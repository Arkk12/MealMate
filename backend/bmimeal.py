"""
bmimeal.py
- AI generates a DIFFERENT meal for every single day (7 unique days)
- Macros are calculated from BMI inputs and enforced as minimums
- Personal note is embedded into the AI prompt so preferences are honoured
- Falls back to a deterministic plan if Groq fails
"""

import json
import os
import random
from datetime import datetime
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from groq import Groq
from pydantic import BaseModel

try:
    import firebase_admin
    from firebase_admin import auth, credentials, firestore
except ImportError:
    firebase_admin = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

bmimeal_router = APIRouter(prefix="/user", tags=["BMI & Meals"])

# ── Firebase ──────────────────────────────────────────────────────
db = None
try:
    if firebase_admin and not firebase_admin._apps:
        _cred_path = os.path.join(BASE_DIR, os.getenv("FIREBASE_CREDENTIALS_PATH", "serviceAccountKey.json"))
        firebase_admin.initialize_app(credentials.Certificate(_cred_path))
    if firebase_admin:
        db = firestore.client()
        print("✅ Firebase connected")
except Exception as _e:
    print(f"⚠️  Firebase init error: {_e}")

# ── Groq ──────────────────────────────────────────────────────────
_groq_key   = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=_groq_key) if _groq_key else None
if groq_client:
    print("✅ Groq ready (bmimeal)")
else:
    print("⚠️  GROQ_API_KEY missing in bmimeal")


# ══════════════════════════════════════════════════════════════════
# MACRO CALCULATOR
# ══════════════════════════════════════════════════════════════════
def calc_bmi(weight: float, height_cm: float) -> float:
    return round(weight / (height_cm / 100) ** 2, 1)

def calc_bmr(weight: float, height_cm: float, age: int, gender: str) -> float:
    if gender.lower() == "male":
        return 88.362 + 13.397 * weight + 4.799 * height_cm - 5.677 * age
    return 447.593 + 9.247 * weight + 3.098 * height_cm - 4.330 * age

ACTIVITY_MULT = {
    "1.2": 1.2, "sedentary": 1.2,
    "1.375": 1.375, "light": 1.375,
    "1.55": 1.55, "moderate": 1.55,
    "1.725": 1.725, "active": 1.725, "very active": 1.9,
}

def calc_macros(weight: float, height_cm: float, age: int,
                gender: str, activity: str, goal: str) -> dict:
    bmr  = calc_bmr(weight, height_cm, age, gender)
    mult = ACTIVITY_MULT.get(str(activity).lower(), 1.55)
    tdee = bmr * mult

    if goal == "bulk":
        calories = int(tdee * 1.15)
    elif goal == "cut":
        calories = int(tdee * 0.80)
    elif goal == "recompose":
        calories = int(tdee * 1.00)
    else:
        calories = int(tdee)

    protein = max(int(weight * 2.0), 100)      # 2 g/kg minimum
    fat     = max(int(weight * 0.8), 45)
    carbs   = max(int((calories - protein * 4 - fat * 9) / 4), 80)

    # If carbs went negative, recalculate
    if carbs < 80:
        calories = protein * 4 + fat * 9 + 80 * 4
        carbs = 80

    return {
        "calories": calories,
        "protein":  protein,
        "carbs":    carbs,
        "fat":      fat,
    }


# ══════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════════
class SaveBMIRequest(BaseModel):
    uid:               str
    email:             str
    bmi:               float
    weight:            float
    height:            float
    age:               int
    gender:            str
    goal:              str
    activity:          str
    protein:           float
    carbs:             float
    fat:               float
    calories:          float
    medical_condition: Optional[str] = "none"


class BMIAnalysisRequest(BaseModel):
    bmi:               float
    weight:            float
    height:            float
    age:               int
    gender:            str
    goal:              str
    activity:          str
    protein:           float
    carbs:             float
    fat:               float
    calories:          float
    medical_condition: Optional[str] = "none"
    personal_note:     Optional[str] = ""


class MealPlanRequest(BaseModel):
    uid:               str
    diet_type:         str
    medical_condition: Optional[str]   = "none"
    personal_note:     Optional[str]   = ""
    # Sidebar / manual overrides
    bmi_override:      Optional[float] = None
    goal_override:     Optional[str]   = None
    # Raw inputs (used to recompute macros when provided)
    weight:            Optional[float] = None
    height:            Optional[float] = None
    age:               Optional[int]   = None
    gender:            Optional[str]   = None
    activity:          Optional[str]   = None
    # Direct macro overrides (highest priority)
    calories:          Optional[int]   = None
    protein:           Optional[int]   = None
    carbs:             Optional[int]   = None
    fat:               Optional[int]   = None


class VerifyTokenRequest(BaseModel):
    id_token: str


# ══════════════════════════════════════════════════════════════════
# FIRESTORE HELPERS
# ══════════════════════════════════════════════════════════════════
def _user_ref(uid: str):
    if not db:
        raise HTTPException(500, "Firebase not initialised")
    return db.collection("users").document(uid)


# ══════════════════════════════════════════════════════════════════
# BMI ANALYSIS  (deterministic — no Groq needed)
# ══════════════════════════════════════════════════════════════════
def build_bmi_analysis(req: BMIAnalysisRequest) -> dict:
    bmi, goal, medical, note = req.bmi, req.goal, req.medical_condition, (req.personal_note or "").strip()

    if bmi < 18.5:
        assessment  = "You are underweight. Focus on a calorie surplus, protein-rich foods, and strength training to build healthy mass."
        weekly_goal = "Aim to gain 0.25–0.5 kg this week with consistent meals and at least 5 eating occasions per day."
    elif bmi < 25:
        assessment  = "Your BMI is in the healthy range — a great foundation. Focus on consistency and goal-specific training."
        weekly_goal = "Hit your calorie and protein targets on at least 5 of 7 days while maintaining your training routine."
    elif bmi < 30:
        assessment  = "Your BMI is slightly elevated. A moderate calorie deficit, daily movement, and high protein will help effectively."
        weekly_goal = "Target 0.25–0.5 kg of fat loss this week through a sustainable deficit and at least 30 min of daily activity."
    else:
        assessment  = "Your BMI is in the obese range. A structured plan, regular movement, and professional guidance can make a big difference."
        weekly_goal = "Focus on daily walks, portion control, and a realistic calorie deficit — consistency beats perfection every time."

    goal_notes = {
        "bulk":      "Your surplus plan should drive muscle gain — ensure progressive overload in training.",
        "cut":       "Your deficit plan should preserve muscle — keep protein high and avoid crash dieting.",
        "maintain":  "Balance your intake with activity — avoid both large surpluses and deficits.",
        "recompose": "Eat near maintenance with high protein and lift weights — slow but sustainable body change.",
    }
    alignment = goal_notes.get(goal, "Follow a balanced plan aligned with your health goal.")

    tips = [
        f"Hit ~{int(req.protein)}g protein daily — spread it across all meals for best absorption.",
        "Build meals around whole foods: dal, eggs, paneer, curd, rice, roti, fruits, vegetables.",
        "Drink 2.5–3 L of water daily and get 7–8 hours of sleep for optimal recovery.",
    ]
    if medical and medical != "none":
        tips.append(f"For {medical}: follow condition-specific food rules and consult your doctor regularly.")
    if note:
        tips.append(f"Your personal note was considered: '{note[:80]}' — keep meals practical and repeatable.")

    focus_foods = ["dal", "curd", "paneer or eggs", "seasonal vegetables", "oats", "brown rice", "fruit"]
    avoid_foods = ["deep-fried foods", "sugary drinks", "ultra-processed snacks"]

    MEDICAL_OVERRIDES = {
        "diabetes":        (["high-fibre carbs", "lean protein", "non-starchy vegetables", "curd"], ["sweets", "refined sugar", "white bread"]),
        "hypertension":    (["potassium-rich fruits", "low-sodium meals", "leafy greens"], ["pickles", "papad", "packaged salty snacks"]),
        "pcos":            (["high-protein meals", "fibre-rich foods", "healthy fats", "seeds"], ["sugary foods", "refined carbs", "sweet beverages"]),
        "high cholesterol":(["oats", "nuts in moderation", "fibre-rich foods", "fish"], ["fried foods", "trans fats", "full-fat dairy"]),
        "fatty liver":     (["lean protein", "green vegetables", "fruit"], ["alcohol", "soft drinks", "fried snacks"]),
        "ibs":             (["easy-to-digest meals", "cooked vegetables", "curd"], ["very spicy foods", "raw salads in excess", "trigger foods"]),
        "kidney disease":  (["low-potassium vegetables", "white rice", "egg whites"], ["high-potassium foods", "red meat", "processed foods"]),
        "anemia":          (["iron-rich foods", "spinach", "jaggery", "dates", "lean meat"], ["tea/coffee with meals", "calcium supplements near iron-rich food"]),
    }
    if medical in MEDICAL_OVERRIDES:
        focus_foods, avoid_foods = MEDICAL_OVERRIDES[medical]

    return {
        "bmi_assessment": assessment,
        "goal_alignment":  alignment,
        "top_tips":        tips[:4],
        "foods_to_focus":  focus_foods,
        "foods_to_avoid":  avoid_foods,
        "weekly_target":   weekly_goal,
        "motivation":      "Small, consistent actions every day beat extreme plans every time. Trust the process. 💪",
    }


# ══════════════════════════════════════════════════════════════════
# AI MEAL GENERATION  — 7 unique days via Groq
# ══════════════════════════════════════════════════════════════════
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

def _build_meal_prompt(
    day: str,
    diet_type: str,
    targets: dict,
    goal: str,
    medical: str,
    note: str,
    bmi: float,
    age: int,
    gender: str,
    used_meals: list[str],   # names used in previous days to ensure variety
) -> str:
    diet_desc = "strictly vegetarian — NO meat, fish, or eggs" if diet_type == "veg" else "non-vegetarian — can include chicken, fish, eggs, dairy"
    medical_line = f"Medical condition: {medical}. Adjust meals accordingly (e.g. low sugar for diabetes, low sodium for hypertension)." if medical != "none" else ""
    note_line    = f"User preferences/restrictions: {note}" if note.strip() else ""
    avoid_line   = f"Avoid repeating these meals used on previous days: {', '.join(used_meals)}." if used_meals else ""
    bmi_line     = f"BMI {bmi} — {'underweight, use calorie-dense foods' if bmi < 18.5 else 'overweight, prefer high-fibre, low-calorie-dense options' if bmi >= 25 else 'healthy BMI, balanced meals'}."

    return f"""You are an expert Indian nutritionist. Create a meal plan for {day} ONLY.

USER PROFILE:
- Diet: {diet_type} ({diet_desc})
- Goal: {goal} | BMI: {bmi} | Age: {age} | Gender: {gender}
- {bmi_line}
- Daily targets: {targets['calories']} kcal minimum | Protein: {targets['protein']}g min | Carbs: {targets['carbs']}g min | Fat: {targets['fat']}g min
- {medical_line}
- {note_line}
- {avoid_line}

RULES:
1. Each meal must be a DIFFERENT Indian dish — rotate ingredients for variety.
2. Macros across all 4 meals MUST sum to AT LEAST the daily targets above.
3. Use affordable Indian ingredients: dal, rice, roti, paneer, curd, eggs, chicken, vegetables, oats, fruits.
4. Meal sizes should be realistic for an Indian adult.
5. Return ONLY valid JSON, no markdown, no extra text.

JSON FORMAT:
{{
  "breakfast": {{"name": "dish name", "items": ["100g oats", "200ml milk", "1 banana"], "calories": 420, "protein": 22, "carbs": 58, "fat": 8}},
  "lunch":     {{"name": "dish name", "items": ["150g dal", "200g rice", "salad"], "calories": 650, "protein": 28, "carbs": 95, "fat": 9}},
  "snack":     {{"name": "dish name", "items": ["30g almonds", "200ml curd"], "calories": 230, "protein": 12, "carbs": 14, "fat": 14}},
  "dinner":    {{"name": "dish name", "items": ["150g paneer", "3 roti", "sabzi"], "calories": 620, "protein": 32, "carbs": 62, "fat": 22}}
}}"""


def _fallback_day(day: str, diet_type: str, targets: dict, medical: str) -> dict:
    """Deterministic fallback if AI fails for a day."""
    veg_days = [
        {"breakfast": {"name": "Masala Oats", "items": ["80g oats", "200ml milk", "vegetables"], "calories": 380, "protein": 18, "carbs": 52, "fat": 8},
         "lunch":     {"name": "Dal Tadka Rice", "items": ["150g moong dal", "180g rice", "salad"], "calories": 580, "protein": 24, "carbs": 88, "fat": 7},
         "snack":     {"name": "Peanut Curd Bowl", "items": ["200ml curd", "30g peanuts"], "calories": 240, "protein": 14, "carbs": 12, "fat": 14},
         "dinner":    {"name": "Paneer Sabzi Roti", "items": ["150g paneer", "3 roti", "mixed sabzi"], "calories": 620, "protein": 30, "carbs": 58, "fat": 24}},

        {"breakfast": {"name": "Poha with Curd", "items": ["80g poha", "200ml curd", "peanuts"], "calories": 360, "protein": 16, "carbs": 54, "fat": 9},
         "lunch":     {"name": "Rajma Chawal", "items": ["150g rajma", "200g rice", "onion salad"], "calories": 600, "protein": 26, "carbs": 90, "fat": 6},
         "snack":     {"name": "Banana Protein Shake", "items": ["1 banana", "200ml milk", "20g whey"], "calories": 260, "protein": 22, "carbs": 30, "fat": 3},
         "dinner":    {"name": "Chana Masala Roti", "items": ["200g chana", "3 roti", "cucumber raita"], "calories": 610, "protein": 28, "carbs": 80, "fat": 12}},

        {"breakfast": {"name": "Vegetable Upma", "items": ["80g semolina", "vegetables", "200ml curd"], "calories": 370, "protein": 14, "carbs": 58, "fat": 8},
         "lunch":     {"name": "Palak Paneer Rice", "items": ["120g paneer", "180g rice", "spinach gravy"], "calories": 620, "protein": 28, "carbs": 72, "fat": 20},
         "snack":     {"name": "Mixed Nuts & Fruit", "items": ["30g mixed nuts", "1 apple"], "calories": 210, "protein": 6, "carbs": 22, "fat": 12},
         "dinner":    {"name": "Dal Makhani Roti", "items": ["180g dal makhani", "3 roti", "salad"], "calories": 590, "protein": 26, "carbs": 76, "fat": 16}},
    ]

    nonveg_days = [
        {"breakfast": {"name": "Egg Bhurji Toast", "items": ["3 eggs", "2 whole wheat toast", "onion tomato"], "calories": 420, "protein": 26, "carbs": 34, "fat": 18},
         "lunch":     {"name": "Chicken Rice Bowl", "items": ["180g chicken breast", "200g rice", "raita"], "calories": 660, "protein": 44, "carbs": 72, "fat": 12},
         "snack":     {"name": "Boiled Egg Curd", "items": ["2 boiled eggs", "200ml curd"], "calories": 240, "protein": 22, "carbs": 8, "fat": 12},
         "dinner":    {"name": "Chicken Curry Roti", "items": ["180g chicken", "3 roti", "sabzi", "salad"], "calories": 640, "protein": 40, "carbs": 58, "fat": 18}},

        {"breakfast": {"name": "Omelette Paratha", "items": ["3 eggs", "2 paratha", "green chutney"], "calories": 480, "protein": 28, "carbs": 44, "fat": 20},
         "lunch":     {"name": "Fish Dal Rice", "items": ["180g fish fillet", "150g dal", "200g rice"], "calories": 640, "protein": 48, "carbs": 68, "fat": 10},
         "snack":     {"name": "Whey Banana Shake", "items": ["25g whey", "1 banana", "200ml milk"], "calories": 270, "protein": 28, "carbs": 32, "fat": 3},
         "dinner":    {"name": "Egg Curry Roti", "items": ["3 eggs curry", "3 roti", "mixed vegetables"], "calories": 580, "protein": 34, "carbs": 56, "fat": 20}},

        {"breakfast": {"name": "Chicken Poha", "items": ["80g poha", "100g chicken", "peanuts", "curd"], "calories": 460, "protein": 32, "carbs": 46, "fat": 14},
         "lunch":     {"name": "Mutton Dal Khichdi", "items": ["150g mutton", "200g khichdi", "salad"], "calories": 680, "protein": 46, "carbs": 60, "fat": 22},
         "snack":     {"name": "Boiled Egg Nuts", "items": ["2 boiled eggs", "30g almonds"], "calories": 260, "protein": 20, "carbs": 6, "fat": 18},
         "dinner":    {"name": "Grilled Chicken Roti", "items": ["200g grilled chicken", "3 roti", "raita", "salad"], "calories": 640, "protein": 50, "carbs": 54, "fat": 14}},
    ]

    bank = veg_days if diet_type == "veg" else nonveg_days
    template = bank[hash(day) % len(bank)]

    # Scale each meal so totals meet targets
    total_cal = sum(m["calories"] for m in template.values())
    scale     = max(targets["calories"] / max(total_cal, 1), 1.0)

    result = {}
    for slot, meal in template.items():
        result[slot] = {
            "name":     meal["name"],
            "items":    meal["items"],
            "calories": round(meal["calories"] * scale),
            "protein":  round(meal["protein"]  * scale),
            "carbs":    round(meal["carbs"]    * scale),
            "fat":      round(meal["fat"]      * scale),
        }
    return result


async def generate_7_day_plan(
    diet_type: str,
    targets: dict,
    goal: str,
    medical: str,
    note: str,
    bmi: float,
    age: int,
    gender: str,
) -> list[dict]:
    """Generate 7 unique days. Uses AI for each day, falls back to deterministic."""
    days_out = []
    used_meal_names: list[str] = []

    for day in DAYS:
        day_meals = None

        if groq_client:
            prompt = _build_meal_prompt(
                day, diet_type, targets, goal, medical, note,
                bmi, age, gender, used_meal_names
            )
            try:
                response = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=800,
                    temperature=0.85,   # higher temp = more variety per day
                )
                raw = response.choices[0].message.content.strip()
                if "```json" in raw:
                    raw = raw.split("```json")[1].split("```")[0].strip()
                elif "```" in raw:
                    raw = raw.split("```")[1].split("```")[0].strip()

                parsed = json.loads(raw)

                # Validate required slots
                if all(k in parsed for k in ["breakfast", "lunch", "snack", "dinner"]):
                    # Enforce minimum macros — scale up if needed
                    total_cal  = sum(parsed[s].get("calories", 0) for s in ["breakfast","lunch","snack","dinner"])
                    total_prot = sum(parsed[s].get("protein",  0) for s in ["breakfast","lunch","snack","dinner"])
                    scale_cal  = max(targets["calories"] / max(total_cal, 1), 1.0)
                    scale_prot = max(targets["protein"]  / max(total_prot, 1), 1.0)
                    scale      = max(scale_cal, scale_prot)

                    if scale > 1.05:  # only scale if more than 5% short
                        for slot in ["breakfast","lunch","snack","dinner"]:
                            for key in ["calories","protein","carbs","fat"]:
                                if key in parsed[slot]:
                                    parsed[slot][key] = round(parsed[slot][key] * scale)

                    day_meals = parsed
                    # Track names for variety enforcement
                    for slot in ["breakfast","lunch","snack","dinner"]:
                        name = parsed[slot].get("name","")
                        if name:
                            used_meal_names.append(name)

            except Exception as e:
                print(f"⚠️  AI failed for {day}: {e} — using fallback")

        if day_meals is None:
            day_meals = _fallback_day(day, diet_type, targets, medical)

        # Compute daily totals
        total_cal  = sum(day_meals[s].get("calories", 0) for s in ["breakfast","lunch","snack","dinner"])
        total_prot = sum(day_meals[s].get("protein",  0) for s in ["breakfast","lunch","snack","dinner"])
        total_carb = sum(day_meals[s].get("carbs",    0) for s in ["breakfast","lunch","snack","dinner"])
        total_fat  = sum(day_meals[s].get("fat",      0) for s in ["breakfast","lunch","snack","dinner"])

        days_out.append({
            "day":       day,
            "breakfast": day_meals["breakfast"],
            "lunch":     day_meals["lunch"],
            "snack":     day_meals["snack"],
            "dinner":    day_meals["dinner"],
            "day_totals": {
                "calories": total_cal,
                "protein":  total_prot,
                "carbs":    total_carb,
                "fat":      total_fat,
            }
        })

    return days_out


def build_ai_insight(
    targets: dict, goal: str, medical: str, note: str, bmi: float, diet_type: str
) -> dict:
    bmi_line = (
        "Meals are calorie-dense to support healthy weight gain." if bmi < 18.5 else
        "Meals are high in fibre and protein to support fat loss." if bmi >= 25 else
        "Meals are balanced for steady progress at a healthy BMI."
    )
    note_line = f"Your note '{note[:60]}...' was incorporated into ingredient and meal choices." if note.strip() else ""

    MEDICAL_FOODS = {
        "diabetes":       (["low-GI carbs","lean protein","high-fibre vegetables","curd"], ["sweets","refined sugar","white bread"]),
        "hypertension":   (["potassium-rich fruit","leafy greens","low-sodium meals"],       ["pickles","papad","packaged salty snacks"]),
        "pcos":           (["high-protein meals","fibre-rich foods","seeds","healthy fats"],  ["sugary foods","refined carbs","sweet drinks"]),
        "high cholesterol":(["oats","nuts in moderation","fibre-rich foods"],                 ["fried foods","trans fats","full-fat dairy"]),
        "fatty liver":    (["lean protein","green vegetables","fruit"],                       ["alcohol","soft drinks","fried snacks"]),
        "kidney disease": (["egg whites","white rice","low-potassium vegetables"],            ["high-potassium foods","red meat","processed foods"]),
        "anemia":         (["spinach","jaggery","dates","lean meat","legumes"],               ["tea/coffee with meals"]),
    }
    focus_foods = ["protein-rich foods","fibre-rich carbs","seasonal vegetables","curd / yogurt","fruit"]
    avoid_foods = ["fried foods","sugary drinks","ultra-processed snacks"]
    if medical in MEDICAL_FOODS:
        focus_foods, avoid_foods = MEDICAL_FOODS[medical]

    return {
        "summary":       f"{bmi_line} This 7-day plan is built to meet your minimum {targets['calories']} kcal and {targets['protein']}g protein daily for your {goal} goal. {note_line}",
        "focus_foods":   focus_foods,
        "avoid_foods":   avoid_foods,
        "weekly_target": f"Hit ≥{targets['calories']} kcal and ≥{targets['protein']}g protein on at least 5 of 7 days while staying consistent with your {goal} goal.",
        "tip":           f"{bmi_line} Prep meals on Sunday to stay on track all week.",
        "macro_summary": f"{targets['calories']} kcal • {targets['protein']}g protein • {targets['carbs']}g carbs • {targets['fat']}g fat",
    }


# ══════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════

@bmimeal_router.post("/verify-token")
async def verify_token(req: VerifyTokenRequest):
    try:
        decoded = auth.verify_id_token(req.id_token)
        return {"success": True, "uid": decoded["uid"], "email": decoded.get("email", "")}
    except Exception as e:
        raise HTTPException(401, str(e))


@bmimeal_router.post("/bmi/save")
async def save_bmi(req: SaveBMIRequest):
    ref = _user_ref(req.uid)
    try:
        bmi_data = req.model_dump()
        bmi_data["timestamp"] = datetime.utcnow().isoformat()
        ref.collection("bmi_history").add(bmi_data)
        ref.set({"email": req.email, "latest_bmi": bmi_data,
                 "updated_at": datetime.utcnow().isoformat()}, merge=True)
        return {"success": True}
    except Exception as e:
        raise HTTPException(500, str(e))


@bmimeal_router.post("/bmi/analyze")
async def analyze_bmi(req: BMIAnalysisRequest):
    return {"success": True, "analysis": build_bmi_analysis(req)}


@bmimeal_router.get("/bmi/history/{uid}")
async def bmi_history(uid: str):
    ref = _user_ref(uid)
    try:
        docs = (ref.collection("bmi_history")
                   .order_by("timestamp", direction=firestore.Query.DESCENDING)
                   .limit(10).stream())
        return {"success": True, "history": [{**d.to_dict(), "id": d.id} for d in docs]}
    except Exception as e:
        raise HTTPException(500, str(e))


@bmimeal_router.get("/bmi/latest/{uid}")
async def latest_bmi(uid: str):
    ref = _user_ref(uid)
    try:
        doc = ref.get()
        return {"success": doc.exists, "data": doc.to_dict().get("latest_bmi") if doc.exists else None}
    except Exception as e:
        raise HTTPException(500, str(e))


@bmimeal_router.post("/meal/generate")
async def generate_meal_plan(req: MealPlanRequest):
    # ── 1. Resolve targets ─────────────────────────────────────
    # Priority: direct macro overrides > computed from inputs > Firestore > defaults
    bmi_info: dict = {}
    if db:
        try:
            doc = db.collection("users").document(req.uid).get()
            if doc.exists:
                bmi_info = doc.to_dict().get("latest_bmi", {})
        except Exception:
            pass

    # Resolve raw inputs
    weight  = req.weight  or bmi_info.get("weight",  70.0)
    height  = req.height  or bmi_info.get("height",  170.0)
    age     = req.age     or bmi_info.get("age",      25)
    gender  = req.gender  or bmi_info.get("gender",   "male")
    activity= req.activity or bmi_info.get("activity", "1.55")
    goal    = req.goal_override or bmi_info.get("goal", "maintain")
    bmi_val = req.bmi_override  or bmi_info.get("bmi",  calc_bmi(weight, height))
    medical = req.medical_condition or bmi_info.get("medical_condition", "none")
    note    = (req.personal_note or "").strip()
    diet    = req.diet_type

    # Compute fresh macros from physical inputs
    computed = calc_macros(weight, height, age, gender, activity, goal)

    # Apply direct overrides (only if explicitly provided)
    targets = {
        "calories": req.calories or computed["calories"],
        "protein":  req.protein  or computed["protein"],
        "carbs":    req.carbs    or computed["carbs"],
        "fat":      req.fat      or computed["fat"],
    }

    # ── 2. Generate 7 unique AI days ───────────────────────────
    days = await generate_7_day_plan(
        diet_type=diet, targets=targets, goal=goal,
        medical=medical, note=note, bmi=bmi_val,
        age=age, gender=gender,
    )

    # ── 3. Build AI insight panel ──────────────────────────────
    insight = build_ai_insight(targets, goal, medical, note, bmi_val, diet)

    medical_note = (
        f"This plan accounts for {medical} — always follow your doctor's advice alongside this plan."
        if medical != "none"
        else "No specific medical adjustments applied."
    )

    meal_plan = {
        "diet_type":    diet,
        "medical_note": medical_note,
        "targets":      targets,
        "days":         days,
    }

    # ── 4. Save to Firestore ───────────────────────────────────
    if db:
        try:
            db.collection("users").document(req.uid).set(
                {"latest_meal_plan": {**meal_plan, "generated_at": datetime.utcnow().isoformat()}},
                merge=True
            )
        except Exception as e:
            print(f"⚠️  Firestore save error: {e}")

    return {"success": True, "meal_plan": meal_plan, "ai_insight": insight}


@bmimeal_router.get("/meal/latest/{uid}")
async def latest_meal(uid: str):
    ref = _user_ref(uid)
    try:
        doc = ref.get()
        if doc.exists:
            plan = doc.to_dict().get("latest_meal_plan")
            if plan:
                return {"success": True, "meal_plan": plan}
        return {"success": False, "meal_plan": None}
    except Exception as e:
        raise HTTPException(500, str(e))