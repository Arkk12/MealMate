"""
MealMate API — main.py
Features:
  • AI generates UNIQUE meals for each of the 7 days (no copy-paste days)
  • Diet types: veg | mixed | nonveg
  • Every meal includes a short Indian/American recipe
  • Macros are calculated from BMI inputs and respected in the plan
  • Personal note from user is embedded into the AI prompt
  • Groq llama-3.1-8b-instant with deterministic fallback
"""

import json
import os
import re
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from groq import Groq
from pydantic import BaseModel

# ── Env ───────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# ── Routers ───────────────────────────────────────────────────
try:
    from .fitness  import fitness_router
    from .bmimeal  import bmimeal_router
except ImportError:
    try:
        import backend.fitness as _f
        import backend.bmimeal as _b
        fitness_router = _f.fitness_router
        bmimeal_router = _b.bmimeal_router
    except ImportError:
        from fitness  import fitness_router
        from bmimeal  import bmimeal_router

# ── App ───────────────────────────────────────────────────────
app = FastAPI(title="MealMate API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(fitness_router)
app.include_router(bmimeal_router)

# ── Groq ──────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client  = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
print("✅ Groq connected" if groq_client else "⚠️  GROQ_API_KEY missing — AI features disabled")

# ── Firestore helper ──────────────────────────────────────────
def get_db():
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
        if not firebase_admin._apps:
            cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "serviceAccountKey.json")
            if not os.path.isabs(cred_path):
                cred_path = os.path.join(BASE_DIR, cred_path)
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        print(f"Firestore error: {e}")
        return None

# ═══════════════════════════════════════════════════════════════
# NUTRITION MATH
# ═══════════════════════════════════════════════════════════════

def calc_bmi(weight: float, height_cm: float) -> float:
    return round(weight / (height_cm / 100) ** 2, 1)

def calc_bmr(weight: float, height_cm: float, age: int, gender: str) -> float:
    if gender.lower() == "male":
        return 88.362 + 13.397 * weight + 4.799 * height_cm - 5.677 * age
    return 447.593 + 9.247 * weight + 3.098 * height_cm - 4.330 * age

ACTIVITY_MULT = {
    "sedentary": 1.2, "1.2": 1.2,
    "light": 1.375,   "1.375": 1.375,
    "moderate": 1.55, "1.55": 1.55,
    "active": 1.725,  "1.725": 1.725,
    "very active": 1.9, "1.725": 1.725,
}

def calc_tdee(bmr: float, activity: str) -> float:
    return bmr * ACTIVITY_MULT.get(str(activity).lower(), 1.55)

def calc_macros(tdee: float, goal: str, weight: float) -> dict:
    if goal == "bulk":       cal = int(tdee * 1.15)
    elif goal == "cut":      cal = int(tdee * 0.82)
    elif goal == "recompose":cal = int(tdee)
    else:                    cal = int(tdee)

    protein = max(int(weight * 2.0), 100)   # 2 g/kg minimum
    fat     = max(int(weight * 0.8), 50)
    carbs   = max(int((cal - protein * 4 - fat * 9) / 4), 80)
    # Recalc calories from macros (source of truth)
    cal = protein * 4 + carbs * 4 + fat * 9
    return {"calories": cal, "protein": protein, "carbs": carbs, "fat": fat}

# ═══════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    message: str

class BmiAnalyzeRequest(BaseModel):
    bmi: float
    weight: float
    height: float
    age: int
    gender: str
    goal: str
    activity: str
    protein: int
    carbs: int
    fat: int
    calories: int
    medical_condition: str = "none"
    personal_note: str = ""

class MealGenerateRequest(BaseModel):
    uid: str
    diet_type: str                        # veg | mixed | nonveg
    medical_condition: str = "none"
    personal_note: str = ""
    bmi_override: Optional[float] = None
    goal_override: Optional[str] = None
    # BMI page inputs passed through
    weight: Optional[float] = None
    height: Optional[float] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    activity: Optional[str] = None
    calories: Optional[int] = None
    protein: Optional[int] = None
    carbs: Optional[int] = None
    fat: Optional[int] = None

class BmiSaveRequest(BaseModel):
    uid: str
    email: str
    bmi: float
    weight: float
    height: float
    age: int
    gender: str
    goal: str
    activity: str
    protein: int
    carbs: int
    fat: int
    calories: int
    medical_condition: str = "none"

# ═══════════════════════════════════════════════════════════════
# CHAT
# ═══════════════════════════════════════════════════════════════

DIET_KW = [
    "diet","meal","nutrition","protein","calorie","calories","carb","carbs",
    "fat","fiber","vitamin","supplement","weight","bmi","fitness","exercise",
    "workout","gym","health","lose","gain","muscle","pcos","diabetes",
    "hypertension","cholesterol","thyroid","food","eating","breakfast","lunch",
    "dinner","snack","recipe","cook","indian","american","keto","vegan",
    "vegetarian","intermittent","fasting","bulk","cut","recompose","maintain",
]

SYSTEM_PROMPT = (
    "You are NutriCore AI, a world-class diet, fitness and medical nutrition assistant. "
    "Give practical, actionable advice. Prefer Indian food options but also mention global options. "
    "Use emojis. Keep responses under 220 words. "
    "For serious medical conditions always suggest consulting a doctor."
)

def is_diet_related(msg: str) -> bool:
    return any(kw in msg.lower() for kw in DIET_KW)

# ═══════════════════════════════════════════════════════════════
# BMI ANALYSIS (rule-based, no AI needed for reliability)
# ═══════════════════════════════════════════════════════════════

def build_bmi_analysis(p: BmiAnalyzeRequest) -> dict:
    bmi, goal, medical, note = p.bmi, p.goal, p.medical_condition, p.personal_note.strip()

    if bmi < 18.5:
        assessment = "You are underweight. Prioritise a calorie surplus, high-protein meals and strength training to build lean mass."
        target = "Gain 0.25–0.5 kg this week through consistent meals and hitting your protein target every day."
    elif bmi < 25:
        assessment = "Your BMI is in the healthy range — great foundation. Focus on your specific goal with consistent nutrition and training."
        target = "Hit your calorie and protein targets on at least 5 of 7 days this week."
    elif bmi < 30:
        assessment = "Your BMI is above the healthy range. A moderate calorie deficit, higher protein intake and daily movement will help."
        target = "Target 0.25–0.5 kg of fat loss this week with a sustainable deficit and regular walking."
    else:
        assessment = "Your BMI is in the obese range. A structured plan, regular activity and medical guidance can make a big difference."
        target = "Focus on daily walks, portion control and a realistic calorie deficit this week."

    goal_map = {
        "bulk":      "Your surplus plan should emphasise progressive strength training and enough recovery between sessions.",
        "cut":       "Your deficit plan should emphasise high protein to preserve muscle, and foods that keep you full longer.",
        "maintain":  "Balance your calorie intake with regular activity and a consistent daily routine.",
        "recompose": "Keep protein high while staying near maintenance calories; resistance training is key.",
    }
    alignment = goal_map.get(goal, "Follow a balanced nutrition plan aligned with your health goal.")

    tips = [
        f"Aim for ~{p.protein}g protein daily — spread across all meals — to support your {goal} goal.",
        "Build meals around whole foods: dal, eggs/chicken, paneer, curd, rice, roti, oats, fruits, vegetables.",
        "Drink 2.5–3 L of water daily and aim for 7–8 hours of sleep for recovery.",
    ]
    med_tips = {
        "diabetes":         "Keep carbs low-GI (oats, brown rice, dal) and pair every meal with protein.",
        "hypertension":     "Reduce sodium — avoid pickles, papad and packaged snacks. Eat more potassium-rich foods.",
        "pcos":             "High protein + high fibre meals help regulate insulin. Limit refined sugar completely.",
        "high cholesterol": "Focus on soluble fibre (oats, dal) and limit saturated fat (ghee, fried foods).",
        "fatty liver":      "Eliminate alcohol and sugary drinks. Eat lean protein and plenty of vegetables.",
        "ibs":              "Eat simple, easy-to-digest meals. Track trigger foods and avoid very spicy items.",
    }
    if medical in med_tips:
        tips.append(med_tips[medical])
    elif note:
        tips.append("Your personal preferences were noted and incorporated into your plan.")

    focus_foods = ["dal / lentils", "curd / Greek yogurt", "paneer / tofu", "oats", "seasonal vegetables", "fruits"]
    avoid_foods = ["deep-fried foods", "sugary drinks", "ultra-processed snacks"]

    med_foods = {
        "diabetes":         (["low-GI carbs","fibre-rich veg","protein with every meal"], ["sweets","refined sugar","white bread"]),
        "hypertension":     (["potassium-rich fruit","low-sodium meals","oats"], ["pickles","packaged salty snacks","papad"]),
        "pcos":             (["protein-rich meals","high-fibre foods","healthy fats"], ["sugary foods","refined snacks","sweet beverages"]),
        "high cholesterol": (["oats","nuts (moderate)","fibre-rich foods"], ["fried foods","trans fats","full-fat dairy"]),
        "fatty liver":      (["lean protein","vegetables","fruit"], ["soft drinks","alcohol","fried foods"]),
        "ibs":              (["simple easy-digest meals","cooked veg"], ["very spicy foods","raw onion","carbonated drinks"]),
    }
    if medical in med_foods:
        focus_foods, avoid_foods = med_foods[medical]

    return {
        "bmi_assessment": assessment,
        "goal_alignment":  alignment,
        "top_tips":        tips[:4],
        "foods_to_focus":  focus_foods,
        "foods_to_avoid":  avoid_foods,
        "weekly_target":   target,
        "motivation":      "Consistency beats perfection. Show up every day and trust the process. 💪",
    }

# ═══════════════════════════════════════════════════════════════
# MEAL PLAN — AI GENERATION
# ═══════════════════════════════════════════════════════════════

DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

def resolve_macros(req: MealGenerateRequest) -> dict:
    """Return final macro targets, calculating from BMI inputs when available."""
    if req.weight and req.height and req.age and req.gender:
        bmr  = calc_bmr(req.weight, req.height, req.age, req.gender)
        tdee = calc_tdee(bmr, req.activity or "moderate")
        goal = req.goal_override or "maintain"
        auto = calc_macros(tdee, goal, req.weight)
        return {
            "calories": req.calories or auto["calories"],
            "protein":  req.protein  or auto["protein"],
            "carbs":    req.carbs    or auto["carbs"],
            "fat":      req.fat      or auto["fat"],
        }
    return {
        "calories": req.calories or 2000,
        "protein":  req.protein  or 120,
        "carbs":    req.carbs    or 220,
        "fat":      req.fat      or 65,
    }

def diet_description(diet: str) -> str:
    if diet == "veg":
        return "strictly vegetarian — NO meat, fish, or eggs. Use paneer, tofu, dal, curd, legumes as protein."
    if diet == "mixed":
        return "mixed (flexitarian) — can use eggs, occasional chicken or fish (2–3 days/week), plus vegetarian options."
    return "non-vegetarian — freely use chicken, fish, eggs, mutton alongside vegetarian staples."

def medical_instruction(medical: str) -> str:
    instructions = {
        "diabetes":         "Keep carbs low-GI (oats, brown rice, dal). Pair every meal with protein. Avoid sweets.",
        "hypertension":     "Use minimal salt. No pickles or papad. Add potassium-rich foods (banana, spinach).",
        "pcos":             "High protein + high fibre. Limit refined carbs. Add seeds (flax, chia) to meals.",
        "high cholesterol": "Use olive oil or minimal ghee. Include oats and fibre. Avoid fried or fatty items.",
        "fatty liver":      "Lean protein only. No fried food, no alcohol, minimal sugar. Lots of green vegetables.",
        "ibs":              "Simple easy-to-digest meals. No very spicy food. Cook all vegetables.",
        "kidney disease":   "Low potassium, low phosphorus. Limit dal and nuts. Consult doctor for protein amount.",
        "anemia":           "Iron-rich foods: spinach, rajma, jaggery, eggs. Pair with vitamin C (lemon, amla).",
        "heart disease":    "Heart-healthy fats (olive oil, nuts). No trans fats or fried food. Low sodium.",
        "pregnancy":        "Include folate (leafy greens), calcium (milk, curd), iron. No raw/undercooked meat.",
    }
    return instructions.get(medical, "")

def build_ai_prompt(req: MealGenerateRequest, targets: dict) -> str:
    goal     = req.goal_override or "maintain"
    bmi_val  = req.bmi_override or (calc_bmi(req.weight, req.height) if req.weight and req.height else "unknown")
    medical  = req.medical_condition
    note     = req.personal_note.strip()
    diet     = req.diet_type

    med_str  = f"\nMEDICAL CONDITION: {medical}\nInstruction: {medical_instruction(medical)}" if medical != "none" else ""
    note_str = f"\nUSER PERSONAL PREFERENCES/NOTE: {note}\nIncorporate these preferences into meal choices, names and variety." if note else ""

    return f"""You are a professional nutritionist. Generate a personalised 7-day Indian + American meal plan.

USER PROFILE:
- BMI: {bmi_val} | Goal: {goal}
- Age: {req.age or 'N/A'} | Gender: {req.gender or 'N/A'} | Weight: {req.weight or 'N/A'} kg
- Diet type: {diet} — {diet_description(diet)}
- Daily targets: {targets['calories']} kcal | Protein: {targets['protein']}g | Carbs: {targets['carbs']}g | Fat: {targets['fat']}g
{med_str}{note_str}

STRICT RULES:
1. Each of the 7 days MUST have COMPLETELY DIFFERENT meals — no repeated dish names across the week.
2. Every meal object must include a "recipe" field: 3-5 numbered steps on how to cook it (concise, practical).
3. Meals should be a MIX of Indian and American/Western dishes spread across the week (e.g. oatmeal bowl, grilled chicken salad, pasta, dal rice, paneer curry etc.)
4. Each meal's calories and protein MUST be realistic and match the portion sizes listed.
5. The SUM of breakfast + lunch + snack + dinner calories must be within 10% of the {targets['calories']} kcal daily target.
6. Protein across meals must total at least {targets['protein']}g per day.
7. Items list should name specific foods with quantities (e.g. "150g paneer", "2 rotis", "1 cup dal").
8. Diet type "{diet}" must be respected strictly throughout all 7 days.

Return ONLY a valid JSON object — no markdown, no explanation, no extra text:
{{
  "ai_insight": {{
    "summary": "2 sentences on why this plan suits the user's BMI and {goal} goal",
    "focus_foods": ["food1","food2","food3","food4"],
    "avoid_foods": ["food1","food2","food3"],
    "weekly_target": "one specific measurable weekly target",
    "tip": "one practical personalised tip based on their profile and note"
  }},
  "days": [
    {{
      "day": "Monday",
      "breakfast": {{
        "name": "Masala Omelette with Toast",
        "items": ["2 eggs","1 slice whole wheat toast","1 tsp olive oil","veggies"],
        "calories": 380,
        "protein": 24,
        "carbs": 28,
        "fat": 14,
        "recipe": "1. Beat 2 eggs with salt, pepper, chopped onion and tomato.\\n2. Heat 1 tsp oil in pan on medium heat.\\n3. Pour egg mixture and cook 2 min each side.\\n4. Toast bread and serve alongside."
      }},
      "lunch": {{
        "name": "Dal Tadka with Brown Rice",
        "items": ["1 cup masoor dal","1 cup brown rice","1 tsp ghee","spices"],
        "calories": 560,
        "protein": 28,
        "carbs": 82,
        "fat": 8,
        "recipe": "1. Pressure cook dal with turmeric for 3 whistles.\\n2. Prepare tadka: heat ghee, add cumin, garlic, dried chilli.\\n3. Pour tadka over dal, add salt and cook 5 min.\\n4. Serve with steamed brown rice."
      }},
      "snack": {{
        "name": "Greek Yogurt Fruit Bowl",
        "items": ["150g Greek yogurt","1 banana","10 almonds"],
        "calories": 210,
        "protein": 14,
        "carbs": 22,
        "fat": 7,
        "recipe": "1. Add Greek yogurt to bowl.\\n2. Slice banana on top.\\n3. Crush almonds and sprinkle.\\n4. Optionally drizzle 1 tsp honey."
      }},
      "dinner": {{
        "name": "Paneer Bhurji with Roti",
        "items": ["150g paneer","2 rotis","capsicum","onion","spices"],
        "calories": 520,
        "protein": 28,
        "carbs": 42,
        "fat": 18,
        "recipe": "1. Crumble paneer. Sauté onion and capsicum in 1 tsp oil.\\n2. Add crumbled paneer, salt, turmeric, garam masala.\\n3. Cook on medium 5 min stirring often.\\n4. Serve with fresh rotis."
      }}
    }}
  ]
}}
Generate all 7 days: Monday through Sunday. Each day must have different, creative meal names."""

def parse_ai_json(raw: str) -> dict | None:
    """Extract JSON from AI response, handling markdown fences."""
    raw = raw.strip()
    # Strip markdown fences
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
    # Find first { to last }
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(raw[start:end])
    except json.JSONDecodeError:
        return None

def verify_and_patch_days(days: list, targets: dict) -> list:
    """Ensure each day has all 4 meals; patch missing fields."""
    meal_keys = ["breakfast","lunch","snack","dinner"]
    for day in days:
        for mk in meal_keys:
            m = day.get(mk)
            if not m or not isinstance(m, dict):
                day[mk] = {
                    "name": f"Balanced {mk.capitalize()}",
                    "items": ["balanced meal"],
                    "calories": targets["calories"] // 4,
                    "protein":  targets["protein"]  // 4,
                    "carbs":    targets["carbs"]     // 4,
                    "fat":      targets["fat"]       // 4,
                    "recipe":   "1. Prepare balanced ingredients.\\n2. Cook as preferred.\\n3. Season to taste.\\n4. Serve fresh.",
                }
            else:
                if "recipe" not in m or not m["recipe"]:
                    m["recipe"] = "1. Prepare ingredients as listed.\\n2. Cook using standard method.\\n3. Season to taste."
                for field in ["calories","protein","carbs","fat"]:
                    if field not in m:
                        m[field] = targets[field] // 4
    return days

# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════

@app.get("/")
def home():
    return {
        "message": "MealMate API 🚀",
        "groq":    "connected" if groq_client else "NOT configured — add GROQ_API_KEY to .env",
    }

@app.get("/auth/login")
async def main_login():
    return RedirectResponse(url="/fitness/auth/login")

@app.post("/chat")
async def chat(req: ChatRequest):
    if not is_diet_related(req.message):
        return {"reply": "I'm NutriCore AI 🥗 I specialise in diet, fitness and health. Ask me anything in those areas!"}
    if not groq_client:
        return {"reply": "⚠️ AI not configured. Add GROQ_API_KEY to .env and restart."}
    try:
        res = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": req.message},
            ],
            max_tokens=400,
            temperature=0.7,
        )
        return {"reply": res.choices[0].message.content}
    except Exception as e:
        print(f"Chat error: {e}")
        return {"reply": "Sorry, I'm having trouble right now. Please try again in a moment."}

@app.post("/user/bmi/save")
async def save_bmi(payload: BmiSaveRequest):
    db = get_db()
    if not db:
        return {"success": False, "detail": "Firestore not configured"}
    try:
        db.collection("users").document(payload.uid).set(
            {"email": payload.email, "latest_bmi": payload.model_dump()},
            merge=True,
        )
        return {"success": True}
    except Exception as e:
        return {"success": False, "detail": str(e)}

@app.post("/user/bmi/analyze")
async def analyze_bmi(payload: BmiAnalyzeRequest):
    return {"success": True, "analysis": build_bmi_analysis(payload)}

@app.post("/user/meal/generate")
async def generate_meal(payload: MealGenerateRequest):
    targets = resolve_macros(payload)
    goal    = payload.goal_override or "maintain"
    medical = payload.medical_condition

    # Build medical note for response
    if medical != "none":
        medical_note = f"Plan adjusted for {medical}. {medical_instruction(medical)}"
    else:
        medical_note = f"Plan aligned with your {goal} goal: {targets['calories']} kcal / {targets['protein']}g protein daily."

    # ── Try AI generation ──────────────────────────────────────
    ai_data     = None
    days        = []
    ai_insight  = None

    if groq_client:
        prompt = build_ai_prompt(payload, targets)
        for attempt in range(2):
            try:
                resp = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4000,
                    temperature=0.7 if attempt == 0 else 0.4,
                )
                raw      = resp.choices[0].message.content
                ai_data  = parse_ai_json(raw)
                if ai_data and "days" in ai_data and len(ai_data["days"]) >= 7:
                    days       = verify_and_patch_days(ai_data["days"], targets)
                    ai_insight = ai_data.get("ai_insight")
                    break
                else:
                    print(f"AI attempt {attempt+1}: incomplete response, retrying...")
            except Exception as e:
                print(f"AI attempt {attempt+1} error: {e}")

    # ── Fallback if AI failed ──────────────────────────────────
    if not days:
        print("Using deterministic fallback meal plan")
        days = build_fallback_plan(payload, targets)
        ai_insight = {
            "summary":      f"This plan is built for your {goal} goal targeting {targets['calories']} kcal and {targets['protein']}g protein daily.",
            "focus_foods":  ["dal","paneer / chicken","oats","curd","seasonal vegetables"],
            "avoid_foods":  ["fried foods","sugary drinks","ultra-processed snacks"],
            "weekly_target":f"Hit {targets['calories']} kcal and {targets['protein']}g protein on at least 5 days this week.",
            "tip":          "Prep your proteins in bulk on Sunday to make weekday meals faster and more consistent.",
        }

    meal_plan = {
        "diet_type":    payload.diet_type,
        "medical_note": medical_note,
        "targets":      targets,
        "days":         days,
    }

    # ── Save to Firestore ──────────────────────────────────────
    db = get_db()
    if db:
        try:
            db.collection("users").document(payload.uid).set(
                {"latest_meal_plan": meal_plan}, merge=True
            )
        except Exception as e:
            print(f"Firestore save error: {e}")

    return {"success": True, "meal_plan": meal_plan, "ai_insight": ai_insight}

@app.get("/user/meal/latest/{uid}")
async def get_latest_meal(uid: str):
    db = get_db()
    if not db:
        return {"success": False, "detail": "Firestore not configured"}
    try:
        doc = db.collection("users").document(uid).get()
        if doc.exists:
            plan = (doc.to_dict() or {}).get("latest_meal_plan")
            if plan:
                return {"success": True, "meal_plan": plan}
        return {"success": False, "detail": "No saved meal plan"}
    except Exception as e:
        return {"success": False, "detail": str(e)}

@app.get("/user/bmi/latest/{uid}")
async def get_latest_bmi(uid: str):
    db = get_db()
    if not db:
        return {"success": False, "data": None}
    try:
        doc = db.collection("users").document(uid).get()
        if doc.exists:
            return {"success": True, "data": (doc.to_dict() or {}).get("latest_bmi")}
        return {"success": False, "data": None}
    except Exception as e:
        return {"success": False, "data": None}

# ═══════════════════════════════════════════════════════════════
# DETERMINISTIC FALLBACK MEAL PLAN
# ═══════════════════════════════════════════════════════════════

def build_fallback_plan(req: MealGenerateRequest, targets: dict) -> list:
    """
    7 unique days of Indian+American meals with recipes.
    Respects veg / mixed / nonveg and medical conditions.
    """
    is_veg   = req.diet_type == "veg"
    is_mixed = req.diet_type == "mixed"
    # non-veg days for mixed (days 1,3,5 = non-veg; rest veg)
    nonveg_days = {0, 2, 4} if is_mixed else (set(range(7)) if not is_veg else set())

    cal = targets["calories"]
    pro = targets["protein"]

    # Split: breakfast 25%, lunch 35%, snack 12%, dinner 28%
    def split(c, p):
        return (round(c*.25), round(c*.35), round(c*.12), round(c*.28),
                round(p*.24), round(p*.36), round(p*.10), round(p*.30))

    bc,lc,sc,dc, bp,lp,sp,dp = split(cal, pro)

    # Unique daily meal sets
    VEG_BREAKFASTS = [
        {
            "name":"Masala Oats Bowl",
            "items":[f"{round(bc/4)}g oats","1 cup milk","1 banana","10 almonds"],
            "recipe":"1. Boil oats in milk 5 min stirring.\n2. Add sliced banana on top.\n3. Crush almonds and sprinkle.\n4. Sweeten with 1 tsp honey if needed."
        },
        {
            "name":"Paneer Paratha with Curd",
            "items":["2 paneer parathas","100g curd","1 tsp pickle"],
            "recipe":"1. Mix crumbled paneer with spices for stuffing.\n2. Stuff into wheat dough and roll flat.\n3. Cook on tawa with 1 tsp ghee each side 3 min.\n4. Serve hot with curd."
        },
        {
            "name":"Avocado Toast with Sprouts",
            "items":["2 whole wheat bread slices","½ avocado","50g sprouts","lemon juice"],
            "recipe":"1. Toast bread slices until golden.\n2. Mash avocado with lemon juice and salt.\n3. Spread avocado on toast.\n4. Top with seasoned sprouts."
        },
        {
            "name":"Vegetable Upma",
            "items":[f"{round(bc/5)}g semolina","mixed veg (carrot, peas)","mustard seeds","curry leaves"],
            "recipe":"1. Dry roast semolina 3 min, set aside.\n2. Sauté mustard seeds, curry leaves, chopped veg 4 min.\n3. Add 1.5x water, bring to boil.\n4. Stir in semolina, cook 5 min until thick."
        },
        {
            "name":"Greek Yogurt Parfait",
            "items":["150g Greek yogurt","30g granola","mixed berries / pomegranate","1 tsp chia seeds"],
            "recipe":"1. Layer Greek yogurt in a bowl.\n2. Add granola on top.\n3. Add berries or pomegranate seeds.\n4. Sprinkle chia seeds and serve."
        },
        {
            "name":"Moong Dal Chilla",
            "items":["100g moong dal (soaked)","green chilli","ginger","coriander"],
            "recipe":"1. Soak moong dal 4h, blend to batter.\n2. Add chopped chilli, ginger, salt.\n3. Pour ladle of batter on hot tawa, spread thin.\n4. Cook 2 min each side with 1 tsp oil."
        },
        {
            "name":"Oatmeal Banana Pancakes",
            "items":["60g oat flour","1 egg (or flaxseed egg)","1 ripe banana","cinnamon"],
            "recipe":"1. Mash banana, mix with oat flour, egg/flaxseed egg and cinnamon.\n2. Add milk to get pourable batter.\n3. Cook on non-stick pan 2 min each side.\n4. Serve with 1 tsp honey."
        },
    ]

    NONVEG_BREAKFASTS = [
        {
            "name":"Masala Omelette with Toast",
            "items":["3 eggs","1 whole wheat toast","onion, tomato, capsicum","1 tsp olive oil"],
            "recipe":"1. Beat eggs with salt, pepper and chopped veg.\n2. Heat oil in pan on medium.\n3. Pour mix and cook 2 min, flip, cook 2 more min.\n4. Serve with toasted bread."
        },
        {
            "name":"Boiled Egg Salad Bowl",
            "items":["3 boiled eggs","cucumber","tomato","lettuce","lemon dressing"],
            "recipe":"1. Boil eggs 10 min, peel and halve.\n2. Chop cucumber, tomato, lettuce.\n3. Mix all with lemon juice, olive oil, salt.\n4. Arrange eggs on top."
        },
        {
            "name":"Chicken Poha",
            "items":["100g flattened rice (poha)","80g shredded cooked chicken","mustard seeds","onion"],
            "recipe":"1. Rinse poha, drain and set aside.\n2. Sauté mustard seeds, onion until golden.\n3. Add poha and cooked chicken, mix well.\n4. Season with salt, lemon and coriander."
        },
    ]

    VEG_LUNCHES = [
        {
            "name":"Dal Tadka with Brown Rice",
            "items":["1 cup masoor dal","1 cup brown rice","1 tsp ghee","cumin, garlic tadka"],
            "recipe":"1. Pressure cook dal with turmeric for 3 whistles.\n2. Heat ghee, add cumin and garlic for tadka.\n3. Pour tadka over dal, cook 5 min.\n4. Serve with steamed brown rice."
        },
        {
            "name":"Chole with Whole Wheat Roti",
            "items":["1 cup chickpeas","2 rotis","tomato-onion gravy","spices"],
            "recipe":"1. Soak chickpeas overnight, pressure cook 4 whistles.\n2. Sauté onion-tomato masala in 1 tsp oil.\n3. Add chickpeas and spices, simmer 10 min.\n4. Serve with fresh rotis."
        },
        {
            "name":"Quinoa Buddha Bowl",
            "items":["100g quinoa","roasted chickpeas","cucumber","tomato","tahini dressing"],
            "recipe":"1. Cook quinoa 1:2 water, 15 min.\n2. Roast chickpeas with olive oil and cumin at 200°C 20 min.\n3. Arrange quinoa, chickpeas and veggies in bowl.\n4. Drizzle with tahini-lemon dressing."
        },
        {
            "name":"Palak Paneer with Roti",
            "items":["150g paneer","2 cups spinach","1 tsp cream","2 rotis","spices"],
            "recipe":"1. Blanch spinach, blend to puree.\n2. Sauté onion-tomato-garlic masala in 1 tsp oil.\n3. Add spinach puree, paneer cubes and cream.\n4. Simmer 8 min. Serve with rotis."
        },
        {
            "name":"Rajma Chawal",
            "items":["1 cup rajma","1 cup basmati rice","tomato onion gravy","spices"],
            "recipe":"1. Soak rajma overnight, pressure cook 5 whistles.\n2. Sauté onion-tomato-ginger-garlic masala.\n3. Add rajma and spices, simmer 15 min.\n4. Serve with steamed basmati rice."
        },
        {
            "name":"Grilled Veggie Sandwich",
            "items":["4 whole wheat slices","paneer slices","capsicum","tomato","green chutney"],
            "recipe":"1. Spread chutney on bread slices.\n2. Layer paneer, capsicum and tomato slices.\n3. Close sandwich and grill 4 min each side.\n4. Cut diagonally and serve hot."
        },
        {
            "name":"Mixed Dal Khichdi",
            "items":["50g yellow dal","50g green moong","100g rice","turmeric","ghee"],
            "recipe":"1. Wash dal and rice together.\n2. Pressure cook with turmeric and salt for 4 whistles.\n3. Heat ghee with cumin and pour as tadka.\n4. Mix well and serve with curd."
        },
    ]

    NONVEG_LUNCHES = [
        {
            "name":"Chicken Rice Bowl",
            "items":["150g grilled chicken breast","1 cup brown rice","salad","lemon"],
            "recipe":"1. Marinate chicken in lemon, olive oil, garlic 30 min.\n2. Grill or pan-cook 6 min each side.\n3. Slice and serve over steamed brown rice.\n4. Add salad on the side."
        },
        {
            "name":"Fish Curry with Rice",
            "items":["150g fish (rohu/tilapia)","1 cup rice","coconut milk curry","spices"],
            "recipe":"1. Sauté onion-tomato-ginger-garlic in coconut oil.\n2. Add spices and coconut milk, bring to simmer.\n3. Add fish pieces, cook 10 min gently.\n4. Serve with steamed rice."
        },
        {
            "name":"Egg Fried Rice",
            "items":["2 eggs","1 cup cooked rice","mixed veg","soy sauce","sesame oil"],
            "recipe":"1. Heat oil in wok on high heat.\n2. Scramble eggs, push to side.\n3. Add cooked rice and veg, stir fry 3 min.\n4. Add soy sauce and sesame oil, toss and serve."
        },
    ]

    VEG_SNACKS = [
        {"name":"Mixed Nuts & Fruit","items":["20g almonds","10g walnuts","1 apple"],"recipe":"1. Portion nuts into a bowl.\n2. Slice apple.\n3. Eat slowly and mindfully."},
        {"name":"Hummus with Veggie Sticks","items":["60g hummus","cucumber sticks","carrot sticks","2 rice cakes"],"recipe":"1. Slice cucumber and carrot into sticks.\n2. Serve with hummus as dip.\n3. Optionally add rice cakes."},
        {"name":"Curd with Berries","items":["150g curd","mixed berries","1 tsp flaxseeds"],"recipe":"1. Add curd to bowl.\n2. Top with berries and flaxseeds.\n3. Serve chilled."},
        {"name":"Banana Peanut Butter","items":["1 banana","2 tsp peanut butter"],"recipe":"1. Peel and slice banana.\n2. Serve with peanut butter for dipping."},
        {"name":"Roasted Makhana","items":["30g makhana (fox nuts)","1 tsp ghee","rock salt"],"recipe":"1. Heat ghee in pan.\n2. Add makhana and roast on low heat 8 min.\n3. Sprinkle rock salt. Let cool before eating."},
        {"name":"Protein Smoothie","items":["1 banana","200ml milk","30g oats","1 tsp honey"],"recipe":"1. Add all ingredients to blender.\n2. Blend 60 seconds until smooth.\n3. Serve immediately."},
        {"name":"Chana Chaat","items":["100g boiled white chana","onion","tomato","chaat masala","lemon"],"recipe":"1. Mix boiled chana with chopped onion and tomato.\n2. Add chaat masala, lemon juice, salt.\n3. Toss well and serve."},
    ]

    VEG_DINNERS = [
        {
            "name":"Paneer Bhurji with Roti",
            "items":["150g paneer","2 rotis","capsicum, onion, tomato","spices"],
            "recipe":"1. Crumble paneer. Sauté onion and capsicum in 1 tsp oil.\n2. Add tomato and cook 3 min.\n3. Add paneer, salt, turmeric, garam masala. Cook 5 min.\n4. Serve hot with rotis."
        },
        {
            "name":"Vegetable Dal Soup with Bread",
            "items":["1 cup mixed dal","mixed veg","2 whole wheat bread slices","herbs"],
            "recipe":"1. Cook dal and veg together with water until soft.\n2. Blend half for creamy texture.\n3. Add herbs, salt and simmer 5 min.\n4. Serve with toasted bread."
        },
        {
            "name":"Tofu Stir Fry with Rice",
            "items":["150g firm tofu","1 cup rice","broccoli, bell pepper","soy sauce"],
            "recipe":"1. Press and cube tofu, pan fry until golden.\n2. Stir fry veg in 1 tsp oil on high heat 4 min.\n3. Add soy sauce and tofu, toss 2 min.\n4. Serve over steamed rice."
        },
        {
            "name":"Methi Thepla with Raita",
            "items":["3 methi theplas","100g curd raita","cucumber"],
            "recipe":"1. Mix flour, fenugreek leaves, spices and curd into dough.\n2. Roll thin and cook on tawa 2 min each side with oil.\n3. Prepare raita with curd, cucumber, cumin.\n4. Serve hot theplas with cool raita."
        },
        {
            "name":"Pasta Primavera (Whole Wheat)",
            "items":["80g whole wheat pasta","mixed veg","tomato sauce","parmesan/nutritional yeast"],
            "recipe":"1. Boil pasta al dente per packet instructions.\n2. Sauté garlic, zucchini, tomato in olive oil 5 min.\n3. Mix pasta into veg with tomato sauce.\n4. Top with nutritional yeast or parmesan."
        },
        {
            "name":"Lentil Vegetable Stew with Roti",
            "items":["1 cup red lentils","mixed veg","coconut milk","2 rotis"],
            "recipe":"1. Sauté onion, garlic in pot. Add spices.\n2. Add lentils, veg and 2 cups water.\n3. Simmer 20 min. Stir in coconut milk.\n4. Serve with rotis."
        },
        {
            "name":"Aloo Gobi with Brown Rice",
            "items":["1 medium potato","1 cup cauliflower","1 cup brown rice","spices"],
            "recipe":"1. Cut potato and cauliflower into pieces.\n2. Sauté onion, ginger, garlic in oil.\n3. Add veg and spices, cook covered 15 min.\n4. Serve with steamed brown rice."
        },
    ]

    NONVEG_DINNERS = [
        {
            "name":"Chicken Curry with Roti",
            "items":["150g chicken","2 rotis","onion-tomato gravy","spices"],
            "recipe":"1. Marinate chicken in yogurt and spices 30 min.\n2. Brown onion-tomato-ginger-garlic masala.\n3. Add chicken, cook covered 20 min.\n4. Garnish with coriander. Serve with rotis."
        },
        {
            "name":"Grilled Salmon with Quinoa",
            "items":["150g salmon fillet","100g quinoa","steamed broccoli","lemon-herb dressing"],
            "recipe":"1. Season salmon with lemon, herbs, olive oil.\n2. Grill 4 min each side.\n3. Cook quinoa, steam broccoli.\n4. Serve salmon over quinoa with broccoli."
        },
        {
            "name":"Egg Curry with Rice",
            "items":["3 boiled eggs","1 cup rice","spicy tomato gravy","coriander"],
            "recipe":"1. Boil eggs, halve and fry lightly.\n2. Make tomato-onion-spice gravy.\n3. Add eggs to gravy, simmer 8 min.\n4. Serve with steamed rice."
        },
    ]

    days = []
    for i, day_name in enumerate(DAYS):
        use_nonveg = i in nonveg_days

        bf_list = NONVEG_BREAKFASTS if use_nonveg and NONVEG_BREAKFASTS else VEG_BREAKFASTS
        ln_list = NONVEG_LUNCHES   if use_nonveg and NONVEG_LUNCHES   else VEG_LUNCHES
        dn_list = NONVEG_DINNERS   if use_nonveg and NONVEG_DINNERS   else VEG_DINNERS

        bf = dict(bf_list[i % len(bf_list)])
        ln = dict(ln_list[i % len(ln_list)])
        sn = dict(VEG_SNACKS[i % len(VEG_SNACKS)])
        dn = dict(dn_list[i % len(dn_list)])

        # Assign macro targets per meal
        bf.update({"calories": bc, "protein": bp, "carbs": targets["carbs"]//4, "fat": targets["fat"]//4})
        ln.update({"calories": lc, "protein": lp, "carbs": targets["carbs"]//3, "fat": targets["fat"]//3})
        sn.update({"calories": sc, "protein": sp, "carbs": targets["carbs"]//8, "fat": targets["fat"]//8})
        dn.update({"calories": dc, "protein": dp, "carbs": targets["carbs"]//3, "fat": targets["fat"]//3})

        days.append({"day": day_name, "breakfast": bf, "lunch": ln, "snack": sn, "dinner": dn})

    return days

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("APP_PORT", 8001)), reload=True)