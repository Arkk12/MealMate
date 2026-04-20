# fitness.py — Real-time Google Fit integration
from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse
import httpx
import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

fitness_router = APIRouter(prefix="/fitness", tags=["Fitness"])

CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI  = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8001/fitness/auth/callback")
FRONTEND_URL  = "http://localhost:5500/frontend/index.html"

if CLIENT_ID:
    print(f"✅ Google OAuth ready ({CLIENT_ID[:36]}...)")
else:
    print("⚠️  GOOGLE_CLIENT_ID missing in .env")

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/fitness.activity.read",
    "https://www.googleapis.com/auth/fitness.body.read",
    "https://www.googleapis.com/auth/fitness.location.read",
]


def get_db():
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
        if not firebase_admin._apps:
            cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "serviceAccountKey.json")
            if not os.path.isabs(cred_path):
                cred_path = os.path.join(BASE_DIR, cred_path)
            firebase_admin.initialize_app(credentials.Certificate(cred_path))
        return firestore.client()
    except Exception as e:
        print(f"Firestore error: {e}")
        return None


def extract_sum(data: dict, key: str = "fpVal") -> float:
    try:
        total = 0.0
        for bucket in data.get("bucket", []):
            for ds in bucket.get("dataset", []):
                for pt in ds.get("point", []):
                    total += pt["value"][0].get(key, 0)
        return total
    except Exception:
        return 0.0


def extract_int_sum(data: dict, key: str = "intVal") -> int:
    try:
        total = 0
        for bucket in data.get("bucket", []):
            for ds in bucket.get("dataset", []):
                for pt in ds.get("point", []):
                    total += int(pt["value"][0].get(key, 0))
        return total
    except Exception:
        return 0


async def save_tokens(email, access_token, refresh_token):
    db = get_db()
    if not db:
        return
    try:
        db.collection("fitness_tokens").document(email).set({
            "access_token":  access_token,
            "refresh_token": refresh_token,
            "updated_at":    datetime.utcnow().isoformat(),
        })
    except Exception as e:
        print(f"Token save error: {e}")


async def get_tokens(email):
    db = get_db()
    if not db:
        return None
    try:
        doc = db.collection("fitness_tokens").document(email).get()
        return doc.to_dict() if doc.exists else None
    except Exception as e:
        print(f"Token fetch error: {e}")
        return None


async def delete_tokens(email):
    db = get_db()
    if not db:
        return
    try:
        db.collection("fitness_tokens").document(email).delete()
    except Exception as e:
        print(f"Token delete error: {e}")


async def refresh_access_token(email):
    tokens = await get_tokens(email)
    if not tokens or not tokens.get("refresh_token"):
        return None
    async with httpx.AsyncClient() as client:
        res = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id":     CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "refresh_token": tokens["refresh_token"],
                "grant_type":    "refresh_token",
            },
        )
        new_token = res.json().get("access_token")
        if new_token:
            await save_tokens(email, new_token, tokens["refresh_token"])
        return new_token


async def fetch_fitness_data(token: str) -> dict:
    now      = datetime.utcnow()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start_ms = int(midnight.timestamp() * 1000)
    end_ms   = int(now.timestamp() * 1000)
    if end_ms <= start_ms:
        end_ms = start_ms + 60_000

    async with httpx.AsyncClient(timeout=15.0) as client:
        headers = {"Authorization": f"Bearer {token}"}

        async def fetch(data_type):
            body = {
                "aggregateBy":     [{"dataTypeName": data_type}],
                "bucketByTime":    {"durationMillis": end_ms - start_ms},
                "startTimeMillis": start_ms,
                "endTimeMillis":   end_ms,
            }
            r = await client.post(
                "https://www.googleapis.com/fitness/v1/users/me/dataset:aggregate",
                json=body, headers=headers,
            )
            return r.json()

        steps_d, cal_d, active_d, workout_d, dist_d = await asyncio.gather(
            fetch("com.google.step_count.delta"),
            fetch("com.google.calories.expended"),
            fetch("com.google.active_minutes"),
            fetch("com.google.activity.segment"),
            fetch("com.google.distance.delta"),
            return_exceptions=True,
        )

    for d in [steps_d, cal_d, active_d, dist_d]:
        if isinstance(d, dict) and d.get("error", {}).get("code") == 401:
            return {"_needs_refresh": True}

    steps       = extract_int_sum(steps_d)           if not isinstance(steps_d,  Exception) else 0
    calories    = round(extract_sum(cal_d, "fpVal")) if not isinstance(cal_d,    Exception) else 0
    active_min  = extract_int_sum(active_d)          if not isinstance(active_d, Exception) else 0
    dist_m      = extract_sum(dist_d, "fpVal")       if not isinstance(dist_d,   Exception) else 0.0
    workouts    = 0
    try:
        workouts = len(workout_d["bucket"][0]["dataset"][0]["point"])
    except Exception:
        pass

    return {
        "success":    True,
        "steps":      steps,
        "calories":   calories,
        "active":     active_min,
        "workouts":   workouts,
        "avg":        round(calories / workouts) if workouts > 0 else calories,
        "distance":   round(dist_m / 1000, 2),
        "fetched_at": datetime.utcnow().isoformat(),
    }


@fitness_router.get("/auth/login")
async def fitness_login():
    if not CLIENT_ID:
        raise HTTPException(500, "GOOGLE_CLIENT_ID not set in .env")
    scope_str = "%20".join(SCOPES)
    return RedirectResponse(
        f"https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}"
        f"&response_type=code&scope={scope_str}"
        f"&access_type=offline&prompt=consent%20select_account"
    )


@fitness_router.get("/auth/callback")
async def fitness_callback(code: str = None, error: str = None):
    if error:
        return RedirectResponse(f"{FRONTEND_URL}?fitness=error&reason={error}")
    if not code:
        raise HTTPException(400, "No authorization code")
    async with httpx.AsyncClient() as client:
        token_res = await client.post(
            "https://oauth2.googleapis.com/token",
            data={"code": code, "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET,
                  "redirect_uri": REDIRECT_URI, "grant_type": "authorization_code"},
        )
        tokens = token_res.json()
        access_token  = tokens.get("access_token")
        refresh_token = tokens.get("refresh_token")
        if not access_token:
            print(f"Token exchange failed: {tokens}")
            return RedirectResponse(f"{FRONTEND_URL}?fitness=error&reason=token_failed")
        user_res = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        email = user_res.json().get("email")
        if not email:
            return RedirectResponse(f"{FRONTEND_URL}?fitness=error&reason=no_email")
    await save_tokens(email, access_token, refresh_token)
    print(f"✅ Fitness connected: {email}")
    return RedirectResponse(f"{FRONTEND_URL}?fitness=connected&email={email}")


@fitness_router.get("/auth/status")
async def fitness_auth_status(email: str = None):
    if not email:
        return {"authenticated": False}
    tokens = await get_tokens(email)
    return {"authenticated": bool(tokens), "email": email if tokens else None}


@fitness_router.get("/data")
async def fitness_data(email: str = None):
    if not email:
        return {"error": "not_authenticated"}
    tokens = await get_tokens(email)
    if not tokens:
        return {"error": "not_authenticated"}
    try:
        data = await fetch_fitness_data(tokens["access_token"])
        if data.get("_needs_refresh"):
            new_token = await refresh_access_token(email)
            if not new_token:
                return {"error": "not_authenticated"}
            data = await fetch_fitness_data(new_token)
            if data.get("_needs_refresh"):
                return {"error": "not_authenticated"}
        data["email"] = email
        return data
    except Exception as e:
        print(f"Fitness data error [{email}]: {e}")
        return {"error": "fetch_failed", "detail": str(e)}


@fitness_router.get("/logout")
async def fitness_logout(email: str = None):
    if not email:
        return {"success": False}
    await delete_tokens(email)
    return {"success": True}


@fitness_router.get("/debug")
async def fitness_debug(email: str = None):
    if not email:
        return {"error": "email required"}
    tokens = await get_tokens(email)
    if not tokens:
        return {"error": "not_authenticated"}
    now   = datetime.utcnow()
    start = int((now - timedelta(days=7)).timestamp() * 1000)
    end   = int(now.timestamp() * 1000)
    async with httpx.AsyncClient(timeout=20.0) as client:
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        async def fetch(dt):
            r = await client.post(
                "https://www.googleapis.com/fitness/v1/users/me/dataset:aggregate",
                json={"aggregateBy": [{"dataTypeName": dt}],
                      "bucketByTime": {"durationMillis": 86400000},
                      "startTimeMillis": start, "endTimeMillis": end},
                headers=headers,
            )
            return r.json()
        steps_r, dist_r, cal_r, active_r = await asyncio.gather(
            fetch("com.google.step_count.delta"), fetch("com.google.distance.delta"),
            fetch("com.google.calories.expended"), fetch("com.google.active_minutes"),
        )
    def pts(d):
        try:
            return [pt for b in d.get("bucket",[]) for ds in b.get("dataset",[]) for pt in ds.get("point",[])]
        except Exception:
            return []
    return {"email": email, "steps": pts(steps_r), "distance": pts(dist_r),
            "calories": pts(cal_r), "active": pts(active_r)}