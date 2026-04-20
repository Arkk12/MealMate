<<<<<<< HEAD
# MealMate
=======
# MealMate 🚀 AI Nutrition + Real-time Fitness

## Quick Start (Double-click)

```
double-click start.bat
```

**3 windows open:**

- Backend API (uvicorn port 8001)
- Frontend server (port 5500)
- Instructions

**Access:** `http://localhost:5500/frontend/index.html`

## Features

- ✅ **BMI Calculator** + AI health analysis
- ✅ **7-day Meal Plans** (veg/mixed/nonveg + recipes)
- ✅ **Live Fitness** (Google Fit → 10s step/calorie tracking)
- ✅ **AI Chat** (nutrition advice)
- ✅ **Firebase** auth + data sync

## Manual

```
# Backend
uvicorn backend.main:app --host 0.0.0.0 --port 8001 --reload

# Frontend (new terminal)
python -m http.server 5500
```

## Google Fit Setup (One-time)

1. Backend shows `✅ Google OAuth ready`
2. Fitness tab → Connect → Google login → Live dashboard!

## Production

```
npm install -g pm2
pm2 start ecosystem.config.js
```

**Enjoy real-time health tracking! 💪**
>>>>>>> 04581cc (clean start)
