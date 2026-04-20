@echo off
title MealMate - Real-time Fitness & Nutrition
cd /d "c:/Users/aryan/MealMate"

echo Starting Backend API...
start "MealMate Backend" cmd /k "uvicorn backend.main:app --host 0.0.0.0 --port 8001 --reload"

timeout /t 3 /nobreak >nul
echo Starting Frontend Server...
start "MealMate Frontend" cmd /k "python -m http.server 5500"

echo.
echo ========================================
echo MealMate LIVE! ^_^ 
echo Backend: http://localhost:8001
echo Frontend: http://localhost:5500/frontend/index.html  
echo ========================================
echo Click Backend/Frontend windows to see logs.
pause
