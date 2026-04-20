# MealMate Nutrient Matching TODO

## Step 1: Add BMI/TDEE Calculator [✅]

- Add `calculate_bmi_tdee()` in main.py using Harris-Benedict formula
- New endpoint `/user/bmi/calculate` [✅]

## Step 2: Food Macro Database [✅]

- Define `FOOD_MACROS` dict with Indian foods (per 100g: cal/protein/carbs/fat) [✅]

## Step 3: Dynamic Quantities & Meal Builder [✅]

- Update `build_daily_meals()`: Dynamic grams based on macros [✅]
- Output: "150g paneer (30g protein)" [✅]

## Step 4: Buffer + Extra Meals [✅]

- Targets +10% buffer [✅]
- If short, add 1-2 extra snacks [✅]

## Step 5: Verify & Integrate [✅]

- Sum validation in `build_meal_plan()` [✅]
- Use BMI-derived targets if no override [✅]

## Step 6: Test [ ]

- BMI calc endpoint
- Meal gen with quantities/extras
