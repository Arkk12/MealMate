from pydantic import BaseModel

class BmiCalculateRequest(BaseModel):
    weight: float
    height: float
    age: int
    gender: str
    activity: str = "moderate"
    goal: str = "maintain"

