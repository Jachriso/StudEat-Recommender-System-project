"""StudEat API (week 2 version: mock recommendations).

Run from the project root:
    python -m uvicorn api.main:app --reload
Interactive docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from api.mock_data import ALLERGENS, DIETS, MEALS

app = FastAPI(title="StudEat API", version="0.2-mock")

# In-memory storage until the SQLite database is ready (week 3).
PROFILES: dict[int, dict] = {}


class Profile(BaseModel):
    name: str = Field(..., min_length=1)
    allergies: list[str] = []
    diet: str = "none"
    budget_min: float = Field(..., ge=0, description="Weekly budget, EUR")
    budget_max: float = Field(..., gt=0, description="Weekly budget, EUR")
    meals_per_week: int = Field(10, ge=1, le=14)

    @model_validator(mode="after")
    def check_values(self):
        if self.budget_min > self.budget_max:
            raise ValueError("budget_min must be lower than budget_max")
        if self.diet not in DIETS:
            raise ValueError(f"diet must be one of {DIETS}")
        unknown = set(self.allergies) - set(ALLERGENS)
        if unknown:
            raise ValueError(f"unknown allergies: {sorted(unknown)}")
        return self


@app.get("/health")
def health():
    return {"status": "ok", "meals_in_catalogue": len(MEALS)}


@app.get("/options")
def options():
    """Values the frontend can display in its forms."""
    return {"allergens": ALLERGENS, "diets": DIETS}


@app.post("/profile")
def create_profile(profile: Profile):
    user_id = max(PROFILES, default=0) + 1
    PROFILES[user_id] = profile.model_dump()
    return {"user_id": user_id, "profile": PROFILES[user_id]}


@app.get("/profile/{user_id}")
def get_profile(user_id: int):
    if user_id not in PROFILES:
        raise HTTPException(status_code=404, detail="Unknown user")
    return PROFILES[user_id]