"""StudEat API
Run from the project root:
    python -m uvicorn api.main:app --reload
Interactive docs: http://localhost:8000/docs
"""

import json
import os
import random
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from api.catalogue import ALLERGENS, DIETS, load_meals
from api.model import EMB_PATH, StudEatModel

app = FastAPI(title="StudEat API", version="0.6-model")

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "studeat.db"

# Meal catalogue: Food.com recipes from the data pipeline (mock catalogue if not available).
MEALS, CATALOGUE_SOURCE = load_meals(DB_PATH)
MEAL_IDS = {m["id"] for m in MEALS}

# Recommendation model (NLGCL embeddings + fold-in + risk control), loaded once at startup.
# Plan B: if the embeddings file is missing, the API still works with random scores.
EMB_FILE = Path(os.getenv("STUDEAT_EMB_PATH", EMB_PATH))
try:
    MODEL = StudEatModel(emb_path=EMB_FILE)
except (FileNotFoundError, OSError, KeyError) as e:
    print(f"[StudEat] Model not loaded ({e}). Falling back to random scores.")
    MODEL = None


def _get_conn() -> sqlite3.Connection:
    """Return a SQLite connection with row_factory so rows behave like dicts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_app_tables() -> None:
    """Create the tables of the app if they do not exist yet.

    They are kept separate from the Food.com tables (recipes, interactions, ...)
    so that student data never mixes with the training data and is not erased
    when the data pipeline is run again.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS app_users (
                user_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                name           TEXT    NOT NULL,
                diet           TEXT    NOT NULL,
                allergies      TEXT    NOT NULL,   -- JSON list, e.g. ["gluten"]
                budget_min     REAL    NOT NULL,
                budget_max     REAL    NOT NULL,
                meals_per_week INTEGER NOT NULL,
                created_at     TEXT    NOT NULL
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS app_ratings (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id   INTEGER NOT NULL REFERENCES app_users(user_id),
                meal_id   INTEGER NOT NULL,
                rating    INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                timestamp TEXT    NOT NULL
            )""")


def load_profile(user_id: int) -> dict | None:
    """Read a student profile from the database (None if unknown)."""
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM app_users WHERE user_id = ?", (user_id,)).fetchone()
    if row is None:
        return None
    profile = dict(row)
    profile["allergies"] = json.loads(profile["allergies"])
    return profile


init_app_tables()


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


def hard_filters(allergies: list[str], diet: str) -> list[dict]:
    """Remove meals the student cannot eat (rules, no ML)."""
    return [
        m for m in MEALS
        if diet in m["diets"] and not set(m["allergens"]) & set(allergies)
    ]


def liked_meal_ids(user_id: int) -> list[int]:
    """Meals the student rated 4 or 5 stars in the app (used for the fold-in)."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT meal_id FROM app_ratings WHERE user_id = ? AND rating >= 4", (user_id,)
        ).fetchall()
    return [r["meal_id"] for r in rows]

def disliked_meal_ids(user_id: int) -> set[int]:
    """Meals whose latest rating by the student is 1 or 2 stars: never put them back in the box."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT meal_id, rating, MAX(rowid) FROM app_ratings WHERE user_id = ? GROUP BY meal_id",
            (user_id,),
        ).fetchall()
    return {r["meal_id"] for r in rows if r["rating"] <= 2}

def predict(user_id: int, budget_min: float, budget_max: float,
            allergies: list[str], diet: str, k: int = 10) -> tuple[list[dict], str]:
    """Rank the meals the student can eat; return (top-k meals with a score, source).

    Agreed interface with MLE-Core / MLE-Ops. The ranking itself is done by
    StudEatModel (api/model.py): fold-in on the student's liked meals, popularity
    if they have not rated anything yet, and risk control. Random scores if the
    model could not be loaded (plan B).
    """
    candidates = hard_filters(allergies, diet)
    disliked = disliked_meal_ids(user_id)
    candidates = [m for m in candidates if m["id"] not in disliked] or candidates  # never an empty box
    if MODEL is None:
        random.shuffle(candidates)
        return [{**m, "score": round(random.random(), 3)} for m in candidates[:k]], "random (model not loaded)"
    ranked, source = MODEL.rank(candidates, liked_meal_ids(user_id))
    return ranked[:k], source


def _copies(candidates: list[dict], k: int) -> list[dict]:
    """Meals available for the box: each candidate once, or several times if
    there are fewer candidates than meals to fill (repeats spread evenly)."""
    times = max(1, -(-k // len(candidates)))  # ceil(k / number of candidates)
    return [m for m in candidates for _ in range(times)]


def cheapest_box_cost(candidates: list[dict], k: int) -> float:
    """Price of the cheapest complete box of k meals (same repeat rule as compose_box)."""
    prices = sorted(m["price"] for m in _copies(candidates, k))
    return round(sum(prices[:k]), 2)


def compose_box(ranked: list[dict], budget_max: float, k: int) -> list[dict]:
    """Pick k meals, best score first, without exceeding budget_max.

    Before adding a meal, we check that the budget left can still pay for the
    cheapest meals remaining for the other slots, so the box is always complete
    when a complete box is affordable (budget_max >= cheapest_box_cost).
    A meal is repeated only if there are fewer candidates than slots.
    """
    pool = _copies(ranked, k)
    used = {}  # meal id -> times already in the box (to spread repeats)
    box, total = [], 0.0
    for slot in range(k):
        after = k - slot - 1                                # slots still to fill after this one
        prices = sorted(m["price"] for m in pool)
        reserve_default = sum(prices[:after])               # cheapest way to fill the other slots
        reserve_if_cheap = sum(prices[:after + 1])          # same, if one of those cheap meals is taken now
        choice = None
        for meal in sorted(pool, key=lambda m: (used.get(m["id"], 0), -m["score"])):
            # Budget kept for the other slots once this meal is in the box
            if after and meal["price"] <= prices[after - 1]:
                reserve = reserve_if_cheap - meal["price"]
            else:
                reserve = reserve_default
            if total + meal["price"] + reserve <= budget_max + 1e-9:
                choice = meal
                break
        if choice is None:
            break
        box.append(choice)
        total += choice["price"]
        pool.remove(choice)
        used[choice["id"]] = used.get(choice["id"], 0) + 1
    return box


@app.get("/health")
def health():
    return {"status": "ok", "meals_in_catalogue": len(MEALS), "catalogue": CATALOGUE_SOURCE,
            "model_loaded": MODEL is not None}


@app.get("/options")
def options():
    """Values the frontend can display in its forms."""
    return {"allergens": ALLERGENS, "diets": DIETS}


@app.post("/profile")
def create_profile(profile: Profile):
    p = profile.model_dump()
    with _get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO app_users (name, diet, allergies, budget_min, budget_max, meals_per_week, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (p["name"], p["diet"], json.dumps(p["allergies"]), p["budget_min"], p["budget_max"],
             p["meals_per_week"], datetime.now(timezone.utc).isoformat()),
        )
        user_id = cur.lastrowid
    return {"user_id": user_id, "profile": p}


@app.get("/profile/{user_id}")
def get_profile(user_id: int):
    profile = load_profile(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Unknown user")
    return profile


@app.get("/recommend/{user_id}")
def recommend(user_id: int):
    p = load_profile(user_id)
    if p is None:
        raise HTTPException(status_code=404, detail="Unknown user")
    k = p["meals_per_week"]

    candidates = hard_filters(p["allergies"], p["diet"])
    if not candidates:
        raise HTTPException(status_code=422, detail="No meal matches these allergies and diet")

    
    # Minimum feasible budget: price of the cheapest complete box with these constraints.
    min_feasible = cheapest_box_cost(candidates, k)
    if p["budget_max"] < min_feasible:
        raise HTTPException(
            status_code=422,
            detail=f"Budget too low: a box of {k} meals starts at {min_feasible} EUR with these constraints",
        )

    ranked, source = predict(user_id, p["budget_min"], p["budget_max"], p["allergies"], p["diet"], k=len(candidates))
    box = compose_box(ranked, p["budget_max"], k)
    total = round(sum(m["price"] for m in box), 2)
    return {
        "user_id": user_id,
        "meals": box,
        "total_price": total,
        "within_budget": p["budget_min"] <= total <= p["budget_max"],
        "min_feasible_budget": min_feasible,
        "model": source,
    }


class Interaction(BaseModel):
    user_id: int
    meal_id: int
    rating: int = Field(..., ge=1, le=5, description="1 to 5 stars")


@app.post("/log_interaction")
def log_interaction(interaction: Interaction):
    if load_profile(interaction.user_id) is None:
        raise HTTPException(status_code=404, detail="Unknown user")
    if interaction.meal_id not in MEAL_IDS:
        raise HTTPException(status_code=404, detail="Unknown meal")
    timestamp = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO app_ratings (user_id, meal_id, rating, timestamp) VALUES (?, ?, ?, ?)",
            (interaction.user_id, interaction.meal_id, interaction.rating, timestamp),
        )
    return {"message": "rating saved", "interaction": {**interaction.model_dump(), "timestamp": timestamp}}


@app.get("/interactions/{user_id}")
def get_interactions(user_id: int):
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT meal_id, rating, timestamp FROM app_ratings WHERE user_id = ? ORDER BY timestamp",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]