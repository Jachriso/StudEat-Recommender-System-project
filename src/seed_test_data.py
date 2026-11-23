"""
Seed test student profiles and ratings into app_users / app_ratings.

Usage:
    python src/seed_test_data.py           # insert test data
    python src/seed_test_data.py --clean   # remove all TEST_ users and their ratings

This lets the SE and MLE team test the full chain (profile → recommend →
log_interaction) without waiting for a real rating campaign.
Test users are identifiable by the "TEST_" prefix on their name.
"""

import argparse
import json
import random
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "studeat.db"

# ── Meal IDs from mock_data.py (IDs 1-30) ────────────────────────────────────
ALL_MEAL_IDS = list(range(1, 31))

# Meals that fit each diet (subset used for realistic ratings)
MEALS_BY_DIET = {
    "none":          list(range(1, 31)),
    "vegetarian":    [2, 5, 8, 10, 12, 14, 16, 18, 20, 23, 25, 28, 30],
    "vegan":         [2, 10, 12, 14, 18, 20, 28],
    "halal":         [1, 2, 4, 5, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 21, 24, 27, 28, 29, 30],
    "pescatarian":   [2, 5, 6, 8, 10, 12, 14, 16, 17, 18, 20, 22, 23, 25, 26, 28, 30],
}

# ── Test profiles ─────────────────────────────────────────────────────────────
TEST_USERS = [
    {
        "name": "TEST_Alice",
        "diet": "vegetarian",
        "allergies": ["gluten"],
        "budget_min": 30.0,
        "budget_max": 50.0,
        "meals_per_week": 7,
        "ratings": {2: 5, 10: 4, 18: 5, 20: 4, 28: 3, 14: 5, 12: 4},
    },
    {
        "name": "TEST_Bob",
        "diet": "none",
        "allergies": [],
        "budget_min": 40.0,
        "budget_max": 60.0,
        "meals_per_week": 10,
        "ratings": {1: 5, 3: 4, 7: 5, 11: 3, 19: 5, 24: 4, 27: 2, 9: 4, 15: 3, 6: 5},
    },
    {
        "name": "TEST_Camille",
        "diet": "vegan",
        "allergies": ["nuts", "sesame"],
        "budget_min": 25.0,
        "budget_max": 40.0,
        "meals_per_week": 7,
        "ratings": {2: 4, 10: 5, 14: 5, 18: 4, 20: 5},
    },
    {
        "name": "TEST_Dylan",
        "diet": "halal",
        "allergies": ["lactose"],
        "budget_min": 35.0,
        "budget_max": 55.0,
        "meals_per_week": 10,
        "ratings": {1: 4, 2: 3, 9: 5, 13: 4, 21: 5, 24: 5, 29: 3, 11: 4, 14: 2, 18: 4},
    },
    {
        "name": "TEST_Emma",
        "diet": "pescatarian",
        "allergies": ["eggs"],
        "budget_min": 45.0,
        "budget_max": 70.0,
        "meals_per_week": 10,
        "ratings": {6: 5, 17: 3, 22: 4, 26: 5, 2: 4, 10: 3, 16: 4, 20: 5, 28: 4, 18: 5},
    },
    {
        "name": "TEST_Felix",
        "diet": "none",
        "allergies": ["peanuts", "soy"],
        "budget_min": 30.0,
        "budget_max": 45.0,
        "meals_per_week": 7,
        "ratings": {1: 3, 2: 4, 15: 5, 16: 4, 18: 5, 19: 4, 20: 3},
    },
    {
        "name": "TEST_Grace",
        "diet": "vegetarian",
        "allergies": [],
        "budget_min": 35.0,
        "budget_max": 60.0,
        "meals_per_week": 10,
        "ratings": {2: 5, 5: 4, 8: 3, 10: 5, 12: 4, 14: 5, 16: 5, 18: 4, 20: 3, 30: 4},
    },
    {
        "name": "TEST_Hugo",
        "diet": "none",
        "allergies": ["gluten", "lactose"],
        "budget_min": 40.0,
        "budget_max": 65.0,
        "meals_per_week": 10,
        "ratings": {1: 4, 2: 5, 14: 4, 18: 5, 19: 3, 20: 4, 21: 5, 24: 4, 28: 3, 29: 5},
    },
]


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_tables() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS app_users (
                user_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                name           TEXT    NOT NULL,
                diet           TEXT    NOT NULL,
                allergies      TEXT    NOT NULL,
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


def clean_test_data() -> None:
    """Remove all TEST_ users and their ratings."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT user_id FROM app_users WHERE name LIKE 'TEST_%'"
        ).fetchall()
        ids = [r["user_id"] for r in rows]
        if not ids:
            print("No test users found.")
            return
        placeholders = ",".join("?" * len(ids))
        conn.execute(f"DELETE FROM app_ratings WHERE user_id IN ({placeholders})", ids)
        conn.execute(f"DELETE FROM app_users  WHERE user_id IN ({placeholders})", ids)
        print(f"Removed {len(ids)} test users and their ratings.")


def seed() -> None:
    """Insert test users and ratings, skipping users that already exist."""
    _ensure_tables()

    with _get_conn() as conn:
        existing = {
            r["name"]
            for r in conn.execute("SELECT name FROM app_users WHERE name LIKE 'TEST_%'")
        }

    inserted_users = 0
    inserted_ratings = 0

    # Spread ratings over the past 30 days for realism
    now = datetime.now(timezone.utc)

    with _get_conn() as conn:
        for user in TEST_USERS:
            if user["name"] in existing:
                print(f"  skip (already exists): {user['name']}")
                continue

            cur = conn.execute(
                "INSERT INTO app_users "
                "(name, diet, allergies, budget_min, budget_max, meals_per_week, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    user["name"],
                    user["diet"],
                    json.dumps(user["allergies"]),
                    user["budget_min"],
                    user["budget_max"],
                    user["meals_per_week"],
                    (now - timedelta(days=random.randint(5, 30))).isoformat(),
                ),
            )
            user_id = cur.lastrowid
            inserted_users += 1

            for meal_id, rating in user["ratings"].items():
                ts = now - timedelta(
                    days=random.randint(0, 20),
                    hours=random.randint(0, 23),
                )
                conn.execute(
                    "INSERT INTO app_ratings (user_id, meal_id, rating, timestamp) "
                    "VALUES (?, ?, ?, ?)",
                    (user_id, meal_id, rating, ts.isoformat()),
                )
                inserted_ratings += 1

            print(f"  inserted: {user['name']} (user_id={user_id}, "
                  f"{len(user['ratings'])} ratings, diet={user['diet']})")

    print(f"\nDone: {inserted_users} users, {inserted_ratings} ratings added.")


def summary() -> None:
    """Print a quick summary of test data in the DB."""
    with _get_conn() as conn:
        n_users = conn.execute(
            "SELECT COUNT(*) FROM app_users WHERE name LIKE 'TEST_%'"
        ).fetchone()[0]
        n_ratings = conn.execute(
            "SELECT COUNT(*) FROM app_ratings WHERE user_id IN "
            "(SELECT user_id FROM app_users WHERE name LIKE 'TEST_%')"
        ).fetchone()[0]
        avg = conn.execute(
            "SELECT AVG(rating) FROM app_ratings WHERE user_id IN "
            "(SELECT user_id FROM app_users WHERE name LIKE 'TEST_%')"
        ).fetchone()[0]
    print(f"\nTest data in DB: {n_users} users, {n_ratings} ratings "
          f"(avg rating: {avg:.2f})" if avg else f"{n_users} users, {n_ratings} ratings")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed or clean test data")
    parser.add_argument("--clean", action="store_true", help="Remove all TEST_ data")
    args = parser.parse_args()

    if args.clean:
        clean_test_data()
    else:
        print("Seeding test users and ratings...")
        seed()
        summary()
