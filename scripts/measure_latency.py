"""Task 11: measure the latency of /recommend over 50 requests (target < 2 s).

Run it with the API already running (python -m uvicorn api.main:app):
    python scripts/measure_latency.py

It plays real students over HTTP, like the Streamlit app does:
- 5 test profiles with different diets and allergies,
- 3 of them rate 3 meals, so both paths are timed:
  "popularity" (new student) and "model" (fold-in on the liked meals),
- 50 timed calls to /recommend, then median, p95 and max,
- the result is written to reports/latency.md,
- the test profiles and their ratings are deleted at the end.
"""

import argparse
import sqlite3
import statistics
import time
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "studeat.db"
REPORT = ROOT / "reports" / "latency.md"
TARGET_S = 2.0

PROFILES = [
    {"diet": "none", "allergies": []},
    {"diet": "vegetarian", "allergies": ["gluten"]},
    {"diet": "vegan", "allergies": ["nuts", "peanuts"]},
    {"diet": "halal", "allergies": ["lactose"]},
    {"diet": "pescatarian", "allergies": ["shellfish"]},
]


def percentile(values, p):
    """p-th percentile (nearest rank): p95 = 95 % of the requests are faster."""
    ordered = sorted(values)
    rank = max(1, round(p / 100 * len(ordered)))
    return ordered[rank - 1]


def create_users(api, users):
    """Fill `users` as we go, so the cleanup also works if a request fails midway."""
    for i, p in enumerate(PROFILES):
        r = requests.post(f"{api}/profile", timeout=10, json={
            "name": f"latency-test-{i}", "diet": p["diet"], "allergies": p["allergies"],
            "budget_min": 30, "budget_max": 50, "meals_per_week": 10})
        r.raise_for_status()
        users.append(r.json()["user_id"])


def rate_some_meals(api, user_ids):
    """The student likes 3 meals of their first box, so the next box uses the model."""
    for uid in user_ids:
        box = requests.get(f"{api}/recommend/{uid}", timeout=30).json()["meals"]
        for meal in box[:3]:
            requests.post(f"{api}/log_interaction", timeout=10,
                          json={"user_id": uid, "meal_id": meal["id"], "rating": 5}).raise_for_status()


def delete_users(user_ids):
    """Remove the test students so they never show up in the app data."""
    marks = ",".join("?" * len(user_ids))
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"DELETE FROM app_ratings WHERE user_id IN ({marks})", user_ids)
        conn.execute(f"DELETE FROM app_users WHERE user_id IN ({marks})", user_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("-n", "--requests", type=int, default=50)
    args = parser.parse_args()

    health = requests.get(f"{args.api}/health", timeout=10).json()
    users = []
    try:
        create_users(args.api, users)
        rate_some_meals(args.api, users[:3])
        requests.get(f"{args.api}/recommend/{users[0]}", timeout=30)  # warm-up, not counted

        times, sources = [], {}
        for i in range(args.requests):
            uid = users[i % len(users)]
            start = time.perf_counter()
            r = requests.get(f"{args.api}/recommend/{uid}", timeout=30)
            times.append(time.perf_counter() - start)
            r.raise_for_status()
            source = r.json().get("model", "?")
            sources[source] = sources.get(source, 0) + 1
    finally:
        if users:
            delete_users(users)

    median, p95, worst = statistics.median(times), percentile(times, 95), max(times)
    verdict = "OK" if p95 < TARGET_S else "TOO SLOW"
    lines = [
        "# Latency of /recommend",
        "",
        f"- Date: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Requests: {len(times)} over HTTP, {len(users)} test students, warm-up excluded",
        f"- Catalogue: {health.get('catalogue')} ({health.get('meals_in_catalogue')} meals), "
        f"model loaded: {health.get('model_loaded')}",
        f"- Ranking source: " + ", ".join(f"{k} = {v}" for k, v in sorted(sources.items())),
        "",
        "| Median | p95 | Max | Target |",
        "|---|---|---|---|",
        f"| {median * 1000:.0f} ms | {p95 * 1000:.0f} ms | {worst * 1000:.0f} ms | < {TARGET_S:.0f} s ({verdict}) |",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"-> {REPORT.relative_to(ROOT)} (test students deleted)")


if __name__ == "__main__":
    main()
