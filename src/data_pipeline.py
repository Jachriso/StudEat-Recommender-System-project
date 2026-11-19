"""
StudEat — Data Engineering Pipeline

"""

import ast
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_config import UNIT_TO_G, CATEGORY_TAG_MAP, STATIC_FALLBACK

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
RAW       = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
DB_PATH   = ROOT / "data" / "studeat.db"

PROCESSED.mkdir(parents=True, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_list(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return []


# ═════════════════════════════════════════════════════════════════════════════
# 1. Load and filter recipes by tags
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1 — Load & filter recipes")
print("=" * 60)

recipes_raw = pd.read_csv(RAW / "RAW_recipes.csv")
print(f"  Raw recipes: {len(recipes_raw):,}")

recipes = recipes_raw[["id", "name", "description", "tags", "nutrition", "ingredients"]].copy()

TARGET_TAGS = {
    "main-dish", "one-dish-meal", "stews", "casseroles",
    "pasta-rice-and-grains", "chicken", "beef", "pork",
    "soups-stews", "curries", "oven",
}

recipes["tags_list"] = recipes["tags"].apply(parse_list)
mask = recipes["tags_list"].apply(lambda t: bool(set(t) & TARGET_TAGS))
recipes = recipes[mask].copy()
print(f"  After tag filter: {len(recipes):,} recipes")


# ═════════════════════════════════════════════════════════════════════════════
# 2. Parse nutrition
# ═════════════════════════════════════════════════════════════════════════════
print("\nSTEP 2 — Parse nutrition")

# Food.com nutrition format: [calories, total_fat_%DV, sugar_%DV, sodium_%DV,
#                              protein_%DV, sat_fat_%DV, carbs_%DV]
NUTR_COLS = ["calories", "total_fat_pdv", "sugar_pdv", "sodium_pdv",
             "protein_pdv", "sat_fat_pdv", "carbs_pdv"]

def parse_nutrition(s):
    vals = parse_list(s)
    return dict(zip(NUTR_COLS, vals)) if len(vals) == 7 else dict.fromkeys(NUTR_COLS)

nutr_df = recipes["nutrition"].apply(parse_nutrition).apply(pd.Series)
recipes = pd.concat([recipes.drop(columns=["nutrition", "tags"]), nutr_df], axis=1)
print(f"  Nutrition columns added: {NUTR_COLS}")


# ═════════════════════════════════════════════════════════════════════════════
# 3. Load and clean interactions
# ═════════════════════════════════════════════════════════════════════════════
print("\nSTEP 3 — Load & clean interactions")

inter_raw = pd.read_csv(RAW / "RAW_interactions.csv")
print(f"  Raw interactions: {len(inter_raw):,}")

inter = inter_raw[["user_id", "recipe_id", "rating", "date"]].copy()
inter["date"] = pd.to_datetime(inter["date"])

inter = inter[inter["rating"] > 0]
print(f"  After removing zero-ratings: {len(inter):,}")

inter = inter[inter["recipe_id"].isin(set(recipes["id"]))]
print(f"  After recipe-tag filter: {len(inter):,}")


# ═════════════════════════════════════════════════════════════════════════════
# 4. Deduplicate recipes by name (Bayesian average score)
# ═════════════════════════════════════════════════════════════════════════════
print("\nSTEP 4 — Deduplicate by name")

rating_stats = (
    inter.groupby("recipe_id")["rating"]
    .agg(avg_rating="mean", n_ratings="count")
    .reset_index()
    .rename(columns={"recipe_id": "id"})
)

recipes = recipes.merge(rating_stats, on="id", how="left")
recipes["avg_rating"] = recipes["avg_rating"].fillna(0)
recipes["n_ratings"]  = recipes["n_ratings"].fillna(0).astype(int)

# Bayesian average: penalises recipes with very few ratings
# score = (n × avg) / (n + M) where M is the minimum rating threshold
M = 5
recipes["_score"] = (
    recipes["n_ratings"] * recipes["avg_rating"]
) / (recipes["n_ratings"] + M)

recipes = recipes.sort_values("_score", ascending=False)
recipes = recipes.drop_duplicates(subset="name", keep="first")
recipes = recipes.drop(columns=["avg_rating", "n_ratings", "_score"])
print(f"  After name deduplication: {len(recipes):,} recipes")

valid_ids = set(recipes["id"])
inter = inter[inter["recipe_id"].isin(valid_ids)]
print(f"  Interactions after dedup: {len(inter):,}")


# ═════════════════════════════════════════════════════════════════════════════
# 5. K-core filtering
#    Separate thresholds: recipes need more signal than users.
#    MIN_K_RECIPE = 5: a recipe rated fewer than 5 times has weak collaborative
#                      signal and would produce a noisy embedding.
#    MIN_K_USER   = 3: even 3 ratings give the GNN enough to learn from,
#                      and a higher threshold would discard too many users.
# ═════════════════════════════════════════════════════════════════════════════
print("\nSTEP 5 — K-core filtering (min 5 per recipe, min 3 per user)")

MIN_K_RECIPE = 5
MIN_K_USER   = 3

prev_size = -1
iteration = 0

while True:
    iteration += 1
    valid_recipes = set(inter["recipe_id"].value_counts()[lambda x: x >= MIN_K_RECIPE].index)
    valid_users   = set(inter["user_id"].value_counts()[lambda x: x >= MIN_K_USER].index)
    inter = inter[
        inter["recipe_id"].isin(valid_recipes) &
        inter["user_id"].isin(valid_users)
    ]
    curr_size = len(inter)
    print(f"  Iteration {iteration}: {curr_size:,} interactions")
    if curr_size == prev_size:
        break
    prev_size = curr_size

recipes = recipes[recipes["id"].isin(set(inter["recipe_id"].unique()))]
recipes = recipes.drop(columns=["tags_list"])

n_users   = inter["user_id"].nunique()
n_recipes = recipes["id"].nunique()
print(f"  Final: {len(inter):,} interactions | {n_recipes:,} recipes | {n_users:,} users")


# ═════════════════════════════════════════════════════════════════════════════
# 6. Temporal split 70 / 15 / 15
# ═════════════════════════════════════════════════════════════════════════════
print("\nSTEP 6 — Temporal split 70/15/15")

inter_sorted = inter.sort_values("date").reset_index(drop=True)
n         = len(inter_sorted)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

train = inter_sorted.iloc[:train_end].copy()
val   = inter_sorted.iloc[train_end:val_end].copy()
test  = inter_sorted.iloc[val_end:].copy()

print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
print(f"  Train period : {train['date'].min().date()} → {train['date'].max().date()}")
print(f"  Val period   : {val['date'].min().date()} → {val['date'].max().date()}")
print(f"  Test period  : {test['date'].min().date()} → {test['date'].max().date()}")


# ═════════════════════════════════════════════════════════════════════════════
# 7. Price estimation
#    Source (primary)  : Open Food Facts Prices parquet — real prices from
#                        French supermarkets (Carrefour, Intermarché, etc.)
#                        Filtered: country=FR, currency=EUR, price > 0.
#                        Price per 100g computed from price_per=KILOGRAM rows
#                        or from (price / product_quantity_g * 100).
#                        Median per ingredient category to remove outliers.
#    Source (fallback) : static €/100g table for ingredients not covered.
#    Method :
#      1. build_price_table_from_parquet() loads the parquet and returns a
#         dict {ingredient_keyword: median_price_per_100g}.
#      2. parse_weight_g() extracts grams from the ingredient text.
#      3. Keyword match (specific before generic) → price in €/100g.
#      4. ingredient cost = (€/100g) × (grams / 100); sum → meal price.
# ═════════════════════════════════════════════════════════════════════════════
print("\nSTEP 7 — Price estimation (Open Food Facts Prices + static fallback)")

_UNITS_PATTERN = "|".join(sorted(UNIT_TO_G.keys(), key=len, reverse=True))
_QTY_RE = re.compile(
    r"(\d+(?:\.\d+)?(?:\s*/\s*\d+)?)\s*(" + _UNITS_PATTERN + r")\b",
    re.IGNORECASE,
)

DEFAULT_WEIGHT_G = 150.0

def parse_weight_g(ingredient_str: str) -> float:
    """Return the weight in grams implied by the quantity + unit in the string."""
    match = _QTY_RE.search(ingredient_str)
    if not match:
        return DEFAULT_WEIGHT_G
    qty_str = match.group(1).replace(" ", "")
    if "/" in qty_str:
        num, den = qty_str.split("/")
        qty = float(num) / float(den)
    else:
        qty = float(qty_str)
    unit = match.group(2).lower()
    return qty * UNIT_TO_G[unit]


def build_price_table_from_parquet(parquet_path: Path) -> dict:
    """
    Build a {ingredient_keyword: median_price_per_100g} dict from the
    Open Food Facts Prices parquet file.

    Filtering:
      - France only (location_osm_address_country_code == 'FR')
      - EUR currency
      - price > 0 and category_tag not null

    Price per 100g computation:
      - If price_per == 'KILOGRAM': price_per_100g = price / 10
      - Else: price_per_100g = price / (product_product_quantity / 100)
        (requires a known product quantity in grams)

    Returns a dict ready to merge with the static fallback table.
    """
    if not parquet_path.exists():
        print(f"  Parquet not found at {parquet_path}, using static table only.")
        return {}

    print(f"  Loading {parquet_path.name} …")
    cols = ["category_tag", "price", "price_per", "currency",
            "price_is_discounted", "price_without_discount",
            "product_product_quantity", "product_product_quantity_unit",
            "location_osm_address_country_code"]
    df = pd.read_parquet(parquet_path, columns=cols)

    # Keep France, EUR, non-null category, positive price
    df = df[
        (df["location_osm_address_country_code"] == "FR") &
        (df["currency"] == "EUR") &
        (df["category_tag"].notna()) &
        (df["category_tag"].isin(CATEGORY_TAG_MAP))
    ].copy()

    # Use undiscounted price when available
    df["_price"] = df.apply(
        lambda r: float(r["price_without_discount"])
        if r["price_is_discounted"] and r["price_without_discount"] is not None
        else float(r["price"]),
        axis=1,
    )
    df = df[df["_price"] > 0]

    # Compute price per 100g
    def price_per_100g(row):
        if row["price_per"] == "KILOGRAM":
            return row["_price"] / 10.0
        qty = row["product_product_quantity"]
        unit = row["product_product_quantity_unit"]
        if pd.notna(qty) and qty > 0 and unit == "g":
            return row["_price"] / (qty / 100.0)
        return None

    df["_p100g"] = df.apply(price_per_100g, axis=1)
    df = df[df["_p100g"].notna() & (df["_p100g"] > 0) & (df["_p100g"] < 50)]

    # Map category tag → ingredient keyword and take median
    df["_keyword"] = df["category_tag"].map(CATEGORY_TAG_MAP)
    medians = df.groupby("_keyword")["_p100g"].median().round(3)

    result = medians.to_dict()
    print(f"  Prices loaded from parquet: {len(result)} ingredients covered")
    for kw, val in sorted(result.items()):
        print(f"    {kw:<25} €{val:.3f}/100g")
    return result


# Parquet-derived prices override the static fallback for the same keyword.
# Compound/specific entries keep their static values (not in parquet).
PRICES_PARQUET = RAW / "prices.parquet"
_parquet_prices = build_price_table_from_parquet(PRICES_PARQUET)
PRICE_TABLE = {**STATIC_FALLBACK, **_parquet_prices}

DEFAULT_PRICE = 0.28  # €/100g fallback for unrecognised ingredients


def estimate_price(ingredients_str: str) -> float:
    """
    Estimate the total cost of a recipe in euros.

    For each ingredient string:
      1. parse_weight_g() extracts the actual quantity in grams from the text.
      2. Keyword match against PRICE_TABLE returns the price in €/100g.
         Specific entries appear before generic ones — first match wins.
      3. ingredient cost = (€/100g) × (grams / 100)
    Returns the sum across all ingredients, rounded to 2 decimal places.
    """
    ingredients = parse_list(ingredients_str)
    total = 0.0
    for ing in ingredients:
        ing_lower = ing.lower()
        weight_g = parse_weight_g(ing_lower)
        price_per_100g = DEFAULT_PRICE
        for keyword, unit_price in PRICE_TABLE.items():
            if keyword in ing_lower:
                price_per_100g = unit_price
                break
        total += price_per_100g * (weight_g / 100.0)
    return round(total, 2)


recipes["estimated_price_eur"] = recipes["ingredients"].apply(estimate_price)

print(f"  Min  price: €{recipes['estimated_price_eur'].min():.2f}")
print(f"  Avg  price: €{recipes['estimated_price_eur'].mean():.2f}")
print(f"  Max  price: €{recipes['estimated_price_eur'].max():.2f}")


# ═════════════════════════════════════════════════════════════════════════════
# 8. Write to SQLite
# ═════════════════════════════════════════════════════════════════════════════
print(f"\nSTEP 8 — Write to SQLite: {DB_PATH}")

recipes["ingredients"] = recipes["ingredients"].astype(str)

with sqlite3.connect(DB_PATH) as conn:
    recipes.to_sql("recipes",          conn, if_exists="replace", index=False)
    inter.to_sql("interactions",       conn, if_exists="replace", index=False)
    train.to_sql("interactions_train", conn, if_exists="replace", index=False)
    val.to_sql("interactions_val",     conn, if_exists="replace", index=False)
    test.to_sql("interactions_test",   conn, if_exists="replace", index=False)

print("  Tables written: recipes, interactions, interactions_train/val/test")


# ═════════════════════════════════════════════════════════════════════════════
# 9. Export processed CSVs
# ═════════════════════════════════════════════════════════════════════════════
print(f"\nSTEP 9 — Export CSVs to {PROCESSED}")

recipes.to_csv(PROCESSED / "recipes.csv",    index=False)
inter.to_csv(PROCESSED / "interactions.csv", index=False)
train.to_csv(PROCESSED / "train.csv",        index=False)
val.to_csv(PROCESSED / "val.csv",            index=False)
test.to_csv(PROCESSED / "test.csv",          index=False)

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
print(f"  recipes      : {len(recipes):,} rows")
print(f"  interactions : {len(inter):,} rows  ({n_users:,} users, {n_recipes:,} recipes)")
print(f"  train/val/test: {len(train):,} / {len(val):,} / {len(test):,}")
print(f"  DB           : {DB_PATH}")
print(f"  CSVs         : {PROCESSED}")
