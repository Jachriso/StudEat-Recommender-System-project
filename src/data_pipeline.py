"""
StudEat — Data Engineering Pipeline
"""

import ast
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# Paths 
ROOT      = Path(__file__).resolve().parent.parent
RAW       = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
DB_PATH   = ROOT / "data" / "studeat.db"

PROCESSED.mkdir(parents=True, exist_ok=True)

# Helpers 
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

M = 5  # Bayesian prior (minimum rating count)
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
# 5. K-core filtering (min 5 interactions per user AND per recipe)
# ═════════════════════════════════════════════════════════════════════════════
print("\nSTEP 5 — K-core filtering (min 5)")

MIN_K = 5
prev_size = -1
iteration = 0

while True:
    iteration += 1
    valid_recipes = set(inter["recipe_id"].value_counts()[lambda x: x >= MIN_K].index)
    valid_users   = set(inter["user_id"].value_counts()[lambda x: x >= MIN_K].index)
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
#    Source : Carrefour France 2024 shelf prices (fresh produce + dry goods)
#    Unit   : €/100g for all entries in PRICE_TABLE
#    Method :
#      1. Parse each ingredient string to extract quantity + unit → grams
#         (e.g. "2 cups flour" → 2 × 240g = 480g)
#      2. Keyword-match the ingredient name against PRICE_TABLE to get €/100g
#         Specific entries (e.g. "cream of chicken soup") are listed BEFORE
#         their generic parent ("chicken") so the first match wins and
#         avoids misclassification.
#      3. price_ingredient = (€/100g) × (grams / 100)
#      4. Sum over all ingredients → estimated meal price in €
# ═════════════════════════════════════════════════════════════════════════════
print("\nSTEP 7 — Price estimation (Carrefour France reference)")

# ── Unit → grams conversion table ────────────────────────────────────────────
# Covers the most common units found in Food.com ingredient strings.
# Liquid volumes are approximated as water-equivalent (1 ml ≈ 1 g) for
# simplicity; this is acceptable for broths, milk, oils at this precision level.
UNIT_TO_G = {
    "kg": 1000, "g": 1,
    "lb": 453,  "lbs": 453, "pound": 453, "pounds": 453,
    "oz": 28,   "ounce": 28, "ounces": 28,
    "cup": 240, "cups": 240,
    "tbsp": 15, "tablespoon": 15, "tablespoons": 15,
    "tsp": 5,   "teaspoon": 5,   "teaspoons": 5,
    "l": 1000,  "liter": 1000,   "liters": 1000,
    "ml": 1,    "milliliter": 1, "milliliters": 1,
    "can": 400, "cans": 400,     # standard 400g can
    "clove": 5, "cloves": 5,     # one garlic clove ≈ 5g
    "slice": 30,"slices": 30,
    "piece": 100,"pieces": 100,
    "bunch": 150,"bunches": 150,
    "stalk": 40, "stalks": 40,
    "sprig": 5,  "sprigs": 5,
}

# Regex: captures an optional quantity (integer, decimal, or fraction like 1/2)
# followed by an optional unit from UNIT_TO_G.
_UNITS_PATTERN = '|'.join(sorted(UNIT_TO_G.keys(), key=len, reverse=True))
_QTY_RE = re.compile(
    r'(\d+(?:\.\d+)?(?:\s*/\s*\d+)?)\s*(' + _UNITS_PATTERN + r')\b',
    re.IGNORECASE
)

DEFAULT_WEIGHT_G = 150.0  # fallback when no recognisable unit is found

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


# ── Price table: €/100g, Carrefour France 2024 ───────────────────────────────
# IMPORTANT: more specific entries must come BEFORE their generic parent so
# that the first-match rule assigns the correct price.
# e.g. "cream of chicken soup" must appear before "chicken", otherwise
#      "chicken" matches first and inflates the price.
PRICE_TABLE = {
    # -- Specific compound/processed products (listed first) -----------------
    "cream of chicken soup":  0.18,   # canned condensed soup ~€1.80/kg
    "cream of mushroom soup": 0.18,
    "chicken broth":          0.14,   # ~€1.40/kg
    "chicken stock":          0.14,
    "beef broth":             0.14,
    "beef stock":             0.14,
    "vegetable broth":        0.12,
    "vegetable stock":        0.12,
    "tomato sauce":           0.22,   # ~€2.20/kg
    "tomato paste":           0.28,
    "canned tomato":          0.14,
    "diced tomato":           0.14,
    "crushed tomato":         0.14,
    "coconut milk":           0.55,
    "soy sauce":              0.38,
    "olive oil":              0.90,   # virgin olive oil ~€9.00/L
    "vegetable oil":          0.18,
    "ground beef":            1.20,   # 5% MG ~€12.00/kg
    "ground pork":            0.85,
    "ground turkey":          1.00,
    "ground chicken":         0.85,
    "sweet potato":           0.18,
    "green bean":             0.30,
    "sour cream":             0.65,
    "heavy cream":            0.80,
    "whipping cream":         0.80,
    "worcestershire":         0.50,
    "hot sauce":              0.45,
    # -- Generic proteins (after compound entries) ----------------------------
    "chicken":                0.79,   # chicken fillet ~€7.90/kg
    "beef":                   1.49,   # minced beef ~€14.90/kg
    "pork":                   0.89,   # pork fillet ~€8.90/kg
    "bacon":                  1.10,   # smoked lardons ~€11.00/kg
    "ham":                    1.30,   # cooked ham ~€13.00/kg
    "sausage":                0.75,
    "salmon":                 2.20,   # atlantic salmon ~€22.00/kg
    "tuna":                   1.00,   # canned tuna ~€10.00/kg
    "shrimp":                 2.50,   # peeled shrimp ~€25.00/kg
    "eggs":                   0.22,   # one egg ≈ 60g at ~€1.30/6 eggs
    "tofu":                   0.55,
    "lamb":                   1.80,   # leg of lamb ~€18.00/kg
    "turkey":                 1.00,
    # -- Starches & carbohydrates --------------------------------------------
    "pasta":                  0.13,   # dry pasta ~€1.30/kg
    "rice":                   0.11,   # long-grain rice ~€1.10/kg
    "potato":                 0.09,   # potatoes ~€0.90/kg
    "bread":                  0.19,
    "flour":                  0.08,   # T45 flour ~€0.80/kg
    "oats":                   0.09,
    "noodles":                0.14,
    "lentil":                 0.19,   # green lentils ~€1.90/kg
    "bean":                   0.13,   # dried beans ~€1.30/kg
    "chickpea":               0.15,
    "couscous":               0.12,
    "quinoa":                 0.55,
    # -- Dairy ---------------------------------------------------------------
    "mozzarella":             1.50,
    "parmesan":               2.80,
    "cheese":                 1.40,   # grated emmental ~€14.00/kg
    "butter":                 1.10,   # unsalted butter ~€11.00/kg
    "milk":                   0.09,   # semi-skimmed ~€0.90/L
    "cream":                  0.70,   # crème fraîche ~€7.00/kg
    "yogurt":                 0.35,
    # -- Fresh vegetables ----------------------------------------------------
    "onion":                  0.07,   # yellow onions ~€0.70/kg
    "garlic":                 0.28,   # garlic ~€2.80/kg
    "tomato":                 0.18,   # round tomatoes ~€1.80/kg
    "carrot":                 0.09,
    "spinach":                0.55,
    "broccoli":               0.35,
    "mushroom":               0.90,   # button mushrooms ~€9.00/kg
    "pepper":                 0.28,   # bell pepper ~€2.80/kg
    "zucchini":               0.22,
    "eggplant":               0.28,
    "corn":                   0.18,
    "peas":                   0.22,
    "celery":                 0.14,
    "cabbage":                0.11,
    "leek":                   0.20,
    "cucumber":               0.15,
    "lettuce":                0.18,
    "asparagus":              0.80,
    # -- Pantry / condiments -------------------------------------------------
    "oil":                    0.20,   # generic oil fallback
    "sugar":                  0.07,
    "salt":                   0.02,
    "broth":                  0.14,
    "stock":                  0.14,
    "lemon":                  0.18,
    "lime":                   0.22,
    "vinegar":                0.12,
    "honey":                  0.75,
    "mustard":                0.30,
    "mayonnaise":             0.38,   # €1.49/395g ≈ €0.38/100g (Carrefour)
    "ketchup":                0.25,
}

DEFAULT_PRICE = 0.28  # €/100g — fallback for unrecognised ingredients


def estimate_price(ingredients_str: str) -> float:
    """
    Estimate the total cost of a recipe in euros.

    For each ingredient string:
      1. parse_weight_g() extracts the actual quantity in grams from the text
         (e.g. '2 cups flour' → 480g).  Falls back to DEFAULT_WEIGHT_G.
      2. keyword match against PRICE_TABLE returns the price in €/100g.
         Specific entries appear before generic ones in the dict, so the
         first match wins (Python 3.7+ dicts preserve insertion order).
      3. ingredient cost = (€/100g) × (grams / 100)
    The function returns the sum across all ingredients, rounded to 2 dp.
    """
    ingredients = parse_list(ingredients_str)
    total = 0.0
    for ing in ingredients:
        ing_lower = ing.lower()

        # Determine weight from the quantity + unit written in the string
        weight_g = parse_weight_g(ing_lower)

        # First keyword match wins; more specific entries are listed first
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
    recipes.to_sql("recipes",            conn, if_exists="replace", index=False)
    inter.to_sql("interactions",         conn, if_exists="replace", index=False)
    train.to_sql("interactions_train",   conn, if_exists="replace", index=False)
    val.to_sql("interactions_val",       conn, if_exists="replace", index=False)
    test.to_sql("interactions_test",     conn, if_exists="replace", index=False)

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
