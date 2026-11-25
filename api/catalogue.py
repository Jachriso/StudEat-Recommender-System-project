"""Meal catalogue of the API, built from the Food.com recipes of the data pipeline.

The `recipes` table of data/studeat.db (produced by src/data_pipeline.py) has
no diet or allergen information and gives the ingredient cost of the whole
recipe. This module turns each recipe into a meal of our box:

- diets and allergens are detected from the ingredient names with keyword rules
  (an approximation, documented as such in the report);
- the meal price = ingredient cost / SERVINGS + SERVICE_FEE (preparation and
  delivery of a ready-made meal).

If the database has no `recipes` table (data pipeline not run yet), the API
falls back to the small mock catalogue of api/mock_data.py.
"""

import ast
import re
import sqlite3
from pathlib import Path

from api import mock_data

# --- Business assumptions (to validate with the team) ----------------------
SERVINGS = 4          # Food.com recipes are assumed to serve 4 people
SERVICE_FEE = 2.50    # EUR per meal: preparation, packaging and delivery
MIN_CALORIES = 250    # kcal per serving: below this, a recipe is not a full meal

ALLERGENS = mock_data.ALLERGENS
DIETS = mock_data.DIETS


def _pattern(words: list[str]) -> re.Pattern:
    """Regex matching any of the words as whole words (so 'egg' does not match 'eggplant')."""
    return re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")s?\b")


MEAT = _pattern([
    "chicken", "beef", "pork", "bacon", "ham", "sausage", "turkey", "lamb", "veal", "steak",
    "meat", "pepperoni", "salami", "prosciutto", "chorizo", "duck", "venison", "lard",
    "hamburger", "meatball", "hot dog", "rib", "brisket", "gelatin", "pancetta", "kielbasa",
    "goose", "rabbit", "bison", "quail", "pheasant", "liver", "gizzard", "oxtail", "pastrami",
    "bologna", "spam", "jerky", "mutton", "sirloin", "tenderloin", "chuck roast", "giblet",
    "roast", "rump", "loin", "pork chop", "cutlet", "drumstick", "chicken wing", "ground round",
])
PORK = _pattern([
    "pork", "bacon", "ham", "sausage", "pepperoni", "salami", "prosciutto", "chorizo",
    "lard", "pancetta", "kielbasa", "hot dog",
])
ALCOHOL = _pattern([
    "wine", "beer", "rum", "vodka", "whiskey", "whisky", "bourbon", "brandy", "sherry",
    "sake", "mirin", "liqueur", "tequila", "gin", "marsala", "cognac", "ale",
])
FISH = _pattern([
    "fish", "salmon", "tuna", "cod", "tilapia", "anchovy", "anchovie", "sardine", "halibut",
    "trout", "catfish", "mahi mahi", "haddock", "mackerel", "fish sauce","snapper", "bass", 
    "swordfish", "flounder", "sole", "perch", "pollock", "grouper", "herring",
    "whitefish", "monkfish", "orange roughy", "mahi", "walleye", "pike", "carp", "turbot",
    "branzino", "caviar", "roe", "seafood", "eel", "worcestershire sauce",
])
SHELLFISH = _pattern([
    "shrimp", "prawn", "crab", "lobster", "scallop", "clam", "mussel", "oyster",
    "squid", "calamari", "crawfish",
])
NOT_DAIRY = re.compile(r"\b(coconut|almond|soy|rice|oat|peanut|cashew) (milk|butter|cream)\b|cream of tartar")
DAIRY = _pattern([
    "milk", "cheese", "butter", "cream", "yogurt", "yoghurt", "parmesan", "mozzarella",
    "cheddar", "ricotta", "feta", "ghee", "buttermilk", "whey", "half-and-half",
])
EGGS = _pattern(["egg", "egg white", "egg yolk", "mayonnaise"])
GLUTEN_FREE_FLOURS = re.compile(r"\b(rice|corn|almond|coconut|potato|chickpea) (flour|noodles?|tortillas?)\b|cornflour|cornstarch")
GLUTEN = _pattern([
    "flour", "bread", "breadcrumb", "bread crumb", "pasta", "spaghetti", "noodle", "macaroni",
    "couscous", "wheat", "barley", "rye", "tortilla", "biscuit", "cracker", "soy sauce", "beer",
    "pie crust", "pastry", "crouton", "bulgur", "semolina", "lasagna", "penne", "fettuccine",
    "linguine", "orzo", "ravioli", "tortellini", "pita", "bun", "panko", "teriyaki sauce",
    "cookie", "cake", "crust", "dough", "wafer", "graham", "brownie", "muffin", "pancake",
    "waffle", "pizza", "bagel", "croissant", "malt", "seitan", "stuffing", "cornbread",
    "bisquick", "gravy", "oat", "oatmeal", "cream of mushroom soup", "cream of chicken soup",
    "roll", "dumpling", "wonton", "phyllo", "puff pastry","baguette", "ciabatta", "loaf", 
    "focaccia", "sourdough", "naan", "brioche", "challah",
    "matzo", "farro", "spelt", "ramen", "udon", "gnocchi", "pierogi", "empanada", "flatbread",
    "shortbread", "gingerbread", "pretzel", "crepe", "lasagne", "vermicelli", "rigatoni", "ziti",
    "farfalle", "rotini", "fusilli", "tart", "pie shell", "batter", "breading",
    "sandwich", "sandwiche", "wrap", "pie", "quiche", "calzone", "stromboli", "hoagie",
    "burrito", "quesadilla", "burger",
])
NUTS = _pattern([
    "almond", "walnut", "pecan", "cashew", "pistachio", "hazelnut", "macadamia", "pine nut",
])
PEANUTS = _pattern(["peanut"])
SOY = _pattern(["soy", "soy sauce", "tofu", "edamame", "miso", "tempeh", "tamari", "teriyaki sauce"])
SESAME = _pattern(["sesame", "tahini"])
HONEY = _pattern(["honey"])


def detect_allergens(text: str) -> list[str]:
    """Allergens found in the ingredient names (lowercase text)."""
    found = []
    if GLUTEN.search(GLUTEN_FREE_FLOURS.sub("", text)):
        found.append("gluten")
    if DAIRY.search(NOT_DAIRY.sub("", text)):
        found.append("lactose")
    for name, pattern in [("eggs", EGGS), ("nuts", NUTS), ("peanuts", PEANUTS), ("fish", FISH),
                          ("shellfish", SHELLFISH), ("soy", SOY), ("sesame", SESAME)]:
        if pattern.search(text):
            found.append(name)
    return found


def detect_diets(text: str, allergens: list[str]) -> list[str]:
    """Diets the meal is compatible with ('none' = no restriction)."""
    has_meat = bool(MEAT.search(text))
    has_seafood = "fish" in allergens or "shellfish" in allergens
    diets = ["none"]
    if not has_meat and not has_seafood:
        diets.append("vegetarian")
        if "lactose" not in allergens and "eggs" not in allergens and not HONEY.search(text):
            diets.append("vegan")
    if not has_meat:
        diets.append("pescatarian")
    # Approximation: no pork and no alcohol (the slaughter method cannot be checked).
    if not PORK.search(text) and not ALCOHOL.search(text):
        diets.append("halal")
    return diets


def recipe_to_meal(row: sqlite3.Row) -> dict:
    ingredients = ast.literal_eval(row["ingredients"])
    text = (" | ".join(ingredients) + " | " + row["name"]).lower()
    allergens = detect_allergens(text)
    return {
        "id": row["id"],
        "name": " ".join(row["name"].split()).capitalize(),  
        "price": round(row["estimated_price_eur"] / SERVINGS + SERVICE_FEE, 2),
        "calories": round(row["calories"]) if row["calories"] is not None else None,
        "diets": detect_diets(text, allergens),
        "allergens": allergens,
    }


def load_meals(db_path: Path) -> tuple[list[dict], str]:
    """Return (meals, source). Falls back to the mock catalogue if there is no recipes table."""
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT id, name, ingredients, calories, estimated_price_eur FROM recipes "
                "WHERE calories >= ?",
                (MIN_CALORIES,),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        finally:
            conn.close()
        if rows:
            return [recipe_to_meal(r) for r in rows], "food.com"
    return mock_data.MEALS, "mock"