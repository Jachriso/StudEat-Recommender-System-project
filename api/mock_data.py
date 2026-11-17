"""Mock meal catalogue used until the cleaned Food.com dataset is ready.

Each meal has an estimated price (EUR), the diets it is compatible with,
and the allergens it contains. In week 3 this list is replaced by the
cleaned Food.com recipes produced by the data pipeline.
"""

ALLERGENS = ["gluten", "lactose", "eggs", "nuts", "peanuts", "fish", "shellfish", "soy", "sesame"]
DIETS = ["none", "vegetarian", "vegan", "halal", "pescatarian"]

MEALS = [
    {"id": 1, "name": "Chili con carne with rice", "price": 4.20, "cuisine": "Mexican", "diets": ["none", "halal"], "allergens": []},
    {"id": 2, "name": "Vegetable curry with basmati rice", "price": 3.60, "cuisine": "Indian", "diets": ["none", "vegetarian", "vegan", "halal", "pescatarian"], "allergens": []},
    {"id": 3, "name": "Spaghetti bolognese", "price": 3.90, "cuisine": "Italian", "diets": ["none", "halal"], "allergens": ["gluten"]},
    {"id": 4, "name": "Chicken tikka masala", "price": 4.80, "cuisine": "Indian", "diets": ["none", "halal"], "allergens": ["lactose"]},
    {"id": 5, "name": "Lentil dahl with naan", "price": 3.20, "cuisine": "Indian", "diets": ["none", "vegetarian", "halal", "pescatarian"], "allergens": ["gluten", "lactose"]},
    {"id": 6, "name": "Salmon teriyaki bowl", "price": 5.90, "cuisine": "Japanese", "diets": ["none", "pescatarian"], "allergens": ["fish", "soy", "sesame"]},
    {"id": 7, "name": "Lasagna", "price": 4.50, "cuisine": "Italian", "diets": ["none"], "allergens": ["gluten", "lactose", "eggs"]},
    {"id": 8, "name": "Vegetable lasagna", "price": 4.30, "cuisine": "Italian", "diets": ["none", "vegetarian", "halal", "pescatarian"], "allergens": ["gluten", "lactose", "eggs"]},
    {"id": 9, "name": "Chicken couscous", "price": 4.60, "cuisine": "Moroccan", "diets": ["none", "halal"], "allergens": ["gluten"]},
    {"id": 10, "name": "Vegetable couscous", "price": 3.70, "cuisine": "Moroccan", "diets": ["none", "vegetarian", "vegan", "halal", "pescatarian"], "allergens": ["gluten"]},
    {"id": 11, "name": "Beef stir-fry with noodles", "price": 5.20, "cuisine": "Chinese", "diets": ["none", "halal"], "allergens": ["gluten", "soy", "sesame"]},
    {"id": 12, "name": "Tofu stir-fry with noodles", "price": 4.10, "cuisine": "Chinese", "diets": ["none", "vegetarian", "vegan", "halal", "pescatarian"], "allergens": ["gluten", "soy", "sesame"]},
    {"id": 13, "name": "Chicken fajitas", "price": 4.90, "cuisine": "Mexican", "diets": ["none", "halal"], "allergens": ["gluten"]},
    {"id": 14, "name": "Black bean burrito bowl", "price": 3.80, "cuisine": "Mexican", "diets": ["none", "vegetarian", "vegan", "halal", "pescatarian"], "allergens": []},
    {"id": 15, "name": "Shepherd's pie", "price": 4.70, "cuisine": "British", "diets": ["none"], "allergens": ["lactose"]},
    {"id": 16, "name": "Mushroom risotto", "price": 4.40, "cuisine": "Italian", "diets": ["none", "vegetarian", "halal", "pescatarian"], "allergens": ["lactose"]},
    {"id": 17, "name": "Tuna pasta bake", "price": 3.90, "cuisine": "Italian", "diets": ["none", "pescatarian"], "allergens": ["gluten", "lactose", "fish"]},
    {"id": 18, "name": "Chickpea and spinach stew", "price": 3.30, "cuisine": "Mediterranean", "diets": ["none", "vegetarian", "vegan", "halal", "pescatarian"], "allergens": []},
    {"id": 19, "name": "Beef bourguignon with potatoes", "price": 5.80, "cuisine": "French", "diets": ["none"], "allergens": []},
    {"id": 20, "name": "Ratatouille with quinoa", "price": 3.90, "cuisine": "French", "diets": ["none", "vegetarian", "vegan", "halal", "pescatarian"], "allergens": []},
    {"id": 21, "name": "Chicken satay with rice", "price": 4.90, "cuisine": "Thai", "diets": ["none", "halal"], "allergens": ["peanuts", "soy"]},
    {"id": 22, "name": "Thai green curry with shrimp", "price": 5.60, "cuisine": "Thai", "diets": ["none", "pescatarian"], "allergens": ["shellfish", "fish"]},
    {"id": 23, "name": "Pad thai with tofu", "price": 4.30, "cuisine": "Thai", "diets": ["none", "vegetarian", "halal", "pescatarian"], "allergens": ["peanuts", "soy", "eggs"]},
    {"id": 24, "name": "Lamb tagine with apricots", "price": 5.90, "cuisine": "Moroccan", "diets": ["none", "halal"], "allergens": ["nuts"]},
    {"id": 25, "name": "Quiche lorraine with salad", "price": 3.80, "cuisine": "French", "diets": ["none"], "allergens": ["gluten", "lactose", "eggs"]},
    {"id": 26, "name": "Cod with mashed potatoes", "price": 5.40, "cuisine": "French", "diets": ["none", "pescatarian"], "allergens": ["fish", "lactose"]},
    {"id": 27, "name": "Bibimbap with beef", "price": 5.30, "cuisine": "Korean", "diets": ["none", "halal"], "allergens": ["soy", "sesame", "eggs"]},
    {"id": 28, "name": "Falafel bowl with hummus", "price": 4.00, "cuisine": "Middle Eastern", "diets": ["none", "vegetarian", "vegan", "halal", "pescatarian"], "allergens": ["sesame"]},
    {"id": 29, "name": "Chicken and vegetable soup", "price": 3.10, "cuisine": "French", "diets": ["none", "halal"], "allergens": []},
    {"id": 30, "name": "Mac and cheese", "price": 3.40, "cuisine": "American", "diets": ["none", "vegetarian", "halal", "pescatarian"], "allergens": ["gluten", "lactose"]},
]