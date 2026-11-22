# StudEat – Recommender System Project

Personalized weekly ready-meal box for students within a chosen budget range.

## Team

| Role | Member |
|---|---|
| Project Lead | Jade PARRILLA |
| Data Engineer | Suzy Cassandra EBOBISSE |
| Lead ML Engineer | Hajar BENLEMQADEM |
| ML Engineer | Annabelle FAUTRAD |
| Systems Engineer | Marie NOMO NGAH |

## Project structure

```
api/
  main.py             FastAPI backend (endpoints, hard filters, placeholder predict())
  mock_data.py        Mock meal catalogue (to be replaced by the Food.com recipes)
app/
  app.py              Streamlit interface (sign-up, weekly box, meal rating)
data/
  raw/                Raw datasets (not versioned, see "Datasets")
  processed/          CSV outputs of the data pipeline
  studeat.db          SQLite database (not versioned)
src/
  data_pipeline.py    Data pipeline: filtering, k-core, temporal split, price estimation
  pipeline_config.py  Static configuration of the pipeline
requirements.txt
```

## Datasets

The raw data files are **not versioned** (too large for git). Download them manually and place them in `data/raw/` before running the pipeline.

### 1 — Food.com Recipes and Interactions (Kaggle)

Required files: `RAW_recipes.csv` and `RAW_interactions.csv`

1. Go to [https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
2. Click **Download** (the main button — not "selected columns")
3. Extract the zip and copy the two CSV files into `data/raw/`

### 2 — Open Food Facts Prices (data.gouv.fr)

Required file: `prices.parquet`

```bash
curl -L "https://www.data.gouv.fr/api/1/datasets/r/49716ed5-aacf-4692-8b2d-3cc6d15bf1d1" -o data/raw/prices.parquet
```

Or download manually from the URL above and save as `data/raw/prices.parquet`.

### 3 — Run the pipeline

```bash
pip install pandas numpy pyarrow
python src/data_pipeline.py
```

Output: `data/studeat.db` (SQLite) and CSVs in `data/processed/`.

---

## Run locally

From the project root:

```bash
python -m pip install -r requirements.txt

# Terminal 1: API on http://localhost:8000 (interactive docs: http://localhost:8000/docs)
python -m uvicorn api.main:app --reload

# Terminal 2: interface on http://localhost:8501
python -m streamlit run app/app.py
```

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | API status and catalogue size |
| GET | `/options` | Allowed diets and allergens |
| POST | `/profile` | Create a student profile, returns `user_id` |
| GET | `/profile/{user_id}` | Read a profile |
| GET | `/recommend/{user_id}` | Weekly box: meals, total price, budget check |
| POST | `/log_interaction` | Save a meal rating (1 to 5 stars) with a timestamp |
| GET | `/interactions/{user_id}` | Ratings given by a student |

## App data (SQLite)

The API stores student data in two tables of `data/studeat.db`, created automatically at startup:

| Table | Content |
|---|---|
| `app_users` | Student profiles (diet, allergies, budget range, meals per week) |
| `app_ratings` | Meal ratings given in the app (user_id, meal_id, rating, timestamp) |

These tables are kept separate from the Food.com tables (`recipes`, `interactions`, `interactions_train`, ...), so that app data never mixes with the training data and is not erased when the data pipeline is run again.

## Model interface (agreed with MLE-Core / MLE-Ops)

```python
predict(user_id, budget_min, budget_max, allergies, diet, k=10) -> list[dict]  # each meal with a "score"
```

Hard filters (allergies, diet) and budget checks are applied by rules in the API; the model only ranks the remaining meals.