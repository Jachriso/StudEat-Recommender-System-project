# StudEat – Recommender System Project

Personalized weekly ready-meal box for students within a chosen budget range.

> **Status (week 2):** the API returns **mock recommendations** (random scores on a small hand-made catalogue).
> The real NLGCL model and the Food.com catalogue will be plugged in during week 3 through the `predict()` function.

## Team

| Role | Member |
|---|---|
| Project Lead | Jade PARILLA |
| Data Engineer | Suzy Cassandra EBOBISSE |
| Lead ML Engineer | Hajar BENLEMQADEM |
| ML Engineer | Annabelle FAUTRAD |
| Systems Engineer | Marie NOMO NGAH |

## Project structure

```
api/
  main.py        FastAPI backend (endpoints + placeholder predict())
  mock_data.py   Mock meal catalogue (replaced by Food.com in week 3)
app/
  app.py         Streamlit interface (sign-up + weekly box)
requirements.txt
```

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

## Model interface (agreed with MLE-Core / MLE-Ops)

```python
predict(user_id, budget_min, budget_max, allergies, diet, k=10) -> list[dict]  # each meal with a "score"
```

Hard filters (allergies, diet) and budget checks are applied by rules in the API; the model only ranks the remaining meals.