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

---

## Quick start (from an empty folder)

**Requirements:** Python 3.10 or newer, Git, and a free Kaggle account for the dataset.

All commands are run from the project root.

### 1. Clone the repository (once)

```bash
git clone https://github.com/Jachriso/StudEat-Recommender-System-project.git
cd StudEat-Recommender-System-project
```

### 2. Create and activate a virtual environment (once)

The virtual environment keeps the project's libraries separate from the rest of your computer.

```bash
python -m venv .venv

# Windows Activation 
.\.venv\Scripts\Activate.ps1

# MacOS / Linux Activation 
source .venv/bin/activate
```

### 3. Install the dependencies (once)

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> Always use `python -m pip` rather than `pip`: it installs into the active virtual environment.

### 4. Download the datasets (once)

The raw data files are not versioned. Put them in `data/raw/`.

**Food.com Recipes and Interactions (Kaggle)** — required: `RAW_recipes.csv` and `RAW_interactions.csv`

1. Go to [https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
2. Click **Download** (the main button — not "selected columns")
3. Extract the zip and copy the two CSV files into `data/raw/`

**Open Food Facts Prices (optional)** — `prices.parquet`. Without it, the pipeline uses a static price table.

```bash
# macOS / Linux
curl -L "https://www.data.gouv.fr/api/1/datasets/r/49716ed5-aacf-4692-8b2d-3cc6d15bf1d1" -o data/raw/prices.parquet

# Windows PowerShell (note: curl.exe, not curl)
curl.exe -L "https://www.data.gouv.fr/api/1/datasets/r/49716ed5-aacf-4692-8b2d-3cc6d15bf1d1" -o data/raw/prices.parquet
```

### 5. Run the data pipeline (once)

```bash
python src/data_pipeline.py
```

Steps: recipe filtering, temporal split 70/15/15, deduplication by name (train scores only), k-core filtering on the training split only, price estimation.
Outputs: `data/studeat.db` (SQLite, with the recipe catalogue used by the API) and `data/processed/{recipes,interactions,train,val,test}.csv`.

### 6. Run the app

The trained model is provided in the repository (`models/embeddings.npz`), so **no training is needed** to run the app.

Open **two terminals** and activate the virtual environment in each of them.

```bash
# Terminal 1 — API: http://localhost:8000/docs
python -m uvicorn api.main:app --reload

# Terminal 2 — interface: http://localhost:8501
python -m streamlit run app/app.py
```

---

## Retrain the model (optional)

Not needed to run the app. Requires steps 1 to 5. Training runs on CPU (about 30–50 s per epoch for NLGCL), with early stopping on validation NDCG@10.

```bash
# Quick check that the model runs (synthetic data, a few seconds)
python src/test_forward.py

# Train NLGCL (hyperparameters in configs/nlgcl.json)
python src/train.py --config configs/nlgcl.json

# Ablation: same model without the contrastive loss (= LightGCN)
python src/train.py --config configs/lightgcn.json

# Loss curves from the training history
python src/plot_history.py models/nlgcl_history.csv

# Baselines (Random, Most-Popular) with the shared evaluation harness
python src/run_baselines.py

# Final benchmark: every model over 3 seeds + baselines, with the shared harness
# (writes reports/results_final.md; about 30 to 60 min on CPU, with early stopping)
python src/run_benchmark.py
# Optional training options (off by default; they did not help in our tuning)
#   --mask-dup     mask duplicate ids as false negatives in the InfoNCE loss
#   --neg-alpha A  sample negatives proportionally to popularity^A (0 = uniform)
python src/train.py --config configs/nlgcl.json --lambda-cl 0.01 --neg-alpha 0.5 --mask-dup

# Blend of the model score with item popularity (weight w chosen on validation, test read once)
python src/eval_blend.py --ckpt models/nlgcl_s0.pt models/nlgcl_s1.pt models/nlgcl_s2.pt

# Cosine + popularity score on fold-in vectors (API); w is chosen on validation (best NDCG@10, or a product choice read on the validation table), test is read once
python src/eval_cosine.py --ckpt models/benchmark/nlgcl_s0.pt models/benchmark/nlgcl_s1.pt models/benchmark/nlgcl_s2.pt

# Results by user-activity level
python src/eval_by_activity.py --ckpt models/nlgcl.pt

# Export the embeddings used by the API (replaces models/embeddings.npz)
python src/export_embeddings.py --ckpt models/nlgcl.pt
```

Each run writes to `models/`: the checkpoint (`.pt`), the per-epoch history (`_history.csv`) and the results (`_results.json`). These files are not versioned; only `models/embeddings.npz` is. Command-line options override the config file, e.g. `--epochs 10`.

---

## Project structure

```
api/
  main.py               FastAPI backend (endpoints, hard filters, placeholder predict())
  mock_data.py          Mock meal catalogue (fallback when the pipeline has not been run)
app/
  app.py                Streamlit interface (sign-up, weekly box, meal rating)
configs/
  nlgcl.json            Hyperparameters of NLGCL
  lightgcn.json         Same configuration without the contrastive loss
data/
  raw/                  Raw datasets (not versioned, see "Datasets")
  processed/            CSV outputs of the data pipeline (not versioned)
  studeat.db            SQLite database (not versioned)
models/
  embeddings.npz        Trained embeddings used by the API (versioned)
                        Checkpoints and histories (not versioned)
reports/figures/        Figures used in the reports
src/
  data_pipeline.py      Data pipeline
  pipeline_config.py    Static configuration of the pipeline
  model_nlgcl.py        NLGCL / LightGCN model
  train.py              Training and evaluation (NDCG@10, Recall@20)
  test_forward.py       Quick model check on synthetic data
  plot_history.py       Training curves
  evaluation.py         Shared evaluation harness (NDCG@10, Recall@20) used for all models
  run_baselines.py      Random and Most-Popular baselines with the shared harness
  run_benchmark.py      Final benchmark: all models over 3 seeds, one results table
  baselines.py          Random and Most-Popular baselines (MLE-Ops version)
  eval_popularity.py    Popularity baseline (MLE-Core version)
  eval_by_activity.py   Results by user-activity level
  export_embeddings.py  Export of the trained embeddings
  recommender.py        Recommender used by the API (popularity fallback for new students)
  seed_test_data.py     Synthetic test students and ratings
requirements.txt
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

## Test data (end-to-end testing)

To test the full chain without a real rating campaign, insert synthetic students and ratings:

```bash
# Insert 8 test students (all diets/allergens covered) and 69 ratings
python src/seed_test_data.py

# Then call GET http://localhost:8000/recommend/{user_id}
# with the user_id printed by the script (e.g. TEST_Alice: vegetarian, gluten-free)

# Remove all test data when done
python src/seed_test_data.py --clean
```

Test students are prefixed with `TEST_`. These ratings are synthetic: they must not be used for training or reported as real results.

## Model interface (agreed with MLE-Core / MLE-Ops)

```python
predict(user_id, budget_min, budget_max, allergies, diet, k=10) -> list[dict]  # each meal with a "score"
```

Hard filters (allergies, diet) and budget checks are applied by rules in the API; the model only ranks the remaining meals.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Windows: `.venv\Scripts\Activate.ps1` "cannot load module" | Add `.\` in front: `.\.venv\Scripts\Activate.ps1` |
| Windows: "running scripts is disabled on this system" | Run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Windows: `pip.exe` blocked by an application control policy | Use `python -m pip ...` instead of `pip ...` |
| `pip` tries to compile numpy and fails ("Unknown compiler") | Your Python version is not supported by a dependency: use Python 3.12 (`py -3.12 -m venv .venv` on Windows) |
| Pipeline: `pyarrow` DLL blocked (Windows Smart App Control) | `python -m pip install fastparquet`, or remove `data/raw/prices.parquet` to use the static prices |
| Streamlit: "API not reachable" | Start the API first (Terminal 1), with the virtual environment activated |
| The app shows only 30 meals | The pipeline has not been run: the API falls back to the mock catalogue (step 5) |