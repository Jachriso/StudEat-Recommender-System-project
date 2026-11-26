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

**Open Food Facts Prices** — required: `prices.parquet`. Used as the primary price source; the static fallback only covers ingredients not found in this file.

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
# Terminal 1 — API: http://127.0.0.1:8000/docs
python -m uvicorn api.main:app --reload

# Terminal 2 — interface: http://127.0.0.1:8501
python -m streamlit run app/app.py
```

---

## Results

Final benchmark on the temporal test split (1,434 students), written by `src/run_benchmark.py` to
[`reports/results_final.md`](reports/results_final.md). Every row uses the shared harness `src/evaluation.py`:
full-catalogue ranking, meals already rated in training masked, positives = ratings ≥ 4.

| Model | Val NDCG@10 | Test NDCG@10 | Test Recall@20 | Unwanted-meal rate |
|---|---|---|---|---|
| Random (5 seeds) | 0.0010 ± 0.0004 | 0.0009 ± 0.0005 | 0.0027 ± 0.0010 | n/a |
| Most-Popular | 0.0375 | 0.0350 | 0.0838 | n/a |
| LightGCN (3 seeds) | 0.0379 ± 0.0002 | 0.0346 ± 0.0004 | 0.0869 ± 0.0017 | n/a |
| NLGCL (3 seeds) | 0.0401 ± 0.0004 | 0.0326 ± 0.0011 | 0.0803 ± 0.0046 | n/a |
| NLGCL fold-in (app score), no risk control | 0.0364 | 0.0321 | 0.0751 | 2.18 % |
| NLGCL fold-in (app score) + risk control (α = 0.015) | 0.0364 | 0.0321 | 0.0751 | 1.95 % |

± = sample standard deviation over seeds. "Unwanted" = a meal in the top 10 that the student rated 2 or less.

**How to read it**

- **Model choice.** NLGCL is the best model on validation (0.0401 vs 0.0379 for LightGCN), so it is the model
  used by the app. The test split is read once and reported as it is.
- **Popularity is hard to beat.** On the test period, LightGCN and Most-Popular are within noise of each other
  and NLGCL is slightly lower. The data is very sparse (median of 5 ratings per student, temporal split),
  so popularity is a strong signal.
- **App score.** The app ranks meals with 0.7 × cosine + 0.3 × popularity on fold-in vectors
  (`src/scoring.py`). It keeps about 90 % of the validation NDCG@10 of the learned vectors, and the box
  changes after the student's ratings.
- **Risk control.** The threshold is calibrated on validation with the same fold-in and score as the app
  (`configs/risk_threshold.json`, α = 0.015). On test, unwanted meals drop from 2.18 % to 1.95 % with no loss
  of accuracy. The test rate stays above α because the base risk is higher on the test period (2.18 % vs
  1.80 % on validation): the guarantee assumes data like the calibration set, and our split is temporal.
  NDCG@10 and Recall@20 are the same with and without risk control: on the full-catalogue ranking, the
  top 10 always scores above the threshold (t = 0.20), so no slot is replaced. The unwanted rate is
  measured like the calibration, on the meals each test student actually rated (12.7 % of them replaced).
- **Latency** of `/recommend` is measured by `scripts/measure_latency.py` (see `reports/latency.md`).

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

# Risk control : conformal calibration of the replacement threshold on validation
# (fold-in vectors + app score, alpha = 0.015), writes configs/risk_threshold.json (versioned)
python src/risk_control.py --foldin

# Final benchmark: every model over 3 seeds + baselines + the app model with and without
# risk control, all with the shared harness (writes reports/results_final.md;
# about 30 to 60 min on CPU, with early stopping). Checkpoints are not versioned,
# so the first run must train.
python src/run_benchmark.py
# Re-evaluate the checkpoints already in models/benchmark/ without retraining
python src/run_benchmark.py --skip-train

# Risk-control rows only, without the other models (writes reports/risk_rows.json)
python src/risk_eval.py

# Optional training options (off by default; they did not help in our tuning)
#   --mask-dup     mask duplicate ids as false negatives in the InfoNCE loss
#   --neg-alpha A  sample negatives proportionally to popularity^A (0 = uniform)
python src/train.py --config configs/nlgcl.json --lambda-cl 0.01 --neg-alpha 0.5 --mask-dup

# Blend of the model score with item popularity (weight w chosen on validation, test read once)
python src/eval_blend.py --ckpt models/benchmark/nlgcl_s0.pt models/benchmark/nlgcl_s1.pt models/benchmark/nlgcl_s2.pt

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
  catalogue.py          Meal catalogue built from the Food.com recipes (diets, allergens, prices)
  main.py               FastAPI backend (endpoints, hard filters, predict() via api/model.py)
  mock_data.py          Mock meal catalogue (fallback when the pipeline has not been run)
  model.py              Ranking for the API (fold-in, popularity fallback, risk control)
.streamlit/
  config.toml           Streamlit theme (StudEat colours and font)
app/
  app.py                Streamlit interface (sign-up, weekly box, meal rating)
configs/
  lightgcn.json         Same configuration without the contrastive loss
  nlgcl.json            Hyperparameters of NLGCL
  risk_threshold.json   Calibrated risk threshold (alpha = 1.5%)
data/
  processed/            CSV outputs of the data pipeline (not versioned)
  raw/                  Raw datasets (not versioned, see "Datasets")
  studeat.db            SQLite database (not versioned)
models/
  embeddings.npz        Trained embeddings used by the API (versioned)
                        Checkpoints and histories (not versioned)
reports/
  figures/              Training curves used in the reports
  posters/              The 3 StudEat posters (PNG + A3 PDF)
  latency.md            Latency of /recommend (scripts/measure_latency.py)
  results_final.md      Final benchmark table (+ results_final.json)
  risk_rows.json        Risk-control rows alone (src/risk_eval.py)
scripts/
  measure_latency.py    Latency test of the API over HTTP
src/
  baselines.py          Random and Most-Popular baselines (MLE-Ops version)
  data_pipeline.py      Data pipeline
  eval_blend.py         Blend of the model score with popularity (weight chosen on validation)
  eval_by_activity.py   Results by user-activity level
  eval_cosine.py        Cosine + popularity score on fold-in vectors (API score)
  eval_popularity.py    Popularity baseline (MLE-Core version)
  evaluation.py         Shared evaluation harness (NDCG@10, Recall@20) used for all models
  export_embeddings.py  Export of the trained embeddings
  model_nlgcl.py        NLGCL / LightGCN model
  pipeline_config.py    Static configuration of the pipeline
  plot_history.py       Training curves
  recommender.py        Recommender used by the API (popularity fallback for new students)
  risk_control.py       Conformal calibration of the risk threshold
  risk_data_check.py    Share of unwanted ratings (<= 2) per split
  risk_eval.py          Risk control on test: rows with and without it (used by run_benchmark.py)
  run_baselines.py      Random and Most-Popular baselines with the shared harness
  run_benchmark.py      Final benchmark: all models over 3 seeds + risk-control rows, one results table
  scoring.py            App score (0.7 x cosine + 0.3 x popularity), shared with the calibration
  seed_test_data.py     Synthetic test students and ratings
  test_forward.py       Quick model check on synthetic data
  train.py              Training and evaluation (NDCG@10, Recall@20)
project_description_2026_2027_figures.ipynb   Figures of the project description
README.md
ROADMAP.md              Team roadmap
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

# Then call GET http://127.0.0.1:8000/recommend/{user_id}
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
