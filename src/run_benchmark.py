"""
StudEat - PL: final benchmark of every model with the shared evaluation harness.

For each model and each seed, this script:
  1. trains the model with src/train.py (its JSON config + the seed),
  2. reloads the best checkpoint (selected on validation by train.py),
  3. evaluates it with src/evaluation.py, exactly like the baselines.
Random and Most-Popular are evaluated with the same harness, on the same split.
The model used by the app (fold-in vectors from models/embeddings.npz + app score) is added with
and without risk control (threshold in configs/risk_threshold.json, see src/risk_eval.py).
The result is ONE table (mean ± std over the seeds), written to reports/results_final.md.

Usage (from the project root):
  python src/run_benchmark.py                          # LightGCN + NLGCL, seeds 0 1 2
  python src/run_benchmark.py --epochs 2               # quick check of the whole chain
  python src/run_benchmark.py --skip-train             # re-evaluate existing checkpoints only
  python src/run_benchmark.py --models nlgcl=configs/nlgcl.json nlgcl_neg=configs/nlgcl_neg.json
  python src/run_benchmark.py --no-risk                # without the two risk-control rows

--extra-rows (optional) adds rows computed elsewhere, as a JSON list:
  [{"model": "My model", "val_ndcg": [..], "test_ndcg": [..], "test_recall": [..], "unwanted_rate": [..]}]
"model" is required. "val_ndcg", "test_ndcg", "test_recall" and "unwanted_rate" hold one value
per seed; a missing metric is shown as "n/a" in the table.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import date

import numpy as np
import torch

from evaluation import (embedding_scores, evaluate_scores, load_splits,
                        most_popular, random_scores)
from model_nlgcl import NLGCL, build_norm_adj
from risk_eval import risk_rows

DEFAULT_MODELS = {"LightGCN": "configs/lightgcn.json", "NLGCL": "configs/nlgcl.json"}
SRC_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_models(pairs):
    """['name=path.json', ...] -> {name: path}; empty -> the default models."""
    if not pairs:
        return dict(DEFAULT_MODELS)
    models = {}
    for pair in pairs:
        name, sep, path = pair.partition("=")
        if not sep or not name or not path:
            sys.exit(f"--models expects name=config.json, got: {pair}")
        models[name] = path
    return models


def train(config, seed, out, epochs, data_dir):
    """Train one model with src/train.py (same Python, so the same virtual environment).

    data_dir is forwarded so that training and evaluation always read the same split.
    """
    cmd = [sys.executable, os.path.join(SRC_DIR, "train.py"),
           "--config", config, "--seed", str(seed), "--out", out, "--data-dir", data_dir]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_embeddings(path, d):
    """Rebuild the model from a train.py checkpoint and return its final embeddings (U, V)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    # the checkpoint's indices must be the harness's indices, or every score is shuffled
    if not (np.array_equal(ck["user_ids"], d["user_ids"])
            and np.array_equal(ck["item_ids"], d["item_ids"])):
        sys.exit(f"{path}: users/items differ from the current split. "
                 "Re-run the pipeline and retrain, or use the matching data.")
    c = ck["config"]
    tr_u, tr_i = d["train_pos"].nonzero()
    adj = build_norm_adj(tr_u, tr_i, d["n_users"], d["n_items"])
    model = NLGCL(d["n_users"], d["n_items"], adj, dim=c["dim"], n_layers=c["layers"],
                  tau=c["tau"], lambda_cl=ck["lambda_cl"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    with torch.no_grad():
        U, V = model.final_embeddings()
    return U.numpy(), V.numpy()


def fmt(values, digits=4):
    """Mean ± sample std over seeds (ddof=1); only the value when there is a single one."""
    v = np.asarray(values, dtype=float)
    if len(v) == 1:
        return f"{v[0]:.{digits}f}"
    return f"{v.mean():.{digits}f} ± {v.std(ddof=1):.{digits}f}"


def cell(row, key):
    """Formatted metric of a row, or "n/a" when the row does not provide it."""
    return fmt(row[key]) if row.get(key) else "n/a"


def main():
    p = argparse.ArgumentParser(description="Final benchmark with the shared harness")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--models", nargs="*", default=None,
                   help="models to train, as name=config.json (default: LightGCN and NLGCL)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=None, help="override the configs' epochs")
    p.add_argument("--random-seeds", type=int, default=5, help="seeds for the Random baseline")
    p.add_argument("--out-dir", default="models/benchmark", help="checkpoints (not versioned)")
    p.add_argument("--skip-train", action="store_true", help="reuse existing checkpoints")
    p.add_argument("--emb", default="models/embeddings.npz", help="embeddings used by the app")
    p.add_argument("--risk", default="configs/risk_threshold.json", help="calibrated risk threshold")
    p.add_argument("--no-risk", action="store_true", help="skip the two risk-control rows")
    p.add_argument("--extra-rows", nargs="*", default=[],
                   help="JSON files with rows computed elsewhere (optional)")
    p.add_argument("--report", default="reports/results_final.md")
    a = p.parse_args()

    models = parse_models(a.models)
    os.makedirs(a.out_dir, exist_ok=True)

    # --- 1. training (one run per model and seed) -----------------------------
    ckpts = {name: [os.path.join(a.out_dir, f"{name.lower()}_s{s}.pt") for s in a.seeds]
             for name in models}
    if not a.skip_train:
        for name, config in models.items():
            for seed, out in zip(a.seeds, ckpts[name]):
                train(config, seed, out, a.epochs, a.data_dir)

    # --- 2. evaluation with the shared harness ---------------------------------
    d = load_splits(a.data_dir)
    val = (d["val_pos"], [d["train_pos"]])                    # same masks as train.py
    test = (d["test_pos"], [d["train_pos"], d["val_pos"]])

    rows = []

    def add_row(model, score_fns):
        r = {"model": model, "val_ndcg": [], "test_ndcg": [], "test_recall": []}
        for fn in score_fns:
            r["val_ndcg"].append(evaluate_scores(fn, *val)[0])
            n, rec, users = evaluate_scores(fn, *test)
            r["test_ndcg"].append(n)
            r["test_recall"].append(rec)
        r["test_users"] = users
        rows.append(r)
        print(f"{model:<28} test NDCG@10 {fmt(r['test_ndcg'])}  Recall@20 {fmt(r['test_recall'])}")

    print("\nEvaluating with src/evaluation.py ...")
    add_row(f"Random ({a.random_seeds} seeds)",
            [random_scores(d["n_items"], s) for s in range(a.random_seeds)])
    add_row("Most-Popular", [most_popular(d["train_pos"])])
    for name in models:
        missing = [c for c in ckpts[name] if not os.path.exists(c)]
        if missing:
            sys.exit(f"Missing checkpoints for {name}: {missing} (run without --skip-train)")
        add_row(f"{name} ({len(a.seeds)} seeds)",
                [embedding_scores(*load_embeddings(c, d)) for c in ckpts[name]])

    # App model with and without risk control (same harness masks, see src/risk_eval.py)
    if not a.no_risk:
        for f_ in (a.emb, a.risk):
            if not os.path.exists(f_):
                sys.exit(f"Missing {f_} (run src/risk_control.py --foldin, or use --no-risk)")
        for r in risk_rows(a.emb, a.data_dir, a.risk, d=d):
            rows.append(r)
            print(f"{r['model']:<28} test NDCG@10 {fmt(r['test_ndcg'])}  unwanted {fmt(r['unwanted_rate'])}")

    for path in a.extra_rows:
        with open(path, encoding="utf-8") as f:
            for r in json.load(f):
                if "model" not in r:
                    sys.exit(f"{path}: every extra row needs a \"model\" name")
                rows.append(r)
                print(f"{r['model']:<28} (from {path})")

    # --- 3. report -------------------------------------------------------------
    has_unwanted = any("unwanted_rate" in r for r in rows)
    header = "| Model | Val NDCG@10 | Test NDCG@10 | Test Recall@20 |"
    sep = "|---|---|---|---|"
    if has_unwanted:
        header += " Unwanted-meal rate |"
        sep += "---|"
    lines = [header, sep]
    for r in rows:
        line = (f"| {r['model']} | {cell(r, 'val_ndcg')} | {cell(r, 'test_ndcg')} "
                f"| {cell(r, 'test_recall')} |")
        if has_unwanted:
            line += f" {cell(r, 'unwanted_rate')} |"
        lines.append(line)
    table = "\n".join(lines)

    # Reproduce block: checkpoints are not versioned, so the command trains (no --skip-train).
    epochs = f" --epochs {a.epochs}" if a.epochs else ""
    reproduce = f"python src/run_benchmark.py{epochs} --seeds {' '.join(map(str, a.seeds))}"

    stats = d["stats"]
    report = f"""# StudEat – final results

*Generated by `src/run_benchmark.py` on {date.today().isoformat()}.*

{table}

**Protocol.** Temporal split produced by `src/data_pipeline.py`; every row is evaluated with the
shared harness `src/evaluation.py` (NDCG@10, Recall@20, full-catalogue ranking, already-seen items
masked, positives = ratings ≥ 4). Models are selected on validation (early stopping in `train.py`);
the test split is used once per checkpoint. Values are mean ± sample std (ddof = 1) over seeds {a.seeds}
(Random: {a.random_seeds} seeds; Most-Popular is deterministic).

**Data.** {d['n_users']:,} users and {d['n_items']:,} recipes in the training split,
{d['train_pos'].nnz:,} positive training interactions; {stats['test']['kept']:,} positive test
interactions from {rows[0]['test_users']:,} users.

**Reproduce.**

```bash
{reproduce}
```

Checkpoints are not versioned: this trains every model, then evaluates it. Add `--skip-train` to
re-evaluate the checkpoints already in `{a.out_dir}`. The risk-control rows use the versioned
`{a.emb}` and `{a.risk}` (recalibrate with `python src/risk_control.py --foldin`).
"""
    os.makedirs(os.path.dirname(a.report) or ".", exist_ok=True)
    with open(a.report, "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.splitext(a.report)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump({"seeds": a.seeds, "models": models, "rows": rows}, f, indent=2)

    print("\n" + table)
    print(f"\nReport written to {a.report} (and {os.path.splitext(a.report)[0]}.json)")


if __name__ == "__main__":
    main()
