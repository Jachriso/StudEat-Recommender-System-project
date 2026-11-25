"""
StudEat - PL: evaluate the Random and Most-Popular baselines with the shared harness.

Usage (from the project root):
  python src/run_baselines.py                 # same protocol as train.py
  python src/run_baselines.py --mask all      # also mask seen ratings < 4 (train, and val on test)
  python src/run_baselines.py --seeds 5       # Random averaged over 5 seeds
"""
import argparse

import numpy as np

from evaluation import evaluate_scores, load_splits, most_popular, random_scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--pos-threshold", type=float, default=4)
    p.add_argument("--mask", choices=["positives", "all"], default="positives",
                   help="seen items excluded from the ranking (train, plus val on the test split): "
                        "positives only (= train.py) or all interactions")
    p.add_argument("--seeds", type=int, default=5, help="number of seeds for Random")
    a = p.parse_args()

    d = load_splits(a.data_dir, pos_threshold=a.pos_threshold)
    # items excluded from the ranking: positives only (= train.py) or all interactions
    train_mask = d["train_pos"] if a.mask == "positives" else d["train_all"]
    val_mask = d["val_pos"] if a.mask == "positives" else d["val_all"]
    print(f"train : {d['n_users']} users, {d['n_items']} items, "
          f"{d['train_pos'].nnz} positive interactions")
    for s in ("val", "test"):
        st = d["stats"][s]
        pct = 100 * st["dropped"] / max(st["kept"] + st["dropped"], 1)
        print(f"{s:5s} : {st['kept']} positives kept, {st['dropped']} ignored "
              f"({pct:.1f} %, user/item unseen in training)")
    print(f"training mask : {a.mask}\n")

    splits = {"VAL": (d["val_pos"], [train_mask]),
              "TEST": (d["test_pos"], [train_mask, val_mask])}

    rows = []
    for split, (truth, masks) in splits.items():
        res = [evaluate_scores(random_scores(d["n_items"], seed), truth, masks)
               for seed in range(a.seeds)]
        nd, rc = np.array([r[0] for r in res]), np.array([r[1] for r in res])
        rows.append((split, f"Random ({a.seeds} seeds)",
                     f"{nd.mean():.4f} ± {nd.std():.4f}",
                     f"{rc.mean():.4f} ± {rc.std():.4f}", res[0][2]))

        n, r, k = evaluate_scores(most_popular(d["train_pos"]), truth, masks)
        rows.append((split, "Most-Popular", f"{n:.4f}", f"{r:.4f}", k))

    print(f"| Split | Model | NDCG@10 | Recall@20 | Users |")
    print(f"|---|---|---|---|---|")
    for row in rows:
        print("| " + " | ".join(str(x) for x in row) + " |")


if __name__ == "__main__":
    main()