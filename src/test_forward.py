"""
StudEat - MLE-Core: NLGCL forward-pass test (week 2 deliverable).

Usage (from the project root):

  # 1) synthetic data (no file needed)
  python src/test_forward.py

  # 2) real data: a CSV of TRAINING interactions with a user column
  #    and an item column (adapt the column names to data/processed/)
  python src/test_forward.py --csv data/processed/train.csv \
      --user-col user_id --item-col recipe_id
"""
import argparse

import numpy as np
import pandas as pd
import torch

from model_nlgcl import NLGCL, build_norm_adj


def load_interactions(args):
    """Returns (users, items, n_users, n_items) with contiguous ids 0..n-1."""
    if args.csv is None:
        rng = np.random.default_rng(0)
        n_users, n_items, n_inter = 200, 100, 3000
        users = rng.integers(0, n_users, n_inter)
        items = rng.integers(0, n_items, n_inter)
        df = pd.DataFrame({"u": users, "i": items}).drop_duplicates()
        return df["u"].values, df["i"].values, n_users, n_items

    df = pd.read_csv(args.csv)[[args.user_col, args.item_col]].drop_duplicates()
    # remap to contiguous indices (to be saved for predict()!)
    user_ids, user_idx = np.unique(df[args.user_col].values, return_inverse=True)
    item_ids, item_idx = np.unique(df[args.item_col].values, return_inverse=True)
    return user_idx, item_idx, len(user_ids), len(item_ids)


def sample_batch(users, items, n_items, batch_size, rng):
    """Batch (u, i+, i-): random negative (not checked for this test)."""
    sel = rng.integers(0, len(users), batch_size)
    u = torch.tensor(users[sel], dtype=torch.long)
    pos = torch.tensor(items[sel], dtype=torch.long)
    neg = torch.tensor(rng.integers(0, n_items, batch_size), dtype=torch.long)
    return u, pos, neg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=None)
    p.add_argument("--user-col", default="user_id")
    p.add_argument("--item-col", default="recipe_id")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    args = p.parse_args()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    users, items, n_users, n_items = load_interactions(args)
    print(f"users={n_users}  items={n_items}  interactions={len(users)}")

    adj = build_norm_adj(users, items, n_users, n_items)
    print("adjacency:", tuple(adj.shape), "nnz =", adj._nnz())

    for name, lam in (("LightGCN (lambda_cl=0)", 0.0), ("NLGCL (lambda_cl=0.1)", 0.1)):
        model = NLGCL(n_users, n_items, adj, dim=args.dim,
                      n_layers=args.layers, lambda_cl=lam)
        u, pos, neg = sample_batch(users, items, n_items, args.batch_size, rng)

        # forward + backward: check that gradients flow
        loss, parts = model(u, pos, neg)
        loss.backward()
        grad_norm = model.emb.weight.grad.norm().item()
        print(f"\n[{name}] loss={loss.item():.4f}  parts={parts}  grad_norm={grad_norm:.4f}")

        # final embeddings + scores for one user
        U, V = model.final_embeddings()
        print("final embeddings:", tuple(U.shape), tuple(V.shape))
        scores = model.score_all_items(0)
        top = torch.topk(scores, k=min(10, n_items)).indices.tolist()
        print("top-10 items for user 0 (untrained model):", top)

    print("\nOK: forward + backward passes work.")


if __name__ == "__main__":
    main()
