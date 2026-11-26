"""StudEat - popularity baseline (same metrics and same masks as train.py).

Usage: python src/eval_popularity.py
"""
import argparse

import numpy as np
import pandas as pd
import torch

from train import evaluate, map_split, read_split, to_csr


class PopularityModel:
    """Same interface as NLGCL to reuse evaluate(): score = item popularity."""

    def __init__(self, pop):
        self.U = torch.ones(1, 1)
        self.V = torch.as_tensor(pop, dtype=torch.float32).unsqueeze(1)

    def eval(self): pass
    def train(self): pass

    def final_embeddings(self):
        return self.U.expand(self._n_users, 1), self.V


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--user-col", default="user_id")
    p.add_argument("--item-col", default="recipe_id")
    p.add_argument("--rating-col", default="rating")
    p.add_argument("--pos-threshold", type=float, default=4)
    a = p.parse_args()
    d, u_, i_, r_ = a.data_dir, a.user_col, a.item_col, a.rating_col

    train = read_split(f"{d}/train.csv", u_, i_, r_, a.pos_threshold)
    val = read_split(f"{d}/val.csv", u_, i_, r_, a.pos_threshold)
    test = read_split(f"{d}/test.csv", u_, i_, r_, a.pos_threshold)
    user_ids, item_ids = np.sort(train[u_].unique()), np.sort(train[i_].unique())
    n_users, n_items = len(user_ids), len(item_ids)
    u2i = pd.Series(np.arange(n_users), index=user_ids)
    i2i = pd.Series(np.arange(n_items), index=item_ids)
    tr_u, tr_i, _ = map_split(train, u2i, i2i, u_, i_)
    va_u, va_i, _ = map_split(val, u2i, i2i, u_, i_)
    te_u, te_i, _ = map_split(test, u2i, i2i, u_, i_)
    train_csr, val_csr, test_csr = (to_csr(tr_u, tr_i, n_users, n_items),
                                    to_csr(va_u, va_i, n_users, n_items),
                                    to_csr(te_u, te_i, n_users, n_items))

    model = PopularityModel(np.bincount(tr_i, minlength=n_items))
    model._n_users = n_users
    dev = torch.device("cpu")
    n, r, k = evaluate(model, val_csr, [train_csr], dev)
    print(f"VAL  popularity NDCG@10={n:.4f}  Recall@20={r:.4f}  ({k} users)")
    n, r, k = evaluate(model, test_csr, [train_csr, val_csr], dev)
    print(f"TEST popularity NDCG@10={n:.4f}  Recall@20={r:.4f}  ({k} users)")


if __name__ == "__main__":
    main()
