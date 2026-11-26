import numpy as np
import pandas as pd
import torch

from train import read_split, map_split, to_csr, evaluate

class FixedScores:
    """Fake model: the same score per item for every user."""
    def __init__(self, item_scores, n_users):
        self.U = torch.ones(n_users, 1)
        self.V = torch.as_tensor(item_scores, dtype=torch.float32).reshape(-1, 1)
    def final_embeddings(self):
        return self.U, self.V
    def eval(self): pass
    def train(self): pass


def main():
    d = "data/processed"
    ucol, icol, rcol, thr = "user_id", "recipe_id", "rating", 4

    # same data and same filter (rating >= 4) as train.py
    train = read_split(f"{d}/train.csv", ucol, icol, rcol, thr)
    val = read_split(f"{d}/val.csv", ucol, icol, rcol, thr)
    test = read_split(f"{d}/test.csv", ucol, icol, rcol, thr)

    # indices built ONLY from the training split (like train.py: no leakage)
    user_ids = np.sort(train[ucol].unique())
    item_ids = np.sort(train[icol].unique())
    n_users, n_items = len(user_ids), len(item_ids)
    u2i = pd.Series(np.arange(n_users), index=user_ids)
    i2i = pd.Series(np.arange(n_items), index=item_ids)

    tr_u, tr_i, _ = map_split(train, u2i, i2i, ucol, icol)
    va_u, va_i, _ = map_split(val, u2i, i2i, ucol, icol)
    te_u, te_i, te_drop = map_split(test, u2i, i2i, ucol, icol)
    print(f"test : {len(te_u)} interactions kept, {te_drop} ignored")

    train_csr = to_csr(tr_u, tr_i, n_users, n_items)
    val_csr = to_csr(va_u, va_i, n_users, n_items)
    test_csr = to_csr(te_u, te_i, n_users, n_items)

    pop = np.bincount(tr_i, minlength=n_items)          # popularity = number of TRAIN interactions
    rnd = np.random.default_rng(0).random(n_items)      # random score, fixed seed
    for name, scores in [("Random", rnd), ("Most-Popular", pop)]:
        ndcg, rec, n = evaluate(FixedScores(scores, n_users), test_csr,
                                [train_csr, val_csr], "cpu")
        print(f"{name:13s} TEST NDCG@10={ndcg:.4f}  Recall@20={rec:.4f}  ({n} users)")


if __name__ == "__main__":
    main()