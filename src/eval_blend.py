"""StudEat - blend of the model score and item popularity.

score(u, i) = model_score(u, i) + w * sigma * z_i
  z_i   = log(1 + number of positive ratings of the item in train), standardised
  sigma = standard deviation of the model scores (so that w is comparable across models)

The weight w is chosen on VALIDATION (mean over seeds). The test split is read
only once, at the chosen w.

Usage :
  python src/eval_blend.py --ckpt models/lam/nlgcl_l0.01_s0.pt models/lam/nlgcl_l0.01_s1.pt models/lam/nlgcl_l0.01_s2.pt
"""
import argparse

import numpy as np
import pandas as pd
import torch

from model_nlgcl import NLGCL, build_norm_adj
from train import evaluate, map_split, read_split, to_csr


class Shim:
    """Same interface as NLGCL, to reuse evaluate()."""
    def __init__(self, U, V):
        self.U, self.V = U, V

    def eval(self): pass
    def train(self): pass
    def final_embeddings(self): return self.U, self.V


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--weights", type=float, nargs="+",
                   default=[0, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5])
    a = p.parse_args()
    dev = torch.device("cpu")

    cfg = torch.load(a.ckpt[0], map_location=dev, weights_only=False)["config"]
    d, u_, i_, r_ = a.data_dir, cfg["user_col"], cfg["item_col"], cfg["rating_col"]
    thr = cfg["pos_threshold"]
    train = read_split(f"{d}/train.csv", u_, i_, r_, thr)
    val = read_split(f"{d}/val.csv", u_, i_, r_, thr)
    test = read_split(f"{d}/test.csv", u_, i_, r_, thr)
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

    lp = np.log1p(np.bincount(tr_i, minlength=n_items)).astype(np.float32)
    z = torch.as_tensor((lp - lp.mean()) / lp.std()).unsqueeze(1)

    adj = build_norm_adj(tr_u, tr_i, n_users, n_items)
    embs = []
    for path in a.ckpt:
        ck = torch.load(path, map_location=dev, weights_only=False)
        c = ck["config"]
        m = NLGCL(n_users, n_items, adj, dim=c["dim"], n_layers=c["layers"],
                  tau=c["tau"], lambda_cl=ck["lambda_cl"]).to(dev)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        with torch.no_grad():
            U, V = m.final_embeddings()
        g = torch.Generator().manual_seed(0)
        us = torch.randint(0, n_users, (2000,), generator=g)
        sigma = float((U[us] @ V.T).std())
        embs.append((U.detach(), V.detach(), sigma))

    def run(truth, masks, w):
        out = []
        for U, V, sigma in embs:
            Ua = torch.cat([U, torch.ones(U.size(0), 1)], 1)
            Va = torch.cat([V, w * sigma * z], 1)
            n, r, _ = evaluate(Shim(Ua, Va), truth, masks, dev)
            out.append((n, r))
        return np.array(out)

    print(f"{len(embs)} models; w is chosen on VALIDATION")
    print(f"{'w':>6}{'val NDCG@10':>14}")
    best_w, best_v = None, -1
    for w in a.weights:
        v = run(val_csr, [train_csr], w)[:, 0]
        print(f"{w:>6}{v.mean():>10.4f} +/- {v.std():.4f}")
        if v.mean() > best_v:
            best_w, best_v = w, v.mean()

    t = run(test_csr, [train_csr, val_csr], best_w)
    print(f"\nchosen w (validation) = {best_w}")
    print(f"TEST NDCG@10 = {t[:, 0].mean():.4f} +/- {t[:, 0].std():.4f}   "
          f"Recall@20 = {t[:, 1].mean():.4f} +/- {t[:, 1].std():.4f}")

    # popularity reference computed on the SAME data / masks (no hardcoded numbers)
    pop = torch.as_tensor(np.bincount(tr_i, minlength=n_items), dtype=torch.float32)
    pop_model = Shim(torch.ones(n_users, 1), pop.unsqueeze(1))
    pv, _, _ = evaluate(pop_model, val_csr, [train_csr], dev)
    pn, pr, _ = evaluate(pop_model, test_csr, [train_csr, val_csr], dev)
    print(f"Most-Popular (same data): val NDCG@10 {pv:.4f} | test NDCG@10 {pn:.4f} | test Recall@20 {pr:.4f}")


if __name__ == "__main__":
    main()
