"""StudEat - cosine + popularity score (for the fold-in in the API).

score(u, i) = cos(u, i) + w * pop01_i
  pop01_i = log(1 + number of positive ratings of the item in train), scaled to [0, 1]
  (the 0.7*cos + 0.3*pop blend corresponds to w = 0.3/0.7 = 0.43)

For each w: NDCG@10 on VALIDATION, and the mean overlap (out of 10) between the model's top 10
and the popularity top 10. w is chosen on validation; the test split is read only once.

Usage :
  python src/eval_cosine.py --ckpt models/final/nlgcl_s0.pt models/final/nlgcl_s1.pt models/final/nlgcl_s2.pt
"""
import argparse

import numpy as np
import pandas as pd
import torch

from model_nlgcl import NLGCL, build_norm_adj
from train import evaluate, map_split, read_split, to_csr


class Shim:
    def __init__(self, U, V):
        self.U, self.V = U, V

    def eval(self): pass
    def train(self): pass
    def final_embeddings(self): return self.U, self.V


def unit(x):
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def top10(U, V, users, seen_csr, chunk=1024):
    out = []
    for s in range(0, len(users), chunk):
        ub = users[s:s + chunk]
        sc = U[torch.as_tensor(ub)] @ V.T
        sc[torch.as_tensor(seen_csr[ub].toarray() > 0)] = -1e9
        out.append(torch.topk(sc, 10).indices)
    return torch.cat(out)


def overlap(Ua, Va, pop_top, users, seen_csr):
    top = top10(Ua, Va, users, seen_csr)
    n = 0
    for a, b in zip(top.tolist(), pop_top.tolist()):
        n += len(set(a) & set(b))
    return n / len(users)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--weights", type=float, nargs="+",
                   default=[0, 0.05, 0.1, 0.2, 0.3, 0.43, 0.6, 1, 2])
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

    counts = np.bincount(tr_i, minlength=n_items).astype(np.float32)
    lp = np.log1p(counts)
    pop01 = torch.as_tensor((lp - lp.min()) / (lp.max() - lp.min())).unsqueeze(1)
    pop_model = Shim(torch.ones(n_users, 1), torch.as_tensor(counts).unsqueeze(1))

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
        embs.append((U.detach(), V.detach()))

    users_val = np.where(np.diff(val_csr.indptr) > 0)[0]
    pop_top = top10(torch.ones(n_users, 1), torch.as_tensor(counts).unsqueeze(1), users_val, train_csr)

    def run(truth, masks, w):
        res = []
        for U, V in embs:
            Ua = torch.cat([unit(U), torch.ones(n_users, 1)], 1)
            Va = torch.cat([unit(V), w * pop01], 1)
            res.append(evaluate(Shim(Ua, Va), truth, masks, dev)[0])
        return np.array(res)

    def ov(w):
        r = []
        for U, V in embs:
            Ua = torch.cat([unit(U), torch.ones(n_users, 1)], 1)
            Va = torch.cat([unit(V), w * pop01], 1)
            r.append(overlap(Ua, Va, pop_top, users_val, train_csr))
        return float(np.mean(r))

    dot_v = np.array([evaluate(Shim(U, V), val_csr, [train_csr], dev)[0] for U, V in embs])
    dot_ov = float(np.mean([overlap(U, V, pop_top, users_val, train_csr) for U, V in embs]))
    print(f"{len(embs)} models. Reference, dot product (current API): "
          f"val {dot_v.mean():.4f}, overlap with the popularity box {dot_ov:.1f}/10")
    print(f"\n{'w':>6}{'val NDCG@10':>16}{'overlap /10':>20}")
    best_w, best_v = None, -1
    for w in a.weights:
        v = run(val_csr, [train_csr], w)
        print(f"{w:>6}{v.mean():>10.4f} +/- {v.std():.4f}{ov(w):>14.1f}")
        if v.mean() > best_v:
            best_w, best_v = w, v.mean()

    t = run(test_csr, [train_csr, val_csr], best_w)
    pn = evaluate(pop_model, test_csr, [train_csr, val_csr], dev)[0]
    print(f"\nchosen w (best VALIDATION) = {best_w}")
    print(f"TEST NDCG@10 = {t.mean():.4f} +/- {t.std():.4f}  (popularity: {pn:.4f})")
    print("Product choice for the demo: a larger w = a less personalised but more accurate box.")


if __name__ == "__main__":
    main()
