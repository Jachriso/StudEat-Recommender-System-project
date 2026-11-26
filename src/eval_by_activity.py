"""StudEat - NDCG/Recall by user-activity bucket: model vs popularity (test).

Usage: python src/eval_by_activity.py --ckpt models/nlgcl.pt
Bucket = number of positive interactions of the user in the training split.
"""
import argparse

import numpy as np
import pandas as pd
import torch

from model_nlgcl import NLGCL, build_norm_adj
from train import map_split, read_split, to_csr


@torch.no_grad()
def per_user(score_fn, truth, masks, chunk=1024, k_ndcg=10, k_rec=20):
    users = np.where(np.diff(truth.indptr) > 0)[0]
    discount = 1.0 / torch.log2(torch.arange(2, k_ndcg + 2).float())
    idcg = torch.cumsum(discount, dim=0)
    nd, rc = [], []
    for s in range(0, len(users), chunk):
        ub = users[s:s + chunk]
        scores = score_fn(ub).clone()
        for m in masks:
            scores = scores.masked_fill(torch.as_tensor(m[ub].toarray() > 0), -1e9)
        rel = torch.as_tensor(truth[ub].toarray() > 0)
        hits = rel.gather(1, torch.topk(scores, max(k_ndcg, k_rec)).indices).float()
        n_rel = rel.sum(1).float()
        rc.append(hits[:, :k_rec].sum(1) / n_rel)
        dcg = (hits[:, :k_ndcg] * discount).sum(1)
        nd.append(dcg / idcg[torch.clamp(n_rel, max=k_ndcg).long() - 1])
    return users, torch.cat(nd).numpy(), torch.cat(rc).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", default=None)
    a = p.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    d = a.data_dir or cfg["data_dir"]
    u_, i_, r_, thr = cfg["user_col"], cfg["item_col"], cfg["rating_col"], cfg["pos_threshold"]

    user_ids, item_ids = ck["user_ids"], ck["item_ids"]
    n_users, n_items = len(user_ids), len(item_ids)
    u2i = pd.Series(np.arange(n_users), index=user_ids)
    i2i = pd.Series(np.arange(n_items), index=item_ids)
    m = {}
    for name in ("train", "val", "test"):
        df = read_split(f"{d}/{name}.csv", u_, i_, r_, thr)
        m[name] = map_split(df, u2i, i2i, u_, i_)[:2]
    csr = {k: to_csr(v[0], v[1], n_users, n_items) for k, v in m.items()}

    adj = build_norm_adj(m["train"][0], m["train"][1], n_users, n_items)
    model = NLGCL(n_users, n_items, adj, dim=cfg["dim"], n_layers=cfg["layers"],
                  tau=cfg["tau"], lambda_cl=ck["lambda_cl"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    with torch.no_grad():
        U, V = model.final_embeddings()
    pop = torch.as_tensor(np.asarray(csr["train"].sum(0)).ravel(), dtype=torch.float32)

    masks = [csr["train"], csr["val"]]
    users, nd_m, rc_m = per_user(lambda ub: U[torch.as_tensor(ub)] @ V.T, csr["test"], masks)
    _, nd_p, rc_p = per_user(lambda ub: pop.unsqueeze(0).repeat(len(ub), 1), csr["test"], masks)
    print(f"\nALL ({len(users)} users) | model NDCG@10={nd_m.mean():.4f} Recall@20={rc_m.mean():.4f}"
          f" | popularity NDCG@10={nd_p.mean():.4f} Recall@20={rc_p.mean():.4f}")

    act = np.diff(csr["train"].indptr)[users]
    print(f"\n{'interactions train':<20}{'users':>7}{'pop NDCG':>11}{'model NDCG':>13}{'delta':>9}"
          f"{'pop Rec':>10}{'model Rec':>12}")
    for lo, hi in [(1, 5), (6, 10), (11, 20), (21, 10**9)]:
        sel = (act >= lo) & (act <= hi)
        if sel.sum() == 0:
            continue
        lab = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
        print(f"{lab:<20}{int(sel.sum()):>7}{nd_p[sel].mean():>11.4f}{nd_m[sel].mean():>13.4f}"
              f"{nd_m[sel].mean() - nd_p[sel].mean():>+9.4f}{rc_p[sel].mean():>10.4f}{rc_m[sel].mean():>12.4f}")


if __name__ == "__main__":
    main()
