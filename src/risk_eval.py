import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from evaluation import load_splits, _top_k
from recommender import Recommender
from risk_control import build_calibration


def aligned_embeddings(npz_path, user_ids, item_ids):
    """Reorder U and V to follow the indices of load_splits (rows of zeros if unknown)."""
    z = np.load(npz_path)
    u_pos = pd.Series(np.arange(len(z["user_ids"])), index=z["user_ids"]).reindex(user_ids).to_numpy()
    i_pos = pd.Series(np.arange(len(z["item_ids"])), index=z["item_ids"]).reindex(item_ids).to_numpy()
    known_u, known_i = ~np.isnan(u_pos), ~np.isnan(i_pos)
    U = np.zeros((len(user_ids), z["U"].shape[1]))
    U[known_u] = z["U"][u_pos[known_u].astype(int)]
    V = np.zeros((len(item_ids), z["V"].shape[1]))
    V[known_i] = z["V"][i_pos[known_i].astype(int)]
    return U, V, known_u, known_i


def evaluate_rc(U, V, known_i, truth, masks, t, k_ndcg=10, k_rec=20, chunk=1024):
    """Same as evaluation.evaluate_scores, but a slot whose score is < t is
    replaced by an already-rated safe meal, so it can no longer be a hit."""
    users = np.where(np.diff(truth.indptr) > 0)[0]
    disc = 1.0 / np.log2(np.arange(2, k_ndcg + 2))
    idcg = np.cumsum(disc)
    nd, rc = [], []
    for s in range(0, len(users), chunk):
        ub = users[s:s + chunk]
        scores = U[ub] @ V.T
        scores[:, ~known_i] = -np.inf
        for m in masks:
            scores[m[ub].toarray() > 0] = -np.inf
        top = _top_k(scores, max(k_ndcg, k_rec))
        kept = np.take_along_axis(scores, top, axis=1) >= t
        rel = truth[ub].toarray() > 0
        hits = (np.take_along_axis(rel, top, axis=1) & kept).astype(float)
        n_rel = rel.sum(axis=1)
        rc.append(hits[:, :k_rec].sum(axis=1) / n_rel)
        nd.append((hits[:, :k_ndcg] * disc).sum(axis=1) / idcg[np.minimum(n_rel, k_ndcg) - 1])
    return float(np.concatenate(nd).mean()), float(np.concatenate(rc).mean()), len(users)


def main():
    with open("configs/risk_threshold.json") as f:
        cfg = json.load(f)
    t_rc, k, umax = cfg["threshold"], cfg["k"], cfg["unwanted_max"]

    # 1) unwanted-meal rate on test (step 1)
    rec = Recommender("models/embeddings.npz")
    S, W, n = build_calibration("data/processed/test.csv", rec, k=k, unwanted_max=umax)

    # 2) NDCG@10 / Recall@20 with and without risk control
    d = load_splits("data/processed")
    U, V, known_u, known_i = aligned_embeddings("models/embeddings.npz", d["user_ids"], d["item_ids"])
    keep = sp.diags(known_u.astype(np.float32))          # only users the model knows
    val_t = (keep @ d["val_pos"]).tocsr()
    val_t.eliminate_zeros()
    test_t = (keep @ d["test_pos"]).tocsr()
    test_t.eliminate_zeros()

    rows = []
    for name, t in [("NLGCL (no risk control)", -np.inf),
                    (f"NLGCL + risk control (alpha = {cfg['alpha']})", t_rc)]:
        vn, _, _ = evaluate_rc(U, V, known_i, val_t, [d["train_all"]], t)
        tn, tr, nu = evaluate_rc(U, V, known_i, test_t, [d["train_all"], d["val_all"]], t)
        risk = W[S >= t].sum() / n
        print(f"{name:40s} val NDCG@10={vn:.4f} | test NDCG@10={tn:.4f} "
              f"Recall@20={tr:.4f} ({nu} users) | unwanted={risk:.4f}")
        rows.append({"model": name, "val_ndcg": [vn], "test_ndcg": [tn],
                     "test_recall": [tr], "unwanted_rate": [float(risk)]})

    with open("reports/risk_rows.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("-> reports/risk_rows.json")


if __name__ == "__main__":
    main()
