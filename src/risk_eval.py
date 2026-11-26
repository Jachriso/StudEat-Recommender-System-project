import argparse
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

from evaluation import load_splits, _top_k
from recommender import Recommender
from risk_control import build_calibration, foldin_vectors
from scoring import blend_scores, pop01


def aligned_embeddings(npz_path, user_ids, item_ids):
    """Reorder U, V and popularity to follow the indices of load_splits (zeros if unknown)."""
    z = np.load(npz_path)
    u_pos = pd.Series(np.arange(len(z["user_ids"])), index=z["user_ids"]).reindex(user_ids).to_numpy()
    i_pos = pd.Series(np.arange(len(z["item_ids"])), index=z["item_ids"]).reindex(item_ids).to_numpy()
    known_u, known_i = ~np.isnan(u_pos), ~np.isnan(i_pos)
    U = np.zeros((len(user_ids), z["U"].shape[1]))
    U[known_u] = z["U"][u_pos[known_u].astype(int)]
    V = np.zeros((len(item_ids), z["V"].shape[1]))
    V[known_i] = z["V"][i_pos[known_i].astype(int)]
    P = np.zeros(len(item_ids))
    P[known_i] = pop01(z["popularity"])[i_pos[known_i].astype(int)]   # scaled on the whole catalogue
    return U, V, P, known_u, known_i


def foldin_matrix(pos, V, known_i):
    """Fold-in user vectors from a positives matrix: mean of V over the liked meals."""
    pos = pos[:, known_i]
    cnt = np.asarray(pos.sum(axis=1)).ravel()
    U = pos @ V[known_i]
    U[cnt > 0] /= cnt[cnt > 0, None]
    return U, cnt > 0


def evaluate_rc(score_fn, known_i, truth, masks, t, k_ndcg=10, k_rec=20, chunk=1024):
    """Same as evaluation.evaluate_scores, but a slot whose score is < t is
    replaced by an already-rated safe meal, so it can no longer be a hit.
    Also returns the share of replaced slots in the real top-10."""
    users = np.where(np.diff(truth.indptr) > 0)[0]
    disc = 1.0 / np.log2(np.arange(2, k_ndcg + 2))
    idcg = np.cumsum(disc)
    nd, rc, rp = [], [], []
    for s in range(0, len(users), chunk):
        ub = users[s:s + chunk]
        scores = np.array(score_fn(ub), dtype=np.float64)
        scores[:, ~known_i] = -np.inf
        for m in masks:
            scores[m[ub].toarray() > 0] = -np.inf
        top = _top_k(scores, max(k_ndcg, k_rec))
        kept = np.take_along_axis(scores, top, axis=1) >= t
        rp.append(1 - kept[:, :k_ndcg].mean(axis=1))
        rel = truth[ub].toarray() > 0
        hits = (np.take_along_axis(rel, top, axis=1) & kept).astype(float)
        n_rel = rel.sum(axis=1)
        rc.append(hits[:, :k_rec].sum(axis=1) / n_rel)
        nd.append((hits[:, :k_ndcg] * disc).sum(axis=1) / idcg[np.minimum(n_rel, k_ndcg) - 1])
    return (float(np.concatenate(nd).mean()), float(np.concatenate(rc).mean()),
            len(users), float(np.concatenate(rp).mean()))


def keep_rows(m, mask):
    out = (sp.diags(mask.astype(np.float32)) @ m).tocsr()
    out.eliminate_zeros()
    return out


def risk_rows(emb="models/embeddings.npz", data_dir="data/processed",
              risk_path="configs/risk_threshold.json", d=None):
    """Rows of the benchmark table with and without risk control (also used by run_benchmark.py).
    d: output of load_splits, to avoid loading the splits twice."""
    with open(risk_path) as f:
        cfg = json.load(f)
    t_rc, k, umax = cfg["threshold"], cfg["k"], cfg["unwanted_max"]
    foldin = cfg.get("mode") == "foldin_blend"
    tag = "NLGCL fold-in (app score)" if foldin else "NLGCL"

    # 1) unwanted-meal rate on test (same scoring as the calibration)
    rec = Recommender(emb)
    vecs = foldin_vectors(rec, [f"{data_dir}/train.csv", f"{data_dir}/val.csv"]) if foldin else None
    S, W, n = build_calibration(f"{data_dir}/test.csv", rec, k, umax, vecs)

    # 2) NDCG@10 / Recall@20, same masks as run_benchmark.py
    if d is None:
        d = load_splits(data_dir)
    U, V, P, known_u, known_i = aligned_embeddings(emb, d["user_ids"], d["item_ids"])
    if foldin:   # val: liked meals of train; test: liked meals of train + val
        U_val, ok_val = foldin_matrix(d["train_pos"], V, known_i)
        U_test, ok_test = foldin_matrix(d["train_pos"] + d["val_pos"], V, known_i)
        val_fn = lambda ub: blend_scores(U_val[ub], V, P)
        test_fn = lambda ub: blend_scores(U_test[ub], V, P)
    else:
        ok_val = ok_test = known_u
        val_fn = test_fn = lambda ub: U[ub] @ V.T
    val_t = keep_rows(d["val_pos"], ok_val)
    test_t = keep_rows(d["test_pos"], ok_test)

    rows = []
    for name, t in [(f"{tag} (no risk control)", -np.inf),
                    (f"{tag} + risk control (alpha = {cfg['alpha']})", t_rc)]:
        vn, _, _, _ = evaluate_rc(val_fn, known_i, val_t, [d["train_pos"]], t)
        tn, tr, nu, rp = evaluate_rc(test_fn, known_i, test_t, [d["train_pos"], d["val_pos"]], t)
        risk = W[S >= t].sum() / n
        print(f"{name:50s} val NDCG@10={vn:.4f} | test NDCG@10={tn:.4f} "
              f"Recall@20={tr:.4f} ({nu} users) | unwanted={risk:.4f} | replaced in top-10={rp:.1%}")
        rows.append({"model": name, "val_ndcg": [vn], "test_ndcg": [tn],
                     "test_recall": [tr], "unwanted_rate": [float(risk)]})
    print(f"test risk n={n} users | meals replaced (calibration protocol): {(S < t_rc).mean():.1%}")

    return rows


def main():
    p = argparse.ArgumentParser(description="Risk control on test + NDCG/Recall ablation")
    p.add_argument("--emb", default="models/embeddings.npz")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--risk", default="configs/risk_threshold.json")
    p.add_argument("--out", default="reports/risk_rows.json")
    a = p.parse_args()

    rows = risk_rows(a.emb, a.data_dir, a.risk)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
