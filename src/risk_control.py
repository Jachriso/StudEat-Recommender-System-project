import argparse
import json

import numpy as np
import pandas as pd

from recommender import Recommender
from scoring import blend_scores, pop01


def foldin_vectors(rec, csv_paths, pos_min=4):
    """User vector = mean of the item embeddings of the meals rated >= pos_min,
    exactly like api/model.py does for a student of the app."""
    df = pd.concat([pd.read_csv(p, usecols=["user_id", "recipe_id", "rating"]) for p in csv_paths])
    df = df[df["rating"] >= pos_min]
    vecs = {}
    for uid, g in df.groupby("user_id"):
        cols = [rec._item_col[int(i)] for i in g["recipe_id"] if int(i) in rec._item_col]
        if cols:
            vecs[uid] = rec.V[cols].mean(axis=0)
    return vecs


def foldin_scores(rec, u, item_ids, pop):
    """Same score as the app (src/scoring.py); unknown meals -> -inf."""
    cols = np.array([rec._item_col.get(int(i), -1) for i in item_ids])
    out = np.full(len(cols), -np.inf)
    ok = cols >= 0
    out[ok] = blend_scores(u, rec.V[cols[ok]], pop[cols[ok]])
    return out


def build_calibration(csv_path, rec, k=10, unwanted_max=2, user_vecs=None):
    """Top-k scores and per-user risk weights. user_vecs=None: trained user
    embeddings (Food.com users); otherwise fold-in vectors (students of the app)."""
    df = pd.read_csv(csv_path, usecols=["user_id", "recipe_id", "rating"])
    all_s, all_w, n = [], [], 0
    pop = pop01(np.expm1(rec.pop))          # rec.pop = log1p(popularity)
    for uid, g in df.groupby("user_id"):
        if user_vecs is None:
            if not rec.knows_user(uid):
                continue
            s, _ = rec.score_items(uid, g["recipe_id"].to_numpy())
        else:
            if uid not in user_vecs:
                continue
            s = foldin_scores(rec, user_vecs[uid], g["recipe_id"].to_numpy(), pop)
        h = g["rating"].to_numpy() <= unwanted_max
        ok = np.isfinite(s)
        s, h = s[ok], h[ok]
        if len(s) == 0:
            continue
        top = np.argsort(-s)[:k]
        all_s.append(s[top])
        all_w.append(h[top] / len(top))
        n += 1
    return np.concatenate(all_s), np.concatenate(all_w), n


def calibrate(scores, w, n, alpha):
    """Smallest threshold t with (n/(n+1)) R(t) + 1/(n+1) <= alpha (Theorem 4.1, B = 1)."""
    order = np.argsort(-scores)
    s_sorted = scores[order]
    risk = np.cumsum(w[order]) / n
    bound = ((n + 1) * alpha - 1) / n
    if risk[-1] <= bound:
        return -np.inf
    j = np.searchsorted(risk, bound, side="right") - 1
    if j < 0:
        return np.inf
    return float(s_sorted[j])


def main():
    p = argparse.ArgumentParser(description="Conformal calibration of the replacement threshold")
    p.add_argument("--emb", default="models/embeddings.npz")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--unwanted-max", type=int, default=2)
    p.add_argument("--foldin", action="store_true",
                   help="calibrate with fold-in vectors and the blended score of the API")
    p.add_argument("--out", default="configs/risk_threshold.json")
    a = p.parse_args()

    rec = Recommender(a.emb)
    vecs = foldin_vectors(rec, [f"{a.data_dir}/train.csv"]) if a.foldin else None
    S, W, n = build_calibration(f"{a.data_dir}/val.csv", rec, a.k, a.unwanted_max, vecs)
    t = calibrate(S, W, n, a.alpha)
    bound = ((n + 1) * a.alpha - 1) / n
    print(f"mode           : {'fold-in + blend (app)' if a.foldin else 'trained embeddings'}")
    print(f"calibration n  : {n} users ({len(S)} meals)")
    print(f"base risk      : {W.sum() / n:.4f}")
    print(f"bound          : {bound:.4f}  (alpha={a.alpha})")
    print(f"threshold t    : {t:.4f}")
    print(f"meals replaced : {(S < t).mean():.1%}")
    with open(a.out, "w") as f:
        json.dump({"alpha": a.alpha, "threshold": t, "n": n, "k": a.k,
                   "unwanted_max": a.unwanted_max,
                   "mode": "foldin_blend" if a.foldin else "trained"}, f, indent=2)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
