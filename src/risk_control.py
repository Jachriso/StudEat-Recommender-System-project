import argparse
import json
import numpy as np
import pandas as pd
from recommender import Recommender


def build_calibration(csv_path, rec, k=10, unwanted_max=2):
    """Pour chaque étudiant connu du modèle : top-k de ses recettes notées.
    Retourne (scores, poids, n) avec poids = 1/k_u si la recette est indésirable, sinon 0."""
    df = pd.read_csv(csv_path, usecols=["user_id", "recipe_id", "rating"])
    all_s, all_w, n = [], [], 0
    for uid, g in df.groupby("user_id"):
        if not rec.knows_user(uid):
            continue
        s, _ = rec.score_items(uid, g["recipe_id"].to_numpy())
        h = g["rating"].to_numpy() <= unwanted_max
        ok = np.isfinite(s)                   # recettes inconnues du modèle : -inf
        s, h = s[ok], h[ok]
        if len(s) == 0:
            continue
        top = np.argsort(-s)[:k]              # la "box" de l'étudiant
        all_s.append(s[top])                  # les scores des recettes de la box
        all_w.append(h[top] / len(top))       # 1/k_u si indésirable, 0 sinon
        n += 1
    return np.concatenate(all_s), np.concatenate(all_w), n


def calibrate(scores, w, n, alpha):
    """Plus petit seuil t tel que (n/(n+1)) R(t) + 1/(n+1) <= alpha."""
    order = np.argsort(-scores)
    s_sorted = scores[order]
    risk = np.cumsum(w[order]) / n            # risk[j] = R(t) avec t = s_sorted[j]
    bound = ((n + 1) * alpha - 1) / n
    if risk[-1] <= bound:          # même sans filtre, on respecte déjà alpha
        return -np.inf
    j = np.searchsorted(risk, bound, side="right") - 1
    if j < 0:                      # même avec une seule recette gardée, c'est trop risqué
        return np.inf
    return float(s_sorted[j])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", default="models/embeddings.npz")
    p.add_argument("--calib", default="data/processed/val.csv")
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--unwanted-max", type=int, default=2)
    p.add_argument("--out", default="configs/risk_threshold.json")
    a = p.parse_args()

    rec = Recommender(a.emb)
    scores, w, n = build_calibration(a.calib, rec, a.k, a.unwanted_max)
    t = calibrate(scores, w, n, a.alpha)

    print(f"calibration users n = {n}  ({len(scores)} rated meals in boxes)")
    print(f"base risk (no filter) = {w.sum() / n:.4f}")
    print(f"bound = {((n + 1) * a.alpha - 1) / n:.4f}   threshold = {t:.4f}")
    print(f"meals replaced = {(scores < t).mean():.1%}")

    with open(a.out, "w") as f:
        json.dump({"alpha": a.alpha, "threshold": t, "n": n, "k": a.k,
                   "unwanted_max": a.unwanted_max}, f, indent=2)


if __name__ == "__main__":
    main()