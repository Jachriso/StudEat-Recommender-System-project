"""StudEat - MLE-Ops: the ONE scoring function for students of the app.

Used by api/model.py (live boxes) AND by risk_control.py / risk_eval.py
(calibration), so the risk threshold is always on the same scale as the app.

score(u, v) = W_COS * cosine(u, v) + W_POP * pop01(v)
  u     : fold-in vector of the student (mean of the embeddings of liked meals)
  pop01 : log1p(popularity) rescaled to [0, 1] over the whole catalogue

Why not the plain dot product: item vector norms correlate with popularity
(r = 0.78), so the dot product mostly returns popular recipes. 
Measured by Hajar on validation with fold-in vectors (what the app uses):
  - dot product   : NDCG@10 0.0389, 8.7/10 meals in common with the popularity box
  - cosine only   : NDCG@10 0.0216
  - 0.7 / 0.3 blend: NDCG@10 0.0365 (~94% of the dot product), 7.2/10 in common
"""
import numpy as np

W_COS, W_POP = 0.7, 0.3


def pop01(popularity):
    """log1p(popularity) rescaled to [0, 1]."""
    p = np.log1p(np.asarray(popularity, dtype=np.float64))
    span = p.max() - p.min()
    return (p - p.min()) / span if span > 0 else np.zeros_like(p)


def blend_scores(U, V, pop):
    """Scores of users (rows of U, or one vector) for items (rows of V).
    pop: pop01 of the same items. Returns (n_users, n_items) or (n_items,)."""
    U = np.asarray(U, dtype=np.float64)
    Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    Un = U / np.maximum(np.linalg.norm(U, axis=-1, keepdims=True), 1e-12)
    return W_COS * (Un @ Vn.T) + W_POP * pop
