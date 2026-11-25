"""StudEat - MLE-Ops: ranking of the meal catalogue for the API.

- Student with ratings >= 4 in the app: user vector = mean of the NLGCL
  embeddings of the meals they liked (fold-in, no retraining needed).
- Student without ratings (cold start): popularity in Food.com.
- Risk control (configs/risk_threshold.json): a meal whose model score is
  below the calibrated threshold t is "risky" and goes after the meals the
  student already liked (Replace strategy of risk_control.py).
"""
import json
from pathlib import Path

import numpy as np

from src.recommender import Recommender

ROOT = Path(__file__).resolve().parent.parent
EMB_PATH = ROOT / "models" / "embeddings.npz"
RISK_PATH = ROOT / "configs" / "risk_threshold.json"


class StudEatModel:
    def __init__(self, emb_path=EMB_PATH, risk_path=RISK_PATH):
        self.rec = Recommender(emb_path)
        self.threshold = -np.inf                      # no risk control if no file
        if Path(risk_path).exists():
            with open(risk_path) as f:
                self.threshold = json.load(f)["threshold"]

    def _cols(self, ids):
        """Rows of V for Food.com ids (-1 if the model does not know the meal)."""
        return np.array([self.rec._item_col.get(int(i), -1) for i in ids])

    def user_vector(self, liked_ids):
        cols = self._cols(liked_ids)
        cols = cols[cols >= 0]
        return self.rec.V[cols].mean(axis=0) if len(cols) else None

    def rank(self, candidates, liked_ids):
        """Return (candidates sorted best first, source). Each meal gets
        score (in [0, 1], 1 = best), model_score and risky."""
        if not candidates:
            return [], "none"
        ids = [m["id"] for m in candidates]
        u = self.user_vector(liked_ids)
        if u is None:
            raw, _ = self.rec.score_items(-1, ids)    # unknown user -> popularity
            source, risky = "popularity", np.zeros(len(ids), dtype=bool)
        else:
            cols = self._cols(ids)
            raw = np.full(len(ids), -np.inf)
            raw[cols >= 0] = self.rec.V[cols[cols >= 0]] @ u
            source = "model"
            liked = np.isin(ids, list(liked_ids))
            risky = (raw < self.threshold) & ~liked   # liked meals are safe replacements
            raw = np.where(liked & (raw < self.threshold), self.threshold, raw)
        # sort: non-risky first, then by decreasing model score
        order = np.lexsort((-raw, risky))
        n = len(order)
        ranked = []
        for r, j in enumerate(order):
            ms = float(raw[j]) if np.isfinite(raw[j]) else None
            ranked.append({**candidates[j], "score": round(1 - r / n, 4),
                           "model_score": None if ms is None else round(ms, 4),
                           "risky": bool(risky[j])})
        return ranked, source
