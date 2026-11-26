"""StudEat - MLE-Ops: ranking of the meal catalogue for the API.

- Student with ratings >= 4 in the app: user vector = mean of the NLGCL
  embeddings of the meals they liked (fold-in, no retraining needed), and
  score = 0.7 x cosine + 0.3 x popularity (src/scoring.py, shared with the
  calibration so the risk threshold is on the same scale).
- Student without ratings (cold start): popularity in Food.com.
- Risk control (configs/risk_threshold.json, calibrated with
  `python src/risk_control.py --foldin`): a meal whose score is below the
  threshold t is "risky" and goes after the meals the student already liked
  (Replace strategy of risk_control.py).
"""
import json
from pathlib import Path

import numpy as np

from src.recommender import Recommender
from src.scoring import blend_scores, pop01

ROOT = Path(__file__).resolve().parent.parent
EMB_PATH = ROOT / "models" / "embeddings.npz"
RISK_PATH = ROOT / "configs" / "risk_threshold.json"


class StudEatModel:
    def __init__(self, emb_path=EMB_PATH, risk_path=RISK_PATH):
        self.rec = Recommender(emb_path)
        self.pop = pop01(np.expm1(self.rec.pop))      # rec.pop = log1p(popularity)
        self.threshold = -np.inf                       # no risk control by default
        if Path(risk_path).exists():
            with open(risk_path) as f:
                cfg = json.load(f)
            if cfg.get("mode") == "foldin_blend":
                self.threshold = cfg["threshold"]
            else:   # a threshold from another scale would filter nothing (or everything)
                print(f"[StudEatModel] {risk_path}: mode {cfg.get('mode')!r} is not "
                      "'foldin_blend', risk control disabled. Re-run "
                      "`python src/risk_control.py --foldin`.")

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
            ok = cols >= 0
            raw = np.full(len(ids), -np.inf)
            raw[ok] = blend_scores(u, self.rec.V[cols[ok]], self.pop[cols[ok]])
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
