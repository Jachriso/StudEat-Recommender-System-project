"""StudEat - MLE-Core: model scores for the API (numpy only, no PyTorch).

  rec = Recommender("models/embeddings.npz")
  rec.score_items(user_id, item_ids)   -> (scores, source)   source = "model" | "popularity"

A student without history has no embedding in the graph: we then fall back on
popularity (cold start) instead of crashing. Hard filters (allergies, diet,
budget) are still applied by the API BEFORE; the model only ranks.
"""
import numpy as np


class Recommender:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.U, self.V = z["U"], z["V"]
        self.item_ids = z["item_ids"]
        self.pop = np.log1p(z["popularity"])
        self._user_row = {int(u): k for k, u in enumerate(z["user_ids"])}
        self._item_col = {int(i): k for k, i in enumerate(self.item_ids)}

    def knows_user(self, user_id):
        return int(user_id) in self._user_row

    def score_items(self, user_id, item_ids):
        """Scores of the requested items (Food.com ids). Items unknown to the model: -inf.
        Returns (np.array of scores, "model" or "popularity")."""
        cols = np.array([self._item_col.get(int(i), -1) for i in item_ids])
        ok = cols >= 0
        out = np.full(len(cols), -np.inf, dtype=np.float32)
        if self.knows_user(user_id):
            out[ok] = self.V[cols[ok]] @ self.U[self._user_row[int(user_id)]]
            return out, "model"
        out[ok] = self.pop[cols[ok]]
        return out, "popularity"

    def top_k(self, user_id, item_ids, k=10, exclude=()):
        """The k best among item_ids (excluding exclude). Returns (list of (id, score), source)."""
        ex = {int(e) for e in exclude}
        cand = [int(i) for i in item_ids if int(i) not in ex]
        scores, src = self.score_items(user_id, cand)
        order = np.argsort(-scores)[:k]
        return [(cand[j], float(scores[j])) for j in order if np.isfinite(scores[j])], src
