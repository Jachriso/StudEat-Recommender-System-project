"""
StudEat - PL: shared evaluation harness (NDCG@10, Recall@20) + baselines.

A single evaluation code path for ALL models (Random, Most-Popular,
LightGCN, NLGCL): a model is a function `score_fn(users) -> scores`
returning a matrix (batch_users x n_items).

Protocol (same as src/train.py):
  - an interaction is "positive" if rating >= pos_threshold (4 by default);
  - user/item indices are built from the TRAINING split ONLY (no leakage);
    users/items unseen in training are ignored;
  - already-seen items are excluded from the ranking (masks);
  - full-catalogue ranking, ties broken by item index.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp


# ----------------------------------------------------------------------------
# Loading the splits
# ----------------------------------------------------------------------------
def load_splits(data_dir="data/processed", user_col="user_id", item_col="recipe_id",
                rating_col="rating", pos_threshold=4):
    """Read train/val/test and return a dict of sparse matrices (users x items).

    Keys: train_pos, train_all, val_pos, val_all, test_pos, n_users, n_items,
          user_ids, item_ids, stats (number of kept/ignored interactions).
    """
    cols = [user_col, item_col, rating_col]
    raw = {s: pd.read_csv(f"{data_dir}/{s}.csv", usecols=cols)
           for s in ("train", "val", "test")}

    def positives(df):
        return df[df[rating_col] >= pos_threshold][[user_col, item_col]].drop_duplicates()

    # indices built from TRAINING positives only (same as train.py)
    tr_pos = positives(raw["train"])
    user_ids = np.sort(tr_pos[user_col].unique())
    item_ids = np.sort(tr_pos[item_col].unique())
    u2i = pd.Series(np.arange(len(user_ids)), index=user_ids)
    i2i = pd.Series(np.arange(len(item_ids)), index=item_ids)
    n_users, n_items = len(user_ids), len(item_ids)

    def to_csr(df):
        u, i = df[user_col].map(u2i), df[item_col].map(i2i)
        keep = u.notna() & i.notna()
        m = sp.csr_matrix(
            (np.ones(int(keep.sum()), dtype=np.float32),
             (u[keep].astype(np.int64), i[keep].astype(np.int64))),
            shape=(n_users, n_items))
        m.data[:] = 1.0                      # possible duplicates -> 1
        return m, int(keep.sum()), int((~keep).sum())

    out = {"n_users": n_users, "n_items": n_items,
           "user_ids": user_ids, "item_ids": item_ids, "stats": {}}
    out["train_pos"], _, _ = to_csr(tr_pos)
    # ALL training interactions (including ratings below threshold), for the mask
    out["train_all"], _, _ = to_csr(raw["train"][[user_col, item_col]].drop_duplicates())
    # ALL validation interactions, to mask them on the test split (--mask all)
    out["val_all"], _, _ = to_csr(raw["val"][[user_col, item_col]].drop_duplicates())
    for s in ("val", "test"):
        out[f"{s}_pos"], kept, dropped = to_csr(positives(raw[s]))
        out["stats"][s] = {"kept": kept, "dropped": dropped}
    return out


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def _top_k(scores, k):
    """Indices of the k highest scores per row (ties: smallest index first).

    A stable sort keeps tied items in index order, so the result is fully
    deterministic, even when ties straddle the k-th position (Most-Popular).
    """
    k = min(k, scores.shape[1])
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def evaluate_scores(score_fn, truth, masks, k_ndcg=10, k_rec=20, chunk=1024):
    """Mean NDCG@k_ndcg and Recall@k_rec over users with at least one positive.

    score_fn: function(users: np.ndarray) -> np.ndarray (len(users), n_items)
    truth   : csr of the items to retrieve (val or test)
    masks   : list of csr of items to exclude from the ranking (already seen)
    Returns (ndcg, recall, n_evaluated_users).
    """
    users = np.where(np.diff(truth.indptr) > 0)[0]
    k_max = max(k_ndcg, k_rec)
    discount = 1.0 / np.log2(np.arange(2, k_ndcg + 2))
    idcg = np.cumsum(discount)                       # IDCG for 1..k_ndcg relevant items

    ndcgs, recalls = [], []
    for s in range(0, len(users), chunk):
        ub = users[s:s + chunk]
        scores = np.array(score_fn(ub), dtype=np.float64, copy=True)
        for m in masks:
            scores[m[ub].toarray() > 0] = -np.inf
        rel = truth[ub].toarray() > 0
        top = _top_k(scores, k_max)
        hits = np.take_along_axis(rel, top, axis=1).astype(np.float64)
        n_rel = rel.sum(axis=1)

        recalls.append(hits[:, :k_rec].sum(axis=1) / n_rel)
        top_ndcg = hits[:, :k_ndcg]
        dcg = (top_ndcg * discount[:top_ndcg.shape[1]]).sum(axis=1)
        ndcgs.append(dcg / idcg[np.minimum(n_rel, k_ndcg) - 1])
    return float(np.concatenate(ndcgs).mean()), float(np.concatenate(recalls).mean()), len(users)


# ----------------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------------
def most_popular(train_pos):
    """Item score = number of positives in TRAINING (same score for every user)."""
    pop = np.asarray(train_pos.sum(axis=0)).ravel().astype(np.float64)
    return lambda users: np.broadcast_to(pop, (len(users), pop.size))


def random_scores(n_items, seed=0):
    """Random ranking, drawn independently for each user."""
    rng = np.random.default_rng(seed)
    return lambda users: rng.random((len(users), n_items))


def embedding_scores(U, V):
    """For an embedding model (LightGCN / NLGCL): score = U[u] . V[i]."""
    U, V = np.asarray(U), np.asarray(V)
    return lambda users: U[users] @ V.T