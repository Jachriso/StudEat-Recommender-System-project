"""
StudEat - MLE-Core : entrainement et evaluation de NLGCL / LightGCN.

Usage (depuis la racine du projet) :

  # NLGCL
  python src/train.py --epochs 30

  # Baseline LightGCN (meme code, loss contrastive desactivee)
  python src/train.py --epochs 30 --lightgcn --out models/lightgcn.pt

Fichiers attendus dans data/processed/ : train.csv, val.csv, test.csv
avec une colonne utilisateur, une colonne recette et (optionnel) une note.
Adapte --user-col / --item-col / --rating-col si les noms different.

Choix a valider avec l'equipe :
  - un "positif" = note >= --pos-threshold (4 par defaut) ;
  - les users/items absents du train sont ignores a l'evaluation (cold start non evalue ici).
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from model_nlgcl import NLGCL, build_norm_adj


# ----------------------------------------------------------------------------
# Donnees
# ----------------------------------------------------------------------------
def read_split(path, ucol, icol, rcol, thr):
    """Lit un CSV et ne garde que les interactions positives (note >= thr)."""
    cols = [ucol, icol] + ([rcol] if rcol else [])
    df = pd.read_csv(path, usecols=cols)
    if rcol:
        df = df[df[rcol] >= thr]
    return df[[ucol, icol]].drop_duplicates()


def map_split(df, u2i, i2i, ucol, icol):
    """Convertit les ids en indices 0..n-1 ; ignore les users/items inconnus du train."""
    u = df[ucol].map(u2i)
    i = df[icol].map(i2i)
    keep = u.notna() & i.notna()
    return (u[keep].astype(np.int64).values,
            i[keep].astype(np.int64).values,
            int((~keep).sum()))


def to_csr(u, i, n_users, n_items):
    return sp.csr_matrix((np.ones(len(u), dtype=np.float32), (u, i)),
                         shape=(n_users, n_items))


# ----------------------------------------------------------------------------
# Echantillonnage des negatifs (item non vu par l'utilisateur dans le train)
# ----------------------------------------------------------------------------
class NegSampler:
    def __init__(self, tr_u, tr_i, n_items, seed=0):
        self.n_items = n_items
        self.keys = np.sort(tr_u.astype(np.int64) * n_items + tr_i)
        self.rng = np.random.default_rng(seed)

    def _seen(self, keys):
        pos = np.searchsorted(self.keys, keys)
        pos[pos == len(self.keys)] = len(self.keys) - 1
        return self.keys[pos] == keys

    def sample(self, users):
        neg = self.rng.integers(0, self.n_items, len(users))
        for _ in range(5):                      # quelques re-tirages suffisent
            bad = self._seen(users.astype(np.int64) * self.n_items + neg)
            if not bad.any():
                break
            neg[bad] = self.rng.integers(0, self.n_items, int(bad.sum()))
        return neg


# ----------------------------------------------------------------------------
# Evaluation : NDCG@10 et Recall@20 (classement complet du catalogue)
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, truth, masks, device, chunk=1024, k_ndcg=10, k_rec=20):
    """truth : csr (users x items) des items a retrouver.
    masks : liste de csr dont les items sont exclus du classement (deja vus)."""
    model.eval()
    U, V = model.final_embeddings()
    users = np.where(np.diff(truth.indptr) > 0)[0]       # users ayant du test
    k_max = min(max(k_ndcg, k_rec), V.size(0))

    discount = 1.0 / torch.log2(torch.arange(2, k_ndcg + 2, device=device).float())
    idcg_table = torch.cumsum(discount, dim=0)            # IDCG pour 1..k_ndcg hits

    ndcgs, recalls = [], []
    for s in range(0, len(users), chunk):
        ub = users[s:s + chunk]
        scores = U[torch.as_tensor(ub, device=device)] @ V.T
        for m in masks:
            seen = torch.as_tensor(m[ub].toarray() > 0, device=device)
            scores = scores.masked_fill(seen, -1e9)
        rel = torch.as_tensor(truth[ub].toarray() > 0, device=device)
        top = torch.topk(scores, k_max).indices
        hits = rel.gather(1, top).float()                  # (B, k_max)
        n_rel = rel.sum(1).float()

        recalls.append(hits[:, :k_rec].sum(1) / n_rel)
        h10 = hits[:, :k_ndcg]
        dcg = (h10 * discount[:h10.size(1)]).sum(1)
        ideal = torch.clamp(n_rel, max=k_ndcg).long() - 1
        ndcgs.append(dcg / idcg_table[ideal])
    model.train()
    return (torch.cat(ndcgs).mean().item(), torch.cat(recalls).mean().item(),
            len(users))

@torch.no_grad()
def val_bpr_loss(model, va_u, va_i, sampler, device, chunk=10000):
    """BPR moyenne sur les paires de validation."""
    model.eval()
    U, V = model.final_embeddings()
    va_n = sampler.sample(va_u)       # un négatif par paire val
    total = 0.0
    for s in range(0, len(va_u), chunk):
        u = U[torch.as_tensor(va_u[s:s + chunk], device=device)]
        p = V[torch.as_tensor(va_i[s:s + chunk], device=device)]   # embeddings des items positifs (attention : V, pas d'offset n_users)
        n = V[torch.as_tensor(va_n[s:s + chunk], device=device)]   # embeddings des items négatifs
        # score = produit scalaire user·item ; BPR = -logsigmoid(score_pos - score_neg)
        total += (-F.logsigmoid((u * p).sum(dim=1) - (u * n).sum(dim=1))).sum().item()   # SOMME sur le paquet (pas la moyenne), en float Python
    model.train()
    return total / len(va_u)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--user-col", default="user_id")
    p.add_argument("--item-col", default="recipe_id")
    p.add_argument("--rating-col", default="rating",
                   help="colonne de note ('' pour ne pas filtrer)")
    p.add_argument("--pos-threshold", type=float, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--tau", type=float, default=0.2)
    p.add_argument("--lambda-cl", type=float, default=0.1)
    p.add_argument("--lightgcn", action="store_true",
                   help="baseline LightGCN (lambda_cl = 0)")
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--patience", type=int, default=5,
                   help="arret si NDCG@10 val ne s'ameliore pas pendant N evaluations")
    p.add_argument("--out", default="models/nlgcl.pt")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lam = 0.0 if args.lightgcn else args.lambda_cl
    print(f"device={device}  modele={'LightGCN' if lam == 0 else 'NLGCL'}  lambda_cl={lam}")

    # --- 1. donnees ---------------------------------------------------------
    d = args.data_dir
    ucol, icol, rcol = args.user_col, args.item_col, args.rating_col
    train = read_split(f"{d}/train.csv", ucol, icol, rcol, args.pos_threshold)
    val = read_split(f"{d}/val.csv", ucol, icol, rcol, args.pos_threshold)
    test_path = f"{d}/test.csv"
    test = (read_split(test_path, ucol, icol, rcol, args.pos_threshold)
            if os.path.exists(test_path) else None)

    # indices contigus construits UNIQUEMENT sur le train (pas de fuite)
    user_ids = np.sort(train[ucol].unique())
    item_ids = np.sort(train[icol].unique())
    n_users, n_items = len(user_ids), len(item_ids)
    u2i = pd.Series(np.arange(n_users), index=user_ids)
    i2i = pd.Series(np.arange(n_items), index=item_ids)

    tr_u, tr_i, _ = map_split(train, u2i, i2i, ucol, icol)
    va_u, va_i, va_drop = map_split(val, u2i, i2i, ucol, icol)
    print(f"train: {len(tr_u)} interactions, {n_users} users, {n_items} items")
    print(f"val  : {len(va_u)} interactions gardees, {va_drop} ignorees (user/item inconnu du train)")
    if test is not None:
        te_u, te_i, te_drop = map_split(test, u2i, i2i, ucol, icol)
        print(f"test : {len(te_u)} interactions gardees, {te_drop} ignorees")

    train_csr = to_csr(tr_u, tr_i, n_users, n_items)
    val_csr = to_csr(va_u, va_i, n_users, n_items)

    # --- 2. modele ----------------------------------------------------------
    adj = build_norm_adj(tr_u, tr_i, n_users, n_items)
    model = NLGCL(n_users, n_items, adj, dim=args.dim, n_layers=args.layers,
                  tau=args.tau, lambda_cl=lam).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sampler = NegSampler(tr_u, tr_i, n_items, seed=args.seed)
    val_sampler = NegSampler(tr_u, tr_i, n_items, seed=args.seed + 1)

    # --- 3. entrainement ----------------------------------------------------
    best_ndcg, best_epoch, bad_evals = -1.0, 0, 0
    history =[]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    def save_ckpt():
        torch.save({"state_dict": model.state_dict(),
                    "user_ids": user_ids, "item_ids": item_ids,
                    "config": vars(args), "lambda_cl": lam}, args.out)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        perm = np.random.permutation(len(tr_u))
        tot, n_b = 0.0, 0
        sum_bpr, sum_cl = 0.0, 0.0 

        for s in range(0, len(perm), args.batch_size):
            idx = perm[s:s + args.batch_size]
            if len(idx) < 2:
                continue
            u = torch.as_tensor(tr_u[idx], device=device)
            pos = torch.as_tensor(tr_i[idx], device=device)
            neg = torch.as_tensor(sampler.sample(tr_u[idx]), device=device)
            loss, parts = model(u, pos, neg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            n_b += 1
            sum_bpr += parts["bpr"]
            sum_cl += parts["cl"]
        n_b = max(n_b, 1)
        vloss = val_bpr_loss(model, va_u, va_i, val_sampler, device)
        msg = f"epoch {epoch:3d}  loss={tot / n_b:.4f}  bpr={sum_bpr / n_b:.4f}  cl={sum_cl / n_b:.4f}  val_bpr={vloss:.4f}  ({time.time() - t0:.0f}s)"

        ndcg, rec = None, None

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            ndcg, rec, n_eval = evaluate(model, val_csr, [train_csr], device)
            msg += f"  | val NDCG@10={ndcg:.4f}  Recall@20={rec:.4f}  ({n_eval} users)"
            if ndcg > best_ndcg:
                best_ndcg, best_epoch, bad_evals = ndcg, epoch, 0
                save_ckpt()
            else:
                bad_evals += 1
        print(msg)

        history.append({"epoch": epoch, "train_loss": tot / n_b, "train_bpr": sum_bpr / n_b, "train_cl": sum_cl / n_b,
                        "val_bpr": vloss, "val_ndcg10": ndcg, "val_recall20": rec})
        pd.DataFrame(history).to_csv(args.out.replace(".pt", "_history.csv"), index=False)

        if bad_evals >= args.patience:
            print(f"arret anticipe (meilleur epoch = {best_epoch})")
            break

    # --- 4. evaluation finale sur le test (une seule fois !) ----------------
    print(f"\nmeilleur val NDCG@10 = {best_ndcg:.4f} (epoch {best_epoch})")
    if test is not None and best_epoch > 0:
        ckpt = torch.load(args.out, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        test_csr = to_csr(te_u, te_i, n_users, n_items)
        ndcg, rec, n_eval = evaluate(model, test_csr, [train_csr, val_csr], device)
        print(f"TEST  NDCG@10={ndcg:.4f}  Recall@20={rec:.4f}  ({n_eval} users)")
    print(f"checkpoint : {args.out}  (ne pas le committer dans git)")


if __name__ == "__main__":
    main()
