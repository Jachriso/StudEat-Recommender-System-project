"""
StudEat - MLE-Core : squelette NLGCL (semaine 2).

NLGCL = LightGCN + une loss contrastive qui utilise les vues "naturelles"
entre couches consecutives du GNN (pas d'augmentation de graphe).

Avec lambda_cl = 0 on obtient exactement LightGCN (baseline / fallback).

ATTENTION : la loss contrastive ci-dessous suit la description du plan de
projet (un noeud est rapproche de ses voisins a la couche suivante, et
eloigne des autres noeuds). Les details exacts (choix des positifs,
negatifs, normalisation) sont a verifier avec le papier et le code officiel
de NLGCL (Xu et al., RecSys 2025).
"""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_norm_adj(train_users, train_items, n_users, n_items):
    """Matrice d'adjacence bipartite normalisee D^-1/2 A D^-1/2 (comme LightGCN).

    A = [[0, R], [R^T, 0]] avec R la matrice d'interactions (users x items).
    ATTENTION : n'utiliser que les interactions du TRAIN (pas de fuite).
    """
    train_users = np.asarray(train_users)
    train_items = np.asarray(train_items)
    R = sp.coo_matrix(
        (np.ones(len(train_users), dtype=np.float32), (train_users, train_items)),
        shape=(n_users, n_items),
    )
    A = sp.bmat([[None, R], [R.T, None]], format="coo")
    deg = np.asarray(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.zeros_like(deg, dtype=np.float32)
    mask = deg > 0
    d_inv_sqrt[mask] = np.power(deg[mask], -0.5)
    D = sp.diags(d_inv_sqrt)
    A_norm = (D @ A @ D).tocoo()
    idx = torch.tensor(np.vstack([A_norm.row, A_norm.col]), dtype=torch.long)
    vals = torch.tensor(A_norm.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, vals, A_norm.shape).coalesce()


class NLGCL(nn.Module):
    def __init__(self, n_users, n_items, adj, dim=64, n_layers=3,
                 tau=0.2, lambda_cl=0.1, reg=1e-4):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.adj = adj                  # sparse (n_users+n_items)^2
        self.n_layers = n_layers
        self.tau = tau                  # temperature InfoNCE
        self.lambda_cl = lambda_cl      # poids de la loss contrastive (0 = LightGCN)
        self.reg = reg                  # poids de la regularisation L2
        # embeddings "ego" (couche 0) : users puis items dans la meme table
        self.emb = nn.Embedding(n_users + n_items, dim)
        nn.init.xavier_uniform_(self.emb.weight)

    def to(self, *args, **kwargs):
        # le tenseur creux n'est pas un parametre : on le deplace a la main
        out = super().to(*args, **kwargs)
        out.adj = out.adj.to(*args, **kwargs)
        return out

    def propagate(self):
        """Liste des embeddings de chaque couche [E0, E1, ..., EL],
        chacun de forme (n_users + n_items, dim). Pas de transformation
        lineaire ni de non-linearite (comme LightGCN)."""
        layers = [self.emb.weight]
        for _ in range(self.n_layers):
            layers.append(torch.sparse.mm(self.adj, layers[-1]))
        return layers

    def final_embeddings(self):
        """Embeddings finaux = moyenne des couches 0..L (LightGCN)."""
        E = torch.stack(self.propagate(), dim=0).mean(dim=0)
        return E[: self.n_users], E[self.n_users:]

    def contrastive_loss(self, layers, users, pos_items):
        """InfoNCE entre couches consecutives.

        Pour chaque couche l (0..L-1) :
          - positif de l'user u (couche l)  = item voisin i (couche l+1)
          - positif de l'item i (couche l)  = user voisin u (couche l+1)
          - negatifs = les autres elements du batch.
        Limite : si un meme user/item apparait plusieurs fois dans le batch,
        certains "negatifs" sont en realite des positifs (a ameliorer plus tard).
        """
        loss = 0.0
        items_idx = self.n_users + pos_items
        labels = torch.arange(users.size(0), device=users.device)
        for l in range(self.n_layers):
            u_l = F.normalize(layers[l][users], dim=1)
            i_l = F.normalize(layers[l][items_idx], dim=1)
            u_next = F.normalize(layers[l + 1][users], dim=1)
            i_next = F.normalize(layers[l + 1][items_idx], dim=1)
            for a, b in ((u_l, i_next), (i_l, u_next)):
                logits = a @ b.T / self.tau       # (B, B), diagonale = positifs
                loss = loss + F.cross_entropy(logits, labels)
        return loss / (2 * self.n_layers)

    def forward(self, users, pos_items, neg_items):
        """Retourne (loss_totale, dict des composantes) pour un batch (u, i+, i-)."""
        layers = self.propagate()
        E = torch.stack(layers, dim=0).mean(dim=0)
        u = E[users]
        p = E[self.n_users + pos_items]
        n = E[self.n_users + neg_items]

        # 1) BPR : l'item aime doit avoir un score > item non vu
        bpr = -F.logsigmoid((u * p).sum(dim=1) - (u * n).sum(dim=1)).mean()

        # 2) regularisation L2 sur les embeddings de depart (couche 0)
        E0 = self.emb.weight
        l2 = (E0[users].pow(2).sum() + E0[self.n_users + pos_items].pow(2).sum()
              + E0[self.n_users + neg_items].pow(2).sum()) / (2 * users.size(0))

        # 3) loss contrastive entre couches voisines (0 si LightGCN)
        if self.lambda_cl > 0:
            cl = self.contrastive_loss(layers, users, pos_items)
        else:
            cl = torch.zeros((), device=users.device)

        total = bpr + self.reg * l2 + self.lambda_cl * cl
        return total, {"bpr": bpr.item(), "l2": l2.item(), "cl": cl.item()}

    @torch.no_grad()
    def score_all_items(self, user_idx):
        """Scores d'un user (int ou tenseur de users) contre tous les items."""
        U, V = self.final_embeddings()
        u = U[user_idx]
        return u @ V.T
