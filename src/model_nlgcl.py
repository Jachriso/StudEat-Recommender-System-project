"""
StudEat - Modèle NLGCL (LightGCN + loss contrastive entre couches).

LightGCN : propagation linéaire sur le graphe user-item normalisé.
NLGCL   : loss contrastive InfoNCE entre couches consécutives (style SGL),
          activée quand lambda_cl > 0. Avec lambda_cl = 0, c'est exactement
          LightGCN (baseline).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp


def build_norm_adj(tr_u, tr_i, n_users, n_items):
    """Matrice d'adjacence normalisée symétriquement (LightGCN).
    A_norm = D^{-1/2} A D^{-1/2}, avec A = [[0, R], [R^T, 0]].
    """
    n_nodes = n_users + n_items
    rows = torch.from_numpy(tr_u.astype('int64'))
    cols = torch.from_numpy(tr_i.astype('int64')) + n_users
    idx = torch.cat([torch.stack([rows, cols], dim=0),
                     torch.stack([cols, rows], dim=0)], dim=1)
    vals = torch.ones(idx.size(1), dtype=torch.float32)

    A = torch.sparse_coo_tensor(idx, vals, (n_nodes, n_nodes)).coalesce()

    # degrés
    deg = torch.sparse.sum(A, dim=1).to_dense()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # D^{-1/2} A D^{-1/2}
    row, col = A.indices()
    v = A.values() * deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return torch.sparse_coo_tensor(A.indices(), v, A.shape).coalesce()


class NLGCL(nn.Module):
    def __init__(self, n_users, n_items, adj, dim=64, n_layers=3,
                 tau=0.2, lambda_cl=0.1, reg=1e-2, mask_dup=False):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.tau = tau
        self.lambda_cl = lambda_cl
        self.mask_dup = mask_dup        # ignorer les doublons du batch comme negatifs
        self.reg = reg                  # poids de la regularisation L2 des embeddings de depart

        self.emb = nn.Embedding(n_users + n_items, dim)
        nn.init.normal_(self.emb.weight, std=0.1)

        # adj stockée comme buffer sparse
        self.register_buffer('adj_idx', adj.indices())
        self.register_buffer('adj_val', adj.values())
        self._adj_shape = adj.shape

    @property
    def adj(self):
        return torch.sparse_coo_tensor(
            self.adj_idx, self.adj_val, self._adj_shape
        ).coalesce()

    def propagate(self):
        """Liste des embeddings de chaque couche [E0, E1, ..., EL]."""
        adj = self.adj
        layers = [self.emb.weight]
        for _ in range(self.n_layers):
            layers.append(torch.sparse.mm(adj, layers[-1]))
        return layers

    def final_embeddings(self):
        """Embeddings finaux = moyenne des couches 0..L (pas de normalisation L2 :
        elle faisait chuter le NDCG@10 de 0.023 a 0.004)."""
        E = torch.stack(self.propagate(), dim=0).mean(dim=0)
        return E[: self.n_users], E[self.n_users:]

    def contrastive_loss(self, layers, users, pos_items):
        """InfoNCE entre couches consécutives (NLGCL).

        Pour chaque couche l (0..L-1) :
          - vue 1 : embeddings de la couche l
          - vue 2 : embeddings de la couche l+1
          - positifs : voisins (user u de la couche l avec l'item i qu'il a aimé
            a la couche l+1, et inversement) ; dans le batch, users[k] et
            pos_items[k] forment une vraie interaction
          - négatifs : autres éléments du batch
        """
        if self.lambda_cl == 0.0:
            return torch.tensor(0.0, device=self.emb.weight.device)

        loss = 0.0
        n_layers_used = 0
        for l in range(len(layers) - 1):
            z1 = layers[l]
            z2 = layers[l + 1]

            # users : batch
            u1 = F.normalize(z1[users], p=2, dim=1)
            u2 = F.normalize(z2[users], p=2, dim=1)

            # items : batch
            i1 = F.normalize(z1[self.n_users + pos_items], p=2, dim=1)
            i2 = F.normalize(z2[self.n_users + pos_items], p=2, dim=1)

            loss += self._info_nce(u1, i2, pos_items)   # colonnes = items du batch
            loss += self._info_nce(i1, u2, users)       # colonnes = users du batch
            n_layers_used += 1

        if n_layers_used == 0:
            return torch.tensor(0.0, device=self.emb.weight.device)
        return loss / (2.0 * n_layers_used)

    def _info_nce(self, z1, z2, col_ids=None):
        """InfoNCE entre deux vues d'un même batch, positifs = paires (i, i).

        Un item populaire apparait plusieurs fois dans un batch de 2048 : ses copies
        seraient comptees comme des negatifs alors que c'est le meme item que le positif.
        Avec mask_dup, on les retire du denominateur (col_ids = identifiant des colonnes).
        """
        sim = (z1 @ z2.T) / self.tau
        if self.mask_dup and col_ids is not None:
            same = col_ids.unsqueeze(0) == col_ids.unsqueeze(1)          # meme id que la colonne k
            same = same & ~torch.eye(len(col_ids), dtype=torch.bool, device=sim.device)
            sim = sim.masked_fill(same, float("-inf"))
        # positifs sur la diagonale
        pos = torch.diag(sim)
        # logsumexp sur toute la ligne (négatifs = autres éléments du batch)
        neg = torch.logsumexp(sim, dim=1)
        return -(pos - neg).mean()

    def forward(self, users, pos_items, neg_items):
        """Retourne (loss_totale, dict des composantes) pour un batch (u, i+, i-).

        Le dict {"bpr", "l2", "cl"} (floats) est lu par train.py pour les logs.
        """
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
