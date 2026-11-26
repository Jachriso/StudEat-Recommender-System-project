"""StudEat - MLE-Core: export a checkpoint to a light .npz file (numpy).

The API then needs neither PyTorch nor the graph to score meals.

Usage (from the project root):
  python src/export_embeddings.py --ckpt models/nlgcl.pt --out models/embeddings.npz
"""
import argparse
import os

import numpy as np
import torch

from model_nlgcl import NLGCL, build_norm_adj
from train import map_split, read_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="models/embeddings.npz")
    p.add_argument("--data-dir", default=None, help="default: the one stored in the checkpoint")
    a = p.parse_args()

    import pandas as pd
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    d = a.data_dir or cfg["data_dir"]
    ucol, icol, rcol = cfg["user_col"], cfg["item_col"], cfg["rating_col"]

    # same construction as train.py: indices built from the training split only
    train = read_split(f"{d}/train.csv", ucol, icol, rcol, cfg["pos_threshold"])
    user_ids, item_ids = ck["user_ids"], ck["item_ids"]
    u2i = pd.Series(np.arange(len(user_ids)), index=user_ids)
    i2i = pd.Series(np.arange(len(item_ids)), index=item_ids)
    tr_u, tr_i, _ = map_split(train, u2i, i2i, ucol, icol)

    adj = build_norm_adj(tr_u, tr_i, len(user_ids), len(item_ids))
    model = NLGCL(len(user_ids), len(item_ids), adj, dim=cfg["dim"], n_layers=cfg["layers"],
                  tau=cfg["tau"], lambda_cl=ck["lambda_cl"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    with torch.no_grad():
        U, V = model.final_embeddings()

    pop = np.bincount(tr_i, minlength=len(item_ids)).astype(np.float32)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, U=U.numpy(), V=V.numpy(), user_ids=np.asarray(user_ids),
                        item_ids=np.asarray(item_ids), popularity=pop)
    print(f"exported: {a.out}  ({len(user_ids)} users, {len(item_ids)} items)")


if __name__ == "__main__":
    main()
