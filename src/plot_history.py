import sys
import pandas as pd
import matplotlib.pyplot as plt

path = sys.argv[1]    # e.g. models/nlgcl_lam0.001_history.csv
h = pd.read_csv(path)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(h["epoch"], h["train_bpr"], marker="o", label="train BPR")
ax1.plot(h["epoch"], h["val_bpr"], marker="o", label="validation BPR")

ax1.set_xlabel("epoch"); ax1.set_ylabel("BPR loss")
ax1.set_title("Training vs validation loss")
ax1.legend()

ev = h.dropna(subset=["val_ndcg10"])       # epochs where NDCG was computed
ax2.plot(ev["epoch"], ev["val_ndcg10"], marker="o", color="tab:green")
ax2.set_xlabel("epoch"); ax2.set_ylabel("NDCG@10")
ax2.set_title("Validation NDCG@10")

fig.tight_layout()
fig.savefig(path.replace("_history.csv", "_curves.png"), dpi=150)