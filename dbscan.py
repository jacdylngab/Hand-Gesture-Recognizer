import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# ================================
# 1) Load data
# ================================

df = pd.read_csv("features_all.csv")

# Keep the top 30% activity for each gesture
top_percent = 0.3

def keep_top30(group):
    n_keep = int(len(group) * top_percent)
    return group.sort_values("activity", ascending=False).iloc[:n_keep]

# Keep only the top 30%
results = df.groupby("src", group_keys=False).apply(keep_top30)

X = results.drop(columns=["src"]).to_numpy()

# =========================
# SCALE → DBSCAN
# =========================

# DBSCAN params (tune eps for your data; try 0.5–2.0)
EPS = 2.0
MINPTS = 100
RANDOM_STATE = 42

scaler = StandardScaler().fit(X)
Xz = scaler.transform(X)

db = DBSCAN(eps=EPS, min_samples=MINPTS, n_jobs=-1)
labels = db.fit_predict(Xz)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
noise_frac = float(np.mean(labels == -1))
print(f"DBSCAN → clusters={n_clusters}, noise={noise_frac:.1%}")

labs = set(labels); labs.discard(-1)
if len(labs) > 1:
    s = silhouette_score(Xz[labels != -1], labels[labels != -1])
    print(f"Silhouette (non-noise): {s:.3f}")

# =========================
# VISUALS
# =========================
# PCA(2) map
pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(Xz)
X2 = pca.transform(Xz)
var2 = pca.explained_variance_ratio_.sum()

plt.figure(figsize=(8,6))
uniq = sorted(set(labels))
for lab in uniq:
    m = (labels == lab)
    color = 'black' if lab == -1 else None
    name  = 'Noise' if lab == -1 else f'C{lab}'
    plt.scatter(X2[m,0], X2[m,1], s=16, c=color, alpha=0.85, label=name)
plt.title(f"DBSCAN — PCA(2) var≈{var2:.2%} | eps={EPS}, min={MINPTS}")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.legend(ncol=2, fontsize=8, frameon=True); plt.grid(True, lw=0.3)
plt.tight_layout()
plt.savefig("DBSCAN Plot")
plt.show()
