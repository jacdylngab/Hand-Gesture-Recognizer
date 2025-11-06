# This script is for using PCA to reduce dimensionality and
# choosing how many principal components to keep

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

names = [label[:-4] for label in results["src"].unique()]
label_to_int = {name : i for i, name in enumerate(names)}
labels = np.array([label_to_int[l[:-4]] for l in results["src"]])

# ================================
# 2) Standardize + PCA 
# ================================

# Standardizes all features -> mean 0, variance 1.
X_scaled = StandardScaler().fit_transform(X)
pca_full = PCA().fit(X_scaled)                  # Full PCA fit. Computes principal components and their variances
X_all = pca_full.transform(X_scaled)            # Full scores. Projects all data onto these new axes (principal components)
exp_var = pca_full.explained_variance_ratio_    # Gives the fraction of the total variance explained by each Principal Component

# ============================================
# 3) Scree Plot (explained variance)
# ============================================

plt.figure(figsize=(9,4))
idx = np.arange(1, len(exp_var)+1)
plt.bar(idx, exp_var, alpha=0.6, label="Explained variance per PC")
plt.plot(idx, np.cumsum(exp_var), marker='o', linewidth=2, label="Cumulative")
plt.axhline(0.95, ls='--', color='r', alpha=0.6, label='95%')
plt.xlabel("Principal component index")
plt.ylabel("Explained variance ratio")
plt.title("Hand gesture — PCA Scree & Cumulative Variance")
plt.legend()
plt.tight_layout()
plt.savefig("PCA Scree Plot")
plt.show()

print(f"Components needed for 95% variance: {np.argmax(np.cumsum(exp_var) >= 0.95) + 1}")

# ============================================
# 4) 2D PCA Projection for visual intuition
# ============================================

pca2 = PCA(n_components=2)
# fit_transform both fits PCA on the training data and transforms it
# This means every data point is now represented by only two coordinates (its projection onto PC1 and PC2).
X_pca2 = pca2.fit_transform(X_scaled)

plt.figure(figsize=(7,5))
scatter = plt.scatter(X_pca2[:, 0], X_pca2[:, 1], c=labels, cmap="tab10", alpha=0.6)


# Add legend
handles, _ = scatter.legend_elements()
plt.legend(handles, [f"{names[i]}" for i in range(len(set(labels)))],
           title="Motions")

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA 2D Projection (Unlabeled Hand Gesture Data)")
plt.grid(True, linewidth=0.3)
plt.tight_layout()
plt.savefig("PCA 2d Projection")
plt.show()

# ============================================
# 5) 3D PCA Projection for visual intuition
# ============================================

pca3 = PCA(n_components=3)
X_pca3 = pca3.fit_transform(X_scaled)

fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca3[:, 0], X_pca3[:, 1], X_pca3[:, 2], c=labels, cmap="tab10", alpha=0.7)

# Add legend
handles, _ = scatter.legend_elements()
plt.legend(handles, [f"{names[i]}" for i in range(len(set(labels)))],
           title="Motions")

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
name = "PCA 3D Projection"
ax.set_title("3D PCA Projection")
#plt.savefig(name)
plt.show()