# This script is for using PCA to reduce dimensionality and
# choosing how many principal components to keep

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import glob
from pathlib import Path 

# ================================
# 1) Load data
# ================================


# Path to your folder containing CSV files
current_directory = Path.cwd()
print(current_directory)

# Get all CSV files
csv_files = glob.glob(str(current_directory) + "/*.csv")

dfs = []
names = []

# Give each file a cluster lable
for i, file in enumerate(csv_files):
    df = pd.read_csv(file)
    df["motion_id"] = i # numeric label for color mapping
    dfs.append(df)
    names.append(Path(file).stem)  # store file name (without extension)

# Read and combine them
combined_df = pd.concat(dfs, ignore_index=True)

X = combined_df.drop(columns=["motion_id"]).to_numpy()
labels = combined_df["motion_id"].to_numpy()

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
ax.scatter(X_pca3[:, 0], X_pca3[:, 1], X_pca3[:, 2], c=labels, cmap="tab10", alpha=0.7)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
name = "PCA 3D Projection"
ax.set_title("3D PCA Projection")
#plt.savefig(name)
plt.show()
