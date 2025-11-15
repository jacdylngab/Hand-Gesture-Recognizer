import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches

# ================================
# 1) Load data
# ================================

df = pd.read_csv("motion_dataset.csv")

# =============================================================
# 2) Transform the dataset
# ==============================================================

# Transform the categorical class column into numerical digits
encoder = LabelEncoder()
y = encoder.fit_transform(df["motion"])

# ==============================================================
# 3) Train/Test Split
# ==============================================================

X = df.drop(columns=["motion", "t_start_ms", "t_end_ms"]).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ==============================================================
# 4) Standardization
# ==============================================================

# Standardizes all features -> mean 0, variance 1.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================================================
# 5) PCA
# ==============================================================

pca2 = PCA(n_components=2)
# fit_transform both fits PCA on the training data and transforms it
# This means every data point is now represented by only two coordinates (its projection onto PC1 and PC2).
X_train_pca2 = pca2.fit_transform(X_train_scaled)
X_test_pca2 = pca2.transform(X_test_scaled)


# ==============================================================
# 6) Random Forest Classifier
# ==============================================================

forest_pca = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    random_state=142,
    n_jobs=-1,
)

# Fit on the training data
forest_pca.fit(X_train_pca2, y_train)

# Make predictions on the training and the test set
yhat_tr = forest_pca.predict(X_train_pca2)
yhat_te = forest_pca.predict(X_test_pca2)


# ======================================================================================
# 7) Plot decision boundary
# ======================================================================================

# Plot decision boundaries
def plot_decision_boundary(model, X, y, title, ax, encoder=None):
    x_min, x_max = X[:, 0].min() - 0.6, X[:, 0].max() + 0.6
    y_min, y_max = X[:, 1].min() - 0.6, X[:, 1].max() + 0.6
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                         np.linspace(y_min, y_max, 400))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    cmap = plt.cm.RdYlBu
    ax.contourf(xx, yy, Z, alpha=0.25, cmap=cmap)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor="k", s=20)
    ax.set_title(title)
    ax.set_xlabel("x1"); ax.set_ylabel("y1")

    if encoder is not None:
        # Get all encoded classes and their names
        classes = encoder.classes_
        colors = [cmap(i / (len(classes) - 1)) for i in range(len(classes))]

        handles = [mpatches.Patch(color=colors[i], label=classes[i]) for i in range(len(classes))]
        ax.legend(handles=handles, title="Motions", bbox_to_anchor=(1.05, 1), loc='upper left')

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
plot_decision_boundary(forest_pca, X_train_pca2, y_train, "TRAIN", axes[0], encoder)
plot_decision_boundary(forest_pca, X_test_pca2,  y_test,  "TEST",  axes[1], encoder)
plt.show()