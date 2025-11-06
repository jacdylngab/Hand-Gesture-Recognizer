import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
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
eps_values = [0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
min_points = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

RANDOM_STATE = 42

scaler = StandardScaler().fit(X)
Xz = scaler.transform(X)

best_silhoutte_score = float('-inf')
best_params = {}

for eps in eps_values:
    for min_point in min_points:
        print(f"eps: {eps}, min_point: {min_point}")
        db = DBSCAN(eps=eps, min_samples=min_point, n_jobs=-1)
        labels = db.fit_predict(Xz)
        labs = set(labels); labs.discard(-1)
        if len(labs) > 1:
            s = silhouette_score(Xz[labels != -1], labels[labels != -1])
            print(f"Silhouette (non-noise): {s:.3f}")
            if s > best_silhoutte_score:
                best_silhoutte_score = s
                best_params = {'eps': eps, 'min_point': min_point}

print(f"Best Silhoutte Score: {best_silhoutte_score}")
print(f"Best Params: {best_params}")