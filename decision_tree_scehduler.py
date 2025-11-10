import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

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

# ============================================================
# 4) Helper functions
# ============================================================

def report_line(tag, acc, f1m, f1w):
    """Nicely formatted printout of model performance metrics."""
    print(f"{tag:<30} | ACC: {acc:.3f} | F1-macro: {f1m:.3f} | F1-weighted: {f1w:.3f}")

def saving_results(criterion=None, max_depth=None, max_leaf_nodes=None, min_samples_split=None, ccp_alpha=None):
    """Save the best Decision Tree hyperparameters to a CSV file."""
    filename = Path("Best_Tree_Hyperparameters.csv")

    data = {
        "criterion"         : [criterion],
        "max_depth"         : [max_depth],
        "max_leaf_nodes"    : [max_leaf_nodes],
        "min_samples_split" : [min_samples_split],
        "ccp_alpha"         : [ccp_alpha]
    }

    df_new = pd.DataFrame(data)

    if filename.exists():
        df_new.to_csv(filename, index=False, mode='a', header=False)
    else:
        df_new.to_csv(filename, index=False, mode='w', header=True)

# ===============================================================
# 5) Define Parameter Grid
# ===============================================================
# The grid explores several configurations:
#   - Varying depth vs. leaf count (but not both simultaneously)
#   - Different impurity criteria ('gini' vs 'entropy')
#   - Range of regularization (ccp_alpha)
#   - Range of min_samples_split
#
# This structure prevents conflicting constraints between max_depth and max_leaf_nodes.
# ===============================================================

# Parameter exploration history (kept for transparency)
#"criterion"         : ['gini', 'entropy'],
#"max_depth"         : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#"ccp_alpha"         : [0.0, 0.0005, 0.001, 0.01, 0.05, 0.1, 0.5]
#"ccp_alpha"         : [0.0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009]
#"max_leaf_nodes"    : [None, 6, 9, 12, 15, 18, 21, 24]
#"max_leaf_nodes"    : [None, 6, 9, 12, 15, 18, 21, 24, 27, 30, 40, 50, 60, 70, 80, 90, 100]
#"max_leaf_nodes"    : [None, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89]

param_grid = [
    # Allow: (A) vary max_depth with max_leaf_nodes=None
    {
        "criterion"         : ['gini', 'entropy'],
        "max_depth"         : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "max_leaf_nodes"    : [None],
        "min_samples_split" : [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "ccp_alpha"         :  [0.0, 0.0005, 0.001, 0.01, 0.05, 0.1, 0.5]
    },
    # Allow: (B) Vary max_leaf_nodes with max_depth=None
    {
        "criterion"         : ['gini', 'entropy'],
        "max_depth"         : [None],
        "max_leaf_nodes"    : [None, 6, 9, 12, 15, 18, 21, 24],
        "min_samples_split" : [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "ccp_alpha"         : [0.0, 0.0005, 0.001, 0.01, 0.05, 0.1, 0.5]
    },
    # Baseline: (C) both unlimited, various mss/alpha
    {
        "criterion"         : ['gini', 'entropy'],
        "max_depth"         : [None],
        "max_leaf_nodes"    : [None],
        "min_samples_split" : [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "ccp_alpha"         : [0.0, 0.0005, 0.001, 0.01, 0.05, 0.1, 0.5]
    }
]

# ============================================================
# 6) k-fold cross-validation using RepeatedStratifiedKFold
# ============================================================

cross_validation = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

# ===============================================================
# 7) Grid Search with Cross-Validation
# ===============================================================
# We use F1-macro as the scoring metric because the dataset is imbalanced.
# n_jobs=-1 uses all CPU cores for parallel computation.
# ===============================================================

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    scoring='f1_macro',          
    cv=cross_validation,        
    n_jobs=-1,                  
    verbose=2,                  
    return_train_score=True
)

# Fit on the training data
grid_search.fit(X_train, y_train)

# Predict on the test set using the best estimator
# The best estimator is the actual trained model (estimator) that achieved the best score on cross-validation for the hyperparameter combination that was specified.
best_clf = grid_search.best_estimator_

# ===============================================================
# 8) Evaluate the Best Model
# ===============================================================

# Make predictions on both the train and test set
yhat_tr = best_clf.predict(X_train)
yhat_te = best_clf.predict(X_test)

print(f"Best Hyperparameters: criterion={grid_search.best_params_['criterion']}, max_depth={grid_search.best_params_['max_depth']} | max_leaf_nodes={grid_search.best_params_['max_leaf_nodes']} | min_samples_split={grid_search.best_params_['min_samples_split']} | ccp_alpha={grid_search.best_params_["ccp_alpha"]}")
# Uncomment this if you want the results to be saved to an external CSV file
saving_results(criterion=grid_search.best_params_['criterion'], max_depth=grid_search.best_params_['max_depth'], max_leaf_nodes=grid_search.best_params_['max_leaf_nodes'], min_samples_split=grid_search.best_params_['min_samples_split'], ccp_alpha=grid_search.best_params_["ccp_alpha"])

# ----------------------- TRAIN ----------------------
print("\n=== TRAIN ===")
print(f"Accuracy: {accuracy_score(y_train, yhat_tr):.3f}")
print(f"F1 (macro): {f1_score(y_train, yhat_tr, average='macro'):.3f}")
print("Confusion matrix:\n", confusion_matrix(y_train, yhat_tr))

# ----------------------- TEST ----------------------
print("\n=== TEST ===")
print(f"Accuracy: {accuracy_score(y_test, yhat_te):.3f}")
print(f"F1 (macro): {f1_score(y_test, yhat_te, average='macro'):.3f}")
print("Confusion matrix:\n", confusion_matrix(y_test, yhat_te))
print("\nClassification report (TEST):")
print(classification_report(y_test, yhat_te, digits=3))