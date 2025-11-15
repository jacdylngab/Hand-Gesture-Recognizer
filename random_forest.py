import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import joblib

# ==============================================================
#  Helper Functions
# ==============================================================

def report_line(tag, acc, f1m, f1w):
    """Nicely formatted printout of model performance metrics."""
    print(f"{tag:<30} | ACC: {acc:.3f} | F1-macro: {f1m:.3f} | F1-weighted: {f1w:.3f}")

def saving_results(Model="None", Macro_F1_Test=None, Macro_F1_Train=None, Accuracy=None):
    """Appends model performance results to Results.csv"""
    filename = Path("Results.csv")

    data = {
        "Model" : [Model],
        "Macro_F1 (Test)" : [Macro_F1_Test],
        "Macro_F1 (Train)": [Macro_F1_Train],
        "Accuracy (Test)" : [Accuracy]
    }

    df_new = pd.DataFrame(data)

    if filename.exists():
        df_new.to_csv(filename, index=False, mode='a', header=False)
    else:
        df_new.to_csv(filename, index=False, mode='w', header=True)

def displaying_the_confusion_matrix(y_pred, y_test_or_train, encoder, t):
    """Displays and optionally saves a confusion matrix plot."""

    # Convert digits -> names
    y_pred_names = encoder.inverse_transform(y_pred)
    y_true_names = encoder.inverse_transform(y_test_or_train)

    cm = confusion_matrix(y_true_names, y_pred_names)

    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
    disp.plot(cmap="Blues", ax=plt.gca())
    plt.title(f"Confusion Matrix - Random Forest ({t})")
    plt.tight_layout()
    saving_name = f"Random Forest ({t}).png"
    plt.savefig(saving_name)
    plt.show()


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
# 4) Random Forest Classifier
# ==============================================================

forest = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    random_state=142,
    n_jobs=-1,
)

# Fit on the training data
forest.fit(X_train, y_train)

# Make predictions on the training and the test set
yhat_tr = forest.predict(X_train)
yhat_te = forest.predict(X_test)


# ==============================================================
# 5) Evaluate Model
# ==============================================================

# ----------------------- TRAIN ----------------------
print("\n=== TRAIN ===")
print(f"Accuracy: {accuracy_score(y_train, yhat_tr):.3f}")
print(f"F1 (macro): {f1_score(y_train, yhat_tr, average='macro'):.3f}")
print("Confusion matrix:\n", confusion_matrix(y_train, yhat_tr))
displaying_the_confusion_matrix(y_pred=yhat_tr, y_test_or_train=y_train, encoder=encoder, t="TRAIN")


# ----------------------- TEST ----------------------
print("\n=== TEST ===")
print(f"Accuracy: {accuracy_score(y_test, yhat_te):.3f}")
print(f"F1 (macro): {f1_score(y_test, yhat_te, average='macro'):.3f}")
print("Confusion matrix:\n", confusion_matrix(y_test, yhat_te))
displaying_the_confusion_matrix(y_pred=yhat_te, y_test_or_train=y_test, encoder=encoder, t="TEST")
print("\nClassification report (TEST):")
print(classification_report(y_test, yhat_te, digits=3))

f1m_test = f1_score(y_test, yhat_te, average='macro')
f1m_train = f1_score(y_train, yhat_tr, average='macro')
acc = accuracy_score(y_test, yhat_te)
# Uncomment this if you want the results to be saved to an external CSV file
#saving_results(Model=best_param, Macro_F1_Test=f1m_test, Macro_F1_Train=f1m_train, Accuracy=acc)

# ======================================================================================
# 6. Save the decision tree model as a pickle file 
# ======================================================================================

joblib.dump(encoder, "label_encoder.pkl")
print("Encoder saved as label_encoder.pkl")
joblib.dump(forest, "random_forest.pkl")
print("Model saved as random_forest.pkl")