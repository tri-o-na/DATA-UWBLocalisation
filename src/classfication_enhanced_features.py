import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
)

np.random.seed(42)

# Load dataset
DATA_PATH = "../data/processed/enhanced_features.csv"

print("Loading dataset")
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

X = df.drop(columns=["NLOS","RANGE"])
feature_names = X.columns
y = df["NLOS"]

print("LOS samples:", (y==0).sum())
print("NLOS samples:", (y==1).sum())

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Models
models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True))
    ]),

    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

results = []
predictions = {}
probabilities = {}
trained_models = {}

# Training loop
for name, model in models.items():

    print("\nTraining", name)

    model.fit(X_train, y_train)
    trained_models[name] = model

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("ROC AUC:", auc)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

    print("CV Accuracy mean:", cv_scores.mean())
    print("CV Accuracy std:", cv_scores.std())

    results.append({
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "cv_accuracy_mean": cv_scores.mean(),
        "cv_accuracy_std": cv_scores.std()
    })

    predictions[name] = preds
    probabilities[name] = probs


# Feature importance (RandomForest)
rf_model = trained_models["RandomForest"]
importances = rf_model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10,6))
plt.title("Feature Importance - RandomForest")
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Importance")

os.makedirs("../results/figures", exist_ok=True)
plt.savefig("../results/figures/classification_random_forest_feature_importance_enhanced_features.png")
plt.close()

# Save metrics
metrics_df = pd.DataFrame(results)
os.makedirs("../results/metrics", exist_ok=True)

metrics_df.to_csv(
    "../results/metrics/classification_metrics_enhanced_features.csv",
    index=False
)

print("\nSaved metrics")

# Best model
best_model = metrics_df.sort_values("accuracy", ascending=False).iloc[0]["model"]

print("Best model:", best_model)

best_preds = predictions[best_model]

# Confusion matrix
cm = confusion_matrix(y_test, best_preds)

disp = ConfusionMatrixDisplay(cm, display_labels=["LOS","NLOS"])
disp.plot()

plt.title(f"Confusion Matrix ({best_model})")

plt.savefig("../results/figures/classification_confusion_matrix_enhanced_features.png")
plt.close()


# ROC curves
plt.figure(figsize=(6,5))

for name in models.keys():

    probs = probabilities[name]
    fpr, tpr, _ = roc_curve(y_test, probs)

    auc_score = roc_auc_score(y_test, probs)

    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.3f})")

plt.plot([0,1],[0,1],'r--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()

plt.savefig("../results/figures/classification_roc_curve_enhanced_features.png")
plt.close()


# Accuracy comparison
plt.figure(figsize=(6,5))

plt.bar(metrics_df["model"], metrics_df["accuracy"])

plt.ylabel("Accuracy")
plt.title("Classification Model Comparison")

plt.savefig("../results/figures/classification_accuracy_comparison_enhanced_features.png")
plt.close()


# Save predictions
pred_df = X_test.copy()
pred_df["true_label"] = y_test.values
pred_df["predicted_label"] = best_preds

pred_df.to_csv(
    "../results/metrics/classification_predictions_enhanced_features.csv",
    index=False
)

print("Finished classification enhanced features")