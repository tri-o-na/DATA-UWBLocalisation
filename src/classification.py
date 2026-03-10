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
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
)

# Ensure consistent results
np.random.seed(42)

# Load dataset
DATA_PATH = "../data/processed/cleaned_data.csv"

print("Loading dataset")
df = pd.read_csv(DATA_PATH)

print("Dataset shape: ", df.shape)

# Baseline Features
FEATURES = [
    "FP_IDX", "FP_AMP1","FP_AMP2","FP_AMP3","STDEV_NOISE","CIR_PWR","MAX_NOISE","RXPACC","CH","FRAME_LEN","PREAM_LEN","BITRATE","PRFR"
]

# Input Features
x = df[FEATURES]

# Target label (0 = LOS, 1 = NLOS)
y = df["NLOS"]

print("LOS samples:" , (y==0).sum())
print("NLOS samples:" , (y==1).sum())

# Train and Test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# Define models
models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("model",LogisticRegression(max_iter=1000))
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

# Train + evaluate models
results = []
predictions = {}
probabilities = {}

# Training Loop
for name, model in models.items():
    print("\nTraining", name)

    # Train model
    model.fit(x_train,y_train)

    # Predictions
    model_predictions = model.predict(x_test)

    # ROC Probability  
    model_probability = model.predict_proba(x_test)[:,1]

    # Compute Metrics
    accuracy = accuracy_score(y_test, model_predictions)
    precision = precision_score(y_test, model_predictions)
    recall = recall_score(y_test, model_predictions)
    f1 = f1_score(y_test, model_predictions)
    auc = roc_auc_score(y_test, model_probability)

    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    print("ROC AUC: ", auc)

    # Cross Validation
    cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring="accuracy")

    print("CV Accuracy mean: ", cv_scores.mean())
    print("CV Accuracy std: ", cv_scores.std())

    results.append({
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1":f1,
        "auc":auc,
        "cv_accuracy_mean":cv_scores.mean(),
        "cv_accuracy_std":cv_scores.std()
    })

    predictions[name] = model_predictions
    probabilities[name] = model_probability

# Feature Importance for RandomForest
if "RandomForest" in models:
    rf_model = models["RandomForest"]
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance - RandomForest")
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [FEATURES[i] for i in indices])
    plt.xlabel("Relative Importance")

    # Save Feature Importance
    os.makedirs("../results/figures", exist_ok=True)
    plt.savefig("../results/figures/classification_random_forest_feature_importance.png")
    plt.close()

# Save Metrics
metrics_df = pd.DataFrame(results)
os.makedirs("../results/metrics", exist_ok=True)
metrics_df.to_csv("../results/metrics/classification_metrics.csv", index=False)

print("\nSaved metrics to results/metrics")

# Determine best model based on highest accuracy
best_model = metrics_df.sort_values("accuracy", ascending=False).iloc[0]["model"]

print("Best model:", best_model)

best_prediction = predictions[best_model]
best_probability = probabilities[best_model]

# Confusion Matrix
cm = confusion_matrix(y_test, best_prediction)

disp = ConfusionMatrixDisplay(cm)
disp.plot()

# Save Confusion Matrix
os.makedirs("../results/figures", exist_ok=True)
plt.title(f"Confusion Matrix ({best_model})")
plt.savefig("../results/figures/classification_confusion_matrix.png")
plt.close()


# ROC Curves
plt.figure(figsize=(6,5))
for name in models.keys():
    model_probability = probabilities[name]

    fpr , tpr, _ = roc_curve(y_test, model_probability)
    plt.plot(fpr,tpr,label=name)

plt.plot([0,1],[0,1],'r--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()

# Save ROC Curves
plt.savefig("../results/figures/classification_roc_curve.png")
plt.close()

# Accuracy Comparison Chart
model_names = metrics_df["model"]
accuracy_values = metrics_df["accuracy"]

plt.figure(figsize=(6,5))
plt.bar(model_names, accuracy_values)

plt.ylabel("Accuracy")
plt.title("Classification Model Comparision")

# Save Accuracy Comparison Chart
plt.savefig("../results/figures/classification_accuracy_comparison.png")
plt.close()

# Save Predictions
predictions_df = x_test.copy()
predictions_df["true_label"] = y_test.values
predictions_df["predicted_label"] = best_prediction

predictions_df.to_csv("../results/metrics/classification_predictions.csv", index=False)

print("Saved classification prediction in ../results/metrics and ../results/figures")