import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

np.random.seed(42)

# -----------------------------
# Load classifier predictions
# -----------------------------
print("Loading classifier predictions...")

pred_df = pd.read_csv(
    "../results/metrics/classification_predictions_enhanced_features.csv"
)

print("Prediction dataset shape:", pred_df.shape)

# -----------------------------
# Load enhanced feature dataset
# -----------------------------
feature_df = pd.read_csv("../data/processed/enhanced_features.csv")

# CIR engineered features
FEATURES = [
    "FP_AMP1",
    "STDEV_NOISE",
    "rms_delay",
    "kurtosis",
    "skewness",
    "peak_amp"
]

# -----------------------------
# Merge RANGE
# -----------------------------
print("Merging with enhanced dataset...")

df = pred_df.merge(
    feature_df[FEATURES + ["RANGE"]],
    on=FEATURES,
    how="left"
)

print("Merged dataset shape:", df.shape)

# Remove rows where merge failed
df = df.dropna(subset=["RANGE"])

print("Dataset after removing NaN RANGE:", df.shape)

# -----------------------------
# Keep predicted LOS
# -----------------------------
df_los = df[df["predicted_label"] == 0].copy()

print("Predicted LOS samples:", len(df_los))

# -----------------------------
# Regression dataset
# -----------------------------
X = df_los[FEATURES]
y = df_los["RANGE"]

print("Regression dataset:", X.shape)

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Models
# -----------------------------
models = {
    "LinearRegression": LinearRegression(),

    "RandomForest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "NeuralNetwork": Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=(64,32),
            max_iter=300,
            random_state=42
        ))
    ])
}

results = []
predictions = {}

# -----------------------------
# Train models
# -----------------------------
for name, model in models.items():

    print("\nTraining", name)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2:", r2)

    results.append({
        "model": name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    })

    predictions[name] = preds


# -----------------------------
# Save metrics
# -----------------------------
metrics_df = pd.DataFrame(results)

os.makedirs("../results/metrics", exist_ok=True)

metrics_df.to_csv(
    "../results/metrics/regression_full_pipeline_metrics.csv",
    index=False
)

print("Saved full pipeline metrics")

# -----------------------------
# Best model
# -----------------------------
best_model = metrics_df.sort_values("rmse").iloc[0]["model"]

print("Best model:", best_model)

best_preds = predictions[best_model]

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(6,5))
plt.scatter(y_test, best_preds, alpha=0.3)

min_val = min(y_test.min(), best_preds.min())
max_val = max(y_test.max(), best_preds.max())

plt.plot([min_val,max_val],[min_val,max_val],'r--')

plt.xlabel("Actual Range")
plt.ylabel("Predicted Range")
plt.title(f"Predicted vs Actual ({best_model})")

os.makedirs("../results/figures", exist_ok=True)

plt.savefig("../results/figures/predicted_vs_actual_full_pipeline.png")

plt.close()

print("Saved prediction plot")