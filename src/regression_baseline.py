import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ensure consistent results
np.random.seed(42)

# Load cleaned dataset
DATA_PATH = "../data/processed/cleaned_data.csv"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

# Filter True LOS samples
df_los = df[df["NLOS"] == 0].copy()

print("LOS samples:", len(df_los))

# Select baseline Features (no CIR yet)
FEATURES = [
    "FP_IDX","FP_AMP1","FP_AMP2","FP_AMP3",
    "STDEV_NOISE","CIR_PWR","MAX_NOISE","RXPACC",
    "CH","FRAME_LEN","PREAM_LEN","BITRATE","PRFR"
]

# input -> signal features, output -> predicted distance
X = df_los[FEATURES]
y = df_los["RANGE"]

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define models
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

# Train + evaluate models
results = []
predictions = {}

# training loop
for name, model in models.items():

    print("\nTraining", name)

    # train the model
    model.fit(X_train, y_train)

    # predict distances
    preds = model.predict(X_test)

    # compute metrics
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2:", r2)

    # cross validation for stability (split data into 5 parts, train/test 5 times)
    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="neg_root_mean_squared_error"
    )

    cv_rmse = -cv_scores

    print("CV RMSE mean:", cv_rmse.mean())
    print("CV RMSE std:", cv_rmse.std())

    results.append({
        "model": name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "cv_rmse_mean": cv_rmse.mean(),
        "cv_rmse_std": cv_rmse.std()
    })

    predictions[name] = preds

# Save metrics
metrics_df = pd.DataFrame(results)

os.makedirs("../results/metrics", exist_ok=True)

metrics_df.to_csv(
    "../results/metrics/regression_baseline_metrics.csv",
    index=False
)

print("\nSaved metrics to results/metrics/")

# Create plots 
# Predicted vs actual

# Scatter Plot 
# select the best model based on lowest rmse
best_model = metrics_df.sort_values("rmse").iloc[0]["model"]
print("Best model:", best_model)

best_preds = predictions[best_model]

plt.figure(figsize=(6,5))
plt.scatter(y_test, best_preds, alpha=0.3)

min_val = min(y_test.min(), best_preds.min())
max_val = max(y_test.max(), best_preds.max())

plt.plot([min_val,max_val],[min_val,max_val],'r--')

plt.xlabel("Actual Range")
plt.ylabel("Predicted Range")
plt.title(f"Predicted vs Actual ({best_model})")

plt.savefig("../results/figures/predicted_vs_actual_baseline.png")

plt.close()

# Residual plot
residuals = y_test - best_preds

plt.figure(figsize=(6,5))
plt.scatter(best_preds, residuals, alpha=0.3)
plt.axhline(0, color="red")

plt.xlabel("Predicted Range")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.savefig("../results/figures/residual_plot_baseline.png")

plt.close()

# RMSE comparison chart
rmse_values = metrics_df["rmse"]
model_names = metrics_df["model"]

plt.figure(figsize=(6,5))
plt.bar(model_names, rmse_values)

plt.ylabel("RMSE")
plt.title("RMSE Comparison")

plt.savefig("../results/figures/rmse_comparison_baseline.png")

plt.close()








