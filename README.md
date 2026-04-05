# UWB Indoor Localisation — LOS/NLOS Classification & Distance Estimation
- A machine learning pipeline for Ultra-Wideband (UWB) indoor localisation, tackling LOS/NLOS signal classification and distance estimation across 7 diverse indoor environments.
---

## Pipeline Overview
Raw CSVs → Data Cleaning → Feature Extraction → Feature Pruning → Classification → Regression → Cross-Environment Validation

## Demo Video
https://youtu.be/DaBEi96-gzc

---

## Setup

### Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy shap
```

### Run Order

Scripts must be run in order from the `src/` directory:

```bash
cd src/
```

---

## Step-by-Step Execution

### Step 1 — Data Cleaning
```bash
python data_cleaning.py
```
- Loads all 7 raw CSV files from `data/raw/`
- Runs EDA, removes duplicates, nulls, invalid RANGE and NLOS values
- Applies Isolation Forest anomaly detection (contamination = 5%)
- **Output:** `data/processed/cleaned_data.csv`, `data/processed/anomalies.csv`
- **Figures:** `results/figures/class_balance.png`

---

### Step 2 — CIR Feature Extraction
```bash
python feature_extraction.py
```
- Extracts 4 CIR-derived features from 1016 raw CIR values:
  `rms_delay`, `kurtosis`, `skewness`, `peak_amp`
- Combines with 13 original scalar features
- **Output:** `data/processed/old_enhanced_features.csv`

---

### Step 3 — Feature Analysis

Run these in any order to analyse the initial feature set:

```bash
python feature_example_plot.py   # CIR signal comparison (LOS vs NLOS)
python feature_importance.py     # SHAP analysis + Gini importance ranking
python feature_correlation.py    # Pearson correlation heatmap
```
- **Figures:** `results/figures/feature_cir_comparison.png`,
  `results/figures/feature_shap.png`,
  `results/figures/feature_gini_importance.png`,
  `results/figures/feature_correlation_heatmap.png`

---

### Step 4 — Feature Pruning
```bash
python improved_feature_extraction.py
```
- Removes low-importance and redundant features identified in Step 3
- Final feature set (9 features): `FP_IDX`, `FP_AMP1`, `STDEV_NOISE`,
  `CIR_PWR`, `MAX_NOISE`, `RXPACC`, `rms_delay`, `kurtosis`, `peak_amp`
- **Output:** `data/processed/enhanced_features.csv`

---

### Step 5 — Feature Performance Comparison
```bash
python feature_performance_comparison.py
```
- Compares accuracy across 3 stages: Baseline → Engineering → Optimisation
- **Figures:** `results/figures/feature_performance_comparison.png`

---

### Step 6 — Classification

**Experiment 1: Baseline (13 scalar features)**
```bash
python classification.py
```
- **Output:** `results/metrics/classification_metrics.csv`,
  `results/metrics/classification_predictions.csv`
- **Figures:** Confusion matrix, ROC curve, feature importance,
  accuracy comparison

**Experiment 2: Enhanced (9 features)**
```bash
python classfication_enhanced_features.py
```
- **Output:** `results/metrics/classification_metrics_enhanced_features.csv`,
  `results/metrics/classification_predictions_enhanced_features.csv`
- **Figures:** Confusion matrix, ROC curve, feature importance,
  accuracy comparison (enhanced)

---

### Step 7 — Regression

**Experiment 1: Baseline regression (ground-truth LOS)**
```bash
python regression_baseline.py
```
- Uses 13 scalar features, true LOS labels
- **Output:** `results/metrics/regression_baseline_metrics.csv`

**Experiment 2: Enhanced regression (ground-truth LOS)**
```bash
python regression_enhanced.py
```
- Uses 9 enhanced features, true LOS labels
- **Output:** `results/metrics/regression_enhanced_metrics.csv`

**Experiment 3: Baseline pipeline (predicted LOS)**
```bash
python regression_classifier_pipeline.py
```
- Uses baseline classifier predictions to filter LOS samples
- Requires `classification_predictions.csv` from Step 6
- **Output:** `results/metrics/regression_classifier_pipeline_metrics.csv`

**Experiment 4: Full enhanced pipeline (predicted LOS)**
```bash
python regression_full_pipeline.py
```
- Uses enhanced classifier predictions + enhanced features
- Requires `classification_predictions_enhanced_features.csv` from Step 6
- **Output:** `results/metrics/regression_full_pipeline_metrics.csv`

---

### Step 8 — Cross-Environment Validation
```bash
# not run in src/
cd..
python cross_env_validation.py
```
- Leave-One-Environment-Out (LOEO) across all 7 environments
- Trains RF Classifier + RF Regressor per iteration
- Requires `enhanced_features.csv` and `cleaned_data.csv`
- Requires `regression_enhanced_metrics.csv` for generalisation gap
- **Output:** `results/metrics/cross_env_results.csv`
- **Figures:** `results/figures/accuracy_per_environment.png`,
  `results/figures/performance_drop_comparison.png`

---

## Key Results

| Metric | Value |
|---|---|
| Avg Classification Accuracy (LOEO) | 86% |
| Cross-Environment RMSE | 1.0048m |
| Random Split Baseline RMSE | 1.0191m |
| Generalisation Gap | -1.4% (improved) |
| Sensitivity Ratio | 1.0× |
| Std Dev of Accuracy | 0.0055 |
| Best Regression RMSE (ground-truth) | 0.996m (Neural Network, Exp 2) |
| Best Regression RMSE (full pipeline) | 1.233m (Random Forest, Exp 4) |

---

## Dataset

7 indoor environments, 6000 samples each (3000 LOS + 3000 NLOS):

| File | Environment |
|---|---|
| uwb_dataset_part1.csv | Office 1 |
| uwb_dataset_part2.csv | Office 2 |
| uwb_dataset_part3.csv | Small Apartment |
| uwb_dataset_part4.csv | Small Workshop |
| uwb_dataset_part5.csv | Kitchen/Living Room |
| uwb_dataset_part6.csv | Bedroom |
| uwb_dataset_part7.csv | Boiler Room |

---

## Notes

- All scripts must be run from the `src/` directory to resolve relative paths correctly
- Random seed is fixed at `42` throughout for reproducibility
- `regression_enhanced_metrics.csv` must exist before running
  `cross_env_validation.py` to generate the generalisation gap chart
```
