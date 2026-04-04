import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 4 of feature engineering pipeline
# General comparison of datasets

# 1. Load your enhanced dataset
# This file contains both original and engineered features
# 1. Load Datasets
# Raw/Cleaned data contains the 1016 CIR samples
df_raw = pd.read_csv('../data/processed/cleaned_data.csv').dropna()
# Enhanced data contains your 3 engineered CIR features
df_enhanced = pd.read_csv('../data/processed/old_enhanced_features.csv').dropna()

# 2. Define the three evolutionary feature sets
# STAGE 1: Basic Original Scalars + CIR vector (Baseline)
basic_cols = [c for c in df_raw.columns if c not in ['NLOS', 'RANGE', 'FP_IDX', 'FP_AMP1', 'FP_AMP2', 'FP_AMP3',
               'STDEV_NOISE', 'CIR_PWR', 'MAX_NOISE', 'RXPACC',
               'CH', 'FRAME_LEN', 'PREAM_LEN', 'BITRATE', 'PRFR', 'source_file']]

# STAGE 2: Feature Engineering (The "Expanded" set)
# Includes hardware configs and all engineered features
enhanced_cols = ['FP_IDX', 'FP_AMP1', 'FP_AMP2', 'FP_AMP3',
               'STDEV_NOISE', 'CIR_PWR', 'MAX_NOISE', 'RXPACC',
               'CH', 'FRAME_LEN', 'PREAM_LEN', 'BITRATE', 'PRFR',
                 'rms_delay', 'kurtosis', 'skewness', 'peak_amp']

# STAGE 3: Feature Optimisation (The "Pruned" set)
# Focuses on high-value targets and engineered features while removing redundancy
optimised_cols = [
    'FP_IDX', 'FP_AMP1',
    'STDEV_NOISE', 'CIR_PWR', 'MAX_NOISE', 'RXPACC',
    'rms_delay', 'kurtosis', 'peak_amp'
]
df = df_enhanced.join(df_raw[basic_cols])

X = df.drop('NLOS', axis=1)
y = df['NLOS']

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Training & Evaluation Functions
def get_accuracy(cols):
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train[cols], y_train)
    return accuracy_score(y_test, model.predict(X_test[cols]))

print("Evaluating Baseline...")
acc_basic = get_accuracy(basic_cols)

print("Evaluating Feature Engineering (Features Added)...")
acc_enhanced = get_accuracy(enhanced_cols)

print("Evaluating Feature Optimisation (Features Pruned)...")
acc_optimised = get_accuracy(optimised_cols)

# 5. Generate Comparison Diagram
plt.figure(figsize=(10, 6))
labels = ['Baseline\n(Original)', 'Engineering\n(Features Added)', 'Optimisation\n(Features Pruned)']
accuracies = [acc_basic, acc_enhanced, acc_optimised]
colors = ['#d3d3d3', '#5bc0de', '#2e6da4'] # Grey, Light Blue, Dark Blue

bars = plt.bar(labels, accuracies, color=colors)
plt.ylim(min(accuracies) - 0.05, 1.0) # Dynamic zoom
plt.ylabel('Accuracy Score')
plt.title('UWB NLOS Detection Performance Evolution')

# Add text labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{height:.2%}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('../results/figures/feature_performance_comparison.png')

print(f"\nRESULTS:")
print(f"Baseline: {acc_basic:.2%}")
print(f"Engineering: {acc_enhanced:.2%}")
print(f"Optimisation: {acc_optimised:.2%}")
print(f"\nTotal Gain: {acc_optimised - acc_basic:.2%}")