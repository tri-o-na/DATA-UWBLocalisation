import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score

# SETUP PATHS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
CLEANED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'cleaned_data.csv')
METRICS_DIR = os.path.join(PROJECT_ROOT, 'results', 'metrics')

# ENVIRONMENT MAPPING
# This maps your filenames to the actual names programmatically
ENV_NAMES = {
    'uwb_dataset_part1.csv': 'Office 1',
    'uwb_dataset_part2.csv': 'Office 2',
    'uwb_dataset_part3.csv': 'Small Apartment',
    'uwb_dataset_part4.csv': 'Small Workshop',
    'uwb_dataset_part5.csv': 'Kitchen/Living Room',
    'uwb_dataset_part6.csv': 'Bedroom',
    'uwb_dataset_part7.csv': 'Boiler Room'
}

def get_features_and_labels(df):
    # Features used in the project
    features = ['FP_AMP1', 'STDEV_NOISE', 'CIR_PWR', 'RXPACC', 'FP_IDX', 'MAX_NOISE']
    X = df[features]
    y_class = df['NLOS']
    y_reg = df['RANGE']
    return X, y_class, y_reg

# Load the master dataset
print(f"Loading cleaned data from: {CLEANED_DATA_PATH}")
full_df = pd.read_csv(CLEANED_DATA_PATH)

# Get unique files from the data
files = sorted(full_df['source_file'].unique())

results = []

# ----- 1. CROSS ENVIRONMENT VALIDATION -----
print(f"========== Cross Environment Validation on {len(files)} environments ========== ")

# LOEO LOOP
for i, file_name in enumerate(files):
    # Lookup the pretty name, fallback to filename if not found
    actual_name = ENV_NAMES.get(file_name, file_name)
    
    print(f"Iteration {i+1}: Testing on {actual_name}...")
    
    # Split using the 'source_file' column
    train_df = full_df[full_df['source_file'] != file_name]
    test_df = full_df[full_df['source_file'] == file_name]
    
    X_train, y_train_c, _ = get_features_and_labels(train_df)
    X_test, y_test_c, _ = get_features_and_labels(test_df)
    
    # A. Train Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train_c)
    clf_acc = accuracy_score(y_test_c, clf.predict(X_test))
    
    # B. Train Regressor (Only on LOS samples)
    train_los = train_df[train_df['NLOS'] == 0]
    test_los = test_df[test_df['NLOS'] == 0]
    
    rmse = None
    if not test_los.empty:
        X_train_r, _, y_train_r_los = get_features_and_labels(train_los)
        X_test_r, _, y_test_r_los = get_features_and_labels(test_los)
        
        reg = RandomForestRegressor(n_estimators=100, random_state=42)
        reg.fit(X_train_r, y_train_r_los)
        preds = reg.predict(X_test_r)
        rmse = np.sqrt(mean_squared_error(y_test_r_los, preds))
    
    # Store results using the ACTUAL name
    results.append({
        'Environment': actual_name,
        'Classifier_Accuracy': clf_acc,
        'Regression_RMSE': rmse
    })

# Save Results
os.makedirs(METRICS_DIR, exist_ok=True)
loeo_df = pd.DataFrame(results)
output_path = os.path.join(METRICS_DIR, 'cross_env_results.csv')
loeo_df.to_csv(output_path, index=False)

print(f"\nValidation Complete. Results saved to: {output_path}")
print(loeo_df)

# ----- 2. DATA VISUALIZATION -----
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

print("\n========== Generating Visualizations ==========")

# CHART 1: Accuracy per Environment (Classifier)
plt.figure(figsize=(12, 6))
# Create colors for each bar
colors = plt.cm.Paired(np.linspace(0, 1, len(loeo_df)))
bars = plt.bar(loeo_df['Environment'], loeo_df['Classifier_Accuracy'], color=colors)

# Add a dashed line for the average
avg_acc = loeo_df['Classifier_Accuracy'].mean()
plt.axhline(y=avg_acc, color='red', linestyle='--', alpha=0.6, label=f'Average: {avg_acc:.2f}')

plt.title('Classifier Accuracy by Indoor Environment (LOS vs NLOS)', fontsize=14)
plt.ylabel('Accuracy Score', fontsize=12)
plt.xticks(rotation=30, ha='right')
plt.ylim(0, 1.1) # Scale to 100%
plt.legend()

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'accuracy_per_environment.png'))
print(f"Saved: accuracy_per_environment.png")


# CHART 2: Performance Drop (Generalization Gap)
RANDOM_METRICS_PATH = os.path.join(PROJECT_ROOT, 'results', 'metrics', 'regression_enhanced_metrics.csv')

if os.path.exists(RANDOM_METRICS_PATH):
    random_results = pd.read_csv(RANDOM_METRICS_PATH)
    try:
        # Get baseline from teammate's RandomForest results
        avg_random_rmse = random_results[random_results['model'] == 'RandomForest']['rmse'].values[0]
        avg_cross_rmse = loeo_df['Regression_RMSE'].mean()

        plt.figure(figsize=(8, 6))
        labels = ['Random Split\n(Same Rooms)', 'Cross-Environment\n(New Room)']
        values = [avg_random_rmse, avg_cross_rmse]
        
        bars = plt.bar(labels, values, color=['#bdc3c7', '#e74c3c'])
        plt.ylabel('RMSE (meters)', fontsize=12)
        plt.title('The Generalization Gap: UWB Distance Error', fontsize=14)
        
        # Labeling 
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.3f}m', ha='center', va='bottom', fontweight='bold')

        # Drop %
        drop_pct = ((avg_cross_rmse - avg_random_rmse) / avg_random_rmse) * 100
        plt.annotate(f'+{drop_pct:.1f}% Error Increase', 
                     xy=(1, avg_cross_rmse), xytext=(0.5, avg_cross_rmse + 0.05),
                     fontsize=12, color='red', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'performance_drop_comparison.png'))
        print(f"Saved: performance_drop_comparison.png")
    except Exception as e:
        print(f"Skipped Gap Chart: {e}")


# ----- 3. PERFORMANCE DROP ANALYTICS -----
print("\n========== PERFORMANCE ANALYSIS ==========")

if os.path.exists(RANDOM_METRICS_PATH):
    # Calculate Percentage Drop
    # Formula: ((New - Old) / Old) * 100
    percentage_drop = ((avg_cross_rmse - avg_random_rmse) / avg_random_rmse) * 100
    
    print(f"Random Split RMSE:  {avg_random_rmse:.4f}m")
    print(f"Cross-Env RMSE:     {avg_cross_rmse:.4f}m")
    print(f"Performance Drop:   {percentage_drop:.1f}% increase in error")
    
    if percentage_drop > 50:
        print("INSIGHT: Significant Generalization Gap detected. The model is highly sensitive to new environments.")
    else:
        print("INSIGHT: The model generalizes well to new environments.")

# Find the hardest environment
# We use 'Regression_RMSE' because that's what we named it in the results append earlier
hardest_env = loeo_df.loc[loeo_df['Regression_RMSE'].idxmax()]
easiest_env = loeo_df.loc[loeo_df['Regression_RMSE'].idxmin()]

print(f"\nHardest Environment: {hardest_env['Environment']} (RMSE: {hardest_env['Regression_RMSE']:.4f}m)")
print(f"Easiest Environment: {easiest_env['Environment']} (RMSE: {easiest_env['Regression_RMSE']:.4f}m)")

# Environment Sensitivity Logic
sensitivity_ratio = hardest_env['Regression_RMSE'] / easiest_env['Regression_RMSE']
print(f"Sensitivity Ratio:   {sensitivity_ratio:.1f}x difference between best and worst rooms.")

# ----- 4. SUMMARY: WHICH MODEL GENERALIZES BEST? -----
print("\n" + "="*40)
print("========== FINAL SUMMARY & WINNER ==========")
print("="*40)

# The model that generalizes best is the one with the lowest variation
std_dev = loeo_df['Classifier_Accuracy'].std()
best_env = loeo_df.loc[loeo_df['Classifier_Accuracy'].idxmax(), 'Environment']
worst_env = loeo_df.loc[loeo_df['Classifier_Accuracy'].idxmin(), 'Environment']

print(f"Average Accuracy across all rooms: {avg_acc:.2f}")
print(f"Environment Consistency (Std Dev): {std_dev:.4f} (Lower is better)")
print(f"Best Generalization: {best_env}")
print(f"Worst Generalization: {worst_env}")

if std_dev < 0.05:
    print("\nVERDICT: The Random Forest model generalizes EXCELLENTLY.")
else:
    print("\nVERDICT: The model is SENSITIVE to room geometry.")
print("="*40)