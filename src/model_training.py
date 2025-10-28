import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from joblib import dump
from pathlib import Path

df = pd.read_csv("data/credit_data.csv")
X = df.drop(columns=["default"])
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Train Logistic Regression Model
print("🔄 Training Logistic Regression Model...")
model_lr = LogisticRegression(max_iter=2000)
model_lr.fit(X_train_s, y_train)

# Train Linear Regression Model (as baseline)
print("🔄 Training Linear Regression Model (Baseline)...")
model_linear = LinearRegression()
model_linear.fit(X_train_s, y_train)

# Define score cutoff threshold
SCORE_CUTOFF = 0.5  # Adjust this value to change the decision threshold

# Get predictions and probabilities for Logistic Regression
proba_lr = model_lr.predict_proba(X_test_s)[:, 1]
y_pred_lr = (proba_lr > SCORE_CUTOFF).astype(int)  # Custom threshold

# Get predictions for Linear Regression (convert to probabilities using sigmoid)
linear_pred = model_linear.predict(X_test_s)
# Convert linear regression outputs to probabilities using sigmoid function
proba_linear = 1 / (1 + np.exp(-linear_pred))
# Convert probabilities to binary predictions using custom threshold
y_pred_linear = (proba_linear > SCORE_CUTOFF).astype(int)

# Calculate metrics for Logistic Regression
auc_lr = roc_auc_score(y_test, proba_lr)
brier_lr = brier_score_loss(y_test, proba_lr)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

# Calculate metrics for Linear Regression
auc_linear = roc_auc_score(y_test, proba_linear)
brier_linear = brier_score_loss(y_test, proba_linear)
accuracy_linear = accuracy_score(y_test, y_pred_linear)
precision_linear = precision_score(y_test, y_pred_linear)
recall_linear = recall_score(y_test, y_pred_linear)
f1_linear = f1_score(y_test, y_pred_linear)

# Additional metrics for Linear Regression
mse_linear = mean_squared_error(y_test, linear_pred)
r2_linear = r2_score(y_test, linear_pred)

# Model Comparison Analysis
print("=" * 80)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 80)
print(f"🎯 Score Cutoff Threshold: {SCORE_CUTOFF}")
print(f"   (Probabilities > {SCORE_CUTOFF} are classified as Default)")
print("=" * 80)

print(f"\n📊 CLASSIFICATION METRICS COMPARISON:")
print(f"{'Metric':<15} {'Logistic Reg':<15} {'Linear Reg':<15} {'Improvement':<15}")
print("-" * 60)
print(f"{'Accuracy':<15} {accuracy_lr:<15.3f} {accuracy_linear:<15.3f} {((accuracy_lr - accuracy_linear) / accuracy_linear * 100):<15.1f}%")
print(f"{'Precision':<15} {precision_lr:<15.3f} {precision_linear:<15.3f} {((precision_lr - precision_linear) / precision_linear * 100):<15.1f}%")
print(f"{'Recall':<15} {recall_lr:<15.3f} {recall_linear:<15.3f} {((recall_lr - recall_linear) / recall_linear * 100):<15.1f}%")
print(f"{'F1-Score':<15} {f1_lr:<15.3f} {f1_linear:<15.3f} {((f1_lr - f1_linear) / f1_linear * 100):<15.1f}%")

print(f"\n🎯 PROBABILITY CALIBRATION COMPARISON:")
print(f"{'Metric':<15} {'Logistic Reg':<15} {'Linear Reg':<15} {'Improvement':<15}")
print("-" * 60)
print(f"{'ROC AUC':<15} {auc_lr:<15.3f} {auc_linear:<15.3f} {((auc_lr - auc_linear) / auc_linear * 100):<15.1f}%")
print(f"{'Brier Score':<15} {brier_lr:<15.4f} {brier_linear:<15.4f} {((brier_linear - brier_lr) / brier_linear * 100):<15.1f}%")

# Brier Score Interpretation for both models
def get_brier_interpretation(brier_score):
    if brier_score < 0.1:
        return "Excellent"
    elif brier_score < 0.2:
        return "Good"
    elif brier_score < 0.3:
        return "Fair"
    else:
        return "Poor"

print(f"\n📈 BRIER SCORE ANALYSIS:")
print(f"  Logistic Regression Brier Score: {brier_lr:.4f} ({get_brier_interpretation(brier_lr)})")
print(f"  Linear Regression Brier Score:   {brier_linear:.4f} ({get_brier_interpretation(brier_linear)})")
print(f"  Random Baseline: 0.2500")
print(f"  Perfect Score: 0.0000")
print(f"  Logistic vs Random: {((0.25 - brier_lr) / 0.25) * 100:.1f}% improvement")
print(f"  Linear vs Random:   {((0.25 - brier_linear) / 0.25) * 100:.1f}% improvement")

# Additional Linear Regression metrics
print(f"\n📏 LINEAR REGRESSION ADDITIONAL METRICS:")
print(f"  MSE (Mean Squared Error): {mse_linear:.4f}")
print(f"  R² Score: {r2_linear:.4f}")

# Probability distribution comparison
print(f"\n🔍 PROBABILITY DISTRIBUTION COMPARISON:")
print(f"  {'Metric':<25} {'Logistic Reg':<15} {'Linear Reg':<15}")
print("-" * 55)
print(f"  {'Mean probability':<25} {proba_lr.mean():<15.3f} {proba_linear.mean():<15.3f}")
print(f"  {'Std probability':<25} {proba_lr.std():<15.3f} {proba_linear.std():<15.3f}")
print(f"  {'Min probability':<25} {proba_lr.min():<15.3f} {proba_linear.min():<15.3f}")
print(f"  {'Max probability':<25} {proba_lr.max():<15.3f} {proba_linear.max():<15.3f}")

# Calibration analysis
print(f"\n⚖️  CALIBRATION ANALYSIS:")
print(f"  Actual default rate: {y_test.mean():.3f}")
print(f"  Logistic predicted rate: {proba_lr.mean():.3f} (diff: {abs(y_test.mean() - proba_lr.mean()):.3f})")
print(f"  Linear predicted rate:    {proba_linear.mean():.3f} (diff: {abs(y_test.mean() - proba_linear.mean()):.3f})")

# Overall improvement summary
print(f"\n🏆 OVERALL IMPROVEMENT SUMMARY:")
print(f"  Logistic Regression shows improvement over Linear Regression in:")
improvements = []
if accuracy_lr > accuracy_linear:
    improvements.append(f"Accuracy (+{((accuracy_lr - accuracy_linear) / accuracy_linear * 100):.1f}%)")
if auc_lr > auc_linear:
    improvements.append(f"ROC AUC (+{((auc_lr - auc_linear) / auc_linear * 100):.1f}%)")
if brier_lr < brier_linear:
    improvements.append(f"Brier Score ({((brier_linear - brier_lr) / brier_linear * 100):.1f}% better)")

for improvement in improvements:
    print(f"    ✅ {improvement}")

if not improvements:
    print("    ⚠️  No significant improvements detected")

# Save results to file
results = {
    'logistic_regression': {
        'accuracy': accuracy_lr,
        'precision': precision_lr,
        'recall': recall_lr,
        'f1_score': f1_lr,
        'roc_auc': auc_lr,
        'brier_score': brier_lr,
        'brier_interpretation': get_brier_interpretation(brier_lr),
        'predicted_default_rate': proba_lr.mean(),
        'calibration_difference': abs(y_test.mean() - proba_lr.mean())
    },
    'linear_regression': {
        'accuracy': accuracy_linear,
        'precision': precision_linear,
        'recall': recall_linear,
        'f1_score': f1_linear,
        'roc_auc': auc_linear,
        'brier_score': brier_linear,
        'brier_interpretation': get_brier_interpretation(brier_linear),
        'predicted_default_rate': proba_linear.mean(),
        'calibration_difference': abs(y_test.mean() - proba_linear.mean()),
        'mse': mse_linear,
        'r2_score': r2_linear
    },
    'comparison': {
        'accuracy_improvement': ((accuracy_lr - accuracy_linear) / accuracy_linear * 100),
        'auc_improvement': ((auc_lr - auc_linear) / auc_linear * 100),
        'brier_improvement': ((brier_linear - brier_lr) / brier_linear * 100),
        'actual_default_rate': y_test.mean()
    },
    'model_config': {
        'score_cutoff': SCORE_CUTOFF,
        'threshold_description': f'Probabilities > {SCORE_CUTOFF} classified as Default'
    }
}

# Save models and results
Path("models").mkdir(exist_ok=True)
dump(model_lr, "models/logreg_pd.joblib")
dump(model_linear, "models/linear_reg.joblib")
dump(scaler, "models/scaler.joblib")

# Save results as JSON
import json
with open("models/training_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n💾 SAVED FILES:")
print("✅ models/logreg_pd.joblib (Logistic Regression)")
print("✅ models/linear_reg.joblib (Linear Regression)")
print("✅ models/scaler.joblib (Feature Scaler)") 
print("✅ models/training_results.json (Results)")

print(f"\n🎉 MODEL TRAINING & COMPARISON COMPLETED!")
print("=" * 80)
