import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from joblib import load
from pathlib import Path

def ks_score(y_true, y_proba, n_bins: int = 20):
    df = pd.DataFrame({"y": y_true, "p": y_proba})
    df["bin"] = pd.qcut(df["p"], n_bins, duplicates="drop")
    grouped = df.groupby("bin")
    cum_good = (grouped["y"].apply(lambda x: (1 - x).sum()).cumsum()) / (1 - df["y"]).sum()
    cum_bad = (grouped["y"].sum().cumsum()) / df["y"].sum()
    return np.max(np.abs(cum_bad.values - cum_good.values))

def psi(expected: pd.Series, actual: pd.Series, n_bins: int = 10):
    e_bins = pd.qcut(expected, n_bins, duplicates="drop")
    a_bins = pd.cut(actual, pd.IntervalIndex(e_bins.cat.categories))
    e_pct = e_bins.value_counts(normalize=True).sort_index()
    a_pct = a_bins.value_counts(normalize=True).sort_index()
    return np.sum((a_pct - e_pct) * np.log((a_pct + 1e-9) / (e_pct + 1e-9)))

def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_plot=True):
    """
    Create and display a confusion matrix with detailed analysis.
    
    Args:
        y_true: Actual labels
        y_pred: Predicted labels
        model_name: Name of the model for display
        save_plot: Whether to save the plot
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Raw counts confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
    ax1.set_title(f'{model_name} - Confusion Matrix (Counts)')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # Plot 2: Normalized confusion matrix (percentages)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax2,
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
    ax2.set_title(f'{model_name} - Confusion Matrix (Percentages)')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    
    if save_plot:
        Path("validation_plots").mkdir(exist_ok=True)
        plt.savefig(f'validation_plots/confusion_matrix_{model_name.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        print(f"✅ Saved confusion matrix plot: validation_plots/confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    
    plt.show()
    
    # Detailed analysis
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n📊 CONFUSION MATRIX ANALYSIS - {model_name}")
    print("=" * 60)
    print(f"True Negatives (TN):  {tn:>6} - Correctly predicted No Default")
    print(f"False Positives (FP): {fp:>6} - Incorrectly predicted Default (Type I Error)")
    print(f"False Negatives (FN): {fn:>6} - Missed Default (Type II Error)")
    print(f"True Positives (TP):  {tp:>6} - Correctly predicted Default")
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📈 PERFORMANCE METRICS:")
    print(f"  Accuracy:    {accuracy:.3f}")
    print(f"  Precision:   {precision:.3f}")
    print(f"  Recall:      {recall:.3f}")
    print(f"  Specificity: {specificity:.3f}")
    print(f"  F1-Score:    {f1_score:.3f}")
    
    # Business impact analysis
    print(f"\n💼 BUSINESS IMPACT ANALYSIS:")
    print(f"  False Positives (Good customers rejected): {fp} ({fp/(fp+tn)*100:.1f}% of good customers)")
    print(f"  False Negatives (Bad customers approved):  {fn} ({fn/(fn+tp)*100:.1f}% of bad customers)")
    
    # Risk assessment
    if fp > fn:
        print("  ⚠️  Model is conservative (rejecting more good customers)")
    elif fn > fp:
        print("  ⚠️  Model is aggressive (approving more bad customers)")
    else:
        print("  ✅ Model is well-balanced")
    
    return cm, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1_score,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }

if __name__ == "__main__":
    print("🔄 Loading trained models and data...")
    
    # Load data
    holdout = pd.read_csv("data/credit_data.csv")
    X = holdout.drop(columns=["default"])
    y = holdout["default"]
    
    # Load trained models
    try:
        model_lr = load("models/logreg_pd.joblib")
        scaler = load("models/scaler.joblib")
        print("✅ Loaded Logistic Regression model and scaler")
    except FileNotFoundError:
        print("❌ Trained models not found. Please run model_training.py first.")
        exit(1)
    
    # Prepare data
    X_scaled = scaler.transform(X)
    
    # Get predictions from Logistic Regression
    y_pred_lr = model_lr.predict(X_scaled)
    y_proba_lr = model_lr.predict_proba(X_scaled)[:, 1]
    
    print("\n" + "=" * 80)
    print("CONFUSION MATRIX ANALYSIS")
    print("=" * 80)
    
    # Create confusion matrix for Logistic Regression
    cm_lr, metrics_lr = plot_confusion_matrix(y, y_pred_lr, "Logistic Regression")
    
    # Traditional validation metrics
    print(f"\n📊 TRADITIONAL VALIDATION METRICS:")
    auc = roc_auc_score(y, y_proba_lr)
    ks = ks_score(y, y_proba_lr)
    
    # Create a simple score for PSI calculation
    rng = np.random.default_rng(123)
    score = 0.5 * (holdout["credit_score"] - holdout["credit_score"].min()) / (
        holdout["credit_score"].max() - holdout["credit_score"].min()
    ) + 0.5 * rng.random(len(holdout))
    
    stability = psi(
        expected=score.sample(frac=0.5, random_state=1),
        actual=score.drop(score.sample(frac=0.5, random_state=1).index)
    )
    
    print(f"  AUC: {auc:.3f}")
    print(f"  KS:  {ks:.3f}")
    print(f"  PSI: {stability:.3f}")
    
    # Model performance summary
    print(f"\n🏆 MODEL PERFORMANCE SUMMARY:")
    print(f"  Overall Accuracy: {metrics_lr['accuracy']:.1%}")
    print(f"  Precision:        {metrics_lr['precision']:.1%}")
    print(f"  Recall:           {metrics_lr['recall']:.1%}")
    print(f"  F1-Score:         {metrics_lr['f1_score']:.1%}")
    
    # Business recommendations
    print(f"\n💡 BUSINESS RECOMMENDATIONS:")
    if metrics_lr['fp'] > metrics_lr['fn']:
        print("  ⚠️  Consider lowering the decision threshold to reduce false positives")
        print("  📈 This will approve more good customers but may increase risk")
    elif metrics_lr['fn'] > metrics_lr['fp']:
        print("  ⚠️  Consider raising the decision threshold to reduce false negatives")
        print("  📉 This will catch more bad customers but may reject good ones")
    else:
        print("  ✅ Current threshold appears well-balanced")
    
    print(f"\n🎉 VALIDATION COMPLETED!")
    print("=" * 80)
