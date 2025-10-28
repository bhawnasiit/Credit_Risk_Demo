
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from joblib import load
from pathlib import Path

def create_enhanced_calibration_plot():
    """
    Create comprehensive calibration plots with multiple calibration techniques.
    """
    print("🔄 Loading data and models...")
    
    # Load data
    df = pd.read_csv("data/credit_data.csv")
    X = df.drop(columns=["default"])
    y = df["default"]
    
    # Split data
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    # Train base model
    print("🔄 Training base logistic regression model...")
    base_model = LogisticRegression(max_iter=2000).fit(X_tr_s, y_tr)
    base_proba = base_model.predict_proba(X_te_s)[:, 1]
    
    # Create calibrated models
    print("🔄 Creating calibrated models...")
    
    # 1. Platt Scaling (Sigmoid calibration)
    platt_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
    platt_model.fit(X_tr_s, y_tr)
    platt_proba = platt_model.predict_proba(X_te_s)[:, 1]
    
    # 2. Isotonic Regression calibration
    isotonic_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
    isotonic_model.fit(X_tr_s, y_tr)
    isotonic_proba = isotonic_model.predict_proba(X_te_s)[:, 1]
    
    # Calculate Brier scores
    base_brier = brier_score_loss(y_te, base_proba)
    platt_brier = brier_score_loss(y_te, platt_proba)
    isotonic_brier = brier_score_loss(y_te, isotonic_proba)
    
    print(f"📊 Brier Scores:")
    print(f"  Base Model:     {base_brier:.4f}")
    print(f"  Platt Scaling:  {platt_brier:.4f}")
    print(f"  Isotonic:       {isotonic_brier:.4f}")
    
    # Create comprehensive calibration plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Original calibration
    ax1 = axes[0, 0]
    frac_pos, mean_pred = calibration_curve(y_te, base_proba, n_bins=10, strategy="quantile")
    ax1.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax1.plot(mean_pred, frac_pos, marker="o", linewidth=2, label=f"Base Model (Brier: {base_brier:.3f})")
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title("Original Model Calibration")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Platt Scaling calibration
    ax2 = axes[0, 1]
    frac_pos_platt, mean_pred_platt = calibration_curve(y_te, platt_proba, n_bins=10, strategy="quantile")
    ax2.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax2.plot(mean_pred_platt, frac_pos_platt, marker="s", linewidth=2, 
             color="green", label=f"Platt Scaling (Brier: {platt_brier:.3f})")
    ax2.set_xlabel("Mean Predicted Probability")
    ax2.set_ylabel("Fraction of Positives")
    ax2.set_title("Platt Scaling Calibration")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Isotonic calibration
    ax3 = axes[1, 0]
    frac_pos_iso, mean_pred_iso = calibration_curve(y_te, isotonic_proba, n_bins=10, strategy="quantile")
    ax3.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax3.plot(mean_pred_iso, frac_pos_iso, marker="^", linewidth=2, 
             color="red", label=f"Isotonic (Brier: {isotonic_brier:.3f})")
    ax3.set_xlabel("Mean Predicted Probability")
    ax3.set_ylabel("Fraction of Positives")
    ax3.set_title("Isotonic Regression Calibration")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Comparison of all methods
    ax4 = axes[1, 1]
    ax4.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax4.plot(mean_pred, frac_pos, marker="o", linewidth=2, label=f"Base Model (Brier: {base_brier:.3f})")
    ax4.plot(mean_pred_platt, frac_pos_platt, marker="s", linewidth=2, 
             color="green", label=f"Platt Scaling (Brier: {platt_brier:.3f})")
    ax4.plot(mean_pred_iso, frac_pos_iso, marker="^", linewidth=2, 
             color="red", label=f"Isotonic (Brier: {isotonic_brier:.3f})")
    ax4.set_xlabel("Mean Predicted Probability")
    ax4.set_ylabel("Fraction of Positives")
    ax4.set_title("Calibration Comparison")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    Path("artifacts").mkdir(exist_ok=True)
    plt.savefig("artifacts/enhanced_calibration_analysis.png", dpi=300, bbox_inches='tight')
    print("✅ Saved artifacts/enhanced_calibration_analysis.png")
    
    # Create probability distribution comparison
    create_probability_distribution_plot(base_proba, platt_proba, isotonic_proba, y_te)
    
    # Save calibrated models
    save_calibrated_models(platt_model, isotonic_model, scaler)
    
    return {
        'base_brier': base_brier,
        'platt_brier': platt_brier,
        'isotonic_brier': isotonic_brier,
        'best_method': min([('base', base_brier), ('platt', platt_brier), ('isotonic', isotonic_brier)], 
                          key=lambda x: x[1])[0]
    }

def create_probability_distribution_plot(base_proba, platt_proba, isotonic_proba, y_true):
    """Create probability distribution comparison plot."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Base model distribution
    axes[0].hist(base_proba[y_true==0], bins=30, alpha=0.7, label='No Default', color='blue')
    axes[0].hist(base_proba[y_true==1], bins=30, alpha=0.7, label='Default', color='red')
    axes[0].set_title('Base Model Probability Distribution')
    axes[0].set_xlabel('Predicted Probability')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Platt scaling distribution
    axes[1].hist(platt_proba[y_true==0], bins=30, alpha=0.7, label='No Default', color='blue')
    axes[1].hist(platt_proba[y_true==1], bins=30, alpha=0.7, label='Default', color='red')
    axes[1].set_title('Platt Scaling Probability Distribution')
    axes[1].set_xlabel('Predicted Probability')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Isotonic distribution
    axes[2].hist(isotonic_proba[y_true==0], bins=30, alpha=0.7, label='No Default', color='blue')
    axes[2].hist(isotonic_proba[y_true==1], bins=30, alpha=0.7, label='Default', color='red')
    axes[2].set_title('Isotonic Regression Probability Distribution')
    axes[2].set_xlabel('Predicted Probability')
    axes[2].set_ylabel('Frequency')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("artifacts/probability_distribution_comparison.png", dpi=300, bbox_inches='tight')
    print("✅ Saved artifacts/probability_distribution_comparison.png")

def save_calibrated_models(platt_model, isotonic_model, scaler):
    """Save the calibrated models for future use."""
    from joblib import dump
    
    Path("models").mkdir(exist_ok=True)
    dump(platt_model, "models/platt_calibrated_model.joblib")
    dump(isotonic_model, "models/isotonic_calibrated_model.joblib")
    dump(scaler, "models/calibration_scaler.joblib")
    
    print("✅ Saved calibrated models:")
    print("  - models/platt_calibrated_model.joblib")
    print("  - models/isotonic_calibrated_model.joblib")
    print("  - models/calibration_scaler.joblib")

if __name__ == "__main__":
    print("🎯 ENHANCED CALIBRATION ANALYSIS")
    print("=" * 60)
    
    results = create_enhanced_calibration_plot()
    
    print(f"\n🏆 CALIBRATION IMPROVEMENT RESULTS:")
    print(f"  Best Method: {results['best_method'].upper()}")
    print(f"  Base Model Brier Score:    {results['base_brier']:.4f}")
    print(f"  Platt Scaling Brier Score: {results['platt_brier']:.4f}")
    print(f"  Isotonic Brier Score:      {results['isotonic_brier']:.4f}")
    
    # Calculate improvements
    platt_improvement = ((results['base_brier'] - results['platt_brier']) / results['base_brier']) * 100
    isotonic_improvement = ((results['base_brier'] - results['isotonic_brier']) / results['base_brier']) * 100
    
    print(f"\n📈 IMPROVEMENTS:")
    print(f"  Platt Scaling:  {platt_improvement:+.1f}%")
    print(f"  Isotonic:       {isotonic_improvement:+.1f}%")
    
    print(f"\n🎉 CALIBRATION ANALYSIS COMPLETED!")
    print("=" * 60)
