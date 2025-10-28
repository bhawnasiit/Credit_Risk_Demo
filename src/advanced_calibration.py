import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from joblib import load, dump
from pathlib import Path

def advanced_calibration_improvement():
    """
    Advanced calibration improvement using multiple techniques and models.
    """
    print("🎯 ADVANCED CALIBRATION IMPROVEMENT")
    print("=" * 60)
    
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
    
    print("🔄 Training multiple base models...")
    
    # Train different base models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_tr_s, y_tr)
        proba = model.predict_proba(X_te_s)[:, 1]
        brier = brier_score_loss(y_te, proba)
        logloss = log_loss(y_te, proba)
        
        results[name] = {
            'model': model,
            'proba': proba,
            'brier': brier,
            'logloss': logloss
        }
    
    print("🔄 Creating calibrated versions...")
    
    # Create calibrated versions of each model
    calibrated_results = {}
    
    for name, model in models.items():
        print(f"  Calibrating {name}...")
        
        # Platt Scaling
        platt_calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=3)
        platt_calibrated.fit(X_tr_s, y_tr)
        platt_proba = platt_calibrated.predict_proba(X_te_s)[:, 1]
        platt_brier = brier_score_loss(y_te, platt_proba)
        
        # Isotonic Regression
        isotonic_calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)
        isotonic_calibrated.fit(X_tr_s, y_tr)
        isotonic_proba = isotonic_calibrated.predict_proba(X_te_s)[:, 1]
        isotonic_brier = brier_score_loss(y_te, isotonic_proba)
        
        calibrated_results[name] = {
            'original': results[name],
            'platt': {
                'model': platt_calibrated,
                'proba': platt_proba,
                'brier': platt_brier
            },
            'isotonic': {
                'model': isotonic_calibrated,
                'proba': isotonic_proba,
                'brier': isotonic_brier
            }
        }
    
    # Create comprehensive comparison plot
    create_calibration_comparison_plot(calibrated_results, y_te)
    
    # Find best model
    best_model = find_best_calibrated_model(calibrated_results)
    
    # Save best model
    save_best_calibrated_model(best_model, scaler)
    
    # Print results
    print_calibration_results(calibrated_results, best_model)
    
    return calibrated_results, best_model

def create_calibration_comparison_plot(calibrated_results, y_true):
    """Create comprehensive calibration comparison plots."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']
    
    for i, (model_name, results) in enumerate(calibrated_results.items()):
        # Original model
        ax1 = axes[0, i]
        frac_pos, mean_pred = calibration_curve(y_true, results['original']['proba'], 
                                              n_bins=10, strategy="quantile")
        ax1.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        ax1.plot(mean_pred, frac_pos, marker=markers[0], linewidth=2, 
                color=colors[0], label=f"Original (Brier: {results['original']['brier']:.3f})")
        ax1.set_xlabel("Mean Predicted Probability")
        ax1.set_ylabel("Fraction of Positives")
        ax1.set_title(f"{model_name} - Original")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Calibrated models
        ax2 = axes[1, i]
        ax2.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        
        # Original
        frac_pos, mean_pred = calibration_curve(y_true, results['original']['proba'], 
                                              n_bins=10, strategy="quantile")
        ax2.plot(mean_pred, frac_pos, marker=markers[0], linewidth=2, 
                color=colors[0], label=f"Original (Brier: {results['original']['brier']:.3f})")
        
        # Platt Scaling
        frac_pos_platt, mean_pred_platt = calibration_curve(y_true, results['platt']['proba'], 
                                                           n_bins=10, strategy="quantile")
        ax2.plot(mean_pred_platt, frac_pos_platt, marker=markers[1], linewidth=2, 
                color=colors[1], label=f"Platt (Brier: {results['platt']['brier']:.3f})")
        
        # Isotonic
        frac_pos_iso, mean_pred_iso = calibration_curve(y_true, results['isotonic']['proba'], 
                                                       n_bins=10, strategy="quantile")
        ax2.plot(mean_pred_iso, frac_pos_iso, marker=markers[2], linewidth=2, 
                color=colors[2], label=f"Isotonic (Brier: {results['isotonic']['brier']:.3f})")
        
        ax2.set_xlabel("Mean Predicted Probability")
        ax2.set_ylabel("Fraction of Positives")
        ax2.set_title(f"{model_name} - Calibration Comparison")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path("artifacts").mkdir(exist_ok=True)
    plt.savefig("artifacts/advanced_calibration_comparison.png", dpi=300, bbox_inches='tight')
    print("✅ Saved artifacts/advanced_calibration_comparison.png")

def find_best_calibrated_model(calibrated_results):
    """Find the best calibrated model based on Brier score."""
    best_score = float('inf')
    best_model_info = None
    
    for model_name, results in calibrated_results.items():
        for method, data in [('original', results['original']), 
                           ('platt', results['platt']), 
                           ('isotonic', results['isotonic'])]:
            if data['brier'] < best_score:
                best_score = data['brier']
                best_model_info = {
                    'model_name': model_name,
                    'method': method,
                    'model': data['model'],
                    'brier': data['brier']
                }
    
    return best_model_info

def save_best_calibrated_model(best_model, scaler):
    """Save the best calibrated model."""
    Path("models").mkdir(exist_ok=True)
    
    dump(best_model['model'], f"models/best_calibrated_{best_model['model_name'].lower().replace(' ', '_')}_{best_model['method']}.joblib")
    dump(scaler, "models/best_calibration_scaler.joblib")
    
    print(f"✅ Saved best model: {best_model['model_name']} with {best_model['method']} calibration")
    print(f"   Brier Score: {best_model['brier']:.4f}")

def print_calibration_results(calibrated_results, best_model):
    """Print comprehensive calibration results."""
    print(f"\n📊 CALIBRATION RESULTS SUMMARY")
    print("=" * 80)
    
    # Create results table
    print(f"{'Model':<20} {'Method':<12} {'Brier Score':<12} {'Improvement':<12}")
    print("-" * 60)
    
    for model_name, results in calibrated_results.items():
        original_brier = results['original']['brier']
        
        # Original
        print(f"{model_name:<20} {'Original':<12} {original_brier:<12.4f} {'Baseline':<12}")
        
        # Platt Scaling
        platt_brier = results['platt']['brier']
        platt_improvement = ((original_brier - platt_brier) / original_brier) * 100
        print(f"{'':<20} {'Platt':<12} {platt_brier:<12.4f} {platt_improvement:+.1f}%")
        
        # Isotonic
        isotonic_brier = results['isotonic']['brier']
        isotonic_improvement = ((original_brier - isotonic_brier) / original_brier) * 100
        print(f"{'':<20} {'Isotonic':<12} {isotonic_brier:<12.4f} {isotonic_improvement:+.1f}%")
        print()
    
    print(f"🏆 BEST MODEL: {best_model['model_name']} with {best_model['method']} calibration")
    print(f"   Brier Score: {best_model['brier']:.4f}")
    
    # Calibration quality assessment
    if best_model['brier'] < 0.1:
        quality = "Excellent"
    elif best_model['brier'] < 0.2:
        quality = "Good"
    elif best_model['brier'] < 0.3:
        quality = "Fair"
    else:
        quality = "Poor"
    
    print(f"   Calibration Quality: {quality}")

if __name__ == "__main__":
    calibrated_results, best_model = advanced_calibration_improvement()
    
    print(f"\n🎉 ADVANCED CALIBRATION IMPROVEMENT COMPLETED!")
    print("=" * 60)
