import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, precision_recall_curve, roc_curve, 
                           accuracy_score, precision_score, recall_score, f1_score)
from joblib import load
from pathlib import Path

def optimize_calibration_threshold():
    """
    Optimize calibration by finding the best threshold for different business objectives.
    """
    print("🎯 CALIBRATION THRESHOLD OPTIMIZATION")
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
    
    # Train model
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_tr_s, y_tr)
    proba = model.predict_proba(X_te_s)[:, 1]
    
    print("🔄 Finding optimal thresholds for different business objectives...")
    
    # 1. Threshold for maximum accuracy
    thresholds = np.linspace(0.1, 0.9, 81)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    brier_scores = []
    
    for threshold in thresholds:
        y_pred = (proba > threshold).astype(int)
        accuracies.append(accuracy_score(y_te, y_pred))
        precisions.append(precision_score(y_te, y_pred))
        recalls.append(recall_score(y_te, y_pred))
        f1_scores.append(f1_score(y_te, y_pred))
        brier_scores.append(brier_score_loss(y_te, proba))
    
    # Find optimal thresholds
    optimal_accuracy_threshold = thresholds[np.argmax(accuracies)]
    optimal_precision_threshold = thresholds[np.argmax(precisions)]
    optimal_recall_threshold = thresholds[np.argmax(recalls)]
    optimal_f1_threshold = thresholds[np.argmax(f1_scores)]
    
    # Business-specific thresholds
    conservative_threshold = 0.3  # More likely to predict default
    aggressive_threshold = 0.7    # Less likely to predict default
    balanced_threshold = 0.5     # Balanced approach
    
    # Create comprehensive analysis
    create_threshold_analysis_plot(thresholds, accuracies, precisions, recalls, f1_scores, brier_scores)
    
    # Evaluate different thresholds
    evaluate_thresholds(proba, y_te, {
        'Optimal Accuracy': optimal_accuracy_threshold,
        'Optimal Precision': optimal_precision_threshold,
        'Optimal Recall': optimal_recall_threshold,
        'Optimal F1': optimal_f1_threshold,
        'Conservative (0.3)': conservative_threshold,
        'Balanced (0.5)': balanced_threshold,
        'Aggressive (0.7)': aggressive_threshold
    })
    
    # Create calibration improvement recommendations
    create_calibration_recommendations(proba, y_te, optimal_f1_threshold)
    
    return {
        'optimal_accuracy': optimal_accuracy_threshold,
        'optimal_precision': optimal_precision_threshold,
        'optimal_recall': optimal_recall_threshold,
        'optimal_f1': optimal_f1_threshold
    }

def create_threshold_analysis_plot(thresholds, accuracies, precisions, recalls, f1_scores, brier_scores):
    """Create comprehensive threshold analysis plot."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Accuracy vs Threshold
    axes[0, 0].plot(thresholds, accuracies, 'b-', linewidth=2, label='Accuracy')
    axes[0, 0].axvline(thresholds[np.argmax(accuracies)], color='b', linestyle='--', alpha=0.7)
    axes[0, 0].set_xlabel('Threshold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy vs Threshold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Precision vs Threshold
    axes[0, 1].plot(thresholds, precisions, 'g-', linewidth=2, label='Precision')
    axes[0, 1].axvline(thresholds[np.argmax(precisions)], color='g', linestyle='--', alpha=0.7)
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision vs Threshold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Recall vs Threshold
    axes[1, 0].plot(thresholds, recalls, 'r-', linewidth=2, label='Recall')
    axes[1, 0].axvline(thresholds[np.argmax(recalls)], color='r', linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].set_title('Recall vs Threshold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # F1 Score vs Threshold
    axes[1, 1].plot(thresholds, f1_scores, 'm-', linewidth=2, label='F1 Score')
    axes[1, 1].axvline(thresholds[np.argmax(f1_scores)], color='m', linestyle='--', alpha=0.7)
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('F1 Score vs Threshold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    Path("artifacts").mkdir(exist_ok=True)
    plt.savefig("artifacts/threshold_optimization_analysis.png", dpi=300, bbox_inches='tight')
    print("✅ Saved artifacts/threshold_optimization_analysis.png")

def evaluate_thresholds(proba, y_true, thresholds_dict):
    """Evaluate different thresholds and print results."""
    print(f"\n📊 THRESHOLD EVALUATION RESULTS")
    print("=" * 80)
    print(f"{'Strategy':<20} {'Threshold':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 80)
    
    results = {}
    
    for strategy, threshold in thresholds_dict.items():
        y_pred = (proba > threshold).astype(int)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        print(f"{strategy:<20} {threshold:<10.3f} {accuracy:<10.3f} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f}")
        
        results[strategy] = {
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return results

def create_calibration_recommendations(proba, y_true, optimal_threshold):
    """Create calibration improvement recommendations."""
    print(f"\n💡 CALIBRATION IMPROVEMENT RECOMMENDATIONS")
    print("=" * 60)
    
    # Calculate current vs optimal performance
    current_pred = (proba > 0.5).astype(int)
    optimal_pred = (proba > optimal_threshold).astype(int)
    
    current_f1 = f1_score(y_true, current_pred)
    optimal_f1 = f1_score(y_true, optimal_pred)
    
    improvement = ((optimal_f1 - current_f1) / current_f1) * 100
    
    print(f"🎯 RECOMMENDED THRESHOLD: {optimal_threshold:.3f}")
    print(f"   Current F1-Score (0.5 threshold): {current_f1:.3f}")
    print(f"   Optimal F1-Score ({optimal_threshold:.3f} threshold): {optimal_f1:.3f}")
    print(f"   Improvement: {improvement:+.1f}%")
    
    # Business impact analysis
    print(f"\n💼 BUSINESS IMPACT ANALYSIS:")
    
    # Calculate confusion matrices
    current_cm = calculate_confusion_matrix(y_true, current_pred)
    optimal_cm = calculate_confusion_matrix(y_true, optimal_pred)
    
    print(f"   Current Threshold (0.5):")
    print(f"     False Positives: {current_cm['fp']} (Good customers rejected)")
    print(f"     False Negatives: {current_cm['fn']} (Bad customers approved)")
    
    print(f"   Optimal Threshold ({optimal_threshold:.3f}):")
    print(f"     False Positives: {optimal_cm['fp']} (Good customers rejected)")
    print(f"     False Negatives: {optimal_cm['fn']} (Bad customers approved)")
    
    # Risk assessment
    if optimal_cm['fp'] < current_cm['fp']:
        print(f"   ✅ Reduces false positives by {current_cm['fp'] - optimal_cm['fp']} (fewer good customers rejected)")
    else:
        print(f"   ⚠️  Increases false positives by {optimal_cm['fp'] - current_cm['fp']} (more good customers rejected)")
    
    if optimal_cm['fn'] < current_cm['fn']:
        print(f"   ✅ Reduces false negatives by {current_cm['fn'] - optimal_cm['fn']} (fewer bad customers approved)")
    else:
        print(f"   ⚠️  Increases false negatives by {optimal_cm['fn'] - current_cm['fn']} (more bad customers approved)")

def calculate_confusion_matrix(y_true, y_pred):
    """Calculate confusion matrix components."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}

if __name__ == "__main__":
    optimal_thresholds = optimize_calibration_threshold()
    
    print(f"\n🏆 OPTIMAL THRESHOLDS FOUND:")
    for metric, threshold in optimal_thresholds.items():
        print(f"   {metric.replace('_', ' ').title()}: {threshold:.3f}")
    
    print(f"\n🎉 THRESHOLD OPTIMIZATION COMPLETED!")
    print("=" * 60)
