"""
evaluate.py
-----------
Loads the best trained model and generates:
  - Classification report (per-class precision, recall, F1)
  - Confusion matrix heatmap     -> outputs/confusion_matrix.png
  - Feature importance plot      -> outputs/feature_importance.png
  - Model comparison bar chart   -> outputs/model_comparison.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix

from preprocess import preprocess
from train import train_all
from columns import LABEL_ENCODING


# Reverse label encoding for display
LABEL_NAMES_MULTI  = {v: k for k, v in LABEL_ENCODING.items()}
LABEL_NAMES_BINARY = {0: 'Normal', 1: 'Attack'}

os.makedirs('outputs', exist_ok=True)


def plot_confusion_matrix(y_test, y_pred, label_names: dict, mode: str):
    cm = confusion_matrix(y_test, y_pred)
    labels = [label_names[i] for i in sorted(label_names)]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax
    )
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label',      fontsize=12)
    ax.set_title(f'Confusion Matrix ({mode})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = 'outputs/confusion_matrix.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_importance(model, feature_names: list, top_n: int = 20):
    if not hasattr(model, 'feature_importances_'):
        print("  Feature importance not available for this model type.")
        return

    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_values   = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_features[::-1], top_values[::-1], color='steelblue', edgecolor='white')
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

    # Annotate bars
    for bar, val in zip(bars, top_values[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=8)

    plt.tight_layout()
    path = 'outputs/feature_importance.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_model_comparison(results: dict):
    names     = list(results.keys())
    accuracies = [results[n]['accuracy'] for n in names]
    f1_scores  = [results[n]['f1']       for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy',  color='steelblue')
    bars2 = ax.bar(x + width/2, f1_scores,  width, label='F1 Score',  color='coral')

    ax.set_ylim(0.85, 1.01)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison — Accuracy vs F1 Score', fontsize=14, fontweight='bold')
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = 'outputs/model_comparison.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def evaluate(mode: str = 'multiclass'):
    print(f"\n{'='*55}")
    print(f"  Evaluation | mode={mode}")
    print(f"{'='*55}\n")

    # Train all models and collect results
    results, feature_names = train_all(mode=mode)

    # Load best model
    best_model = joblib.load('models/best_model.pkl')

    # Re-run preprocessing to get test set
    _, X_test, _, y_test, _, label_names = preprocess(mode=mode, save_artifacts=False)
    y_pred = best_model.predict(X_test)

    label_names = LABEL_NAMES_MULTI if mode == 'multiclass' else LABEL_NAMES_BINARY

    # --- Classification Report ---
    target_names = [label_names[i] for i in sorted(label_names)]
    print("\n--- Classification Report (Best Model) ---")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # --- Plots ---
    print("\nGenerating plots...")
    plot_confusion_matrix(y_test, y_pred, label_names, mode)
    plot_feature_importance(best_model, feature_names)
    plot_model_comparison(results)

    print("\nAll outputs saved to outputs/")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['binary', 'multiclass'],
                        default='multiclass')
    args = parser.parse_args()
    evaluate(mode=args.mode)
