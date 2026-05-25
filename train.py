"""
train.py
--------
Trains three classifiers on the NSL-KDD dataset:
  1. Random Forest
  2. Gradient Boosting
  3. MLP Neural Network

Saves the best model to models/best_model.pkl.
"""

import os
import time
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

from preprocess import preprocess


def get_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            n_jobs=-1,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42
        )
    }


def train_all(mode: str = 'multiclass'):
    X_train, X_test, y_train, y_test, feature_names, _ = preprocess(mode=mode)

    models   = get_models()
    results  = {}
    best_f1  = -1
    best_name = None
    best_model = None

    print("\n" + "="*55)
    print(f"  Training {len(models)} models | mode={mode}")
    print("="*55)

    sample_weight = compute_sample_weight('balanced', y_train)
    print(f"  Sample weights computed for {len(np.unique(y_train))} classes.")

    for name, model in models.items():
        print(f"\n>> {name}")
        t0 = time.time()
        model.fit(X_train, y_train, sample_weight=sample_weight)
        elapsed = time.time() - t0

        y_pred   = model.predict(X_test)
        acc      = accuracy_score(y_test, y_pred)
        f1       = f1_score(y_test, y_pred, average='weighted')

        results[name] = {
            'model':    model,
            'accuracy': acc,
            'f1':       f1,
            'time':     elapsed
        }

        print(f"   Accuracy : {acc:.4f}")
        print(f"   F1 Score : {f1:.4f}")
        print(f"   Time     : {elapsed:.1f}s")

        if f1 > best_f1:
            best_f1   = f1
            best_name = name
            best_model = model

    print("\n" + "="*55)
    print(f"  Best Model : {best_name}  (F1={best_f1:.4f})")
    print("="*55)

    # Save best model and metadata
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.pkl')

    # Save all models individually
    for name, res in results.items():
        safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        joblib.dump(res['model'], f'models/{safe_name}.pkl')

    # Save feature names for later use
    joblib.dump(feature_names, 'models/feature_names.pkl')

    print("\nAll models saved to models/")
    return results, feature_names


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['binary', 'multiclass'],
                        default='multiclass', help='Classification mode')
    args = parser.parse_args()
    train_all(mode=args.mode)
