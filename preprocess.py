"""
preprocess.py
-------------
Loads and preprocesses the NSL-KDD dataset.
Supports both binary classification (normal vs attack)
and multiclass classification (normal, DoS, Probe, R2L, U2R).
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from columns import COLUMNS, CATEGORICAL_COLS, ATTACK_MAP, LABEL_ENCODING


def load_raw(path: str) -> pd.DataFrame:
    """Load raw NSL-KDD .txt file into a DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download from: https://www.unb.ca/cic/datasets/nsl.html\n"
            "Place KDDTrain+.txt and KDDTest+.txt inside the data/ folder."
        )
    df = pd.read_csv(path, header=None, names=COLUMNS)
    return df


def encode_categoricals(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    """
    Label-encode categorical columns.
    If fit=True, fit new encoders. Otherwise use provided encoders (for test set).
    Returns (df, encoders_dict).
    """
    if encoders is None:
        encoders = {}

    df = df.copy()
    for col in CATEGORICAL_COLS:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            # Handle unseen labels gracefully
            df[col] = df[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    return df, encoders


def map_labels(df: pd.DataFrame, mode: str = 'binary') -> pd.DataFrame:
    """
    Map raw attack labels to binary or multiclass targets.

    mode='binary'     -> 0 = normal, 1 = attack
    mode='multiclass' -> 0=normal, 1=DoS, 2=Probe, 3=R2L, 4=U2R
    """
    df = df.copy()

    # Strip trailing dots (some NSL-KDD labels have them)
    df['label'] = df['label'].str.strip().str.rstrip('.')

    if mode == 'binary':
        df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

    elif mode == 'multiclass':
        df['label'] = df['label'].map(ATTACK_MAP)
        df['label'] = df['label'].map(LABEL_ENCODING)
        # Drop rows with unknown labels
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)

    else:
        raise ValueError("mode must be 'binary' or 'multiclass'")

    return df


def preprocess(
    train_path: str = 'data/KDDTrain+.txt',
    test_path: str  = 'data/KDDTest+.txt',
    mode: str       = 'multiclass',
    save_artifacts: bool = True
):
    """
    Full preprocessing pipeline.

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    print(f"[1/5] Loading data (mode={mode})...")
    train_df = load_raw(train_path)
    test_df  = load_raw(test_path)

    # Drop difficulty column (not a feature)
    train_df.drop(columns=['difficulty'], inplace=True)
    test_df.drop(columns=['difficulty'], inplace=True)

    print("[2/5] Mapping labels...")
    train_df = map_labels(train_df, mode=mode)
    test_df  = map_labels(test_df,  mode=mode)

    print("[3/5] Encoding categorical features...")
    train_df, encoders = encode_categoricals(train_df, fit=True)
    test_df, _         = encode_categoricals(test_df, encoders=encoders, fit=False)

    # Separate features and labels
    feature_names = [c for c in COLUMNS if c not in ('label', 'difficulty')]
    X_train = train_df[feature_names].values.astype(np.float32)
    y_train = train_df['label'].values
    X_test  = test_df[feature_names].values.astype(np.float32)
    y_test  = test_df['label'].values

    print("[4/5] Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print("[5/5] Done.")
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"  Label distribution (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")

    if save_artifacts:
        os.makedirs('models', exist_ok=True)
        joblib.dump(scaler,   'models/scaler.pkl')
        joblib.dump(encoders, 'models/encoders.pkl')
        print("  Saved scaler and encoders to models/")

    return X_train, X_test, y_train, y_test, feature_names


if __name__ == '__main__':
    preprocess()
