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

UNSW_TRAIN_FILE = 'UNSW_NB15_training-set.parquet'
UNSW_TEST_FILE  = 'UNSW_NB15_testing-set.parquet'
UNSW_CATEGORICAL_COLS = ['proto', 'service', 'state']
UNSW_ONEHOT_COLS = ['service', 'state']
UNSW_FREQ_COL = 'proto_freq'
NSL_LABEL_NAMES = {v: k for k, v in LABEL_ENCODING.items()}
LABEL_NAMES_BINARY = {0: 'Normal', 1: 'Attack'}


def load_raw(path: str) -> pd.DataFrame:
    """Load raw NSL-KDD or UNSW-NB15 dataset into a DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Place KDDTrain+.txt and KDDTest+.txt inside the data/ folder,"
            " or place NLS-KDD.txt or UNSW_NB15 training/testing parquet files in the repo root."
        )

    if path.lower().endswith('.parquet'):
        return pd.read_parquet(path)

    return pd.read_csv(path, header=None, names=COLUMNS)


def discover_dataset_paths(train_path: str, test_path: str):
    if os.path.exists(UNSW_TRAIN_FILE) and os.path.exists(UNSW_TEST_FILE):
        return UNSW_TRAIN_FILE, UNSW_TEST_FILE

    if os.path.exists(train_path) and os.path.exists(test_path):
        return train_path, test_path

    combined_path = 'NLS-KDD.txt'
    if os.path.exists(combined_path):
        print(f"[INFO] Found combined dataset file '{combined_path}'. Splitting into train/test.")
        return combined_path, None

    return train_path, test_path


def map_unsw_labels(train_df: pd.DataFrame, test_df: pd.DataFrame, mode: str = 'binary'):
    if mode == 'binary':
        train_df['label'] = train_df['label'].astype(int)
        test_df['label']  = test_df['label'].astype(int)
        label_names = {0: 'Normal', 1: 'Attack'}

    elif mode == 'multiclass':
        encoder = LabelEncoder()
        all_cats = pd.concat([
            train_df['attack_cat'].astype(str),
            test_df['attack_cat'].astype(str)
        ], ignore_index=True)
        encoder.fit(all_cats)

        train_df['label'] = encoder.transform(train_df['attack_cat'].astype(str))
        test_df['label']  = encoder.transform(test_df['attack_cat'].astype(str))
        label_names = {int(i): name for i, name in enumerate(encoder.classes_)}

    else:
        raise ValueError("mode must be 'binary' or 'multiclass'")

    return train_df, test_df, label_names


def add_unsw_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create UNSW-specific numeric features from packet and byte counts."""
    df = df.copy()
    df['total_bytes'] = df['sbytes'] + df['dbytes']
    df['total_pkts'] = df['spkts'] + df['dpkts']
    df['avg_pkt_size'] = df['total_bytes'] / (df['total_pkts'] + 1)
    df['dur_per_pkt'] = df['dur'] / (df['total_pkts'] + 1)
    df['byte_ratio'] = df['sbytes'] / (df['dbytes'] + 1)
    df['pkt_ratio'] = df['spkts'] / (df['dpkts'] + 1)
    df['synack_ratio'] = df['synack'] / (df['ackdat'] + 1)
    return df


def add_unsw_categorical_dummies(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """One-hot encode service and state, aligned between train and test."""
    train_df = pd.get_dummies(train_df, columns=UNSW_ONEHOT_COLS, prefix=UNSW_ONEHOT_COLS)
    test_df = pd.get_dummies(test_df, columns=UNSW_ONEHOT_COLS, prefix=UNSW_ONEHOT_COLS)

    all_cols = sorted(set(train_df.columns).union(test_df.columns))
    train_df = train_df.reindex(columns=all_cols, fill_value=0)
    test_df = test_df.reindex(columns=all_cols, fill_value=0)
    return train_df, test_df


def add_unsw_proto_frequency(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_df = train_df.copy()
    test_df = test_df.copy()
    proto_freq = train_df['proto'].value_counts(normalize=True).to_dict()
    train_df[UNSW_FREQ_COL] = train_df['proto'].map(proto_freq).fillna(0.0)
    test_df[UNSW_FREQ_COL] = test_df['proto'].map(proto_freq).fillna(0.0)
    return train_df, test_df


def encode_categoricals(df: pd.DataFrame, categorical_cols: list, encoders: dict = None, fit: bool = True):
    """
    Label-encode categorical columns.
    If fit=True, fit new encoders. Otherwise use provided encoders (for test set).
    Returns (df, encoders_dict).
    """
    if encoders is None:
        encoders = {}

    df = df.copy()
    for col in categorical_cols:
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
        X_train, X_test, y_train, y_test, feature_names, label_names
    """
    print(f"[1/5] Loading data (mode={mode})...")
    train_path, test_path = discover_dataset_paths(train_path, test_path)

    if test_path is None:
        full_df = load_raw(train_path)
        full_df.drop(columns=['difficulty'], inplace=True)

        print("[2/5] Mapping labels...")
        full_df = map_labels(full_df, mode=mode)

        print("[2.5/5] Splitting combined dataset into train/test...")
        train_df, test_df = train_test_split(
            full_df,
            test_size=0.20,
            stratify=full_df['label'],
            random_state=42
        )
        print(f"  Split: train={len(train_df)} rows, test={len(test_df)} rows")
        label_names = LABEL_NAMES_BINARY if mode == 'binary' else NSL_LABEL_NAMES

    elif train_path.lower().endswith('.parquet'):
        train_df = load_raw(train_path)
        test_df  = load_raw(test_path)

        print("[2/5] Mapping labels for UNSW-NB15...")
        train_df, test_df, label_names = map_unsw_labels(train_df, test_df, mode=mode)

        print("[2.5/5] Building derived UNSW features...")
        train_df = add_unsw_derived_features(train_df)
        test_df = add_unsw_derived_features(test_df)

        print("[2.6/5] Adding UNSW categorical dummies...")
        train_df, test_df = add_unsw_categorical_dummies(train_df, test_df)

        print("[2.7/5] Adding UNSW proto frequency feature...")
        train_df, test_df = add_unsw_proto_frequency(train_df, test_df)

    else:
        train_df = load_raw(train_path)
        test_df  = load_raw(test_path)

        # Drop difficulty column (not a feature)
        train_df.drop(columns=['difficulty'], inplace=True)
        test_df.drop(columns=['difficulty'], inplace=True)

        print("[2/5] Mapping labels...")
        train_df = map_labels(train_df, mode=mode)
        test_df  = map_labels(test_df,  mode=mode)
        label_names = LABEL_NAMES_BINARY if mode == 'binary' else NSL_LABEL_NAMES

    print("[3/5] Encoding categorical features...")
    if train_path.lower().endswith('.parquet'):
        train_df, encoders = encode_categoricals(train_df, UNSW_CATEGORICAL_COLS, fit=True)
        test_df, _         = encode_categoricals(test_df,  UNSW_CATEGORICAL_COLS, encoders=encoders, fit=False)
        feature_names = [c for c in train_df.columns if c not in ('label', 'attack_cat')]
    else:
        train_df, encoders = encode_categoricals(train_df, CATEGORICAL_COLS, fit=True)
        test_df, _         = encode_categoricals(test_df, CATEGORICAL_COLS, encoders=encoders, fit=False)
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

    return X_train, X_test, y_train, y_test, feature_names, label_names


if __name__ == '__main__':
    preprocess()
