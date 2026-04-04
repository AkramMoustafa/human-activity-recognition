import os
import glob
import zipfile
import numpy as np
import pandas as pd

def extract_data(zip_path, extract_dir):
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)
        print("✅ Extracted log files.")
    else:
        print("⚠️ ZIP already extracted.")

    files = glob.glob(os.path.join(extract_dir, "*.csv"))
    print(f"📁 Found {len(files)} CSV files")
    return files

def expand_labels(labels, back=5):
    new_labels = labels.copy()
    for i in range(len(labels)):
        if labels[i] == 1:
            start = max(0, i - back)
            new_labels[start:i] = 1
    return new_labels

def load_gesture_soon_dataset():
    """
    Wrapper that runs your full dataset pipeline
    """
    zip_path = "data/snaptic_logs.zip"
    extract_dir = "data/logs"

    extract_data(zip_path, extract_dir)

    features, labels = load_all_data(extract_dir)

    X, Y = create_clean_dataset(features, labels)

    return X, Y

def load_all_data(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.csv"))

    all_features = []
    all_labels = []

    for file in files:
        df = pd.read_csv(file)

        feature_cols = [col for col in df.columns if "acc" in col or "gyro" in col]
        if len(feature_cols) == 0:
            feature_cols = df.select_dtypes(include=['number']).columns.tolist()
            feature_cols = [c for c in feature_cols if c != "label"]

        if "label" in df.columns:
            label_col = "label"
        elif "gesture" in df.columns:
            label_col = "gesture"
        else:
            raise ValueError(f"No label column in {file}")

        features = df[feature_cols].values
        labels = df[label_col].values

        labels = np.array([0 if l == "no_gesture" else 1 for l in labels])

        all_features.append(features)
        all_labels.append(labels)

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    labels = expand_labels(labels)

    return features, labels

def create_clean_dataset(features, labels, window_size=20, stride=5, K=20, max_gap=10):

    X = []
    Y = []

    for t in range(window_size, len(labels) - K, stride):

        if np.sum(labels[t - window_size:t][-5:]) > 0:
            continue

        window = features[t - window_size:t]
        movement = np.linalg.norm(np.diff(window, axis=0), axis=1).mean()
        future = labels[t:t + K]

        onset_idx = None
        for i in range(1, len(future)):
            if future[i] == 1 and future[i - 1] == 0:
                onset_idx = i
                break

        if onset_idx is None:
            y = 0
        elif onset_idx <= max_gap:
            y = 1
        else:
            continue

        if y == 1 and movement < 5:
            continue

        X.append(window)
        Y.append(y)

    return np.array(X), np.array(Y)

def add_features(X):
    velocity = np.diff(X, axis=1, prepend=X[:, :1, :])
    X = np.concatenate([X, velocity], axis=2)

    magnitude = np.linalg.norm(X, axis=2, keepdims=True)
    X = np.concatenate([X, magnitude], axis=2)

    return X

def prepare_dataloaders(X, Y, batch_size=64):
    from sklearn.model_selection import train_test_split
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    mean = X_train.mean(axis=(0, 1))
    std = X_train.std(axis=(0, 1)) + 1e-8

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    return train_loader, test_loader, mean, std