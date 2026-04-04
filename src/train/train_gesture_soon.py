import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from src.dataset.gesture_soon_dataset import load_gesture_soon_dataset
from src.models.gesture_soon_classifier import LSTMModel

X, Y = load_gesture_soon_dataset()

print("X shape:", X.shape)
print("Y shape:", Y.shape)

X_feat = X.copy()

velocity = np.diff(X_feat, axis=1, prepend=X_feat[:, :1, :])
X_feat = np.concatenate([X_feat, velocity], axis=2)

magnitude = np.linalg.norm(X_feat, axis=2, keepdims=True)
X_feat = np.concatenate([X_feat, magnitude], axis=2)

X_train, X_test, y_train, y_test = train_test_split(
    X_feat, Y, test_size=0.2, random_state=42, stratify=Y
)

mean = X_train.mean(axis=(0, 1))
std = X_train.std(axis=(0, 1)) + 1e-8

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

input_size = X_feat.shape[2]
model = LSTMModel(input_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

pos_count = y_train.sum().item()
neg_count = len(y_train) - pos_count

pos_weight = torch.tensor([neg_count / (pos_count + 1e-8)]).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 20

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)

        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        total_loss += loss.item() * x_batch.size(0)

    total_loss /= len(train_loader.dataset)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            logits = model(x_batch)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.3).float()

            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    acc = correct / total

    print(f"Epoch {epoch+1}")
    print(f"Loss: {total_loss:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    print("-" * 40)

def predict_window(model, window, mean, std):
    model.eval()

    # 🔥 SAME FEATURE ENGINEERING
    velocity = np.diff(window, axis=0, prepend=window[:1])
    window = np.concatenate([window, velocity], axis=1)

    magnitude = np.linalg.norm(window, axis=1, keepdims=True)
    window = np.concatenate([window, magnitude], axis=1)

    window = (window - mean) / std

    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)[0]
        prob = torch.sigmoid(logits).item()

    return prob

sample_idx = 0
prob = predict_window(model, X[sample_idx], mean, std)

print("\nPrediction probability:", prob)
print("Predicted class:", 1 if prob > 0.5 else 0)
print("True label:", Y[sample_idx])
print("Positive ratio:", Y.mean())