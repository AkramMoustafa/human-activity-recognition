import os
import glob
import zipfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import joblib

from src.config_loader import load_config
from src.dataset.time_to_gesture_dataset import TimeToGestureDataset
from src.models.time_to_gesture_regressor import TimeToGestureRegressor
from src.train_utils import set_seed, get_device, save_model

cfg = load_config()
global_cfg = cfg["global"]
task_cfg = cfg["time_to_gesture"]

ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT, "data")
ZIP_PATH = os.path.join(DATA_DIR, "snaptic_logs.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "logs")

SCALER_PATH = os.path.join(ROOT, global_cfg["scaler_path"])
MODEL_SAVE_PATH = os.path.join(
    ROOT,
    global_cfg["save_dir"],
    task_cfg["model_name"],
)
os.makedirs(
    os.path.join(ROOT, global_cfg["save_dir"]),
    exist_ok=True,
)

if not os.path.exists(EXTRACT_DIR):
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)
    print("Extracted log files.")
else:
    print("ZIP already extracted.")

print("Loading scaler...")
scaler = joblib.load(SCALER_PATH)

csv_files = glob.glob(os.path.join(EXTRACT_DIR, "*.csv"))
print(f"Found {len(csv_files)} CSV files.")

gesture_to_idx = cfg["global"]["gesture_map"]

all_datasets = []

for file in csv_files:
    df = pd.read_csv(file)
    df["gesture"] = df["gesture"].ffill().fillna("no_gesture")

    imu = df[
        ["acc_x", "acc_y", "acc_z",
         "gyro_x", "gyro_y", "gyro_z"]
    ].values

    imu = scaler.transform(imu)
    gest = df["gesture"].map(gesture_to_idx).values

    ds = TimeToGestureDataset(
        imu_data=imu,
        gesture_labels=gest,
        window_size=task_cfg["window_size"],
        max_time=task_cfg["max_time"],
    )

    all_datasets.append(ds)
from torch.utils.data import ConcatDataset, random_split

# combine all datasets
full_dataset = ConcatDataset(all_datasets)

val_ratio = 0.2
val_size = int(len(full_dataset) * val_ratio)
train_size = len(full_dataset) - val_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(global_cfg["seed"])
)

train_loader = DataLoader(
    train_dataset,
    batch_size=task_cfg["batch_size"],
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=task_cfg["batch_size"],
    shuffle=False
)

all_targets = []

all_targets = []

for ds in all_datasets:
    for i in range(len(ds)):
        _, y = ds[i]
        all_targets.append(y.item())

set_seed(global_cfg["seed"])
device = get_device()

model = TimeToGestureRegressor(
    input_dim=global_cfg["input_dim"],
    hidden_dim=task_cfg["hidden_dim"],
    num_layers=task_cfg["num_layers"],
).to(device)

model.eval()

test_loader = DataLoader(
    all_datasets[0],
    batch_size=task_cfg["batch_size"],
    shuffle=False
)

with torch.no_grad():
    x_batch, _ = next(iter(test_loader))
    x_batch = x_batch.to(device)

    enc_out, _ = model.encoder(x_batch)
    print("Encoder std:", enc_out.std().item())

    weights = torch.softmax(model.attn(enc_out), dim=1)
    print("Attention std:", weights.std().item())

    h_attn = (weights * enc_out).sum(dim=1)
    print("Context std:", h_attn.std().item())

    out = model(x_batch)
    print("Output std:", out.std().item())

model.train()

criterion = torch.nn.SmoothL1Loss(reduction="none")
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=task_cfg["learning_rate"],
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

epochs = task_cfg["epochs"]

best_val_loss = float("inf")
patience = 7
patience_counter = 0
best_model_state = None

for epoch in range(50):

    model.train()
    total_loss = 0
    epoch_grad = 0.0
    batch_count = 0
    all_preds = []
    all_targets = []
    for x, y in train_loader:

      x, y = x.to(device), y.to(device)

      optimizer.zero_grad()
      pred = model(x)
      loss = criterion(pred, y).mean()
      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

      batch_grad = 0
      param_count = 0
      for param in model.parameters():
          if param.grad is not None:
              batch_grad += param.grad.abs().mean().item()
              param_count += 1

      if param_count > 0:
        epoch_grad += batch_grad / param_count
      batch_count += 1

      optimizer.step()

      total_loss += loss.item()
      all_preds.append(pred.detach().cpu())
      all_targets.append(y.detach().cpu())
    model.eval()

    val_preds = []
    val_targets = []
    val_loss = 0

    with torch.no_grad():
        for x, y in val_loader:

            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = criterion(pred, y).mean()

            val_loss += loss.item()

            val_preds.append(pred.cpu())
            val_targets.append(y.cpu())

    val_preds = torch.cat(val_preds)
    val_targets = torch.cat(val_targets)

    val_mse = torch.mean((val_preds - val_targets) ** 2)
    val_corr = torch.corrcoef(
        torch.stack([val_preds.view(-1), val_targets.view(-1)])
    )[0, 1]

    # ---- epoch metrics ----
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    baseline = all_targets.mean()
    baseline_pred = torch.ones_like(all_targets) * baseline

    baseline_mse = torch.mean((baseline_pred - all_targets) ** 2)
    model_mse = torch.mean((all_preds - all_targets) ** 2)
    train_corr = torch.corrcoef(
    torch.stack([all_preds.view(-1), all_targets.view(-1)])
    )[0, 1]
    print(f"\nEpoch {epoch+1}/{epochs}")

    print("Avg grad magnitude:", epoch_grad / batch_count)

    print("Train loss:", total_loss / len(train_loader))
    print(f"Train MSE: {model_mse.item():.6f}")
    print(f"Train Corr: {train_corr.item():.4f}")

    print(f"Val loss: {val_loss / len(val_loader):.6f}")
    print(f"Val MSE:  {val_mse.item():.6f}")
    print(f"Val Corr: {val_corr.item():.4f}")
    val_loss_epoch = val_loss / len(val_loader)
    scheduler.step(val_loss_epoch)
    # Early stopping check
    if val_loss_epoch < best_val_loss:
        best_val_loss = val_loss_epoch
        patience_counter = 0
        best_model_state = model.state_dict()
        print("✓ New best validation loss")
    else:
        patience_counter += 1
        print(f"No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print("\nEarly stopping triggered.")
            break

if best_model_state is not None:
    model.load_state_dict(best_model_state)

save_model(model, MODEL_SAVE_PATH)
print(f"Best model saved → {MODEL_SAVE_PATH}")
