#!/usr/bin/env python3
"""
MLP 购房预测训练 — Ames Housing Dataset
使用 PyTorch 实现 MLP，MSE 损失函数，随机初始化网络参数。
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os

# ---------- 0. 全局设置 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ---------- 1. 数据加载 ----------
train_raw = pd.read_csv("dataset/train.csv")
test_raw  = pd.read_csv("dataset/test.csv")

# 合并 train+test 以便统一预处理
train_id = train_raw["Id"]
test_id  = test_raw["Id"]
y_full   = train_raw["SalePrice"]
train_raw.drop(columns=["Id", "SalePrice"], inplace=True)
test_raw.drop(columns=["Id"], inplace=True)

all_data = pd.concat([train_raw, test_raw], axis=0, ignore_index=True)

# ---------- 2. 缺失值处理 ----------
# 数值列：用中位数填充
num_cols = list(all_data.select_dtypes(include=["int64", "float64"]).columns)
for c in num_cols:
    all_data[c] = all_data[c].fillna(all_data[c].median())

# 类别列：用 "None" 填充
cat_cols = list(all_data.select_dtypes(include=["object"]).columns)
for c in cat_cols:
    all_data[c] = all_data[c].fillna("None")

# ---------- 3. 特征工程 ----------
# 3a. 类别特征 → Label Encoding
for c in cat_cols:
    le = LabelEncoder()
    all_data[c] = le.fit_transform(all_data[c].astype(str))

# 3b. 所有特征标准化
scaler = StandardScaler()
all_data_scaled = scaler.fit_transform(all_data)

# 拆分回 train / test
X_train_all = all_data_scaled[:len(train_raw)]
X_test       = all_data_scaled[len(train_raw):]

# 对 SalePrice 做 log 变换（使分布更接近正态，利于 MSE 优化）
y_log = np.log1p(y_full.values)  # log(1+x)

# ---------- 4. 训练 / 验证 划分 ----------
X_train, X_val, y_train, y_val = train_test_split(
    X_train_all, y_log, test_size=0.2, random_state=RANDOM_SEED
)

# 转为 PyTorch Tensor
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val,   dtype=torch.float32).view(-1, 1)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)

# DataLoader
train_ds = TensorDataset(X_train_t, y_train_t)
val_ds   = TensorDataset(X_val_t,   y_val_t)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)

input_dim = X_train.shape[1]
print(f"Input dimension: {input_dim}")
print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")


# ---------- 5. 模型定义 ----------
class MLP(nn.Module):
    """多层感知机，用于房价回归预测。"""

    def __init__(self, input_dim, hidden_dims, output_dim=1, dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# 超参数
HIDDEN_DIMS = [256, 128, 64, 32]  # 四层隐藏层
DROPOUT = 0.2
LR = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 500

model = MLP(input_dim, HIDDEN_DIMS, dropout=DROPOUT).to(device)
print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# 损失函数 & 优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=20
)


# ---------- 6. 训练 ----------
def rmse_metric(y_pred, y_true):
    """Root Mean Squared Error（log 空间）。"""
    return torch.sqrt(criterion(y_pred, y_true))


train_losses, val_losses = [], []
best_val_loss = float("inf")
best_model_state = None
patience_counter = 0
EARLY_STOP_PATIENCE = 80

for epoch in range(1, EPOCHS + 1):
    # ---- Train ----
    model.train()
    train_loss_sum = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(Xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item() * Xb.size(0)

    avg_train_loss = train_loss_sum / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    # ---- Valid ----
    model.eval()
    val_loss_sum = 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            pred = model(Xb)
            loss = criterion(pred, yb)
            val_loss_sum += loss.item() * Xb.size(0)

    avg_val_loss = val_loss_sum / len(val_loader.dataset)
    val_losses.append(avg_val_loss)

    scheduler.step(avg_val_loss)

    # 打印进度
    if epoch % 50 == 0 or epoch == 1:
        train_rmse = np.sqrt(avg_train_loss)
        val_rmse   = np.sqrt(avg_val_loss)
        print(f"Epoch {epoch:4d}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
              f"Train RMSE (log): {train_rmse:.4f} | Val RMSE (log): {val_rmse:.4f}")

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= EARLY_STOP_PATIENCE:
        print(f"Early stopping at epoch {epoch}")
        break

# 加载最佳模型
model.load_state_dict(best_model_state)
print(f"Best val loss: {best_val_loss:.6f}  (RMSE log: {np.sqrt(best_val_loss):.4f})")


# ---------- 7. 结果可视化 ----------
os.makedirs("figures", exist_ok=True)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses,   label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss (log space)")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# 在验证集上展示预测 vs 真实
model.eval()
with torch.no_grad():
    val_pred = model(X_val_t.to(device)).cpu().numpy().flatten()
val_true = y_val

# 换回原始 SalePrice 空间
val_pred_price = np.expm1(val_pred)
val_true_price = np.expm1(val_true)

plt.scatter(val_true_price, val_pred_price, alpha=0.5, s=8)
plt.plot([val_true_price.min(), val_true_price.max()],
         [val_true_price.min(), val_true_price.max()], "r--", lw=1)
plt.xlabel("True SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title(f"Validation Set Predictions (n={len(val_true)})")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures/training_curve.png", dpi=150)
print("Figure saved to figures/training_curve.png")


# ---------- 8. 测试集预测 ----------
model.eval()
with torch.no_grad():
    test_pred_log = model(X_test_t.to(device)).cpu().numpy().flatten()

test_pred_price = np.expm1(test_pred_log)
test_pred_price = np.clip(test_pred_price, 0, None)  # 确保非负

submission = pd.DataFrame({"Id": test_id, "SalePrice": test_pred_price})
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
print(f"Prediction stats: mean={test_pred_price.mean():.1f}, "
      f"min={test_pred_price.min():.1f}, max={test_pred_price.max():.1f}")
