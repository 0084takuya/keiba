import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np

# データ読み込み
X = pd.read_pickle('keiba_sanshutsu/deep_rl_analysis/X.pkl')
y = pd.read_pickle('keiba_sanshutsu/deep_rl_analysis/y.pkl')

# numpy変換
X_np = X.values.astype(np.float32)
y_np = y.values.astype(np.float32)

# PyTorch Tensor化
X_tensor = torch.tensor(X_np)
y_tensor = torch.tensor(y_np).unsqueeze(1)

dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256)

# DNNモデル定義
class Predictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

model = Predictor(X.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学習
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    print(f"Epoch {epoch+1}, train loss: {total_loss/len(train_loader.dataset):.4f}")

# 評価
model.eval()
correct = 0
n = 0
with torch.no_grad():
    for xb, yb in val_loader:
        pred = model(xb)
        pred_label = (pred > 0.5).float()
        correct += (pred_label == yb).sum().item()
        n += len(xb)
print(f"Validation accuracy: {correct/n:.3f}")

# 特徴量重要度（重みの絶対値で簡易評価）
with torch.no_grad():
    importance = model.net[0].weight.abs().sum(dim=0).cpu().numpy()
    feature_importance = sorted(zip(X.columns, importance), key=lambda x: -x[1])
    print("特徴量重要度（上位10件）:")
    for name, score in feature_importance[:10]:
        print(f"{name}: {score:.3f}") 