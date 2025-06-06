import pandas as pd
import torch
import torch.nn as nn
import os

# 데이터 로드
df = pd.read_csv("backend/model/fight_data.csv")

X = df[["height_diff", "weight_diff", "reach_diff"]].values
y = df[["winner"]].values

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# 모델 정의
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# 학습
model = MLP()
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1000):
    model.train()
    pred = model(X)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.4f}")

# 모델 저장
os.makedirs("backend/model", exist_ok=True)
torch.save(model.state_dict(), "backend/model/model.pt")
print("✅ model.pt 저장 완료")
