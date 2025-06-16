import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os

# 현재 스크립트 기준으로 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "fight_data.csv")
model_path = os.path.join(BASE_DIR, "model.pt")

# 데이터 로드 및 대칭 데이터 생성
df = pd.read_csv(csv_path)

# 원본: A - B, 결과는 A 기준 승리 확률
X_orig = df[["height_diff", "weight_diff", "reach_diff"]].values
y_orig = df[["winner"]].values

# 대칭: B - A, 결과는 1 - winner
X_flip = -X_orig
y_flip = 1 - y_orig

# 데이터 합치기
X_total = np.vstack([X_orig, X_flip])
y_total = np.vstack([y_orig, y_flip])

# 텐서 변환
X = torch.tensor(X_total, dtype=torch.float32)
y = torch.tensor(y_total, dtype=torch.float32)

# MLP 모델 정의
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
torch.save(model.state_dict(), model_path)
print(f"모델 저장 완료: {model_path}")
