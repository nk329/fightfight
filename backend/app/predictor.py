import torch
import torch.nn as nn

# 모델 정의 (입력 크기: 3)
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

# 모델 불러오기
def load_model(model_path="backend/model/model.pt"):
    model = MLP()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

# 예측 함수
def predict_winner(model, height_diff, weight_diff, reach_diff):
    """
    입력값:
      height_diff: float
      weight_diff: float
      reach_diff: float

    출력값:
      A가 이길 확률 (0.0 ~ 1.0)
    """
    input_tensor = torch.tensor([[height_diff, weight_diff, reach_diff]], dtype=torch.float32)

    with torch.no_grad():
        prob = model(input_tensor).item()

    return prob

# 테스트
if __name__ == "__main__":
    model = load_model()
    result = predict_winner(model, height_diff=5.0, weight_diff=0.0, reach_diff=7.0)
    print(f"Player A 승리 확률: {result:.2f}")
