import torch
from models import MyModel

print("===== モデルテスト =====")

# モデルの作成
model = MyModel()

# 入力テンソルの作成（32 × 3 × 256 × 256）
input_tensor = torch.ones(32, 3, 256, 256)
print(f"入力テンソルの形状: {input_tensor.shape}")
print()

# モデルの実行
output = model(input_tensor)

print()
print("===== 最終出力 =====")
print(f"出力テンソルの形状: {output.shape}")
print(f"期待値: torch.Size([32, 64])")
