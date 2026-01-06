import torch
import torch.nn as nn

print("===== 問題1 =====")
# 問題1: 入力用のテンソルとして、32 × 1024 のテンソルを定義
input_tensor = torch.ones(32, 1024)
print("入力テンソルの形状:", input_tensor.shape)
print()

print("===== 問題2 =====")
# 問題2: 出力が 32 × 256 となるように全結合層を定義して、適用
# Linear(入力サイズ, 出力サイズ)
fc2 = nn.Linear(in_features=1024, out_features=256)
output2 = fc2(input_tensor)
print("出力テンソルの形状:", output2.shape)
print("出力の例（最初の5要素）:", output2[0, :5])
print()

print("===== 問題3 =====")
# 問題3: 出力が 32 × 2048 となるように全結合層を定義して、適用
fc3 = nn.Linear(in_features=1024, out_features=2048)
output3 = fc3(input_tensor)
print("出力テンソルの形状:", output3.shape)
print("出力の例（最初の5要素）:", output3[0, :5])
print()

print("===== 結果確認 =====")
print(f"入力: {input_tensor.shape} (期待値: torch.Size([32, 1024]))")
print(f"問題2の出力: {output2.shape} (期待値: torch.Size([32, 256]))")
print(f"問題3の出力: {output3.shape} (期待値: torch.Size([32, 2048]))")
