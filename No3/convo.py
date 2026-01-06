import torch
import torch.nn as nn

# 問題1: 32 × 3 × 128 × 128 のテンソルを作成
print("===== 問題1 =====")
input_tensor = torch.ones(32, 3, 128, 128)
print("入力テンソルの形状:", input_tensor.shape)
print()

# 問題2: 出力が 32 × 64 × 126 × 126 となるように畳み込みを定義
print("===== 問題2 =====")
# 出力サイズ = (入力 - kernel + 2*padding) / stride + 1
# 126 = (128 - 3 + 2*padding) / 1 + 1
# 126 = 127 + 2*padding
# 2*padding = -1 => padding = 0（paddingはいらない）
conv2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=0, stride=1)
output2 = conv2(input_tensor)
print("出力テンソルの形状:", output2.shape)
print()

# 問題3: 出力が 32 × 256 × 64 × 64 となるように畳み込みを定義
print("===== 問題3 =====")
# 出力サイズ = (入力 - kernel + 2*padding) / stride + 1
# 64 = (128 - 3 + 2*padding) / stride + 1
# stride = 2 の場合を試す：
# 64 = (128 - 3 + 2*padding) / 2 + 1
# 63 = (128 - 3 + 2*padding) / 2
# 126 = 128 - 3 + 2*padding
# 2*padding = 1 => padding = 0.5 (不可)
# stride = 2, padding = 1：
# (128 - 3 + 2*1) / 2 + 1 = 127 / 2 + 1 = 63.5 + 1 = 64.5 (不可)
# stride = 2, padding = 0：
# (128 - 3 + 0) / 2 + 1 = 125 / 2 + 1 = 62.5 + 1 = 63.5 (不可)
# 正しい計算：stride = 2, padding = 1
# floor((128 + 2*1 - 3) / 2) + 1 = floor(127 / 2) + 1 = 63 + 1 = 64 ✓
conv3 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, padding=1, stride=2)
output3 = conv3(input_tensor)
print("出力テンソルの形状:", output3.shape)
print()

# 問題4: kernel_size = 5 で同様の結果が得られるように定義
print("===== 問題4 =====")
# 問題2の場合（出力 32 × 64 × 126 × 126）
# kernel_size = 5：126 = (128 - 5 + 2*padding) / 1 + 1
# 125 = 128 - 5 + 2*padding
# 2*padding = 2 => padding = 1
conv4_2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=1, stride=1)
output4_2 = conv4_2(input_tensor)
print("問題2相当（stride=1）の出力形状:", output4_2.shape)

# 問題3の場合（出力 32 × 256 × 64 × 64）
# kernel_size = 5, stride = 2：
# 64 = floor((128 + 2*padding - 5) / 2) + 1
# 63 = floor((128 + 2*padding - 5) / 2)
# 126 = 128 + 2*padding - 5
# 2*padding = 3 => padding = 1.5 (不可)
# padding = 2, stride = 2：
# floor((128 + 2*2 - 5) / 2) + 1 = floor(127 / 2) + 1 = 63 + 1 = 64 ✓
conv4_3 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=5, padding=2, stride=2)
output4_3 = conv4_3(input_tensor)
print("問題3相当（stride=2）の出力形状:", output4_3.shape)
print()

# 結果確認
print("===== 結果確認 =====")
print(f"問題2の出力: {output2.shape} (期待値: torch.Size([32, 64, 126, 126]))")
print(f"問題3の出力: {output3.shape} (期待値: torch.Size([32, 256, 64, 64]))")
print(f"問題4-2の出力: {output4_2.shape} (期待値: torch.Size([32, 64, 126, 126]))")
print(f"問題4-3の出力: {output4_3.shape} (期待値: torch.Size([32, 256, 64, 64]))")
