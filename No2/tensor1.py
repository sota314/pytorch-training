import numpy as np
import torch

data = np.array([
    [85, 78], [67, 82], [92, 88], [75, 70], [60, 64],
    [70, 68], [77, 72], [85, 90], [60, 65], [78, 76],
    [80, 84], [88, 87], [66, 68], [72, 73], [64, 60]
])

# 問題1: Numpy配列をTensorに変換
tensor_data = torch.tensor(data, dtype=torch.float)
print("【問題1】Tensorに変換")
print(tensor_data)
print()

# 問題2: 2科目、3クラス、各クラス5人に並び替える
# 現在のデータ: (15, 2) -> (3, 5, 2) に変換後、(2, 3, 5) に並び替え
reshaped_tensor = tensor_data.reshape(3, 5, 2)  # (3, 5, 2): クラス、人数、科目
reshaped_tensor = reshaped_tensor.permute(2, 0, 1)  # (2, 3, 5): 科目、クラス、人数
print("【問題2】2科目、3クラス、各クラス5人に並び替え")
print("形状:", reshaped_tensor.shape)
print(reshaped_tensor)
print()

# 問題3: クラスごと、個々人の2科目合計点
# (2, 3, 5) -> (3, 5) に変換（dim=0の科目を合計）
sum_per_student = reshaped_tensor.sum(dim=0)  # 2科目を合計
print("【問題3】クラスごと、個々人の2科目合計点")
print(sum_per_student)
print()

# 問題4: クラスごと、2科目合計点の平均
# (3, 5) の各クラス（dim=1）の5人の平均
mean_per_class = sum_per_student.mean(dim=1)
print("【問題4】クラスごと、2科目合計点の平均")
print(mean_per_class)
print()

# 問題5: torch.meanを使わずに4と同じ値を導出
# 合計を人数で割る
total_sum_per_class = sum_per_student.sum(dim=1)
num_students = sum_per_student.shape[1]
mean_without_torch = total_sum_per_class / num_students
print("【問題5】torch.meanを使わずに4と同じもの")
print(mean_without_torch)
print()

# 確認：問題4と問題5が同じ値か
print("【確認】問題4と問題5が同じか：")
print(torch.allclose(mean_per_class, mean_without_torch))

# tensor_data = torch.from_numpy(data)
# print("NumPy Array:\n", data)
# print("PyTorch Tensor:\n", tensor_data)


