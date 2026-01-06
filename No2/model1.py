import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, mytensor, elem_add, elem_multiply):
        super().__init__()
        # インスタンス変数として、mytensor（Tensor型）、elem_add、elem_multiplyを持つ
        self.mytensor = mytensor
        self.elem_add = elem_add
        self.elem_multiply = elem_multiply
    
    def forward(self, x):
        # 入力に mytensor を加算
        x = x + self.mytensor
        print("===== 操作2: mytensor を加算後 =====")
        print(x)
        print()
        
        # テンソル全体に elem_add を加算
        x = x + self.elem_add
        print("===== 操作3: elem_add を加算後 =====")
        print(x)
        print()
        
        # テンソル全体に elem_multiply を乗算
        x = x * self.elem_multiply
        print("===== 操作4: elem_multiply を乗算後 =====")
        print(x)
        print()
        
        return x

if __name__ == "__main__":
    # テンソルの作成
    mytensor = torch.tensor([[2., 2.], [2., 2.]])
    elem_add = 4.
    elem_multiply = 6.
    
    # モデルの作成
    model = MyModel(mytensor, elem_add, elem_multiply)
    
    # 入力テンソルの作成
    input_tensor = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    
    print("===== 入力テンソル =====")
    print(input_tensor)
    print()
    
    # モデルの呼び出し
    output = model(input_tensor)
    
    print("===== 最終出力 =====")
    print(output)
