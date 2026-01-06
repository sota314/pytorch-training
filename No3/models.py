import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Convolution: in_channels=3, out_channels=256, kernel_size=5, stride=8
        # padding は自分で調整（padding=2 にしてみる）
        self.conv = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=5, stride=8, padding=2)
        
        # Batch Normalization
        self.bn = nn.BatchNorm2d(256)
        
        # ReLU
        self.relu = nn.ReLU()
        
        # Linear層
        # 入力画像サイズが 32 × 256 × 256 とすると、
        # Convolution出力: 256 × floor((256 + 2*2 - 5) / 8 + 1) × floor((256 + 2*2 - 5) / 8 + 1)
        # = 256 × floor(253 / 8 + 1) × floor(253 / 8 + 1)
        # = 256 × floor(32.625) × floor(32.625)
        # = 256 × 32 × 32 = 262144
        # Linear: 262144 -> 64
        self.linear = nn.Linear(256 * 32 * 32, 64)
    
    def forward(self, x):
        # Convolution
        x = self.conv(x)
        print(f"After Conv: {x.shape}")
        
        # Batch Normalization
        x = self.bn(x)
        print(f"After BN: {x.shape}")
        
        # ReLU
        x = self.relu(x)
        print(f"After ReLU: {x.shape}")
        
        # Flatten
        x = x.view(x.size(0), -1)
        print(f"After Flatten: {x.shape}")
        
        # Linear
        x = self.linear(x)
        print(f"After Linear: {x.shape}")
        
        return x
