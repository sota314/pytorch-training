from torch import nn
from torchinfo import summary

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.l1 = nn.Linear(in_features=8*8*16, out_features=128)
        self.bn3 = nn.BatchNorm1d(128)
        self.l2 = nn.Linear(in_features=128, out_features=64)
        self.bn4 = nn.BatchNorm1d(64)
        self.l3 = nn.Linear(in_features=64, out_features=32)
        self.bn5 = nn.BatchNorm1d(32)
        self.l4 = nn.Linear(in_features=32, out_features=10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size()[0], -1)
        x = self.relu(self.bn3(self.l1(x)))
        x = self.relu(self.bn4(self.l2(x)))
        x = self.relu(self.bn5(self.l3(x)))
        x = self.l4(x)
        return x

if __name__=='__main__':
    model = CNN()
    summary(model, input_size=(64,3,32,32), col_names=["output_size", "num_params"], verbose=2)

