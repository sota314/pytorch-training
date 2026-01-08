import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import cifar_datasets
from model import CNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('gpu is available')


def run_epoch(model, dataloader, criterion, optimizer=None, is_train=True):
    model.train() if is_train else model.eval()

    total_loss = 0.
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            _, predict = outputs.max(1)
            total += labels.size(0)
            correct += predict.eq(labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion):
    model.eval()
    
    total_loss = 0.
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predict = outputs.max(1)
            total += labels.size(0)
            correct += predict.eq(labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


# 保存されたモデルのパス
model_path = './models/cifar_cnn.pth'

_, test_data = cifar_datasets()
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

model = CNN()
model = model.to(device)

# 保存されたモデルを読み込み
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

criterion = nn.CrossEntropyLoss()
print(f"Model loaded from {model_path}")
test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)')
