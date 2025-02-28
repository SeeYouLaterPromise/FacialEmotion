import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm  # 进度条库

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 参数初始化
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.04)


# -------------------- 数据集定义 --------------------
class FaceDataset(data.Dataset):
    def __init__(self, root):
        super(FaceDataset, self).__init__()
        self.root = root
        df = pd.read_csv(root + '\\dataset.csv', header=None)
        self.path = df[0].values
        self.label = df[1].values

    def __getitem__(self, item):
        face = cv2.imread(self.root + '\\' + self.path[item])
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_hist = cv2.equalizeHist(face_gray)
        face_normalized = face_hist.reshape(1, 48, 48) / 255.0
        face_tensor = torch.from_numpy(face_normalized).float()
        label = self.label[item]
        return face_tensor, label

    def __len__(self):
        return len(self.path)


# -------------------- 网络定义 --------------------
class FaceCNN(nn.Module):
    def __init__(self):
        super(FaceCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.RReLU(), nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.RReLU(), nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.RReLU(), nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(256 * 6 * 6, 4096), nn.RReLU(),
            nn.Dropout(0.5), nn.Linear(4096, 1024), nn.RReLU(),
            nn.Linear(1024, 256), nn.RReLU(), nn.Linear(256, 7)
        )
        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.conv3.apply(gaussian_weights_init)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# -------------------- 验证函数（带进度条）--------------------
def validate(model, dataset, batch_size):
    val_loader = data.DataLoader(dataset, batch_size, shuffle=False)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# -------------------- 训练函数（带进度条和可视化）--------------------
def train(train_dataset, val_dataset, batch_size, epochs, lr, weight_decay):
    train_loader = data.DataLoader(train_dataset, batch_size, shuffle=True)
    model = FaceCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 训练历史记录
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    # 主训练循环
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # 使用tqdm包装训练数据加载器
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False)
        for images, labels in train_progress:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # 实时更新进度条描述
            train_progress.set_postfix({'loss': f"{loss.item():.4f}"})

        # 计算指标
        avg_loss = running_loss / len(train_loader)
        train_acc = validate(model, train_dataset, batch_size)
        val_acc = validate(model, val_dataset, batch_size)

        # 记录历史
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # 打印结果
        print(f"Epoch {epoch + 1:03d}/{epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), 'face_cnn.pth')
    return model, history


# -------------------- 主函数 --------------------
def main():
    # 加载数据
    train_dataset = FaceDataset(root='train')
    val_dataset = FaceDataset(root='val')

    # 训练参数
    BATCH_SIZE = 128
    EPOCHS = 100
    LR = 1e-3
    WEIGHT_DECAY = 0

    # 开始训练
    model, history = train(train_dataset, val_dataset, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY)


if __name__ == '__main__':
    main()