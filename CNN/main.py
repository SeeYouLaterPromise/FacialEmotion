import json
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from FaceDataset import FaceDataset
from FaceCNN import FaceCNN
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- 验证函数（带进度条）--------------------
def validate(model, dataset, batch_size):
    val_loader = data.DataLoader(dataset, batch_size, shuffle=False)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=True):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# -------------------- 训练函数（带进度条和可视化）--------------------
def train(model, train_dataset, val_dataset, batch_size, epochs, lr, weight_decay):
    train_loader = data.DataLoader(train_dataset, batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 使用 ReduceLROnPlateau 调度器，根据验证集的损失动态调整学习率
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)

    # 训练历史记录
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0  # 用于保存表现最佳的模型

    # 主训练循环
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # 使用 tqdm 包装训练数据加载器
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=True)
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

        # 如果验证准确率提高，保存模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            now = datetime.datetime.now()
            current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, 'model/best_face_cnn.pth')
            print(f"Best model saved at epoch {epoch + 1}")

        # 动态调整学习率
        scheduler.step(avg_loss)

    return model, history


# -------------------- 绘制损失和准确率曲线 --------------------
def plot_history(history):
    # 绘制损失曲线
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train Loss Curve')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()


# -------------------- 主函数 --------------------
def main():
    # 加载数据
    train_dataset = FaceDataset(root='../data/fer2013/train')
    val_dataset = FaceDataset(root='../data/fer2013/val')

    # 训练参数
    BATCH_SIZE = 128
    EPOCHS = 500
    LR = 1e-3
    WEIGHT_DECAY = 0

    # 初始化模型
    model = FaceCNN().to(device)
    try:
        model.load_state_dict(torch.load("2025-03-01_18-12-22_best_face_cnn.pth"))
    except Exception as e:
        print(f"Error loading model: {e}")

    # 开始训练
    model, history = train(model, train_dataset, val_dataset, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY)

    # 保存训练历史
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    with open(current_time + "_history.json", 'w') as f:
        json.dump(history, f)
    print("History saved!")

    # 绘制损失和准确率曲线
    plot_history(history)


if __name__ == '__main__':
    main()