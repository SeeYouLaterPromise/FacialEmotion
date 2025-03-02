import torchvision.models as models
import torch.nn as nn
import torch
import pandas as pd
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- 数据集定义 --------------------
class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        super(FaceDataset, self).__init__()
        self.root = root
        df = pd.read_csv(root + '\\dataset.csv', header=None)
        self.path = df[0].values
        self.label = df[1].values

    def __getitem__(self, item):
        face = cv2.imread(self.root + '\\' + self.path[item])
        face_resized = cv2.resize(face, (224, 224))  # 调整大小为 224x224
        face_tensor = torch.from_numpy(face_resized.transpose(2, 0, 1)).float() / 255.0  # 转换为 (C, H, W) 格式
        label = self.label[item]
        return face_tensor, label

    def __len__(self):
        return len(self.path)

# -------------------- 网络定义 --------------------
def get_vgg_model(num_classes=7):
    # 加载预训练的 VGG16 模型
    model = models.vgg16(pretrained=True)

    # 冻结卷积层参数（可选）
    for param in model.features.parameters():
        param.requires_grad = False

    # 修改最后一层全连接层以适应 7 分类任务
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model