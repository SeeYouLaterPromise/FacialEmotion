import torchvision.models as models
import torch.nn as nn
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
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