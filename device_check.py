import torch
import sys

print(sys.version)
print(torch.__version__)          # 查看 PyTorch 版本
print(torch.cuda.is_available())  # 应输出 True
print(torch.cuda.get_device_name(0))  # 输出显卡型号（如 NVIDIA RTX 3090）
