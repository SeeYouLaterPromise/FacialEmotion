import torch

# 检查CUDA是否可用
if torch.cuda.is_available():
    print("CUDA is available! PyTorch can use GPUs.")
else:
    print("CUDA is not available. PyTorch will use CPUs.")

# import torch
# 获取当前默认设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: {device}")
# 获取GPU数量
gpu_count = torch.cuda.device_count()
print(f"Number of GPUs: {gpu_count}")

# 获取每个GPU的名称
for i in range(gpu_count):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    tensor = torch.randn(100, 100).to(device)
    result = tensor * tensor
    print("Tensor operation on GPU completed successfully.")
else:
    print("CUDA is not available. Cannot perform GPU operations.")
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())