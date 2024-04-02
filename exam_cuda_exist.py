import torch

# 检查CUDA是否可用
cuda_available = torch.cuda.is_available()

# 获取当前CUDA设备的编号和名称（如果CUDA可用）
cuda_device_count = torch.cuda.device_count()
cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else "CUDA设备不可用"

print(f"{cuda_available}, {cuda_device_count}, {cuda_device_name}")


# 查看CUDA和CUDA Toolkit的版本

cuda_version = torch.version.cuda  # CUDA的版本
cudnn_version = torch.backends.cudnn.version()  # cuDNN的版本

print(f"CUDA版本: {cuda_version}, cuDNN版本: {cudnn_version}")