import torch

cuda_available = torch.cuda.is_available()
torch_version = torch.__version__

print(f"CUDA available: {cuda_available}")
print(f"PyTorch version: {torch_version}")

with open("cuda_check.txt", "w") as f:
    f.write(f"CUDA available: {cuda_available}\n")
    f.write(f"PyTorch version: {torch_version}\n")
