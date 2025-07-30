import torch
import cv2

# Check if PyTorch can use the GPU
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

# Check if OpenCV can use the GPU (often False with pip install)
print(f"OpenCV CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}")