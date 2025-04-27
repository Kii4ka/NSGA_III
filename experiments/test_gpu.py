#!/usr/bin/env python
from gpu_config import HAS_GPU, MPS_ENABLED, GPU_NAME, print_gpu_info

print("\nGPU Configuration Test:")
print(f"Has GPU: {HAS_GPU}")
print(f"MPS Enabled: {MPS_ENABLED}")
print(f"GPU Name: {GPU_NAME}")

# Test PyTorch MPS
try:
    import torch
    print("\nPyTorch MPS Test:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    # Create a sample tensor on MPS device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        x = torch.rand(5, 3, device=device)
        print(f"Sample tensor on MPS device: {x.device}")
        print("MPS is working correctly!")
    else:
        print("MPS is not available on this device")
except ImportError:
    print("PyTorch not installed. Install with: pip install torch")