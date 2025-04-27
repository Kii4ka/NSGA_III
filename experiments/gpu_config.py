#!/usr/bin/env python
"""GPU configuration for NSGA-III using Apple Metal Performance Shaders"""
import os
import sys
import numpy as np
import platform
import time

# Check if running on macOS with Apple Silicon
IS_MACOS = platform.system() == "Darwin"
IS_APPLE_SILICON = IS_MACOS and ("arm" in platform.machine() or "Apple" in platform.processor())

# Initialize variables
HAS_GPU = False
MPS_ENABLED = False
GPU_NAME = "None"

# Configure Metal for NumPy if possible
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "1"

# Check for Apple Silicon GPU
if IS_APPLE_SILICON:
    try:
        # Try to import PyTorch which has reliable MPS support
        import torch
        if torch.backends.mps.is_available():
            HAS_GPU = True
            MPS_ENABLED = True
            # Enable PyTorch MPS backend
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            
            # Detect GPU model
            import subprocess
            try:
                result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                       capture_output=True, text=True)
                output = result.stdout
                if "Apple M" in output:
                    for line in output.split('\n'):
                        if "Chipset Model" in line:
                            GPU_NAME = line.split(":")[1].strip()
                            if "GPU" not in GPU_NAME:
                                GPU_NAME += " (GPU)"
                            break
                if not GPU_NAME or GPU_NAME == "None":
                    GPU_NAME = "Apple Silicon GPU"
            except:
                GPU_NAME = "Apple Silicon GPU"
    except ImportError:
        print("PyTorch not found. Install with: pip install torch")

def print_gpu_info():
    """Print GPU configuration information"""
    print("NumPy configuration:")
    print(f"- NumPy version: {np.__version__}")
    print(f"- Is using MPS: {MPS_ENABLED}")
    if HAS_GPU:
        print(f"- Device: {GPU_NAME}")
        
        # Try to get more detailed GPU info if available
        try:
            import torch
            print(f"- PyTorch MPS available: {torch.backends.mps.is_available()}")
            print(f"- PyTorch MPS built: {torch.backends.mps.is_built()}")
        except ImportError:
            pass
    else:
        print("- No GPU acceleration available")

print_gpu_info()

class GPUTimer:
    """Timer for GPU operations with memory transfer tracking"""
    def __init__(self):
        self.start_time = 0
        self.end_time = 0
        self.transfer_times = []
        self.kernel_times = []
        
    def start(self):
        """Start the timer"""
        self.start_time = time.time()
        return self
        
    def end(self):
        """End the timer"""
        self.end_time = time.time()
        return self
    
    def record_transfer(self, start, end):
        """Record a memory transfer"""
        self.transfer_times.append(end - start)
        return end - start
    
    def record_kernel(self, start, end):
        """Record a kernel execution"""
        self.kernel_times.append(end - start)
        return end - start
    
    @property
    def elapsed(self):
        """Get elapsed time in seconds"""
        return self.end_time - self.start_time
    
    @property
    def total_transfer_time(self):
        """Get total transfer time in seconds"""
        return sum(self.transfer_times)
    
    @property
    def total_kernel_time(self):
        """Get total kernel execution time in seconds"""
        return sum(self.kernel_times)