#!/usr/bin/env python
"""GPU configuration for NSGA-III using Apple Metal Performance Shaders"""
import os
import sys
import numpy as np
import platform
import time
import psutil

# Add a flag to control printing on import
QUIET_MODE = os.environ.get("NSGA_QUIET_MODE", "0") == "1"

# Check if running on macOS with Apple Silicon
IS_MACOS = platform.system() == "Darwin"
IS_APPLE_SILICON = IS_MACOS and ("arm" in platform.machine() or "Apple" in platform.processor())

# Initialize variables
HAS_GPU = False
MPS_ENABLED = False
HAS_TORCH = False
GPU_NAME = "None"

# Configure Metal for NumPy if possible
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "1"

# Check for Apple Silicon GPU
if IS_APPLE_SILICON:
    try:
        # Try to import PyTorch which has reliable MPS support
        import torch
        HAS_TORCH = True
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
                            break
                if not GPU_NAME or GPU_NAME == "None":
                    GPU_NAME = "Apple Silicon GPU"
            except:
                GPU_NAME = "Apple Silicon GPU"
    except ImportError:
        print("PyTorch not found. Install with: pip install torch")

def get_detailed_hardware_info():
    """Collect detailed hardware specifications"""
    hw_info = {
        'platform': platform.platform(),
        'system': platform.system(),
        'release': platform.release(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'cpu_count_physical': psutil.cpu_count(logical=False),
        'cpu_count_logical': psutil.cpu_count(logical=True),
        'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'gpu_name': GPU_NAME,
        'has_gpu': HAS_GPU,
        'has_torch': HAS_TORCH
    }
    
    # Apple Silicon specific info
    if IS_APPLE_SILICON:
        # Get detailed Apple Silicon info
        try:
            import subprocess
            result = subprocess.run(['sysctl', 'hw.model'], capture_output=True, text=True)
            if result.returncode == 0:
                hw_info['apple_model'] = result.stdout.split(':')[1].strip()
                
            # Get GPU core count
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if "Total Number of Cores" in line:
                        hw_info['gpu_cores'] = line.split(':')[1].strip()
        except:
            pass
    
    # PyTorch GPU info if available
    if HAS_TORCH:
        import torch
        hw_info['torch_version'] = torch.__version__
        hw_info['torch_mps_available'] = torch.backends.mps.is_available()
        hw_info['torch_mps_built'] = torch.backends.mps.is_built()
    
    return hw_info

class GPUTimer:
    """Timer for GPU operations with memory transfer tracking"""
    def __init__(self):
        self.start_time = 0
        self.end_time = 0
        self.transfer_times = []
        self.kernel_times = []
        
    def start(self):
        self.start_time = time.time()
        return self
        
    def end(self):
        self.end_time = time.time()
        return self
    
    def record_transfer(self, start, end):
        self.transfer_times.append(end - start)
        return end - start
    
    def record_kernel(self, start, end):
        self.kernel_times.append(end - start)
        return end - start
    
    @property
    def elapsed(self):
        return self.end_time - self.start_time
    
    @property
    def total_transfer_time(self):
        return sum(self.transfer_times)
    
    @property
    def total_kernel_time(self):
        return sum(self.kernel_times)

def print_gpu_info():
    """Print GPU configuration information"""
    if QUIET_MODE:
        return
        
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

# Only print on import if not in quiet mode
if not QUIET_MODE:
    print_gpu_info()