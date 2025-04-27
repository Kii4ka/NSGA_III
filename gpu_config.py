import os
import numpy as np

# Enable Metal Performance Shaders for NumPy operations (M-series acceleration)
os.environ['ACCELERATE_ENABLE_METAL'] = '1'

# Print GPU info
print("NumPy configuration:")
print(f"- NumPy version: {np.__version__}")
print(f"- Is using MPS: {'ACCELERATE_ENABLE_METAL' in os.environ}")
print(f"- Device: Apple M4 Max (40 GPU cores)")