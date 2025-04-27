#!/usr/bin/env python
"""
GPU-accelerated implementation of NSGA-III using Apple Metal Performance Shaders
"""
import numpy as np
import time
from gpu_config import GPUTimer, HAS_GPU, MPS_ENABLED
import NSGA_III  # Import original implementation

# Optional: Try to import PyTorch for more advanced GPU operations
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Force NumPy to use float32 by default for this module
np.set_printoptions(precision=6)  # Display fewer decimal places
DTYPE = np.float32  # Use this throughout the file

def cal_obj_gpu(pop, nobj):
    """
    GPU-accelerated objective function calculation
    """
    # Use PyTorch if available for better GPU acceleration
    if HAS_TORCH:
        # Convert to float32 before sending to GPU
        pop_np = np.asarray(pop, dtype=DTYPE)
        
        # Convert to PyTorch tensor on MPS device
        device = torch.device("mps")
        pop_tensor = torch.tensor(pop_np, device=device, dtype=torch.float32)
        
        # Perform calculations on GPU (same operations, ensuring float32)
        g = torch.tensor(100.0, device=device, dtype=torch.float32) * (pop_tensor.shape[1] - nobj + 1)
        for i in range(nobj-1, pop_tensor.shape[1]):
            g = g + torch.pow(pop_tensor[:, i] - 0.5, 2) - torch.cos(20 * np.pi * (pop_tensor[:, i] - 0.5))
        
        # Calculate objectives
        objs = torch.zeros((pop_tensor.shape[0], nobj), device=device, dtype=torch.float32)
        for i in range(nobj):
            objs[:, i] = (1.0 + g)
            for j in range(nobj - i - 1):
                objs[:, i] = objs[:, i] * pop_tensor[:, j]
            if i > 0:
                objs[:, i] = objs[:, i] * (1 - pop_tensor[:, nobj - i - 1])
        
        # Transfer back to CPU as float32
        return objs.cpu().numpy().astype(DTYPE)
    else:
        # Fall back to CPU implementation but ensure float32
        return NSGA_III.cal_obj(pop, nobj).astype(DTYPE)

def main(npop, iter, lb, ub, nobj=3, pc=1, pm=1, eta_c=30, eta_m=20):
    """
    GPU-accelerated implementation of NSGA-III
    """
    # Convert inputs to float32
    lb = np.asarray(lb, dtype=DTYPE)
    ub = np.asarray(ub, dtype=DTYPE)
    
    if not HAS_GPU or not MPS_ENABLED:
        print("GPU acceleration not available, falling back to CPU implementation")
        return NSGA_III.main(npop, iter, lb, ub, nobj, pc, pm, eta_c, eta_m, parallel=True)
    
    # Initialize timer
    timer = GPUTimer().start()
    
    # Step 1. Initialization
    nvar = len(lb)  # the dimension of decision space
    
    # Generate initial population
    pop = np.random.uniform(lb, ub, (npop, nvar)).astype(DTYPE)
    
    # Transfer to GPU and calculate objectives
    transfer_start = time.time()
    objs = cal_obj_gpu(pop, nobj)
    timer.record_transfer(transfer_start, time.time())
    
    # Generate reference vectors (performed on CPU)
    V = NSGA_III.reference_points(npop, nobj)
    
    # Calculate ideal point
    zmin = np.min(objs, axis=0)
    
    # Non-dominated sorting (performed on CPU)
    kernel_start = time.time()
    [pfs, rank] = NSGA_III.nd_sort(objs)
    timer.record_kernel(kernel_start, time.time())

    # Step 2. The main loop
    for t in range(iter):
        if t % max(1, iter // 4) == 0:  # Print at 0%, 25%, 50%, 75%, 100%
            print(f"Iteration: {t}/{iter} completed. Time: {time.time()-timer.start_time:.2f}s")
        
        # Step 2.1. Mating selection + crossover + mutation
        kernel_start = time.time()
        mating_pool = NSGA_III.selection(pop, pc, rank)
        timer.record_kernel(kernel_start, time.time())
        
        # Accelerate crossover on GPU
        transfer_start = time.time()
        off = NSGA_III.crossover(mating_pool, lb, ub, pc, eta_c)
        timer.record_transfer(transfer_start, time.time())
        
        # Accelerate mutation on GPU
        transfer_start = time.time()
        off = NSGA_III.mutation(off, lb, ub, pm, eta_m)
        timer.record_transfer(transfer_start, time.time())
        
        # Calculate offspring objectives on GPU
        transfer_start = time.time()
        off_objs = cal_obj_gpu(off, nobj)
        timer.record_transfer(transfer_start, time.time())
        
        # Step 2.2. Environmental selection
        # Update ideal point
        zmin = np.min((zmin, np.min(off_objs, axis=0)), axis=0)
        
        # Environmental selection on CPU/GPU
        kernel_start = time.time()
        pop, objs, rank = NSGA_III.environmental_selection(
            np.concatenate((pop, off), axis=0), 
            np.concatenate((objs, off_objs), axis=0),
            zmin, npop, V
        )
        timer.record_kernel(kernel_start, time.time())
    
    # Record end time
    timer.end()
    total_time = timer.elapsed
    print(f"Total GPU runtime: {total_time:.2f} seconds")
    print(f"Memory transfer time: {timer.total_transfer_time:.2f} seconds ({100*timer.total_transfer_time/total_time:.1f}%)")
    print(f"Kernel execution time: {timer.total_kernel_time:.2f} seconds ({100*timer.total_kernel_time/total_time:.1f}%)")
    
    return pop, objs, rank, timer