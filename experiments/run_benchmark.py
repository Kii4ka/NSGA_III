#!/usr/bin/env python
import os
import sys
import numpy as np
import time
import psutil
from performance_metrics import PerformanceMetrics
import NSGA_III

# Try to import GPU modules
try:
    from gpu_config import HAS_GPU, MPS_ENABLED, print_gpu_info
    import NSGA_III_GPU
    has_gpu_support = HAS_GPU and MPS_ENABLED
except ImportError:
    has_gpu_support = False
    print("GPU modules not available, skipping GPU benchmarks")

# Performance metrics instance
performance_metrics = PerformanceMetrics()

def run_only_benchmarks():
    """Run only the NSGA-III benchmarks without other experiments"""
    global performance_metrics
    
    # Problem parameters (reduced for quicker testing)
    npop = 50
    iter = 50
    lb = np.array([0] * 7) 
    ub = np.array([1] * 7)
    nobj = 3
    
    print("Running NSGA-III standalone benchmarks...")
    
    try:
        # Sequential first
        print("Testing CPU Sequential implementation...")
        start_time = time.time()
        start_mem = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Run sequential implementation
        NSGA_III.main(npop, iter, lb, ub, nobj=nobj, parallel=False)
        
        end_time = time.time()
        end_mem = psutil.Process().memory_info().rss / (1024 * 1024)
        seq_time = (end_time - start_time) * 1000  # Convert to ms
        
        performance_metrics.record_implementation(
            "CPU Sequential",
            time_ms=seq_time,
            memory_mb=end_mem - start_mem,
            throughput=npop * iter / (end_time - start_time)
        )
        
        print(f"Sequential implementation completed in {seq_time:.2f} ms")
        
        # Then parallel
        print("Testing CPU Parallel implementation...")
        start_time = time.time()
        start_mem = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Run parallel implementation
        NSGA_III.main(npop, iter, lb, ub, nobj=nobj, parallel=True)
        
        end_time = time.time()
        end_mem = psutil.Process().memory_info().rss / (1024 * 1024)
        par_time = (end_time - start_time) * 1000  # Convert to ms
        
        performance_metrics.record_implementation(
            "CPU Parallel",
            time_ms=par_time,
            memory_mb=end_mem - start_mem,
            throughput=npop * iter / (end_time - start_time)
        )
        
        print(f"Parallel implementation completed in {par_time:.2f} ms")
        
        # Run GPU implementation if available
        if has_gpu_support:
            print("Testing GPU implementation...")
            start_time = time.time()
            start_mem = psutil.Process().memory_info().rss / (1024 * 1024)
            
            # Run GPU implementation
            _, _, _, gpu_timer = NSGA_III_GPU.main(npop, iter, lb, ub, nobj=nobj)
            
            end_time = time.time()
            end_mem = psutil.Process().memory_info().rss / (1024 * 1024)
            gpu_time = (end_time - start_time) * 1000  # Convert to ms
            
            performance_metrics.record_implementation(
                "GPU MPS",
                time_ms=gpu_time,
                memory_mb=end_mem - start_mem,
                throughput=npop * iter / (end_time - start_time),
                transfer_time=gpu_timer.total_transfer_time * 1000,  # Convert to ms
                kernel_time=(gpu_timer.elapsed - gpu_timer.total_transfer_time) * 1000  # Convert to ms
            )
            
            print(f"GPU implementation completed in {gpu_time:.2f} ms")
            print(f"Memory transfer time: {gpu_timer.total_transfer_time * 1000:.2f} ms")
        
        # Only export metrics once at the very end
        print("Exporting final performance metrics...")
        try:
            performance_metrics.export_to_csv("nsga_performance.csv")
            print("Performance metrics saved to nsga_performance.csv")
        except Exception as e:
            print(f"Error exporting metrics to CSV: {e}")

        try:
            performance_metrics.print_comparison()
        except Exception as e:
            # Fall back to basic print if method is missing
            print("\n==== NSGA-III Performance Comparison ====")
            for impl_name, data in performance_metrics.metrics.items():
                print(f"{impl_name}: {data['time_ms']:.2f} ms, Speedup: {data.get('speedup', 1.0):.2f}x")
        
    except Exception as e:
        print(f"Error during benchmarking: {e}")

if __name__ == "__main__":
    run_only_benchmarks()