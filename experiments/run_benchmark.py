#!/usr/bin/env python
import os
import sys
import numpy as np
import time
import psutil
from multiprocessing import Pool, shared_memory
from performance_metrics import PerformanceMetrics
import NSGA_III

# Try to import GPU modules
try:
    # Import the specific function we need
    from gpu_config import get_detailed_hardware_info, HAS_GPU, MPS_ENABLED, HAS_TORCH
    import NSGA_III_GPU
    has_gpu_support = HAS_GPU and MPS_ENABLED
except ImportError:
    # Define a minimal hardware info function if import fails
    def get_detailed_hardware_info():
        return {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'has_gpu': False,
            'gpu_name': 'None'
        }
    has_gpu_support = False
    print("GPU modules not available, skipping GPU benchmarks")

# Performance metrics instance
performance_metrics = PerformanceMetrics()

def get_test_configurations():
    """Define test configurations with different sizes"""
    return [
        # {
        #     'name': 'Small',
        #     'npop': 75,
        #     'iter': 75,
        #     'nvars': 7,
        #     'nobj': 3
        # },
        # {
        #     'name': 'Medium',
        #     'npop': 125,
        #     'iter': 75,
        #     'nvars': 7,
        #     'nobj': 3
        # },
        {
            'name': 'Large',
            'npop': 250, 
            'iter': 75,
            'nvars': 7,
            'nobj': 3
        }
    ]

def run_only_benchmarks():
    """Run only the NSGA-III benchmarks without other experiments"""
    global performance_metrics
    
    # Create pool inside the function instead of at module level
    pool = None
    
    # Get detailed hardware info
    try:
        hw_info = get_detailed_hardware_info()
        
        print("\n==== Hardware Specifications ====")
        print(f"System: {hw_info.get('platform', 'Unknown')}")
        print(f"Processor: {hw_info.get('processor', 'Unknown')}")
        print(f"Physical cores: {hw_info.get('cpu_count_physical', 'Unknown')}")
        print(f"Logical cores: {hw_info.get('cpu_count_logical', 'Unknown')}")
        print(f"Total memory: {hw_info.get('memory_total_gb', 'Unknown')} GB")
        
        if 'apple_model' in hw_info:
            print(f"Apple model: {hw_info['apple_model']}")
        
        if 'gpu_cores' in hw_info:
            print(f"GPU cores: {hw_info['gpu_cores']}")
            
        print("\n==== GPU Configuration ====")
        print(f"GPU available: {hw_info.get('has_gpu', False)}")
        print(f"GPU name: {hw_info.get('gpu_name', 'Unknown')}")
        
        if hw_info.get('has_torch', False):
            print(f"PyTorch version: {hw_info.get('torch_version', 'Unknown')}")
            print(f"MPS available: {hw_info.get('torch_mps_available', False)}")
        
        # Store hardware info with performance metrics
        try:
            performance_metrics.set_hardware_info(hw_info)
        except AttributeError:
            performance_metrics.hw_info = hw_info
            print("Hardware info saved as attribute (no method available)")
    except Exception as e:
        print(f"Error collecting hardware information: {e}")
        import traceback
        traceback.print_exc()
    
    # Get the test configurations
    configs = get_test_configurations()
    
    print("\nRunning NSGA-III benchmarks with multiple sizes...")
    
    # Run each configuration
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Testing configuration: {config['name']} (npop={config['npop']}, iter={config['iter']})")
        print(f"{'='*60}")
        
        # Extract parameters
        npop = config['npop']
        iter = config['iter']
        nvars = config['nvars']
        nobj = config['nobj']
        lb = np.array([0] * nvars)
        ub = np.array([1] * nvars)
        
        try:
            # Sequential implementation
            print(f"\nTesting CPU Sequential ({config['name']})...")
            start_time = time.time()
            start_mem = psutil.Process().memory_info().rss / (1024 * 1024)
            
            NSGA_III.main(npop, iter, lb, ub, nobj=nobj, parallel=False)
            
            end_time = time.time()
            end_mem = psutil.Process().memory_info().rss / (1024 * 1024)
            seq_time = (end_time - start_time) * 1000  # Convert to ms
            
            performance_metrics.record_implementation(
                f"CPU Sequential ({config['name']})",
                time_ms=seq_time,
                memory_mb=end_mem - start_mem,
                throughput=npop * iter / (end_time - start_time)
            )
            
            print(f"Sequential implementation completed in {seq_time:.2f} ms")
            
            # Set quiet mode before running parallel to avoid repeated prints
            os.environ["NSGA_QUIET_MODE"] = "1"
            
            # Parallel implementation
            print(f"\nTesting CPU Parallel ({config['name']})...")
            start_time = time.time()
            start_mem = psutil.Process().memory_info().rss / (1024 * 1024)
            
            # Create a process pool inside the function
            if pool is None:
                pool = Pool(processes=8)  # Try different values (4, 8, 12)
            
            # Run the parallel NSGA-III
            NSGA_III.main(npop, iter, lb, ub, nobj=nobj, parallel=True)
            
            end_time = time.time()
            end_mem = psutil.Process().memory_info().rss / (1024 * 1024)
            par_time = (end_time - start_time) * 1000  # Convert to ms
            
            performance_metrics.record_implementation(
                f"CPU Parallel ({config['name']})",
                time_ms=par_time,
                memory_mb=end_mem - start_mem,
                throughput=npop * iter / (end_time - start_time)
            )
            
            print(f"Parallel implementation completed in {par_time:.2f} ms")
            
            # Reset quiet mode
            os.environ["NSGA_QUIET_MODE"] = "0"
            
            # GPU implementation if available
            if has_gpu_support:
                print(f"\nTesting GPU MPS ({config['name']})...")
                start_time = time.time()
                start_mem = psutil.Process().memory_info().rss / (1024 * 1024)
                
                # Run GPU implementation
                _, _, _, gpu_timer = NSGA_III_GPU.main(npop, iter, lb, ub, nobj=nobj)
                
                end_time = time.time()
                end_mem = psutil.Process().memory_info().rss / (1024 * 1024)
                gpu_time = (end_time - start_time) * 1000  # Convert to ms
                
                performance_metrics.record_implementation(
                    f"GPU MPS ({config['name']})",
                    time_ms=gpu_time,
                    memory_mb=end_mem - start_mem,
                    throughput=npop * iter / (end_time - start_time),
                    transfer_time=gpu_timer.total_transfer_time * 1000,
                    kernel_time=(gpu_timer.elapsed - gpu_timer.total_transfer_time) * 1000
                )
                
                print(f"GPU implementation completed in {gpu_time:.2f} ms")
                print(f"Memory transfer time: {gpu_timer.total_transfer_time * 1000:.2f} ms")
                transfer_percent = 100 * gpu_timer.total_transfer_time / gpu_timer.elapsed
                print(f"Transfer percentage: {transfer_percent:.1f}%")
                
        except Exception as e:
            print(f"Error running benchmark for {config['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Close the pool at the end
    if pool is not None:
        pool.close()
        pool.join()
    
    # Print summary table with all configurations
    try:
        print("\n==== NSGA-III Performance Comparison Across Sizes ====")
        headers = ["Implementation", "Time (ms)", "Speedup", "Memory (MB)", "Throughput"]
        print(f"{headers[0]:<25} | {headers[1]:<10} | {headers[2]:<10} | {headers[3]:<12} | {headers[4]:<15}")
        print("-" * 80)
        
        # Group by configuration
        for config in configs:
            print(f"\n--- {config['name']} Configuration (npop={config['npop']}, iter={config['iter']}) ---")
            seq_impl = f"CPU Sequential ({config['name']})"
            seq_time = performance_metrics.metrics.get(seq_impl, {}).get('time_ms', 1)
            
            for impl_type in ["CPU Sequential", "CPU Parallel", "GPU MPS"]:
                impl_name = f"{impl_type} ({config['name']})"
                if impl_name in performance_metrics.metrics:
                    data = performance_metrics.metrics[impl_name]
                    speedup = seq_time / data['time_ms'] if seq_impl in performance_metrics.metrics else 1.0
                    print(f"{impl_name:<25} | {data['time_ms']:<10.2f} | {speedup:<10.2f}x | "
                          f"{data['memory_mb']:<12.2f} | {data['throughput']:<15.2f}")
        
    except Exception as e:
        print(f"Error printing comparison table: {e}")

    # Export results at the end
    try:
        performance_metrics.export_to_csv("nsga_performance_multisizes.csv")
        print("Performance metrics saved to nsga_performance_multisizes.csv")
    except Exception as e:
        print(f"Error exporting metrics to CSV: {e}")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # Required on Windows
    run_only_benchmarks()