# Add this at the VERY TOP of your file, before any imports
import os
os.environ["NSGA_QUIET_MODE"] = "1"  # Set quiet mode BEFORE any imports
os.environ["PYTHONWARNINGS"] = "ignore"  # Suppress warnings too

#python3.13 experiment.py
import json
import pathlib
import sys, os, csv, time, io
import random
from contextlib import redirect_stdout
from math import exp
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from typing import Union
import sklearn.preprocessing as preprocessing
from sklearn.pipeline import Pipeline
import math
import multiprocessing as mp
from functools import partial
import concurrent.futures
import psutil
from performance_metrics import PerformanceMetrics
import tempfile
from multiprocessing import Lock
from pathlib import Path
import logging
import io
import NSGA_III, NSGA_III_GPU, gpu_config


# Use a simpler approach for output control
VERBOSE = False
QUIET_MODE = os.environ.get("NSGA_QUIET_MODE", "0") == "1"

# Number of CPU cores to use (leave 2 for system)
NUM_CPU_CORES = max(1, mp.cpu_count() - 2)

# Create a sentinel file to track whether config has been printed
CONFIG_PRINTED = False
CONFIG_LOCK = Lock()  # Shared memory lock for multiprocessing
CONFIG_LOCKFILE = os.path.join(tempfile.gettempdir(), "nsga_config_printed.lock")

# Add these variables near the top of the file
CONFIG_SHOWN = False
WORKER_QUIET = True
has_gpu_support = False

# Fix the incomplete try-except block
try:
    if hasattr(gpu_config, 'HAS_GPU') and gpu_config.HAS_GPU:
        has_gpu_support = True
except Exception as e:
    # Handle any exceptions when checking GPU support
    print(f"Warning: Error checking GPU support: {e}")
    has_gpu_support = False

def print_msg(message, always_print=False):
    """Print a message if verbose or always_print is True"""
    if VERBOSE or always_print and not QUIET_MODE:
        print(message)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ucb
import stats

def initialize_modules(verbose=True):
    """Initialize all modules with proper output control"""
    global has_gpu_support
    
    # Import modules
    if "NSGA_III" not in sys.modules:
        try:
            # Do a comprehensive GPU check once
            has_gpu_support = (hasattr(gpu_config, 'HAS_GPU') and gpu_config.HAS_GPU and
                              hasattr(gpu_config, 'MPS_ENABLED') and gpu_config.MPS_ENABLED)
            
            # Print GPU details if verbose
            if verbose and has_gpu_support:
                print(f"GPU detected: {gpu_config.DEVICE_NAME if hasattr(gpu_config, 'DEVICE_NAME') else 'Unknown'}")
                
        except ImportError:
            has_gpu_support = False

# Print configuration once at the start
def print_config_once():
    """Print system configuration exactly once"""
    global CONFIG_PRINTED
    
    with CONFIG_LOCK:  # Use lock to ensure atomic operation
        if not CONFIG_PRINTED:
            # Print configuration
            print(f"Using {NUM_CPU_CORES} CPU cores for parallel processing")
            print(f"GPU support: {'Available' if has_gpu_support else 'Not available'}")
            CONFIG_PRINTED = True

additional_values = {}
number = Union[float, int]  #
performance_metrics = PerformanceMetrics()

def medianSd(a: list[number]) -> tuple[number, number]:
    a = sorted(a)
    return a[int(0.5 * len(a))], (a[int(0.9 * len(a))] - a[int(0.1 * len(a))])


def preprocess_dataset(dataset):
    df = pd.DataFrame(dataset)
    df.columns = df.columns.astype(str).str.replace(' ', '_')
    # Convert all columns to numeric, coercing errors
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Fill missing values with column means, then fill remaining NaNs with zeroes
    df.fillna(df.mean(), inplace=True)
    df.fillna(0, inplace=True)  # Ensure no NaN values remain
    # Add the goal attribute to the relevant columns
    for col in df.columns:
        if 'goal' not in df[col].attrs:
            df[col].attrs['goal'] = 0  # or set to an appropriate default value

    # Convert the DataFrame back to a NumPy array
    processed_array = df.values
    return processed_array


def exploit_score(B, R):
    return B - R

def explore_score(B, R):
    return (exp(B) + exp(R)) / (1E-30 + abs(exp(B) - exp(R)))

def random_score(B, R):
    return random.random()

# Initialize an empty list instead
scoring_policies = []

# Add this function to initialize scoring policies after imports
def initialize_scoring_policies():
    """Set up scoring policies after modules are imported"""
    global scoring_policies
    
    # Only set up if not already defined
    if not scoring_policies:
        scoring_policies = [
            ('NSGA_III', NSGA_III.nsga3_score),
            ('NSGA_III_GPU', NSGA_III_GPU.nsga3_score)
        ]

def run_experiment_trial(trial_data):
    """Single trial of an experiment for parallel execution"""
    d, how, max_label, rnd = trial_data
    try:
        tmp = d.shuffle().activeLearning(score=how, max_label=max_label)
        return rnd(d.d2h(tmp[0]))
    except Exception as e:
        print_msg(f"Error in trial: {e}", always_print=True)
        return None

# Run experiment on a dataset and collect results into SOME instances
def run_experiment(data_path):
    print_msg(f"Processing file: {data_path}", always_print=True)
    results = []
    somes = []
    data_dict = {}  # Dictionary to store the data for each dataset
    
    # Load the data
    try:
        d = ucb.DATA().adds(ucb.csv(data_path))
    except Exception as e:
        print_msg(f"Error loading data from {data_path}: {e}", always_print=True)
        return []

    # Ensure that the goal attribute is set for the relevant columns
    for col in d.cols.y:
        if not hasattr(col, 'goal'):
            col.goal = 0  # or set to an appropriate default value

    # Filter out columns that do not have the norm method
    d.cols.y = [col for col in d.cols.y if hasattr(col, 'norm')]

    # Baseline "asIs" distances to heaven (chebyshev distances)
    b4 = [d.d2h(row) for row in d.rows]

    assert len(b4) == len(d.rows), "Baseline d2h should match the number of rows."
    somes.append(stats.SOME(b4, f"asIs,{len(d.rows)}"))
    asIs, div = medianSd(b4)

    # Additional values to be added to the report
    global additional_values
    additional_values = {
        "datapath": data_path,
        "asIs": round(asIs, 3),
        "div": round(div, 3),
        "rows": len(d.rows),
        "xcols": len(d.cols.x),
        "ycols": len(d.cols.y)
    }

    # Define scoring policies
    rows_num = len(d.rows)
    rnd = lambda z: z
    global scoring_policies
    global timing_results
    timing_results = {}
    repeats = 20

    max_label_values = []
    
    # Helper function to ensure even values
    def make_even(n):
        return n + (n % 2)  # Add 1 if odd
    
    # First budget: 1% of rows, at least 4, max 20, always even
    first_budget = make_even(min(20, max(4, int(rows_num * 0.01))))
    max_label_values.append(first_budget)
    
    # Second budget: first budget + 2% of rows (even), max 40
    second_increment = make_even(int(rows_num * 0.02))
    second_budget = make_even(min(40, first_budget + second_increment))
    max_label_values.append(second_budget)
    
    # Third budget: second budget + 2% of rows (even), max 60
    third_increment = make_even(int(rows_num * 0.02))
    third_budget = make_even(min(60, second_budget + third_increment))
    max_label_values.append(third_budget)
    
    print_msg(f"Using label budgets: {max_label_values}", always_print=True)

    # Process each max_label setting ONCE
    for max_label in max_label_values:
        # Run each scoring policy
        for what, how in scoring_policies:
            print_msg(f"Running active learning with policy: {what}", always_print=True)

            start = time.time()  # Start timing
            result = []
            
            # Run trials serially instead of with multiprocessing
            for _ in range(repeats):
                try:
                    tmp = d.shuffle().activeLearning(score=how, max_label=max_label)
                    result.append(d.d2h(tmp[0]))  # Apply the identity function directly
                except Exception as e:
                    print_msg(f"Error in trial: {e}", always_print=True)
            
            smart_time = (time.time() - start) / repeats  # Average time
            pre = f"{what}"
            tag = f"{pre}, {max_label}"
            print_msg(f"{tag}: {smart_time:.2f} secs", always_print=True)
            somes.append(stats.SOME(result, tag))
            timing_results[tag] = smart_time
            
    return somes

def process_dataset(dataset_info):
    """Process a single dataset (for parallel execution)"""
    if 'file' in dataset_info:
        data_path = dataset_info['file']
        try:
            somes = run_experiment(data_path)
            
            # Get the filename from the path
            file_name = data_path.split(os.path.sep)[-1].split('.')[0] + '.csv'
            output_path = os.path.join('results', dataset_info.get('folder', 'unknown'), file_name)
            
            # Save results to CSV
            save_results_to_csv(somes, output_path, additional_values)
            return True
        except Exception as e:
            print_msg(f"Error processing {data_path}: {e}", always_print=True)
            return False
    else:
        print_msg(f"Skipping dataset due to missing 'file' key: {dataset_info}", always_print=True)
        return False

def save_results_to_csv(somes, output_file, additional_values):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=':')

        # Write additional values
        for key, value in additional_values.items():
            writer.writerow([key, value])
            print_msg(f"{key:10}: {value}", always_print=True)

        # Write each timing result in a new row
        for key, value in timing_results.items():
            writer.writerow([key, round(value, 2)])

        # Capture the output of stats.report
        report_output = io.StringIO()
        with redirect_stdout(report_output):
            stats.report(somes, 0.01)
        report_output.seek(0)

        # Write the captured report to the CSV file
        print_msg("Stats_report:", always_print=True)
        for line in report_output:
            writer.writerow([line.strip()])

        # Also print the captured report to the console
        report_output.seek(0)
        print_msg(report_output.read(), always_print=True)
        print_msg("_" * 100, always_print=True)

# Load CSV containing file paths
def load_csv(file_path):
    datasets = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'file' in row:
                datasets.append(row)
            else:
                print_msg(f"Skipping dataset due to missing 'file' key: {row}", always_print=True)
    return datasets

def logging_initializer():
    """Initialize logging settings for worker processes"""
    setattr(logging.getLogger(), 'disabled', not VERBOSE)

def process_datasets(datasets, folder_name):
    # Add folder name to dataset info
    for dataset in datasets:
        dataset['folder'] = folder_name
    
    # Process datasets in parallel
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(5, NUM_CPU_CORES),
        initializer=worker_initializer  # Much simpler initializer
    ) as executor:
        results = list(executor.map(process_dataset, datasets))
    
    success_count = sum(1 for result in results if result)
    print(f"Successfully processed {success_count} out of {len(datasets)} datasets in {folder_name}")

def run_nsga_benchmark(collect_metrics=True):
    """Run NSGA-III benchmarks for different implementations"""
    global performance_metrics, has_gpu_support
    
    # Use the global has_gpu_support directly
    if has_gpu_support:
        print_msg("Using GPU-accelerated implementation", always_print=True)
    else:
        print_msg("Using CPU-only implementation", always_print=True)
    
    # Problem parameters (consistent for fair comparison)
    npop = 100
    iter = 75
    lb = np.array([0] * 7) 
    ub = np.array([1] * 7)
    nobj = 3
    
    print_msg("Running NSGA-III benchmarks...", always_print=True)
    
    try:
        # CPU Parallel implementation
        print_msg("Testing CPU Parallel implementation...", always_print=True)
        start_time = time.time()
        start_mem = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Update this line to handle potential 4th return value (timer)
        results_cpu = NSGA_III.main(npop, iter, lb, ub, nobj=nobj, parallel=True)
        if len(results_cpu) == 3:
            pop_cpu, objs_cpu, rank_cpu = results_cpu
            cpu_timer = None
        else:
            pop_cpu, objs_cpu, rank_cpu, cpu_timer = results_cpu
        
        end_time = time.time()
        end_mem = psutil.Process().memory_info().rss / (1024 * 1024)
        cpu_time = end_time - start_time
        
        performance_metrics.record_implementation(
            "CPU Parallel",
            time_ms=cpu_time * 1000,
            memory_mb=end_mem - start_mem,
            throughput=npop * iter / cpu_time
        )
        
        # GPU implementation (if available)
        if has_gpu_support:
            print_msg("Testing GPU implementation...", always_print=True)
            start_time = time.time()
            start_mem = psutil.Process().memory_info().rss / (1024 * 1024)
            
            pop_gpu, objs_gpu, rank_gpu, gpu_timer = NSGA_III_GPU.main(npop, iter, lb, ub, nobj=nobj)
            
            end_time = time.time()
            end_mem = psutil.Process().memory_info().rss / (1024 * 1024)
            gpu_time = end_time - start_time
            
            performance_metrics.record_implementation(
                "GPU MPS",
                time_ms=gpu_time * 1000,
                memory_mb=end_mem - start_mem,
                throughput=npop * iter / gpu_time,
                transfer_time=gpu_timer.total_transfer_time * 1000 if hasattr(gpu_timer, 'total_transfer_time') else 0
            )
            
            # Calculate speedup
            speedup = cpu_time / gpu_time
            print_msg(f"GPU Speedup: {speedup:.2f}x", always_print=True)
            
            # Compare solution quality (Pareto front size)
            pf_cpu = objs_cpu[rank_cpu == 0]
            pf_gpu = objs_gpu[rank_gpu == 0]
            print_msg(f"CPU Pareto front size: {len(pf_cpu)}", always_print=True)
            print_msg(f"GPU Pareto front size: {len(pf_gpu)}", always_print=True)
        
        # Export metrics
        if collect_metrics:
            performance_metrics.export_to_csv("nsga_performance.csv")
            performance_metrics.print_comparison()
            print_msg("Performance metrics saved to nsga_performance.csv", always_print=True)
            
    except Exception as e:
        print_msg(f"Error during benchmarking: {e}", always_print=True)
        import traceback
        traceback.print_exc()
        # Still try to export whatever metrics we collected
        if collect_metrics and performance_metrics.metrics:
            performance_metrics.export_to_csv("nsga_partial_results.csv")
            print_msg("Partial metrics saved to nsga_partial_results.csv", always_print=True)

def compare_nsga_implementations(datasets=None):
    """Compare NSGA-III CPU vs GPU implementations on your existing datasets"""
    import glob
    import os
    
    print_msg("Comparing NSGA-III implementations...", always_print=True)
    
    # Define GPU support variable
    has_gpu_support = False
    try:
        # Check if HAS_GPU and MPS_ENABLED are available
        if 'gpu_config' in sys.modules:
            has_gpu_support = (hasattr(gpu_config, 'HAS_GPU') and gpu_config.HAS_GPU and 
                             hasattr(gpu_config, 'MPS_ENABLED') and gpu_config.MPS_ENABLED)
        elif 'HAS_GPU' in globals() and 'MPS_ENABLED' in globals():
            has_gpu_support = HAS_GPU and MPS_ENABLED
    except Exception as e:
        print_msg(f"Error checking GPU support: {e}", always_print=True)
    
    print_msg(f"GPU Support: {'Available' if has_gpu_support else 'Not Available'}", always_print=True)
    
    # If no datasets provided, find some automatically
    if datasets is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        if not os.path.exists(data_dir):
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        
        print_msg(f"Looking for datasets in {data_dir}", always_print=True)
        
        # Find CSV files to use for comparison
        csv_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        
        if not csv_files:
            print_msg("No datasets found. Please provide datasets manually.", always_print=True)
            return
            
        # Use the first 3 CSV files found
        datasets = [{'file': f} for f in csv_files[:3]]
        print_msg(f"Using {len(datasets)} datasets for comparison", always_print=True)
    
    results = {
        'CPU': {},
        'GPU': {}
    }
    
    # Run both implementations on each dataset
    for dataset in datasets:
        data_path = dataset['file']
        print_msg(f"\nComparing NSGA-III implementations on {os.path.basename(data_path)}", always_print=True)
        
        # Load data
        try:
            d = ucb.DATA().adds(ucb.csv(data_path))
        except Exception as e:
            print_msg(f"Error loading data: {e}", always_print=True)
            continue
            
        # Settings for comparison
        max_label = 10  # Use consistent value for comparison
        repeats = 3     # Number of trials for each implementation
        
        # Compare CPU and GPU implementations
        for impl_name, impl_type in [('NSGA-III CPU', 'CPU'), ('NSGA-III GPU', 'GPU')]:
            # Skip GPU if not available
            if impl_type == 'GPU' and (not has_gpu_support):
                print_msg("GPU implementation not available, skipping", always_print=True)
                continue
                
            # Choose appropriate scoring function
            if impl_type == 'CPU':
                scoring_fn = NSGA_III.nsga3_score
            else:
                scoring_fn = NSGA_III_GPU.nsga3_score
                
            print_msg(f"Testing {impl_name}...", always_print=True)
            
            # Measure memory before
            start_mem = psutil.Process().memory_info().rss / (1024 * 1024)
            
            # Run multiple trials and time them
            start_time = time.time()
            trial_results = []
            
            for _ in range(repeats):
                try:
                    # Use your existing activeLearning method
                    tmp = d.shuffle().activeLearning(score=scoring_fn, max_label=max_label)
                    result = tmp[0] if isinstance(tmp, tuple) and len(tmp) > 0 else 0
                    trial_results.append(result)
                except Exception as e:
                    print_msg(f"Error in trial: {e}", always_print=True)
            
            # Calculate metrics
            elapsed_time = time.time() - start_time
            avg_time = elapsed_time / max(1, len(trial_results))
            end_mem = psutil.Process().memory_info().rss / (1024 * 1024)
            mem_used = end_mem - start_mem
            
            # Store results
            results[impl_type][data_path] = {
                'time': avg_time,
                'memory': mem_used,
                'results': trial_results,
                'avg_result': sum(trial_results) / max(1, len(trial_results)) if trial_results else 0
            }
            
            print_msg(f"  Average time: {avg_time:.4f} seconds", always_print=True)
            print_msg(f"  Memory used: {mem_used:.2f} MB", always_print=True)
            print_msg(f"  Average score: {results[impl_type][data_path]['avg_result']:.4f}", always_print=True)
    
    # Calculate and display speedups
    print_msg("\n===== NSGA-III Performance Comparison =====", always_print=True)
    for dataset in datasets:
        data_path = dataset['file']
        data_name = os.path.basename(data_path)
        
        if data_path in results['CPU'] and data_path in results['GPU']:
            cpu_time = results['CPU'][data_path]['time']
            gpu_time = results['GPU'][data_path]['time']
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            
            cpu_mem = results['CPU'][data_path]['memory']
            gpu_mem = results['GPU'][data_path]['memory']
            mem_ratio = cpu_mem / gpu_mem if gpu_mem > 0 else 0
            
            cpu_score = results['CPU'][data_path]['avg_result']
            gpu_score = results['GPU'][data_path]['avg_result']
            score_diff = ((gpu_score - cpu_score) / cpu_score) * 100 if cpu_score > 0 else 0
            
            print_msg(f"\nDataset: {data_name}", always_print=True)
            print_msg(f"  Time    - CPU: {cpu_time:.4f}s, GPU: {gpu_time:.4f}s, Speedup: {speedup:.2f}x", always_print=True)
            print_msg(f"  Memory  - CPU: {cpu_mem:.2f}MB, GPU: {gpu_mem:.2f}MB, Ratio: {mem_ratio:.2f}x", always_print=True)
            print_msg(f"  Quality - CPU: {cpu_score:.4f}, GPU: {gpu_score:.4f}, Diff: {score_diff:.2f}%", always_print=True)
    
    # Add summary statistics at the end
    if len(results['CPU']) > 0 and len(results['GPU']) > 0:
        cpu_times = [results['CPU'][path]['time'] for path in results['CPU']]
        gpu_times = [results['GPU'][path]['time'] for path in results['GPU']]
        
        avg_cpu_time = sum(cpu_times) / len(cpu_times) if cpu_times else 0
        avg_gpu_time = sum(gpu_times) / len(gpu_times) if gpu_times else 0
        avg_speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else 0
        
        print_msg("\n===== Overall Performance Summary =====", always_print=True)
        print_msg(f"Average CPU time: {avg_cpu_time:.4f} seconds", always_print=True)
        print_msg(f"Average GPU time: {avg_gpu_time:.4f} seconds", always_print=True)
        print_msg(f"Average speedup: {avg_speedup:.2f}x", always_print=True)
    
    # Save results to CSV
    import csv
    with open('nsga_comparison_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'CPU Time (s)', 'GPU Time (s)', 'Speedup', 
                        'CPU Memory (MB)', 'GPU Memory (MB)', 'Memory Ratio',
                        'CPU Score', 'GPU Score', 'Score Difference (%)'])
        
        for dataset in datasets:
            data_path = dataset['file']
            data_name = os.path.basename(data_path)
            
            if data_path in results['CPU'] and data_path in results['GPU']:
                cpu_time = results['CPU'][data_path]['time']
                gpu_time = results['GPU'][data_path]['time']
                speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                
                cpu_mem = results['CPU'][data_path]['memory']
                gpu_mem = results['GPU'][data_path]['memory']
                mem_ratio = cpu_mem / gpu_mem if gpu_mem > 0 else 0
                
                cpu_score = results['CPU'][data_path]['avg_result']
                gpu_score = results['GPU'][data_path]['avg_result']
                score_diff = ((gpu_score - cpu_score) / cpu_score) * 100 if cpu_score > 0 else 0
                
                writer.writerow([data_name, cpu_time, gpu_time, speedup,
                                cpu_mem, gpu_mem, mem_ratio,
                                cpu_score, gpu_score, score_diff])
    
    print_msg("\nDetailed results saved to nsga_comparison_results.csv", always_print=True)
    return results

# Define a more robust initializer for worker processes
def worker_initializer():
    """Completely suppress output in worker processes"""
    global QUIET_MODE, WORKER_QUIET
    
    # Force quiet mode in all worker processes
    QUIET_MODE = True
    WORKER_QUIET = True
    
    # Signal to other modules that we're in a worker process
    os.environ["NSGA_QUIET_MODE"] = "1"
    
    # Disable logging
    logging.getLogger().setLevel(logging.CRITICAL)
    
    # Completely redirect stdout/stderr to nowhere
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

def main():
    # Remove any stale lock file
    try:
        if os.path.exists(CONFIG_LOCKFILE):
            os.remove(CONFIG_LOCKFILE)
    except:
        pass
    
    # Initialize modules FIRST (this will set has_gpu_support correctly)
    initialize_modules()
    
    # Then print config once
    print_config_once()
    
    # Define the directory containing the datasets CSV files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check for data directory in multiple possible locations
    possible_data_paths = [
        os.path.join(script_dir, 'data'),
        script_dir,
        os.path.join(os.path.dirname(script_dir), 'data'),
        os.path.abspath(os.path.join(script_dir, '..', 'data'))
    ]
    
    # Find the first existing data directory
    data_dir = None
    for path in possible_data_paths:
        if os.path.exists(path):
            print_msg(f"Found data directory at: {path}", always_print=True)
            data_dir = path
            break
    
    if not data_dir:
        print_msg("Could not find data directory. Creating one in the current script directory.", always_print=True)
        data_dir = os.path.join(script_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
    print_msg(f"Using data directory: {data_dir}", always_print=True)
    
    # Ensure results directory exists
    os.makedirs(os.path.join(script_dir, 'results'), exist_ok=True)
    
    # Load datasets from CSV files
    datasets_files = [
        ('0-10_xcols.csv', 'ds_10'),
        ('11-18_xcols.csv', 'ds_11_18'),
        ('19-29_xcols.csv', 'ds_19_29'),
        ('30-54_xcols.csv', 'ds_30_54'),
        ('55-1000_xcols.csv', 'ds_55_1000')
    ]
    
    # Process each dataset file sequentially, but datasets within files in parallel
    for file, folder in datasets_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            print_msg(f"Processing datasets from {file}...", always_print=True)
            datasets = load_csv(file_path)
            if datasets:
                process_datasets(datasets, folder)
        else:
            print_msg(f"Dataset file not found: {file_path}", always_print=True)
            # Check if file exists without directory prefix (for relative path issues)
            if os.path.exists(file):
                print_msg(f"  However, file exists at relative path: {file}", always_print=True)
                datasets = load_csv(file)
                if datasets:
                    process_datasets(datasets, folder)
                    
    print_msg("\nRunning NSGA-III performance benchmarks...", always_print=True)
    run_nsga_benchmark()
    
    print_msg("\nComparing NSGA-III implementations...", always_print=True)
    compare_nsga_implementations()
    
    print_msg("\nRunning NSGA-III CPU vs GPU comparison...", always_print=True)
    
    # # Select a subset of your datasets for comparison (one from each category)
    # comparison_datasets = []
    # for file, folder in datasets_files:
    #     file_path = os.path.join(data_dir, file)
    #     if os.path.exists(file_path):
    #         datasets = load_csv(file_path)
    #         if datasets:
    #             comparison_datasets.append(datasets)  # Add first dataset from each category
    
    # # To use all selected datasets (one from each category):
    # compare_nsga_implementations(comparison_datasets)

    # OR to include ALL datasets from all categories:
    comparison_datasets = []
    for file, folder in datasets_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            datasets = load_csv(file_path)
            if datasets:
                comparison_datasets.extend(datasets)  # Add ALL datasets

    # Run comparison
    compare_nsga_implementations(comparison_datasets)  # Limit to 3 datasets for quicker testing
    
    print_msg("All experiments completed!", always_print=True)

# Run the main function
if __name__ == "__main__":
    # Initialize multiprocessing with spawn method for better compatibility
    mp.set_start_method('spawn', force=True)
    main()