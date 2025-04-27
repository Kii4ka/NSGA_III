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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ucb
import stats
import NSGA_III

# Import Metal GPU acceleration config
try:
    from gpu_config import *
except ImportError:
    print("GPU config not found, running on CPU only")

# Number of CPU cores to use (leave 2 for system)
NUM_CPU_CORES = max(1, mp.cpu_count() - 2)
print(f"Using {NUM_CPU_CORES} CPU cores for parallel processing")

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

scoring_policies = [
    # ('exploit', exploit_score),
    # ('explore', explore_score),
    # ('Random', random_score),
    # ('UCB_GPM', 'UCB_GPM'), 
    ('NSGA_III', NSGA_III.nsga3_score),
]

def run_experiment_trial(trial_data):
    """Single trial of an experiment for parallel execution"""
    d, how, max_label, rnd = trial_data
    try:
        tmp = d.shuffle().activeLearning(score=how, max_label=max_label)
        return rnd(d.d2h(tmp[0]))
    except Exception as e:
        print(f"Error in trial: {e}")
        return None

# Run experiment on a dataset and collect results into SOME instances
def run_experiment(data_path):
    print(f"Processing file: {data_path}")
    results = []
    somes = []
    data_dict = {}  # Dictionary to store the data for each dataset
    
    # Load the data
    try:
        d = ucb.DATA().adds(ucb.csv(data_path))
    except Exception as e:
        print(f"Error loading data from {data_path}: {e}")
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
    
    print(f"Using label budgets: {max_label_values}")

    # Process each max_label setting ONCE
    for max_label in max_label_values:
        # Run each scoring policy
        for what, how in scoring_policies:
            print(f"Running active learning with policy: {what}")

            start = time.time()  # Start timing
            result = []
            
            # Run trials serially instead of with multiprocessing
            for _ in range(repeats):
                try:
                    tmp = d.shuffle().activeLearning(score=how, max_label=max_label)
                    result.append(d.d2h(tmp[0]))  # Apply the identity function directly
                except Exception as e:
                    print(f"Error in trial: {e}")
            
            smart_time = (time.time() - start) / repeats  # Average time
            pre = f"{what}"
            tag = f"{pre}, {max_label}"
            print(tag, f": {smart_time:.2f} secs")
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
            print(f"Error processing {data_path}: {e}")
            return False
    else:
        print(f"Skipping dataset due to missing 'file' key: {dataset_info}")
        return False

def save_results_to_csv(somes, output_file, additional_values):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=':')

        # Write additional values
        for key, value in additional_values.items():
            writer.writerow([key, value])
            print(f"{key:10}: {value}")

        # Write each timing result in a new row
        for key, value in timing_results.items():
            writer.writerow([key, round(value, 2)])

        # Capture the output of stats.report
        report_output = io.StringIO()
        with redirect_stdout(report_output):
            stats.report(somes, 0.01)
        report_output.seek(0)

        # Write the captured report to the CSV file
        print("Stats_report:")
        for line in report_output:
            writer.writerow([line.strip()])

        # Also print the captured report to the console
        report_output.seek(0)
        print(report_output.read())
        print("_" * 100)

# Load CSV containing file paths
def load_csv(file_path):
    datasets = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'file' in row:
                datasets.append(row)
            else:
                print(f"Skipping dataset due to missing 'file' key: {row}")
    return datasets

def process_datasets(datasets, folder_name):
    # Add folder name to dataset info
    for dataset in datasets:
        dataset['folder'] = folder_name
    
    # Process datasets in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(5, NUM_CPU_CORES)) as executor:
        results = list(executor.map(process_dataset, datasets))
    
    success_count = sum(1 for result in results if result)
    print(f"Successfully processed {success_count} out of {len(datasets)} datasets in {folder_name}")

def run_nsga_benchmark(collect_metrics=True):
    """Run NSGA-III benchmarks for different implementations"""
    global performance_metrics
    
    # Problem parameters (reduce size for testing)
    npop = 50  # Reduced from 100 for faster testing
    iter = 50  # Reduced from 100 for faster testing
    lb = np.array([0] * 7) 
    ub = np.array([1] * 7)
    nobj = 3
    
    print("Running NSGA-III benchmarks...")
    
    try:
        # CPU Parallel - run with explicit timeout
        print("Testing CPU Parallel implementation...")
        start_time = time.time()
        start_mem = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Run parallel implementation with timeout protection
        NSGA_III.main(npop, iter, lb, ub, nobj=nobj, parallel=True)
        
        end_time = time.time()
        end_mem = psutil.Process().memory_info().rss / (1024 * 1024)
        
        performance_metrics.record_implementation(
            "CPU Parallel",
            time_ms=(end_time - start_time) * 1000,
            memory_mb=end_mem - start_mem,
            throughput=npop * iter / (end_time - start_time)
        )
        
        # Export metrics
        if collect_metrics:
            performance_metrics.export_to_csv("nsga_performance.csv")
            performance_metrics.print_comparison()
            print("Performance metrics saved to nsga_performance.csv")
            
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        # Still try to export whatever metrics we collected
        if collect_metrics and performance_metrics.metrics:
            performance_metrics.export_to_csv("nsga_partial_results.csv")
            print("Partial metrics saved to nsga_partial_results.csv")

def main():
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
            print(f"Found data directory at: {path}")
            data_dir = path
            break
    
    if not data_dir:
        print("Could not find data directory. Creating one in the current script directory.")
        data_dir = os.path.join(script_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
    print(f"Using data directory: {data_dir}")
    
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
            print(f"Processing datasets from {file}...")
            datasets = load_csv(file_path)
            if datasets:
                process_datasets(datasets, folder)
        else:
            print(f"Dataset file not found: {file_path}")
            # Check if file exists without directory prefix (for relative path issues)
            if os.path.exists(file):
                print(f"  However, file exists at relative path: {file}")
                datasets = load_csv(file)
                if datasets:
                    process_datasets(datasets, folder)
                    
    print("\nRunning NSGA-III performance benchmarks...")
    run_nsga_benchmark()
    
    print("All experiments completed!")

# Run the main function
if __name__ == "__main__":
    # Initialize multiprocessing with spawn method for better compatibility
    mp.set_start_method('spawn', force=True)
    main()