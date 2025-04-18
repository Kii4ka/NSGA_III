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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ucb
import stats
import NSGA_III

additional_values = {}
number = Union[float, int]  #


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


scoring_policies = [
    ('exploit', lambda B, R: B - R),
    ('explore', lambda B, R: (exp(B) + exp(R)) / (1E-30 + abs(exp(B) - exp(R)))),
    ('Random', lambda B, R: random.random()),
    ('UCB_GPM', 'UCB_GPM'), 
    ('NSGA_III', NSGA_III.nsga3_score),

]


# Run experiment on a dataset and collect results into SOME instances
def run_experiment(data_path):
    print(f"Processing file: {data_path}")
    results = []
    somes = []
    data_dict = {}  # Dictionary to store the data for each dataset
    # Load the data

    d = ucb.DATA().adds(ucb.csv(data_path))

    # Ensure that the goal attribute is set for the relevant columns
    for col in d.cols.y:
        if not hasattr(col, 'goal'):
            col.goal = 0  # or set to an appropriate default value

    # Filter out columns that do not have the norm method
    d.cols.y = [col for col in d.cols.y if hasattr(col, 'norm')]

    # Baseline "asIs" distances to heaven (chebyshev distances)
    b4 = [d.d2h(row) for row in d.rows]

    assert len(b4) == len( d.rows), "Baseline d2h should match the number of rows."
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
        # for what, how in [scoring_policies[i] for i in [0, 1, 2, 4, 5, 6]]:

            print(f"Running active learning with policy: {what}")

            start = time.time()  # Start timing for smart guesses
            result = []
            for _ in range(repeats):
                tmp = d.shuffle().activeLearning(score=how, max_label=max_label)
                result.append(rnd(d.d2h(tmp[0])))
            smart_time = (time.time() - start) / repeats  # End timing for smart guesses
            pre = f"{what}"
            tag = f"{pre}, {max_label}"
            print(tag, f": {smart_time:.2f} secs")
            somes.append(stats.SOME(result, tag))
            timing_results[tag] = smart_time
            
    #     # Active Learning with ALBD
    #     start = time.time()
    #     result_AL = []

    #     for repeat in range(repeats):
    #         result_labled_set = d.shuffle().activeLearningALBD(
    #             [scoring_policies[i] for i in range(len(scoring_policies)) if i not in [3, 7, 9]], 
    #             max_label=max_label
    #         )
    #         # Calculate distance for the labeled set
    #         d2h_combined = rnd(d.d2h(result_labled_set[0]))
    #         result_AL.append(d2h_combined)

    #     # Calculate time for this active learning round
    #     smart_time = (time.time() - start) / repeats
    #     pre = f"ALBD"
    #     tag = f"{pre}, {max_label}"
    #     print(tag, f": {smart_time:.2f} secs")
    #     somes.append(stats.SOME(result_AL, tag))
    #     timing_results[tag] = smart_time

    # # Add data to the dictionary (for in-memory use only)
    # set_name = os.path.basename(data_path).split('.')[0]
    # data_dict[set_name] = {
    #     "timing_results": timing_results
    # }
    
    return somes

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
        print(
            "_______________________________________________________________________________________________________________________________")


# Load CSV containing file paths
def load_csv(file_path):
    datasets = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            datasets.append(row)
    return datasets


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
    for dataset in datasets:
        if 'file' in dataset:
            data_path = dataset['file']
            somes = run_experiment(data_path)

            file_name = data_path.split(os.path.sep)[-1].split('.')[0] + '.csv'
            save_results_to_csv(somes,
                                os.path.join('results', folder_name, file_name),
                                additional_values)
        else:
            print(f"Skipping dataset due to missing 'file' key: {dataset}")


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
    
    # Load datasets from CSV files
    datasets_files = [
        ('0-10_xcols.csv', 'ds_10'),
        ('11-18_xcols.csv', 'ds_11_18'),
        ('19-29_xcols.csv', 'ds_19_29'),
        ('30-54_xcols.csv', 'ds_30_54'),
        ('55-1000_xcols.csv', 'ds_55_1000')
    ]
    
    # List files in the data directory to help with debugging
    print("Files in data directory:")
    try:
        for filename in os.listdir(data_dir):
            print(f"  - {filename}")
    except Exception as e:
        print(f"Error listing directory: {e}")

    for file, folder in datasets_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
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


# Run the main function
if __name__ == "__main__":
    main()