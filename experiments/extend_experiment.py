#  #python3.13 -B extend_experiment.py ../data/optimize/[comp]*/*.csv
import sys
import os
import csv as py_csv
from typing import Union  # Use typing.Union for older Python versions

# Add the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import with try/except to handle different Python versions
try:
    from ezr import the, DATA, csv as ezr_csv, dot
except TypeError:
    # If the import fails due to type annotation issues
    print("Modifying import approach for older Python version...")
    # You may need to modify ezr.py or use a compatible version
    sys.exit("Please use Python 3.10+ or update the type hints in ezr.py to use typing.Union")

def myfun(train):
    try:
        d = DATA().adds(ezr_csv(train))
        x = len(d.cols.x)
        return [x, len(d.cols.y), len(d.rows), train]
    except Exception as e:
        print(f"Error processing {train}: {e}")
        return [0, 0, 0, train]  # Return default values on error

# Files to write output based on xcols condition
file_ranges = {
    '0-10_xcols.csv': (0, 10),
    '11-18_xcols.csv': (11, 18),
    '19-29_xcols.csv': (19, 29),
    '30-54_xcols.csv': (30, 54),
    '55-1000_xcols.csv': (55, 1000)
}

# Create output directory structure if it doesn't exist
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(output_dir, exist_ok=True)

# Open all files and write headers
writers = {}
for file_name, _ in file_ranges.items():
    file_path = os.path.join(output_dir, file_name)
    file = open(file_path, 'w', newline='')
    writer = py_csv.writer(file)
    writer.writerow(["xcols", "ycols", "rows", "file"])
    writers[file_name] = (file, writer)

# Process each CSV file in the data folder
script_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.abspath(os.path.join(script_dir, "..", "data"))

print(f"Looking for data in: {data_folder}")

if not os.path.exists(data_folder):
    print(f"Data folder not found at {data_folder}")
    # Try other common locations
    alt_data_folder = os.path.join(script_dir, "data")
    if os.path.exists(alt_data_folder):
        data_folder = alt_data_folder
        print(f"Using alternative data folder: {data_folder}")
    else:
        print("No data folder found. Please verify the data location.")

for root, dirs, files in os.walk(data_folder):
    for file in files:
        print("Processing: ", file, " in ", root)
        if file.startswith("."):
            continue
        if file.endswith(".md") or file.endswith(".ipynb"):
            continue
        if file.endswith(".csv"):
            file_path = os.path.join(root, file)
            try:
                data_info = myfun(file_path)
                xcols = data_info[0]
                for file_name, (low, high) in file_ranges.items():
                    if low <= xcols <= high:
                        writers[file_name][1].writerow(data_info)
                        break
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

# Close all files
for file, _ in writers.values():
    file.close()

# Function to print the content of a CSV file
def print_file_content(file_name):
    file_path = os.path.join(output_dir, file_name)
    print(f"\nContents of {file_path}:")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
        
    with open(file_path, 'r') as f:
        reader = py_csv.reader(f)
        rows = list(reader)
        
        if len(rows) > 0:
            # Determine the maximum width of each column for proper formatting
            col_widths = [max(len(str(item)) for item in column) for column in zip(*rows)]
            
            # Print the headers
            headers = rows[0]
            header_row = " | ".join(f"{headers[i].ljust(col_widths[i])}" for i in range(len(headers)))
            print(header_row)
            print("-" * len(header_row))
            
            # Print each row with aligned columns
            for row in rows[1:]:
                formatted_row = " | ".join(f"{row[i].ljust(col_widths[i])}" for i in range(len(row)))
                print(formatted_row)

# Print the contents of the files
for file_name in file_ranges.keys():
    print_file_content(file_name)
