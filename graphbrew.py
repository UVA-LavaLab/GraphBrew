import json
import csv
import re
import subprocess
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from collections import defaultdict

# Setting a default font for matplotlib to handle more character glyphs
rcParams['font.family'] = 'DejaVu Sans'

# Loading configuration settings from a JSON file
with open('config/lite.json', 'r') as f:
    config = json.load(f)
reorderings = config['reorderings']  # Dictionary of reordering strategies with corresponding codes
KERNELS = config['kernels']          # List of kernels to use in benchmarks
graph_suites = config['graph_suites']  # Graph suites with their respective details
suite_dir = graph_suites.pop('suite_dir')  # Base directory for graph data, removing it from graph_suites

# Define constants for benchmark settings
NUM_TRIALS = 1        # Number of times to run each benchmark
NUM_ITERATIONS = 1    # Number of iterations per trial (if applicable)
FLUSH_CACHE = 0       # Whether to flush cache before each run
PARALLEL = os.cpu_count()  # Use all available CPU cores

# Directory setup for storing results
results_dir = "bench/results"
graph_csv_dir = os.path.join(results_dir, "data_csv")      # Directory for CSV files
graph_charts_dir = os.path.join(results_dir, "data_charts")  # Directory for charts
graph_raw_dir = os.path.join(results_dir, "data_raw")      # Directory for raw outputs

# Ensure all directories exist
os.makedirs(graph_csv_dir, exist_ok=True)
os.makedirs(graph_charts_dir, exist_ok=True)
os.makedirs(graph_raw_dir, exist_ok=True)

# Regular expressions for parsing timing data from benchmark outputs
time_patterns = {
    'reorder_time': {
        'Original': re.compile(r"Original Time:\s+([\d\.]+)"),
        'Random': re.compile(r"Random Map Time:\s+([\d\.]+)"),
        'Sort': re.compile(r"Sort Map Time:\s+([\d\.]+)"),
        'HubSort': re.compile(r"HubSort Map Time:\s+([\d\.]+)"),
        'HubCluster': re.compile(r"HubCluster Map Time:\s+([\d\.]+)"),
        'DBG': re.compile(r"DBG Map Time:\s+([\d\.]+)"),
        'HubSortDBG': re.compile(r"HubSortDBG Map Time:\s+([\d\.]+)"),
        'HubClusterDBG': re.compile(r"HubClusterDBG Map Time:\s+([\d\.]+)"),
        'RabbitOrder': re.compile(r"RabbitOrder Time:\s+([\d\.]+)"),
        'Gorder': re.compile(r"Gorder Time:\s+([\d\.]+)"),
        'Corder': re.compile(r"Corder Time:\s+([\d\.]+)"),
        'RCM': re.compile(r"RCMorder Time:\s+([\d\.]+)"),
        'Leiden': re.compile(r"Leiden Time:\s+([\d\.]+)")
    },
    'trial_time': {
        'Average': re.compile(r"Average Time:\s+([\d\.]+)")
    }
}


def clear_cpu_cache(size=100*1024*1024):  # 100 MB
    """
    A simple attempt to 'flush' CPU cache by loading large data into memory,
    which likely causes much of the cache to be replaced.
    """
    try:
        _ = bytearray(size)  # Allocate a large amount of data
        # print("Performed large memory allocation to disrupt CPU cache.")
    except Exception as e:
        print(f"Failed to disrupt CPU cache: {e}")

# Function to run benchmarks, handle output, and save raw data
def run_benchmark(kernel, graph_path, reorder_code, graph_name, reorder_name):
    # Prepare benchmark command with parameters
    RUN_PARAMS = [f"-n{NUM_TRIALS}", f"-o{reorder_code}"]
    GRAPH_BENCH = ["-f", f"{graph_path}"]
    clear_cpu_cache();

    if kernel in ['pr', 'pr_spmv']:
        RUN_PARAMS.append(f"-i {NUM_ITERATIONS}")  # Add iterations for specific kernels
    if kernel == 'tc':
        RUN_PARAMS.append("-s")  # Special flag for 'tc' kernel

    cmd = [
        f"make run-{kernel}",
        f"GRAPH_BENCH='{ ' '.join(GRAPH_BENCH) }'",
        f"RUN_PARAMS='{ ' '.join(RUN_PARAMS) }'",
        f"FLUSH_CACHE={FLUSH_CACHE}",
        f"PARALLEL={PARALLEL}"
    ]

    cmd = " ".join(cmd)
    try:
        output = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output_text = output.stdout.decode()
        error_text = output.stderr.decode()

        if error_text:
            print("Error Output:", error_text)

        # Save both output and errors to the file
        output_filename = f"{kernel}_{graph_name}_{reorder_name}_output.txt"
        output_filepath = os.path.join(graph_raw_dir, output_filename)
        with open(output_filepath, 'w') as f:
            f.write("Standard Output:\n" + output_text + "\n\nError Output:\n" + error_text)

        return output_text if output_text else error_text
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}\nError: {e.stderr.decode()}")
        return None

# Function to parse and extract timing data from benchmark output
def parse_timing_data(output):
    """
    Parses output to extract timing data based on defined regular expressions in time_patterns.
    """
    time_data = {}
    for category, patterns in time_patterns.items():
        time_data[category] = {}
        for key, regex in patterns.items():
            match = regex.search(output)
            if match:
                time_data[category][key] = float(match.group(1))
            # else:
            #     print(f"Warning: No data found for {key} in category {category}.")
            #     time_data[category][key] = None  # Optionally handle missing data
    return time_data
    
def generate_svg_chart(csv_path):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 5))
    for column in df.columns[1:]:  # Skip 'Graph Name' for plotting
        plt.plot(df['Graph Name'], df[column], marker='o', label=column)

    plt.title('Benchmark Results for ' + os.path.basename(csv_path).replace('.csv', ''))
    plt.xlabel('Graph Name')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    plt.legend(title='Reordering Strategies')
    plt.tight_layout()

    svg_path = csv_path.replace('.csv', '.svg')
    plt.savefig(svg_path)
    plt.close()
    print(f"Chart saved as {svg_path}")

def write_csv_for_category(category, data, base_dir):
    csv_filename = f"{category}_results.csv"
    csv_path = os.path.join(base_dir, csv_filename)
    fieldnames = ['Graph Name'] + sorted(data[next(iter(data))].keys())  # Dynamic field names based on keys

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for graph_name, timings in data.items():
            row = {'Graph Name': graph_name}
            row.update(timings)
            writer.writerow(row)

    shutil.copy(csv_path, graph_charts_dir)
    generate_svg_chart(csv_path)

def process_kernel_results(kernel_results):
    """
    Process kernel results for all categories defined in time_patterns.
    Each category's data is written to a separate CSV file and a corresponding chart is generated.

    :param kernel_results: Dictionary structured by graph and categories containing timing data.
    """
    # Iterate over all categories defined in time_patterns
    for category in time_patterns.keys():
        # Prepare data for this category
        category_data = {graph: data.get(category, {}) for graph, data in kernel_results.items()}
        # Write data to CSV and generate chart
        write_csv_for_category(category, category_data, graph_csv_dir)


# Function to initialize the kernel results data structure
def initialize_kernel_results():
    return defaultdict(lambda: defaultdict(dict))

kernel_results = initialize_kernel_results()

# Example of data collection (modify as per your existing setup)
for kernel in KERNELS:
    for suite_name, details in graph_suites.items():
        for graph in details["graphs"]:
            graph_path = f"{suite_dir}/{suite_name}/{graph}/{details['file_type']}"
            for reorder_name, reorder_code in reorderings.items():
                output = run_benchmark(kernel, graph_path, reorder_code, graph, reorder_name)
                if output:
                    time_data = parse_timing_data(output)
                    for category, times in time_data.items():
                        for key, value in times.items():
                            kernel_results[category][graph][key] = value
                            print(f"Processed {category} for {graph} under {key}: {value}")

    # print(kernel_results)

process_kernel_results(kernel_results)

print("Benchmarking completed and data recorded in designated folders.")
