import json
import csv
import re
import subprocess
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

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
FLUSH_CACHE = 1       # Whether to flush cache before each run
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
        'Original': re.compile(r"Original time:\s+([\d\.]+)"),
        'Random Map': re.compile(r"Random Map Time:\s+([\d\.]+)"),
        'Sort Map': re.compile(r"Sort Map Time:\s+([\d\.]+)"),
        'HubSort Map': re.compile(r"HubSort Map Time:\s+([\d\.]+)"),
        'HubCluster Map': re.compile(r"HubCluster Map Time:\s+([\d\.]+)"),
        'DBG Map': re.compile(r"DBG Map Time:\s+([\d\.]+)"),
        'HubSortDBG Map': re.compile(r"HubSortDBG Map Time:\s+([\d\.]+)"),
        'HubClusterDBG Map': re.compile(r"HubClusterDBG Map Time:\s+([\d\.]+)"),
        'RabbitOrder': re.compile(r"RabbitOrder time:\s+([\d\.]+)"),
        'Gorder': re.compile(r"Gorder time:\s+([\d\.]+)"),
        'Corder': re.compile(r"Corder Time:\s+([\d\.]+)"),
        'RCMorder': re.compile(r"RCMorder time:\s+([\d\.]+)"),
        'Leiden': re.compile(r"Leiden time:\s+([\d\.]+)")
    },
    'trial_time': {
        'Average': re.compile(r"Average Time:\s+([\d\.]+)")
    }
}

# Function to run benchmarks, handle output, and save raw data
def run_benchmark(kernel, graph_path, reorder_code, graph_name, reorder_name):
    # Prepare benchmark command with parameters
    RUN_PARAMS = [f"-n{NUM_TRIALS}", f"-r{reorder_code}"]
    GRAPH_BENCH = ["-f", f"{graph_path}"]

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

        # Save the raw output to a file
        output_filename = f"{kernel}_{graph_name}_{reorder_name}_output.txt"
        output_filepath = os.path.join(graph_raw_dir, output_filename)
        with open(output_filepath, 'w') as f:
            f.write(output_text)

        return output_text
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}\nError: {e.stderr.decode()}")
        return None

# Function to parse and extract timing data from benchmark output
def parse_timing_data(output):
    time_data = {}
    # Iterate over both categories and all strategy patterns
    for category, patterns in time_patterns.items():
        time_data[category] = {}
        for key, pattern in patterns.items():
            match = pattern.search(output)
            if match:
                time_data[category][key] = float(match.group(1))
    return time_data

# Function to write results to CSV, copy them to designated directories, and generate charts
def write_to_csv_and_copy(kernel, results):
    for category, data in results.items():
        csv_filename = f"{kernel}_{category}.csv"
        csv_path = os.path.join(graph_csv_dir, csv_filename)
        with open(csv_path, mode='w', newline='') as file:
            fieldnames = ['Graph Name'] + list(time_patterns[category].keys())
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for graph_name, timings in data.items():
                row = {'Graph Name': graph_name}
                row.update(timings)
                writer.writerow(row)
        shutil.copy(csv_path, graph_charts_dir)
        generate_svg_chart(csv_path)

# Main execution loop to process each kernel and graph suite
for kernel in KERNELS:
    kernel_results = {}
    for suite_name, details in graph_suites.items():
        for graph in details["graphs"]:
            graph_results = {'reorder_time': {}, 'trial_time': {}}
            graph_path = f"{suite_dir}/{suite_name}/{graph}/{details['file_type']}"
            for reorder_name, reorder_code in reorderings.items():
                print(f"Processing {kernel} on {graph} with {reorder_name} reordering...")
                output = run_benchmark(kernel, graph_path, reorder_code, graph, reorder_name)
                if output:
                    time_data = parse_timing_data(output)
                    for category in time_data:
                        graph_results[category][reorder_name] = time_data[category]
            kernel_results[graph] = graph_results
    write_to_csv_and_copy(kernel, kernel_results)

print("Benchmarking completed and data recorded in designated folders.")