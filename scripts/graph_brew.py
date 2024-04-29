import json
import csv
import re
import subprocess
import os
import sys
import shutil
import importlib.util
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from collections import defaultdict

# Setting a default font for matplotlib to handle more character glyphs
rcParams['font.family'] = 'DejaVu Sans'

reorderings    = None  # Dictionary of reordering strategies with corresponding codes
KERNELS        = None          # List of kernels to use in benchmarks
graph_suites   = None  # Graph suites with their respective details
suite_dir      = None  # Base directory for graph data, removing it from graph_suites
kernel_results = None
# Define constants for benchmark settings
NUM_TRIALS     = 1        # Number of times to run each benchmark
NUM_ITERATIONS = 1    # Number of iterations per trial (if applicable)
FLUSH_CACHE    = 0       # Whether to flush cache before each run
PARALLEL       = os.cpu_count()  # Use all available CPU cores

# Directory setup for storing results
results_dir      = None
graph_csv_dir    = None      # Directory for CSV files
graph_charts_dir = None  # Directory for charts
graph_raw_dir    = None      # Directory for raw outputs

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

def import_check_install(package_name):
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        print(f"{package_name} is not installed. Installing...")
        subprocess.run(["pip", "install", package_name])

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
def run_benchmark(kernel, graph_path, reorder_code, graph_symbol, reorder_name):
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
        output_filename = f"{kernel}_{graph_symbol}_{reorder_name}_output.txt"
        output_filepath = os.path.join(graph_raw_dir, output_filename)
        with open(output_filepath, 'w') as f:
            f.write("Standard Output:\n" + output_text + "\n\nError Output:\n" + error_text)

        return output_text if output_text else error_text
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}\nError: {e.stderr.decode()}")
        return None

def initialize_kernel_results():
    """ Initializes a nested dictionary for storing benchmark results by kernel and category. """
    return {kernel: defaultdict(lambda: defaultdict(dict)) for kernel in KERNELS}


def graph_results_from_csv():
    """Parses the benchmark output to extract timing data based on predefined patterns."""
    for kernel in KERNELS:
        for category, patterns in time_patterns.items():
            filename = f"{kernel}_{category}_results"
            csv_file_path = os.path.join(graph_csv_dir, f"{filename}.csv")
            print(csv_file_path, graph_charts_dir, category)
            # with open(csv_path, 'w', newline='') as file:
            # create_pandas_bar_graph(csv_path, chart_path, category)
            if os.path.exists(csv_file_path):
                create_seaborn_bar_graph(csv_file_path, graph_charts_dir, category)

def write_results_to_csv(config_file_name, kernel, kernel_data):
    for category, category_data in kernel_data.items():
        # Prepare CSV file path
        filename = f"{kernel}_{category}_results"
        csv_file_path = os.path.join(graph_csv_dir, f"{filename}.csv")
        
        # Determine the fieldnames based on available metrics
        fieldnames = ['Graph'] + list(next(iter(category_data.values())).keys())

        with open(csv_file_path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            # Write data to CSV
            for graph, graph_data in category_data.items():
                row = {'Graph': graph}
                for reordering, value in graph_data.items():
                    row[reordering] = value[next(iter(value))]
                writer.writerow(row)

            print(f"Data written to {csv_file_path}")

        create_seaborn_bar_graph(csv_file_path, graph_charts_dir, category)

def create_seaborn_bar_graph(csv_file, output_folder, category):
    if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
        print(f"File {csv_file} does not exist or is empty.")
        return

    df = pd.read_csv(csv_file)
    if df.empty:
        print("DataFrame is empty.")
        return

    df_long = df.melt(id_vars=[df.columns[0]], var_name='Group', value_name='Value')
    category_title = category.replace('_', ' ').capitalize()
    if 'time' in category.lower():
        category_title += ' (s)'

    base_palette = [
        '#1f88e5', '#91caf9',   # Original blues
        '#ffe082', '#ffa000',   # Original yellows
        '#174a7e', '#7ab8bf',   # Additional blues and teals
        '#ffc857', '#dbab09',   # More yellows and golds
        '#44a248',              # Introducing green
        '#606c38', '#283d3b'    # Neutrals and dark accents
    ]

    num_groups = df_long['Group'].nunique()
    if num_groups > len(base_palette):
        # Extend the palette by repeating it
        repeats = -(-num_groups // len(base_palette))  # Ceiling division
        color_palette = (base_palette * repeats)[:num_groups]
    else:
        color_palette = base_palette[:num_groups]

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Graph', y='Value', hue='Group', data=df_long, palette=color_palette, edgecolor='black', width=0.5, linewidth=2.5)
    plt.xticks(rotation=45, ha='right', fontsize=14, fontweight='bold')
    plt.xlabel('Graphs', fontsize=18, fontweight='bold')
    plt.ylabel(category_title, fontsize=16, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)
    plt.tight_layout()

    filename = os.path.splitext(os.path.basename(csv_file))[0]
    for ext in ['svg', 'pdf']:
        output_path = os.path.join(output_folder, f"{filename}_{category}.{ext}")
        plt.savefig(output_path)
    plt.close()

def create_seaborn_bar_graph(csv_file, output_folder, category):
    if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
        print(f"File {csv_file} does not exist or is empty.")
        return

    df = pd.read_csv(csv_file)
    if df.empty:
        print("DataFrame is empty.")
        return

    df_long = df.melt(id_vars=[df.columns[0]], var_name='Group', value_name='Value')
    category_title = category.replace('_', ' ').capitalize()
    if 'time' in category.lower():
        category_title += ' (s)'

    base_palette = [
        '#1f88e5', '#91caf9', '#ffe082', '#ffa000', '#174a7e', '#7ab8bf',
        '#ffc857', '#dbab09', '#44a248', '#606c38', '#283d3b'
    ]
    num_groups = df_long['Group'].nunique()
    if num_groups > len(base_palette):
        repeats = -(-num_groups // len(base_palette))  # Ceiling division
        color_palette = (base_palette * repeats)[:num_groups]
    else:
        color_palette = base_palette[:num_groups]

    plt.figure(figsize=(12, 6))
    bar_plot = sns.barplot(x='Graph', y='Value', hue='Group', data=df_long, palette=color_palette, edgecolor='black', width=0.5, linewidth=2.5)

    # Calculate and plot geometric mean
    geom_means = df_long.groupby('Group')['Value'].apply(lambda x: np.exp(np.log(x).mean())).reset_index()
    geom_means['Graph'] = 'Geometric Mean'
    sns.barplot(x='Graph', y='Value', hue='Group', data=geom_means, palette=color_palette, edgecolor='black', width=0.5, linewidth=2.5, ax=plt.gca(), legend=False)

    # Find the position for the vertical line
    unique_graphs = df_long['Graph'].nunique()  # Number of unique graphs
    # The position is after the last plot of the initial set of graphs
    line_position = unique_graphs - 0.5

    plt.axvline(x=line_position, color='gray', linestyle='--', linewidth=2)

    plt.xticks(rotation=45, ha='right', fontsize=14, fontweight='bold')
    plt.xlabel('Graphs', fontsize=18, fontweight='bold')
    plt.ylabel(category_title, fontsize=16, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)
    plt.tight_layout()

    # Position the legend outside the plot area
    lgd = plt.legend(title='Reordering', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=12, title_fontsize=14)
    lgd.get_frame().set_linewidth(1.5)  # Increase the frame width


    filename = os.path.splitext(os.path.basename(csv_file))[0]
    for ext in ['svg', 'pdf']:
        output_path = os.path.join(output_folder, f"{filename}_{category}.{ext}")
        plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def parse_timing_data(output):
    """Parses the benchmark output to extract timing data based on predefined patterns."""
    time_data = {}
    for category, patterns in time_patterns.items():
        time_data[category] = {}
        for key, regex in patterns.items():
            match = regex.search(output)
            if match:
                time_data[category][key] = float(match.group(1))
    return time_data

def run_and_parse_benchmarks(config_file_name):
    global kernel_results
    
    """ Executes benchmarks for each kernel, graph, and reordering, then parses and stores the results. """
    for kernel in KERNELS:
        for suite_name, details in graph_suites.items():
            for graph in details["graphs"]:
                graph_name   = graph["name"]
                graph_symbol = graph["symbol"]
                graph_type   = graph["type"]
                graph_path = f"{suite_dir}/{suite_name}/{graph_symbol}/graph.{graph_type}"
                for reorder_name, reorder_code in reorderings.items():
                    output = run_benchmark(kernel, graph_path, reorder_code, graph_symbol, reorder_name)
                    if output:
                        time_data = parse_timing_data(output)
                        # print(time_data)
                        for category, patterns in time_patterns.items():
                        # Update reorder_time
                            if category in time_data:
                                for key, value in time_data[category].items():
                                    if reorder_name not in kernel_results[kernel][category][graph_symbol]:
                                        kernel_results[kernel][category][graph_symbol][reorder_name] = {}
                                    kernel_results[kernel][category][graph_symbol][reorder_name][key] = value
                                    print(f"{kernel:<7} {graph_symbol:<7} {category:<15} {key:<13}: {value:7.7f}(s)")

        if kernel_results[kernel]:  # Ensure there is data to process
            write_results_to_csv(config_file_name, kernel, kernel_results[kernel])




def main(config_file):
    global reorderings
    global graph_suites
    global suite_dir
    global KERNELS 
    global kernel_results
    # Directory setup for storing results
    global results_dir
    global graph_csv_dir
    global graph_charts_dir
    global graph_raw_dir

    # config_file    = "scripts/config/lite.json"  # Specify the path to your JSON configuration file
    graph_download_script = "./scripts/graph_download.py"  # Specify the path to your other Python script
   
    config_file_name = os.path.splitext(os.path.basename(config_file))[0]
    # Directory setup for storing results
    results_dir = f"bench/results/{config_file_name}"
    graph_csv_dir = os.path.join(results_dir, "data_csv")      # Directory for CSV files
    graph_charts_dir = os.path.join(results_dir, "data_charts")  # Directory for charts
    graph_raw_dir = os.path.join(results_dir, "data_raw")      # Directory for raw outputs

    # Load configuration settings from the specified JSON file
    with open(config_file, 'r') as f:
        config = json.load(f)
        reorderings = config['reorderings']  # Dictionary of reordering strategies with corresponding codes
        KERNELS = config['kernels']          # List of kernels to use in benchmarks
        graph_suites = config['graph_suites']  # Graph suites with their respective details
        suite_dir = graph_suites.pop('suite_dir')  # Base directory for graph data, removing it from graph_suites

        if os.path.exists(graph_csv_dir):
            print(f"Suite directory {graph_csv_dir} already exists.")
            graph_results_from_csv()
            return

        # Ensure all directories exist
        os.makedirs(graph_csv_dir, exist_ok=True)
        os.makedirs(graph_charts_dir, exist_ok=True)
        os.makedirs(graph_raw_dir, exist_ok=True)

        # Call the other Python script with the specified configuration file
        subprocess.run(["python3", graph_download_script, config_file])

    kernel_results = initialize_kernel_results()
    run_and_parse_benchmarks(config_file_name)

    print("Benchmarking completed and data recorded in designated folders.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python graph_download.py config_file.json")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    main(config_file)