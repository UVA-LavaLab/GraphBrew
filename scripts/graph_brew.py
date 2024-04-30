#!/usr/bin/env python3

import json
import csv
import re
import subprocess
import os
import sys
import shutil
from pathlib import Path
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
prereorder_codes  = None
postreorder_codes = None
baselines_speedup = None
baselines_overhead= None
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
        'Corder': re.compile(r"\bCorder\b Time:\s*([\d\.]+)"),
        'DBG': re.compile(r"\bDBG\b Map Time:\s*([\d\.]+)"),
        'Gorder': re.compile(r"\bGorder\b Time:\s*([\d\.]+)"),
        'HubClusterDBG': re.compile(r"\bHubClusterDBG\b Map Time:\s*([\d\.]+)"),
        'HubCluster': re.compile(r"\bHubCluster\b Map Time:\s*([\d\.]+)"),
        'HubSortDBG': re.compile(r"\bHubSortDBG\b Map Time:\s*([\d\.]+)"),
        'HubSort': re.compile(r"\bHubSort\b Map Time:\s*([\d\.]+)"),
        'Leiden': re.compile(r"\bLeiden\b Time:\s*([\d\.]+)"),
        'Original': re.compile(r"\bOriginal\b Time:\s*([\d\.]+)"),
        'RabbitOrder': re.compile(r"\bRabbitOrder\b Time:\s*([\d\.]+)"),
        'Random': re.compile(r"\bRandom\b Map Time:\s*([\d\.]+)"),
        'RCM': re.compile(r"\bRCMorder\b Time:\s*([\d\.]+)"),
        'Sort': re.compile(r"\bSort\b Map Time:\s*([\d\.]+)")
    },
    'trial_time': {
        'Average': re.compile(r"\bAverage\b Time:\s*([\d\.]+)")
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
def run_benchmark(kernel, graph_path, reorder_code, graph_symbol, reorder_name, prereorder_codes, postreorder_codes):
    # Clear CPU cache to simulate a fresh start for benchmarking
    clear_cpu_cache()

    RUN_PARAMS = []
    # Start with the number of trials parameter
    if kernel != 'converter':
        RUN_PARAMS.append(f"-n{NUM_TRIALS}")

    EXTENTION_PARAMS = []
    # Add prereorder codes if any
    RUN_PARAMS.extend([f"-o{code}" for code in prereorder_codes])
    EXTENTION_PARAMS.extend([f"{code}" for code in prereorder_codes])
    # Main reorder code
    RUN_PARAMS.append(f"-o{reorder_code}")
    EXTENTION_PARAMS.append(f"{reorder_code}")
    # Add postreorder codes if any
    RUN_PARAMS.extend([f"-o{code}" for code in postreorder_codes])
    EXTENTION_PARAMS.extend([f"{code}" for code in postreorder_codes])

    # Specify the graph file
    GRAPH_BENCH  = ["-f", f"{graph_path}"]
    OUTPUT_BENCH = []
    cmd = []

    # Additional kernel-specific parameters
    if kernel in ['pr', 'pr_spmv']:
        RUN_PARAMS.append(f"-i {NUM_ITERATIONS}")  # PageRank iterations
    if kernel == 'tc':
        RUN_PARAMS.append("-s")  # Special flag for 'tc' kernel

    if kernel == 'converter':
        output_graph_path = Path(graph_path)
        new_graph_path = output_graph_path.with_name(f"{output_graph_path.stem}_{'_'.join(EXTENTION_PARAMS)}.sg")
        OUTPUT_BENCH = ["-b", f"{new_graph_path}"]
        GRAPH_BENCH.extend(OUTPUT_BENCH)
        cmd = [
            f"make run-{kernel}",
            f"GRAPH_BENCH='{ ' '.join(GRAPH_BENCH) }'",
            f"RUN_PARAMS='{ ' '.join(RUN_PARAMS) }'",
            f"FLUSH_CACHE={FLUSH_CACHE}",
            f"PARALLEL={PARALLEL}"
        ]
    else:
        # Assemble the command
        cmd = [
            f"make run-{kernel}",
            f"GRAPH_BENCH='{ ' '.join(GRAPH_BENCH) }'",
            f"RUN_PARAMS='{ ' '.join(RUN_PARAMS) }'",
            f"FLUSH_CACHE={FLUSH_CACHE}",
            f"PARALLEL={PARALLEL}"
        ]

    # Convert list to a space-separated string for subprocess execution
    cmd = " ".join(cmd)
    print(cmd)
    # Execute the command
    try:
        output = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output_text = output.stdout.decode()
        error_text = output.stderr.decode()

        # Output handling
        if error_text:
            print("Error Output:", error_text)

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


def load_and_prepare_data(csv_file):
    """Load the CSV file into a DataFrame and melt it for Seaborn plotting."""
    df = pd.read_csv(csv_file)
    df_melted = df.melt(id_vars='Graph', var_name='Reordering', value_name='Trial Time')
    return df_melted

def load_and_prepare_data_speedup(csv_file, baseline):
    """Load the CSV file into a DataFrame and melt it for Seaborn plotting."""
    df = pd.read_csv(csv_file)
    new_df = df[['Graph']].copy()
    baseline_time = df[baseline]
    for column in df.columns:
        if column not in ['Graph', baseline]:  # Avoid the graph and baseline column itself
            new_df[column] = baseline_time / df[column]

    df_melted = new_df.melt(id_vars='Graph', var_name='Reordering', value_name='Trial Time')
    return df_melted

def load_and_prepare_data_overhead(csv_file, baseline):
    """Load the CSV file into a DataFrame and melt it for Seaborn plotting."""
    df = pd.read_csv(csv_file)
    new_df = df[['Graph']].copy()
    baseline_time = df[baseline]
    for column in df.columns:
        if column not in ['Graph', baseline]:  # Avoid the graph and baseline column itself
            new_df[column] = df[column] / baseline_time 

    df_melted = new_df.melt(id_vars='Graph', var_name='Reordering', value_name='Trial Time')
    return df_melted

def add_geometric_means(df):
    """Add a row for geometric means to the DataFrame, calculated across all graphs for each reordering."""
    # Calculate geometric means for each reordering, ignoring non-positive values
    gm = df[df['Trial Time'] > 0].groupby('Reordering')['Trial Time'].apply(
        lambda x: np.exp(np.mean(np.log(x)))
    ).reset_index()
    gm['Graph'] = 'GM'
    # Concatenate geometric means with the main data frame
    return pd.concat([df, pd.DataFrame(gm)], ignore_index=True)

def plot_data(df,csv_file, output_folder, category, value_name="time"):
    """Create a Seaborn bar plot from the DataFrame."""

    base_palette = [
        '#1f88e5', '#91caf9', '#ffe082', '#ffa000', '#174a7e', '#7ab8bf',
        '#ffc857', '#dbab09', '#44a248', '#606c38', '#283d3b'
    ]
    num_groups = df['Reordering'].nunique()
    if num_groups > len(base_palette):
        repeats = -(-num_groups // len(base_palette))  # Ceiling division
        color_palette = (base_palette * repeats)[:num_groups]
    else:
        color_palette = base_palette[:num_groups]

    category_title = category.replace('_', ' ').capitalize()

    # plt.figure(figsize=(8, 4))
    # bar_plot = sns.barplot(x='Graph', y='Trial Time', hue='Reordering', data=df, palette=color_palette, edgecolor='black', width=0.7, linewidth=2.5)
# Calculate the accurate position for the vertical line
    num_graphs = len(df['Graph'].unique())  # Total number of graphs including 'GM'
    bar_width = 0.8  # Adjust this if you want wider or narrower bars
    gm_x_position = num_graphs - 1 - bar_width/(1 + bar_width) # Position the line right before the 'GM' group

    plt.figure(figsize=(10, 6)) 
    bar_plot = sns.barplot(x='Graph', y='Trial Time', hue='Reordering', data=df, 
                           palette=color_palette, edgecolor='black', width=bar_width, linewidth=2.5) 

    plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold') 
    plt.axvline(x=gm_x_position, color='red', linestyle='--', linewidth=2, label='Geometric Mean')  

    plt.xlabel('Graphs', fontsize=18, fontweight='bold')
    plt.ylabel(category_title, fontsize=16, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)
    plt.tight_layout()

    # Adjust legend position to be outside the plot
    plt.legend(title='Reordering', loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    
    # Adjust the layout to make space for the legend
    plt.subplots_adjust(right=0.75)  # Adjust this value to fit your specific plot and legend size


    filename = os.path.splitext(os.path.basename(csv_file))[0]
    for ext in ['svg', 'pdf']:
        output_path = os.path.join(output_folder, f"{filename}_{category}_{value_name}.{ext}")
        plt.savefig(output_path)
    plt.close()

def clean_category_name(category):
    """Remove '_time' suffix from category name if present, using split and join."""
    parts = category.split('_')
    if "time" in parts:
        parts.remove('time')  # Remove the 'time' part from the list
    cleaned_category = '_'.join(parts)  # Rejoin the parts without 'time'
    return cleaned_category

def create_seaborn_bar_graph(csv_file, output_folder, category):
    baseline_speedup_columns = baselines_speedup.keys()  # Get baseline column names
    baselines_overhead_columns = baselines_overhead.keys()  # Get baseline column names
    
     # Calculate speedup for each column against each baseline
    df = load_and_prepare_data(csv_file)
    df = add_geometric_means(df)
    plot_data(df,csv_file, output_folder, category)
    
    if 'trial_time' in category:
        for baseline in baseline_speedup_columns:
            clean_category = clean_category_name(category)
            df = load_and_prepare_data_speedup(csv_file, baseline)
            df = add_geometric_means(df)
            plot_data(df,csv_file, output_folder, f"{clean_category}_Speedup",f"speedup_{baseline}")
    if 'reorder_time' in category:   
        for baseline in baselines_overhead_columns:
            clean_category = clean_category_name(category)
            df = load_and_prepare_data_speedup(csv_file, baseline)
            df = add_geometric_means(df)
            plot_data(df,csv_file, output_folder, f"{clean_category}_Overhead",f"overhead_{baseline}")


def parse_timing_data(output, reorderings, prereorder_codes, postreorder_codes):
    """Parses the benchmark output to extract timing data based on predefined patterns."""
    found_reorder_times = {}
    time_data = {category: {} for category in time_patterns}

    match_count = 0
    main_match = len(prereorder_codes)+1
    # Process each line of the output
    for line in output.strip().split('\n'):
        for category, patterns in time_patterns.items():
            for key, regex in patterns.items():
                matches = regex.findall(line)
                if matches:
                    if category == 'reorder_time':
                        match_count += 1
                        if(match_count == main_match):
                            reorder_code = reorderings[key]
                            time_key = key  # Use the main reorder time
                            found_reorder_times.setdefault(reorder_code, []).append((time_key, float(matches[0])))
                    else:
                        # Handle other times
                        time_data[category][key] = float(matches[0])

    if found_reorder_times:
        for code, times_list in found_reorder_times.items():
            sorted_times = sorted(times_list, key=lambda x: x[0] not in prereorder_codes and x[0] not in postreorder_codes)
            for time_key, time_value in sorted_times:
                time_data['reorder_time'][time_key] = time_value

    return time_data


def run_and_parse_benchmarks(config_file_name):
    global kernel_results
    global prereorder_codes
    global postreorder_codes
    
    """ Executes benchmarks for each kernel, graph, and reordering, then parses and stores the results. """
    for kernel in KERNELS:
        for suite_name, details in graph_suites.items():
            for graph in details["graphs"]:
                graph_name   = graph["name"]
                graph_symbol = graph["symbol"]
                graph_type   = graph["type"]
                graph_path = f"{suite_dir}/{suite_name}/{graph_symbol}/graph.{graph_type}"
                for reorder_name, reorder_code in reorderings.items():
                    output = run_benchmark(kernel, graph_path, reorder_code, graph_symbol, reorder_name, prereorder_codes, postreorder_codes)
                    if output:
                        time_data = parse_timing_data(output, reorderings, prereorder_codes, postreorder_codes)
                        # print(time_data)
                        for category, patterns in time_patterns.items():
                        # Update reorder_time
                            if category in time_data:
                                for key, value in time_data[category].items():
                                    if reorder_name not in kernel_results[kernel][category][graph_symbol]:
                                        kernel_results[kernel][category][graph_symbol][reorder_name] = {}
                                    kernel_results[kernel][category][graph_symbol][reorder_name][key] = value
                                    print(f"{kernel:<7} {graph_symbol:<7} {category:<15} {key:<13}: {value}(s)")

        if kernel_results[kernel]:  # Ensure there is data to process
            write_results_to_csv(config_file_name, kernel, kernel_results[kernel])




def main(config_file):
    global reorderings
    global graph_suites
    global suite_dir
    global KERNELS 
    global kernel_results
    global prereorder_codes
    global postreorder_codes
    # Directory setup for storing results
    global results_dir
    global graph_csv_dir
    global graph_charts_dir
    global graph_raw_dir
    global baselines_speedup
    global baselines_overhead

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
        config       = json.load(f)
        reorderings  = config['reorderings']  # Dictionary of reordering strategies with corresponding codes
        KERNELS      = config['kernels']          # List of kernels to use in benchmarks
        graph_suites = config['graph_suites']  # Graph suites with their respective details
        suite_dir    = graph_suites.pop('suite_dir')  # Base directory for graph data, removing it from graph_suites
        # Extract prereorder and postreorder codes
        prereorder_codes   = [config['prereorder'].get(key, []) for key in config.get('prereorder', {})]
        postreorder_codes  = [config['postreorder'].get(key, []) for key in config.get('postreorder', {})]
        baselines_speedup  = config.get('baselines_speedup', {})
        baselines_overhead = config.get('baselines_overhead', {})

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