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
results_dir = "bench/results"
graph_csv_dir = os.path.join(results_dir, "data_csv")      # Directory for CSV files
graph_charts_dir = os.path.join(results_dir, "data_charts")  # Directory for charts
graph_raw_dir = os.path.join(results_dir, "data_raw")      # Directory for raw outputs

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
            csv_path = os.path.join("bench", "results", "data_csv", f"{filename}.csv")
            chart_path = os.path.join("bench", "results", "data_charts")
            print(csv_path, chart_path, category)
            # with open(csv_path, 'w', newline='') as file:
            # create_pandas_bar_graph(csv_path, chart_path, category)
            if os.path.exists(csv_path):
                create_pandas_bar_graph(csv_path, chart_path, category)

def write_results_to_csv(kernel, kernel_data):
    for category, category_data in kernel_data.items():
        # Prepare CSV file path
        filename = f"{kernel}_{category}_results"
        csv_path = os.path.join("bench", "results", "data_csv", f"{filename}.csv")
        chart_path = os.path.join("bench", "results", "data_charts")
        
        # Determine the fieldnames based on available metrics
        fieldnames = ['Graph'] + list(next(iter(category_data.values())).keys())

        with open(csv_path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            # Write data to CSV
            for graph, graph_data in category_data.items():
                row = {'Graph': graph}
                for reordering, value in graph_data.items():
                    row[reordering] = value[next(iter(value))]
                writer.writerow(row)

            print(f"Data written to {csv_path}")

        create_seaborn_bar_graph(csv_path, chart_path, category)

def create_seaborn_bar_graph(csv_file, output_folder, category):
    # Read CSV file into a pandas DataFrame
    if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
        df = pd.read_csv(csv_file)
        # Check if DataFrame is not empty
        if not df.empty:
            # Prepare the data in long-form for seaborn
            df_long = df.melt(id_vars=[df.columns[0]], var_name='Group', value_name='Value')

            # Modify category label for the title and remove underscores
            category_title = category.replace('_', ' ').capitalize()
            if 'time' in category.lower():
                category_title += ' (s)'

            # Define the color palette from the image provided
            color_palette = ['#1f88e5', '#91caf9', '#ffe082', '#ffa000']

            # Initialize the matplotlib figure
            f, ax = plt.subplots(figsize=(12, 6))

            # Create grouped bar plot with seaborn
            sns.barplot(x='Graph', y='Value', hue='Group', errorbar=None, data=df_long, palette=color_palette, edgecolor='black', width=0.5, linewidth=2.5)

            # Fixing tick positions and labels for robustness
            ticks = ax.get_xticks()
            labels = [label.get_text() for label in ax.get_xticklabels()]
            ax.set_xticks(ticks)  # Ensure ticks are correctly set
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=14, fontweight='bold')  # Now safe to use

            # Set labels with the modified layout
            ax.set_xlabel('Graphs', fontsize=18, fontweight='bold')
            ax.set_ylabel(category_title, fontsize=16, fontweight='bold')

            # Add horizontal grid lines
            ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)

            # Set font size for ticks and adjust the layout
            ax.tick_params(axis='both', which='major', labelsize=14, width=2)
            plt.yscale('log')  # Set y-axis to logarithmic scale
            plt.tight_layout()

            # Remove legend if not needed
            # ax.legend().set_visible(False)

            # Save the plot as SVG and PDF in the output folder
            filename = os.path.splitext(os.path.basename(csv_file))[0]
            for ext in ['svg', 'pdf']:
                output_path = os.path.join(output_folder, f"{filename}_{category}.{ext}")
                plt.savefig(output_path)

def create_pandas_bar_graph(csv_file, output_folder, category):
    # Read CSV file into a pandas DataFrame
    if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
        df = pd.read_csv(csv_file)

        # Check if DataFrame is not empty
        if not df.empty:
            # Extract labels and data
            labels = df.iloc[:, 0]
            data = df.iloc[:, 1:]

            # Modify category label for the title and remove underscores
            category_title = category.replace('_', ' ').capitalize()
            if 'time' in category.lower():
                category_title += ' (s)'

            # Define the color palette from the image provided
            color_palette = ['#1f88e5', '#91caf9', '#ffe082', '#ffa000']

            # Create grouped bar plot with the new color palette
            # Adjust 'width' for narrow bars and 'linewidth' for thicker borders
            ax = data.plot(kind='bar', color=color_palette, figsize=(12, 6), width=0.5, edgecolor='black', linewidth=2.5, logy=True)

            # Set labels and title with the modified layout
            # Adjust 'fontsize' and 'fontweight' for thicker and larger font
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=14, fontweight='bold')
            ax.set_xlabel('Graphs', fontsize=18, fontweight='bold')
            ax.set_ylabel(category_title, fontsize=16, fontweight='bold')
            # ax.set_title(f"{category_title}", fontsize=18, fontweight='bold')

            # Add horizontal grid lines
            ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)

            # Set font size for ticks and adjust the layout
            ax.tick_params(axis='both', which='major', labelsize=14, width=2)
            plt.yscale('log')  # Set y-axis to logarithmic scale
            plt.tight_layout()

            # Save the plot as SVG and PDF in the output folder
            filename = os.path.splitext(os.path.basename(csv_file))[0]
            for ext in ['svg', 'pdf']:
                output_path = os.path.join(output_folder, f"{filename}_{category}.{ext}")
                plt.savefig(output_path)
        else:
            print(f"CSV file {csv_file} is empty. Skipping graph generation.")
    else:
        print(f"CSV file {csv_file} does not exist or is empty. No graph will be generated.")


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

def run_and_parse_benchmarks():
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
                        
                        # Update reorder_time
                        if 'reorder_time' in time_data:
                            for key, value in time_data['reorder_time'].items():
                                if reorder_name not in kernel_results[kernel]['reorder_time'][graph_symbol]:
                                    kernel_results[kernel]['reorder_time'][graph_symbol][reorder_name] = {}
                                kernel_results[kernel]['reorder_time'][graph_symbol][reorder_name][key] = value
                                print(f"{kernel:<7} {graph_symbol:<7} reorder_time    {key:<13}: {value:<7}(s)")
                        
                        # Update trial_time
                        if 'trial_time' in time_data:
                            for key, value in time_data['trial_time'].items():
                                if key == 'Average':  # Only store average trial time
                                    if reorder_name not in kernel_results[kernel]['trial_time'][graph_symbol]:
                                        kernel_results[kernel]['trial_time'][graph_symbol][reorder_name] = {}
                                    kernel_results[kernel]['trial_time'][graph_symbol][reorder_name][key] = value
                                    print(f"{kernel:<7} {graph_symbol:<7} trial_time      {key:<13}: {value:<7}(s)")

        if kernel_results[kernel]:  # Ensure there is data to process
            write_results_to_csv(kernel, kernel_results[kernel])




def main():
    global reorderings
    global graph_suites
    global suite_dir
    global KERNELS 
    global kernel_results

    dependencies = ['re', 'json', 'shutil', 'tarfile', 'csv', 'matplotlib', 'collections', 'seaborn']
    for dependency in dependencies:
        import_check_install(dependency)

    config_file    = "scripts/config/lite.json"  # Specify the path to your JSON configuration file
    graph_download_script = "./scripts/graph_download.py"  # Specify the path to your other Python script
    csv_dir = os.path.join("bench", "results", "data_csv")


    # Load configuration settings from the specified JSON file
    with open(config_file, 'r') as f:
        config = json.load(f)
        reorderings = config['reorderings']  # Dictionary of reordering strategies with corresponding codes
        KERNELS = config['kernels']          # List of kernels to use in benchmarks
        graph_suites = config['graph_suites']  # Graph suites with their respective details
        suite_dir = graph_suites.pop('suite_dir')  # Base directory for graph data, removing it from graph_suites

        if os.path.exists(csv_dir):
            print(f"Suite directory {csv_dir} already exists.")
            graph_results_from_csv()
            return

        # Ensure all directories exist
        os.makedirs(graph_csv_dir, exist_ok=True)
        os.makedirs(graph_charts_dir, exist_ok=True)
        os.makedirs(graph_raw_dir, exist_ok=True)

        # Call the other Python script with the specified configuration file
        subprocess.run(["python3", graph_download_script, config_file])

    kernel_results = initialize_kernel_results()
    run_and_parse_benchmarks()

    print("Benchmarking completed and data recorded in designated folders.")

if __name__ == "__main__":
    main()

   
