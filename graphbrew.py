import csv
import subprocess
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set a default font that includes more glyphs
rcParams['font.family'] = 'DejaVu Sans'

# Define the graph suites with their corresponding graph names and file types
graph_suites = {
    # "LAW": {
    #     "graphs": ["amazon-2008", "arabic-2005", "cnr-2000", "dblp-2010", "enron", "eu-2005",
    #                "hollywood-2009", "in-2004", "indochina-2004", "it-2004", "ljournal-2008",
    #                "sk-2005", "uk-2002", "uk-2005", "webbase-2001"],
    #     "file_type": "graph.el"
    # },
    "SNAP": {
        "graphs": ["cit-Patents", "com-Orkut", "soc-LiveJournal1", "soc-Pokec", "web-Google"],
        "file_type": "graph.el"
    },
    # "GAP": {
    #     "graphs": ["kron", "road", "twitter", "urand", "web"],
    #     "file_type": "graph.wel"
    # }
}

# Reordering strategies
# reorderings = {
#     "ORIGINAL": 0, "RANDOM": 1, "SORT": 2, "HUBSORT": 3, "HUBCLUSTER": 4,
#     "DBG": 5, "HUBSORTDBG": 6, "HUBCLUSTERDBG": 7, "RABBITORDER": 8,
#     "GORDER": 9, "CORDER": 10, "RCM": 11, "LeidenOrder": 12
# }
# Kernel configurations
# KERNELS = ['bc', 'bfs', 'cc', 'cc_sv', 'pr', 'pr_spmv', 'sssp', 'tc']

reorderings = {
    "ORIGINAL": 0,"RABBITORDER": 8, "LeidenOrder": 12
}
KERNELS = ['bfs', 'pr_spmv']

NUM_TRIALS   = 1
NUM_ITERAIONS= 1
FLUSH_CACHE  = 1
PARALLEL     = os.cpu_count()  # Use all available CPU cores

# Setup directories
results_dir = "bench/results"
raw_csv_dir = os.path.join(results_dir, "csv_tables")
graph_charts_dir = os.path.join(results_dir, "charts")

os.makedirs(raw_csv_dir, exist_ok=True)
os.makedirs(graph_charts_dir, exist_ok=True)

# Function to execute the benchmark command
def run_benchmark(kernel, graph_path, reorder_code):
    
    RUN_PARAMS = [f"-n{NUM_TRIALS}", f"-r{reorder_code}"]
    GRAPH_BENCH= ["-f", f"{graph_path}" ]

    if kernel in ['pr', 'pr_spmv']:
        RUN_PARAMS.append(f"-i {NUM_ITERAIONS}")  # Number of iterations for PageRank kernels
    if kernel == 'tc':
        RUN_PARAMS.append("-s")  # Specific flag for 'tc' kernel

    cmd = [
        f"make run-{kernel}",
        f"GRAPH_BENCH='{ ' '.join(GRAPH_BENCH) }'",
        f"RUN_PARAMS='{ ' '.join(RUN_PARAMS) }'",
        f"FLUSH_CACHE={FLUSH_CACHE}",
        f"PARALLEL={PARALLEL}"
    ]
   
    # Join command list into a single string to execute
    cmd = " ".join(cmd)
    try:
        output = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output.stdout.decode()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}\nError: {e.stderr.decode()}")
        return None

# Function to generate SVG charts from CSV data
def generate_svg_chart(csv_path):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(12, 6))
    for column in df.columns[1:]:  # Exclude the 'graph_name' column for plotting
        plt.plot(df['graph_name'], df[column], marker='o', label=column)
    plt.title('Reordering Strategy Performance')
    plt.xlabel('Graph Name')
    plt.ylabel('Execution Time (s)')
    plt.xticks(rotation=45)
    plt.legend(title='Reordering Strategies')
    plt.tight_layout()
    svg_path = os.path.join(graph_charts_dir, os.path.basename(csv_path).replace('.csv', '.svg'))
    plt.savefig(svg_path)
    plt.close()

# Write results to CSV and copy to directories, and generate charts
def write_to_csv_and_copy(kernel, results):
    csv_filename = f"{kernel}.csv"
    csv_path = os.path.join(raw_csv_dir, csv_filename)
    with open(csv_path, mode='w', newline='') as file:
        fieldnames = ['graph_name'] + list(reorderings.keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for graph_name, timings in results.items():
            row = {'graph_name': graph_name}
            row.update(timings)
            writer.writerow(row)
    # Copy CSV to graph charts directory and generate SVG chart
    shutil.copy(csv_path, graph_charts_dir)
    generate_svg_chart(csv_path)

# Main execution loop
for kernel in KERNELS:
    kernel_results = {}
    for suite, details in graph_suites.items():
        for graph in details["graphs"]:
            graph_results = {}
            graph_path = f"/media/cmv6ru/Data/00_GraphDatasets/{suite}/{graph}/{details['file_type']}"
            for reorder_name, reorder_code in reorderings.items():
                print(f"Processing {kernel} on {graph} with {reorder_name} reordering...")
                result = run_benchmark(kernel, graph_path, reorder_code)
                graph_results[reorder_name] = result  # Storing command output directly
            kernel_results[graph] = graph_results
    write_to_csv_and_copy(kernel, kernel_results)

print("Benchmarking completed and data recorded in designated folders.")
