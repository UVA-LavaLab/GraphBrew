import csv
import random
import os

# Define kernels
KERNELS = ['bc', 'bfs', 'cc', 'cc_sv', 'pr', 'pr_spmv', 'sssp', 'tc']

# Define the graph suites with their corresponding graph names and file types
graph_suites = {
    "LAW": {
        "graphs": [
            "amazon-2008", "arabic-2005", "cnr-2000", "dblp-2010", "enron", "eu-2005",
            "hollywood-2009", "in-2004", "indochina-2004", "it-2004", "ljournal-2008",
            "sk-2005", "uk-2002", "uk-2005", "webbase-2001"
        ],
        "file_type": "graph.el"
    },
    "SNAP": {
        "graphs": [
            "cit-Patents", "com-Orkut", "soc-LiveJournal1", "soc-Pokec", "web-Google"
        ],
        "file_type": "graph.el"
    },
    "GAP": {
        "graphs": [
            "kron", "road", "twitter", "urand", "web"
        ],
        "file_type": "graph.wel"
    }
}

# Reordering strategies
REORDERING_STRATEGIES = [
    "Original", "Random", "Sort", "HubSort", "HubCluster", "DBG", "HubSortDBG",
    "HubClusterDBG", "RabbitOrder", "Gorder", "Corder", "RCM", "Leiden"
]

# Placeholder function to simulate running a command
def run_command(command):
    print("Running command:", command)
    return random.uniform(0.00010, 0.00050)  # Simulated execution time

# Main execution loop
for kernel in KERNELS:
    results = {}
    for suite, details in graph_suites.items():
        for graph_name in details["graphs"]:
            file_type = details["file_type"]
            graph_path = f'/media/cmv6ru/Data/00_GraphDatasets/{suite}/{graph_name}/{file_type}'
            results[graph_name] = {}
            for strategy in REORDERING_STRATEGIES:
                # Construct the command
                command = f"run-{kernel} -f {graph_path}"
                if strategy != "Original":
                    command += f" -r {REORDERING_STRATEGIES.index(strategy)}"
                # Execute the command and record the time
                time_taken = run_command(command)
                results[graph_name][strategy] = time_taken

    # Write results to a CSV file
    csv_filename = f"{kernel}.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['graph_name'] + REORDERING_STRATEGIES
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for graph, timings in results.items():
            row = {'graph_name': graph}
            row.update(timings)
            writer.writerow(row)

    print(f"Results written to {csv_filename}")
