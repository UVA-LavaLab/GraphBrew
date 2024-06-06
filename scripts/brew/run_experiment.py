import os
import re
import subprocess
import csv

# Define the base directory containing the graph datasets
BASE_DIR   = "/media/Data/00_GraphDatasets/GBREW"
RESULT_DIR = "bench/results"
LOG_DIR_RUN   = os.path.join(RESULT_DIR, "logs_run")
LOG_DIR_ORDER = os.path.join(RESULT_DIR, "logs_order")
os.makedirs(LOG_DIR_RUN, exist_ok=True)
os.makedirs(LOG_DIR_ORDER, exist_ok=True)

# Define the list of graphs and their extensions
graph_extensions = {
    "RD": "mtx",
    "SLJ1": "mtx",
    "CPAT": "mtx",
    "CORKT": "mtx",
    "SPKC": "mtx",
    "GPLUS": "mtx",
    "WIKLE": "mtx",
    "WEB01": "mtx",
    "TWTR": "mtx"
}

# Define the list of kernels
kernels = [
    {"name": "bc", "trials": 20, "iterations": 10},
    {"name": "bfs", "trials": 20, "iterations": 10},
    {"name": "cc", "trials": 20, "iterations": 10},
    {"name": "cc_sv", "trials": 20, "iterations": 10},
    {"name": "pr", "trials": 20, "iterations": 10},
    {"name": "pr_spmv", "trials": 20, "iterations": 10},
    {"name": "sssp", "trials": 20, "iterations": 10}
]

# Regular expressions for parsing timing data from benchmark outputs
time_patterns = {
    "reorder_time": {
        "HubClusterDBG": re.compile(r"\bHubClusterDBG\b Map Time:\s*([\d\.]+)"),
        "HubCluster": re.compile(r"\bHubCluster\b Map Time:\s*([\d\.]+)"),
        "HubSortDBG": re.compile(r"\bHubSortDBG\b Map Time:\s*([\d\.]+)"),
        "HubSort": re.compile(r"\bHubSort\b Map Time:\s*([\d\.]+)"),
        "LeidenFull": re.compile(r"\bLeidenFullOrder\b Map Time:\s*([\d\.]+)"),
        "Leiden": re.compile(r"\bLeidenOrder\b Map Time:\s*([\d\.]+)"),
        "Original": re.compile(r"\bOriginal\b Map Time:\s*([\d\.]+)"),
        "RabbitOrder": re.compile(r"\bRabbitOrder\b Map Time:\s*([\d\.]+)"),
        "Random": re.compile(r"\bRandom\b Map Time:\s*([\d\.]+)"),
        "Corder": re.compile(r"\bCOrder\b Map Time:\s*([\d\.]+)"),
        "Gorder": re.compile(r"\bGOrder\b Map Time:\s*([\d\.]+)"),
        "DBG": re.compile(r"\bDBG\b Map Time:\s*([\d\.]+)"),
        "RCM": re.compile(r"\bRCMOrder\b Map Time:\s*([\d\.]+)"),
        "Sort": re.compile(r"\bSort\b Map Time:\s*([\d\.]+)")
    },
    "trial_time": {
        "Average": re.compile(r"\bAverage\b Time:\s*([\d\.]+)")
    }
}

reorder_option_mapping = {
    "Random": "-o0", # this is your baseline
    # "Sort": "-o2",
    # "HubSort": "-o3",
    # "HubCluster": "-o4",
    "DBG": "-o5",
    # "HubSortDBG": "-o6",
    # "HubClusterDBG": "-o7",
    "RabbitOrder": "-o8",
    "Gorder": "-o9",
    "Corder": "-o10",
    "RCM": "-o11",
    "Leiden": "-o12",
    "LeidenFull": "-o8 -o12"
}

def parse_reorder_output(output):
    timings = {}
    for key, pattern in time_patterns["reorder_time"].items():
        match = pattern.search(output)
        if match:
            timings[key] = float(match.group(1))
    return timings

def parse_kernel_output(output):
    match = time_patterns["trial_time"]["Average"].search(output)
    if match:
        return float(match.group(1))
    return None

def run_reorders():
    print("Starting reorder process...")
    
    results = {}
    
    # Iterate over each graph
    for graph, ext in graph_extensions.items():
        print(f"Processing graph: {graph}")
        
        # Construct the graph file path
        graph_file = os.path.join(BASE_DIR, graph, f"graph.{ext}")
        random_graph_file = os.path.join(BASE_DIR, graph, f"graph_1.sg")

        reorder_name   = "Random" 
        reorder_option = "-o1"        
        # Construct a random graph if it does not exist
        if not os.path.isfile(random_graph_file):
            print(f"Running converter with reorder {reorder_name} option: {reorder_option}")
            print(f"Output file: {random_graph_file}")
            make_command = f"make run-converter GRAPH_BENCH='-f {graph_file} -b {random_graph_file}' RUN_PARAMS='{reorder_option}' FLUSH_CACHE=0 PARALLEL=16"
            log_file = os.path.join(LOG_DIR_ORDER, f"{graph}_initial.log")
            with open(log_file, 'w') as log:
                print(f"Executing command: {make_command}")
                subprocess.run(make_command, shell=True, check=True, stdout=log, stderr=log)
        
        # Check if the random graph file exists
        if os.path.isfile(random_graph_file):
            print(f"Graph file found: {random_graph_file}")
            
            results[graph] = {}
            
            # Iterate over each reorder option
            for reorder_name, reorder_option in list(reorder_option_mapping.items()):
                if ' ' in reorder_option:
                    # Handle multiple options
                    option_numbers = '_'.join([opt.split('o')[1] for opt in reorder_option.split()])
                    output_file = os.path.join(BASE_DIR, graph, f"graph_{option_numbers}.mtx")
                else:
                    # Handle single option
                    option_number = reorder_option.split('o')[1]
                    output_file = os.path.join(BASE_DIR, graph, f"graph_{option_number}.mtx")
                
                # Skip if the output file already exists
                if os.path.isfile(output_file):
                    print(f"Output file already exists, skipping: {output_file}")
                    continue
                
                # Print the current stage
                print(f"Running converter with reorder {reorder_name} option: {reorder_option}")
                print(f"Output file: {output_file}")
                
                # Construct and run the make command
                make_command = f"make run-converter GRAPH_BENCH='-f {random_graph_file} -p {output_file}' RUN_PARAMS='{reorder_option}' FLUSH_CACHE=0 PARALLEL=16"
                log_file = os.path.join(LOG_DIR_ORDER, f"{graph}_{reorder_name}.log")
                with open(log_file, 'w') as log:
                    print(f"Executing command: {make_command}")
                    result = subprocess.run(make_command, shell=True, check=True, stdout=log, stderr=log)
                
                # Parse the output from the log file
                with open(log_file, 'r') as log:
                    timings = parse_reorder_output(log.read())
                
                # Record the results
                for key, time in timings.items():
                    if reorder_name in reorder_option_mapping:
                        results[graph][reorder_name] = time
                
                print(f"Completed conversion for reorder option: {reorder_option}\n")
        else:
            print(f"Graph file not found: {random_graph_file}")
    
    # Check if results are empty
    if not results:
        print("No new conversions were performed. All graph files already exist.")
        return
    
    # Write results to CSV
    csv_file = os.path.join(RESULT_DIR, "reorder_results.csv")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ["Graph"] + list(reorder_option_mapping.keys())
        writer.writerow(header)
        
        for graph, timings in results.items():
            row = [graph] + [timings.get(reorder_name, '') for reorder_name in reorder_option_mapping.keys()]
            writer.writerow(row)
    
    print("Reorder process completed.")

def run_kernels():
    print("Starting kernel execution process...")
    
    kernel_results = {kernel["name"]: {} for kernel in kernels}
    
    # Iterate over each graph
    for graph in graph_extensions.keys():
        print(f"Processing graph: {graph}")
        
        # Iterate over each reorder option
        for reorder_name, reorder_option in reorder_option_mapping.items():
            if ' ' in reorder_option:
                # Handle multiple options
                option_numbers = '_'.join([opt.split('o')[1] for opt in reorder_option.split()])
                output_file = os.path.join(BASE_DIR, graph, f"graph_{option_numbers}.sg")
            else:
                # Handle single option
                option_number = reorder_option.split('o')[1]
                output_file = os.path.join(BASE_DIR, graph, f"graph_{option_number}.sg")
            
            # Check if the converted graph file exists
            if os.path.isfile(output_file):
                print(f"Converted graph file found: {output_file}")
                
                # Run kernels on the converted graph file
                for kernel in kernels:
                    kernel_command = f"make run-{kernel['name']} GRAPH_BENCH='-f {output_file}' RUN_PARAMS='-n {kernel['trials']}' FLUSH_CACHE=1 PARALLEL=16"
                    log_file = os.path.join(LOG_DIR_RUN, f"{graph}_{reorder_name}_{kernel['name']}.log")
                    print(f"Running kernel: {kernel['name']} with {kernel['trials']} trials and {kernel['iterations']} iterations")
                    print(f"Executing command: {kernel_command}")
                    
                    # Run the command and log the output
                    with open(log_file, 'w') as log:
                        result = subprocess.run(kernel_command, shell=True, check=True, stdout=log, stderr=log)
                    
                    # Parse the output from the log file
                    with open(log_file, 'r') as log:
                        average_time = parse_kernel_output(log.read())
                    
                    if average_time is not None:
                        if graph not in kernel_results[kernel['name']]:
                            kernel_results[kernel['name']][graph] = {}
                        kernel_results[kernel['name']][graph][reorder_name] = average_time
                    
                    print(f"Completed kernel: {kernel['name']}\n")
            else:
                print(f"Converted graph file not found: {output_file}")
    
    # Check if kernel results are empty
    if all(not results for results in kernel_results.values()):
        print("No kernels were executed. All converted graph files already exist or were not found.")
        return
    
    # Write results to CSV for each kernel
    for kernel_name, results in kernel_results.items():
        if results:
            csv_file = os.path.join(RESULT_DIR, f"{kernel_name}_trial_time_results.csv")
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                header = ["Graph"] + list(reorder_option_mapping.keys())
                writer.writerow(header)
                
                for graph, timings in results.items():
                    row = [graph] + [timings.get(reorder_name, '') for reorder_name in reorder_option_mapping.keys()]
                    writer.writerow(row)
    
    print("Kernel execution process completed.")

def run_convert():
    print("Starting reorder process...")
    
    results = {}
    
    # Iterate over each graph
    for graph, ext in graph_extensions.items():
        print(f"Processing graph: {graph}")
        
        results[graph] = {}
            
        # Iterate over each reorder option
        for reorder_name, reorder_option in list(reorder_option_mapping.items()):
            if ' ' in reorder_option:
                # Handle multiple options
                option_numbers = '_'.join([opt.split('o')[1] for opt in reorder_option.split()])
                output_file = os.path.join(BASE_DIR, graph, f"graph_{option_numbers}.mtx")
                output_file_conv = os.path.join(BASE_DIR, graph, f"graph_{option_numbers}.sg")
            else:
                # Handle single option
                option_number = reorder_option.split('o')[1]
                output_file = os.path.join(BASE_DIR, graph, f"graph_{option_number}.mtx")
                output_file_conv = os.path.join(BASE_DIR, graph, f"graph_{option_number}.sg")
            
            # Skip if the output file already exists
            if os.path.isfile(output_file_conv):
                print(f"Output file already exists, skipping: {output_file_conv}")
                continue
            
            # Print the current stage
            print(f"Running converter with reorder {reorder_name} option: {reorder_option}")
            print(f"Output file: {output_file}")
            
            # Construct and run the make command
            make_command = f"make run-converter GRAPH_BENCH='-f {output_file} -b {output_file_conv}' RUN_PARAMS='-o0' FLUSH_CACHE=0 PARALLEL=32"
            log_file = os.path.join(LOG_DIR_ORDER, f"{graph}_{reorder_name}.log")
            with open(log_file, 'w') as log:
                print(f"Executing command: {make_command}")
                result = subprocess.run(make_command, shell=True, check=True, stdout=log, stderr=log)
            
            # Parse the output from the log file
            with open(log_file, 'r') as log:
                timings = parse_reorder_output(log.read())
            
            # Record the results
            for key, time in timings.items():
                if reorder_name in reorder_option_mapping:
                    results[graph][reorder_name] = time
            
            print(f"Completed conversion for reorder option: {reorder_option}\n")
    
    print("Convert process completed.")

if __name__ == "__main__":
    run_convert()
    # run_reorders()
    # run_kernels()
