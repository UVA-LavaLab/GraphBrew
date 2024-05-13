#!/usr/bin/env python3

from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import gzip
import importlib.util
import json
import os
import re
import requests
import shutil
import subprocess
import sys
import tarfile
import zipfile

prereorder_codes = None
postreorder_codes = None

PARALLEL = os.cpu_count()  # Use all available CPU cores

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

def import_check_install(package_name):
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        print(f"{package_name} is not installed. Installing...")
        subprocess.run(["pip", "install", package_name])


def parse_synthetic_link(synthetic_link):
    # Regular expression pattern to match either -g or -u followed by a number, and optionally -k followed by another number
    pattern = r"(-[gu]\d+)\s*(-k\d+)?"
    # Find all matches in the synthetic_link string
    matches = re.findall(pattern, synthetic_link)
    parsed_values = []
    for match in matches:
        # Extract the -g or -u number
        g_or_u_number = match[0]
        # Extract the -k number if present, otherwise set it to None
        k_number = match[1] if match[1] else None
        parsed_values.append((g_or_u_number, k_number))
    return parsed_values


def download_and_extract_graph(graph):
    global prereorder_codes
    global postreorder_codes

    symbol = graph["symbol"]
    graph_download_type = graph.get("download_type", "")
    graph_download_link = graph.get("download_link", "")
    graph_synthetic_link = graph.get("synthetic_link", "")
    parsed_synthetic_link = parse_synthetic_link(graph_synthetic_link)

    suite_dir_path = graph["suite_dir_path"]
    graph_fullname = graph["graph_fullname"]

    download_dir = os.path.join(suite_dir_path, symbol)
    graph_file_path = os.path.join(download_dir, graph_fullname)

    # Create a subdirectory for extraction
    extract_dir = os.path.join(download_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)

    # Check if suite directory exists
    if os.path.exists(graph_file_path):
        print(f"Suite graph {graph_file_path} already exists.")
        return

    os.makedirs(download_dir, exist_ok=True)

    DEFAULT_RUN_PARAMS = []
    # Add prereorder codes if any
    DEFAULT_RUN_PARAMS.extend([f"-o{code}" for code in prereorder_codes])
    # Main reorder code
    # Add postreorder codes if any
    DEFAULT_RUN_PARAMS.extend([f"-o{code}" for code in postreorder_codes])

    # Download the graph file
    if parsed_synthetic_link:
        # Construct file path using synthetic link
        file_name = f"{symbol}_synthetic"
        file_path = graph_file_path
        # Assuming here that you have a function to generate synthetic graph data
        # Replace this with your actual function call to generate the synthetic graph data
        # Assuming you want to call the make command after generating the synthetic graph
        graph_bench = " ".join([f"{g_or_u_number} {k_number}" for g_or_u_number, k_number in parsed_synthetic_link])
        # print(parsed_synthetic_link)
        order_params = ' '.join(DEFAULT_RUN_PARAMS)
        run_params = f"-b {file_path}"
        run_params = order_params + ' ' + run_params

        cmd = [
            "make run-converter",
            f"RUN_PARAMS='{run_params}'",
            f"GRAPH_BENCH='{graph_bench}'",
            "FLUSH_CACHE=0",
            f"PARALLEL={PARALLEL}",
        ]

        # Convert list to a space-separated string for subprocess execution
        cmd = " ".join(cmd)
        print(cmd)
        try:
            output = subprocess.run(
                cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {cmd}\nError: {e.stderr.decode()}")
            return

        return

    # Check if it is a Google Drive link
    elif 'drive.google.com' in graph_download_link:
        # Extract the file ID from the link
        match = re.search(r'd/([^/]+)', graph_download_link)
        if match:
            file_id = match.group(1)
            file_path = os.path.join(download_dir, graph_fullname)
            download_file_from_google_drive(file_id, file_path)
        else:
            print("Failed to extract File ID from the Google Drive link.")

    else:
        file_name = os.path.basename(graph_download_link)
        file_path = os.path.join(download_dir, file_name)

        progress_desc = f"Downloading {symbol}"
        with requests.get(graph_download_link, stream=True) as response:
            total_size = int(response.headers.get("content-length", 0))
            progress_bar = tqdm(
                total=total_size, unit="B", unit_scale=True, leave=False, desc=progress_desc
            )
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))
        progress_bar.close()

    # Extract the graph file if it's a tar.gz archive
    if file_name.endswith(".tar.gz"):
        print(f"Extracting {symbol}...")
        with tarfile.open(file_path, "r:gz") as tar:
            largest_size = 0
            largest_file = None
            for member in tar.getmembers():
                if member.size > largest_size:
                    largest_size = member.size
                    largest_file = member
            if largest_file:
                tar.extract(largest_file, path=extract_dir)
                # Rename the largest file to 'graph.extension'
                extracted_path = os.path.join(extract_dir, largest_file.name)
                shutil.move(extracted_path, graph_file_path)
                # print(f"Extracted and renamed {largest_file.name} to {graph_fullname}")
                # Remove the rest of the files
                for member in tar.getmembers():
                    if member.name != largest_file.name:
                        tar.extract(member, path=extract_dir)
                        if os.path.isdir(os.path.join(extract_dir, member.name)):
                            shutil.rmtree(os.path.join(extract_dir, member.name))
                        else:
                            os.remove(os.path.join(extract_dir, member.name))
                # print("Deleted other files")
                # Remove the extracted directory
                shutil.rmtree(extract_dir)
            else:
                print("No files found in the archive")

        # Remove the downloaded .tar.gz file
        os.remove(file_path)
        # print(f"Deleted {file_name}")

    elif file_name.endswith(".gz"):
        # Handle .gz files using gzip
        # Extract the .gz file
        with gzip.open(file_path, "rb") as gz_file:
            # Read the contents of the .gz file
            content = gz_file.read()

            # Write the contents to a new file
            extracted_file_path = os.path.join(
                extract_dir, symbol + "." + graph_download_type
            )
            with open(extracted_file_path, "wb") as extracted_file:
                extracted_file.write(content)

        # Move the extracted file to the desired location
        shutil.move(extracted_file_path, graph_file_path)

        # Remove the extracted directory
        shutil.rmtree(extract_dir)
        os.remove(file_path)

    elif file_name.endswith(".zip"):
        # Handle .zip files using zipfile
        # Extract the .zip file
        with zipfile.ZipFile(file_path, "r") as zip_file:
            # Get the list of files in the .zip file
            file_list = zip_file.namelist()

            # Choose the largest file in the .zip file
            largest_file = max(file_list, key=lambda x: zip_file.getinfo(x).file_size)

            if largest_file:
                # Extract the largest file
                zip_file.extract(largest_file, path=extract_dir)

                # Rename the extracted file to 'graph.extension'
                extracted_path = os.path.join(extract_dir, largest_file)
                shutil.move(extracted_path, graph_file_path)

                # Remove the rest of the files
                for file in file_list:
                    if file != largest_file:
                        zip_file.extract(file, path=extract_dir)
                        if os.path.isdir(os.path.join(extract_dir, file)):
                            shutil.rmtree(os.path.join(extract_dir, file))
                        else:
                            os.remove(os.path.join(extract_dir, file))

                # Remove the extracted directory
                shutil.rmtree(extract_dir)
            else:
                print("No files found in the archive")
            os.remove(file_path)
    else:
        print("Unsupported file format.")


def download_and_extract_graphs(config):
    global prereorder_codes
    global postreorder_codes

    graph_suites = config["graph_suites"]
    suite_dir = graph_suites.pop("suite_dir")  # Extracting suite_dir from graph_suites

    # Extract prereorder and postreorder codes
    prereorder_codes = [
        config["prereorder"].get(key, []) for key in config.get("prereorder", {})
    ]
    postreorder_codes = [
        config["postreorder"].get(key, []) for key in config.get("postreorder", {})
    ]

    threads = []
    for suite_name, details in graph_suites.items():
        if suite_name == "suite_dir":
            continue

        suite_dir_path = os.path.join(suite_dir, suite_name)
        os.makedirs(suite_dir_path, exist_ok=True)

        graphs = details["graphs"]
        graph_basename = details.get("graph_basename", "graph")

        for graph in graphs:
            graph_download_type = graph.get("download_type", "el")
            graph["suite_dir_path"] = suite_dir_path
            graph["graph_fullname"] = f"{graph_basename}.{graph_download_type}"
            thread = ThreadPool(processes=1).apply_async(
                download_and_extract_graph, (graph,)
            )
            threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.get()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python graph_download.py config_file.json")
        sys.exit(1)

    config_file = sys.argv[1]

    # Loading configuration settings from the specified JSON file
    with open(config_file, "r") as f:
        config = json.load(f)
        download_and_extract_graphs(config)
