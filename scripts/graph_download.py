#!/usr/bin/env python3

import os
import requests
import tarfile
import shutil
import json
import sys
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import importlib.util
import gzip
import zipfile


def import_check_install(package_name):
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        print(f"{package_name} is not installed. Installing...")
        subprocess.run(["pip", "install", package_name])


def download_and_extract_graph(graph):
    symbol = graph["symbol"]
    graph_download_type = graph["download_type"]
    download_link = graph["download_link"]
    suite_dir_path = graph["suite_dir_path"]
    graph_fullname = graph["graph_fullname"]

    download_dir = os.path.join(suite_dir_path, symbol)
    graph_file_path = os.path.join(download_dir, f"{graph_fullname}")
    # Create a subdirectory for extraction
    extract_dir = os.path.join(download_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    # Check if suite directory exists
    if os.path.exists(graph_file_path):
        print(f"Suite graph {graph_file_path} already exists.")
        return
    os.makedirs(download_dir, exist_ok=True)
    # Download the graph file
    file_name = os.path.basename(download_link)
    file_path = os.path.join(download_dir, file_name)
    progress_desc = f"Downloading {symbol}"
    with requests.get(download_link, stream=True) as response:
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
            extracted_file_path = os.path.join(extract_dir, symbol + "." + graph_download_type)
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
    graph_suites = config["graph_suites"]
    suite_dir = graph_suites.pop("suite_dir")  # Extracting suite_dir from graph_suites

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
