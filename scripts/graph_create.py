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
from bs4 import BeautifulSoup

prereorder_codes = None
postreorder_codes = None

PARALLEL = os.cpu_count()  # Use all available CPU cores

def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
        save_response_content(response, destination)
    else:
        soup = BeautifulSoup(response.content, 'html.parser')
        download_link = find_confirm_link(soup)
        if download_link:
            response = session.get(download_link, stream=True)
            save_response_content(response, destination)
        else:
            print("Failed to find download link in the HTML file.")
            return

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def find_confirm_link(soup):
    download_link = None
    for a in soup.find_all('a'):
        if 'Download anyway' in a.text:
            download_link = a['href']
            break
    return download_link

def save_response_content(response, destination):
    CHUNK_SIZE = 32768  # 32KB chunks

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

def parse_synthetic_link(synthetic_link):
    pattern = r"(-[gu]\d+)\s*(-k\d+)?"
    matches = re.findall(pattern, synthetic_link)
    parsed_values = []
    for match in matches:
        g_or_u_number = match[0]
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

    extract_dir = os.path.join(download_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)

    if os.path.exists(graph_file_path):
        print(f"Suite graph {graph_file_path} already exists.")
        return

    os.makedirs(download_dir, exist_ok=True)

    DEFAULT_RUN_PARAMS = []
    DEFAULT_RUN_PARAMS.extend([f"-o{code}" for code in prereorder_codes])
    DEFAULT_RUN_PARAMS.extend([f"-o{code}" for code in postreorder_codes])

    file_name = None

    if parsed_synthetic_link:
        file_name = f"{symbol}_synthetic"
        file_path = graph_file_path
        graph_bench = " ".join([f"{g_or_u_number} {k_number}" for g_or_u_number, k_number in parsed_synthetic_link])
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

        cmd = " ".join(cmd)
        print(cmd)
        try:
            output = subprocess.run(
                cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            print(output.stdout.decode())
            print(output.stderr.decode())
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {cmd}\nError: {e.stderr.decode()}")
            return
        return

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

    try:
        if file_name.endswith(".tar.gz"):
            mode = "r:gz"
        elif file_name.endswith(".tar.bz2"):
            mode = "r:bz2"
        else:
            mode = None

        if mode:
            print(f"Extracting {symbol} with mode {mode}...")
            with tarfile.open(file_path, mode) as tar:
                print(f"Opened {file_path}")
                members = tar.getmembers()
                print(f"Members count: {len(members)}")
                largest_size = 0
                largest_file = None
                for member in members:
                    if member.size > largest_size:
                        largest_size = member.size
                        largest_file = member
                if largest_file:
                    print(f"Largest file in the archive: {largest_file.name} ({largest_size} bytes)")
                    tar.extract(largest_file, path=extract_dir)
                    extracted_path = os.path.join(extract_dir, largest_file.name)
                    shutil.move(extracted_path, graph_file_path)
                    for member in members:
                        if member.name != largest_file.name:
                            tar.extract(member, path=extract_dir)
                            if os.path.isdir(os.path.join(extract_dir, member.name)):
                                shutil.rmtree(os.path.join(extract_dir, member.name))
                            else:
                                os.remove(os.path.join(extract_dir, member.name))
                    shutil.rmtree(extract_dir)
                else:
                    print("No files found in the archive")
            os.remove(file_path)

        elif file_name.endswith(".gz"):
            print(f"Extracting .gz file {file_name}...")
            with gzip.open(file_path, "rb") as gz_file:
                content = gz_file.read()
                extracted_file_path = os.path.join(
                    extract_dir, symbol + "." + graph_download_type
                )
                with open(extracted_file_path, "wb") as extracted_file:
                    extracted_file.write(content)
            shutil.move(extracted_file_path, graph_file_path)
            shutil.rmtree(extract_dir)
            os.remove(file_path)

        elif file_name.endswith(".zip"):
            print(f"Extracting .zip file {file_name}...")
            with zipfile.ZipFile(file_path, "r") as zip_file:
                file_list = zip_file.namelist()
                print(f"Files in the zip archive: {file_list}")
                largest_file = max(file_list, key=lambda x: zip_file.getinfo(x).file_size)
                if largest_file:
                    zip_file.extract(largest_file, path=extract_dir)
                    extracted_path = os.path.join(extract_dir, largest_file)
                    shutil.move(extracted_path, graph_file_path)
                    for file in file_list:
                        if file != largest_file:
                            zip_file.extract(file, path=extract_dir)
                            if os.path.isdir(os.path.join(extract_dir, file)):
                                shutil.rmtree(os.path.join(extract_dir, file))
                            else:
                                os.remove(os.path.join(extract_dir, file))
                    shutil.rmtree(extract_dir)
                else:
                    print("No files found in the archive")
            os.remove(file_path)

        else:
            print(f"Unsupported file format for {file_name}.")

    except (tarfile.TarError, gzip.BadGzipFile, zipfile.BadZipFile) as e:
        print(f"Error extracting {file_name}: {e}")

def download_and_extract_graphs(config):
    global prereorder_codes
    global postreorder_codes

    graph_suites = config["graph_suites"]
    suite_dir = graph_suites.pop("suite_dir")

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

    for thread in threads:
        thread.get()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python graph_download.py config_file.json")
        sys.exit(1)

    config_file = sys.argv[1]

    with open(config_file, "r") as f:
        config = json.load(f)
        download_and_extract_graphs(config)
