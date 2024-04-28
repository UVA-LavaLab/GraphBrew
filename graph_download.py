import os
import requests
import tarfile
import shutil
import json
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import importlib.util

def import_check_install(package_name):
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        print(f"{package_name} is not installed. Installing...")
        subprocess.run(["pip", "install", package_name])

def download_and_extract_graph(graph):
    symbol = graph['symbol']
    download_link = graph['download_link']
    suite_dir_path = graph['suite_dir_path']
    file_type = graph['file_type']
    
    download_dir = os.path.join(suite_dir_path, symbol)
    os.makedirs(download_dir, exist_ok=True)
    
    # Download the graph file
    file_name = os.path.basename(download_link)
    file_path = os.path.join(download_dir, file_name)
    progress_desc = f"Downloading {symbol}"
    with requests.get(download_link, stream=True) as response:
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, leave=False, desc=progress_desc)
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
    progress_bar.close()
    
    # Extract the graph file if it's a tar.gz archive
    if file_name.endswith('.tar.gz'):
        # print(f"Extracting {symbol}...")
        with tarfile.open(file_path, 'r:gz') as tar:
            largest_size = 0
            largest_file = None
            for member in tar.getmembers():
                if member.size > largest_size:
                    largest_size = member.size
                    largest_file = member
            if largest_file:
                # Create a subdirectory for extraction
                extract_dir = os.path.join(download_dir, 'extracted')
                os.makedirs(extract_dir, exist_ok=True)
                tar.extract(largest_file, path=extract_dir)
                # Rename the largest file to 'graph.extension'
                extracted_path = os.path.join(extract_dir, largest_file.name)
                graph_file_path = os.path.join(download_dir, f'{file_type}')
                shutil.move(extracted_path, graph_file_path)
                # print(f"Extracted and renamed {largest_file.name} to {file_type}")
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

def download_and_extract_graphs(config):
    graph_suites = config['graph_suites']
    suite_dir = graph_suites.pop('suite_dir')  # Extracting suite_dir from graph_suites
    
    threads = []
    for suite_name, details in graph_suites.items():
        if suite_name == 'suite_dir':
            continue
        
        suite_dir_path = os.path.join(suite_dir, suite_name)
        os.makedirs(suite_dir_path, exist_ok=True)
        
        graphs = details['graphs']
        file_type = details['file_type']
        
        for graph in graphs:
            graph['suite_dir_path'] = suite_dir_path
            graph['file_type'] = file_type
            thread = ThreadPool(processes=1).apply_async(download_and_extract_graph, (graph,))
            threads.append(thread)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.get()

dependencies = ['requests', 'tqdm', 'shutil', 'tarfile', 'requests', 'multiprocessing', 'importlib']
for dependency in dependencies:
    import_check_install(dependency)

# Loading configuration settings from a JSON file
with open('config/lite.json', 'r') as f:
    config = json.load(f)
    download_and_extract_graphs(config)
