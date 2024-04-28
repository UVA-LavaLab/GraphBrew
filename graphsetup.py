import os
import requests
import json

def download_graphs(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    reorderings = data['reorderings']
    graph_suites = data['graph_suites']
    
    for suite_name, suite_info in graph_suites.items():
        suite_dir = suite_info['suite_dir']
        graphs = suite_info['graphs']
        file_type = suite_info['file_type']
        
        suite_dir = os.path.join(suite_dir, suite_name)
        os.makedirs(suite_dir, exist_ok=True)
        
        for graph_name in graphs:
            download_url = f"{suite_info['suite_dir']}/{suite_name}/{graph_name}.{file_type}"
            download_path = os.path.join(suite_dir, f"{graph_name}.{file_type}")
            
            # Download the graph file
            response = requests.get(download_url)
            if response.status_code == 200:
                with open(download_path, 'wb') as file:
                    file.write(response.content)
                print(f"Downloaded {graph_name}.{file_type}")
                
                # Augment the graphs list with graph name and download link
                graph_info = {
                    'graph_name': graph_name,
                    'download_link': download_url,
                    'suite_directory': suite_dir
                }
                print(graph_info)
            else:
                print(f"Failed to download {graph_name}.{file_type}")

if __name__ == "__main__":
    json_file = "your_json_file.json"  # Replace with the path to your JSON file
    download_graphs(json_file)