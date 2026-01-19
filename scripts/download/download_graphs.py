#!/usr/bin/env python3
"""
Graph Download Script for GraphBrew Benchmarks

Downloads a curated set of real-world graphs for comprehensive benchmarking.
Graphs are selected to cover different domains and characteristics:
- Social networks (high clustering, power-law)
- Web graphs (directed, power-law)
- Road networks (low clustering, planar-like)
- Citation networks (directed, hierarchical)
- Collaboration networks (bipartite-like)

Usage:
    python3 scripts/download/download_graphs.py [--output-dir DIR] [--size SMALL|MEDIUM|LARGE|ALL]
    
Examples:
    python3 scripts/download/download_graphs.py                     # Download medium graphs
    python3 scripts/download/download_graphs.py --size LARGE        # Download large graphs
    python3 scripts/download/download_graphs.py --size ALL          # Download all graphs
"""

import os
import sys
import argparse
import subprocess
import tarfile
import gzip
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# ============================================================================
# Graph Catalog
# ============================================================================

@dataclass
class GraphInfo:
    """Information about a downloadable graph."""
    name: str           # Short name for the graph
    url: str            # Download URL
    size_mb: int        # Approximate download size in MB
    nodes: int          # Approximate number of nodes
    edges: int          # Approximate number of edges
    format: str         # File format (mtx, el, wel, etc.)
    symmetric: bool     # Whether graph is symmetric/undirected
    description: str    # Brief description
    category: str       # Graph category (social, web, road, citation, etc.)


# =============================================================================
# VERIFIED GRAPH CATALOG
# All graphs verified to have proper square adjacency matrix format
# =============================================================================

# Small graphs (< 100MB, good for testing)
SMALL_GRAPHS = [
    GraphInfo(
        name="email-Enron",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/email-Enron.tar.gz",
        size_mb=5,
        nodes=36692,
        edges=183831,
        format="mtx",
        symmetric=True,
        description="Enron email communication network",
        category="communication"
    ),
    GraphInfo(
        name="ca-AstroPh",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/ca-AstroPh.tar.gz",
        size_mb=3,
        nodes=18772,
        edges=198110,
        format="mtx",
        symmetric=True,
        description="Arxiv Astro Physics collaboration network",
        category="collaboration"
    ),
    GraphInfo(
        name="ca-CondMat",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/ca-CondMat.tar.gz",
        size_mb=2,
        nodes=23133,
        edges=93497,
        format="mtx",
        symmetric=True,
        description="Condensed Matter collaboration network",
        category="collaboration"
    ),
    GraphInfo(
        name="p2p-Gnutella31",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/p2p-Gnutella31.tar.gz",
        size_mb=2,
        nodes=62586,
        edges=147892,
        format="mtx",
        symmetric=False,
        description="Gnutella peer-to-peer network",
        category="p2p"
    ),
]

# Medium graphs (100MB - 2GB, good balance of size and diversity)
# VERIFIED: All have proper square adjacency matrix format
MEDIUM_GRAPHS = [
    GraphInfo(
        name="wiki-Talk",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/wiki-Talk.tar.gz",
        size_mb=80,
        nodes=2394385,
        edges=5021410,
        format="mtx",
        symmetric=False,
        description="Wikipedia talk network",
        category="communication"
    ),
    GraphInfo(
        name="cit-Patents",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/cit-Patents.tar.gz",
        size_mb=262,
        nodes=3774768,
        edges=16518948,
        format="mtx",
        symmetric=False,
        description="US Patent citation network",
        category="citation"
    ),
    GraphInfo(
        name="roadNet-PA",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/roadNet-PA.tar.gz",
        size_mb=40,
        nodes=1090920,
        edges=1541898,
        format="mtx",
        symmetric=True,
        description="Pennsylvania road network",
        category="road"
    ),
    GraphInfo(
        name="roadNet-CA",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/roadNet-CA.tar.gz",
        size_mb=60,
        nodes=1971281,
        edges=2766607,
        format="mtx",
        symmetric=True,
        description="California road network",
        category="road"
    ),
    GraphInfo(
        name="roadNet-TX",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/roadNet-TX.tar.gz",
        size_mb=45,
        nodes=1393383,
        edges=1921660,
        format="mtx",
        symmetric=True,
        description="Texas road network",
        category="road"
    ),
    GraphInfo(
        name="soc-Epinions1",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/soc-Epinions1.tar.gz",
        size_mb=12,
        nodes=75888,
        edges=508837,
        format="mtx",
        symmetric=False,
        description="Epinions social network",
        category="social"
    ),
    GraphInfo(
        name="soc-Slashdot0811",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/soc-Slashdot0811.tar.gz",
        size_mb=10,
        nodes=77360,
        edges=905468,
        format="mtx",
        symmetric=False,
        description="Slashdot social network",
        category="social"
    ),
    GraphInfo(
        name="amazon0302",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/amazon0302.tar.gz",
        size_mb=15,
        nodes=262111,
        edges=1234877,
        format="mtx",
        symmetric=False,
        description="Amazon product co-purchasing Mar 2003",
        category="commerce"
    ),
    GraphInfo(
        name="amazon0601",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/amazon0601.tar.gz",
        size_mb=25,
        nodes=403394,
        edges=3387388,
        format="mtx",
        symmetric=False,
        description="Amazon product co-purchasing Jun 2001",
        category="commerce"
    ),
    GraphInfo(
        name="soc-sign-epinions",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/soc-sign-epinions.tar.gz",
        size_mb=10,
        nodes=131828,
        edges=841372,
        format="mtx",
        symmetric=False,
        description="Epinions signed trust network",
        category="social"
    ),
    GraphInfo(
        name="web-NotreDame",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/web-NotreDame.tar.gz",
        size_mb=15,
        nodes=325729,
        edges=1497134,
        format="mtx",
        symmetric=False,
        description="Notre Dame web graph",
        category="web"
    ),
    GraphInfo(
        name="web-Stanford",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/web-Stanford.tar.gz",
        size_mb=20,
        nodes=281903,
        edges=2312497,
        format="mtx",
        symmetric=False,
        description="Stanford web graph",
        category="web"
    ),
    GraphInfo(
        name="web-BerkStan",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/web-BerkStan.tar.gz",
        size_mb=50,
        nodes=685230,
        edges=7600595,
        format="mtx",
        symmetric=False,
        description="Berkeley-Stanford web graph",
        category="web"
    ),
]

# "Large" graphs (1GB - 8GB, substantial but manageable)
# These were previously in the "Large" category - good for serious benchmarking
# VERIFIED: All have proper square adjacency matrix format
MID_LARGE_GRAPHS = [
    GraphInfo(
        name="soc-LiveJournal1",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/soc-LiveJournal1.tar.gz",
        size_mb=1024,
        nodes=4847571,
        edges=68993773,
        format="mtx",
        symmetric=False,
        description="LiveJournal social network",
        category="social"
    ),
    GraphInfo(
        name="hollywood-2009",
        url="https://suitesparse-collection-website.herokuapp.com/MM/LAW/hollywood-2009.tar.gz",
        size_mb=600,
        nodes=1139905,
        edges=57515616,
        format="mtx",
        symmetric=True,
        description="Hollywood actor collaboration",
        category="collaboration"
    ),
    GraphInfo(
        name="indochina-2004",
        url="https://suitesparse-collection-website.herokuapp.com/MM/LAW/indochina-2004.tar.gz",
        size_mb=2500,
        nodes=7414866,
        edges=194109311,
        format="mtx",
        symmetric=False,
        description="Indochina web crawl 2004",
        category="web"
    ),
    GraphInfo(
        name="uk-2002",
        url="https://suitesparse-collection-website.herokuapp.com/MM/LAW/uk-2002.tar.gz",
        size_mb=4000,
        nodes=18520486,
        edges=298113762,
        format="mtx",
        symmetric=False,
        description="UK domain web crawl 2002",
        category="web"
    ),
    GraphInfo(
        name="GAP-road",
        url="https://suitesparse-collection-website.herokuapp.com/MM/GAP/GAP-road.tar.gz",
        size_mb=628,
        nodes=23947347,
        edges=57708624,
        format="mtx",
        symmetric=True,
        description="Full USA road network (GAP benchmark)",
        category="road"
    ),
    GraphInfo(
        name="arabic-2005",
        url="https://suitesparse-collection-website.herokuapp.com/MM/LAW/arabic-2005.tar.gz",
        size_mb=3200,
        nodes=22744080,
        edges=639999458,
        format="mtx",
        symmetric=False,
        description="Arabic web crawl 2005",
        category="web"
    ),
    GraphInfo(
        name="it-2004",
        url="https://suitesparse-collection-website.herokuapp.com/MM/LAW/it-2004.tar.gz",
        size_mb=3500,
        nodes=41291594,
        edges=1150725436,
        format="mtx",
        symmetric=False,
        description="Italian web crawl 2004",
        category="web"
    ),
]

# Large graphs (8GB - 16GB, for comprehensive benchmarking)
# VERIFIED: All have proper square adjacency matrix format
LARGE_GRAPHS = [
    GraphInfo(
        name="GAP-urand",
        url="https://suitesparse-collection-website.herokuapp.com/MM/GAP/GAP-urand.tar.gz",
        size_mb=16000,
        nodes=134217726,
        edges=4294966654,
        format="mtx",
        symmetric=True,
        description="Uniform random graph (GAP benchmark)",
        category="synthetic"
    ),
    GraphInfo(
        name="GAP-kron",
        url="https://suitesparse-collection-website.herokuapp.com/MM/GAP/GAP-kron.tar.gz",
        size_mb=14000,
        nodes=134217726,
        edges=2111634222,
        format="mtx",
        symmetric=True,
        description="Kronecker synthetic graph (GAP benchmark)",
        category="synthetic"
    ),
    GraphInfo(
        name="webbase-2001",
        url="https://suitesparse-collection-website.herokuapp.com/MM/LAW/webbase-2001.tar.gz",
        size_mb=8500,
        nodes=118142155,
        edges=1019903190,
        format="mtx",
        symmetric=False,
        description="WebBase 2001 web crawl",
        category="web"
    ),
]

# Extra Large graphs (>16GB, for comprehensive benchmarking on powerful machines)
# VERIFIED: All have proper square adjacency matrix format  
EXTRA_LARGE_GRAPHS = [
    GraphInfo(
        name="GAP-twitter",
        url="https://suitesparse-collection-website.herokuapp.com/MM/GAP/GAP-twitter.tar.gz",
        size_mb=31400,
        nodes=61578415,
        edges=1468365182,
        format="mtx",
        symmetric=False,
        description="Twitter follower network (GAP benchmark)",
        category="social"
    ),
    GraphInfo(
        name="GAP-web",
        url="https://suitesparse-collection-website.herokuapp.com/MM/GAP/GAP-web.tar.gz",
        size_mb=17000,
        nodes=50636151,
        edges=1930292948,
        format="mtx",
        symmetric=False,
        description="ClueWeb09 web graph (GAP benchmark)",
        category="web"
    ),
    GraphInfo(
        name="uk-2005",
        url="https://suitesparse-collection-website.herokuapp.com/MM/LAW/uk-2005.tar.gz",
        size_mb=20000,
        nodes=39459925,
        edges=936364282,
        format="mtx",
        symmetric=False,
        description="UK web crawl 2005",
        category="web"
    ),
    GraphInfo(
        name="sk-2005",
        url="https://suitesparse-collection-website.herokuapp.com/MM/LAW/sk-2005.tar.gz",
        size_mb=18000,
        nodes=50636154,
        edges=1949412601,
        format="mtx",
        symmetric=False,
        description="SK domain web crawl 2005",
        category="web"
    ),
]

# ============================================================================
# Download Functions
# ============================================================================

def download_file(url: str, output_path: str, verbose: bool = True) -> bool:
    """Download a file using wget or curl."""
    if verbose:
        print(f"  Downloading from {url}...")
    
    # Try wget first
    try:
        result = subprocess.run(
            ["wget", "-q", "--show-progress", "-O", output_path, url],
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Fall back to curl
    try:
        result = subprocess.run(
            ["curl", "-L", "-o", output_path, url],
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    print(f"  Error: Could not download {url}")
    return False


def extract_archive(archive_path: str, output_dir: str, verbose: bool = True) -> bool:
    """Extract a tar.gz archive."""
    if verbose:
        print(f"  Extracting {archive_path}...")
    
    try:
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(output_dir)
        return True
    except Exception as e:
        print(f"  Error extracting {archive_path}: {e}")
        return False


def find_graph_file(directory: str, formats: List[str] = ['mtx', 'el', 'wel']) -> Optional[str]:
    """Find the graph file in extracted directory."""
    for root, dirs, files in os.walk(directory):
        for f in files:
            for fmt in formats:
                if f.endswith(f'.{fmt}'):
                    return os.path.join(root, f)
    return None


def validate_mtx_file(filepath: str) -> Tuple[bool, str]:
    """
    Validate that an MTX file has a square adjacency matrix.
    
    Returns (is_valid, message).
    """
    try:
        with open(filepath, 'r') as f:
            # Skip comments
            line = f.readline()
            while line.startswith('%'):
                line = f.readline()
            
            # Parse dimensions: rows cols nnz
            parts = line.strip().split()
            if len(parts) < 2:
                return False, f"Invalid header: {line.strip()}"
            
            rows = int(parts[0])
            cols = int(parts[1])
            nnz = int(parts[2]) if len(parts) > 2 else 0
            
            if rows != cols:
                return False, f"Non-square matrix: {rows} x {cols} (this is likely a bipartite/community matrix, not an adjacency matrix)"
            
            if rows == 0 or cols == 0:
                return False, f"Invalid dimensions: {rows} x {cols}"
            
            return True, f"Valid square matrix: {rows} x {cols}, {nnz} edges"
    except Exception as e:
        return False, f"Error reading file: {e}"


def download_graph(
    graph: GraphInfo,
    output_dir: str,
    verbose: bool = True,
    force: bool = False,
    validate: bool = True
) -> Optional[str]:
    """
    Download and extract a graph.
    
    Returns the path to the graph file, or None if failed.
    """
    graph_dir = os.path.join(output_dir, graph.name)
    final_path = os.path.join(graph_dir, f"graph.{graph.format}")
    
    # Check if already exists
    if os.path.exists(final_path) and not force:
        if verbose:
            print(f"  {graph.name}: Already exists at {final_path}")
        
        # Validate existing file
        if validate and graph.format == 'mtx':
            is_valid, msg = validate_mtx_file(final_path)
            if not is_valid:
                print(f"    WARNING: {msg}")
                print(f"    Re-downloading with --force may help, or this graph may be incompatible")
                return None
        
        return final_path
    
    # Create directory
    os.makedirs(graph_dir, exist_ok=True)
    
    # Download
    archive_name = os.path.basename(graph.url)
    archive_path = os.path.join(graph_dir, archive_name)
    
    if not download_file(graph.url, archive_path, verbose):
        return None
    
    # Extract
    if not extract_archive(archive_path, graph_dir, verbose):
        return None
    
    # Find and rename graph file
    graph_file = find_graph_file(graph_dir, [graph.format, 'mtx', 'el', 'wel'])
    if graph_file and graph_file != final_path:
        shutil.move(graph_file, final_path)
    
    # Clean up archive
    if os.path.exists(archive_path):
        os.remove(archive_path)
    
    # Clean up extracted subdirectories
    for item in os.listdir(graph_dir):
        item_path = os.path.join(graph_dir, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
    
    if os.path.exists(final_path):
        # Validate the downloaded file
        if validate and graph.format == 'mtx':
            is_valid, msg = validate_mtx_file(final_path)
            if not is_valid:
                print(f"  {graph.name}: INVALID - {msg}")
                # Remove invalid file
                os.remove(final_path)
                return None
            if verbose:
                print(f"  {graph.name}: {msg}")
        
        if verbose:
            print(f"  {graph.name}: Downloaded to {final_path}")
        return final_path
    else:
        print(f"  {graph.name}: Failed to locate graph file")
        return None


# ============================================================================
# Main Functions
# ============================================================================

def get_graphs_by_size(size: str) -> List[GraphInfo]:
    """Get list of graphs by size category.
    
    Size categories:
    - SMALL: <100MB (quick testing)
    - MEDIUM: 100MB-1GB (standard benchmarking)
    - MID_LARGE: 1GB-8GB (serious benchmarking)
    - LARGE: 8GB-16GB (comprehensive benchmarking)
    - EXTRA_LARGE/XL: >16GB (powerful machines only)
    """
    size = size.upper()
    if size == "SMALL":
        return SMALL_GRAPHS
    elif size == "MEDIUM":
        return SMALL_GRAPHS + MEDIUM_GRAPHS
    elif size == "MID_LARGE" or size == "ML":
        return SMALL_GRAPHS + MEDIUM_GRAPHS + MID_LARGE_GRAPHS
    elif size == "LARGE":
        return SMALL_GRAPHS + MEDIUM_GRAPHS + MID_LARGE_GRAPHS + LARGE_GRAPHS
    elif size == "EXTRA_LARGE" or size == "XL":
        return SMALL_GRAPHS + MEDIUM_GRAPHS + MID_LARGE_GRAPHS + LARGE_GRAPHS + EXTRA_LARGE_GRAPHS
    elif size == "ALL":
        return SMALL_GRAPHS + MEDIUM_GRAPHS + MID_LARGE_GRAPHS + LARGE_GRAPHS + EXTRA_LARGE_GRAPHS
    elif size == "SMALL_ONLY":
        return SMALL_GRAPHS
    elif size == "MEDIUM_ONLY":
        return MEDIUM_GRAPHS
    elif size == "MID_LARGE_ONLY" or size == "ML_ONLY":
        return MID_LARGE_GRAPHS
    elif size == "LARGE_ONLY":
        return LARGE_GRAPHS
    elif size == "EXTRA_LARGE_ONLY" or size == "XL_ONLY":
        return EXTRA_LARGE_GRAPHS
    else:
        return MEDIUM_GRAPHS


def print_catalog(graphs: List[GraphInfo]):
    """Print the graph catalog."""
    print("\nGraph Catalog:")
    print("-" * 100)
    print(f"{'Name':<20} {'Category':<15} {'Nodes':>12} {'Edges':>15} {'Size':>10} {'Format':<6}")
    print("-" * 100)
    
    total_size = 0
    for g in graphs:
        size_str = f"{g.size_mb}MB" if g.size_mb < 1024 else f"{g.size_mb/1024:.1f}GB"
        print(f"{g.name:<20} {g.category:<15} {g.nodes:>12,} {g.edges:>15,} {size_str:>10} {g.format:<6}")
        total_size += g.size_mb
    
    print("-" * 100)
    total_str = f"{total_size}MB" if total_size < 1024 else f"{total_size/1024:.1f}GB"
    print(f"Total: {len(graphs)} graphs, {total_str} download size")


def download_all(
    graphs: List[GraphInfo],
    output_dir: str,
    verbose: bool = True,
    force: bool = False
) -> Dict[str, str]:
    """
    Download all graphs in the list.
    
    Returns dict mapping graph name to file path.
    """
    results = {}
    
    print(f"\nDownloading {len(graphs)} graphs to {output_dir}...")
    print("=" * 70)
    
    for i, graph in enumerate(graphs, 1):
        print(f"\n[{i}/{len(graphs)}] {graph.name} ({graph.size_mb}MB)")
        path = download_graph(graph, output_dir, verbose, force)
        if path:
            results[graph.name] = path
    
    print("\n" + "=" * 70)
    print(f"Downloaded {len(results)}/{len(graphs)} graphs successfully")
    
    return results


def create_graph_config(
    graphs: Dict[str, str],
    output_path: str,
    symmetric_flags: Dict[str, bool] = None
):
    """Create a JSON config file for the downloaded graphs."""
    config = {
        "graphs": {},
        "description": "Auto-generated graph configuration"
    }
    
    # Build symmetric flags from catalog
    if symmetric_flags is None:
        symmetric_flags = {}
        for g in SMALL_GRAPHS + MEDIUM_GRAPHS + LARGE_GRAPHS:
            symmetric_flags[g.name] = g.symmetric
    
    for name, path in graphs.items():
        sym = symmetric_flags.get(name, True)
        config["graphs"][name] = {
            "path": path,
            "symmetric": sym,
            "args": f"-f {path}" + (" -s" if sym else "")
        }
    
    import json
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nCreated graph config: {output_path}")


def validate_existing_graphs(output_dir: str) -> Dict[str, Tuple[bool, str]]:
    """
    Validate all existing graph files in the output directory.
    
    Returns dict mapping graph name to (is_valid, message).
    """
    results = {}
    
    if not os.path.exists(output_dir):
        print(f"Directory not found: {output_dir}")
        return results
    
    print(f"\nValidating graphs in {output_dir}...")
    print("=" * 70)
    
    for name in sorted(os.listdir(output_dir)):
        graph_dir = os.path.join(output_dir, name)
        if not os.path.isdir(graph_dir):
            continue
        
        mtx_path = os.path.join(graph_dir, "graph.mtx")
        if os.path.exists(mtx_path):
            is_valid, msg = validate_mtx_file(mtx_path)
            status = "✓" if is_valid else "✗"
            print(f"{status} {name:<24} {msg}")
            results[name] = (is_valid, msg)
    
    valid_count = sum(1 for v, _ in results.values() if v)
    print("=" * 70)
    print(f"Valid: {valid_count}/{len(results)} graphs")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Download graphs for GraphBrew benchmarking"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./graphs",
        help="Output directory for downloaded graphs (default: ./graphs)"
    )
    parser.add_argument(
        "--size", "-s",
        choices=["SMALL", "MEDIUM", "MID_LARGE", "ML", "LARGE", "EXTRA_LARGE", "XL", "ALL", 
                 "SMALL_ONLY", "MEDIUM_ONLY", "MID_LARGE_ONLY", "ML_ONLY", 
                 "LARGE_ONLY", "EXTRA_LARGE_ONLY", "XL_ONLY"],
        default="MEDIUM",
        help="Size: SMALL(<100MB), MEDIUM(<1GB), MID_LARGE/ML(1-8GB), LARGE(8-16GB), XL(>16GB)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available graphs without downloading"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download even if files exist"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to save graph configuration JSON"
    )
    parser.add_argument(
        "--validate", "-v",
        action="store_true",
        help="Validate existing graphs without downloading"
    )
    
    args = parser.parse_args()
    
    graphs = get_graphs_by_size(args.size)
    
    if args.list:
        print_catalog(graphs)
        return 0
    
    if args.validate:
        output_dir = os.path.abspath(args.output_dir)
        validate_existing_graphs(output_dir)
        return 0
    
    # Check disk space
    total_size_mb = sum(g.size_mb for g in graphs)
    print(f"\nWill download approximately {total_size_mb/1024:.1f}GB of data")
    
    # Download
    output_dir = os.path.abspath(args.output_dir)
    results = download_all(graphs, output_dir, force=args.force)
    
    # Create config
    if args.config:
        create_graph_config(results, args.config)
    else:
        config_path = os.path.join(output_dir, "graphs.json")
        create_graph_config(results, config_path)
    
    return 0 if len(results) == len(graphs) else 1


if __name__ == "__main__":
    sys.exit(main())
