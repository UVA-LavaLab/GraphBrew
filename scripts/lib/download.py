#!/usr/bin/env python3
"""
Graph downloading utilities for GraphBrew.

Downloads benchmark graphs from SuiteSparse Matrix Collection.
Can be used standalone or as a library.

Standalone usage:
    python -m scripts.lib.download --size SMALL
    python -m scripts.lib.download --list
    python -m scripts.lib.download --graph email-Enron

Library usage:
    from scripts.lib.download import download_graphs, GRAPH_CATALOG
    
    download_graphs(size="SMALL")
    download_graphs(graphs=["email-Enron", "web-Google"])
"""

import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from .utils import GRAPHS_DIR, log, ensure_directories


# =============================================================================
# Graph Catalog
# =============================================================================

@dataclass
class GraphInfo:
    """Information about a downloadable graph."""
    name: str
    url: str
    size_mb: int
    nodes: int
    edges: int
    category: str
    description: str = ""


# Small graphs (< 100MB, good for testing)
SMALL_GRAPHS = {
    "email-Enron": GraphInfo(
        name="email-Enron",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/email-Enron.tar.gz",
        size_mb=5, nodes=36692, edges=183831,
        category="communication",
        description="Enron email network"
    ),
    "ca-AstroPh": GraphInfo(
        name="ca-AstroPh",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/ca-AstroPh.tar.gz",
        size_mb=3, nodes=18772, edges=198110,
        category="collaboration",
        description="Arxiv Astro Physics collaboration"
    ),
    "ca-CondMat": GraphInfo(
        name="ca-CondMat",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/ca-CondMat.tar.gz",
        size_mb=2, nodes=23133, edges=93497,
        category="collaboration",
        description="Condensed Matter collaboration"
    ),
    "ca-GrQc": GraphInfo(
        name="ca-GrQc",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/ca-GrQc.tar.gz",
        size_mb=1, nodes=5242, edges=14496,
        category="collaboration",
        description="General Relativity collaboration"
    ),
    "ca-HepPh": GraphInfo(
        name="ca-HepPh",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/ca-HepPh.tar.gz",
        size_mb=2, nodes=12008, edges=118521,
        category="collaboration",
        description="High Energy Physics collaboration"
    ),
    "ca-HepTh": GraphInfo(
        name="ca-HepTh",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/ca-HepTh.tar.gz",
        size_mb=1, nodes=9877, edges=25998,
        category="collaboration",
        description="High Energy Physics Theory collaboration"
    ),
    "p2p-Gnutella31": GraphInfo(
        name="p2p-Gnutella31",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/p2p-Gnutella31.tar.gz",
        size_mb=2, nodes=62586, edges=147892,
        category="p2p",
        description="Gnutella P2P network"
    ),
    "soc-Epinions1": GraphInfo(
        name="soc-Epinions1",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/soc-Epinions1.tar.gz",
        size_mb=6, nodes=75879, edges=508837,
        category="social",
        description="Epinions social network"
    ),
    "soc-Slashdot0811": GraphInfo(
        name="soc-Slashdot0811",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/soc-Slashdot0811.tar.gz",
        size_mb=5, nodes=77360, edges=905468,
        category="social",
        description="Slashdot social network"
    ),
    "wiki-Vote": GraphInfo(
        name="wiki-Vote",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/wiki-Vote.tar.gz",
        size_mb=1, nodes=7115, edges=103689,
        category="social",
        description="Wikipedia voting network"
    ),
    "cit-HepPh": GraphInfo(
        name="cit-HepPh",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/cit-HepPh.tar.gz",
        size_mb=10, nodes=34546, edges=421578,
        category="citation",
        description="High Energy Physics citation"
    ),
    "cit-HepTh": GraphInfo(
        name="cit-HepTh",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/cit-HepTh.tar.gz",
        size_mb=5, nodes=27770, edges=352807,
        category="citation",
        description="High Energy Physics Theory citation"
    ),
}

# Medium graphs (100MB - 1GB)
MEDIUM_GRAPHS = {
    "web-Google": GraphInfo(
        name="web-Google",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/web-Google.tar.gz",
        size_mb=65, nodes=916428, edges=5105039,
        category="web",
        description="Google web graph"
    ),
    "web-NotreDame": GraphInfo(
        name="web-NotreDame",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/web-NotreDame.tar.gz",
        size_mb=16, nodes=325729, edges=1497134,
        category="web",
        description="Notre Dame web graph"
    ),
    "web-Stanford": GraphInfo(
        name="web-Stanford",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/web-Stanford.tar.gz",
        size_mb=20, nodes=281903, edges=2312497,
        category="web",
        description="Stanford web graph"
    ),
    "web-BerkStan": GraphInfo(
        name="web-BerkStan",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/web-BerkStan.tar.gz",
        size_mb=65, nodes=685230, edges=7600595,
        category="web",
        description="Berkeley-Stanford web graph"
    ),
    "roadNet-CA": GraphInfo(
        name="roadNet-CA",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/roadNet-CA.tar.gz",
        size_mb=35, nodes=1965206, edges=5533214,
        category="road",
        description="California road network"
    ),
    "roadNet-PA": GraphInfo(
        name="roadNet-PA",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/roadNet-PA.tar.gz",
        size_mb=20, nodes=1088092, edges=3083796,
        category="road",
        description="Pennsylvania road network"
    ),
    "roadNet-TX": GraphInfo(
        name="roadNet-TX",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/roadNet-TX.tar.gz",
        size_mb=25, nodes=1379917, edges=3843320,
        category="road",
        description="Texas road network"
    ),
    "amazon0302": GraphInfo(
        name="amazon0302",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/amazon0302.tar.gz",
        size_mb=15, nodes=262111, edges=1234877,
        category="commerce",
        description="Amazon product co-purchasing"
    ),
    "amazon0601": GraphInfo(
        name="amazon0601",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/amazon0601.tar.gz",
        size_mb=35, nodes=403394, edges=3387388,
        category="commerce",
        description="Amazon product co-purchasing"
    ),
    "wiki-Talk": GraphInfo(
        name="wiki-Talk",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/wiki-Talk.tar.gz",
        size_mb=35, nodes=2394385, edges=5021410,
        category="communication",
        description="Wikipedia talk network"
    ),
    "email-EuAll": GraphInfo(
        name="email-EuAll",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/email-EuAll.tar.gz",
        size_mb=15, nodes=265214, edges=420045,
        category="communication",
        description="EU email network"
    ),
}

# Large graphs (> 1GB)  
LARGE_GRAPHS = {
    "soc-LiveJournal1": GraphInfo(
        name="soc-LiveJournal1",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/soc-LiveJournal1.tar.gz",
        size_mb=500, nodes=4847571, edges=68993773,
        category="social",
        description="LiveJournal social network"
    ),
    "cit-Patents": GraphInfo(
        name="cit-Patents",
        url="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/cit-Patents.tar.gz",
        size_mb=200, nodes=3774768, edges=16518948,
        category="citation",
        description="US Patent citation network"
    ),
}

# Combined catalog
GRAPH_CATALOG = {
    "SMALL": SMALL_GRAPHS,
    "MEDIUM": MEDIUM_GRAPHS,
    "LARGE": LARGE_GRAPHS,
}


# =============================================================================
# Download Functions
# =============================================================================

def get_catalog(size: str = None) -> Dict[str, GraphInfo]:
    """Get graph catalog for specified size(s)."""
    if size is None or size.upper() == "ALL":
        result = {}
        for s in ["SMALL", "MEDIUM", "LARGE"]:
            result.update(GRAPH_CATALOG.get(s, {}))
        return result
    return GRAPH_CATALOG.get(size.upper(), {})


def download_file(url: str, dest_path: Path) -> bool:
    """Download a file from URL to destination."""
    try:
        log.debug(f"Downloading {url}")
        urllib.request.urlretrieve(url, dest_path)
        return True
    except Exception as e:
        log.error(f"Failed to download {url}: {e}")
        return False


def extract_tarball(tar_path: Path, dest_dir: Path) -> bool:
    """Extract tarball to destination directory."""
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(dest_dir)
        return True
    except Exception as e:
        log.error(f"Failed to extract {tar_path}: {e}")
        return False


def download_graph(
    name: str,
    info: GraphInfo,
    dest_dir: Path = None,
    force: bool = False
) -> Optional[Path]:
    """
    Download and extract a single graph.
    
    Args:
        name: Graph name
        info: GraphInfo object
        dest_dir: Destination directory (default: GRAPHS_DIR)
        force: Force re-download even if exists
        
    Returns:
        Path to extracted graph file, or None if failed
    """
    if dest_dir is None:
        dest_dir = GRAPHS_DIR
    
    graph_dir = dest_dir / name
    
    # Check if already exists
    if graph_dir.exists() and not force:
        mtx_files = list(graph_dir.glob("*.mtx"))
        if mtx_files:
            log.debug(f"Graph {name} already exists at {mtx_files[0]}")
            return mtx_files[0]
    
    # Create directory
    graph_dir.mkdir(parents=True, exist_ok=True)
    
    # Download
    tar_path = graph_dir / f"{name}.tar.gz"
    log.info(f"Downloading {name} ({info.size_mb}MB)...")
    
    if not download_file(info.url, tar_path):
        return None
    
    # Extract
    log.info(f"Extracting {name}...")
    if not extract_tarball(tar_path, graph_dir):
        return None
    
    # Clean up tarball
    tar_path.unlink()
    
    # Find extracted .mtx file
    mtx_files = list(graph_dir.rglob("*.mtx"))
    if not mtx_files:
        log.error(f"No .mtx file found after extracting {name}")
        return None
    
    log.success(f"Downloaded {name} to {mtx_files[0]}")
    return mtx_files[0]


def download_graphs(
    size: str = "SMALL",
    graphs: List[str] = None,
    category: str = None,
    dest_dir: Path = None,
    max_workers: int = 4,
    force: bool = False
) -> List[Path]:
    """
    Download multiple graphs.
    
    Args:
        size: Size category ("SMALL", "MEDIUM", "LARGE", "ALL")
        graphs: Specific graph names to download (overrides size)
        category: Filter by category (e.g., "social", "web", "road")
        dest_dir: Destination directory
        max_workers: Parallel download threads
        force: Force re-download
        
    Returns:
        List of paths to downloaded graph files
    """
    ensure_directories()
    
    if dest_dir is None:
        dest_dir = GRAPHS_DIR
    
    # Build download list
    catalog = get_catalog(size)
    
    if graphs:
        # Filter to specific graphs
        catalog = {k: v for k, v in catalog.items() if k in graphs}
    
    if category:
        catalog = {k: v for k, v in catalog.items() if v.category == category}
    
    if not catalog:
        log.warning("No graphs match the specified criteria")
        return []
    
    total_mb = sum(g.size_mb for g in catalog.values())
    log.info(f"Downloading {len(catalog)} graphs (~{total_mb}MB total)...")
    
    # Download with threading
    downloaded = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_graph, name, info, dest_dir, force): name
            for name, info in catalog.items()
        }
        
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                if result:
                    downloaded.append(result)
            except Exception as e:
                log.error(f"Failed to download {name}: {e}")
    
    log.info(f"Downloaded {len(downloaded)}/{len(catalog)} graphs")
    return downloaded


def list_available_graphs() -> None:
    """Print available graphs in catalog."""
    print("\n" + "=" * 70)
    print("Available Graphs in Catalog")
    print("=" * 70)
    
    for size_name in ["SMALL", "MEDIUM", "LARGE"]:
        catalog = GRAPH_CATALOG.get(size_name, {})
        total_mb = sum(g.size_mb for g in catalog.values())
        print(f"\n{size_name} ({len(catalog)} graphs, {total_mb}MB total)")
        print("-" * 60)
        
        for name, info in sorted(catalog.items()):
            print(f"  {name:25s} {info.category:15s} {info.size_mb:5d}MB  {info.nodes:>10,} nodes")


def list_downloaded_graphs(dest_dir: Path = None) -> List[Dict]:
    """List downloaded graphs with their info."""
    if dest_dir is None:
        dest_dir = GRAPHS_DIR
    
    graphs = []
    if not dest_dir.exists():
        return graphs
    
    for graph_dir in dest_dir.iterdir():
        if not graph_dir.is_dir():
            continue
        
        mtx_files = list(graph_dir.glob("*.mtx"))
        if mtx_files:
            mtx = mtx_files[0]
            size_mb = mtx.stat().st_size / (1024 * 1024)
            graphs.append({
                "name": graph_dir.name,
                "path": str(mtx),
                "size_mb": size_mb
            })
    
    return graphs


# =============================================================================
# Standalone CLI
# =============================================================================

def main():
    """CLI for graph downloading."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download benchmark graphs for GraphBrew",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m scripts.lib.download --size SMALL
    python -m scripts.lib.download --graph email-Enron web-Google
    python -m scripts.lib.download --list
    python -m scripts.lib.download --list-downloaded
        """
    )
    
    parser.add_argument("--size", choices=["SMALL", "MEDIUM", "LARGE", "ALL"],
                       default="SMALL", help="Size category to download")
    parser.add_argument("--graph", nargs="+", dest="graphs",
                       help="Specific graphs to download")
    parser.add_argument("--category", help="Filter by category (social, web, road, etc.)")
    parser.add_argument("--list", action="store_true", help="List available graphs")
    parser.add_argument("--list-downloaded", action="store_true",
                       help="List already downloaded graphs")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--workers", type=int, default=4,
                       help="Parallel download workers")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_graphs()
        return
    
    if args.list_downloaded:
        graphs = list_downloaded_graphs()
        print(f"\nDownloaded graphs ({len(graphs)}):")
        for g in graphs:
            print(f"  {g['name']:30s} {g['size_mb']:8.1f}MB  {g['path']}")
        return
    
    # Download
    downloaded = download_graphs(
        size=args.size,
        graphs=args.graphs,
        category=args.category,
        max_workers=args.workers,
        force=args.force
    )
    
    print(f"\nSuccessfully downloaded {len(downloaded)} graphs")


if __name__ == "__main__":
    main()
