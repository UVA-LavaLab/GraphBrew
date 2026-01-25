#!/usr/bin/env python3
"""
Graph feature computation and system utilities for GraphBrew.

This module provides:
- Graph type detection and classification
- Topological feature computation (clustering coefficient, diameter, etc.)
- Graph properties caching
- System resource utilities (memory, disk)

Can be used standalone or as a library.

Standalone usage:
    python -m scripts.lib.features --detect-type graph.mtx
    python -m scripts.lib.features --system-info
"""

import json
import math
import os
import random
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# =============================================================================
# Graph Type Constants (must match C++ GraphType enum in builder.h)
# =============================================================================

GRAPH_TYPE_GENERIC = "generic"
GRAPH_TYPE_SOCIAL = "social"
GRAPH_TYPE_ROAD = "road"
GRAPH_TYPE_WEB = "web"
GRAPH_TYPE_POWERLAW = "powerlaw"
GRAPH_TYPE_UNIFORM = "uniform"

ALL_GRAPH_TYPES = [
    GRAPH_TYPE_GENERIC,
    GRAPH_TYPE_SOCIAL,
    GRAPH_TYPE_ROAD,
    GRAPH_TYPE_WEB,
    GRAPH_TYPE_POWERLAW,
    GRAPH_TYPE_UNIFORM,
]

# Memory estimation constants (bytes per edge/node for graph algorithms)
# Based on CSR format: ~24 bytes/edge + 8 bytes/node for working memory
BYTES_PER_EDGE = 24
BYTES_PER_NODE = 8
MEMORY_SAFETY_FACTOR = 1.5  # Add 50% buffer for algorithm overhead


# =============================================================================
# Graph Properties Cache
# =============================================================================

# Global cache for graph properties (populated during benchmark runs)
_graph_properties_cache: Dict[str, Dict] = {}


def get_graph_properties_cache_file(output_dir: str = "results") -> str:
    """Get the path to the graph properties cache file."""
    return os.path.join(output_dir, "graph_properties_cache.json")


def load_graph_properties_cache(output_dir: str = "results") -> Dict[str, Dict]:
    """Load graph properties cache from file."""
    global _graph_properties_cache
    cache_file = get_graph_properties_cache_file(output_dir)
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                _graph_properties_cache = json.load(f)
        except Exception:
            _graph_properties_cache = {}
    return _graph_properties_cache


def save_graph_properties_cache(output_dir: str = "results"):
    """Save graph properties cache to file."""
    global _graph_properties_cache
    cache_file = get_graph_properties_cache_file(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(_graph_properties_cache, f, indent=2)


def update_graph_properties(graph_name: str, properties: Dict, output_dir: str = "results"):
    """
    Update graph properties in the cache.
    
    Args:
        graph_name: Name of the graph
        properties: Dict with keys like 'modularity', 'degree_variance', 
                   'hub_concentration', 'avg_degree', 'graph_type', 'nodes', 'edges'
    """
    global _graph_properties_cache
    if graph_name not in _graph_properties_cache:
        _graph_properties_cache[graph_name] = {}
    
    # Update with new properties (don't overwrite existing valid values with None)
    for key, value in properties.items():
        if value is not None:
            _graph_properties_cache[graph_name][key] = value
    
    # Auto-detect graph type if we have enough features
    if 'graph_type' not in _graph_properties_cache[graph_name]:
        props = _graph_properties_cache[graph_name]
        if all(k in props for k in ['modularity', 'degree_variance', 'hub_concentration']):
            avg_degree = props.get('avg_degree', 0)
            if avg_degree == 0 and 'nodes' in props and 'edges' in props:
                avg_degree = 2 * props['edges'] / props['nodes'] if props['nodes'] > 0 else 0
            
            _graph_properties_cache[graph_name]['graph_type'] = detect_graph_type(
                props['modularity'],
                props['degree_variance'],
                props['hub_concentration'],
                avg_degree
            )


def get_graph_properties(graph_name: str) -> Dict:
    """Get cached properties for a graph."""
    global _graph_properties_cache
    return _graph_properties_cache.get(graph_name, {})


def clear_graph_properties_cache():
    """Clear the in-memory graph properties cache."""
    global _graph_properties_cache
    _graph_properties_cache = {}


# =============================================================================
# Graph Type Detection
# =============================================================================

def detect_graph_type(modularity: float, degree_variance: float, 
                      hub_concentration: float, avg_degree: float,
                      num_nodes: int = 0) -> str:
    """
    Detect graph type based on topological features.
    
    Uses the same decision tree as C++ DetectGraphType() in builder.h.
    
    Args:
        modularity: Leiden modularity score (0-1)
        degree_variance: Coefficient of variation of degrees
        hub_concentration: Fraction of edges to top 10% nodes
        avg_degree: Average node degree
        num_nodes: Number of nodes (for additional heuristics)
    
    Returns:
        One of: 'generic', 'social', 'road', 'web', 'powerlaw', 'uniform'
    """
    # Decision tree matching C++ implementation
    
    # Road networks: very regular, low degree variance, sparse
    if modularity < 0.1 and degree_variance < 0.5 and avg_degree < 10:
        return GRAPH_TYPE_ROAD
    
    # Social networks: high modularity, high degree variance
    if modularity > 0.3 and degree_variance > 0.8:
        return GRAPH_TYPE_SOCIAL
    
    # Web graphs: high hub concentration, power-law like
    if hub_concentration > 0.5 and degree_variance > 1.0:
        return GRAPH_TYPE_WEB
    
    # Power-law (scale-free) graphs: very high degree variance
    if degree_variance > 1.5 and modularity < 0.3:
        return GRAPH_TYPE_POWERLAW
    
    # Uniform random graphs: low variance, no clear structure
    if degree_variance < 0.5 and hub_concentration < 0.3 and modularity < 0.1:
        return GRAPH_TYPE_UNIFORM
    
    # Default fallback
    return GRAPH_TYPE_GENERIC


def get_graph_type_from_name(graph_name: str) -> str:
    """
    Heuristic graph type detection based on graph name.
    
    Used as a fallback when we don't have computed features.
    """
    name_lower = graph_name.lower()
    
    # Social networks
    if any(x in name_lower for x in ['soc-', 'social', 'twitter', 'facebook', 'friendster', 'orkut']):
        return GRAPH_TYPE_SOCIAL
    
    # Road networks
    if any(x in name_lower for x in ['road', 'osm', 'tiger', 'rgg_']):
        return GRAPH_TYPE_ROAD
    
    # Web graphs
    if any(x in name_lower for x in ['web-', 'uk-', 'it-', 'arabic', 'webbase', 'indochina']):
        return GRAPH_TYPE_WEB
    
    # Power-law / scale-free
    if any(x in name_lower for x in ['kron', 'rmat', 'gap-kron']):
        return GRAPH_TYPE_POWERLAW
    
    # Uniform random
    if any(x in name_lower for x in ['urand', 'erdos', 'gap-urand', 'random']):
        return GRAPH_TYPE_UNIFORM
    
    return GRAPH_TYPE_GENERIC


def get_graph_type_from_properties(graph_name: str, fallback_to_name: bool = True) -> str:
    """
    Get graph type from cached properties, or fall back to name heuristic.
    
    Args:
        graph_name: Name of the graph
        fallback_to_name: If True, use name-based heuristic when no cached properties
    
    Returns:
        Graph type string
    """
    global _graph_properties_cache
    
    if graph_name in _graph_properties_cache:
        props = _graph_properties_cache[graph_name]
        
        # If we have pre-computed graph type, use it
        if 'graph_type' in props:
            return props['graph_type']
        
        # Try to detect from stored features
        if all(k in props for k in ['modularity', 'degree_variance', 'hub_concentration']):
            avg_degree = props.get('avg_degree', 0)
            if avg_degree == 0 and 'nodes' in props and 'edges' in props:
                avg_degree = 2 * props['edges'] / props['nodes'] if props['nodes'] > 0 else 0
            
            return detect_graph_type(
                props['modularity'],
                props['degree_variance'],
                props['hub_concentration'],
                avg_degree
            )
    
    # Fall back to name-based heuristic
    if fallback_to_name:
        return get_graph_type_from_name(graph_name)
    
    return GRAPH_TYPE_GENERIC


# =============================================================================
# Topological Feature Computation
# =============================================================================

def compute_clustering_coefficient_sample(adjacency_list: Dict[int, List[int]], 
                                          sample_size: int = 1000) -> float:
    """
    Compute average local clustering coefficient using sampling for large graphs.
    Clustering coefficient measures how connected a node's neighbors are to each other.
    
    Args:
        adjacency_list: Dict mapping node ID to list of neighbor IDs
        sample_size: Number of nodes to sample (for large graphs)
    
    Returns:
        Average clustering coefficient in [0, 1]
    """
    if not adjacency_list:
        return 0.0
    
    nodes = list(adjacency_list.keys())
    if len(nodes) > sample_size:
        nodes = random.sample(nodes, sample_size)
    
    total_cc = 0.0
    valid_nodes = 0
    
    for node in nodes:
        neighbors = adjacency_list.get(node, [])
        k = len(neighbors)
        if k < 2:
            continue
        
        # Count edges between neighbors
        neighbor_set = set(neighbors)
        triangles = 0
        for neighbor in neighbors:
            for n2 in adjacency_list.get(neighbor, []):
                if n2 in neighbor_set and n2 != node:
                    triangles += 1
        
        # Each triangle is counted twice
        triangles //= 2
        possible_triangles = k * (k - 1) // 2
        
        if possible_triangles > 0:
            total_cc += triangles / possible_triangles
            valid_nodes += 1
    
    return total_cc / valid_nodes if valid_nodes > 0 else 0.0


def estimate_diameter_bfs(adjacency_list: Dict[int, List[int]], 
                          num_samples: int = 10) -> Tuple[float, float]:
    """
    Estimate graph diameter and average path length using BFS from random samples.
    
    Args:
        adjacency_list: Dict mapping node ID to list of neighbor IDs
        num_samples: Number of BFS runs from random starting nodes
    
    Returns:
        (diameter_estimate, avg_path_length) tuple
    """
    if not adjacency_list:
        return 0.0, 0.0
    
    nodes = list(adjacency_list.keys())
    if len(nodes) < 2:
        return 0.0, 0.0
    
    # Sample starting nodes
    sample_nodes = random.sample(nodes, min(num_samples, len(nodes)))
    
    max_distance = 0
    total_distance = 0
    path_count = 0
    
    for start in sample_nodes:
        # BFS from start
        distances = {start: 0}
        queue = deque([start])
        
        while queue:
            node = queue.popleft()
            current_dist = distances[node]
            
            for neighbor in adjacency_list.get(node, []):
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
                    
                    if distances[neighbor] > max_distance:
                        max_distance = distances[neighbor]
        
        # Accumulate path lengths
        for dist in distances.values():
            if dist > 0:
                total_distance += dist
                path_count += 1
    
    avg_path = total_distance / path_count if path_count > 0 else 0.0
    return float(max_distance), avg_path


def count_subcommunities_quick(adjacency_list: Dict[int, List[int]], 
                               threshold: int = 100) -> int:
    """
    Quick estimate of number of connected components/communities using union-find.
    For small graphs, returns exact count. For large graphs, estimates.
    
    Args:
        adjacency_list: Dict mapping node ID to list of neighbor IDs
        threshold: Not currently used, kept for API compatibility
    
    Returns:
        Number of connected components
    """
    if not adjacency_list:
        return 0
    
    # Union-Find
    parent = {node: node for node in adjacency_list}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Build connected components
    for node, neighbors in adjacency_list.items():
        for neighbor in neighbors:
            if neighbor in parent:
                union(node, neighbor)
    
    # Count unique roots
    roots = set(find(node) for node in adjacency_list)
    return len(roots)


def compute_extended_features(nodes: int, edges: int, density: float, 
                              degree_variance: float, hub_concentration: float,
                              adjacency_list: Optional[Dict[int, List[int]]] = None) -> Dict:
    """
    Compute extended feature set for a subcommunity/subgraph.
    
    Args:
        nodes: Number of nodes
        edges: Number of edges
        density: Edge density
        degree_variance: Coefficient of variation of degrees
        hub_concentration: Fraction of edges to top nodes
        adjacency_list: Optional adjacency list for computing additional features
    
    Returns:
        Dictionary with all computed features
    """
    features = {
        'nodes': nodes,
        'edges': edges,
        'density': density,
        'degree_variance': degree_variance,
        'hub_concentration': hub_concentration,
        'avg_degree': (2 * edges / nodes) if nodes > 0 else 0,
        'clustering_coefficient': 0.0,
        'avg_path_length': 0.0,
        'diameter_estimate': 0.0,
        'community_count': 1,
    }
    
    # If we have the adjacency list, compute additional features
    if adjacency_list and len(adjacency_list) > 0 and len(adjacency_list) < 50000:
        try:
            features['clustering_coefficient'] = compute_clustering_coefficient_sample(adjacency_list)
            diameter, avg_path = estimate_diameter_bfs(adjacency_list)
            features['diameter_estimate'] = diameter
            features['avg_path_length'] = avg_path
            features['community_count'] = count_subcommunities_quick(adjacency_list)
        except Exception:
            pass  # Use defaults on error
    
    return features


# =============================================================================
# System Resource Utilities
# =============================================================================

def get_available_memory_gb() -> float:
    """Get available system RAM in GB."""
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    # Value is in kB
                    return int(line.split()[1]) / (1024 * 1024)
    except:
        pass
    # Fallback: try psutil if available
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except:
        pass
    # Default to 8GB if we can't detect
    return 8.0


def get_total_memory_gb() -> float:
    """Get total system RAM in GB."""
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    return int(line.split()[1]) / (1024 * 1024)
    except:
        pass
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except:
        pass
    return 16.0


def estimate_graph_memory_gb(nodes: int, edges: int) -> float:
    """
    Estimate memory required for a graph in GB.
    
    Based on CSR format storage plus working memory for algorithms.
    """
    memory_bytes = (edges * BYTES_PER_EDGE + nodes * BYTES_PER_NODE) * MEMORY_SAFETY_FACTOR
    return memory_bytes / (1024 ** 3)


def get_available_disk_gb(path: str = ".") -> float:
    """Get available disk space in GB for the given path."""
    try:
        import shutil
        usage = shutil.disk_usage(path)
        return usage.free / (1024 ** 3)
    except:
        pass
    try:
        # Fallback using statvfs
        stat = os.statvfs(path)
        return (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
    except:
        pass
    # Default to 100GB if we can't detect
    return 100.0


def get_total_disk_gb(path: str = ".") -> float:
    """Get total disk space in GB for the given path."""
    try:
        import shutil
        usage = shutil.disk_usage(path)
        return usage.total / (1024 ** 3)
    except:
        pass
    try:
        stat = os.statvfs(path)
        return (stat.f_blocks * stat.f_frsize) / (1024 ** 3)
    except:
        pass
    return 500.0


def get_num_threads() -> int:
    """Get the number of available CPU threads."""
    try:
        return os.cpu_count() or 1
    except:
        return 1


# =============================================================================
# Main (for standalone usage)
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Graph feature utilities")
    parser.add_argument("--system-info", action="store_true", help="Show system info")
    parser.add_argument("--list-types", action="store_true", help="List graph types")
    args = parser.parse_args()
    
    if args.system_info:
        print(f"Available memory: {get_available_memory_gb():.1f} GB")
        print(f"Total memory: {get_total_memory_gb():.1f} GB")
        print(f"Available disk: {get_available_disk_gb():.1f} GB")
        print(f"Total disk: {get_total_disk_gb():.1f} GB")
        print(f"CPU threads: {get_num_threads()}")
    
    if args.list_types:
        print("Graph types:", ", ".join(ALL_GRAPH_TYPES))
