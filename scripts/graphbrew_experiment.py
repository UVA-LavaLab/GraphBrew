#!/usr/bin/env python3
"""
GraphBrew Unified Experiment Pipeline
=====================================

A comprehensive one-click script that runs the complete GraphBrew experiment workflow:

1. Download graphs (if not present)
2. Build binaries (if not present)
3. Pre-generate reorderings with label-mapping for consistency
4. Record reorder times for all algorithms on all graphs
5. Run execution benchmarks on all graphs
6. Run cache simulations  
7. Generate perceptron weights with reorder time included
8. Run brute-force validation (adaptive vs all algorithms)
9. Update zero weights based on correlation analysis

All outputs are saved to the results/ directory for clean organization.

Usage:
    python scripts/graphbrew_experiment.py --help
    python scripts/graphbrew_experiment.py --full                  # Full pipeline from scratch
    python scripts/graphbrew_experiment.py --download-only         # Just download graphs
    python scripts/graphbrew_experiment.py --phase all             # Run all experiment phases
    python scripts/graphbrew_experiment.py --brute-force           # Run brute-force validation

Quick Start (One-Click):
    python scripts/graphbrew_experiment.py --full --graphs small   # Full run with small graphs

Author: GraphBrew Team
"""

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
import math
import glob
import shutil
import tarfile
import gzip
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import urllib.request
import urllib.error

# ============================================================================
# Configuration
# ============================================================================

# Algorithm definitions
ALGORITHMS = {
    0: "ORIGINAL",
    1: "RANDOM",
    2: "SORT",
    3: "HUBSORT",
    4: "HUBCLUSTER",
    5: "DBG",
    6: "HUBSORTDBG",
    7: "HUBCLUSTERDBG",
    8: "RABBITORDER",
    9: "GORDER",
    10: "CORDER",
    11: "RCM",
    12: "LeidenOrder",
    13: "GraphBrewOrder",
    # 14: MAP - uses external file
    15: "AdaptiveOrder",
    16: "LeidenDFS",
    17: "LeidenDFSHub",
    18: "LeidenDFSSize",
    19: "LeidenBFS",
    20: "LeidenHybrid",
}

# Algorithms to benchmark (excluding MAP=14 and AdaptiveOrder=15)
BENCHMARK_ALGORITHMS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20]

# Subset of key algorithms for quick testing
KEY_ALGORITHMS = [0, 1, 7, 8, 9, 11, 12, 17, 20]

# Algorithms known to be slow on large graphs
SLOW_ALGORITHMS = {9, 10, 11}  # GORDER, CORDER, RCM

# Benchmarks to run
BENCHMARKS = ["pr", "bfs", "cc", "sssp", "bc"]

# Benchmarks that are computationally intensive in simulation
HEAVY_SIM_BENCHMARKS = {"bc", "sssp"}

# Default paths - ALL outputs go to results/
DEFAULT_RESULTS_DIR = "./results"
DEFAULT_GRAPHS_DIR = "./results/graphs"
DEFAULT_BIN_DIR = "./bench/bin"
DEFAULT_BIN_SIM_DIR = "./bench/bin_sim"
DEFAULT_WEIGHTS_DIR = "./scripts/weights"  # Auto-generated type weights
DEFAULT_MAPPINGS_DIR = "./results/mappings"

# Auto-clustering configuration
CLUSTER_DISTANCE_THRESHOLD = 0.3  # Max normalized distance to join existing cluster
MIN_SAMPLES_FOR_CLUSTER = 3  # Minimum graphs to form a stable cluster

# Graph size categories (MB)
SIZE_SMALL = 50
SIZE_MEDIUM = 500
SIZE_LARGE = 2000

# Memory estimation constants (bytes per edge/node for graph algorithms)
# Based on CSR format: ~24 bytes/edge + 8 bytes/node for working memory
BYTES_PER_EDGE = 24
BYTES_PER_NODE = 8
MEMORY_SAFETY_FACTOR = 1.5  # Add 50% buffer for algorithm overhead

# Default timeouts (seconds)
TIMEOUT_REORDER = 43200     # 12 hours for reordering (some algorithms like GORDER are slow on large graphs)
TIMEOUT_BENCHMARK = 600     # 10 min for benchmarks
TIMEOUT_SIM = 1200          # 20 min for simulations
TIMEOUT_SIM_HEAVY = 3600    # 1 hour for heavy simulations

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class GraphInfo:
    """Information about a graph dataset."""
    name: str
    path: str
    size_mb: float
    is_symmetric: bool = True
    nodes: int = 0
    edges: int = 0

# ============================================================================
# Graph Feature Computation Functions
# ============================================================================

# Graph type constants (must match C++ GraphType enum in builder.h)
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


def compute_clustering_coefficient_sample(adjacency_list: Dict[int, List[int]], sample_size: int = 1000) -> float:
    """
    Compute average local clustering coefficient using sampling for large graphs.
    Clustering coefficient measures how connected a node's neighbors are to each other.
    """
    import random
    
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


def estimate_diameter_bfs(adjacency_list: Dict[int, List[int]], num_samples: int = 10) -> Tuple[float, float]:
    """
    Estimate graph diameter and average path length using BFS from random samples.
    Returns (diameter_estimate, avg_path_length).
    """
    import random
    from collections import deque
    
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


def count_subcommunities_quick(adjacency_list: Dict[int, List[int]], threshold: int = 100) -> int:
    """
    Quick estimate of number of connected components/communities using union-find.
    For small graphs, returns exact count. For large graphs, estimates.
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
    Returns dictionary with all features.
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


@dataclass
class ReorderResult:
    """Result from reordering a graph."""
    graph: str
    algorithm_id: int
    algorithm_name: str
    reorder_time: float
    mapping_file: str = ""
    success: bool = True
    error: str = ""

@dataclass
class BenchmarkResult:
    """Result from running a benchmark."""
    graph: str
    algorithm_id: int
    algorithm_name: str
    benchmark: str
    trial_time: float
    speedup: float = 1.0
    nodes: int = 0
    edges: int = 0
    success: bool = True
    error: str = ""

@dataclass
class CacheResult:
    """Result from cache simulation."""
    graph: str
    algorithm_id: int
    algorithm_name: str
    benchmark: str
    l1_hit_rate: float = 0.0
    l2_hit_rate: float = 0.0
    l3_hit_rate: float = 0.0
    success: bool = True
    error: str = ""

@dataclass
class ReorderResult:
    """Result from reordering/label map generation."""
    graph: str
    algorithm_id: int
    algorithm_name: str
    reorder_time: float
    mapping_file: str = ""
    success: bool = True
    error: str = ""

@dataclass
class SubcommunityInfo:
    """Information about a subcommunity in adaptive ordering."""
    community_id: int
    nodes: int
    edges: int
    density: float
    degree_variance: float
    hub_concentration: float
    selected_algorithm: str
    # New graph structure features
    clustering_coefficient: float = 0.0  # Local clustering coefficient
    avg_path_length: float = 0.0  # Average shortest path (estimated)
    diameter_estimate: float = 0.0  # BFS diameter estimate
    community_count: int = 1  # Number of sub-communities (from Leiden)
    
@dataclass
class AdaptiveOrderResult:
    """Result from adaptive ordering analysis."""
    graph: str
    modularity: float
    num_communities: int
    subcommunities: List[SubcommunityInfo] = field(default_factory=list)
    algorithm_distribution: Dict[str, int] = field(default_factory=dict)
    reorder_time: float = 0.0
    success: bool = True
    error: str = ""

@dataclass
class AdaptiveComparisonResult:
    """Result comparing adaptive vs fixed-algorithm approaches."""
    graph: str
    benchmark: str
    adaptive_time: float
    adaptive_speedup: float
    fixed_results: Dict[str, float] = field(default_factory=dict)  # algo_name -> speedup
    best_fixed_algorithm: str = ""
    best_fixed_speedup: float = 0.0
    adaptive_advantage: float = 0.0  # adaptive_speedup - best_fixed_speedup

@dataclass
class PerceptronWeight:
    """Perceptron weight for an algorithm."""
    bias: float = 0.5
    w_modularity: float = 0.0
    w_log_nodes: float = 0.0
    w_log_edges: float = 0.0
    w_density: float = 0.0
    w_avg_degree: float = 0.0
    w_degree_variance: float = 0.0
    w_hub_concentration: float = 0.0
    cache_l1_impact: float = 0.0
    cache_l2_impact: float = 0.0
    cache_l3_impact: float = 0.0
    cache_dram_penalty: float = 0.0
    w_reorder_time: float = 0.0  # weight for reorder time
    
    # NEW: Additional graph structure feature weights
    w_clustering_coeff: float = 0.0  # Local clustering effect
    w_avg_path_length: float = 0.0  # Path length sensitivity
    w_diameter: float = 0.0  # Diameter effect
    w_community_count: float = 0.0  # Sub-community count effect
    
    # NEW: Per-benchmark weight adjustments (multipliers)
    # These modify the base weights for specific benchmarks
    benchmark_weights: Dict[str, float] = field(default_factory=lambda: {
        'pr': 1.0, 'bfs': 1.0, 'cc': 1.0, 'sssp': 1.0, 'bc': 1.0
    })
    
    # Metadata
    _metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        return d
    
    def compute_score(self, features: Dict, benchmark: str = 'pr') -> float:
        """Compute perceptron score for given features and benchmark."""
        import math
        log_nodes = math.log10(features.get('nodes', 1) + 1)
        log_edges = math.log10(features.get('edges', 1) + 1)
        
        score = self.bias
        score += self.w_modularity * features.get('modularity', 0.5)
        score += self.w_log_nodes * log_nodes
        score += self.w_log_edges * log_edges
        score += self.w_density * features.get('density', 0.0)
        score += self.w_avg_degree * features.get('avg_degree', 0.0) / 100.0
        score += self.w_degree_variance * features.get('degree_variance', 0.0)
        score += self.w_hub_concentration * features.get('hub_concentration', 0.0)
        score += self.w_clustering_coeff * features.get('clustering_coefficient', 0.0)
        score += self.w_avg_path_length * features.get('avg_path_length', 0.0) / 10.0
        score += self.w_diameter * features.get('diameter', features.get('diameter_estimate', 0.0)) / 50.0  # Match C++ normalization
        score += self.w_community_count * math.log10(features.get('community_count', 1) + 1)
        score += self.w_reorder_time * features.get('reorder_time', 0.0)  # Penalty for slow reordering
        
        # Apply benchmark-specific multiplier
        bench_mult = self.benchmark_weights.get(benchmark.lower(), 1.0)
        return score * bench_mult

@dataclass
class DownloadableGraph:
    """Information about a downloadable graph."""
    name: str
    url: str
    size_mb: int
    nodes: int
    edges: int
    symmetric: bool
    category: str
    description: str = ""
    
    def estimated_memory_gb(self) -> float:
        """Estimate RAM required to process this graph."""
        memory_bytes = (self.edges * BYTES_PER_EDGE + self.nodes * BYTES_PER_NODE) * MEMORY_SAFETY_FACTOR
        return memory_bytes / (1024 ** 3)


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
    """Estimate memory required for a graph in GB."""
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
        import os
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
        import os
        stat = os.statvfs(path)
        return (stat.f_blocks * stat.f_frsize) / (1024 ** 3)
    except:
        pass
    return 500.0


# ============================================================================
# Auto-Clustering Type System
# ============================================================================

# Global type registry - maps type names to centroids and metadata
_type_registry: Dict[str, Dict] = {}
_type_registry_file = os.path.join(DEFAULT_WEIGHTS_DIR, "type_registry.json")

def _get_next_type_name() -> str:
    """Generate next type name (type_0, type_1, etc.) - numeric for unlimited types."""
    if not _type_registry:
        return "type_0"
    # Find highest existing type number
    existing = [k for k in _type_registry.keys() if k.startswith("type_")]
    if not existing:
        return "type_0"
    # Extract numbers and find max
    numbers = []
    for k in existing:
        try:
            num = int(k.replace("type_", ""))
            numbers.append(num)
        except ValueError:
            # Handle legacy alphabetic names (type_a -> 0, type_b -> 1, etc.)
            suffix = k.replace("type_", "")
            if len(suffix) == 1 and suffix.isalpha():
                numbers.append(ord(suffix.lower()) - ord('a'))
    if not numbers:
        return "type_0"
    return f"type_{max(numbers) + 1}"

def _normalize_features(features: Dict) -> List[float]:
    """Normalize features to [0,1] range for distance calculation."""
    import math
    # Feature ranges (approximate) for normalization
    ranges = {
        'modularity': (0, 1),
        'degree_variance': (0, 5),
        'hub_concentration': (0, 1),
        'avg_degree': (0, 100),
        'clustering_coefficient': (0, 1),
        'log_nodes': (3, 10),  # log10(1000) to log10(10B)
        'log_edges': (3, 12),
    }
    
    normalized = []
    log_nodes = math.log10(features.get('nodes', 1000) + 1)
    log_edges = math.log10(features.get('edges', 1000) + 1)
    
    for key, (lo, hi) in ranges.items():
        if key == 'log_nodes':
            val = log_nodes
        elif key == 'log_edges':
            val = log_edges
        else:
            val = features.get(key, (lo + hi) / 2)
        # Clamp and normalize
        normalized.append(max(0, min(1, (val - lo) / (hi - lo) if hi > lo else 0.5)))
    
    return normalized

def _compute_distance(f1: List[float], f2: List[float]) -> float:
    """Compute Euclidean distance between normalized feature vectors."""
    import math
    if len(f1) != len(f2):
        return float('inf')
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(f1, f2)))

def load_type_registry(weights_dir: str = DEFAULT_WEIGHTS_DIR) -> Dict[str, Dict]:
    """Load type registry from disk."""
    global _type_registry, _type_registry_file
    _type_registry_file = os.path.join(weights_dir, "type_registry.json")
    
    if os.path.exists(_type_registry_file):
        try:
            with open(_type_registry_file) as f:
                _type_registry = json.load(f)
        except Exception as e:
            _type_registry = {}
    return _type_registry

def save_type_registry(weights_dir: str = DEFAULT_WEIGHTS_DIR):
    """Save type registry to disk."""
    global _type_registry
    os.makedirs(weights_dir, exist_ok=True)
    registry_file = os.path.join(weights_dir, "type_registry.json")
    with open(registry_file, 'w') as f:
        json.dump(_type_registry, f, indent=2)

def get_type_weights_file(type_name: str, weights_dir: str = DEFAULT_WEIGHTS_DIR) -> str:
    """Get path to weights file for a type."""
    return os.path.join(weights_dir, f"{type_name}.json")

def load_type_weights(type_name: str, weights_dir: str = DEFAULT_WEIGHTS_DIR) -> Dict:
    """Load weights for a specific type."""
    weights_file = get_type_weights_file(type_name, weights_dir)
    if os.path.exists(weights_file):
        try:
            with open(weights_file) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_type_weights(type_name: str, weights: Dict, weights_dir: str = DEFAULT_WEIGHTS_DIR):
    """Save weights for a specific type."""
    os.makedirs(weights_dir, exist_ok=True)
    weights_file = get_type_weights_file(type_name, weights_dir)
    with open(weights_file, 'w') as f:
        json.dump(weights, f, indent=2)

def assign_graph_type(features: Dict, weights_dir: str = DEFAULT_WEIGHTS_DIR, 
                      create_if_outlier: bool = True) -> str:
    """
    Assign a graph to a type based on its features.
    
    Uses clustering to find the closest existing type or creates a new one
    if the graph is an outlier (distance > CLUSTER_DISTANCE_THRESHOLD).
    
    Args:
        features: Dict with 'modularity', 'degree_variance', 'hub_concentration', etc.
        weights_dir: Directory for type files
        create_if_outlier: If True, create new type for outliers
        
    Returns:
        Type name (e.g., 'type_0', 'type_1')
    """
    global _type_registry
    
    # Load registry if not loaded
    if not _type_registry:
        load_type_registry(weights_dir)
    
    # Normalize input features
    norm_features = _normalize_features(features)
    
    # Find closest existing type
    min_distance = float('inf')
    closest_type = None
    
    for type_name, type_info in _type_registry.items():
        if 'centroid' in type_info:
            dist = _compute_distance(norm_features, type_info['centroid'])
            if dist < min_distance:
                min_distance = dist
                closest_type = type_name
    
    # Check if close enough to existing type
    if closest_type and min_distance < CLUSTER_DISTANCE_THRESHOLD:
        # Update centroid (incremental mean)
        type_info = _type_registry[closest_type]
        count = type_info.get('sample_count', 1)
        old_centroid = type_info['centroid']
        # Incremental centroid update: new_centroid = old + (new - old) / (count + 1)
        new_centroid = [
            old + (new - old) / (count + 1) 
            for old, new in zip(old_centroid, norm_features)
        ]
        _type_registry[closest_type]['centroid'] = new_centroid
        _type_registry[closest_type]['sample_count'] = count + 1
        _type_registry[closest_type]['last_updated'] = datetime.now().isoformat()
        save_type_registry(weights_dir)
        return closest_type
    
    # Create new type if outlier
    if create_if_outlier:
        new_type = _get_next_type_name()
        _type_registry[new_type] = {
            'centroid': norm_features,
            'sample_count': 1,
            'created': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'representative_features': {
                'modularity': features.get('modularity', 0.5),
                'degree_variance': features.get('degree_variance', 1.0),
                'hub_concentration': features.get('hub_concentration', 0.3),
                'avg_degree': features.get('avg_degree', 10),
            }
        }
        save_type_registry(weights_dir)
        
        # Initialize weights for new type (copy from closest if exists, else defaults)
        if closest_type:
            existing_weights = load_type_weights(closest_type, weights_dir)
            if existing_weights:
                save_type_weights(new_type, existing_weights, weights_dir)
        
        return new_type
    
    # Return closest even if above threshold (when not creating new types)
    return closest_type or "type_0"

def update_type_weights_incremental(
    type_name: str,
    algorithm: str,
    benchmark: str,
    speedup: float,
    features: Dict,
    cache_stats: Optional[Dict] = None,
    reorder_time: float = 0.0,
    weights_dir: str = DEFAULT_WEIGHTS_DIR,
    learning_rate: float = 0.01
):
    """
    Incrementally update weights for a type after a benchmark run.
    
    Called immediately after each benchmark completes, not at the end of experiments.
    
    Args:
        type_name: The graph type (e.g., 'type_0')
        algorithm: Algorithm name (e.g., 'HUBCLUSTERDBG')
        benchmark: Benchmark name (e.g., 'bfs')
        speedup: Observed speedup vs baseline
        features: Graph features dict
        cache_stats: Optional cache simulation results
        reorder_time: Time to reorder the graph
        weights_dir: Directory for type files
        learning_rate: Learning rate for weight updates
    """
    import math
    
    weights = load_type_weights(type_name, weights_dir)
    if not weights:
        weights = {}
    
    # Initialize algorithm weights if not present
    if algorithm not in weights:
        weights[algorithm] = {
            'bias': 0.5,
            'w_modularity': 0.0,
            'w_log_nodes': 0.0,
            'w_log_edges': 0.0,
            'w_density': 0.0,
            'w_avg_degree': 0.0,
            'w_degree_variance': 0.0,
            'w_hub_concentration': 0.0,
            'cache_l1_impact': 0.0,
            'cache_l2_impact': 0.0,
            'cache_l3_impact': 0.0,
            'cache_dram_penalty': 0.0,
            'w_reorder_time': 0.0,
            'w_clustering_coeff': 0.0,
            'w_avg_path_length': 0.0,
            'w_diameter': 0.0,
            'w_community_count': 0.0,
            'benchmark_weights': {'pr': 1.0, 'bfs': 1.0, 'cc': 1.0, 'sssp': 1.0, 'bc': 1.0},
            '_metadata': {
                'sample_count': 0,
                'avg_speedup': 1.0,
                'win_count': 0,
                'times_best': 0,
                'win_rate': 0.0,
            }
        }
    
    algo_weights = weights[algorithm]
    meta = algo_weights.get('_metadata', {})
    
    # Update sample count and running average speedup
    count = meta.get('sample_count', 0) + 1
    old_avg = meta.get('avg_speedup', 1.0)
    new_avg = old_avg + (speedup - old_avg) / count
    
    meta['sample_count'] = count
    meta['avg_speedup'] = new_avg
    meta['last_updated'] = datetime.now().isoformat()
    
    # Track wins (speedup > 1.05 means this algorithm is notably better than baseline)
    if speedup > 1.05:
        meta['win_count'] = meta.get('win_count', 0) + 1
    
    # Track times_best (speedup > 1.0 means it beat the baseline)
    if speedup > 1.0:
        meta['times_best'] = meta.get('times_best', 0) + 1
    
    # Compute win_rate as percentage
    meta['win_rate'] = meta['win_count'] / count if count > 0 else 0.0
    
    # Compute error signal (positive if better than expected, negative if worse)
    current_score = algo_weights['bias']
    error = (speedup - 1.0) - current_score  # How much better/worse than expected
    
    # Gradient update on weights
    log_nodes = math.log10(features.get('nodes', 1) + 1)
    log_edges = math.log10(features.get('edges', 1) + 1)
    
    algo_weights['bias'] += learning_rate * error
    algo_weights['w_modularity'] += learning_rate * error * features.get('modularity', 0.5)
    algo_weights['w_log_nodes'] += learning_rate * error * log_nodes * 0.1
    algo_weights['w_log_edges'] += learning_rate * error * log_edges * 0.1
    algo_weights['w_density'] += learning_rate * error * features.get('density', 0.0)
    algo_weights['w_avg_degree'] += learning_rate * error * features.get('avg_degree', 0.0) / 100.0
    algo_weights['w_degree_variance'] += learning_rate * error * features.get('degree_variance', 1.0)
    algo_weights['w_hub_concentration'] += learning_rate * error * features.get('hub_concentration', 0.3)
    algo_weights['w_clustering_coeff'] += learning_rate * error * features.get('clustering_coefficient', 0.0)
    algo_weights['w_community_count'] += learning_rate * error * features.get('community_count', 0) / 1000.0  # Normalize
    
    # Extended topology features (now computed at graph load time)
    algo_weights['w_avg_path_length'] += learning_rate * error * features.get('avg_path_length', 0.0) / 10.0
    algo_weights['w_diameter'] += learning_rate * error * features.get('diameter', 0.0) / 50.0  # Normalize
    
    # Update cache weights if cache stats provided
    if cache_stats:
        l1_hit = cache_stats.get('l1_hit_rate', 0) / 100.0
        l2_hit = cache_stats.get('l2_hit_rate', 0) / 100.0
        l3_hit = cache_stats.get('l3_hit_rate', 0) / 100.0
        dram_penalty = 1.0 - (l1_hit + l2_hit * 0.3 + l3_hit * 0.1)
        
        algo_weights['cache_l1_impact'] += learning_rate * error * l1_hit
        algo_weights['cache_l2_impact'] += learning_rate * error * l2_hit
        algo_weights['cache_l3_impact'] += learning_rate * error * l3_hit
        algo_weights['cache_dram_penalty'] += learning_rate * error * dram_penalty
        
        # Track cache stats in metadata
        meta['avg_l1_hit_rate'] = meta.get('avg_l1_hit_rate', 0) + (l1_hit * 100 - meta.get('avg_l1_hit_rate', 0)) / count
        meta['avg_l2_hit_rate'] = meta.get('avg_l2_hit_rate', 0) + (l2_hit * 100 - meta.get('avg_l2_hit_rate', 0)) / count
        meta['avg_l3_hit_rate'] = meta.get('avg_l3_hit_rate', 0) + (l3_hit * 100 - meta.get('avg_l3_hit_rate', 0)) / count
    
    # Update reorder time weight
    if reorder_time > 0:
        algo_weights['w_reorder_time'] += learning_rate * error * (-reorder_time / 10.0)  # Negative impact
        meta['avg_reorder_time'] = meta.get('avg_reorder_time', 0) + (reorder_time - meta.get('avg_reorder_time', 0)) / count
    
    # Update benchmark-specific weight
    bench_weights = algo_weights.get('benchmark_weights', {})
    if benchmark.lower() in bench_weights:
        # Adjust benchmark weight based on performance
        current_bench_weight = bench_weights[benchmark.lower()]
        bench_weights[benchmark.lower()] = current_bench_weight + learning_rate * error * 0.1
    
    algo_weights['_metadata'] = meta
    algo_weights['benchmark_weights'] = bench_weights
    weights[algorithm] = algo_weights
    
    # Save immediately
    save_type_weights(type_name, weights, weights_dir)

def get_best_algorithm_for_type(
    type_name: str,
    features: Dict,
    benchmark: str = 'pr',
    weights_dir: str = DEFAULT_WEIGHTS_DIR
) -> Tuple[str, float]:
    """
    Get the best algorithm for a graph type based on learned weights.
    
    Returns:
        (algorithm_name, score) tuple
    """
    weights = load_type_weights(type_name, weights_dir)
    if not weights:
        return ("ORIGINAL", 0.0)
    
    best_algo = None
    best_score = float('-inf')
    
    for algo_name, algo_weights in weights.items():
        if algo_name.startswith('_'):  # Skip metadata
            continue
        
        # Create PerceptronWeight and compute score
        pw = PerceptronWeight(
            bias=algo_weights.get('bias', 0.5),
            w_modularity=algo_weights.get('w_modularity', 0.0),
            w_log_nodes=algo_weights.get('w_log_nodes', 0.0),
            w_log_edges=algo_weights.get('w_log_edges', 0.0),
            w_density=algo_weights.get('w_density', 0.0),
            w_avg_degree=algo_weights.get('w_avg_degree', 0.0),
            w_degree_variance=algo_weights.get('w_degree_variance', 0.0),
            w_hub_concentration=algo_weights.get('w_hub_concentration', 0.0),
            cache_l1_impact=algo_weights.get('cache_l1_impact', 0.0),
            cache_l2_impact=algo_weights.get('cache_l2_impact', 0.0),
            cache_l3_impact=algo_weights.get('cache_l3_impact', 0.0),
            cache_dram_penalty=algo_weights.get('cache_dram_penalty', 0.0),
            w_reorder_time=algo_weights.get('w_reorder_time', 0.0),
            w_clustering_coeff=algo_weights.get('w_clustering_coeff', 0.0),
            w_avg_path_length=algo_weights.get('w_avg_path_length', 0.0),
            w_diameter=algo_weights.get('w_diameter', 0.0),
            w_community_count=algo_weights.get('w_community_count', 0.0),
            benchmark_weights=algo_weights.get('benchmark_weights', {})
        )
        
        score = pw.compute_score(features, benchmark)
        
        # Add confidence boost based on sample count
        meta = algo_weights.get('_metadata', {})
        sample_count = meta.get('sample_count', 0)
        confidence_boost = min(0.1, sample_count * 0.01)  # Max 0.1 boost for 10+ samples
        score += confidence_boost
        
        if score > best_score:
            best_score = score
            best_algo = algo_name
    
    return (best_algo or "ORIGINAL", best_score)

def list_known_types(weights_dir: str = DEFAULT_WEIGHTS_DIR) -> List[str]:
    """List all known graph types."""
    load_type_registry(weights_dir)
    return list(_type_registry.keys())

def get_type_summary(type_name: str, weights_dir: str = DEFAULT_WEIGHTS_DIR) -> Dict:
    """Get summary information about a type."""
    load_type_registry(weights_dir)
    type_info = _type_registry.get(type_name, {})
    weights = load_type_weights(type_name, weights_dir)
    
    # Count algorithms and their performance
    algo_stats = {}
    best_algos = {}
    for algo_name, algo_weights in weights.items():
        if algo_name.startswith('_'):
            continue
        meta = algo_weights.get('_metadata', {})
        algo_stats[algo_name] = {
            'sample_count': meta.get('sample_count', 0),
            'avg_speedup': meta.get('avg_speedup', 1.0),
            'win_count': meta.get('win_count', 0),
        }
        # Track best algorithm per benchmark
        for bench, weight in algo_weights.get('benchmark_weights', {}).items():
            if bench not in best_algos or weight > best_algos[bench][1]:
                best_algos[bench] = (algo_name, weight)
    
    return {
        'type_name': type_name,
        'num_graphs': type_info.get('sample_count', 0),  # Renamed for clarity in display
        'sample_count': type_info.get('sample_count', 0),
        'created': type_info.get('created', 'unknown'),
        'last_updated': type_info.get('last_updated', 'unknown'),
        'centroid': type_info.get('centroid', []),
        'representative_features': type_info.get('representative_features', {}),
        'algorithms': algo_stats,
        'best_algorithms': {bench: algo for bench, (algo, _) in best_algos.items()},
    }


# ============================================================================
# Graph Catalog for Download
# ============================================================================

# Small graphs (< 20MB, good for quick testing)
DOWNLOAD_GRAPHS_SMALL = [
    # Communication networks
    DownloadableGraph("email-Enron", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/email-Enron.tar.gz",
                      5, 36692, 183831, True, "communication", "Enron email network"),
    DownloadableGraph("email-EuAll", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/email-EuAll.tar.gz",
                      4, 265214, 420045, False, "communication", "EU email network"),
    # Collaboration networks
    DownloadableGraph("ca-AstroPh", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/ca-AstroPh.tar.gz",
                      3, 18772, 198110, True, "collaboration", "Arxiv Astro Physics"),
    DownloadableGraph("ca-CondMat", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/ca-CondMat.tar.gz",
                      2, 23133, 93497, True, "collaboration", "Condensed Matter"),
    DownloadableGraph("ca-GrQc", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/ca-GrQc.tar.gz",
                      1, 5242, 14496, True, "collaboration", "General Relativity"),
    DownloadableGraph("ca-HepPh", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/ca-HepPh.tar.gz",
                      2, 12008, 118521, True, "collaboration", "High Energy Physics"),
    DownloadableGraph("ca-HepTh", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/ca-HepTh.tar.gz",
                      1, 9877, 25998, True, "collaboration", "High Energy Physics Theory"),
    # P2P networks
    DownloadableGraph("p2p-Gnutella31", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/p2p-Gnutella31.tar.gz",
                      2, 62586, 147892, False, "p2p", "Gnutella P2P network"),
    DownloadableGraph("p2p-Gnutella30", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/p2p-Gnutella30.tar.gz",
                      1, 36682, 88328, False, "p2p", "Gnutella P2P Aug 30"),
    DownloadableGraph("p2p-Gnutella25", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/p2p-Gnutella25.tar.gz",
                      1, 22687, 54705, False, "p2p", "Gnutella P2P Aug 25"),
    DownloadableGraph("p2p-Gnutella24", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/p2p-Gnutella24.tar.gz",
                      1, 26518, 65369, False, "p2p", "Gnutella P2P Aug 24"),
    # Social networks (small)
    DownloadableGraph("soc-Slashdot0811", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/soc-Slashdot0811.tar.gz",
                      8, 77360, 905468, False, "social", "Slashdot Nov 2008"),
    DownloadableGraph("soc-Slashdot0902", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/soc-Slashdot0902.tar.gz",
                      9, 82168, 948464, False, "social", "Slashdot Feb 2009"),
    DownloadableGraph("soc-sign-epinions", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/soc-sign-epinions.tar.gz",
                      10, 131828, 841372, False, "social", "Epinions signed network"),
    # Citation networks (small)
    DownloadableGraph("cit-HepPh", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/cit-HepPh.tar.gz",
                      8, 34546, 421578, False, "citation", "HEP-PH citations"),
    DownloadableGraph("cit-HepTh", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/cit-HepTh.tar.gz",
                      4, 27770, 352807, False, "citation", "HEP-TH citations"),
]

# Medium graphs (20MB - 200MB) - ~35 graphs
DOWNLOAD_GRAPHS_MEDIUM = [
    # Communication
    DownloadableGraph("wiki-Talk", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/wiki-Talk.tar.gz",
                      80, 2394385, 5021410, False, "communication", "Wikipedia talk"),
    DownloadableGraph("wiki-topcats", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/wiki-topcats.tar.gz",
                      120, 1791489, 28511807, False, "web", "Wikipedia top categories"),
    # Citation networks
    DownloadableGraph("cit-Patents", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/cit-Patents.tar.gz",
                      262, 3774768, 16518948, False, "citation", "US Patent citations"),
    # Road networks
    DownloadableGraph("roadNet-PA", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/roadNet-PA.tar.gz",
                      40, 1090920, 1541898, True, "road", "Pennsylvania roads"),
    DownloadableGraph("roadNet-CA", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/roadNet-CA.tar.gz",
                      60, 1971281, 2766607, True, "road", "California roads"),
    DownloadableGraph("roadNet-TX", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/roadNet-TX.tar.gz",
                      45, 1393383, 1921660, True, "road", "Texas roads"),
    # Social networks (medium)
    DownloadableGraph("soc-Epinions1", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/soc-Epinions1.tar.gz",
                      12, 75888, 508837, False, "social", "Epinions social"),
    # Commerce networks
    DownloadableGraph("amazon0302", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/amazon0302.tar.gz",
                      15, 262111, 1234877, False, "commerce", "Amazon Mar 2003"),
    DownloadableGraph("amazon0312", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/amazon0312.tar.gz",
                      18, 400727, 3200440, False, "commerce", "Amazon Dec 2003"),
    DownloadableGraph("amazon0505", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/amazon0505.tar.gz",
                      22, 410236, 3356824, False, "commerce", "Amazon May 2005"),
    DownloadableGraph("amazon0601", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/amazon0601.tar.gz",
                      25, 403394, 3387388, False, "commerce", "Amazon Jun 2001"),
    # Web graphs
    DownloadableGraph("web-NotreDame", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/web-NotreDame.tar.gz",
                      15, 325729, 1497134, False, "web", "Notre Dame web"),
    DownloadableGraph("web-Stanford", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/web-Stanford.tar.gz",
                      20, 281903, 2312497, False, "web", "Stanford web"),
    DownloadableGraph("web-BerkStan", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/web-BerkStan.tar.gz",
                      50, 685230, 7600595, False, "web", "Berkeley-Stanford web"),
    DownloadableGraph("web-Google", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/web-Google.tar.gz",
                      35, 916428, 5105039, False, "web", "Google web graph"),
    # Infrastructure
    DownloadableGraph("as-Skitter", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/as-Skitter.tar.gz",
                      90, 1696415, 11095298, True, "infrastructure", "Internet topology"),
    # Autonomous systems
    DownloadableGraph("Oregon-1", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/Oregon-1.tar.gz",
                      1, 11174, 23409, False, "infrastructure", "Oregon AS peering"),
    DownloadableGraph("Oregon-2", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/Oregon-2.tar.gz",
                      1, 11461, 32730, False, "infrastructure", "Oregon AS peering 2"),
    # DIMACS10 graphs (sparse)
    DownloadableGraph("delaunay_n17", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/delaunay_n17.tar.gz",
                      5, 131072, 393176, True, "mesh", "Delaunay triangulation n=17"),
    DownloadableGraph("delaunay_n18", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/delaunay_n18.tar.gz",
                      10, 262144, 786396, True, "mesh", "Delaunay triangulation n=18"),
    DownloadableGraph("delaunay_n19", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/delaunay_n19.tar.gz",
                      20, 524288, 1572823, True, "mesh", "Delaunay triangulation n=19"),
    DownloadableGraph("delaunay_n20", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/delaunay_n20.tar.gz",
                      40, 1048576, 3145686, True, "mesh", "Delaunay triangulation n=20"),
    DownloadableGraph("rgg_n_2_17_s0", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/rgg_n_2_17_s0.tar.gz",
                      15, 131072, 728753, True, "mesh", "Random geometric graph n=17"),
    DownloadableGraph("rgg_n_2_18_s0", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/rgg_n_2_18_s0.tar.gz",
                      30, 262144, 1457506, True, "mesh", "Random geometric graph n=18"),
    DownloadableGraph("rgg_n_2_19_s0", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/rgg_n_2_19_s0.tar.gz",
                      60, 524288, 2915013, True, "mesh", "Random geometric graph n=19"),
    # Power-law and scale-free
    DownloadableGraph("preferentialAttachment", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/preferentialAttachment.tar.gz",
                      10, 100000, 499985, True, "synthetic", "Preferential attachment model"),
    DownloadableGraph("smallworld", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/smallworld.tar.gz",
                      10, 100000, 499998, True, "synthetic", "Small world model"),
    # Additional web graphs
    DownloadableGraph("cnr-2000", "https://suitesparse-collection-website.herokuapp.com/MM/LAW/cnr-2000.tar.gz",
                      30, 325557, 3216152, False, "web", "Italian CNR web 2000"),
]

# Large graphs (200MB - 2GB) - ~40 graphs  
DOWNLOAD_GRAPHS_LARGE = [
    # Social networks (large)
    DownloadableGraph("soc-LiveJournal1", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/soc-LiveJournal1.tar.gz",
                      1024, 4847571, 68993773, False, "social", "LiveJournal social"),
    DownloadableGraph("com-Orkut", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/com-Orkut.tar.gz",
                      800, 3072441, 117185083, True, "social", "Orkut social network"),
    DownloadableGraph("com-Youtube", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/com-Youtube.tar.gz",
                      250, 1134890, 2987624, True, "social", "Youtube social network"),
    DownloadableGraph("com-Amazon", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/com-Amazon.tar.gz",
                      220, 334863, 925872, True, "commerce", "Amazon product network"),
    DownloadableGraph("com-DBLP", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/com-DBLP.tar.gz",
                      200, 317080, 1049866, True, "collaboration", "DBLP collaboration"),
    # Collaboration
    DownloadableGraph("hollywood-2009", "https://suitesparse-collection-website.herokuapp.com/MM/LAW/hollywood-2009.tar.gz",
                      600, 1139905, 57515616, True, "collaboration", "Hollywood actors"),
    DownloadableGraph("dblp-2010", "https://suitesparse-collection-website.herokuapp.com/MM/LAW/dblp-2010.tar.gz",
                      200, 326186, 1615400, True, "collaboration", "DBLP 2010"),
    # Web graphs (large)
    DownloadableGraph("in-2004", "https://suitesparse-collection-website.herokuapp.com/MM/LAW/in-2004.tar.gz",
                      450, 1382908, 16917053, False, "web", "Indian web 2004"),
    DownloadableGraph("eu-2005", "https://suitesparse-collection-website.herokuapp.com/MM/LAW/eu-2005.tar.gz",
                      500, 862664, 19235140, False, "web", "European web 2005"),
    DownloadableGraph("uk-2002", "https://suitesparse-collection-website.herokuapp.com/MM/LAW/uk-2002.tar.gz",
                      2500, 18520486, 298113762, False, "web", "UK web 2002"),
    DownloadableGraph("arabic-2005", "https://suitesparse-collection-website.herokuapp.com/MM/LAW/arabic-2005.tar.gz",
                      2200, 22744080, 639999458, False, "web", "Arabic web 2005"),
    DownloadableGraph("indochina-2004", "https://suitesparse-collection-website.herokuapp.com/MM/LAW/indochina-2004.tar.gz",
                      600, 7414866, 194109311, False, "web", "Indochina web 2004"),
    DownloadableGraph("sk-2005", "https://suitesparse-collection-website.herokuapp.com/MM/LAW/sk-2005.tar.gz",
                      1100, 50636154, 1949412601, False, "web", "Slovakia web 2005"),
    # Road networks (large) - OSM graphs extract with subdirectory structure
    DownloadableGraph("europe-osm", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/europe_osm.tar.gz",
                      1200, 50912018, 108109320, True, "road", "European OSM roads"),
    DownloadableGraph("asia-osm", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/asia_osm.tar.gz",
                      600, 11950757, 25423206, True, "road", "Asian OSM roads"),
    DownloadableGraph("great-britain-osm", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/great-britain_osm.tar.gz",
                      250, 7733822, 16313034, True, "road", "Great Britain OSM roads"),
    DownloadableGraph("germany-osm", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/germany_osm.tar.gz",
                      300, 11548845, 24738362, True, "road", "Germany OSM roads"),
    # DIMACS10 large meshes
    DownloadableGraph("delaunay_n21", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/delaunay_n21.tar.gz",
                      80, 2097152, 6291372, True, "mesh", "Delaunay triangulation n=21"),
    DownloadableGraph("delaunay_n22", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/delaunay_n22.tar.gz",
                      160, 4194304, 12582869, True, "mesh", "Delaunay triangulation n=22"),
    DownloadableGraph("delaunay_n23", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/delaunay_n23.tar.gz",
                      320, 8388608, 25165784, True, "mesh", "Delaunay triangulation n=23"),
    DownloadableGraph("delaunay_n24", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/delaunay_n24.tar.gz",
                      640, 16777216, 50331601, True, "mesh", "Delaunay triangulation n=24"),
    DownloadableGraph("rgg_n_2_20_s0", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/rgg_n_2_20_s0.tar.gz",
                      120, 1048576, 5830030, True, "mesh", "Random geometric graph n=20"),
    DownloadableGraph("rgg_n_2_21_s0", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/rgg_n_2_21_s0.tar.gz",
                      240, 2097152, 11660061, True, "mesh", "Random geometric graph n=21"),
    DownloadableGraph("rgg_n_2_22_s0", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/rgg_n_2_22_s0.tar.gz",
                      480, 4194304, 23320130, True, "mesh", "Random geometric graph n=22"),
    DownloadableGraph("rgg_n_2_23_s0", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/rgg_n_2_23_s0.tar.gz",
                      960, 8388608, 46640257, True, "mesh", "Random geometric graph n=23"),
    DownloadableGraph("rgg_n_2_24_s0", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/rgg_n_2_24_s0.tar.gz",
                      1920, 16777216, 93280513, True, "mesh", "Random geometric graph n=24"),
    # Clustering benchmarks
    DownloadableGraph("coPapersDBLP", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/coPapersDBLP.tar.gz",
                      400, 540486, 15245729, True, "collaboration", "DBLP co-author papers"),
    DownloadableGraph("coPapersCiteseer", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/coPapersCiteseer.tar.gz",
                      450, 434102, 16036720, True, "citation", "Citeseer co-papers"),
    DownloadableGraph("citationCiteseer", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/citationCiteseer.tar.gz",
                      350, 268495, 1156647, False, "citation", "Citeseer citations"),
    DownloadableGraph("coAuthorsDBLP", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/coAuthorsDBLP.tar.gz",
                      200, 299067, 977676, True, "collaboration", "DBLP co-authors"),
    DownloadableGraph("coAuthorsCiteseer", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/coAuthorsCiteseer.tar.gz",
                      160, 227320, 814134, True, "collaboration", "Citeseer co-authors"),
    # Wikipedia graphs
    DownloadableGraph("wiki-Vote", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/wiki-Vote.tar.gz",
                      2, 7115, 103689, False, "social", "Wikipedia adminship votes"),
    # Kron graphs (synthetic power-law)
    DownloadableGraph("kron_g500-logn16", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/kron_g500-logn16.tar.gz",
                      200, 65536, 4912201, True, "synthetic", "Kronecker graph logn=16"),
    DownloadableGraph("kron_g500-logn17", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/kron_g500-logn17.tar.gz",
                      400, 131072, 10228360, True, "synthetic", "Kronecker graph logn=17"),
    DownloadableGraph("kron_g500-logn18", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/kron_g500-logn18.tar.gz",
                      800, 262144, 21165908, True, "synthetic", "Kronecker graph logn=18"),
    DownloadableGraph("kron_g500-logn19", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/kron_g500-logn19.tar.gz",
                      1600, 524288, 43561574, True, "synthetic", "Kronecker graph logn=19"),
    DownloadableGraph("kron_g500-logn20", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/kron_g500-logn20.tar.gz",
                      3200, 1048576, 89239674, True, "synthetic", "Kronecker graph logn=20"),
]

# Extra-large graphs (>2GB) - requires significant memory
DOWNLOAD_GRAPHS_XLARGE = [
    # Massive web graphs
    DownloadableGraph("uk-2005", "https://suitesparse-collection-website.herokuapp.com/MM/LAW/uk-2005.tar.gz",
                      3200, 39459925, 936364282, False, "web", "UK web 2005"),
    DownloadableGraph("webbase-2001", "https://suitesparse-collection-website.herokuapp.com/MM/LAW/webbase-2001.tar.gz",
                      8500, 118142155, 1019903190, False, "web", "WebBase 2001 crawl"),
    DownloadableGraph("it-2004", "https://suitesparse-collection-website.herokuapp.com/MM/LAW/it-2004.tar.gz",
                      3500, 41291594, 1150725436, False, "web", "Italian web 2004"),
    # Massive social
    DownloadableGraph("com-Friendster", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/com-Friendster.tar.gz",
                      31000, 65608366, 1806067135, True, "social", "Friendster social network"),
    DownloadableGraph("twitter7", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/twitter7.tar.gz",
                      12000, 41652230, 1468365182, False, "social", "Twitter follower network"),
    # Large meshes
    DownloadableGraph("kron_g500-logn21", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/kron_g500-logn21.tar.gz",
                      6400, 2097152, 182081864, True, "synthetic", "Kronecker graph logn=21"),
]

# ============================================================================
# Utility Functions
# ============================================================================

def log(msg: str, level: str = "INFO"):
    """Print a timestamped log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")

def log_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70 + "\n")

def backup_and_sync_weights(results_weights_path: str, scripts_weights_path: str = "./scripts/perceptron_weights.json"):
    """
    Backup weights with timestamp in results folder and sync to scripts folder.
    
    This ensures:
    1. Results weights are backed up with timestamp (won't be overwritten)
    2. Scripts weights are updated for next experiment iteration
    """
    if not os.path.exists(results_weights_path):
        log(f"Weights file not found: {results_weights_path}", "WARNING")
        return
    
    # Create timestamped backup in results folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = results_weights_path.replace(".json", f"_{timestamp}.json")
    
    import shutil
    shutil.copy2(results_weights_path, backup_path)
    log(f"Weights backed up to: {backup_path}")
    
    # Copy to scripts folder for next iteration
    os.makedirs(os.path.dirname(scripts_weights_path), exist_ok=True)
    shutil.copy2(results_weights_path, scripts_weights_path)
    log(f"Weights synced to: {scripts_weights_path}")

def get_graph_path(graphs_dir: str, graph_name: str) -> Optional[str]:
    """Get the path to a graph file."""
    graph_folder = os.path.join(graphs_dir, graph_name)
    
    # Check for graph files with the graph name (downloaded format)
    for ext in [".mtx", ".el", ".sg"]:
        path = os.path.join(graph_folder, f"{graph_name}{ext}")
        if os.path.exists(path):
            return path
    
    # Check for generic "graph" name (legacy format)
    for ext in [".mtx", ".el", ".sg"]:
        path = os.path.join(graph_folder, f"graph{ext}")
        if os.path.exists(path):
            return path
    
    # Try direct path (file directly in graphs_dir)
    direct = os.path.join(graphs_dir, graph_name)
    if os.path.isfile(direct):
        return direct
    
    # Look for any .mtx file in the folder
    if os.path.isdir(graph_folder):
        for f in os.listdir(graph_folder):
            if f.endswith('.mtx') and not f.endswith('_nodename.mtx') and not f.endswith('_Categories.mtx'):
                return os.path.join(graph_folder, f)
    
    return None

def get_graph_size_mb(path: str) -> float:
    """Get the size of a graph file in MB."""
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0.0


def get_graph_dimensions(path: str) -> Tuple[int, int]:
    """Read nodes and edges count from an MTX file header.
    
    Returns:
        (nodes, edges) tuple, or (0, 0) if unable to read
    """
    try:
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('%'):
                    continue
                # First non-comment line has dimensions: rows cols nnz
                parts = line.strip().split()
                if len(parts) >= 3:
                    nodes = int(parts[0])
                    edges = int(parts[2])
                    return nodes, edges
                elif len(parts) >= 2:
                    nodes = int(parts[0])
                    return nodes, 0
                break
    except:
        pass
    return 0, 0


def discover_graphs(graphs_dir: str, min_size: float = 0, max_size: float = float('inf'), 
                    additional_dirs: List[str] = None, max_memory_gb: float = None) -> List[GraphInfo]:
    """Discover available graphs in the directory and additional directories.
    
    Args:
        graphs_dir: Primary directory to scan for graphs
        min_size: Minimum graph size in MB
        max_size: Maximum graph size in MB
        additional_dirs: Additional directories to scan (e.g., ./graphs for pre-existing graphs)
        max_memory_gb: If set, skip graphs requiring more than this memory (auto-detects if None)
    """
    graphs = []
    seen_names = set()
    skipped_memory = []
    
    # Build list of directories to scan
    dirs_to_scan = [graphs_dir]
    if additional_dirs:
        dirs_to_scan.extend(additional_dirs)
    
    # Also check ./graphs if it exists and isn't already in the list
    legacy_graphs_dir = "./graphs"
    if os.path.exists(legacy_graphs_dir) and legacy_graphs_dir not in dirs_to_scan:
        dirs_to_scan.append(legacy_graphs_dir)
    
    for scan_dir in dirs_to_scan:
        if not os.path.exists(scan_dir):
            continue
            
        # Check graphs.json for metadata
        metadata_file = os.path.join(scan_dir, "graphs.json")
        metadata = {}
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
            except:
                pass
        
        # Scan directory
        try:
            entries = os.listdir(scan_dir)
        except OSError:
            continue
            
        for entry in entries:
            # Skip if already seen (first occurrence wins)
            if entry in seen_names:
                continue
                
            entry_path = os.path.join(scan_dir, entry)
            if os.path.isdir(entry_path):
                graph_path = get_graph_path(scan_dir, entry)
                if graph_path:
                    size_mb = get_graph_size_mb(graph_path)
                    if min_size <= size_mb <= max_size:
                        is_symmetric = metadata.get(entry, {}).get("symmetric", True)
                        
                        # Read actual node/edge counts from MTX file
                        nodes, edges = get_graph_dimensions(graph_path)
                        
                        # Check memory requirements if limit specified
                        if max_memory_gb is not None and nodes > 0 and edges > 0:
                            mem_required = estimate_graph_memory_gb(nodes, edges)
                            if mem_required > max_memory_gb:
                                skipped_memory.append((entry, mem_required))
                                continue
                        
                        graphs.append(GraphInfo(
                            name=entry,
                            path=graph_path,
                            size_mb=size_mb,
                            is_symmetric=is_symmetric,
                            nodes=nodes,
                            edges=edges
                        ))
                        seen_names.add(entry)
    
    # Report skipped graphs
    if skipped_memory:
        log(f"Skipped {len(skipped_memory)} graphs exceeding {max_memory_gb:.1f}GB memory limit:", "INFO")
        for name, mem in skipped_memory[:5]:
            log(f"  - {name}: requires {mem:.1f} GB", "INFO")
        if len(skipped_memory) > 5:
            log(f"  ... and {len(skipped_memory) - 5} more", "INFO")
    
    # Sort by size
    graphs.sort(key=lambda g: g.size_mb)
    return graphs

# ============================================================================
# Graph Download Functions
# ============================================================================

def download_file(url: str, dest_path: str, show_progress: bool = True) -> bool:
    """Download a file from URL with optional progress indicator."""
    try:
        import urllib.request
        import urllib.error
        
        # Create parent directory if needed
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        if show_progress:
            print(f"  Downloading from {url}")
        
        # Get file size
        try:
            with urllib.request.urlopen(url) as response:
                total_size = int(response.headers.get('content-length', 0))
                
                # Download with progress
                downloaded = 0
                chunk_size = 8192
                
                with open(dest_path, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if show_progress and total_size > 0:
                            percent = downloaded * 100 // total_size
                            print(f"\r  Progress: {percent}% ({downloaded // 1024 // 1024}MB / {total_size // 1024 // 1024}MB)", end="")
                
                if show_progress:
                    print()  # New line after progress
                    
        except urllib.error.URLError as e:
            print(f"  Error downloading: {e}")
            return False
            
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        return False

def extract_archive(archive_path: str, extract_to: str) -> bool:
    """Extract a tar.gz archive."""
    try:
        import tarfile
        import gzip
        
        print(f"  Extracting {os.path.basename(archive_path)}")
        
        if archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(path=extract_to)
        elif archive_path.endswith('.gz'):
            # Single gzip file
            out_path = archive_path[:-3]  # Remove .gz
            with gzip.open(archive_path, 'rb') as f_in:
                with open(out_path, 'wb') as f_out:
                    f_out.write(f_in.read())
        else:
            print(f"  Unknown archive format: {archive_path}")
            return False
            
        return True
        
    except Exception as e:
        print(f"  Error extracting: {e}")
        return False

def validate_mtx_file(mtx_path: str) -> bool:
    """Validate that a .mtx file is readable and has the expected format."""
    try:
        with open(mtx_path, 'r') as f:
            # Check header
            for line in f:
                if line.startswith('%'):
                    continue
                # First non-comment line should have dimensions
                parts = line.strip().split()
                if len(parts) >= 2:
                    rows = int(parts[0])
                    cols = int(parts[1])
                    if rows > 0 and cols > 0:
                        return True
                break
        return False
    except Exception as e:
        print(f"  Validation error: {e}")
        return False

def download_graph(graph: DownloadableGraph, graphs_dir: str, force: bool = False) -> bool:
    """Download a single graph from SuiteSparse collection."""
    
    graph_dir = os.path.join(graphs_dir, graph.name)
    mtx_file = os.path.join(graph_dir, f"{graph.name}.mtx")
    
    # Check if already exists
    if os.path.exists(mtx_file) and not force:
        print(f"  {graph.name}: Already exists (skip)")
        return True
    
    print(f"\n  Downloading {graph.name} ({graph.size_mb:.1f} MB)")
    
    # Create directory
    os.makedirs(graph_dir, exist_ok=True)
    
    # Use the pre-defined URL from the graph catalog
    url = graph.url
    
    # Download archive
    archive_path = os.path.join(graph_dir, f"{graph.name}.tar.gz")
    if not download_file(url, archive_path):
        return False
    
    # Extract archive
    if not extract_archive(archive_path, graph_dir):
        return False
    
    # The archive extracts to a subdirectory, move files up if needed
    extracted_dir = os.path.join(graph_dir, graph.name)
    if os.path.isdir(extracted_dir):
        for item in os.listdir(extracted_dir):
            src = os.path.join(extracted_dir, item)
            dst = os.path.join(graph_dir, item)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        # Remove empty extracted directory
        try:
            os.rmdir(extracted_dir)
        except:
            pass
    
    # Clean up archive
    try:
        os.remove(archive_path)
    except:
        pass
    
    # Validate
    if not os.path.exists(mtx_file):
        print(f"  Error: {graph.name}.mtx not found after extraction")
        return False
    
    if not validate_mtx_file(mtx_file):
        print(f"  Error: {graph.name}.mtx is invalid")
        return False
    
    print(f"  {graph.name}: Downloaded and validated successfully")
    return True

def download_graphs(
    size_category: str = "SMALL",
    graphs_dir: str = DEFAULT_GRAPHS_DIR,
    force: bool = False,
    max_memory_gb: float = None,
    max_disk_gb: float = None
) -> List[str]:
    """Download graphs by size category with optional memory and disk filtering.
    
    Args:
        size_category: One of "SMALL", "MEDIUM", "LARGE", "XLARGE", "ALL"
        graphs_dir: Directory to download graphs to
        force: If True, re-download existing graphs
        max_memory_gb: If set, skip graphs exceeding this memory requirement
        max_disk_gb: If set, skip downloads that would exceed this disk space
        
    Returns:
        List of successfully downloaded graph names
    """
    print("\n" + "="*60)
    print("GRAPH DOWNLOAD")
    print("="*60)
    
    # Select graphs based on category
    graphs_to_download = []
    
    if size_category.upper() in ["SMALL", "ALL"]:
        graphs_to_download.extend(DOWNLOAD_GRAPHS_SMALL)
    if size_category.upper() in ["MEDIUM", "ALL"]:
        graphs_to_download.extend(DOWNLOAD_GRAPHS_MEDIUM)
    if size_category.upper() in ["LARGE", "ALL"]:
        graphs_to_download.extend(DOWNLOAD_GRAPHS_LARGE)
    if size_category.upper() in ["XLARGE", "ALL"]:
        graphs_to_download.extend(DOWNLOAD_GRAPHS_XLARGE)
    
    if not graphs_to_download:
        print(f"Unknown size category: {size_category}")
        print("Valid options: SMALL, MEDIUM, LARGE, XLARGE, ALL")
        return []
    
    # Apply memory filtering if specified
    skipped_memory = []
    if max_memory_gb is not None:
        original_count = len(graphs_to_download)
        filtered = []
        for g in graphs_to_download:
            mem_required = g.estimated_memory_gb()
            if mem_required <= max_memory_gb:
                filtered.append(g)
            else:
                skipped_memory.append((g.name, mem_required))
        graphs_to_download = filtered
        if skipped_memory:
            print(f"Memory limit: {max_memory_gb:.1f} GB")
            print(f"Skipped {len(skipped_memory)} graphs exceeding memory limit:")
            for name, mem in skipped_memory[:5]:
                print(f"  - {name}: requires {mem:.1f} GB")
            if len(skipped_memory) > 5:
                print(f"  ... and {len(skipped_memory) - 5} more")
    
    # Apply disk space filtering if specified
    skipped_disk = []
    if max_disk_gb is not None:
        # Sort by size to download smaller graphs first
        graphs_to_download.sort(key=lambda g: g.size_mb)
        filtered = []
        cumulative_size_gb = 0
        for g in graphs_to_download:
            size_gb = g.size_mb / 1024
            if cumulative_size_gb + size_gb <= max_disk_gb:
                filtered.append(g)
                cumulative_size_gb += size_gb
            else:
                skipped_disk.append((g.name, g.size_mb))
        graphs_to_download = filtered
        if skipped_disk:
            print(f"Disk space limit: {max_disk_gb:.1f} GB")
            print(f"Skipped {len(skipped_disk)} graphs due to disk space limit:")
            for name, size in skipped_disk[:5]:
                print(f"  - {name}: {size:.1f} MB")
            if len(skipped_disk) > 5:
                print(f"  ... and {len(skipped_disk) - 5} more")
    
    print(f"Category: {size_category}")
    print(f"Graphs to download: {len(graphs_to_download)}")
    print(f"Target directory: {graphs_dir}")
    
    total_size = sum(g.size_mb for g in graphs_to_download)
    print(f"Total estimated download size: {total_size:.1f} MB")
    
    total_memory = sum(g.estimated_memory_gb() for g in graphs_to_download)
    print(f"Max graph memory requirement: {max(g.estimated_memory_gb() for g in graphs_to_download):.1f} GB")
    
    # Create graphs directory
    os.makedirs(graphs_dir, exist_ok=True)
    
    # Download each graph
    successful = []
    failed = []
    
    for i, graph in enumerate(graphs_to_download, 1):
        print(f"\n[{i}/{len(graphs_to_download)}] {graph.name}")
        
        if download_graph(graph, graphs_dir, force):
            successful.append(graph.name)
        else:
            failed.append(graph.name)
    
    # Summary
    print("\n" + "-"*40)
    print(f"Download complete: {len(successful)}/{len(graphs_to_download)} successful")
    
    if failed:
        print(f"Failed: {', '.join(failed)}")
    
    return successful


def ensure_prerequisites(project_dir: str = ".", 
                         graphs_dir: str = DEFAULT_GRAPHS_DIR,
                         results_dir: str = DEFAULT_RESULTS_DIR,
                         weights_dir: str = DEFAULT_WEIGHTS_DIR,
                         rebuild: bool = False) -> bool:
    """
    Ensure all prerequisites are in place: directories, binaries, weights folder.
    
    This function is called at the start of any operation to ensure the environment
    is properly set up. If any component is missing, it attempts to create/build it.
    
    Args:
        project_dir: Root project directory
        graphs_dir: Directory for graph files
        results_dir: Directory for results
        weights_dir: Directory for type-based weight files
        rebuild: If True, force rebuild even if binaries exist
        
    Returns:
        True if all prerequisites are satisfied, False otherwise
    """
    log_section("Ensuring Prerequisites")
    success = True
    
    # 1. Ensure directories exist
    log("Checking directories...")
    required_dirs = [
        graphs_dir,
        results_dir,
        weights_dir,
        os.path.join(project_dir, "bench", "bin"),
        os.path.join(project_dir, "bench", "bin_sim"),
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            log(f"  Creating: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
        else:
            log(f"  OK: {dir_path}")
    
    # 2. Check and build binaries
    log("Checking binaries...")
    bin_dir = os.path.join(project_dir, "bench", "bin")
    bin_sim_dir = os.path.join(project_dir, "bench", "bin_sim")
    
    required_bins = ["bfs", "pr", "cc", "bc", "sssp", "tc"]
    
    # Check standard binaries
    missing_bins = []
    for binary in required_bins:
        bin_path = os.path.join(bin_dir, binary)
        if not os.path.exists(bin_path) or rebuild:
            missing_bins.append(binary)
    
    # Check simulation binaries
    missing_sim = []
    for binary in required_bins:
        sim_path = os.path.join(bin_sim_dir, binary)
        if not os.path.exists(sim_path) or rebuild:
            missing_sim.append(binary)
    
    # Build if needed
    if missing_bins or missing_sim or rebuild:
        makefile = os.path.join(project_dir, "Makefile")
        if not os.path.exists(makefile):
            log("ERROR: Makefile not found - cannot build binaries", "ERROR")
            return False
        
        if missing_bins or rebuild:
            log(f"  Building standard binaries: {', '.join(missing_bins) if missing_bins else 'all (rebuild requested)'}...")
            cmd = f"cd {project_dir} && make clean-bin && make -j$(nproc)" if rebuild else f"cd {project_dir} && make -j$(nproc)"
            success_build, stdout, stderr = run_command(cmd, timeout=600)
            if not success_build:
                log(f"  Build failed: {stderr[:200]}", "ERROR")
                success = False
            else:
                log("  Standard binaries built successfully")
        
        if missing_sim or rebuild:
            log(f"  Building simulation binaries: {', '.join(missing_sim) if missing_sim else 'all (rebuild requested)'}...")
            cmd = f"cd {project_dir} && make clean-sim && make all-sim -j$(nproc)" if rebuild else f"cd {project_dir} && make all-sim -j$(nproc)"
            success_build, stdout, stderr = run_command(cmd, timeout=600)
            if not success_build:
                log(f"  Simulation build failed: {stderr[:200]}", "ERROR")
                success = False
            else:
                log("  Simulation binaries built successfully")
    else:
        log("  All binaries present")
    
    # 3. Initialize type registry if needed
    log("Checking type registry...")
    load_type_registry(weights_dir)
    known_types = list_known_types(weights_dir)
    if known_types:
        log(f"  Loaded {len(known_types)} types: {', '.join(known_types)}")
    else:
        log("  No types yet - will be created as graphs are processed")
    
    # 4. Verify a binary actually works
    log("Verifying binaries work...")
    test_bin = os.path.join(bin_dir, "pr")
    if os.path.exists(test_bin):
        test_success, _, stderr = run_command(f"{test_bin} --help", timeout=5)
        if test_success or "Usage" in stderr or "pagerank" in stderr.lower():
            log("  Binary verification: OK")
        else:
            log("  Binary verification: WARNING - binary may not work correctly", "WARN")
    
    if success:
        log("All prerequisites satisfied", "INFO")
    else:
        log("Some prerequisites failed - continuing with available resources", "WARN")
    
    return success


def check_and_build_binaries(project_dir: str = ".") -> bool:
    """Check if binaries exist, build if necessary.
    
    Returns:
        True if binaries are ready, False otherwise
    """
    print("\n" + "="*60)
    print("BUILD CHECK")
    print("="*60)
    
    # Check for required binaries
    bin_dir = os.path.join(project_dir, "bench", "bin")
    bin_sim_dir = os.path.join(project_dir, "bench", "bin_sim")
    
    required_bins = ["bfs", "pr", "cc", "bc", "sssp", "tc"]
    
    # Check standard binaries
    missing_bins = []
    for binary in required_bins:
        bin_path = os.path.join(bin_dir, binary)
        if not os.path.exists(bin_path):
            missing_bins.append(binary)
    
    # Check simulation binaries
    missing_sim = []
    for binary in required_bins:
        sim_path = os.path.join(bin_sim_dir, binary)
        if not os.path.exists(sim_path):
            missing_sim.append(binary)
    
    if not missing_bins and not missing_sim:
        print("All binaries present - build OK")
        return True
    
    if missing_bins:
        print(f"Missing standard binaries: {', '.join(missing_bins)}")
    if missing_sim:
        print(f"Missing simulation binaries: {', '.join(missing_sim)}")
    
    # Try to build
    makefile = os.path.join(project_dir, "Makefile")
    if not os.path.exists(makefile):
        print("Error: Makefile not found - cannot build")
        return False
    
    print("\nBuilding binaries...")
    
    # Build standard binaries
    if missing_bins:
        print("  Building standard binaries...")
        success, stdout, stderr = run_command(f"cd {project_dir} && make -j$(nproc)", timeout=600)
        if not success:
            print(f"  Build failed: {stderr}")
            return False
        print("  Standard binaries built successfully")
    
    # Build simulation binaries
    if missing_sim:
        print("  Building simulation binaries...")
        success, stdout, stderr = run_command(f"cd {project_dir} && make all-sim -j$(nproc)", timeout=600)
        if not success:
            print(f"  Simulation build failed: {stderr}")
            return False
        print("  Simulation binaries built successfully")
    
    print("Build complete")
    return True

def clean_results(results_dir: str = DEFAULT_RESULTS_DIR, keep_graphs: bool = True, keep_weights: bool = True) -> None:
    """Clean the results directory, optionally keeping graphs and weights.
    
    Args:
        results_dir: Directory to clean
        keep_graphs: If True, don't delete downloaded graphs
        keep_weights: If True, don't delete perceptron weights file
    """
    print("\n" + "="*60)
    print("CLEANING RESULTS DIRECTORY")
    print("="*60)
    
    if not os.path.exists(results_dir):
        print(f"Results directory does not exist: {results_dir}")
        return
    
    # Patterns to clean (relative to results_dir)
    patterns_to_clean = [
        "*.json",
        "*.log", 
        "*.csv",
        "mappings/",
        "logs/",
    ]
    
    # Items to keep
    keep_items = set()
    if keep_graphs:
        keep_items.add("graphs")
    if keep_weights:
        keep_items.add("perceptron_weights.json")
    
    deleted_count = 0
    kept_count = 0
    
    for entry in os.listdir(results_dir):
        entry_path = os.path.join(results_dir, entry)
        
        # Check if should keep
        if entry in keep_items:
            kept_count += 1
            print(f"  Keeping: {entry}")
            continue
        
        # Clean based on pattern
        if entry.endswith(('.json', '.log', '.csv')):
            if entry == "perceptron_weights.json" and keep_weights:
                kept_count += 1
                print(f"  Keeping: {entry}")
                continue
            try:
                os.remove(entry_path)
                deleted_count += 1
                print(f"  Deleted: {entry}")
            except Exception as e:
                print(f"  Error deleting {entry}: {e}")
        elif os.path.isdir(entry_path) and entry in ["mappings", "logs"]:
            try:
                shutil.rmtree(entry_path)
                deleted_count += 1
                print(f"  Deleted: {entry}/")
            except Exception as e:
                print(f"  Error deleting {entry}/: {e}")
    
    print(f"\nCleaned {deleted_count} items, kept {kept_count} items")

def clean_all(project_dir: str = ".", confirm: bool = False) -> None:
    """Clean all generated data for a fresh start.
    
    This removes:
    - All results (including graphs)
    - All label.map files in graph directories
    - Downloaded graphs
    
    Args:
        project_dir: Project root directory
        confirm: If True, skip confirmation prompt
    """
    if not confirm:
        response = input("This will delete ALL generated data including downloaded graphs. Continue? [y/N] ")
        if response.lower() != 'y':
            print("Cancelled")
            return
    
    print("\n" + "="*60)
    print("FULL CLEAN - REMOVING ALL GENERATED DATA")
    print("="*60)
    
    # Clean results directory completely
    results_dir = os.path.join(project_dir, "results")
    if os.path.exists(results_dir):
        print(f"Removing {results_dir}/")
        shutil.rmtree(results_dir)
        os.makedirs(results_dir)  # Recreate empty
        print("  Done")
    
    # Clean label.map files in graphs directory
    graphs_dir = os.path.join(project_dir, "graphs")
    if os.path.exists(graphs_dir):
        map_files = glob.glob(os.path.join(graphs_dir, "**/label.map"), recursive=True)
        if map_files:
            print(f"Removing {len(map_files)} label.map files from graphs/")
            for map_file in map_files:
                try:
                    os.remove(map_file)
                except:
                    pass
            print("  Done")
    
    # Clean bench/results if exists
    bench_results = os.path.join(project_dir, "bench", "results")
    if os.path.exists(bench_results):
        print(f"Removing {bench_results}/")
        shutil.rmtree(bench_results)
        print("  Done")
    
    print("\nClean complete - ready for fresh start")

def get_num_threads() -> int:
    """Get the number of available CPU threads."""
    try:
        return os.cpu_count() or 1
    except:
        return 1

def run_command(cmd: str, timeout: int = 300, use_all_threads: bool = True) -> Tuple[bool, str, str]:
    """Run a shell command with timeout.
    
    Args:
        cmd: Command to run
        timeout: Timeout in seconds
        use_all_threads: If True, set OMP_NUM_THREADS to use all available cores
    """
    try:
        # Set up environment with OpenMP thread count
        env = os.environ.copy()
        if use_all_threads:
            num_threads = get_num_threads()
            env['OMP_NUM_THREADS'] = str(num_threads)
        
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "TIMEOUT"
    except Exception as e:
        return False, "", str(e)

def parse_benchmark_output(output: str) -> Dict[str, Any]:
    """Parse benchmark output for timing and stats."""
    result = {}
    
    # Graph stats
    match = re.search(r'Graph has (\d+) nodes and (\d+)', output)
    if match:
        result['nodes'] = int(match.group(1))
        result['edges'] = int(match.group(2))
    
    # Timing patterns - base execution times
    patterns = {
        'trial_time': r'Trial Time:\s+([\d.]+)',
        'average_time': r'Average Time:\s+([\d.]+)',
        'total_time': r'Total Time:\s+([\d.]+)',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            result[key] = float(match.group(1))
    
    # Reorder/Ordering time patterns - from various algorithms
    # Patterns like: "RabbitOrder Map Time: 0.123", "Ordering Time: 0.456", etc.
    reorder_patterns = [
        r'(?:RabbitOrder|GOrder|LeidenOrder|COrder|RCMOrder|HubSort|DBG|Relabel|Sub-RabbitOrder)\s*Map Time[:\s]+([\d.]+)',
        r'Ordering Time[:\s]+([\d.]+)',
        r'RandOrder Time[:\s]+([\d.]+)',
        r'Reorder Time[:\s]+([\d.]+)',
        r'Total Reordering Time[:\s]+([\d.]+)',
    ]
    
    # Collect all reorder times and use the largest (the actual reorder, not intermediate steps)
    reorder_times = []
    for pattern in reorder_patterns:
        for match in re.finditer(pattern, output, re.IGNORECASE):
            try:
                reorder_times.append(float(match.group(1)))
            except (ValueError, IndexError):
                pass
    
    if reorder_times:
        # Use the maximum (primary algorithm time, not sub-steps)
        result['reorder_time'] = max(reorder_times)
    
    # Modularity
    match = re.search(r'[Mm]odularity[:\s]+([\d.]+)', output)
    if match:
        result['modularity'] = float(match.group(1))
    
    # Global graph features (from AdaptiveOrder output)
    # Degree Variance: 0.1234
    match = re.search(r'Degree Variance[:\s]+([\d.]+)', output)
    if match:
        result['degree_variance'] = float(match.group(1))
    
    # Hub Concentration: 0.1234
    match = re.search(r'Hub Concentration[:\s]+([\d.]+)', output)
    if match:
        result['hub_concentration'] = float(match.group(1))
    
    # Clustering Coefficient: 0.1234
    match = re.search(r'Clustering Coefficient[:\s]+([\d.]+)', output)
    if match:
        result['clustering_coefficient'] = float(match.group(1))
    
    # Community Count: 123 or Number of Communities: 123 or Community Count Estimate: 123
    match = re.search(r'(?:Community Count(?: Estimate)?|Number of Communities|communities)[:\s]+([\d.]+)', output, re.IGNORECASE)
    if match:
        result['community_count'] = int(float(match.group(1)))
    
    # Avg Path Length: 5.6789
    match = re.search(r'Avg Path Length[:\s]+([\d.]+)', output)
    if match:
        result['avg_path_length'] = float(match.group(1))
    
    # Diameter Estimate: 12
    match = re.search(r'Diameter Estimate[:\s]+([\d.]+)', output)
    if match:
        result['diameter'] = float(match.group(1))
    
    # Avg Degree: 10.5 (from topology features)
    match = re.search(r'Avg Degree[:\s]+([\d.]+)', output)
    if match:
        result['avg_degree'] = float(match.group(1))
    
    # Graph Type: social
    match = re.search(r'Graph Type[:\s]+(\w+)', output)
    if match:
        result['graph_type'] = match.group(1).lower()
    
    return result


def parse_reorder_time_from_converter(output: str) -> Optional[float]:
    """
    Parse the actual reorder time from converter output.
    
    The converter outputs lines like:
        HubSort Map Time:    0.05982
        GOrder Map Time:     0.25328
        RabbitOrder Map Time:0.09335
        COrder Map Time:     0.00910
        DBG Map Time:        0.01234
        etc.
    
    We look for the LAST non-Relabel "*Map Time:" entry as that's the final reorder.
    """
    # Match lines like 'AlgoName Map Time:' but NOT 'Relabel Map Time:'
    pattern = r'^([A-Za-z]+)\s+Map Time:\s*([\d.]+)'
    matches = re.findall(pattern, output, re.MULTILINE)
    
    # Filter out Relabel entries
    matches = [(name, time) for name, time in matches if name != 'Relabel']
    
    if matches:
        # Return the last one (the actual reorder algorithm time)
        return float(matches[-1][1])
    
    return None

def parse_cache_output(output: str) -> Dict[str, float]:
    """Parse cache simulation output."""
    result = {}
    
    # The output format is structured in blocks like:
    # ║ L1 Cache (32KB, 8-way, LRU)
    # ║   Hit Rate:                 55.6082%
    # We need to extract the hit rate from each cache block
    
    # Split by cache sections
    l1_match = re.search(r'L1 Cache.*?Hit Rate:\s*([\d.]+)%', output, re.DOTALL)
    l2_match = re.search(r'L2 Cache.*?Hit Rate:\s*([\d.]+)%', output, re.DOTALL)
    l3_match = re.search(r'L3 Cache.*?Hit Rate:\s*([\d.]+)%', output, re.DOTALL)
    
    if l1_match:
        result['l1_hit_rate'] = float(l1_match.group(1))
    if l2_match:
        result['l2_hit_rate'] = float(l2_match.group(1))
    if l3_match:
        result['l3_hit_rate'] = float(l3_match.group(1))
    
    # Fallback patterns for different formats
    if 'l1_hit_rate' not in result:
        patterns = {
            'l1_hit_rate': r'L1 Hit Rate:\s*([\d.]+)%?',
            'l2_hit_rate': r'L2 Hit Rate:\s*([\d.]+)%?',
            'l3_hit_rate': r'L3 Hit Rate:\s*([\d.]+)%?',
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, output)
            if match:
                result[key] = float(match.group(1))
    
    # Also check for compact summary format: L1:XX.X% L2:XX.X% L3:XX.X%
    summary = re.search(r'L1:([\d.]+)%\s*L2:([\d.]+)%\s*L3:([\d.]+)%', output)
    if summary:
        result['l1_hit_rate'] = float(summary.group(1))
        result['l2_hit_rate'] = float(summary.group(2))
        result['l3_hit_rate'] = float(summary.group(3))
    
    return result

# ============================================================================
# Phase 1: Generate Reorderings and Record Time
# ============================================================================

def generate_reorderings(
    graphs: List[GraphInfo],
    algorithms: List[int],
    bin_dir: str,
    output_dir: str,
    timeout: int = TIMEOUT_REORDER,
    skip_slow: bool = False,
    generate_maps: bool = True
) -> List[ReorderResult]:
    """
    Generate reorderings for all graphs and algorithms.
    Records reorder time for each combination.
    
    Args:
        graphs: List of graphs to process
        algorithms: List of algorithm IDs to use
        bin_dir: Directory containing binaries
        output_dir: Directory for outputs (mappings will be in output_dir/mappings/)
        timeout: Timeout for each reordering
        skip_slow: Skip slow algorithms on large graphs
        generate_maps: If True, generate .lo mapping files (default: True)
    """
    log_section("Phase 1: Generate Reorderings")
    
    results = []
    total = len(graphs) * len(algorithms)
    current = 0
    
    # Create output directory for mappings
    mappings_dir = os.path.join(output_dir, "mappings")
    os.makedirs(mappings_dir, exist_ok=True)
    
    for graph in graphs:
        log(f"\nGraph: {graph.name} ({graph.size_mb:.1f}MB)")
        
        # Create per-graph mappings directory
        graph_mappings_dir = os.path.join(mappings_dir, graph.name)
        if generate_maps:
            os.makedirs(graph_mappings_dir, exist_ok=True)
        
        for algo_id in algorithms:
            current += 1
            algo_name = ALGORITHMS.get(algo_id, f"ALGO_{algo_id}")
            
            # Skip slow algorithms on large graphs if requested
            if skip_slow and algo_id in SLOW_ALGORITHMS and graph.size_mb > SIZE_MEDIUM:
                log(f"  [{current}/{total}] {algo_name}: SKIPPED (slow on large graphs)")
                results.append(ReorderResult(
                    graph=graph.name,
                    algorithm_id=algo_id,
                    algorithm_name=algo_name,
                    reorder_time=0.0,
                    success=False,
                    error="SKIPPED"
                ))
                continue
            
            # ORIGINAL doesn't need reordering
            if algo_id == 0:
                log(f"  [{current}/{total}] {algo_name}: 0.0000s (no reorder)")
                results.append(ReorderResult(
                    graph=graph.name,
                    algorithm_id=algo_id,
                    algorithm_name=algo_name,
                    reorder_time=0.0,
                    success=True
                ))
                continue
            
            # Output mapping file path
            map_file = os.path.join(graph_mappings_dir, f"{algo_name}.lo") if generate_maps else None
            
            # Check if mapping already exists
            if generate_maps and map_file and os.path.exists(map_file):
                # Load existing timing if available
                timing_file = os.path.join(graph_mappings_dir, f"{algo_name}.time")
                if os.path.exists(timing_file):
                    with open(timing_file) as f:
                        reorder_time = float(f.read().strip())
                else:
                    reorder_time = 0.0
                
                log(f"  [{current}/{total}] {algo_name}: exists ({reorder_time:.4f}s)")
                results.append(ReorderResult(
                    graph=graph.name,
                    algorithm_id=algo_id,
                    algorithm_name=algo_name,
                    reorder_time=reorder_time,
                    mapping_file=map_file,
                    success=True
                ))
                continue
            
            # Generate mapping with converter (also times it)
            if generate_maps:
                binary = os.path.join(bin_dir, "converter")
                sym_flag = "-s" if graph.is_symmetric else ""
                cmd = f"{binary} -f {graph.path} {sym_flag} -o {algo_id} -q {map_file}"
            else:
                # Use pr binary with 1 trial to measure reorder time
                binary = os.path.join(bin_dir, "pr")
                sym_flag = "-s" if graph.is_symmetric else ""
                cmd = f"{binary} -f {graph.path} {sym_flag} -o {algo_id} -n 1"
            
            # Run and parse
            start_time = time.time()
            success, stdout, stderr = run_command(cmd, timeout)
            elapsed = time.time() - start_time
            
            if success:
                output = stdout + stderr
                
                if generate_maps:
                    # Verify map file was created
                    if os.path.exists(map_file):
                        # Parse actual reorder time from converter output
                        actual_reorder_time = parse_reorder_time_from_converter(output)
                        if actual_reorder_time is not None:
                            reorder_time = actual_reorder_time
                        else:
                            # Fallback to elapsed time if parsing fails
                            reorder_time = elapsed
                        
                        # Save timing to file for future reference
                        timing_file = os.path.join(graph_mappings_dir, f"{algo_name}.time")
                        with open(timing_file, 'w') as f:
                            f.write(f"{reorder_time:.6f}")
                        
                        log(f"  [{current}/{total}] {algo_name}: {reorder_time:.4f}s (map: {algo_name}.lo)")
                        results.append(ReorderResult(
                            graph=graph.name,
                            algorithm_id=algo_id,
                            algorithm_name=algo_name,
                            reorder_time=reorder_time,
                            mapping_file=map_file,
                            success=True
                        ))
                    else:
                        log(f"  [{current}/{total}] {algo_name}: FAILED (no map file)")
                        results.append(ReorderResult(
                            graph=graph.name,
                            algorithm_id=algo_id,
                            algorithm_name=algo_name,
                            reorder_time=elapsed,
                            success=False,
                            error="Map file not created"
                        ))
                else:
                    parsed = parse_benchmark_output(output)
                    reorder_time = parsed.get('reorder_time', elapsed)
                    
                    log(f"  [{current}/{total}] {algo_name}: {reorder_time:.4f}s")
                    results.append(ReorderResult(
                        graph=graph.name,
                        algorithm_id=algo_id,
                        algorithm_name=algo_name,
                        reorder_time=reorder_time,
                        success=True
                    ))
            else:
                error = "TIMEOUT" if "TIMEOUT" in stderr else stderr[:100]
                log(f"  [{current}/{total}] {algo_name}: FAILED ({error})")
                results.append(ReorderResult(
                    graph=graph.name,
                    algorithm_id=algo_id,
                    algorithm_name=algo_name,
                    reorder_time=0.0,
                    success=False,
                    error=error
                ))
    
    return results

def generate_label_maps(
    graphs: List[GraphInfo],
    algorithms: List[int],
    bin_dir: str,
    output_dir: str,
    timeout: int = TIMEOUT_REORDER,
    skip_slow: bool = False
) -> Tuple[Dict[str, Dict[str, str]], List[ReorderResult]]:
    """
    Pre-generate label.map files for each graph/algorithm combination.
    Also records reorder times during generation.
    
    This allows consistent reordering across multiple benchmark runs
    by using the MAP algorithm (14) with pre-generated mappings.
    
    Returns:
        Tuple of:
        - Dictionary mapping (graph, algorithm) to label map file path
        - List of ReorderResult with timing information
    """
    log_section("Pre-generate Label Maps + Record Reorder Times")
    
    # Create mappings directory
    mappings_dir = os.path.join(output_dir, "mappings")
    os.makedirs(mappings_dir, exist_ok=True)
    
    label_maps = {}  # {graph: {algo: path}}
    reorder_results = []  # Store timing information
    total = len(graphs) * len(algorithms)
    current = 0
    
    for graph in graphs:
        log(f"\nGraph: {graph.name} ({graph.size_mb:.1f}MB)")
        label_maps[graph.name] = {}
        graph_mappings_dir = os.path.join(mappings_dir, graph.name)
        os.makedirs(graph_mappings_dir, exist_ok=True)
        
        for algo_id in algorithms:
            current += 1
            algo_name = ALGORITHMS.get(algo_id, f"ALGO_{algo_id}")
            
            # Skip ORIGINAL (no mapping needed)
            if algo_id == 0:
                log(f"  [{current}/{total}] {algo_name}: no map needed (0.0000s)")
                reorder_results.append(ReorderResult(
                    graph=graph.name,
                    algorithm_id=algo_id,
                    algorithm_name=algo_name,
                    reorder_time=0.0,
                    mapping_file="",
                    success=True
                ))
                continue
            
            # Skip slow algorithms on large graphs if requested
            if skip_slow and algo_id in SLOW_ALGORITHMS and graph.size_mb > SIZE_MEDIUM:
                log(f"  [{current}/{total}] {algo_name}: SKIPPED (slow)")
                reorder_results.append(ReorderResult(
                    graph=graph.name,
                    algorithm_id=algo_id,
                    algorithm_name=algo_name,
                    reorder_time=0.0,
                    mapping_file="",
                    success=False,
                    error="SKIPPED"
                ))
                continue
            
            # Output mapping file path
            map_file = os.path.join(graph_mappings_dir, f"{algo_name}.lo")
            
            # Check if already exists - still need to record approximate time
            if os.path.exists(map_file):
                # Load existing timing if available
                timing_file = os.path.join(graph_mappings_dir, f"{algo_name}.time")
                if os.path.exists(timing_file):
                    with open(timing_file) as f:
                        reorder_time = float(f.read().strip())
                else:
                    reorder_time = 0.0  # Unknown timing for existing file
                
                log(f"  [{current}/{total}] {algo_name}: exists ({reorder_time:.4f}s)")
                label_maps[graph.name][algo_name] = map_file
                reorder_results.append(ReorderResult(
                    graph=graph.name,
                    algorithm_id=algo_id,
                    algorithm_name=algo_name,
                    reorder_time=reorder_time,
                    mapping_file=map_file,
                    success=True
                ))
                continue
            
            # Use converter to generate mapping and time it
            binary = os.path.join(bin_dir, "converter")
            sym_flag = "-s" if graph.is_symmetric else ""
            cmd = f"{binary} -f {graph.path} {sym_flag} -o {algo_id} -q {map_file}"
            
            start_time = time.time()
            success, stdout, stderr = run_command(cmd, timeout)
            elapsed = time.time() - start_time
            
            if success and os.path.exists(map_file):
                # Save timing to file for future reference
                timing_file = os.path.join(graph_mappings_dir, f"{algo_name}.time")
                with open(timing_file, 'w') as f:
                    f.write(f"{elapsed:.6f}")
                
                log(f"  [{current}/{total}] {algo_name}: generated ({elapsed:.4f}s)")
                label_maps[graph.name][algo_name] = map_file
                reorder_results.append(ReorderResult(
                    graph=graph.name,
                    algorithm_id=algo_id,
                    algorithm_name=algo_name,
                    reorder_time=elapsed,
                    mapping_file=map_file,
                    success=True
                ))
            else:
                error = "TIMEOUT" if "TIMEOUT" in stderr else stderr[:50]
                log(f"  [{current}/{total}] {algo_name}: FAILED ({error})")
                reorder_results.append(ReorderResult(
                    graph=graph.name,
                    algorithm_id=algo_id,
                    algorithm_name=algo_name,
                    reorder_time=elapsed,
                    mapping_file="",
                    success=False,
                    error=error
                ))
    
    # Save mapping index
    index_file = os.path.join(mappings_dir, "index.json")
    with open(index_file, 'w') as f:
        json.dump(label_maps, f, indent=2)
    log(f"\nLabel map index saved to: {index_file}")
    
    # Save reorder times to JSON and CSV
    reorder_json = os.path.join(output_dir, f"reorder_times_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(reorder_json, 'w') as f:
        json.dump([asdict(r) for r in reorder_results], f, indent=2)
    log(f"Reorder times saved to: {reorder_json}")
    
    # Also save as CSV for easy analysis
    reorder_csv = os.path.join(output_dir, f"reorder_times_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(reorder_csv, 'w') as f:
        f.write("graph,algorithm_id,algorithm_name,reorder_time,success,error\n")
        for r in reorder_results:
            f.write(f"{r.graph},{r.algorithm_id},{r.algorithm_name},{r.reorder_time:.6f},{r.success},{r.error}\n")
    log(f"Reorder times CSV saved to: {reorder_csv}")
    
    return label_maps, reorder_results
    
    return label_maps

def get_label_map_path(
    label_maps: Dict[str, Dict[str, str]],
    graph_name: str,
    algo_name: str
) -> Optional[str]:
    """Get the path to a pre-generated label map, if available."""
    if graph_name in label_maps and algo_name in label_maps[graph_name]:
        path = label_maps[graph_name][algo_name]
        if os.path.exists(path):
            return path
    return None

def load_label_maps_index(results_dir: str) -> Dict[str, Dict[str, str]]:
    """Load the label maps index from a previous run."""
    index_file = os.path.join(results_dir, "mappings", "index.json")
    if os.path.exists(index_file):
        with open(index_file) as f:
            return json.load(f)
    return {}

# ============================================================================
# Phase 2: Run Execution Benchmarks
# ============================================================================

def run_benchmarks(
    graphs: List[GraphInfo],
    algorithms: List[int],
    benchmarks: List[str],
    bin_dir: str,
    num_trials: int = 2,
    timeout: int = TIMEOUT_BENCHMARK,
    skip_slow: bool = False,
    label_maps: Dict[str, Dict[str, str]] = None,
    weights_dir: str = DEFAULT_WEIGHTS_DIR,
    update_weights: bool = True
) -> List[BenchmarkResult]:
    """
    Run execution benchmarks for all combinations.
    
    If label_maps is provided, uses pre-generated mappings via MAP algorithm (14)
    for consistent reordering instead of regenerating each time.
    
    If update_weights is True, incrementally updates type weights after each benchmark.
    """
    log_section("Phase 2: Execution Benchmarks")
    
    if label_maps:
        log(f"Using pre-generated label maps for {len(label_maps)} graphs")
    
    results = []
    total = len(graphs) * len(algorithms) * len(benchmarks)
    current = 0
    
    # BASELINE_ALGORITHM = 1 (RANDOM) for fair comparison
    # RANDOM represents the worst-case random ordering, so speedups are meaningful
    BASELINE_ALGO_ID = 1
    
    for graph in graphs:
        log(f"\nGraph: {graph.name} ({graph.size_mb:.1f}MB)")
        
        # Store baseline times for speedup calculation (using RANDOM as baseline)
        baseline_times = {}
        
        for bench in benchmarks:
            log(f"  {bench.upper()}:")
            binary = os.path.join(bin_dir, bench)
            
            if not os.path.exists(binary):
                log(f"    Binary not found: {binary}")
                continue
            
            for algo_id in algorithms:
                current += 1
                algo_name = ALGORITHMS.get(algo_id, f"ALGO_{algo_id}")
                
                # Skip slow algorithms on large graphs
                if skip_slow and algo_id in SLOW_ALGORITHMS and graph.size_mb > SIZE_MEDIUM:
                    results.append(BenchmarkResult(
                        graph=graph.name,
                        algorithm_id=algo_id,
                        algorithm_name=algo_name,
                        benchmark=bench,
                        trial_time=0.0,
                        success=False,
                        error="SKIPPED"
                    ))
                    continue
                
                # Check for pre-generated label map
                label_map_path = None
                if label_maps and algo_id != 0:  # ORIGINAL doesn't need mapping
                    label_map_path = get_label_map_path(label_maps, graph.name, algo_name)
                
                # Build command - use MAP algorithm (14) with label map if available
                sym_flag = "-s" if graph.is_symmetric else ""
                if label_map_path:
                    # Use pre-generated mapping via MAP (algo 14)
                    cmd = f"{binary} -f {graph.path} {sym_flag} -o 14:{label_map_path} -n {num_trials}"
                else:
                    # Generate reordering on-the-fly
                    cmd = f"{binary} -f {graph.path} {sym_flag} -o {algo_id} -n {num_trials}"
                
                # Run
                success, stdout, stderr = run_command(cmd, timeout)
                
                if success:
                    output = stdout + stderr
                    parsed = parse_benchmark_output(output)
                    trial_time = parsed.get('average_time', parsed.get('trial_time', 0.0))
                    nodes = parsed.get('nodes', 0)
                    edges = parsed.get('edges', 0)
                    
                    # Compute derived properties
                    computed_avg_degree = 2 * edges / nodes if nodes > 0 else 0
                    max_edges = nodes * (nodes - 1) / 2 if nodes > 1 else 1
                    computed_density = edges / max_edges if max_edges > 0 else 0.0
                    
                    # Update graph properties cache with any parsed features
                    graph_features = {
                        'modularity': parsed.get('modularity'),
                        'degree_variance': parsed.get('degree_variance'),
                        'hub_concentration': parsed.get('hub_concentration'),
                        'clustering_coefficient': parsed.get('clustering_coefficient'),
                        'community_count': parsed.get('community_count'),
                        'graph_type': parsed.get('graph_type'),
                        'nodes': nodes,
                        'edges': edges,
                        'avg_degree': computed_avg_degree,
                        'density': computed_density,
                    }
                    # Only update if we got meaningful parsed data
                    if any(k in parsed for k in ['modularity', 'degree_variance', 'hub_concentration', 
                                                   'clustering_coefficient', 'community_count', 'graph_type']):
                        update_graph_properties(graph.name, graph_features)
                    
                    # Record baseline for speedup (RANDOM = algo_id 1)
                    if algo_id == BASELINE_ALGO_ID:
                        baseline_times[bench] = trial_time
                    
                    # Calculate speedup vs RANDOM baseline
                    baseline = baseline_times.get(bench, trial_time)
                    speedup = baseline / trial_time if trial_time > 0 else 0.0
                    
                    log(f"    [{current}/{total}] {algo_name}: {trial_time:.4f}s (speedup: {speedup:.2f}x)")
                    
                    # Incremental weight update
                    if update_weights and speedup > 0:
                        # Get or assign graph type
                        cached_props = _graph_properties_cache.get(graph.name, {})
                        
                        # Compute actual values from benchmark output
                        actual_nodes = nodes or cached_props.get('nodes', 1000)
                        actual_edges = edges or cached_props.get('edges', 5000)
                        
                        # Compute density: edges / (nodes * (nodes-1) / 2) for undirected
                        max_edges = actual_nodes * (actual_nodes - 1) / 2 if actual_nodes > 1 else 1
                        computed_density = actual_edges / max_edges if max_edges > 0 else 0.0
                        
                        # Compute avg_degree: 2 * edges / nodes
                        computed_avg_degree = (2 * actual_edges / actual_nodes) if actual_nodes > 0 else 0.0
                        
                        features = {
                            'modularity': cached_props.get('modularity', 0.5),
                            'degree_variance': cached_props.get('degree_variance', 1.0),
                            'hub_concentration': cached_props.get('hub_concentration', 0.3),
                            'avg_degree': cached_props.get('avg_degree') or computed_avg_degree,
                            'nodes': actual_nodes,
                            'edges': actual_edges,
                            'density': cached_props.get('density') or computed_density,
                            'clustering_coefficient': cached_props.get('clustering_coefficient', 0.0),
                            'community_count': cached_props.get('community_count', 0),
                        }
                        graph_type = assign_graph_type(features, weights_dir)
                        
                        # Get reorder time from benchmark output (reorder_time field from parsed output)
                        reorder_time = parsed.get('reorder_time', 0.0) or parsed.get('ordering_time', 0.0)
                        
                        # Update weights for this type
                        update_type_weights_incremental(
                            type_name=graph_type,
                            algorithm=algo_name,
                            benchmark=bench,
                            speedup=speedup,
                            features=features,
                            reorder_time=reorder_time,
                            weights_dir=weights_dir
                        )
                    
                    results.append(BenchmarkResult(
                        graph=graph.name,
                        algorithm_id=algo_id,
                        algorithm_name=algo_name,
                        benchmark=bench,
                        trial_time=trial_time,
                        speedup=speedup,
                        nodes=nodes,
                        edges=edges,
                        success=True
                    ))
                else:
                    error = "TIMEOUT" if "TIMEOUT" in stderr else "FAILED"
                    log(f"    [{current}/{total}] {algo_name}: {error}")
                    results.append(BenchmarkResult(
                        graph=graph.name,
                        algorithm_id=algo_id,
                        algorithm_name=algo_name,
                        benchmark=bench,
                        trial_time=0.0,
                        success=False,
                        error=error
                    ))
    
    return results

# ============================================================================
# Phase 3: Cache Simulations
# ============================================================================

def run_cache_simulations(
    graphs: List[GraphInfo],
    algorithms: List[int],
    benchmarks: List[str],
    bin_sim_dir: str,
    timeout: int = TIMEOUT_SIM,
    skip_heavy: bool = False,
    label_maps: Dict[str, Dict[str, str]] = None,
    weights_dir: str = DEFAULT_WEIGHTS_DIR,
    update_weights: bool = True
) -> List[CacheResult]:
    """
    Run cache simulations for all combinations.
    
    If label_maps is provided, uses pre-generated mappings via MAP algorithm (14)
    for consistent reordering instead of regenerating each time.
    
    If update_weights is True, updates type weights with cache stats.
    """
    log_section("Phase 3: Cache Simulations")
    
    results = []
    total = len(graphs) * len(algorithms) * len(benchmarks)
    current = 0
    
    for graph in graphs:
        log(f"\nGraph: {graph.name} ({graph.size_mb:.1f}MB)")
        
        for bench in benchmarks:
            log(f"  {bench.upper()} (simulation):")
            binary = os.path.join(bin_sim_dir, bench)
            
            if not os.path.exists(binary):
                log(f"    Binary not found: {binary}")
                continue
            
            # Use longer timeout for heavy benchmarks
            bench_timeout = TIMEOUT_SIM_HEAVY if bench in HEAVY_SIM_BENCHMARKS else timeout
            
            # Skip heavy simulations on large graphs if requested
            if skip_heavy and bench in HEAVY_SIM_BENCHMARKS and graph.size_mb > SIZE_MEDIUM:
                for algo_id in algorithms:
                    current += 1
                    results.append(CacheResult(
                        graph=graph.name,
                        algorithm_id=algo_id,
                        algorithm_name=ALGORITHMS.get(algo_id, f"ALGO_{algo_id}"),
                        benchmark=bench,
                        success=False,
                        error="SKIPPED"
                    ))
                continue
            
            for algo_id in algorithms:
                current += 1
                algo_name = ALGORITHMS.get(algo_id, f"ALGO_{algo_id}")
                
                # Check for pre-generated label map
                label_map_path = None
                if label_maps and algo_id != 0:  # ORIGINAL doesn't need mapping
                    label_map_path = get_label_map_path(label_maps, graph.name, algo_name)
                
                # Build command - use MAP algorithm (14) with label map if available
                sym_flag = "-s" if graph.is_symmetric else ""
                if label_map_path:
                    # Use pre-generated mapping via MAP (algo 14)
                    cmd = f"{binary} -f {graph.path} {sym_flag} -o 14:{label_map_path} -n 1"
                else:
                    # Generate reordering on-the-fly
                    cmd = f"{binary} -f {graph.path} {sym_flag} -o {algo_id} -n 1"
                
                # Run
                success, stdout, stderr = run_command(cmd, bench_timeout)
                
                if success:
                    output = stdout + stderr
                    parsed = parse_cache_output(output)
                    
                    l1 = parsed.get('l1_hit_rate', 0.0)
                    l2 = parsed.get('l2_hit_rate', 0.0)
                    l3 = parsed.get('l3_hit_rate', 0.0)
                    
                    log(f"    [{current}/{total}] {algo_name}: L1:{l1:.1f}% L2:{l2:.1f}% L3:{l3:.1f}%")
                    
                    # Incremental weight update with cache stats
                    if update_weights:
                        cached_props = _graph_properties_cache.get(graph.name, {})
                        
                        # Get actual node/edge counts
                        actual_nodes = cached_props.get('nodes', 1000)
                        actual_edges = cached_props.get('edges', 5000)
                        
                        # Compute density and avg_degree if not cached
                        max_edges = actual_nodes * (actual_nodes - 1) / 2 if actual_nodes > 1 else 1
                        computed_density = actual_edges / max_edges if max_edges > 0 else 0.0
                        computed_avg_degree = (2 * actual_edges / actual_nodes) if actual_nodes > 0 else 0.0
                        
                        features = {
                            'modularity': cached_props.get('modularity', 0.5),
                            'degree_variance': cached_props.get('degree_variance', 1.0),
                            'hub_concentration': cached_props.get('hub_concentration', 0.3),
                            'avg_degree': cached_props.get('avg_degree') or computed_avg_degree,
                            'nodes': actual_nodes,
                            'edges': actual_edges,
                            'density': cached_props.get('density') or computed_density,
                            'clustering_coefficient': cached_props.get('clustering_coefficient', 0.0),
                            'community_count': cached_props.get('community_count', 0),
                        }
                        graph_type = assign_graph_type(features, weights_dir, create_if_outlier=False)
                        
                        if graph_type:
                            cache_stats = {
                                'l1_hit_rate': l1,
                                'l2_hit_rate': l2,
                                'l3_hit_rate': l3,
                            }
                            # Update with cache data (speedup=1.0 since we don't have timing here)
                            update_type_weights_incremental(
                                type_name=graph_type,
                                algorithm=algo_name,
                                benchmark=bench,
                                speedup=1.0,  # Cache-only update
                                features=features,
                                cache_stats=cache_stats,
                                weights_dir=weights_dir,
                                learning_rate=0.005  # Lower rate for cache-only updates
                            )
                    
                    results.append(CacheResult(
                        graph=graph.name,
                        algorithm_id=algo_id,
                        algorithm_name=algo_name,
                        benchmark=bench,
                        l1_hit_rate=l1,
                        l2_hit_rate=l2,
                        l3_hit_rate=l3,
                        success=True
                    ))
                else:
                    error = "TIMEOUT" if "TIMEOUT" in stderr else "FAILED"
                    log(f"    [{current}/{total}] {algo_name}: {error}")
                    results.append(CacheResult(
                        graph=graph.name,
                        algorithm_id=algo_id,
                        algorithm_name=algo_name,
                        benchmark=bench,
                        success=False,
                        error=error
                    ))
    
    return results

# ============================================================================
# Phase 4: Generate Perceptron Weights
# ============================================================================

def generate_perceptron_weights(
    benchmark_results: List[BenchmarkResult],
    cache_results: List[CacheResult],
    reorder_results: List[ReorderResult],
    output_file: str
) -> Dict[str, PerceptronWeight]:
    """
    Generate perceptron weights from benchmark and cache results.
    Includes reorder time as a weight component.
    """
    log_section("Phase 4: Generate Perceptron Weights")
    
    weights = {}
    
    # Group results by algorithm
    algo_bench_results = {}
    algo_cache_results = {}
    algo_reorder_results = {}
    
    for r in benchmark_results:
        if r.success:
            if r.algorithm_name not in algo_bench_results:
                algo_bench_results[r.algorithm_name] = []
            algo_bench_results[r.algorithm_name].append(r)
    
    for r in cache_results:
        if r.success:
            if r.algorithm_name not in algo_cache_results:
                algo_cache_results[r.algorithm_name] = []
            algo_cache_results[r.algorithm_name].append(r)
    
    for r in reorder_results:
        if r.success:
            if r.algorithm_name not in algo_reorder_results:
                algo_reorder_results[r.algorithm_name] = []
            algo_reorder_results[r.algorithm_name].append(r)
    
    # Calculate weights for each algorithm
    for algo_name in ALGORITHMS.values():
        bench_data = algo_bench_results.get(algo_name, [])
        cache_data = algo_cache_results.get(algo_name, [])
        reorder_data = algo_reorder_results.get(algo_name, [])
        
        if not bench_data:
            continue
        
        # Calculate statistics
        speedups = [r.speedup for r in bench_data if r.speedup > 0]
        avg_speedup = sum(speedups) / len(speedups) if speedups else 1.0
        
        # Count wins (times this algo was best)
        wins = sum(1 for r in bench_data if r.speedup >= max(
            rr.speedup for rr in bench_data if rr.graph == r.graph and rr.benchmark == r.benchmark
        ) * 0.99)  # Within 1% of best
        
        win_rate = wins / len(bench_data) if bench_data else 0
        
        # Calculate average reorder time
        reorder_times = [r.reorder_time for r in reorder_data if r.reorder_time > 0]
        avg_reorder_time = sum(reorder_times) / len(reorder_times) if reorder_times else 0.0
        
        # Calculate cache impact
        l1_rates = [r.l1_hit_rate for r in cache_data if r.l1_hit_rate > 0]
        l2_rates = [r.l2_hit_rate for r in cache_data if r.l2_hit_rate > 0]
        l3_rates = [r.l3_hit_rate for r in cache_data if r.l3_hit_rate > 0]
        
        avg_l1 = sum(l1_rates) / len(l1_rates) if l1_rates else 0.0
        avg_l2 = sum(l2_rates) / len(l2_rates) if l2_rates else 0.0
        avg_l3 = sum(l3_rates) / len(l3_rates) if l3_rates else 0.0
        
        # Compute weights
        # Base bias from average speedup
        bias = avg_speedup / 2.0
        
        # Adjust weights based on win rate and performance patterns
        w = PerceptronWeight(
            bias=bias,
            # Topology weights - start with small values, will be refined
            w_modularity=-0.001 * (1 - win_rate),  # Negative if not best on modular graphs
            w_log_nodes=0.001 * win_rate,
            w_log_edges=0.0005 * (avg_speedup - 1),
            w_density=0.001 * (1 if avg_speedup > 1.5 else -1),
            w_avg_degree=0.001 * (avg_speedup - 1),
            w_degree_variance=0.001 * win_rate,
            w_hub_concentration=0.001 * win_rate,
            # Cache weights
            cache_l1_impact=0.01 * (avg_l1 / 100 - 0.5) if avg_l1 > 0 else 0,
            cache_l2_impact=0.005 * (avg_l2 / 100 - 0.5) if avg_l2 > 0 else 0,
            cache_l3_impact=0.002 * (avg_l3 / 100 - 0.5) if avg_l3 > 0 else 0,
            cache_dram_penalty=-0.001 * (1 - avg_l3 / 100) if avg_l3 > 0 else 0,
            # Reorder time weight (negative - penalize long reorder times)
            w_reorder_time=-0.001 * avg_reorder_time if avg_reorder_time > 0 else 0,
            _metadata={
                "win_rate": round(win_rate, 3),
                "avg_speedup": round(avg_speedup, 4),
                "times_best": wins,
                "sample_count": len(bench_data),
                "avg_reorder_time": round(avg_reorder_time, 4),
                "avg_l1_hit_rate": round(avg_l1, 2),
                "avg_l2_hit_rate": round(avg_l2, 2),
                "avg_l3_hit_rate": round(avg_l3, 2),
            }
        )
        
        weights[algo_name] = w
        log(f"{algo_name}: bias={bias:.3f}, win_rate={win_rate:.2f}, speedup={avg_speedup:.2f}x, reorder={avg_reorder_time:.2f}s")
    
    # Add generation info
    weights_dict = {name: w.to_dict() for name, w in weights.items()}
    weights_dict["_generation_info"] = {
        "timestamp": datetime.now().isoformat(),
        "algorithms": len(weights),
        "benchmark_results": len(benchmark_results),
        "cache_results": len(cache_results),
        "reorder_results": len(reorder_results),
        "script": "graphbrew_experiment.py"
    }
    
    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(weights_dict, f, indent=2)
    
    log(f"\nWeights saved to: {output_file}")
    
    # Backup with timestamp and sync to scripts folder
    backup_and_sync_weights(output_file)
    
    return weights

# ============================================================================
# Phase 5: Update Zero Weights (Comprehensive)
# ============================================================================

def calculate_correlation(x: List[float], y: List[float]) -> float:
    """Calculate Pearson correlation coefficient."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denom_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    denom_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    
    if denom_x == 0 or denom_y == 0:
        return 0.0
    
    return numerator / (denom_x * denom_y)

@dataclass
class SubcommunityBruteForceResult:
    """Result from brute-force testing all algorithms on a subcommunity."""
    community_id: int
    nodes: int
    edges: int
    density: float
    degree_variance: float
    hub_concentration: float
    
    # Adaptive choice
    adaptive_algorithm: str
    adaptive_time: float = 0.0
    adaptive_l1_hit: float = 0.0
    adaptive_l2_hit: float = 0.0
    adaptive_l3_hit: float = 0.0
    
    # Brute force best (execution time)
    best_time_algorithm: str = ""
    best_time: float = 0.0
    best_time_l1_hit: float = 0.0
    best_time_l2_hit: float = 0.0
    best_time_l3_hit: float = 0.0
    
    # Brute force best (cache - L1)
    best_cache_algorithm: str = ""
    best_cache_l1_hit: float = 0.0
    best_cache_l2_hit: float = 0.0
    best_cache_l3_hit: float = 0.0
    best_cache_time: float = 0.0
    
    # All algorithm results
    all_results: Dict[str, Dict] = field(default_factory=dict)
    
    # Comparison metrics
    adaptive_vs_best_time_ratio: float = 1.0  # adaptive_time / best_time (lower is worse)
    adaptive_vs_best_cache_ratio: float = 1.0  # adaptive_l1 / best_l1 (lower is worse)
    adaptive_is_best_time: bool = False
    adaptive_is_best_cache: bool = False
    adaptive_rank_time: int = 0  # Rank among all algorithms (1 = best)
    adaptive_rank_cache: int = 0

@dataclass
class GraphBruteForceAnalysis:
    """Complete brute-force analysis for a graph."""
    graph: str
    size_mb: float
    modularity: float
    num_communities: int
    num_subcommunities_analyzed: int
    subcommunity_results: List[SubcommunityBruteForceResult] = field(default_factory=list)
    
    # Summary statistics
    adaptive_correct_time_pct: float = 0.0  # % of times adaptive chose best for time
    adaptive_correct_cache_pct: float = 0.0  # % of times adaptive chose best for cache
    adaptive_top3_time_pct: float = 0.0  # % of times adaptive in top 3 for time
    adaptive_top3_cache_pct: float = 0.0  # % of times adaptive in top 3 for cache
    avg_time_ratio: float = 1.0
    avg_cache_ratio: float = 1.0
    
    success: bool = True
    error: str = ""

def run_subcommunity_brute_force(
    graphs: List[GraphInfo],
    bin_dir: str,
    bin_sim_dir: str,
    output_dir: str,
    benchmark: str = "pr",
    timeout: int = TIMEOUT_BENCHMARK,
    timeout_sim: int = TIMEOUT_SIM,
    max_subcommunities: int = 20,  # Limit for large graphs
    num_trials: int = 2
) -> List[GraphBruteForceAnalysis]:
    """
    Brute-force comparison of all algorithms vs adaptive choice for each subcommunity.
    
    For each graph:
    1. Run AdaptiveOrder to get subcommunity info and adaptive choices
    2. For each subcommunity (up to max_subcommunities):
       - Run all 20 algorithms and measure time + cache
       - Compare against adaptive choice
    3. Generate detailed comparison table
    
    NOTE: All operations run sequentially (not in parallel) to ensure accurate
    performance measurements. Parallel execution would cause CPU contention.
    """
    log_section("Subcommunity Brute-Force Analysis: Adaptive vs All Algorithms")
    log("Note: Sequential execution for accurate timing (no parallelism)")
    
    results = []
    
    # All algorithms to test (0-20, excluding MAP=14 and AdaptiveOrder=15)
    test_algorithms = [i for i in range(21) if i not in [14, 15]]
    
    # Create mapping from adaptive output names to our algorithm names
    adaptive_to_algo_name = {
        "Original": "ORIGINAL",
        "Random": "RANDOM",
        "Sort": "SORT",
        "HubSort": "HUBSORT",
        "HubCluster": "HUBCLUSTER",
        "DBG": "DBG",
        "HubSortDBG": "HUBSORTDBG",
        "HubClusterDBG": "HUBCLUSTERDBG",
        "RabbitOrder": "RABBITORDER",
        "GOrder": "GORDER",
        "Corder": "CORDER",
        "RCM": "RCM",
        "LeidenOrder": "LeidenOrder",
        "GraphBrewOrder": "GraphBrewOrder",
        "LeidenDFS": "LeidenDFS",
        "LeidenDFSHub": "LeidenDFSHub",
        "LeidenDFSSize": "LeidenDFSSize",
        "LeidenBFS": "LeidenBFS",
        "LeidenHybrid": "LeidenHybrid",
    }
    
    for graph in graphs:
        log(f"\n{'='*60}")
        log(f"Graph: {graph.name} ({graph.size_mb:.1f}MB)")
        log(f"{'='*60}")
        
        analysis = GraphBruteForceAnalysis(
            graph=graph.name,
            size_mb=graph.size_mb,
            modularity=0.0,
            num_communities=0,
            num_subcommunities_analyzed=0
        )
        
        # Step 1: Run AdaptiveOrder to get subcommunity info
        binary = os.path.join(bin_dir, benchmark)
        if not os.path.exists(binary):
            log(f"  Binary not found: {binary}", "ERROR")
            analysis.success = False
            analysis.error = "Binary not found"
            results.append(analysis)
            continue
        
        sym_flag = "-s" if graph.is_symmetric else ""
        cmd = f"{binary} -f {graph.path} {sym_flag} -o 15 -n 1"
        
        success, stdout, stderr = run_command(cmd, timeout)
        if not success:
            log(f"  AdaptiveOrder failed", "ERROR")
            analysis.success = False
            analysis.error = "AdaptiveOrder failed"
            results.append(analysis)
            continue
        
        output = stdout + stderr
        modularity, num_communities, subcommunities, algo_distribution = parse_adaptive_output(output)
        
        analysis.modularity = modularity
        analysis.num_communities = num_communities
        
        log(f"  Modularity: {modularity:.4f}")
        log(f"  Total communities: {num_communities}")
        log(f"  Subcommunities with features: {len(subcommunities)}")
        
        if not subcommunities:
            log(f"  No subcommunities to analyze", "WARN")
            results.append(analysis)
            continue
        
        # Sort by size and take largest subcommunities for analysis
        subcommunities_sorted = sorted(subcommunities, key=lambda s: s.nodes, reverse=True)
        subcommunities_to_test = subcommunities_sorted[:max_subcommunities]
        
        log(f"  Testing top {len(subcommunities_to_test)} largest subcommunities")
        log(f"\n  {'Comm':<6} {'Nodes':<8} {'Edges':<10} {'Adaptive':<12} {'BestTime':<12} {'BestCache':<12} {'TimeRatio':<10} {'CacheRatio':<10}")
        log(f"  {'-'*80}")
        
        subcommunity_results = []
        
        for idx, subcomm in enumerate(subcommunities_to_test):
            # Normalize adaptive algorithm name
            adaptive_algo_raw = subcomm.selected_algorithm
            adaptive_algo = adaptive_to_algo_name.get(adaptive_algo_raw, adaptive_algo_raw)
            
            # Initialize result
            sc_result = SubcommunityBruteForceResult(
                community_id=subcomm.community_id,
                nodes=subcomm.nodes,
                edges=subcomm.edges,
                density=subcomm.density,
                degree_variance=subcomm.degree_variance,
                hub_concentration=subcomm.hub_concentration,
                adaptive_algorithm=adaptive_algo
            )
            
            # Run all algorithms on this graph (we approximate subcommunity performance
            # by using graph-level performance since we can't easily isolate subcommunities)
            algo_times = {}
            algo_cache = {}
            
            for algo_id in test_algorithms:
                algo_name = ALGORITHMS.get(algo_id, f"ALGO_{algo_id}")
                
                # Run benchmark for time
                cmd_bench = f"{binary} -f {graph.path} {sym_flag} -o {algo_id} -n {num_trials}"
                success_bench, stdout_bench, stderr_bench = run_command(cmd_bench, timeout)
                
                if success_bench:
                    parsed = parse_benchmark_output(stdout_bench + stderr_bench)
                    algo_times[algo_name] = parsed.get('average_time', parsed.get('trial_time', 999999))
                else:
                    algo_times[algo_name] = 999999  # Failed
                
                # Run cache simulation
                sim_binary = os.path.join(bin_sim_dir, benchmark)
                if os.path.exists(sim_binary):
                    cmd_sim = f"{sim_binary} -f {graph.path} {sym_flag} -o {algo_id} -n 1"
                    success_sim, stdout_sim, stderr_sim = run_command(cmd_sim, timeout_sim)
                    
                    if success_sim:
                        cache_parsed = parse_cache_output(stdout_sim + stderr_sim)
                        algo_cache[algo_name] = {
                            'l1': cache_parsed.get('l1_hit_rate', 0),
                            'l2': cache_parsed.get('l2_hit_rate', 0),
                            'l3': cache_parsed.get('l3_hit_rate', 0)
                        }
                    else:
                        algo_cache[algo_name] = {'l1': 0, 'l2': 0, 'l3': 0}
                else:
                    algo_cache[algo_name] = {'l1': 0, 'l2': 0, 'l3': 0}
            
            # Store all results
            for algo_name in algo_times:
                sc_result.all_results[algo_name] = {
                    'time': algo_times.get(algo_name, 999999),
                    'l1_hit': algo_cache.get(algo_name, {}).get('l1', 0),
                    'l2_hit': algo_cache.get(algo_name, {}).get('l2', 0),
                    'l3_hit': algo_cache.get(algo_name, {}).get('l3', 0)
                }
            
            # Find best by time
            valid_times = {k: v for k, v in algo_times.items() if v < 999999}
            if valid_times:
                best_time_algo = min(valid_times, key=valid_times.get)
                sc_result.best_time_algorithm = best_time_algo
                sc_result.best_time = valid_times[best_time_algo]
                sc_result.best_time_l1_hit = algo_cache.get(best_time_algo, {}).get('l1', 0)
                sc_result.best_time_l2_hit = algo_cache.get(best_time_algo, {}).get('l2', 0)
                sc_result.best_time_l3_hit = algo_cache.get(best_time_algo, {}).get('l3', 0)
            
            # Find best by cache (L1 hit rate)
            valid_cache = {k: v.get('l1', 0) for k, v in algo_cache.items() if v.get('l1', 0) > 0}
            if valid_cache:
                best_cache_algo = max(valid_cache, key=valid_cache.get)
                sc_result.best_cache_algorithm = best_cache_algo
                sc_result.best_cache_l1_hit = valid_cache[best_cache_algo]
                sc_result.best_cache_l2_hit = algo_cache.get(best_cache_algo, {}).get('l2', 0)
                sc_result.best_cache_l3_hit = algo_cache.get(best_cache_algo, {}).get('l3', 0)
                sc_result.best_cache_time = algo_times.get(best_cache_algo, 0)
            
            # Get adaptive algorithm results
            adaptive_algo = sc_result.adaptive_algorithm
            sc_result.adaptive_time = algo_times.get(adaptive_algo, 999999)
            sc_result.adaptive_l1_hit = algo_cache.get(adaptive_algo, {}).get('l1', 0)
            sc_result.adaptive_l2_hit = algo_cache.get(adaptive_algo, {}).get('l2', 0)
            sc_result.adaptive_l3_hit = algo_cache.get(adaptive_algo, {}).get('l3', 0)
            
            # Calculate comparison metrics
            if sc_result.best_time > 0:
                sc_result.adaptive_vs_best_time_ratio = sc_result.best_time / sc_result.adaptive_time if sc_result.adaptive_time > 0 else 0
            if sc_result.best_cache_l1_hit > 0:
                sc_result.adaptive_vs_best_cache_ratio = sc_result.adaptive_l1_hit / sc_result.best_cache_l1_hit
            
            sc_result.adaptive_is_best_time = (adaptive_algo == sc_result.best_time_algorithm)
            sc_result.adaptive_is_best_cache = (adaptive_algo == sc_result.best_cache_algorithm)
            
            # Calculate ranks
            sorted_by_time = sorted(valid_times.items(), key=lambda x: x[1])
            sorted_by_cache = sorted(valid_cache.items(), key=lambda x: -x[1])
            
            for rank, (algo, _) in enumerate(sorted_by_time, 1):
                if algo == adaptive_algo:
                    sc_result.adaptive_rank_time = rank
                    break
            
            for rank, (algo, _) in enumerate(sorted_by_cache, 1):
                if algo == adaptive_algo:
                    sc_result.adaptive_rank_cache = rank
                    break
            
            subcommunity_results.append(sc_result)
            
            # Print row
            time_ratio_str = f"{sc_result.adaptive_vs_best_time_ratio:.3f}" if sc_result.adaptive_vs_best_time_ratio > 0 else "N/A"
            cache_ratio_str = f"{sc_result.adaptive_vs_best_cache_ratio:.3f}" if sc_result.adaptive_vs_best_cache_ratio > 0 else "N/A"
            best_marker_time = "✓" if sc_result.adaptive_is_best_time else ""
            best_marker_cache = "✓" if sc_result.adaptive_is_best_cache else ""
            
            log(f"  {subcomm.community_id:<6} {subcomm.nodes:<8} {subcomm.edges:<10} {adaptive_algo:<12} {sc_result.best_time_algorithm:<12} {sc_result.best_cache_algorithm:<12} {time_ratio_str:<10} {cache_ratio_str:<10}")
            
            # Only test first subcommunity per graph to save time (algorithms are same across subcommunities)
            # The key insight is that algorithm performance on the whole graph approximates subcommunity performance
            break
        
        analysis.subcommunity_results = subcommunity_results
        analysis.num_subcommunities_analyzed = len(subcommunity_results)
        
        # Calculate summary statistics
        if subcommunity_results:
            correct_time = sum(1 for r in subcommunity_results if r.adaptive_is_best_time)
            correct_cache = sum(1 for r in subcommunity_results if r.adaptive_is_best_cache)
            top3_time = sum(1 for r in subcommunity_results if r.adaptive_rank_time <= 3)
            top3_cache = sum(1 for r in subcommunity_results if r.adaptive_rank_cache <= 3)
            
            n = len(subcommunity_results)
            analysis.adaptive_correct_time_pct = 100 * correct_time / n
            analysis.adaptive_correct_cache_pct = 100 * correct_cache / n
            analysis.adaptive_top3_time_pct = 100 * top3_time / n
            analysis.adaptive_top3_cache_pct = 100 * top3_cache / n
            
            valid_time_ratios = [r.adaptive_vs_best_time_ratio for r in subcommunity_results if r.adaptive_vs_best_time_ratio > 0]
            valid_cache_ratios = [r.adaptive_vs_best_cache_ratio for r in subcommunity_results if r.adaptive_vs_best_cache_ratio > 0]
            
            if valid_time_ratios:
                analysis.avg_time_ratio = sum(valid_time_ratios) / len(valid_time_ratios)
            if valid_cache_ratios:
                analysis.avg_cache_ratio = sum(valid_cache_ratios) / len(valid_cache_ratios)
        
        results.append(analysis)
        
        # Print detailed table for this graph
        log(f"\n  === Detailed Algorithm Comparison for {graph.name} ===")
        if subcommunity_results and subcommunity_results[0].all_results:
            log(f"  {'Algorithm':<16} {'Time(s)':<12} {'L1 Hit%':<10} {'L2 Hit%':<10} {'L3 Hit%':<10} {'Notes':<20}")
            log(f"  {'-'*78}")
            
            all_res = subcommunity_results[0].all_results
            sorted_algos = sorted(all_res.items(), key=lambda x: x[1].get('time', 999999))
            
            adaptive_algo = subcommunity_results[0].adaptive_algorithm
            best_time_algo = subcommunity_results[0].best_time_algorithm
            best_cache_algo = subcommunity_results[0].best_cache_algorithm
            
            for algo_name, data in sorted_algos:
                time_val = data.get('time', 0)
                l1 = data.get('l1_hit', 0)
                l2 = data.get('l2_hit', 0)
                l3 = data.get('l3_hit', 0)
                
                notes = []
                if algo_name == adaptive_algo:
                    notes.append("ADAPTIVE")
                if algo_name == best_time_algo:
                    notes.append("BEST-TIME")
                if algo_name == best_cache_algo:
                    notes.append("BEST-CACHE")
                
                time_str = f"{time_val:.4f}" if time_val < 999999 else "FAILED"
                log(f"  {algo_name:<16} {time_str:<12} {l1:<10.2f} {l2:<10.2f} {l3:<10.2f} {', '.join(notes):<20}")
        
        log(f"\n  Summary:")
        log(f"    Adaptive chose best for time: {analysis.adaptive_correct_time_pct:.1f}%")
        log(f"    Adaptive chose best for cache: {analysis.adaptive_correct_cache_pct:.1f}%")
        log(f"    Adaptive in top 3 for time: {analysis.adaptive_top3_time_pct:.1f}%")
        log(f"    Adaptive in top 3 for cache: {analysis.adaptive_top3_cache_pct:.1f}%")
        log(f"    Avg time ratio (best/adaptive): {analysis.avg_time_ratio:.3f}")
        log(f"    Avg cache ratio (adaptive/best): {analysis.avg_cache_ratio:.3f}")
    
    # Save results
    results_file = os.path.join(output_dir, f"brute_force_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    results_data = []
    for analysis in results:
        data = {
            "graph": analysis.graph,
            "size_mb": analysis.size_mb,
            "modularity": analysis.modularity,
            "num_communities": analysis.num_communities,
            "num_subcommunities_analyzed": analysis.num_subcommunities_analyzed,
            "adaptive_correct_time_pct": analysis.adaptive_correct_time_pct,
            "adaptive_correct_cache_pct": analysis.adaptive_correct_cache_pct,
            "adaptive_top3_time_pct": analysis.adaptive_top3_time_pct,
            "adaptive_top3_cache_pct": analysis.adaptive_top3_cache_pct,
            "avg_time_ratio": analysis.avg_time_ratio,
            "avg_cache_ratio": analysis.avg_cache_ratio,
            "success": analysis.success,
            "error": analysis.error,
            "subcommunity_results": [
                {
                    "community_id": r.community_id,
                    "nodes": r.nodes,
                    "edges": r.edges,
                    "density": r.density,
                    "degree_variance": r.degree_variance,
                    "hub_concentration": r.hub_concentration,
                    "adaptive_algorithm": r.adaptive_algorithm,
                    "adaptive_time": r.adaptive_time,
                    "adaptive_l1_hit": r.adaptive_l1_hit,
                    "best_time_algorithm": r.best_time_algorithm,
                    "best_time": r.best_time,
                    "best_cache_algorithm": r.best_cache_algorithm,
                    "best_cache_l1_hit": r.best_cache_l1_hit,
                    "adaptive_vs_best_time_ratio": r.adaptive_vs_best_time_ratio,
                    "adaptive_vs_best_cache_ratio": r.adaptive_vs_best_cache_ratio,
                    "adaptive_is_best_time": r.adaptive_is_best_time,
                    "adaptive_is_best_cache": r.adaptive_is_best_cache,
                    "adaptive_rank_time": r.adaptive_rank_time,
                    "adaptive_rank_cache": r.adaptive_rank_cache,
                    "all_results": r.all_results
                }
                for r in analysis.subcommunity_results
            ]
        }
        results_data.append(data)
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    log(f"\n{'='*60}")
    log(f"Brute-force analysis saved to: {results_file}")
    
    # Print overall summary
    log(f"\n{'='*60}")
    log("OVERALL SUMMARY")
    log(f"{'='*60}")
    
    successful = [r for r in results if r.success]
    if successful:
        avg_correct_time = sum(r.adaptive_correct_time_pct for r in successful) / len(successful)
        avg_correct_cache = sum(r.adaptive_correct_cache_pct for r in successful) / len(successful)
        avg_top3_time = sum(r.adaptive_top3_time_pct for r in successful) / len(successful)
        avg_top3_cache = sum(r.adaptive_top3_cache_pct for r in successful) / len(successful)
        
        log(f"Graphs analyzed: {len(successful)}")
        log(f"Avg % adaptive chose best for time: {avg_correct_time:.1f}%")
        log(f"Avg % adaptive chose best for cache: {avg_correct_cache:.1f}%")
        log(f"Avg % adaptive in top 3 for time: {avg_top3_time:.1f}%")
        log(f"Avg % adaptive in top 3 for cache: {avg_top3_cache:.1f}%")
    
    return results

def update_zero_weights(
    weights_file: str, 
    benchmark_results: List[BenchmarkResult],
    cache_results: List[CacheResult] = None,
    reorder_results: List[ReorderResult] = None,
    graphs_dir: str = DEFAULT_GRAPHS_DIR
):
    """
    Comprehensive update of weights based on correlation analysis.
    
    This function:
    1. Identifies weights that are 0.0 or very small
    2. Calculates correlations between graph features and algorithm performance
    3. Updates weights based on actual correlation patterns
    4. Normalizes negative values appropriately (negative = penalizes that feature)
    5. Ensures weights are within reasonable bounds
    """
    log_section("Phase 5: Update Zero Weights (Comprehensive)")
    
    with open(weights_file) as f:
        weights = json.load(f)
    
    # Collect graph features for correlation analysis
    graph_features = {}
    for graph_dir in os.listdir(graphs_dir):
        graph_path = os.path.join(graphs_dir, graph_dir)
        if not os.path.isdir(graph_path):
            continue
        
        # Try to load graph info
        info_file = os.path.join(graph_path, "info.json")
        if os.path.exists(info_file):
            with open(info_file) as f:
                info = json.load(f)
                nodes = info.get("nodes", 0)
                edges = info.get("edges", 0)
        else:
            # Estimate from file size
            mtx_path = os.path.join(graph_path, "graph.mtx")
            if os.path.exists(mtx_path):
                size_mb = os.path.getsize(mtx_path) / (1024 * 1024)
                # Rough estimate: ~100 bytes per edge for mtx format
                edges = int(size_mb * 1024 * 1024 / 100)
                nodes = int(edges / 5)  # Assume average degree ~10
            else:
                continue
        
        if nodes > 0 and edges > 0:
            density = edges / (nodes * (nodes - 1) / 2) if nodes > 1 else 0
            avg_degree = 2 * edges / nodes if nodes > 0 else 0
            
            graph_features[graph_dir] = {
                "log_nodes": math.log10(nodes) if nodes > 0 else 0,
                "log_edges": math.log10(edges) if edges > 0 else 0,
                "density": density,
                "avg_degree": avg_degree / 100,  # Normalized
                "nodes": nodes,
                "edges": edges
            }
    
    log(f"Loaded features for {len(graph_features)} graphs")
    
    # Process each algorithm
    updated_count = 0
    for algo_name, algo_weights in weights.items():
        if algo_name.startswith("_") or not isinstance(algo_weights, dict):
            continue
        
        # Get results for this algorithm
        algo_results = [r for r in benchmark_results if r.algorithm_name == algo_name and r.success]
        
        if not algo_results:
            continue
        
        # Group results by graph
        graph_speedups = {}
        for r in algo_results:
            if r.graph not in graph_speedups:
                graph_speedups[r.graph] = []
            graph_speedups[r.graph].append(r.speedup)
        
        # Calculate average speedup per graph
        avg_speedups = {g: sum(s)/len(s) for g, s in graph_speedups.items()}
        
        # Prepare data for correlation
        graphs_with_data = [g for g in avg_speedups if g in graph_features]
        
        if len(graphs_with_data) < 3:
            continue
        
        speedup_list = [avg_speedups[g] for g in graphs_with_data]
        
        updates = {}
        explanations = {}
        
        # Calculate correlations for each feature
        feature_correlations = {}
        for feature_name in ["log_nodes", "log_edges", "density", "avg_degree"]:
            feature_values = [graph_features[g][feature_name] for g in graphs_with_data]
            corr = calculate_correlation(feature_values, speedup_list)
            feature_correlations[feature_name] = corr
        
        # Update weights based on correlations
        weight_mapping = {
            "w_log_nodes": "log_nodes",
            "w_log_edges": "log_edges", 
            "w_density": "density",
            "w_avg_degree": "avg_degree"
        }
        
        for weight_name, feature_name in weight_mapping.items():
            current_val = algo_weights.get(weight_name, 0)
            corr = feature_correlations.get(feature_name, 0)
            
            # Only update if current value is zero/tiny or if we have strong correlation
            if abs(current_val) < 0.0001 or (abs(corr) > 0.3 and abs(current_val) < abs(corr * 0.001)):
                # Scale correlation to weight magnitude
                new_val = corr * 0.001  # Keep weights small
                updates[weight_name] = round(new_val, 6)
                explanations[weight_name] = f"corr={corr:.3f}"
        
        # Update cache-related weights if we have cache data
        if cache_results:
            algo_cache = [r for r in cache_results if r.algorithm_name == algo_name and r.success]
            if algo_cache:
                avg_l1 = sum(r.l1_hit_rate for r in algo_cache) / len(algo_cache)
                avg_l2 = sum(r.l2_hit_rate for r in algo_cache) / len(algo_cache)
                avg_l3 = sum(r.l3_hit_rate for r in algo_cache) / len(algo_cache)
                
                # Update cache impact weights if they're zero
                if algo_weights.get("cache_l1_impact", 0) == 0 and avg_l1 > 0:
                    updates["cache_l1_impact"] = round(0.01 * (avg_l1 / 100 - 0.5), 6)
                    explanations["cache_l1_impact"] = f"avg_l1={avg_l1:.1f}%"
                
                if algo_weights.get("cache_l2_impact", 0) == 0 and avg_l2 > 0:
                    updates["cache_l2_impact"] = round(0.005 * (avg_l2 / 100 - 0.5), 6)
                    explanations["cache_l2_impact"] = f"avg_l2={avg_l2:.1f}%"
        
        # Update reorder time weight if we have reorder data
        if reorder_results:
            algo_reorder = [r for r in reorder_results if r.algorithm_name == algo_name and r.success]
            if algo_reorder:
                avg_reorder_time = sum(r.reorder_time for r in algo_reorder) / len(algo_reorder)
                
                if algo_weights.get("w_reorder_time", 0) == 0 and avg_reorder_time > 0:
                    # Penalize longer reorder times (negative weight)
                    updates["w_reorder_time"] = round(-0.001 * avg_reorder_time, 6)
                    explanations["w_reorder_time"] = f"avg_time={avg_reorder_time:.2f}s"
        
        # Apply updates
        if updates:
            for key, val in updates.items():
                algo_weights[key] = val
            updated_count += 1
            log(f"{algo_name}: Updated {len(updates)} weights")
            for key, val in updates.items():
                exp = explanations.get(key, "")
                sign = "+" if val >= 0 else ""
                log(f"    {key}: {sign}{val} ({exp})")
    
    # Add metadata about this update
    weights["_zero_weight_update"] = {
        "timestamp": datetime.now().isoformat(),
        "algorithms_updated": updated_count,
        "graphs_analyzed": len(graph_features),
        "benchmark_results_used": len(benchmark_results),
        "note": "Negative weights penalize the feature; positive weights reward it"
    }
    
    # Save updated weights
    with open(weights_file, 'w') as f:
        json.dump(weights, f, indent=2)
    
    log(f"\nUpdated {updated_count} algorithms")
    log(f"Weights saved to: {weights_file}")
    log("\nNote: Negative weights (like -0.0) are NORMAL and mean that feature is penalized")
    
    # Backup with timestamp and sync to scripts folder
    backup_and_sync_weights(weights_file)


def compute_benchmark_weights(
    weights_file: str,
    benchmark_results: List[BenchmarkResult]
) -> None:
    """
    Compute per-benchmark weights based on algorithm performance on each benchmark.
    
    For each algorithm, calculates how well it performs on pr, bfs, cc, sssp, bc
    relative to other algorithms. Higher weight = algorithm is better for that benchmark.
    """
    log("Computing benchmark-specific weights...")
    
    # Load existing weights
    with open(weights_file) as f:
        weights = json.load(f)
    
    # Group results by benchmark
    benchmarks = {}
    for r in benchmark_results:
        if not r.success:
            continue
        if r.benchmark not in benchmarks:
            benchmarks[r.benchmark] = []
        benchmarks[r.benchmark].append(r)
    
    if not benchmarks:
        log("No benchmark results to compute weights from")
        return
    
    # For each algorithm, compute relative performance on each benchmark
    for algo_name, algo_weights in weights.items():
        if algo_name.startswith("_") or not isinstance(algo_weights, dict):
            continue
        
        if "benchmark_weights" not in algo_weights:
            algo_weights["benchmark_weights"] = {}
        
        for benchmark, results in benchmarks.items():
            # Get this algorithm's results for this benchmark
            algo_results = [r for r in results if r.algorithm_name == algo_name]
            
            if not algo_results:
                continue
            
            # Calculate average speedup for this algorithm on this benchmark
            avg_speedup = sum(r.speedup for r in algo_results) / len(algo_results)
            
            # Calculate average speedup across ALL algorithms for this benchmark
            all_speedups = [r.speedup for r in results]
            overall_avg = sum(all_speedups) / len(all_speedups) if all_speedups else 1.0
            
            # Relative performance: >1 means better than average, <1 means worse
            relative_perf = avg_speedup / overall_avg if overall_avg > 0 else 1.0
            
            # Convert to weight: clamp to [0.5, 2.0] range
            bw = max(0.5, min(2.0, relative_perf))
            
            algo_weights["benchmark_weights"][benchmark] = round(bw, 3)
        
        weights[algo_name] = algo_weights
    
    # Add metadata
    weights["_benchmark_weights_update"] = {
        "timestamp": datetime.now().isoformat(),
        "benchmarks_analyzed": list(benchmarks.keys()),
        "results_used": len(benchmark_results)
    }
    
    # Save
    with open(weights_file, 'w') as f:
        json.dump(weights, f, indent=2)
    
    log(f"Updated benchmark weights for {len(benchmarks)} benchmarks")
    
    # Show summary
    for benchmark in benchmarks:
        algo_weights_for_bm = []
        for algo, w in weights.items():
            if algo.startswith("_") or not isinstance(w, dict):
                continue
            bw = w.get("benchmark_weights", {}).get(benchmark, 1.0)
            algo_weights_for_bm.append((algo, bw))
        
        # Sort by weight, show top 3
        algo_weights_for_bm.sort(key=lambda x: -x[1])
        top3 = algo_weights_for_bm[:3]
        log(f"  {benchmark}: best={top3[0][0]}({top3[0][1]:.2f}), {top3[1][0]}({top3[1][1]:.2f}), {top3[2][0]}({top3[2][1]:.2f})")


def generate_per_type_weights(
    base_weights_file: str,
    benchmark_results: List[BenchmarkResult],
    graphs_dir: str,
    output_dir: str = "scripts"
) -> Dict[str, str]:
    """
    Generate per-graph-type weight files using auto-clustering.
    
    This function:
    1. Groups benchmark results by auto-detected graph type (using feature clustering)
    2. For each cluster, computes specialized weights based on results for graphs in that cluster
    3. Saves weights to type-specific files (e.g., scripts/weights/type_0.json)
    
    Args:
        base_weights_file: Path to the base weights file (used as starting point)
        benchmark_results: List of benchmark results from all phases
        graphs_dir: Directory containing graph files
        output_dir: Directory to save per-type weight files
    
    Returns:
        Dict mapping graph type to weight file path
    """
    log("Generating per-graph-type weight files...")
    
    # Load graph properties cache if not already loaded
    load_graph_properties_cache(output_dir)
    
    # Load base weights as template
    with open(base_weights_file) as f:
        base_weights = json.load(f)
    
    # Group benchmark results by graph type
    type_results: Dict[str, List[BenchmarkResult]] = {t: [] for t in ALL_GRAPH_TYPES}
    graph_type_cache: Dict[str, str] = {}
    detection_method: Dict[str, str] = {}  # Track how each graph type was detected
    
    for result in benchmark_results:
        if not result.success:
            continue
        
        # Get or detect graph type from properties (falls back to name heuristic)
        if result.graph not in graph_type_cache:
            # Check if we have properties for this graph
            has_properties = (result.graph in _graph_properties_cache and 
                            any(k in _graph_properties_cache[result.graph] 
                                for k in ['modularity', 'degree_variance', 'hub_concentration', 'graph_type']))
            
            # Use properties-based detection, with name fallback
            graph_type = get_graph_type_from_properties(result.graph, fallback_to_name=True)
            graph_type_cache[result.graph] = graph_type
            detection_method[result.graph] = "properties" if has_properties else "name"
        
        graph_type = graph_type_cache[result.graph]
        type_results[graph_type].append(result)
    
    # Log distribution with detection method
    log("  Graph type distribution:")
    for gtype, results in type_results.items():
        if results:
            unique_graphs = set(r.graph for r in results)
            # Count by detection method
            from_props = sum(1 for g in unique_graphs if detection_method.get(g) == "properties")
            from_name = sum(1 for g in unique_graphs if detection_method.get(g) == "name")
            log(f"    {gtype}: {len(unique_graphs)} graphs ({from_props} from properties, {from_name} from name), {len(results)} results")
    
    # Generate weights for each graph type
    output_files = {}
    
    for graph_type in ALL_GRAPH_TYPES:
        results = type_results[graph_type]
        
        if not results and graph_type != GRAPH_TYPE_GENERIC:
            # No data for this type - skip (will fall back to generic)
            continue
        
        # Start with base weights
        type_weights = copy.deepcopy(base_weights)
        
        # Update metadata
        type_weights["_graph_type"] = graph_type
        type_weights["_generation_date"] = datetime.now().isoformat()
        type_weights["_graphs_used"] = list(set(r.graph for r in results)) if results else []
        
        if results:
            # Compute type-specific algorithm performance
            algo_performance: Dict[str, Dict] = {}
            
            for result in results:
                if result.algorithm_name not in algo_performance:
                    algo_performance[result.algorithm_name] = {
                        'speedups': [],
                        'times': [],
                        'wins': 0,
                        'count': 0
                    }
                algo_performance[result.algorithm_name]['speedups'].append(result.speedup)
                algo_performance[result.algorithm_name]['times'].append(result.avg_time)
                algo_performance[result.algorithm_name]['count'] += 1
            
            # Compute wins (best speedup for each graph/benchmark combo)
            from collections import defaultdict
            best_for_combo: Dict[tuple, tuple] = {}  # (graph, benchmark) -> (algo, speedup)
            
            for result in results:
                key = (result.graph, result.benchmark)
                if key not in best_for_combo or result.speedup > best_for_combo[key][1]:
                    best_for_combo[key] = (result.algorithm_name, result.speedup)
            
            for (graph, benchmark), (winner, _) in best_for_combo.items():
                if winner in algo_performance:
                    algo_performance[winner]['wins'] += 1
            
            # Adjust biases based on win rate for this graph type
            total_combos = len(best_for_combo)
            
            for algo_name, perf in algo_performance.items():
                if algo_name not in type_weights or algo_name.startswith("_"):
                    continue
                
                win_rate = perf['wins'] / total_combos if total_combos > 0 else 0
                avg_speedup = sum(perf['speedups']) / len(perf['speedups']) if perf['speedups'] else 1.0
                
                # Adjust bias based on win rate and average speedup
                current_bias = type_weights[algo_name].get('bias', 0.5)
                
                # Bias adjustment: scale by win rate (0-50%) and avg speedup
                adjustment = (win_rate - 0.1) * 0.5 + (avg_speedup - 1.0) * 0.1
                new_bias = max(0.1, min(1.0, current_bias + adjustment))
                
                type_weights[algo_name]['bias'] = round(new_bias, 4)
                
                # Store performance metadata
                type_weights[algo_name]['_metadata'] = type_weights[algo_name].get('_metadata', {})
                type_weights[algo_name]['_metadata'].update({
                    f'{graph_type}_win_rate': round(win_rate, 4),
                    f'{graph_type}_avg_speedup': round(avg_speedup, 4),
                    f'{graph_type}_sample_count': perf['count']
                })
        
        # Save to type-specific file
        if graph_type == GRAPH_TYPE_GENERIC:
            output_file = base_weights_file  # Generic is the base file
        else:
            base_name = os.path.basename(base_weights_file).replace('.json', '')
            output_file = os.path.join(output_dir, f"{base_name}_{graph_type}.json")
        
        with open(output_file, 'w') as f:
            json.dump(type_weights, f, indent=2)
        
        output_files[graph_type] = output_file
        log(f"  Saved {graph_type} weights to: {output_file}")
    
    # Also copy to scripts directory for C++ to find
    scripts_dir = "scripts"
    if output_dir != scripts_dir:
        for graph_type, weight_file in output_files.items():
            if graph_type != GRAPH_TYPE_GENERIC:
                dest = os.path.join(scripts_dir, os.path.basename(weight_file))
                shutil.copy2(weight_file, dest)
                log(f"  Synced to: {dest}")
    
    return output_files


# ============================================================================
# Phase 5b: Iterative Weight Training with Feedback Loop
# ============================================================================

@dataclass
class TrainingIterationResult:
    """Result from one iteration of training."""
    iteration: int
    accuracy_time_pct: float
    accuracy_cache_pct: float
    accuracy_top3_pct: float
    avg_time_ratio: float
    avg_cache_ratio: float
    graphs_tested: int
    weights_updated: int
    
@dataclass
class TrainingResult:
    """Final result from iterative training."""
    final_accuracy_time_pct: float
    final_accuracy_cache_pct: float
    final_accuracy_top3_pct: float
    iterations_run: int
    target_accuracy: float
    target_reached: bool
    iteration_history: List[TrainingIterationResult] = field(default_factory=list)
    best_weights_iteration: int = 0
    best_weights_file: str = ""


def initialize_enhanced_weights(weights_file: str, algorithms: List[str] = None) -> Dict:
    """
    Initialize or upgrade weights file with enhanced feature support.
    
    Creates a new weights file with all extended features, or upgrades
    an existing weights file to include new features.
    """
    default_algorithms = [
        "Original", "RCMOrder", "DegSort", "HubSort", "HubClusterDBG",
        "HubClusterDeg", "Sort", "Gorder", "RabbitOrder", "MineOrder",
        "LDGOrder", "Corder", "SlashBurn", "LeidenDFS", "LeidenBFS",
        "Adaptive", "LeidenDFSHub", "LeidenBFSHub", "MAP"
    ]
    
    if algorithms is None:
        algorithms = default_algorithms
    
    # Default template for algorithm weights
    def make_default_weights(algo_name: str) -> Dict:
        # Baseline biases based on prior experiments
        base_biases = {
            "LeidenDFS": 3.5, "LeidenDFSHub": 3.3, "LeidenBFS": 3.2,
            "LeidenBFSHub": 3.0, "RabbitOrder": 2.5, "Gorder": 2.3,
            "HubClusterDBG": 2.0, "HubSort": 1.8, "DegSort": 1.5,
            "RCMOrder": 1.2, "Corder": 1.0, "Original": 0.5
        }
        return {
            "bias": base_biases.get(algo_name, 1.0),
            "w_modularity": 0.1,
            "w_density": 0.05,
            "w_degree_variance": 0.03,
            "w_hub_concentration": 0.05,
            "w_log_nodes": 0.02,
            "w_log_edges": 0.02,
            # New extended features
            "w_clustering_coeff": 0.04,
            "w_avg_path_length": 0.02,
            "w_diameter": 0.01,
            "w_community_count": 0.03,
            # Per-benchmark multipliers
            "benchmark_weights": {
                "pr": 1.0,
                "bfs": 1.0,
                "cc": 1.0,
                "sssp": 1.0,
                "bc": 1.0
            }
        }
    
    # Try to load existing weights
    weights = {}
    if os.path.exists(weights_file):
        try:
            with open(weights_file) as f:
                weights = json.load(f)
        except (json.JSONDecodeError, IOError):
            weights = {}
    
    # Initialize or upgrade each algorithm
    for algo in algorithms:
        if algo not in weights:
            weights[algo] = make_default_weights(algo)
        else:
            # Upgrade existing weights with missing fields
            existing = weights[algo]
            if isinstance(existing, dict):
                defaults = make_default_weights(algo)
                for key, value in defaults.items():
                    if key not in existing:
                        existing[key] = value
                    elif key == "benchmark_weights" and isinstance(existing.get(key), dict):
                        # Merge benchmark weights
                        for bk, bv in value.items():
                            if bk not in existing[key]:
                                existing[key][bk] = bv
                weights[algo] = existing
    
    # Add metadata
    if '_metadata' not in weights:
        weights['_metadata'] = {}
    weights['_metadata']['enhanced_features'] = True
    weights['_metadata']['last_updated'] = datetime.now().isoformat()
    
    # Save updated weights
    with open(weights_file, 'w') as f:
        json.dump(weights, f, indent=2)
    
    return weights


def train_adaptive_weights_large_scale(
    graphs: List[GraphInfo],
    bin_dir: str,
    bin_sim_dir: str,
    output_dir: str,
    weights_file: str,
    benchmarks: List[str] = None,
    target_accuracy: float = 80.0,
    max_iterations: int = 10,
    batch_size: int = 8,
    timeout: int = TIMEOUT_REORDER,
    timeout_sim: int = TIMEOUT_SIM,
    num_trials: int = 2,
    learning_rate: float = 0.15,
    algorithms: List[int] = None
) -> TrainingResult:
    """
    Large-scale training with batching and multi-benchmark support.
    
    This extends train_adaptive_weights_iterative with:
    - Batch processing for organized training (sequential execution for accurate timing)
    - Multi-benchmark training (pr, bfs, cc, etc.)
    - Progressive learning rate decay
    - Cross-validation between batches
    
    NOTE: All graph algorithms and reordering operations run sequentially (not in parallel)
    to ensure accurate performance measurements. Parallel execution would cause CPU
    contention and skew timing results.
    
    Args:
        graphs: List of all graphs to train on
        benchmarks: List of benchmarks to train on (default: ['pr', 'bfs', 'cc'])
        batch_size: Number of graphs per batch (processed sequentially)
        Other args same as train_adaptive_weights_iterative
    """
    log_section(f"Large-Scale Adaptive Training ({len(graphs)} graphs)")
    log("Note: All algorithms run sequentially for accurate performance measurement")
    
    if benchmarks is None:
        benchmarks = ['pr', 'bfs', 'cc']
    
    if algorithms is None:
        algorithms = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13]
    
    # Setup
    training_dir = os.path.join(output_dir, f"large_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(training_dir, exist_ok=True)
    
    # Initialize weights
    initialize_enhanced_weights(weights_file)
    
    result = TrainingResult(
        final_accuracy_time_pct=0.0,
        final_accuracy_cache_pct=0.0,
        final_accuracy_top3_pct=0.0,
        iterations_run=0,
        target_accuracy=target_accuracy,
        target_reached=False
    )
    
    best_accuracy = 0.0
    
    # Split graphs into batches (for organized progress tracking, not parallel execution)
    import random
    shuffled_graphs = graphs.copy()
    random.shuffle(shuffled_graphs)
    batches = [shuffled_graphs[i:i+batch_size] for i in range(0, len(shuffled_graphs), batch_size)]
    
    log(f"Created {len(batches)} batches of ~{batch_size} graphs each (sequential processing)")
    log(f"Training on benchmarks: {benchmarks}")
    
    for iteration in range(1, max_iterations + 1):
        # Learning rate decay
        current_lr = learning_rate * (0.9 ** (iteration - 1))
        
        log(f"\n{'='*60}")
        log(f"ITERATION {iteration}/{max_iterations} (LR: {current_lr:.4f})")
        log(f"{'='*60}")
        
        iteration_accuracy = []
        
        for benchmark in benchmarks:
            log(f"\n--- Benchmark: {benchmark.upper()} ---")
            
            for batch_idx, batch in enumerate(batches):
                log(f"\nBatch {batch_idx+1}/{len(batches)}: {[g.name for g in batch]}")
                
                # Run training iteration on this batch
                iter_result = train_adaptive_weights_iterative(
                    graphs=batch,
                    bin_dir=bin_dir,
                    bin_sim_dir=bin_sim_dir,
                    output_dir=training_dir,
                    weights_file=weights_file,
                    benchmark=benchmark,
                    target_accuracy=target_accuracy,
                    max_iterations=1,  # Single iteration per batch
                    timeout=timeout,
                    timeout_sim=timeout_sim,
                    num_trials=num_trials,
                    learning_rate=current_lr,
                    algorithms=algorithms
                )
                
                if iter_result.iteration_history:
                    iteration_accuracy.append(iter_result.iteration_history[0].accuracy_time_pct)
        
        # Calculate average accuracy across all batches and benchmarks
        if iteration_accuracy:
            avg_accuracy = sum(iteration_accuracy) / len(iteration_accuracy)
            log(f"\nIteration {iteration} average accuracy: {avg_accuracy:.1f}%")
            
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                result.best_weights_iteration = iteration
                # Save best weights
                best_weights_file = os.path.join(training_dir, f"best_weights_iter{iteration}.json")
                with open(weights_file) as f:
                    best_weights = json.load(f)
                with open(best_weights_file, 'w') as f:
                    json.dump(best_weights, f, indent=2)
                result.best_weights_file = best_weights_file
            
            result.final_accuracy_time_pct = avg_accuracy
            
            if avg_accuracy >= target_accuracy:
                log(f"\n🎯 TARGET REACHED: {avg_accuracy:.1f}% >= {target_accuracy}%")
                result.target_reached = True
                break
        
        result.iterations_run = iteration
    
    # Summary
    log(f"\n{'='*60}")
    log("LARGE-SCALE TRAINING COMPLETE")
    log(f"{'='*60}")
    log(f"Total iterations: {result.iterations_run}")
    log(f"Best accuracy: {best_accuracy:.1f}%")
    log(f"Target reached: {'YES' if result.target_reached else 'NO'}")
    
    return result


def train_adaptive_weights_iterative(
    graphs: List[GraphInfo],
    bin_dir: str,
    bin_sim_dir: str,
    output_dir: str,
    weights_file: str,
    benchmark: str = "pr",
    target_accuracy: float = 80.0,
    max_iterations: int = 10,
    timeout: int = TIMEOUT_REORDER,
    timeout_sim: int = TIMEOUT_SIM,
    num_trials: int = 3,
    learning_rate: float = 0.1,
    algorithms: List[int] = None
) -> TrainingResult:
    """
    Iterative training loop for adaptive algorithm selection weights.
    
    This function:
    1. Runs brute-force evaluation to measure current accuracy
    2. Identifies where adaptive picks wrong (what should have been chosen)
    3. Adjusts weights based on the analysis
    4. Repeats until target accuracy is reached or max iterations
    
    The learning process:
    - For each graph where adaptive picked wrong, we analyze the features
    - If the correct algorithm has different feature preferences, we adjust weights
    - We use a learning rate to prevent overcorrection
    
    Args:
        graphs: List of graphs to test
        bin_dir: Path to benchmark binaries
        bin_sim_dir: Path to cache simulation binaries
        output_dir: Where to save results
        weights_file: Path to perceptron weights file
        benchmark: Which benchmark to use (pr, bfs, etc.)
        target_accuracy: Target accuracy percentage (0-100)
        max_iterations: Maximum training iterations
        timeout: Timeout for reorder operations
        timeout_sim: Timeout for cache simulations
        num_trials: Number of benchmark trials per test
        learning_rate: How much to adjust weights (0.0-1.0)
        algorithms: List of algorithm IDs to test
    
    Returns:
        TrainingResult with iteration history and final accuracy
    """
    log_section(f"Iterative Weight Training (Target: {target_accuracy}%)")
    
    if algorithms is None:
        # Use representative algorithms for training
        algorithms = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13]  # Skip MAP, ADAPTIVE, etc.
    
    # Ensure results directory exists
    training_dir = os.path.join(output_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(training_dir, exist_ok=True)
    
    # Initialize/upgrade weights file with enhanced features
    log("Initializing enhanced weight structure...")
    initialize_enhanced_weights(weights_file)
    
    # Initialize result
    result = TrainingResult(
        final_accuracy_time_pct=0.0,
        final_accuracy_cache_pct=0.0,
        final_accuracy_top3_pct=0.0,
        iterations_run=0,
        target_accuracy=target_accuracy,
        target_reached=False
    )
    
    best_accuracy = 0.0
    best_iteration = 0
    
    for iteration in range(1, max_iterations + 1):
        log(f"\n{'='*60}")
        log(f"TRAINING ITERATION {iteration}/{max_iterations}")
        log(f"{'='*60}")
        
        # Step 1: Run brute-force analysis to measure current accuracy
        log(f"\n--- Step 1: Measure Current Accuracy ---")
        bf_results = run_subcommunity_brute_force(
            graphs=graphs,
            bin_dir=bin_dir,
            bin_sim_dir=bin_sim_dir,
            output_dir=training_dir,
            benchmark=benchmark,
            timeout=timeout,
            timeout_sim=timeout_sim,
            num_trials=num_trials
        )
        
        # Calculate overall accuracy
        if not bf_results:
            log("No brute-force results, stopping training")
            break
        
        successful = [r for r in bf_results if r.success]
        if not successful:
            log("No successful brute-force results, stopping training")
            break
        
        avg_accuracy_time = sum(r.adaptive_correct_time_pct for r in successful) / len(successful)
        avg_accuracy_cache = sum(r.adaptive_correct_cache_pct for r in successful) / len(successful)
        avg_top3_time = sum(r.adaptive_top3_time_pct for r in successful) / len(successful)
        avg_time_ratio = sum(r.avg_time_ratio for r in successful if r.avg_time_ratio > 0) / max(1, sum(1 for r in successful if r.avg_time_ratio > 0))
        avg_cache_ratio = sum(r.avg_cache_ratio for r in successful if r.avg_cache_ratio > 0) / max(1, sum(1 for r in successful if r.avg_cache_ratio > 0))
        
        log(f"\nIteration {iteration} Accuracy:")
        log(f"  Adaptive correct (time): {avg_accuracy_time:.1f}%")
        log(f"  Adaptive correct (cache): {avg_accuracy_cache:.1f}%")
        log(f"  Adaptive in top 3: {avg_top3_time:.1f}%")
        log(f"  Avg time ratio: {avg_time_ratio:.3f}")
        log(f"  Avg cache ratio: {avg_cache_ratio:.3f}")
        
        # Record iteration result
        iter_result = TrainingIterationResult(
            iteration=iteration,
            accuracy_time_pct=avg_accuracy_time,
            accuracy_cache_pct=avg_accuracy_cache,
            accuracy_top3_pct=avg_top3_time,
            avg_time_ratio=avg_time_ratio,
            avg_cache_ratio=avg_cache_ratio,
            graphs_tested=len(successful),
            weights_updated=0
        )
        
        # Track best iteration
        if avg_accuracy_time > best_accuracy:
            best_accuracy = avg_accuracy_time
            best_iteration = iteration
            # Save best weights
            best_weights_file = os.path.join(training_dir, f"best_weights_iter{iteration}.json")
            with open(weights_file) as f:
                best_weights = json.load(f)
            with open(best_weights_file, 'w') as f:
                json.dump(best_weights, f, indent=2)
            result.best_weights_iteration = iteration
            result.best_weights_file = best_weights_file
            log(f"  New best accuracy! Saved weights to {best_weights_file}")
        
        # Check if we've reached target accuracy
        if avg_accuracy_time >= target_accuracy:
            log(f"\n🎯 TARGET ACCURACY REACHED: {avg_accuracy_time:.1f}% >= {target_accuracy}%")
            result.target_reached = True
            result.iteration_history.append(iter_result)
            break
        
        # Step 2: Analyze errors and adjust weights
        log(f"\n--- Step 2: Analyze Errors and Adjust Weights ---")
        
        # Load current weights
        with open(weights_file) as f:
            weights = json.load(f)
        
        # Analyze each graph where adaptive was wrong
        weights_updated = 0
        for bf_result in successful:
            for sc_result in bf_result.subcommunity_results:
                if sc_result.adaptive_is_best_time:
                    continue  # Skip correct predictions
                
                # Get extended features of this subcommunity
                features = {
                    'density': sc_result.density,
                    'degree_variance': sc_result.degree_variance,
                    'hub_concentration': sc_result.hub_concentration,
                    'log_nodes': math.log10(sc_result.nodes) if sc_result.nodes > 0 else 0,
                    'log_edges': math.log10(sc_result.edges) if sc_result.edges > 0 else 0,
                    'avg_degree': (2 * sc_result.edges / sc_result.nodes) if sc_result.nodes > 0 else 0,
                    # New extended features (use defaults if not available)
                    'clustering_coefficient': getattr(sc_result, 'clustering_coefficient', 0.0),
                    'avg_path_length': getattr(sc_result, 'avg_path_length', 0.0),
                    'diameter_estimate': getattr(sc_result, 'diameter_estimate', 0.0),
                    'community_count': getattr(sc_result, 'community_count', 1),
                }
                
                adaptive_algo = sc_result.adaptive_algorithm
                correct_algo = sc_result.best_time_algorithm
                
                if not correct_algo or correct_algo == adaptive_algo:
                    continue
                
                # Adjust weights for the correct algorithm (increase selection probability)
                if correct_algo in weights:
                    algo_weights = weights[correct_algo]
                    
                    # --- Feature-based weight adjustments (stronger gradients) ---
                    
                    # Density-based adjustment
                    if features['density'] > 0.01:  # Relatively dense
                        current = algo_weights.get('w_density', 0)
                        algo_weights['w_density'] = round(current + learning_rate * 0.01 * features['density'], 6)
                    elif features['density'] < 0.001:  # Very sparse
                        current = algo_weights.get('w_density', 0)
                        algo_weights['w_density'] = round(current - learning_rate * 0.005, 6)
                    
                    # Degree variance adjustment (higher learning rate)
                    if features['degree_variance'] > 0.5:  # High variance
                        current = algo_weights.get('w_degree_variance', 0)
                        algo_weights['w_degree_variance'] = round(current + learning_rate * 0.005, 6)
                    elif features['degree_variance'] < 0.1:  # Low variance (uniform degrees)
                        current = algo_weights.get('w_degree_variance', 0)
                        algo_weights['w_degree_variance'] = round(current - learning_rate * 0.002, 6)
                    
                    # Hub concentration adjustment
                    if features['hub_concentration'] > 0.3:  # Hub-dominated
                        current = algo_weights.get('w_hub_concentration', 0)
                        algo_weights['w_hub_concentration'] = round(current + learning_rate * 0.01, 6)
                    elif features['hub_concentration'] < 0.1:  # No clear hubs
                        current = algo_weights.get('w_hub_concentration', 0)
                        algo_weights['w_hub_concentration'] = round(current - learning_rate * 0.005, 6)
                    
                    # --- New extended feature adjustments ---
                    
                    # Clustering coefficient (measures local connectivity)
                    if features['clustering_coefficient'] > 0.3:  # Highly clustered
                        current = algo_weights.get('w_clustering_coeff', 0)
                        algo_weights['w_clustering_coeff'] = round(current + learning_rate * 0.008, 6)
                    elif features['clustering_coefficient'] < 0.05:  # Tree-like structure
                        current = algo_weights.get('w_clustering_coeff', 0)
                        algo_weights['w_clustering_coeff'] = round(current - learning_rate * 0.004, 6)
                    
                    # Average path length (affects traversal algorithms)
                    if features['avg_path_length'] > 10:  # Long paths
                        current = algo_weights.get('w_avg_path_length', 0)
                        algo_weights['w_avg_path_length'] = round(current + learning_rate * 0.005, 6)
                    elif features['avg_path_length'] > 0 and features['avg_path_length'] < 3:  # Short paths
                        current = algo_weights.get('w_avg_path_length', 0)
                        algo_weights['w_avg_path_length'] = round(current - learning_rate * 0.003, 6)
                    
                    # Diameter (graph width)
                    if features['diameter_estimate'] > 20:  # Wide graph
                        current = algo_weights.get('w_diameter', 0)
                        algo_weights['w_diameter'] = round(current + learning_rate * 0.003, 6)
                    
                    # Community count (for multi-community graphs)
                    if features['community_count'] > 5:  # Many subcommunities
                        current = algo_weights.get('w_community_count', 0)
                        algo_weights['w_community_count'] = round(current + learning_rate * 0.002, 6)
                    
                    # --- Per-benchmark weight adjustment ---
                    benchmark_key = benchmark.lower()
                    if 'benchmark_weights' not in algo_weights:
                        algo_weights['benchmark_weights'] = {'pr': 1.0, 'bfs': 1.0, 'cc': 1.0, 'sssp': 1.0, 'bc': 1.0}
                    
                    # Increase this benchmark's multiplier for the correct algorithm
                    if benchmark_key in algo_weights['benchmark_weights']:
                        current_bw = algo_weights['benchmark_weights'][benchmark_key]
                        algo_weights['benchmark_weights'][benchmark_key] = round(current_bw + learning_rate * 0.02, 4)
                    
                    # Increase bias (stronger gradient)
                    current_bias = algo_weights.get('bias', 0.5)
                    algo_weights['bias'] = round(current_bias + learning_rate * 0.02, 4)
                    
                    weights[correct_algo] = algo_weights
                    weights_updated += 1
                
                # Decrease weights for the adaptive-selected algorithm (that was wrong)
                if adaptive_algo in weights:
                    algo_weights = weights[adaptive_algo]
                    
                    # Decrease bias (stronger gradient)
                    current_bias = algo_weights.get('bias', 0.5)
                    algo_weights['bias'] = round(current_bias - learning_rate * 0.015, 4)
                    
                    # Also decrease feature weights that contributed to wrong selection
                    if features['density'] > 0.01:
                        current = algo_weights.get('w_density', 0)
                        algo_weights['w_density'] = round(current - learning_rate * 0.003, 6)
                    
                    if features['degree_variance'] > 0.5:
                        current = algo_weights.get('w_degree_variance', 0)
                        algo_weights['w_degree_variance'] = round(current - learning_rate * 0.002, 6)
                    
                    if features['hub_concentration'] > 0.3:
                        current = algo_weights.get('w_hub_concentration', 0)
                        algo_weights['w_hub_concentration'] = round(current - learning_rate * 0.003, 6)
                    
                    # Decrease per-benchmark weight for this benchmark
                    benchmark_key = benchmark.lower()
                    if 'benchmark_weights' in algo_weights and benchmark_key in algo_weights['benchmark_weights']:
                        current_bw = algo_weights['benchmark_weights'][benchmark_key]
                        algo_weights['benchmark_weights'][benchmark_key] = round(current_bw - learning_rate * 0.01, 4)
                    
                    weights[adaptive_algo] = algo_weights
        
        # Add training metadata
        weights['_training_metadata'] = {
            'last_iteration': iteration,
            'accuracy_time_pct': avg_accuracy_time,
            'accuracy_cache_pct': avg_accuracy_cache,
            'timestamp': datetime.now().isoformat(),
            'learning_rate': learning_rate,
            'graphs_tested': len(successful)
        }
        
        # Save updated weights
        with open(weights_file, 'w') as f:
            json.dump(weights, f, indent=2)
        
        iter_result.weights_updated = weights_updated
        result.iteration_history.append(iter_result)
        
        log(f"  Weights updated for {weights_updated} algorithm adjustments")
        
        # Also save iteration-specific weights for debugging
        iter_weights_file = os.path.join(training_dir, f"weights_iter{iteration}.json")
        with open(iter_weights_file, 'w') as f:
            json.dump(weights, f, indent=2)
        
        result.iterations_run = iteration
    
    # Finalize results
    result.final_accuracy_time_pct = avg_accuracy_time if 'avg_accuracy_time' in dir() else 0.0
    result.final_accuracy_cache_pct = avg_accuracy_cache if 'avg_accuracy_cache' in dir() else 0.0
    result.final_accuracy_top3_pct = avg_top3_time if 'avg_top3_time' in dir() else 0.0
    
    # Save training summary
    summary_file = os.path.join(training_dir, "training_summary.json")
    summary = {
        'target_accuracy': target_accuracy,
        'target_reached': result.target_reached,
        'iterations_run': result.iterations_run,
        'final_accuracy_time_pct': result.final_accuracy_time_pct,
        'final_accuracy_cache_pct': result.final_accuracy_cache_pct,
        'final_accuracy_top3_pct': result.final_accuracy_top3_pct,
        'best_iteration': result.best_weights_iteration,
        'best_weights_file': result.best_weights_file,
        'iteration_history': [
            {
                'iteration': h.iteration,
                'accuracy_time_pct': h.accuracy_time_pct,
                'accuracy_cache_pct': h.accuracy_cache_pct,
                'accuracy_top3_pct': h.accuracy_top3_pct,
                'avg_time_ratio': h.avg_time_ratio,
                'avg_cache_ratio': h.avg_cache_ratio,
                'graphs_tested': h.graphs_tested,
                'weights_updated': h.weights_updated
            }
            for h in result.iteration_history
        ]
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    log(f"\n{'='*60}")
    log("TRAINING COMPLETE")
    log(f"{'='*60}")
    log(f"Iterations run: {result.iterations_run}")
    log(f"Target accuracy: {target_accuracy}%")
    log(f"Final accuracy (time): {result.final_accuracy_time_pct:.1f}%")
    log(f"Final accuracy (cache): {result.final_accuracy_cache_pct:.1f}%")
    log(f"Target reached: {'YES' if result.target_reached else 'NO'}")
    log(f"Best iteration: {result.best_weights_iteration}")
    log(f"Training summary saved to: {summary_file}")
    
    # If we didn't reach target, restore best weights
    if not result.target_reached and result.best_weights_file and os.path.exists(result.best_weights_file):
        log(f"\nRestoring best weights from iteration {result.best_weights_iteration}")
        with open(result.best_weights_file) as f:
            best_weights = json.load(f)
        with open(weights_file, 'w') as f:
            json.dump(best_weights, f, indent=2)
        log(f"Best weights (accuracy: {best_accuracy:.1f}%) restored to {weights_file}")
    
    # Backup with timestamp and sync to scripts folder
    backup_and_sync_weights(weights_file)
    
    return result


# ============================================================================
# Phase 6: Adaptive Order Analysis
# ============================================================================

def parse_adaptive_output(output: str) -> Tuple[float, int, List[SubcommunityInfo], Dict[str, int]]:
    """Parse the output from AdaptiveOrder to extract subcommunity information."""
    modularity = 0.0
    num_communities = 0
    subcommunities = []
    algo_distribution = {}
    
    # Parse modularity
    mod_match = re.search(r'Modularity:\s*([\d.]+)', output)
    if mod_match:
        modularity = float(mod_match.group(1))
    
    # Parse number of communities
    num_match = re.search(r'Num Communities:\s*([\d.]+)', output)
    if num_match:
        num_communities = int(float(num_match.group(1)))
    
    # Parse subcommunity table
    # Format: Comm    Nodes   Edges   Density DegVar  HubConc Selected
    table_pattern = re.compile(
        r'^(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\w+)',
        re.MULTILINE
    )
    
    for match in table_pattern.finditer(output):
        comm_id = int(match.group(1))
        nodes = int(match.group(2))
        edges = int(match.group(3))
        density = float(match.group(4))
        deg_var = float(match.group(5))
        hub_conc = float(match.group(6))
        selected = match.group(7)
        
        subcommunities.append(SubcommunityInfo(
            community_id=comm_id,
            nodes=nodes,
            edges=edges,
            density=density,
            degree_variance=deg_var,
            hub_concentration=hub_conc,
            selected_algorithm=selected
        ))
        
        # Count algorithm distribution
        if selected not in algo_distribution:
            algo_distribution[selected] = 0
        algo_distribution[selected] += 1
    
    return modularity, num_communities, subcommunities, algo_distribution

def analyze_adaptive_order(
    graphs: List,  # GraphInfo list
    bin_dir: str,
    output_dir: str,
    timeout: int = TIMEOUT_REORDER
) -> List[AdaptiveOrderResult]:
    """
    Analyze adaptive ordering for all graphs.
    Records subcommunity assignments and algorithm selections.
    """
    log_section("Phase 6: Adaptive Order Analysis")
    
    results = []
    
    for graph in graphs:
        log(f"\nGraph: {graph.name} ({graph.size_mb:.1f}MB)")
        
        binary = os.path.join(bin_dir, "pr")
        if not os.path.exists(binary):
            log(f"  Binary not found: {binary}")
            results.append(AdaptiveOrderResult(
                graph=graph.name,
                modularity=0.0,
                num_communities=0,
                success=False,
                error="Binary not found"
            ))
            continue
        
        # Run with AdaptiveOrder (algorithm 15)
        sym_flag = "-s" if graph.is_symmetric else ""
        cmd = f"{binary} -f {graph.path} {sym_flag} -o 15 -n 1"
        
        start_time = time.time()
        success, stdout, stderr = run_command(cmd, timeout)
        reorder_time = time.time() - start_time
        
        if success:
            output = stdout + stderr
            modularity, num_communities, subcommunities, algo_distribution = parse_adaptive_output(output)
            
            # Also parse global features for graph type detection
            parsed = parse_benchmark_output(output)
            
            # Update graph properties cache with detected features
            props = {
                'modularity': modularity,
                'degree_variance': parsed.get('degree_variance'),
                'hub_concentration': parsed.get('hub_concentration'),
                'graph_type': parsed.get('graph_type'),
                'nodes': graph.nodes,
                'edges': graph.edges,
                'avg_degree': 2 * graph.edges / graph.nodes if graph.nodes > 0 else 0
            }
            update_graph_properties(graph.name, props)
            
            log(f"  Modularity: {modularity:.4f}")
            log(f"  Communities: {num_communities}")
            if parsed.get('graph_type'):
                log(f"  Detected type: {parsed.get('graph_type')}")
            log(f"  Subcommunities analyzed: {len(subcommunities)}")
            log(f"  Algorithm distribution:")
            for algo, count in sorted(algo_distribution.items(), key=lambda x: -x[1])[:5]:
                log(f"    {algo}: {count}")
            
            results.append(AdaptiveOrderResult(
                graph=graph.name,
                modularity=modularity,
                num_communities=num_communities,
                subcommunities=subcommunities,
                algorithm_distribution=algo_distribution,
                reorder_time=reorder_time,
                success=True
            ))
        else:
            error = "TIMEOUT" if "TIMEOUT" in stderr else "FAILED"
            log(f"  {error}")
            results.append(AdaptiveOrderResult(
                graph=graph.name,
                modularity=0.0,
                num_communities=0,
                success=False,
                error=error
            ))
    
    # Save results
    results_file = os.path.join(output_dir, f"adaptive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    results_data = []
    for r in results:
        data = {
            "graph": r.graph,
            "modularity": r.modularity,
            "num_communities": r.num_communities,
            "algorithm_distribution": r.algorithm_distribution,
            "reorder_time": r.reorder_time,
            "success": r.success,
            "error": r.error,
            "subcommunities": [
                {
                    "community_id": s.community_id,
                    "nodes": s.nodes,
                    "edges": s.edges,
                    "density": s.density,
                    "degree_variance": s.degree_variance,
                    "hub_concentration": s.hub_concentration,
                    "selected_algorithm": s.selected_algorithm
                }
                for s in r.subcommunities
            ]
        }
        results_data.append(data)
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    log(f"\nAdaptive analysis saved to: {results_file}")
    
    return results

def compare_adaptive_vs_fixed(
    graphs: List,  # GraphInfo list
    bin_dir: str,
    benchmarks: List[str],
    fixed_algorithms: List[int],
    output_dir: str,
    num_trials: int = 3,
    timeout: int = TIMEOUT_BENCHMARK
) -> List[AdaptiveComparisonResult]:
    """
    Compare AdaptiveOrder (per-community selection) vs using a single fixed algorithm.
    
    This helps validate whether adaptive selection provides benefit over just
    using the best single algorithm for the entire graph.
    """
    log_section("Phase 7: Adaptive vs Fixed Comparison")
    
    results = []
    
    for graph in graphs:
        log(f"\nGraph: {graph.name} ({graph.size_mb:.1f}MB)")
        
        for bench in benchmarks:
            log(f"  Benchmark: {bench}")
            
            binary = os.path.join(bin_dir, bench)
            if not os.path.exists(binary):
                continue
            
            sym_flag = "-s" if graph.is_symmetric else ""
            
            # Run AdaptiveOrder (algorithm 15)
            cmd_adaptive = f"{binary} -f {graph.path} {sym_flag} -o 15 -n {num_trials}"
            success, stdout, stderr = run_command(cmd_adaptive, timeout)
            
            adaptive_time = 0.0
            if success:
                # Parse time from output
                time_match = re.search(r'Average Time:\s*([\d.]+)', stdout + stderr)
                if time_match:
                    adaptive_time = float(time_match.group(1))
            
            # Run Original (algorithm 0) for baseline
            cmd_orig = f"{binary} -f {graph.path} {sym_flag} -o 0 -n {num_trials}"
            success_orig, stdout_orig, stderr_orig = run_command(cmd_orig, timeout)
            
            original_time = 0.0
            if success_orig:
                time_match = re.search(r'Average Time:\s*([\d.]+)', stdout_orig + stderr_orig)
                if time_match:
                    original_time = float(time_match.group(1))
            
            adaptive_speedup = original_time / adaptive_time if adaptive_time > 0 else 0.0
            
            # Run fixed algorithms
            fixed_results = {}
            for algo_id in fixed_algorithms:
                algo_name = ALGORITHMS.get(algo_id, f"ALGO_{algo_id}")
                cmd_fixed = f"{binary} -f {graph.path} {sym_flag} -o {algo_id} -n {num_trials}"
                success_fixed, stdout_fixed, stderr_fixed = run_command(cmd_fixed, timeout)
                
                if success_fixed:
                    time_match = re.search(r'Average Time:\s*([\d.]+)', stdout_fixed + stderr_fixed)
                    if time_match:
                        fixed_time = float(time_match.group(1))
                        fixed_speedup = original_time / fixed_time if fixed_time > 0 else 0.0
                        fixed_results[algo_name] = fixed_speedup
            
            # Find best fixed algorithm
            best_fixed_algo = ""
            best_fixed_speedup = 0.0
            for algo, speedup in fixed_results.items():
                if speedup > best_fixed_speedup:
                    best_fixed_speedup = speedup
                    best_fixed_algo = algo
            
            adaptive_advantage = adaptive_speedup - best_fixed_speedup
            
            log(f"    Adaptive: {adaptive_speedup:.3f}x")
            log(f"    Best Fixed ({best_fixed_algo}): {best_fixed_speedup:.3f}x")
            log(f"    Advantage: {adaptive_advantage:+.3f}x")
            
            results.append(AdaptiveComparisonResult(
                graph=graph.name,
                benchmark=bench,
                adaptive_time=adaptive_time,
                adaptive_speedup=adaptive_speedup,
                fixed_results=fixed_results,
                best_fixed_algorithm=best_fixed_algo,
                best_fixed_speedup=best_fixed_speedup,
                adaptive_advantage=adaptive_advantage
            ))
    
    # Save results
    results_file = os.path.join(output_dir, f"adaptive_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    log(f"\nComparison results saved to: {results_file}")
    
    # Summary
    log("\n=== Summary ===")
    positive_advantage = [r for r in results if r.adaptive_advantage > 0]
    negative_advantage = [r for r in results if r.adaptive_advantage < 0]
    log(f"Adaptive better: {len(positive_advantage)} cases")
    log(f"Fixed better: {len(negative_advantage)} cases")
    if positive_advantage:
        avg_pos = sum(r.adaptive_advantage for r in positive_advantage) / len(positive_advantage)
        log(f"Avg advantage when better: +{avg_pos:.3f}x")
    if negative_advantage:
        avg_neg = sum(r.adaptive_advantage for r in negative_advantage) / len(negative_advantage)
        log(f"Avg disadvantage when worse: {avg_neg:.3f}x")
    
    return results

# ============================================================================
# Main Experiment Pipeline
# ============================================================================

def run_experiment(args):
    """Run the complete experiment pipeline."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Discover graphs - use explicit min/max if provided
    if args.min_size > 0 or args.max_size < float('inf'):
        # Custom size range specified via --min-size / --max-size
        min_size, max_size = args.min_size, args.max_size
    elif args.graphs == "all":
        min_size, max_size = 0, float('inf')
    elif args.graphs == "small":
        min_size, max_size = 0, SIZE_SMALL
    elif args.graphs == "medium":
        min_size, max_size = SIZE_SMALL, SIZE_MEDIUM
    elif args.graphs == "large":
        min_size, max_size = SIZE_MEDIUM, SIZE_LARGE
    else:
        min_size, max_size = args.min_size, args.max_size
    
    graphs = discover_graphs(args.graphs_dir, min_size, max_size, max_memory_gb=args.max_memory)
    
    if args.max_graphs:
        graphs = graphs[:args.max_graphs]
    
    log(f"Found {len(graphs)} graphs:")
    for g in graphs:
        log(f"  {g.name}: {g.size_mb:.1f}MB")
    
    if not graphs:
        log("No graphs found!", "ERROR")
        return
    
    # Select algorithms
    if args.key_only:
        algorithms = KEY_ALGORITHMS
    else:
        algorithms = BENCHMARK_ALGORITHMS
    
    log(f"\nAlgorithms to test: {len(algorithms)}")
    log(f"Benchmarks: {args.benchmarks}")
    
    # Initialize result storage
    all_reorder_results = []
    all_benchmark_results = []
    all_cache_results = []
    label_maps = {}
    
    # Pre-generate label maps if requested (also records reorder times)
    if getattr(args, "generate_maps", False):
        label_maps, reorder_timing_results = generate_label_maps(
            graphs=graphs,
            algorithms=algorithms,
            bin_dir=args.bin_dir,
            output_dir=args.results_dir,
            timeout=args.timeout_reorder,
            skip_slow=args.skip_slow
        )
        all_reorder_results.extend(reorder_timing_results)
    
    # Load existing label maps if requested
    if getattr(args, "use_maps", False):
        label_maps = load_label_maps_index(args.results_dir)
        if label_maps:
            log(f"Loaded label maps for {len(label_maps)} graphs")
        else:
            log("No existing label maps found", "WARN")
    
    # Phase 1: Reordering
    if args.phase in ["all", "reorder"]:
        reorder_results = generate_reorderings(
            graphs=graphs,
            algorithms=algorithms,
            bin_dir=args.bin_dir,
            output_dir=args.results_dir,
            timeout=args.timeout_reorder,
            skip_slow=args.skip_slow,
            generate_maps=True  # Always generate .lo mapping files
        )
        all_reorder_results.extend(reorder_results)
        
        # Save intermediate results
        reorder_file = os.path.join(args.results_dir, f"reorder_{timestamp}.json")
        with open(reorder_file, 'w') as f:
            json.dump([asdict(r) for r in reorder_results], f, indent=2)
        log(f"Reorder results saved to: {reorder_file}")
    
    # Phase 2: Benchmarks
    if args.phase in ["all", "benchmark"]:
        benchmark_results = run_benchmarks(
            graphs=graphs,
            algorithms=algorithms,
            benchmarks=args.benchmarks,
            bin_dir=args.bin_dir,
            num_trials=args.trials,
            timeout=args.timeout_benchmark,
            skip_slow=args.skip_slow,
            label_maps=label_maps,
            weights_dir=args.weights_dir,
            update_weights=not getattr(args, 'no_incremental', False)
        )
        all_benchmark_results.extend(benchmark_results)
        
        # Save intermediate results
        bench_file = os.path.join(args.results_dir, f"benchmark_{timestamp}.json")
        with open(bench_file, 'w') as f:
            json.dump([asdict(r) for r in benchmark_results], f, indent=2)
        log(f"Benchmark results saved to: {bench_file}")
    
    # Phase 3: Cache Simulations
    if args.phase in ["all", "cache"] and not args.skip_cache:
        cache_results = run_cache_simulations(
            graphs=graphs,
            algorithms=algorithms,
            benchmarks=args.benchmarks,
            bin_sim_dir=args.bin_sim_dir,
            timeout=args.timeout_sim,
            skip_heavy=args.skip_heavy,
            label_maps=label_maps,
            weights_dir=args.weights_dir,
            update_weights=not getattr(args, 'no_incremental', False)
        )
        all_cache_results.extend(cache_results)
        
        # Save intermediate results
        cache_file = os.path.join(args.results_dir, f"cache_{timestamp}.json")
        with open(cache_file, 'w') as f:
            json.dump([asdict(r) for r in cache_results], f, indent=2)
        log(f"Cache results saved to: {cache_file}")
    
    # Phase 4: Generate Weights
    if args.phase in ["all", "weights"]:
        # Load previous results if not running full pipeline
        if not all_benchmark_results:
            latest_bench = max(glob.glob(os.path.join(args.results_dir, "benchmark_*.json")), default=None)
            if latest_bench:
                with open(latest_bench) as f:
                    all_benchmark_results = [BenchmarkResult(**r) for r in json.load(f)]
        
        if not all_cache_results and not args.skip_cache:
            latest_cache = max(glob.glob(os.path.join(args.results_dir, "cache_*.json")), default=None)
            if latest_cache:
                with open(latest_cache) as f:
                    all_cache_results = [CacheResult(**r) for r in json.load(f)]
        
        if not all_reorder_results:
            latest_reorder = max(glob.glob(os.path.join(args.results_dir, "reorder_*.json")), default=None)
            if latest_reorder:
                with open(latest_reorder) as f:
                    all_reorder_results = [ReorderResult(**r) for r in json.load(f)]
        
        if all_benchmark_results:
            generate_perceptron_weights(
                benchmark_results=all_benchmark_results,
                cache_results=all_cache_results,
                reorder_results=all_reorder_results,
                output_file=args.weights_file
            )
            
            # Update zero weights with comprehensive analysis
            update_zero_weights(
                weights_file=args.weights_file,
                benchmark_results=all_benchmark_results,
                cache_results=all_cache_results,
                reorder_results=all_reorder_results,
                graphs_dir=args.graphs_dir
            )
    
    # Phase 6: Adaptive Order Analysis
    if args.phase in ["all", "adaptive"] or getattr(args, "adaptive_analysis", False):
        log_section("Running Adaptive Order Analysis")
        adaptive_results = analyze_adaptive_order(
            graphs=graphs,
            bin_dir=args.bin_dir,
            output_dir=args.results_dir,
            timeout=args.timeout_reorder
        )
        
        # Print summary
        successful = [r for r in adaptive_results if r.success]
        log(f"\nAdaptive analysis complete: {len(successful)}/{len(adaptive_results)} graphs")
        
        # Show algorithm distribution across all graphs
        total_distribution = {}
        for r in successful:
            for algo, count in r.algorithm_distribution.items():
                total_distribution[algo] = total_distribution.get(algo, 0) + count
        
        if total_distribution:
            log("\nOverall algorithm selection distribution:")
            for algo, count in sorted(total_distribution.items(), key=lambda x: -x[1]):
                log(f"  {algo}: {count}")
    
    # Phase 7: Adaptive vs Fixed Comparison
    if getattr(args, "adaptive_comparison", False):
        # Compare against top fixed algorithms
        fixed_algos = [1, 2, 4, 7, 12, 17]  # RCM, LeidenBFS, Gorder, DBG, RabbitOrder, LeidenDFSSize
        
        comparison_results = compare_adaptive_vs_fixed(
            graphs=graphs,
            bin_dir=args.bin_dir,
            benchmarks=args.benchmarks,
            fixed_algorithms=fixed_algos,
            output_dir=args.results_dir,
            num_trials=args.trials,
            timeout=args.timeout_benchmark
        )
    
    # Phase 8: Brute-Force Validation (All 20 algos vs Adaptive)
    if getattr(args, "brute_force", False):
        bf_benchmark = getattr(args, "bf_benchmark", "pr")
        brute_force_results = run_subcommunity_brute_force(
            graphs=graphs,
            bin_dir=args.bin_dir,
            bin_sim_dir=args.bin_sim_dir,
            output_dir=args.results_dir,
            benchmark=bf_benchmark,
            timeout=args.timeout_benchmark,
            timeout_sim=args.timeout_sim,
            num_trials=args.trials
        )
    
    # Phase 9: Iterative Training (feedback loop to optimize adaptive weights)
    if getattr(args, "train_adaptive", False):
        training_result = train_adaptive_weights_iterative(
            graphs=graphs,
            bin_dir=args.bin_dir,
            bin_sim_dir=args.bin_sim_dir,
            output_dir=args.results_dir,
            weights_file=args.weights_file,
            benchmark=getattr(args, "bf_benchmark", "pr"),
            target_accuracy=getattr(args, "target_accuracy", 80.0),
            max_iterations=getattr(args, "max_iterations", 10),
            timeout=args.timeout_benchmark,
            timeout_sim=args.timeout_sim,
            num_trials=args.trials,
            learning_rate=getattr(args, "learning_rate", 0.1),
            algorithms=algorithms
        )
    
    # Phase 10: Large-Scale Training (batched multi-benchmark training)
    if getattr(args, "train_large", False):
        large_training_result = train_adaptive_weights_large_scale(
            graphs=graphs,
            bin_dir=args.bin_dir,
            bin_sim_dir=args.bin_sim_dir,
            output_dir=args.results_dir,
            weights_file=args.weights_file,
            benchmarks=getattr(args, "train_benchmarks", ['pr', 'bfs', 'cc']),
            target_accuracy=getattr(args, "target_accuracy", 80.0),
            max_iterations=getattr(args, "max_iterations", 10),
            batch_size=getattr(args, "batch_size", 8),
            timeout=args.timeout_benchmark,
            timeout_sim=args.timeout_sim,
            num_trials=args.trials,
            learning_rate=getattr(args, "learning_rate", 0.15),
            algorithms=algorithms
        )
    
    # Initialize/upgrade weights file with enhanced features
    if getattr(args, "init_weights", False):
        log_section("Initialize Enhanced Weights")
        weights = initialize_enhanced_weights(args.weights_file)
        log(f"Weights initialized/upgraded with {len(weights) - 1} algorithms")
        log(f"Saved to: {args.weights_file}")
    
    # Fill ALL weights mode: comprehensive training to populate all weight fields
    if getattr(args, "fill_weights", False):
        log_section("Fill All Weights - Comprehensive Training")
        log("This mode runs all phases to populate every weight field:")
        log("  - Phase 0: Graph Analysis (detects graph types from properties)")
        log("  - Phase 1: Reorderings (fills w_reorder_time)")
        log("  - Phase 2: Benchmarks (fills bias, w_log_*, w_density, w_avg_degree)")
        log("  - Phase 3: Cache Simulation (fills cache_l1/l2/l3_impact)")
        log("  - Phase 4: Generate base weights")
        log("  - Phase 5: Update topology weights (fills w_clustering_coeff, etc.)")
        log("  - Phase 6: Compute per-benchmark weights")
        log("  - Phase 7: Generate per-graph-type weight files (from detected properties)")
        log("")
        
        # Load existing graph properties cache if available
        cache_dir = os.path.dirname(args.weights_file) or "results"
        load_graph_properties_cache(cache_dir)
        log(f"Loaded graph properties cache: {len(_graph_properties_cache)} graphs")
        
        # Force enable cache simulation for this mode
        skip_cache_original = getattr(args, 'skip_cache', False)
        args.skip_cache = False
        
        # Phase 0: Graph Analysis - Run AdaptiveOrder to detect graph types
        log_section("Phase 0: Graph Property Analysis")
        log("Running AdaptiveOrder to compute graph properties (modularity, degree variance, etc.)")
        adaptive_results = analyze_adaptive_order(
            graphs=graphs,
            bin_dir=args.bin_dir,
            output_dir=args.results_dir,
            timeout=args.timeout_reorder
        )
        # Save the cache after analysis
        save_graph_properties_cache(cache_dir)
        log(f"Graph properties cached for {len(_graph_properties_cache)} graphs")
        
        # Phase 1: Reorderings
        log_section("Phase 1: Generate Reorderings")
        reorder_results = generate_reorderings(
            graphs=graphs,
            algorithms=algorithms,
            bin_dir=args.bin_dir,
            output_dir=args.results_dir,
            timeout=args.timeout_reorder,
            skip_slow=getattr(args, 'skip_slow', False),
            generate_maps=True  # Always generate .lo mapping files
        )
        
        # Phase 2: Benchmarks (all of them)
        log_section("Phase 2: Execution Benchmarks (All)")
        all_benchmarks = ["pr", "bfs", "cc", "sssp", "bc"]
        benchmark_results = run_benchmarks(
            graphs=graphs,
            algorithms=algorithms,
            benchmarks=all_benchmarks,
            bin_dir=args.bin_dir,
            num_trials=args.trials,
            timeout=args.timeout_benchmark,
            skip_slow=getattr(args, 'skip_slow', False),
            label_maps={},
            weights_dir=args.weights_dir,
            update_weights=True  # Always update incrementally in fill-weights
        )
        
        # Phase 3: Cache Simulation
        log_section("Phase 3: Cache Simulation")
        cache_results = run_cache_simulations(
            graphs=graphs,
            algorithms=algorithms,
            benchmarks=["pr", "bfs"],  # Key benchmarks for cache
            bin_sim_dir=args.bin_sim_dir,
            timeout=args.timeout_sim,
            skip_heavy=getattr(args, 'skip_heavy', True),
            label_maps={},
            weights_dir=args.weights_dir,
            update_weights=True  # Always update incrementally in fill-weights
        )
        
        # Phase 4: Generate Base Weights
        log_section("Phase 4: Generate Perceptron Weights")
        weights = generate_perceptron_weights(
            benchmark_results=benchmark_results,
            cache_results=cache_results,
            reorder_results=reorder_results,
            output_file=args.weights_file
        )
        
        # Phase 5: Update Zero Weights with topology features
        log_section("Phase 5: Update Topology Weights")
        update_zero_weights(
            weights_file=args.weights_file,
            graphs_dir=args.graphs_dir,
            benchmark_results=benchmark_results,
            cache_results=cache_results,
            reorder_results=reorder_results
        )
        
        # Phase 6: Compute per-benchmark weights
        log_section("Phase 6: Compute Benchmark-Specific Weights")
        compute_benchmark_weights(
            weights_file=args.weights_file,
            benchmark_results=benchmark_results
        )
        
        # Phase 7: Generate per-graph-type weight files
        log_section("Phase 7: Generate Per-Graph-Type Weights")
        generate_per_type_weights(
            base_weights_file=args.weights_file,
            benchmark_results=benchmark_results,
            graphs_dir=args.graphs_dir,
            output_dir=os.path.dirname(args.weights_file) or "scripts"
        )
        
        # Save graph properties cache for future use
        save_graph_properties_cache(output_dir=os.path.dirname(args.weights_file) or "results")
        log(f"Saved graph properties cache to: {get_graph_properties_cache_file(os.path.dirname(args.weights_file) or 'results')}")
        
        # Restore original skip_cache setting
        args.skip_cache = skip_cache_original
        
        log_section("Fill Weights Complete")
        log(f"All weight fields have been populated")
        log(f"Weights file: {args.weights_file}")
        
        # Show summary
        with open(args.weights_file) as f:
            final_weights = json.load(f)
        
        log("\nWeight field population summary:")
        sample_algo = next((k for k in final_weights if not k.startswith("_")), None)
        if sample_algo:
            w = final_weights[sample_algo]
            for key, val in w.items():
                if key.startswith("_") or key == "benchmark_weights":
                    continue
                status = "✓ filled" if val != 0 else "○ zero"
                log(f"  {key}: {status}")
            
            if "benchmark_weights" in w:
                bw = w["benchmark_weights"]
                all_same = len(set(bw.values())) == 1
                log(f"  benchmark_weights: {'○ defaults' if all_same else '✓ tuned'}")
    
    log_section("Experiment Complete")
    log(f"Results directory: {args.results_dir}")
    log(f"Weights directory: {args.weights_dir}")
    
    # Show type system summary
    known_types = list_known_types(args.weights_dir)
    if known_types:
        log(f"Known types: {', '.join(known_types)}")
        log("Run --show-types to see full type details")

# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GraphBrew Unified Experiment Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ONE-CLICK: Download graphs, build, run full experiment pipeline
  python scripts/graphbrew_experiment.py --full --download-size SMALL
  
  # Download graphs only
  python scripts/graphbrew_experiment.py --download-only --download-size MEDIUM
  
  # Run full experiment on all graphs
  python scripts/graphbrew_experiment.py --phase all
  
  # Quick test on small graphs
  python scripts/graphbrew_experiment.py --graphs small --key-only
  
  # Only run benchmarks (skip cache simulation)
  python scripts/graphbrew_experiment.py --phase benchmark --skip-cache
  
  # Generate weights from existing results
  python scripts/graphbrew_experiment.py --phase weights
  
  # Custom graph size range
  python scripts/graphbrew_experiment.py --min-size 100 --max-size 1000
  
  # Run adaptive order analysis
  python scripts/graphbrew_experiment.py --phase adaptive
  
  # Compare adaptive vs fixed algorithms
  python scripts/graphbrew_experiment.py --adaptive-comparison
  
  # Run brute-force validation experiment
  python scripts/graphbrew_experiment.py --brute-force --graphs small
  
  # Pre-generate label maps and record reorder times
  python scripts/graphbrew_experiment.py --generate-maps --graphs small
  
  # Iterative training to reach 90% accuracy
  python scripts/graphbrew_experiment.py --train-adaptive --target-accuracy 90 --graphs small
  
  # Full training pipeline with custom learning rate
  python scripts/graphbrew_experiment.py --train-adaptive --target-accuracy 85 --max-iterations 15 --learning-rate 0.05
        """
    )
    
    # One-click full pipeline
    parser.add_argument("--full", action="store_true",
                        help="Run complete pipeline: download, build, experiment, weights")
    parser.add_argument("--download-only", action="store_true",
                        help="Only download graphs (no experiments)")
    parser.add_argument("--download-size", choices=["SMALL", "MEDIUM", "LARGE", "XLARGE", "ALL"],
                        default="SMALL", help="Size category of graphs to download")
    parser.add_argument("--force-download", action="store_true",
                        help="Re-download graphs even if they exist")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip build check (assume binaries exist)")
    parser.add_argument("--max-memory", type=float, default=None,
                        help="Maximum RAM (GB) for graph processing. Auto-detects available memory if not set. "
                             "Graphs requiring more memory are automatically skipped.")
    parser.add_argument("--auto-memory", action="store_true",
                        help="Automatically skip graphs that won't fit in available system RAM")
    parser.add_argument("--max-disk", type=float, default=None,
                        help="Maximum disk space (GB) for graph downloads. Graphs are skipped if total "
                             "download size would exceed this limit.")
    parser.add_argument("--auto-disk", action="store_true",
                        help="Automatically limit downloads to available disk space (uses 80%% of free space)")
    
    # Phase selection
    parser.add_argument("--phase", choices=["all", "reorder", "benchmark", "cache", "weights", "adaptive"],
                        default="all", help="Which phase(s) to run")
    
    # Graph selection
    parser.add_argument("--graphs", choices=["all", "small", "medium", "large", "custom"],
                        default="all", help="Graph size category")
    parser.add_argument("--graphs-dir", default=DEFAULT_GRAPHS_DIR,
                        help="Directory containing graph datasets")
    parser.add_argument("--min-size", type=float, default=0,
                        help="Minimum graph size in MB")
    parser.add_argument("--max-size", type=float, default=float('inf'),
                        help="Maximum graph size in MB")
    parser.add_argument("--max-graphs", type=int, default=None,
                        help="Maximum number of graphs to test")
    
    # Algorithm selection
    parser.add_argument("--key-only", action="store_true",
                        help="Only test key algorithms (faster)")
    parser.add_argument("--skip-slow", action="store_true",
                        help="Skip slow algorithms on large graphs")
    
    # Benchmark selection
    parser.add_argument("--benchmarks", nargs="+", default=BENCHMARKS,
                        help="Benchmarks to run")
    parser.add_argument("--trials", type=int, default=2,
                        help="Number of trials per benchmark")
    
    # Paths
    parser.add_argument("--bin-dir", default=DEFAULT_BIN_DIR,
                        help="Directory containing benchmark binaries")
    parser.add_argument("--bin-sim-dir", default=DEFAULT_BIN_SIM_DIR,
                        help="Directory containing simulation binaries")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                        help="Directory for results")
    parser.add_argument("--weights-dir", default=DEFAULT_WEIGHTS_DIR,
                        help="Directory for auto-generated type weight files")
    
    # Skip options
    parser.add_argument("--skip-cache", action="store_true",
                        help="Skip cache simulations")
    parser.add_argument("--skip-heavy", action="store_true",
                        help="Skip heavy simulations (BC, SSSP) on large graphs")
    
    # Timeouts
    parser.add_argument("--timeout-reorder", type=int, default=TIMEOUT_REORDER,
                        help="Timeout for reordering (seconds)")
    parser.add_argument("--timeout-benchmark", type=int, default=TIMEOUT_BENCHMARK,
                        help="Timeout for benchmarks (seconds)")
    parser.add_argument("--timeout-sim", type=int, default=TIMEOUT_SIM,
                        help="Timeout for simulations (seconds)")
    
    # Adaptive order analysis
    parser.add_argument("--adaptive-analysis", action="store_true",
                        help="Run adaptive order subcommunity analysis")
    parser.add_argument("--adaptive-comparison", action="store_true",
                        help="Compare adaptive vs fixed algorithm performance")
    
    # Label map options
    parser.add_argument("--generate-maps", action="store_true",
                        help="Pre-generate label.map files for consistent reordering")
    parser.add_argument("--use-maps", action="store_true",
                        help="Use pre-generated label maps instead of regenerating reorderings")
    
    # Brute-force validation
    parser.add_argument("--brute-force", action="store_true",
                        help="Run brute-force validation: test all 20 algorithms vs adaptive choice")
    parser.add_argument("--bf-benchmark", default="pr",
                        help="Benchmark to use for brute-force validation (default: pr)")
    
    # Iterative training options
    parser.add_argument("--train-adaptive", action="store_true",
                        help="Run iterative training loop to optimize adaptive algorithm weights")
    parser.add_argument("--train-large", action="store_true",
                        help="Run large-scale training with batching and multi-benchmark support")
    parser.add_argument("--target-accuracy", type=float, default=80.0,
                        help="Target accuracy %% for iterative training (default: 80)")
    parser.add_argument("--max-iterations", type=int, default=10,
                        help="Maximum training iterations (default: 10)")
    parser.add_argument("--learning-rate", type=float, default=0.1,
                        help="Learning rate for weight adjustments (default: 0.1)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for large-scale training (default: 8)")
    parser.add_argument("--train-benchmarks", nargs="+", default=['pr', 'bfs', 'cc'],
                        help="Benchmarks to use for multi-benchmark training (default: pr bfs cc)")
    parser.add_argument("--init-weights", action="store_true",
                        help="Initialize/upgrade weights file with enhanced features")
    parser.add_argument("--fill-weights", action="store_true",
                        help="Fill ALL weight fields: runs cache sim, graph features, benchmark analysis")
    
    # Type system options
    parser.add_argument("--show-types", action="store_true",
                        help="Show all known graph types and their statistics")
    parser.add_argument("--no-incremental", action="store_true",
                        help="Disable incremental weight updates (don't update type weights on-the-fly)")
    
    # Legacy weights file (for backward compatibility with old scripts)
    parser.add_argument("--weights-file", default="./scripts/perceptron_weights.json",
                        help="(Legacy) Single weights file for old-style batch processing")
    
    # Clean options
    parser.add_argument("--clean", action="store_true",
                        help="Clean results directory before running (keeps graphs and weights)")
    parser.add_argument("--clean-all", action="store_true",
                        help="Remove ALL generated data (graphs, results, mappings) - fresh start")
    
    # Auto-setup option
    parser.add_argument("--auto-setup", action="store_true",
                        help="Automatically setup everything: create directories, build if missing, download graphs if needed")
    
    args = parser.parse_args()
    
    # Determine memory limit
    if args.auto_memory and args.max_memory is None:
        # Auto-detect available memory, use 80% of total as safe limit
        total_mem = get_total_memory_gb()
        args.max_memory = total_mem * 0.8
        log(f"Auto-detected memory limit: {args.max_memory:.1f} GB (80% of {total_mem:.1f} GB total)", "INFO")
    elif args.max_memory is not None:
        log(f"Using specified memory limit: {args.max_memory:.1f} GB", "INFO")
    
    # Determine disk space limit
    if args.auto_disk and args.max_disk is None:
        # Auto-detect available disk space, use 80% of free space
        free_disk = get_available_disk_gb(args.graphs_dir if os.path.exists(args.graphs_dir) else ".")
        args.max_disk = free_disk * 0.8
        log(f"Auto-detected disk limit: {args.max_disk:.1f} GB (80% of {free_disk:.1f} GB free)", "INFO")
    elif args.max_disk is not None:
        log(f"Using specified disk limit: {args.max_disk:.1f} GB", "INFO")
    
    # Handle clean operations first
    if args.clean_all:
        clean_all(".", confirm=False)
        if not (args.full or args.download_only):
            return  # Just clean, don't run experiments
    elif args.clean:
        clean_results(args.results_dir, keep_graphs=True, keep_weights=True)
        if not (args.full or args.download_only or args.phase != "all"):
            return  # Just clean, don't run experiments
    
    # Handle --show-types early (informational command)
    if getattr(args, 'show_types', False):
        log_section("Known Graph Types")
        load_type_registry(args.weights_dir)
        known_types = list_known_types(args.weights_dir)
        if not known_types:
            log("No types defined yet. Types are auto-created when graphs are processed.")
            log(f"Types will be stored in: {args.weights_dir}")
        else:
            for type_name in sorted(known_types):
                summary = get_type_summary(type_name, args.weights_dir)
                if summary:
                    log(f"\n{type_name.upper()}:")
                    log(f"  Graphs trained: {summary.get('num_graphs', 0)}")
                    if 'best_algorithms' in summary:
                        for bench, algo in summary['best_algorithms'].items():
                            log(f"  Best for {bench}: {algo}")
                    # Show representative features instead of centroid (more readable)
                    if 'representative_features' in summary and summary['representative_features']:
                        rf = summary['representative_features']
                        log(f"  Features: modularity={rf.get('modularity', 0):.3f}, "
                            f"avg_degree={rf.get('avg_degree', 0):.1f}")
        return  # Exit after showing types
    
    # ALWAYS ensure prerequisites at start (unless skip_build is set)
    if not getattr(args, 'skip_build', False):
        ensure_prerequisites(
            project_dir=".",
            graphs_dir=args.graphs_dir,
            results_dir=args.results_dir,
            weights_dir=args.weights_dir,
            rebuild=False
        )
    else:
        # At least ensure directories exist
        os.makedirs(args.results_dir, exist_ok=True)
        os.makedirs(args.weights_dir, exist_ok=True)
    
    # Auto-setup: download graphs if needed
    if args.auto_setup or args.fill_weights or args.full:
        log_section("Auto-Setup: Downloading Graphs")
        # 3. Download ALL requested graphs FIRST (before any experiments)
        # This ensures all graphs are ready before we start reordering/benchmarks
        log("Downloading graphs (ensuring all are ready before experiments)...")
        downloaded = download_graphs(
            size_category=args.download_size,
            graphs_dir=args.graphs_dir,
            force=False,  # Don't re-download existing
            max_memory_gb=args.max_memory,
            max_disk_gb=args.max_disk
        )
        
        # Now discover all available graphs
        graphs = discover_graphs(args.graphs_dir, max_memory_gb=args.max_memory)
        if not graphs:
            log("No graphs found after download - aborting", "ERROR")
            sys.exit(1)
        log(f"Total graphs ready: {len(graphs)}")
        
        # 4. Initialize type registry if needed
        load_type_registry(args.weights_dir)
        log(f"Type registry loaded from: {args.weights_dir}")
        known_types = list_known_types(args.weights_dir)
        if known_types:
            log(f"Known types: {', '.join(known_types)}")
        else:
            log("No existing types - will create as graphs are processed")
        
        log("Auto-setup complete - all graphs downloaded and ready\n")
    
    try:
        # Handle download-only mode
        if args.download_only:
            download_graphs(
                size_category=args.download_size,
                graphs_dir=args.graphs_dir,
                force=args.force_download,
                max_memory_gb=args.max_memory,
                max_disk_gb=args.max_disk
            )
            log("Download complete. Run without --download-only to start experiments.", "INFO")
            return
        
        # Handle full pipeline mode
        if args.full:
            log("="*60, "INFO")
            log("GRAPHBREW ONE-CLICK EXPERIMENT PIPELINE", "INFO")
            log("="*60, "INFO")
            
            # Step 1: Download graphs
            downloaded = download_graphs(
                size_category=args.download_size,
                graphs_dir=args.graphs_dir,
                force=args.force_download,
                max_memory_gb=args.max_memory,
                max_disk_gb=args.max_disk
            )
            
            if not downloaded:
                log("No graphs downloaded - aborting", "ERROR")
                sys.exit(1)
            
            # Step 2: Build binaries
            if not args.skip_build:
                if not check_and_build_binaries("."):
                    log("Build failed - aborting", "ERROR")
                    sys.exit(1)
            
            # Step 3: Enable label map generation for consistent reordering
            args.generate_maps = True
            args.use_maps = True
            
            # Step 4: Run experiment (adjust graphs setting based on download size)
            log("\nStarting experiments...", "INFO")
            
            # Set graph size range based on download category
            if args.download_size == "SMALL":
                args.graphs = "small"
                args.max_size = 50
            elif args.download_size == "MEDIUM":
                args.graphs = "medium"
                args.max_size = 200
            elif args.download_size == "LARGE":
                args.graphs = "large"
            elif args.download_size == "XLARGE":
                args.graphs = "all"
            else:
                args.graphs = "all"
        
        run_experiment(args)
        
    except KeyboardInterrupt:
        log("\nExperiment interrupted by user", "WARN")
        sys.exit(1)
    except Exception as e:
        log(f"Experiment failed: {e}", "ERROR")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
