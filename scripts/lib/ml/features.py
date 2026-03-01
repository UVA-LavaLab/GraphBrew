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
    python -m scripts.lib.ml.features --detect-type graph.mtx
    python -m scripts.lib.ml.features --system-info
"""

import json
import os
import random
from collections import deque
from typing import Dict, List, Optional, Tuple

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
# Graph Properties — delegates to centralized GraphPropsStore
# =============================================================================
#
# All graph property data lives in ONE file: results/data/graph_properties.json
# Accessed via the GraphPropsStore singleton in datastore.py.
# The functions below are thin wrappers kept for backward compatibility.
#

def _get_store():
    """Lazy import to avoid circular dependency at module load time."""
    from scripts.lib.core.datastore import get_props_store
    return get_props_store()


def get_graph_properties_cache_file(output_dir: str = "") -> str:
    """Return the canonical graph_properties.json path.

    The *output_dir* parameter is accepted for backward compatibility but
    ignored — the path is always ``results/data/graph_properties.json``.
    """
    return str(_get_store().path)


def load_graph_properties_cache(output_dir: str = "results") -> Dict[str, Dict]:
    """Return the full graph-properties dict (loads from disk on first call)."""
    return _get_store().all()


def save_graph_properties_cache(output_dir: str = "results"):
    """Persist the current graph-properties dict to disk."""
    _get_store().save()


def update_graph_properties(graph_name: str, properties: Dict, output_dir: str = "results"):
    """Merge *properties* into the entry for *graph_name*.

    Auto-detects ``graph_type`` when enough features are present.
    """
    store = _get_store()
    store.update(graph_name, properties)

    # Auto-detect graph type if we have enough features
    props = store.get(graph_name) or {}
    if 'graph_type' not in props:
        if all(k in props for k in ['modularity', 'degree_variance', 'hub_concentration']):
            avg_degree = props.get('avg_degree', 0)
            if avg_degree == 0 and 'nodes' in props and 'edges' in props:
                avg_degree = 2 * props['edges'] / props['nodes'] if props['nodes'] > 0 else 0
            store.update(graph_name, {
                'graph_type': detect_graph_type(
                    props['modularity'],
                    props['degree_variance'],
                    props['hub_concentration'],
                    avg_degree,
                )
            })


def get_graph_properties(graph_name: str) -> Dict:
    """Get cached properties for a graph."""
    return _get_store().get(graph_name) or {}


def clear_graph_properties_cache():
    """Reset the in-memory store (mainly for tests)."""
    from scripts.lib.core.datastore import _props_store
    import scripts.lib.core.datastore as _ds
    _ds._props_store = None


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
    props = get_graph_properties(graph_name)
    
    if props:
        
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


def compute_packing_factor(adjacency_list: Dict[int, List[int]],
                           sample_size: int = 500) -> float:
    """
    Compute packing factor (IISWC'18): measures how many hub neighbors
    are already co-located in memory (have nearby IDs).
    
    High packing = hub neighbors already have nearby IDs → less benefit
    from hub-based reordering.
    
    Matches C++ ComputeSampledDegreeFeatures() packing factor logic.
    
    Args:
        adjacency_list: Dict mapping node ID to list of neighbor IDs
        sample_size: Number of hub nodes to sample
    
    Returns:
        Fraction of hub neighbors with nearby IDs, in [0, 1]
    """
    if not adjacency_list:
        return 0.0
    
    nodes = sorted(adjacency_list.keys())
    num_nodes = len(nodes)
    if num_nodes < 10:
        return 0.0
    
    # Find top-degree hub nodes
    node_degrees = [(node, len(adjacency_list[node])) for node in nodes]
    node_degrees.sort(key=lambda x: x[1], reverse=True)
    
    hub_count = max(1, min(sample_size, num_nodes) // 10)
    locality_window = max(64, num_nodes // 100)
    
    total_neighbors = 0
    colocated_neighbors = 0
    
    for node, deg in node_degrees[:hub_count]:
        if deg < 2:
            continue
        for neighbor in adjacency_list[node]:
            total_neighbors += 1
            if abs(neighbor - node) <= locality_window:
                colocated_neighbors += 1
    
    return colocated_neighbors / total_neighbors if total_neighbors > 0 else 0.0


def compute_forward_edge_fraction(adjacency_list: Dict[int, List[int]],
                                  sample_size: int = 2000) -> float:
    """
    Compute forward edge fraction (GoGraph): fraction of edges (u,v) where
    ID(u) < ID(v).
    
    High forward fraction = ordering already respects data flow direction.
    Important for async iterative algorithms (PR, SSSP convergence).
    
    Matches C++ ComputeSampledDegreeFeatures() forward edge fraction logic.
    
    Args:
        adjacency_list: Dict mapping node ID to list of neighbor IDs
        sample_size: Number of nodes to sample
    
    Returns:
        Fraction of forward edges, in [0, 1]
    """
    if not adjacency_list:
        return 0.5
    
    nodes = sorted(adjacency_list.keys())
    num_nodes = len(nodes)
    
    # Sample uniformly
    if num_nodes > sample_size:
        step = num_nodes / sample_size
        sampled = [nodes[int(i * step)] for i in range(sample_size)]
    else:
        sampled = nodes
    
    forward_count = 0
    total_count = 0
    
    for node in sampled:
        for neighbor in adjacency_list.get(node, []):
            total_count += 1
            if node < neighbor:
                forward_count += 1
    
    return forward_count / total_count if total_count > 0 else 0.5


def compute_vertex_significance_skewness(adjacency_list: Dict[int, List[int]],
                                          sample_size: int = 2000) -> float:
    """
    Compute vertex significance skewness (DON-RL, Zhao et al.).
    
    Coefficient of variation of per-vertex locality contributions
    (degree × local fraction). High skewness → hub-dominated graph →
    HubSort works well. Low skewness → uniform → community-based better.
    
    Matches C++ ComputeSampledDegreeFeatures() vertex_significance_skewness.
    
    Args:
        adjacency_list: Adjacency list {node: [neighbors]}
        sample_size: Number of vertices to sample
    
    Returns:
        CV (std/mean) of per-vertex significance values
    """
    import math
    nodes = sorted(adjacency_list.keys())
    n = len(nodes)
    if n < 10:
        return 0.0
    
    stride = max(1, n // sample_size)
    sampled = nodes[::stride][:sample_size]
    locality_window = max(64, n // 100)
    
    significance = []
    for node in sampled:
        neighbors = adjacency_list.get(node, [])
        deg = len(neighbors)
        if deg == 0:
            significance.append(0.0)
            continue
        local_count = sum(1 for nb in neighbors if abs(nb - node) <= locality_window)
        significance.append(deg * (local_count / deg))
    
    if not significance:
        return 0.0
    
    mean = sum(significance) / len(significance)
    if mean < 1e-12:
        return 0.0
    variance = sum((s - mean) ** 2 for s in significance) / max(1, len(significance) - 1)
    return math.sqrt(variance) / mean


def compute_window_neighbor_overlap(adjacency_list: Dict[int, List[int]],
                                     sample_size: int = 2000) -> float:
    """
    Compute window neighbor overlap (DON-RL, Zhao et al.).
    
    Mean fraction of a vertex's neighbors within a locality window.
    Generalizes packing_factor with uniform sampling across the graph
    (packing_factor only samples hubs).
    
    Matches C++ ComputeSampledDegreeFeatures() window_neighbor_overlap.
    
    Args:
        adjacency_list: Adjacency list {node: [neighbors]}
        sample_size: Number of vertices to sample
    
    Returns:
        Mean neighbor-in-window fraction across sampled vertices
    """
    nodes = sorted(adjacency_list.keys())
    n = len(nodes)
    if n < 10:
        return 0.0
    
    stride = max(1, n // sample_size)
    sampled = nodes[::stride][:sample_size]
    locality_window = max(64, n // 100)
    
    total_overlap = 0.0
    counted = 0
    
    for node in sampled:
        neighbors = adjacency_list.get(node, [])
        deg = len(neighbors)
        if deg == 0:
            continue
        in_window = sum(1 for nb in neighbors if abs(nb - node) <= locality_window)
        total_overlap += in_window / deg
        counted += 1
    
    return total_overlap / counted if counted > 0 else 0.0


def compute_sampled_locality_score(adjacency_list: Dict[int, List[int]],
                                    sample_size: int = 1000) -> float:
    """
    Compute sampled locality score — F(σ) approximation (P1 3.1d).
    
    Measures how well the CURRENT vertex ordering preserves graph locality,
    using a narrow cache-line-sized window (~16 vertices or N/1000).
    
    High value → current ordering is already cache-friendly → ORIGINAL likely wins.
    Low value  → current ordering is poor → reordering has high potential benefit.
    
    Complementary to window_neighbor_overlap which uses a broader window (N/100).
    
    Matches C++ ComputeSampledDegreeFeatures() sampled_locality_score.
    
    Args:
        adjacency_list: Adjacency list {node: [neighbors]}
        sample_size: Number of vertices to sample
    
    Returns:
        Mean neighbor-in-cache-window fraction across sampled vertices
    """
    nodes = sorted(adjacency_list.keys())
    n = len(nodes)
    if n < 10:
        return 0.0
    
    stride = max(1, n // sample_size)
    sampled = nodes[::stride][:sample_size]
    # Narrow window approximating L1 cache line reach (~16 vertices)
    cache_window = max(16, n // 1000)
    
    total_locality = 0.0
    counted = 0
    
    for node in sampled:
        neighbors = adjacency_list.get(node, [])
        deg = len(neighbors)
        if deg == 0:
            continue
        in_cache = sum(1 for nb in neighbors if abs(nb - node) <= cache_window)
        total_locality += in_cache / deg
        counted += 1
    
    return total_locality / counted if counted > 0 else 0.0


def compute_avg_reuse_distance(adjacency_list: Dict[int, List[int]],
                                sample_size: int = 1000) -> float:
    """
    Compute average transpose reuse distance (P3 3.2, P-OPT inspired).
    
    For sampled vertices, computes the mean absolute distance to in-neighbors
    (predecessors in the transpose graph). High reuse distance indicates
    the graph benefits from reordering to reduce cache misses.
    
    Matches C++ ComputeSampledDegreeFeatures() avg_reuse_distance.
    
    Args:
        adjacency_list: Adjacency list {node: [neighbors]}
        sample_size: Number of vertices to sample
    
    Returns:
        Mean reuse distance normalized by graph size [0, ~0.5]
    """
    nodes = sorted(adjacency_list.keys())
    n = len(nodes)
    if n < 10:
        return 0.0
    
    # Build reverse adjacency list (transpose)
    reverse_adj: Dict[int, List[int]] = {}
    for src, neighbors in adjacency_list.items():
        for dst in neighbors:
            if dst not in reverse_adj:
                reverse_adj[dst] = []
            reverse_adj[dst].append(src)
    
    stride = max(1, n // sample_size)
    sampled = nodes[::stride][:sample_size]
    
    total_reuse = 0.0
    counted = 0
    
    for node in sampled:
        predecessors = reverse_adj.get(node, [])
        in_deg = len(predecessors)
        if in_deg == 0:
            continue
        sum_dist = sum(abs(pred - node) for pred in predecessors)
        total_reuse += sum_dist / in_deg
        counted += 1
    
    # Normalize by graph size
    return (total_reuse / counted) / n if counted > 0 and n > 0 else 0.0


def compute_packing_factor_cl(adjacency_list: Dict[int, List[int]],
                               sample_size: int = 500) -> float:
    """
    Compute IISWC'18 cache-line packing factor — paper-aligned variant.
    
    Fraction of a hub's neighbors that map to the SAME cache line
    (vertex_id // CL_VERTS) as the hub. More faithful to Su et al.'s
    definition than the ID-distance proxy (compute_packing_factor).
    
    Args:
        adjacency_list: Dict mapping node ID to list of neighbor IDs
        sample_size: Number of hub nodes to sample
    
    Returns:
        Fraction of hub neighbors on same cache line, in [0, 1]
    """
    CL_VERTS = 64 // 4  # 16 vertices per 64-byte cache line
    
    if not adjacency_list:
        return 0.0
    
    nodes = sorted(adjacency_list.keys())
    num_nodes = len(nodes)
    if num_nodes < 10:
        return 0.0
    
    node_degrees = [(node, len(adjacency_list[node])) for node in nodes]
    node_degrees.sort(key=lambda x: x[1], reverse=True)
    
    hub_count = max(1, min(sample_size, num_nodes) // 10)
    total_neighbors = 0
    same_cl_neighbors = 0
    
    for node, deg in node_degrees[:hub_count]:
        if deg < 2:
            continue
        node_cl = node // CL_VERTS
        for neighbor in adjacency_list[node]:
            total_neighbors += 1
            if neighbor // CL_VERTS == node_cl:
                same_cl_neighbors += 1
    
    return same_cl_neighbors / total_neighbors if total_neighbors > 0 else 0.0


def compute_locality_score_pairwise(adjacency_list: Dict[int, List[int]],
                                     sample_size: int = 2000) -> float:
    """
    Compute DON-RL pairwise locality score — paper-aligned variant.
    
    Sampled mean of 1/|σ(u)−σ(v)| over edges, approximating the paper's
    F(σ) = Σ_{(u,v)∈E} 1/(|σ(u)−σ(v)|) objective.
    
    Args:
        adjacency_list: Dict mapping node ID to list of neighbor IDs
        sample_size: Number of nodes to sample
    
    Returns:
        Mean reciprocal ID-distance over sampled edges
    """
    if not adjacency_list:
        return 0.0
    
    nodes = sorted(adjacency_list.keys())
    n = len(nodes)
    if n < 10:
        return 0.0
    
    stride = max(1, n // sample_size)
    sampled = nodes[::stride][:sample_size]
    
    total_recip = 0.0
    counted_edges = 0
    
    for node in sampled:
        for neighbor in adjacency_list.get(node, []):
            diff = abs(neighbor - node)
            if diff > 0:
                total_recip += 1.0 / diff
                counted_edges += 1
    
    return total_recip / counted_edges if counted_edges > 0 else 0.0


def compute_reuse_distance_lru(adjacency_list: Dict[int, List[int]],
                                sample_size: int = 500) -> float:
    """
    Compute P-OPT LRU stack distance — paper-aligned variant.
    
    Simulates a small LRU cache (64 cache lines) and measures the mean
    stack distance when accessing each vertex's neighbors in order.
    More faithful to P-OPT's temporal reuse distance than the spatial
    ID-distance proxy (compute_avg_reuse_distance).
    
    Args:
        adjacency_list: Dict mapping node ID to list of neighbor IDs
        sample_size: Number of vertices to sample
    
    Returns:
        Mean LRU stack distance normalized by cache size, in [0, 1]
    """
    CL_VERTS = 64 // 4  # 16 vertices per cache line
    LRU_CAP = 64
    
    if not adjacency_list:
        return 0.0
    
    nodes = sorted(adjacency_list.keys())
    n = len(nodes)
    if n < 10:
        return 0.0
    
    stride = max(1, n // sample_size)
    sampled = nodes[::stride][:sample_size]
    
    total_stack_dist = 0.0
    counted_access = 0
    
    for node in sampled:
        neighbors = adjacency_list.get(node, [])
        if len(neighbors) < 2:
            continue
        
        lru_stack = []  # most-recent at end
        for neighbor in neighbors:
            cl = neighbor // CL_VERTS
            try:
                idx = lru_stack.index(cl)
                # Hit: distance = items between this and top
                total_stack_dist += len(lru_stack) - idx - 1
                lru_stack.pop(idx)
                lru_stack.append(cl)
            except ValueError:
                # Miss: distance = current LRU size
                total_stack_dist += len(lru_stack)
                lru_stack.append(cl)
                if len(lru_stack) > LRU_CAP:
                    lru_stack.pop(0)
            counted_access += 1
    
    return (total_stack_dist / counted_access) / LRU_CAP if counted_access > 0 else 0.0


def compute_working_set_ratio(nodes: int, edges: int) -> float:
    """
    Compute working set ratio (P-OPT): graph_bytes / LLC_size.
    
    Estimates how much of the graph's working set overflows the last-level cache.
    ratio ≈ 1 → graph fits in cache → reordering has limited benefit
    ratio >> 1 → graph exceeds cache → reordering can significantly help
    
    Matches C++ ComputeSampledDegreeFeatures() working set ratio logic.
    
    Args:
        nodes: Number of nodes in the graph
        edges: Number of directed edges (or 2× undirected edges)
    
    Returns:
        graph_bytes / LLC_size ratio (0 if LLC size unknown)
    """
    if nodes <= 0:
        return 0.0
    
    # CSR working set ≈ offsets + edges + vertex data
    # Matches C++ calculation:
    #   offsets: (num_nodes+1) * sizeof(int64_t) = 8 bytes
    #   edges:   num_edges * sizeof(int32_t) = 4 bytes
    #   vertex:  num_nodes * sizeof(double)  = 8 bytes
    graph_bytes = (nodes + 1) * 8 + edges * 4 + nodes * 8
    
    llc_bytes = get_llc_size_bytes()
    return graph_bytes / llc_bytes if llc_bytes > 0 else 0.0


def get_llc_size_bytes() -> int:
    """
    Get Last-Level Cache size in bytes.
    
    Matches C++ GetLLCSizeBytes() in reorder_types.h.
    Reads from /sys/devices/system/cpu/ on Linux.
    
    Returns:
        LLC size in bytes, or default 30MB if detection fails
    """
    # Try Linux sysfs
    try:
        import glob
        cache_dirs = sorted(glob.glob('/sys/devices/system/cpu/cpu0/cache/index*'))
        max_size = 0
        for cache_dir in cache_dirs:
            try:
                with open(os.path.join(cache_dir, 'size'), 'r') as f:
                    size_str = f.read().strip()
                    # Parse "8192K" or "32M" format
                    if size_str.endswith('K'):
                        size = int(size_str[:-1]) * 1024
                    elif size_str.endswith('M'):
                        size = int(size_str[:-1]) * 1024 * 1024
                    elif size_str.endswith('G'):
                        size = int(size_str[:-1]) * 1024 * 1024 * 1024
                    else:
                        size = int(size_str)
                    max_size = max(max_size, size)
            except (IOError, ValueError):
                continue
        if max_size > 0:
            return max_size
    except ImportError:
        pass
    
    # Default: 30MB (same as C++ GetLLCSizeBytes fallback)
    return 30 * 1024 * 1024


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
        Dictionary with all computed features including locality metrics
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
        'packing_factor': 0.0,
        'forward_edge_fraction': 0.5,
        'working_set_ratio': 0.0,
        # DON-RL features (Zhao et al.)
        'vertex_significance_skewness': 0.0,
        'window_neighbor_overlap': 0.0,
        # P1 3.1d: Sampled locality score
        'sampled_locality_score': 0.0,
        # P3 3.2: Transpose reuse distance
        'avg_reuse_distance': 0.0,
    }
    
    # If we have the adjacency list, compute additional features
    if adjacency_list and len(adjacency_list) > 0 and len(adjacency_list) < 50000:
        try:
            features['clustering_coefficient'] = compute_clustering_coefficient_sample(adjacency_list)
            diameter, avg_path = estimate_diameter_bfs(adjacency_list)
            features['diameter_estimate'] = diameter
            features['avg_path_length'] = avg_path
            features['community_count'] = count_subcommunities_quick(adjacency_list)
            
            # Locality features (match C++ ComputeSampledDegreeFeatures)
            features['packing_factor'] = compute_packing_factor(adjacency_list)
            features['forward_edge_fraction'] = compute_forward_edge_fraction(adjacency_list)
            
            # DON-RL features (Zhao et al.)
            features['vertex_significance_skewness'] = compute_vertex_significance_skewness(adjacency_list)
            features['window_neighbor_overlap'] = compute_window_neighbor_overlap(adjacency_list)
            # P1 3.1d: Sampled locality score
            features['sampled_locality_score'] = compute_sampled_locality_score(adjacency_list)
            # P3 3.2: Transpose reuse distance
            features['avg_reuse_distance'] = compute_avg_reuse_distance(adjacency_list)
            # Paper-aligned feature variants
            features['packing_factor_cl'] = compute_packing_factor_cl(adjacency_list)
            features['locality_score_pairwise'] = compute_locality_score_pairwise(adjacency_list)
            features['reuse_distance_lru'] = compute_reuse_distance_lru(adjacency_list)
        except Exception:
            pass  # Use defaults on error
    
    # Working set ratio (can compute without adjacency list)
    features['working_set_ratio'] = compute_working_set_ratio(nodes, edges)
    
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
    except (OSError, ValueError):
        pass
    # Fallback: try psutil if available
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
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
    except (OSError, ValueError):
        pass
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
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
    except (OSError, TypeError):
        pass
    try:
        # Fallback using statvfs
        stat = os.statvfs(path)
        return (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
    except OSError:
        pass
    # Default to 100GB if we can't detect
    return 100.0


def get_total_disk_gb(path: str = ".") -> float:
    """Get total disk space in GB for the given path."""
    try:
        import shutil
        usage = shutil.disk_usage(path)
        return usage.total / (1024 ** 3)
    except (OSError, TypeError):
        pass
    try:
        stat = os.statvfs(path)
        return (stat.f_blocks * stat.f_frsize) / (1024 ** 3)
    except OSError:
        pass
    return 500.0


def get_num_threads() -> int:
    """Get the number of available CPU threads."""
    try:
        return os.cpu_count() or 1
    except Exception:
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
