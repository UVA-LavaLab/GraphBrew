#!/usr/bin/env python3
"""
Weight management utilities for GraphBrew.

Handles type-based perceptron weights for adaptive algorithm selection.
Implements auto-clustering type system for graph classification.

Standalone usage:
    python -m scripts.lib.weights --list-types
    python -m scripts.lib.weights --show-type type_0
    python -m scripts.lib.weights --best-algo type_0 --benchmark pr

Library usage:
    from scripts.lib.weights import (
        assign_graph_type, load_type_weights, update_type_weights_incremental,
        get_best_algorithm_for_type
    )
"""

import os
import json
import math
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from .utils import (
    PROJECT_ROOT, WEIGHTS_DIR, ACTIVE_WEIGHTS_DIR, ALGORITHMS,
    Logger, get_timestamp,
    WEIGHT_PATH_LENGTH_NORMALIZATION, WEIGHT_REORDER_TIME_NORMALIZATION,
    WEIGHT_AVG_DEGREE_DEFAULT,
    RABBITORDER_VARIANTS, RABBITORDER_DEFAULT_VARIANT,
    GRAPHBREW_VARIANTS, GRAPHBREW_DEFAULT_VARIANT,
)

# Initialize logger
log = Logger()

# =============================================================================
# Constants
# =============================================================================

# Default weights directory (active/ subfolder where C++ reads from)
DEFAULT_WEIGHTS_DIR = str(ACTIVE_WEIGHTS_DIR)

# Auto-clustering configuration
CLUSTER_DISTANCE_THRESHOLD = 0.15  # Max normalized distance to join existing cluster
MIN_SAMPLES_FOR_CLUSTER = 2  # Minimum graphs to form a stable cluster


# =============================================================================
# Data Classes
# =============================================================================

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
    w_reorder_time: float = 0.0
    w_clustering_coeff: float = 0.0
    w_avg_path_length: float = 0.0
    w_diameter: float = 0.0
    w_community_count: float = 0.0
    w_packing_factor: float = 0.0
    w_forward_edge_fraction: float = 0.0
    w_working_set_ratio: float = 0.0
    w_dv_x_hub: float = 0.0
    w_mod_x_logn: float = 0.0
    w_pf_x_wsr: float = 0.0
    w_fef_convergence: float = 0.0
    benchmark_weights: Dict[str, float] = field(default_factory=lambda: {
        'pr': 1.0, 'bfs': 1.0, 'cc': 1.0, 'sssp': 1.0, 'bc': 1.0, 'tc': 1.0
    })
    _metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'PerceptronWeight':
        """Create PerceptronWeight from a dict (e.g., JSON weight entry)."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**kwargs)
    
    def compute_score(self, features: Dict, benchmark: str = 'pr') -> float:
        """Compute perceptron score for given features and benchmark.
        
        Matches C++ scoreBase() + score() in reorder_types.h.
        """
        log_nodes = math.log10(features.get('nodes', 1) + 1)
        log_edges = math.log10(features.get('edges', 1) + 1)
        
        score = self.bias
        score += self.w_modularity * features.get('modularity', 0.0)
        score += self.w_log_nodes * log_nodes
        score += self.w_log_edges * log_edges
        score += self.w_density * features.get('density', 0.0)
        score += self.w_avg_degree * features.get('avg_degree', 0.0) / 100.0
        score += self.w_degree_variance * features.get('degree_variance', 0.0)
        score += self.w_hub_concentration * features.get('hub_concentration', 0.0)
        # Accept both key names for clustering (experiment uses 'clustering_coefficient',
        # emulator/C++ use 'clustering_coeff')
        clustering = features.get('clustering_coefficient', features.get('clustering_coeff', 0.0))
        score += self.w_clustering_coeff * clustering
        score += self.w_avg_path_length * features.get('avg_path_length', 0.0) / WEIGHT_PATH_LENGTH_NORMALIZATION
        score += self.w_diameter * features.get('diameter', features.get('diameter_estimate', 0.0)) / 50.0
        score += self.w_community_count * math.log10(features.get('community_count', 1) + 1)
        score += self.w_reorder_time * features.get('reorder_time', 0.0)
        score += self.w_packing_factor * features.get('packing_factor', 0.0)
        score += self.w_forward_edge_fraction * features.get('forward_edge_fraction', 0.0)
        score += self.w_working_set_ratio * math.log2(features.get('working_set_ratio', 0.0) + 1.0)
        
        # Quadratic interaction terms
        log_wsr = math.log2(features.get('working_set_ratio', 0.0) + 1.0)
        log_n = math.log10(features.get('nodes', 1) + 1)
        score += self.w_dv_x_hub * features.get('degree_variance', 0.0) * features.get('hub_concentration', 0.0)
        score += self.w_mod_x_logn * features.get('modularity', 0.0) * log_n
        score += self.w_pf_x_wsr * features.get('packing_factor', 0.0) * log_wsr
        
        # Cache impact weights — matches C++ scoreBase():
        #   s += cache_l1_impact * 0.5 + cache_l2_impact * 0.3
        #      + cache_l3_impact * 0.2 + cache_dram_penalty
        score += self.cache_l1_impact * 0.5
        score += self.cache_l2_impact * 0.3
        score += self.cache_l3_impact * 0.2
        score += self.cache_dram_penalty
        
        # Convergence bonus for iterative algorithms (PR, SSSP)
        # Use the benchmark parameter passed to this function, not from features dict
        bench_name = benchmark if benchmark else features.get('benchmark', '')
        if bench_name in ('pr', 'sssp'):
            score += self.w_fef_convergence * features.get('forward_edge_fraction', 0.0)
        
        # Apply benchmark-specific multiplier
        bench_mult = self.benchmark_weights.get(bench_name.lower(), 1.0)
        return score * bench_mult


# =============================================================================
# Global Type Registry
# =============================================================================

_type_registry: Dict[str, Dict] = {}
_type_registry_file: str = os.path.join(DEFAULT_WEIGHTS_DIR, "type_registry.json")


def _get_next_type_name() -> str:
    """Generate next type name (type_0, type_1, etc.)."""
    if not _type_registry:
        return "type_0"
    
    existing = [k for k in _type_registry.keys() if k.startswith("type_")]
    if not existing:
        return "type_0"
    
    numbers = []
    for k in existing:
        try:
            num = int(k.replace("type_", ""))
            numbers.append(num)
        except ValueError:
            suffix = k.replace("type_", "")
            if len(suffix) == 1 and suffix.isalpha():
                numbers.append(ord(suffix.lower()) - ord('a'))
    
    if not numbers:
        return "type_0"
    return f"type_{max(numbers) + 1}"


def _normalize_features(features: Dict) -> List[float]:
    """Normalize features to [0,1] range for distance calculation."""
    ranges = {
        'modularity': (0, 1),
        'degree_variance': (0, 5),
        'hub_concentration': (0, 1),
        'avg_degree': (0, 100),
        'clustering_coefficient': (0, 1),
        'log_nodes': (3, 10),
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
        normalized.append(max(0, min(1, (val - lo) / (hi - lo) if hi > lo else 0.5)))
    
    return normalized


def _compute_distance(f1: List[float], f2: List[float]) -> float:
    """Compute Euclidean distance between normalized feature vectors."""
    if len(f1) != len(f2):
        return float('inf')
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(f1, f2)))


# =============================================================================
# Type Registry Functions
# =============================================================================

def load_type_registry(weights_dir: str = DEFAULT_WEIGHTS_DIR) -> Dict[str, Dict]:
    """Load type registry from disk."""
    global _type_registry, _type_registry_file
    _type_registry_file = os.path.join(weights_dir, "type_registry.json")
    
    if os.path.exists(_type_registry_file):
        try:
            with open(_type_registry_file) as f:
                _type_registry = json.load(f)
        except Exception:
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


def _normalize_legacy_weight_keys(weights: Dict) -> Dict:
    """Normalize legacy bare algorithm names in loaded weights.
    
    Maps old keys like 'GraphBrewOrder' → 'GraphBrewOrder_leiden' so that
    weight lookups work consistently after the variant-everywhere change.
    """
    _LEGACY = {'GraphBrewOrder': 'GraphBrewOrder_leiden'}
    normalized = {}
    for key, value in weights.items():
        new_key = _LEGACY.get(key, key)
        # If the new key already exists, merge by keeping the one with higher bias
        if new_key in normalized:
            existing_bias = normalized[new_key].get('bias', 0)
            new_bias = value.get('bias', 0) if isinstance(value, dict) else 0
            if new_bias > existing_bias:
                normalized[new_key] = value
        else:
            normalized[new_key] = value
    return normalized


def load_type_weights(type_name: str, weights_dir: str = DEFAULT_WEIGHTS_DIR) -> Dict:
    """Load weights for a specific type.
    
    Normalizes legacy algorithm names (e.g., 'GraphBrewOrder' →
    'GraphBrewOrder_leiden') for backward compatibility.
    """
    weights_file = get_type_weights_file(type_name, weights_dir)
    if os.path.exists(weights_file):
        try:
            with open(weights_file) as f:
                weights = json.load(f)
            return _normalize_legacy_weight_keys(weights)
        except Exception:
            pass
    return {}


def save_type_weights(type_name: str, weights: Dict, weights_dir: str = DEFAULT_WEIGHTS_DIR):
    """Save weights for a specific type."""
    os.makedirs(weights_dir, exist_ok=True)
    weights_file = get_type_weights_file(type_name, weights_dir)
    with open(weights_file, 'w') as f:
        json.dump(weights, f, indent=2)


# =============================================================================
# Type Assignment and Weight Updates
# =============================================================================

def assign_graph_type(
    features: Dict,
    weights_dir: str = DEFAULT_WEIGHTS_DIR,
    create_if_outlier: bool = True,
    graph_name: str = None,
) -> str:
    """
    Assign a graph to a type based on its features.
    
    Uses clustering to find the closest existing type or creates a new one
    if the graph is an outlier (distance > CLUSTER_DISTANCE_THRESHOLD).
    
    Args:
        features: Dict with 'modularity', 'degree_variance', 'hub_concentration', etc.
        weights_dir: Directory for type files
        create_if_outlier: If True, create new type for outliers
        graph_name: Optional graph name to track in registry
        
    Returns:
        Type name (e.g., 'type_0', 'type_1')
    """
    global _type_registry
    
    if not _type_registry:
        load_type_registry(weights_dir)
    
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
        type_info = _type_registry[closest_type]
        count = type_info.get('sample_count', 1)
        old_centroid = type_info['centroid']
        new_centroid = [
            old + (new - old) / (count + 1)
            for old, new in zip(old_centroid, norm_features)
        ]
        _type_registry[closest_type]['centroid'] = new_centroid
        _type_registry[closest_type]['sample_count'] = count + 1
        _type_registry[closest_type]['graph_count'] = count + 1
        _type_registry[closest_type]['last_updated'] = datetime.now().isoformat()
        
        # Track graph names
        if graph_name:
            if 'graphs' not in _type_registry[closest_type]:
                _type_registry[closest_type]['graphs'] = []
            if graph_name not in _type_registry[closest_type]['graphs']:
                _type_registry[closest_type]['graphs'].append(graph_name)
        
        save_type_registry(weights_dir)
        return closest_type
    
    # Create new type if outlier
    if create_if_outlier:
        new_type = _get_next_type_name()
        _type_registry[new_type] = {
            'centroid': norm_features,
            'sample_count': 1,
            'graph_count': 1,
            'algorithms': [],
            'graphs': [graph_name] if graph_name else [],
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
        
        if closest_type:
            existing_weights = load_type_weights(closest_type, weights_dir)
            if existing_weights:
                save_type_weights(new_type, existing_weights, weights_dir)
        
        return new_type
    
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
    weights = load_type_weights(type_name, weights_dir)
    if not weights:
        weights = {}
    
    # Track algorithm in type registry
    global _type_registry
    if not _type_registry:
        load_type_registry(weights_dir)
    if type_name in _type_registry:
        if 'algorithms' not in _type_registry[type_name]:
            _type_registry[type_name]['algorithms'] = []
        if algorithm not in _type_registry[type_name]['algorithms']:
            _type_registry[type_name]['algorithms'].append(algorithm)
            save_type_registry(weights_dir)
    
    # Initialize algorithm weights if not present
    if algorithm not in weights:
        weights[algorithm] = _create_default_weight_entry()
    
    algo_weights = weights[algorithm]
    meta = algo_weights.get('_metadata', {})
    
    # Update sample count and running average speedup
    count = meta.get('sample_count', 0) + 1
    old_avg = meta.get('avg_speedup', 1.0)
    new_avg = old_avg + (speedup - old_avg) / count
    
    meta['sample_count'] = count
    meta['avg_speedup'] = new_avg
    meta['last_updated'] = datetime.now().isoformat()
    
    if speedup > 1.05:
        meta['win_count'] = meta.get('win_count', 0) + 1
    if speedup > 1.0:
        meta['times_best'] = meta.get('times_best', 0) + 1
    
    meta['win_rate'] = meta['win_count'] / count if count > 0 else 0.0
    
    # Compute error signal
    current_score = algo_weights['bias']
    error = (speedup - 1.0) - current_score
    
    # Gradient update
    log_nodes = math.log10(features.get('nodes', 1) + 1)
    log_edges = math.log10(features.get('edges', 1) + 1)
    
    # Accept both clustering key names
    clustering = features.get('clustering_coefficient', features.get('clustering_coeff', 0.0))
    
    algo_weights['bias'] += learning_rate * error
    algo_weights['w_modularity'] += learning_rate * error * features.get('modularity', 0.0)
    algo_weights['w_log_nodes'] += learning_rate * error * log_nodes * 0.1
    algo_weights['w_log_edges'] += learning_rate * error * log_edges * 0.1
    algo_weights['w_density'] += learning_rate * error * features.get('density', 0.0)
    algo_weights['w_avg_degree'] += learning_rate * error * features.get('avg_degree', 0.0) / 100.0
    algo_weights['w_degree_variance'] += learning_rate * error * features.get('degree_variance', 0.0)
    algo_weights['w_hub_concentration'] += learning_rate * error * features.get('hub_concentration', 0.0)
    algo_weights['w_clustering_coeff'] += learning_rate * error * clustering
    algo_weights['w_community_count'] += learning_rate * error * features.get('community_count', 0) / 1000.0
    algo_weights['w_avg_path_length'] += learning_rate * error * features.get('avg_path_length', 0.0) / WEIGHT_PATH_LENGTH_NORMALIZATION
    algo_weights['w_diameter'] += learning_rate * error * features.get('diameter', 0.0) / 50.0
    
    # Locality features (IISWC'18 Packing Factor, GoGraph forward edge fraction)
    algo_weights['w_packing_factor'] += learning_rate * error * features.get('packing_factor', 0.0)
    algo_weights['w_forward_edge_fraction'] += learning_rate * error * features.get('forward_edge_fraction', 0.0)
    algo_weights['w_working_set_ratio'] += learning_rate * error * math.log2(features.get('working_set_ratio', 0.0) + 1.0)
    
    # Quadratic interaction gradients
    log_wsr = math.log2(features.get('working_set_ratio', 0.0) + 1.0)
    log_n = math.log10(features.get('nodes', 1) + 1)
    algo_weights['w_dv_x_hub'] = algo_weights.get('w_dv_x_hub', 0.0) + learning_rate * error * features.get('degree_variance', 0.0) * features.get('hub_concentration', 0.0)
    algo_weights['w_mod_x_logn'] = algo_weights.get('w_mod_x_logn', 0.0) + learning_rate * error * features.get('modularity', 0.0) * log_n
    algo_weights['w_pf_x_wsr'] = algo_weights.get('w_pf_x_wsr', 0.0) + learning_rate * error * features.get('packing_factor', 0.0) * log_wsr
    
    # Convergence bonus gradient (only for iterative benchmarks)
    # Use the benchmark parameter passed to this function, not from features dict
    if benchmark in ('pr', 'sssp'):
        algo_weights['w_fef_convergence'] = algo_weights.get('w_fef_convergence', 0.0) + learning_rate * error * features.get('forward_edge_fraction', 0.0)
    
    # Update cache weights
    if cache_stats:
        l1_hit = cache_stats.get('l1_hit_rate', 0) / 100.0
        l2_hit = cache_stats.get('l2_hit_rate', 0) / 100.0
        l3_hit = cache_stats.get('l3_hit_rate', 0) / 100.0
        dram_penalty = 1.0 - (l1_hit + l2_hit * 0.3 + l3_hit * 0.1)
        
        algo_weights['cache_l1_impact'] += learning_rate * error * l1_hit
        algo_weights['cache_l2_impact'] += learning_rate * error * l2_hit
        algo_weights['cache_l3_impact'] += learning_rate * error * l3_hit
        algo_weights['cache_dram_penalty'] += learning_rate * error * dram_penalty
        
        meta['avg_l1_hit_rate'] = meta.get('avg_l1_hit_rate', 0) + (l1_hit * 100 - meta.get('avg_l1_hit_rate', 0)) / count
        meta['avg_l2_hit_rate'] = meta.get('avg_l2_hit_rate', 0) + (l2_hit * 100 - meta.get('avg_l2_hit_rate', 0)) / count
        meta['avg_l3_hit_rate'] = meta.get('avg_l3_hit_rate', 0) + (l3_hit * 100 - meta.get('avg_l3_hit_rate', 0)) / count
    
    if reorder_time > 0:
        algo_weights['w_reorder_time'] += learning_rate * error * (-reorder_time / WEIGHT_REORDER_TIME_NORMALIZATION)
        meta['avg_reorder_time'] = meta.get('avg_reorder_time', 0) + (reorder_time - meta.get('avg_reorder_time', 0)) / count
    
    # Update benchmark-specific weight
    bench_weights = algo_weights.get('benchmark_weights', {})
    if benchmark.lower() in bench_weights:
        current_bench_weight = bench_weights[benchmark.lower()]
        bench_weights[benchmark.lower()] = current_bench_weight + learning_rate * error * 0.1
    
    # L2 weight decay (regularization) to prevent weight explosion
    # Applied to all feature weights (not bias, not metadata, not benchmark_weights)
    # Decay rate: 1e-4 per update — gentle enough to not interfere with learning
    WEIGHT_DECAY = 1e-4
    weight_keys = [k for k in algo_weights.keys() 
                   if k.startswith('w_') or k.startswith('cache_')]
    for k in weight_keys:
        if isinstance(algo_weights[k], (int, float)):
            algo_weights[k] *= (1.0 - WEIGHT_DECAY)
    
    algo_weights['_metadata'] = meta
    algo_weights['benchmark_weights'] = bench_weights
    weights[algorithm] = algo_weights
    
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
        if algo_name.startswith('_'):
            continue
        
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
            w_packing_factor=algo_weights.get('w_packing_factor', 0.0),
            w_forward_edge_fraction=algo_weights.get('w_forward_edge_fraction', 0.0),
            w_working_set_ratio=algo_weights.get('w_working_set_ratio', 0.0),
            w_dv_x_hub=algo_weights.get('w_dv_x_hub', 0.0),
            w_mod_x_logn=algo_weights.get('w_mod_x_logn', 0.0),
            w_pf_x_wsr=algo_weights.get('w_pf_x_wsr', 0.0),
            w_fef_convergence=algo_weights.get('w_fef_convergence', 0.0),
            benchmark_weights=algo_weights.get('benchmark_weights', {})
        )
        
        score = pw.compute_score(features, benchmark)
        
        # Confidence boost
        meta = algo_weights.get('_metadata', {})
        sample_count = meta.get('sample_count', 0)
        confidence_boost = min(0.1, sample_count * 0.01)
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
        for bench, weight in algo_weights.get('benchmark_weights', {}).items():
            if bench not in best_algos or weight > best_algos[bench][1]:
                best_algos[bench] = (algo_name, weight)
    
    return {
        'type_name': type_name,
        'num_graphs': type_info.get('sample_count', 0),
        'sample_count': type_info.get('sample_count', 0),
        'created': type_info.get('created', 'unknown'),
        'last_updated': type_info.get('last_updated', 'unknown'),
        'centroid': type_info.get('centroid', []),
        'representative_features': type_info.get('representative_features', {}),
        'algorithms': algo_stats,
        'best_algorithms': {bench: algo for bench, (algo, _) in best_algos.items()},
    }


def compute_weights_from_results(
    benchmark_results: List,
    cache_results: List = None,
    reorder_results: List = None,
    output_file: str = None,
    weights_dir: str = None,
) -> Dict:
    """
    Compute perceptron weights from benchmark results.
    
    This function analyzes benchmark results to determine which algorithms
    perform best for different graph types, and computes weights accordingly.
    
    Args:
        benchmark_results: List of BenchmarkResult objects
        cache_results: Optional list of CacheResult objects
        reorder_results: Optional list of ReorderResult objects
        output_file: Optional file to save weights
        weights_dir: Optional directory to save type-based weights
        
    Returns:
        Dict of weights by algorithm
    """
    cache_results = cache_results or []
    reorder_results = reorder_results or []
    if weights_dir is None:
        weights_dir = DEFAULT_WEIGHTS_DIR
    
    # Initialize weights - start with default then add from results
    weights = initialize_default_weights()
    
    # Normalize legacy bare algo names → variant-suffixed names in results
    # so old results keyed as "GraphBrewOrder" map to "GraphBrewOrder_leiden"
    _LEGACY_NAME_MAP = {'GraphBrewOrder': 'GraphBrewOrder_leiden'}
    
    for r in benchmark_results:
        if r.algorithm in _LEGACY_NAME_MAP:
            r.algorithm = _LEGACY_NAME_MAP[r.algorithm]
    for r in reorder_results:
        algo_name = getattr(r, 'algorithm_name', None) or getattr(r, 'algorithm', '')
        if algo_name in _LEGACY_NAME_MAP:
            if hasattr(r, 'algorithm_name'):
                r.algorithm_name = _LEGACY_NAME_MAP[algo_name]
            elif hasattr(r, 'algorithm'):
                r.algorithm = _LEGACY_NAME_MAP[algo_name]
    
    # Collect all unique algorithm names from results (includes variants)
    all_algo_names = set()
    for r in benchmark_results:
        if r.algorithm:
            all_algo_names.add(r.algorithm)
    for r in reorder_results:
        algo_name = getattr(r, 'algorithm_name', None) or getattr(r, 'algorithm', '')
        if algo_name:
            all_algo_names.add(algo_name)
    
    # Add variant algorithms that aren't in default weights
    for algo_name in all_algo_names:
        if algo_name not in weights and not algo_name.startswith('_'):
            weights[algo_name] = _create_default_weight_entry()
    
    # Group results by graph and benchmark
    results_by_graph = {}
    for r in benchmark_results:
        if not r.success or r.time_seconds <= 0:
            continue
        
        graph_name = r.graph
        if graph_name not in results_by_graph:
            results_by_graph[graph_name] = {}
        
        bench = r.benchmark
        if bench not in results_by_graph[graph_name]:
            results_by_graph[graph_name][bench] = []
        
        results_by_graph[graph_name][bench].append(r)
    
    # Find best algorithm per (graph, benchmark) and collect win statistics
    algo_win_count = {}   # algo -> number of (graph, benchmark) contexts won
    algo_speedups = {}    # algo -> list of speedups when winning
    total_contexts = 0
    
    for graph_name, benchmarks in results_by_graph.items():
        for bench, results in benchmarks.items():
            if not results:
                continue
            total_contexts += 1
            
            # Find baseline (ORIGINAL or first result) — use total time (exec + reorder)
            baseline_time = None
            for r in results:
                if 'ORIGINAL' in r.algorithm:
                    baseline_time = r.time_seconds + r.reorder_time
                    break
            if baseline_time is None:
                baseline_time = results[0].time_seconds + results[0].reorder_time
            
            # Find best by total time (exec + reorder) — practical end-to-end metric
            best_result = min(results, key=lambda r: r.time_seconds + r.reorder_time)
            best_algo = best_result.algorithm
            
            if best_algo in weights and baseline_time > 0:
                best_total = best_result.time_seconds + best_result.reorder_time
                speedup = baseline_time / best_total
                algo_win_count[best_algo] = algo_win_count.get(best_algo, 0) + 1
                if best_algo not in algo_speedups:
                    algo_speedups[best_algo] = []
                algo_speedups[best_algo].append(speedup)
    
    # Compute bias from win rate: scale to [0.3, 1.0] range
    # This gives a max spread of 0.7 — features can easily override this
    for algo in weights:
        if algo.startswith('_'):
            continue
        wins = algo_win_count.get(algo, 0)
        win_rate = wins / total_contexts if total_contexts > 0 else 0.0
        # Bias = 0.3 + 0.7 * win_rate  (never won → 0.3, won everything → 1.0)
        weights[algo]['bias'] = round(0.3 + 0.7 * win_rate, 4)
        
        # Update metadata
        meta = weights[algo].get('_metadata', {})
        meta['win_count'] = wins
        meta['sample_count'] = wins
        if algo in algo_speedups:
            meta['avg_speedup'] = sum(algo_speedups[algo]) / len(algo_speedups[algo])
        weights[algo]['_metadata'] = meta
    
    # =========================================================================
    # Multi-class Perceptron Training
    # =========================================================================
    # Train a multi-class perceptron via the standard update rule:
    #   For each training example (graph_features, best_algorithm):
    #     predicted = argmax_algo score(algo, features)
    #     if predicted != best_algorithm:
    #       w[best] += learning_rate * features
    #       w[predicted] -= learning_rate * features
    # This directly optimizes for correct algorithm selection.
    
    import math
    from .features import load_graph_properties_cache
    from .utils import RESULTS_DIR
    
    graph_props = load_graph_properties_cache(RESULTS_DIR)
    
    # Feature keys used in C++ scoreBase()
    feat_to_weight = {
        'modularity': 'w_modularity',
        'degree_variance': 'w_degree_variance',
        'hub_concentration': 'w_hub_concentration',
        'log_nodes': 'w_log_nodes',
        'log_edges': 'w_log_edges',
        'density': 'w_density',
        'avg_degree': 'w_avg_degree',
        'clustering_coefficient': 'w_clustering_coeff',
        'avg_path_length': 'w_avg_path_length',
        'diameter': 'w_diameter',
        'community_count': 'w_community_count',
    }
    
    # Collect features per graph
    graph_features = {}
    for graph_name, props in graph_props.items():
        nodes = props.get('nodes', 1000)
        edges = props.get('edges', 5000)
        cc = props.get('clustering_coefficient', 0.0)
        avg_degree = props.get('avg_degree', WEIGHT_AVG_DEGREE_DEFAULT)
        
        # Align features to match what C++ computes at runtime:
        # - modularity: C++ uses estimated_modularity = min(0.9, clustering_coeff * 1.5)
        # - density: C++ uses internal_density = avg_degree / (num_nodes - 1)
        # - avg_path_length, diameter, community_count: C++ doesn't compute these
        estimated_modularity = min(0.9, cc * 1.5)
        internal_density = avg_degree / (nodes - 1) if nodes > 1 else 0
        
        graph_features[graph_name] = {
            'modularity': estimated_modularity,
            'degree_variance': props.get('degree_variance', 1.0),
            'hub_concentration': props.get('hub_concentration', 0.3),
            'avg_degree': avg_degree,
            'log_nodes': math.log10(nodes + 1) if nodes > 0 else 0,
            'log_edges': math.log10(edges + 1) if edges > 0 else 0,
            'density': internal_density,
            'clustering_coefficient': cc,
            'avg_path_length': 0.0,  # C++ doesn't compute at runtime
            'diameter': 0.0,         # C++ doesn't compute at runtime
            'community_count': 0.0,  # C++ doesn't compute at runtime
        }
    
    # Build training examples: (feature_vector, best_algorithm)
    # Map variant names to base algorithms for training (more examples per class)
    _VARIANT_PREFIXES = [
        'GraphBrewOrder_', 'RABBITORDER_'
    ]
    
    def _get_base(name):
        for prefix in _VARIANT_PREFIXES:
            if name.startswith(prefix):
                return prefix.rstrip('_')
        return name
    
    training_data = []
    for graph_name, benchmarks in results_by_graph.items():
        if graph_name not in graph_features:
            continue
        feats = graph_features[graph_name]
        
        for bench, results in benchmarks.items():
            if not results:
                continue
            # Oracle by total time (exec + reorder) — practical end-to-end metric
            best_result = min(results, key=lambda r: r.time_seconds + r.reorder_time)
            best_algo = _get_base(best_result.algorithm)
            
            # Build feature vector matching C++ scoreBase() scaling
            fv = [
                feats['modularity'],
                feats['degree_variance'],
                feats['hub_concentration'],
                feats['log_nodes'],
                feats['log_edges'],
                feats['density'],
                feats['avg_degree'] / 100.0,
                feats['clustering_coefficient'],
                feats['avg_path_length'] / 10.0,
                feats['diameter'] / 50.0,
                math.log10(feats['community_count'] + 1) if feats['community_count'] > 0 else 0,
            ]
            
            training_data.append((fv, best_algo))
    
    if training_data and graph_features:
        # Get base algorithm names for training
        base_algos = sorted(set(_get_base(a) for a in weights if not a.startswith('_')))
        
        # =====================================================================
        # Per-benchmark perceptron training → C++ compatible output
        # =====================================================================
        # Strategy:
        #   1. Train separate perceptrons per benchmark (high accuracy each)
        #   2. Feature weights = average across benchmarks (shared scoreBase)
        #   3. Per-bench residual → benchmark_weights multiplier
        #   4. De-normalize features for C++ raw-feature scoring
        #
        # This matches C++ model: score = scoreBase(feat) * benchmarkMultiplier(bench)
        
        import random
        random.seed(42)
        
        weight_keys = list(feat_to_weight.values())
        n_feat = len(weight_keys)
        
        # Collect bench names
        bench_names = sorted(set(
            bench for benchmarks in results_by_graph.values()
            for bench in benchmarks.keys()
        ))
        
        # Build per-benchmark training data with raw features
        per_bench_data_raw = {bn: [] for bn in bench_names}
        for graph_name, benchmarks in results_by_graph.items():
            if graph_name not in graph_features:
                continue
            feats = graph_features[graph_name]
            fv = [
                feats['modularity'],
                feats['degree_variance'],
                feats['hub_concentration'],
                feats['log_nodes'],
                feats['log_edges'],
                feats['density'],
                feats['avg_degree'] / 100.0,
                feats['clustering_coefficient'],
                feats['avg_path_length'] / 10.0,
                feats['diameter'] / 50.0,
                math.log10(feats['community_count'] + 1) if feats['community_count'] > 0 else 0,
            ]
            for bench, results in benchmarks.items():
                if not results:
                    continue
                # Oracle by total time (exec + reorder) — practical end-to-end metric
                best_result = min(results, key=lambda r: r.time_seconds + r.reorder_time)
                best_algo = _get_base(best_result.algorithm)
                per_bench_data_raw[bench].append((fv, best_algo))
        
        # Feature normalization stats
        all_fvs = [fv for bn in bench_names for fv, _ in per_bench_data_raw[bn]]
        feat_means = [0.0] * n_feat
        feat_stds = [1.0] * n_feat
        if all_fvs:
            for i in range(n_feat):
                vals = [fv[i] for fv in all_fvs]
                feat_means[i] = sum(vals) / len(vals)
                var = sum((v - feat_means[i])**2 for v in vals) / len(vals)
                feat_stds[i] = max(math.sqrt(var), 1e-8)
        
        # Normalize per-benchmark data
        per_bench_data = {}
        for bn in bench_names:
            per_bench_data[bn] = [
                ([(fv[i] - feat_means[i]) / feat_stds[i] for i in range(n_feat)], algo)
                for fv, algo in per_bench_data_raw[bn]
            ]
        
        # Train one perceptron per benchmark with multiple random restarts
        per_bench_w = {}  # bench -> {base -> {'bias': float, 'w': [float]}}
        N_RESTARTS = 5
        N_EPOCHS = 800
        
        for bn in bench_names:
            data = per_bench_data[bn]
            if not data:
                continue
            
            global_best_acc = 0
            global_best_snap = None
            
            for restart in range(N_RESTARTS):
                # Use benchmark index for deterministic seeding
                bn_idx = bench_names.index(bn)
                random.seed(42 + restart * 1000 + bn_idx * 100)
                
                # Initialize with small random weights for diversity
                bw = {base: {
                    'bias': random.gauss(0, 0.1),
                    'w': [random.gauss(0, 0.1) for _ in range(n_feat)]
                } for base in base_algos}
                
                lr = 0.05
                best_acc = 0
                best_snap = None
                
                for epoch in range(N_EPOCHS):
                    random.shuffle(data)
                    correct = 0
                    for fv_n, true_base in data:
                        if true_base not in bw:
                            continue
                        # Predict
                        best_s, pred = float('-inf'), None
                        for base in base_algos:
                            s = bw[base]['bias'] + sum(bw[base]['w'][i]*fv_n[i] for i in range(n_feat))
                            if s > best_s:
                                best_s, pred = s, base
                        if pred == true_base:
                            correct += 1
                        else:
                            bw[true_base]['bias'] += lr
                            bw[pred]['bias'] -= lr
                            for i in range(n_feat):
                                bw[true_base]['w'][i] += lr * fv_n[i]
                                bw[pred]['w'][i] -= lr * fv_n[i]
                    
                    acc = correct / len(data) if data else 0
                    if acc > best_acc:
                        best_acc = acc
                        best_snap = {
                            base: {'bias': d['bias'], 'w': list(d['w'])}
                            for base, d in bw.items()
                        }
                    lr *= 0.997
                
                if best_acc > global_best_acc:
                    global_best_acc = best_acc
                    global_best_snap = best_snap
            
            if global_best_snap:
                per_bench_w[bn] = global_best_snap
            log.info(f"  {bn}: accuracy = {global_best_acc:.1%} ({len(data)} examples, "
                     f"{N_RESTARTS} restarts)")
        
        # =====================================================================
        # Combine into C++ weight format
        # =====================================================================
        # For each algo, score on all training graphs with each per-bench model.
        # Use the per-bench model that "matches" the C++ multiplicative form best.
        #
        # Approach: compute each algo's ranking score on each training graph
        # using per-bench models, then find bias + feature_weights + bench_multipliers
        # that best reproduce those scores.
        #
        # =====================================================================
        # Save per-benchmark perceptrons as separate weight files (type_0_{bench}.json)
        # These are loaded by C++ when benchmark type hint is available, giving
        # much higher accuracy than the averaged scoreBase × multiplier model.
        # =====================================================================
        
        for bn, bn_weights in per_bench_w.items():
            per_bench_cpp = {}
            for base in base_algos:
                if base not in bn_weights:
                    continue
                bw_raw = bn_weights[base]
                # De-normalize for C++ raw features (same transform as averaged weights)
                bias_adj = sum(bw_raw['w'][i] * feat_means[i] / feat_stds[i] for i in range(n_feat))
                denorm_bias = bw_raw['bias'] - bias_adj
                denorm_w = {weight_keys[i]: bw_raw['w'][i] / feat_stds[i] for i in range(n_feat)}
                
                entry = {'bias': denorm_bias}
                entry.update(denorm_w)
                # No benchmark_weights needed - this IS the benchmark-specific perceptron
                entry['benchmark_weights'] = {}
                entry['_metadata'] = {}
                per_bench_cpp[base] = entry
            
            if per_bench_cpp and weights_dir:
                bench_file = os.path.join(weights_dir, f'type_0_{bn}.json')
                with open(bench_file, 'w') as f:
                    json.dump(per_bench_cpp, f, indent=2)
                log.info(f"  Saved per-benchmark weights: {bench_file} ({len(per_bench_cpp)} algorithms)")
        
        # =====================================================================
        # ScoreBase from averaged per-bench perceptrons + regret-aware grid search
        # =====================================================================
        
        base_weights = {}
        bench_multipliers = {}
        
        # Build scoreBase from averaged per-bench perceptrons
        for base in base_algos:
            avg_bias = 0.0
            avg_w = [0.0] * n_feat
            n_benches = len(per_bench_w)
            
            for bn in bench_names:
                if bn not in per_bench_w:
                    continue
                avg_bias += per_bench_w[bn][base]['bias']
                for i in range(n_feat):
                    avg_w[i] += per_bench_w[bn][base]['w'][i]
            
            if n_benches > 0:
                avg_bias /= n_benches
                avg_w = [w / n_benches for w in avg_w]
            
            # De-normalize for C++ raw features
            bias_adj = sum(avg_w[i] * feat_means[i] / feat_stds[i] for i in range(n_feat))
            denorm_bias = avg_bias - bias_adj
            denorm_w = {weight_keys[i]: avg_w[i] / feat_stds[i] for i in range(n_feat)}
            
            base_weights[base] = {'bias': denorm_bias}
            base_weights[base].update(denorm_w)
            bench_multipliers[base] = {bn: 1.0 for bn in bench_names}
        
        # Helper: compute raw feature vector for a graph
        def _make_fv(feats):
            return [
                feats['modularity'],
                feats['degree_variance'],
                feats['hub_concentration'],
                feats['log_nodes'],
                feats['log_edges'],
                feats['density'],
                feats['avg_degree'] / 100.0,
                feats['clustering_coefficient'],
                feats['avg_path_length'] / 10.0,
                feats['diameter'] / 50.0,
                math.log10(feats['community_count'] + 1) if feats['community_count'] > 0 else 0,
            ]
        
        # Helper: compute scoreBase for an algo on a graph
        def _score_base(bw, fv):
            return bw['bias'] + sum(bw.get(weight_keys[i], 0) * fv[i] for i in range(n_feat))
        
        # Build ground truth + timing data for all benchmarks
        bench_truth_all = {}  # bench -> {graph -> best_base}
        bench_times_all = {}  # bench -> {graph -> {base: best_time}}
        bench_best_time_all = {}  # bench -> {graph -> best_time}
        graph_fvs = {}  # graph -> feature vector
        
        for gn, benchmarks_data in results_by_graph.items():
            if gn not in graph_features:
                continue
            graph_fvs[gn] = _make_fv(graph_features[gn])
            
            for bn, results in benchmarks_data.items():
                if not results:
                    continue
                best = min(results, key=lambda r: r.time_seconds)
                
                if bn not in bench_truth_all:
                    bench_truth_all[bn] = {}
                    bench_times_all[bn] = {}
                    bench_best_time_all[bn] = {}
                
                bench_truth_all[bn][gn] = _get_base(best.algorithm)
                bench_best_time_all[bn][gn] = best.time_seconds
                
                base_t = {}
                for r in results:
                    b = _get_base(r.algorithm)
                    if b not in base_t or r.time_seconds < base_t[b]:
                        base_t[b] = r.time_seconds
                bench_times_all[bn][gn] = base_t
        
        graph_names_list = sorted(graph_fvs.keys())
        
        # Precompute scoreBase matrix
        score_base_matrix = {}  # (base, graph) -> score
        for base in base_algos:
            bw = base_weights[base]
            for gn in graph_names_list:
                score_base_matrix[(base, gn)] = _score_base(bw, graph_fvs[gn])
        
        MULT_GRID = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4,
                     0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                     1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0,
                     3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0,
                     12.0, 15.0, 20.0]
        
        # Regret-aware grid search for multipliers
        for bn in bench_names:
            if bn not in bench_truth_all:
                continue
            
            current_mults = {base: 1.0 for base in base_algos}
            
            def _eval_bench(mults):
                """(accuracy, -mean_regret) for tie-breaking."""
                correct = 0
                total_regret = 0.0
                for gn in bench_truth_all[bn]:
                    scores = {}
                    for base in base_algos:
                        scores[base] = score_base_matrix.get((base, gn), 0) * mults[base]
                    pred = max(scores, key=scores.get)
                    if pred == bench_truth_all[bn][gn]:
                        correct += 1
                    else:
                        bt = bench_best_time_all[bn].get(gn, 1.0)
                        pt = bench_times_all[bn].get(gn, {}).get(pred, bt * 10)
                        total_regret += min((pt - bt) / max(bt, 1e-6), 50.0)
                n = max(len(bench_truth_all[bn]), 1)
                return (correct, -total_regret / n)
            
            for iteration in range(30):
                improved = False
                for opt_base in base_algos:
                    best_mult = current_mults[opt_base]
                    best_score = _eval_bench(current_mults)
                    
                    for trial_mult in MULT_GRID:
                        old_mult = current_mults[opt_base]
                        current_mults[opt_base] = trial_mult
                        trial_score = _eval_bench(current_mults)
                        if trial_score > best_score:
                            best_score = trial_score
                            best_mult = trial_mult
                            improved = True
                        current_mults[opt_base] = old_mult
                    
                    current_mults[opt_base] = best_mult
                
                if not improved:
                    break
            
            for base in base_algos:
                bench_multipliers[base][bn] = round(current_mults[base], 4)
            
            # Log
            correct = 0
            for gn in bench_truth_all[bn]:
                scores = {}
                for base in base_algos:
                    scores[base] = score_base_matrix.get((base, gn), 0) * current_mults[base]
                pred = max(scores, key=scores.get)
                if pred == bench_truth_all[bn][gn]:
                    correct += 1
            total_bn = len(bench_truth_all[bn])
            log.info(f"  {bn}: mult-opt accuracy = {correct}/{total_bn} "
                     f"= {correct/total_bn:.1%}")
        
        log.info(f"Per-benchmark perceptron → C++ weights: "
                 f"{len(bench_names)} benchmarks × {len(base_algos)} algorithms")
        
        # Map trained base weights to the best variant per base
        # For each base, pick the variant with the LOWEST median slowdown
        # relative to the best variant for each graph. This avoids picking
        # a variant that wins often but loses catastrophically sometimes.
        
        # Build variant performance data: for each variant on each graph,
        # compute ratio vs best variant of same base
        variant_ratios = {}  # variant -> [ratio1, ratio2, ...]
        variant_wins = {}  # variant -> win count
        for _, benchmarks_data in results_by_graph.items():
            for _, results in benchmarks_data.items():
                if not results:
                    continue
                best = min(results, key=lambda r: r.time_seconds)
                variant_wins[best.algorithm] = variant_wins.get(best.algorithm, 0) + 1
                
                # Group by base, find best per base
                base_best = {}  # base -> best_time
                base_variants = {}  # base -> [(variant, time)]
                for r in results:
                    b = _get_base(r.algorithm)
                    if b not in base_variants:
                        base_variants[b] = []
                    base_variants[b].append((r.algorithm, r.time_seconds))
                    if b not in base_best or r.time_seconds < base_best[b]:
                        base_best[b] = r.time_seconds
                
                for b, variants in base_variants.items():
                    bt = base_best[b]
                    if bt <= 0:
                        continue
                    for var, t in variants:
                        if var not in variant_ratios:
                            variant_ratios[var] = []
                        variant_ratios[var].append(t / bt)
        
        # For each base, pick variant with lowest median ratio
        base_best_variant = {}  # base -> best variant name
        for algo in weights:
            if algo.startswith('_'):
                continue
            base = _get_base(algo)
            ratios = variant_ratios.get(algo, [])
            median_ratio = sorted(ratios)[len(ratios)//2] if ratios else float('inf')
            
            if base not in base_best_variant:
                base_best_variant[base] = (algo, median_ratio)
            else:
                _, best_ratio = base_best_variant[base]
                if median_ratio < best_ratio:
                    base_best_variant[base] = (algo, median_ratio)
        
        base_best_variant = {b: v[0] for b, v in base_best_variant.items()}
        
        # Apply trained base weights to all variants
        # Map per-benchmark multipliers to C++ benchmark_weights format
        for algo in weights:
            if algo.startswith('_'):
                continue
            base = _get_base(algo)
            if base in base_weights:
                bw = base_weights[base]
                weights[algo]['bias'] = bw['bias']
                for wk in feat_to_weight.values():
                    weights[algo][wk] = bw.get(wk, 0)
                
                # Set benchmark_weights from per-benchmark perceptron ratios
                bm = bench_multipliers.get(base, {})
                bw_dict = weights[algo].get('benchmark_weights', {})
                for bn in bench_names:
                    bw_dict[bn] = bm.get(bn, 1.0)
                weights[algo]['benchmark_weights'] = bw_dict
                
                # Give the best variant per base a small bias boost
                # so C++ variant collapsing picks it
                if algo == base_best_variant.get(base):
                    weights[algo]['bias'] += 0.01
        
        # Restore metadata
        for algo in [a for a in weights if not a.startswith('_')]:
            meta = weights[algo].get('_metadata', {})
            meta['win_count'] = algo_win_count.get(algo, 0)
            meta['sample_count'] = algo_win_count.get(algo, 0)
            if algo in algo_speedups:
                meta['avg_speedup'] = sum(algo_speedups[algo]) / len(algo_speedups[algo])
            weights[algo]['_metadata'] = meta
    
    # Update reorder time weights from reorder results
    reorder_times = {}
    for r in reorder_results:
        algo = getattr(r, 'algorithm_name', None) or getattr(r, 'algorithm', '')
        if not algo:
            continue
        if algo not in reorder_times:
            reorder_times[algo] = []
        reorder_time = getattr(r, 'reorder_time', 0.0) or getattr(r, 'time_seconds', 0.0)
        if reorder_time > 0:
            reorder_times[algo].append(reorder_time)
    
    for algo, times in reorder_times.items():
        if algo in weights and times:
            avg_time = sum(times) / len(times)
            weights[algo]['w_reorder_time'] = -avg_time / WEIGHT_REORDER_TIME_NORMALIZATION
            meta = weights[algo].get('_metadata', {})
            meta['avg_reorder_time'] = avg_time
            weights[algo]['_metadata'] = meta
    
    # Save if output file specified (DEPRECATED - kept for backward compatibility)
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(weights, f, indent=2)
    
    # =========================================================================
    # Pre-collapse variants before saving to JSON
    # =========================================================================
    # C++ ParseWeightsFromJSON collapses variants by highest bias, but with
    # discriminative feature weights, different variants may be better for
    # different graph types. To work around C++ highest-bias collapsing:
    #   For each base algorithm, score each variant on the MEAN feature vector
    #   of the training graphs, and keep only the best-scoring variant.
    #   Set that variant's bias to be highest so C++ will select it.
    
    # Compute mean features across training graphs
    if graph_features:
        feat_keys = list(next(iter(graph_features.values())).keys())
        mean_features = {}
        for fk in feat_keys:
            vals = [gf[fk] for gf in graph_features.values() if fk in gf]
            mean_features[fk] = sum(vals) / len(vals) if vals else 0.0
        
        # Score each variant on mean features
        def score_on_mean(data):
            s = data.get('bias', 0.3)
            s += data.get('w_modularity', 0) * mean_features.get('modularity', 0.5)
            s += data.get('w_log_nodes', 0) * mean_features.get('log_nodes', 5.0)
            s += data.get('w_log_edges', 0) * mean_features.get('log_edges', 6.0)
            s += data.get('w_density', 0) * mean_features.get('density', 0.001)
            s += data.get('w_degree_variance', 0) * mean_features.get('degree_variance', 1.0)
            s += data.get('w_hub_concentration', 0) * mean_features.get('hub_concentration', 0.3)
            s += data.get('w_avg_degree', 0) * mean_features.get('avg_degree', 10.0) / 100.0
            return s
        
        # Group variants by base algorithm
        _VARIANT_PREFIXES = [
            'GraphBrewOrder_', 'RABBITORDER_'
        ]
        
        def get_base(name):
            for prefix in _VARIANT_PREFIXES:
                if name.startswith(prefix):
                    return prefix.rstrip('_')
            return name
        
        base_groups = {}  # base -> [(variant_name, data, wins)]
        for algo, data in weights.items():
            if algo.startswith('_'):
                continue
            base = get_base(algo)
            wins = variant_wins.get(algo, 0)
            if base not in base_groups:
                base_groups[base] = []
            base_groups[base].append((algo, data, wins))
        
        # For each base with multiple variants, keep only the one with most wins
        collapsed_weights = {}
        for base, variants in base_groups.items():
            if len(variants) == 1:
                algo, data, _ = variants[0]
                collapsed_weights[algo] = data
            else:
                # Sort by win count descending
                variants.sort(key=lambda x: -x[2])
                best_algo, best_data, best_wins = variants[0]
                
                # Only emit the best variant (others are redundant for C++)
                collapsed_weights[best_algo] = best_data
        
        # Preserve metadata keys
        for k, v in weights.items():
            if k.startswith('_'):
                collapsed_weights[k] = v
        
        log.info(f"Pre-collapsed {len(weights) - len(collapsed_weights)} redundant variants "
                     f"({len(weights)} → {len(collapsed_weights)} entries)")
        
        save_weights_to_active_type(collapsed_weights, weights_dir, type_name="type_0")
    else:
        # No graph features available, save all variants
        save_weights_to_active_type(weights, weights_dir, type_name="type_0")
    
    return weights


def cross_validate_logo(
    benchmark_results: List,
    reorder_results: List = None,
    weights_dir: str = None,
) -> Dict:
    """
    Leave-One-Graph-Out (LOGO) cross-validation for perceptron weights.

    For each graph in the results:
      1. Train weights on all OTHER graphs (full perceptron, per-benchmark)
      2. Use trained weights + held-out graph features to predict best algorithm
      3. Compare prediction to actual best algorithm (oracle by total time)

    This measures generalization accuracy: how well the model predicts on
    graphs it has never seen during training.

    Args:
        benchmark_results: List of BenchmarkResult objects
        reorder_results: Optional list of ReorderResult objects
        weights_dir: Weights directory (used for type registry and graph props)

    Returns:
        Dict with keys: accuracy, per_graph, overfitting_score, regret metrics
    """
    import math, tempfile, json
    from .features import load_graph_properties_cache
    from .utils import RESULTS_DIR

    reorder_results = reorder_results or []
    if weights_dir is None:
        weights_dir = DEFAULT_WEIGHTS_DIR

    _VARIANT_PREFIXES = ['GraphBrewOrder_', 'RABBITORDER_']

    def _get_base(name):
        for prefix in _VARIANT_PREFIXES:
            if name.startswith(prefix):
                return prefix.rstrip('_')
        return name

    def _build_features(props):
        """Build C++-aligned feature dict from graph properties."""
        nodes = props.get('nodes', 1000)
        edges = props.get('edges', 5000)
        cc = props.get('clustering_coefficient', 0.0)
        avg_degree = props.get('avg_degree', 10.0)
        return {
            'modularity': min(0.9, cc * 1.5),
            'degree_variance': props.get('degree_variance', 1.0),
            'hub_concentration': props.get('hub_concentration', 0.3),
            'avg_degree': avg_degree,
            'log_nodes': math.log10(nodes + 1) if nodes > 0 else 0,
            'log_edges': math.log10(edges + 1) if edges > 0 else 0,
            'density': avg_degree / (nodes - 1) if nodes > 1 else 0,
            'clustering_coefficient': cc,
        }

    def _score(algo_data, feats):
        """Mimic C++ scoreBase() — must match reorder_adaptive.h."""
        s = algo_data.get('bias', 0.5)
        s += algo_data.get('w_modularity', 0) * feats.get('modularity', 0.0)
        s += algo_data.get('w_log_nodes', 0) * feats.get('log_nodes', 5.0)
        s += algo_data.get('w_log_edges', 0) * feats.get('log_edges', 6.0)
        s += algo_data.get('w_density', 0) * feats.get('density', 0.0)
        s += algo_data.get('w_avg_degree', 0) * feats.get('avg_degree', 10.0) / 100.0
        s += algo_data.get('w_degree_variance', 0) * feats.get('degree_variance', 1.0)
        s += algo_data.get('w_hub_concentration', 0) * feats.get('hub_concentration', 0.3)
        s += algo_data.get('w_clustering_coeff', 0) * feats.get('clustering_coefficient', 0.0)
        return s

    # Load graph properties for feature-based scoring
    graph_props = load_graph_properties_cache(RESULTS_DIR)

    # Collect graphs and group results
    graphs = set()
    for r in benchmark_results:
        if r.success and r.time_seconds > 0:
            graphs.add(r.graph)
    graphs = sorted(graphs)

    if len(graphs) < 3:
        return {'accuracy': 0.0, 'per_graph': {}, 'overfitting_score': 0.0,
                'error': 'Need at least 3 graphs for LOGO validation'}

    # Build per-(graph, bench) result lists using total time
    gb_results = {}
    for r in benchmark_results:
        if r.success and r.time_seconds > 0:
            key = (r.graph, r.benchmark)
            if key not in gb_results:
                gb_results[key] = []
            gb_results[key].append((r.algorithm, r.time_seconds + r.reorder_time))

    correct = 0
    total = 0
    regrets = []
    per_graph = {}
    per_bench_names = ['pr', 'bfs', 'cc', 'sssp', 'bc', 'tc', 'pr_spmv', 'cc_sv']

    log.info(f"LOGO: {len(graphs)} graphs, {len(gb_results)} (graph, bench) tasks")

    for held_out in graphs:
        # Train on all graphs except held_out
        train_results = [r for r in benchmark_results if r.graph != held_out]
        train_reorder = [r for r in reorder_results
                         if getattr(r, 'graph', '') != held_out]

        with tempfile.TemporaryDirectory() as tmpdir:
            compute_weights_from_results(
                train_results,
                reorder_results=train_reorder,
                weights_dir=tmpdir,
            )

            # Load the produced weight files (per-benchmark when available)
            type0_file = os.path.join(tmpdir, 'type_0.json')
            if not os.path.isfile(type0_file):
                continue
            with open(type0_file) as f:
                saved_algos = {k: v for k, v in json.load(f).items()
                               if not k.startswith('_')}

            per_bench_weights = {}
            for bn in per_bench_names:
                bfile = os.path.join(tmpdir, f'type_0_{bn}.json')
                if os.path.isfile(bfile):
                    with open(bfile) as f:
                        per_bench_weights[bn] = json.load(f)

        # Build features for held-out graph
        if held_out not in graph_props:
            continue
        feats = _build_features(graph_props[held_out])

        graph_correct = 0
        graph_total = 0
        graph_regrets = []

        for bench in sorted(set(b for (g, b) in gb_results if g == held_out)):
            key = (held_out, bench)
            if key not in gb_results:
                continue
            algo_times = gb_results[key]

            # Predict with feature-based scoring (per-bench weights preferred)
            scoring_algos = per_bench_weights.get(bench, saved_algos)
            best_score = float('-inf')
            predicted_algo = None
            for algo, data in scoring_algos.items():
                if algo.startswith('_'):
                    continue
                s = _score(data, feats)
                if s > best_score:
                    best_score = s
                    predicted_algo = algo

            # Ground truth: fastest algorithm by total time
            sorted_times = sorted(algo_times, key=lambda x: x[1])
            actual_algo = sorted_times[0][0]
            best_time = sorted_times[0][1]

            # Compare using base names (variant-aware)
            pred_base = _get_base(predicted_algo) if predicted_algo else ''
            actual_base = _get_base(actual_algo)
            is_correct = pred_base == actual_base

            if is_correct:
                graph_correct += 1
                correct += 1
            total += 1
            graph_total += 1

            # Regret
            pred_time = None
            for a, t in algo_times:
                if a == predicted_algo:
                    pred_time = t
                    break
            if pred_time is None:
                for a, t in algo_times:
                    if _get_base(a) == pred_base:
                        pred_time = t
                        break
            if pred_time is None:
                pred_time = sorted_times[-1][1]  # worst case

            if best_time > 0:
                r = (pred_time - best_time) / best_time * 100
                regrets.append(r)
                graph_regrets.append(r)

        per_graph[held_out] = {
            'correct': graph_correct,
            'total': graph_total,
            'accuracy': graph_correct / graph_total if graph_total > 0 else 0.0,
            'avg_regret': sum(graph_regrets) / len(graph_regrets) if graph_regrets else 0.0,
        }

    logo_accuracy = correct / total if total > 0 else 0.0
    avg_regret = sum(regrets) / len(regrets) if regrets else 0.0
    sorted_regrets = sorted(regrets) if regrets else [0.0]
    median_regret = sorted_regrets[len(sorted_regrets) // 2]

    # Compare to in-sample accuracy (full training on all graphs)
    full_weights = compute_weights_from_results(
        benchmark_results, reorder_results=reorder_results, weights_dir=weights_dir
    )
    # Reload produced files for scoring
    type0_file = os.path.join(weights_dir, 'type_0.json')
    full_algos = {}
    full_per_bench = {}
    if os.path.isfile(type0_file):
        with open(type0_file) as f:
            full_algos = {k: v for k, v in json.load(f).items()
                          if not k.startswith('_')}
    for bn in per_bench_names:
        bfile = os.path.join(weights_dir, f'type_0_{bn}.json')
        if os.path.isfile(bfile):
            with open(bfile) as f:
                full_per_bench[bn] = json.load(f)

    full_correct = 0
    full_total = 0
    for (graph_name, bench), algo_times in gb_results.items():
        if graph_name not in graph_props:
            continue
        feats = _build_features(graph_props[graph_name])
        scoring_algos = full_per_bench.get(bench, full_algos)
        best_score = float('-inf')
        predicted = None
        for algo, data in scoring_algos.items():
            if algo.startswith('_'):
                continue
            s = _score(data, feats)
            if s > best_score:
                best_score = s
                predicted = algo
        actual = sorted(algo_times, key=lambda x: x[1])[0][0]
        if _get_base(predicted or '') == _get_base(actual):
            full_correct += 1
        full_total += 1

    full_accuracy = full_correct / full_total if full_total > 0 else 0.0
    overfitting_score = full_accuracy - logo_accuracy

    return {
        'accuracy': logo_accuracy,
        'full_training_accuracy': full_accuracy,
        'overfitting_score': overfitting_score,
        'avg_regret': avg_regret,
        'median_regret': median_regret,
        'num_graphs': len(graphs),
        'correct': correct,
        'total': total,
        'per_graph': per_graph,
        'warning': 'Possible overfitting' if overfitting_score > 0.2 else 'OK',
    }


def save_weights_to_active_type(
    weights: Dict,
    weights_dir: str = None,
    type_name: str = "type_0",
    graphs: List[str] = None,
) -> str:
    """
    Save weights to active type-based weights directory for C++ to use.
    
    This creates/updates:
    - scripts/weights/active/type_N.json - Algorithm weights
    - scripts/weights/active/type_registry.json - Type registry
    
    Args:
        weights: Dictionary of algorithm weights
        weights_dir: Directory to save (default: scripts/weights/active/)
        type_name: Type name (default: type_0)
        graphs: Optional list of graph names that contributed to these weights
        
    Returns:
        Path to saved type file
    """
    global _type_registry
    
    if weights_dir is None:
        weights_dir = DEFAULT_WEIGHTS_DIR
    
    os.makedirs(weights_dir, exist_ok=True)
    
    # Save weights to type file
    type_file = os.path.join(weights_dir, f"{type_name}.json")
    with open(type_file, 'w') as f:
        json.dump(weights, f, indent=2)
    
    # Update type registry
    if not _type_registry:
        load_type_registry(weights_dir)
    
    # Get list of algorithms from weights
    algo_list = [k for k in weights.keys() if not k.startswith('_')]
    
    # Create or update registry entry
    if type_name not in _type_registry:
        _type_registry[type_name] = {
            'centroid': [0.5] * 7,  # Default centroid
            'sample_count': 1,
            'graph_count': len(graphs) if graphs else 1,
            'algorithms': algo_list,
            'graphs': graphs or [],
            'created': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
        }
    else:
        _type_registry[type_name]['algorithms'] = algo_list
        _type_registry[type_name]['last_updated'] = datetime.now().isoformat()
        if graphs:
            existing_graphs = _type_registry[type_name].get('graphs', [])
            for g in graphs:
                if g not in existing_graphs:
                    existing_graphs.append(g)
            _type_registry[type_name]['graphs'] = existing_graphs
            _type_registry[type_name]['graph_count'] = len(existing_graphs)
    
    save_type_registry(weights_dir)
    log.info(f"Saved weights to {type_file} ({len(algo_list)} algorithms)")
    
    return type_file


def _create_default_weight_entry() -> Dict:
    """Create a default weight entry for an algorithm."""
    return {
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
        'w_packing_factor': 0.0,
        'w_forward_edge_fraction': 0.0,
        'w_working_set_ratio': 0.0,
        'w_dv_x_hub': 0.0,
        'w_mod_x_logn': 0.0,
        'w_pf_x_wsr': 0.0,
        'w_fef_convergence': 0.0,
        'benchmark_weights': {'pr': 1.0, 'bfs': 1.0, 'cc': 1.0, 'sssp': 1.0, 'bc': 1.0, 'tc': 1.0},
        '_metadata': {
            'sample_count': 0,
            'avg_speedup': 1.0,
            'win_count': 0,
            'times_best': 0,
            'win_rate': 0.0,
        }
    }


def get_all_algorithm_variant_names() -> List[str]:
    """Get all algorithm names including variant-expanded names.
    
    Returns names for:
    - Base algorithms (ORIGINAL, RANDOM, SORT, HUBSORT, etc.)
    - RabbitOrder variants: RABBITORDER_csr, RABBITORDER_boost
    - GraphBrewOrder variants: GraphBrewOrder_leiden, GraphBrewOrder_rabbit, etc.
    """
    names = []
    
    # Algorithms that have variants (with their base IDs)
    VARIANT_ALGO_IDS = {8, 12}  # RabbitOrder, GraphBrewOrder
    
    for algo_id, algo_name in ALGORITHMS.items():
        if algo_name in ['MAP', 'AdaptiveOrder']:
            continue
        
        if algo_id == 8:  # RabbitOrder → expand to RABBITORDER_csr, RABBITORDER_boost
            for variant in RABBITORDER_VARIANTS:
                names.append(f"RABBITORDER_{variant}")
        elif algo_id == 12:  # GraphBrewOrder → expand to GraphBrewOrder_leiden, etc.
            for variant in GRAPHBREW_VARIANTS:
                names.append(f"GraphBrewOrder_{variant}")
        else:
            names.append(algo_name)
    
    return names


def initialize_default_weights(weights_dir: str = DEFAULT_WEIGHTS_DIR) -> Dict:
    """Initialize default weights for all algorithms including all variants.
    
    Creates entries for every variant of every algorithm so training can
    capture performance differences between variants (e.g., GraphBrewOrder_leiden
    vs GraphBrewOrder_rabbit vs GraphBrewOrder_hubcluster).
    """
    weights = {}
    
    for name in get_all_algorithm_variant_names():
        weights[name] = _create_default_weight_entry()
    
    return weights


def store_per_graph_results(
    benchmark_results: List = None,
    cache_results: List = None,
    reorder_results: List = None,
    graphs_dir: str = None,
    data_dir: str = None,
    run_timestamp: str = None,
    create_new_run: bool = True,
) -> str:
    """
    Store benchmark/reorder/cache results per-graph for later analysis.
    
    This enables:
    1. Re-analyzing data without re-running experiments
    2. Partial experiment runs (redo specific graphs)
    3. Multiple perceptron flavors comparison
    4. Historical tracking
    
    Args:
        benchmark_results: List of BenchmarkResult objects
        cache_results: List of CacheResult objects
        reorder_results: List of ReorderResult objects
        graphs_dir: Directory with graph files (for feature extraction)
        data_dir: Output directory for per-graph data (default: results/graphs/)
        run_timestamp: Explicit timestamp for this run (default: generate new)
        create_new_run: If True, create a new run instead of updating latest (default: True)
        
    Returns:
        The run timestamp used for storage
    """
    from .utils import get_timestamp
    
    benchmark_results = benchmark_results or []
    cache_results = cache_results or []
    reorder_results = reorder_results or []
    
    if data_dir is None:
        from .utils import RESULTS_DIR
        data_dir = os.path.join(RESULTS_DIR, "graphs")
    
    # Generate a shared timestamp for all graphs in this run
    if run_timestamp is None and create_new_run:
        run_timestamp = get_timestamp()
    
    # Import graph_data module
    from .graph_data import (
        GraphDataStore, GraphFeatures,
        AlgorithmBenchmarkData, AlgorithmReorderData,
    )
    
    # Load graph properties for features
    from .features import load_graph_properties_cache
    graph_props = load_graph_properties_cache(graphs_dir or "results")
    
    # Group results by graph
    graph_benchmarks = {}  # graph_name -> [BenchmarkResult, ...]
    graph_reorders = {}    # graph_name -> [ReorderResult, ...]
    graph_caches = {}      # graph_name -> [CacheResult, ...]
    
    for r in benchmark_results:
        graph_name = r.graph
        if graph_name not in graph_benchmarks:
            graph_benchmarks[graph_name] = []
        graph_benchmarks[graph_name].append(r)
    
    for r in reorder_results:
        graph_name = r.graph
        if graph_name not in graph_reorders:
            graph_reorders[graph_name] = []
        graph_reorders[graph_name].append(r)
    
    for r in cache_results:
        graph_name = r.graph if hasattr(r, 'graph') else getattr(r, 'graph_name', 'unknown')
        if graph_name not in graph_caches:
            graph_caches[graph_name] = []
        graph_caches[graph_name].append(r)
    
    # Get all unique graphs
    all_graphs = set(graph_benchmarks.keys()) | set(graph_reorders.keys()) | set(graph_caches.keys())
    
    log.info(f"Storing per-graph data for {len(all_graphs)} graphs")
    
    for graph_name in all_graphs:
        # Use shared timestamp for all graphs in this run
        store = GraphDataStore(graph_name, data_dir, run_timestamp=run_timestamp, 
                              create_new_run=create_new_run)
        
        # Store features from properties cache
        props = graph_props.get(graph_name, {})
        if props:
            features = GraphFeatures(
                graph_name=graph_name,
                nodes=props.get('nodes', 0),
                edges=props.get('edges', 0),
                avg_degree=props.get('avg_degree', 0.0),
                density=props.get('density', 0.0),
                modularity=props.get('modularity', 0.0),
                degree_variance=props.get('degree_variance', 0.0),
                hub_concentration=props.get('hub_concentration', 0.0),
                clustering_coefficient=props.get('clustering_coefficient', 0.0),
                avg_path_length=props.get('avg_path_length', 0.0),
                diameter_estimate=props.get('diameter', 0.0),
                community_count=props.get('community_count', 0),
                graph_type=props.get('graph_type', 'unknown'),
            )
            store.save_features(features)
        
        # Store benchmark results
        # Compute baseline time for speedup calculation
        benchmarks = graph_benchmarks.get(graph_name, [])
        baseline_times = {}  # benchmark -> baseline time (algo=0 or ORIGINAL)
        
        for r in benchmarks:
            # Handle both algorithm_name (graph_types.py) and algorithm (utils.py) attributes
            algo_name = getattr(r, 'algorithm_name', None) or getattr(r, 'algorithm', '')
            if r.algorithm_id == 0 or algo_name == 'ORIGINAL':
                bench = r.benchmark
                time_val = getattr(r, 'avg_time', 0) or getattr(r, 'time_seconds', 0)
                if time_val > 0:
                    baseline_times[bench] = time_val
        
        for r in benchmarks:
            time_val = getattr(r, 'avg_time', 0) or getattr(r, 'time_seconds', 0)
            trial_times = getattr(r, 'trial_times', [time_val]) if hasattr(r, 'trial_times') else [time_val]
            # Handle both algorithm_name (graph_types.py) and algorithm (utils.py) attributes
            algo_name = getattr(r, 'algorithm_name', None) or getattr(r, 'algorithm', '')
            
            # Compute speedup vs baseline
            baseline = baseline_times.get(r.benchmark, time_val)
            speedup = baseline / time_val if time_val > 0 else 1.0
            
            bench_data = AlgorithmBenchmarkData(
                graph_name=graph_name,
                algorithm_id=r.algorithm_id,
                algorithm_name=algo_name,
                benchmark=r.benchmark,
                avg_time=time_val,
                trial_times=trial_times,
                speedup=speedup,
                num_trials=len(trial_times),
                success=getattr(r, 'success', True),
                error=getattr(r, 'error', ''),
            )
            store.save_benchmark_result(bench_data)
        
        # Store reorder results
        for r in graph_reorders.get(graph_name, []):
            reorder_time = getattr(r, 'reorder_time', 0.0) or getattr(r, 'time_seconds', 0.0)
            # Handle both algorithm_name (graph_types.py) and algorithm (utils.py) attributes
            algo_name = getattr(r, 'algorithm_name', None) or getattr(r, 'algorithm', '')
            
            reorder_data = AlgorithmReorderData(
                graph_name=graph_name,
                algorithm_id=r.algorithm_id,
                algorithm_name=algo_name,
                reorder_time=reorder_time,
                modularity=getattr(r, 'modularity', 0.0),
                communities=getattr(r, 'communities', 0),
                isolated_vertices=getattr(r, 'isolated_vertices', 0),
                mapping_file=getattr(r, 'mapping_file', ''),
                success=getattr(r, 'success', True),
                error=getattr(r, 'error', ''),
            )
            store.save_reorder_result(reorder_data)
        
        # Store cache stats with benchmark results
        for r in graph_caches.get(graph_name, []):
            # Find matching benchmark and update cache stats
            algo_name = r.algorithm_name if hasattr(r, 'algorithm_name') else str(getattr(r, 'algorithm_id', 0))
            bench = getattr(r, 'benchmark', 'pr')
            
            # Load existing benchmark and update with cache stats
            existing = store.load_benchmark_result(bench, algo_name)
            if existing:
                existing.l1_hit_rate = 100.0 - getattr(r, 'l1_miss_rate', 0.0)
                existing.l2_hit_rate = 100.0 - getattr(r, 'l2_miss_rate', 0.0) 
                existing.l3_hit_rate = 100.0 - getattr(r, 'l3_miss_rate', 0.0)
                existing.llc_misses = getattr(r, 'llc_misses', 0)
                store.save_benchmark_result(existing)
    
    # Use the actual timestamp used (could be existing or new)
    used_timestamp = run_timestamp or store.run_timestamp if all_graphs else run_timestamp
    log.info(f"Stored per-graph data in {data_dir} (run: {used_timestamp})")
    
    return used_timestamp


def update_zero_weights(
    weights_file: str = None,
    benchmark_results: List = None,
    cache_results: List = None,
    reorder_results: List = None,
    graphs_dir: str = None,
    store_per_graph: bool = True,
    weights_dir: str = None,  # Alias for weights_file (for backward compatibility)
) -> None:
    """
    Update zero/default weights with actual benchmark data.
    
    This fills in missing weight values based on empirical benchmark results.
    Useful for bootstrapping the weight system with real performance data.
    
    Args:
        weights_file: Path to weights file to update
        benchmark_results: List of BenchmarkResult objects
        cache_results: Optional list of CacheResult objects
        reorder_results: Optional list of ReorderResult objects
        graphs_dir: Directory containing graphs (for feature extraction)
        store_per_graph: If True, also store results in per-graph directory structure
        weights_dir: Alias for weights_file (for backward compatibility)
    """
    # Handle alias
    if weights_dir and not weights_file:
        weights_file = weights_dir
        
    benchmark_results = benchmark_results or []
    cache_results = cache_results or []
    reorder_results = reorder_results or []
    
    # Store per-graph results for later analysis (before aggregation)
    if store_per_graph:
        try:
            store_per_graph_results(
                benchmark_results=benchmark_results,
                cache_results=cache_results,
                reorder_results=reorder_results,
                graphs_dir=graphs_dir
            )
        except Exception as e:
            log.warning(f"Failed to store per-graph results: {e}")
    
    # Load graph properties cache for features
    from .features import load_graph_properties_cache
    graph_props = load_graph_properties_cache(graphs_dir or "results")
    
    # Load existing weights or initialize
    weights = {}
    if weights_file and os.path.exists(weights_file):
        try:
            with open(weights_file) as f:
                weights = json.load(f)
        except:
            pass
    
    if not weights:
        weights = initialize_default_weights()
    
    # Collect all unique algorithm names from results (includes variants)
    all_algo_names = set()
    for r in benchmark_results:
        algo = getattr(r, 'algorithm', None) or getattr(r, 'algorithm_name', '')
        if algo:
            all_algo_names.add(algo)
    for r in reorder_results:
        algo = getattr(r, 'algorithm_name', None) or getattr(r, 'algorithm', '')
        if algo:
            all_algo_names.add(algo)
    
    # Add variant algorithms that aren't in weights yet
    for algo_name in all_algo_names:
        if algo_name not in weights and not algo_name.startswith('_'):
            weights[algo_name] = _create_default_weight_entry()
    
    # Update reorder time weights from reorder results
    reorder_times = {}
    for r in reorder_results:
        algo = r.algorithm_name
        if algo not in reorder_times:
            reorder_times[algo] = []
        reorder_time = getattr(r, 'reorder_time', 0.0) or getattr(r, 'time_seconds', 0.0)
        if reorder_time > 0:
            reorder_times[algo].append(reorder_time)
    
    for algo, times in reorder_times.items():
        if algo in weights and times:
            avg_time = sum(times) / len(times)
            # Penalize slow reordering algorithms
            weights[algo]['w_reorder_time'] = -avg_time / WEIGHT_REORDER_TIME_NORMALIZATION  # Normalize
            
            # Calibrate _metadata.avg_reorder_time from actual measurements
            # This is critical for MODE_BEST_AMORTIZATION which uses
            # avg_reorder_time to compute iterations-to-amortize
            meta = weights[algo].get('_metadata', {})
            meta['avg_reorder_time'] = avg_time
            weights[algo]['_metadata'] = meta
    
    # Update cache impact weights from cache results
    # Aggregate by algorithm, then average
    cache_stats = {}
    for r in cache_results:
        if not r.success:
            continue
        algo = r.algorithm_name if hasattr(r, 'algorithm_name') else str(r.algorithm_id)
        if algo not in cache_stats:
            cache_stats[algo] = {'l1': [], 'l2': [], 'l3': []}
        
        l1_miss = getattr(r, 'l1_miss_rate', 0.0)
        l2_miss = getattr(r, 'l2_miss_rate', 0.0)
        l3_miss = getattr(r, 'l3_miss_rate', 0.0)
        
        cache_stats[algo]['l1'].append(l1_miss)
        cache_stats[algo]['l2'].append(l2_miss)
        cache_stats[algo]['l3'].append(l3_miss)
    
    for algo, stats in cache_stats.items():
        if algo in weights:
            # Cache impact = (1 - miss_rate), so lower miss = higher impact
            if stats['l1']:
                avg_l1_miss = sum(stats['l1']) / len(stats['l1'])
                weights[algo]['cache_l1_impact'] = 1.0 - avg_l1_miss
            if stats['l2']:
                avg_l2_miss = sum(stats['l2']) / len(stats['l2'])
                weights[algo]['cache_l2_impact'] = 1.0 - avg_l2_miss
            if stats['l3']:
                avg_l3_miss = sum(stats['l3']) / len(stats['l3'])
                weights[algo]['cache_l3_impact'] = 1.0 - avg_l3_miss
            
            # DRAM penalty = average miss rate across all levels (normalized)
            all_misses = stats['l1'] + stats['l2'] + stats['l3']
            if all_misses:
                avg_miss = sum(all_misses) / len(all_misses)
                weights[algo]['cache_dram_penalty'] = -avg_miss * 0.5  # Penalty (negative weight)
    
    # Update feature weights from benchmark results
    # Group results by graph and benchmark to compute correlations
    if benchmark_results:
        import math
        
        # Collect speedups and features per algorithm
        algo_speedups = {}  # algo -> [(speedup, features), ...]
        
        for r in benchmark_results:
            if not r.success or r.time_seconds <= 0:
                continue
            
            algo = r.algorithm
            if algo not in algo_speedups:
                algo_speedups[algo] = []
            
            # Get features from graph properties cache first, then fallback to result
            graph_name = r.graph
            props = graph_props.get(graph_name, {})
            
            features = {
                'modularity': props.get('modularity', getattr(r, 'modularity', 0.5)),
                'degree_variance': props.get('degree_variance', getattr(r, 'degree_variance', 1.0)),
                'hub_concentration': props.get('hub_concentration', getattr(r, 'hub_concentration', 0.3)),
                'avg_degree': props.get('avg_degree', getattr(r, 'avg_degree', WEIGHT_AVG_DEGREE_DEFAULT)),
                'nodes': props.get('nodes', getattr(r, 'nodes', 1000)),
                'edges': props.get('edges', getattr(r, 'edges', 5000)),
                'clustering_coefficient': props.get('clustering_coefficient', 0.0),
                'avg_path_length': props.get('avg_path_length', 0.0),
                'diameter': props.get('diameter', 0.0),
                'community_count': props.get('community_count', 0.0),
            }
            
            # Compute derived features
            nodes = features['nodes']
            edges = features['edges']
            features['log_nodes'] = math.log10(nodes + 1) if nodes > 0 else 0
            features['log_edges'] = math.log10(edges + 1) if edges > 0 else 0
            features['density'] = 2 * edges / (nodes * (nodes - 1)) if nodes > 1 else 0
            
            # Estimate speedup by comparing to baseline
            algo_speedups[algo].append({
                'time': r.time_seconds,
                'graph': r.graph,
                'features': features
            })
        
        # Find baseline times per graph for speedup calculation
        baseline_times = {}
        for algo, results in algo_speedups.items():
            if 'ORIGINAL' in algo:
                for r in results:
                    baseline_times[r['graph']] = r['time']
        
        # Update feature weights using simple correlation-based heuristics
        for algo, results in algo_speedups.items():
            if algo not in weights:
                continue
            # ORIGINAL is now trained as a regular algorithm:
            # When ORIGINAL has speedup >= 1.0 (meaning no reorder beats it),
            # its weights are updated positively so it can win perceptron selection.
            
            # Collect feature arrays for correlation
            speedups = []
            feature_arrays = {
                'modularity': [], 'degree_variance': [], 'hub_concentration': [],
                'log_nodes': [], 'log_edges': [], 'density': [], 'avg_degree': [],
                'clustering_coefficient': [], 'avg_path_length': [], 
                'diameter': [], 'community_count': []
            }
            
            for r in results:
                graph = r['graph']
                baseline = baseline_times.get(graph, r['time'])
                if baseline > 0:
                    speedup = baseline / r['time']
                    speedups.append(speedup)
                    for feat_name in feature_arrays:
                        feature_arrays[feat_name].append(r['features'].get(feat_name, 0))
            
            if len(speedups) >= 2:
                # Simple correlation-based weight update
                avg_speedup = sum(speedups) / len(speedups)
                
                # Update bias based on average speedup (capped at 1.5)
                if avg_speedup > 1.0:
                    weights[algo]['bias'] = min(1.5, 0.5 + (avg_speedup - 1.0) * 0.5)
                
                # Update metadata to reflect the correlation-based analysis
                meta = weights[algo].get('_metadata', {})
                meta['sample_count'] = max(meta.get('sample_count', 0), len(speedups))
                meta['avg_speedup'] = avg_speedup
                weights[algo]['_metadata'] = meta
                
                # Update feature weights based on correlation direction
                # (positive correlation = feature helps this algo, negative = hurts)
                def correlation(x, y):
                    if len(x) < 2:
                        return 0.0
                    n = len(x)
                    mean_x = sum(x) / n
                    mean_y = sum(y) / n
                    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
                    den_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
                    den_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
                    if den_x * den_y > 0:
                        return num / (den_x * den_y)
                    return 0.0
                
                # Compute correlations and update all feature weights
                weight_map = {
                    'modularity': ('w_modularity', 0.5),
                    'degree_variance': ('w_degree_variance', 0.3),
                    'hub_concentration': ('w_hub_concentration', 0.3),
                    'log_nodes': ('w_log_nodes', 0.2),
                    'log_edges': ('w_log_edges', 0.2),
                    'density': ('w_density', 0.3),
                    'avg_degree': ('w_avg_degree', 0.2),
                    'clustering_coefficient': ('w_clustering_coeff', 0.3),
                    'avg_path_length': ('w_avg_path_length', 0.2),
                    'diameter': ('w_diameter', 0.2),
                    'community_count': ('w_community_count', 0.2),
                }
                
                for feat_name, (weight_name, scale) in weight_map.items():
                    feat_vals = feature_arrays[feat_name]
                    # Only compute if we have variance in the feature
                    if len(set(feat_vals)) > 1:
                        corr = correlation(feat_vals, speedups)
                        weights[algo][weight_name] = corr * scale
    
    # Save updated weights (DEPRECATED flat file - kept for backward compatibility)
    if weights_file:
        os.makedirs(os.path.dirname(weights_file), exist_ok=True)
        with open(weights_file, 'w') as f:
            json.dump(weights, f, indent=2)
    
    # Always save to type_0.json in active weights directory (used by C++)
    # Get list of unique graphs from results
    graph_names = list(set(r.graph for r in benchmark_results if hasattr(r, 'graph')))
    save_weights_to_active_type(weights, DEFAULT_WEIGHTS_DIR, type_name="type_0", graphs=graph_names)


# Alias for backward compatibility and phases.py import
generate_perceptron_weights_from_results = compute_weights_from_results


# =============================================================================
# Standalone CLI
# =============================================================================

def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="GraphBrew Weight Management Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m scripts.lib.weights --list-types
    python -m scripts.lib.weights --show-type type_0
    python -m scripts.lib.weights --best-algo type_0 --benchmark pr
    python -m scripts.lib.weights --init-type type_0
"""
    )
    
    parser.add_argument("--list-types", action="store_true",
                        help="List all known graph types")
    parser.add_argument("--show-type", help="Show summary for a type")
    parser.add_argument("--best-algo", help="Get best algorithm for a type")
    parser.add_argument("--benchmark", "-b", default="pr",
                        help="Benchmark for algorithm selection (default: pr)")
    parser.add_argument("--init-type", help="Initialize weights for a new type")
    parser.add_argument("--weights-dir", default=DEFAULT_WEIGHTS_DIR,
                        help=f"Weights directory (default: {DEFAULT_WEIGHTS_DIR})")
    
    args = parser.parse_args()
    
    if args.list_types:
        types = list_known_types(args.weights_dir)
        print(f"\nKnown Graph Types ({len(types)}):")
        print("-" * 40)
        for t in types:
            summary = get_type_summary(t, args.weights_dir)
            print(f"  {t}: {summary['num_graphs']} graphs, {len(summary['algorithms'])} algorithms")
        return
    
    if args.show_type:
        summary = get_type_summary(args.show_type, args.weights_dir)
        print(f"\nType: {summary['type_name']}")
        print(f"  Graphs: {summary['num_graphs']}")
        print(f"  Created: {summary['created']}")
        print(f"  Last updated: {summary['last_updated']}")
        print(f"  Representative features: {summary['representative_features']}")
        print(f"\n  Algorithms ({len(summary['algorithms'])}):")
        for algo, stats in sorted(summary['algorithms'].items(), key=lambda x: -x[1]['avg_speedup']):
            print(f"    {algo}: avg_speedup={stats['avg_speedup']:.3f}, wins={stats['win_count']}/{stats['sample_count']}")
        print(f"\n  Best by benchmark: {summary['best_algorithms']}")
        return
    
    if args.best_algo:
        features = {'nodes': 10000, 'edges': 50000, 'modularity': 0.5, 'degree_variance': 1.0}
        algo, score = get_best_algorithm_for_type(args.best_algo, features, args.benchmark, args.weights_dir)
        print(f"\nBest algorithm for {args.best_algo} ({args.benchmark}): {algo} (score={score:.4f})")
        return
    
    if args.init_type:
        weights = initialize_default_weights(args.weights_dir)
        save_type_weights(args.init_type, weights, args.weights_dir)
        print(f"Initialized weights for {args.init_type} with {len(weights)} algorithms")
        return
    
    parser.print_help()


if __name__ == "__main__":
    main()
