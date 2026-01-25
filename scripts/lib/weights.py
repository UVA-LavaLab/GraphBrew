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
    PROJECT_ROOT, WEIGHTS_DIR, ALGORITHMS,
    Logger, get_timestamp,
)

# Initialize logger
log = Logger()

# =============================================================================
# Constants
# =============================================================================

# Default weights directory
DEFAULT_WEIGHTS_DIR = str(WEIGHTS_DIR)

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
    benchmark_weights: Dict[str, float] = field(default_factory=lambda: {
        'pr': 1.0, 'bfs': 1.0, 'cc': 1.0, 'sssp': 1.0, 'bc': 1.0
    })
    _metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def compute_score(self, features: Dict, benchmark: str = 'pr') -> float:
        """Compute perceptron score for given features and benchmark."""
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
        score += self.w_diameter * features.get('diameter', features.get('diameter_estimate', 0.0)) / 50.0
        score += self.w_community_count * math.log10(features.get('community_count', 1) + 1)
        score += self.w_reorder_time * features.get('reorder_time', 0.0)
        
        # Apply benchmark-specific multiplier
        bench_mult = self.benchmark_weights.get(benchmark.lower(), 1.0)
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


# =============================================================================
# Type Assignment and Weight Updates
# =============================================================================

def assign_graph_type(
    features: Dict,
    weights_dir: str = DEFAULT_WEIGHTS_DIR,
    create_if_outlier: bool = True
) -> str:
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
    
    algo_weights['bias'] += learning_rate * error
    algo_weights['w_modularity'] += learning_rate * error * features.get('modularity', 0.5)
    algo_weights['w_log_nodes'] += learning_rate * error * log_nodes * 0.1
    algo_weights['w_log_edges'] += learning_rate * error * log_edges * 0.1
    algo_weights['w_density'] += learning_rate * error * features.get('density', 0.0)
    algo_weights['w_avg_degree'] += learning_rate * error * features.get('avg_degree', 0.0) / 100.0
    algo_weights['w_degree_variance'] += learning_rate * error * features.get('degree_variance', 1.0)
    algo_weights['w_hub_concentration'] += learning_rate * error * features.get('hub_concentration', 0.3)
    algo_weights['w_clustering_coeff'] += learning_rate * error * features.get('clustering_coefficient', 0.0)
    algo_weights['w_community_count'] += learning_rate * error * features.get('community_count', 0) / 1000.0
    algo_weights['w_avg_path_length'] += learning_rate * error * features.get('avg_path_length', 0.0) / 10.0
    algo_weights['w_diameter'] += learning_rate * error * features.get('diameter', 0.0) / 50.0
    
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
        algo_weights['w_reorder_time'] += learning_rate * error * (-reorder_time / 10.0)
        meta['avg_reorder_time'] = meta.get('avg_reorder_time', 0) + (reorder_time - meta.get('avg_reorder_time', 0)) / count
    
    # Update benchmark-specific weight
    bench_weights = algo_weights.get('benchmark_weights', {})
    if benchmark.lower() in bench_weights:
        current_bench_weight = bench_weights[benchmark.lower()]
        bench_weights[benchmark.lower()] = current_bench_weight + learning_rate * error * 0.1
    
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


def initialize_default_weights(weights_dir: str = DEFAULT_WEIGHTS_DIR) -> Dict:
    """Initialize default weights for all algorithms."""
    weights = {}
    
    for algo_id, algo_name in ALGORITHMS.items():
        if algo_name in ['MAP', 'AdaptiveOrder']:
            continue
        
        weights[algo_name] = {
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
    
    return weights


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
