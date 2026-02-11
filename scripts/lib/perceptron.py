#!/usr/bin/env python3
"""
Perceptron Experimentation Script for GraphBrew.

Allows experimenting with different perceptron configurations WITHOUT re-running
the expensive benchmark/reorder/cache phases. Uses existing results to:

1. Try different weight initializations (bias-based, performance-based, random)
2. Experiment with clustering (k-means with different k values)
3. Tweak feature weights and observe selection changes
4. Compare configurations against ground truth (best algorithm from benchmarks)
5. Export optimized weights to active directory for C++ to use

Usage:
    # Via entry point
    python3 scripts/graphbrew_experiment.py --perceptron

    # Or directly via module
    python3 -m scripts.lib.perceptron --show
    python3 -m scripts.lib.perceptron --grid-search
    python3 -m scripts.lib.perceptron --train --clusters 3 --method speedup
    python3 -m scripts.lib.perceptron --interactive
    python3 -m scripts.lib.perceptron --export --config best_config.json

Author: GraphBrew Team
"""

import os
import sys
import json
import math
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import copy

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.utils import ALGORITHMS, Logger, PROJECT_ROOT as PR, BENCHMARKS
from scripts.lib.weights import (
    save_weights_to_active_type, load_type_weights,
    PerceptronWeight, _create_default_weight_entry
)

log = Logger()

# =============================================================================
# Constants
# =============================================================================

RESULTS_DIR = PR / "results"
WEIGHTS_DIR = PR / "scripts" / "weights"
ACTIVE_DIR = WEIGHTS_DIR / "active"
EXPERIMENTS_DIR = WEIGHTS_DIR / "experiments"

# Features used for clustering
CLUSTER_FEATURES = [
    'modularity', 'log_nodes', 'log_edges', 'density', 
    'avg_degree', 'degree_variance', 'hub_concentration'
]

# BENCHMARKS imported from lib/utils.py (Single Source of Truth)

# Algorithm taxonomy - group by type for analysis
ALGORITHM_TAXONOMY = {
    'basic': ['ORIGINAL', 'RANDOM', 'SORT'],
    'hub': ['HUBSORT', 'HUBCLUSTER', 'DBG', 'HUBSORTDBG', 'HUBCLUSTERDBG'],
    'community': ['RABBITORDER', 'RABBITORDER_csr', 'RABBITORDER_boost', 'GORDER', 'CORDER', 'RCM'],
    'leiden': ['LeidenOrder',
               'GraphBrewOrder_leiden', 'GraphBrewOrder_rabbit',
               'GraphBrewOrder_hubcluster'],
    'composite': ['GraphBrewOrder', 'AdaptiveOrder'],
}

# Reverse mapping: algorithm -> category
ALGO_TO_CATEGORY = {}
for cat, algos in ALGORITHM_TAXONOMY.items():
    for algo in algos:
        ALGO_TO_CATEGORY[algo] = cat

# Graph type heuristics based on name patterns
GRAPH_TYPE_PATTERNS = {
    'social': ['soc-', 'facebook', 'twitter', 'friendster'],
    'web': ['web-', 'uk-', 'it-', 'arabic-', 'indochina-'],
    'road': ['road', 'Road', 'GAP-road'],
    'citation': ['cit-', 'ca-', 'hep'],
    'p2p': ['p2p-', 'gnutella'],
    'email': ['email-', 'enron'],
    'random': ['GAP-urand', 'GAP-kron', 'random', 'er-'],
}

def get_graph_type(graph_name: str) -> str:
    """Infer graph type from name."""
    name_lower = graph_name.lower()
    for gtype, patterns in GRAPH_TYPE_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in name_lower:
                return gtype
    return 'unknown'

def get_algo_category(algo_name: str) -> str:
    """Get algorithm category."""
    return ALGO_TO_CATEGORY.get(algo_name, 'unknown')


# =============================================================================
# Data Loading
# =============================================================================

def load_all_results() -> Dict[str, Any]:
    """Load all benchmark, reorder, and cache results from results/ directory."""
    results = {
        'benchmarks': [],
        'reorder_times': [],
        'cache': [],
        'graphs': set(),
        'algorithms': set(),
    }
    
    # Find most recent result files
    result_files = list(RESULTS_DIR.glob("*.json"))
    
    # Group by type and get latest
    benchmark_files = sorted([f for f in result_files if f.name.startswith("benchmark_")])
    reorder_files = sorted([f for f in result_files if f.name.startswith("reorder_times_")])
    cache_files = sorted([f for f in result_files if f.name.startswith("cache_")])
    
    # Load all benchmark results (merge all files)
    for bf in benchmark_files:
        try:
            with open(bf) as f:
                data = json.load(f)
                if isinstance(data, list):
                    results['benchmarks'].extend(data)
                    for item in data:
                        results['graphs'].add(item.get('graph', ''))
                        results['algorithms'].add(item.get('algorithm', ''))
        except Exception as e:
            log.warning(f"Failed to load {bf}: {e}")
    
    # Load all reorder times
    for rf in reorder_files:
        try:
            with open(rf) as f:
                data = json.load(f)
                if isinstance(data, list):
                    results['reorder_times'].extend(data)
        except Exception as e:
            log.warning(f"Failed to load {rf}: {e}")
    
    # Load all cache results
    for cf in cache_files:
        try:
            with open(cf) as f:
                data = json.load(f)
                if isinstance(data, list):
                    results['cache'].extend(data)
        except Exception as e:
            log.warning(f"Failed to load {cf}: {e}")
    
    results['graphs'] = sorted(results['graphs'])
    results['algorithms'] = sorted(results['algorithms'])
    
    # Track which phases are available
    results['has_benchmarks'] = len(results['benchmarks']) > 0
    results['has_reorder'] = len(results['reorder_times']) > 0
    results['has_cache'] = len(results['cache']) > 0
    
    log.info(f"Loaded {len(results['benchmarks'])} benchmark results")
    log.info(f"Loaded {len(results['reorder_times'])} reorder times")
    if results['has_cache']:
        log.info(f"Loaded {len(results['cache'])} cache results")
    else:
        log.warning("No cache results (--skip-cache was used)")
    
    # Categorize graphs and algorithms
    results['graph_types'] = {g: get_graph_type(g) for g in results['graphs']}
    results['algo_categories'] = {a: get_algo_category(a) for a in results['algorithms']}
    log.info(f"Graphs: {len(results['graphs'])}, Algorithms: {len(results['algorithms'])}")
    
    return results


def compute_graph_features(graph_name: str, results: Dict) -> Dict[str, float]:
    """
    Compute/estimate graph features from available data.
    
    Merges features from per-graph features.json AND graph_properties_cache.json,
    preferring non-zero values over zero/default ones so that partial data from
    one source doesn't overwrite richer data from another.
    """
    features = {
        'modularity': 0.5,
        'nodes': 1000,
        'edges': 5000,
        'density': 0.01,
        'avg_degree': 10.0,
        'degree_variance': 1.0,
        'hub_concentration': 0.3,
        'clustering_coefficient': 0.1,
        'community_count': 10,
    }

    # Metadata keys that should always be overwritten (not numeric features)
    _metadata_keys = {'graph_name', 'graph_type', 'source_file', 'last_updated'}

    def _smart_merge(dst: dict, src: dict) -> None:
        """Merge src into dst, but never overwrite a non-zero numeric with zero."""
        for k, v in src.items():
            if k in _metadata_keys:
                dst[k] = v
                continue
            if isinstance(v, (int, float)):
                # Only overwrite if new value is non-zero or destination is still default
                existing = dst.get(k)
                if v != 0 or existing is None:
                    dst[k] = v
            else:
                dst[k] = v

    # Load from per-graph features.json first (may have 0.0 for some fields)
    feat_file = RESULTS_DIR / "graphs" / graph_name / "features.json"
    if feat_file.exists():
        try:
            with open(feat_file) as f:
                _smart_merge(features, json.load(f))
        except Exception:
            pass

    # Load from graph_properties_cache.json last — it typically has the richest
    # data including modularity, density, etc. computed from full graph analysis.
    cache_file = RESULTS_DIR / "graph_properties_cache.json"
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                cache = json.load(f)
                if graph_name in cache:
                    _smart_merge(features, cache[graph_name])
        except Exception:
            pass
    
    # Compute derived features
    features['log_nodes'] = math.log10(features.get('nodes', 1000) + 1)
    features['log_edges'] = math.log10(features.get('edges', 5000) + 1)
    
    return features


def build_performance_matrix(results: Dict) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Build a performance matrix: graph -> algorithm -> benchmark -> time.
    
    Returns:
        {graph: {algorithm: {benchmark: time_seconds}}}
    """
    matrix = defaultdict(lambda: defaultdict(dict))
    
    for bench in results['benchmarks']:
        graph = bench.get('graph', '')
        algo = bench.get('algorithm', '')
        benchmark = bench.get('benchmark', '')
        time = bench.get('time_seconds', float('inf'))
        
        if graph and algo and benchmark and bench.get('success', False):
            # Keep best time if multiple runs
            existing = matrix[graph][algo].get(benchmark, float('inf'))
            matrix[graph][algo][benchmark] = min(existing, time)
    
    return dict(matrix)


def find_best_algorithm(perf_matrix: Dict, graph: str, benchmark: str) -> Tuple[str, float]:
    """Find the best algorithm for a graph/benchmark combination."""
    if graph not in perf_matrix:
        return 'ORIGINAL', float('inf')
    
    best_algo = 'ORIGINAL'
    best_time = float('inf')
    
    for algo, benchmarks in perf_matrix[graph].items():
        time = benchmarks.get(benchmark, float('inf'))
        if time < best_time:
            best_time = time
            best_algo = algo
    
    return best_algo, best_time


# =============================================================================
# Weight Training Methods
# =============================================================================

@dataclass
class PerceptronConfig:
    """Configuration for perceptron training."""
    method: str = 'speedup'  # 'speedup', 'winrate', 'rank', 'hybrid'
    n_clusters: int = 1
    feature_weights: Dict[str, float] = field(default_factory=dict)
    bias_scale: float = 1.0
    normalize_features: bool = True
    benchmark_weights: Dict[str, float] = field(default_factory=lambda: {
        'pr': 1.0, 'bfs': 1.0, 'cc': 1.0, 'sssp': 0.8, 'bc': 0.5, 'tc': 1.0
    })


def train_weights_speedup(
    perf_matrix: Dict,
    graphs: List[str],
    config: PerceptronConfig,
) -> Dict[str, Dict]:
    """
    Train weights based on speedup over ORIGINAL baseline.
    
    bias = average_speedup_across_benchmarks
    """
    weights = {}
    algorithms = set()
    
    for graph in graphs:
        if graph in perf_matrix:
            algorithms.update(perf_matrix[graph].keys())
    
    for algo in algorithms:
        speedups = []
        
        for graph in graphs:
            if graph not in perf_matrix or algo not in perf_matrix[graph]:
                continue
            
            for bench, time in perf_matrix[graph][algo].items():
                # Get ORIGINAL baseline
                baseline = perf_matrix[graph].get('ORIGINAL', {}).get(bench, time)
                if baseline > 0 and time > 0:
                    speedup = baseline / time
                    bench_weight = config.benchmark_weights.get(bench, 1.0)
                    speedups.append(speedup * bench_weight)
        
        if speedups:
            avg_speedup = sum(speedups) / len(speedups)
            bias = avg_speedup * config.bias_scale
        else:
            bias = 0.5
        
        weights[algo] = _create_default_weight_entry()
        weights[algo]['bias'] = round(bias, 4)
        weights[algo]['_metadata'] = {
            'method': 'speedup',
            'sample_count': len(speedups),
            'avg_speedup': round(sum(speedups)/len(speedups), 4) if speedups else 0,
        }
    
    return weights


def train_weights_winrate(
    perf_matrix: Dict,
    graphs: List[str],
    config: PerceptronConfig,
) -> Dict[str, Dict]:
    """
    Train weights based on win rate (how often algorithm is best).
    
    bias = win_rate * scale
    """
    weights = {}
    win_counts = defaultdict(int)
    total_counts = defaultdict(int)
    algorithms = set()
    
    for graph in graphs:
        if graph not in perf_matrix:
            continue
        algorithms.update(perf_matrix[graph].keys())
        
        for bench in BENCHMARKS:
            best_algo, _ = find_best_algorithm(perf_matrix, graph, bench)
            if best_algo:
                win_counts[best_algo] += 1
            for algo in perf_matrix[graph].keys():
                if bench in perf_matrix[graph][algo]:
                    total_counts[algo] += 1
    
    for algo in algorithms:
        if total_counts[algo] > 0:
            win_rate = win_counts[algo] / total_counts[algo]
            bias = win_rate * config.bias_scale * 10  # Scale up for differentiation
        else:
            bias = 0.5
        
        weights[algo] = _create_default_weight_entry()
        weights[algo]['bias'] = round(bias, 4)
        weights[algo]['_metadata'] = {
            'method': 'winrate',
            'wins': win_counts[algo],
            'total': total_counts[algo],
            'win_rate': round(win_rate, 4) if total_counts[algo] > 0 else 0,
        }
    
    return weights


def train_weights_rank(
    perf_matrix: Dict,
    graphs: List[str],
    config: PerceptronConfig,
) -> Dict[str, Dict]:
    """
    Train weights based on average rank across benchmarks.
    
    bias = (n_algorithms - avg_rank) / n_algorithms * scale
    """
    weights = {}
    rank_sums = defaultdict(float)
    rank_counts = defaultdict(int)
    algorithms = set()
    
    for graph in graphs:
        if graph not in perf_matrix:
            continue
        algorithms.update(perf_matrix[graph].keys())
        
        for bench in BENCHMARKS:
            # Get all times for this graph/benchmark
            times = []
            for algo, benchmarks in perf_matrix[graph].items():
                if bench in benchmarks:
                    times.append((algo, benchmarks[bench]))
            
            if not times:
                continue
            
            # Sort by time (ascending = better)
            times.sort(key=lambda x: x[1])
            
            # Assign ranks
            for rank, (algo, _) in enumerate(times, 1):
                rank_sums[algo] += rank
                rank_counts[algo] += 1
    
    n_algorithms = len(algorithms) if algorithms else 1
    
    for algo in algorithms:
        if rank_counts[algo] > 0:
            avg_rank = rank_sums[algo] / rank_counts[algo]
            # Higher bias for lower (better) rank
            bias = (n_algorithms - avg_rank + 1) / n_algorithms * config.bias_scale * 5
        else:
            bias = 0.5
        
        weights[algo] = _create_default_weight_entry()
        weights[algo]['bias'] = round(max(0.1, bias), 4)
        weights[algo]['_metadata'] = {
            'method': 'rank',
            'avg_rank': round(rank_sums[algo]/rank_counts[algo], 2) if rank_counts[algo] > 0 else 0,
            'n_samples': rank_counts[algo],
        }
    
    return weights


def train_weights_per_benchmark(
    perf_matrix: Dict,
    graphs: List[str],
    config: PerceptronConfig,
) -> Dict[str, Dict]:
    """
    Train weights with per-benchmark multipliers.
    
    Each algorithm gets benchmark-specific weights that affect selection
    when a specific benchmark is being run.
    """
    weights = {}
    algorithms = set()
    
    for graph in graphs:
        if graph in perf_matrix:
            algorithms.update(perf_matrix[graph].keys())
    
    # For each algorithm, compute performance per benchmark
    for algo in algorithms:
        bench_scores = {}
        
        for bench in BENCHMARKS:
            speedups = []
            for graph in graphs:
                if graph not in perf_matrix or algo not in perf_matrix[graph]:
                    continue
                if bench not in perf_matrix[graph][algo]:
                    continue
                
                time = perf_matrix[graph][algo][bench]
                baseline = perf_matrix[graph].get('ORIGINAL', {}).get(bench, time)
                
                if baseline > 0 and time > 0:
                    speedups.append(baseline / time)
            
            if speedups:
                bench_scores[bench] = sum(speedups) / len(speedups)
            else:
                bench_scores[bench] = 1.0
        
        # Compute base bias as average speedup
        if bench_scores:
            avg_speedup = sum(bench_scores.values()) / len(bench_scores)
        else:
            avg_speedup = 0.5
        
        weights[algo] = _create_default_weight_entry()
        weights[algo]['bias'] = round(avg_speedup * config.bias_scale, 4)
        
        # Add benchmark-specific multipliers
        # If algorithm is better than average for a benchmark, boost it
        weights[algo]['benchmark_weights'] = {}
        for bench, score in bench_scores.items():
            # Relative to this algorithm's average
            if avg_speedup > 0:
                multiplier = score / avg_speedup
            else:
                multiplier = 1.0
            weights[algo]['benchmark_weights'][bench] = round(multiplier, 4)
        
        weights[algo]['_metadata'] = {
            'method': 'per_benchmark',
            'bench_scores': bench_scores,
            'avg_speedup': avg_speedup,
        }
    
    return weights


def train_weights_hybrid(
    perf_matrix: Dict,
    graphs: List[str],
    config: PerceptronConfig,
) -> Dict[str, Dict]:
    """
    Hybrid method: combine speedup, winrate, and rank.
    
    bias = 0.4 * speedup_score + 0.4 * winrate_score + 0.2 * rank_score
    """
    speedup_weights = train_weights_speedup(perf_matrix, graphs, config)
    winrate_weights = train_weights_winrate(perf_matrix, graphs, config)
    rank_weights = train_weights_rank(perf_matrix, graphs, config)
    
    weights = {}
    all_algos = set(speedup_weights.keys()) | set(winrate_weights.keys()) | set(rank_weights.keys())
    
    for algo in all_algos:
        s_bias = speedup_weights.get(algo, {}).get('bias', 0.5)
        w_bias = winrate_weights.get(algo, {}).get('bias', 0.5)
        r_bias = rank_weights.get(algo, {}).get('bias', 0.5)
        
        # Normalize to similar scales before combining
        combined = 0.4 * s_bias + 0.4 * w_bias + 0.2 * r_bias
        
        weights[algo] = _create_default_weight_entry()
        weights[algo]['bias'] = round(combined, 4)
        weights[algo]['_metadata'] = {
            'method': 'hybrid',
            'speedup_bias': s_bias,
            'winrate_bias': w_bias,
            'rank_bias': r_bias,
        }
    
    return weights


def train_weights_perceptron(
    perf_matrix: Dict,
    graphs: List[str],
    config: PerceptronConfig,
    results: Dict,
    learning_rate: float = 0.1,
    epochs: int = 50,
) -> Dict[str, Dict]:
    """
    Train weights using an online perceptron update rule.

    For each graph/benchmark, the algorithm with the highest score should be the
    actual best algorithm.  When the predicted best differs from the actual best,
    we push the actual-best score up and the predicted-best score down by
    adjusting bias and feature weights proportionally.

    This is the only training method that produces non-zero feature weights,
    enabling the perceptron to specialize its selections per-graph.
    """
    # Collect all algorithms
    algorithms = set()
    for graph in graphs:
        if graph in perf_matrix:
            algorithms.update(perf_matrix[graph].keys())
    algorithms = sorted(algorithms)

    # Initialize weights from rank-based bias (good starting point)
    weights = train_weights_rank(perf_matrix, graphs, config)
    # Ensure all algorithms have entries
    for algo in algorithms:
        if algo not in weights:
            weights[algo] = _create_default_weight_entry()

    # Compute features once
    graph_features = {g: compute_graph_features(g, results) for g in graphs}

    # Feature names and their weight keys (same mapping as add_feature_weights)
    feat_map = {
        'modularity': 'w_modularity',
        'density': 'w_density',
        'degree_variance': 'w_degree_variance',
        'hub_concentration': 'w_hub_concentration',
        'clustering_coefficient': 'w_clustering_coeff',
        'avg_degree': 'w_avg_degree',
        'packing_factor': 'w_packing_factor',
        'forward_edge_fraction': 'w_forward_edge_fraction',
        'working_set_ratio': 'w_working_set_ratio',
        'community_count': 'w_community_count',
    }

    # Identify benchmarks that have data
    active_benchmarks = set()
    for g in graphs:
        for algo in perf_matrix.get(g, {}):
            for b in perf_matrix[g][algo]:
                active_benchmarks.add(b)
    active_benchmarks = sorted(active_benchmarks)

    updates = 0

    for epoch in range(epochs):
        # Shuffle training order for stability
        order = list(graphs)
        random.shuffle(order)
        epoch_updates = 0

        for graph in order:
            if graph not in perf_matrix:
                continue
            features = graph_features.get(graph, {})

            for bench in active_benchmarks:
                # Find actual best algorithm for this graph/benchmark
                actual_best, actual_time = find_best_algorithm(perf_matrix, graph, bench)
                if actual_time == float('inf') or actual_best not in weights:
                    continue

                # Find predicted best (highest score)
                best_score = float('-inf')
                predicted_best = None
                for algo, w in weights.items():
                    if algo.startswith('_') or not isinstance(w, dict):
                        continue
                    if algo not in perf_matrix.get(graph, {}):
                        continue
                    if bench not in perf_matrix[graph][algo]:
                        continue
                    pw = PerceptronWeight.from_dict(w)
                    score = pw.compute_score(features, bench)
                    if score > best_score:
                        best_score = score
                        predicted_best = algo

                if predicted_best is None or predicted_best == actual_best:
                    continue  # Correct prediction, no update needed

                # Perceptron update: push actual_best up, predicted_best down
                epoch_updates += 1

                # Update bias
                weights[actual_best]['bias'] += learning_rate
                weights[predicted_best]['bias'] -= learning_rate

                # Update feature weights
                for feat_name, w_key in feat_map.items():
                    fval = features.get(feat_name, 0.0)
                    if fval == 0.0:
                        continue
                    # Normalize large features
                    if feat_name == 'avg_degree':
                        fval /= 100.0
                    elif feat_name == 'community_count':
                        fval = math.log10(fval + 1)
                    elif feat_name == 'working_set_ratio':
                        fval = math.log2(fval + 1.0)

                    if w_key in weights[actual_best]:
                        weights[actual_best][w_key] += learning_rate * fval
                    if w_key in weights[predicted_best]:
                        weights[predicted_best][w_key] -= learning_rate * fval

                # Also update benchmark_weights for the specific benchmark
                bw_key = bench.lower()
                if bw_key in weights[actual_best].get('benchmark_weights', {}):
                    weights[actual_best]['benchmark_weights'][bw_key] += learning_rate * 0.5
                if bw_key in weights[predicted_best].get('benchmark_weights', {}):
                    weights[predicted_best]['benchmark_weights'][bw_key] = max(
                        0.1, weights[predicted_best]['benchmark_weights'][bw_key] - learning_rate * 0.5
                    )

        updates += epoch_updates
        if epoch_updates == 0:
            break  # Converged

    # Round all weights for clean JSON output
    for algo in weights:
        if algo.startswith('_') or not isinstance(weights[algo], dict):
            continue
        for k, v in weights[algo].items():
            if isinstance(v, float):
                weights[algo][k] = round(v, 4)
        weights[algo]['_metadata'] = {
            'method': 'perceptron',
            'epochs': epoch + 1,
            'total_updates': updates,
            'learning_rate': learning_rate,
        }

    return weights


def add_feature_weights(
    weights: Dict[str, Dict],
    perf_matrix: Dict,
    graphs: List[str],
    results: Dict,
) -> Dict[str, Dict]:
    """
    Add feature-based weights by correlating features with performance.
    
    For each algorithm, compute correlation between each feature and speedup.
    """
    # Compute features for all graphs
    graph_features = {g: compute_graph_features(g, results) for g in graphs}
    
    for algo in weights.keys():
        if algo.startswith('_'):
            continue
        
        # Collect (feature_value, speedup) pairs
        feature_speedups = defaultdict(list)
        
        for graph in graphs:
            if graph not in perf_matrix or algo not in perf_matrix[graph]:
                continue
            
            features = graph_features.get(graph, {})
            
            for bench in BENCHMARKS:
                if bench not in perf_matrix[graph][algo]:
                    continue
                
                time = perf_matrix[graph][algo][bench]
                baseline = perf_matrix[graph].get('ORIGINAL', {}).get(bench, time)
                
                if baseline > 0 and time > 0:
                    speedup = baseline / time
                    
                    # Map feature names to their weight-key names in the dict.
                    # Most follow the pattern w_{feature}, but clustering_coefficient
                    # maps to w_clustering_coeff.
                    _FEAT_TO_WEIGHT = {
                        'modularity': 'w_modularity',
                        'density': 'w_density',
                        'degree_variance': 'w_degree_variance',
                        'hub_concentration': 'w_hub_concentration',
                        'clustering_coefficient': 'w_clustering_coeff',
                        'avg_degree': 'w_avg_degree',
                        'packing_factor': 'w_packing_factor',
                        'forward_edge_fraction': 'w_forward_edge_fraction',
                        'working_set_ratio': 'w_working_set_ratio',
                        'community_count': 'w_community_count',
                    }
                    for feat_name in _FEAT_TO_WEIGHT:
                        if feat_name in features:
                            feature_speedups[feat_name].append((features[feat_name], speedup))
        
        # Compute simple correlation (positive/negative tendency)
        for feat_name, pairs in feature_speedups.items():
            if len(pairs) < 3:
                continue
            
            # Simple: if high feature correlates with high speedup, positive weight
            avg_feat = sum(p[0] for p in pairs) / len(pairs)
            high_feat_speedups = [p[1] for p in pairs if p[0] > avg_feat]
            low_feat_speedups = [p[1] for p in pairs if p[0] <= avg_feat]
            
            if high_feat_speedups and low_feat_speedups:
                high_avg = sum(high_feat_speedups) / len(high_feat_speedups)
                low_avg = sum(low_feat_speedups) / len(low_feat_speedups)
                
                # Weight based on difference
                diff = (high_avg - low_avg) / max(high_avg, low_avg, 0.01)
                weight_name = _FEAT_TO_WEIGHT.get(feat_name, f'w_{feat_name}')
                
                if weight_name in weights[algo]:
                    weights[algo][weight_name] = round(diff * 0.5, 4)
    
    return weights


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_weights(
    weights: Dict[str, Dict],
    perf_matrix: Dict,
    graphs: List[str],
    results: Dict,
    benchmark: str = 'pr',
) -> Dict[str, Any]:
    """
    Evaluate perceptron weights against ground truth.
    
    Returns accuracy, confusion matrix, and per-graph details.
    """
    correct = 0
    total = 0
    details = []
    confusion = defaultdict(lambda: defaultdict(int))
    
    graph_features = {g: compute_graph_features(g, results) for g in graphs}
    
    for graph in graphs:
        if graph not in perf_matrix:
            continue
        
        features = graph_features.get(graph, {})
        
        # Find actual best algorithm
        actual_best, actual_time = find_best_algorithm(perf_matrix, graph, benchmark)
        
        # Find predicted best algorithm using perceptron
        # Only consider algorithms that have benchmark data for this graph/benchmark
        best_score = float('-inf')
        predicted_best = 'ORIGINAL'
        
        graph_algos = perf_matrix.get(graph, {})
        
        for algo, w in weights.items():
            if algo.startswith('_') or not isinstance(w, dict):
                continue
            # Only consider algorithms with actual benchmark data for this graph
            if algo not in graph_algos or benchmark not in graph_algos[algo]:
                continue
            
            # Use canonical PerceptronWeight.compute_score() with all 24 features
            pw = PerceptronWeight.from_dict(w)
            score = pw.compute_score(features, benchmark)
            
            if score > best_score:
                best_score = score
                predicted_best = algo
        
        is_correct = (predicted_best == actual_best)
        if is_correct:
            correct += 1
        total += 1
        
        confusion[actual_best][predicted_best] += 1
        
        # Compute regret (how much slower than optimal)
        predicted_time = perf_matrix[graph].get(predicted_best, {}).get(benchmark, float('inf'))
        regret = (predicted_time / actual_time - 1.0) * 100 if actual_time > 0 else 0
        
        details.append({
            'graph': graph,
            'actual_best': actual_best,
            'actual_time': actual_time,
            'predicted_best': predicted_best,
            'predicted_time': predicted_time,
            'correct': is_correct,
            'regret_pct': round(regret, 2),
        })
    
    accuracy = correct / total * 100 if total > 0 else 0
    avg_regret = sum(d['regret_pct'] for d in details) / len(details) if details else 0
    
    # Tolerance-based accuracy: count predictions within X% of optimal as correct
    within_5 = sum(1 for d in details if d['regret_pct'] <= 5.0)
    within_10 = sum(1 for d in details if d['regret_pct'] <= 10.0)
    within_20 = sum(1 for d in details if d['regret_pct'] <= 20.0)
    
    return {
        'accuracy': round(accuracy, 2),
        'accuracy_5pct': round(within_5 / total * 100, 2) if total > 0 else 0,
        'accuracy_10pct': round(within_10 / total * 100, 2) if total > 0 else 0,
        'accuracy_20pct': round(within_20 / total * 100, 2) if total > 0 else 0,
        'correct': correct,
        'total': total,
        'avg_regret_pct': round(avg_regret, 2),
        'confusion': dict(confusion),
        'details': details,
        'benchmark': benchmark,
    }


def evaluate_all_benchmarks(
    weights: Dict[str, Dict],
    perf_matrix: Dict,
    graphs: List[str],
    results: Dict,
) -> Dict[str, Any]:
    """Evaluate weights across all benchmarks."""
    all_results = {}
    
    for bench in BENCHMARKS:
        # Check if we have data for this benchmark
        has_data = any(
            bench in perf_matrix.get(g, {}).get(algo, {})
            for g in graphs
            for algo in perf_matrix.get(g, {}).keys()
        )
        
        if has_data:
            all_results[bench] = evaluate_weights(weights, perf_matrix, graphs, results, bench)
    
    # Compute average
    if all_results:
        avg_accuracy = sum(r['accuracy'] for r in all_results.values()) / len(all_results)
        avg_acc_5 = sum(r.get('accuracy_5pct', 0) for r in all_results.values()) / len(all_results)
        avg_acc_10 = sum(r.get('accuracy_10pct', 0) for r in all_results.values()) / len(all_results)
        avg_regret = sum(r['avg_regret_pct'] for r in all_results.values()) / len(all_results)
    else:
        avg_accuracy = avg_acc_5 = avg_acc_10 = 0
        avg_regret = 0
    
    return {
        'per_benchmark': all_results,
        'avg_accuracy': round(avg_accuracy, 2),
        'avg_accuracy_5pct': round(avg_acc_5, 2),
        'avg_accuracy_10pct': round(avg_acc_10, 2),
        'avg_regret_pct': round(avg_regret, 2),
    }


# =============================================================================
# Grid Search
# =============================================================================

def grid_search(
    perf_matrix: Dict,
    graphs: List[str],
    results: Dict,
) -> List[Dict]:
    """
    Run grid search over different configurations.
    
    Tests:
    - Training methods: speedup, winrate, rank, hybrid, perceptron
    - Bias scales: 0.5, 1.0, 2.0, 5.0
    - Learning rates (perceptron only): 0.01, 0.05, 0.1, 0.5
    - With/without feature weights
    """
    configs = []
    
    # Classic methods with bias-only training
    for method in ['speedup', 'winrate', 'rank', 'hybrid']:
        for scale in [0.5, 1.0, 2.0, 5.0]:
            for use_features in [False, True]:
                configs.append({
                    'method': method,
                    'scale': scale,
                    'use_features': use_features,
                    'learning_rate': 0.0,
                    'config': PerceptronConfig(method=method, bias_scale=scale),
                })
    
    # Perceptron method with online learning (feature weights trained)
    for lr in [0.01, 0.05, 0.1, 0.5]:
        for scale in [0.5, 1.0, 2.0]:
            configs.append({
                'method': 'perceptron',
                'scale': scale,
                'use_features': True,  # always uses features
                'learning_rate': lr,
                'config': PerceptronConfig(method='perceptron', bias_scale=scale),
            })
    
    results_list = []
    
    log.info(f"Running grid search with {len(configs)} configurations...")
    
    for i, cfg in enumerate(configs):
        # Train weights
        method = cfg['method']
        if method == 'speedup':
            weights = train_weights_speedup(perf_matrix, graphs, cfg['config'])
        elif method == 'winrate':
            weights = train_weights_winrate(perf_matrix, graphs, cfg['config'])
        elif method == 'rank':
            weights = train_weights_rank(perf_matrix, graphs, cfg['config'])
        elif method == 'perceptron':
            weights = train_weights_perceptron(
                perf_matrix, graphs, cfg['config'], results,
                learning_rate=cfg['learning_rate'],
            )
        else:
            weights = train_weights_hybrid(perf_matrix, graphs, cfg['config'])
        
        if cfg['use_features'] and method != 'perceptron':
            weights = add_feature_weights(weights, perf_matrix, graphs, results)
        
        # Evaluate
        eval_result = evaluate_all_benchmarks(weights, perf_matrix, graphs, results)
        
        config_desc = {'method': method, 'scale': cfg['scale'], 'use_features': cfg['use_features']}
        if method == 'perceptron':
            config_desc['learning_rate'] = cfg['learning_rate']
        
        results_list.append({
            'config': config_desc,
            'accuracy': eval_result['avg_accuracy'],
            'accuracy_5pct': eval_result.get('avg_accuracy_5pct', 0),
            'accuracy_10pct': eval_result.get('avg_accuracy_10pct', 0),
            'regret': eval_result['avg_regret_pct'],
            'per_benchmark': {k: v['accuracy'] for k, v in eval_result['per_benchmark'].items()},
            'weights': weights,
        })
        
        extra = f", lr={cfg['learning_rate']}" if method == 'perceptron' else ""
        log.info(f"  [{i+1}/{len(configs)}] {method}, scale={cfg['scale']}, features={cfg['use_features']}{extra}: "
                f"acc={eval_result['avg_accuracy']:.1f}% (≤5%:{eval_result.get('avg_accuracy_5pct',0):.0f}%, "
                f"≤10%:{eval_result.get('avg_accuracy_10pct',0):.0f}%), "
                f"regret={eval_result['avg_regret_pct']:.1f}%")
    
    # Sort by accuracy (descending), then regret (ascending)
    results_list.sort(key=lambda x: (-x['accuracy'], x['regret']))
    
    return results_list


# =============================================================================
# Clustering
# =============================================================================

def cluster_graphs(
    graphs: List[str],
    results: Dict,
    n_clusters: int = 2,
) -> Dict[int, List[str]]:
    """
    Cluster graphs by feature similarity using simple k-means.
    """
    if n_clusters <= 1:
        return {0: graphs}
    
    # Compute features for all graphs
    features_list = []
    for g in graphs:
        f = compute_graph_features(g, results)
        # Normalize features
        vec = [
            f.get('modularity', 0.5),
            f.get('log_nodes', 3.0) / 6.0,  # Normalize to ~0-1
            f.get('log_edges', 4.0) / 8.0,
            f.get('density', 0.01) * 100,
            f.get('avg_degree', 10.0) / 100.0,
            f.get('degree_variance', 1.0) / 10.0,
            f.get('hub_concentration', 0.3),
        ]
        features_list.append(vec)
    
    # Simple k-means
    # Initialize centroids randomly
    random.seed(42)
    indices = random.sample(range(len(graphs)), min(n_clusters, len(graphs)))
    centroids = [features_list[i] for i in indices]
    
    clusters = {i: [] for i in range(n_clusters)}
    
    # Iterate
    for _ in range(10):
        # Assign points to nearest centroid
        clusters = {i: [] for i in range(n_clusters)}
        
        for i, (g, feat) in enumerate(zip(graphs, features_list)):
            best_cluster = 0
            best_dist = float('inf')
            
            for c_idx, centroid in enumerate(centroids):
                dist = sum((a - b) ** 2 for a, b in zip(feat, centroid)) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_cluster = c_idx
            
            clusters[best_cluster].append(g)
        
        # Update centroids
        for c_idx in range(n_clusters):
            if clusters[c_idx]:
                cluster_feats = [features_list[graphs.index(g)] for g in clusters[c_idx]]
                centroids[c_idx] = [
                    sum(f[d] for f in cluster_feats) / len(cluster_feats)
                    for d in range(len(cluster_feats[0]))
                ]
    
    return clusters


# =============================================================================
# Export & Import
# =============================================================================

def export_weights(weights: Dict[str, Dict], name: str = None) -> str:
    """Export weights to active directory."""
    if name is None:
        name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Save to active directory
    graphs = list(weights.get('_metadata', {}).get('graphs', []))
    save_weights_to_active_type(weights, str(ACTIVE_DIR), 'type_0', graphs)
    
    # Also save to experiments directory
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    exp_file = EXPERIMENTS_DIR / f"{name}.json"
    with open(exp_file, 'w') as f:
        json.dump(weights, f, indent=2)
    
    log.info(f"Exported weights to {ACTIVE_DIR}/type_0.json")
    log.info(f"Saved experiment to {exp_file}")
    
    return str(exp_file)


def save_experiment_results(results: List[Dict], filename: str = None) -> str:
    """Save grid search or experiment results."""
    if filename is None:
        filename = f"grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = EXPERIMENTS_DIR / filename
    
    # Don't save full weights in results (too large)
    results_clean = []
    for r in results:
        r_clean = {k: v for k, v in r.items() if k != 'weights'}
        results_clean.append(r_clean)
    
    with open(filepath, 'w') as f:
        json.dump(results_clean, f, indent=2)
    
    log.info(f"Saved experiment results to {filepath}")
    return str(filepath)


# =============================================================================
# Interactive Mode
# =============================================================================

def interactive_mode(perf_matrix: Dict, graphs: List[str], results: Dict):
    """Interactive mode for manual weight tweaking."""
    print("\n" + "="*60)
    print("PERCEPTRON EXPERIMENTATION - INTERACTIVE MODE")
    print("="*60)
    
    # Start with current active weights or defaults
    try:
        weights = load_type_weights('type_0', str(ACTIVE_DIR))
        print(f"Loaded current weights from {ACTIVE_DIR}/type_0.json")
    except:
        weights = train_weights_hybrid(perf_matrix, graphs, PerceptronConfig())
        print("Started with hybrid-trained weights")
    
    while True:
        print("\nCommands:")
        print("  show            - Show current weights (top 10 by bias)")
        print("  eval [bench]    - Evaluate accuracy (default: pr)")
        print("  set ALGO BIAS   - Set algorithm bias (e.g., 'set GraphBrewOrder 5.0')")
        print("  train METHOD    - Retrain (speedup/winrate/rank/hybrid)")
        print("  grid            - Run grid search")
        print("  export          - Export to active directory")
        print("  quit            - Exit")
        
        try:
            cmd = input("\n> ").strip().lower().split()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not cmd:
            continue
        
        if cmd[0] == 'quit' or cmd[0] == 'q':
            break
        
        elif cmd[0] == 'show':
            # Show top algorithms by bias
            sorted_algos = sorted(
                [(k, v.get('bias', 0)) for k, v in weights.items() if not k.startswith('_')],
                key=lambda x: -x[1]
            )[:10]
            print("\nTop 10 algorithms by bias:")
            for algo, bias in sorted_algos:
                print(f"  {algo:25s}: {bias:.4f}")
        
        elif cmd[0] == 'eval':
            bench = cmd[1] if len(cmd) > 1 else 'pr'
            result = evaluate_weights(weights, perf_matrix, graphs, results, bench)
            print(f"\nAccuracy for {bench}: {result['accuracy']:.1f}%")
            print(f"Average regret: {result['avg_regret_pct']:.1f}%")
            print(f"Correct: {result['correct']}/{result['total']}")
        
        elif cmd[0] == 'set' and len(cmd) >= 3:
            algo = cmd[1]
            try:
                bias = float(cmd[2])
                if algo in weights:
                    weights[algo]['bias'] = bias
                    print(f"Set {algo} bias to {bias}")
                else:
                    print(f"Algorithm '{algo}' not found")
            except ValueError:
                print("Invalid bias value")
        
        elif cmd[0] == 'train' and len(cmd) >= 2:
            method = cmd[1]
            config = PerceptronConfig(method=method)
            
            if method == 'speedup':
                weights = train_weights_speedup(perf_matrix, graphs, config)
            elif method == 'winrate':
                weights = train_weights_winrate(perf_matrix, graphs, config)
            elif method == 'rank':
                weights = train_weights_rank(perf_matrix, graphs, config)
            elif method == 'hybrid':
                weights = train_weights_hybrid(perf_matrix, graphs, config)
            else:
                print(f"Unknown method: {method}")
                continue
            
            weights = add_feature_weights(weights, perf_matrix, graphs, results)
            print(f"Retrained with {method} method")
        
        elif cmd[0] == 'grid':
            search_results = grid_search(perf_matrix, graphs, results)
            print("\nTop 5 configurations:")
            for i, r in enumerate(search_results[:5]):
                print(f"  {i+1}. {r['config']}: acc={r['accuracy']:.1f}%, regret={r['regret']:.1f}%")
            
            # Offer to use best
            try:
                choice = input("\nUse best config? [y/n]: ").strip().lower()
                if choice == 'y':
                    weights = search_results[0]['weights']
                    print("Applied best configuration")
            except:
                pass
        
        elif cmd[0] == 'export':
            export_weights(weights)
        
        else:
            print(f"Unknown command: {cmd[0]}")
    
    print("\nExiting interactive mode")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Experiment with perceptron configurations without re-running phases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show current accuracy
  python3 -m scripts.lib.perceptron --show
  
  # Run grid search to find best configuration
  python3 -m scripts.lib.perceptron --grid-search
  
  # Train with specific method and export
  python3 -m scripts.lib.perceptron --train --method hybrid --export
  
  # Interactive mode
  python3 -m scripts.lib.perceptron --interactive
        """
    )
    
    parser.add_argument('--show', action='store_true',
                       help='Show current weights and evaluate accuracy')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze data: show graph types, algorithm categories, best per category')
    parser.add_argument('--grid-search', action='store_true',
                       help='Run grid search over configurations')
    parser.add_argument('--train', action='store_true',
                       help='Train new weights')
    parser.add_argument('--method', choices=['speedup', 'winrate', 'rank', 'hybrid', 'per_benchmark', 'perceptron'],
                       default='hybrid', help='Training method (default: hybrid)')
    parser.add_argument('--scale', type=float, default=1.0,
                       help='Bias scale factor (default: 1.0)')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                       help='Learning rate for perceptron method (default: 0.1)')
    parser.add_argument('--clusters', type=int, default=1,
                       help='Number of graph clusters (default: 1)')
    parser.add_argument('--benchmark', default='pr',
                       help='Benchmark to evaluate (default: pr)')
    parser.add_argument('--export', action='store_true',
                       help='Export weights to active directory')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Enter interactive mode')
    parser.add_argument('--save-results', type=str,
                       help='Save results to file')
    
    args = parser.parse_args()
    
    # Load data
    log.info("Loading results...")
    all_results = load_all_results()
    
    if not all_results['benchmarks']:
        log.error("No benchmark results found in results/")
        log.error("Run: python3 scripts/graphbrew_experiment.py --full --size small")
        sys.exit(1)
    
    perf_matrix = build_performance_matrix(all_results)
    graphs = list(perf_matrix.keys())
    
    log.info(f"Loaded data for {len(graphs)} graphs")
    
    if args.interactive:
        interactive_mode(perf_matrix, graphs, all_results)
        return
    
    if args.analyze:
        # Analyze data taxonomy
        print("\n" + "="*60)
        print("DATA TAXONOMY ANALYSIS")
        print("="*60)
        
        # Graph types
        print("\n📊 GRAPHS BY TYPE:")
        graph_types = all_results.get('graph_types', {})
        by_type = defaultdict(list)
        for g, t in graph_types.items():
            by_type[t].append(g)
        for gtype, glist in sorted(by_type.items()):
            print(f"  {gtype:12s}: {', '.join(glist)}")
        
        # Algorithm categories
        print("\n🔧 ALGORITHMS BY CATEGORY:")
        algo_cats = all_results.get('algo_categories', {})
        by_cat = defaultdict(list)
        for a, c in algo_cats.items():
            by_cat[c].append(a)
        for cat, alist in sorted(by_cat.items()):
            print(f"  {cat:12s}: {', '.join(sorted(alist))}")
        
        # Best algorithm per category per benchmark
        print("\n🏆 BEST ALGORITHM BY CATEGORY (per benchmark):")
        for bench in ['pr', 'bfs', 'cc', 'sssp', 'bc']:
            has_data = any(
                bench in perf_matrix.get(g, {}).get(a, {})
                for g in graphs for a in perf_matrix.get(g, {}).keys()
            )
            if not has_data:
                continue
            
            print(f"\n  {bench.upper()}:")
            cat_winners = defaultdict(lambda: {'algo': None, 'wins': 0, 'total_speedup': 0})
            
            for g in graphs:
                if g not in perf_matrix:
                    continue
                
                # Find best in each category for this graph
                for cat in ['basic', 'hub', 'community', 'leiden']:
                    best_algo = None
                    best_time = float('inf')
                    
                    for algo in by_cat.get(cat, []):
                        if algo in perf_matrix[g] and bench in perf_matrix[g][algo]:
                            t = perf_matrix[g][algo][bench]
                            if t < best_time:
                                best_time = t
                                best_algo = algo
                    
                    if best_algo:
                        # Check if this is overall best
                        overall_best, overall_time = find_best_algorithm(perf_matrix, g, bench)
                        if best_algo == overall_best:
                            cat_winners[cat]['wins'] += 1
                        if overall_time > 0:
                            cat_winners[cat]['total_speedup'] += overall_time / best_time
                        cat_winners[cat]['algo'] = best_algo
            
            for cat in ['basic', 'hub', 'community', 'leiden']:
                w = cat_winners[cat]
                if w['algo']:
                    print(f"    {cat:12s}: {w['algo']:25s} (wins: {w['wins']}/{len(graphs)})")
        
        # Data availability
        print("\n📁 DATA AVAILABILITY:")
        print(f"  Benchmarks: {'✅' if all_results['has_benchmarks'] else '❌'} ({len(all_results['benchmarks'])} records)")
        print(f"  Reorder:    {'✅' if all_results['has_reorder'] else '❌'} ({len(all_results['reorder_times'])} records)")
        print(f"  Cache:      {'✅' if all_results['has_cache'] else '⚠️ skipped'} ({len(all_results['cache'])} records)")
        
        return
    
    if args.show:
        # Load current weights
        try:
            weights = load_type_weights('type_0', str(ACTIVE_DIR))
            log.info(f"Loaded weights from {ACTIVE_DIR}/type_0.json")
        except Exception as e:
            log.warning(f"No active weights found: {e}")
            weights = {}
        
        # Show top algorithms
        if weights:
            sorted_algos = sorted(
                [(k, v.get('bias', 0)) for k, v in weights.items() if not k.startswith('_')],
                key=lambda x: -x[1]
            )[:15]
            print("\nTop 15 algorithms by bias:")
            for algo, bias in sorted_algos:
                print(f"  {algo:25s}: {bias:.4f}")
        
        # Evaluate
        if weights and perf_matrix:
            result = evaluate_all_benchmarks(weights, perf_matrix, graphs, all_results)
            print(f"\nOverall accuracy: {result['avg_accuracy']:.1f}%")
            print(f"Average regret: {result['avg_regret_pct']:.1f}%")
            print("\nPer-benchmark accuracy:")
            for bench, res in result['per_benchmark'].items():
                print(f"  {bench:6s}: {res['accuracy']:5.1f}% (regret: {res['avg_regret_pct']:.1f}%)")
        
        return
    
    if args.grid_search:
        results = grid_search(perf_matrix, graphs, all_results)
        
        print("\n" + "="*60)
        print("GRID SEARCH RESULTS (sorted by accuracy)")
        print("="*60)
        
        for i, r in enumerate(results[:15]):
            print(f"\n{i+1}. {r['config']}")
            acc5 = r.get('accuracy_5pct', 0)
            acc10 = r.get('accuracy_10pct', 0)
            print(f"   Accuracy: {r['accuracy']:.1f}% exact, ≤5%:{acc5:.0f}%, ≤10%:{acc10:.0f}%, Regret: {r['regret']:.1f}%")
            if r['per_benchmark']:
                bench_str = ", ".join(f"{k}:{v:.0f}%" for k, v in r['per_benchmark'].items())
                print(f"   Per-benchmark: {bench_str}")
        
        if args.save_results:
            save_experiment_results(results, args.save_results)
        else:
            save_experiment_results(results)
        
        # Offer to export best
        if args.export and results:
            export_weights(results[0]['weights'])
        
        return
    
    if args.train:
        config = PerceptronConfig(
            method=args.method,
            bias_scale=args.scale,
        )
        
        # Optionally cluster graphs
        if args.clusters > 1:
            clusters = cluster_graphs(graphs, all_results, args.clusters)
            log.info(f"Clustered graphs into {len(clusters)} groups")
            
            # Train per cluster (simplified: just use largest cluster for now)
            largest_cluster = max(clusters.values(), key=len)
            graphs = largest_cluster
        
        # Train
        log.info(f"Training with method={args.method}, scale={args.scale}")
        
        if args.method == 'speedup':
            weights = train_weights_speedup(perf_matrix, graphs, config)
        elif args.method == 'winrate':
            weights = train_weights_winrate(perf_matrix, graphs, config)
        elif args.method == 'rank':
            weights = train_weights_rank(perf_matrix, graphs, config)
        elif args.method == 'per_benchmark':
            weights = train_weights_per_benchmark(perf_matrix, graphs, config)
        elif args.method == 'perceptron':
            weights = train_weights_perceptron(perf_matrix, graphs, config, all_results,
                                               learning_rate=args.learning_rate)
        else:
            weights = train_weights_hybrid(perf_matrix, graphs, config)
        
        # Add feature weights (perceptron already trains its own)
        if args.method != 'perceptron':
            weights = add_feature_weights(weights, perf_matrix, graphs, all_results)
        
        # Add metadata
        weights['_metadata'] = {
            'method': args.method,
            'scale': args.scale,
            'graphs': graphs,
            'created': datetime.now().isoformat(),
        }
        
        # Evaluate
        result = evaluate_all_benchmarks(weights, perf_matrix, graphs, all_results)
        
        print(f"\nTraining Results:")
        print(f"  Method: {args.method}")
        print(f"  Scale: {args.scale}")
        print(f"  Graphs: {len(graphs)}")
        print(f"  Overall accuracy: {result['avg_accuracy']:.1f}%")
        print(f"  Average regret: {result['avg_regret_pct']:.1f}%")
        
        # Show top algorithms
        sorted_algos = sorted(
            [(k, v.get('bias', 0)) for k, v in weights.items() if not k.startswith('_')],
            key=lambda x: -x[1]
        )[:10]
        print("\nTop 10 algorithms by bias:")
        for algo, bias in sorted_algos:
            print(f"  {algo:25s}: {bias:.4f}")
        
        if args.export:
            export_weights(weights)
        
        return
    
    # Default: show help
    parser.print_help()


if __name__ == '__main__':
    main()
