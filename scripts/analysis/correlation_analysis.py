#!/usr/bin/env python3
"""
Correlation Analysis for Graph Reordering

Analyzes the correlation between graph topology features and optimal reordering:
- Runs all algorithms on each graph
- Identifies the best reordering for each graph and algorithm
- Correlates graph features (nodes, edges, density, modularity, etc.) with best algorithm
- Generates recommendations and perceptron training data

Usage:
    python3 scripts/analysis/correlation_analysis.py [--graphs-dir DIR]
    
Examples:
    python3 scripts/analysis/correlation_analysis.py --quick
    python3 scripts/analysis/correlation_analysis.py --graphs-dir ./graphs --benchmark pr bfs
"""

import os
import sys
import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.common import (
    ALGORITHMS, COMMUNITY_ALGORITHMS, LEIDEN_ALGORITHMS,
    BenchmarkResult, GraphFeatures,
    run_benchmark, extract_graph_features, compute_speedups,
    Colors, print_header, print_subheader, format_time, format_speedup
)

# ============================================================================
# Configuration
# ============================================================================

BENCHMARKS = ["pr", "bfs", "cc"]
DEFAULT_TRIALS = 3

# Synthetic graphs for testing
SYNTHETIC_GRAPHS = {
    "rmat_12": "-g 12",
    "rmat_14": "-g 14",
    "rmat_16": "-g 16",
}

# ============================================================================
# Correlation Analysis
# ============================================================================

@dataclass
class CorrelationResult:
    """Result of correlation analysis for a single graph."""
    graph_name: str
    features: GraphFeatures
    best_algorithm: Dict[str, Tuple[int, str, float]]  # benchmark -> (algo_id, name, time)
    all_speedups: Dict[str, Dict[int, float]]  # benchmark -> algo_id -> speedup


def run_benchmark_for_correlation(
    graph_name: str,
    graph_args: str,
    benchmarks: List[str],
    algorithms: Dict[int, str] = None,
    num_trials: int = DEFAULT_TRIALS,
    timeout: int = 300,
    verbose: bool = True
) -> Optional[CorrelationResult]:
    """
    Run benchmarks and collect correlation data for a single graph.
    """
    if algorithms is None:
        algorithms = ALGORITHMS
    
    # Extract graph features
    features = extract_graph_features(graph_args)
    if features is None:
        if verbose:
            print(f"  Could not extract features for {graph_name}")
        return None
    features.name = graph_name
    
    best_algorithm = {}
    all_speedups = {}
    
    for benchmark in benchmarks:
        binary = f"./bench/bin/{benchmark}"
        results = []
        
        if verbose:
            print(f"  {benchmark}:", end=" ", flush=True)
        
        for algo_id, algo_name in sorted(algorithms.items()):
            parsed, output = run_benchmark(
                binary=binary,
                graph_args=graph_args,
                algo_id=algo_id,
                num_trials=num_trials,
                timeout=timeout
            )
            
            if parsed:
                trial_time = parsed.get('average_time', float('inf'))
                results.append((algo_id, algo_name, trial_time))
        
        if results:
            # Filter out invalid results (NaN, inf, None)
            import math
            valid_results = [(a, n, t) for a, n, t in results 
                           if t is not None and not math.isnan(t) and not math.isinf(t) and t > 0]
            
            if not valid_results:
                if verbose:
                    print("No valid results")
                continue
            
            # Find baseline (ORIGINAL)
            baseline_time = next((t for aid, _, t in valid_results if aid == 0), valid_results[0][2])
            
            # Compute speedups
            speedups = {}
            for algo_id, algo_name, trial_time in valid_results:
                if trial_time > 0:
                    speedups[algo_id] = baseline_time / trial_time
            
            all_speedups[benchmark] = speedups
            
            # Find best
            best = min(valid_results, key=lambda x: x[2])
            best_algorithm[benchmark] = best
            
            if verbose:
                print(f"{best[1]} ({format_time(best[2])})", end=" ")
        
        if verbose:
            print()
    
    return CorrelationResult(
        graph_name=graph_name,
        features=features,
        best_algorithm=best_algorithm,
        all_speedups=all_speedups
    )


def analyze_correlations(
    results: List[CorrelationResult],
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Analyze correlations between graph features and best algorithms.
    
    Returns correlation coefficients for each feature-algorithm pair.
    """
    # Collect data for correlation
    feature_names = ['log_nodes', 'log_edges', 'density', 'avg_degree']
    
    correlations = defaultdict(dict)
    
    # Group by best algorithm
    algo_features = defaultdict(lambda: defaultdict(list))
    
    for result in results:
        for benchmark, (algo_id, algo_name, _) in result.best_algorithm.items():
            for fname in feature_names:
                fval = getattr(result.features, fname, 0)
                algo_features[algo_name][fname].append(fval)
    
    # Compute statistics
    if verbose:
        print_subheader("Feature Statistics by Best Algorithm")
        print(f"{'Algorithm':<20}", end="")
        for fname in feature_names:
            print(f"{fname:>15}", end="")
        print(f"{'Count':>10}")
        print("-" * (20 + 15 * len(feature_names) + 10))
    
    for algo_name, features in sorted(algo_features.items()):
        if verbose:
            print(f"{algo_name:<20}", end="")
        for fname in feature_names:
            vals = features[fname]
            if vals:
                avg = sum(vals) / len(vals)
                if verbose:
                    print(f"{avg:>15.3f}", end="")
                correlations[algo_name][fname] = avg
        if verbose:
            count = len(features[feature_names[0]])
            print(f"{count:>10}")
    
    return correlations


def compute_algorithm_scores(
    results: List[CorrelationResult]
) -> Dict[str, Dict[str, float]]:
    """
    Compute aggregate scores for each algorithm across all graphs.
    """
    scores = defaultdict(lambda: {
        'wins': 0,
        'total_speedup': 0,
        'avg_speedup': 0,
        'graphs_tested': 0
    })
    
    for result in results:
        for benchmark, speedups in result.all_speedups.items():
            if not speedups:
                continue
            
            best_algo = max(speedups.items(), key=lambda x: x[1])
            
            for algo_id, speedup in speedups.items():
                algo_name = ALGORITHMS.get(algo_id, f"Unknown({algo_id})")
                scores[algo_name]['total_speedup'] += speedup
                scores[algo_name]['graphs_tested'] += 1
                
                if algo_id == best_algo[0]:
                    scores[algo_name]['wins'] += 1
    
    # Compute averages
    for algo_name, data in scores.items():
        if data['graphs_tested'] > 0:
            data['avg_speedup'] = data['total_speedup'] / data['graphs_tested']
    
    return dict(scores)


def print_algorithm_rankings(scores: Dict[str, Dict[str, float]]):
    """Print algorithm rankings based on aggregate scores."""
    print_subheader("Algorithm Rankings")
    
    # Sort by wins then by average speedup
    ranked = sorted(
        scores.items(),
        key=lambda x: (x[1]['wins'], x[1]['avg_speedup']),
        reverse=True
    )
    
    print(f"{'Rank':<6} {'Algorithm':<20} {'Wins':>8} {'Avg Speedup':>12} {'Tests':>8}")
    print("-" * 60)
    
    for rank, (algo_name, data) in enumerate(ranked, 1):
        speedup_str = format_speedup(data['avg_speedup'])
        print(f"{rank:<6} {algo_name:<20} {data['wins']:>8} {speedup_str:>12} {data['graphs_tested']:>8}")


def generate_recommendations(
    results: List[CorrelationResult],
    correlations: Dict[str, Dict[str, float]]
) -> Dict[str, str]:
    """
    Generate algorithm recommendations based on graph features.
    """
    recommendations = {}
    
    # Simple heuristic based on observed patterns
    # This would be replaced with proper ML in production
    
    # Analyze patterns
    large_graph_winners = []
    small_graph_winners = []
    dense_graph_winners = []
    sparse_graph_winners = []
    
    for result in results:
        is_large = result.features.log_nodes > 5
        is_dense = result.features.density > 0.001
        
        for benchmark, (algo_id, algo_name, _) in result.best_algorithm.items():
            if is_large:
                large_graph_winners.append(algo_name)
            else:
                small_graph_winners.append(algo_name)
            if is_dense:
                dense_graph_winners.append(algo_name)
            else:
                sparse_graph_winners.append(algo_name)
    
    # Find most common winners
    from collections import Counter
    
    recommendations['large_graphs'] = Counter(large_graph_winners).most_common(1)[0][0] if large_graph_winners else "LeidenHybrid"
    recommendations['small_graphs'] = Counter(small_graph_winners).most_common(1)[0][0] if small_graph_winners else "HUBCLUSTERDBG"
    recommendations['dense_graphs'] = Counter(dense_graph_winners).most_common(1)[0][0] if dense_graph_winners else "LeidenHybrid"
    recommendations['sparse_graphs'] = Counter(sparse_graph_winners).most_common(1)[0][0] if sparse_graph_winners else "RCM"
    
    return recommendations


def compute_perceptron_weights(
    results: List[CorrelationResult],
    output_file: str = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute perceptron weights from benchmark results for ALL algorithms (0-20).
    
    For each algorithm, computes weights that correlate features with performance.
    The weights represent how well the algorithm performs given certain features.
    
    - Generates weights for ALL algorithms in ALGORITHMS dict
    - Loads existing weights file and merges/updates with new data
    - Uses sensible defaults for algorithms without benchmark data
    
    Output format (JSON):
    {
        "ORIGINAL": {"bias": 1.0, "w_modularity": 0.3, ...},
        "LeidenHybrid": {"bias": 0.95, "w_modularity": 0.45, ...},
        ...
    }
    """
    from collections import defaultdict
    import numpy as np
    
    feature_names = ['w_modularity', 'w_log_nodes', 'w_log_edges', 
                     'w_density', 'w_avg_degree', 'w_degree_variance', 
                     'w_hub_concentration']
    
    # Default weights for each algorithm category (tuned heuristics)
    DEFAULT_WEIGHTS = {
        # Basic algorithms - conservative, work well on small/simple graphs
        "ORIGINAL": {"bias": 0.6, "w_modularity": 0.0, "w_log_nodes": -0.1, "w_log_edges": -0.1, 
                     "w_density": 0.1, "w_avg_degree": 0.0, "w_degree_variance": 0.0, "w_hub_concentration": 0.0},
        "RANDOM": {"bias": 0.3, "w_modularity": 0.0, "w_log_nodes": 0.0, "w_log_edges": 0.0, 
                   "w_density": 0.0, "w_avg_degree": 0.0, "w_degree_variance": 0.0, "w_hub_concentration": 0.0},
        "SORT": {"bias": 0.4, "w_modularity": 0.0, "w_log_nodes": 0.0, "w_log_edges": 0.0, 
                 "w_density": 0.0, "w_avg_degree": 0.0, "w_degree_variance": 0.0, "w_hub_concentration": 0.0},
        
        # Hub-based algorithms - good for power-law graphs
        "HUBSORT": {"bias": 0.55, "w_modularity": 0.05, "w_log_nodes": 0.0, "w_log_edges": 0.05, 
                    "w_density": -0.05, "w_avg_degree": 0.1, "w_degree_variance": 0.15, "w_hub_concentration": 0.2},
        "HUBCLUSTER": {"bias": 0.6, "w_modularity": 0.1, "w_log_nodes": 0.0, "w_log_edges": 0.05, 
                       "w_density": -0.05, "w_avg_degree": 0.1, "w_degree_variance": 0.15, "w_hub_concentration": 0.25},
        
        # DBG-based algorithms - good for locality
        "DBG": {"bias": 0.65, "w_modularity": 0.1, "w_log_nodes": 0.05, "w_log_edges": 0.05, 
                "w_density": 0.0, "w_avg_degree": 0.1, "w_degree_variance": 0.1, "w_hub_concentration": 0.1},
        "HUBSORTDBG": {"bias": 0.7, "w_modularity": 0.1, "w_log_nodes": 0.05, "w_log_edges": 0.05, 
                       "w_density": -0.05, "w_avg_degree": 0.15, "w_degree_variance": 0.15, "w_hub_concentration": 0.2},
        "HUBCLUSTERDBG": {"bias": 0.75, "w_modularity": 0.15, "w_log_nodes": 0.05, "w_log_edges": 0.05, 
                          "w_density": -0.05, "w_avg_degree": 0.15, "w_degree_variance": 0.15, "w_hub_concentration": 0.25},
        
        # Community-based algorithms - excellent for modular graphs
        "RABBITORDER": {"bias": 0.7, "w_modularity": 0.2, "w_log_nodes": 0.1, "w_log_edges": 0.1, 
                        "w_density": -0.1, "w_avg_degree": 0.1, "w_degree_variance": 0.1, "w_hub_concentration": 0.15},
        "GORDER": {"bias": 0.65, "w_modularity": 0.1, "w_log_nodes": 0.05, "w_log_edges": 0.05, 
                   "w_density": 0.0, "w_avg_degree": 0.1, "w_degree_variance": 0.05, "w_hub_concentration": 0.1},
        "CORDER": {"bias": 0.6, "w_modularity": 0.1, "w_log_nodes": 0.05, "w_log_edges": 0.05, 
                   "w_density": 0.0, "w_avg_degree": 0.05, "w_degree_variance": 0.05, "w_hub_concentration": 0.1},
        "RCM": {"bias": 0.55, "w_modularity": 0.0, "w_log_nodes": 0.0, "w_log_edges": 0.0, 
                "w_density": 0.1, "w_avg_degree": -0.05, "w_degree_variance": 0.0, "w_hub_concentration": 0.0},
        
        # Leiden-based algorithms - best for community structure
        "LeidenOrder": {"bias": 0.75, "w_modularity": 0.25, "w_log_nodes": 0.1, "w_log_edges": 0.1, 
                        "w_density": -0.1, "w_avg_degree": 0.1, "w_degree_variance": 0.1, "w_hub_concentration": 0.15},
        "GraphBrewOrder": {"bias": 0.8, "w_modularity": 0.25, "w_log_nodes": 0.1, "w_log_edges": 0.1, 
                           "w_density": -0.1, "w_avg_degree": 0.15, "w_degree_variance": 0.15, "w_hub_concentration": 0.2},
        "AdaptiveOrder": {"bias": 0.85, "w_modularity": 0.2, "w_log_nodes": 0.1, "w_log_edges": 0.1, 
                          "w_density": 0.0, "w_avg_degree": 0.15, "w_degree_variance": 0.15, "w_hub_concentration": 0.2},
        "LeidenDFS": {"bias": 0.75, "w_modularity": 0.2, "w_log_nodes": 0.1, "w_log_edges": 0.1, 
                      "w_density": -0.1, "w_avg_degree": 0.1, "w_degree_variance": 0.1, "w_hub_concentration": 0.15},
        "LeidenDFSHub": {"bias": 0.8, "w_modularity": 0.2, "w_log_nodes": 0.1, "w_log_edges": 0.1, 
                         "w_density": -0.1, "w_avg_degree": 0.15, "w_degree_variance": 0.15, "w_hub_concentration": 0.25},
        "LeidenDFSSize": {"bias": 0.78, "w_modularity": 0.2, "w_log_nodes": 0.15, "w_log_edges": 0.1, 
                          "w_density": -0.1, "w_avg_degree": 0.1, "w_degree_variance": 0.1, "w_hub_concentration": 0.15},
        "LeidenBFS": {"bias": 0.75, "w_modularity": 0.2, "w_log_nodes": 0.1, "w_log_edges": 0.1, 
                      "w_density": -0.1, "w_avg_degree": 0.1, "w_degree_variance": 0.1, "w_hub_concentration": 0.15},
        "LeidenHybrid": {"bias": 0.85, "w_modularity": 0.25, "w_log_nodes": 0.1, "w_log_edges": 0.1, 
                         "w_density": -0.05, "w_avg_degree": 0.15, "w_degree_variance": 0.15, "w_hub_concentration": 0.25},
    }
    
    # Determine output path (default: scripts/perceptron_weights.json)
    if output_file is None:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_file = os.path.join(script_dir, "perceptron_weights.json")
    
    # Start with defaults for ALL algorithms
    perceptron_weights = {}
    for algo_id, algo_name in ALGORITHMS.items():
        if algo_name in DEFAULT_WEIGHTS:
            perceptron_weights[algo_name] = DEFAULT_WEIGHTS[algo_name].copy()
        else:
            # Fallback for any algorithm not in defaults
            perceptron_weights[algo_name] = {
                "bias": 0.5, "w_modularity": 0.0, "w_log_nodes": 0.0, "w_log_edges": 0.0,
                "w_density": 0.0, "w_avg_degree": 0.0, "w_degree_variance": 0.0, "w_hub_concentration": 0.0
            }
    
    # Load existing weights file and merge (preserves manually tuned values)
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                existing_weights = json.load(f)
            # Merge existing weights (they take precedence for algorithms not being updated)
            for algo_name, weights in existing_weights.items():
                if algo_name in perceptron_weights:
                    perceptron_weights[algo_name].update(weights)
            print(f"Loaded existing weights from: {output_file}")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load existing weights: {e}")
    
    # Collect training data from benchmark results
    algo_data = defaultdict(lambda: {
        'speedups': [],
        'features': [],
        'wins': 0,
        'total_tests': 0
    })
    
    for result in results:
        features = [
            getattr(result.features, 'modularity', 0.5),  # estimated modularity
            result.features.log_nodes,
            result.features.log_edges,
            result.features.density,
            result.features.avg_degree / 100.0,  # normalized
            getattr(result.features, 'degree_variance', 0.5),
            getattr(result.features, 'hub_concentration', 0.3),
        ]
        
        for benchmark, speedups in result.all_speedups.items():
            if not speedups:
                continue
            
            best_algo_id = max(speedups.items(), key=lambda x: x[1])[0]
            
            for algo_id, speedup in speedups.items():
                algo_name = ALGORITHMS.get(algo_id, f"Unknown({algo_id})")
                algo_data[algo_name]['speedups'].append(speedup)
                algo_data[algo_name]['features'].append(features)
                algo_data[algo_name]['total_tests'] += 1
                
                if algo_id == best_algo_id:
                    algo_data[algo_name]['wins'] += 1
    
    # Update weights for algorithms with benchmark data
    updated_algos = []
    for algo_name, data in algo_data.items():
        if data['total_tests'] == 0:
            continue
        
        avg_speedup = sum(data['speedups']) / len(data['speedups'])
        win_rate = data['wins'] / data['total_tests']
        
        # Simple heuristic: base bias on win rate and average speedup
        bias = 0.5 + 0.3 * win_rate + 0.2 * min(avg_speedup / 5.0, 1.0)
        
        # Compute feature correlations with speedup
        weights = {'bias': round(bias, 3)}
        
        if len(data['speedups']) >= 3:
            speedups_arr = np.array(data['speedups'])
            features_arr = np.array(data['features'])
            
            # Simple correlation-based weights
            for i, fname in enumerate(feature_names):
                if features_arr.shape[0] > 1:
                    feature_col = features_arr[:, i]
                    if np.std(feature_col) > 0:
                        corr = np.corrcoef(feature_col, speedups_arr)[0, 1]
                        if not np.isnan(corr):
                            weights[fname] = round(corr * 0.3, 3)  # Scale correlation
                        else:
                            weights[fname] = perceptron_weights[algo_name].get(fname, 0.0)
                    else:
                        weights[fname] = perceptron_weights[algo_name].get(fname, 0.0)
                else:
                    weights[fname] = perceptron_weights[algo_name].get(fname, 0.0)
        else:
            # Not enough data, keep defaults
            for fname in feature_names:
                weights[fname] = perceptron_weights[algo_name].get(fname, 0.0)
        
        # Update weights for this algorithm
        perceptron_weights[algo_name].update(weights)
        updated_algos.append(algo_name)
    
    # Create backup if file exists
    if os.path.exists(output_file):
        backup_file = output_file + ".backup"
        import shutil
        shutil.copy2(output_file, backup_file)
        print(f"Backup created: {backup_file}")
    
    # Save to file (sorted by algorithm ID for consistency)
    sorted_weights = {}
    algo_name_to_id = {name: id for id, name in ALGORITHMS.items()}
    for algo_name in sorted(perceptron_weights.keys(), key=lambda x: algo_name_to_id.get(x, 999)):
        sorted_weights[algo_name] = perceptron_weights[algo_name]
    
    with open(output_file, 'w') as f:
        json.dump(sorted_weights, f, indent=2)
    
    print(f"\nPerceptron weights saved to: {output_file}")
    print(f"  {len(sorted_weights)} algorithms configured (all 0-20)")
    if updated_algos:
        print(f"  Updated from benchmarks: {', '.join(updated_algos)}")
    else:
        print(f"  Using default weights (no benchmark data)")
    print(f"  C++ will automatically load these weights at runtime")
    
    return sorted_weights


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze correlation between graph features and optimal reordering"
    )
    parser.add_argument(
        "--graphs-dir", "-g",
        default=None,
        help="Directory containing graphs"
    )
    parser.add_argument(
        "--graphs-config",
        default=None,
        help="JSON config file with graph definitions"
    )
    parser.add_argument(
        "--benchmark", "-b",
        nargs='+',
        default=["pr"],
        help="Benchmarks to analyze (default: pr)"
    )
    parser.add_argument(
        "--algorithms", "-a",
        default=None,
        help="Comma-separated list of algorithm IDs (default: all)"
    )
    parser.add_argument(
        "--trials", "-n",
        type=int,
        default=DEFAULT_TRIALS,
        help=f"Number of trials (default: {DEFAULT_TRIALS})"
    )
    parser.add_argument(
        "--output", "-o",
        default="./bench/results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick test with synthetic graphs"
    )
    parser.add_argument(
        "--weights-file", "-w",
        default=None,
        help="Output file for perceptron weights (default: scripts/perceptron_weights.json)"
    )
    
    args = parser.parse_args()
    
    # Determine graphs to use
    if args.quick:
        graphs = SYNTHETIC_GRAPHS
    elif args.graphs_config:
        with open(args.graphs_config, 'r') as f:
            config = json.load(f)
        graphs = {name: info.get("args", f"-f {info['path']} -s")
                  for name, info in config.get("graphs", {}).items()}
    elif args.graphs_dir:
        # Load from directory
        graphs = {}
        config_path = os.path.join(args.graphs_dir, "graphs.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            for name, info in config.get("graphs", {}).items():
                if "args" in info:
                    graphs[name] = info["args"]
                elif "path" in info:
                    sym = "-s" if info.get("symmetric", True) else ""
                    graphs[name] = f"-f {info['path']} {sym}"
    else:
        graphs = SYNTHETIC_GRAPHS
    
    # Determine algorithms
    if args.algorithms:
        algo_ids = [int(x.strip()) for x in args.algorithms.split(',')]
        algorithms = {aid: ALGORITHMS.get(aid, f"Unknown({aid})") for aid in algo_ids}
    else:
        algorithms = ALGORITHMS
    
    # Run analysis
    print_header("Correlation Analysis")
    print(f"Graphs: {len(graphs)}")
    print(f"Benchmarks: {', '.join(args.benchmark)}")
    print(f"Algorithms: {len(algorithms)}")
    
    results = []
    for graph_name, graph_args in graphs.items():
        print_subheader(f"Graph: {graph_name}")
        
        result = run_benchmark_for_correlation(
            graph_name=graph_name,
            graph_args=graph_args,
            benchmarks=args.benchmark,
            algorithms=algorithms,
            num_trials=args.trials
        )
        
        if result:
            results.append(result)
            print(f"  Features: nodes={result.features.nodes:,}, edges={result.features.edges:,}, "
                  f"density={result.features.density:.6f}")
    
    if not results:
        print("No results collected")
        return 1
    
    # Analyze correlations
    correlations = analyze_correlations(results)
    
    # Compute scores
    scores = compute_algorithm_scores(results)
    print_algorithm_rankings(scores)
    
    # Generate recommendations
    recommendations = generate_recommendations(results, correlations)
    
    print_subheader("Recommendations")
    for category, algo in recommendations.items():
        print(f"  {category.replace('_', ' ').title()}: {algo}")
    
    # Compute and save perceptron weights
    print_subheader("Computing Perceptron Weights")
    perceptron_weights = compute_perceptron_weights(
        results, 
        output_file=args.weights_file
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output, exist_ok=True)
    
    output_data = {
        'correlations': correlations,
        'scores': scores,
        'recommendations': recommendations,
        'results': [
            {
                'graph': r.graph_name,
                'features': r.features.to_dict(),
                'best_algorithm': {b: {'id': a[0], 'name': a[1], 'time': a[2]}
                                   for b, a in r.best_algorithm.items()},
                'speedups': r.all_speedups
            }
            for r in results
        ]
    }
    
    json_path = os.path.join(args.output, f"correlation_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {json_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
