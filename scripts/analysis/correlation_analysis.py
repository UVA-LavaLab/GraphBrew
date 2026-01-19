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
            # Find baseline (ORIGINAL)
            baseline_time = next((t for aid, _, t in results if aid == 0), results[0][2])
            
            # Compute speedups
            speedups = {}
            for algo_id, algo_name, trial_time in results:
                if trial_time > 0:
                    speedups[algo_id] = baseline_time / trial_time
            
            all_speedups[benchmark] = speedups
            
            # Find best
            best = min(results, key=lambda x: x[2])
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
