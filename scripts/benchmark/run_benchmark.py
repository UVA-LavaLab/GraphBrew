#!/usr/bin/env python3
"""
Comprehensive Benchmark Suite for GraphBrew

Runs all algorithms on a set of graphs and collects:
- Preprocessing time (reordering time)
- Trial time (algorithm execution time)
- Speedups relative to ORIGINAL baseline

For BFS, SSSP, BC: Uses multiple source nodes for stable timing.

Usage:
    python3 scripts/benchmark/run_benchmark.py [--graphs-dir DIR] [--output DIR]
    
Examples:
    python3 scripts/benchmark/run_benchmark.py --graphs-dir ./graphs --benchmark pr
    python3 scripts/benchmark/run_benchmark.py --quick   # Quick test with rmat graphs
"""

import os
import sys
import argparse
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.common import (
    ALGORITHMS, QUICK_ALGORITHMS, MULTI_SOURCE_BENCHMARKS,
    DEFAULT_SOURCE_TRIALS, BenchmarkResult, GraphFeatures,
    run_benchmark, run_multi_source_benchmark, extract_graph_features,
    compute_speedups, save_results_json, save_results_csv,
    Colors, print_header, print_subheader, format_time, format_speedup
)

# ============================================================================
# Configuration
# ============================================================================

BENCHMARKS = ["pr", "bfs", "cc", "sssp", "bc", "tc"]
DEFAULT_TRIALS = 5
MULTI_SOURCE_TRIALS = 16  # For BFS, SSSP, BC

# Synthetic graphs for quick testing
SYNTHETIC_GRAPHS = {
    "rmat_14": "-g 14",
    "rmat_16": "-g 16",
    "rmat_18": "-g 18",
    "rmat_20": "-g 20",
}

# ============================================================================
# Benchmark Functions
# ============================================================================

def run_single_benchmark(
    benchmark: str,
    graph_name: str,
    graph_args: str,
    algo_id: int,
    num_trials: int = DEFAULT_TRIALS,
    timeout: int = 600
) -> Optional[BenchmarkResult]:
    """Run a single benchmark configuration."""
    binary = f"./bench/bin/{benchmark}"
    
    # Use more trials for multi-source benchmarks
    if benchmark in MULTI_SOURCE_BENCHMARKS:
        actual_trials = max(num_trials, MULTI_SOURCE_TRIALS)
    else:
        actual_trials = num_trials
    
    parsed, output = run_benchmark(
        binary=binary,
        graph_args=graph_args,
        algo_id=algo_id,
        num_trials=actual_trials,
        verify=False,
        timeout=timeout
    )
    
    if parsed is None:
        return None
    
    result = BenchmarkResult(
        algorithm_id=algo_id,
        algorithm_name=ALGORITHMS.get(algo_id, f"Unknown({algo_id})"),
        graph_name=graph_name,
        benchmark=benchmark,
        reorder_time=parsed.get('reorder_time', 0) + parsed.get('relabel_time', 0),
        trial_time=parsed.get('average_time', 0),
        total_time=parsed.get('total_time', 0),
        nodes=parsed.get('nodes', 0),
        edges=parsed.get('edges', 0),
        avg_degree=parsed.get('avg_degree', 0),
        iterations=parsed.get('iterations', 0),
        verified=parsed.get('verified', False),
    )
    
    return result


def run_all_algorithms(
    benchmark: str,
    graph_name: str,
    graph_args: str,
    algorithms: Dict[int, str] = None,
    num_trials: int = DEFAULT_TRIALS,
    timeout: int = 600,
    verbose: bool = True
) -> List[BenchmarkResult]:
    """Run all specified algorithms on a single graph."""
    if algorithms is None:
        algorithms = ALGORITHMS
    
    results = []
    
    for algo_id, algo_name in sorted(algorithms.items()):
        if verbose:
            print(f"  {algo_name:15} ({algo_id:2d})...", end=" ", flush=True)
        
        result = run_single_benchmark(
            benchmark=benchmark,
            graph_name=graph_name,
            graph_args=graph_args,
            algo_id=algo_id,
            num_trials=num_trials,
            timeout=timeout
        )
        
        if result:
            results.append(result)
            if verbose:
                print(f"{format_time(result.trial_time):>10} (reorder: {format_time(result.reorder_time)})")
        else:
            if verbose:
                print(f"{Colors.YELLOW}SKIPPED{Colors.RESET}")
    
    return results


def run_benchmark_suite(
    graphs: Dict[str, str],
    benchmarks: List[str],
    algorithms: Dict[int, str] = None,
    num_trials: int = DEFAULT_TRIALS,
    timeout: int = 600,
    verbose: bool = True
) -> List[BenchmarkResult]:
    """
    Run full benchmark suite across all graphs, benchmarks, and algorithms.
    """
    if algorithms is None:
        algorithms = ALGORITHMS
    
    all_results = []
    
    total_configs = len(graphs) * len(benchmarks) * len(algorithms)
    print_header(f"GraphBrew Benchmark Suite")
    print(f"Graphs: {len(graphs)}")
    print(f"Benchmarks: {', '.join(benchmarks)}")
    print(f"Algorithms: {len(algorithms)}")
    print(f"Total configurations: {total_configs}")
    
    for graph_name, graph_args in graphs.items():
        print_subheader(f"Graph: {graph_name}")
        
        for benchmark in benchmarks:
            if verbose:
                print(f"\n  Benchmark: {benchmark}")
                print(f"  {'-'*50}")
            
            results = run_all_algorithms(
                benchmark=benchmark,
                graph_name=graph_name,
                graph_args=graph_args,
                algorithms=algorithms,
                num_trials=num_trials,
                timeout=timeout,
                verbose=verbose
            )
            
            all_results.extend(results)
    
    # Compute speedups
    all_results = compute_speedups(all_results, baseline_algo=0)
    
    return all_results


# ============================================================================
# Results Analysis
# ============================================================================

def print_results_table(results: List[BenchmarkResult], graph_name: str = None):
    """Print results in a formatted table."""
    # Filter by graph if specified
    if graph_name:
        results = [r for r in results if r.graph_name == graph_name]
    
    if not results:
        print("No results to display")
        return
    
    # Group by benchmark
    by_benchmark = {}
    for r in results:
        if r.benchmark not in by_benchmark:
            by_benchmark[r.benchmark] = []
        by_benchmark[r.benchmark].append(r)
    
    for benchmark, bench_results in by_benchmark.items():
        print(f"\n{Colors.BOLD}{benchmark.upper()}{Colors.RESET}")
        print("-" * 80)
        print(f"{'Algorithm':<18} {'Reorder':>12} {'Trial':>12} {'Total':>12} {'Speedup':>10}")
        print("-" * 80)
        
        # Sort by trial time
        bench_results.sort(key=lambda x: x.trial_time)
        
        for r in bench_results:
            speedup_str = format_speedup(r.speedup)
            print(f"{r.algorithm_name:<18} "
                  f"{format_time(r.reorder_time):>12} "
                  f"{format_time(r.trial_time):>12} "
                  f"{format_time(r.total_time):>12} "
                  f"{speedup_str:>10}")
        
        # Find best
        best = min(bench_results, key=lambda x: x.trial_time)
        print("-" * 80)
        print(f"{Colors.GREEN}Best: {best.algorithm_name} ({format_time(best.trial_time)}){Colors.RESET}")


def print_summary(results: List[BenchmarkResult]):
    """Print overall summary of results."""
    print_header("Summary")
    
    # Group by graph
    by_graph = {}
    for r in results:
        if r.graph_name not in by_graph:
            by_graph[r.graph_name] = []
        by_graph[r.graph_name].append(r)
    
    # Find best algorithm per graph
    print(f"\n{'Graph':<20} {'Best Algorithm':<20} {'Speedup':>10}")
    print("-" * 55)
    
    algo_wins = {}
    for graph_name, graph_results in sorted(by_graph.items()):
        best = max(graph_results, key=lambda x: x.speedup)
        print(f"{graph_name:<20} {best.algorithm_name:<20} {format_speedup(best.speedup)}")
        
        if best.algorithm_name not in algo_wins:
            algo_wins[best.algorithm_name] = 0
        algo_wins[best.algorithm_name] += 1
    
    # Algorithm win counts
    print(f"\n{Colors.BOLD}Algorithm Win Counts:{Colors.RESET}")
    for algo, wins in sorted(algo_wins.items(), key=lambda x: -x[1]):
        print(f"  {algo}: {wins} wins")


# ============================================================================
# Graph Loading
# ============================================================================

def load_graphs_from_dir(graphs_dir: str) -> Dict[str, str]:
    """Load graphs from a directory."""
    graphs = {}
    
    # Look for config file
    config_path = os.path.join(graphs_dir, "graphs.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        for name, info in config.get("graphs", {}).items():
            if "args" in info:
                graphs[name] = info["args"]
            elif "path" in info:
                sym = "-s" if info.get("symmetric", True) else ""
                graphs[name] = f"-f {info['path']} {sym}"
        return graphs
    
    # Scan directory for graph files
    for item in os.listdir(graphs_dir):
        item_path = os.path.join(graphs_dir, item)
        if os.path.isdir(item_path):
            # Look for graph file in subdirectory
            for fmt in ['mtx', 'el', 'wel', 'graph']:
                graph_file = os.path.join(item_path, f"graph.{fmt}")
                if os.path.exists(graph_file):
                    graphs[item] = f"-f {graph_file} -s"
                    break
        elif item.endswith(('.mtx', '.el', '.wel', '.graph')):
            name = os.path.splitext(item)[0]
            graphs[name] = f"-f {item_path} -s"
    
    return graphs


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive benchmark suite for GraphBrew"
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
        help="Benchmarks to run (default: pr)"
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
        "--timeout", "-t",
        type=int,
        default=600,
        help="Timeout per benchmark in seconds (default: 600)"
    )
    
    args = parser.parse_args()
    
    # Determine graphs to use
    if args.quick:
        graphs = SYNTHETIC_GRAPHS
        print("Using synthetic RMAT graphs for quick test")
    elif args.graphs_config:
        with open(args.graphs_config, 'r') as f:
            config = json.load(f)
        graphs = {name: info.get("args", f"-f {info['path']} -s")
                  for name, info in config.get("graphs", {}).items()}
    elif args.graphs_dir:
        graphs = load_graphs_from_dir(args.graphs_dir)
    else:
        # Default to small synthetic graphs
        graphs = {"rmat_14": "-g 14", "rmat_16": "-g 16"}
        print("No graphs specified, using default synthetic graphs")
    
    if not graphs:
        print("Error: No graphs found")
        return 1
    
    # Determine algorithms
    if args.algorithms:
        algo_ids = [int(x.strip()) for x in args.algorithms.split(',')]
        algorithms = {aid: ALGORITHMS.get(aid, f"Unknown({aid})") for aid in algo_ids}
    else:
        algorithms = ALGORITHMS
    
    # Run benchmarks
    results = run_benchmark_suite(
        graphs=graphs,
        benchmarks=args.benchmark,
        algorithms=algorithms,
        num_trials=args.trials,
        timeout=args.timeout
    )
    
    # Print results
    for graph_name in graphs.keys():
        print_results_table(results, graph_name)
    
    print_summary(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output, exist_ok=True)
    
    json_path = os.path.join(args.output, f"benchmark_{timestamp}.json")
    csv_path = os.path.join(args.output, f"benchmark_{timestamp}.csv")
    
    save_results_json(results, json_path)
    save_results_csv(results, csv_path)
    
    print(f"\nResults saved to:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
