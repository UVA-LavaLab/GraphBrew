#!/usr/bin/env python3
"""
Example: Algorithm Comparison
=============================

This script compares different reordering algorithms on a set of graphs
and generates a detailed comparison report.

**Usage:**

    # Compare algorithms on all graphs
    python scripts/examples/compare_algorithms.py
    
    # On specific graphs
    python scripts/examples/compare_algorithms.py --graphs email-Enron web-Stanford
    
    # With specific algorithms
    python scripts/examples/compare_algorithms.py --algorithms 0 1 8 15 17

**Output:**

- Speedup matrix (algorithm vs graph)
- Best algorithm per graph
- Average speedups across all graphs
- Detailed CSV export

**Algorithm IDs:**

    0  = original (baseline)
    1  = random
    2  = sort
    3  = sort-degree
    4  = HubSort
    5  = HubCluster
    6  = DBG
    7  = corder
    8  = gorder
    9  = rorder
    10 = sorder
    11 = morder
    12 = bcorder
    13 = rabbit
    14 = minla
    15 = rcm
    16 = lorder
    17 = GraphBrewOrder
"""

import argparse
import os
import sys
from collections import defaultdict

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.phases import PhaseConfig, run_benchmark_phase
from lib.progress import ProgressTracker
from lib import ALGORITHMS, BENCHMARKS


def find_graphs(graphs_dir: str, graph_names: list = None) -> list:
    """
    Find graphs in the graphs directory.
    
    Args:
        graphs_dir: Path to graphs directory
        graph_names: Optional list of specific graph names
    
    Returns:
        List of (name, path) tuples
    """
    graphs = []
    
    # If specific graphs requested
    if graph_names:
        for name in graph_names:
            # Check various locations
            for ext in ['.mtx', '.el', '.sg']:
                path = os.path.join(graphs_dir, name, f"{name}{ext}")
                if os.path.exists(path):
                    graphs.append((name, path))
                    break
        return graphs
    
    # Otherwise, find all graphs
    for entry in sorted(os.listdir(graphs_dir)):
        graph_dir = os.path.join(graphs_dir, entry)
        if os.path.isdir(graph_dir):
            for ext in ['.mtx', '.el', '.sg']:
                path = os.path.join(graph_dir, f"{entry}{ext}")
                if os.path.exists(path):
                    graphs.append((entry, path))
                    break
    
    return graphs


def compute_speedups(results: list, baseline_algo: int = 0) -> dict:
    """
    Compute speedups relative to baseline algorithm.
    
    Args:
        results: List of benchmark results
        baseline_algo: Algorithm ID to use as baseline (default: 0 = original)
    
    Returns:
        Dictionary mapping (graph, benchmark, algo) -> speedup
    
    Example:
        >>> speedups = compute_speedups(results)
        >>> print(speedups[('email-Enron', 'pr', 8)])  # gorder speedup on PR
        1.45
    """
    # Group by (graph, benchmark)
    groups = defaultdict(dict)
    for r in results:
        key = (r.graph_name, r.benchmark)
        groups[key][r.algorithm_id] = r.avg_time
    
    # Compute speedups
    speedups = {}
    for (graph, bench), times in groups.items():
        baseline = times.get(baseline_algo, 0)
        if baseline <= 0:
            continue
            
        for algo_id, time in times.items():
            if time > 0:
                speedups[(graph, bench, algo_id)] = baseline / time
    
    return speedups


def print_speedup_table(speedups: dict, graphs: list, algorithms: list, benchmark: str):
    """
    Print a speedup table for a specific benchmark.
    
    The table shows speedups for each algorithm on each graph,
    with the best algorithm highlighted.
    
    Args:
        speedups: Speedup dictionary from compute_speedups()
        graphs: List of graph names
        algorithms: List of algorithm IDs
        benchmark: Benchmark name (e.g., 'pr')
    """
    print(f"\n  {benchmark.upper()} Speedup Matrix:")
    print("  " + "-" * (15 + 10 * len(algorithms)))
    
    # Header
    header = f"  {'Graph':15s}"
    for algo in algorithms:
        header += f" {ALGORITHMS.get(algo, str(algo)):>8s}"
    print(header)
    print("  " + "-" * (15 + 10 * len(algorithms)))
    
    # Data rows
    for graph in graphs:
        row = f"  {graph[:15]:15s}"
        best_speedup = 0
        best_algo = None
        
        # Find best
        for algo in algorithms:
            speedup = speedups.get((graph, benchmark, algo), 0)
            if speedup > best_speedup:
                best_speedup = speedup
                best_algo = algo
        
        # Print values
        for algo in algorithms:
            speedup = speedups.get((graph, benchmark, algo), 0)
            if speedup > 0:
                # Highlight best
                if algo == best_algo and algo != 0:
                    row += f" \033[32m{speedup:8.2f}x\033[0m"
                elif speedup < 1.0:
                    row += f" \033[31m{speedup:8.2f}x\033[0m"
                else:
                    row += f" {speedup:8.2f}x"
            else:
                row += f" {'N/A':>8s}"
        
        print(row)
    
    print("  " + "-" * (15 + 10 * len(algorithms)))


def print_best_algorithms(speedups: dict, graphs: list, algorithms: list, benchmark: str):
    """
    Print the best algorithm for each graph on a specific benchmark.
    
    Args:
        speedups: Speedup dictionary
        graphs: List of graph names
        algorithms: List of algorithm IDs
        benchmark: Benchmark name
    """
    print(f"\n  Best Algorithm per Graph ({benchmark.upper()}):")
    
    for graph in graphs:
        best_speedup = 0
        best_algo = 'original'
        
        for algo in algorithms:
            speedup = speedups.get((graph, benchmark, algo), 0)
            if speedup > best_speedup:
                best_speedup = speedup
                best_algo = ALGORITHMS.get(algo, str(algo))
        
        if best_speedup > 1.0:
            print(f"    {graph:20s} → {best_algo:15s} ({best_speedup:.2f}x)")
        else:
            print(f"    {graph:20s} → original (no improvement)")


def main():
    # =========================================================================
    # ARGUMENT PARSING
    # =========================================================================
    parser = argparse.ArgumentParser(
        description="Compare reordering algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Algorithm IDs:
  0=original  1=random  2=sort  3=sort-degree  4=HubSort
  5=HubCluster  6=DBG  7=corder  8=gorder  9=rorder
  10=sorder  11=morder  12=bcorder  13=rabbit  14=minla
  15=rcm  16=lorder  17=GraphBrewOrder
        """
    )
    parser.add_argument('--graphs-dir', '-d', default='results/graphs',
                        help='Graphs directory (default: results/graphs)')
    parser.add_argument('--graphs', '-g', nargs='+',
                        help='Specific graph names to test')
    parser.add_argument('--algorithms', '-a', type=int, nargs='+',
                        default=[0, 1, 8, 12, 15],
                        help='Algorithm IDs (default: 0,1,8,12,15)')
    parser.add_argument('--benchmarks', '-b', nargs='+',
                        default=['pr', 'bfs'],
                        help='Benchmarks (default: pr, bfs)')
    parser.add_argument('--output', '-o',
                        help='Output CSV file')
    
    args = parser.parse_args()
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    progress = ProgressTracker()
    progress.banner("ALGORITHM COMPARISON")
    
    # Find graphs
    graphs = find_graphs(args.graphs_dir, args.graphs)
    if not graphs:
        progress.error(f"No graphs found in {args.graphs_dir}")
        return 1
    
    graph_names = [g[0] for g in graphs]
    
    progress.info(f"Graphs: {len(graphs)}")
    progress.info(f"Algorithms: {[ALGORITHMS.get(a, str(a)) for a in args.algorithms]}")
    progress.info(f"Benchmarks: {args.benchmarks}")
    print()
    
    # =========================================================================
    # RUN BENCHMARKS
    # =========================================================================
    all_results = []
    
    for idx, (graph_name, graph_path) in enumerate(graphs, 1):
        print(f"\n[{idx}/{len(graphs)}] Processing {graph_name}...")
        
        config = PhaseConfig(
            graph_path=graph_path,
            output_dir=f"results/compare/{graph_name}",
            algorithms=args.algorithms,
            benchmarks=args.benchmarks,
            iterations=5,
            warmup=2
        )
        
        try:
            results = run_benchmark_phase(config)
            all_results.extend(results)
        except Exception as e:
            progress.warning(f"Failed {graph_name}: {e}")
            continue
    
    # =========================================================================
    # COMPUTE AND DISPLAY RESULTS
    # =========================================================================
    speedups = compute_speedups(all_results)
    
    print("\n" + "=" * 70)
    print("  COMPARISON RESULTS")
    print("=" * 70)
    
    # Print speedup tables for each benchmark
    for bench in args.benchmarks:
        print_speedup_table(speedups, graph_names, args.algorithms, bench)
        print_best_algorithms(speedups, graph_names, args.algorithms, bench)
    
    # =========================================================================
    # AVERAGE SPEEDUPS
    # =========================================================================
    print("\n  Average Speedup Across All Graphs:")
    print("  " + "-" * 40)
    
    for algo in args.algorithms:
        if algo == 0:
            continue  # Skip baseline
            
        algo_speedups = [v for (g, b, a), v in speedups.items() if a == algo]
        if algo_speedups:
            avg = sum(algo_speedups) / len(algo_speedups)
            bar = "█" * int(avg * 5)
            print(f"    {ALGORITHMS.get(algo, str(algo)):15s} {avg:5.2f}x {bar}")
    
    print("=" * 70)
    
    # =========================================================================
    # EXPORT CSV
    # =========================================================================
    if args.output:
        import csv
        with open(args.output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['graph', 'benchmark', 'algorithm', 'speedup'])
            
            for (graph, bench, algo), speedup in speedups.items():
                writer.writerow([graph, bench, ALGORITHMS.get(algo, str(algo)), f"{speedup:.4f}"])
        
        progress.success(f"Results saved to {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
