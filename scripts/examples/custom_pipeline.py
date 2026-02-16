#!/usr/bin/env python3
"""
Example: Custom GraphBrew Pipeline
===================================

This script demonstrates how to create custom experiment pipelines using
the lib/phases.py module. It shows the flexibility of the library while
keeping the code simple and readable.

**What This Example Shows:**

1. How to discover graphs in a directory
2. How to configure experiments with PhaseConfig
3. How to run individual phases (reorder, benchmark, cache)
4. How to display and analyze results
5. How to compare specific algorithms

**Running This Example:**

    # Run on all graphs in graphs/ directory
    python scripts/examples/custom_pipeline.py
    
    # Run on a specific graph
    python scripts/examples/custom_pipeline.py --graph graphs/email-Enron
    
    # Quick test mode (fewer algorithms, skip cache)
    python scripts/examples/custom_pipeline.py --quick
    
    # Compare two specific algorithms
    python scripts/examples/custom_pipeline.py --compare 1 8
    
    # Run only specific phases
    python scripts/examples/custom_pipeline.py --phases reorder benchmark

**Output:**

    Results are saved to:
    - results/reorder_<timestamp>.json
    - results/benchmark_<timestamp>.json
    - results/cache_<timestamp>.json (if cache phase is run)
"""

import argparse
import os
import sys
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# Add scripts directory to path for imports
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import (
    # Phase orchestration
    PhaseConfig,
    run_reorder_phase,
    run_benchmark_phase,
    run_cache_phase,
    run_adaptive_analysis_phase,
    compare_algorithms,
    # Progress tracking
    ProgressTracker,
    # Utilities
    ALGORITHMS,
)
from lib.graph_types import GraphInfo


# =============================================================================
# Graph Discovery Helpers
# =============================================================================

def get_graph_dimensions(path: str) -> tuple:
    """
    Read node and edge counts from an MTX file header.
    
    MTX (Matrix Market) files have a header format like:
        %%MatrixMarket matrix coordinate ...
        % comment lines...
        rows cols entries
    
    Args:
        path: Path to MTX file
        
    Returns:
        Tuple of (nodes, edges). Returns (0, 0) if parsing fails.
    """
    try:
        with open(path, 'r') as f:
            for line in f:
                # Skip comment lines starting with %
                if line.startswith('%'):
                    continue
                # First non-comment line has dimensions
                parts = line.strip().split()
                if len(parts) >= 3:
                    return int(parts[0]), int(parts[2])
                elif len(parts) >= 2:
                    return int(parts[0]), 0
                break
    except Exception:
        pass
    return 0, 0


def discover_graphs(graphs_dir: str, max_size_mb: float = None) -> list:
    """
    Discover graph files in a directory structure.
    
    Expected structure:
        graphs_dir/
            graph_name_1/
                graph_name_1.mtx
            graph_name_2/
                graph_name_2.mtx
    
    Args:
        graphs_dir: Root directory containing graph folders
        max_size_mb: Optional size limit in MB
        
    Returns:
        List of GraphInfo objects
    """
    graphs = []
    
    if not os.path.isdir(graphs_dir):
        print(f"Warning: graphs directory not found: {graphs_dir}")
        return graphs
    
    for name in sorted(os.listdir(graphs_dir)):
        graph_path = os.path.join(graphs_dir, name)
        
        # Skip non-directories
        if not os.path.isdir(graph_path):
            continue
        
        # Find graph file (prefer .mtx, then .el, then .sg)
        for ext in ['.mtx', '.el', '.sg']:
            path = os.path.join(graph_path, f"{name}{ext}")
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                
                # Apply size filter if specified
                if max_size_mb and size_mb > max_size_mb:
                    continue
                    
                nodes, edges = get_graph_dimensions(path)
                graphs.append(GraphInfo(
                    name=name,
                    path=path,
                    size_mb=size_mb,
                    nodes=nodes,
                    edges=edges
                ))
                break  # Found a graph file, move to next directory
    
    return graphs


# =============================================================================
# Result Display Helpers
# =============================================================================

def display_benchmark_results(benchmark_results: list, benchmarks: list):
    """
    Display benchmark results as a formatted summary.
    
    Groups results by graph and shows best algorithm for each benchmark.
    """
    # Group results by graph
    by_graph = defaultdict(list)
    for r in benchmark_results:
        if r.time_seconds > 0:  # Only successful runs
            by_graph[r.graph].append(r)
    
    if not by_graph:
        print("\nNo successful benchmark results to display.")
        return
    
    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    
    for graph_name, results in sorted(by_graph.items()):
        print(f"\n  {graph_name}:")
        
        # Find best algorithm for each benchmark
        for bench in benchmarks:
            bench_results = [(r.algorithm, r.time_seconds) 
                           for r in results if r.benchmark == bench]
            if bench_results:
                best_algo, best_time = min(bench_results, key=lambda x: x[1])
                
                # Also show how much faster than ORIGINAL
                original_time = next((t for a, t in bench_results if 'ORIGINAL' in a), None)
                if original_time and original_time != best_time:
                    speedup = original_time / best_time
                    print(f"    {bench:6s}: {best_algo:20s} {best_time:.4f}s ({speedup:.2f}x faster)")
                else:
                    print(f"    {bench:6s}: {best_algo:20s} {best_time:.4f}s")


def display_algorithm_comparison(comparison_results: dict, algo_a: int, algo_b: int):
    """
    Display comparison between two specific algorithms.
    """
    algo_a_name = ALGORITHMS.get(algo_a, f"ALG_{algo_a}")
    algo_b_name = ALGORITHMS.get(algo_b, f"ALG_{algo_b}")
    
    print("\n" + "=" * 70)
    print(f"  ALGORITHM COMPARISON: {algo_a_name} vs {algo_b_name}")
    print("=" * 70)
    
    for graph_name, benchmarks in sorted(comparison_results.items()):
        print(f"\n  {graph_name}:")
        for bench_name, times in benchmarks.items():
            time_a = times.get(algo_a_name, 0)
            time_b = times.get(algo_b_name, 0)
            
            if time_a > 0 and time_b > 0:
                if time_a < time_b:
                    winner = algo_a_name
                    speedup = time_b / time_a
                else:
                    winner = algo_b_name
                    speedup = time_a / time_b
                    
                print(f"    {bench_name}: {algo_a_name}={time_a:.4f}s, {algo_b_name}={time_b:.4f}s "
                      f"→ {winner} wins ({speedup:.2f}x)")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for the custom pipeline example."""
    
    # ─────────────────────────────────────────────────────────────────────────
    # Parse command-line arguments
    # ─────────────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Custom GraphBrew Pipeline Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --quick                    # Quick test with key algorithms
    %(prog)s --graph graphs/email-Enron # Test specific graph
    %(prog)s --compare 1 8              # Compare RANDOM vs RABBITORDER
    %(prog)s --phases reorder benchmark # Run only these phases
        """
    )
    
    parser.add_argument('--graphs-dir', default='results/graphs', 
                        help='Directory containing graph folders')
    parser.add_argument('--graph', 
                        help='Single graph directory to benchmark')
    parser.add_argument('--phases', nargs='+', 
                        choices=['reorder', 'benchmark', 'cache', 'weights', 'adaptive'],
                        default=['reorder', 'benchmark'],
                        help='Phases to run (default: reorder, benchmark)')
    parser.add_argument('--quick', action='store_true', 
                        help='Quick mode: fewer algorithms, skip cache')
    parser.add_argument('--benchmarks', nargs='+', default=['pr', 'bfs'],
                        help='Benchmarks to run')
    parser.add_argument('--compare', type=int, nargs=2, metavar=('ALG_A', 'ALG_B'),
                        help='Compare two specific algorithms')
    parser.add_argument('--max-size', type=float, default=100,
                        help='Maximum graph size in MB')
    parser.add_argument('--trials', type=int, default=3,
                        help='Number of benchmark trials')
    
    args = parser.parse_args()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Initialize progress tracker
    # ─────────────────────────────────────────────────────────────────────────
    progress = ProgressTracker()
    progress.banner("CUSTOM GRAPHBREW PIPELINE", "Using lib/phases.py")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Discover graphs
    # ─────────────────────────────────────────────────────────────────────────
    if args.graph:
        # Single graph mode
        name = os.path.basename(args.graph.rstrip('/'))
        
        # Find graph file
        graph_found = False
        for ext in ['.mtx', '.el', '.sg']:
            path = os.path.join(args.graph, f"{name}{ext}")
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                nodes, edges = get_graph_dimensions(path)
                graphs = [GraphInfo(
                    name=name, 
                    path=path, 
                    size_mb=size_mb, 
                    nodes=nodes, 
                    edges=edges
                )]
                graph_found = True
                break
        
        if not graph_found:
            print(f"Error: No graph file found in {args.graph}")
            return 1
    else:
        # Discover all graphs
        graphs = discover_graphs(args.graphs_dir, max_size_mb=args.max_size)
    
    if not graphs:
        print("Error: No graphs found!")
        return 1
    
    # Display discovered graphs
    progress.info(f"Found {len(graphs)} graph(s):")
    for g in graphs[:10]:
        progress.info(f"  • {g.name}: {g.nodes:,} nodes, {g.edges:,} edges ({g.size_mb:.1f} MB)")
    if len(graphs) > 10:
        progress.info(f"  ... and {len(graphs) - 10} more")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Select algorithms
    # ─────────────────────────────────────────────────────────────────────────
    if args.quick:
        # Quick mode: just key algorithms
        algorithms = [0, 1, 8, 15]  # ORIGINAL, RANDOM, RABBITORDER, LeidenOrder
        progress.info("Quick mode: Testing ORIGINAL, RANDOM, RABBITORDER, LeidenOrder")
    else:
        # Full mode: comprehensive algorithm set
        algorithms = [0, 1, 2, 4, 7, 8, 9, 12, 15]
        progress.info(f"Testing {len(algorithms)} algorithms")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Handle algorithm comparison mode
    # ─────────────────────────────────────────────────────────────────────────
    if args.compare:
        progress.phase_start("ALGORITHM COMPARISON", 
                            f"Comparing {ALGORITHMS.get(args.compare[0])} vs {ALGORITHMS.get(args.compare[1])}")
        
        comparison = compare_algorithms(
            graphs=graphs,
            algorithm_a=args.compare[0],
            algorithm_b=args.compare[1],
            benchmarks=args.benchmarks
        )
        
        display_algorithm_comparison(comparison, args.compare[0], args.compare[1])
        progress.phase_end()
        return 0
    
    # ─────────────────────────────────────────────────────────────────────────
    # Create configuration
    # ─────────────────────────────────────────────────────────────────────────
    config = PhaseConfig(
        benchmarks=args.benchmarks,
        trials=args.trials,
        skip_slow=args.quick,
        skip_cache=args.quick or 'cache' not in args.phases,
        progress=progress
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Run phases
    # ─────────────────────────────────────────────────────────────────────────
    results = {}
    label_maps = {}
    
    # Phase 1: Reordering
    if 'reorder' in args.phases:
        reorder_results, label_maps = run_reorder_phase(
            graphs=graphs,
            algorithms=algorithms,
            config=config
        )
        results['reorder'] = reorder_results
        
        # Summary
        successful = sum(1 for r in reorder_results if r.success)
        progress.success(f"Reordering: {successful}/{len(reorder_results)} successful")
    
    # Phase 2: Benchmarking
    if 'benchmark' in args.phases:
        benchmark_results = run_benchmark_phase(
            graphs=graphs,
            algorithms=algorithms,
            label_maps=label_maps,
            config=config
        )
        results['benchmark'] = benchmark_results
        
        # Display results
        display_benchmark_results(benchmark_results, args.benchmarks)
    
    # Phase 3: Cache Simulation
    if 'cache' in args.phases:
        cache_results = run_cache_phase(
            graphs=graphs,
            algorithms=algorithms,
            label_maps=label_maps,
            config=config
        )
        results['cache'] = cache_results
    
    # Phase 4: Adaptive Analysis
    if 'adaptive' in args.phases:
        adaptive_results = run_adaptive_analysis_phase(
            graphs=graphs,
            config=config
        )
        results['adaptive'] = adaptive_results
    
    # ─────────────────────────────────────────────────────────────────────────
    # Final summary
    # ─────────────────────────────────────────────────────────────────────────
    progress.banner("COMPLETE", f"Processed {len(graphs)} graph(s) through {len(args.phases)} phase(s)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
