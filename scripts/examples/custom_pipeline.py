#!/usr/bin/env python3
"""
Example: Custom GraphBrew Pipeline

This script demonstrates how to create custom experiment pipelines using
the lib/phases.py module. You can select specific phases to run and
customize the configuration.

Usage:
    python scripts/examples/custom_pipeline.py --graphs-dir graphs --phases reorder benchmark
    python scripts/examples/custom_pipeline.py --graph graphs/web-Stanford --quick
"""

import argparse
import os
import sys

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import (
    # Phase orchestration
    PhaseConfig,
    run_reorder_phase,
    run_benchmark_phase,
    run_cache_phase,
    run_weights_phase,
    run_adaptive_analysis_phase,
    run_full_pipeline,
    quick_benchmark,
    compare_algorithms,
    # Progress tracking
    ProgressTracker,
    # Utilities
    ALGORITHMS,
)
from lib.types import GraphInfo


def get_graph_dimensions(path: str) -> tuple:
    """Read nodes and edges count from an MTX file header."""
    try:
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('%'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 3:
                    return int(parts[0]), int(parts[2])
                elif len(parts) >= 2:
                    return int(parts[0]), 0
                break
    except:
        pass
    return 0, 0


def discover_graphs_simple(graphs_dir: str) -> list:
    """Simple graph discovery function."""
    graphs = []
    if not os.path.isdir(graphs_dir):
        return graphs
    
    for name in sorted(os.listdir(graphs_dir)):
        graph_path = os.path.join(graphs_dir, name)
        if not os.path.isdir(graph_path):
            continue
        
        # Find graph file
        for ext in ['.mtx', '.el', '.sg']:
            path = os.path.join(graph_path, f"{name}{ext}")
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                nodes, edges = get_graph_dimensions(path)
                graphs.append(GraphInfo(
                    name=name,
                    path=path,
                    size_mb=size_mb,
                    nodes=nodes,
                    edges=edges
                ))
                break
    
    return graphs


def main():
    parser = argparse.ArgumentParser(description="Custom GraphBrew Pipeline Example")
    parser.add_argument('--graphs-dir', default='graphs', help='Directory containing graphs')
    parser.add_argument('--graph', help='Single graph to benchmark (path to directory)')
    parser.add_argument('--phases', nargs='+', 
                        choices=['reorder', 'benchmark', 'cache', 'weights', 'adaptive'],
                        default=['reorder', 'benchmark'],
                        help='Phases to run')
    parser.add_argument('--quick', action='store_true', 
                        help='Quick mode: only test key algorithms')
    parser.add_argument('--benchmarks', nargs='+', default=['pr', 'bfs'],
                        help='Benchmarks to run')
    parser.add_argument('--compare', type=int, nargs=2, metavar=('ALG_A', 'ALG_B'),
                        help='Compare two specific algorithms')
    args = parser.parse_args()
    
    # Initialize progress tracker
    progress = ProgressTracker()
    progress.banner("CUSTOM GRAPHBREW PIPELINE", "Example using lib/phases.py")
    
    # Discover graphs
    if args.graph:
        # Single graph mode
        name = os.path.basename(args.graph.rstrip('/'))
        for ext in ['.mtx', '.el', '.sg']:
            path = os.path.join(args.graph, f"{name}{ext}")
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                nodes, edges = get_graph_dimensions(path)
                graphs = [GraphInfo(name=name, path=path, size_mb=size_mb, nodes=nodes, edges=edges)]
                break
        else:
            print(f"No graph file found in {args.graph}")
            return
    else:
        graphs = discover_graphs_simple(args.graphs_dir)
    
    if not graphs:
        print("No graphs found!")
        return
    
    progress.info(f"Found {len(graphs)} graphs")
    for g in graphs:
        progress.info(f"  - {g.name}: {g.nodes:,} nodes, {g.edges:,} edges")
    
    # Select algorithms
    if args.quick:
        algorithms = [0, 1, 7, 15, 16]  # ORIGINAL, RANDOM, HUBCLUSTERDBG, LeidenOrder, LeidenDendrogram
    else:
        algorithms = [0, 1, 2, 4, 7, 8, 9, 15, 16, 17]
    
    progress.info(f"Algorithms: {[ALGORITHMS.get(a, f'ALG_{a}') for a in algorithms]}")
    
    # Comparison mode
    if args.compare:
        progress.phase_start("ALGORITHM COMPARISON", "Comparing two algorithms")
        results = compare_algorithms(
            graphs=graphs,
            algorithm_a=args.compare[0],
            algorithm_b=args.compare[1],
            benchmarks=args.benchmarks
        )
        
        # Print results
        for graph_name, benchmarks in results.items():
            print(f"\n{graph_name}:")
            for bench, times in benchmarks.items():
                print(f"  {bench}:")
                for algo, time in times.items():
                    print(f"    {algo}: {time:.4f}s")
        return
    
    # Create configuration
    config = PhaseConfig(
        benchmarks=args.benchmarks,
        trials=3,
        skip_slow=args.quick,
        progress=progress
    )
    
    # Run phases
    results = {}
    label_maps = {}
    
    if 'reorder' in args.phases:
        reorder_results, label_maps = run_reorder_phase(
            graphs=graphs,
            algorithms=algorithms,
            config=config
        )
        results['reorder'] = reorder_results
        progress.success(f"Reordering complete: {len(reorder_results)} results")
    
    if 'benchmark' in args.phases:
        benchmark_results = run_benchmark_phase(
            graphs=graphs,
            algorithms=algorithms,
            label_maps=label_maps,
            config=config
        )
        results['benchmark'] = benchmark_results
        
        # Show best algorithms per graph
        progress.phase_start("RESULTS SUMMARY", "Best algorithms per graph")
        from collections import defaultdict
        by_graph = defaultdict(list)
        for r in benchmark_results:
            if r.avg_time > 0:
                by_graph[r.graph].append((r.algorithm_name, r.benchmark, r.avg_time))
        
        for graph, runs in by_graph.items():
            print(f"\n{graph}:")
            for bench in args.benchmarks:
                bench_runs = [(a, t) for a, b, t in runs if b == bench]
                if bench_runs:
                    best = min(bench_runs, key=lambda x: x[1])
                    print(f"  {bench}: {best[0]} ({best[1]:.4f}s)")
        progress.phase_end()
    
    if 'cache' in args.phases:
        cache_results = run_cache_phase(
            graphs=graphs,
            algorithms=algorithms,
            label_maps=label_maps,
            config=config
        )
        results['cache'] = cache_results
    
    if 'adaptive' in args.phases:
        adaptive_results = run_adaptive_analysis_phase(
            graphs=graphs,
            config=config
        )
        results['adaptive'] = adaptive_results
    
    progress.banner("COMPLETE", f"Processed {len(graphs)} graphs through {len(args.phases)} phases")


if __name__ == "__main__":
    main()
