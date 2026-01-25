#!/usr/bin/env python3
"""
Example: Quick Graph Benchmark
==============================

A simple script that benchmarks a single graph with common algorithms.
This is the easiest way to test if GraphBrew is working correctly.

**Usage:**

    # Benchmark a single graph
    python scripts/examples/quick_test.py graphs/email-Enron
    
    # With specific benchmarks
    python scripts/examples/quick_test.py graphs/email-Enron --benchmarks pr bfs cc
    
    # Just PageRank
    python scripts/examples/quick_test.py graphs/email-Enron --benchmarks pr

**What It Does:**

1. Loads the specified graph
2. Runs reordering with key algorithms
3. Benchmarks each reordering
4. Prints results as a table

This is useful for:
- Verifying installation
- Quick performance tests
- Comparing algorithms on a new graph
"""

import argparse
import os
import sys

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.phases import quick_benchmark
from lib.progress import ProgressTracker
from lib import ALGORITHMS


def main():
    parser = argparse.ArgumentParser(
        description="Quick benchmark a single graph",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('graph', help='Path to graph directory (e.g., graphs/email-Enron)')
    parser.add_argument('--benchmarks', '-b', nargs='+', default=['pr', 'bfs'],
                        help='Benchmarks to run (default: pr, bfs)')
    parser.add_argument('--algorithms', '-a', type=int, nargs='+',
                        default=[0, 1, 8, 15],
                        help='Algorithm IDs (default: 0,1,8,15)')
    
    args = parser.parse_args()
    
    # Initialize progress
    progress = ProgressTracker()
    progress.banner("QUICK GRAPH BENCHMARK")
    
    # Find graph file
    graph_dir = args.graph.rstrip('/')
    graph_name = os.path.basename(graph_dir)
    
    graph_file = None
    for ext in ['.mtx', '.el', '.sg']:
        path = os.path.join(graph_dir, f"{graph_name}{ext}")
        if os.path.exists(path):
            graph_file = path
            break
    
    if not graph_file:
        print(f"Error: No graph file found in {graph_dir}")
        return 1
    
    progress.info(f"Graph: {graph_name}")
    progress.info(f"File: {graph_file}")
    progress.info(f"Algorithms: {[ALGORITHMS.get(a, str(a)) for a in args.algorithms]}")
    progress.info(f"Benchmarks: {args.benchmarks}")
    print()
    
    # Run quick benchmark
    results = quick_benchmark(
        graph_path=graph_file,
        algorithms=args.algorithms,
        benchmarks=args.benchmarks
    )
    
    # Display results
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    
    # Group by benchmark
    for bench in args.benchmarks:
        bench_results = [(r.algorithm_name, r.avg_time) 
                        for r in results if r.benchmark == bench and r.avg_time > 0]
        if bench_results:
            print(f"\n  {bench.upper()}:")
            
            # Sort by time
            bench_results.sort(key=lambda x: x[1])
            baseline = bench_results[0][1]  # Best time
            
            for algo, time in bench_results:
                speedup = baseline / time if time > 0 else 0
                bar = "█" * int(speedup * 10)
                print(f"    {algo:20s} {time:8.4f}s  {bar}")
    
    print("\n" + "=" * 70)
    progress.success("Benchmark complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
