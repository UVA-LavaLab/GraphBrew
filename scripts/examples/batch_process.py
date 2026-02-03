#!/usr/bin/env python3
"""
Example: Batch Graph Processing
===============================

This script demonstrates how to process multiple graphs in batch mode.
It's useful for running large-scale experiments across a graph collection.

**Usage:**

    # Process all graphs in a directory
    python scripts/examples/batch_process.py graphs/ --output results/batch
    
    # Only graphs matching a pattern
    python scripts/examples/batch_process.py graphs/ --pattern "web-*"
    
    # With specific phases
    python scripts/examples/batch_process.py graphs/ --phases reorder benchmark

**Features:**

- Automatic graph discovery
- Progress tracking with ETA
- Results saved to JSON/CSV
- Resumable (skips completed graphs)
"""

import argparse
import os
import sys
import json
import glob
from datetime import datetime

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.phases import PhaseConfig, run_reorder_phase, run_benchmark_phase
from lib.progress import ProgressTracker
from lib.graph_types import BenchmarkResult


def discover_graphs(base_dir: str, pattern: str = "*") -> list:
    """
    Find all graph directories matching the pattern.
    
    Args:
        base_dir: Base directory to search (e.g., "graphs/")
        pattern: Glob pattern for graph names (e.g., "web-*")
    
    Returns:
        List of (graph_name, graph_path) tuples
    
    Example:
        >>> graphs = discover_graphs("graphs/", "soc-*")
        >>> print(graphs)
        [('soc-Epinions1', 'graphs/soc-Epinions1/soc-Epinions1.mtx'), ...]
    """
    graphs = []
    
    # Check if pattern has wildcard
    if '*' not in pattern:
        pattern = f"*{pattern}*"
    
    # Find matching directories
    search_path = os.path.join(base_dir, pattern)
    for graph_dir in sorted(glob.glob(search_path)):
        if not os.path.isdir(graph_dir):
            continue
            
        graph_name = os.path.basename(graph_dir)
        
        # Look for graph file
        for ext in ['.mtx', '.el', '.sg']:
            graph_file = os.path.join(graph_dir, f"{graph_name}{ext}")
            if os.path.exists(graph_file):
                graphs.append((graph_name, graph_file))
                break
    
    return graphs


def load_completed(output_dir: str) -> set:
    """
    Load list of already-completed graphs from the output directory.
    This enables resuming interrupted batch processing.
    
    Args:
        output_dir: Output directory containing results
    
    Returns:
        Set of completed graph names
    """
    completed = set()
    results_file = os.path.join(output_dir, "results.json")
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            data = json.load(f)
            completed = set(data.get('completed_graphs', []))
    
    return completed


def save_progress(output_dir: str, completed: set, results: list):
    """
    Save current progress to disk.
    
    Args:
        output_dir: Output directory
        completed: Set of completed graph names
        results: List of result dictionaries
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'completed_graphs': list(completed),
            'results': results
        }, f, indent=2)


def main():
    # =========================================================================
    # ARGUMENT PARSING
    # =========================================================================
    parser = argparse.ArgumentParser(
        description="Batch process multiple graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('input_dir', help='Directory containing graphs')
    parser.add_argument('--output', '-o', default='results/batch',
                        help='Output directory (default: results/batch)')
    parser.add_argument('--pattern', '-p', default='*',
                        help='Graph name pattern (default: *)')
    parser.add_argument('--phases', nargs='+', default=['reorder', 'benchmark'],
                        choices=['reorder', 'benchmark', 'cache'],
                        help='Phases to run')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from previous run')
    parser.add_argument('--algorithms', '-a', type=int, nargs='+',
                        default=[0, 1, 8],
                        help='Algorithm IDs (default: 0, 1, 8)')
    parser.add_argument('--benchmarks', '-b', nargs='+',
                        default=['pr', 'bfs'],
                        help='Benchmarks (default: pr, bfs)')
    
    args = parser.parse_args()
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    progress = ProgressTracker()
    progress.banner("BATCH GRAPH PROCESSING")
    
    # Discover graphs
    graphs = discover_graphs(args.input_dir, args.pattern)
    if not graphs:
        progress.error(f"No graphs found in {args.input_dir} matching '{args.pattern}'")
        return 1
    
    progress.info(f"Found {len(graphs)} graphs")
    
    # Load completed graphs if resuming
    completed = set()
    if args.resume:
        completed = load_completed(args.output)
        if completed:
            progress.info(f"Resuming: {len(completed)} graphs already completed")
    
    # Filter out completed
    pending = [(name, path) for name, path in graphs if name not in completed]
    progress.info(f"Processing: {len(pending)} graphs")
    print()
    
    # =========================================================================
    # BATCH PROCESSING
    # =========================================================================
    all_results = []
    
    for idx, (graph_name, graph_path) in enumerate(pending, 1):
        # Print progress header
        print(f"\n{'=' * 70}")
        print(f"  Graph {idx}/{len(pending)}: {graph_name}")
        print(f"{'=' * 70}")
        
        # Create config for this graph
        config = PhaseConfig(
            graph_path=graph_path,
            output_dir=os.path.join(args.output, graph_name),
            algorithms=args.algorithms,
            benchmarks=args.benchmarks,
            iterations=3,  # Less iterations for batch mode
            warmup=1
        )
        
        try:
            # Run reorder phase
            if 'reorder' in args.phases:
                progress.phase("Reordering")
                run_reorder_phase(config)
            
            # Run benchmark phase
            if 'benchmark' in args.phases:
                progress.phase("Benchmarking")
                results = run_benchmark_phase(config)
                
                # Collect results
                for r in results:
                    all_results.append({
                        'graph': graph_name,
                        'algorithm': r.algorithm_name,
                        'benchmark': r.benchmark,
                        'avg_time': r.avg_time,
                        'stddev': r.stddev
                    })
            
            # Mark as completed
            completed.add(graph_name)
            
            # Save progress after each graph
            save_progress(args.output, completed, all_results)
            
            progress.success(f"Completed {graph_name}")
            
        except Exception as e:
            progress.error(f"Failed {graph_name}: {e}")
            continue
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("  BATCH PROCESSING COMPLETE")
    print(f"{'=' * 70}")
    progress.stats_summary("Results", {
        'Total Graphs': len(graphs),
        'Completed': len(completed),
        'Results': len(all_results),
        'Output': args.output
    })
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
