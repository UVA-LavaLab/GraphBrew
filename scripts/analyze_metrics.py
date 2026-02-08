#!/usr/bin/env python3
"""
Analyze amortization and ordering quality from existing benchmark results.

Computes derived metrics without running new benchmarks:
  - Break-even iterations for each (graph, algorithm, benchmark)
  - End-to-end speedup at 1, 10, 100 iterations
  - Head-to-head variant comparison with crossover points

Usage:
    # Amortization report from latest benchmark results
    python3 scripts/analyze_metrics.py --results-dir results/

    # Head-to-head: rabbit vs vibe:hrab
    python3 scripts/analyze_metrics.py --results-dir results/ \\
        --compare RABBITORDER_csr LeidenCSR_vibe:hrab

    # Filter to specific benchmarks/graphs
    python3 scripts/analyze_metrics.py --results-dir results/ \\
        --benchmarks pr bfs --graphs web-Google soc-Slashdot0902
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add scripts/ to path
sys.path.insert(0, str(Path(__file__).parent))

from lib.metrics import (
    compute_amortization,
    compute_variant_comparison,
    format_amortization_table,
    format_comparison_table,
)


def load_latest_results(results_dir: str, prefix: str) -> list:
    """Load most recent results JSON file with given prefix."""
    candidates = sorted(
        [f for f in os.listdir(results_dir) if f.startswith(prefix) and f.endswith('.json')],
        reverse=True
    )
    if not candidates:
        return []
    path = os.path.join(results_dir, candidates[0])
    print(f"Loading: {path}")
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Analyze amortization and ordering quality metrics")
    parser.add_argument('--results-dir', default='results/', help='Directory with result JSON files')
    parser.add_argument('--benchmark-file', help='Specific benchmark JSON file')
    parser.add_argument('--reorder-file', help='Specific reorder JSON file')
    parser.add_argument('--compare', nargs=2, metavar=('ALGO_A', 'ALGO_B'),
                       help='Head-to-head comparison of two algorithms')
    parser.add_argument('--benchmarks', nargs='+', help='Filter to specific benchmarks')
    parser.add_argument('--graphs', nargs='+', help='Filter to specific graphs')
    parser.add_argument('--reference', default='ORIGINAL', help='Reference algorithm (default: ORIGINAL)')
    parser.add_argument('--max-rows', type=int, default=100, help='Max table rows')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    args = parser.parse_args()
    
    # Load data
    if args.benchmark_file:
        with open(args.benchmark_file) as f:
            bench_data = json.load(f)
    else:
        bench_data = load_latest_results(args.results_dir, 'benchmark_')
    
    if args.reorder_file:
        with open(args.reorder_file) as f:
            reorder_data = json.load(f)
    else:
        reorder_data = load_latest_results(args.results_dir, 'reorder_')
    
    if not bench_data:
        print("No benchmark results found. Run graphbrew_experiment.py first.")
        sys.exit(1)
    
    # Apply filters
    if args.benchmarks:
        bench_data = [r for r in bench_data if r.get('benchmark') in args.benchmarks]
    if args.graphs:
        bench_data = [r for r in bench_data if r.get('graph') in args.graphs]
        reorder_data = [r for r in reorder_data if r.get('graph') in args.graphs]
    
    print(f"Loaded {len(bench_data)} benchmark results, {len(reorder_data)} reorder results\n")
    
    # Compute amortization
    report = compute_amortization(bench_data, reorder_data, reference_algo=args.reference)
    
    if args.json:
        import dataclasses
        print(json.dumps([dataclasses.asdict(e) for e in report.entries], indent=2, default=str))
    else:
        print(format_amortization_table(report, max_rows=args.max_rows))
    
    # Head-to-head comparison
    if args.compare:
        va, vb = args.compare
        comps = compute_variant_comparison(bench_data, reorder_data, va, vb)
        if comps:
            print()
            print(format_comparison_table(comps))
        else:
            print(f"\nNo overlapping (graph, benchmark) pairs for {va} vs {vb}")


if __name__ == '__main__':
    main()
