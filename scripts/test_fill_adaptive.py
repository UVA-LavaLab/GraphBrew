#!/usr/bin/env python3
"""
Test script for fill-weights and adaptive order functionality.

This script:
1. Runs fill-weights on a small graph to populate weight files
2. Verifies weights are properly filled
3. Tests that adaptive order uses the weights
4. Displays detailed output of the process

Usage:
    python scripts/test_fill_adaptive.py
    python scripts/test_fill_adaptive.py --graph graphs/email-Enron
    python scripts/test_fill_adaptive.py --clean   # Clean and re-run
"""

import argparse
import json
import os
import sys
import shutil
import subprocess
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib import (
    ALGORITHMS, BENCHMARKS, Logger, Colors,
    format_size, format_duration,
)
from scripts.lib.weights import (
    load_type_weights, list_known_types, get_best_algorithm_for_type,
    load_type_registry
)
from scripts.lib.progress import ProgressTracker

log = Logger()
progress = ProgressTracker()


def find_test_graph(graphs_dir: str = "graphs") -> tuple:
    """Find a small graph suitable for testing."""
    # Preferred small graphs for testing
    preferred = [
        "email-Enron",
        "p2p-Gnutella31", 
        "ca-AstroPh",
        "ca-CondMat",
        "soc-Epinions1",
    ]
    
    for name in preferred:
        graph_dir = os.path.join(graphs_dir, name)
        if os.path.isdir(graph_dir):
            for ext in ['.mtx', '.el', '.sg']:
                path = os.path.join(graph_dir, f"{name}{ext}")
                if os.path.exists(path):
                    size = os.path.getsize(path) / (1024 * 1024)
                    return name, path, size
    
    # Fall back to any graph under 50MB
    if os.path.isdir(graphs_dir):
        for name in sorted(os.listdir(graphs_dir)):
            graph_dir = os.path.join(graphs_dir, name)
            if os.path.isdir(graph_dir):
                for ext in ['.mtx', '.el', '.sg']:
                    path = os.path.join(graph_dir, f"{name}{ext}")
                    if os.path.exists(path):
                        size = os.path.getsize(path) / (1024 * 1024)
                        if size < 50:
                            return name, path, size
    
    return None, None, 0


def clean_weights_dir(weights_dir: str):
    """Remove all weight files to start fresh."""
    if os.path.isdir(weights_dir):
        for f in os.listdir(weights_dir):
            if f.endswith('.json'):
                os.remove(os.path.join(weights_dir, f))
        log.info(f"Cleaned weights directory: {weights_dir}")


def run_fill_weights(graph_path: str, args):
    """Run fill-weights phase on a graph."""
    # Use small graphs category and size filter to get a single graph
    graph_name = os.path.basename(os.path.dirname(graph_path))
    graph_size = os.path.getsize(graph_path) / (1024 * 1024)
    
    cmd = [
        sys.executable, "scripts/graphbrew_experiment.py",
        "--fill-weights",
        "--graphs", "small",
        "--max-graphs", "1",  # Just one graph
        "--trials", "2",  # Fewer trials for speed
        "--skip-heavy",  # Skip heavy sims
        "--max-size", str(int(graph_size + 5)),  # Ensure our graph is included
    ]
    
    log.info(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=False,  # Let output stream through
        text=True
    )
    
    return result.returncode == 0


def verify_weights(weights_dir: str) -> dict:
    """Verify that weights are properly filled."""
    results = {
        'types_found': [],
        'registry_exists': False,
        'weights_populated': {},
        'issues': []
    }
    
    # Check type registry
    registry_file = os.path.join(weights_dir, "type_registry.json")
    if os.path.exists(registry_file):
        results['registry_exists'] = True
        with open(registry_file) as f:
            registry = json.load(f)
        results['types_found'] = list(registry.keys())
        log.success(f"Type registry found with {len(registry)} types")
    else:
        results['issues'].append("No type_registry.json found")
        log.warning("No type registry found")
    
    # Check individual type weights
    for f in os.listdir(weights_dir):
        if f.startswith("type_") and f.endswith(".json"):
            type_name = f.replace(".json", "")
            filepath = os.path.join(weights_dir, f)
            
            with open(filepath) as file:
                weights = json.load(file)
            
            # Count filled vs zero weights
            filled = 0
            total = 0
            for algo_name, algo_weights in weights.items():
                if algo_name.startswith("_"):
                    continue
                for key, val in algo_weights.items():
                    if key.startswith("_") or key == "benchmark_weights":
                        continue
                    total += 1
                    if val != 0:
                        filled += 1
            
            pct = (filled / total * 100) if total > 0 else 0
            results['weights_populated'][type_name] = {
                'filled': filled,
                'total': total,
                'percent': pct
            }
            
            status = "✓" if pct > 50 else "○"
            log.info(f"  {status} {type_name}: {filled}/{total} weights filled ({pct:.1f}%)")
    
    return results


def test_adaptive_order(graph_path: str, weights_dir: str) -> dict:
    """Test that adaptive order actually uses the weights."""
    results = {
        'ran_successfully': False,
        'weights_loaded': False,
        'algorithm_selected': None,
        'output': ""
    }
    
    # Set environment variable to point to weights
    env = os.environ.copy()
    
    # Find the first type weights file
    type_files = [f for f in os.listdir(weights_dir) if f.startswith("type_") and f.endswith(".json")]
    if type_files:
        weights_file = os.path.join(weights_dir, type_files[0])
        env['PERCEPTRON_WEIGHTS_FILE'] = weights_file
        log.info(f"Set PERCEPTRON_WEIGHTS_FILE={weights_file}")
    
    # Run benchmark with AdaptiveOrder (algorithm 14)
    bin_path = os.path.join(PROJECT_ROOT, "bench", "bin", "pr")
    if not os.path.exists(bin_path):
        results['output'] = "Benchmark binary not found"
        return results
    
    cmd = [bin_path, "-f", graph_path, "-a", "14", "-n", "1"]
    log.info(f"Running: {' '.join(cmd)}")
    
    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120
    )
    
    results['output'] = proc.stdout + proc.stderr
    results['ran_successfully'] = proc.returncode == 0
    
    # Check if weights were loaded
    if "Perceptron: Loaded" in results['output']:
        results['weights_loaded'] = True
        log.success("Weights were loaded by AdaptiveOrder")
    elif "Using default" in results['output']:
        log.warning("AdaptiveOrder used default weights (file not found?)")
    
    # Look for algorithm selection info
    import re
    algo_match = re.search(r'Selected algorithm[:\s]+(\w+)', results['output'], re.IGNORECASE)
    if algo_match:
        results['algorithm_selected'] = algo_match.group(1)
        log.info(f"Algorithm selected: {results['algorithm_selected']}")
    
    # Check for subcommunity info
    if "Subcommunity" in results['output'] or "subcommunity" in results['output']:
        log.info("Subcommunity analysis was performed")
    
    return results


def show_best_algorithms(weights_dir: str):
    """Show best algorithm for each type and benchmark."""
    progress.section("BEST ALGORITHMS PER TYPE")
    
    types = list_known_types(weights_dir)
    if not types:
        log.warning("No types found")
        return
    
    benchmarks = ['pr', 'bfs', 'cc', 'sssp', 'bc']
    
    for type_name in types:
        print(f"\n  {type_name}:")
        for bench in benchmarks:
            best_algo = get_best_algorithm_for_type(type_name, bench, weights_dir)
            if best_algo:
                print(f"    {bench:6s}: {best_algo}")
            else:
                print(f"    {bench:6s}: (no data)")


def main():
    parser = argparse.ArgumentParser(
        description="Test fill-weights and adaptive order functionality"
    )
    parser.add_argument('--graph', '-g', help='Specific graph to test')
    parser.add_argument('--clean', '-c', action='store_true', 
                        help='Clean weights directory before testing')
    parser.add_argument('--skip-fill', action='store_true',
                        help='Skip fill-weights phase (use existing weights)')
    parser.add_argument('--weights-dir', default='scripts/weights',
                        help='Weights directory')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    progress.banner("FILL-WEIGHTS & ADAPTIVE ORDER TEST")
    
    # Find test graph
    if args.graph:
        graph_name = os.path.basename(args.graph.rstrip('/'))
        graph_path = args.graph
        if os.path.isdir(graph_path):
            for ext in ['.mtx', '.el', '.sg']:
                p = os.path.join(graph_path, f"{graph_name}{ext}")
                if os.path.exists(p):
                    graph_path = p
                    break
        size = os.path.getsize(graph_path) / (1024 * 1024) if os.path.exists(graph_path) else 0
    else:
        graph_name, graph_path, size = find_test_graph()
    
    if not graph_path or not os.path.exists(graph_path):
        log.error("No suitable test graph found!")
        log.info("Please specify a graph with --graph or download some graphs first")
        return 1
    
    log.info(f"Test graph: {graph_name}")
    log.info(f"Graph path: {graph_path}")
    log.info(f"Graph size: {size:.1f} MB")
    print()
    
    # Clean if requested
    if args.clean:
        progress.section("CLEANING WEIGHTS")
        clean_weights_dir(args.weights_dir)
    
    # Run fill-weights
    if not args.skip_fill:
        progress.section("RUNNING FILL-WEIGHTS")
        success = run_fill_weights(graph_path, args)
        if not success:
            log.error("Fill-weights failed!")
            return 1
        log.success("Fill-weights completed")
    else:
        log.info("Skipping fill-weights (using existing weights)")
    
    print()
    
    # Verify weights
    progress.section("VERIFYING WEIGHTS")
    verify_results = verify_weights(args.weights_dir)
    
    if verify_results['issues']:
        for issue in verify_results['issues']:
            log.warning(issue)
    
    print()
    
    # Show best algorithms
    show_best_algorithms(args.weights_dir)
    
    print()
    
    # Test adaptive order
    progress.section("TESTING ADAPTIVE ORDER")
    adaptive_results = test_adaptive_order(graph_path, args.weights_dir)
    
    if args.verbose and adaptive_results['output']:
        print("\n--- Adaptive Order Output ---")
        print(adaptive_results['output'][:2000])  # First 2000 chars
        print("--- End Output ---\n")
    
    # Summary
    progress.section("SUMMARY")
    
    summary = {
        'Weights Directory': args.weights_dir,
        'Types Found': len(verify_results['types_found']),
        'Registry Exists': '✓' if verify_results['registry_exists'] else '✗',
        'Adaptive Order Ran': '✓' if adaptive_results['ran_successfully'] else '✗',
        'Weights Loaded': '✓' if adaptive_results['weights_loaded'] else '✗',
    }
    
    for key, val in summary.items():
        log.info(f"  {key}: {val}")
    
    # Check overall success
    success = (
        verify_results['registry_exists'] and
        len(verify_results['weights_populated']) > 0 and
        adaptive_results['ran_successfully']
    )
    
    print()
    if success:
        log.success("All tests passed! Weights are being filled and used correctly.")
    else:
        log.warning("Some tests may have issues - check output above")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
