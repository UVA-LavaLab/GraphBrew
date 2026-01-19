#!/usr/bin/env python3
"""
Full Correlation Scan with Sequential Execution

Runs comprehensive benchmarks across all graphs and reordering algorithms.
Each benchmark runs sequentially with FULL thread allocation (no parallel interference).

Key features:
- Sequential execution (one benchmark at a time)
- Uses RANDOM (1) as baseline for speedup calculation
- Tests all algorithms 0-20 (except 13, 14 which need special params)
- Skips TC (triangle counting) as reordering doesn't help
- Saves progress incrementally to allow resumption
- Full cache simulation integration

Usage:
    python3 scripts/analysis/full_correlation_scan.py --quick     # Small test
    python3 scripts/analysis/full_correlation_scan.py --small     # SMALL graphs only
    python3 scripts/analysis/full_correlation_scan.py --medium    # SMALL + MEDIUM graphs
    python3 scripts/analysis/full_correlation_scan.py --full      # All graphs
"""

import os
import sys
import argparse
import json
import time
import math
import subprocess
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

# ============================================================================
# Configuration
# ============================================================================

# All reordering algorithms (skip 13=GraphBrew, 14=MAP which need special params)
ALL_ALGORITHMS = {
    0: "ORIGINAL",
    1: "RANDOM",
    2: "SORT",
    3: "HUBSORT",
    4: "HUBCLUSTER",
    5: "DBG",
    6: "HUBSORTDBG",
    7: "HUBCLUSTERDBG",
    8: "RABBITORDER",
    9: "GORDER",
    10: "CORDER",
    11: "RCM",
    12: "LeidenOrder",
    15: "AdaptiveOrder",
    16: "LeidenDFS",
    17: "LeidenDFSHub",
    18: "LeidenDFSDepth",
    19: "LeidenDFSBFS",
    20: "LeidenHybrid",
}

# Baseline algorithm for speedup calculation
BASELINE_ALGO = 1  # RANDOM

# Benchmarks to run (skip tc - reordering doesn't help)
BENCHMARKS = ["pr", "bfs", "cc", "sssp", "bc"]

# Quick test configuration
QUICK_ALGORITHMS = {0: "ORIGINAL", 1: "RANDOM", 7: "HUBCLUSTERDBG", 12: "LeidenOrder", 20: "LeidenHybrid"}

# Synthetic graphs for quick testing
SYNTHETIC_GRAPHS = {
    "rmat_10": "-g 10",
    "rmat_12": "-g 12",
}

# Graph size categories (estimated based on typical sizes)
SMALL_GRAPH_LIMIT = 100 * 1024 * 1024  # 100MB
MEDIUM_GRAPH_LIMIT = 1024 * 1024 * 1024  # 1GB

# ============================================================================
# Helper Functions
# ============================================================================

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print a formatted header."""
    width = 70
    print()
    print("=" * width)
    print(f"{text:^{width}}")
    print("=" * width)


def print_subheader(text: str):
    """Print a formatted subheader."""
    width = 70
    print()
    print("-" * width)
    print(text)
    print("-" * width)


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 0.001:
        return f"{seconds*1e6:.1f}µs"
    elif seconds < 1:
        return f"{seconds*1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.3f}s"
    else:
        return f"{seconds/60:.1f}min"


def format_eta(seconds: float) -> str:
    """Format ETA in human-readable format."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


# ============================================================================
# Benchmark Execution
# ============================================================================

def run_single_benchmark(
    binary: str,
    graph_args: str,
    algo_id: int,
    num_trials: int = 3,
    timeout: int = 600
) -> Tuple[Optional[Dict], str]:
    """
    Run a SINGLE benchmark with full thread allocation.
    
    This function blocks until the benchmark completes.
    No other benchmarks should run concurrently.
    """
    cmd = f"{binary} {graph_args} -o {algo_id} -n {num_trials}"
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        output = result.stdout + result.stderr
        parsed = parse_output(output)
        return parsed, output
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    except Exception as e:
        return None, str(e)


def parse_output(output: str) -> Dict[str, Any]:
    """Parse benchmark output into structured data."""
    result = {}
    
    # Graph statistics
    graph_match = re.search(r'Graph has (\d+) nodes and (\d+)', output)
    if graph_match:
        result['nodes'] = int(graph_match.group(1))
        result['edges'] = int(graph_match.group(2))
    
    # Timing
    patterns = {
        'reorder_time': r'Reorder Time:\s+([\d.]+)',
        'relabel_time': r'Relabel.*Time:\s+([\d.]+)',
        'trial_time': r'Trial Time:\s+([\d.]+)',
        'average_time': r'Average Time:\s+([\d.]+)',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            result[key] = float(match.group(1))
    
    # PageRank iterations
    iter_match = re.search(r'(\d+)\s+iterations', output)
    if iter_match:
        result['iterations'] = int(iter_match.group(1))
    
    # Modularity
    mod_match = re.search(r'[Mm]odularity[:\s]+([\d.]+)', output)
    if mod_match:
        result['modularity'] = float(mod_match.group(1))
    
    return result


# ============================================================================
# Graph Discovery
# ============================================================================

def discover_graphs(graphs_dir: str, size_limit: int = None) -> List[Tuple[str, str, int]]:
    """
    Discover graph files in a directory.
    
    Returns list of (name, path, size_bytes) tuples, sorted by size.
    """
    graphs = []
    graphs_path = Path(graphs_dir)
    
    if not graphs_path.exists():
        return graphs
    
    for subdir in sorted(graphs_path.iterdir()):
        if subdir.is_dir():
            # Look for graph files
            for ext in ['.el', '.wel', '.gr', '.mtx', '.sg']:
                candidates = list(subdir.glob(f'*{ext}'))
                if candidates:
                    graph_file = candidates[0]
                    size = graph_file.stat().st_size
                    
                    # Apply size filter
                    if size_limit and size > size_limit:
                        continue
                    
                    graphs.append((subdir.name, str(graph_file), size))
                    break
    
    # Sort by size (smallest first)
    graphs.sort(key=lambda x: x[2])
    return graphs


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.1f}KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes/(1024*1024):.1f}MB"
    else:
        return f"{size_bytes/(1024*1024*1024):.2f}GB"


# ============================================================================
# Main Scan Logic
# ============================================================================

@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    graph_name: str
    benchmark: str
    algo_id: int
    algo_name: str
    average_time: float
    reorder_time: float
    total_time: float
    nodes: int = 0
    edges: int = 0
    iterations: int = 0
    modularity: float = 0.0


def run_full_scan(
    graphs: List[Tuple[str, str, int]],
    algorithms: Dict[int, str],
    benchmarks: List[str],
    num_trials: int = 3,
    timeout: int = 600,
    output_dir: str = "./results",
    resume: bool = True
) -> List[BenchmarkResult]:
    """
    Run full correlation scan with SEQUENTIAL execution.
    
    Each benchmark runs one at a time with full thread allocation.
    Progress is saved after each benchmark to allow resumption.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load previous results if resuming
    results_file = os.path.join(output_dir, "scan_results.json")
    completed = set()
    all_results = []
    
    if resume and os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
                all_results = [BenchmarkResult(**r) for r in data.get('results', [])]
                completed = set(data.get('completed', []))
                print(f"Resuming from {len(completed)} completed benchmarks")
        except:
            pass
    
    # Calculate total work
    total_runs = len(graphs) * len(benchmarks) * len(algorithms)
    completed_runs = len(completed)
    
    print_header("Full Correlation Scan (Sequential Execution)")
    print(f"Graphs: {len(graphs)}")
    print(f"Benchmarks: {', '.join(benchmarks)}")
    print(f"Algorithms: {len(algorithms)} ({', '.join(str(a) for a in sorted(algorithms.keys()))})")
    print(f"Total runs: {total_runs}")
    print(f"Already completed: {completed_runs}")
    print(f"Remaining: {total_runs - completed_runs}")
    print(f"Baseline for speedup: {ALL_ALGORITHMS.get(BASELINE_ALGO, 'RANDOM')} (ID: {BASELINE_ALGO})")
    print()
    
    start_time = time.time()
    runs_completed = 0
    
    # Iterate: graph -> benchmark -> algorithm (sequential)
    for graph_idx, (graph_name, graph_path, graph_size) in enumerate(graphs):
        print_subheader(f"Graph {graph_idx+1}/{len(graphs)}: {graph_name} ({format_size(graph_size)})")
        graph_args = f"-f {graph_path} -s"
        
        for bench_idx, benchmark in enumerate(benchmarks):
            binary = f"./bench/bin/{benchmark}"
            
            if not os.path.exists(binary):
                print(f"  {Colors.YELLOW}Skipping {benchmark} (binary not found){Colors.RESET}")
                continue
            
            print(f"\n  [{benchmark}] ", end="", flush=True)
            
            bench_results = []
            
            for algo_id, algo_name in sorted(algorithms.items()):
                # Check if already completed
                run_key = f"{graph_name}:{benchmark}:{algo_id}"
                if run_key in completed:
                    # Find existing result
                    existing = next((r for r in all_results 
                                   if r.graph_name == graph_name 
                                   and r.benchmark == benchmark 
                                   and r.algo_id == algo_id), None)
                    if existing:
                        bench_results.append(existing)
                    print(".", end="", flush=True)
                    continue
                
                # Run the benchmark (SEQUENTIAL - one at a time!)
                parsed, output = run_single_benchmark(
                    binary=binary,
                    graph_args=graph_args,
                    algo_id=algo_id,
                    num_trials=num_trials,
                    timeout=timeout
                )
                
                runs_completed += 1
                
                if parsed and 'average_time' in parsed:
                    avg_time = parsed['average_time']
                    reorder_time = parsed.get('reorder_time', 0) + parsed.get('relabel_time', 0)
                    
                    result = BenchmarkResult(
                        graph_name=graph_name,
                        benchmark=benchmark,
                        algo_id=algo_id,
                        algo_name=algo_name,
                        average_time=avg_time,
                        reorder_time=reorder_time,
                        total_time=avg_time + reorder_time,
                        nodes=parsed.get('nodes', 0),
                        edges=parsed.get('edges', 0),
                        iterations=parsed.get('iterations', 0),
                        modularity=parsed.get('modularity', 0.0)
                    )
                    all_results.append(result)
                    bench_results.append(result)
                    completed.add(run_key)
                    
                    print(f"{Colors.GREEN}✓{Colors.RESET}", end="", flush=True)
                else:
                    print(f"{Colors.RED}✗{Colors.RESET}", end="", flush=True)
                
                # Save progress after each run
                save_results(all_results, completed, output_dir)
                
                # Print ETA
                elapsed = time.time() - start_time
                if runs_completed > 0:
                    rate = elapsed / runs_completed
                    remaining = total_runs - len(completed)
                    eta = rate * remaining
                    print(f" ETA: {format_eta(eta)}", end="", flush=True)
            
            # Print benchmark summary
            if bench_results:
                # Find baseline and best
                baseline = next((r for r in bench_results if r.algo_id == BASELINE_ALGO), None)
                if baseline and baseline.average_time > 0:
                    best = min(bench_results, key=lambda r: r.average_time)
                    speedup = baseline.average_time / best.average_time
                    print(f"\n    Best: {best.algo_name} ({format_time(best.average_time)}, "
                          f"{Colors.GREEN}{speedup:.2f}x vs RANDOM{Colors.RESET})")
    
    # Final save
    save_results(all_results, completed, output_dir)
    
    elapsed = time.time() - start_time
    print_header(f"Scan Complete ({format_eta(elapsed)})")
    print(f"Total results: {len(all_results)}")
    print(f"Results saved to: {output_dir}")
    
    return all_results


def save_results(results: List[BenchmarkResult], completed: set, output_dir: str):
    """Save results incrementally."""
    results_file = os.path.join(output_dir, "scan_results.json")
    data = {
        'timestamp': datetime.now().isoformat(),
        'completed': list(completed),
        'results': [asdict(r) for r in results]
    }
    with open(results_file, 'w') as f:
        json.dump(data, f, indent=2)


# ============================================================================
# Analysis
# ============================================================================

def analyze_results(results: List[BenchmarkResult], output_dir: str):
    """Analyze results and generate correlation data."""
    
    print_header("Analysis Results")
    
    # Group by graph and benchmark
    by_graph_bench = {}
    for r in results:
        key = (r.graph_name, r.benchmark)
        if key not in by_graph_bench:
            by_graph_bench[key] = []
        by_graph_bench[key].append(r)
    
    # Calculate speedups relative to RANDOM baseline
    speedup_data = []
    best_algorithms = {}
    
    for (graph_name, benchmark), bench_results in by_graph_bench.items():
        # Find baseline (RANDOM)
        baseline = next((r for r in bench_results if r.algo_id == BASELINE_ALGO), None)
        if not baseline or baseline.average_time <= 0:
            continue
        
        baseline_time = baseline.average_time
        
        # Calculate speedups
        for r in bench_results:
            if r.average_time > 0:
                speedup = baseline_time / r.average_time
                speedup_data.append({
                    'graph': graph_name,
                    'benchmark': benchmark,
                    'algo_id': r.algo_id,
                    'algo_name': r.algo_name,
                    'time': r.average_time,
                    'speedup': speedup,
                    'reorder_time': r.reorder_time,
                })
        
        # Find best algorithm
        best = min(bench_results, key=lambda r: r.average_time)
        key = (graph_name, benchmark)
        best_algorithms[key] = {
            'algo_id': best.algo_id,
            'algo_name': best.algo_name,
            'time': best.average_time,
            'speedup': baseline_time / best.average_time if best.average_time > 0 else 0
        }
    
    # Print summary by algorithm
    print_subheader("Average Speedup by Algorithm (vs RANDOM baseline)")
    
    algo_speedups = {}
    for sd in speedup_data:
        aid = sd['algo_id']
        if aid not in algo_speedups:
            algo_speedups[aid] = {'name': sd['algo_name'], 'speedups': [], 'times': []}
        algo_speedups[aid]['speedups'].append(sd['speedup'])
        algo_speedups[aid]['times'].append(sd['time'])
    
    print(f"{'ID':<4} {'Algorithm':<20} {'Avg Speedup':>12} {'Best':>8} {'Worst':>8} {'Count':>6}")
    print("-" * 60)
    
    for algo_id in sorted(algo_speedups.keys()):
        data = algo_speedups[algo_id]
        speedups = data['speedups']
        avg_speedup = sum(speedups) / len(speedups)
        best_speedup = max(speedups)
        worst_speedup = min(speedups)
        
        color = Colors.GREEN if avg_speedup > 1.0 else Colors.RED
        print(f"{algo_id:<4} {data['name']:<20} {color}{avg_speedup:>11.3f}x{Colors.RESET} "
              f"{best_speedup:>7.2f}x {worst_speedup:>7.2f}x {len(speedups):>6}")
    
    # Print best algorithm counts
    print_subheader("Best Algorithm Frequency")
    
    best_counts = {}
    for (graph, bench), best in best_algorithms.items():
        aid = best['algo_id']
        if aid not in best_counts:
            best_counts[aid] = {'name': best['algo_name'], 'count': 0, 'avg_speedup': []}
        best_counts[aid]['count'] += 1
        best_counts[aid]['avg_speedup'].append(best['speedup'])
    
    print(f"{'ID':<4} {'Algorithm':<20} {'Times Best':>12} {'Avg Speedup When Best':>22}")
    print("-" * 60)
    
    for algo_id in sorted(best_counts.keys(), key=lambda x: best_counts[x]['count'], reverse=True):
        data = best_counts[algo_id]
        avg = sum(data['avg_speedup']) / len(data['avg_speedup'])
        print(f"{algo_id:<4} {data['name']:<20} {data['count']:>12} {avg:>21.3f}x")
    
    # Save analysis
    analysis_file = os.path.join(output_dir, "analysis.json")
    analysis = {
        'timestamp': datetime.now().isoformat(),
        'baseline_algorithm': BASELINE_ALGO,
        'speedup_data': speedup_data,
        'best_algorithms': {f"{k[0]}:{k[1]}": v for k, v in best_algorithms.items()},
        'algorithm_summary': {
            str(aid): {
                'name': data['name'],
                'avg_speedup': sum(data['speedups']) / len(data['speedups']),
                'best_speedup': max(data['speedups']),
                'worst_speedup': min(data['speedups']),
                'count': len(data['speedups'])
            }
            for aid, data in algo_speedups.items()
        }
    }
    
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nAnalysis saved to: {analysis_file}")
    
    return analysis


# ============================================================================
# Perceptron Weight Update
# ============================================================================

def update_perceptron_weights(analysis: Dict, output_dir: str):
    """Update perceptron weights based on correlation analysis."""
    
    print_subheader("Updating Perceptron Weights")
    
    # Load existing weights
    weights_file = "./scripts/perceptron_weights.json"
    if os.path.exists(weights_file):
        with open(weights_file, 'r') as f:
            weights = json.load(f)
    else:
        weights = {}
    
    # Update weights based on algorithm performance
    algo_summary = analysis.get('algorithm_summary', {})
    
    for algo_id_str, data in algo_summary.items():
        algo_id = int(algo_id_str)
        avg_speedup = data['avg_speedup']
        
        # Normalize speedup to 0-1 range for bias
        # Higher speedup = higher bias (more likely to be selected)
        normalized_bias = min(1.0, avg_speedup / 2.0)
        
        if str(algo_id) not in weights:
            weights[str(algo_id)] = {
                'bias': normalized_bias,
                'weights': {}
            }
        else:
            # Blend with existing weights (70% new, 30% old)
            old_bias = weights[str(algo_id)].get('bias', 0.5)
            weights[str(algo_id)]['bias'] = 0.7 * normalized_bias + 0.3 * old_bias
    
    # Save updated weights
    new_weights_file = os.path.join(output_dir, "perceptron_weights.json")
    with open(new_weights_file, 'w') as f:
        json.dump(weights, f, indent=2)
    
    print(f"Updated perceptron weights saved to: {new_weights_file}")
    
    # Also update main weights file
    with open(weights_file, 'w') as f:
        json.dump(weights, f, indent=2)
    
    print(f"Main weights file updated: {weights_file}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Full Correlation Scan with Sequential Execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test with synthetic graphs
    python3 scripts/analysis/full_correlation_scan.py --quick
    
    # SMALL graphs only (< 100MB)
    python3 scripts/analysis/full_correlation_scan.py --small
    
    # SMALL + MEDIUM graphs (< 1GB)
    python3 scripts/analysis/full_correlation_scan.py --medium
    
    # Full scan (all graphs)
    python3 scripts/analysis/full_correlation_scan.py --full
    
    # Resume interrupted scan
    python3 scripts/analysis/full_correlation_scan.py --resume
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with synthetic graphs and few algorithms')
    parser.add_argument('--small', action='store_true',
                       help='Run on SMALL graphs only (< 100MB)')
    parser.add_argument('--medium', action='store_true',
                       help='Run on SMALL + MEDIUM graphs (< 1GB)')
    parser.add_argument('--full', action='store_true',
                       help='Run on ALL graphs')
    parser.add_argument('--graphs-dir', default='./graphs',
                       help='Directory containing graph files')
    parser.add_argument('--benchmarks', nargs='+', default=BENCHMARKS,
                       help=f'Benchmarks to run (default: {BENCHMARKS})')
    parser.add_argument('--trials', type=int, default=3,
                       help='Number of trials per benchmark')
    parser.add_argument('--timeout', type=int, default=600,
                       help='Timeout per benchmark (seconds)')
    parser.add_argument('--output', default='./results',
                       help='Output directory')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous incomplete scan')
    parser.add_argument('--no-analysis', action='store_true',
                       help='Skip analysis phase')
    
    args = parser.parse_args()
    
    # Determine graphs and algorithms based on mode
    if args.quick:
        graphs = [(name, args, 0) for name, args in SYNTHETIC_GRAPHS.items()]
        algorithms = QUICK_ALGORITHMS
        size_limit = None
    elif args.small:
        graphs = discover_graphs(args.graphs_dir, SMALL_GRAPH_LIMIT)
        algorithms = ALL_ALGORITHMS
        size_limit = SMALL_GRAPH_LIMIT
    elif args.medium:
        graphs = discover_graphs(args.graphs_dir, MEDIUM_GRAPH_LIMIT)
        algorithms = ALL_ALGORITHMS
        size_limit = MEDIUM_GRAPH_LIMIT
    elif args.full:
        graphs = discover_graphs(args.graphs_dir)
        algorithms = ALL_ALGORITHMS
        size_limit = None
    else:
        # Default: SMALL + MEDIUM
        graphs = discover_graphs(args.graphs_dir, MEDIUM_GRAPH_LIMIT)
        algorithms = ALL_ALGORITHMS
        size_limit = MEDIUM_GRAPH_LIMIT
    
    # Add synthetic graphs if no real graphs found
    if not graphs:
        print(f"No graphs found in {args.graphs_dir}, using synthetic graphs")
        graphs = [(name, graph_args, 0) for name, graph_args in SYNTHETIC_GRAPHS.items()]
    
    # For synthetic graphs, convert format
    processed_graphs = []
    for item in graphs:
        if len(item) == 3:
            name, path_or_args, size = item
            if path_or_args.startswith('-g'):
                # Synthetic graph
                processed_graphs.append((name, path_or_args, size))
            else:
                # Real graph file
                processed_graphs.append((name, path_or_args, size))
        else:
            processed_graphs.append(item)
    
    graphs = processed_graphs
    
    print(f"Found {len(graphs)} graphs")
    for name, path, size in graphs[:10]:
        print(f"  - {name}: {format_size(size) if size else 'synthetic'}")
    if len(graphs) > 10:
        print(f"  ... and {len(graphs) - 10} more")
    
    # Run scan
    results = run_full_scan(
        graphs=graphs,
        algorithms=algorithms,
        benchmarks=args.benchmarks,
        num_trials=args.trials,
        timeout=args.timeout,
        output_dir=args.output,
        resume=args.resume
    )
    
    if not args.no_analysis and results:
        # Analyze results
        analysis = analyze_results(results, args.output)
        
        # Update perceptron weights
        update_perceptron_weights(analysis, args.output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
