#!/usr/bin/env python3
"""
Full Benchmark Scan

Runs comprehensive benchmarks across all algorithms, graphs, and benchmarks.
Collects data for perceptron training and final evaluation.

Usage:
    python3 scripts/analysis/full_benchmark_scan.py [OPTIONS]
"""

import os
import sys
import json
import subprocess
import re
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.common import ALGORITHMS

# ============================================================================
# Configuration
# ============================================================================

# Key algorithms for testing
KEY_ALGORITHMS = [0, 7, 8, 9, 11, 12, 15, 17, 20]

# All algorithms
ALL_ALGORITHMS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20]

# Benchmarks
BENCHMARKS = ['pr', 'bfs', 'cc', 'sssp']

# Graph size limits (bytes)
SMALL_LIMIT = 100 * 1024 * 1024  # 100MB
MEDIUM_LIMIT = 1024 * 1024 * 1024  # 1GB

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class BenchmarkResult:
    graph: str
    benchmark: str
    algorithm_id: int
    algorithm_name: str
    nodes: int
    edges: int
    trial_time: float
    reorder_time: float
    total_time: float
    speedup: float = 1.0


@dataclass
class GraphFeatures:
    name: str
    nodes: int
    edges: int
    density: float
    avg_degree: float
    file_size: int


# ============================================================================
# Utilities
# ============================================================================

def discover_graphs(graphs_dir: str, max_size: int = MEDIUM_LIMIT) -> List[Tuple[str, str, int]]:
    """Discover graphs in directory, filtering by size."""
    graphs = []
    graphs_path = Path(graphs_dir)
    
    if not graphs_path.exists():
        return graphs
    
    for subdir in sorted(graphs_path.iterdir()):
        if subdir.is_dir():
            for ext in ['.mtx', '.el', '.wel', '.gr']:
                candidates = list(subdir.glob(f'*{ext}'))
                if candidates:
                    graph_file = candidates[0]
                    file_size = graph_file.stat().st_size
                    if file_size <= max_size:
                        graphs.append((subdir.name, str(graph_file), file_size))
                    break
    
    return sorted(graphs, key=lambda x: x[2])


def parse_output(output: str) -> Dict[str, Any]:
    """Parse benchmark output."""
    result = {}
    
    # Graph stats
    graph_match = re.search(r'Graph has (\d+) nodes and (\d+)', output)
    if graph_match:
        result['nodes'] = int(graph_match.group(1))
        result['edges'] = int(graph_match.group(2))
    
    # Times
    patterns = {
        'trial_time': r'Trial Time:\s+([\d.]+)',
        'average_time': r'Average Time:\s+([\d.]+)',
        'reorder_time': r'Reorder Time:\s+([\d.]+)',
        'relabel_time': r'Relabel.*Time:\s+([\d.]+)',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            result[key] = float(match.group(1))
    
    return result


def run_single_benchmark(
    binary: str,
    graph_path: str,
    algo_id: int,
    trials: int = 3,
    timeout: int = 300
) -> Optional[Dict]:
    """Run a single benchmark."""
    cmd = f"{binary} -f {graph_path} -s -o {algo_id} -n {trials}"
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        output = result.stdout + result.stderr
        return parse_output(output)
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


# ============================================================================
# Main Benchmark Runner
# ============================================================================

def run_full_scan(
    graphs_dir: str,
    output_dir: str,
    algorithms: List[int],
    benchmarks: List[str],
    trials: int = 3,
    max_size: int = MEDIUM_LIMIT,
    timeout: int = 300
) -> Dict[str, Any]:
    """
    Run full benchmark scan.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Discover graphs
    graphs = discover_graphs(graphs_dir, max_size)
    print(f"Found {len(graphs)} graphs under {max_size/1024/1024:.0f}MB")
    
    # Check binaries
    bin_dir = "./bench/bin"
    available_benchmarks = [b for b in benchmarks if os.path.exists(f"{bin_dir}/{b}")]
    print(f"Available benchmarks: {', '.join(available_benchmarks)}")
    
    results = []
    total_runs = len(graphs) * len(available_benchmarks) * len(algorithms)
    current = 0
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for graph_name, graph_path, file_size in graphs:
        print(f"\n{'='*60}")
        print(f"Graph: {graph_name} ({file_size/1024/1024:.1f}MB)")
        print(f"{'='*60}")
        
        graph_results = {
            'graph': graph_name,
            'file_size': file_size,
            'benchmarks': {}
        }
        
        for benchmark in available_benchmarks:
            binary = f"{bin_dir}/{benchmark}"
            print(f"\n  {benchmark}:")
            
            benchmark_results = {}
            baseline_time = None
            
            for algo_id in algorithms:
                current += 1
                algo_name = ALGORITHMS.get(algo_id, f"Unknown({algo_id})")
                
                print(f"    [{current}/{total_runs}] {algo_name}...", end=" ", flush=True)
                
                parsed = run_single_benchmark(
                    binary, graph_path, algo_id, trials, timeout
                )
                
                if parsed and 'average_time' in parsed:
                    time = parsed['average_time']
                    reorder_time = parsed.get('reorder_time', 0) + parsed.get('relabel_time', 0)
                    
                    if algo_id == 0:
                        baseline_time = time
                    
                    speedup = baseline_time / time if baseline_time and time > 0 else 1.0
                    
                    benchmark_results[algo_id] = {
                        'algorithm': algo_name,
                        'time': time,
                        'reorder_time': reorder_time,
                        'speedup': speedup,
                        'nodes': parsed.get('nodes', 0),
                        'edges': parsed.get('edges', 0)
                    }
                    
                    results.append(BenchmarkResult(
                        graph=graph_name,
                        benchmark=benchmark,
                        algorithm_id=algo_id,
                        algorithm_name=algo_name,
                        nodes=parsed.get('nodes', 0),
                        edges=parsed.get('edges', 0),
                        trial_time=time,
                        reorder_time=reorder_time,
                        total_time=time + reorder_time,
                        speedup=speedup
                    ))
                    
                    print(f"{time:.4f}s (speedup: {speedup:.2f}x)")
                else:
                    print("FAILED")
            
            # Find best
            if benchmark_results:
                best_id = min(benchmark_results, key=lambda x: benchmark_results[x]['time'])
                best = benchmark_results[best_id]
                print(f"  Best: {best['algorithm']} ({best['time']:.4f}s, {best['speedup']:.2f}x speedup)")
            
            graph_results['benchmarks'][benchmark] = benchmark_results
        
        # Save intermediate results
        interim_file = os.path.join(output_dir, f"interim_{timestamp}.json")
        with open(interim_file, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
    
    # Final results
    final_results = {
        'timestamp': timestamp,
        'config': {
            'graphs_dir': graphs_dir,
            'algorithms': algorithms,
            'benchmarks': available_benchmarks,
            'trials': trials,
            'max_size_mb': max_size / 1024 / 1024
        },
        'results': [asdict(r) for r in results],
        'summary': compute_summary(results)
    }
    
    # Save final results
    final_file = os.path.join(output_dir, f"full_benchmark_{timestamp}.json")
    with open(final_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to: {final_file}")
    
    return final_results


def compute_summary(results: List[BenchmarkResult]) -> Dict:
    """Compute summary statistics."""
    summary = {
        'by_algorithm': {},
        'by_benchmark': {},
        'best_per_graph': {}
    }
    
    # Group by algorithm
    by_algo = defaultdict(list)
    for r in results:
        by_algo[r.algorithm_id].append(r)
    
    for algo_id, algo_results in by_algo.items():
        speedups = [r.speedup for r in algo_results if r.speedup > 0]
        times = [r.trial_time for r in algo_results if r.trial_time > 0]
        
        summary['by_algorithm'][algo_id] = {
            'name': ALGORITHMS.get(algo_id, f"Unknown({algo_id})"),
            'avg_speedup': sum(speedups) / len(speedups) if speedups else 0,
            'max_speedup': max(speedups) if speedups else 0,
            'min_speedup': min(speedups) if speedups else 0,
            'avg_time': sum(times) / len(times) if times else 0,
            'count': len(algo_results)
        }
    
    # Group by benchmark
    by_bench = defaultdict(list)
    for r in results:
        by_bench[r.benchmark].append(r)
    
    for bench, bench_results in by_bench.items():
        # Find best algorithm per graph
        by_graph = defaultdict(list)
        for r in bench_results:
            by_graph[r.graph].append(r)
        
        best_counts = defaultdict(int)
        for graph, graph_results in by_graph.items():
            if graph_results:
                best = min(graph_results, key=lambda x: x.trial_time)
                best_counts[best.algorithm_id] += 1
        
        summary['by_benchmark'][bench] = {
            'best_algorithm_counts': dict(best_counts),
            'total_graphs': len(by_graph)
        }
    
    return summary


def generate_perceptron_weights(results: List[BenchmarkResult], output_file: str):
    """Generate perceptron weights from results."""
    # Group results by graph and benchmark
    by_graph_bench = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_graph_bench[r.graph][r.benchmark].append(r)
    
    # Compute weights based on which algorithms perform best
    algo_wins = defaultdict(int)
    algo_speedups = defaultdict(list)
    
    for graph, benchmarks in by_graph_bench.items():
        for bench, bench_results in benchmarks.items():
            if bench_results:
                best = min(bench_results, key=lambda x: x.trial_time)
                algo_wins[best.algorithm_id] += 1
                
                for r in bench_results:
                    algo_speedups[r.algorithm_id].append(r.speedup)
    
    # Generate weights
    weights = {}
    total_wins = sum(algo_wins.values())
    
    for algo_id in set(algo_wins.keys()) | set(algo_speedups.keys()):
        win_rate = algo_wins.get(algo_id, 0) / total_wins if total_wins > 0 else 0
        speedups = algo_speedups.get(algo_id, [1.0])
        avg_speedup = sum(speedups) / len(speedups)
        
        weights[algo_id] = {
            'bias': min(1.0, avg_speedup / 2),
            'win_rate': win_rate,
            'avg_speedup': avg_speedup,
            'weights': {
                'modularity': 0.1 * win_rate,
                'log_nodes': 0.05,
                'log_edges': 0.05,
                'density': 0.1 * (1 - win_rate),
                'avg_degree': 0.05,
                'degree_variance': 0.05,
                'hub_concentration': 0.1 * win_rate,
                # Cache features
                'l1_hit_rate': 0.1 * avg_speedup,
                'l2_hit_rate': 0.05 * avg_speedup,
                'l3_hit_rate': 0.05 * avg_speedup,
                'dram_access_rate': -0.1 * avg_speedup,
                'l1_eviction_rate': -0.05,
                'l2_eviction_rate': -0.05,
                'l3_eviction_rate': -0.05,
            }
        }
    
    # Save weights
    with open(output_file, 'w') as f:
        json.dump(weights, f, indent=2)
    
    print(f"Perceptron weights saved to: {output_file}")
    return weights


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Full Benchmark Scan")
    parser.add_argument('--graphs-dir', '-g', default='./graphs',
                        help='Directory containing graphs')
    parser.add_argument('--output', '-o', default='./results/full_scan',
                        help='Output directory')
    parser.add_argument('--algorithms', '-a', default=None,
                        help='Comma-separated algorithm IDs (default: key algorithms)')
    parser.add_argument('--benchmarks', '-b', nargs='+', default=['pr', 'bfs', 'cc'],
                        help='Benchmarks to run')
    parser.add_argument('--trials', '-n', type=int, default=3,
                        help='Number of trials')
    parser.add_argument('--max-size', type=int, default=500,
                        help='Maximum graph size in MB')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Timeout per benchmark in seconds')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Quick mode (small graphs, key algorithms)')
    parser.add_argument('--full', action='store_true',
                        help='Full mode (all algorithms)')
    parser.add_argument('--weights-file', '-w', default='./scripts/perceptron_weights.json',
                        help='Output file for perceptron weights')
    
    args = parser.parse_args()
    
    # Configure algorithms
    if args.algorithms:
        algorithms = [int(x) for x in args.algorithms.split(',')]
    elif args.quick:
        algorithms = [0, 7, 12, 15, 20]
    elif args.full:
        algorithms = ALL_ALGORITHMS
    else:
        algorithms = KEY_ALGORITHMS
    
    # Configure max size
    if args.quick:
        max_size = 50 * 1024 * 1024  # 50MB
    else:
        max_size = args.max_size * 1024 * 1024
    
    print("="*60)
    print("Full Benchmark Scan")
    print("="*60)
    print(f"Graphs directory: {args.graphs_dir}")
    print(f"Output directory: {args.output}")
    print(f"Algorithms: {[ALGORITHMS.get(a, a) for a in algorithms]}")
    print(f"Benchmarks: {args.benchmarks}")
    print(f"Trials: {args.trials}")
    print(f"Max graph size: {max_size/1024/1024:.0f}MB")
    print()
    
    # Run scan
    results = run_full_scan(
        graphs_dir=args.graphs_dir,
        output_dir=args.output,
        algorithms=algorithms,
        benchmarks=args.benchmarks,
        trials=args.trials,
        max_size=max_size,
        timeout=args.timeout
    )
    
    # Generate perceptron weights
    if 'results' in results:
        result_objects = [
            BenchmarkResult(**r) for r in results['results']
        ]
        generate_perceptron_weights(result_objects, args.weights_file)
    
    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    if 'summary' in results:
        summary = results['summary']
        
        print("\nBest algorithm by average speedup:")
        by_algo = summary.get('by_algorithm', {})
        sorted_algos = sorted(by_algo.items(), key=lambda x: x[1].get('avg_speedup', 0), reverse=True)
        for algo_id, stats in sorted_algos[:5]:
            print(f"  {stats['name']}: avg={stats['avg_speedup']:.2f}x, max={stats['max_speedup']:.2f}x")
        
        print("\nBest algorithm wins by benchmark:")
        for bench, stats in summary.get('by_benchmark', {}).items():
            print(f"  {bench}:")
            counts = stats.get('best_algorithm_counts', {})
            for algo_id, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"    {ALGORITHMS.get(int(algo_id), algo_id)}: {count} wins")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
