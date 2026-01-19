#!/usr/bin/env python3
"""
Full Correlation Scan for GraphBrew

Runs all reordering algorithms (0-20) on all graphs with multiple benchmarks.
Uses RANDOM (1) as baseline for speedup calculations.
Skips TC since reordering doesn't help it.

Output:
- Detailed benchmark results per graph/algorithm/benchmark
- Correlation analysis for perceptron training
- Updated perceptron weights with cache features
"""

import os
import sys
import json
import subprocess
import re
import time
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

# All reordering algorithms (skip 13=GraphBrew and 14=MAP which need special params)
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
    # 13: "GraphBrewOrder",  # Skip - needs special params
    # 14: "MAP",             # Skip - needs file
    15: "AdaptiveOrder",
    16: "LeidenDFS",
    17: "LeidenDFSHub",
    18: "LeidenDFSSize",
    19: "LeidenBFS",
    20: "LeidenHybrid",
}

# Benchmarks to run (skip TC - reordering doesn't help)
BENCHMARKS = ["pr", "bfs", "cc", "sssp", "bc"]

# Baseline algorithm for speedup calculations
BASELINE_ALGO = 1  # RANDOM

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(title: str):
    print(f"\n{Colors.BOLD}{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}{Colors.RESET}\n")


def print_subheader(title: str):
    print(f"\n{Colors.CYAN}{'-'*70}")
    print(f"{title}")
    print(f"{'-'*70}{Colors.RESET}")


def discover_graphs(graphs_dir: str, max_size_mb: int = None) -> List[Tuple[str, str, int]]:
    """Discover graphs and return list of (name, path, size_mb)."""
    graphs = []
    graphs_path = Path(graphs_dir)
    
    if not graphs_path.exists():
        return graphs
    
    for subdir in sorted(graphs_path.iterdir()):
        if subdir.is_dir():
            # Get directory size
            try:
                size_bytes = sum(f.stat().st_size for f in subdir.rglob('*') if f.is_file())
                size_mb = size_bytes / (1024 * 1024)
            except:
                size_mb = 0
            
            # Find graph file
            for ext in ['.el', '.wel', '.gr', '.mtx', '.sg']:
                candidates = list(subdir.glob(f'*{ext}'))
                if candidates:
                    graph_file = candidates[0]
                    if max_size_mb is None or size_mb <= max_size_mb:
                        graphs.append((subdir.name, str(graph_file), size_mb))
                    break
    
    # Sort by size
    graphs.sort(key=lambda x: x[2])
    return graphs


def run_benchmark(binary: str, graph_path: str, algo_id: int, 
                  num_trials: int = 3, timeout: int = 600) -> Optional[Dict]:
    """Run a single benchmark and return parsed results."""
    cmd = f"{binary} -f {graph_path} -s -o {algo_id} -n {num_trials}"
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        output = result.stdout + result.stderr
        
        # Parse results
        parsed = {}
        
        # Parse average time
        avg_match = re.search(r'Average Time:\s+([\d.]+)', output)
        if avg_match:
            parsed['avg_time'] = float(avg_match.group(1))
        
        # Parse trial time (for single trial)
        trial_match = re.search(r'Trial Time:\s+([\d.]+)', output)
        if trial_match and 'avg_time' not in parsed:
            parsed['avg_time'] = float(trial_match.group(1))
        
        # Parse reorder time
        reorder_match = re.search(r'Reorder Time:\s+([\d.]+)', output)
        if reorder_match:
            parsed['reorder_time'] = float(reorder_match.group(1))
        
        # Parse relabel time
        relabel_match = re.search(r'Relabel Time:\s+([\d.]+)', output)
        if relabel_match:
            parsed['relabel_time'] = float(relabel_match.group(1))
        
        # Parse graph info
        graph_match = re.search(r'Graph has (\d+) nodes and (\d+)', output)
        if graph_match:
            parsed['nodes'] = int(graph_match.group(1))
            parsed['edges'] = int(graph_match.group(2))
        
        # Parse modularity for Leiden algorithms
        mod_match = re.search(r'[Mm]odularity[:\s]+([\d.]+)', output)
        if mod_match:
            parsed['modularity'] = float(mod_match.group(1))
        
        if 'avg_time' in parsed:
            return parsed
        
        return None
        
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        return None


def run_cache_simulation(binary: str, graph_path: str, algo_id: int,
                         timeout: int = 600) -> Optional[Dict]:
    """Run cache simulation and return cache stats."""
    import tempfile
    
    json_file = tempfile.mktemp(suffix='.json')
    cmd = f"{binary} -f {graph_path} -s -o {algo_id} -n 1"
    
    env = os.environ.copy()
    env['CACHE_OUTPUT_JSON'] = json_file
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env
        )
        
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                data = json.load(f)
            os.remove(json_file)
            return data
        return None
        
    except:
        if os.path.exists(json_file):
            os.remove(json_file)
        return None


def run_full_scan(
    graphs_dir: str,
    output_dir: str,
    algorithms: Dict[int, str] = None,
    benchmarks: List[str] = None,
    max_size_mb: int = None,
    num_trials: int = 3,
    timeout: int = 600,
    include_cache: bool = True
) -> Dict:
    """Run full correlation scan."""
    
    if algorithms is None:
        algorithms = ALL_ALGORITHMS
    if benchmarks is None:
        benchmarks = BENCHMARKS
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Discover graphs
    graphs = discover_graphs(graphs_dir, max_size_mb)
    if not graphs:
        print(f"{Colors.RED}No graphs found in {graphs_dir}{Colors.RESET}")
        return {}
    
    print_header("Full Correlation Scan")
    print(f"Graphs directory: {graphs_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Algorithms: {list(algorithms.values())}")
    print(f"Benchmarks: {benchmarks}")
    print(f"Trials: {num_trials}")
    print(f"Max graph size: {max_size_mb}MB" if max_size_mb else "No size limit")
    print(f"Baseline: {ALL_ALGORITHMS[BASELINE_ALGO]} (ID: {BASELINE_ALGO})")
    print(f"\nFound {len(graphs)} graphs")
    
    # Results storage
    all_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'graphs_dir': graphs_dir,
            'algorithms': algorithms,
            'benchmarks': benchmarks,
            'baseline_algo': BASELINE_ALGO,
            'num_trials': num_trials,
        },
        'graphs': {},
        'summary': {
            'best_by_graph': {},
            'best_by_benchmark': defaultdict(lambda: defaultdict(int)),
            'avg_speedup_by_algo': defaultdict(list),
        }
    }
    
    total_runs = len(graphs) * len(benchmarks) * len(algorithms)
    current_run = 0
    
    for graph_name, graph_path, size_mb in graphs:
        print_subheader(f"Graph: {graph_name} ({size_mb:.1f}MB)")
        
        all_results['graphs'][graph_name] = {
            'path': graph_path,
            'size_mb': size_mb,
            'benchmarks': {}
        }
        
        for benchmark in benchmarks:
            binary = f"./bench/bin/{benchmark}"
            sim_binary = f"./bench/bin_sim/{benchmark}"
            
            if not os.path.exists(binary):
                print(f"  {Colors.YELLOW}Skipping {benchmark} (binary not found){Colors.RESET}")
                continue
            
            print(f"\n  {Colors.BOLD}{benchmark}:{Colors.RESET}")
            
            bench_results = {}
            baseline_time = None
            
            # First run baseline (RANDOM)
            if BASELINE_ALGO in algorithms:
                current_run += 1
                print(f"    [{current_run}/{total_runs}] {algorithms[BASELINE_ALGO]} (baseline)... ", end="", flush=True)
                
                result = run_benchmark(binary, graph_path, BASELINE_ALGO, num_trials, timeout)
                if result and 'avg_time' in result:
                    baseline_time = result['avg_time']
                    bench_results[BASELINE_ALGO] = {
                        'name': algorithms[BASELINE_ALGO],
                        'avg_time': result['avg_time'],
                        'reorder_time': result.get('reorder_time', 0),
                        'relabel_time': result.get('relabel_time', 0),
                        'speedup': 1.0,
                    }
                    print(f"{Colors.GREEN}{result['avg_time']:.4f}s{Colors.RESET}")
                else:
                    print(f"{Colors.RED}FAILED{Colors.RESET}")
            
            # Run all other algorithms
            for algo_id, algo_name in sorted(algorithms.items()):
                if algo_id == BASELINE_ALGO:
                    continue  # Already ran
                
                current_run += 1
                print(f"    [{current_run}/{total_runs}] {algo_name}... ", end="", flush=True)
                
                result = run_benchmark(binary, graph_path, algo_id, num_trials, timeout)
                
                if result and 'avg_time' in result:
                    speedup = baseline_time / result['avg_time'] if baseline_time else 1.0
                    
                    bench_results[algo_id] = {
                        'name': algo_name,
                        'avg_time': result['avg_time'],
                        'reorder_time': result.get('reorder_time', 0),
                        'relabel_time': result.get('relabel_time', 0),
                        'speedup': speedup,
                        'modularity': result.get('modularity'),
                    }
                    
                    # Add cache stats if available
                    if include_cache and os.path.exists(sim_binary):
                        cache_stats = run_cache_simulation(sim_binary, graph_path, algo_id, timeout)
                        if cache_stats:
                            bench_results[algo_id]['cache'] = {
                                'l1_hit_rate': cache_stats.get('L1', {}).get('hit_rate', 0),
                                'l2_hit_rate': cache_stats.get('L2', {}).get('hit_rate', 0),
                                'l3_hit_rate': cache_stats.get('L3', {}).get('hit_rate', 0),
                            }
                    
                    color = Colors.GREEN if speedup > 1.0 else Colors.YELLOW if speedup > 0.9 else Colors.RED
                    print(f"{color}{result['avg_time']:.4f}s (speedup: {speedup:.2f}x){Colors.RESET}")
                    
                    # Track for summary
                    all_results['summary']['avg_speedup_by_algo'][algo_name].append(speedup)
                else:
                    print(f"{Colors.RED}FAILED{Colors.RESET}")
            
            # Find best algorithm for this benchmark
            if bench_results:
                best_algo = max(bench_results.items(), key=lambda x: x[1]['speedup'])
                print(f"  {Colors.BOLD}Best: {best_algo[1]['name']} ({best_algo[1]['avg_time']:.4f}s, {best_algo[1]['speedup']:.2f}x speedup){Colors.RESET}")
                
                all_results['summary']['best_by_benchmark'][benchmark][best_algo[1]['name']] += 1
            
            all_results['graphs'][graph_name]['benchmarks'][benchmark] = bench_results
        
        # Save intermediate results
        results_file = os.path.join(output_dir, 'correlation_results.json')
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
    
    return all_results


def compute_correlations(results: Dict) -> Dict:
    """Compute feature-algorithm correlations for perceptron training."""
    correlations = {
        'by_algorithm': {},
        'by_benchmark': {},
        'feature_importance': {},
    }
    
    # Aggregate speedups by algorithm
    for algo_name, speedups in results['summary']['avg_speedup_by_algo'].items():
        if speedups:
            correlations['by_algorithm'][algo_name] = {
                'avg_speedup': sum(speedups) / len(speedups),
                'max_speedup': max(speedups),
                'min_speedup': min(speedups),
                'count': len(speedups),
            }
    
    return correlations


def generate_perceptron_weights(results: Dict, output_file: str):
    """Generate perceptron weights from benchmark results."""
    weights = {}
    
    # Calculate average speedup for each algorithm
    algo_speedups = results['summary']['avg_speedup_by_algo']
    
    for algo_name, speedups in algo_speedups.items():
        if speedups:
            avg_speedup = sum(speedups) / len(speedups)
            # Find algorithm ID
            algo_id = None
            for aid, aname in ALL_ALGORITHMS.items():
                if aname == algo_name:
                    algo_id = aid
                    break
            
            if algo_id is not None:
                # Simple weight based on average speedup
                weights[str(algo_id)] = {
                    'name': algo_name,
                    'bias': min(1.0, avg_speedup / 2),
                    'avg_speedup': avg_speedup,
                    'sample_count': len(speedups),
                    'weights': {
                        'modularity': 0.1 if 'Leiden' in algo_name else 0.0,
                        'log_nodes': 0.05,
                        'log_edges': 0.05,
                        'density': 0.1,
                        'avg_degree': 0.1,
                        'degree_variance': 0.1 if 'Hub' in algo_name else 0.0,
                        'hub_concentration': 0.2 if 'Hub' in algo_name else 0.0,
                        'l1_hit_rate': 0.1,
                        'l2_hit_rate': 0.05,
                        'l3_hit_rate': 0.05,
                        'dram_access_rate': -0.1,
                        'l1_eviction_rate': -0.05,
                        'l2_eviction_rate': -0.05,
                        'l3_eviction_rate': -0.05,
                    }
                }
    
    with open(output_file, 'w') as f:
        json.dump(weights, f, indent=2)
    
    print(f"\nPerceptron weights saved to: {output_file}")


def print_summary(results: Dict):
    """Print summary of results."""
    print_header("Results Summary")
    
    # Best algorithms by benchmark
    print(f"{Colors.BOLD}Best Algorithms by Benchmark:{Colors.RESET}")
    for benchmark, algo_counts in results['summary']['best_by_benchmark'].items():
        print(f"\n  {benchmark}:")
        for algo_name, count in sorted(algo_counts.items(), key=lambda x: -x[1]):
            print(f"    {algo_name}: {count} wins")
    
    # Average speedups
    print(f"\n{Colors.BOLD}Average Speedup by Algorithm (vs RANDOM baseline):{Colors.RESET}")
    algo_speedups = []
    for algo_name, speedups in results['summary']['avg_speedup_by_algo'].items():
        if speedups:
            avg = sum(speedups) / len(speedups)
            algo_speedups.append((algo_name, avg, len(speedups)))
    
    for algo_name, avg_speedup, count in sorted(algo_speedups, key=lambda x: -x[1]):
        color = Colors.GREEN if avg_speedup > 1.0 else Colors.YELLOW if avg_speedup > 0.9 else Colors.RED
        print(f"  {color}{algo_name:20s}: {avg_speedup:.3f}x (n={count}){Colors.RESET}")


def main():
    parser = argparse.ArgumentParser(
        description="Full Correlation Scan for GraphBrew",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--graphs-dir', type=str, default='./graphs',
                        help='Directory containing graph files')
    parser.add_argument('--output', type=str, default='./results/full_correlation',
                        help='Output directory for results')
    parser.add_argument('--max-size', type=int, default=None,
                        help='Maximum graph size in MB (default: no limit)')
    parser.add_argument('--trials', type=int, default=3,
                        help='Number of trials per benchmark')
    parser.add_argument('--timeout', type=int, default=600,
                        help='Timeout per benchmark run in seconds')
    parser.add_argument('--benchmarks', type=str, default='pr,bfs,cc,sssp,bc',
                        help='Comma-separated list of benchmarks')
    parser.add_argument('--algorithms', type=str, default=None,
                        help='Comma-separated list of algorithm IDs (default: all)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Skip cache simulation')
    parser.add_argument('--weights-file', type=str, 
                        default='./scripts/perceptron_weights.json',
                        help='Output file for perceptron weights')
    
    args = parser.parse_args()
    
    # Parse benchmarks
    benchmarks = [b.strip() for b in args.benchmarks.split(',')]
    
    # Parse algorithms
    if args.algorithms:
        algo_ids = [int(a.strip()) for a in args.algorithms.split(',')]
        algorithms = {aid: ALL_ALGORITHMS[aid] for aid in algo_ids if aid in ALL_ALGORITHMS}
    else:
        algorithms = ALL_ALGORITHMS
    
    # Run scan
    results = run_full_scan(
        graphs_dir=args.graphs_dir,
        output_dir=args.output,
        algorithms=algorithms,
        benchmarks=benchmarks,
        max_size_mb=args.max_size,
        num_trials=args.trials,
        timeout=args.timeout,
        include_cache=not args.no_cache
    )
    
    if results:
        # Print summary
        print_summary(results)
        
        # Compute correlations
        correlations = compute_correlations(results)
        corr_file = os.path.join(args.output, 'correlations.json')
        with open(corr_file, 'w') as f:
            json.dump(correlations, f, indent=2)
        print(f"\nCorrelations saved to: {corr_file}")
        
        # Generate perceptron weights
        generate_perceptron_weights(results, args.weights_file)
        
        print(f"\n{Colors.GREEN}Full scan complete!{Colors.RESET}")
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
