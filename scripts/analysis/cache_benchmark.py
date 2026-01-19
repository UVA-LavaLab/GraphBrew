#!/usr/bin/env python3
"""
Cache Performance Benchmark Suite

Runs cache simulations across all algorithms, graphs, and reordering techniques.
Generates comprehensive cache performance data for correlation analysis and
perceptron model training.

Features:
- Multi-algorithm cache simulation (pr, bfs, cc, bc, sssp, tc)
- Multi-graph support (synthetic and real-world)
- Reordering comparison
- JSON export for further analysis
- Correlation with execution time

Usage:
    python3 scripts/analysis/cache_benchmark.py [OPTIONS]

Examples:
    # Quick test with synthetic graphs
    python3 scripts/analysis/cache_benchmark.py --quick
    
    # Full benchmark on real graphs
    python3 scripts/analysis/cache_benchmark.py --graphs-dir ./graphs
    
    # Specific algorithm and reorderings
    python3 scripts/analysis/cache_benchmark.py --algorithms pr bfs --reorders 0 7 12 20
"""

import os
import sys
import argparse
import json
import subprocess
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.common import (
    ALGORITHMS, QUICK_ALGORITHMS,
    Colors, print_header, print_subheader,
    format_time, format_speedup
)

# ============================================================================
# Configuration
# ============================================================================

# Simulation binaries directory
SIM_BIN_DIR = "./bench/bin_sim"

# Algorithms available for cache simulation
SIM_ALGORITHMS = ["pr", "bfs", "cc", "bc", "sssp", "tc"]

# Reordering algorithms to test
DEFAULT_REORDERS = [0, 7, 8, 12, 17, 20]
QUICK_REORDERS = [0, 7, 12, 20]

# Synthetic graphs for quick testing
SYNTHETIC_GRAPHS = {
    "rmat_10": "-g 10",
    "rmat_12": "-g 12",
    "rmat_14": "-g 14",
    "rmat_16": "-g 16",
}

QUICK_SYNTHETIC = {
    "rmat_10": "-g 10",
    "rmat_12": "-g 12",
}

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class CacheStats:
    """Statistics from a single cache level."""
    level: str
    size_bytes: int
    ways: int
    hits: int
    misses: int
    hit_rate: float
    evictions: int
    
    def __post_init__(self):
        self.total = self.hits + self.misses
        self.miss_rate = 1.0 - self.hit_rate if self.hit_rate else 0.0
        self.eviction_rate = self.evictions / self.total if self.total > 0 else 0.0


@dataclass
class CacheResult:
    """Complete cache simulation result."""
    algorithm: str
    graph_name: str
    graph_args: str
    reorder_id: int
    reorder_name: str
    
    # Cache statistics
    total_accesses: int = 0
    memory_accesses: int = 0
    overall_hit_rate: float = 0.0
    
    l1_stats: Optional[CacheStats] = None
    l2_stats: Optional[CacheStats] = None
    l3_stats: Optional[CacheStats] = None
    
    # Timing
    trial_time: float = 0.0
    reorder_time: float = 0.0
    
    # Graph info
    nodes: int = 0
    edges: int = 0
    
    def get_feature_vector(self) -> List[float]:
        """Return cache features as a vector for ML models."""
        return [
            self.l1_stats.hit_rate if self.l1_stats else 0.0,
            self.l2_stats.hit_rate if self.l2_stats else 0.0,
            self.l3_stats.hit_rate if self.l3_stats else 0.0,
            self.memory_accesses / self.total_accesses if self.total_accesses > 0 else 0.0,
            self.l1_stats.eviction_rate if self.l1_stats else 0.0,
            self.l2_stats.eviction_rate if self.l2_stats else 0.0,
            self.l3_stats.eviction_rate if self.l3_stats else 0.0,
        ]
    
    def to_dict(self) -> Dict:
        result = {
            'algorithm': self.algorithm,
            'graph_name': self.graph_name,
            'reorder_id': self.reorder_id,
            'reorder_name': self.reorder_name,
            'total_accesses': self.total_accesses,
            'memory_accesses': self.memory_accesses,
            'overall_hit_rate': self.overall_hit_rate,
            'trial_time': self.trial_time,
            'reorder_time': self.reorder_time,
            'nodes': self.nodes,
            'edges': self.edges,
        }
        if self.l1_stats:
            result['l1_hit_rate'] = self.l1_stats.hit_rate
            result['l1_eviction_rate'] = self.l1_stats.eviction_rate
        if self.l2_stats:
            result['l2_hit_rate'] = self.l2_stats.hit_rate
            result['l2_eviction_rate'] = self.l2_stats.eviction_rate
        if self.l3_stats:
            result['l3_hit_rate'] = self.l3_stats.hit_rate
            result['l3_eviction_rate'] = self.l3_stats.eviction_rate
        return result


# ============================================================================
# Cache Simulation Runner
# ============================================================================

def run_cache_simulation(
    algorithm: str,
    graph_args: str,
    reorder_id: int = 0,
    graph_name: str = "",
    timeout: int = 300,
    json_output: str = None
) -> Optional[CacheResult]:
    """
    Run a cache simulation for a specific algorithm and configuration.
    
    Args:
        algorithm: Algorithm name (pr, bfs, cc, etc.)
        graph_args: Graph arguments (-g N or -f path)
        reorder_id: Reordering algorithm ID
        graph_name: Human-readable graph name
        timeout: Maximum execution time in seconds
        json_output: Path for JSON output (optional)
    
    Returns:
        CacheResult with statistics, or None if failed
    """
    binary = f"{SIM_BIN_DIR}/{algorithm}"
    
    if not os.path.exists(binary):
        print(f"{Colors.RED}Error: Binary not found: {binary}{Colors.RESET}")
        return None
    
    # Build command
    cmd = f"{binary} {graph_args} -o {reorder_id} -n 1"
    
    # Set up environment for JSON output
    env = os.environ.copy()
    if json_output:
        env['CACHE_OUTPUT_JSON'] = json_output
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env
        )
        
        output = result.stdout + result.stderr
        
        # Parse result
        cache_result = CacheResult(
            algorithm=algorithm,
            graph_name=graph_name or graph_args,
            graph_args=graph_args,
            reorder_id=reorder_id,
            reorder_name=ALGORITHMS.get(reorder_id, f"Unknown({reorder_id})"),
        )
        
        # Parse graph stats
        graph_match = re.search(r'Graph has (\d+) nodes and (\d+)', output)
        if graph_match:
            cache_result.nodes = int(graph_match.group(1))
            cache_result.edges = int(graph_match.group(2))
        
        # Parse timing
        trial_match = re.search(r'(?:Trial|Average) Time:\s+([\d.]+)', output)
        if trial_match:
            cache_result.trial_time = float(trial_match.group(1))
        
        reorder_match = re.search(r'Reorder Time:\s+([\d.]+)', output)
        if reorder_match:
            cache_result.reorder_time = float(reorder_match.group(1))
        
        # Parse from JSON if available
        if json_output and os.path.exists(json_output):
            with open(json_output, 'r') as f:
                data = json.load(f)
            
            cache_result.total_accesses = data.get('total_accesses', 0)
            cache_result.memory_accesses = data.get('memory_accesses', 0)
            
            if 'L1' in data:
                l1 = data['L1']
                cache_result.l1_stats = CacheStats(
                    level='L1',
                    size_bytes=l1.get('size_bytes', 0),
                    ways=l1.get('ways', 0),
                    hits=l1.get('hits', 0),
                    misses=l1.get('misses', 0),
                    hit_rate=l1.get('hit_rate', 0.0),
                    evictions=l1.get('evictions', 0),
                )
            
            if 'L2' in data:
                l2 = data['L2']
                cache_result.l2_stats = CacheStats(
                    level='L2',
                    size_bytes=l2.get('size_bytes', 0),
                    ways=l2.get('ways', 0),
                    hits=l2.get('hits', 0),
                    misses=l2.get('misses', 0),
                    hit_rate=l2.get('hit_rate', 0.0),
                    evictions=l2.get('evictions', 0),
                )
            
            if 'L3' in data:
                l3 = data['L3']
                cache_result.l3_stats = CacheStats(
                    level='L3',
                    size_bytes=l3.get('size_bytes', 0),
                    ways=l3.get('ways', 0),
                    hits=l3.get('hits', 0),
                    misses=l3.get('misses', 0),
                    hit_rate=l3.get('hit_rate', 0.0),
                    evictions=l3.get('evictions', 0),
                )
            
            if cache_result.total_accesses > 0:
                cache_result.overall_hit_rate = (
                    cache_result.total_accesses - cache_result.memory_accesses
                ) / cache_result.total_accesses
        
        else:
            # Parse from stdout if no JSON
            total_match = re.search(r'Total Accesses:\s+(\d+)', output)
            if total_match:
                cache_result.total_accesses = int(total_match.group(1))
            
            mem_match = re.search(r'Memory Accesses:\s+(\d+)', output)
            if mem_match:
                cache_result.memory_accesses = int(mem_match.group(1))
            
            # Parse L1 stats
            l1_match = re.search(
                r'L1 Cache.*?Hits:\s+(\d+).*?Misses:\s+(\d+).*?Hit Rate:\s+([\d.]+)%.*?Evictions:\s+(\d+)',
                output, re.DOTALL
            )
            if l1_match:
                cache_result.l1_stats = CacheStats(
                    level='L1',
                    size_bytes=32768,
                    ways=8,
                    hits=int(l1_match.group(1)),
                    misses=int(l1_match.group(2)),
                    hit_rate=float(l1_match.group(3)) / 100.0,
                    evictions=int(l1_match.group(4)),
                )
        
        return cache_result
        
    except subprocess.TimeoutExpired:
        print(f"{Colors.YELLOW}Timeout: {algorithm} on {graph_name}{Colors.RESET}")
        return None
    except Exception as e:
        print(f"{Colors.RED}Error running {algorithm}: {e}{Colors.RESET}")
        return None


# ============================================================================
# Benchmark Suite
# ============================================================================

def discover_graphs(graphs_dir: str) -> List[Tuple[str, str]]:
    """
    Discover graph files in a directory.
    
    Returns list of (name, path) tuples.
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
                    graphs.append((subdir.name, str(graph_file)))
                    break
    
    return graphs


def run_cache_benchmark_suite(
    algorithms: List[str] = None,
    reorders: List[int] = None,
    graphs: List[Tuple[str, str]] = None,
    synthetic: Dict[str, str] = None,
    output_dir: str = "./cache_results",
    timeout: int = 300,
) -> List[CacheResult]:
    """
    Run comprehensive cache benchmark suite.
    
    Args:
        algorithms: List of algorithms to test (default: all)
        reorders: List of reordering IDs to test
        graphs: List of (name, path) tuples for real graphs
        synthetic: Dict of synthetic graph names to args
        output_dir: Directory for output files
        timeout: Per-run timeout in seconds
    
    Returns:
        List of CacheResult objects
    """
    if algorithms is None:
        algorithms = SIM_ALGORITHMS
    
    if reorders is None:
        reorders = DEFAULT_REORDERS
    
    if synthetic is None:
        synthetic = SYNTHETIC_GRAPHS
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    total_runs = len(algorithms) * len(reorders) * (len(synthetic) + len(graphs or []))
    current_run = 0
    
    print_header("Cache Performance Benchmark Suite")
    print(f"Algorithms: {', '.join(algorithms)}")
    print(f"Reorders: {[ALGORITHMS.get(r, r) for r in reorders]}")
    print(f"Total runs: {total_runs}")
    print()
    
    # Run on synthetic graphs
    if synthetic:
        print_subheader("Synthetic Graphs")
        for graph_name, graph_args in synthetic.items():
            for algo in algorithms:
                for reorder_id in reorders:
                    current_run += 1
                    print(f"[{current_run}/{total_runs}] {algo}/{graph_name}/{ALGORITHMS.get(reorder_id, reorder_id)}... ", end="", flush=True)
                    
                    json_file = os.path.join(output_dir, f"{algo}_{graph_name}_{reorder_id}.json")
                    result = run_cache_simulation(
                        algorithm=algo,
                        graph_args=f"{graph_args} -o {reorder_id}",
                        reorder_id=reorder_id,
                        graph_name=graph_name,
                        timeout=timeout,
                        json_output=json_file
                    )
                    
                    if result:
                        results.append(result)
                        l1_rate = result.l1_stats.hit_rate * 100 if result.l1_stats else 0
                        print(f"{Colors.GREEN}L1: {l1_rate:.1f}% Time: {result.trial_time:.4f}s{Colors.RESET}")
                    else:
                        print(f"{Colors.RED}FAILED{Colors.RESET}")
    
    # Run on real graphs
    if graphs:
        print_subheader("Real Graphs")
        for graph_name, graph_path in graphs:
            for algo in algorithms:
                for reorder_id in reorders:
                    current_run += 1
                    print(f"[{current_run}/{total_runs}] {algo}/{graph_name}/{ALGORITHMS.get(reorder_id, reorder_id)}... ", end="", flush=True)
                    
                    json_file = os.path.join(output_dir, f"{algo}_{graph_name}_{reorder_id}.json")
                    result = run_cache_simulation(
                        algorithm=algo,
                        graph_args=f"-f {graph_path} -s",
                        reorder_id=reorder_id,
                        graph_name=graph_name,
                        timeout=timeout,
                        json_output=json_file
                    )
                    
                    if result:
                        results.append(result)
                        l1_rate = result.l1_stats.hit_rate * 100 if result.l1_stats else 0
                        print(f"{Colors.GREEN}L1: {l1_rate:.1f}% Time: {result.trial_time:.4f}s{Colors.RESET}")
                    else:
                        print(f"{Colors.RED}FAILED{Colors.RESET}")
    
    return results


# ============================================================================
# Correlation Analysis
# ============================================================================

def compute_correlations(results: List[CacheResult]) -> Dict[str, Any]:
    """
    Compute correlations between cache performance and execution time.
    
    Returns correlation data for analysis.
    """
    correlations = {
        'by_algorithm': {},
        'by_graph': {},
        'by_reorder': {},
        'feature_correlations': {},
    }
    
    # Group by algorithm
    by_algo = defaultdict(list)
    for r in results:
        by_algo[r.algorithm].append(r)
    
    for algo, algo_results in by_algo.items():
        l1_rates = [r.l1_stats.hit_rate if r.l1_stats else 0 for r in algo_results]
        times = [r.trial_time for r in algo_results]
        
        if len(l1_rates) > 1:
            # Simple correlation
            mean_l1 = sum(l1_rates) / len(l1_rates)
            mean_time = sum(times) / len(times)
            
            cov = sum((l1_rates[i] - mean_l1) * (times[i] - mean_time) for i in range(len(l1_rates)))
            std_l1 = (sum((x - mean_l1)**2 for x in l1_rates) / len(l1_rates)) ** 0.5
            std_time = (sum((x - mean_time)**2 for x in times) / len(times)) ** 0.5
            
            corr = cov / (std_l1 * std_time * len(l1_rates)) if std_l1 > 0 and std_time > 0 else 0
            
            correlations['by_algorithm'][algo] = {
                'l1_time_correlation': corr,
                'avg_l1_hit_rate': mean_l1,
                'avg_time': mean_time,
                'samples': len(algo_results),
            }
    
    # Group by reorder
    by_reorder = defaultdict(list)
    for r in results:
        by_reorder[r.reorder_name].append(r)
    
    for reorder, reorder_results in by_reorder.items():
        l1_rates = [r.l1_stats.hit_rate if r.l1_stats else 0 for r in reorder_results]
        l2_rates = [r.l2_stats.hit_rate if r.l2_stats else 0 for r in reorder_results]
        l3_rates = [r.l3_stats.hit_rate if r.l3_stats else 0 for r in reorder_results]
        times = [r.trial_time for r in reorder_results]
        
        correlations['by_reorder'][reorder] = {
            'avg_l1_hit_rate': sum(l1_rates) / len(l1_rates) if l1_rates else 0,
            'avg_l2_hit_rate': sum(l2_rates) / len(l2_rates) if l2_rates else 0,
            'avg_l3_hit_rate': sum(l3_rates) / len(l3_rates) if l3_rates else 0,
            'avg_time': sum(times) / len(times) if times else 0,
            'samples': len(reorder_results),
        }
    
    return correlations


def generate_perceptron_features(results: List[CacheResult]) -> Dict[str, Any]:
    """
    Generate perceptron feature data from cache results.
    
    Returns feature vectors suitable for perceptron training.
    """
    features = {
        'feature_names': [
            'l1_hit_rate',
            'l2_hit_rate', 
            'l3_hit_rate',
            'dram_access_rate',
            'l1_eviction_rate',
            'l2_eviction_rate',
            'l3_eviction_rate',
        ],
        'samples': [],
    }
    
    for r in results:
        sample = {
            'graph': r.graph_name,
            'algorithm': r.algorithm,
            'reorder': r.reorder_name,
            'reorder_id': r.reorder_id,
            'features': r.get_feature_vector(),
            'time': r.trial_time,
        }
        features['samples'].append(sample)
    
    return features


# ============================================================================
# Reporting
# ============================================================================

def print_summary_report(results: List[CacheResult], correlations: Dict):
    """Print a summary report of cache performance."""
    print()
    print_header("Cache Performance Summary")
    
    # By reorder algorithm
    print_subheader("By Reordering Algorithm")
    print(f"{'Algorithm':<20} {'L1 Hit%':>10} {'L2 Hit%':>10} {'L3 Hit%':>10} {'Avg Time':>12}")
    print("-" * 64)
    
    for reorder, stats in sorted(correlations['by_reorder'].items()):
        print(f"{reorder:<20} {stats['avg_l1_hit_rate']*100:>9.2f}% {stats['avg_l2_hit_rate']*100:>9.2f}% "
              f"{stats['avg_l3_hit_rate']*100:>9.2f}% {stats['avg_time']:>11.4f}s")
    
    # By benchmark algorithm
    print()
    print_subheader("By Benchmark Algorithm")
    print(f"{'Algorithm':<15} {'L1-Time Corr':>12} {'Avg L1 Hit%':>12} {'Avg Time':>12} {'Samples':>8}")
    print("-" * 61)
    
    for algo, stats in sorted(correlations['by_algorithm'].items()):
        print(f"{algo:<15} {stats['l1_time_correlation']:>12.4f} {stats['avg_l1_hit_rate']*100:>11.2f}% "
              f"{stats['avg_time']:>11.4f}s {stats['samples']:>8}")


def export_results(
    results: List[CacheResult],
    correlations: Dict,
    features: Dict,
    output_dir: str
):
    """Export results to JSON files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export raw results
    results_file = os.path.join(output_dir, f"cache_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    print(f"Results exported to: {results_file}")
    
    # Export correlations
    corr_file = os.path.join(output_dir, f"cache_correlations_{timestamp}.json")
    with open(corr_file, 'w') as f:
        json.dump(correlations, f, indent=2)
    print(f"Correlations exported to: {corr_file}")
    
    # Export perceptron features
    features_file = os.path.join(output_dir, f"cache_features_{timestamp}.json")
    with open(features_file, 'w') as f:
        json.dump(features, f, indent=2)
    print(f"Features exported to: {features_file}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cache Performance Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test with small synthetic graphs
    python3 scripts/analysis/cache_benchmark.py --quick
    
    # Full benchmark with all algorithms
    python3 scripts/analysis/cache_benchmark.py --graphs-dir ./graphs
    
    # Specific algorithms and reorders
    python3 scripts/analysis/cache_benchmark.py --algorithms pr bfs --reorders 0 7 12 20
    
    # Custom output directory
    python3 scripts/analysis/cache_benchmark.py --output ./my_results
        """
    )
    
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick mode: fewer graphs and reorders'
    )
    parser.add_argument(
        '--algorithms', nargs='+', default=None,
        help=f'Algorithms to benchmark (default: all). Options: {SIM_ALGORITHMS}'
    )
    parser.add_argument(
        '--reorders', nargs='+', type=int, default=None,
        help=f'Reordering algorithms to test (default: {DEFAULT_REORDERS})'
    )
    parser.add_argument(
        '--graphs-dir', type=str, default='./graphs',
        help='Directory containing graph files'
    )
    parser.add_argument(
        '--synthetic-only', action='store_true',
        help='Only run on synthetic graphs'
    )
    parser.add_argument(
        '--output', type=str, default='./cache_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--timeout', type=int, default=300,
        help='Timeout per benchmark run (seconds)'
    )
    
    args = parser.parse_args()
    
    # Configure based on mode
    if args.quick:
        algorithms = args.algorithms or ['pr', 'bfs']
        reorders = args.reorders or QUICK_REORDERS
        synthetic = QUICK_SYNTHETIC
    else:
        algorithms = args.algorithms or SIM_ALGORITHMS
        reorders = args.reorders or DEFAULT_REORDERS
        synthetic = SYNTHETIC_GRAPHS
    
    # Discover graphs
    graphs = []
    if not args.synthetic_only:
        graphs = discover_graphs(args.graphs_dir)
        if graphs:
            print(f"Found {len(graphs)} graphs in {args.graphs_dir}")
    
    # Run benchmarks
    results = run_cache_benchmark_suite(
        algorithms=algorithms,
        reorders=reorders,
        graphs=graphs,
        synthetic=synthetic,
        output_dir=args.output,
        timeout=args.timeout,
    )
    
    if not results:
        print(f"{Colors.RED}No results collected!{Colors.RESET}")
        return 1
    
    # Compute correlations
    correlations = compute_correlations(results)
    
    # Generate perceptron features
    features = generate_perceptron_features(results)
    
    # Print summary
    print_summary_report(results, correlations)
    
    # Export results
    print()
    print_subheader("Exporting Results")
    export_results(results, correlations, features, args.output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
