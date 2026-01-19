#!/usr/bin/env python3
"""
Common utilities for GraphBrew benchmark scripts.

This module provides shared functionality for:
- Running benchmarks with different algorithms
- Parsing benchmark output
- Graph feature extraction
- Result aggregation
"""

import subprocess
import re
import os
import json
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

# ============================================================================
# Algorithm Definitions
# ============================================================================

ALGORITHMS = {
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
    13: "GraphBrewOrder",
    # 14: MAP - requires external file
    15: "AdaptiveOrder",
    16: "LeidenDFS",
    17: "LeidenDFSHub",
    18: "LeidenDFSSize",
    19: "LeidenBFS",
    20: "LeidenHybrid",
}

# Algorithms that are fast to run (for quick tests)
QUICK_ALGORITHMS = {0, 7, 8, 12, 15, 20}

# Community-based algorithms
COMMUNITY_ALGORITHMS = {8, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20}

# Leiden family
LEIDEN_ALGORITHMS = {12, 16, 17, 18, 19, 20}

# Benchmarks that need multiple source nodes for stable results
MULTI_SOURCE_BENCHMARKS = {"bfs", "sssp", "bc"}

# Default number of source nodes for multi-source benchmarks
DEFAULT_SOURCE_TRIALS = 16

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    algorithm_id: int
    algorithm_name: str
    graph_name: str
    benchmark: str
    
    # Timing results
    reorder_time: float = 0.0
    trial_time: float = 0.0
    total_time: float = 0.0
    
    # Graph statistics
    nodes: int = 0
    edges: int = 0
    avg_degree: float = 0.0
    
    # Optional fields
    iterations: int = 0  # For PageRank
    verified: bool = False
    speedup: float = 1.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class GraphFeatures:
    """Features extracted from a graph for ML/correlation analysis."""
    name: str
    nodes: int
    edges: int
    avg_degree: float
    density: float
    log_nodes: float
    log_edges: float
    degree_variance: float = 0.0
    hub_concentration: float = 0.0
    modularity: float = 0.0
    num_communities: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_feature_vector(self) -> List[float]:
        """Return features as a vector for ML models."""
        return [
            self.modularity,
            self.log_nodes,
            self.log_edges,
            self.density,
            self.avg_degree,
            self.degree_variance,
            self.hub_concentration,
        ]


# ============================================================================
# Benchmark Execution
# ============================================================================

def run_benchmark(
    binary: str,
    graph_args: str,
    algo_id: int,
    num_trials: int = 1,
    verify: bool = False,
    timeout: int = 600,
    extra_args: str = ""
) -> Tuple[Optional[Dict], str]:
    """
    Run a benchmark with specified algorithm.
    
    Args:
        binary: Path to benchmark binary (e.g., "./bench/bin/pr")
        graph_args: Graph specification (e.g., "-g 20" or "-f graph.mtx -s")
        algo_id: Algorithm ID (0-20)
        num_trials: Number of trials to run
        verify: Whether to verify results
        timeout: Timeout in seconds
        extra_args: Additional arguments
    
    Returns:
        Tuple of (parsed_results dict, raw_output string)
    """
    verify_flag = "-v" if verify else ""
    cmd = f"{binary} {graph_args} -o {algo_id} -n {num_trials} {verify_flag} {extra_args}"
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        output = result.stdout + result.stderr
        parsed = parse_benchmark_output(output)
        return parsed, output
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    except Exception as e:
        return None, str(e)


def parse_benchmark_output(output: str) -> Dict[str, Any]:
    """Parse benchmark output into structured data."""
    result = {}
    
    # Graph statistics
    # Format: "Graph has X nodes and Y undirected edges for degree: Z"
    graph_match = re.search(r'Graph has (\d+) nodes and (\d+) (?:undirected |directed )?edges', output)
    if graph_match:
        result['nodes'] = int(graph_match.group(1))
        result['edges'] = int(graph_match.group(2))
    
    degree_match = re.search(r'for degree:\s*(\d+)', output)
    if degree_match:
        result['avg_degree'] = int(degree_match.group(1))
    
    # Timing results
    patterns = {
        'generate_time': r'Generate Time:\s+([\d.]+)',
        'build_time': r'Build Time:\s+([\d.]+)',
        'reorder_time': r'Reorder Time:\s+([\d.]+)',
        'relabel_time': r'Relabel.*Time:\s+([\d.]+)',
        'trial_time': r'Trial Time:\s+([\d.]+)',
        'average_time': r'Average Time:\s+([\d.]+)',
        'total_time': r'Total Time:\s+([\d.]+)',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            result[key] = float(match.group(1))
    
    # PageRank iterations
    iter_match = re.search(r'(\d+)\s+iterations', output)
    if iter_match:
        result['iterations'] = int(iter_match.group(1))
    
    # Verification
    result['verified'] = 'Verification:           PASS' in output or 'Verification: PASS' in output
    
    # Modularity (if printed)
    mod_match = re.search(r'[Mm]odularity[:\s]+([\d.]+)', output)
    if mod_match:
        result['modularity'] = float(mod_match.group(1))
    
    # Number of communities
    comm_match = re.search(r'(\d+)\s+communities', output)
    if comm_match:
        result['num_communities'] = int(comm_match.group(1))
    
    return result


def run_multi_source_benchmark(
    binary: str,
    graph_args: str,
    algo_id: int,
    num_sources: int = DEFAULT_SOURCE_TRIALS,
    timeout: int = 600
) -> Tuple[Optional[Dict], str]:
    """
    Run benchmark with multiple random source nodes (for BFS, SSSP, BC).
    
    This provides more stable timing results by averaging over many sources.
    """
    cmd = f"{binary} {graph_args} -o {algo_id} -n {num_sources}"
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        output = result.stdout + result.stderr
        parsed = parse_benchmark_output(output)
        return parsed, output
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    except Exception as e:
        return None, str(e)


# ============================================================================
# Graph Feature Extraction
# ============================================================================

def extract_graph_features(
    graph_args: str,
    binary: str = "./bench/bin/pr",
    timeout: int = 300
) -> Optional[GraphFeatures]:
    """
    Extract features from a graph by running a quick benchmark.
    
    Uses ORIGINAL ordering with 1 trial to get graph statistics.
    """
    parsed, output = run_benchmark(binary, graph_args, algo_id=0, num_trials=1, timeout=timeout)
    
    if not parsed:
        return None
    
    nodes = parsed.get('nodes', 0)
    edges = parsed.get('edges', 0)
    
    if nodes == 0:
        return None
    
    avg_degree = edges / nodes if nodes > 0 else 0
    max_edges = nodes * (nodes - 1) / 2
    density = edges / max_edges if max_edges > 0 else 0
    
    return GraphFeatures(
        name=graph_args,
        nodes=nodes,
        edges=edges,
        avg_degree=avg_degree,
        density=density,
        log_nodes=math.log10(nodes) if nodes > 0 else 0,
        log_edges=math.log10(edges) if edges > 0 else 0,
        modularity=parsed.get('modularity', 0),
        num_communities=parsed.get('num_communities', 0),
    )


def compute_degree_stats(graph_args: str, binary: str = "./bench/bin/bfs") -> Dict[str, float]:
    """
    Compute degree distribution statistics.
    
    Note: This requires a modified binary that outputs degree stats,
    or we estimate from available data.
    """
    # For now, return estimates based on graph type
    # TODO: Add actual degree extraction
    return {
        'degree_variance': 0.0,
        'hub_concentration': 0.0,
    }


# ============================================================================
# Result Aggregation
# ============================================================================

def compute_speedups(
    results: List[BenchmarkResult],
    baseline_algo: int = 0
) -> List[BenchmarkResult]:
    """
    Compute speedups relative to baseline algorithm.
    
    Args:
        results: List of benchmark results
        baseline_algo: Algorithm ID to use as baseline (default: ORIGINAL)
    
    Returns:
        Updated results with speedup field populated
    """
    # Group by graph and benchmark
    grouped = {}
    for r in results:
        key = (r.graph_name, r.benchmark)
        if key not in grouped:
            grouped[key] = {}
        grouped[key][r.algorithm_id] = r
    
    # Compute speedups
    for key, algos in grouped.items():
        if baseline_algo in algos:
            baseline_time = algos[baseline_algo].trial_time
            if baseline_time > 0:
                for algo_id, result in algos.items():
                    result.speedup = baseline_time / result.trial_time if result.trial_time > 0 else 0
    
    return results


def find_best_algorithm(
    results: List[BenchmarkResult],
    metric: str = "trial_time"
) -> Dict[str, Tuple[int, str, float]]:
    """
    Find the best algorithm for each graph based on specified metric.
    
    Args:
        results: List of benchmark results
        metric: Metric to optimize ("trial_time", "total_time", "speedup")
    
    Returns:
        Dict mapping graph_name to (best_algo_id, algo_name, metric_value)
    """
    # Group by graph
    grouped = {}
    for r in results:
        if r.graph_name not in grouped:
            grouped[r.graph_name] = []
        grouped[r.graph_name].append(r)
    
    best = {}
    for graph_name, graph_results in grouped.items():
        if metric == "speedup":
            # Maximize speedup
            best_result = max(graph_results, key=lambda x: x.speedup)
        else:
            # Minimize time
            best_result = min(graph_results, key=lambda x: getattr(x, metric, float('inf')))
        
        best[graph_name] = (
            best_result.algorithm_id,
            best_result.algorithm_name,
            getattr(best_result, metric)
        )
    
    return best


# ============================================================================
# File I/O
# ============================================================================

def save_results_json(results: List[BenchmarkResult], filepath: str):
    """Save benchmark results to JSON file."""
    data = [r.to_dict() for r in results]
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_results_json(filepath: str) -> List[BenchmarkResult]:
    """Load benchmark results from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return [BenchmarkResult(**d) for d in data]


def save_results_csv(results: List[BenchmarkResult], filepath: str):
    """Save benchmark results to CSV file."""
    import csv
    
    if not results:
        return
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fieldnames = list(results[0].to_dict().keys())
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())


# ============================================================================
# Terminal Output Formatting
# ============================================================================

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(title: str, width: int = 70):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{'='*width}{Colors.RESET}")
    print(f"{Colors.BOLD}{title.center(width)}{Colors.RESET}")
    print(f"{'='*width}")


def print_subheader(title: str, width: int = 70):
    """Print a formatted subheader."""
    print(f"\n{Colors.CYAN}{'-'*width}{Colors.RESET}")
    print(f"{Colors.CYAN}{title}{Colors.RESET}")
    print(f"{'-'*width}")


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 0.001:
        return f"{seconds*1000000:.1f}µs"
    elif seconds < 1:
        return f"{seconds*1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.3f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"


def format_speedup(speedup: float) -> str:
    """Format speedup with color coding."""
    if speedup >= 2.0:
        return f"{Colors.GREEN}{speedup:.2f}x{Colors.RESET}"
    elif speedup >= 1.0:
        return f"{Colors.YELLOW}{speedup:.2f}x{Colors.RESET}"
    else:
        return f"{Colors.RED}{speedup:.2f}x{Colors.RESET}"
