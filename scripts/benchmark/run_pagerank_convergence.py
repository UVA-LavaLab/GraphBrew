#!/usr/bin/env python3
"""
PageRank Convergence Analysis

Analyzes the relationship between graph reordering and PageRank convergence:
- Number of iterations until convergence for each reordering technique
- Correlation between reordering and iteration count
- Effect on total computation time

Usage:
    python3 scripts/benchmark/run_pagerank_convergence.py [--graphs-dir DIR]
    
Examples:
    python3 scripts/benchmark/run_pagerank_convergence.py --quick
    python3 scripts/benchmark/run_pagerank_convergence.py --graphs-dir ./graphs
"""

import os
import sys
import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.common import (
    ALGORITHMS, Colors, print_header, print_subheader,
    format_time, format_speedup
)

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_TOLERANCE = 1e-4
DEFAULT_MAX_ITERATIONS = 100

# Synthetic graphs for quick testing
SYNTHETIC_GRAPHS = {
    "rmat_14": "-g 14",
    "rmat_16": "-g 16",
    "rmat_18": "-g 18",
}

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PRConvergenceResult:
    """Results from a PageRank convergence test."""
    algorithm_id: int
    algorithm_name: str
    graph_name: str
    iterations: int
    converged: bool
    trial_time: float
    avg_iter_time: float
    tolerance: float
    
    def to_dict(self) -> Dict:
        return {
            'algorithm_id': self.algorithm_id,
            'algorithm_name': self.algorithm_name,
            'graph_name': self.graph_name,
            'iterations': self.iterations,
            'converged': self.converged,
            'trial_time': self.trial_time,
            'avg_iter_time': self.avg_iter_time,
            'tolerance': self.tolerance,
        }


# ============================================================================
# Benchmark Functions
# ============================================================================

def run_pagerank_convergence(
    graph_args: str,
    algo_id: int,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    timeout: int = 300
) -> Optional[PRConvergenceResult]:
    """
    Run PageRank and capture convergence information.
    """
    import subprocess
    
    binary = "./bench/bin/pr"
    # Use -l flag to enable logging which prints iteration info
    cmd = f"{binary} {graph_args} -o {algo_id} -n 1 -t {tolerance} -i {max_iterations} -l"
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        output = result.stdout + result.stderr
        
        # Parse output
        iterations = 0
        trial_time = 0.0
        converged = False
        
        # Count iteration lines: format is "   N   error" where N is iteration number
        # Pattern matches lines like "    0                1.04832"
        iter_lines = re.findall(r'^\s*(\d+)\s+[\d.]+\s*$', output, re.MULTILINE)
        if iter_lines:
            iterations = len(iter_lines)
            converged = iterations < max_iterations
        
        # Look for trial time
        time_match = re.search(r'(?:Trial|Average) Time:\s+([\d.]+)', output)
        if time_match:
            trial_time = float(time_match.group(1))
        
        avg_iter_time = trial_time / iterations if iterations > 0 else trial_time
        
        return PRConvergenceResult(
            algorithm_id=algo_id,
            algorithm_name=ALGORITHMS.get(algo_id, f"Unknown({algo_id})"),
            graph_name=graph_args,
            iterations=iterations,
            converged=converged,
            trial_time=trial_time,
            avg_iter_time=avg_iter_time,
            tolerance=tolerance,
        )
        
    except Exception as e:
        return None


def analyze_convergence(
    graphs: Dict[str, str],
    algorithms: Dict[int, str] = None,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    verbose: bool = True
) -> List[PRConvergenceResult]:
    """
    Analyze PageRank convergence for all graphs and algorithms.
    """
    if algorithms is None:
        algorithms = ALGORITHMS
    
    all_results = []
    
    print_header("PageRank Convergence Analysis")
    print(f"Tolerance: {tolerance}")
    print(f"Max iterations: {max_iterations}")
    
    for graph_name, graph_args in graphs.items():
        print_subheader(f"Graph: {graph_name}")
        print(f"{'Algorithm':<18} {'Iterations':>12} {'Time':>12} {'Time/Iter':>12} {'Status':<10}")
        print("-" * 70)
        
        for algo_id, algo_name in sorted(algorithms.items()):
            result = run_pagerank_convergence(
                graph_args=graph_args,
                algo_id=algo_id,
                tolerance=tolerance,
                max_iterations=max_iterations
            )
            
            if result:
                result.graph_name = graph_name
                all_results.append(result)
                
                status = f"{Colors.GREEN}✓{Colors.RESET}" if result.converged else f"{Colors.RED}✗{Colors.RESET}"
                print(f"{algo_name:<18} "
                      f"{result.iterations:>12} "
                      f"{format_time(result.trial_time):>12} "
                      f"{format_time(result.avg_iter_time):>12} "
                      f"{status}")
            else:
                print(f"{algo_name:<18} {Colors.YELLOW}SKIPPED{Colors.RESET}")
    
    return all_results


def compute_correlation(results: List[PRConvergenceResult]) -> Dict[str, float]:
    """
    Compute correlation between reordering and iteration count.
    """
    from collections import defaultdict
    
    # Group by graph
    by_graph = defaultdict(list)
    for r in results:
        by_graph[r.graph_name].append(r)
    
    correlations = {}
    
    for graph_name, graph_results in by_graph.items():
        # Sort by iteration count
        sorted_results = sorted(graph_results, key=lambda x: x.iterations)
        
        if len(sorted_results) >= 2:
            min_iters = sorted_results[0].iterations
            max_iters = sorted_results[-1].iterations
            
            if max_iters > min_iters:
                # Compute relative variation
                variation = (max_iters - min_iters) / min_iters
                correlations[graph_name] = variation
    
    return correlations


def print_convergence_summary(results: List[PRConvergenceResult]):
    """Print summary of convergence analysis."""
    from collections import defaultdict
    
    print_header("Convergence Summary")
    
    # Group by graph
    by_graph = defaultdict(list)
    for r in results:
        by_graph[r.graph_name].append(r)
    
    print(f"\n{'Graph':<20} {'Min Iters':>12} {'Max Iters':>12} {'Best Algorithm':<20}")
    print("-" * 70)
    
    algo_best_count = defaultdict(int)
    
    for graph_name, graph_results in sorted(by_graph.items()):
        if not graph_results:
            continue
        
        min_result = min(graph_results, key=lambda x: x.iterations)
        max_result = max(graph_results, key=lambda x: x.iterations)
        
        print(f"{graph_name:<20} "
              f"{min_result.iterations:>12} "
              f"{max_result.iterations:>12} "
              f"{min_result.algorithm_name:<20}")
        
        algo_best_count[min_result.algorithm_name] += 1
    
    # Print algorithm that achieves fastest convergence most often
    print(f"\n{Colors.BOLD}Algorithms with fastest convergence:{Colors.RESET}")
    for algo, count in sorted(algo_best_count.items(), key=lambda x: -x[1]):
        print(f"  {algo}: {count} graphs")
    
    # Compute and print correlations
    correlations = compute_correlation(results)
    if correlations:
        print(f"\n{Colors.BOLD}Iteration count variation by graph:{Colors.RESET}")
        for graph_name, variation in sorted(correlations.items(), key=lambda x: -x[1]):
            print(f"  {graph_name}: {variation:.1%} variation")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze PageRank convergence with different reorderings"
    )
    parser.add_argument(
        "--graphs-dir", "-g",
        default=None,
        help="Directory containing graphs"
    )
    parser.add_argument(
        "--graphs-config",
        default=None,
        help="JSON config file with graph definitions"
    )
    parser.add_argument(
        "--algorithms", "-a",
        default=None,
        help="Comma-separated list of algorithm IDs (default: all)"
    )
    parser.add_argument(
        "--tolerance", "-t",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"Convergence tolerance (default: {DEFAULT_TOLERANCE})"
    )
    parser.add_argument(
        "--max-iterations", "-i",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"Maximum iterations (default: {DEFAULT_MAX_ITERATIONS})"
    )
    parser.add_argument(
        "--output", "-o",
        default="./bench/results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick test with synthetic graphs"
    )
    
    args = parser.parse_args()
    
    # Determine graphs to use
    if args.quick:
        graphs = SYNTHETIC_GRAPHS
    elif args.graphs_config:
        with open(args.graphs_config, 'r') as f:
            config = json.load(f)
        graphs = {name: info.get("args", f"-f {info['path']} -s")
                  for name, info in config.get("graphs", {}).items()}
    elif args.graphs_dir:
        # Load from directory
        graphs = {}
        config_path = os.path.join(args.graphs_dir, "graphs.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            for name, info in config.get("graphs", {}).items():
                if "args" in info:
                    graphs[name] = info["args"]
                elif "path" in info:
                    sym = "-s" if info.get("symmetric", True) else ""
                    graphs[name] = f"-f {info['path']} {sym}"
    else:
        graphs = {"rmat_14": "-g 14", "rmat_16": "-g 16"}
    
    # Determine algorithms
    if args.algorithms:
        algo_ids = [int(x.strip()) for x in args.algorithms.split(',')]
        algorithms = {aid: ALGORITHMS.get(aid, f"Unknown({aid})") for aid in algo_ids}
    else:
        algorithms = ALGORITHMS
    
    # Run analysis
    results = analyze_convergence(
        graphs=graphs,
        algorithms=algorithms,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations
    )
    
    # Print summary
    print_convergence_summary(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output, exist_ok=True)
    
    json_path = os.path.join(args.output, f"pr_convergence_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    
    print(f"\nResults saved to: {json_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
