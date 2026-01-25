#!/usr/bin/env python3
"""
Cache simulation utilities for GraphBrew.

Runs cache simulations to analyze memory access patterns for different reordering algorithms.
Can be used standalone or as a library.

Standalone usage:
    python -m scripts.lib.cache --graph graphs/email-Enron/email-Enron.mtx
    python -m scripts.lib.cache --graph test.mtx --algorithms 0,8,9 --benchmarks pr,bfs

Library usage:
    from scripts.lib.cache import run_cache_simulations, parse_cache_output
    
    results = run_cache_simulations(graphs, algorithms=[0, 8], benchmarks=["pr", "bfs"])
"""

import os
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from .utils import (
    PROJECT_ROOT, BIN_SIM_DIR, RESULTS_DIR,
    ALGORITHMS, Logger, run_command,
)
from .reorder import get_label_map_path

# Initialize logger
log = Logger()

# =============================================================================
# Constants
# =============================================================================

# Default timeouts (seconds)
TIMEOUT_SIM = 1200          # 20 min for simulations
TIMEOUT_SIM_HEAVY = 3600    # 1 hour for heavy simulations

# Heavy benchmarks (computationally intensive)
HEAVY_SIM_BENCHMARKS = {"bc", "sssp"}

# Graph size threshold (MB)
SIZE_MEDIUM = 500


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CacheResult:
    """Result from cache simulation."""
    graph: str
    algorithm_id: int
    algorithm_name: str
    benchmark: str
    l1_hit_rate: float = 0.0
    l2_hit_rate: float = 0.0
    l3_hit_rate: float = 0.0
    l1_misses: int = 0
    l2_misses: int = 0
    l3_misses: int = 0
    success: bool = True
    error: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class GraphInfo:
    """Basic information about a graph for simulations."""
    name: str
    path: str
    size_mb: float = 0.0
    is_symmetric: bool = True


# =============================================================================
# Output Parsing
# =============================================================================

def parse_cache_output(output: str) -> Dict:
    """
    Parse cache simulation output for hit rates and miss counts.
    
    Args:
        output: Stdout from cache simulation
        
    Returns:
        Dict with l1_hit_rate, l2_hit_rate, l3_hit_rate, and miss counts
    """
    result = {
        'l1_hit_rate': 0.0,
        'l2_hit_rate': 0.0,
        'l3_hit_rate': 0.0,
        'l1_misses': 0,
        'l2_misses': 0,
        'l3_misses': 0,
    }
    
    # Cache hit rate patterns
    # Format: "L1 Hit Rate: 95.5%" or "L1 hit rate: 95.5"
    patterns = {
        'l1_hit_rate': r'L1\s*(?:Hit\s*)?Rate[:\s]*([\d.]+)',
        'l2_hit_rate': r'L2\s*(?:Hit\s*)?Rate[:\s]*([\d.]+)',
        'l3_hit_rate': r'L3\s*(?:Hit\s*)?Rate[:\s]*([\d.]+)',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            result[key] = float(match.group(1))
    
    # Cache miss patterns
    miss_patterns = {
        'l1_misses': r'L1\s*(?:Cache\s*)?Misses[:\s]*(\d+)',
        'l2_misses': r'L2\s*(?:Cache\s*)?Misses[:\s]*(\d+)',
        'l3_misses': r'L3\s*(?:Cache\s*)?Misses[:\s]*(\d+)',
    }
    
    for key, pattern in miss_patterns.items():
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            result[key] = int(match.group(1))
    
    return result


# =============================================================================
# Core Simulation Functions
# =============================================================================

def run_cache_simulation(
    benchmark: str,
    graph_path: str,
    algorithm: int = 0,
    label_map_path: str = None,
    symmetric: bool = True,
    timeout: int = TIMEOUT_SIM,
    bin_sim_dir: str = None
) -> CacheResult:
    """
    Run a single cache simulation.
    
    Args:
        benchmark: Benchmark name (pr, bfs, cc, etc.)
        graph_path: Path to graph file
        algorithm: Algorithm ID
        label_map_path: Optional path to pre-generated label map
        symmetric: Use symmetric graph flag
        timeout: Timeout in seconds
        bin_sim_dir: Directory containing simulation binaries
        
    Returns:
        CacheResult with hit rates and miss counts
    """
    if bin_sim_dir is None:
        bin_sim_dir = str(BIN_SIM_DIR)
    
    graph_path = Path(graph_path)
    graph_name = graph_path.stem
    algo_name = ALGORITHMS.get(algorithm, f"ALGO_{algorithm}")
    
    binary = Path(bin_sim_dir) / benchmark
    if not binary.exists():
        return CacheResult(
            graph=graph_name,
            algorithm_id=algorithm,
            algorithm_name=algo_name,
            benchmark=benchmark,
            success=False,
            error=f"Binary not found: {binary}"
        )
    
    # Build command
    sym_flag = "-s" if symmetric else ""
    if label_map_path and os.path.exists(label_map_path) and algorithm != 0:
        # Use pre-generated mapping via MAP (algo 13)
        cmd = f"{binary} -f {graph_path} {sym_flag} -o 13:{label_map_path} -n 1"
    else:
        cmd = f"{binary} -f {graph_path} {sym_flag} -o {algorithm} -n 1"
    
    # Run simulation
    success, stdout, stderr = run_command(cmd, timeout)
    
    if success:
        output = stdout + stderr
        parsed = parse_cache_output(output)
        
        return CacheResult(
            graph=graph_name,
            algorithm_id=algorithm,
            algorithm_name=algo_name,
            benchmark=benchmark,
            l1_hit_rate=parsed['l1_hit_rate'],
            l2_hit_rate=parsed['l2_hit_rate'],
            l3_hit_rate=parsed['l3_hit_rate'],
            l1_misses=parsed['l1_misses'],
            l2_misses=parsed['l2_misses'],
            l3_misses=parsed['l3_misses'],
            success=True
        )
    else:
        error = "TIMEOUT" if "TIMEOUT" in stderr else stderr[:100]
        return CacheResult(
            graph=graph_name,
            algorithm_id=algorithm,
            algorithm_name=algo_name,
            benchmark=benchmark,
            success=False,
            error=error
        )


def run_cache_simulations(
    graphs: List[GraphInfo],
    algorithms: List[int],
    benchmarks: List[str],
    bin_sim_dir: str = None,
    timeout: int = TIMEOUT_SIM,
    skip_heavy: bool = False,
    label_maps: Dict[str, Dict[str, str]] = None
) -> List[CacheResult]:
    """
    Run cache simulations for all combinations.
    
    Args:
        graphs: List of graphs to process
        algorithms: List of algorithm IDs
        benchmarks: List of benchmark names
        bin_sim_dir: Directory containing simulation binaries
        timeout: Timeout per simulation
        skip_heavy: Skip heavy benchmarks on large graphs
        label_maps: Optional pre-generated label maps
        
    Returns:
        List of CacheResult with hit rates and miss counts
    """
    if bin_sim_dir is None:
        bin_sim_dir = str(BIN_SIM_DIR)
    
    log.info(f"Running cache simulations: {len(graphs)} graphs × {len(algorithms)} algorithms × {len(benchmarks)} benchmarks")
    
    results = []
    total = len(graphs) * len(algorithms) * len(benchmarks)
    current = 0
    
    for graph in graphs:
        log.info(f"Graph: {graph.name} ({graph.size_mb:.1f}MB)")
        
        for bench in benchmarks:
            binary = os.path.join(bin_sim_dir, bench)
            
            if not os.path.exists(binary):
                log.warning(f"  Binary not found: {binary}")
                continue
            
            # Use longer timeout for heavy benchmarks
            bench_timeout = TIMEOUT_SIM_HEAVY if bench in HEAVY_SIM_BENCHMARKS else timeout
            
            # Skip heavy simulations on large graphs if requested
            if skip_heavy and bench in HEAVY_SIM_BENCHMARKS and graph.size_mb > SIZE_MEDIUM:
                log.info(f"  {bench.upper()}: SKIPPED (heavy on large graph)")
                for algo_id in algorithms:
                    current += 1
                    results.append(CacheResult(
                        graph=graph.name,
                        algorithm_id=algo_id,
                        algorithm_name=ALGORITHMS.get(algo_id, f"ALGO_{algo_id}"),
                        benchmark=bench,
                        success=False,
                        error="SKIPPED"
                    ))
                continue
            
            log.info(f"  {bench.upper()} (simulation):")
            
            for algo_id in algorithms:
                current += 1
                algo_name = ALGORITHMS.get(algo_id, f"ALGO_{algo_id}")
                
                # Check for pre-generated label map
                label_map_path = None
                if label_maps and algo_id != 0:
                    label_map_path = get_label_map_path(label_maps, graph.name, algo_name)
                
                result = run_cache_simulation(
                    benchmark=bench,
                    graph_path=graph.path,
                    algorithm=algo_id,
                    label_map_path=label_map_path,
                    symmetric=graph.is_symmetric,
                    timeout=bench_timeout,
                    bin_sim_dir=bin_sim_dir
                )
                
                if result.success:
                    log.info(f"    [{current}/{total}] {algo_name}: L1:{result.l1_hit_rate:.1f}% L2:{result.l2_hit_rate:.1f}% L3:{result.l3_hit_rate:.1f}%")
                else:
                    log.error(f"    [{current}/{total}] {algo_name}: {result.error}")
                
                results.append(result)
    
    return results


def get_cache_stats_summary(results: List[CacheResult]) -> Dict:
    """
    Compute summary statistics from cache results.
    
    Args:
        results: List of CacheResult objects
        
    Returns:
        Dict with summary statistics
    """
    successful = [r for r in results if r.success]
    
    if not successful:
        return {"total": len(results), "successful": 0}
    
    # Group by algorithm
    by_algo = {}
    for r in successful:
        if r.algorithm_name not in by_algo:
            by_algo[r.algorithm_name] = []
        by_algo[r.algorithm_name].append(r)
    
    # Compute averages
    algo_stats = {}
    for algo, algo_results in by_algo.items():
        algo_stats[algo] = {
            "count": len(algo_results),
            "avg_l1_hit_rate": sum(r.l1_hit_rate for r in algo_results) / len(algo_results),
            "avg_l2_hit_rate": sum(r.l2_hit_rate for r in algo_results) / len(algo_results),
            "avg_l3_hit_rate": sum(r.l3_hit_rate for r in algo_results) / len(algo_results),
        }
    
    return {
        "total": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "by_algorithm": algo_stats
    }


# =============================================================================
# Standalone CLI
# =============================================================================

def main():
    """CLI entry point."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(
        description="GraphBrew Cache Simulation Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m scripts.lib.cache --graph graphs/email-Enron/email-Enron.mtx
    python -m scripts.lib.cache --graph test.mtx --algorithms 0,8,9 --benchmarks pr,bfs
    python -m scripts.lib.cache --graph test.mtx -o results/cache_results.json
"""
    )
    
    parser.add_argument("--graph", "-g", required=True, help="Path to graph file")
    parser.add_argument("--algorithms", "-a", default="0,1,8",
                        help="Comma-separated algorithm IDs (default: 0,1,8)")
    parser.add_argument("--benchmarks", "-b", default="pr,bfs",
                        help="Comma-separated benchmarks (default: pr,bfs)")
    parser.add_argument("--output", "-o", help="Output JSON file for results")
    parser.add_argument("--timeout", type=int, default=TIMEOUT_SIM,
                        help=f"Timeout per simulation (default: {TIMEOUT_SIM})")
    parser.add_argument("--skip-heavy", action="store_true",
                        help="Skip heavy benchmarks on large graphs")
    
    args = parser.parse_args()
    
    # Parse arguments
    algo_ids = [int(x.strip()) for x in args.algorithms.split(",")]
    benchmarks = [x.strip() for x in args.benchmarks.split(",")]
    
    # Create GraphInfo
    graph_path = Path(args.graph)
    graph = GraphInfo(
        name=graph_path.stem,
        path=str(graph_path),
        size_mb=graph_path.stat().st_size / (1024 * 1024) if graph_path.exists() else 0,
        is_symmetric=True
    )
    
    # Run simulations
    results = run_cache_simulations(
        graphs=[graph],
        algorithms=algo_ids,
        benchmarks=benchmarks,
        timeout=args.timeout,
        skip_heavy=args.skip_heavy
    )
    
    # Print summary
    summary = get_cache_stats_summary(results)
    print(f"\nCache Simulation Summary:")
    print(f"  Total: {summary['total']}, Successful: {summary['successful']}, Failed: {summary.get('failed', 0)}")
    
    if 'by_algorithm' in summary:
        print("\nBy Algorithm:")
        for algo, stats in summary['by_algorithm'].items():
            print(f"  {algo}: L1={stats['avg_l1_hit_rate']:.1f}% L2={stats['avg_l2_hit_rate']:.1f}% L3={stats['avg_l3_hit_rate']:.1f}%")
    
    # Save results if output specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
