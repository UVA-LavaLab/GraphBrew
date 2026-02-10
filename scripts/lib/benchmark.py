#!/usr/bin/env python3
"""
Benchmark execution utilities for GraphBrew.

Runs graph algorithm benchmarks with various reordering strategies.
Can be used standalone or as a library.

Standalone usage:
    python -m scripts.lib.benchmark --graph graphs/email-Enron/email-Enron.mtx -a 0,1,8
    python -m scripts.lib.benchmark --graph test.mtx --leiden-variants

Library usage:
    from scripts.lib.benchmark import run_benchmark, run_benchmark_suite
    
    result = run_benchmark("pr", "graph.mtx", algorithm="12:community")
    results = run_benchmark_suite("graph.mtx", algorithms=["0", "1", "8"])
"""

import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .utils import (
    BIN_DIR, ALGORITHMS, BENCHMARKS,
    BenchmarkResult, log, run_command, check_binary_exists,
    get_results_file, save_json, get_algorithm_name, parse_algorithm_option
)
from .reorder import get_algorithm_name_with_variant
from .features import update_graph_properties, save_graph_properties_cache

# Enable run logging (saves command outputs per graph)
ENABLE_RUN_LOGGING = True


# =============================================================================
# Adaptive Timeout
# =============================================================================

def compute_adaptive_timeout(edges: int, base_timeout: int = 600) -> int:
    """
    Compute a timeout that scales with graph size.

    Small graphs (<1M edges) get the base timeout (default 600s).
    Medium graphs (1M–10M) get 2× base.
    Large graphs (10M–100M) get 4× base.
    Very large graphs (>100M) get 8× base.

    This prevents false-positive timeouts on large graphs while still
    catching hangs and bugs quickly on small ones.

    Args:
        edges: Number of edges in the graph.
        base_timeout: Base timeout in seconds (applied to <1M-edge graphs).

    Returns:
        Adjusted timeout in seconds.
    """
    if edges <= 0:
        return base_timeout
    if edges < 1_000_000:
        return base_timeout
    elif edges < 10_000_000:
        return base_timeout * 2
    elif edges < 100_000_000:
        return base_timeout * 4
    else:
        return base_timeout * 8


# =============================================================================
# Output Parsing
# =============================================================================

def parse_benchmark_output(output: str) -> Tuple[float, float, Dict]:
    """
    Parse benchmark stdout to extract timing information.
    
    Args:
        output: Stdout from benchmark execution
        
    Returns:
        Tuple of (average_time, reorder_time, extra_info)
    """
    avg_time = 0.0
    reorder_time = 0.0
    extra = {}
    
    for line in output.split("\n"):
        line_lower = line.lower()
        
        # Average time - various formats
        if "average" in line_lower and ("time" in line_lower or ":" in line):
            match = re.search(r"[\d.]+", line.split(":")[-1] if ":" in line else line)
            if match:
                avg_time = float(match.group())
        
        # Reorder time
        if "reorder" in line_lower and "time" in line_lower:
            match = re.search(r"[\d.]+", line.split(":")[-1] if ":" in line else line)
            if match:
                reorder_time = float(match.group())
        
        # Trial times
        if "trial" in line_lower and "time" in line_lower:
            match = re.search(r"trial\s*(\d+).*?([\d.]+)", line_lower)
            if match:
                extra[f"trial_{match.group(1)}"] = float(match.group(2))
        
        # MTEPS (for BFS)
        if "mteps" in line_lower:
            match = re.search(r"[\d.]+", line)
            if match:
                extra["mteps"] = float(match.group())
        
        # PageRank iterations
        if "iteration" in line_lower:
            match = re.search(r"(\d+)\s*iteration", line_lower)
            if match:
                extra["iterations"] = int(match.group(1))
    
    # Extract topology features for weight learning
    # These are printed by the C++ code during graph loading
    dv_match = re.search(r'Degree Variance:\s*([\d.]+)', output)
    if dv_match:
        extra['degree_variance'] = float(dv_match.group(1))
    
    hc_match = re.search(r'Hub Concentration:\s*([\d.]+)', output)
    if hc_match:
        extra['hub_concentration'] = float(hc_match.group(1))
    
    ad_match = re.search(r'Avg Degree:\s*([\d.]+)', output)
    if ad_match:
        extra['avg_degree'] = float(ad_match.group(1))
    
    cc_match = re.search(r'Clustering Coefficient:\s*([\d.]+)', output)
    if cc_match:
        extra['clustering_coefficient'] = float(cc_match.group(1))
    
    apl_match = re.search(r'Avg Path Length:\s*([\d.]+)', output)
    if apl_match:
        extra['avg_path_length'] = float(apl_match.group(1))
    
    diam_match = re.search(r'Diameter Estimate:\s*([\d.]+)', output)
    if diam_match:
        extra['diameter'] = float(diam_match.group(1))
    
    comm_match = re.search(r'Community Count Estimate:\s*([\d.]+)', output)
    if comm_match:
        extra['community_count'] = float(comm_match.group(1))
    
    mod_match = re.search(r'Modularity:\s*([\d.]+)', output)
    if mod_match:
        extra['modularity'] = float(mod_match.group(1))
    
    return avg_time, reorder_time, extra


# =============================================================================
# Benchmark Execution
# =============================================================================

def run_benchmark(
    benchmark: str,
    graph_path: str,
    algorithm: str = "0",
    trials: int = 3,
    symmetric: bool = True,
    timeout: int = 600,
    extra_args: List[str] = None,
    bin_dir: str = None
) -> BenchmarkResult:
    """
    Run a single benchmark with specified algorithm.
    
    Args:
        benchmark: Benchmark name (pr, bfs, cc, etc.)
        graph_path: Path to graph file
        algorithm: Algorithm option string (e.g., "0", "12:community", "15:1.0")
        trials: Number of trials
        symmetric: Use symmetric graph flag (-s)
        timeout: Timeout in seconds
        extra_args: Additional command line arguments
        bin_dir: Directory containing benchmark binaries
        
    Returns:
        BenchmarkResult with timing information
    """
    graph_path = Path(graph_path)
    graph_name = graph_path.stem
    bin_dir_path = Path(bin_dir) if bin_dir else BIN_DIR
    
    algo_id, _ = parse_algorithm_option(algorithm)
    algo_name = get_algorithm_name(algorithm)
    
    # Build command
    binary = bin_dir_path / benchmark
    if not binary.exists():
        return BenchmarkResult(
            graph=graph_name,
            algorithm=algo_name,
            algorithm_id=algo_id,
            benchmark=benchmark,
            time_seconds=0.0,
            success=False,
            error=f"Binary not found: {binary}"
        )
    
    cmd = [str(binary), "-f", str(graph_path), "-o", algorithm, "-n", str(trials)]
    
    if symmetric:
        cmd.append("-s")
    
    if extra_args:
        cmd.extend(extra_args)
    
    # Run benchmark
    try:
        start_time = time.time()
        result = run_command(cmd, timeout=timeout, check=False)
        elapsed = time.time() - start_time
        
        # Save run log
        if ENABLE_RUN_LOGGING:
            try:
                from .graph_data import save_run_log
                save_run_log(
                    graph_name=graph_name,
                    operation='benchmark',
                    algorithm=algo_name,
                    benchmark=benchmark,
                    output=result.stdout + "\n--- STDERR ---\n" + result.stderr if result.stderr else result.stdout,
                    command=' '.join(str(c) for c in cmd),
                    exit_code=result.returncode,
                    duration=elapsed
                )
            except Exception as e:
                log.debug(f"Failed to save run log: {e}")
        
        if result.returncode != 0:
            error_msg = result.stderr[:500] if result.stderr else f"Exit code {result.returncode}"
            return BenchmarkResult(
                graph=graph_name,
                algorithm=algo_name,
                algorithm_id=algo_id,
                benchmark=benchmark,
                time_seconds=0.0,
                success=False,
                error=error_msg
            )
        
        # Parse output
        avg_time, reorder_time, extra = parse_benchmark_output(result.stdout)
        
        return BenchmarkResult(
            graph=graph_name,
            algorithm=algo_name,
            algorithm_id=algo_id,
            benchmark=benchmark,
            time_seconds=avg_time,
            reorder_time=reorder_time,
            trials=trials,
            success=True,
            extra=extra
        )
        
    except Exception as e:
        return BenchmarkResult(
            graph=graph_name,
            algorithm=algo_name,
            algorithm_id=algo_id,
            benchmark=benchmark,
            time_seconds=0.0,
            success=False,
            error=str(e)
        )


def run_benchmark_suite(
    graph_path: str,
    algorithms: List[str] = None,
    benchmarks: List[str] = None,
    trials: int = 3,
    timeout: int = 600
) -> List[BenchmarkResult]:
    """
    Run a suite of benchmarks on a graph.
    
    Args:
        graph_path: Path to graph file
        algorithms: List of algorithm option strings (default: ["0", "1", "8"])
        benchmarks: List of benchmark names (default: ["pr", "bfs", "cc"])
        trials: Number of trials per config
        timeout: Timeout in seconds
        
    Returns:
        List of BenchmarkResult
    """
    if algorithms is None:
        algorithms = ["0", "1", "8"]
    if benchmarks is None:
        benchmarks = ["pr", "bfs", "cc"]
    
    results = []
    graph_name = Path(graph_path).stem
    
    log.info(f"Running {len(benchmarks)} benchmarks × {len(algorithms)} algorithms on {graph_name}")
    
    for bench in benchmarks:
        if not check_binary_exists(bench):
            log.warning(f"Skipping {bench}: binary not found")
            continue
        
        for algo in algorithms:
            algo_name = get_algorithm_name(algo)
            log.info(f"  {bench} with {algo_name}...")
            
            result = run_benchmark(
                benchmark=bench,
                graph_path=graph_path,
                algorithm=algo,
                trials=trials,
                timeout=timeout
            )
            results.append(result)
            
            if result.success:
                log.info(f"    {result.time_seconds:.4f}s")
            else:
                log.warning(f"    FAILED: {result.error[:50]}")
    
    return results


def run_benchmarks_multi_graph(
    graphs: List,  # List of GraphInfo objects
    algorithms: List[int],
    benchmarks: List[str],
    bin_dir: str = None,
    num_trials: int = 3,
    timeout: int = 600,
    label_maps: Dict[str, Dict[str, str]] = None,
    weights_dir: str = None,
    update_weights: bool = True,
    skip_slow: bool = False,
    progress = None  # Optional ProgressTracker
) -> List[BenchmarkResult]:
    """
    Run benchmarks across multiple graphs.
    
    This is the main multi-graph benchmarking function used by the experiment pipeline.
    
    Args:
        graphs: List of GraphInfo objects
        algorithms: List of algorithm IDs
        benchmarks: List of benchmark names
        bin_dir: Binary directory (default: bench/bin)
        num_trials: Number of trials per configuration
        timeout: Timeout in seconds
        label_maps: Pre-computed label maps {graph_name: {algo_name: path}}
        weights_dir: Directory for weight files
        update_weights: Whether to update weights incrementally
        skip_slow: Skip slow algorithms (GORDER, CORDER, RCM)
        progress: Optional progress tracker
        
    Returns:
        List of BenchmarkResult objects
    """
    bin_dir = bin_dir or str(BIN_DIR)
    label_maps = label_maps or {}
    results = []
    
    # Filter slow algorithms if requested
    if skip_slow:
        from .utils import SLOW_ALGORITHMS
        algorithms = [a for a in algorithms if a not in SLOW_ALGORITHMS]
    
    total_configs = len(graphs) * len(algorithms) * len(benchmarks)
    completed = 0
    skipped = 0
    
    # Track (graph, benchmark) combos where ORIGINAL or first algo timed out / crashed.
    # If the baseline is intractable, every reordering variant will be too — skip them
    # to avoid burning hours of timeout budget on a single graph×benchmark pair.
    timed_out_combos: set = set()
    
    for graph in graphs:
        graph_name = graph.name
        graph_path = graph.path
        graph_label_maps = label_maps.get(graph_name, {})
        
        if progress:
            progress.info(f"Benchmarking: {graph_name}")
        
        # Adaptive timeout based on graph size
        graph_timeout = compute_adaptive_timeout(graph.edges, timeout)
        if graph_timeout != timeout and progress:
            progress.info(f"  Adaptive timeout: {graph_timeout}s (edges={graph.edges:,})")

        for bench in benchmarks:
            if not check_binary_exists(bench, bin_dir):
                log.warning(f"Skipping {bench}: binary not found")
                continue
            
            for algo_id in algorithms:
                # Always include variant in name for algorithms that have variants
                algo_name = get_algorithm_name_with_variant(algo_id)
                
                # Early-exit: skip remaining algorithms if this graph×benchmark
                # already proved intractable (timeout or crash on a prior algorithm)
                combo_key = (graph_name, bench)
                if combo_key in timed_out_combos:
                    result = BenchmarkResult(
                        graph=graph_name,
                        algorithm=algo_name,
                        algorithm_id=algo_id,
                        benchmark=bench,
                        time_seconds=0.0,
                        success=False,
                        error="SKIPPED: prior algorithm timed out on this graph+benchmark"
                    )
                    result.nodes = graph.nodes
                    result.edges = graph.edges
                    results.append(result)
                    completed += 1
                    skipped += 1
                    continue
                
                # Check if we have a label map
                label_map_path = graph_label_maps.get(algo_name, "")
                
                # Build algorithm option string
                # Use MAP (algo 13) with label map when available, except for ORIGINAL (algo 0)
                if label_map_path and os.path.exists(label_map_path) and algo_id != 0:
                    algo_opt = f"13:{label_map_path}"
                    using_label_map = True
                else:
                    algo_opt = str(algo_id)
                    using_label_map = False
                
                result = run_benchmark(
                    benchmark=bench,
                    graph_path=graph_path,
                    algorithm=algo_opt,
                    trials=num_trials,
                    timeout=graph_timeout,
                    bin_dir=bin_dir
                )
                
                # Detect timeout or crash — mark this graph×benchmark as intractable
                if not result.success:
                    err_lower = (result.error or "").lower()
                    is_timeout = "timed out" in err_lower or "timeout" in err_lower
                    is_crash = "exit code -" in err_lower or "signal" in err_lower
                    if is_timeout or is_crash:
                        timed_out_combos.add(combo_key)
                        remaining = len(algorithms) - (algorithms.index(algo_id) + 1) if algo_id in algorithms else 0
                        if progress:
                            reason = "TIMEOUT" if is_timeout else f"CRASH ({result.error[:60]})"
                            progress.info(
                                f"  ⚠ {reason}: {algo_name} on {bench}/{graph_name} — "
                                f"skipping {remaining} remaining algorithms for this combo"
                            )
                
                # Enrich result with metadata
                result.graph = graph_name
                result.nodes = graph.nodes
                result.edges = graph.edges
                
                # Cache graph features from first successful benchmark run
                # The extra dict now contains topology features parsed from C++ output
                if result.success and result.extra:
                    features_to_cache = {k: v for k, v in result.extra.items() 
                                        if k in ('degree_variance', 'hub_concentration', 'avg_degree',
                                                'clustering_coefficient', 'avg_path_length', 
                                                'diameter', 'community_count', 'modularity')}
                    if features_to_cache:
                        features_to_cache['nodes'] = graph.nodes
                        features_to_cache['edges'] = graph.edges
                        update_graph_properties(graph_name, features_to_cache, "results")
                
                # Preserve original algorithm name when using label map
                if using_label_map:
                    result.algorithm = algo_name
                    result.algorithm_id = algo_id
                
                results.append(result)
                completed += 1
                
                if progress and completed % 10 == 0:
                    progress.info(f"  Progress: {completed}/{total_configs}")
    
    if skipped > 0:
        log.info(f"Benchmark early-exit: skipped {skipped}/{total_configs} runs due to timeout/crash")
    
    # Save the graph properties cache after all benchmarks
    try:
        save_graph_properties_cache("results")
    except Exception as e:
        log.warning(f"Failed to save graph properties cache: {e}")
    
    # Note: compute_speedups returns a dict of {algorithm: {benchmark: speedup}}
    # We don't overwrite results here - callers can use compute_speedups() separately
    
    return results


def run_leiden_variant_comparison(
    graph_path: str,
    benchmarks: List[str] = None,
    trials: int = 3,
    include_baselines: bool = True
) -> List[BenchmarkResult]:
    """
    Run comprehensive comparison of all Leiden variants.
    
    Args:
        graph_path: Path to graph file
        benchmarks: Benchmarks to run (default: pr, bfs, cc)
        trials: Number of trials per config
        include_baselines: Include ORIGINAL, RANDOM, RABBITORDER
        
    Returns:
        List of BenchmarkResult
    """
    if benchmarks is None:
        benchmarks = ["pr", "bfs", "cc"]
    
    # Build algorithm list
    algorithms = []
    
    if include_baselines:
        algorithms.extend(["0", "1", "8"])  # ORIGINAL, RANDOM, RABBITORDER
    
    # GraphBrewOrder (12)
    algorithms.append("12")
    
    # LeidenOrder (15)
    algorithms.append("15")
    
    return run_benchmark_suite(graph_path, algorithms, benchmarks, trials)


# =============================================================================
# Results Analysis
# =============================================================================

def compute_speedups(
    results: List[BenchmarkResult],
    baseline_algo: str = "RANDOM"
) -> Dict[str, Dict[str, float]]:
    """
    Compute speedups relative to baseline.
    
    Args:
        results: List of benchmark results
        baseline_algo: Baseline algorithm name (partial match)
        
    Returns:
        Dict of {algorithm: {benchmark: speedup}}
    """
    # Find baseline times by (graph, benchmark)
    baselines = {}
    for r in results:
        if baseline_algo in r.algorithm and r.success:
            key = (r.graph, r.benchmark)
            baselines[key] = r.time_seconds
    
    # Compute speedups
    speedups = {}
    for r in results:
        if not r.success or r.time_seconds <= 0:
            continue
        
        key = (r.graph, r.benchmark)
        baseline_time = baselines.get(key, r.time_seconds)
        
        if r.algorithm not in speedups:
            speedups[r.algorithm] = {}
        
        if baseline_time > 0:
            speedups[r.algorithm][r.benchmark] = baseline_time / r.time_seconds
        else:
            speedups[r.algorithm][r.benchmark] = 1.0
    
    return speedups


def format_results_table(
    results: List[BenchmarkResult],
    baseline_algo: str = "RANDOM"
) -> str:
    """Format results as a text table with speedups."""
    speedups = compute_speedups(results, baseline_algo)
    
    # Get unique benchmarks and algorithms
    benchmarks = sorted(set(r.benchmark for r in results if r.success))
    algorithms = sorted(speedups.keys())
    
    if not benchmarks or not algorithms:
        return "No successful results to display"
    
    # Build header
    lines = []
    header = f"{'Algorithm':<35}"
    for b in benchmarks:
        header += f" {b:>8}"
    header += f" {'Avg':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    
    # Build rows
    for algo in algorithms:
        row = f"{algo:<35}"
        algo_speedups = []
        for b in benchmarks:
            s = speedups[algo].get(b, 1.0)
            row += f" {s:>7.2f}x"
            algo_speedups.append(s)
        avg = sum(algo_speedups) / len(algo_speedups) if algo_speedups else 1.0
        row += f" {avg:>7.2f}x"
        lines.append(row)
    
    return "\n".join(lines)


# =============================================================================
# Standalone CLI
# =============================================================================

def main():
    """CLI for benchmark execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run GraphBrew benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m scripts.lib.benchmark --graph graph.mtx -a 0,1,8
    python -m scripts.lib.benchmark --graph graph.mtx --leiden-variants
    python -m scripts.lib.benchmark --graph graph.mtx -a 0,8,12 --expand
        """
    )
    
    parser.add_argument("--graph", "-g", required=True, help="Graph file path")
    parser.add_argument("-a", "--algorithms", default="0,1,8",
                       help="Comma-separated algorithm options")
    parser.add_argument("-b", "--benchmarks", nargs="+", default=["pr", "bfs", "cc"],
                       help="Benchmarks to run")
    parser.add_argument("-n", "--trials", type=int, default=3, help="Trials per config")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds")
    parser.add_argument("--leiden-variants", action="store_true",
                       help="Run Leiden variant comparison (baselines + GraphBrew + LeidenOrder)")
    parser.add_argument("--expand", action="store_true",
                       help="Expand variant-based algorithms to all variants")
    parser.add_argument("-o", "--output", help="Output JSON file")
    
    args = parser.parse_args()
    
    # Verify graph exists
    graph_path = Path(args.graph)
    if not graph_path.exists():
        log.error(f"Graph not found: {graph_path}")
        return 1
    
    # Run benchmarks
    if args.leiden_variants:
        log.info("Running Leiden variant comparison...")
        results = run_leiden_variant_comparison(
            str(graph_path),
            benchmarks=args.benchmarks,
            trials=args.trials
        )
    else:
        algorithms = args.algorithms.split(",")
        
        results = run_benchmark_suite(
            str(graph_path),
            algorithms=algorithms,
            benchmarks=args.benchmarks,
            trials=args.trials,
            timeout=args.timeout
        )
    
    # Display results
    print("\n" + "=" * 70)
    print("Results (speedup vs RANDOM)")
    print("=" * 70)
    print(format_results_table(results))
    
    # Save to JSON
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = get_results_file("benchmark")
    
    save_json([r.to_dict() for r in results], output_path)
    print(f"\nResults saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
