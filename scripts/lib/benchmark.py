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
    
    result = run_benchmark("pr", "graph.mtx", algorithm="16:1.0:hybrid")
    results = run_benchmark_suite("graph.mtx", algorithms=["0", "1", "8"])
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .utils import (
    BIN_DIR, ALGORITHMS, BENCHMARKS,
    LEIDEN_DENDROGRAM_VARIANTS, LEIDEN_CSR_VARIANTS,
    BenchmarkResult, log, run_command, check_binary_exists,
    get_results_file, save_json, get_algorithm_name, parse_algorithm_option
)


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
    extra_args: List[str] = None
) -> BenchmarkResult:
    """
    Run a single benchmark with specified algorithm.
    
    Args:
        benchmark: Benchmark name (pr, bfs, cc, etc.)
        graph_path: Path to graph file
        algorithm: Algorithm option string (e.g., "0", "16:1.0:hybrid")
        trials: Number of trials
        symmetric: Use symmetric graph flag (-s)
        timeout: Timeout in seconds
        extra_args: Additional command line arguments
        
    Returns:
        BenchmarkResult with timing information
    """
    graph_path = Path(graph_path)
    graph_name = graph_path.stem
    
    algo_id, _ = parse_algorithm_option(algorithm)
    algo_name = get_algorithm_name(algorithm)
    
    # Build command
    binary = BIN_DIR / benchmark
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
        result = run_command(cmd, timeout=timeout, check=False)
        
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
    
    # LeidenDendrogram variants (16)
    for variant in LEIDEN_DENDROGRAM_VARIANTS:
        algorithms.append(f"16:1.0:{variant}")
    
    # LeidenCSR variants (17)
    for variant in LEIDEN_CSR_VARIANTS:
        algorithms.append(f"17:1.0:3:{variant}")
    
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
    python -m scripts.lib.benchmark --graph graph.mtx -a 16,17 --expand
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
                       help="Run all Leiden variant comparison")
    parser.add_argument("--expand", action="store_true",
                       help="Expand Leiden algorithms (16,17) to all variants")
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
        
        # Expand Leiden variants if requested
        if args.expand:
            expanded = []
            for algo in algorithms:
                algo_id, _ = parse_algorithm_option(algo)
                if algo_id == 16:
                    expanded.extend([f"16:1.0:{v}" for v in LEIDEN_DENDROGRAM_VARIANTS])
                elif algo_id == 17:
                    expanded.extend([f"17:1.0:3:{v}" for v in LEIDEN_CSR_VARIANTS])
                else:
                    expanded.append(algo)
            algorithms = expanded
        
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
