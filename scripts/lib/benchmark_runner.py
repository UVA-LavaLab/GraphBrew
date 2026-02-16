#!/usr/bin/env python3
"""
Fresh benchmark runner: run all AdaptiveOrder-eligible algorithms on .sg graphs.

Replaces the legacy benchmark_fresh.sh shell script with a pure-Python
implementation that integrates with the GraphBrew lib/ infrastructure.

Usage (standalone):
    python -m scripts.lib.benchmark_runner
    python -m scripts.lib.benchmark_runner --trials 5 --benchmarks bfs pr cc
    python -m scripts.lib.benchmark_runner --graphs soc-Epinions1 cnr-2000

Usage (library):
    from scripts.lib.benchmark_runner import run_fresh_benchmarks
    results = run_fresh_benchmarks(trials=3, benchmarks=['bfs', 'pr', 'cc'])
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .utils import (
    ALGORITHMS, BIN_DIR, RESULTS_DIR, GRAPHS_DIR,
    BenchmarkResult, Logger,
)

log = Logger()

# ============================================================================
# Constants
# ============================================================================

# AdaptiveOrder-eligible algorithms (by -o ID)
# Excludes MAP (13) and AdaptiveOrder itself (14)
ADAPTIVE_ELIGIBLE_ALGOS: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15]

# Human-readable names matching C++ output
ALGO_NAMES: Dict[int, str] = {
    0: "ORIGINAL",
    1: "Random",
    2: "Sort",
    3: "HubSort",
    4: "HubCluster",
    5: "DBG",
    6: "HubSortDBG",
    7: "HubClusterDBG",
    8: "RabbitOrder",
    9: "GOrder",
    10: "COrder",
    11: "RCMOrder",
    12: "GraphBrewOrder",
    15: "LeidenOrder",
}

# Complexity guards: skip algorithms that are too slow on large graphs
ALGO_NODE_LIMITS: Dict[int, int] = {
    9: 500_000,    # GOrder: O(n*m*w) reorder (use -o 9:csr for faster CSR variant)
    12: 500_000,   # GraphBrewOrder: slow community detection
    10: 2_000_000, # COrder: slow on very large graphs
}

# Default benchmarks (all 5 types)
DEFAULT_BENCHMARKS: List[str] = ["bfs", "pr", "pr_spmv", "cc", "cc_sv"]

# Default timeout per benchmark invocation (seconds)
DEFAULT_TIMEOUT: int = 120


# ============================================================================
# Graph Discovery
# ============================================================================

def discover_sg_graphs(
    graphs_dir: str = None,
    graph_names: List[str] = None,
) -> List[Tuple[str, str, int]]:
    """
    Discover .sg graph files and their node counts.

    Returns:
        List of (graph_name, sg_path, node_count) sorted by node count.
    """
    graphs_dir = Path(graphs_dir or GRAPHS_DIR)
    results = []

    for sg_path in sorted(graphs_dir.glob("*/*.sg")):
        name = sg_path.parent.name
        if graph_names and name not in graph_names:
            continue

        # Read node count from features.json
        feat_path = sg_path.parent / "features.json"
        nodes = 0
        if feat_path.is_file():
            try:
                with open(feat_path) as f:
                    nodes = json.load(f).get("nodes", 0)
            except Exception:
                pass
        results.append((name, str(sg_path), nodes))

    # Sort by node count (smallest first) for predictable ordering
    results.sort(key=lambda x: x[2])
    return results


# ============================================================================
# Single Benchmark Run
# ============================================================================

def _run_single(
    binary: str,
    sg_path: str,
    algo_id: int,
    trials: int,
    timeout: int,
) -> Optional[Tuple[float, float]]:
    """
    Run a single benchmark invocation.

    Returns:
        (avg_time, reorder_time) or None on failure.
    """
    cmd = [binary, "-f", sg_path, "-o", str(algo_id), "-n", str(trials)]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            return None

        output = result.stdout
        avg_match = re.search(r"Average Time:\s+([\d.]+)", output)
        reorder_match = re.search(r"Reorder Time:\s+([\d.]+)", output)

        if avg_match:
            avg_time = float(avg_match.group(1))
            reorder_time = float(reorder_match.group(1)) if reorder_match else 0.0
            return (avg_time, reorder_time)
        return None
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


# ============================================================================
# Main Runner
# ============================================================================

def run_fresh_benchmarks(
    graphs_dir: str = None,
    graph_names: List[str] = None,
    benchmarks: List[str] = None,
    algos: List[int] = None,
    trials: int = 3,
    timeout: int = DEFAULT_TIMEOUT,
    bin_dir: str = None,
    output_file: str = None,
) -> List[dict]:
    """
    Run all AdaptiveOrder-eligible algorithms on all .sg graphs.

    This is the Python equivalent of benchmark_fresh.sh. It:
    1. Discovers .sg graphs (or uses the provided list)
    2. Runs every (graph, benchmark, algorithm) combination
    3. Skips algorithms that exceed node-count guards
    4. Returns results as a list of dicts (also saves to JSON)

    Args:
        graphs_dir: Directory containing graph subdirectories with .sg files
        graph_names: Optional list of specific graph names to benchmark
        benchmarks: List of benchmark types (default: bfs, pr, pr_spmv, cc, cc_sv)
        algos: List of algorithm IDs (default: all AdaptiveOrder-eligible)
        trials: Number of trials per (graph, bench, algo) combination
        timeout: Timeout per benchmark invocation in seconds
        bin_dir: Directory containing benchmark binaries
        output_file: Path to save JSON results (default: results/benchmark_fresh.json)

    Returns:
        List of result dicts compatible with eval_weights.py
    """
    graphs_dir = str(graphs_dir or GRAPHS_DIR)
    benchmarks = benchmarks or DEFAULT_BENCHMARKS
    algos = algos or ADAPTIVE_ELIGIBLE_ALGOS
    bin_dir = str(bin_dir or BIN_DIR)
    if output_file is None:
        output_file = str(Path(RESULTS_DIR) / "benchmark_fresh.json")

    # Discover graphs
    graphs = discover_sg_graphs(graphs_dir, graph_names)
    if not graphs:
        log.error("No .sg graphs found")
        return []

    log.info(f"Found {len(graphs)} graphs with .sg files")
    log.info(f"Algorithms: {[ALGO_NAMES.get(a, str(a)) for a in algos]}")
    log.info(f"Benchmarks: {benchmarks}")

    entries: List[dict] = []
    total = 0
    failed = 0

    for graph_name, sg_path, nodes in graphs:
        print(f"\n{'='*64}")
        print(f"=== {graph_name} ({nodes:,} nodes) ===")
        print(f"{'='*64}")

        for bench in benchmarks:
            binary = os.path.join(bin_dir, bench)
            if not os.path.isfile(binary):
                log.warning(f"Binary not found: {binary}")
                continue

            for algo_id in algos:
                algo_name = ALGO_NAMES.get(algo_id)
                if not algo_name:
                    continue

                # Apply complexity guards
                node_limit = ALGO_NODE_LIMITS.get(algo_id)
                if node_limit and nodes > node_limit:
                    continue

                sys.stdout.write(f"  {bench}/{algo_name}... ")
                sys.stdout.flush()

                result = _run_single(binary, sg_path, algo_id, trials, timeout)
                if result:
                    avg_time, reorder_time = result
                    print(f"{avg_time}s (reorder: {reorder_time}s)")
                    entries.append({
                        "graph": graph_name,
                        "algorithm": algo_name,
                        "algorithm_id": algo_id,
                        "benchmark": bench,
                        "time_seconds": avg_time,
                        "reorder_time": reorder_time,
                        "trials": trials,
                        "success": True,
                        "error": "",
                        "extra": "sg_benchmark",
                    })
                    total += 1
                else:
                    print("TIMEOUT/ERROR")
                    failed += 1

    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(entries, f, indent=2)

    print(f"\n{'='*64}")
    print(f"Done. {total} entries in {output_file} ({failed} failed)")
    print(f"{'='*64}")

    return entries


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run fresh benchmarks on all .sg graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--graphs-dir", default=None,
                        help="Directory containing graph subdirectories")
    parser.add_argument("--graphs", nargs="+", default=None,
                        help="Specific graph names to benchmark")
    parser.add_argument("--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS,
                        help=f"Benchmark types (default: {DEFAULT_BENCHMARKS})")
    parser.add_argument("--algos", nargs="+", type=int, default=None,
                        help=f"Algorithm IDs (default: {ADAPTIVE_ELIGIBLE_ALGOS})")
    parser.add_argument("--trials", type=int, default=3,
                        help="Trials per benchmark (default: 3)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help=f"Timeout per run in seconds (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--bin-dir", default=None,
                        help="Directory containing benchmark binaries")
    parser.add_argument("--output", "-o", default=None,
                        help="Output JSON file (default: results/benchmark_fresh.json)")
    args = parser.parse_args()

    run_fresh_benchmarks(
        graphs_dir=args.graphs_dir,
        graph_names=args.graphs,
        benchmarks=args.benchmarks,
        algos=args.algos,
        trials=args.trials,
        timeout=args.timeout,
        bin_dir=args.bin_dir,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
