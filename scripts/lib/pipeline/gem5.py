#!/usr/bin/env python3
"""
gem5 simulation pipeline module for GraphBrew.

Mirrors the interface of scripts/lib/pipeline/cache.py but executes
graph benchmarks under gem5 instead of the standalone cache simulator.
Supports all graph-aware replacement policies (GRASP, P-OPT, ECG) and
the DROPLET indirect prefetcher.

Usage (standalone):
    python -m scripts.lib.pipeline.gem5 --graph results/graphs/soc-pokec/soc-pokec.sg \\
        --benchmark pr --algorithm 5 --policy GRASP

Usage (library):
    from scripts.lib.pipeline.gem5 import run_gem5_simulation, run_gem5_simulations
    result = run_gem5_simulation(graph_path, "pr", algo_id=5, policy="GRASP")
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from ..core.utils import (
    PROJECT_ROOT, BENCH_DIR, BIN_DIR,
    ALGORITHMS, Logger, run_command,
    canonical_algo_key, algo_converter_opt,
    GRAPHS_DIR,
)
from ..core.graph_types import GraphInfo
from .cache import CacheResult

log = Logger()

# =============================================================================
# Constants
# =============================================================================

GEM5_SIM_DIR = BENCH_DIR / "include" / "gem5_sim"
GEM5_DIR = GEM5_SIM_DIR / "gem5"
GEM5_CONFIGS_DIR = GEM5_SIM_DIR / "configs" / "graphbrew"
GEM5_SCRIPTS_DIR = GEM5_SIM_DIR / "scripts"
GEM5_METADATA_DIR = PROJECT_ROOT / "results" / "gem5_metadata"

# Default gem5 binary (X86 opt build)
DEFAULT_GEM5_BINARY = GEM5_DIR / "build" / "X86" / "gem5.opt"

# Default timeouts (gem5 is ~100-1000x slower than native)
TIMEOUT_GEM5 = 7200        # 2 hours for standard benchmarks
TIMEOUT_GEM5_HEAVY = 14400  # 4 hours for BC, SSSP

HEAVY_BENCHMARKS = {"bc", "sssp"}

# Supported policies
SUPPORTED_POLICIES = ["LRU", "FIFO", "SRRIP", "RANDOM", "GRASP", "POPT", "ECG"]


# =============================================================================
# gem5 Binary Discovery
# =============================================================================

def find_gem5_binary(isa: str = "X86", build_type: str = "opt") -> Path:
    """Find the gem5 binary for the given ISA and build type."""
    binary = GEM5_DIR / "build" / isa / f"gem5.{build_type}"
    if not binary.exists():
        raise FileNotFoundError(
            f"gem5 binary not found: {binary}\n"
            f"Run: python scripts/setup_gem5.py --isa {isa}"
        )
    return binary


def find_benchmark_binary(benchmark: str) -> Path:
    """Find the native benchmark binary (used by gem5 SE mode)."""
    binary = BIN_DIR / benchmark
    if not binary.exists():
        raise FileNotFoundError(
            f"Benchmark binary not found: {binary}\n"
            f"Run: make {benchmark}"
        )
    return binary


# =============================================================================
# Metadata Export
# =============================================================================

def export_gem5_metadata(graph_name: str, graph_path: str,
                         metadata: Optional[Dict] = None) -> Path:
    """Export graph metadata to JSON for gem5 consumption.

    Creates results/gem5_metadata/{graph_name}/context.json with property
    region info, topology, and rereference matrix paths.

    Args:
        graph_name: Graph name (e.g., "soc-pokec")
        graph_path: Path to graph .sg file
        metadata: Optional pre-computed metadata dict. If None, creates
                  a minimal placeholder.

    Returns:
        Path to the created JSON file.
    """
    output_dir = GEM5_METADATA_DIR / graph_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "context.json"

    if metadata is None:
        metadata = {
            "graph_name": graph_name,
            "graph_path": str(graph_path),
            "property_regions": [],
            "topology": {
                "num_vertices": 0,
                "num_edges": 0,
                "avg_degree": 0.0,
                "num_buckets": 11,
                "bucket_vertex_counts": [],
            },
            "mask_config": {
                "mask_width": 8,
                "dbg_bits": 2,
                "popt_bits": 4,
                "prefetch_bits": 2,
                "num_buckets": 11,
                "rrpv_max": 7,
                "ecg_mode": "DBG_PRIMARY",
                "enabled": False,
            },
            "rereference": {
                "matrix_file": "",
                "num_epochs": 256,
                "num_cache_lines": 0,
                "epoch_size": 0,
                "base_address": 0,
                "enabled": False,
            },
            "edge_array": {
                "base_address": 0,
                "size": 0,
                "elem_size": 4,
            },
        }

    with open(output_file, "w") as f:
        json.dump(metadata, f, indent=2)

    return output_file


# =============================================================================
# Single Simulation
# =============================================================================

def run_gem5_simulation(
    graph_path: str,
    benchmark: str,
    algo_id: int = 0,
    algo_variant: str = "",
    policy: str = "LRU",
    ecg_mode: str = "DBG_PRIMARY",
    prefetcher: str = "none",
    cpu_type: str = "timing",
    graph_metadata_path: str = "",
    gem5_binary: str = "",
    timeout: int = 0,
    output_dir: str = "",
    symmetric: bool = True,
) -> CacheResult:
    """Run a single graph benchmark under gem5 and return cache results.

    Args:
        graph_path: Path to the graph file (.sg or .mtx)
        benchmark: Benchmark name (pr, bfs, bc, cc, sssp, tc, etc.)
        algo_id: Reordering algorithm ID (0-16)
        algo_variant: Algorithm variant (e.g., "csr", "leiden")
        policy: Cache replacement policy for L3
        ecg_mode: ECG mode (only used with policy=ECG)
        prefetcher: Prefetcher ("none" or "DROPLET")
        cpu_type: CPU model ("timing", "O3", "minor")
        graph_metadata_path: Path to graph metadata JSON
        gem5_binary: Override gem5 binary path
        timeout: Timeout in seconds (0 = use default)
        output_dir: Override gem5 output directory
        symmetric: Whether graph is symmetric (-s flag)

    Returns:
        CacheResult with miss rates and metadata.
    """
    # Find binaries
    if not gem5_binary:
        gem5_binary = str(find_gem5_binary())
    bench_binary = str(find_benchmark_binary(benchmark))

    # Build algorithm option string
    if algo_variant:
        algo_opt = f"{algo_id}:{algo_variant}"
    else:
        algo_opt = str(algo_id)

    # Build benchmark options
    sym_flag = "-s" if symmetric else ""
    bench_options = f"-f {graph_path} {sym_flag} -o {algo_opt} -n 1"

    # Determine output directory
    if not output_dir:
        algo_name = canonical_algo_key(algo_id, algo_variant)
        output_dir = str(
            PROJECT_ROOT / "results" / "gem5_runs" /
            f"{Path(graph_path).stem}_{benchmark}_{algo_name}_{policy}"
        )

    # Build gem5 command
    config_script = str(GEM5_CONFIGS_DIR / "graph_se.py")

    cmd_parts = [
        gem5_binary,
        f"--outdir={output_dir}",
        config_script,
        f"--binary={bench_binary}",
        f'--options="{bench_options}"',
        f"--policy={policy}",
        f"--cpu-type={cpu_type}",
        f"--prefetcher={prefetcher}",
    ]

    if policy == "ECG":
        cmd_parts.append(f"--ecg-mode={ecg_mode}")

    if graph_metadata_path:
        cmd_parts.append(f"--graph-metadata={graph_metadata_path}")

    cmd = " ".join(cmd_parts)

    # Determine timeout
    if timeout <= 0:
        timeout = TIMEOUT_GEM5_HEAVY if benchmark in HEAVY_BENCHMARKS else TIMEOUT_GEM5

    # Construct result
    algo_name = canonical_algo_key(algo_id, algo_variant)
    result = CacheResult(
        graph=Path(graph_path).stem,
        algorithm_id=algo_id,
        algorithm_name=algo_name,
        benchmark=benchmark,
    )

    log.info(f"  gem5: {benchmark} × {algo_name} × {policy}"
             f" ({cpu_type}, {prefetcher})")

    # Run gem5
    start_time = time.time()
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout,
        )
        elapsed = time.time() - start_time

        if proc.returncode != 0:
            result.success = False
            result.error = f"gem5 exited with code {proc.returncode}"
            stderr_tail = proc.stderr[-500:] if proc.stderr else ""
            if stderr_tail:
                result.error += f": {stderr_tail}"
            return result

    except subprocess.TimeoutExpired:
        result.success = False
        result.error = f"TIMEOUT after {timeout}s"
        return result

    # Parse stats
    stats_file = Path(output_dir) / "stats.txt"
    if not stats_file.exists():
        result.success = False
        result.error = f"stats.txt not found in {output_dir}"
        return result

    # Import parser
    sys.path.insert(0, str(GEM5_SCRIPTS_DIR))
    from parse_stats import parse_gem5_stats

    stats = parse_gem5_stats(str(stats_file))

    result.l1_miss_rate = stats["l1_miss_rate"]
    result.l2_miss_rate = stats["l2_miss_rate"]
    result.l3_miss_rate = stats["l3_miss_rate"]
    result.l1_misses = stats["l1_misses"]
    result.l2_misses = stats["l2_misses"]
    result.l3_misses = stats["l3_misses"]
    result.success = stats["success"]
    result.error = stats["error"]

    if result.success:
        log.info(f"    L1={result.l1_miss_rate*100:.1f}% "
                 f"L2={result.l2_miss_rate*100:.1f}% "
                 f"L3={result.l3_miss_rate*100:.1f}% "
                 f"({elapsed:.1f}s)")

    return result


# =============================================================================
# Batch Simulation
# =============================================================================

def run_gem5_simulations(
    graphs: List[GraphInfo],
    algorithms: List[int] = None,
    benchmarks: List[str] = None,
    policy: str = "LRU",
    ecg_mode: str = "DBG_PRIMARY",
    prefetcher: str = "none",
    cpu_type: str = "timing",
    **kwargs,
) -> List[CacheResult]:
    """Run gem5 simulations for a matrix of graphs × algorithms × benchmarks.

    Mirrors the interface of scripts/lib/pipeline/cache.run_cache_simulations().

    Args:
        graphs: List of GraphInfo objects
        algorithms: List of algorithm IDs (default: all eligible)
        benchmarks: List of benchmark names (default: all)
        policy: Cache replacement policy
        ecg_mode: ECG mode
        prefetcher: Prefetcher name
        cpu_type: CPU model
        **kwargs: Extra args passed to run_gem5_simulation

    Returns:
        List of CacheResult objects.
    """
    from ..core.utils import ELIGIBLE_ALGORITHMS

    if algorithms is None:
        algorithms = ELIGIBLE_ALGORITHMS
    if benchmarks is None:
        benchmarks = ["pr", "bfs", "bc", "cc", "sssp", "tc"]

    results = []
    total = len(graphs) * len(algorithms) * len(benchmarks)
    count = 0

    for graph in graphs:
        graph_path = str(graph.sg_path) if hasattr(graph, 'sg_path') else str(graph.path)
        graph_name = graph.name if hasattr(graph, 'name') else Path(graph_path).stem

        # Export metadata for this graph
        metadata_path = export_gem5_metadata(graph_name, graph_path)

        for bench in benchmarks:
            for algo_id in algorithms:
                count += 1
                log.info(f"[{count}/{total}] gem5: {graph_name} × {bench} "
                         f"× algo={algo_id} × {policy}")

                result = run_gem5_simulation(
                    graph_path=graph_path,
                    benchmark=bench,
                    algo_id=algo_id,
                    policy=policy,
                    ecg_mode=ecg_mode,
                    prefetcher=prefetcher,
                    cpu_type=cpu_type,
                    graph_metadata_path=str(metadata_path),
                    symmetric=getattr(graph, 'symmetric', True),
                    **kwargs,
                )
                results.append(result)

    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run gem5 graph simulation")
    parser.add_argument("--graph", required=True, help="Path to graph file")
    parser.add_argument("--benchmark", required=True, help="Benchmark name")
    parser.add_argument("--algorithm", type=int, default=0, help="Algorithm ID")
    parser.add_argument("--policy", default="LRU", help="Cache policy")
    parser.add_argument("--ecg-mode", default="DBG_PRIMARY")
    parser.add_argument("--prefetcher", default="none")
    parser.add_argument("--cpu-type", default="timing")

    args = parser.parse_args()

    result = run_gem5_simulation(
        graph_path=args.graph,
        benchmark=args.benchmark,
        algo_id=args.algorithm,
        policy=args.policy,
        ecg_mode=args.ecg_mode,
        prefetcher=args.prefetcher,
        cpu_type=args.cpu_type,
    )

    if result.success:
        print(f"L1 miss rate: {result.l1_miss_rate*100:.2f}%")
        print(f"L2 miss rate: {result.l2_miss_rate*100:.2f}%")
        print(f"L3 miss rate: {result.l3_miss_rate*100:.2f}%")
    else:
        print(f"Error: {result.error}")
        sys.exit(1)
