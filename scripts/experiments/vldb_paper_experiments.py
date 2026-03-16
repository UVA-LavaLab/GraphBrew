#!/usr/bin/env python3
"""
VLDB 2026 GraphBrew Paper — Experiment Runner.

Reproduces all figures and tables from the paper. Each experiment
dumps structured JSON results; a final step generates publication-
ready figures (PNG/PDF) and LaTeX table snippets.

Usage:
    # Full reproducibility run (experiments + figures):
    python scripts/experiments/vldb_paper_experiments.py --all --graph-dir /data/graphs

    # Experiments only (no figure generation):
    python scripts/experiments/vldb_paper_experiments.py --all --graph-dir /data/graphs --no-figures

    # Figures only (from previously saved results):
    python scripts/experiments/vldb_paper_experiments.py --figures-only

    # Preview mode (small graphs, 1 trial, 2 benchmarks):
    python scripts/experiments/vldb_paper_experiments.py --all --preview --graph-dir /data/graphs

    # Dry run (print commands without executing):
    python scripts/experiments/vldb_paper_experiments.py --all --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from experiments.vldb_config import (
    ABLATION_CONFIGS,
    ALL_ALGORITHMS,
    BASELINE_ALGORITHMS,
    BENCHMARKS,
    BENCHMARKS_PREVIEW,
    BIN_DIR,
    BIN_SIM_DIR,
    CACHE_SIZES,
    CHAINED_ORDERINGS,
    EVAL_GRAPHS,
    EVAL_GRAPHS_64GB,
    FIGURES_DIR,
    GRAPH_TYPE_GROUPS,
    GRAPHBREW_VARIANTS,
    PREVIEW_GRAPHS,
    RESULTS_DIR,
    TABLES_DIR,
    THREAD_COUNTS,
    TIMEOUT_FULL,
    TIMEOUT_PREVIEW,
    TRIALS_FULL,
    TRIALS_PREVIEW,
    VLDB_GRAPH_SOURCES,
    get_converter_flags,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vldb_paper")


# ============================================================================
# Helpers
# ============================================================================


def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_cmd(
    cmd: list[str],
    dry_run: bool = False,
    timeout: int = 3600,
    env: Optional[dict] = None,
) -> Optional[str]:
    """Run a command and return stdout, or None on failure."""
    cmd_str = " ".join(str(c) for c in cmd)
    log.info(f"  CMD: {cmd_str}")
    if dry_run:
        return ""
    merged_env = {**os.environ, **(env or {})}
    try:
        result = subprocess.run(
            cmd, timeout=timeout, capture_output=True, text=True, env=merged_env,
        )
        if result.returncode != 0:
            log.error(f"  FAILED (rc={result.returncode}): {result.stderr[:500]}")
            return None
        return result.stdout
    except subprocess.TimeoutExpired:
        log.error(f"  TIMEOUT after {timeout}s")
        return None
    except Exception as e:
        log.error(f"  ERROR: {e}")
        return None


def save_json(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    log.info(f"  Saved: {path}")


def parse_timing(output: Optional[str]) -> dict:
    """Extract timing fields from benchmark stdout.

    GraphBrew binaries print lines like:
        Trial Time:       0.1234
        Reorder Time:     0.0567
        Average Time:     0.1100

    For chained orderings, multiple ``Reorder Time:`` lines may appear.
    These are **summed** to give the total reorder cost.
    """
    result: dict = {}
    if not output:
        return result
    reorder_times: list[float] = []
    for line in output.splitlines():
        line = line.strip()
        # Collect all Reorder Time values (summed below)
        if line.startswith("Reorder Time"):
            try:
                reorder_times.append(float(line.split(":", 1)[1].strip().split()[0]))
            except (ValueError, IndexError):
                pass
            continue
        for key in ("Trial Time", "Average Time",
                    "Preprocessing Time", "Total Time",
                    "Topology Analysis Time", "Read Time",
                    "Relabel Map Time"):
            if line.startswith(key):
                try:
                    result[key.lower().replace(" ", "_")] = float(
                        line.split(":", 1)[1].strip().split()[0]
                    )
                except (ValueError, IndexError):
                    pass
    if reorder_times:
        result["reorder_time"] = sum(reorder_times)
    return result


def parse_cache_sim(output: Optional[str]) -> dict:
    """Extract cache simulation metrics from sim binary stdout.

    The sim binary outputs a formatted table like:
        ║ L1 Cache (32KB, 8-way, Clock)
        ║   Hits:                       110358
        ║   Misses:                         67
        ║   Hit Rate:                 99.9393%
        ║ L2 Cache ...
        ║ Total Accesses:                98188
        ║ Memory Accesses:                  64
        ║ Overall Hit Rate:           99.9348%
    """
    result: dict = {}
    if not output:
        return result
    current_level = ""
    for line in output.splitlines():
        stripped = line.strip().strip("║").strip()
        if not stripped:
            continue
        # Detect cache level header: "L1 Cache (32KB, ...)"
        if stripped.startswith("L1 Cache"):
            current_level = "l1"
        elif stripped.startswith("L2 Cache"):
            current_level = "l2"
        elif stripped.startswith("L3 Cache"):
            current_level = "l3"
        elif stripped.startswith("SUMMARY"):
            current_level = "summary"
        elif current_level and ":" in stripped:
            key_part, _, val_part = stripped.partition(":")
            key_part = key_part.strip().lower().replace(" ", "_")
            val_part = val_part.strip().rstrip("%")
            try:
                val = float(val_part)
                if current_level == "summary":
                    result[key_part] = val
                else:
                    result[f"{current_level}_{key_part}"] = val
            except (ValueError, IndexError):
                pass
    return result


def git_revision() -> str:
    """Return short git hash, or 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT), text=True, timeout=5,
        ).strip()
    except Exception:
        return "unknown"


def save_manifest(args: argparse.Namespace, elapsed: float) -> None:
    """Write a reproducibility manifest with config + environment info."""
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "git_revision": git_revision(),
        "platform": platform.platform(),
        "python": sys.version,
        "args": vars(args),
        "elapsed_seconds": round(elapsed, 1),
        "config": {
            "baselines": list(BASELINE_ALGORITHMS.values()),
            "graphbrew_variants": GRAPHBREW_VARIANTS,
            "chained_orderings": [c[0] for c in CHAINED_ORDERINGS],
            "benchmarks": BENCHMARKS,
            "graphs": [g["name"] for g in EVAL_GRAPHS],
        },
    }
    save_json(manifest, RESULTS_DIR / "MANIFEST.json")


# ---------------------------------------------------------------------------
# Mapping (.lo) pre-generation infrastructure
# ---------------------------------------------------------------------------

MAPPINGS_DIR = PROJECT_ROOT / "results" / "vldb_mappings"


def _lo_path(graph_name: str, algo_key: str) -> Path:
    """Path for a pre-generated label-order mapping file."""
    safe = algo_key.replace(":", "_").replace("/", "_")
    return MAPPINGS_DIR / graph_name / f"{safe}.lo"


def _time_path(graph_name: str, algo_key: str) -> Path:
    """Path for the recorded reorder time alongside a .lo file."""
    safe = algo_key.replace(":", "_").replace("/", "_")
    return MAPPINGS_DIR / graph_name / f"{safe}.time"


def _load_reorder_time(graph_name: str, algo_key: str) -> float:
    """Load pre-recorded reorder time from .time file, or 0.0."""
    tp = _time_path(graph_name, algo_key)
    if tp.exists():
        try:
            return float(tp.read_text().strip())
        except (ValueError, OSError):
            pass
    return 0.0


def _pregenerate_mappings(
    graphs: list[dict],
    graph_dir: str,
    dry_run: bool = False,
    timeout: int = 1800,
) -> None:
    """Pre-generate .lo mapping files for all (graph, algorithm) pairs.

    Runs the converter once per pair with ``-q {lo_path}`` to produce
    a vertex-permutation file.  Also records ``Reorder Time:`` from
    converter stdout into a ``.time`` companion file.

    Subsequent experiments use MAP mode (``-o 13:{lo_path}``) so the
    benchmark binary loads the pre-computed ordering with zero reorder
    cost, and all benchmarks for the same graph×algorithm see exactly
    the same ordering.
    """
    converter = BIN_DIR / "converter"
    if not converter.exists():
        log.warning("  Converter binary not found — skipping .lo pre-generation")
        return

    # Build the full algo list: baselines + GB variants + chained
    algo_list: list[tuple[str, list[str]]] = []  # (key, flags)
    for aid, _aname in BASELINE_ALGORITHMS.items():
        if aid == 0:
            continue  # ORIGINAL — no mapping needed
        algo_list.append((str(aid), ["-o", str(aid)]))
    for v in GRAPHBREW_VARIANTS:
        algo_list.append((f"12:{v}", ["-o", f"12:{v}"]))
    for chain_name, chain_flags in CHAINED_ORDERINGS:
        algo_list.append((f"chain:{chain_name}", chain_flags))
    # Ablation configs that aren't already covered
    for cfg in ABLATION_CONFIGS:
        key = cfg["algo"]
        if key == "0":
            continue
        if not any(k == key for k, _ in algo_list):
            algo_list.append((key, get_converter_flags(key)))

    generated = 0
    skipped = 0
    failed = 0

    for graph in graphs:
        gname = graph["name"]
        sg = resolve_graph_path(gname, graph_dir, ext=".sg")
        if not Path(sg).exists():
            log.warning(f"  {gname}: no .sg file — skipping")
            continue

        for algo_key, aflags in algo_list:
            lo = _lo_path(gname, algo_key)
            tf = _time_path(gname, algo_key)

            if lo.exists() and lo.stat().st_size > 0:
                skipped += 1
                continue

            if dry_run:
                log.info(f"  [dry-run] {gname} → {algo_key}")
                skipped += 1
                continue

            lo.parent.mkdir(parents=True, exist_ok=True)

            # Converter needs -b (output .sg) even though we only want -q (.lo).
            # Use a tempfile that gets discarded.
            with tempfile.NamedTemporaryFile(suffix=".sg", delete=True) as tmp:
                cmd = [str(converter), "-f", sg, "-s"]
                cmd.extend(aflags)
                cmd.extend(["-b", tmp.name, "-q", str(lo)])
                output = run_cmd(cmd, dry_run=False, timeout=timeout)

            if output is None or not lo.exists() or lo.stat().st_size == 0:
                log.warning(f"  FAILED: {gname} → {algo_key}")
                if lo.exists():
                    lo.unlink()
                failed += 1
                continue

            # Save reorder time (sum of all Reorder Time: lines)
            reorder_times = re.findall(r"Reorder Time:\s*([\d.]+)", output)
            if reorder_times:
                total = sum(float(t) for t in reorder_times)
                tf.parent.mkdir(parents=True, exist_ok=True)
                tf.write_text(str(total))

            generated += 1

    log.info(f"  Mappings: {generated} generated, {skipped} existing, {failed} failed")


def algo_flags_or_map(
    algo_key: str, algo_flags: list[str], graph_name: str,
) -> tuple[list[str], float]:
    """Return (flags, prerecorded_reorder_time) using MAP mode if .lo exists.

    If a pre-generated .lo file exists for this (graph, algo), returns
    ``["-o", "13:{lo_path}"]`` so the benchmark loads the cached
    ordering with zero runtime reorder cost.  The recorded reorder time
    from the ``.time`` file is returned as the second element.

    Otherwise falls back to the original *algo_flags* (runtime reorder).
    """
    lo = _lo_path(graph_name, algo_key)
    if lo.exists() and lo.stat().st_size > 0:
        rt = _load_reorder_time(graph_name, algo_key)
        return ["-o", f"13:{lo}"], rt
    return algo_flags, 0.0


def build_benchmark_cmd(
    benchmark: str, graph_path: str, algo_flags: list[str], trials: int = 3,
    sim: bool = False,
) -> list[str]:
    """Build the CLI command to run a benchmark with a reorder algorithm."""
    bin_dir = BIN_SIM_DIR if sim else BIN_DIR
    binary = bin_dir / benchmark
    cmd = [str(binary), "-f", graph_path, "-s", "-n", str(trials)]
    cmd.extend(algo_flags)
    return cmd


def resolve_graph_path(graph_name: str, graph_dir: str, ext: str = ".sg") -> str:
    """Build the full path to a graph file.

    Checks two layouts:
      1. flat:   graph_dir/name.sg
      2. nested: graph_dir/name/name.sg   (created by auto-setup download)
    Returns the first that exists, or the flat path if neither does.
    """
    flat = Path(graph_dir) / f"{graph_name}{ext}"
    nested = Path(graph_dir) / graph_name / f"{graph_name}{ext}"
    if nested.exists():
        return str(nested)
    return str(flat)


# ============================================================================
# Experiment 1: Cache Performance Analysis
# ============================================================================


def exp1_cache_performance(
    graphs: list[dict], benchmarks: list[str], trials: int,
    timeout: int, dry_run: bool, graph_dir: str = ".",
) -> None:
    log.info("=" * 60)
    log.info("EXPERIMENT 1: Cache Performance Analysis")
    log.info("=" * 60)

    out_dir = ensure_dir(RESULTS_DIR / "exp1_cache")
    results = []

    # Use PR for cache simulation (canonical benchmark)
    cache_bench = "pr"
    if cache_bench not in benchmarks:
        cache_bench = benchmarks[0]

    for graph in graphs:
        gname = graph["name"]
        log.info(f"  Graph: {gname}")

        # Collect all algorithms to test
        algo_list: list[tuple[str, str, list[str]]] = []  # (name, key, flags)

        # Baselines
        for aid, aname in BASELINE_ALGORITHMS.items():
            algo_list.append((aname, str(aid), ["-o", str(aid)]))

        # GraphBrew variants
        for v in GRAPHBREW_VARIANTS:
            algo_list.append((f"GB-{v}", f"12:{v}", ["-o", f"12:{v}"]))

        for aname, akey, aflags in algo_list:
            gpath = resolve_graph_path(gname, graph_dir)
            flags, pregen_rt = algo_flags_or_map(akey, aflags, gname)
            cmd = build_benchmark_cmd(cache_bench, gpath, flags, trials, sim=True)
            log.info(f"    {aname}")
            output = run_cmd(cmd, dry_run=dry_run, timeout=timeout)
            timing = parse_timing(output)
            cache = parse_cache_sim(output)
            if pregen_rt > 0 and "reorder_time" not in timing:
                timing["reorder_time"] = pregen_rt
            results.append({
                "graph": gname, "algorithm": aname, "benchmark": cache_bench,
                **timing, **cache,
            })

    save_json(results, out_dir / "cache_results.json")


# ============================================================================
# Experiment 2: Kernel Speedup
# ============================================================================


def exp2_kernel_speedup(
    graphs: list[dict], benchmarks: list[str], trials: int,
    timeout: int, dry_run: bool, graph_dir: str = ".",
) -> None:
    log.info("=" * 60)
    log.info("EXPERIMENT 2: Kernel Speedup")
    log.info("=" * 60)

    out_dir = ensure_dir(RESULTS_DIR / "exp2_speedup")
    results = []

    for graph in graphs:
        gname = graph["name"]
        log.info(f"  Graph: {gname}")

        for bench in benchmarks:
            log.info(f"    Benchmark: {bench}")

            # All baselines
            for aid, aname in BASELINE_ALGORITHMS.items():
                gpath = resolve_graph_path(gname, graph_dir)
                flags, pregen_rt = algo_flags_or_map(str(aid), ["-o", str(aid)], gname)
                cmd = build_benchmark_cmd(bench, gpath, flags, trials)
                output = run_cmd(cmd, dry_run=dry_run, timeout=timeout)
                timing = parse_timing(output)
                if pregen_rt > 0 and "reorder_time" not in timing:
                    timing["reorder_time"] = pregen_rt
                results.append({
                    "graph": gname, "algorithm": aname, "benchmark": bench,
                    "algo_id": aid, **timing,
                })

            # GraphBrew variants
            for v in GRAPHBREW_VARIANTS:
                gpath = resolve_graph_path(gname, graph_dir)
                flags, pregen_rt = algo_flags_or_map(f"12:{v}", ["-o", f"12:{v}"], gname)
                cmd = build_benchmark_cmd(bench, gpath, flags, trials)
                output = run_cmd(cmd, dry_run=dry_run, timeout=timeout)
                timing = parse_timing(output)
                if pregen_rt > 0 and "reorder_time" not in timing:
                    timing["reorder_time"] = pregen_rt
                results.append({
                    "graph": gname, "algorithm": f"GB-{v}", "benchmark": bench,
                    "algo_id": f"12:{v}", **timing,
                })

    save_json(results, out_dir / "speedup_results.json")


# ============================================================================
# Experiment 3: Reorder Overhead & Amortization
# ============================================================================


def exp3_reorder_overhead(
    graphs: list[dict], benchmarks: list[str], trials: int,
    timeout: int, dry_run: bool, graph_dir: str = ".",
) -> None:
    log.info("=" * 60)
    log.info("EXPERIMENT 3: Reorder Overhead & Amortization")
    log.info("=" * 60)

    out_dir = ensure_dir(RESULTS_DIR / "exp3_overhead")
    results = []

    for graph in graphs:
        gname = graph["name"]
        log.info(f"  Graph: {gname}")

        converter = BIN_DIR / "converter"
        for aid, aname in {**BASELINE_ALGORITHMS}.items():
            if aid == 0:
                continue  # No reorder for original
            # Try .sg first (auto-setup creates .sg); fall back to .el
            gpath = resolve_graph_path(gname, graph_dir, ext=".sg")
            if not Path(gpath).exists():
                gpath = resolve_graph_path(gname, graph_dir, ext=".el")
            cmd = [str(converter), "-f", gpath, "-o", str(aid)]
            output = run_cmd(cmd, dry_run=dry_run, timeout=timeout)
            timing = parse_timing(output)
            results.append({
                "graph": gname, "algorithm": aname, "algo_id": aid, **timing,
            })

        for v in GRAPHBREW_VARIANTS:
            gpath = resolve_graph_path(gname, graph_dir, ext=".sg")
            if not Path(gpath).exists():
                gpath = resolve_graph_path(gname, graph_dir, ext=".el")
            cmd = [str(converter), "-f", gpath, "-o", f"12:{v}"]
            output = run_cmd(cmd, dry_run=dry_run, timeout=timeout)
            timing = parse_timing(output)
            results.append({
                "graph": gname, "algorithm": f"GB-{v}", "algo_id": f"12:{v}", **timing,
            })

    save_json(results, out_dir / "overhead_results.json")


# ============================================================================
# Experiment 4: End-to-End Performance
# ============================================================================


def exp4_end_to_end(
    graphs: list[dict], benchmarks: list[str], trials: int,
    timeout: int, dry_run: bool, graph_dir: str = ".",
) -> None:
    log.info("=" * 60)
    log.info("EXPERIMENT 4: End-to-End Performance")
    log.info("=" * 60)
    log.info("  (Combines reorder overhead + kernel execution)")
    log.info("  Results derived from Exp 2 + Exp 3 data.")

    # E2E is computed as reorder_time + trials * kernel_time
    # No additional commands needed; analysis done in figure generation.
    out_dir = ensure_dir(RESULTS_DIR / "exp4_e2e")
    save_json({"note": "Derived from exp2 + exp3 data"}, out_dir / "e2e_note.json")


# ============================================================================
# Experiment 5: Variant Ablation Study
# ============================================================================


def exp5_ablation(
    graphs: list[dict], benchmarks: list[str], trials: int,
    timeout: int, dry_run: bool, graph_dir: str = ".",
) -> None:
    log.info("=" * 60)
    log.info("EXPERIMENT 5: Variant Ablation Study")
    log.info("=" * 60)

    out_dir = ensure_dir(RESULTS_DIR / "exp5_ablation")
    results = []

    # Focus on PR for ablation (most representative iterative algorithm)
    abl_bench = "pr"

    for graph in graphs:
        gname = graph["name"]
        log.info(f"  Graph: {gname}")

        for config in ABLATION_CONFIGS:
            algo_key = config["algo"]
            aflags = get_converter_flags(algo_key)
            gpath = resolve_graph_path(gname, graph_dir)
            flags, pregen_rt = algo_flags_or_map(algo_key, aflags, gname)
            cmd = build_benchmark_cmd(abl_bench, gpath, flags, trials)
            output = run_cmd(cmd, dry_run=dry_run, timeout=timeout)
            timing = parse_timing(output)
            if pregen_rt > 0 and "reorder_time" not in timing:
                timing["reorder_time"] = pregen_rt
            results.append({
                "graph": gname, "config": config["name"],
                "algo": config["algo"], "desc": config["desc"],
                "benchmark": abl_bench, **timing,
            })

    save_json(results, out_dir / "ablation_results.json")


# ============================================================================
# Experiment 6: Graph-Type Sensitivity
# ============================================================================


def exp6_sensitivity(
    graphs: list[dict], benchmarks: list[str], trials: int,
    timeout: int, dry_run: bool, graph_dir: str = ".",
) -> None:
    log.info("=" * 60)
    log.info("EXPERIMENT 6: Graph-Type Sensitivity")
    log.info("=" * 60)
    log.info("  Results derived from Exp 2 data, grouped by graph type.")

    out_dir = ensure_dir(RESULTS_DIR / "exp6_sensitivity")
    save_json(
        {"note": "Analysis performed by vldb_generate_figures.py from exp2 data",
         "groups": GRAPH_TYPE_GROUPS},
        out_dir / "sensitivity_note.json",
    )


# ============================================================================
# Experiment 7: Chained Ordering Analysis
# ============================================================================


def exp7_chained(
    graphs: list[dict], benchmarks: list[str], trials: int,
    timeout: int, dry_run: bool, graph_dir: str = ".",
) -> None:
    log.info("=" * 60)
    log.info("EXPERIMENT 7: Chained Ordering Analysis")
    log.info("=" * 60)

    out_dir = ensure_dir(RESULTS_DIR / "exp7_chained")
    results = []

    chain_bench = "pr"

    for graph in graphs:
        gname = graph["name"]
        log.info(f"  Graph: {gname}")

        for chain_name, chain_flags in CHAINED_ORDERINGS:
            gpath = resolve_graph_path(gname, graph_dir)
            chain_key = f"chain:{chain_name}"
            flags, pregen_rt = algo_flags_or_map(chain_key, chain_flags, gname)
            cmd = build_benchmark_cmd(chain_bench, gpath, flags, trials)
            output = run_cmd(cmd, dry_run=dry_run, timeout=timeout)
            timing = parse_timing(output)
            if pregen_rt > 0 and "reorder_time" not in timing:
                timing["reorder_time"] = pregen_rt
            results.append({
                "graph": gname, "chain": chain_name,
                "flags": chain_flags, "benchmark": chain_bench, **timing,
            })

    save_json(results, out_dir / "chained_results.json")


# ============================================================================
# Experiment 8: Reorder Scalability
# ============================================================================


def exp8_scalability(
    graphs: list[dict], benchmarks: list[str], trials: int,
    timeout: int, dry_run: bool, graph_dir: str = ".",
) -> None:
    log.info("=" * 60)
    log.info("EXPERIMENT 8: Reorder Scalability")
    log.info("=" * 60)

    out_dir = ensure_dir(RESULTS_DIR / "exp8_scalability")
    results = []

    # Test GraphBrew variants and Gorder at different thread counts
    test_algos = [
        ("GB-leiden", ["-o", "12:leiden"]),
        ("GB-hrab",   ["-o", "12:hrab"]),
        ("GB-rabbit", ["-o", "12:rabbit"]),
        ("Gorder",    ["-o", "9"]),
        ("RabbitOrder",["-o", "8"]),
    ]

    converter = BIN_DIR / "converter"
    for graph in graphs:
        gname = graph["name"]
        log.info(f"  Graph: {gname}")

        for aname, aflags in test_algos:
            for nthreads in THREAD_COUNTS:
                env = {"OMP_NUM_THREADS": str(nthreads)}
                # Try .sg first (auto-setup creates .sg); fall back to .el
                gpath = resolve_graph_path(gname, graph_dir, ext=".sg")
                if not Path(gpath).exists():
                    gpath = resolve_graph_path(gname, graph_dir, ext=".el")
                cmd = [str(converter), "-f", gpath] + aflags
                output = run_cmd(cmd, dry_run=dry_run, timeout=timeout, env=env)
                timing = parse_timing(output)
                results.append({
                    "graph": gname, "algorithm": aname, "threads": nthreads,
                    **timing,
                })

    save_json(results, out_dir / "scalability_results.json")


# ============================================================================
# Main
# ============================================================================

EXPERIMENTS = {
    1: ("Cache Performance Analysis", exp1_cache_performance),
    2: ("Kernel Speedup", exp2_kernel_speedup),
    3: ("Reorder Overhead & Amortization", exp3_reorder_overhead),
    4: ("End-to-End Performance", exp4_end_to_end),
    5: ("Variant Ablation Study", exp5_ablation),
    6: ("Graph-Type Sensitivity", exp6_sensitivity),
    7: ("Chained Ordering Analysis", exp7_chained),
    8: ("Reorder Scalability", exp8_scalability),
}


# ============================================================================
# Auto-Setup: dependencies, binaries, graph download & conversion
# ============================================================================

def _setup_environment(
    graph_dir: str,
    graphs: list[dict],
    dry_run: bool = False,
    skip_download: bool = False,
) -> str:
    """Ensure binaries are built, graphs are downloaded, and .sg files exist.

    Returns the *resolved* graph directory (may differ from input when
    graphs live under ``results/graphs``).
    """
    log.info("=" * 60)
    log.info("  AUTO-SETUP")
    log.info("=" * 60)

    graphs_path = Path(graph_dir) if graph_dir != "." else PROJECT_ROOT / "results" / "graphs"
    graphs_path.mkdir(parents=True, exist_ok=True)
    graph_dir_resolved = str(graphs_path)

    if dry_run:
        log.info("  [dry-run] Skipping auto-setup")
        return graph_dir_resolved

    # ── 1. Python dependencies ──────────────────────────────────────────
    log.info("\n── Step 1/5: Python dependencies ──")
    try:
        import matplotlib  # noqa: F401
        log.info("  matplotlib: OK")
    except ImportError:
        log.warning("  matplotlib not installed — figures will be skipped")
        log.info("  Install with: pip install matplotlib numpy")

    # ── 2. Build binaries ───────────────────────────────────────────────
    log.info("\n── Step 2/5: Build binaries ──")
    _setup_build_binaries()

    # ── 3. Download graphs ──────────────────────────────────────────────
    log.info("\n── Step 3/5: Download graphs ──")
    if skip_download:
        log.info("  Skipping download (--skip-download)")
    else:
        _setup_download_graphs(graphs, graphs_path)

    # ── 4. Convert .mtx → .sg ──────────────────────────────────────────
    log.info("\n── Step 4/5: Convert graphs to .sg ──")
    _setup_convert_graphs(graphs, graphs_path)

    # ── 5. Pre-generate .lo mapping files ──────────────────────────────
    log.info("\n── Step 5/5: Pre-generate reorder mappings (.lo) ──")
    _pregenerate_mappings(graphs, graph_dir_resolved, dry_run=dry_run)

    log.info("\n" + "=" * 60)
    log.info("  AUTO-SETUP COMPLETE")
    log.info("=" * 60 + "\n")
    return graph_dir_resolved


def _setup_build_binaries() -> None:
    """Build standard and cache-simulation binaries if missing."""
    missing_std = [b for b in BENCHMARKS if not (BIN_DIR / b).exists()]
    missing_sim = [b for b in BENCHMARKS if not (BIN_SIM_DIR / b).exists()]
    converter = BIN_DIR / "converter"

    if not missing_std and not missing_sim and converter.exists():
        log.info("  All binaries present ✓")
        return

    makefile = PROJECT_ROOT / "Makefile"
    if not makefile.exists():
        log.error("  Makefile not found — cannot build binaries!")
        return

    if missing_std or not converter.exists():
        log.info(f"  Building standard binaries ({len(missing_std)} missing)...")
        result = subprocess.run(
            ["make", "-j", str(os.cpu_count() or 4)],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True,
        )
        if result.returncode != 0:
            log.error(f"  Build failed: {result.stderr[:300]}")
        else:
            log.info("  Standard binaries: OK ✓")

    if missing_sim:
        log.info(f"  Building simulation binaries ({len(missing_sim)} missing)...")
        result = subprocess.run(
            ["make", "all-sim", "-j", str(os.cpu_count() or 4)],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True,
        )
        if result.returncode != 0:
            log.error(f"  Sim build failed: {result.stderr[:300]}")
        else:
            log.info("  Simulation binaries: OK ✓")


def _setup_download_graphs(graphs: list[dict], dest_dir: Path) -> None:
    """Download evaluation graphs from SuiteSparse; report manual-download graphs."""
    # Collect graphs that need downloading from catalog
    catalog_names = []
    manual_graphs = []

    for g in graphs:
        name = g["name"]
        sg_path = dest_dir / name / f"{name}.sg"
        el_path = dest_dir / name / f"{name}.el"
        # Already have .sg or .el — skip
        if sg_path.exists() or el_path.exists():
            log.info(f"  {name}: already present ✓")
            continue

        src = VLDB_GRAPH_SOURCES.get(name, {})
        if src.get("source") == "catalog":
            catalog_names.append(name)
        elif src.get("source") == "manual":
            manual_graphs.append((name, src))
        else:
            # Graph not in VLDB sources — check if .mtx exists
            mtx_dir = dest_dir / name
            if mtx_dir.exists():
                log.info(f"  {name}: directory exists (will convert)")
            else:
                log.warning(f"  {name}: not in download catalog — place .sg/.el manually")

    # Download from catalog
    if catalog_names:
        log.info(f"  Downloading {len(catalog_names)} graphs from SuiteSparse...")
        try:
            from lib.pipeline.download import (
                download_graphs_parallel,
                get_graph_info,
                DownloadableGraph,
            )
            # Build DownloadableGraph list for graphs in the catalog
            to_download = []
            for name in catalog_names:
                info = get_graph_info(name)
                if info:
                    to_download.append(info)
                else:
                    log.warning(f"  {name}: not found in download catalog")
            if to_download:
                paths, failed = download_graphs_parallel(
                    graphs=to_download,
                    dest_dir=dest_dir,
                    max_workers=min(4, len(to_download)),
                    show_progress=True,
                )
                log.info(f"  Downloaded {len(paths)} graphs, {len(failed)} failed")
                for name in failed:
                    log.warning(f"    FAILED: {name}")
        except Exception as e:
            log.error(f"  Download failed: {e}")

    # Report manual-download graphs
    if manual_graphs:
        log.info("")
        log.info("  ┌─ MANUAL DOWNLOAD REQUIRED ─────────────────────────────")
        for name, src in manual_graphs:
            log.info(f"  │ {name}:")
            for line in src.get("instructions", "").split("\n"):
                log.info(f"  │   {line}")
        log.info("  └─────────────────────────────────────────────────────────")


def _setup_convert_graphs(graphs: list[dict], graphs_dir: Path) -> None:
    """Convert downloaded .mtx files to .sg format."""
    converter = BIN_DIR / "converter"
    if not converter.exists():
        log.warning("  Converter binary not found — skipping conversion")
        return

    converted = 0
    skipped = 0
    for g in graphs:
        name = g["name"]
        graph_subdir = graphs_dir / name
        sg_path = graph_subdir / f"{name}.sg"

        if sg_path.exists() and sg_path.stat().st_size > 0:
            skipped += 1
            continue

        # Find a convertible file (.mtx or .el)
        # Prefer exact-name match (name.mtx) to avoid picking auxiliary files
        # like *_nodename.mtx which are metadata, not graph data.
        input_file = None
        if graph_subdir.exists():
            for pattern in ("**/*.mtx", "**/*.el"):
                matches = sorted(graph_subdir.glob(pattern))
                if matches:
                    # Prefer file named exactly {name}.{ext}
                    exact = [m for m in matches if m.stem == name]
                    input_file = exact[0] if exact else matches[0]
                    break

        if not input_file:
            continue

        log.info(f"  Converting {name}...")
        result = subprocess.run(
            [str(converter), "-f", str(input_file), "-s", "-o", "1",
             "-b", str(sg_path)],
            capture_output=True, text=True, timeout=1800,
        )
        if result.returncode == 0 and sg_path.exists():
            sz_mb = sg_path.stat().st_size / (1024 * 1024)
            log.info(f"    → {sz_mb:.0f} MB ✓")
            converted += 1
        else:
            log.warning(f"    Conversion failed for {name}")

    log.info(f"  Conversion: {converted} new, {skipped} already existed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VLDB 2026 GraphBrew Paper — Experiment Runner"
    )
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--exp", nargs="+", type=int, choices=range(1, 9),
                        help="Run specific experiment(s) by number")
    parser.add_argument("--preview", action="store_true",
                        help="Preview mode: small graphs, 1 trial, 2 benchmarks")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--graphs", nargs="+",
                        help="Override graph list (by name)")
    parser.add_argument("--graph-dir", type=str, default=".",
                        help="Directory containing graph files (.sg, .el)")
    parser.add_argument("--64gb", action="store_true", dest="use_64gb",
                        help="Use 64 GB graph set (11 auto-downloadable graphs, no >1B-edge graphs)")
    parser.add_argument("--skip-setup", action="store_true",
                        help="Skip auto-setup (build, download, convert)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip graph download (use existing graphs only)")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip figure generation after experiments")
    parser.add_argument("--figures-only", action="store_true",
                        help="Only generate figures from existing results (no experiments)")
    args = parser.parse_args()

    if not args.all and not args.exp and not args.figures_only:
        parser.print_help()
        sys.exit(1)

    # ---- Figures-only mode ----
    if args.figures_only:
        log.info("Generating figures from existing results...")
        _generate_figures()
        return

    # Select configuration
    if args.preview:
        graphs = PREVIEW_GRAPHS
        benchmarks = BENCHMARKS_PREVIEW
        trials = TRIALS_PREVIEW
        timeout = TIMEOUT_PREVIEW
    elif getattr(args, "use_64gb", False):
        graphs = EVAL_GRAPHS_64GB
        benchmarks = BENCHMARKS
        trials = TRIALS_FULL
        timeout = TIMEOUT_FULL
    else:
        graphs = EVAL_GRAPHS
        benchmarks = BENCHMARKS
        trials = TRIALS_FULL
        timeout = TIMEOUT_FULL

    # Override graphs if specified
    if args.graphs:
        graphs = [g for g in (EVAL_GRAPHS + PREVIEW_GRAPHS) if g["name"] in args.graphs]
        if not graphs:
            graphs = [{"name": g, "short": g, "type": "unknown", "vertices_m": 0, "edges_m": 0}
                      for g in args.graphs]

    # Determine which experiments to run
    exp_ids = list(range(1, 9)) if args.all else (args.exp or [])

    # ── Auto-setup: build, download, convert ──
    if not args.skip_setup:
        graph_dir_resolved = _setup_environment(
            args.graph_dir, graphs,
            dry_run=args.dry_run,
            skip_download=getattr(args, "skip_download", False),
        )
    else:
        # Still resolve default graph directory even when skipping setup
        if args.graph_dir == ".":
            graph_dir_resolved = str(PROJECT_ROOT / "results" / "graphs")
        else:
            graph_dir_resolved = args.graph_dir

    log.info(f"GraphBrew VLDB Paper Experiments")
    log.info(f"  Mode: {'preview' if args.preview else 'full'}")
    log.info(f"  Graphs: {len(graphs)} in {graph_dir_resolved}")
    log.info(f"  Benchmarks: {benchmarks}")
    log.info(f"  Trials: {trials}")
    log.info(f"  Experiments: {exp_ids}")
    log.info(f"  Dry run: {args.dry_run}")
    log.info("")

    ensure_dir(RESULTS_DIR)
    start = time.time()

    for eid in exp_ids:
        name, func = EXPERIMENTS[eid]
        log.info(f"\n{'#' * 60}")
        log.info(f"# Starting Experiment {eid}: {name}")
        log.info(f"{'#' * 60}\n")
        func(graphs, benchmarks, trials, timeout, args.dry_run,
             graph_dir=graph_dir_resolved)

    elapsed = time.time() - start

    # Save reproducibility manifest
    save_manifest(args, elapsed)

    log.info(f"\nAll experiments completed in {elapsed:.1f}s")
    log.info(f"Results: {RESULTS_DIR}")
    log.info(f"Manifest: {RESULTS_DIR / 'MANIFEST.json'}")

    # Auto-generate figures unless --no-figures
    if not args.no_figures and not args.dry_run:
        log.info("\n--- Generating figures ---")
        _generate_figures()


def _generate_figures() -> None:
    """Invoke the figure generator on saved results."""
    fig_script = PROJECT_ROOT / "scripts" / "experiments" / "vldb_generate_figures.py"
    cmd = [sys.executable, str(fig_script)]

    # If results don't have real data yet, use sample data
    has_data = (RESULTS_DIR / "exp2_speedup" / "speedup_results.json").exists()
    if not has_data:
        cmd.append("--sample-data")

    log.info(f"  CMD: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(PROJECT_ROOT))


if __name__ == "__main__":
    main()
