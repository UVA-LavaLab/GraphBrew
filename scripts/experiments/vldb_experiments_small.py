#!/usr/bin/env python3
"""
VLDB Experiments — Local Preview Runner.

A lightweight wrapper around vldb_experiments.py that runs a quick subset
of each experiment so you can verify correctness before running the full
suite on a lab machine.

Differences from the full runner:
  - 1 trial (not 3)
  - 5 baseline algorithms: ORIGINAL, SORT, GORDER, GraphBrewOrder, GoGraphOrder
  - 2 benchmarks: pr, bfs (not 4)
  - Medium-tier graphs only, auto-filtered to those already on disk
  - 300 s per-command timeout (not 7200 s)
  - Output goes to results/vldb_preview/ (not vldb_experiments/)
  - Exp 3 (LOGO CV): unchanged (already fast)
  - Exp 4 (Ablation): 3 groups only (no_packing, no_types, all_features)
  - Exp 6 (Cache): simulation only (no HW perf counters)

Usage:
    python scripts/experiments/vldb_experiments_small.py --all --dry-run
    python scripts/experiments/vldb_experiments_small.py --exp 1 7
    python scripts/experiments/vldb_experiments_small.py --all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Constants — tuned for local preview
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PREVIEW_DIR = RESULTS_DIR / "vldb_preview"
BIN_DIR = PROJECT_ROOT / "bench" / "bin"
EXPERIMENT_SCRIPT = PROJECT_ROOT / "scripts" / "graphbrew_experiment.py"

# Reduced algorithm set (5 representative baselines)
ALGORITHMS = {
    0: "ORIGINAL",
    2: "SORT",
    9: "GORDER",
    12: "GraphBrewOrder",
    14: "AdaptiveOrder",
    16: "GoGraphOrder",
}
BASELINE_IDS = [0, 2, 9, 12, 16]  # everything except AdaptiveOrder

BENCHMARKS = ["pr", "bfs"]  # 2 instead of 4
NUM_TRIALS = 1
TIMEOUT = 300  # 5 min per command

# Medium-tier graphs that are quick to process
PREVIEW_GRAPHS = [
    "web-Google",
    "cit-Patents",
    "web-BerkStan",
    "as-Skitter",
    "roadNet-CA",
]

log = logging.getLogger("vldb_preview")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def available_graphs(candidates: Optional[List[str]] = None) -> List[str]:
    """Return the subset of *candidates* whose .sg file exists on disk."""
    targets = candidates or PREVIEW_GRAPHS
    found = []
    for g in targets:
        sg = RESULTS_DIR / "graphs" / g / f"{g}.sg"
        if sg.exists():
            found.append(g)
        else:
            log.info(f"  Graph {g} not on disk, skipping ({sg})")
    if not found:
        log.warning(
            "  No graph .sg files found — run the download step first, "
            "or use --dry-run to check commands."
        )
    return found


def run_cmd(cmd: str, dry_run: bool = False) -> bool:
    """Run a shell command with the preview timeout."""
    log.info(f"  CMD: {cmd}")
    if dry_run:
        return True
    try:
        result = subprocess.run(
            cmd, shell=True, timeout=TIMEOUT, capture_output=False
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        log.error(f"  TIMEOUT after {TIMEOUT}s")
        return False
    except Exception as e:
        log.error(f"  ERROR: {e}")
        return False


def run_experiment_script(args: str, dry_run: bool = False) -> bool:
    """Run graphbrew_experiment.py with given arguments."""
    return run_cmd(f"python {EXPERIMENT_SCRIPT} {args}", dry_run=dry_run)


# ---------------------------------------------------------------------------
# Experiments  (small / preview versions)
# ---------------------------------------------------------------------------


def exp1_oracle_gap(dry_run: bool = False, graphs: Optional[List[str]] = None) -> None:
    """Exp 1: Oracle Gap — subset of algorithms on available graphs."""
    log.info("=" * 60)
    log.info("PREVIEW — Experiment 1: Oracle Gap Analysis")
    log.info("=" * 60)

    out_dir = PREVIEW_DIR / "exp1_oracle_gap"
    os.makedirs(out_dir, exist_ok=True)

    algo_str = " ".join(str(a) for a in BASELINE_IDS)
    bench_str = " ".join(BENCHMARKS)
    target = available_graphs(graphs)

    for g in target:
        run_experiment_script(
            f"--full --precompute -a {algo_str} -b {bench_str} "
            f"-n {NUM_TRIALS} --graphs {g} --output-dir {out_dir}",
            dry_run=dry_run,
        )


def exp2_adaptive_vs_baselines(
    dry_run: bool = False, graphs: Optional[List[str]] = None
) -> None:
    """Exp 2: AdaptiveOrder — mode 1 only (fastest-exec)."""
    log.info("=" * 60)
    log.info("PREVIEW — Experiment 2: AdaptiveOrder vs Baselines")
    log.info("=" * 60)

    out_dir = PREVIEW_DIR / "exp2_adaptive"
    os.makedirs(out_dir, exist_ok=True)

    bench_str = " ".join(BENCHMARKS)
    target = available_graphs(graphs)

    # Only mode 1 for preview
    for g in target:
        run_experiment_script(
            f"--full --precompute -a 14 -b {bench_str} "
            f"-n {NUM_TRIALS} --graphs {g} "
            f"--output-dir {out_dir}/fastest_exec",
            dry_run=dry_run,
        )


def exp3_logo_cv(dry_run: bool = False) -> None:
    """Exp 3: LOGO CV — identical to full (already fast)."""
    log.info("=" * 60)
    log.info("PREVIEW — Experiment 3: Selection Accuracy (LOGO CV)")
    log.info("=" * 60)

    out_dir = PREVIEW_DIR / "exp3_logo_cv"
    os.makedirs(out_dir, exist_ok=True)

    cmd = (
        f"python scripts/evaluate_all_modes.py --all --json "
        f"2>&1 | tee {out_dir}/logo_cv_results.log"
    )
    run_cmd(cmd, dry_run=dry_run)


def exp4_feature_ablation(dry_run: bool = False) -> None:
    """Exp 4: Ablation — 3 groups only."""
    log.info("=" * 60)
    log.info("PREVIEW — Experiment 4: Feature Ablation (subset)")
    log.info("=" * 60)

    out_dir = PREVIEW_DIR / "exp4_feature_ablation"
    os.makedirs(out_dir, exist_ok=True)

    ablation_groups = {
        "no_packing":   "ADAPTIVE_ZERO_FEATURES=packing",
        "no_types":     "ADAPTIVE_NO_TYPES=1",
        "all_features": "",
    }

    bench_str = " ".join(BENCHMARKS)

    for group_name, env_prefix in ablation_groups.items():
        log.info(f"  Ablation: {group_name}")
        cmd = (
            f"{env_prefix} python {EXPERIMENT_SCRIPT} "
            f"--full --precompute --size medium "
            f"-a 14 -b {bench_str} -n {NUM_TRIALS} --auto "
            f"--output-dir {out_dir}/{group_name}"
        )
        run_cmd(cmd, dry_run=dry_run)


def exp5_cold_start(dry_run: bool = False) -> None:
    """Exp 5: Cold-start — medium graphs, 1 trial."""
    log.info("=" * 60)
    log.info("PREVIEW — Experiment 5: Cold-Start Learning Curve")
    log.info("=" * 60)

    out_dir = PREVIEW_DIR / "exp5_cold_start"
    os.makedirs(out_dir, exist_ok=True)

    algo_str = " ".join(str(a) for a in BASELINE_IDS)
    bench_str = " ".join(BENCHMARKS)

    run_experiment_script(
        f"--full --precompute --size medium "
        f"-a {algo_str} 14 -b {bench_str} -n {NUM_TRIALS} --auto "
        f"--output-dir {out_dir}",
        dry_run=dry_run,
    )


def exp6_cache_performance(
    dry_run: bool = False, graphs: Optional[List[str]] = None
) -> None:
    """Exp 6: Cache — simulation only (no HW perf counters)."""
    log.info("=" * 60)
    log.info("PREVIEW — Experiment 6: Cache Simulation (no HW counters)")
    log.info("=" * 60)

    out_dir = PREVIEW_DIR / "exp6_cache"
    os.makedirs(out_dir, exist_ok=True)

    target = available_graphs(graphs)

    for graph in target:
        sg_path = f"results/graphs/{graph}/{graph}.sg"
        for algo_id in [0, 9, 12, 14]:
            algo_name = ALGORITHMS.get(algo_id, str(algo_id))
            cmd = (
                f"./bench/bin_sim/pr -f {sg_path} -s -o {algo_id} -n 1 "
                f"2>&1 | tee {out_dir}/cache_sim_{graph}_{algo_name}.log"
            )
            run_cmd(cmd, dry_run=dry_run)


def exp7_gograph_convergence(
    dry_run: bool = False, graphs: Optional[List[str]] = None
) -> None:
    """Exp 7: GoGraph convergence — parse PR iteration counts."""
    log.info("=" * 60)
    log.info("PREVIEW — Experiment 7: GoGraph Convergence")
    log.info("=" * 60)

    out_dir = PREVIEW_DIR / "exp7_convergence"
    os.makedirs(out_dir, exist_ok=True)

    target = available_graphs(graphs)
    results = []

    for graph in target:
        sg_path = f"results/graphs/{graph}/{graph}.sg"
        for algo_id in [0, 9, 12, 16]:
            algo_name = ALGORITHMS.get(algo_id, str(algo_id))
            log.info(f"  {graph} × {algo_name}")

            cmd = (
                f"./bench/bin/pr -f {sg_path} -s -o {algo_id} -n 1 -l 2>&1"
            )
            if dry_run:
                run_cmd(cmd, dry_run=True)
                continue

            try:
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True,
                    timeout=TIMEOUT,
                )
                iter_match = re.search(r"Iterations\s*:\s*(\d+)", result.stdout)
                time_match = re.search(r"Average Time\s*:\s*([\d.]+)", result.stdout)
                entry = {
                    "graph": graph,
                    "algorithm": algo_name,
                    "algorithm_id": algo_id,
                    "iterations": int(iter_match.group(1)) if iter_match else None,
                    "avg_time": float(time_match.group(1)) if time_match else None,
                }
                results.append(entry)
                log.info(f"    iterations={entry['iterations']}, time={entry['avg_time']}")
            except Exception as e:
                log.error(f"    Failed: {e}")

    if results and not dry_run:
        out_path = out_dir / f"convergence_{timestamp()}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"  Saved {len(results)} entries → {out_path}")


def exp8_scalability(dry_run: bool = False) -> None:
    """Exp 8: Scalability — only graphs already on disk."""
    log.info("=" * 60)
    log.info("PREVIEW — Experiment 8: Scalability (available graphs)")
    log.info("=" * 60)

    out_dir = PREVIEW_DIR / "exp8_scalability"
    os.makedirs(out_dir, exist_ok=True)

    # Check a broad list; use whatever is on disk
    candidates = [
        "ca-AstroPh", "ca-HepPh",
        "web-Google", "cit-Patents", "as-Skitter",
        "soc-LiveJournal1", "com-Orkut",
    ]
    target = available_graphs(candidates)

    for algo_id in [9, 12, 14, 16]:
        algo_name = ALGORITHMS.get(algo_id, str(algo_id))
        for graph in target:
            sg_path = f"results/graphs/{graph}/{graph}.sg"
            log.info(f"  {graph} × {algo_name}")
            cmd = (
                f"./bench/bin/pr -f {sg_path} -s -o {algo_id} -n 1 "
                f"2>&1 | tee {out_dir}/scale_{graph}_{algo_name}.log"
            )
            run_cmd(cmd, dry_run=dry_run)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="VLDB Experiments — Local Preview Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--exp", nargs="+", type=int, choices=range(1, 9),
        help="Experiment number(s) to preview (1-8)",
    )
    parser.add_argument(
        "--all", action="store_true", help="Preview all 8 experiments",
    )
    parser.add_argument(
        "--graphs", nargs="+",
        help="Specific graph names (overrides built-in list)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.exp and not args.all:
        parser.print_help()
        sys.exit(1)

    experiments = list(range(1, 9)) if args.all else args.exp
    os.makedirs(PREVIEW_DIR, exist_ok=True)

    log.info(f"GraphBrew VLDB Preview Runner — {timestamp()}")
    log.info(f"Experiments: {experiments}")
    log.info(f"Trials: {NUM_TRIALS}, Timeout: {TIMEOUT}s, Benchmarks: {BENCHMARKS}")
    if args.dry_run:
        log.info("DRY RUN MODE — no commands will be executed")
    log.info("")

    start = time.time()

    dispatch = {
        1: lambda: exp1_oracle_gap(args.dry_run, args.graphs),
        2: lambda: exp2_adaptive_vs_baselines(args.dry_run, args.graphs),
        3: lambda: exp3_logo_cv(args.dry_run),
        4: lambda: exp4_feature_ablation(args.dry_run),
        5: lambda: exp5_cold_start(args.dry_run),
        6: lambda: exp6_cache_performance(args.dry_run, args.graphs),
        7: lambda: exp7_gograph_convergence(args.dry_run, args.graphs),
        8: lambda: exp8_scalability(args.dry_run),
    }

    for exp_num in experiments:
        dispatch[exp_num]()
        log.info("")

    elapsed = time.time() - start
    log.info(f"All previews complete. Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
