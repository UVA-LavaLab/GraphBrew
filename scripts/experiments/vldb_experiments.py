#!/usr/bin/env python3
"""
VLDB 2026 Experiment Runner for GraphBrew AdaptiveOrder-ML Paper.

Orchestrates the 8 experiments described in the experiment plan:
    1. Oracle Gap Analysis (motivation figure)
    2. AdaptiveOrder vs Static Baselines (main result)
    3. Algorithm Selection Accuracy (LOGO CV)
    4. Feature Importance Ablation
    5. Streaming Database Cold-Start → Warm
    6. Cache Performance (simulation + HW counters)
    7. GoGraph Convergence Analysis
    8. Scalability (overhead vs graph size)

Usage:
    # Run all experiments on LARGE graphs:
    python scripts/experiments/vldb_experiments.py --all --size large

    # Run specific experiment(s):
    python scripts/experiments/vldb_experiments.py --exp 1 2 --size large

    # Dry run (print commands without executing):
    python scripts/experiments/vldb_experiments.py --all --dry-run

    # Custom graph set:
    python scripts/experiments/vldb_experiments.py --exp 2 --graphs web-Google cit-Patents
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
EXPERIMENT_DIR = RESULTS_DIR / "vldb_experiments"
BIN_DIR = PROJECT_ROOT / "bench" / "bin"
EXPERIMENT_SCRIPT = PROJECT_ROOT / "scripts" / "graphbrew_experiment.py"

# All reorder algorithm IDs and their names
ALGORITHMS = {
    0: "ORIGINAL",
    2: "SORT",
    3: "HUBSORT",
    5: "DBG",
    7: "HUBCLUSTERDBG",
    8: "RABBITORDER",
    9: "GORDER",
    10: "CORDER",
    11: "RCM",
    12: "GraphBrewOrder",
    14: "AdaptiveOrder",
    15: "LeidenOrder",
    16: "GoGraphOrder",
}

# Baseline set (all except AdaptiveOrder)
BASELINE_IDS = [0, 2, 3, 5, 7, 8, 9, 10, 11, 12, 15, 16]

# Benchmarks for evaluation
BENCHMARKS = ["pr", "bfs", "sssp", "cc"]

# Representative graphs for focused experiments
REPRESENTATIVE_GRAPHS = [
    "web-Google",
    "cit-Patents",
    "soc-LiveJournal1",
    "com-Orkut",
    "twitter7",
]

log = logging.getLogger("vldb_experiments")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_cmd(cmd: str, dry_run: bool = False, timeout: int = 7200) -> bool:
    """Run a shell command, return success."""
    log.info(f"  CMD: {cmd}")
    if dry_run:
        return True
    try:
        result = subprocess.run(
            cmd, shell=True, timeout=timeout, capture_output=False
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        log.error(f"  TIMEOUT after {timeout}s")
        return False
    except Exception as e:
        log.error(f"  ERROR: {e}")
        return False


def run_experiment_script(
    args: str, dry_run: bool = False, timeout: int = 14400
) -> bool:
    """Run graphbrew_experiment.py with given arguments."""
    cmd = f"python {EXPERIMENT_SCRIPT} {args}"
    return run_cmd(cmd, dry_run=dry_run, timeout=timeout)


# ---------------------------------------------------------------------------
# Experiment 1: Oracle Gap Analysis
# ---------------------------------------------------------------------------


def exp1_oracle_gap(
    size: str = "large",
    num_trials: int = 3,
    dry_run: bool = False,
    graphs: Optional[List[str]] = None,
) -> None:
    """Run ALL candidate algorithms on all graphs to find the oracle.

    This produces the data for the motivation figure showing that no single
    algorithm dominates across all graphs.
    """
    log.info("=" * 60)
    log.info("EXPERIMENT 1: Oracle Gap Analysis")
    log.info("=" * 60)

    out_dir = EXPERIMENT_DIR / "exp1_oracle_gap"
    os.makedirs(out_dir, exist_ok=True)

    # Run brute-force: all algorithms × all benchmarks × all graphs
    algo_str = " ".join(str(a) for a in BASELINE_IDS)
    bench_str = " ".join(BENCHMARKS)

    if graphs:
        for g in graphs:
            run_experiment_script(
                f"--full --precompute -a {algo_str} -b {bench_str} "
                f"-n {num_trials} --graphs {g} "
                f"--output-dir {out_dir}",
                dry_run=dry_run,
            )
    else:
        run_experiment_script(
            f"--full --precompute --size {size} "
            f"-a {algo_str} -b {bench_str} -n {num_trials} --auto "
            f"--output-dir {out_dir}",
            dry_run=dry_run,
        )


# ---------------------------------------------------------------------------
# Experiment 2: AdaptiveOrder vs Static Baselines
# ---------------------------------------------------------------------------


def exp2_adaptive_vs_baselines(
    size: str = "large",
    num_trials: int = 3,
    dry_run: bool = False,
    graphs: Optional[List[str]] = None,
) -> None:
    """Run AdaptiveOrder on the same graph set and compare to baselines."""
    log.info("=" * 60)
    log.info("EXPERIMENT 2: AdaptiveOrder vs Static Baselines")
    log.info("=" * 60)

    out_dir = EXPERIMENT_DIR / "exp2_adaptive"
    os.makedirs(out_dir, exist_ok=True)

    bench_str = " ".join(BENCHMARKS)

    # Run AdaptiveOrder in all selection modes
    for mode in [1, 2, 3]:  # fastest-exec, best-e2e, best-amortization
        mode_name = {1: "fastest_exec", 2: "best_e2e", 3: "best_amort"}[mode]
        log.info(f"  Mode {mode}: {mode_name}")

        if graphs:
            for g in graphs:
                run_experiment_script(
                    f"--full --precompute -a 14 -b {bench_str} "
                    f"-n {num_trials} --graphs {g} "
                    f"--output-dir {out_dir}/{mode_name}",
                    dry_run=dry_run,
                )
        else:
            run_experiment_script(
                f"--full --precompute --size {size} "
                f"-a 14 -b {bench_str} -n {num_trials} --auto "
                f"--output-dir {out_dir}/{mode_name}",
                dry_run=dry_run,
            )


# ---------------------------------------------------------------------------
# Experiment 3: Algorithm Selection Accuracy (LOGO CV)
# ---------------------------------------------------------------------------


def exp3_logo_cv(dry_run: bool = False) -> None:
    """Run LOGO cross-validation using the existing evaluation infrastructure."""
    log.info("=" * 60)
    log.info("EXPERIMENT 3: Selection Accuracy (LOGO CV)")
    log.info("=" * 60)

    out_dir = EXPERIMENT_DIR / "exp3_logo_cv"
    os.makedirs(out_dir, exist_ok=True)

    # evaluate_all_modes.py accepts: --logo, --all, --weights, --json
    # Run with --all --json and capture output to our experiment directory
    cmd = (
        f"python scripts/evaluate_all_modes.py --all --json "
        f"2>&1 | tee {out_dir}/logo_cv_results.log"
    )
    run_cmd(cmd, dry_run=dry_run)


# ---------------------------------------------------------------------------
# Experiment 4: Feature Importance Ablation
# ---------------------------------------------------------------------------


def exp4_feature_ablation(
    size: str = "medium",
    dry_run: bool = False,
) -> None:
    """Run feature ablation study: remove one feature group at a time."""
    log.info("=" * 60)
    log.info("EXPERIMENT 4: Feature Importance Ablation")
    log.info("=" * 60)

    out_dir = EXPERIMENT_DIR / "exp4_feature_ablation"
    os.makedirs(out_dir, exist_ok=True)

    # C++ AblationConfig reads ADAPTIVE_ZERO_FEATURES (comma-delimited)
    # and ADAPTIVE_NO_* boolean toggles — NOT individual weight env vars.
    # Valid feature group names: packing, fef, wsr, quadratic
    ablation_groups = {
        "no_packing":    "ADAPTIVE_ZERO_FEATURES=packing",
        "no_fef":        "ADAPTIVE_ZERO_FEATURES=fef",
        "no_wsr":        "ADAPTIVE_ZERO_FEATURES=wsr",
        "no_quadratic":  "ADAPTIVE_ZERO_FEATURES=quadratic",
        "no_packing_fef_wsr_quad": "ADAPTIVE_ZERO_FEATURES=packing,fef,wsr,quadratic",
        "no_types":      "ADAPTIVE_NO_TYPES=1",
        "no_ood":        "ADAPTIVE_NO_OOD=1",
        "no_margin":     "ADAPTIVE_NO_MARGIN=1",
        "no_leiden":     "ADAPTIVE_NO_LEIDEN=1",
        "all_features":  "",  # baseline: all features enabled
    }

    bench_str = " ".join(BENCHMARKS)

    for group_name, env_prefix in ablation_groups.items():
        log.info(f"  Ablation: {group_name}")
        cmd = (
            f"{env_prefix} python {EXPERIMENT_SCRIPT} "
            f"--full --precompute --size {size} "
            f"-a 14 -b {bench_str} -n 3 --auto "
            f"--output-dir {out_dir}/{group_name}"
        )
        run_cmd(cmd, dry_run=dry_run)


# ---------------------------------------------------------------------------
# Experiment 5: Streaming Database Cold-Start → Warm
# ---------------------------------------------------------------------------


def exp5_cold_start(
    size: str = "medium",
    dry_run: bool = False,
) -> None:
    """Simulate cold-start: process graphs one at a time, measure accuracy."""
    log.info("=" * 60)
    log.info("EXPERIMENT 5: Cold-Start → Warm Learning Curve")
    log.info("=" * 60)

    out_dir = EXPERIMENT_DIR / "exp5_cold_start"
    os.makedirs(out_dir, exist_ok=True)

    # This experiment needs a custom driver that:
    # 1. Starts with empty streaming DB
    # 2. Processes graphs sequentially
    # 3. After each graph, evaluates selection accuracy on remaining
    #
    # For now, generate the brute-force data needed for the simulation
    algo_str = " ".join(str(a) for a in BASELINE_IDS)
    bench_str = " ".join(BENCHMARKS)

    run_experiment_script(
        f"--full --precompute --size {size} "
        f"-a {algo_str} 14 -b {bench_str} -n 3 --auto "
        f"--output-dir {out_dir}",
        dry_run=dry_run,
    )

    log.info(
        "  NOTE: Run scripts/lib/analysis/cold_start_sim.py on the collected "
        "data to generate the learning curve plot."
    )


# ---------------------------------------------------------------------------
# Experiment 6: Cache Performance (simulation + HW counters)
# ---------------------------------------------------------------------------


def exp6_cache_performance(
    dry_run: bool = False,
    graphs: Optional[List[str]] = None,
) -> None:
    """Run cache simulation sweep + hardware perf counters."""
    log.info("=" * 60)
    log.info("EXPERIMENT 6: Cache Performance")
    log.info("=" * 60)

    out_dir = EXPERIMENT_DIR / "exp6_cache"
    os.makedirs(out_dir, exist_ok=True)

    target_graphs = graphs or REPRESENTATIVE_GRAPHS

    # Part A: Cache simulation (using bin_sim/ binaries)
    log.info("  Part A: Cache simulation")
    for graph in target_graphs:
        sg_path = f"results/graphs/{graph}/{graph}.sg"
        if not Path(sg_path).exists():
            log.warning(f"  Skipping {graph}: {sg_path} not found")
            continue
        for algo_id in [0, 9, 12, 14]:
            algo_name = ALGORITHMS[algo_id]
            cmd = (
                f"./bench/bin_sim/pr -f {sg_path} -s -o {algo_id} -n 1 "
                f"2>&1 | tee {out_dir}/cache_sim_{graph}_{algo_name}.log"
            )
            run_cmd(cmd, dry_run=dry_run, timeout=600)

    # Part B: Hardware perf counters (using regular binaries)
    log.info("  Part B: Hardware perf counters")
    perf_algos = {
        "ORIGINAL": "-o 0",
        "Gorder": "-o 9",
        "GraphBrew_leiden": "-o 12",
        "AdaptiveOrder": "-o 14",
        "GoGraph": "-o 16",
    }
    for graph in target_graphs:
        sg_path = f"results/graphs/{graph}/{graph}.sg"
        if not Path(sg_path).exists():
            continue
        cmd = (
            f"python -m scripts.lib.analysis.perf_counters "
            f"-f {sg_path} -b pr bfs -n 3 "
            f"--algorithms "
            + " ".join(f'"{k}:{v}"' for k, v in perf_algos.items())
            + f" -o {out_dir}/perf_{graph}.json"
        )
        run_cmd(cmd, dry_run=dry_run, timeout=600)


# ---------------------------------------------------------------------------
# Experiment 7: GoGraph Convergence Analysis
# ---------------------------------------------------------------------------


def exp7_gograph_convergence(
    dry_run: bool = False,
    graphs: Optional[List[str]] = None,
) -> None:
    """Measure PR iteration counts with different orderings.

    This experiment demonstrates that GoGraph's forward-edge maximization
    reduces the number of PageRank iterations needed for convergence.
    """
    log.info("=" * 60)
    log.info("EXPERIMENT 7: GoGraph Convergence Analysis")
    log.info("=" * 60)

    out_dir = EXPERIMENT_DIR / "exp7_convergence"
    os.makedirs(out_dir, exist_ok=True)

    target_graphs = graphs or REPRESENTATIVE_GRAPHS
    results = []

    for graph in target_graphs:
        sg_path = f"results/graphs/{graph}/{graph}.sg"
        if not Path(sg_path).exists():
            log.warning(f"  Skipping {graph}: {sg_path} not found")
            continue

        for algo_id in [0, 9, 12, 16]:  # ORIGINAL, Gorder, GraphBrew, GoGraph
            algo_name = ALGORITHMS[algo_id]
            log.info(f"  {graph} × {algo_name}")

            # Run PR with logging enabled to capture iteration count
            cmd = (
                f"./bench/bin/pr -f {sg_path} -s -o {algo_id} "
                f"-n 1 -l 2>&1"
            )
            if dry_run:
                run_cmd(cmd, dry_run=True)
                continue

            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                stdout = result.stdout

                # Parse iteration count from output
                import re

                iter_match = re.search(r"Iterations\s*:\s*(\d+)", stdout)
                time_match = re.search(
                    r"Average Time\s*:\s*([\d.]+)", stdout
                )

                entry = {
                    "graph": graph,
                    "algorithm": algo_name,
                    "algorithm_id": algo_id,
                    "iterations": (
                        int(iter_match.group(1)) if iter_match else None
                    ),
                    "avg_time": (
                        float(time_match.group(1)) if time_match else None
                    ),
                }
                results.append(entry)
                log.info(
                    f"    iterations={entry['iterations']}, "
                    f"time={entry['avg_time']}"
                )
            except Exception as e:
                log.error(f"    Failed: {e}")

    if results and not dry_run:
        out_path = out_dir / f"convergence_{timestamp()}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"  Saved {len(results)} entries → {out_path}")


# ---------------------------------------------------------------------------
# Experiment 8: Scalability (overhead vs graph size)
# ---------------------------------------------------------------------------


def exp8_scalability(
    dry_run: bool = False,
) -> None:
    """Measure reorder time across graph sizes (10K → 1B+ edges).

    Shows near-linear scaling for AdaptiveOrder vs superlinear for Gorder.
    """
    log.info("=" * 60)
    log.info("EXPERIMENT 8: Scalability Analysis")
    log.info("=" * 60)

    out_dir = EXPERIMENT_DIR / "exp8_scalability"
    os.makedirs(out_dir, exist_ok=True)

    # Graphs spanning multiple orders of magnitude
    scalability_graphs = [
        # Small (~10K-100K edges)
        "ca-AstroPh",
        "ca-HepPh",
        # Medium (~100K-10M edges)
        "web-Google",
        "cit-Patents",
        "as-Skitter",
        # Large (~10M-100M edges)
        "soc-LiveJournal1",
        "com-Orkut",
        "hollywood-2009",
        # XLarge (~100M-1B+ edges)
        "uk-2002",
        "arabic-2005",
        "twitter7",
        "sk-2005",
    ]

    # Only reorder time matters here — run with -n 1 and parse reorder time
    for algo_id in [9, 12, 14, 16]:  # Gorder, GraphBrew, Adaptive, GoGraph
        algo_name = ALGORITHMS[algo_id]
        for graph in scalability_graphs:
            sg_path = f"results/graphs/{graph}/{graph}.sg"
            if not Path(sg_path).exists():
                log.info(f"  Skipping {graph}: not downloaded yet")
                continue

            log.info(f"  {graph} × {algo_name}")
            cmd = (
                f"./bench/bin/pr -f {sg_path} -s -o {algo_id} -n 1 "
                f"2>&1 | tee {out_dir}/scale_{graph}_{algo_name}.log"
            )
            run_cmd(cmd, dry_run=dry_run, timeout=3600)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="VLDB 2026 Experiment Runner for GraphBrew",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--exp",
        nargs="+",
        type=int,
        choices=range(1, 9),
        help="Experiment number(s) to run (1-8)",
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all 8 experiments"
    )
    parser.add_argument(
        "--size",
        choices=["small", "medium", "large", "xlarge"],
        default="large",
        help="Graph size tier (default: large)",
    )
    parser.add_argument(
        "--graphs",
        nargs="+",
        help="Specific graph names to use (overrides --size)",
    )
    parser.add_argument(
        "-n",
        "--trials",
        type=int,
        default=3,
        help="Number of benchmark trials (default: 3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose logging"
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

    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    log.info(f"GraphBrew VLDB Experiment Runner — {timestamp()}")
    log.info(f"Experiments: {experiments}")
    log.info(f"Size: {args.size}, Trials: {args.trials}")
    if args.dry_run:
        log.info("DRY RUN MODE — no commands will be executed")
    log.info("")

    start_time = time.time()

    dispatch = {
        1: lambda: exp1_oracle_gap(
            args.size, args.trials, args.dry_run, args.graphs
        ),
        2: lambda: exp2_adaptive_vs_baselines(
            args.size, args.trials, args.dry_run, args.graphs
        ),
        3: lambda: exp3_logo_cv(args.dry_run),
        4: lambda: exp4_feature_ablation(args.size, args.dry_run),
        5: lambda: exp5_cold_start(args.size, args.dry_run),
        6: lambda: exp6_cache_performance(args.dry_run, args.graphs),
        7: lambda: exp7_gograph_convergence(args.dry_run, args.graphs),
        8: lambda: exp8_scalability(args.dry_run),
    }

    for exp_num in experiments:
        dispatch[exp_num]()
        log.info("")

    elapsed = time.time() - start_time
    log.info(f"All experiments complete. Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
