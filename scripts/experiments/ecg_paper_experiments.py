#!/usr/bin/env python3
"""
ECG Paper Experiments Runner
=============================
Runs all experiments for the ECG paper:
  "Expressing Locality and Prefetching for Optimal Caching in Graph Structures"

6 experiments:
  1. Policy Comparison    — All 9 policies × benchmarks × graphs
  2. Reorder Interaction  — DBG/Rabbit/GraphBrew × GRASP/P-OPT/ECG
  3. Cache Size Sweep     — L3 32KB–64MB with key policies
  4. Algorithm Analysis   — Group by access pattern (iterative/traversal)
  5. Graph Sensitivity    — Group by topology (social/road/citation)
  6. Fat-ID Analysis      — Bit allocation per graph size (analytical)

Usage:
  python3 scripts/experiments/ecg_paper_experiments.py --all --graph-dir /path/to/graphs
  python3 scripts/experiments/ecg_paper_experiments.py --exp 1 --preview
  python3 scripts/experiments/ecg_paper_experiments.py --exp 6
  python3 scripts/experiments/ecg_paper_experiments.py --all --dry-run
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.ecg_config import (
    BIN_SIM_DIR, RESULTS_DIR,
    ALL_POLICIES, PREVIEW_POLICIES, GRAPH_AWARE_POLICIES,
    BASELINE_POLICIES, REORDER_POLICY_PAIRS,
    BENCHMARKS, BENCHMARKS_PREVIEW,
    ITERATIVE_BENCHMARKS, TRAVERSAL_BENCHMARKS,
    DEFAULT_CACHE, CACHE_SIZES_SWEEP,
    EVAL_GRAPHS, EVAL_GRAPHS_PREVIEW,
    TIMEOUT_SIM, TIMEOUT_SIM_HEAVY, TRIALS,
    policy_env, format_cache_size,
)


# ============================================================================
# Output Parsing
# ============================================================================

def parse_cache_output(output):
    """Parse cache simulation stdout for L1/L2/L3 hit/miss stats."""
    result = {}
    for level in ["L1", "L2", "L3"]:
        hits = re.search(rf"{level}.*?Hits:\s+(\d+)", output, re.DOTALL)
        misses = re.search(rf"{level}.*?Misses:\s+(\d+)", output, re.DOTALL)
        if hits and misses:
            h, m = int(hits.group(1)), int(misses.group(1))
            total = h + m
            result[f"{level.lower()}_hits"] = h
            result[f"{level.lower()}_misses"] = m
            result[f"{level.lower()}_miss_rate"] = round(m / total, 6) if total > 0 else 0.0
            result[f"{level.lower()}_hit_rate"] = round(h / total, 6) if total > 0 else 0.0

    # Parse Graph Cache Context summary if present
    hot_match = re.search(r"hot=([\d.]+)%", output)
    if hot_match:
        result["hot_fraction_pct"] = float(hot_match.group(1))

    # Parse timing
    time_match = re.search(r"Average:\s+([\d.]+)", output)
    if time_match:
        result["avg_time_s"] = float(time_match.group(1))

    return result


def run_sim(benchmark, graph_path, reorder_opt, policy,
            cache_config=None, timeout=TIMEOUT_SIM, dry_run=False):
    """Run a single cache simulation and return parsed results."""
    binary = BIN_SIM_DIR / benchmark
    if not binary.exists() and not dry_run:
        return {"error": f"Binary not found: {binary}"}

    cmd = [str(binary), "-f", graph_path, "-s"]
    cmd += reorder_opt.split()
    cmd += ["-n", str(TRIALS)]
    env = policy_env(policy, cache_config)

    if dry_run:
        env_str = f"CACHE_POLICY={policy}"
        if cache_config and "CACHE_L3_SIZE" in cache_config:
            env_str += f" CACHE_L3_SIZE={cache_config['CACHE_L3_SIZE']}"
        print(f"  [DRY] {env_str} {' '.join(cmd)}")
        return {"dry_run": True}

    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout, env=env)
        if result.returncode != 0:
            return {"error": result.stderr[:200] if result.stderr else "nonzero exit"}
        parsed = parse_cache_output(result.stdout)
        if not parsed:
            return {"error": "no cache stats in output"}
        return parsed
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)[:100]}


# ============================================================================
# Experiment 1: Policy Comparison
# ============================================================================

def exp1_policy_comparison(graphs, benchmarks, policies, dry_run, graph_dir):
    """Compare all cache policies across benchmarks and graphs.

    All runs use DBG reordering (-o 5) so GRASP/ECG regions are meaningful.
    For comparison, also runs Original (-o 0) with LRU as absolute baseline.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: Cache Policy Comparison")
    print(f"  {len(graphs)} graphs × {len(benchmarks)} benchmarks × {len(policies)} policies")
    print("=" * 70)

    results = []
    total = len(graphs) * len(benchmarks) * (len(policies) + 1)  # +1 for Original+LRU baseline
    done = 0

    for g in graphs:
        gpath = str(Path(graph_dir) / g["name"] / "graph.sg")

        for bench in benchmarks:
            # Baseline: Original ordering + LRU
            done += 1
            print(f"  [{done}/{total}] {g['short']}/{bench}/Original+LRU", end="", flush=True)
            timeout = TIMEOUT_SIM_HEAVY if bench in ("bc", "sssp") else TIMEOUT_SIM
            r = run_sim(bench, gpath, "-o 0", "LRU", timeout=timeout, dry_run=dry_run)
            r.update({"graph": g["short"], "graph_type": g["type"],
                       "benchmark": bench, "policy": "Original+LRU", "reorder": "-o 0"})
            results.append(r)
            _print_result(r)

            # All policies with DBG reordering
            for policy in policies:
                done += 1
                print(f"  [{done}/{total}] {g['short']}/{bench}/DBG+{policy}", end="", flush=True)
                r = run_sim(bench, gpath, "-o 5", policy, timeout=timeout, dry_run=dry_run)
                r.update({"graph": g["short"], "graph_type": g["type"],
                           "benchmark": bench, "policy": f"DBG+{policy}", "reorder": "-o 5"})
                results.append(r)
                _print_result(r)

    return results


# ============================================================================
# Experiment 2: Reorder × Policy Interaction
# ============================================================================

def exp2_reorder_interaction(graphs, benchmarks, dry_run, graph_dir):
    """Show how different reorderings interact with graph-aware policies."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Reordering × Policy Interaction")
    print(f"  {len(graphs)} graphs × {len(benchmarks)} benchmarks × {len(REORDER_POLICY_PAIRS)} pairs")
    print("=" * 70)

    results = []
    total = len(graphs) * len(benchmarks) * len(REORDER_POLICY_PAIRS)
    done = 0

    for g in graphs:
        gpath = str(Path(graph_dir) / g["name"] / "graph.sg")
        for bench in benchmarks:
            for reorder_opt, policy, label in REORDER_POLICY_PAIRS:
                done += 1
                print(f"  [{done}/{total}] {g['short']}/{bench}/{label}", end="", flush=True)
                timeout = TIMEOUT_SIM_HEAVY if bench in ("bc", "sssp") else TIMEOUT_SIM
                r = run_sim(bench, gpath, reorder_opt, policy, timeout=timeout, dry_run=dry_run)
                r.update({"graph": g["short"], "graph_type": g["type"],
                           "benchmark": bench, "reorder": reorder_opt,
                           "policy": policy, "label": label})
                results.append(r)
                _print_result(r)

    return results


# ============================================================================
# Experiment 3: Cache Size Sweep
# ============================================================================

def exp3_cache_sweep(graphs, benchmarks, policies, dry_run, graph_dir):
    """Sweep L3 cache size from 32KB to 64MB with key policies."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: Cache Size Sensitivity")
    print(f"  {len(graphs)} graphs × {len(benchmarks)} benchmarks × "
          f"{len(policies)} policies × {len(CACHE_SIZES_SWEEP)} sizes")
    print("=" * 70)

    results = []
    total = len(graphs) * len(benchmarks) * len(policies) * len(CACHE_SIZES_SWEEP)
    done = 0

    for g in graphs:
        gpath = str(Path(graph_dir) / g["name"] / "graph.sg")
        for bench in benchmarks:
            for policy in policies:
                for cache_size in CACHE_SIZES_SWEEP:
                    done += 1
                    sz_str = format_cache_size(cache_size)
                    print(f"  [{done}/{total}] {g['short']}/{bench}/{policy}/{sz_str}",
                          end="", flush=True)
                    config = dict(DEFAULT_CACHE)
                    config["CACHE_L3_SIZE"] = str(cache_size)
                    r = run_sim(bench, gpath, "-o 5", policy,
                                cache_config=config, dry_run=dry_run)
                    r.update({"graph": g["short"], "benchmark": bench,
                              "policy": policy, "cache_size": cache_size,
                              "cache_size_str": sz_str})
                    results.append(r)
                    _print_result(r)

    return results


# ============================================================================
# Experiment 4: Algorithm-Type Analysis (derived from Exp1)
# ============================================================================

def exp4_algorithm_analysis(exp1_results):
    """Group Exp1 results by algorithm access pattern."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 4: Algorithm-Type Analysis (derived from Exp1)")
    print("=" * 70)

    analysis = {}
    for category, benches in [("Iterative", ITERATIVE_BENCHMARKS),
                               ("Traversal", TRAVERSAL_BENCHMARKS)]:
        analysis[category] = {}
        for r in exp1_results:
            if r.get("benchmark") in benches and "l3_miss_rate" in r:
                policy = r["policy"]
                analysis[category].setdefault(policy, []).append(r["l3_miss_rate"])

        print(f"\n  {category} algorithms (geo-mean L3 miss rate):")
        for policy in sorted(analysis[category].keys()):
            rates = analysis[category][policy]
            if rates:
                geo_mean = _geo_mean(rates)
                print(f"    {policy:20s}: {geo_mean:.4f} ({len(rates)} samples)")

    return analysis


# ============================================================================
# Experiment 5: Graph-Type Sensitivity (derived from Exp1)
# ============================================================================

def exp5_graph_sensitivity(exp1_results):
    """Group Exp1 results by graph topology type."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 5: Graph-Type Sensitivity (derived from Exp1)")
    print("=" * 70)

    analysis = {}
    for r in exp1_results:
        if "l3_miss_rate" not in r:
            continue
        gtype = r.get("graph_type", "Unknown")
        policy = r["policy"]
        analysis.setdefault(gtype, {}).setdefault(policy, []).append(r["l3_miss_rate"])

    graph_types = sorted(analysis.keys())
    for gtype in graph_types:
        print(f"\n  {gtype} graphs (geo-mean L3 miss rate):")
        for policy in sorted(analysis[gtype].keys()):
            rates = analysis[gtype][policy]
            if rates:
                geo_mean = _geo_mean(rates)
                print(f"    {policy:20s}: {geo_mean:.4f} ({len(rates)} samples)")

    return analysis


# ============================================================================
# Experiment 6: Fat-ID Bit Allocation Analysis (analytical, no simulation)
# ============================================================================

def exp6_fatid_analysis(graphs):
    """Show adaptive fat-ID bit allocation for each graph size."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 6: Fat-ID Bit Allocation Analysis")
    print("=" * 70)
    print(f"  {'Graph':12s} | {'|V|':>10s} | {'ID':>3s} | "
          f"{'--- 32-bit ---':^26s} | {'--- 64-bit ---':^26s} | vs P-OPT")
    print(f"  {'':12s} | {'':>10s} | {'':>3s} | "
          f"{'spare':>5s} {'DBG':>3s} {'POPT':>4s} {'PFX':>3s} {'levels':>6s} | "
          f"{'spare':>5s} {'DBG':>3s} {'POPT':>4s} {'PFX':>3s} {'levels':>6s} |")
    print("  " + "-" * 95)

    results = []
    for g in graphs:
        v = int(g.get("vertices_m", 1) * 1_000_000)
        id_bits = max(1, math.ceil(math.log2(max(v, 2))))

        # 32-bit allocation
        spare_32 = 32 - id_bits
        d32, p32, f32 = _allocate_bits(spare_32)
        levels_32 = 2 ** p32 if p32 > 0 else 0

        # 64-bit allocation
        spare_64 = 64 - id_bits
        d64, p64, f64 = _allocate_bits(spare_64)
        levels_64 = 2 ** p64 if p64 > 0 else 0

        vs_popt = f"{levels_64 / 128 * 100:.0f}%" if levels_64 else "N/A"

        entry = {
            "graph": g["short"], "vertices": v, "id_bits": id_bits,
            "spare_32": spare_32, "dbg_32": d32, "popt_32": p32, "pfx_32": f32,
            "levels_32": levels_32,
            "spare_64": spare_64, "dbg_64": d64, "popt_64": p64, "pfx_64": f64,
            "levels_64": levels_64, "vs_popt_matrix": vs_popt,
        }
        results.append(entry)

        print(f"  {g['short']:12s} | {v:>10,} | {id_bits:>3d} | "
              f"{spare_32:>5d} {d32:>3d} {p32:>4d} {f32:>3d} {levels_32:>6d} | "
              f"{spare_64:>5d} {d64:>3d} {p64:>4d} {f64:>3d} {levels_64:>6d} | "
              f"{vs_popt:>6s}")

    print("\n  P-OPT matrix uses 7-bit precision (128 levels), consuming 2+ LLC ways.")
    print("  Fat-ID encoding uses 0 LLC capacity. 64-bit mode exceeds P-OPT precision.")

    return results


# ============================================================================
# Helpers
# ============================================================================

def _print_result(r):
    """Print a compact result line."""
    if "l3_miss_rate" in r:
        print(f"  L3_miss={r['l3_miss_rate']:.4f}")
    elif "error" in r:
        print(f"  ERROR: {r['error'][:50]}")
    elif "dry_run" in r:
        pass  # Already printed by run_sim
    else:
        print()


def _geo_mean(values):
    """Geometric mean of positive values."""
    if not values:
        return 0.0
    product = 1.0
    for v in values:
        product *= max(v, 1e-10)  # Avoid log(0)
    return product ** (1.0 / len(values))


def _allocate_bits(spare):
    """Allocate metadata bits from spare bits (matching FatIDConfig logic)."""
    if spare >= 16:
        return 2, 8, min(spare - 10, 6)
    elif spare >= 10:
        return 2, 4, min(spare - 6, 4)
    elif spare >= 6:
        return 2, 2, spare - 4
    elif spare >= 4:
        return 2, 2, 0
    elif spare >= 2:
        return 2, 0, 0
    else:
        return spare, 0, 0


def save_results(results, name):
    """Save results to JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  → Saved to: {path}")
    return path


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ECG Paper Experiments Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 %(prog)s --all --graph-dir /data/graphs
  python3 %(prog)s --exp 1 6 --preview
  python3 %(prog)s --exp 6                         # Analytical — no graphs needed
  python3 %(prog)s --all --dry-run
        """)
    parser.add_argument("--all", action="store_true", help="Run all 6 experiments")
    parser.add_argument("--exp", nargs="+", type=int, choices=range(1, 7),
                        help="Run specific experiments (1-6)")
    parser.add_argument("--preview", action="store_true",
                        help="Use smaller graph/benchmark/policy set")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--graph-dir", default=".", help="Base directory for graphs")

    args = parser.parse_args()

    if not args.all and not args.exp:
        parser.print_help()
        return

    experiments = list(range(1, 7)) if args.all else args.exp
    graphs = EVAL_GRAPHS_PREVIEW if args.preview else EVAL_GRAPHS
    benchmarks = BENCHMARKS_PREVIEW if args.preview else BENCHMARKS
    policies = PREVIEW_POLICIES if args.preview else ALL_POLICIES

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'=' * 70}")
    print(f"  ECG Paper Experiments — {ts}")
    print(f"  Graphs: {len(graphs)} | Benchmarks: {len(benchmarks)} | Policies: {len(policies)}")
    print(f"  Experiments: {experiments}")
    print(f"{'=' * 70}")

    exp1_results = None

    for exp_num in sorted(experiments):
        t0 = time.time()

        if exp_num == 1:
            exp1_results = exp1_policy_comparison(graphs, benchmarks, policies,
                                                   args.dry_run, args.graph_dir)
            if not args.dry_run:
                save_results(exp1_results, f"exp1_policy_comparison_{ts}")

        elif exp_num == 2:
            r = exp2_reorder_interaction(graphs, benchmarks, args.dry_run, args.graph_dir)
            if not args.dry_run:
                save_results(r, f"exp2_reorder_interaction_{ts}")

        elif exp_num == 3:
            sweep_policies = PREVIEW_POLICIES if args.preview else \
                ["LRU", "SRRIP", "GRASP", "POPT", "ECG"]
            sweep_benches = ["pr"] if args.preview else ["pr", "bfs"]
            r = exp3_cache_sweep(graphs, sweep_benches, sweep_policies,
                                args.dry_run, args.graph_dir)
            if not args.dry_run:
                save_results(r, f"exp3_cache_sweep_{ts}")

        elif exp_num == 4:
            if exp1_results is None:
                print("\n  Exp4 requires Exp1. Running Exp1 first...")
                exp1_results = exp1_policy_comparison(
                    graphs, benchmarks, policies, args.dry_run, args.graph_dir)
            r = exp4_algorithm_analysis(exp1_results)
            if not args.dry_run:
                save_results(r, f"exp4_algorithm_analysis_{ts}")

        elif exp_num == 5:
            if exp1_results is None:
                print("\n  Exp5 requires Exp1. Running Exp1 first...")
                exp1_results = exp1_policy_comparison(
                    graphs, benchmarks, policies, args.dry_run, args.graph_dir)
            r = exp5_graph_sensitivity(exp1_results)
            if not args.dry_run:
                save_results(r, f"exp5_graph_sensitivity_{ts}")

        elif exp_num == 6:
            r = exp6_fatid_analysis(graphs)
            save_results(r, f"exp6_fatid_analysis_{ts}")

        elapsed = time.time() - t0
        print(f"\n  Experiment {exp_num} completed in {elapsed:.1f}s")

    print(f"\n{'=' * 70}")
    print(f"  All experiments complete. Results in: {RESULTS_DIR}/")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
