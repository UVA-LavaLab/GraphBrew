#!/usr/bin/env python3
"""
ECG Paper Experiments Runner
=============================
Runs all experiments for the ECG paper:
  "Expressing Locality and Prefetching for Optimal Caching in Graph Structures"

Organized into two sections:

  Section A: Accuracy Validation (faithfulness to reference papers)
    A1. GRASP Invariants    -- verify 3 claims from Faldu et al., HPCA 2020
    A2. P-OPT Invariants   -- verify 3 claims from Balaji et al., HPCA 2021
    A3. ECG Mode Equiv.     -- ECG(DBG_ONLY)~GRASP, ECG(POPT_PRIMARY)~P-OPT

  Section B: Performance Showcase (comparison and reorder effects)
    B1. Policy Comparison   -- All 9 policies x benchmarks x graphs
    B2. Reorder Effect      -- How reordering affects each policy
    B3. Reorder x Policy    -- Full interaction matrix
    B4. Cache Size Sweep    -- L3 32KB-64MB sensitivity
    B5. Algorithm Analysis  -- Iterative vs traversal access patterns
    B6. Graph Sensitivity   -- Social/road/citation topology effects
    B7. ECG Mode Comparison -- DBG_PRIMARY vs POPT_PRIMARY vs DBG_ONLY
    B8. Fat-ID Analysis     -- Bit allocation per graph size (analytical)

Usage:
  python3 scripts/experiments/ecg_paper_experiments.py --all --graph-dir /path/to/graphs
  python3 scripts/experiments/ecg_paper_experiments.py --section A --preview
  python3 scripts/experiments/ecg_paper_experiments.py --exp A1 A2 A3
  python3 scripts/experiments/ecg_paper_experiments.py --exp B1 B7
  python3 scripts/experiments/ecg_paper_experiments.py --exp B8
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
    BASELINE_POLICIES, REORDER_POLICY_PAIRS, REORDER_VARIANTS,
    BENCHMARKS, BENCHMARKS_PREVIEW,
    ITERATIVE_BENCHMARKS, TRAVERSAL_BENCHMARKS,
    DEFAULT_CACHE, CACHE_SIZES_SWEEP,
    EVAL_GRAPHS, EVAL_GRAPHS_PREVIEW, ACCURACY_GRAPHS,
    ACCURACY_PAIRS, ECG_MODES,
    TIMEOUT_SIM, TIMEOUT_SIM_HEAVY, TRIALS,
    policy_env, format_cache_size,
)


# ============================================================================
# Graph File Discovery
# ============================================================================

def find_graph_file(graph_dir, graph_name):
    """Find the best available graph file in a graph directory.

    Prefers .sg (fast binary) over .mtx (text) over .el (edge list).
    Searches both the graph directory and common nested layouts from
    SuiteSparse downloads.

    Returns:
        str: Path to graph file, or None if not found.
    """
    base = Path(graph_dir) / graph_name
    if not base.exists():
        return None

    # Priority 1: Pre-converted .sg files (fastest loading)
    for pattern in [
        f"{graph_name}.sg",
        "graph.sg",
        "*.sg",
    ]:
        matches = sorted(base.glob(pattern))
        for m in matches:
            if m.is_file() or (m.is_symlink() and m.resolve().exists()):
                return str(m)

    # Priority 2: Matrix Market .mtx files (need -s flag at runtime)
    #   Check nested SuiteSparse layout: {name}/{Name}/{Name}.mtx
    for pattern in [
        f"{graph_name}.mtx",
        f"*/{graph_name}.mtx",
        f"*/*{graph_name}*.mtx",
    ]:
        matches = sorted(base.glob(pattern), key=lambda p: len(str(p)))
        for m in matches:
            if m.is_file() and m.stat().st_size > 1_000_000:
                return str(m)

    # Priority 3: Any .mtx larger than 1MB (skip metadata .mtx files)
    for m in sorted(base.rglob("*.mtx"), key=lambda p: -p.stat().st_size):
        if m.stat().st_size > 1_000_000:
            return str(m)

    # Priority 4: Edge list
    for m in base.glob("*.el"):
        if m.is_file():
            return str(m)

    return None


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

    hot_match = re.search(r"hot=([\d.]+)%", output)
    if hot_match:
        result["hot_fraction_pct"] = float(hot_match.group(1))

    time_match = re.search(r"Average:\s+([\d.]+)", output)
    if time_match:
        result["avg_time_s"] = float(time_match.group(1))

    return result


def run_sim(benchmark, graph_path, reorder_opt, policy,
            cache_config=None, extra_env=None, timeout=TIMEOUT_SIM,
            dry_run=False):
    """Run a single cache simulation and return parsed results."""
    binary = BIN_SIM_DIR / benchmark
    if not binary.exists() and not dry_run:
        return {"error": f"Binary not found: {binary}"}

    cmd = [str(binary), "-f", graph_path, "-s"]
    cmd += reorder_opt.split()
    cmd += ["-n", str(TRIALS)]
    env = policy_env(policy, cache_config, extra_env)

    if dry_run:
        env_str = f"CACHE_POLICY={policy}"
        if extra_env:
            env_str += " " + " ".join(f"{k}={v}" for k, v in extra_env.items())
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


# ############################################################################
#
#  SECTION A: ACCURACY VALIDATION
#
# ############################################################################

# ============================================================================
# A1: GRASP Invariants (Faldu et al., HPCA 2020)
# ============================================================================

def expA1_grasp_accuracy(graphs, benchmarks, dry_run, graph_dir):
    """Verify GRASP faithfulness to the reference paper.

    Tests three claims:
      1. DBG+GRASP < DBG+SRRIP (miss rate) -- degree-aware insertion helps
      2. Original+GRASP ~ Original+SRRIP -- without DBG, no region info
      3. P-OPT > GRASP (P-OPT still wins -- GRASP is heuristic, not oracle)
    """
    print("\n" + "=" * 70)
    print("  A1: GRASP Accuracy Validation (Faldu et al., HPCA 2020)")
    print("  Testing: GRASP beats SRRIP with DBG, equals SRRIP without")
    print("=" * 70)

    configs = [
        ("-o 5",  "SRRIP", {},  "DBG+SRRIP"),
        ("-o 5",  "GRASP", {},  "DBG+GRASP"),
        ("-o 0",  "SRRIP", {},  "Original+SRRIP"),
        ("-o 0",  "GRASP", {},  "Original+GRASP"),
        ("-o 5",  "POPT",  {},  "DBG+P-OPT"),       # Upper bound reference
        ("-o 0",  "LRU",   {},  "Original+LRU"),     # Lower bound reference
    ]

    results = []
    total = len(graphs) * len(benchmarks) * len(configs)
    done = 0

    for g in graphs:
        gpath = find_graph_file(graph_dir, g["name"])
        if not gpath:
            print(f"  [SKIP] Graph not found: {g['name']} in {graph_dir}")
            continue
        for bench in benchmarks:
            for reorder, policy, extra, label in configs:
                done += 1
                print(f"  [{done}/{total}] {g['short']}/{bench}/{label}", end="", flush=True)
                timeout = TIMEOUT_SIM_HEAVY if bench in ("bc", "sssp") else TIMEOUT_SIM
                r = run_sim(bench, gpath, reorder, policy,
                            extra_env=extra, timeout=timeout, dry_run=dry_run)
                r.update({"graph": g["short"], "graph_type": g["type"],
                          "benchmark": bench, "policy": policy,
                          "reorder": reorder, "label": label})
                results.append(r)
                _print_result(r)

    if not dry_run:
        _validate_grasp(results)

    return results


def _validate_grasp(results):
    """Check GRASP invariant claims against collected results."""
    print("\n  --- GRASP Validation Report ---")
    pass_count, fail_count = 0, 0

    groups = {}
    for r in results:
        if "l3_miss_rate" not in r:
            continue
        key = (r["graph"], r["benchmark"])
        groups.setdefault(key, {})[r["label"]] = r["l3_miss_rate"]

    # Claim 1: DBG+GRASP < DBG+SRRIP
    print("\n  Claim 1: DBG+GRASP should have lower L3 miss rate than DBG+SRRIP")
    for (g, b), rates in sorted(groups.items()):
        if "DBG+GRASP" in rates and "DBG+SRRIP" in rates:
            ok = rates["DBG+GRASP"] <= rates["DBG+SRRIP"]
            status = "PASS" if ok else "FAIL"
            delta = rates["DBG+SRRIP"] - rates["DBG+GRASP"]
            print(f"    {status}: {g}/{b}: GRASP={rates['DBG+GRASP']:.4f} "
                  f"SRRIP={rates['DBG+SRRIP']:.4f} (delta={delta:+.4f})")
            if ok:
                pass_count += 1
            else:
                fail_count += 1

    # Claim 2: Original+GRASP ~ Original+SRRIP (within 5%)
    print("\n  Claim 2: Original+GRASP ~ Original+SRRIP (within 5% relative)")
    for (g, b), rates in sorted(groups.items()):
        if "Original+GRASP" in rates and "Original+SRRIP" in rates:
            grasp = rates["Original+GRASP"]
            srrip = rates["Original+SRRIP"]
            rel_diff = abs(grasp - srrip) / max(srrip, 1e-10)
            ok = rel_diff < 0.05
            status = "PASS" if ok else "FAIL"
            print(f"    {status}: {g}/{b}: GRASP={grasp:.4f} "
                  f"SRRIP={srrip:.4f} (rel_diff={rel_diff:.1%})")
            if ok:
                pass_count += 1
            else:
                fail_count += 1

    # Claim 3: P-OPT > GRASP (lower miss rate)
    print("\n  Claim 3: P-OPT should beat GRASP (lower L3 miss rate)")
    for (g, b), rates in sorted(groups.items()):
        if "DBG+P-OPT" in rates and "DBG+GRASP" in rates:
            ok = rates["DBG+P-OPT"] <= rates["DBG+GRASP"]
            status = "PASS" if ok else "FAIL"
            print(f"    {status}: {g}/{b}: P-OPT={rates['DBG+P-OPT']:.4f} "
                  f"GRASP={rates['DBG+GRASP']:.4f}")
            if ok:
                pass_count += 1
            else:
                fail_count += 1

    print(f"\n  Summary: {pass_count} PASS, {fail_count} FAIL")


# ============================================================================
# A2: P-OPT Invariants (Balaji et al., HPCA 2021)
# ============================================================================

def expA2_popt_accuracy(graphs, benchmarks, dry_run, graph_dir):
    """Verify P-OPT faithfulness to the reference paper.

    Tests three claims:
      1. P-OPT beats all RRIP variants (LRU, SRRIP, GRASP)
      2. P-OPT is reorder-agnostic (Original+P-OPT ~ DBG+P-OPT)
      3. P-OPT beats LRU significantly (>10% L3 miss reduction)
    """
    print("\n" + "=" * 70)
    print("  A2: P-OPT Accuracy Validation (Balaji et al., HPCA 2021)")
    print("  Testing: P-OPT beats RRIP variants, reorder-agnostic")
    print("=" * 70)

    configs = [
        ("-o 0", "LRU",   {}, "Original+LRU"),
        ("-o 0", "SRRIP", {}, "Original+SRRIP"),
        ("-o 0", "POPT",  {}, "Original+P-OPT"),
        ("-o 5", "SRRIP", {}, "DBG+SRRIP"),
        ("-o 5", "GRASP", {}, "DBG+GRASP"),
        ("-o 5", "POPT",  {}, "DBG+P-OPT"),
    ]

    results = []
    total = len(graphs) * len(benchmarks) * len(configs)
    done = 0

    for g in graphs:
        gpath = find_graph_file(graph_dir, g["name"])
        if not gpath:
            print(f"  [SKIP] Graph not found: {g['name']} in {graph_dir}")
            continue
        for bench in benchmarks:
            for reorder, policy, extra, label in configs:
                done += 1
                print(f"  [{done}/{total}] {g['short']}/{bench}/{label}", end="", flush=True)
                timeout = TIMEOUT_SIM_HEAVY if bench in ("bc", "sssp") else TIMEOUT_SIM
                r = run_sim(bench, gpath, reorder, policy,
                            extra_env=extra, timeout=timeout, dry_run=dry_run)
                r.update({"graph": g["short"], "graph_type": g["type"],
                          "benchmark": bench, "policy": policy,
                          "reorder": reorder, "label": label})
                results.append(r)
                _print_result(r)

    if not dry_run:
        _validate_popt(results)

    return results


def _validate_popt(results):
    """Check P-OPT invariant claims."""
    print("\n  --- P-OPT Validation Report ---")
    pass_count, fail_count = 0, 0

    groups = {}
    for r in results:
        if "l3_miss_rate" not in r:
            continue
        key = (r["graph"], r["benchmark"])
        groups.setdefault(key, {})[r["label"]] = r["l3_miss_rate"]

    # Claim 1: P-OPT beats all RRIP variants
    print("\n  Claim 1: P-OPT should beat LRU, SRRIP, and GRASP")
    for (g, b), rates in sorted(groups.items()):
        popt = rates.get("DBG+P-OPT") or rates.get("Original+P-OPT")
        if popt is None:
            continue
        for rival_label in ["Original+LRU", "Original+SRRIP", "DBG+SRRIP", "DBG+GRASP"]:
            if rival_label in rates:
                ok = popt <= rates[rival_label]
                status = "PASS" if ok else "FAIL"
                print(f"    {status}: {g}/{b}: P-OPT={popt:.4f} vs "
                      f"{rival_label}={rates[rival_label]:.4f}")
                if ok:
                    pass_count += 1
                else:
                    fail_count += 1

    # Claim 2: P-OPT is reorder-agnostic (within 10% relative)
    print("\n  Claim 2: Original+P-OPT ~ DBG+P-OPT (within 10% relative)")
    for (g, b), rates in sorted(groups.items()):
        if "Original+P-OPT" in rates and "DBG+P-OPT" in rates:
            orig = rates["Original+P-OPT"]
            dbg = rates["DBG+P-OPT"]
            rel_diff = abs(orig - dbg) / max(orig, 1e-10)
            ok = rel_diff < 0.10
            status = "PASS" if ok else "WARN"
            print(f"    {status}: {g}/{b}: Original={orig:.4f} DBG={dbg:.4f} "
                  f"(rel_diff={rel_diff:.1%})")
            if ok:
                pass_count += 1
            else:
                fail_count += 1

    # Claim 3: P-OPT improvement vs LRU > 10%
    print("\n  Claim 3: P-OPT reduces L3 miss rate >10% vs LRU")
    for (g, b), rates in sorted(groups.items()):
        popt = rates.get("Original+P-OPT")
        lru = rates.get("Original+LRU")
        if popt is not None and lru is not None and lru > 0:
            reduction = (lru - popt) / lru
            ok = reduction > 0.10
            status = "PASS" if ok else "WARN"
            print(f"    {status}: {g}/{b}: LRU={lru:.4f} P-OPT={popt:.4f} "
                  f"(reduction={reduction:.1%})")
            if ok:
                pass_count += 1
            else:
                fail_count += 1

    print(f"\n  Summary: {pass_count} PASS, {fail_count} FAIL/WARN")


# ============================================================================
# A3: ECG Mode Equivalence Validation
# ============================================================================

def expA3_ecg_mode_accuracy(graphs, benchmarks, dry_run, graph_dir):
    """Verify ECG mode behavior matches expectations.

    Tests three claims:
      1. ECG(DBG_ONLY) ~ GRASP (within 3% -- same DBG insert + SRRIP eviction)
      2. ECG(POPT_PRIMARY) ~ P-OPT (within 5% -- P-OPT as primary tiebreaker)
      3. ECG(DBG_PRIMARY) between GRASP and P-OPT (combines both signals)
    """
    print("\n" + "=" * 70)
    print("  A3: ECG Mode Accuracy Validation")
    print("  Testing: DBG_ONLY~GRASP, POPT_PRIMARY~P-OPT, DBG_PRIMARY=sweet spot")
    print("=" * 70)

    configs = [
        ("-o 5", "GRASP", {},                          "DBG+GRASP"),
        ("-o 5", "POPT",  {},                          "DBG+P-OPT"),
        ("-o 5", "ECG",   {"ECG_MODE": "DBG_ONLY"},    "DBG+ECG(DBG_ONLY)"),
        ("-o 5", "ECG",   {"ECG_MODE": "POPT_PRIMARY"},"DBG+ECG(POPT_PRIMARY)"),
        ("-o 5", "ECG",   {"ECG_MODE": "DBG_PRIMARY"}, "DBG+ECG(DBG_PRIMARY)"),
    ]

    results = []
    total = len(graphs) * len(benchmarks) * len(configs)
    done = 0

    for g in graphs:
        gpath = find_graph_file(graph_dir, g["name"])
        if not gpath:
            print(f"  [SKIP] Graph not found: {g['name']} in {graph_dir}")
            continue
        for bench in benchmarks:
            for reorder, policy, extra, label in configs:
                done += 1
                print(f"  [{done}/{total}] {g['short']}/{bench}/{label}", end="", flush=True)
                timeout = TIMEOUT_SIM_HEAVY if bench in ("bc", "sssp") else TIMEOUT_SIM
                r = run_sim(bench, gpath, reorder, policy,
                            extra_env=extra, timeout=timeout, dry_run=dry_run)
                r.update({"graph": g["short"], "graph_type": g["type"],
                          "benchmark": bench, "policy": policy,
                          "reorder": reorder, "label": label})
                results.append(r)
                _print_result(r)

    if not dry_run:
        _validate_ecg_modes(results)

    return results


def _validate_ecg_modes(results):
    """Check ECG mode equivalence claims."""
    print("\n  --- ECG Mode Validation Report ---")
    pass_count, fail_count = 0, 0

    groups = {}
    for r in results:
        if "l3_miss_rate" not in r:
            continue
        key = (r["graph"], r["benchmark"])
        groups.setdefault(key, {})[r["label"]] = r["l3_miss_rate"]

    # Claim 1: ECG(DBG_ONLY) ~ GRASP (within 3%)
    print("\n  Claim 1: ECG(DBG_ONLY) ~ GRASP (within 3% relative)")
    for (g, b), rates in sorted(groups.items()):
        ecg = rates.get("DBG+ECG(DBG_ONLY)")
        grasp = rates.get("DBG+GRASP")
        if ecg is not None and grasp is not None:
            rel_diff = abs(ecg - grasp) / max(grasp, 1e-10)
            ok = rel_diff < 0.03
            status = "PASS" if ok else "FAIL"
            print(f"    {status}: {g}/{b}: ECG(DBG_ONLY)={ecg:.4f} "
                  f"GRASP={grasp:.4f} (rel_diff={rel_diff:.1%})")
            if ok:
                pass_count += 1
            else:
                fail_count += 1

    # Claim 2: ECG(POPT_PRIMARY) ~ P-OPT (within 5%)
    print("\n  Claim 2: ECG(POPT_PRIMARY) ~ P-OPT (within 5% relative)")
    for (g, b), rates in sorted(groups.items()):
        ecg = rates.get("DBG+ECG(POPT_PRIMARY)")
        popt = rates.get("DBG+P-OPT")
        if ecg is not None and popt is not None:
            rel_diff = abs(ecg - popt) / max(popt, 1e-10)
            ok = rel_diff < 0.05
            status = "PASS" if ok else "WARN"
            print(f"    {status}: {g}/{b}: ECG(POPT_PRIMARY)={ecg:.4f} "
                  f"P-OPT={popt:.4f} (rel_diff={rel_diff:.1%})")
            if ok:
                pass_count += 1
            else:
                fail_count += 1

    # Claim 3: ECG(DBG_PRIMARY) between GRASP and P-OPT
    print("\n  Claim 3: P-OPT <= ECG(DBG_PRIMARY) <= GRASP (miss rate ordering)")
    for (g, b), rates in sorted(groups.items()):
        ecg = rates.get("DBG+ECG(DBG_PRIMARY)")
        grasp = rates.get("DBG+GRASP")
        popt = rates.get("DBG+P-OPT")
        if ecg is not None and grasp is not None and popt is not None:
            # Miss rate: P-OPT <= ECG <= GRASP (lower is better)
            ok = popt <= ecg + 0.001 and ecg <= grasp + 0.001
            status = "PASS" if ok else "WARN"
            print(f"    {status}: {g}/{b}: P-OPT={popt:.4f} <= "
                  f"ECG(DBG_PRIMARY)={ecg:.4f} <= GRASP={grasp:.4f}")
            if ok:
                pass_count += 1
            else:
                fail_count += 1

    print(f"\n  Summary: {pass_count} PASS, {fail_count} FAIL/WARN")


# ############################################################################
#
#  SECTION B: PERFORMANCE SHOWCASE
#
# ############################################################################

# ============================================================================
# B1: Policy Comparison (all policies, fixed reorder)
# ============================================================================

def expB1_policy_comparison(graphs, benchmarks, policies, dry_run, graph_dir):
    """Compare all cache policies across benchmarks and graphs.

    All runs use DBG reordering (-o 5) so GRASP/ECG regions are meaningful.
    Also runs Original (-o 0) with LRU as absolute baseline.
    """
    print("\n" + "=" * 70)
    print("  B1: Cache Policy Comparison")
    print(f"  {len(graphs)} graphs x {len(benchmarks)} benchmarks x "
          f"{len(policies)} policies")
    print("=" * 70)

    results = []
    total = len(graphs) * len(benchmarks) * (len(policies) + 1)
    done = 0

    for g in graphs:
        gpath = find_graph_file(graph_dir, g["name"])
        if not gpath:
            print(f"  [SKIP] Graph not found: {g['name']} in {graph_dir}")
            continue
        for bench in benchmarks:
            # Baseline: Original+LRU
            done += 1
            print(f"  [{done}/{total}] {g['short']}/{bench}/Original+LRU",
                  end="", flush=True)
            timeout = TIMEOUT_SIM_HEAVY if bench in ("bc", "sssp") else TIMEOUT_SIM
            r = run_sim(bench, gpath, "-o 0", "LRU",
                        timeout=timeout, dry_run=dry_run)
            r.update({"graph": g["short"], "graph_type": g["type"],
                      "benchmark": bench, "policy": "Original+LRU",
                      "reorder": "-o 0"})
            results.append(r)
            _print_result(r)

            # All policies with DBG
            for policy in policies:
                done += 1
                print(f"  [{done}/{total}] {g['short']}/{bench}/DBG+{policy}",
                      end="", flush=True)
                r = run_sim(bench, gpath, "-o 5", policy,
                            timeout=timeout, dry_run=dry_run)
                r.update({"graph": g["short"], "graph_type": g["type"],
                          "benchmark": bench, "policy": f"DBG+{policy}",
                          "reorder": "-o 5"})
                results.append(r)
                _print_result(r)

    return results


# ============================================================================
# B2: Reorder Effect (how reordering alone affects each policy)
# ============================================================================

def expB2_reorder_effect(graphs, benchmarks, dry_run, graph_dir):
    """Show the effect of vertex reordering on each cache policy.

    For each reordering variant, tests key policies to isolate:
    - Reorder-only effect (LRU, same policy, different ordering)
    - Reorder+policy synergy (GRASP needs DBG, P-OPT is agnostic)
    """
    print("\n" + "=" * 70)
    print("  B2: Reorder Effect on Cache Policies")
    print(f"  {len(graphs)} graphs x {len(benchmarks)} benchmarks x "
          f"{len(REORDER_VARIANTS)} reorders x key policies")
    print("=" * 70)

    key_policies = ["LRU", "SRRIP", "GRASP", "POPT", "ECG"]
    results = []
    total = len(graphs) * len(benchmarks) * len(REORDER_VARIANTS) * len(key_policies)
    done = 0

    for g in graphs:
        gpath = find_graph_file(graph_dir, g["name"])
        if not gpath:
            print(f"  [SKIP] Graph not found: {g['name']} in {graph_dir}")
            continue
        for bench in benchmarks:
            for reorder_opt, reorder_name in REORDER_VARIANTS:
                for policy in key_policies:
                    done += 1
                    label = f"{reorder_name}+{policy}"
                    print(f"  [{done}/{total}] {g['short']}/{bench}/{label}",
                          end="", flush=True)
                    timeout = TIMEOUT_SIM_HEAVY if bench in ("bc", "sssp") \
                        else TIMEOUT_SIM
                    r = run_sim(bench, gpath, reorder_opt, policy,
                                timeout=timeout, dry_run=dry_run)
                    r.update({"graph": g["short"], "graph_type": g["type"],
                              "benchmark": bench, "reorder": reorder_opt,
                              "reorder_name": reorder_name,
                              "policy": policy, "label": label})
                    results.append(r)
                    _print_result(r)

    if not dry_run and results:
        _summarize_reorder_effect(results)

    return results


def _summarize_reorder_effect(results):
    """Print summary table of reorder x policy miss rates."""
    print("\n  --- Reorder Effect Summary (geo-mean L3 miss rate) ---\n")

    data = {}
    for r in results:
        if "l3_miss_rate" not in r:
            continue
        key = (r.get("reorder_name", "?"), r["policy"])
        data.setdefault(key, []).append(r["l3_miss_rate"])

    reorders = sorted(set(k[0] for k in data.keys()))
    policies = sorted(set(k[1] for k in data.keys()))

    header = f"  {'Reorder':15s}"
    for p in policies:
        header += f" | {p:>8s}"
    print(header)
    print("  " + "-" * len(header))

    for reorder in reorders:
        row = f"  {reorder:15s}"
        for policy in policies:
            rates = data.get((reorder, policy), [])
            if rates:
                gm = _geo_mean(rates)
                row += f" | {gm:8.4f}"
            else:
                row += f" | {'N/A':>8s}"
        print(row)


# ============================================================================
# B3: Reorder x Policy Interaction (full matrix)
# ============================================================================

def expB3_reorder_interaction(graphs, benchmarks, dry_run, graph_dir):
    """Full reorder x policy interaction matrix."""
    print("\n" + "=" * 70)
    print("  B3: Reordering x Policy Interaction")
    print(f"  {len(graphs)} graphs x {len(benchmarks)} benchmarks x "
          f"{len(REORDER_POLICY_PAIRS)} pairs")
    print("=" * 70)

    results = []
    total = len(graphs) * len(benchmarks) * len(REORDER_POLICY_PAIRS)
    done = 0

    for g in graphs:
        gpath = find_graph_file(graph_dir, g["name"])
        if not gpath:
            print(f"  [SKIP] Graph not found: {g['name']} in {graph_dir}")
            continue
        for bench in benchmarks:
            for reorder_opt, policy, label in REORDER_POLICY_PAIRS:
                done += 1
                print(f"  [{done}/{total}] {g['short']}/{bench}/{label}",
                      end="", flush=True)
                timeout = TIMEOUT_SIM_HEAVY if bench in ("bc", "sssp") \
                    else TIMEOUT_SIM
                r = run_sim(bench, gpath, reorder_opt, policy,
                            timeout=timeout, dry_run=dry_run)
                r.update({"graph": g["short"], "graph_type": g["type"],
                          "benchmark": bench, "reorder": reorder_opt,
                          "policy": policy, "label": label})
                results.append(r)
                _print_result(r)

    return results


# ============================================================================
# B4: Cache Size Sweep
# ============================================================================

def expB4_cache_sweep(graphs, benchmarks, policies, dry_run, graph_dir):
    """Sweep L3 cache size from 32KB to 64MB with key policies."""
    print("\n" + "=" * 70)
    print("  B4: Cache Size Sensitivity")
    print(f"  {len(graphs)} graphs x {len(benchmarks)} benchmarks x "
          f"{len(policies)} policies x {len(CACHE_SIZES_SWEEP)} sizes")
    print("=" * 70)

    results = []
    total = len(graphs) * len(benchmarks) * len(policies) * len(CACHE_SIZES_SWEEP)
    done = 0

    for g in graphs:
        gpath = find_graph_file(graph_dir, g["name"])
        if not gpath:
            print(f"  [SKIP] Graph not found: {g['name']} in {graph_dir}")
            continue
        for bench in benchmarks:
            for policy in policies:
                for cache_size in CACHE_SIZES_SWEEP:
                    done += 1
                    sz_str = format_cache_size(cache_size)
                    print(f"  [{done}/{total}] {g['short']}/{bench}/"
                          f"{policy}/{sz_str}", end="", flush=True)
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
# B5: Algorithm-Type Analysis (derived from B1)
# ============================================================================

def expB5_algorithm_analysis(b1_results):
    """Group B1 results by algorithm access pattern."""
    print("\n" + "=" * 70)
    print("  B5: Algorithm-Type Analysis (derived from B1)")
    print("=" * 70)

    analysis = {}
    for category, benches in [("Iterative", ITERATIVE_BENCHMARKS),
                               ("Traversal", TRAVERSAL_BENCHMARKS)]:
        analysis[category] = {}
        for r in b1_results:
            if r.get("benchmark") in benches and "l3_miss_rate" in r:
                policy = r["policy"]
                analysis[category].setdefault(policy, []).append(
                    r["l3_miss_rate"])

        print(f"\n  {category} algorithms (geo-mean L3 miss rate):")
        for policy in sorted(analysis[category].keys()):
            rates = analysis[category][policy]
            if rates:
                geo_mean = _geo_mean(rates)
                print(f"    {policy:25s}: {geo_mean:.4f} "
                      f"({len(rates)} samples)")

    return analysis


# ============================================================================
# B6: Graph-Type Sensitivity (derived from B1)
# ============================================================================

def expB6_graph_sensitivity(b1_results):
    """Group B1 results by graph topology type."""
    print("\n" + "=" * 70)
    print("  B6: Graph-Type Sensitivity (derived from B1)")
    print("=" * 70)

    analysis = {}
    for r in b1_results:
        if "l3_miss_rate" not in r:
            continue
        gtype = r.get("graph_type", "Unknown")
        policy = r["policy"]
        analysis.setdefault(gtype, {}).setdefault(policy, []).append(
            r["l3_miss_rate"])

    for gtype in sorted(analysis.keys()):
        print(f"\n  {gtype} graphs (geo-mean L3 miss rate):")
        for policy in sorted(analysis[gtype].keys()):
            rates = analysis[gtype][policy]
            if rates:
                geo_mean = _geo_mean(rates)
                print(f"    {policy:25s}: {geo_mean:.4f} "
                      f"({len(rates)} samples)")

    return analysis


# ============================================================================
# B7: ECG Mode Comparison
# ============================================================================

def expB7_ecg_mode_comparison(graphs, benchmarks, dry_run, graph_dir):
    """Compare the three ECG modes across all benchmarks and graphs.

    Shows the trade-off between DBG structure and P-OPT oracle:
    - DBG_PRIMARY: cheap structure + rare oracle tiebreak (default)
    - POPT_PRIMARY: oracle primary + structure tiebreak
    - DBG_ONLY: pure structure, no oracle overhead
    """
    print("\n" + "=" * 70)
    print("  B7: ECG Mode Comparison")
    print(f"  {len(graphs)} graphs x {len(benchmarks)} benchmarks x "
          f"3 modes + references")
    print("=" * 70)

    configs = [
        ("-o 5", "GRASP", {},                           "GRASP"),
        ("-o 5", "POPT",  {},                           "P-OPT"),
        ("-o 5", "ECG",   {"ECG_MODE": "DBG_ONLY"},     "ECG(DBG_ONLY)"),
        ("-o 5", "ECG",   {"ECG_MODE": "POPT_PRIMARY"}, "ECG(POPT_PRIMARY)"),
        ("-o 5", "ECG",   {"ECG_MODE": "DBG_PRIMARY"},  "ECG(DBG_PRIMARY)"),
    ]

    results = []
    total = len(graphs) * len(benchmarks) * len(configs)
    done = 0

    for g in graphs:
        gpath = find_graph_file(graph_dir, g["name"])
        if not gpath:
            print(f"  [SKIP] Graph not found: {g['name']} in {graph_dir}")
            continue
        for bench in benchmarks:
            for reorder, policy, extra, label in configs:
                done += 1
                print(f"  [{done}/{total}] {g['short']}/{bench}/{label}",
                      end="", flush=True)
                timeout = TIMEOUT_SIM_HEAVY if bench in ("bc", "sssp") \
                    else TIMEOUT_SIM
                r = run_sim(bench, gpath, reorder, policy,
                            extra_env=extra, timeout=timeout, dry_run=dry_run)
                r.update({"graph": g["short"], "graph_type": g["type"],
                          "benchmark": bench, "policy": policy,
                          "reorder": reorder, "label": label})
                results.append(r)
                _print_result(r)

    if not dry_run and results:
        print("\n  --- ECG Mode Summary (geo-mean L3 miss rate) ---")
        data = {}
        for r in results:
            if "l3_miss_rate" in r:
                data.setdefault(r["label"], []).append(r["l3_miss_rate"])
        for label in ["GRASP", "ECG(DBG_ONLY)", "ECG(DBG_PRIMARY)",
                      "ECG(POPT_PRIMARY)", "P-OPT"]:
            rates = data.get(label, [])
            if rates:
                print(f"    {label:22s}: {_geo_mean(rates):.4f} "
                      f"({len(rates)} samples)")

    return results


# ============================================================================
# B8: Fat-ID Bit Allocation Analysis (analytical, no simulation)
# ============================================================================

def expB8_fatid_analysis(graphs):
    """Show adaptive fat-ID bit allocation for each graph size."""
    print("\n" + "=" * 70)
    print("  B8: Fat-ID Bit Allocation Analysis")
    print("=" * 70)
    print(f"  {'Graph':12s} | {'|V|':>10s} | {'ID':>3s} | "
          f"{'--- 32-bit ---':^26s} | {'--- 64-bit ---':^26s} | vs P-OPT")
    print(f"  {'':12s} | {'':>10s} | {'':>3s} | "
          f"{'spare':>5s} {'DBG':>3s} {'POPT':>4s} {'PFX':>3s} "
          f"{'levels':>6s} | "
          f"{'spare':>5s} {'DBG':>3s} {'POPT':>4s} {'PFX':>3s} "
          f"{'levels':>6s} |")
    print("  " + "-" * 95)

    results = []
    for g in graphs:
        v = int(g.get("vertices_m", 1) * 1_000_000)
        id_bits = max(1, math.ceil(math.log2(max(v, 2))))

        spare_32 = 32 - id_bits
        d32, p32, f32 = _allocate_bits(spare_32)
        levels_32 = 2 ** p32 if p32 > 0 else 0

        spare_64 = 64 - id_bits
        d64, p64, f64 = _allocate_bits(spare_64)
        levels_64 = 2 ** p64 if p64 > 0 else 0

        vs_popt = f"{levels_64 / 128 * 100:.0f}%" if levels_64 else "N/A"

        entry = {
            "graph": g["short"], "vertices": v, "id_bits": id_bits,
            "spare_32": spare_32, "dbg_32": d32, "popt_32": p32,
            "pfx_32": f32, "levels_32": levels_32,
            "spare_64": spare_64, "dbg_64": d64, "popt_64": p64,
            "pfx_64": f64, "levels_64": levels_64,
            "vs_popt_matrix": vs_popt,
        }
        results.append(entry)

        print(f"  {g['short']:12s} | {v:>10,} | {id_bits:>3d} | "
              f"{spare_32:>5d} {d32:>3d} {p32:>4d} {f32:>3d} "
              f"{levels_32:>6d} | "
              f"{spare_64:>5d} {d64:>3d} {p64:>4d} {f64:>3d} "
              f"{levels_64:>6d} | "
              f"{vs_popt:>6s}")

    print("\n  P-OPT matrix uses 7-bit precision (128 levels), "
          "consuming 2+ LLC ways.")
    print("  Fat-ID encoding uses 0 LLC capacity. "
          "64-bit mode exceeds P-OPT precision.")

    return results


# ############################################################################
#
#  HELPERS
#
# ############################################################################

def _print_result(r):
    """Print a compact result line."""
    if "l3_miss_rate" in r:
        print(f"  L3_miss={r['l3_miss_rate']:.4f}")
    elif "error" in r:
        print(f"  ERROR: {r['error'][:50]}")
    elif "dry_run" in r:
        pass
    else:
        print()


def _geo_mean(values):
    """Geometric mean of positive values."""
    if not values:
        return 0.0
    product = 1.0
    for v in values:
        product *= max(v, 1e-10)
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
    print(f"  -> Saved to: {path}")
    return path


# ############################################################################
#
#  MAIN
#
# ############################################################################

ALL_EXPERIMENTS = {
    "A1": ("GRASP Accuracy Validation",     "accuracy"),
    "A2": ("P-OPT Accuracy Validation",     "accuracy"),
    "A3": ("ECG Mode Accuracy Validation",  "accuracy"),
    "B1": ("Policy Comparison",             "performance"),
    "B2": ("Reorder Effect",                "performance"),
    "B3": ("Reorder x Policy Interaction",  "performance"),
    "B4": ("Cache Size Sweep",              "performance"),
    "B5": ("Algorithm-Type Analysis",       "performance"),
    "B6": ("Graph-Type Sensitivity",        "performance"),
    "B7": ("ECG Mode Comparison",           "performance"),
    "B8": ("Fat-ID Analysis",               "performance"),
}


def main():
    parser = argparse.ArgumentParser(
        description="ECG Paper Experiments Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Experiments are organized into two sections:

  Section A: Accuracy Validation (faithfulness to reference papers)
    A1  GRASP Invariants    -- Faldu et al., HPCA 2020
    A2  P-OPT Invariants   -- Balaji et al., HPCA 2021
    A3  ECG Mode Equiv.     -- DBG_ONLY~GRASP, POPT_PRIMARY~P-OPT

  Section B: Performance Showcase (comparison and reorder effects)
    B1  Policy Comparison   -- All 9 policies x benchmarks x graphs
    B2  Reorder Effect      -- How reordering affects each policy
    B3  Reorder x Policy    -- Full interaction matrix
    B4  Cache Size Sweep    -- L3 32KB-64MB sensitivity
    B5  Algorithm Analysis  -- Iterative vs traversal (derived from B1)
    B6  Graph Sensitivity   -- Social/road/citation (derived from B1)
    B7  ECG Mode Comparison -- DBG_PRIMARY vs POPT_PRIMARY vs DBG_ONLY
    B8  Fat-ID Analysis     -- Bit allocation per graph size (analytical)

Examples:
  python3 %(prog)s --all --graph-dir /data/graphs
  python3 %(prog)s --section A --preview      # Accuracy only
  python3 %(prog)s --section B --preview      # Performance only
  python3 %(prog)s --exp A1 A2 A3             # Specific experiments
  python3 %(prog)s --exp B8                   # Analytical -- no graphs needed
  python3 %(prog)s --all --dry-run
        """)
    parser.add_argument("--all", action="store_true",
                        help="Run all experiments")
    parser.add_argument("--section", choices=["A", "B"],
                        help="Run entire section (A=Accuracy, B=Performance)")
    parser.add_argument("--exp", nargs="+",
                        choices=list(ALL_EXPERIMENTS.keys()),
                        help="Run specific experiments")
    parser.add_argument("--preview", action="store_true",
                        help="Use smaller graph/benchmark set")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands only")
    parser.add_argument("--graph-dir", default=".",
                        help="Base directory for graphs")

    args = parser.parse_args()

    if not args.all and not args.section and not args.exp:
        parser.print_help()
        return

    # Determine which experiments to run
    if args.all:
        experiments = list(ALL_EXPERIMENTS.keys())
    elif args.section:
        experiments = [k for k in ALL_EXPERIMENTS if k.startswith(args.section)]
    else:
        experiments = args.exp

    # Select graph/benchmark sets
    if args.preview:
        perf_graphs = EVAL_GRAPHS_PREVIEW
        acc_graphs = EVAL_GRAPHS_PREVIEW[:2]
        benchmarks = BENCHMARKS_PREVIEW
        policies = PREVIEW_POLICIES
    else:
        perf_graphs = EVAL_GRAPHS
        acc_graphs = ACCURACY_GRAPHS
        benchmarks = BENCHMARKS
        policies = ALL_POLICIES

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'=' * 70}")
    print(f"  ECG Paper Experiments -- {ts}")
    print(f"  Experiments: {experiments}")
    print(f"  Accuracy graphs: {len(acc_graphs)} | "
          f"Perf graphs: {len(perf_graphs)}")
    print(f"  Benchmarks: {len(benchmarks)} | Policies: {len(policies)}")
    print(f"{'=' * 70}")

    b1_results = None

    for exp_id in sorted(experiments):
        t0 = time.time()
        print(f"\n{'-' * 70}")
        print(f"  Starting {exp_id}: {ALL_EXPERIMENTS[exp_id][0]}")
        print(f"{'-' * 70}")

        if exp_id == "A1":
            r = expA1_grasp_accuracy(
                acc_graphs, benchmarks, args.dry_run, args.graph_dir)
            if not args.dry_run:
                save_results(r, f"expA1_grasp_accuracy_{ts}")

        elif exp_id == "A2":
            r = expA2_popt_accuracy(
                acc_graphs, benchmarks, args.dry_run, args.graph_dir)
            if not args.dry_run:
                save_results(r, f"expA2_popt_accuracy_{ts}")

        elif exp_id == "A3":
            r = expA3_ecg_mode_accuracy(
                acc_graphs, benchmarks, args.dry_run, args.graph_dir)
            if not args.dry_run:
                save_results(r, f"expA3_ecg_mode_accuracy_{ts}")

        elif exp_id == "B1":
            b1_results = expB1_policy_comparison(
                perf_graphs, benchmarks, policies,
                args.dry_run, args.graph_dir)
            if not args.dry_run:
                save_results(b1_results, f"expB1_policy_comparison_{ts}")

        elif exp_id == "B2":
            r = expB2_reorder_effect(
                perf_graphs, benchmarks, args.dry_run, args.graph_dir)
            if not args.dry_run:
                save_results(r, f"expB2_reorder_effect_{ts}")

        elif exp_id == "B3":
            r = expB3_reorder_interaction(
                perf_graphs, benchmarks, args.dry_run, args.graph_dir)
            if not args.dry_run:
                save_results(r, f"expB3_reorder_interaction_{ts}")

        elif exp_id == "B4":
            sweep_policies = PREVIEW_POLICIES if args.preview else \
                ["LRU", "SRRIP", "GRASP", "POPT", "ECG"]
            sweep_benches = ["pr"] if args.preview else ["pr", "bfs"]
            r = expB4_cache_sweep(
                perf_graphs, sweep_benches, sweep_policies,
                args.dry_run, args.graph_dir)
            if not args.dry_run:
                save_results(r, f"expB4_cache_sweep_{ts}")

        elif exp_id == "B5":
            if b1_results is None:
                print("\n  B5 requires B1 data. Running B1 first...")
                b1_results = expB1_policy_comparison(
                    perf_graphs, benchmarks, policies,
                    args.dry_run, args.graph_dir)
            r = expB5_algorithm_analysis(b1_results)
            if not args.dry_run:
                save_results(r, f"expB5_algorithm_analysis_{ts}")

        elif exp_id == "B6":
            if b1_results is None:
                print("\n  B6 requires B1 data. Running B1 first...")
                b1_results = expB1_policy_comparison(
                    perf_graphs, benchmarks, policies,
                    args.dry_run, args.graph_dir)
            r = expB6_graph_sensitivity(b1_results)
            if not args.dry_run:
                save_results(r, f"expB6_graph_sensitivity_{ts}")

        elif exp_id == "B7":
            r = expB7_ecg_mode_comparison(
                perf_graphs, benchmarks, args.dry_run, args.graph_dir)
            if not args.dry_run:
                save_results(r, f"expB7_ecg_mode_comparison_{ts}")

        elif exp_id == "B8":
            r = expB8_fatid_analysis(perf_graphs)
            save_results(r, f"expB8_fatid_analysis_{ts}")

        elapsed = time.time() - t0
        print(f"\n  {exp_id} completed in {elapsed:.1f}s")

    print(f"\n{'=' * 70}")
    print(f"  All experiments complete. Results in: {RESULTS_DIR}/")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
