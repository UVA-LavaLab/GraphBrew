#!/usr/bin/env python3
"""
Oracle Analysis for AdaptiveOrder (Algorithm 14).

Phase 4 deliverable: computes selection accuracy, regret distribution,
and confusion matrix by comparing AdaptiveOrder's choices against the
whole-graph oracle (best single algorithm per graph×benchmark).

Usage:
    # Via entry point
    python3 scripts/graphbrew_experiment.py --oracle-analysis

    # Or directly via module
    python3 -m scripts.lib.oracle --results-dir results/
    python3 -m scripts.lib.oracle --run-experiment --size small
    python3 -m scripts.lib.oracle --results-dir results/ --benchmarks pr bfs
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .utils import _VARIANT_ALGO_REGISTRY, ALGORITHMS, DISPLAY_TO_CANONICAL


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OracleEntry:
    """Oracle best for one graph×benchmark."""
    graph: str
    benchmark: str
    oracle_algo: str
    oracle_time: float
    adaptive_algo: str  # what adaptive's dominant selection was
    adaptive_time: float  # time if we used adaptive's dominant algo for whole graph
    regret: float  # (adaptive_time - oracle_time) / oracle_time
    rank: int  # rank of adaptive's choice among all algorithms
    total_algos: int
    all_times: Dict[str, float] = field(default_factory=dict)


@dataclass
class OracleReport:
    """Aggregated oracle comparison report."""
    # Per graph×benchmark entries
    entries: List[OracleEntry] = field(default_factory=list)
    # Summary stats
    accuracy: float = 0.0           # fraction where adaptive == oracle
    top3_rate: float = 0.0          # fraction where adaptive rank <= 3
    mean_regret: float = 0.0        # average regret
    median_regret: float = 0.0
    p95_regret: float = 0.0
    max_regret: float = 0.0
    # Confusion matrix: oracle_algo -> adaptive_algo -> count
    confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # Per-benchmark breakdown
    per_benchmark: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Regret distribution buckets
    regret_buckets: Dict[str, int] = field(default_factory=dict)
    # Algorithm frequency (how often each is oracle / adaptive)
    oracle_freq: Dict[str, int] = field(default_factory=dict)
    adaptive_freq: Dict[str, int] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm name normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_algo_name(name: str) -> str:
    """
    Normalize algorithm names for consistent comparison.
    
    Maps adaptive output names (e.g., 'GOrder', 'HubClusterDBG') to
    benchmark result names (e.g., 'GORDER', 'HUBCLUSTERDBG').
    Uses DISPLAY_TO_CANONICAL from SSOT (utils.py).
    """
    return DISPLAY_TO_CANONICAL.get(name, name)


# Candidate algorithms: the ones AdaptiveOrder can select from.
# These must be present in benchmark results for oracle to work.
CANDIDATE_ALGOS = {
    "ORIGINAL", "RANDOM", "SORT", "HUBSORT", "HUBCLUSTER",
    "DBG", "HUBSORTDBG", "HUBCLUSTERDBG",
    "GORDER", "CORDER", "RCM_default",
}

# Also include RabbitOrder variants and GraphBrew variants
CANDIDATE_PREFIXES = {"RABBITORDER", "GraphBrewOrder", "LeidenOrder"}


def is_candidate(algo_name: str) -> bool:
    """Check if an algorithm is a candidate that AdaptiveOrder could select."""
    if algo_name in CANDIDATE_ALGOS:
        return True
    for prefix in CANDIDATE_PREFIXES:
        if algo_name.startswith(prefix):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_benchmark_data(results_dir: str) -> List[dict]:
    """Load and merge all benchmark JSON files."""
    pattern = os.path.join(results_dir, "benchmark_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No benchmark files found in {results_dir}")
        return []
    
    all_records = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
            all_records.extend(data)
    
    print(f"Loaded {len(all_records)} benchmark records from {len(files)} files")
    return all_records


def load_adaptive_data(results_dir: str) -> List[dict]:
    """Load and merge all adaptive analysis JSON files."""
    pattern = os.path.join(results_dir, "adaptive_analysis_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No adaptive analysis files found in {results_dir}")
        return []
    
    all_records = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
            all_records.extend(data)
    
    print(f"Loaded {len(all_records)} adaptive analysis records from {len(files)} files")
    return all_records


def build_time_lookup(
    bench_data: List[dict],
    benchmarks: Optional[List[str]] = None
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Build lookup: graph -> benchmark -> algorithm -> time.
    
    When multiple records exist for the same triple, keeps the one
    with the most trials (most reliable).
    """
    # Track best record by trial count
    best: Dict[str, Dict[str, Dict[str, Tuple[float, int]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    
    for rec in bench_data:
        if not rec.get("success", True):
            continue
        graph = rec["graph"]
        bench = rec["benchmark"]
        algo = rec["algorithm"]
        time_s = rec.get("time_seconds", 0)
        trials = rec.get("trials", 1)
        
        if benchmarks and bench not in benchmarks:
            continue
        if time_s <= 0:
            continue
        
        existing = best[graph][bench].get(algo)
        if existing is None or trials > existing[1]:
            best[graph][bench][algo] = (time_s, trials)
    
    # Flatten to graph -> bench -> algo -> time
    lookup: Dict[str, Dict[str, Dict[str, float]]] = {}
    for graph in best:
        lookup[graph] = {}
        for bench in best[graph]:
            lookup[graph][bench] = {
                algo: time_trials[0]
                for algo, time_trials in best[graph][bench].items()
            }
    
    return lookup


def build_adaptive_lookup(
    adaptive_data: List[dict]
) -> Dict[str, dict]:
    """
    Build lookup: graph -> adaptive analysis record.
    
    Keeps the most recent record per graph (by position in merged list).
    """
    lookup = {}
    for rec in adaptive_data:
        graph = rec.get("graph", "")
        if graph:
            lookup[graph] = rec  # last one wins (most recent)
    return lookup


# ─────────────────────────────────────────────────────────────────────────────
# Oracle computation
# ─────────────────────────────────────────────────────────────────────────────

def get_dominant_adaptive_algo(adaptive_rec: dict) -> str:
    """
    Get the dominant algorithm selected by AdaptiveOrder for a graph.
    
    Uses algorithm_distribution to find the most frequently selected algorithm.
    """
    dist = adaptive_rec.get("algorithm_distribution", {})
    if not dist:
        return "ORIGINAL"
    
    # Find most frequently selected
    dominant = max(dist, key=dist.get)
    return normalize_algo_name(dominant)


def find_best_matching_algo(
    adaptive_algo: str,
    available_algos: Dict[str, float]
) -> Tuple[str, float]:
    """
    Find the benchmark algorithm that best matches the adaptive selection.
    
    Handles name mapping between adaptive output and benchmark result names.
    Returns (algo_name, time).
    """
    # Direct match
    if adaptive_algo in available_algos:
        return adaptive_algo, available_algos[adaptive_algo]
    
    # Try case-insensitive match
    upper = adaptive_algo.upper()
    for algo, time in available_algos.items():
        if algo.upper() == upper:
            return algo, time
    
    # Try prefix match (e.g., "RABBITORDER" matches "RABBITORDER_csr")
    for algo, time in available_algos.items():
        if algo.startswith(adaptive_algo) or adaptive_algo.startswith(algo):
            return algo, time
    
    # Default variant matches — auto-generated from SSOT variant registry
    _DEFAULT_VARIANTS = {
        ALGORITHMS[aid]: [f"{pfx}{v}" for v in variants]
        for aid, (pfx, variants, _) in _VARIANT_ALGO_REGISTRY.items()
    }
    if adaptive_algo in _DEFAULT_VARIANTS:
        for variant in _DEFAULT_VARIANTS[adaptive_algo]:
            if variant in available_algos:
                return variant, available_algos[variant]
    
    return adaptive_algo, float("inf")


def compute_oracle(
    time_lookup: Dict[str, Dict[str, Dict[str, float]]],
    adaptive_lookup: Dict[str, dict],
    benchmarks: Optional[List[str]] = None,
    candidate_only: bool = True,
) -> OracleReport:
    """
    Compute oracle comparison for all graph×benchmark pairs.
    
    Args:
        time_lookup: graph -> benchmark -> algorithm -> time
        adaptive_lookup: graph -> adaptive analysis record
        benchmarks: restrict to these benchmarks (None = all)
        candidate_only: only consider candidate algorithms for oracle
    """
    report = OracleReport()
    confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    oracle_freq: Counter = Counter()
    adaptive_freq: Counter = Counter()
    per_bench: Dict[str, List[OracleEntry]] = defaultdict(list)
    
    graphs_with_adaptive = set(adaptive_lookup.keys()) & set(time_lookup.keys())
    
    if not graphs_with_adaptive:
        print("WARNING: No graphs have both benchmark data and adaptive analysis data")
        print(f"  Benchmark graphs: {sorted(time_lookup.keys())[:5]}...")
        print(f"  Adaptive graphs: {sorted(adaptive_lookup.keys())[:5]}...")
        return report
    
    for graph in sorted(graphs_with_adaptive):
        adaptive_rec = adaptive_lookup[graph]
        dominant_algo = get_dominant_adaptive_algo(adaptive_rec)
        
        for bench in sorted(time_lookup[graph]):
            if benchmarks and bench not in benchmarks:
                continue
            
            algo_times = time_lookup[graph][bench]
            
            # Filter to candidates only if requested
            if candidate_only:
                filtered = {a: t for a, t in algo_times.items() if is_candidate(a)}
            else:
                filtered = dict(algo_times)
            
            if not filtered:
                continue
            
            # Oracle: best algorithm
            oracle_algo = min(filtered, key=filtered.get)
            oracle_time = filtered[oracle_algo]
            
            # Adaptive: find its time
            adaptive_name, adaptive_time = find_best_matching_algo(
                dominant_algo, algo_times
            )
            
            # Rank of adaptive among all candidates
            sorted_algos = sorted(filtered.items(), key=lambda x: x[1])
            rank = len(sorted_algos)  # worst case
            for i, (a, _) in enumerate(sorted_algos, 1):
                if a == adaptive_name:
                    rank = i
                    break
            
            # Regret
            if oracle_time > 0 and adaptive_time < float("inf"):
                regret = (adaptive_time - oracle_time) / oracle_time
            else:
                regret = 0.0
            
            entry = OracleEntry(
                graph=graph,
                benchmark=bench,
                oracle_algo=oracle_algo,
                oracle_time=oracle_time,
                adaptive_algo=adaptive_name,
                adaptive_time=adaptive_time,
                regret=regret,
                rank=rank,
                total_algos=len(filtered),
                all_times=filtered,
            )
            report.entries.append(entry)
            per_bench[bench].append(entry)
            
            # Update confusion matrix
            confusion[oracle_algo][adaptive_name] += 1
            oracle_freq[oracle_algo] += 1
            adaptive_freq[adaptive_name] += 1
    
    if not report.entries:
        return report
    
    # Compute summary stats
    n = len(report.entries)
    correct = sum(1 for e in report.entries if e.adaptive_algo == e.oracle_algo)
    top3 = sum(1 for e in report.entries if e.rank <= 3)
    regrets = sorted(e.regret for e in report.entries)
    
    report.accuracy = correct / n
    report.top3_rate = top3 / n
    report.mean_regret = sum(regrets) / n
    report.median_regret = regrets[n // 2]
    report.p95_regret = regrets[int(n * 0.95)]
    report.max_regret = regrets[-1]
    report.confusion = {k: dict(v) for k, v in confusion.items()}
    report.oracle_freq = dict(oracle_freq)
    report.adaptive_freq = dict(adaptive_freq)
    
    # Regret distribution buckets
    buckets = {"0%": 0, "0-5%": 0, "5-10%": 0, "10-20%": 0, "20-50%": 0, ">50%": 0}
    for r in regrets:
        if r <= 0:
            buckets["0%"] += 1
        elif r <= 0.05:
            buckets["0-5%"] += 1
        elif r <= 0.10:
            buckets["5-10%"] += 1
        elif r <= 0.20:
            buckets["10-20%"] += 1
        elif r <= 0.50:
            buckets["20-50%"] += 1
        else:
            buckets[">50%"] += 1
    report.regret_buckets = buckets
    
    # Per-benchmark stats
    for bench, entries in per_bench.items():
        nb = len(entries)
        c = sum(1 for e in entries if e.adaptive_algo == e.oracle_algo)
        t3 = sum(1 for e in entries if e.rank <= 3)
        regs = [e.regret for e in entries]
        report.per_benchmark[bench] = {
            "accuracy": c / nb,
            "top3_rate": t3 / nb,
            "mean_regret": sum(regs) / nb,
            "median_regret": sorted(regs)[nb // 2],
            "count": nb,
        }
    
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────

def print_report(report: OracleReport) -> None:
    """Pretty-print the oracle comparison report."""
    n = len(report.entries)
    if n == 0:
        print("No entries to report.")
        return
    
    print("\n" + "=" * 72)
    print("  ADAPTIVE ORDER ORACLE ANALYSIS")
    print("=" * 72)
    
    # Overall summary
    print(f"\n{'Metric':<30} {'Value':>15}")
    print("-" * 45)
    print(f"{'Graph×Benchmark pairs':<30} {n:>15}")
    print(f"{'Selection accuracy':<30} {report.accuracy:>14.1%}")
    print(f"{'Top-3 rate':<30} {report.top3_rate:>14.1%}")
    print(f"{'Mean regret':<30} {report.mean_regret:>14.1%}")
    print(f"{'Median regret':<30} {report.median_regret:>14.1%}")
    print(f"{'P95 regret':<30} {report.p95_regret:>14.1%}")
    print(f"{'Max regret':<30} {report.max_regret:>14.1%}")
    
    # Regret distribution
    print(f"\n{'Regret Bucket':<20} {'Count':>8} {'Fraction':>10}")
    print("-" * 38)
    for bucket, count in report.regret_buckets.items():
        print(f"{bucket:<20} {count:>8} {count/n:>9.1%}")
    
    # Per-benchmark breakdown
    print(f"\n{'Benchmark':<12} {'Accuracy':>10} {'Top-3':>10} {'Mean Regret':>12} {'N':>6}")
    print("-" * 50)
    for bench in sorted(report.per_benchmark):
        s = report.per_benchmark[bench]
        print(f"{bench:<12} {s['accuracy']:>9.1%} {s['top3_rate']:>9.1%} "
              f"{s['mean_regret']:>11.1%} {s['count']:>6}")
    
    # Algorithm frequency table
    all_algos = sorted(set(report.oracle_freq.keys()) | set(report.adaptive_freq.keys()))
    print(f"\n{'Algorithm':<25} {'Oracle':>8} {'Adaptive':>10} {'Match':>8}")
    print("-" * 55)
    for algo in all_algos:
        o = report.oracle_freq.get(algo, 0)
        a = report.adaptive_freq.get(algo, 0)
        m = min(o, a)
        print(f"{algo:<25} {o:>8} {a:>10} {m:>8}")
    
    # Confusion matrix (top 8 x 8)
    top_oracle = [a for a, _ in sorted(report.oracle_freq.items(), key=lambda x: -x[1])[:8]]
    top_adaptive = [a for a, _ in sorted(report.adaptive_freq.items(), key=lambda x: -x[1])[:8]]
    
    if top_oracle and top_adaptive:
        print("\nConfusion Matrix (rows=oracle, cols=adaptive):")
        # Header
        header = f"{'Oracle\\Adaptive':<20}"
        for a in top_adaptive:
            short = a[:10]
            header += f" {short:>10}"
        print(header)
        print("-" * (20 + 11 * len(top_adaptive)))
        
        for o in top_oracle:
            row = f"{o[:20]:<20}"
            for a in top_adaptive:
                count = report.confusion.get(o, {}).get(a, 0)
                if count > 0:
                    row += f" {count:>10}"
                else:
                    row += f" {'·':>10}"
            print(row)
    
    # Worst cases (highest regret)
    worst = sorted(report.entries, key=lambda e: -e.regret)[:10]
    if worst and worst[0].regret > 0:
        print("\nTop-10 Highest Regret Cases:")
        print(f"{'Graph':<25} {'Bench':<6} {'Oracle':<20} {'Adaptive':<20} {'Regret':>8}")
        print("-" * 85)
        for e in worst:
            print(f"{e.graph[:25]:<25} {e.benchmark:<6} {e.oracle_algo[:20]:<20} "
                  f"{e.adaptive_algo[:20]:<20} {e.regret:>7.1%}")
    
    print()


def save_report(report: OracleReport, output_dir: str) -> str:
    """Save the oracle report to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(output_dir, f"oracle_analysis_{ts}.json")
    
    data = {
        "timestamp": ts,
        "summary": {
            "count": len(report.entries),
            "accuracy": report.accuracy,
            "top3_rate": report.top3_rate,
            "mean_regret": report.mean_regret,
            "median_regret": report.median_regret,
            "p95_regret": report.p95_regret,
            "max_regret": report.max_regret,
        },
        "regret_buckets": report.regret_buckets,
        "per_benchmark": report.per_benchmark,
        "oracle_freq": report.oracle_freq,
        "adaptive_freq": report.adaptive_freq,
        "confusion": report.confusion,
        "entries": [
            {
                "graph": e.graph,
                "benchmark": e.benchmark,
                "oracle_algo": e.oracle_algo,
                "oracle_time": e.oracle_time,
                "adaptive_algo": e.adaptive_algo,
                "adaptive_time": e.adaptive_time,
                "regret": e.regret,
                "rank": e.rank,
                "total_algos": e.total_algos,
            }
            for e in report.entries
        ],
    }
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Report saved to: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Live experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_oracle_experiment(
    bin_dir: str,
    graph_paths: List[str],
    benchmarks: List[str],
    num_trials: int = 3,
    timeout: int = 300,
) -> Tuple[List[dict], List[dict]]:
    """
    Run AdaptiveOrder + all candidates on given graphs for oracle comparison.
    
    Returns (benchmark_records, adaptive_records).
    """
    # Candidate algorithm IDs to test
    candidate_ids = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    benchmark_records = []
    adaptive_records = []
    
    for graph_path in graph_paths:
        graph_name = os.path.splitext(os.path.basename(graph_path))[0]
        print(f"\n--- {graph_name} ---")
        
        # Check if symmetric (heuristic: use -s for undirected graphs)
        sym_flag = ""  # Let the binary figure it out
        
        # Run AdaptiveOrder (algo 14) to get selections
        for bench in benchmarks:
            binary = os.path.join(bin_dir, bench)
            if not os.path.exists(binary):
                print(f"  Binary not found: {binary}")
                continue
            
            cmd = f"{binary} -f {graph_path} {sym_flag} -o 14 -n {num_trials}"
            print(f"  Running: AdaptiveOrder × {bench}")
            try:
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=timeout
                )
                output = result.stdout + result.stderr
                
                # Parse time
                time_match = re.search(r'(?:Average|Trial) Time:\s*([\d.e+-]+)', output)
                if time_match:
                    adaptive_time = float(time_match.group(1))
                    benchmark_records.append({
                        "graph": graph_name,
                        "algorithm": "AdaptiveOrder",
                        "algorithm_id": 14,
                        "benchmark": bench,
                        "time_seconds": adaptive_time,
                        "trials": num_trials,
                        "success": True,
                    })
                
                # Parse community info (only need once per graph)
                if bench == benchmarks[0]:
                    mod_match = re.search(r'modularity[:\s]*([\d.]+)', output, re.I)
                    comm_matches = re.findall(
                        r'Community\s+(\d+):\s*(\d+)\s+nodes,\s*(\d+)\s+edges\s*->\s*(\w+)',
                        output
                    )
                    small_match = re.search(
                        r'Grouped\s+(\d+)\s+small\s+communities.*?->\s+(\w+)',
                        output
                    )
                    
                    algo_dist = Counter()
                    subcommunities = []
                    for cid, nodes, edges, algo in comm_matches:
                        algo_dist[algo] += 1
                        subcommunities.append({
                            "community_id": int(cid),
                            "nodes": int(nodes),
                            "edges": int(edges),
                            "selected_algorithm": algo,
                        })
                    if small_match:
                        algo_dist[small_match.group(2)] += int(small_match.group(1))
                    
                    adaptive_records.append({
                        "graph": graph_name,
                        "modularity": float(mod_match.group(1)) if mod_match else 0.0,
                        "algorithm_distribution": dict(algo_dist),
                        "subcommunities": subcommunities,
                        "success": True,
                    })
            except (subprocess.TimeoutExpired, Exception) as e:
                print(f"    FAILED: {e}")
        
        # Run each candidate algorithm
        for algo_id in candidate_ids:
            for bench in benchmarks:
                binary = os.path.join(bin_dir, bench)
                if not os.path.exists(binary):
                    continue
                
                cmd = f"{binary} -f {graph_path} {sym_flag} -o {algo_id} -n {num_trials}"
                algo_name = {
                    0: "ORIGINAL", 2: "SORT", 3: "HUBSORT", 4: "HUBCLUSTER",
                    5: "DBG", 6: "HUBSORTDBG", 7: "HUBCLUSTERDBG",
                    8: "RABBITORDER_csr", 9: "GORDER", 10: "CORDER", 11: "RCM",
                }.get(algo_id, f"ALGO_{algo_id}")
                
                print(f"  Running: {algo_name} × {bench}")
                try:
                    result = subprocess.run(
                        cmd, shell=True, capture_output=True, text=True, timeout=timeout
                    )
                    output = result.stdout + result.stderr
                    
                    time_match = re.search(r'(?:Average|Trial) Time:\s*([\d.e+-]+)', output)
                    if time_match:
                        benchmark_records.append({
                            "graph": graph_name,
                            "algorithm": algo_name,
                            "algorithm_id": algo_id,
                            "benchmark": bench,
                            "time_seconds": float(time_match.group(1)),
                            "trials": num_trials,
                            "success": True,
                        })
                except (subprocess.TimeoutExpired, Exception) as e:
                    print(f"    FAILED: {e}")
    
    return benchmark_records, adaptive_records


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Oracle analysis for AdaptiveOrder (Algorithm 14)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="Directory containing benchmark_*.json and adaptive_analysis_*.json"
    )
    parser.add_argument(
        "--benchmarks", nargs="+", default=None,
        help="Restrict to these benchmarks (e.g., pr bfs)"
    )
    parser.add_argument(
        "--all-algos", action="store_true",
        help="Include all algorithms in oracle, not just candidates AdaptiveOrder selects from"
    )
    parser.add_argument(
        "--run-experiment", action="store_true",
        help="Run a fresh experiment instead of analyzing existing data"
    )
    parser.add_argument(
        "--graphs", nargs="+", default=None,
        help="Graph .sg file paths for --run-experiment"
    )
    parser.add_argument(
        "--bin-dir", default="bench/bin",
        help="Directory containing benchmark binaries"
    )
    parser.add_argument(
        "--trials", type=int, default=3,
        help="Number of trials for --run-experiment"
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Output directory for report JSON"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print JSON report instead of formatted text"
    )
    
    args = parser.parse_args()
    
    if args.run_experiment:
        # Run fresh experiment
        if not args.graphs:
            # Auto-discover graphs
            graph_dir = os.path.join(args.results_dir, "graphs")
            graph_files = glob.glob(os.path.join(graph_dir, "*", "*.sg"))
            if not graph_files:
                print(f"No .sg files found in {graph_dir}/*/")
                print("Use --graphs to specify graph file paths")
                sys.exit(1)
            args.graphs = sorted(graph_files)[:5]  # Limit for sanity
            print(f"Auto-discovered {len(args.graphs)} graphs")
        
        benchmarks = args.benchmarks or ["pr", "bfs"]
        bench_data, adaptive_data = run_oracle_experiment(
            bin_dir=args.bin_dir,
            graph_paths=args.graphs,
            benchmarks=benchmarks,
            num_trials=args.trials,
        )
    else:
        # Load existing data
        bench_data = load_benchmark_data(args.results_dir)
        adaptive_data = load_adaptive_data(args.results_dir)
    
    if not bench_data:
        print("No benchmark data available. Use --run-experiment to generate data.")
        sys.exit(1)
    
    # Build lookups
    time_lookup = build_time_lookup(bench_data, args.benchmarks)
    adaptive_lookup = build_adaptive_lookup(adaptive_data)
    
    print(f"\nGraphs with benchmark data: {len(time_lookup)}")
    print(f"Graphs with adaptive data: {len(adaptive_lookup)}")
    
    # Compute oracle
    report = compute_oracle(
        time_lookup=time_lookup,
        adaptive_lookup=adaptive_lookup,
        benchmarks=args.benchmarks,
        candidate_only=not args.all_algos,
    )
    
    if args.json:
        # JSON output
        data = {
            "summary": {
                "count": len(report.entries),
                "accuracy": report.accuracy,
                "top3_rate": report.top3_rate,
                "mean_regret": report.mean_regret,
                "median_regret": report.median_regret,
                "p95_regret": report.p95_regret,
                "max_regret": report.max_regret,
            },
            "per_benchmark": report.per_benchmark,
            "regret_buckets": report.regret_buckets,
        }
        print(json.dumps(data, indent=2))
    else:
        print_report(report)
    
    # Save report
    save_report(report, args.output_dir)


if __name__ == "__main__":
    main()
