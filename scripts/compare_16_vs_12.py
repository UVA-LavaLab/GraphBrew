#!/usr/bin/env python3
"""
DEPRECATED: Compare Algorithm 16 (LeidenCSR) vs Algorithm 12 (GraphBrewOrder).

LeidenCSR (16) has been deprecated and merged into GraphBrewOrder (12).
This script is kept for historical reference only.

Original purpose:
  Tests whether GraphBrew (algo 12) can replicate and beat LeidenCSR (algo 16),
  potentially making algo 16 redundant. (Result: yes — algo 16 was removed.)
"""

import subprocess
import sys
import re
import json
import os
import statistics
from datetime import datetime

# ─── Configuration ─────────────────────────────────────────────
BINARY = "./bench/bin/pr"
GRAPH_SIZES = [15, 17, 19, 20]  # Uniform random graphs (-g N)
REAL_GRAPHS = []  # Will check for available real graphs
NUM_ITERATIONS = 3  # PageRank iterations per run
NUM_RUNS = 3  # Repeat each config this many times

# Configs to compare: (label, -o flag)
CONFIGS = [
    ("baseline (no reorder)", "0"),
    ("16:gveopt2 (LeidenCSR default)", "16"),
    ("16:gve (LeidenCSR GVE)", "16:gve"),
    ("16:fast (LeidenCSR fast)", "16:fast"),
    ("12:community (GraphBrew community-sort)", "12:community"),
    ("12 (GraphBrew LAYER default)", "12"),
    ("12:hrab (GraphBrew Hybrid Rabbit)", "12:hrab"),
    ("12:conn (GraphBrew Connectivity BFS)", "12:conn"),
]


def find_real_graphs(results_dir="results/graphs"):
    """Find available real graph files."""
    graphs = []
    if os.path.isdir(results_dir):
        for name in sorted(os.listdir(results_dir)):
            gdir = os.path.join(results_dir, name)
            if os.path.isdir(gdir):
                # Look for .sg or .el files
                for ext in [".sg", ".el", ".mtx"]:
                    candidates = [f for f in os.listdir(gdir) if f.endswith(ext)]
                    if candidates:
                        graphs.append(os.path.join(gdir, candidates[0]))
                        break
    return graphs[:4]  # Limit to 4 real graphs


def run_benchmark(graph_flag, order_flag, num_iters=3):
    """Run a single benchmark and extract timing info."""
    cmd = [BINARY, graph_flag[0], graph_flag[1], "-n", str(num_iters), "-o", order_flag]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            env={**os.environ, "OMP_NUM_THREADS": str(os.cpu_count())}
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "cmd": " ".join(cmd)}
    except Exception as e:
        return {"error": str(e), "cmd": " ".join(cmd)}

    # Parse output
    data = {"cmd": " ".join(cmd), "output": output}

    # Reorder time
    m = re.search(r"Reorder\s+Time:\s*([\d.]+)", output)
    if m:
        data["reorder_time"] = float(m.group(1))

    # Average iteration time (or trial time)
    m = re.search(r"Average\s+Time:\s*([\d.]+)", output)
    if m:
        data["avg_time"] = float(m.group(1))

    # Total trial time
    m = re.search(r"Trial\s+Time:\s*([\d.]+)", output)
    if m:
        data["trial_time"] = float(m.group(1))

    # Number of communities
    m = re.search(r"communities[:\s]+(\d+)", output, re.IGNORECASE)
    if m:
        data["communities"] = int(m.group(1))

    # Graph info
    m = re.search(r"(\d+)\s+Nodes", output)
    if m:
        data["nodes"] = int(m.group(1))
    m = re.search(r"(\d+)\s+Edges", output)
    if m:
        data["edges"] = int(m.group(1))

    # GraphBrew ordering info
    m = re.search(r"ordering=(\S+)", output)
    if m:
        data["ordering"] = m.group(1)

    # Variant info for algo 16
    m = re.search(r"variant=(\S+)", output)
    if m:
        data["variant"] = m.group(1)

    return data


def run_comparison(graph_label, graph_flag, configs, num_runs=3, num_iters=3):
    """Run all configs for a single graph, repeated num_runs times."""
    results = {}
    for label, order in configs:
        runs = []
        for run_i in range(num_runs):
            print(f"  [{run_i+1}/{num_runs}] {label} ... ", end="", flush=True)
            data = run_benchmark(graph_flag, order, num_iters)
            if "error" in data:
                print(f"ERROR: {data['error']}")
                runs.append(data)
            else:
                reorder = data.get("reorder_time", 0.0)
                avg = data.get("avg_time", 0.0)
                total = reorder + avg
                data["total_time"] = total
                runs.append(data)
                print(f"reorder={reorder:.4f}s  avg_pr={avg:.4f}s  total={total:.4f}s")

        # Compute statistics
        valid_runs = [r for r in runs if "total_time" in r]
        if valid_runs:
            reorder_times = [r["reorder_time"] for r in valid_runs]
            avg_times = [r["avg_time"] for r in valid_runs]
            total_times = [r["total_time"] for r in valid_runs]

            results[label] = {
                "order_flag": order,
                "runs": len(valid_runs),
                "reorder_time": {
                    "mean": statistics.mean(reorder_times),
                    "median": statistics.median(reorder_times),
                    "min": min(reorder_times),
                    "max": max(reorder_times),
                    "stdev": statistics.stdev(reorder_times) if len(reorder_times) > 1 else 0,
                },
                "pr_time": {
                    "mean": statistics.mean(avg_times),
                    "median": statistics.median(avg_times),
                    "min": min(avg_times),
                    "max": max(avg_times),
                    "stdev": statistics.stdev(avg_times) if len(avg_times) > 1 else 0,
                },
                "total_time": {
                    "mean": statistics.mean(total_times),
                    "median": statistics.median(total_times),
                    "min": min(total_times),
                    "max": max(total_times),
                    "stdev": statistics.stdev(total_times) if len(total_times) > 1 else 0,
                },
                "communities": valid_runs[-1].get("communities"),
                "nodes": valid_runs[-1].get("nodes"),
                "edges": valid_runs[-1].get("edges"),
                "ordering": valid_runs[-1].get("ordering", ""),
                "variant": valid_runs[-1].get("variant", ""),
            }
        else:
            results[label] = {"order_flag": order, "error": "all runs failed"}

    return results


def print_results_table(graph_label, results):
    """Print a formatted comparison table."""
    print(f"\n{'='*90}")
    print(f"  RESULTS: {graph_label}")
    print(f"{'='*90}")

    # Get baseline for speedup calculation
    baseline_total = None
    for label, data in results.items():
        if "baseline" in label.lower():
            baseline_total = data.get("total_time", {}).get("mean")
            break

    # Header
    print(f"{'Algorithm':<45} {'Reorder(s)':<12} {'PR(s)':<12} {'Total(s)':<12} {'Speedup':<10}")
    print(f"{'-'*45} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

    # Sort by total time
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get("total_time", {}).get("mean", float("inf"))
    )

    for label, data in sorted_results:
        if "error" in data:
            print(f"{label:<45} {'ERROR':<12}")
            continue
        rt = data["reorder_time"]["mean"]
        pt = data["pr_time"]["mean"]
        tt = data["total_time"]["mean"]
        speedup = ""
        if baseline_total and baseline_total > 0:
            sp = baseline_total / tt
            speedup = f"{sp:.2f}x"
            if "baseline" in label.lower():
                speedup = "1.00x"
        print(f"{label:<45} {rt:<12.4f} {pt:<12.4f} {tt:<12.4f} {speedup:<10}")

    # Direct 16 vs 12:community comparison
    algo16_data = None
    algo12c_data = None
    algo12_data = None
    algo12h_data = None

    for label, data in results.items():
        if "error" in data:
            continue
        if "16:gveopt2" in label:
            algo16_data = data
        if "12:community" in label:
            algo12c_data = data
        if "12 (GraphBrew LAYER" in label:
            algo12_data = data
        if "12:hrab" in label:
            algo12h_data = data

    print(f"\n--- Head-to-Head: LeidenCSR (16) vs GraphBrew equivalents ---")
    if algo16_data and algo12c_data:
        t16 = algo16_data["total_time"]["mean"]
        t12c = algo12c_data["total_time"]["mean"]
        diff_pct = ((t16 - t12c) / t16) * 100
        winner = "12:community WINS" if t12c < t16 else "16:gveopt2 WINS"
        print(f"  16:gveopt2 total = {t16:.4f}s")
        print(f"  12:community total = {t12c:.4f}s")
        print(f"  Difference: {abs(diff_pct):.1f}% → {winner}")

    if algo16_data and algo12_data:
        t16 = algo16_data["total_time"]["mean"]
        t12 = algo12_data["total_time"]["mean"]
        diff_pct = ((t16 - t12) / t16) * 100
        winner = "12 WINS" if t12 < t16 else "16:gveopt2 WINS"
        print(f"  12 (LAYER) total = {t12:.4f}s")
        print(f"  Difference: {abs(diff_pct):.1f}% → {winner}")

    if algo16_data and algo12h_data:
        t16 = algo16_data["total_time"]["mean"]
        t12h = algo12h_data["total_time"]["mean"]
        diff_pct = ((t16 - t12h) / t16) * 100
        winner = "12:hrab WINS" if t12h < t16 else "16:gveopt2 WINS"
        print(f"  12:hrab total = {t12h:.4f}s")
        print(f"  Difference: {abs(diff_pct):.1f}% → {winner}")


def main():
    print("=" * 90)
    print("  GraphBrew Performance Comparison: Algo 16 (LeidenCSR) vs Algo 12 (GraphBrew)")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Binary: {BINARY}")
    print(f"  Runs per config: {NUM_RUNS}")
    print(f"  PR iterations: {NUM_ITERATIONS}")
    print(f"  CPUs: {os.cpu_count()}")
    print("=" * 90)

    all_results = {}

    # Synthetic graphs
    for g_size in GRAPH_SIZES:
        graph_label = f"Uniform Random -g {g_size} ({2**g_size} nodes)"
        graph_flag = ("-g", str(g_size))
        print(f"\n{'─'*90}")
        print(f"  Graph: {graph_label}")
        print(f"{'─'*90}")
        results = run_comparison(graph_label, graph_flag, CONFIGS, NUM_RUNS, NUM_ITERATIONS)
        print_results_table(graph_label, results)
        all_results[graph_label] = results

    # Real graphs if available
    real_graphs = find_real_graphs()
    for gpath in real_graphs:
        gname = os.path.basename(os.path.dirname(gpath))
        graph_label = f"Real: {gname}"
        graph_flag = ("-f", gpath)
        print(f"\n{'─'*90}")
        print(f"  Graph: {graph_label}")
        print(f"{'─'*90}")
        results = run_comparison(graph_label, graph_flag, CONFIGS, NUM_RUNS, NUM_ITERATIONS)
        print_results_table(graph_label, results)
        all_results[graph_label] = results

    # ─── Final Verdict ─────────────────────────────────────────
    print(f"\n{'='*90}")
    print("  FINAL VERDICT")
    print(f"{'='*90}")

    wins_16 = 0
    wins_12 = 0
    total_comparisons = 0

    for graph_label, results in all_results.items():
        algo16_total = None
        best_12_total = float("inf")
        best_12_label = None

        for label, data in results.items():
            if "error" in data:
                continue
            tt = data["total_time"]["mean"]
            if "16:gveopt2" in label:
                algo16_total = tt
            if label.startswith("12"):
                if tt < best_12_total:
                    best_12_total = tt
                    best_12_label = label

        if algo16_total is not None and best_12_label is not None:
            total_comparisons += 1
            if best_12_total < algo16_total:
                wins_12 += 1
                print(f"  {graph_label}: ALGO 12 WINS ({best_12_label}: {best_12_total:.4f}s vs 16: {algo16_total:.4f}s)")
            else:
                wins_16 += 1
                print(f"  {graph_label}: ALGO 16 WINS (16: {algo16_total:.4f}s vs {best_12_label}: {best_12_total:.4f}s)")

    print(f"\n  Score: Algo 12 wins {wins_12}/{total_comparisons}, Algo 16 wins {wins_16}/{total_comparisons}")

    if wins_12 >= total_comparisons * 0.75:
        print("\n  RECOMMENDATION: Algo 16 can be DEPRECATED — GraphBrew (12) consistently outperforms it.")
    elif wins_12 >= total_comparisons * 0.5:
        print("\n  RECOMMENDATION: Mixed results — consider keeping both or further investigation.")
    else:
        print("\n  RECOMMENDATION: Algo 16 still has value — keep it for now.")

    # Save results
    output_file = f"results/algo16_vs_12_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("results", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_file}")


if __name__ == "__main__":
    main()
