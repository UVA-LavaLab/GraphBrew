#!/usr/bin/env python3
"""
VLDB 2026 GraphBrew Paper — Figure & Table Generator.

Reads experiment results from results/vldb_paper/ and generates
publication-quality figures (PNG) and LaTeX table snippets.

Usage:
    # Generate all figures from experiment results:
    python scripts/experiments/vldb_generate_figures.py

    # Generate with sample/placeholder data (for layout preview):
    python scripts/experiments/vldb_generate_figures.py --sample-data

    # Generate specific figure:
    python scripts/experiments/vldb_generate_figures.py --fig 1 2 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from experiments.vldb_config import (
    ALL_ALGORITHMS,
    BASELINE_ALGORITHMS,
    BENCHMARKS,
    EVAL_GRAPHS,
    FIGURES_DIR,
    GRAPHBREW_VARIANTS,
    GRAPH_TYPE_GROUPS,
    CHAINED_ORDERINGS,
    ABLATION_CONFIGS,
    RESULTS_DIR,
    TABLES_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("vldb_figures")

# Paper figure directory (for direct LaTeX inclusion)
PAPER_DIR = PROJECT_ROOT / "research" / (
    "GraphBrew__Multilayered_Graph_Reordering_Techniques_for_"
    "Accelerated_Graph_Processing__VLDB_2024_"
)
PAPER_CHARTS_DIR = PAPER_DIR / "dataCharts"

# Try importing matplotlib; if not available, skip figure generation
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    log.warning("matplotlib not available; will generate LaTeX tables only")

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False


# ============================================================================
# Helpers
# ============================================================================


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: Path) -> Any:
    if not path.exists():
        log.warning(f"  Missing: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def save_latex_table(content: str, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w") as f:
        f.write(content)
    log.info(f"  Saved LaTeX table: {path}")


def copy_to_paper(src: Path, subdir: str, filename: Optional[str] = None) -> None:
    """Copy a generated figure into the paper's dataCharts directory."""
    import shutil
    dst_dir = PAPER_CHARTS_DIR / subdir
    ensure_dir(dst_dir)
    dst = dst_dir / (filename or src.name)
    shutil.copy2(src, dst)
    log.info(f"  Copied to paper: {dst.relative_to(PROJECT_ROOT)}")


def generate_sample_speedup_data() -> dict:
    """Generate plausible sample data for layout preview."""
    import random
    random.seed(42)

    data = {}
    algos = list(BASELINE_ALGORITHMS.values()) + [f"GB-{v}" for v in GRAPHBREW_VARIANTS]
    graphs = [g["short"] for g in EVAL_GRAPHS]
    benchmarks_sample = ["bfs", "pr", "sssp", "cc", "bc"]

    for bench in benchmarks_sample:
        data[bench] = {}
        for graph in graphs:
            data[bench][graph] = {}
            for algo in algos:
                base = 1.0
                if "GB" in algo:
                    base = random.uniform(1.3, 2.1)
                elif algo == "GORDER":
                    base = random.uniform(1.4, 2.2)
                elif algo == "RABBITORDER":
                    base = random.uniform(1.2, 1.8)
                elif algo in ("DBG", "HUBCLUSTERDBG"):
                    base = random.uniform(1.1, 1.5)
                elif algo in ("ORIGINAL", "RANDOM"):
                    base = random.uniform(0.8, 1.1)
                else:
                    base = random.uniform(1.0, 1.6)
                data[bench][graph][algo] = round(base, 3)

    return data


def generate_sample_overhead_data() -> dict:
    """Generate sample reorder time data."""
    import random
    random.seed(43)

    data = {}
    algos = list(BASELINE_ALGORITHMS.values()) + [f"GB-{v}" for v in GRAPHBREW_VARIANTS]
    graphs = [g["short"] for g in EVAL_GRAPHS]

    for graph in graphs:
        data[graph] = {}
        scale = next((g["edges_m"] for g in EVAL_GRAPHS if g["short"] == graph), 100)
        for algo in algos:
            if algo == "ORIGINAL":
                data[graph][algo] = 0.0
            elif algo == "GORDER":
                data[graph][algo] = round(scale * random.uniform(0.5, 2.0), 2)
            elif "GB" in algo or algo == "RABBITORDER":
                data[graph][algo] = round(scale * random.uniform(0.01, 0.1), 2)
            else:
                data[graph][algo] = round(scale * random.uniform(0.005, 0.05), 2)

    return data


# ============================================================================
# Figure 1: Cache Miss Rate vs Cache Size
# ============================================================================


def fig1_cache_performance(sample: bool = False) -> None:
    log.info("Figure 1: Cache Miss Rate vs Cache Size")
    if not HAS_MPL or not HAS_NP:
        log.warning("  Skipped (no matplotlib/numpy)")
        return

    ensure_dir(FIGURES_DIR)

    if sample:
        # Placeholder with random data
        cache_sizes_kb = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
        algos = ["ORIGINAL", "DBG", "RABBITORDER", "GORDER", "GB-leiden", "GB-hrab"]
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for ax, graph in zip(axes.flat, ["G1", "G2", "G3", "G4", "G5", "GM"]):
            for algo in algos:
                base = np.random.uniform(0.3, 0.8)
                ax.plot(cache_sizes_kb, [max(0.01, base * (1 - 0.06*i) + np.random.normal(0, 0.02))
                        for i in range(len(cache_sizes_kb))], marker="o", markersize=3, label=algo)
            ax.set_title(graph); ax.set_xscale("log", base=2); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
        axes[0, 0].legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "fig1_cache_performance.png", dpi=300); plt.close()
        log.info(f"  Saved (sample): {FIGURES_DIR / 'fig1_cache_performance.png'}")
        return

    # ---- Real data from exp1 ----
    data = load_json(RESULTS_DIR / "exp1_cache" / "cache_results.json")
    if not isinstance(data, list) or not data:
        log.warning("  Skipped (no cache data)")
        return

    # Data has single-cache-size results from sim benchmarks.  Plot miss rate
    # per graph as a grouped bar chart (algo on x-axis, miss rate on y-axis).
    from collections import defaultdict

    # Group: graph -> algo -> l3_miss_rate  (average across benchmarks)
    graph_algo_miss: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in data:
        miss = r.get("l3_miss_rate") or r.get("l3_hit_rate")
        if miss is None:
            continue
        # If we got hit_rate, convert
        if "l3_miss_rate" not in r and "l3_hit_rate" in r:
            miss = 100.0 - r["l3_hit_rate"]
        graph_algo_miss[r["graph"]][r["algorithm"]].append(miss)

    graphs = sorted(graph_algo_miss.keys())
    if not graphs:
        log.warning("  Skipped (no valid miss rate data)")
        return

    # Select representative algorithms for readability
    show_algos = ["ORIGINAL", "DBG", "RABBITORDER", "GORDER", "GB-Leiden", "GB-HRAB",
                  "GB-Rabbit", "GB-Hubcluster", "GB-TQR", "GoGraphOrder", "RCM"]
    # Map display names
    all_algos_in_data = set()
    for g in graphs:
        all_algos_in_data.update(graph_algo_miss[g].keys())
    show_algos = [a for a in show_algos if a in all_algos_in_data]
    if not show_algos:
        show_algos = sorted(all_algos_in_data)[:10]

    colors_map = {
        "ORIGINAL": "#888888", "RANDOM": "#aaaaaa", "DBG": "#1f77b4",
        "RABBITORDER": "#9467bd", "GORDER": "#d62728", "GoGraphOrder": "#17becf",
        "RCM": "#7f7f7f",
    }
    gb_colors = ["#2ca02c", "#98df8a", "#006400", "#228B22", "#32CD32",
                 "#3CB371", "#66CDAA", "#8FBC8F", "#556B2F", "#6B8E23"]

    ncols = min(3, len(graphs) + 1)
    nrows = (len(graphs) + ncols) // ncols  # +1 for GM
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    x = np.arange(len(show_algos))
    width = 0.7

    for idx, graph in enumerate(graphs):
        ax = axes[idx]
        vals = []
        for algo in show_algos:
            rates = graph_algo_miss[graph].get(algo, [])
            vals.append(np.mean(rates) if rates else 0)
        c = [colors_map.get(a, gb_colors[i % len(gb_colors)])
             for i, a in enumerate(show_algos)]
        ax.bar(x, vals, width, color=c, edgecolor="black", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([a[:10] for a in show_algos], rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("L3 Miss Rate (%)")
        ax.set_title(graph, fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    # Turn off unused axes
    for i in range(len(graphs), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    out = FIGURES_DIR / "fig1_cache_performance.png"
    plt.savefig(out, dpi=300); plt.close()
    log.info(f"  Saved: {out}")
    copy_to_paper(out, "cache", "cacheGM.png")


# ============================================================================
# Figure 2: Kernel Speedup Bar Charts
# ============================================================================


def fig2_kernel_speedup(sample: bool = False) -> None:
    log.info("Figure 2: Kernel Speedup")
    if not HAS_MPL or not HAS_NP:
        log.warning("  Skipped (no matplotlib/numpy)")
        return

    ensure_dir(FIGURES_DIR)

    data = load_json(RESULTS_DIR / "exp2_speedup" / "speedup_results.json")
    if not isinstance(data, list) or not data:
        log.warning("  No data available")
        return

    from collections import defaultdict

    # Build baseline: ORIGINAL average_time per (graph, benchmark)
    baseline: Dict[tuple, float] = {}
    for r in data:
        if r.get("algorithm") == "ORIGINAL" and r.get("average_time"):
            baseline[(r["graph"], r["benchmark"])] = r["average_time"]

    # Compute speedup per (benchmark, algorithm) — geo-mean across graphs
    bench_algo_speedups: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in data:
        algo = r.get("algorithm", "")
        if algo == "ORIGINAL":
            continue
        graph, bench = r.get("graph", ""), r.get("benchmark", "")
        avg_t = r.get("average_time")
        key = (graph, bench)
        if key in baseline and baseline[key] > 0 and avg_t and avg_t > 0:
            bench_algo_speedups[bench][algo].append(baseline[key] / avg_t)

    benchmarks_plot = [b for b in BENCHMARKS if b in bench_algo_speedups]
    if not benchmarks_plot:
        log.warning("  No benchmark data")
        return

    # Select key algorithms for readability
    key_algos = ["DBG", "RABBITORDER", "GORDER", "GB-Leiden", "GB-HRAB",
                 "GB-Rabbit", "GB-Hubcluster", "GoGraphOrder", "RCM"]
    all_algos = set()
    for b in benchmarks_plot:
        all_algos.update(bench_algo_speedups[b].keys())
    key_algos = [a for a in key_algos if a in all_algos]
    if not key_algos:
        key_algos = sorted(all_algos)[:10]

    algo_colors = {
        "DBG": "#1f77b4", "RABBITORDER": "#9467bd", "GORDER": "#d62728",
        "GoGraphOrder": "#17becf", "RCM": "#7f7f7f",
        "GB-Leiden": "#2ca02c", "GB-HRAB": "#006400", "GB-Rabbit": "#98df8a",
        "GB-Hubcluster": "#228B22", "GB-TQR": "#3CB371", "GB-Hcache": "#66CDAA",
        "GB-Streaming": "#556B2F", "GB-Rabbit:dbg": "#6B8E23",
        "GB-Rabbit:hubcluster": "#8FBC8F", "GB-Rcm": "#808000",
    }

    ncols = min(4, len(benchmarks_plot) + 1)
    nrows = (len(benchmarks_plot) + ncols) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    # Per-benchmark subplot
    for idx, bench in enumerate(benchmarks_plot):
        ax = axes[idx]
        means = []
        for algo in key_algos:
            vals = bench_algo_speedups[bench].get(algo, [])
            means.append(_geo_mean(vals) if vals else 1.0)
        colors = [algo_colors.get(a, "#aaaaaa") for a in key_algos]
        x = np.arange(len(key_algos))
        ax.bar(x, means, 0.7, color=colors, edgecolor="black", linewidth=0.3)
        ax.axhline(y=1.0, color="red", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace("GB-", "") for a in key_algos],
                           rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("Speedup (vs Original)")
        ax.set_title(bench.upper(), fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    # Aggregate (geo-mean across benchmarks)
    if len(benchmarks_plot) < len(axes):
        ax = axes[len(benchmarks_plot)]
        gm_vals = []
        for algo in key_algos:
            all_speedups = []
            for bench in benchmarks_plot:
                vals = bench_algo_speedups[bench].get(algo, [])
                if vals:
                    all_speedups.append(_geo_mean(vals))
            gm_vals.append(_geo_mean(all_speedups) if all_speedups else 1.0)
        colors = [algo_colors.get(a, "#aaaaaa") for a in key_algos]
        x = np.arange(len(key_algos))
        ax.bar(x, gm_vals, 0.7, color=colors, edgecolor="black", linewidth=0.3)
        ax.axhline(y=1.0, color="red", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace("GB-", "") for a in key_algos],
                           rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("Geo-Mean Speedup")
        ax.set_title("Aggregate (GM)", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    for i in range(min(len(benchmarks_plot) + 1, len(axes)), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    out = FIGURES_DIR / "fig2_kernel_speedup.png"
    plt.savefig(out, dpi=300); plt.close()
    log.info(f"  Saved: {out}")
    copy_to_paper(out, "speedup", "aggregateSpeedups.png")

    # Also generate per-benchmark per-graph charts
    for bench in benchmarks_plot:
        fig, ax = plt.subplots(figsize=(12, 5))
        graphs_in_bench = sorted(set(r["graph"] for r in data if r["benchmark"] == bench))
        x = np.arange(len(graphs_in_bench))
        n_algos = len(key_algos)
        width = 0.8 / n_algos
        for i, algo in enumerate(key_algos):
            vals = []
            for g in graphs_in_bench:
                bl = baseline.get((g, bench), 1.0)
                rec = [r for r in data if r["graph"] == g and r["benchmark"] == bench
                       and r["algorithm"] == algo]
                if rec and rec[0].get("average_time") and bl > 0:
                    vals.append(bl / rec[0]["average_time"])
                else:
                    vals.append(0)
            ax.bar(x + i * width - 0.4 + width/2, vals, width,
                   label=algo.replace("GB-", ""),
                   color=algo_colors.get(algo, "#aaaaaa"), edgecolor="black", linewidth=0.2)
        ax.set_xticks(x)
        short_names = {g["name"]: g["short"] for gl in [EVAL_GRAPHS] for g in gl}
        ax.set_xticklabels([short_names.get(g, g[:12]) for g in graphs_in_bench],
                           rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Speedup")
        ax.set_title(f"{bench.upper()} — Per-Graph Speedup")
        ax.axhline(y=1.0, color="red", linestyle="--", linewidth=0.5)
        ax.legend(fontsize=5, ncol=3, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        out_b = FIGURES_DIR / f"fig2_{bench}.png"
        plt.savefig(out_b, dpi=300); plt.close()
        log.info(f"  Saved: {out_b}")
        copy_to_paper(out_b, "speedup", f"{bench.upper()}.png")


# ============================================================================
# Figure 3: Reorder Overhead
# ============================================================================


def fig3_reorder_overhead(sample: bool = False) -> None:
    log.info("Figure 3: Reorder Overhead")
    if not HAS_MPL or not HAS_NP:
        return

    ensure_dir(FIGURES_DIR)

    data = load_json(RESULTS_DIR / "exp3_overhead" / "overhead_results.json")
    if not isinstance(data, list) or not data:
        log.warning("  No data available")
        return

    from collections import defaultdict

    # Group: graph -> algo -> reorder_time
    graph_algo_time: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in data:
        algo = r.get("algorithm", "")
        graph = r.get("graph", "")
        rt = r.get("reorder_time")
        if rt is not None and rt > 0 and algo != "ORIGINAL":
            graph_algo_time[graph][algo] = rt

    graphs = sorted(graph_algo_time.keys())
    if not graphs:
        log.warning("  No valid overhead data")
        return

    # Select key algorithms
    key_algos = ["DBG", "RABBITORDER", "GORDER", "GB-Leiden", "GB-HRAB",
                 "GB-Rabbit", "GB-Hubcluster", "GoGraphOrder", "RCM"]
    all_algos = set()
    for g in graphs:
        all_algos.update(graph_algo_time[g].keys())
    key_algos = [a for a in key_algos if a in all_algos]
    if not key_algos:
        key_algos = sorted(all_algos)[:10]

    algo_colors = {
        "DBG": "#1f77b4", "RABBITORDER": "#9467bd", "GORDER": "#d62728",
        "GoGraphOrder": "#17becf", "RCM": "#7f7f7f",
        "GB-Leiden": "#2ca02c", "GB-HRAB": "#006400", "GB-Rabbit": "#98df8a",
        "GB-Hubcluster": "#228B22", "GB-TQR": "#3CB371",
    }

    fig, ax = plt.subplots(figsize=(max(10, len(graphs) * 1.5), 5))
    x = np.arange(len(graphs))
    n_algos = len(key_algos)
    width = 0.8 / n_algos

    for i, algo in enumerate(key_algos):
        vals = [graph_algo_time[g].get(algo, 0) for g in graphs]
        ax.bar(x + i * width - 0.4 + width/2, vals, width,
               label=algo.replace("GB-", ""),
               color=algo_colors.get(algo, "#aaaaaa"), edgecolor="black", linewidth=0.2)

    short_names = {g["name"]: g["short"] for gl in [EVAL_GRAPHS] for g in gl}
    ax.set_xticks(x)
    ax.set_xticklabels([short_names.get(g, g[:12]) for g in graphs],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Reorder Time (s)")
    ax.set_title("Reorder Overhead")
    ax.set_yscale("log")
    ax.legend(fontsize=6, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = FIGURES_DIR / "fig3_reorder_overhead.png"
    plt.savefig(out, dpi=300); plt.close()
    log.info(f"  Saved: {out}")
    copy_to_paper(out, "speedup", "overheadReorder.png")


# ============================================================================
# Figure 5: Ablation Study
# ============================================================================


def _geo_mean(values: list[float]) -> float:
    """Compute geometric mean of positive values."""
    if not values or any(v <= 0 for v in values):
        return 0.0
    import math
    return math.exp(sum(math.log(v) for v in values) / len(values))


def fig5_ablation(sample: bool = False) -> None:
    log.info("Figure 5: Variant Ablation")

    ensure_dir(TABLES_DIR)

    # Try loading experiment data
    data = load_json(RESULTS_DIR / "exp5_ablation" / "ablation_results.json")
    config_stats: Dict[str, tuple] = {}  # name -> (speedup_str, reorder_str)

    if isinstance(data, list) and data:
        # Build baseline: Original average_time per graph
        baseline = {}
        for r in data:
            if r.get("config") == "Original" and r.get("average_time"):
                baseline[r["graph"]] = r["average_time"]

        # Compute per-config geo-mean speedup and avg reorder time
        from collections import defaultdict
        config_times: Dict[str, list] = defaultdict(list)
        config_reorder: Dict[str, list] = defaultdict(list)
        for r in data:
            cfg = r.get("config", "")
            graph = r.get("graph", "")
            avg_t = r.get("average_time")
            reorder_t = r.get("reorder_time")
            if avg_t and graph in baseline and baseline[graph] > 0:
                speedup = baseline[graph] / avg_t
                config_times[cfg].append(speedup)
            if reorder_t is not None:
                config_reorder[cfg].append(reorder_t)

        for config in ABLATION_CONFIGS:
            name = config["name"]
            speedups = config_times.get(name, [])
            reorders = config_reorder.get(name, [])
            su_str = f"{_geo_mean(speedups):.2f}$\\times$" if speedups else "\\emph{TBD}"
            ro_str = f"{sum(reorders)/len(reorders):.4f}" if reorders else "\\emph{TBD}"
            config_stats[name] = (su_str, ro_str)

    rows = []
    for config in ABLATION_CONFIGS:
        su, ro = config_stats.get(config["name"], ("\\emph{TBD}", "\\emph{TBD}"))
        rows.append(f"        {config['name']:<30s} & {su} & {ro} \\\\")

    latex = (
        "\\begin{table}[t]\n"
        "    \\centering\n"
        "    \\caption{Variant ablation: incremental layer contribution (PR, geo-mean).}\n"
        "    \\begin{tabular}{@{}lcc@{}}\n"
        "        \\toprule\n"
        "        \\textbf{Configuration} & \\textbf{Speedup} & \\textbf{Reorder (s)} \\\\\n"
        "        \\midrule\n"
        + "\n".join(rows) + "\n"
        "        \\bottomrule\n"
        "    \\end{tabular}\n"
        "    \\label{table:ablation}\n"
        "\\end{table}\n"
    )

    save_latex_table(latex, TABLES_DIR / "table_ablation.tex")


# ============================================================================
# Figure 6: Graph-Type Sensitivity
# ============================================================================


def fig6_sensitivity(sample: bool = False) -> None:
    log.info("Figure 6: Graph-Type Sensitivity")

    ensure_dir(TABLES_DIR)

    # Try loading exp2 speedup data (sensitivity is derived from it)
    data = load_json(RESULTS_DIR / "exp2_speedup" / "speedup_results.json")
    type_stats: Dict[str, tuple] = {}  # gtype -> (best_variant, speedup_str, runner_up)

    if isinstance(data, list) and data:
        from collections import defaultdict

        # Build baseline: ORIGINAL average_time per (graph, benchmark)
        baseline: Dict[tuple, float] = {}
        for r in data:
            if r.get("algorithm") == "ORIGINAL" and r.get("average_time"):
                baseline[(r["graph"], r["benchmark"])] = r["average_time"]

        # Map graph names to types
        graph_type_map = {}
        for gtype, gnames in GRAPH_TYPE_GROUPS.items():
            for gn in gnames:
                graph_type_map[gn] = gtype

        # Collect speedups per (graph_type, algorithm)
        type_algo_speedups: Dict[tuple, list] = defaultdict(list)
        for r in data:
            algo = r.get("algorithm", "")
            if algo == "ORIGINAL":
                continue
            graph = r.get("graph", "")
            bench = r.get("benchmark", "")
            avg_t = r.get("average_time")
            key = (graph, bench)
            gtype = graph_type_map.get(graph)
            if gtype and key in baseline and baseline[key] > 0 and avg_t and avg_t > 0:
                speedup = baseline[key] / avg_t
                type_algo_speedups[(gtype, algo)].append(speedup)

        # Find best and runner-up per type
        for gtype in GRAPH_TYPE_GROUPS:
            algo_means = {}
            for (gt, algo), vals in type_algo_speedups.items():
                if gt == gtype:
                    algo_means[algo] = _geo_mean(vals)
            if algo_means:
                ranked = sorted(algo_means.items(), key=lambda x: x[1], reverse=True)
                best_name, best_su = ranked[0]
                runner = ranked[1][0] if len(ranked) > 1 else "---"
                type_stats[gtype] = (best_name, f"{best_su:.2f}$\\times$", runner)

    rows = []
    for gtype in GRAPH_TYPE_GROUPS:
        best, su, runner = type_stats.get(gtype, ("\\emph{TBD}", "\\emph{TBD}", "\\emph{TBD}"))
        rows.append(f"        {gtype:<15s} & {best} & {su} & {runner} \\\\")

    latex = (
        "\\begin{table}[t]\n"
        "    \\centering\n"
        "    \\caption{Best GraphBrew variant per graph topology (kernel speedup, PR).}\n"
        "    \\begin{tabular}{@{}llcc@{}}\n"
        "        \\toprule\n"
        "        \\textbf{Graph Type} & \\textbf{Best Variant} & \\textbf{Speedup} & \\textbf{Runner-Up} \\\\\n"
        "        \\midrule\n"
        + "\n".join(rows) + "\n"
        "        \\bottomrule\n"
        "    \\end{tabular}\n"
        "    \\label{table:sensitivity}\n"
        "\\end{table}\n"
    )

    save_latex_table(latex, TABLES_DIR / "table_sensitivity.tex")


# ============================================================================
# Figure 7: Chained Ordering
# ============================================================================


def fig7_chained(sample: bool = False) -> None:
    log.info("Figure 7: Chained Ordering")

    ensure_dir(TABLES_DIR)

    # Try loading chained results
    data = load_json(RESULTS_DIR / "exp7_chained" / "chained_results.json")
    chain_stats: Dict[str, tuple] = {}  # chain_name -> (speedup_str, reorder_str)

    if isinstance(data, list) and data:
        # We need a baseline. Try exp5 ablation (has "Original" config on PR)
        # or exp2 speedup (has "ORIGINAL" on PR).
        baseline_data = load_json(RESULTS_DIR / "exp5_ablation" / "ablation_results.json")
        baseline: Dict[str, float] = {}
        if isinstance(baseline_data, list):
            for r in baseline_data:
                if r.get("config") == "Original" and r.get("average_time"):
                    baseline[r["graph"]] = r["average_time"]

        # If no ablation data, try exp2
        if not baseline:
            exp2_data = load_json(RESULTS_DIR / "exp2_speedup" / "speedup_results.json")
            if isinstance(exp2_data, list):
                for r in exp2_data:
                    if r.get("algorithm") == "ORIGINAL" and r.get("benchmark") == "pr" and r.get("average_time"):
                        baseline[r["graph"]] = r["average_time"]

        from collections import defaultdict
        chain_speedups: Dict[str, list] = defaultdict(list)
        chain_reorder: Dict[str, list] = defaultdict(list)

        for r in data:
            chain = r.get("chain", "")
            graph = r.get("graph", "")
            avg_t = r.get("average_time")
            reorder_t = r.get("reorder_time")
            if avg_t and graph in baseline and baseline[graph] > 0:
                chain_speedups[chain].append(baseline[graph] / avg_t)
            if reorder_t is not None:
                chain_reorder[chain].append(reorder_t)

        for chain_name, _ in CHAINED_ORDERINGS:
            speedups = chain_speedups.get(chain_name, [])
            reorders = chain_reorder.get(chain_name, [])
            su_str = f"{_geo_mean(speedups):.2f}$\\times$" if speedups else "\\emph{TBD}"
            ro_str = f"{sum(reorders)/len(reorders):.4f}" if reorders else "\\emph{TBD}"
            chain_stats[chain_name] = (su_str, ro_str)

    rows = []
    for chain_name, _ in CHAINED_ORDERINGS:
        su, ro = chain_stats.get(chain_name, ("\\emph{TBD}", "\\emph{TBD}"))
        rows.append(f"        {chain_name:<25s} & {su} & {ro} \\\\")

    latex = (
        "\\begin{table}[t]\n"
        "    \\centering\n"
        "    \\caption{Chained ordering vs.\\ standalone (PR, geo-mean across graphs).}\n"
        "    \\begin{tabular}{@{}lcc@{}}\n"
        "        \\toprule\n"
        "        \\textbf{Ordering} & \\textbf{Speedup} & \\textbf{Reorder (s)} \\\\\n"
        "        \\midrule\n"
        + "\n".join(rows) + "\n"
        "        \\bottomrule\n"
        "    \\end{tabular}\n"
        "    \\label{table:chained}\n"
        "\\end{table}\n"
    )

    save_latex_table(latex, TABLES_DIR / "table_chained.tex")


# ============================================================================
# Variant Summary Table
# ============================================================================


def table_variants() -> None:
    log.info("Table: Variant Summary")

    ensure_dir(TABLES_DIR)

    latex = r"""\begin{table*}[t]
    \centering
    \caption{Summary of GraphBrew reordering variants.}
    \begin{tabular}{@{}llllll@{}}
        \toprule
        \textbf{Variant} & \textbf{Community Detection} & \textbf{Intra-community} & \textbf{Inter-community} & \textbf{Complexity} & \textbf{Best For} \\
        \midrule
        Leiden (default) & Multi-pass Leiden & Connectivity BFS & Hierarchical sort & $O(n\log n + m)$ & General-purpose \\
        Rabbit & Incremental aggregation & Dendrogram DFS & Dendrogram order & $O(n\log n + m)$ & Fast reordering \\
        HubCluster & Multi-pass Leiden & Hub-first grouping & Hierarchical sort & $O(n\log n + m)$ & Power-law graphs \\
        HRAB & Multi-pass Leiden & BFS within community & RabbitOrder super-graph & $O(n\log n + m)$ & Best locality \\
        TQR & Cache-line tiling & Community-aware BFS & RabbitOrder tile graph & $O(n\log n + m)$ & Cache-aligned \\
        HCache & Multi-pass Leiden (all) & BFS within fine community & Multi-level hierarchy & $O(n\log n + m)$ & Deep hierarchies \\
        Streaming & Leiden + lazy aggregation & Connectivity BFS & Hierarchical sort & $O(n\log n + m)$ & Memory-efficient \\
        \bottomrule
    \end{tabular}
    \label{table:variants}
\end{table*}
"""

    save_latex_table(latex, TABLES_DIR / "table_variants.tex")


# ============================================================================
# Main
# ============================================================================

FIGURES = {
    1: ("Cache Performance", fig1_cache_performance),
    2: ("Kernel Speedup", fig2_kernel_speedup),
    3: ("Reorder Overhead", fig3_reorder_overhead),
    5: ("Variant Ablation", fig5_ablation),
    6: ("Graph-Type Sensitivity", fig6_sensitivity),
    7: ("Chained Ordering", fig7_chained),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VLDB 2026 GraphBrew Paper — Figure & Table Generator"
    )
    parser.add_argument("--sample-data", action="store_true",
                        help="Generate with sample/placeholder data")
    parser.add_argument("--fig", nargs="+", type=int,
                        help="Generate specific figure(s) by number")
    args = parser.parse_args()

    fig_ids = args.fig if args.fig else list(FIGURES.keys())

    log.info("GraphBrew VLDB Paper — Figure Generator")
    log.info(f"  Sample data: {args.sample_data}")
    log.info(f"  Figures: {fig_ids}")
    log.info("")

    ensure_dir(FIGURES_DIR)
    ensure_dir(TABLES_DIR)

    # Always generate tables
    table_variants()

    for fid in fig_ids:
        if fid in FIGURES:
            name, func = FIGURES[fid]
            log.info(f"\n--- Figure {fid}: {name} ---")
            func(sample=args.sample_data)
        else:
            log.warning(f"  Unknown figure: {fid}")

    log.info(f"\nFigures saved to: {FIGURES_DIR}")
    log.info(f"Tables saved to: {TABLES_DIR}")
    log.info(f"Paper charts dir: {PAPER_CHARTS_DIR}")


if __name__ == "__main__":
    main()
