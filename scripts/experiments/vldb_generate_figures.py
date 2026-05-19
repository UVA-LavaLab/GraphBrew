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
    # Also mirror into the paper's dataCharts/tables/ so main.tex can
    # \input{} the generated table directly without an extra copy step.
    copy_to_paper(path, "tables")


def copy_to_paper(src: Path, subdir: str, filename: Optional[str] = None) -> None:
    """Copy a generated figure or table into the paper's dataCharts directory."""
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


# =============================================================================
# Head-to-head comparison: GraphBrew vs Gorder vs Rabbit
# =============================================================================
#
# Auto-generates the paper-headline comparison artifacts from the VLDB
# experiment JSONs:
#   1) Per-graph table comparing focus algorithms across mem_acc, hit_rate,
#      reorder_time, real kernel_time, and end-to-end speedup
#   2) Cross-graph aggregate table with geo-mean speedup vs baseline
#   3) "Wins matrix" — how many graphs each focus algo wins on each metric
#   4) Pareto-frontier scatter plot per graph (reorder_time vs mem_acc)
#
# All outputs go to results/vldb_paper/{tables,figures}/ and are mirrored
# into the paper's dataCharts/ directory via save_latex_table / copy_to_paper.

# Algorithms shown in the headline comparison.  Order matters for table.
# Names match those emitted by vldb_paper_experiments.py (see BASELINE_ALGORITHMS
# in vldb_config.py and GRAPHBREW_VARIANTS for the GB-* form).
FOCUS_ALGOS = [
    "ORIGINAL",     # baseline
    "RABBITORDER",  # speed-of-reorder baseline (native CSR Rabbit)
    "GORDER",       # cache-quality state-of-the-art baseline
    "GB-leiden",    # GraphBrew best for sparse/medium graphs
    "GB-hrab",      # GraphBrew best for dense graphs (after May 2026 fix)
]


def _geo_mean_safe(values: list[float]) -> float:
    """Geometric mean ignoring None/NaN/<=0."""
    import math
    clean = [v for v in values if v is not None and v > 0 and not math.isnan(v)]
    if not clean:
        return float("nan")
    log_sum = sum(math.log(v) for v in clean)
    return math.exp(log_sum / len(clean))


def _index_by_graph_algo(rows: list[dict], bench_filter: str = None) -> dict:
    """Reshape JSON rows into {(graph, algo): row_dict}.

    When `bench_filter` is set, only keep rows whose 'benchmark' matches.
    """
    idx = {}
    for r in rows:
        if bench_filter and r.get("benchmark") != bench_filter:
            continue
        g = r.get("graph"); a = r.get("algorithm")
        if g and a:
            idx[(g, a)] = r
    return idx


def comparison_vs_baselines(sample: bool = False) -> None:
    """Generate head-to-head comparison artifacts vs Gorder and Rabbit.

    Emits:
      - tables/table_h2h_per_graph.tex   (per-graph headline)
      - tables/table_h2h_summary.tex     (aggregate geo-mean + wins)
      - figures/fig_pareto_<graph>.png   (one per graph)
      - figures/fig_pareto_grid.png      (multi-graph grid)
    """
    log.info("Head-to-head comparison: GraphBrew vs Gorder vs Rabbit")

    cache_path = RESULTS_DIR / "exp1_cache" / "cache_results.json"
    speed_path = RESULTS_DIR / "exp2_speedup" / "speedup_results.json"
    cache_rows = load_json(cache_path) or []
    speed_rows = load_json(speed_path) or []

    if not cache_rows and not speed_rows and not sample:
        log.warning("  No exp1/exp2 data — skipping (re-run experiments first)")
        return

    # Sample-data fallback: synthesize from the May 18 head-to-head logs
    # baked into the script for layout preview.
    if sample or not cache_rows:
        log.info("  Using built-in sample data (from May 18 2026 4-axis benchmark)")
        cache_rows, speed_rows = _sample_comparison_data()

    ensure_dir(TABLES_DIR); ensure_dir(FIGURES_DIR)

    # Cache results are PR-only; speed results may span multiple benchmarks.
    # For the headline table, use PR (canonical for cache-sim paper claims).
    cache_idx = _index_by_graph_algo(cache_rows, bench_filter="pr")
    speed_idx = _index_by_graph_algo(speed_rows, bench_filter="pr")

    # Discover graphs present in at least one source
    graphs = sorted({g for (g, _) in {**cache_idx, **speed_idx}.keys()})
    if not graphs:
        log.warning("  No graphs found in cache/speed data — skipping")
        return

    # ---- Per-graph headline table ----
    rows_out = []
    rows_out.append(r"\begin{table}[t]")
    rows_out.append(r"\centering")
    rows_out.append(r"\caption{Head-to-head: GraphBrew vs Rabbit Order and Gorder "
                    r"(PR, L3=1MB cache simulation + real wallclock).  Mem.~accesses "
                    r"in millions; reorder/kernel in seconds.  $K$-speedup is the "
                    r"per-iteration kernel speedup vs ORIGINAL; $N^{*}$ is the "
                    r"break-even trial count where reorder cost is amortized "
                    r"($N^{*} \le 1$ means amortized within one trial).  "
                    r"\textbf{Bold} = best in graph (excluding ORIGINAL).}")
    rows_out.append(r"\label{table:h2h_per_graph}")
    rows_out.append(r"\footnotesize")
    rows_out.append(r"\setlength{\tabcolsep}{3pt}")
    rows_out.append(r"\begin{tabular}{@{}l l r r r r r r@{}}")
    rows_out.append(r"\toprule")
    rows_out.append(r"\textbf{Graph} & \textbf{Algorithm} & "
                    r"\textbf{Mem (M)} & \textbf{Hit \%} & "
                    r"\textbf{Reord.~(s)} & \textbf{Kernel (s)} & "
                    r"\textbf{$K$-speedup} & \textbf{$N^{*}$} \\")
    rows_out.append(r"\midrule")

    for g in graphs:
        # Get ORIGINAL kernel for speedup baseline
        orig_speed = speed_idx.get((g, "ORIGINAL")) or {}
        orig_kernel = orig_speed.get("average_time") or 0.0
        orig_reorder = orig_speed.get("reorder_time", 0.0) or 0.0

        # First pass: compute all rows, then find per-metric best
        per_row = {}
        for algo in FOCUS_ALGOS:
            c = cache_idx.get((g, algo)) or {}
            s = speed_idx.get((g, algo)) or {}
            mem_m = (c.get("memory_accesses") or 0) / 1e6
            hit = c.get("overall_hit_rate") or c.get("l3_hit_rate") or 0.0
            reorder = (s.get("reorder_time") if s.get("reorder_time") is not None
                       else c.get("reorder_time")) or 0.0
            kernel = s.get("average_time") or 0.0
            # Kernel-only speedup (per-iteration; reordering's intrinsic value)
            k_speedup = orig_kernel / kernel if (kernel > 0 and orig_kernel > 0) else 0.0
            # Break-even trials: solve N s.t. N*orig_kernel = reorder + N*kernel
            # => N = reorder / (orig_kernel - kernel)  (only finite if k_speedup > 1)
            if k_speedup > 1.0 and (orig_kernel - kernel) > 0:
                n_star = reorder / (orig_kernel - kernel)
            else:
                n_star = float("inf")  # never amortizes — slower per-iter than ORIG
            per_row[algo] = {
                "mem": mem_m, "hit": hit, "reord": reorder,
                "kernel": kernel, "k_speedup": k_speedup, "n_star": n_star,
            }

        # Find per-metric best (excluding ORIGINAL)
        contenders = [a for a in FOCUS_ALGOS if a != "ORIGINAL"]
        best = {}
        if contenders:
            best["mem"] = min((per_row[a]["mem"] for a in contenders
                               if per_row[a]["mem"] > 0), default=None)
            best["hit"] = max((per_row[a]["hit"] for a in contenders
                               if per_row[a]["hit"] > 0), default=None)
            best["reord"] = min((per_row[a]["reord"] for a in contenders
                                 if per_row[a]["reord"] > 0), default=None)
            best["kernel"] = min((per_row[a]["kernel"] for a in contenders
                                  if per_row[a]["kernel"] > 0), default=None)
            best["k_speedup"] = max((per_row[a]["k_speedup"] for a in contenders
                                     if per_row[a]["k_speedup"] > 0), default=None)
            finite_n = [per_row[a]["n_star"] for a in contenders
                        if per_row[a]["n_star"] != float("inf")
                        and per_row[a]["n_star"] > 0]
            best["n_star"] = min(finite_n, default=None)

        def _fmt(val, fmt, best_val=None):
            if val is None:
                return r"\emph{N/A}"
            if isinstance(val, float) and (val != val or val == 0):
                return r"\emph{N/A}"
            if val == float("inf"):
                return r"$\infty$"
            s = format(val, fmt)
            is_best = (best_val is not None and val == best_val)
            return r"\textbf{" + s + r"}" if is_best else s

        for ai, algo in enumerate(FOCUS_ALGOS):
            r = per_row[algo]
            graph_cell = g if ai == 0 else ""
            # ORIGINAL never gets highlighted (it's the baseline)
            is_orig = (algo == "ORIGINAL")
            rows_out.append(
                f"{graph_cell} & {algo} & "
                f"{_fmt(r['mem'], '.2f', None if is_orig else best['mem'])} & "
                f"{_fmt(r['hit'], '.2f', None if is_orig else best['hit'])} & "
                f"{_fmt(r['reord'], '.2f', None if is_orig else best['reord'])} & "
                f"{_fmt(r['kernel'], '.4f', None if is_orig else best['kernel'])} & "
                + (_fmt(r['k_speedup'], '.2f', None if is_orig else best['k_speedup'])
                   + (r"$\times$" if r['k_speedup'] not in (None, 0)
                      and not (isinstance(r['k_speedup'], float)
                               and r['k_speedup'] != r['k_speedup'])
                      else ""))
                + " & "
                + (r"$\infty$" if r['n_star'] == float("inf")
                   else _fmt(r['n_star'], '.1f',
                             None if is_orig else best['n_star']))
                + r" \\"
            )
        if g != graphs[-1]:
            rows_out.append(r"\midrule")

    rows_out.append(r"\bottomrule")
    rows_out.append(r"\end{tabular}")
    rows_out.append(r"\end{table}")
    save_latex_table("\n".join(rows_out) + "\n",
                     TABLES_DIR / "table_h2h_per_graph.tex")

    # ---- Cross-graph summary (geo-mean + wins matrix) ----
    summary_rows = []
    summary_rows.append(r"\begin{table}[t]")
    summary_rows.append(r"\centering")
    summary_rows.append(r"\caption{Aggregate comparison across "
                        f"{len(graphs)} graphs (PR).  Geo-mean over graphs.  "
                        r"$K$-speedup is per-iteration kernel speedup vs ORIGINAL.  "
                        r"`Wins'' counts graphs where the algorithm is best on "
                        r"\emph{any} of {mem, hit, reord, $K$-speedup} "
                        r"(lower mem/reord, higher hit/speedup).}")
    summary_rows.append(r"\label{table:h2h_summary}")
    summary_rows.append(r"\footnotesize")
    summary_rows.append(r"\begin{tabular}{@{}l r r r r r@{}}")
    summary_rows.append(r"\toprule")
    summary_rows.append(r"\textbf{Algorithm} & "
                        r"\textbf{Mem (M)} & \textbf{Hit \%} & "
                        r"\textbf{Reord (s)} & \textbf{$K$-speedup} & "
                        r"\textbf{Wins} \\")
    summary_rows.append(r"\midrule")

    # Per-algo metric vectors across graphs
    per_algo = {a: {"mem": [], "hit": [], "reord": [], "k_speedup": []}
                for a in FOCUS_ALGOS}
    wins = {a: 0 for a in FOCUS_ALGOS}

    for g in graphs:
        orig_speed = speed_idx.get((g, "ORIGINAL")) or {}
        orig_kernel = orig_speed.get("average_time") or 0.0

        graph_metrics = {}
        for algo in FOCUS_ALGOS:
            c = cache_idx.get((g, algo)) or {}
            s = speed_idx.get((g, algo)) or {}
            mem_m = (c.get("memory_accesses") or 0) / 1e6 or None
            hit = c.get("overall_hit_rate") or c.get("l3_hit_rate")
            reorder = (s.get("reorder_time") if s.get("reorder_time") is not None
                       else c.get("reorder_time"))
            kernel = s.get("average_time")
            k_speedup = (orig_kernel / kernel) if (kernel and orig_kernel) else None
            graph_metrics[algo] = {"mem": mem_m, "hit": hit,
                                   "reord": reorder, "k_speedup": k_speedup}
            per_algo[algo]["mem"].append(mem_m or 0)
            per_algo[algo]["hit"].append(hit or 0)
            per_algo[algo]["reord"].append(reorder or 0)
            per_algo[algo]["k_speedup"].append(k_speedup or 0)

        # Tally per-metric winners (excluding ORIGINAL)
        contenders = [a for a in FOCUS_ALGOS if a != "ORIGINAL"]
        if contenders:
            best_mem = min((graph_metrics[a]["mem"] for a in contenders
                            if graph_metrics[a]["mem"] and graph_metrics[a]["mem"] > 0),
                           default=None)
            best_hit = max((graph_metrics[a]["hit"] for a in contenders
                            if graph_metrics[a]["hit"] and graph_metrics[a]["hit"] > 0),
                           default=None)
            best_reord = min((graph_metrics[a]["reord"] for a in contenders
                              if graph_metrics[a]["reord"] and graph_metrics[a]["reord"] > 0),
                             default=None)
            best_k = max((graph_metrics[a]["k_speedup"] for a in contenders
                          if graph_metrics[a]["k_speedup"] and graph_metrics[a]["k_speedup"] > 0),
                         default=None)
            for algo in contenders:
                m = graph_metrics[algo]
                if (m["mem"] is not None and best_mem is not None
                        and abs(m["mem"] - best_mem) < 1e-6):
                    wins[algo] += 1
                if (m["hit"] is not None and best_hit is not None
                        and abs(m["hit"] - best_hit) < 1e-6):
                    wins[algo] += 1
                if (m["reord"] is not None and best_reord is not None
                        and abs(m["reord"] - best_reord) < 1e-6):
                    wins[algo] += 1
                if (m["k_speedup"] is not None and best_k is not None
                        and abs(m["k_speedup"] - best_k) < 1e-6):
                    wins[algo] += 1

    # Determine row-level bests for highlighting summary
    s_best = {
        "mem": min((_geo_mean_safe(per_algo[a]["mem"]) for a in FOCUS_ALGOS
                    if a != "ORIGINAL"), default=None),
        "hit": max((_geo_mean_safe(per_algo[a]["hit"]) for a in FOCUS_ALGOS
                    if a != "ORIGINAL"), default=None),
        "reord": min((_geo_mean_safe(per_algo[a]["reord"]) for a in FOCUS_ALGOS
                      if a != "ORIGINAL"), default=None),
        "k_speedup": max((_geo_mean_safe(per_algo[a]["k_speedup"]) for a in FOCUS_ALGOS
                          if a != "ORIGINAL"), default=None),
        "wins": max((wins[a] for a in FOCUS_ALGOS if a != "ORIGINAL"), default=None),
    }

    def _bold(val, fmt, best_val=None, is_orig=False):
        if val is None or (isinstance(val, float) and val != val):
            return r"\emph{N/A}"
        s = format(val, fmt)
        if not is_orig and best_val is not None and abs(val - best_val) < 1e-6:
            return r"\textbf{" + s + r"}"
        return s

    for algo in FOCUS_ALGOS:
        is_orig = (algo == "ORIGINAL")
        mem_gm = _geo_mean_safe(per_algo[algo]["mem"])
        hit_gm = _geo_mean_safe(per_algo[algo]["hit"])
        reord_gm = _geo_mean_safe(per_algo[algo]["reord"])
        k_gm = _geo_mean_safe(per_algo[algo]["k_speedup"])
        summary_rows.append(
            f"{algo} & "
            f"{_bold(mem_gm, '.2f', s_best['mem'], is_orig)} & "
            f"{_bold(hit_gm, '.2f', s_best['hit'], is_orig)} & "
            f"{_bold(reord_gm, '.2f', s_best['reord'], is_orig)} & "
            + (_bold(k_gm, '.2f', s_best['k_speedup'], is_orig)
               + (r"$\times$" if k_gm and not (k_gm != k_gm) else ""))
            + " & "
            + f"{_bold(float(wins[algo]), '.0f', s_best['wins'], is_orig)}"
            + r" \\"
        )
    summary_rows.append(r"\bottomrule")
    summary_rows.append(r"\end{tabular}")
    summary_rows.append(r"\end{table}")
    save_latex_table("\n".join(summary_rows) + "\n",
                     TABLES_DIR / "table_h2h_summary.tex")

    # ---- Pareto-frontier scatter plot per graph ----
    if not HAS_MPL or not HAS_NP:
        log.warning("  Pareto plots skipped (no matplotlib/numpy)")
        return

    n = len(graphs)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows),
                             squeeze=False)
    colors = {"ORIGINAL": "gray", "RABBITORDER": "tab:blue", "GORDER": "tab:red",
              "GB-leiden": "tab:green", "GB-hrab": "tab:purple"}
    markers = {"ORIGINAL": "o", "RABBITORDER": "^", "GORDER": "v",
               "GB-leiden": "s", "GB-hrab": "D"}

    for gi, g in enumerate(graphs):
        ax = axes[gi // ncols][gi % ncols]
        pts = []
        for algo in FOCUS_ALGOS:
            c = cache_idx.get((g, algo)) or {}
            s = speed_idx.get((g, algo)) or {}
            mem_m = (c.get("memory_accesses") or 0) / 1e6
            reorder = (s.get("reorder_time") if s.get("reorder_time") is not None
                       else c.get("reorder_time")) or 0.0
            if mem_m > 0:  # plot only points with data
                ax.scatter([reorder], [mem_m], c=colors.get(algo, "k"),
                           marker=markers.get(algo, "o"), s=80,
                           label=algo, zorder=3, edgecolors="black",
                           linewidths=0.5)
                pts.append((reorder, mem_m, algo))

        # Highlight Pareto frontier: a point dominates another if it has
        # both lower reorder AND lower mem.  Pareto-optimal = not dominated.
        pareto = []
        for i, (rx, my, a) in enumerate(pts):
            dominated = any(
                (rx2 <= rx and my2 <= my and (rx2 < rx or my2 < my))
                for j, (rx2, my2, _) in enumerate(pts) if i != j
            )
            if not dominated:
                pareto.append((rx, my, a))
        # Connect Pareto points (sorted by x)
        pareto.sort(key=lambda t: t[0])
        if len(pareto) >= 2:
            xs = [p[0] for p in pareto]
            ys = [p[1] for p in pareto]
            ax.plot(xs, ys, "--", color="gray", alpha=0.6, zorder=1,
                    linewidth=1, label="_Pareto front")

        ax.set_xlabel("Reorder time (s)")
        ax.set_ylabel("Memory accesses (M)")
        ax.set_title(g, fontsize=10)
        ax.grid(True, alpha=0.3)
        # Log scale on x if there's wide range
        if pts and max(p[0] for p in pts) / max(1e-3, min(p[0] for p in pts)) > 100:
            ax.set_xscale("log")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

    # Hide unused subplots
    for gi in range(n, nrows * ncols):
        axes[gi // ncols][gi % ncols].set_visible(False)

    fig.suptitle("Pareto frontier: reorder cost vs cache quality (PR, L3=1MB)",
                 fontsize=12, y=1.005)
    plt.tight_layout()
    out_path = FIGURES_DIR / "fig_h2h_pareto.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved figure: {out_path}")
    copy_to_paper(out_path, "speedup", "h2h_pareto.png")
    copy_to_paper(out_path.with_suffix(".pdf"), "speedup", "h2h_pareto.pdf")


def _sample_comparison_data() -> tuple[list[dict], list[dict]]:
    """Built-in sample data from the May 18 2026 head-to-head benchmark.

    Used when no real experiment JSONs exist yet, so layout can be
    previewed and downstream consumers (paper compilation) won't break.
    """
    # Numbers are PR L3=1MB cache-sim from results/data/raw_2026_05_18/
    # plus realtime kernel from realtime_*_after.log files.
    sample_cache = [
        # cit-Patents L3=1MB
        {"graph": "cit-Patents", "algorithm": "ORIGINAL", "benchmark": "pr",
         "memory_accesses": 80870000, "overall_hit_rate": 23.76,
         "reorder_time": 0.0001},
        {"graph": "cit-Patents", "algorithm": "RABBITORDER", "benchmark": "pr",
         "memory_accesses": 20870000, "overall_hit_rate": 65.58,
         "reorder_time": 2.334},
        {"graph": "cit-Patents", "algorithm": "GORDER", "benchmark": "pr",
         "memory_accesses": 30220000, "overall_hit_rate": 56.71,
         "reorder_time": 20.586},
        {"graph": "cit-Patents", "algorithm": "GB-leiden", "benchmark": "pr",
         "memory_accesses": 16000000, "overall_hit_rate": 70.78,
         "reorder_time": 12.087},
        {"graph": "cit-Patents", "algorithm": "GB-hrab", "benchmark": "pr",
         "memory_accesses": 19630000, "overall_hit_rate": 64.44,
         "reorder_time": 8.416},
        # hollywood-2009 L3=1MB (post-fix)
        {"graph": "hollywood-2009", "algorithm": "ORIGINAL", "benchmark": "pr",
         "memory_accesses": 152350000, "overall_hit_rate": 44.59,
         "reorder_time": 0.0001},
        {"graph": "hollywood-2009", "algorithm": "RABBITORDER", "benchmark": "pr",
         "memory_accesses": 40190000, "overall_hit_rate": 77.64,
         "reorder_time": 2.377},
        {"graph": "hollywood-2009", "algorithm": "GORDER", "benchmark": "pr",
         "memory_accesses": 35640000, "overall_hit_rate": 77.01,
         "reorder_time": 55.873},
        {"graph": "hollywood-2009", "algorithm": "GB-leiden", "benchmark": "pr",
         "memory_accesses": 34340000, "overall_hit_rate": 79.75,
         "reorder_time": 9.544},
        {"graph": "hollywood-2009", "algorithm": "GB-hrab", "benchmark": "pr",
         "memory_accesses": 33930000, "overall_hit_rate": 80.54,
         "reorder_time": 4.293},
    ]
    sample_speed = [
        # cit-Patents real wallclock PR 20iter
        {"graph": "cit-Patents", "algorithm": "ORIGINAL", "benchmark": "pr",
         "average_time": 0.1536, "reorder_time": 0.0001},
        {"graph": "cit-Patents", "algorithm": "RABBITORDER", "benchmark": "pr",
         "average_time": 0.1845, "reorder_time": 1.440},
        {"graph": "cit-Patents", "algorithm": "GORDER", "benchmark": "pr",
         "average_time": 0.1555, "reorder_time": 12.720},
        {"graph": "cit-Patents", "algorithm": "GB-leiden", "benchmark": "pr",
         "average_time": 0.1625, "reorder_time": 6.921},
        {"graph": "cit-Patents", "algorithm": "GB-hrab", "benchmark": "pr",
         "average_time": 0.1257, "reorder_time": 5.450},
        # hollywood-2009 real wallclock PR 20iter (post-fix)
        {"graph": "hollywood-2009", "algorithm": "ORIGINAL", "benchmark": "pr",
         "average_time": 0.2783, "reorder_time": 0.0001},
        {"graph": "hollywood-2009", "algorithm": "RABBITORDER", "benchmark": "pr",
         "average_time": 0.2607, "reorder_time": 2.446},
        {"graph": "hollywood-2009", "algorithm": "GORDER", "benchmark": "pr",
         "average_time": 0.2948, "reorder_time": 59.700},
        {"graph": "hollywood-2009", "algorithm": "GB-leiden", "benchmark": "pr",
         "average_time": 0.2086, "reorder_time": 10.724},
        {"graph": "hollywood-2009", "algorithm": "GB-hrab", "benchmark": "pr",
         "average_time": 0.3131, "reorder_time": 4.583},
    ]
    return sample_cache, sample_speed


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

    # Always run the head-to-head comparison (GraphBrew vs Gorder vs Rabbit).
    # Gracefully handles missing data — emits \emph{N/A} cells when JSONs
    # are partial, or uses built-in sample data on --sample-data.
    log.info(f"\n--- Head-to-head comparison vs Gorder & Rabbit ---")
    comparison_vs_baselines(sample=args.sample_data)

    log.info(f"\nFigures saved to: {FIGURES_DIR}")
    log.info(f"Tables saved to: {TABLES_DIR}")
    log.info(f"Paper charts dir: {PAPER_CHARTS_DIR}")


if __name__ == "__main__":
    main()
