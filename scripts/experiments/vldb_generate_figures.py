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
PAPER_DIR = PROJECT_ROOT / "research" / "research" / (
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
    if not HAS_MPL:
        log.warning("  Skipped (no matplotlib)")
        return

    ensure_dir(FIGURES_DIR)
    # Placeholder: generate sample cache curves
    if sample and HAS_NP:
        cache_sizes_kb = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
        algos = ["ORIGINAL", "DBG", "RABBITORDER", "GORDER", "GB-leiden", "GB-hrab", "GB-tqr"]
        colors = ["gray", "orange", "blue", "red", "green", "darkgreen", "purple"]

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        graphs = ["road", "twitter", "gplus", "wikipedia", "webbase", "GM"]

        for ax, graph in zip(axes.flat, graphs):
            for algo, color in zip(algos, colors):
                base_miss = np.random.uniform(0.3, 0.8)
                misses = [max(0.01, base_miss * (1.0 - 0.06 * i) + np.random.normal(0, 0.02))
                          for i in range(len(cache_sizes_kb))]
                ax.plot(cache_sizes_kb, misses, marker="o", markersize=3, label=algo, color=color)
            ax.set_title(graph, fontsize=10)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Cache Size (KB)")
            ax.set_ylabel("Miss Rate")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

        axes[0, 0].legend(fontsize=6, loc="upper right")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "fig1_cache_performance.png", dpi=300)
        plt.close()
        log.info(f"  Saved: {FIGURES_DIR / 'fig1_cache_performance.png'}")
        copy_to_paper(FIGURES_DIR / "fig1_cache_performance.png", "cache", "cacheGM.png")
    else:
        log.info("  Skipped (no data or no numpy)")


# ============================================================================
# Figure 2: Kernel Speedup Bar Charts
# ============================================================================


def fig2_kernel_speedup(sample: bool = False) -> None:
    log.info("Figure 2: Kernel Speedup")
    if not HAS_MPL:
        log.warning("  Skipped (no matplotlib)")
        return

    ensure_dir(FIGURES_DIR)
    data = generate_sample_speedup_data() if sample else load_json(RESULTS_DIR / "exp2_speedup" / "speedup_results.json")
    if data is None:
        log.warning("  No data available")
        return

    if sample:
        benchmarks = list(data.keys())
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))

        for idx, (ax, bench) in enumerate(zip(axes.flat, benchmarks + ["Aggregate"])):
            if bench == "Aggregate":
                ax.set_title("Geometric Mean", fontsize=10)
                ax.text(0.5, 0.5, "TBD", transform=ax.transAxes, ha="center", fontsize=14)
                continue

            bench_data = data.get(bench, {})
            if not bench_data:
                continue

            # Average across graphs for each algorithm
            algos = list(next(iter(bench_data.values())).keys()) if bench_data else []
            means = []
            for algo in algos:
                vals = [bench_data[g].get(algo, 1.0) for g in bench_data]
                means.append(sum(vals) / len(vals) if vals else 1.0)

            colors = ["green" if "GB" in a else "gray" if a in ("ORIGINAL", "RANDOM")
                       else "steelblue" for a in algos]
            bars = ax.barh(range(len(algos)), means, color=colors, edgecolor="black", linewidth=0.5)
            ax.set_yticks(range(len(algos)))
            ax.set_yticklabels([a[:12] for a in algos], fontsize=6)
            ax.set_xlabel("Speedup (vs Original)")
            ax.set_title(bench.upper(), fontsize=10)
            ax.axvline(x=1.0, color="red", linestyle="--", linewidth=0.5)

        if len(benchmarks) < 6:
            for ax in axes.flat[len(benchmarks):]:
                ax.axis("off")

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "fig2_kernel_speedup.png", dpi=300)
        plt.close()
        log.info(f"  Saved: {FIGURES_DIR / 'fig2_kernel_speedup.png'}")
        copy_to_paper(FIGURES_DIR / "fig2_kernel_speedup.png", "speedup", "aggregateSpeedups.png")


# ============================================================================
# Figure 3: Reorder Overhead
# ============================================================================


def fig3_reorder_overhead(sample: bool = False) -> None:
    log.info("Figure 3: Reorder Overhead")
    if not HAS_MPL:
        return

    ensure_dir(FIGURES_DIR)
    data = generate_sample_overhead_data() if sample else None
    if data is None and not sample:
        log.warning("  No data available")
        return

    if sample:
        fig, ax = plt.subplots(figsize=(12, 6))
        graphs = list(data.keys())
        algos = ["GORDER", "RABBITORDER", "GB-leiden", "GB-hrab", "GB-rabbit", "GB-streaming"]
        x = range(len(graphs))
        width = 0.12

        for i, algo in enumerate(algos):
            vals = [data[g].get(algo, 0) for g in graphs]
            offset = (i - len(algos) / 2) * width
            ax.bar([xi + offset for xi in x], vals, width, label=algo)

        ax.set_xticks(x)
        ax.set_xticklabels(graphs, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Reorder Time (s)")
        ax.set_title("Reorder Overhead Comparison")
        ax.legend(fontsize=7, ncol=3)
        ax.set_yscale("log")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "fig3_reorder_overhead.png", dpi=300)
        plt.close()
        log.info(f"  Saved: {FIGURES_DIR / 'fig3_reorder_overhead.png'}")
        copy_to_paper(FIGURES_DIR / "fig3_reorder_overhead.png", "speedup", "overheadReorder.png")


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
