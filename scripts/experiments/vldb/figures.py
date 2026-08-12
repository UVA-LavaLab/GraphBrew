#!/usr/bin/env python3
"""
VLDB 2026 GraphBrew Paper — Figure & Table Generator.

Reads experiment results from results/vldb_paper/ and generates
publication-quality figures (PNG) and LaTeX table snippets.

Usage:
    # Generate all figures from experiment results:
    python scripts/experiments/vldb/figures.py

    # Generate with sample/placeholder data (for layout preview):
    python scripts/experiments/vldb/figures.py --sample-data

    # Generate specific figure:
    python scripts/experiments/vldb/figures.py --fig 1 2 5
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import logging
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from experiments.vldb.config import (
    ABLATION_CONTRASTS,
    ALL_ALGORITHMS,
    BASELINE_ALGORITHMS,
    BENCHMARKS,
    CACHE_ALGORITHM_KEYS,
    CACHE_GRAPH_NAMES,
    CACHE_SIZES,
    E2E_PAPER_ALGORITHM_KEYS,
    EVAL_GRAPHS,
    FIGURES_DIR,
    GRAPHBREW_VARIANTS,
    GRAPH_TYPE_GROUPS,
    REORDER_TIMING_ANCHOR_ALGOS,
    REORDER_TIMING_REUSE_GRAPHS,
    SCALABILITY_ALGORITHM_KEYS,
    SCALABILITY_GRAPH_NAMES,
    SCALABILITY_TWITTER_ALGORITHM_KEYS,
    SCALABILITY_TWITTER_GRAPH,
    CHAINED_ORDERINGS,
    ABLATION_CONFIGS,
    RESULTS_DIR,
    TABLES_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("vldb_figures")

# Paper figure directory (for direct LaTeX inclusion).
# The canonical paper source lives at paper/ — figures are copied into
# paper/dataCharts/<subdir>/ so main.tex can \includegraphics{dataCharts/...}.
PAPER_DIR = PROJECT_ROOT / "research"
PAPER_CHARTS_DIR = PAPER_DIR / "dataCharts"
PUBLISH_TO_PAPER = os.environ.get("GRAPHBREW_PUBLISH_PAPER_FIGURES") == "1"

# Try importing matplotlib; if not available, skip figure generation
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.patches import Patch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    log.warning("matplotlib not available; will generate LaTeX tables only")

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False


def run_metric(row: dict) -> Optional[float]:
    """Return the per-run median, falling back to the recorded average."""
    trial_times = [
        float(value) for value in row.get("trial_times", [])
        if isinstance(value, (int, float)) and value > 0
    ]
    if trial_times:
        return statistics.median(trial_times)
    value = row.get("median_time", row.get("average_time"))
    return float(value) if isinstance(value, (int, float)) and value > 0 else None


def select_measurement_cohort(rows: list[dict], label: str) -> list[dict]:
    """Select one complete policy cohort instead of mixing incompatible runs."""
    cohorts: Dict[str, Dict[str, dict]] = {}
    for row in rows:
        cohort_id = row.get("cohort_id")
        if cohort_id:
            cohort = cohorts.setdefault(str(cohort_id), {})
            cell_key = str(row.get("cell_key") or row.get("policy_id") or id(row))
            current = cohort.get(cell_key)
            current_quality = (
                0 if current and current.get("overhead_timeout")
                else 1 if current and current.get("weighted_apply_timeout")
                else 2 if current else -1
            )
            row_quality = (
                0 if row.get("overhead_timeout")
                else 1 if row.get("weighted_apply_timeout")
                else 2
            )
            if (
                current is None
                or row_quality > current_quality
                or (
                    row_quality == current_quality
                    and str(row.get("measured_at", ""))
                    >= str(current.get("measured_at", ""))
                )
            ):
                cohort[cell_key] = row
    if not cohorts:
        log.warning(f"  {label}: legacy rows have no cohort_id; using all rows")
        return rows
    selected_id, selected_cells = max(
        cohorts.items(),
        key=lambda item: (
            len(item[1]),
            max(
                (str(row.get("measured_at", "")) for row in item[1].values()),
                default="",
            ),
        ),
    )
    selected = list(selected_cells.values())
    all_cells = {
        str(row.get("cell_key") or row.get("policy_id") or id(row))
        for row in rows
    }
    missing_cells = all_cells - set(selected_cells)
    if missing_cells:
        raise RuntimeError(
            f"{label}: no complete measurement cohort covers "
            f"{len(missing_cells)} recorded cell(s)"
        )
    dropped = len(rows) - len(selected)
    log.info(
        f"  {label}: cohort {selected_id} ({len(selected)} rows"
        f"{f', ignored {dropped} incompatible rows' if dropped else ''})"
    )
    return selected


# ============================================================================
# Paper style — matches paper/dataCharts/*.png (LibreOffice-Calc palette)
# ============================================================================

# Canonical palette pulled from the existing paper figures. Order is the
# usual baselines-then-GraphBrew progression: deep-blue, light-blue, cream,
# orange, green, grey for "other".
PAPER_PALETTE = {
    "blue":       "#2E75B6",   # DBG / first baseline
    "lightblue":  "#9DC3E6",   # Rabbit / second baseline
    "cream":      "#FFE699",   # Gorder / third baseline
    "orange":     "#ED7D31",   # GraphBrew headline
    "green":      "#548235",   # extra (used in cache plots)
    "grey":       "#A6A6A6",   # neutral / RCM / fallback
    "darkorange": "#C55A11",   # GB compose variants
    "darkblue":   "#1F4E79",   # SHUFFLED baseline outline
}

# Per-algorithm assignment so the same algo always gets the same colour
# across all figures in the paper. Keys cover canonical aliases, GraphBrew
# variants, and compose recipes.
ALGO_COLORS = {
    # Baselines
    "SHUFFLED":      PAPER_PALETTE["grey"],
    "RANDOM":        PAPER_PALETTE["grey"],
    "DBG":           PAPER_PALETTE["blue"],
    "HUBSORTDBG":    PAPER_PALETTE["blue"],
    "HUBCLUSTERDBG": PAPER_PALETTE["blue"],
    "HUBSORT":       PAPER_PALETTE["lightblue"],
    "HUBCLUSTER":    PAPER_PALETTE["lightblue"],
    "SORT":          PAPER_PALETTE["lightblue"],
    "RABBITORDER":   PAPER_PALETTE["lightblue"],
    "RabbitOrder (CSR)": PAPER_PALETTE["lightblue"],
    "RabbitOrder (Boost)": PAPER_PALETTE["blue"],
    "Rabbit":        PAPER_PALETTE["lightblue"],
    "GORDER":        PAPER_PALETTE["cream"],
    "Gorder":        PAPER_PALETTE["cream"],
    "GoGraphOrder":  PAPER_PALETTE["green"],
    "RCM":           PAPER_PALETTE["grey"],
    # GraphBrew headline
    "GraphBrew":     PAPER_PALETTE["orange"],
    "GB-Leiden":     PAPER_PALETTE["orange"],
    "GB-leiden":     PAPER_PALETTE["orange"],
    "GB-HRAB":       PAPER_PALETTE["darkorange"],
    "GB-hrab":       PAPER_PALETTE["darkorange"],
    "GB-HRAB-BFS":   PAPER_PALETTE["orange"],
    "GB-HRAB-RCM":   PAPER_PALETTE["darkorange"],
    "GB-Rabbit":     PAPER_PALETTE["orange"],
    "GB-rabbit":     PAPER_PALETTE["orange"],
    "GB-Hubcluster": PAPER_PALETTE["orange"],
    "GB-hubcluster": PAPER_PALETTE["orange"],
    "GB-TQR":        PAPER_PALETTE["darkorange"],
    "GB-tqr":        PAPER_PALETTE["darkorange"],
    "GB-Hcache":     PAPER_PALETTE["darkorange"],
    "GB-hcache":     PAPER_PALETTE["darkorange"],
    "GB-Streaming":  PAPER_PALETTE["darkorange"],
    "GB-streaming":  PAPER_PALETTE["darkorange"],
    "GB-Rcm":        PAPER_PALETTE["grey"],
    "GB-rcm":        PAPER_PALETTE["grey"],
    # Compose recipes — keep the orange family
    "GB-Leiden-Gorder8": PAPER_PALETTE["darkorange"],
    "GB-Leiden-HubSort": PAPER_PALETTE["orange"],
    "GB-Leiden-DegreeAsc": PAPER_PALETTE["darkorange"],
    "GB-Leiden-RCMpp": PAPER_PALETTE["darkorange"],
    "GB-Leiden-CommDegree-HubSort": PAPER_PALETTE["orange"],
    "GB-Leiden-CommDegree-DegreeAsc": PAPER_PALETTE["darkorange"],
    "GB-Leiden-CommDegree-RCMpp": PAPER_PALETTE["darkorange"],
    "GB-Rabbit-HubSort": PAPER_PALETTE["darkorange"],
    "GB-SuperRabbit-HubSort": PAPER_PALETTE["orange"],
}
ALGO_COLORS.update({
    ALL_ALGORITHMS[spec]: (
        PAPER_PALETTE["darkorange"]
        if any(
            token in spec
            for token in ("hrab", "tqr", "hcache", "gorder", "rcmpp")
        )
        else PAPER_PALETTE["orange"]
    )
    for spec in ALL_ALGORITHMS
    if spec.startswith("12:")
})


def algo_color(name: str) -> str:
    """Return the paper-palette colour for an algorithm name."""
    return ALGO_COLORS.get(name, PAPER_PALETTE["grey"])


CACHE_ALGO_STYLES = {
    "0": ("SHUFFLED", "#222222", "x", "--"),
    "5": ("DBG", "#4C78A8", "o", "-"),
    "8:csr": ("Rabbit CSR", "#72B7B2", "s", "-"),
    "8:boost": ("Rabbit Boost", "#54A24B", "^", "-"),
    "9:csr": ("Gorder", "#ECA82C", "D", "-"),
    "11": ("RCM", "#B279A2", "v", "-"),
    "12:leiden": ("Leiden", "#E45756", "*", "-"),
    "12:hrab:bfs_intra": ("HRAB-BFS", "#F58518", "P", "-."),
    "12:hrab": ("HRAB-RCM", "#FF9DA6", "X", ":"),
    "12:tqr": ("TQR", "#9D755D", "h", "-."),
}

CACHE_GRAPH_LABELS = {
    "cit-Patents": "cit-Patents",
    "com-Orkut": "com-Orkut",
    "hollywood-2009": "hollywood",
    "USA-road-d.USA": "USA-road",
}


def cache_hierarchy_lookups(row: dict) -> float:
    """Count modeled lookups across L1, L2, L3, and memory."""
    fields = ("total_accesses", "l1_misses", "l2_misses", "l3_misses")
    values: list[float] = []
    for field in fields:
        value = row.get(field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value < 0
        ):
            raise ValueError(f"Invalid cache counter {field}={value!r}")
        values.append(float(value))
    return sum(values)


def apply_paper_style() -> None:
    """Apply the paper's matplotlib rcParams (Times-like, compact, IEEE-2col).

    Idempotent; safe to call multiple times.
    """
    if not HAS_MPL:
        return
    plt.rcParams.update({
        # Fonts — readable when the figure is shrunk to a 2-col paper column.
        "font.family":       "DejaVu Sans",   # widely available, sans-serif
        "font.size":          9,
        "axes.titlesize":     9,
        "axes.labelsize":     8,
        "xtick.labelsize":    7,
        "ytick.labelsize":    7,
        "legend.fontsize":    7,
        "legend.title_fontsize": 7,
        # Axes / spines / grid
        "axes.edgecolor":    "#333333",
        "axes.linewidth":     0.6,
        "axes.grid":          True,
        "axes.axisbelow":     True,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "grid.color":        "#CCCCCC",
        "grid.linewidth":     0.4,
        # Bar / line defaults
        "patch.linewidth":    0.5,    # black border on every bar
        "patch.edgecolor":   "black",
        "lines.linewidth":    1.2,
        "lines.markersize":   4,
        # Legend
        "legend.frameon":     True,
        "legend.framealpha":  0.9,
        "legend.edgecolor":  "#888888",
        "legend.borderpad":   0.3,
        "legend.columnspacing": 1.0,
        "legend.handlelength":  1.2,
        "legend.handletextpad": 0.4,
        # Output
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
    })


# IEEE 2-column figure widths (inches)
COL_WIDTH_IN      = 3.4   # single-column
TWOCOL_WIDTH_IN   = 7.0   # full text width
ROW_HEIGHT_IN     = 1.8   # short row, tweak per plot


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


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def save_latex_table(content: str, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w") as f:
        f.write(content)
    log.info(f"  Saved LaTeX table: {path}")
    if PUBLISH_TO_PAPER:
        import shutil
        paper_tables = ensure_dir(PAPER_CHARTS_DIR / "tables")
        dst = paper_tables / path.name
        shutil.copy2(path, dst)
        log.info(f"  Copied to paper: {display_path(dst)}")


def copy_to_paper(src: Path, subdir: str, filename: Optional[str] = None) -> None:
    """Copy a generated figure into the paper's dataCharts directory."""
    if not PUBLISH_TO_PAPER:
        log.info("  Paper chart copy disabled (set GRAPHBREW_PUBLISH_PAPER_FIGURES=1)")
        return
    import shutil
    dst_dir = PAPER_CHARTS_DIR / subdir
    ensure_dir(dst_dir)
    dst = dst_dir / (filename or src.name)
    shutil.copy2(src, dst)
    log.info(f"  Copied to paper: {display_path(dst)}")


def generate_sample_speedup_data() -> dict:
    """Generate plausible sample data for layout preview."""
    import random
    random.seed(42)

    data = {}
    algos = list(BASELINE_ALGORITHMS.values()) + [
        ALL_ALGORITHMS[f"12:{variant}"]
        for variant in GRAPHBREW_VARIANTS
    ]
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
                elif algo in ("SHUFFLED", "RANDOM"):
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
    algos = list(BASELINE_ALGORITHMS.values()) + [
        ALL_ALGORITHMS[f"12:{variant}"]
        for variant in GRAPHBREW_VARIANTS
    ]
    graphs = [g["short"] for g in EVAL_GRAPHS]

    for graph in graphs:
        data[graph] = {}
        scale = next((g["edges_m"] for g in EVAL_GRAPHS if g["short"] == graph), 100)
        for algo in algos:
            if algo == "SHUFFLED":
                data[graph][algo] = 0.0
            elif algo == "GORDER":
                data[graph][algo] = round(scale * random.uniform(0.5, 2.0), 2)
            elif "GB" in algo or algo == "RABBITORDER":
                data[graph][algo] = round(scale * random.uniform(0.01, 0.1), 2)
            else:
                data[graph][algo] = round(scale * random.uniform(0.005, 0.05), 2)

    return data


# ============================================================================
# Figure 1: Hierarchy Lookup Cost vs LLC Capacity
# ============================================================================


def fig1_cache_performance(sample: bool = False) -> None:
    log.info("Figure 1: Cache Hierarchy Lookups vs LLC Capacity")
    if not HAS_MPL or not HAS_NP:
        log.warning("  Skipped (no matplotlib/numpy)")
        return

    ensure_dir(FIGURES_DIR)

    if sample:
        cache_sizes_mib = [2, 8, 22, 32, 64]
        fig, axes = plt.subplots(2, 3, figsize=(TWOCOL_WIDTH_IN, 4.0))
        for ax, graph in zip(axes.flat, ["G1", "G2", "G3", "G4", "GM"]):
            for algo_key, (label, color, marker, linestyle) in (
                CACHE_ALGO_STYLES.items()
            ):
                base = 1.0 if algo_key == "0" else np.random.uniform(1.2, 1.8)
                ax.plot(
                    cache_sizes_mib,
                    [max(1.0, base - 0.03 * i) for i in range(5)],
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    label=label,
                )
            ax.set_title(graph)
            ax.set_xscale("log", base=2)
            ax.set_ylim(0.95, 1.9)
            ax.grid(True, alpha=0.3)
        axes[1, 2].axis("off")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        axes[1, 2].legend(
            handles, labels, fontsize=5.5, ncol=2, loc="center"
        )
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "fig1_cache_performance.png", dpi=300); plt.close()
        log.info(f"  Saved (sample): {FIGURES_DIR / 'fig1_cache_performance.png'}")
        return

    # ---- Real data from exp1 ----
    apply_paper_style()
    data = load_json(RESULTS_DIR / "exp1_cache" / "cache_results.json")
    if not isinstance(data, list) or not data:
        log.warning("  Skipped (no cache data)")
        return
    data = select_measurement_cohort(data, "exp1")

    modes = {str(row.get("cache_mode")) for row in data}
    sample_rates = {
        int(row.get("cache_sample_rate", 1))
        for row in data
        if isinstance(row.get("cache_sample_rate", 1), (int, float))
    }
    if modes != {"ultrafast"} or sample_rates != {1}:
        raise RuntimeError(
            "exp1 paper figure requires one every-access ultrafast cohort"
        )
    observed_cells = [
        (
            str(row.get("graph")),
            str(row.get("algo_key")),
            int(row.get("cache_size_bytes", -1)),
        )
        for row in data
    ]
    expected_cells = {
        (graph, algo_key, cache_size)
        for graph in CACHE_GRAPH_NAMES
        for algo_key in CACHE_ALGORITHM_KEYS
        for cache_size in CACHE_SIZES
    }
    observed_cell_set = set(observed_cells)
    if (
        len(observed_cells) != len(observed_cell_set)
        or observed_cell_set != expected_cells
    ):
        message = (
            "exp1 paper figure requires the exact 4x10x5 publication matrix "
            f"({len(expected_cells)} cells); found "
            f"{len(observed_cell_set)} unique cells"
        )
        if PUBLISH_TO_PAPER:
            raise RuntimeError(message)
        log.warning(f"  {message}")

    # Group: graph -> canonical algorithm key -> capacity -> lookup samples.
    graph_algo_lookup: Dict[str, Dict[str, Dict[int, list]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    property_working_set: Dict[str, int] = {}
    accesses_by_cell: Dict[tuple[str, int], set[float]] = defaultdict(set)
    for r in data:
        cache_size = r.get("cache_size_bytes")
        if not isinstance(cache_size, (int, float)) or cache_size <= 0:
            raise ValueError(f"Invalid cache size: {cache_size!r}")
        graph = r["graph"]
        algo_key = str(r["algo_key"])
        graph_algo_lookup[graph][algo_key][int(cache_size)].append(
            cache_hierarchy_lookups(r)
        )
        accesses = r.get("total_accesses")
        if not isinstance(accesses, (int, float)) or accesses < 0:
            raise ValueError(f"Invalid total_accesses: {accesses!r}")
        accesses_by_cell[(graph, int(cache_size))].add(float(accesses))
        ws = r.get("property_working_set_bytes")
        if isinstance(ws, (int, float)) and ws > 0:
            property_working_set[graph] = int(ws)

    mismatched_work = [
        cell for cell, values in accesses_by_cell.items() if len(values) != 1
    ]
    if mismatched_work:
        raise RuntimeError(
            "exp1 cache rows do not execute identical fixed work for "
            f"{len(mismatched_work)} graph/capacity cell(s)"
        )

    preferred_graph_order = [
        "cit-Patents",
        "com-Orkut",
        "hollywood-2009",
        "USA-road-d.USA",
    ]
    graphs = [
        graph for graph in preferred_graph_order
        if graph in graph_algo_lookup
    ]
    graphs.extend(sorted(set(graph_algo_lookup) - set(graphs)))
    if not graphs:
        log.warning("  Skipped (no valid cache data)")
        return
    all_algos_in_data: set[str] = set()
    for g in graphs:
        all_algos_in_data.update(graph_algo_lookup[g].keys())
    unknown_algos = all_algos_in_data - set(CACHE_ALGO_STYLES)
    if unknown_algos:
        raise RuntimeError(
            "exp1 figure has no explicit visual style for: "
            + ", ".join(sorted(unknown_algos))
        )
    show_algos = [
        algo_key for algo_key in CACHE_ALGO_STYLES
        if algo_key in all_algos_in_data
    ]
    if "0" not in show_algos:
        raise RuntimeError("exp1 figure requires the SHUFFLED baseline")

    panels = [*graphs, "GM"]
    ncols = min(3, max(1, len(panels)))
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(TWOCOL_WIDTH_IN,
                                      ROW_HEIGHT_IN * nrows + 0.5),
                             sharey=True)
    axes = np.array(axes).flatten()

    all_sizes = sorted({
        size
        for graph in graphs
        for algo in graph_algo_lookup[graph].values()
        for size in algo
    })
    size_mib = [size / 1024**2 for size in all_sizes]
    for idx, graph in enumerate(panels):
        ax = axes[idx]
        for algo_key in show_algos:
            xs: list[float] = []
            ys: list[float] = []
            for size in all_sizes:
                if graph == "GM":
                    values: list[float] = []
                    for source_graph in graphs:
                        baseline = graph_algo_lookup[source_graph].get(
                            "0", {}
                        ).get(size, [])
                        samples = graph_algo_lookup[source_graph].get(
                            algo_key, {}
                        ).get(size, [])
                        if baseline and samples:
                            values.append(
                                float(np.mean(baseline))
                                / float(np.mean(samples))
                            )
                else:
                    baseline = graph_algo_lookup[graph].get("0", {}).get(
                        size, []
                    )
                    samples = graph_algo_lookup[graph].get(algo_key, {}).get(
                        size, []
                    )
                    values = (
                        [float(np.mean(baseline)) / float(np.mean(samples))]
                        if baseline and samples else []
                    )
                if values:
                    if graph == "GM" and len(values) != len(graphs):
                        raise RuntimeError(
                            "exp1 geometric mean requires a common graph set"
                        )
                    xs.append(size / 1024**2)
                    ys.append(
                        _geo_mean([float(value) for value in values])
                        if graph == "GM" else float(np.mean(values))
                    )
            if xs:
                label, color, marker, linestyle = CACHE_ALGO_STYLES[algo_key]
                ax.plot(
                    xs,
                    ys,
                    marker=marker,
                    markersize=3.0,
                    linewidth=1.0,
                    linestyle=linestyle,
                    label=label,
                    color=color,
                )
        if graph != "GM" and graph in property_working_set:
            ws_mib = property_working_set[graph] / 1024**2
            if size_mib and size_mib[0] <= ws_mib <= size_mib[-1]:
                ax.axvline(
                    ws_mib, color="#666666", linestyle=":", linewidth=0.8,
                )
        ax.set_xscale("log", base=2)
        ax.set_xticks(size_mib)
        ax.set_xticklabels([f"{size:g}" for size in size_mib])
        ax.axhline(1.0, color="#555555", linewidth=0.6, linestyle="--")
        if idx % ncols == 0:
            ax.set_ylabel("Lookup reduction vs SHUFFLED", fontsize=8)
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Modeled LLC capacity (MiB)", fontsize=8)
        if graph == "GM":
            title = "Geometric mean"
        else:
            title = CACHE_GRAPH_LABELS.get(graph, graph)
            if graph in property_working_set:
                title += (
                    f" (property WS "
                    f"{property_working_set[graph] / 1024**2:.1f} MiB)"
                )
        ax.set_title(title, fontsize=7.5, pad=2)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=6, pad=1)

    from matplotlib.lines import Line2D

    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D(
        [0], [0], color="#666666", linestyle=":", linewidth=0.8,
    ))
    labels.append("PR property WS")
    if len(panels) < len(axes):
        legend_axis = axes[len(panels)]
        legend_axis.axis("off")
        legend_axis.legend(
            handles,
            labels,
            fontsize=5.5,
            ncol=2,
            frameon=True,
            loc="center",
        )
        for i in range(len(panels) + 1, len(axes)):
            axes[i].axis("off")
    else:
        fig.legend(
            handles,
            labels,
            fontsize=5.5,
            ncol=5,
            frameon=True,
            loc="lower center",
        )

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.6)
    out = FIGURES_DIR / "fig1_cache_performance.png"
    plt.savefig(out); plt.close()
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
    apply_paper_style()

    data = load_json(RESULTS_DIR / "exp2_speedup" / "speedup_results.json")
    if not isinstance(data, list) or not data:
        log.warning("  No data available")
        return
    data = select_measurement_cohort(data, "exp2")

    from collections import defaultdict

    # Build SHUFFLED baseline average_time per (graph, benchmark)
    baseline: Dict[tuple, float] = {}
    for r in data:
        metric = run_metric(r)
        if r.get("algorithm") == "SHUFFLED" and metric:
            baseline[(r["graph"], r["benchmark"])] = metric

    # Compute speedup per (benchmark, algorithm) — geo-mean across graphs
    bench_algo_speedups: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in data:
        algo = r.get("algorithm", "")
        if algo == "SHUFFLED":
            continue
        graph, bench = r.get("graph", ""), r.get("benchmark", "")
        avg_t = run_metric(r)
        key = (graph, bench)
        if key in baseline and baseline[key] > 0 and avg_t and avg_t > 0:
            bench_algo_speedups[bench][algo].append(baseline[key] / avg_t)

    benchmarks_plot = [b for b in BENCHMARKS if b in bench_algo_speedups]
    if not benchmarks_plot:
        log.warning("  No benchmark data")
        return

    # Select key algorithms for readability
    key_algos = [
        "DBG", "RabbitOrder (CSR)", "RabbitOrder (Boost)", "GORDER",
        ALL_ALGORITHMS["12:leiden"],
        ALL_ALGORITHMS["12:hrab:bfs_intra"],
        ALL_ALGORITHMS["12:hrab"],
        ALL_ALGORITHMS["12:rabbit"],
        ALL_ALGORITHMS["12:hubcluster"],
        "GoGraphOrder", "RCM",
    ]
    all_algos = set()
    for b in benchmarks_plot:
        all_algos.update(bench_algo_speedups[b].keys())
    key_algos = [a for a in key_algos if a in all_algos]
    if not key_algos:
        key_algos = sorted(all_algos)[:10]

    algo_colors = {a: algo_color(a) for a in key_algos}

    # Compact layout for paper inclusion: per-benchmark stacked vertically,
    # full text width.  One axes per benchmark + one aggregate.
    n_panels = len(benchmarks_plot) + 1
    ncols = min(4, n_panels)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(TWOCOL_WIDTH_IN,
                                      ROW_HEIGHT_IN * nrows + 0.4),
                             sharey=False)
    axes = np.array(axes).flatten()

    # Per-benchmark subplot
    for idx, bench in enumerate(benchmarks_plot):
        ax = axes[idx]
        means = []
        for algo in key_algos:
            vals = bench_algo_speedups[bench].get(algo, [])
            means.append(_geo_mean(vals) if vals else 1.0)
        colors = [algo_colors[a] for a in key_algos]
        x = np.arange(len(key_algos))
        ax.bar(x, means, 0.78, color=colors,
               edgecolor="black", linewidth=0.5)
        ax.axhline(y=1.0, color="#666666", linestyle="--", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace("GB-", "") for a in key_algos],
                           rotation=40, ha="right", fontsize=6)
        if idx % ncols == 0:
            ax.set_ylabel("Speedup", fontsize=8)
        ax.set_title(bench.upper(), fontsize=8, pad=2)
        ax.tick_params(axis="y", pad=1)
        ax.margins(x=0.02)

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
        colors = [algo_colors[a] for a in key_algos]
        x = np.arange(len(key_algos))
        ax.bar(x, gm_vals, 0.78, color=colors,
               edgecolor="black", linewidth=0.5)
        ax.axhline(y=1.0, color="#666666", linestyle="--", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace("GB-", "") for a in key_algos],
                           rotation=40, ha="right", fontsize=6)
        ax.set_title("GM", fontsize=8, pad=2)
        ax.tick_params(axis="y", pad=1)
        ax.margins(x=0.02)

    for i in range(min(len(benchmarks_plot) + 1, len(axes)), len(axes)):
        axes[i].axis("off")

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.6)
    out = FIGURES_DIR / "fig2_kernel_speedup.png"
    plt.savefig(out, dpi=300); plt.close()
    log.info(f"  Saved: {out}")
    copy_to_paper(out, "speedup", "aggregateSpeedups.png")

    # Also generate per-benchmark per-graph charts (one PNG each, 2-col wide)
    for bench in benchmarks_plot:
        fig, ax = plt.subplots(figsize=(TWOCOL_WIDTH_IN, 2.2))
        graphs_in_bench = sorted(set(r["graph"] for r in data if r["benchmark"] == bench))
        x = np.arange(len(graphs_in_bench))
        n_algos = len(key_algos)
        width = 0.84 / max(n_algos, 1)
        for i, algo in enumerate(key_algos):
            vals = []
            for g in graphs_in_bench:
                bl = baseline.get((g, bench), 1.0)
                rec = [r for r in data if r["graph"] == g and r["benchmark"] == bench
                       and r["algorithm"] == algo]
                metric = run_metric(rec[0]) if rec else None
                if metric and bl > 0:
                    vals.append(bl / metric)
                else:
                    vals.append(0)
            ax.bar(x + i * width - 0.42 + width/2, vals, width,
                   label=algo.replace("GB-", ""),
                   color=algo_colors[algo], edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        short_names = {g["name"]: g["short"] for gl in [EVAL_GRAPHS] for g in gl}
        ax.set_xticklabels([short_names.get(g, g[:12]) for g in graphs_in_bench],
                           rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("Speedup", fontsize=8)
        ax.set_title(f"{bench.upper()} — per-graph", fontsize=8, pad=2)
        ax.axhline(y=1.0, color="#666666", linestyle="--", linewidth=0.5)
        ax.legend(fontsize=6, ncol=min(len(key_algos), 5),
                  loc="upper center", bbox_to_anchor=(0.5, 1.22),
                  frameon=True)
        plt.tight_layout(pad=0.3)
        out_b = FIGURES_DIR / f"fig2_{bench}.png"
        plt.savefig(out_b); plt.close()
        log.info(f"  Saved: {out_b}")
        paper_name = {
            "bfs": "BFS.png",
            "pr": "PR.png",
            "pr_spmv": "SpMV.png",
            "sssp": "SSSP.png",
            "cc": "CC.png",
            "cc_sv": "CC_SV.png",
            "bc": "BC.png",
        }[bench]
        copy_to_paper(out_b, "speedup", paper_name)

    controlled_benchmarks = {"pr", "pr_spmv", "sssp", "bc"}
    controlled_by_algo_graph: Dict[
        str, Dict[str, list[float]]
    ] = defaultdict(lambda: defaultdict(list))
    all_by_algo: Dict[str, list[float]] = defaultdict(list)
    for row in data:
        algorithm = str(row.get("algorithm", ""))
        if algorithm == "SHUFFLED":
            continue
        graph = str(row.get("graph", ""))
        benchmark = str(row.get("benchmark", ""))
        metric = run_metric(row)
        shuffled = baseline.get((graph, benchmark))
        if not metric or not shuffled:
            continue
        speedup = shuffled / metric
        all_by_algo[algorithm].append(speedup)
        if benchmark in controlled_benchmarks:
            controlled_by_algo_graph[algorithm][graph].append(speedup)

    controlled_stats: dict[
        str, tuple[float, float, float, int, float]
    ] = {}
    controlled_per_graph: dict[str, list[float]] = {}
    for algorithm, graph_values in controlled_by_algo_graph.items():
        if set(graph_values) != {graph["name"] for graph in EVAL_GRAPHS}:
            raise RuntimeError(
                f"Incomplete controlled-work kernel cohort for {algorithm}"
            )
        per_graph = [
            _geo_mean(graph_values[graph["name"]])
            for graph in EVAL_GRAPHS
        ]
        if any(len(values) != len(controlled_benchmarks)
               for values in graph_values.values()):
            raise RuntimeError(
                f"Incomplete controlled-work benchmark set for {algorithm}"
            )
        controlled_per_graph[algorithm] = per_graph
        ci_low, ci_high = _bootstrap_geo_ci(per_graph)
        controlled_stats[algorithm] = (
            _geo_mean(per_graph),
            ci_low,
            ci_high,
            sum(value > 1.0 for value in per_graph),
            _geo_mean(all_by_algo[algorithm]),
        )

    ranked_algorithms = sorted(
        controlled_stats,
        key=lambda algorithm: controlled_stats[algorithm][0],
        reverse=True,
    )
    selected_algorithms = ranked_algorithms[:5]
    for algorithm in (
        "RabbitOrder (CSR)",
        "RabbitOrder (Boost)",
        "GORDER",
        "DBG",
    ):
        if algorithm in controlled_stats and algorithm not in selected_algorithms:
            selected_algorithms.append(algorithm)
    selected_algorithms.sort(
        key=lambda algorithm: controlled_stats[algorithm][0],
        reverse=True,
    )

    summary_rows = []
    leader = ranked_algorithms[0]
    leader_per_graph = controlled_per_graph[leader]
    comparison_count = len(selected_algorithms) - 1
    for algorithm in selected_algorithms:
        point, ci_low, ci_high, graph_wins, all_kernel = (
            controlled_stats[algorithm]
        )
        escaped_algorithm = algorithm.replace("_", r"\_")
        if algorithm == leader:
            paired_text = "---"
            adjusted_text = "---"
        else:
            paired_values = [
                leader_value / method_value
                for leader_value, method_value in zip(
                    leader_per_graph,
                    controlled_per_graph[algorithm],
                )
            ]
            paired_low, paired_high = _bootstrap_geo_ci(paired_values)
            paired_text = (
                f"{_geo_mean(paired_values):.3f}$\\times$ "
                f"[{paired_low:.3f}, {paired_high:.3f}]"
            )
            adjusted_low, adjusted_high = _bootstrap_geo_ci(
                paired_values,
                alpha=0.05 / comparison_count,
            )
            adjusted_text = (
                f"[{adjusted_low:.3f}, {adjusted_high:.3f}]"
            )
        summary_rows.append(
            f"        {escaped_algorithm} & {point:.3f}$\\times$ "
            f"& [{ci_low:.3f}, {ci_high:.3f}] "
            f"& {graph_wins}/11 & {paired_text} "
            f"& {adjusted_text} "
            f"& {all_kernel:.3f}$\\times$ \\\\"
        )
    kernel_latex = (
        "\\begin{table*}[t]\n"
        "    \\centering\n"
        "    \\caption{Kernel-only speedup over SHUFFLED. Controlled-work "
        "point estimates aggregate fixed-work PR/PR-SpMV, fixed-source SSSP, "
        "and BC. Marginal intervals use a graph-level percentile bootstrap "
        "(20,000 resamples, seed 7) over each graph's four-kernel geometric "
        "mean in canonical evaluation-graph order. Leiden/method reports the "
        "paired ratio and paired 95\\% interval under the same resamples. "
        "Bonferroni CI uses $\\alpha=0.05/8$ across the eight printed paired "
        "comparisons. "
        "The all-kernel GM is descriptive and additionally includes BFS and "
        "CC/CC-SV, whose traversal or executed work can depend on ordering.}\n"
        "    \\resizebox{\\textwidth}{!}{%\n"
        "    \\begin{tabular}{@{}lcccccc@{}}\n"
        "        \\toprule\n"
        "        \\textbf{Algorithm} & \\textbf{Controlled GM} & "
        "\\textbf{Marginal 95\\% CI} & \\textbf{Graphs $>1$} & "
        "\\textbf{Leiden/method paired 95\\% CI} & "
        "\\textbf{Bonferroni CI} & "
        "\\textbf{All-kernel GM} \\\\\n"
        "        \\midrule\n"
        + "\n".join(summary_rows)
        + "\n        \\bottomrule\n"
        "    \\end{tabular}}\n"
        "    \\label{table:kernel-speedup}\n"
        "\\end{table*}\n"
    )
    save_latex_table(
        kernel_latex,
        TABLES_DIR / "table_kernel_speedup.tex",
    )


# ============================================================================
# Figure 3: Reorder Overhead
# ============================================================================


def _complete_reorder_calibration_factors(
    rows: list[dict],
) -> tuple[dict[str, float], float]:
    ratios_by_algo: Dict[str, list[float]] = defaultdict(list)
    all_ratios: list[float] = []
    for row in rows:
        algo_key = str(row.get("algo_id"))
        if (
            str(row.get("graph")) in REORDER_TIMING_REUSE_GRAPHS
            and algo_key not in REORDER_TIMING_ANCHOR_ALGOS
        ):
            continue
        ratio = row.get("complete_reorder_calibration_ratio")
        if isinstance(ratio, (int, float)) and ratio > 0:
            ratios_by_algo[algo_key].append(float(ratio))
            all_ratios.append(float(ratio))
    factors = {
        algo_key: statistics.median(values)
        for algo_key, values in ratios_by_algo.items()
    }
    return factors, statistics.median(all_ratios) if all_ratios else 1.0


def _calibrated_complete_reorder_time(
    row: dict,
    factors: dict[str, float],
    global_factor: float,
) -> Optional[float]:
    value = row.get("reorder_time")
    if not isinstance(value, (int, float)) or value < 0:
        return None
    if row.get("timing_source") != "stage02-sidecar":
        return float(value)
    factor = factors.get(str(row.get("algo_id")), global_factor)
    return float(value) * factor


def fig3_reorder_overhead(sample: bool = False) -> None:
    log.info("Figure 3: Reorder Overhead")
    if not HAS_MPL or not HAS_NP:
        return

    ensure_dir(FIGURES_DIR)

    data = load_json(RESULTS_DIR / "exp3_overhead" / "overhead_results.json")
    if not isinstance(data, list) or not data:
        log.warning("  No data available")
        return
    calibration_factors, global_calibration = (
        _complete_reorder_calibration_factors(data)
    )
    data = select_measurement_cohort(data, "exp3")

    # Group: graph -> algo -> calibrated complete reorder time.
    graph_algo_time: Dict[str, Dict[str, float]] = defaultdict(dict)
    censored: Dict[tuple[str, str], float] = {}
    reused: set[tuple[str, str]] = set()
    for r in data:
        algo = r.get("algorithm", "")
        graph = r.get("graph", "")
        rt = _calibrated_complete_reorder_time(
            r, calibration_factors, global_calibration,
        )
        if rt is not None and rt > 0 and algo != "SHUFFLED":
            if r.get("timing_source") == "stage02-sidecar":
                reused.add((graph, algo))
            graph_algo_time[graph][algo] = float(rt)
        elif r.get("overhead_timeout") is True:
            timeout_value = r.get("timeout_seconds")
            if isinstance(timeout_value, (int, float)) and timeout_value > 0:
                censored[(graph, algo)] = float(timeout_value)

    graphs = sorted(
        set(graph_algo_time)
        | {graph for graph, _algo in censored}
    )
    if not graphs:
        log.warning("  No valid overhead data")
        return

    # Select key algorithms
    key_algos = [
        "DBG", "RabbitOrder (CSR)", "RabbitOrder (Boost)", "GORDER",
        ALL_ALGORITHMS["12:leiden"],
        ALL_ALGORITHMS["12:hrab:bfs_intra"],
        ALL_ALGORITHMS["12:hrab"],
        ALL_ALGORITHMS["12:rabbit"],
        ALL_ALGORITHMS["12:hubcluster"],
        "GoGraphOrder", "RCM",
    ]
    all_algos = set()
    for g in graphs:
        all_algos.update(graph_algo_time[g].keys())
    all_algos.update(algo for _graph, algo in censored)
    key_algos = [a for a in key_algos if a in all_algos]
    if not key_algos:
        key_algos = sorted(all_algos)[:10]

    algo_colors = {algo: algo_color(algo) for algo in key_algos}

    fig, ax = plt.subplots(figsize=(max(10, len(graphs) * 1.5), 5))
    x = np.arange(len(graphs))
    n_algos = len(key_algos)
    width = 0.8 / n_algos

    for i, algo in enumerate(key_algos):
        vals = [
            graph_algo_time[g].get(
                algo,
                censored.get((g, algo), np.nan),
            )
            for g in graphs
        ]
        bars = ax.bar(
            x + i * width - 0.4 + width/2,
            vals,
            width,
            label=algo.replace("GB-", ""),
            color=algo_colors.get(algo, "#aaaaaa"),
            edgecolor="black",
            linewidth=0.2,
        )
        for graph, bar in zip(graphs, bars):
            if (graph, algo) in censored:
                bar.set_hatch("//")
                bar.set_fill(False)
            elif (graph, algo) in reused:
                bar.set_hatch("..")

    short_names = {g["name"]: g["short"] for gl in [EVAL_GRAPHS] for g in gl}
    ax.set_xticks(x)
    ax.set_xticklabels([short_names.get(g, g[:12]) for g in graphs],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Complete Reorder Time (s)")
    ax.set_title("Calibrated Complete Reorder Overhead")
    ax.set_yscale("log")
    handles, labels = ax.get_legend_handles_labels()
    if censored:
        handles.append(Patch(
            facecolor="none",
            edgecolor="black",
            hatch="//",
            label="Timed out (bar = timeout bound)",
        ))
    if reused:
        handles.append(Patch(
            facecolor="white",
            edgecolor="black",
            hatch="..",
            label="Stage-02 timing (calibrated)",
        ))
    ax.legend(handles=handles, fontsize=6, ncol=3, loc="upper left")
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


def _bootstrap_geo_ci(
    values: list[float],
    *,
    resamples: int = 20000,
    seed: int = 7,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    import random
    rng = random.Random(seed)
    samples = sorted(
        _geo_mean([
            values[rng.randrange(len(values))]
            for _ in values
        ])
        for _ in range(resamples)
    )
    tail = alpha / 2
    return (
        samples[int(tail * (resamples - 1))],
        samples[int((1 - tail) * (resamples - 1))],
    )


def _format_signed_tenth(value: float) -> str:
    return "0.0" if abs(value) < 0.05 else f"{value:+.1f}"


def fig5_ablation(sample: bool = False) -> None:
    log.info("Figure 5: Controlled Ablation")

    ensure_dir(TABLES_DIR)

    # Try loading experiment data
    data = load_json(RESULTS_DIR / "exp5_ablation" / "ablation_results.json")
    config_stats: Dict[str, tuple[Optional[float], Optional[float]]] = {}
    overhead_data = load_json(
        RESULTS_DIR / "exp3_overhead" / "overhead_results.json"
    )
    overhead_by_cell: Dict[tuple[str, str], float] = {}
    if isinstance(overhead_data, list) and overhead_data:
        calibration_factors, global_calibration = (
            _complete_reorder_calibration_factors(overhead_data)
        )
        overhead_data = select_measurement_cohort(
            overhead_data,
            "exp3-for-exp5",
        )
        for row in overhead_data:
            value = _calibrated_complete_reorder_time(
                row, calibration_factors, global_calibration,
            )
            if (
                isinstance(value, (int, float))
                and value >= 0
                and not row.get("overhead_timeout")
            ):
                overhead_by_cell[
                    (str(row.get("graph")), str(row.get("algo_id")))
                ] = float(value)

    if isinstance(data, list) and data:
        data = select_measurement_cohort(data, "exp5")
        measurements: Dict[tuple[str, str], tuple[float, Optional[float]]] = {}
        for r in data:
            cfg = r.get("config", "")
            graph = r.get("graph", "")
            avg_t = run_metric(r)
            reorder_t = (
                0.0
                if str(r.get("algo")) == "0"
                else overhead_by_cell.get(
                    (str(graph), str(r.get("algo"))),
                )
            )
            if avg_t:
                measurements[(graph, cfg)] = (
                    avg_t,
                    float(reorder_t) if reorder_t is not None else None,
                )
        config_stats["__measurements__"] = measurements

    rows = []
    names = {
        config["algo"]: config["name"]
        for config in ABLATION_CONFIGS
    }
    contrasts = [
        {
            "name": "Overall anchor",
            "base": "0",
            "variant": (
                "12:rabbit:compose:"
                "sg_none:comm_identity:intra_hubsort"
            ),
            "contrast_type": "anchor",
        },
        *ABLATION_CONTRASTS,
    ]
    for contrast in contrasts:
        base_name = names[contrast["base"]]
        variant_name = names[contrast["variant"]]
        measurements = config_stats.get("__measurements__", {})
        common_graphs = sorted({
            graph
            for graph, config_name in measurements
            if config_name == base_name
            and (graph, variant_name) in measurements
        })
        kernel_ratios = [
            measurements[(graph, base_name)][0]
            / measurements[(graph, variant_name)][0]
            for graph in common_graphs
        ]
        reorder_deltas = [
            measurements[(graph, variant_name)][1]
            - measurements[(graph, base_name)][1]
            for graph in common_graphs
            if measurements[(graph, variant_name)][1] is not None
            and measurements[(graph, base_name)][1] is not None
        ]
        reorder_graph_count = len(reorder_deltas)
        ci_low, ci_high = _bootstrap_geo_ci(kernel_ratios)
        kernel_ratio = (
            f"{_geo_mean(kernel_ratios):.3f}$\\times$"
            if kernel_ratios else "\\emph{TBD}"
        )
        kernel_ci = (
            f"[{ci_low:.3f}, {ci_high:.3f}]"
            if kernel_ratios else "\\emph{TBD}"
        )
        kernel_range = (
            f"[{min(kernel_ratios):.3f}, {max(kernel_ratios):.3f}]"
            if kernel_ratios else "\\emph{TBD}"
        )
        reorder_delta = (
            (
                f"{_format_signed_tenth(statistics.fmean(reorder_deltas))} / "
                f"{_format_signed_tenth(statistics.median(reorder_deltas))}"
            )
            if reorder_deltas else "\\emph{TBD}"
        )
        rows.append(
            f"        {contrast['name']} & {base_name} & {variant_name} "
            f"& {kernel_ratio} & {kernel_ci} & {kernel_range} "
            f"& {sum(ratio > 1.0 for ratio in kernel_ratios)}/"
            f"{len(kernel_ratios)} & {reorder_delta} & {len(common_graphs)} "
            f"& {reorder_graph_count} \\\\"
        )

    latex = (
        "\\begin{table*}[t]\n"
        "    \\centering\n"
        "    \\caption{PR contrast analysis over the SHUFFLED corpus. Most "
        "rows are one-axis controls; ``Partitioner stack'' is a bundled "
        "Rabbit-to-LeidenGVE substitution including preprocessing changes. "
        "Kernel speedup is base time divided by variant time. Reorder deltas "
        "are arithmetic mean / median seconds; four large graphs reuse "
        "Stage-02 timing calibrated by the final live/sidecar policy. "
        "Intervals are unadjusted pointwise summaries, not simultaneous "
        "family-wise guarantees.}\n"
        "    \\resizebox{\\textwidth}{!}{%\n"
        "    \\begin{tabular}{@{}lllccccccc@{}}\n"
        "        \\toprule\n"
        "        \\textbf{Axis} & \\textbf{Base} & \\textbf{Variant} "
        "& \\textbf{Kernel speedup (base/variant)} & \\textbf{95\\% CI} "
        "& \\textbf{Graph range} & \\textbf{Wins} "
        "& \\textbf{$\\Delta$ reorder mean/median (s)} "
        "& \\textbf{Kernel graphs} & \\textbf{Reorder graphs} \\\\\n"
        "        \\midrule\n"
        + "\n".join(rows) + "\n"
        "        \\bottomrule\n"
        "    \\end{tabular}}\n"
        "    \\label{table:ablation}\n"
        "\\end{table*}\n"
    )

    save_latex_table(latex, TABLES_DIR / "table_ablation.tex")


# ============================================================================
# Figure 6: Graph-Type Sensitivity
# ============================================================================


def fig6_sensitivity(sample: bool = False) -> None:
    log.info("Figure 6: Graph-Type Sensitivity")
    del sample

    ensure_dir(TABLES_DIR)

    data = load_json(RESULTS_DIR / "exp2_speedup" / "speedup_results.json")
    type_stats: Dict[str, tuple[str, float, str, float, float, int]] = {}

    if isinstance(data, list) and data:
        data = select_measurement_cohort(data, "exp2 sensitivity")
        controlled_benchmarks = {"pr", "pr_spmv", "sssp", "bc"}

        baseline: Dict[tuple, float] = {}
        for r in data:
            metric = run_metric(r)
            if (
                r.get("algorithm") == "SHUFFLED"
                and r.get("benchmark") in controlled_benchmarks
                and metric
            ):
                baseline[(r["graph"], r["benchmark"])] = metric

        graph_type_map: dict[str, str] = {}
        for gtype, gnames in GRAPH_TYPE_GROUPS.items():
            for gn in gnames:
                graph_type_map[gn] = gtype

        type_algo_speedups: Dict[
            tuple[str, str], Dict[tuple[str, str], float]
        ] = defaultdict(dict)
        for r in data:
            algo = r.get("algorithm", "")
            bench = r.get("benchmark", "")
            if algo == "SHUFFLED" or bench not in controlled_benchmarks:
                continue
            graph = r.get("graph", "")
            avg_t = run_metric(r)
            key = (graph, bench)
            gtype = graph_type_map.get(graph)
            if (
                gtype
                and key in baseline
                and baseline[key] > 0
                and avg_t
                and avg_t > 0
            ):
                if key in type_algo_speedups[(gtype, algo)]:
                    raise RuntimeError(
                        f"Duplicate sensitivity cell for {gtype}/{algo}/{key}"
                    )
                type_algo_speedups[(gtype, algo)][key] = (
                    baseline[key] / avg_t
                )

        available_graphs = {str(row.get("graph")) for row in data}
        for gtype, configured_graphs in GRAPH_TYPE_GROUPS.items():
            type_graphs = [
                graph for graph in configured_graphs
                if graph in available_graphs
            ]
            expected_cells = {
                (graph, benchmark)
                for graph in type_graphs
                for benchmark in controlled_benchmarks
            }
            algo_means: dict[str, float] = {}
            for (gt, algo), cells in type_algo_speedups.items():
                if gt == gtype:
                    if set(cells) != expected_cells:
                        raise RuntimeError(
                            f"Incomplete sensitivity cohort for {gtype}/{algo}: "
                            f"{len(cells)}/{len(expected_cells)} cells"
                        )
                    algo_means[algo] = _geo_mean(list(cells.values()))
            if algo_means:
                ranked = sorted(
                    algo_means.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
                best_name, best_su = ranked[0]
                runner_name, runner_su = (
                    ranked[1] if len(ranked) > 1 else ("---", best_su)
                )
                fixed_leiden = algo_means.get(
                    ALL_ALGORITHMS["12:leiden"]
                )
                if not fixed_leiden:
                    raise RuntimeError(
                        f"Missing fixed Leiden control for {gtype}"
                    )
                type_stats[gtype] = (
                    best_name,
                    best_su,
                    runner_name,
                    runner_su,
                    fixed_leiden,
                    len(expected_cells),
                )

    missing_types = [
        gtype for gtype in GRAPH_TYPE_GROUPS
        if gtype not in type_stats
    ]
    if missing_types and PUBLISH_TO_PAPER:
        raise RuntimeError(
            "Exp6 publication table is missing graph types: "
            + ", ".join(missing_types)
        )
    if missing_types:
        log.warning(
            "  Exp6 omits unavailable graph types: "
            + ", ".join(missing_types)
        )

    rows = []
    for gtype in GRAPH_TYPE_GROUPS:
        stats = type_stats.get(gtype)
        if stats is None:
            continue
        best, best_su, runner, runner_su, fixed_leiden, cell_count = stats
        best = best.replace("_", r"\_")
        runner = runner.replace("_", r"\_")
        rows.append(
            f"        {gtype:<15s} & {best} & {best_su:.3f}$\\times$ "
            f"& {runner} & {runner_su:.3f}$\\times$ "
            f"& {100 * (best_su / fixed_leiden - 1):.1f}\\% "
            f"& {cell_count} \\\\"
        )

    latex = (
        "\\begin{table*}[t]\n"
        "    \\centering\n"
        "    \\caption{Graph-type sensitivity over the controlled-work "
        "cohort (fixed-work PR/PR-SpMV, fixed-source SSSP, and BC). "
        "Speedups are geometric means over all graph--kernel cells relative "
        "to SHUFFLED. Fixed-Leiden regret is the additional speedup available "
        "from the type-specific best over using LeidenGVE--Blocks--Rabbit for "
        "every type; singleton types remain descriptive.}\n"
        "    \\resizebox{\\textwidth}{!}{%\n"
        "    \\begin{tabular}{@{}llclccc@{}}\n"
        "        \\toprule\n"
        "        \\textbf{Graph Type} & \\textbf{Best Method} & "
        "\\textbf{GM} & \\textbf{Runner-Up} & \\textbf{GM} & "
        "\\textbf{Fixed-Leiden regret} & \\textbf{Cells} \\\\\n"
        "        \\midrule\n"
        + "\n".join(rows) + "\n"
        "        \\bottomrule\n"
        "    \\end{tabular}}\n"
        "    \\label{table:sensitivity}\n"
        "\\end{table*}\n"
    )

    save_latex_table(latex, TABLES_DIR / "table_sensitivity.tex")


# ============================================================================
# Figure 7: Chained Ordering
# ============================================================================


def fig7_chained(sample: bool = False) -> None:
    log.info("Figure 7: Chained Ordering")

    ensure_dir(TABLES_DIR)
    data = load_json(RESULTS_DIR / "exp7_chained" / "chained_results.json")
    overhead_data = load_json(
        RESULTS_DIR / "exp3_overhead" / "overhead_results.json"
    )
    rows = []
    if isinstance(data, list) and data:
        data = select_measurement_cohort(data, "exp7")
        calibration_factors, global_calibration = (
            _complete_reorder_calibration_factors(overhead_data)
            if isinstance(overhead_data, list) and overhead_data
            else ({}, 1.0)
        )
        overhead_rows = (
            select_measurement_cohort(overhead_data, "exp3-for-exp7")
            if isinstance(overhead_data, list) and overhead_data
            else []
        )
        by_cell = {
            (
                str(row["graph"]),
                str(row["chain"]),
                str(row["benchmark"]),
            ): run_metric(row)
            for row in data
        }
        overhead_by_cell = {
            (str(row["graph"]), str(row["algo_id"])): row
            for row in overhead_rows
        }
        stage1 = {
            "GB-Leiden+DBG": "Leiden standalone",
            "GB-Leiden+HubCluster": "Leiden standalone",
            "GB-HRAB+DBG": "HRAB standalone",
            "GB-Leiden+GoGraph": "Leiden standalone",
            "RabbitOrder+DBG": "RabbitOrder CSR standalone",
        }
        graphs = sorted({str(row["graph"]) for row in data})
        for chain_name, _flags in CHAINED_ORDERINGS:
            fixed_ratios = []
            incremental = []
            convergence = []
            reorder_costs = []
            break_even = []
            wins = 0
            for graph in graphs:
                shuffled = by_cell.get((graph, "SHUFFLED", "pr"))
                chain = by_cell.get((graph, chain_name, "pr"))
                standalone = by_cell.get(
                    (graph, stage1[chain_name], "pr")
                )
                shuffled_conv = by_cell.get(
                    (graph, "SHUFFLED", "pr_convergence")
                )
                chain_conv = by_cell.get(
                    (graph, chain_name, "pr_convergence")
                )
                if (
                    shuffled and chain and standalone
                    and shuffled_conv and chain_conv
                ):
                    fixed_ratios.append(shuffled / chain)
                    gain = standalone / chain
                    incremental.append(gain)
                    wins += gain > 1
                    convergence.append(shuffled_conv / chain_conv)
                overhead = overhead_by_cell.get(
                    (graph, f"chain:{chain_name}")
                )
                if overhead and chain and shuffled:
                    cost = _calibrated_complete_reorder_time(
                        overhead,
                        calibration_factors,
                        global_calibration,
                    )
                    if cost is None:
                        continue
                    reorder_costs.append(cost)
                    savings = shuffled - chain
                    break_even.append(
                        max(1, math.ceil(cost / savings))
                        if savings > 0 else math.inf
                    )
            ci_low, ci_high = _bootstrap_geo_ci(incremental)
            be_median = statistics.median(break_even)
            be_text = (
                r"$\infty$"
                if math.isinf(be_median)
                else (
                    f"{be_median:.1f}"
                    if not float(be_median).is_integer()
                    else f"{be_median:.0f}"
                )
            )
            rows.append(
                f"        {chain_name} & {_geo_mean(fixed_ratios):.3f}"
                f"$\\times$ & {_geo_mean(incremental):.3f}$\\times$"
                f" & [{ci_low:.3f}, {ci_high:.3f}]"
                f" & {wins}/{len(incremental)}"
                f" & {_geo_mean(convergence):.3f}$\\times$"
                f" & {statistics.median(reorder_costs):.1f}"
                f" & {be_text} \\\\"
            )

    latex = (
        "\\begin{table*}[t]\n"
        "    \\centering\n"
        "    \\caption{Chained PR orderings. Fixed-work speedup is relative "
        "to SHUFFLED; incremental speedup is stage-1 standalone time divided "
        "by chained time in the same cohort. Convergence reports time to "
        "tolerance. Reorder cost is the calibrated median across graphs.}\n"
        "    \\resizebox{\\textwidth}{!}{%\n"
        "    \\begin{tabular}{@{}lccccccc@{}}\n"
        "        \\toprule\n"
        "        \\textbf{Chain} & \\textbf{Fixed vs. shuffled} "
        "& \\textbf{Increment vs. stage 1} & \\textbf{95\\% CI} "
        "& \\textbf{Wins} & \\textbf{Convergence vs. shuffled} "
        "& \\textbf{Reorder median (s)} & \\textbf{Median BE} \\\\\n"
        "        \\midrule\n"
        + "\n".join(rows) + "\n"
        "        \\bottomrule\n"
        "    \\end{tabular}}\n"
        "    \\label{table:chained}\n"
        "\\end{table*}\n"
    )

    save_latex_table(latex, TABLES_DIR / "table_chained.tex")


# ============================================================================
# Figure 8: Representative Reorder Scalability
# ============================================================================


def fig8_scalability(sample: bool = False) -> None:
    log.info("Figure 8: Representative Reorder Scalability")
    del sample
    data = load_json(
        RESULTS_DIR / "exp8_scalability" / "scalability_results.json"
    )
    if not isinstance(data, list) or not data:
        log.warning("  Skipped (no scalability data)")
        return
    data = select_measurement_cohort(data, "exp8")
    grouped: Dict[tuple[str, str, int], list[float]] = defaultdict(list)
    repeats_by_cell: Dict[tuple[str, str, int], set[int]] = defaultdict(set)
    for row in data:
        value = row.get("mapping_generation_time")
        if isinstance(value, (int, float)) and value > 0:
            grouped[(
                str(row["graph"]),
                str(row["algorithm"]),
                int(row["threads"]),
            )].append(float(value))
            repeats_by_cell[(
                str(row["graph"]),
                str(row["algorithm"]),
                int(row["threads"]),
            )].add(int(row.get("repeat", -1)))
    for key, repeats in repeats_by_cell.items():
        if repeats != {0, 1, 2} or len(grouped[key]) != 3:
            raise RuntimeError(
                f"Incomplete scalability repeats for {key}: "
                f"{sorted(repeats)}"
            )
    graph_algorithms = {
        (key[0], key[1]) for key in grouped
    }
    for graph_algorithm in graph_algorithms:
        thread_points = {
            key[2] for key in grouped
            if key[:2] == graph_algorithm
        }
        if thread_points != {1, 2, 4, 8, 16}:
            raise RuntimeError(
                "Incomplete scalability thread points for "
                f"{graph_algorithm}: {sorted(thread_points)}"
            )
    medians = {
        key: statistics.median(values)
        for key, values in grouped.items()
    }
    algorithm_by_key: dict[str, str] = {}
    for row in data:
        algo_key = str(row["algo_key"])
        algorithm = str(row["algorithm"])
        current = algorithm_by_key.get(algo_key)
        if current is not None and current != algorithm:
            raise RuntimeError(
                f"Conflicting Exp8 name for {algo_key}: "
                f"{current!r} vs {algorithm!r}"
            )
        algorithm_by_key[algo_key] = algorithm
    core_algorithms = [
        algorithm_by_key[key] for key in SCALABILITY_ALGORITHM_KEYS
    ]
    twitter_algorithms = {
        algorithm_by_key[key]
        for key in SCALABILITY_TWITTER_ALGORITHM_KEYS
    }
    expected_graph_algorithms = {
        (graph, algorithm)
        for graph in SCALABILITY_GRAPH_NAMES
        for algorithm in core_algorithms
    } | {
        (SCALABILITY_TWITTER_GRAPH, algorithm)
        for algorithm in twitter_algorithms
    }
    if graph_algorithms != expected_graph_algorithms:
        message = (
            "exp8 publication table requires the exact core plus Twitter "
            f"addendum matrix; found {len(graph_algorithms)}/"
            f"{len(expected_graph_algorithms)} graph-algorithm pairs"
        )
        if PUBLISH_TO_PAPER:
            raise RuntimeError(message)
        log.warning(f"  {message}")

    rows = []
    for algorithm in core_algorithms:
        core_times = [
            medians.get((graph, algorithm, 16))
            for graph in SCALABILITY_GRAPH_NAMES
        ]
        core_speedups = [
            medians[(graph, algorithm, 1)]
            / medians[(graph, algorithm, 16)]
            for graph in SCALABILITY_GRAPH_NAMES
            if (graph, algorithm, 1) in medians
            and (graph, algorithm, 16) in medians
        ]
        if any(value is None for value in core_times) or (
            len(core_speedups) != len(SCALABILITY_GRAPH_NAMES)
        ):
            raise RuntimeError(
                f"Incomplete core scalability summary for {algorithm}"
            )
        twitter_time = medians.get(
            (SCALABILITY_TWITTER_GRAPH, algorithm, 16)
        )
        twitter_one = medians.get(
            (SCALABILITY_TWITTER_GRAPH, algorithm, 1)
        )
        twitter_speedup = (
            twitter_one / twitter_time
            if twitter_one and twitter_time else None
        )
        escaped_algorithm = algorithm.replace("_", r"\_")
        twitter_time_text = (
            f"{twitter_time:.3g}" if twitter_time is not None else "---"
        )
        twitter_speedup_text = (
            f"{twitter_speedup:.3f}$\\times$"
            if twitter_speedup is not None else "---"
        )
        rows.append(
            f"        {escaped_algorithm} & "
            + " & ".join(f"{value:.3g}" for value in core_times)
            + f" & {_geo_mean(core_speedups):.3f}$\\times$ "
            f"& {twitter_time_text} & {twitter_speedup_text} \\\\"
        )
    latex = (
        "\\begin{table*}[t]\n"
        "    \\centering\n"
        "    \\caption{Representative mapping-generation scalability. "
        "Absolute 16-thread times are medians of three runs. Core GM speedup "
        "is self-relative to one thread across cit-Patents, USA-road, and "
        "com-Orkut. Twitter is a disclosed reduced four-algorithm addendum. "
        "Schedule-sensitive methods may produce different mappings across "
        "thread counts, so speedup is not fixed-permutation strong scaling.}\n"
        "    \\resizebox{\\textwidth}{!}{%\n"
        "    \\begin{tabular}{@{}lcccccc@{}}\n"
        "        \\toprule\n"
        "        \\textbf{Algorithm} & \\textbf{cit 16T (s)} & "
        "\\textbf{road 16T (s)} & \\textbf{Orkut 16T (s)} & "
        "\\textbf{Core GM speedup} & \\textbf{Twitter 16T (s)} & "
        "\\textbf{Twitter speedup} \\\\\n"
        "        \\midrule\n"
        + "\n".join(rows)
        + "\n        \\bottomrule\n"
        "    \\end{tabular}}\n"
        "    \\label{table:scalability}\n"
        "\\end{table*}\n"
    )
    save_latex_table(latex, TABLES_DIR / "table_scalability.tex")

    if not HAS_MPL:
        return
    ensure_dir(FIGURES_DIR)
    graphs = sorted({key[0] for key in medians})
    for graph in graphs:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        algorithms = sorted({
            key[1] for key in medians if key[0] == graph
        })
        for algorithm in algorithms:
            thread_values = sorted(
                key[2]
                for key in medians
                if key[0] == graph and key[1] == algorithm
            )
            baseline = medians.get((graph, algorithm, 1))
            if not baseline:
                continue
            speedups = [
                baseline / medians[(graph, algorithm, threads)]
                for threads in thread_values
            ]
            ax.plot(
                thread_values,
                speedups,
                marker="o",
                label=algorithm,
            )
        ax.plot([1, 16], [1, 16], "--", color="black", label="Ideal")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16])
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.set_xlabel("Threads")
        ax.set_ylabel("Mapping-generation speedup")
        ax.set_title(f"Reorder scalability: {graph}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=2)
        plt.tight_layout()
        out = FIGURES_DIR / f"fig8_scalability_{graph}.png"
        plt.savefig(out, dpi=300)
        plt.close()


# ============================================================================
# End-to-End Summary Table
# ============================================================================


def table_end_to_end(sample: bool = False) -> None:
    log.info("Table: End-to-End Performance")
    payload = load_json(
        RESULTS_DIR / "exp4_e2e" / "e2e_results.json"
    )
    if not isinstance(payload, dict):
        log.warning("  Skipped (no end-to-end data)")
        return
    if not payload.get("rows"):
        if payload.get("schema") == "end_to_end_results/v3":
            raise RuntimeError(
                "End-to-end table has no algorithms in the primary cohort"
            )
        log.warning("  Skipped (no end-to-end data)")
        return
    primary_benchmarks = set(
        payload.get("primary_cohort", {}).get(
            "benchmarks",
            ["pr", "pr_spmv", "sssp", "bc"],
        )
    )
    expected_cells_per_algorithm = int(
        payload.get("primary_cohort", {}).get(
            "cells_per_algorithm",
            0,
        )
    )
    grouped: Dict[str, list[dict]] = defaultdict(list)
    algorithm_by_key: dict[str, str] = {}
    for row in payload["rows"]:
        if str(row.get("algo_id")) == "0":
            continue
        if row.get("benchmark") not in primary_benchmarks:
            continue
        algorithm = str(row.get("algorithm", row.get("algo_id")))
        algo_key = str(row.get("algo_id"))
        current = algorithm_by_key.get(algo_key)
        if current is not None and current != algorithm:
            raise RuntimeError(
                f"Conflicting end-to-end name for {algo_key}: "
                f"{current!r} vs {algorithm!r}"
            )
        algorithm_by_key[algo_key] = algorithm
        grouped[algorithm].append(row)
    complete_cell_keys = all(
        "graph" in row and "benchmark" in row
        for rows in grouped.values()
        for row in rows
    )
    common_cells: Optional[set[tuple[str, str]]] = (
        set.intersection(*[
            {
                (str(row["graph"]), str(row["benchmark"]))
                for row in rows
            }
            for rows in grouped.values()
        ])
        if grouped and complete_cell_keys
        else None
    )
    all_cells = {
        (str(row["graph"]), str(row["benchmark"]))
        for rows in grouped.values()
        for row in rows
        if "graph" in row and "benchmark" in row
    }
    excluded_cells = (
        sorted(all_cells - common_cells)
        if common_cells is not None else []
    )
    if excluded_cells:
        log.warning(
            "  End-to-end common-cell intersection excludes: "
            + ", ".join(
                f"{graph}/{benchmark}"
                for graph, benchmark in excluded_cells
            )
        )
    censored_cells = payload.get("censored_cells", [])
    censored_algorithms = {
        str(row.get("algorithm", row.get("algo_id")))
        for row in censored_cells
    }
    if common_cells is not None and not common_cells:
        raise RuntimeError(
            "End-to-end table has no graph-kernel cells common "
            "to every algorithm"
        )
    if not grouped:
        raise RuntimeError(
            "End-to-end table has no algorithms in the primary cohort"
        )
    if (
        common_cells is not None
        and expected_cells_per_algorithm > 0
        and len(common_cells) != expected_cells_per_algorithm
    ):
        raise RuntimeError(
            "End-to-end table primary cohort has "
            f"{len(common_cells)} common cells; expected "
            f"{expected_cells_per_algorithm}"
        )
    if sample:
        paper_algorithms = sorted(grouped)
    else:
        missing_paper_algorithms = [
            key for key in E2E_PAPER_ALGORITHM_KEYS
            if key not in algorithm_by_key
        ]
        if missing_paper_algorithms:
            raise RuntimeError(
                "End-to-end paper subset is missing: "
                + ", ".join(missing_paper_algorithms)
            )
        paper_algorithms = [
            algorithm_by_key[key] for key in E2E_PAPER_ALGORITHM_KEYS
        ]
    lines = []
    for algorithm in paper_algorithms:
        rows = grouped[algorithm]
        if common_cells is not None:
            rows = [
                row for row in rows
                if (str(row["graph"]), str(row["benchmark"]))
                in common_cells
            ]
        one_run = [
            float(row["one_run_end_to_end_speedup"])
            for row in rows
            if row.get("one_run_end_to_end_speedup", 0) > 0
        ]
        ten_run = [
            float(row["reuse_counts"]["10"].get(
                "point",
                row["reuse_counts"]["10"].get("speedup", 0),
            ))
            for row in rows
            if row.get("reuse_counts", {}).get("10", {}).get(
                "point",
                row.get("reuse_counts", {}).get("10", {}).get(
                    "speedup", 0
                ),
            ) > 0
        ]
        hundred_run = [
            float(row["reuse_counts"]["100"].get(
                "point",
                row["reuse_counts"]["100"].get("speedup", 0),
            ))
            for row in rows
            if row.get("reuse_counts", {}).get("100", {}).get(
                "point",
                row.get("reuse_counts", {}).get("100", {}).get(
                    "speedup", 0
                ),
            ) > 0
        ]
        break_even_population = [
            (
                float(row["break_even"]["point"])
                if row.get("break_even", {}).get("status") == "finite"
                and isinstance(
                    row.get("break_even", {}).get("point"),
                    int,
                )
                else math.inf
            )
            for row in rows
        ]
        never_fraction = (
            sum(
                row.get("amortization_status") == "never"
                for row in rows
            )
            / len(rows)
            if rows else 0.0
        )
        indeterminate_fraction = (
            sum(
                row.get("amortization_status") == "indeterminate"
                for row in rows
            )
            / len(rows)
            if rows else 0.0
        )
        escaped = algorithm.replace("_", r"\_")
        if algorithm in censored_algorithms:
            escaped += r"\textsuperscript{\dag}"
        break_even_median = (
            statistics.median(break_even_population)
            if break_even_population else math.inf
        )
        break_even_text = (
            r"$\infty$"
            if math.isinf(break_even_median)
            else (
                f"{break_even_median:.1f}"
                if not float(break_even_median).is_integer()
                else f"{break_even_median:.0f}"
            )
        )
        lines.append(
            f"        {escaped} & {_geo_mean(one_run):.3g}$\\times$"
            f" & {_geo_mean(ten_run):.3g}$\\times$"
            f" & {_geo_mean(hundred_run):.3g}$\\times$"
            f" & {break_even_text}"
            f" & {100 * never_fraction:.1f}\\%"
            f" & {100 * indeterminate_fraction:.1f}\\%"
            f" & {len(rows)} \\\\"
        )
    dagger_note = (
        "$\\dag$ marks an algorithm with a censored overhead cell. "
        if censored_algorithms else ""
    )
    latex = (
        "\\begin{table*}[t]\n"
        "    \\centering\n"
        "    \\caption{End-to-end speedup over the SHUFFLED baseline for the "
        "controlled-work primary cohort (fixed-work PR/PR-SpMV, fixed-source "
        "SSSP, and BC). "
        "Geometric means include mapping generation, validation, CSR "
        "relocation, and kernel execution and use the graph--kernel "
        "intersection common to every algorithm. "
        + dagger_note
        + "Median BE "
        "is the median over all primary cells, assigning $\\infty$ to never-"
        "amortizing and statistically indeterminate cells; Never and Indet. "
        "report those fractions. The paper subset contains the primary "
        "conventional competitors and all ten core GraphBrew variants; "
        "controlled compose candidates are reported in Experiments 5 and 7, "
        "and the artifact retains the full 31-method matrix."
        + (
            f" {len(excluded_cells)} graph--kernel cells are omitted by the "
            "common-cell rule."
            if excluded_cells else ""
        )
        + "}\n"
        "    \\resizebox{\\textwidth}{!}{%\n"
        "    \\begin{tabular}{@{}lccccccc@{}}\n"
        "        \\toprule\n"
        "        \\textbf{Algorithm} & \\textbf{1 run} & "
        "\\textbf{10 runs} & \\textbf{100 runs} & "
        "\\textbf{Median BE} & \\textbf{Never} & "
        "\\textbf{Indet.} & \\textbf{Cells} \\\\\n"
        "        \\midrule\n"
        + "\n".join(lines)
        + "\n        \\bottomrule\n"
        "    \\end{tabular}}\n"
        "    \\label{table:end-to-end}\n"
        "\\end{table*}\n"
    )
    save_latex_table(latex, TABLES_DIR / "table_end_to_end.tex")


# ============================================================================
# Main
# ============================================================================

FIGURES = {
    1: ("Cache Performance", fig1_cache_performance),
    2: ("Kernel Speedup", fig2_kernel_speedup),
    3: ("Reorder Overhead", fig3_reorder_overhead),
    4: ("End-to-End Performance", table_end_to_end),
    5: ("Controlled Ablation", fig5_ablation),
    6: ("Graph-Type Sensitivity", fig6_sensitivity),
    7: ("Chained Ordering", fig7_chained),
    8: ("Representative Reorder Scalability", fig8_scalability),
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
