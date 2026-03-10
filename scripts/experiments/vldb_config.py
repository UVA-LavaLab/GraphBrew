#!/usr/bin/env python3
"""
Shared configuration for VLDB 2026 GraphBrew paper experiments.

Defines all algorithm IDs, GraphBrew variants, chained orderings,
graph datasets, benchmarks, and evaluation parameters used across
the experiment runner and figure generator.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BIN_DIR = PROJECT_ROOT / "bench" / "bin"
BIN_SIM_DIR = PROJECT_ROOT / "bench" / "bin_sim"
RESULTS_DIR = PROJECT_ROOT / "results" / "vldb_paper"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# ---------------------------------------------------------------------------
# Algorithm Definitions
# ---------------------------------------------------------------------------

# Baseline reorder algorithm IDs (no GraphBrew variants)
BASELINE_ALGORITHMS = {
    0: "ORIGINAL",
    1: "RANDOM",
    2: "SORT",
    3: "HUBSORT",
    4: "HUBCLUSTER",
    5: "DBG",
    6: "HUBSORTDBG",
    7: "HUBCLUSTERDBG",
    8: "RABBITORDER",
    9: "GORDER",
    11: "RCM",
    16: "GoGraphOrder",
}

# GraphBrew variants (all use algorithm ID 12)
GRAPHBREW_VARIANTS = [
    "leiden",       # Default multi-pass Leiden + BFS
    "rabbit",       # Single-pass RabbitOrder
    "hubcluster",   # Leiden + hub-first ordering
    "hrab",         # Hybrid Leiden + RabbitOrder super-graph
    "tqr",          # Tile-Quantized RabbitOrder
    "hcache",       # Hierarchical cache-aware
    "streaming",    # Leiden + lazy aggregation
]

# Chained orderings: list of (display_name, cli_flags) tuples
CHAINED_ORDERINGS = [
    ("SORT+RabbitOrder",     ["-o", "2", "-o", "8:csr"]),
    ("SORT+GB-Leiden",       ["-o", "2", "-o", "12:leiden"]),
    ("DBG+GB-Leiden",        ["-o", "5", "-o", "12:leiden"]),
    ("SORT+GB-HRAB",         ["-o", "2", "-o", "12:hrab"]),
    ("HubClusterDBG+Rabbit", ["-o", "7", "-o", "8:csr"]),
]

# Full algorithm list for experiments: ID → display name
ALL_ALGORITHMS = {
    **BASELINE_ALGORITHMS,
    **{f"12:{v}": f"GB-{v.capitalize()}" for v in GRAPHBREW_VARIANTS},
}

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

# Core benchmarks (match paper Section 2.2)
BENCHMARKS = ["bfs", "pr", "pr_spmv", "sssp", "cc", "cc_sv", "bc"]

# Quick preview benchmarks
BENCHMARKS_PREVIEW = ["pr", "bfs"]

# ---------------------------------------------------------------------------
# Graph Datasets
# ---------------------------------------------------------------------------

# Graphs for full evaluation (match paper Table 2)
EVAL_GRAPHS = [
    {"name": "cit-Patents",        "short": "patents",   "type": "citation",      "vertices_m": 6.01,   "edges_m": 16.52},
    {"name": "soc-pokec",          "short": "pokec",     "type": "social",        "vertices_m": 1.63,   "edges_m": 30.62},
    {"name": "USA-road-d.USA",     "short": "road",      "type": "road",          "vertices_m": 23.95,  "edges_m": 58.33},
    {"name": "soc-LiveJournal1",   "short": "journal",   "type": "social",        "vertices_m": 4.85,   "edges_m": 68.99},
    {"name": "delaunay_n24",       "short": "delaunay",  "type": "mesh",          "vertices_m": 16.78,  "edges_m": 100.66},
    {"name": "com-Orkut",          "short": "orkut",     "type": "social",        "vertices_m": 3.07,   "edges_m": 117.19},
    {"name": "hollywood-2009",     "short": "hollywood", "type": "collaboration", "vertices_m": 1.14,   "edges_m": 113.89},
    {"name": "wikipedia_link_en",  "short": "wikipedia", "type": "content",       "vertices_m": 12.15,  "edges_m": 378.14},
    {"name": "Gong-gplus",         "short": "gplus",     "type": "social",        "vertices_m": 28.94,  "edges_m": 462.99},
    {"name": "webbase-2001",       "short": "webbase",   "type": "web",           "vertices_m": 118.14, "edges_m": 1019.90},
    {"name": "twitter7",           "short": "twitter",   "type": "social",        "vertices_m": 61.79,  "edges_m": 1468.36},
]

# Small graphs for preview mode
PREVIEW_GRAPHS = [
    {"name": "email-Eu-core",      "short": "email",     "type": "social",   "vertices_m": 0.001,  "edges_m": 0.025},
    {"name": "cit-Patents",        "short": "patents",   "type": "citation", "vertices_m": 6.01,   "edges_m": 16.52},
]

# ---------------------------------------------------------------------------
# Experiment Parameters
# ---------------------------------------------------------------------------

# Number of algorithm trials per benchmark
TRIALS_FULL = 3
TRIALS_PREVIEW = 1

# Timeout per command (seconds)
TIMEOUT_FULL = 3600
TIMEOUT_PREVIEW = 300

# Cache simulation sweep sizes (bytes)
CACHE_SIZES = [
    32 * 1024,        # 32 KB
    64 * 1024,        # 64 KB
    128 * 1024,       # 128 KB
    256 * 1024,       # 256 KB
    512 * 1024,       # 512 KB
    1 * 1024**2,      # 1 MB
    2 * 1024**2,      # 2 MB
    4 * 1024**2,      # 4 MB
    8 * 1024**2,      # 8 MB
    16 * 1024**2,     # 16 MB
    32 * 1024**2,     # 32 MB
    64 * 1024**2,     # 64 MB
]

# Thread counts for scalability experiment
THREAD_COUNTS = [1, 2, 4, 8, 16, 32]

# ---------------------------------------------------------------------------
# Ablation study configurations (Experiment 5)
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = [
    {"name": "Original",                  "algo": "0",            "desc": "No reordering"},
    {"name": "Leiden-only",               "algo": "12:leiden",    "desc": "Leiden + hierarchical sort"},
    {"name": "Leiden+BFS",                "algo": "12:leiden",    "desc": "Leiden + BFS intra-community"},
    {"name": "Leiden+HubCluster",         "algo": "12:hubcluster","desc": "Leiden + hub-first ordering"},
    {"name": "HRAB",                      "algo": "12:hrab",      "desc": "Leiden + RabbitOrder super-graph"},
    {"name": "HRAB+hubx",                "algo": "12:hrab:hubx", "desc": "HRAB + hub extraction"},
    {"name": "Gorder",                    "algo": "9",            "desc": "Gorder (reference)"},
]

# ---------------------------------------------------------------------------
# Graph type groupings (Experiment 6)
# ---------------------------------------------------------------------------

GRAPH_TYPE_GROUPS = {
    "social":        ["soc-pokec", "com-Orkut", "soc-LiveJournal1", "Gong-gplus", "twitter7"],
    "web":           ["webbase-2001"],
    "road":          ["USA-road-d.USA"],
    "citation":      ["cit-Patents"],
    "content":       ["wikipedia_link_en"],
    "collaboration": ["hollywood-2009"],
    "mesh":          ["delaunay_n24"],
}


def get_converter_flags(algo_key: str) -> list[str]:
    """Convert an algorithm key like '12:hrab' to converter CLI flags."""
    if ":" in algo_key:
        parts = algo_key.split(":", 1)
        return ["-o", f"{parts[0]}:{parts[1]}"]
    return ["-o", algo_key]


# ---------------------------------------------------------------------------
# VLDB Graph Download Sources
# ---------------------------------------------------------------------------
# Maps EVAL_GRAPHS name → download info.  Graphs marked source="suitesparse"
# are fetched automatically; others require manual download.

VLDB_GRAPH_SOURCES = {
    # ── Auto-download from SuiteSparse (already in catalog) ──
    "email-Eu-core":     {"source": "catalog"},
    "cit-Patents":       {"source": "catalog"},
    "soc-pokec":         {"source": "catalog"},
    "USA-road-d.USA":    {"source": "catalog"},
    "soc-LiveJournal1":  {"source": "catalog"},
    "delaunay_n24":      {"source": "catalog"},
    "com-Orkut":         {"source": "catalog"},
    "hollywood-2009":    {"source": "catalog"},
    "webbase-2001":      {"source": "catalog"},
    "twitter7":          {"source": "catalog"},

    # ── Manual download required ──
    "wikipedia_link_en": {
        "source": "manual",
        "url": "http://konect.cc/networks/wikipedia_link_en/",
        "instructions": (
            "Download from KONECT: http://konect.cc/networks/wikipedia_link_en/\n"
            "Extract and convert to Matrix Market (.mtx) or edge-list (.el) format,\n"
            "then place as results/graphs/wikipedia_link_en/wikipedia_link_en.el"
        ),
    },
    "Gong-gplus": {
        "source": "manual",
        "url": "https://people.duke.edu/~zg70/gplus.html",
        "gdrive_id": "1HF8Q2N_hxsaQ26MarKYxZEQhqI66qAxV",
        "instructions": (
            "Download from https://people.duke.edu/~zg70/gplus.html\n"
            "  (Google Drive: https://drive.google.com/file/d/"
            "1HF8Q2N_hxsaQ26MarKYxZEQhqI66qAxV/view)\n"
            "Extract snapshot 4 (keep all edges with TimeID 0-3), convert to\n"
            "edge-list format, and place as results/graphs/Gong-gplus/Gong-gplus.el"
        ),
    },
}
