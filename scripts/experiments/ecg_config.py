#!/usr/bin/env python3
"""
ECG Paper Experiment Configuration
===================================
Configuration for cache replacement policy experiments comparing:
  - Baselines: LRU, FIFO, RANDOM, LFU, SRRIP
  - Graph-aware: GRASP, P-OPT, ECG

Designed for the ECG paper: "Expressing Locality and Prefetching
for Optimal Caching in Graph Structures"

Organized into two sections:
  Section A: Accuracy Validation — verify GRASP and P-OPT faithfulness
  Section B: Performance Showcase — method comparison and reorder effects
"""

from pathlib import Path
import os

# ============================================================================
# Paths
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BIN_SIM_DIR = PROJECT_ROOT / "bench" / "bin_sim"
RESULTS_DIR = PROJECT_ROOT / "results" / "ecg_experiments"

# ============================================================================
# Cache Replacement Policies (9 total)
# ============================================================================
BASELINE_POLICIES = ["LRU", "FIFO", "RANDOM", "LFU", "SRRIP"]
GRAPH_AWARE_POLICIES = ["GRASP", "POPT", "ECG"]
ALL_POLICIES = BASELINE_POLICIES + GRAPH_AWARE_POLICIES
PREVIEW_POLICIES = ["LRU", "SRRIP", "GRASP", "POPT", "ECG"]

# ECG modes (from ECGMode enum: DBG_PRIMARY, POPT_PRIMARY, DBG_ONLY, ECG_EMBEDDED)
ECG_MODES = ["DBG_PRIMARY", "POPT_PRIMARY", "DBG_ONLY", "ECG_EMBEDDED"]

# ============================================================================
# Section A: Accuracy Validation — GRASP and P-OPT faithfulness
# ============================================================================
# Pairs designed to test specific paper claims.
#
# A1: GRASP invariants (Faldu et al., HPCA 2020)
#   Claim 1: With DBG reordering, GRASP should always beat SRRIP (same eviction,
#            but degree-aware insertion gives hot vertices lower RRPV)
#   Claim 2: GRASP on original ordering ~ SRRIP (no DBG -> no region info)
#   Claim 3: Hot hub vertices (bucket-0) get RRPV much lower than cold vertices
#
# A2: P-OPT invariants (Balaji et al., HPCA 2021)
#   Claim 1: P-OPT should approach OPT miss rate (within 1-5% on small graphs)
#   Claim 2: P-OPT is reorder-agnostic — rereference matrix captures vertex ID
#            patterns regardless of ordering (similar miss rate with/without DBG)
#   Claim 3: P-OPT should beat all RRIP variants (pure look-ahead dominates aging)
#
# A3: ECG layered correctness
#   Claim 1: ECG(DBG_ONLY mode) ~ GRASP (same DBG insertion + SRRIP eviction)
#   Claim 2: ECG(POPT_PRIMARY) should approach P-OPT miss rate when matrix present
#   Claim 3: ECG(DBG_PRIMARY) is the sweet spot — DBG structure + P-OPT tiebreak

ACCURACY_PAIRS = [
    # (reorder, policy, env_extra, label, expected_relation)
    # --- GRASP vs SRRIP ---
    ("-o 5",  "SRRIP", {},                          "DBG+SRRIP",           "baseline"),
    ("-o 5",  "GRASP", {},                          "DBG+GRASP",           "grasp_beats_srrip"),
    ("-o 0",  "GRASP", {},                          "Original+GRASP",      "grasp_no_dbg_eq_srrip"),
    ("-o 0",  "SRRIP", {},                          "Original+SRRIP",      "baseline"),
    # --- P-OPT invariants ---
    ("-o 0",  "POPT",  {},                          "Original+P-OPT",      "popt_best"),
    ("-o 5",  "POPT",  {},                          "DBG+P-OPT",           "popt_reorder_agnostic"),
    ("-o 0",  "LRU",   {},                          "Original+LRU",        "baseline"),
    # --- ECG mode equivalences ---
    ("-o 5",  "ECG",   {"ECG_MODE": "DBG_ONLY"},    "DBG+ECG(DBG_ONLY)",   "ecg_dbg_eq_grasp"),
    ("-o 5",  "ECG",   {"ECG_MODE": "POPT_PRIMARY"},"DBG+ECG(POPT_PRIMARY)","ecg_popt_approach"),
    ("-o 5",  "ECG",   {"ECG_MODE": "DBG_PRIMARY"}, "DBG+ECG(DBG_PRIMARY)", "ecg_sweet_spot"),
]

# ============================================================================
# Section B: Reorder x Policy Interaction Pairs (Performance)
# ============================================================================
REORDER_POLICY_PAIRS = [
    # (reorder_opt, policy, label)
    ("-o 0",         "LRU",   "Original+LRU"),
    ("-o 0",         "SRRIP", "Original+SRRIP"),
    ("-o 5",         "LRU",   "DBG+LRU"),
    ("-o 5",         "SRRIP", "DBG+SRRIP"),
    ("-o 5",         "GRASP", "DBG+GRASP"),
    ("-o 5",         "POPT",  "DBG+P-OPT"),
    ("-o 5",         "ECG",   "DBG+ECG"),
    ("-o 0",         "POPT",  "Original+P-OPT"),
    ("-o 8:csr",     "LRU",   "Rabbit+LRU"),
    ("-o 12:leiden", "LRU",   "GraphBrew+LRU"),
    ("-o 12:leiden", "ECG",   "GraphBrew+ECG"),
]

# Reorderings to test in isolation (for reorder effect analysis)
REORDER_VARIANTS = [
    ("-o 0",         "Original"),
    ("-o 5",         "DBG"),
    ("-o 8:csr",     "RabbitOrder"),
    ("-o 12:leiden", "GraphBrew"),
]

# ============================================================================
# Benchmark Algorithms
# ============================================================================
BENCHMARKS = ["pr", "pr_spmv", "bfs", "cc", "cc_sv", "sssp", "bc"]
BENCHMARKS_PREVIEW = ["pr", "bfs"]

# Algorithm categories for analysis grouping
ITERATIVE_BENCHMARKS = ["pr", "pr_spmv", "cc_sv"]
TRAVERSAL_BENCHMARKS = ["bfs", "sssp", "bc"]

# ============================================================================
# Cache Configuration
# ============================================================================
DEFAULT_CACHE = {
    "CACHE_L1_SIZE": "32768", "CACHE_L1_WAYS": "8",
    "CACHE_L2_SIZE": "262144", "CACHE_L2_WAYS": "4",
    "CACHE_L3_SIZE": "8388608", "CACHE_L3_WAYS": "16",
    "CACHE_LINE_SIZE": "64",
}

CACHE_SIZES_SWEEP = [
    32 * 1024, 64 * 1024, 128 * 1024, 256 * 1024, 512 * 1024,
    1024 * 1024, 2 * 1024 * 1024, 4 * 1024 * 1024, 8 * 1024 * 1024,
    16 * 1024 * 1024, 32 * 1024 * 1024, 64 * 1024 * 1024,
]

# ============================================================================
# Evaluation Graphs
# ============================================================================
EVAL_GRAPHS = [
    {"name": "soc-pokec",         "short": "pokec",      "type": "Social",   "vertices_m": 1.63,  "edges_m": 30.62},
    {"name": "soc-LiveJournal1",  "short": "livejournal", "type": "Social",  "vertices_m": 4.85,  "edges_m": 68.99},
    {"name": "com-orkut",         "short": "orkut",      "type": "Social",   "vertices_m": 3.07,  "edges_m": 117.19},
    {"name": "cit-Patents",       "short": "patents",    "type": "Citation", "vertices_m": 6.01,  "edges_m": 16.52},
    {"name": "USA-Road",          "short": "road",       "type": "Road",     "vertices_m": 23.95, "edges_m": 58.33},
    {"name": "wikipedia_link_en", "short": "wikipedia",  "type": "Content",  "vertices_m": 12.15, "edges_m": 378.14},
]
EVAL_GRAPHS_PREVIEW = EVAL_GRAPHS[:2]

# Accuracy validation uses fewer graphs for focused testing
ACCURACY_GRAPHS = EVAL_GRAPHS[:3]  # pokec, livejournal, orkut (social -- most relevant)

# ============================================================================
# Timeouts
# ============================================================================
TIMEOUT_SIM = 600
TIMEOUT_SIM_HEAVY = 1800
TRIALS = 1  # Cache simulation is deterministic


def policy_env(policy, cache_config=None, extra_env=None):
    """Build environment variables for a cache sim run."""
    env = dict(os.environ)
    env["CACHE_POLICY"] = policy
    env["CACHE_ULTRAFAST"] = "0"
    env.update(cache_config or DEFAULT_CACHE)
    if extra_env:
        env.update(extra_env)
    return env


def format_cache_size(size_bytes):
    """Format bytes to human-readable cache size."""
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes // (1024 * 1024)}MB"
    if size_bytes >= 1024:
        return f"{size_bytes // 1024}KB"
    return f"{size_bytes}B"
