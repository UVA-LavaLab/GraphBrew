#!/usr/bin/env python3
"""
ECG Paper Experiment Configuration
===================================
Configuration for cache replacement policy experiments comparing:
  - Baselines: LRU, FIFO, RANDOM, LFU, SRRIP
  - Graph-aware: GRASP, P-OPT, ECG

Designed for the ECG paper: "Expressing Locality and Prefetching
for Optimal Caching in Graph Structures"
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

# ============================================================================
# Reorder × Policy Interaction Pairs
# ============================================================================
# GRASP/ECG require DBG reordering; P-OPT is reorder-agnostic
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

# ============================================================================
# Timeouts
# ============================================================================
TIMEOUT_SIM = 600
TIMEOUT_SIM_HEAVY = 1800
TRIALS = 1  # Cache simulation is deterministic


def policy_env(policy, cache_config=None):
    """Build environment variables for a cache sim run."""
    env = dict(os.environ)
    env["CACHE_POLICY"] = policy
    # Graph-aware policies need full CacheHierarchy (not UltraFast clock-based)
    env["CACHE_ULTRAFAST"] = "0"
    env.update(cache_config or DEFAULT_CACHE)
    return env


def format_cache_size(size_bytes):
    """Format bytes to human-readable cache size."""
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes // (1024 * 1024)}MB"
    if size_bytes >= 1024:
        return f"{size_bytes // 1024}KB"
    return f"{size_bytes}B"
