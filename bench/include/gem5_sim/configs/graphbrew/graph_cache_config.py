#!/usr/bin/env python3
"""
gem5 cache hierarchy configuration matching GraphBrew/ECG defaults.

Provides factory functions to create cache hierarchy objects with parameters
matching the standalone cache_sim defaults:

    L1D: 32KB, 8-way, 64B lines
    L1I: 32KB, 8-way, 64B lines
    L2:  256KB, 4-way, 64B lines  (private per core)
    L3:  8MB, 16-way, 64B lines   (shared)

These match:
    - bench/include/cache_sim/cache_sim.h CacheHierarchy defaults
    - scripts/experiments/ecg_config.py DEFAULT_CACHE
    - Makefile CACHE_L1_SIZE/CACHE_L2_SIZE/CACHE_L3_SIZE

Reference: scripts/experiments/ecg_config.py lines 72-78
"""

import m5
from m5.objects import *


# =============================================================================
# Default cache parameters (matching ECG standalone simulator)
# =============================================================================
DEFAULTS = {
    "l1d_size":  "32kB",
    "l1d_assoc": 8,
    "l1i_size":  "32kB",
    "l1i_assoc": 8,
    "l2_size":   "256kB",
    "l2_assoc":  4,
    "l3_size":   "8MB",
    "l3_assoc":  16,
    "line_size":  64,
}


# =============================================================================
# Replacement Policy Factory
# =============================================================================
POLICY_MAP = {
    "LRU":   lambda: LRURP(),
    "FIFO":  lambda: FIFORP(),
    "SRRIP": lambda: BRRIPRP(btp=0),  # BRRIP with btp=0 == SRRIP
    "BRRIP": lambda: BRRIPRP(),
    "RANDOM": lambda: RandomRP(),
}


def make_replacement_policy(name, **kwargs):
    """Create a replacement policy SimObject by name.

    For graph-aware policies (GRASP, POPT, ECG), imports from the
    GraphBrew overlay SimObjects.

    Args:
        name: Policy name (LRU, FIFO, SRRIP, GRASP, POPT, ECG, etc.)
        **kwargs: Extra parameters passed to the policy constructor.

    Returns:
        gem5 SimObject for the replacement policy.
    """
    upper = name.upper()

    if upper in POLICY_MAP:
        return POLICY_MAP[upper]()

    if upper == "GRASP":
        return GraphGraspRP(
            max_rrpv=kwargs.get("max_rrpv", 7),
            num_buckets=kwargs.get("num_buckets", 11),
            hot_fraction=kwargs.get("hot_fraction", 0.1),
        )
    elif upper in ("POPT", "P-OPT"):
        return GraphPoptRP(
            max_rrpv=kwargs.get("max_rrpv", 7),
        )
    elif upper == "ECG":
        return GraphEcgRP(
            rrpv_max=kwargs.get("rrpv_max", 7),
            num_buckets=kwargs.get("num_buckets", 11),
            ecg_mode=kwargs.get("ecg_mode", "DBG_PRIMARY"),
        )
    else:
        print(f"Warning: Unknown policy '{name}', defaulting to LRU")
        return LRURP()


# =============================================================================
# Cache Hierarchy Builder
# =============================================================================
def make_l1d_cache(policy="LRU", **policy_kwargs):
    """Create L1 data cache matching ECG defaults."""
    return Cache(
        size=DEFAULTS["l1d_size"],
        assoc=DEFAULTS["l1d_assoc"],
        tag_latency=2,
        data_latency=2,
        response_latency=2,
        mshrs=4,
        tgts_per_mshr=20,
        replacement_policy=make_replacement_policy(policy, **policy_kwargs),
    )


def make_l1i_cache(policy="LRU", **policy_kwargs):
    """Create L1 instruction cache."""
    return Cache(
        size=DEFAULTS["l1i_size"],
        assoc=DEFAULTS["l1i_assoc"],
        tag_latency=2,
        data_latency=2,
        response_latency=2,
        mshrs=4,
        tgts_per_mshr=20,
        replacement_policy=make_replacement_policy(policy, **policy_kwargs),
    )


def make_l2_cache(policy="LRU", **policy_kwargs):
    """Create L2 private cache matching ECG defaults."""
    return Cache(
        size=DEFAULTS["l2_size"],
        assoc=DEFAULTS["l2_assoc"],
        tag_latency=10,
        data_latency=10,
        response_latency=10,
        mshrs=20,
        tgts_per_mshr=12,
        replacement_policy=make_replacement_policy(policy, **policy_kwargs),
    )


def make_l3_cache(policy="LRU", **policy_kwargs):
    """Create L3 shared cache matching ECG defaults (8MB, 16-way)."""
    return Cache(
        size=DEFAULTS["l3_size"],
        assoc=DEFAULTS["l3_assoc"],
        tag_latency=20,
        data_latency=20,
        response_latency=20,
        mshrs=32,
        tgts_per_mshr=16,
        replacement_policy=make_replacement_policy(policy, **policy_kwargs),
    )


def make_droplet_prefetcher(**kwargs):
    """Create DROPLET indirect graph prefetcher."""
    return GraphDropletPrefetcher(
        prefetch_degree=kwargs.get("prefetch_degree", 4),
        indirect_degree=kwargs.get("indirect_degree", 8),
        stride_table_size=kwargs.get("stride_table_size", 16),
    )
