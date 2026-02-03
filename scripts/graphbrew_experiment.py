#!/usr/bin/env python3
"""
GraphBrew Unified Experiment Pipeline
=====================================

A comprehensive one-click script that runs the complete GraphBrew experiment workflow:

**Core Pipeline (--phase all):**
1. Download graphs (if not present) - via --download-only or --full
2. Build binaries (if not present) - via --build or --full
3. Phase 1: Generate reorderings with label-mapping for consistency
4. Phase 2: Run execution benchmarks on all graphs
5. Phase 3: Run cache simulations (optional, skip with --skip-cache)
6. Phase 4: Generate type-based perceptron weights (scripts/weights/active/type_*.json)

**Validation & Analysis:**
7. Phase 6: Adaptive order analysis (--adaptive-analysis)
8. Phase 7: Adaptive vs fixed comparison (--adaptive-comparison)
9. Phase 8: Brute-force validation (--brute-force)

**Training Modes:**
10. Standard training (--train): One-pass pipeline that runs all phases
11. Iterative training (--train-iterative): Repeatedly adjusts weights until target accuracy
12. Batched training (--train-batched): Process graphs in batches for large datasets

**Algorithm Variant Testing:**
    For LeidenCSR (17), LeidenDendrogram (16), and RabbitOrder (8), you can test
    specific variants or all variants:
    
    # Test all algorithm variants
    python scripts/graphbrew_experiment.py --train --all-variants --size small
    
    # Test specific variants only
    python scripts/graphbrew_experiment.py --train --csr-variants gve gveopt gveopt2 --size small
    
    # Test new optimized variants (best performance)
    python scripts/graphbrew_experiment.py --train --csr-variants gveopt2 gveadaptive --size medium
    
    # With custom Leiden parameters
    python scripts/graphbrew_experiment.py --train --all-variants \\
        --resolution 1.0 --passes 5 --size medium
    
    RabbitOrder (8) variants: csr (default), boost
    LeidenCSR (17) variants:
      - gve (default): Standard GVE-Leiden with refinement
      - gveopt: Cache-optimized with prefetching
      - gveopt2: CSR-based aggregation (fastest reordering) ⭐
      - gveadaptive: Dynamic resolution adjustment (best for unknown graphs) ⭐
      - gveoptsort: Multi-level sort ordering
      - gveturbo: Speed-optimized (optional refinement skip)
      - gvefast: CSR buffer reuse (leiden.hxx style)
      - gvedendo/gveoptdendo: Incremental dendrogram building
      - gverabbit: GVE-Rabbit hybrid (fastest)
      - dfs, bfs, hubsort, modularity: Alternative ordering strategies
    LeidenDendrogram (16) variants: dfs, dfshub, dfssize, bfs, hybrid
    
    Resolution modes (for --resolution):
      - Fixed: 1.5 (use specified value)
      - Auto: auto or 0 (compute from graph density/CV)
      - Dynamic: dynamic (adjust per-pass, gveadaptive only)
      - Dynamic+Init: dynamic_2.0 (start at 2.0, adjust per-pass)

All outputs are saved to the results/ directory for clean organization.
Type-based weights are saved to scripts/weights/active/type_*.json.

Usage:
    python scripts/graphbrew_experiment.py --help
    python scripts/graphbrew_experiment.py --full --size small     # Full pipeline with small graphs
    python scripts/graphbrew_experiment.py --train --size medium   # Train on medium graphs
    python scripts/graphbrew_experiment.py --download-only         # Just download graphs
    python scripts/graphbrew_experiment.py --phase all             # Run all experiment phases
    python scripts/graphbrew_experiment.py --brute-force           # Run brute-force validation

Quick Start (One-Click):
    python scripts/graphbrew_experiment.py --full --size small --auto              # Small graphs, auto resources
    python scripts/graphbrew_experiment.py --full --size large --auto --quick      # Large graphs, quick mode
    python scripts/graphbrew_experiment.py --train --all-variants --auto --size medium  # Train all variants

Author: GraphBrew Team
"""

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
import math
import glob
import shutil
import tarfile
import gzip
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import urllib.request
import urllib.error

# Ensure project root is in path for lib imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import shared utilities from lib/ module (REQUIRED)
# The lib/ module is part of this project and must be available
from scripts.lib import (
    # Core constants
    ALGORITHMS as LIB_ALGORITHMS,
    BENCHMARKS as LIB_BENCHMARKS,
    # RabbitOrder variants (csr default)
    RABBITORDER_VARIANTS as LIB_RABBITORDER_VARIANTS,
    RABBITORDER_DEFAULT_VARIANT as LIB_RABBITORDER_DEFAULT_VARIANT,
    # GraphBrewOrder variants (leiden default for backward compat)
    GRAPHBREW_VARIANTS as LIB_GRAPHBREW_VARIANTS,
    GRAPHBREW_DEFAULT_VARIANT as LIB_GRAPHBREW_DEFAULT_VARIANT,
    # Leiden variants (gve default for LeidenCSR)
    LEIDEN_CSR_VARIANTS as LIB_LEIDEN_CSR_VARIANTS,
    LEIDEN_CSR_DEFAULT_VARIANT as LIB_LEIDEN_CSR_DEFAULT_VARIANT,
    LEIDEN_DENDROGRAM_VARIANTS as LIB_LEIDEN_DENDROGRAM_VARIANTS,
    LEIDEN_DEFAULT_RESOLUTION as LIB_LEIDEN_DEFAULT_RESOLUTION,
    LEIDEN_DEFAULT_PASSES as LIB_LEIDEN_DEFAULT_PASSES,
    # Paths
    PROJECT_ROOT as LIB_PROJECT_ROOT,
    BIN_DIR as LIB_BIN_DIR,
    BIN_SIM_DIR as LIB_BIN_SIM_DIR,
    GRAPHS_DIR as LIB_GRAPHS_DIR,
    RESULTS_DIR as LIB_RESULTS_DIR,
    WEIGHTS_DIR as LIB_WEIGHTS_DIR,
    # Utils
    Logger,
    run_command as lib_run_command,
    get_timestamp,
    # Features (imported directly - removes need for local duplicates)
    GRAPH_TYPE_GENERIC,
    GRAPH_TYPE_SOCIAL,
    GRAPH_TYPE_ROAD,
    GRAPH_TYPE_WEB,
    GRAPH_TYPE_POWERLAW,
    GRAPH_TYPE_UNIFORM,
    ALL_GRAPH_TYPES,
    BYTES_PER_EDGE,
    BYTES_PER_NODE,
    MEMORY_SAFETY_FACTOR,
    load_graph_properties_cache,
    save_graph_properties_cache,
    update_graph_properties,
    get_graph_properties,
    detect_graph_type,
    get_graph_type_from_name,
    get_graph_type_from_properties,
    compute_clustering_coefficient_sample,
    estimate_diameter_bfs,
    count_subcommunities_quick,
    compute_extended_features,
    get_available_memory_gb,
    get_total_memory_gb,
    estimate_graph_memory_gb,
    get_available_disk_gb,
    get_total_disk_gb,
    get_num_threads,
    # Download
    GRAPH_CATALOG,
    DOWNLOAD_GRAPHS_SMALL,
    DOWNLOAD_GRAPHS_MEDIUM,
    DOWNLOAD_GRAPHS_LARGE,
    DOWNLOAD_GRAPHS_XLARGE,
    DownloadableGraph,
    download_graph as lib_download_graph,
    download_graphs as lib_download_graphs,
    download_graphs_parallel as lib_download_graphs_parallel,
    get_graph_info as lib_get_graph_info,
    get_graphs_by_size as lib_get_graphs_by_size,
    # Build
    build_binaries as lib_build_binaries,
    check_binaries as lib_check_binaries,
    ensure_binaries as lib_ensure_binaries,
    # Reorder
    ReorderResult,
    generate_reorderings as lib_generate_reorderings,
    generate_label_maps as lib_generate_label_maps,
    generate_reorderings_with_variants as lib_generate_reorderings_with_variants,
    # Benchmark
    BenchmarkResult,
    run_benchmark as lib_run_benchmark,
    parse_benchmark_output as lib_parse_benchmark_output,
    # Cache
    CacheResult,
    run_cache_simulation as lib_run_cache_simulation,
    run_cache_simulations as lib_run_cache_simulations,
    parse_cache_output as lib_parse_cache_output,
    # Weights (imported directly - removes need for local duplicates)
    PerceptronWeight,
    load_type_registry,
    save_type_registry,
    assign_graph_type,
    update_type_weights_incremental,
    get_best_algorithm_for_type,
    list_known_types,
    get_type_weights_file,
    load_type_weights,
    save_type_weights,
    get_type_summary,
    CLUSTER_DISTANCE_THRESHOLD,
    # Progress
    ProgressTracker,
    create_progress as lib_create_progress,
    format_duration as lib_format_duration,
    # Results
    ResultsManager,
    read_json as lib_read_json,
    write_json as lib_write_json,
    filter_results as lib_filter_results,
    # Analysis
    SubcommunityInfo,
    AdaptiveOrderResult,
    AdaptiveComparisonResult,
    GraphBruteForceAnalysis,
    parse_adaptive_output as lib_parse_adaptive_output,
    analyze_adaptive_order as lib_analyze_adaptive_order,
    compare_adaptive_vs_fixed as lib_compare_adaptive_vs_fixed,
    run_subcommunity_brute_force as lib_run_subcommunity_brute_force,
    # Training
    TrainingResult,
    TrainingIterationResult,
    initialize_enhanced_weights as lib_initialize_enhanced_weights,
    train_adaptive_weights_iterative as lib_train_adaptive_weights_iterative,
    train_adaptive_weights_large_scale as lib_train_adaptive_weights_large_scale,
    # Phase orchestration
    PhaseConfig,
    run_reorder_phase,
    run_benchmark_phase,
    run_cache_phase,
    run_weights_phase,
    run_fill_weights_phase,
    run_adaptive_analysis_phase,
    run_comparison_phase,
    run_brute_force_phase,
    run_training_phase,
    run_large_scale_training_phase,
    run_full_pipeline,
)

# Try to import dependency manager
try:
    from scripts.lib.dependencies import (
        check_dependencies as lib_check_dependencies,
        install_dependencies as lib_install_dependencies,
        print_install_instructions as lib_print_install_instructions,
    )
    HAS_DEPENDENCY_MANAGER = True
except ImportError:
    HAS_DEPENDENCY_MANAGER = False

# ============================================================================
# Configuration
# ============================================================================

# Algorithm definitions
ALGORITHMS = {
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
    10: "CORDER",
    11: "RCM",
    12: "GraphBrewOrder",
    # 13: MAP - uses external file
    14: "AdaptiveOrder",
    # Leiden algorithms (15-17) - grouped together for easier sweeping
    # Format: 15:resolution
    15: "LeidenOrder",
    # Format: 16:variant:resolution where variant = dfs/dfshub/dfssize/bfs/hybrid
    16: "LeidenDendrogram",
    # Format: 17:variant:resolution:iterations:passes
    # Resolution: fixed (e.g., 1.5), auto, 0, dynamic, dynamic_2.0
    17: "LeidenCSR",
}

# Algorithms to benchmark (excluding MAP=13)
BENCHMARK_ALGORITHMS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17]

# Subset of key algorithms for quick testing
KEY_ALGORITHMS = [0, 1, 7, 8, 9, 11, 15, 16, 17]

# Algorithms known to be slow on large graphs
SLOW_ALGORITHMS = {9, 10, 11}  # GORDER, CORDER, RCM

# RabbitOrder variant configurations (default: csr)
# Format: -o 8:variant where variant = csr (default, native CSR) or boost (original Boost-based)
RABBITORDER_VARIANTS = ["csr", "boost"]
RABBITORDER_DEFAULT_VARIANT = "csr"

# GraphBrewOrder variant configurations (default: leiden for backward compat)
# Format: -o 12:cluster_variant:final_algo:resolution:levels
# cluster_variant: leiden (default), gve, gveopt, gvefast, gveoptfast, rabbit, hubcluster
GRAPHBREW_VARIANTS = ["leiden", "gve", "gveopt", "gvefast", "gveoptfast", "rabbit", "hubcluster"]
GRAPHBREW_DEFAULT_VARIANT = "leiden"  # Original Leiden library (backward compatible)

# Leiden variant configurations for sweeping
LEIDEN_DENDROGRAM_VARIANTS = ["dfs", "dfshub", "dfssize", "bfs", "hybrid"]
# LeidenCSR variants - gve (GVE-Leiden) is default for best modularity quality
# gveopt is cache-optimized with prefetching and flat arrays for large graphs
# gveopt2 uses CSR-based aggregation (fastest reordering, best PR performance)
# gveadaptive dynamically adjusts resolution at each pass based on runtime metrics
# gveoptsort uses LeidenOrder-style multi-level sort ordering
# gveturbo is speed-optimized (optional refinement skip)
# gvefast uses CSR buffer reuse (leiden.hxx style aggregation)
# gvedendo/gveoptdendo: RabbitOrder-inspired incremental dendrogram building
# gverabbit is GVE-Rabbit hybrid (fastest, good quality)
LEIDEN_CSR_VARIANTS = [
    "gve", "gveopt", "gveopt2", "gveadaptive", "gveoptsort", "gveturbo",
    "gvefast", "gvedendo", "gveoptdendo", "gverabbit", "dfs", "bfs", "hubsort", "modularity"
]
LEIDEN_CSR_DEFAULT_VARIANT = "gve"

# Recommended variants for different use cases
LEIDEN_CSR_FAST_VARIANTS = ["gveopt2", "gveadaptive", "gveturbo", "gvefast", "gverabbit"]  # Speed priority
LEIDEN_CSR_QUALITY_VARIANTS = ["gve", "gveopt", "gveopt2", "gveadaptive"]  # Quality priority

# Resolution modes for LeidenCSR
# - Fixed: numeric value (e.g., "1.5")
# - Auto: "auto" or "0" (compute from graph density/CV)
# - Dynamic: "dynamic" (adjust per-pass, gveadaptive only)
# - Dynamic+Init: "dynamic_2.0" (start at 2.0, adjust per-pass)
LEIDEN_RESOLUTION_MODES = ["auto", "dynamic", "1.0", "1.5", "2.0"]

# Default Leiden parameters
LEIDEN_DEFAULT_RESOLUTION = "auto"  # Auto-compute from graph (or use "dynamic", "1.0", etc)
LEIDEN_DEFAULT_PASSES = 3

# ============================================================================
# Algorithm Configuration with Variant Support
# ============================================================================

@dataclass
class AlgorithmConfig:
    """Configuration for an algorithm, including variant support."""
    algo_id: int           # Base algorithm ID (e.g., 17 for LeidenCSR)
    name: str              # Display name (e.g., "LeidenCSR_gve")
    option_string: str     # Full option string for -o flag (e.g., "17:gve:1.0:20:10")
    variant: str = ""      # Variant name if applicable (e.g., "gve")
    resolution: float = 1.0
    passes: int = 10
    
    @property
    def base_name(self) -> str:
        """Get base algorithm name without variant suffix."""
        return ALGORITHMS.get(self.algo_id, f"ALGO_{self.algo_id}")


def expand_algorithms_with_variants(
    algorithms: List[int],
    expand_leiden_variants: bool = False,
    leiden_resolution: str = LEIDEN_DEFAULT_RESOLUTION,
    leiden_passes: int = LEIDEN_DEFAULT_PASSES,
    leiden_csr_variants: List[str] = None,
    leiden_dendrogram_variants: List[str] = None,
    rabbit_variants: List[str] = None,
    graphbrew_variants: List[str] = None
) -> List[AlgorithmConfig]:
    """
    Expand algorithm IDs into AlgorithmConfig objects.
    
    For Leiden algorithms (16, 17), optionally expand into their variants.
    For RabbitOrder (8), optionally expand into csr/boost variants.
    For GraphBrewOrder (12), optionally expand into leiden/gve/gveopt/rabbit/hubcluster variants.
    
    Args:
        algorithms: List of algorithm IDs
        expand_leiden_variants: If True, expand variant algorithms into all their variants
        leiden_resolution: Resolution parameter for Leiden algorithms
        leiden_passes: Number of passes for LeidenCSR
        leiden_csr_variants: Which LeidenCSR variants to include (default: all)
        leiden_dendrogram_variants: Which LeidenDendrogram variants to include (default: all)
        rabbit_variants: Which RabbitOrder variants to include (default: csr only)
        graphbrew_variants: Which GraphBrewOrder variants to include (default: leiden only)
    
    Returns:
        List of AlgorithmConfig objects
    """
    if leiden_csr_variants is None:
        leiden_csr_variants = LEIDEN_CSR_VARIANTS
    if leiden_dendrogram_variants is None:
        leiden_dendrogram_variants = LEIDEN_DENDROGRAM_VARIANTS
    if rabbit_variants is None:
        # When expand_leiden_variants is True (--all-variants), include both RabbitOrder variants
        rabbit_variants = RABBITORDER_VARIANTS if expand_leiden_variants else [RABBITORDER_DEFAULT_VARIANT]
    if graphbrew_variants is None:
        # When expand_leiden_variants is True (--all-variants), include all GraphBrewOrder variants
        graphbrew_variants = GRAPHBREW_VARIANTS if expand_leiden_variants else [GRAPHBREW_DEFAULT_VARIANT]
    
    configs = []
    
    for algo_id in algorithms:
        base_name = ALGORITHMS.get(algo_id, f"ALGO_{algo_id}")
        
        if algo_id == 17 and expand_leiden_variants:
            # LeidenCSR: expand into variants
            # Format: 17:variant (let C++ use auto-resolution)
            for variant in leiden_csr_variants:
                option_str = f"{algo_id}:{variant}"
                configs.append(AlgorithmConfig(
                    algo_id=algo_id,
                    name=f"LeidenCSR_{variant}",
                    option_string=option_str,
                    variant=variant,
                    resolution=-1.0,  # -1 indicates auto-resolution
                    passes=leiden_passes
                ))
        elif algo_id == 16 and expand_leiden_variants:
            # LeidenDendrogram: expand into variants
            # Format: 16:variant (let C++ use auto-resolution)
            for variant in leiden_dendrogram_variants:
                option_str = f"{algo_id}:{variant}"
                configs.append(AlgorithmConfig(
                    algo_id=algo_id,
                    name=f"LeidenDendrogram_{variant}",
                    option_string=option_str,
                    variant=variant,
                    resolution=-1.0  # -1 indicates auto-resolution
                ))
        elif algo_id == 8 and expand_leiden_variants and len(rabbit_variants) > 1:
            # RabbitOrder: expand into variants if multiple specified
            for variant in rabbit_variants:
                option_str = f"{algo_id}:{variant}"
                configs.append(AlgorithmConfig(
                    algo_id=algo_id,
                    name=f"RABBITORDER_{variant}",
                    option_string=option_str,
                    variant=variant
                ))
        elif algo_id == 12 and expand_leiden_variants:
            # GraphBrewOrder: expand into clustering variants
            # Format: 12:cluster_variant:final_algo:resolution:levels
            for variant in graphbrew_variants:
                # Default: final_algo=8 (RabbitOrder), resolution=1.0, levels=2
                option_str = f"{algo_id}:{variant}"
                configs.append(AlgorithmConfig(
                    algo_id=algo_id,
                    name=f"GraphBrewOrder_{variant}",
                    option_string=option_str,
                    variant=variant,
                    resolution=leiden_resolution
                ))
        elif algo_id == 12:
            # GraphBrewOrder: use specified variant (default: leiden)
            variant = graphbrew_variants[0] if graphbrew_variants else GRAPHBREW_DEFAULT_VARIANT
            option_str = f"{algo_id}:{variant}"
            configs.append(AlgorithmConfig(
                algo_id=algo_id,
                name=f"GraphBrewOrder_{variant}",
                option_string=option_str,
                variant=variant,
                resolution=-1.0  # -1 indicates auto-resolution
            ))
        elif algo_id == 8:
            # RabbitOrder: use specified variant (default: csr)
            variant = rabbit_variants[0] if rabbit_variants else RABBITORDER_DEFAULT_VARIANT
            option_str = f"{algo_id}:{variant}"
            configs.append(AlgorithmConfig(
                algo_id=algo_id,
                name=base_name,
                option_string=option_str,
                variant=variant
            ))
        elif algo_id == 15:
            # LeidenOrder: no parameters (use auto-resolution)
            option_str = f"{algo_id}"
            configs.append(AlgorithmConfig(
                algo_id=algo_id,
                name=base_name,
                option_string=option_str,
                resolution=-1.0  # -1 indicates auto-resolution
            ))
        else:
            # Non-Leiden algorithms: just use ID
            configs.append(AlgorithmConfig(
                algo_id=algo_id,
                name=base_name,
                option_string=str(algo_id)
            ))
    
    return configs


def get_algorithm_config_by_name(name: str, configs: List[AlgorithmConfig]) -> Optional[AlgorithmConfig]:
    """Find an AlgorithmConfig by name."""
    for cfg in configs:
        if cfg.name == name:
            return cfg
    return None


def get_best_leiden_variant(
    type_name: str,
    base_algo_id: int,
    benchmark: str = 'pr',
    weights_dir: str = None
) -> Optional[str]:
    """
    Get the best variant for a Leiden algorithm based on learned weights.
    
    For LeidenCSR (17), variants are: gve (default), gveopt, dfs, bfs, hubsort, fast, modularity
    For LeidenDendrogram (16), variants are: dfs, dfshub, dfssize, bfs, hybrid
    
    Args:
        type_name: Graph type (e.g., 'type_0')
        base_algo_id: Algorithm ID (16 or 17)
        benchmark: Benchmark to optimize for
        weights_dir: Directory containing type weights
    
    Returns:
        Best variant name or None if no data available
    """
    if weights_dir is None:
        weights_dir = DEFAULT_WEIGHTS_DIR
    
    # Get variants for this algorithm
    if base_algo_id == 17:
        base_name = "LeidenCSR"
        variants = LEIDEN_CSR_VARIANTS
    elif base_algo_id == 16:
        base_name = "LeidenDendrogram"
        variants = LEIDEN_DENDROGRAM_VARIANTS
    else:
        return None  # Not a Leiden algorithm
    
    # Load type weights
    weights = load_type_weights(type_name, weights_dir)
    if not weights:
        return variants[0]  # Default to first variant if no weights
    
    # Find best variant based on win rate and average speedup
    best_variant = None
    best_score = float('-inf')
    
    for variant in variants:
        variant_name = f"{base_name}_{variant}"
        if variant_name in weights:
            algo_weights = weights[variant_name]
            meta = algo_weights.get('_metadata', {})
            
            # Score = win_rate * 0.7 + normalized_avg_speedup * 0.3
            win_rate = meta.get('win_rate', 0.0)
            avg_speedup = meta.get('avg_speedup', 1.0)
            
            # Benchmark-specific bonus
            bench_weights = algo_weights.get('benchmark_weights', {})
            bench_bonus = bench_weights.get(benchmark.lower(), 1.0) - 1.0
            
            score = win_rate * 0.7 + (avg_speedup - 1.0) * 0.3 + bench_bonus * 0.1
            
            if score > best_score:
                best_score = score
                best_variant = variant
    
    return best_variant if best_variant else variants[0]


def get_leiden_variant_rankings(
    type_name: str,
    base_algo_id: int,
    benchmark: str = 'pr',
    weights_dir: str = None
) -> List[Tuple[str, float]]:
    """
    Get ranked list of Leiden variants with their scores.
    
    Args:
        type_name: Graph type
        base_algo_id: Algorithm ID (16 or 17)
        benchmark: Benchmark to optimize for
        weights_dir: Directory containing type weights
    
    Returns:
        List of (variant_name, score) tuples, sorted by score descending
    """
    if weights_dir is None:
        weights_dir = DEFAULT_WEIGHTS_DIR
    
    if base_algo_id == 17:
        base_name = "LeidenCSR"
        variants = LEIDEN_CSR_VARIANTS
    elif base_algo_id == 16:
        base_name = "LeidenDendrogram"
        variants = LEIDEN_DENDROGRAM_VARIANTS
    else:
        return []
    
    weights = load_type_weights(type_name, weights_dir)
    rankings = []
    
    for variant in variants:
        variant_name = f"{base_name}_{variant}"
        if variant_name in weights:
            algo_weights = weights[variant_name]
            meta = algo_weights.get('_metadata', {})
            
            win_rate = meta.get('win_rate', 0.0)
            avg_speedup = meta.get('avg_speedup', 1.0)
            bench_weights = algo_weights.get('benchmark_weights', {})
            bench_bonus = bench_weights.get(benchmark.lower(), 1.0) - 1.0
            
            score = win_rate * 0.7 + (avg_speedup - 1.0) * 0.3 + bench_bonus * 0.1
            rankings.append((variant_name, score))
        else:
            rankings.append((variant_name, 0.0))  # No data yet
    
    rankings.sort(key=lambda x: x[1], reverse=True)
    return rankings


# Benchmarks to run
BENCHMARKS = ["pr", "bfs", "cc", "sssp", "bc"]

# Benchmarks that are computationally intensive in simulation
HEAVY_SIM_BENCHMARKS = {"bc", "sssp"}

# Default paths - ALL outputs go to results/
DEFAULT_RESULTS_DIR = "./results"
DEFAULT_GRAPHS_DIR = "./results/graphs"
DEFAULT_BIN_DIR = "./bench/bin"
DEFAULT_BIN_SIM_DIR = "./bench/bin_sim"
DEFAULT_WEIGHTS_DIR = "./scripts/weights/active"  # Active weights (C++ reads from here)
DEFAULT_MAPPINGS_DIR = "./results/mappings"

# Auto-clustering configuration
CLUSTER_DISTANCE_THRESHOLD = 0.15  # Max normalized distance to join existing cluster (lower = more clusters)
MIN_SAMPLES_FOR_CLUSTER = 2  # Minimum graphs to form a stable cluster

# Graph size categories (MB)
SIZE_SMALL = 50
SIZE_MEDIUM = 500
SIZE_LARGE = 2000

# Minimum edges for training (skip small graphs that introduce noise/skew)
MIN_EDGES_FOR_TRAINING = 100000  # 100K edges - graphs below this are too noisy for perceptron training

# Memory estimation constants (bytes per edge/node for graph algorithms)
# Based on CSR format: ~24 bytes/edge + 8 bytes/node for working memory
BYTES_PER_EDGE = 24
BYTES_PER_NODE = 8
MEMORY_SAFETY_FACTOR = 1.5  # Add 50% buffer for algorithm overhead

# Default timeouts (seconds)
TIMEOUT_REORDER = 43200     # 12 hours for reordering (some algorithms like GORDER are slow on large graphs)
TIMEOUT_BENCHMARK = 600     # 10 min for benchmarks
TIMEOUT_SIM = 1200          # 20 min for simulations
TIMEOUT_SIM_HEAVY = 3600    # 1 hour for heavy simulations

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class GraphInfo:
    """Information about a graph dataset."""
    name: str
    path: str
    size_mb: float
    is_symmetric: bool = True
    nodes: int = 0
    edges: int = 0

# ============================================================================
# Graph Feature Functions (from lib/features.py)
# ============================================================================
# All graph feature computation functions are now imported from lib/features.py:
# - Graph type constants: GRAPH_TYPE_*, ALL_GRAPH_TYPES
# - Memory constants: BYTES_PER_EDGE, BYTES_PER_NODE, MEMORY_SAFETY_FACTOR
# - Graph properties cache: load_graph_properties_cache, save_graph_properties_cache,
#   update_graph_properties, get_graph_properties
# - Graph type detection: detect_graph_type, get_graph_type_from_name, get_graph_type_from_properties
# - Topological features: compute_clustering_coefficient_sample, estimate_diameter_bfs,
#   count_subcommunities_quick, compute_extended_features
# - System utilities: get_available_memory_gb, get_total_memory_gb, estimate_graph_memory_gb,
#   get_available_disk_gb, get_total_disk_gb, get_num_threads


# =============================================================================
# Data Classes (from lib/)
# =============================================================================
# Core data classes are imported from lib/. BenchmarkResult comes from utils.py


class PerceptronWeightExtended(PerceptronWeight):
    """Extended PerceptronWeight with compute_score and benchmark_weights."""
    
    def __init__(self, *args, benchmark_weights: Dict[str, float] = None, 
                 _metadata: Dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.benchmark_weights = benchmark_weights or {'pr': 1.0, 'bfs': 1.0, 'cc': 1.0, 'sssp': 1.0, 'bc': 1.0}
        self._metadata = _metadata or {}
    
    def compute_score(self, features: Dict, benchmark: str = 'pr') -> float:
        """Compute perceptron score for given features and benchmark."""
        log_nodes = math.log10(features.get('nodes', 1) + 1)
        log_edges = math.log10(features.get('edges', 1) + 1)
        
        score = self.bias
        score += self.w_modularity * features.get('modularity', 0.5)
        score += self.w_log_nodes * log_nodes
        score += self.w_log_edges * log_edges
        score += self.w_density * features.get('density', 0.0)
        score += self.w_avg_degree * features.get('avg_degree', 0.0) / 100.0
        score += self.w_degree_variance * features.get('degree_variance', 0.0)
        score += self.w_hub_concentration * features.get('hub_concentration', 0.0)
        score += getattr(self, 'w_clustering_coeff', 0.0) * features.get('clustering_coefficient', 0.0)
        score += getattr(self, 'w_avg_path_length', 0.0) * features.get('avg_path_length', 0.0) / 10.0
        score += getattr(self, 'w_diameter', 0.0) * features.get('diameter', features.get('diameter_estimate', 0.0)) / 50.0
        score += getattr(self, 'w_community_count', 0.0) * math.log10(features.get('community_count', 1) + 1)
        score += self.w_reorder_time * features.get('reorder_time', 0.0)
        
        # Apply benchmark-specific multiplier
        bench_mult = self.benchmark_weights.get(benchmark.lower(), 1.0) if self.benchmark_weights else 1.0
        return score * bench_mult


# Note: These are imported from lib/ directly:
# - ReorderResult, CacheResult, SubcommunityInfo
# - AdaptiveOrderResult, AdaptiveComparisonResult
# - PerceptronWeight, ProgressTracker, ResultsManager
# - TrainingResult, TrainingIterationResult, GraphBruteForceAnalysis
# - DownloadableGraph, DOWNLOAD_GRAPHS_*

# ============================================================================
# System Utilities (from lib/features.py)
# ============================================================================
# These functions are now imported from lib/features.py:
# - get_available_memory_gb, get_total_memory_gb, estimate_graph_memory_gb
# - get_available_disk_gb, get_total_disk_gb, get_num_threads
# - BYTES_PER_EDGE, BYTES_PER_NODE, MEMORY_SAFETY_FACTOR

# ============================================================================
# Type System (using lib/weights.py)
# ============================================================================
# All type system functions are now imported from lib/weights.py:
# - load_type_registry, save_type_registry
# - assign_graph_type, update_type_weights_incremental
# - get_best_algorithm_for_type, list_known_types
# - load_type_weights, save_type_weights, get_type_weights_file
# - get_type_summary, CLUSTER_DISTANCE_THRESHOLD


# ============================================================================
# Graph Catalog and Progress Tracking (from lib/)
# ============================================================================
# These are imported directly from lib/download.py and lib/progress.py:
# - DOWNLOAD_GRAPHS_SMALL, DOWNLOAD_GRAPHS_MEDIUM, DOWNLOAD_GRAPHS_LARGE, DOWNLOAD_GRAPHS_XLARGE
# - DownloadableGraph
# - ProgressTracker

# Global progress tracker instance
_progress = ProgressTracker()


# ============================================================================
# Utility Functions
# ============================================================================

def log(msg: str, level: str = "INFO"):
    """Print a timestamped log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    level_colors = {
        'INFO': '',
        'WARN': '\033[93m',
        'ERROR': '\033[91m',
        'SUCCESS': '\033[92m',
    }
    color = level_colors.get(level, '')
    end_color = '\033[0m' if color else ''
    print(f"[{timestamp}] {color}[{level}]{end_color} {msg}")

def log_section(title: str):
    """Print a section header."""
    _progress.phase_start(title)

# NOTE: Legacy backup_and_sync_weights removed - now using type-based weights in scripts/weights/

def get_graph_path(graphs_dir: str, graph_name: str) -> Optional[str]:
    """Get the path to a graph file."""
    graph_folder = os.path.join(graphs_dir, graph_name)
    
    # Try variations of the graph name (hyphen vs underscore)
    name_variants = [graph_name, graph_name.replace('-', '_'), graph_name.replace('_', '-')]
    
    # Check for graph files with the graph name (downloaded format)
    for name in name_variants:
        for ext in [".mtx", ".el", ".sg"]:
            path = os.path.join(graph_folder, f"{name}{ext}")
            if os.path.exists(path):
                return path
    
    # Check for generic "graph" name (legacy format)
    for ext in [".mtx", ".el", ".sg"]:
        path = os.path.join(graph_folder, f"graph{ext}")
        if os.path.exists(path):
            return path
    
    # Try direct path (file directly in graphs_dir)
    direct = os.path.join(graphs_dir, graph_name)
    if os.path.isfile(direct):
        return direct
    
    # Look in subdirectories (e.g., germany-osm/germany_osm/germany_osm.mtx)
    if os.path.isdir(graph_folder):
        for subdir in os.listdir(graph_folder):
            subdir_path = os.path.join(graph_folder, subdir)
            if os.path.isdir(subdir_path):
                for name in name_variants + [subdir]:
                    for ext in [".mtx", ".el", ".sg"]:
                        path = os.path.join(subdir_path, f"{name}{ext}")
                        if os.path.exists(path):
                            return path
    
    # Last resort: look for any .mtx file in the folder (excluding auxiliary files)
    if os.path.isdir(graph_folder):
        for f in sorted(os.listdir(graph_folder)):
            # Skip auxiliary files (coords, nodenames, categories, etc.)
            if f.endswith('.mtx') and not any(skip in f for skip in ['_coord', '_nodename', '_Categories', '_b.mtx']):
                full_path = os.path.join(graph_folder, f)
                if os.path.isfile(full_path):
                    return full_path
    
    return None

def get_graph_size_mb(path: str) -> float:
    """Get the size of a graph file in MB."""
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0.0


def get_graph_dimensions(path: str) -> Tuple[int, int]:
    """Read nodes and edges count from an MTX file header.
    
    Returns:
        (nodes, edges) tuple, or (0, 0) if unable to read
    """
    try:
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('%'):
                    continue
                # First non-comment line has dimensions: rows cols nnz
                parts = line.strip().split()
                if len(parts) >= 3:
                    nodes = int(parts[0])
                    edges = int(parts[2])
                    return nodes, edges
                elif len(parts) >= 2:
                    nodes = int(parts[0])
                    return nodes, 0
                break
    except:
        pass
    return 0, 0


def discover_graphs(graphs_dir: str, min_size: float = 0, max_size: float = float('inf'), 
                    additional_dirs: List[str] = None, max_memory_gb: float = None,
                    min_edges: int = 0) -> List[GraphInfo]:
    """Discover available graphs in the directory and additional directories.
    
    Args:
        graphs_dir: Primary directory to scan for graphs
        min_size: Minimum graph size in MB
        max_size: Maximum graph size in MB
        additional_dirs: Additional directories to scan (e.g., ./graphs for pre-existing graphs)
        max_memory_gb: If set, skip graphs requiring more than this memory (auto-detects if None)
        min_edges: Minimum number of edges (skip smaller graphs - they introduce noise in training)
    """
    graphs = []
    seen_names = set()
    skipped_memory = []
    skipped_edges = []
    
    # Build list of directories to scan
    dirs_to_scan = [graphs_dir]
    if additional_dirs:
        dirs_to_scan.extend(additional_dirs)
    
    # Also check ./graphs if it exists and isn't already in the list
    legacy_graphs_dir = "./results/graphs"
    if os.path.exists(legacy_graphs_dir) and legacy_graphs_dir not in dirs_to_scan:
        dirs_to_scan.append(legacy_graphs_dir)
    
    for scan_dir in dirs_to_scan:
        if not os.path.exists(scan_dir):
            continue
            
        # Check graphs.json for metadata
        metadata_file = os.path.join(scan_dir, "graphs.json")
        metadata = {}
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
            except:
                pass
        
        # Scan directory
        try:
            entries = os.listdir(scan_dir)
        except OSError:
            continue
            
        for entry in entries:
            # Skip if already seen (first occurrence wins)
            if entry in seen_names:
                continue
                
            entry_path = os.path.join(scan_dir, entry)
            if os.path.isdir(entry_path):
                graph_path = get_graph_path(scan_dir, entry)
                if graph_path:
                    size_mb = get_graph_size_mb(graph_path)
                    if min_size <= size_mb <= max_size:
                        is_symmetric = metadata.get(entry, {}).get("symmetric", True)
                        
                        # Read actual node/edge counts from MTX file
                        nodes, edges = get_graph_dimensions(graph_path)
                        
                        # Skip graphs with too few edges (they introduce noise in training)
                        if min_edges > 0 and edges > 0 and edges < min_edges:
                            skipped_edges.append((entry, edges))
                            continue
                        
                        # Check memory requirements if limit specified
                        if max_memory_gb is not None and nodes > 0 and edges > 0:
                            mem_required = estimate_graph_memory_gb(nodes, edges)
                            if mem_required > max_memory_gb:
                                skipped_memory.append((entry, mem_required))
                                continue
                        
                        graphs.append(GraphInfo(
                            name=entry,
                            path=graph_path,
                            size_mb=size_mb,
                            is_symmetric=is_symmetric,
                            nodes=nodes,
                            edges=edges
                        ))
                        seen_names.add(entry)
    
    # Report skipped graphs
    if skipped_memory:
        log(f"Skipped {len(skipped_memory)} graphs exceeding {max_memory_gb:.1f}GB memory limit:", "INFO")
        for name, mem in skipped_memory[:5]:
            log(f"  - {name}: requires {mem:.1f} GB", "INFO")
        if len(skipped_memory) > 5:
            log(f"  ... and {len(skipped_memory) - 5} more", "INFO")
    
    if skipped_edges:
        log(f"Skipped {len(skipped_edges)} graphs with fewer than {min_edges:,} edges:", "INFO")
        for name, edge_count in skipped_edges[:5]:
            log(f"  - {name}: {edge_count:,} edges", "INFO")
        if len(skipped_edges) > 5:
            log(f"  ... and {len(skipped_edges) - 5} more", "INFO")
    
    # Sort by size
    graphs.sort(key=lambda g: g.size_mb)
    return graphs

# ============================================================================
# Graph Download Functions
# ============================================================================

def download_file(url: str, dest_path: str, show_progress: bool = True) -> bool:
    """Download a file from URL with optional progress indicator."""
    try:
        import urllib.request
        import urllib.error
        
        # Create parent directory if needed
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        if show_progress:
            print(f"  Downloading from {url}")
        
        # Get file size
        try:
            with urllib.request.urlopen(url) as response:
                total_size = int(response.headers.get('content-length', 0))
                
                # Download with progress
                downloaded = 0
                chunk_size = 8192
                
                with open(dest_path, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if show_progress and total_size > 0:
                            percent = downloaded * 100 // total_size
                            print(f"\r  Progress: {percent}% ({downloaded // 1024 // 1024}MB / {total_size // 1024 // 1024}MB)", end="")
                
                if show_progress:
                    print()  # New line after progress
                    
        except urllib.error.URLError as e:
            print(f"  Error downloading: {e}")
            return False
            
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        return False

def extract_archive(archive_path: str, extract_to: str) -> bool:
    """Extract a tar.gz archive."""
    try:
        import tarfile
        import gzip
        
        print(f"  Extracting {os.path.basename(archive_path)}")
        
        if archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(path=extract_to)
        elif archive_path.endswith('.gz'):
            # Single gzip file
            out_path = archive_path[:-3]  # Remove .gz
            with gzip.open(archive_path, 'rb') as f_in:
                with open(out_path, 'wb') as f_out:
                    f_out.write(f_in.read())
        else:
            print(f"  Unknown archive format: {archive_path}")
            return False
            
        return True
        
    except Exception as e:
        print(f"  Error extracting: {e}")
        return False

def validate_mtx_file(mtx_path: str) -> bool:
    """Validate that a .mtx file is readable and has the expected format."""
    try:
        with open(mtx_path, 'r') as f:
            # Check header
            for line in f:
                if line.startswith('%'):
                    continue
                # First non-comment line should have dimensions
                parts = line.strip().split()
                if len(parts) >= 2:
                    rows = int(parts[0])
                    cols = int(parts[1])
                    if rows > 0 and cols > 0:
                        return True
                break
        return False
    except Exception as e:
        print(f"  Validation error: {e}")
        return False

def download_graph(graph: DownloadableGraph, graphs_dir: str, force: bool = False) -> bool:
    """Download a single graph from SuiteSparse collection."""
    
    graph_dir = os.path.join(graphs_dir, graph.name)
    mtx_file = os.path.join(graph_dir, f"{graph.name}.mtx")
    
    # Check if already exists
    if os.path.exists(mtx_file) and not force:
        print(f"  {graph.name}: Already exists (skip)")
        return True
    
    print(f"\n  Downloading {graph.name} ({graph.size_mb:.1f} MB)")
    
    # Create directory
    os.makedirs(graph_dir, exist_ok=True)
    
    # Use the pre-defined URL from the graph catalog
    url = graph.url
    
    # Download archive
    archive_path = os.path.join(graph_dir, f"{graph.name}.tar.gz")
    if not download_file(url, archive_path):
        return False
    
    # Extract archive
    if not extract_archive(archive_path, graph_dir):
        return False
    
    # The archive extracts to a subdirectory, move files up if needed
    extracted_dir = os.path.join(graph_dir, graph.name)
    if os.path.isdir(extracted_dir):
        for item in os.listdir(extracted_dir):
            src = os.path.join(extracted_dir, item)
            dst = os.path.join(graph_dir, item)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        # Remove empty extracted directory
        try:
            os.rmdir(extracted_dir)
        except:
            pass
    
    # Clean up archive
    try:
        os.remove(archive_path)
    except:
        pass
    
    # Validate
    if not os.path.exists(mtx_file):
        print(f"  Error: {graph.name}.mtx not found after extraction")
        return False
    
    if not validate_mtx_file(mtx_file):
        print(f"  Error: {graph.name}.mtx is invalid")
        return False
    
    print(f"  {graph.name}: Downloaded and validated successfully")
    return True

def download_graphs(
    size_category: str = "SMALL",
    graphs_dir: str = DEFAULT_GRAPHS_DIR,
    force: bool = False,
    max_memory_gb: float = None,
    max_disk_gb: float = None,
    parallel: bool = True,
    max_workers: int = 4,
) -> List[str]:
    """Download graphs by size category with optional memory and disk filtering.
    
    Downloads are performed in parallel to optimize bandwidth utilization.
    The function blocks until ALL downloads complete before returning,
    ensuring graphs are ready before running experiments.
    
    Args:
        size_category: One of "SMALL", "MEDIUM", "LARGE", "XLARGE", "ALL"
        graphs_dir: Directory to download graphs to
        force: If True, re-download existing graphs
        max_memory_gb: If set, skip graphs exceeding this memory requirement
        max_disk_gb: If set, skip downloads that would exceed this disk space
        parallel: If True, download graphs in parallel (default: True)
        max_workers: Number of parallel download threads (default: 4)
        
    Returns:
        List of successfully downloaded graph names
    """
    # Select graphs based on category
    graphs_to_download = []
    
    if size_category.upper() in ["SMALL", "ALL"]:
        graphs_to_download.extend(DOWNLOAD_GRAPHS_SMALL)
    if size_category.upper() in ["MEDIUM", "ALL"]:
        graphs_to_download.extend(DOWNLOAD_GRAPHS_MEDIUM)
    if size_category.upper() in ["LARGE", "ALL"]:
        graphs_to_download.extend(DOWNLOAD_GRAPHS_LARGE)
    if size_category.upper() in ["XLARGE", "ALL"]:
        graphs_to_download.extend(DOWNLOAD_GRAPHS_XLARGE)
    
    if not graphs_to_download:
        print(f"Unknown size category: {size_category}")
        print("Valid options: SMALL, MEDIUM, LARGE, XLARGE, ALL")
        return []
    
    # Apply memory filtering if specified
    skipped_memory = []
    if max_memory_gb is not None:
        filtered = []
        for g in graphs_to_download:
            mem_required = g.estimated_memory_gb()
            if mem_required <= max_memory_gb:
                filtered.append(g)
            else:
                skipped_memory.append((g.name, mem_required))
        graphs_to_download = filtered
    
    # Apply disk space filtering if specified
    skipped_disk = []
    if max_disk_gb is not None:
        # Sort by size to download smaller graphs first
        graphs_to_download.sort(key=lambda g: g.size_mb)
        filtered = []
        cumulative_size_gb = 0
        for g in graphs_to_download:
            size_gb = g.size_mb / 1024
            if cumulative_size_gb + size_gb <= max_disk_gb:
                filtered.append(g)
                cumulative_size_gb += size_gb
            else:
                skipped_disk.append((g.name, g.size_mb))
        graphs_to_download = filtered
    
    if not graphs_to_download:
        print("No graphs to download after filtering")
        return []
    
    # Print filtering summary if any graphs were skipped
    if skipped_memory:
        print(f"\n  ⚠ Skipped {len(skipped_memory)} graphs exceeding memory limit ({max_memory_gb:.1f} GB)")
    if skipped_disk:
        print(f"\n  ⚠ Skipped {len(skipped_disk)} graphs exceeding disk limit ({max_disk_gb:.1f} GB)")
    
    # Create graphs directory
    os.makedirs(graphs_dir, exist_ok=True)
    
    # Use parallel download (blocks until all complete)
    if parallel:
        successful_paths, failed_names = lib_download_graphs_parallel(
            graphs=[g.name for g in graphs_to_download],
            dest_dir=Path(graphs_dir),
            max_workers=max_workers,
            force=force,
            show_progress=True,
            wait_for_all=True,  # Always wait for all downloads
        )
        successful = [p.parent.name for p in successful_paths]
    else:
        # Sequential fallback
        print("\n" + "="*60)
        print("  GRAPH DOWNLOAD (Sequential)")
        print("="*60)
        
        successful = []
        for i, graph in enumerate(graphs_to_download, 1):
            print(f"\n[{i}/{len(graphs_to_download)}] {graph.name}")
            if download_graph(graph, graphs_dir, force):
                successful.append(graph.name)
        
        print("\n" + "-"*40)
        print(f"Download complete: {len(successful)}/{len(graphs_to_download)} successful")
    
    return successful


def ensure_prerequisites(project_dir: str = ".", 
                         graphs_dir: str = DEFAULT_GRAPHS_DIR,
                         results_dir: str = DEFAULT_RESULTS_DIR,
                         weights_dir: str = DEFAULT_WEIGHTS_DIR,
                         rebuild: bool = False) -> bool:
    """
    Ensure all prerequisites are in place: directories, binaries, weights folder.
    
    This function is called at the start of any operation to ensure the environment
    is properly set up. If any component is missing, it attempts to create/build it.
    
    Args:
        project_dir: Root project directory
        graphs_dir: Directory for graph files
        results_dir: Directory for results
        weights_dir: Directory for type-based weight files
        rebuild: If True, force rebuild even if binaries exist
        
    Returns:
        True if all prerequisites are satisfied, False otherwise
    """
    log_section("Ensuring Prerequisites")
    success = True
    
    # 1. Ensure directories exist
    log("Checking directories...")
    required_dirs = [
        graphs_dir,
        results_dir,
        weights_dir,
        os.path.join(project_dir, "bench", "bin"),
        os.path.join(project_dir, "bench", "bin_sim"),
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            log(f"  Creating: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
        else:
            log(f"  OK: {dir_path}")
    
    # 2. Check and build binaries
    log("Checking binaries...")
    bin_dir = os.path.join(project_dir, "bench", "bin")
    bin_sim_dir = os.path.join(project_dir, "bench", "bin_sim")
    
    required_bins = ["bfs", "pr", "cc", "bc", "sssp", "tc"]
    
    # Check standard binaries
    missing_bins = []
    for binary in required_bins:
        bin_path = os.path.join(bin_dir, binary)
        if not os.path.exists(bin_path) or rebuild:
            missing_bins.append(binary)
    
    # Check simulation binaries
    missing_sim = []
    for binary in required_bins:
        sim_path = os.path.join(bin_sim_dir, binary)
        if not os.path.exists(sim_path) or rebuild:
            missing_sim.append(binary)
    
    # Build if needed
    if missing_bins or missing_sim or rebuild:
        makefile = os.path.join(project_dir, "Makefile")
        if not os.path.exists(makefile):
            log("ERROR: Makefile not found - cannot build binaries", "ERROR")
            return False
        
        if missing_bins or rebuild:
            log(f"  Building standard binaries: {', '.join(missing_bins) if missing_bins else 'all (rebuild requested)'}...")
            cmd = f"cd {project_dir} && make clean-bin && make -j$(nproc)" if rebuild else f"cd {project_dir} && make -j$(nproc)"
            success_build, stdout, stderr = run_command(cmd, timeout=600)
            if not success_build:
                log(f"  Build failed: {stderr[:200]}", "ERROR")
                success = False
            else:
                log("  Standard binaries built successfully")
        
        if missing_sim or rebuild:
            log(f"  Building simulation binaries: {', '.join(missing_sim) if missing_sim else 'all (rebuild requested)'}...")
            cmd = f"cd {project_dir} && make clean-sim && make all-sim -j$(nproc)" if rebuild else f"cd {project_dir} && make all-sim -j$(nproc)"
            success_build, stdout, stderr = run_command(cmd, timeout=600)
            if not success_build:
                log(f"  Simulation build failed: {stderr[:200]}", "ERROR")
                success = False
            else:
                log("  Simulation binaries built successfully")
    else:
        log("  All binaries present")
    
    # 3. Initialize type registry if needed
    log("Checking type registry...")
    load_type_registry(weights_dir)
    known_types = list_known_types(weights_dir)
    if known_types:
        log(f"  Loaded {len(known_types)} types: {', '.join(known_types)}")
    else:
        log("  No types yet - will be created as graphs are processed")
    
    # 4. Verify a binary actually works
    log("Verifying binaries work...")
    test_bin = os.path.join(bin_dir, "pr")
    if os.path.exists(test_bin):
        test_success, _, stderr = run_command(f"{test_bin} --help", timeout=5)
        if test_success or "Usage" in stderr or "pagerank" in stderr.lower():
            log("  Binary verification: OK")
        else:
            log("  Binary verification: WARNING - binary may not work correctly", "WARN")
    
    if success:
        log("All prerequisites satisfied", "INFO")
    else:
        log("Some prerequisites failed - continuing with available resources", "WARN")
    
    return success


def check_and_build_binaries(project_dir: str = ".") -> bool:
    """Check if binaries exist, build if necessary.
    
    Returns:
        True if binaries are ready, False otherwise
    """
    print("\n" + "="*60)
    print("BUILD CHECK")
    print("="*60)
    
    # Check for required binaries
    bin_dir = os.path.join(project_dir, "bench", "bin")
    bin_sim_dir = os.path.join(project_dir, "bench", "bin_sim")
    
    required_bins = ["bfs", "pr", "cc", "bc", "sssp", "tc"]
    
    # Check standard binaries
    missing_bins = []
    for binary in required_bins:
        bin_path = os.path.join(bin_dir, binary)
        if not os.path.exists(bin_path):
            missing_bins.append(binary)
    
    # Check simulation binaries
    missing_sim = []
    for binary in required_bins:
        sim_path = os.path.join(bin_sim_dir, binary)
        if not os.path.exists(sim_path):
            missing_sim.append(binary)
    
    if not missing_bins and not missing_sim:
        print("All binaries present - build OK")
        return True
    
    if missing_bins:
        print(f"Missing standard binaries: {', '.join(missing_bins)}")
    if missing_sim:
        print(f"Missing simulation binaries: {', '.join(missing_sim)}")
    
    # Try to build
    makefile = os.path.join(project_dir, "Makefile")
    if not os.path.exists(makefile):
        print("Error: Makefile not found - cannot build")
        return False
    
    print("\nBuilding binaries...")
    
    # Build standard binaries
    if missing_bins:
        print("  Building standard binaries...")
        success, stdout, stderr = run_command(f"cd {project_dir} && make -j$(nproc)", timeout=600)
        if not success:
            print(f"  Build failed: {stderr}")
            return False
        print("  Standard binaries built successfully")
    
    # Build simulation binaries
    if missing_sim:
        print("  Building simulation binaries...")
        success, stdout, stderr = run_command(f"cd {project_dir} && make all-sim -j$(nproc)", timeout=600)
        if not success:
            print(f"  Simulation build failed: {stderr}")
            return False
        print("  Simulation binaries built successfully")
    
    print("Build complete")
    return True

def clean_results(results_dir: str = DEFAULT_RESULTS_DIR, keep_graphs: bool = True, keep_weights: bool = True) -> None:
    """Clean the results directory, optionally keeping graphs.
    
    Args:
        results_dir: Directory to clean
        keep_graphs: If True, don't delete downloaded graphs
        keep_weights: (Deprecated) Kept for backward compatibility, no longer used
    """
    print("\n" + "="*60)
    print("CLEANING RESULTS DIRECTORY")
    print("="*60)
    
    if not os.path.exists(results_dir):
        print(f"Results directory does not exist: {results_dir}")
        return
    
    # Patterns to clean (relative to results_dir)
    patterns_to_clean = [
        "*.json",
        "*.log", 
        "*.csv",
        "mappings/",
        "logs/",
    ]
    
    # Items to keep
    keep_items = set()
    if keep_graphs:
        keep_items.add("graphs")
    
    deleted_count = 0
    kept_count = 0
    
    for entry in os.listdir(results_dir):
        entry_path = os.path.join(results_dir, entry)
        
        # Check if should keep
        if entry in keep_items:
            kept_count += 1
            print(f"  Keeping: {entry}")
            continue
        
        # Clean based on pattern
        if entry.endswith(('.json', '.log', '.csv')):
            try:
                os.remove(entry_path)
                deleted_count += 1
                print(f"  Deleted: {entry}")
            except Exception as e:
                print(f"  Error deleting {entry}: {e}")
        elif os.path.isdir(entry_path) and entry in ["mappings", "logs"]:
            try:
                shutil.rmtree(entry_path)
                deleted_count += 1
                print(f"  Deleted: {entry}/")
            except Exception as e:
                print(f"  Error deleting {entry}/: {e}")
    
    print(f"\nCleaned {deleted_count} items, kept {kept_count} items")

def clean_all(project_dir: str = ".", confirm: bool = False) -> None:
    """Clean all generated data for a fresh start.
    
    This removes:
    - All results (including graphs)
    - All label.map files in graph directories
    - Downloaded graphs
    - Compiled binaries (bench/bin/, bench/bin_sim/)
    
    Args:
        project_dir: Project root directory
        confirm: If True, skip confirmation prompt
    """
    if not confirm:
        response = input("This will delete ALL generated data including downloaded graphs and binaries. Continue? [y/N] ")
        if response.lower() != 'y':
            print("Cancelled")
            return
    
    print("\n" + "="*60)
    print("FULL CLEAN - REMOVING ALL GENERATED DATA")
    print("="*60)
    
    # Clean results directory completely
    results_dir = os.path.join(project_dir, "results")
    if os.path.exists(results_dir):
        print(f"Removing {results_dir}/")
        shutil.rmtree(results_dir)
        os.makedirs(results_dir)  # Recreate empty
        print("  Done")
    
    # Clean compiled binaries
    bin_dirs = [
        os.path.join(project_dir, "bench", "bin"),
        os.path.join(project_dir, "bench", "bin_sim"),
    ]
    for bin_dir in bin_dirs:
        if os.path.exists(bin_dir):
            # Remove all files in bin directory but keep the directory
            for f in os.listdir(bin_dir):
                fpath = os.path.join(bin_dir, f)
                if os.path.isfile(fpath):
                    os.remove(fpath)
            print(f"Cleaned binaries in {bin_dir}/")
    
    # Clean label.map files in graphs directory
    graphs_dir = os.path.join(project_dir, "results", "graphs")
    if os.path.exists(graphs_dir):
        map_files = glob.glob(os.path.join(graphs_dir, "**/label.map"), recursive=True)
        if map_files:
            print(f"Removing {len(map_files)} label.map files from graphs/")
            for map_file in map_files:
                try:
                    os.remove(map_file)
                except:
                    pass
            print("  Done")
    
    # Clean bench/results if exists
    bench_results = os.path.join(project_dir, "bench", "results")
    if os.path.exists(bench_results):
        print(f"Removing {bench_results}/")
        shutil.rmtree(bench_results)
        print("  Done")
    
    print("\nClean complete - ready for fresh start")


def clean_reorder_cache(graphs_dir: str = None):
    """Clean all .lo and .time files to force fresh reordering.
    
    Args:
        graphs_dir: Directory containing graph data (default: auto-detect)
    """
    if graphs_dir is None:
        graphs_dir = os.environ.get("GRAPH_DATA_DIR", DEFAULT_GRAPHS_DIR)
    
    log_section("Cleaning Reorder Cache")
    
    # Find all .lo files
    lo_files = glob.glob(os.path.join(graphs_dir, "**/*.lo"), recursive=True)
    time_files = glob.glob(os.path.join(graphs_dir, "**/*.time"), recursive=True)
    
    total = len(lo_files) + len(time_files)
    if total == 0:
        log("No cached reorder files found.")
        return
    
    log(f"Found {len(lo_files)} .lo files and {len(time_files)} .time files")
    
    removed = 0
    for f in lo_files + time_files:
        try:
            os.remove(f)
            removed += 1
        except Exception as e:
            log(f"  Failed to remove {f}: {e}", "WARN")
    
    log(f"Removed {removed}/{total} cached files")
    log("Fresh reordering will be performed on next run.")


# get_num_threads() is imported from lib/features.py

def run_command(cmd: str, timeout: int = 300, use_all_threads: bool = True) -> Tuple[bool, str, str]:
    """Run a shell command with timeout.
    
    Args:
        cmd: Command to run
        timeout: Timeout in seconds
        use_all_threads: If True, set OMP_NUM_THREADS to use all available cores
    """
    try:
        # Set up environment with OpenMP thread count
        env = os.environ.copy()
        if use_all_threads:
            num_threads = get_num_threads()
            env['OMP_NUM_THREADS'] = str(num_threads)
        
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "TIMEOUT"
    except Exception as e:
        return False, "", str(e)

def parse_benchmark_output(output: str) -> Dict[str, Any]:
    """Parse benchmark output for timing and stats."""
    result = {}
    
    # Graph stats
    match = re.search(r'Graph has (\d+) nodes and (\d+)', output)
    if match:
        result['nodes'] = int(match.group(1))
        result['edges'] = int(match.group(2))
    
    # Timing patterns - base execution times
    patterns = {
        'trial_time': r'Trial Time:\s+([\d.]+)',
        'average_time': r'Average Time:\s+([\d.]+)',
        'total_time': r'Total Time:\s+([\d.]+)',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            result[key] = float(match.group(1))
    
    # Reorder/Ordering time patterns - from various algorithms
    # Patterns like: "RabbitOrder Map Time: 0.123", "Ordering Time: 0.456", etc.
    reorder_patterns = [
        r'(?:RabbitOrder|GOrder|LeidenOrder|COrder|RCMOrder|HubSort|DBG|Relabel|Sub-RabbitOrder)\s*Map Time[:\s]+([\d.]+)',
        r'Ordering Time[:\s]+([\d.]+)',
        r'RandOrder Time[:\s]+([\d.]+)',
        r'Reorder Time[:\s]+([\d.]+)',
        r'Total Reordering Time[:\s]+([\d.]+)',
    ]
    
    # Collect all reorder times and use the largest (the actual reorder, not intermediate steps)
    reorder_times = []
    for pattern in reorder_patterns:
        for match in re.finditer(pattern, output, re.IGNORECASE):
            try:
                reorder_times.append(float(match.group(1)))
            except (ValueError, IndexError):
                pass
    
    if reorder_times:
        # Use the maximum (primary algorithm time, not sub-steps)
        result['reorder_time'] = max(reorder_times)
    
    # Modularity
    match = re.search(r'[Mm]odularity[:\s]+([\d.]+)', output)
    if match:
        result['modularity'] = float(match.group(1))
    
    # Global graph features (from AdaptiveOrder output)
    # Degree Variance: 0.1234
    match = re.search(r'Degree Variance[:\s]+([\d.]+)', output)
    if match:
        result['degree_variance'] = float(match.group(1))
    
    # Hub Concentration: 0.1234
    match = re.search(r'Hub Concentration[:\s]+([\d.]+)', output)
    if match:
        result['hub_concentration'] = float(match.group(1))
    
    # Clustering Coefficient: 0.1234
    match = re.search(r'Clustering Coefficient[:\s]+([\d.]+)', output)
    if match:
        result['clustering_coefficient'] = float(match.group(1))
    
    # Community Count: 123 or Number of Communities: 123 or Community Count Estimate: 123
    match = re.search(r'(?:Community Count(?: Estimate)?|Number of Communities|communities)[:\s]+([\d.]+)', output, re.IGNORECASE)
    if match:
        result['community_count'] = int(float(match.group(1)))
    
    # Avg Path Length: 5.6789
    match = re.search(r'Avg Path Length[:\s]+([\d.]+)', output)
    if match:
        result['avg_path_length'] = float(match.group(1))
    
    # Diameter Estimate: 12
    match = re.search(r'Diameter Estimate[:\s]+([\d.]+)', output)
    if match:
        result['diameter'] = float(match.group(1))
    
    # Avg Degree: 10.5 (from topology features)
    match = re.search(r'Avg Degree[:\s]+([\d.]+)', output)
    if match:
        result['avg_degree'] = float(match.group(1))
    
    # Graph Type: social
    match = re.search(r'Graph Type[:\s]+(\w+)', output)
    if match:
        result['graph_type'] = match.group(1).lower()
    
    return result


def parse_cache_output(output: str) -> Dict[str, float]:
    """Parse cache simulation output."""
    result = {}
    
    # The output format is structured in blocks like:
    # ║ L1 Cache (32KB, 8-way, LRU)
    # ║   Hit Rate:                 55.6082%
    # We need to extract the hit rate from each cache block
    
    # Split by cache sections
    l1_match = re.search(r'L1 Cache.*?Hit Rate:\s*([\d.]+)%', output, re.DOTALL)
    l2_match = re.search(r'L2 Cache.*?Hit Rate:\s*([\d.]+)%', output, re.DOTALL)
    l3_match = re.search(r'L3 Cache.*?Hit Rate:\s*([\d.]+)%', output, re.DOTALL)
    
    if l1_match:
        result['l1_hit_rate'] = float(l1_match.group(1))
    if l2_match:
        result['l2_hit_rate'] = float(l2_match.group(1))
    if l3_match:
        result['l3_hit_rate'] = float(l3_match.group(1))
    
    # Fallback patterns for different formats
    if 'l1_hit_rate' not in result:
        patterns = {
            'l1_hit_rate': r'L1 Hit Rate:\s*([\d.]+)%?',
            'l2_hit_rate': r'L2 Hit Rate:\s*([\d.]+)%?',
            'l3_hit_rate': r'L3 Hit Rate:\s*([\d.]+)%?',
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, output)
            if match:
                result[key] = float(match.group(1))
    
    # Also check for compact summary format: L1:XX.X% L2:XX.X% L3:XX.X%
    summary = re.search(r'L1:([\d.]+)%\s*L2:([\d.]+)%\s*L3:([\d.]+)%', output)
    if summary:
        result['l1_hit_rate'] = float(summary.group(1))
        result['l2_hit_rate'] = float(summary.group(2))
        result['l3_hit_rate'] = float(summary.group(3))
    
    return result


# ============================================================================
# Phase Functions (Delegated to lib/ modules)
# ============================================================================
#
# The following phase functions are now imported from lib/ modules:
#
# From lib/reorder.py:
#   - generate_reorderings() - Phase 1: Generate vertex reorderings
#   - generate_reorderings_with_variants() - Phase 1 with Leiden variants
#   - generate_label_maps() - Generate .lo mapping files
#   - load_label_maps_index() - Load existing label maps
#   - get_label_map_path() - Get path to a label map file
#
# From lib/benchmark.py:
#   - run_benchmark_suite() - Phase 2: Run performance benchmarks
#
# From lib/cache.py:
#   - run_cache_simulations() - Phase 3: Run cache simulations
#
# From lib/weights.py:
#   - generate_perceptron_weights_from_results() - Phase 4: Generate weights
#   - update_zero_weights() - Phase 5: Update zero weights
#
# From lib/analysis.py:
#   - analyze_adaptive_order() - Phase 6: Adaptive analysis
#   - compare_adaptive_vs_fixed() - Phase 7: Comparison
#   - run_subcommunity_brute_force() - Phase 8: Brute force validation
#
# From lib/training.py:
#   - train_adaptive_weights_iterative() - Phase 9: Iterative training
#   - train_adaptive_weights_large_scale() - Phase 10: Large-scale training
#
# From lib/phases.py:
#   - run_reorder_phase(), run_benchmark_phase(), run_cache_phase(), etc.
#   - run_full_pipeline() - Run all phases
#   - PhaseConfig - Configuration for phase execution
#
# Legacy local implementations have been removed to avoid duplication.
# Use the run_phases() function or import directly from lib/.
#
# ============================================================================

# Alias imports for backward compatibility with existing code
from scripts.lib.reorder import (
    generate_reorderings,
    generate_reorderings_with_variants,
    generate_label_maps,
    load_label_maps_index,
    get_label_map_path,
)
from scripts.lib.benchmark import run_benchmark_suite, run_benchmarks_multi_graph
from scripts.lib.cache import run_cache_simulations, run_cache_simulations_with_variants
from scripts.lib.analysis import (
    analyze_adaptive_order,
    compare_adaptive_vs_fixed,
    run_subcommunity_brute_force,
)
from scripts.lib.training import (
    train_adaptive_weights_iterative,
    train_adaptive_weights_large_scale,
    initialize_enhanced_weights,
)


# ============================================================================
# Local helper functions that wrap lib/ calls
# ============================================================================

def run_benchmarks_with_variants(
    graphs: List[GraphInfo],
    label_maps: Dict[str, Dict[str, str]],
    benchmarks: List[str],
    bin_dir: str,
    num_trials: int = 3,
    timeout: int = TIMEOUT_BENCHMARK,
    weights_dir: str = DEFAULT_WEIGHTS_DIR,
    update_weights: bool = True,
    progress: 'ProgressTracker' = None
) -> List[BenchmarkResult]:
    """
    Run benchmarks with variant-expanded label maps.
    
    This iterates directly over the algorithm names in label_maps (which include
    variant suffixes like LeidenCSR_gve, RABBITORDER_csr) to ensure the results
    contain the full variant names.
    
    When using .lo files (MAP mode), loads reorder_time from the corresponding
    .time file instead of parsing from benchmark output.
    """
    from scripts.lib.benchmark import run_benchmark, check_binary_exists
    from pathlib import Path
    
    def load_reorder_time(label_map_path: str) -> float:
        """Load reorder time from .time file corresponding to .lo file."""
        if not label_map_path:
            return 0.0
        time_file = Path(label_map_path).with_suffix('.time')
        if time_file.exists():
            try:
                return float(time_file.read_text().strip())
            except (ValueError, IOError):
                return 0.0
        return 0.0
    
    results = []
    
    # Collect all unique algorithm names from label_maps
    all_algo_names = set()
    for graph_maps in label_maps.values():
        all_algo_names.update(graph_maps.keys())
    
    # Always include ORIGINAL (algo_id=0) - it doesn't need a label map
    all_algo_names.add("ORIGINAL")
    
    # Sort for consistent ordering (ORIGINAL first, then alphabetically)
    algo_names_sorted = ["ORIGINAL"] + sorted([n for n in all_algo_names if n != "ORIGINAL"])
    
    total_configs = len(graphs) * len(algo_names_sorted) * len(benchmarks)
    completed = 0
    
    for graph in graphs:
        graph_name = graph.name
        graph_path = graph.path
        graph_label_maps = label_maps.get(graph_name, {})
        
        if progress:
            progress.info(f"Benchmarking: {graph_name} ({graph.size_mb:.1f}MB)")
        
        for bench in benchmarks:
            if not check_binary_exists(bench, bin_dir):
                log.warning(f"Skipping {bench}: binary not found")
                continue
            
            if progress:
                progress.info(f"  {bench.upper()}:")
            
            for algo_name in algo_names_sorted:
                # Determine algorithm ID from name
                algo_id = 0
                for aid, aname in ALGORITHMS.items():
                    if algo_name == aname or algo_name.startswith(aname + "_"):
                        algo_id = aid
                        break
                
                # Get label map path for this algorithm (if not ORIGINAL)
                label_map_path = ""
                if algo_name == "ORIGINAL":
                    # ORIGINAL uses algo_id=0, no label map needed
                    algo_opt = "0"
                else:
                    label_map_path = graph_label_maps.get(algo_name, "")
                    if not label_map_path:
                        # Skip if no label map for this graph/algorithm combo
                        continue
                    # Use MAP mode with label file
                    algo_opt = f"13:{label_map_path}"
                
                result = run_benchmark(
                    benchmark=bench,
                    graph_path=graph_path,
                    algorithm=algo_opt,
                    trials=num_trials,
                    timeout=timeout,
                    bin_dir=bin_dir
                )
                
                # Set the algorithm name to include variant suffix
                result.algorithm = algo_name
                result.algorithm_id = algo_id
                result.graph = graph_name
                result.nodes = graph.nodes
                result.edges = graph.edges
                
                # Load reorder_time from .time file when using .lo files (MAP mode)
                if label_map_path:
                    result.reorder_time = load_reorder_time(label_map_path)
                
                results.append(result)
                completed += 1
                
                # Log progress
                status = "✓" if result.success else "✗"
                time_str = f"{result.time_seconds:.4f}s" if result.success else result.error[:30]
                if progress:
                    progress.info(f"    [{completed}/{total_configs}] {algo_name}: {time_str}")
    
    return results


def generate_perceptron_weights(
    benchmark_results: List[BenchmarkResult],
    cache_results: List[CacheResult],
    reorder_results: List[ReorderResult],
    output_file: str
) -> Dict[str, PerceptronWeight]:
    """Generate perceptron weights. Delegates to lib/weights."""
    from scripts.lib.weights import compute_weights_from_results
    return compute_weights_from_results(
        benchmark_results=benchmark_results,
        cache_results=cache_results,
        reorder_results=reorder_results,
        output_file=output_file
    )


def update_zero_weights(
    weights_file: str,
    benchmark_results: List[BenchmarkResult],
    cache_results: List[CacheResult],
    reorder_results: List[ReorderResult],
    graphs_dir: str
) -> None:
    """Update zero weights with comprehensive analysis. Delegates to lib/weights."""
    from scripts.lib.weights import update_zero_weights as lib_update_zero_weights
    lib_update_zero_weights(
        weights_file=weights_file,
        benchmark_results=benchmark_results,
        cache_results=cache_results,
        reorder_results=reorder_results,
        graphs_dir=graphs_dir
    )


def validate_adaptive_accuracy(
    graphs: List[GraphInfo],
    bin_dir: str,
    output_dir: str,
    benchmarks: List[str] = None,
    timeout: int = TIMEOUT_BENCHMARK,
    num_trials: int = 3,
    force_reorder: bool = False
) -> List[Dict]:
    """Validate adaptive accuracy. Delegates to lib/analysis."""
    from scripts.lib.analysis import validate_adaptive_accuracy as lib_validate
    return lib_validate(
        graphs=graphs,
        bin_dir=bin_dir,
        output_dir=output_dir,
        benchmarks=benchmarks or ['pr', 'bfs', 'cc'],
        timeout=timeout,
        num_trials=num_trials,
        force_reorder=force_reorder
    )



# ============================================================================
# Simplified Phase-Based Orchestration (using lib/phases.py)
# ============================================================================

def run_phases(args, graphs: List[GraphInfo], algorithms: List[int]) -> Dict[str, Any]:
    """
    Run experiment phases using the lib/phases.py orchestration module.
    
    This is a simplified version of run_experiment that delegates to lib modules.
    
    Args:
        args: Command line arguments
        graphs: List of GraphInfo objects
        algorithms: List of algorithm IDs to benchmark
    
    Returns:
        Dictionary with results from each phase
    """
    # Create phase configuration from args
    config = PhaseConfig.from_args(args)
    
    # Initialize results storage
    results = {
        'reorder': [],
        'benchmark': [],
        'cache': [],
        'label_maps': {},
    }
    
    # ==========================================================================
    # Load previous results when running individual phases
    # This allows running phases separately: --phase reorder, then --phase benchmark, etc.
    # ==========================================================================
    def load_previous_results():
        """Load results from previous phase runs."""
        loaded_any = False
        
        # Load label maps (needed for benchmark and cache phases)
        if args.phase in ["benchmark", "cache"] or getattr(args, "use_maps", False):
            from scripts.lib.reorder import load_label_maps_index
            results['label_maps'] = load_label_maps_index(args.results_dir)
            if results['label_maps']:
                config.progress.info(f"Loaded label maps for {len(results['label_maps'])} graphs")
                loaded_any = True
        
        # Load reorder results (needed for weights phase)
        if args.phase in ["weights"]:
            latest_reorder = max(glob.glob(os.path.join(args.results_dir, "reorder_*.json")), default=None, key=os.path.getmtime)
            if latest_reorder:
                try:
                    with open(latest_reorder) as f:
                        results['reorder'] = [ReorderResult(**r) for r in json.load(f)]
                    config.progress.info(f"Loaded {len(results['reorder'])} reorder results from {os.path.basename(latest_reorder)}")
                    loaded_any = True
                except Exception as e:
                    config.progress.warning(f"Failed to load reorder results: {e}")
        
        # Load benchmark results (needed for weights phase)
        if args.phase in ["weights"]:
            latest_bench = max(glob.glob(os.path.join(args.results_dir, "benchmark_*.json")), default=None, key=os.path.getmtime)
            if latest_bench:
                try:
                    with open(latest_bench) as f:
                        results['benchmark'] = [BenchmarkResult(**r) for r in json.load(f)]
                    config.progress.info(f"Loaded {len(results['benchmark'])} benchmark results from {os.path.basename(latest_bench)}")
                    loaded_any = True
                except Exception as e:
                    config.progress.warning(f"Failed to load benchmark results: {e}")
        
        # Load cache results (needed for weights phase)
        if args.phase in ["weights"] and not args.skip_cache:
            latest_cache = max(glob.glob(os.path.join(args.results_dir, "cache_*.json")), default=None, key=os.path.getmtime)
            if latest_cache:
                try:
                    with open(latest_cache) as f:
                        results['cache'] = [CacheResult(**r) for r in json.load(f)]
                    config.progress.info(f"Loaded {len(results['cache'])} cache results from {os.path.basename(latest_cache)}")
                    loaded_any = True
                except Exception as e:
                    config.progress.warning(f"Failed to load cache results: {e}")
        
        return loaded_any
    
    # Load previous results if running a phase that needs them
    if args.phase != "all":
        config.progress.phase_start("LOADING PREVIOUS RESULTS", f"Preparing for --phase {args.phase}")
        if load_previous_results():
            config.progress.success("Previous results loaded")
        else:
            config.progress.info("No previous results found (this may be the first run)")
        config.progress.phase_end()
    
    # Load existing label maps if explicitly requested
    if getattr(args, "use_maps", False) and not results['label_maps']:
        config.progress.phase_start("LOADING MAPS", "Loading pre-generated label mappings")
        from scripts.lib.reorder import load_label_maps_index
        results['label_maps'] = load_label_maps_index(args.results_dir)
        if results['label_maps']:
            config.progress.success(f"Loaded maps for {len(results['label_maps'])} graphs")
        config.progress.phase_end()
    
    # Phase 1: Reordering
    if args.phase in ["all", "reorder"]:
        reorder_results, label_maps = run_reorder_phase(
            graphs=graphs,
            algorithms=algorithms,
            config=config,
            label_maps=results['label_maps']
        )
        results['reorder'] = reorder_results
        results['label_maps'] = label_maps
    
    # Phase 2: Benchmarking
    if args.phase in ["all", "benchmark"]:
        benchmark_results = run_benchmark_phase(
            graphs=graphs,
            algorithms=algorithms,
            label_maps=results['label_maps'],
            config=config
        )
        results['benchmark'] = benchmark_results
    
    # Phase 3: Cache Simulation
    if args.phase in ["all", "cache"] and not args.skip_cache:
        cache_results = run_cache_phase(
            graphs=graphs,
            algorithms=algorithms,
            label_maps=results['label_maps'],
            config=config
        )
        results['cache'] = cache_results
    
    # Phase 4: Weights
    if args.phase in ["all", "weights"]:
        weights = run_weights_phase(
            benchmark_results=results['benchmark'],
            cache_results=results['cache'],
            reorder_results=results['reorder'],
            config=config
        )
        results['weights'] = weights
    
    # Phase 5: Fill Weights
    if getattr(args, "fill_weights", False):
        run_fill_weights_phase(
            benchmark_results=results['benchmark'],
            cache_results=results['cache'],
            reorder_results=results['reorder'],
            config=config
        )
    
    # Phase 6: Adaptive Analysis
    if args.phase in ["all", "adaptive"] or getattr(args, "adaptive_analysis", False):
        adaptive_results = run_adaptive_analysis_phase(graphs=graphs, config=config)
        results['adaptive'] = adaptive_results
    
    # Phase 7: Comparison
    if getattr(args, "adaptive_comparison", False):
        comparison_results = run_comparison_phase(graphs=graphs, config=config)
        results['comparison'] = comparison_results
    
    # Phase 8: Brute Force
    if getattr(args, "brute_force", False):
        bf_results = run_brute_force_phase(
            graphs=graphs,
            config=config,
            benchmark=getattr(args, "bf_benchmark", "pr")
        )
        results['brute_force'] = bf_results
    
    # Phase 9: Training
    if getattr(args, "train_adaptive", False):
        training_result = run_training_phase(
            graphs=graphs,
            config=config,
            weights_file=getattr(args, 'weights_file', None)
        )
        results['training'] = training_result
    
    # Phase 10: Large-Scale Training
    if getattr(args, "train_large", False):
        large_training_result = run_large_scale_training_phase(
            graphs=graphs,
            config=config,
            weights_file=getattr(args, 'weights_file', None)
        )
        results['large_training'] = large_training_result
    
    return results


# ============================================================================
# Main Experiment Pipeline (Full Implementation)
# ============================================================================

def run_experiment(args):
    """Run the complete experiment pipeline with comprehensive progress tracking."""
    
    global _progress
    _progress = ProgressTracker()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize session for log grouping
    from scripts.lib.graph_data import set_session_id
    session_id = set_session_id(timestamp)
    
    # Show experiment banner
    _progress.banner(
        "GRAPHBREW EXPERIMENT PIPELINE",
        f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Session: {session_id}"
    )
    
    # Phase 0: Configuration Summary
    _progress.phase_start("CONFIGURATION", "Gathering experiment parameters")
    _progress.info(f"Results directory: {args.results_dir}")
    _progress.info(f"Graphs directory: {args.graphs_dir}")
    _progress.info(f"Bin directory: {args.bin_dir}")
    _progress.info(f"Phase to run: {args.phase}")
    _progress.info(f"Benchmarks: {', '.join(args.benchmarks)}")
    _progress.info(f"Trials per benchmark: {args.trials}")
    if getattr(args, 'expand_variants', False):
        _progress.info("Leiden variant expansion: ENABLED")
    _progress.phase_end()
    
    # Discover graphs
    _progress.phase_start("GRAPH DISCOVERY", "Finding available graph datasets")
    
    # Determine size range
    if args.min_size > 0 or args.max_size < float('inf'):
        min_size, max_size = args.min_size, args.max_size
    elif args.graphs == "all":
        min_size, max_size = 0, float('inf')
    elif args.graphs == "small":
        min_size, max_size = 0, SIZE_SMALL
    elif args.graphs == "medium":
        min_size, max_size = SIZE_SMALL, SIZE_MEDIUM
    elif args.graphs == "large":
        min_size, max_size = SIZE_MEDIUM, SIZE_LARGE
    else:
        min_size, max_size = args.min_size, args.max_size
    
    _progress.info(f"Size filter: {min_size:.1f}MB - {max_size:.1f}MB")
    
    graphs = discover_graphs(args.graphs_dir, min_size, max_size, max_memory_gb=args.max_memory,
                             min_edges=getattr(args, 'min_edges', 0))
    
    # Filter by specific graph name(s) if provided
    if args.graph_list:
        # Multiple graphs specified
        graph_names_lower = [n.lower() for n in args.graph_list]
        filtered = []
        for g in graphs:
            if g.name.lower() in graph_names_lower or any(n in g.name.lower() for n in graph_names_lower):
                filtered.append(g)
        if not filtered:
            _progress.error(f"None of the specified graphs found: {args.graph_list}")
            _progress.phase_end("Graphs not available")
            return
        graphs = filtered
        _progress.info(f"Filtered to {len(graphs)} graphs: {', '.join(g.name for g in graphs)}")
    elif args.graph_name:
        # Single graph specified
        graphs = [g for g in graphs if g.name.lower() == args.graph_name.lower() or 
                  args.graph_name.lower() in g.name.lower()]
        if not graphs:
            _progress.error(f"Graph '{args.graph_name}' not found!")
            _progress.phase_end("Graph not available")
            return
        _progress.info(f"Filtered to graph: {args.graph_name}")
    
    if args.max_graphs:
        graphs = graphs[:args.max_graphs]
    
    if not graphs:
        _progress.error("No graphs found!")
        _progress.phase_end("No graphs available")
        return
    
    # Display discovered graphs in a table
    _progress.success(f"Found {len(graphs)} graphs")
    total_size = sum(g.size_mb for g in graphs)
    _progress.info(f"Total size: {total_size:.1f} MB")
    print()
    for i, g in enumerate(graphs, 1):
        nodes_str = f"{g.nodes:,}" if g.nodes else "?"
        edges_str = f"{g.edges:,}" if g.edges else "?"
        print(f"    {i:2d}. {g.name:<25} {g.size_mb:8.1f} MB  │ V={nodes_str:<12} E={edges_str}")
    
    _progress.phase_end(f"Found {len(graphs)} graphs totaling {total_size:.1f} MB")
    
    # Select algorithms
    if args.key_only:
        algorithms = KEY_ALGORITHMS
    else:
        algorithms = BENCHMARK_ALGORITHMS
    
    # Filter by specific algorithm name(s) if provided
    algo_filter_names = args.algo_list if args.algo_list else ([args.algo_name] if args.algo_name else None)
    if algo_filter_names:
        # Build reverse mapping from name to id
        name_to_id = {v.upper(): k for k, v in ALGORITHMS.items()}
        # Also add common variants
        name_to_id.update({
            "RABBIT": 8, "RABBITORDER_BOOST": 8, "RABBITORDER_CSR": 8,
            "LEIDEN": 15, "LEIDENORDER": 15,
            "LEIDENDENDROGRAM": 16, "LEIDEN_DENDROGRAM": 16,
            "LEIDENCSR": 17, "LEIDEN_CSR": 17, "LEIDENCSR_GVE": 17, "GVE": 17,
        })
        
        filtered_algos = set()
        filtered_names = []
        for name in algo_filter_names:
            name_upper = name.upper().replace("-", "_")
            if name_upper in name_to_id:
                filtered_algos.add(name_to_id[name_upper])
                filtered_names.append(name)
            else:
                # Try partial match
                for algo_name, algo_id in name_to_id.items():
                    if name_upper in algo_name or algo_name in name_upper:
                        filtered_algos.add(algo_id)
                        filtered_names.append(ALGORITHMS.get(algo_id, name))
                        break
        
        if filtered_algos:
            algorithms = [a for a in algorithms if a in filtered_algos]
            if not algorithms:
                # If no overlap with current set, use filtered set directly
                algorithms = sorted(list(filtered_algos))
            _progress.info(f"Filtered to {len(algorithms)} algorithms: {', '.join(filtered_names)}")
        else:
            _progress.error(f"No matching algorithms found for: {algo_filter_names}")
            _progress.info(f"Available algorithms: {', '.join(ALGORITHMS.values())}")
            return

    _progress.phase_start("ALGORITHM SELECTION", "Determining which algorithms to test")
    _progress.info(f"Algorithm set: {'KEY (fast subset)' if args.key_only else 'FULL benchmark set'}")
    _progress.info(f"Total algorithms: {len(algorithms)}")
    algo_names = [ALGORITHMS.get(a, f"ALG_{a}") for a in algorithms[:10]]
    _progress.info(f"Algorithms: {', '.join(algo_names)}{'...' if len(algorithms) > 10 else ''}")
    _progress.info(f"Benchmarks: {', '.join(args.benchmarks)}")
    
    # Calculate total operations
    total_ops = len(graphs) * len(algorithms) * len(args.benchmarks) * args.trials
    _progress.info(f"Estimated operations: {total_ops:,} (graphs × algorithms × benchmarks × trials)")
    _progress.phase_end()
    
    # Initialize result storage
    all_reorder_results = []
    all_benchmark_results = []
    all_cache_results = []
    label_maps = {}
    
    # Pre-generate label maps if requested (also records reorder times)
    if getattr(args, "generate_maps", False):
        _progress.phase_start("LABEL MAP GENERATION", "Pre-generating reordering mappings (.lo files)")
        
        # Check if variant expansion is requested
        if getattr(args, "expand_variants", False):
            _progress.info("Leiden variant expansion: ENABLED")
            _progress.info(f"  Resolution: {getattr(args, 'leiden_resolution', LEIDEN_DEFAULT_RESOLUTION)}")
            _progress.info(f"  Passes: {getattr(args, 'leiden_passes', LEIDEN_DEFAULT_PASSES)}")
            
            # Use variant-aware mapping generation
            label_maps, reorder_timing_results = generate_reorderings_with_variants(
                graphs=graphs,
                algorithms=algorithms,
                bin_dir=args.bin_dir,
                output_dir=args.results_dir,
                expand_leiden_variants=True,
                leiden_resolution=getattr(args, "leiden_resolution", LEIDEN_DEFAULT_RESOLUTION),
                leiden_passes=getattr(args, "leiden_passes", LEIDEN_DEFAULT_PASSES),
                leiden_csr_variants=getattr(args, "leiden_csr_variants", None),
                leiden_dendrogram_variants=getattr(args, "leiden_dendrogram_variants", None),
                timeout=args.timeout_reorder,
                skip_slow=args.skip_slow,
                force_reorder=getattr(args, "force_reorder", False)
            )
        else:
            # Use standard mapping generation (no variant expansion)
            label_maps, reorder_timing_results = generate_label_maps(
                graphs=graphs,
                algorithms=algorithms,
                bin_dir=args.bin_dir,
                output_dir=args.results_dir,
                timeout=args.timeout_reorder,
                skip_slow=args.skip_slow
            )
        all_reorder_results.extend(reorder_timing_results)
        _progress.phase_end(f"Generated {len(label_maps)} graph mappings")
    
    # Load existing label maps if requested
    if getattr(args, "use_maps", False):
        _progress.phase_start("LOADING EXISTING MAPS", "Loading pre-generated label mappings")
        label_maps = load_label_maps_index(args.results_dir)
        if label_maps:
            _progress.success(f"Loaded label maps for {len(label_maps)} graphs")
        else:
            _progress.warning("No existing label maps found")
        _progress.phase_end()
    
    # Phase 1: Reordering
    if args.phase in ["all", "reorder"]:
        _progress.phase_start("REORDERING", "Generating vertex reorderings for all graphs")
        
        # Check if variant expansion is requested
        if getattr(args, "expand_variants", False):
            _progress.info("Leiden/RabbitOrder variant expansion: ENABLED")
            
            # Use variant-aware reordering
            variant_label_maps, reorder_results = generate_reorderings_with_variants(
                graphs=graphs,
                algorithms=algorithms,
                bin_dir=args.bin_dir,
                output_dir=args.results_dir,
                expand_leiden_variants=True,
                leiden_resolution=getattr(args, "leiden_resolution", LEIDEN_DEFAULT_RESOLUTION),
                leiden_passes=getattr(args, "leiden_passes", LEIDEN_DEFAULT_PASSES),
                leiden_csr_variants=getattr(args, "leiden_csr_variants", None),
                leiden_dendrogram_variants=getattr(args, "leiden_dendrogram_variants", None),
                rabbit_variants=getattr(args, "rabbit_variants", None),
                graphbrew_variants=getattr(args, "graphbrew_variants", None),
                timeout=args.timeout_reorder,
                skip_slow=args.skip_slow,
                force_reorder=getattr(args, "force_reorder", False)
            )
            
            # Merge variant label maps into main label_maps
            for graph_name, algo_maps in variant_label_maps.items():
                if graph_name not in label_maps:
                    label_maps[graph_name] = {}
                label_maps[graph_name].update(algo_maps)
        else:
            # Standard reordering (no variant expansion)
            reorder_results = generate_reorderings(
                graphs=graphs,
                algorithms=algorithms,
                bin_dir=args.bin_dir,
                output_dir=args.results_dir,
                timeout=args.timeout_reorder,
                skip_slow=args.skip_slow,
                generate_maps=True,  # Always generate .lo mapping files
                force_reorder=getattr(args, "force_reorder", False)
            )
            
            # Build label_maps from successful reorder results if not already populated
            if not label_maps:
                for r in reorder_results:
                    if r.success and r.mapping_file:
                        if r.graph not in label_maps:
                            label_maps[r.graph] = {}
                        label_maps[r.graph][r.algorithm_name] = r.mapping_file
                if label_maps:
                    _progress.info(f"Built label_maps for {len(label_maps)} graphs from reorder results")
        
        all_reorder_results.extend(reorder_results)
        
        # Save intermediate results
        reorder_file = os.path.join(args.results_dir, f"reorder_{timestamp}.json")
        with open(reorder_file, 'w') as f:
            json.dump([asdict(r) for r in reorder_results], f, indent=2)
        _progress.success(f"Reorder results saved to: {reorder_file}")
        
        _progress.phase_end(f"Generated {len(reorder_results)} reorderings")
    
    # Phase 2: Benchmarks
    if args.phase in ["all", "benchmark"]:
        _progress.phase_start("BENCHMARKING", "Running performance benchmarks")
        
        # Check if we have variant-expanded label maps
        has_variant_maps = (label_maps and 
                           any('_' in algo_name for g in label_maps.values() for algo_name in g.keys()))
        
        if has_variant_maps and getattr(args, "expand_variants", False):
            # Use variant-aware benchmarking with pre-generated variant mappings
            _progress.info("Mode: Variant-aware benchmarking (LeidenCSR_fast, LeidenCSR_hubsort, etc.)")
            benchmark_results = run_benchmarks_with_variants(
                graphs=graphs,
                label_maps=label_maps,
                benchmarks=args.benchmarks,
                bin_dir=args.bin_dir,
                num_trials=args.trials,
                timeout=args.timeout_benchmark,
                weights_dir=args.weights_dir,
                update_weights=not getattr(args, 'no_incremental', False),
                progress=_progress
            )
        else:
            # Standard benchmarking
            _progress.info("Mode: Standard benchmarking")
            benchmark_results = run_benchmarks_multi_graph(
                graphs=graphs,
                algorithms=algorithms,
                benchmarks=args.benchmarks,
                bin_dir=args.bin_dir,
                num_trials=args.trials,
                timeout=args.timeout_benchmark,
                skip_slow=args.skip_slow,
                label_maps=label_maps,
                weights_dir=args.weights_dir,
                update_weights=not getattr(args, 'no_incremental', False)
            )
        all_benchmark_results.extend(benchmark_results)
        
        # Save intermediate results
        bench_file = os.path.join(args.results_dir, f"benchmark_{timestamp}.json")
        with open(bench_file, 'w') as f:
            json.dump([asdict(r) for r in benchmark_results], f, indent=2)
        _progress.success(f"Benchmark results saved to: {bench_file}")
        
        # Show summary statistics
        if benchmark_results:
            successful = [r for r in benchmark_results if r.time_seconds > 0]
            _progress.stats_box("Benchmark Statistics", {
                "Total runs": len(benchmark_results),
                "Successful": len(successful),
                "Failed/Timeout": len(benchmark_results) - len(successful),
                "Avg time": f"{sum(r.time_seconds for r in successful) / len(successful):.4f}s" if successful else "N/A"
            })
        
        _progress.phase_end(f"Completed {len(benchmark_results)} benchmark runs")
    
    # Phase 3: Cache Simulations
    if args.phase in ["all", "cache"] and not args.skip_cache:
        _progress.phase_start("CACHE SIMULATION", "Running cache miss simulations")
        
        # Check if we have variant-expanded label maps
        has_variant_maps = (label_maps and 
                           any('_' in algo_name for g in label_maps.values() for algo_name in g.keys()))
        
        if has_variant_maps and getattr(args, "expand_variants", False):
            # Use variant-aware cache simulation
            _progress.info("Mode: Variant-aware cache simulation (LeidenCSR_fast, LeidenCSR_hubsort, etc.)")
            cache_results = run_cache_simulations_with_variants(
                graphs=graphs,
                label_maps=label_maps,
                benchmarks=args.benchmarks,
                bin_sim_dir=args.bin_sim_dir,
                timeout=args.timeout_sim,
                skip_heavy=args.skip_heavy
            )
        else:
            # Standard cache simulation with variant support
            _progress.info("Mode: Standard cache simulation")
            
            # Pass variant lists if specified
            leiden_csr_variants = getattr(args, 'leiden_csr_variants', None)
            rabbit_variants = getattr(args, 'rabbit_variants', None)
            leiden_dendrogram_variants = getattr(args, 'leiden_dendrogram_variants', None)
            
            if leiden_csr_variants or rabbit_variants or leiden_dendrogram_variants:
                _progress.info(f"  LeidenCSR variants: {leiden_csr_variants or ['gve (default)']}")
                _progress.info(f"  RabbitOrder variants: {rabbit_variants or ['csr (default)']}")
            
            cache_results = run_cache_simulations(
                graphs=graphs,
                algorithms=algorithms,
                benchmarks=args.benchmarks,
                bin_sim_dir=args.bin_sim_dir,
                timeout=args.timeout_sim,
                skip_heavy=args.skip_heavy,
                label_maps=label_maps,
                leiden_csr_variants=leiden_csr_variants,
                rabbit_variants=rabbit_variants,
                leiden_dendrogram_variants=leiden_dendrogram_variants,
                resolution=getattr(args, 'leiden_resolution', 1.0),
                passes=getattr(args, 'leiden_passes', 3)
            )
        all_cache_results.extend(cache_results)
        
        # Save intermediate results
        cache_file = os.path.join(args.results_dir, f"cache_{timestamp}.json")
        with open(cache_file, 'w') as f:
            json.dump([asdict(r) for r in cache_results], f, indent=2)
        log(f"Cache results saved to: {cache_file}")
    
    # Phase 4: Generate Weights
    if args.phase in ["all", "weights"]:
        # Load previous results if not running full pipeline
        # Note: Multiple Result classes exist with different field names.
        # - utils.py: BenchmarkResult (algorithm, time_seconds)
        # - reorder.py: ReorderResult (reorder_time)
        # - types.py: ReorderResult (time_seconds), BenchmarkResult (algorithm_name, avg_time)
        # The JSON files are saved using the working versions (utils/reorder), so we import those
        
        if not all_benchmark_results:
            latest_bench = max(glob.glob(os.path.join(args.results_dir, "benchmark_*.json")), default=None)
            if latest_bench:
                with open(latest_bench) as f:
                    raw_results = json.load(f)
                    from scripts.lib.utils import BenchmarkResult as UtilsBenchmarkResult
                    all_benchmark_results = [UtilsBenchmarkResult(**r) for r in raw_results]
        
        if not all_cache_results and not args.skip_cache:
            latest_cache = max(glob.glob(os.path.join(args.results_dir, "cache_*.json")), default=None)
            if latest_cache:
                with open(latest_cache) as f:
                    all_cache_results = [CacheResult(**r) for r in json.load(f)]
        
        if not all_reorder_results:
            latest_reorder = max(glob.glob(os.path.join(args.results_dir, "reorder_*.json")), default=None)
            if latest_reorder:
                with open(latest_reorder) as f:
                    # Use reorder.py version which has 'reorder_time' field
                    from scripts.lib.reorder import ReorderResult as ReorderReorderResult
                    all_reorder_results = [ReorderReorderResult(**r) for r in json.load(f)]
        
        if all_benchmark_results:
            _progress.phase_start("WEIGHTS", "Generating perceptron weights")
            generate_perceptron_weights(
                benchmark_results=all_benchmark_results,
                cache_results=all_cache_results,
                reorder_results=all_reorder_results,
                output_file=args.weights_file  # Deprecated, but kept for compatibility
            )
            
            # Update zero weights with comprehensive analysis
            update_zero_weights(
                weights_file=args.weights_file,  # Deprecated, but kept for compatibility
                benchmark_results=all_benchmark_results,
                cache_results=all_cache_results,
                reorder_results=all_reorder_results,
                graphs_dir=args.graphs_dir
            )
            _progress.phase_end("Weights saved to scripts/weights/active/type_0.json")
    
    # Phase 6: Adaptive Order Analysis
    if args.phase in ["all", "adaptive"] or getattr(args, "adaptive_analysis", False):
        log_section("Running Adaptive Order Analysis")
        adaptive_results = analyze_adaptive_order(
            graphs=graphs,
            bin_dir=args.bin_dir,
            output_dir=args.results_dir,
            timeout=args.timeout_reorder
        )
        
        # Print summary
        successful = [r for r in adaptive_results if r.success]
        log(f"\nAdaptive analysis complete: {len(successful)}/{len(adaptive_results)} graphs")
        
        # Show algorithm distribution across all graphs
        total_distribution = {}
        for r in successful:
            for algo, count in r.algorithm_distribution.items():
                total_distribution[algo] = total_distribution.get(algo, 0) + count
        
        if total_distribution:
            log("\nOverall algorithm selection distribution:")
            for algo, count in sorted(total_distribution.items(), key=lambda x: -x[1]):
                log(f"  {algo}: {count}")
    
    # Phase 7: Adaptive vs Fixed Comparison
    if getattr(args, "adaptive_comparison", False):
        # Compare against top fixed algorithms
        fixed_algos = [1, 2, 4, 7, 15, 16]  # RANDOM, SORT, HUBCLUSTER, HUBCLUSTERDBG, LeidenOrder, LeidenDendrogram
        
        comparison_results = compare_adaptive_vs_fixed(
            graphs=graphs,
            bin_dir=args.bin_dir,
            benchmarks=args.benchmarks,
            fixed_algorithms=fixed_algos,
            output_dir=args.results_dir,
            num_trials=args.trials,
            timeout=args.timeout_benchmark
        )
    
    # Phase 8: Brute-Force Validation (All 20 algos vs Adaptive)
    if getattr(args, "brute_force", False):
        bf_benchmark = getattr(args, "bf_benchmark", "pr")
        brute_force_results = run_subcommunity_brute_force(
            graphs=graphs,
            bin_dir=args.bin_dir,
            bin_sim_dir=args.bin_sim_dir,
            output_dir=args.results_dir,
            benchmark=bf_benchmark,
            timeout=args.timeout_benchmark,
            timeout_sim=args.timeout_sim,
            num_trials=args.trials
        )
    
    # Phase 8b: Validate Adaptive Accuracy (faster than brute-force)
    if getattr(args, "validate_adaptive", False):
        validation_results = validate_adaptive_accuracy(
            graphs=graphs,
            bin_dir=args.bin_dir,
            output_dir=args.results_dir,
            benchmarks=getattr(args, "benchmarks", ['pr', 'bfs', 'cc']),
            timeout=args.timeout_benchmark,
            num_trials=args.trials,
            force_reorder=getattr(args, "force_reorder", False)
        )
    
    # Phase 9: Iterative Training (feedback loop to optimize adaptive weights)
    if getattr(args, "train_adaptive", False):
        training_result = train_adaptive_weights_iterative(
            graphs=graphs,
            bin_dir=args.bin_dir,
            bin_sim_dir=args.bin_sim_dir,
            output_dir=args.results_dir,
            weights_file=args.weights_file,
            benchmark=getattr(args, "bf_benchmark", "pr"),
            target_accuracy=getattr(args, "target_accuracy", 80.0),
            max_iterations=getattr(args, "max_iterations", 10),
            timeout=args.timeout_benchmark,
            timeout_sim=args.timeout_sim,
            num_trials=args.trials,
            learning_rate=getattr(args, "learning_rate", 0.1),
            algorithms=algorithms,
            weights_dir=args.weights_dir  # Type-based weights directory
        )
    
    # Phase 10: Large-Scale Training (batched multi-benchmark training)
    if getattr(args, "train_large", False):
        large_training_result = train_adaptive_weights_large_scale(
            graphs=graphs,
            bin_dir=args.bin_dir,
            bin_sim_dir=args.bin_sim_dir,
            output_dir=args.results_dir,
            weights_file=args.weights_file,
            benchmarks=getattr(args, "train_benchmarks", ['pr', 'bfs', 'cc']),
            target_accuracy=getattr(args, "target_accuracy", 80.0),
            max_iterations=getattr(args, "max_iterations", 10),
            batch_size=getattr(args, "batch_size", 8),
            timeout=args.timeout_benchmark,
            timeout_sim=args.timeout_sim,
            num_trials=args.trials,
            learning_rate=getattr(args, "learning_rate", 0.15),
            algorithms=algorithms
        )
    
    # Initialize/upgrade weights file with enhanced features
    if getattr(args, "init_weights", False):
        log_section("Initialize Enhanced Weights")
        weights = initialize_enhanced_weights(args.weights_file)
        log(f"Weights initialized/upgraded with {len(weights) - 1} algorithms")
        log(f"Saved to: {args.weights_file}")
    
    # Fill ALL weights mode: comprehensive training to populate all weight fields
    if getattr(args, "fill_weights", False):
        log_section("Fill All Weights - Comprehensive Training")
        log("This mode runs all phases to populate every weight field:")
        log("  - Phase 0: Graph Analysis (detects graph types from properties)")
        log("  - Phase 1: Reorderings (fills w_reorder_time)")
        log("  - Phase 2: Benchmarks (fills bias, w_log_*, w_density, w_avg_degree)")
        log("  - Phase 3: Cache Simulation (fills cache_l1/l2/l3_impact)")
        log("  - Phase 4: Generate base weights")
        log("  - Phase 5: Update topology weights (fills w_clustering_coeff, etc.)")
        log("  - Phase 6: Compute per-benchmark weights")
        log("  - Phase 7: Generate per-graph-type weight files (from detected properties)")
        log("")
        
        # Load existing graph properties cache if available
        weights_file = getattr(args, 'weights_file', None) or os.path.join(args.weights_dir, 'weights.json')
        cache_dir = os.path.dirname(weights_file) or args.weights_dir
        props_cache = load_graph_properties_cache(cache_dir)
        log(f"Loaded graph properties cache: {len(props_cache)} graphs")
        
        # Force enable cache simulation for this mode
        skip_cache_original = getattr(args, 'skip_cache', False)
        args.skip_cache = False
        
        # Phase 0: Graph Analysis - Run AdaptiveOrder to detect graph types
        log_section("Phase 0: Graph Property Analysis")
        log("Running AdaptiveOrder to compute graph properties (modularity, degree variance, etc.)")
        adaptive_results = analyze_adaptive_order(
            graphs=graphs,
            bin_dir=args.bin_dir,
            output_dir=args.results_dir,
            timeout=args.timeout_reorder
        )
        # Save the cache after analysis
        save_graph_properties_cache(cache_dir)
        props_cache = load_graph_properties_cache(cache_dir)
        log(f"Graph properties cached for {len(props_cache)} graphs")
        
        # Phase 1: Reorderings
        log_section("Phase 1: Generate Reorderings")
        reorder_results = generate_reorderings(
            graphs=graphs,
            algorithms=algorithms,
            bin_dir=args.bin_dir,
            output_dir=args.results_dir,
            timeout=args.timeout_reorder,
            skip_slow=getattr(args, 'skip_slow', False),
            generate_maps=True,  # Always generate .lo mapping files
            force_reorder=getattr(args, "force_reorder", False)
        )
        
        # Phase 2: Benchmarks (all of them)
        log_section("Phase 2: Execution Benchmarks (All)")
        all_benchmarks = ["pr", "bfs", "cc", "sssp", "bc"]
        benchmark_results = run_benchmarks_multi_graph(
            graphs=graphs,
            algorithms=algorithms,
            benchmarks=all_benchmarks,
            bin_dir=args.bin_dir,
            num_trials=args.trials,
            timeout=args.timeout_benchmark,
            skip_slow=getattr(args, 'skip_slow', False),
            label_maps={},
            weights_dir=args.weights_dir,
            update_weights=True  # Always update incrementally in fill-weights
        )
        
        # Phase 3: Cache Simulation
        log_section("Phase 3: Cache Simulation")
        cache_results = run_cache_simulations(
            graphs=graphs,
            algorithms=algorithms,
            benchmarks=["pr", "bfs"],  # Key benchmarks for cache
            bin_sim_dir=args.bin_sim_dir,
            timeout=args.timeout_sim,
            skip_heavy=getattr(args, 'skip_heavy', True),
            label_maps={}
        )
        
        # Phase 4: Generate Base Weights
        log_section("Phase 4: Generate Perceptron Weights")
        weights = generate_perceptron_weights(
            benchmark_results=benchmark_results,
            cache_results=cache_results,
            reorder_results=reorder_results,
            output_file=args.weights_file
        )
        
        # Phase 5: Update Zero Weights with topology features
        log_section("Phase 5: Update Topology Weights")
        update_zero_weights(
            weights_file=args.weights_file,
            graphs_dir=args.graphs_dir,
            benchmark_results=benchmark_results,
            cache_results=cache_results,
            reorder_results=reorder_results
        )
        
        # Phase 6: Update type-based weights from results
        log_section("Phase 6: Assign Graph Types & Update Weights")
        
        # Extract features for each graph and assign to types
        for graph_info in graphs:
            graph_name = graph_info.name
            graph_path = getattr(graph_info, 'mtx_path', None) or graph_info.path
            
            # Get modularity from adaptive results
            modularity = 0.5
            for ar in adaptive_results:
                if ar.graph == graph_name:
                    modularity = ar.modularity
                    break
            
            # Run a quick benchmark to get topology features
            binary = os.path.join(args.bin_dir, "pr")
            cmd = f"{binary} -f {graph_path} -a 0 -n 1"
            try:
                import subprocess
                result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=60)
                output = result.stdout + result.stderr
                
                # Parse all topology features from C++ output
                dv_match = re.search(r'Degree Variance:\s*([\d.]+)', output)
                hc_match = re.search(r'Hub Concentration:\s*([\d.]+)', output)
                ad_match = re.search(r'Avg Degree:\s*([\d.]+)', output)
                cc_match = re.search(r'Clustering Coefficient:\s*([\d.]+)', output)
                apl_match = re.search(r'Avg Path Length:\s*([\d.]+)', output)
                diam_match = re.search(r'Diameter Estimate:\s*([\d.]+)', output)
                comm_match = re.search(r'Community Count Estimate:\s*([\d.]+)', output)
                
                degree_variance = float(dv_match.group(1)) if dv_match else 1.0
                hub_concentration = float(hc_match.group(1)) if hc_match else 0.3
                avg_degree = float(ad_match.group(1)) if ad_match else 10.0
                clustering_coeff = float(cc_match.group(1)) if cc_match else 0.0
                avg_path_length = float(apl_match.group(1)) if apl_match else 0.0
                diameter = float(diam_match.group(1)) if diam_match else 0.0
                community_count = float(comm_match.group(1)) if comm_match else 0.0
                
                # Assign to type
                features = {
                    'modularity': modularity,
                    'degree_variance': degree_variance,
                    'hub_concentration': hub_concentration,
                    'avg_degree': avg_degree,
                    'clustering_coefficient': clustering_coeff,
                    'avg_path_length': avg_path_length,
                    'diameter': diameter,
                    'community_count': community_count,
                }
                
                # Save all features to graph properties cache
                update_graph_properties(graph_name, {
                    'modularity': modularity,
                    'degree_variance': degree_variance,
                    'hub_concentration': hub_concentration,
                    'avg_degree': avg_degree,
                    'clustering_coefficient': clustering_coeff,
                    'avg_path_length': avg_path_length,
                    'diameter': diameter,
                    'community_count': community_count,
                    'nodes': getattr(graph_info, 'nodes', 0),
                    'edges': getattr(graph_info, 'edges', 0),
                })
                
                type_name = assign_graph_type(features, args.weights_dir, create_if_outlier=True)
                log(f"  {graph_name} → {type_name} (mod={modularity:.3f}, dv={degree_variance:.3f}, hc={hub_concentration:.3f})")
                
                # Update type weights with benchmark results for this graph
                graph_benchmark_results = [r for r in benchmark_results if r.graph == graph_name]
                graph_cache_results = [r for r in cache_results if r.graph == graph_name]
                
                # Find best algorithm for each benchmark
                for bench in set(r.benchmark for r in graph_benchmark_results):
                    bench_results = [r for r in graph_benchmark_results if r.benchmark == bench and r.success]
                    if not bench_results:
                        continue
                    
                    # Find baseline and best
                    baseline = next((r for r in bench_results if 'ORIGINAL' in r.algorithm), bench_results[0])
                    best = min(bench_results, key=lambda r: r.time_seconds)
                    
                    if baseline.time_seconds > 0:
                        speedup = baseline.time_seconds / best.time_seconds
                        
                        # Get cache stats for best algorithm
                        cache_stat = next((c for c in graph_cache_results 
                                          if c.algorithm_name == best.algorithm and c.benchmark == bench), None)
                        
                        # Get reorder time
                        reorder = next((r for r in reorder_results 
                                       if r.graph == graph_name and r.algorithm_name == best.algorithm), None)
                        reorder_time = reorder.reorder_time if reorder else 0.0
                        
                        # Update type weights
                        update_type_weights_incremental(
                            type_name=type_name,
                            algorithm=best.algorithm,
                            benchmark=bench,
                            speedup=speedup,
                            features=features,
                            cache_stats={'l1_hit_rate': (1.0 - cache_stat.l1_miss_rate) * 100 if cache_stat else 0,
                                        'l2_hit_rate': (1.0 - cache_stat.l2_miss_rate) * 100 if cache_stat else 0,
                                        'l3_hit_rate': (1.0 - cache_stat.l3_miss_rate) * 100 if cache_stat else 0} if cache_stat else None,
                            reorder_time=reorder_time,
                            weights_dir=args.weights_dir
                        )
                        
            except Exception as e:
                log(f"  {graph_name}: could not extract features ({e})")
        
        # Show types summary
        known_types = list_known_types(args.weights_dir)
        if known_types:
            log(f"\nTypes created/updated: {len(known_types)}")
            for type_name in known_types:
                type_file = os.path.join(args.weights_dir, f"{type_name}.json")
                if os.path.exists(type_file):
                    with open(type_file) as f:
                        type_data = json.load(f)
                    algo_count = len([k for k in type_data.keys() if not k.startswith('_')])
                    log(f"  {type_name}: {algo_count} algorithms trained")
        else:
            log("No types created (weights are in main perceptron_weights.json)")
        
        # Save graph properties cache for future use
        save_graph_properties_cache(output_dir=os.path.dirname(args.weights_file) or "results")
        cache_file_path = os.path.join(os.path.dirname(args.weights_file) or 'results', "graph_properties_cache.json")
        log(f"Saved graph properties cache to: {cache_file_path}")
        
        # Re-run weight update now that graph properties cache is populated
        # This ensures feature weights (w_modularity, etc.) are computed
        update_zero_weights(
            weights_file=args.weights_file,
            graphs_dir=os.path.dirname(args.weights_file) or "results",
            benchmark_results=benchmark_results,
            cache_results=cache_results,
            reorder_results=reorder_results
        )
        
        # Restore original skip_cache setting
        args.skip_cache = skip_cache_original
        
        log_section("Fill Weights Complete")
        log(f"All weight fields have been populated")
        log(f"Weights file: {args.weights_file}")
        
        # Show summary
        with open(args.weights_file) as f:
            final_weights = json.load(f)
        
        log("\nWeight field population summary:")
        # Pick a non-ORIGINAL algorithm to show (ORIGINAL doesn't learn feature weights)
        sample_algo = next((k for k in final_weights if not k.startswith("_") and k != "ORIGINAL"), None)
        if not sample_algo:
            sample_algo = next((k for k in final_weights if not k.startswith("_")), None)
        if sample_algo:
            w = final_weights[sample_algo]
            for key, val in w.items():
                if key.startswith("_") or key == "benchmark_weights":
                    continue
                status = "✓ filled" if val != 0 else "○ zero"
                log(f"  {key}: {status}")
            
            if "benchmark_weights" in w:
                bw = w["benchmark_weights"]
                all_same = len(set(bw.values())) == 1
                log(f"  benchmark_weights: {'○ defaults' if all_same else '✓ tuned'}")
        
        # Auto-merge weights from this run with previous runs
        if not getattr(args, 'no_merge', False):
            try:
                from scripts.lib.weight_merger import auto_merge_after_run
                log("\nMerging weights with previous runs...")
                merge_summary = auto_merge_after_run()
                if "error" not in merge_summary:
                    log(f"  Merged {merge_summary.get('runs_merged', 0)} runs -> {merge_summary.get('total_types', 0)} types")
                    log(f"  Merged weights saved to: {merge_summary.get('output_dir', 'scripts/weights/merged/')}")
            except ImportError:
                pass  # weight_merger not available
            except Exception as e:
                log(f"  Warning: Weight merge failed: {e}")
    
    # Show final summary with statistics
    _progress.phase_end()
    _progress.final_summary()
    
    log(f"Results directory: {args.results_dir}")
    log(f"Weights directory: {args.weights_dir}")
    
    # Show type system summary
    known_types = list_known_types(args.weights_dir)
    if known_types:
        log(f"Known types: {', '.join(known_types)}")
        log("Run --show-types to see full type details")

# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GraphBrew Unified Experiment Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
=== EVALUATION MODES ===

  # REORDER EVALUATION: Test reordering algorithms only (no graph algorithm benchmarks)
  python scripts/graphbrew_experiment.py --phase reorder --size small

  # BENCHMARK EVALUATION: Run graph algorithm benchmarks (BFS, PR, CC, etc.)
  python scripts/graphbrew_experiment.py --phase benchmark --size small --skip-cache

  # END-TO-END EVALUATION: Full pipeline without weight training
  python scripts/graphbrew_experiment.py --full --size small --auto

  # VALIDATION: Compare AdaptiveOrder vs all fixed algorithms
  python scripts/graphbrew_experiment.py --brute-force --validation-benchmark pr --size small

=== TRAINING MODES ===

  # TRAIN: Complete training pipeline (reorder → benchmark → cache sim → weights)
  python scripts/graphbrew_experiment.py --train --size small --auto

  # TRAIN ITERATIVE: Repeatedly adjust weights until target accuracy
  python scripts/graphbrew_experiment.py --train-iterative --target-accuracy 90 --size small

  # TRAIN BATCHED: Process graphs in batches for large datasets
  python scripts/graphbrew_experiment.py --train-batched --size medium --batch-size 8

=== QUICK START ===

  # ONE-CLICK: Download, build, run full experiment
  python scripts/graphbrew_experiment.py --full --size small --auto

  # QUICK TEST: Key algorithms only (faster)
  python scripts/graphbrew_experiment.py --full --size small --quick --auto

  # ALL VARIANTS: Test all algorithm variants
  python scripts/graphbrew_experiment.py --train --all-variants --size small

=== RUN PHASES SEPARATELY ===

  # Phase 1: Reordering only (generates .lo label maps)
  python scripts/graphbrew_experiment.py --phase reorder --size small

  # Phase 2: Benchmarking only (uses .lo maps from Phase 1)
  python scripts/graphbrew_experiment.py --phase benchmark --size small

  # Phase 3: Cache simulation only (uses .lo maps from Phase 1)
  python scripts/graphbrew_experiment.py --phase cache --size small

  # Phase 4: Generate weights (loads benchmark + cache results from files)
  python scripts/graphbrew_experiment.py --phase weights

  # Run phases sequentially (no --train needed)
  python scripts/graphbrew_experiment.py --phase reorder --size small && \\
  python scripts/graphbrew_experiment.py --phase benchmark --size small && \\
  python scripts/graphbrew_experiment.py --phase cache --size small && \\
  python scripts/graphbrew_experiment.py --phase weights

=== UTILITY COMMANDS ===

  # Download graphs only
  python scripts/graphbrew_experiment.py --download-only --size medium

  # Pre-generate label maps for consistent reordering
  python scripts/graphbrew_experiment.py --precompute --size small

  # Clean and start fresh
  python scripts/graphbrew_experiment.py --clean-all
        """
    )
    
    # Dependency management
    parser.add_argument("--check-deps", action="store_true",
                        help="Check system dependencies (Boost, g++, libnuma, etc.) and exit")
    parser.add_argument("--install-deps", action="store_true",
                        help="Install missing system dependencies (may require sudo)")
    parser.add_argument("--install-boost", action="store_true",
                        help="Download and install Boost 1.58.0 for RabbitOrder compatibility")
    
    # One-click full pipeline
    parser.add_argument("--full", action="store_true",
                        help="Run complete pipeline: download, build, experiment, weights")
    parser.add_argument("--download-only", action="store_true",
                        help="Only download graphs (no experiments)")
    
    # Unified size parameter (user-friendly)
    parser.add_argument("--size", choices=["small", "medium", "large", "xlarge", "all"],
                        default=None, 
                        help="Graph size category - controls both download and filtering. "
                             "small=<50MB, medium=50-500MB, large=500MB-2GB, xlarge=>2GB, all=everything")
    
    # Legacy parameter (kept for backwards compatibility)
    parser.add_argument("--download-size", choices=["SMALL", "MEDIUM", "LARGE", "XLARGE", "ALL"],
                        default=None, help="[DEPRECATED: use --size instead] Size category of graphs to download")
    
    parser.add_argument("--force-download", action="store_true",
                        help="Re-download graphs even if they exist")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip build check (assume binaries exist)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip graph download phase (use existing graphs only)")
    
    # Resource limits
    parser.add_argument("--max-memory", type=float, default=None,
                        help="Maximum RAM (GB) for graph processing. Auto-detects available memory if not set. "
                             "Graphs requiring more memory are automatically skipped.")
    parser.add_argument("--max-disk", type=float, default=None,
                        help="Maximum disk space (GB) for graph downloads. Graphs are skipped if total "
                             "download size would exceed this limit.")
    
    # Auto resource detection (user-friendly combined flag)
    parser.add_argument("--auto", action="store_true",
                        help="Automatically detect memory and disk limits (combines --auto-memory and --auto-disk)")
    parser.add_argument("--auto-memory", action="store_true",
                        help="Automatically skip graphs that won't fit in available system RAM")
    parser.add_argument("--auto-disk", action="store_true",
                        help="Automatically limit downloads to available disk space (uses 80%% of free space)")
    
    parser.add_argument("--min-edges", type=int, default=0,
                        help="Minimum number of edges for graph inclusion. Graphs with fewer edges are skipped. "
                             f"Recommended: {MIN_EDGES_FOR_TRAINING:,} for weight training to avoid noise.")
    
    # Phase selection
    parser.add_argument("--phase", choices=["all", "reorder", "benchmark", "cache", "weights", "adaptive"],
                        default="all", help="Which phase(s) to run")
    
    # Graph selection (kept for backwards compatibility, use --size instead)
    parser.add_argument("--graphs", choices=["all", "small", "medium", "large", "custom"],
                        default=None, help="[DEPRECATED: use --size instead] Graph size category filter")
    parser.add_argument("--graphs-dir", default=DEFAULT_GRAPHS_DIR,
                        help="Directory containing graph datasets")
    parser.add_argument("--graph", "--graph-name", type=str, default=None, dest="graph_name",
                        help="Run on a specific graph by name (e.g., wiki-Talk)")
    parser.add_argument("--graph-list", nargs="+", type=str, default=None, dest="graph_list",
                        help="Run on specific graphs (e.g., --graph-list wiki-Talk web-Google)")
    parser.add_argument("--min-mb", type=float, default=0, dest="min_size",
                        help="Minimum graph file size in MB (for custom filtering)")
    parser.add_argument("--max-mb", type=float, default=float('inf'), dest="max_size",
                        help="Maximum graph file size in MB (for custom filtering)")
    parser.add_argument("--max-graphs", type=int, default=None,
                        help="Maximum number of graphs to test")
    
    # Algorithm selection
    parser.add_argument("--quick", action="store_true", dest="key_only",
                        help="Quick mode: test only key algorithms (Original, Random, HubClusterDBG, "
                             "RabbitOrder, Gorder, RCM, Leiden, LeidenDendrogram, LeidenCSR)")
    parser.add_argument("--algo", "--algorithm", type=str, default=None, dest="algo_name",
                        help="Run only a specific algorithm (e.g., RABBITORDER_boost, LeidenCSR_gve)")
    parser.add_argument("--algo-list", nargs="+", type=str, default=None, dest="algo_list",
                        help="Run specific algorithms (e.g., --algo-list RABBITORDER_boost LeidenCSR_gve GORDER)")
    parser.add_argument("--skip-slow", action="store_true",
                        help="Skip slow algorithms (Gorder, Corder, RCM) on large graphs")
    
    # Benchmark selection
    parser.add_argument("--benchmarks", nargs="+", default=BENCHMARKS,
                        help="Benchmarks to run")
    parser.add_argument("--trials", type=int, default=2,
                        help="Number of trials per benchmark")
    
    # Paths
    parser.add_argument("--bin-dir", default=DEFAULT_BIN_DIR,
                        help="Directory containing benchmark binaries")
    parser.add_argument("--bin-sim-dir", default=DEFAULT_BIN_SIM_DIR,
                        help="Directory containing simulation binaries")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                        help="Directory for results")
    parser.add_argument("--weights-dir", default=DEFAULT_WEIGHTS_DIR,
                        help="Directory for auto-generated type weight files")
    
    # Skip options
    parser.add_argument("--skip-cache", action="store_true",
                        help="Skip cache simulations (saves time, loses cache analysis data)")
    parser.add_argument("--skip-expensive", action="store_true", dest="skip_heavy",
                        help="Skip expensive benchmarks (BC, SSSP) on large graphs (>100MB)")
    
    # Timeouts
    parser.add_argument("--timeout-reorder", type=int, default=TIMEOUT_REORDER,
                        help="Timeout for reordering (seconds)")
    parser.add_argument("--timeout-benchmark", type=int, default=TIMEOUT_BENCHMARK,
                        help="Timeout for benchmarks (seconds)")
    parser.add_argument("--timeout-sim", type=int, default=TIMEOUT_SIM,
                        help="Timeout for simulations (seconds)")
    
    # Adaptive order analysis
    parser.add_argument("--adaptive-analysis", action="store_true",
                        help="Run adaptive order subcommunity analysis")
    parser.add_argument("--adaptive-comparison", action="store_true",
                        help="Compare adaptive vs fixed algorithm performance")
    
    # Label map options (for consistent, reproducible reorderings)
    parser.add_argument("--precompute", action="store_true",
                        help="Pre-generate and use label maps for consistent reordering (combines --generate-maps --use-maps)")
    parser.add_argument("--generate-maps", action="store_true",
                        help="Pre-generate .lo label map files for each algorithm")
    parser.add_argument("--use-maps", action="store_true",
                        help="Use pre-generated label maps (faster, reproducible results)")
    parser.add_argument("--force-reorder", action="store_true",
                        help="Force regeneration of reorderings even if .lo/.time files exist")
    parser.add_argument("--clean-reorder-cache", action="store_true",
                        help="Remove all .lo and .time files to force fresh reordering")
    
    # Leiden variant expansion options
    parser.add_argument("--all-variants", action="store_true", dest="expand_variants",
                        help="Test ALL algorithm variants (Leiden, RabbitOrder) instead of just defaults")
    parser.add_argument("--csr-variants", nargs="+", dest="leiden_csr_variants",
                        default=None, choices=["gve", "gveopt", "gveopt2", "gveadaptive", "gveoptsort", "gveturbo", "gvefast",
                                               "gvedendo", "gveoptdendo", "gverabbit", "dfs", "bfs", "hubsort", "modularity"],
                        help="LeidenCSR variants: gve, gveopt, gveopt2 (CSR aggregation), gveadaptive (dynamic resolution), "
                             "gveoptsort, gveturbo, gvefast (CSR buffer reuse), gvedendo, gveoptdendo, gverabbit, dfs, bfs, hubsort, modularity")
    parser.add_argument("--dendrogram-variants", nargs="+", dest="leiden_dendrogram_variants",
                        default=None, choices=["dfs", "dfshub", "dfssize", "bfs", "hybrid"],
                        help="LeidenDendrogram variants: dfs, dfshub, dfssize, bfs, hybrid")
    parser.add_argument("--rabbit-variants", nargs="+",
                        default=None, choices=["csr", "boost"],
                        help="RabbitOrder variants: csr (default, no deps), boost (requires libboost-graph-dev)")
    parser.add_argument("--graphbrew-variants", nargs="+", dest="graphbrew_variants",
                        default=None, choices=["leiden", "gve", "gveopt", "gvefast", "gveoptfast", "rabbit", "hubcluster"],
                        help="GraphBrewOrder variants: leiden (default), gve, gveopt, gvefast, gveoptfast, rabbit, hubcluster")
    parser.add_argument("--resolution", type=str, default="1.0", dest="leiden_resolution",
                        help="Leiden resolution: fixed (1.5), auto, 0, dynamic, dynamic_2.0 (default: 1.0)")
    parser.add_argument("--passes", type=int, default=3, dest="leiden_passes",
                        help="LeidenCSR refinement passes - higher = better quality (default: 3)")
    
    # Brute-force validation
    parser.add_argument("--brute-force", action="store_true",
                        help="Run brute-force validation: test all algorithms vs AdaptiveOrder choice")
    parser.add_argument("--validation-benchmark", default="pr", dest="bf_benchmark",
                        help="Benchmark to use for brute-force validation (default: pr)")
    parser.add_argument("--validate-adaptive", action="store_true",
                        help="Validate adaptive algorithm accuracy: compare predicted vs actual best")
    
    # Training options
    parser.add_argument("--train", action="store_true", dest="fill_weights",
                        help="Train perceptron weights: runs reorder → benchmark → cache sim → compute weights")
    parser.add_argument("--train-iterative", action="store_true", dest="train_adaptive",
                        help="Iterative training: repeatedly adjust weights until target accuracy is reached")
    parser.add_argument("--train-batched", action="store_true", dest="train_large",
                        help="Batched training: process graphs in batches with multiple benchmarks")
    parser.add_argument("--target-accuracy", type=float, default=80.0,
                        help="Target accuracy %% for iterative training (default: 80)")
    parser.add_argument("--max-iterations", type=int, default=10,
                        help="Maximum training iterations (default: 10)")
    parser.add_argument("--learning-rate", type=float, default=0.1,
                        help="Learning rate for weight adjustments (default: 0.1)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for batched training (default: 8)")
    parser.add_argument("--train-benchmarks", nargs="+", default=['pr', 'bfs', 'cc'],
                        help="Benchmarks to use for multi-benchmark training (default: pr bfs cc)")
    parser.add_argument("--init-weights", action="store_true",
                        help="Initialize empty weights file (run once before first training)")
    
    # Type system options
    parser.add_argument("--show-types", action="store_true",
                        help="Show all known graph types and their statistics")
    parser.add_argument("--batch-only", action="store_true", dest="no_incremental",
                        help="Only update weights at end of run (disable per-graph incremental updates)")
    
    # Weight merging options
    parser.add_argument("--isolate-run", action="store_true", dest="no_merge",
                        help="Keep this run's weights isolated (don't merge with previous runs)")
    parser.add_argument("--list-runs", action="store_true",
                        help="List all saved weight runs")
    parser.add_argument("--merge-runs", nargs="*", metavar="TIMESTAMP",
                        help="Merge specific runs (or all if no args)")
    parser.add_argument("--use-run", metavar="TIMESTAMP",
                        help="Use weights from a specific run instead of merged")
    parser.add_argument("--use-merged", action="store_true",
                        help="Use merged weights (default after merge)")
    
    # Legacy weights file (DEPRECATED - weights now saved to scripts/weights/active/type_0.json)
    parser.add_argument("--weights-file", default=None,
                        help="(DEPRECATED) Legacy flat file. Weights are now saved to scripts/weights/active/type_0.json for C++ to use.")
    
    # Clean options
    parser.add_argument("--clean", action="store_true",
                        help="Clean results directory before running (keeps graphs and weights)")
    parser.add_argument("--clean-all", action="store_true",
                        help="Remove ALL generated data (graphs, results, mappings) - fresh start")
    
    # Auto-setup option
    parser.add_argument("--auto-setup", action="store_true",
                        help="Automatically setup everything: create directories, build if missing, download graphs if needed")
    
    args = parser.parse_args()
    
    # ==========================================================================
    # Parameter Resolution: Unify related flags
    # ==========================================================================
    
    # Resolve --auto flag (combines --auto-memory and --auto-disk)
    if args.auto:
        args.auto_memory = True
        args.auto_disk = True
    
    # Resolve --precompute flag (combines --generate-maps and --use-maps)
    if args.precompute:
        args.generate_maps = True
        args.use_maps = True
    
    # Auto-enable --all-variants when specific variant lists are provided
    if (args.leiden_csr_variants or args.leiden_dendrogram_variants or args.rabbit_variants):
        if not args.expand_variants:
            args.expand_variants = True
            log("Auto-enabling variant expansion (specific variants requested)", "INFO")
    
    # Resolve unified --size parameter
    # Priority: --size > --graphs > --download-size > default (all)
    if args.size is not None:
        # New unified --size parameter takes precedence
        size_lower = args.size.lower()
        args.graphs = size_lower if size_lower != "xlarge" else "all"
        args.download_size = args.size.upper()
        log(f"Using --size {args.size}: graphs={args.graphs}, download={args.download_size}", "INFO")
    elif args.graphs is not None:
        # Legacy --graphs parameter
        args.download_size = args.graphs.upper() if args.download_size is None else args.download_size
        log(f"Using --graphs {args.graphs} (deprecated, use --size instead)", "INFO")
    elif args.download_size is not None:
        # Legacy --download-size parameter
        args.graphs = args.download_size.lower() if args.download_size != "XLARGE" else "all"
        log(f"Using --download-size {args.download_size} (deprecated, use --size instead)", "INFO")
    else:
        # Default: all graphs
        args.graphs = "all"
        args.download_size = "SMALL"  # Conservative default for downloads
    
    # Handle dependency management (before anything else)
    if args.check_deps or args.install_deps or args.install_boost:
        if not HAS_DEPENDENCY_MANAGER:
            print("ERROR: Dependency manager not available. Check scripts/lib/dependencies.py")
            sys.exit(1)
        
        if args.install_boost:
            # Import Boost-specific functions
            try:
                from scripts.lib.dependencies import install_boost_158, check_boost_158
            except ImportError:
                print("ERROR: Could not import Boost installation functions")
                sys.exit(1)
            
            # Check if already installed
            is_installed, msg = check_boost_158()
            if is_installed:
                print(f"✓ {msg}")
                sys.exit(0)
            
            print("Downloading and installing Boost 1.58.0 for RabbitOrder...")
            print("(This may require sudo password)")
            success, msg = install_boost_158(verbose=True)
            print(msg)
            if success:
                print("\nVerifying installation...")
                is_ok, verify_msg = check_boost_158()
                print(f"  {'✓' if is_ok else '✗'} {verify_msg}")
            sys.exit(0 if success else 1)
        
        if args.check_deps:
            all_ok, status = lib_check_dependencies(verbose=True)
            if not all_ok:
                print("\nTo install missing dependencies:")
                print("  python scripts/graphbrew_experiment.py --install-deps")
                print("\nOr see manual instructions:")
                print("  python -m scripts.lib.dependencies --instructions")
            sys.exit(0 if all_ok else 1)
        
        if args.install_deps:
            print("Installing missing system dependencies...")
            print("(This may require sudo password)")
            success, msg = lib_install_dependencies(verbose=True)
            print(msg)
            if success:
                print("\nVerifying installation...")
                lib_check_dependencies(verbose=True)
            sys.exit(0 if success else 1)
    
    # Determine memory limit
    if args.auto_memory and args.max_memory is None:
        # Auto-detect available memory, use 60% of total as safe limit
        total_mem = get_total_memory_gb()
        args.max_memory = total_mem * 0.6
        log(f"Auto-detected memory limit: {args.max_memory:.1f} GB (60% of {total_mem:.1f} GB total)", "INFO")
    elif args.max_memory is not None:
        log(f"Using specified memory limit: {args.max_memory:.1f} GB", "INFO")
    
    # Determine disk space limit
    if args.auto_disk and args.max_disk is None:
        # Auto-detect available disk space, use 80% of free space
        free_disk = get_available_disk_gb(args.graphs_dir if os.path.exists(args.graphs_dir) else ".")
        args.max_disk = free_disk * 0.8
        log(f"Auto-detected disk limit: {args.max_disk:.1f} GB (80% of {free_disk:.1f} GB free)", "INFO")
    elif args.max_disk is not None:
        log(f"Using specified disk limit: {args.max_disk:.1f} GB", "INFO")
    
    # Log min_edges filter if specified (works with any download_size)
    if getattr(args, 'min_edges', 0) > 0:
        log(f"Min edges filter: {args.min_edges:,} (graphs with fewer edges will be skipped for training)", "INFO")
    
    # Handle clean operations first
    if args.clean_all:
        clean_all(".", confirm=False)
        if not (args.full or args.download_only):
            return  # Just clean, don't run experiments
    elif args.clean:
        clean_results(args.results_dir, keep_graphs=True, keep_weights=True)
        if not (args.full or args.download_only or args.phase != "all"):
            return  # Just clean, don't run experiments
    
    # Handle --clean-reorder-cache early
    if getattr(args, 'clean_reorder_cache', False):
        clean_reorder_cache(args.graphs_dir)
        if not (args.full or args.download_only or args.phase != "all" or 
                getattr(args, 'validate_adaptive', False) or getattr(args, 'brute_force', False)):
            return  # Just clean cache, don't run experiments
    
    # Handle weight management commands early
    if getattr(args, 'list_runs', False):
        from scripts.lib.weight_merger import list_runs
        runs = list_runs()
        if not runs:
            log("No saved runs found")
        else:
            log(f"Available runs ({len(runs)}):\n")
            for run in runs:
                log(f"  {run.timestamp}:")
                log(f"    Types: {len(run.types)}")
                for tid, tinfo in run.types.items():
                    log(f"      {tid}: {tinfo.graph_count} graphs, {len(tinfo.algorithms)} algos")
        return
    
    if getattr(args, 'merge_runs', None) is not None:
        from scripts.lib.weight_merger import merge_runs, use_merged
        run_timestamps = args.merge_runs if args.merge_runs else None
        summary = merge_runs(run_timestamps=run_timestamps)
        if "error" not in summary:
            use_merged()
            log(f"Merged {summary.get('runs_merged', 0)} runs -> {summary.get('total_types', 0)} types")
        return
    
    if getattr(args, 'use_run', None):
        from scripts.lib.weight_merger import use_run
        if use_run(args.use_run):
            log(f"Now using weights from run: {args.use_run}")
        return
    
    if getattr(args, 'use_merged', False):
        from scripts.lib.weight_merger import use_merged
        if use_merged():
            log("Now using merged weights")
        return
    
    # Handle --show-types early (informational command)
    if getattr(args, 'show_types', False):
        log_section("Known Graph Types")
        load_type_registry(args.weights_dir)
        known_types = list_known_types(args.weights_dir)
        if not known_types:
            log("No types defined yet. Types are auto-created when graphs are processed.")
            log(f"Types will be stored in: {args.weights_dir}")
        else:
            for type_name in sorted(known_types):
                summary = get_type_summary(type_name, args.weights_dir)
                if summary:
                    log(f"\n{type_name.upper()}:")
                    log(f"  Graphs trained: {summary.get('num_graphs', 0)}")
                    if 'best_algorithms' in summary:
                        for bench, algo in summary['best_algorithms'].items():
                            log(f"  Best for {bench}: {algo}")
                    # Show representative features instead of centroid (more readable)
                    if 'representative_features' in summary and summary['representative_features']:
                        rf = summary['representative_features']
                        log(f"  Features: modularity={rf.get('modularity', 0):.3f}, "
                            f"avg_degree={rf.get('avg_degree', 0):.1f}")
                    
                    # Show best Leiden variants for each benchmark
                    log(f"  Leiden variant rankings:")
                    for bench in ['pr', 'bfs', 'cc']:
                        # LeidenCSR variants
                        csr_rankings = get_leiden_variant_rankings(type_name, 17, bench, args.weights_dir)
                        if csr_rankings and csr_rankings[0][1] > 0:
                            best_csr = csr_rankings[0]
                            log(f"    {bench}/LeidenCSR: {best_csr[0].split('_')[-1]} (score: {best_csr[1]:.3f})")
                        
                        # LeidenDendrogram variants
                        dendro_rankings = get_leiden_variant_rankings(type_name, 16, bench, args.weights_dir)
                        if dendro_rankings and dendro_rankings[0][1] > 0:
                            best_dendro = dendro_rankings[0]
                            log(f"    {bench}/LeidenDendro: {best_dendro[0].split('_')[-1]} (score: {best_dendro[1]:.3f})")
        return  # Exit after showing types
    
    # ALWAYS ensure prerequisites at start (unless skip_build is set)
    if not getattr(args, 'skip_build', False):
        ensure_prerequisites(
            project_dir=".",
            graphs_dir=args.graphs_dir,
            results_dir=args.results_dir,
            weights_dir=args.weights_dir,
            rebuild=False
        )
    else:
        # At least ensure directories exist
        os.makedirs(args.results_dir, exist_ok=True)
        os.makedirs(args.weights_dir, exist_ok=True)
    
    # Auto-setup: download graphs if needed
    if args.auto_setup or args.fill_weights or args.full:
        log_section("Auto-Setup: Downloading Graphs")
        # 3. Download ALL requested graphs FIRST (before any experiments)
        # This ensures all graphs are ready before we start reordering/benchmarks
        log("Downloading graphs (ensuring all are ready before experiments)...")
        downloaded = download_graphs(
            size_category=args.download_size,
            graphs_dir=args.graphs_dir,
            force=False,  # Don't re-download existing
            max_memory_gb=args.max_memory,
            max_disk_gb=args.max_disk
        )
        
        # Now discover all available graphs
        graphs = discover_graphs(args.graphs_dir, max_memory_gb=args.max_memory,
                                 min_edges=getattr(args, 'min_edges', 0))
        if not graphs:
            log("No graphs found after download - aborting", "ERROR")
            sys.exit(1)
        log(f"Total graphs ready: {len(graphs)}")
        
        # 4. Initialize type registry if needed
        load_type_registry(args.weights_dir)
        log(f"Type registry loaded from: {args.weights_dir}")
        known_types = list_known_types(args.weights_dir)
        if known_types:
            log(f"Known types: {', '.join(known_types)}")
        else:
            log("No existing types - will create as graphs are processed")
        
        log("Auto-setup complete - all graphs downloaded and ready\n")
    
    try:
        # Handle download-only mode
        if args.download_only:
            download_graphs(
                size_category=args.download_size,
                graphs_dir=args.graphs_dir,
                force=args.force_download,
                max_memory_gb=args.max_memory,
                max_disk_gb=args.max_disk
            )
            log("Download complete. Run without --download-only to start experiments.", "INFO")
            return
        
        # Handle full pipeline mode
        if args.full:
            log("="*60, "INFO")
            log("GRAPHBREW ONE-CLICK EXPERIMENT PIPELINE", "INFO")
            log("="*60, "INFO")
            log(f"Configuration: size={args.download_size}, graphs={args.graphs}", "INFO")
            
            # Step 1: Download graphs (unless --skip-download is set)
            if not args.skip_download:
                downloaded = download_graphs(
                    size_category=args.download_size,
                    graphs_dir=args.graphs_dir,
                    force=args.force_download,
                    max_memory_gb=args.max_memory,
                    max_disk_gb=args.max_disk
                )
                
                if not downloaded:
                    log("No graphs downloaded - aborting", "ERROR")
                    sys.exit(1)
            else:
                log("Skipping download (--skip-download), using existing graphs", "INFO")
            
            # Step 2: Build binaries
            if not args.skip_build:
                if not check_and_build_binaries("."):
                    log("Build failed - aborting", "ERROR")
                    sys.exit(1)
            
            # Step 3: Enable label map generation for consistent reordering
            args.generate_maps = True
            args.use_maps = True
            
            # Step 4: Run experiment
            log("\nStarting experiments...", "INFO")
            # Note: args.graphs and args.download_size are already synchronized
            # by the parameter resolution at the top of main()
        
        run_experiment(args)
        
    except KeyboardInterrupt:
        log("\nExperiment interrupted by user", "WARN")
        sys.exit(1)
    except Exception as e:
        log(f"Experiment failed: {e}", "ERROR")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
