#!/usr/bin/env python3
"""
GraphBrew Unified Experiment Pipeline
=====================================

A comprehensive one-click script that runs the complete GraphBrew experiment workflow:

**Core Pipeline (--phase all):**
 1. Download graphs (if not present) — via --download-only or --full
 2. Build binaries (if not present) — via --build or --full
 3. Convert .mtx → .sg with RANDOM baseline (--random-baseline, default ON)
 4. Pre-generate reordered .sg per algorithm (--pregenerate-sg, default ON)
    Falls back to real-time reordering if disk space is insufficient.
 5. Phase 1: Generate reorderings (12 algorithms; baselines ORIGINAL/RANDOM are skipped)
 6. Phase 2: Run benchmarks (14 algorithms × 7 benchmarks; TC excluded by default)
 7. Phase 3: Run cache simulations (optional, skip with --skip-cache)
 8. Phase 4: Generate type-based perceptron weights (results/weights/type_*/weights.json)

**Validation & Analysis:**
 9. Phase 5: Adaptive order analysis (--adaptive-analysis)
10. Phase 6: Adaptive vs fixed comparison (--adaptive-comparison)
11. Phase 7: Brute-force validation (--brute-force)

**Training Modes:**
12. Standard training (--train): One-pass pipeline that runs all phases
13. Iterative training (--train-iterative): Repeatedly adjusts weights until target accuracy
14. Batched training (--train-batched): Process graphs in batches for large datasets

**Algorithm Variant Testing:**
    For GraphBrewOrder (12) and RabbitOrder (8), you can test
    specific variants or all variants.
    
    # Test all algorithm variants
    python scripts/graphbrew_experiment.py --train --all-variants --size small
    
    # Test specific variants only
    python scripts/graphbrew_experiment.py --train --graphbrew-variants leiden rabbit --size small
    
    # With custom Leiden parameters
    python scripts/graphbrew_experiment.py --train --all-variants \\
        --resolution 1.0 --passes 5 --size medium
    
    RabbitOrder (8) variants: csr (default), boost
    GraphBrewOrder (12) ordering strategies:
      - (default): Leiden + per-community RabbitOrder
      - hrab: Hybrid Leiden+RabbitOrder (best locality) ⭐
      - dfs, bfs: Dendrogram traversal orderings
      - conn: Connectivity BFS within communities
      - rabbit: RabbitOrder single-pass pipeline
    
    Resolution modes (for --resolution):
      - Fixed: 1.5 (use specified value)
      - Auto: auto or 0 (compute from graph density/CV)
      - Dynamic: dynamic (adjust per-pass)

All outputs are saved to the results/ directory for clean organization.
Type-based weights are saved to results/weights/type_*/weights.json.

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
import glob
import json
import os
import re
import shutil
import sys
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure project root is in path for lib imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import shared utilities from lib/ module (REQUIRED)
# The lib/ module is part of this project and must be available
from scripts.lib import (
    # Core constants
    BENCHMARKS,
    # Variant definitions
    RABBITORDER_VARIANTS,
    # Graph property helpers
    load_graph_properties_cache,
    save_graph_properties_cache,
    update_graph_properties,
    get_total_memory_gb,
    estimate_graph_memory_gb,
    get_available_disk_gb,
    # Download
    DOWNLOAD_GRAPHS_SMALL,
    DOWNLOAD_GRAPHS_MEDIUM,
    DOWNLOAD_GRAPHS_LARGE,
    DOWNLOAD_GRAPHS_XLARGE,
    download_graphs_parallel,
    download_graph as lib_download_graph,
    # Cache
    run_cache_simulations,
    load_type_registry,
    assign_graph_type,
    update_type_weights_incremental,
    list_known_types,
    get_type_weights_file,
    get_type_summary,
    # Progress
    ProgressTracker,
)

# Try to import dependency manager
try:
    from scripts.lib.dependencies import (
        check_dependencies,
        install_dependencies,
    )
    HAS_DEPENDENCY_MANAGER = True
except ImportError:
    HAS_DEPENDENCY_MANAGER = False

# ============================================================================
# Configuration - Import from Single Source of Truth (lib/utils.py)
# ============================================================================

# Import algorithm definitions and constants from utils.py (Single Source of Truth)
from scripts.lib.utils import (
    ALGORITHMS, ALGORITHM_IDS, ELIGIBLE_ALGORITHMS, REORDER_ALGORITHMS,
    RESULTS_DIR, GRAPHS_DIR, BIN_DIR, BIN_SIM_DIR, WEIGHTS_DIR,
    SIZE_SMALL, SIZE_MEDIUM, SIZE_LARGE,
    LEIDEN_DEFAULT_PASSES,
    TIMEOUT_BUILD, TIMEOUT_REORDER, TIMEOUT_BENCHMARK, TIMEOUT_SIM,
    _VARIANT_ALGO_REGISTRY, GORDER_VARIANTS,
    CHAINED_ORDERINGS,
    is_variant_prefixed,
    canonical_algo_key, algo_converter_opt, get_algo_variants,
    run_command,
    get_graph_dimensions,
    EXPERIMENT_BENCHMARKS,
)

# Named baseline IDs — prefer these over magic numbers 0, 1
_ORIGINAL_ID = ALGORITHM_IDS["ORIGINAL"]   # 0
_RANDOM_ID   = ALGORITHM_IDS["RANDOM"]     # 1

# Algorithms to benchmark — SSOT alias (excludes MAP=13, AdaptiveOrder=14)
# Includes baselines ORIGINAL (0) and RANDOM (1) which establish reference times.
BENCHMARK_ALGORITHMS = ELIGIBLE_ALGORITHMS

# Subset of key algorithms for quick testing
KEY_ALGORITHMS = [
    ALGORITHM_IDS["ORIGINAL"], ALGORITHM_IDS["RANDOM"],
    ALGORITHM_IDS["HUBCLUSTERDBG"], ALGORITHM_IDS["RABBITORDER"],
    ALGORITHM_IDS["GORDER"], ALGORITHM_IDS["RCM"],
    ALGORITHM_IDS["GraphBrewOrder"], ALGORITHM_IDS["LeidenOrder"],
]
assert all(a in ALGORITHMS for a in KEY_ALGORITHMS), "KEY_ALGORITHMS contains unknown IDs"

# ============================================================================
# Algorithm Configuration with Variant Support
# ============================================================================

# Default paths — derived from SSOT (lib/utils.py), converted to strings for argparse
DEFAULT_RESULTS_DIR = str(RESULTS_DIR)
DEFAULT_GRAPHS_DIR = str(GRAPHS_DIR)
DEFAULT_BIN_DIR = str(BIN_DIR)
DEFAULT_BIN_SIM_DIR = str(BIN_SIM_DIR)
DEFAULT_WEIGHTS_DIR = str(WEIGHTS_DIR)

# Training defaults
DEFAULT_TRAIN_BENCHMARKS = ['pr', 'bfs', 'cc']  # Fast subset of EXPERIMENT_BENCHMARKS
assert all(b in BENCHMARKS for b in DEFAULT_TRAIN_BENCHMARKS), "DEFAULT_TRAIN_BENCHMARKS has unknown entries"

# Minimum edges for training (skip small graphs that introduce noise/skew)
MIN_EDGES_FOR_TRAINING = 100000  # 100K edges

# Default modularity for graphs without adaptive analysis results.
# 0.5 is a neutral mid-range value (range 0-1); real graphs typically 0.3-0.8.
DEFAULT_MODULARITY = 0.5

# Auto-resource safety margins
AUTO_MEMORY_FRACTION = 0.6   # Use 60% of total RAM
AUTO_DISK_FRACTION = 0.8     # Use 80% of free disk

# Cache simulation benchmarks (subset of BENCHMARKS for speed)
CACHE_KEY_BENCHMARKS = ["pr", "bfs"]
assert all(b in BENCHMARKS for b in CACHE_KEY_BENCHMARKS), "CACHE_KEY_BENCHMARKS has unknown entries"

# Feature extraction timeout (quick PR run, not a full benchmark)
TIMEOUT_FEATURE_EXTRACTION = 60

# Graph file search: preferred extension order and MTX auxiliary file suffixes to skip
_GRAPH_EXTENSIONS = [".sg", ".mtx", ".el"]
_MTX_AUX_SUFFIXES = ['_coord', '_nodename', '_Categories', '_b.mtx']

# Size category → (min_mb, max_mb) mapping for graph discovery
_SIZE_RANGES = {
    "all":    (0, float('inf')),
    "small":  (0, SIZE_SMALL),
    "medium": (SIZE_SMALL, SIZE_MEDIUM),
    "large":  (SIZE_MEDIUM, SIZE_LARGE),
    "xlarge": (SIZE_LARGE, float('inf')),
}

# Feature patterns parsed from C++ topology output (hoisted to module level)
_FEATURE_PATTERNS = [
    # (regex_label, feature_key, default_value)
    (r'Degree Variance:\s*([\d.]+)', 'degree_variance', 1.0),
    (r'Hub Concentration:\s*([\d.]+)', 'hub_concentration', 0.3),
    (r'Avg Degree:\s*([\d.]+)', 'avg_degree', 10.0),
    (r'Clustering Coefficient:\s*([\d.]+)', 'clustering_coefficient', 0.0),
    (r'Avg Path Length:\s*([\d.]+)', 'avg_path_length', 0.0),
    (r'Diameter Estimate:\s*([\d.]+)', 'diameter', 0.0),
    (r'Community Count Estimate:\s*([\d.]+)', 'community_count', 0.0),
    (r'Packing Factor:\s*([\d.]+)', 'packing_factor', 0.0),
    (r'Forward Edge Fraction:\s*([\d.]+)', 'forward_edge_fraction', 0.5),
    (r'Working Set Ratio:\s*([\d.]+)', 'working_set_ratio', 0.0),
    (r'Graph Density:\s*([\d.eE+-]+)', 'density', 0.0),
]

# ============================================================================
# Data Classes
# ============================================================================

# Use the canonical GraphInfo from lib/graph_types.py (includes extended properties)
from scripts.lib.graph_types import GraphInfo

# Pipeline stage imports from lib/ modules
from scripts.lib.reorder import (
    generate_reorderings,
    generate_reorderings_with_variants,
    generate_label_maps,
    load_label_maps_index,
)
from scripts.lib.benchmark import run_benchmarks_multi_graph, run_benchmarks_with_variants
from scripts.lib.cache import run_cache_simulations_with_variants
from scripts.lib.analysis import (
    analyze_adaptive_order,
    compare_adaptive_vs_fixed,
    run_subcommunity_brute_force,
    validate_adaptive_accuracy,
)
from scripts.lib.training import (
    train_adaptive_weights_iterative,
    train_adaptive_weights_large_scale,
    initialize_enhanced_weights,
)
from scripts.lib.weights import compute_weights_from_results, update_zero_weights
from scripts.lib.metrics import compute_amortization, format_amortization_table
from scripts.lib.graph_data import set_session_id
from scripts.lib.weight_merger import auto_merge_after_run


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

def _print_amortization_report(benchmark_results, reorder_results, max_rows: int = 25):
    """Print amortization report if data is available."""
    if not benchmark_results or not reorder_results:
        return
    try:
        bench_dicts = [asdict(r) for r in benchmark_results]
        reorder_dicts = [asdict(r) for r in reorder_results]
        report = compute_amortization(bench_dicts, reorder_dicts)
        if report.entries:
            log("\n" + format_amortization_table(report, max_rows=max_rows))
    except Exception as e:
        log(f"  Note: Amortization report skipped: {e}")

def get_graph_path(graphs_dir: str, graph_name: str) -> Optional[str]:
    """Get the path to a graph file.

    Preferred order: .sg > .mtx > .el.  When a randomly-reordered .sg
    baseline exists it is returned instead of the raw .mtx so that all
    benchmark measurements are relative to a random-baseline ordering.
    """
    graph_folder = os.path.join(graphs_dir, graph_name)

    # Try variations of the graph name (hyphen vs underscore)
    name_variants = [graph_name, graph_name.replace('-', '_'), graph_name.replace('_', '-')]

    # Check for graph files with the graph name (downloaded format)
    for name in name_variants:
        for ext in _GRAPH_EXTENSIONS:
            path = os.path.join(graph_folder, f"{name}{ext}")
            if os.path.exists(path):
                return path

    # Check for generic "graph" name (legacy format)
    for ext in _GRAPH_EXTENSIONS:
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
                    for ext in _GRAPH_EXTENSIONS:
                        path = os.path.join(subdir_path, f"{name}{ext}")
                        if os.path.exists(path):
                            return path

    # Last resort: look for any .mtx file in the folder (excluding auxiliary files)
    if os.path.isdir(graph_folder):
        for f in sorted(os.listdir(graph_folder)):
            if f.endswith('.mtx') and not any(skip in f for skip in _MTX_AUX_SUFFIXES):
                full_path = os.path.join(graph_folder, f)
                if os.path.isfile(full_path):
                    return full_path

    return None

def get_graph_size_mb(path: str) -> float:
    """Get the size of a graph file in MB."""
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0.0


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
    
    # Also check SSOT graphs dir if it exists and isn't already in the list
    ssot_graphs_dir = str(GRAPHS_DIR)
    if os.path.exists(ssot_graphs_dir) and ssot_graphs_dir not in dirs_to_scan:
        dirs_to_scan.append(ssot_graphs_dir)
    
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
            except (OSError, json.JSONDecodeError):
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
        log(f"Unknown size category: {size_category}", "WARN")
        log("Valid options: SMALL, MEDIUM, LARGE, XLARGE, ALL")
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
        log("No graphs to download after filtering", "WARN")
        return []
    
    # Log filtering summary if any graphs were skipped
    if skipped_memory:
        log(f"Skipped {len(skipped_memory)} graphs exceeding memory limit ({max_memory_gb:.1f} GB)", "WARN")
    if skipped_disk:
        log(f"Skipped {len(skipped_disk)} graphs exceeding disk limit ({max_disk_gb:.1f} GB)", "WARN")
    
    # Create graphs directory
    os.makedirs(graphs_dir, exist_ok=True)
    
    # Use parallel download (blocks until all complete)
    if parallel:
        successful_paths, failed_names = download_graphs_parallel(
            graphs=[g.name for g in graphs_to_download],
            dest_dir=Path(graphs_dir),
            max_workers=max_workers,
            force=force,
            show_progress=True,
            wait_for_all=True,  # Always wait for all downloads
        )
        if failed_names:
            log(f"Failed to download {len(failed_names)} graphs: {', '.join(failed_names)}", "WARN")
        successful = [p.parent.name for p in successful_paths]
    else:
        # Sequential fallback
        log_section("GRAPH DOWNLOAD (Sequential)")
        
        successful = []
        for i, graph in enumerate(graphs_to_download, 1):
            log(f"[{i}/{len(graphs_to_download)}] {graph.name}")
            result = lib_download_graph(graph, dest_dir=Path(graphs_dir), force=force)
            if result is not None:
                successful.append(graph.name)
        
        log(f"Download complete: {len(successful)}/{len(graphs_to_download)} successful")
    
    return successful


# ============================================================================
# Random-Baseline .sg Conversion
# ============================================================================

def _find_mtx_for_graph(graph_dir: str, graph_name: str) -> Optional[str]:
    """Locate the .mtx file for a graph, searching name variants and subdirs."""
    name_variants = [graph_name, graph_name.replace('-', '_'), graph_name.replace('_', '-')]
    for name in name_variants:
        p = os.path.join(graph_dir, f"{name}.mtx")
        if os.path.isfile(p):
            return p
    # Check subdirectories (SuiteSparse layout)
    for subdir in os.listdir(graph_dir) if os.path.isdir(graph_dir) else []:
        sp = os.path.join(graph_dir, subdir)
        if os.path.isdir(sp):
            for name in name_variants + [subdir]:
                p = os.path.join(sp, f"{name}.mtx")
                if os.path.isfile(p):
                    return p
    # Last resort: any .mtx excluding auxiliary files
    if os.path.isdir(graph_dir):
        for f in sorted(os.listdir(graph_dir)):
            if f.endswith('.mtx') and not any(x in f for x in _MTX_AUX_SUFFIXES):
                fp = os.path.join(graph_dir, f)
                if os.path.isfile(fp):
                    return fp
    return None


def convert_graph_to_sg(
    mtx_path: str,
    sg_path: str,
    order: int = 1,
    bin_dir: str = None,
    timeout: int = 600,
) -> bool:
    """Convert a single graph from .mtx to .sg with the given reordering.

    Args:
        mtx_path: Path to input .mtx file.
        sg_path:  Path for the output .sg file.
        order:    Reorder algorithm ID applied during conversion
                  (default 1 = RANDOM).
        bin_dir:  Directory containing the converter binary.
        timeout:  Timeout in seconds.

    Returns:
        True if conversion succeeded, False otherwise.
    """
    bin_dir = bin_dir or str(BIN_DIR)
    converter = os.path.join(bin_dir, "converter")
    if not os.path.isfile(converter):
        log(f"Converter binary not found: {converter}", "ERROR")
        return False

    cmd = f"{converter} -f {mtx_path} -s -o {order} -b {sg_path}"
    success, stdout, stderr = run_command(cmd, timeout=timeout)
    if not success:
        log(f"Conversion failed for {mtx_path}: {stderr[:200]}", "ERROR")
        # Remove partial output
        if os.path.isfile(sg_path):
            os.remove(sg_path)
        return False

    if not os.path.isfile(sg_path) or os.path.getsize(sg_path) == 0:
        log(f"Converter produced empty .sg for {mtx_path}", "ERROR")
        if os.path.isfile(sg_path):
            os.remove(sg_path)
        return False

    return True


def convert_graphs_to_sg(
    graphs_dir: str = DEFAULT_GRAPHS_DIR,
    order: int = 1,
    bin_dir: str = None,
    force: bool = False,
    graph_names: List[str] = None,
    timeout: int = 600,
) -> int:
    """Convert all discovered .mtx graphs to .sg with the specified ordering.

    When *order=1* (RANDOM, the default) this creates a random-baseline
    .sg so that subsequent benchmark runs with ``-o 0`` (ORIGINAL) measure
    performance on a randomly-ordered graph, and every reordering algorithm
    shows its *improvement* over that worst-case baseline.

    Args:
        graphs_dir:   Root directory containing graph sub-directories.
        order:        Reorder algorithm ID applied during conversion
                      (default 1 = RANDOM).
        bin_dir:      Directory containing the converter binary.
        force:        If True, overwrite existing .sg files.
        graph_names:  Optional list limiting conversion to these graph names.
        timeout:      Per-graph conversion timeout in seconds.

    Returns:
        Number of graphs successfully converted.
    """
    bin_dir = bin_dir or str(BIN_DIR)
    algo_label = canonical_algo_key(order)

    log_section(f"CONVERTING GRAPHS TO .sg (baseline={algo_label})")

    converted = 0
    skipped = 0
    failed = 0
    entries = sorted(os.listdir(graphs_dir)) if os.path.isdir(graphs_dir) else []

    for entry in entries:
        entry_path = os.path.join(graphs_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        if graph_names and entry not in graph_names:
            continue

        # Determine output .sg path
        sg_path = os.path.join(entry_path, f"{entry}.sg")

        # Skip if .sg already exists and force is not set
        if os.path.isfile(sg_path) and os.path.getsize(sg_path) > 0 and not force:
            skipped += 1
            continue

        # Locate the .mtx source
        mtx_path = _find_mtx_for_graph(entry_path, entry)
        if not mtx_path:
            continue  # no .mtx — nothing to convert

        log(f"  Converting {entry} ({algo_label}) ...")
        if convert_graph_to_sg(mtx_path, sg_path, order=order, bin_dir=bin_dir, timeout=timeout):
            converted += 1
            sg_size_mb = os.path.getsize(sg_path) / (1024 * 1024)
            log(f"    -> {sg_path} ({sg_size_mb:.1f} MB)")
        else:
            failed += 1

    log(f"Conversion complete: {converted} converted, {skipped} skipped (exist), {failed} failed")
    return converted


# ============================================================================
# Pre-generated Reordered .sg Files
# ============================================================================

def _iter_algo_variants(
    algorithms: List[int],
    include_chains: bool = True,
) -> List[Tuple[int, str, str]]:
    """Expand algorithm IDs into ``(algo_id, canonical_name, converter_opt)`` triples.

    For variant algorithms (RABBITORDER, GraphBrewOrder, RCM) each variant is
    emitted separately — they produce *different* reorderings and therefore
    need their own ``.sg`` file::

        8 → [(8, 'RABBITORDER_csr',      '8:csr'),
             (8, 'RABBITORDER_boost',     '8:boost')]
       12 → [(12, 'GraphBrewOrder_leiden',  '12:leiden'),
             (12, 'GraphBrewOrder_rabbit',  '12:rabbit'),
             (12, 'GraphBrewOrder_hubcluster', '12:hubcluster')]
        2 → [(2, 'SORT', '2')]

    When *include_chains* is ``True`` (default), ``CHAINED_ORDERINGS`` from the
    SSOT are appended.  Chained orderings apply multiple ``-o`` flags
    sequentially and produce a compound reordered ``.sg``::

        ('SORT+RABBITORDER_csr', '-o 2 -o 8:csr')  →  email-Enron_SORT+RABBITORDER_csr.sg

    The *algo_id* for chained orderings is ``-1`` (no single ID).
    """
    result: List[Tuple[int, str, str]] = []
    for algo_id in algorithms:
        variants = get_algo_variants(algo_id)
        if variants:
            for v in variants:
                result.append((
                    algo_id,
                    canonical_algo_key(algo_id, v),
                    algo_converter_opt(algo_id, v),
                ))
        else:
            result.append((algo_id, canonical_algo_key(algo_id), algo_converter_opt(algo_id)))

    # Append chained (multi-pass) orderings from SSOT
    if include_chains:
        for canonical, converter_opts in CHAINED_ORDERINGS:
            result.append((-1, canonical, converter_opts))

    return result


def get_pregenerated_sg_path(graph_dir: str, graph_name: str, algo_id: int,
                             variant: str = None) -> str:
    """Return the expected path for a pre-generated reordered .sg file.

    Naming convention: ``{graph_name}_{CANONICAL_NAME}.sg``

    For variant algorithms the *variant* suffix is appended automatically::

        algo_id=2             → email-Enron_SORT.sg
        algo_id=7             → email-Enron_HUBCLUSTERDBG.sg
        algo_id=8, variant='' → email-Enron_RABBITORDER_csr.sg   (default)
        algo_id=8, variant='boost' → email-Enron_RABBITORDER_boost.sg
        algo_id=12, variant='leiden' → email-Enron_GraphBrewOrder_leiden.sg
    """
    canonical = canonical_algo_key(algo_id, variant)
    return os.path.join(graph_dir, f"{graph_name}_{canonical}.sg")


def estimate_pregeneration_size(
    graphs_dir: str,
    algorithms: List[int] = None,
    graph_names: List[str] = None,
) -> Tuple[int, int, bool]:
    """Estimate disk space needed to pre-generate reordered .sg files.

    Each reordered .sg is approximately the same size as the baseline .sg
    (vertex/edge counts are preserved; only vertex ordering changes).

    Args:
        graphs_dir:   Root directory containing graph sub-directories.
        algorithms:   Reorder algorithm IDs (default: REORDER_ALGORITHMS).
        graph_names:  Optional list limiting estimation to these graphs.

    Returns:
        ``(estimated_bytes, available_bytes, fits)`` — *fits* is True when
        the estimated space does not exceed the free disk space.
    """
    algorithms = algorithms or REORDER_ALGORITHMS
    algo_variants = _iter_algo_variants(algorithms)
    estimated = 0
    entries = sorted(os.listdir(graphs_dir)) if os.path.isdir(graphs_dir) else []

    for entry in entries:
        entry_path = os.path.join(graphs_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        if graph_names and entry not in graph_names:
            continue

        # Find baseline .sg
        sg_path = os.path.join(entry_path, f"{entry}.sg")
        if not os.path.isfile(sg_path):
            continue

        baseline_size = os.path.getsize(sg_path)
        # Only count files that don't already exist
        for _aid, canonical, _opt in algo_variants:
            pregen = os.path.join(entry_path, f"{entry}_{canonical}.sg")
            if not os.path.isfile(pregen):
                estimated += baseline_size

    available = shutil.disk_usage(graphs_dir).free
    return estimated, available, estimated <= available


def pregenerate_reordered_sgs(
    graphs_dir: str = DEFAULT_GRAPHS_DIR,
    algorithms: List[int] = None,
    bin_dir: str = None,
    force: bool = False,
    graph_names: List[str] = None,
    timeout: int = 600,
) -> Tuple[int, int]:
    """Pre-generate reordered .sg files so benchmarks skip runtime reordering.

    For every graph that already has a baseline ``.sg`` (typically RANDOM-ordered),
    this creates ``{graph_name}_{ALGORITHM_NAME}.sg`` for each algorithm in
    *algorithms*.  At benchmark time the pre-generated file is loaded with
    ``-o 0`` (ORIGINAL) so no runtime reorder overhead is incurred.

    Before generating, the function checks whether the estimated storage fits
    on disk.  If not, it logs a warning and returns ``(0, 0)`` so the caller
    can fall back to real-time reordering.

    Args:
        graphs_dir:   Root directory containing graph sub-directories.
        algorithms:   Reorder algorithm IDs (default: REORDER_ALGORITHMS).
        bin_dir:      Directory containing the ``converter`` binary.
        force:        If True, regenerate even when the file already exists.
        graph_names:  Optional list limiting generation to these graphs.
        timeout:      Per-conversion timeout in seconds.

    Returns:
        ``(generated, skipped)`` counts.
    """
    algorithms = algorithms or REORDER_ALGORITHMS
    bin_dir = bin_dir or str(BIN_DIR)
    converter = os.path.join(bin_dir, "converter")

    if not os.path.isfile(converter):
        log(f"Converter binary not found: {converter}", "ERROR")
        return 0, 0

    algo_variants = _iter_algo_variants(algorithms)
    canonical_names = [name for _, name, _ in algo_variants]
    log_section(f"PRE-GENERATING REORDERED .sg FILES ({len(algo_variants)} targets)")
    log(f"  Targets: {', '.join(canonical_names)}")

    # ── Disk-space safety check ──────────────────────────────────────
    est_bytes, avail_bytes, fits = estimate_pregeneration_size(
        graphs_dir, algorithms, graph_names
    )
    est_mb = est_bytes / (1024 * 1024)
    avail_mb = avail_bytes / (1024 * 1024)
    log(f"  Estimated additional disk: {est_mb:.1f} MB")
    log(f"  Available disk space:      {avail_mb:.1f} MB")

    if not fits:
        log(
            f"Insufficient disk space ({est_mb:.1f} MB needed, "
            f"{avail_mb:.1f} MB free). Falling back to real-time reordering.",
            "WARN",
        )
        return 0, 0

    # ── Generate ─────────────────────────────────────────────────────
    generated = 0
    skipped = 0
    failed = 0
    entries = sorted(os.listdir(graphs_dir)) if os.path.isdir(graphs_dir) else []

    for entry in entries:
        entry_path = os.path.join(graphs_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        if graph_names and entry not in graph_names:
            continue

        baseline_sg = os.path.join(entry_path, f"{entry}.sg")
        if not os.path.isfile(baseline_sg):
            continue

        for _algo_id, canonical, converter_opt in algo_variants:
            out_path = os.path.join(entry_path, f"{entry}_{canonical}.sg")

            if os.path.isfile(out_path) and os.path.getsize(out_path) > 0 and not force:
                skipped += 1
                continue

            log(f"  {entry} → {canonical} ...")
            # Chained orderings already contain `-o` prefixes (e.g. "-o 2 -o 8:csr").
            # Single orderings are bare (e.g. "8:csr") and need a `-o` prefix.
            if converter_opt.startswith("-o"):
                o_flags = converter_opt
            else:
                o_flags = f"-o {converter_opt}"
            cmd = f"{converter} -f {baseline_sg} -s {o_flags} -b {out_path}"
            success, stdout, stderr = run_command(cmd, timeout=timeout)

            if not success:
                log(f"    FAILED: {stderr[:200]}", "ERROR")
                if os.path.isfile(out_path):
                    os.remove(out_path)
                failed += 1
                continue

            if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
                log(f"    Converter produced empty output for {entry}/{canonical}", "ERROR")
                if os.path.isfile(out_path):
                    os.remove(out_path)
                failed += 1
                continue

            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            log(f"    -> {out_path} ({size_mb:.1f} MB)")
            generated += 1

    log(
        f"Pre-generation complete: {generated} generated, "
        f"{skipped} existing, {failed} failed"
    )
    return generated, skipped


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
    
    required_bins = BENCHMARKS
    
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
            cmd = f"cd {project_dir} && make clean && make -j$(nproc)" if rebuild else f"cd {project_dir} && make -j$(nproc)"
            success_build, stdout, stderr = run_command(cmd, timeout=TIMEOUT_BUILD)
            if not success_build:
                log(f"  Build failed: {stderr[:200]}", "ERROR")
                success = False
            else:
                log("  Standard binaries built successfully")
        
        if missing_sim or rebuild:
            log(f"  Building simulation binaries: {', '.join(missing_sim) if missing_sim else 'all (rebuild requested)'}...")
            cmd = f"cd {project_dir} && make clean-sim && make all-sim -j$(nproc)" if rebuild else f"cd {project_dir} && make all-sim -j$(nproc)"
            success_build, stdout, stderr = run_command(cmd, timeout=TIMEOUT_BUILD)
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


def clean_results(results_dir: str = DEFAULT_RESULTS_DIR, keep_graphs: bool = True) -> None:
    """Clean the results directory, optionally keeping graphs.
    
    Args:
        results_dir: Directory to clean
        keep_graphs: If True, don't delete downloaded graphs
    """
    log_section("CLEANING RESULTS DIRECTORY")
    
    if not os.path.exists(results_dir):
        log(f"Results directory does not exist: {results_dir}")
        return
    
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
            log(f"  Keeping: {entry}")
            continue
        
        # Clean based on pattern
        if entry.endswith(('.json', '.log', '.csv')):
            try:
                os.remove(entry_path)
                deleted_count += 1
                log(f"  Deleted: {entry}")
            except Exception as e:
                log(f"  Error deleting {entry}: {e}", "WARN")
        elif os.path.isdir(entry_path) and entry in ["mappings", "logs"]:
            try:
                shutil.rmtree(entry_path)
                deleted_count += 1
                log(f"  Deleted: {entry}/")
            except Exception as e:
                log(f"  Error deleting {entry}/: {e}", "WARN")
    
    log(f"Cleaned {deleted_count} items, kept {kept_count} items")

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
            log("Cancelled")
            return
    
    log_section("FULL CLEAN - REMOVING ALL GENERATED DATA")
    
    # Clean results directory completely
    results_dir = os.path.join(project_dir, "results")
    if os.path.exists(results_dir):
        log(f"Removing {results_dir}/")
        shutil.rmtree(results_dir)
        os.makedirs(results_dir)  # Recreate empty
        log("  Done")
    
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
            log(f"Cleaned binaries in {bin_dir}/")
    
    # Clean label.map files in graphs directory
    graphs_dir = os.path.join(project_dir, "results", "graphs")
    if os.path.exists(graphs_dir):
        map_files = glob.glob(os.path.join(graphs_dir, "**/label.map"), recursive=True)
        if map_files:
            log(f"Removing {len(map_files)} label.map files from graphs/")
            for map_file in map_files:
                try:
                    os.remove(map_file)
                except OSError:
                    pass
            log("  Done")
    
    # Clean bench/results if exists
    bench_results = os.path.join(project_dir, "bench", "results")
    if os.path.exists(bench_results):
        log(f"Removing {bench_results}/")
        shutil.rmtree(bench_results)
        log("  Done")
    
    log("Clean complete - ready for fresh start")


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


# ============================================================================
# Main Experiment Pipeline (Full Implementation)
# ============================================================================

def run_experiment(args):
    """Run the complete experiment pipeline with comprehensive progress tracking."""
    
    global _progress
    _progress = ProgressTracker()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize session for log grouping
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
    if args.expand_variants:
        _progress.info("Leiden variant expansion: ENABLED")
    if args.random_baseline:
        _progress.info("Random baseline: ENABLED (graphs converted to .sg with RANDOM ordering)")
    if args.pregenerate_sg:
        _progress.info("Pre-generated .sg: ENABLED (reordered graphs cached on disk)")
    _progress.phase_end()
    
    # Discover graphs
    _progress.phase_start("GRAPH DISCOVERY", "Finding available graph datasets")
    
    # Determine size range
    if args.min_size > 0 or args.max_size < float('inf'):
        min_size, max_size = args.min_size, args.max_size
    else:
        min_size, max_size = _SIZE_RANGES.get(args.graphs, (args.min_size, args.max_size))
    
    _progress.info(f"Size filter: {min_size:.1f}MB - {max_size:.1f}MB")
    
    # Auto-download: ensure graphs are available (skips already-downloaded, safe to call multiple times)
    if args.auto and not args.skip_download:
        _progress.info(f"Auto-download: ensuring all {args.download_size} graphs are available...")
        download_graphs(
            size_category=args.download_size,
            graphs_dir=args.graphs_dir,
            force=False,  # Don't re-download existing
            max_memory_gb=args.max_memory,
            max_disk_gb=args.max_disk
        )
    
    # Random-baseline .sg conversion: convert .mtx → .sg with RANDOM ordering
    # so all benchmarks measure improvement over a worst-case random baseline.
    if args.random_baseline:
        graph_name_filter = args.graph_list if args.graph_list else None
        convert_graphs_to_sg(
            graphs_dir=args.graphs_dir,
            order=1,  # RANDOM
            bin_dir=args.bin_dir,
            force=args.force_convert,
            graph_names=graph_name_filter,
        )
    
    # Pre-generate reordered .sg files: create {graph}_{ALGO}.sg from the
    # RANDOM baseline so benchmarks can load the pre-ordered graph with -o 0
    # and skip runtime reorder overhead entirely.
    pregenerated_available = False
    if getattr(args, 'pregenerate_sg', True):
        graph_name_filter = args.graph_list if args.graph_list else None
        gen_count, skip_count = pregenerate_reordered_sgs(
            graphs_dir=args.graphs_dir,
            bin_dir=args.bin_dir,
            force=args.force_convert,
            graph_names=graph_name_filter,
        )
        pregenerated_available = (gen_count + skip_count) > 0
        if pregenerated_available:
            _progress.info(
                f"Pre-generated .sg files ready ({gen_count} new, "
                f"{skip_count} cached) — benchmarks will skip runtime reordering"
            )
        else:
            _progress.info("Pre-generated .sg files: unavailable — using real-time reordering")
    
    graphs = discover_graphs(args.graphs_dir, min_size, max_size, max_memory_gb=args.max_memory,
                             min_edges=args.min_edges)
    
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
    
    # Select algorithms for benchmarking (includes baselines ORIGINAL/RANDOM)
    if args.key_only:
        algorithms = KEY_ALGORITHMS
    else:
        algorithms = BENCHMARK_ALGORITHMS
    
    # Reorder-only algorithms: exclude baselines.
    # Baselines are graph states, not reordering techniques — no mapping to generate.
    reorder_algorithms = [a for a in algorithms if a not in (_ORIGINAL_ID, _RANDOM_ID)]
    
    # Filter by specific algorithm name(s) if provided
    algo_filter_names = args.algo_list if args.algo_list else ([args.algo_name] if args.algo_name else None)
    if algo_filter_names:
        # Build reverse mapping from name to id (auto-generated from SSOT)
        name_to_id = {v.upper(): k for k, v in ALGORITHMS.items()}
        # Add variant names → base algo IDs (from SSOT API)
        from scripts.lib.utils import get_all_algorithm_variant_names
        for name in get_all_algorithm_variant_names():
            algo_id_for_name = next(
                (aid for aid in ALGORITHMS
                 if aid in _VARIANT_ALGO_REGISTRY
                 and name.startswith(_VARIANT_ALGO_REGISTRY[aid][0])),
                None,
            )
            if algo_id_for_name is not None:
                name_to_id[name.upper()] = algo_id_for_name
        # Add GOrder implementation variant names (not in registry, share weight)
        for v in GORDER_VARIANTS:
            name_to_id[f"GORDER_{v}".upper()] = ALGORITHM_IDS["GORDER"]
        # Common shorthand aliases (use ALGORITHM_IDS from SSOT)
        name_to_id.update({
            "RABBIT": ALGORITHM_IDS["RABBITORDER"],
            "LEIDEN": ALGORITHM_IDS["LeidenOrder"],
            "LEIDENORDER": ALGORITHM_IDS["LeidenOrder"],
            "GRAPHBREW": ALGORITHM_IDS["GraphBrewOrder"],
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
                        filtered_names.append(canonical_algo_key(algo_id))
                        break
        
        if filtered_algos:
            algorithms = [a for a in algorithms if a in filtered_algos]
            reorder_algorithms = [a for a in reorder_algorithms if a in filtered_algos]
            if not algorithms:
                # If no overlap with current set, use filtered set directly
                algorithms = sorted(list(filtered_algos))
                reorder_algorithms = [a for a in algorithms if a not in (_ORIGINAL_ID, _RANDOM_ID)]
            _progress.info(f"Filtered to {len(algorithms)} algorithms: {', '.join(filtered_names)}")
        else:
            _progress.error(f"No matching algorithms found for: {algo_filter_names}")
            _progress.info(f"Available algorithms: {', '.join(ALGORITHMS.values())}")
            return

    _progress.phase_start("ALGORITHM SELECTION", "Determining which algorithms to test")
    _progress.info(f"Algorithm set: {'KEY (fast subset)' if args.key_only else 'FULL benchmark set'}")
    _progress.info(f"Benchmark algorithms: {len(algorithms)} (includes baselines ORIGINAL, RANDOM)")
    _progress.info(f"Reorder algorithms: {len(reorder_algorithms)} (excludes baselines)")
    algo_names = [canonical_algo_key(a) for a in algorithms[:10]]
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
    if args.generate_maps:
        _progress.phase_start("LABEL MAP GENERATION", "Pre-generating reordering mappings (.lo files)")
        
        # Check if variant expansion is requested
        if args.expand_variants:
            _progress.info("Leiden variant expansion: ENABLED")
            _progress.info(f"  Resolution: {args.leiden_resolution}")
            _progress.info(f"  Passes: {args.leiden_passes}")
            
            # Use variant-aware mapping generation
            label_maps, reorder_timing_results = generate_reorderings_with_variants(
                graphs=graphs,
                algorithms=reorder_algorithms,
                bin_dir=args.bin_dir,
                output_dir=args.results_dir,
                expand_leiden_variants=True,
                leiden_resolution=args.leiden_resolution,
                leiden_passes=args.leiden_passes,
                gorder_variants=args.gorder_variants,
                timeout=args.timeout_reorder,
                skip_slow=args.skip_slow,
                force_reorder=args.force_reorder
            )
        else:
            # Use standard mapping generation (no variant expansion)
            label_maps, reorder_timing_results = generate_label_maps(
                graphs=graphs,
                algorithms=reorder_algorithms,
                bin_dir=args.bin_dir,
                output_dir=args.results_dir,
                timeout=args.timeout_reorder,
                skip_slow=args.skip_slow
            )
        all_reorder_results.extend(reorder_timing_results)
        _progress.phase_end(f"Generated {len(label_maps)} graph mappings")
    
    # Load existing label maps if requested
    if args.use_maps:
        _progress.phase_start("LOADING EXISTING MAPS", "Loading pre-generated label mappings")
        label_maps = load_label_maps_index(args.results_dir)
        if label_maps:
            _progress.success(f"Loaded label maps for {len(label_maps)} graphs")
        else:
            _progress.warning("No existing label maps found")
        _progress.phase_end()
    
    # Phase 1: Reordering
    # (skip if fill_weights is active — it has its own reorder phase)
    if args.phase in ["all", "reorder"] and not args.fill_weights:
        _progress.phase_start("REORDERING", "Generating vertex reorderings for all graphs")
        
        # Check if variant expansion is requested
        if args.expand_variants:
            _progress.info("Leiden/RabbitOrder variant expansion: ENABLED")
            
            # Use variant-aware reordering
            variant_label_maps, reorder_results = generate_reorderings_with_variants(
                graphs=graphs,
                algorithms=reorder_algorithms,
                bin_dir=args.bin_dir,
                output_dir=args.results_dir,
                expand_leiden_variants=True,
                leiden_resolution=args.leiden_resolution,
                leiden_passes=args.leiden_passes,
                rabbit_variants=args.rabbit_variants,
                graphbrew_variants=args.graphbrew_variants,
                gorder_variants=args.gorder_variants,
                timeout=args.timeout_reorder,
                skip_slow=args.skip_slow,
                force_reorder=args.force_reorder
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
                algorithms=reorder_algorithms,
                bin_dir=args.bin_dir,
                output_dir=args.results_dir,
                timeout=args.timeout_reorder,
                skip_slow=args.skip_slow,
                generate_maps=True,  # Always generate .lo mapping files
                force_reorder=args.force_reorder
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
    
    # Pre-compute variant map detection once (used by Phase 2 & 3)
    has_variant_maps = (label_maps and
                       any(is_variant_prefixed(algo_name)
                           for g in label_maps.values() for algo_name in g.keys()))
    
    # Phase 2: Benchmarks
    # (skip if fill_weights is active — it has its own benchmark phase)
    if args.phase in ["all", "benchmark"] and not args.fill_weights:
        _progress.phase_start("BENCHMARKING", "Running performance benchmarks")
        
        if has_variant_maps and args.expand_variants:
            # Use variant-aware benchmarking with pre-generated variant mappings
            _progress.info("Mode: Variant-aware benchmarking (GraphBrewOrder_leiden, GraphBrewOrder_rabbit, etc.)")
            benchmark_results = run_benchmarks_with_variants(
                graphs=graphs,
                label_maps=label_maps,
                benchmarks=args.benchmarks,
                bin_dir=args.bin_dir,
                num_trials=args.trials,
                timeout=args.timeout_benchmark,
                weights_dir=args.weights_dir,
                update_weights=not args.no_incremental,
                progress=_progress
            )
        else:
            # Standard benchmarking
            _progress.info("Mode: Standard benchmarking")
            if pregenerated_available:
                _progress.info("  Using pre-generated .sg files (no runtime reorder overhead)")
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
                update_weights=not args.no_incremental,
                use_pregenerated=pregenerated_available,
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
        
        # Amortization report (auto-generated after benchmarking)
        _print_amortization_report(benchmark_results, all_reorder_results)
    
    # Phase 3: Cache Simulations
    # (skip if fill_weights is active — it has its own cache phase)
    if args.phase in ["all", "cache"] and not args.skip_cache and not args.fill_weights:
        _progress.phase_start("CACHE SIMULATION", "Running cache miss simulations")
        
        if has_variant_maps and args.expand_variants:
            # Use variant-aware cache simulation
            _progress.info("Mode: Variant-aware cache simulation (GraphBrewOrder_leiden, GraphBrewOrder_rabbit, etc.)")
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
            rabbit_variants = args.rabbit_variants
            
            if rabbit_variants:
                _progress.info(f"  RabbitOrder variants: {rabbit_variants or ['csr (default)']}")
            
            cache_results = run_cache_simulations(
                graphs=graphs,
                algorithms=algorithms,
                benchmarks=args.benchmarks,
                bin_sim_dir=args.bin_sim_dir,
                timeout=args.timeout_sim,
                skip_heavy=args.skip_heavy,
                label_maps=label_maps,
                rabbit_variants=rabbit_variants,
                resolution=args.leiden_resolution,
                passes=args.leiden_passes
            )
        all_cache_results.extend(cache_results)
        
        # Save intermediate results
        cache_file = os.path.join(args.results_dir, f"cache_{timestamp}.json")
        with open(cache_file, 'w') as f:
            json.dump([asdict(r) for r in cache_results], f, indent=2)
        _progress.success(f"Cache results saved to: {cache_file}")
        _progress.phase_end(f"Completed {len(cache_results)} cache simulations")
    
    # Phase 6: Adaptive Order Analysis
    if args.phase in ["all", "adaptive"] or args.adaptive_analysis:
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
    if args.adaptive_comparison:
        # Compare against top fixed algorithms
        fixed_algos = [
            ALGORITHM_IDS["RANDOM"], ALGORITHM_IDS["SORT"],
            ALGORITHM_IDS["HUBCLUSTER"], ALGORITHM_IDS["HUBCLUSTERDBG"],
            ALGORITHM_IDS["LeidenOrder"],
        ]
        
        compare_adaptive_vs_fixed(
            graphs=graphs,
            bin_dir=args.bin_dir,
            benchmarks=args.benchmarks,
            fixed_algorithms=fixed_algos,
            output_dir=args.results_dir,
            num_trials=args.trials,
            timeout=args.timeout_benchmark
        )
    
    # Phase 8: Brute-Force Validation (all eligible algos vs Adaptive)
    if args.brute_force:
        bf_benchmark = args.bf_benchmark
        run_subcommunity_brute_force(
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
    if args.validate_adaptive:
        validate_adaptive_accuracy(
            graphs=graphs,
            bin_dir=args.bin_dir,
            output_dir=args.results_dir,
            benchmarks=args.benchmarks,
            timeout=args.timeout_benchmark,
            num_trials=args.trials,
            force_reorder=args.force_reorder
        )
    
    # Phase 9: Iterative Training (feedback loop to optimize adaptive weights)
    if args.train_adaptive:
        train_adaptive_weights_iterative(
            graphs=graphs,
            bin_dir=args.bin_dir,
            bin_sim_dir=args.bin_sim_dir,
            output_dir=args.results_dir,
            weights_file=args.weights_file,
            benchmark=args.bf_benchmark,
            target_accuracy=args.target_accuracy,
            max_iterations=args.max_iterations,
            timeout=args.timeout_benchmark,
            timeout_sim=args.timeout_sim,
            num_trials=args.trials,
            learning_rate=args.learning_rate,
            algorithms=algorithms,
            weights_dir=args.weights_dir  # Type-based weights directory
        )
    
    # Phase 10: Large-Scale Training (batched multi-benchmark training)
    if args.train_large:
        train_adaptive_weights_large_scale(
            graphs=graphs,
            bin_dir=args.bin_dir,
            bin_sim_dir=args.bin_sim_dir,
            output_dir=args.results_dir,
            weights_file=args.weights_file,
            benchmarks=args.train_benchmarks,
            target_accuracy=args.target_accuracy,
            max_iterations=args.max_iterations,
            batch_size=args.batch_size,
            timeout=args.timeout_benchmark,
            timeout_sim=args.timeout_sim,
            num_trials=args.trials,
            learning_rate=args.learning_rate,
            algorithms=algorithms
        )
    
    # Initialize/upgrade weights file with enhanced features
    if args.init_weights:
        log_section("Initialize Enhanced Weights")
        weights = initialize_enhanced_weights(args.weights_file)
        log(f"Weights initialized/upgraded with {len(weights) - 1} algorithms")
        log(f"Saved to: {args.weights_file}")
    
    # Fill ALL weights mode: comprehensive training to populate all weight fields
    # Triggered by --train, --full, or --phase weights
    if args.fill_weights or args.phase == "weights":
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
        weights_file = args.weights_file or os.path.join(args.weights_dir, 'weights.json')
        cache_dir = os.path.dirname(weights_file) or args.weights_dir
        props_cache = load_graph_properties_cache(cache_dir)
        log(f"Loaded graph properties cache: {len(props_cache)} graphs")
        
        # Respect --skip-cache flag
        skip_cache_original = args.skip_cache
        
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
            skip_slow=args.skip_slow,
            generate_maps=True,  # Always generate .lo mapping files
            force_reorder=args.force_reorder
        )
        
        # Phase 2: Benchmarks (all of them)
        log_section("Phase 2: Execution Benchmarks (All)")
        all_benchmarks = BENCHMARKS
        if has_variant_maps and args.expand_variants:
            # Variant-aware benchmarking (uses pre-generated .sg files or label maps)
            log("  Mode: Variant-aware benchmarking")
            benchmark_results = run_benchmarks_with_variants(
                graphs=graphs,
                label_maps=label_maps,
                benchmarks=all_benchmarks,
                bin_dir=args.bin_dir,
                num_trials=args.trials,
                timeout=args.timeout_benchmark,
                weights_dir=args.weights_dir,
                update_weights=True,
            )
        else:
            benchmark_results = run_benchmarks_multi_graph(
                graphs=graphs,
                algorithms=algorithms,
                benchmarks=all_benchmarks,
                bin_dir=args.bin_dir,
                num_trials=args.trials,
                timeout=args.timeout_benchmark,
                skip_slow=args.skip_slow,
                label_maps=label_maps,
                weights_dir=args.weights_dir,
                update_weights=True  # Always update incrementally in fill-weights
            )
        
        # Amortization report after fill-weights benchmarks
        _print_amortization_report(benchmark_results, reorder_results)
        
        # Phase 3: Cache Simulation
        log_section("Phase 3: Cache Simulation")
        if skip_cache_original:
            log("SKIPPED (--skip-cache specified)")
            cache_results = []
        else:
            # Get variant lists for expanded cache simulation
            expand_variants = args.expand_variants
            rabbit_variants = args.rabbit_variants or RABBITORDER_VARIANTS
            
            cache_results = run_cache_simulations(
                graphs=graphs,
                algorithms=algorithms,
                benchmarks=CACHE_KEY_BENCHMARKS,
                bin_sim_dir=args.bin_sim_dir,
                timeout=args.timeout_sim,
                skip_heavy=args.skip_heavy,
                label_maps={},
                rabbit_variants=rabbit_variants if expand_variants else ['csr']
            )
        
        # Phase 4: Generate Base Weights
        log_section("Phase 4: Generate Perceptron Weights")
        weights = compute_weights_from_results(
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
            graph_path = graph_info.path
            
            # Get modularity from adaptive results
            modularity = DEFAULT_MODULARITY
            for ar in adaptive_results:
                if ar.graph == graph_name:
                    modularity = ar.modularity
                    break
            
            # Run a quick benchmark to get topology features
            binary = os.path.join(args.bin_dir, "pr")
            cmd = f"{binary} -f {graph_path} -a 0 -n 1"
            try:
                success, stdout, stderr = run_command(cmd, timeout=TIMEOUT_FEATURE_EXTRACTION)
                if not success:
                    log(f"  {graph_name}: feature extraction command failed", "WARN")
                output = stdout + stderr
                
                # Parse all topology features from C++ output
                parsed = {}
                for pattern, key, default in _FEATURE_PATTERNS:
                    m = re.search(pattern, output)
                    parsed[key] = float(m.group(1)) if m else default
                
                graph_nodes = graph_info.nodes
                graph_edges = graph_info.edges
                
                # Assign to type
                features = {
                    'modularity': modularity,
                    **parsed,
                    'nodes': graph_nodes,
                    'edges': graph_edges,
                }
                
                # Save all features to graph properties cache
                update_graph_properties(graph_name, features)
                
                type_name = assign_graph_type(features, args.weights_dir, create_if_outlier=True)
                log(f"  {graph_name} → {type_name} (mod={modularity:.3f}, dv={parsed['degree_variance']:.3f}, hc={parsed['hub_concentration']:.3f})")
                
                # Update type weights with benchmark results for this graph
                graph_benchmark_results = [r for r in benchmark_results if r.graph == graph_name]
                graph_cache_results = [r for r in cache_results if r.graph == graph_name]
                
                # Find best algorithm for each benchmark
                for bench in set(r.benchmark for r in graph_benchmark_results):
                    bench_results = [r for r in graph_benchmark_results if r.benchmark == bench and r.success]
                    if not bench_results:
                        continue
                    
                    # Find baseline and best
                    baseline = next((r for r in bench_results if r.algorithm == ALGORITHMS[_ORIGINAL_ID]), bench_results[0])
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
                type_file = get_type_weights_file(type_name, args.weights_dir)
                if os.path.exists(type_file):
                    with open(type_file) as f:
                        type_data = json.load(f)
                    algo_count = len([k for k in type_data.keys() if not k.startswith('_')])
                    log(f"  {type_name}: {algo_count} algorithms trained")
        else:
            log("No types created (weights are in main perceptron_weights.json)")
        
        # Save graph properties cache for future use
        cache_output_dir = args.weights_dir or "results"
        save_graph_properties_cache(output_dir=cache_output_dir)
        cache_file_path = os.path.join(cache_output_dir, "graph_properties_cache.json")
        log(f"Saved graph properties cache to: {cache_file_path}")
        
        # Re-run weight update now that graph properties cache is populated
        # This ensures feature weights (w_modularity, etc.) are computed
        # Only if we have a legacy weights file
        if args.weights_file:
            update_zero_weights(
                weights_file=args.weights_file,
                graphs_dir=os.path.dirname(args.weights_file) or "results",
                benchmark_results=benchmark_results,
                cache_results=cache_results,
                reorder_results=reorder_results
            )
        
        log_section("Fill Weights Complete")
        log("All weight fields have been populated")
        log(f"Weights directory: {args.weights_dir}")
        
        # Show summary from type-based weights
        summary_weights_file = None
        if args.weights_file and os.path.exists(args.weights_file):
            summary_weights_file = args.weights_file
        else:
            # Try type_0/weights.json as the main type-based weights file
            type_0_file = get_type_weights_file('type_0', args.weights_dir)
            if os.path.exists(type_0_file):
                summary_weights_file = type_0_file
        
        if summary_weights_file:
            with open(summary_weights_file) as f:
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
        if not args.no_merge:
            try:
                log("\nMerging weights with previous runs...")
                merge_summary = auto_merge_after_run()
                if "error" not in merge_summary:
                    log(f"  Merged {merge_summary.get('runs_merged', 0)} runs -> {merge_summary.get('total_types', 0)} types")
                    log(f"  Merged weights saved to: {merge_summary.get('output_dir', 'results/weights/merged/')}")
            except Exception as e:
                log(f"  Warning: Weight merge failed: {e}")
    
    # Show final summary with statistics
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
    
    parser.add_argument("--force-download", action="store_true",
                        help="Re-download graphs even if they exist")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip build check (assume binaries exist)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip graph download phase (use existing graphs only)")
    
    # Random-baseline .sg conversion
    parser.add_argument("--random-baseline", action="store_true", default=True,
                        help="Convert graphs to .sg with RANDOM ordering (default: enabled). "
                             "This ensures benchmarks measure improvement over a random baseline.")
    parser.add_argument("--no-random-baseline", action="store_false", dest="random_baseline",
                        help="Disable random-baseline conversion (use original .mtx files directly)")
    parser.add_argument("--force-convert", action="store_true",
                        help="Force re-conversion of .sg files even if they already exist")
    
    # Pre-generated reordered .sg files
    parser.add_argument("--pregenerate-sg", action="store_true", default=True,
                        help="Pre-generate reordered .sg files for each algorithm so "
                             "benchmarks skip runtime reordering (default: enabled). "
                             "Falls back to real-time reordering if disk space is insufficient.")
    parser.add_argument("--no-pregenerate-sg", action="store_false", dest="pregenerate_sg",
                        help="Disable pre-generation of reordered .sg files "
                             "(use real-time reordering during benchmarks)")
    
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
    
    # Graph selection
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
                             "RabbitOrder, Gorder, RCM, Leiden)")
    parser.add_argument("--algo", "--algorithm", type=str, default=None, dest="algo_name",
                        help="Run only a specific algorithm (e.g., RABBITORDER_boost, GraphBrewOrder_leiden)")
    parser.add_argument("--algo-list", nargs="+", type=str, default=None, dest="algo_list",
                        help="Run specific algorithms (e.g., --algo-list RABBITORDER_csr GraphBrewOrder_leiden GORDER GORDER_csr)")
    parser.add_argument("--skip-slow", action="store_true",
                        help="Skip slow algorithms (Gorder, Corder, RCM) on large graphs")
    
    # Benchmark selection
    parser.add_argument("--benchmarks", nargs="+", default=EXPERIMENT_BENCHMARKS,
                        help="Benchmarks to run (default: excludes TC)")
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
    parser.add_argument("--rabbit-variants", nargs="+",
                        default=None, choices=["csr", "boost"],
                        help="RabbitOrder variants: csr (default, no deps), boost (requires libboost-graph-dev)")
    parser.add_argument("--gorder-variants", nargs="+",
                        default=None, choices=["default", "csr", "fast"],
                        help="GOrder implementation variants: default, csr, fast (differ in speed, same ordering)")
    parser.add_argument("--graphbrew-variants", nargs="+", dest="graphbrew_variants",
                        default=None, choices=["leiden", "rabbit", "hubcluster"],
                        help="GraphBrewOrder variants (GraphBrew-powered): leiden (default), rabbit, hubcluster")
    parser.add_argument("--resolution", type=str, default="dynamic", dest="leiden_resolution",
                        help="Leiden resolution: dynamic (default, best PR), auto, fixed (1.5), dynamic_2.0")
    parser.add_argument("--passes", type=int, default=LEIDEN_DEFAULT_PASSES, dest="leiden_passes",
                        help=f"Leiden refinement passes - higher = better quality (default: {LEIDEN_DEFAULT_PASSES})")
    
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
    parser.add_argument("--train-benchmarks", nargs="+", default=DEFAULT_TRAIN_BENCHMARKS,
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
    
    parser.add_argument("--weights-file", default=os.path.join(DEFAULT_WEIGHTS_DIR, "weights.json"),
                        help="Path to weights file (default: results/weights/weights.json)")
    
    # Clean options
    parser.add_argument("--clean", action="store_true",
                        help="Clean results directory before running (keeps graphs and weights)")
    parser.add_argument("--clean-all", action="store_true",
                        help="Remove ALL generated data (graphs, results, mappings) - fresh start")
    
    # Auto-setup option
    parser.add_argument("--auto-setup", action="store_true",
                        help="Automatically setup everything: create directories, build if missing, download graphs if needed")

    # ── Standalone sub-workflows (replace shell scripts) ─────────────
    parser.add_argument("--benchmark-fresh", action="store_true",
                        help="Run all AdaptiveOrder-eligible algorithms on all .sg graphs (replaces benchmark_fresh.sh)")
    parser.add_argument("--ab-test", action="store_true",
                        help="A/B test: AdaptiveOrder vs Original on all .sg graphs (replaces ab_test.sh)")
    parser.add_argument("--eval-weights", action="store_true",
                        help="Train perceptron weights and evaluate prediction accuracy (replaces eval_weights.py)")
    parser.add_argument("--logo", action="store_true",
                        help="[eval-weights] Run Leave-One-Graph-Out cross-validation to measure generalization")
    parser.add_argument("--sg-only", action="store_true",
                        help="[eval-weights] Only use .sg benchmark data for training")
    parser.add_argument("--benchmark-file", default=None,
                        help="[eval-weights] Load a specific benchmark JSON file")
    parser.add_argument("--timeout", type=int, default=TIMEOUT_BENCHMARK,
                        help=f"[benchmark-fresh/ab-test] Timeout per benchmark invocation in seconds (default: {TIMEOUT_BENCHMARK})")

    # ── Tools (moved from standalone scripts) ────────────────────────
    parser.add_argument("--emulator", action="store_true",
                        help="Run AdaptiveOrder emulator for weight analysis (pass extra args after --)")
    parser.add_argument("--oracle-analysis", action="store_true",
                        help="Oracle analysis: selection accuracy, regret, confusion matrix")
    parser.add_argument("--perceptron", action="store_true",
                        help="Perceptron experimentation: grid search, training, interactive mode")
    parser.add_argument("--cache-compare", action="store_true",
                        help="Quick cache simulation comparison across algorithm variants")
    parser.add_argument("--regen-features", action="store_true",
                        help="Regenerate features.json for all .sg graphs via C++ binary")
    parser.add_argument("--check-includes", action="store_true",
                        help="Scan C++ sources for legacy include paths")
    parser.add_argument("--generate-figures", action="store_true",
                        help="Generate reordering visualization SVG figures for docs")
    parser.add_argument("--compare-leiden", action="store_true",
                        help="Compare Leiden/RabbitOrder/GraphBrew community detection variants")

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
    if args.rabbit_variants:
        if not args.expand_variants:
            args.expand_variants = True
            log("Auto-enabling variant expansion (specific variants requested)", "INFO")
    
    # Auto-enable --all-variants when running --full pipeline
    # Training on all variants is critical for the perceptron to learn which
    # variant works best for each graph structure (e.g., GraphBrewOrder_leiden vs
    # GraphBrewOrder_rabbit, RABBITORDER_csr vs RABBITORDER_boost, etc.)
    if args.full and not args.expand_variants:
        args.expand_variants = True
        log("Auto-enabling variant expansion for --full pipeline (train on all variants)", "INFO")
    
    # Resolve unified --size parameter
    if args.size is not None:
        size_lower = args.size.lower()
        args.graphs = size_lower if size_lower != "xlarge" else "all"
        args.download_size = args.size.upper()
        log(f"Using --size {args.size}: graphs={args.graphs}, download={args.download_size}", "INFO")
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
            all_ok, status = check_dependencies(verbose=True)
            if not all_ok:
                print("\nTo install missing dependencies:")
                print("  python scripts/graphbrew_experiment.py --install-deps")
                print("\nOr see manual instructions:")
                print("  python -m scripts.lib.dependencies --instructions")
            sys.exit(0 if all_ok else 1)
        
        if args.install_deps:
            print("Installing missing system dependencies...")
            print("(This may require sudo password)")
            success, msg = install_dependencies(verbose=True)
            print(msg)
            if success:
                print("\nVerifying installation...")
                check_dependencies(verbose=True)
            sys.exit(0 if success else 1)
    
    # Determine memory limit
    if args.auto_memory and args.max_memory is None:
        # Auto-detect available memory, use 60% of total as safe limit
        total_mem = get_total_memory_gb()
        args.max_memory = total_mem * AUTO_MEMORY_FRACTION
        log(f"Auto-detected memory limit: {args.max_memory:.1f} GB ({int(AUTO_MEMORY_FRACTION*100)}% of {total_mem:.1f} GB total)", "INFO")
    elif args.max_memory is not None:
        log(f"Using specified memory limit: {args.max_memory:.1f} GB", "INFO")
    
    # Determine disk space limit
    if args.auto_disk and args.max_disk is None:
        # Auto-detect available disk space, use 80% of free space
        free_disk = get_available_disk_gb(args.graphs_dir if os.path.exists(args.graphs_dir) else ".")
        args.max_disk = free_disk * AUTO_DISK_FRACTION
        log(f"Auto-detected disk limit: {args.max_disk:.1f} GB ({int(AUTO_DISK_FRACTION*100)}% of {free_disk:.1f} GB free)", "INFO")
    elif args.max_disk is not None:
        log(f"Using specified disk limit: {args.max_disk:.1f} GB", "INFO")
    
    # Log min_edges filter if specified (works with any download_size)
    if args.min_edges > 0:
        log(f"Min edges filter: {args.min_edges:,} (graphs with fewer edges will be skipped for training)", "INFO")
    
    # Handle clean operations first
    if args.clean_all:
        clean_all(".", confirm=False)
        if not (args.full or args.download_only):
            return  # Just clean, don't run experiments
    elif args.clean:
        clean_results(args.results_dir, keep_graphs=True)
        if not (args.full or args.download_only or args.phase != "all"):
            return  # Just clean, don't run experiments
    
    # Handle --clean-reorder-cache early
    if args.clean_reorder_cache:
        clean_reorder_cache(args.graphs_dir)
        if not (args.full or args.download_only or args.phase != "all" or 
                args.validate_adaptive or args.brute_force):
            return  # Just clean cache, don't run experiments
    
    # Handle weight management commands early
    if args.list_runs:
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
    
    if args.merge_runs is not None:
        from scripts.lib.weight_merger import merge_runs, use_merged
        run_timestamps = args.merge_runs if args.merge_runs else None
        summary = merge_runs(run_timestamps=run_timestamps)
        if "error" not in summary:
            use_merged()
            log(f"Merged {summary.get('runs_merged', 0)} runs -> {summary.get('total_types', 0)} types")
        return
    
    if args.use_run:
        from scripts.lib.weight_merger import use_run
        if use_run(args.use_run):
            log(f"Now using weights from run: {args.use_run}")
        return
    
    if args.use_merged:
        from scripts.lib.weight_merger import use_merged
        if use_merged():
            log("Now using merged weights")
        return
    
    # Handle --show-types early (informational command)
    if args.show_types:
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
                    

        return  # Exit after showing types
    
    # ALWAYS ensure prerequisites at start (unless skip_build is set)
    if not args.skip_build:
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
        download_graphs(
            size_category=args.download_size,
            graphs_dir=args.graphs_dir,
            force=False,  # Don't re-download existing
            max_memory_gb=args.max_memory,
            max_disk_gb=args.max_disk
        )
        
        # Convert to random-baseline .sg if enabled
        if args.random_baseline:
            convert_graphs_to_sg(
                graphs_dir=args.graphs_dir,
                order=1,  # RANDOM
                bin_dir=args.bin_dir,
                force=args.force_convert,
            )
        
        # Now discover all available graphs
        graphs = discover_graphs(args.graphs_dir, max_memory_gb=args.max_memory,
                                 min_edges=args.min_edges)
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
    
    # ── Standalone sub-workflows (early exit) ────────────────────────
    if args.benchmark_fresh:
        from scripts.lib.benchmark_runner import run_fresh_benchmarks
        graph_list = args.graph_list or None
        run_fresh_benchmarks(
            graphs_dir=args.graphs_dir,
            graph_names=graph_list,
            trials=args.trials,
            timeout=args.timeout,
        )
        return

    if args.ab_test:
        from scripts.lib.ab_test import run_ab_test
        graph_list = args.graph_list or None
        run_ab_test(
            graphs_dir=args.graphs_dir,
            graph_names=graph_list,
            trials=args.trials,
            timeout=args.timeout,
        )
        return

    if args.eval_weights:
        from scripts.lib.eval_weights import train_and_evaluate
        train_and_evaluate(
            results_dir=args.results_dir,
            sg_only=args.sg_only,
            benchmark_file=args.benchmark_file,
            logo=args.logo,
        )
        return

    # ── Dispatchers for moved standalone tools ───────────────────────
    if args.emulator:
        from scripts.lib.adaptive_emulator import main as emulator_main
        sys.argv = [sys.argv[0]]  # clear args so emulator's argparse takes over
        emulator_main()
        return

    if args.oracle_analysis:
        from scripts.lib.oracle import main as oracle_main
        sys.argv = [sys.argv[0], "--results-dir", args.results_dir]
        oracle_main()
        return

    if args.perceptron:
        from scripts.lib.perceptron import main as perceptron_main
        sys.argv = [sys.argv[0]]
        perceptron_main()
        return

    if args.cache_compare:
        from scripts.lib.cache_compare import main as cache_compare_main
        cache_compare_main()
        return

    if args.regen_features:
        from scripts.lib.regen_features import main as regen_main
        regen_main()
        return

    if args.check_includes:
        from scripts.lib.check_includes import main as check_main
        sys.argv = [sys.argv[0]]
        check_main()
        return

    if args.generate_figures:
        from scripts.lib.figures import main as figures_main
        figures_main()
        return

    if args.compare_leiden:
        from scripts.lib.leiden_compare import main as leiden_main
        sys.argv = [sys.argv[0]]
        leiden_main()
        return

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
            # Convert to random-baseline .sg if enabled
            if args.random_baseline:
                convert_graphs_to_sg(
                    graphs_dir=args.graphs_dir,
                    order=1,  # RANDOM
                    bin_dir=args.bin_dir,
                    force=args.force_convert,
                )
            log("Download complete. Run without --download-only to start experiments.", "INFO")
            return
        
        # Handle full pipeline mode
        if args.full:
            log("="*60, "INFO")
            log("GRAPHBREW ONE-CLICK EXPERIMENT PIPELINE", "INFO")
            log("="*60, "INFO")
            log(f"Configuration: size={args.download_size}, graphs={args.graphs}", "INFO")
            
            # Downloads already handled by auto-setup block above
            
            # Enable label map generation for consistent reordering
            args.generate_maps = True
            args.use_maps = True
            
            # --full includes weight training (fill_weights handles its own
            # reorder/benchmark/cache phases internally)
            args.fill_weights = True
            
            # Run experiment
            log("\nStarting experiments...", "INFO")
        
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
