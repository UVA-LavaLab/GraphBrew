#!/usr/bin/env python3
"""
Shared utilities for GraphBrew scripts.

This module provides common functions used across all GraphBrew library modules.
Can be used standalone or imported by other modules.

Standalone usage:
    python -m scripts.lib.utils --check-deps
    python -m scripts.lib.utils --list-algorithms
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, asdict

# =============================================================================
# Path Constants
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
BENCH_DIR = PROJECT_ROOT / "bench"
BIN_DIR = BENCH_DIR / "bin"
BIN_SIM_DIR = BENCH_DIR / "bin_sim"
RESULTS_DIR = PROJECT_ROOT / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"
WEIGHTS_DIR = RESULTS_DIR / "weights"
ACTIVE_WEIGHTS_DIR = WEIGHTS_DIR  # Where C++ reads from


def weights_registry_path(weights_dir: str = "") -> str:
    """Path to the graph-type cluster registry file."""
    d = weights_dir or str(ACTIVE_WEIGHTS_DIR)
    return os.path.join(d, "registry.json")


def weights_type_path(type_name: str, weights_dir: str = "") -> str:
    """Path to generic weights for a graph-type cluster (e.g. type_0/weights.json)."""
    d = weights_dir or str(ACTIVE_WEIGHTS_DIR)
    return os.path.join(d, type_name, "weights.json")


def weights_bench_path(type_name: str, bench_name: str, weights_dir: str = "") -> str:
    """Path to per-benchmark specialised weights (e.g. type_0/pr.json)."""
    d = weights_dir or str(ACTIVE_WEIGHTS_DIR)
    return os.path.join(d, type_name, f"{bench_name}.json")

# =============================================================================
# Unified Reorder Configuration Constants
# Match C++ reorder::ReorderConfig in reorder_types.h
# Used by: GraphBrew, Leiden, GraphBrew, RabbitOrder, Adaptive
# =============================================================================

# Core defaults (single source of truth)
REORDER_DEFAULT_RESOLUTION = 1.0         # Modularity resolution (auto-computed from graph)
REORDER_DEFAULT_TOLERANCE = 1e-2         # Node movement convergence (0.01)
REORDER_DEFAULT_AGGREGATION_TOLERANCE = 0.8
REORDER_DEFAULT_TOLERANCE_DROP = 10.0    # Tolerance reduction per pass
REORDER_DEFAULT_MAX_ITERATIONS = 10      # Max iterations per pass
REORDER_DEFAULT_MAX_PASSES = 10          # Max aggregation passes

# Aliases for backward compatibility
LEIDEN_DEFAULT_RESOLUTION = REORDER_DEFAULT_RESOLUTION
LEIDEN_DEFAULT_TOLERANCE = REORDER_DEFAULT_TOLERANCE
LEIDEN_DEFAULT_AGGREGATION_TOLERANCE = REORDER_DEFAULT_AGGREGATION_TOLERANCE
LEIDEN_DEFAULT_QUALITY_FACTOR = REORDER_DEFAULT_TOLERANCE_DROP
LEIDEN_DEFAULT_MAX_ITERATIONS = REORDER_DEFAULT_MAX_ITERATIONS
LEIDEN_DEFAULT_MAX_PASSES = REORDER_DEFAULT_MAX_PASSES
LEIDEN_MODULARITY_MAX_ITERATIONS = 20    # For quality-focused runs
LEIDEN_MODULARITY_MAX_PASSES = 20

# Normalization factors for weight computation
WEIGHT_PATH_LENGTH_NORMALIZATION = 10.0
WEIGHT_REORDER_TIME_NORMALIZATION = 10.0
WEIGHT_AVG_DEGREE_DEFAULT = 10.0

# =============================================================================
# Algorithm Definitions (must match builder.h ReorderingAlgo enum)
# =============================================================================

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
    13: "MAP",  # Load from file
    14: "AdaptiveOrder",
    15: "LeidenOrder",
}

# Reverse mapping
ALGORITHM_IDS = {v: k for k, v in ALGORITHMS.items()}

# Algorithm categories
SLOW_ALGORITHMS = {9, 10, 11}  # Gorder, Corder, RCM - can be slow on large graphs

# =============================================================================
# VARIANT REGISTRY — Single Source of Truth (SSOT)
# =============================================================================
# To add a new trainable variant:
#   1. Add the suffix here (e.g. "leiden_dfs" in GRAPHBREW_VARIANTS)
#   2. In C++ reorder.h, add dispatch logic ONLY if new handling is needed
#   3. Rebuild C++ (make -j$(nproc)), run experiments, train weights
#
# The canonical variant name is auto-generated: {PREFIX}{suffix}
#   e.g. "leiden" → "GraphBrewOrder_leiden"
#
# C++ side auto-discovers variant names via prefix matching in
# ResolveVariantSelection() and ParseWeightsFromJSON() — NO C++ source
# changes needed for new variants.
# =============================================================================

# Variant prefixes — shared between Python and C++ (matches VARIANT_PREFIXES
# in reorder_types.h).  Used for auto-discovery and name resolution.
VARIANT_PREFIXES = ("GraphBrewOrder_", "RABBITORDER_", "RCM_")

# RabbitOrder variants: -o 8:variant
RABBITORDER_VARIANTS = ("csr", "boost")
RABBITORDER_DEFAULT_VARIANT = "csr"

# GOrder implementation variants: -o 9:variant
# These differ only in implementation speed — they produce equivalent orderings.
# NOT in _VARIANT_ALGO_REGISTRY because they share a single perceptron weight.
GORDER_VARIANTS = ("default", "csr", "fast")
GORDER_DEFAULT_VARIANT = "default"

# RCM variants: -o 11:variant
RCM_VARIANTS = ("default", "bnf")
RCM_DEFAULT_VARIANT = "default"

# GraphBrewOrder variants: -o 12:variant
# Compound variants use '_' separator: "leiden_dfs" → options=["leiden","dfs"]
GRAPHBREW_VARIANTS = ("leiden", "rabbit", "hubcluster")
GRAPHBREW_DEFAULT_VARIANT = "leiden"

# =============================================================================
# GraphBrew Multi-Layer Configuration (mirrors C++ parseGraphBrewConfig)
# =============================================================================
# GraphBrewOrder (algo 12) is a multi-layer pipeline.  Each "layer" is an
# independent dimension that can be combined freely via the CLI:
#
#   -o 12:preset:ordering:aggregation:features...
#
# Example:  -o 12:leiden:hrab:gvecsr:merge:hubx:0.75
#           preset=leiden, ordering=hrab, aggregation=gvecsr,
#           features=[merge,hubx], resolution=0.75
#
# The layers are:
#
#  Layer 0  PRESET (algorithm skeleton)
#  ────────────────────────────────────
#  Selects the top-level algorithm and its default config.
#  Only ONE preset is active.
#
#  Layer 1  ORDERING STRATEGY
#  ────────────────────────────────────
#  How vertices are permuted within/across communities after detection.
#  Only ONE ordering is active at a time.
#
#  Layer 2  AGGREGATION STRATEGY
#  ────────────────────────────────────
#  How the community super-graph is built between Leiden passes.
#  Only ONE aggregation is active.
#
#  Layer 3  FEATURE FLAGS (additive)
#  ────────────────────────────────────
#  Post-processing / optimization toggles.  Multiple can be combined.
#
#  Layer 4  GRAPHBREW MODE (recursive dispatch)
#  ────────────────────────────────────
#  Per-community external algorithm dispatch: "graphbrew" or "gb" token.
#  Enables finalAlgoId (0-11), recursiveDepth, subAlgoId.
#  Default depth=-1 (auto): recurses when max community > LLC capacity.
#
#  Layer 5  RESOLUTION / NUMERIC
#  ────────────────────────────────────
#  Resolution (float 0.1-3.0), max iterations, max passes.
#
# Combinatorial space per config:
#   3 presets × 13 orderings × 4 aggregations × 2^9 feature combos
#     × 12 finalAlgos × 11 depths × 12 subAlgos × continuous resolution
#   = effectively infinite; but the *trainable* space is the set of
#     compound variant names registered in GRAPHBREW_VARIANTS above.
# =============================================================================

GRAPHBREW_LAYERS = {
    # Layer 0: Preset (one-of) — sets algorithm skeleton + defaults
    "preset": {
        "leiden":      "GVE-CSR Leiden + per-community RabbitOrder (default)",
        "rabbit":      "Full RabbitOrder pipeline (single-pass, no Leiden)",
        "hubcluster":  "Leiden + hub-cluster native ordering",
    },
    # Layer 1: Ordering strategy (one-of) — vertex permutation method
    "ordering": (
        "hrab",          # Hybrid Leiden+Rabbit BFS (best general locality)
        "dfs",           # DFS traversal of community dendrogram
        "bfs",           # BFS traversal of community dendrogram
        "conn",          # Connectivity BFS within communities (Boost-style)
        "dbg",           # Degree-Based Grouping within communities
        "corder",        # Hot/cold partitioning within communities
        "dbg-global",    # DBG across all vertices (post-clustering)
        "corder-global", # Corder across all vertices (post-clustering)
        "community",     # Simple sort by final community + degree
        "hubcluster",    # Hub-first within communities
        "hierarchical",  # Multi-level sort by all passes (leiden.hxx style)
        "hcache",        # Cache-aware hierarchical ordering
        "tqr",           # Tile-quantized graph + RabbitOrder
    ),
    # Layer 2: Aggregation strategy (one-of) — super-graph construction
    "aggregation": (
        "gvecsr",        # GVE-style explicit super-graph (best quality)
        "leiden",        # Standard Leiden CSR aggregation
        "streaming",     # RabbitOrder-style lazy incremental merge (fast)
        "hybrid",        # Lazy for early passes, CSR for final
    ),
    # Layer 3: Feature flags (additive, any combination)
    "features": (
        "merge",         # Merge small communities for cache locality
        "hubx",          # Extract high-degree hubs before ordering
        "gord",          # Gorder-inspired intra-community optimization
        "hsort",         # Post-process: pack hubs by descending degree
        "rcm",           # RCM on super-graph instead of dendrogram DFS
        "norefine",      # Disable Leiden refinement step
        "lazyupdate",    # Batch community weight updates
        "verify",        # Verify topology after reordering
        "graphbrew",     # Activate LAYER ordering (per-community dispatch)
        "recursive",     # Force recursive sub-community dispatch (depth>=1)
        "flat",          # Force flat dispatch (depth=0, no recursion)
    ),
    # Layer 4: GraphBrew recursive dispatch (when "graphbrew" feature active)
    "graphbrew_dispatch": {
        "final_algo":   "0-11 (algorithm ID for per-community reordering, default=8/RabbitOrder)",
        "depth":        "-1=auto (default: recurse when max community > LLC), 0=flat, 1-10=force recursive",
        "sub_algo":     "-1 or 0-11 (-1=adaptive per-sub-community, else fixed)",
    },
    # Layer 5: Numeric parameters
    "numeric": {
        "resolution": "auto | dynamic | <float 0.1-3.0>",
        "max_iterations": "<int 1-100>",
        "max_passes": "<int 1-50>",
    },
}

# Backward-compatible alias
GRAPHBREW_OPTIONS = {
    "presets":              GRAPHBREW_LAYERS["preset"],
    "ordering_strategies":  list(GRAPHBREW_LAYERS["ordering"]),
    "aggregation":          list(GRAPHBREW_LAYERS["aggregation"]),
    "features":             list(GRAPHBREW_LAYERS["features"]),
    "resolution":           ["auto", "dynamic", "<float 0.1-3.0>"],
}

# ---- Derived helpers (computed from the registry above) ----

# Map: algo_id → (prefix, variants, default)
# NOTE: Only algorithms whose variants produce *different reorderings* (and thus
# need separate perceptron weights) belong here.  GOrder variants (default/csr/
# fast) differ only in implementation speed — they produce equivalent orderings
# so they share the single "GORDER" weight entry.  LeidenOrder resolution is a
# continuous parameter, not a discrete variant for training.
_VARIANT_ALGO_REGISTRY = {
    8:  ("RABBITORDER_",     RABBITORDER_VARIANTS,  RABBITORDER_DEFAULT_VARIANT),
    11: ("RCM_",             RCM_VARIANTS,          RCM_DEFAULT_VARIANT),
    12: ("GraphBrewOrder_",  GRAPHBREW_VARIANTS,    GRAPHBREW_DEFAULT_VARIANT),
}

# Set of algo IDs that expand to variants
VARIANT_ALGO_IDS = frozenset(_VARIANT_ALGO_REGISTRY.keys())

# C++ display-name → canonical training name (case-insensitive lookup)
# Variant-prefixed names auto-pass unchanged — only base aliases listed here.
DISPLAY_TO_CANONICAL: dict[str, str] = {
    # Base names already canonical
    "ORIGINAL": "ORIGINAL", "SORT": "SORT", "HUBSORT": "HUBSORT",
    "HUBCLUSTER": "HUBCLUSTER", "DBG": "DBG", "HUBSORTDBG": "HUBSORTDBG",
    "HUBCLUSTERDBG": "HUBCLUSTERDBG", "GORDER": "GORDER", "CORDER": "CORDER",
    "RCM": "RCM", "RANDOM": "RANDOM",
    # Mixed-case C++ display → uppercase canonical
    "Random": "RANDOM", "Sort": "SORT", "HubSort": "HUBSORT",
    "HubCluster": "HUBCLUSTER", "HubSortDBG": "HUBSORTDBG",
    "HubClusterDBG": "HUBCLUSTERDBG", "COrder": "CORDER", "RCMOrder": "RCM",
    "GOrder": "GORDER", "Original": "ORIGINAL",
    # Bare C++ display → default variant (backward compat)
    "RabbitOrder": "RABBITORDER_csr",
    "GraphBrewOrder": "GraphBrewOrder_leiden",
    # LeidenOrder is its own algorithm (C++ enum 15)
    "LeidenOrder": "LeidenOrder",
}


def get_all_algorithm_variant_names() -> list[str]:
    """Get all canonical algorithm names including variant-expanded names.

    This is the SSOT list of names that appear in weight files and
    perceptron scoring.  Derived from ALGORITHMS + the variant registry.

    Returns:
        Sorted list of canonical names like:
        ["CORDER", "DBG", ..., "GraphBrewOrder_leiden", "GraphBrewOrder_rabbit",
         "RABBITORDER_csr", "RABBITORDER_boost", "RCM_default", "RCM_bnf", ...]
    """
    names: list[str] = []
    for algo_id, algo_name in ALGORITHMS.items():
        if algo_name in ("MAP", "AdaptiveOrder"):
            continue
        if algo_id in _VARIANT_ALGO_REGISTRY:
            prefix, variants, _ = _VARIANT_ALGO_REGISTRY[algo_id]
            for v in variants:
                names.append(f"{prefix}{v}")
        else:
            names.append(algo_name)
    return sorted(names)


def resolve_canonical_name(display_name: str) -> str:
    """Resolve a C++ display name to a canonical training name.

    Variant-prefixed names (GraphBrewOrder_*, RABBITORDER_*, RCM_*) pass
    through unchanged.  Base names are looked up in DISPLAY_TO_CANONICAL.
    Unknown names pass through unchanged.
    """
    for prefix in VARIANT_PREFIXES:
        if display_name.startswith(prefix):
            return display_name
    return DISPLAY_TO_CANONICAL.get(display_name, display_name)


def is_variant_prefixed(name: str) -> bool:
    """Check if a name uses a variant prefix (auto-pass in resolution)."""
    return any(name.startswith(p) for p in VARIANT_PREFIXES)


def enumerate_graphbrew_multilayer() -> dict[str, Any]:
    """Enumerate the full multi-layer configuration space for GraphBrewOrder.

    Returns a dict with:
        layers:         per-layer option counts
        compound_variants: all preset × ordering compound names
        feature_combos: number of additive feature flag combinations
        total_configs:  approximate total unique configurations
        active_trained: currently trained variant names (from SSOT)
        untrained:      compound names not yet in training
    """
    presets = tuple(GRAPHBREW_LAYERS["preset"].keys())
    orderings = GRAPHBREW_LAYERS["ordering"]
    aggregations = GRAPHBREW_LAYERS["aggregation"]
    features = GRAPHBREW_LAYERS["features"]

    # All preset × ordering compound names
    compounds: list[str] = []
    for preset in presets:
        compounds.append(f"GraphBrewOrder_{preset}")  # base preset
        for ordering in orderings:
            compounds.append(f"GraphBrewOrder_{preset}_{ordering}")

    active = set(get_all_algorithm_variant_names())
    active_gb = sorted(c for c in compounds if c in active)
    untrained = sorted(c for c in compounds if c not in active)

    feature_combos = 2 ** len(features)  # each feature on/off
    # GraphBrew dispatch: 12 finalAlgos × 11 depths × 13 subAlgos
    dispatch_combos = 12 * 11 * 13

    total_configs = (
        len(presets)
        * len(orderings)
        * len(aggregations)
        * feature_combos
        * dispatch_combos
        # × continuous resolution (uncountable)
    )

    return {
        "layers": {
            "presets": len(presets),
            "orderings": len(orderings),
            "aggregations": len(aggregations),
            "features": len(features),
            "feature_combos": feature_combos,
            "dispatch_combos": dispatch_combos,
        },
        "compound_variants": compounds,
        "active_trained": active_gb,
        "untrained": untrained,
        "total_discrete_configs": total_configs,
    }

# Leiden default resolution mode for experiments
# NOTE: LEIDEN_DEFAULT_RESOLUTION constant (= 1.0) defined above is for algorithm defaults
# "dynamic" gives best PR performance on web graphs (adjusts per-pass)
# "auto" computes once from graph properties (density, degree CV)
LEIDEN_DEFAULT_PASSES = 3

# Benchmark definitions
BENCHMARKS = ["pr", "pr_spmv", "bfs", "cc", "cc_sv", "sssp", "bc", "tc"]

# =============================================================================
# Graph Size Thresholds (MB) - Single Source of Truth
# =============================================================================
# Used for skip logic (slow algorithms, heavy simulations) and categorization
SIZE_SMALL = 50       # < 50 MB: quick experiments
SIZE_MEDIUM = 500     # < 500 MB: moderate size, may skip slow algorithms
SIZE_LARGE = 2000     # < 2 GB: large graphs, skip heavy operations

# =============================================================================
# Timeout Constants (seconds) - Single Source of Truth
# =============================================================================
TIMEOUT_REORDER = 43200       # 12 hours for reordering (GORDER can be slow)
TIMEOUT_BENCHMARK = 600       # 10 min for benchmarks
TIMEOUT_SIM = 1200            # 20 min for cache simulations
TIMEOUT_SIM_HEAVY = 3600      # 1 hour for heavy simulations (bc, sssp)

# =============================================================================
# Graph Dimension Parsing
# =============================================================================

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
    except Exception:
        pass
    return 0, 0


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    graph: str
    algorithm: str
    algorithm_id: int
    benchmark: str
    time_seconds: float
    reorder_time: float = 0.0
    trials: int = 1
    nodes: int = 0
    edges: int = 0
    success: bool = True
    error: str = ""
    extra: Dict = None
    
    def __post_init__(self):
        if self.extra is None:
            self.extra = {}
    
    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# Directory Management
# =============================================================================

def ensure_directories() -> None:
    """Create all required directories if they don't exist."""
    for directory in [RESULTS_DIR, GRAPHS_DIR, WEIGHTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def get_timestamp() -> str:
    """Return current timestamp string for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_results_file(prefix: str, extension: str = "json") -> Path:
    """Generate a timestamped results file path."""
    ensure_directories()
    return RESULTS_DIR / f"{prefix}_{get_timestamp()}.{extension}"


# =============================================================================
# Logging
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'


class Logger:
    """
    Simple logger with levels, colors, and optional file output.
    
    Provides clean, consistent output formatting across all GraphBrew modules.
    Supports color output when running in a terminal.
    """
    
    LEVELS = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
    
    # Status symbols for visual feedback
    SYMBOLS = {
        "success": "✓",
        "error": "✗",
        "warning": "⚠",
        "info": "→",
        "debug": "·",
        "progress": "↓",
        "done": "●",
        "skip": "○",
    }
    
    def __init__(self, level: str = "INFO", log_file: Optional[Path] = None, 
                 use_colors: bool = True, compact: bool = False):
        """
        Initialize logger.
        
        Args:
            level: Minimum log level (DEBUG, INFO, WARNING, ERROR)
            log_file: Optional file to write logs to
            use_colors: Enable colored output (auto-disabled if not TTY)
            compact: Use compact output format (no timestamps)
        """
        self.level = self.LEVELS.get(level.upper(), 1)
        self.log_file = log_file
        self.use_colors = use_colors and sys.stdout.isatty()
        self.compact = compact
        
    def _log(self, level: str, message: str, color: str = "", 
             symbol: str = "", prefix: str = "") -> None:
        """Internal log method."""
        if self.LEVELS.get(level, 0) < self.level:
            return
            
        # Build formatted message
        if self.compact:
            if symbol:
                formatted = f"  {symbol} {message}"
            elif prefix:
                formatted = f"  {prefix} {message}"
            else:
                formatted = f"  {message}"
        else:
            timestamp = datetime.now().strftime("%H:%M:%S")
            if symbol:
                formatted = f"[{timestamp}] {symbol} {message}"
            else:
                formatted = f"[{timestamp}] [{level}] {message}"
        
        # Print with optional color
        if color and self.use_colors:
            print(f"{color}{formatted}{Colors.RESET}")
        else:
            print(formatted)
            
        # Write to file (no colors)
        if self.log_file:
            with open(self.log_file, "a") as f:
                clean = formatted if not color else formatted
                f.write(clean + "\n")
    
    def debug(self, message: str) -> None:
        """Log debug message (dimmed)."""
        self._log("DEBUG", message, Colors.DIM, self.SYMBOLS["debug"])
    
    def info(self, message: str) -> None:
        """Log info message."""
        self._log("INFO", message)
    
    def success(self, message: str) -> None:
        """Log success message (green with checkmark)."""
        self._log("INFO", message, Colors.GREEN, self.SYMBOLS["success"])
    
    def warning(self, message: str) -> None:
        """Log warning message (yellow with warning symbol)."""
        self._log("WARNING", message, Colors.YELLOW, self.SYMBOLS["warning"])
    
    def error(self, message: str) -> None:
        """Log error message (red with X)."""
        self._log("ERROR", message, Colors.RED, self.SYMBOLS["error"])
    
    def progress(self, current: int, total: int, item: str = "", 
                 extra: str = "") -> None:
        """
        Log progress update (overwrites previous line).
        
        Args:
            current: Current item number (1-indexed)
            total: Total number of items
            item: Name of current item
            extra: Additional info to display
        """
        if self.LEVELS.get("INFO", 1) < self.level:
            return
            
        pct = (current / total * 100) if total > 0 else 0
        bar_width = 20
        filled = int(bar_width * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)
        
        line = f"  [{current:3d}/{total:3d}] [{bar}] {pct:5.1f}%"
        if item:
            line += f"  {item}"
        if extra:
            line += f" │ {extra}"
        
        # Use carriage return for in-place updates
        end = '\n' if current == total else '\r'
        print(line.ljust(100), end=end, flush=True)
    
    def header(self, text: str, char: str = "═", width: int = 60) -> None:
        """Print a section header."""
        print()
        print(char * width)
        if self.use_colors:
            print(f"  {Colors.BOLD}{text}{Colors.RESET}")
        else:
            print(f"  {text}")
        print(char * width)
    
    def section(self, text: str) -> None:
        """Print a subsection header."""
        print()
        if self.use_colors:
            print(f"  {Colors.CYAN}▸ {text}{Colors.RESET}")
        else:
            print(f"  ▸ {text}")
    
    def item(self, key: str, value: Any = None, indent: int = 0) -> None:
        """Print a key-value item."""
        prefix = "  " * (indent + 1) + "• "
        if value is not None:
            print(f"{prefix}{key}: {value}")
        else:
            print(f"{prefix}{key}")


# Global logger instance
log = Logger()


# =============================================================================
# System Utilities
# =============================================================================

def run_command(
    cmd,
    timeout: Optional[int] = None,
    capture_output: bool = True,
    check: bool = False,
    cwd: Path = None
):
    """
    Run a shell command with timeout and error handling.
    
    Args:
        cmd: Command as string or list of strings
        timeout: Timeout in seconds (None for no timeout)
        capture_output: Whether to capture stdout/stderr
        check: Whether to raise exception on non-zero exit
        cwd: Working directory
        
    Returns:
        If cmd is a string: Tuple of (success, stdout, stderr)
        If cmd is a list: CompletedProcess instance
    """
    # Handle string commands (simple interface for other modules)
    if isinstance(cmd, str):
        log.debug(f"Running: {cmd}")
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                timeout=timeout,
                capture_output=True,
                text=True,
                cwd=cwd
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            log.debug(f"Command timed out after {timeout}s")
            return False, "", "TIMEOUT"
        except Exception as e:
            log.debug(f"Command failed: {e}")
            return False, "", str(e)
    
    # Handle list commands (CompletedProcess interface)
    log.debug(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=capture_output,
            text=True,
            check=check,
            cwd=cwd
        )
        return result
    except subprocess.TimeoutExpired:
        log.error(f"Command timed out after {timeout}s: {' '.join(cmd)}")
        raise
    except subprocess.CalledProcessError as e:
        log.error(f"Command failed with exit code {e.returncode}: {' '.join(cmd)}")
        raise


def check_binary_exists(binary_name: str, bin_dir: str = None) -> bool:
    """Check if a benchmark binary exists."""
    binary_dir = Path(bin_dir) if bin_dir else BIN_DIR
    binary_path = binary_dir / binary_name
    return binary_path.exists() and os.access(binary_path, os.X_OK)


def get_available_benchmarks(bin_dir: str = None) -> List[str]:
    """Get list of available benchmark binaries."""
    return [b for b in BENCHMARKS if check_binary_exists(b, bin_dir)]


# =============================================================================
# Algorithm Utilities
# =============================================================================

def parse_algorithm_option(option: str) -> Tuple[int, List[str]]:
    """
    Parse algorithm option string into ID and parameters.
    
    Examples:
        "0" -> (0, [])
        "12:graphbrew:quality" -> (12, ["graphbrew", "quality"])
    """
    parts = option.split(":")
    algo_id = int(parts[0])
    params = parts[1:] if len(parts) > 1 else []
    return algo_id, params


def get_algorithm_name(option: str) -> str:
    """
    Get canonical variant name for an algorithm option string.
    
    Uses _VARIANT_ALGO_REGISTRY to build names — no hardcoded algo IDs.
    
    Examples:
        "0"           → "ORIGINAL"
        "8"           → "RABBITORDER_csr"    (default variant)
        "8:boost"     → "RABBITORDER_boost"
        "12:leiden"   → "GraphBrewOrder_leiden"
        "12"          → "GraphBrewOrder_leiden" (default variant)
    """
    algo_id, params = parse_algorithm_option(option)
    base_name = ALGORITHMS.get(algo_id, f"Unknown({algo_id})")
    
    if algo_id in _VARIANT_ALGO_REGISTRY:
        prefix, _, default = _VARIANT_ALGO_REGISTRY[algo_id]
        suffix = params[0] if params else default
        return f"{prefix}{suffix}"
    
    if params:
        return f"{base_name}_{':'.join(params)}"
    return base_name


# =============================================================================
# JSON Utilities
# =============================================================================

def save_json(data: Any, path: Path, indent: int = 2) -> None:
    """Save data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)
    log.debug(f"Saved JSON to {path}")


# =============================================================================
# Graph File Utilities  
# =============================================================================

def find_graphs(graph_dir: Path = None, extensions: List[str] = None) -> List[Path]:
    """
    Find all graph files in directory.
    
    Args:
        graph_dir: Directory to search (default: GRAPHS_DIR)
        extensions: File extensions to match (default: [".mtx", ".el", ".wel"])
        
    Returns:
        List of graph file paths
    """
    if graph_dir is None:
        graph_dir = GRAPHS_DIR
    if extensions is None:
        extensions = [".mtx", ".el", ".wel"]
    
    graphs = []
    for ext in extensions:
        graphs.extend(graph_dir.rglob(f"*{ext}"))
    
    return sorted(graphs)


# =============================================================================
# Output Formatting Utilities
# =============================================================================

def format_size(size_bytes: int) -> str:
    """Format size in bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}PB"


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 0:
        return "0s"
    
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    days, hours = divmod(hours, 24)
    
    if days > 0:
        return f"{days}d {hours}h {mins}m"
    elif hours > 0:
        return f"{hours}h {mins}m {secs}s"
    elif mins > 0:
        return f"{mins}m {secs}s"
    elif seconds >= 1:
        return f"{secs}s"
    else:
        return f"{seconds*1000:.0f}ms"


def format_number(num: int) -> str:
    """Format large numbers with commas."""
    return f"{num:,}"


def format_table(
    headers: List[str],
    rows: List[List[Any]],
    alignments: List[str] = None,
    title: str = None
) -> str:
    """
    Format data as an ASCII table.
    
    Args:
        headers: Column headers
        rows: Data rows (list of lists)
        alignments: Column alignments ('l', 'c', 'r') per column
        title: Optional table title
        
    Returns:
        Formatted table string
    """
    if not rows:
        return "No data"
    
    # Determine column widths
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    
    # Default alignments (right for numbers, left for text)
    if alignments is None:
        alignments = ['r' if isinstance(rows[0][i], (int, float)) else 'l' 
                      for i in range(len(headers))]
    
    # Build format string
    def align(text: str, width: int, alignment: str) -> str:
        text = str(text)
        if alignment == 'l':
            return text.ljust(width)
        elif alignment == 'r':
            return text.rjust(width)
        else:
            return text.center(width)
    
    lines = []
    
    # Title
    if title:
        total_width = sum(widths) + 3 * (len(headers) - 1) + 4
        lines.append("")
        lines.append("  " + "─" * (total_width - 2))
        lines.append(f"  {title}")
        lines.append("  " + "─" * (total_width - 2))
    
    # Header
    header_line = "  │ " + " │ ".join(
        align(h, widths[i], 'c') for i, h in enumerate(headers)
    ) + " │"
    lines.append(header_line)
    
    # Separator
    sep_line = "  ├─" + "─┼─".join("─" * w for w in widths) + "─┤"
    lines.append(sep_line)
    
    # Rows
    for row in rows:
        row_line = "  │ " + " │ ".join(
            align(cell, widths[i], alignments[i]) for i, cell in enumerate(row)
        ) + " │"
        lines.append(row_line)
    
    # Footer
    footer_line = "  └─" + "─┴─".join("─" * w for w in widths) + "─┘"
    lines.append(footer_line)
    
    return "\n".join(lines)


def print_summary_box(title: str, items: Dict[str, Any], width: int = 50) -> None:
    """
    Print a summary box with key-value pairs.
    
    Args:
        title: Box title
        items: Dictionary of items to display
        width: Box width
    """
    print()
    print("  ┌" + "─" * width + "┐")
    print("  │ " + title.center(width - 2) + " │")
    print("  ├" + "─" * width + "┤")
    for key, value in items.items():
        line = f"  {key}: {value}"
        print("  │ " + line.ljust(width - 2) + " │")
    print("  └" + "─" * width + "┘")


# =============================================================================
# Standalone CLI
# =============================================================================

def main():
    """CLI for utility functions."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GraphBrew utilities")
    parser.add_argument("--check-deps", action="store_true",
                       help="Check if benchmark binaries are available")
    parser.add_argument("--list-graphs", action="store_true",
                       help="List all graphs in graphs directory")
    parser.add_argument("--list-algorithms", action="store_true",
                       help="List all algorithms with IDs")
    
    args = parser.parse_args()
    
    if args.check_deps:
        print("Checking benchmark binaries...")
        available = get_available_benchmarks()
        for bench in BENCHMARKS:
            status = "✓" if bench in available else "✗"
            print(f"  {status} {bench}")
        print(f"\nAvailable: {len(available)}/{len(BENCHMARKS)}")
    
    if args.list_graphs:
        graphs = find_graphs()
        print(f"Found {len(graphs)} graphs:")
        for g in graphs[:20]:
            print(f"  {g.name}")
        if len(graphs) > 20:
            print(f"  ... and {len(graphs) - 20} more")
    
    if args.list_algorithms:
        print("Algorithms (ID: Name):")
        for aid, name in sorted(ALGORITHMS.items()):
            print(f"  {aid:2d}: {name}")


if __name__ == "__main__":
    main()
