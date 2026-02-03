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
import shutil
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
WEIGHTS_DIR = SCRIPT_DIR / "weights"
ACTIVE_WEIGHTS_DIR = WEIGHTS_DIR / "active"  # Where C++ reads type_*.json from

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
    16: "LeidenDendrogram",
    17: "LeidenCSR",
}

# Reverse mapping
ALGORITHM_IDS = {v: k for k, v in ALGORITHMS.items()}

# Algorithm categories
QUICK_ALGORITHMS = {0, 1, 2, 3, 4, 5, 6, 7, 8}  # Fast algorithms
SLOW_ALGORITHMS = {9, 10, 11}  # Gorder, Corder, RCM - can be slow on large graphs
LEIDEN_ALGORITHMS = {15, 16, 17}
COMMUNITY_ALGORITHMS = {8, 12, 14, 15, 16, 17}

# RabbitOrder variant definitions (default: csr)
# Format: -o 8:variant where variant = csr (default) or boost
RABBITORDER_VARIANTS = ["csr", "boost"]
RABBITORDER_DEFAULT_VARIANT = "csr"  # Native CSR (faster, no external deps)

# GraphBrewOrder variant definitions (default: leiden for backward compat)
# Format: -o 12:cluster_variant:final_algo:resolution:levels
# cluster_variant: leiden (default), gve, gveopt, rabbit, hubcluster
# "fast" suffix: use HubSortDBG (6) instead of RabbitOrder (8) for final layer
# Examples: gvefast, gveoptfast, fast (= gvefast)
GRAPHBREW_VARIANTS = ["leiden", "gve", "gveopt", "gvefast", "gveoptfast", "rabbit", "hubcluster"]
GRAPHBREW_DEFAULT_VARIANT = "leiden"  # Original Leiden library (backward compatible)

# Leiden variant definitions
LEIDEN_DENDROGRAM_VARIANTS = ["dfs", "dfshub", "dfssize", "bfs", "hybrid"]
# LeidenCSR variants (default: gve for best quality)
# Format: -o 17:variant:resolution:iterations:passes
# Resolution modes:
#   - Fixed: 1.5 (use specified value)
#   - Auto: "auto" or "0" (compute from graph density/CV)
#   - Dynamic: "dynamic" (adjust per-pass, gveadaptive only)
#   - Dynamic+Init: "dynamic_2.0" (start at 2.0, adjust per-pass)
# New optimized variants:
#   gveopt2: CSR-based aggregation (fastest reordering, best PR performance)
#   gveadaptive: Dynamic resolution adjustment (best for unknown graphs)
#   gveoptsort: Multi-level sort ordering (LeidenOrder-style)
#   gveturbo: Speed-optimized (optional refinement skip)
#   gvefast: CSR buffer reuse (leiden.hxx style)
# Legacy variants: gvedendo/gveoptdendo (incremental dendrogram)
LEIDEN_CSR_VARIANTS = [
    "gve", "gveopt", "gveopt2", "gveadaptive", "gveoptsort", "gveturbo", "gvefast",
    "gvedendo", "gveoptdendo", "gverabbit", "dfs", "bfs", "hubsort", "modularity"
]
LEIDEN_CSR_DEFAULT_VARIANT = "gve"  # GVE-Leiden (best modularity quality)

# Recommended variants for different use cases
LEIDEN_CSR_FAST_VARIANTS = ["gveopt2", "gveadaptive", "gveturbo", "gvefast", "gverabbit"]  # Speed priority
LEIDEN_CSR_QUALITY_VARIANTS = ["gve", "gveopt", "gveopt2", "gveadaptive"]  # Quality priority

# Resolution modes
LEIDEN_RESOLUTION_MODES = ["auto", "dynamic", "1.0", "1.5", "2.0"]

# Leiden default parameters
# "dynamic" gives best PR performance on most graphs (especially web graphs)
# "auto" is good for unknown graphs, fixed values for specific tuning
LEIDEN_DEFAULT_RESOLUTION = "dynamic"  # Dynamic adjustment per-pass
LEIDEN_DEFAULT_PASSES = 3

# Benchmark definitions
BENCHMARKS = ["pr", "bfs", "cc", "sssp", "bc", "tc"]
BENCHMARKS_DEFAULT = ["pr", "bfs", "cc", "sssp", "bc"]  # TC skipped by default (minimal reorder benefit)

# =============================================================================
# Graph Size Thresholds (MB) - Single Source of Truth
# =============================================================================
# Used for skip logic (slow algorithms, heavy simulations) and categorization
SIZE_SMALL = 50       # < 50 MB: quick experiments
SIZE_MEDIUM = 500     # < 500 MB: moderate size, may skip slow algorithms
SIZE_LARGE = 2000     # < 2 GB: large graphs, skip heavy operations
SIZE_XLARGE = 10000   # >= 2 GB: very large graphs

# =============================================================================
# Timeout Constants (seconds) - Single Source of Truth
# =============================================================================
TIMEOUT_REORDER = 43200       # 12 hours for reordering (GORDER can be slow)
TIMEOUT_BENCHMARK = 600       # 10 min for benchmarks
TIMEOUT_SIM = 1200            # 20 min for cache simulations
TIMEOUT_SIM_HEAVY = 3600      # 1 hour for heavy simulations (bc, sssp)

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
    success: bool = True
    error: str = ""
    extra: Dict = None
    
    def __post_init__(self):
        if self.extra is None:
            self.extra = {}
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass  
class GraphProperties:
    """Properties extracted from graph analysis."""
    name: str
    num_vertices: int = 0
    num_edges: int = 0
    avg_degree: float = 0.0
    max_degree: int = 0
    degree_variance: float = 0.0
    hub_concentration: float = 0.0
    density: float = 0.0
    clustering_coefficient: float = 0.0
    modularity: float = 0.5
    
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
        
    def _colorize(self, text: str, color: str) -> str:
        """Apply color if colors are enabled."""
        if self.use_colors:
            return f"{color}{text}{Colors.RESET}"
        return text
        
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
        "16:hybrid:1.0" -> (16, ["hybrid", "1.0"])
        "17:gve:1.0:20:10" -> (17, ["gve", "1.0", "20", "10"])
    """
    parts = option.split(":")
    algo_id = int(parts[0])
    params = parts[1:] if len(parts) > 1 else []
    return algo_id, params


def format_algorithm_option(algo_id: int, params: List[str] = None) -> str:
    """Format algorithm ID and parameters into option string."""
    if params:
        return f"{algo_id}:{':'.join(params)}"
    return str(algo_id)


def get_algorithm_name(option: str) -> str:
    """
    Get display name for algorithm option string.
    
    ALWAYS includes variant in name for algorithms that have variants:
    - RABBITORDER (8): RABBITORDER_csr (default) or RABBITORDER_boost
    - GraphBrewOrder (12): GraphBrewOrder_leiden (default), GraphBrewOrder_gve, etc.
    - LeidenDendrogram (16): LeidenDendrogram_dfs (default), etc.
    - LeidenCSR (17): LeidenCSR_gve (default), LeidenCSR_gveopt, etc.
    
    For other algorithms, returns the base name.
    """
    algo_id, params = parse_algorithm_option(option)
    base_name = ALGORITHMS.get(algo_id, f"Unknown({algo_id})")
    
    # If params provided, use them to build name
    if params:
        # For variant-based algorithms, extract just the variant (first param)
        if algo_id in [12, 16, 17]:  # GraphBrewOrder, LeidenDendrogram, LeidenCSR
            return f"{base_name}_{params[0]}"
        # For RabbitOrder, include variant
        elif algo_id == 8:
            return f"RABBITORDER_{params[0]}"
        else:
            return f"{base_name}_{':'.join(params)}"
    
    # No params - use default variant for algorithms that have variants
    if algo_id == 8:  # RABBITORDER
        return f"RABBITORDER_{RABBITORDER_DEFAULT_VARIANT}"
    elif algo_id == 12:  # GraphBrewOrder
        return f"{base_name}_{GRAPHBREW_DEFAULT_VARIANT}"
    elif algo_id == 17:  # LeidenCSR
        return f"{base_name}_{LEIDEN_CSR_DEFAULT_VARIANT}"
    elif algo_id == 16:  # LeidenDendrogram
        return f"{base_name}_{LEIDEN_DENDROGRAM_VARIANTS[0]}"  # Default: dfs
    
    return base_name


def expand_algorithm_variants(algo_id: int) -> List[str]:
    """
    Expand algorithm ID to all its variants.
    
    Returns list of option strings for all variants.
    Supported:
    - RabbitOrder (8): 8:csr, 8:boost
    - GraphBrewOrder (12): 12:leiden, 12:gve, 12:gveopt, 12:rabbit, 12:hubcluster
    - LeidenDendrogram (16): 16:dfs, 16:dfshub, etc.
    - LeidenCSR (17): 17:gve, 17:gveopt, 17:gvedendo, etc.
    """
    if algo_id == 8:  # RABBITORDER
        return [f"8:{v}" for v in RABBITORDER_VARIANTS]
    elif algo_id == 12:  # GraphBrewOrder
        return [f"12:{v}" for v in GRAPHBREW_VARIANTS]
    elif algo_id == 16:  # LeidenDendrogram
        return [f"16:{v}:1.0" for v in LEIDEN_DENDROGRAM_VARIANTS]
    elif algo_id == 17:  # LeidenCSR
        return [f"17:{v}:1.0:20:10" for v in LEIDEN_CSR_VARIANTS]
    else:
        return [str(algo_id)]


# Backward compatibility alias
def expand_leiden_variants(algo_id: int) -> List[str]:
    """Deprecated: Use expand_algorithm_variants instead."""
    return expand_algorithm_variants(algo_id)


# =============================================================================
# JSON Utilities
# =============================================================================

def load_json(path: Path) -> Dict:
    """Load JSON file with error handling."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        log.warning(f"JSON file not found: {path}")
        return {}
    except json.JSONDecodeError as e:
        log.error(f"Invalid JSON in {path}: {e}")
        return {}


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


def get_graph_name(graph_path: Path) -> str:
    """Extract clean graph name from path."""
    return graph_path.stem


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
    parser.add_argument("--list-leiden-variants", action="store_true",
                       help="List all Leiden variants")
    
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
    
    if args.list_leiden_variants:
        print("LeidenDendrogram (16) variants (format: 16:variant:resolution):")
        for v in LEIDEN_DENDROGRAM_VARIANTS:
            print(f"  -o 16:{v}:1.0")
        print("\nLeidenCSR (17) variants (format: 17:variant:resolution:iterations:passes):")
        for v in LEIDEN_CSR_VARIANTS:
            print(f"  -o 17:{v}:1.0:20:10")


if __name__ == "__main__":
    main()
