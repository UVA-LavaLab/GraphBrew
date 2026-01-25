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
GRAPHS_DIR = PROJECT_ROOT / "graphs"
WEIGHTS_DIR = SCRIPT_DIR / "weights"

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

# Leiden variant definitions
LEIDEN_DENDROGRAM_VARIANTS = ["dfs", "dfshub", "dfssize", "bfs", "hybrid"]
LEIDEN_CSR_VARIANTS = ["dfs", "bfs", "hubsort", "fast", "modularity"]

# Leiden default parameters
LEIDEN_DEFAULT_RESOLUTION = 1.0
LEIDEN_DEFAULT_PASSES = 3

# Benchmark definitions
BENCHMARKS = ["pr", "bfs", "cc", "sssp", "bc", "tc"]

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
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


class Logger:
    """Simple logger with levels and optional file output."""
    
    LEVELS = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
    
    def __init__(self, level: str = "INFO", log_file: Optional[Path] = None):
        self.level = self.LEVELS.get(level.upper(), 1)
        self.log_file = log_file
        
    def _log(self, level: str, message: str, color: str = "") -> None:
        if self.LEVELS.get(level, 0) >= self.level:
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted = f"[{timestamp}] [{level}] {message}"
            if color and sys.stdout.isatty():
                print(f"{color}{formatted}{Colors.RESET}")
            else:
                print(formatted)
            if self.log_file:
                with open(self.log_file, "a") as f:
                    f.write(formatted + "\n")
    
    def debug(self, message: str) -> None:
        self._log("DEBUG", message)
    
    def info(self, message: str) -> None:
        self._log("INFO", message)
    
    def success(self, message: str) -> None:
        self._log("INFO", message, Colors.GREEN)
    
    def warning(self, message: str) -> None:
        self._log("WARNING", message, Colors.YELLOW)
    
    def error(self, message: str) -> None:
        self._log("ERROR", message, Colors.RED)


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


def check_binary_exists(binary_name: str) -> bool:
    """Check if a benchmark binary exists."""
    binary_path = BIN_DIR / binary_name
    return binary_path.exists() and os.access(binary_path, os.X_OK)


def get_available_benchmarks() -> List[str]:
    """Get list of available benchmark binaries."""
    return [b for b in BENCHMARKS if check_binary_exists(b)]


# =============================================================================
# Algorithm Utilities
# =============================================================================

def parse_algorithm_option(option: str) -> Tuple[int, List[str]]:
    """
    Parse algorithm option string into ID and parameters.
    
    Examples:
        "0" -> (0, [])
        "16:1.0:hybrid" -> (16, ["1.0", "hybrid"])
        "17:1.0:3:fast" -> (17, ["1.0", "3", "fast"])
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
    """Get display name for algorithm option string."""
    algo_id, params = parse_algorithm_option(option)
    base_name = ALGORITHMS.get(algo_id, f"Unknown({algo_id})")
    if params:
        return f"{base_name}_{':'.join(params)}"
    return base_name


def expand_leiden_variants(algo_id: int) -> List[str]:
    """
    Expand Leiden algorithm ID to all its variants.
    
    Returns list of option strings for all variants.
    """
    if algo_id == 16:  # LeidenDendrogram
        return [f"16:1.0:{v}" for v in LEIDEN_DENDROGRAM_VARIANTS]
    elif algo_id == 17:  # LeidenCSR
        return [f"17:1.0:3:{v}" for v in LEIDEN_CSR_VARIANTS]
    else:
        return [str(algo_id)]


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
        print("LeidenDendrogram (16) variants:")
        for v in LEIDEN_DENDROGRAM_VARIANTS:
            print(f"  -o 16:1.0:{v}")
        print("\nLeidenCSR (17) variants:")
        for v in LEIDEN_CSR_VARIANTS:
            print(f"  -o 17:1.0:3:{v}")


if __name__ == "__main__":
    main()
