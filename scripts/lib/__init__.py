"""
GraphBrew Library - Modular utilities for graph reordering experiments.

This library provides functions for:
- Downloading benchmark graphs from SuiteSparse
- Running performance benchmarks  
- Analyzing graph properties
- Training ML weights for AdaptiveOrder

Usage as library:
    from scripts.lib.utils import ALGORITHMS, BENCHMARKS, Logger
    from scripts.lib.download import download_graphs, GRAPH_CATALOG
    from scripts.lib.benchmark import run_benchmark, BenchmarkResult
    
Each module can also run standalone:
    python -m scripts.lib.download --size SMALL
    python -m scripts.lib.benchmark --graph test.mtx --algorithms 0,1,8
    python -m scripts.lib.utils --list-algorithms
"""

__version__ = "1.0.0"
__all__ = ["download", "benchmark", "utils"]

# Re-export key symbols for convenience
from .utils import (
    ALGORITHMS,
    BENCHMARKS,
    LEIDEN_CSR_VARIANTS,
    LEIDEN_DENDROGRAM_VARIANTS,
    PROJECT_ROOT,
    BIN_DIR,
    RESULTS_DIR,
    GRAPHS_DIR,
    Logger,
    BenchmarkResult,
)
from .download import GRAPH_CATALOG, download_graph, download_graphs
from .benchmark import run_benchmark
