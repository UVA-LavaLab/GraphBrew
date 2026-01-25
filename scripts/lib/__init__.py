"""
GraphBrew Library - Modular utilities for graph reordering experiments.

This library provides functions for:
- Downloading benchmark graphs from SuiteSparse
- Building benchmark binaries
- Generating vertex reorderings
- Running performance benchmarks
- Running cache simulations
- Managing type-based weights for AdaptiveOrder
- Training ML weights

**Module Overview:**
- `utils`: Core constants (ALGORITHMS, BENCHMARKS), logging, command execution
- `download`: Graph catalog and download functions
- `build`: Binary compilation utilities
- `reorder`: Vertex reordering generation
- `benchmark`: Performance benchmark execution
- `cache`: Cache simulation analysis
- `weights`: Type-based weight management for AdaptiveOrder

**Standalone Usage:**
    python -m scripts.lib.download --size SMALL --list --stats
    python -m scripts.lib.build --check --build
    python -m scripts.lib.reorder --graph test.mtx --algorithms 0,8,9
    python -m scripts.lib.benchmark --graph test.mtx --algorithms 0,1,8
    python -m scripts.lib.cache --graph test.mtx --benchmarks pr,bfs
    python -m scripts.lib.weights --list-types --show-type type_0
    python -m scripts.lib.utils --list-algorithms

**Library Usage:**
    from scripts.lib import ALGORITHMS, BENCHMARKS, GRAPH_CATALOG
    from scripts.lib.download import download_graphs, DOWNLOAD_GRAPHS_SMALL
    from scripts.lib.build import build_binaries, check_binaries
    from scripts.lib.reorder import generate_reorderings, generate_label_maps
    from scripts.lib.benchmark import run_benchmark, run_benchmark_suite
    from scripts.lib.cache import run_cache_simulations
    from scripts.lib.weights import assign_graph_type, update_type_weights_incremental
"""

__version__ = "1.1.0"
__all__ = [
    "utils", "download", "build", "reorder", "benchmark", "cache", "weights"
]

# =============================================================================
# Core utilities
# =============================================================================
from .utils import (
    # Constants
    ALGORITHMS,
    ALGORITHM_IDS,
    BENCHMARKS,
    LEIDEN_CSR_VARIANTS,
    LEIDEN_DENDROGRAM_VARIANTS,
    LEIDEN_DEFAULT_RESOLUTION,
    LEIDEN_DEFAULT_PASSES,
    # Paths
    PROJECT_ROOT,
    BIN_DIR,
    BIN_SIM_DIR,
    RESULTS_DIR,
    GRAPHS_DIR,
    WEIGHTS_DIR,
    # Utilities
    Logger,
    run_command,
    get_timestamp,
)

# =============================================================================
# Graph download
# =============================================================================
from .download import (
    GRAPH_CATALOG,
    DOWNLOAD_GRAPHS_SMALL,
    DOWNLOAD_GRAPHS_MEDIUM,
    DOWNLOAD_GRAPHS_LARGE,
    DOWNLOAD_GRAPHS_XLARGE,
    DownloadableGraph,
    download_graph,
    download_graphs,
    get_graph_info,
    get_graphs_by_size,
    get_catalog_stats,
)

# =============================================================================
# Build utilities
# =============================================================================
from .build import (
    build_binaries,
    check_binaries,
    clean_build,
    ensure_binaries,
    check_build_requirements,
)

# =============================================================================
# Reordering
# =============================================================================
from .reorder import (
    ReorderResult,
    AlgorithmConfig,
    generate_reorderings,
    generate_label_maps,
    generate_reorderings_with_variants,
    expand_algorithms_with_variants,
    load_label_maps_index,
    get_label_map_path,
)

# =============================================================================
# Benchmarking
# =============================================================================
from .benchmark import (
    run_benchmark,
    parse_benchmark_output,
)

# =============================================================================
# Cache simulation
# =============================================================================
from .cache import (
    CacheResult,
    run_cache_simulation,
    run_cache_simulations,
    parse_cache_output,
    get_cache_stats_summary,
)

# =============================================================================
# Weight management
# =============================================================================
from .weights import (
    PerceptronWeight,
    load_type_registry,
    save_type_registry,
    load_type_weights,
    save_type_weights,
    assign_graph_type,
    update_type_weights_incremental,
    get_best_algorithm_for_type,
    list_known_types,
    get_type_summary,
    initialize_default_weights,
)
