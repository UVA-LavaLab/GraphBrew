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
- Progress tracking and reporting
- Result file management
- Adaptive order analysis

**Module Overview:**
- `utils`: Core constants (ALGORITHMS, BENCHMARKS), logging, command execution
- `download`: Graph catalog and download functions
- `build`: Binary compilation utilities
- `reorder`: Vertex reordering generation
- `benchmark`: Performance benchmark execution
- `cache`: Cache simulation analysis
- `weights`: Type-based weight management for AdaptiveOrder
- `progress`: Visual progress tracking and reporting
- `results`: Result file I/O and aggregation
- `analysis`: Adaptive order analysis functions
- `training`: Weight training functions

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
    from scripts.lib.progress import ProgressTracker, create_progress
    from scripts.lib.results import ResultsManager, read_json, write_json
    from scripts.lib.analysis import analyze_adaptive_order, compare_adaptive_vs_fixed
    from scripts.lib.training import train_adaptive_weights_iterative
"""

__version__ = "1.2.0"
__all__ = [
    "utils", "download", "build", "reorder", "benchmark", "cache", "weights",
    "progress", "results", "analysis", "training", "features", "phases", "types"
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
# Graph features and system utilities
# =============================================================================
from .features import (
    # Graph type constants
    GRAPH_TYPE_GENERIC,
    GRAPH_TYPE_SOCIAL,
    GRAPH_TYPE_ROAD,
    GRAPH_TYPE_WEB,
    GRAPH_TYPE_POWERLAW,
    GRAPH_TYPE_UNIFORM,
    ALL_GRAPH_TYPES,
    # Memory constants
    BYTES_PER_EDGE,
    BYTES_PER_NODE,
    MEMORY_SAFETY_FACTOR,
    # Graph properties cache
    load_graph_properties_cache,
    save_graph_properties_cache,
    update_graph_properties,
    get_graph_properties,
    clear_graph_properties_cache,
    # Graph type detection
    detect_graph_type,
    get_graph_type_from_name,
    get_graph_type_from_properties,
    # Topological features
    compute_clustering_coefficient_sample,
    estimate_diameter_bfs,
    count_subcommunities_quick,
    compute_extended_features,
    # System utilities
    get_available_memory_gb,
    get_total_memory_gb,
    estimate_graph_memory_gb,
    get_available_disk_gb,
    get_total_disk_gb,
    get_num_threads,
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
    get_type_weights_file,
    assign_graph_type,
    update_type_weights_incremental,
    get_best_algorithm_for_type,
    list_known_types,
    get_type_summary,
    initialize_default_weights,
    CLUSTER_DISTANCE_THRESHOLD,
)

# =============================================================================
# Progress tracking
# =============================================================================
from .progress import (
    ProgressTracker,
    ConsoleColors,
    Timer,
    format_duration,
    create_progress,
    get_progress,
    reset_progress,
)

# =============================================================================
# Results management
# =============================================================================
from .results import (
    ResultsManager,
    read_json,
    write_json,
    read_csv,
    write_csv,
    append_csv,
    generate_timestamp,
    generate_result_filename,
    flatten_dict,
    filter_results,
    group_results,
    compute_statistics,
    ResultSchema,
    BENCHMARK_RESULT_SCHEMA,
    REORDER_RESULT_SCHEMA,
    CACHE_RESULT_SCHEMA,
)

# =============================================================================
# Analysis functions
# =============================================================================
from .analysis import (
    SubcommunityInfo,
    AdaptiveOrderResult,
    AdaptiveComparisonResult,
    SubcommunityBruteForceResult,
    GraphBruteForceAnalysis,
    parse_adaptive_output,
    analyze_adaptive_order,
    compare_adaptive_vs_fixed,
    run_subcommunity_brute_force,
)

# =============================================================================
# Training functions
# =============================================================================
from .training import (
    TrainingIterationResult,
    TrainingResult,
    initialize_enhanced_weights,
    train_adaptive_weights_iterative,
    train_adaptive_weights_large_scale,
)
# =============================================================================
# Phase orchestration
# =============================================================================
from .phases import (
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
    quick_benchmark,
    compare_algorithms,
)

# =============================================================================
# Type definitions
# =============================================================================
from .types import (
    GraphInfo,
    ReorderResult,
    BenchmarkResult,
    CacheResult,
    AdaptiveAnalysisResult,
)