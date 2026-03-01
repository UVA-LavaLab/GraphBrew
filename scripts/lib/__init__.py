"""
GraphBrew Library - Modular utilities for graph reordering experiments.

This library provides functions for:
- Downloading benchmark graphs from SuiteSparse
- Building benchmark binaries
- Generating vertex reorderings
- Running performance benchmarks
- Running cache simulations
- Streaming database model (benchmarks.json → C++ runtime predictions)
- Managing type-based weights for AdaptiveOrder
- Training & evaluating perceptron weights
- Progress tracking and reporting
- Centralized benchmark data storage (SSO)
- Adaptive order analysis

**Module Overview (SSO Architecture):**

Sub-packages:
- `core/`:     Constants, logging, data types, storage primitives
  - `utils`:            Core constants (ALGORITHMS, BENCHMARKS), logging, command execution
  - `graph_types`:      Data classes (GraphInfo, BenchmarkResult)
  - `datastore`:        Centralized benchmark data store (BenchmarkStore)
  - `graph_data`:       Per-graph data storage and run logging
- `pipeline/`: Experiment pipeline stages
  - `download`:         Graph catalog and download functions
  - `build`:            Binary compilation utilities
  - `dependencies`:     System dependency detection
  - `reorder`:          Vertex reordering generation
  - `benchmark`:        Performance benchmark execution
  - `cache`:            Cache simulation analysis
  - `phases`:           Phase orchestration
  - `progress`:         Visual progress tracking and reporting
- `ml/`:       Machine learning models
  - `weights`:          SSO for perceptron scoring & training (PerceptronWeight)
  - `eval_weights`:     Weight evaluation, data loading
  - `training`:         Iterative training loop
  - `adaptive_emulator`:Python emulator of C++ AdaptiveOrder
  - `oracle`:           Oracle (best-known) analysis
  - `features`:         Graph feature extraction
- `analysis/`: Result analysis and visualization
  - `adaptive`:         Adaptive order analysis functions
  - `metrics`:          Amortization & end-to-end metrics
  - `figures`:          SVG figure generation
- `tools/`:    Standalone maintenance tools
  - `check_includes`:   CI: scan C++ for legacy includes
  - `regen_features`:   Regenerate features.json via C++ binary

**Standalone Usage:**
    python -m scripts.lib.pipeline.download --size SMALL --list --stats
    python -m scripts.lib.pipeline.build --check --build
    python -m scripts.lib.pipeline.reorder --graph test.mtx --algorithms 0,8,9
    python -m scripts.lib.pipeline.benchmark --graph test.mtx --algorithms 0,1,8
    python -m scripts.lib.pipeline.cache --graph test.mtx --benchmarks pr,bfs
    python -m scripts.lib.ml.weights --list-types
    python -m scripts.lib.ml.eval_weights --logo
    python -m scripts.lib.core.utils --list-algorithms

**Library Usage:**
    from scripts.lib import ALGORITHMS, BENCHMARKS, GRAPH_CATALOG
    from scripts.lib.pipeline.download import download_graphs, DOWNLOAD_GRAPHS_SMALL
    from scripts.lib.pipeline.build import build_binaries, check_binaries
    from scripts.lib.pipeline.reorder import generate_reorderings, generate_label_maps
    from scripts.lib.pipeline.benchmark import run_benchmark, run_benchmarks_multi_graph
    from scripts.lib.pipeline.cache import run_cache_simulations, run_cache_simulations_with_variants
    from scripts.lib.ml.weights import assign_graph_type, update_type_weights_incremental, PerceptronWeight
    from scripts.lib.ml.eval_weights import load_all_results, compute_graph_features, find_best_algorithm
    from scripts.lib.core.datastore import get_benchmark_store, get_props_store
    from scripts.lib.pipeline.progress import ProgressTracker, create_progress
    from scripts.lib.analysis import analyze_adaptive_order, compare_adaptive_vs_fixed
    from scripts.lib.ml.training import train_adaptive_weights_iterative
"""

__version__ = "2.0.0"
__all__ = [
    # Sub-packages
    "core", "pipeline", "ml", "analysis", "tools",
]

# =============================================================================
# Core utilities
# =============================================================================
from .core.utils import (
    # Constants
    ALGORITHMS,
    ALGORITHM_IDS,
    ELIGIBLE_ALGORITHMS,
    BENCHMARKS,
    # Variant SSOT
    VARIANT_PREFIXES,
    VARIANT_ALGO_IDS,
    DISPLAY_TO_CANONICAL,
    # RabbitOrder variants (csr default, boost optional)
    RABBITORDER_VARIANTS,
    RABBITORDER_DEFAULT_VARIANT,
    # RCM variants (default=GoGraph, bnf=CSR-native BNF)
    RCM_VARIANTS,
    RCM_DEFAULT_VARIANT,
    # GraphBrewOrder variants
    GRAPHBREW_VARIANTS,
    GRAPHBREW_DEFAULT_VARIANT,
    # Variant helpers
    get_all_algorithm_variant_names,
    resolve_canonical_name,
    is_variant_prefixed,
    enumerate_graphbrew_multilayer,
    # Multi-layer config
    GRAPHBREW_LAYERS,
    GRAPHBREW_OPTIONS,
    # Leiden resolution/pass settings
    LEIDEN_DEFAULT_RESOLUTION,
    LEIDEN_DEFAULT_PASSES,
    # Leiden SSOT constants (match C++ reorder_graphbrew.h)
    LEIDEN_DEFAULT_TOLERANCE,
    LEIDEN_DEFAULT_AGGREGATION_TOLERANCE,
    LEIDEN_DEFAULT_QUALITY_FACTOR,
    LEIDEN_DEFAULT_MAX_ITERATIONS,
    LEIDEN_DEFAULT_MAX_PASSES,
    LEIDEN_MODULARITY_MAX_ITERATIONS,
    LEIDEN_MODULARITY_MAX_PASSES,
    # Weight normalization constants
    WEIGHT_PATH_LENGTH_NORMALIZATION,
    WEIGHT_REORDER_TIME_NORMALIZATION,
    WEIGHT_AVG_DEGREE_DEFAULT,
    # Paths
    PROJECT_ROOT,
    BIN_DIR,
    BIN_SIM_DIR,
    RESULTS_DIR,
    GRAPHS_DIR,
    WEIGHTS_DIR,
    MODELS_DIR,
    DATA_DIR,
    GRAPH_PROPS_FILE,
    BENCHMARK_DATA_FILE,
    # Utilities
    Logger,
    Colors,
    run_command,
    get_timestamp,
    # Formatting
    format_size,
    format_duration,
    format_number,
    format_table,
    print_summary_box,
)

# =============================================================================
# Graph features and system utilities
# =============================================================================
from .ml.features import (
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
from .pipeline.download import (
    GRAPH_CATALOG,
    DOWNLOAD_GRAPHS_SMALL,
    DOWNLOAD_GRAPHS_MEDIUM,
    DOWNLOAD_GRAPHS_LARGE,
    DOWNLOAD_GRAPHS_XLARGE,
    DownloadableGraph,
    download_graph,
    download_graphs,
    download_graphs_parallel,
    ParallelDownloadManager,
    DownloadStatus,
    DownloadProgress,
    get_graph_info,
    get_graphs_by_size,
    get_catalog_stats,
)

# =============================================================================
# Build utilities
# =============================================================================
from .pipeline.build import (
    build_binaries,
    check_binaries,
    clean_build,
    ensure_binaries,
    check_build_requirements,
    ensure_dependencies,
)

# =============================================================================
# Dependency management
# =============================================================================
try:
    from .pipeline.dependencies import (
        check_dependencies,
        install_dependencies,
        install_boost_158,
        check_boost_158,
        print_install_instructions,
        detect_platform,
        get_package_manager,
        get_boost_version,
    )
except ImportError:
    # Dependencies module may not be available in minimal installs
    pass

# =============================================================================
# Reordering
# =============================================================================
from .pipeline.reorder import (
    ReorderResult,
    AlgorithmConfig,
    generate_reorderings,
    generate_label_maps,
    generate_reorderings_with_variants,
    expand_algorithms_with_variants,
    load_label_maps_index,
    get_label_map_path,
    get_algorithm_name_with_variant,
)

# =============================================================================
# Benchmarking
# =============================================================================
from .pipeline.benchmark import (
    run_benchmark,
    parse_benchmark_output,
)

# =============================================================================
# Cache simulation
# =============================================================================
from .pipeline.cache import (
    CacheResult,
    run_cache_simulation,
    run_cache_simulations,
    run_cache_simulations_with_variants,
    parse_cache_output,
    get_cache_stats_summary,
)

# =============================================================================
# Weight management (SSO: scoring & training)
# NOTE: Type system functions (load_type_registry, assign_graph_type, etc.)
# are DEPRECATED — C++ now trains models from raw DB data at runtime.
# They are kept for backward compatibility but should not be used in new code.
# =============================================================================
from .ml.weights import (
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
    update_zero_weights,
    CLUSTER_DISTANCE_THRESHOLD,
)

# =============================================================================
# Weight evaluation & data loading
# =============================================================================
from .ml.eval_weights import (
    load_all_results,
    build_performance_matrix,
    compute_graph_features,
    find_best_algorithm,
    evaluate_predictions,
    evaluate_weights,
    train_and_evaluate,
)

# =============================================================================
# Progress tracking
# =============================================================================
from .pipeline.progress import (
    ProgressTracker,
    ConsoleColors,
    Timer,
    create_progress,
    get_progress,
    reset_progress,
)

# =============================================================================
# Analysis functions
# =============================================================================
from .analysis.adaptive import (
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
from .ml.training import (
    TrainingIterationResult,
    TrainingResult,
    compute_significance_weight,
    initialize_enhanced_weights,
    train_adaptive_weights_iterative,
    train_adaptive_weights_large_scale,
)
# =============================================================================
# Phase orchestration
# =============================================================================
from .pipeline.phases import (
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
# Type definitions (GraphInfo lives in graph_types to avoid circular imports)
# =============================================================================
from .core.graph_types import GraphInfo
from .core.utils import BenchmarkResult

# =============================================================================
# Per-graph data storage and run logging
# =============================================================================
from .core.graph_data import (
    # Data stores
    GraphDataStore,
    # Data classes
    GraphFeatures,
    AlgorithmBenchmarkData,
    AlgorithmReorderData,
    # Convenience functions
    save_graph_features,
    load_all_graph_data,
    # Run management
    list_runs,
    get_latest_run,
    cleanup_old_runs,
    # Run logging
    LOGS_DIR,
    GRAPH_DATA_DIR,
    save_run_log,
    list_graph_logs,
    list_sessions,
    read_log,
    read_log_by_name,
    get_latest_log,
    cleanup_old_logs,
    set_session_id,
    get_session_id,
    clear_session_id,
    # Migration
    migrate_old_structure,
    migrate_all_graphs,
)

# =============================================================================
# Centralized benchmark data store
# =============================================================================
from .core.datastore import (
    BenchmarkStore,
    GraphPropsStore,
    get_benchmark_store,
    get_props_store,
)