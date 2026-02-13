"""
GraphBrew Phase Orchestration Module
=====================================

This module provides clean, reusable phase orchestration functions that wrap
the underlying lib modules. It's designed to make building custom experiment
pipelines simple and maintainable.

**Architecture Overview:**

    ┌─────────────────────────────────────────────────────────────────────┐
    │                     YOUR EXPERIMENT SCRIPT                          │
    │  (e.g., graphbrew_experiment.py or custom_pipeline.py)             │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                      lib/phases.py (THIS MODULE)                    │
    │  Provides: run_reorder_phase, run_benchmark_phase, etc.            │
    │  Handles: Configuration, progress tracking, result saving          │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                     LOW-LEVEL LIB MODULES                           │
    │  lib/reorder.py, lib/benchmark.py, lib/cache.py, lib/weights.py   │
    │  These do the actual work but require manual setup                  │
    └─────────────────────────────────────────────────────────────────────┘

**Available Phases:**

    1. REORDER    - Generate vertex reorderings (label maps)
    2. BENCHMARK  - Run performance benchmarks
    3. CACHE      - Run cache miss simulations  
    4. WEIGHTS    - Generate perceptron weights from results
    5. FILL       - Update zero weights with defaults
    6. ADAPTIVE   - Analyze adaptive ordering decisions
    7. COMPARISON - Compare adaptive vs fixed algorithms
    8. BRUTE      - Brute-force validate all algorithms
    9. TRAINING   - Iteratively train adaptive weights
    10. LARGE_TRAINING - Batched training for large datasets

**Usage Examples:**

    # Simple: Run full pipeline
    from lib.phases import run_full_pipeline, PhaseConfig
    
    config = PhaseConfig(benchmarks=['pr', 'bfs', 'cc'])
    results = run_full_pipeline(graphs, algorithms, config)
    
    # Custom: Run specific phases
    from lib.phases import run_reorder_phase, run_benchmark_phase
    
    reorder_results, label_maps = run_reorder_phase(graphs, algorithms, config)
    benchmark_results = run_benchmark_phase(graphs, algorithms, label_maps, config)
    
    # Quick: Benchmark a single graph
    from lib.phases import quick_benchmark
    
    results = quick_benchmark("graphs/email-Enron/email-Enron.mtx")

**Configuration:**

    Use PhaseConfig to customize behavior:
    
    config = PhaseConfig(
        bin_dir="bench/bin",      # Directory with binaries
        results_dir="results",     # Where to save results
        benchmarks=['pr', 'bfs'], # Which benchmarks to run
        trials=3,                  # Trials per configuration
        skip_slow=True,           # Skip slow algorithms (GORDER, etc)
        expand_variants=True,     # Test all Leiden variants
    )

**Phase Results:**

    Each phase returns specific result types:
    - reorder  -> (List[ReorderResult], Dict[str, Dict[str, str]])
    - benchmark -> List[BenchmarkResult]
    - cache    -> List[CacheResult]
    - adaptive -> List[AdaptiveAnalysisResult]
    - training -> TrainingResult

**Error Handling:**

    Phases are designed to be resilient:
    - Individual failures don't stop the pipeline
    - Errors are captured in result objects (result.error)
    - Progress tracker records warnings/errors
    - Results are auto-saved after each phase

See Also:
    - lib/graph_types.py for data class definitions
    - lib/progress.py for progress tracking
    - scripts/examples/ for usage examples
"""

import os
import json
import glob
from datetime import datetime
from dataclasses import asdict
from typing import List, Dict, Optional, Any, Tuple

# Import from lib modules
from .graph_types import (
    GraphInfo, ReorderResult, BenchmarkResult, CacheResult,
    AdaptiveAnalysisResult, TrainingResult
)
from .reorder import (
    generate_reorderings, generate_reorderings_with_variants,
    load_label_maps_index, expand_algorithms_with_variants
)
from .benchmark import run_benchmark_suite
from .cache import run_cache_simulations, get_cache_stats_summary
from .analysis import (
    analyze_adaptive_order, compare_adaptive_vs_fixed,
    run_subcommunity_brute_force
)
from .training import (
    train_adaptive_weights_iterative, train_adaptive_weights_large_scale,
    initialize_enhanced_weights
)
from .weights import (
    load_type_registry, save_type_registry,
    update_type_weights_incremental, get_best_algorithm_for_type,
    initialize_default_weights
)
from .progress import ProgressTracker
from .utils import (
    RABBITORDER_VARIANTS,
    GRAPHBREW_VARIANTS,
    LEIDEN_DEFAULT_RESOLUTION,
    LEIDEN_DEFAULT_PASSES,
)


# =============================================================================
# Phase Configuration
# =============================================================================

class PhaseConfig:
    """
    Configuration container for pipeline phases.
    
    This class centralizes all settings needed by the various phases.
    Using a single config object makes it easy to:
    - Pass consistent settings across phases
    - Create from command-line arguments
    - Modify settings for different experiments
    
    Attributes:
        bin_dir: Path to benchmark binaries (default: "bench/bin")
        bin_sim_dir: Path to cache simulation binaries (default: "bench/bin_sim")
        graphs_dir: Path to graph files (default: "results/graphs")
        results_dir: Where to save results (default: "results")
        weights_dir: Where perceptron weights are stored (default: "results/weights")
        
        timeout_reorder: Max seconds for reordering (default: 300)
        timeout_benchmark: Max seconds per benchmark (default: 120)
        timeout_sim: Max seconds for cache simulation (default: 600)
        
        benchmarks: List of benchmarks to run (default: ['pr', 'bfs', 'cc'])
        trials: Number of trials per configuration (default: 3)
        
        skip_slow: Skip slow algorithms like GORDER, CORDER, RCM (default: False)
        skip_cache: Skip cache simulation phase (default: False)
        skip_heavy: Skip heavy simulations for BC/SSSP (default: False)
        force_reorder: Re-generate reorderings even if they exist (default: False)
        expand_variants: Expand Leiden algorithms to all variants (default: False)
        update_weights: Update weights incrementally during phases (default: True)
        
        leiden_resolution: Resolution: "dynamic" (default, best PR), "auto", "1.0", etc.
        leiden_passes: Number of Leiden optimization passes (default: 10)
        
        target_accuracy: Target accuracy for training (default: 0.95)
        max_iterations: Max training iterations (default: 10)
        learning_rate: Learning rate for weight updates (default: 0.1)
        batch_size: Batch size for large-scale training (default: 5)
        
        progress: ProgressTracker instance for visual feedback
    
    Example:
        # Basic usage
        config = PhaseConfig(benchmarks=['pr', 'bfs'], trials=5)
        
        # From argparse
        args = parser.parse_args()
        config = PhaseConfig.from_args(args)
        
        # Quick testing setup
        config = PhaseConfig(
            skip_slow=True,
            skip_cache=True,
            trials=1
        )
    """
    
    def __init__(
        self,
        # ─────────────────────────────────────────────────────────────────────
        # Directory paths
        # ─────────────────────────────────────────────────────────────────────
        bin_dir: str = "bench/bin",
        bin_sim_dir: str = "bench/bin_sim",
        graphs_dir: str = "results/graphs",
        results_dir: str = "results",
        weights_dir: str = "results/weights",
        
        # ─────────────────────────────────────────────────────────────────────
        # Timeout settings (seconds)
        # ─────────────────────────────────────────────────────────────────────
        timeout_reorder: int = 300,
        timeout_benchmark: int = 120,
        timeout_sim: int = 600,
        
        # ─────────────────────────────────────────────────────────────────────
        # Benchmark settings
        # ─────────────────────────────────────────────────────────────────────
        benchmarks: List[str] = None,
        trials: int = 3,
        
        # ─────────────────────────────────────────────────────────────────────
        # Feature flags
        # ─────────────────────────────────────────────────────────────────────
        skip_slow: bool = False,
        skip_cache: bool = False,
        skip_heavy: bool = False,
        force_reorder: bool = False,
        expand_variants: bool = False,
        update_weights: bool = True,
        
        # ─────────────────────────────────────────────────────────────────
        # Leiden-specific settings
        # ─────────────────────────────────────────────────────────────────────
        leiden_resolution: str = "dynamic",  # "dynamic" (default), "auto", "1.0", etc.
        leiden_passes: int = LEIDEN_DEFAULT_PASSES,
        rabbit_variants: List[str] = None,
        graphbrew_variants: List[str] = None,
        
        # ─────────────────────────────────────────────────────────────────────
        # Training settings
        # ─────────────────────────────────────────────────────────────────────
        target_accuracy: float = 0.95,
        max_iterations: int = 10,
        learning_rate: float = 0.1,
        batch_size: int = 5,
        
        # ─────────────────────────────────────────────────────────────────────
        # Progress tracking
        # ─────────────────────────────────────────────────────────────────────
        progress: ProgressTracker = None,
    ):
        """Initialize configuration with given settings."""
        # Directories
        self.bin_dir = bin_dir
        self.bin_sim_dir = bin_sim_dir
        self.graphs_dir = graphs_dir
        self.results_dir = results_dir
        self.weights_dir = weights_dir
        
        # Timeouts
        self.timeout_reorder = timeout_reorder
        self.timeout_benchmark = timeout_benchmark
        self.timeout_sim = timeout_sim
        
        # Benchmarks
        self.benchmarks = benchmarks or ['pr', 'bfs', 'cc']
        self.trials = trials
        
        # Feature flags
        self.skip_slow = skip_slow
        self.skip_cache = skip_cache
        self.skip_heavy = skip_heavy
        self.force_reorder = force_reorder
        self.expand_variants = expand_variants
        self.update_weights = update_weights
        
        # Leiden settings - use lib/utils.py as single source of truth
        self.leiden_resolution = leiden_resolution
        self.leiden_passes = leiden_passes
        self.rabbit_variants = rabbit_variants or ['csr']  # Default: csr only (not all variants)
        self.graphbrew_variants = graphbrew_variants or ['leiden']  # Default: leiden only (not all variants)
        
        # Training settings
        self.target_accuracy = target_accuracy
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Progress tracker (create default if not provided)
        self.progress = progress or ProgressTracker()
    
    @classmethod
    def from_args(cls, args) -> 'PhaseConfig':
        """Create config from argparse namespace."""
        return cls(
            bin_dir=getattr(args, 'bin_dir', 'bench/bin'),
            bin_sim_dir=getattr(args, 'bin_sim_dir', 'bench/bin_sim'),
            graphs_dir=getattr(args, 'graphs_dir', 'results/graphs'),
            results_dir=getattr(args, 'results_dir', 'results'),
            weights_dir=getattr(args, 'weights_dir', 'results/weights'),
            timeout_reorder=getattr(args, 'timeout_reorder', 300),
            timeout_benchmark=getattr(args, 'timeout_benchmark', 120),
            timeout_sim=getattr(args, 'timeout_sim', 600),
            benchmarks=getattr(args, 'benchmarks', ['pr', 'bfs', 'cc']),
            trials=getattr(args, 'trials', 3),
            skip_slow=getattr(args, 'skip_slow', False),
            skip_cache=getattr(args, 'skip_cache', False),
            skip_heavy=getattr(args, 'skip_heavy', False),
            force_reorder=getattr(args, 'force_reorder', False),
            expand_variants=getattr(args, 'expand_variants', False),
            update_weights=not getattr(args, 'no_incremental', False),
            leiden_resolution=getattr(args, 'leiden_resolution', 'dynamic'),
            leiden_passes=getattr(args, 'leiden_passes', 10),
            rabbit_variants=getattr(args, 'rabbit_variants', None),
            graphbrew_variants=getattr(args, 'graphbrew_variants', None),
            target_accuracy=getattr(args, 'target_accuracy', 0.95),
            max_iterations=getattr(args, 'max_iterations', 10),
            learning_rate=getattr(args, 'learning_rate', 0.1),
            batch_size=getattr(args, 'batch_size', 5),
        )


# =============================================================================
# Utility Functions
# =============================================================================

def _get_timestamp() -> str:
    """Get current timestamp for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _save_results(results: List, filename: str, results_dir: str) -> str:
    """Save results to JSON file."""
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)
    with open(filepath, 'w') as f:
        json.dump([asdict(r) if hasattr(r, '__dataclass_fields__') else r for r in results], f, indent=2)
    return filepath


def _load_latest_results(pattern: str, results_dir: str, result_class) -> List:
    """Load latest results matching pattern."""
    files = glob.glob(os.path.join(results_dir, pattern))
    if not files:
        return []
    latest = max(files)
    with open(latest) as f:
        data = json.load(f)
    return [result_class(**r) for r in data]


# =============================================================================
# Phase 1: Reordering
# =============================================================================

def run_reorder_phase(
    graphs: List[GraphInfo],
    algorithms: List[int],
    config: PhaseConfig,
    label_maps: Dict[str, Dict[str, str]] = None,
) -> Tuple[List[ReorderResult], Dict[str, Dict[str, str]]]:
    """
    Run the reordering phase.
    
    Args:
        graphs: List of graphs to process
        algorithms: List of algorithm IDs
        config: Phase configuration
        label_maps: Existing label maps (optional)
    
    Returns:
        Tuple of (reorder_results, updated_label_maps)
    """
    config.progress.phase_start("REORDERING", "Generating vertex reorderings")
    
    timestamp = _get_timestamp()
    label_maps = label_maps or {}
    
    if config.expand_variants:
        # Expand to include Leiden, RabbitOrder, and GraphBrewOrder variants
        expanded_algorithms = expand_algorithms_with_variants(
            algorithms,
            expand_leiden_variants=True,
            rabbit_variants=config.rabbit_variants,
            graphbrew_variants=config.graphbrew_variants
        )
        
        # generate_reorderings_with_variants returns (label_maps, results) tuple
        variant_label_maps, results = generate_reorderings_with_variants(
            graphs=graphs,
            algorithms=expanded_algorithms,
            bin_dir=config.bin_dir,
            output_dir=config.results_dir,
            timeout=config.timeout_reorder,
            skip_slow=config.skip_slow,
            force_reorder=config.force_reorder,
            leiden_resolution=config.leiden_resolution,
            leiden_passes=config.leiden_passes
        )
        # Merge the variant label maps into our label_maps
        for graph_name, algo_maps in variant_label_maps.items():
            if graph_name not in label_maps:
                label_maps[graph_name] = {}
            label_maps[graph_name].update(algo_maps)
    else:
        results = generate_reorderings(
            graphs=graphs,
            algorithms=algorithms,
            bin_dir=config.bin_dir,
            output_dir=config.results_dir,
            timeout=config.timeout_reorder,
            skip_slow=config.skip_slow,
            generate_maps=True,
            force_reorder=config.force_reorder,
            progress=config.progress
        )
        # Update label maps from results (non-variant mode)
        for r in results:
            if r.success and r.mapping_file:
                if r.graph not in label_maps:
                    label_maps[r.graph] = {}
                label_maps[r.graph][r.algorithm_name] = r.mapping_file
    
    # Save results
    filepath = _save_results(results, f"reorder_{timestamp}.json", config.results_dir)
    config.progress.success(f"Reorder results saved to: {filepath}")
    
    config.progress.phase_end(f"Generated {len(results)} reorderings")
    return results, label_maps


# =============================================================================
# Phase 2: Benchmarking
# =============================================================================

def run_benchmark_phase(
    graphs: List[GraphInfo],
    algorithms: List[int],
    label_maps: Dict[str, Dict[str, str]],
    config: PhaseConfig,
) -> List[BenchmarkResult]:
    """
    Run the benchmarking phase.
    
    Args:
        graphs: List of graphs to benchmark
        algorithms: List of algorithm IDs
        label_maps: Label maps from reorder phase
        config: Phase configuration
    
    Returns:
        List of benchmark results
    """
    config.progress.phase_start("BENCHMARKING", "Running performance benchmarks")
    
    timestamp = _get_timestamp()
    
    # Check if we have variant-expanded label maps
    has_variant_maps = (label_maps and 
                       any('_' in algo_name for g in label_maps.values() for algo_name in g.keys()))
    
    if has_variant_maps and config.expand_variants:
        config.progress.info("Mode: Variant-aware benchmarking")
        # Import variant-aware benchmark function
        from .benchmark import run_benchmarks_with_variants
        results = run_benchmarks_with_variants(
            graphs=graphs,
            label_maps=label_maps,
            benchmarks=config.benchmarks,
            bin_dir=config.bin_dir,
            num_trials=config.trials,
            timeout=config.timeout_benchmark,
            weights_dir=config.weights_dir,
            update_weights=config.update_weights,
            progress=config.progress
        )
    else:
        config.progress.info("Mode: Standard benchmarking")
        results = run_benchmark_suite(
            graphs=graphs,
            algorithms=algorithms,
            benchmarks=config.benchmarks,
            bin_dir=config.bin_dir,
            num_trials=config.trials,
            timeout=config.timeout_benchmark,
            skip_slow=config.skip_slow,
            label_maps=label_maps,
            weights_dir=config.weights_dir,
            update_weights=config.update_weights,
            progress=config.progress
        )
    
    # Save results
    filepath = _save_results(results, f"benchmark_{timestamp}.json", config.results_dir)
    config.progress.success(f"Benchmark results saved to: {filepath}")
    
    # Show summary
    if results:
        successful = [r for r in results if r.avg_time > 0]
        config.progress.stats_summary("Benchmark Statistics", {
            "Total runs": len(results),
            "Successful": len(successful),
            "Failed/Timeout": len(results) - len(successful),
            "Avg time": f"{sum(r.avg_time for r in successful) / len(successful):.4f}s" if successful else "N/A"
        })
    
    config.progress.phase_end(f"Completed {len(results)} benchmark runs")
    return results


# =============================================================================
# Phase 3: Cache Simulation
# =============================================================================

def run_cache_phase(
    graphs: List[GraphInfo],
    algorithms: List[int],
    label_maps: Dict[str, Dict[str, str]],
    config: PhaseConfig,
) -> List[CacheResult]:
    """
    Run the cache simulation phase.
    
    Args:
        graphs: List of graphs to simulate
        algorithms: List of algorithm IDs
        label_maps: Label maps from reorder phase
        config: Phase configuration
    
    Returns:
        List of cache simulation results
    """
    if config.skip_cache:
        config.progress.info("Skipping cache simulation phase (--skip-cache)")
        return []
    
    config.progress.phase_start("CACHE SIMULATION", "Running cache miss simulations")
    
    timestamp = _get_timestamp()
    
    # Check if we have variant-expanded label maps
    has_variant_maps = (label_maps and 
                       any('_' in algo_name for g in label_maps.values() for algo_name in g.keys()))
    
    if has_variant_maps and config.expand_variants:
        config.progress.info("Mode: Variant-aware cache simulation")
        # Use variant-aware cache simulation
        from .cache import run_cache_simulations_with_variants
        results = run_cache_simulations_with_variants(
            graphs=graphs,
            label_maps=label_maps,
            benchmarks=config.benchmarks,
            bin_sim_dir=config.bin_sim_dir,
            timeout=config.timeout_sim,
            skip_heavy=config.skip_heavy,
            weights_dir=config.weights_dir,
            update_weights=config.update_weights
        )
    else:
        config.progress.info("Mode: Standard cache simulation")
        results = run_cache_simulations(
            graphs=graphs,
            algorithms=algorithms,
            benchmarks=config.benchmarks,
            bin_sim_dir=config.bin_sim_dir,
            timeout=config.timeout_sim,
            skip_heavy=config.skip_heavy,
            label_maps=label_maps,
            weights_dir=config.weights_dir,
            update_weights=config.update_weights
        )
    
    # Save results
    filepath = _save_results(results, f"cache_{timestamp}.json", config.results_dir)
    config.progress.success(f"Cache results saved to: {filepath}")
    
    # Show summary
    if results:
        stats = get_cache_stats_summary(results)
        config.progress.stats_summary("Cache Statistics", stats)
    
    config.progress.phase_end(f"Completed {len(results)} cache simulations")
    return results


# =============================================================================
# Phase 4: Weight Generation
# =============================================================================

def run_weights_phase(
    benchmark_results: List[BenchmarkResult],
    cache_results: List[CacheResult],
    reorder_results: List[ReorderResult],
    config: PhaseConfig,
) -> Dict:
    """
    Run the weight generation phase.
    
    Args:
        benchmark_results: Results from benchmark phase
        cache_results: Results from cache phase
        reorder_results: Results from reorder phase
        config: Phase configuration
    
    Returns:
        Generated weights dictionary
    """
    config.progress.phase_start("WEIGHTS", "Generating perceptron weights")
    
    # Load previous results if not provided
    if not benchmark_results:
        benchmark_results = _load_latest_results("benchmark_*.json", config.results_dir, BenchmarkResult)
    if not cache_results and not config.skip_cache:
        cache_results = _load_latest_results("cache_*.json", config.results_dir, CacheResult)
    if not reorder_results:
        reorder_results = _load_latest_results("reorder_*.json", config.results_dir, ReorderResult)
    
    if not benchmark_results:
        config.progress.warning("No benchmark results available for weight generation")
        config.progress.phase_end("Skipped - no data")
        return {}
    
    # Import weight generation function
    from .weights import generate_perceptron_weights_from_results
    
    weights = generate_perceptron_weights_from_results(
        benchmark_results=benchmark_results,
        cache_results=cache_results or [],
        reorder_results=reorder_results or [],
        weights_dir=config.weights_dir
    )
    
    # Initialize default weights if needed
    if not weights:
        weights = initialize_default_weights(config.weights_dir)
    
    config.progress.phase_end(f"Generated weights for {len(weights)} graph types")
    return weights


# =============================================================================
# Phase 5: Zero Weight Update
# =============================================================================

def run_fill_weights_phase(
    benchmark_results: List[BenchmarkResult],
    cache_results: List[CacheResult],
    reorder_results: List[ReorderResult],
    config: PhaseConfig,
) -> None:
    """
    Run the zero weight filling phase.
    
    Args:
        benchmark_results: Results from benchmark phase
        cache_results: Results from cache phase
        reorder_results: Results from reorder phase
        config: Phase configuration
    """
    config.progress.phase_start("FILL WEIGHTS", "Updating zero weights")
    
    # Load results if needed
    if not benchmark_results:
        benchmark_results = _load_latest_results("benchmark_*.json", config.results_dir, BenchmarkResult)
    if not cache_results and not config.skip_cache:
        cache_results = _load_latest_results("cache_*.json", config.results_dir, CacheResult)
    if not reorder_results:
        reorder_results = _load_latest_results("reorder_*.json", config.results_dir, ReorderResult)
    
    # Import update function
    from .weights import update_zero_weights
    
    update_zero_weights(
        weights_dir=config.weights_dir,
        benchmark_results=benchmark_results,
        cache_results=cache_results or [],
        reorder_results=reorder_results or [],
        graphs_dir=config.graphs_dir,
        store_per_graph=True,  # Enable per-graph data storage for analysis
    )
    
    config.progress.phase_end("Zero weights updated")


# =============================================================================
# Phase 6: Adaptive Analysis
# =============================================================================

def run_adaptive_analysis_phase(
    graphs: List[GraphInfo],
    config: PhaseConfig,
) -> List[AdaptiveAnalysisResult]:
    """
    Run adaptive order analysis.
    
    Args:
        graphs: List of graphs to analyze
        config: Phase configuration
    
    Returns:
        List of adaptive analysis results
    """
    config.progress.phase_start("ADAPTIVE ANALYSIS", "Analyzing adaptive ordering")
    
    results = analyze_adaptive_order(
        graphs=graphs,
        bin_dir=config.bin_dir,
        output_dir=config.results_dir,
        timeout=config.timeout_reorder
    )
    
    # Show summary
    successful = [r for r in results if r.success]
    config.progress.info(f"Analysis complete: {len(successful)}/{len(results)} graphs")
    
    # Algorithm distribution
    total_distribution = {}
    for r in successful:
        for algo, count in r.algorithm_distribution.items():
            total_distribution[algo] = total_distribution.get(algo, 0) + count
    
    if total_distribution:
        config.progress.info("Algorithm distribution:")
        for algo, count in sorted(total_distribution.items(), key=lambda x: -x[1])[:5]:
            config.progress.info(f"  {algo}: {count}")
    
    config.progress.phase_end(f"Analyzed {len(results)} graphs")
    return results


# =============================================================================
# Phase 7: Adaptive vs Fixed Comparison
# =============================================================================

def run_comparison_phase(
    graphs: List[GraphInfo],
    config: PhaseConfig,
    fixed_algorithms: List[int] = None,
) -> List[Dict]:
    """
    Compare adaptive ordering vs fixed algorithms.
    
    Args:
        graphs: List of graphs to compare
        config: Phase configuration
        fixed_algorithms: Algorithm IDs to compare against
    
    Returns:
        Comparison results
    """
    config.progress.phase_start("COMPARISON", "Comparing adaptive vs fixed algorithms")
    
    fixed_algorithms = fixed_algorithms or [1, 2, 4, 7, 15]
    
    results = compare_adaptive_vs_fixed(
        graphs=graphs,
        bin_dir=config.bin_dir,
        benchmarks=config.benchmarks,
        fixed_algorithms=fixed_algorithms,
        output_dir=config.results_dir,
        num_trials=config.trials,
        timeout=config.timeout_benchmark
    )
    
    config.progress.phase_end(f"Compared {len(results)} configurations")
    return results


# =============================================================================
# Phase 8: Brute Force Validation
# =============================================================================

def run_brute_force_phase(
    graphs: List[GraphInfo],
    config: PhaseConfig,
    benchmark: str = "pr",
) -> List[Dict]:
    """
    Run brute-force validation of all algorithms.
    
    Args:
        graphs: List of graphs to validate
        config: Phase configuration
        benchmark: Benchmark to use for validation
    
    Returns:
        Brute force validation results
    """
    config.progress.phase_start("BRUTE FORCE", f"Validating all algorithms on {benchmark}")
    
    results = run_subcommunity_brute_force(
        graphs=graphs,
        bin_dir=config.bin_dir,
        bin_sim_dir=config.bin_sim_dir,
        output_dir=config.results_dir,
        benchmark=benchmark,
        timeout=config.timeout_benchmark,
        timeout_sim=config.timeout_sim,
        num_trials=config.trials
    )
    
    config.progress.phase_end(f"Validated {len(results)} graphs")
    return results


# =============================================================================
# Phase 9: Iterative Training
# =============================================================================

def run_training_phase(
    graphs: List[GraphInfo],
    config: PhaseConfig,
    weights_file: str = None,
) -> TrainingResult:
    """
    Run iterative training of adaptive weights.
    
    Args:
        graphs: List of training graphs
        config: Phase configuration
        weights_file: Path to weights file
    
    Returns:
        Training result
    """
    config.progress.phase_start("TRAINING", "Training adaptive weights")
    
    weights_file = weights_file or os.path.join(config.weights_dir, "perceptron_weights.json")
    
    result = train_adaptive_weights_iterative(
        graphs=graphs,
        bin_dir=config.bin_dir,
        weights_file=weights_file,
        benchmarks=config.benchmarks,
        target_accuracy=config.target_accuracy,
        max_iterations=config.max_iterations,
        learning_rate=config.learning_rate,
        output_dir=config.results_dir,
        timeout=config.timeout_benchmark,
        num_trials=config.trials,
        progress=config.progress
    )
    
    config.progress.phase_end(f"Training complete: {result.final_accuracy:.2%} accuracy")
    return result


# =============================================================================
# Phase 10: Large-Scale Training
# =============================================================================

def run_large_scale_training_phase(
    graphs: List[GraphInfo],
    config: PhaseConfig,
    weights_file: str = None,
) -> TrainingResult:
    """
    Run large-scale batched training.
    
    Args:
        graphs: List of training graphs
        config: Phase configuration
        weights_file: Path to weights file
    
    Returns:
        Training result
    """
    config.progress.phase_start("LARGE-SCALE TRAINING", "Batched training on large graphs")
    
    weights_file = weights_file or os.path.join(config.weights_dir, "perceptron_weights.json")
    
    result = train_adaptive_weights_large_scale(
        graphs=graphs,
        bin_dir=config.bin_dir,
        weights_file=weights_file,
        benchmarks=config.benchmarks,
        target_accuracy=config.target_accuracy,
        max_iterations=config.max_iterations,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        output_dir=config.results_dir,
        timeout=config.timeout_benchmark,
        progress=config.progress
    )
    
    config.progress.phase_end(f"Large-scale training complete")
    return result


# =============================================================================
# Full Pipeline
# =============================================================================

def run_full_pipeline(
    graphs: List[GraphInfo],
    algorithms: List[int],
    config: PhaseConfig,
    phases: List[str] = None,
) -> Dict[str, Any]:
    """
    Run the full experiment pipeline.
    
    Args:
        graphs: List of graphs to process
        algorithms: List of algorithm IDs
        config: Phase configuration
        phases: List of phases to run (default: all)
    
    Returns:
        Dictionary with all results
    """
    phases = phases or ['reorder', 'benchmark', 'cache', 'weights']
    results = {}
    label_maps = {}
    
    # Try to load existing label maps
    existing_maps = load_label_maps_index(config.results_dir)
    if existing_maps:
        label_maps = existing_maps
        config.progress.info(f"Loaded existing label maps for {len(label_maps)} graphs")
    
    # Phase 1: Reordering
    if 'reorder' in phases:
        reorder_results, label_maps = run_reorder_phase(graphs, algorithms, config, label_maps)
        results['reorder'] = reorder_results
    
    # Phase 2: Benchmarking
    if 'benchmark' in phases:
        benchmark_results = run_benchmark_phase(graphs, algorithms, label_maps, config)
        results['benchmark'] = benchmark_results
    
    # Phase 3: Cache Simulation
    if 'cache' in phases:
        cache_results = run_cache_phase(graphs, algorithms, label_maps, config)
        results['cache'] = cache_results
    
    # Phase 4: Weight Generation
    if 'weights' in phases:
        weights = run_weights_phase(
            results.get('benchmark', []),
            results.get('cache', []),
            results.get('reorder', []),
            config
        )
        results['weights'] = weights
    
    # Phase 5: Fill Weights
    if 'fill_weights' in phases:
        run_fill_weights_phase(
            results.get('benchmark', []),
            results.get('cache', []),
            results.get('reorder', []),
            config
        )
    
    # Phase 6: Adaptive Analysis
    if 'adaptive' in phases:
        adaptive_results = run_adaptive_analysis_phase(graphs, config)
        results['adaptive'] = adaptive_results
    
    # Phase 7: Comparison
    if 'comparison' in phases:
        comparison_results = run_comparison_phase(graphs, config)
        results['comparison'] = comparison_results
    
    # Phase 8: Brute Force
    if 'brute_force' in phases:
        bf_results = run_brute_force_phase(graphs, config)
        results['brute_force'] = bf_results
    
    # Phase 9: Training
    if 'training' in phases:
        training_result = run_training_phase(graphs, config)
        results['training'] = training_result
    
    # Phase 10: Large-Scale Training
    if 'large_training' in phases:
        large_training_result = run_large_scale_training_phase(graphs, config)
        results['large_training'] = large_training_result
    
    return results


# =============================================================================
# Convenience Functions for Custom Scripts
# =============================================================================

def quick_benchmark(
    graph_path: str,
    algorithms: List[int] = None,
    benchmarks: List[str] = None,
) -> List[BenchmarkResult]:
    """
    Quick benchmark a single graph.
    
    Args:
        graph_path: Path to graph file
        algorithms: Algorithm IDs (default: common algorithms)
        benchmarks: Benchmarks to run (default: pr, bfs, cc)
    
    Returns:
        Benchmark results
    """
    from .utils import get_graph_dimensions
    
    algorithms = algorithms or [0, 1, 2, 4, 7, 15]
    benchmarks = benchmarks or ['pr', 'bfs', 'cc']
    
    # Create graph info
    nodes, edges = get_graph_dimensions(graph_path)
    name = os.path.basename(os.path.dirname(graph_path))
    graph = GraphInfo(name=name, path=graph_path, nodes=nodes, edges=edges)
    
    config = PhaseConfig(benchmarks=benchmarks)
    
    # Run reorder + benchmark
    reorder_results, label_maps = run_reorder_phase([graph], algorithms, config)
    benchmark_results = run_benchmark_phase([graph], algorithms, label_maps, config)
    
    return benchmark_results


def compare_algorithms(
    graphs: List[GraphInfo],
    algorithm_a: int,
    algorithm_b: int,
    benchmarks: List[str] = None,
) -> Dict[str, Dict]:
    """
    Compare two algorithms across multiple graphs.
    
    Args:
        graphs: List of graphs
        algorithm_a: First algorithm ID
        algorithm_b: Second algorithm ID
        benchmarks: Benchmarks to use
    
    Returns:
        Comparison results by graph
    """
    algorithms = [algorithm_a, algorithm_b]
    benchmarks = benchmarks or ['pr', 'bfs', 'cc']
    
    config = PhaseConfig(benchmarks=benchmarks)
    
    reorder_results, label_maps = run_reorder_phase(graphs, algorithms, config)
    benchmark_results = run_benchmark_phase(graphs, algorithms, label_maps, config)
    
    # Organize results for comparison
    comparison = {}
    for r in benchmark_results:
        if r.graph not in comparison:
            comparison[r.graph] = {}
        if r.benchmark not in comparison[r.graph]:
            comparison[r.graph][r.benchmark] = {}
        comparison[r.graph][r.benchmark][r.algorithm_name] = r.avg_time
    
    return comparison


# =============================================================================
# Main (for testing)
# =============================================================================

def main():
    """Test the phases module."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test phase orchestration")
    parser.add_argument('--graphs-dir', default='results/graphs', help='Graphs directory')
    parser.add_argument('--phase', choices=['reorder', 'benchmark', 'cache', 'weights', 'all'],
                        default='all', help='Phase to run')
    args = parser.parse_args()
    
    print("Phase orchestration module loaded successfully")
    print(f"Available phases: reorder, benchmark, cache, weights, adaptive, comparison, brute_force, training")


if __name__ == "__main__":
    main()
