#!/usr/bin/env python3
"""
Training functions for GraphBrew adaptive weight learning.

This module provides:
- train_adaptive_weights_iterative: Iterative training with feedback loop
- train_adaptive_weights_large_scale: Large-scale batch training
- TrainingResult/TrainingIterationResult: Result data classes
- Weight initialization and update utilities

Example usage:
    from scripts.lib.training import train_adaptive_weights_iterative
    
    result = train_adaptive_weights_iterative(
        graphs=graphs,
        bin_dir="bench/bin",
        bin_sim_dir="bench/bin_sim",
        output_dir="results",
        weights_file="results/weights/type_0/weights.json",
        target_accuracy=80.0
    )
    
    print(f"Final accuracy: {result.final_accuracy_time_pct:.1f}%")
"""

import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Set

from .utils import Logger, WEIGHTS_DIR, TIMEOUT_SIM, TIMEOUT_BENCHMARK, get_all_algorithm_variant_names
from .analysis import (
    run_subcommunity_brute_force,
)
from .benchmark import load_reorder_time_for_algo
from .weights import (
    assign_graph_type,
    update_type_weights_incremental,
    load_type_registry,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingIterationResult:
    """Result from one iteration of training."""
    iteration: int
    accuracy_time_pct: float
    accuracy_cache_pct: float
    accuracy_top3_pct: float
    avg_time_ratio: float
    avg_cache_ratio: float
    graphs_tested: int
    weights_updated: int


@dataclass
class TrainingResult:
    """Final result from iterative training."""
    final_accuracy_time_pct: float
    final_accuracy_cache_pct: float
    final_accuracy_top3_pct: float
    iterations_run: int
    target_accuracy: float
    target_reached: bool
    iteration_history: List[TrainingIterationResult] = field(default_factory=list)
    best_weights_iteration: int = 0
    best_weights_file: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Weight Initialization
# ─────────────────────────────────────────────────────────────────────────────

def initialize_enhanced_weights(weights_file: str, algorithms: List[str] = None) -> Dict:
    """
    Initialize or upgrade weights file with enhanced feature support.
    
    Creates a new weights file with all extended features, or upgrades
    an existing weights file to include new features.
    
    Args:
        weights_file: Path to weights JSON file
        algorithms: List of algorithm names (uses defaults if None)
        
    Returns:
        Initialized weights dictionary
    """
    default_algorithms = get_all_algorithm_variant_names()
    
    if algorithms is None:
        algorithms = default_algorithms
    
    def make_default_weights(algo_name: str) -> Dict:
        """Create default weights for an algorithm."""
        biases = {
            "GraphBrewOrder_leiden": 0.6, "LeidenOrder": 0.55,
            "GORDER": 0.52, "RABBITORDER_csr": 0.5, "CORDER": 0.48,
            "HUBCLUSTERDBG": 0.45, "HUBSORT": 0.42, "RCM_default": 0.4,
        }
        
        return {
            "bias": biases.get(algo_name, 0.5),
            "w_modularity": 0.0,
            "w_log_nodes": 0.0,
            "w_log_edges": 0.0,
            "w_density": 0.0,
            "w_avg_degree": 0.0,
            "w_degree_variance": 0.0,
            "w_hub_concentration": 0.0,
            "w_clustering_coeff": 0.0,
            "w_avg_path_length": 0.0,
            "w_diameter": 0.0,
            "w_community_count": 0.0,
            "w_packing_factor": 0.0,
            "w_forward_edge_fraction": 0.0,
            "w_working_set_ratio": 0.0,
            "w_dv_x_hub": 0.0,
            "w_mod_x_logn": 0.0,
            "w_pf_x_wsr": 0.0,
            "w_fef_convergence": 0.0,
            "w_reorder_time": 0.0,
            "cache_l1_impact": 0.0,
            "cache_l2_impact": 0.0,
            "cache_l3_impact": 0.0,
            "cache_dram_penalty": 0.0,
            "benchmark_weights": {
                "pr": 1.0, "pr_spmv": 1.0, "bfs": 1.0, "cc": 1.0,
                "cc_sv": 1.0, "sssp": 1.0, "bc": 1.0, "tc": 1.0,
            }
        }
    
    # Try to load existing weights
    weights = {}
    if os.path.exists(weights_file):
        try:
            with open(weights_file) as f:
                weights = json.load(f)
        except (json.JSONDecodeError, IOError):
            weights = {}
    
    # Ensure all algorithms have weights
    for algo in algorithms:
        if algo not in weights or algo.startswith('_'):
            continue
        
        # Upgrade existing weights with new fields
        existing = weights[algo]
        default = make_default_weights(algo)
        
        for key, value in default.items():
            if key not in existing:
                existing[key] = value
        
        weights[algo] = existing
    
    # Add missing algorithms
    for algo in algorithms:
        if algo not in weights:
            weights[algo] = make_default_weights(algo)
    
    # Save initialized weights
    os.makedirs(os.path.dirname(weights_file) or '.', exist_ok=True)
    with open(weights_file, 'w') as f:
        json.dump(weights, f, indent=2)
    
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Training Functions
# ─────────────────────────────────────────────────────────────────────────────

def train_adaptive_weights_iterative(
    graphs: List,  # GraphInfo objects
    bin_dir: str,
    bin_sim_dir: str,
    output_dir: str,
    weights_file: str,
    benchmark: str = "pr",
    target_accuracy: float = 80.0,
    max_iterations: int = 10,
    timeout: int = TIMEOUT_BENCHMARK,
    timeout_sim: int = TIMEOUT_SIM,
    num_trials: int = 3,
    learning_rate: float = 0.1,
    algorithms: List[int] = None,
    weights_dir: str = None,
    logger: Logger = None
) -> TrainingResult:
    """
    Iterative training loop for adaptive algorithm selection weights.
    
    This function uses the type-based weight system:
    1. Runs brute-force evaluation to measure current accuracy
    2. Identifies where adaptive picks wrong (what should have been chosen)
    3. Detects the graph type for each graph/subcommunity
    4. Updates type-specific weights using update_type_weights_incremental()
    5. Repeats until target accuracy is reached or max iterations
    
    Args:
        graphs: List of GraphInfo objects to test
        bin_dir: Path to benchmark binaries
        bin_sim_dir: Path to cache simulation binaries
        output_dir: Where to save results
        weights_file: Path to intermediate weights file
        benchmark: Which benchmark to use (pr, bfs, etc.)
        target_accuracy: Target accuracy percentage (0-100)
        max_iterations: Maximum training iterations
        timeout: Timeout for reorder operations
        timeout_sim: Timeout for cache simulations
        num_trials: Number of benchmark trials per test
        learning_rate: How much to adjust weights (0.0-1.0)
        algorithms: List of algorithm IDs to test
        weights_dir: Directory for type-based weight files
        logger: Optional logger instance
        
    Returns:
        TrainingResult with iteration history and final accuracy
    """
    log = logger.info if logger else print
    
    if weights_dir is None:
        weights_dir = str(WEIGHTS_DIR)
    
    log(f"Iterative Weight Training (Target: {target_accuracy}%)")
    
    if algorithms is None:
        algorithms = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 15]
    
    # Ensure directories exist
    training_dir = os.path.join(output_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    
    # Load type registry
    log("Loading type registry for type-based weight system...")
    load_type_registry(weights_dir)
    
    # Initialize weights file
    log("Initializing enhanced weight structure...")
    initialize_enhanced_weights(weights_file)
    
    # Track types updated per iteration
    types_updated_per_iter: Dict[int, Set[str]] = {}
    
    # Initialize result
    result = TrainingResult(
        final_accuracy_time_pct=0.0,
        final_accuracy_cache_pct=0.0,
        final_accuracy_top3_pct=0.0,
        iterations_run=0,
        target_accuracy=target_accuracy,
        target_reached=False
    )
    
    best_accuracy = 0.0
    avg_accuracy_time = 0.0
    avg_accuracy_cache = 0.0
    avg_top3_time = 0.0
    
    for iteration in range(1, max_iterations + 1):
        log(f"\n{'='*60}")
        log(f"TRAINING ITERATION {iteration}/{max_iterations}")
        log(f"{'='*60}")
        
        # Step 1: Run brute-force analysis
        log("\n--- Step 1: Measure Current Accuracy ---")
        bf_results = run_subcommunity_brute_force(
            graphs=graphs,
            bin_dir=bin_dir,
            bin_sim_dir=bin_sim_dir,
            output_dir=training_dir,
            benchmark=benchmark,
            timeout=timeout,
            timeout_sim=timeout_sim,
            num_trials=num_trials,
            logger=logger
        )
        
        if not bf_results:
            log("No brute-force results, stopping training")
            break
        
        successful = [r for r in bf_results if r.success]
        if not successful:
            log("No successful brute-force results, stopping training")
            break
        
        # Calculate overall accuracy
        avg_accuracy_time = sum(r.adaptive_correct_time_pct for r in successful) / len(successful)
        avg_accuracy_cache = sum(r.adaptive_correct_cache_pct for r in successful) / len(successful)
        avg_top3_time = sum(r.adaptive_top3_time_pct for r in successful) / len(successful)
        
        valid_time_ratios = [r.avg_time_ratio for r in successful if r.avg_time_ratio > 0]
        valid_cache_ratios = [r.avg_cache_ratio for r in successful if r.avg_cache_ratio > 0]
        avg_time_ratio = sum(valid_time_ratios) / len(valid_time_ratios) if valid_time_ratios else 0
        avg_cache_ratio = sum(valid_cache_ratios) / len(valid_cache_ratios) if valid_cache_ratios else 0
        
        log(f"\nIteration {iteration} Accuracy:")
        log(f"  Adaptive correct (time): {avg_accuracy_time:.1f}%")
        log(f"  Adaptive correct (cache): {avg_accuracy_cache:.1f}%")
        log(f"  Adaptive in top 3: {avg_top3_time:.1f}%")
        
        # Record iteration result
        iter_result = TrainingIterationResult(
            iteration=iteration,
            accuracy_time_pct=avg_accuracy_time,
            accuracy_cache_pct=avg_accuracy_cache,
            accuracy_top3_pct=avg_top3_time,
            avg_time_ratio=avg_time_ratio,
            avg_cache_ratio=avg_cache_ratio,
            graphs_tested=len(successful),
            weights_updated=0
        )
        
        # Track best iteration
        if avg_accuracy_time > best_accuracy:
            best_accuracy = avg_accuracy_time
            best_weights_file = os.path.join(training_dir, f"best_weights_iter{iteration}.json")
            with open(weights_file) as f:
                best_weights = json.load(f)
            with open(best_weights_file, 'w') as f:
                json.dump(best_weights, f, indent=2)
            result.best_weights_iteration = iteration
            result.best_weights_file = best_weights_file
            log(f"  New best accuracy! Saved weights to {best_weights_file}")
        
        # Check if target reached
        if avg_accuracy_time >= target_accuracy:
            log(f"\n🎯 TARGET ACCURACY REACHED: {avg_accuracy_time:.1f}% >= {target_accuracy}%")
            result.target_reached = True
            result.iteration_history.append(iter_result)
            break
        
        # Step 2: Analyze errors and adjust weights
        log("\n--- Step 2: Analyze Errors and Adjust Weights ---")
        
        with open(weights_file) as f:
            weights = json.load(f)
        
        types_updated_this_iter: Set[str] = set()
        weights_updated = 0
        
        for bf_result in successful:
            for sc_result in bf_result.subcommunity_results:
                if sc_result.adaptive_is_best_time:
                    continue
                
                # Get features
                features = {
                    'modularity': getattr(sc_result, 'modularity', 0.5),
                    'density': sc_result.density,
                    'degree_variance': sc_result.degree_variance,
                    'hub_concentration': sc_result.hub_concentration,
                    'nodes': sc_result.nodes,
                    'edges': sc_result.edges,
                    'log_nodes': math.log10(sc_result.nodes) if sc_result.nodes > 0 else 0,
                    'log_edges': math.log10(sc_result.edges) if sc_result.edges > 0 else 0,
                    'avg_degree': (2 * sc_result.edges / sc_result.nodes) if sc_result.nodes > 0 else 0,
                    'clustering_coefficient': getattr(sc_result, 'clustering_coefficient', 0.0),
                    'avg_path_length': getattr(sc_result, 'avg_path_length', 0.0),
                    'diameter_estimate': getattr(sc_result, 'diameter_estimate', 0.0),
                    'diameter': getattr(sc_result, 'diameter_estimate', 0.0),
                    'community_count': getattr(sc_result, 'community_count', 1),
                }
                
                # Detect graph type
                graph_type = assign_graph_type(features, weights_dir, create_if_outlier=False)
                
                adaptive_algo = sc_result.adaptive_algorithm
                correct_algo = sc_result.best_time_algorithm
                
                if not correct_algo or correct_algo == adaptive_algo:
                    continue
                
                # Load reorder times from .time files (written during
                # pregeneration).  This lets w_reorder_time learn that
                # fast-to-reorder algorithms are preferable when they
                # give comparable benchmark performance.
                graph_name = bf_result.graph
                correct_reorder_time = load_reorder_time_for_algo(
                    graph_name, correct_algo)
                adaptive_reorder_time = load_reorder_time_for_algo(
                    graph_name, adaptive_algo)
                
                # Update type-based weights
                correct_speedup = 1.2  # Positive reinforcement
                wrong_speedup = 0.8   # Negative reinforcement
                
                update_type_weights_incremental(
                    type_name=graph_type,
                    algorithm=correct_algo,
                    benchmark=benchmark.lower(),
                    speedup=correct_speedup,
                    features=features,
                    cache_stats=None,
                    reorder_time=correct_reorder_time,
                    weights_dir=weights_dir,
                    learning_rate=learning_rate
                )
                types_updated_this_iter.add(graph_type)
                weights_updated += 1
                
                update_type_weights_incremental(
                    type_name=graph_type,
                    algorithm=adaptive_algo,
                    benchmark=benchmark.lower(),
                    speedup=wrong_speedup,
                    features=features,
                    cache_stats=None,
                    reorder_time=adaptive_reorder_time,
                    weights_dir=weights_dir,
                    learning_rate=learning_rate
                )
                # Legacy weights file update removed — type-based perceptron
                # (update_type_weights_incremental) is the sole training path.
        
        # Add training metadata
        weights['_training_metadata'] = {
            'last_iteration': iteration,
            'accuracy_time_pct': avg_accuracy_time,
            'accuracy_cache_pct': avg_accuracy_cache,
            'timestamp': datetime.now().isoformat(),
            'learning_rate': learning_rate,
            'graphs_tested': len(successful),
            'types_updated': list(types_updated_this_iter),
            'weight_system': 'type-based'
        }
        
        # Save updated weights
        with open(weights_file, 'w') as f:
            json.dump(weights, f, indent=2)
        
        types_updated_per_iter[iteration] = types_updated_this_iter
        
        iter_result.weights_updated = weights_updated
        result.iteration_history.append(iter_result)
        
        log(f"  Weights updated for {weights_updated} algorithm adjustments")
        log(f"  Types updated: {len(types_updated_this_iter)}")
        
        # Save iteration weights
        iter_weights_file = os.path.join(training_dir, f"weights_iter{iteration}.json")
        with open(iter_weights_file, 'w') as f:
            json.dump(weights, f, indent=2)
        
        result.iterations_run = iteration
    
    # Finalize results
    result.final_accuracy_time_pct = avg_accuracy_time
    result.final_accuracy_cache_pct = avg_accuracy_cache
    result.final_accuracy_top3_pct = avg_top3_time
    
    # Save training summary
    summary_file = os.path.join(training_dir, "training_summary.json")
    summary = {
        'target_accuracy': target_accuracy,
        'target_reached': result.target_reached,
        'iterations_run': result.iterations_run,
        'final_accuracy_time_pct': result.final_accuracy_time_pct,
        'final_accuracy_cache_pct': result.final_accuracy_cache_pct,
        'final_accuracy_top3_pct': result.final_accuracy_top3_pct,
        'best_iteration': result.best_weights_iteration,
        'best_weights_file': result.best_weights_file,
        'iteration_history': [asdict(h) for h in result.iteration_history]
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    log(f"\n{'='*60}")
    log("TRAINING COMPLETE (TYPE-BASED WEIGHTS)")
    log(f"{'='*60}")
    log(f"Iterations run: {result.iterations_run}")
    log(f"Final accuracy (time): {result.final_accuracy_time_pct:.1f}%")
    log(f"Target reached: {'YES' if result.target_reached else 'NO'}")
    log(f"Training summary saved to: {summary_file}")
    
    # Restore best weights if target not reached
    if not result.target_reached and result.best_weights_file and os.path.exists(result.best_weights_file):
        log(f"\nRestoring best weights from iteration {result.best_weights_iteration}")
        with open(result.best_weights_file) as f:
            best_weights = json.load(f)
        with open(weights_file, 'w') as f:
            json.dump(best_weights, f, indent=2)
    
    return result


def train_adaptive_weights_large_scale(
    graphs: List,  # GraphInfo objects
    bin_dir: str,
    bin_sim_dir: str,
    output_dir: str,
    weights_file: str,
    benchmarks: List[str] = None,
    target_accuracy: float = 80.0,
    max_iterations: int = 10,
    batch_size: int = 8,
    timeout: int = TIMEOUT_BENCHMARK,
    timeout_sim: int = TIMEOUT_SIM,
    num_trials: int = 2,
    learning_rate: float = 0.15,
    algorithms: List[int] = None,
    weights_dir: str = None,
    logger: Logger = None
) -> TrainingResult:
    """
    Large-scale training with batching and multi-benchmark support.
    
    This extends train_adaptive_weights_iterative with:
    - Batch processing for organized training
    - Multi-benchmark training (pr, bfs, cc, etc.)
    - Progressive learning rate decay
    - Cross-validation between batches
    
    NOTE: All operations run sequentially for accurate performance measurements.
    
    Args:
        graphs: List of GraphInfo objects to train on
        bin_dir: Path to benchmark binaries
        bin_sim_dir: Path to cache simulation binaries
        output_dir: Where to save results
        weights_file: Path to weights file
        benchmarks: List of benchmarks to train on (default: ['pr', 'bfs', 'cc'])
        target_accuracy: Target accuracy percentage
        max_iterations: Maximum training iterations
        batch_size: Number of graphs per batch
        timeout: Timeout for benchmark runs
        timeout_sim: Timeout for cache simulations
        num_trials: Number of trials per benchmark
        learning_rate: Initial learning rate
        algorithms: List of algorithm IDs to test
        weights_dir: Directory for type-based weight files
        logger: Optional logger instance
        
    Returns:
        TrainingResult with final accuracy and iteration history
    """
    log = logger.info if logger else print
    
    if weights_dir is None:
        weights_dir = str(WEIGHTS_DIR)
    
    log(f"Large-Scale Adaptive Training ({len(graphs)} graphs)")
    log("Note: All algorithms run sequentially for accurate measurement")
    
    if benchmarks is None:
        # Fast-training subset (full SSOT: BENCHMARKS in utils.py has 8)
        benchmarks = ['pr', 'bfs', 'cc']
    
    if algorithms is None:
        algorithms = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 15]
    
    # Setup
    training_dir = os.path.join(output_dir, f"large_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(training_dir, exist_ok=True)
    
    # Initialize weights
    initialize_enhanced_weights(weights_file)
    
    result = TrainingResult(
        final_accuracy_time_pct=0.0,
        final_accuracy_cache_pct=0.0,
        final_accuracy_top3_pct=0.0,
        iterations_run=0,
        target_accuracy=target_accuracy,
        target_reached=False
    )
    
    best_accuracy = 0.0
    
    # Split graphs into batches
    shuffled_graphs = graphs.copy()
    random.shuffle(shuffled_graphs)
    batches = [shuffled_graphs[i:i+batch_size] for i in range(0, len(shuffled_graphs), batch_size)]
    
    log(f"Created {len(batches)} batches of ~{batch_size} graphs each")
    log(f"Training on benchmarks: {benchmarks}")
    
    for iteration in range(1, max_iterations + 1):
        # Learning rate decay
        current_lr = learning_rate * (0.9 ** (iteration - 1))
        
        log(f"\n{'='*60}")
        log(f"ITERATION {iteration}/{max_iterations} (LR: {current_lr:.4f})")
        log(f"{'='*60}")
        
        iteration_accuracy = []
        
        for benchmark in benchmarks:
            log(f"\n--- Benchmark: {benchmark.upper()} ---")
            
            for batch_idx, batch in enumerate(batches):
                log(f"\nBatch {batch_idx+1}/{len(batches)}: {[g.name for g in batch]}")
                
                # Run training iteration on this batch
                iter_result = train_adaptive_weights_iterative(
                    graphs=batch,
                    bin_dir=bin_dir,
                    bin_sim_dir=bin_sim_dir,
                    output_dir=training_dir,
                    weights_file=weights_file,
                    benchmark=benchmark,
                    target_accuracy=target_accuracy,
                    max_iterations=1,
                    timeout=timeout,
                    timeout_sim=timeout_sim,
                    num_trials=num_trials,
                    learning_rate=current_lr,
                    algorithms=algorithms,
                    weights_dir=weights_dir,
                    logger=logger
                )
                
                if iter_result.iteration_history:
                    iteration_accuracy.append(iter_result.iteration_history[0].accuracy_time_pct)
        
        # Calculate average accuracy
        if iteration_accuracy:
            avg_accuracy = sum(iteration_accuracy) / len(iteration_accuracy)
            log(f"\nIteration {iteration} average accuracy: {avg_accuracy:.1f}%")
            
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                result.best_weights_iteration = iteration
                best_weights_file = os.path.join(training_dir, f"best_weights_iter{iteration}.json")
                with open(weights_file) as f:
                    best_weights = json.load(f)
                with open(best_weights_file, 'w') as f:
                    json.dump(best_weights, f, indent=2)
                result.best_weights_file = best_weights_file
            
            result.final_accuracy_time_pct = avg_accuracy
            
            if avg_accuracy >= target_accuracy:
                log(f"\n🎯 TARGET REACHED: {avg_accuracy:.1f}% >= {target_accuracy}%")
                result.target_reached = True
                break
        
        result.iterations_run = iteration
    
    # Summary
    log(f"\n{'='*60}")
    log("LARGE-SCALE TRAINING COMPLETE")
    log(f"{'='*60}")
    log(f"Total iterations: {result.iterations_run}")
    log(f"Best accuracy: {best_accuracy:.1f}%")
    log(f"Target reached: {'YES' if result.target_reached else 'NO'}")
    
    return result


# _update_legacy_weights removed — the ad-hoc gradient only adjusted 3 of 17+
# weights and didn't match the perceptron's forward pass. All training now goes
# through update_type_weights_incremental() which uses the correct gradient.


__all__ = [
    'TrainingIterationResult',
    'TrainingResult',
    'initialize_enhanced_weights',
    'train_adaptive_weights_iterative',
    'train_adaptive_weights_large_scale',
]
