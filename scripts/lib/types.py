"""
GraphBrew Type Definitions

Central module for all data class definitions used across the GraphBrew library.
This ensures consistent type definitions and avoids duplication.

Usage:
    from lib.types import GraphInfo, ReorderResult, BenchmarkResult, CacheResult
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# =============================================================================
# Graph Information
# =============================================================================

@dataclass
class GraphInfo:
    """Information about a graph dataset."""
    name: str
    path: str
    size_mb: float = 0.0
    is_symmetric: bool = True
    nodes: int = 0
    edges: int = 0
    
    # Optional extended properties
    avg_degree: float = 0.0
    density: float = 0.0
    degree_variance: float = 0.0
    hub_concentration: float = 0.0
    
    # Graph type (from detection)
    graph_type: str = "generic"
    
    def __post_init__(self):
        if self.nodes > 0 and self.edges > 0 and self.avg_degree == 0:
            self.avg_degree = 2 * self.edges / self.nodes if self.is_symmetric else self.edges / self.nodes


# =============================================================================
# Reordering Results
# =============================================================================

@dataclass
class ReorderResult:
    """Result from a reordering operation."""
    graph: str
    algorithm_id: int
    algorithm_name: str
    success: bool
    time_seconds: float = 0.0
    error: str = ""
    mapping_file: str = ""  # Path to .lo file
    nodes: int = 0
    edges: int = 0
    
    # Leiden-specific
    variant: str = ""
    resolution: float = 1.0
    passes: int = 10


@dataclass
class AlgorithmConfig:
    """Configuration for a reordering algorithm."""
    algorithm_id: int
    name: str
    variant: str = ""
    resolution: float = 1.0
    passes: int = 10
    
    @property
    def full_name(self) -> str:
        """Get full algorithm name including variant."""
        if self.variant:
            return f"{self.name}_{self.variant}"
        return self.name


# =============================================================================
# Benchmark Results
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result from running a benchmark."""
    graph: str
    algorithm_id: int
    algorithm_name: str
    benchmark: str
    avg_time: float = 0.0
    trial_times: List[float] = field(default_factory=list)
    speedup: float = 1.0
    nodes: int = 0
    edges: int = 0
    success: bool = True
    error: str = ""
    
    # Additional metrics
    memory_gb: float = 0.0
    iterations: int = 0
    
    # For compatibility with older code
    @property
    def trial_time(self) -> float:
        """Alias for avg_time for backward compatibility."""
        return self.avg_time


# =============================================================================
# Cache Simulation Results
# =============================================================================

@dataclass
class CacheResult:
    """Result from cache simulation."""
    graph: str
    algorithm_id: int
    algorithm_name: str
    benchmark: str
    l1_miss_rate: float = 0.0
    l2_miss_rate: float = 0.0
    l3_miss_rate: float = 0.0
    total_accesses: int = 0
    success: bool = True
    error: str = ""
    
    # Raw counts
    l1_misses: int = 0
    l2_misses: int = 0
    l3_misses: int = 0


# =============================================================================
# Adaptive Order Results
# =============================================================================

@dataclass
class SubcommunityInfo:
    """Information about a subcommunity in adaptive ordering."""
    id: int
    size: int
    algorithm: str
    algorithm_id: int
    nodes_start: int = 0
    nodes_end: int = 0


@dataclass
class AdaptiveOrderResult:
    """Result from running adaptive order."""
    graph: str
    success: bool
    total_time: float = 0.0
    num_subcommunities: int = 0
    subcommunities: List[SubcommunityInfo] = field(default_factory=list)
    algorithm_distribution: Dict[str, int] = field(default_factory=dict)
    error: str = ""


@dataclass
class AdaptiveAnalysisResult:
    """Detailed analysis result from adaptive ordering."""
    graph: str
    success: bool
    total_time: float = 0.0
    num_subcommunities: int = 0
    subcommunities: List[SubcommunityInfo] = field(default_factory=list)
    algorithm_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Benchmark performance
    benchmark_times: Dict[str, float] = field(default_factory=dict)
    speedups: Dict[str, float] = field(default_factory=dict)
    
    error: str = ""


@dataclass
class AdaptiveComparisonResult:
    """Result comparing adaptive vs fixed algorithm."""
    graph: str
    benchmark: str
    adaptive_time: float
    fixed_times: Dict[str, float] = field(default_factory=dict)
    adaptive_better: bool = False
    best_fixed_algo: str = ""
    improvement: float = 0.0


# =============================================================================
# Brute Force Analysis
# =============================================================================

@dataclass
class SubcommunityBruteForceResult:
    """Result from brute-force algorithm selection for a subcommunity."""
    subcommunity_id: int
    size: int
    best_algorithm: str
    best_time: float
    predicted_algorithm: str
    prediction_correct: bool
    all_times: Dict[str, float] = field(default_factory=dict)


@dataclass
class GraphBruteForceAnalysis:
    """Complete brute-force analysis for a graph."""
    graph: str
    success: bool
    num_subcommunities: int = 0
    subcommunity_results: List[SubcommunityBruteForceResult] = field(default_factory=list)
    overall_accuracy: float = 0.0
    total_time: float = 0.0
    error: str = ""


# =============================================================================
# Training Results
# =============================================================================

@dataclass
class TrainingIterationResult:
    """Result from a single training iteration."""
    iteration: int
    accuracy: float
    correct: int
    total: int
    adjustments: int
    graphs_processed: List[str] = field(default_factory=list)


@dataclass
class TrainingResult:
    """Result from training adaptive weights."""
    success: bool
    final_accuracy: float = 0.0
    iterations: int = 0
    iteration_results: List[TrainingIterationResult] = field(default_factory=list)
    converged: bool = False
    weights_file: str = ""
    error: str = ""


# =============================================================================
# Weight Types
# =============================================================================

@dataclass
class PerceptronWeight:
    """Perceptron weights for an algorithm."""
    algorithm_id: int
    algorithm_name: str
    bias: float = 0.0
    w_modularity: float = 0.0
    w_log_nodes: float = 0.0
    w_log_edges: float = 0.0
    w_density: float = 0.0
    w_avg_degree: float = 0.0
    w_degree_variance: float = 0.0
    w_hub_concentration: float = 0.0
    w_clustering_coeff: float = 0.0
    w_avg_path_length: float = 0.0
    w_diameter: float = 0.0
    w_community_count: float = 0.0
    w_reorder_time: float = 0.0
    
    # Performance tracking
    wins: int = 0
    total_comparisons: int = 0
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        return self.wins / self.total_comparisons if self.total_comparisons > 0 else 0.0


# =============================================================================
# Pipeline Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the full experiment pipeline."""
    # Directories
    bin_dir: str = "bench/bin"
    bin_sim_dir: str = "bench/bin_sim"
    graphs_dir: str = "graphs"
    results_dir: str = "results"
    weights_dir: str = "scripts/weights"
    
    # Timeouts
    timeout_reorder: int = 300
    timeout_benchmark: int = 120
    timeout_sim: int = 600
    
    # Benchmark settings
    benchmarks: List[str] = field(default_factory=lambda: ['pr', 'bfs', 'cc'])
    trials: int = 3
    
    # Flags
    skip_slow: bool = False
    skip_cache: bool = False
    skip_heavy: bool = False
    force_reorder: bool = False
    expand_variants: bool = False
    update_weights: bool = True


# =============================================================================
# Downloadable Graph Info
# =============================================================================

@dataclass
class DownloadableGraph:
    """Information about a downloadable graph from SuiteSparse."""
    name: str
    group: str
    id: int
    rows: int
    cols: int
    nnz: int
    kind: str = ""
    symmetric: bool = True
    url: str = ""
    size_mb: float = 0.0
    
    @property
    def full_name(self) -> str:
        return f"{self.group}/{self.name}"


# =============================================================================
# Export all types
# =============================================================================

__all__ = [
    'GraphInfo',
    'ReorderResult',
    'AlgorithmConfig',
    'BenchmarkResult',
    'CacheResult',
    'SubcommunityInfo',
    'AdaptiveOrderResult',
    'AdaptiveAnalysisResult',
    'AdaptiveComparisonResult',
    'SubcommunityBruteForceResult',
    'GraphBruteForceAnalysis',
    'TrainingIterationResult',
    'TrainingResult',
    'PerceptronWeight',
    'PipelineConfig',
    'DownloadableGraph',
]
