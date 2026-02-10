"""
GraphBrew Type Definitions
==========================

Central module for all data class definitions used across the GraphBrew library.
This ensures consistent type definitions and avoids circular imports.

Type Categories
---------------
The types are organized into logical groups:

**Graph Information:**
    - GraphInfo: Metadata about a graph (nodes, edges, degree stats)
    - DownloadableGraph: Info about a graph available for download

**Reordering:**
    - ReorderResult: Output from running a reordering algorithm
    - AlgorithmConfig: Configuration for an algorithm

**Benchmarking:**
    - BenchmarkResult: Timing results from benchmark runs
    - CacheResult: Cache simulation statistics

**Adaptive Ordering:**
    - SubcommunityInfo: Info about a detected subcommunity
    - AdaptiveOrderResult: Output from adaptive ordering
    - AdaptiveAnalysisResult: Detailed analysis with benchmarks
    - AdaptiveComparisonResult: Comparison of adaptive vs fixed

**Training:**
    - PerceptronWeight: Learned weights for an algorithm
    - TrainingIterationResult: Stats from one training iteration
    - TrainingResult: Complete training output

**Configuration:**
    - PipelineConfig: Settings for experiment pipeline

Usage Examples
--------------
Import individual types:

    >>> from lib.graph_types import GraphInfo, BenchmarkResult
    >>> info = GraphInfo(name="test", path="/path/to/graph", nodes=100, edges=500)
    >>> print(info.avg_degree)  # Auto-calculated
    10.0

Import all types:

    >>> from lib.graph_types import *
    >>> result = BenchmarkResult(
    ...     graph="test",
    ...     algorithm_id=8,
    ...     algorithm_name="gorder",
    ...     benchmark="pr",
    ...     avg_time=0.5
    ... )

Type Checking
-------------
All types are dataclasses with type hints for IDE support:

    >>> from lib.graph_types import ReorderResult
    >>> def process_result(result: ReorderResult) -> float:
    ...     return result.time_seconds

See Also
--------
- lib/phases.py: Uses these types for phase I/O
- lib/benchmark.py: Produces BenchmarkResult
- lib/reorder.py: Produces ReorderResult
- lib/cache.py: Produces CacheResult
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# =============================================================================
# Graph Information
# =============================================================================

@dataclass
class GraphInfo:
    """
    Information about a graph dataset.
    
    This class holds metadata about a graph including structural properties
    and topology metrics. It's used throughout the library to pass graph
    information between modules.
    
    Attributes:
        name: Short name of the graph (e.g., "email-Enron")
        path: Full path to the graph file
        size_mb: File size in megabytes
        is_symmetric: Whether graph is undirected (symmetric adjacency)
        nodes: Number of vertices in the graph
        edges: Number of edges (or directed arcs if asymmetric)
        avg_degree: Average vertex degree (auto-calculated if not provided)
        density: Graph density (edges / possible edges)
        degree_variance: Variance in vertex degrees
        hub_concentration: Measure of how concentrated edges are in hubs
        graph_type: Detected graph type ("social", "web", "road", etc.)
    
    Example:
        >>> info = GraphInfo(
        ...     name="email-Enron",
        ...     path="graphs/email-Enron/email-Enron.mtx",
        ...     nodes=36692,
        ...     edges=183831,
        ...     is_symmetric=True
        ... )
        >>> print(f"Avg degree: {info.avg_degree:.1f}")
        Avg degree: 10.0
    
    Notes:
        - avg_degree is auto-calculated in __post_init__ if nodes/edges given
        - For symmetric graphs, avg_degree = 2 * edges / nodes
        - For directed graphs, avg_degree = edges / nodes
    """
    name: str
    path: str
    size_mb: float = 0.0
    is_symmetric: bool = True
    nodes: int = 0
    edges: int = 0
    
    # Optional extended properties (populated by topology analysis)
    avg_degree: float = 0.0
    density: float = 0.0
    degree_variance: float = 0.0
    hub_concentration: float = 0.0
    
    # Graph type (detected from structure)
    graph_type: str = "generic"
    
    def __post_init__(self):
        """Auto-calculate avg_degree if not provided."""
        if self.nodes > 0 and self.edges > 0 and self.avg_degree == 0:
            self.avg_degree = 2 * self.edges / self.nodes if self.is_symmetric else self.edges / self.nodes


# =============================================================================
# Reordering Results
# =============================================================================

@dataclass
class ReorderResult:
    """
    Result from a reordering operation.
    
    Captures the output of running a graph reordering algorithm, including
    timing information and the path to the generated mapping file.
    
    Attributes:
        graph: Name of the graph that was reordered
        algorithm_id: Numeric ID of the algorithm (0-15)
        algorithm_name: Human-readable algorithm name
        success: Whether reordering completed successfully
        time_seconds: Time taken to reorder
        error: Error message if reordering failed
        mapping_file: Path to the .lo (layout) file with vertex mapping
        nodes: Number of nodes in the graph
        edges: Number of edges in the graph
        variant: For Leiden, the variant used (mod/cpm)
        resolution: For Leiden, the resolution parameter
        passes: For Leiden, number of optimization passes
    
    Example:
        >>> result = ReorderResult(
        ...     graph="email-Enron",
        ...     algorithm_id=8,
        ...     algorithm_name="gorder",
        ...     success=True,
        ...     time_seconds=2.5,
        ...     mapping_file="results/email-Enron/gorder.lo"
        ... )
        >>> if result.success:
        ...     print(f"Reordered in {result.time_seconds:.2f}s")
    """
    graph: str
    algorithm_id: int
    algorithm_name: str
    success: bool
    time_seconds: float = 0.0
    error: str = ""
    mapping_file: str = ""  # Path to .lo file with vertex mapping
    nodes: int = 0
    edges: int = 0
    
    # Leiden-specific parameters
    variant: str = ""       # "mod" or "cpm"
    resolution: float = 1.0
    passes: int = 10


@dataclass
class AlgorithmConfig:
    """
    Configuration for a reordering algorithm.
    
    Used to specify which algorithm to run and with what parameters.
    
    Attributes:
        algorithm_id: Numeric ID (see lib/__init__.py for mapping)
        name: Algorithm name
        variant: Optional variant (e.g., "mod" or "cpm" for Leiden)
        resolution: Leiden resolution parameter
        passes: Leiden optimization passes
        
    Example:
        >>> config = AlgorithmConfig(
        ...     algorithm_id=12,
        ...     name="GraphBrewOrder",
        ...     variant="community",
        ...     resolution=1.0
        ... )
        >>> print(config.full_name)
        lorder_cpm
    """
    algorithm_id: int
    name: str
    variant: str = ""
    resolution: float = 1.0
    passes: int = 10
    
    @property
    def full_name(self) -> str:
        """Get full algorithm name including variant suffix."""
        if self.variant:
            return f"{self.name}_{self.variant}"
        return self.name


# =============================================================================
# Benchmark Results
# =============================================================================

@dataclass
class BenchmarkResult:
    """
    Result from running a benchmark.
    
    Contains timing data from executing a graph algorithm benchmark on
    a specific graph with a specific vertex ordering.
    
    Attributes:
        graph: Name of the graph
        algorithm_id: ID of the reordering algorithm used
        algorithm_name: Name of the reordering algorithm
        benchmark: Benchmark name (pr, bfs, cc, sssp, bc, tc)
        avg_time: Average execution time across trials
        trial_times: List of individual trial times
        speedup: Speedup vs original ordering (baseline=1.0)
        nodes: Number of nodes
        edges: Number of edges
        success: Whether benchmark completed successfully
        error: Error message if failed
        memory_gb: Peak memory usage in GB
        iterations: For iterative algorithms, number of iterations
        
    Example:
        >>> result = BenchmarkResult(
        ...     graph="email-Enron",
        ...     algorithm_id=8,
        ...     algorithm_name="gorder",
        ...     benchmark="pr",
        ...     avg_time=0.45,
        ...     trial_times=[0.44, 0.45, 0.46],
        ...     speedup=1.35
        ... )
        >>> print(f"PageRank: {result.avg_time:.3f}s ({result.speedup:.2f}x faster)")
        
    Notes:
        - trial_time property is alias for avg_time (backward compat)
        - stddev can be computed from trial_times if needed
    """
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
    
    @property
    def trial_time(self) -> float:
        """Alias for avg_time (backward compatibility)."""
        return self.avg_time
    
    @property
    def stddev(self) -> float:
        """Calculate standard deviation from trial times."""
        if len(self.trial_times) < 2:
            return 0.0
        mean = sum(self.trial_times) / len(self.trial_times)
        variance = sum((t - mean) ** 2 for t in self.trial_times) / (len(self.trial_times) - 1)
        return variance ** 0.5
    
    @property 
    def graph_name(self) -> str:
        """Alias for graph (compatibility with some code)."""
        return self.graph


# =============================================================================
# Cache Simulation Results
# =============================================================================

@dataclass
class CacheResult:
    """
    Result from cache simulation.
    
    Contains cache miss rates from running a benchmark through the
    cache simulator. These metrics help understand memory access patterns.
    
    Attributes:
        graph: Name of the graph
        algorithm_id: ID of reordering algorithm
        algorithm_name: Name of reordering algorithm
        benchmark: Benchmark that was simulated
        l1_miss_rate: L1 cache miss rate (0.0 to 1.0)
        l2_miss_rate: L2 cache miss rate (0.0 to 1.0)
        l3_miss_rate: L3 cache miss rate (0.0 to 1.0)
        total_accesses: Total memory accesses
        success: Whether simulation completed
        error: Error message if failed
        l1_misses: Raw count of L1 misses
        l2_misses: Raw count of L2 misses
        l3_misses: Raw count of L3 misses
        
    Example:
        >>> result = CacheResult(
        ...     graph="email-Enron",
        ...     algorithm_id=8,
        ...     algorithm_name="gorder",
        ...     benchmark="pr",
        ...     l1_miss_rate=0.15,
        ...     l2_miss_rate=0.08,
        ...     l3_miss_rate=0.02
        ... )
        >>> print(f"L1 misses: {result.l1_miss_rate*100:.1f}%")
        L1 misses: 15.0%
        
    Notes:
        - Lower miss rates = better cache locality
        - l3_miss_rate most correlates with execution time
        - Good reorderings typically reduce miss rates significantly
    """
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
    
    # Raw counts (for detailed analysis)
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
    graphs_dir: str = "results/graphs"
    results_dir: str = "results"
    weights_dir: str = "scripts/weights/active"
    
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
