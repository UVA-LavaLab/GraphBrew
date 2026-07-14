"""
GraphBrew Type Definitions
==========================

Central type for graph metadata used across the GraphBrew library.

Canonical type definitions live in the modules that produce them:
    - ``ReorderResult``, ``AlgorithmConfig`` → :pymod:`lib.pipeline.reorder`
    - ``BenchmarkResult``                   → :pymod:`lib.core.utils`
    - ``CacheResult``                       → :pymod:`lib.pipeline.cache`
    - ``TrainingResult``, ``TrainingIterationResult`` → :pymod:`lib.ml.training`
    - ``PerceptronWeight``                  → :pymod:`lib.ml.weights`
    - ``SubcommunityInfo``, ``AdaptiveOrderResult``,
      ``AdaptiveComparisonResult``, ``SubcommunityBruteForceResult``,
      ``GraphBruteForceAnalysis``           → :pymod:`lib.analysis.adaptive`
    - ``DownloadableGraph``                 → :pymod:`lib.pipeline.download`

This module defines only **GraphInfo**, which is imported by several
modules that cannot import from one another without creating circular
dependencies.
"""

from dataclasses import dataclass


# =============================================================================
# Graph Information
# =============================================================================

@dataclass
class GraphInfo:
    """
    Information about a graph dataset.

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
        ...     is_symmetric=True,
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
            self.avg_degree = (
                2 * self.edges / self.nodes
                if self.is_symmetric
                else self.edges / self.nodes
            )


__all__ = [
    "GraphInfo",
]
