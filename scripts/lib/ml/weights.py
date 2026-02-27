#!/usr/bin/env python3
"""
Weight management utilities for GraphBrew.

Handles type-based perceptron weights for adaptive algorithm selection.
Implements auto-clustering type system for graph classification.

Standalone usage:
    python -m scripts.lib.ml.weights --list-types
    python -m scripts.lib.ml.weights --show-type type_0
    python -m scripts.lib.ml.weights --best-algo type_0 --benchmark pr

Library usage:
    from scripts.lib.ml.weights import (
        assign_graph_type, load_type_weights, update_type_weights_incremental,
        get_best_algorithm_for_type
    )
"""

import os
import json
import math
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from ..core.utils import (
    ACTIVE_WEIGHTS_DIR,
    EXPERIMENT_BENCHMARKS,
    Logger, get_timestamp,
    WEIGHT_PATH_LENGTH_NORMALIZATION, WEIGHT_REORDER_TIME_NORMALIZATION,
    WEIGHT_AVG_DEGREE_DEFAULT,
    get_all_algorithm_variant_names,
    is_chained_ordering_name,
    weights_registry_path, weights_type_path, weights_bench_path,
)

# Initialize logger
log = Logger()

# =============================================================================
# Constants
# =============================================================================

# Default weights directory (where C++ reads from)
DEFAULT_WEIGHTS_DIR = str(ACTIVE_WEIGHTS_DIR)

# Auto-clustering configuration
CLUSTER_DISTANCE_THRESHOLD = 0.15  # Max normalized distance to join existing cluster

# Dead feature keys — features that are ALWAYS 0 in training data.
# C++ now computes avg_path_length, diameter, community_count at runtime
# via ComputeExtendedFeatures(), so they are no longer dead.
_DEAD_WEIGHT_KEYS: set = set()

# =============================================================================
# Algorithm Groups (Mode 4+3)
# =============================================================================
# Group algorithms by underlying strategy for hierarchical prediction:
#   1. Predict group (3-class → higher accuracy)
#   2. Within predicted group, rank by perceptron score and try top-K
#
# GROUP_COMMUNITY: Community-detection based orderings
# GROUP_BANDWIDTH: Structural / bandwidth-reduction orderings
# GROUP_CACHE: Cache-aware / degree-based heuristic orderings
ALGO_GROUPS = {
    'COMMUNITY': frozenset({
        'LeidenOrder',
        'GraphBrewOrder_leiden',
        'GraphBrewOrder_hubcluster',
        'GraphBrewOrder_rabbit',
    }),
    'BANDWIDTH': frozenset({
        'GORDER',
        'CORDER',
        'RCM_default',
        'RCM_bnf',
    }),
    'CACHE': frozenset({
        'RABBITORDER_csr',
        'RABBITORDER_boost',
        'HUBSORT',
        'HUBCLUSTER',
        'SORT',
        'DBG',
        'HUBSORTDBG',
        'HUBCLUSTERDBG',
    }),
}

# Inverse mapping: algorithm → group name
ALGO_TO_GROUP = {}
for _gname, _algos in ALGO_GROUPS.items():
    for _a in _algos:
        ALGO_TO_GROUP[_a] = _gname

# =============================================================================
# Perceptron Candidate Algorithm Configuration
# =============================================================================
# PERCEPTRON_EXCLUDED: algorithms the perceptron should NEVER select.
# ORIGINAL/RANDOM are graph states used for baseline / speedup measurement,
# not reordering algorithms.  MAP and AdaptiveOrder are meta-algorithms.
# Chained orderings are filtered separately (they can't be selected at C++
# runtime since the perceptron picks a single algorithm ID).
#
# To exclude an algorithm from perceptron selection, add it here.
# To include it again, remove it from this set.
PERCEPTRON_EXCLUDED: frozenset = frozenset({
    'ORIGINAL',
    'RANDOM',
})


def get_perceptron_candidates(extra_exclude: set = None) -> frozenset:
    """Return the set of algorithms the perceptron may select from.

    Computed dynamically from the algorithm registry minus:
      - PERCEPTRON_EXCLUDED (baseline states)
      - Chained orderings (can't be selected at C++ runtime)
      - Any additional names in *extra_exclude*

    Returns:
        frozenset of canonical algorithm names eligible for training /
        inference.
    """
    base = set(get_all_algorithm_variant_names())
    # Remove chained orderings (already excluded by get_all_algorithm_variant_names
    # for the base set, but guard against future changes)
    base = {n for n in base if not is_chained_ordering_name(n)}
    # Remove excluded baselines
    base -= PERCEPTRON_EXCLUDED
    if extra_exclude:
        base -= set(extra_exclude)
    return frozenset(base)


# =============================================================================
# Data Classes — Single Source of Truth (SSO) for Perceptron Weights
# =============================================================================
#
# This is the CANONICAL Python implementation of the perceptron scoring model.
# The C++ mirror lives in bench/include/graphbrew/reorder/reorder_types.h
# (struct PerceptronWeights, scoreBase(), scoreBaseNormalized(), score()).
#
# ANY change to weight fields or scoring logic MUST be mirrored in BOTH files.
# No other Python module should re-implement scoring — all must delegate here.
#
# Weight Conceptual Reference (Paper Motivation)
# ───────────────────────────────────────────────
#
# ┌─────────────────────────┬─────────────────────────────────────────────────┐
# │ Weight                  │ Conceptual Meaning                              │
# ├─────────────────────────┼─────────────────────────────────────────────────┤
# │ bias                    │ Algorithm-level prior: base preference before   │
# │                         │ any graph features are considered. Captures the │
# │                         │ algorithm's overall effectiveness.              │
# ├─────────────────────────┼─────────────────────────────────────────────────┤
# │ w_modularity            │ Community structure quality (Leiden/Louvain Q).  │
# │                         │ High → GraphBrewOrder & community-aware algos   │
# │                         │ can exploit strong partition structure.          │
# ├─────────────────────────┼─────────────────────────────────────────────────┤
# │ w_log_nodes             │ Graph scale (log₁₀(N+1)). Some algorithms have │
# │                         │ scale-dependent advantages — e.g., RCM benefits │
# │                         │ from smaller graphs, hub-sort from larger ones. │
# ├─────────────────────────┼─────────────────────────────────────────────────┤
# │ w_log_edges             │ Edge count scale (log₁₀(E+1)). Complements     │
# │                         │ node count; sparse vs dense at equal N.         │
# ├─────────────────────────┼─────────────────────────────────────────────────┤
# │ w_density               │ Edge density (E / (N*(N-1)/2)). Dense graphs    │
# │                         │ have different locality patterns.               │
# ├─────────────────────────┼─────────────────────────────────────────────────┤
# │ w_avg_degree            │ Average degree / 100. High avg-degree graphs    │
# │                         │ benefit from cache-line packing optimizations.  │
# ├─────────────────────────┼─────────────────────────────────────────────────┤
# │ w_degree_variance       │ Degree distribution skewness. Power-law graphs  │
# │                         │ with high DV benefit from hub-aware orderings.  │
# ├─────────────────────────┼─────────────────────────────────────────────────┤
# │ w_hub_concentration     │ Fraction of edges touching top-1% nodes.       │
# │                         │ High → HubSort/HubCluster effectively isolate  │
# │                         │ hot vertices into cache-resident blocks.        │
# ├─────────────────────────┼─────────────────────────────────────────────────┤
# │ w_clustering_coeff      │ Local clustering coefficient (sampled).        │
# │                         │ High → triangles are dense → community-aware   │
# │                         │ orderings improve spatial locality.             │
# ├─────────────────────────┼─────────────────────────────────────────────────┤
# │ w_avg_path_length       │ Average shortest-path length / 10. Long paths  │
# │                         │ → road-network-like → RCM/BFS-order excel.     │
# ├─────────────────────────┼─────────────────────────────────────────────────┤
# │ w_diameter              │ Graph diameter estimate / 50. Large diameter    │
# │                         │ favors bandwidth-reducing orderings (RCM).      │
# ├─────────────────────────┼─────────────────────────────────────────────────┤
# │ w_community_count       │ log₁₀(community_count + 1). More communities   │
# │                         │ → finer partition → GraphBrewOrder can exploit  │
# │                         │ per-community local ordering.                   │
# ├─────────────────────────┼─────────────────────────────────────────────────┤
# │ w_packing_factor        │ Cache-line utilization ratio (IISWC 2018).     │
# │                         │ Measures how well neighbors pack into cache    │
# │                         │ lines. High PF → ordering preserves locality.  │
# ├─────────────────────────┼─────────────────────────────────────────────────┤
# │ w_forward_edge_fraction │ Fraction of edges pointing to higher-numbered  │
# │                         │ vertices (GoGraph). High FEF → ordering aligns │
# │                         │ with data-flow direction.                      │
# ├─────────────────────────┼─────────────────────────────────────────────────┤
# │ w_working_set_ratio     │ log₂(WSR + 1). Working-set / cache size ratio. │
# │                         │ Measures cache pressure — high WSR means the   │
# │                         │ active working set exceeds available cache.     │
# ├─────────────────────────┼─────────────────────────────────────────────────┤
# │ w_dv_x_hub              │ DV × HC cross-term. Captures hub-dominated     │
# │                         │ power-law structure where both skewness AND    │
# │                         │ hub concentration are high simultaneously.      │
# ├─────────────────────────┼─────────────────────────────────────────────────┤
# │ w_mod_x_logn            │ Modularity × log(N). Community structure value │
# │                         │ at scale — strong communities matter MORE in   │
# │                         │ large graphs.                                  │
# ├─────────────────────────┼─────────────────────────────────────────────────┤
# │ w_pf_x_wsr              │ Packing Factor × log₂(WSR+1). Captures the    │
# │                         │ interaction between cache utilization quality  │
# │                         │ and cache pressure — locality gains matter     │
# │                         │ more when cache is under pressure.             │
# ├─────────────────────────┼─────────────────────────────────────────────────┤
# │ w_fef_convergence       │ FEF convergence bonus (PR/PR_SPMV/SSSP only). │
# │                         │ Iterative algorithms converge faster when      │
# │                         │ edges align with Gauss-Seidel update order.    │
# ├─────────────────────────┼─────────────────────────────────────────────────┤
# │ cache_l1/l2/l3_impact   │ Per-cache-level impact (×0.5, ×0.3, ×0.2).    │
# │                         │ From cache simulation; captures how well the   │
# │                         │ ordering fits each cache tier.                 │
# ├─────────────────────────┼─────────────────────────────────────────────────┤
# │ cache_dram_penalty      │ DRAM access penalty (×1.0). High → ordering   │
# │                         │ causes excessive main-memory traffic.           │
# ├─────────────────────────┼─────────────────────────────────────────────────┤
# │ w_reorder_time          │ Reordering cost (seconds). Penalizes slow     │
# │                         │ algorithms — amortization trade-off.           │
# ├─────────────────────────┼─────────────────────────────────────────────────┤
# │ benchmark_weights       │ Per-benchmark multiplier {pr: 1.0, bfs: 1.2,  │
# │                         │ ...}. Trained to specialize algorithm scoring  │
# │                         │ per workload type.                             │
# └─────────────────────────┴─────────────────────────────────────────────────┘
#
# Feature Vector (17 elements, matches C++ scoreBase() exactly):
#   [modularity, dv, hc, log₁₀(N+1), log₁₀(E+1), density, avg_deg/100,
#    cc, apl/10, dia/50, log₁₀(comm+1), pf, fef, log₂(wsr+1),
#    dv×hc, mod×log(N), pf×log₂(wsr+1)]
#
# Training: Multi-class Jimenez perceptron with θ-margin, averaged weights,
# W_MAX=16 clamping, 5 restarts, 800 epochs. See compute_weights_from_results().
#
# Normalization: Mode 5 uses z-score normalization — features are centered
# (mean=0, std=1) before scoring; trained weights are in z-score space.
# See C++ scoreBaseNormalized() and Python _logo_score(norm_data=...).
# =============================================================================

@dataclass
class PerceptronWeight:
    """Single Source of Truth (SSO) perceptron weight vector for one algorithm.
    
    Each field corresponds to a learned weight for a graph feature.
    The scoring formula is:
    
        score = bias
              + Σ(wᵢ × transform(featureᵢ))       # 14 linear features
              + Σ(wⱼ × featureₐ × featureᵦ)        # 3 quadratic cross-terms
              + cache_l1 × 0.5 + cache_l2 × 0.3     # cache impact constants
              + cache_l3 × 0.2 + cache_dram          #   (from simulation)
              + w_fef_convergence × fef              # convergence bonus (PR/SSSP only)
        final = score × benchmark_multiplier
    
    This MUST stay in sync with C++ PerceptronWeights::scoreBase() and score()
    in bench/include/graphbrew/reorder/reorder_types.h.
    """
    # --- Algorithm prior ---
    bias: float = 0.0                    # Base score before any features
                                         # (matches C++ PerceptronWeights default)
    
    # --- Core topology weights (paper: community & degree structure) ---
    w_modularity: float = 0.0            # Leiden/Louvain modularity Q
    w_log_nodes: float = 0.0             # log₁₀(N+1) — graph scale
    w_log_edges: float = 0.0             # log₁₀(E+1) — edge scale
    w_density: float = 0.0               # Edge density E/(N·(N-1)/2)
    w_avg_degree: float = 0.0            # Average degree (÷100 at scoring)
    w_degree_variance: float = 0.0       # Degree distribution skewness
    w_hub_concentration: float = 0.0     # Edge fraction on top-1% hubs
    
    # --- Cache impact constants (from cache simulation) ---
    cache_l1_impact: float = 0.0         # L1 cache hit impact (×0.5)
    cache_l2_impact: float = 0.0         # L2 cache hit impact (×0.3)
    cache_l3_impact: float = 0.0         # L3 cache hit impact (×0.2)
    cache_dram_penalty: float = 0.0      # DRAM access penalty (×1.0)
    
    # --- Reordering cost ---
    w_reorder_time: float = 0.0          # Reorder time penalty (seconds)
    
    # --- Extended structural weights (paper: locality & path structure) ---
    w_clustering_coeff: float = 0.0      # Local clustering coefficient
    w_avg_path_length: float = 0.0       # Average shortest path (÷10)
    w_diameter: float = 0.0              # Diameter estimate (÷50)
    w_community_count: float = 0.0       # log₁₀(communities + 1)
    
    # --- Locality features (IISWC'18 packing factor, GoGraph FEF) ---
    w_packing_factor: float = 0.0        # Cache-line utilization ratio
    w_forward_edge_fraction: float = 0.0 # Forward edge fraction
    w_working_set_ratio: float = 0.0     # log₂(WSR + 1) cache pressure
    
    # --- Quadratic cross-terms (paper: feature interactions) ---
    w_dv_x_hub: float = 0.0             # DV × HC — hub-dominated structure
    w_mod_x_logn: float = 0.0           # Modularity × Scale
    w_pf_x_wsr: float = 0.0            # Packing × Cache pressure
    
    # --- Convergence bonus (iterative benchmarks only) ---
    w_fef_convergence: float = 0.0       # FEF bonus for PR/PR_SPMV/SSSP
    
    # --- Per-benchmark multiplier ---
    benchmark_weights: Dict[str, float] = field(default_factory=lambda: {
        b: 1.0 for b in EXPERIMENT_BENCHMARKS
    })
    
    # --- Training metadata (used by amortization mode, not in scoring) ---
    avg_speedup: float = 1.0             # Average speedup vs ORIGINAL
    avg_reorder_time: float = 0.0        # Average reorder time (seconds)
    
    # --- Auto-generated metadata (not used in scoring) ---
    _metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        # Backward compat: also store in _metadata for legacy consumers
        d.setdefault('_metadata', {})['avg_speedup'] = self.avg_speedup
        d['_metadata']['avg_reorder_time'] = self.avg_reorder_time
        return d
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'PerceptronWeight':
        """Create PerceptronWeight from a dict (e.g., JSON weight entry)."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in d.items() if k in valid_fields}
        # Backward compat: pull avg_speedup/avg_reorder_time from _metadata
        meta = d.get('_metadata', {})
        if 'avg_speedup' not in kwargs and 'avg_speedup' in meta:
            kwargs['avg_speedup'] = meta['avg_speedup']
        if 'avg_reorder_time' not in kwargs and 'avg_reorder_time' in meta:
            kwargs['avg_reorder_time'] = meta['avg_reorder_time']
        return cls(**kwargs)
    
    def compute_score(self, features: Dict, benchmark: str = 'pr') -> float:
        """Compute perceptron score — SINGLE SOURCE OF TRUTH for Python scoring.
        
        This is the CANONICAL scoring function. All other Python code MUST
        delegate here. Mirrors C++ scoreBase() + score() in reorder_types.h.
        
        Score formula:
            1. Base score = bias + Σ(wᵢ × transformᵢ(featureᵢ))
            2. + quadratic cross-terms (dv×hc, mod×logN, pf×log₂(wsr+1))
            3. + cache impact (l1×0.5 + l2×0.3 + l3×0.2 + dram)
            4. + convergence bonus (w_fef_convergence × fef, PR/SSSP only)
            5. × benchmark_multiplier
        
        Feature transforms (matching C++ exactly):
            - log_nodes  = log₁₀(N + 1)
            - log_edges  = log₁₀(E + 1)
            - avg_degree → ÷ 100
            - avg_path_length → ÷ 10 (WEIGHT_PATH_LENGTH_NORMALIZATION)
            - diameter → ÷ 50
            - community_count → log₁₀(count + 1)
            - working_set_ratio → log₂(wsr + 1)
        
        Args:
            features: Dict with graph properties (nodes, edges, modularity, etc.)
            benchmark: Benchmark name for convergence bonus + multiplier
        
        Returns:
            Perceptron score (higher = algorithm more suitable for this graph)
        """
        # Derived scale features
        log_nodes = math.log10(features.get('nodes', 1) + 1)
        log_edges = math.log10(features.get('edges', 1) + 1)
        
        # ── Layer 1: Linear features ──────────────────────────────────
        score = self.bias
        score += self.w_modularity * features.get('modularity', 0.0)
        score += self.w_log_nodes * log_nodes
        score += self.w_log_edges * log_edges
        score += self.w_density * features.get('density', 0.0)
        score += self.w_avg_degree * features.get('avg_degree', 0.0) / 100.0
        score += self.w_degree_variance * features.get('degree_variance', 0.0)
        score += self.w_hub_concentration * features.get('hub_concentration', 0.0)
        # Accept both key names for clustering (experiment uses 'clustering_coefficient',
        # emulator/C++ use 'clustering_coeff')
        clustering = features.get('clustering_coefficient', features.get('clustering_coeff', 0.0))
        score += self.w_clustering_coeff * clustering
        score += self.w_avg_path_length * features.get('avg_path_length', 0.0) / WEIGHT_PATH_LENGTH_NORMALIZATION
        score += self.w_diameter * features.get('diameter', features.get('diameter_estimate', 0.0)) / 50.0
        score += self.w_community_count * math.log10(features.get('community_count', 1) + 1)
        score += self.w_reorder_time * features.get('reorder_time', 0.0)
        score += self.w_packing_factor * features.get('packing_factor', 0.0)
        score += self.w_forward_edge_fraction * features.get('forward_edge_fraction', 0.0)
        score += self.w_working_set_ratio * math.log2(features.get('working_set_ratio', 0.0) + 1.0)
        
        # ── Layer 2: Quadratic interaction terms ──────────────────────
        log_wsr = math.log2(features.get('working_set_ratio', 0.0) + 1.0)
        log_n = math.log10(features.get('nodes', 1) + 1)
        score += self.w_dv_x_hub * features.get('degree_variance', 0.0) * features.get('hub_concentration', 0.0)
        score += self.w_mod_x_logn * features.get('modularity', 0.0) * log_n
        score += self.w_pf_x_wsr * features.get('packing_factor', 0.0) * log_wsr
        
        # ── Layer 3: Cache impact constants ───────────────────────────
        # Matches C++ scoreBase():
        #   s += cache_l1_impact * 0.5 + cache_l2_impact * 0.3
        #      + cache_l3_impact * 0.2 + cache_dram_penalty
        score += self.cache_l1_impact * 0.5
        score += self.cache_l2_impact * 0.3
        score += self.cache_l3_impact * 0.2
        score += self.cache_dram_penalty
        
        # ── Layer 4: Convergence bonus (iterative benchmarks only) ────
        # Matches C++ score(): bench == BENCH_PR || BENCH_PR_SPMV || BENCH_SSSP
        # Iterative algorithms converge faster when edges align with
        # Gauss-Seidel update order (high forward-edge fraction).
        bench_name = benchmark if benchmark else features.get('benchmark', '')
        if bench_name in ('pr', 'pr_spmv', 'sssp'):
            score += self.w_fef_convergence * features.get('forward_edge_fraction', 0.0)
        
        # ── Layer 5: Benchmark-specific multiplier ────────────────────
        bench_mult = self.benchmark_weights.get(bench_name.lower(), 1.0)
        return score * bench_mult

    def compute_score_normalized(
        self,
        features: Dict,
        benchmark: str = 'pr',
        norm_mean: List[float] = None,
        norm_std: List[float] = None,
    ) -> float:
        """Compute z-score-normalized perceptron score.

        Mirrors C++ ``scoreBaseNormalized()`` in reorder_types.h.

        The 17-element raw feature vector (in the same order as C++):
            [0]  modularity
            [1]  degree_variance
            [2]  hub_concentration
            [3]  log10(N+1)
            [4]  log10(E+1)
            [5]  density
            [6]  avg_degree / 100
            [7]  clustering_coeff
            [8]  avg_path_length / 10
            [9]  diameter / 50
            [10] log10(community_count + 1)
            [11] packing_factor
            [12] forward_edge_fraction
            [13] log2(wsr + 1)
            [14] degree_variance * hub_concentration  (quadratic)
            [15] modularity * log10(N+1)               (quadratic)
            [16] packing_factor * log2(wsr+1)          (quadratic)

        Each raw[i] is z-normalized: ``z = (raw - mean[i]) / std[i]``
        (skipped when ``std < 1e-12``).

        Cache and reorder_time terms are **outside** the z-score loop
        (matching C++).

        Args:
            features: Dict with graph properties.
            benchmark: Benchmark name for convergence bonus + multiplier.
            norm_mean: 17-element list of per-feature means.
            norm_std: 17-element list of per-feature standard deviations.

        Returns:
            Perceptron score (higher = algorithm more suitable).
        """
        if norm_mean is None or norm_std is None:
            # Without normalization data, fall back to the regular scorer.
            return self.compute_score(features, benchmark)

        log_n = math.log10(features.get('nodes', 1) + 1)
        log_e = math.log10(features.get('edges', 1) + 1)
        clustering = features.get(
            'clustering_coefficient', features.get('clustering_coeff', 0.0))
        pf = features.get('packing_factor', 0.0)
        fef = features.get('forward_edge_fraction', 0.0)
        log_wsr = math.log2(features.get('working_set_ratio', 0.0) + 1.0)
        dv = features.get('degree_variance', 0.0)
        hc = features.get('hub_concentration', 0.0)
        mod = features.get('modularity', 0.0)

        raw = [
            mod,
            dv,
            hc,
            log_n,
            log_e,
            features.get('density', 0.0),
            features.get('avg_degree', 0.0) / 100.0,
            clustering,
            features.get('avg_path_length', 0.0) / WEIGHT_PATH_LENGTH_NORMALIZATION,
            features.get('diameter', features.get('diameter_estimate', 0.0)) / 50.0,
            math.log10(features.get('community_count', 1) + 1),
            pf,
            fef,
            log_wsr,
            dv * hc,
            mod * log_n,
            pf * log_wsr,
        ]

        weights_17 = [
            self.w_modularity,
            self.w_degree_variance,
            self.w_hub_concentration,
            self.w_log_nodes,
            self.w_log_edges,
            self.w_density,
            self.w_avg_degree,
            self.w_clustering_coeff,
            self.w_avg_path_length,
            self.w_diameter,
            self.w_community_count,
            self.w_packing_factor,
            self.w_forward_edge_fraction,
            self.w_working_set_ratio,
            self.w_dv_x_hub,
            self.w_mod_x_logn,
            self.w_pf_x_wsr,
        ]

        score = self.bias
        for i in range(17):
            if i < len(norm_std) and norm_std[i] >= 1e-12:
                z = (raw[i] - norm_mean[i]) / norm_std[i]
                score += weights_17[i] * z

        # Cache and reorder terms are outside the z-score loop (matching C++)
        score += self.cache_l1_impact * 0.5
        score += self.cache_l2_impact * 0.3
        score += self.cache_l3_impact * 0.2
        score += self.cache_dram_penalty
        score += self.w_reorder_time * features.get('reorder_time', 0.0)

        # Convergence bonus
        bench_name = benchmark if benchmark else features.get('benchmark', '')
        if bench_name in ('pr', 'pr_spmv', 'sssp'):
            score += self.w_fef_convergence * fef

        bench_mult = self.benchmark_weights.get(bench_name.lower(), 1.0)
        return score * bench_mult


# =============================================================================
# Global Type Registry
# =============================================================================

_type_registry: Dict[str, Dict] = {}
_type_registry_file: str = weights_registry_path(DEFAULT_WEIGHTS_DIR)


def _get_next_type_name() -> str:
    """Generate next type name (type_0, type_1, etc.)."""
    if not _type_registry:
        return "type_0"
    
    existing = [k for k in _type_registry.keys() if k.startswith("type_")]
    if not existing:
        return "type_0"
    
    numbers = []
    for k in existing:
        try:
            num = int(k.replace("type_", ""))
            numbers.append(num)
        except ValueError:
            suffix = k.replace("type_", "")
            if len(suffix) == 1 and suffix.isalpha():
                numbers.append(ord(suffix.lower()) - ord('a'))
    
    if not numbers:
        return "type_0"
    return f"type_{max(numbers) + 1}"


def _normalize_features(features: Dict) -> List[float]:
    """Normalize features to [0,1] range for distance calculation."""
    ranges = {
        'modularity': (0, 1),
        'degree_variance': (0, 5),
        'hub_concentration': (0, 1),
        'avg_degree': (0, 100),
        'clustering_coefficient': (0, 1),
        'log_nodes': (3, 10),
        'log_edges': (3, 12),
    }
    
    normalized = []
    log_nodes = math.log10(features.get('nodes', 1000) + 1)
    log_edges = math.log10(features.get('edges', 1000) + 1)
    
    for key, (lo, hi) in ranges.items():
        if key == 'log_nodes':
            val = log_nodes
        elif key == 'log_edges':
            val = log_edges
        else:
            val = features.get(key, (lo + hi) / 2)
        normalized.append(max(0, min(1, (val - lo) / (hi - lo) if hi > lo else 0.5)))
    
    return normalized


def _compute_distance(f1: List[float], f2: List[float]) -> float:
    """Compute Euclidean distance between normalized feature vectors."""
    if len(f1) != len(f2):
        return float('inf')
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(f1, f2)))


def _pearson_correlation(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient between two sequences."""
    if len(x) < 2:
        return 0.0
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    den_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    if den_x * den_y > 0:
        return num / (den_x * den_y)
    return 0.0


# =============================================================================
# LOGO / CV Shared Helpers  (used by cross_validate_logo[_grouped])
# =============================================================================

# Weight keys in canonical order — must match C++ scoreBaseNormalized().
_LOGO_WEIGHT_KEYS = [
    'w_modularity', 'w_degree_variance', 'w_hub_concentration',
    'w_log_nodes', 'w_log_edges', 'w_density', 'w_avg_degree',
    'w_clustering_coeff', 'w_avg_path_length', 'w_diameter',
    'w_community_count', 'w_packing_factor', 'w_forward_edge_fraction',
    'w_working_set_ratio', 'w_dv_x_hub', 'w_mod_x_logn', 'w_pf_x_wsr',
]


def _build_logo_features(props: Dict) -> Dict:
    """Build INFERENCE feature dict matching what C++ runtime computes.

    Uses ESTIMATED modularity (CC*1.5) — NOT real modularity from
    computeFastModularity — because C++ AdaptiveOrder only has the
    estimate at runtime.  Training uses real modularity for better
    signal, but LOGO/inference must simulate runtime conditions.
    """
    nodes = props.get('nodes', 1000)
    edges = props.get('edges', 5000)
    cc = props.get('clustering_coefficient', 0.0)
    avg_degree = props.get('avg_degree', 10.0)
    estimated_mod = min(0.9, cc * 1.5)
    log_n = math.log10(nodes + 1) if nodes > 0 else 0
    pf = props.get('packing_factor', 0.0)
    wsr = props.get('working_set_ratio', 0.0)
    log_wsr = math.log2(wsr + 1.0)
    return {
        'modularity': estimated_mod,
        'degree_variance': props.get('degree_variance', 1.0),
        'hub_concentration': props.get('hub_concentration', 0.3),
        'avg_degree': avg_degree,
        'log_nodes': log_n,
        'log_edges': math.log10(edges + 1) if edges > 0 else 0,
        'density': avg_degree / (nodes - 1) if nodes > 1 else 0,
        'clustering_coefficient': cc,
        'avg_path_length': props.get('avg_path_length', 0.0),
        'diameter': props.get('diameter_estimate', 0.0),
        'community_count': props.get('community_count', 0.0),
        'packing_factor': pf,
        'forward_edge_fraction': props.get('forward_edge_fraction', 0.5),
        'log_working_set_ratio': log_wsr,
        'working_set_ratio': wsr,
        # Quadratic interaction terms (use estimated modularity)
        'dv_x_hub': props.get('degree_variance', 1.0) * props.get('hub_concentration', 0.3),
        'mod_x_logn': estimated_mod * log_n,
        'pf_x_wsr': pf * log_wsr,
    }


def _make_logo_score_fv(feats: Dict) -> list:
    """Build 17-element feature vector with C++ transforms applied."""
    dv = feats.get('degree_variance', 1.0)
    hc = feats.get('hub_concentration', 0.3)
    modularity = feats.get('modularity', 0.0)
    log_nodes = feats.get('log_nodes', 5.0)
    pf = feats.get('packing_factor', 0.0)
    wsr = feats.get('working_set_ratio', 0.0)
    log_wsr = math.log2(wsr + 1.0)
    comm = feats.get('community_count', 1.0)
    return [
        modularity,
        dv,
        hc,
        log_nodes,
        feats.get('log_edges', 6.0),
        feats.get('density', 0.0),
        feats.get('avg_degree', 10.0) / 100.0,
        feats.get('clustering_coefficient', 0.0),
        feats.get('avg_path_length', 0.0) / 10.0,
        feats.get('diameter', feats.get('diameter_estimate', 0.0)) / 50.0,
        math.log10(comm + 1) if comm > 0 else 0,
        pf,
        feats.get('forward_edge_fraction', 0.5),
        log_wsr,
        dv * hc,
        modularity * log_nodes,
        pf * log_wsr,
    ]


def _logo_score(algo_data: Dict, feats: Dict, norm_data: Dict = None) -> float:
    """Score an algorithm for given features.

    Covers the 17 core linear + quadratic terms used in LOGO cross-validation.
    Intentionally omits cache constant offsets and convergence bonus —
    these are graph-independent constants and per-benchmark conditionals
    that don't affect relative ranking between algorithms within one graph.

    When *norm_data* is provided, z-normalize the feature vector before
    the dot product (matching C++ scoreBaseNormalized).
    """
    fv = _make_logo_score_fv(feats)
    s = algo_data.get('bias', 0.5)
    if norm_data is not None:
        means = norm_data['feat_means']
        stds = norm_data['feat_stds']
        for i in range(len(fv)):
            if i < len(stds) and stds[i] > 1e-12:
                z = (fv[i] - means[i]) / stds[i]
                s += algo_data.get(_LOGO_WEIGHT_KEYS[i], 0) * z
    else:
        for i in range(len(fv)):
            s += algo_data.get(_LOGO_WEIGHT_KEYS[i], 0) * fv[i]
    return s


# =============================================================================
# Type Registry Functions
# =============================================================================

def load_type_registry(weights_dir: str = DEFAULT_WEIGHTS_DIR) -> Dict[str, Dict]:
    """Load type registry from disk."""
    global _type_registry, _type_registry_file
    _type_registry_file = weights_registry_path(weights_dir)
    
    if os.path.exists(_type_registry_file):
        try:
            with open(_type_registry_file) as f:
                _type_registry = json.load(f)
        except Exception:
            _type_registry = {}
    return _type_registry


def save_type_registry(weights_dir: str = DEFAULT_WEIGHTS_DIR):
    """Save type registry to disk."""
    global _type_registry
    os.makedirs(weights_dir, exist_ok=True)
    registry_file = weights_registry_path(weights_dir)
    with open(registry_file, 'w') as f:
        json.dump(_type_registry, f, indent=2)


def get_type_weights_file(type_name: str, weights_dir: str = DEFAULT_WEIGHTS_DIR) -> str:
    """Get path to weights file for a type (e.g. type_0/weights.json)."""
    return weights_type_path(type_name, weights_dir)


def load_type_weights(type_name: str, weights_dir: str = DEFAULT_WEIGHTS_DIR) -> Dict:
    """Load weights for a specific type."""
    weights_file = get_type_weights_file(type_name, weights_dir)
    if os.path.exists(weights_file):
        try:
            with open(weights_file) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_type_weights(type_name: str, weights: Dict, weights_dir: str = DEFAULT_WEIGHTS_DIR):
    """Save weights for a specific type."""
    weights_file = get_type_weights_file(type_name, weights_dir)
    os.makedirs(os.path.dirname(weights_file), exist_ok=True)
    with open(weights_file, 'w') as f:
        json.dump(weights, f, indent=2)


# =============================================================================
# Type Assignment and Weight Updates
# =============================================================================

def assign_graph_type(
    features: Dict,
    weights_dir: str = DEFAULT_WEIGHTS_DIR,
    create_if_outlier: bool = True,
    graph_name: str = None,
) -> str:
    """
    Assign a graph to a type based on its features.
    
    Uses clustering to find the closest existing type or creates a new one
    if the graph is an outlier (distance > CLUSTER_DISTANCE_THRESHOLD).
    
    Args:
        features: Dict with 'modularity', 'degree_variance', 'hub_concentration', etc.
        weights_dir: Directory for type files
        create_if_outlier: If True, create new type for outliers
        graph_name: Optional graph name to track in registry
        
    Returns:
        Type name (e.g., 'type_0', 'type_1')
    """
    global _type_registry
    
    if not _type_registry:
        load_type_registry(weights_dir)
    
    norm_features = _normalize_features(features)
    
    # Find closest existing type
    min_distance = float('inf')
    closest_type = None
    
    for type_name, type_info in _type_registry.items():
        if 'centroid' in type_info:
            dist = _compute_distance(norm_features, type_info['centroid'])
            if dist < min_distance:
                min_distance = dist
                closest_type = type_name
    
    # Check if close enough to existing type
    if closest_type and min_distance < CLUSTER_DISTANCE_THRESHOLD:
        type_info = _type_registry[closest_type]
        count = type_info.get('sample_count', 1)
        old_centroid = type_info['centroid']
        new_centroid = [
            old + (new - old) / (count + 1)
            for old, new in zip(old_centroid, norm_features)
        ]
        _type_registry[closest_type]['centroid'] = new_centroid
        _type_registry[closest_type]['sample_count'] = count + 1
        _type_registry[closest_type]['graph_count'] = count + 1
        _type_registry[closest_type]['last_updated'] = datetime.now().isoformat()
        
        # Track graph names
        if graph_name:
            if 'graphs' not in _type_registry[closest_type]:
                _type_registry[closest_type]['graphs'] = []
            if graph_name not in _type_registry[closest_type]['graphs']:
                _type_registry[closest_type]['graphs'].append(graph_name)
        
        save_type_registry(weights_dir)
        return closest_type
    
    # Create new type if outlier
    if create_if_outlier:
        new_type = _get_next_type_name()
        _type_registry[new_type] = {
            'centroid': norm_features,
            'sample_count': 1,
            'graph_count': 1,
            'algorithms': [],
            'graphs': [graph_name] if graph_name else [],
            'created': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'representative_features': {
                'modularity': features.get('modularity', 0.5),
                'degree_variance': features.get('degree_variance', 1.0),
                'hub_concentration': features.get('hub_concentration', 0.3),
                'avg_degree': features.get('avg_degree', 10),
            }
        }
        save_type_registry(weights_dir)
        
        if closest_type:
            existing_weights = load_type_weights(closest_type, weights_dir)
            if existing_weights:
                save_type_weights(new_type, existing_weights, weights_dir)
        
        return new_type
    
    return closest_type or "type_0"


def update_type_weights_incremental(
    type_name: str,
    algorithm: str,
    benchmark: str,
    speedup: float,
    features: Dict,
    cache_stats: Optional[Dict] = None,
    reorder_time: float = 0.0,
    weights_dir: str = DEFAULT_WEIGHTS_DIR,
    learning_rate: float = 0.01
):
    """
    Incrementally update weights for a type after a benchmark run.
    
    Args:
        type_name: The graph type (e.g., 'type_0')
        algorithm: Algorithm name (e.g., 'HUBCLUSTERDBG')
        benchmark: Benchmark name (e.g., 'bfs')
        speedup: Observed speedup vs baseline
        features: Graph features dict
        cache_stats: Optional cache simulation results
        reorder_time: Time to reorder the graph
        weights_dir: Directory for type files
        learning_rate: Learning rate for weight updates
    """
    weights = load_type_weights(type_name, weights_dir)
    if not weights:
        weights = {}
    
    # Skip algorithms not eligible for perceptron selection
    if is_chained_ordering_name(algorithm) or algorithm in PERCEPTRON_EXCLUDED:
        return
    
    # Track algorithm in type registry
    global _type_registry
    if not _type_registry:
        load_type_registry(weights_dir)
    if type_name in _type_registry:
        if 'algorithms' not in _type_registry[type_name]:
            _type_registry[type_name]['algorithms'] = []
        if algorithm not in _type_registry[type_name]['algorithms']:
            _type_registry[type_name]['algorithms'].append(algorithm)
            save_type_registry(weights_dir)
    
    # Initialize algorithm weights if not present
    if algorithm not in weights:
        weights[algorithm] = _create_default_weight_entry()
    
    algo_weights = weights[algorithm]
    meta = algo_weights.get('_metadata', {})
    
    # Update sample count and running average speedup
    count = meta.get('sample_count', 0) + 1
    old_avg = meta.get('avg_speedup', 1.0)
    new_avg = old_avg + (speedup - old_avg) / count
    
    meta['sample_count'] = count
    meta['avg_speedup'] = new_avg
    meta['last_updated'] = datetime.now().isoformat()
    
    if speedup > 1.05:
        meta['win_count'] = meta.get('win_count', 0) + 1
    if speedup > 1.0:
        meta['times_best'] = meta.get('times_best', 0) + 1
    
    meta['win_rate'] = meta['win_count'] / count if count > 0 else 0.0
    
    # Compute error signal
    current_score = algo_weights['bias']
    error = (speedup - 1.0) - current_score
    
    # Gradient update — feature transforms must match C++ scoreBase() exactly.
    # This ensures the gradient ∂L/∂w uses the same feature values as the
    # forward pass:  score = w · f(x), so ∂score/∂w = f(x).
    log_nodes = math.log10(features.get('nodes', 1) + 1)
    log_edges = math.log10(features.get('edges', 1) + 1)
    
    # Accept both clustering key names
    clustering = features.get('clustering_coefficient', features.get('clustering_coeff', 0.0))
    
    # Use real modularity if available (C++ now computes it via
    # computeFastModularity). Fall back to CC×1.5 estimate only if
    # no modularity value is provided.
    modularity_val = features.get('modularity', None)
    if modularity_val is None:
        modularity_val = min(0.9, clustering * 1.5)
    
    algo_weights['bias'] += learning_rate * error
    algo_weights['w_modularity'] += learning_rate * error * modularity_val
    algo_weights['w_log_nodes'] += learning_rate * error * log_nodes
    algo_weights['w_log_edges'] += learning_rate * error * log_edges
    algo_weights['w_density'] += learning_rate * error * features.get('density', 0.0)
    algo_weights['w_avg_degree'] += learning_rate * error * features.get('avg_degree', 0.0) / 100.0
    algo_weights['w_degree_variance'] += learning_rate * error * features.get('degree_variance', 0.0)
    algo_weights['w_hub_concentration'] += learning_rate * error * features.get('hub_concentration', 0.0)
    algo_weights['w_clustering_coeff'] += learning_rate * error * clustering
    algo_weights['w_community_count'] += learning_rate * error * math.log10(features.get('community_count', 0) + 1)
    algo_weights['w_avg_path_length'] += learning_rate * error * features.get('avg_path_length', 0.0) / WEIGHT_PATH_LENGTH_NORMALIZATION
    algo_weights['w_diameter'] += learning_rate * error * features.get('diameter', features.get('diameter_estimate', 0.0)) / 50.0
    
    # Locality features (IISWC'18 Packing Factor, GoGraph forward edge fraction)
    algo_weights['w_packing_factor'] += learning_rate * error * features.get('packing_factor', 0.0)
    algo_weights['w_forward_edge_fraction'] += learning_rate * error * features.get('forward_edge_fraction', 0.0)
    algo_weights['w_working_set_ratio'] += learning_rate * error * math.log2(features.get('working_set_ratio', 0.0) + 1.0)
    
    # Quadratic interaction gradients
    log_wsr = math.log2(features.get('working_set_ratio', 0.0) + 1.0)
    log_n = math.log10(features.get('nodes', 1) + 1)
    algo_weights['w_dv_x_hub'] = algo_weights.get('w_dv_x_hub', 0.0) + learning_rate * error * features.get('degree_variance', 0.0) * features.get('hub_concentration', 0.0)
    algo_weights['w_mod_x_logn'] = algo_weights.get('w_mod_x_logn', 0.0) + learning_rate * error * modularity_val * log_n
    algo_weights['w_pf_x_wsr'] = algo_weights.get('w_pf_x_wsr', 0.0) + learning_rate * error * features.get('packing_factor', 0.0) * log_wsr
    
    # Convergence bonus gradient (only for iterative benchmarks)
    # Matches C++ score(): bench == BENCH_PR || BENCH_PR_SPMV || BENCH_SSSP
    if benchmark in ('pr', 'pr_spmv', 'sssp'):
        algo_weights['w_fef_convergence'] = algo_weights.get('w_fef_convergence', 0.0) + learning_rate * error * features.get('forward_edge_fraction', 0.0)
    
    # Update cache weights
    if cache_stats:
        l1_hit = cache_stats.get('l1_hit_rate', 0) / 100.0
        l2_hit = cache_stats.get('l2_hit_rate', 0) / 100.0
        l3_hit = cache_stats.get('l3_hit_rate', 0) / 100.0
        dram_penalty = 1.0 - (l1_hit + l2_hit * 0.3 + l3_hit * 0.1)
        
        algo_weights['cache_l1_impact'] += learning_rate * error * l1_hit
        algo_weights['cache_l2_impact'] += learning_rate * error * l2_hit
        algo_weights['cache_l3_impact'] += learning_rate * error * l3_hit
        algo_weights['cache_dram_penalty'] += learning_rate * error * dram_penalty
        
        meta['avg_l1_hit_rate'] = meta.get('avg_l1_hit_rate', 0) + (l1_hit * 100 - meta.get('avg_l1_hit_rate', 0)) / count
        meta['avg_l2_hit_rate'] = meta.get('avg_l2_hit_rate', 0) + (l2_hit * 100 - meta.get('avg_l2_hit_rate', 0)) / count
        meta['avg_l3_hit_rate'] = meta.get('avg_l3_hit_rate', 0) + (l3_hit * 100 - meta.get('avg_l3_hit_rate', 0)) / count
    
    if reorder_time > 0:
        algo_weights['w_reorder_time'] += learning_rate * error * (-reorder_time)
        meta['avg_reorder_time'] = meta.get('avg_reorder_time', 0) + (reorder_time - meta.get('avg_reorder_time', 0)) / count
    
    # Update benchmark-specific weight
    bench_weights = algo_weights.get('benchmark_weights', {})
    if benchmark.lower() in bench_weights:
        current_bench_weight = bench_weights[benchmark.lower()]
        bench_weights[benchmark.lower()] = current_bench_weight + learning_rate * error * 0.1
    
    # L2 weight decay (regularization) to prevent weight explosion
    # Applied to all feature weights (not bias, not metadata, not benchmark_weights)
    # Decay rate: 1e-4 per update — gentle enough to not interfere with learning
    WEIGHT_DECAY = 1e-4
    weight_keys = [k for k in algo_weights.keys() 
                   if k.startswith('w_') or k.startswith('cache_')]
    for k in weight_keys:
        if isinstance(algo_weights[k], (int, float)):
            algo_weights[k] *= (1.0 - WEIGHT_DECAY)
    
    algo_weights['_metadata'] = meta
    algo_weights['benchmark_weights'] = bench_weights
    weights[algorithm] = algo_weights
    
    save_type_weights(type_name, weights, weights_dir)


def get_best_algorithm_for_type(
    type_name: str,
    features: Dict,
    benchmark: str = 'pr',
    weights_dir: str = DEFAULT_WEIGHTS_DIR
) -> Tuple[str, float]:
    """
    Get the best algorithm for a graph type based on learned weights.
    
    Returns:
        (algorithm_name, score) tuple
    """
    weights = load_type_weights(type_name, weights_dir)
    if not weights:
        return ("SORT", 0.0)  # fallback: simplest real reordering
    
    candidates = get_perceptron_candidates()
    best_algo = None
    best_score = float('-inf')
    
    for algo_name, algo_weights in weights.items():
        if algo_name.startswith('_') or algo_name not in candidates:
            continue
        
        pw = PerceptronWeight.from_dict(algo_weights)
        
        score = pw.compute_score(features, benchmark)
        
        # Confidence boost
        meta = algo_weights.get('_metadata', {})
        sample_count = meta.get('sample_count', 0)
        confidence_boost = min(0.1, sample_count * 0.01)
        score += confidence_boost
        
        if score > best_score:
            best_score = score
            best_algo = algo_name
    
    return (best_algo or "SORT", best_score)  # fallback: simplest real reordering


def list_known_types(weights_dir: str = DEFAULT_WEIGHTS_DIR) -> List[str]:
    """List all known graph types."""
    load_type_registry(weights_dir)
    return list(_type_registry.keys())


def get_type_summary(type_name: str, weights_dir: str = DEFAULT_WEIGHTS_DIR) -> Dict:
    """Get summary information about a type."""
    load_type_registry(weights_dir)
    type_info = _type_registry.get(type_name, {})
    weights = load_type_weights(type_name, weights_dir)
    
    algo_stats = {}
    best_algos = {}
    
    for algo_name, algo_weights in weights.items():
        if algo_name.startswith('_'):
            continue
        meta = algo_weights.get('_metadata', {})
        algo_stats[algo_name] = {
            'sample_count': meta.get('sample_count', 0),
            'avg_speedup': meta.get('avg_speedup', 1.0),
            'win_count': meta.get('win_count', 0),
        }
        for bench, weight in algo_weights.get('benchmark_weights', {}).items():
            if bench not in best_algos or weight > best_algos[bench][1]:
                best_algos[bench] = (algo_name, weight)
    
    return {
        'type_name': type_name,
        'num_graphs': type_info.get('sample_count', 0),
        'sample_count': type_info.get('sample_count', 0),
        'created': type_info.get('created', 'unknown'),
        'last_updated': type_info.get('last_updated', 'unknown'),
        'centroid': type_info.get('centroid', []),
        'representative_features': type_info.get('representative_features', {}),
        'algorithms': algo_stats,
        'best_algorithms': {bench: algo for bench, (algo, _) in best_algos.items()},
    }


def compute_weights_from_results(
    benchmark_results: List,
    cache_results: List = None,
    reorder_results: List = None,
    output_file: str = None,
    weights_dir: str = None,
    candidate_algos: frozenset = None,
) -> Dict:
    """
    Compute perceptron weights from benchmark results.
    
    This function analyzes benchmark results to determine which algorithms
    perform best for different graph types, and computes weights accordingly.
    
    Args:
        benchmark_results: List of BenchmarkResult objects
        cache_results: Optional list of CacheResult objects
        reorder_results: Optional list of ReorderResult objects
        output_file: Optional file to save weights
        weights_dir: Optional directory to save type-based weights
        candidate_algos: Optional frozenset of algorithm names the perceptron
            may select from.  Defaults to ``get_perceptron_candidates()``.
            Pass a custom set to restrict or expand the candidate pool.
        
    Returns:
        Dict of weights by algorithm
    """
    cache_results = cache_results or []
    reorder_results = reorder_results or []
    if weights_dir is None:
        weights_dir = DEFAULT_WEIGHTS_DIR
    
    # Initialize weights - start with default then add from results
    weights = initialize_default_weights()
    
    # Collect all unique algorithm names from results (includes variants)
    all_algo_names = set()
    for r in benchmark_results:
        if r.algorithm:
            all_algo_names.add(r.algorithm)
    for r in reorder_results:
        algo_name = getattr(r, 'algorithm_name', None) or getattr(r, 'algorithm', '')
        if algo_name:
            all_algo_names.add(algo_name)
    
    # Filter to perceptron-eligible algorithms only.
    # Chained orderings can't be selected at C++ runtime (perceptron picks
    # a single algorithm ID).  PERCEPTRON_EXCLUDED removes baseline states
    # (ORIGINAL, RANDOM) that are not reordering techniques.
    candidates = candidate_algos if candidate_algos is not None else get_perceptron_candidates()
    log.info(f"Perceptron candidates ({len(candidates)}): "
             f"{sorted(candidates)}")
    all_algo_names = {n for n in all_algo_names
                      if not is_chained_ordering_name(n) and n in candidates}
    benchmark_results = [r for r in benchmark_results
                         if not is_chained_ordering_name(r.algorithm or '')]
    reorder_results = [r for r in reorder_results
                       if not is_chained_ordering_name(
                           getattr(r, 'algorithm_name', None) or getattr(r, 'algorithm', ''))]
    
    # Add variant algorithms that aren't in default weights
    for algo_name in all_algo_names:
        if algo_name not in weights and not algo_name.startswith('_'):
            weights[algo_name] = _create_default_weight_entry()
    
    # Group results by graph and benchmark
    results_by_graph = {}
    for r in benchmark_results:
        if not r.success or r.time_seconds <= 0:
            continue
        
        graph_name = r.graph
        if graph_name not in results_by_graph:
            results_by_graph[graph_name] = {}
        
        bench = r.benchmark
        if bench not in results_by_graph[graph_name]:
            results_by_graph[graph_name][bench] = []
        
        results_by_graph[graph_name][bench].append(r)
    
    # Find best algorithm per (graph, benchmark) and collect win statistics
    algo_win_count = {}   # algo -> number of (graph, benchmark) contexts won
    algo_speedups = {}    # algo -> list of speedups when winning
    total_contexts = 0
    
    for _graph_name, benchmarks in results_by_graph.items():
        for _bench, results in benchmarks.items():
            if not results:
                continue
            total_contexts += 1
            
            # Find baseline (ORIGINAL or first result) — use total time (exec + reorder)
            baseline_time = None
            for r in results:
                if 'ORIGINAL' in r.algorithm:
                    baseline_time = r.time_seconds + r.reorder_time
                    break
            if baseline_time is None:
                baseline_time = results[0].time_seconds + results[0].reorder_time
            
            # Find best by total time (exec + reorder) — practical end-to-end metric
            best_result = min(results, key=lambda r: r.time_seconds + r.reorder_time)
            best_algo = best_result.algorithm
            
            if best_algo in weights and baseline_time > 0:
                best_total = best_result.time_seconds + best_result.reorder_time
                speedup = baseline_time / best_total
                algo_win_count[best_algo] = algo_win_count.get(best_algo, 0) + 1
                if best_algo not in algo_speedups:
                    algo_speedups[best_algo] = []
                algo_speedups[best_algo].append(speedup)
    
    # Compute bias from win rate: scale to [0.3, 1.0] range
    # This gives a max spread of 0.7 — features can easily override this
    for algo in weights:
        if algo.startswith('_'):
            continue
        wins = algo_win_count.get(algo, 0)
        win_rate = wins / total_contexts if total_contexts > 0 else 0.0
        # Bias = 0.3 + 0.7 * win_rate  (never won → 0.3, won everything → 1.0)
        weights[algo]['bias'] = round(0.3 + 0.7 * win_rate, 4)
        
        # Update metadata
        meta = weights[algo].get('_metadata', {})
        meta['win_count'] = wins
        meta['sample_count'] = wins
        if algo in algo_speedups:
            meta['avg_speedup'] = sum(algo_speedups[algo]) / len(algo_speedups[algo])
        weights[algo]['_metadata'] = meta
    
    # =========================================================================
    # Multi-class Perceptron Training
    # =========================================================================
    # Train a multi-class perceptron via the standard update rule:
    #   For each training example (graph_features, best_algorithm):
    #     predicted = argmax_algo score(algo, features)
    #     if predicted != best_algorithm:
    #       w[best] += learning_rate * features
    #       w[predicted] -= learning_rate * features
    # This directly optimizes for correct algorithm selection.
    
    from .features import load_graph_properties_cache
    from scripts.lib.core.utils import RESULTS_DIR
    graph_props = load_graph_properties_cache(RESULTS_DIR)
    
    # Feature keys used in C++ scoreBase()
    feat_to_weight = {
        'modularity': 'w_modularity',
        'degree_variance': 'w_degree_variance',
        'hub_concentration': 'w_hub_concentration',
        'log_nodes': 'w_log_nodes',
        'log_edges': 'w_log_edges',
        'density': 'w_density',
        'avg_degree': 'w_avg_degree',
        'clustering_coefficient': 'w_clustering_coeff',
        'avg_path_length': 'w_avg_path_length',
        'diameter': 'w_diameter',
        'community_count': 'w_community_count',
        # IISWC'18 / GoGraph / P-OPT locality features
        'packing_factor': 'w_packing_factor',
        'forward_edge_fraction': 'w_forward_edge_fraction',
        'log_working_set_ratio': 'w_working_set_ratio',
        # Quadratic interaction terms (matches C++ scoreBase cross-features)
        'dv_x_hub': 'w_dv_x_hub',
        'mod_x_logn': 'w_mod_x_logn',
        'pf_x_wsr': 'w_pf_x_wsr',
    }
    
    # Collect features per graph
    #
    # TRAINING vs INFERENCE feature distinction:
    #   - Training (here): use REAL modularity from computeFastModularity
    #     (stored in data/graph_properties.json).  This gives the SGD a
    #     much higher-quality signal (CC*1.5 estimate is 2-178× off).
    #   - LOGO / C++ runtime: use ESTIMATED modularity = min(0.9, CC*1.5)
    #     since that's all the C++ runtime has without the expensive
    #     computeFastModularity call.
    #
    # The de-normalized weights will be calibrated for real modularity
    # scale.  At C++ runtime the estimated value is smaller, so the
    # modularity contribution shrinks — but relative ordering across
    # graphs is preserved, which is what the perceptron needs.
    graph_features = {}
    for graph_name, props in graph_props.items():
        nodes = props.get('nodes', 1000)
        edges = props.get('edges', 5000)
        cc = props.get('clustering_coefficient', 0.0)
        avg_degree = props.get('avg_degree', WEIGHT_AVG_DEGREE_DEFAULT)
        
        # Use real modularity from cache if available, else fall back to estimate
        real_modularity = props.get('modularity', None)
        estimated_modularity = min(0.9, cc * 1.5)
        train_modularity = real_modularity if real_modularity is not None else estimated_modularity
        
        internal_density = avg_degree / (nodes - 1) if nodes > 1 else 0
        log_n = math.log10(nodes + 1) if nodes > 0 else 0
        
        graph_features[graph_name] = {
            'modularity': train_modularity,
            'degree_variance': props.get('degree_variance', 1.0),
            'hub_concentration': props.get('hub_concentration', 0.3),
            'avg_degree': avg_degree,
            'log_nodes': log_n,
            'log_edges': math.log10(edges + 1) if edges > 0 else 0,
            'density': internal_density,
            'clustering_coefficient': cc,
            'avg_path_length': props.get('avg_path_length', 0.0),
            'diameter': props.get('diameter_estimate', 0.0),
            'community_count': props.get('community_count', 1.0),
            # Locality features — match C++ transforms:
            # packing_factor: raw (IISWC'18)
            # forward_edge_fraction: raw (GoGraph)
            # working_set_ratio: C++ uses log2(wsr + 1.0)
            'packing_factor': props.get('packing_factor', 0.0),
            'forward_edge_fraction': props.get('forward_edge_fraction', 0.5),
            'log_working_set_ratio': math.log2(props.get('working_set_ratio', 0.0) + 1.0),
            # Quadratic interaction terms — use training modularity for mod_x_logn
            'dv_x_hub': props.get('degree_variance', 1.0) * props.get('hub_concentration', 0.3),
            'mod_x_logn': train_modularity * log_n,
            'pf_x_wsr': props.get('packing_factor', 0.0) * math.log2(props.get('working_set_ratio', 0.0) + 1.0),
            # Estimated values for de-normalization (C++ runtime only has these)
            '_est_modularity': estimated_modularity,
            '_est_mod_x_logn': estimated_modularity * log_n,
        }
    
    # Build training examples: (feature_vector, best_algorithm)
    # Each canonical variant name (e.g., GraphBrewOrder_leiden, RCM_bnf)
    # is a separate perceptron class for variant-level prediction.
    
    training_data = []
    for graph_name, benchmarks in results_by_graph.items():
        if graph_name not in graph_features:
            continue
        feats = graph_features[graph_name]
        
        for _bench, results in benchmarks.items():
            if not results:
                continue
            # Oracle by execution time only — reorder cost is handled
            # separately by the C++ reorder_time_penalty in scoreBase().
            # Including raw reorder_time here causes ORIGINAL to dominate,
            # masking the per-algorithm execution differences.
            # Filter to perceptron candidates: if the overall winner is a
            # baseline state (ORIGINAL/RANDOM), pick the best candidate
            # algorithm instead so training targets a selectable algo.
            candidate_results = [r for r in results
                                 if r.algorithm in candidates]
            if not candidate_results:
                continue  # no candidate ran on this graph/bench — skip
            best_result = min(candidate_results, key=lambda r: r.time_seconds)
            best_algo = best_result.algorithm
            
            # Build feature vector matching C++ scoreBase() scaling
            fv = [
                feats['modularity'],
                feats['degree_variance'],
                feats['hub_concentration'],
                feats['log_nodes'],
                feats['log_edges'],
                feats['density'],
                feats['avg_degree'] / 100.0,
                feats['clustering_coefficient'],
                feats['avg_path_length'] / 10.0,
                feats['diameter'] / 50.0,
                math.log10(feats['community_count'] + 1) if feats['community_count'] > 0 else 0,
                # IISWC'18 / GoGraph / P-OPT locality features
                feats['packing_factor'],
                feats['forward_edge_fraction'],
                feats['log_working_set_ratio'],
                # Quadratic interaction terms
                feats['dv_x_hub'],
                feats['mod_x_logn'],
                feats['pf_x_wsr'],
            ]
            
            training_data.append((fv, best_algo))
    
    if training_data and graph_features:
        # Get algorithm names for training (canonical variant names)
        # Include all algorithms from weights PLUS any that appear as winners
        # in training data, filtered to perceptron candidates only.
        # Baseline states (ORIGINAL, RANDOM) are excluded even if they win
        # on execution time — the perceptron should only select real
        # reordering algorithms.
        training_labels = set(algo for _fv, algo in training_data)
        base_algos = sorted(
            (set(a for a in weights if not a.startswith('_')) | training_labels)
            & candidates  # intersect with perceptron-eligible algorithms
        )
        
        # =====================================================================
        # Per-benchmark perceptron training → C++ compatible output
        # =====================================================================
        # Strategy:
        #   1. Train separate perceptrons per benchmark (high accuracy each)
        #   2. Feature weights = average across benchmarks (shared scoreBase)
        #   3. Per-bench residual → benchmark_weights multiplier
        #   4. De-normalize features for C++ raw-feature scoring
        #
        # This matches C++ model: score = scoreBase(feat) * benchmarkMultiplier(bench)
        
        import random
        random.seed(42)
        
        weight_keys = list(feat_to_weight.values())
        n_feat = len(weight_keys)
        
        # Collect bench names
        bench_names = sorted(set(
            bench for benchmarks in results_by_graph.values()
            for bench in benchmarks.keys()
        ))
        
        # Build per-benchmark training data with raw features
        per_bench_data_raw = {bn: [] for bn in bench_names}
        for graph_name, benchmarks in results_by_graph.items():
            if graph_name not in graph_features:
                continue
            feats = graph_features[graph_name]
            fv = [
                feats['modularity'],
                feats['degree_variance'],
                feats['hub_concentration'],
                feats['log_nodes'],
                feats['log_edges'],
                feats['density'],
                feats['avg_degree'] / 100.0,
                feats['clustering_coefficient'],
                feats['avg_path_length'] / 10.0,
                feats['diameter'] / 50.0,
                math.log10(feats['community_count'] + 1) if feats['community_count'] > 0 else 0,
                # IISWC'18 / GoGraph / P-OPT locality features
                feats['packing_factor'],
                feats['forward_edge_fraction'],
                feats['log_working_set_ratio'],
                # Quadratic interaction terms
                feats['dv_x_hub'],
                feats['mod_x_logn'],
                feats['pf_x_wsr'],
            ]
            for bench, results in benchmarks.items():
                if not results:
                    continue
                # Oracle by execution time — reorder cost handled by C++ penalty
                # Filter to perceptron candidates only (exclude baselines)
                cand_results = [r for r in results if r.algorithm in candidates]
                if not cand_results:
                    continue
                best_result = min(cand_results, key=lambda r: r.time_seconds)
                best_algo = best_result.algorithm
                per_bench_data_raw[bench].append((fv, best_algo))
        
        # Feature normalization stats (computed from REAL modularity for training)
        all_fvs = [fv for bn in bench_names for fv, _ in per_bench_data_raw[bn]]
        feat_means = [0.0] * n_feat
        feat_stds = [1.0] * n_feat
        if all_fvs:
            for i in range(n_feat):
                vals = [fv[i] for fv in all_fvs]
                feat_means[i] = sum(vals) / len(vals)
                var = sum((v - feat_means[i])**2 for v in vals) / len(vals)
                feat_stds[i] = max(math.sqrt(var), 1e-8)
        
        # De-normalization stats: use ESTIMATED modularity for mod-related features.
        # Training uses real modularity (better gradient signal), but C++ runtime
        # only has estimated_mod = min(0.9, CC*1.5).  De-normalizing with real
        # stats would bake the real mean into the bias, causing sign flips when
        # C++ feeds in much-smaller estimated values.
        # Indices: 0 = 'modularity', n_feat-2 = 'mod_x_logn'
        MOD_IDX = 0
        MOD_LOGN_IDX = n_feat - 2  # mod_x_logn is second-to-last feature
        feat_means_denorm = list(feat_means)
        feat_stds_denorm = list(feat_stds)
        # Collect estimated values per unique training graph
        training_graphs = [gn for gn in results_by_graph if gn in graph_features]
        if training_graphs:
            est_mod_vals = [graph_features[gn]['_est_modularity'] for gn in training_graphs]
            est_mln_vals = [graph_features[gn]['_est_mod_x_logn'] for gn in training_graphs]
            em_mean = sum(est_mod_vals) / len(est_mod_vals)
            em_var = sum((v - em_mean)**2 for v in est_mod_vals) / len(est_mod_vals)
            feat_means_denorm[MOD_IDX] = em_mean
            feat_stds_denorm[MOD_IDX] = max(math.sqrt(em_var), 1e-8)
            eml_mean = sum(est_mln_vals) / len(est_mln_vals)
            eml_var = sum((v - eml_mean)**2 for v in est_mln_vals) / len(est_mln_vals)
            feat_means_denorm[MOD_LOGN_IDX] = eml_mean
            feat_stds_denorm[MOD_LOGN_IDX] = max(math.sqrt(eml_var), 1e-8)
            log.info(f"  Modularity de-norm: real mean={feat_means[MOD_IDX]:.4f} std={feat_stds[MOD_IDX]:.4f}"
                     f" → estimated mean={feat_means_denorm[MOD_IDX]:.4f} std={feat_stds_denorm[MOD_IDX]:.4f}")
        
        # Normalize per-benchmark data
        per_bench_data = {}
        for bn in bench_names:
            per_bench_data[bn] = [
                ([(fv[i] - feat_means[i]) / feat_stds[i] for i in range(n_feat)], algo)
                for fv, algo in per_bench_data_raw[bn]
            ]
        
        # Train one perceptron per benchmark with multiple random restarts
        # Using margin-based (theta) training, weight clamping, and averaged
        # perceptron — inspired by Jimenez & Lin HPCA 2001 / Jimenez MICRO 2016
        per_bench_w = {}  # bench -> {base -> {'bias': float, 'w': [float]}}
        N_RESTARTS = 5
        N_EPOCHS = 800

        # Theta threshold: update on wrong OR low-confidence correct predictions
        # Jimenez formula: theta = floor(1.93 * h + 14) for binary features
        # with weight cap 127.  Scale proportionally for our z-normalized
        # features with W_MAX capped weights:
        #   theta_scaled = floor((1.93*h + 14) * W_MAX / 127)
        # This preserves the theta/W_MAX ratio (~0.36) from the original paper.
        # Weight saturation: prevents overfitting on few training examples
        # Jimenez uses [-127, +127] for binary {+1,-1} features;
        # for z-normalized real features, use a proportional cap.
        W_MAX = 16.0
        THETA = max(1, int((1.93 * n_feat + 14) * W_MAX / 127))  # ~5 for 17 feat, W_MAX=16
        
        for bn in bench_names:
            data = per_bench_data[bn]
            if not data:
                continue

            # Restrict training to algorithms that actually appear as winners
            # in this benchmark's training data.  With 5 graphs & 22 classes,
            # the 15-17 classes with NO training data only add noise.
            # Intersect with candidates as a safety guard.
            active_algos = sorted(set(algo for _fv, algo in data) & candidates)
            log.info(f"  {bn}: {len(active_algos)} active classes "
                     f"(of {len(base_algos)}) → {active_algos}")
            
            global_best_acc = 0
            global_best_snap = None
            
            for restart in range(N_RESTARTS):
                # Use benchmark index for deterministic seeding
                bn_idx = bench_names.index(bn)
                random.seed(42 + restart * 1000 + bn_idx * 100)
                
                # Initialize with small random weights for diversity
                bw = {base: {
                    'bias': random.gauss(0, 0.1),
                    'w': [random.gauss(0, 0.1) for _ in range(n_feat)]
                } for base in active_algos}
                
                # Averaged perceptron: accumulate weight sums across all updates
                # (Freund & Schapire 1999 — provably better generalization)
                avg_bw = {base: {
                    'bias': 0.0,
                    'w': [0.0] * n_feat
                } for base in active_algos}
                avg_count = 0  # total steps for averaging

                lr = 0.05
                best_acc = 0
                best_snap = None
                # Adaptive theta (Jimenez MICRO 2016): auto-tune the margin
                # threshold per restart.  Start from the static formula value.
                # With few examples (5-10), naive ±1 steps cause theta to grow
                # unboundedly since nearly all predictions are correct.
                # Cap at 3× initial and use fractional steps ∝ 1/n_examples.
                theta = float(THETA)
                theta_max = THETA * 3.0
                theta_step = max(0.1, 1.0 / max(len(data), 1))
                
                for _epoch in range(N_EPOCHS):
                    random.shuffle(data)
                    correct = 0
                    for fv_n, true_base in data:
                        if true_base not in bw:
                            continue
                        # Compute scores for all active classes
                        scores = {}
                        for base in active_algos:
                            scores[base] = bw[base]['bias'] + sum(
                                bw[base]['w'][i] * fv_n[i] for i in range(n_feat))

                        # Find predicted (best) and runner-up
                        pred = max(scores, key=scores.get)
                        true_score = scores[true_base]
                        pred_score = scores[pred]

                        is_correct = (pred == true_base)
                        if is_correct:
                            correct += 1
                            # Margin = true_score - runner_up_score
                            runner_up = max(
                                (s for b, s in scores.items() if b != true_base),
                                default=float('-inf'))
                            margin = true_score - runner_up
                        else:
                            margin = -1  # always below theta

                        # Adaptive theta update (Jimenez MICRO 2016):
                        #   correct & confident (margin > theta) → increment theta
                        #     (demand wider margins → more updates → stronger generalization)
                        #   wrong → decrement theta
                        #     (relax threshold → fewer forced updates → let model stabilize)
                        if is_correct and margin > theta:
                            theta = min(theta_max, theta + theta_step)
                        elif not is_correct:
                            theta = max(0.0, theta - theta_step)

                        # Jimenez update rule: update on wrong OR margin <= theta
                        if not is_correct or margin <= theta:
                            # Promote true class, demote predicted class
                            bw[true_base]['bias'] += lr
                            bw[true_base]['bias'] = max(-W_MAX, min(W_MAX, bw[true_base]['bias']))
                            if not is_correct:
                                bw[pred]['bias'] -= lr
                                bw[pred]['bias'] = max(-W_MAX, min(W_MAX, bw[pred]['bias']))
                            for i in range(n_feat):
                                bw[true_base]['w'][i] += lr * fv_n[i]
                                bw[true_base]['w'][i] = max(-W_MAX, min(W_MAX, bw[true_base]['w'][i]))
                                if not is_correct:
                                    bw[pred]['w'][i] -= lr * fv_n[i]
                                    bw[pred]['w'][i] = max(-W_MAX, min(W_MAX, bw[pred]['w'][i]))

                        # Accumulate for averaged perceptron
                        avg_count += 1
                        for base in active_algos:
                            avg_bw[base]['bias'] += bw[base]['bias']
                            for i in range(n_feat):
                                avg_bw[base]['w'][i] += bw[base]['w'][i]
                    
                    acc = correct / len(data) if data else 0
                    if acc > best_acc:
                        best_acc = acc
                        best_snap = {
                            base: {'bias': d['bias'], 'w': list(d['w'])}
                            for base, d in bw.items()
                        }
                    lr *= 0.997
                
                # Use averaged weights (better generalization, especially with few examples)
                if avg_count > 0:
                    avg_snap = {
                        base: {
                            'bias': avg_bw[base]['bias'] / avg_count,
                            'w': [avg_bw[base]['w'][i] / avg_count for i in range(n_feat)]
                        } for base in active_algos
                    }
                else:
                    avg_snap = best_snap

                # Evaluate averaged weights on training data
                avg_acc = 0
                if avg_snap is not None:
                    avg_correct = 0
                    for fv_n, true_base in data:
                        if true_base not in avg_snap:
                            continue
                        best_s, apred = float('-inf'), None
                        for base in active_algos:
                            s = avg_snap[base]['bias'] + sum(
                                avg_snap[base]['w'][i] * fv_n[i] for i in range(n_feat))
                            if s > best_s:
                                best_s, apred = s, base
                        if apred == true_base:
                            avg_correct += 1
                    avg_acc = avg_correct / len(data) if data else 0

                # Pick whichever is better: averaged or best-snapshot
                use_acc = max(avg_acc, best_acc)
                use_snap = (avg_snap if avg_acc >= best_acc and avg_snap is not None
                            else best_snap)

                if use_acc > global_best_acc:
                    global_best_acc = use_acc
                    global_best_snap = use_snap
            
            if global_best_snap:
                per_bench_w[bn] = global_best_snap
            log.info(f"  {bn}: accuracy = {global_best_acc:.1%} ({len(data)} examples, "
                     f"{N_RESTARTS} restarts, theta_init={THETA}, theta_final={theta:.0f}, W_max={W_MAX})")
        
        # =====================================================================
        # Combine into C++ weight format
        # =====================================================================
        # For each algo, score on all training graphs with each per-bench model.
        # Use the per-bench model that "matches" the C++ multiplicative form best.
        #
        # Approach: compute each algo's ranking score on each training graph
        # using per-bench models, then find bias + feature_weights + bench_multipliers
        # that best reproduce those scores.
        #
        # =====================================================================
        # Save per-benchmark perceptrons as separate weight files ({type}/{bench}.json)
        # These are loaded by C++ when benchmark type hint is available, giving
        # much higher accuracy than the averaged scoreBase × multiplier model.
        #
        # Per-bench files are propagated to ALL types in the registry so that
        # C++ LoadPerceptronWeightsForFeatures can find them regardless of which
        # type the graph is matched to.
        # =====================================================================
        
        # Build per-bench weight dicts (one per benchmark)
        # Mode 5: save z-score weights directly — C++ will z-normalize
        # features before scoring.  Eliminates de-normalization weight
        # explosion (density 1/std=10k → millions-scale weights).
        per_bench_dicts = {}  # bn -> {algo: weights}
        for bn, bn_weights in per_bench_w.items():
            per_bench_cpp = {}
            for base in base_algos:
                if base not in bn_weights:
                    continue
                bw_raw = bn_weights[base]
                # Save z-score weights directly (all within [-W_MAX, W_MAX])
                entry = {'bias': bw_raw['bias']}
                entry.update({weight_keys[i]: bw_raw['w'][i] for i in range(n_feat)})
                # No benchmark_weights needed - this IS the benchmark-specific perceptron
                entry['benchmark_weights'] = {}
                entry['_metadata'] = {}
                # Per-bench files use the canonical variant name directly
                per_bench_cpp[base] = entry
            # Shared normalization stats for C++ z-score normalization.
            # Uses estimated-modularity stats for mod-related indices
            # because C++ only has estimated_mod = min(0.9, CC*1.5).
            per_bench_cpp['_normalization'] = {
                'feat_means': [round(v, 10) for v in feat_means_denorm],
                'feat_stds': [round(v, 10) for v in feat_stds_denorm],
                'weight_keys': list(weight_keys),
            }
            per_bench_dicts[bn] = per_bench_cpp
        
        # Save per-bench files to ALL types in the registry
        if weights_dir and per_bench_dicts:
            load_type_registry(weights_dir)
            all_types = list(_type_registry.keys()) if _type_registry else ['type_0']
            if 'type_0' not in all_types:
                all_types.insert(0, 'type_0')
            for type_name in all_types:
                for bn, per_bench_cpp in per_bench_dicts.items():
                    if not per_bench_cpp:
                        continue
                    bench_file = weights_bench_path(type_name, bn, weights_dir)
                    os.makedirs(os.path.dirname(bench_file), exist_ok=True)
                    with open(bench_file, 'w') as f:
                        json.dump(per_bench_cpp, f, indent=2)
                log.info(f"  Saved per-benchmark weights: {type_name}/ ({len(per_bench_dicts)} benchmarks, "
                         f"{len(next(iter(per_bench_dicts.values())))} algorithms)")
        
        # =====================================================================
        # ScoreBase from averaged per-bench perceptrons + regret-aware grid search
        # =====================================================================
        
        base_weights = {}
        bench_multipliers = {}
        
        # Build scoreBase from averaged per-bench perceptrons
        for base in base_algos:
            avg_bias = 0.0
            avg_w = [0.0] * n_feat
            n_counted = 0
            
            for bn in bench_names:
                if bn not in per_bench_w:
                    continue
                if base not in per_bench_w[bn]:
                    continue  # this algo wasn't an active class for this bench
                n_counted += 1
                avg_bias += per_bench_w[bn][base]['bias']
                for i in range(n_feat):
                    avg_w[i] += per_bench_w[bn][base]['w'][i]
            
            if n_counted > 0:
                avg_bias /= n_counted
                avg_w = [w / n_counted for w in avg_w]
            
            # Mode 5: save z-score weights directly (no de-normalization)
            base_weights[base] = {'bias': avg_bias}
            base_weights[base].update({weight_keys[i]: avg_w[i] for i in range(n_feat)})
            bench_multipliers[base] = {bn: 1.0 for bn in bench_names}
        
        # Helper: compute raw feature vector for a graph
        def _make_fv(feats):
            return [
                feats['modularity'],
                feats['degree_variance'],
                feats['hub_concentration'],
                feats['log_nodes'],
                feats['log_edges'],
                feats['density'],
                feats['avg_degree'] / 100.0,
                feats['clustering_coefficient'],
                feats['avg_path_length'] / 10.0,
                feats['diameter'] / 50.0,
                math.log10(feats['community_count'] + 1) if feats['community_count'] > 0 else 0,
                # IISWC'18 / GoGraph / P-OPT locality features
                feats['packing_factor'],
                feats['forward_edge_fraction'],
                feats['log_working_set_ratio'],
                # Quadratic interaction terms
                feats['dv_x_hub'],
                feats['mod_x_logn'],
                feats['pf_x_wsr'],
            ]
        
        # Helper: compute scoreBase for an algo on a graph
        # Mode 5: z-normalize raw features before dot product with z-score weights
        def _score_base(bw, fv):
            s = bw['bias']
            for i in range(n_feat):
                z = (fv[i] - feat_means_denorm[i]) / feat_stds_denorm[i]
                s += bw.get(weight_keys[i], 0) * z
            return s
        
        # Build ground truth + timing data for all benchmarks
        bench_truth_all = {}  # bench -> {graph -> best_base}
        bench_times_all = {}  # bench -> {graph -> {base: best_time}}
        bench_best_time_all = {}  # bench -> {graph -> best_time}
        graph_fvs = {}  # graph -> feature vector
        
        for gn, benchmarks_data in results_by_graph.items():
            if gn not in graph_features:
                continue
            graph_fvs[gn] = _make_fv(graph_features[gn])
            
            for bn, results in benchmarks_data.items():
                if not results:
                    continue
                best = min(results, key=lambda r: r.time_seconds)
                
                if bn not in bench_truth_all:
                    bench_truth_all[bn] = {}
                    bench_times_all[bn] = {}
                    bench_best_time_all[bn] = {}
                
                bench_truth_all[bn][gn] = best.algorithm
                bench_best_time_all[bn][gn] = best.time_seconds
                
                algo_t = {}
                for r in results:
                    a = r.algorithm
                    if a not in algo_t or r.time_seconds < algo_t[a]:
                        algo_t[a] = r.time_seconds
                bench_times_all[bn][gn] = algo_t
        
        graph_names_list = sorted(graph_fvs.keys())
        
        # Precompute scoreBase matrix
        score_base_matrix = {}  # (base, graph) -> score
        for base in base_algos:
            bw = base_weights[base]
            for gn in graph_names_list:
                score_base_matrix[(base, gn)] = _score_base(bw, graph_fvs[gn])
        
        MULT_GRID = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4,
                     0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                     1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0,
                     3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0,
                     12.0, 15.0, 20.0]
        
        # Regret-aware grid search for multipliers
        for bn in bench_names:
            if bn not in bench_truth_all:
                continue
            
            current_mults = {base: 1.0 for base in base_algos}
            
            def _eval_bench(mults):
                """(accuracy, -mean_regret) for tie-breaking."""
                correct = 0
                total_regret = 0.0
                for gn in bench_truth_all[bn]:
                    scores = {}
                    for base in base_algos:
                        scores[base] = score_base_matrix.get((base, gn), 0) * mults[base]
                    pred = max(scores, key=scores.get)
                    if pred == bench_truth_all[bn][gn]:
                        correct += 1
                    else:
                        bt = bench_best_time_all[bn].get(gn, 1.0)
                        pt = bench_times_all[bn].get(gn, {}).get(pred, bt * 10)
                        total_regret += min((pt - bt) / max(bt, 1e-6), 50.0)
                n = max(len(bench_truth_all[bn]), 1)
                return (correct, -total_regret / n)
            
            for _iteration in range(30):
                improved = False
                for opt_base in base_algos:
                    best_mult = current_mults[opt_base]
                    best_score = _eval_bench(current_mults)
                    
                    for trial_mult in MULT_GRID:
                        old_mult = current_mults[opt_base]
                        current_mults[opt_base] = trial_mult
                        trial_score = _eval_bench(current_mults)
                        if trial_score > best_score:
                            best_score = trial_score
                            best_mult = trial_mult
                            improved = True
                        current_mults[opt_base] = old_mult
                    
                    current_mults[opt_base] = best_mult
                
                if not improved:
                    break
            
            for base in base_algos:
                bench_multipliers[base][bn] = round(current_mults[base], 4)
            
            # Log
            correct = 0
            for gn in bench_truth_all[bn]:
                scores = {}
                for base in base_algos:
                    scores[base] = score_base_matrix.get((base, gn), 0) * current_mults[base]
                pred = max(scores, key=scores.get)
                if pred == bench_truth_all[bn][gn]:
                    correct += 1
            total_bn = len(bench_truth_all[bn])
            log.info(f"  {bn}: mult-opt accuracy = {correct}/{total_bn} "
                     f"= {correct/total_bn:.1%}")
        
        log.info("Per-benchmark perceptron → C++ weights: "
                 f"{len(bench_names)} benchmarks × {len(base_algos)} algorithms")
        
        # Apply trained weights directly to each algorithm (variant-level training)
        # Each canonical variant name has its own perceptron weight vector
        for algo in weights:
            if algo.startswith('_'):
                continue
            if algo in base_weights:
                bw = base_weights[algo]
                weights[algo]['bias'] = bw['bias']
                for wk in feat_to_weight.values():
                    weights[algo][wk] = bw.get(wk, 0)
                
                # Set benchmark_weights from per-benchmark perceptron ratios
                bm = bench_multipliers.get(algo, {})
                bw_dict = weights[algo].get('benchmark_weights', {})
                for bn in bench_names:
                    bw_dict[bn] = bm.get(bn, 1.0)
                weights[algo]['benchmark_weights'] = bw_dict
        
        # Restore metadata
        for algo in [a for a in weights if not a.startswith('_')]:
            meta = weights[algo].get('_metadata', {})
            meta['win_count'] = algo_win_count.get(algo, 0)
            meta['sample_count'] = algo_win_count.get(algo, 0)
            if algo in algo_speedups:
                meta['avg_speedup'] = sum(algo_speedups[algo]) / len(algo_speedups[algo])
            weights[algo]['_metadata'] = meta
    
    # Update reorder time weights from reorder results
    reorder_times = {}
    for r in reorder_results:
        algo = getattr(r, 'algorithm_name', None) or getattr(r, 'algorithm', '')
        if not algo:
            continue
        if algo not in reorder_times:
            reorder_times[algo] = []
        reorder_time = getattr(r, 'reorder_time', 0.0) or getattr(r, 'time_seconds', 0.0)
        if reorder_time > 0:
            reorder_times[algo].append(reorder_time)
    
    for algo, times in reorder_times.items():
        if algo in weights and times:
            avg_time = sum(times) / len(times)
            weights[algo]['w_reorder_time'] = -avg_time / WEIGHT_REORDER_TIME_NORMALIZATION
            meta = weights[algo].get('_metadata', {})
            meta['avg_reorder_time'] = avg_time
            weights[algo]['_metadata'] = meta
    
    # Save to flat file if output path specified
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(weights, f, indent=2)
    
    # =========================================================================
    # Pre-collapse variants before saving to JSON
    # =========================================================================
    # C++ ParseWeightsFromJSON collapses variants by highest bias, but with
    # discriminative feature weights, different variants may be better for
    # different graph types. To work around C++ highest-bias collapsing:
    #   For each base algorithm, score each variant on the MEAN feature vector
    #   of the training graphs, and keep only the best-scoring variant.
    #   Set that variant's bias to be highest so C++ will select it.
    
    # Compute mean features across training graphs
    if graph_features:
        feat_keys = list(next(iter(graph_features.values())).keys())
        mean_features = {}
        for fk in feat_keys:
            vals = [gf[fk] for gf in graph_features.values() if fk in gf]
            mean_features[fk] = sum(vals) / len(vals) if vals else 0.0
        
        # With string-keyed weights in C++, each variant has its own entry.
        # No collapsing needed — save all variants directly.
        # Mode 5: embed normalization stats so C++ can z-normalize features
        # Only available when per-benchmark training produced feat_means/feat_stds.
        try:
            _means = feat_means_denorm
            _stds = feat_stds_denorm
        except NameError:
            try:
                _means = list(feat_means)
                _stds = list(feat_stds)
            except NameError:
                _means = None
                _stds = None
        if _means is not None and _stds is not None:
            try:
                weights['_normalization'] = {
                    'feat_means': [round(v, 10) for v in _means],
                    'feat_stds': [round(v, 10) for v in _stds],
                    'weight_keys': list(weight_keys),
                }
            except NameError:
                pass  # weight_keys not defined — skip normalization
        save_weights_to_active_type(weights, weights_dir, type_name="type_0")
    
    return weights


def cross_validate_logo(
    benchmark_results: List,
    reorder_results: List = None,
    weights_dir: str = None,
) -> Dict:
    """
    Leave-One-Graph-Out (LOGO) cross-validation for perceptron weights.

    For each graph in the results:
      1. Train weights on all OTHER graphs (full perceptron, per-benchmark)
      2. Use trained weights + held-out graph features to predict best algorithm
      3. Compare prediction to actual best algorithm (oracle by total time)

    This measures generalization accuracy: how well the model predicts on
    graphs it has never seen during training.

    Args:
        benchmark_results: List of BenchmarkResult objects
        reorder_results: Optional list of ReorderResult objects
        weights_dir: Weights directory (used for type registry and graph props)

    Returns:
        Dict with keys: accuracy, per_graph, overfitting_score, regret metrics
    """
    import tempfile
    from .features import load_graph_properties_cache
    from scripts.lib.core.utils import RESULTS_DIR

    reorder_results = reorder_results or []
    if weights_dir is None:
        weights_dir = DEFAULT_WEIGHTS_DIR

    # ---- Scoring helpers: use module-level SSO functions ----
    # _build_logo_features, _make_logo_score_fv, _logo_score, _LOGO_WEIGHT_KEYS

    def _predict_top_k(scoring_algos, feats, k, norm_data=None):
        """Return top-K algorithms sorted by descending score."""
        scores = []
        for algo, data in scoring_algos.items():
            if algo.startswith('_'):
                continue
            s = _logo_score(data, feats, norm_data)
            scores.append((algo, s))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]

    # Load graph properties for feature-based scoring
    graph_props = load_graph_properties_cache(RESULTS_DIR)

    # Collect graphs and group results
    graphs = set()
    for r in benchmark_results:
        if r.success and r.time_seconds > 0:
            graphs.add(r.graph)
    graphs = sorted(graphs)

    if len(graphs) < 3:
        return {'accuracy': 0.0, 'per_graph': {}, 'overfitting_score': 0.0,
                'error': 'Need at least 3 graphs for LOGO validation'}

    # Build per-(graph, bench) result lists using total time
    # Filter to perceptron candidates so oracle matches what the model can predict
    candidates = get_perceptron_candidates()
    gb_results = {}
    for r in benchmark_results:
        if r.success and r.time_seconds > 0 and r.algorithm in candidates:
            key = (r.graph, r.benchmark)
            if key not in gb_results:
                gb_results[key] = {}
            total_time = r.time_seconds + r.reorder_time
            algo = r.algorithm
            # Deduplicate: keep minimum total time per algorithm
            if algo not in gb_results[key] or total_time < gb_results[key][algo]:
                gb_results[key][algo] = total_time
    # Convert to list-of-tuples format for downstream consumption
    gb_results = {k: list(v.items()) for k, v in gb_results.items()}

    correct = 0
    total = 0
    regrets = []
    per_graph = {}
    per_bench_names = list(EXPERIMENT_BENCHMARKS)

    # Top-K tracking (K = 1, 2, 3)
    TOP_K_VALUES = [1, 2, 3]
    topk_correct = {k: 0 for k in TOP_K_VALUES}
    topk_regrets = {k: [] for k in TOP_K_VALUES}
    log.info(f"LOGO: {len(graphs)} graphs, {len(gb_results)} (graph, bench) tasks")

    for held_out in graphs:
        # Train on all graphs except held_out
        train_results = [r for r in benchmark_results if r.graph != held_out]
        train_reorder = [r for r in reorder_results
                         if getattr(r, 'graph', '') != held_out]

        with tempfile.TemporaryDirectory() as tmpdir:
            compute_weights_from_results(
                train_results,
                reorder_results=train_reorder,
                weights_dir=tmpdir,
            )

            # Load the produced weight files (per-benchmark when available)
            type0_file = weights_type_path('type_0', tmpdir)
            if not os.path.isfile(type0_file):
                continue
            with open(type0_file) as f:
                type0_data = json.load(f)
            saved_algos = {k: v for k, v in type0_data.items()
                           if not k.startswith('_')}
            # Mode 5: extract normalization data if present
            norm_data = type0_data.get('_normalization', None)

            per_bench_weights = {}
            per_bench_norm = {}  # per-bench normalization data
            for bn in per_bench_names:
                bfile = weights_bench_path('type_0', bn, tmpdir)
                if os.path.isfile(bfile):
                    with open(bfile) as f:
                        bdata = json.load(f)
                    per_bench_weights[bn] = bdata
                    # Per-bench files may have their own _normalization
                    per_bench_norm[bn] = bdata.get('_normalization', norm_data)

        # Build features for held-out graph
        if held_out not in graph_props:
            continue
        feats = _build_logo_features(graph_props[held_out])

        graph_correct = 0
        graph_total = 0
        graph_regrets = []

        for bench in sorted(set(b for (g, b) in gb_results if g == held_out)):
            key = (held_out, bench)
            if key not in gb_results:
                continue
            algo_times = gb_results[key]

            # Predict with feature-based scoring (per-bench weights preferred)
            scoring_algos = per_bench_weights.get(bench, saved_algos)
            bench_norm = per_bench_norm.get(bench, norm_data)

            # Get top-K predictions (max K we need)
            max_k = max(TOP_K_VALUES)
            top_k_preds = _predict_top_k(scoring_algos, feats, max_k, bench_norm)
            predicted_algo = top_k_preds[0][0] if top_k_preds else None

            # Ground truth: fastest algorithm by total time
            sorted_times = sorted(algo_times, key=lambda x: x[1])
            actual_algo = sorted_times[0][0]
            best_time = sorted_times[0][1]
            algo_time_map = {a: t for a, t in algo_times}
            # Median time for fallback when predicted algo has no data
            median_time = sorted_times[len(sorted_times) // 2][1]

            # Top-1 accuracy (same as before)
            is_correct = (predicted_algo or '') == actual_algo
            if is_correct:
                graph_correct += 1
                correct += 1
            total += 1
            graph_total += 1

            # Top-K accuracy and regret
            for k in TOP_K_VALUES:
                top_k_names = {p[0] for p in top_k_preds[:k]}
                if actual_algo in top_k_names:
                    topk_correct[k] += 1
                # Top-K regret: time of best candidate in top-K that we have data for
                best_topk_time = None
                for pname, _ in top_k_preds[:k]:
                    if pname in algo_time_map:
                        t = algo_time_map[pname]
                        if best_topk_time is None or t < best_topk_time:
                            best_topk_time = t
                if best_topk_time is None:
                    best_topk_time = median_time  # median fallback
                if best_time > 0:
                    topk_regrets[k].append(
                        (best_topk_time - best_time) / best_time * 100)

            # Top-1 regret (backward compat)
            pred_time = algo_time_map.get(predicted_algo)
            if pred_time is None:
                pred_time = median_time  # median fallback

            if best_time > 0:
                r = (pred_time - best_time) / best_time * 100
                regrets.append(r)
                graph_regrets.append(r)

        per_graph[held_out] = {
            'correct': graph_correct,
            'total': graph_total,
            'accuracy': graph_correct / graph_total if graph_total > 0 else 0.0,
            'avg_regret': sum(graph_regrets) / len(graph_regrets) if graph_regrets else 0.0,
        }

    logo_accuracy = correct / total if total > 0 else 0.0
    avg_regret = sum(regrets) / len(regrets) if regrets else 0.0
    sorted_regrets = sorted(regrets) if regrets else [0.0]
    median_regret = sorted_regrets[len(sorted_regrets) // 2]

    # Compare to in-sample accuracy (full training on all graphs)
    compute_weights_from_results(
        benchmark_results, reorder_results=reorder_results, weights_dir=weights_dir
    )
    # Reload produced files for scoring
    type0_file = weights_type_path('type_0', weights_dir)
    full_algos = {}
    full_per_bench = {}
    full_norm_data = None
    if os.path.isfile(type0_file):
        with open(type0_file) as f:
            type0_data = json.load(f)
        full_algos = {k: v for k, v in type0_data.items()
                      if not k.startswith('_')}
        full_norm_data = type0_data.get('_normalization', None)
    full_per_bench_norm = {}
    for bn in per_bench_names:
        bfile = weights_bench_path('type_0', bn, weights_dir)
        if os.path.isfile(bfile):
            with open(bfile) as f:
                bdata = json.load(f)
            full_per_bench[bn] = bdata
            full_per_bench_norm[bn] = bdata.get('_normalization', full_norm_data)

    full_correct = 0
    full_total = 0
    for (graph_name, bench), algo_times in gb_results.items():
        if graph_name not in graph_props:
            continue
        feats = _build_logo_features(graph_props[graph_name])
        scoring_algos = full_per_bench.get(bench, full_algos)
        bench_norm = full_per_bench_norm.get(bench, full_norm_data)
        top_preds = _predict_top_k(scoring_algos, feats, 1, bench_norm)
        predicted = top_preds[0][0] if top_preds else None
        actual = sorted(algo_times, key=lambda x: x[1])[0][0]
        if (predicted or '') == actual:
            full_correct += 1
        full_total += 1

    full_accuracy = full_correct / full_total if full_total > 0 else 0.0
    overfitting_score = full_accuracy - logo_accuracy

    # Compute top-K metrics
    topk_metrics = {}
    for k in TOP_K_VALUES:
        k_acc = topk_correct[k] / total if total > 0 else 0.0
        k_regs = topk_regrets[k]
        k_avg_reg = sum(k_regs) / len(k_regs) if k_regs else 0.0
        k_sorted = sorted(k_regs) if k_regs else [0.0]
        k_med_reg = k_sorted[len(k_sorted) // 2]
        topk_metrics[f'top_{k}'] = {
            'accuracy': k_acc,
            'avg_regret': k_avg_reg,
            'median_regret': k_med_reg,
            'correct': topk_correct[k],
        }

    return {
        'accuracy': logo_accuracy,
        'full_training_accuracy': full_accuracy,
        'overfitting_score': overfitting_score,
        'avg_regret': avg_regret,
        'median_regret': median_regret,
        'num_graphs': len(graphs),
        'correct': correct,
        'total': total,
        'per_graph': per_graph,
        'topk': topk_metrics,
        'warning': 'Possible overfitting' if overfitting_score > 0.2 else 'OK',
    }


def cross_validate_logo_grouped(
    benchmark_results: List,
    reorder_results: List = None,
    weights_dir: str = None,
    within_group_k: int = 2,
) -> Dict:
    """
    Mode 4+3: Group prediction + within-group Top-K LOGO evaluation.

    Two-stage prediction:
      1. Predict algorithm GROUP (3-class: COMMUNITY, BANDWIDTH, CACHE)
         using majority vote of per-algorithm scores within each group.
      2. Within the predicted group, rank algorithms by perceptron score
         and select top-K candidates.

    Measures:
      - Group accuracy: how often the predicted group contains the oracle
      - Final accuracy: how often the oracle algorithm is in the top-K
        within the predicted group
      - Regret: time penalty of the best candidate in the predicted group's
        top-K vs the true oracle

    Args:
        benchmark_results: List of BenchmarkResult objects
        reorder_results: Optional list of ReorderResult objects
        weights_dir: Weights directory
        within_group_k: K for within-group ranking (default 2)

    Returns:
        Dict with group_accuracy, final_accuracy, regret, per_group stats
    """
    import tempfile
    from .features import load_graph_properties_cache
    from scripts.lib.core.utils import RESULTS_DIR

    reorder_results = reorder_results or []
    if weights_dir is None:
        weights_dir = DEFAULT_WEIGHTS_DIR

    # Use module-level SSO helpers: _build_logo_features, _logo_score, _LOGO_WEIGHT_KEYS

    graph_props = load_graph_properties_cache(RESULTS_DIR)

    graphs = sorted(set(r.graph for r in benchmark_results
                        if r.success and r.time_seconds > 0))
    if len(graphs) < 3:
        return {'error': 'Need at least 3 graphs'}

    candidates = get_perceptron_candidates()
    gb_results = {}
    for r in benchmark_results:
        if r.success and r.time_seconds > 0 and r.algorithm in candidates:
            key = (r.graph, r.benchmark)
            if key not in gb_results:
                gb_results[key] = {}
            total_time = r.time_seconds + r.reorder_time
            algo = r.algorithm
            # Deduplicate: keep minimum total time per algorithm
            if algo not in gb_results[key] or total_time < gb_results[key][algo]:
                gb_results[key][algo] = total_time
    # Convert to list-of-tuples format
    gb_results = {k: list(v.items()) for k, v in gb_results.items()}

    per_bench_names = list(EXPERIMENT_BENCHMARKS)

    # Tracking
    group_correct = 0
    final_correct = 0
    total = 0
    regrets = []
    per_group_stats = {g: {'oracle': 0, 'predicted': 0, 'correct': 0}
                       for g in ALGO_GROUPS}

    log.info(f"LOGO-Grouped: {len(graphs)} graphs, K={within_group_k}")

    for held_out in graphs:
        train_results = [r for r in benchmark_results if r.graph != held_out]
        train_reorder = [r for r in reorder_results
                         if getattr(r, 'graph', '') != held_out]

        with tempfile.TemporaryDirectory() as tmpdir:
            compute_weights_from_results(
                train_results, reorder_results=train_reorder,
                weights_dir=tmpdir)

            type0_file = weights_type_path('type_0', tmpdir)
            if not os.path.isfile(type0_file):
                continue
            with open(type0_file) as f:
                type0_data = json.load(f)
            saved_algos = {k: v for k, v in type0_data.items()
                           if not k.startswith('_')}
            norm_data = type0_data.get('_normalization', None)

            per_bench_weights = {}
            per_bench_norm = {}
            for bn in per_bench_names:
                bfile = weights_bench_path('type_0', bn, tmpdir)
                if os.path.isfile(bfile):
                    with open(bfile) as f:
                        bdata = json.load(f)
                    per_bench_weights[bn] = bdata
                    per_bench_norm[bn] = bdata.get('_normalization', norm_data)

        if held_out not in graph_props:
            continue
        feats = _build_logo_features(graph_props[held_out])

        for bench in sorted(set(b for (g, b) in gb_results if g == held_out)):
            key = (held_out, bench)
            if key not in gb_results:
                continue
            algo_times = gb_results[key]
            algo_time_map = {a: t for a, t in algo_times}

            scoring_algos = per_bench_weights.get(bench, saved_algos)
            bench_norm = per_bench_norm.get(bench, norm_data)

            # Score all algorithms
            all_scores = []
            for algo, data in scoring_algos.items():
                if algo.startswith('_'):
                    continue
                s = _logo_score(data, feats, bench_norm)
                all_scores.append((algo, s))

            # Stage 1: Predict group by max mean score (size-normalized)
            group_scores = {}
            for gname, gmembers in ALGO_GROUPS.items():
                member_scores = [s for a, s in all_scores if a in gmembers]
                gs = sum(member_scores) / len(member_scores) if member_scores else float('-inf')
                group_scores[gname] = gs
            predicted_group = max(group_scores, key=group_scores.get)

            # Ground truth
            sorted_times = sorted(algo_times, key=lambda x: x[1])
            actual_algo = sorted_times[0][0]
            best_time = sorted_times[0][1]
            actual_group = ALGO_TO_GROUP.get(actual_algo, 'UNKNOWN')

            is_group_correct = (predicted_group == actual_group)
            if is_group_correct:
                group_correct += 1
            total += 1

            per_group_stats[actual_group]['oracle'] += 1
            per_group_stats[predicted_group]['predicted'] += 1
            if is_group_correct:
                per_group_stats[actual_group]['correct'] += 1

            # Stage 2: Within predicted group, rank and take top-K
            group_algos = [(a, s) for a, s in all_scores
                           if a in ALGO_GROUPS.get(predicted_group, set())]
            group_algos.sort(key=lambda x: -x[1])
            topk_in_group = [a for a, _ in group_algos[:within_group_k]]

            # Check if oracle is in top-K within predicted group
            if actual_algo in topk_in_group:
                final_correct += 1

            # Regret: best available time among top-K in predicted group
            best_topk_time = None
            for a in topk_in_group:
                if a in algo_time_map:
                    t = algo_time_map[a]
                    if best_topk_time is None or t < best_topk_time:
                        best_topk_time = t
            if best_topk_time is None:
                best_topk_time = sorted_times[len(sorted_times) // 2][1]  # median fallback
            if best_time > 0:
                regrets.append(
                    (best_topk_time - best_time) / best_time * 100)

    group_accuracy = group_correct / total if total > 0 else 0.0
    final_accuracy = final_correct / total if total > 0 else 0.0
    avg_regret = sum(regrets) / len(regrets) if regrets else 0.0
    sorted_regrets = sorted(regrets) if regrets else [0.0]
    median_regret = sorted_regrets[len(sorted_regrets) // 2]

    return {
        'group_accuracy': group_accuracy,
        'final_accuracy': final_accuracy,
        'avg_regret': avg_regret,
        'median_regret': median_regret,
        'group_correct': group_correct,
        'final_correct': final_correct,
        'total': total,
        'within_group_k': within_group_k,
        'per_group': per_group_stats,
    }


def save_weights_to_active_type(
    weights: Dict,
    weights_dir: str = None,
    type_name: str = "type_0",
    graphs: List[str] = None,
) -> str:
    """
    Save weights to active type-based weights directory for C++ to use.
    
    This creates/updates (DEPRECATED — C++ trains at runtime):
    - results/models/perceptron/type_N/weights.json - Legacy algorithm weights
    - results/models/perceptron/registry.json - Legacy type registry
    
    Args:
        weights: Dictionary of algorithm weights
        weights_dir: Directory to save (default: results/models/perceptron/)
        type_name: Type name (default: type_0)
        graphs: Optional list of graph names that contributed to these weights
        
    Returns:
        Path to saved type file
    """
    global _type_registry
    
    if weights_dir is None:
        weights_dir = DEFAULT_WEIGHTS_DIR
    
    os.makedirs(weights_dir, exist_ok=True)
    
    # Save weights to type file
    type_file = weights_type_path(type_name, weights_dir)
    os.makedirs(os.path.dirname(type_file), exist_ok=True)
    with open(type_file, 'w') as f:
        json.dump(weights, f, indent=2)
    
    # Update type registry
    if not _type_registry:
        load_type_registry(weights_dir)
    
    # Registry algorithms list: always include ALL known canonical variants
    # (not just those from the current training run).
    # This ensures C++ type matching sees a consistent algorithm universe.
    canonical_algos = sorted(get_all_algorithm_variant_names())
    
    # Create or update registry entry
    if type_name not in _type_registry:
        _type_registry[type_name] = {
            'centroid': [0.5] * 7,  # Default centroid
            'sample_count': 1,
            'graph_count': len(graphs) if graphs else 1,
            'algorithms': canonical_algos,
            'graphs': graphs or [],
            'created': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
        }
    else:
        _type_registry[type_name]['algorithms'] = canonical_algos
        _type_registry[type_name]['last_updated'] = datetime.now().isoformat()
        if graphs:
            existing_graphs = _type_registry[type_name].get('graphs', [])
            for g in graphs:
                if g not in existing_graphs:
                    existing_graphs.append(g)
            _type_registry[type_name]['graphs'] = existing_graphs
            _type_registry[type_name]['graph_count'] = len(existing_graphs)
    
    save_type_registry(weights_dir)
    log.info(f"Saved weights to {type_file} ({len(canonical_algos)} algorithms)")
    
    # M1: auto-regenerate unified adaptive_models.json so C++ always has
    # up-to-date model data in a single file.
    try:
        from scripts.lib.core.datastore import export_unified_models
        export_unified_models()
    except Exception as e:
        log.warning(f"export_unified_models() failed (non-fatal): {e}")
    
    return type_file


def _create_default_weight_entry() -> Dict:
    """Create a default weight entry for an algorithm."""
    return {
        'bias': 0.5,
        'w_modularity': 0.0,
        'w_log_nodes': 0.0,
        'w_log_edges': 0.0,
        'w_density': 0.0,
        'w_avg_degree': 0.0,
        'w_degree_variance': 0.0,
        'w_hub_concentration': 0.0,
        'cache_l1_impact': 0.0,
        'cache_l2_impact': 0.0,
        'cache_l3_impact': 0.0,
        'cache_dram_penalty': 0.0,
        'w_reorder_time': 0.0,
        'w_clustering_coeff': 0.0,
        'w_avg_path_length': 0.0,
        'w_diameter': 0.0,
        'w_community_count': 0.0,
        'w_packing_factor': 0.0,
        'w_forward_edge_fraction': 0.0,
        'w_working_set_ratio': 0.0,
        'w_dv_x_hub': 0.0,
        'w_mod_x_logn': 0.0,
        'w_pf_x_wsr': 0.0,
        'w_fef_convergence': 0.0,
        'benchmark_weights': {b: 1.0 for b in EXPERIMENT_BENCHMARKS},
        '_metadata': {
            'sample_count': 0,
            'avg_speedup': 1.0,
            'win_count': 0,
            'times_best': 0,
            'win_rate': 0.0,
        }
    }


# get_all_algorithm_variant_names() has moved to utils.py (SSOT).
# Re-exported via the import at top-of-file for backward compatibility.


def initialize_default_weights(weights_dir: str = DEFAULT_WEIGHTS_DIR) -> Dict:
    """Initialize default weights for all algorithms including all variants.
    
    Creates entries for every variant of every algorithm so training can
    capture performance differences between variants (e.g., GraphBrewOrder_leiden
    vs GraphBrewOrder_rabbit vs GraphBrewOrder_hubcluster).
    """
    weights = {}
    
    for name in get_all_algorithm_variant_names():
        weights[name] = _create_default_weight_entry()
    
    return weights


def store_per_graph_results(
    benchmark_results: List = None,
    cache_results: List = None,
    reorder_results: List = None,
    graphs_dir: str = None,
    data_dir: str = None,
    run_timestamp: str = None,
    create_new_run: bool = True,
) -> str:
    """
    DEPRECATED: Per-graph results are now stored by the C++ binary directly
    into the central database (results/data/benchmarks.json).

    This function is kept as a no-op for backward compatibility.
    Use ``datastore.BenchmarkStore`` to query results instead.
    """
    import warnings
    warnings.warn(
        "store_per_graph_results is deprecated. "
        "C++ now writes directly to results/data/benchmarks.json. "
        "Use datastore.BenchmarkStore to query results.",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_timestamp or ""


def update_zero_weights(
    weights_file: str = None,
    benchmark_results: List = None,
    cache_results: List = None,
    reorder_results: List = None,
    graphs_dir: str = None,
    store_per_graph: bool = True,
    weights_dir: str = None,  # Alias for weights_file (for backward compatibility)
) -> None:
    """
    Update zero/default weights with actual benchmark data.
    
    This fills in missing weight values based on empirical benchmark results.
    Useful for bootstrapping the weight system with real performance data.
    
    Args:
        weights_file: Path to weights file to update
        benchmark_results: List of BenchmarkResult objects
        cache_results: Optional list of CacheResult objects
        reorder_results: Optional list of ReorderResult objects
        graphs_dir: Directory containing graphs (for feature extraction)
        store_per_graph: If True, also store results in per-graph directory structure
        weights_dir: Alias for weights_file (for backward compatibility)
    """
    # Handle alias
    if weights_dir and not weights_file:
        weights_file = weights_dir
        
    benchmark_results = benchmark_results or []
    cache_results = cache_results or []
    reorder_results = reorder_results or []
    
    # Per-graph storage is deprecated — C++ writes to benchmarks.json directly.
    # The store_per_graph parameter is retained for API compat but ignored.
    
    # Load graph properties cache for features
    from .features import load_graph_properties_cache
    graph_props = load_graph_properties_cache(graphs_dir or "results")
    
    # Load existing weights or initialize
    weights = {}
    if weights_file and os.path.exists(weights_file):
        try:
            with open(weights_file) as f:
                weights = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    
    if not weights:
        weights = initialize_default_weights()
    
    # Collect all unique algorithm names from results (includes variants)
    all_algo_names = set()
    for r in benchmark_results:
        algo = getattr(r, 'algorithm', None) or getattr(r, 'algorithm_name', '')
        if algo:
            all_algo_names.add(algo)
    for r in reorder_results:
        algo = getattr(r, 'algorithm_name', None) or getattr(r, 'algorithm', '')
        if algo:
            all_algo_names.add(algo)
    
    # Filter out chained orderings — analysis-only, not trainable (see SSOT)
    all_algo_names = {n for n in all_algo_names if not is_chained_ordering_name(n)}
    
    # Add variant algorithms that aren't in weights yet
    for algo_name in all_algo_names:
        if algo_name not in weights and not algo_name.startswith('_'):
            weights[algo_name] = _create_default_weight_entry()
    
    # Update reorder time weights from reorder results
    reorder_times = {}
    for r in reorder_results:
        algo = r.algorithm_name
        if algo not in reorder_times:
            reorder_times[algo] = []
        reorder_time = getattr(r, 'reorder_time', 0.0) or getattr(r, 'time_seconds', 0.0)
        if reorder_time > 0:
            reorder_times[algo].append(reorder_time)
    
    for algo, times in reorder_times.items():
        if algo in weights and times:
            avg_time = sum(times) / len(times)
            # Penalize slow reordering algorithms
            weights[algo]['w_reorder_time'] = -avg_time / WEIGHT_REORDER_TIME_NORMALIZATION  # Normalize
            
            # Calibrate _metadata.avg_reorder_time from actual measurements
            # This is critical for MODE_BEST_AMORTIZATION which uses
            # avg_reorder_time to compute iterations-to-amortize
            meta = weights[algo].get('_metadata', {})
            meta['avg_reorder_time'] = avg_time
            weights[algo]['_metadata'] = meta
    
    # Update cache impact weights from cache results
    # Aggregate by algorithm, then average
    cache_stats = {}
    for r in cache_results:
        if not r.success:
            continue
        algo = r.algorithm_name if hasattr(r, 'algorithm_name') else str(r.algorithm_id)
        if algo not in cache_stats:
            cache_stats[algo] = {'l1': [], 'l2': [], 'l3': []}
        
        l1_miss = getattr(r, 'l1_miss_rate', 0.0)
        l2_miss = getattr(r, 'l2_miss_rate', 0.0)
        l3_miss = getattr(r, 'l3_miss_rate', 0.0)
        
        cache_stats[algo]['l1'].append(l1_miss)
        cache_stats[algo]['l2'].append(l2_miss)
        cache_stats[algo]['l3'].append(l3_miss)
    
    for algo, stats in cache_stats.items():
        if algo in weights:
            # Cache impact = (1 - miss_rate), so lower miss = higher impact
            if stats['l1']:
                avg_l1_miss = sum(stats['l1']) / len(stats['l1'])
                weights[algo]['cache_l1_impact'] = 1.0 - avg_l1_miss
            if stats['l2']:
                avg_l2_miss = sum(stats['l2']) / len(stats['l2'])
                weights[algo]['cache_l2_impact'] = 1.0 - avg_l2_miss
            if stats['l3']:
                avg_l3_miss = sum(stats['l3']) / len(stats['l3'])
                weights[algo]['cache_l3_impact'] = 1.0 - avg_l3_miss
            
            # DRAM penalty = average miss rate across all levels (normalized)
            all_misses = stats['l1'] + stats['l2'] + stats['l3']
            if all_misses:
                avg_miss = sum(all_misses) / len(all_misses)
                weights[algo]['cache_dram_penalty'] = -avg_miss * 0.5  # Penalty (negative weight)
    
    # Update feature weights from benchmark results
    # Group results by graph and benchmark to compute correlations
    if benchmark_results:
        # Collect speedups and features per algorithm
        algo_speedups = {}  # algo -> [(speedup, features), ...]
        
        for r in benchmark_results:
            if not r.success or r.time_seconds <= 0:
                continue
            
            algo = r.algorithm
            if algo not in algo_speedups:
                algo_speedups[algo] = []
            
            # Get features from graph properties cache first, then fallback to result
            graph_name = r.graph
            props = graph_props.get(graph_name, {})
            
            features = {
                'modularity': props.get('modularity', getattr(r, 'modularity', 0.5)),
                'degree_variance': props.get('degree_variance', getattr(r, 'degree_variance', 1.0)),
                'hub_concentration': props.get('hub_concentration', getattr(r, 'hub_concentration', 0.3)),
                'avg_degree': props.get('avg_degree', getattr(r, 'avg_degree', WEIGHT_AVG_DEGREE_DEFAULT)),
                'nodes': props.get('nodes', getattr(r, 'nodes', 1000)),
                'edges': props.get('edges', getattr(r, 'edges', 5000)),
                'clustering_coefficient': props.get('clustering_coefficient', 0.0),
                'avg_path_length': props.get('avg_path_length', 0.0),
                'diameter': props.get('diameter', 0.0),
                'community_count': props.get('community_count', 0.0),
                # IISWC'18 / GoGraph / P-OPT locality features
                'packing_factor': props.get('packing_factor', 0.0),
                'forward_edge_fraction': props.get('forward_edge_fraction', 0.5),
            }
            
            # Compute derived features
            nodes = features['nodes']
            edges = features['edges']
            features['log_nodes'] = math.log10(nodes + 1) if nodes > 0 else 0
            features['log_edges'] = math.log10(edges + 1) if edges > 0 else 0
            features['density'] = 2 * edges / (nodes * (nodes - 1)) if nodes > 1 else 0
            # C++ uses log2(wsr + 1.0) for working_set_ratio
            wsr = props.get('working_set_ratio', 0.0)
            features['log_working_set_ratio'] = math.log2(wsr + 1.0)
            # Quadratic interaction terms
            estimated_modularity = min(0.9, features['clustering_coefficient'] * 1.5)
            features['dv_x_hub'] = features['degree_variance'] * features['hub_concentration']
            features['mod_x_logn'] = estimated_modularity * features['log_nodes']
            features['pf_x_wsr'] = features['packing_factor'] * features['log_working_set_ratio']
            
            # Estimate speedup by comparing to baseline
            algo_speedups[algo].append({
                'time': r.time_seconds,
                'graph': r.graph,
                'features': features
            })
        
        # Find baseline times per graph for speedup calculation
        baseline_times = {}
        for algo, results in algo_speedups.items():
            if 'ORIGINAL' in algo:
                for r in results:
                    baseline_times[r['graph']] = r['time']
        
        # Update feature weights using simple correlation-based heuristics
        for algo, results in algo_speedups.items():
            if algo not in weights:
                continue
            
            # Compute average speedup for metadata
            speedups = []
            for r in results:
                graph = r['graph']
                baseline = baseline_times.get(graph, r['time'])
                if baseline > 0:
                    speedups.append(baseline / r['time'])
            
            if len(speedups) >= 2:
                avg_speedup = sum(speedups) / len(speedups)
                
                # Only update metadata — do NOT overwrite bias or feature
                # weights that were already trained by compute_weights_from_results
                # (Phase 4 perceptron SGD).  The old code set
                #   bias = min(1.5, 0.5 + (speedup-1)*0.5)
                # which capped 9+ algorithms at the same 1.5 bias, making the
                # perceptron unable to distinguish between them.
                meta = weights[algo].get('_metadata', {})
                meta['sample_count'] = max(meta.get('sample_count', 0), len(speedups))
                meta['avg_speedup'] = avg_speedup
                weights[algo]['_metadata'] = meta
    
    # Save updated weights to flat file if path specified
    if weights_file:
        os.makedirs(os.path.dirname(weights_file), exist_ok=True)
        with open(weights_file, 'w') as f:
            json.dump(weights, f, indent=2)
    
    # Always save to type_0.json in active weights directory (used by C++)
    # Get list of unique graphs from results
    graph_names = list(set(r.graph for r in benchmark_results if hasattr(r, 'graph')))
    save_weights_to_active_type(weights, DEFAULT_WEIGHTS_DIR, type_name="type_0", graphs=graph_names)



# =============================================================================
# Standalone CLI
# =============================================================================

def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="GraphBrew Weight Management Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m scripts.lib.ml.weights --list-types
    python -m scripts.lib.ml.weights --show-type type_0
    python -m scripts.lib.ml.weights --best-algo type_0 --benchmark pr
    python -m scripts.lib.ml.weights --init-type type_0
"""
    )
    
    parser.add_argument("--list-types", action="store_true",
                        help="List all known graph types")
    parser.add_argument("--show-type", help="Show summary for a type")
    parser.add_argument("--best-algo", help="Get best algorithm for a type")
    parser.add_argument("--benchmark", "-b", default="pr",
                        help="Benchmark for algorithm selection (default: pr)")
    parser.add_argument("--init-type", help="Initialize weights for a new type")
    parser.add_argument("--weights-dir", default=DEFAULT_WEIGHTS_DIR,
                        help=f"Weights directory (default: {DEFAULT_WEIGHTS_DIR})")
    
    args = parser.parse_args()
    
    if args.list_types:
        types = list_known_types(args.weights_dir)
        print(f"\nKnown Graph Types ({len(types)}):")
        print("-" * 40)
        for t in types:
            summary = get_type_summary(t, args.weights_dir)
            print(f"  {t}: {summary['num_graphs']} graphs, {len(summary['algorithms'])} algorithms")
        return
    
    if args.show_type:
        summary = get_type_summary(args.show_type, args.weights_dir)
        print(f"\nType: {summary['type_name']}")
        print(f"  Graphs: {summary['num_graphs']}")
        print(f"  Created: {summary['created']}")
        print(f"  Last updated: {summary['last_updated']}")
        print(f"  Representative features: {summary['representative_features']}")
        print(f"\n  Algorithms ({len(summary['algorithms'])}):")
        for algo, stats in sorted(summary['algorithms'].items(), key=lambda x: -x[1]['avg_speedup']):
            print(f"    {algo}: avg_speedup={stats['avg_speedup']:.3f}, wins={stats['win_count']}/{stats['sample_count']}")
        print(f"\n  Best by benchmark: {summary['best_algorithms']}")
        return
    
    if args.best_algo:
        features = {'nodes': 10000, 'edges': 50000, 'modularity': 0.5, 'degree_variance': 1.0}
        algo, score = get_best_algorithm_for_type(args.best_algo, features, args.benchmark, args.weights_dir)
        print(f"\nBest algorithm for {args.best_algo} ({args.benchmark}): {algo} (score={score:.4f})")
        return
    
    if args.init_type:
        weights = initialize_default_weights(args.weights_dir)
        save_type_weights(args.init_type, weights, args.weights_dir)
        print(f"Initialized weights for {args.init_type} with {len(weights)} algorithms")
        return
    
    parser.print_help()


if __name__ == "__main__":
    main()
