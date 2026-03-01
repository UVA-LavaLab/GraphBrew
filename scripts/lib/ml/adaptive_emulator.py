#!/usr/bin/env python3
"""
AdaptiveOrder Emulator - Fast Python implementation for weight analysis.

This emulator replicates the C++ AdaptiveOrder algorithm selection logic:
  Layer 1: Graph type matching (Euclidean distance to centroids)
  Layer 2: Algorithm selection (perceptron scoring)

Selection Modes:
  - fastest-reorder:    Minimize reordering time only
  - fastest-execution:  Minimize algorithm execution time (default)
  - best-endtoend:      Minimize (reorder_time + execution_time)
  - best-amortization:  Minimize iterations needed to amortize reordering cost

Features:
  - Toggle individual weights on/off to analyze their impact
  - Compare emulated selections against actual benchmark results
  - Fast iteration without recompiling C++

Usage:
  python3 scripts/graphbrew_experiment.py --emulator
  python3 -m scripts.lib.ml.adaptive_emulator --graph results/graphs/email-Enron/email-Enron.mtx
  python3 -m scripts.lib.ml.adaptive_emulator --all-graphs --disable-weight w_modularity
  python3 -m scripts.lib.ml.adaptive_emulator --compare-benchmark results/benchmark_*.json
  python3 -m scripts.lib.ml.adaptive_emulator --mode best-endtoend --compare-benchmark results/benchmark.json
"""

import argparse
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# =============================================================================
# Selection Model × Criterion (clean 2D architecture)
# =============================================================================

class SelectionModel(Enum):
    """Which prediction model to use (how to predict)."""
    PERCEPTRON = "perceptron"          # Multi-class perceptron with SSO-trained weights
    DECISION_TREE = "decision-tree"    # Trained DT classifier (12D features, per-benchmark)
    HYBRID = "hybrid"                  # DT structure with per-leaf perceptron weights
    KNN_DATABASE = "knn-database"      # Distance-weighted k-NN from benchmark database
    HEURISTIC = "heuristic"            # Feature-based heuristic (Python-only, experimental)
    TYPE_BENCH = "type-bench"          # Type+benchmark recommendations (Python-only, experimental)
    # ── Label granularity modes ──
    # These control how many distinct labels the model predicts:
    #   FAMILY:     7 labels  (ORIGINAL/SORT/RCM/HUBSORT/GORDER/RABBIT/LEIDEN)
    #   TOPN:       Top-N individual algorithms (default N=8)
    #   INDIVIDUAL: All ~17 individual algorithm variants
    FAMILY = "family"                  # Predict at family level (7 classes)
    TOPN = "topn"                      # Predict top-N algorithms (8 classes)
    INDIVIDUAL = "individual"          # Predict individual algorithm (all ~17)


class SelectionCriterion(Enum):
    """What metric to optimize (what to optimize)."""
    FASTEST_REORDER = "fastest-reorder"      # Minimize reordering time only
    FASTEST_EXECUTION = "fastest-execution"  # Minimize algorithm execution time
    BEST_ENDTOEND = "best-endtoend"          # Minimize (reorder_time + execution_time)
    BEST_AMORTIZATION = "best-amortization"  # Minimize iterations to amortize


# =============================================================================
# Legacy SelectionMode — kept for backward compatibility
# =============================================================================

class SelectionMode(Enum):
    """Algorithm selection optimization target (legacy, conflated enum)."""
    FASTEST_REORDER = "fastest-reorder"      # Minimize reordering time
    FASTEST_EXECUTION = "fastest-execution"  # Minimize execution time (default)
    BEST_ENDTOEND = "best-endtoend"          # Minimize reorder + execution time
    BEST_AMORTIZATION = "best-amortization"  # Minimize iterations to amortize
    DECISION_TREE = "decision-tree"          # Decision Tree classifier (C++ MODE_DECISION_TREE=4)
    HYBRID = "hybrid"                        # Hybrid DT+Perceptron Model Tree (C++ MODE_HYBRID=5)
    DATABASE = "database"                    # Database-driven kNN (mirrors C++ MODE_DATABASE=6)
    HEURISTIC = "heuristic"                  # Feature-based heuristic (Python-only, experimental)
    TYPE_BENCH = "type-bench"                # Type+benchmark recommendations (Python-only, experimental)


def decompose_selection_mode(mode: SelectionMode) -> tuple:
    """Decompose legacy SelectionMode into (SelectionModel, SelectionCriterion).

    Mirrors C++ DecomposeSelectionMode().
    """
    _MAP = {
        SelectionMode.FASTEST_REORDER:    (SelectionModel.PERCEPTRON, SelectionCriterion.FASTEST_REORDER),
        SelectionMode.FASTEST_EXECUTION:  (SelectionModel.PERCEPTRON, SelectionCriterion.FASTEST_EXECUTION),
        SelectionMode.BEST_ENDTOEND:      (SelectionModel.PERCEPTRON, SelectionCriterion.BEST_ENDTOEND),
        SelectionMode.BEST_AMORTIZATION:  (SelectionModel.PERCEPTRON, SelectionCriterion.BEST_AMORTIZATION),
        SelectionMode.DECISION_TREE:      (SelectionModel.DECISION_TREE, SelectionCriterion.FASTEST_EXECUTION),
        SelectionMode.HYBRID:             (SelectionModel.HYBRID, SelectionCriterion.FASTEST_EXECUTION),
        SelectionMode.DATABASE:           (SelectionModel.KNN_DATABASE, SelectionCriterion.FASTEST_EXECUTION),
        SelectionMode.HEURISTIC:          (SelectionModel.HEURISTIC, SelectionCriterion.FASTEST_EXECUTION),
        SelectionMode.TYPE_BENCH:         (SelectionModel.TYPE_BENCH, SelectionCriterion.FASTEST_EXECUTION),
    }
    return _MAP.get(mode, (SelectionModel.PERCEPTRON, SelectionCriterion.FASTEST_EXECUTION))


# =============================================================================
# Constants
# =============================================================================

# Path constants from SSOT (lib/core/utils.py)
from ..core.utils import (
    WEIGHTS_DIR, RESULTS_DIR, GRAPHS_DIR, PROJECT_ROOT,
    weights_registry_path, weights_type_path,
)

# All weight fields — derived from PerceptronWeight dataclass (SSO).
# This list is used by WeightConfig to enable/disable individual weights
# for ablation experiments.  If you add a field to PerceptronWeight,
# add it here too.
from .weights import PerceptronWeight as _PW
import dataclasses as _dc
WEIGHT_FIELDS = [f.name for f in _dc.fields(_PW)]

# Import ALGORITHMS from lib/core/utils.py (Single Source of Truth)
from ..core.utils import ALGORITHMS

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GraphFeatures:
    """Features extracted from a graph for type matching and scoring."""
    name: str
    path: str
    # Core features
    num_nodes: int = 0
    num_edges: int = 0
    modularity: float = 0.0
    density: float = 0.0
    avg_degree: float = 0.0
    degree_variance: float = 0.0
    hub_concentration: float = 0.0
    clustering_coeff: float = 0.0
    avg_path_length: float = 0.0
    diameter: float = 0.0
    community_count: float = 0.0
    reorder_time: float = 0.0
    packing_factor: float = 0.0
    forward_edge_fraction: float = 0.5
    working_set_ratio: float = 0.0
    vertex_significance_skewness: float = 0.0
    window_neighbor_overlap: float = 0.0
    
    @property
    def log_nodes(self) -> float:
        """log10(num_nodes + 1) — matches C++ scoreBase() in reorder_types.h."""
        return math.log10(self.num_nodes + 1) if self.num_nodes >= 0 else 0
    
    @property
    def log_edges(self) -> float:
        """log10(num_edges + 1) — matches C++ scoreBase() in reorder_types.h."""
        return math.log10(self.num_edges + 1) if self.num_edges >= 0 else 0
    
    def to_type_vector(self) -> List[float]:
        """Convert to normalized feature vector for type matching.
        
        MUST match the training format from lib/weights.py _normalize_features():
        [modularity, degree_variance, hub_concentration, avg_degree, 
         clustering_coefficient, log_nodes, log_edges]
        
        Features are normalized to [0,1] ranges:
        - modularity: [0, 1]
        - degree_variance: [0, 5]
        - hub_concentration: [0, 1]
        - avg_degree: [0, 100]
        - clustering_coefficient: [0, 1]
        - log_nodes: [3, 10]
        - log_edges: [3, 12]
        """
        def normalize(val, lo, hi):
            return max(0.0, min(1.0, (val - lo) / (hi - lo) if hi > lo else 0.5))
        
        return [
            normalize(self.modularity, 0, 1),
            normalize(self.degree_variance, 0, 5),
            normalize(self.hub_concentration, 0, 1),
            normalize(self.avg_degree, 0, 100),
            normalize(self.clustering_coeff, 0, 1),
            normalize(self.log_nodes, 3, 10),
            normalize(self.log_edges, 3, 12),
        ]
    
    def to_scoring_dict(self) -> Dict[str, float]:
        """Convert to RAW feature dict for PerceptronWeight.compute_score().
        
        Returns RAW (untransformed) features. All normalization and transform
        logic (avg_degree/100, log10(community+1), etc.) is applied inside
        PerceptronWeight.compute_score() — the Single Source of Truth.
        
        This replaces the old pre-normalized approach where transforms were
        applied here and scoring was a simple dot product. Now scoring is
        fully delegated to PerceptronWeight.compute_score() from weights.py.
        """
        return {
            "nodes": self.num_nodes,
            "edges": self.num_edges,
            "modularity": self.modularity,
            "density": self.density,
            "avg_degree": self.avg_degree,
            "degree_variance": self.degree_variance,
            "hub_concentration": self.hub_concentration,
            "clustering_coeff": self.clustering_coeff,
            "avg_path_length": self.avg_path_length,
            "diameter": self.diameter,
            "community_count": self.community_count,
            "reorder_time": self.reorder_time,
            "packing_factor": self.packing_factor,
            "forward_edge_fraction": self.forward_edge_fraction,
            "working_set_ratio": self.working_set_ratio,
        }


@dataclass
class WeightConfig:
    """Configuration for which weights are enabled/disabled."""
    enabled_weights: Dict[str, bool] = field(default_factory=dict)
    weight_multipliers: Dict[str, float] = field(default_factory=dict)
    bias_cap: Optional[float] = None  # Maximum allowed bias value
    weight_cap: Optional[float] = None  # Maximum allowed weight value (for all weights)
    normalize_bias: bool = False  # Normalize biases to [0, 1]
    normalize_scores: bool = False  # Normalize final scores to [0, 1]
    
    def __post_init__(self):
        # Default: all weights enabled with multiplier 1.0
        for w in WEIGHT_FIELDS:
            if w not in self.enabled_weights:
                self.enabled_weights[w] = True
            if w not in self.weight_multipliers:
                self.weight_multipliers[w] = 1.0
    
    def disable(self, weight_name: str):
        """Disable a specific weight."""
        self.enabled_weights[weight_name] = False
    
    def enable(self, weight_name: str):
        """Enable a specific weight."""
        self.enabled_weights[weight_name] = True
    
    def is_enabled(self, weight_name: str) -> bool:
        return self.enabled_weights.get(weight_name, True)
    
    def get_multiplier(self, weight_name: str) -> float:
        return self.weight_multipliers.get(weight_name, 1.0)
    
    def apply_cap(self, value: float) -> float:
        """Apply weight cap if configured."""
        if self.weight_cap is not None:
            return min(max(value, -self.weight_cap), self.weight_cap)
        return value


@dataclass
class EmulationResult:
    """Result of AdaptiveOrder emulation."""
    graph_name: str
    matched_type: str
    type_distance: float
    selected_algorithm: str
    algorithm_scores: Dict[str, float]
    features: GraphFeatures


# =============================================================================
# Layer 1: Type Matching
# =============================================================================

class TypeMatcher:
    """Matches graphs to types using Euclidean distance to centroids."""
    
    def __init__(self, registry_path: Path = None):
        self.registry_path = registry_path or Path(weights_registry_path(str(WEIGHTS_DIR)))
        self.registry = {}
        self.centroids = {}
        self.radii = {}  # P2 2.2: per-type OOD radius
        self._load_registry()
    
    def _load_registry(self):
        """Load type registry with centroids."""
        if not self.registry_path.exists():
            print(f"Warning: Type registry not found at {self.registry_path}")
            return
        
        with open(self.registry_path) as f:
            self.registry = json.load(f)
        
        # Extract centroids and radii from nested structure
        # Format: {"type_N": {"centroid": [...], "radius": 0.1, ...}}
        self.centroids = {}
        self.radii = {}
        for type_name, type_info in self.registry.items():
            if isinstance(type_info, dict) and "centroid" in type_info:
                self.centroids[type_name] = type_info["centroid"]
                self.radii[type_name] = type_info.get("radius", 0.0)
        
        if not self.centroids:
            print("Warning: No centroids found in registry")
        else:
            print(f"Loaded {len(self.centroids)} type centroids")
    
    def euclidean_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute Euclidean distance between two vectors."""
        if len(vec1) != len(vec2):
            # Pad shorter vector with zeros
            max_len = max(len(vec1), len(vec2))
            vec1 = vec1 + [0.0] * (max_len - len(vec1))
            vec2 = vec2 + [0.0] * (max_len - len(vec2))
        
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))
    
    def find_best_type(self, features: GraphFeatures) -> Tuple[str, float]:
        """Find the best matching type for given graph features."""
        if not self.centroids:
            return "type_0", 999.0
        
        feature_vector = features.to_type_vector()
        
        best_type = None
        best_distance = float('inf')
        
        for type_name, centroid in self.centroids.items():
            if isinstance(centroid, list):
                distance = self.euclidean_distance(feature_vector, centroid)
                if distance < best_distance:
                    best_distance = distance
                    best_type = type_name
        
        return best_type or "type_0", best_distance


# =============================================================================
# Layer 2: Algorithm Selection (Perceptron Scoring)
# =============================================================================

class AlgorithmSelector:
    """Selects the best algorithm using perceptron scoring."""
    
    def __init__(self, weights_dir: Path = None):
        self.weights_dir = weights_dir or WEIGHTS_DIR
        self.weights_cache: Dict[str, Dict] = {}
    
    def load_weights(self, type_name: str) -> Dict[str, Dict]:
        """Load weights for a specific type."""
        if type_name in self.weights_cache:
            return self.weights_cache[type_name]
        
        weights_path = Path(weights_type_path(type_name, str(self.weights_dir)))
        if not weights_path.exists():
            print(f"Warning: Weights file not found: {weights_path}")
            return {}
        
        with open(weights_path) as f:
            weights = json.load(f)
        
        self.weights_cache[type_name] = weights
        return weights
    
    def _load_regime_weights(self, regime: str, type_name: str) -> Optional[Dict[str, Dict]]:
        """P3 3.3: Load regime-specific weights for hierarchical gating.
        
        Looks for weights in <weights_dir>/<regime>/weights.json.
        Falls back to None if not found (caller uses default weights).
        
        Args:
            regime: Graph regime from _classify_graph_type() (e.g. 'sparse_hub')
            type_name: Original type name for cache key namespacing
        
        Returns:
            Regime-specific weight dict, or None if not available
        """
        cache_key = f"__regime_{regime}_{type_name}"
        if cache_key in self.weights_cache:
            return self.weights_cache[cache_key]
        
        regime_path = self.weights_dir / regime / "weights.json"
        if not regime_path.exists():
            return None
        
        try:
            with open(regime_path) as f:
                weights = json.load(f)
            self.weights_cache[cache_key] = weights
            return weights
        except (json.JSONDecodeError, IOError):
            return None
    
    def compute_score(
        self,
        algo_weights: Dict[str, float],
        features: Dict[str, float],
        config: WeightConfig,
        benchmark: str = None
    ) -> float:
        """Compute perceptron score — delegates to PerceptronWeight.compute_score().
        
        This method applies WeightConfig ablation (disable/cap/multiplier)
        by modifying the weight dict before delegating to the SSO scorer.
        
        The SSO scoring function (PerceptronWeight.compute_score in weights.py)
        handles all feature transforms (log10, /100, /50, etc.) internally.
        """
        # Apply ablation config: zero disabled weights, apply caps/multipliers
        ablated = dict(algo_weights)
        for field_name in WEIGHT_FIELDS:
            if field_name.startswith('_') or field_name == 'benchmark_weights':
                continue
            if not config.is_enabled(field_name):
                ablated[field_name] = 0.0
            else:
                val = ablated.get(field_name, 0.0)
                if isinstance(val, (int, float)):
                    val = config.apply_cap(val)
                    val *= config.get_multiplier(field_name)
                    ablated[field_name] = val
        
        # Apply bias cap if configured
        if config.bias_cap is not None and 'bias' in ablated:
            ablated['bias'] = min(ablated['bias'], config.bias_cap)
        
        # Delegate to SSO scoring function
        pw = _PW.from_dict(ablated)
        return pw.compute_score(features, benchmark or '')
    
    def select_algorithm(
        self,
        type_name: str,
        features: GraphFeatures,
        config: WeightConfig,
        benchmark: str = None,
        type_distance: float = 0.0,
        type_radius: float = 0.0,
        hierarchical: bool = False
    ) -> Tuple[str, Dict[str, float]]:
        """Select the best algorithm for given features.
        
        Args:
            type_name: Matched graph type (e.g. "type_0")
            features: Graph features for scoring
            config: Weight configuration
            benchmark: Optional benchmark name for per-benchmark multiplier
            type_distance: Euclidean distance to nearest centroid (0 = unknown)
            type_radius: P2 2.2 per-type OOD radius (p95 of training distances)
            hierarchical: P3 3.3 hierarchical gating — try regime-specific weights
        """
        # OOD guardrail: if graph is too far from known centroid,
        # predictions are unreliable — fall back to ORIGINAL.
        # P2 2.2: Use per-type radius when available, else global threshold.
        if type_distance > 0:
            if type_radius > 0:
                # Per-type OOD: distance / radius > OOD_RADIUS_RATIO → OOD
                from scripts.lib.ml.weights import OOD_RADIUS_RATIO
                if type_distance / type_radius > OOD_RADIUS_RATIO:
                    return "ORIGINAL", {}
            else:
                # Fallback: global threshold
                OOD_DISTANCE_THRESHOLD = 1.5
                if type_distance > OOD_DISTANCE_THRESHOLD:
                    return "ORIGINAL", {}
        
        # P3 3.3: Hierarchical gating — try regime-specific weights first
        if hierarchical:
            regime = self._classify_graph_type(features)
            regime_weights = self._load_regime_weights(regime, type_name)
            if regime_weights:
                weights = regime_weights
            else:
                weights = self.load_weights(type_name)
        else:
            weights = self.load_weights(type_name)
        if not weights:
            return "ORIGINAL", {}
        
        # Raw features — PerceptronWeight.compute_score() handles all transforms
        feature_dict = features.to_scoring_dict()
        
        scores = {}
        
        # First pass: collect all biases for normalization if enabled
        all_biases = {}
        if config.normalize_bias:
            for algo_name, algo_weights in weights.items():
                if algo_name.startswith("_") or not isinstance(algo_weights, dict):
                    continue
                all_biases[algo_name] = algo_weights.get("bias", 0.5)
            
            # Normalize: map to [0, 1] range
            if all_biases:
                min_bias = min(all_biases.values())
                max_bias = max(all_biases.values())
                bias_range = max_bias - min_bias
                if bias_range > 0:
                    for algo in all_biases:
                        all_biases[algo] = (all_biases[algo] - min_bias) / bias_range
        
        for algo_name, algo_weights in weights.items():
            if algo_name.startswith("_"):  # Skip metadata
                continue
            if not isinstance(algo_weights, dict):
                continue
            
            # If normalizing bias, use normalized version
            if config.normalize_bias and algo_name in all_biases:
                # Create a modified copy of weights with normalized bias
                modified_weights = dict(algo_weights)
                modified_weights["bias"] = all_biases[algo_name]
                score = self.compute_score(modified_weights, feature_dict, config, benchmark)
            else:
                score = self.compute_score(algo_weights, feature_dict, config, benchmark)
            
            scores[algo_name] = score
        
        if not scores:
            return "ORIGINAL", {}
        
        best_algo = max(scores, key=scores.get)
        
        # Margin-based ORIGINAL fallback (IISWC'18):
        # If best algorithm doesn't beat ORIGINAL by sufficient margin,
        # reordering overhead likely exceeds the benefit.
        ORIGINAL_MARGIN_THRESHOLD = 0.05
        if best_algo != "ORIGINAL" and "ORIGINAL" in scores:
            margin = scores[best_algo] - scores["ORIGINAL"]
            
            # Determine if margin is below threshold
            below_threshold = False
            
            # P1 1.4: Platt-calibrated margin threshold.
            # If Platt params are available for the selected algo, use calibrated
            # probability instead of fixed threshold.
            algo_w = weights.get(best_algo, {})
            platt_a = algo_w.get('platt_A', 0.0) if isinstance(algo_w, dict) else 0.0
            if platt_a != 0.0:
                platt_b = algo_w.get('platt_B', 0.0) if isinstance(algo_w, dict) else 0.0
                from scripts.lib.ml.weights import platt_probability
                conf = platt_probability(margin, platt_a, platt_b)
                below_threshold = (conf < 0.55)  # PLATT_CONFIDENCE_THRESHOLD
            else:
                below_threshold = (margin < ORIGINAL_MARGIN_THRESHOLD)
            
            if below_threshold:
                # P2 2.1: ε-greedy bandit exploration for low-margin cases.
                # When ADAPTIVE_BANDIT=1, with probability ε, explore by using
                # the model's top pick despite low margin.  ε decays with
                # log(evaluations) to converge toward exploitation.
                import os
                if os.environ.get('ADAPTIVE_BANDIT', '') in ('1', 'true'):
                    import random as _rng
                    if not hasattr(self, '_bandit_eval_count'):
                        self._bandit_eval_count = 0
                    self._bandit_eval_count += 1
                    
                    eps0_str = os.environ.get('ADAPTIVE_BANDIT_EPSILON', '0.1')
                    try:
                        eps0 = float(eps0_str)
                    except ValueError:
                        eps0 = 0.1
                    import math
                    eps = eps0 / (1.0 + 0.1 * math.log(self._bandit_eval_count))
                    
                    # Safety: never explore with algos whose avg_reorder_time
                    # exceeds estimated kernel time × overhead ratio
                    safe = True
                    avg_rt = algo_w.get('avg_reorder_time', 0.0) if isinstance(algo_w, dict) else 0.0
                    if avg_rt and avg_rt > 0:
                        est_kernel = feature_dict.get('num_edges', 0) * 1e-8
                        if avg_rt > est_kernel * 0.5:
                            safe = False
                    
                    if safe and _rng.random() < eps:
                        return best_algo, scores  # explore
                
                best_algo = "ORIGINAL"  # exploit
        
        # P3 3.1f: DON-Lite neural ordering override.
        # When enabled and the perceptron margin is low on a large community,
        # override with DON_LITE (the C++ side dispatches to GenerateDonLiteMapping).
        DON_LITE_MIN_COMMUNITY = 50000
        DON_LITE_MARGIN_THRESHOLD = 0.1
        if os.environ.get('ADAPTIVE_DON_LITE', '') in ('1', 'true'):
            num_nodes = feature_dict.get('num_nodes', 0)
            if num_nodes >= DON_LITE_MIN_COMMUNITY and scores:
                sorted_scores = sorted(scores.values(), reverse=True)
                margin = (sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else abs(sorted_scores[0])
                if margin < DON_LITE_MARGIN_THRESHOLD:
                    best_algo = "DON_LITE"
        
        return best_algo, scores


# =============================================================================
# Algorithm Family Mapping (mirrors C++ AlgoToFamily)
# =============================================================================

# Map fine-grained algorithm names to coarse family names.
# Must match C++ AlgoToFamily() in reorder_database.h.
# C++ has exactly 7 families: ORIGINAL, SORT, RCM, HUBSORT, GORDER, RABBIT, LEIDEN.
_ALGO_FAMILY_MAP = {
    "ORIGINAL": "ORIGINAL",
    # SORT family (C++: RANDOM|SORT → "SORT")
    "RANDOM": "SORT",
    "SORT": "SORT",
    # HUBSORT family (C++: anything with "HUB" or "DBG" → "HUBSORT")
    "HUBSORT": "HUBSORT",
    "HUBCLUSTER": "HUBSORT",
    "DBG": "HUBSORT",
    "HUBSORTDBG": "HUBSORT",
    "HUBCLUSTERDBG": "HUBSORT",
    # RCM family
    "RCM": "RCM",
    # GORDER family (C++: GORDER|CORDER → "GORDER")
    "GORDER": "GORDER",
    "CORDER": "GORDER",
    # RABBIT family
    "RABBITORDER": "RABBIT",
    # LEIDEN family (C++: Leiden|GraphBrew → "LEIDEN")
    "LeidenOrder": "LEIDEN",
    "GraphBrewOrder": "LEIDEN",
    # GoGraph family (P3 3.4)
    "GoGraphOrder": "GOGRAPH",
    "GOGRAPHORDER": "GOGRAPH",
    # Mechanism/meta — not real families
    "MAP": "MAP",
    "AdaptiveOrder": "ADAPTIVE",
}

# The 7 canonical families recognised by C++ AlgoToFamily().
CANONICAL_FAMILIES = ["ORIGINAL", "SORT", "RCM", "HUBSORT", "GORDER", "RABBIT", "LEIDEN", "GOGRAPH"]


def algo_to_family(algo_name: str) -> str:
    """Map an algorithm name to its family (mirrors C++ AlgoToFamily).

    Follows the exact C++ matching order:
    1. Direct lookup in ``_ALGO_FAMILY_MAP``
    2. Strip variant suffix (``RABBITORDER_csr`` → ``RABBITORDER``)
    3. Substring matching (HUB, RCM, GORDER, RABBIT, Leiden, GraphBrew)
    4. Compound ordering (``SORT+RABBITORDER_csr`` → family of second part)
    5. Fall back to ``ORIGINAL``
    """
    # Direct lookup
    fam = _ALGO_FAMILY_MAP.get(algo_name)
    if fam:
        return fam
    # Strip variant suffix (e.g. RABBITORDER_csr → RABBITORDER)
    base = algo_name.split("_")[0]
    fam = _ALGO_FAMILY_MAP.get(base)
    if fam:
        return fam
    # Substring matching (mirrors C++ .find() checks)
    if "RCM" in algo_name:
        return "RCM"
    if "HUB" in algo_name or "DBG" in algo_name:
        return "HUBSORT"
    if "GORDER" in algo_name or algo_name == "CORDER":
        return "GORDER"
    if "RABBIT" in algo_name:
        return "RABBIT"
    if "Leiden" in algo_name or "GraphBrew" in algo_name:
        return "LEIDEN"
    if algo_name == "RANDOM" or algo_name == "SORT":
        return "SORT"
    # Compound ordering: SORT+RABBITORDER_csr → family of the second part
    if "+" in algo_name:
        _, suffix = algo_name.split("+", 1)
        return algo_to_family(suffix)
    return "ORIGINAL"


# =============================================================================
# Layer 3: Database Selector (mirrors C++ select_for_mode / knn_algo_scores)
# =============================================================================

class DatabaseSelector:
    """Selects algorithms using the central benchmark DB via kNN.

    Mirrors the exact C++ logic in ``BenchmarkDatabase::select_for_mode()``
    and ``knn_algo_scores()`` from ``reorder_database.h``.
    This allows verifying C++ correctness from Python and predicting
    how AdaptiveOrder will behave without running the binary.
    """

    KNN_K = 5

    def __init__(self):
        from scripts.lib.core.datastore import get_benchmark_store, get_props_store
        self._bench_store = get_benchmark_store()
        self._props_store = get_props_store()

    # ---- public helpers to allow injection in tests ----

    def reload(self):
        """Reload stores from disk (after C++ writes new data)."""
        from scripts.lib.core.datastore import BenchmarkStore, GraphPropsStore, BENCHMARKS_FILE, GRAPH_PROPS_FILE
        self._bench_store = BenchmarkStore(BENCHMARKS_FILE)
        self._props_store = GraphPropsStore(GRAPH_PROPS_FILE)

    # ---- feature vector (14 elements, same order as C++ GraphFeatureVec) ----

    @staticmethod
    def make_feature_vec(props: Dict) -> List[float]:
        """Build 14-element feature vector from raw graph properties.

        Transforms match ``GraphFeatureVec`` construction in C++
        ``reorder_database.h``.
        """
        clustering = props.get('clustering_coefficient',
                               props.get('clustering_coeff', 0.0))
        diameter = props.get('diameter', props.get('diameter_estimate', 0.0))
        return [
            props.get('modularity', 0.0),
            props.get('hub_concentration', 0.0),
            math.log10(props.get('nodes', 1) + 1),          # log_nodes
            math.log10(props.get('edges', 1) + 1),          # log_edges
            props.get('density', 0.0),
            props.get('avg_degree', 0.0) / 100.0,           # avg_degree_100
            clustering,
            props.get('packing_factor', 0.0),
            props.get('forward_edge_fraction', 0.0),
            math.log2(props.get('working_set_ratio', 0.0) + 1.0),  # log2_wsr
            math.log10(props.get('community_count', 1) + 1),       # log10_cc
            diameter / 50.0,                                        # diameter_50
            props.get('vertex_significance_skewness', 0.0),         # DON-RL
            props.get('window_neighbor_overlap', 0.0),              # DON-RL
        ]

    # ---- Euclidean distance ----

    @staticmethod
    def _euclidean(a: List[float], b: List[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    # ---- Oracle (exact-match) lookup ----

    def oracle_lookup_individual(
        self, graph_name: str, benchmark: str
    ) -> Optional[Tuple[str, Dict[str, float]]]:
        """Return the oracle-best individual algorithm for *graph_name*.

        Like ``oracle_lookup`` but returns individual algorithm names
        instead of family names.
        """
        records = self._bench_store.query(graph=graph_name, benchmark=benchmark)
        if not records:
            return None
        algo_times: Dict[str, float] = {}
        for r in records:
            a = r.get('algorithm', '')
            t = r.get('time_seconds', float('inf'))
            if a not in algo_times or t < algo_times[a]:
                algo_times[a] = t
        if not algo_times:
            return None
        best = min(algo_times, key=algo_times.get)
        return best, algo_times

    def knn_scores_individual(
        self, query_features: Dict, benchmark: str, k: int = None,
    ) -> List[Dict]:
        """Per-algorithm weighted kNN scores (individual granularity).

        Same as ``knn_scores()`` but does NOT aggregate by family —
        each algorithm variant gets its own score entry.
        """
        k = k or self.KNN_K
        query_vec = self.make_feature_vec(query_features)

        all_props = self._props_store.all()
        if not all_props:
            return []

        neighbors: List[Tuple[float, str]] = []
        for gname, gprops in all_props.items():
            gvec = self.make_feature_vec(gprops)
            dist = self._euclidean(query_vec, gvec)
            neighbors.append((dist, gname))

        neighbors.sort(key=lambda x: x[0])
        actual_k = min(k, len(neighbors))
        nearest = neighbors[:actual_k]

        algo_kernel: Dict[str, float] = defaultdict(float)
        algo_reorder: Dict[str, float] = defaultdict(float)
        algo_weight: Dict[str, float] = defaultdict(float)
        algo_count: Dict[str, int] = defaultdict(int)

        for dist, gname in nearest:
            w = 1.0 / (dist + 1e-8)
            records = self._bench_store.query(graph=gname, benchmark=benchmark)
            for r in records:
                a = r.get('algorithm', '')
                t = r.get('time_seconds', float('inf'))
                rt = r.get('reorder_time', 0.0) or 0.0
                algo_kernel[a] += w * t
                algo_reorder[a] += w * rt
                algo_weight[a] += w
                algo_count[a] += 1

        scores = []
        for a in algo_kernel:
            wsum = algo_weight[a]
            if wsum <= 0:
                continue
            scores.append({
                'algorithm': a,
                'family': algo_to_family(a),
                'avg_kernel_time': algo_kernel[a] / wsum,
                'avg_reorder_time': algo_reorder[a] / wsum,
                'vote_weight': wsum,
                'vote_count': algo_count[a],
            })

        scores.sort(key=lambda s: s['avg_kernel_time'])
        return scores

    def select_individual(
        self,
        features: Dict,
        benchmark: str,
        graph_name: str = None,
        *,
        criterion: 'SelectionCriterion' = None,
        top_n: int = 0,
    ) -> Optional[Tuple[str, Dict[str, float]]]:
        """Select best individual algorithm (or top-N subset).

        When ``top_n > 0``, only the top-N fastest algorithms (by kernel
        time across all graphs) are considered candidates.

        Returns ``(algorithm, scores_dict)`` or ``None`` if DB is empty.
        """
        if criterion is None:
            criterion = SelectionCriterion.FASTEST_EXECUTION

        # Oracle shortcut
        if graph_name and criterion != SelectionCriterion.FASTEST_REORDER:
            oracle = self.oracle_lookup_individual(graph_name, benchmark)
            if oracle is not None:
                algo, algo_times = oracle
                if top_n > 0:
                    # Filter to top-N candidates
                    top_algos = self._top_n_algorithms(benchmark, top_n)
                    filtered = {a: t for a, t in algo_times.items() if a in top_algos}
                    if filtered:
                        algo = min(filtered, key=filtered.get)
                        return algo, filtered
                return algo, algo_times

        # kNN
        scores = self.knn_scores_individual(features, benchmark)
        if not scores:
            return None

        # Filter to top-N if requested
        if top_n > 0:
            top_algos = self._top_n_algorithms(benchmark, top_n)
            scores = [s for s in scores if s['algorithm'] in top_algos]
            if not scores:
                return None

        # Criterion dispatch
        if criterion == SelectionCriterion.FASTEST_REORDER:
            by_reorder = sorted(scores, key=lambda s: s['avg_reorder_time'])
            winner = by_reorder[0]
            return winner['algorithm'], {s['algorithm']: s['avg_reorder_time'] for s in scores}

        if criterion == SelectionCriterion.FASTEST_EXECUTION:
            winner = scores[0]
            return winner['algorithm'], {s['algorithm']: s['avg_kernel_time'] for s in scores}

        if criterion == SelectionCriterion.BEST_ENDTOEND:
            total = [(s, s['avg_kernel_time'] + s['avg_reorder_time']) for s in scores]
            total.sort(key=lambda x: x[1])
            winner = total[0][0]
            return winner['algorithm'], {s['algorithm']: s['avg_kernel_time'] + s['avg_reorder_time'] for s in scores}

        if criterion == SelectionCriterion.BEST_AMORTIZATION:
            orig_time = None
            for s in scores:
                if s['algorithm'] == 'ORIGINAL':
                    orig_time = s['avg_kernel_time']
                    break
            if orig_time is None:
                orig_time = max(s['avg_kernel_time'] for s in scores)
            best_algo = None
            best_val = float('inf')
            amort_dict: Dict[str, float] = {}
            for s in scores:
                saving = orig_time - s['avg_kernel_time']
                if saving <= 0 or s['algorithm'] == 'ORIGINAL':
                    amort_dict[s['algorithm']] = float('inf')
                    continue
                iters = s['avg_reorder_time'] / saving
                amort_dict[s['algorithm']] = iters
                if iters < best_val:
                    best_val = iters
                    best_algo = s['algorithm']
            if best_algo is None:
                return 'ORIGINAL', amort_dict
            return best_algo, amort_dict

        winner = scores[0]
        return winner['algorithm'], {s['algorithm']: s['avg_kernel_time'] for s in scores}

    def _top_n_algorithms(self, benchmark: str, n: int) -> set:
        """Return the set of top-N individual algorithms by average kernel
        time across all graphs in the DB."""
        perf = self._bench_store.perf_matrix()
        algo_sums: Dict[str, float] = defaultdict(float)
        algo_counts: Dict[str, int] = defaultdict(int)
        for g, algos in perf.items():
            for a, benches in algos.items():
                if benchmark in benches:
                    algo_sums[a] += benches[benchmark]
                    algo_counts[a] += 1
        # Average
        algo_avg = {}
        for a in algo_sums:
            if algo_counts[a] > 0:
                algo_avg[a] = algo_sums[a] / algo_counts[a]
        # Top N
        sorted_algos = sorted(algo_avg, key=algo_avg.get)
        return set(sorted_algos[:n])

    def oracle_lookup(
        self, graph_name: str, benchmark: str
    ) -> Optional[Tuple[str, Dict[str, float]]]:
        """Return the oracle-best family for *graph_name* if it is in the DB.

        Returns ``(family, {family: time})`` or ``None`` if graph is unknown.
        """
        records = self._bench_store.query(graph=graph_name, benchmark=benchmark)
        if not records:
            return None

        family_times: Dict[str, float] = {}
        for r in records:
            fam = algo_to_family(r.get('algorithm', ''))
            t = r.get('time_seconds', float('inf'))
            if fam not in family_times or t < family_times[fam]:
                family_times[fam] = t

        if not family_times:
            return None
        best_fam = min(family_times, key=family_times.get)
        return best_fam, family_times

    # ---- kNN scoring (mirrors C++ knn_algo_scores) ----

    def knn_scores(
        self, query_features: Dict, benchmark: str, k: int = None,
    ) -> List[Dict]:
        """Compute per-family weighted kNN scores.

        Args:
            query_features: Raw graph property dict (nodes, edges, modularity, …).
            benchmark: Benchmark name (e.g. ``"pr"``).
            k: Number of neighbors (default 5, matching C++).

        Returns:
            Sorted list of dicts ``{family, avg_kernel_time, avg_reorder_time,
            vote_weight, vote_count}`` — lowest ``avg_kernel_time`` first.
        """
        k = k or self.KNN_K
        query_vec = self.make_feature_vec(query_features)

        # Compute distance to every graph in the props store
        all_props = self._props_store.all()
        if not all_props:
            return []

        neighbors: List[Tuple[float, str]] = []
        for gname, gprops in all_props.items():
            gvec = self.make_feature_vec(gprops)
            dist = self._euclidean(query_vec, gvec)
            neighbors.append((dist, gname))

        neighbors.sort(key=lambda x: x[0])
        actual_k = min(k, len(neighbors))
        nearest = neighbors[:actual_k]

        # Accumulate per-family weighted sums
        fam_kernel: Dict[str, float] = defaultdict(float)
        fam_reorder: Dict[str, float] = defaultdict(float)
        fam_weight: Dict[str, float] = defaultdict(float)
        fam_count: Dict[str, int] = defaultdict(int)

        for dist, gname in nearest:
            w = 1.0 / (dist + 1e-8)
            records = self._bench_store.query(graph=gname, benchmark=benchmark)
            for r in records:
                fam = algo_to_family(r.get('algorithm', ''))
                t = r.get('time_seconds', float('inf'))
                rt = r.get('reorder_time', 0.0) or 0.0
                fam_kernel[fam] += w * t
                fam_reorder[fam] += w * rt
                fam_weight[fam] += w
                fam_count[fam] += 1

        scores = []
        for fam in fam_kernel:
            wsum = fam_weight[fam]
            if wsum <= 0:
                continue
            scores.append({
                'family': fam,
                'avg_kernel_time': fam_kernel[fam] / wsum,
                'avg_reorder_time': fam_reorder[fam] / wsum,
                'vote_weight': wsum,
                'vote_count': fam_count[fam],
            })

        scores.sort(key=lambda s: s['avg_kernel_time'])
        return scores

    # ---- Mode-aware selection (mirrors C++ select_for_mode) ----

    def select_for_mode(
        self,
        features: Dict,
        benchmark: str,
        mode: 'SelectionMode' = None,
        graph_name: str = None,
        *,
        criterion: 'SelectionCriterion' = None,
    ) -> Optional[Tuple[str, Dict[str, float]]]:
        """Select the best algorithm family from the central DB.

        Mirrors C++ ``BenchmarkDatabase::select_for_mode()`` logic:

        1. Try oracle first (if graph exists in DB and criterion != fastest-reorder).
        2. Fall back to kNN scoring.
        3. Apply criterion-specific metric to pick the winner.

        Accepts either a legacy ``mode`` or the new ``criterion``.
        If ``criterion`` is given, it takes precedence.

        Returns ``(family, scores_dict)`` or ``None`` if DB is empty.
        """
        # Convert legacy mode to criterion if needed
        if criterion is None and mode is not None:
            _, criterion = decompose_selection_mode(mode)
        if criterion is None:
            criterion = SelectionCriterion.FASTEST_EXECUTION

        # Step 1: Oracle shortcut (direct DB match)
        if graph_name and criterion != SelectionCriterion.FASTEST_REORDER:
            oracle = self.oracle_lookup(graph_name, benchmark)
            if oracle is not None:
                fam, fam_times = oracle
                return fam, fam_times

        # Step 2: kNN scores
        scores = self.knn_scores(features, benchmark)
        if not scores:
            return None

        # Step 3: Criterion dispatch
        if criterion == SelectionCriterion.FASTEST_REORDER:
            # Sort by avg_reorder_time ascending, pick lowest
            by_reorder = sorted(scores, key=lambda s: s['avg_reorder_time'])
            winner = by_reorder[0]
            return winner['family'], {s['family']: s['avg_reorder_time'] for s in scores}

        if criterion == SelectionCriterion.FASTEST_EXECUTION:
            # Already sorted by avg_kernel_time
            winner = scores[0]
            return winner['family'], {s['family']: s['avg_kernel_time'] for s in scores}

        if criterion == SelectionCriterion.BEST_ENDTOEND:
            total = [(s, s['avg_kernel_time'] + s['avg_reorder_time']) for s in scores]
            total.sort(key=lambda x: x[1])
            winner = total[0][0]
            return winner['family'], {s['family']: s['avg_kernel_time'] + s['avg_reorder_time'] for s in scores}

        if criterion == SelectionCriterion.BEST_AMORTIZATION:
            # Find ORIGINAL kernel time (baseline)
            orig_time = None
            for s in scores:
                if s['family'] == 'ORIGINAL':
                    orig_time = s['avg_kernel_time']
                    break
            if orig_time is None:
                orig_time = max(s['avg_kernel_time'] for s in scores)

            best_fam = None
            best_val = float('inf')
            amort_dict: Dict[str, float] = {}
            for s in scores:
                saving = orig_time - s['avg_kernel_time']
                if saving <= 0 or s['family'] == 'ORIGINAL':
                    amort_dict[s['family']] = float('inf')
                    continue
                iters = s['avg_reorder_time'] / saving
                amort_dict[s['family']] = iters
                if iters < best_val:
                    best_val = iters
                    best_fam = s['family']

            if best_fam is None:
                return 'ORIGINAL', amort_dict
            return best_fam, amort_dict

        # Default: fastest execution
        winner = scores[0]
        return winner['family'], {s['family']: s['avg_kernel_time'] for s in scores}

    # ---- Comparison: DB vs perceptron vs oracle ----

    def compare_with_perceptron(
        self,
        emulator: 'AdaptiveOrderEmulator',
        benchmark: str = 'pr',
    ) -> Dict:
        """Compare DB-driven, perceptron, and oracle selections for all graphs.

        Returns summary with agreement rates and per-graph details.
        """
        all_props = self._props_store.all()
        details = []
        agree_db_oracle = 0
        agree_perc_oracle = 0
        agree_db_perc = 0
        total = 0

        for gname, gprops in all_props.items():
            # Oracle
            oracle = self.oracle_lookup(gname, benchmark)
            if oracle is None:
                continue
            oracle_fam, _ = oracle

            # kNN-DB
            db_result = self.select_for_mode(
                gprops, benchmark, SelectionMode.DATABASE, graph_name=None,
                criterion=SelectionCriterion.FASTEST_EXECUTION,
            )
            db_fam = db_result[0] if db_result else 'ORIGINAL'

            # Perceptron
            gf = GraphFeatures(
                name=gname, path='',
                num_nodes=gprops.get('nodes', 0),
                num_edges=gprops.get('edges', 0),
                modularity=gprops.get('modularity', 0.0),
                density=gprops.get('density', 0.0),
                avg_degree=gprops.get('avg_degree', 0.0),
                degree_variance=gprops.get('degree_variance', 0.0),
                hub_concentration=gprops.get('hub_concentration', 0.0),
                clustering_coeff=gprops.get('clustering_coefficient',
                                            gprops.get('clustering_coeff', 0.0)),
                avg_path_length=gprops.get('avg_path_length', 0.0),
                diameter=gprops.get('diameter', gprops.get('diameter_estimate', 0.0)),
                community_count=gprops.get('community_count', 0.0),
                packing_factor=gprops.get('packing_factor', 0.0),
                forward_edge_fraction=gprops.get('forward_edge_fraction', 0.5),
                working_set_ratio=gprops.get('working_set_ratio', 0.0),
            )
            perc_result = emulator.emulate(gf, benchmark)
            perc_fam = algo_to_family(perc_result.selected_algorithm)

            total += 1
            if db_fam == oracle_fam:
                agree_db_oracle += 1
            if perc_fam == oracle_fam:
                agree_perc_oracle += 1
            if db_fam == perc_fam:
                agree_db_perc += 1

            details.append({
                'graph': gname,
                'oracle': oracle_fam,
                'database': db_fam,
                'perceptron': perc_fam,
                'db_correct': db_fam == oracle_fam,
                'perc_correct': perc_fam == oracle_fam,
            })

        return {
            'total': total,
            'db_oracle_agree': agree_db_oracle,
            'perc_oracle_agree': agree_perc_oracle,
            'db_perc_agree': agree_db_perc,
            'db_accuracy': agree_db_oracle / total if total else 0,
            'perc_accuracy': agree_perc_oracle / total if total else 0,
            'details': details,
        }


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_features_from_graph(graph_path: str, bin_dir: Path = None) -> Optional[GraphFeatures]:
    """Extract features from a graph by running the benchmark binary."""
    bin_dir = bin_dir or (PROJECT_ROOT / "bench" / "bin")
    pr_binary = bin_dir / "pr"
    
    if not pr_binary.exists():
        print(f"Error: Binary not found: {pr_binary}")
        return None
    
    # Run with AdaptiveOrder to get features printed
    cmd = f"{pr_binary} -f {graph_path} -s -o 14 -n 1"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print(f"Timeout extracting features from {graph_path}")
        return None
    
    # Parse features from output
    features = GraphFeatures(
        name=Path(graph_path).stem,
        path=graph_path
    )
    
    # Parse graph size
    match = re.search(r"Graph has (\d+) nodes and (\d+) edges", output)
    if match:
        features.num_nodes = int(match.group(1))
        features.num_edges = int(match.group(2))
    
    # Parse topology features
    patterns = {
        "modularity": r"Modularity:\s+([\d.]+)",
        "clustering_coeff": r"Clustering Coefficient:\s*([\d.]+)",
        "avg_path_length": r"Avg Path Length:\s+([\d.]+)",
        "diameter": r"Diameter Estimate:\s+([\d.]+)",
        "community_count": r"Community Count Estimate:\s*([\d.]+)",
        "degree_variance": r"Degree Variance:\s+([\d.]+)",
        "hub_concentration": r"Hub Concentration:\s+([\d.]+)",
        "avg_degree": r"Avg Degree:\s+([\d.]+)",
    }
    
    for attr, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            setattr(features, attr, float(match.group(1)))
    
    # Compute density
    if features.num_nodes > 1:
        max_edges = features.num_nodes * (features.num_nodes - 1) / 2
        features.density = features.num_edges / max_edges if max_edges > 0 else 0
    
    return features


def load_cached_features(cache_path: Path = None) -> Dict[str, GraphFeatures]:
    """Load cached graph features from JSON."""
    if cache_path is None:
        from .features import get_graph_properties_cache_file
        cache_path = Path(get_graph_properties_cache_file())
    
    if not cache_path.exists():
        return {}
    
    with open(cache_path) as f:
        data = json.load(f)
    
    features = {}
    for graph_name, props in data.items():
        if isinstance(props, dict):
            features[graph_name] = GraphFeatures(
                name=graph_name,
                path=props.get("path", ""),
                num_nodes=props.get("nodes", 0),
                num_edges=props.get("edges", 0),
                modularity=props.get("modularity", 0.0),
                density=props.get("density", 0.0),
                avg_degree=props.get("avg_degree", 0.0),
                degree_variance=props.get("degree_variance", 0.0),
                hub_concentration=props.get("hub_concentration", 0.0),
                clustering_coeff=props.get("clustering_coefficient", 0.0),
                avg_path_length=props.get("avg_path_length", 0.0),
                diameter=props.get("diameter_estimate", props.get("diameter", 0.0)),
                community_count=props.get("community_count", 0.0),
                packing_factor=props.get("packing_factor", 0.0),
                forward_edge_fraction=props.get("forward_edge_fraction", 0.5),
                working_set_ratio=props.get("working_set_ratio", 0.0),
                vertex_significance_skewness=props.get("vertex_significance_skewness", 0.0),
                window_neighbor_overlap=props.get("window_neighbor_overlap", 0.0),
            )
    
    return features


# =============================================================================
# Emulator
# =============================================================================


class AdaptiveOrderEmulator:
    """Main emulator class combining type matching and algorithm selection."""
    
    def __init__(self, weights_dir: Path = None):
        self.type_matcher = TypeMatcher()
        self.algorithm_selector = AlgorithmSelector(weights_dir)
        self.config = WeightConfig()
        self.selection_mode = SelectionMode.DATABASE  # Default: DB-driven (mirrors C++)
        # New 2D defaults
        self.selection_model = SelectionModel.KNN_DATABASE
        self.selection_criterion = SelectionCriterion.FASTEST_EXECUTION
        # P3 3.3: Hierarchical gating (set via ADAPTIVE_HIERARCHICAL=1)
        self.hierarchical = os.environ.get('ADAPTIVE_HIERARCHICAL', '') in ('1', 'true')
    
    def get_reorder_time_weights(self, type_name: str) -> Dict[str, float]:
        """Get w_reorder_time weights for all algorithms from type weights.
        
        Higher (less negative) w_reorder_time = faster reordering.
        """
        weights = self.algorithm_selector.load_weights(type_name)
        reorder_weights = {}
        for algo, w in weights.items():
            if algo.startswith("_"):
                continue
            if isinstance(w, dict):
                reorder_weights[algo] = w.get("w_reorder_time", 0.0)
        return reorder_weights

    def emulate(
        self,
        features: GraphFeatures,
        benchmark: str = None,
        mode: SelectionMode = None,
        *,
        model: SelectionModel = None,
        criterion: SelectionCriterion = None,
    ) -> EmulationResult:
        """Emulate AdaptiveOrder for given graph features.

        Supports both legacy ``mode`` and new ``(model, criterion)`` API.
        If ``model`` or ``criterion`` are given they take precedence.

        **Models** (how to predict):
          PERCEPTRON, DECISION_TREE, HYBRID, KNN_DATABASE, HEURISTIC, TYPE_BENCH

        **Criteria** (what to optimize):
          FASTEST_REORDER, FASTEST_EXECUTION, BEST_ENDTOEND, BEST_AMORTIZATION
        """
        # Resolve model & criterion
        if model is None and criterion is None:
            # Legacy path: use mode
            mode = mode or self.selection_mode
            model, criterion = decompose_selection_mode(mode)
        else:
            model = model or self.selection_model
            criterion = criterion or self.selection_criterion
        
        # Layer 1: Type matching - finds the closest type centroid
        matched_type, type_distance = self.type_matcher.find_best_type(features)
        
        # P2 2.2: Get per-type OOD radius
        type_radius = self.type_matcher.radii.get(matched_type, 0.0)
        
        # Layer 2: Model × Criterion dispatch
        selected_algo, scores = self._dispatch(
            model, criterion, matched_type, features, benchmark,
            type_distance=type_distance, type_radius=type_radius,
        )
        
        return EmulationResult(
            graph_name=features.name,
            matched_type=matched_type,
            type_distance=type_distance,
            selected_algorithm=selected_algo,
            algorithm_scores=scores,
            features=features,
        )

    def _dispatch(
        self,
        model: SelectionModel,
        criterion: SelectionCriterion,
        matched_type: str,
        features: GraphFeatures,
        benchmark: str,
        type_distance: float = 0.0,
        type_radius: float = 0.0,
    ) -> Tuple[str, Dict[str, float]]:
        """Two-dimensional dispatch: model × criterion.

        Mirrors C++ SelectReorderingWithModelCriterion().
        """
        # ---- KNN_DATABASE ----
        if model == SelectionModel.KNN_DATABASE:
            db_sel = DatabaseSelector()
            raw_props = features.__dict__
            result = db_sel.select_for_mode(
                raw_props, benchmark or 'pr',
                graph_name=features.name,
                criterion=criterion,
            )
            if result is not None:
                return result
            # DB empty — fall through to perceptron

        # ---- DECISION_TREE / HYBRID ----
        if model in (SelectionModel.DECISION_TREE, SelectionModel.HYBRID):
            subdir = 'decision_tree' if model == SelectionModel.DECISION_TREE else 'hybrid'
            try:
                algo, scores = self._select_by_model_tree(features, benchmark, subdir)
                return algo, scores
            except Exception:
                pass  # fall through to perceptron

        # ---- HEURISTIC ----
        if model == SelectionModel.HEURISTIC:
            return self._select_by_heuristic(features)

        # ---- TYPE_BENCH ----
        if model == SelectionModel.TYPE_BENCH:
            return self._select_by_type_bench(features, benchmark)

        # ---- FAMILY (label granularity: 7 families via DB) ----
        if model == SelectionModel.FAMILY:
            db_sel = DatabaseSelector()
            raw_props = features.__dict__
            result = db_sel.select_for_mode(
                raw_props, benchmark or 'pr',
                graph_name=features.name,
                criterion=criterion,
            )
            if result is not None:
                return result
            # Fallback to perceptron (already family-level)
            return self._perceptron_for_criterion(
                criterion, matched_type, features, benchmark,
                type_distance=type_distance, type_radius=type_radius,
            )

        # ---- INDIVIDUAL (label granularity: all ~17 algorithms via DB) ----
        if model == SelectionModel.INDIVIDUAL:
            db_sel = DatabaseSelector()
            raw_props = features.__dict__
            result = db_sel.select_individual(
                raw_props, benchmark or 'pr',
                graph_name=features.name,
                criterion=criterion,
            )
            if result is not None:
                return result
            return self._perceptron_for_criterion(
                criterion, matched_type, features, benchmark,
                type_distance=type_distance, type_radius=type_radius,
            )

        # ---- TOPN (label granularity: top-8 algorithms via DB) ----
        if model == SelectionModel.TOPN:
            db_sel = DatabaseSelector()
            raw_props = features.__dict__
            result = db_sel.select_individual(
                raw_props, benchmark or 'pr',
                graph_name=features.name,
                criterion=criterion,
                top_n=8,
            )
            if result is not None:
                return result
            return self._perceptron_for_criterion(
                criterion, matched_type, features, benchmark,
                type_distance=type_distance, type_radius=type_radius,
            )

        # ---- PERCEPTRON (default / fallback) ----
        return self._perceptron_for_criterion(
            criterion, matched_type, features, benchmark,
            type_distance=type_distance, type_radius=type_radius,
        )

    def _perceptron_for_criterion(
        self,
        criterion: SelectionCriterion,
        matched_type: str,
        features: GraphFeatures,
        benchmark: str,
        type_distance: float = 0.0,
        type_radius: float = 0.0,
    ) -> Tuple[str, Dict[str, float]]:
        """Perceptron selection parameterized by criterion."""
        hierarchical = self.hierarchical
        if criterion == SelectionCriterion.FASTEST_REORDER:
            return self._select_fastest_reorder(matched_type, features)
        if criterion == SelectionCriterion.FASTEST_EXECUTION:
            return self.algorithm_selector.select_algorithm(
                matched_type, features, self.config, benchmark,
                type_distance=type_distance, type_radius=type_radius,
                hierarchical=hierarchical,
            )
        if criterion == SelectionCriterion.BEST_ENDTOEND:
            return self._select_best_endtoend(matched_type, features, benchmark)
        if criterion == SelectionCriterion.BEST_AMORTIZATION:
            return self._select_best_amortization(matched_type, features, benchmark)
        # Default
        return self.algorithm_selector.select_algorithm(
            matched_type, features, self.config, benchmark,
            type_distance=type_distance, type_radius=type_radius,
            hierarchical=hierarchical,
        )

    # -- Model Tree (DT / Hybrid) cache --
    _model_tree_cache: Dict = None

    def _load_model_trees(self) -> Dict:
        """Load model trees from adaptive_models.json (lazy, cached)."""
        if AdaptiveOrderEmulator._model_tree_cache is None:
            from .model_tree import load_adaptive_models
            AdaptiveOrderEmulator._model_tree_cache = load_adaptive_models()
        return AdaptiveOrderEmulator._model_tree_cache

    def _select_by_model_tree(
        self,
        features: GraphFeatures,
        benchmark: str,
        subdir: str,  # 'decision_tree' or 'hybrid'
    ) -> Tuple[str, Dict[str, float]]:
        """Select algorithm using a Decision Tree or Hybrid Model Tree.

        Mirrors C++ SelectAlgorithmModelTreeFromDB() → ModelTree::predict().
        Falls back to perceptron if no model exists for this benchmark.
        """
        from .model_tree import extract_dt_features

        models = self._load_model_trees()
        bench_models = models.get(subdir, {})
        bench = benchmark or 'pr'

        tree = bench_models.get(bench)
        if tree is None or not tree.nodes:
            # No model — fall back to perceptron (matches C++ fallback)
            matched_type, _ = self.type_matcher.find_best_type(features)
            return self.algorithm_selector.select_algorithm(
                matched_type, features, self.config, benchmark
            )

        # Build feature dict from GraphFeatures for extract_dt_features
        props = {
            'nodes': features.num_nodes,
            'edges': features.num_edges,
            'modularity': features.modularity,
            'hub_concentration': features.hub_concentration,
            'avg_degree': features.avg_degree,
            'clustering_coefficient': features.clustering_coeff,
            'packing_factor': features.packing_factor,
            'forward_edge_fraction': features.forward_edge_fraction,
            'working_set_ratio': features.working_set_ratio,
            'community_count': features.community_count,
            'diameter_estimate': features.diameter_estimate,
        }
        feats_12d = extract_dt_features(props)
        predicted = tree.predict(feats_12d)

        # Build scores dict (single winner)
        scores = {predicted: 1.0}
        return predicted, scores
    
    def _select_by_heuristic(
        self,
        features: GraphFeatures
    ) -> Tuple[str, Dict[str, float]]:
        """Select algorithm using feature-based heuristics.
        
        This is more robust than the perceptron when weights are overfitted.
        Based on empirical analysis of graph characteristics and algorithm performance.
        """
        cluster = features.clustering_coeff
        avg_deg = features.avg_degree
        hub = features.hub_concentration
        n = features.num_nodes
        
        all_algos = list(ALGORITHMS.values())
        scores = {a: 0.1 for a in all_algos}
        
        # Rule 1: Very low clustering (<0.01) - sparse graphs like p2p networks
        # These benefit from simple algorithms or hub-based clustering
        if cluster < 0.01:
            scores['RANDOM'] = 1.0
            scores['HUBCLUSTER'] = 0.95
            scores['HUBCLUSTERDBG'] = 0.93
            scores['SORT'] = 0.9
            scores['DBG'] = 0.85
            scores['GraphBrewOrder'] = 0.8
        
        # Rule 2: High clustering (>0.5) - strong community structure
        # Community-detection and hub-based algorithms work well
        elif cluster > 0.5:
            scores['DBG'] = 1.0
            scores['HUBSORT'] = 0.95
            scores['LeidenOrder'] = 0.9
            scores['GraphBrewOrder'] = 0.85
            scores['RABBITORDER'] = 0.8
        
        # Rule 3: Medium clustering (0.1-0.5) - moderate community structure
        # Leiden algorithms typically perform well
        elif cluster > 0.1:
            scores['GraphBrewOrder'] = 1.0
            scores['DBG'] = 0.9
            scores['SORT'] = 0.85
            scores['RABBITORDER'] = 0.8
        
        # Rule 4: Low-medium clustering (0.01-0.1)
        # DBG and sorting work well
        else:
            scores['DBG'] = 1.0
            scores['SORT'] = 0.95
            scores['RANDOM'] = 0.9
            scores['RCM'] = 0.85
        
        # Adjust for graph size: larger graphs benefit more from sophisticated reordering
        if n > 100000:
            # Large graphs: boost community-based algorithms
            scores['RABBITORDER'] = scores.get('RABBITORDER', 0.5) + 0.2
            scores['LeidenOrder'] = scores.get('LeidenOrder', 0.5) + 0.15
        elif n < 10000:
            # Small graphs: simple algorithms are often sufficient
            scores['SORT'] = scores.get('SORT', 0.5) + 0.1
            scores['RANDOM'] = scores.get('RANDOM', 0.5) + 0.1
        
        # Exclude GORDER and CORDER (known to have overfitted weights)
        # Keep them with low score in case user wants them
        scores['GORDER'] = 0.05
        scores['CORDER'] = 0.05
        
        best_algo = max(scores, key=scores.get)
        return best_algo, scores
    
    def _classify_graph_type(self, features: GraphFeatures) -> str:
        """Classify graph into a structural type.
        
        Types:
        - sparse_hub: Low density (avg_deg < 5) with hub concentration > 0.3
        - sparse_uniform: Low density without strong hubs
        - dense_clustered: Higher density with clustering > 0.2
        - dense_flat: Higher density without strong clustering
        """
        clustering = features.clustering_coeff
        hub_concentration = features.hub_concentration
        avg_degree = features.avg_degree
        
        is_sparse = avg_degree < 5
        is_hub_heavy = hub_concentration > 0.3
        
        if is_sparse:
            return 'sparse_hub' if is_hub_heavy else 'sparse_uniform'
        else:
            return 'dense_clustered' if clustering > 0.2 else 'dense_flat'
    
    def _select_by_type_bench(
        self,
        features: GraphFeatures,
        benchmark: str = None
    ) -> Tuple[str, Dict[str, float]]:
        """Select algorithm using type+benchmark recommendations.
        
        This approach achieves ~44% exact match and ~100% top-3 accuracy
        based on empirical analysis of algorithm performance by graph type
        and benchmark workload.
        """
        graph_type = self._classify_graph_type(features)
        
        # Load type+benchmark recommendations
        reco_path = self.algorithm_selector.weights_dir / "type_bench_recommendations.json"
        if reco_path.exists():
            with open(reco_path) as f:
                recommendations = json.load(f)
        else:
            # Fallback to hardcoded recommendations (from training)
            recommendations = {
                # Dense clustered graphs
                'dense_clustered_bc': ['RABBITORDER', 'LeidenOrder', 'GORDER'],
                'dense_clustered_bfs': ['LeidenOrder', 'GORDER', 'RABBITORDER'],
                'dense_clustered_cc': ['SORT', 'GORDER'],
                'dense_clustered_pr': ['DBG', 'GraphBrewOrder', 'LeidenOrder'],
                'dense_clustered_sssp': ['RABBITORDER', 'GORDER', 'LeidenOrder'],
                # Dense flat graphs
                'dense_flat_bc': ['RCM', 'GraphBrewOrder', 'LeidenOrder'],
                'dense_flat_bfs': ['RCM', 'RABBITORDER', 'HUBCLUSTER'],
                'dense_flat_cc': ['SORT', 'GraphBrewOrder', 'GORDER'],
                'dense_flat_pr': ['RANDOM', 'GraphBrewOrder', 'LeidenOrder'],
                'dense_flat_sssp': ['GORDER', 'RCM', 'GraphBrewOrder'],
                # Sparse hub graphs
                'sparse_hub_bc': ['RABBITORDER', 'LeidenOrder', 'CORDER'],
                'sparse_hub_bfs': ['HUBCLUSTER', 'GraphBrewOrder', 'HUBCLUSTERDBG'],
                'sparse_hub_cc': ['GraphBrewOrder', 'HUBCLUSTER', 'GORDER'],
                'sparse_hub_pr': ['RANDOM', 'DBG', 'RABBITORDER'],
                'sparse_hub_sssp': ['RABBITORDER', 'LeidenOrder', 'RCM'],
                # Sparse uniform graphs (fallback)
                'sparse_uniform_bc': ['RABBITORDER', 'RCM', 'SORT'],
                'sparse_uniform_bfs': ['RCM', 'SORT', 'RANDOM'],
                'sparse_uniform_cc': ['SORT', 'RCM', 'RANDOM'],
                'sparse_uniform_pr': ['RANDOM', 'SORT', 'DBG'],
                'sparse_uniform_sssp': ['RCM', 'SORT', 'RANDOM'],
            }
        
        # Normalize benchmark name
        bench = benchmark.lower() if benchmark else 'pr'
        if bench not in ['bc', 'bfs', 'cc', 'pr', 'sssp', 'tc']:
            bench = 'pr'  # Default to PageRank
        
        # Get recommendations for this type+benchmark
        key = f"{graph_type}_{bench}"
        algo_ranking = recommendations.get(key, ['RABBITORDER', 'GORDER', 'RCM'])
        
        # Convert to scores (higher rank = higher score)
        all_algos = list(ALGORITHMS.values())
        scores = {a: 0.1 for a in all_algos}
        
        for i, algo in enumerate(algo_ranking):
            scores[algo] = 1.0 - (i * 0.1)  # 1.0, 0.9, 0.8, ...
        
        best_algo = algo_ranking[0] if algo_ranking else 'RABBITORDER'
        return best_algo, scores
    
    def _select_fastest_reorder(
        self,
        matched_type: str,
        features: GraphFeatures
    ) -> Tuple[str, Dict[str, float]]:
        """Select algorithm with fastest reorder time using w_reorder_time weights.
        
        Higher w_reorder_time = faster reordering.
        """
        reorder_weights = self.get_reorder_time_weights(matched_type)
        
        if not reorder_weights:
            # Fall back to heuristics: simple algorithms are fastest
            fast_algorithms = ["RANDOM", "SORT", "HUBSORT", "DBG", "HUBCLUSTERDBG"]
            return fast_algorithms[0], {a: 1.0/float(i+1) for i, a in enumerate(fast_algorithms)}
        
        # Exclude ORIGINAL (no reordering)
        # Higher w_reorder_time = faster, so we want max
        reorder_scores = {}
        for algo, w_rt in reorder_weights.items():
            if algo != "ORIGINAL":
                # Convert to positive score: higher = faster
                # w_reorder_time is typically negative, so negate it
                reorder_scores[algo] = -w_rt  # More negative w_rt → lower score
        
        if not reorder_scores:
            return "RANDOM", {"RANDOM": 1.0}
        
        best_algo = max(reorder_scores, key=reorder_scores.get)
        return best_algo, reorder_scores
    
    def _select_best_endtoend(
        self,
        matched_type: str,
        features: GraphFeatures,
        benchmark: str = None
    ) -> Tuple[str, Dict[str, float]]:
        """Select algorithm with best end-to-end (reorder + execution) time.
        
        Uses perceptron scores (which already include w_reorder_time) and
        adds extra weight to w_reorder_time for end-to-end optimization.
        """
        # Get execution time scores from perceptron
        _, exec_scores = self.algorithm_selector.select_algorithm(
            matched_type, features, self.config, benchmark
        )
        
        # Get reorder time weights from type
        reorder_weights = self.get_reorder_time_weights(matched_type)
        
        # Combine: add extra boost for fast reordering
        REORDER_WEIGHT_BOOST = 2.0
        combined_scores = {}
        for algo, exec_score in exec_scores.items():
            w_rt = reorder_weights.get(algo, 0.0)
            # w_reorder_time is already part of exec_score, add extra boost
            reorder_bonus = w_rt * REORDER_WEIGHT_BOOST
            combined_scores[algo] = exec_score + reorder_bonus
        
        if not combined_scores:
            return "ORIGINAL", {"ORIGINAL": 1.0}
        
        best_algo = max(combined_scores, key=combined_scores.get)
        return best_algo, combined_scores
    
    def get_algorithm_metadata(self, type_name: str) -> Dict[str, Dict[str, float]]:
        """Get metadata (avg_speedup, avg_reorder_time) for all algorithms from type weights."""
        weights = self.algorithm_selector.load_weights(type_name)
        metadata = {}
        for algo, w in weights.items():
            if algo.startswith("_"):
                continue
            if isinstance(w, dict):
                meta = w.get("_metadata", {})
                metadata[algo] = {
                    "avg_speedup": meta.get("avg_speedup", 1.0),
                    "avg_reorder_time": meta.get("avg_reorder_time", 0.0),
                }
        return metadata
    
    def _select_best_amortization(
        self,
        matched_type: str,
        features: GraphFeatures,
        benchmark: str = None
    ) -> Tuple[str, Dict[str, float]]:
        """Select algorithm that needs fewest iterations to amortize reordering cost.
        
        Uses actual avg_speedup and avg_reorder_time from training metadata.
        
        Formula: iterations_to_amortize = reorder_time / time_saved_per_iter
        Where: time_saved_per_iter = (speedup - 1) / speedup  (normalized to 1s baseline)
        
        Lower iterations = better (amortizes faster)
        """
        # Get metadata for all algorithms
        metadata = self.get_algorithm_metadata(matched_type)
        
        # Calculate iterations to amortize for each algorithm
        iterations_scores = {}  # Lower = better
        
        for algo, meta in metadata.items():
            if algo == "ORIGINAL":
                continue  # ORIGINAL has no reorder cost
            
            speedup = meta.get("avg_speedup", 1.0)
            reorder_time = meta.get("avg_reorder_time", 0.0)
            
            if speedup <= 1.0:
                # No speedup = never amortizes
                iterations_scores[algo] = float('inf')
            else:
                # time_saved_per_iter = (speedup - 1) / speedup
                time_saved_per_iter = (speedup - 1.0) / speedup
                if time_saved_per_iter <= 0:
                    iterations_scores[algo] = float('inf')
                else:
                    iterations_scores[algo] = reorder_time / time_saved_per_iter
        
        if not iterations_scores:
            return "ORIGINAL", {"ORIGINAL": 1.0}
        
        # Select algorithm with minimum iterations (fastest amortization)
        best_algo = min(iterations_scores, key=iterations_scores.get)
        
        # Convert to scores (inverse of iterations, so higher = better for display)
        display_scores = {}
        for algo, iters in iterations_scores.items():
            if iters == float('inf'):
                display_scores[algo] = 0.0
            else:
                display_scores[algo] = 1.0 / (iters + 0.001)  # Avoid div by zero
        
        return best_algo, display_scores
    
    def emulate_with_config(
        self,
        features: GraphFeatures,
        config: WeightConfig,
        benchmark: str = None
    ) -> EmulationResult:
        """Emulate with a specific weight configuration."""
        old_config = self.config
        self.config = config
        result = self.emulate(features, benchmark)
        self.config = old_config
        return result
    
    def analyze_weight_impact(
        self,
        features: GraphFeatures,
        benchmark: str = None
    ) -> Dict[str, Dict]:
        """Analyze the impact of each weight on algorithm selection."""
        # Get baseline result with all weights enabled
        baseline_config = WeightConfig()
        baseline = self.emulate_with_config(features, baseline_config, benchmark)
        
        impacts = {
            "baseline": {
                "selected": baseline.selected_algorithm,
                "scores": baseline.algorithm_scores.copy(),
            }
        }
        
        # Test disabling each weight
        for weight in WEIGHT_FIELDS:
            test_config = WeightConfig()
            test_config.disable(weight)
            result = self.emulate_with_config(features, test_config, benchmark)
            
            # Check if selection changed
            changed = result.selected_algorithm != baseline.selected_algorithm
            score_diff = {
                algo: result.algorithm_scores.get(algo, 0) - baseline.algorithm_scores.get(algo, 0)
                for algo in baseline.algorithm_scores
            }
            
            impacts[weight] = {
                "selected": result.selected_algorithm,
                "changed": changed,
                "score_diff": score_diff,
            }
        
        return impacts
    
    def grid_search_caps(
        self,
        features_list: List[GraphFeatures],
        benchmark_path: Path,
        bias_caps: List[float],
        weight_caps: List[float],
    ) -> List[Dict]:
        """Grid search to find optimal cap values."""
        
        # Load benchmark data
        with open(benchmark_path) as f:
            benchmark_data = json.load(f)
        
        # Group by graph and benchmark
        actual_best = defaultdict(dict)
        all_times = defaultdict(dict)
        
        for entry in benchmark_data:
            if not entry.get("success", False):
                continue
            graph = entry["graph"]
            bench = entry["benchmark"]
            algo = entry["algorithm"]
            time = entry["time_seconds"]
            
            key = (graph, bench)
            all_times[key][algo] = time
            if key not in actual_best or time < actual_best[key]["time"]:
                actual_best[key] = {"algorithm": algo, "time": time}
        
        # Build features lookup
        features_lookup = {f.name: f for f in features_list}
        
        results = []
        
        for bias_cap in bias_caps:
            for weight_cap in weight_caps:
                # Set caps
                self.config.bias_cap = bias_cap if bias_cap > 0 else None
                self.config.weight_cap = weight_cap if weight_cap > 0 else None
                
                # Evaluate
                matches = 0
                top_2 = 0
                top_3 = 0
                total = 0
                speedup_gaps = []
                
                for (graph, bench), best in actual_best.items():
                    if graph not in features_lookup:
                        continue
                    
                    features = features_lookup[graph]
                    emulated = self.emulate(features, bench)
                    
                    emulated_algo = emulated.selected_algorithm
                    best_time = best["time"]
                    emulated_time = all_times[(graph, bench)].get(emulated_algo, best_time * 10)
                    
                    speedup_gap = emulated_time / best_time if best_time > 0 else 1.0
                    speedup_gaps.append(speedup_gap)
                    
                    # Rank
                    sorted_algos = sorted(all_times[(graph, bench)].items(), key=lambda x: x[1])
                    rank = next((i + 1 for i, (a, _) in enumerate(sorted_algos) if a == emulated_algo), 999)
                    
                    total += 1
                    if emulated_algo == best["algorithm"]:
                        matches += 1
                    if rank <= 2:
                        top_2 += 1
                    if rank <= 3:
                        top_3 += 1
                
                avg_gap = sum(speedup_gaps) / len(speedup_gaps) if speedup_gaps else 999
                
                results.append({
                    "bias_cap": bias_cap,
                    "weight_cap": weight_cap,
                    "matches": matches,
                    "top_2": top_2,
                    "top_3": top_3,
                    "total": total,
                    "match_rate": matches / total if total > 0 else 0,
                    "top_3_rate": top_3 / total if total > 0 else 0,
                    "avg_slowdown": avg_gap,
                })
        
        return results


# =============================================================================
# CLI Interface
# =============================================================================

def print_emulation_result(result: EmulationResult, verbose: bool = False):
    """Pretty print an emulation result."""
    print(f"\n{'='*70}")
    print(f"Graph: {result.graph_name}")
    print(f"{'='*70}")
    print(f"  Matched Type:      {result.matched_type} (distance: {result.type_distance:.4f})")
    print(f"  Selected Algorithm: {result.selected_algorithm}")
    
    if verbose and result.algorithm_scores:
        print("\n  Algorithm Scores (sorted):")
        sorted_scores = sorted(result.algorithm_scores.items(), key=lambda x: -x[1])
        for algo, score in sorted_scores[:10]:
            marker = " ← SELECTED" if algo == result.selected_algorithm else ""
            print(f"    {algo:<20}: {score:>8.4f}{marker}")


def print_weight_impact(impacts: Dict[str, Dict], graph_name: str):
    """Pretty print weight impact analysis."""
    baseline = impacts.get("baseline", {})
    
    print(f"\n{'='*70}")
    print(f"WEIGHT IMPACT ANALYSIS: {graph_name}")
    print(f"{'='*70}")
    print(f"Baseline selection: {baseline.get('selected', 'N/A')}")
    print()
    
    print(f"{'Weight':<25} {'Selection':<20} {'Changed?':<10}")
    print("-" * 55)
    
    for weight in WEIGHT_FIELDS:
        if weight not in impacts:
            continue
        info = impacts[weight]
        changed_marker = "YES ⚠️" if info.get("changed") else "no"
        print(f"{weight:<25} {info.get('selected', 'N/A'):<20} {changed_marker:<10}")


def load_reorder_times_from_mappings(graph_name: str) -> Dict[str, float]:
    """Load reorder times from .time files in mappings directory."""
    mappings_dir = RESULTS_DIR / "mappings" / graph_name
    reorder_times = {}
    
    if not mappings_dir.exists():
        return reorder_times
    
    for time_file in mappings_dir.glob("*.time"):
        algo_name = time_file.stem
        try:
            with open(time_file) as f:
                reorder_times[algo_name] = float(f.read().strip())
        except (ValueError, IOError):
            pass
    
    # Add ORIGINAL with 0 reorder time
    reorder_times["ORIGINAL"] = 0.0
    
    return reorder_times


def load_benchmark_data(benchmark_path: Path) -> Dict:
    """Load and parse benchmark data into structured format."""
    with open(benchmark_path) as f:
        benchmark_data = json.load(f)
    
    # Collect all graphs to load reorder times
    graphs = set()
    for entry in benchmark_data:
        if entry.get("success"):
            graphs.add(entry["graph"])
    
    # Load reorder times from mappings
    reorder_times_by_graph = {}
    for graph in graphs:
        reorder_times_by_graph[graph] = load_reorder_times_from_mappings(graph)
    
    # Collect all data by (graph, benchmark, algorithm)
    all_data = defaultdict(lambda: defaultdict(dict))  # graph -> bench -> algo -> {time, reorder_time}
    
    for entry in benchmark_data:
        if not entry.get("success", False):
            continue
        graph = entry["graph"]
        bench = entry["benchmark"]
        algo = entry["algorithm"]
        exec_time = entry.get("time_seconds", 999)
        
        # Get reorder time: prefer from benchmark, then from mappings
        reorder_time = entry.get("reorder_time", 0) or entry.get("reorder_time_seconds", 0)
        if reorder_time == 0 and graph in reorder_times_by_graph:
            reorder_time = reorder_times_by_graph[graph].get(algo, 0)
        
        all_data[graph][bench][algo] = {
            "exec_time": exec_time,
            "reorder_time": reorder_time,
            "endtoend_time": reorder_time + exec_time,
        }
    
    return all_data


def find_optimal_algorithm(
    algo_data: Dict[str, Dict],
    mode: SelectionMode,
    original_time: float = None
) -> Tuple[str, Dict]:
    """Find the optimal algorithm based on selection mode."""
    if not algo_data:
        return "ORIGINAL", {}
    
    # Get ORIGINAL execution time for amortization calculation
    if original_time is None:
        original_time = algo_data.get("ORIGINAL", {}).get("exec_time", 999)
    
    best_algo = None
    best_value = float('inf')
    metrics = {}
    
    for algo, data in algo_data.items():
        exec_time = data.get("exec_time", 999)
        reorder_time = data.get("reorder_time", 0)
        
        if mode == SelectionMode.FASTEST_REORDER:
            # Skip ORIGINAL for reorder comparison (it has no reordering)
            if algo == "ORIGINAL":
                value = float('inf')  # ORIGINAL doesn't reorder
            else:
                value = reorder_time
        
        elif mode == SelectionMode.FASTEST_EXECUTION:
            value = exec_time
        
        elif mode == SelectionMode.BEST_ENDTOEND:
            value = reorder_time + exec_time
        
        elif mode == SelectionMode.BEST_AMORTIZATION:
            # Iterations to amortize = reorder_time / time_saved_per_iteration
            # time_saved = original_time - exec_time
            time_saved = original_time - exec_time
            if algo == "ORIGINAL" or time_saved <= 0:
                value = float('inf')  # No benefit or slower than original
            else:
                value = reorder_time / time_saved  # iterations needed
        
        else:
            value = exec_time  # default
        
        metrics[algo] = {
            "exec_time": exec_time,
            "reorder_time": reorder_time,
            "endtoend_time": reorder_time + exec_time,
            "value": value,
        }
        
        # Calculate amortization
        time_saved = original_time - exec_time
        if time_saved > 0 and reorder_time > 0:
            metrics[algo]["amortization_iters"] = reorder_time / time_saved
        else:
            metrics[algo]["amortization_iters"] = float('inf') if algo != "ORIGINAL" else 0
        
        if value < best_value:
            best_value = value
            best_algo = algo
    
    return best_algo, metrics


def compare_with_benchmark(
    emulator: "AdaptiveOrderEmulator",
    benchmark_path: Path,
    features_cache: Dict[str, "GraphFeatures"],
    mode: SelectionMode = SelectionMode.DATABASE
) -> Dict:
    """Compare emulated selections against actual benchmark results for a given mode."""
    all_data = load_benchmark_data(benchmark_path)
    
    results = {
        "mode": mode.value,
        "matches": 0,
        "top_2": 0,
        "top_3": 0,
        "total": 0,
        "avg_gap": 0.0,
        "details": []
    }
    
    gaps = []
    
    for graph, bench_data in all_data.items():
        if graph not in features_cache:
            continue
        
        features = features_cache[graph]
        
        for bench, algo_data in bench_data.items():
            # Find actual optimal for this mode
            original_time = algo_data.get("ORIGINAL", {}).get("exec_time", 999)
            optimal_algo, metrics = find_optimal_algorithm(algo_data, mode, original_time)
            
            if optimal_algo is None:
                continue
            
            # Get emulated selection with mode-specific logic
            emulated = emulator.emulate(features, bench, mode)
            emulated_algo = emulated.selected_algorithm
            
            # Get metrics for emulated algorithm
            emulated_metrics = metrics.get(emulated_algo, {})
            optimal_metrics = metrics.get(optimal_algo, {})
            
            emulated_value = emulated_metrics.get("value", float('inf'))
            optimal_value = optimal_metrics.get("value", 1.0)
            
            # Calculate gap (ratio of emulated / optimal)
            if optimal_value > 0 and optimal_value != float('inf'):
                gap = emulated_value / optimal_value if emulated_value != float('inf') else 100
            else:
                gap = 1.0 if emulated_algo == optimal_algo else 100
            gaps.append(min(gap, 100))  # Cap at 100x
            
            # Rank of emulated algorithm
            sorted_algos = sorted(
                [(a, m.get("value", float('inf'))) for a, m in metrics.items()],
                key=lambda x: x[1]
            )
            rank = next((i + 1 for i, (a, _) in enumerate(sorted_algos) if a == emulated_algo), 999)
            
            match = emulated_algo == optimal_algo
            results["total"] += 1
            if match:
                results["matches"] += 1
            if rank <= 2:
                results["top_2"] += 1
            if rank <= 3:
                results["top_3"] += 1
            
            results["details"].append({
                "graph": graph,
                "benchmark": bench,
                "emulated": emulated_algo,
                "optimal": optimal_algo,
                "match": match,
                "rank": rank,
                "gap": gap,
                "emulated_value": emulated_value,
                "optimal_value": optimal_value,
                "emulated_exec": emulated_metrics.get("exec_time", 0),
                "emulated_reorder": emulated_metrics.get("reorder_time", 0),
                "emulated_amortization": emulated_metrics.get("amortization_iters", 0),
                "optimal_exec": optimal_metrics.get("exec_time", 0),
                "optimal_reorder": optimal_metrics.get("reorder_time", 0),
                "optimal_amortization": optimal_metrics.get("amortization_iters", 0),
            })
    
    if gaps:
        results["avg_gap"] = sum(gaps) / len(gaps)
    
    return results


def compare_all_modes(
    emulator: "AdaptiveOrderEmulator",
    benchmark_path: Path,
    features_cache: Dict[str, "GraphFeatures"]
) -> Dict[str, Dict]:
    """Compare against all selection modes."""
    results = {}
    for mode in SelectionMode:
        results[mode.value] = compare_with_benchmark(emulator, benchmark_path, features_cache, mode)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="AdaptiveOrder Emulator - Analyze weight impacts on algorithm selection"
    )
    
    # Input options
    parser.add_argument("--graph", "-g", type=str, help="Path to a single graph file")
    parser.add_argument("--all-graphs", "-a", action="store_true", help="Process all graphs in results/graphs")
    parser.add_argument("--use-cache", action="store_true", help="Use cached features instead of extracting")
    
    # Selection mode
    parser.add_argument("--mode", "-m", type=str, default="fastest-execution",
                        choices=["fastest-reorder", "fastest-execution", "best-endtoend", "best-amortization", "all"],
                        help="Selection optimization target: "
                             "fastest-reorder (min reorder time), "
                             "fastest-execution (min exec time, default), "
                             "best-endtoend (min reorder+exec), "
                             "best-amortization (min iterations to amortize), "
                             "all (evaluate all modes)")
    
    # Weight configuration
    parser.add_argument("--disable-weight", "-d", type=str, action="append", 
                        help="Disable specific weight(s)")
    parser.add_argument("--only-bias", action="store_true", 
                        help="Only use bias weights (disable all feature weights)")
    parser.add_argument("--no-cache-weights", action="store_true",
                        help="Disable cache impact weights")
    parser.add_argument("--bias-cap", type=float, default=None,
                        help="Cap bias values at this maximum (e.g., 1.0)")
    parser.add_argument("--weight-cap", type=float, default=None,
                        help="Cap ALL weight values (feature + cache) at this maximum")
    parser.add_argument("--normalize-bias", action="store_true",
                        help="Normalize all biases to [0, 1] range")
    
    # Analysis options
    parser.add_argument("--analyze-impact", action="store_true",
                        help="Analyze impact of each weight")
    parser.add_argument("--compare-benchmark", "-c", type=str,
                        help="Compare against benchmark results JSON")
    parser.add_argument("--benchmark", "-b", type=str, default=None,
                        help="Specific benchmark (pr, bfs, cc, sssp, bc)")
    parser.add_argument("--grid-search", action="store_true",
                        help="Grid search for optimal bias/weight cap values")
    
    # Output options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    # Parse selection mode
    if args.mode == "all":
        selection_mode = None  # Will run all modes
    else:
        selection_mode = SelectionMode(args.mode)
    
    # Initialize emulator
    emulator = AdaptiveOrderEmulator()
    
    # Configure weights
    if args.disable_weight:
        for w in args.disable_weight:
            emulator.config.disable(w)
            print(f"Disabled weight: {w}")
    
    if args.only_bias:
        for w in WEIGHT_FIELDS:
            if w != "bias":
                emulator.config.disable(w)
        print("Only using bias weights")
    
    if args.no_cache_weights:
        for w in ["cache_l1_impact", "cache_l2_impact", "cache_l3_impact", "cache_dram_penalty"]:
            emulator.config.disable(w)
        print("Disabled cache impact weights")
    
    if args.bias_cap is not None:
        emulator.config.bias_cap = args.bias_cap
        print(f"Bias cap set to: {args.bias_cap}")
    
    if args.weight_cap is not None:
        emulator.config.weight_cap = args.weight_cap
        print(f"Weight cap set to: {args.weight_cap}")
    
    if args.normalize_bias:
        emulator.config.normalize_bias = True
        print("Bias normalization enabled")
    
    # Load features
    features_cache = {}
    if args.use_cache:
        features_cache = load_cached_features()
        print(f"Loaded {len(features_cache)} cached graph features")
    
    # Process graphs
    graphs_to_process = []
    
    if args.graph:
        if args.use_cache:
            graph_name = Path(args.graph).stem
            if graph_name in features_cache:
                graphs_to_process.append(features_cache[graph_name])
            else:
                print(f"Graph {graph_name} not in cache, extracting features...")
                features = extract_features_from_graph(args.graph)
                if features:
                    graphs_to_process.append(features)
        else:
            features = extract_features_from_graph(args.graph)
            if features:
                graphs_to_process.append(features)
    
    elif args.all_graphs:
        if args.use_cache:
            graphs_to_process = list(features_cache.values())
        else:
            for graph_dir in sorted(GRAPHS_DIR.iterdir()):
                if graph_dir.is_dir():
                    mtx_files = list(graph_dir.glob("*.mtx"))
                    if mtx_files:
                        features = extract_features_from_graph(str(mtx_files[0]))
                        if features:
                            graphs_to_process.append(features)
    
    # Run emulation
    results = []
    for features in graphs_to_process:
        if args.analyze_impact:
            impacts = emulator.analyze_weight_impact(features, args.benchmark)
            print_weight_impact(impacts, features.name)
        else:
            result = emulator.emulate(features, args.benchmark)
            results.append(result)
            print_emulation_result(result, args.verbose)
    
    # Compare with benchmark if requested
    if args.compare_benchmark:
        benchmark_path = Path(args.compare_benchmark)
        if benchmark_path.exists():
            if selection_mode is None:
                # Run all modes
                all_results = compare_all_modes(emulator, benchmark_path, features_cache)
                
                print(f"\n{'='*90}")
                print("COMPARISON WITH ACTUAL BENCHMARK RESULTS - ALL MODES")
                print(f"{'='*90}")
                print(f"\n{'Mode':<22} {'Match%':>8} {'Top2%':>8} {'Top3%':>8} {'AvgGap':>10}")
                print("-" * 60)
                
                for mode_name, comparison in all_results.items():
                    total = comparison['total']
                    if total > 0:
                        print(f"{mode_name:<22} {comparison['matches']/total*100:>7.1f}% "
                              f"{comparison['top_2']/total*100:>7.1f}% "
                              f"{comparison['top_3']/total*100:>7.1f}% "
                              f"{comparison['avg_gap']:>9.2f}x")
                
                # Detailed output for all modes if verbose
                if args.verbose:
                    for mode_name, comparison in all_results.items():
                        print(f"\n{'='*90}")
                        print(f"MODE: {mode_name.upper()}")
                        print(f"{'='*90}")
                        print(f"\n{'Graph':<18} {'Bench':<6} {'Emulated':<15} {'Optimal':<15} {'Rank':>4} {'Gap':>8}")
                        print("-" * 75)
                        for d in sorted(comparison["details"], key=lambda x: (x["graph"], x["benchmark"])):
                            match_marker = "✓" if d["match"] else " "
                            gap_str = f"{d['gap']:.2f}x" if d['gap'] < 100 else "inf"
                            print(f"{d['graph']:<18} {d['benchmark']:<6} {d['emulated']:<15} "
                                  f"{d['optimal']:<15} {d['rank']:>4} {gap_str:>8} {match_marker}")
            else:
                # Single mode comparison
                comparison = compare_with_benchmark(emulator, benchmark_path, features_cache, selection_mode)
                
                print(f"\n{'='*80}")
                print(f"COMPARISON WITH ACTUAL BENCHMARK RESULTS - {selection_mode.value.upper()}")
                print(f"{'='*80}")
                total = comparison['total']
                if total > 0:
                    print(f"Exact Matches:     {comparison['matches']:3d}/{total} ({comparison['matches']/total*100:5.1f}%)")
                    print(f"Top 2 (rank ≤ 2):  {comparison['top_2']:3d}/{total} ({comparison['top_2']/total*100:5.1f}%)")
                    print(f"Top 3 (rank ≤ 3):  {comparison['top_3']:3d}/{total} ({comparison['top_3']/total*100:5.1f}%)")
                    print(f"Avg Gap:           {comparison['avg_gap']:.2f}x (1.0 = optimal)")
                
                if args.verbose:
                    print(f"\n{'Graph':<18} {'Bench':<6} {'Emulated':<15} {'Optimal':<15} {'Rank':>4} {'Gap':>8}")
                    print("-" * 75)
                    for d in sorted(comparison["details"], key=lambda x: (x["graph"], x["benchmark"])):
                        match_marker = "✓" if d["match"] else " "
                        gap_str = f"{d['gap']:.2f}x" if d['gap'] < 100 else "inf"
                        print(f"{d['graph']:<18} {d['benchmark']:<6} {d['emulated']:<15} "
                              f"{d['optimal']:<15} {d['rank']:>4} {gap_str:>8} {match_marker}")
    
    # Grid search for optimal caps
    if args.grid_search and args.compare_benchmark:
        benchmark_path = Path(args.compare_benchmark)
        if benchmark_path.exists() and features_cache:
            print(f"\n{'='*80}")
            print("GRID SEARCH FOR OPTIMAL CAP VALUES")
            print(f"{'='*80}")
            
            bias_caps = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]
            weight_caps = [0, 0.1, 0.2, 0.5, 1.0, 2.0]
            
            grid_results = emulator.grid_search_caps(
                list(features_cache.values()),
                benchmark_path,
                bias_caps,
                weight_caps
            )
            
            # Sort by top_3_rate descending, then avg_slowdown ascending
            grid_results.sort(key=lambda x: (-x["top_3_rate"], x["avg_slowdown"]))
            
            print(f"\n{'Bias Cap':>10} {'Weight Cap':>12} {'Match%':>8} {'Top3%':>8} {'AvgSlowdown':>12}")
            print("-" * 55)
            for r in grid_results[:15]:
                bias_str = f"{r['bias_cap']:.1f}" if r['bias_cap'] > 0 else "None"
                weight_str = f"{r['weight_cap']:.1f}" if r['weight_cap'] > 0 else "None"
                print(f"{bias_str:>10} {weight_str:>12} {r['match_rate']*100:>7.1f}% "
                      f"{r['top_3_rate']*100:>7.1f}% {r['avg_slowdown']:>11.2f}x")
            
            # Best by metric
            best_match = max(grid_results, key=lambda x: x["match_rate"])
            best_top3 = max(grid_results, key=lambda x: x["top_3_rate"])
            best_slow = min(grid_results, key=lambda x: x["avg_slowdown"])
            
            print(f"\nBest by exact match: bias_cap={best_match['bias_cap']}, weight_cap={best_match['weight_cap']}")
            print(f"Best by top 3 rate:  bias_cap={best_top3['bias_cap']}, weight_cap={best_top3['weight_cap']}")
            print(f"Best by avg slowdown: bias_cap={best_slow['bias_cap']}, weight_cap={best_slow['weight_cap']}")
    
    # JSON output
    if args.json and results:
        output = [
            {
                "graph": r.graph_name,
                "matched_type": r.matched_type,
                "type_distance": r.type_distance,
                "selected_algorithm": r.selected_algorithm,
                "scores": r.algorithm_scores,
            }
            for r in results
        ]
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
