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
  python3 scripts/adaptive_emulator.py --graph results/graphs/email-Enron/email-Enron.mtx
  python3 scripts/adaptive_emulator.py --all-graphs --disable-weight w_modularity
  python3 scripts/adaptive_emulator.py --compare-benchmark results/benchmark_*.json
  python3 scripts/adaptive_emulator.py --mode best-endtoend --compare-benchmark results/benchmark.json
"""

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict


# =============================================================================
# Selection Modes
# =============================================================================

class SelectionMode(Enum):
    """Algorithm selection optimization target."""
    FASTEST_REORDER = "fastest-reorder"      # Minimize reordering time
    FASTEST_EXECUTION = "fastest-execution"  # Minimize execution time (default)
    BEST_ENDTOEND = "best-endtoend"          # Minimize reorder + execution time
    BEST_AMORTIZATION = "best-amortization"  # Minimize iterations to amortize
    HEURISTIC = "heuristic"                  # Feature-based heuristic (more robust)


# =============================================================================
# Constants
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
WEIGHTS_DIR = SCRIPT_DIR / "weights" / "active"
RESULTS_DIR = PROJECT_ROOT / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"

# All weight fields used in perceptron scoring
WEIGHT_FIELDS = [
    "bias",
    "w_modularity",
    "w_log_nodes",
    "w_log_edges", 
    "w_density",
    "w_avg_degree",
    "w_degree_variance",
    "w_hub_concentration",
    "w_clustering_coeff",
    "w_avg_path_length",
    "w_diameter",
    "w_community_count",
    "w_reorder_time",
    "cache_l1_impact",
    "cache_l2_impact",
    "cache_l3_impact",
    "cache_dram_penalty",
]

# Features used for type matching (Layer 1)
TYPE_MATCHING_FEATURES = [
    "modularity",
    "log_nodes",
    "log_edges",
    "density",
    "avg_degree",
    "degree_variance",
    "hub_concentration",
    "clustering_coeff",
    "community_count",
]

# Features used for algorithm scoring (Layer 2)
SCORING_FEATURES = [
    ("modularity", "w_modularity"),
    ("log_nodes", "w_log_nodes"),
    ("log_edges", "w_log_edges"),
    ("density", "w_density"),
    ("avg_degree", "w_avg_degree"),
    ("degree_variance", "w_degree_variance"),
    ("hub_concentration", "w_hub_concentration"),
    ("clustering_coeff", "w_clustering_coeff"),
    ("avg_path_length", "w_avg_path_length"),
    ("diameter", "w_diameter"),
    ("community_count", "w_community_count"),
    ("reorder_time", "w_reorder_time"),
]

ALGORITHMS = {
    0: "ORIGINAL",
    1: "RANDOM",
    2: "SORT",
    3: "HUBSORT",
    4: "HUBCLUSTER",
    5: "DBG",
    6: "HUBSORTDBG",
    7: "HUBCLUSTERDBG",
    8: "RABBITORDER",
    9: "GORDER",
    10: "CORDER",
    11: "RCM",
    12: "GraphBrewOrder",
    15: "LeidenOrder",
    16: "LeidenDendrogram",
    17: "LeidenCSR",
}

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
    
    @property
    def log_nodes(self) -> float:
        return math.log10(self.num_nodes) if self.num_nodes > 0 else 0
    
    @property
    def log_edges(self) -> float:
        return math.log10(self.num_edges) if self.num_edges > 0 else 0
    
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
        """Convert to dict for algorithm scoring."""
        return {
            "modularity": self.modularity,
            "log_nodes": self.log_nodes,
            "log_edges": self.log_edges,
            "density": self.density,
            "avg_degree": self.avg_degree / 100.0,  # Normalized as in C++
            "degree_variance": self.degree_variance / 100.0,  # Normalized
            "hub_concentration": self.hub_concentration,
            "clustering_coeff": self.clustering_coeff,
            "avg_path_length": self.avg_path_length / 10.0,  # Normalized
            "diameter": self.diameter / 50.0,  # Normalized
            "community_count": math.log10(self.community_count) if self.community_count > 0 else 0,
            "reorder_time": self.reorder_time,
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
    
    def set_multiplier(self, weight_name: str, multiplier: float):
        """Set a multiplier for a specific weight."""
        self.weight_multipliers[weight_name] = multiplier
    
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
        self.registry_path = registry_path or (WEIGHTS_DIR / "type_registry.json")
        self.registry = {}
        self.centroids = {}
        self._load_registry()
    
    def _load_registry(self):
        """Load type registry with centroids."""
        if not self.registry_path.exists():
            print(f"Warning: Type registry not found at {self.registry_path}")
            return
        
        with open(self.registry_path) as f:
            self.registry = json.load(f)
        
        # Extract centroids from nested structure
        # Format: {"type_N": {"centroid": [...], "graph_count": N, ...}}
        self.centroids = {}
        for type_name, type_info in self.registry.items():
            if isinstance(type_info, dict) and "centroid" in type_info:
                self.centroids[type_name] = type_info["centroid"]
        
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
    
    def get_all_distances(self, features: GraphFeatures) -> Dict[str, float]:
        """Get distances to all type centroids."""
        feature_vector = features.to_type_vector()
        distances = {}
        
        for type_name, centroid in self.centroids.items():
            if isinstance(centroid, list):
                distances[type_name] = self.euclidean_distance(feature_vector, centroid)
        
        return distances


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
        
        weights_path = self.weights_dir / f"{type_name}.json"
        if not weights_path.exists():
            print(f"Warning: Weights file not found: {weights_path}")
            return {}
        
        with open(weights_path) as f:
            weights = json.load(f)
        
        self.weights_cache[type_name] = weights
        return weights
    
    def compute_score(
        self,
        algo_weights: Dict[str, float],
        features: Dict[str, float],
        config: WeightConfig,
        benchmark: str = None
    ) -> float:
        """Compute perceptron score for an algorithm given features."""
        score = 0.0
        
        # Add bias if enabled
        if config.is_enabled("bias"):
            bias = algo_weights.get("bias", 0.5)
            # Apply bias cap if configured
            if config.bias_cap is not None:
                bias = min(bias, config.bias_cap)
            score += bias * config.get_multiplier("bias")
        
        # Add weighted features
        for feature_name, weight_name in SCORING_FEATURES:
            if not config.is_enabled(weight_name):
                continue
            
            weight = algo_weights.get(weight_name, 0.0)
            # Apply weight cap if configured
            weight = config.apply_cap(weight)
            feature_value = features.get(feature_name, 0.0)
            multiplier = config.get_multiplier(weight_name)
            
            score += weight * feature_value * multiplier
        
        # Add cache impacts if enabled
        for cache_weight in ["cache_l1_impact", "cache_l2_impact", "cache_l3_impact", "cache_dram_penalty"]:
            if config.is_enabled(cache_weight):
                cache_w = config.apply_cap(algo_weights.get(cache_weight, 0.0))
                score += cache_w * config.get_multiplier(cache_weight)
        
        # Apply benchmark-specific multiplier if available
        if benchmark:
            bench_weights = algo_weights.get("benchmark_weights", {})
            bench_multiplier = bench_weights.get(benchmark, 1.0)
            score *= bench_multiplier
        
        return score
    
    def select_algorithm(
        self,
        type_name: str,
        features: GraphFeatures,
        config: WeightConfig,
        benchmark: str = None
    ) -> Tuple[str, Dict[str, float]]:
        """Select the best algorithm for given features."""
        weights = self.load_weights(type_name)
        if not weights:
            return "ORIGINAL", {}
        
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
        return best_algo, scores


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
    
    import re
    
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
    cache_path = cache_path or (RESULTS_DIR / "graph_properties_cache.json")
    
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
                diameter=props.get("diameter_estimate", 0.0),
                community_count=props.get("community_count", 0.0),
            )
    
    return features


# =============================================================================
# Emulator
# =============================================================================

# Threshold for flagging a graph as "far" from known types (for informational purposes)
# This is calibrated based on typical type distances (range: 7-50+)
# Note: This does NOT trigger a fallback - perceptron still uses closest type's weights
UNKNOWN_TYPE_DISTANCE_THRESHOLD = 50.0


class AdaptiveOrderEmulator:
    """Main emulator class combining type matching and algorithm selection."""
    
    def __init__(self, weights_dir: Path = None):
        self.type_matcher = TypeMatcher()
        self.algorithm_selector = AlgorithmSelector(weights_dir)
        self.config = WeightConfig()
        self.selection_mode = SelectionMode.FASTEST_EXECUTION  # Default mode
    
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
    
    def is_distant_graph_type(self, type_distance: float) -> bool:
        """Check if a graph is far from known types (for informational purposes only)."""
        return type_distance > UNKNOWN_TYPE_DISTANCE_THRESHOLD
    
    def emulate(
        self,
        features: GraphFeatures,
        benchmark: str = None,
        mode: SelectionMode = None
    ) -> EmulationResult:
        """Emulate AdaptiveOrder for given graph features.
        
        For ALL graphs (including unknown/untrained), we use the perceptron
        approach: extract features, find closest type centroid, and use that
        type's weights to select the best algorithm. This is the whole point
        of the type-based perceptron system - it generalizes to new graphs.
        """
        mode = mode or self.selection_mode
        
        # Layer 1: Type matching - finds the closest type centroid
        matched_type, type_distance = self.type_matcher.find_best_type(features)
        
        # Note: For unknown graphs (high distance), we still use perceptron with
        # the closest type's weights. The type matching finds the most similar
        # graph type, and we use those learned weights. No fallback needed.
        
        # Layer 2: Algorithm selection (mode-dependent)
        # All modes use the type weights - no .time files needed
        if mode == SelectionMode.FASTEST_REORDER:
            selected_algo, scores = self._select_fastest_reorder(matched_type, features)
        elif mode == SelectionMode.HEURISTIC:
            selected_algo, scores = self._select_by_heuristic(features)
        elif mode == SelectionMode.FASTEST_EXECUTION:
            selected_algo, scores = self.algorithm_selector.select_algorithm(
                matched_type, features, self.config, benchmark
            )
        elif mode == SelectionMode.BEST_ENDTOEND:
            # Use standard selection but weight reorder_time heavily
            selected_algo, scores = self._select_best_endtoend(
                matched_type, features, benchmark
            )
        elif mode == SelectionMode.BEST_AMORTIZATION:
            selected_algo, scores = self._select_best_amortization(
                matched_type, features, benchmark
            )
        else:
            selected_algo, scores = self.algorithm_selector.select_algorithm(
                matched_type, features, self.config, benchmark
            )
        
        return EmulationResult(
            graph_name=features.name,
            matched_type=matched_type,
            type_distance=type_distance,
            selected_algorithm=selected_algo,
            algorithm_scores=scores,
            features=features,
        )
    
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
            scores['LeidenCSR'] = 0.8
        
        # Rule 2: High clustering (>0.5) - strong community structure
        # Community-detection and hub-based algorithms work well
        elif cluster > 0.5:
            scores['DBG'] = 1.0
            scores['HUBSORT'] = 0.95
            scores['LeidenOrder'] = 0.9
            scores['LeidenCSR'] = 0.85
            scores['RABBITORDER'] = 0.8
        
        # Rule 3: Medium clustering (0.1-0.5) - moderate community structure
        # Leiden algorithms typically perform well
        elif cluster > 0.1:
            scores['LeidenCSR'] = 1.0
            scores['LeidenDendrogram'] = 0.95
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
        from collections import defaultdict
        
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
        print(f"\n  Algorithm Scores (sorted):")
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
    mode: SelectionMode = SelectionMode.FASTEST_EXECUTION
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
