#!/usr/bin/env python3
"""
AdaptiveOrder Emulator - Fast Python implementation for weight analysis.

This emulator replicates the C++ AdaptiveOrder algorithm selection logic:
  Layer 1: Graph type matching (Euclidean distance to centroids)
  Layer 2: Algorithm selection (perceptron scoring)

Features:
  - Toggle individual weights on/off to analyze their impact
  - Compare emulated selections against actual benchmark results
  - Fast iteration without recompiling C++

Usage:
  python3 scripts/adaptive_emulator.py --graph results/graphs/email-Enron/email-Enron.mtx
  python3 scripts/adaptive_emulator.py --all-graphs --disable-weight w_modularity
  python3 scripts/adaptive_emulator.py --compare-benchmark results/benchmark_*.json
"""

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

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
        """Convert to feature vector for type matching."""
        return [
            self.modularity,
            self.log_nodes,
            self.log_edges,
            self.density,
            self.avg_degree,
            self.degree_variance,
            self.hub_concentration,
            self.clustering_coeff,
            self.community_count,
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

class AdaptiveOrderEmulator:
    """Main emulator class combining type matching and algorithm selection."""
    
    def __init__(self, weights_dir: Path = None):
        self.type_matcher = TypeMatcher()
        self.algorithm_selector = AlgorithmSelector(weights_dir)
        self.config = WeightConfig()
    
    def emulate(
        self,
        features: GraphFeatures,
        benchmark: str = None
    ) -> EmulationResult:
        """Emulate AdaptiveOrder for given graph features."""
        # Layer 1: Type matching
        matched_type, type_distance = self.type_matcher.find_best_type(features)
        
        # Layer 2: Algorithm selection
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


def compare_with_benchmark(
    emulator: AdaptiveOrderEmulator,
    benchmark_path: Path,
    features_cache: Dict[str, GraphFeatures]
) -> Dict:
    """Compare emulated selections against actual benchmark results."""
    with open(benchmark_path) as f:
        benchmark_data = json.load(f)
    
    # Group by graph and benchmark, find actual best and all times
    from collections import defaultdict
    actual_best = defaultdict(dict)
    all_times = defaultdict(dict)  # (graph, bench) -> {algo: time}
    
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
    
    # Compare emulated vs actual
    results = {
        "matches": 0,
        "top_2": 0,
        "top_3": 0,
        "total": 0,
        "avg_speedup_gap": 0.0,
        "details": []
    }
    
    speedup_gaps = []
    
    for (graph, bench), best in actual_best.items():
        if graph not in features_cache:
            continue
        
        features = features_cache[graph]
        emulated = emulator.emulate(features, bench)
        
        emulated_algo = emulated.selected_algorithm
        best_algo = best["algorithm"]
        best_time = best["time"]
        
        # Get the time for the emulated algorithm
        emulated_time = all_times[(graph, bench)].get(emulated_algo, best_time * 10)
        
        # Calculate speedup gap: how much slower than optimal
        speedup_gap = emulated_time / best_time if best_time > 0 else 1.0
        speedup_gaps.append(speedup_gap)
        
        # Rank of emulated algorithm
        sorted_algos = sorted(all_times[(graph, bench)].items(), key=lambda x: x[1])
        rank = 1
        for i, (algo, time) in enumerate(sorted_algos):
            if algo == emulated_algo:
                rank = i + 1
                break
        
        match = emulated_algo == best_algo
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
            "actual_best": best_algo,
            "match": match,
            "rank": rank,
            "speedup_gap": speedup_gap,
            "emulated_time": emulated_time,
            "best_time": best_time,
        })
    
    if speedup_gaps:
        results["avg_speedup_gap"] = sum(speedup_gaps) / len(speedup_gaps)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="AdaptiveOrder Emulator - Analyze weight impacts on algorithm selection"
    )
    
    # Input options
    parser.add_argument("--graph", "-g", type=str, help="Path to a single graph file")
    parser.add_argument("--all-graphs", "-a", action="store_true", help="Process all graphs in results/graphs")
    parser.add_argument("--use-cache", action="store_true", help="Use cached features instead of extracting")
    
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
            comparison = compare_with_benchmark(emulator, benchmark_path, features_cache)
            
            print(f"\n{'='*80}")
            print("COMPARISON WITH ACTUAL BENCHMARK RESULTS")
            print(f"{'='*80}")
            total = comparison['total']
            if total > 0:
                print(f"Exact Matches:     {comparison['matches']:3d}/{total} ({comparison['matches']/total*100:5.1f}%)")
                print(f"Top 2 (rank ≤ 2):  {comparison['top_2']:3d}/{total} ({comparison['top_2']/total*100:5.1f}%)")
                print(f"Top 3 (rank ≤ 3):  {comparison['top_3']:3d}/{total} ({comparison['top_3']/total*100:5.1f}%)")
                print(f"Avg Slowdown:      {comparison['avg_speedup_gap']:.2f}x (1.0 = optimal)")
            
            if args.verbose:
                print(f"\n{'Graph':<20} {'Bench':<6} {'Emulated':<15} {'Actual Best':<15} {'Rank':>4} {'Gap':>6}")
                print("-" * 75)
                for d in sorted(comparison["details"], key=lambda x: (x["graph"], x["benchmark"])):
                    match_marker = "✓" if d["match"] else " "
                    print(f"{d['graph']:<20} {d['benchmark']:<6} {d['emulated']:<15} "
                          f"{d['actual_best']:<15} {d['rank']:>4} {d['speedup_gap']:>5.2f}x {match_marker}")
    
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
