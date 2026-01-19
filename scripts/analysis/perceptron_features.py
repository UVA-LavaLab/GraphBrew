#!/usr/bin/env python3
"""
Perceptron Feature Generator and Trainer

Generates perceptron features from graph characteristics and correlates
them with optimal reordering algorithms. Outputs weights for the
AdaptiveOrder perceptron selector.

Features extracted:
- Modularity (community structure strength)
- Log nodes, log edges (scale)
- Density (edge density)
- Average degree
- Degree variance (heterogeneity)
- Hub concentration (power-law degree)

Usage:
    python3 scripts/analysis/perceptron_features.py [--graphs-dir DIR]
    
Examples:
    python3 scripts/analysis/perceptron_features.py --quick
    python3 scripts/analysis/perceptron_features.py --graphs-dir ./graphs --output weights.json
"""

import os
import sys
import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.common import (
    ALGORITHMS, COMMUNITY_ALGORITHMS, LEIDEN_ALGORITHMS,
    run_benchmark, Colors, print_header, print_subheader,
    format_time, format_speedup
)

# ============================================================================
# Configuration
# ============================================================================

# Graph structural feature names used by the perceptron
STRUCTURAL_FEATURE_NAMES = [
    'modularity',
    'log_nodes',
    'log_edges',
    'density',
    'avg_degree',
    'degree_variance',
    'hub_concentration',
]

# Cache performance feature names (from cache simulation)
CACHE_FEATURE_NAMES = [
    'l1_hit_rate',
    'l2_hit_rate',
    'l3_hit_rate',
    'dram_access_rate',
    'l1_eviction_rate',
    'l2_eviction_rate',
    'l3_eviction_rate',
]

# Combined feature names (structural + cache)
FEATURE_NAMES = STRUCTURAL_FEATURE_NAMES + CACHE_FEATURE_NAMES

# Algorithms supported by the perceptron
PERCEPTRON_ALGORITHMS = {
    7: "HUBCLUSTERDBG",
    8: "RABBITORDER",
    12: "LeidenOrder",
    17: "LeidenDFSHub",
    20: "LeidenHybrid",
}

# Synthetic graphs for testing
SYNTHETIC_GRAPHS = {
    "rmat_12": "-g 12",
    "rmat_14": "-g 14",
    "rmat_16": "-g 16",
    "rmat_18": "-g 18",
}

# ============================================================================
# Feature Extraction
# ============================================================================

@dataclass
class PerceptronFeatures:
    """Features for perceptron-based algorithm selection."""
    graph_name: str
    # Structural features
    modularity: float = 0.0
    log_nodes: float = 0.0
    log_edges: float = 0.0
    density: float = 0.0
    avg_degree: float = 0.0
    degree_variance: float = 0.0
    hub_concentration: float = 0.0
    # Cache performance features
    l1_hit_rate: float = 0.0
    l2_hit_rate: float = 0.0
    l3_hit_rate: float = 0.0
    dram_access_rate: float = 0.0
    l1_eviction_rate: float = 0.0
    l2_eviction_rate: float = 0.0
    l3_eviction_rate: float = 0.0
    
    def to_vector(self) -> List[float]:
        """Return features as a vector."""
        return [
            # Structural features
            self.modularity,
            self.log_nodes,
            self.log_edges,
            self.density,
            self.avg_degree,
            self.degree_variance,
            self.hub_concentration,
            # Cache features
            self.l1_hit_rate,
            self.l2_hit_rate,
            self.l3_hit_rate,
            self.dram_access_rate,
            self.l1_eviction_rate,
            self.l2_eviction_rate,
            self.l3_eviction_rate,
        ]
    
    def to_structural_vector(self) -> List[float]:
        """Return only structural features as a vector."""
        return [
            self.modularity,
            self.log_nodes,
            self.log_edges,
            self.density,
            self.avg_degree,
            self.degree_variance,
            self.hub_concentration,
        ]
    
    def to_cache_vector(self) -> List[float]:
        """Return only cache features as a vector."""
        return [
            self.l1_hit_rate,
            self.l2_hit_rate,
            self.l3_hit_rate,
            self.dram_access_rate,
            self.l1_eviction_rate,
            self.l2_eviction_rate,
            self.l3_eviction_rate,
        ]
    
    def to_dict(self) -> Dict[str, float]:
        return {
            # Structural
            'modularity': self.modularity,
            'log_nodes': self.log_nodes,
            'log_edges': self.log_edges,
            'density': self.density,
            'avg_degree': self.avg_degree,
            'degree_variance': self.degree_variance,
            'hub_concentration': self.hub_concentration,
            # Cache
            'l1_hit_rate': self.l1_hit_rate,
            'l2_hit_rate': self.l2_hit_rate,
            'l3_hit_rate': self.l3_hit_rate,
            'dram_access_rate': self.dram_access_rate,
            'l1_eviction_rate': self.l1_eviction_rate,
            'l2_eviction_rate': self.l2_eviction_rate,
            'l3_eviction_rate': self.l3_eviction_rate,
        }


def extract_features(graph_args: str, timeout: int = 300) -> Optional[PerceptronFeatures]:
    """
    Extract perceptron features from a graph.
    
    Uses LeidenOrder to get modularity and community info.
    """
    import re
    import subprocess
    
    # Run with LeidenOrder to get modularity
    binary = "./bench/bin/pr"
    cmd = f"{binary} {graph_args} -o 12 -n 1"
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        output = result.stdout + result.stderr
        
        features = PerceptronFeatures(graph_name=graph_args)
        
        # Parse graph stats
        graph_match = re.search(r'Graph has (\d+) nodes and (\d+)', output)
        if graph_match:
            nodes = int(graph_match.group(1))
            edges = int(graph_match.group(2))
            features.log_nodes = math.log10(nodes) if nodes > 0 else 0
            features.log_edges = math.log10(edges) if edges > 0 else 0
            features.avg_degree = edges / nodes if nodes > 0 else 0
            max_edges = nodes * (nodes - 1) / 2
            features.density = edges / max_edges if max_edges > 0 else 0
        
        # Parse modularity
        mod_match = re.search(r'[Mm]odularity[:\s]+([\d.]+)', output)
        if mod_match:
            features.modularity = float(mod_match.group(1))
        
        # Estimate degree variance and hub concentration from graph type
        # For RMAT graphs, these are typically high
        if '-g' in graph_args:
            features.degree_variance = 0.5
            features.hub_concentration = 0.3
        else:
            features.degree_variance = 0.3
            features.hub_concentration = 0.2
        
        return features
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None


def extract_cache_features(
    graph_args: str,
    algorithm: str = "pr",
    reorder_id: int = 0,
    timeout: int = 300,
    json_output: str = None
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Extract cache performance features from a simulation.
    
    Args:
        graph_args: Graph arguments (-g N or -f path)
        algorithm: Algorithm to simulate (pr, bfs, cc, etc.)
        reorder_id: Reordering algorithm ID
        timeout: Timeout in seconds
        json_output: Path for JSON output file
    
    Returns:
        Tuple of (l1_hit_rate, l2_hit_rate, l3_hit_rate, dram_access_rate,
                  l1_eviction_rate, l2_eviction_rate, l3_eviction_rate)
    """
    import tempfile
    import subprocess
    
    # Default return values
    default = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    binary = f"./bench/bin_sim/{algorithm}"
    if not os.path.exists(binary):
        return default
    
    # Use temp file for JSON if not provided
    if json_output is None:
        json_output = tempfile.mktemp(suffix='.json')
        cleanup_json = True
    else:
        cleanup_json = False
    
    cmd = f"{binary} {graph_args} -o {reorder_id} -n 1"
    env = os.environ.copy()
    env['CACHE_OUTPUT_JSON'] = json_output
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env
        )
        
        if not os.path.exists(json_output):
            return default
        
        with open(json_output, 'r') as f:
            data = json.load(f)
        
        total_accesses = data.get('total_accesses', 0)
        memory_accesses = data.get('memory_accesses', 0)
        
        l1 = data.get('L1', {})
        l2 = data.get('L2', {})
        l3 = data.get('L3', {})
        
        l1_hit_rate = l1.get('hit_rate', 0.0)
        l2_hit_rate = l2.get('hit_rate', 0.0)
        l3_hit_rate = l3.get('hit_rate', 0.0)
        dram_access_rate = memory_accesses / total_accesses if total_accesses > 0 else 0.0
        
        # Compute eviction rates
        l1_total = l1.get('hits', 0) + l1.get('misses', 0)
        l2_total = l2.get('hits', 0) + l2.get('misses', 0)
        l3_total = l3.get('hits', 0) + l3.get('misses', 0)
        
        l1_eviction_rate = l1.get('evictions', 0) / l1_total if l1_total > 0 else 0.0
        l2_eviction_rate = l2.get('evictions', 0) / l2_total if l2_total > 0 else 0.0
        l3_eviction_rate = l3.get('evictions', 0) / l3_total if l3_total > 0 else 0.0
        
        return (l1_hit_rate, l2_hit_rate, l3_hit_rate, dram_access_rate,
                l1_eviction_rate, l2_eviction_rate, l3_eviction_rate)
        
    except Exception as e:
        print(f"Error extracting cache features: {e}")
        return default
    finally:
        if cleanup_json and os.path.exists(json_output):
            try:
                os.remove(json_output)
            except:
                pass


def extract_all_features(
    graph_args: str,
    algorithm: str = "pr",
    reorder_id: int = 0,
    timeout: int = 300,
    include_cache: bool = True
) -> Optional[PerceptronFeatures]:
    """
    Extract both structural and cache features for a graph.
    
    Args:
        graph_args: Graph arguments
        algorithm: Algorithm for cache simulation
        reorder_id: Reordering algorithm ID
        timeout: Timeout in seconds
        include_cache: Whether to include cache features
    
    Returns:
        PerceptronFeatures with all features populated
    """
    # Get structural features
    features = extract_features(graph_args, timeout)
    if features is None:
        return None
    
    # Get cache features if simulation binaries exist
    if include_cache and os.path.exists(f"./bench/bin_sim/{algorithm}"):
        cache_features = extract_cache_features(
            graph_args, algorithm, reorder_id, timeout
        )
        features.l1_hit_rate = cache_features[0]
        features.l2_hit_rate = cache_features[1]
        features.l3_hit_rate = cache_features[2]
        features.dram_access_rate = cache_features[3]
        features.l1_eviction_rate = cache_features[4]
        features.l2_eviction_rate = cache_features[5]
        features.l3_eviction_rate = cache_features[6]
    
    return features


# ============================================================================
# Benchmark Collection
# ============================================================================

@dataclass
class AlgorithmPerformance:
    """Performance of an algorithm on a graph."""
    algorithm_id: int
    algorithm_name: str
    trial_time: float
    reorder_time: float
    total_time: float
    speedup: float = 1.0


def benchmark_algorithms(
    graph_args: str,
    algorithms: Dict[int, str] = None,
    benchmark: str = "pr",
    num_trials: int = 3,
    timeout: int = 300
) -> List[AlgorithmPerformance]:
    """
    Benchmark all algorithms on a graph.
    """
    if algorithms is None:
        algorithms = PERCEPTRON_ALGORITHMS
    
    results = []
    binary = f"./bench/bin/{benchmark}"
    
    # Get baseline
    baseline_time = float('inf')
    
    for algo_id, algo_name in sorted(algorithms.items()):
        parsed, output = run_benchmark(
            binary=binary,
            graph_args=graph_args,
            algo_id=algo_id,
            num_trials=num_trials,
            timeout=timeout
        )
        
        if parsed:
            trial_time = parsed.get('average_time', float('inf'))
            reorder_time = parsed.get('reorder_time', 0) + parsed.get('relabel_time', 0)
            total_time = trial_time + reorder_time
            
            if algo_id == 0:
                baseline_time = trial_time
            
            results.append(AlgorithmPerformance(
                algorithm_id=algo_id,
                algorithm_name=algo_name,
                trial_time=trial_time,
                reorder_time=reorder_time,
                total_time=total_time,
            ))
    
    # Compute speedups
    if baseline_time < float('inf'):
        for r in results:
            r.speedup = baseline_time / r.trial_time if r.trial_time > 0 else 0
    
    return results


# ============================================================================
# Perceptron Training
# ============================================================================

def train_perceptron_weights(
    features_list: List[PerceptronFeatures],
    performance_list: List[List[AlgorithmPerformance]],
    algorithms: Dict[int, str] = None
) -> Dict[int, Dict[str, float]]:
    """
    Train perceptron weights from benchmark data.
    
    Uses a simple correlation-based approach:
    - For each algorithm, compute correlation of each feature with performance
    - Use correlation as weight
    - Add bias based on average speedup
    """
    if algorithms is None:
        algorithms = PERCEPTRON_ALGORITHMS
    
    weights = {}
    
    for algo_id, algo_name in algorithms.items():
        # Collect feature-performance pairs
        feature_perf = []
        
        for features, performances in zip(features_list, performance_list):
            # Find this algorithm's performance
            algo_perf = next((p for p in performances if p.algorithm_id == algo_id), None)
            if algo_perf:
                feature_perf.append((features.to_vector(), algo_perf.speedup))
        
        if not feature_perf:
            continue
        
        # Compute correlation for each feature
        n = len(feature_perf)
        n_features = len(FEATURE_NAMES)
        
        feature_weights = [0.0] * n_features
        avg_speedup = sum(fp[1] for fp in feature_perf) / n
        
        for i in range(n_features):
            # Simple correlation: mean(feature * speedup) / mean(feature)
            feature_vals = [fp[0][i] for fp in feature_perf]
            speedup_vals = [fp[1] for fp in feature_perf]
            
            if feature_vals:
                mean_f = sum(feature_vals) / len(feature_vals)
                mean_s = sum(speedup_vals) / len(speedup_vals)
                
                if mean_f > 0:
                    # Normalized correlation
                    cov = sum((f - mean_f) * (s - mean_s) for f, s in zip(feature_vals, speedup_vals)) / n
                    var_f = sum((f - mean_f) ** 2 for f in feature_vals) / n
                    
                    if var_f > 0:
                        feature_weights[i] = cov / var_f * 0.1  # Scale down
        
        weights[algo_id] = {
            'bias': min(1.0, avg_speedup / 2),  # Normalize to ~0-1 range
            'weights': dict(zip(FEATURE_NAMES, feature_weights))
        }
    
    return weights


def format_weights_for_cpp(weights: Dict[int, Dict[str, float]], include_cache: bool = True) -> str:
    """
    Format weights for inclusion in C++ code.
    
    Args:
        weights: Dictionary of algorithm ID -> weight data
        include_cache: Whether to include cache feature weights
    """
    lines = []
    lines.append("// Perceptron weights for AdaptiveOrder")
    lines.append("// Generated by scripts/analysis/perceptron_features.py")
    lines.append("// Features: " + ", ".join(FEATURE_NAMES if include_cache else STRUCTURAL_FEATURE_NAMES))
    lines.append("")
    
    for algo_id, data in sorted(weights.items()):
        algo_name = ALGORITHMS.get(algo_id, f"Unknown({algo_id})")
        bias = data['bias']
        w = data['weights']
        
        lines.append(f"// {algo_name} (ID: {algo_id})")
        lines.append(f"perceptron_weights[{algo_id}] = {{")
        lines.append(f"    {bias:.4f},  // bias")
        
        # Structural feature weights
        structural = (f"{w.get('modularity', 0):.4f}, {w.get('log_nodes', 0):.4f}, "
                     f"{w.get('log_edges', 0):.4f}, {w.get('density', 0):.4f}, "
                     f"{w.get('avg_degree', 0):.4f}, {w.get('degree_variance', 0):.4f}, "
                     f"{w.get('hub_concentration', 0):.4f}")
        
        if include_cache:
            # Cache feature weights
            cache = (f"{w.get('l1_hit_rate', 0):.4f}, {w.get('l2_hit_rate', 0):.4f}, "
                    f"{w.get('l3_hit_rate', 0):.4f}, {w.get('dram_access_rate', 0):.4f}, "
                    f"{w.get('l1_eviction_rate', 0):.4f}, {w.get('l2_eviction_rate', 0):.4f}, "
                    f"{w.get('l3_eviction_rate', 0):.4f}")
            lines.append(f"    {{{structural},  // structural features")
            lines.append(f"     {cache}}}  // cache features")
        else:
            lines.append(f"    {{{structural}}}  // structural features")
        
        lines.append("};")
        lines.append("")
    
    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate perceptron features and train weights"
    )
    parser.add_argument(
        "--graphs-dir", "-g",
        default=None,
        help="Directory containing graphs"
    )
    parser.add_argument(
        "--graphs-config",
        default=None,
        help="JSON config file with graph definitions"
    )
    parser.add_argument(
        "--benchmark", "-b",
        default="pr",
        help="Benchmark to use for training (default: pr)"
    )
    parser.add_argument(
        "--trials", "-n",
        type=int,
        default=3,
        help="Number of trials per configuration (default: 3)"
    )
    parser.add_argument(
        "--output", "-o",
        default="./bench/results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick test with synthetic graphs"
    )
    
    args = parser.parse_args()
    
    # Determine graphs to use
    if args.quick:
        graphs = SYNTHETIC_GRAPHS
    elif args.graphs_config:
        with open(args.graphs_config, 'r') as f:
            config = json.load(f)
        graphs = {name: info.get("args", f"-f {info['path']} -s")
                  for name, info in config.get("graphs", {}).items()}
    elif args.graphs_dir:
        graphs = {}
        config_path = os.path.join(args.graphs_dir, "graphs.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            for name, info in config.get("graphs", {}).items():
                if "args" in info:
                    graphs[name] = info["args"]
                elif "path" in info:
                    sym = "-s" if info.get("symmetric", True) else ""
                    graphs[name] = f"-f {info['path']} {sym}"
    else:
        graphs = SYNTHETIC_GRAPHS
    
    print_header("Perceptron Feature Generator")
    print(f"Graphs: {len(graphs)}")
    print(f"Benchmark: {args.benchmark}")
    
    # Collect features and performance
    features_list = []
    performance_list = []
    
    for graph_name, graph_args in graphs.items():
        print_subheader(f"Graph: {graph_name}")
        
        # Extract features
        print("  Extracting features...", end=" ", flush=True)
        features = extract_features(graph_args)
        if features:
            features.graph_name = graph_name
            print(f"OK (mod={features.modularity:.3f}, log_n={features.log_nodes:.2f})")
        else:
            print("FAILED")
            continue
        
        # Benchmark algorithms
        print("  Benchmarking algorithms...")
        performances = benchmark_algorithms(
            graph_args=graph_args,
            benchmark=args.benchmark,
            num_trials=args.trials
        )
        
        for p in sorted(performances, key=lambda x: x.trial_time):
            speedup_str = format_speedup(p.speedup)
            print(f"    {p.algorithm_name:<18} {format_time(p.trial_time):>10} {speedup_str}")
        
        features_list.append(features)
        performance_list.append(performances)
    
    if not features_list:
        print("No data collected")
        return 1
    
    # Train weights
    print_header("Training Perceptron Weights")
    weights = train_perceptron_weights(features_list, performance_list)
    
    # Print weights
    print("\nLearned weights:")
    print("-" * 70)
    for algo_id, data in sorted(weights.items()):
        algo_name = ALGORITHMS.get(algo_id, f"Unknown({algo_id})")
        print(f"\n{algo_name} (ID: {algo_id}):")
        print(f"  bias: {data['bias']:.4f}")
        for fname, w in data['weights'].items():
            print(f"  {fname}: {w:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output, exist_ok=True)
    
    # JSON output
    json_path = os.path.join(args.output, f"perceptron_weights_{timestamp}.json")
    output_data = {
        'weights': weights,
        'features': [f.to_dict() for f in features_list],
        'feature_names': FEATURE_NAMES,
    }
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # C++ output
    cpp_path = os.path.join(args.output, f"perceptron_weights_{timestamp}.cpp")
    with open(cpp_path, 'w') as f:
        f.write(format_weights_for_cpp(weights))
    
    print(f"\nResults saved to:")
    print(f"  JSON: {json_path}")
    print(f"  C++:  {cpp_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
