#!/usr/bin/env python3
"""
Per-graph data storage for GraphBrew experiments.

Stores benchmark results, reorder times, features, and computed weights 
per-graph, enabling:
1. Incremental experiment runs (can redo just some graphs)
2. Different perceptron flavor analysis
3. Historical tracking of experiment data
4. Easy data inspection and debugging

Directory structure:
    results/graphs/<graph_name>/
        features.json       - Graph topology features (nodes, edges, modularity, etc.)
        benchmarks/
            <benchmark>_<algorithm>.json  - Benchmark results per algo
        reorder/
            <algorithm>.json - Reorder time and mapping info
        weights/
            perceptron.json  - Per-graph computed weights
            
Standalone usage:
    python -m scripts.lib.graph_data --list-graphs
    python -m scripts.lib.graph_data --show-graph email-Enron
    python -m scripts.lib.graph_data --export-csv results/all_data.csv

Library usage:
    from scripts.lib.graph_data import (
        GraphDataStore, save_graph_features, save_benchmark_result,
        save_reorder_result, load_all_graph_data
    )
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import csv

from .utils import (
    PROJECT_ROOT, RESULTS_DIR, Logger, get_timestamp,
)

# Initialize logger
log = Logger()

# =============================================================================
# Constants
# =============================================================================

GRAPH_DATA_DIR = os.path.join(RESULTS_DIR, "graphs")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GraphFeatures:
    """Graph topology features for perceptron training."""
    graph_name: str
    nodes: int = 0
    edges: int = 0
    avg_degree: float = 0.0
    density: float = 0.0
    modularity: float = 0.0
    degree_variance: float = 0.0
    hub_concentration: float = 0.0
    clustering_coefficient: float = 0.0
    avg_path_length: float = 0.0
    diameter_estimate: float = 0.0
    community_count: int = 0
    graph_type: str = "unknown"  # social, road, web, etc.
    
    # Metadata
    source_file: str = ""
    last_updated: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GraphFeatures':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AlgorithmBenchmarkData:
    """Benchmark data for a single algorithm on a single graph."""
    graph_name: str
    algorithm_id: int
    algorithm_name: str
    benchmark: str  # pr, bfs, cc, sssp, bc, tc
    
    # Timing data
    avg_time: float = 0.0
    trial_times: List[float] = field(default_factory=list)
    speedup: float = 1.0  # vs original ordering
    
    # Cache metrics (if available)
    l1_hit_rate: float = 0.0
    l2_hit_rate: float = 0.0
    l3_hit_rate: float = 0.0
    llc_misses: int = 0
    
    # Metadata
    num_trials: int = 0
    success: bool = True
    error: str = ""
    timestamp: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AlgorithmBenchmarkData':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AlgorithmReorderData:
    """Reorder data for a single algorithm on a single graph."""
    graph_name: str
    algorithm_id: int
    algorithm_name: str
    
    # Timing
    reorder_time: float = 0.0  # seconds
    
    # Quality metrics
    modularity: float = 0.0
    communities: int = 0
    isolated_vertices: int = 0
    
    # Mapping info
    mapping_file: str = ""  # path to .lo file if saved
    
    # Metadata
    success: bool = True
    error: str = ""
    timestamp: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AlgorithmReorderData':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass 
class GraphPerceptronWeights:
    """Per-graph computed perceptron weights (for analysis)."""
    graph_name: str
    
    # Best algorithm per benchmark
    best_algo: Dict[str, str] = field(default_factory=dict)  # benchmark -> algo
    best_speedup: Dict[str, float] = field(default_factory=dict)  # benchmark -> speedup
    
    # All algorithm scores per benchmark
    algo_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Computed weights for this graph
    weights: Dict[str, Dict] = field(default_factory=dict)  # algo -> weight dict
    
    # Type assignment
    assigned_type: str = ""
    type_distance: float = 0.0
    
    # Metadata
    timestamp: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GraphPerceptronWeights':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Graph Data Store
# =============================================================================

class GraphDataStore:
    """
    Manages per-graph experiment data storage.
    
    Example:
        store = GraphDataStore("email-Enron")
        store.save_features(features)
        store.save_benchmark_result(result)
        store.save_reorder_result(reorder)
        
        # Later, load all data
        all_data = store.load_all()
    """
    
    def __init__(self, graph_name: str, data_dir: str = GRAPH_DATA_DIR):
        """Initialize store for a specific graph."""
        self.graph_name = graph_name
        self.base_dir = os.path.join(data_dir, graph_name)
        self.benchmarks_dir = os.path.join(self.base_dir, "benchmarks")
        self.reorder_dir = os.path.join(self.base_dir, "reorder")
        self.weights_dir = os.path.join(self.base_dir, "weights")
        
        # Create directories
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.benchmarks_dir, exist_ok=True)
        os.makedirs(self.reorder_dir, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Features
    # -------------------------------------------------------------------------
    
    def save_features(self, features: GraphFeatures) -> None:
        """Save graph topology features."""
        features.last_updated = get_timestamp()
        features_file = os.path.join(self.base_dir, "features.json")
        with open(features_file, 'w') as f:
            json.dump(features.to_dict(), f, indent=2)
        log.debug(f"Saved features for {self.graph_name}")
    
    def load_features(self) -> Optional[GraphFeatures]:
        """Load graph topology features."""
        features_file = os.path.join(self.base_dir, "features.json")
        if not os.path.exists(features_file):
            return None
        try:
            with open(features_file) as f:
                data = json.load(f)
            return GraphFeatures.from_dict(data)
        except Exception as e:
            log.warn(f"Failed to load features for {self.graph_name}: {e}")
            return None
    
    # -------------------------------------------------------------------------
    # Benchmark Results
    # -------------------------------------------------------------------------
    
    def save_benchmark_result(self, result: AlgorithmBenchmarkData) -> None:
        """Save benchmark result for an algorithm."""
        result.timestamp = get_timestamp()
        filename = f"{result.benchmark}_{result.algorithm_name}.json"
        filepath = os.path.join(self.benchmarks_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        log.debug(f"Saved benchmark {result.benchmark}/{result.algorithm_name} for {self.graph_name}")
    
    def load_benchmark_result(self, benchmark: str, algorithm_name: str) -> Optional[AlgorithmBenchmarkData]:
        """Load benchmark result for an algorithm."""
        filename = f"{benchmark}_{algorithm_name}.json"
        filepath = os.path.join(self.benchmarks_dir, filename)
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath) as f:
                data = json.load(f)
            return AlgorithmBenchmarkData.from_dict(data)
        except Exception as e:
            log.warn(f"Failed to load benchmark {benchmark}/{algorithm_name} for {self.graph_name}: {e}")
            return None
    
    def load_all_benchmarks(self) -> List[AlgorithmBenchmarkData]:
        """Load all benchmark results for this graph."""
        results = []
        if not os.path.exists(self.benchmarks_dir):
            return results
        
        for filename in os.listdir(self.benchmarks_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.benchmarks_dir, filename)
                try:
                    with open(filepath) as f:
                        data = json.load(f)
                    results.append(AlgorithmBenchmarkData.from_dict(data))
                except Exception:
                    continue
        return results
    
    # -------------------------------------------------------------------------
    # Reorder Results
    # -------------------------------------------------------------------------
    
    def save_reorder_result(self, result: AlgorithmReorderData) -> None:
        """Save reorder result for an algorithm."""
        result.timestamp = get_timestamp()
        filename = f"{result.algorithm_name}.json"
        filepath = os.path.join(self.reorder_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        log.debug(f"Saved reorder {result.algorithm_name} for {self.graph_name}")
    
    def load_reorder_result(self, algorithm_name: str) -> Optional[AlgorithmReorderData]:
        """Load reorder result for an algorithm."""
        filename = f"{algorithm_name}.json"
        filepath = os.path.join(self.reorder_dir, filename)
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath) as f:
                data = json.load(f)
            return AlgorithmReorderData.from_dict(data)
        except Exception as e:
            log.warn(f"Failed to load reorder {algorithm_name} for {self.graph_name}: {e}")
            return None
    
    def load_all_reorders(self) -> List[AlgorithmReorderData]:
        """Load all reorder results for this graph."""
        results = []
        if not os.path.exists(self.reorder_dir):
            return results
        
        for filename in os.listdir(self.reorder_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.reorder_dir, filename)
                try:
                    with open(filepath) as f:
                        data = json.load(f)
                    results.append(AlgorithmReorderData.from_dict(data))
                except Exception:
                    continue
        return results
    
    # -------------------------------------------------------------------------
    # Weights
    # -------------------------------------------------------------------------
    
    def save_weights(self, weights: GraphPerceptronWeights) -> None:
        """Save computed perceptron weights for this graph."""
        weights.timestamp = get_timestamp()
        weights_file = os.path.join(self.weights_dir, "perceptron.json")
        with open(weights_file, 'w') as f:
            json.dump(weights.to_dict(), f, indent=2)
        log.debug(f"Saved weights for {self.graph_name}")
    
    def load_weights(self) -> Optional[GraphPerceptronWeights]:
        """Load computed perceptron weights for this graph."""
        weights_file = os.path.join(self.weights_dir, "perceptron.json")
        if not os.path.exists(weights_file):
            return None
        try:
            with open(weights_file) as f:
                data = json.load(f)
            return GraphPerceptronWeights.from_dict(data)
        except Exception as e:
            log.warn(f"Failed to load weights for {self.graph_name}: {e}")
            return None
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    
    def get_summary(self) -> Dict:
        """Get summary of all data for this graph."""
        features = self.load_features()
        benchmarks = self.load_all_benchmarks()
        reorders = self.load_all_reorders()
        weights = self.load_weights()
        
        # Find best algorithm per benchmark
        best_per_bench = {}
        for bench_result in benchmarks:
            bench = bench_result.benchmark
            if bench not in best_per_bench or bench_result.speedup > best_per_bench[bench]['speedup']:
                best_per_bench[bench] = {
                    'algorithm': bench_result.algorithm_name,
                    'speedup': bench_result.speedup,
                    'time': bench_result.avg_time
                }
        
        return {
            'graph_name': self.graph_name,
            'has_features': features is not None,
            'num_benchmarks': len(benchmarks),
            'num_reorders': len(reorders),
            'has_weights': weights is not None,
            'features': features.to_dict() if features else None,
            'best_per_benchmark': best_per_bench,
            'algorithms_tested': list(set(b.algorithm_name for b in benchmarks)),
            'benchmarks_run': list(set(b.benchmark for b in benchmarks)),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def save_graph_features(graph_name: str, features: Dict, data_dir: str = GRAPH_DATA_DIR) -> None:
    """Save features for a graph from a dictionary."""
    store = GraphDataStore(graph_name, data_dir)
    graph_features = GraphFeatures(
        graph_name=graph_name,
        nodes=features.get('nodes', 0),
        edges=features.get('edges', 0),
        avg_degree=features.get('avg_degree', 0.0),
        density=features.get('density', 0.0),
        modularity=features.get('modularity', 0.0),
        degree_variance=features.get('degree_variance', 0.0),
        hub_concentration=features.get('hub_concentration', 0.0),
        clustering_coefficient=features.get('clustering_coefficient', features.get('clustering_coeff', 0.0)),
        avg_path_length=features.get('avg_path_length', 0.0),
        diameter_estimate=features.get('diameter', features.get('diameter_estimate', 0.0)),
        community_count=features.get('community_count', 0),
        graph_type=features.get('graph_type', 'unknown'),
        source_file=features.get('source_file', ''),
    )
    store.save_features(graph_features)


def save_benchmark_result(
    graph_name: str,
    algorithm_id: int,
    algorithm_name: str,
    benchmark: str,
    avg_time: float,
    trial_times: List[float],
    speedup: float,
    cache_stats: Optional[Dict] = None,
    data_dir: str = GRAPH_DATA_DIR
) -> None:
    """Save a single benchmark result."""
    store = GraphDataStore(graph_name, data_dir)
    result = AlgorithmBenchmarkData(
        graph_name=graph_name,
        algorithm_id=algorithm_id,
        algorithm_name=algorithm_name,
        benchmark=benchmark,
        avg_time=avg_time,
        trial_times=trial_times,
        speedup=speedup,
        l1_hit_rate=cache_stats.get('l1_hit_rate', 0.0) if cache_stats else 0.0,
        l2_hit_rate=cache_stats.get('l2_hit_rate', 0.0) if cache_stats else 0.0,
        l3_hit_rate=cache_stats.get('l3_hit_rate', 0.0) if cache_stats else 0.0,
        llc_misses=cache_stats.get('llc_misses', 0) if cache_stats else 0,
        num_trials=len(trial_times),
        success=True,
    )
    store.save_benchmark_result(result)


def save_reorder_result(
    graph_name: str,
    algorithm_id: int,
    algorithm_name: str,
    reorder_time: float,
    modularity: float = 0.0,
    communities: int = 0,
    isolated_vertices: int = 0,
    mapping_file: str = "",
    data_dir: str = GRAPH_DATA_DIR
) -> None:
    """Save a single reorder result."""
    store = GraphDataStore(graph_name, data_dir)
    result = AlgorithmReorderData(
        graph_name=graph_name,
        algorithm_id=algorithm_id,
        algorithm_name=algorithm_name,
        reorder_time=reorder_time,
        modularity=modularity,
        communities=communities,
        isolated_vertices=isolated_vertices,
        mapping_file=mapping_file,
        success=True,
    )
    store.save_reorder_result(result)


def list_graphs(data_dir: str = GRAPH_DATA_DIR) -> List[str]:
    """List all graphs with stored data."""
    if not os.path.exists(data_dir):
        return []
    return [d for d in os.listdir(data_dir) 
            if os.path.isdir(os.path.join(data_dir, d))]


def load_all_graph_data(data_dir: str = GRAPH_DATA_DIR) -> Dict[str, Dict]:
    """Load summary data for all graphs."""
    graphs = list_graphs(data_dir)
    all_data = {}
    for graph_name in graphs:
        store = GraphDataStore(graph_name, data_dir)
        all_data[graph_name] = store.get_summary()
    return all_data


def export_to_csv(
    output_file: str,
    data_dir: str = GRAPH_DATA_DIR,
    benchmarks: List[str] = None
) -> None:
    """Export all graph data to CSV for analysis."""
    benchmarks = benchmarks or ['pr', 'bfs', 'cc', 'sssp', 'bc']
    all_data = load_all_graph_data(data_dir)
    
    # Collect all algorithms
    all_algos = set()
    for graph_data in all_data.values():
        all_algos.update(graph_data.get('algorithms_tested', []))
    all_algos = sorted(all_algos)
    
    # Build header
    header = ['graph', 'nodes', 'edges', 'modularity', 'degree_variance', 'hub_concentration']
    for bench in benchmarks:
        header.append(f'best_algo_{bench}')
        header.append(f'best_speedup_{bench}')
        for algo in all_algos:
            header.append(f'{bench}_{algo}_time')
            header.append(f'{bench}_{algo}_speedup')
    
    # Write CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for graph_name, data in all_data.items():
            row = [graph_name]
            
            # Features
            features = data.get('features') or {}
            row.extend([
                features.get('nodes', 0),
                features.get('edges', 0),
                features.get('modularity', 0),
                features.get('degree_variance', 0),
                features.get('hub_concentration', 0),
            ])
            
            # Best per benchmark
            best_per_bench = data.get('best_per_benchmark', {})
            
            # Load full benchmark data
            store = GraphDataStore(graph_name, data_dir)
            all_benchmarks = store.load_all_benchmarks()
            
            # Organize by benchmark and algorithm
            bench_algo_data = {}
            for br in all_benchmarks:
                key = (br.benchmark, br.algorithm_name)
                bench_algo_data[key] = br
            
            for bench in benchmarks:
                best = best_per_bench.get(bench, {})
                row.append(best.get('algorithm', ''))
                row.append(best.get('speedup', 0))
                
                for algo in all_algos:
                    br = bench_algo_data.get((bench, algo))
                    if br:
                        row.append(br.avg_time)
                        row.append(br.speedup)
                    else:
                        row.append('')
                        row.append('')
            
            writer.writerow(row)
    
    log.info(f"Exported {len(all_data)} graphs to {output_file}")


def compute_and_save_weights(
    graph_name: str,
    data_dir: str = GRAPH_DATA_DIR,
    weights_module = None
) -> Optional[GraphPerceptronWeights]:
    """
    Compute perceptron weights for a graph based on its benchmark data.
    
    This analyzes the benchmark results to determine:
    - Which algorithm performed best for each benchmark
    - Relative speedups
    - Correlation with graph features
    """
    store = GraphDataStore(graph_name, data_dir)
    features = store.load_features()
    benchmarks = store.load_all_benchmarks()
    
    if not features or not benchmarks:
        log.warn(f"Insufficient data to compute weights for {graph_name}")
        return None
    
    # Group by benchmark
    by_benchmark: Dict[str, List[AlgorithmBenchmarkData]] = {}
    for br in benchmarks:
        if br.benchmark not in by_benchmark:
            by_benchmark[br.benchmark] = []
        by_benchmark[br.benchmark].append(br)
    
    # Find best per benchmark
    best_algo = {}
    best_speedup = {}
    algo_scores = {}
    
    for bench, results in by_benchmark.items():
        best = max(results, key=lambda x: x.speedup)
        best_algo[bench] = best.algorithm_name
        best_speedup[bench] = best.speedup
        
        # Store all scores
        algo_scores[bench] = {r.algorithm_name: r.speedup for r in results}
    
    # Create weights object
    weights = GraphPerceptronWeights(
        graph_name=graph_name,
        best_algo=best_algo,
        best_speedup=best_speedup,
        algo_scores=algo_scores,
    )
    
    # Assign type if weights module available
    if weights_module:
        try:
            features_dict = features.to_dict()
            type_name = weights_module.assign_graph_type(
                features_dict,
                graph_name=graph_name
            )
            weights.assigned_type = type_name
        except Exception as e:
            log.warn(f"Could not assign type for {graph_name}: {e}")
    
    store.save_weights(weights)
    return weights


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for graph data inspection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Graph data store utility")
    parser.add_argument("--list-graphs", action="store_true",
                       help="List all graphs with stored data")
    parser.add_argument("--show-graph", type=str,
                       help="Show summary for a specific graph")
    parser.add_argument("--export-csv", type=str,
                       help="Export all data to CSV file")
    parser.add_argument("--data-dir", type=str, default=GRAPH_DATA_DIR,
                       help="Data directory")
    parser.add_argument("--compute-weights", type=str,
                       help="Compute weights for a specific graph")
    parser.add_argument("--compute-all-weights", action="store_true",
                       help="Compute weights for all graphs")
    
    args = parser.parse_args()
    
    if args.list_graphs:
        graphs = list_graphs(args.data_dir)
        print(f"\nGraphs with stored data ({len(graphs)} total):")
        for g in sorted(graphs):
            store = GraphDataStore(g, args.data_dir)
            summary = store.get_summary()
            print(f"  {g}: {summary['num_benchmarks']} benchmarks, "
                  f"{summary['num_reorders']} reorders, "
                  f"features={'✓' if summary['has_features'] else '✗'}")
    
    elif args.show_graph:
        store = GraphDataStore(args.show_graph, args.data_dir)
        summary = store.get_summary()
        print(f"\nGraph: {args.show_graph}")
        print(json.dumps(summary, indent=2))
    
    elif args.export_csv:
        export_to_csv(args.export_csv, args.data_dir)
    
    elif args.compute_weights:
        weights = compute_and_save_weights(args.compute_weights, args.data_dir)
        if weights:
            print(f"Computed weights for {args.compute_weights}")
            print(json.dumps(weights.to_dict(), indent=2))
    
    elif args.compute_all_weights:
        graphs = list_graphs(args.data_dir)
        for graph in graphs:
            weights = compute_and_save_weights(graph, args.data_dir)
            if weights:
                print(f"  {graph}: best_algo = {weights.best_algo}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
