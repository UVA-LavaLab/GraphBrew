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
                             This is STATIC data about the graph itself
    
    results/logs/<graph_name>/
        runs/<timestamp>/
            benchmarks/
                <benchmark>_<algorithm>.json  - Benchmark results
            reorder/
                <algorithm>.json - Reorder time and mapping info
            weights/
                perceptron.json  - Per-graph computed weights
            summary.json         - Run summary with metadata
        
        reorder_<algorithm>_<timestamp>.log   - Reorder command output
        benchmark_<bench>_<algorithm>_<timestamp>.log - Benchmark output
        cache_<bench>_<algorithm>_<timestamp>.log    - Cache simulation output
            
Standalone usage:
    python -m scripts.lib.graph_data --list-graphs
    python -m scripts.lib.graph_data --show-graph email-Enron
    python -m scripts.lib.graph_data --export-csv results/all_data.csv
    python -m scripts.lib.graph_data --list-logs email-Enron
    python -m scripts.lib.graph_data --show-log email-Enron reorder_HUBCLUSTERDBG_20260127_123456.log
    python -m scripts.lib.graph_data --list-runs email-Enron
    python -m scripts.lib.graph_data --show-run email-Enron 20260127_145547

Library usage:
    from scripts.lib.graph_data import (
        GraphDataStore, GraphRunStore, save_graph_features,
        save_run_log, list_graph_logs, read_log, get_latest_run
    )
"""

import os
import json
import glob
import shutil
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

# Static graph features go here (topology data)
GRAPH_DATA_DIR = os.path.join(RESULTS_DIR, "graphs")

# Run-specific data and logs go here
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")

# Global session ID for grouping logs from one experiment run
_current_session_id = None


def set_session_id(session_id: str = None) -> str:
    """
    Set the current session ID for log grouping.
    
    If session_id is None, generates a new timestamp-based ID.
    Returns the session ID.
    """
    global _current_session_id
    if session_id is None:
        _current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        _current_session_id = session_id
    return _current_session_id


def get_session_id() -> str:
    """Get current session ID, creating one if needed."""
    global _current_session_id
    if _current_session_id is None:
        set_session_id()
    return _current_session_id


def clear_session_id():
    """Clear the session ID (for testing or manual runs)."""
    global _current_session_id
    _current_session_id = None


# =============================================================================
# Run Logging Functions (Individual Operation Logs)
# =============================================================================

def get_log_dir(graph_name: str, logs_dir: str = LOGS_DIR, use_session: bool = True) -> str:
    """
    Get log directory for a graph, creating if needed.
    
    Structure: logs/run_YYYYMMDD_HHMMSS/graph_name/
    Or without session: logs/graph_name/
    """
    if use_session:
        session_id = get_session_id()
        log_dir = os.path.join(logs_dir, f"run_{session_id}", graph_name)
    else:
        log_dir = os.path.join(logs_dir, graph_name)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def save_run_log(
    graph_name: str,
    operation: str,
    algorithm: str,
    output: str,
    benchmark: str = None,
    command: str = None,
    exit_code: int = None,
    duration: float = None,
    logs_dir: str = LOGS_DIR,
    use_session: bool = True
) -> str:
    """
    Save command output to a timestamped log file.
    
    Logs are organized by session (experiment run):
        logs/run_YYYYMMDD_HHMMSS/graph_name/operation_algorithm_timestamp.log
    
    Args:
        graph_name: Name of the graph
        operation: Type of operation ('reorder', 'benchmark', 'cache')
        algorithm: Algorithm name (e.g., 'HUBCLUSTERDBG', 'LeidenCSR')
        output: Command stdout/stderr output
        benchmark: Benchmark name for benchmark operations (e.g., 'pr', 'bfs')
        command: The command that was run
        exit_code: Command exit code
        duration: Execution duration in seconds
        logs_dir: Base logs directory
        use_session: If True, group logs under session folder
    
    Returns:
        Path to the saved log file
    """
    log_dir = get_log_dir(graph_name, logs_dir, use_session=use_session)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build filename
    if benchmark:
        filename = f"{operation}_{benchmark}_{algorithm}_{timestamp}.log"
    else:
        filename = f"{operation}_{algorithm}_{timestamp}.log"
    
    filepath = os.path.join(log_dir, filename)
    
    # Build log content with metadata header
    header_lines = [
        f"# GraphBrew Run Log",
        f"# Graph: {graph_name}",
        f"# Operation: {operation}",
        f"# Algorithm: {algorithm}",
    ]
    if benchmark:
        header_lines.append(f"# Benchmark: {benchmark}")
    header_lines.append(f"# Timestamp: {timestamp}")
    if command:
        header_lines.append(f"# Command: {command}")
    if exit_code is not None:
        header_lines.append(f"# Exit Code: {exit_code}")
    if duration is not None:
        header_lines.append(f"# Duration: {duration:.3f}s")
    header_lines.append("#" + "=" * 70)
    header_lines.append("")
    
    content = "\n".join(header_lines) + output
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    log.debug(f"Saved run log: {filepath}")
    return filepath


def list_graph_logs(graph_name: str, logs_dir: str = LOGS_DIR) -> List[Dict]:
    """
    List all log files for a graph.
    
    Searches both session-based and legacy (flat) log directories.
    
    Returns:
        List of dicts with 'filename', 'operation', 'algorithm', 'benchmark', 'timestamp', 'session'
    """
    logs = []
    
    # Check session-based directories (logs/run_*/graph_name/)
    if os.path.exists(logs_dir):
        for entry in os.listdir(logs_dir):
            if entry.startswith('run_'):
                session_log_dir = os.path.join(logs_dir, entry, graph_name)
                if os.path.exists(session_log_dir):
                    logs.extend(_parse_log_dir(session_log_dir, session=entry[4:]))  # Remove 'run_' prefix
    
    # Check legacy flat directory (logs/graph_name/)
    legacy_dir = os.path.join(logs_dir, graph_name)
    if os.path.exists(legacy_dir) and os.path.isdir(legacy_dir):
        # Only if it's not a run_* directory
        if not graph_name.startswith('run_'):
            logs.extend(_parse_log_dir(legacy_dir, session=None))
    
    return sorted(logs, key=lambda x: x.get('timestamp', ''), reverse=True)


def _parse_log_dir(log_dir: str, session: str = None) -> List[Dict]:
    """Parse log files from a directory."""
    logs = []
    for filename in os.listdir(log_dir):
        if not filename.endswith('.log'):
            continue
        
        # Parse filename: operation_[benchmark_]algorithm_timestamp.log
        parts = filename[:-4].split('_')  # Remove .log
        
        if len(parts) >= 3:
            operation = parts[0]
            timestamp = '_'.join(parts[-2:])  # Last two parts are YYYYMMDD_HHMMSS
            
            # Check if there's a benchmark (operations: benchmark, cache have benchmarks)
            # Format with benchmark: operation_benchmark_algorithm_YYYYMMDD_HHMMSS.log (5+ parts)
            # Format without: operation_algorithm_YYYYMMDD_HHMMSS.log (4 parts)
            if len(parts) >= 5 and operation in ('benchmark', 'cache'):
                benchmark = parts[1]
                algorithm = '_'.join(parts[2:-2])
            else:
                benchmark = None
                algorithm = '_'.join(parts[1:-2])
            
            logs.append({
                'filename': filename,
                'filepath': os.path.join(log_dir, filename),
                'operation': operation,
                'algorithm': algorithm,
                'benchmark': benchmark,
                'timestamp': timestamp,
                'session': session,
            })
    
    return logs


def list_sessions(logs_dir: str = LOGS_DIR) -> List[Dict]:
    """
    List all experiment sessions.
    
    Returns:
        List of dicts with 'session_id', 'path', 'graphs'
    """
    sessions = []
    if not os.path.exists(logs_dir):
        return sessions
    
    for entry in sorted(os.listdir(logs_dir), reverse=True):
        if entry.startswith('run_'):
            session_dir = os.path.join(logs_dir, entry)
            if os.path.isdir(session_dir):
                graphs = [g for g in os.listdir(session_dir) if os.path.isdir(os.path.join(session_dir, g))]
                sessions.append({
                    'session_id': entry[4:],  # Remove 'run_' prefix
                    'path': session_dir,
                    'graphs': graphs,
                    'graph_count': len(graphs),
                })
    
    return sessions


def read_log(filepath: str) -> Optional[str]:
    """Read contents of a log file by its full path."""
    if not os.path.exists(filepath):
        return None
    with open(filepath) as f:
        return f.read()


def read_log_by_name(graph_name: str, filename: str, logs_dir: str = LOGS_DIR, session: str = None) -> Optional[str]:
    """
    Read contents of a log file by graph name and filename.
    
    Args:
        graph_name: Name of the graph
        filename: Log filename
        logs_dir: Base logs directory
        session: Session ID (if None, searches legacy directory)
    """
    if session:
        filepath = os.path.join(logs_dir, f"run_{session}", graph_name, filename)
    else:
        filepath = os.path.join(logs_dir, graph_name, filename)
    
    if not os.path.exists(filepath):
        return None
    with open(filepath) as f:
        return f.read()


def get_latest_log(
    graph_name: str,
    operation: str = None,
    algorithm: str = None,
    benchmark: str = None,
    logs_dir: str = LOGS_DIR
) -> Optional[str]:
    """
    Get the most recent log file matching the criteria.
    
    Returns:
        Path to the latest matching log file, or None
    """
    logs = list_graph_logs(graph_name, logs_dir)
    
    # Filter by criteria
    if operation:
        logs = [l for l in logs if l['operation'] == operation]
    if algorithm:
        logs = [l for l in logs if l['algorithm'] == algorithm]
    if benchmark:
        logs = [l for l in logs if l['benchmark'] == benchmark]
    
    if not logs:
        return None
    
    # Sort by timestamp (already sorted by filename, but be explicit)
    logs.sort(key=lambda x: x['timestamp'], reverse=True)
    return logs[0]['filepath']


def cleanup_old_logs(
    graph_name: str = None,
    max_logs_per_graph: int = 100,
    max_age_days: int = 30,
    logs_dir: str = LOGS_DIR
) -> int:
    """
    Clean up old log files to prevent disk bloat.
    
    Args:
        graph_name: Specific graph to clean, or None for all
        max_logs_per_graph: Keep at most this many logs per graph
        max_age_days: Delete logs older than this
        logs_dir: Base logs directory
    
    Returns:
        Number of files deleted
    """
    deleted = 0
    cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
    
    if graph_name:
        graphs = [graph_name]
    else:
        graphs = [d for d in os.listdir(logs_dir) 
                  if os.path.isdir(os.path.join(logs_dir, d))]
    
    for graph in graphs:
        graph_log_dir = os.path.join(logs_dir, graph)
        if not os.path.exists(graph_log_dir):
            continue
        
        log_files = []
        for filename in os.listdir(graph_log_dir):
            if filename.endswith('.log'):
                filepath = os.path.join(graph_log_dir, filename)
                mtime = os.path.getmtime(filepath)
                log_files.append((filepath, mtime))
        
        # Sort by modification time (newest first)
        log_files.sort(key=lambda x: x[1], reverse=True)
        
        for i, (filepath, mtime) in enumerate(log_files):
            # Delete if too old or exceeds max count
            if mtime < cutoff_time or i >= max_logs_per_graph:
                try:
                    os.remove(filepath)
                    deleted += 1
                except OSError:
                    pass
    
    if deleted > 0:
        log.info(f"Cleaned up {deleted} old log files")
    
    return deleted


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GraphFeatures:
    """Graph topology features (STATIC - stored in graphs/ folder)."""
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
    """Benchmark data for a single algorithm on a single graph (run-specific)."""
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
    """Reorder data for a single algorithm on a single graph (run-specific)."""
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
    """Per-graph computed perceptron weights (run-specific, for analysis)."""
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
# Graph Features Store (Static Data - in graphs/ folder)
# =============================================================================

class GraphFeaturesStore:
    """
    Manages STATIC graph topology features.
    
    Features are stored in results/graphs/<graph>/ since they describe
    the graph itself and don't change between experiment runs.
    """
    
    def __init__(self, graph_name: str, data_dir: str = GRAPH_DATA_DIR):
        """Initialize store for a specific graph."""
        self.graph_name = graph_name
        self.base_dir = os.path.join(data_dir, graph_name)
        os.makedirs(self.base_dir, exist_ok=True)
    
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


# =============================================================================
# Graph Run Store (Run-Specific Data - in logs/ folder)
# =============================================================================

class GraphRunStore:
    """
    Manages RUN-SPECIFIC experiment data (benchmarks, reorder times, weights).
    
    Each run is stored in results/logs/<graph>/runs/<timestamp>/ with:
    - benchmarks/<bench>_<algo>.json
    - reorder/<algo>.json
    - weights/perceptron.json
    - summary.json
    
    This allows keeping historical run data while the static features
    remain in the graphs/ folder.
    """
    
    def __init__(self, graph_name: str, run_timestamp: str = None, logs_dir: str = LOGS_DIR):
        """
        Initialize store for a specific graph run.
        
        Args:
            graph_name: Name of the graph
            run_timestamp: Timestamp for this run (default: current time)
            logs_dir: Base logs directory
        """
        self.graph_name = graph_name
        self.run_timestamp = run_timestamp or get_timestamp()
        self.logs_dir = logs_dir
        
        self.runs_dir = os.path.join(logs_dir, graph_name, "runs")
        self.run_dir = os.path.join(self.runs_dir, self.run_timestamp)
        self.benchmarks_dir = os.path.join(self.run_dir, "benchmarks")
        self.reorder_dir = os.path.join(self.run_dir, "reorder")
        self.weights_dir = os.path.join(self.run_dir, "weights")
        
        # Create directories
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.benchmarks_dir, exist_ok=True)
        os.makedirs(self.reorder_dir, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Benchmark Results
    # -------------------------------------------------------------------------
    
    def save_benchmark_result(self, result: AlgorithmBenchmarkData) -> None:
        """Save benchmark result for an algorithm."""
        result.timestamp = self.run_timestamp
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
            log.warn(f"Failed to load benchmark {benchmark}/{algorithm_name}: {e}")
            return None
    
    def load_all_benchmarks(self) -> List[AlgorithmBenchmarkData]:
        """Load all benchmark results for this run."""
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
        result.timestamp = self.run_timestamp
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
            log.warn(f"Failed to load reorder {algorithm_name}: {e}")
            return None
    
    def load_all_reorders(self) -> List[AlgorithmReorderData]:
        """Load all reorder results for this run."""
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
        """Save computed perceptron weights for this run."""
        weights.timestamp = self.run_timestamp
        weights_file = os.path.join(self.weights_dir, "perceptron.json")
        with open(weights_file, 'w') as f:
            json.dump(weights.to_dict(), f, indent=2)
        log.debug(f"Saved weights for {self.graph_name}")
    
    def load_weights(self) -> Optional[GraphPerceptronWeights]:
        """Load computed perceptron weights for this run."""
        weights_file = os.path.join(self.weights_dir, "perceptron.json")
        if not os.path.exists(weights_file):
            return None
        try:
            with open(weights_file) as f:
                data = json.load(f)
            return GraphPerceptronWeights.from_dict(data)
        except Exception as e:
            log.warn(f"Failed to load weights: {e}")
            return None
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    
    def save_summary(self, metadata: Dict = None) -> None:
        """Save run summary."""
        benchmarks = self.load_all_benchmarks()
        reorders = self.load_all_reorders()
        
        # Find best per benchmark
        best_per_bench = {}
        for br in benchmarks:
            bench = br.benchmark
            if bench not in best_per_bench or br.speedup > best_per_bench[bench]['speedup']:
                best_per_bench[bench] = {
                    'algorithm': br.algorithm_name,
                    'speedup': br.speedup,
                    'time': br.avg_time
                }
        
        summary = {
            'graph_name': self.graph_name,
            'run_timestamp': self.run_timestamp,
            'num_benchmarks': len(benchmarks),
            'num_reorders': len(reorders),
            'best_per_benchmark': best_per_bench,
            'algorithms_tested': list(set(b.algorithm_name for b in benchmarks)),
            'benchmarks_run': list(set(b.benchmark for b in benchmarks)),
            'metadata': metadata or {},
        }
        
        summary_file = os.path.join(self.run_dir, "summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def load_summary(self) -> Optional[Dict]:
        """Load run summary."""
        summary_file = os.path.join(self.run_dir, "summary.json")
        if not os.path.exists(summary_file):
            return None
        try:
            with open(summary_file) as f:
                return json.load(f)
        except Exception:
            return None
    
    def get_summary(self) -> Dict:
        """Get summary (load or compute)."""
        summary = self.load_summary()
        if summary:
            return summary
        
        # Compute on the fly
        benchmarks = self.load_all_benchmarks()
        reorders = self.load_all_reorders()
        weights = self.load_weights()
        
        best_per_bench = {}
        for br in benchmarks:
            bench = br.benchmark
            if bench not in best_per_bench or br.speedup > best_per_bench[bench]['speedup']:
                best_per_bench[bench] = {
                    'algorithm': br.algorithm_name,
                    'speedup': br.speedup,
                    'time': br.avg_time
                }
        
        return {
            'graph_name': self.graph_name,
            'run_timestamp': self.run_timestamp,
            'num_benchmarks': len(benchmarks),
            'num_reorders': len(reorders),
            'has_weights': weights is not None,
            'best_per_benchmark': best_per_bench,
            'algorithms_tested': list(set(b.algorithm_name for b in benchmarks)),
            'benchmarks_run': list(set(b.benchmark for b in benchmarks)),
        }


# =============================================================================
# Combined Graph Data Store (for backward compatibility)
# =============================================================================

class GraphDataStore:
    """
    Unified interface for graph data (features + latest run data).
    
    This maintains backward compatibility with the previous API while
    using the new separated storage structure:
    - Static features in results/graphs/<graph>/
    - Run data in results/logs/<graph>/runs/<timestamp>/
    """
    
    def __init__(self, graph_name: str, data_dir: str = GRAPH_DATA_DIR, 
                 logs_dir: str = LOGS_DIR, run_timestamp: str = None,
                 create_new_run: bool = False):
        """
        Initialize combined store.
        
        Args:
            graph_name: Name of the graph
            data_dir: Directory for static features (default: results/graphs/)
            logs_dir: Directory for run data (default: results/logs/)
            run_timestamp: Specific run timestamp to use (default: use latest or create new)
            create_new_run: If True, always create a new run instead of using latest
        """
        self.graph_name = graph_name
        self.data_dir = data_dir
        self.logs_dir = logs_dir
        
        # Features store (static)
        self.features_store = GraphFeaturesStore(graph_name, data_dir)
        
        # Run store (dynamic) - use latest if not specified, or create new
        if run_timestamp is not None:
            # Explicit timestamp provided
            self.run_timestamp = run_timestamp
        elif create_new_run:
            # Force new run
            self.run_timestamp = get_timestamp()
        else:
            # Use latest or create new
            self.run_timestamp = self._get_latest_run_timestamp() or get_timestamp()
        
        self.run_store = GraphRunStore(graph_name, self.run_timestamp, logs_dir)
        
        # Legacy compatibility paths
        self.base_dir = self.features_store.base_dir
        self.benchmarks_dir = self.run_store.benchmarks_dir
        self.reorder_dir = self.run_store.reorder_dir
        self.weights_dir = self.run_store.weights_dir
    
    def _get_latest_run_timestamp(self) -> Optional[str]:
        """Get the most recent run timestamp for this graph."""
        runs_dir = os.path.join(self.logs_dir, self.graph_name, "runs")
        if not os.path.exists(runs_dir):
            return None
        runs = sorted([d for d in os.listdir(runs_dir) 
                      if os.path.isdir(os.path.join(runs_dir, d))], reverse=True)
        return runs[0] if runs else None
    
    # -------------------------------------------------------------------------
    # Features (static - delegates to features store)
    # -------------------------------------------------------------------------
    
    def save_features(self, features: GraphFeatures) -> None:
        """Save graph topology features."""
        self.features_store.save_features(features)
    
    def load_features(self) -> Optional[GraphFeatures]:
        """Load graph topology features."""
        return self.features_store.load_features()
    
    # -------------------------------------------------------------------------
    # Benchmark Results (run-specific - delegates to run store)
    # -------------------------------------------------------------------------
    
    def save_benchmark_result(self, result: AlgorithmBenchmarkData) -> None:
        """Save benchmark result."""
        self.run_store.save_benchmark_result(result)
    
    def load_benchmark_result(self, benchmark: str, algorithm_name: str) -> Optional[AlgorithmBenchmarkData]:
        """Load benchmark result."""
        return self.run_store.load_benchmark_result(benchmark, algorithm_name)
    
    def load_all_benchmarks(self) -> List[AlgorithmBenchmarkData]:
        """Load all benchmarks from latest run."""
        return self.run_store.load_all_benchmarks()
    
    # -------------------------------------------------------------------------
    # Reorder Results (run-specific - delegates to run store)
    # -------------------------------------------------------------------------
    
    def save_reorder_result(self, result: AlgorithmReorderData) -> None:
        """Save reorder result."""
        self.run_store.save_reorder_result(result)
    
    def load_reorder_result(self, algorithm_name: str) -> Optional[AlgorithmReorderData]:
        """Load reorder result."""
        return self.run_store.load_reorder_result(algorithm_name)
    
    def load_all_reorders(self) -> List[AlgorithmReorderData]:
        """Load all reorders from latest run."""
        return self.run_store.load_all_reorders()
    
    # -------------------------------------------------------------------------
    # Weights (run-specific - delegates to run store)
    # -------------------------------------------------------------------------
    
    def save_weights(self, weights: GraphPerceptronWeights) -> None:
        """Save weights."""
        self.run_store.save_weights(weights)
    
    def load_weights(self) -> Optional[GraphPerceptronWeights]:
        """Load weights."""
        return self.run_store.load_weights()
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    
    def get_summary(self) -> Dict:
        """Get combined summary of features + latest run."""
        features = self.load_features()
        run_summary = self.run_store.get_summary()
        
        return {
            'graph_name': self.graph_name,
            'has_features': features is not None,
            'features': features.to_dict() if features else None,
            'run_timestamp': self.run_timestamp,
            'num_benchmarks': run_summary.get('num_benchmarks', 0),
            'num_reorders': run_summary.get('num_reorders', 0),
            'has_weights': run_summary.get('has_weights', False),
            'best_per_benchmark': run_summary.get('best_per_benchmark', {}),
            'algorithms_tested': run_summary.get('algorithms_tested', []),
            'benchmarks_run': run_summary.get('benchmarks_run', []),
        }


# =============================================================================
# Run Management Functions
# =============================================================================

def list_runs(graph_name: str, logs_dir: str = LOGS_DIR) -> List[str]:
    """List all run timestamps for a graph."""
    runs_dir = os.path.join(logs_dir, graph_name, "runs")
    if not os.path.exists(runs_dir):
        return []
    return sorted([d for d in os.listdir(runs_dir) 
                   if os.path.isdir(os.path.join(runs_dir, d))], reverse=True)


def get_latest_run(graph_name: str, logs_dir: str = LOGS_DIR) -> Optional[GraphRunStore]:
    """Get the latest run store for a graph."""
    runs = list_runs(graph_name, logs_dir)
    if not runs:
        return None
    return GraphRunStore(graph_name, runs[0], logs_dir)


def cleanup_old_runs(
    graph_name: str = None,
    max_runs_per_graph: int = 10,
    logs_dir: str = LOGS_DIR
) -> int:
    """
    Clean up old run directories to prevent disk bloat.
    
    Returns:
        Number of runs deleted
    """
    deleted = 0
    
    if graph_name:
        graphs = [graph_name]
    else:
        graphs = [d for d in os.listdir(logs_dir) 
                  if os.path.isdir(os.path.join(logs_dir, d))]
    
    for graph in graphs:
        runs = list_runs(graph, logs_dir)
        
        # Keep only the newest max_runs_per_graph
        for run_ts in runs[max_runs_per_graph:]:
            run_dir = os.path.join(logs_dir, graph, "runs", run_ts)
            try:
                shutil.rmtree(run_dir)
                deleted += 1
            except OSError:
                pass
    
    if deleted > 0:
        log.info(f"Cleaned up {deleted} old run directories")
    
    return deleted


# =============================================================================
# Convenience Functions
# =============================================================================

def save_graph_features(graph_name: str, features: Dict, data_dir: str = GRAPH_DATA_DIR) -> None:
    """Save features for a graph from a dictionary."""
    store = GraphFeaturesStore(graph_name, data_dir)
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


def list_graphs(data_dir: str = GRAPH_DATA_DIR) -> List[str]:
    """List all graphs with stored data (features)."""
    if not os.path.exists(data_dir):
        return []
    return [d for d in os.listdir(data_dir) 
            if os.path.isdir(os.path.join(data_dir, d))]


def load_all_graph_data(data_dir: str = GRAPH_DATA_DIR, logs_dir: str = LOGS_DIR) -> Dict[str, Dict]:
    """Load summary data for all graphs."""
    graphs = list_graphs(data_dir)
    all_data = {}
    for graph_name in graphs:
        store = GraphDataStore(graph_name, data_dir, logs_dir)
        all_data[graph_name] = store.get_summary()
    return all_data


def export_to_csv(
    output_file: str,
    data_dir: str = GRAPH_DATA_DIR,
    logs_dir: str = LOGS_DIR,
    benchmarks: List[str] = None
) -> None:
    """Export all graph data to CSV for analysis."""
    benchmarks = benchmarks or ['pr', 'bfs', 'cc', 'sssp', 'bc']
    all_data = load_all_graph_data(data_dir, logs_dir)
    
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
            store = GraphDataStore(graph_name, data_dir, logs_dir)
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


# =============================================================================
# Migration Helper
# =============================================================================

def migrate_old_structure(graph_name: str, data_dir: str = GRAPH_DATA_DIR, 
                          logs_dir: str = LOGS_DIR) -> bool:
    """
    Migrate data from old structure (all in graphs/) to new structure.
    
    Old: results/graphs/<graph>/benchmarks/, reorder/, weights/
    New: results/logs/<graph>/runs/<timestamp>/benchmarks/, reorder/, weights/
    """
    old_base = os.path.join(data_dir, graph_name)
    old_benchmarks = os.path.join(old_base, "benchmarks")
    old_reorder = os.path.join(old_base, "reorder")
    old_weights = os.path.join(old_base, "weights")
    
    # Check if old structure exists
    has_old_data = (os.path.exists(old_benchmarks) or 
                    os.path.exists(old_reorder) or 
                    os.path.exists(old_weights))
    
    if not has_old_data:
        return False
    
    # Create new run with migration timestamp
    migration_ts = get_timestamp()
    new_run = GraphRunStore(graph_name, f"migrated_{migration_ts}", logs_dir)
    
    # Move benchmarks
    if os.path.exists(old_benchmarks):
        for f in os.listdir(old_benchmarks):
            src = os.path.join(old_benchmarks, f)
            dst = os.path.join(new_run.benchmarks_dir, f)
            shutil.copy2(src, dst)
        shutil.rmtree(old_benchmarks)
    
    # Move reorder
    if os.path.exists(old_reorder):
        for f in os.listdir(old_reorder):
            src = os.path.join(old_reorder, f)
            dst = os.path.join(new_run.reorder_dir, f)
            shutil.copy2(src, dst)
        shutil.rmtree(old_reorder)
    
    # Move weights
    if os.path.exists(old_weights):
        for f in os.listdir(old_weights):
            src = os.path.join(old_weights, f)
            dst = os.path.join(new_run.weights_dir, f)
            shutil.copy2(src, dst)
        shutil.rmtree(old_weights)
    
    # Save summary
    new_run.save_summary({'migrated_from': 'old_structure', 'migration_ts': migration_ts})
    
    log.info(f"Migrated {graph_name} data to new structure")
    return True


def migrate_all_graphs(data_dir: str = GRAPH_DATA_DIR, logs_dir: str = LOGS_DIR) -> int:
    """Migrate all graphs from old structure to new."""
    migrated = 0
    for graph_name in list_graphs(data_dir):
        if migrate_old_structure(graph_name, data_dir, logs_dir):
            migrated += 1
    return migrated


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
                       help="Data directory (for features)")
    
    # Run management
    parser.add_argument("--list-runs", type=str, metavar="GRAPH",
                       help="List all runs for a specific graph")
    parser.add_argument("--show-run", nargs=2, metavar=("GRAPH", "TIMESTAMP"),
                       help="Show summary for a specific run")
    parser.add_argument("--cleanup-runs", action="store_true",
                       help="Clean up old run directories")
    parser.add_argument("--max-runs", type=int, default=10,
                       help="Max runs per graph for cleanup (default: 10)")
    
    # Log management options
    parser.add_argument("--list-logs", type=str, metavar="GRAPH",
                       help="List all logs for a specific graph")
    parser.add_argument("--show-log", nargs=2, metavar=("GRAPH", "FILENAME"),
                       help="Show contents of a specific log file")
    parser.add_argument("--latest-log", type=str, metavar="GRAPH",
                       help="Show the latest log for a graph")
    parser.add_argument("--cleanup-logs", action="store_true",
                       help="Clean up old log files")
    parser.add_argument("--max-logs", type=int, default=100,
                       help="Max logs per graph for cleanup (default: 100)")
    parser.add_argument("--max-age-days", type=int, default=30,
                       help="Max age in days for cleanup (default: 30)")
    parser.add_argument("--logs-dir", type=str, default=LOGS_DIR,
                       help="Logs directory")
    
    # Migration
    parser.add_argument("--migrate", action="store_true",
                       help="Migrate old data structure to new")
    parser.add_argument("--migrate-graph", type=str,
                       help="Migrate a specific graph")
    
    args = parser.parse_args()
    
    if args.list_graphs:
        graphs = list_graphs(args.data_dir)
        print(f"\nGraphs with stored data ({len(graphs)} total):")
        for g in sorted(graphs):
            store = GraphDataStore(g, args.data_dir, args.logs_dir)
            summary = store.get_summary()
            runs = list_runs(g, args.logs_dir)
            print(f"  {g}: {summary['num_benchmarks']} benchmarks, "
                  f"{summary['num_reorders']} reorders, "
                  f"features={'✓' if summary['has_features'] else '✗'}, "
                  f"runs={len(runs)}")
    
    elif args.show_graph:
        store = GraphDataStore(args.show_graph, args.data_dir, args.logs_dir)
        summary = store.get_summary()
        print(f"\nGraph: {args.show_graph}")
        print(json.dumps(summary, indent=2))
    
    elif args.list_runs:
        runs = list_runs(args.list_runs, args.logs_dir)
        print(f"\nRuns for {args.list_runs} ({len(runs)} total):")
        for run_ts in runs:
            run_store = GraphRunStore(args.list_runs, run_ts, args.logs_dir)
            summary = run_store.get_summary()
            print(f"  {run_ts}: {summary['num_benchmarks']} benchmarks, "
                  f"{summary['num_reorders']} reorders")
    
    elif args.show_run:
        graph, timestamp = args.show_run
        run_store = GraphRunStore(graph, timestamp, args.logs_dir)
        summary = run_store.get_summary()
        print(f"\nRun {timestamp} for {graph}:")
        print(json.dumps(summary, indent=2))
    
    elif args.cleanup_runs:
        deleted = cleanup_old_runs(max_runs_per_graph=args.max_runs, logs_dir=args.logs_dir)
        print(f"Cleaned up {deleted} old run directories")
    
    elif args.export_csv:
        export_to_csv(args.export_csv, args.data_dir, args.logs_dir)
    
    elif args.list_logs:
        logs = list_graph_logs(args.list_logs, args.logs_dir)
        print(f"\nLogs for {args.list_logs} ({len(logs)} total):")
        for l in logs:
            bench_str = f" [{l['benchmark']}]" if l['benchmark'] else ""
            print(f"  {l['filename']}: {l['operation']}{bench_str} {l['algorithm']}")
    
    elif args.show_log:
        graph, filename = args.show_log
        content = read_log(graph, filename, args.logs_dir)
        if content:
            print(content)
        else:
            print(f"Log not found: {graph}/{filename}")
    
    elif args.latest_log:
        filepath = get_latest_log(args.latest_log, logs_dir=args.logs_dir)
        if filepath:
            print(f"Latest log: {filepath}")
            with open(filepath) as f:
                print(f.read())
        else:
            print(f"No logs found for {args.latest_log}")
    
    elif args.cleanup_logs:
        deleted = cleanup_old_logs(
            max_logs_per_graph=args.max_logs,
            max_age_days=args.max_age_days,
            logs_dir=args.logs_dir
        )
        print(f"Cleaned up {deleted} old log files")
    
    elif args.migrate:
        migrated = migrate_all_graphs(args.data_dir, args.logs_dir)
        print(f"Migrated {migrated} graphs to new structure")
    
    elif args.migrate_graph:
        if migrate_old_structure(args.migrate_graph, args.data_dir, args.logs_dir):
            print(f"Migrated {args.migrate_graph} to new structure")
        else:
            print(f"No old data found for {args.migrate_graph}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
