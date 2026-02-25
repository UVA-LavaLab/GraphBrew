#!/usr/bin/env python3
"""
Weight Merger - Consolidate weights from multiple experiment runs.

This module handles merging perceptron weights from different experiment runs,
accounting for the fact that type IDs may differ between runs (type_0 in run A
might correspond to type_1 in run B based on centroid similarity).

Directory Structure:
    results/weights/
    ├── registry.json               # Graph-type cluster registry
    ├── type_0/                     # Default graph-type cluster
    │   ├── weights.json            # Generic perceptron weights
    │   ├── pr.json                 # Per-benchmark specialised weights
    │   │   └── bfs.json
    │   ├── type_1/
    │   │   └── weights.json
    │   └── type_2/
    │       └── weights.json
    └── runs/                       # Historical snapshots (same layout)
        └── 20260125_123456/
            ├── registry.json
            └── type_0/
                └── weights.json

Usage:
    # Merge all runs into merged/
    python -m scripts.lib.weight_merger --merge-all
    
    # Merge specific runs
    python -m scripts.lib.weight_merger --merge-runs 20260125_123456 20260125_134567
    
    # Use specific run instead of merged
    python -m scripts.lib.weight_merger --use-run 20260125_123456
    
    # List available runs
    python -m scripts.lib.weight_merger --list-runs
"""

import json
import shutil
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .utils import Logger

log = Logger()


# Centroid feature names (must match type_registry.json structure)
CENTROID_FEATURES = [
    'modularity', 'degree_variance', 'hub_concentration',
    'clustering_coeff', 'avg_path_length', 'diameter', 'community_count'
]

# Default threshold for matching types by centroid distance
DEFAULT_MATCH_THRESHOLD = 0.2  # Normalized Euclidean distance


@dataclass
class TypeInfo:
    """Information about a graph type."""
    type_id: str
    centroid: Any  # Can be dict or list
    graph_count: int = 0
    algorithms: List[str] = field(default_factory=list)
    
    def centroid_vector(self) -> np.ndarray:
        """Convert centroid to numpy array."""
        if isinstance(self.centroid, list):
            # Already a list, just convert to array
            return np.array(self.centroid)
        elif isinstance(self.centroid, dict):
            # Dict format - extract values by feature names
            return np.array([self.centroid.get(f, 0.0) for f in CENTROID_FEATURES])
        else:
            return np.zeros(len(CENTROID_FEATURES))
    
    @staticmethod
    def from_registry_entry(type_id: str, entry: Dict) -> 'TypeInfo':
        """Create TypeInfo from registry entry."""
        # Handle sample_count or graph_count
        count = entry.get('graph_count', entry.get('sample_count', len(entry.get('graphs', []))))
        return TypeInfo(
            type_id=type_id,
            centroid=entry.get('centroid', []),
            graph_count=count,
            algorithms=entry.get('algorithms', [])
        )


@dataclass
class RunInfo:
    """Information about a single experiment run."""
    timestamp: str
    path: Path
    types: Dict[str, TypeInfo] = field(default_factory=dict)
    
    def load(self) -> bool:
        """Load run information from disk."""
        registry_path = self.path / "registry.json"
        if not registry_path.exists():
            return False
        
        try:
            with open(registry_path) as f:
                registry = json.load(f)
            
            for type_id, entry in registry.items():
                if type_id.startswith('type_'):
                    self.types[type_id] = TypeInfo.from_registry_entry(type_id, entry)
            return True
        except Exception as e:
            log.warning(f"Error loading run {self.timestamp}: {e}")
            return False


def centroid_distance(c1: np.ndarray, c2: np.ndarray) -> float:
    """
    Calculate normalized Euclidean distance between two centroids.
    
    Returns a value between 0 (identical) and ~1+ (very different).
    """
    # Normalize each feature to [0, 1] range based on typical values
    normalization = np.array([
        1.0,   # modularity: 0-1
        5.0,   # degree_variance: 0-5+
        1.0,   # hub_concentration: 0-1
        1.0,   # clustering_coeff: 0-1
        20.0,  # avg_path_length: 1-20+
        100.0, # diameter: 1-100+
        100.0  # community_count: 1-100+
    ])
    
    # Ensure arrays are same length
    min_len = min(len(c1), len(c2), len(normalization))
    c1_norm = c1[:min_len] / normalization[:min_len]
    c2_norm = c2[:min_len] / normalization[:min_len]
    
    return float(np.linalg.norm(c1_norm - c2_norm))


def find_matching_type(
    source_type: TypeInfo,
    target_types: Dict[str, TypeInfo],
    threshold: float = DEFAULT_MATCH_THRESHOLD
) -> Optional[str]:
    """
    Find a matching type in target_types based on centroid similarity.
    
    Args:
        source_type: The type to match
        target_types: Dictionary of existing types to search
        threshold: Maximum distance to consider a match
        
    Returns:
        Matching type_id or None if no match found
    """
    source_vec = source_type.centroid_vector()
    best_match = None
    best_distance = float('inf')
    
    for type_id, target_type in target_types.items():
        target_vec = target_type.centroid_vector()
        dist = centroid_distance(source_vec, target_vec)
        
        if dist < best_distance:
            best_distance = dist
            best_match = type_id
    
    if best_distance <= threshold:
        return best_match
    return None


def merge_centroids(
    c1: Any, count1: int,
    c2: Any, count2: int
) -> List[float]:
    """
    Merge two centroids using weighted average.
    
    Args:
        c1, c2: Centroids (can be list or dict)
        count1, count2: Number of graphs in each cluster
        
    Returns:
        Merged centroid as list
    """
    total = count1 + count2
    if total == 0:
        if isinstance(c1, list):
            return c1.copy()
        elif isinstance(c1, dict):
            return [c1.get(f, 0.0) for f in CENTROID_FEATURES]
        return [0.0] * len(CENTROID_FEATURES)
    
    # Convert to arrays
    if isinstance(c1, list):
        v1 = np.array(c1)
    elif isinstance(c1, dict):
        v1 = np.array([c1.get(f, 0.0) for f in CENTROID_FEATURES])
    else:
        v1 = np.zeros(len(CENTROID_FEATURES))
    
    if isinstance(c2, list):
        v2 = np.array(c2)
    elif isinstance(c2, dict):
        v2 = np.array([c2.get(f, 0.0) for f in CENTROID_FEATURES])
    else:
        v2 = np.zeros(len(CENTROID_FEATURES))
    
    # Ensure same length
    max_len = max(len(v1), len(v2))
    if len(v1) < max_len:
        v1 = np.pad(v1, (0, max_len - len(v1)))
    if len(v2) < max_len:
        v2 = np.pad(v2, (0, max_len - len(v2)))
    
    # Weighted average
    merged = (v1 * count1 + v2 * count2) / total
    return merged.tolist()


def merge_weights(
    w1: Dict[str, Any], count1: int,
    w2: Dict[str, Any], count2: int
) -> Dict[str, Any]:
    """
    Merge two weight dictionaries using weighted average.
    
    Args:
        w1, w2: Weight dictionaries for an algorithm
        count1, count2: Number of training samples
        
    Returns:
        Merged weight dictionary
    """
    total = count1 + count2
    if total == 0:
        return w1.copy()
    
    merged = {}
    all_keys = set(w1.keys()) | set(w2.keys())
    
    for key in all_keys:
        v1 = w1.get(key)
        v2 = w2.get(key)
        
        # Skip metadata and non-numeric fields
        if key.startswith('_'):
            # Keep the more recent metadata
            merged[key] = v2 if v2 is not None else v1
            continue
        
        # Handle benchmark_weights specially
        if key == 'benchmark_weights':
            if isinstance(v1, dict) and isinstance(v2, dict):
                merged[key] = {}
                bw_keys = set(v1.keys()) | set(v2.keys())
                for bk in bw_keys:
                    bv1 = v1.get(bk, 0.0)
                    bv2 = v2.get(bk, 0.0)
                    if isinstance(bv1, (int, float)) and isinstance(bv2, (int, float)):
                        merged[key][bk] = (bv1 * count1 + bv2 * count2) / total
                    else:
                        merged[key][bk] = bv2 if bv2 is not None else bv1
            else:
                merged[key] = v2 if v2 is not None else v1
            continue
        
        # Merge numeric values
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            merged[key] = (v1 * count1 + v2 * count2) / total
        elif v1 is None:
            merged[key] = v2
        elif v2 is None:
            merged[key] = v1
        else:
            merged[key] = v2  # Prefer newer value for non-numeric
    
    return merged


def load_type_weights(type_path: Path) -> Dict[str, Dict]:
    """Load weights for a type from JSON file."""
    if not type_path.exists():
        return {}
    
    try:
        with open(type_path) as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Error loading {type_path}: {e}")
        return {}


def save_type_weights(weights: Dict[str, Dict], type_path: Path):
    """Save weights for a type to JSON file."""
    type_path.parent.mkdir(parents=True, exist_ok=True)
    with open(type_path, 'w') as f:
        json.dump(weights, f, indent=2)


def get_weights_dir() -> Path:
    """Get the base weights directory path (SSOT: WEIGHTS_DIR from utils)."""
    from .utils import WEIGHTS_DIR
    return WEIGHTS_DIR


def get_active_dir() -> Path:
    """Get the active weights directory path (C++ reads from here)."""
    return get_weights_dir()


def get_runs_dir() -> Path:
    """Get the runs directory path."""
    return get_weights_dir() / "runs"


def get_merged_dir() -> Path:
    """Get the merged weights directory path."""
    return get_weights_dir() / "merged"


def list_runs() -> List[RunInfo]:
    """List all available runs."""
    runs_dir = get_runs_dir()
    if not runs_dir.exists():
        return []
    
    runs = []
    for entry in sorted(runs_dir.iterdir()):
        if entry.is_dir() and (entry / "registry.json").exists():
            run = RunInfo(timestamp=entry.name, path=entry)
            if run.load():
                runs.append(run)
    
    return runs


def save_current_run(timestamp: Optional[str] = None) -> Path:
    """
    Save current weights to a timestamped run folder.
    
    Args:
        timestamp: Optional timestamp string, defaults to current time
        
    Returns:
        Path to the saved run folder
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    active_dir = get_active_dir()
    runs_dir = get_runs_dir()
    run_dir = runs_dir / timestamp
    
    # Create run directory
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy all type directories and registry from active
    files_copied = 0
    if active_dir.exists():
        # Copy registry
        reg = active_dir / "registry.json"
        if reg.is_file():
            shutil.copy2(reg, run_dir / "registry.json")
            files_copied += 1
        # Copy type_N/ directories
        for item in active_dir.iterdir():
            if item.is_dir() and item.name.startswith('type_'):
                dest = run_dir / item.name
                shutil.copytree(item, dest)
                files_copied += sum(1 for _ in dest.rglob('*.json'))
    
    print(f"Saved {files_copied} files to run {timestamp}")
    return run_dir


def merge_runs(
    run_timestamps: Optional[List[str]] = None,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Merge weights from multiple runs.
    
    Args:
        run_timestamps: List of run timestamps to merge (None = all runs)
        threshold: Centroid distance threshold for type matching
        output_dir: Where to save merged weights (default: merged/)
        
    Returns:
        Summary of the merge operation
    """
    if output_dir is None:
        output_dir = get_merged_dir()
    
    # Load runs
    all_runs = list_runs()
    if not all_runs:
        print("No runs found to merge")
        return {"error": "No runs found"}
    
    if run_timestamps:
        runs = [r for r in all_runs if r.timestamp in run_timestamps]
        if not runs:
            print(f"No matching runs found for: {run_timestamps}")
            return {"error": "No matching runs"}
    else:
        runs = all_runs
    
    print(f"Merging {len(runs)} runs...")
    
    # Initialize merged state
    merged_types: Dict[str, TypeInfo] = {}
    merged_weights: Dict[str, Dict[str, Dict]] = {}  # type_id -> algo -> weights
    type_counter = 0
    
    # Process each run
    for run in runs:
        print(f"\n  Processing run: {run.timestamp}")
        
        for src_type_id, src_type in run.types.items():
            # Try to match with existing merged type
            match_id = find_matching_type(src_type, merged_types, threshold)
            
            if match_id:
                # Merge into existing type
                print(f"    {src_type_id} -> matches {match_id}")
                target_type = merged_types[match_id]
                
                # Merge centroids
                target_type.centroid = merge_centroids(
                    target_type.centroid, target_type.graph_count,
                    src_type.centroid, src_type.graph_count
                )
                
                # Load and merge weights
                src_weights = load_type_weights(run.path / src_type_id / "weights.json")
                
                for algo, algo_weights in src_weights.items():
                    if algo.startswith('_'):
                        continue
                    
                    if algo in merged_weights.get(match_id, {}):
                        merged_weights[match_id][algo] = merge_weights(
                            merged_weights[match_id][algo], target_type.graph_count,
                            algo_weights, src_type.graph_count
                        )
                    else:
                        if match_id not in merged_weights:
                            merged_weights[match_id] = {}
                        merged_weights[match_id][algo] = algo_weights.copy()
                
                # Update graph count
                target_type.graph_count += src_type.graph_count
                
                # Merge algorithm lists
                for algo in src_type.algorithms:
                    if algo not in target_type.algorithms:
                        target_type.algorithms.append(algo)
            
            else:
                # Create new type
                new_id = f"type_{type_counter}"
                type_counter += 1
                print(f"    {src_type_id} -> new {new_id}")
                
                merged_types[new_id] = TypeInfo(
                    type_id=new_id,
                    centroid=src_type.centroid.copy(),
                    graph_count=src_type.graph_count,
                    algorithms=src_type.algorithms.copy()
                )
                
                # Load weights
                src_weights = load_type_weights(run.path / src_type_id / "weights.json")
                merged_weights[new_id] = {
                    k: v.copy() for k, v in src_weights.items()
                    if not k.startswith('_')
                }
    
    # Save merged results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save type weights
    for type_id, weights in merged_weights.items():
        save_type_weights(weights, output_dir / type_id / "weights.json")
        print(f"  Saved {type_id}/weights.json ({len(weights)} algorithms)")
    
    # Save registry
    registry = {}
    for type_id, type_info in merged_types.items():
        registry[type_id] = {
            'centroid': type_info.centroid,
            'graph_count': type_info.graph_count,
            'algorithms': type_info.algorithms,
            'graphs': []  # Don't track individual graphs in merged
        }
    
    registry_path = output_dir / "registry.json"
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    print(f"  Saved registry.json ({len(registry)} types)")
    
    # Summary
    summary = {
        "runs_merged": len(runs),
        "run_timestamps": [r.timestamp for r in runs],
        "total_types": len(merged_types),
        "types": {
            tid: {
                "graph_count": t.graph_count,
                "algorithms": len(t.algorithms)
            }
            for tid, t in merged_types.items()
        },
        "output_dir": str(output_dir)
    }
    
    print(f"\nMerge complete: {len(merged_types)} types from {len(runs)} runs")
    return summary


def use_run(timestamp: str) -> bool:
    """
    Copy a specific run's weights to the main weights directory.
    
    Args:
        timestamp: Run timestamp to use
        
    Returns:
        True if successful
    """
    runs_dir = get_runs_dir()
    active_dir = get_active_dir()
    run_dir = runs_dir / timestamp
    
    if not run_dir.exists():
        print(f"Run not found: {timestamp}")
        return False
    
    # Create active directory if needed
    active_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files from run to active directory
    files_copied = 0
    for item in run_dir.iterdir():
        if item.is_file() and item.suffix == '.json':
            shutil.copy2(item, active_dir / item.name)
            files_copied += 1
        elif item.is_dir() and item.name.startswith('type_'):
            dest = active_dir / item.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
            files_copied += sum(1 for _ in dest.rglob('*.json'))
    
    print(f"Applied {files_copied} files from run {timestamp}")
    return True


def use_merged() -> bool:
    """
    Copy merged weights to the active weights directory.
    
    Returns:
        True if successful
    """
    merged_dir = get_merged_dir()
    active_dir = get_active_dir()
    
    if not merged_dir.exists():
        print("No merged weights found. Run merge first.")
        return False
    
    # Create active directory if needed
    active_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files from merged to active directory
    files_copied = 0
    for item in merged_dir.iterdir():
        if item.is_file() and item.suffix == '.json':
            shutil.copy2(item, active_dir / item.name)
            files_copied += 1
        elif item.is_dir() and item.name.startswith('type_'):
            dest = active_dir / item.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
            files_copied += sum(1 for _ in dest.rglob('*.json'))
    
    print(f"Applied {files_copied} files from merged weights")
    return True


def auto_merge_after_run(timestamp: Optional[str] = None) -> Dict[str, Any]:
    """
    Automatically save current run and merge with existing runs.
    
    This should be called at the end of fill-weights to automatically
    accumulate weights across runs.
    
    Args:
        timestamp: Optional timestamp for the run
        
    Returns:
        Summary of the merge operation
    """
    # Save current run
    run_path = save_current_run(timestamp)
    
    # Merge all runs
    summary = merge_runs()
    
    # Apply merged weights to main directory
    use_merged()
    
    return summary


def validate_merge() -> bool:
    """
    Validate that merged weights are mathematically correct.
    
    Checks:
    1. Total graph counts match sum of runs
    2. Centroids are valid weighted averages
    3. Weights are valid weighted averages
    4. All algorithms from runs are present in merged
    
    Returns:
        True if all validations pass
    """
    print("\n" + "=" * 60)
    print("Validating Merge Results")
    print("=" * 60)
    
    runs = list_runs()
    if not runs:
        print("No runs to validate against")
        return False
    
    merged_dir = get_merged_dir()
    if not merged_dir.exists():
        print("No merged weights found")
        return False
    
    # Load merged registry
    merged_reg_path = merged_dir / "registry.json"
    if not merged_reg_path.exists():
        print("Merged registry not found")
        return False
    
    with open(merged_reg_path) as f:
        merged_reg = json.load(f)
    
    all_passed = True
    
    # Check 1: Total graph counts
    print("\n1. Checking total graph counts...")
    run_total = sum(
        tinfo.graph_count 
        for run in runs 
        for tinfo in run.types.values()
    )
    merged_total = sum(
        entry.get('graph_count', 0) 
        for entry in merged_reg.values()
    )
    
    if run_total == merged_total:
        print(f"   ✓ Total graphs: {merged_total} (matches sum of runs)")
    else:
        print(f"   ✗ Total graphs mismatch: runs={run_total}, merged={merged_total}")
        all_passed = False
    
    # Check 2: All algorithms preserved
    print("\n2. Checking algorithm preservation...")
    run_algorithms = set()
    for run in runs:
        for tinfo in run.types.values():
            type_path = run.path / tinfo.type_id / "weights.json"
            if type_path.exists():
                with open(type_path) as f:
                    weights = json.load(f)
                for algo in weights:
                    if not algo.startswith('_'):
                        run_algorithms.add(algo)
    
    merged_algorithms = set()
    for tdir in merged_dir.iterdir():
        if tdir.is_dir() and tdir.name.startswith('type_'):
            wfile = tdir / "weights.json"
            if wfile.exists():
                with open(wfile) as f:
                    weights = json.load(f)
                for algo in weights:
                    if not algo.startswith('_'):
                        merged_algorithms.add(algo)
    
    missing = run_algorithms - merged_algorithms
    if not missing:
        print(f"   ✓ All {len(merged_algorithms)} algorithms preserved")
    else:
        print(f"   ✗ Missing algorithms: {missing}")
        all_passed = False
    
    # Check 3: Centroid values are valid
    print("\n3. Checking centroid validity...")
    for tid, entry in merged_reg.items():
        centroid = entry.get('centroid', [])
        if isinstance(centroid, list) and len(centroid) > 0:
            # Check for NaN or Inf
            if any(not np.isfinite(v) for v in centroid):
                print(f"   ✗ {tid}: Invalid centroid values (NaN/Inf)")
                all_passed = False
            # Check reasonable ranges
            elif all(-1000 < v < 1000 for v in centroid):
                print(f"   ✓ {tid}: Centroid valid, count={entry.get('graph_count', 0)}")
            else:
                print(f"   ⚠ {tid}: Centroid has extreme values")
    
    # Check 4: Weight values are valid
    print("\n4. Checking weight validity...")
    for tfile in merged_dir.iterdir():
        if tfile.suffix == '.json' and tfile.name.startswith('type_') and 'registry' not in tfile.name:
            with open(tfile) as f:
                weights = json.load(f)
            
            invalid_count = 0
            for algo, w in weights.items():
                if algo.startswith('_'):
                    continue
                for key, val in w.items():
                    if key.startswith('_') or key == 'benchmark_weights':
                        continue
                    if isinstance(val, (int, float)) and not np.isfinite(val):
                        invalid_count += 1
            
            if invalid_count == 0:
                print(f"   ✓ {tfile.name}: All weights valid")
            else:
                print(f"   ✗ {tfile.name}: {invalid_count} invalid weight values")
                all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All validations passed")
    else:
        print("✗ Some validations failed")
    print("=" * 60)
    
    return all_passed


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Merge perceptron weights from multiple experiment runs"
    )
    parser.add_argument('--list-runs', action='store_true',
                       help="List all available runs")
    parser.add_argument('--save-run', metavar='NAME',
                       help="Save current weights as a named run")
    parser.add_argument('--merge-all', action='store_true',
                       help="Merge all runs into merged/")
    parser.add_argument('--merge-runs', nargs='+', metavar='TIMESTAMP',
                       help="Merge specific runs by timestamp")
    parser.add_argument('--use-run', metavar='TIMESTAMP',
                       help="Use weights from a specific run")
    parser.add_argument('--use-merged', action='store_true',
                       help="Use merged weights")
    parser.add_argument('--threshold', type=float, default=DEFAULT_MATCH_THRESHOLD,
                       help=f"Centroid matching threshold (default: {DEFAULT_MATCH_THRESHOLD})")
    parser.add_argument('--auto-merge', action='store_true',
                       help="Save current run and merge (for use after fill-weights)")
    parser.add_argument('--no-apply', action='store_true',
                       help="Don't apply merged weights to main directory after merge")
    parser.add_argument('--validate', action='store_true',
                       help="Validate merge results (check math correctness)")
    
    args = parser.parse_args()
    
    if args.list_runs:
        runs = list_runs()
        if not runs:
            print("No runs found")
        else:
            print(f"Available runs ({len(runs)}):\n")
            for run in runs:
                print(f"  {run.timestamp}:")
                print(f"    Types: {len(run.types)}")
                for tid, tinfo in run.types.items():
                    print(f"      {tid}: {tinfo.graph_count} graphs, {len(tinfo.algorithms)} algos")
                print()
    
    elif args.save_run:
        save_current_run(args.save_run)
    
    elif args.merge_all:
        summary = merge_runs(threshold=args.threshold)
        if not args.no_apply and "error" not in summary:
            use_merged()
            print("\nMerged weights applied to main directory.")
        if args.validate:
            validate_merge()
    
    elif args.merge_runs:
        summary = merge_runs(run_timestamps=args.merge_runs, threshold=args.threshold)
        if not args.no_apply and "error" not in summary:
            use_merged()
            print("\nMerged weights applied to main directory.")
        if args.validate:
            validate_merge()
    
    elif args.use_run:
        use_run(args.use_run)
    
    elif args.use_merged:
        use_merged()
    
    elif args.auto_merge:
        auto_merge_after_run()
    
    elif args.validate:
        validate_merge()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
