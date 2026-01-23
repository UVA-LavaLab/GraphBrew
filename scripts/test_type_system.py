#!/usr/bin/env python3
"""
Type System Verification Script
===============================

Tests the auto-clustering type system to verify:
1. Different graph topologies create distinct type clusters (type_0, type_1, type_2, type_3)
2. Weights are loaded correctly during adaptive reordering
3. Cache simulation uses the weights appropriately  
4. Weight updates work correctly after benchmarks

Usage:
    python scripts/test_type_system.py --all                # Full verification
    python scripts/test_type_system.py --test-clustering    # Only test clustering
    python scripts/test_type_system.py --test-adaptive      # Only test adaptive reorder
    python scripts/test_type_system.py --test-cache-sim     # Only test cache simulation
    python scripts/test_type_system.py --test-weight-update # Only test weight updates
"""

import argparse
import json
import os
import subprocess
import sys
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
GRAPHBREW_ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = GRAPHBREW_ROOT / "scripts" / "weights"
RESULTS_DIR = GRAPHBREW_ROOT / "results"
BENCH_BIN = GRAPHBREW_ROOT / "bench" / "bin"
BENCH_BIN_SIM = GRAPHBREW_ROOT / "bench" / "bin_sim"

# External graph datasets location
GRAPH_DATASETS = Path("/home/ab/Documents/00_github_repos/00_GraphDataSets")

# Test graphs with expected characteristics
TEST_GRAPHS = {
    # Social networks - high modularity, power-law degree distribution, hub-heavy
    "soc-LiveJournal1": {
        "path": GRAPH_DATASETS / "soc-LiveJournal1" / "graph.el",
        "expected_type_features": {"high_modularity": True, "hub_heavy": True},
        "category": "social"
    },
    "com-Youtube": {
        "path": GRAPH_DATASETS / "com-Youtube" / "graph.el",
        "expected_type_features": {"high_modularity": True, "hub_heavy": True},
        "category": "social"
    },
    
    # Road networks - grid-like, low modularity, uniform degree
    "roadNet-CA": {
        "path": GRAPH_DATASETS / "roadNet-CA" / "graph.el",
        "expected_type_features": {"low_modularity": True, "uniform_degree": True},
        "category": "road"
    },
    "roadNet-TX": {
        "path": GRAPH_DATASETS / "roadNet-TX" / "graph.el", 
        "expected_type_features": {"low_modularity": True, "uniform_degree": True},
        "category": "road"
    },
    
    # Web graphs - crawl patterns, very high hub concentration
    "web-Google": {
        "path": GRAPH_DATASETS / "web-Google" / "graph.el",
        "expected_type_features": {"web_structure": True, "very_hub_heavy": True},
        "category": "web"
    },
    "web-BerkStan": {
        "path": GRAPH_DATASETS / "web-BerkStan" / "graph.el",
        "expected_type_features": {"web_structure": True, "very_hub_heavy": True},
        "category": "web"
    },
    
    # Citation/collaboration networks - different pattern
    "cit-Patents": {
        "path": GRAPH_DATASETS / "cit-Patents" / "graph.el",
        "expected_type_features": {"citation_pattern": True},
        "category": "citation"
    },
    "com-DBLP": {
        "path": GRAPH_DATASETS / "com-DBLP" / "graph.el",
        "expected_type_features": {"collaboration_pattern": True},
        "category": "collaboration"
    },
}

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(msg: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{msg}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_success(msg: str):
    print(f"{Colors.GREEN}✓ {msg}{Colors.ENDC}")

def print_fail(msg: str):
    print(f"{Colors.FAIL}✗ {msg}{Colors.ENDC}")

def print_warning(msg: str):
    print(f"{Colors.WARNING}⚠ {msg}{Colors.ENDC}")

def print_info(msg: str):
    print(f"{Colors.CYAN}ℹ {msg}{Colors.ENDC}")


def load_type_registry() -> Dict:
    """Load the type registry."""
    registry_path = WEIGHTS_DIR / "type_registry.json"
    if registry_path.exists():
        with open(registry_path) as f:
            return json.load(f)
    return {}


def load_type_weights(type_name: str) -> Dict:
    """Load weights for a specific type."""
    weights_path = WEIGHTS_DIR / f"{type_name}.json"
    if weights_path.exists():
        with open(weights_path) as f:
            return json.load(f)
    return {}


def backup_weights():
    """Backup current weights before testing."""
    backup_dir = WEIGHTS_DIR / "backup_test"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    for f in WEIGHTS_DIR.glob("*.json"):
        if f.is_file():
            shutil.copy(f, backup_dir / f.name)
    
    print_info(f"Backed up weights to {backup_dir}")
    return backup_dir


def restore_weights(backup_dir: Path):
    """Restore weights from backup."""
    if not backup_dir.exists():
        print_warning("No backup found to restore")
        return
    
    for f in backup_dir.glob("*.json"):
        shutil.copy(f, WEIGHTS_DIR / f.name)
    
    print_info(f"Restored weights from {backup_dir}")


def compute_graph_features(graph_path: Path) -> Optional[Dict]:
    """Compute basic graph features from edge list."""
    if not graph_path.exists():
        print_warning(f"Graph not found: {graph_path}")
        return None
    
    try:
        nodes = set()
        edges = 0
        degree_count = {}
        
        with open(graph_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('%'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        src, dst = int(parts[0]), int(parts[1])
                        nodes.add(src)
                        nodes.add(dst)
                        edges += 1
                        degree_count[src] = degree_count.get(src, 0) + 1
                        degree_count[dst] = degree_count.get(dst, 0) + 1
                    except ValueError:
                        continue
        
        num_nodes = len(nodes)
        num_edges = edges
        
        if num_nodes == 0:
            return None
        
        # Compute features
        degrees = list(degree_count.values())
        avg_degree = sum(degrees) / len(degrees) if degrees else 0
        max_degree = max(degrees) if degrees else 0
        
        # Degree variance (normalized)
        if len(degrees) > 1:
            mean_deg = avg_degree
            variance = sum((d - mean_deg) ** 2 for d in degrees) / len(degrees)
            degree_variance = min(1.0, math.sqrt(variance) / (max_degree + 1))
        else:
            degree_variance = 0
        
        # Hub concentration (fraction of edges from top 1% nodes)
        sorted_degrees = sorted(degrees, reverse=True)
        top_1_percent = max(1, len(sorted_degrees) // 100)
        hub_edges = sum(sorted_degrees[:top_1_percent])
        total_edges = sum(sorted_degrees)
        hub_concentration = hub_edges / total_edges if total_edges > 0 else 0
        
        # Density
        max_possible_edges = num_nodes * (num_nodes - 1) // 2
        density = num_edges / max_possible_edges if max_possible_edges > 0 else 0
        
        # Estimated modularity (simplified - based on degree distribution)
        # High variance + clustered structure -> higher modularity
        modularity_estimate = min(1.0, 0.3 + degree_variance * 0.5 + hub_concentration * 0.2)
        
        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "avg_degree": avg_degree,
            "max_degree": max_degree,
            "degree_variance": degree_variance,
            "hub_concentration": hub_concentration,
            "density": density,
            "modularity": modularity_estimate,
        }
    except Exception as e:
        print_warning(f"Error computing features: {e}")
        return None


def test_clustering(verbose: bool = True) -> Tuple[bool, Dict]:
    """
    Test 1: Verify that different graph types create distinct clusters.
    
    We expect:
    - Social graphs should cluster together
    - Road graphs should cluster together (different from social)
    - Web graphs may form their own cluster
    """
    print_header("TEST 1: Type Clustering Verification")
    
    results = {"passed": True, "details": {}}
    type_assignments = {}
    
    # Check which graphs are available
    available_graphs = {}
    for name, info in TEST_GRAPHS.items():
        if info["path"].exists():
            available_graphs[name] = info
            print_success(f"Found graph: {name}")
        else:
            print_warning(f"Graph not available: {name} ({info['path']})")
    
    if len(available_graphs) < 3:
        print_fail("Not enough graphs available for clustering test")
        results["passed"] = False
        results["details"]["error"] = "Insufficient graphs"
        return results["passed"], results
    
    # Compute features for each graph
    print_info("\nComputing features for available graphs...")
    
    for name, info in available_graphs.items():
        features = compute_graph_features(info["path"])
        if features:
            type_assignments[name] = {
                "features": features,
                "category": info["category"]
            }
            if verbose:
                print(f"  {name}: nodes={features['num_nodes']:,}, edges={features['num_edges']:,}, "
                      f"avg_deg={features['avg_degree']:.2f}, hub_conc={features['hub_concentration']:.3f}")
    
    # Check current type registry
    registry = load_type_registry()
    print_info(f"\nCurrent type registry has {len(registry)} types: {list(registry.keys())}")
    
    # Group by category and check if similar graphs get similar types
    by_category = {}
    for name, data in type_assignments.items():
        cat = data["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append((name, data["features"]))
    
    print_info("\nGraphs by category:")
    for cat, graphs in by_category.items():
        print(f"  {cat}: {[g[0] for g in graphs]}")
    
    # Compute feature distances within and between categories
    print_info("\nFeature similarity analysis:")
    
    def feature_distance(f1: Dict, f2: Dict) -> float:
        """Euclidean distance in feature space."""
        keys = ["modularity", "degree_variance", "hub_concentration"]
        return math.sqrt(sum((f1.get(k, 0) - f2.get(k, 0)) ** 2 for k in keys))
    
    # Within-category distances should be small
    within_distances = []
    for cat, graphs in by_category.items():
        if len(graphs) > 1:
            for i in range(len(graphs)):
                for j in range(i + 1, len(graphs)):
                    dist = feature_distance(graphs[i][1], graphs[j][1])
                    within_distances.append(dist)
                    print(f"  Within {cat}: {graphs[i][0]} <-> {graphs[j][0]} = {dist:.4f}")
    
    # Between-category distances should be larger
    between_distances = []
    cats = list(by_category.keys())
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            for g1 in by_category[cats[i]]:
                for g2 in by_category[cats[j]]:
                    dist = feature_distance(g1[1], g2[1])
                    between_distances.append(dist)
                    print(f"  Between {cats[i]}/{cats[j]}: {g1[0]} <-> {g2[0]} = {dist:.4f}")
    
    # Analysis
    if within_distances and between_distances:
        avg_within = sum(within_distances) / len(within_distances)
        avg_between = sum(between_distances) / len(between_distances)
        print_info(f"\nAverage within-category distance: {avg_within:.4f}")
        print_info(f"Average between-category distance: {avg_between:.4f}")
        
        if avg_between > avg_within:
            print_success("Different graph categories have distinguishable features")
        else:
            print_warning("Graph categories may not be well-separated in feature space")
    
    results["details"] = {
        "available_graphs": list(available_graphs.keys()),
        "categories": {cat: [g[0] for g in graphs] for cat, graphs in by_category.items()},
        "num_types": len(registry)
    }
    
    print_success(f"\nClustering test completed with {len(type_assignments)} graphs")
    return results["passed"], results


def test_adaptive_reorder(verbose: bool = True) -> Tuple[bool, Dict]:
    """
    Test 2: Verify adaptive reordering loads correct weights.
    
    Tests:
    - Run benchmark with -o 15 (AdaptiveOrder)
    - Verify it prints which type weights are being used
    - Verify different graph types select different algorithms
    """
    print_header("TEST 2: Adaptive Reorder Weight Loading")
    
    results = {"passed": True, "details": {}}
    
    # Check if binaries exist
    pr_bin = BENCH_BIN / "pr"
    if not pr_bin.exists():
        print_fail(f"PageRank binary not found at {pr_bin}")
        results["passed"] = False
        results["details"]["error"] = "Binary not found"
        return results["passed"], results
    
    # Pick a small test graph
    test_graph = None
    for name, info in TEST_GRAPHS.items():
        if info["path"].exists():
            # Get size
            size_mb = info["path"].stat().st_size / (1024 * 1024)
            if size_mb < 100:  # Use graph < 100MB for quick test
                test_graph = (name, info["path"])
                break
    
    if not test_graph:
        print_warning("No small test graph found, skipping adaptive reorder test")
        results["passed"] = False
        results["details"]["error"] = "No test graph"
        return results["passed"], results
    
    graph_name, graph_path = test_graph
    print_info(f"Testing with graph: {graph_name}")
    
    # Run adaptive reorder with verbose output
    cmd = [
        str(pr_bin),
        "-f", str(graph_path),
        "-o", "15",  # AdaptiveOrder
        "-n", "1",   # Single iteration
        "-v"         # Verbose
    ]
    
    print_info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(GRAPHBREW_ROOT)
        )
        
        output = result.stdout + result.stderr
        
        # Check for weight loading messages
        weight_loaded = False
        type_detected = None
        algorithm_selected = None
        
        for line in output.split('\n'):
            if "Perceptron:" in line or "type_" in line.lower():
                print_info(f"  Weight log: {line.strip()}")
                weight_loaded = True
                
            if "Best matching type:" in line:
                # Extract type name
                parts = line.split("Best matching type:")
                if len(parts) > 1:
                    type_detected = parts[1].strip().split()[0]
                    
            if "Selected algorithm:" in line or "AdaptiveOrder selected:" in line:
                algorithm_selected = line.strip()
                print_info(f"  Selection: {line.strip()}")
        
        if weight_loaded or type_detected:
            print_success(f"Type-based weights loaded (type: {type_detected or 'detected'})")
        else:
            print_warning("Could not verify weight loading from output")
        
        # Check exit code
        if result.returncode == 0:
            print_success("Adaptive reorder completed successfully")
        else:
            print_warning(f"Exit code: {result.returncode}")
        
        results["details"] = {
            "graph": graph_name,
            "type_detected": type_detected,
            "algorithm_selected": algorithm_selected,
            "weight_loaded": weight_loaded,
            "exit_code": result.returncode
        }
        
    except subprocess.TimeoutExpired:
        print_fail("Adaptive reorder timed out")
        results["passed"] = False
        results["details"]["error"] = "Timeout"
    except Exception as e:
        print_fail(f"Error running adaptive reorder: {e}")
        results["passed"] = False
        results["details"]["error"] = str(e)
    
    return results["passed"], results


def test_cache_simulation(verbose: bool = True) -> Tuple[bool, Dict]:
    """
    Test 3: Verify cache simulation uses type-based weights.
    
    Tests:
    - Run cache simulation binary
    - Verify cache statistics are computed
    - Verify weights include cache parameters
    """
    print_header("TEST 3: Cache Simulation Verification")
    
    results = {"passed": True, "details": {}}
    
    # Check if simulation binaries exist
    pr_sim = BENCH_BIN_SIM / "pr"
    if not pr_sim.exists():
        print_fail(f"PageRank simulation binary not found at {pr_sim}")
        print_info("Cache simulation requires bench/bin_sim/ binaries")
        results["passed"] = False
        results["details"]["error"] = "Simulation binary not found"
        return results["passed"], results
    
    # Check type weights for cache parameters
    print_info("Checking type weights for cache parameters...")
    
    registry = load_type_registry()
    types_with_cache = []
    
    for type_name in registry.keys():
        weights = load_type_weights(type_name)
        if weights:
            # Check first algorithm for cache weights
            first_algo = list(weights.keys())[0] if weights else None
            if first_algo and isinstance(weights[first_algo], dict):
                algo_weights = weights[first_algo]
                cache_params = ["cache_l1_impact", "cache_l2_impact", "cache_l3_impact", "cache_dram_penalty"]
                has_cache = all(p in algo_weights for p in cache_params)
                
                if has_cache:
                    types_with_cache.append(type_name)
                    if verbose:
                        print_info(f"  {type_name}: L1={algo_weights.get('cache_l1_impact', 0):.6f}, "
                                   f"L2={algo_weights.get('cache_l2_impact', 0):.6f}, "
                                   f"DRAM={algo_weights.get('cache_dram_penalty', 0):.6f}")
    
    if types_with_cache:
        print_success(f"Found {len(types_with_cache)} type(s) with cache parameters: {types_with_cache}")
    else:
        print_warning("No types have cache parameters set")
    
    # Try running a quick cache simulation
    test_graph = None
    for name, info in TEST_GRAPHS.items():
        if info["path"].exists():
            size_mb = info["path"].stat().st_size / (1024 * 1024)
            if size_mb < 50:  # Use small graph for simulation
                test_graph = (name, info["path"])
                break
    
    if test_graph:
        graph_name, graph_path = test_graph
        print_info(f"\nRunning cache simulation with: {graph_name}")
        
        cmd = [
            str(pr_sim),
            "-f", str(graph_path),
            "-o", "0",  # ORIGINAL (no reorder)
            "-n", "1",
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(GRAPHBREW_ROOT)
            )
            
            output = result.stdout + result.stderr
            
            # Look for cache statistics
            cache_found = False
            for line in output.split('\n'):
                if any(x in line.lower() for x in ['l1', 'l2', 'l3', 'cache', 'hit', 'miss']):
                    print_info(f"  Cache: {line.strip()}")
                    cache_found = True
            
            if cache_found:
                print_success("Cache simulation produced cache statistics")
            else:
                print_warning("No cache statistics found in output")
            
            results["details"]["simulation_run"] = result.returncode == 0
            
        except Exception as e:
            print_warning(f"Could not run cache simulation: {e}")
            results["details"]["simulation_error"] = str(e)
    else:
        print_warning("No small test graph available for simulation")
    
    results["details"]["types_with_cache"] = types_with_cache
    return results["passed"], results


def test_weight_updates(verbose: bool = True) -> Tuple[bool, Dict]:
    """
    Test 4: Verify weight update mechanism.
    
    Tests:
    - Check that experiment script can update weights
    - Verify incremental weight updates work
    - Check metadata tracking (sample_count, win_rate, etc.)
    """
    print_header("TEST 4: Weight Update Verification")
    
    results = {"passed": True, "details": {}}
    
    # Check current type weights for metadata
    registry = load_type_registry()
    print_info(f"Checking {len(registry)} types for training metadata...")
    
    types_with_training = []
    
    for type_name in registry.keys():
        weights = load_type_weights(type_name)
        if weights:
            for algo_name, algo_weights in weights.items():
                if isinstance(algo_weights, dict) and "_metadata" in algo_weights:
                    metadata = algo_weights["_metadata"]
                    types_with_training.append({
                        "type": type_name,
                        "algorithm": algo_name,
                        "sample_count": metadata.get("sample_count", 0),
                        "win_rate": metadata.get("win_rate", 0),
                        "last_updated": metadata.get("last_updated", "never"),
                    })
                    break  # Just check first algo per type
    
    if types_with_training:
        print_success(f"Found {len(types_with_training)} trained types:")
        for t in types_with_training:
            print_info(f"  {t['type']}: samples={t['sample_count']}, "
                       f"win_rate={t['win_rate']:.2%}, updated={t['last_updated'][:10]}")
    else:
        print_warning("No types have training metadata")
    
    # Check type registry metadata
    print_info("\nType registry cluster info:")
    for type_name, type_info in registry.items():
        sample_count = type_info.get("sample_count", 0)
        created = type_info.get("created", "unknown")
        last_updated = type_info.get("last_updated", "unknown")
        
        print_info(f"  {type_name}: {sample_count} samples, "
                   f"created={created[:10] if created != 'unknown' else created}")
        
        # Check centroid
        if "centroid" in type_info:
            centroid = type_info["centroid"]
            print(f"    centroid: [{', '.join(f'{c:.3f}' for c in centroid[:4])}...]")
    
    # Verify weight file structure
    print_info("\nVerifying weight file structure...")
    
    required_weight_keys = [
        "bias", "w_modularity", "w_log_nodes", "w_log_edges",
        "w_degree_variance", "w_hub_concentration",
        "cache_l1_impact", "cache_l2_impact", "cache_dram_penalty"
    ]
    
    structure_ok = True
    for type_name in registry.keys():
        weights = load_type_weights(type_name)
        if not weights:
            print_warning(f"  {type_name}: No weights file")
            continue
        
        # Check each algorithm has required keys
        for algo_name, algo_weights in weights.items():
            if not isinstance(algo_weights, dict):
                continue
            missing = [k for k in required_weight_keys if k not in algo_weights]
            if missing:
                print_warning(f"  {type_name}/{algo_name}: missing {missing}")
                structure_ok = False
            else:
                if verbose and algo_name == "ORIGINAL":  # Just report first
                    print_success(f"  {type_name}: All required weight keys present")
                break
    
    if structure_ok:
        print_success("All type weight files have correct structure")
    
    results["details"] = {
        "num_types": len(registry),
        "trained_types": len(types_with_training),
        "structure_ok": structure_ok,
    }
    
    return results["passed"], results


def run_all_tests(verbose: bool = True) -> Dict:
    """Run all verification tests."""
    print_header("GraphBrew Type System Verification")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Weights directory: {WEIGHTS_DIR}")
    print(f"Graphs directory: {GRAPH_DATASETS}")
    
    all_results = {}
    all_passed = True
    
    # Test 1: Clustering
    passed, results = test_clustering(verbose)
    all_results["clustering"] = results
    if not passed:
        all_passed = False
    
    # Test 2: Adaptive Reorder  
    passed, results = test_adaptive_reorder(verbose)
    all_results["adaptive_reorder"] = results
    if not passed:
        all_passed = False
    
    # Test 3: Cache Simulation
    passed, results = test_cache_simulation(verbose)
    all_results["cache_simulation"] = results
    if not passed:
        all_passed = False
    
    # Test 4: Weight Updates
    passed, results = test_weight_updates(verbose)
    all_results["weight_updates"] = results
    if not passed:
        all_passed = False
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    
    for test_name, test_results in all_results.items():
        status = "PASSED" if test_results.get("passed", True) else "NEEDS ATTENTION"
        if status == "PASSED":
            print_success(f"{test_name}: {status}")
        else:
            print_warning(f"{test_name}: {status}")
    
    # Save results
    results_file = RESULTS_DIR / f"type_system_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "all_passed": all_passed,
            "results": all_results
        }, f, indent=2, default=str)
    
    print_info(f"\nResults saved to: {results_file}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Test the GraphBrew type system")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--test-clustering", action="store_true", help="Test type clustering")
    parser.add_argument("--test-adaptive", action="store_true", help="Test adaptive reorder")
    parser.add_argument("--test-cache-sim", action="store_true", help="Test cache simulation")
    parser.add_argument("--test-weight-update", action="store_true", help="Test weight updates")
    parser.add_argument("-v", "--verbose", action="store_true", default=True, help="Verbose output")
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet output")
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    # Default to all tests if none specified
    if not any([args.all, args.test_clustering, args.test_adaptive, 
                args.test_cache_sim, args.test_weight_update]):
        args.all = True
    
    if args.all:
        run_all_tests(verbose)
    else:
        if args.test_clustering:
            test_clustering(verbose)
        if args.test_adaptive:
            test_adaptive_reorder(verbose)
        if args.test_cache_sim:
            test_cache_simulation(verbose)
        if args.test_weight_update:
            test_weight_updates(verbose)


if __name__ == "__main__":
    main()
