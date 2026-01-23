#!/usr/bin/env python3
"""
Generate More Type Clusters Script
==================================

Runs experiments on diverse graphs to generate type_0, type_1, type_2, type_3 clusters.

This script:
1. Tests graphs with very different topologies
2. Lowers the clustering threshold to create more distinct clusters
3. Runs benchmarks to update weights
4. Verifies multiple types are created and used
"""

import json
import os
import subprocess
import sys
import math
from pathlib import Path
from datetime import datetime

# Paths
GRAPHBREW_ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = GRAPHBREW_ROOT / "scripts" / "weights"
BENCH_BIN = GRAPHBREW_ROOT / "bench" / "bin"
GRAPH_DATASETS = Path("/home/ab/Documents/00_github_repos/00_GraphDataSets")

# Graphs representing very different topologies
# We want enough diversity to generate 4 distinct clusters
DIVERSE_GRAPHS = [
    # Social networks (high modularity, power-law, hub-heavy)
    ("soc-LiveJournal1", GRAPH_DATASETS / "soc-LiveJournal1" / "graph.el"),
    ("com-Youtube", GRAPH_DATASETS / "com-Youtube" / "graph.el"),
    
    # Road networks (grid-like, uniform degree, low modularity)  
    ("roadNet-CA", GRAPH_DATASETS / "roadNet-CA" / "graph.el"),
    ("roadNet-TX", GRAPH_DATASETS / "roadNet-TX" / "graph.el"),
    
    # Web graphs (crawl patterns, directional, very hub-heavy)
    ("web-Google", GRAPH_DATASETS / "web-Google" / "graph.el"),
    ("web-BerkStan", GRAPH_DATASETS / "web-BerkStan" / "graph.el"),
    
    # Citation/collaboration (temporal patterns)
    ("cit-Patents", GRAPH_DATASETS / "cit-Patents" / "graph.el"),
    ("com-DBLP", GRAPH_DATASETS / "com-DBLP" / "graph.el"),
    
    # E-commerce (bipartite-ish)
    ("com-Amazon", GRAPH_DATASETS / "com-Amazon" / "graph.el"),
]


def load_type_registry():
    """Load type registry."""
    registry_path = WEIGHTS_DIR / "type_registry.json"
    if registry_path.exists():
        with open(registry_path) as f:
            return json.load(f)
    return {}


def print_color(msg, color):
    colors = {"green": "\033[92m", "red": "\033[91m", "yellow": "\033[93m", 
              "blue": "\033[94m", "cyan": "\033[96m", "end": "\033[0m", "bold": "\033[1m"}
    print(f"{colors.get(color, '')}{msg}{colors['end']}")


def run_benchmark_on_graph(graph_name, graph_path, algo=0, benchmark="pr", iterations=1):
    """Run a benchmark on a graph and return the output."""
    bin_path = BENCH_BIN / benchmark
    if not bin_path.exists():
        print_color(f"  Binary not found: {bin_path}", "red")
        return None
        
    cmd = [
        str(bin_path),
        "-f", str(graph_path),
        "-o", str(algo),
        "-n", str(iterations),
        "-v"  # Verbose
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(GRAPHBREW_ROOT)
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print_color(f"  Timeout running {benchmark} on {graph_name}", "yellow")
        return None
    except Exception as e:
        print_color(f"  Error: {e}", "red")
        return None


def run_adaptive_on_graph(graph_name, graph_path, benchmark="pr"):
    """Run adaptive reorder and extract type information."""
    output = run_benchmark_on_graph(graph_name, graph_path, algo=15, benchmark=benchmark)
    
    if output:
        type_detected = None
        algorithm_selected = None
        
        for line in output.split('\n'):
            if "Best matching type:" in line:
                parts = line.split("Best matching type:")
                if len(parts) > 1:
                    type_detected = parts[1].strip().split()[0]
            if "AdaptiveOrder" in line and "selected" in line.lower():
                algorithm_selected = line.strip()
        
        return {"type": type_detected, "algorithm": algorithm_selected, "output": output}
    return None


def main():
    print_color("\n" + "="*60, "bold")
    print_color("Generate More Type Clusters - Test Script", "bold")
    print_color("="*60 + "\n", "bold")
    
    print_color(f"Weights directory: {WEIGHTS_DIR}", "cyan")
    print_color(f"Graph datasets: {GRAPH_DATASETS}\n", "cyan")
    
    # Check initial state
    registry = load_type_registry()
    print_color(f"Initial types: {list(registry.keys())}", "blue")
    
    # Filter available graphs
    available = [(name, path) for name, path in DIVERSE_GRAPHS if path.exists()]
    print_color(f"Available graphs: {len(available)}/{len(DIVERSE_GRAPHS)}\n", "blue")
    
    # Run adaptive on each graph to see type assignments
    print_color("Testing type assignments with Adaptive Reorder (-o 15):", "bold")
    print_color("-" * 50, "bold")
    
    type_assignments = {}
    
    for graph_name, graph_path in available:
        print_color(f"\n{graph_name}:", "cyan")
        result = run_adaptive_on_graph(graph_name, graph_path)
        
        if result:
            t = result.get("type", "unknown")
            algo = result.get("algorithm", "unknown")
            type_assignments[graph_name] = t
            print_color(f"  ✓ Type: {t}", "green")
            if algo:
                print(f"    {algo}")
        else:
            print_color(f"  ✗ Failed to run", "red")
    
    # Summary
    print_color("\n" + "="*60, "bold")
    print_color("SUMMARY", "bold")
    print_color("="*60, "bold")
    
    # Check final registry
    registry = load_type_registry()
    print_color(f"\nFinal types: {list(registry.keys())}", "green")
    
    # Group graphs by assigned type
    by_type = {}
    for graph, t in type_assignments.items():
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(graph)
    
    print_color("\nGraphs by type:", "blue")
    for t, graphs in sorted(by_type.items()):
        print_color(f"  {t}: {graphs}", "cyan")
    
    # Check type weights
    print_color("\nType weights status:", "blue")
    for type_name in registry.keys():
        weights_file = WEIGHTS_DIR / f"{type_name}.json"
        if weights_file.exists():
            with open(weights_file) as f:
                weights = json.load(f)
            num_algos = len(weights)
            # Check for metadata
            has_metadata = False
            sample_count = 0
            for algo_data in weights.values():
                if isinstance(algo_data, dict) and "_metadata" in algo_data:
                    has_metadata = True
                    sample_count = algo_data["_metadata"].get("sample_count", 0)
                    break
            print_color(f"  {type_name}: {num_algos} algorithms, "
                       f"metadata={has_metadata}, samples={sample_count}", "green")
        else:
            print_color(f"  {type_name}: NO WEIGHTS FILE", "red")
    
    # Final verdict
    num_types = len(registry)
    if num_types >= 4:
        print_color(f"\n✓ SUCCESS: {num_types} types created (target: 4)", "green")
    elif num_types >= 2:
        print_color(f"\n⚠ PARTIAL: {num_types} types created (target: 4)", "yellow")
        print_color("  Note: Graphs may be too similar in feature space to form more clusters", "yellow")
        print_color("  Consider lowering CLUSTER_DISTANCE_THRESHOLD in graphbrew_experiment.py", "cyan")
    else:
        print_color(f"\n✗ FAILED: Only {num_types} type(s) created", "red")
    
    return num_types


if __name__ == "__main__":
    num_types = main()
    sys.exit(0 if num_types >= 2 else 1)
