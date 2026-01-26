#!/usr/bin/env python3
"""
Compare Leiden Variants - Community Quality and Performance Benchmark

This script compares all Leiden-based reordering algorithms:
- LeidenOrder (15): Native optimized Leiden library (reference for quality)
- LeidenDendrogram (16): Leiden + dendrogram traversal variants
- LeidenCSR (17): Fast native CSR implementation  
- GraphBrewOrder (12): Leiden + per-community RabbitOrder

Metrics compared:
- Number of communities detected
- Number of Leiden passes
- Reorder time (fair timing excluding conversion)
- Modularity (if available)
"""

import os
import sys
import json
import subprocess
import re
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from lib.reorder import parse_reorder_time_from_converter

@dataclass
class LeidenResult:
    """Result from a single Leiden variant run"""
    algorithm: str
    algo_id: int
    variant: str
    reorder_time: float
    num_communities: int
    num_passes: int
    resolution: float
    modularity: float = 0.0
    raw_output: str = ""

def get_project_root() -> Path:
    """Get the project root directory"""
    return SCRIPT_DIR.parent

def find_converter() -> Path:
    """Find the converter binary"""
    root = get_project_root()
    converter = root / "bench" / "bin" / "converter"
    if not converter.exists():
        raise FileNotFoundError(f"Converter not found at {converter}. Run 'make all' first.")
    return converter

def find_graphs(min_edges: int = 10000, max_edges: int = 10000000) -> List[Path]:
    """Find available graphs within edge count range"""
    root = get_project_root()
    graphs_json = root / "graphs" / "graphs.json"
    
    graphs = []
    
    # Check if graphs.json exists
    if graphs_json.exists():
        with open(graphs_json) as f:
            graph_data = json.load(f)
        
        for name, info in graph_data.items():
            edges = info.get("edges", 0)
            if min_edges <= edges <= max_edges:
                # Find the graph file
                graph_dir = root / "graphs" / name
                if graph_dir.exists():
                    for ext in [".el", ".wel", ".sg", ".wsg"]:
                        graph_file = graph_dir / f"{name}{ext}"
                        if graph_file.exists():
                            graphs.append((graph_file, info))
                            break
    
    # Also check test graphs
    test_graphs = root / "test" / "graphs"
    if test_graphs.exists():
        for el_file in test_graphs.glob("*.el"):
            # Count edges
            try:
                with open(el_file) as f:
                    edge_count = sum(1 for _ in f)
                if min_edges <= edge_count <= max_edges:
                    graphs.append((el_file, {"name": el_file.stem, "edges": edge_count}))
            except:
                pass
    
    return graphs

def parse_leiden_output(output: str, algo_id: int) -> Dict:
    """Parse Leiden algorithm output for metrics"""
    result = {
        "num_communities": 0,
        "num_passes": 0,
        "resolution": 1.0,
        "modularity": 0.0,
        "reorder_time": 0.0
    }
    
    # Parse based on algorithm
    patterns = {
        "num_communities": [
            r"Num Communities:\s*([\d.]+)",
            r"Num Comm:\s*([\d.]+)",  # GraphBrewOrder uses this
            r"LeidenCSR Communities:\s*([\d.]+)",
            r"Dendrogram Roots:\s*([\d.]+)",
            r"communities:\s*(\d+)",
        ],
        "num_passes": [
            r"Num Passes:\s*([\d.]+)",
            r"LeidenCSR Passes:\s*([\d.]+)",
            r"Leiden Passes:\s*([\d.]+)",
            r"Community Passes Stored:\s*([\d.]+)",
        ],
        "resolution": [
            r"Resolution:\s*([\d.]+)",
            r"resolution=([\d.]+)",
            r"Leiden Resolution:\s*([\d.]+)",
        ],
        "modularity": [
            r"Modularity:\s*([\d.]+)",
        ],
    }
    
    for key, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, output)
            if match:
                result[key] = float(match.group(1))
                break
    
    # Get fair reorder time
    result["reorder_time"] = parse_reorder_time_from_converter(output)
    
    return result

def run_leiden_variant(converter: Path, graph: Path, algo_id: int, 
                       variant: str = "", resolution: float = None) -> Optional[LeidenResult]:
    """Run a single Leiden variant and collect metrics"""
    
    # Build command - format: -o algo_id:options
    # For LeidenCSR (17): -o 17:variant:resolution:iterations:passes
    cmd = [str(converter), "-f", str(graph), "-b", "/tmp/leiden_test.sg"]
    
    if variant:
        cmd.extend(["-o", f"{algo_id}:{variant}"])
    else:
        cmd.extend(["-o", str(algo_id)])
    
    algo_names = {
        12: "GraphBrewOrder",
        15: "LeidenOrder",
        16: "LeidenDendrogram", 
        17: "LeidenCSR"
    }
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout + result.stderr
        
        metrics = parse_leiden_output(output, algo_id)
        
        return LeidenResult(
            algorithm=algo_names.get(algo_id, f"Algorithm_{algo_id}"),
            algo_id=algo_id,
            variant=variant or "default",
            reorder_time=metrics["reorder_time"] or 0.0,
            num_communities=int(metrics["num_communities"]),
            num_passes=int(metrics["num_passes"]),
            resolution=metrics["resolution"],
            modularity=metrics["modularity"],
            raw_output=output
        )
    except subprocess.TimeoutExpired:
        print(f"  Timeout for algo {algo_id} variant {variant}")
        return None
    except Exception as e:
        print(f"  Error running algo {algo_id}: {e}")
        return None

def compare_leiden_variants(graph: Path, graph_info: Dict, 
                           include_variants: bool = True) -> List[LeidenResult]:
    """Compare all Leiden variants on a single graph"""
    
    converter = find_converter()
    results = []
    
    # Define variants to test
    # Format for CLI: -o algo:variant:resolution:iterations:passes
    # LeidenCSR (17): -o 17:hubsort:1.0:10:5
    variants_to_test = [
        # (algo_id, variant_options, description)
        (15, "", "LeidenOrder (native Leiden)"),
        (16, "hybrid", "LeidenDendrogram-hybrid"),
        (16, "dfs", "LeidenDendrogram-dfs"),
        (16, "dfshub", "LeidenDendrogram-dfshub"),
        (16, "bfs", "LeidenDendrogram-bfs"),
        (17, "hubsort:1.0:10:1", "LeidenCSR-hubsort-1pass"),
        (17, "hubsort:1.0:10:5", "LeidenCSR-hubsort-5pass"),
        (17, "hubsort:1.0:10:10", "LeidenCSR-hubsort-10pass"),
        (17, "dfs:1.0:10:5", "LeidenCSR-dfs-5pass"),
        (12, "", "GraphBrewOrder"),
    ]
    
    if not include_variants:
        # Just test default variants
        variants_to_test = [
            (15, "", "LeidenOrder"),
            (16, "hybrid", "LeidenDendrogram"),
            (17, "hubsort:1.0:10:5", "LeidenCSR-5pass"),
            (12, "", "GraphBrewOrder"),
        ]
    
    for algo_id, variant, desc in variants_to_test:
        print(f"  Testing {desc}...", end=" ", flush=True)
        result = run_leiden_variant(converter, graph, algo_id, variant)
        if result:
            results.append(result)
            print(f"✓ {result.num_communities} communities, {result.reorder_time:.4f}s")
        else:
            print("✗ failed")
    
    return results

def print_comparison_table(results: List[LeidenResult], graph_name: str):
    """Print a formatted comparison table"""
    
    if not results:
        print("No results to display")
        return
    
    print(f"\n{'='*80}")
    print(f"LEIDEN VARIANTS COMPARISON - {graph_name}")
    print(f"{'='*80}")
    
    # Header
    print(f"{'Algorithm':<30} {'Communities':>12} {'Passes':>8} {'Time (s)':>10} {'Resolution':>10}")
    print(f"{'-'*30} {'-'*12} {'-'*8} {'-'*10} {'-'*10}")
    
    # Sort by algorithm name
    sorted_results = sorted(results, key=lambda x: (x.algo_id, x.variant))
    
    # Reference values from LeidenOrder (native Leiden)
    leiden_order = next((r for r in results if r.algo_id == 15), None)
    ref_communities = leiden_order.num_communities if leiden_order else 0
    
    for r in sorted_results:
        name = f"{r.algorithm}"
        if r.variant and r.variant != "default":
            name += f"-{r.variant}"
        
        # Highlight community count differences from LeidenOrder
        comm_diff = ""
        if ref_communities > 0 and r.num_communities != ref_communities:
            diff = r.num_communities - ref_communities
            comm_diff = f" ({diff:+d})"
        
        print(f"{name:<30} {r.num_communities:>12}{comm_diff:<6} {r.num_passes:>8} {r.reorder_time:>10.4f} {r.resolution:>10.2f}")
    
    # Summary
    print(f"\n{'='*80}")
    print("QUALITY COMPARISON (vs LeidenOrder - native Leiden reference)")
    print(f"{'='*80}")
    
    if leiden_order:
        print(f"Reference: LeidenOrder detected {leiden_order.num_communities} communities in {leiden_order.num_passes} passes")
        print()
        
        for r in sorted_results:
            if r.algo_id == 15:
                continue
            
            name = f"{r.algorithm}"
            if r.variant and r.variant != "default":
                name += f"-{r.variant}"
            
            comm_ratio = r.num_communities / ref_communities if ref_communities > 0 else 0
            speed_ratio = leiden_order.reorder_time / r.reorder_time if r.reorder_time > 0 else 0
            
            status = "≈" if 0.8 <= comm_ratio <= 1.2 else ("+" if comm_ratio > 1 else "-")
            
            print(f"  {name:<28}: {r.num_communities:>4} communities ({comm_ratio:.2f}x) {status}, "
                  f"{speed_ratio:.1f}x faster")

def main():
    parser = argparse.ArgumentParser(description="Compare Leiden variant quality and performance")
    parser.add_argument("--graph", "-g", type=str, help="Path to specific graph file")
    parser.add_argument("--min-edges", type=int, default=100, help="Minimum edges for auto-discovery")
    parser.add_argument("--max-edges", type=int, default=10000000, help="Maximum edges for auto-discovery")
    parser.add_argument("--all-variants", "-a", action="store_true", help="Test all variant options")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file for results")
    
    args = parser.parse_args()
    
    print("="*80)
    print("LEIDEN VARIANTS COMMUNITY QUALITY COMPARISON")
    print("="*80)
    print()
    print("Comparing:")
    print("  - LeidenOrder (15): Native optimized Leiden library [REFERENCE]")
    print("  - LeidenDendrogram (16): Leiden + dendrogram traversal")
    print("  - LeidenCSR (17): Fast native CSR implementation")
    print("  - GraphBrewOrder (12): Leiden + per-community RabbitOrder")
    print()
    
    all_results = {}
    
    if args.graph:
        # Test specific graph
        graph_path = Path(args.graph)
        if not graph_path.exists():
            print(f"Error: Graph file not found: {graph_path}")
            sys.exit(1)
        
        print(f"Testing graph: {graph_path.name}")
        results = compare_leiden_variants(graph_path, {"name": graph_path.stem}, args.all_variants)
        print_comparison_table(results, graph_path.stem)
        all_results[graph_path.stem] = [asdict(r) for r in results]
    else:
        # Auto-discover graphs
        graphs = find_graphs(args.min_edges, args.max_edges)
        
        if not graphs:
            print(f"No graphs found with {args.min_edges}-{args.max_edges} edges")
            print("Using test graphs...")
            root = get_project_root()
            test_graphs = list((root / "test" / "graphs").glob("*.el"))
            if test_graphs:
                graphs = [(g, {"name": g.stem, "edges": 0}) for g in test_graphs]
        
        for graph_path, graph_info in graphs:
            print(f"\n{'='*80}")
            print(f"Testing: {graph_path.name}")
            if "edges" in graph_info:
                print(f"  Edges: {graph_info.get('edges', 'unknown')}")
            
            results = compare_leiden_variants(graph_path, graph_info, args.all_variants)
            print_comparison_table(results, graph_path.stem)
            all_results[graph_path.stem] = [asdict(r) for r in results]
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        root = get_project_root()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = root / "results" / f"leiden_comparison_{timestamp}.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "graphs": all_results
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
