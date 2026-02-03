#!/usr/bin/env python3
"""
Quick Cache Comparison - Compare GVE Variants vs RabbitOrder
============================================================

Runs cache simulation benchmarks comparing different community detection
variants to measure their impact on cache performance:

- GVE-Leiden variants (gve, gveopt, gveopt2, gveadaptive, gvedendo, gveoptdendo)
- RabbitOrder variants (csr, boost)

Outputs:
- L1/L2 hit rates
- Memory accesses  
- Reorder time
- Modularity (if available)

Usage:
    python scripts/quick_cache_compare.py

Prerequisites:
    - Cache simulation binaries built: make sim
    - Graphs downloaded: python scripts/graphbrew_experiment.py --download-only

Author: GraphBrew Team
"""
import subprocess
import re
import sys
import os
from pathlib import Path

# Config
GRAPHS = ["web-Google", "web-BerkStan", "as-Skitter", "wiki-Talk", "roadNet-CA"]  # Variety of graph types
VARIANTS = [
    ("17:gve:auto", "GVE-Auto"),
    ("17:gveopt:auto", "GVEOpt-Auto"),
    ("17:gveopt2:auto", "GVEOpt2-Auto"),          # CSR-based aggregation, auto resolution
    ("17:gveopt2:2.0", "GVEOpt2-R2.0"),           # CSR-based aggregation, fixed resolution
    ("17:gveadaptive:dynamic", "GVEAdaptive"),    # Dynamic resolution adjustment
    ("17:gveadaptive:dynamic_2.0", "GVEAdaptive-2.0"),  # Dynamic starting at 2.0
    ("17:gveoptsort:auto", "GVEOptSort"),         # Multi-level sort
    ("17:gveturbo:auto", "GVETurbo"),             # Speed-optimized
    ("17:gvefast:auto", "GVEFast"),               # CSR buffer reuse
    ("17:gvedendo:auto", "GVEDendo"),
    ("17:gveoptdendo:auto", "GVEOptDendo"),
    ("8:csr", "RabbitCSR"),
    ("8:boost", "RabbitBoost"),
]
BENCHMARKS = ["pr"]  # Just PR for now
BIN_SIM = "bench/bin_sim"
GRAPHS_DIR = "results/graphs"
TIMEOUT = 300  # 5 min timeout per run

def run_cache_sim(graph, variant_opt, benchmark):
    """Run cache simulation and extract results."""
    graph_path = f"{GRAPHS_DIR}/{graph}/{graph}.mtx"
    cmd = [
        f"{BIN_SIM}/{benchmark}",
        "-f", graph_path,
        "-s",
        "-o", variant_opt,
        "-n", "1"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
        output = result.stdout + result.stderr
        
        # Parse cache results
        l1_match = re.search(r'L1 Hit Rate:\s*([\d.]+)', output)
        l2_match = re.search(r'L2 Hit Rate:\s*([\d.]+)', output)
        mem_match = re.search(r'Memory Accesses:\s*(\d+)', output)
        reorder_match = re.search(r'Reorder Time:\s*([\d.]+)', output)
        mod_match = re.search(r'[Mm]odularity[:\s]*([\d.]+)', output)
        
        return {
            'l1_hit': float(l1_match.group(1)) if l1_match else None,
            'l2_hit': float(l2_match.group(1)) if l2_match else None,
            'mem_access': int(mem_match.group(1)) if mem_match else None,
            'reorder_time': float(reorder_match.group(1)) if reorder_match else None,
            'modularity': float(mod_match.group(1)) if mod_match else None,
            'output': output
        }
    except subprocess.TimeoutExpired:
        return {'error': 'timeout'}
    except Exception as e:
        return {'error': str(e)}

def main():
    print("=" * 80)
    print("GVE vs RabbitOrder Cache Performance Comparison")
    print("=" * 80)
    
    results = {}
    
    for graph in GRAPHS:
        print(f"\n{'='*60}")
        print(f"Graph: {graph}")
        print(f"{'='*60}")
        
        results[graph] = {}
        
        for variant_opt, variant_name in VARIANTS:
            print(f"  Running {variant_name}...", end=" ", flush=True)
            
            for bench in BENCHMARKS:
                r = run_cache_sim(graph, variant_opt, bench)
                
                if 'error' in r:
                    print(f"ERROR: {r['error']}")
                    continue
                    
                results[graph][variant_name] = r
                print(f"L1={r['l1_hit']:.1f}% L2={r['l2_hit']:.1f}% Mem={r['mem_access']:,} Time={r['reorder_time']:.2f}s")
    
    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE (PR Benchmark)")
    print("=" * 100)
    print(f"{'Graph':<15} {'Variant':<12} {'L1 Hit%':>10} {'L2 Hit%':>10} {'Mem Access':>15} {'Reorder(s)':>12} {'Modularity':>10}")
    print("-" * 100)
    
    for graph in GRAPHS:
        for variant_opt, variant_name in VARIANTS:
            if variant_name in results.get(graph, {}):
                r = results[graph][variant_name]
                if 'error' not in r:
                    print(f"{graph:<15} {variant_name:<12} {r['l1_hit']:>10.2f} {r['l2_hit']:>10.2f} {r['mem_access']:>15,} {r['reorder_time']:>12.2f} {r['modularity'] or 0:>10.4f}")
        print()

if __name__ == "__main__":
    main()
