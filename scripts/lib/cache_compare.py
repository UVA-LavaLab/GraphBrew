#!/usr/bin/env python3
"""
Quick Cache Comparison - Compare GraphBrew +Variants vs RabbitOrder
=============================================================

Runs cache simulation benchmarks comparing different community detection
variants to measure their impact on cache performance:

- GraphBrew variants (leiden, rabbit, hubcluster + ordering strategies)
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

from .utils import BIN_SIM_DIR, GRAPHS_DIR, GRAPHBREW_LAYERS, TIMEOUT_BENCHMARK

# Config
GRAPHS = ["web-Google", "web-BerkStan", "as-Skitter", "wiki-Talk", "roadNet-CA"]  # Variety of graph types


def _build_cache_compare_variants():
    """Build VARIANTS list from SSOT registry + GRAPHBREW_LAYERS.

    Returns list of (cli_option, display_name) tuples.
    """
    variants = []
    # GraphBrew presets + a few key compound strategies
    presets = list(GRAPHBREW_LAYERS["preset"].keys()) if GRAPHBREW_LAYERS else ["leiden", "rabbit", "hubcluster"]
    key_strategies = ["hrab", "dfs", "bfs", "conn"]  # Most interesting for cache comparison
    # Base presets
    for preset in presets:
        variants.append((f"12:{preset}", f"GraphBrew-{preset.title()}"))
    # Compound: leiden × key strategies
    for strat in key_strategies:
        variants.append((f"12:leiden:{strat}", f"GraphBrew-Leiden-{strat.upper()}"))
    # RabbitOrder variants
    variants.append(("8:csr", "RabbitCSR"))
    variants.append(("8:boost", "RabbitBoost"))
    # LeidenOrder baseline
    variants.append(("15", "LeidenOrder"))
    return variants


VARIANTS = _build_cache_compare_variants()
_COMPARE_BENCHMARKS = ["pr"]  # Just PR for now (cache sim subset)
BIN_SIM = str(BIN_SIM_DIR)
_GRAPHS_DIR = str(GRAPHS_DIR)
TIMEOUT = TIMEOUT_BENCHMARK

def run_cache_sim(graph, variant_opt, benchmark):
    """Run cache simulation and extract results."""
    graph_path = f"{_GRAPHS_DIR}/{graph}/{graph}.mtx"
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
    print("GraphBrew vs RabbitOrder Cache Performance Comparison")
    print("=" * 80)
    
    results = {}
    
    for graph in GRAPHS:
        print(f"\n{'='*60}")
        print(f"Graph: {graph}")
        print(f"{'='*60}")
        
        results[graph] = {}
        
        for variant_opt, variant_name in VARIANTS:
            print(f"  Running {variant_name}...", end=" ", flush=True)
            
            for bench in _COMPARE_BENCHMARKS:
                r = run_cache_sim(graph, variant_opt, bench)
                
                if 'error' in r:
                    print(f"ERROR: {r['error']}")
                    continue
                    
                results[graph][variant_name] = r
                l1 = r.get('l1_hit')
                l2 = r.get('l2_hit')
                mem = r.get('mem_access')
                rt = r.get('reorder_time')
                print(f"L1={l1:.1f}% L2={l2:.1f}% Mem={mem:,} Time={rt:.2f}s" if all(v is not None for v in (l1, l2, mem, rt)) else "Parse error (partial output)")
    
    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE (PR Benchmark)")
    print("=" * 100)
    print(f"{'Graph':<15} {'Variant':<12} {'L1 Hit%':>10} {'L2 Hit%':>10} {'Mem Access':>15} {'Reorder(s)':>12} {'Modularity':>10}")
    print("-" * 100)
    
    for graph in GRAPHS:
        for _variant_opt, variant_name in VARIANTS:
            if variant_name in results.get(graph, {}):
                r = results[graph][variant_name]
                if 'error' not in r and r.get('l1_hit') is not None:
                    print(f"{graph:<15} {variant_name:<12} {r['l1_hit']:>10.2f} {r['l2_hit'] or 0:>10.2f} {r['mem_access'] or 0:>15,} {r['reorder_time'] or 0:>12.2f} {r['modularity'] or 0:>10.4f}")
        print()

if __name__ == "__main__":
    main()
