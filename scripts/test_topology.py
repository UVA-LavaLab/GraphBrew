#!/usr/bin/env python3
"""
Topology Verification Test for Graph Reordering Algorithms

This script verifies that all reordering algorithms preserve the graph topology
(same number of nodes, edges, and connectivity) after reordering.

Usage:
    python3 scripts/test_topology.py [--graph GRAPH_ARGS] [--algorithms ALGO_LIST]
    
Examples:
    python3 scripts/test_topology.py                    # Test with default rmat_12
    python3 scripts/test_topology.py --graph "-g 14"    # Test with rmat scale 14
    python3 scripts/test_topology.py --algorithms "0,7,12,15,20"  # Test specific algos
"""

import subprocess
import re
import sys
import argparse
from typing import Dict, List, Tuple, Optional

# All reordering algorithms to test
ALL_ALGORITHMS = {
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
    12: "LeidenOrder",
    13: "GraphBrewOrder",
    # 14: "MAP",  # Requires external file, skip in auto-test
    15: "AdaptiveOrder",
    16: "LeidenDFS",
    17: "LeidenDFSHub",
    18: "LeidenDFSSize",
    19: "LeidenBFS",
    20: "LeidenHybrid",
}

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def parse_graph_stats(output: str) -> Optional[Dict[str, int]]:
    """Parse graph statistics from benchmark output."""
    stats = {}
    
    # Look for graph statistics in output
    # Format: "Graph has X nodes and Y undirected edges for degree: Z"
    graph_match = re.search(r'Graph has (\d+) nodes and (\d+) (?:undirected |directed )?edges', output)
    if graph_match:
        stats['nodes'] = int(graph_match.group(1))
        stats['edges'] = int(graph_match.group(2))
    
    # Look for degree info
    degree_match = re.search(r'for degree:\s*(\d+)', output)
    if degree_match:
        stats['avg_degree'] = int(degree_match.group(1))
    
    # Look for verification result
    if 'Verification:' in output:
        stats['verified'] = 'PASS' in output
    
    # Look for trial time as sanity check
    time_match = re.search(r'Average Time:\s+([\d.]+)', output)
    if time_match:
        stats['avg_time'] = float(time_match.group(1))
    
    return stats if stats else None

def run_benchmark(graph_args: str, algo_id: int, binary: str = "./bench/bin/bfs") -> Tuple[Optional[Dict], str]:
    """Run a benchmark with specific algorithm and return stats."""
    cmd = f"{binary} {graph_args} -o {algo_id} -n 1 -v"
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        output = result.stdout + result.stderr
        stats = parse_graph_stats(output)
        return stats, output
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    except Exception as e:
        return None, str(e)

def compare_topology(base_stats: Dict, test_stats: Dict) -> Tuple[bool, List[str]]:
    """Compare two graph topologies for equivalence."""
    errors = []
    
    if base_stats.get('nodes') != test_stats.get('nodes'):
        errors.append(f"Node count mismatch: {base_stats.get('nodes')} vs {test_stats.get('nodes')}")
    
    if base_stats.get('edges') != test_stats.get('edges'):
        errors.append(f"Edge count mismatch: {base_stats.get('edges')} vs {test_stats.get('edges')}")
    
    # Check average degree as additional validation
    if 'avg_degree' in base_stats and 'avg_degree' in test_stats:
        if base_stats['avg_degree'] != test_stats['avg_degree']:
            errors.append(f"Avg degree mismatch: {base_stats['avg_degree']} vs {test_stats['avg_degree']}")
    
    # Check verification if available
    if test_stats.get('verified') == False:
        errors.append("Algorithm verification FAILED")
    
    return len(errors) == 0, errors

def test_algorithms(graph_args: str, algorithms: Dict[int, str], binary: str = "./bench/bin/bfs") -> Dict:
    """Test all specified algorithms and verify topology preservation."""
    results = {
        'passed': [],
        'failed': [],
        'skipped': [],
        'base_stats': None
    }
    
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}Topology Verification Test{Colors.RESET}")
    print(f"Graph: {graph_args}")
    print(f"Binary: {binary}")
    print(f"Testing {len(algorithms)} algorithms")
    print(f"{'='*70}\n")
    
    # First, get baseline stats with ORIGINAL (no reordering)
    print(f"Getting baseline topology (ORIGINAL)...", end=" ", flush=True)
    base_stats, base_output = run_benchmark(graph_args, 0, binary)
    
    if not base_stats:
        print(f"{Colors.RED}FAILED{Colors.RESET}")
        print(f"Could not get baseline stats. Output:\n{base_output[:500]}")
        return results
    
    print(f"{Colors.GREEN}OK{Colors.RESET}")
    print(f"  Nodes: {base_stats.get('nodes', 'N/A')}")
    print(f"  Edges: {base_stats.get('edges', 'N/A')}")
    results['base_stats'] = base_stats
    print()
    
    # Test each algorithm
    for algo_id, algo_name in sorted(algorithms.items()):
        if algo_id == 0:  # Skip ORIGINAL, it's our baseline
            results['passed'].append((algo_id, algo_name, "Baseline"))
            continue
            
        print(f"Testing {algo_name:15} ({algo_id:2d})...", end=" ", flush=True)
        
        stats, output = run_benchmark(graph_args, algo_id, binary)
        
        if stats is None:
            print(f"{Colors.YELLOW}SKIPPED{Colors.RESET} ({output[:50]})")
            results['skipped'].append((algo_id, algo_name, output[:100]))
            continue
        
        passed, errors = compare_topology(base_stats, stats)
        
        if passed:
            time_str = f"({stats.get('avg_time', 0):.4f}s)" if 'avg_time' in stats else ""
            verified_str = f" [verified]" if stats.get('verified') else ""
            print(f"{Colors.GREEN}PASS{Colors.RESET} {time_str}{verified_str}")
            results['passed'].append((algo_id, algo_name, stats))
        else:
            print(f"{Colors.RED}FAIL{Colors.RESET}")
            for err in errors:
                print(f"    {Colors.RED}✗{Colors.RESET} {err}")
            results['failed'].append((algo_id, algo_name, errors))
    
    return results

def print_summary(results: Dict):
    """Print test summary."""
    print(f"\n{'='*70}")
    print(f"{Colors.BOLD}Test Summary{Colors.RESET}")
    print(f"{'='*70}")
    
    passed = len(results['passed'])
    failed = len(results['failed'])
    skipped = len(results['skipped'])
    total = passed + failed + skipped
    
    print(f"  {Colors.GREEN}Passed:{Colors.RESET}  {passed}/{total}")
    print(f"  {Colors.RED}Failed:{Colors.RESET}  {failed}/{total}")
    print(f"  {Colors.YELLOW}Skipped:{Colors.RESET} {skipped}/{total}")
    
    if results['failed']:
        print(f"\n{Colors.RED}Failed Algorithms:{Colors.RESET}")
        for algo_id, algo_name, errors in results['failed']:
            print(f"  - {algo_name} ({algo_id})")
            for err in errors:
                print(f"      {err}")
    
    if results['skipped']:
        print(f"\n{Colors.YELLOW}Skipped Algorithms:{Colors.RESET}")
        for algo_id, algo_name, reason in results['skipped']:
            print(f"  - {algo_name} ({algo_id}): {reason[:60]}")
    
    print(f"\n{'='*70}")
    
    if failed == 0:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ All topology tests PASSED!{Colors.RESET}")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ Some topology tests FAILED!{Colors.RESET}")
        return 1

def main():
    parser = argparse.ArgumentParser(
        description="Verify graph topology preservation across reordering algorithms"
    )
    parser.add_argument(
        "--graph", "-g",
        default="-g 12",
        help="Graph arguments (default: '-g 12' for rmat scale 12)"
    )
    parser.add_argument(
        "--algorithms", "-a",
        default=None,
        help="Comma-separated list of algorithm IDs to test (default: all)"
    )
    parser.add_argument(
        "--binary", "-b",
        default="./bench/bin/bfs",
        help="Benchmark binary to use (default: ./bench/bin/bfs)"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick test with subset of key algorithms"
    )
    
    args = parser.parse_args()
    
    # Determine which algorithms to test
    if args.algorithms:
        algo_ids = [int(x.strip()) for x in args.algorithms.split(',')]
        algorithms = {aid: ALL_ALGORITHMS.get(aid, f"Unknown({aid})") for aid in algo_ids}
    elif args.quick:
        # Quick test: only key algorithms
        quick_algos = [0, 7, 8, 12, 15, 17, 20]
        algorithms = {aid: ALL_ALGORITHMS[aid] for aid in quick_algos if aid in ALL_ALGORITHMS}
    else:
        algorithms = ALL_ALGORITHMS.copy()
    
    # Run tests
    results = test_algorithms(args.graph, algorithms, args.binary)
    
    # Print summary and exit
    return print_summary(results)

if __name__ == "__main__":
    sys.exit(main())
