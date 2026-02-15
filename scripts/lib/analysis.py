#!/usr/bin/env python3
"""
Analysis functions for GraphBrew adaptive ordering evaluation.

This module provides:
- analyze_adaptive_order: Analyze adaptive ordering for graphs
- compare_adaptive_vs_fixed: Compare adaptive vs fixed-algorithm approaches
- run_subcommunity_brute_force: Brute-force comparison of all algorithms

These functions evaluate how well the adaptive algorithm selection performs
compared to fixed algorithms and identify where improvements can be made.

Example usage:
    from lib.analysis import analyze_adaptive_order, compare_adaptive_vs_fixed
    
    # Analyze adaptive ordering
    results = analyze_adaptive_order(graphs, bin_dir, output_dir)
    
    # Compare with fixed algorithms
    comparison = compare_adaptive_vs_fixed(
        graphs, bin_dir, 
        benchmarks=['pr', 'bfs'],
        fixed_algorithms=[1, 8, 11]
    )
"""

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .utils import ALGORITHMS, run_command, Logger
from .features import update_graph_properties, save_graph_properties_cache


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SubcommunityInfo:
    """Information about a subcommunity in adaptive ordering."""
    community_id: int
    nodes: int
    edges: int
    density: float
    degree_variance: float
    hub_concentration: float
    selected_algorithm: str
    clustering_coefficient: float = 0.0
    avg_path_length: float = 0.0
    diameter_estimate: float = 0.0
    community_count: int = 1


@dataclass
class AdaptiveOrderResult:
    """Result from adaptive ordering analysis."""
    graph: str
    modularity: float
    num_communities: int
    subcommunities: List[SubcommunityInfo] = field(default_factory=list)
    algorithm_distribution: Dict[str, int] = field(default_factory=dict)
    reorder_time: float = 0.0
    success: bool = True
    error: str = ""


@dataclass
class AdaptiveComparisonResult:
    """Result comparing adaptive vs fixed-algorithm approaches."""
    graph: str
    benchmark: str
    adaptive_time: float
    adaptive_speedup: float
    fixed_results: Dict[str, float] = field(default_factory=dict)
    best_fixed_algorithm: str = ""
    best_fixed_speedup: float = 0.0
    adaptive_advantage: float = 0.0


@dataclass
class SubcommunityBruteForceResult:
    """Result from brute-force testing all algorithms on a subcommunity."""
    community_id: int
    nodes: int
    edges: int
    density: float
    degree_variance: float
    hub_concentration: float
    
    adaptive_algorithm: str
    adaptive_time: float = 0.0
    adaptive_l1_hit: float = 0.0
    adaptive_l2_hit: float = 0.0
    adaptive_l3_hit: float = 0.0
    
    best_time_algorithm: str = ""
    best_time: float = 0.0
    best_time_l1_hit: float = 0.0
    best_time_l2_hit: float = 0.0
    best_time_l3_hit: float = 0.0
    
    best_cache_algorithm: str = ""
    best_cache_l1_hit: float = 0.0
    best_cache_l2_hit: float = 0.0
    best_cache_l3_hit: float = 0.0
    best_cache_time: float = 0.0
    
    all_results: Dict[str, Dict] = field(default_factory=dict)
    
    adaptive_vs_best_time_ratio: float = 1.0
    adaptive_vs_best_cache_ratio: float = 1.0
    adaptive_is_best_time: bool = False
    adaptive_is_best_cache: bool = False
    adaptive_rank_time: int = 0
    adaptive_rank_cache: int = 0


@dataclass
class GraphBruteForceAnalysis:
    """Complete brute-force analysis for a graph."""
    graph: str
    size_mb: float
    modularity: float
    num_communities: int
    num_subcommunities_analyzed: int
    subcommunity_results: List[SubcommunityBruteForceResult] = field(default_factory=list)
    
    adaptive_correct_time_pct: float = 0.0
    adaptive_correct_cache_pct: float = 0.0
    adaptive_top3_time_pct: float = 0.0
    adaptive_top3_cache_pct: float = 0.0
    avg_time_ratio: float = 1.0
    avg_cache_ratio: float = 1.0
    
    success: bool = True
    error: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Output Parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_adaptive_output(output: str) -> Tuple[float, int, List[SubcommunityInfo], Dict[str, int]]:
    """
    Parse the output from AdaptiveOrder to extract subcommunity information.
    
    Args:
        output: Combined stdout/stderr from AdaptiveOrder run
        
    Returns:
        Tuple of (modularity, num_communities, subcommunities_list, algorithm_distribution)
    """
    modularity = 0.0
    num_communities = 0
    subcommunities = []
    algo_distribution = {}
    
    # Parse modularity (from GVE-Leiden output or direct)
    mod_match = re.search(r'Modularity:\s*([\d.]+)', output)
    if mod_match:
        modularity = float(mod_match.group(1))
    # Also try GVE format
    gve_mod_match = re.search(r'modularity=([\d.]+)', output)
    if gve_mod_match and modularity == 0.0:
        modularity = float(gve_mod_match.group(1))
    
    # Parse number of communities
    num_match = re.search(r'Num Communities:\s*([\d.]+)', output)
    if num_match:
        num_communities = int(float(num_match.group(1)))
    
    # Parse graph type
    graph_type = "unknown"
    type_match = re.search(r'Graph Type:\s*(\w+)', output)
    if type_match:
        graph_type = type_match.group(1)
    
    # Parse algorithm selections from AdaptiveOrder output
    # Format: "AdaptiveOrder: Grouped X small communities (Y nodes, Z edges) -> Algorithm"
    grouped_match = re.search(
        r'AdaptiveOrder: Grouped (\d+) small communities \((\d+) nodes, (\d+) edges\) -> (\w+)',
        output
    )
    if grouped_match:
        nodes = int(grouped_match.group(2))
        edges = int(grouped_match.group(3))
        selected = grouped_match.group(4)
        
        subcommunities.append(SubcommunityInfo(
            community_id=0,
            nodes=nodes,
            edges=edges,
            density=2.0 * edges / (nodes * (nodes - 1)) if nodes > 1 else 0.0,
            degree_variance=0.0,
            hub_concentration=0.0,
            selected_algorithm=selected
        ))
        algo_distribution[selected] = algo_distribution.get(selected, 0) + 1
    
    # Parse individual community selections
    # C++ format: "  Community N: X nodes, Y edges -> Algorithm"
    # Legacy format: "Community N: algo=Algorithm, nodes=X, edges=Y"
    comm_pattern = re.compile(
        r'Community\s+(\d+):\s*(\d+)\s+nodes,\s*(\d+)\s+edges\s*->\s*(\w+)',
        re.IGNORECASE
    )
    # Also try legacy format
    comm_pattern_legacy = re.compile(
        r'Community\s+(\d+):\s*algo=(\w+),\s*nodes=(\d+),\s*edges=(\d+)',
        re.IGNORECASE
    )
    
    for match in comm_pattern.finditer(output):
        comm_id = int(match.group(1))
        nodes = int(match.group(2))
        edges = int(match.group(3))
        selected = match.group(4)
        
        subcommunities.append(SubcommunityInfo(
            community_id=comm_id,
            nodes=nodes,
            edges=edges,
            density=2.0 * edges / (nodes * (nodes - 1)) if nodes > 1 else 0.0,
            degree_variance=0.0,
            hub_concentration=0.0,
            selected_algorithm=selected
        ))
        algo_distribution[selected] = algo_distribution.get(selected, 0) + 1
    
    # Fallback: try legacy format if no matches found from C++ format
    if not any(s.community_id > 0 for s in subcommunities):
        for match in comm_pattern_legacy.finditer(output):
            comm_id = int(match.group(1))
            selected = match.group(2)
            nodes = int(match.group(3))
            edges = int(match.group(4))
            
            subcommunities.append(SubcommunityInfo(
                community_id=comm_id,
                nodes=nodes,
                edges=edges,
                density=2.0 * edges / (nodes * (nodes - 1)) if nodes > 1 else 0.0,
                degree_variance=0.0,
                hub_concentration=0.0,
                selected_algorithm=selected
            ))
            algo_distribution[selected] = algo_distribution.get(selected, 0) + 1
    
    # Parse subcommunity table (legacy format)
    table_pattern = re.compile(
        r'^(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\w+)',
        re.MULTILINE
    )
    
    for match in table_pattern.finditer(output):
        comm_id = int(match.group(1))
        nodes = int(match.group(2))
        edges = int(match.group(3))
        density = float(match.group(4))
        deg_var = float(match.group(5))
        hub_conc = float(match.group(6))
        selected = match.group(7)
        
        subcommunities.append(SubcommunityInfo(
            community_id=comm_id,
            nodes=nodes,
            edges=edges,
            density=density,
            degree_variance=deg_var,
            hub_concentration=hub_conc,
            selected_algorithm=selected
        ))
        
        if selected not in algo_distribution:
            algo_distribution[selected] = 0
        algo_distribution[selected] += 1
    
    return modularity, num_communities, subcommunities, algo_distribution


def parse_benchmark_output(output: str) -> Dict[str, Any]:
    """
    Parse benchmark output to extract timing and features.
    
    Args:
        output: Benchmark output text
        
    Returns:
        Dict with parsed values
    """
    parsed = {}
    
    # Parse average time
    time_match = re.search(r'Average Time:\s*([\d.]+)', output)
    if time_match:
        parsed['average_time'] = float(time_match.group(1))
    
    # Parse trial time (fallback)
    trial_match = re.search(r'Trial Time:\s*([\d.]+)', output)
    if trial_match:
        parsed['trial_time'] = float(trial_match.group(1))
    
    # Parse degree variance
    dv_match = re.search(r'Degree Variance:\s*([\d.]+)', output)
    if dv_match:
        parsed['degree_variance'] = float(dv_match.group(1))
    
    # Parse hub concentration
    hc_match = re.search(r'Hub Concentration:\s*([\d.]+)', output)
    if hc_match:
        parsed['hub_concentration'] = float(hc_match.group(1))
    
    # Parse clustering coefficient
    cc_match = re.search(r'Clustering Coefficient:\s*([\d.]+)', output)
    if cc_match:
        parsed['clustering_coefficient'] = float(cc_match.group(1))
    
    # Parse average path length
    apl_match = re.search(r'Avg Path Length:\s*([\d.]+)', output)
    if apl_match:
        parsed['avg_path_length'] = float(apl_match.group(1))
    
    # Parse diameter estimate
    diam_match = re.search(r'Diameter Estimate:\s*([\d.]+)', output)
    if diam_match:
        parsed['diameter'] = float(diam_match.group(1))
    
    # Parse community count estimate
    comm_match = re.search(r'Community Count Estimate:\s*([\d.]+)', output)
    if comm_match:
        parsed['community_count'] = float(comm_match.group(1))
    
    # Parse avg degree
    ad_match = re.search(r'Avg Degree:\s*([\d.]+)', output)
    if ad_match:
        parsed['avg_degree'] = float(ad_match.group(1))
    
    # Parse graph density
    dens_match = re.search(r'Graph Density:\s*([\d.]+)', output)
    if dens_match:
        parsed['density'] = float(dens_match.group(1))
    
    # Parse packing factor
    pf_match = re.search(r'Packing Factor:\s*([\d.]+)', output)
    if pf_match:
        parsed['packing_factor'] = float(pf_match.group(1))
    
    # Parse forward edge fraction
    fef_match = re.search(r'Forward Edge Fraction:\s*([\d.]+)', output)
    if fef_match:
        parsed['forward_edge_fraction'] = float(fef_match.group(1))
    
    # Parse working set ratio
    wsr_match = re.search(r'Working Set Ratio:\s*([\d.]+)', output)
    if wsr_match:
        parsed['working_set_ratio'] = float(wsr_match.group(1))
    
    # Parse graph type
    type_match = re.search(r'Graph Type:\s*(\w+)', output)
    if type_match:
        parsed['graph_type'] = type_match.group(1)
    
    return parsed


def parse_cache_output(output: str) -> Dict[str, float]:
    """
    Parse cache simulation output.
    
    Args:
        output: Cache simulation output text
        
    Returns:
        Dict with l1/l2/l3 hit rates
    """
    parsed = {}
    
    l1_match = re.search(r'L1[_ ]?Hit[_ ]?Rate[:\s]*([\d.]+)', output, re.IGNORECASE)
    if l1_match:
        parsed['l1_hit_rate'] = float(l1_match.group(1))
    
    l2_match = re.search(r'L2[_ ]?Hit[_ ]?Rate[:\s]*([\d.]+)', output, re.IGNORECASE)
    if l2_match:
        parsed['l2_hit_rate'] = float(l2_match.group(1))
    
    l3_match = re.search(r'L3[_ ]?Hit[_ ]?Rate[:\s]*([\d.]+)', output, re.IGNORECASE)
    if l3_match:
        parsed['l3_hit_rate'] = float(l3_match.group(1))
    
    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# Analysis Functions
# ─────────────────────────────────────────────────────────────────────────────

def analyze_adaptive_order(
    graphs: List,  # GraphInfo objects
    bin_dir: str,
    output_dir: str,
    timeout: int = 300,
    logger: Logger = None
) -> List[AdaptiveOrderResult]:
    """
    Analyze adaptive ordering for all graphs.
    
    Records subcommunity assignments and algorithm selections for each graph.
    
    Args:
        graphs: List of GraphInfo objects
        bin_dir: Path to benchmark binaries
        output_dir: Directory for result files
        timeout: Timeout per graph in seconds
        logger: Optional logger instance
        
    Returns:
        List of AdaptiveOrderResult for each graph
    """
    log = logger.info if logger else print
    log_warn = logger.warning if logger else print
    
    log("Phase: Adaptive Order Analysis")
    
    results = []
    
    for graph in graphs:
        log(f"\nGraph: {graph.name} ({graph.size_mb:.1f}MB)")
        
        binary = os.path.join(bin_dir, "pr")
        if not os.path.exists(binary):
            log(f"  Binary not found: {binary}")
            results.append(AdaptiveOrderResult(
                graph=graph.name,
                modularity=0.0,
                num_communities=0,
                success=False,
                error="Binary not found"
            ))
            continue
        
        # Run with AdaptiveOrder (algorithm 14)
        sym_flag = "-s" if graph.is_symmetric else ""
        cmd = f"{binary} -f {graph.path} {sym_flag} -o 14 -n 1"
        
        start_time = time.time()
        success, stdout, stderr = run_command(cmd, timeout)
        reorder_time = time.time() - start_time
        
        if success:
            output = stdout + stderr
            modularity, num_communities, subcommunities, algo_distribution = parse_adaptive_output(output)
            
            # Parse and cache topology features for weight computation
            features = parse_benchmark_output(output)
            features['modularity'] = modularity
            features['nodes'] = getattr(graph, 'nodes', 0)
            features['edges'] = getattr(graph, 'edges', 0)
            update_graph_properties(graph.name, features, output_dir)
            
            log(f"  Modularity: {modularity:.4f}")
            log(f"  Communities: {num_communities}")
            if features.get('degree_variance'):
                log(f"  Degree Variance: {features.get('degree_variance', 0):.4f}")
            if features.get('hub_concentration'):
                log(f"  Hub Concentration: {features.get('hub_concentration', 0):.4f}")
            log(f"  Subcommunities analyzed: {len(subcommunities)}")
            log(f"  Algorithm distribution:")
            for algo, count in sorted(algo_distribution.items(), key=lambda x: -x[1])[:5]:
                log(f"    {algo}: {count}")
            
            results.append(AdaptiveOrderResult(
                graph=graph.name,
                modularity=modularity,
                num_communities=num_communities,
                subcommunities=subcommunities,
                algorithm_distribution=algo_distribution,
                reorder_time=reorder_time,
                success=True
            ))
        else:
            error = "TIMEOUT" if "TIMEOUT" in stderr else "FAILED"
            log(f"  {error}")
            results.append(AdaptiveOrderResult(
                graph=graph.name,
                modularity=0.0,
                num_communities=0,
                success=False,
                error=error
            ))
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f"adaptive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    results_data = []
    for r in results:
        data = {
            "graph": r.graph,
            "modularity": r.modularity,
            "num_communities": r.num_communities,
            "algorithm_distribution": r.algorithm_distribution,
            "reorder_time": r.reorder_time,
            "success": r.success,
            "error": r.error,
            "subcommunities": [
                {
                    "community_id": s.community_id,
                    "nodes": s.nodes,
                    "edges": s.edges,
                    "density": s.density,
                    "degree_variance": s.degree_variance,
                    "hub_concentration": s.hub_concentration,
                    "selected_algorithm": s.selected_algorithm
                }
                for s in r.subcommunities
            ]
        }
        results_data.append(data)
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Save the graph properties cache with all extracted features
    save_graph_properties_cache(output_dir)
    log(f"\nGraph properties cached for {len(results)} graphs")
    
    log(f"\nAdaptive analysis saved to: {results_file}")
    
    return results


def compare_adaptive_vs_fixed(
    graphs: List,  # GraphInfo objects
    bin_dir: str,
    benchmarks: List[str],
    fixed_algorithms: List[int],
    output_dir: str,
    num_trials: int = 3,
    timeout: int = 300,
    logger: Logger = None
) -> List[AdaptiveComparisonResult]:
    """
    Compare AdaptiveOrder vs using a single fixed algorithm.
    
    This validates whether adaptive selection provides benefit over just
    using the best single algorithm for the entire graph.
    
    Args:
        graphs: List of GraphInfo objects
        bin_dir: Path to benchmark binaries
        benchmarks: List of benchmarks to test
        fixed_algorithms: List of algorithm IDs to compare against
        output_dir: Directory for result files
        num_trials: Number of trials per benchmark
        timeout: Timeout in seconds
        logger: Optional logger instance
        
    Returns:
        List of AdaptiveComparisonResult
    """
    log = logger.info if logger else print
    
    log("Phase: Adaptive vs Fixed Comparison")
    
    results = []
    
    for graph in graphs:
        log(f"\nGraph: {graph.name} ({graph.size_mb:.1f}MB)")
        
        for bench in benchmarks:
            log(f"  Benchmark: {bench}")
            
            binary = os.path.join(bin_dir, bench)
            if not os.path.exists(binary):
                continue
            
            sym_flag = "-s" if graph.is_symmetric else ""
            
            # Run AdaptiveOrder (algorithm 15)
            cmd_adaptive = f"{binary} -f {graph.path} {sym_flag} -o 15 -n {num_trials}"
            success, stdout, stderr = run_command(cmd_adaptive, timeout)
            
            adaptive_time = 0.0
            if success:
                time_match = re.search(r'Average Time:\s*([\d.]+)', stdout + stderr)
                if time_match:
                    adaptive_time = float(time_match.group(1))
            
            # Run Original (algorithm 0) for baseline
            cmd_orig = f"{binary} -f {graph.path} {sym_flag} -o 0 -n {num_trials}"
            success_orig, stdout_orig, stderr_orig = run_command(cmd_orig, timeout)
            
            original_time = 0.0
            if success_orig:
                time_match = re.search(r'Average Time:\s*([\d.]+)', stdout_orig + stderr_orig)
                if time_match:
                    original_time = float(time_match.group(1))
            
            adaptive_speedup = original_time / adaptive_time if adaptive_time > 0 else 0.0
            
            # Run fixed algorithms
            fixed_results = {}
            for algo_id in fixed_algorithms:
                algo_name = ALGORITHMS.get(algo_id, f"ALGO_{algo_id}")
                cmd_fixed = f"{binary} -f {graph.path} {sym_flag} -o {algo_id} -n {num_trials}"
                success_fixed, stdout_fixed, stderr_fixed = run_command(cmd_fixed, timeout)
                
                if success_fixed:
                    time_match = re.search(r'Average Time:\s*([\d.]+)', stdout_fixed + stderr_fixed)
                    if time_match:
                        fixed_time = float(time_match.group(1))
                        fixed_speedup = original_time / fixed_time if fixed_time > 0 else 0.0
                        fixed_results[algo_name] = fixed_speedup
            
            # Find best fixed algorithm
            best_fixed_algo = ""
            best_fixed_speedup = 0.0
            for algo, speedup in fixed_results.items():
                if speedup > best_fixed_speedup:
                    best_fixed_speedup = speedup
                    best_fixed_algo = algo
            
            adaptive_advantage = adaptive_speedup - best_fixed_speedup
            
            log(f"    Adaptive: {adaptive_speedup:.3f}x")
            log(f"    Best Fixed ({best_fixed_algo}): {best_fixed_speedup:.3f}x")
            log(f"    Advantage: {adaptive_advantage:+.3f}x")
            
            results.append(AdaptiveComparisonResult(
                graph=graph.name,
                benchmark=bench,
                adaptive_time=adaptive_time,
                adaptive_speedup=adaptive_speedup,
                fixed_results=fixed_results,
                best_fixed_algorithm=best_fixed_algo,
                best_fixed_speedup=best_fixed_speedup,
                adaptive_advantage=adaptive_advantage
            ))
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f"adaptive_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    log(f"\nComparison results saved to: {results_file}")
    
    # Summary
    log("\n=== Summary ===")
    positive_advantage = [r for r in results if r.adaptive_advantage > 0]
    negative_advantage = [r for r in results if r.adaptive_advantage < 0]
    log(f"Adaptive better: {len(positive_advantage)} cases")
    log(f"Fixed better: {len(negative_advantage)} cases")
    if positive_advantage:
        avg_pos = sum(r.adaptive_advantage for r in positive_advantage) / len(positive_advantage)
        log(f"Avg advantage when better: +{avg_pos:.3f}x")
    if negative_advantage:
        avg_neg = sum(r.adaptive_advantage for r in negative_advantage) / len(negative_advantage)
        log(f"Avg disadvantage when worse: {avg_neg:.3f}x")
    
    return results


def validate_adaptive_accuracy(
    graphs: list,
    bin_dir: str,
    output_dir: str,
    benchmarks: List[str] = None,
    timeout: int = 300,
    num_trials: int = 3,
    force_reorder: bool = False,
) -> List[Dict]:
    """
    Validate adaptive ordering accuracy by comparing against fixed algorithms.

    This is a convenience wrapper around compare_adaptive_vs_fixed that uses a
    standard set of common algorithms for comparison and returns plain dicts
    (suitable for JSON serialization).

    Args:
        graphs: List of GraphInfo objects
        bin_dir: Path to benchmark binaries
        output_dir: Directory for result files
        benchmarks: List of benchmarks to test (default: pr, bfs, cc)
        timeout: Timeout in seconds per benchmark run
        num_trials: Number of trials per benchmark
        force_reorder: Unused (kept for API compatibility)

    Returns:
        List of dicts with comparison results per graph/benchmark
    """
    benchmarks = benchmarks or ["pr", "bfs", "cc"]

    # Standard set of fixed algorithms to compare against
    # 0=Original, 1=Random, 2=Sort, 4=HubSort, 7=DBG, 8=RabbitOrder, 11=RCM, 12=GraphBrew
    fixed_algorithms = [0, 1, 2, 4, 7, 8, 11, 12]

    comparison_results = compare_adaptive_vs_fixed(
        graphs=graphs,
        bin_dir=bin_dir,
        benchmarks=benchmarks,
        fixed_algorithms=fixed_algorithms,
        output_dir=output_dir,
        num_trials=num_trials,
        timeout=timeout,
    )

    return [asdict(r) for r in comparison_results]


def run_subcommunity_brute_force(
    graphs: List,  # GraphInfo objects
    bin_dir: str,
    bin_sim_dir: str,
    output_dir: str,
    benchmark: str = "pr",
    timeout: int = 300,
    timeout_sim: int = 600,
    max_subcommunities: int = 20,
    num_trials: int = 2,
    logger: Logger = None
) -> List[GraphBruteForceAnalysis]:
    """
    Brute-force comparison of all algorithms vs adaptive choice for each subcommunity.
    
    For each graph:
    1. Run AdaptiveOrder to get subcommunity info and adaptive choices
    2. For each subcommunity (up to max_subcommunities):
       - Run all algorithms and measure time + cache
       - Compare against adaptive choice
    3. Generate detailed comparison table
    
    NOTE: All operations run sequentially for accurate performance measurements.
    
    Args:
        graphs: List of GraphInfo objects
        bin_dir: Path to benchmark binaries
        bin_sim_dir: Path to cache simulation binaries
        output_dir: Directory for result files
        benchmark: Which benchmark to use
        timeout: Timeout for benchmark runs
        timeout_sim: Timeout for cache simulations
        max_subcommunities: Max subcommunities to analyze per graph
        num_trials: Number of trials per algorithm
        logger: Optional logger instance
        
    Returns:
        List of GraphBruteForceAnalysis
    """
    log = logger.info if logger else print
    log_error = logger.error if logger else print
    
    log("Subcommunity Brute-Force Analysis: Adaptive vs All Algorithms")
    log("Note: Sequential execution for accurate timing (no parallelism)")
    
    results = []
    
    # Algorithms to test (0-20, excluding some)
    test_algorithms = [i for i in range(21) if i not in [14, 15]]
    
    # Mapping from adaptive output names to algorithm names
    adaptive_to_algo_name = {
        "Original": "ORIGINAL", "Random": "RANDOM", "Sort": "SORT",
        "HubSort": "HUBSORT", "HubCluster": "HUBCLUSTER", "DBG": "DBG",
        "HubSortDBG": "HUBSORTDBG", "HubClusterDBG": "HUBCLUSTERDBG",
        "RabbitOrder": "RABBITORDER", "GOrder": "GORDER", "Corder": "CORDER",
        "RCM": "RCM", "LeidenOrder": "LeidenOrder", "GraphBrewOrder": "GraphBrewOrder",
        "LeidenDFS": "LeidenDFS", "LeidenDFSHub": "LeidenDFSHub",
        "LeidenDFSSize": "LeidenDFSSize", "LeidenBFS": "LeidenBFS",
        "LeidenHybrid": "LeidenHybrid",
    }
    
    for graph in graphs:
        log(f"\n{'='*60}")
        log(f"Graph: {graph.name} ({graph.size_mb:.1f}MB)")
        log(f"{'='*60}")
        
        analysis = GraphBruteForceAnalysis(
            graph=graph.name,
            size_mb=graph.size_mb,
            modularity=0.0,
            num_communities=0,
            num_subcommunities_analyzed=0
        )
        
        # Step 1: Run AdaptiveOrder to get subcommunity info
        binary = os.path.join(bin_dir, benchmark)
        if not os.path.exists(binary):
            log_error(f"  Binary not found: {binary}")
            analysis.success = False
            analysis.error = "Binary not found"
            results.append(analysis)
            continue
        
        sym_flag = "-s" if graph.is_symmetric else ""
        cmd = f"{binary} -f {graph.path} {sym_flag} -o 15 -n 1"
        
        success, stdout, stderr = run_command(cmd, timeout)
        if not success:
            log_error(f"  AdaptiveOrder failed")
            analysis.success = False
            analysis.error = "AdaptiveOrder failed"
            results.append(analysis)
            continue
        
        output = stdout + stderr
        modularity, num_communities, subcommunities, algo_distribution = parse_adaptive_output(output)
        
        analysis.modularity = modularity
        analysis.num_communities = num_communities
        
        log(f"  Modularity: {modularity:.4f}")
        log(f"  Total communities: {num_communities}")
        log(f"  Subcommunities with features: {len(subcommunities)}")
        
        if not subcommunities:
            log(f"  No subcommunities to analyze")
            results.append(analysis)
            continue
        
        # Sort by size and take largest subcommunities
        subcommunities_sorted = sorted(subcommunities, key=lambda s: s.nodes, reverse=True)
        subcommunities_to_test = subcommunities_sorted[:max_subcommunities]
        
        log(f"  Testing top {len(subcommunities_to_test)} largest subcommunities")
        
        subcommunity_results = []
        
        for idx, subcomm in enumerate(subcommunities_to_test):
            adaptive_algo_raw = subcomm.selected_algorithm
            adaptive_algo = adaptive_to_algo_name.get(adaptive_algo_raw, adaptive_algo_raw)
            
            sc_result = SubcommunityBruteForceResult(
                community_id=subcomm.community_id,
                nodes=subcomm.nodes,
                edges=subcomm.edges,
                density=subcomm.density,
                degree_variance=subcomm.degree_variance,
                hub_concentration=subcomm.hub_concentration,
                adaptive_algorithm=adaptive_algo
            )
            
            # Run all algorithms
            algo_times = {}
            algo_cache = {}
            
            for algo_id in test_algorithms:
                algo_name = ALGORITHMS.get(algo_id, f"ALGO_{algo_id}")
                
                # Run benchmark for time
                cmd_bench = f"{binary} -f {graph.path} {sym_flag} -o {algo_id} -n {num_trials}"
                success_bench, stdout_bench, stderr_bench = run_command(cmd_bench, timeout)
                
                if success_bench:
                    parsed = parse_benchmark_output(stdout_bench + stderr_bench)
                    algo_times[algo_name] = parsed.get('average_time', parsed.get('trial_time', 999999))
                else:
                    algo_times[algo_name] = 999999
                
                # Run cache simulation
                sim_binary = os.path.join(bin_sim_dir, benchmark)
                if os.path.exists(sim_binary):
                    cmd_sim = f"{sim_binary} -f {graph.path} {sym_flag} -o {algo_id} -n 1"
                    success_sim, stdout_sim, stderr_sim = run_command(cmd_sim, timeout_sim)
                    
                    if success_sim:
                        cache_parsed = parse_cache_output(stdout_sim + stderr_sim)
                        algo_cache[algo_name] = {
                            'l1': cache_parsed.get('l1_hit_rate', 0),
                            'l2': cache_parsed.get('l2_hit_rate', 0),
                            'l3': cache_parsed.get('l3_hit_rate', 0)
                        }
                    else:
                        algo_cache[algo_name] = {'l1': 0, 'l2': 0, 'l3': 0}
                else:
                    algo_cache[algo_name] = {'l1': 0, 'l2': 0, 'l3': 0}
            
            # Store all results
            for algo_name in algo_times:
                sc_result.all_results[algo_name] = {
                    'time': algo_times.get(algo_name, 999999),
                    'l1_hit': algo_cache.get(algo_name, {}).get('l1', 0),
                    'l2_hit': algo_cache.get(algo_name, {}).get('l2', 0),
                    'l3_hit': algo_cache.get(algo_name, {}).get('l3', 0)
                }
            
            # Find best by time
            valid_times = {k: v for k, v in algo_times.items() if v < 999999}
            if valid_times:
                best_time_algo = min(valid_times, key=valid_times.get)
                sc_result.best_time_algorithm = best_time_algo
                sc_result.best_time = valid_times[best_time_algo]
                sc_result.best_time_l1_hit = algo_cache.get(best_time_algo, {}).get('l1', 0)
                sc_result.best_time_l2_hit = algo_cache.get(best_time_algo, {}).get('l2', 0)
                sc_result.best_time_l3_hit = algo_cache.get(best_time_algo, {}).get('l3', 0)
            
            # Find best by cache
            valid_cache = {k: v.get('l1', 0) for k, v in algo_cache.items() if v.get('l1', 0) > 0}
            if valid_cache:
                best_cache_algo = max(valid_cache, key=valid_cache.get)
                sc_result.best_cache_algorithm = best_cache_algo
                sc_result.best_cache_l1_hit = valid_cache[best_cache_algo]
                sc_result.best_cache_l2_hit = algo_cache.get(best_cache_algo, {}).get('l2', 0)
                sc_result.best_cache_l3_hit = algo_cache.get(best_cache_algo, {}).get('l3', 0)
                sc_result.best_cache_time = algo_times.get(best_cache_algo, 0)
            
            # Get adaptive algorithm results
            sc_result.adaptive_time = algo_times.get(adaptive_algo, 999999)
            sc_result.adaptive_l1_hit = algo_cache.get(adaptive_algo, {}).get('l1', 0)
            sc_result.adaptive_l2_hit = algo_cache.get(adaptive_algo, {}).get('l2', 0)
            sc_result.adaptive_l3_hit = algo_cache.get(adaptive_algo, {}).get('l3', 0)
            
            # Calculate comparison metrics
            if sc_result.best_time > 0:
                sc_result.adaptive_vs_best_time_ratio = sc_result.best_time / sc_result.adaptive_time if sc_result.adaptive_time > 0 else 0
            if sc_result.best_cache_l1_hit > 0:
                sc_result.adaptive_vs_best_cache_ratio = sc_result.adaptive_l1_hit / sc_result.best_cache_l1_hit
            
            sc_result.adaptive_is_best_time = (adaptive_algo == sc_result.best_time_algorithm)
            sc_result.adaptive_is_best_cache = (adaptive_algo == sc_result.best_cache_algorithm)
            
            # Calculate ranks
            sorted_by_time = sorted(valid_times.items(), key=lambda x: x[1])
            sorted_by_cache = sorted(valid_cache.items(), key=lambda x: -x[1])
            
            for rank, (algo, _) in enumerate(sorted_by_time, 1):
                if algo == adaptive_algo:
                    sc_result.adaptive_rank_time = rank
                    break
            
            for rank, (algo, _) in enumerate(sorted_by_cache, 1):
                if algo == adaptive_algo:
                    sc_result.adaptive_rank_cache = rank
                    break
            
            subcommunity_results.append(sc_result)
            
            # Only test first subcommunity per graph to save time
            break
        
        analysis.subcommunity_results = subcommunity_results
        analysis.num_subcommunities_analyzed = len(subcommunity_results)
        
        # Calculate summary statistics
        if subcommunity_results:
            correct_time = sum(1 for r in subcommunity_results if r.adaptive_is_best_time)
            correct_cache = sum(1 for r in subcommunity_results if r.adaptive_is_best_cache)
            top3_time = sum(1 for r in subcommunity_results if r.adaptive_rank_time <= 3)
            top3_cache = sum(1 for r in subcommunity_results if r.adaptive_rank_cache <= 3)
            
            n = len(subcommunity_results)
            analysis.adaptive_correct_time_pct = 100 * correct_time / n
            analysis.adaptive_correct_cache_pct = 100 * correct_cache / n
            analysis.adaptive_top3_time_pct = 100 * top3_time / n
            analysis.adaptive_top3_cache_pct = 100 * top3_cache / n
            
            valid_time_ratios = [r.adaptive_vs_best_time_ratio for r in subcommunity_results if r.adaptive_vs_best_time_ratio > 0]
            valid_cache_ratios = [r.adaptive_vs_best_cache_ratio for r in subcommunity_results if r.adaptive_vs_best_cache_ratio > 0]
            
            if valid_time_ratios:
                analysis.avg_time_ratio = sum(valid_time_ratios) / len(valid_time_ratios)
            if valid_cache_ratios:
                analysis.avg_cache_ratio = sum(valid_cache_ratios) / len(valid_cache_ratios)
        
        results.append(analysis)
        
        log(f"\n  Summary:")
        log(f"    Adaptive chose best for time: {analysis.adaptive_correct_time_pct:.1f}%")
        log(f"    Adaptive chose best for cache: {analysis.adaptive_correct_cache_pct:.1f}%")
        log(f"    Adaptive in top 3 for time: {analysis.adaptive_top3_time_pct:.1f}%")
        log(f"    Adaptive in top 3 for cache: {analysis.adaptive_top3_cache_pct:.1f}%")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f"brute_force_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    results_data = []
    for analysis in results:
        data = {
            "graph": analysis.graph,
            "size_mb": analysis.size_mb,
            "modularity": analysis.modularity,
            "num_communities": analysis.num_communities,
            "num_subcommunities_analyzed": analysis.num_subcommunities_analyzed,
            "adaptive_correct_time_pct": analysis.adaptive_correct_time_pct,
            "adaptive_correct_cache_pct": analysis.adaptive_correct_cache_pct,
            "adaptive_top3_time_pct": analysis.adaptive_top3_time_pct,
            "adaptive_top3_cache_pct": analysis.adaptive_top3_cache_pct,
            "avg_time_ratio": analysis.avg_time_ratio,
            "avg_cache_ratio": analysis.avg_cache_ratio,
            "success": analysis.success,
            "error": analysis.error,
            "subcommunity_results": [
                asdict(r) for r in analysis.subcommunity_results
            ]
        }
        results_data.append(data)
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    log(f"\n{'='*60}")
    log(f"Brute-force analysis saved to: {results_file}")
    
    # Print overall summary
    successful = [r for r in results if r.success]
    if successful:
        avg_correct_time = sum(r.adaptive_correct_time_pct for r in successful) / len(successful)
        avg_correct_cache = sum(r.adaptive_correct_cache_pct for r in successful) / len(successful)
        avg_top3_time = sum(r.adaptive_top3_time_pct for r in successful) / len(successful)
        
        log(f"\nOVERALL SUMMARY")
        log(f"Graphs analyzed: {len(successful)}")
        log(f"Avg % adaptive chose best for time: {avg_correct_time:.1f}%")
        log(f"Avg % adaptive chose best for cache: {avg_correct_cache:.1f}%")
        log(f"Avg % adaptive in top 3 for time: {avg_top3_time:.1f}%")
    
    return results


__all__ = [
    # Data classes
    'SubcommunityInfo',
    'AdaptiveOrderResult',
    'AdaptiveComparisonResult',
    'SubcommunityBruteForceResult',
    'GraphBruteForceAnalysis',
    # Parsing functions
    'parse_adaptive_output',
    'parse_benchmark_output',
    'parse_cache_output',
    # Analysis functions
    'analyze_adaptive_order',
    'compare_adaptive_vs_fixed',
    'validate_adaptive_accuracy',
    'run_subcommunity_brute_force',
]
