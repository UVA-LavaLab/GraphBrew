#!/usr/bin/env python3
"""
Reordering utilities for GraphBrew.

Generates vertex reorderings (label mappings) for graphs using various algorithms.
Can be used standalone or as a library.

Standalone usage:
    python -m scripts.lib.reorder --graph graphs/email-Enron/email-Enron.mtx
    python -m scripts.lib.reorder --graph test.mtx --algorithms 0,8,9 --output results/mappings
    python -m scripts.lib.reorder --graph test.mtx --expand-variants

Library usage:
    from scripts.lib.reorder import generate_reorderings, generate_label_maps
    
    results = generate_reorderings(graphs, algorithms=[0, 8, 9], bin_dir="bench/bin")
    maps, times = generate_label_maps(graphs, algorithms, output_dir="results")
"""

import os
import time
import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from .utils import (
    PROJECT_ROOT, BIN_DIR, RESULTS_DIR,
    ALGORITHMS, ALGORITHM_IDS, SLOW_ALGORITHMS, SIZE_MEDIUM,
    LEIDEN_CSR_VARIANTS,
    LEIDEN_DEFAULT_RESOLUTION, LEIDEN_DEFAULT_PASSES,
    LEIDEN_MODULARITY_MAX_ITERATIONS,
    RABBITORDER_VARIANTS, RABBITORDER_DEFAULT_VARIANT,
    GRAPHBREW_VARIANTS, GRAPHBREW_DEFAULT_VARIANT,
    TIMEOUT_REORDER,
    Logger, run_command, get_timestamp,
)

# Initialize logger
log = Logger()

# =============================================================================
# Constants
# =============================================================================

# Enable run logging (saves command outputs per graph)
ENABLE_RUN_LOGGING = True

# Backward-compat: old bare algorithm names → new variant-suffixed names.
# Before the variant-everywhere change, GraphBrewOrder was stored without a
# variant suffix.  Old label-map indices and .lo files still use the bare
# name so we need fallback lookups.
_LEGACY_ALGO_NAMES = {
    'GraphBrewOrder': 'GraphBrewOrder_leiden',
}
_LEGACY_ALGO_NAMES_REV = {v: k for k, v in _LEGACY_ALGO_NAMES.items()}


def safe_filename(name: str) -> str:
    """Sanitize algorithm name for use in filenames.
    
    Colons in filenames (e.g., LeidenCSR_vibe:rabbit.lo) break the C++ CLI
    parser which splits -o arguments on ALL colons. Replace colons with
    underscores to produce safe filenames (e.g., LeidenCSR_vibe_rabbit.lo).
    """
    return name.replace(':', '_')
# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ReorderResult:
    """Result from reordering/label map generation."""
    graph: str
    algorithm_id: int
    algorithm_name: str
    reorder_time: float
    mapping_file: str = ""
    success: bool = True
    error: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AlgorithmConfig:
    """Configuration for an algorithm, including variant support."""
    algo_id: int           # Base algorithm ID (e.g., 16 for LeidenCSR)
    name: str              # Display name (e.g., "LeidenCSR_vibe")
    option_string: str     # Full option string for -o flag (e.g., "16:vibe:quality")
    variant: str = ""      # Variant name if applicable (e.g., "vibe")
    resolution: str = "auto"  # Resolution: "auto", "dynamic", "1.0", etc.
    passes: int = LEIDEN_DEFAULT_PASSES
    
    @property
    def base_name(self) -> str:
        """Get base algorithm name without variant suffix."""
        return ALGORITHMS.get(self.algo_id, f"ALGO_{self.algo_id}")


@dataclass
class GraphInfo:
    """Basic information about a graph for reordering."""
    name: str
    path: str
    size_mb: float = 0.0
    is_symmetric: bool = True
    nodes: int = 0
    edges: int = 0


def get_algorithm_name_with_variant(algo_id: int, variant: str = None) -> str:
    """
    Get algorithm name with variant suffix for algorithms that have variants.
    
    ALWAYS includes variant in name for:
    - RABBITORDER (8): RABBITORDER_csr or RABBITORDER_boost
    - GraphBrewOrder (12): GraphBrewOrder_leiden (default), GraphBrewOrder_gve, etc.
    - LeidenCSR (16): LeidenCSR_vibe, LeidenCSR_vibe:hrab, etc.
    
    For other algorithms, returns the base name.
    
    Args:
        algo_id: Algorithm ID
        variant: Variant name (optional, uses default if not provided)
        
    Returns:
        Algorithm name with variant suffix where applicable
    """
    base_name = ALGORITHMS.get(algo_id, f"ALGO_{algo_id}")
    
    if algo_id == 8:  # RABBITORDER
        variant = variant or RABBITORDER_DEFAULT_VARIANT
        return f"RABBITORDER_{variant}"
    elif algo_id == 12:  # GraphBrewOrder
        variant = variant or GRAPHBREW_DEFAULT_VARIANT
        return f"GraphBrewOrder_{variant}"
    elif algo_id == 16:  # LeidenCSR
        variant = variant or LEIDEN_CSR_VARIANTS[0]  # Default: vibe
        return f"LeidenCSR_{variant}"
    else:
        return base_name


def _find_legacy_map_file(graph_mappings_dir: str, algo_name: str) -> Optional[str]:
    """Check for a .lo file under the legacy (bare) algorithm name.
    
    When algo_name is 'GraphBrewOrder_leiden', also checks for
    'GraphBrewOrder.lo' which may exist from older runs.
    Returns the path if found, None otherwise.
    """
    legacy_name = _LEGACY_ALGO_NAMES_REV.get(algo_name)
    if not legacy_name:
        return None
    legacy_file = os.path.join(graph_mappings_dir, f"{safe_filename(legacy_name)}.lo")
    if os.path.exists(legacy_file):
        return legacy_file
    return None


# =============================================================================
# Algorithm Configuration
# =============================================================================

def expand_algorithms_with_variants(
    algorithms: List[int],
    expand_leiden_variants: bool = False,
    leiden_resolution: str = LEIDEN_DEFAULT_RESOLUTION,
    leiden_passes: int = LEIDEN_DEFAULT_PASSES,
    leiden_csr_variants: List[str] = None,
    rabbit_variants: List[str] = None,
    graphbrew_variants: List[str] = None
) -> List[AlgorithmConfig]:
    """
    Expand algorithm IDs into AlgorithmConfig objects.
    
    For Leiden algorithm (16), optionally expand into its variants.
    For RabbitOrder (8), optionally expand into csr/boost variants.
    For GraphBrewOrder (12), optionally expand into leiden/gve/gveopt/rabbit/hubcluster variants.
    
    Args:
        algorithms: List of algorithm IDs
        expand_leiden_variants: If True, expand LeidenCSR into variants
        leiden_resolution: Resolution parameter for Leiden algorithms
        leiden_passes: Number of passes for LeidenCSR
        leiden_csr_variants: Which LeidenCSR variants to include (default: all)
        rabbit_variants: Which RabbitOrder variants to include (default: csr only)
        graphbrew_variants: Which GraphBrewOrder variants to include (default: leiden only)
    
    Returns:
        List of AlgorithmConfig objects
    """
    if leiden_csr_variants is None:
        leiden_csr_variants = LEIDEN_CSR_VARIANTS
    if rabbit_variants is None:
        # When expanding variants, include both RabbitOrder variants; otherwise just csr
        rabbit_variants = RABBITORDER_VARIANTS if expand_leiden_variants else [RABBITORDER_DEFAULT_VARIANT]
    if graphbrew_variants is None:
        # When expanding variants, include all GraphBrewOrder variants; otherwise just leiden
        graphbrew_variants = GRAPHBREW_VARIANTS if expand_leiden_variants else [GRAPHBREW_DEFAULT_VARIANT]
    
    configs = []
    
    for algo_id in algorithms:
        base_name = ALGORITHMS.get(algo_id, f"ALGO_{algo_id}")
        
        if algo_id == 16 and expand_leiden_variants:
            # LeidenCSR: expand into variants
            # Format: 16:variant:resolution:iterations:passes
            for variant in leiden_csr_variants:
                max_iterations = LEIDEN_MODULARITY_MAX_ITERATIONS
                option_str = f"{algo_id}:{variant}:{leiden_resolution}:{max_iterations}:{leiden_passes}"
                configs.append(AlgorithmConfig(
                    algo_id=algo_id,
                    name=f"LeidenCSR_{variant}",
                    option_string=option_str,
                    variant=variant,
                    resolution=leiden_resolution,
                    passes=leiden_passes
                ))
        elif algo_id == 8 and expand_leiden_variants and len(rabbit_variants) > 1:
            # RabbitOrder: expand into variants if multiple specified
            for variant in rabbit_variants:
                option_str = f"{algo_id}:{variant}"
                configs.append(AlgorithmConfig(
                    algo_id=algo_id,
                    name=f"RABBITORDER_{variant}",
                    option_string=option_str,
                    variant=variant
                ))
        elif algo_id == 8:
            # RabbitOrder: use specified variant (default: csr) - ALWAYS include variant in name
            variant = rabbit_variants[0] if rabbit_variants else RABBITORDER_DEFAULT_VARIANT
            option_str = f"{algo_id}:{variant}"
            configs.append(AlgorithmConfig(
                algo_id=algo_id,
                name=f"RABBITORDER_{variant}",  # Always show variant
                option_string=option_str,
                variant=variant
            ))
        elif algo_id == 16:
            # LeidenCSR: use default variant when not expanding - ALWAYS include variant in name
            variant = leiden_csr_variants[0] if leiden_csr_variants else LEIDEN_CSR_VARIANTS[0]
            max_iterations = LEIDEN_MODULARITY_MAX_ITERATIONS
            option_str = f"{algo_id}:{variant}:{leiden_resolution}:{max_iterations}:{leiden_passes}"
            configs.append(AlgorithmConfig(
                algo_id=algo_id,
                name=f"LeidenCSR_{variant}",  # Always show variant
                option_string=option_str,
                variant=variant,
                resolution=leiden_resolution,
                passes=leiden_passes
            ))
        elif algo_id == 15:
            # LeidenOrder: just resolution
            option_str = f"{algo_id}:{leiden_resolution}"
            configs.append(AlgorithmConfig(
                algo_id=algo_id,
                name=base_name,
                option_string=option_str,
                resolution=leiden_resolution
            ))
        elif algo_id == 12 and expand_leiden_variants:
            # GraphBrewOrder: expand into clustering variants
            # Format: 12:variant
            for variant in graphbrew_variants:
                option_str = f"{algo_id}:{variant}"
                configs.append(AlgorithmConfig(
                    algo_id=algo_id,
                    name=f"GraphBrewOrder_{variant}",
                    option_string=option_str,
                    variant=variant,
                    resolution=leiden_resolution
                ))
        elif algo_id == 12:
            # GraphBrewOrder: use default variant (leiden) - ALWAYS include variant in name
            variant = graphbrew_variants[0] if graphbrew_variants else GRAPHBREW_DEFAULT_VARIANT
            option_str = f"{algo_id}:{variant}"
            configs.append(AlgorithmConfig(
                algo_id=algo_id,
                name=f"GraphBrewOrder_{variant}",  # Always show variant
                option_string=option_str,
                variant=variant,
                resolution=leiden_resolution
            ))
        else:
            # Non-Leiden algorithms: just use ID
            configs.append(AlgorithmConfig(
                algo_id=algo_id,
                name=base_name,
                option_string=str(algo_id)
            ))
    
    return configs


# =============================================================================
# Output Parsing
# =============================================================================

def parse_reorder_time_from_converter(output: str) -> Optional[float]:
    """
    Parse the actual reordering algorithm time from converter output.
    
    FAIR TIMING PHILOSOPHY:
    - INCLUDE: Actual algorithm work (community detection, ordering generation)
    - EXCLUDE: Data structure conversion overhead (library-specific preprocessing)
    
    Why exclude conversion overhead?
    External libraries (RabbitOrder, GOrder, GVE-Leiden) require their own
    data structures. If we had native CSR implementations, there would be no
    conversion. For fair comparison, we only measure the actual ordering algorithm.
    
    CONVERSION OVERHEAD TO EXCLUDE:
    - "DiGraph Build Time:"     - CSR → GVE-Leiden DiGraph (LeidenOrder)
    - "DiGraph graph:"          - CSR → DiGraph (LeidenOrder - legacy naming)
    - "GOrder graph:"           - CSR → GOrder internal format
    - "Sort Map Time:" + first "Relabel Map Time:" - RabbitOrder preprocessing
    
    ALGORITHM TIME TO INCLUDE:
    - "Leiden Time:" + "Ordering Time:"           - Legacy Leiden algorithm
    - "LeidenOrder Map Time:" + "GenID Time:"     - LeidenOrder algorithm
    - "LeidenCSR Community Detection/Ordering:"   - LeidenCSR (native CSR, fast)
    - "GOrder Map Time:"                          - GOrder actual ordering
    - "RabbitOrder Map Time:"                     - RabbitOrder actual ordering
    - "*Map Time:" for native CSR algorithms      - HubSort, DBG, Sort, Random, etc.
    
    Returns the reorder time in seconds, or None if not found.
    """
    # =========================================================================
    # STRATEGY: Parse detailed component times and sum only algorithm work
    # =========================================================================
    
    # Helper to parse a time value from a pattern
    def get_time(pattern: str) -> Optional[float]:
        match = re.search(pattern, output, re.MULTILINE)
        return float(match.group(1)) if match else None
    
    # -------------------------------------------------------------------------
    # 1. Legacy Leiden: Leiden Time + Ordering Time (exclude DiGraph Build)
    # -------------------------------------------------------------------------
    leiden_time = get_time(r'^Leiden Time:\s*([\d.]+)')
    ordering_time = get_time(r'^Ordering Time:\s*([\d.]+)')
    
    if leiden_time is not None and ordering_time is not None:
        # Legacy Leiden detected: sum algorithm parts only
        return leiden_time + ordering_time
    
    # -------------------------------------------------------------------------
    # 2. LeidenOrder (legacy): LeidenOrder Map Time + GenID Time (exclude DiGraph graph)
    # -------------------------------------------------------------------------
    leiden_order_time = get_time(r'^LeidenOrder Map Time:\s*([\d.]+)')
    genid_time = get_time(r'^GenID Time:\s*([\d.]+)')
    
    if leiden_order_time is not None and genid_time is not None:
        # LeidenOrder detected: sum algorithm parts only
        return leiden_order_time + genid_time
    
    # -------------------------------------------------------------------------
    # 3. LeidenCSR: Community Detection + Ordering (native CSR, all included)
    # -------------------------------------------------------------------------
    leiden_csr_community = get_time(r'^LeidenCSR Community Detection:\s*([\d.]+)')
    leiden_csr_ordering = get_time(r'^LeidenCSR Ordering:\s*([\d.]+)')
    
    if leiden_csr_community is not None and leiden_csr_ordering is not None:
        return leiden_csr_community + leiden_csr_ordering
    
    # -------------------------------------------------------------------------
    # 4. GOrder: Only GOrder Map Time (exclude "GOrder graph:" build time)
    # -------------------------------------------------------------------------
    gorder_map_time = get_time(r'^GOrder Map Time:\s*([\d.]+)')
    if gorder_map_time is not None:
        return gorder_map_time
    
    # -------------------------------------------------------------------------
    # 5. RabbitOrder: Only RabbitOrder Map Time (exclude Sort + Relabel prep)
    # -------------------------------------------------------------------------
    rabbit_map_time = get_time(r'^RabbitOrder Map Time:\s*([\d.]+)')
    if rabbit_map_time is not None:
        return rabbit_map_time
    
    # -------------------------------------------------------------------------
    # 5. Native CSR algorithms: Use their Map Time directly (no conversion)
    #    HubSort, HubCluster, DBG, Sort, Random, COrder, RCMOrder, etc.
    # -------------------------------------------------------------------------
    # Standard Map Time pattern
    pattern = r'^([A-Za-z]+)\s+Map Time:\s*([\d.]+)'
    matches = re.findall(pattern, output, re.MULTILINE)
    
    # Filter out intermediate steps (Relabel, Sort, Total) and get actual algorithm time
    # 'total' excluded: RabbitOrderCSR outputs "Total Map Time: 0.00000" as a statistic
    excluded_names = {'relabel', 'sort', 'gorder', 'total'}  # Already handled above
    valid_times = [
        (name, float(t)) for name, t in matches 
        if name.lower() not in excluded_names
    ]
    
    if valid_times:
        # Return the last valid algorithm time
        return valid_times[-1][1]
    
    # -------------------------------------------------------------------------
    # FALLBACK: Unified "Reorder Time:" (total wall clock - may include overhead)
    # Only use if no detailed component parsing succeeded
    # -------------------------------------------------------------------------
    reorder_time_match = get_time(r'^Reorder Time:\s*([\d.]+)')
    if reorder_time_match is not None:
        return reorder_time_match
    
    return None


# =============================================================================
# Core Reordering Functions
# =============================================================================

def generate_reorderings(
    graphs: List[GraphInfo],
    algorithms: List[int],
    bin_dir: str = None,
    output_dir: str = None,
    timeout: int = TIMEOUT_REORDER,
    skip_slow: bool = False,
    generate_maps: bool = True,
    force_reorder: bool = False
) -> List[ReorderResult]:
    """
    Generate reorderings for all graphs and algorithms.
    Records reorder time for each combination.
    
    Args:
        graphs: List of graphs to process
        algorithms: List of algorithm IDs to use
        bin_dir: Directory containing binaries (default: bench/bin)
        output_dir: Directory for outputs (default: results)
        timeout: Timeout for each reordering
        skip_slow: Skip slow algorithms on large graphs
        generate_maps: If True, generate .lo mapping files
        force_reorder: If True, regenerate even if .lo/.time files exist
        
    Returns:
        List of ReorderResult with timing information
    """
    if bin_dir is None:
        bin_dir = str(BIN_DIR)
    if output_dir is None:
        output_dir = str(RESULTS_DIR)
    
    log.info(f"Generating reorderings for {len(graphs)} graphs × {len(algorithms)} algorithms")
    if force_reorder:
        log.info("Force reorder enabled - will regenerate all reorderings")
    
    results = []
    total = len(graphs) * len(algorithms)
    current = 0
    
    # Create output directory for mappings
    mappings_dir = os.path.join(output_dir, "mappings")
    os.makedirs(mappings_dir, exist_ok=True)
    
    for graph_idx, graph in enumerate(graphs, 1):
        log.info(f"Graph [{graph_idx}/{len(graphs)}]: {graph.name} ({graph.size_mb:.1f}MB)")
        
        # Create per-graph mappings directory
        graph_mappings_dir = os.path.join(mappings_dir, graph.name)
        if generate_maps:
            os.makedirs(graph_mappings_dir, exist_ok=True)
        
        for algo_id in algorithms:
            current += 1
            # Always include variant in name for algorithms that have variants
            algo_name = get_algorithm_name_with_variant(algo_id)
            
            # Skip slow algorithms on large graphs if requested
            if skip_slow and algo_id in SLOW_ALGORITHMS and graph.size_mb > SIZE_MEDIUM:
                log.info(f"  [{current}/{total}] {algo_name}: SKIPPED (slow on large graphs)")
                results.append(ReorderResult(
                    graph=graph.name,
                    algorithm_id=algo_id,
                    algorithm_name=algo_name,
                    reorder_time=0.0,
                    success=False,
                    error="SKIPPED"
                ))
                continue
            
            # ORIGINAL doesn't need reordering
            if algo_id == 0:
                log.info(f"  [{current}/{total}] {algo_name}: 0.0000s (no reorder)")
                results.append(ReorderResult(
                    graph=graph.name,
                    algorithm_id=algo_id,
                    algorithm_name=algo_name,
                    reorder_time=0.0,
                    success=True
                ))
                continue
            
            # Output mapping file path
            safe_name = safe_filename(algo_name)
            map_file = os.path.join(graph_mappings_dir, f"{safe_name}.lo") if generate_maps else None
            
            # Check if mapping already exists (unless force_reorder is set)
            # Also check legacy filename (e.g. GraphBrewOrder.lo → GraphBrewOrder_leiden.lo)
            actual_map_file = map_file
            if generate_maps and map_file and not os.path.exists(map_file) and not force_reorder:
                legacy = _find_legacy_map_file(graph_mappings_dir, algo_name)
                if legacy:
                    actual_map_file = legacy
            
            if generate_maps and actual_map_file and os.path.exists(actual_map_file) and not force_reorder:
                timing_file = os.path.join(graph_mappings_dir, f"{safe_name}.time")
                # Also try legacy timing file
                if not os.path.exists(timing_file):
                    legacy_name = _LEGACY_ALGO_NAMES_REV.get(algo_name)
                    if legacy_name:
                        timing_file = os.path.join(graph_mappings_dir, f"{safe_filename(legacy_name)}.time")
                if os.path.exists(timing_file):
                    with open(timing_file) as f:
                        reorder_time = float(f.read().strip())
                else:
                    reorder_time = 0.0
                
                log.info(f"  [{current}/{total}] {algo_name}: exists ({reorder_time:.4f}s)")
                results.append(ReorderResult(
                    graph=graph.name,
                    algorithm_id=algo_id,
                    algorithm_name=algo_name,
                    reorder_time=reorder_time,
                    mapping_file=actual_map_file,
                    success=True
                ))
                continue
            
            # Remove existing files if force_reorder
            if force_reorder and map_file and os.path.exists(map_file):
                os.remove(map_file)
                timing_file = os.path.join(graph_mappings_dir, f"{safe_name}.time")
                if os.path.exists(timing_file):
                    os.remove(timing_file)
            
            # Generate mapping with converter
            if generate_maps:
                binary = os.path.join(bin_dir, "converter")
                sym_flag = "-s" if graph.is_symmetric else ""
                cmd = f"{binary} -f {graph.path} {sym_flag} -o {algo_id} -q {map_file}"
            else:
                binary = os.path.join(bin_dir, "pr")
                sym_flag = "-s" if graph.is_symmetric else ""
                cmd = f"{binary} -f {graph.path} {sym_flag} -o {algo_id} -n 1"
            
            # Run and parse
            start_time = time.time()
            success, stdout, stderr = run_command(cmd, timeout)
            elapsed = time.time() - start_time
            
            # Save run log
            if ENABLE_RUN_LOGGING:
                try:
                    from .graph_data import save_run_log
                    save_run_log(
                        graph_name=graph.name,
                        operation='reorder',
                        algorithm=algo_name,
                        output=stdout + "\n--- STDERR ---\n" + stderr if stderr else stdout,
                        command=cmd,
                        exit_code=0 if success else 1,
                        duration=elapsed
                    )
                except Exception as e:
                    log.debug(f"Failed to save run log: {e}")
            
            if success:
                output = stdout + stderr
                
                if generate_maps:
                    if os.path.exists(map_file):
                        actual_reorder_time = parse_reorder_time_from_converter(output)
                        reorder_time = actual_reorder_time if actual_reorder_time else elapsed
                        
                        timing_file = os.path.join(graph_mappings_dir, f"{safe_name}.time")
                        with open(timing_file, 'w') as f:
                            f.write(f"{reorder_time:.6f}")
                        
                        log.info(f"  [{current}/{total}] {algo_name}: {reorder_time:.4f}s (map: {safe_name}.lo)")
                        results.append(ReorderResult(
                            graph=graph.name,
                            algorithm_id=algo_id,
                            algorithm_name=algo_name,
                            reorder_time=reorder_time,
                            mapping_file=map_file,
                            success=True
                        ))
                    else:
                        log.error(f"  [{current}/{total}] {algo_name}: FAILED (no map file)")
                        results.append(ReorderResult(
                            graph=graph.name,
                            algorithm_id=algo_id,
                            algorithm_name=algo_name,
                            reorder_time=elapsed,
                            success=False,
                            error="Map file not created"
                        ))
                else:
                    log.info(f"  [{current}/{total}] {algo_name}: {elapsed:.4f}s")
                    results.append(ReorderResult(
                        graph=graph.name,
                        algorithm_id=algo_id,
                        algorithm_name=algo_name,
                        reorder_time=elapsed,
                        success=True
                    ))
            else:
                error = "TIMEOUT" if "TIMEOUT" in stderr else stderr[:100]
                log.error(f"  [{current}/{total}] {algo_name}: FAILED ({error})")
                results.append(ReorderResult(
                    graph=graph.name,
                    algorithm_id=algo_id,
                    algorithm_name=algo_name,
                    reorder_time=0.0,
                    success=False,
                    error=error
                ))
    
    return results


def generate_label_maps(
    graphs: List[GraphInfo],
    algorithms: List[int],
    bin_dir: str = None,
    output_dir: str = None,
    timeout: int = TIMEOUT_REORDER,
    skip_slow: bool = False
) -> Tuple[Dict[str, Dict[str, str]], List[ReorderResult]]:
    """
    Pre-generate label.map files for each graph/algorithm combination.
    Also records reorder times during generation.
    
    Args:
        graphs: List of graphs to process
        algorithms: List of algorithm IDs to use
        bin_dir: Directory containing binaries
        output_dir: Directory for outputs
        timeout: Timeout for each reordering
        skip_slow: Skip slow algorithms on large graphs
        
    Returns:
        Tuple of:
        - Dictionary mapping (graph, algorithm) to label map file path
        - List of ReorderResult with timing information
    """
    if bin_dir is None:
        bin_dir = str(BIN_DIR)
    if output_dir is None:
        output_dir = str(RESULTS_DIR)
    
    log.info(f"Pre-generating label maps for {len(graphs)} graphs")
    
    # Create mappings directory
    mappings_dir = os.path.join(output_dir, "mappings")
    os.makedirs(mappings_dir, exist_ok=True)
    
    label_maps = {}
    reorder_results = []
    total = len(graphs) * len(algorithms)
    current = 0
    
    for graph_idx, graph in enumerate(graphs, 1):
        log.info(f"Graph [{graph_idx}/{len(graphs)}]: {graph.name} ({graph.size_mb:.1f}MB)")
        label_maps[graph.name] = {}
        graph_mappings_dir = os.path.join(mappings_dir, graph.name)
        os.makedirs(graph_mappings_dir, exist_ok=True)
        
        for algo_id in algorithms:
            current += 1
            # Always include variant in name for algorithms that have variants
            algo_name = get_algorithm_name_with_variant(algo_id)
            
            # Skip ORIGINAL (no mapping needed)
            if algo_id == 0:
                log.info(f"  [{current}/{total}] {algo_name}: no map needed (0.0000s)")
                reorder_results.append(ReorderResult(
                    graph=graph.name,
                    algorithm_id=algo_id,
                    algorithm_name=algo_name,
                    reorder_time=0.0,
                    mapping_file="",
                    success=True
                ))
                continue
            
            # Skip slow algorithms on large graphs if requested
            if skip_slow and algo_id in SLOW_ALGORITHMS and graph.size_mb > SIZE_MEDIUM:
                log.info(f"  [{current}/{total}] {algo_name}: SKIPPED (slow)")
                reorder_results.append(ReorderResult(
                    graph=graph.name,
                    algorithm_id=algo_id,
                    algorithm_name=algo_name,
                    reorder_time=0.0,
                    mapping_file="",
                    success=False,
                    error="SKIPPED"
                ))
                continue
            
            # Output mapping file path
            safe_name = safe_filename(algo_name)
            map_file = os.path.join(graph_mappings_dir, f"{safe_name}.lo")
            timing_file = os.path.join(graph_mappings_dir, f"{safe_name}.time")
            
            # Check if already exists (also check legacy filename)
            actual_map_file = map_file
            if not os.path.exists(map_file):
                legacy = _find_legacy_map_file(graph_mappings_dir, algo_name)
                if legacy:
                    actual_map_file = legacy
            
            if os.path.exists(actual_map_file):
                actual_timing = timing_file
                if not os.path.exists(timing_file):
                    legacy_name = _LEGACY_ALGO_NAMES_REV.get(algo_name)
                    if legacy_name:
                        legacy_tf = os.path.join(graph_mappings_dir, f"{safe_filename(legacy_name)}.time")
                        if os.path.exists(legacy_tf):
                            actual_timing = legacy_tf
                if os.path.exists(actual_timing):
                    with open(actual_timing) as f:
                        reorder_time = float(f.read().strip())
                else:
                    reorder_time = 0.0
                
                log.info(f"  [{current}/{total}] {algo_name}: exists ({reorder_time:.4f}s)")
                label_maps[graph.name][algo_name] = actual_map_file
                reorder_results.append(ReorderResult(
                    graph=graph.name,
                    algorithm_id=algo_id,
                    algorithm_name=algo_name,
                    reorder_time=reorder_time,
                    mapping_file=actual_map_file,
                    success=True
                ))
                continue
            
            # Use converter to generate mapping
            binary = os.path.join(bin_dir, "converter")
            sym_flag = "-s" if graph.is_symmetric else ""
            cmd = f"{binary} -f {graph.path} {sym_flag} -o {algo_id} -q {map_file}"
            
            start_time = time.time()
            success, stdout, stderr = run_command(cmd, timeout)
            elapsed = time.time() - start_time
            
            # Save run log
            if ENABLE_RUN_LOGGING:
                try:
                    from .graph_data import save_run_log
                    save_run_log(
                        graph_name=graph.name,
                        operation='reorder',
                        algorithm=algo_name,
                        output=stdout + "\n--- STDERR ---\n" + stderr if stderr else stdout,
                        command=cmd,
                        exit_code=0 if success else 1,
                        duration=elapsed
                    )
                except Exception as e:
                    log.debug(f"Failed to save run log: {e}")
            
            if success and os.path.exists(map_file):
                # Save timing
                with open(timing_file, 'w') as f:
                    f.write(f"{elapsed:.6f}")
                
                log.info(f"  [{current}/{total}] {algo_name}: generated ({elapsed:.4f}s)")
                label_maps[graph.name][algo_name] = map_file
                reorder_results.append(ReorderResult(
                    graph=graph.name,
                    algorithm_id=algo_id,
                    algorithm_name=algo_name,
                    reorder_time=elapsed,
                    mapping_file=map_file,
                    success=True
                ))
            else:
                error = "TIMEOUT" if "TIMEOUT" in stderr else stderr[:50]
                log.error(f"  [{current}/{total}] {algo_name}: FAILED ({error})")
                reorder_results.append(ReorderResult(
                    graph=graph.name,
                    algorithm_id=algo_id,
                    algorithm_name=algo_name,
                    reorder_time=elapsed,
                    mapping_file="",
                    success=False,
                    error=error
                ))
    
    # Save mapping index
    index_file = os.path.join(mappings_dir, "index.json")
    with open(index_file, 'w') as f:
        json.dump(label_maps, f, indent=2)
    log.info(f"Label map index saved to: {index_file}")
    
    # Save reorder times
    timestamp = get_timestamp()
    reorder_json = os.path.join(output_dir, f"reorder_times_{timestamp}.json")
    with open(reorder_json, 'w') as f:
        json.dump([r.to_dict() for r in reorder_results], f, indent=2)
    
    return label_maps, reorder_results


def generate_reorderings_with_variants(
    graphs: List[GraphInfo],
    algorithms: List[int],
    bin_dir: str = None,
    output_dir: str = None,
    expand_leiden_variants: bool = True,
    leiden_resolution: float = LEIDEN_DEFAULT_RESOLUTION,
    leiden_passes: int = LEIDEN_DEFAULT_PASSES,
    leiden_csr_variants: List[str] = None,
    rabbit_variants: List[str] = None,
    graphbrew_variants: List[str] = None,
    timeout: int = TIMEOUT_REORDER,
    skip_slow: bool = False,
    force_reorder: bool = False
) -> Tuple[Dict[str, Dict[str, str]], List[ReorderResult]]:
    """
    Generate reorderings with Leiden variant expansion.
    
    Creates separate mappings for each variant:
        - LeidenCSR_vibe.lo
        - LeidenCSR_vibe:hrab.lo
        - GraphBrewOrder_leiden.lo
        - GraphBrewOrder_gve.lo
    
    Args:
        graphs: List of graphs to process
        algorithms: List of algorithm IDs or AlgorithmConfig objects
        bin_dir: Directory containing binaries
        output_dir: Directory for outputs
        expand_leiden_variants: If True, expand Leiden into variants
        leiden_resolution: Resolution parameter
        leiden_passes: Number of passes for LeidenCSR
        leiden_csr_variants: Which LeidenCSR variants
        graphbrew_variants: Which GraphBrewOrder variants
        timeout: Timeout for each reordering
        skip_slow: Skip slow algorithms on large graphs
        force_reorder: Regenerate even if files exist
        
    Returns:
        Tuple of (label_maps, reorder_results)
    """
    if bin_dir is None:
        bin_dir = str(BIN_DIR)
    if output_dir is None:
        output_dir = str(RESULTS_DIR)
    
    if expand_leiden_variants:
        log.info(f"Leiden variant expansion enabled")
        log.info(f"  LeidenCSR variants: {leiden_csr_variants or LEIDEN_CSR_VARIANTS}")
    
    # Handle both algorithm ID lists and pre-expanded AlgorithmConfig lists
    if algorithms and isinstance(algorithms[0], AlgorithmConfig):
        # Already expanded - use directly
        configs = algorithms
    else:
        # Expand algorithm IDs to configs
        configs = expand_algorithms_with_variants(
            algorithms,
            expand_leiden_variants=expand_leiden_variants,
            leiden_resolution=leiden_resolution,
            leiden_passes=leiden_passes,
            leiden_csr_variants=leiden_csr_variants,
            rabbit_variants=rabbit_variants,
            graphbrew_variants=graphbrew_variants
        )
    
    results = []
    label_maps = {}
    total = len(graphs) * len(configs)
    current = 0
    
    # Create mappings directory
    mappings_dir = os.path.join(output_dir, "mappings")
    os.makedirs(mappings_dir, exist_ok=True)
    
    for graph_idx, graph in enumerate(graphs, 1):
        log.info(f"Graph [{graph_idx}/{len(graphs)}]: {graph.name} ({graph.size_mb:.1f}MB)")
        label_maps[graph.name] = {}
        
        graph_mappings_dir = os.path.join(mappings_dir, graph.name)
        os.makedirs(graph_mappings_dir, exist_ok=True)
        
        for cfg in configs:
            current += 1
            
            # Skip slow algorithms on large graphs
            if skip_slow and cfg.algo_id in SLOW_ALGORITHMS and graph.size_mb > SIZE_MEDIUM:
                log.info(f"  [{current}/{total}] {cfg.name}: SKIPPED (slow)")
                results.append(ReorderResult(
                    graph=graph.name,
                    algorithm_id=cfg.algo_id,
                    algorithm_name=cfg.name,
                    reorder_time=0.0,
                    success=False,
                    error="SKIPPED"
                ))
                continue
            
            # ORIGINAL doesn't need reordering
            if cfg.algo_id == 0:
                log.info(f"  [{current}/{total}] {cfg.name}: 0.0000s (no reorder)")
                results.append(ReorderResult(
                    graph=graph.name,
                    algorithm_id=cfg.algo_id,
                    algorithm_name=cfg.name,
                    reorder_time=0.0,
                    success=True
                ))
                continue
            
            safe_name = safe_filename(cfg.name)
            map_file = os.path.join(graph_mappings_dir, f"{safe_name}.lo")
            timing_file = os.path.join(graph_mappings_dir, f"{safe_name}.time")
            
            # Check if exists (also check legacy filename for backward compat)
            actual_map_file = map_file
            if not os.path.exists(map_file) and not force_reorder:
                legacy = _find_legacy_map_file(graph_mappings_dir, cfg.name)
                if legacy:
                    actual_map_file = legacy
            
            if os.path.exists(actual_map_file) and not force_reorder:
                actual_timing = timing_file
                if not os.path.exists(timing_file):
                    legacy_name = _LEGACY_ALGO_NAMES_REV.get(cfg.name)
                    if legacy_name:
                        legacy_tf = os.path.join(graph_mappings_dir, f"{safe_filename(legacy_name)}.time")
                        if os.path.exists(legacy_tf):
                            actual_timing = legacy_tf
                if os.path.exists(actual_timing):
                    with open(actual_timing) as f:
                        reorder_time = float(f.read().strip())
                else:
                    reorder_time = 0.0
                
                log.info(f"  [{current}/{total}] {cfg.name}: exists ({reorder_time:.4f}s)")
                label_maps[graph.name][cfg.name] = actual_map_file
                results.append(ReorderResult(
                    graph=graph.name,
                    algorithm_id=cfg.algo_id,
                    algorithm_name=cfg.name,
                    reorder_time=reorder_time,
                    mapping_file=actual_map_file,
                    success=True
                ))
                continue
            
            # Remove if force_reorder
            if force_reorder:
                if os.path.exists(map_file):
                    os.remove(map_file)
                if os.path.exists(timing_file):
                    os.remove(timing_file)
            
            # Generate using full option string
            binary = os.path.join(bin_dir, "converter")
            sym_flag = "-s" if graph.is_symmetric else ""
            cmd = f"{binary} -f {graph.path} {sym_flag} -o {cfg.option_string} -q {map_file}"
            
            start_time = time.time()
            success, stdout, stderr = run_command(cmd, timeout)
            elapsed = time.time() - start_time
            
            # Save run log
            if ENABLE_RUN_LOGGING:
                try:
                    from .graph_data import save_run_log
                    save_run_log(
                        graph_name=graph.name,
                        operation='reorder',
                        algorithm=cfg.name,
                        output=stdout + "\n--- STDERR ---\n" + stderr if stderr else stdout,
                        command=cmd,
                        exit_code=0 if success else 1,
                        duration=elapsed
                    )
                except Exception as e:
                    log.debug(f"Failed to save run log: {e}")
            
            if success and os.path.exists(map_file):
                actual_time = parse_reorder_time_from_converter(stdout + stderr)
                reorder_time = actual_time if actual_time else elapsed
                
                with open(timing_file, 'w') as f:
                    f.write(f"{reorder_time:.6f}")
                
                log.info(f"  [{current}/{total}] {cfg.name}: {reorder_time:.4f}s")
                label_maps[graph.name][cfg.name] = map_file
                results.append(ReorderResult(
                    graph=graph.name,
                    algorithm_id=cfg.algo_id,
                    algorithm_name=cfg.name,
                    reorder_time=reorder_time,
                    mapping_file=map_file,
                    success=True
                ))
            else:
                error = "TIMEOUT" if "TIMEOUT" in stderr else stderr[:100]
                log.error(f"  [{current}/{total}] {cfg.name}: FAILED ({error})")
                results.append(ReorderResult(
                    graph=graph.name,
                    algorithm_id=cfg.algo_id,
                    algorithm_name=cfg.name,
                    reorder_time=0.0,
                    success=False,
                    error=error
                ))
    
    # Save index
    index_file = os.path.join(mappings_dir, "index.json")
    with open(index_file, 'w') as f:
        json.dump(label_maps, f, indent=2)
    
    return label_maps, results


def get_label_map_path(
    label_maps: Dict[str, Dict[str, str]],
    graph_name: str,
    algo_name: str
) -> Optional[str]:
    """Get the path to a pre-generated label map, if available.
    
    Supports backward compatibility: if algo_name is 'GraphBrewOrder_leiden'
    but the index only has 'GraphBrewOrder' (or vice versa), the fallback
    name is tried automatically.
    """
    if graph_name not in label_maps:
        return None
    
    graph_maps = label_maps[graph_name]
    
    # Try exact match first
    if algo_name in graph_maps:
        path = graph_maps[algo_name]
        if os.path.exists(path):
            return path
    
    # Try legacy fallback (new name → old bare name, or old → new)
    fallback = _LEGACY_ALGO_NAMES_REV.get(algo_name) or _LEGACY_ALGO_NAMES.get(algo_name)
    if fallback and fallback in graph_maps:
        path = graph_maps[fallback]
        if os.path.exists(path):
            return path
    
    return None


def load_label_maps_index(results_dir: str = None) -> Dict[str, Dict[str, str]]:
    """Load the label maps index from a previous run."""
    if results_dir is None:
        results_dir = str(RESULTS_DIR)
    index_file = os.path.join(results_dir, "mappings", "index.json")
    if os.path.exists(index_file):
        with open(index_file) as f:
            return json.load(f)
    return {}


# =============================================================================
# Standalone CLI
# =============================================================================

def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="GraphBrew Reordering Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m scripts.lib.reorder --graph graphs/email-Enron/email-Enron.mtx
    python -m scripts.lib.reorder --graph test.mtx --algorithms 0,8,9
    python -m scripts.lib.reorder --graph test.mtx --expand-variants
    python -m scripts.lib.reorder --list-algorithms
"""
    )
    
    parser.add_argument("--graph", "-g", help="Path to graph file")
    parser.add_argument("--algorithms", "-a", default="0,1,8",
                        help="Comma-separated algorithm IDs (default: 0,1,8)")
    parser.add_argument("--output", "-o", default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--expand-variants", action="store_true",
                        help="Expand Leiden algorithms into variants")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Force regeneration even if files exist")
    parser.add_argument("--skip-slow", action="store_true",
                        help="Skip slow algorithms on large graphs")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Timeout per reordering in seconds")
    parser.add_argument("--list-algorithms", action="store_true",
                        help="List available algorithms")
    
    args = parser.parse_args()
    
    if args.list_algorithms:
        print("\nAvailable Reordering Algorithms:")
        print("-" * 40)
        for algo_id, name in sorted(ALGORITHMS.items()):
            slow_marker = " (slow)" if algo_id in SLOW_ALGORITHMS else ""
            print(f"  {algo_id:2d}: {name}{slow_marker}")
        print()
        return
    
    if not args.graph:
        parser.print_help()
        return
    
    # Parse algorithms
    algo_ids = [int(x.strip()) for x in args.algorithms.split(",")]
    
    # Create GraphInfo
    from pathlib import Path
    graph_path = Path(args.graph)
    graph = GraphInfo(
        name=graph_path.stem,
        path=str(graph_path),
        size_mb=graph_path.stat().st_size / (1024 * 1024) if graph_path.exists() else 0,
        is_symmetric=True
    )
    
    # Generate reorderings
    if args.expand_variants:
        label_maps, results = generate_reorderings_with_variants(
            graphs=[graph],
            algorithms=algo_ids,
            output_dir=args.output,
            expand_leiden_variants=True,
            timeout=args.timeout,
            skip_slow=args.skip_slow,
            force_reorder=args.force
        )
    else:
        results = generate_reorderings(
            graphs=[graph],
            algorithms=algo_ids,
            output_dir=args.output,
            timeout=args.timeout,
            skip_slow=args.skip_slow,
            force_reorder=args.force
        )
    
    # Print summary
    print(f"\nGenerated {sum(1 for r in results if r.success)} reorderings")
    for r in results:
        status = "OK" if r.success else f"FAIL: {r.error}"
        print(f"  {r.algorithm_name}: {r.reorder_time:.4f}s - {status}")


if __name__ == "__main__":
    main()
