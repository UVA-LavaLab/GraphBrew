#!/usr/bin/env python3
"""
Reordering utilities for GraphBrew.

Generates vertex reorderings (label mappings) for graphs using various algorithms.
Can be used standalone or as a library.

Standalone usage:
    python -m scripts.lib.pipeline.reorder --graph graphs/email-Enron/email-Enron.mtx
    python -m scripts.lib.pipeline.reorder --graph test.mtx --algorithms 0,8,9 --output results/mappings
    python -m scripts.lib.pipeline.reorder --graph test.mtx --expand-variants

Library usage:
    from scripts.lib.pipeline.reorder import generate_reorderings, generate_label_maps
    
    results = generate_reorderings(graphs, algorithms=[0, 8, 9], bin_dir="bench/bin")
    maps, times = generate_label_maps(graphs, algorithms, output_dir="results")
"""

import os
import time
import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from ..core.utils import (
    BIN_DIR, RESULTS_DIR,
    ALGORITHMS, SLOW_ALGORITHMS, SIZE_MEDIUM,
    LEIDEN_DEFAULT_RESOLUTION, LEIDEN_DEFAULT_PASSES,
    RABBITORDER_VARIANTS, RABBITORDER_DEFAULT_VARIANT,
    GRAPHBREW_VARIANTS, GRAPHBREW_DEFAULT_VARIANT,
    GOGRAPH_VARIANTS, GOGRAPH_DEFAULT_VARIANT,
    TIMEOUT_REORDER,
    Logger, run_command, get_timestamp,
    canonical_algo_key, algo_converter_opt,
)
from ..core.graph_types import GraphInfo
from .reorder_timing import (
    metadata_path as reorder_time_metadata_path,
    read_reorder_time,
    write_reorder_time,
)

# Initialize logger
log = Logger()

# =============================================================================
# Constants
# =============================================================================

from ..core.utils import ENABLE_RUN_LOGGING

# Legacy name migration — bare algorithm names from pre-variant era.
# Maps old bare names → new canonical variant names (and reverse).
_LEGACY_ALGO_NAMES: dict[str, str] = {
    "GraphBrewOrder": "GraphBrewOrder_leiden",
}
_LEGACY_ALGO_NAMES_REV: dict[str, str] = {
    v: k for k, v in _LEGACY_ALGO_NAMES.items()
}


def safe_filename(name: str) -> str:
    """Sanitize algorithm name for use in filenames.
    
    Colons in filenames (e.g., GraphBrewOrder_leiden:dfs.lo) break the C++ CLI
    parser which splits -o arguments on ALL colons. Replace colons with
    underscores to produce safe filenames (e.g., GraphBrewOrder_leiden_dfs.lo).
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
    algo_id: int           # Base algorithm ID (e.g., 12 for GraphBrewOrder)
    name: str              # Display name (e.g., "GraphBrewOrder_leiden")
    option_string: str     # Full option string for -o flag (e.g., "12:hrab")
    variant: str = ""      # Variant name if applicable (e.g., "graphbrew")
    resolution: str = "auto"  # Resolution: "auto", "dynamic", "1.0", etc.
    passes: int = LEIDEN_DEFAULT_PASSES
    
    @property
    def base_name(self) -> str:
        """Get base algorithm name without variant suffix."""
        return ALGORITHMS.get(self.algo_id, f"ALGO_{self.algo_id}")


def get_algorithm_name_with_variant(algo_id: int, variant: str = None) -> str:
    """Get algorithm name with variant suffix for algorithms that have variants.
    
    .. deprecated:: Use ``canonical_algo_key()`` from utils.py directly.
       This wrapper is retained only for backward compatibility with existing
       imports; new code should call ``canonical_algo_key(algo_id, variant)``.
    """
    return canonical_algo_key(algo_id, variant)


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
    leiden_resolution: float = LEIDEN_DEFAULT_RESOLUTION,
    leiden_passes: int = LEIDEN_DEFAULT_PASSES,
    rabbit_variants: List[str] = None,
    graphbrew_variants: List[str] = None,
    gorder_variants: List[str] = None,
    gograph_variants: List[str] = None,
) -> List[AlgorithmConfig]:
    """
    Expand algorithm IDs into AlgorithmConfig objects.
    
    For GOrder (9), optionally expand into default/gograph/csr/fast variants.
    For RabbitOrder (8), optionally expand into csr/boost variants.
    For GraphBrewOrder (12), optionally expand into all preset/strategy variants.
    For GoGraphOrder (16), optionally expand into default/fast/naive variants.
    
    Args:
        algorithms: List of algorithm IDs
        expand_leiden_variants: If True, expand variant-based algorithms into all variants
        leiden_resolution: Resolution parameter for Leiden algorithms
        leiden_passes: Number of passes for Leiden
        rabbit_variants: Which RabbitOrder variants to include (default: csr only)
        graphbrew_variants: Which GraphBrewOrder variants to include (default: leiden only)
        gorder_variants: Which GOrder implementation variants to include (default: None = plain GOrder)
        gograph_variants: Which GoGraphOrder variants to include (default: default only)
    
    Returns:
        List of AlgorithmConfig objects
    """
    from scripts.lib.core.utils import (
        CORDER_DEFAULT_VARIANT,
        CORDER_VARIANTS,
        GORDER_VARIANTS,
        GORDER_DEFAULT_VARIANT,
    )
    
    if rabbit_variants is None:
        # When expanding variants, include both RabbitOrder variants; otherwise just csr
        rabbit_variants = RABBITORDER_VARIANTS if expand_leiden_variants else [RABBITORDER_DEFAULT_VARIANT]
    if graphbrew_variants is None:
        # When expanding variants, include all GraphBrewOrder variants; otherwise just leiden
        graphbrew_variants = GRAPHBREW_VARIANTS if expand_leiden_variants else [GRAPHBREW_DEFAULT_VARIANT]
    if gograph_variants is None:
        # When expanding variants, include all GoGraphOrder variants; otherwise just default
        gograph_variants = GOGRAPH_VARIANTS if expand_leiden_variants else [GOGRAPH_DEFAULT_VARIANT]
    
    configs = []
    
    for algo_id in algorithms:
        base_name = ALGORITHMS.get(algo_id, f"ALGO_{algo_id}")
        
        if algo_id == 9 and gorder_variants and len(gorder_variants) > 1:
            # GOrder: expand into implementation variants. The faithful
            # gograph/csr paths are equivalent; fast is a relaxed heuristic.
            # NOTE: GOrder is intentionally NOT in _VARIANT_ALGO_REGISTRY— its variants
            # share one perceptron weight.
            # We use f"GORDER_{variant}" for filename differentiation only.
            for variant in gorder_variants:
                if variant == "default":
                    option_str = str(algo_id)
                else:
                    option_str = f"{algo_id}:{variant}"
                configs.append(AlgorithmConfig(
                    algo_id=algo_id,
                    name=canonical_algo_key(algo_id, variant),
                    option_string=option_str,
                    variant=variant
                ))
        elif algo_id == 9 and gorder_variants:
            # GOrder: single specific variant
            variant = gorder_variants[0]
            if variant == "default":
                option_str = str(algo_id)
            else:
                option_str = f"{algo_id}:{variant}"
            configs.append(AlgorithmConfig(
                algo_id=algo_id,
                name=canonical_algo_key(algo_id, variant),
                option_string=option_str,
                variant=variant
            ))
        elif algo_id == 8 and expand_leiden_variants and len(rabbit_variants) > 1:
            # RabbitOrder: expand into variants if multiple specified
            for variant in rabbit_variants:
                configs.append(AlgorithmConfig(
                    algo_id=algo_id,
                    name=canonical_algo_key(algo_id, variant),
                    option_string=algo_converter_opt(algo_id, variant),
                    variant=variant
                ))
        elif algo_id == 8:
            # RabbitOrder: use specified variant (default: csr) - ALWAYS include variant in name
            variant = rabbit_variants[0] if rabbit_variants else RABBITORDER_DEFAULT_VARIANT
            configs.append(AlgorithmConfig(
                algo_id=algo_id,
                name=canonical_algo_key(algo_id, variant),
                option_string=algo_converter_opt(algo_id, variant),
                variant=variant
            ))
        elif algo_id == 10 and expand_leiden_variants:
            for variant in CORDER_VARIANTS:
                configs.append(AlgorithmConfig(
                    algo_id=algo_id,
                    name=canonical_algo_key(algo_id, variant),
                    option_string=algo_converter_opt(algo_id, variant),
                    variant=variant,
                ))
        elif algo_id == 10:
            configs.append(AlgorithmConfig(
                algo_id=algo_id,
                name=canonical_algo_key(
                    algo_id, CORDER_DEFAULT_VARIANT),
                option_string=algo_converter_opt(algo_id),
                variant=CORDER_DEFAULT_VARIANT,
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
            for variant in graphbrew_variants:
                configs.append(AlgorithmConfig(
                    algo_id=algo_id,
                    name=canonical_algo_key(algo_id, variant),
                    option_string=algo_converter_opt(algo_id, variant),
                    variant=variant,
                    resolution=leiden_resolution
                ))
        elif algo_id == 12:
            # GraphBrewOrder: use default variant (leiden) - ALWAYS include variant in name
            variant = graphbrew_variants[0] if graphbrew_variants else GRAPHBREW_DEFAULT_VARIANT
            configs.append(AlgorithmConfig(
                algo_id=algo_id,
                name=canonical_algo_key(algo_id, variant),
                option_string=algo_converter_opt(algo_id, variant),
                variant=variant,
                resolution=leiden_resolution
            ))
        elif algo_id == 16 and expand_leiden_variants and len(gograph_variants) > 1:
            # GoGraphOrder: expand into all variants (different orderings)
            for variant in gograph_variants:
                configs.append(AlgorithmConfig(
                    algo_id=algo_id,
                    name=canonical_algo_key(algo_id, variant),
                    option_string=algo_converter_opt(algo_id, variant),
                    variant=variant
                ))
        elif algo_id == 16:
            # GoGraphOrder: single variant - ALWAYS include variant in name
            variant = gograph_variants[0] if gograph_variants else GOGRAPH_DEFAULT_VARIANT
            configs.append(AlgorithmConfig(
                algo_id=algo_id,
                name=canonical_algo_key(algo_id, variant),
                option_string=algo_converter_opt(algo_id, variant),
                variant=variant
            ))
        else:
            # Non-variant algorithms: use canonical key (which includes
            # variant suffix for any algorithms in _VARIANT_ALGO_REGISTRY
            # like RCM_default that don't have explicit handlers above)
            configs.append(AlgorithmConfig(
                algo_id=algo_id,
                name=canonical_algo_key(algo_id),
                option_string=algo_converter_opt(algo_id)
            ))
    
    return configs


# =============================================================================
# Output Parsing
# =============================================================================

def parse_reorder_time_from_converter(output: str) -> Optional[float]:
    """Return complete reorder cost from the shared timing contract.

    New binaries report core mapping work, permutation validation, and CSR
    application separately; their sum is the reusable reorder cost.  Legacy
    outputs without explicit boundaries fall back to their historical unified
    ``Reorder Time`` value.
    """
    # Lazy import avoids the benchmark↔reorder compatibility import cycle.
    from .benchmark import parse_complete_reorder_time

    return parse_complete_reorder_time(output)


def _load_mapping_reorder_time(
    time_path: str,
    mapping_path: str,
) -> float:
    from .benchmark import mapping_permutation_fingerprint

    fingerprint = (
        mapping_permutation_fingerprint(mapping_path)
        if os.path.isfile(mapping_path)
        else None
    )
    value = read_reorder_time(
        time_path,
        expected_mapping_fingerprint=fingerprint,
        allow_legacy=(
            os.environ.get("GRAPHBREW_ALLOW_LEGACY_TIME") == "1"
        ),
    )
    return value if value is not None else 0.0


def _remove_stale_mapping_artifacts(
    mapping_path: str,
    timing_path: str,
) -> None:
    Path(mapping_path).unlink(missing_ok=True)
    Path(timing_path).unlink(missing_ok=True)
    reorder_time_metadata_path(timing_path).unlink(missing_ok=True)


def _write_mapping_reorder_time(
    time_path: str,
    output: str,
) -> float:
    from .benchmark import parse_benchmark_output

    _average, complete, timing = parse_benchmark_output(output)
    if "reorder_time_passes" not in timing:
        raise ValueError("Converter output is missing reorder timing")
    write_reorder_time(
        time_path,
        complete_reorder_time=complete,
        mapping_fingerprint=str(timing.get("mapping_fingerprint", "")),
        algorithm_spec=str(
            timing.get("resolved_algorithm_spec", "")),
    )
    return complete


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
            
            # ORIGINAL and RANDOM are baselines — no reorder mapping to generate
            if algo_id in (0, 1):
                baseline_label = "ORIGINAL" if algo_id == 0 else "RANDOM"
                log.info(f"  [{current}/{total}] {algo_name}: 0.0000s ({baseline_label} baseline, no reorder)")
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
                if not (
                    reorder_time_metadata_path(timing_file).exists()
                    or (
                        os.environ.get("GRAPHBREW_ALLOW_LEGACY_TIME") == "1"
                        and os.path.exists(timing_file)
                    )
                ):
                    legacy_name = _LEGACY_ALGO_NAMES_REV.get(algo_name)
                    if legacy_name:
                        timing_file = os.path.join(graph_mappings_dir, f"{safe_filename(legacy_name)}.time")
                try:
                    reorder_time = _load_mapping_reorder_time(
                        timing_file, actual_map_file)
                    timing_valid = (
                        reorder_time_metadata_path(timing_file).exists()
                        or (
                            os.environ.get(
                                "GRAPHBREW_ALLOW_LEGACY_TIME") == "1"
                            and os.path.exists(timing_file)
                        )
                    )
                except (OSError, ValueError):
                    timing_valid = False
                if timing_valid:
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
                log.warning(
                    f"  [{current}/{total}] {algo_name}: stale timing "
                    "metadata; regenerating mapping")
                os.remove(actual_map_file)
            
            # Remove existing files if force_reorder
            if force_reorder and map_file and os.path.exists(map_file):
                os.remove(map_file)
                timing_file = os.path.join(graph_mappings_dir, f"{safe_name}.time")
                if os.path.exists(timing_file):
                    os.remove(timing_file)
                reorder_time_metadata_path(timing_file).unlink(
                    missing_ok=True)
            
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
                    from scripts.lib.core.graph_data import save_run_log
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
                        timing_file = os.path.join(graph_mappings_dir, f"{safe_name}.time")
                        reorder_time = _write_mapping_reorder_time(
                            timing_file, output)
                        
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
            
            # Skip baselines ORIGINAL and RANDOM (no mapping needed)
            if algo_id in (0, 1):
                baseline_label = "ORIGINAL" if algo_id == 0 else "RANDOM"
                log.info(f"  [{current}/{total}] {algo_name}: no map needed ({baseline_label} baseline)")
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
                if not (
                    reorder_time_metadata_path(timing_file).exists()
                    or (
                        os.environ.get("GRAPHBREW_ALLOW_LEGACY_TIME") == "1"
                        and os.path.exists(timing_file)
                    )
                ):
                    legacy_name = _LEGACY_ALGO_NAMES_REV.get(algo_name)
                    if legacy_name:
                        legacy_tf = os.path.join(graph_mappings_dir, f"{safe_filename(legacy_name)}.time")
                        if (
                            reorder_time_metadata_path(legacy_tf).exists()
                            or (
                                os.environ.get(
                                    "GRAPHBREW_ALLOW_LEGACY_TIME") == "1"
                                and os.path.exists(legacy_tf)
                            )
                        ):
                            actual_timing = legacy_tf
                try:
                    reorder_time = _load_mapping_reorder_time(
                        actual_timing, actual_map_file)
                    timing_valid = (
                        reorder_time_metadata_path(actual_timing).exists()
                        or (
                            os.environ.get(
                                "GRAPHBREW_ALLOW_LEGACY_TIME") == "1"
                            and os.path.exists(actual_timing)
                        )
                    )
                except (OSError, ValueError):
                    timing_valid = False
                if timing_valid:
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
                log.warning(
                    f"  [{current}/{total}] {algo_name}: stale timing "
                    "metadata; regenerating mapping")
                os.remove(actual_map_file)
            
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
                    from scripts.lib.core.graph_data import save_run_log
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
                reorder_time = _write_mapping_reorder_time(
                    timing_file, stdout + stderr)
                
                log.info(f"  [{current}/{total}] {algo_name}: generated ({reorder_time:.4f}s)")
                label_maps[graph.name][algo_name] = map_file
                reorder_results.append(ReorderResult(
                    graph=graph.name,
                    algorithm_id=algo_id,
                    algorithm_name=algo_name,
                    reorder_time=reorder_time,
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
    rabbit_variants: List[str] = None,
    graphbrew_variants: List[str] = None,
    gorder_variants: List[str] = None,
    timeout: int = TIMEOUT_REORDER,
    skip_slow: bool = False,
    force_reorder: bool = False
) -> Tuple[Dict[str, Dict[str, str]], List[ReorderResult]]:
    """
    Generate reorderings with variant expansion.
    
    Creates separate mappings for each variant:
        - GraphBrewOrder_leiden.lo
        - GraphBrewOrder_rabbit.lo
        - GORDER_csr.lo
    
    Args:
        graphs: List of graphs to process
        algorithms: List of algorithm IDs or AlgorithmConfig objects
        bin_dir: Directory containing binaries
        output_dir: Directory for outputs
        expand_leiden_variants: If True, expand variant algorithms into all variants
        leiden_resolution: Resolution parameter
        leiden_passes: Number of passes
        graphbrew_variants: Which GraphBrewOrder variants
        gorder_variants: Which GOrder implementation variants
            (default/gograph/csr/fast)
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
        log.info("Variant expansion enabled")
    
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
            rabbit_variants=rabbit_variants,
            graphbrew_variants=graphbrew_variants,
            gorder_variants=gorder_variants
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
            
            # ORIGINAL and RANDOM are baselines — no reorder mapping to generate
            if cfg.algo_id in (0, 1):
                baseline_label = "ORIGINAL" if cfg.algo_id == 0 else "RANDOM"
                log.info(f"  [{current}/{total}] {cfg.name}: 0.0000s ({baseline_label} baseline, no reorder)")
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
                if not (
                    reorder_time_metadata_path(timing_file).exists()
                    or (
                        os.environ.get("GRAPHBREW_ALLOW_LEGACY_TIME") == "1"
                        and os.path.exists(timing_file)
                    )
                ):
                    legacy_name = _LEGACY_ALGO_NAMES_REV.get(cfg.name)
                    if legacy_name:
                        legacy_tf = os.path.join(graph_mappings_dir, f"{safe_filename(legacy_name)}.time")
                        if (
                            reorder_time_metadata_path(legacy_tf).exists()
                            or (
                                os.environ.get(
                                    "GRAPHBREW_ALLOW_LEGACY_TIME") == "1"
                                and os.path.exists(legacy_tf)
                            )
                        ):
                            actual_timing = legacy_tf
                try:
                    reorder_time = _load_mapping_reorder_time(
                        actual_timing, actual_map_file)
                except ValueError:
                    log.warning(
                        f"  [{current}/{total}] {cfg.name}: "
                        "stale mapping timing; regenerating")
                    _remove_stale_mapping_artifacts(
                        actual_map_file, actual_timing)
                else:
                    log.info(
                        f"  [{current}/{total}] {cfg.name}: "
                        f"exists ({reorder_time:.4f}s)")
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
                reorder_time_metadata_path(timing_file).unlink(
                    missing_ok=True)
            
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
                    from scripts.lib.core.graph_data import save_run_log
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
                reorder_time = _write_mapping_reorder_time(
                    timing_file, stdout + stderr)
                
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
    python -m scripts.lib.pipeline.reorder --graph graphs/email-Enron/email-Enron.mtx
    python -m scripts.lib.pipeline.reorder --graph test.mtx --algorithms 0,8,9
    python -m scripts.lib.pipeline.reorder --graph test.mtx --expand-variants
    python -m scripts.lib.pipeline.reorder --list-algorithms
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
