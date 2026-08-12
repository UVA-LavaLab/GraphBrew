#!/usr/bin/env python3
"""
Benchmark execution utilities for GraphBrew.

Runs graph algorithm benchmarks with various reordering strategies.
Can be used standalone or as a library.

Standalone usage:
    python -m scripts.lib.pipeline.benchmark --graph graphs/email-Enron/email-Enron.mtx -a 0,1,8
    python -m scripts.lib.pipeline.benchmark --graph test.mtx --leiden-variants

Library usage:
    from scripts.lib.pipeline.benchmark import run_benchmark, run_benchmark_suite
    
    result = run_benchmark("pr", "graph.mtx", algorithm="12:community")
    results = run_benchmark_suite("graph.mtx", algorithms=["0", "1", "8"])
"""

import json
import os
import re
import hashlib
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ..core.utils import (
    BIN_DIR, ALGORITHMS, ALGORITHM_IDS, BENCHMARKS,
    BenchmarkResult, log, run_command, check_binary_exists,
    get_results_file, save_json, get_algorithm_name, parse_algorithm_option,
    ENABLE_RUN_LOGGING, canonical_algo_key, algo_converter_opt,
    ELIGIBLE_ALGORITHMS, GRAPHS_DIR, RESULTS_DIR, TIMEOUT_BENCHMARK,
    normalize_graph_name, benchmark_condition_key,
)
from .reorder import get_algorithm_name_with_variant  # deprecated; kept for compat
from ..ml.features import update_graph_properties, save_graph_properties_cache


# =============================================================================
# Adaptive Timeout
# =============================================================================

def compute_adaptive_timeout(edges: int, base_timeout: int = 600) -> int:
    """
    Compute a timeout that scales with graph size.

    Small graphs (<1M edges) get the base timeout (default 600s).
    Medium graphs (1M–10M) get 2× base.
    Large graphs (10M–100M) get 4× base.
    Very large graphs (>100M) get 8× base.

    This prevents false-positive timeouts on large graphs while still
    catching hangs and bugs quickly on small ones.

    Args:
        edges: Number of edges in the graph.
        base_timeout: Base timeout in seconds (applied to <1M-edge graphs).

    Returns:
        Adjusted timeout in seconds.
    """
    if edges <= 0:
        return base_timeout
    if edges < 1_000_000:
        return base_timeout
    elif edges < 10_000_000:
        return base_timeout * 2
    elif edges < 100_000_000:
        return base_timeout * 4
    else:
        return base_timeout * 8


# =============================================================================
# Reorder Time Utilities
# =============================================================================

# Default directory for mappings (relative to results dir)
_DEFAULT_MAPPINGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))), "results", "mappings")
_MAPPING_IDENTITY_CACHE: Dict[Tuple[str, int, int], str] = {}


def mapping_artifact_identity(mapping_path: str | os.PathLike) -> str:
    """Return a cached content identity for a pre-generated mapping.

    Mapping filenames identify the algorithm, but not a regenerated artifact.
    The SHA-256 digest makes resume exact while the stat-keyed cache ensures
    each mapping is read at most once per harness process.
    """
    path = Path(mapping_path).resolve()
    before = path.stat()
    cache_key = (str(path), before.st_size, before.st_mtime_ns)
    cached = _MAPPING_IDENTITY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"Mapping changed while hashing: {path}")

    identity = f"map:{path.name}:sha256:{digest.hexdigest()}"
    _MAPPING_IDENTITY_CACHE[cache_key] = identity
    return identity


def load_reorder_time_for_algo(graph_name: str, algo_name: str,
                               mappings_dir: str = None) -> float:
    """Load reorder time from a .time file for a given graph and algorithm.

    Checks ``{mappings_dir}/{graph_name}/{algo_name}.time``.  Returns 0.0
    if the file does not exist.
    """
    mappings_dir = mappings_dir or _DEFAULT_MAPPINGS_DIR
    time_file = os.path.join(mappings_dir, graph_name, f"{algo_name}.time")
    if os.path.isfile(time_file):
        try:
            return float(Path(time_file).read_text().strip())
        except (ValueError, IOError):
            return 0.0
    return 0.0


# =============================================================================
# Output Parsing
# =============================================================================

def parse_benchmark_output(output: str) -> Tuple[float, float, Dict]:
    """
    Parse benchmark stdout to extract timing information.

    Returns a flat dict in ``extra`` containing (when present):
      - ``trial_times``: list of per-trial wall-clock times (seconds)
      - ``average_time``: same as the tuple's first element
      - ``representation_build_time``: input-to-canonical-CSR construction
      - ``reorder_core_time``: mapping construction/load
      - ``reorder_validation_time``: permutation validation
      - ``reorder_apply_time``: CSR relabel/application
      - ``total_preprocessing_time``: direct pre-kernel wall clock
      - ``preprocessing_time``, ``total_time``: legacy standalone timings
      - ``read_time``, ``topology_analysis_time``, ``relabel_map_time``:
        GraphBrew-specific load/topology timings
      - ``reorder_time_passes``: complete per-pass reorder times for chained
        orderings; the returned ``reorder_time`` is their sum
      - ``mapping_generation_time``: compatibility alias for
        ``reorder_core_time``
      - ``mteps``, ``iterations``: BFS / PR specific
      - weighted SSSP policy and answer fingerprints
      - topology features (degree_variance, hub_concentration, modularity, ...)

    Args:
        output: Stdout from benchmark execution

    Returns:
        Tuple of (average_time, reorder_time_total, extra_info)
    """
    avg_time = 0.0
    extra: Dict = {}
    trial_times: List[float] = []
    legacy_core_passes: List[float] = []
    core_passes: List[float] = []
    validation_passes: List[float] = []
    apply_passes: List[float] = []
    end_to_end_passes: List[float] = []
    iteration_counts: List[int] = []
    final_errors: List[float] = []

    def _num(line: str) -> float | None:
        try:
            return float(line.split(":", 1)[1].strip().split()[0])
        except (ValueError, IndexError):
            return None

    for raw in output.split("\n"):
        line = raw.strip()
        if not line:
            continue
        line_lower = line.lower()

        # ---- Reorder phases (may repeat for chained orderings)
        if line.startswith("Reorder Core Time"):
            v = _num(line)
            if v is not None:
                core_passes.append(v)
            continue
        if line.startswith("Reorder Time"):
            v = _num(line)
            if v is not None:
                legacy_core_passes.append(v)
            continue
        if line.startswith("Reorder Validation Time"):
            v = _num(line)
            if v is not None:
                validation_passes.append(v)
            continue
        if line.startswith("Reorder Apply Time"):
            v = _num(line)
            if v is not None:
                apply_passes.append(v)
            continue
        if line.startswith("Reorder End-to-End Time"):
            v = _num(line)
            if v is not None:
                end_to_end_passes.append(v)
            continue

        # ---- Trial Time:   <secs>     (one per trial, ordered)
        if line.startswith("Trial Time"):
            v = _num(line)
            if v is not None:
                trial_times.append(v)
            continue

        # ---- Average Time
        if line.startswith("Average Time"):
            v = _num(line)
            if v is not None:
                avg_time = v
                extra["average_time"] = v
            continue

        # ---- Other standalone timings produced by GraphBrew binaries
        for key in (
            "Representation Build Time", "Total Preprocessing Time",
            "Preprocessing Time", "Total Time", "Read Time",
            "Topology Analysis Time", "Relabel Map Time",
            "Adaptive Feature Time", "Adaptive Model Time",
            "Adaptive Selection Time", "Adaptive Arm Map Time",
            "Adaptive Confidence",
            "Property Working Set Bytes", "LLC Capacity Bytes",
            "Property WSR LLC",
        ):
            if line.startswith(key):
                v = _num(line)
                if v is not None:
                    extra[key.lower().replace(" ", "_")] = v
                break

        if line.startswith("Adaptive Tier0 Features:"):
            _, payload = line.split(":", 1)
            try:
                extra["adaptive_tier0_features"] = json.loads(
                    payload.strip()
                )
            except json.JSONDecodeError as error:
                raise ValueError(
                    "Malformed Adaptive Tier0 feature record"
                ) from error
            continue

        for label, key in (
            ("Adaptive Predicted", "adaptive_predicted"),
            ("Adaptive Applied", "adaptive_applied"),
            (
                "Adaptive Override Reason",
                "adaptive_override_reason",
            ),
            ("Adaptive Weight Source", "adaptive_weight_source"),
            ("Adaptive Tier0 Trained", "adaptive_tier0_trained"),
            ("Leiden Layout", "leiden_layout"),
            ("Leiden Seed", "leiden_seed"),
        ):
            if line.startswith(label + ":"):
                extra[key] = line.split(":", 1)[1].strip()
                break

        # ---- MTEPS (BFS)
        if "mteps" in line_lower:
            m = re.search(r"[\d.]+", line)
            if m:
                extra["mteps"] = float(m.group())

        # ---- PageRank / SSSP iteration count
        if line.startswith("Iterations"):
            v = _num(line)
            if v is not None:
                iteration_counts.append(int(v))
            continue
        if line.startswith("Final Error"):
            v = _num(line)
            if v is not None:
                final_errors.append(v)
            continue
        if "iteration" in line_lower:
            m = re.search(r"(\d+)\s*iteration", line_lower)
            if m:
                extra["iterations"] = int(m.group(1))

    if trial_times:
        extra["trial_times"] = trial_times
    if iteration_counts:
        extra["iteration_counts"] = iteration_counts
        extra["iterations"] = iteration_counts[-1]
    if final_errors:
        extra["final_errors"] = final_errors
        extra["final_error"] = final_errors[-1]
    if core_passes and legacy_core_passes and core_passes != legacy_core_passes:
        raise ValueError(
            "Reorder Core Time and legacy Reorder Time disagree"
        )
    effective_core_passes = core_passes or legacy_core_passes
    if effective_core_passes:
        extra["reorder_core_time_passes"] = effective_core_passes
        extra["reorder_core_time"] = sum(effective_core_passes)
        # Historical compatibility: consumers previously called the mapping
        # construction/load phase "mapping generation".
        extra["mapping_generation_time_passes"] = effective_core_passes
        extra["mapping_generation_time"] = sum(effective_core_passes)
    if validation_passes:
        extra["reorder_validation_time_passes"] = validation_passes
        extra["reorder_validation_time"] = sum(validation_passes)
    if apply_passes:
        extra["reorder_apply_time_passes"] = apply_passes
        extra["reorder_apply_time"] = sum(apply_passes)
    if end_to_end_passes:
        if validation_passes or apply_passes:
            if not (
                len(effective_core_passes)
                == len(validation_passes)
                == len(apply_passes)
                == len(end_to_end_passes)
            ):
                raise ValueError(
                    "Incomplete explicit reorder phase timing"
                )
            for core, validation, apply, complete in zip(
                effective_core_passes,
                validation_passes,
                apply_passes,
                end_to_end_passes,
            ):
                expected = core + validation + apply
                if abs(complete - expected) > 1e-3:
                    raise ValueError(
                        "Complete reorder time disagrees with phase timings"
                    )
        extra["reorder_time_passes"] = end_to_end_passes
        extra["complete_reorder_time"] = sum(end_to_end_passes)
    elif effective_core_passes:
        extra["reorder_time_passes"] = effective_core_passes
        extra["complete_reorder_time"] = sum(effective_core_passes)
    reorder_time = sum(
        end_to_end_passes if end_to_end_passes else effective_core_passes
    )
    
    # Extract topology features for weight learning
    # These are printed by the C++ code during graph loading
    dv_match = re.search(r'Degree Variance:\s*([\d.]+)', output)
    if dv_match:
        extra['degree_variance'] = float(dv_match.group(1))
    
    hc_match = re.search(r'Hub Concentration:\s*([\d.]+)', output)
    if hc_match:
        extra['hub_concentration'] = float(hc_match.group(1))
    
    ad_match = re.search(r'Avg Degree:\s*([\d.]+)', output)
    if ad_match:
        extra['avg_degree'] = float(ad_match.group(1))
    
    cc_match = re.search(r'Clustering Coefficient:\s*([\d.]+)', output)
    if cc_match:
        extra['clustering_coefficient'] = float(cc_match.group(1))
    
    apl_match = re.search(r'Avg Path Length:\s*([\d.]+)', output)
    if apl_match:
        extra['avg_path_length'] = float(apl_match.group(1))
    
    diam_match = re.search(r'Diameter Estimate:\s*([\d.]+)', output)
    if diam_match:
        extra['diameter'] = float(diam_match.group(1))
    
    comm_match = re.search(r'Community Count Estimate:\s*([\d.]+)', output)
    if comm_match:
        extra['community_count'] = float(comm_match.group(1))
    
    mod_match = re.search(r'Modularity:\s*([\d.]+)', output)
    if mod_match:
        extra['modularity'] = float(mod_match.group(1))

    sampled_span = re.search(
        r"Mapping Sampled Edge Span:\s*([\d.]+)",
        output,
    )
    if sampled_span:
        extra["mapping_sampled_edge_span"] = float(
            sampled_span.group(1)
        )

    sampled_edges = re.search(
        r"Mapping Sampled Edges:\s*(\d+)",
        output,
    )
    if sampled_edges:
        extra["mapping_sampled_edges"] = int(sampled_edges.group(1))

    weight_scheme = re.search(r"Weight Scheme:\s*(\S+)", output)
    if weight_scheme:
        extra["weight_scheme"] = weight_scheme.group(1)

    weight_checksum = re.search(r"Weight Checksum:\s*([0-9a-fA-F]+)", output)
    if weight_checksum:
        extra["weight_checksum"] = weight_checksum.group(1).lower()

    delta = re.search(r"Delta:\s*(\d+)", output)
    if delta:
        extra["delta"] = int(delta.group(1))

    source_originals = [
        int(value)
        for value in re.findall(r"Source Original:\s*(-?\d+)", output)
    ]
    if source_originals:
        extra["source_originals"] = source_originals

    source_internals = [
        int(value)
        for value in re.findall(r"Source Internal:\s*(-?\d+)", output)
    ]
    if source_internals:
        extra["source_internals"] = source_internals

    source_out_degrees = [
        int(value)
        for value in re.findall(
            r"Source Out Degree:\s*(-?\d+)",
            output,
        )
    ]
    if source_out_degrees:
        extra["source_out_degrees"] = source_out_degrees

    distance_fingerprints = [
        value.lower()
        for value in re.findall(
            r"Distance Fingerprint:\s*([0-9a-fA-F]+)", output,
        )
    ]
    if distance_fingerprints:
        extra["distance_fingerprints"] = distance_fingerprints

    mapping_fingerprints = [
        value.lower()
        for value in re.findall(
            r"Mapping Fingerprint:\s*([0-9a-fA-F]+)", output,
        )
    ]
    if mapping_fingerprints:
        extra["mapping_fingerprints"] = mapping_fingerprints
        extra["mapping_fingerprint"] = mapping_fingerprints[-1]

    schedule_sensitive = re.findall(
        r"Reorder Schedule Sensitive:\s*(true|false)",
        output,
        flags=re.IGNORECASE,
    )
    if schedule_sensitive:
        extra["reorder_schedule_sensitive"] = (
            schedule_sensitive[-1].lower() == "true"
        )
    thread_policy_sensitive = re.findall(
        r"Reorder Thread Policy Sensitive:\s*(true|false)",
        output,
        flags=re.IGNORECASE,
    )
    if thread_policy_sensitive:
        extra["reorder_thread_policy_sensitive"] = (
            thread_policy_sensitive[-1].lower() == "true"
        )

    work_labels = {
        "bfs_td_edges": "BFS TD Edges",
        "bfs_bu_edges": "BFS BU Edges",
        "bfs_edges_examined": "BFS Edges Examined",
        "bfs_steps": "BFS Steps",
        "sssp_edges_examined": "SSSP Edges Examined",
        "sssp_relax_successes": "SSSP Relax Successes",
        "sssp_frontier_entries": "SSSP Frontier Entries",
        "sssp_bucket_iterations": "SSSP Bucket Iterations",
        "cc_sampled_edges": "CC Sampled Edges",
        "cc_final_edges": "CC Final Edges",
        "cc_compress_steps": "CC Compress Steps",
        "cc_skipped_vertices": "CC Skipped Vertices",
        "cc_sv_iterations": "CC-SV Iterations",
        "cc_sv_edges_examined": "CC-SV Edges Examined",
        "cc_sv_compress_steps": "CC-SV Compress Steps",
        "bc_bfs_edges": "BC BFS Edges",
        "bc_backprop_edges": "BC Backprop Edges",
        "bc_max_depth": "BC Max Depth",
    }
    for key, label in work_labels.items():
        values = [
            int(value)
            for value in re.findall(
                rf"{re.escape(label)}:\s*(\d+)", output,
            )
        ]
        if values:
            extra[key] = values

    pr_mode = re.search(r"PR Mode:\s*(\S+)", output)
    if pr_mode:
        extra["pr_mode"] = pr_mode.group(1)

    verification = re.findall(
        r"Verification:\s*(PASS|FAIL)", output,
    )
    extra["verification"] = verification
    extra["verification_state"] = (
        "fail" if "FAIL" in verification
        else "pass" if verification
        else "not-run"
    )
    
    return avg_time, reorder_time, extra


def parse_complete_reorder_time(output: str) -> float | None:
    """Return complete reorder cost, with a legacy core-only fallback."""
    _average, reorder_time, timing = parse_benchmark_output(output)
    if "reorder_time_passes" not in timing:
        return None
    return reorder_time


# =============================================================================
# Benchmark Execution
# =============================================================================

def format_source_list(source_originals: Sequence[int]) -> str:
    sources = [int(source) for source in source_originals]
    if not sources:
        raise ValueError("Source list cannot be empty")
    if any(source < 0 for source in sources):
        raise ValueError("Source IDs must be non-negative")
    if len(sources) != len(set(sources)):
        raise ValueError("Source IDs must be unique")
    return ",".join(str(source) for source in sources)


class SourceContractError(RuntimeError):
    """A frozen adaptive source contract was violated."""


def attach_source_trial_metadata(
    extra: Dict,
    *,
    process_id: int,
    measurement_mode: str,
    source_policy_id: str,
    source_repeats: int,
    expected_sources: Sequence[int] = None,
    expected_internals: Sequence[int] = None,
    expected_out_degrees: Sequence[int] = None,
) -> None:
    """Normalize parsed source output into the adaptive trial contract."""
    if process_id < 0:
        raise ValueError("process_id must be non-negative")
    if measurement_mode not in {"cold-process", "warm-block"}:
        raise ValueError("Unknown source measurement mode")
    if not source_policy_id:
        raise ValueError("source_policy_id is required")
    if source_repeats <= 0:
        raise ValueError("source_repeats must be positive")

    times = extra.get("trial_times", [])
    originals = extra.get("source_originals", [])
    internals = extra.get("source_internals", [])
    degrees = extra.get("source_out_degrees", [])
    lengths = {
        len(times),
        len(originals),
        len(internals),
        len(degrees),
    }
    if len(lengths) != 1 or not times:
        raise SourceContractError(
            "Source trial times and source metadata must be aligned")

    repetitions: Dict[int, int] = {}
    source_trials = []
    for trial_time, source, internal, degree in zip(
        times,
        originals,
        internals,
        degrees,
    ):
        source = int(source)
        repetition = repetitions.get(source, 0)
        repetitions[source] = repetition + 1
        source_trials.append({
            "process_id": process_id,
            "source_id": source,
            "source_internal": int(internal),
            "source_out_degree": int(degree),
            "repetition_index": repetition,
            "measurement_mode": measurement_mode,
            "trial_time": float(trial_time),
        })
    if any(count != source_repeats for count in repetitions.values()):
        raise SourceContractError(
            "Source trial repetitions do not match source_repeats")
    if expected_sources is not None:
        expected = [
            int(source)
            for source in expected_sources
            for _ in range(source_repeats)
        ]
        observed = [trial["source_id"] for trial in source_trials]
        if observed != expected:
            raise SourceContractError(
                "Observed sources do not match the frozen source manifest")
    if expected_out_degrees is not None:
        expected_degrees = [
            int(degree)
            for degree in expected_out_degrees
            for _ in range(source_repeats)
        ]
        observed_degrees = [
            trial["source_out_degree"] for trial in source_trials
        ]
        if observed_degrees != expected_degrees:
            raise SourceContractError(
                "Observed source degrees do not match the frozen manifest")
    if expected_internals is not None:
        expected_internal_values = [
            int(internal)
            for internal in expected_internals
            for _ in range(source_repeats)
        ]
        observed_internals = [
            trial["source_internal"] for trial in source_trials
        ]
        if observed_internals != expected_internal_values:
            raise SourceContractError(
                "Observed source internals do not match the frozen labeling")
    extra["source_policy_id"] = source_policy_id
    extra["source_trials"] = source_trials


def run_benchmark(
    benchmark: str,
    graph_path: str,
    algorithm: str = "0",
    trials: int = 3,
    symmetric: bool = True,
    timeout: int = 600,
    extra_args: List[str] = None,
    bin_dir: str = None,
    log_algorithm: str = None,
    log_graph_name: str = None,
    result_graph_name: str = None,
    source_originals: Sequence[int] = None,
    source_repeats: int = 1,
    source_policy_id: str = None,
    process_id: int = None,
    measurement_mode: str = None,
    expected_source_out_degrees: Sequence[int] = None,
    expected_source_internals: Sequence[int] = None,
    labeling: str = "natural",
    threads: int = 0,
    mapping_identity_id: str = "direct",
    algorithm_spec: str = None,
    attempt: int = 1,
    self_record: bool = False,
) -> BenchmarkResult:
    """
    Run a single benchmark with specified algorithm.
    
    Args:
        benchmark: Benchmark name (pr, bfs, cc, etc.)
        graph_path: Path to graph file
        algorithm: Algorithm option string (e.g., "0", "12:community", "15:1.0")
        trials: Number of trials
        symmetric: Use symmetric graph flag (-s)
        timeout: Timeout in seconds
        extra_args: Additional command line arguments
        bin_dir: Directory containing benchmark binaries
        log_algorithm: Override algorithm name for log filenames.  When using
            pre-generated .sg files (algorithm="0") or MAP mode
            (algorithm="13:..."), pass the real algorithm name here so
            logs are named correctly (e.g. "GORDER" instead of "ORIGINAL"
            or "MAP").
        log_graph_name: Override graph name for log directory.  When
            benchmarking a pre-generated .sg file (e.g. ca-GrQc_GORDER.sg),
            pass the base graph name ("ca-GrQc") so logs are grouped
            correctly.
        result_graph_name: Explicit graph key stored in the result record.
        labeling: Observation labeling (e.g. ``natural`` / ``shuffled``).
        threads: Thread policy recorded on the observation.
        mapping_identity_id: Mapping identity for this run (``direct`` for a
            runtime reorder, or a ``map:<file>`` identity for a pre-generated
            mapping) so direct and MAP inputs never collide.
        algorithm_spec: Exact ordered reordering specification. Defaults to
            the literal ``algorithm`` argument.
        attempt: Attempt index for repeated draws of the same condition.
        self_record: When False (default) the C++ subprocess is launched with
            ``GRAPHBREW_DB_DIR=''`` so no ambient data-dir writer is inherited
            — the Python harness is the sole official writer for generic runs.
            When True, the ambient environment is inherited so an explicit
            ``-D`` flag or exported ``GRAPHBREW_DB_DIR`` can enable C++
            self-recording.
        
    Returns:
        BenchmarkResult with timing information
    """
    graph_path = Path(graph_path)
    graph_name = result_graph_name or normalize_graph_name(graph_path)
    bin_dir_path = Path(bin_dir) if bin_dir else BIN_DIR
    
    algo_id, _ = parse_algorithm_option(algorithm)
    algo_name = get_algorithm_name(algorithm)
    resolved_mode = measurement_mode or "process"

    def _make_result(**overrides) -> BenchmarkResult:
        """Build a BenchmarkResult with the observation condition populated."""
        fields = dict(
            graph=graph_name,
            algorithm=algo_name,
            algorithm_id=algo_id,
            benchmark=benchmark,
            time_seconds=0.0,
            run_id=uuid.uuid4().hex,
            algorithm_spec=algorithm_spec or algorithm,
            labeling=labeling,
            measurement_mode=resolved_mode,
            threads=threads,
            mapping_identity_id=mapping_identity_id,
            process_id=process_id if process_id is not None else 0,
            attempt=attempt,
        )
        fields.update(overrides)
        return BenchmarkResult(**fields)

    # Build command
    binary = bin_dir_path / benchmark
    if not binary.exists():
        return _make_result(
            success=False,
            error=f"Binary not found: {binary}",
            error_kind="missing-binary",
        )
    
    if source_repeats <= 0:
        raise ValueError("source_repeats must be positive")
    if source_originals and len(source_originals) > 1:
        expected_trials = len(source_originals) * source_repeats
        if trials != expected_trials:
            raise ValueError(
                "trials must equal source_count * source_repeats"
            )
    cmd = [str(binary), "-f", str(graph_path), "-o", algorithm, "-n", str(trials)]
    if source_originals:
        cmd.extend(["-r", format_source_list(source_originals)])
        if source_repeats != 1:
            cmd.extend(["-R", str(source_repeats)])
        if benchmark == "bc":
            cmd.extend(["-i", "1"])
    
    if symmetric:
        cmd.append("-s")
    
    if extra_args:
        cmd.extend(extra_args)

    # Child environment: for generic runs, disable inherited C++ self-recording
    # so the Python harness stays the sole official writer.  Explicit
    # self-recording (self_record=True) inherits the ambient environment.
    child_env = os.environ.copy()
    if not self_record:
        child_env["GRAPHBREW_DB_DIR"] = ""
    
    # Run benchmark
    try:
        start_time = time.time()
        result = run_command(cmd, timeout=timeout, check=False, env=child_env)
        elapsed = time.time() - start_time
        
        # Save run log
        if ENABLE_RUN_LOGGING:
            try:
                from scripts.lib.core.graph_data import save_run_log
                save_run_log(
                    graph_name=log_graph_name or graph_name,
                    operation='benchmark',
                    algorithm=log_algorithm or algo_name,
                    benchmark=benchmark,
                    output=result.stdout + "\n--- STDERR ---\n" + result.stderr if result.stderr else result.stdout,
                    command=' '.join(str(c) for c in cmd),
                    exit_code=result.returncode,
                    duration=elapsed
                )
            except Exception as e:
                log.debug(f"Failed to save run log: {e}")
        
        if result.returncode != 0:
            error_msg = result.stderr[:500] if result.stderr else f"Exit code {result.returncode}"
            return _make_result(
                success=False,
                error=error_msg,
                error_kind="process-failure",
            )
        
        # Parse output
        avg_time, reorder_time, extra = parse_benchmark_output(result.stdout)
        if source_policy_id is not None:
            if process_id is None or measurement_mode is None:
                raise ValueError(
                    "Adaptive source metadata requires policy, process, "
                    "and measurement mode")
            attach_source_trial_metadata(
                extra,
                process_id=process_id,
                measurement_mode=measurement_mode,
                source_policy_id=source_policy_id,
                source_repeats=source_repeats,
                expected_sources=source_originals,
                expected_internals=expected_source_internals,
                expected_out_degrees=expected_source_out_degrees,
            )
        
        return _make_result(
            time_seconds=avg_time,
            reorder_time=reorder_time,
            representation_build_time=float(
                extra.get("representation_build_time", 0.0)
            ),
            reorder_core_time=float(
                extra.get("reorder_core_time", 0.0)
            ),
            reorder_validation_time=float(
                extra.get("reorder_validation_time", 0.0)
            ),
            reorder_apply_time=float(
                extra.get("reorder_apply_time", 0.0)
            ),
            total_preprocessing_time=float(
                extra.get("total_preprocessing_time", 0.0)
            ),
            mapping_fingerprint=str(
                extra.get("mapping_fingerprint", "")
            ),
            reorder_schedule_sensitive=bool(
                extra.get("reorder_schedule_sensitive", False)
            ),
            reorder_thread_policy_sensitive=bool(
                extra.get(
                    "reorder_thread_policy_sensitive", False)
            ),
            trials=trials,
            success=True,
            extra=extra,
        )
        
    except SourceContractError:
        raise
    except subprocess.TimeoutExpired as error:
        return _make_result(
            success=False,
            error=str(error),
            error_kind="timeout",
        )
    except Exception as e:
        return _make_result(
            success=False,
            error=str(e),
            error_kind="process-failure",
        )


def run_benchmark_suite(
    graph_path: str,
    algorithms: List[str] = None,
    benchmarks: List[str] = None,
    trials: int = 3,
    timeout: int = 600
) -> List[BenchmarkResult]:
    """
    Run a suite of benchmarks on a graph.
    
    Args:
        graph_path: Path to graph file
        algorithms: List of algorithm option strings (default: ORIGINAL, RANDOM, RABBITORDER)
        benchmarks: List of benchmark names (default: BENCHMARKS from utils)
        trials: Number of trials per config
        timeout: Timeout in seconds
        
    Returns:
        List of BenchmarkResult
    """
    if algorithms is None:
        algorithms = [str(ALGORITHM_IDS["ORIGINAL"]),
                      str(ALGORITHM_IDS["RANDOM"]),
                      str(ALGORITHM_IDS["RABBITORDER"])]
    if benchmarks is None:
        benchmarks = list(BENCHMARKS)
    
    results = []
    graph_name = Path(graph_path).stem
    
    log.info(f"Running {len(benchmarks)} benchmarks × {len(algorithms)} algorithms on {graph_name}")
    
    for bench in benchmarks:
        if not check_binary_exists(bench):
            log.warning(f"Skipping {bench}: binary not found")
            continue
        
        for algo in algorithms:
            algo_name = get_algorithm_name(algo)
            log.info(f"  {bench} with {algo_name}...")
            
            result = run_benchmark(
                benchmark=bench,
                graph_path=graph_path,
                algorithm=algo,
                trials=trials,
                timeout=timeout
            )
            results.append(result)
            
            if result.success:
                log.info(f"    {result.time_seconds:.4f}s")
            else:
                log.warning(f"    FAILED: {result.error[:50]}")
    
    return results


def run_benchmarks_multi_graph(
    graphs: List,  # List of GraphInfo objects
    algorithms: List[int],
    benchmarks: List[str],
    bin_dir: str = None,
    num_trials: int = 3,
    timeout: int = 600,
    label_maps: Dict[str, Dict[str, str]] = None,
    weights_dir: str = None,
    update_weights: bool = True,
    skip_slow: bool = False,
    progress = None,  # Optional ProgressTracker
    use_pregenerated: bool = False,
    on_graph_complete = None,  # Optional[Callable[[str, List[BenchmarkResult]], None]]
    skip_existing: set = None,  # Optional[Set[condition-key]] already in DB
    labeling: str = "natural",
    measurement_mode: str = "process",
    threads: int = 0,
) -> List[BenchmarkResult]:
    """
    Run benchmarks across multiple graphs.
    
    This is the main multi-graph benchmarking function used by the experiment pipeline.
    
    Args:
        graphs: List of GraphInfo objects
        algorithms: List of algorithm IDs
        benchmarks: List of benchmark names
        bin_dir: Binary directory (default: bench/bin)
        num_trials: Number of trials per configuration
        timeout: Timeout in seconds
        label_maps: Pre-computed label maps {graph_name: {algo_name: path}}
        weights_dir: Directory for weight files
        update_weights: Whether to update weights incrementally
        skip_slow: Skip slow algorithms (GORDER, CORDER, RCM)
        progress: Optional progress tracker
        use_pregenerated: If True, use pre-generated ``{graph}_{ALGO}.sg``
            files (loaded with ``-o 0``) to skip runtime reorder overhead
        on_graph_complete: Optional callback invoked after each graph's
            benchmarks finish.  Receives ``(graph_name, graph_results)``.
            Useful for per-graph incremental flushing to the datastore.
        skip_existing: Optional set of shared ``benchmark_condition_key``
            tuples already in the database.  A run is skipped only when its
            fully-resolved condition key (including labeling, measurement mode,
            thread policy and mapping identity) matches — so direct and MAP
            inputs resume independently.
        labeling: Observation labeling for every result (e.g. ``natural`` /
            ``shuffled``).
        measurement_mode: Measurement mode recorded on every result.
        threads: Thread policy recorded on every result.
        
    Returns:
        List of BenchmarkResult objects
    """
    bin_dir = bin_dir or str(BIN_DIR)
    label_maps = label_maps or {}
    results = []
    skip_existing = skip_existing or set()

    def _condition_key(algo_name: str, algorithm_spec: str, bench: str,
                       graph_name: str, mapping_identity_id: str,
                       attempt: int = 1):
        """Compute the shared condition key for resume comparison."""
        return benchmark_condition_key({
            "graph": graph_name,
            "algorithm": algo_name,
            "algorithm_spec": algorithm_spec,
            "benchmark": bench,
            "labeling": labeling,
            "measurement_mode": measurement_mode,
            "threads": threads,
            "mapping_identity_id": mapping_identity_id,
            "attempt": attempt,
        })
    
    # Filter slow algorithms if requested
    if skip_slow:
        from scripts.lib.core.utils import SLOW_ALGORITHMS
        algorithms = [a for a in algorithms if a not in SLOW_ALGORITHMS]
    
    total_configs = len(graphs) * len(algorithms) * len(benchmarks)
    completed = 0
    skipped = 0
    skipped_existing = 0
    
    # Track (graph, benchmark) combos where ORIGINAL or first algo timed out / crashed.
    timed_out_combos: set = set()
    
    for graph in graphs:
        graph_name = graph.name
        graph_path = graph.path
        graph_label_maps = label_maps.get(graph_name, {})
        graph_results = []  # Per-graph results for on_graph_complete callback
        
        if progress:
            progress.info(f"Benchmarking: {graph_name}")
        
        # Adaptive timeout based on graph size
        graph_timeout = compute_adaptive_timeout(graph.edges, timeout)
        if graph_timeout != timeout and progress:
            progress.info(f"  Adaptive timeout: {graph_timeout}s (edges={graph.edges:,})")

        for bench in benchmarks:
            if not check_binary_exists(bench, bin_dir):
                log.warning(f"Skipping {bench}: binary not found")
                continue
            
            for algo_id in algorithms:
                # Always include variant in name for algorithms that have variants
                algo_name = get_algorithm_name_with_variant(algo_id)
                combo_key = (graph_name, bench)

                # ── Resolve the mapping mode first so the observation
                #    condition (direct vs pre-generated MAP) is known before
                #    the resume check and result recording. ──
                label_map_path = graph_label_maps.get(algo_name, "")
                mappings_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(graph_path))),
                    'mappings', graph_name,
                )
                pregen_lo = os.path.join(mappings_dir, f"{algo_name}.lo")

                if (use_pregenerated and algo_id not in (0, 1)
                        and os.path.isfile(pregen_lo)):
                    algo_opt = f"13:{pregen_lo}"
                    mapping_identity_id = mapping_artifact_identity(pregen_lo)
                    run_mode = "pregen"
                elif (label_map_path and os.path.exists(label_map_path)
                        and algo_id != 0):
                    algo_opt = f"13:{label_map_path}"
                    mapping_identity_id = mapping_artifact_identity(
                        label_map_path
                    )
                    run_mode = "labelmap"
                else:
                    algo_opt = str(algo_id)
                    mapping_identity_id = "direct"
                    run_mode = "direct"

                # Resume: skip runs already in the database.  The key is the
                # shared benchmark_condition_key computed *after* the mapping
                # mode is known, so direct and MAP inputs resume independently.
                if _condition_key(algo_name, algo_opt, bench, graph_name,
                                  mapping_identity_id) in skip_existing:
                    completed += 1
                    skipped_existing += 1
                    continue

                # Early-exit: skip remaining algorithms if this graph×benchmark
                # already proved intractable (timeout or crash on a prior algorithm)
                if combo_key in timed_out_combos:
                    result = BenchmarkResult(
                        graph=graph_name,
                        algorithm=algo_name,
                        algorithm_id=algo_id,
                        benchmark=bench,
                        time_seconds=0.0,
                        success=False,
                        error="SKIPPED: prior algorithm timed out on this graph+benchmark",
                        algorithm_spec=algo_opt,
                        labeling=labeling,
                        measurement_mode=measurement_mode,
                        threads=threads,
                        mapping_identity_id=mapping_identity_id,
                    )
                    result.nodes = graph.nodes
                    result.edges = graph.edges
                    results.append(result)
                    graph_results.append(result)
                    completed += 1
                    skipped += 1
                    continue

                result = run_benchmark(
                    benchmark=bench,
                    graph_path=graph_path,
                    algorithm=algo_opt,
                    trials=num_trials,
                    timeout=graph_timeout,
                    bin_dir=bin_dir,
                    log_algorithm=algo_name,
                    log_graph_name=graph_name if run_mode != "direct" else None,
                    labeling=labeling,
                    measurement_mode=measurement_mode,
                    threads=threads,
                    mapping_identity_id=mapping_identity_id,
                )

                # ── Pre-generated .lo mapping path ───────────────────
                if run_mode == "pregen":
                    # Preserve original algo identity for analysis
                    result.algorithm = algo_name
                    result.algorithm_id = algo_id
                    result.graph = graph_name
                    result.nodes = graph.nodes
                    result.edges = graph.edges
                    # Load reorder_time from .time file (written during pregeneration)
                    if result.reorder_time <= 0:
                        result.reorder_time = load_reorder_time_for_algo(
                            graph_name, algo_name, mappings_dir=mappings_dir)
                    results.append(result)
                    graph_results.append(result)
                    completed += 1
                    if progress and completed % 10 == 0:
                        progress.info(f"  Progress: {completed}/{total_configs}")
                    continue

                # Detect timeout or crash — mark this graph×benchmark as intractable
                if not result.success:
                    err_lower = (result.error or "").lower()
                    is_timeout = "timed out" in err_lower or "timeout" in err_lower
                    is_crash = "exit code -" in err_lower or "signal" in err_lower
                    if is_timeout or is_crash:
                        timed_out_combos.add(combo_key)
                        remaining = len(algorithms) - (algorithms.index(algo_id) + 1) if algo_id in algorithms else 0
                        if progress:
                            reason = "TIMEOUT" if is_timeout else f"CRASH ({result.error[:60]})"
                            progress.info(
                                f"  ⚠ {reason}: {algo_name} on {bench}/{graph_name} — "
                                f"skipping {remaining} remaining algorithms for this combo"
                            )
                
                # Enrich result with metadata
                result.graph = graph_name
                result.nodes = graph.nodes
                result.edges = graph.edges
                
                # Cache graph features from first successful benchmark run
                # The extra dict now contains topology features parsed from C++ output
                if result.success and result.extra:
                    features_to_cache = {k: v for k, v in result.extra.items() 
                                        if k in ('degree_variance', 'hub_concentration', 'avg_degree',
                                                'clustering_coefficient', 'avg_path_length', 
                                                'diameter', 'community_count', 'modularity')}
                    if features_to_cache:
                        features_to_cache['nodes'] = graph.nodes
                        features_to_cache['edges'] = graph.edges
                        update_graph_properties(graph_name, features_to_cache, "results")
                
                # Preserve original algorithm name when using label map
                if run_mode == "labelmap":
                    result.algorithm = algo_name
                    result.algorithm_id = algo_id
                
                results.append(result)
                graph_results.append(result)
                completed += 1
                
                if progress and completed % 10 == 0:
                    progress.info(f"  Progress: {completed}/{total_configs}")
        
        # ── Per-graph callback (incremental flush) ──
        if on_graph_complete and graph_results:
            try:
                on_graph_complete(graph_name, graph_results)
            except Exception as e:
                log.warning(f"on_graph_complete callback failed for {graph_name}: {e}")
    
    if skipped > 0:
        log.info(f"Benchmark early-exit: skipped {skipped}/{total_configs} runs due to timeout/crash")
    if skipped_existing > 0:
        log.info(f"Benchmark resume: skipped {skipped_existing} already-completed runs")
    
    # ── Benchmark chained (multi-pass) pre-generated .lo mappings ──
    # Chained orderings like SORT+RABBITORDER_csr have no single algo_id;
    # they exist only as pre-generated .lo mapping files.  We scan for
    # them and run with MAP mode (-o 13:path.lo).
    if use_pregenerated:
        from scripts.lib.core.utils import CHAINED_ORDERINGS
        for graph in graphs:
            graph_name = graph.name
            graph_timeout = compute_adaptive_timeout(graph.edges, timeout)
            mappings_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(graph.path))),
                'mappings', graph_name,
            )
            for canonical, _converter_opts in CHAINED_ORDERINGS:
                pregen_lo = os.path.join(mappings_dir, f"{canonical}.lo")
                if not os.path.isfile(pregen_lo):
                    continue
                mapping_identity_id = mapping_artifact_identity(pregen_lo)
                for bench in benchmarks:
                    if not check_binary_exists(bench, bin_dir):
                        continue
                    combo_key = (graph_name, bench)
                    if combo_key in timed_out_combos:
                        continue
                    # Resume: chained orderings resume on the same shared key.
                    map_spec = f"13:{pregen_lo}"
                    if _condition_key(canonical, map_spec, bench, graph_name,
                                      mapping_identity_id) in skip_existing:
                        skipped_existing += 1
                        continue
                    result = run_benchmark(
                        benchmark=bench,
                        graph_path=graph.path,
                        algorithm=map_spec,  # MAP mode with .lo
                        trials=num_trials,
                        timeout=graph_timeout,
                        bin_dir=bin_dir,
                        log_algorithm=canonical,
                        log_graph_name=graph_name,
                        labeling=labeling,
                        measurement_mode=measurement_mode,
                        threads=threads,
                        mapping_identity_id=mapping_identity_id,
                    )
                    result.algorithm = canonical
                    result.algorithm_id = -1
                    result.graph = graph_name
                    result.nodes = graph.nodes
                    result.edges = graph.edges
                    # Load reorder_time from .time file (written during pregeneration)
                    if result.reorder_time <= 0:
                        result.reorder_time = load_reorder_time_for_algo(
                            graph_name, canonical, mappings_dir=mappings_dir)
                    results.append(result)
    
    # Save the graph properties cache after all benchmarks
    try:
        save_graph_properties_cache("results")
    except Exception as e:
        log.warning(f"Failed to save graph properties cache: {e}")
    
    # Note: compute_speedups returns a dict of {algorithm: {benchmark: speedup}}
    # We don't overwrite results here - callers can use compute_speedups() separately
    
    return results


def run_leiden_variant_comparison(
    graph_path: str,
    benchmarks: List[str] = None,
    trials: int = 3,
    include_baselines: bool = True
) -> List[BenchmarkResult]:
    """
    Run comprehensive comparison of all Leiden variants.
    
    Args:
        graph_path: Path to graph file
        benchmarks: Benchmarks to run (default: pr, bfs, cc)
        trials: Number of trials per config
        include_baselines: Include ORIGINAL, RANDOM, RABBITORDER
        
    Returns:
        List of BenchmarkResult
    """
    if benchmarks is None:
        benchmarks = list(BENCHMARKS)
    
    # Build algorithm list
    algorithms = []
    
    if include_baselines:
        algorithms.extend([str(ALGORITHM_IDS["ORIGINAL"]),
                           str(ALGORITHM_IDS["RANDOM"]),
                           str(ALGORITHM_IDS["RABBITORDER"])])
    
    # GraphBrewOrder
    algorithms.append(str(ALGORITHM_IDS["GraphBrewOrder"]))
    
    # LeidenOrder
    algorithms.append(str(ALGORITHM_IDS["LeidenOrder"]))
    
    return run_benchmark_suite(graph_path, algorithms, benchmarks, trials)


# =============================================================================
# Results Analysis
# =============================================================================

def compute_speedups(
    results: List[BenchmarkResult],
    baseline_algo: str = "RANDOM"
) -> Dict[str, Dict[str, float]]:
    """
    Compute speedups relative to baseline.
    
    Args:
        results: List of benchmark results
        baseline_algo: Baseline algorithm name (partial match)
        
    Returns:
        Dict of {algorithm: {benchmark: speedup}}
    """
    # Find baseline times by (graph, benchmark)
    baselines = {}
    for r in results:
        if baseline_algo in r.algorithm and r.success:
            key = (r.graph, r.benchmark)
            baselines[key] = r.time_seconds
    
    # Compute speedups
    speedups = {}
    for r in results:
        if not r.success or r.time_seconds <= 0:
            continue
        
        key = (r.graph, r.benchmark)
        baseline_time = baselines.get(key, r.time_seconds)
        
        if r.algorithm not in speedups:
            speedups[r.algorithm] = {}
        
        if baseline_time > 0:
            speedups[r.algorithm][r.benchmark] = baseline_time / r.time_seconds
        else:
            speedups[r.algorithm][r.benchmark] = 1.0
    
    return speedups


def format_results_table(
    results: List[BenchmarkResult],
    baseline_algo: str = "RANDOM"
) -> str:
    """Format results as a text table with speedups."""
    speedups = compute_speedups(results, baseline_algo)
    
    # Get unique benchmarks and algorithms
    benchmarks = sorted(set(r.benchmark for r in results if r.success))
    algorithms = sorted(speedups.keys())
    
    if not benchmarks or not algorithms:
        return "No successful results to display"
    
    # Build header
    lines = []
    header = f"{'Algorithm':<35}"
    for b in benchmarks:
        header += f" {b:>8}"
    header += f" {'Avg':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    
    # Build rows
    for algo in algorithms:
        row = f"{algo:<35}"
        algo_speedups = []
        for b in benchmarks:
            s = speedups[algo].get(b, 1.0)
            row += f" {s:>7.2f}x"
            algo_speedups.append(s)
        avg = sum(algo_speedups) / len(algo_speedups) if algo_speedups else 1.0
        row += f" {avg:>7.2f}x"
        lines.append(row)
    
    return "\n".join(lines)


# =============================================================================
# Variant-Aware Benchmarking
# =============================================================================

def run_benchmarks_with_variants(
    graphs: list,
    label_maps: Dict[str, Dict[str, str]],
    benchmarks: List[str],
    bin_dir: str,
    num_trials: int = 3,
    timeout: int = 600,
    weights_dir: str = "",
    update_weights: bool = True,
    progress=None,
    labeling: str = "natural",
    measurement_mode: str = "process",
    threads: int = 0,
) -> List[BenchmarkResult]:
    """
    Run benchmarks with variant-expanded label maps.

    This iterates directly over the algorithm names in label_maps (which include
    variant suffixes like GraphBrewOrder_leiden, RABBITORDER_csr) to ensure the results
    contain the full variant names.

    When using .lo files (MAP mode), loads reorder_time from the corresponding
    .time file instead of parsing from benchmark output.

    The ``labeling`` / ``measurement_mode`` / ``threads`` arguments populate the
    observation condition of every success and failure result so distinct
    conditions never collide, and mapping identity distinguishes direct
    (``ORIGINAL``) inputs from pre-generated MAP inputs.
    """
    from pathlib import Path as _Path

    def load_reorder_time(label_map_path: str) -> float:
        """Load reorder time from .time file corresponding to .lo file."""
        if not label_map_path:
            return 0.0
        time_file = _Path(label_map_path).with_suffix('.time')
        if time_file.exists():
            try:
                return float(time_file.read_text().strip())
            except (ValueError, IOError):
                return 0.0
        return 0.0

    results = []

    # Collect all unique algorithm names from label_maps
    all_algo_names: set = set()
    for graph_maps in label_maps.values():
        all_algo_names.update(graph_maps.keys())

    # Always include ORIGINAL (algo_id=0) - it doesn't need a label map
    all_algo_names.add("ORIGINAL")

    # Sort for consistent ordering (ORIGINAL first, then alphabetically)
    algo_names_sorted = ["ORIGINAL"] + sorted(
        [n for n in all_algo_names if n != "ORIGINAL"]
    )

    total_configs = len(graphs) * len(algo_names_sorted) * len(benchmarks)
    completed = 0

    # Track (graph, benchmark) combos that timed out / crashed —
    # skip remaining algorithms to avoid burning timeout budget
    timed_out_combos: set = set()

    for graph in graphs:
        graph_name = graph.name
        graph_path = graph.path
        graph_label_maps = label_maps.get(graph_name, {})

        # Adaptive timeout based on graph size
        graph_timeout = compute_adaptive_timeout(graph.edges, timeout)

        if progress:
            timeout_note = (
                f", timeout={graph_timeout}s" if graph_timeout != timeout else ""
            )
            progress.info(
                f"Benchmarking: {graph_name} ({graph.size_mb:.1f}MB, "
                f"{graph.edges:,} edges{timeout_note})"
            )

        for bench in benchmarks:
            if not check_binary_exists(bench, bin_dir):
                log.warning(f"Skipping {bench}: binary not found")
                continue

            if progress:
                progress.info(f"  {bench.upper()}:")

            for algo_name in algo_names_sorted:
                combo_key = (graph_name, bench)

                # Early-exit: skip if this graph×benchmark already timed out
                if combo_key in timed_out_combos:
                    algo_id = 0
                    for aid, aname in ALGORITHMS.items():
                        if algo_name == aname or algo_name.startswith(aname + "_"):
                            algo_id = aid
                            break
                    result = BenchmarkResult(
                        graph=graph_name,
                        algorithm=algo_name,
                        algorithm_id=algo_id,
                        benchmark=bench,
                        time_seconds=0.0,
                        success=False,
                        error="SKIPPED: prior algorithm timed out on this graph+benchmark",
                        labeling=labeling,
                        measurement_mode=measurement_mode,
                        threads=threads,
                    )
                    result.nodes = graph.nodes
                    result.edges = graph.edges
                    results.append(result)
                    completed += 1
                    if progress:
                        progress.info(
                            f"    [{completed}/{total_configs}] {algo_name}: SKIPPED (timeout)"
                        )
                    continue

                # Determine algorithm ID from name
                algo_id = 0
                for aid, aname in ALGORITHMS.items():
                    if algo_name == aname or algo_name.startswith(aname + "_"):
                        algo_id = aid
                        break

                # Get label map path for this algorithm (if not ORIGINAL)
                label_map_path = ""
                if algo_name == "ORIGINAL":
                    algo_opt = "0"
                    mapping_identity_id = "direct"
                else:
                    # ── Pre-generated .lo mapping shortcut ───────────────
                    # When a pre-generated .lo mapping exists in the
                    # mappings directory, use MAP mode (-o 13:path.lo) to
                    # skip runtime reorder overhead.
                    mappings_dir = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.dirname(graph_path))),
                        'mappings', graph_name,
                    )
                    pregen_lo = os.path.join(mappings_dir, f"{algo_name}.lo")
                    if os.path.isfile(pregen_lo):
                        result = run_benchmark(
                            benchmark=bench,
                            graph_path=graph_path,
                            algorithm=f"13:{pregen_lo}",  # MAP mode with .lo
                            trials=num_trials,
                            timeout=graph_timeout,
                            bin_dir=bin_dir,
                            log_algorithm=algo_name,
                            log_graph_name=graph_name,
                            labeling=labeling,
                            measurement_mode=measurement_mode,
                            threads=threads,
                            mapping_identity_id=mapping_artifact_identity(
                                pregen_lo
                            ),
                        )
                        # Preserve original algo identity for analysis
                        result.algorithm = algo_name
                        result.algorithm_id = algo_id
                        result.graph = graph_name
                        result.nodes = graph.nodes
                        result.edges = graph.edges
                        # Load reorder_time from .time file (written during pregeneration)
                        if result.reorder_time <= 0:
                            result.reorder_time = load_reorder_time_for_algo(graph_name, algo_name)
                        results.append(result)
                        completed += 1
                        if progress:
                            time_str = (
                                f"{result.time_seconds:.4f}s"
                                if result.success
                                else result.error[:30]
                            )
                            progress.info(
                                f"    [{completed}/{total_configs}] {algo_name}: {time_str} [.lo]"
                            )
                        continue

                    label_map_path = graph_label_maps.get(algo_name, "")
                    if not label_map_path:
                        continue
                    algo_opt = f"13:{label_map_path}"
                    mapping_identity_id = mapping_artifact_identity(
                        label_map_path
                    )

                result = run_benchmark(
                    benchmark=bench,
                    graph_path=graph_path,
                    algorithm=algo_opt,
                    trials=num_trials,
                    timeout=graph_timeout,
                    bin_dir=bin_dir,
                    log_algorithm=algo_name,
                    labeling=labeling,
                    measurement_mode=measurement_mode,
                    threads=threads,
                    mapping_identity_id=mapping_identity_id,
                )

                # Detect timeout or crash
                if not result.success:
                    err_lower = (result.error or "").lower()
                    is_timeout = "timed out" in err_lower or "timeout" in err_lower
                    is_crash = (
                        "exit code -" in err_lower or "signal" in err_lower
                    )
                    if is_timeout or is_crash:
                        timed_out_combos.add(combo_key)
                        reason = (
                            "TIMEOUT"
                            if is_timeout
                            else f"CRASH ({result.error[:60]})"
                        )
                        if progress:
                            progress.info(
                                f"  ⚠ {reason}: {algo_name} on {bench}/{graph_name} — "
                                "skipping remaining algorithms for this combo"
                            )

                # Set the algorithm name to include variant suffix
                result.algorithm = algo_name
                result.algorithm_id = algo_id
                result.graph = graph_name
                result.nodes = graph.nodes
                result.edges = graph.edges

                # Load reorder_time from .time file when using .lo files
                if label_map_path:
                    result.reorder_time = load_reorder_time(label_map_path)

                results.append(result)
                completed += 1

                # Log progress
                time_str = (
                    f"{result.time_seconds:.4f}s"
                    if result.success
                    else result.error[:30]
                )
                if progress:
                    progress.info(
                        f"    [{completed}/{total_configs}] {algo_name}: {time_str}"
                    )

    return results


# =============================================================================
# Standalone CLI
# =============================================================================

def main():
    """CLI for benchmark execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run GraphBrew benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m scripts.lib.pipeline.benchmark --graph graph.mtx -a 0,1,8
    python -m scripts.lib.pipeline.benchmark --graph graph.mtx --leiden-variants
    python -m scripts.lib.pipeline.benchmark --graph graph.mtx -a 0,8,12 --expand
        """
    )
    
    parser.add_argument("--graph", "-g", required=True, help="Graph file path")
    parser.add_argument("-a", "--algorithms", default="0,1,8",
                       help="Comma-separated algorithm options")
    parser.add_argument("-b", "--benchmarks", nargs="+", default=list(BENCHMARKS),
                       help="Benchmarks to run")
    parser.add_argument("-n", "--trials", type=int, default=3, help="Trials per config")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds")
    parser.add_argument("--leiden-variants", action="store_true",
                       help="Run Leiden variant comparison (baselines + GraphBrew + LeidenOrder)")
    parser.add_argument("--expand", action="store_true",
                       help="Expand variant-based algorithms to all variants")
    parser.add_argument("-o", "--output", help="Output JSON file")
    
    args = parser.parse_args()
    
    # Verify graph exists
    graph_path = Path(args.graph)
    if not graph_path.exists():
        log.error(f"Graph not found: {graph_path}")
        return 1
    
    # Run benchmarks
    if args.leiden_variants:
        log.info("Running Leiden variant comparison...")
        results = run_leiden_variant_comparison(
            str(graph_path),
            benchmarks=args.benchmarks,
            trials=args.trials
        )
    else:
        algorithms = args.algorithms.split(",")
        
        results = run_benchmark_suite(
            str(graph_path),
            algorithms=algorithms,
            benchmarks=args.benchmarks,
            trials=args.trials,
            timeout=args.timeout
        )
    
    # Display results
    print("\n" + "=" * 70)
    print("Results (speedup vs RANDOM)")
    print("=" * 70)
    print(format_results_table(results))
    
    # Save to JSON
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = get_results_file("benchmark")
    
    save_json([r.to_dict() for r in results], output_path)
    print(f"\nResults saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())


# =============================================================================
# Fresh Benchmark Runner (merged from benchmark_runner.py)
# =============================================================================

# Complexity guards: skip algorithms that are too slow on large graphs
_FRESH_ALGO_NODE_LIMITS: Dict[int, int] = {
    9: 500_000,     # GOrder: O(n*m*w) reorder
    12: 500_000,    # GraphBrewOrder: slow community detection
    10: 2_000_000,  # COrder: slow on very large graphs
}


def discover_sg_graphs(
    graphs_dir: str = None,
    graph_names: List[str] = None,
) -> List[Tuple[str, str, int]]:
    """Discover .sg graph files and their node counts.

    Returns list of (graph_name, sg_path, node_count) sorted by node count.
    """
    gdir = Path(graphs_dir or GRAPHS_DIR)
    results = []
    for sg_path in sorted(gdir.glob("*/*.sg")):
        name = sg_path.parent.name
        if graph_names and name not in graph_names:
            continue
        # Read node count from central GraphPropsStore (no per-graph features.json)
        from scripts.lib.ml.features import get_graph_properties
        nodes = get_graph_properties(name).get("nodes", 0)
        results.append((name, str(sg_path), nodes))
    results.sort(key=lambda x: x[2])
    return results


def run_fresh_benchmarks(
    graphs_dir: str = None,
    graph_names: List[str] = None,
    benchmarks: List[str] = None,
    algos: List[int] = None,
    trials: int = 3,
    timeout: int = TIMEOUT_BENCHMARK,
    bin_dir: str = None,
    output_file: str = None,
) -> List[dict]:
    """Run all AdaptiveOrder-eligible algorithms on all .sg graphs.

    For each (graph, benchmark, algorithm) combination:
    1. Discovers .sg graphs (or uses the provided list)
    2. Runs every combination with complexity guards
    3. Saves results to JSON

    Args:
        graphs_dir: Directory containing graph subdirectories with .sg files
        graph_names: Optional list of specific graph names to benchmark
        benchmarks: List of benchmark types (default: all)
        algos: List of algorithm IDs (default: all eligible)
        trials: Number of trials per combination
        timeout: Timeout per benchmark invocation in seconds
        bin_dir: Directory containing benchmark binaries
        output_file: Path to save JSON results

    Returns:
        List of result dicts.
    """
    benchmarks = benchmarks or list(BENCHMARKS)
    algos = algos or ELIGIBLE_ALGORITHMS
    bin_dir = str(bin_dir or BIN_DIR)
    if output_file is None:
        output_file = str(Path(RESULTS_DIR) / "benchmark_fresh.json")

    graphs = discover_sg_graphs(graphs_dir, graph_names)
    if not graphs:
        log.error("No .sg graphs found")
        return []

    log.info(f"Found {len(graphs)} graphs, {len(algos)} algos, {len(benchmarks)} benchmarks")

    entries: List[dict] = []
    total = 0
    failed = 0

    for graph_name, sg_path, nodes in graphs:
        print(f"\n{'='*64}")
        print(f"=== {graph_name} ({nodes:,} nodes) ===")
        print(f"{'='*64}")

        for bench in benchmarks:
            binary = os.path.join(bin_dir, bench)
            if not os.path.isfile(binary):
                continue

            for algo_id in algos:
                algo_name = canonical_algo_key(algo_id)

                node_limit = _FRESH_ALGO_NODE_LIMITS.get(algo_id)
                if node_limit and nodes > node_limit:
                    continue

                sys.stdout.write(f"  {bench}/{algo_name}... ")
                sys.stdout.flush()

                opt = algo_converter_opt(algo_id)
                cmd = [binary, "-f", sg_path, "-o", opt, "-n", str(trials)]
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True,
                                            timeout=timeout)
                    if result.returncode != 0:
                        print("ERROR")
                        failed += 1
                        continue
                    avg_time, reorder_time, timing = (
                        parse_benchmark_output(result.stdout)
                    )
                    if avg_time > 0:
                        print(f"{avg_time}s (reorder: {reorder_time}s)")
                        entries.append({
                            "graph": graph_name,
                            "algorithm": algo_name,
                            "algorithm_id": algo_id,
                            "benchmark": bench,
                            "time_seconds": avg_time,
                            "reorder_time": reorder_time,
                            "representation_build_time": timing.get(
                                "representation_build_time", 0.0
                            ),
                            "reorder_core_time": timing.get(
                                "reorder_core_time", 0.0
                            ),
                            "reorder_validation_time": timing.get(
                                "reorder_validation_time", 0.0
                            ),
                            "reorder_apply_time": timing.get(
                                "reorder_apply_time", 0.0
                            ),
                            "total_preprocessing_time": timing.get(
                                "total_preprocessing_time", 0.0
                            ),
                            "trials": trials,
                            "success": True,
                        })
                        total += 1
                    else:
                        print("PARSE ERROR")
                        failed += 1
                except subprocess.TimeoutExpired:
                    print("TIMEOUT")
                    failed += 1
                except Exception:
                    print("ERROR")
                    failed += 1

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"\nDone. {total} entries saved to {output_file} ({failed} failed)")
    return entries
