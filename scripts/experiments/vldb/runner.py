#!/usr/bin/env python3
"""
GraphBrew Frozen Evaluation Runner.

Reproduces all figures and tables from the paper. Each experiment
dumps structured JSON results; a final step generates publication-
ready figures (PNG/PDF) and LaTeX table snippets.

Usage:
    # Full reproducibility run (experiments + figures):
    python scripts/experiments/vldb/runner.py --all --graph-dir /data/graphs

    # Experiments only (no figure generation):
    python scripts/experiments/vldb/runner.py --all --graph-dir /data/graphs --no-figures

    # Figures only (from previously saved results):
    python scripts/experiments/vldb/runner.py --figures-only

    # Preview mode (small graphs, 1 trial, 2 benchmarks):
    python scripts/experiments/vldb/runner.py --all --preview \
        --graph-dir /data/graphs --artifact-root /data/artifacts \
        --threads 4 --cpu-list 24-27

    # Dry run (print commands without executing):
    python scripts/experiments/vldb/runner.py --all --dry-run
"""

from __future__ import annotations

import argparse
import fcntl
import filecmp
import json
import logging
import math
import os
import platform
import re
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.vldb.config import (
    ADAPTIVE_CPU_EXPANSION_GRAPHS,
    ABLATION_CONFIGS,
    ALGORITHM_GRAPH_EXCLUSION_EVIDENCE,
    ALGORITHM_GRAPH_EXCLUSIONS,
    ALL_ALGORITHMS,
    BASELINE_ALGORITHMS,
    BC_SOURCE_ITERATIONS,
    BENCHMARKS,
    BENCHMARKS_PREVIEW,
    BIN_DIR,
    BIN_SIM_DIR,
    BIN_WORK_DIR,
    CACHE_ALGORITHM_KEYS,
    CACHE_ALGORITHM_KEYS_PREVIEW,
    CACHE_GRAPH_NAMES,
    CACHE_SIZES,
    CACHE_SIZES_PREVIEW,
    CACHE_TRIALS,
    CACHE_PR_ITERATIONS,
    CHAINED_ORDERINGS,
    COMPOSITION_P0_ALGORITHM_KEYS,
    COMPOSITION_P0_CONFIGS,
    EVALUATION_BASELINES,
    E2E_REUSE_COUNTS,
    EVAL_GRAPHS,
    EVAL_GRAPHS_64GB,
    EVAL_GRAPHS_LOCAL,
    FIGURES_DIR,
    GRAPH_TYPE_GROUPS,
    GRAPHBREW_VARIANTS,
    COMPOSE_VARIANTS,
    DIAGNOSTIC_CONFIGS,
    DUAL_ARM_S0_CONFIGS,
    DUAL_ARM_S2_CONFIGS,
    PREVIEW_GRAPHS,
    PR_CONVERGENCE_MAX_ITERATIONS,
    PR_FIXED_ITERATIONS,
    PR_TOLERANCE,
    RANDOM_BASELINE_SEED,
    REORDER_SEMANTICS_VERSION,
    PROMOTED_GORDER_GRAPHS,
    PAPER_ARTIFACT_ROOT,
    PAPER_GRAPH_ROOT,
    RESULTS_DIR,
    REORDER_TRIALS_FULL,
    REORDER_TRIALS_PREVIEW,
    REORDER_TIMEOUT_FULL,
    REORDER_TIMING_REUSE_GRAPHS,
    REORDER_TIMING_ANCHOR_ALGOS,
    RABBIT_MAPPING_DRAWS,
    SSSP_DELTA_CANDIDATES,
    SSSP_POLICY,
    SSSP_POLICY_PATH,
    SSSP_POLICY_SELECTION_RULE_ID,
    SSSP_SELECTION_RULE_ID,
    SSSP_TUNING_ORDER_POLICY,
    SSSP_TUNING_PRACTICAL_TIE_RATIO,
    SSSP_TUNING_REPLICATES,
    SSSP_TUNING_REPEATS,
    SSSP_TUNING_SNAPSHOT_PATH,
    SSSP_TUNING_SOURCES,
    SSSP_TUNING_TRIALS,
    SSSP_TUNING_T_CRITICAL_95_DF8,
    SSSP_WEIGHT_SCHEME,
    SCALABILITY_REPEATS,
    SCALABILITY_ALGORITHM_KEYS,
    SCALABILITY_GRAPH_NAMES,
    TABLES_DIR,
    THREAD_COUNTS,
    TIMEOUT_FULL,
    TIMEOUT_PREVIEW,
    TRIALS_FULL,
    TRIALS_PREVIEW,
    VERIFICATION_TIMEOUT_MULTIPLIER,
    VLDB_ROOT,
    VLDB_GRAPH_SOURCES,
    algorithm_exclusion_reason,
    get_converter_flags,
)
from scripts.lib.pipeline.reorder_config import (
    GRAPHBREW_EFFECTIVE_CONFIG_PREFIX,
    GRAPHBREW_REALIZED_CONFIG_PREFIX,
    expected_graphbrew_config,
    extract_graphbrew_order_specs,
    graphbrew_schedule_sensitive,
    parse_graphbrew_effective_configs,
    parse_graphbrew_realized_configs,
    validate_graphbrew_effective_configs,
    validate_graphbrew_realized_configs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vldb_paper")

_RUNTIME_THREADS: Optional[int] = None
_RUNTIME_CPU_LIST: Optional[str] = None
_RUNTIME_ENV: dict[str, str] = {}
_PREVIEW_MODE = False
_CACHE_MODE = "ultrafast"
_CACHE_SAMPLE_RATE = 64
_CACHE_ALL_ALGORITHMS = False
_CACHE_SIZE_OVERRIDE: Optional[list[int]] = None
_DRY_RUN_MODE = False
_CAMPAIGN_ID: Optional[str] = None
_ACTIVE_VERIFICATION_GATE_ID: Optional[str] = None
_FILE_FINGERPRINT_CACHE: dict[tuple[str, int, int], dict[str, int | str]] = {}
_FILE_CRC32_CACHE: dict[tuple[str, int, int], str] = {}
_ALGORITHM_FILTER: Optional[set[str]] = None
_MEASUREMENT_GENERATION_OVERRIDE: Optional[str] = None
_ALGORITHM_ENV_KEYS = (
    "GORDER_FAST_BATCH",
    "GORDER_WINDOW",
    "RABBIT_RESOLUTION",
)


# ============================================================================
# Helpers
# ============================================================================


def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _short_id(payload: Any) -> str:
    """Return a compact non-cryptographic identifier for internal indexing."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return f"{zlib.crc32(encoded) & 0xffffffff:08x}"


def _file_crc32(path: str | Path) -> str:
    file_path = Path(path)
    stat = file_path.stat()
    key = (str(file_path.resolve()), stat.st_size, stat.st_mtime_ns)
    cached = _FILE_CRC32_CACHE.get(key)
    if cached is not None:
        return cached
    value = 0
    with file_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value = zlib.crc32(block, value)
    digest = f"{value & 0xffffffff:08x}"
    _FILE_CRC32_CACHE[key] = digest
    return digest


def ensure_dir(p: Path) -> Path:
    if _DRY_RUN_MODE:
        return p
    p.mkdir(parents=True, exist_ok=True)
    return p


def _expand_cpu_list(spec: Optional[str]) -> list[int]:
    """Expand a taskset-style CPU list such as ``0-3,8``."""
    if not spec:
        return []
    cpus: list[int] = []
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start, end = int(start_s), int(end_s)
            if start < 0 or end < start:
                raise ValueError(f"Invalid CPU range: {token}")
            cpus.extend(range(start, end + 1))
        else:
            cpu = int(token)
            if cpu < 0:
                raise ValueError(f"Invalid CPU id: {token}")
            cpus.append(cpu)
    if len(cpus) != len(set(cpus)):
        raise ValueError(f"CPU list contains duplicates: {spec}")
    return cpus


def _format_cpu_list(cpus: list[int]) -> str:
    return ",".join(str(cpu) for cpu in cpus)


def configure_runtime_policy(threads: int, cpu_list: Optional[str]) -> None:
    """Set the base OpenMP and CPU-affinity policy for all experiment commands."""
    global _RUNTIME_THREADS, _RUNTIME_CPU_LIST, _RUNTIME_ENV
    if threads < 1:
        raise ValueError("threads must be positive")
    cpus = _expand_cpu_list(cpu_list)
    if cpus and len(cpus) < threads:
        raise ValueError(
            f"CPU list {cpu_list!r} provides {len(cpus)} CPUs for {threads} threads"
        )
    if cpu_list and shutil.which("taskset") is None:
        raise RuntimeError("taskset is required when --cpu-list is used")
    _RUNTIME_THREADS = threads
    _RUNTIME_CPU_LIST = cpu_list
    _RUNTIME_ENV = {
        "OMP_NUM_THREADS": str(threads),
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
        "OMP_DYNAMIC": "FALSE",
        "GRAPHBREW_DB_DIR": "",
        "GRAPHBREW_TOPOLOGY_ANALYSIS": "0",
    }


def configure_cache_policy(
    *,
    preview: bool,
    mode: str,
    sample_rate: int,
    all_algorithms: bool,
    sizes_kib: Optional[list[int]] = None,
) -> None:
    """Set the cache-simulation policy used by experiment 1."""
    global _PREVIEW_MODE, _CACHE_MODE, _CACHE_SAMPLE_RATE
    global _CACHE_ALL_ALGORITHMS, _CACHE_SIZE_OVERRIDE
    if mode not in {"accurate", "fast", "ultrafast", "sampled"}:
        raise ValueError(f"Unsupported cache mode: {mode}")
    if sample_rate < 1:
        raise ValueError("cache sample rate must be positive")
    _PREVIEW_MODE = preview
    _CACHE_MODE = mode
    _CACHE_SAMPLE_RATE = sample_rate
    _CACHE_ALL_ALGORITHMS = all_algorithms
    if sizes_kib:
        if any(size <= 0 for size in sizes_kib):
            raise ValueError("cache sizes must be positive")
        _CACHE_SIZE_OVERRIDE = [size * 1024 for size in sizes_kib]
    else:
        _CACHE_SIZE_OVERRIDE = None


def configure_algorithm_filter(algorithms: Optional[list[str]]) -> None:
    """Restrict rapid runs to exact canonical algorithm keys."""
    global _ALGORITHM_FILTER
    if not algorithms:
        _ALGORITHM_FILTER = None
        return
    known = set(ALL_ALGORITHMS)
    known.update(config["algo"] for config in ABLATION_CONFIGS)
    known.update(config["algo"] for config in DIAGNOSTIC_CONFIGS)
    known.update(config["algo"] for config in DUAL_ARM_S0_CONFIGS)
    known.update(config["algo"] for config in DUAL_ARM_S2_CONFIGS)
    known.update(config["algo"] for config in COMPOSITION_P0_CONFIGS)
    known.update(f"chain:{name}" for name, _flags in CHAINED_ORDERINGS)
    unknown = sorted(set(algorithms) - known)
    if unknown:
        raise ValueError(
            f"Unknown algorithm key(s): {', '.join(unknown)}. "
            f"Valid keys include: {', '.join(list(ALL_ALGORITHMS)[:12])}"
        )
    _ALGORITHM_FILTER = set(algorithms)


def resolve_benchmark_policy(
    default_benchmarks: list[str],
    default_trials: int,
    requested_benchmarks: Optional[list[str]],
    requested_trials: Optional[int],
) -> tuple[list[str], int]:
    benchmarks = list(default_benchmarks)
    trials = int(default_trials)
    if requested_benchmarks is not None:
        unknown = sorted(set(requested_benchmarks) - set(BENCHMARKS))
        if (
            not requested_benchmarks
            or len(requested_benchmarks) != len(set(requested_benchmarks))
            or unknown
        ):
            raise ValueError(
                "Invalid benchmark override: "
                + ", ".join(unknown or requested_benchmarks)
            )
        benchmarks = list(requested_benchmarks)
    if requested_trials is not None:
        if requested_trials <= 0:
            raise ValueError("Trial override must be positive")
        trials = int(requested_trials)
    return benchmarks, trials


def configure_execution_mode(*, dry_run: bool) -> None:
    global _DRY_RUN_MODE
    _DRY_RUN_MODE = dry_run


def configure_measurement_generation(generation: Optional[str]) -> None:
    global _MEASUREMENT_GENERATION_OVERRIDE
    if generation is not None and re.fullmatch(
        r"[A-Za-z0-9_.-]{1,128}", generation,
    ) is None:
        raise ValueError("Invalid measurement generation identifier")
    _MEASUREMENT_GENERATION_OVERRIDE = generation


def configure_artifact_root(root: str | Path) -> None:
    """Route paper results, mappings, and run sidecars to one artifact root."""
    global RESULTS_DIR, FIGURES_DIR, TABLES_DIR, MAPPINGS_DIR, KERNEL_RUNS_DIR
    global _CAMPAIGN_ID, _ACTIVE_VERIFICATION_GATE_ID
    artifact_root = Path(root).resolve()
    os.environ["GRAPHBREW_VLDB_ROOT"] = str(artifact_root)
    RESULTS_DIR = artifact_root / "vldb_paper"
    FIGURES_DIR = RESULTS_DIR / "figures"
    TABLES_DIR = RESULTS_DIR / "tables"
    MAPPINGS_DIR = artifact_root / "vldb_mappings"
    KERNEL_RUNS_DIR = artifact_root / "vldb_runs"
    _CAMPAIGN_ID = None
    _ACTIVE_VERIFICATION_GATE_ID = None


def _effective_env(env: Optional[dict] = None) -> dict[str, str]:
    return {**_RUNTIME_ENV, **{str(k): str(v) for k, v in (env or {}).items()}}


def _algorithm_environment() -> dict[str, str]:
    return {
        key: os.environ[key]
        for key in _ALGORITHM_ENV_KEYS
        if key in os.environ
    }


def measurement_policy_id(
    kind: str,
    *,
    trials: int,
    env: Optional[dict] = None,
    cpu_list: Optional[str] = None,
    executable: Optional[str | Path] = None,
    extra: Optional[dict] = None,
) -> str:
    """Identify the semantic runtime policy for one measured cell."""
    executable_identity: Optional[dict[str, Any]] = None
    if executable is not None:
        executable_path = Path(executable)
        executable_identity = {
            "name": executable_path.name,
            **_graph_fingerprint(executable_path),
        }
    payload = {
        "schema": "measurement_policy/v2",
        "kind": kind,
        "trials": trials,
        "env": _effective_env(env),
        "cpu_list": cpu_list if cpu_list is not None else _RUNTIME_CPU_LIST,
        "algorithm_env": _algorithm_environment(),
        "executable": executable_identity,
        "timing_machine": (
            None if kind == "cache" else timing_machine_metadata()
        ),
        "extra": extra or {},
    }
    return _short_id(payload)


def measurement_cohort_id(
    kind: str,
    *,
    trials: int,
    extra: Optional[dict] = None,
) -> str:
    """Identify one coherent experiment campaign across all of its cells."""
    explicit_generation = (
        _MEASUREMENT_GENERATION_OVERRIDE
        or os.environ.get("GRAPHBREW_MEASUREMENT_GENERATION")
    )
    payload = {
        "schema": "measurement_cohort/v2",
        "kind": kind,
        "trials": trials,
        "base_env": _RUNTIME_ENV,
        "base_cpu_list": _RUNTIME_CPU_LIST,
        "algorithm_env": _algorithm_environment(),
        "preview": _PREVIEW_MODE,
        "cache_mode": _CACHE_MODE,
        "cache_sample_rate": _CACHE_SAMPLE_RATE,
        "timing_machine": (
            None if kind == "cache" else timing_machine_metadata()
        ),
        "explicit_generation": explicit_generation,
        "extra": extra or {},
    }
    return _short_id(payload)


def _mapping_identity_id(mapping_identity: dict[str, Any]) -> str:
    return _short_id(mapping_identity)


def _e2e_join_context_id() -> str:
    """Bind kernel and reorder cohorts without requiring one process run."""
    payload = {
        "schema": "e2e_join_context/v1",
        "runtime_env": _RUNTIME_ENV,
        "cpu_list": _RUNTIME_CPU_LIST,
        "preview": _PREVIEW_MODE,
        "algorithm_env": _algorithm_environment(),
        "timing_machine": timing_machine_metadata(),
    }
    return _short_id(payload)


def sssp_policy_for_graph(graph_name: str) -> dict[str, Any]:
    policy = SSSP_POLICY.get(graph_name)
    if policy is None:
        raise RuntimeError(
            f"No frozen weighted-SSSP policy for {graph_name}; run the "
            "canonical SSSP delta tuner, complete layered review, then run "
            "it with --freeze-sssp-policy"
        )
    required = {
        "weight_scheme",
        "weight_checksum",
        "delta",
        "conversion_policy_id",
    }
    missing = required - set(policy)
    if missing:
        raise RuntimeError(
            f"Incomplete weighted-SSSP policy for {graph_name}: "
            f"missing {sorted(missing)}"
        )
    if policy["weight_scheme"] != SSSP_WEIGHT_SCHEME:
        raise RuntimeError(
            f"Unexpected SSSP weight scheme for {graph_name}: "
            f"{policy['weight_scheme']}"
        )
    if (
        type(policy["delta"]) is not int
        or policy["delta"] <= 0
        or policy["delta"] not in SSSP_DELTA_CANDIDATES
    ):
        raise RuntimeError(f"Invalid SSSP delta for {graph_name}")
    checksum = policy["weight_checksum"]
    if (
        type(checksum) is not str
        or re.fullmatch(r"[0-9a-f]{32}", checksum) is None
    ):
        raise RuntimeError(f"Invalid SSSP weight checksum for {graph_name}")
    return dict(policy)


_SSSP_SNAPSHOT_IDENTITY_KEYS = (
    "weight_scheme",
    "delta_candidates",
    "trials_per_candidate",
    "trials_per_invocation",
    "source_count",
    "repeats_per_source",
    "invocation_replicates",
    "candidate_order_policy",
    "runtime_env",
    "cpu_list",
    "measurement_protocol_id",
    "selection_rule_id",
    "practical_tie_ratio",
    "paired_t_critical",
)


def _sssp_policy_validation_snapshot(
    artifact: dict[str, Any],
) -> dict[str, Any]:
    """Project a reviewed tuning artifact to runtime validation fields."""
    snapshot = {
        "schema": artifact.get("schema"),
        "artifact_kind": "policy-validation-snapshot",
        "preview": artifact.get("preview"),
        "eligible_for_freeze": artifact.get("eligible_for_freeze"),
        **{
            key: artifact.get(key)
            for key in _SSSP_SNAPSHOT_IDENTITY_KEYS
        },
        "graphs": {
            name: {"graph_info": row.get("graph_info")}
            for name, row in artifact.get("graphs", {}).items()
        },
        "recommendations": artifact.get("recommendations"),
    }
    return snapshot


def _validate_sssp_policy_source() -> dict[str, Any]:
    if SSSP_POLICY_SELECTION_RULE_ID != SSSP_SELECTION_RULE_ID:
        raise RuntimeError(
            "Frozen SSSP policy is missing its reviewed selection rule"
        )
    tuning_path = SSSP_TUNING_SNAPSHOT_PATH
    if not tuning_path.is_file():
        raise RuntimeError(
            f"Missing SSSP tuning source artifact: {tuning_path}"
        )
    artifact = json.loads(tuning_path.read_text())
    if (
        artifact.get("schema") != "sssp_delta_tuning/v2"
        or artifact.get("artifact_kind")
            != "policy-validation-snapshot"
        or
        artifact.get("preview") is not False
        or artifact.get("eligible_for_freeze") is not True
    ):
        raise RuntimeError("SSSP tuning snapshot is not final-path eligible")
    if artifact.get("selection_rule_id") != SSSP_SELECTION_RULE_ID:
        raise RuntimeError("SSSP tuning selection rule does not match policy")
    expected_identity = {
        "weight_scheme": SSSP_WEIGHT_SCHEME,
        "delta_candidates": SSSP_DELTA_CANDIDATES,
        "trials_per_candidate": SSSP_TUNING_TRIALS,
        "trials_per_invocation":
            SSSP_TUNING_SOURCES * SSSP_TUNING_REPEATS,
        "source_count": SSSP_TUNING_SOURCES,
        "repeats_per_source": SSSP_TUNING_REPEATS,
        "invocation_replicates": SSSP_TUNING_REPLICATES,
        "candidate_order_policy": SSSP_TUNING_ORDER_POLICY,
        "practical_tie_ratio": SSSP_TUNING_PRACTICAL_TIE_RATIO,
        "paired_t_critical": SSSP_TUNING_T_CRITICAL_95_DF8,
        "runtime_env": _effective_env(),
        "cpu_list": _RUNTIME_CPU_LIST,
        "measurement_protocol_id": _sssp_measurement_protocol_id(),
    }
    mismatches = {
        key: (artifact.get(key), value)
        for key, value in expected_identity.items()
        if artifact.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Frozen SSSP policy runtime identity mismatch: {mismatches}"
        )
    artifact_policy_bytes = json.dumps(
        artifact.get("recommendations"),
        sort_keys=True,
        separators=(",", ":"),
    )
    frozen_policy_bytes = json.dumps(
        SSSP_POLICY,
        sort_keys=True,
        separators=(",", ":"),
    )
    if artifact_policy_bytes != frozen_policy_bytes:
        raise RuntimeError(
            "Frozen SSSP policy differs from reviewed tuning recommendations"
        )
    return artifact


def _validate_sssp_policy_graph(
    graph_name: str,
    graph_path: str | Path,
) -> dict[str, Any]:
    graph_path = Path(graph_path)
    artifact = _validate_sssp_policy_source()
    policy = sssp_policy_for_graph(graph_name)
    provenance_path = _graph_provenance_path(graph_path)
    if not _graph_provenance_valid(
        graph_path,
        graph_name=graph_name,
    ):
        raise RuntimeError(
            f"Weighted SSSP requires current canonical graph provenance: "
            f"{graph_path}"
        )
    provenance = json.loads(provenance_path.read_text())
    tuned_graph = artifact.get("graphs", {}).get(graph_name, {})
    current_graph_info = _serialized_graph_info(graph_path)
    if tuned_graph.get("graph_info") != current_graph_info:
        raise RuntimeError(
            f"Frozen weighted-SSSP graph dimensions mismatch for "
            f"{graph_name}: expected {tuned_graph.get('graph_info')}, "
            f"found {current_graph_info}"
        )
    actual = {
        "conversion_policy_id":
            provenance.get("conversion_policy_id"),
    }
    mismatches = {
        key: (policy.get(key), value)
        for key, value in actual.items()
        if policy.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Frozen weighted-SSSP graph identity mismatch for "
            f"{graph_name}: {mismatches}"
        )
    return policy


def _benchmark_runtime_policy(
    benchmark: str,
    graph_name: str,
    graph_path: str | Path,
    *,
    fixed_pr_iterations: int = PR_FIXED_ITERATIONS,
) -> dict[str, Any]:
    policy: dict[str, Any] = {"schema": "benchmark_runtime/v1"}
    if benchmark == "sssp":
        policy["sssp"] = _validate_sssp_policy_graph(
            graph_name, graph_path,
        )
    if benchmark in {"pr", "pr_spmv"}:
        policy["pagerank"] = {
            "mode": "fixed-work",
            "iterations": fixed_pr_iterations,
            "tolerance": PR_TOLERANCE,
        }
    elif benchmark in {
        "pr_convergence", "pr_spmv_convergence",
    }:
        policy["pagerank"] = {
            "mode": "convergence",
            "max_iterations": PR_CONVERGENCE_MAX_ITERATIONS,
            "tolerance": PR_TOLERANCE,
        }
    return policy


def preflight_benchmark_policies(
    graphs: list[dict],
    benchmarks: list[str],
    graph_dir: str,
) -> None:
    if "sssp" not in benchmarks:
        return
    for graph in graphs:
        graph_name = graph["name"]
        _validate_sssp_policy_graph(
            graph_name,
            resolve_graph_path(graph_name, graph_dir),
        )


def _sssp_measurement_protocol_id(
    *,
    trials: int = SSSP_TUNING_TRIALS,
    sources: int = SSSP_TUNING_SOURCES,
    repeats: int = SSSP_TUNING_REPEATS,
    replicates: int = SSSP_TUNING_REPLICATES,
    candidates: Optional[list[int]] = None,
) -> str:
    payload = {
        "schema": "sssp_measurement_protocol/v2",
        "weight_scheme": SSSP_WEIGHT_SCHEME,
        "delta_candidates": candidates or SSSP_DELTA_CANDIDATES,
        "trials": trials,
        "sources": sources,
        "repeats": repeats,
        "replicates": replicates,
        "candidate_order_policy": SSSP_TUNING_ORDER_POLICY,
    }
    return _short_id(payload)


def _sssp_candidate_execution_orders(
    candidates: list[int],
    replicates: int,
) -> list[list[int]]:
    if not candidates or replicates <= 0:
        raise ValueError("SSSP candidate schedule requires positive inputs")
    return [
        (
            candidates[(replicate * len(candidates)) // replicates:]
            + candidates[:(replicate * len(candidates)) // replicates]
        )
        for replicate in range(replicates)
    ]


def _select_sssp_delta(
    candidate_rows: list[dict],
) -> tuple[dict, dict, list[int]]:
    if not candidate_rows:
        raise RuntimeError("Cannot select from an empty SSSP candidate set")
    fastest = min(
        candidate_rows,
        key=lambda row: (row["median_time"], row["delta"]),
    )
    baseline_times = fastest.get("trial_times")
    if not isinstance(baseline_times, list) or not baseline_times:
        raise RuntimeError("Fastest SSSP candidate has no block times")
    tie_set: list[int] = []
    for row in candidate_rows:
        trial_times = row.get("trial_times")
        if (
            not isinstance(trial_times, list)
            or len(trial_times) != len(baseline_times)
        ):
            raise RuntimeError("SSSP candidates have incompatible block times")
        paired_slowdowns = [
            candidate - baseline
            for candidate, baseline in zip(
                trial_times, baseline_times,
            )
        ]
        paired_mean = statistics.fmean(paired_slowdowns)
        paired_stddev = (
            statistics.stdev(paired_slowdowns)
            if len(paired_slowdowns) > 1 else 0.0
        )
        paired_se = (
            paired_stddev / math.sqrt(len(paired_slowdowns))
            if paired_slowdowns else 0.0
        )
        statistically_indistinguishable = (
            paired_mean
            <= SSSP_TUNING_T_CRITICAL_95_DF8 * paired_se
        )
        within_practical_band = (
            row["median_time"]
            <= (
                SSSP_TUNING_PRACTICAL_TIE_RATIO
                * fastest["median_time"]
            )
        )
        in_tie_set = (
            statistically_indistinguishable
            and within_practical_band
        )
        row.update({
            "paired_block_slowdowns": paired_slowdowns,
            "paired_mean_slowdown": paired_mean,
            "paired_stddev_slowdown": paired_stddev,
            "paired_standard_error": paired_se,
            "paired_t_statistic": (
                paired_mean / paired_se
                if paired_se > 0 else
                0.0 if paired_mean == 0 else None
            ),
            "paired_one_sided_95_lower_bound":
                paired_mean
                - SSSP_TUNING_T_CRITICAL_95_DF8 * paired_se,
            "paired_tie_threshold":
                SSSP_TUNING_T_CRITICAL_95_DF8 * paired_se,
            "statistically_indistinguishable":
                statistically_indistinguishable,
            "within_practical_band": within_practical_band,
            "in_tie_set": in_tie_set,
        })
        if in_tie_set:
            tie_set.append(row["delta"])
    if not tie_set:
        raise RuntimeError("SSSP selection produced an empty tie set")
    winner = next(
        row for row in candidate_rows
        if row["delta"] == min(tie_set)
    )
    return fastest, winner, sorted(tie_set)


def _kernel_policy_ids(
    *,
    graph_path: str,
    kind: str,
    trials: int,
    executable: str | Path,
    env: Optional[dict] = None,
    cpu_list: Optional[str] = None,
    extra: Optional[dict] = None,
    cohort_extra: Optional[dict] = None,
    benchmark_name: Optional[str] = None,
    fixed_pr_iterations: int = PR_FIXED_ITERATIONS,
) -> tuple[str, str]:
    benchmark = benchmark_name or Path(executable).name
    graph_name = Path(graph_path).stem
    runtime_policy = _benchmark_runtime_policy(
        benchmark,
        graph_name,
        graph_path,
        fixed_pr_iterations=fixed_pr_iterations,
    )
    cohort_id = measurement_cohort_id(
        kind,
        trials=trials,
        extra=cohort_extra,
    )
    policy_id = measurement_policy_id(
        kind,
        trials=trials,
        env=env,
        cpu_list=cpu_list,
        executable=executable,
        extra={
            "graph": graph_name,
            "benchmark_runtime_policy": runtime_policy,
            **(extra or {}),
        },
    )
    return cohort_id, policy_id


def _cpu_list_for_threads(threads: int) -> Optional[str]:
    """Return the first ``threads`` CPUs from the configured affinity mask."""
    cpus = _expand_cpu_list(_RUNTIME_CPU_LIST)
    if not cpus:
        return None
    if threads > len(cpus):
        raise ValueError(
            f"Cannot place {threads} threads on configured CPU list {_RUNTIME_CPU_LIST}"
        )
    return _format_cpu_list(cpus[:threads])


def run_cmd(
    cmd: list[str],
    dry_run: bool = False,
    timeout: int = 3600,
    env: Optional[dict] = None,
    cpu_list: Optional[str] = None,
    failure_details: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Run a command and return stdout, or None on failure."""
    effective_cpu_list = cpu_list if cpu_list is not None else _RUNTIME_CPU_LIST
    effective_cmd = [str(c) for c in cmd]
    if effective_cpu_list:
        effective_cmd = ["taskset", "-c", effective_cpu_list, *effective_cmd]
    cmd_str = " ".join(effective_cmd)
    log.info(f"  CMD: {cmd_str}")
    effective_env = _effective_env(env)
    if effective_env:
        shown = " ".join(
            f"{key}={effective_env[key]}"
            for key in sorted(effective_env)
            if (
                key.startswith("OMP_")
                or key.startswith("CACHE_")
                or key.startswith("ECG_")
            )
        )
        if shown:
            log.info(f"  ENV: {shown}")
    if dry_run:
        return ""
    merged_env = {**os.environ, **effective_env}
    started = time.monotonic()
    try:
        result = subprocess.run(
            effective_cmd, timeout=timeout, capture_output=True, text=True,
            env=merged_env,
        )
        if result.returncode != 0:
            if failure_details is not None:
                failure_details.update({
                    "failure_mode": "nonzero-exit",
                    "returncode": result.returncode,
                    "elapsed_seconds": time.monotonic() - started,
                    "stderr": result.stderr[:1000],
                })
            log.error(f"  FAILED (rc={result.returncode}): {result.stderr[:500]}")
            if result.returncode == 3:
                raise RuntimeError(
                    f"Semantic benchmark verification failed: {cmd_str}"
                )
            return None
        return result.stdout
    except subprocess.TimeoutExpired:
        if failure_details is not None:
            failure_details.update({
                "failure_mode": "timeout",
                "elapsed_seconds": time.monotonic() - started,
                "timeout_seconds": timeout,
            })
        log.error(f"  TIMEOUT after {timeout}s")
        return None
    except OSError as e:
        if failure_details is not None:
            failure_details.update({
                "failure_mode": "os-error",
                "elapsed_seconds": time.monotonic() - started,
                "error": str(e),
            })
        log.error(f"  ERROR: {e}")
        return None


def save_json(data: Any, path: Path) -> None:
    if _DRY_RUN_MODE:
        log.info(f"  [dry-run] Would save: {path}")
        return
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    log.info(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# ResultsStore — checkpoint-after-every-cell + resume on restart.
# CRITICAL for SLURM / remote-cluster runs: a job killed mid-sweep loses
# nothing, and rerunning with --resume picks up exactly where it stopped.
# ---------------------------------------------------------------------------
class ResultsStore:
    """JSON-backed result accumulator with per-cell checkpoint + resume.

    Usage:
        store = ResultsStore(out_dir / "results.json",
                             key_fields=["graph", "algorithm", "benchmark"])
        for ... in ...:
            row = {"graph": g, "algorithm": a, "benchmark": b, ...}
            if store.has(row): continue       # resume: skip done cells
            ... run command ...
            store.add(row)                    # appends + flushes to disk
    """
    def __init__(self, path: Path, key_fields: list[str]):
        self.path = Path(path)
        self.key_fields = key_fields
        self.results: list[dict] = []
        self._seen: set[tuple] = set()
        if self.path.exists():
            try:
                with self.path.open() as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    self.results = loaded
                    for r in self.results:
                        self._seen.add(self._key(r))
                    log.info(f"  Resume: loaded {len(self.results)} existing "
                             f"results from {self.path.name}")
            except Exception as e:
                log.warning(f"  Could not load existing results ({e}); starting fresh")

    def _key(self, row: dict) -> tuple:
        return tuple(row.get(k) for k in self.key_fields)

    def has(self, row: dict) -> bool:
        return self._key(row) in self._seen

    def add(self, row: dict) -> None:
        self.results.append(row)
        self._seen.add(self._key(row))
        if _DRY_RUN_MODE:
            return
        ensure_dir(self.path.parent)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            merged: dict[tuple, dict] = {}
            if self.path.exists():
                try:
                    with self.path.open() as stream:
                        disk_rows = json.load(stream)
                    if isinstance(disk_rows, list):
                        for disk_row in disk_rows:
                            if isinstance(disk_row, dict):
                                merged[self._key(disk_row)] = disk_row
                except (OSError, ValueError):
                    pass
            for current in self.results:
                merged[self._key(current)] = current
            self.results = list(merged.values())
            self._seen = set(merged)
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json.tmp",
                dir=self.path.parent,
                delete=False,
            ) as tmp:
                json.dump(self.results, tmp, indent=2)
                tmp_path = Path(tmp.name)
            tmp_path.replace(self.path)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def parse_timing(output: Optional[str]) -> dict:
    """Extract timing fields from benchmark stdout.

    Delegates to :func:`scripts.lib.pipeline.benchmark.parse_benchmark_output`
    (the shared, rich parser) and flattens its ``(avg, reorder, extra)``
    return into the flat dict that the rest of this runner consumes.

    Captures (when present in stdout):
      - ``trial_times`` (per-trial wall-clock list)
      - ``average_time``, ``preprocessing_time``, ``total_time``
      - ``representation_build_time``, ``reorder_core_time``
      - ``reorder_validation_time``, ``reorder_apply_time``
      - ``total_preprocessing_time``
      - ``read_time``, ``topology_analysis_time``, ``relabel_map_time``
      - ``reorder_time`` (complete core + validation + apply cost)
        + ``reorder_time_passes`` (per-pass list)
      - ``mteps`` (BFS), ``iterations`` (PR/SSSP)
      - Topology features (degree_variance, hub_concentration, modularity, ...)
    """
    if not output:
        return {}
    _avg, reorder_total, extra = _lib_parse_bench(output)
    result = dict(extra)
    trial_times = [
        float(value) for value in result.get("trial_times", [])
        if isinstance(value, (int, float)) and value > 0
    ]
    if trial_times:
        result["median_time"] = statistics.median(trial_times)
        result["mean_time"] = statistics.fmean(trial_times)
        result["stddev_time"] = (
            statistics.stdev(trial_times) if len(trial_times) > 1 else 0.0
        )
        iteration_counts = result.get("iteration_counts", [])
        if (
            len(iteration_counts) == len(trial_times)
            and all(count > 0 for count in iteration_counts)
        ):
            per_iteration = [
                seconds / count
                for seconds, count in zip(
                    trial_times, iteration_counts,
                )
            ]
            result["time_per_iteration"] = per_iteration
            result["median_time_per_iteration"] = statistics.median(
                per_iteration
            )
    if reorder_total > 0:
        result["reorder_time"] = reorder_total
    return result


_GRAPHBREW_CONFIG_PREFIX = GRAPHBREW_EFFECTIVE_CONFIG_PREFIX
_GRAPHBREW_REALIZED_PREFIX = GRAPHBREW_REALIZED_CONFIG_PREFIX
_graphbrew_specs = extract_graphbrew_order_specs
_expected_graphbrew_config = expected_graphbrew_config


def parse_cache_sim(output: Optional[str]) -> dict:
    """Extract cache simulation metrics from sim binary stdout.

    The sim binary outputs a formatted table like:
        ║ L1 Cache (32KB, 8-way, Clock)
        ║   Hits:                       110358
        ║   Misses:                         67
        ║   Hit Rate:                 99.9393%
        ║ L2 Cache ...
        ║ Total Accesses:                98188
        ║ Memory Accesses:                  64
        ║ Overall Hit Rate:           99.9348%
    """
    result: dict = {
        "cache_schema": "cache_metrics/v2",
        "cache_rate_unit": "percent",
    }
    if not output:
        return result
    current_level = ""
    for line in output.splitlines():
        stripped = line.strip().strip("║").strip()
        if not stripped:
            continue
        # Detect cache level header: "L1 Cache (32KB, ...)"
        if stripped.startswith("L1 Cache"):
            current_level = "l1"
        elif stripped.startswith("L2 Cache"):
            current_level = "l2"
        elif stripped.startswith("L3 Cache"):
            current_level = "l3"
        elif stripped.startswith("SUMMARY"):
            current_level = "summary"
        elif current_level and ":" in stripped:
            key_part, _, val_part = stripped.partition(":")
            key_part = key_part.strip().lower().replace(" ", "_")
            val_part = val_part.strip().rstrip("%").replace(",", "")
            number = re.match(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", val_part)
            try:
                if not number:
                    continue
                val = float(number.group(0))
                if current_level == "summary":
                    result[key_part] = val
                else:
                    result[f"{current_level}_{key_part}"] = val
            except (ValueError, IndexError):
                pass
    return result


def git_revision() -> str:
    """Return short git hash, or 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT), text=True, timeout=5,
        ).strip()
    except Exception:
        return "unknown"


def _read_text(path: str) -> Optional[str]:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def machine_metadata() -> dict:
    """Capture the host details that materially affect timing reproducibility."""
    cpu_model = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
        "cpu_list": _RUNTIME_CPU_LIST,
        "threads": _RUNTIME_THREADS,
        "omp_env": dict(_RUNTIME_ENV),
        "cpu_governor": _read_text(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
        ),
        "intel_pstate_no_turbo": _read_text(
            "/sys/devices/system/cpu/intel_pstate/no_turbo"
        ),
        "rabbit_enable_env": os.environ.get("RABBIT_ENABLE", "1 (Makefile default)"),
    }


def timing_machine_metadata() -> dict[str, Optional[str]]:
    cpus = _expand_cpu_list(_RUNTIME_CPU_LIST) or [0]
    governors = sorted({
        value
        for cpu in cpus
        if (
            value := _read_text(
                f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/"
                "scaling_governor"
            )
        ) is not None
    })
    return {
        "cpu_governors": governors,
        "intel_pstate_no_turbo": _read_text(
            "/sys/devices/system/cpu/intel_pstate/no_turbo"
        ),
    }


def require_timing_machine_policy(*, preview: bool = False) -> None:
    if preview or os.environ.get("GRAPHBREW_ALLOW_UNSTABLE_TIMING") == "1":
        return
    expected_governor = os.environ.get(
        "GRAPHBREW_EXPECTED_GOVERNOR",
        "performance",
    )
    expected_no_turbo = os.environ.get(
        "GRAPHBREW_EXPECTED_NO_TURBO",
        "1",
    )
    expected_host = os.environ.get(
        "GRAPHBREW_EXPECTED_HOST",
        "jaguar",
    )
    expected_threads = int(os.environ.get(
        "GRAPHBREW_EXPECTED_THREADS",
        "16",
    ))
    expected_cpu_list = os.environ.get(
        "GRAPHBREW_EXPECTED_CPU_LIST",
        "0-15",
    )
    actual = timing_machine_metadata()
    if (
        platform.node().split(".", 1)[0] != expected_host
        or _RUNTIME_THREADS != expected_threads
        or _RUNTIME_CPU_LIST != expected_cpu_list
        or actual["cpu_governors"] != [expected_governor]
        or actual["intel_pstate_no_turbo"] != expected_no_turbo
    ):
        raise RuntimeError(
            "Final timing requires host/threads/CPUs "
            f"{expected_host}/{expected_threads}/{expected_cpu_list}, "
            "all pinned CPUs to use governor "
            f"{expected_governor!r} and intel_pstate/no_turbo="
            f"{expected_no_turbo}; found host="
            f"{platform.node().split('.', 1)[0]}, threads="
            f"{_RUNTIME_THREADS}, cpu_list={_RUNTIME_CPU_LIST}, "
            f"timing={actual}. Run: "
            "sudo cpupower frequency-set -g performance && "
            "echo 1 | sudo tee "
            "/sys/devices/system/cpu/intel_pstate/no_turbo"
        )


def verification_machine_identity(metadata: dict) -> dict:
    return {
        key: value
        for key, value in metadata.items()
        if key not in {
            "cpu_governor",
            "cpu_governors",
            "intel_pstate_no_turbo",
        }
    }


def _fingerprint_or_missing(path: str | Path) -> dict[str, Any]:
    try:
        return _stable_file_fingerprint(path)
    except OSError:
        return {"path": str(path), "missing": True}


def configure_campaign(
    *,
    graphs: list[dict],
    graph_dir: str,
    experiment_ids: list[int],
    benchmarks: list[str],
    trials: int,
) -> str:
    """Freeze one resumable measurement campaign from all material inputs."""
    global _CAMPAIGN_ID
    graph_manifest = {}
    mapping_manifest = {}
    for graph in graphs:
        name = graph["name"]
        graph_path = resolve_graph_path(name, graph_dir)
        graph_manifest[name] = {
            "path": str(Path(graph_path).resolve()),
            **_fingerprint_or_missing(graph_path),
        }
        mapping_dir = MAPPINGS_DIR / name
        if mapping_dir.is_dir():
            mapping_manifest[name] = {
                path.name: _fingerprint_or_missing(path)
                for path in sorted(mapping_dir.iterdir())
                if path.is_file() and path.suffix in {".lo", ".json"}
            }

    binaries = {"converter": {"path": str(BIN_DIR / "converter")}}
    if any(exp_id != 1 for exp_id in experiment_ids):
        binaries["cpu"] = {
            bench: {"path": str(BIN_DIR / bench)}
            for bench in benchmarks
        }
    if 1 in experiment_ids:
        binaries["cache"] = {
            "pr": {"path": str(BIN_SIM_DIR / "pr")},
        }

    payload = {
        "schema": "experiment_campaign/v1",
        "git_revision": git_revision(),
        "machine": machine_metadata(),
        "runtime_env": _RUNTIME_ENV,
        "cpu_list": _RUNTIME_CPU_LIST,
        "preview": _PREVIEW_MODE,
        "cache_mode": _CACHE_MODE,
        "cache_sample_rate": _CACHE_SAMPLE_RATE,
        "algorithm_env": _algorithm_environment(),
        "cache_sizes": _CACHE_SIZE_OVERRIDE,
        "algorithm_filter": sorted(_ALGORITHM_FILTER or []),
        "measurement_cohort_id":
            measurement_cohort_id("kernel", trials=trials),
        "experiment_ids": experiment_ids,
        "benchmarks": benchmarks,
        "trials": trials,
        "sssp_policy": SSSP_POLICY,
        "algorithm_graph_exclusions": ALGORITHM_GRAPH_EXCLUSIONS,
        "algorithm_graph_exclusion_evidence":
            _algorithm_exclusion_evidence_payload(graphs),
        "graphs": graph_manifest,
        "mappings": mapping_manifest,
        "binaries": binaries,
    }
    _CAMPAIGN_ID = _short_id(payload)
    payload["campaign_id"] = _CAMPAIGN_ID
    save_json(payload, RESULTS_DIR / "campaigns" / f"{_CAMPAIGN_ID}.json")
    return _CAMPAIGN_ID


def ensure_campaign(
    graphs: list[dict],
    graph_dir: str,
    experiment_id: int,
    benchmarks: list[str],
    trials: int,
) -> str:
    if _CAMPAIGN_ID is None:
        return configure_campaign(
            graphs=graphs,
            graph_dir=graph_dir,
            experiment_ids=[experiment_id],
            benchmarks=benchmarks,
            trials=trials,
        )
    return _CAMPAIGN_ID


def save_manifest(args: argparse.Namespace, elapsed: float) -> None:
    """Write a reproducibility manifest with config + environment info."""
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "git_revision": git_revision(),
        "campaign_id": _CAMPAIGN_ID,
        "measurement_generation_override":
            _MEASUREMENT_GENERATION_OVERRIDE,
        "platform": platform.platform(),
        "python": sys.version,
        "args": vars(args),
        "elapsed_seconds": round(elapsed, 1),
        "artifact_root": str(RESULTS_DIR.parent),
        "machine": machine_metadata(),
        "cache_policy": {
            "mode": _CACHE_MODE,
            "sample_rate": _CACHE_SAMPLE_RATE,
            "preview": _PREVIEW_MODE,
            "all_algorithms": _CACHE_ALL_ALGORITHMS,
        },
        "config": {
            "baselines": list(EVALUATION_BASELINES.values()),
            "graphbrew_variants": GRAPHBREW_VARIANTS,
            "chained_orderings": [c[0] for c in CHAINED_ORDERINGS],
            "benchmarks": BENCHMARKS,
            "graphs": [g["name"] for g in EVAL_GRAPHS],
            "sssp_policy": SSSP_POLICY,
        },
    }
    save_json(manifest, RESULTS_DIR / "MANIFEST.json")


# ---------------------------------------------------------------------------
# Mapping (.lo) pre-generation infrastructure
# ---------------------------------------------------------------------------

MAPPINGS_DIR = VLDB_ROOT / "vldb_mappings"
KERNEL_RUNS_DIR = VLDB_ROOT / "vldb_runs"

# Library parser: superset of the runner's old parse_timing(), plus per-trial
# vectors, MTEPS, iteration counts, topology features, and chained-reorder
# summing. Delegated to here so the rich data flows through to sidecars.
from scripts.lib.pipeline.benchmark import (  # noqa: E402
    file_sha256,
    mapping_permutation_fingerprint,
    parse_benchmark_output as _lib_parse_bench,
    repository_scope_state,
    repository_scope_semantics,
)
from scripts.lib.core.utils import get_graph_dimensions  # noqa: E402
from scripts.lib.ml.working_set import modeled_property_bytes  # noqa: E402


def _lo_path(graph_name: str, algo_key: str) -> Path:
    """Path for a pre-generated label-order mapping file."""
    safe = algo_key.replace(":", "_").replace("/", "_")
    return MAPPINGS_DIR / graph_name / f"{safe}.lo"


def _meta_path(graph_name: str, algo_key: str) -> Path:
    """Rich JSON sidecar with full reorder parameters + stdout."""
    safe = algo_key.replace(":", "_").replace("/", "_")
    return MAPPINGS_DIR / graph_name / f"{safe}.json"


def _load_reorder_meta(graph_name: str, algo_key: str) -> dict:
    """Load the current reorder sidecar, or ``{}`` if not yet generated."""
    mp = _meta_path(graph_name, algo_key)
    if mp.exists():
        try:
            return json.loads(mp.read_text())
        except (ValueError, OSError):
            pass
    return {}


def _load_reorder_time(graph_name: str, algo_key: str) -> float:
    """Return the cached precompute wall-clock from the sidecar, or 0.0."""
    return float(_load_reorder_meta(graph_name, algo_key).get("reorder_time", 0.0))


def _graph_provenance_path(graph_path: str | Path) -> Path:
    return Path(graph_path).with_suffix(".sg.meta.json")


def _graph_conversion_policy_id(
    provenance: dict,
    *,
    include_revision: bool = False,
) -> str:
    repository_state = provenance.get("conversion_repository_state", {})
    if not include_revision:
        repository_state = repository_scope_semantics(repository_state)
    payload = {
        "schema": "graph_conversion_policy/v1",
        "symmetrized": provenance.get("symmetrized"),
        "directed": provenance.get("directed"),
        "random_order_algorithm": provenance.get("random_order_algorithm"),
        "random_seed": provenance.get("random_seed"),
        "reorder_semantics_version":
            provenance.get("reorder_semantics_version"),
        "source_crc32": provenance.get("source_crc32"),
        "output_crc32": provenance.get("output_crc32"),
        "converter_sha256": provenance.get("converter_sha256"),
        "conversion_repository_state": repository_state,
        "expected_nodes": provenance.get("expected_nodes"),
        "expected_undirected_edges":
            provenance.get("expected_undirected_edges"),
    }
    return _short_id(payload)


def _conversion_repository_state() -> dict[str, Any]:
    """Fingerprint tracked and untracked converter-relevant repository state."""
    return repository_scope_state(
        PROJECT_ROOT,
        (
            "Makefile",
            "bench/src/converter.cc",
            "bench/include",
        ),
    )


def _serialized_graph_info(graph_path: str | Path) -> dict[str, int | bool]:
    with Path(graph_path).open("rb") as stream:
        header = stream.read(17)
    if len(header) != 17:
        raise RuntimeError(f"Serialized graph header is truncated: {graph_path}")
    directed, edges, nodes = struct.unpack("<?qq", header)
    return {"directed": directed, "edges": edges, "nodes": nodes}


def _graph_provenance_valid(
    graph_path: str | Path,
    *,
    graph_name: Optional[str] = None,
    canonical_output_path: Optional[str | Path] = None,
    expected_nodes: Optional[int] = None,
    expected_undirected_edges: Optional[int] = None,
) -> bool:
    graph_path = Path(graph_path)
    provenance_path = _graph_provenance_path(graph_path)
    if not graph_path.exists() or not provenance_path.exists():
        return False
    try:
        provenance = json.loads(provenance_path.read_text())
        graph_info = _serialized_graph_info(graph_path)
        converter_args = provenance.get("converter_args", [])
        source_path = Path(provenance.get("source_path", ""))
        recorded_output_path = Path(
            provenance.get("output_path", ""))
        expected_output_path = Path(
            canonical_output_path or graph_path).resolve()
        random_arg = (
            converter_args.index("-o") + 1 < len(converter_args)
            and converter_args[converter_args.index("-o") + 1] == "1"
            if "-o" in converter_args else False
        )
        return (
            provenance.get("schema") == "graph_source/v2"
            and provenance.get("reorder_semantics_version")
            == REORDER_SEMANTICS_VERSION
            and (
                graph_name is None
                or provenance.get("graph") == graph_name
            )
            and provenance.get("random_order_algorithm") == "1"
            and provenance.get("random_seed") == RANDOM_BASELINE_SEED
            and provenance.get("converter_sha256")
            == file_sha256(BIN_DIR / "converter")
            and repository_scope_semantics(
                provenance.get("conversion_repository_state", {})
            ) == repository_scope_semantics(
                _conversion_repository_state())
            and (
                not source_path.is_file()
                or (
                    provenance.get("source_bytes")
                    == source_path.stat().st_size
                    and provenance.get("source_crc32")
                    == _file_crc32(source_path)
                )
                and recorded_output_path.name == expected_output_path.name
            )
            and isinstance(converter_args, list)
            and "-s" in converter_args
            and random_arg
            and "-f" in converter_args
            and "-b" in converter_args
            and graph_info["directed"] is False
            and provenance.get("directed") is False
            and provenance.get("symmetrized") is True
            and provenance.get("nodes") == graph_info["nodes"]
            and provenance.get("directed_edges") == graph_info["edges"]
            and provenance.get("expected_nodes")
            == (
                expected_nodes
                if expected_nodes is not None
                else provenance.get("nodes")
            )
            and provenance.get("expected_undirected_edges")
            == (
                expected_undirected_edges
                if expected_undirected_edges is not None
                else provenance.get("undirected_edges")
            )
            and provenance.get("nodes")
            == provenance.get("expected_nodes")
            and provenance.get("undirected_edges")
            == provenance.get("expected_undirected_edges")
            and provenance.get("output_bytes") == graph_path.stat().st_size
            and provenance.get("output_crc32")
            == _file_crc32(graph_path)
            and provenance.get("conversion_policy_id") in {
                _graph_conversion_policy_id(provenance),
                _graph_conversion_policy_id(
                    provenance, include_revision=True),
            }
        )
    except (IndexError, OSError, ValueError, RuntimeError):
        return False


def _graph_fingerprint(
    path: str | Path,
    *,
    use_cache: bool = True,
) -> dict[str, int | str]:
    """Return lightweight file metadata without content hashing."""
    graph_path = Path(path)
    stat = graph_path.stat()
    cache_key = (str(graph_path.resolve()), stat.st_size, stat.st_mtime_ns)
    if use_cache:
        cached = _FILE_FINGERPRINT_CACHE.get(cache_key)
        if cached is not None:
            return dict(cached)
    fingerprint = {
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    _FILE_FINGERPRINT_CACHE[cache_key] = fingerprint
    return dict(fingerprint)


def _stable_file_fingerprint(
    path: str | Path,
    *,
    use_cache: bool = True,
) -> dict[str, int | str]:
    """Return copy-stable file metadata without content hashing."""
    fingerprint = _graph_fingerprint(path, use_cache=use_cache)
    fingerprint.pop("mtime_ns", None)
    return fingerprint


def _algorithm_exclusion_evidence_payload(
    graphs: list[dict],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Validate semantic exclusion evidence for selected graphs."""
    if set(ALGORITHM_GRAPH_EXCLUSIONS) != set(
        ALGORITHM_GRAPH_EXCLUSION_EVIDENCE
    ):
        raise RuntimeError(
            "Algorithm exclusion and evidence graph sets differ"
        )
    for graph_name, exclusions in ALGORITHM_GRAPH_EXCLUSIONS.items():
        if set(exclusions) != set(
            ALGORITHM_GRAPH_EXCLUSION_EVIDENCE[graph_name]
        ):
            raise RuntimeError(
                f"Algorithm exclusion and evidence keys differ for "
                f"{graph_name}"
            )

    selected = {graph["name"] for graph in graphs}
    payload: dict[str, dict[str, dict[str, Any]]] = {}
    artifact_root = RESULTS_DIR.parent
    for graph_name, algorithms in ALGORITHM_GRAPH_EXCLUSION_EVIDENCE.items():
        if graph_name not in selected:
            continue
        payload[graph_name] = {}
        for algorithm_key, record in algorithms.items():
            repository_path = PROJECT_ROOT / record["artifact"]
            evidence_path = (
                repository_path
                if repository_path.is_file()
                else artifact_root
                / record.get("external_artifact", record["artifact"])
            )
            try:
                evidence = json.loads(evidence_path.read_text())
            except (OSError, ValueError) as error:
                raise RuntimeError(
                    f"Missing exclusion evidence for "
                    f"{graph_name}/{algorithm_key}: {evidence_path}"
                ) from error
            if (
                evidence.get("schema") != record["schema"]
                or evidence.get("graph") != graph_name
                or evidence.get("algorithm_key") != algorithm_key
                or evidence.get("applicability_timeout_seconds")
                != record["timeout_seconds"]
            ):
                raise RuntimeError(
                    f"Invalid exclusion evidence for "
                    f"{graph_name}/{algorithm_key}: {evidence_path}"
                )
            payload[graph_name][algorithm_key] = {
                **record,
                "artifact": str(evidence_path),
            }
    return payload


def _mapping_draw_count(algo_flags: list[str]) -> int:
    """Return repeated mapping draws for schedule-sensitive pipelines."""
    for index, flag in enumerate(algo_flags[:-1]):
        if flag != "-o":
            continue
        spec = algo_flags[index + 1]
        algorithm_id = spec.split(":", 1)[0]
        if algorithm_id == "8":
            return RABBIT_MAPPING_DRAWS
        if algorithm_id != "12":
            continue
        expected = _expected_graphbrew_config(spec)
        if (
            spec in COMPOSITION_P0_ALGORITHM_KEYS
            or graphbrew_schedule_sensitive(expected)
        ):
            return RABBIT_MAPPING_DRAWS
    return 1


def _validate_mapping_draw_cohort(
    algo_key: str,
    draw_records: list[dict[str, Any]],
) -> None:
    if not draw_records:
        raise RuntimeError(f"No mapping draws recorded for {algo_key}")
    deterministic_membership_indexes = [
        index
        for index, effective in enumerate(
            draw_records[0]["graphbrew_effective_configs"]
        )
        if effective["deterministic_community_detection"]
    ]
    for config_index in deterministic_membership_indexes:
        membership_fingerprints = {
            record["graphbrew_realized_configs"][config_index][
                "membership_fingerprint"
            ]
            for record in draw_records
        }
        if len(membership_fingerprints) != 1:
            raise RuntimeError(
                f"Deterministic membership drift for "
                f"{algo_key}/config{config_index}: "
                f"{sorted(membership_fingerprints)}"
            )
    if algo_key in COMPOSITION_P0_ALGORITHM_KEYS:
        mapping_fingerprints = {
            record["mapping_fingerprint"]
            for record in draw_records
        }
        if len(mapping_fingerprints) != 1:
            raise RuntimeError(
                f"Composition P0 mapping drift for {algo_key}: "
                f"{sorted(mapping_fingerprints)}"
            )


def _mapping_generation_policy_id(
    graph_path: str | Path,
    algo_flags: list[str],
) -> str:
    return _short_id({
        "schema": "mapping-generation-policy/v1",
        "reorder_semantics_version": REORDER_SEMANTICS_VERSION,
        "graph_crc32": _file_crc32(graph_path),
        "converter_flags": list(algo_flags),
        "omp_env": dict(_RUNTIME_ENV),
        "cpu_list": _RUNTIME_CPU_LIST,
        "algorithm_env": _algorithm_environment(),
    })


def _mapping_is_valid(
    graph_name: str,
    algo_key: str,
    graph_path: str | Path,
    algo_flags: list[str],
) -> bool:
    lo = _lo_path(graph_name, algo_key)
    graph_path = Path(graph_path)
    if (
        not graph_path.is_file()
        or not lo.exists()
        or lo.stat().st_size == 0
    ):
        return False
    meta = _load_reorder_meta(graph_name, algo_key)
    try:
        schema = meta.get("schema")
        if schema != "reorder_meta/v6":
            return False
        if (
            meta.get("reorder_semantics_version")
            != REORDER_SEMANTICS_VERSION
        ):
            return False
        if meta.get("graph_crc32") != _file_crc32(graph_path):
            return False
        if meta.get("generation_policy_id") != (
            _mapping_generation_policy_id(graph_path, algo_flags)
        ):
            return False
        top_effective = meta.get("graphbrew_effective_configs", [])
        top_realized = meta.get("graphbrew_realized_configs", [])
        validate_graphbrew_effective_configs(list(algo_flags), top_effective)
        validate_graphbrew_realized_configs(
            list(algo_flags), top_effective, top_realized,
        )
        draw_count = _mapping_draw_count(algo_flags)
        draw_records = meta.get("mapping_draws")
        meta_dir = _meta_path(graph_name, algo_key).parent
        resolved_draws = []
        if isinstance(draw_records, list):
            for record in draw_records:
                record_path = Path(record.get("path", ""))
                if not record_path.is_absolute():
                    record_path = meta_dir / record_path
                resolved_draws.append((record, record_path))
        draws_valid = (
            isinstance(draw_records, list)
            and len(draw_records) == draw_count
            and all(
                record_path.is_file()
                and record_path.stat().st_size > 0
                and record.get("mapping_fingerprint")
                == mapping_permutation_fingerprint(record_path)
                and _mapping_draw_config_is_valid(
                    list(algo_flags), record,
                )
                for record, record_path in resolved_draws
            )
            and resolved_draws[0][1].stat().st_size == lo.stat().st_size
        )
        command = meta.get("cmd")
        command_template = meta.get("cmd_template")
        command_valid = (
            isinstance(command, list)
            and len(command) >= 2
            and len(resolved_draws) > 0
            and Path(command[-1]).is_absolute()
            and Path(command[-1]).name
            == Path(draw_records[0].get("path", "")).name
            and isinstance(command_template, list)
            and command_template[-1] == draw_records[0].get("path")
        )
        timing_valid = all(
            isinstance(meta.get(field), (int, float))
            and meta[field] >= 0
            for field in (
                "representation_build_time",
                "reorder_core_time",
                "reorder_validation_time",
                "reorder_apply_time",
                "total_preprocessing_time",
            )
        )
        if timing_valid:
            complete = (
                float(meta["reorder_core_time"])
                + float(meta["reorder_validation_time"])
                + float(meta["reorder_apply_time"])
            )
            timing_valid = (
                float(meta["total_preprocessing_time"]) + 1e-4
                >= float(meta["representation_build_time"]) + complete
            )
        mapping_fingerprint = mapping_permutation_fingerprint(lo)
        return (
            meta.get("graph") == graph_name
            and meta.get("graph_info")
            == _serialized_graph_info(graph_path)
            and meta.get("converter_flags") == list(algo_flags)
            and meta.get("lo_path") == lo.name
            and meta.get("lo_bytes") == lo.stat().st_size
            and meta.get("mapping_fingerprint")
            == mapping_fingerprint
            and meta.get("mapping_draw_count") == draw_count
            and draws_valid
            and command_valid
            and timing_valid
        )
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return False


def _mapping_draw_config_is_valid(
    algo_flags: list[str],
    record: dict,
) -> bool:
    effective = record.get("graphbrew_effective_configs", [])
    realized = record.get("graphbrew_realized_configs", [])
    validate_graphbrew_effective_configs(algo_flags, effective)
    validate_graphbrew_realized_configs(algo_flags, effective, realized)
    return True


def _algo_identity(algo_key: str) -> int | str:
    return int(algo_key) if algo_key.isdigit() else algo_key


def _paper_algorithm_specs(
    *, include_compose: bool = False,
) -> list[tuple[str, str, list[str]]]:
    """Return canonical (key, display name, converter flags) experiment specs."""
    specs = [
        (key, name, get_converter_flags(key))
        for key, name in EVALUATION_BASELINES.items()
    ]
    specs.extend(
        (f"12:{variant}", ALL_ALGORITHMS[f"12:{variant}"], ["-o", f"12:{variant}"])
        for variant in GRAPHBREW_VARIANTS
    )
    if include_compose:
        specs.extend(
            (spec, ALL_ALGORITHMS[spec], ["-o", spec])
            for _label, spec in COMPOSE_VARIANTS
        )
    if _ALGORITHM_FILTER is not None:
        by_key = {spec[0]: spec for spec in specs}
        specs = []
        for key in sorted(_ALGORITHM_FILTER):
            if key.startswith("chain:"):
                continue
            specs.append(
                by_key[key]
                if key in by_key
                else _algorithm_spec_for_key(key)
            )
    return specs


def _algorithm_spec_for_key(key: str) -> tuple[str, str, list[str]]:
    if key in ALL_ALGORITHMS:
        return key, ALL_ALGORITHMS[key], get_converter_flags(key)
    for config in ABLATION_CONFIGS:
        if config["algo"] == key:
            return key, config["name"], get_converter_flags(key)
    for config in DIAGNOSTIC_CONFIGS:
        if config["algo"] == key:
            return key, config["name"], get_converter_flags(key)
    for config in DUAL_ARM_S0_CONFIGS:
        if config["algo"] == key:
            return key, config["name"], get_converter_flags(key)
    for config in DUAL_ARM_S2_CONFIGS:
        if config["algo"] == key:
            return key, config["name"], get_converter_flags(key)
    for config in COMPOSITION_P0_CONFIGS:
        if config["algo"] == key:
            return key, config["name"], get_converter_flags(key)
    for chain_name, chain_flags in CHAINED_ORDERINGS:
        if key == f"chain:{chain_name}":
            return key, chain_name, list(chain_flags)
    raise KeyError(f"No experiment specification registered for {key}")


def _cache_algorithm_specs() -> list[tuple[str, str, list[str]]]:
    if _CACHE_ALL_ALGORITHMS:
        return _paper_algorithm_specs(include_compose=True)
    if _ALGORITHM_FILTER is not None:
        keys = sorted(_ALGORITHM_FILTER)
    else:
        keys = CACHE_ALGORITHM_KEYS_PREVIEW if _PREVIEW_MODE else CACHE_ALGORITHM_KEYS
    specs: list[tuple[str, str, list[str]]] = []
    for key in keys:
        specs.append(_algorithm_spec_for_key(key))
    if not specs:
        raise RuntimeError("The selected algorithm filter produced no cache cells")
    return specs


def _overhead_algorithm_specs() -> list[tuple[str, str, list[str]]]:
    keys = [key for key, _name, _flags in _paper_algorithm_specs(include_compose=True)]
    keys.extend(config["algo"] for config in ABLATION_CONFIGS)
    if _ALGORITHM_FILTER is not None:
        keys.extend(
            config["algo"]
            for config in DIAGNOSTIC_CONFIGS
            if config["algo"] in _ALGORITHM_FILTER
        )
        keys.extend(
            config["algo"]
            for config in DUAL_ARM_S0_CONFIGS
            if config["algo"] in _ALGORITHM_FILTER
        )
        keys.extend(
            config["algo"]
            for config in DUAL_ARM_S2_CONFIGS
            if config["algo"] in _ALGORITHM_FILTER
        )
        keys.extend(
            config["algo"]
            for config in COMPOSITION_P0_CONFIGS
            if config["algo"] in _ALGORITHM_FILTER
        )
    keys.extend(f"chain:{name}" for name, _flags in CHAINED_ORDERINGS)
    if _ALGORITHM_FILTER is not None:
        keys = [key for key in keys if key in _ALGORITHM_FILTER]
    specs: list[tuple[str, str, list[str]]] = []
    seen: set[str] = set()
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        specs.append(_algorithm_spec_for_key(key))
    return specs


def _pregenerate_mappings(
    graphs: list[dict],
    graph_dir: str,
    dry_run: bool = False,
    timeout: int = REORDER_TIMEOUT_FULL,
) -> None:
    """Pre-generate .lo mapping files for all (graph, algorithm) pairs.

    Runs the converter with ``-q {lo_path}`` to produce a vertex-permutation
    file, and writes a reorder_meta/v6 ``.json``
    sidecar next to it with the full cmd / env / timing / stdout tail.
    Schedule-sensitive Rabbit pipelines retain multiple named draws while
    pinning draw 0 as the mapping used by measured kernels.

    Subsequent experiments use MAP mode (``-o 13:{lo_path}``) so the
    benchmark binary loads the pre-computed ordering with zero reorder
    cost, and all benchmarks for the same graph×algorithm see exactly
    the same ordering.
    """
    converter = BIN_DIR / "converter"
    if not converter.exists():
        if dry_run:
            log.info("  [dry-run] Converter binary is not present")
            return
        raise RuntimeError("Converter binary not found; cannot generate required mappings")
    if not dry_run:
        _algorithm_exclusion_evidence_payload(graphs)

    # Build the full algo list: baselines + GB variants + chained
    algo_list: list[tuple[str, list[str]]] = []  # (key, flags)
    for algo_key, _aname in EVALUATION_BASELINES.items():
        if algo_key == "0":
            continue  # SHUFFLED baseline — no additional mapping needed
        algo_list.append((algo_key, get_converter_flags(algo_key)))
    for v in GRAPHBREW_VARIANTS:
        algo_list.append((f"12:{v}", ["-o", f"12:{v}"]))
    for chain_name, chain_flags in CHAINED_ORDERINGS:
        algo_list.append((f"chain:{chain_name}", chain_flags))
    # COMPOSE_VARIANTS include explicit one-axis Rabbit super-graph controls.
    # Pre-generating these guarantees the same ordering is reused across every
    # kernel for a given graph: reorder cost paid once per (graph, algo) and
    # every kernel sees byte-identical layout.
    for _label, spec in COMPOSE_VARIANTS:
        if not any(k == spec for k, _ in algo_list):
            algo_list.append((spec, ["-o", spec]))
    # Ablation configs that aren't already covered
    for cfg in ABLATION_CONFIGS:
        key = cfg["algo"]
        if key == "0":
            continue
        if not any(k == key for k, _ in algo_list):
            algo_list.append((key, get_converter_flags(key)))
    if _ALGORITHM_FILTER is not None:
        for cfg in (
            *DIAGNOSTIC_CONFIGS,
            *DUAL_ARM_S0_CONFIGS,
            *DUAL_ARM_S2_CONFIGS,
            *COMPOSITION_P0_CONFIGS,
        ):
            key = cfg["algo"]
            if (
                key in _ALGORITHM_FILTER
                and not any(k == key for k, _ in algo_list)
            ):
                algo_list.append((key, get_converter_flags(key)))
    if _ALGORITHM_FILTER is not None:
        algo_list = [
            (key, flags) for key, flags in algo_list
            if key in _ALGORITHM_FILTER
        ]

    generated = 0
    skipped = 0
    failed = 0
    planned_mappings = 0
    planned_draws = 0

    for graph in graphs:
        gname = graph["name"]
        graph_planned_mappings = 0
        graph_planned_draws = 0
        sg = resolve_graph_path(gname, graph_dir, ext=".sg")
        if not Path(sg).exists():
            if dry_run:
                log.info(f"  [dry-run] {gname}: no .sg file")
                continue
            raise RuntimeError(f"{gname}: required randomized .sg file is missing")
        provenance_valid = _graph_provenance_valid(
            sg,
            graph_name=gname,
        )
        if not provenance_valid:
            if dry_run:
                log.warning(
                    f"  [dry-run] {gname}: randomized .sg provenance "
                    "must be refreshed before execution"
                )
            else:
                raise RuntimeError(
                    f"{gname}: randomized .sg provenance is missing or stale"
                )

        for algo_key, aflags in algo_list:
            exclusion = algorithm_exclusion_reason(gname, algo_key)
            if exclusion:
                log.info(
                    f"  EXCLUDED: {gname} → {algo_key}: {exclusion}"
                )
                continue
            draw_count = _mapping_draw_count(aflags)
            graph_planned_mappings += 1
            graph_planned_draws += draw_count
            planned_mappings += 1
            planned_draws += draw_count
            if dry_run:
                log.info(
                    f"  [dry-run] {gname} → {algo_key} "
                    f"({draw_count} draw(s))"
                )
                continue

            lo = _lo_path(gname, algo_key)

            if lo.exists() and lo.stat().st_size > 0:
                if _mapping_is_valid(gname, algo_key, sg, aflags):
                    skipped += 1
                    continue
                log.warning(f"  {gname} → {algo_key}: invalidating stale mapping")
                lo.unlink()
                for draw_path in lo.parent.glob(f"{lo.stem}.draw*.lo"):
                    draw_path.unlink()
                meta_path = _meta_path(gname, algo_key)
                if meta_path.exists():
                    meta_path.unlink()

            lo.parent.mkdir(parents=True, exist_ok=True)
            draw_records: list[dict[str, Any]] = []
            draw_outputs: list[str] = []
            generated_draw_paths: list[Path] = []
            draw_failed = False
            for draw in range(draw_count):
                draw_path = (
                    lo if draw_count == 1
                    else lo.with_name(f"{lo.stem}.draw{draw}.lo")
                )
                draw_path.unlink(missing_ok=True)
                generated_draw_paths.append(draw_path)
                cmd = [str(converter), "-f", sg, "-s"]
                cmd.extend(aflags)
                cmd.extend(["-q", str(draw_path)])
                output = run_cmd(
                    cmd,
                    dry_run=False,
                    timeout=timeout,
                    env={"GRAPHBREW_MAPPING_QUALITY": "1"},
                )
                if (
                    output is None
                    or not draw_path.exists()
                    or draw_path.stat().st_size == 0
                ):
                    log.warning(
                        f"  FAILED: {gname} → {algo_key}, draw {draw}"
                    )
                    draw_failed = True
                    break

                effective_configs = parse_graphbrew_effective_configs(output)
                validate_graphbrew_effective_configs(
                    aflags, effective_configs,
                )
                realized_configs = parse_graphbrew_realized_configs(output)
                validate_graphbrew_realized_configs(
                    aflags, effective_configs, realized_configs,
                )
                timing = parse_timing(output)
                mapping_fingerprint = timing.get(
                    "mapping_fingerprint")
                if (
                    not isinstance(mapping_fingerprint, str)
                    or mapping_fingerprint
                    != mapping_permutation_fingerprint(draw_path)
                ):
                    raise RuntimeError(
                        f"Mapping fingerprint mismatch for "
                        f"{gname}/{algo_key}/draw{draw}"
                    )
                core_times = timing.get("reorder_core_time_passes", [])
                validation_times = timing.get(
                    "reorder_validation_time_passes", []
                )
                apply_times = timing.get("reorder_apply_time_passes", [])
                end_to_end_times = timing.get("reorder_time_passes", [])
                if not all(
                    isinstance(values, list)
                    for values in (
                        core_times,
                        validation_times,
                        apply_times,
                        end_to_end_times,
                    )
                ) or not (
                    len(core_times)
                    == len(validation_times)
                    == len(apply_times)
                    == len(end_to_end_times)
                ):
                    raise RuntimeError(
                        f"Incomplete end-to-end reorder timing for "
                        f"{gname}/{algo_key}/draw{draw}"
                    )
                edge_spans = [
                    float(value)
                    for value in re.findall(
                        r"Mapping Sampled Edge Span:\s*([\d.]+)",
                        output,
                    )
                ]
                sampled_edges = [
                    int(value)
                    for value in re.findall(
                        r"Mapping Sampled Edges:\s*(\d+)", output,
                    )
                ]
                draw_records.append({
                    "draw": draw,
                    "path": draw_path.name,
                    "mapping_fingerprint": mapping_fingerprint,
                    "cmd": cmd,
                    "reorder_time": sum(end_to_end_times),
                    "reorder_time_passes": end_to_end_times,
                    "representation_build_time":
                        timing.get("representation_build_time", 0.0),
                    "reorder_core_time": sum(core_times),
                    "reorder_core_time_passes": core_times,
                    "mapping_generation_time": sum(core_times),
                    "mapping_generation_time_passes": core_times,
                    "reorder_validation_time": sum(validation_times),
                    "reorder_validation_time_passes": validation_times,
                    "reorder_apply_time": sum(apply_times),
                    "reorder_apply_time_passes": apply_times,
                    "total_preprocessing_time":
                        timing.get("total_preprocessing_time", 0.0),
                    "mapping_sampled_edge_span": (
                        edge_spans[-1] if edge_spans else None
                    ),
                    "mapping_sampled_edges": (
                        sampled_edges[-1] if sampled_edges else None
                    ),
                    "graphbrew_effective_configs": effective_configs,
                    "graphbrew_realized_configs": realized_configs,
                    "stdout_tail": output.strip().splitlines()[-40:],
                })
                draw_outputs.append(output)

            if draw_failed:
                lo.unlink(missing_ok=True)
                for draw_path in generated_draw_paths:
                    draw_path.unlink(missing_ok=True)
                failed += 1
                continue
            try:
                _validate_mapping_draw_cohort(algo_key, draw_records)
            except RuntimeError as exc:
                raise RuntimeError(f"{gname}: {exc}") from exc
            if draw_count > 1:
                selected_draw_path = (
                    lo.parent / draw_records[0]["path"]
                )
                lo.unlink(missing_ok=True)
                try:
                    os.link(selected_draw_path, lo)
                except OSError:
                    shutil.copyfile(selected_draw_path, lo)

            output = draw_outputs[0]
            effective_configs = draw_records[0][
                "graphbrew_effective_configs"
            ]
            realized_configs = draw_records[0][
                "graphbrew_realized_configs"
            ]
            reorder_times = draw_records[0]["reorder_time_passes"]
            total = draw_records[0]["reorder_time"]
            timing = parse_timing(output)
            meta = {
                "schema": "reorder_meta/v6",
                "reorder_semantics_version":
                    REORDER_SEMANTICS_VERSION,
                "graph": gname,
                "graph_info": _serialized_graph_info(sg),
                "graph_crc32": _file_crc32(sg),
                "algo_key": algo_key,
                "converter_flags": list(aflags),
                "generation_policy_id":
                    _mapping_generation_policy_id(sg, aflags),
                "omp_env": dict(_RUNTIME_ENV),
                "cpu_list": _RUNTIME_CPU_LIST,
                "algorithm_env": _algorithm_environment(),
                "cmd": draw_records[0]["cmd"],
                "cmd_template": [
                    str(converter), "-f", sg, "-s",
                    *aflags, "-q", draw_records[0]["path"],
                ],
                "omp_num_threads": _effective_env().get("OMP_NUM_THREADS"),
                "cpu_list": _RUNTIME_CPU_LIST,
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "reorder_time": total,
                "reorder_time_passes": reorder_times,
                "representation_build_time":
                    draw_records[0]["representation_build_time"],
                "reorder_core_time":
                    draw_records[0]["reorder_core_time"],
                "reorder_core_time_passes":
                    draw_records[0]["reorder_core_time_passes"],
                "mapping_generation_time":
                    draw_records[0]["mapping_generation_time"],
                "mapping_generation_time_passes":
                    draw_records[0]["mapping_generation_time_passes"],
                "reorder_validation_time":
                    draw_records[0]["reorder_validation_time"],
                "reorder_validation_time_passes":
                    draw_records[0]["reorder_validation_time_passes"],
                "reorder_apply_time":
                    draw_records[0]["reorder_apply_time"],
                "reorder_apply_time_passes":
                    draw_records[0]["reorder_apply_time_passes"],
                "total_preprocessing_time":
                    draw_records[0]["total_preprocessing_time"],
                "lo_path": lo.name,
                "lo_bytes": lo.stat().st_size,
                "mapping_fingerprint":
                    draw_records[0]["mapping_fingerprint"],
                "mapping_draw_count": draw_count,
                "selected_draw": 0,
                "mapping_draws": draw_records,
                "timing": timing,
                "graphbrew_effective_configs": effective_configs,
                "graphbrew_realized_configs": realized_configs,
                "stdout_tail": output.strip().splitlines()[-40:],
            }
            _meta_path(gname, algo_key).write_text(json.dumps(meta, indent=2))

            generated += 1

        if dry_run:
            log.info(
                f"  [dry-run] {gname}: planned {graph_planned_mappings} "
                f"mapping(s), {graph_planned_draws} named draw(s)"
            )

    if dry_run:
        log.info(
            f"  Mappings planned: {planned_mappings}, "
            f"named draws: {planned_draws}"
        )
    if not dry_run:
        log.info(
            f"  Mappings: {generated} generated, {skipped} existing, "
            f"{failed} failed"
        )
    if failed:
        raise RuntimeError(f"{failed} required mapping(s) failed to generate")


def algo_flags_or_map(
    algo_key: str, algo_flags: list[str], graph_name: str,
    graph_path: Optional[str] = None,
) -> tuple[list[str], float, dict]:
    """Return flags, prerecorded reorder time, and exact mapping identity.

    If a pre-generated .lo file exists for this (graph, algo), returns
    ``["-o", "13:{lo_path}"]`` so the benchmark loads the cached
    ordering with zero runtime reorder cost.  The recorded reorder time
    from the sidecar JSON is returned as the second element.

    Otherwise falls back to the original *algo_flags* (runtime reorder).
    """
    lo = _lo_path(graph_name, algo_key)
    if lo.exists() and lo.stat().st_size > 0:
        if graph_path and not _mapping_is_valid(
            graph_name, algo_key, graph_path, algo_flags,
        ):
            if _DRY_RUN_MODE:
                return algo_flags, 0.0, {
                    "source": "dry-run-direct",
                    "algo_flags": list(algo_flags),
                }
            raise RuntimeError(
                f"Mapping for {graph_name}/{algo_key} does not match the "
                "current graph dimensions or algorithm specification"
            )
        rt = _load_reorder_time(graph_name, algo_key)
        meta = _load_reorder_meta(graph_name, algo_key)
        draw_files = []
        for record in meta.get("mapping_draws", []):
            draw_path = lo.parent / record.get("path", "")
            if draw_path.is_file():
                draw_files.append({
                    "path": record.get("path"),
                    "bytes": draw_path.stat().st_size,
                    "mapping_fingerprint":
                        record.get("mapping_fingerprint"),
                })
        return ["-o", f"13:{lo}"], rt, {
            "source": "map",
            "path": lo.name,
            "bytes": lo.stat().st_size,
            "mapping_fingerprint": meta.get("mapping_fingerprint"),
            "generation_policy_id": meta.get("generation_policy_id"),
            "selected_draw": meta.get("selected_draw"),
            "mapping_draws": draw_files,
        }
    if algo_key != "0" and not _DRY_RUN_MODE:
        raise RuntimeError(
            f"Required mapping is missing for {graph_name}/{algo_key}; "
            "run stage 02 before measured experiments"
        )
    return algo_flags, 0.0, {
        "source": "direct",
        "algo_flags": list(algo_flags),
    }


def build_benchmark_cmd(
    benchmark: str, graph_path: str, algo_flags: list[str], trials: int = 3,
    sim: bool = False,
    work_metrics: bool = False,
) -> list[str]:
    """Build the CLI command to run a benchmark with a reorder algorithm."""
    if sim and work_metrics:
        raise ValueError("Cache simulation and work metrics are separate paths")
    bin_dir = (
        BIN_SIM_DIR if sim
        else BIN_WORK_DIR if work_metrics
        else BIN_DIR
    )
    binary_name = {
        "pr_convergence": "pr",
        "pr_spmv_convergence": "pr_spmv",
    }.get(benchmark, benchmark)
    binary = bin_dir / binary_name
    cmd = [str(binary), "-f", graph_path, "-s", "-n", str(trials)]
    if benchmark in {"pr", "pr_spmv"}:
        cmd.extend([
            "-F",
            "-i", str(PR_FIXED_ITERATIONS),
            "-t", str(PR_TOLERANCE),
        ])
    if benchmark == "bc":
        cmd.extend(["-i", str(BC_SOURCE_ITERATIONS)])
    elif benchmark in {
        "pr_convergence", "pr_spmv_convergence",
    }:
        cmd.extend([
            "-i", str(PR_CONVERGENCE_MAX_ITERATIONS),
            "-t", str(PR_TOLERANCE),
        ])
    if benchmark == "sssp":
        policy = sssp_policy_for_graph(Path(graph_path).stem)
        cmd.extend([
            "-W", str(policy["weight_scheme"]),
            "-d", str(policy["delta"]),
        ])
    cmd.extend(algo_flags)
    return cmd


def _verification_benchmark_name(benchmark: str) -> str:
    if benchmark == "pr":
        return "pr_convergence"
    if benchmark == "pr_spmv":
        return "pr_spmv_convergence"
    return benchmark


def _verification_gate_payload(
    graphs: list[dict],
    benchmarks: list[str],
    graph_dir: str,
) -> dict[str, Any]:
    graph_payload = {}
    mapping_identities = {}
    specs = _overhead_algorithm_specs()
    for graph in graphs:
        graph_name = graph["name"]
        graph_path = resolve_graph_path(graph_name, graph_dir)
        graph_payload[graph_name] = {
            "path": str(Path(graph_path).resolve()),
            **_serialized_graph_info(graph_path),
        }
        mapping_identities[graph_name] = {}
        for key, _name, flags in specs:
            if algorithm_exclusion_reason(graph_name, str(key)):
                continue
            try:
                _resolved, _time, identity = algo_flags_or_map(
                    key,
                    flags,
                    graph_name,
                    graph_path,
                )
            except RuntimeError:
                identity = {
                    "source": "missing",
                    "algo_flags": list(flags),
                }
            mapping_identities[graph_name][str(key)] = identity
    return {
        "schema": "verification_gate_policy/v1",
        "machine": machine_metadata(),
        "runtime_env": _RUNTIME_ENV,
        "cpu_list": _RUNTIME_CPU_LIST,
        "graphs": graph_payload,
        "benchmarks": list(benchmarks),
        "algorithms": [
            {
                "key": key,
                "flags": flags,
            }
            for key, _name, flags in specs
        ],
        "mapping_identities": mapping_identities,
        "binaries": {
            benchmark: {
                "timed": _graph_fingerprint(BIN_DIR / benchmark),
                **(
                    {
                        "work": _graph_fingerprint(
                            BIN_WORK_DIR / benchmark,
                        ),
                    }
                    if benchmark in {
                        "bfs", "bc", "cc", "cc_sv", "sssp",
                    }
                    else {}
                ),
            }
            for benchmark in benchmarks
        },
        "cache_sim_binaries": (
            {"pr": _graph_fingerprint(BIN_SIM_DIR / "pr")}
            if "pr" in benchmarks
            else {}
        ),
        "algorithm_graph_exclusions": ALGORITHM_GRAPH_EXCLUSIONS,
        "algorithm_graph_exclusion_evidence":
            _algorithm_exclusion_evidence_payload(graphs),
        "sssp_policy": (
            {
                graph["name"]: sssp_policy_for_graph(graph["name"])
                for graph in graphs
            }
            if "sssp" in benchmarks else {}
        ),
    }


def verification_gate_id(
    graphs: list[dict],
    benchmarks: list[str],
    graph_dir: str,
) -> str:
    payload = _verification_gate_payload(
        graphs, benchmarks, graph_dir,
    )
    return _short_id(payload)


def run_verification_gate(
    *,
    graphs: list[dict],
    benchmarks: list[str],
    graph_dir: str,
    timeout: int,
    dry_run: bool,
) -> None:
    preflight_benchmark_policies(graphs, benchmarks, graph_dir)
    starting_policy = _verification_gate_payload(
        graphs,
        benchmarks,
        graph_dir,
    )
    gate_id = _short_id(starting_policy)
    out_dir = ensure_dir(RESULTS_DIR / "verification_gate")
    store = ResultsStore(
        out_dir / "verification_results.json",
        key_fields=["gate_id", "graph", "algo_key", "benchmark"],
    )
    specs = _overhead_algorithm_specs()
    specs.sort(key=lambda spec: 0 if str(spec[0]) == "0" else 1)
    if "sssp" in benchmarks and not any(
        str(key) == "0" for key, _name, _flags in specs
    ):
        raise RuntimeError(
            "SSSP verification requires SHUFFLED (algorithm 0) in the filter"
        )
    simulator_smoke = None
    if "pr" in benchmarks and not dry_run:
        smoke_cmd = [
            str(BIN_SIM_DIR / "pr"),
            "-f", str(PROJECT_ROOT / "scripts" / "test" / "data" / "tiny.el"),
            "-s", "-n", "1",
            "-i", str(PR_CONVERGENCE_MAX_ITERATIONS),
            "-t", str(PR_TOLERANCE),
            "-o", "0",
            "-v",
        ]
        smoke_output = run_cmd(
            smoke_cmd,
            timeout=timeout * VERIFICATION_TIMEOUT_MULTIPLIER,
            env=_cache_env(1024 * 1024),
            cpu_list=_cpu_list_for_threads(1),
        )
        if smoke_output is None:
            raise RuntimeError("Cache-simulator semantic smoke failed")
        smoke_timing = parse_timing(smoke_output)
        if smoke_timing.get("verification_state") != "pass":
            raise RuntimeError("Cache-simulator verifier did not pass")
        simulator_smoke = {
            "cmd": smoke_cmd,
            "verification_state": "pass",
            "binary": str(BIN_SIM_DIR / "pr"),
            "stdout_tail": smoke_output.strip().splitlines()[-40:],
        }

    for graph in graphs:
        graph_name = graph["name"]
        graph_path = resolve_graph_path(graph_name, graph_dir)
        if not _graph_provenance_valid(
            graph_path, graph_name=graph_name,
        ):
            raise RuntimeError(
                f"Verification gate requires current graph provenance: "
                f"{graph_name}"
            )
        for benchmark in benchmarks:
            verification_benchmark = _verification_benchmark_name(
                benchmark,
            )
            reference_sssp = next(
                (
                    row for row in store.results
                    if row.get("gate_id") == gate_id
                    and row.get("graph") == graph_name
                    and row.get("benchmark") == benchmark
                    and str(row.get("algo_key")) == "0"
                ),
                None,
            )
            for algo_key, algorithm_name, algorithm_flags in specs:
                if algorithm_exclusion_reason(graph_name, str(algo_key)):
                    continue
                key_row = {
                    "gate_id": gate_id,
                    "graph": graph_name,
                    "algo_key": str(algo_key),
                    "benchmark": benchmark,
                }
                if store.has(key_row):
                    continue
                flags, _reorder_time, mapping_identity = algo_flags_or_map(
                    algo_key, algorithm_flags, graph_name, graph_path,
                )
                cmd = build_benchmark_cmd(
                    verification_benchmark,
                    graph_path,
                    flags,
                    trials=1,
                )
                serial_verifier = (
                    benchmark != "sssp" or str(algo_key) == "0"
                )
                if serial_verifier:
                    cmd.append("-v")
                output = run_cmd(
                    cmd,
                    dry_run=dry_run,
                    timeout=timeout * VERIFICATION_TIMEOUT_MULTIPLIER,
                )
                if dry_run:
                    continue
                if output is None:
                    raise RuntimeError(
                        f"Verification failed for "
                        f"{graph_name}/{algo_key}/{benchmark}"
                    )
                timing = parse_timing(output)
                _validate_kernel_policy(
                    benchmark=verification_benchmark,
                    graph_name=graph_name,
                    timing=timing,
                    trials=1,
                    require_work=False,
                )
                work_timing: dict = {}
                work_cmd: list[str] = []
                work_output = ""
                if verification_benchmark in {
                    "bfs", "bc", "cc", "cc_sv", "sssp",
                }:
                    work_cmd = build_benchmark_cmd(
                        verification_benchmark,
                        graph_path,
                        flags,
                        trials=1,
                        work_metrics=True,
                    )
                    if serial_verifier:
                        work_cmd.append("-v")
                    work_output = run_cmd(
                        work_cmd,
                        dry_run=False,
                        timeout=timeout * VERIFICATION_TIMEOUT_MULTIPLIER,
                    )
                    if work_output is None:
                        raise RuntimeError(
                            f"Work-accounting run failed for "
                            f"{graph_name}/{algo_key}/{benchmark}"
                        )
                    work_timing = parse_timing(work_output)
                    _validate_kernel_policy(
                        benchmark=verification_benchmark,
                        graph_name=graph_name,
                        timing=work_timing,
                        trials=1,
                        require_work=True,
                    )
                    if (
                        serial_verifier
                        and work_timing.get("verification_state") != "pass"
                    ):
                        raise RuntimeError(
                            f"Work-binary verifier did not pass for "
                            f"{graph_name}/{algo_key}/{benchmark}"
                        )
                    for identity_key in (
                        "source_originals",
                        "source_internals",
                        "distance_fingerprints",
                        "weight_checksum",
                    ):
                        if (
                            identity_key in timing
                            and work_timing.get(identity_key)
                            != timing.get(identity_key)
                        ):
                            raise RuntimeError(
                                f"Timed/work identity mismatch for "
                                f"{graph_name}/{algo_key}/{benchmark}: "
                                f"{identity_key}"
                            )
                if serial_verifier:
                    if timing.get("verification_state") != "pass":
                        raise RuntimeError(
                            f"Verifier did not pass for "
                            f"{graph_name}/{algo_key}/{benchmark}"
                        )
                    verification_method = "semantic-verifier"
                elif benchmark == "sssp":
                    if reference_sssp is None:
                        raise RuntimeError(
                            f"Missing SHUFFLED SSSP reference for {graph_name}"
                        )
                    expected = {
                        "source_originals":
                            reference_sssp["source_originals"],
                        "distance_fingerprints":
                            reference_sssp["distance_fingerprints"],
                        "weight_checksum":
                            reference_sssp["weight_checksum"],
                    }
                    mismatches = {
                        key: (timing.get(key), value)
                        for key, value in expected.items()
                        if timing.get(key) != value
                    }
                    if mismatches:
                        raise RuntimeError(
                            f"SSSP verification signature mismatch for "
                            f"{graph_name}/{algo_key}: {mismatches}"
                        )
                    verification_method = "shuffled-answer-signature"
                    timing["verification_state"] = "pass"
                row = {
                    **key_row,
                    "algorithm": algorithm_name,
                    "verification_method": verification_method,
                    "mapping": mapping_identity,
                    "cmd": cmd,
                    "verified_at":
                        datetime.now().isoformat(timespec="seconds"),
                    "stdout_tail": output.strip().splitlines()[-40:],
                    "work_cmd": work_cmd,
                    "work_metrics": work_timing,
                    "work_stdout_tail": (
                        work_output.strip().splitlines()[-40:]
                        if work_output else []
                    ),
                    **timing,
                }
                store.add(row)
                if benchmark == "sssp" and str(algo_key) == "0":
                    reference_sssp = row

    if dry_run:
        return
    expected_cells = sum(
        1
        for graph in graphs
        for _benchmark in benchmarks
        for key, _name, _flags in specs
        if not algorithm_exclusion_reason(graph["name"], str(key))
    )
    gate_rows = [
        row for row in store.results
        if row.get("gate_id") == gate_id
    ]
    if len(gate_rows) != expected_cells:
        raise RuntimeError(
            f"Verification gate incomplete: {len(gate_rows)}/"
            f"{expected_cells} cells"
        )
    policy = _verification_gate_payload(
        graphs, benchmarks, graph_dir,
    )
    if policy != starting_policy:
        raise RuntimeError(
            "Verification inputs changed while the gate was running"
        )
    for row in gate_rows:
        expected_mapping = policy["mapping_identities"][
            row["graph"]
        ][str(row["algo_key"])]
        if row.get("mapping") != expected_mapping:
            raise RuntimeError(
                f"Verification mapping changed for "
                f"{row['graph']}/{row['algo_key']}"
            )
    manifest = {
        "schema": "verification_gate/v1",
        "gate_id": gate_id,
        "expected_cells": expected_cells,
        "completed_cells": len(gate_rows),
        "policy": policy,
        "completed_at":
            datetime.now().isoformat(timespec="seconds"),
        "simulator_smoke": simulator_smoke,
        "adjudication": {
            "semantic_verifier_passes": sum(
                row.get("verification_method") == "semantic-verifier"
                for row in gate_rows
            ),
            "shuffled_answer_signature_passes": sum(
                row.get("verification_method")
                == "shuffled-answer-signature"
                for row in gate_rows
            ),
            "total_passes": sum(
                row.get("verification_state") == "pass"
                for row in gate_rows
            ),
        },
    }
    save_json(manifest, out_dir / f"manifest-{gate_id}.json")


def require_verification_gate(
    graphs: list[dict],
    benchmarks: list[str],
    graph_dir: str,
) -> None:
    global _ACTIVE_VERIFICATION_GATE_ID
    expected_id = verification_gate_id(
        graphs, benchmarks, graph_dir,
    )
    gate_dir = RESULTS_DIR / "verification_gate"
    requested_policy = _verification_gate_payload(
        graphs, benchmarks, graph_dir,
    )
    manifest_paths = [
        gate_dir / f"manifest-{expected_id}.json",
        *sorted(gate_dir.glob("manifest-*.json")),
    ]
    manifests = []
    seen_paths = set()
    for path in manifest_paths:
        if path in seen_paths or not path.is_file():
            continue
        seen_paths.add(path)
        manifest = json.loads(path.read_text())
        policy = manifest.get("policy", {})
        exact = (
            manifest.get("gate_id") == expected_id
            and policy == requested_policy
        )
        superset = (
            policy.get("schema") == requested_policy.get("schema")
            and verification_machine_identity(policy.get("machine", {}))
            == verification_machine_identity(
                requested_policy.get("machine", {}),
            )
            and policy.get("runtime_env")
            == requested_policy.get("runtime_env")
            and policy.get("cpu_list") == requested_policy.get("cpu_list")
            and set(requested_policy.get("graphs", {})).issubset(
                policy.get("graphs", {})
            )
            and all(
                policy["graphs"].get(name)
                == requested_policy["graphs"][name]
                for name in requested_policy.get("graphs", {})
            )
            and set(requested_policy.get("benchmarks", [])).issubset(
                policy.get("benchmarks", [])
            )
            and all(
                algorithm in policy.get("algorithms", [])
                for algorithm in requested_policy.get("algorithms", [])
            )
            and all(
                policy.get("binaries", {}).get(name) == identity
                for name, identity in requested_policy.get(
                    "binaries", {},
                ).items()
            )
            and all(
                policy.get("cache_sim_binaries", {}).get(name)
                == identity
                for name, identity in requested_policy.get(
                    "cache_sim_binaries", {},
                ).items()
            )
            and all(
                policy.get("sssp_policy", {}).get(name)
                == requested_policy.get("sssp_policy", {}).get(name)
                for name in requested_policy.get("sssp_policy", {})
            )
            and all(
                policy.get("mapping_identities", {}).get(name, {}).get(key)
                == identity
                for name, identities in requested_policy.get(
                    "mapping_identities", {},
                ).items()
                for key, identity in identities.items()
            )
            and policy.get("algorithm_graph_exclusions")
            == requested_policy.get("algorithm_graph_exclusions")
            and all(
                policy.get(
                    "algorithm_graph_exclusion_evidence", {},
                ).get(graph_name, {}).get(algorithm_key)
                == evidence
                for graph_name, graph_evidence
                in requested_policy.get(
                    "algorithm_graph_exclusion_evidence", {},
                ).items()
                for algorithm_key, evidence in graph_evidence.items()
            )
        )
        if exact or superset:
            manifests.append(manifest)
    if not manifests:
        raise RuntimeError(
            "Stage 03 requires the untimed verification gate; run "
            "03_cpu_perf.py --verify-gate first"
        )
    results_path = (
        gate_dir / "verification_results.json"
    )
    rows = (
        json.loads(results_path.read_text())
        if results_path.is_file() else []
    )
    for manifest in manifests:
        manifest_id = manifest.get("gate_id")
        manifest_policy = manifest.get("policy", {})
        gate_rows = [
            row for row in rows
            if row.get("gate_id") == manifest_id
        ]
        rows_match_mappings = all(
            row.get("mapping")
            == manifest_policy.get("mapping_identities", {})
            .get(row.get("graph"), {})
            .get(str(row.get("algo_key")))
            for row in gate_rows
        )
        if (
            manifest.get("schema") == "verification_gate/v1"
            and manifest.get("completed_cells")
            == manifest.get("expected_cells")
            and len(gate_rows) == manifest.get("expected_cells")
            and rows_match_mappings
            and (
                "pr" not in requested_policy.get("benchmarks", [])
                or manifest.get("simulator_smoke", {}).get(
                    "verification_state"
                ) == "pass"
            )
        ):
            _ACTIVE_VERIFICATION_GATE_ID = str(manifest_id)
            return
    raise RuntimeError(
        "Verification gate is incomplete for this graph, benchmark, "
        "algorithm, or policy selection"
    )


def resolve_graph_path(graph_name: str, graph_dir: str, ext: str = ".sg") -> str:
    """Build the full path to a graph file.

    Checks two layouts:
      1. flat:   graph_dir/name.sg
      2. nested: graph_dir/name/name.sg   (created by auto-setup download)
    Returns the first that exists, or the flat path if neither does.
    """
    flat = Path(graph_dir) / f"{graph_name}{ext}"
    nested = Path(graph_dir) / graph_name / f"{graph_name}{ext}"
    if nested.exists():
        return str(nested)
    return str(flat)


# ---------------------------------------------------------------------------
# Per-kernel-run JSON sidecar (schema kernel_run/v1)
# Mirrors the reorder_meta/v1 sidecar in vldb_mappings/ but for kernel runs.
# Layout: results/vldb_runs/<graph>/<safe_algo_key>__<benchmark>.json
# ---------------------------------------------------------------------------

def _kernel_sidecar_path(
    graph_name: str, algo_key: str, benchmark: str, policy_id: str,
) -> Path:
    safe = str(algo_key).replace(":", "_").replace("/", "_")
    return KERNEL_RUNS_DIR / graph_name / f"{safe}__{benchmark}__{policy_id}.json"


def _save_kernel_sidecar(
    *,
    graph_name: str,
    algo_key: str,
    benchmark: str,
    cmd: list[str],
    output: Optional[str],
    timing: dict,
    cache: Optional[dict] = None,
    pregen_rt: float = 0.0,
    env: Optional[dict] = None,
    cpu_list: Optional[str] = None,
    sim: bool = False,
    policy_id: Optional[str] = None,
    cohort_id: Optional[str] = None,
    fixed_pr_iterations: int = PR_FIXED_ITERATIONS,
) -> None:
    """Write a full ``kernel_run/v1`` JSON record for a single kernel invocation.

    Captures everything needed to reproduce + analyse the run: command,
    env, parsed timings, cache metrics (if any), pregen reorder cost,
    and the tail of stdout.
    """
    effective_env = _effective_env(env)
    effective_cpu_list = cpu_list if cpu_list is not None else _RUNTIME_CPU_LIST
    trials = 1
    if "-n" in cmd:
        try:
            trials = int(cmd[cmd.index("-n") + 1])
        except (ValueError, IndexError):
            pass
    graph_path = cmd[cmd.index("-f") + 1]
    policy_id = policy_id or measurement_policy_id(
        "cache" if sim else "kernel",
        trials=trials,
        env=env,
        cpu_list=effective_cpu_list,
        extra={
            "sim": sim,
            "benchmark_runtime_policy":
                _benchmark_runtime_policy(
                    benchmark,
                    graph_name,
                    graph_path,
                    fixed_pr_iterations=fixed_pr_iterations,
                ),
        },
    )
    path = _kernel_sidecar_path(graph_name, algo_key, benchmark, policy_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec: dict = {
        "schema": "kernel_run/v2",
        "graph": graph_name,
        "algo_key": str(algo_key),
        "benchmark": benchmark,
        "policy_id": policy_id,
        "cohort_id": cohort_id,
        "sim_binary": sim,
        "cmd": [str(c) for c in cmd],
        "env": effective_env,
        "cpu_list": effective_cpu_list,
        "omp_num_threads": effective_env.get("OMP_NUM_THREADS"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "pregen_reorder_time": pregen_rt,
        "reorder_source": (
            "precomputed-map"
            if _command_uses_mapping(cmd)
            else "direct"
        ),
        "timing": timing,
        "cache": cache or {},
        "benchmark_runtime_policy":
            _benchmark_runtime_policy(
                benchmark,
                graph_name,
                graph_path,
                fixed_pr_iterations=fixed_pr_iterations,
            ),
        "stdout_tail": (output.strip().splitlines()[-40:] if output else []),
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json.tmp",
        dir=path.parent,
        delete=False,
    ) as tmp:
        json.dump(rec, tmp, indent=2)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _command_uses_mapping(cmd: list[str]) -> bool:
    return any(
        cmd[index] == "-o"
        and str(cmd[index + 1]).startswith("13:")
        for index in range(len(cmd) - 1)
    )


def _merge_pregen(
    timing: dict,
    pregen_rt: float,
    *,
    precomputed: bool,
) -> dict:
    """Apply the cached-mapping bookkeeping in one place.

    Mapping generation time is provenance, not the paper-facing reorder metric.
    Experiment 3 is the SSOT for reorder overhead.
    """
    if precomputed:
        map_load = timing.pop("mapping_generation_time", None)
        if isinstance(map_load, (int, float)):
            timing["map_load_time"] = map_load
        if "reorder_time" in timing:
            timing["mapping_application_time"] = timing.pop(
                "reorder_time"
            )
        timing["mapping_generation_time"] = pregen_rt
        timing["reorder_source"] = "precomputed-map"
    return timing


def _validate_kernel_policy(
    *,
    benchmark: str,
    graph_name: str,
    timing: dict,
    trials: int,
    require_work: bool = False,
    fixed_pr_iterations: int = PR_FIXED_ITERATIONS,
) -> None:
    if benchmark == "bfs":
        mismatches = {}
        for key in (
            "source_originals",
            "source_internals",
        ):
            values = timing.get(key)
            if not isinstance(values, list) or len(values) != trials:
                mismatches[key] = (
                    0 if not isinstance(values, list) else len(values),
                    trials,
                )
        if require_work:
            for key in (
                "bfs_td_edges",
                "bfs_bu_edges",
                "bfs_edges_examined",
                "bfs_steps",
            ):
                values = timing.get(key)
                if not isinstance(values, list) or len(values) != trials:
                    mismatches[key] = (
                        0 if not isinstance(values, list) else len(values),
                        trials,
                    )
        if mismatches:
            raise RuntimeError(
                f"BFS work/source policy mismatch for "
                f"{graph_name}: {mismatches}"
            )
        return
    if benchmark == "cc":
        if not require_work:
            return
        required = (
            "cc_sampled_edges",
            "cc_final_edges",
            "cc_compress_steps",
            "cc_skipped_vertices",
        )
        mismatches = {
            key: timing.get(key)
            for key in required
            if not isinstance(timing.get(key), list)
            or len(timing[key]) != trials
        }
        if mismatches:
            raise RuntimeError(
                f"CC work policy mismatch for {graph_name}: {mismatches}"
            )
        return
    if benchmark == "cc_sv":
        if not require_work:
            return
        required = (
            "cc_sv_iterations",
            "cc_sv_edges_examined",
            "cc_sv_compress_steps",
        )
        mismatches = {
            key: timing.get(key)
            for key in required
            if not isinstance(timing.get(key), list)
            or len(timing[key]) != trials
        }
        if mismatches:
            raise RuntimeError(
                f"CC-SV work policy mismatch for "
                f"{graph_name}: {mismatches}"
            )
        return
    if benchmark == "bc":
        mismatches = {}
        if require_work:
            for key in (
                "bc_bfs_edges",
                "bc_backprop_edges",
                "bc_max_depth",
            ):
                values = timing.get(key)
                if not isinstance(values, list) or len(values) != trials:
                    mismatches[key] = values
        expected_sources = trials * BC_SOURCE_ITERATIONS
        for key in ("source_originals", "source_internals"):
            values = timing.get(key)
            if (
                not isinstance(values, list)
                or len(values) != expected_sources
            ):
                mismatches[key] = (values, expected_sources)
        if mismatches:
            raise RuntimeError(
                f"BC work/source policy mismatch for "
                f"{graph_name}: {mismatches}"
            )
        return
    if benchmark in {
        "pr", "pr_spmv", "pr_convergence",
        "pr_spmv_convergence",
    }:
        mismatches = {}
        expected_mode = (
            "convergence"
            if benchmark in {
                "pr_convergence", "pr_spmv_convergence",
            }
            else "fixed-work"
        )
        if timing.get("pr_mode") != expected_mode:
            mismatches["pr_mode"] = (
                timing.get("pr_mode"), expected_mode,
            )
        iteration_counts = timing.get("iteration_counts")
        counts_valid = (
            isinstance(iteration_counts, list)
            and len(iteration_counts) == trials
        )
        if counts_valid and benchmark in {
            "pr_convergence", "pr_spmv_convergence",
        }:
            counts_valid = all(
                0 < count <= PR_CONVERGENCE_MAX_ITERATIONS
                for count in iteration_counts
            )
        elif counts_valid:
            counts_valid = all(
                count == fixed_pr_iterations
                for count in iteration_counts
            )
        if not counts_valid:
            mismatches["iteration_counts"] = (
                iteration_counts,
                (
                    f"1..{PR_CONVERGENCE_MAX_ITERATIONS}"
                    if benchmark in {
                        "pr_convergence", "pr_spmv_convergence",
                    }
                    else [fixed_pr_iterations] * trials
                ),
            )
        final_errors = timing.get("final_errors")
        if (
            not isinstance(final_errors, list)
            or len(final_errors) != trials
            or any(
                not isinstance(error, (int, float))
                or not math.isfinite(error)
                for error in (final_errors or [])
            )
        ):
            mismatches["final_errors"] = (final_errors, trials)
        elif benchmark in {
            "pr_convergence", "pr_spmv_convergence",
        } and any(
            error >= PR_TOLERANCE for error in final_errors
        ):
            mismatches["convergence"] = (
                final_errors, f"< {PR_TOLERANCE}",
            )
        if mismatches:
            raise RuntimeError(
                f"PageRank fixed-work policy mismatch for "
                f"{graph_name}/{benchmark}: {mismatches}"
            )
        return
    if benchmark != "sssp":
        return
    policy = sssp_policy_for_graph(graph_name)
    expected = {
        "weight_scheme": policy["weight_scheme"],
        "weight_checksum": policy["weight_checksum"],
        "delta": policy["delta"],
    }
    mismatches = {
        key: (timing.get(key), value)
        for key, value in expected.items()
        if timing.get(key) != value
    }
    for key in (
        "source_originals",
        "source_internals",
        "distance_fingerprints",
    ):
        values = timing.get(key)
        if not isinstance(values, list) or len(values) != trials:
            mismatches[key] = (
                0 if not isinstance(values, list) else len(values),
                trials,
            )
    if require_work:
        for key in (
            "sssp_edges_examined",
            "sssp_relax_successes",
            "sssp_frontier_entries",
            "sssp_bucket_iterations",
        ):
            values = timing.get(key)
            if not isinstance(values, list) or len(values) != trials:
                mismatches[key] = (
                    0 if not isinstance(values, list) else len(values),
                    trials,
                )
    if mismatches:
        raise RuntimeError(
            f"Weighted SSSP policy mismatch for {graph_name}: {mismatches}"
        )


def _answer_invariant_path(
    graph_name: str,
    benchmark: str,
    cohort_id: Optional[str],
) -> Path:
    safe_graph = graph_name.replace("/", "_")
    safe_cohort = (cohort_id or "adhoc").replace("/", "_")
    return (
        RESULTS_DIR / "answer_invariants" / safe_cohort
        / f"{safe_graph}__{benchmark}.json"
    )


def _validate_cross_ordering_answers(
    *,
    graph_name: str,
    algo_key: str,
    benchmark: str,
    cohort_id: Optional[str],
    timing: dict,
) -> None:
    if benchmark != "sssp":
        return
    signature = {
        "weight_scheme": timing["weight_scheme"],
        "weight_checksum": timing["weight_checksum"],
        "delta": timing["delta"],
        "source_originals": timing["source_originals"],
        "distance_fingerprints": timing["distance_fingerprints"],
    }
    path = _answer_invariant_path(graph_name, benchmark, cohort_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists():
            reference = json.loads(path.read_text())
            if reference.get("schema") != "answer_invariant/v1":
                raise RuntimeError(
                    f"Unsupported answer-invariant schema at {path}"
                )
            if reference.get("signature") != signature:
                raise RuntimeError(
                    "Cross-ordering SSSP answer mismatch for "
                    f"{graph_name}: reference={reference.get('algo_key')}, "
                    f"candidate={algo_key}"
                )
        else:
            payload = {
                "schema": "answer_invariant/v1",
                "graph": graph_name,
                "benchmark": benchmark,
                "cohort_id": cohort_id,
                "algo_key": algo_key,
                "signature": signature,
            }
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(path)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _run_kernel(
    *,
    cmd: list[str],
    graph_name: str,
    algo_key: str,
    benchmark: str,
    pregen_rt: float,
    dry_run: bool,
    timeout: int,
    env: Optional[dict] = None,
    cpu_list: Optional[str] = None,
    sim: bool = False,
    parse_cache: bool = False,
    policy_id: Optional[str] = None,
    cohort_id: Optional[str] = None,
    fixed_pr_iterations: int = PR_FIXED_ITERATIONS,
) -> tuple[dict, dict]:
    """One-stop helper: run a kernel binary, parse rich timings, apply cache
    bookkeeping, write the per-kernel sidecar, and return ``(timing, cache)``.

    Replaces the repeated ``run_cmd + parse_timing + pregen-merge`` block
    that previously appeared in every experiment function.
    """
    output = run_cmd(
        cmd, dry_run=dry_run, timeout=timeout, env=env, cpu_list=cpu_list,
    )
    if output is None:
        raise RuntimeError(
            f"Kernel command failed for {graph_name}/{algo_key}/{benchmark}"
        )
    timing = parse_timing(output)
    timing["timing_machine"] = timing_machine_metadata()
    trials = 1
    if "-n" in cmd:
        trials = int(cmd[cmd.index("-n") + 1])
    if not dry_run:
        _validate_kernel_policy(
            benchmark=benchmark,
            graph_name=graph_name,
            timing=timing,
            trials=trials,
            require_work=(
                Path(cmd[0]).parent == BIN_WORK_DIR
            ),
            fixed_pr_iterations=fixed_pr_iterations,
        )
        _validate_cross_ordering_answers(
            graph_name=graph_name,
            algo_key=algo_key,
            benchmark=benchmark,
            cohort_id=cohort_id,
            timing=timing,
        )
    if benchmark in {
        "pr", "pr_spmv", "pr_convergence",
        "pr_spmv_convergence",
    }:
        iteration_counts = timing.get("iteration_counts")
        if isinstance(iteration_counts, list):
            _nodes, directed_edges = get_graph_dimensions(
                cmd[cmd.index("-f") + 1],
            )
            timing["directed_edges_processed"] = [
                directed_edges * count
                for count in iteration_counts
            ]
    cache: dict = parse_cache_sim(output) if parse_cache else {}
    _merge_pregen(
        timing,
        pregen_rt,
        precomputed=_command_uses_mapping(cmd),
    )
    if not dry_run:
        _save_kernel_sidecar(
            graph_name=graph_name, algo_key=algo_key, benchmark=benchmark,
            cmd=cmd, output=output, timing=timing, cache=cache,
            pregen_rt=pregen_rt, env=env, cpu_list=cpu_list, sim=sim,
            policy_id=policy_id, cohort_id=cohort_id,
            fixed_pr_iterations=fixed_pr_iterations,
        )
    return timing, cache


def tune_sssp_deltas(
    *,
    graphs: list[dict],
    graph_dir: str,
    timeout: int,
    dry_run: bool,
    trials: int = SSSP_TUNING_TRIALS,
    freeze_policy: bool = False,
) -> dict[str, dict[str, Any]]:
    """Tune weighted SSSP on SHUFFLED and emit a reviewable policy artifact."""
    binary = BIN_DIR / "sssp"
    if not binary.exists() and not dry_run:
        raise RuntimeError("SSSP binary is missing; build bench/bin/sssp first")
    recommendations: dict[str, dict[str, Any]] = {}
    eligible_for_freeze = (
        not _PREVIEW_MODE
        and trials >= SSSP_TUNING_TRIALS
    )
    source_count = (
        SSSP_TUNING_SOURCES if eligible_for_freeze else max(1, trials)
    )
    source_repeats = SSSP_TUNING_REPEATS
    replicate_count = (
        SSSP_TUNING_REPLICATES if eligible_for_freeze else 1
    )
    trials_per_invocation = source_count * source_repeats
    total_trials = trials_per_invocation * replicate_count
    filename = (
        "sssp_delta_tuning.json"
        if eligible_for_freeze
        else "sssp_delta_tuning_preview.json"
    )
    path = RESULTS_DIR / filename
    measurement_identity: dict[str, Any] = {
        "schema": "sssp_delta_tuning/v2",
        "preview": _PREVIEW_MODE,
        "eligible_for_freeze": eligible_for_freeze,
        "weight_scheme": SSSP_WEIGHT_SCHEME,
        "delta_candidates": SSSP_DELTA_CANDIDATES,
        "trials_per_candidate": total_trials,
        "trials_per_invocation": trials_per_invocation,
        "source_count": source_count,
        "repeats_per_source": source_repeats,
        "invocation_replicates": replicate_count,
        "candidate_order_policy": SSSP_TUNING_ORDER_POLICY,
        "runtime_env": _effective_env(),
        "cpu_list": _RUNTIME_CPU_LIST,
        "measurement_protocol_id": _sssp_measurement_protocol_id(
            trials=total_trials,
            sources=source_count,
            repeats=source_repeats,
            replicates=replicate_count,
            candidates=list(SSSP_DELTA_CANDIDATES),
        ),
    }
    analysis_identity: dict[str, Any] = {
        "selection_rule_id": SSSP_SELECTION_RULE_ID,
        "practical_tie_ratio": SSSP_TUNING_PRACTICAL_TIE_RATIO,
        "paired_t_critical": SSSP_TUNING_T_CRITICAL_95_DF8,
    }
    artifact: dict[str, Any] = {
        **measurement_identity,
        **analysis_identity,
        "graphs": {},
    }
    reviewed_artifact_bytes: Optional[bytes] = None
    if not dry_run and path.is_file():
        reviewed_artifact_bytes = path.read_bytes()
        existing = json.loads(path.read_text())
        if all(
            existing.get(key) == value
            for key, value in measurement_identity.items()
        ):
            artifact = existing
            artifact.update(analysis_identity)
            recommendations.update(
                existing.get("recommendations", {})
            )
        else:
            raise RuntimeError(
                "Existing SSSP tuning artifact has a different measurement "
                f"protocol: {path}"
            )
    execution_orders = _sssp_candidate_execution_orders(
        list(SSSP_DELTA_CANDIDATES),
        replicate_count,
    )

    def checkpoint() -> None:
        if dry_run or freeze_policy:
            return
        if eligible_for_freeze:
            artifact["recommendations"] = recommendations
        else:
            artifact["preview_candidates"] = {
                name: data["selected_policy"]
                for name, data in artifact["graphs"].items()
                if "selected_policy" in data
            }
        ensure_dir(path.parent)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json.tmp",
            dir=path.parent,
            delete=False,
        ) as tmp:
            json.dump(artifact, tmp, indent=2)
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)

    def normalize_candidate_rows(
        rows: list[dict],
        *,
        graph_name: str,
    ) -> tuple[list[dict], tuple[str, list[int], list[str]]]:
        by_delta: dict[int, dict] = {}
        for raw_row in rows:
            delta = raw_row.get("delta")
            if type(delta) is not int or delta not in SSSP_DELTA_CANDIDATES:
                raise RuntimeError(
                    f"Invalid cached SSSP delta for {graph_name}: {delta}"
                )
            if delta in by_delta:
                raise RuntimeError(
                    f"Duplicate cached SSSP delta for {graph_name}: {delta}"
                )
            by_delta[delta] = dict(raw_row)
        if set(by_delta) != set(SSSP_DELTA_CANDIDATES):
            raise RuntimeError(
                f"Incomplete cached SSSP candidates for {graph_name}"
            )
        normalized = []
        reference: Optional[tuple[str, list[int], list[str]]] = None
        for delta in SSSP_DELTA_CANDIDATES:
            row = by_delta[delta]
            raw_invocations = row.get("invocations")
            if not isinstance(raw_invocations, list):
                raise RuntimeError(
                    f"Invalid cached SSSP invocations for "
                    f"{graph_name}, delta={delta}"
                )
            by_replicate: dict[int, dict] = {}
            for raw_invocation in raw_invocations:
                if not isinstance(raw_invocation, dict):
                    raise RuntimeError(
                        f"Invalid cached SSSP invocation for "
                        f"{graph_name}, delta={delta}"
                    )
                replicate = raw_invocation.get("replicate")
                if (
                    type(replicate) is not int
                    or replicate < 0
                    or replicate >= replicate_count
                    or replicate in by_replicate
                ):
                    raise RuntimeError(
                        f"Invalid cached SSSP replicate for "
                        f"{graph_name}, delta={delta}: {replicate}"
                    )
                by_replicate[replicate] = dict(raw_invocation)
            if set(by_replicate) != set(range(replicate_count)):
                raise RuntimeError(
                    f"Incomplete cached SSSP invocations for "
                    f"{graph_name}, delta={delta}"
                )

            invocations: list[dict] = []
            flat_times: list[float] = []
            flat_sources: list[int] = []
            flat_fingerprints: list[str] = []
            checksum: Optional[str] = None
            positions: list[int] = []
            for replicate in range(replicate_count):
                invocation = by_replicate[replicate]
                invocation_times = invocation.get("trial_times")
                invocation_sources = invocation.get("source_originals")
                invocation_fingerprints = invocation.get(
                    "distance_fingerprints"
                )
                invocation_checksum = invocation.get("weight_checksum")
                expected_position = execution_orders[replicate].index(delta)
                if (
                    invocation.get("execution_index") != expected_position
                    or not isinstance(invocation_times, list)
                    or len(invocation_times) != trials_per_invocation
                    or any(
                        type(value) not in {int, float}
                        or not math.isfinite(value) or value <= 0
                        for value in invocation_times
                    )
                    or not isinstance(invocation_sources, list)
                    or len(invocation_sources) != trials_per_invocation
                    or any(type(value) is not int for value in invocation_sources)
                    or not isinstance(invocation_fingerprints, list)
                    or len(invocation_fingerprints) != trials_per_invocation
                    or any(
                        type(value) is not str
                        or re.fullmatch(r"[0-9a-f]{32}", value) is None
                        for value in invocation_fingerprints
                    )
                    or type(invocation_checksum) is not str
                    or re.fullmatch(
                        r"[0-9a-f]{32}", invocation_checksum,
                    ) is None
                ):
                    raise RuntimeError(
                        f"Invalid cached SSSP invocation data for "
                        f"{graph_name}, delta={delta}, replicate={replicate}"
                    )
                source_blocks = [
                    invocation_sources[index:index + source_repeats]
                    for index in range(
                        0, trials_per_invocation, source_repeats,
                    )
                ]
                fingerprint_blocks = [
                    invocation_fingerprints[index:index + source_repeats]
                    for index in range(
                        0, trials_per_invocation, source_repeats,
                    )
                ]
                if (
                    any(
                        len(block) != source_repeats
                        or len(set(block)) != 1
                        for block in source_blocks
                    )
                    or any(
                        len(block) != source_repeats
                        or len(set(block)) != 1
                        for block in fingerprint_blocks
                    )
                    or (
                        eligible_for_freeze
                        and (
                            len(source_blocks) != SSSP_TUNING_SOURCES
                            or len({block[0] for block in source_blocks})
                            != SSSP_TUNING_SOURCES
                        )
                    )
                ):
                    raise RuntimeError(
                        f"Invalid cached SSSP source pattern for "
                        f"{graph_name}, delta={delta}, replicate={replicate}"
                    )
                if checksum is None:
                    checksum = invocation_checksum
                elif checksum != invocation_checksum:
                    raise RuntimeError(
                        f"SSSP weight checksum changed between invocations "
                        f"for {graph_name}, delta={delta}"
                    )
                invocation = {
                    "replicate": replicate,
                    "execution_index": expected_position,
                    "trial_times": [float(value) for value in invocation_times],
                    "source_originals": list(invocation_sources),
                    "distance_fingerprints":
                        list(invocation_fingerprints),
                    "weight_checksum": invocation_checksum,
                }
                invocations.append(invocation)
                positions.append(expected_position)
                flat_times.extend(invocation["trial_times"])
                flat_sources.extend(invocation["source_originals"])
                flat_fingerprints.extend(
                    invocation["distance_fingerprints"]
                )

            if checksum is None:
                raise RuntimeError(
                    f"No SSSP checksum for {graph_name}, delta={delta}"
                )
            answers = (
                checksum,
                flat_sources,
                flat_fingerprints,
            )
            if reference is None:
                reference = answers
            elif answers != reference:
                raise RuntimeError(
                    f"Cached SSSP answers changed with delta on {graph_name}"
                )
            source_medians = [
                statistics.median(
                    [
                        value
                        for invocation in invocations
                        for value in invocation["trial_times"][
                            source_index * source_repeats:
                            (source_index + 1) * source_repeats
                        ]
                    ]
                )
                for source_index in range(source_count)
            ]
            for key in list(row):
                if (
                    key.startswith("paired_")
                    or key in {
                        "statistically_indistinguishable",
                        "within_practical_band",
                        "in_tie_set",
                    }
                ):
                    row.pop(key, None)
            row.update({
                "invocations": invocations,
                "candidate_execution_positions": positions,
                "trial_times": flat_times,
                "source_median_times": source_medians,
                "invocation_median_times": [
                    statistics.median(invocation["trial_times"])
                    for invocation in invocations
                ],
                "median_time": statistics.median(flat_times),
                "mean_time": statistics.fmean(flat_times),
                "stddev_time": (
                    statistics.stdev(flat_times)
                    if len(flat_times) > 1 else 0.0
                ),
                "weight_checksum": checksum,
                "source_originals": flat_sources,
                "distance_fingerprints": flat_fingerprints,
            })
            normalized.append(row)
        if reference is None:
            raise RuntimeError(f"No SSSP candidates for {graph_name}")
        return normalized, reference

    for graph in graphs:
        graph_name = graph["name"]
        graph_path = resolve_graph_path(graph_name, graph_dir)
        if not Path(graph_path).is_file() and not dry_run:
            raise RuntimeError(f"Missing graph for SSSP tuning: {graph_path}")
        provenance_path = _graph_provenance_path(graph_path)
        provenance_valid = (
            False if dry_run else _graph_provenance_valid(
                graph_path,
                graph_name=graph_name,
            )
        )
        if eligible_for_freeze and not dry_run and not provenance_valid:
            raise RuntimeError(
                f"Final SSSP tuning requires current canonical provenance: "
                f"{graph_path}"
            )
        graph_info = (
            None if dry_run else _serialized_graph_info(graph_path)
        )
        conversion_policy_id = (
            json.loads(provenance_path.read_text()).get(
                "conversion_policy_id"
            )
            if provenance_path.is_file() else None
        )
        existing_graph = artifact.get("graphs", {}).get(graph_name)
        matching_existing_graph = (
            existing_graph
            and existing_graph.get("graph_info") == graph_info
            and existing_graph.get("conversion_policy_id")
            == conversion_policy_id
        )
        recommendations.pop(graph_name, None)
        candidate_rows = (
            [dict(row) for row in existing_graph.get("candidates", [])]
            if matching_existing_graph else []
        )
        existing_invocations = sum(
            len(row.get("invocations", []))
            for row in candidate_rows
            if isinstance(row, dict)
        )
        if existing_invocations:
            log.info(
                f"  Re-deriving SSSP policy for {graph_name} "
                f"from {existing_invocations} checkpointed invocations"
            )
        else:
            log.info(f"  Tuning weighted SSSP delta for {graph_name}")
        cached_deltas = [
            row.get("delta")
            for row in candidate_rows
            if isinstance(row, dict)
        ]
        if len(cached_deltas) != len(set(cached_deltas)):
            raise RuntimeError(
                f"Duplicate checkpointed SSSP candidates for {graph_name}"
            )
        rows_by_delta = {
            row.get("delta"): row
            for row in candidate_rows
            if isinstance(row, dict)
            and row.get("delta") in SSSP_DELTA_CANDIDATES
        }
        candidate_rows = []
        for delta in SSSP_DELTA_CANDIDATES:
            row = rows_by_delta.get(delta, {
                "delta": delta,
                "invocations": [],
            })
            row.setdefault("invocations", [])
            candidate_rows.append(row)

        reference_invocation_answers: Optional[
            tuple[str, list[int], list[str]]
        ] = None
        for row in candidate_rows:
            for invocation in row.get("invocations", []):
                answers = (
                    invocation.get("weight_checksum"),
                    invocation.get("source_originals"),
                    invocation.get("distance_fingerprints"),
                )
                if reference_invocation_answers is None:
                    reference_invocation_answers = answers
                elif answers != reference_invocation_answers:
                    raise RuntimeError(
                        "Checkpointed SSSP invocation answers disagree for "
                        f"{graph_name}"
                    )
        artifact["graphs"][graph_name] = {
            "graph_info": graph_info,
            "provenance_valid": provenance_valid,
            "conversion_policy_id": conversion_policy_id,
            "candidate_execution_order": execution_orders,
            "candidates": candidate_rows,
        }

        for replicate, delta_order in enumerate(execution_orders):
            for execution_index, delta in enumerate(delta_order):
                row = next(
                    item for item in candidate_rows
                    if item["delta"] == delta
                )
                if any(
                    invocation.get("replicate") == replicate
                    for invocation in row.get("invocations", [])
                ):
                    continue
                if freeze_policy:
                    raise RuntimeError(
                        "Freeze validation found missing SSSP invocation "
                        f"{graph_name}/delta={delta}/replicate={replicate}"
                    )
                cmd = [
                    str(binary),
                    "-f", graph_path,
                    "-s",
                    "-n", str(trials_per_invocation),
                    "-R", str(source_repeats),
                    "-W", SSSP_WEIGHT_SCHEME,
                    "-d", str(delta),
                    "-o", "0",
                ]
                output = run_cmd(cmd, dry_run=dry_run, timeout=timeout)
                if dry_run:
                    continue
                if output is None:
                    raise RuntimeError(
                        "SSSP delta tuning failed for "
                        f"{graph_name}, delta={delta}, replicate={replicate}"
                    )
                timing = parse_timing(output)
                invocation_times = timing.get("trial_times", [])
                invocation_sources = timing.get("source_originals", [])
                invocation_fingerprints = timing.get(
                    "distance_fingerprints", []
                )
                checksum = timing.get("weight_checksum")
                if (
                    len(invocation_times) != trials_per_invocation
                    or len(invocation_sources) != trials_per_invocation
                    or len(invocation_fingerprints) != trials_per_invocation
                    or not isinstance(checksum, str)
                ):
                    raise RuntimeError(
                        "Incomplete SSSP tuning output for "
                        f"{graph_name}, delta={delta}, replicate={replicate}"
                    )
                source_blocks = [
                    invocation_sources[index:index + source_repeats]
                    for index in range(
                        0, trials_per_invocation, source_repeats,
                    )
                ]
                fingerprint_blocks = [
                    invocation_fingerprints[index:index + source_repeats]
                    for index in range(
                        0, trials_per_invocation, source_repeats,
                    )
                ]
                if (
                    any(
                        len(block) != source_repeats
                        or len(set(block)) != 1
                        for block in source_blocks
                    )
                    or any(
                        len(block) != source_repeats
                        or len(set(block)) != 1
                        for block in fingerprint_blocks
                    )
                    or (
                        eligible_for_freeze
                        and (
                            len(source_blocks) != SSSP_TUNING_SOURCES
                            or len({block[0] for block in source_blocks})
                            != SSSP_TUNING_SOURCES
                        )
                    )
                ):
                    raise RuntimeError(
                        "Invalid SSSP source pattern for "
                        f"{graph_name}, delta={delta}, replicate={replicate}"
                    )
                answers = (
                    checksum,
                    invocation_sources,
                    invocation_fingerprints,
                )
                if reference_invocation_answers is None:
                    reference_invocation_answers = answers
                elif answers != reference_invocation_answers:
                    raise RuntimeError(
                        f"SSSP answers changed with delta on {graph_name}: "
                        f"delta={delta}, replicate={replicate}"
                    )
                row.setdefault("invocations", []).append({
                    "replicate": replicate,
                    "execution_index": execution_index,
                    "trial_times": invocation_times,
                    "source_originals": invocation_sources,
                    "distance_fingerprints": invocation_fingerprints,
                    "weight_checksum": checksum,
                })
                artifact["graphs"][graph_name]["candidates"] = candidate_rows
                checkpoint()

        if dry_run:
            continue
        candidate_rows, reference_answers = normalize_candidate_rows(
            candidate_rows,
            graph_name=graph_name,
        )
        artifact["graphs"][graph_name]["candidates"] = candidate_rows
        fastest, winner, tie_set = _select_sssp_delta(candidate_rows)
        if (
            eligible_for_freeze
            and fastest["delta"] == SSSP_DELTA_CANDIDATES[-1]
        ):
            artifact["graphs"][graph_name]["selection_rule"] = {
                "status": "upper-bound-not-bracketed",
                "fastest_delta": fastest["delta"],
            }
            checkpoint()
            raise RuntimeError(
                f"SSSP delta optimum is not bracketed for {graph_name}; "
                f"extend SSSP_DELTA_CANDIDATES above {fastest['delta']}"
            )
        policy = {
            "weight_scheme": SSSP_WEIGHT_SCHEME,
            "weight_checksum": reference_answers[0],
            "delta": winner["delta"],
            "conversion_policy_id": conversion_policy_id,
        }
        if eligible_for_freeze:
            recommendations[graph_name] = policy
        artifact["graphs"][graph_name].update({
            "conversion_policy_id": policy["conversion_policy_id"],
            "selected_policy": policy,
            "selection_rule": {
                "selection_rule_id": SSSP_SELECTION_RULE_ID,
                "fastest_delta": fastest["delta"],
                "fastest_median_time": fastest["median_time"],
                "selected_delta": winner["delta"],
                "selected_median_time": winner["median_time"],
                "tie_set": sorted(tie_set),
                "practical_tie_ratio":
                    SSSP_TUNING_PRACTICAL_TIE_RATIO,
                "paired_t_critical":
                    SSSP_TUNING_T_CRITICAL_95_DF8,
                "paired_blocks": len(fastest["trial_times"]),
                "candidate_execution_order": execution_orders,
                "lower_bound": {
                    "status": (
                        "domain-floor"
                        if winner["delta"] == SSSP_DELTA_CANDIDATES[0]
                        else "interior"
                    ),
                    "minimum_delta": SSSP_DELTA_CANDIDATES[0],
                    "generated_weight_range": [1, 255],
                },
                "rule": (
                    "fastest pooled median over nine source-by-replicate "
                    "blocks; choose the smallest delta within both the "
                    "one-sided paired 95% tie band and 2% practical band"
                ),
            },
        })
        log.info(
            f"    selected delta={winner['delta']} "
            f"(fastest={fastest['delta']}, tie_set={sorted(tie_set)}, "
            f"median {winner['median_time']:.6f}s)"
        )
        checkpoint()

    if not dry_run:
        artifact["recommendations"] = recommendations
        if not freeze_policy:
            checkpoint()
            log.info(f"  Wrote SSSP tuning artifact: {path}")
        if eligible_for_freeze:
            log.info(
                "  Candidate SSSP_POLICY entries pending layered review:\n"
                + json.dumps(recommendations, indent=2, sort_keys=True)
            )
        else:
            log.info(
                "  Preview tuning is diagnostic only and cannot freeze policy"
            )
        if freeze_policy:
            expected_graphs = {graph["name"] for graph in graphs}
            canonical_graphs = {graph["name"] for graph in EVAL_GRAPHS}
            if (
                not eligible_for_freeze
                or expected_graphs != canonical_graphs
                or set(recommendations) != expected_graphs
            ):
                raise RuntimeError(
                    "SSSP policy freeze requires the complete canonical "
                    "11-graph final-path recommendation set"
                )
            if reviewed_artifact_bytes is None:
                raise RuntimeError(
                    "SSSP policy freeze requires an existing reviewed artifact"
                )
            reviewed_artifact = json.loads(reviewed_artifact_bytes)
            canonical_reviewed = json.dumps(
                reviewed_artifact,
                sort_keys=True,
                separators=(",", ":"),
            )
            canonical_rederived = json.dumps(
                artifact,
                sort_keys=True,
                separators=(",", ":"),
            )
            if canonical_reviewed != canonical_rederived:
                raise RuntimeError(
                    "Reviewed SSSP artifact does not match validation-only "
                    "re-derivation"
                )
            snapshot_bytes = json.dumps(
                _sssp_policy_validation_snapshot(reviewed_artifact),
                indent=2,
                sort_keys=True,
            ).encode()
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".json.tmp",
                dir=SSSP_TUNING_SNAPSHOT_PATH.parent,
                delete=False,
            ) as tmp:
                tmp.write(snapshot_bytes)
                snapshot_tmp_path = Path(tmp.name)
            snapshot_tmp_path.replace(SSSP_TUNING_SNAPSHOT_PATH)
            policy_payload = {
                "schema": "sssp_policy/v1",
                "selection_rule_id": SSSP_SELECTION_RULE_ID,
                "policies": recommendations,
            }
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json.tmp",
                dir=SSSP_POLICY_PATH.parent,
                delete=False,
            ) as tmp:
                json.dump(policy_payload, tmp, indent=2, sort_keys=True)
                tmp_path = Path(tmp.name)
            tmp_path.replace(SSSP_POLICY_PATH)
            log.info(f"  Froze reviewed SSSP policy: {SSSP_POLICY_PATH}")
    return recommendations


# ============================================================================
# Experiment 1: Cache Performance Analysis
# ============================================================================


def _validate_cache_geometry(
    name: str, size_bytes: int, ways: int, line_size: int = 64,
) -> None:
    set_bytes = ways * line_size
    if size_bytes < set_bytes or size_bytes % set_bytes != 0:
        raise ValueError(
            f"{name}: capacity must be an exact multiple of line_size * ways"
        )
    sets = size_bytes // set_bytes
    if sets & (sets - 1):
        raise ValueError(f"{name}: cache set count must be a power of two")


def _cache_env(cache_size: int) -> dict[str, str]:
    """Return a reproducible single-thread cache hierarchy for one capacity."""
    l1_size = min(cache_size, 32 * 1024)
    l2_size = min(cache_size, 256 * 1024)
    l3_ways = 11 if cache_size == 22 * 1024**2 else 16
    _validate_cache_geometry("L1", l1_size, 8)
    _validate_cache_geometry("L2", l2_size, 8)
    _validate_cache_geometry("L3", cache_size, l3_ways)
    env = {
        "OMP_NUM_THREADS": "1",
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
        "OMP_DYNAMIC": "FALSE",
        "CACHE_L1_SIZE": str(l1_size),
        "CACHE_L2_SIZE": str(l2_size),
        "CACHE_L3_SIZE": str(cache_size),
        "CACHE_L3_WAYS": str(l3_ways),
        "CACHE_LINE_SIZE": "64",
        "CACHE_POLICY": (
            "LRU" if _CACHE_MODE == "accurate" else "CLOCK"
        ),
        "CACHE_MULTICORE": "0",
        "CACHE_SAMPLED": "0",
        "CACHE_ULTRAFAST": "0",
        "CACHE_FAST": "0",
        "ECG_PREFETCH_MODE": "0",
        "ECG_PREFETCH_WINDOW": "8",
        "ECG_PFX_BITS": "0",
    }
    if _CACHE_MODE == "fast":
        env["CACHE_FAST"] = "1"
    elif _CACHE_MODE == "ultrafast":
        env["CACHE_ULTRAFAST"] = "1"
    elif _CACHE_MODE == "sampled":
        env["CACHE_SAMPLED"] = "1"
        env["CACHE_SAMPLE_RATE"] = str(_CACHE_SAMPLE_RATE)
    return env


def _pr_working_set(graph_path: str) -> dict[str, int | float]:
    """Estimate the cache-simulated PR footprint from exact graph dimensions."""
    nodes, edges = get_graph_dimensions(graph_path)
    property_bytes = modeled_property_bytes("pr", nodes, edges)
    tracked_edge_bytes = edges * 4
    return {
        "nodes": nodes,
        "edges": edges,
        "property_working_set_bytes": property_bytes,
        "tracked_edge_array_bytes": tracked_edge_bytes,
        "estimated_modeled_working_set_bytes":
            property_bytes + tracked_edge_bytes,
    }


def exp1_cache_performance(
    graphs: list[dict], benchmarks: list[str], trials: int,
    timeout: int, dry_run: bool, graph_dir: str = ".",
) -> None:
    log.info("=" * 60)
    log.info("EXPERIMENT 1: Cache Performance Analysis")
    log.info("=" * 60)
    ensure_campaign(graphs, graph_dir, 1, benchmarks, CACHE_TRIALS)

    out_dir = ensure_dir(RESULTS_DIR / "exp1_cache")
    store = ResultsStore(
        out_dir / "cache_results.json",
        key_fields=[
            "graph", "algo_key", "benchmark", "cache_size_bytes",
            "cache_schema", "policy_id",
        ],
    )

    # Use PR for cache simulation (canonical benchmark)
    cache_bench = "pr"
    if cache_bench not in benchmarks:
        cache_bench = benchmarks[0]
    cache_sizes = (
        _CACHE_SIZE_OVERRIDE
        or (CACHE_SIZES_PREVIEW if _PREVIEW_MODE else CACHE_SIZES)
    )
    algo_list = _cache_algorithm_specs()
    cache_cpu_list = _cpu_list_for_threads(1)

    for graph in graphs:
        gname = graph["name"]
        log.info(f"  Graph: {gname}")
        gpath = resolve_graph_path(gname, graph_dir)
        working_set = _pr_working_set(gpath)

        for akey, aname, aflags in algo_list:
            flags, pregen_rt, mapping_identity = algo_flags_or_map(
                akey, aflags, gname, gpath,
            )
            for cache_size in cache_sizes:
                key_row = {
                    "graph": gname,
                    "algo_key": akey,
                    "benchmark": cache_bench,
                    "cache_size_bytes": cache_size,
                    "cache_schema": "cache_metrics/v2",
                    "cache_mode": _CACHE_MODE,
                }
                env = _cache_env(cache_size)
                cohort_id, policy_id = _kernel_policy_ids(
                    graph_path=gpath,
                    kind="cache",
                    trials=CACHE_TRIALS,
                    executable=BIN_SIM_DIR / cache_bench,
                    env=env,
                    cpu_list=cache_cpu_list,
                    extra={
                        "mapping": mapping_identity,
                        "cache_size_bytes": cache_size,
                        "cache_mode": _CACHE_MODE,
                        "cache_sample_rate": (
                            _CACHE_SAMPLE_RATE if _CACHE_MODE == "sampled" else 1
                        ),
                        "cache_iterations": CACHE_PR_ITERATIONS,
                    },
                    fixed_pr_iterations=CACHE_PR_ITERATIONS,
                )
                key_row["policy_id"] = policy_id
                cell_key = f"{gname}|{akey}|{cache_bench}|{cache_size}"
                if store.has(key_row):
                    continue
                cmd = build_benchmark_cmd(
                    cache_bench, gpath, flags, CACHE_TRIALS, sim=True,
                )
                if "-i" in cmd:
                    cmd[cmd.index("-i") + 1] = str(
                        CACHE_PR_ITERATIONS
                    )
                log.info(
                    f"    {aname}: {cache_size // 1024} KiB ({_CACHE_MODE})"
                )
                timing, cache = _run_kernel(
                    cmd=cmd, graph_name=gname, algo_key=akey,
                    benchmark=cache_bench, pregen_rt=pregen_rt,
                    dry_run=dry_run, timeout=timeout, env=env,
                    cpu_list=cache_cpu_list, sim=True, parse_cache=True,
                    policy_id=policy_id, cohort_id=cohort_id,
                    fixed_pr_iterations=CACHE_PR_ITERATIONS,
                )
                overall_hit = cache.get("overall_hit_rate")
                overall_miss = (
                    100.0 - overall_hit
                    if isinstance(overall_hit, (int, float)) else None
                )
                property_bytes = int(working_set["property_working_set_bytes"])
                store.add({
                    **key_row,
                    "algorithm": aname,
                    "cache_l1_bytes": int(env["CACHE_L1_SIZE"]),
                    "cache_l2_bytes": int(env["CACHE_L2_SIZE"]),
                    "cache_l3_bytes": int(env["CACHE_L3_SIZE"]),
                    "cache_l3_ways": int(env["CACHE_L3_WAYS"]),
                    "cache_iterations": CACHE_PR_ITERATIONS,
                    "cache_sample_rate": (
                        _CACHE_SAMPLE_RATE if _CACHE_MODE == "sampled" else 1
                    ),
                    "policy_id": policy_id,
                    "cohort_id": cohort_id,
                    "cell_key": cell_key,
                    "measured_at": datetime.now().isoformat(timespec="seconds"),
                    "cache_to_property_ratio": (
                        cache_size / property_bytes if property_bytes > 0 else None
                    ),
                    "overall_miss_rate": overall_miss,
                    **working_set,
                    **timing,
                    **cache,
                })

    log.info(f"  exp1: {len(store.results)} total result rows in {store.path.name}")


# ============================================================================
# Experiment 2: Kernel Speedup
# ============================================================================


def _run_pr_convergence_diagnostics(
    *,
    graphs: list[dict],
    trials: int,
    timeout: int,
    dry_run: bool,
    graph_dir: str,
) -> None:
    out_dir = ensure_dir(RESULTS_DIR / "exp2_speedup")
    store = ResultsStore(
        out_dir / "pr_convergence_results.json",
        key_fields=["graph", "algo_id", "benchmark", "policy_id"],
    )
    for graph in graphs:
        graph_name = graph["name"]
        graph_path = resolve_graph_path(graph_name, graph_dir)
        for algo_key, algorithm_name, algorithm_flags in (
            _paper_algorithm_specs(include_compose=True)
        ):
            algo_id = _algo_identity(algo_key)
            flags, pregen_rt, mapping_identity = algo_flags_or_map(
                algo_key, algorithm_flags, graph_name, graph_path,
            )
            cohort_id, policy_id = _kernel_policy_ids(
                graph_path=graph_path,
                kind="pr-convergence",
                trials=trials,
                executable=BIN_DIR / "pr",
                benchmark_name="pr_convergence",
                extra={"mapping": mapping_identity},
            )
            key_row = {
                "graph": graph_name,
                "algo_id": algo_id,
                "benchmark": "pr_convergence",
                "policy_id": policy_id,
            }
            if store.has(key_row):
                continue
            cmd = build_benchmark_cmd(
                "pr_convergence",
                graph_path,
                flags,
                trials,
            )
            timing, _ = _run_kernel(
                cmd=cmd,
                graph_name=graph_name,
                algo_key=algo_key,
                benchmark="pr_convergence",
                pregen_rt=pregen_rt,
                dry_run=dry_run,
                timeout=timeout,
                policy_id=policy_id,
                cohort_id=cohort_id,
            )
            store.add({
                **key_row,
                "algorithm": algorithm_name,
                "cohort_id": cohort_id,
                "cell_key":
                    f"{graph_name}|{algo_id}|pr_convergence",
                "measured_at":
                    datetime.now().isoformat(timespec="seconds"),
                **timing,
            })


def exp2_kernel_speedup(
    graphs: list[dict], benchmarks: list[str], trials: int,
    timeout: int, dry_run: bool, graph_dir: str = ".",
) -> None:
    log.info("=" * 60)
    log.info("EXPERIMENT 2: Kernel Speedup")
    log.info("=" * 60)
    ensure_campaign(graphs, graph_dir, 2, benchmarks, trials)

    out_dir = ensure_dir(RESULTS_DIR / "exp2_speedup")
    store = ResultsStore(
        out_dir / "speedup_results.json",
        key_fields=["graph", "algo_id", "benchmark", "policy_id"],
    )
    gate_id = (
        _ACTIVE_VERIFICATION_GATE_ID
        or verification_gate_id(graphs, benchmarks, graph_dir)
    )

    for graph in graphs:
        gname = graph["name"]
        log.info(f"  Graph: {gname}")

        for bench in benchmarks:
            log.info(f"    Benchmark: {bench}")
            gpath = resolve_graph_path(gname, graph_dir)

            for algo_key, aname, aflags in _paper_algorithm_specs(
                include_compose=True,
            ):
                if algorithm_exclusion_reason(gname, str(algo_key)):
                    continue
                algo_id = _algo_identity(algo_key)
                cell_key = f"{gname}|{algo_id}"
                flags, pregen_rt, mapping_identity = algo_flags_or_map(
                    algo_key, aflags, gname, gpath,
                )
                cohort_id, policy_id = _kernel_policy_ids(
                    graph_path=gpath,
                    kind="kernel",
                    trials=trials,
                    executable=BIN_DIR / bench,
                    extra={"mapping": mapping_identity},
                )
                cell_key = f"{gname}|{algo_id}|{bench}"
                key_row = {
                    "graph": gname, "algo_id": algo_id, "benchmark": bench,
                    "policy_id": policy_id,
                }
                if store.has(key_row):
                    continue
                cmd = build_benchmark_cmd(bench, gpath, flags, trials)
                timing, _ = _run_kernel(
                    cmd=cmd, graph_name=gname, algo_key=algo_key,
                    benchmark=bench, pregen_rt=pregen_rt,
                    dry_run=dry_run, timeout=timeout,
                    policy_id=policy_id, cohort_id=cohort_id,
                )
                store.add({
                    "graph": gname, "algorithm": aname, "benchmark": bench,
                    "algo_id": algo_id,
                    "e2e_join_context_id": _e2e_join_context_id(),
                    "mapping_identity_id":
                        _mapping_identity_id(mapping_identity),
                    "mapping_identity": mapping_identity,
                    "policy_id": policy_id,
                    "cohort_id": cohort_id,
                    "cell_key": cell_key,
                    "verification_gate_id": gate_id,
                    "verification_gate_status": "pass",
                    "verification_gate_method": (
                        "shuffled-answer-signature"
                        if bench == "sssp" and str(algo_key) != "0"
                        else "semantic-verifier"
                    ),
                    "measured_at": datetime.now().isoformat(timespec="seconds"),
                    **timing,
                })

    if "pr" in benchmarks:
        _run_pr_convergence_diagnostics(
            graphs=graphs,
            trials=trials,
            timeout=timeout,
            dry_run=dry_run,
            graph_dir=graph_dir,
        )

    log.info(f"  exp2: {len(store.results)} total result rows in {store.path.name}")


# ============================================================================
# Experiment 3: Reorder Overhead & Amortization
# ============================================================================


def _measure_reorder(
    cmd: list[str],
    *,
    repeats: int,
    dry_run: bool,
    timeout: int,
    allow_timeout: bool = False,
) -> dict:
    timings: list[dict] = []
    for _ in range(repeats):
        failure_details: dict[str, Any] = {}
        output = run_cmd(
            cmd,
            dry_run=dry_run,
            timeout=timeout,
            failure_details=failure_details,
        )
        if output is None:
            if (
                allow_timeout
                and failure_details.get("failure_mode") == "timeout"
                and failure_details.get("elapsed_seconds", 0)
                >= 0.95 * timeout
            ):
                return {
                    "overhead_timeout": True,
                    "timeout_seconds": timeout,
                    "elapsed_seconds":
                        failure_details.get("elapsed_seconds"),
                    "failure_mode": "timeout",
                    "timing_machine": timing_machine_metadata(),
                }
            raise RuntimeError(
                f"Reorder command failed "
                f"({failure_details.get('failure_mode', 'unknown')}): "
                f"{' '.join(cmd)}"
            )
        timing = parse_timing(output)
        timings.append(timing)
    result = dict(timings[-1]) if timings else {}
    for metric in (
        "representation_build_time",
        "reorder_core_time",
        "reorder_time",
        "mapping_generation_time",
        "reorder_validation_time",
        "reorder_apply_time",
        "total_preprocessing_time",
    ):
        values = [
            float(timing[metric])
            for timing in timings
            if isinstance(timing.get(metric), (int, float))
            and timing[metric] >= 0
        ]
        if not values:
            continue
        stem = metric.removesuffix("_time")
        result[f"{stem}_times"] = values
        result[metric] = statistics.median(values)
        result[f"{stem}_mean_time"] = statistics.fmean(values)
        result[f"{stem}_stddev_time"] = (
            statistics.stdev(values) if len(values) > 1 else None
        )
    result["timing_machine"] = timing_machine_metadata()
    return result


def _mapping_sidecar_timing(meta: dict) -> Optional[dict]:
    """Aggregate Stage-02 draw timings without regenerating the mapping."""
    draw_records = meta.get("mapping_draws")
    if not isinstance(draw_records, list) or not draw_records:
        return None
    metrics = (
        "mapping_generation_time",
        "reorder_validation_time",
        "reorder_apply_time",
    )
    values: dict[str, list[float]] = {metric: [] for metric in metrics}
    end_to_end: list[float] = []
    for record in draw_records:
        if not isinstance(record, dict):
            return None
        row: dict[str, float] = {}
        for metric in metrics:
            value = record.get(metric)
            if not isinstance(value, (int, float)):
                return None
            if metric == "mapping_generation_time":
                if value <= 0:
                    return None
            elif value < 0:
                return None
            row[metric] = float(value)
            values[metric].append(float(value))
        end_to_end.append(sum(row.values()))

    result = {
        "timing_source": "stage02-sidecar",
        "mapping_timing_machine": {
            "state": "unrecorded",
            "recorded_at": meta.get("timestamp"),
        },
        "reorder_time": end_to_end[0],
        "reorder_times": end_to_end,
        "reorder_mean_time": statistics.fmean(end_to_end),
        "reorder_stddev_time": (
            statistics.stdev(end_to_end) if len(end_to_end) > 1 else None
        ),
    }
    for metric, metric_values in values.items():
        stem = metric.removesuffix("_time")
        result[metric] = metric_values[0]
        result[f"{stem}_times"] = metric_values
        result[f"{stem}_mean_time"] = statistics.fmean(metric_values)
        result[f"{stem}_stddev_time"] = (
            statistics.stdev(metric_values)
            if len(metric_values) > 1 else None
        )
    result["reorder_core_time"] = result["mapping_generation_time"]
    result["reorder_core_times"] = result["mapping_generation_times"]
    result["reorder_core_mean_time"] = result[
        "mapping_generation_mean_time"
    ]
    result["reorder_core_stddev_time"] = result[
        "mapping_generation_stddev_time"
    ]
    for metric in (
        "representation_build_time",
        "total_preprocessing_time",
    ):
        metric_values = [
            float(record[metric])
            for record in draw_records
            if isinstance(record.get(metric), (int, float))
            and record[metric] >= 0
        ]
        if len(metric_values) != len(draw_records):
            continue
        stem = metric.removesuffix("_time")
        result[metric] = metric_values[0]
        result[f"{stem}_times"] = metric_values
        result[f"{stem}_mean_time"] = statistics.fmean(metric_values)
        result[f"{stem}_stddev_time"] = (
            statistics.stdev(metric_values)
            if len(metric_values) > 1 else None
        )
    apply_passes = draw_records[0].get(
        "reorder_apply_time_passes"
    )
    if isinstance(apply_passes, list):
        result["reorder_apply_time_passes"] = [
            float(value)
            for value in apply_passes
            if isinstance(value, (int, float))
        ]
    return result


def _overhead_checkpoint_is_complete(
    rows: list[dict],
    timeout: int,
    *,
    require_equivalence: bool = False,
) -> bool:
    if any(
        not row.get("overhead_timeout")
        and not row.get("weighted_apply_timeout")
        and (
            not require_equivalence
            or row.get("mapping_equivalence_checked") is True
        )
        for row in rows
    ):
        return True
    return any(
        (
            row.get("overhead_timeout")
            and (row.get("timeout_seconds") or 0) >= timeout
        )
        or (
            row.get("weighted_apply_timeout")
            and (row.get("weighted_timeout_seconds") or 0)
            >= timeout
            and (
                not require_equivalence
                or row.get("mapping_equivalence_checked") is True
            )
        )
        for row in rows
    )


def exp3_reorder_overhead(
    graphs: list[dict], benchmarks: list[str], trials: int,
    timeout: int, dry_run: bool, graph_dir: str = ".",
) -> None:
    log.info("=" * 60)
    log.info("EXPERIMENT 3: Reorder Overhead & Amortization")
    log.info("=" * 60)

    speedup_path = RESULTS_DIR / "exp2_speedup" / "speedup_results.json"
    if not speedup_path.is_file():
        raise RuntimeError(
            "Experiment 3 requires completed Experiment 2 results"
        )
    speedup_rows = json.loads(speedup_path.read_text())
    speedup_join_ids = {
        row.get("e2e_join_context_id")
        for row in speedup_rows
        if isinstance(row.get("e2e_join_context_id"), str)
    }
    current_join_id = _e2e_join_context_id()
    if speedup_join_ids != {current_join_id}:
        raise RuntimeError(
            "Experiment 3 runtime policy does not match Experiment 2: "
            f"expected {sorted(speedup_join_ids)}, found "
            f"{current_join_id}"
        )

    out_dir = ensure_dir(RESULTS_DIR / "exp3_overhead")
    store = ResultsStore(
        out_dir / "overhead_results.json",
        key_fields=["graph", "algo_id", "policy_id"],
    )
    live_repeats = 1
    weighted_apply_repeats = 1
    ensure_campaign(graphs, graph_dir, 3, benchmarks, live_repeats)
    expected_promoted: set[tuple[str, str]] = set()
    promoted_cohort_ids: dict[tuple[str, str], str] = {}
    promoted_mapping_identities: dict[
        tuple[str, str], dict[str, Any]
    ] = {}
    for graph in graphs:
        gname = graph["name"]
        for algo_key, _name, _flags in _overhead_algorithm_specs():
            meta = _load_reorder_meta(gname, algo_key)
            if (
                str(algo_key) == "9:csr"
                and gname in PROMOTED_GORDER_GRAPHS
                and not algorithm_exclusion_reason(
                    gname,
                    str(algo_key),
                )
            ):
                expected_promoted.add((gname, str(algo_key)))
    if expected_promoted:
        log.info(
            f"  Promoted mappings requiring live equivalence checks: "
            f"{len(expected_promoted)}"
        )
    current_cohort_id = measurement_cohort_id(
        "reorder",
        trials=live_repeats,
    )
    log.info(f"  Reorder cohort: {current_cohort_id}")

    for graph in graphs:
        gname = graph["name"]
        log.info(f"  Graph: {gname}")

        converter = BIN_DIR / "converter"
        converter_identity = _graph_fingerprint(converter)
        for algo_key, aname, aflags in _overhead_algorithm_specs():
            if algo_key == "0":
                continue  # No reorder for original
            exclusion = algorithm_exclusion_reason(gname, str(algo_key))
            if exclusion:
                log.info(
                    f"    EXCLUDED {algo_key}: {exclusion}"
                )
                continue
            gpath = resolve_graph_path(gname, graph_dir, ext=".sg")
            if not Path(gpath).exists():
                gpath = resolve_graph_path(gname, graph_dir, ext=".el")
            mapping_identity: dict[str, Any] = {"dry_run": True}
            map_flags: list[str] = []
            meta: dict = {}
            reused_timing: Optional[dict] = None
            if not dry_run:
                map_flags, _pregen_time, mapping_identity = algo_flags_or_map(
                    algo_key, aflags, gname, gpath,
                )
                meta = _load_reorder_meta(gname, algo_key)
                reused_timing = _mapping_sidecar_timing(meta)
            is_promoted = (
                str(algo_key) == "9:csr"
                and gname in PROMOTED_GORDER_GRAPHS
            )
            reuse_sidecar = (
                gname in REORDER_TIMING_REUSE_GRAPHS
                and reused_timing is not None
                and not is_promoted
                and str(algo_key)
                not in REORDER_TIMING_ANCHOR_ALGOS
            )
            timing_source = (
                "stage02-sidecar"
                if reuse_sidecar
                else "live-final"
            )
            cohort_id, policy_id = _kernel_policy_ids(
                graph_path=gpath,
                kind="reorder",
                trials=live_repeats,
                executable=converter,
                extra={
                    "mapping_timing_source": timing_source,
                    "live_repeats": live_repeats,
                    "weighted_apply_repeats": weighted_apply_repeats,
                    "timeout_seconds": timeout,
                    "apply_profiles": ["unweighted", "weighted"],
                    "mapping": mapping_identity,
                },
            )
            algo_id = _algo_identity(algo_key)
            if is_promoted:
                promoted_cohort_ids[(gname, str(algo_key))] = cohort_id
                promoted_mapping_identities[
                    (gname, str(algo_key))
                ] = mapping_identity
            cell_key = f"{gname}|{algo_id}"
            key_row = {
                "graph": gname,
                "algo_id": algo_id,
                "policy_id": policy_id,
                "cohort_id": cohort_id,
                "cell_key": cell_key,
                "measured_at": datetime.now().isoformat(timespec="seconds"),
            }
            existing_rows = [
                row for row in store.results
                if row.get("graph") == gname
                and row.get("algo_id") == algo_id
                and row.get("mapping_identity") == mapping_identity
                and row.get("e2e_join_context_id")
                == _e2e_join_context_id()
                and row.get("cohort_id") == cohort_id
                and row.get("converter_identity") == converter_identity
                and row.get("timing_source") == timing_source
            ]
            if _overhead_checkpoint_is_complete(
                existing_rows,
                timeout,
                require_equivalence=is_promoted,
            ):
                continue
            if reuse_sidecar:
                timing = reused_timing
                log.info(
                    f"    Reusing Stage-02 timing for {algo_key} "
                    f"({len(timing['mapping_generation_times'])} draw(s))"
                )
            else:
                cmd = [str(converter), "-f", gpath, "-s", *aflags]
                equivalence_scratch: Optional[Path] = None
                equivalence_result: Optional[bool] = None
                if is_promoted:
                    safe_algo = str(algo_key).replace(":", "_")
                    equivalence_dir = ensure_dir(
                        out_dir / "equivalence_checks" / gname
                    )
                    equivalence_scratch = (
                        equivalence_dir / f"{safe_algo}.live.lo"
                    )
                    equivalence_scratch.unlink(missing_ok=True)
                    cmd.extend(["-q", str(equivalence_scratch)])
                try:
                    timing = _measure_reorder(
                        cmd,
                        repeats=live_repeats,
                        dry_run=dry_run,
                        timeout=timeout,
                        allow_timeout=True,
                    )
                    if (
                        not dry_run
                        and equivalence_scratch is not None
                        and not timing.get("overhead_timeout")
                    ):
                        promoted_path = _lo_path(gname, algo_key)
                        equivalent = filecmp.cmp(
                            equivalence_scratch,
                            promoted_path,
                            shallow=False,
                        )
                        equivalence_result = equivalent
                        evidence = {
                            "schema": "mapping_equivalence/v1",
                            "graph": gname,
                            "algorithm": str(algo_key),
                            "live_path": str(equivalence_scratch),
                            "promoted_path": str(promoted_path),
                            "live_bytes":
                                equivalence_scratch.stat().st_size,
                            "promoted_bytes":
                                promoted_path.stat().st_size,
                            "equal": equivalent,
                            "checked_at":
                                datetime.now().isoformat(timespec="seconds"),
                        }
                        evidence_path = (
                            equivalence_dir
                            / f"{safe_algo}.equivalence.json"
                        )
                        evidence_path.write_text(
                            json.dumps(evidence, indent=2)
                        )
                        timing["mapping_equivalence_evidence"] = (
                            str(evidence_path)
                        )
                        if not equivalent:
                            raise RuntimeError(
                                "Live Gorder CSR mapping differs from "
                                f"promoted mapping for {gname}"
                            )
                        timing["mapping_equivalence_checked"] = True
                finally:
                    if (
                        equivalence_scratch is not None
                        and equivalence_result is not False
                    ):
                        equivalence_scratch.unlink(missing_ok=True)
                timing["timing_source"] = "live-final"
                timing["mapping_timing_machine"] = (
                    timing_machine_metadata()
                )
                if (
                    reused_timing is not None
                    and not timing.get("overhead_timeout")
                ):
                    timing["sidecar_reference"] = reused_timing
                    timing["mapping_generation_calibration_ratio"] = (
                        timing["mapping_generation_time"]
                        / reused_timing["mapping_generation_time"]
                    )
                    timing["complete_reorder_calibration_ratio"] = (
                        timing["reorder_time"]
                        / reused_timing["reorder_time"]
                    )
            if not dry_run and not timing.get("overhead_timeout"):
                weighted_cmd = [
                    str(converter), "-f", gpath, "-s", "-w", *map_flags,
                ]
                weighted_timing = _measure_reorder(
                    weighted_cmd,
                    repeats=weighted_apply_repeats,
                    dry_run=False,
                    timeout=timeout,
                    allow_timeout=True,
                )
                if weighted_timing.get("overhead_timeout"):
                    timing["weighted_apply_timeout"] = True
                    timing["weighted_timeout_seconds"] = (
                        weighted_timing.get("timeout_seconds")
                    )
                    timing["weighted_elapsed_seconds"] = (
                        weighted_timing.get("elapsed_seconds")
                    )
                    timing["weighted_failure_mode"] = (
                        weighted_timing.get("failure_mode")
                    )
                elif "reorder_apply_time" not in weighted_timing:
                    raise RuntimeError(
                        "Weighted CSR application timing missing for "
                        f"{gname}/{algo_key}"
                    )
                else:
                    for key in (
                        "reorder_apply_time",
                        "reorder_apply_times",
                        "reorder_apply_mean_time",
                        "reorder_apply_stddev_time",
                    ):
                        timing[f"weighted_{key}"] = weighted_timing[key]
                    timing["weighted_timing_machine"] = (
                        weighted_timing["timing_machine"]
                    )
            store.add({
                "graph": gname,
                "algorithm": aname,
                "algo_id": algo_id,
                "e2e_join_context_id": _e2e_join_context_id(),
                "mapping_identity_id":
                    _mapping_identity_id(mapping_identity),
                "mapping_identity": mapping_identity,
                "policy_id": policy_id,
                "cohort_id": cohort_id,
                "cell_key": cell_key,
                "measured_at": datetime.now().isoformat(timespec="seconds"),
                "reorder_trials": len(
                    timing.get("mapping_generation_times", [])
                ),
                "weighted_apply_trials": weighted_apply_repeats,
                "converter_identity": converter_identity,
                **timing,
            })
    checked_promoted = {
        (str(row.get("graph")), str(row.get("algo_id")))
        for row in store.results
        if row.get("mapping_equivalence_checked") is True
        and row.get("cohort_id") == promoted_cohort_ids.get(
            (str(row.get("graph")), str(row.get("algo_id")))
        )
        and row.get("mapping_identity")
        == promoted_mapping_identities.get(
            (str(row.get("graph")), str(row.get("algo_id")))
        )
    }
    censored_promoted = {
        (str(row.get("graph")), str(row.get("algo_id")))
        for row in store.results
        if row.get("overhead_timeout") is True
        and row.get("cohort_id") == promoted_cohort_ids.get(
            (str(row.get("graph")), str(row.get("algo_id")))
        )
        and row.get("mapping_identity")
        == promoted_mapping_identities.get(
            (str(row.get("graph")), str(row.get("algo_id")))
        )
    }
    if not dry_run:
        unresolved_promoted = (
            expected_promoted - checked_promoted - censored_promoted
        )
        if unresolved_promoted:
            raise RuntimeError(
                "Promoted mappings were not equivalence-checked: "
                f"{sorted(unresolved_promoted)}"
            )
    log.info(f"  exp3: {len(store.results)} total result rows in {store.path.name}")


# ============================================================================
# Experiment 4: End-to-End Performance
# ============================================================================


def exp4_end_to_end(
    graphs: list[dict], benchmarks: list[str], trials: int,
    timeout: int, dry_run: bool, graph_dir: str = ".",
) -> None:
    log.info("=" * 60)
    log.info("EXPERIMENT 4: End-to-End Performance")
    log.info("=" * 60)
    log.info("  (Combines reorder overhead + kernel execution)")
    log.info("  Materializing one-run and explicit-reuse end-to-end results.")

    speedup_path = RESULTS_DIR / "exp2_speedup" / "speedup_results.json"
    overhead_path = RESULTS_DIR / "exp3_overhead" / "overhead_results.json"
    if not speedup_path.is_file() or not overhead_path.is_file():
        raise RuntimeError(
            "Experiment 4 requires completed Experiment 2 and 3 results"
        )
    speedup_rows = json.loads(speedup_path.read_text())
    overhead_rows = json.loads(overhead_path.read_text())

    def grouped_cohorts(
        rows: list[dict],
        cell_fields: tuple[str, ...],
    ) -> dict[tuple[str, str], list[dict]]:
        grouped: dict[tuple[str, str], dict[tuple[str, ...], dict]] = {}
        for row in rows:
            join_id = row.get("e2e_join_context_id")
            cohort_id = row.get("cohort_id")
            if not isinstance(join_id, str) or not isinstance(cohort_id, str):
                continue
            cohort = grouped.setdefault((join_id, cohort_id), {})
            cell = tuple(str(row.get(field)) for field in cell_fields)
            current = cohort.get(cell)
            current_quality = (
                0 if current and current.get("overhead_timeout")
                else 1 if current and current.get("weighted_apply_timeout")
                else 2 if current else -1
            )
            row_quality = (
                0 if row.get("overhead_timeout")
                else 1 if row.get("weighted_apply_timeout")
                else 2
            )
            if (
                current is None
                or row_quality > current_quality
                or (
                    row_quality == current_quality
                    and str(row.get("measured_at", ""))
                    >= str(current.get("measured_at", ""))
                )
            ):
                cohort[cell] = row
        return {
            key: list(cells.values())
            for key, cells in grouped.items()
        }

    speedup_cohorts = grouped_cohorts(
        speedup_rows, ("graph", "algo_id", "benchmark"),
    )
    overhead_cohorts = grouped_cohorts(
        overhead_rows, ("graph", "algo_id"),
    )
    candidates: list[tuple[list[dict], list[dict]]] = []
    expected_speed_cells: Optional[set[tuple[str, str, str]]] = None
    if graphs and benchmarks:
        expected_specs = _paper_algorithm_specs(include_compose=True)
        expected_speed_cells = {
            (
                str(graph["name"]),
                str(_algo_identity(key)),
                str(benchmark),
            )
            for graph in graphs
            for key, _name, _flags in expected_specs
            if not algorithm_exclusion_reason(
                graph["name"],
                str(key),
            )
            for benchmark in benchmarks
        }

    for (speed_join, _speed_cohort), speed_values in speedup_cohorts.items():
        actual_speed_cells = {
            (
                str(row.get("graph")),
                str(row.get("algo_id")),
                str(row.get("benchmark")),
            )
            for row in speed_values
        }
        if (
            expected_speed_cells is not None
            and actual_speed_cells != expected_speed_cells
        ):
            continue
        baseline_cells = {
            (str(row.get("graph")), str(row.get("benchmark")))
            for row in speed_values
            if str(row.get("algo_id")) == "0"
        }
        required_baselines = {
            (str(row.get("graph")), str(row.get("benchmark")))
            for row in speed_values
        }
        if baseline_cells != required_baselines:
            continue
        for (
            overhead_join, _overhead_cohort
        ), overhead_values in overhead_cohorts.items():
            if overhead_join != speed_join:
                continue
            overhead_by_cell = {
                (str(row.get("graph")), str(row.get("algo_id"))): row
                for row in overhead_values
            }
            compatible = True
            for row in speed_values:
                if str(row.get("algo_id")) == "0":
                    continue
                cell = (str(row.get("graph")), str(row.get("algo_id")))
                overhead_row = overhead_by_cell.get(cell)
                if (
                    overhead_row is None
                    or row.get("mapping_identity")
                    != overhead_row.get("mapping_identity")
                ):
                    compatible = False
                    break
            if compatible:
                candidates.append((speed_values, overhead_values))

    if not candidates:
        raise RuntimeError(
            "Experiment 4 found no complete compatible Experiment 2/3 "
            "cohort pair with matching per-cell mapping identity"
        )
    speedup_rows, overhead_rows = max(
        candidates,
        key=lambda pair: (
            len(pair[0]),
            max(
                (
                    str(row.get("measured_at", ""))
                    for rows in pair for row in rows
                ),
                default="",
            ),
        ),
    )
    overhead = {
        (str(row["graph"]), str(row["algo_id"])): row
        for row in overhead_rows
    }
    calibration_by_algo: dict[str, list[float]] = {}
    mapping_calibration_by_algo: dict[str, list[float]] = {}
    all_calibration_ratios: list[float] = []
    for row in overhead_rows:
        algo_id = str(row["algo_id"])
        if (
            str(row.get("graph")) in REORDER_TIMING_REUSE_GRAPHS
            and algo_id not in REORDER_TIMING_ANCHOR_ALGOS
        ):
            continue
        ratio = row.get("complete_reorder_calibration_ratio")
        if isinstance(ratio, (int, float)) and ratio > 0:
            calibration_by_algo.setdefault(
                algo_id, [],
            ).append(float(ratio))
            all_calibration_ratios.append(float(ratio))
        mapping_ratio = row.get(
            "mapping_generation_calibration_ratio"
        )
        reference_mapping_time = (
            row.get("sidecar_reference", {})
            .get("mapping_generation_time")
        )
        if (
            isinstance(mapping_ratio, (int, float))
            and mapping_ratio > 0
            and isinstance(reference_mapping_time, (int, float))
            and reference_mapping_time >= 1.0
        ):
            mapping_calibration_by_algo.setdefault(
                algo_id, [],
            ).append(float(mapping_ratio))
    def quantile(values: list[float], probability: float) -> float:
        ordered = sorted(values)
        if not ordered:
            raise ValueError("quantile requires values")
        if len(ordered) == 1:
            return ordered[0]
        position = probability * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return (
            ordered[lower] * (1 - weight)
            + ordered[upper] * weight
        )

    global_calibration = (
        statistics.median(all_calibration_ratios)
        if all_calibration_ratios else 1.0
    )
    global_calibration_upper = (
        quantile(all_calibration_ratios, 0.90)
        if all_calibration_ratios else 1.0
    )

    def reorder_cost(row: dict, benchmark: str) -> dict[str, Any]:
        mapping = row.get("mapping_generation_time")
        validation = row.get("reorder_validation_time")
        apply_key = (
            "weighted_reorder_apply_time"
            if benchmark == "sssp" else
            "reorder_apply_time"
        )
        apply = row.get(apply_key)
        if not all(
            isinstance(value, (int, float))
            for value in (mapping, validation, apply)
        ) or not (
            mapping > 0
            and validation >= 0
            and apply >= 0
        ):
            raise RuntimeError(
                "Incomplete reorder cost components for "
                f"{row.get('graph')}/{row.get('algo_id')}/{benchmark}: "
                f"mapping={mapping}, validation={validation}, "
                f"{apply_key}={apply}"
            )
        apply_passes = row.get("reorder_apply_time_passes", [])
        intermediate_apply = (
            sum(
                float(value)
                for value in apply_passes[:-1]
                if isinstance(value, (int, float))
            )
            if isinstance(apply_passes, list)
            and len(apply_passes) > 1
            else 0.0
        )
        raw_cost = (
            float(mapping)
            + float(validation)
            + float(apply)
            + (intermediate_apply if benchmark == "sssp" else 0.0)
        )
        timing_source = row.get("timing_source", "live-final")
        factor = 1.0
        upper_factor = 1.0
        mapping_factor = 1.0
        mapping_upper_factor = 1.0
        calibration_scope = "live"
        calibration_sample_size = 0
        if timing_source == "stage02-sidecar":
            if (
                str(row.get("algo_id", "")).startswith("chain:")
                and not apply_passes
            ):
                raise RuntimeError(
                    "Missing intermediate apply-pass timing for "
                    f"{row.get('graph')}/{row.get('algo_id')}"
                )
            ratios = calibration_by_algo.get(str(row.get("algo_id")), [])
            if ratios:
                factor = statistics.median(ratios)
                upper_factor = quantile(ratios, 0.90)
                calibration_scope = "same-algorithm"
                calibration_sample_size = len(ratios)
            else:
                factor = global_calibration
                upper_factor = global_calibration_upper
                calibration_scope = "global-sensitivity"
            mapping_ratios = mapping_calibration_by_algo.get(
                str(row.get("algo_id")), [],
            )
            if mapping_ratios:
                mapping_factor = statistics.median(mapping_ratios)
                mapping_upper_factor = quantile(
                    mapping_ratios,
                    0.90,
                )
            else:
                mapping_factor = factor
                mapping_upper_factor = upper_factor
        calibrated_cost = (
            float(mapping) * mapping_factor
            + float(validation) * factor
            + intermediate_apply * factor
            + float(apply)
            if benchmark == "sssp"
            else raw_cost * factor
        )
        conservative_cost = (
            float(mapping) * mapping_upper_factor
            + float(validation) * upper_factor
            + intermediate_apply * upper_factor
            + float(apply)
            if benchmark == "sssp"
            else raw_cost * upper_factor
        )
        return {
            "calibrated": calibrated_cost,
            "uncalibrated": raw_cost,
            "conservative": conservative_cost,
            "calibration_factor": factor,
            "calibration_upper_factor": upper_factor,
            "mapping_calibration_factor": mapping_factor,
            "mapping_calibration_upper_factor":
                mapping_upper_factor,
            "calibration_scope": calibration_scope,
            "calibration_sample_size": calibration_sample_size,
            "timing_source": timing_source,
            "mapping_timing_machine":
                row.get("mapping_timing_machine"),
            "weighted_timing_machine":
                row.get("weighted_timing_machine"),
            "apply_profile": (
                "weighted-csr"
                if benchmark == "sssp"
                else "unweighted-csr"
            ),
        }
    baseline = {}
    baseline_trials: dict[tuple[str, str], list[float]] = {}
    for row in speedup_rows:
        if str(row.get("algo_id")) != "0":
            continue
        metric = (
            statistics.median(row["trial_times"])
            if row.get("trial_times")
            else row.get("median_time", row.get("average_time"))
        )
        if isinstance(metric, (int, float)) and metric > 0:
            baseline[(row["graph"], row["benchmark"])] = float(metric)
            baseline_trials[(row["graph"], row["benchmark"])] = [
                float(value)
                for value in row.get("trial_times", [])
                if isinstance(value, (int, float)) and value > 0
            ]

    derived = []
    censored_cells = []
    for row in speedup_rows:
        metric = (
            statistics.median(row["trial_times"])
            if row.get("trial_times")
            else row.get("median_time", row.get("average_time"))
        )
        if not isinstance(metric, (int, float)) or metric <= 0:
            continue
        graph_name = str(row["graph"])
        benchmark = str(row["benchmark"])
        algo_id = str(row["algo_id"])
        original_time = baseline.get((graph_name, benchmark))
        if original_time is None:
            raise RuntimeError(
                f"Missing Shuffled baseline for {graph_name}/{benchmark}"
            )
        overhead_row = overhead.get((graph_name, algo_id))
        if algo_id != "0" and overhead_row is None:
            raise RuntimeError(
                f"Missing reorder overhead for {graph_name}/{algo_id}"
            )
        if (
            algo_id != "0"
            and (
                overhead_row.get("overhead_timeout") is True
                or (
                    benchmark == "sssp"
                    and overhead_row.get("weighted_apply_timeout")
                    is True
                )
            )
        ):
            censored_cells.append({
                "graph": graph_name,
                "algorithm": row.get("algorithm"),
                "algo_id": row.get("algo_id"),
                "benchmark": benchmark,
                "timeout_seconds":
                    (
                        overhead_row.get("weighted_timeout_seconds")
                        if benchmark == "sssp"
                        and overhead_row.get("weighted_apply_timeout")
                        else overhead_row.get("timeout_seconds")
                    ),
                "elapsed_seconds":
                    (
                        overhead_row.get("weighted_elapsed_seconds")
                        if benchmark == "sssp"
                        and overhead_row.get("weighted_apply_timeout")
                        else overhead_row.get("elapsed_seconds")
                    ),
                "failure_mode":
                    (
                        overhead_row.get("weighted_failure_mode")
                        if benchmark == "sssp"
                        and overhead_row.get("weighted_apply_timeout")
                        else overhead_row.get("failure_mode")
                    ),
            })
            continue
        if algo_id == "0":
            reorder_time = 0.0
            reorder_time_uncalibrated = 0.0
            reorder_time_conservative = 0.0
            calibration_factor = 1.0
            calibration_upper_factor = 1.0
            mapping_calibration_factor = 1.0
            mapping_calibration_upper_factor = 1.0
            calibration_scope = "none"
            calibration_sample_size = 0
            mapping_timing_source = "none"
            mapping_timing_machine = None
            weighted_timing_machine = None
            reorder_apply_profile = "none"
        else:
            cost = reorder_cost(
                overhead_row, benchmark,
            )
            reorder_time = cost["calibrated"]
            reorder_time_uncalibrated = cost["uncalibrated"]
            reorder_time_conservative = cost["conservative"]
            calibration_factor = cost["calibration_factor"]
            calibration_upper_factor = cost[
                "calibration_upper_factor"
            ]
            mapping_calibration_factor = cost[
                "mapping_calibration_factor"
            ]
            mapping_calibration_upper_factor = cost[
                "mapping_calibration_upper_factor"
            ]
            calibration_scope = cost["calibration_scope"]
            calibration_sample_size = cost["calibration_sample_size"]
            mapping_timing_source = cost["timing_source"]
            mapping_timing_machine = cost["mapping_timing_machine"]
            weighted_timing_machine = cost["weighted_timing_machine"]
            reorder_apply_profile = cost["apply_profile"]
        kernel_time = float(metric)
        savings = original_time - kernel_time
        baseline_vector = baseline_trials.get(
            (graph_name, benchmark), [],
        )
        row_vector = [
            float(value)
            for value in row.get("trial_times", [])
            if isinstance(value, (int, float)) and value > 0
        ]
        differences = [
            left - right
            for left, right in zip(baseline_vector, row_vector)
        ]
        mean_savings = (
            statistics.fmean(differences)
            if differences else savings
        )
        savings_se = (
            statistics.stdev(differences)
            / math.sqrt(len(differences))
            if len(differences) > 1 else 0.0
        )
        if algo_id == "0":
            amortization_status = "baseline"
        elif abs(mean_savings) <= 2 * savings_se:
            amortization_status = "indeterminate"
        elif savings <= 0 or mean_savings <= 0:
            amortization_status = "never"
        else:
            amortization_status = "finite"

        def break_even_for(cost: float) -> Optional[int]:
            if amortization_status != "finite":
                return None
            return max(1, math.ceil(cost / savings))

        break_even_runs = (
            0 if algo_id == "0" else break_even_for(reorder_time)
        )
        break_even_uncalibrated = (
            0 if algo_id == "0"
            else break_even_for(reorder_time_uncalibrated)
        )
        break_even_conservative = (
            0 if algo_id == "0"
            else break_even_for(reorder_time_conservative)
        )
        reuse = {}
        reuse_uncalibrated = {}
        for count in E2E_REUSE_COUNTS:
            baseline_total = count * original_time
            reordered_total = reorder_time + count * kernel_time
            uncalibrated_total = (
                reorder_time_uncalibrated + count * kernel_time
            )
            sensitivity_total = (
                reorder_time_conservative + count * kernel_time
            )
            reuse[str(count)] = {
                "baseline_time": baseline_total,
                "raw": baseline_total / uncalibrated_total,
                "point": baseline_total / reordered_total,
                "conservative":
                    baseline_total / sensitivity_total,
                "range": sorted([
                    baseline_total / reordered_total,
                    baseline_total / sensitivity_total,
                ]),
                "end_to_end_time": reordered_total,
                "speedup": baseline_total / reordered_total,
            }
            reuse_uncalibrated[str(count)] = {
                "baseline_time": baseline_total,
                "end_to_end_time": uncalibrated_total,
                "speedup": baseline_total / uncalibrated_total,
            }
        derived.append({
            "graph": graph_name,
            "benchmark": benchmark,
            "algorithm": row.get("algorithm"),
            "algo_id": row.get("algo_id"),
            "kernel_time": kernel_time,
            "shuffled_kernel_time": original_time,
            "reorder_time": reorder_time,
            "reorder_time_uncalibrated": reorder_time_uncalibrated,
            "reorder_time_conservative":
                reorder_time_conservative,
            "reorder_calibration_factor": calibration_factor,
            "reorder_calibration_upper_factor":
                calibration_upper_factor,
            "mapping_generation_calibration_factor":
                mapping_calibration_factor,
            "mapping_generation_calibration_upper_factor":
                mapping_calibration_upper_factor,
            "reorder_calibration_scope": calibration_scope,
            "reorder_calibration_sample_size":
                calibration_sample_size,
            "mapping_timing_source": mapping_timing_source,
            "mapping_timing_machine": mapping_timing_machine,
            "weighted_timing_machine": weighted_timing_machine,
            "reorder_apply_profile": reorder_apply_profile,
            "one_run_end_to_end_time": reorder_time + kernel_time,
            "one_run_end_to_end_speedup":
                original_time / (reorder_time + kernel_time),
            "amortization_status": amortization_status,
            "mean_kernel_savings": mean_savings,
            "kernel_savings_standard_error": savings_se,
            "break_even_runs": break_even_runs,
            "break_even_runs_uncalibrated":
                break_even_uncalibrated,
            "break_even_runs_conservative":
                break_even_conservative,
            "break_even": {
                "status": amortization_status,
                "raw": break_even_uncalibrated,
                "point": break_even_runs,
                "conservative": break_even_conservative,
                "range": (
                    [
                        min(
                            break_even_runs,
                            break_even_conservative,
                        ),
                        max(
                            break_even_runs,
                            break_even_conservative,
                        ),
                    ]
                    if isinstance(break_even_runs, int)
                    and isinstance(
                        break_even_conservative, int
                    )
                    else None
                ),
            },
            "reuse_counts": reuse,
            "reuse_counts_uncalibrated": reuse_uncalibrated,
            "kernel_policy_id": row.get("policy_id"),
            "kernel_cohort_id": row.get("cohort_id"),
        })

    out_dir = ensure_dir(RESULTS_DIR / "exp4_e2e")
    save_json({
        "schema": "end_to_end_results/v3",
        "reuse_counts": E2E_REUSE_COUNTS,
        "primary_cohort": {
            "name": "controlled-work",
            "benchmarks": ["pr", "pr_spmv", "sssp", "bc"],
            "cells_per_algorithm": 44,
        },
        "sensitivity_cohorts": {
            "all": {
                "benchmarks": [
                    "bfs", "pr", "pr_spmv", "sssp",
                    "cc", "cc_sv", "bc",
                ],
                "cells_per_algorithm": 77,
            },
            "ordering-dependent-work": {
                "benchmarks": ["cc", "cc_sv"],
                "cells_per_algorithm": 22,
            },
            "ordering-dependent-traversal": {
                "benchmarks": ["bfs"],
                "cells_per_algorithm": 11,
            },
        },
        "amortization_screen": {
            "rule": "paired two-standard-error heuristic",
            "interpretation": "noise screen, not a 95% confidence interval",
        },
        "calibration_sensitivity": {
            "point": "per-algorithm median live/sidecar ratio when available, otherwise global median",
            "upper": "per-algorithm p90 live/sidecar ratio when available, otherwise global p90",
            "interpretation": "cross-machine calibration sensitivity, not a confidence interval",
        },
        "mixed_timing_state": any(
            row.get("mapping_timing_source") == "stage02-sidecar"
            for row in derived
        ),
        "global_reorder_calibration_factor": global_calibration,
        "global_reorder_calibration_upper_factor":
            global_calibration_upper,
        "censored_cells": censored_cells,
        "rows": derived,
    }, out_dir / "e2e_results.json")


# ============================================================================
# Experiment 5: Controlled Contrasts
# ============================================================================


def exp5_ablation(
    graphs: list[dict], benchmarks: list[str], trials: int,
    timeout: int, dry_run: bool, graph_dir: str = ".",
) -> None:
    log.info("=" * 60)
    log.info("EXPERIMENT 5: Controlled Contrasts")
    log.info("=" * 60)
    ensure_campaign(graphs, graph_dir, 5, benchmarks, trials)

    out_dir = ensure_dir(RESULTS_DIR / "exp5_ablation")
    store = ResultsStore(
        out_dir / "ablation_results.json",
        key_fields=["graph", "algo", "benchmark", "policy_id"],
    )

    # Focus on PR for ablation (most representative iterative algorithm)
    abl_bench = "pr"

    for graph in graphs:
        gname = graph["name"]
        log.info(f"  Graph: {gname}")
        gpath = resolve_graph_path(gname, graph_dir)

        configs = ABLATION_CONFIGS
        if _ALGORITHM_FILTER is not None:
            configs = [
                config for config in configs
                if config["algo"] in _ALGORITHM_FILTER
            ]
        for config in configs:
            algo_key = config["algo"]
            exclusion = algorithm_exclusion_reason(gname, algo_key)
            if exclusion:
                log.info(
                    f"    EXCLUDED {algo_key}: {exclusion}"
                )
                continue
            aflags = get_converter_flags(algo_key)
            flags, pregen_rt, mapping_identity = algo_flags_or_map(
                algo_key, aflags, gname, gpath,
            )
            cohort_id, policy_id = _kernel_policy_ids(
                graph_path=gpath,
                kind="ablation",
                trials=trials,
                executable=BIN_DIR / abl_bench,
                extra={"mapping": mapping_identity},
            )
            cell_key = f"{gname}|{algo_key}|{abl_bench}"
            key_row = {
                "graph": gname,
                "algo": algo_key,
                "benchmark": abl_bench,
                "policy_id": policy_id,
            }
            if store.has(key_row):
                continue
            cmd = build_benchmark_cmd(abl_bench, gpath, flags, trials)
            timing, _ = _run_kernel(
                cmd=cmd, graph_name=gname, algo_key=algo_key,
                benchmark=abl_bench, pregen_rt=pregen_rt,
                dry_run=dry_run, timeout=timeout,
                policy_id=policy_id, cohort_id=cohort_id,
            )
            store.add({
                "graph": gname, "config": config["name"],
                "algo": config["algo"], "desc": config["desc"],
                "benchmark": abl_bench,
                "policy_id": policy_id,
                "cohort_id": cohort_id,
                "cell_key": cell_key,
                "measured_at": datetime.now().isoformat(timespec="seconds"),
                **timing,
            })

    log.info(f"  exp5: {len(store.results)} total result rows in {store.path.name}")


# ============================================================================
# Experiment 6: Graph-Type Sensitivity
# ============================================================================


def exp6_sensitivity(
    graphs: list[dict], benchmarks: list[str], trials: int,
    timeout: int, dry_run: bool, graph_dir: str = ".",
) -> None:
    log.info("=" * 60)
    log.info("EXPERIMENT 6: Graph-Type Sensitivity")
    log.info("=" * 60)
    log.info("  Results derived from Exp 2 data, grouped by graph type.")

    out_dir = ensure_dir(RESULTS_DIR / "exp6_sensitivity")
    save_json(
        {"note": "Analysis performed by vldb_generate_figures.py from exp2 data",
         "groups": GRAPH_TYPE_GROUPS},
        out_dir / "sensitivity_note.json",
    )


# ============================================================================
# Experiment 7: Chained Ordering Analysis
# ============================================================================


def exp7_chained(
    graphs: list[dict], benchmarks: list[str], trials: int,
    timeout: int, dry_run: bool, graph_dir: str = ".",
) -> None:
    log.info("=" * 60)
    log.info("EXPERIMENT 7: Chained Ordering Analysis")
    log.info("=" * 60)
    ensure_campaign(graphs, graph_dir, 7, benchmarks, trials)

    out_dir = ensure_dir(RESULTS_DIR / "exp7_chained")
    store = ResultsStore(
        out_dir / "chained_results.json",
        key_fields=["graph", "chain", "benchmark", "policy_id"],
    )

    chain_bench = "pr"
    standalone_configs = [
        ("Leiden standalone", "12:leiden", ["-o", "12:leiden"]),
        ("HRAB standalone", "12:hrab", ["-o", "12:hrab"]),
        ("RabbitOrder CSR standalone", "8:csr", ["-o", "8:csr"]),
        ("DBG standalone", "5", ["-o", "5"]),
        ("HubCluster standalone", "4", ["-o", "4"]),
        ("GoGraph standalone", "16", ["-o", "16"]),
    ]
    gate_id = (
        _ACTIVE_VERIFICATION_GATE_ID
        or verification_gate_id(graphs, benchmarks, graph_dir)
    )
    expected_chain_cohort = measurement_cohort_id(
        "chained",
        trials=trials,
    )
    existing_chain_cohorts = {
        str(row.get("cohort_id"))
        for row in store.results
        if row.get("cohort_id") is not None
    }
    if (
        existing_chain_cohorts
        and existing_chain_cohorts != {expected_chain_cohort}
    ):
        raise RuntimeError(
            "Experiment 7 existing rows use a different cohort: "
            f"expected {expected_chain_cohort}, found "
            f"{sorted(existing_chain_cohorts)}"
        )

    for graph in graphs:
        gname = graph["name"]
        log.info(f"  Graph: {gname}")
        gpath = resolve_graph_path(gname, graph_dir)

        original_flags, original_rt, original_identity = algo_flags_or_map(
            "0", ["-o", "0"], gname, gpath,
        )
        original_cohort, original_policy = _kernel_policy_ids(
            graph_path=gpath,
            kind="chained",
            trials=trials,
            executable=BIN_DIR / chain_bench,
            extra={"mapping": original_identity},
        )
        original_key = {
            "graph": gname,
            "chain": "SHUFFLED",
            "benchmark": chain_bench,
            "policy_id": original_policy,
        }
        if not store.has(original_key):
            cmd = build_benchmark_cmd(
                chain_bench, gpath, original_flags, trials,
            )
            timing, _ = _run_kernel(
                cmd=cmd,
                graph_name=gname,
                algo_key="0",
                benchmark=chain_bench,
                pregen_rt=original_rt,
                dry_run=dry_run,
                timeout=timeout,
                policy_id=original_policy,
                cohort_id=original_cohort,
            )
            store.add({
                **original_key,
                "entry_type": "baseline",
                "flags": ["-o", "0"],
                "verification_gate_id": gate_id,
                "verification_gate_status": "pass",
                "verification_gate_method": "semantic-verifier",
                "cohort_id": original_cohort,
                "cell_key": f"{gname}|SHUFFLED|{chain_bench}",
                "measured_at": datetime.now().isoformat(timespec="seconds"),
                **timing,
            })

        chains = CHAINED_ORDERINGS
        if _ALGORITHM_FILTER is not None:
            chains = [
                (name, flags) for name, flags in chains
                if f"chain:{name}" in _ALGORITHM_FILTER
            ]
        for chain_name, chain_flags in chains:
            chain_key = f"chain:{chain_name}"
            if algorithm_exclusion_reason(gname, chain_key):
                continue
            flags, pregen_rt, mapping_identity = algo_flags_or_map(
                chain_key, chain_flags, gname, gpath,
            )
            cohort_id, policy_id = _kernel_policy_ids(
                graph_path=gpath,
                kind="chained",
                trials=trials,
                executable=BIN_DIR / chain_bench,
                extra={"mapping": mapping_identity},
            )
            cell_key = f"{gname}|{chain_name}|{chain_bench}"
            key_row = {
                "graph": gname,
                "chain": chain_name,
                "benchmark": chain_bench,
                "policy_id": policy_id,
            }
            if store.has(key_row):
                continue
            cmd = build_benchmark_cmd(chain_bench, gpath, flags, trials)
            timing, _ = _run_kernel(
                cmd=cmd, graph_name=gname, algo_key=chain_key,
                benchmark=chain_bench, pregen_rt=pregen_rt,
                dry_run=dry_run, timeout=timeout,
                policy_id=policy_id, cohort_id=cohort_id,
            )
            store.add({
                "graph": gname, "chain": chain_name,
                "entry_type": "chain",
                "flags": chain_flags, "benchmark": chain_bench,
                "verification_gate_id": gate_id,
                "verification_gate_status": "pass",
                "verification_gate_method": "semantic-verifier",
                "policy_id": policy_id,
                "cohort_id": cohort_id,
                "cell_key": cell_key,
                "measured_at": datetime.now().isoformat(timespec="seconds"),
                **timing,
            })

        for display_name, algo_key, algo_flags in standalone_configs:
            if (
                _ALGORITHM_FILTER is not None
                and algo_key not in _ALGORITHM_FILTER
            ):
                continue
            flags, pregen_rt, mapping_identity = algo_flags_or_map(
                algo_key, algo_flags, gname, gpath,
            )
            cohort_id, policy_id = _kernel_policy_ids(
                graph_path=gpath,
                kind="chained",
                trials=trials,
                executable=BIN_DIR / chain_bench,
                extra={"mapping": mapping_identity},
            )
            key_row = {
                "graph": gname,
                "chain": display_name,
                "benchmark": chain_bench,
                "policy_id": policy_id,
            }
            if store.has(key_row):
                continue
            cmd = build_benchmark_cmd(
                chain_bench, gpath, flags, trials,
            )
            timing, _ = _run_kernel(
                cmd=cmd,
                graph_name=gname,
                algo_key=algo_key,
                benchmark=chain_bench,
                pregen_rt=pregen_rt,
                dry_run=dry_run,
                timeout=timeout,
                policy_id=policy_id,
                cohort_id=cohort_id,
            )
            store.add({
                **key_row,
                "entry_type": "standalone",
                "algo_key": algo_key,
                "flags": algo_flags,
                "cohort_id": cohort_id,
                "cell_key":
                    f"{gname}|standalone:{algo_key}|{chain_bench}",
                "verification_gate_id": gate_id,
                "verification_gate_status": "pass",
                "verification_gate_method": "semantic-verifier",
                "measured_at":
                    datetime.now().isoformat(timespec="seconds"),
                **timing,
            })

        convergence_entries = [
            ("SHUFFLED", "0", ["-o", "0"]),
            *[
                (name, f"chain:{name}", flags)
                for name, flags in chains
            ],
        ]
        for display_name, algo_key, algo_flags in convergence_entries:
            flags, pregen_rt, mapping_identity = algo_flags_or_map(
                algo_key, algo_flags, gname, gpath,
            )
            cohort_id, policy_id = _kernel_policy_ids(
                graph_path=gpath,
                kind="chained",
                trials=trials,
                executable=BIN_DIR / "pr",
                benchmark_name="pr_convergence",
                extra={"mapping": mapping_identity},
            )
            key_row = {
                "graph": gname,
                "chain": display_name,
                "benchmark": "pr_convergence",
                "policy_id": policy_id,
            }
            if store.has(key_row):
                continue
            cmd = build_benchmark_cmd(
                "pr_convergence", gpath, flags, trials,
            )
            timing, _ = _run_kernel(
                cmd=cmd,
                graph_name=gname,
                algo_key=algo_key,
                benchmark="pr_convergence",
                pregen_rt=pregen_rt,
                dry_run=dry_run,
                timeout=timeout,
                policy_id=policy_id,
                cohort_id=cohort_id,
            )
            store.add({
                **key_row,
                "entry_type": (
                    "baseline" if algo_key == "0" else "chain"
                ),
                "flags": algo_flags,
                "cohort_id": cohort_id,
                "cell_key":
                    f"{gname}|{display_name}|pr_convergence",
                "verification_gate_id": gate_id,
                "verification_gate_status": "pass",
                "verification_gate_method": "semantic-verifier",
                "measured_at":
                    datetime.now().isoformat(timespec="seconds"),
                **timing,
            })

    log.info(f"  exp7: {len(store.results)} total result rows in {store.path.name}")


# ============================================================================
# Experiment 8: Reorder Scalability
# ============================================================================


def exp8_scalability(
    graphs: list[dict], benchmarks: list[str], trials: int,
    timeout: int, dry_run: bool, graph_dir: str = ".",
) -> None:
    log.info("=" * 60)
    log.info("EXPERIMENT 8: Reorder Scalability")
    log.info("=" * 60)
    ensure_campaign(graphs, graph_dir, 8, benchmarks, 1)

    out_dir = ensure_dir(RESULTS_DIR / "exp8_scalability")
    store = ResultsStore(
        out_dir / "scalability_results.json",
        key_fields=[
            "graph", "algorithm", "threads", "repeat", "policy_id",
        ],
    )

    test_algos = [
        (name, key, flags)
        for key in SCALABILITY_ALGORITHM_KEYS
        for _key, name, flags in [_algorithm_spec_for_key(key)]
    ]
    if _ALGORITHM_FILTER is not None:
        test_algos = [
            entry for entry in test_algos if entry[1] in _ALGORITHM_FILTER
        ]

    converter = BIN_DIR / "converter"
    available_cpus = len(_expand_cpu_list(_RUNTIME_CPU_LIST))
    max_threads = available_cpus or (_RUNTIME_THREADS or max(THREAD_COUNTS))
    thread_counts = [value for value in THREAD_COUNTS if value <= max_threads]
    if thread_counts != THREAD_COUNTS:
        log.info(
            f"  Scalability sweep capped at {max_threads} threads by runtime policy"
        )
    for graph in graphs:
        gname = graph["name"]
        log.info(f"  Graph: {gname}")
        gpath = resolve_graph_path(gname, graph_dir, ext=".sg")
        if not Path(gpath).exists():
            gpath = resolve_graph_path(gname, graph_dir, ext=".el")

        for aname, _algo_key, aflags in test_algos:
            if algorithm_exclusion_reason(gname, _algo_key):
                continue
            for nthreads in thread_counts:
                for repeat in range(SCALABILITY_REPEATS):
                    env = {
                        "OMP_NUM_THREADS": str(nthreads),
                        "GRAPHBREW_MAPPING_QUALITY": "1",
                    }
                    cpu_list = _cpu_list_for_threads(nthreads)
                    cohort_id, policy_id = _kernel_policy_ids(
                        graph_path=gpath,
                        kind="scalability",
                        trials=SCALABILITY_REPEATS,
                        executable=converter,
                        env=env,
                        cpu_list=cpu_list,
                        extra={"threads": nthreads},
                    )
                    cell_key = (
                        f"{gname}|{aname}|{nthreads}|{repeat}"
                    )
                    key_row = {
                        "graph": gname,
                        "algorithm": aname,
                        "threads": nthreads,
                        "repeat": repeat,
                        "policy_id": policy_id,
                    }
                    if store.has(key_row):
                        continue
                    cmd = [str(converter), "-f", gpath] + aflags
                    output = run_cmd(
                        cmd,
                        dry_run=dry_run,
                        timeout=timeout,
                        env=env,
                        cpu_list=cpu_list,
                    )
                    if output is None:
                        raise RuntimeError(
                            "Scalability command failed for "
                            f"{gname}/{aname}/{nthreads}/repeat{repeat}"
                        )
                    timing = parse_timing(output)
                    timing["timing_machine"] = timing_machine_metadata()
                    store.add({
                        "graph": gname,
                        "algorithm": aname,
                        "algo_key": _algo_key,
                        "threads": nthreads,
                        "repeat": repeat,
                        "cpu_list": cpu_list,
                        "policy_id": policy_id,
                        "cohort_id": cohort_id,
                        "cell_key": cell_key,
                        "measured_at":
                            datetime.now().isoformat(timespec="seconds"),
                        "stdout_tail":
                            output.strip().splitlines()[-40:],
                        **timing,
                    })

    log.info(f"  exp8: {len(store.results)} total result rows in {store.path.name}")


# ============================================================================
# Main
# ============================================================================

EXPERIMENTS = {
    1: ("Cache Performance Analysis", exp1_cache_performance),
    2: ("Kernel Speedup", exp2_kernel_speedup),
    3: ("Reorder Overhead & Amortization", exp3_reorder_overhead),
    4: ("End-to-End Performance", exp4_end_to_end),
    5: ("Controlled Contrasts", exp5_ablation),
    6: ("Graph-Type Sensitivity", exp6_sensitivity),
    7: ("Chained Ordering Analysis", exp7_chained),
    8: ("Reorder Scalability", exp8_scalability),
}


# ============================================================================
# Auto-Setup: dependencies, binaries, graph download & conversion
# ============================================================================

def _setup_environment(
    graph_dir: str,
    graphs: list[dict],
    dry_run: bool = False,
    skip_download: bool = False,
    timeout: int = 7200,
) -> str:
    """Ensure binaries are built, graphs are downloaded, and .sg files exist.

    Returns the *resolved* graph directory (may differ from input when
    graphs live under ``results/graphs``).
    """
    log.info("=" * 60)
    log.info("  AUTO-SETUP")
    log.info("=" * 60)

    graphs_path = Path(graph_dir) if graph_dir != "." else PROJECT_ROOT / "results" / "graphs"
    graph_dir_resolved = str(graphs_path)

    if dry_run:
        log.info("  [dry-run] Skipping auto-setup")
        return graph_dir_resolved
    graphs_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Python dependencies ──────────────────────────────────────────
    log.info("\n── Step 1/5: Python dependencies ──")
    try:
        import matplotlib  # noqa: F401
        log.info("  matplotlib: OK")
    except ImportError:
        log.warning("  matplotlib not installed — figures will be skipped")
        log.info("  Install with: pip install matplotlib numpy")

    # ── 2. Build binaries ───────────────────────────────────────────────
    log.info("\n── Step 2/5: Build binaries ──")
    _setup_build_binaries()

    # ── 3. Download graphs ──────────────────────────────────────────────
    log.info("\n── Step 3/5: Download graphs ──")
    if skip_download:
        log.info("  Skipping download (--skip-download)")
    else:
        _setup_download_graphs(graphs, graphs_path)

    # ── 4. Convert .mtx → .sg ──────────────────────────────────────────
    log.info("\n── Step 4/5: Convert graphs to .sg ──")
    _setup_convert_graphs(graphs, graphs_path, timeout=timeout)

    # ── 5. Pre-generate .lo mapping files ──────────────────────────────
    log.info("\n── Step 5/5: Pre-generate reorder mappings (.lo) ──")
    _pregenerate_mappings(graphs, graph_dir_resolved, dry_run=dry_run)

    log.info("\n" + "=" * 60)
    log.info("  AUTO-SETUP COMPLETE")
    log.info("=" * 60 + "\n")
    return graph_dir_resolved


def _refresh_graph_corpus(
    graph_dir: str,
    graphs: list[dict],
    *,
    dry_run: bool = False,
    skip_download: bool = False,
    timeout: int = 7200,
) -> str:
    """Transactionally refresh graph_source/v2 corpus artifacts only."""
    graphs_path = (
        Path(graph_dir)
        if graph_dir != "."
        else PROJECT_ROOT / "results" / "graphs"
    )
    if dry_run:
        log.info(
            f"[dry-run] Would refresh {len(graphs)} graphs in {graphs_path}")
        return str(graphs_path)

    graphs_path.mkdir(parents=True, exist_ok=True)
    _setup_build_binaries(
        benchmarks=[],
        include_standard=False,
        include_sim=False,
        include_converter=True,
    )
    if not skip_download:
        _setup_download_graphs(graphs, graphs_path)
    _setup_convert_graphs(graphs, graphs_path, timeout=timeout)
    return str(graphs_path)


def select_requested_graphs(names: list[str]) -> list[dict]:
    """Resolve exact graph names once, preserving the requested order."""
    catalog = {}
    for graph in (
        EVAL_GRAPHS
        + PREVIEW_GRAPHS
        + ADAPTIVE_CPU_EXPANSION_GRAPHS
    ):
        catalog.setdefault(str(graph["name"]), graph)
    selected = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        selected.append(dict(catalog.get(name, {
            "name": name,
            "short": name,
            "type": "unknown",
            "vertices_m": 0,
            "edges_m": 0,
        })))
    return selected


def _setup_build_binaries(
    benchmarks: Optional[list[str]] = None,
    *,
    include_standard: bool = True,
    include_sim: bool = True,
    include_converter: bool = True,
    include_work: bool = False,
    sim_benchmarks: Optional[list[str]] = None,
) -> None:
    """Ask make to rebuild every required canonical binary when stale."""
    makefile = PROJECT_ROOT / "Makefile"
    if not makefile.exists():
        raise RuntimeError("Makefile not found — cannot build required binaries")

    selected = BENCHMARKS if benchmarks is None else benchmarks
    targets: list[str] = []
    if include_standard:
        targets.extend(
            str((BIN_DIR / bench).relative_to(PROJECT_ROOT))
            for bench in selected
        )
    if include_converter:
        targets.append(str((BIN_DIR / "converter").relative_to(PROJECT_ROOT)))
    if include_sim:
        selected_sim = (
            selected if sim_benchmarks is None else sim_benchmarks
        )
        targets.extend(
            str((BIN_SIM_DIR / bench).relative_to(PROJECT_ROOT))
            for bench in selected_sim
        )
    if include_work:
        targets.extend(
            str((BIN_WORK_DIR / bench).relative_to(PROJECT_ROOT))
            for bench in selected
            if bench in {"bfs", "bc", "cc", "cc_sv", "sssp"}
        )
    jobs = min(4, _RUNTIME_THREADS or 4)
    log.info(f"  Checking/rebuilding {len(targets)} canonical binaries...")
    lock_dir = PROJECT_ROOT / "bench" / "obj"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / ".build.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        result = subprocess.run(
            ["make", "-j", str(jobs), *targets],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True,
        )
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    if result.returncode != 0:
        raise RuntimeError(f"Required binary build failed: {result.stderr[-1000:]}")
    log.info("  Required binaries are fresh ✓")


def _setup_download_graphs(graphs: list[dict], dest_dir: Path) -> None:
    """Download evaluation graphs from SuiteSparse; report manual-download graphs."""
    # Collect graphs that need downloading from catalog
    catalog_names = []
    manual_graphs = []

    for g in graphs:
        name = g["name"]
        sg_path = dest_dir / name / f"{name}.sg"
        el_path = dest_dir / name / f"{name}.el"
        source_sg = dest_dir / name / f"{name}.source.sg"
        flat_sg = dest_dir / f"{name}.sg"
        flat_el = dest_dir / f"{name}.el"
        # Already have .sg or .el — skip
        if (
            sg_path.exists() or el_path.exists() or source_sg.exists()
            or flat_sg.exists() or flat_el.exists()
        ):
            log.info(f"  {name}: already present ✓")
            continue

        src = VLDB_GRAPH_SOURCES.get(name, {})
        if src.get("source") == "catalog":
            catalog_names.append(name)
        elif src.get("source") == "manual":
            manual_graphs.append((name, src))
        else:
            # Graph not in VLDB sources — check if .mtx exists
            mtx_dir = dest_dir / name
            if mtx_dir.exists():
                log.info(f"  {name}: directory exists (will convert)")
            else:
                log.warning(f"  {name}: not in download catalog — place .sg/.el manually")

    # Download from catalog
    if catalog_names:
        log.info(f"  Downloading {len(catalog_names)} graphs from SuiteSparse...")
        try:
            from scripts.lib.pipeline.download import (
                download_graphs_parallel,
                get_graph_info,
                DownloadableGraph,
            )
            # Build DownloadableGraph list for graphs in the catalog
            to_download = []
            for name in catalog_names:
                info = get_graph_info(name)
                if info:
                    to_download.append(info)
                else:
                    log.warning(f"  {name}: not found in download catalog")
            if to_download:
                paths, failed = download_graphs_parallel(
                    graphs=to_download,
                    dest_dir=dest_dir,
                    max_workers=min(4, len(to_download)),
                    show_progress=True,
                )
                log.info(f"  Downloaded {len(paths)} graphs, {len(failed)} failed")
                for name in failed:
                    log.warning(f"    FAILED: {name}")
        except Exception as e:
            log.error(f"  Download failed: {e}")

    # Report manual-download graphs
    if manual_graphs:
        log.info("")
        log.info("  ┌─ MANUAL DOWNLOAD REQUIRED ─────────────────────────────")
        for name, src in manual_graphs:
            log.info(f"  │ {name}:")
            for line in src.get("instructions", "").split("\n"):
                log.info(f"  │   {line}")
        log.info("  └─────────────────────────────────────────────────────────")


_AUXILIARY_GRAPH_TOKENS = {
    "nodename", "completionpercentage", "public", "userid",
    "categories", "category", "coord",
}


def _normalize_graph_name(value: str) -> str:
    normalized = value.lower()
    if normalized.endswith(".source"):
        normalized = normalized[:-len(".source")]
    return re.sub(r"[^a-z0-9]", "", normalized)


def _select_graph_input(graph_subdir: Path, graph_name: str) -> Optional[Path]:
    if not graph_subdir.exists():
        return None
    matches = sorted({
        *graph_subdir.glob("**/*.mtx"),
        *graph_subdir.glob("**/*.el"),
        *graph_subdir.glob("**/*.source.sg"),
    })
    exact = [
        match for match in matches
        if _normalize_graph_name(match.stem) == _normalize_graph_name(graph_name)
    ]
    if exact:
        return max(exact, key=lambda path: path.stat().st_size)
    graph_candidates = [
        match for match in matches
        if not any(
            token in _normalize_graph_name(match.stem)
            for token in _AUXILIARY_GRAPH_TOKENS
        )
    ]
    if len(graph_candidates) == 1:
        return graph_candidates[0]
    if len(graph_candidates) > 1:
        raise RuntimeError(
            f"Ambiguous graph inputs for {graph_name}: "
            + ", ".join(str(path) for path in graph_candidates)
        )
    return None


def _recorded_graph_input(
    provenance_path: Path,
    graph_name: str,
) -> Optional[Path]:
    """Reuse an exact external source recorded by prior valid provenance."""
    if not provenance_path.is_file():
        return None
    try:
        provenance = json.loads(provenance_path.read_text())
        source_path = Path(provenance["source_path"])
    except (KeyError, OSError, TypeError, ValueError):
        return None
    if (
        provenance.get("schema") not in {
            "graph_source/v1", "graph_source/v2"}
        or provenance.get("graph") != graph_name
        or not source_path.is_file()
    ):
        return None
    if provenance.get("source_bytes") != source_path.stat().st_size:
        raise RuntimeError(
            f"{graph_name}: recorded graph source changed: {source_path} "
            f"from {provenance_path}"
        )
    if (
        provenance.get("schema") == "graph_source/v2"
        and provenance.get("source_crc32")
        != _file_crc32(source_path)
    ):
        raise RuntimeError(
            f"{graph_name}: recorded graph source content changed: "
            f"{source_path}")
    return source_path


def _graph_conversion_transaction_path(graph_path: Path) -> Path:
    return graph_path.with_suffix(".sg.transaction.json")


def _cleanup_graph_conversion_candidates(
    transaction_path: Path,
    graph_path: Path,
) -> None:
    try:
        transaction = json.loads(transaction_path.read_text())
    except (OSError, ValueError):
        transaction = {}
    expected_prefix = f".{graph_path.stem}.candidate-"
    for field in ("candidate_graph", "candidate_provenance"):
        raw_path = transaction.get(field)
        if not isinstance(raw_path, str):
            continue
        candidate = Path(raw_path)
        if (
            candidate.parent.resolve() == graph_path.parent.resolve()
            and candidate.name.startswith(expected_prefix)
        ):
            candidate.unlink(missing_ok=True)


def _write_graph_conversion_transaction(
    *,
    transaction_path: Path,
    graph_name: str,
    phase: str,
    candidate_path: Path,
    candidate_provenance: Path,
    backup_graph: Path,
    backup_provenance: Path,
) -> None:
    marker_tmp = transaction_path.with_suffix(
        transaction_path.suffix + ".tmp"
    )
    marker_tmp.unlink(missing_ok=True)
    marker_tmp.write_text(json.dumps({
        "schema": "graph_conversion_transaction/v1",
        "graph": graph_name,
        "phase": phase,
        "candidate_graph": str(candidate_path.resolve()),
        "candidate_provenance": str(candidate_provenance.resolve()),
        "backup_graph": str(backup_graph.resolve()),
        "backup_provenance": str(backup_provenance.resolve()),
    }, indent=2))
    marker_tmp.replace(transaction_path)


def _recover_graph_conversion_transaction(
    *,
    graph_name: str,
    graph_path: Path,
    provenance_path: Path,
    backup_graph: Path,
    backup_provenance: Path,
    transaction_path: Path,
) -> None:
    """Recover the canonical pair after an interrupted atomic replacement."""
    marker_tmp = transaction_path.with_suffix(
        transaction_path.suffix + ".tmp"
    )
    if not (
        transaction_path.exists()
        or marker_tmp.exists()
        or backup_graph.exists()
        or backup_provenance.exists()
    ):
        return
    if not transaction_path.exists() and marker_tmp.exists():
        marker_tmp.replace(transaction_path)

    current_valid = (
        graph_path.exists()
        and provenance_path.exists()
        and _graph_provenance_valid(
            graph_path, graph_name=graph_name,
        )
    )
    if current_valid:
        backup_graph.unlink(missing_ok=True)
        backup_provenance.unlink(missing_ok=True)
    elif backup_graph.exists() or backup_provenance.exists():
        if backup_graph.exists():
            graph_path.unlink(missing_ok=True)
            backup_graph.replace(graph_path)
        if backup_provenance.exists():
            provenance_path.unlink(missing_ok=True)
            backup_provenance.replace(provenance_path)
        log.warning(
            f"  {graph_name}: recovered interrupted graph conversion"
        )

    if transaction_path.exists():
        _cleanup_graph_conversion_candidates(
            transaction_path, graph_path,
        )
    transaction_path.unlink(missing_ok=True)
    marker_tmp.unlink(missing_ok=True)


def _setup_convert_graphs(
    graphs: list[dict],
    graphs_dir: Path,
    timeout: int = 7200,
) -> None:
    """Convert inputs to deterministic random-label .sg files with provenance."""
    converter = BIN_DIR / "converter"
    if not converter.exists():
        log.warning("  Converter binary not found — skipping conversion")
        return

    converted = 0
    skipped = 0
    failed = 0
    converter_sha256 = file_sha256(converter)
    repository_state = _conversion_repository_state()

    for g in graphs:
        name = g["name"]
        frozen_nodes = int(g.get("nodes", 0) or 0)
        frozen_undirected_edges = int(
            g.get("undirected_edges", 0) or 0)
        graph_subdir = graphs_dir / name
        nested_sg = graph_subdir / f"{name}.sg"
        flat_sg = graphs_dir / f"{name}.sg"
        sg_path = nested_sg if nested_sg.exists() or not flat_sg.exists() else flat_sg
        provenance_path = _graph_provenance_path(sg_path)
        backup_graph = sg_path.with_suffix(".sg.previous")
        backup_provenance = provenance_path.with_suffix(
            ".json.previous"
        )
        transaction_path = _graph_conversion_transaction_path(sg_path)

        sg_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = sg_path.with_suffix(".sg.lock")
        with lock_path.open("a+") as graph_lock:
            fcntl.flock(graph_lock.fileno(), fcntl.LOCK_EX)
            _recover_graph_conversion_transaction(
                graph_name=name,
                graph_path=sg_path,
                provenance_path=provenance_path,
                backup_graph=backup_graph,
                backup_provenance=backup_provenance,
                transaction_path=transaction_path,
            )

            # Prefer local canonical inputs, then reuse an exact external
            # source recorded by recovered provenance.
            input_file = _select_graph_input(graph_subdir, name)
            if input_file is None:
                for candidate in (
                    graphs_dir / f"{name}.mtx",
                    graphs_dir / f"{name}.el",
                ):
                    if candidate.exists():
                        input_file = candidate
                        break
            if input_file is None:
                input_file = _recorded_graph_input(
                    provenance_path,
                    name,
                )

            if sg_path.exists() and sg_path.stat().st_size > 0:
                provenance: dict = {}
                if provenance_path.exists():
                    try:
                        provenance = json.loads(provenance_path.read_text())
                    except (OSError, ValueError):
                        provenance = {}
                verified = _graph_provenance_valid(
                    sg_path,
                    graph_name=name,
                    expected_nodes=(
                        frozen_nodes if frozen_nodes > 0 else None),
                    expected_undirected_edges=(
                        frozen_undirected_edges
                        if frozen_undirected_edges > 0 else None),
                )
                if verified and input_file:
                    verified = (
                        provenance.get("source_path")
                        == str(input_file.resolve())
                        and provenance.get("source_bytes")
                        == input_file.stat().st_size
                    )
                if verified:
                    skipped += 1
                    fcntl.flock(graph_lock.fileno(), fcntl.LOCK_UN)
                    continue
                if not input_file:
                    raise RuntimeError(
                        f"{name}: existing .sg has no verified random-label "
                        "provenance and no source .mtx/.el is available"
                    )
                log.warning(
                    f"  {name}: rebuilding unverified .sg with "
                    "deterministic RANDOM labels"
                )

            if not input_file:
                failed += 1
                log.warning(f"  {name}: no graph source file found")
                fcntl.flock(graph_lock.fileno(), fcntl.LOCK_UN)
                continue

            with tempfile.NamedTemporaryFile(
                prefix=f".{name}.candidate-",
                suffix=".sg",
                dir=sg_path.parent,
                delete=False,
            ) as candidate_handle:
                candidate_path = Path(candidate_handle.name)
            candidate_path.unlink()
            candidate_provenance = _graph_provenance_path(candidate_path)
            candidate_provenance.unlink(missing_ok=True)
            _write_graph_conversion_transaction(
                transaction_path=transaction_path,
                graph_name=name,
                phase="building",
                candidate_path=candidate_path,
                candidate_provenance=candidate_provenance,
                backup_graph=backup_graph,
                backup_provenance=backup_provenance,
            )

            log.info(f"  Converting {name}...")
            logical_cmd = [
                str(converter), "-f", str(input_file), "-s", "-o", "1",
                "-b", str(sg_path),
            ]
            execution_cmd = [
                *logical_cmd[:-1], str(candidate_path),
            ]
            output = run_cmd(
                execution_cmd, dry_run=False, timeout=timeout,
            )
            if (
                output is None
                or not candidate_path.exists()
                or candidate_path.stat().st_size == 0
            ):
                log.warning(f"    Conversion failed for {name}")
                candidate_path.unlink(missing_ok=True)
                candidate_provenance.unlink(missing_ok=True)
                transaction_path.unlink(missing_ok=True)
                failed += 1
                fcntl.flock(graph_lock.fileno(), fcntl.LOCK_UN)
                continue

            source_stat = input_file.stat()
            graph_info = _serialized_graph_info(candidate_path)
            if graph_info["directed"]:
                candidate_path.unlink(missing_ok=True)
                transaction_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"{name}: converter did not produce a symmetrized graph"
                )
            if (
                frozen_nodes > 0
                and graph_info["nodes"] != frozen_nodes
            ):
                raise RuntimeError(
                    f"{name}: converter node count changed: "
                    f"{graph_info['nodes']} != {frozen_nodes}"
                )
            if (
                frozen_undirected_edges > 0
                and graph_info["edges"] // 2
                != frozen_undirected_edges
            ):
                raise RuntimeError(
                    f"{name}: converter edge count changed: "
                    f"{graph_info['edges'] // 2} != "
                    f"{frozen_undirected_edges}"
                )
            expected_nodes = (
                frozen_nodes
                if frozen_nodes > 0 else int(graph_info["nodes"])
            )
            expected_undirected_edges = (
                frozen_undirected_edges
                if frozen_undirected_edges > 0
                else int(graph_info["edges"]) // 2
            )
            provenance = {
                "schema": "graph_source/v2",
                "reorder_semantics_version":
                    REORDER_SEMANTICS_VERSION,
                "graph": name,
                "source_path": str(input_file.resolve()),
                "source_bytes": source_stat.st_size,
                "source_mtime_ns": source_stat.st_mtime_ns,
                "source_crc32": _file_crc32(input_file),
                "output_path": str(sg_path.resolve()),
                "output_bytes": candidate_path.stat().st_size,
                "output_crc32": _file_crc32(candidate_path),
                "converter_sha256": converter_sha256,
                "conversion_repository_state": repository_state,
                "converter_args": logical_cmd,
                "directed": graph_info["directed"],
                "symmetrized": not graph_info["directed"],
                "nodes": graph_info["nodes"],
                "directed_edges": graph_info["edges"],
                "undirected_edges": graph_info["edges"] // 2,
                "expected_nodes": expected_nodes,
                "expected_undirected_edges":
                    expected_undirected_edges,
                "random_order_algorithm": "1",
                "random_seed": RANDOM_BASELINE_SEED,
                "omp_num_threads":
                    _effective_env().get("OMP_NUM_THREADS"),
                "cpu_list": _RUNTIME_CPU_LIST,
                "created_at":
                    datetime.now().isoformat(timespec="seconds"),
            }
            provenance["conversion_policy_id"] = (
                _graph_conversion_policy_id(provenance)
            )
            candidate_provenance.write_text(
                json.dumps(provenance, indent=2)
            )
            if not _graph_provenance_valid(
                candidate_path,
                graph_name=name,
                canonical_output_path=sg_path,
                expected_nodes=expected_nodes,
                expected_undirected_edges=
                    expected_undirected_edges,
            ):
                candidate_path.unlink(missing_ok=True)
                candidate_provenance.unlink(missing_ok=True)
                transaction_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"{name}: candidate graph does not match the current "
                    "conversion policy and canonical graph dimensions"
                )

            try:
                _write_graph_conversion_transaction(
                    transaction_path=transaction_path,
                    graph_name=name,
                    phase="installing",
                    candidate_path=candidate_path,
                    candidate_provenance=candidate_provenance,
                    backup_graph=backup_graph,
                    backup_provenance=backup_provenance,
                )
                if sg_path.exists():
                    sg_path.replace(backup_graph)
                if provenance_path.exists():
                    provenance_path.replace(backup_provenance)
                candidate_path.replace(sg_path)
                candidate_provenance.replace(provenance_path)
                if not _graph_provenance_valid(
                    sg_path,
                    graph_name=name,
                    expected_nodes=expected_nodes,
                    expected_undirected_edges=
                        expected_undirected_edges,
                ):
                    raise RuntimeError(
                        f"{name}: installed graph failed provenance validation"
                    )
            except Exception:
                sg_path.unlink(missing_ok=True)
                provenance_path.unlink(missing_ok=True)
                if backup_graph.exists():
                    backup_graph.replace(sg_path)
                if backup_provenance.exists():
                    backup_provenance.replace(provenance_path)
                transaction_path.unlink(missing_ok=True)
                raise
            else:
                backup_graph.unlink(missing_ok=True)
                backup_provenance.unlink(missing_ok=True)
                transaction_path.unlink(missing_ok=True)
                sz_mb = sg_path.stat().st_size / (1024 * 1024)
                log.info(f"    → {sz_mb:.0f} MB ✓")
                converted += 1
            finally:
                candidate_path.unlink(missing_ok=True)
                candidate_provenance.unlink(missing_ok=True)
                fcntl.flock(graph_lock.fileno(), fcntl.LOCK_UN)

    log.info(
        f"  Conversion: {converted} new, {skipped} already existed, "
        f"{failed} failed"
    )
    if failed:
        raise RuntimeError(f"{failed} graph conversion(s) failed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GraphBrew frozen evaluation runner"
    )
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--exp", nargs="+", type=int, choices=range(1, 9),
                        help="Run specific experiment(s) by number")
    parser.add_argument("--preview", action="store_true",
                        help="Preview mode: small graphs, 1 trial, 2 benchmarks")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--graphs", nargs="+",
                        help="Override graph list (by name)")
    parser.add_argument("--algorithms", nargs="+",
                        help="Restrict to exact canonical keys (e.g. 8:csr 12:hrab)")
    parser.add_argument(
        "--benchmarks", nargs="+",
        help="Override the benchmark subset for a bounded run",
    )
    parser.add_argument(
        "--trials", type=int,
        help="Override process trials for a bounded run",
    )
    parser.add_argument("--graph-dir", type=str, default=str(PAPER_GRAPH_ROOT),
                        help="Directory containing graph files (.sg, .el)")
    parser.add_argument(
        "--artifact-root", type=str, default=str(PAPER_ARTIFACT_ROOT),
        help="Root for vldb_paper/, vldb_mappings/, and vldb_runs/",
    )
    parser.add_argument(
        "--measurement-generation",
        help="Explicit shared generation ID for distributed/fan-out runs",
    )
    parser.add_argument("--threads", type=int,
                        help="Base OpenMP thread count (default: 4 preview, 16 full)")
    parser.add_argument("--cpu-list", type=str,
                        help="taskset CPU list used for timing isolation (e.g. 0-15)")
    parser.add_argument("--cache-mode",
                        choices=["accurate", "fast", "ultrafast", "sampled"],
                        help="Cache simulator mode (default: ultrafast)")
    parser.add_argument("--cache-sample-rate", type=int, default=64,
                        help="Access sampling interval for --cache-mode sampled")
    parser.add_argument("--cache-sizes-kib", nargs="+", type=int,
                        help="Override cache-capacity sweep in KiB")
    parser.add_argument("--cache-all-algorithms", action="store_true",
                        help="Sweep the full paper algorithm matrix in experiment 1")
    parser.add_argument("--publish-paper-figures", action="store_true",
                        help="Copy generated figures/tables into the private paper workspace")
    parser.add_argument(
        "--paper-dir",
        type=str,
        default=os.environ.get("GRAPHBREW_PRIVATE_PAPER_ROOT"),
        help="Private paper workspace for optional publication export",
    )
    parser.add_argument("--64gb", action="store_true", dest="use_64gb",
                        help="Use 64 GB graph set (11 auto-downloadable graphs, no >1B-edge graphs)")
    parser.add_argument("--local", action="store_true", dest="use_local",
                        help="Use local graph set (6 graphs ≤117M edges, fits 64GB RAM, covers all types)")
    parser.add_argument("--skip-setup", action="store_true",
                        help="Skip auto-setup (build, download, convert)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip graph download (use existing graphs only)")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip figure generation after experiments")
    parser.add_argument("--figures-only", action="store_true",
                        help="Only generate figures from existing results (no experiments)")
    parser.add_argument("--tune-sssp-delta", action="store_true",
                        help="Tune weighted SSSP delta on SHUFFLED and exit")
    parser.add_argument(
        "--freeze-sssp-policy",
        action="store_true",
        help="Write reviewed tuner recommendations to the frozen policy SSOT",
    )
    parser.add_argument("--verify-gate", action="store_true",
                        help="Run the untimed semantic verification gate and exit")
    parser.add_argument(
        "--refresh-corpus",
        action="store_true",
        help="Refresh graph_source/v2 corpus artifacts and exit",
    )
    args = parser.parse_args()
    if args.freeze_sssp_policy and not args.tune_sssp_delta:
        parser.error("--freeze-sssp-policy requires --tune-sssp-delta")
    if args.publish_paper_figures and not args.paper_dir:
        parser.error(
            "--publish-paper-figures requires --paper-dir or "
            "GRAPHBREW_PRIVATE_PAPER_ROOT"
        )
    if args.paper_dir:
        os.environ["GRAPHBREW_PRIVATE_PAPER_ROOT"] = str(
            Path(args.paper_dir).resolve())
    if args.publish_paper_figures:
        os.environ["GRAPHBREW_PUBLISH_PAPER_FIGURES"] = "1"

    if (
        not args.all and not args.exp and not args.figures_only
        and not args.tune_sssp_delta and not args.verify_gate
        and not args.refresh_corpus
    ):
        parser.print_help()
        sys.exit(1)

    configure_artifact_root(args.artifact_root)
    configure_measurement_generation(args.measurement_generation)
    configure_runtime_policy(
        args.threads or (4 if args.preview else 16),
        args.cpu_list,
    )
    configure_cache_policy(
        preview=args.preview,
        mode=args.cache_mode or "ultrafast",
        sample_rate=args.cache_sample_rate,
        all_algorithms=args.cache_all_algorithms,
        sizes_kib=args.cache_sizes_kib,
    )
    configure_algorithm_filter(args.algorithms)
    configure_execution_mode(dry_run=args.dry_run)

    # ---- Figures-only mode ----
    if args.figures_only:
        log.info("Generating figures from existing results...")
        _generate_figures()
        return

    # Select configuration
    if args.preview:
        graphs = PREVIEW_GRAPHS
        benchmarks = BENCHMARKS_PREVIEW
        trials = TRIALS_PREVIEW
        timeout = TIMEOUT_PREVIEW
    elif getattr(args, "use_64gb", False):
        graphs = EVAL_GRAPHS_64GB
        benchmarks = BENCHMARKS
        trials = TRIALS_FULL
        timeout = TIMEOUT_FULL
    elif getattr(args, "use_local", False):
        graphs = EVAL_GRAPHS_LOCAL
        benchmarks = BENCHMARKS
        trials = TRIALS_FULL
        timeout = TIMEOUT_FULL
    else:
        graphs = EVAL_GRAPHS
        benchmarks = BENCHMARKS
        trials = TRIALS_FULL
        timeout = TIMEOUT_FULL

    benchmarks, trials = resolve_benchmark_policy(
        benchmarks,
        trials,
        args.benchmarks,
        args.trials,
    )

    # Override graphs if specified
    if args.graphs:
        graphs = select_requested_graphs(args.graphs)

    # Determine which experiments to run
    exp_ids = list(range(1, 9)) if args.all else (args.exp or [])
    if exp_ids == [1] and not args.preview and not args.graphs:
        graph_by_name = {graph["name"]: graph for graph in graphs}
        graphs = [
            graph_by_name[name] for name in CACHE_GRAPH_NAMES
            if name in graph_by_name
        ]
    elif exp_ids == [8] and not args.preview and not args.graphs:
        graph_by_name = {graph["name"]: graph for graph in graphs}
        graphs = [
            graph_by_name[name] for name in SCALABILITY_GRAPH_NAMES
            if name in graph_by_name
        ]

    if args.tune_sssp_delta:
        graph_dir_resolved = (
            str(PROJECT_ROOT / "results" / "graphs")
            if args.graph_dir == "."
            else args.graph_dir
        )
        if not args.freeze_sssp_policy:
            _setup_build_binaries(
                benchmarks=["sssp"],
                include_standard=True,
                include_sim=False,
                include_converter=False,
            )
        ensure_dir(RESULTS_DIR)
        tune_sssp_deltas(
            graphs=graphs,
            graph_dir=graph_dir_resolved,
            timeout=timeout,
            dry_run=args.dry_run,
            trials=1 if args.preview else SSSP_TUNING_TRIALS,
            freeze_policy=args.freeze_sssp_policy,
        )
        return

    if args.refresh_corpus:
        graph_dir_resolved = _refresh_graph_corpus(
            args.graph_dir,
            graphs,
            dry_run=args.dry_run,
            skip_download=args.skip_download,
            timeout=timeout,
        )
        log.info(
            "Refreshed current-semantics graph corpus at "
            f"{graph_dir_resolved}"
        )
        return

    if args.verify_gate:
        graph_dir_resolved = (
            str(PROJECT_ROOT / "results" / "graphs")
            if args.graph_dir == "."
            else args.graph_dir
        )
        _setup_build_binaries(
            benchmarks=benchmarks,
            include_standard=True,
            include_sim="pr" in benchmarks,
            include_converter=False,
            include_work=True,
            sim_benchmarks=["pr"],
        )
        ensure_dir(RESULTS_DIR)
        run_verification_gate(
            graphs=graphs,
            benchmarks=benchmarks,
            graph_dir=graph_dir_resolved,
            timeout=timeout,
            dry_run=args.dry_run,
        )
        return

    # ── Auto-setup: build, download, convert ──
    if not args.skip_setup:
        graph_dir_resolved = _setup_environment(
            args.graph_dir, graphs,
            dry_run=args.dry_run,
            skip_download=getattr(args, "skip_download", False),
            timeout=timeout,
        )
    else:
        # Still resolve default graph directory even when skipping setup
        if args.graph_dir == ".":
            graph_dir_resolved = str(PROJECT_ROOT / "results" / "graphs")
        else:
            graph_dir_resolved = args.graph_dir

    log.info(f"GraphBrew VLDB Paper Experiments")
    log.info(f"  Mode: {'preview' if args.preview else 'full'}")
    log.info(f"  Graphs: {len(graphs)} in {graph_dir_resolved}")
    log.info(f"  Benchmarks: {benchmarks}")
    log.info(f"  Trials: {trials}")
    log.info(f"  Threads: {_RUNTIME_THREADS}")
    log.info(f"  CPU list: {_RUNTIME_CPU_LIST or 'scheduler-managed'}")
    log.info(f"  Artifact root: {RESULTS_DIR.parent}")
    log.info(f"  Cache mode: {_CACHE_MODE}")
    log.info(f"  Experiments: {exp_ids}")
    log.info(f"  Dry run: {args.dry_run}")
    log.info("")

    preflight_benchmark_policies(
        graphs, benchmarks, graph_dir_resolved,
    )
    if args.preview:
        try:
            require_verification_gate(
                graphs, benchmarks, graph_dir_resolved,
            )
        except RuntimeError:
            log.info(
                "Preview gate missing or stale; running it automatically"
            )
            _setup_build_binaries(
                benchmarks=benchmarks,
                include_standard=True,
                include_sim="pr" in benchmarks,
                include_converter=False,
                include_work=True,
                sim_benchmarks=["pr"],
            )
            run_verification_gate(
                graphs=graphs,
                benchmarks=benchmarks,
                graph_dir=graph_dir_resolved,
                timeout=timeout,
                dry_run=args.dry_run,
            )
    else:
        require_verification_gate(
            graphs, benchmarks, graph_dir_resolved,
        )
        if any(eid != 1 for eid in exp_ids):
            require_timing_machine_policy()
    configure_campaign(
        graphs=graphs,
        graph_dir=graph_dir_resolved,
        experiment_ids=exp_ids,
        benchmarks=benchmarks,
        trials=trials,
    )
    ensure_dir(RESULTS_DIR)
    start = time.time()

    for eid in exp_ids:
        name, func = EXPERIMENTS[eid]
        log.info(f"\n{'#' * 60}")
        log.info(f"# Starting Experiment {eid}: {name}")
        log.info(f"{'#' * 60}\n")
        experiment_timeout = (
            REORDER_TIMEOUT_FULL
            if eid in {3, 8} and not args.preview
            else timeout
        )
        experiment_graphs = graphs
        if eid == 1 and not args.preview and not args.graphs:
            graph_by_name = {graph["name"]: graph for graph in graphs}
            experiment_graphs = [
                graph_by_name[name] for name in CACHE_GRAPH_NAMES
                if name in graph_by_name
            ]
        elif eid == 8 and not args.preview and not args.graphs:
            graph_by_name = {graph["name"]: graph for graph in graphs}
            experiment_graphs = [
                graph_by_name[name] for name in SCALABILITY_GRAPH_NAMES
                if name in graph_by_name
            ]
        func(experiment_graphs, benchmarks, trials, experiment_timeout, args.dry_run,
             graph_dir=graph_dir_resolved)

    elapsed = time.time() - start

    # Save reproducibility manifest
    save_manifest(args, elapsed)

    log.info(f"\nAll experiments completed in {elapsed:.1f}s")
    log.info(f"Results: {RESULTS_DIR}")
    log.info(f"Manifest: {RESULTS_DIR / 'MANIFEST.json'}")

    # Auto-generate figures unless --no-figures
    if not args.no_figures and not args.dry_run:
        log.info("\n--- Generating figures ---")
        _generate_figures()


def _generate_figures() -> None:
    """Invoke the figure generator on saved results."""
    fig_script = PROJECT_ROOT / "scripts" / "experiments" / "vldb" / "figures.py"
    cmd = [sys.executable, str(fig_script)]

    log.info(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError("VLDB figure generation failed")


if __name__ == "__main__":
    main()
