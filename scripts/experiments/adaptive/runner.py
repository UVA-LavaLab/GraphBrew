#!/usr/bin/env python3
"""Adaptive-selector Sprint experiment runner."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.vldb.config import (
    BC_SOURCE_ITERATIONS,
    EVAL_GRAPHS,
    PR_FIXED_ITERATIONS,
    PR_TOLERANCE,
    SSSP_POLICY,
)
from scripts.lib.core.experiment_policy import (
    ADAPTIVE_CACHE_BENCHMARK_ORDER,
    CACHE_CAPACITY_CANDIDATES_MIB,
    CACHE_PR_ITERATIONS,
    PAPER_BENCHMARK_ORDER,
    REORDER_SEMANTICS_VERSION,
)
from scripts.lib.pipeline.reorder_config import (
    parse_graphbrew_effective_configs,
    parse_graphbrew_realized_configs,
    validate_graphbrew_effective_configs,
    validate_graphbrew_realized_configs,
)
from scripts.lib.ml.feature_schema import TIER0_FEATURE_NAMES
from scripts.lib.ml.portfolio import (
    DEPLOYABLE_ARM_SPECS,
    CHARACTERIZATION_BASELINE_ARM_SPECS,
    CHARACTERIZATION_DENDROGRAM_ANCHOR,
)
from scripts.lib.ml.source_policy import (
    ADAPTIVE_SOURCE_COUNT,
    ADAPTIVE_SOURCE_MIN_REACHABILITY,
    ADAPTIVE_SOURCE_POLICY_ID,
    ADAPTIVE_SOURCE_SEED,
    SOURCE_DRIVEN_KERNELS,
)
from scripts.lib.ml.working_set import modeled_property_bytes
from scripts.lib.pipeline.benchmark import (
    SourceContractError,
    attach_source_trial_metadata,
    file_crc32,
    file_sha256,
    parse_benchmark_output,
    repository_scope_state,
)

SPRINT1_BUDGET_SCHEMA = "adaptive-sprint1-budget/v1"
SPRINT1_BUDGET_HOURS = 168.0
PROJECTION_SAFETY_FACTOR = 1.25
PROCESS_REPETITIONS = 10
PILOT_PROCESS_BLOCKS = 3
PILOT_RETRY_ATTEMPTS = (0, 1)
WARM_PROCESS_BLOCKS = 5
LARGE_GRAPH_EDGE_THRESHOLD = 100_000_000
BENCHMARKS = list(PAPER_BENCHMARK_ORDER)
CACHE_KERNELS = ADAPTIVE_CACHE_BENCHMARK_ORDER
CACHE_CAPACITY_MIB = CACHE_CAPACITY_CANDIDATES_MIB
HARDWARE_REPETITIONS = 3
HARDWARE_SINGLE_THREAD_FACTOR = 16
H4_KERNEL_HIGH_MULTIPLIER = 2.0
CACHE_MICRO_WALL_CAP_MULTIPLIER = 4.0
PILOT_DEFAULT_WALL_CAP_MULTIPLIER = 4.0
PROCESS_STARTUP_ALLOWANCE_SECONDS = 5.0
CACHE_SETUP_SINGLE_THREAD_FACTOR = 16.0
PILOT_FORBIDDEN_AMBIENT_ENV = (
    "GOMP_CPU_AFFINITY",
    "OMP_WAIT_POLICY",
    "OMP_SCHEDULE",
    "OMP_THREAD_LIMIT",
    "LD_PRELOAD",
    "MALLOC_ARENA_MAX",
    "GORDER_WINDOW",
    "MODEL_TREE_PATH",
)
PILOT_FORBIDDEN_APPLICATION_PREFIXES = (
    "OMP_",
    "ADAPTIVE_",
    "RABBIT_",
    "GORDER_",
    "MODEL_TREE_",
    "PERCEPTRON_",
    "ECG_",
    "CACHE_",
    "GRAPHBREW_",
)
PILOT_CONSUMER_REQUIRED_KEYS = frozenset({
    "command_id",
    "idempotency_key",
    "attempt",
    "phase",
    "graph",
    "graph_path",
    "command",
    "environment",
    "environment_mode",
    "timeout_seconds",
    "timeout_interpretation",
    "stdout_path",
    "stderr_path",
    "result_path",
    "depends_on",
    "retry_attempts",
    "graph_output_bytes",
    "graph_mtime_ns",
    "graph_crc32",
    "binary_provenance",
})
H4_DENDROGRAM_ANCHOR = CHARACTERIZATION_DENDROGRAM_ANCHOR
H4_KERNEL_PROXY = (
    "12:rabbit:compose:sg_none:"
    "comm_identity:intra_hubsort"
)
ALL_TIMING_ARMS = CHARACTERIZATION_BASELINE_ARM_SPECS
_KERNEL_EVIDENCE_PROXIES = {
    "10:canonical": "9:csr",
    "11:mind": "11",
    "11:bnf": "11",
    "15:1.0:10:10:hierarchy-degree": "12:leiden",
    "15:1.0:10:10:final-stable": "12:leiden",
    "15:1.0:10:10:final-degree": "12:leiden",
    H4_DENDROGRAM_ANCHOR: H4_KERNEL_PROXY,
}
_MAPPING_EVIDENCE_PROXIES = {
    "10:canonical": "9:csr",
    "11:mind": "11",
    "11:bnf": "11",
    "15:1.0:10:10:hierarchy-degree": "12:leiden",
    "15:1.0:10:10:final-stable": "12:leiden",
    "15:1.0:10:10:final-degree": "12:leiden",
}
_PROXY_HIGH_MULTIPLIER = 2.0
NATURAL_PILOT_GRAPHS = ("hollywood-2009",)
NATURAL_PILOT_EXCLUSIONS = {
    "twitter7": "available upstream .sg is already randomly labeled",
    "webbase-2001": "available upstream .sg is already randomly labeled",
}
NATURAL_PILOT_EXCLUSION_EVIDENCE = {
    "twitter7": {
        "upstream_path":
            "/media/Data/00_GraphDatasets/GBREW/TWTR/graph_0.sg",
        "upstream_normalized_edge_span": 0.3393342360777594,
        "upstream_window_neighbor_overlap": 0.0,
        "randomized_normalized_edge_span": 0.3378149564230687,
        "randomized_window_neighbor_overlap": 0.0,
    },
    "webbase-2001": {
        "upstream_path":
            "/media/Data/00_GraphDatasets/GBREW/WEB01/graph_0.sg",
        "upstream_normalized_edge_span": 0.3330649665115265,
        "upstream_window_neighbor_overlap": 0.0,
        "randomized_normalized_edge_span": 0.33120100716144896,
        "randomized_window_neighbor_overlap": 0.0,
    },
}
_CONVERSION_REPOSITORY_STATE_CACHE: dict[str, object] | None = None


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required timing artifact not found: {path}")
    with open(path) as stream:
        return json.load(stream)


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _artifact_binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path, use_cache=False),
    }


def _validate_artifact_binding(binding: dict[str, Any]) -> None:
    path = Path(str(binding.get("path", "")))
    if (
        not path.is_file()
        or path.stat().st_size != int(binding.get("bytes", -1))
        or file_sha256(path, use_cache=False)
            != binding.get("sha256")
    ):
        raise RuntimeError(
            f"Pilot input artifact changed after freeze: {path}")


def _execution_input_artifacts(
    budget_path: Path,
    source_path: Path,
    natural_path: Path,
    contract_path: Path,
) -> dict[str, dict[str, Any]]:
    return {
        "budget_projection": _artifact_binding(budget_path),
        "source_manifest": _artifact_binding(source_path),
        "natural_manifest": _artifact_binding(natural_path),
        "contract_weights": _artifact_binding(contract_path),
    }


def _conversion_repository_state() -> dict[str, object]:
    global _CONVERSION_REPOSITORY_STATE_CACHE
    if _CONVERSION_REPOSITORY_STATE_CACHE is None:
        _CONVERSION_REPOSITORY_STATE_CACHE = repository_scope_state(
            PROJECT_ROOT,
            (
                "Makefile",
                "bench/src/converter.cc",
                "bench/include",
            ),
        )
    return dict(_CONVERSION_REPOSITORY_STATE_CACHE)


def _current_graph_metadata(
    graph_path: Path,
    *,
    graph_name: str,
    natural: bool,
    verify_content: bool,
    expected_nodes: int,
    expected_undirected_edges: int,
) -> dict[str, Any]:
    metadata_path = graph_path.with_suffix(
        graph_path.suffix + ".meta.json")
    metadata = _load_json(metadata_path)
    expected_schema = (
        "adaptive-natural-graph/v2"
        if natural else "graph_source/v2"
    )
    required = {
        "schema": expected_schema,
        "reorder_semantics_version": REORDER_SEMANTICS_VERSION,
        "graph": graph_name,
        "output_path": str(graph_path.resolve()),
        "output_bytes": graph_path.stat().st_size,
        "converter_sha256": file_sha256(
            PROJECT_ROOT / "bench" / "bin" / "converter"
        ),
        "conversion_repository_state":
            _conversion_repository_state(),
        "expected_nodes": expected_nodes,
        "expected_undirected_edges": expected_undirected_edges,
        "nodes": expected_nodes,
        "undirected_edges": expected_undirected_edges,
    }
    for key, value in required.items():
        if metadata.get(key) != value:
            raise ValueError(
                "Current graph metadata mismatch: "
                f"{graph_name}/{key}")
    expected_crc = metadata.get("output_crc32")
    if not isinstance(expected_crc, str) or not re.fullmatch(
        r"[0-9a-f]{8}", expected_crc
    ):
        raise ValueError(
            f"Current graph CRC is missing: {graph_name}")
    if verify_content and file_crc32(graph_path) != expected_crc:
        raise ValueError(
            f"Current graph content changed: {graph_name}")
    return metadata


def _atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _atomic_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def select_medium_pilot_graph(
    graphs: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> str:
    """Select the median graph by frozen undirected edge count."""
    ordered = sorted(graphs, key=lambda graph: graph["undirected_edges"])
    return str(ordered[len(ordered) // 2]["name"])


def sprint1_pilot_graphs() -> tuple[str, str, str]:
    return (
        "twitter7",
        "webbase-2001",
        select_medium_pilot_graph(EVAL_GRAPHS),
    )


def reduced_capacity_set_mib(
    benchmark: str,
    nodes: int,
    directed_edges: int,
) -> tuple[int, ...]:
    """Return the pre-registered reduced H2 capacity set."""
    working_set = modeled_property_bytes(
        benchmark,
        nodes,
        directed_edges,
    )

    def nearest(target_bytes: float) -> int:
        return min(
            CACHE_CAPACITY_MIB,
            key=lambda capacity: abs(math.log(
                capacity * 1024 * 1024 / max(target_bytes, 1.0)
            )),
        )

    selected = {
        nearest(0.5 * working_set),
        nearest(working_set),
        nearest(2.0 * working_set),
        22,
    }
    selected.add(
        max(CACHE_CAPACITY_MIB)
        if working_set <= 22 * 1024 * 1024
        else min(CACHE_CAPACITY_MIB)
    )
    return tuple(sorted(selected))


def _index_unique(
    rows: list[dict[str, Any]],
    fields: tuple[str, ...],
    label: str,
) -> dict[tuple[str, ...], dict[str, Any]]:
    indexed = {}
    for row in rows:
        key = tuple(str(row[field]) for field in fields)
        if key in indexed:
            raise ValueError(f"Duplicate {label} row: {'|'.join(key)}")
        indexed[key] = row
    return indexed


def _index_required_overhead(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    required = {
        _MAPPING_EVIDENCE_PROXIES.get(arm, arm)
        for arm in ALL_TIMING_ARMS
        if arm != "0"
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        arm = str(row.get("algo_id"))
        if arm not in required:
            continue
        key = (str(row.get("graph")), arm)
        grouped.setdefault(key, []).append(row)

    indexed = {}
    for key, candidates in grouped.items():
        if len(candidates) == 1:
            indexed[key] = candidates[0]
            continue
        live = [
            row for row in candidates
            if row.get("timing_source") == "live-final"
        ]
        if len(live) != 1:
            raise ValueError(
                "Ambiguous Experiment 3 rows for "
                f"{key[0]}/{key[1]}"
            )
        indexed[key] = live[0]
    return indexed


def _trial_seconds_bounds(
    row: dict[str, Any],
) -> tuple[float, float]:
    values = [
        float(value)
        for value in row.get("trial_times", [])
        if float(value) > 0
    ]
    if values:
        return statistics.median(values), max(values)
    value = float(row.get("median_time", row.get("average_time", 0.0)))
    if value <= 0:
        raise ValueError(
            "Timing evidence lacks a positive kernel trial for "
            f"{row.get('graph')}/{row.get('algo_id')}/"
            f"{row.get('benchmark')}"
        )
    return value, value


def _cache_rate_by_graph_proxy(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    rates: dict[str, dict[str, list[float]]] = {
        "0": {},
        "5": {},
        "8:csr": {},
    }
    for row in rows:
        if row.get("benchmark") != "pr":
            raise ValueError(
                "Cache projection evidence must contain only PR rows")
        key = str(row.get("algo_key"))
        if key not in rates:
            continue
        graph = str(row.get("graph"))
        processed = row.get("directed_edges_processed", 0)
        if isinstance(processed, list):
            processed = processed[0] if processed else 0
        processed = float(processed)
        elapsed = float(row.get("average_time", 0.0))
        if processed > 0 and elapsed > 0:
            rates[key].setdefault(graph, []).append(elapsed / processed)
    missing = [key for key, by_graph in rates.items() if not by_graph]
    if missing:
        raise ValueError(
            "Cache timing evidence is missing proxy arms: "
            + " ".join(missing)
        )
    return {
        key: {
            graph: statistics.median(values)
            for graph, values in by_graph.items()
        }
        for key, by_graph in rates.items()
    }


def _artifact_provenance(
    path: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    measured = sorted({
        str(row["measured_at"])
        for row in rows
        if row.get("measured_at")
    })
    return {
        "path": str(path),
        "row_count": len(rows),
        "cohort_ids": sorted({
            str(row["cohort_id"])
            for row in rows
            if row.get("cohort_id")
        }),
        "policy_ids": sorted({
            str(row["policy_id"])
            for row in rows
            if row.get("policy_id")
        }),
        "measured_at_first": measured[0] if measured else None,
        "measured_at_last": measured[-1] if measured else None,
    }


def _cache_rate_bounds(
    rates: dict[str, dict[str, float]],
    proxy: str,
    target_edges: int,
) -> tuple[float, float, str]:
    graph_edges = {
        str(graph["name"]): int(graph["undirected_edges"])
        for graph in EVAL_GRAPHS
    }
    candidates = rates[proxy]
    nearest_graph = min(
        candidates,
        key=lambda graph: abs(math.log(
            graph_edges[graph] / max(target_edges, 1)
        )),
    )
    return (
        candidates[nearest_graph],
        max(candidates.values()),
        nearest_graph,
    )


def _has_fit_boundary(
    benchmark: str,
    nodes: int,
    directed_edges: int,
) -> bool:
    working_set = modeled_property_bytes(
        benchmark,
        nodes,
        directed_edges,
    )
    return any(
        0.5 <= capacity * 1024 * 1024 / working_set <= 2.0
        for capacity in CACHE_CAPACITY_MIB
    )


def _kernel_evidence_arm(arm: str) -> str:
    return _KERNEL_EVIDENCE_PROXIES.get(arm, arm)


def _mapping_evidence_arm(arm: str) -> str:
    return _MAPPING_EVIDENCE_PROXIES.get(arm, arm)


def _characterization_arm_role(arm: str) -> str:
    return (
        "diagnostic-anchor"
        if arm == H4_DENDROGRAM_ANCHOR
        else "attributed-candidate"
    )


def _cache_proxy_arm(arm: str) -> str:
    return arm if arm in {"0", "5", "8:csr"} else "8:csr"


def _measurement_shape(
    benchmark: str,
    undirected_edges: int,
) -> tuple[str, int, int]:
    if benchmark not in SOURCE_DRIVEN_KERNELS:
        return "cold-process", PROCESS_REPETITIONS, PROCESS_REPETITIONS
    if undirected_edges > LARGE_GRAPH_EDGE_THRESHOLD:
        return (
            "warm-block",
            WARM_PROCESS_BLOCKS,
            ADAPTIVE_SOURCE_COUNT * PROCESS_REPETITIONS,
        )
    return (
        "cold-process",
        PROCESS_REPETITIONS,
        ADAPTIVE_SOURCE_COUNT * PROCESS_REPETITIONS,
    )


def build_sprint1_budget_projection(
    artifact_root: Path,
    graph_root: Path,
    *,
    budget_hours: float = SPRINT1_BUDGET_HOURS,
    safety_factor: float = PROJECTION_SAFETY_FACTOR,
) -> dict[str, Any]:
    """Build the pre-data Sprint-1 node-hour projection."""
    if budget_hours <= 0 or safety_factor < 1:
        raise ValueError("Budget must be positive and safety factor at least 1")
    missing_graphs = [
        graph["name"]
        for graph in EVAL_GRAPHS
        if not (
            graph_root / graph["name"] / f"{graph['name']}.sg"
        ).is_file()
    ]
    if missing_graphs:
        raise FileNotFoundError(
            "Frozen Sprint-1 graph files are missing: "
            + " ".join(missing_graphs)
        )

    paper_root = artifact_root / "vldb_paper"
    exp2 = _load_json(
        paper_root / "exp2_speedup" / "speedup_results.json")
    exp3 = _load_json(
        paper_root / "exp3_overhead" / "overhead_results.json")
    cache = _load_json(
        paper_root / "exp1_cache" / "cache_results.json")
    if not all(isinstance(rows, list) for rows in (exp2, exp3, cache)):
        raise ValueError("VLDB timing artifacts have an invalid top-level shape")

    speed = _index_unique(
        exp2,
        ("graph", "algo_id", "benchmark"),
        "Experiment 2",
    )
    overhead = _index_required_overhead(exp3)
    cache_rates = _cache_rate_by_graph_proxy(cache)
    graph_metadata = {
        str(graph["name"]): _current_graph_metadata(
            graph_root
            / str(graph["name"])
            / f"{graph['name']}.sg",
            graph_name=str(graph["name"]),
            natural=False,
            verify_content=True,
            expected_nodes=int(graph["nodes"]),
            expected_undirected_edges=
                int(graph["undirected_edges"]),
        )
        for graph in EVAL_GRAPHS
    }
    graph_output_bytes = {
        graph_name: int(metadata["output_bytes"])
        for graph_name, metadata in graph_metadata.items()
    }
    max_read_by_graph = {}
    for graph in EVAL_GRAPHS:
        graph_name = str(graph["name"])
        max_read_by_graph[graph_name] = max(
            float(row.get("read_time", 0.0))
            for row in exp2
            if str(row.get("graph")) == graph_name
        )
        if max_read_by_graph[graph_name] <= 0:
            raise ValueError(
                f"Read timing evidence is missing for {graph_name}")
        for arm in ALL_TIMING_ARMS:
            evidence_arm = _kernel_evidence_arm(arm)
            if not all(
                (graph_name, evidence_arm, benchmark) in speed
                for benchmark in BENCHMARKS
            ):
                raise ValueError(
                    f"Kernel timing evidence is incomplete: "
                    f"{graph_name}/{evidence_arm}")

    kernel_rows: list[dict[str, Any]] = []
    kernel_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for graph in EVAL_GRAPHS:
        graph_name = str(graph["name"])
        undirected_edges = int(graph["undirected_edges"])
        for arm in ALL_TIMING_ARMS:
            evidence_arm = _kernel_evidence_arm(arm)
            map_seconds = 0.0
            map_seconds_high = 0.0
            mapping_evidence_arm = None
            if arm != "0":
                mapping_evidence_arm = _mapping_evidence_arm(arm)
                try:
                    map_seconds = float(
                        overhead[
                            (graph_name, mapping_evidence_arm)
                        ]["reorder_time"])
                except KeyError as error:
                    raise ValueError(
                        "Missing reorder evidence for "
                        f"{graph_name}/{mapping_evidence_arm} "
                        f"(characterization arm {arm})"
                    ) from error
                map_seconds_high = map_seconds * (
                    _PROXY_HIGH_MULTIPLIER
                    if mapping_evidence_arm != arm else 1.0
                )
            for benchmark in BENCHMARKS:
                try:
                    evidence = speed[
                        (graph_name, evidence_arm, benchmark)]
                except KeyError as error:
                    raise ValueError(
                        "Missing kernel evidence for "
                        f"{graph_name}/{evidence_arm}/{benchmark}"
                    ) from error
                read_seconds = float(evidence.get("read_time", 0.0))
                kernel_seconds, kernel_seconds_high = (
                    _trial_seconds_bounds(evidence)
                )
                if evidence_arm != arm:
                    kernel_seconds_high *= _PROXY_HIGH_MULTIPLIER
                if arm == H4_DENDROGRAM_ANCHOR:
                    kernel_seconds_high = max(
                        kernel_seconds_high,
                        kernel_seconds * H4_KERNEL_HIGH_MULTIPLIER,
                    )
                mode, process_blocks, timed_trials = _measurement_shape(
                    benchmark,
                    undirected_edges,
                )
                raw_seconds_low = (
                    process_blocks * (read_seconds + map_seconds)
                    + timed_trials * kernel_seconds
                )
                raw_seconds_high = (
                    process_blocks
                    * (read_seconds + map_seconds_high)
                    + timed_trials * kernel_seconds_high
                )
                cap_read_seconds = max_read_by_graph[graph_name]
                cap_floor_raw_seconds = (
                    process_blocks
                    * (
                        cap_read_seconds
                        + map_seconds_high
                        + PROCESS_STARTUP_ALLOWANCE_SECONDS
                    )
                    + timed_trials * kernel_seconds_high
                )
                row = {
                    "phase": "randomized-kernel",
                    "graph": graph_name,
                    "kernel": benchmark,
                    "arm": arm,
                    "arm_role": _characterization_arm_role(arm),
                    "measurement_mode": mode,
                    "page_cache_regime": "retained-page-cache",
                    "labeling": "randomized",
                    "graph_path": str(
                        graph_root
                        / graph_name
                        / f"{graph_name}.sg"),
                    "process_blocks": process_blocks,
                    "timed_trials": timed_trials,
                    "read_seconds_per_block": read_seconds,
                    "map_seconds_per_block": map_seconds,
                    "map_seconds_per_block_high": map_seconds_high,
                    "kernel_seconds_per_trial": kernel_seconds,
                    "kernel_seconds_per_trial_high":
                        kernel_seconds_high,
                    "raw_seconds_low": raw_seconds_low,
                    "raw_seconds_high": raw_seconds_high,
                    "cap_floor_raw_seconds": cap_floor_raw_seconds,
                    "cap_read_seconds_per_block": cap_read_seconds,
                    "cap_implied_read_bandwidth_mib_s": (
                        graph_output_bytes[graph_name]
                        / max(cap_read_seconds, 1e-12)
                        / (1024 * 1024)
                    ),
                    "process_startup_allowance_seconds":
                        PROCESS_STARTUP_ALLOWANCE_SECONDS,
                    "buffered_node_hours_low":
                        raw_seconds_low * safety_factor / 3600.0,
                    "buffered_node_hours_high":
                        raw_seconds_high * safety_factor / 3600.0,
                    "kernel_evidence_arm": evidence_arm,
                    "mapping_evidence_arm": mapping_evidence_arm,
                    "mapping_evidence": (
                        "none"
                        if arm == "0"
                        else (
                            "historical-proxy"
                            if mapping_evidence_arm != arm
                            else "exp3-live-final"
                        )
                    ),
                }
                kernel_rows.append(row)
                kernel_lookup[(graph_name, arm, benchmark)] = row

    cache_rows: list[dict[str, Any]] = []
    excluded_cache_cells: list[dict[str, str]] = []
    for graph in EVAL_GRAPHS:
        graph_name = str(graph["name"])
        nodes = int(graph["nodes"])
        directed_edges = 2 * int(graph["undirected_edges"])
        for arm in ALL_TIMING_ARMS:
            proxy = _cache_proxy_arm(arm)
            pr_seconds = kernel_lookup[
                (graph_name, arm, "pr")
            ]["kernel_seconds_per_trial"]
            for benchmark in CACHE_KERNELS:
                if not _has_fit_boundary(
                    benchmark,
                    nodes,
                    directed_edges,
                ):
                    excluded_cache_cells.append({
                        "graph": graph_name,
                        "kernel": benchmark,
                        "reason": "no-0.5x-to-2x-capacity",
                    })
                    continue
                native_seconds = kernel_lookup[
                    (graph_name, arm, benchmark)
                ]["kernel_seconds_per_trial"]
                native_ratio_low = max(
                    0.25,
                    native_seconds / pr_seconds,
                )
                native_ratio_high = max(
                    1.0,
                    2.0 * native_ratio_low,
                )
                capacities = reduced_capacity_set_mib(
                    benchmark,
                    nodes,
                    directed_edges,
                )
                rate_low, rate_high, rate_graph = _cache_rate_bounds(
                    cache_rates,
                    proxy,
                    int(graph["undirected_edges"]),
                )
                simulated_seconds_low = (
                    rate_low
                    * directed_edges
                    * CACHE_PR_ITERATIONS
                    * native_ratio_low
                )
                simulated_seconds_high = (
                    rate_high
                    * directed_edges
                    * CACHE_PR_ITERATIONS
                    * native_ratio_high
                )
                source_multiplier = (
                    ADAPTIVE_SOURCE_COUNT
                    if benchmark in SOURCE_DRIVEN_KERNELS
                    else 1
                )
                evidence = kernel_lookup[
                    (graph_name, arm, benchmark)]
                raw_seconds_low = len(capacities) * (
                    evidence["read_seconds_per_block"]
                    + evidence["map_seconds_per_block"]
                    + source_multiplier * simulated_seconds_low
                )
                raw_seconds_high = len(capacities) * (
                    evidence["read_seconds_per_block"]
                    + evidence["map_seconds_per_block_high"]
                    + source_multiplier * simulated_seconds_high
                )
                cap_floor_raw_seconds = (
                    CACHE_SETUP_SINGLE_THREAD_FACTOR
                    * (
                        evidence["cap_read_seconds_per_block"]
                        + evidence["map_seconds_per_block_high"]
                    )
                    + simulated_seconds_high
                    + PROCESS_STARTUP_ALLOWANCE_SECONDS
                )
                cache_rows.append({
                    "phase": "randomized-cache",
                    "graph": graph_name,
                    "kernel": benchmark,
                    "arm": arm,
                    "arm_role": _characterization_arm_role(arm),
                    "measurement_mode": "cache-simulator",
                    "capacities_mib": ",".join(map(str, capacities)),
                    "capacity_count": len(capacities),
                    "source_multiplier": source_multiplier,
                    "cache_proxy_arm": proxy,
                    "cache_rate_evidence_graph": rate_graph,
                    "cache_rate_low": rate_low,
                    "cache_rate_high": rate_high,
                    "native_kernel_ratio_low": native_ratio_low,
                    "native_kernel_ratio_high": native_ratio_high,
                    "read_seconds_per_capacity":
                        evidence["read_seconds_per_block"],
                    "map_seconds_per_capacity":
                        evidence["map_seconds_per_block"],
                    "simulated_seconds_low": simulated_seconds_low,
                    "simulated_seconds_high": simulated_seconds_high,
                    "raw_seconds_low": raw_seconds_low,
                    "raw_seconds_high": raw_seconds_high,
                    "cap_floor_raw_seconds": cap_floor_raw_seconds,
                    "buffered_node_hours_low":
                        raw_seconds_low * safety_factor / 3600.0,
                    "buffered_node_hours_high":
                        raw_seconds_high * safety_factor / 3600.0,
                })

    pilot_graphs = sprint1_pilot_graphs()
    natural_pilot_rows: list[dict[str, Any]] = []
    for graph_name in NATURAL_PILOT_GRAPHS:
        for arm in ALL_TIMING_ARMS:
            for benchmark in BENCHMARKS:
                evidence = kernel_lookup[
                    (graph_name, arm, benchmark)]
                timed_trials = (
                    PILOT_PROCESS_BLOCKS * ADAPTIVE_SOURCE_COUNT
                    if benchmark in SOURCE_DRIVEN_KERNELS
                    else PILOT_PROCESS_BLOCKS
                )
                raw_seconds_low = (
                    PILOT_PROCESS_BLOCKS
                    * (
                        evidence["read_seconds_per_block"]
                        + evidence["map_seconds_per_block"]
                    )
                    + timed_trials
                    * evidence["kernel_seconds_per_trial"]
                )
                raw_seconds_high = (
                    PILOT_PROCESS_BLOCKS
                    * (
                        evidence["read_seconds_per_block"]
                        + evidence["map_seconds_per_block_high"]
                    )
                    + timed_trials
                    * evidence["kernel_seconds_per_trial_high"]
                )
                natural_pilot_rows.append({
                    **evidence,
                    "phase": "natural-label-pilot",
                    "measurement_mode": "cold-process",
                    "labeling": "natural",
                    "graph_path": str(
                        graph_root
                        / graph_name
                        / f"{graph_name}.natural.sg"),
                    "claim_eligible": False,
                    "pilot_only": True,
                    "process_blocks": PILOT_PROCESS_BLOCKS,
                    "timed_trials": timed_trials,
                    "raw_seconds_low": raw_seconds_low,
                    "raw_seconds_high": raw_seconds_high,
                    "cap_floor_raw_seconds": (
                        PILOT_PROCESS_BLOCKS
                        * (
                            evidence["cap_read_seconds_per_block"]
                            + evidence["map_seconds_per_block_high"]
                            + PROCESS_STARTUP_ALLOWANCE_SECONDS
                        )
                        + timed_trials
                        * evidence["kernel_seconds_per_trial_high"]
                    ),
                    "buffered_node_hours_low":
                        raw_seconds_low * safety_factor / 3600.0,
                    "buffered_node_hours_high":
                        raw_seconds_high * safety_factor / 3600.0,
                })
    randomized_pilot_rows: list[dict[str, Any]] = []
    for graph_name in pilot_graphs:
        for arm in ALL_TIMING_ARMS:
            for benchmark in BENCHMARKS:
                evidence = kernel_lookup[
                    (graph_name, arm, benchmark)]
                timed_trials = (
                    PILOT_PROCESS_BLOCKS * ADAPTIVE_SOURCE_COUNT
                    if benchmark in SOURCE_DRIVEN_KERNELS
                    else PILOT_PROCESS_BLOCKS
                )
                raw_seconds_low = (
                    PILOT_PROCESS_BLOCKS
                    * (
                        evidence["read_seconds_per_block"]
                        + evidence["map_seconds_per_block"]
                    )
                    + timed_trials
                    * evidence["kernel_seconds_per_trial"]
                )
                raw_seconds_high = (
                    PILOT_PROCESS_BLOCKS
                    * (
                        evidence["read_seconds_per_block"]
                        + evidence["map_seconds_per_block_high"]
                    )
                    + timed_trials
                    * evidence["kernel_seconds_per_trial_high"]
                )
                randomized_pilot_rows.append({
                    **evidence,
                    "phase": "randomized-pilot",
                    "measurement_mode": "cold-process",
                    "labeling": "randomized",
                    "claim_eligible": False,
                    "pilot_only": True,
                    "process_blocks": PILOT_PROCESS_BLOCKS,
                    "timed_trials": timed_trials,
                    "raw_seconds_low": raw_seconds_low,
                    "raw_seconds_high": raw_seconds_high,
                    "cap_floor_raw_seconds": (
                        PILOT_PROCESS_BLOCKS
                        * (
                            evidence["cap_read_seconds_per_block"]
                            + evidence["map_seconds_per_block_high"]
                            + PROCESS_STARTUP_ALLOWANCE_SECONDS
                        )
                        + timed_trials
                        * evidence["kernel_seconds_per_trial_high"]
                    ),
                    "buffered_node_hours_low":
                        raw_seconds_low * safety_factor / 3600.0,
                    "buffered_node_hours_high":
                        raw_seconds_high * safety_factor / 3600.0,
                })

    materialization_rows = []
    for graph_name in NATURAL_PILOT_GRAPHS:
        metadata = _load_json(
            graph_root / graph_name / f"{graph_name}.sg.meta.json")
        output_bytes = int(metadata.get("output_bytes", 0))
        if output_bytes <= 0:
            raise ValueError(
                f"Graph metadata lacks output_bytes for {graph_name}")
        read_seconds = float(
            kernel_lookup[(graph_name, "0", "pr")]
            ["read_seconds_per_block"]
        )
        raw_seconds_low = (
            read_seconds + output_bytes / (500 * 1024 * 1024)
        )
        raw_seconds_high = max(
            2.0 * read_seconds + output_bytes / (200 * 1024 * 1024),
            3600.0 if graph_name in {"twitter7", "webbase-2001"}
            else 600.0,
        )
        materialization_rows.append({
            "phase": "natural-label-materialization",
            "graph": graph_name,
            "kernel": "none",
            "arm": "0",
            "arm_role": "dataset",
            "measurement_mode": "converter-symmetrize-original-labels",
            "raw_seconds_low": raw_seconds_low,
            "raw_seconds_high": raw_seconds_high,
            "buffered_node_hours_low":
                raw_seconds_low * safety_factor / 3600.0,
            "buffered_node_hours_high":
                raw_seconds_high * safety_factor / 3600.0,
        })

    rss_rows = []
    for graph_name in ("twitter7", "webbase-2001"):
        for arm in ALL_TIMING_ARMS:
            evidence = kernel_lookup[(graph_name, arm, "pr")]
            raw_seconds_low = (
                evidence["read_seconds_per_block"]
                + evidence["map_seconds_per_block"]
            )
            raw_seconds_high = (
                evidence["read_seconds_per_block"]
                + evidence["map_seconds_per_block_high"]
            )
            rss_rows.append({
                "phase": "rss-pilot",
                "graph": graph_name,
                "kernel": "none",
                "arm": arm,
                "arm_role": _characterization_arm_role(arm),
                "measurement_mode": "peak-rss",
                "raw_seconds_low": raw_seconds_low,
                "raw_seconds_high": raw_seconds_high,
                "cap_floor_raw_seconds": (
                    evidence["cap_read_seconds_per_block"]
                    + evidence["map_seconds_per_block_high"]
                    + evidence["kernel_seconds_per_trial_high"]
                    / PR_FIXED_ITERATIONS
                    + PROCESS_STARTUP_ALLOWANCE_SECONDS
                ),
                "buffered_node_hours_low":
                    raw_seconds_low * safety_factor / 3600.0,
                "buffered_node_hours_high":
                    raw_seconds_high * safety_factor / 3600.0,
            })

    representative_costs = [
        row["read_seconds_per_block"]
        + row["map_seconds_per_block"]
        + row["kernel_seconds_per_trial"]
        for row in kernel_rows
        if row["arm_role"] == "attributed-candidate"
    ]
    hardware_seconds_low = 30 * statistics.median(representative_costs)
    hardware_seconds_high = (
        hardware_seconds_low
        * HARDWARE_REPETITIONS
        * HARDWARE_SINGLE_THREAD_FACTOR
    )
    hardware_rows = [{
        "phase": "hardware-validation",
        "graph": "representative-30-cell-cohort",
        "kernel": "mixed",
        "arm": "mixed",
        "arm_role": "validation",
        "measurement_mode": "perf-single-thread",
        "raw_seconds_low": hardware_seconds_low,
        "raw_seconds_high": hardware_seconds_high,
        "buffered_node_hours_low":
            hardware_seconds_low * safety_factor / 3600.0,
        "buffered_node_hours_high":
            hardware_seconds_high * safety_factor / 3600.0,
    }]

    cache_micro_specs = (
        ("twitter7", "pr", "8:csr", 22, None, "graph-factor"),
        ("webbase-2001", "pr", "8:csr", 22, None, "graph-factor"),
        ("hollywood-2009", "pr", "8:csr", 22, None, "star-center"),
        ("hollywood-2009", "bfs", "8:csr", 22, 0, "kernel-factor"),
        ("hollywood-2009", "pr", "0", 22, None, "arm-factor"),
        ("hollywood-2009", "pr", "5", 22, None, "arm-factor"),
        ("hollywood-2009", "pr", "8:csr", 512, None, "capacity-factor"),
        ("twitter7", "bfs", "8:csr", 22, 0, "graph-kernel-interaction"),
        ("hollywood-2009", "bfs", "8:csr", 22, 7, "source-dispersion"),
    )
    cache_micro_pilot_rows = []
    for (
        graph_name,
        benchmark,
        arm,
        capacity_mib,
        source_index,
        probe_role,
    ) in cache_micro_specs:
        evidence = next(
            row for row in cache_rows
            if row["graph"] == graph_name
            and row["kernel"] == benchmark
            and row["arm"] == arm
        )
        raw_seconds_low = (
            float(evidence["read_seconds_per_capacity"])
            + float(evidence["map_seconds_per_capacity"])
            + float(evidence["simulated_seconds_low"])
        )
        raw_seconds_high = (
            float(evidence["read_seconds_per_capacity"])
            + float(evidence["map_seconds_per_capacity"])
            + float(evidence["simulated_seconds_high"])
        )
        cap_multiplier = (
            10.0 if benchmark == "bfs"
            else CACHE_MICRO_WALL_CAP_MULTIPLIER
        )
        cache_micro_pilot_rows.append({
            **evidence,
            "phase": "cache-micro-pilot",
            "measurement_mode":
                f"cache-simulator-{capacity_mib}mib",
            "capacities_mib": str(capacity_mib),
            "capacity_count": 1,
            "source_multiplier": 1,
            "raw_seconds_low": raw_seconds_low,
            "raw_seconds_high": raw_seconds_high,
            "buffered_node_hours_low":
                raw_seconds_low * safety_factor / 3600.0,
            "buffered_node_hours_high":
                raw_seconds_high * safety_factor / 3600.0,
            "wall_clock_cap_seconds":
                cap_multiplier * max(
                    raw_seconds_high,
                    float(evidence["cap_floor_raw_seconds"]),
                ),
            "wall_clock_cap_multiplier": cap_multiplier,
            "timeout_interpretation":
                "right-censored-lower-bound",
            "source_policy_id": (
                ADAPTIVE_SOURCE_POLICY_ID
                if benchmark in SOURCE_DRIVEN_KERNELS
                else None
            ),
            "source_index": source_index,
            "requested_octile": (
                source_index
                if benchmark in SOURCE_DRIVEN_KERNELS
                else None
            ),
            "source_vertex_id": None,
            "probe_role": probe_role,
        })

    feature_pilot_rows = []
    for graph_name in pilot_graphs:
        evidence = kernel_lookup[(graph_name, "0", "pr")]
        labelings = ["randomized"]
        if graph_name in NATURAL_PILOT_GRAPHS:
            labelings.append("natural")
        for labeling in labelings:
            feature_allowance = (
                300.0
                if graph_name in {"twitter7", "webbase-2001"}
                else 120.0
            )
            raw_seconds_low = (
                evidence["read_seconds_per_block"]
                + evidence["kernel_seconds_per_trial"]
            )
            raw_seconds_high = (
                evidence["read_seconds_per_block"]
                + evidence["kernel_seconds_per_trial_high"]
                + feature_allowance
            )
            feature_pilot_rows.append({
                "phase": "feature-cost-pilot",
                "graph": graph_name,
                "kernel": "pr",
                "arm": "14:perceptron-contract-original",
                "arm_role": "feature-extractor",
                "labeling": labeling,
                "measurement_mode": "adaptive-original-contract",
                "raw_seconds_low": raw_seconds_low,
                "raw_seconds_high": raw_seconds_high,
                "cap_floor_raw_seconds": (
                    evidence["cap_read_seconds_per_block"]
                    + evidence["kernel_seconds_per_trial_high"]
                    + feature_allowance
                    + PROCESS_STARTUP_ALLOWANCE_SECONDS
                ),
                "buffered_node_hours_low":
                    raw_seconds_low * safety_factor / 3600.0,
                "buffered_node_hours_high":
                    raw_seconds_high * safety_factor / 3600.0,
                "feature_model_apply_allowance_seconds":
                    feature_allowance,
                "tier0_trained": False,
                "wall_clock_cap_seconds":
                    CACHE_MICRO_WALL_CAP_MULTIPLIER
                    * max(
                        raw_seconds_high,
                        evidence["cap_read_seconds_per_block"]
                        + evidence["kernel_seconds_per_trial_high"]
                        + feature_allowance
                        + PROCESS_STARTUP_ALLOWANCE_SECONDS,
                    ),
                "timeout_interpretation":
                    "right-censored-lower-bound",
            })

    escalation_rows = []
    for row in kernel_rows:
        if row["kernel"] not in SOURCE_DRIVEN_KERNELS:
            continue
        extra_blocks = (
            2 * int(row["process_blocks"])
            if row["measurement_mode"] == "warm-block"
            else 20
        )
        extra_trials = 2 * int(row["timed_trials"])
        raw_seconds_high = (
            extra_blocks
            * (
                float(row["read_seconds_per_block"])
                + float(row["map_seconds_per_block_high"])
            )
            + extra_trials
            * float(row["kernel_seconds_per_trial_high"])
        )
        escalation_rows.append({
            "phase": "escalation-reserve",
            "graph": row["graph"],
            "kernel": row["kernel"],
            "arm": row["arm"],
            "arm_role": row["arm_role"],
            "measurement_mode":
                f"{row['measurement_mode']}-30-repetition-reserve",
            "raw_seconds_low": 0.0,
            "raw_seconds_high": raw_seconds_high,
            "buffered_node_hours_low": 0.0,
            "buffered_node_hours_high":
                raw_seconds_high * safety_factor / 3600.0,
        })

    all_rows = (
        kernel_rows + cache_rows + natural_pilot_rows
        + materialization_rows + rss_rows
        + hardware_rows + escalation_rows
    )
    phase_hours_low: dict[str, float] = {}
    phase_hours_high: dict[str, float] = {}
    for row in all_rows:
        phase = str(row["phase"])
        phase_hours_low[phase] = (
            phase_hours_low.get(phase, 0.0)
            + float(row["buffered_node_hours_low"])
        )
        phase_hours_high[phase] = (
            phase_hours_high.get(phase, 0.0)
            + float(row["buffered_node_hours_high"])
        )
    projected_hours_low = sum(phase_hours_low.values())
    projected_hours_high = sum(phase_hours_high.values())
    full_status = (
        "within-budget"
        if projected_hours_high <= budget_hours
        else "budget-amendment-required"
    )
    pilot_high = (
        sum(
            float(row["buffered_node_hours_high"])
            for row in randomized_pilot_rows
        )
        + phase_hours_high.get("natural-label-pilot", 0.0)
        + phase_hours_high.get(
            "natural-label-materialization", 0.0)
        + phase_hours_high.get("rss-pilot", 0.0)
        + sum(
            float(row["buffered_node_hours_high"])
            for row in cache_micro_pilot_rows
        )
        + sum(
            float(row["buffered_node_hours_high"])
            for row in feature_pilot_rows
        )
    )
    authorized_phases = (
        "randomized-pilot",
        "natural-label-pilot",
        "natural-label-materialization",
        "rss-pilot",
        "cache-micro-pilot",
        "feature-cost-pilot",
    )
    row_groups = (
        kernel_rows,
        cache_rows,
        randomized_pilot_rows,
        cache_micro_pilot_rows,
        feature_pilot_rows,
        natural_pilot_rows,
        materialization_rows,
        rss_rows,
        hardware_rows,
        escalation_rows,
    )
    for rows in row_groups:
        for row in rows:
            row["authorized_for_collection"] = (
                row["phase"] in authorized_phases
            )
            if row["authorized_for_collection"]:
                row.setdefault("pilot_only", True)
                row.setdefault("claim_eligible", False)
            if (
                row["authorized_for_collection"]
                and "wall_clock_cap_seconds" not in row
            ):
                row["wall_clock_cap_seconds"] = (
                    PILOT_DEFAULT_WALL_CAP_MULTIPLIER
                    * max(
                        float(row["raw_seconds_high"]),
                        float(row.get(
                            "cap_floor_raw_seconds",
                            row["raw_seconds_high"],
                        )),
                    )
                )
                row["wall_clock_cap_multiplier"] = (
                    PILOT_DEFAULT_WALL_CAP_MULTIPLIER
                )
                row["timeout_interpretation"] = (
                    "right-censored-lower-bound"
                )
    pilot_phase_hours_low: dict[str, float] = {}
    pilot_phase_hours_high: dict[str, float] = {}
    for rows in row_groups:
        for row in rows:
            if not row["authorized_for_collection"]:
                continue
            phase = str(row["phase"])
            pilot_phase_hours_low[phase] = (
                pilot_phase_hours_low.get(phase, 0.0)
                + float(row["buffered_node_hours_low"])
            )
            pilot_phase_hours_high[phase] = (
                pilot_phase_hours_high.get(phase, 0.0)
                + float(row["buffered_node_hours_high"])
            )
    feature_hours_low = sum(
        float(row["buffered_node_hours_low"])
        for row in feature_pilot_rows
    )
    feature_hours_high = sum(
        float(row["buffered_node_hours_high"])
        for row in feature_pilot_rows
    )
    randomized_pilot_hours_low = sum(
        float(row["buffered_node_hours_low"])
        for row in randomized_pilot_rows
    )
    randomized_pilot_hours_high = sum(
        float(row["buffered_node_hours_high"])
        for row in randomized_pilot_rows
    )
    cache_micro_hours_low = sum(
        float(row["buffered_node_hours_low"])
        for row in cache_micro_pilot_rows
    )
    cache_micro_hours_high = sum(
        float(row["buffered_node_hours_high"])
        for row in cache_micro_pilot_rows
    )
    capped_rows = [
        row
        for rows in row_groups
        for row in rows
        if row["authorized_for_collection"]
    ]
    pilot_worst_case = (
        pilot_high
        - sum(
            float(row["buffered_node_hours_high"])
            for row in capped_rows
        )
        + sum(
            (
                float(row["wall_clock_cap_seconds"])
                * (
                    1
                    if row["phase"] == "natural-label-materialization"
                    else len(PILOT_RETRY_ATTEMPTS)
                )
                / 3600.0
            )
            for row in capped_rows
        )
    )
    status = (
        "pilot-approved-repricing-required"
        if pilot_worst_case <= budget_hours
        else "pilot-budget-amendment-required"
    )
    source_manifest_path = (
        artifact_root
        / "adaptive_selector"
        / "sprint1"
        / "source_manifest.json"
    )
    natural_manifest_path = (
        artifact_root
        / "adaptive_selector"
        / "sprint1"
        / "natural_manifest.json"
    )
    precondition_artifacts = {}
    if source_manifest_path.is_file():
        source_manifest = _load_json(source_manifest_path)
        try:
            _validate_source_bundle_graph_provenance(
                source_manifest)
        except (FileNotFoundError, RuntimeError, ValueError):
            precondition_artifacts["source_manifest"] = {
                "path": str(source_manifest_path),
                "status": "stale",
            }
        else:
            for row in cache_micro_pilot_rows:
                if row.get("source_index") is None:
                    continue
                source = source_manifest["graphs"][
                    row["graph"]]["sources"][int(row["source_index"])]
                row["source_vertex_id"] = int(source["source_id"])
                row["expected_source_internal"] = int(
                    source["source_internal"])
                row["expected_source_out_degree"] = int(
                    source["source_out_degree"])
            precondition_artifacts["source_manifest"] = {
                "path": str(source_manifest_path),
                "sha256": file_sha256(
                    source_manifest_path, use_cache=False),
                "schema": source_manifest.get("schema"),
                "policy_id": source_manifest.get("policy_id"),
                "seed": source_manifest.get("seed"),
                "source_lists": source_manifest.get("source_lists"),
                "graph_provenance":
                    source_manifest.get("graph_provenance"),
                "graph_provenance_sha256":
                    source_manifest.get(
                        "graph_provenance_sha256"),
            }
    if natural_manifest_path.is_file():
        natural_manifest = _load_json(natural_manifest_path)
        try:
            _validate_natural_manifest_graph_provenance(
                natural_manifest)
        except (FileNotFoundError, RuntimeError, ValueError):
            precondition_artifacts["natural_manifest"] = {
                "path": str(natural_manifest_path),
                "status": "stale",
            }
        else:
            precondition_artifacts["natural_manifest"] = {
                "path": str(natural_manifest_path),
                "sha256": file_sha256(
                    natural_manifest_path, use_cache=False),
                "schema": natural_manifest.get("schema"),
                "policy_id": natural_manifest.get("policy_id"),
                "source_invariance":
                    natural_manifest.get("source_invariance"),
                "graphs": sorted(natural_manifest.get("graphs", {})),
                "excluded_graphs":
                    natural_manifest.get("excluded_graphs", {}),
                "graph_provenance":
                    natural_manifest.get("graph_provenance"),
                "graph_provenance_sha256":
                    natural_manifest.get(
                        "graph_provenance_sha256"),
            }
    return {
        "schema": SPRINT1_BUDGET_SCHEMA,
        "source_artifacts": {
            "exp2": _artifact_provenance(
                paper_root / "exp2_speedup" / "speedup_results.json",
                exp2,
            ),
            "exp3": _artifact_provenance(
                paper_root / "exp3_overhead" / "overhead_results.json",
                exp3,
            ),
            "cache": _artifact_provenance(
                paper_root / "exp1_cache" / "cache_results.json",
                cache,
            ),
        },
        "policy": {
            "budget_hours": budget_hours,
            "safety_factor": safety_factor,
            "process_repetitions": PROCESS_REPETITIONS,
            "warm_process_blocks": WARM_PROCESS_BLOCKS,
            "source_count": ADAPTIVE_SOURCE_COUNT,
            "large_graph_edge_threshold":
                LARGE_GRAPH_EDGE_THRESHOLD,
            "cache_capacity_candidates_mib":
                list(CACHE_CAPACITY_MIB),
            "cache_pr_iterations": CACHE_PR_ITERATIONS,
            "cache_cell_rule":
                "at-least-one-capacity-in-0.5x-to-2x-working-set",
            "cache_graphs": [graph["name"] for graph in EVAL_GRAPHS],
            "cache_kernels": list(CACHE_KERNELS),
            "pilot_process_blocks": PILOT_PROCESS_BLOCKS,
            "pilot_retry_attempts":
                list(PILOT_RETRY_ATTEMPTS),
            "characterization_baseline_arms":
                list(ALL_TIMING_ARMS),
            "natural_pilot_graphs": list(NATURAL_PILOT_GRAPHS),
            "natural_pilot_exclusions": NATURAL_PILOT_EXCLUSIONS,
            "native_ratio_low_floor": 0.25,
            "native_ratio_high_floor": 1.0,
            "h4_kernel_proxy": H4_KERNEL_PROXY,
            "h4_kernel_high_multiplier": H4_KERNEL_HIGH_MULTIPLIER,
            "hardware_repetitions": HARDWARE_REPETITIONS,
            "hardware_single_thread_factor":
                HARDWARE_SINGLE_THREAD_FACTOR,
            "cache_micro_wall_cap_multiplier":
                CACHE_MICRO_WALL_CAP_MULTIPLIER,
            "cache_setup_single_thread_factor":
                CACHE_SETUP_SINGLE_THREAD_FACTOR,
            "process_startup_allowance_seconds":
                PROCESS_STARTUP_ALLOWANCE_SECONDS,
            "pilot_default_wall_cap_multiplier":
                PILOT_DEFAULT_WALL_CAP_MULTIPLIER,
            "cache_repricing_model": "separable-star/v1",
            "cache_repricing_center":
                "hollywood-2009|pr|8:csr|22MiB",
            "cache_repricing_factors": {
                "graph": [
                    "twitter7|pr|8:csr|22MiB",
                    "webbase-2001|pr|8:csr|22MiB",
                ],
                "kernel": "hollywood-2009|bfs|8:csr|22MiB",
                "arm": [
                    "hollywood-2009|pr|0|22MiB",
                    "hollywood-2009|pr|5|22MiB",
                ],
                "capacity":
                    "hollywood-2009|pr|8:csr|512MiB",
                "interaction":
                    "twitter7|bfs|8:csr|22MiB|source-index-0",
                "source_dispersion":
                    "hollywood-2009|bfs|8:csr|22MiB|source-index-7",
            },
            "cache_repricing_max_interaction_residual": 0.25,
            "cache_censored_factor_policy":
                "lower-bound-only-no-full-authorization",
            "cache_unprobed_kernel_policy": {
                "cc": "amendment-gated",
                "sssp": "amendment-gated",
            },
        },
        "pilot_graphs": list(pilot_graphs),
        "medium_graph_rule": "median-undirected-edge-count",
        "kernel_rows": kernel_rows,
        "cache_rows": cache_rows,
        "randomized_pilot_rows": randomized_pilot_rows,
        "cache_micro_pilot_rows": cache_micro_pilot_rows,
        "feature_pilot_rows": feature_pilot_rows,
        "natural_pilot_rows": natural_pilot_rows,
        "materialization_rows": materialization_rows,
        "rss_rows": rss_rows,
        "hardware_rows": hardware_rows,
        "escalation_rows": escalation_rows,
        "excluded_cache_cells": excluded_cache_cells,
        "phase_buffered_node_hours_low": phase_hours_low,
        "phase_buffered_node_hours_high": phase_hours_high,
        "pilot_phase_buffered_node_hours_low":
            pilot_phase_hours_low,
        "pilot_phase_buffered_node_hours_high":
            pilot_phase_hours_high,
        "projected_buffered_node_hours_low": projected_hours_low,
        "projected_buffered_node_hours_high": projected_hours_high,
        "program_total_if_full_after_pilot_low":
            projected_hours_low
            + randomized_pilot_hours_low
            + cache_micro_hours_low
            + feature_hours_low,
        "program_total_if_full_after_pilot_high":
            projected_hours_high
            + randomized_pilot_hours_high
            + cache_micro_hours_high
            + feature_hours_high,
        "pilot_buffered_node_hours_high": pilot_high,
        "pilot_projection_if_all_defined_caps_bind":
            pilot_worst_case,
        "remaining_budget_hours_high":
            budget_hours - projected_hours_high,
        "status": status,
        "full_projection_status": full_status,
        "collection_allowed": {
            "three_graph_pilot": pilot_worst_case <= budget_hours,
            "cache_micro_pilot": pilot_worst_case <= budget_hours,
            "feature_cost_pilot": pilot_worst_case <= budget_hours,
            "hardware_validation": False,
            "randomized_kernel_corpus": False,
            "randomized_cache_corpus": False,
            "natural_label_extension": False,
        },
        "pilot_evidence_required_for_full_collection": True,
        "launch_preconditions": [
            "frozen-source-manifest-with-source-vertex-ids",
            "hollywood-natural-manifest-with-source-invariance-and-"
            "labeling-distinctness",
        ],
        "precondition_artifacts": precondition_artifacts,
        "authorized_phases": list(authorized_phases),
        "low_bound_role": "non-decision floor",
        "requires_repricing_after_pilot": [
            "randomized-cache",
            "source-precision-escalation",
            "hardware-validation",
        ],
        "limitations": [
            "Kernel and mapping costs are projections from final VLDB "
            "Exp2/Exp3 evidence; Sprint-1 measurements must be fresh.",
            "H4 kernel time uses Rabbit-HubSort as an optimistic low proxy; "
            "the high estimate doubles that proxy.",
            "Cache low estimates use the nearest measured graph rate; high "
            "estimates use the per-proxy maximum rate and conservative "
            "kernel-ratio floors.",
            "Only graph/kernel cells with a modeled working-set boundary "
            "inside the capacity ladder are eligible for H2.",
            "Existing Exp1 evidence varies by at most about 10% across "
            "2-64 MiB for a fixed graph/arm; the 512 MiB micro-probe tests "
            "whether that capacity-neutral throughput assumption extends.",
            "The pilot must replace cache and source-time bounds before any "
            "full-corpus collection.",
            "Natural-label pilot rows are additive and remain pilot-only.",
        ],
    }


def write_sprint1_budget_projection(
    projection: dict[str, Any],
    artifact_root: Path,
    *,
    refreeze: bool = False,
) -> tuple[Path, Path, Path]:
    out_dir = artifact_root / "adaptive_selector" / "sprint1"
    json_path = out_dir / "budget_projection.json"
    cell_path = out_dir / "budget_projection_cells.csv"
    summary_path = out_dir / "budget_projection_summary.csv"
    if json_path.is_file():
        existing = _load_json(json_path)
        freeze_fields = (
            "schema",
            "pilot_graphs",
            "authorized_phases",
            "collection_allowed",
            "launch_preconditions",
            "precondition_artifacts",
        )
        changed = any(
            existing.get(field) != projection.get(field)
            for field in freeze_fields
        )
        existing_rows = [
            {
                key: row.get(key)
                for key in (
                    "phase",
                    "graph",
                    "kernel",
                    "arm",
                    "labeling",
                    "measurement_mode",
                    "source_index",
                    "source_vertex_id",
                    "capacities_mib",
                    "wall_clock_cap_seconds",
                    "authorized_for_collection",
                )
            }
            for row in (
                existing.get("randomized_pilot_rows", [])
                + existing.get("natural_pilot_rows", [])
                + existing.get("materialization_rows", [])
                + existing.get("rss_rows", [])
                + existing.get("cache_micro_pilot_rows", [])
                + existing.get("feature_pilot_rows", [])
            )
        ]
        projected_rows = [
            {
                key: row.get(key)
                for key in (
                    "phase",
                    "graph",
                    "kernel",
                    "arm",
                    "labeling",
                    "measurement_mode",
                    "source_index",
                    "source_vertex_id",
                    "capacities_mib",
                    "wall_clock_cap_seconds",
                    "authorized_for_collection",
                )
            }
            for row in (
                projection["randomized_pilot_rows"]
                + projection["natural_pilot_rows"]
                + projection["materialization_rows"]
                + projection["rss_rows"]
                + projection["cache_micro_pilot_rows"]
                + projection["feature_pilot_rows"]
            )
        ]
        if (changed or existing_rows != projected_rows) and not refreeze:
            raise RuntimeError(
                "Frozen adaptive budget authorization changed; "
                "use --refreeze-budget after review")
    _atomic_json(projection, json_path)
    all_rows = (
        projection["kernel_rows"]
        + projection["cache_rows"]
        + projection["randomized_pilot_rows"]
        + projection["cache_micro_pilot_rows"]
        + projection["feature_pilot_rows"]
        + projection["natural_pilot_rows"]
        + projection["materialization_rows"]
        + projection["rss_rows"]
        + projection["hardware_rows"]
        + projection["escalation_rows"]
    )
    _atomic_csv(all_rows, cell_path)
    summary_rows = [
        {
            "scope": "full-projection",
            "phase": phase,
            "buffered_node_hours": hours,
        }
        for phase, hours in sorted(
            projection["phase_buffered_node_hours_high"].items())
    ]
    summary_rows.extend({
        "scope": "authorized-pilot",
        "phase": phase,
        "buffered_node_hours": hours,
    } for phase, hours in sorted(
        projection["pilot_phase_buffered_node_hours_high"].items()
    ))
    summary_rows.extend([{
        "scope": "full-projection",
        "phase": "TOTAL_HIGH",
        "buffered_node_hours":
            projection["projected_buffered_node_hours_high"],
    }, {
        "scope": "full-projection",
        "phase": "TOTAL_LOW",
        "buffered_node_hours":
            projection["projected_buffered_node_hours_low"],
    }, {
        "scope": "authorized-pilot",
        "phase": "PILOT_HIGH",
        "buffered_node_hours":
            projection["pilot_buffered_node_hours_high"],
    }, {
        "scope": "authorized-pilot",
        "phase": "PILOT_DEFINED_CAPS",
        "buffered_node_hours":
            projection[
                "pilot_projection_if_all_defined_caps_bind"],
    }, {
        "scope": "program",
        "phase": "PROGRAM_TOTAL_HIGH",
        "buffered_node_hours":
            projection["program_total_if_full_after_pilot_high"],
    }])
    _atomic_csv(summary_rows, summary_path)
    return json_path, cell_path, summary_path


def _validate_source_manifest(
    payload: dict[str, Any],
    graph: dict[str, Any],
    *,
    expected_graph_path: Path | None = None,
    graph_provenance: dict[str, Any] | None = None,
) -> None:
    if payload.get("schema") != "adaptive-source-manifest/v1":
        raise ValueError("Unsupported adaptive source manifest schema")
    if payload.get("policy_id") != ADAPTIVE_SOURCE_POLICY_ID:
        raise ValueError("Adaptive source manifest policy mismatch")
    if payload.get("seed") != ADAPTIVE_SOURCE_SEED:
        raise ValueError("Adaptive source manifest seed mismatch")
    if payload.get("source_count") != ADAPTIVE_SOURCE_COUNT:
        raise ValueError("Adaptive source manifest count mismatch")
    if payload.get("component_verification") != "pass":
        raise ValueError(
            "Adaptive source manifest lacks component verification")
    if payload.get("component_verifier") != "CCVerifier/v1":
        raise ValueError("Adaptive source component verifier mismatch")
    if (
        payload.get("candidate_order")
        != "seeded-cyclic-octile-scan/v1"
    ):
        raise ValueError("Adaptive source candidate order mismatch")
    labeling_features = payload.get("labeling_features")
    if not isinstance(labeling_features, dict):
        raise ValueError("Adaptive source labeling features are missing")
    for name in ("normalized_edge_span", "window_neighbor_overlap"):
        value = float(labeling_features.get(name, math.nan))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Adaptive source labeling feature is invalid: {name}")
    if labeling_features.get("sample_policy") != (
        "sqrt-clamped-1024-8192/v1"
    ):
        raise ValueError("Adaptive labeling sample policy mismatch")
    if int(labeling_features.get("sample_size", 0)) <= 0:
        raise ValueError("Adaptive labeling sample size is invalid")
    if payload.get("graph") != graph["name"]:
        raise ValueError("Adaptive source manifest graph mismatch")
    if payload.get("nodes") != graph["nodes"]:
        raise ValueError("Adaptive source manifest node mismatch")
    if payload.get("undirected_edges") != graph["undirected_edges"]:
        raise ValueError("Adaptive source manifest edge mismatch")
    if (
        expected_graph_path is not None
        and payload.get("graph_path") != str(expected_graph_path)
    ):
        raise ValueError("Adaptive source manifest graph path mismatch")
    if (
        graph_provenance is not None
        and payload.get("graph_provenance") != graph_provenance
    ):
        raise ValueError("Adaptive source graph provenance mismatch")
    if (
        float(payload.get("minimum_reachability_fraction", -1))
        != ADAPTIVE_SOURCE_MIN_REACHABILITY
    ):
        raise ValueError(
            "Adaptive source manifest reachability policy mismatch")

    sources = payload.get("sources")
    if not isinstance(sources, list) or len(sources) != ADAPTIVE_SOURCE_COUNT:
        raise ValueError("Adaptive source manifest must contain eight sources")
    largest_size = int(payload.get("largest_component_size", 0))
    if largest_size <= 0:
        raise ValueError("Adaptive source largest component is invalid")
    if int(payload.get("second_largest_component_size", -1)) < 0:
        raise ValueError(
            "Adaptive source second-largest component is missing")
    if int(payload.get("largest_component_min_original", -1)) < 0:
        raise ValueError(
            "Adaptive source component tie-break provenance is missing")
    original_ids = []
    internal_ids = []
    previous_degree = -1
    for index, source in enumerate(sources):
        if source.get("source_index") != index:
            raise ValueError("Adaptive source indices are not ordered")
        if source.get("requested_octile") != index:
            raise ValueError("Adaptive source requested octile changed")
        if not 0 <= int(source.get("realized_octile", -1)) < 8:
            raise ValueError("Adaptive source realized octile is invalid")
        if int(source.get("source_out_degree", 0)) <= 0:
            raise ValueError("Adaptive source is isolated")
        degree = int(source["source_out_degree"])
        if degree < previous_degree:
            raise ValueError("Adaptive source octile degrees are not monotone")
        previous_degree = degree
        expected_start = index * largest_size // ADAPTIVE_SOURCE_COUNT
        expected_end = (
            (index + 1) * largest_size // ADAPTIVE_SOURCE_COUNT
        )
        if (
            int(source.get("octile_start", -1)) != expected_start
            or int(source.get("octile_end", -1)) != expected_end
        ):
            raise ValueError("Adaptive source octile boundaries changed")
        rank = int(source.get("rank", -1))
        if not expected_start <= rank < expected_end:
            raise ValueError("Adaptive source rank is outside its octile")
        if (
            int(source.get("reachable_vertices", -1))
            != largest_size
        ):
            raise ValueError(
                "Adaptive source reachability does not match the LCC")
        if not source.get("replacement_path"):
            raise ValueError("Adaptive source replacement path is missing")
        if (
            float(source.get("reachable_fraction", 0.0))
            < ADAPTIVE_SOURCE_MIN_REACHABILITY
        ):
            raise ValueError("Adaptive source fails reachability threshold")
        expected_fraction = largest_size / int(payload["nodes"])
        if not math.isclose(
            float(source["reachable_fraction"]),
            expected_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Adaptive source reachable fraction is inconsistent")
        original_ids.append(int(source["source_id"]))
        internal_ids.append(int(source["source_internal"]))
    if len(set(original_ids)) != ADAPTIVE_SOURCE_COUNT:
        raise ValueError("Adaptive source manifest contains duplicates")
    if len(set(internal_ids)) != ADAPTIVE_SOURCE_COUNT:
        raise ValueError(
            "Adaptive source manifest contains duplicate internal IDs")


def _source_graph_provenance(
    graph_path: Path,
) -> dict[str, Any]:
    metadata = _load_json(
        graph_path.with_suffix(graph_path.suffix + ".meta.json"))
    required = (
        "schema",
        "reorder_semantics_version",
        "conversion_policy_id",
        "source_crc32",
        "source_mtime_ns",
        "source_bytes",
        "output_path",
        "output_bytes",
        "output_crc32",
        "converter_sha256",
        "conversion_repository_state",
        "expected_nodes",
        "expected_undirected_edges",
        "nodes",
        "directed_edges",
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(
            "Graph metadata lacks source provenance: "
            + " ".join(missing)
        )
    return {key: metadata[key] for key in required}


def _validate_source_bundle_graph_provenance(
    bundle: dict[str, Any],
) -> None:
    graph_provenance = bundle.get("graph_provenance")
    if (
        not isinstance(graph_provenance, dict)
        or bundle.get("graph_provenance_sha256")
        != _canonical_json_sha256(graph_provenance)
    ):
        raise ValueError(
            "Adaptive source bundle lacks graph provenance binding")
    graphs = bundle.get("graphs", {})
    if set(graph_provenance) != set(graphs):
        raise ValueError(
            "Adaptive source bundle graph provenance coverage changed")
    for graph_name, payload in graphs.items():
        graph_path = Path(str(payload.get("graph_path", "")))
        current = _source_graph_provenance(graph_path)
        if (
            graph_provenance.get(graph_name) != current
            or payload.get("graph_provenance") != current
        ):
            raise ValueError(
                f"Adaptive source bundle is stale: {graph_name}")


def _validate_natural_manifest_graph_provenance(
    manifest: dict[str, Any],
) -> None:
    graph_provenance = manifest.get("graph_provenance")
    if (
        not isinstance(graph_provenance, dict)
        or manifest.get("graph_provenance_sha256")
        != _canonical_json_sha256(graph_provenance)
    ):
        raise ValueError(
            "Natural manifest lacks graph provenance binding")
    graphs = manifest.get("graphs", {})
    if set(graph_provenance) != set(graphs):
        raise ValueError(
            "Natural manifest graph provenance coverage changed")
    for graph_name, record in graphs.items():
        graph_path = Path(str(record.get("natural_graph", "")))
        current = _source_graph_provenance(graph_path)
        if (
            graph_provenance.get(graph_name) != current
            or record.get("graph_provenance") != current
        ):
            raise ValueError(
                f"Natural manifest is stale: {graph_name}")


def generate_sprint1_source_manifests(
    artifact_root: Path,
    graph_root: Path,
    *,
    threads: int,
    cpu_list: str | None = None,
    force: bool = False,
    refreeze: bool = False,
) -> Path:
    """Generate and freeze source manifests for the three pilot graphs."""
    if threads <= 0:
        raise ValueError("Source generation threads must be positive")
    budget_path = (
        artifact_root
        / "adaptive_selector"
        / "sprint1"
        / "budget_projection.json"
    )
    budget = _load_json(budget_path)
    if budget.get("schema") != SPRINT1_BUDGET_SCHEMA:
        raise ValueError("Adaptive source generation requires budget v1")
    if not budget.get("collection_allowed", {}).get("three_graph_pilot"):
        raise ValueError("Adaptive budget does not authorize pilot preparation")
    if not budget.get("pilot_evidence_required_for_full_collection"):
        raise ValueError("Adaptive budget lost pilot-evidence gate")

    binary = PROJECT_ROOT / "bench" / "bin" / "cc"
    subprocess.run(
        [
            "make",
            f"-j{min(threads, 4)}",
            "bench/bin/cc",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    if not binary.is_file():
        raise FileNotFoundError(f"CC binary not found: {binary}")

    graph_by_name = {
        str(graph["name"]): graph for graph in EVAL_GRAPHS
    }
    out_dir = (
        artifact_root / "adaptive_selector" / "sprint1" / "sources")
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_manifests = {}
    commands = {}
    for graph_name in budget["pilot_graphs"]:
        graph = graph_by_name[graph_name]
        graph_path = (
            graph_root / graph_name / f"{graph_name}.sg")
        graph_provenance = _source_graph_provenance(graph_path)
        output_path = out_dir / f"{graph_name}.json"
        log_path = out_dir / f"{graph_name}.log"
        base_command = [
            str(binary),
            "-f",
            str(graph_path),
            "-Y",
            str(output_path),
        ]
        command = (
            ["taskset", "-c", cpu_list, *base_command]
            if cpu_list else base_command
        )
        if output_path.is_file() and not force:
            payload = _load_json(output_path)
            if (
                payload.get("graph_provenance") != graph_provenance
                or payload.get("generator_command") != command
                or payload.get("omp_num_threads") != threads
                or payload.get("cpu_list") != cpu_list
            ):
                raise RuntimeError(
                    "Existing adaptive source manifest is stale; "
                    "regenerate with --force-sources and "
                    "--refreeze-sources"
                )
            _validate_source_manifest(
                payload,
                graph,
                expected_graph_path=graph_path,
                graph_provenance=graph_provenance,
            )
            graph_manifests[graph_name] = payload
            commands[graph_name] = payload["generator_command"]
            continue

        environment = {
            **os.environ,
            "OMP_NUM_THREADS": str(threads),
            "GRAPHBREW_DB_DIR": "",
            "GRAPHBREW_TOPOLOGY_ANALYSIS": "0",
        }
        with open(log_path, "w") as log:
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=12 * 60 * 60,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"Adaptive source generation failed for {graph_name}; "
                f"see {log_path}"
            )
        payload = _load_json(output_path)
        payload["generator_command"] = command
        payload["omp_num_threads"] = threads
        payload["cpu_list"] = cpu_list
        payload["graph_provenance"] = graph_provenance
        _atomic_json(payload, output_path)
        _validate_source_manifest(
            payload,
            graph,
            expected_graph_path=graph_path,
            graph_provenance=graph_provenance,
        )
        graph_manifests[graph_name] = payload
        commands[graph_name] = command

    bundle = {
        "schema": "adaptive-source-bundle/v1",
        "policy_id": ADAPTIVE_SOURCE_POLICY_ID,
        "seed": ADAPTIVE_SOURCE_SEED,
        "source_count": ADAPTIVE_SOURCE_COUNT,
        "minimum_reachability_fraction":
            ADAPTIVE_SOURCE_MIN_REACHABILITY,
        "budget_projection": str(budget_path),
        "pilot_graphs": list(budget["pilot_graphs"]),
        "threads": threads,
        "cpu_list": cpu_list,
        "commands": commands,
        "graphs": graph_manifests,
        "graph_provenance": {
            graph_name: payload["graph_provenance"]
            for graph_name, payload in graph_manifests.items()
        },
        "source_lists": {
            graph_name: [
                int(source["source_id"])
                for source in payload["sources"]
            ]
            for graph_name, payload in graph_manifests.items()
        },
    }
    bundle["graph_provenance_sha256"] = _canonical_json_sha256(
        bundle["graph_provenance"]
    )
    bundle_path = (
        artifact_root
        / "adaptive_selector"
        / "sprint1"
        / "source_manifest.json"
    )
    if bundle_path.is_file():
        existing = _load_json(bundle_path)
        frozen_fields = (
            "policy_id",
            "seed",
            "source_lists",
            "graph_provenance",
            "graph_provenance_sha256",
        )
        changed = any(
            existing.get(field) != bundle.get(field)
            for field in frozen_fields
        )
        if changed and not refreeze:
            raise RuntimeError(
                "Frozen adaptive source bundle changed; "
                "use --refreeze-sources after review")
    _atomic_json(bundle, bundle_path)
    return bundle_path


def materialize_sprint1_natural_graphs(
    artifact_root: Path,
    graph_root: Path,
    *,
    threads: int,
    cpu_list: str | None = None,
    force: bool = False,
) -> Path:
    """Materialize natural-label pilot graphs and verify source invariance."""
    source_bundle_path = (
        artifact_root
        / "adaptive_selector"
        / "sprint1"
        / "source_manifest.json"
    )
    source_bundle = _load_json(source_bundle_path)
    if source_bundle.get("schema") != "adaptive-source-bundle/v1":
        raise ValueError("Natural materialization requires source bundle v1")
    if source_bundle.get("policy_id") != ADAPTIVE_SOURCE_POLICY_ID:
        raise ValueError("Natural materialization source policy mismatch")

    converter = PROJECT_ROOT / "bench" / "bin" / "converter"
    cc_binary = PROJECT_ROOT / "bench" / "bin" / "cc"
    subprocess.run(
        [
            "make",
            f"-j{min(threads, 4)}",
            "bench/bin/converter",
            "bench/bin/cc",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    if not converter.is_file() or not cc_binary.is_file():
        raise FileNotFoundError("Natural materialization binaries are missing")

    expected_output_bytes = sum(
        int(source_bundle["graphs"][name]["graph_provenance"]["output_bytes"])
        for name in NATURAL_PILOT_GRAPHS
    )
    free_bytes = shutil.disk_usage(graph_root).free
    if free_bytes < int(1.2 * expected_output_bytes):
        raise RuntimeError(
            "Insufficient external storage for natural pilot graphs")

    out_dir = (
        artifact_root / "adaptive_selector" / "sprint1" / "natural")
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_by_name = {
        str(graph["name"]): graph for graph in EVAL_GRAPHS
    }
    records = {}
    for graph_name in NATURAL_PILOT_GRAPHS:
        graph = graph_by_name[graph_name]
        canonical_graph_path = (
            graph_root / graph_name / f"{graph_name}.sg")
        canonical_meta = _current_graph_metadata(
            canonical_graph_path,
            graph_name=graph_name,
            natural=False,
            verify_content=True,
            expected_nodes=int(graph["nodes"]),
            expected_undirected_edges=
                int(graph["undirected_edges"]),
        )
        source_path = Path(str(canonical_meta["source_path"]))
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Natural source graph is missing: {source_path}")
        natural_path = (
            graph_root / graph_name / f"{graph_name}.natural.sg")
        natural_meta_path = natural_path.with_suffix(
            natural_path.suffix + ".meta.json")
        converter_log = out_dir / f"{graph_name}.converter.log"
        source_log = out_dir / f"{graph_name}.sources.log"
        natural_source_path = out_dir / f"{graph_name}.sources.json"

        converter_command = [
            str(converter),
            "-f",
            str(source_path),
            "-s",
            "-o",
            "0",
            "-b",
            str(natural_path),
        ]
        if cpu_list:
            converter_command = [
                "taskset", "-c", cpu_list, *converter_command]
        if force or not natural_path.is_file():
            environment = {
                **os.environ,
                "OMP_NUM_THREADS": str(threads),
                "GRAPHBREW_DB_DIR": "",
                "GRAPHBREW_TOPOLOGY_ANALYSIS": "0",
            }
            with open(converter_log, "w") as log:
                result = subprocess.run(
                    converter_command,
                    cwd=PROJECT_ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=12 * 60 * 60,
                )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Natural conversion failed for {graph_name}; "
                    f"see {converter_log}"
                )
            natural_metadata = {
                "schema": "adaptive-natural-graph/v2",
                "reorder_semantics_version":
                    REORDER_SEMANTICS_VERSION,
                "conversion_policy_id": "adaptive-natural-original/v1",
                "graph": graph_name,
                "source_path": str(source_path.resolve()),
                "source_bytes": source_path.stat().st_size,
                "source_mtime_ns": source_path.stat().st_mtime_ns,
                "source_crc32": canonical_meta["source_crc32"],
                "output_path": str(natural_path.resolve()),
                "output_bytes": natural_path.stat().st_size,
                "output_crc32": file_crc32(
                    natural_path, use_cache=False),
                "nodes": graph["nodes"],
                "directed_edges": 2 * graph["undirected_edges"],
                "undirected_edges": graph["undirected_edges"],
                "expected_nodes": graph["nodes"],
                "expected_undirected_edges":
                    graph["undirected_edges"],
                "symmetrized": True,
                "ordering": "0",
                "omp_num_threads": threads,
                "cpu_list": cpu_list,
                "converter_args": converter_command,
                "converter_sha256": file_sha256(converter),
                "conversion_repository_state":
                    _conversion_repository_state(),
            }
            _atomic_json(natural_metadata, natural_meta_path)
        else:
            natural_metadata = _load_json(natural_meta_path)
            required = {
                "schema": "adaptive-natural-graph/v2",
                "reorder_semantics_version":
                    REORDER_SEMANTICS_VERSION,
                "conversion_policy_id":
                    "adaptive-natural-original/v1",
                "graph": graph_name,
                "source_path": str(source_path.resolve()),
                "source_bytes": source_path.stat().st_size,
                "source_mtime_ns": source_path.stat().st_mtime_ns,
                "source_crc32": canonical_meta["source_crc32"],
                "output_path": str(natural_path.resolve()),
                "nodes": graph["nodes"],
                "undirected_edges": graph["undirected_edges"],
                "expected_nodes": graph["nodes"],
                "expected_undirected_edges":
                    graph["undirected_edges"],
                "ordering": "0",
                "converter_sha256": file_sha256(converter),
                "conversion_repository_state":
                    _conversion_repository_state(),
            }
            for key, value in required.items():
                if natural_metadata.get(key) != value:
                    raise ValueError(
                        f"Natural graph metadata mismatch: {graph_name}/{key}")
            if natural_metadata.get("output_bytes") != natural_path.stat().st_size:
                raise ValueError(
                    f"Natural graph size changed: {graph_name}")
            if (
                natural_metadata.get("output_crc32")
                != file_crc32(natural_path)
            ):
                raise ValueError(
                    f"Natural graph content changed: {graph_name}")

        source_command = [
            str(cc_binary),
            "-f",
            str(natural_path),
            "-Y",
            str(natural_source_path),
        ]
        if cpu_list:
            source_command = ["taskset", "-c", cpu_list, *source_command]
        generated_natural_sources = (
            force or not natural_source_path.is_file()
        )
        if generated_natural_sources:
            environment = {
                **os.environ,
                "OMP_NUM_THREADS": str(threads),
                "GRAPHBREW_DB_DIR": "",
                "GRAPHBREW_TOPOLOGY_ANALYSIS": "0",
            }
            with open(source_log, "w") as log:
                result = subprocess.run(
                    source_command,
                    cwd=PROJECT_ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=12 * 60 * 60,
                )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Natural source validation failed for {graph_name}; "
                    f"see {source_log}"
                )
        natural_sources = _load_json(natural_source_path)
        natural_graph_provenance = _source_graph_provenance(
            natural_path)
        expected_source_contract = {
            "generator_command": source_command,
            "omp_num_threads": threads,
            "cpu_list": cpu_list,
            "graph_provenance": natural_graph_provenance,
        }
        if generated_natural_sources:
            natural_sources.update(expected_source_contract)
            _atomic_json(natural_sources, natural_source_path)
        elif any(
            natural_sources.get(key) != value
            for key, value in expected_source_contract.items()
        ):
            raise RuntimeError(
                "Existing natural source manifest is stale; "
                "regenerate with --force-sources"
            )
        _validate_source_manifest(
            natural_sources,
            graph,
            expected_graph_path=natural_path,
            graph_provenance=natural_graph_provenance,
        )

        frozen_sources = source_bundle["graphs"][graph_name]["sources"]
        if len(frozen_sources) != len(natural_sources["sources"]):
            raise ValueError(
                f"Natural/randomized source count mismatch: {graph_name}")
        comparable_fields = (
            "source_id",
            "source_out_degree",
            "requested_octile",
            "realized_octile",
            "rank",
            "octile_start",
            "octile_end",
        )
        for frozen, natural in zip(frozen_sources, natural_sources["sources"]):
            for field in comparable_fields:
                if frozen[field] != natural[field]:
                    raise ValueError(
                        "Natural/randomized source invariance failed for "
                        f"{graph_name}/{field}")
        randomized_features = source_bundle["graphs"][graph_name][
            "labeling_features"]
        natural_features = natural_sources["labeling_features"]
        feature_differences = {
            name: abs(
                float(randomized_features[name])
                - float(natural_features[name])
            )
            for name in (
                "normalized_edge_span",
                "window_neighbor_overlap",
            )
        }
        if not (
            float(natural_features["window_neighbor_overlap"]) >= 0.01
            and float(natural_features["normalized_edge_span"]) <= 0.2
            and float(natural_features["window_neighbor_overlap"])
                >= float(
                    randomized_features["window_neighbor_overlap"])
                + 0.01
        ):
            raise ValueError(
                f"Natural labeling is not measurably distinct: {graph_name}")
        records[graph_name] = {
            "natural_graph": str(natural_path),
            "natural_metadata": str(natural_meta_path),
            "natural_source_manifest": str(natural_source_path),
            "graph_provenance": natural_graph_provenance,
            "source_ids": [
                int(source["source_id"])
                for source in natural_sources["sources"]
            ],
            "source_out_degrees": [
                int(source["source_out_degree"])
                for source in natural_sources["sources"]
            ],
            "source_invariance": "pass",
            "randomized_labeling_features": randomized_features,
            "natural_labeling_features": natural_features,
            "labeling_feature_absolute_differences":
                feature_differences,
            "labeling_distinctness": "pass",
        }

    manifest = {
        "schema": "adaptive-natural-pilot/v1",
        "source_bundle": str(source_bundle_path),
        "policy_id": ADAPTIVE_SOURCE_POLICY_ID,
        "threads": threads,
        "cpu_list": cpu_list,
        "graphs": records,
        "graph_provenance": {
            graph_name: record["graph_provenance"]
            for graph_name, record in records.items()
        },
        "excluded_graphs": NATURAL_PILOT_EXCLUSIONS,
        "exclusion_evidence": NATURAL_PILOT_EXCLUSION_EVIDENCE,
        "source_invariance": "pass",
    }
    manifest["graph_provenance_sha256"] = _canonical_json_sha256(
        manifest["graph_provenance"]
    )
    manifest_path = (
        artifact_root
        / "adaptive_selector"
        / "sprint1"
        / "natural_manifest.json"
    )
    _atomic_json(manifest, manifest_path)
    return manifest_path


def _rotate(values: list[int], offset: int) -> list[int]:
    if not values:
        return []
    offset %= len(values)
    return values[offset:] + values[:offset]


def _taskset_command(command: list[str], cpu_list: str | None) -> list[str]:
    return (
        ["taskset", "-c", cpu_list, *command]
        if cpu_list else command
    )


def _command_binary_provenance(
    command: list[str],
) -> list[dict[str, Any]]:
    binaries = []
    seen = set()
    for token in command:
        candidate = Path(token)
        is_repo_binary = token.startswith(
            str(PROJECT_ROOT / "bench" / "bin"))
        is_known_tool = candidate.name in {
            "taskset", "dd", "time", "python3",
        }
        if not is_repo_binary and not is_known_tool:
            continue
        if not candidate.is_absolute():
            resolved = shutil.which(token)
            candidate = Path(resolved) if resolved else candidate
        if (
            candidate.is_file()
            and os.access(candidate, os.X_OK)
            and candidate not in seen
        ):
            seen.add(candidate)
            stat = candidate.stat()
            binaries.append({
                "path": str(candidate),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": file_sha256(candidate),
            })
    if not binaries:
        raise ValueError("Pilot command has no executable binary")
    return binaries


def _validate_command_binaries(commands: list[dict[str, Any]]) -> None:
    dependency_paths = [PROJECT_ROOT / "Makefile"]
    for pattern in ("*.h", "*.hpp", "*.hxx", "*.def"):
        dependency_paths.extend(
            (PROJECT_ROOT / "bench" / "include").rglob(pattern))
    newest_header = max(
        path.stat().st_mtime_ns
        for path in dependency_paths
        if path.is_file()
    )
    binaries = {
        Path(token)
        for record in commands
        for token in record["command"]
        if token.startswith(str(PROJECT_ROOT / "bench" / "bin"))
    }
    for binary in binaries:
        if not binary.is_file():
            raise FileNotFoundError(
                f"Pilot binary is missing: {binary}")
        relative = binary.relative_to(PROJECT_ROOT / "bench")
        source_dir = (
            PROJECT_ROOT / "bench" / "src_sim"
            if relative.parts[0] == "bin_sim"
            else PROJECT_ROOT / "bench" / "src"
        )
        source = source_dir / f"{binary.name}.cc"
        newest_dependency = max(
            newest_header,
            source.stat().st_mtime_ns if source.is_file() else 0,
        )
        if binary.stat().st_mtime_ns < newest_dependency:
            raise RuntimeError(
                f"Pilot binary is stale: {binary}; rebuild its Make target")


def _validate_binary_provenance(command: dict[str, Any]) -> None:
    for binary in command["binary_provenance"]:
        path = Path(binary["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(binary["bytes"])
            or path.stat().st_mtime_ns != int(binary["mtime_ns"])
            or file_sha256(path) != binary.get("sha256")
        ):
            raise RuntimeError(
                f"Pilot binary changed after freeze: {path}")


def _kernel_policy_args(
    benchmark: str,
    graph_name: str,
) -> list[str]:
    if benchmark in {"pr", "pr_spmv"}:
        return [
            "-F",
            "-i", str(PR_FIXED_ITERATIONS),
            "-t", str(PR_TOLERANCE),
        ]
    if benchmark == "bc":
        return ["-i", str(BC_SOURCE_ITERATIONS)]
    if benchmark == "sssp":
        policy = SSSP_POLICY.get(graph_name)
        if not isinstance(policy, dict):
            raise ValueError(f"SSSP policy is missing for {graph_name}")
        return [
            "-W", str(policy["weight_scheme"]),
            "-d", str(policy["delta"]),
        ]
    return []


def _contract_weight_payload() -> dict[str, Any]:
    entry = {
        "bias": 0.0,
        **{
            f"w_t0_{name}": 0.0
            for name in TIER0_FEATURE_NAMES
        },
    }
    return {
        "_schema": "adaptive-tier0/v1",
        "weights": {
            arm: dict(entry)
            for arm in DEPLOYABLE_ARM_SPECS
        },
    }


def _authorized_pilot_budget_rows(
    budget: dict[str, Any],
) -> list[dict[str, Any]]:
    return (
        budget["randomized_pilot_rows"]
        + budget["natural_pilot_rows"]
        + budget["materialization_rows"]
        + budget["rss_rows"]
        + budget["cache_micro_pilot_rows"]
        + budget["feature_pilot_rows"]
    )


def _expected_pilot_command_ids(
    budget: dict[str, Any],
) -> tuple[dict[str, int], set[str]]:
    expected: dict[str, int] = {}
    rows = _authorized_pilot_budget_rows(budget)
    for row_index, row in enumerate(rows):
        phase = str(row["phase"])
        graph = str(row["graph"])
        if phase == "natural-label-materialization":
            continue
        if phase in {"randomized-pilot", "natural-label-pilot"}:
            for process_id in range(int(row["process_blocks"])):
                command_id = (
                    f"{phase}|{graph}|{row['kernel']}|"
                    f"{row['arm']}|p{process_id}"
                )
                if command_id in expected:
                    raise ValueError(
                        f"Duplicate expected pilot command: {command_id}")
                expected[command_id] = row_index
            continue
        if phase == "cache-micro-pilot":
            command_id = (
                f"{phase}|{graph}|{row['kernel']}|{row['arm']}|"
                f"{row['capacities_mib']}MiB|"
                f"s{row.get('source_index')}"
            )
        elif phase == "feature-cost-pilot":
            command_id = (
                f"{phase}|{graph}|{row.get('labeling')}"
            )
        elif phase == "rss-pilot":
            command_id = f"{phase}|{graph}|{row['arm']}"
        else:
            raise ValueError(
                f"Unsupported authorized pilot phase: {phase}")
        if command_id in expected:
            raise ValueError(
                f"Duplicate expected pilot command: {command_id}")
        expected[command_id] = row_index
    priming = {
        f"page-cache-prime|{graph_name}|randomized"
        for graph_name in budget["pilot_graphs"]
    } | {
        f"page-cache-prime|{graph_name}|natural"
        for graph_name in NATURAL_PILOT_GRAPHS
    }
    return expected, priming


def _verify_budget_preconditions(
    budget: dict[str, Any],
    source_bundle: dict[str, Any],
    natural_manifest: dict[str, Any],
) -> None:
    if budget.get("schema") != SPRINT1_BUDGET_SCHEMA:
        raise ValueError("Pilot launcher requires budget schema v1")
    if budget.get("policy", {}).get(
        "characterization_baseline_arms"
    ) != list(ALL_TIMING_ARMS):
        raise ValueError(
            "Pilot characterization arm contract changed")
    if budget.get("policy", {}).get(
        "pilot_retry_attempts"
    ) != list(PILOT_RETRY_ATTEMPTS):
        raise ValueError("Pilot retry budget contract changed")
    if not budget.get("pilot_evidence_required_for_full_collection"):
        raise ValueError("Pilot launcher lost full-collection gate")
    allowed = budget.get("collection_allowed", {})
    if not all((
        allowed.get("three_graph_pilot"),
        allowed.get("cache_micro_pilot"),
        allowed.get("feature_cost_pilot"),
    )):
        raise ValueError("Pilot budget does not authorize all pilot phases")
    if any((
        allowed.get("hardware_validation"),
        allowed.get("randomized_kernel_corpus"),
        allowed.get("randomized_cache_corpus"),
        allowed.get("natural_label_extension"),
    )):
        raise ValueError("Pilot budget unexpectedly authorizes corpus work")

    bindings = budget.get("precondition_artifacts", {})
    _validate_source_bundle_graph_provenance(source_bundle)
    _validate_natural_manifest_graph_provenance(natural_manifest)
    source_binding = bindings.get("source_manifest", {})
    if (
        source_binding.get("sha256")
            != file_sha256(
                Path(source_binding.get("path", "")),
                use_cache=False,
            )
        or source_binding.get("schema") != source_bundle.get("schema")
        or source_binding.get("policy_id") != source_bundle.get("policy_id")
        or source_binding.get("seed") != source_bundle.get("seed")
        or source_binding.get("source_lists")
            != source_bundle.get("source_lists")
        or source_binding.get("graph_provenance")
            != source_bundle.get("graph_provenance")
        or source_binding.get("graph_provenance_sha256")
            != source_bundle.get("graph_provenance_sha256")
    ):
        raise ValueError("Budget/source manifest binding mismatch")
    natural_binding = bindings.get("natural_manifest", {})
    if (
        natural_binding.get("sha256")
            != file_sha256(
                Path(natural_binding.get("path", "")),
                use_cache=False,
            )
        or natural_binding.get("schema")
            != natural_manifest.get("schema")
        or natural_binding.get("policy_id")
            != natural_manifest.get("policy_id")
        or natural_binding.get("source_invariance")
            != natural_manifest.get("source_invariance")
        or natural_binding.get("graphs")
            != sorted(natural_manifest.get("graphs", {}))
        or natural_binding.get("excluded_graphs")
            != natural_manifest.get("excluded_graphs")
        or natural_binding.get("graph_provenance")
            != natural_manifest.get("graph_provenance")
        or natural_binding.get("graph_provenance_sha256")
            != natural_manifest.get("graph_provenance_sha256")
    ):
        raise ValueError("Budget/natural manifest binding mismatch")


def prepare_sprint1_pilot_execution(
    artifact_root: Path,
    graph_root: Path,
    *,
    threads: int,
    cpu_list: str | None,
    refreeze: bool = False,
) -> Path:
    """Expand authorized budget rows into a dry-run-only execution manifest."""
    sprint_root = artifact_root / "adaptive_selector" / "sprint1"
    budget_path = sprint_root / "budget_projection.json"
    source_path = sprint_root / "source_manifest.json"
    natural_path = sprint_root / "natural_manifest.json"
    budget = _load_json(budget_path)
    source_bundle = _load_json(source_path)
    natural_manifest = _load_json(natural_path)
    _verify_budget_preconditions(
        budget, source_bundle, natural_manifest)

    contract_path = sprint_root / "tier0_contract_weights.json"
    contract_payload = _contract_weight_payload()
    if contract_path.is_file():
        if _load_json(contract_path) != contract_payload:
            raise RuntimeError(
                "Tier-0 contract weights changed unexpectedly")
    else:
        _atomic_json(contract_payload, contract_path)
    output_root = sprint_root / "pilot_runs"
    output_root.mkdir(parents=True, exist_ok=True)

    authorized_phases = set(budget["authorized_phases"])
    all_budget_rows = _authorized_pilot_budget_rows(budget)
    expected_group_sizes = {
        "randomized_pilot_rows":
            len(budget["pilot_graphs"])
            * len(ALL_TIMING_ARMS)
            * len(BENCHMARKS),
        "natural_pilot_rows":
            len(NATURAL_PILOT_GRAPHS)
            * len(ALL_TIMING_ARMS)
            * len(BENCHMARKS),
        "materialization_rows": len(NATURAL_PILOT_GRAPHS),
        "rss_rows": 2 * len(ALL_TIMING_ARMS),
        "cache_micro_pilot_rows": 9,
        "feature_pilot_rows":
            len(budget["pilot_graphs"]) + len(NATURAL_PILOT_GRAPHS),
    }
    for key, expected in expected_group_sizes.items():
        if len(budget[key]) != expected:
            raise ValueError(
                f"Pilot authorization {key} changed: "
                f"expected {expected}, got {len(budget[key])}"
            )
    if any(
        not row.get("authorized_for_collection")
        or row.get("phase") not in authorized_phases
        or "wall_clock_cap_seconds" not in row
        for row in all_budget_rows
    ):
        raise ValueError("Pilot contains an unauthorized or uncapped row")

    commands = []
    completed_preparation = []
    base_environment = {
        "GRAPHBREW_DB_DIR": "",
        "GRAPHBREW_TOPOLOGY_ANALYSIS": "0",
        "OMP_NUM_THREADS": str(threads),
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
        "OMP_DYNAMIC": "FALSE",
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
        "CACHE_FAST": "0",
        "CACHE_ULTRAFAST": "0",
        "CACHE_SAMPLED": "0",
        "CACHE_MULTICORE": "0",
    }
    source_graphs = source_bundle["graphs"]
    natural_graphs = natural_manifest["graphs"]

    for row_index, row in enumerate(all_budget_rows):
        phase = str(row["phase"])
        graph_name = str(row["graph"])
        labeling = row.get("labeling")
        if phase == "natural-label-materialization":
            record = natural_graphs.get(graph_name)
            if not record or record.get("source_invariance") != "pass":
                raise ValueError(
                    f"Natural preparation is incomplete: {graph_name}")
            completed_preparation.append({
                "budget_row_index": row_index,
                "phase": phase,
                "graph": graph_name,
                "state": "completed",
                "manifest": str(natural_path),
            })
            continue

        if labeling == "natural":
            natural_record = natural_graphs[graph_name]
            graph_path = Path(natural_record["natural_graph"])
            source_manifest = _load_json(
                Path(natural_record[
                    "natural_source_manifest"]))
            if (
                [
                    int(source["source_id"])
                    for source in source_manifest["sources"]
                ] != natural_record["source_ids"]
                or [
                    int(source["source_out_degree"])
                    for source in source_manifest["sources"]
                ] != natural_record["source_out_degrees"]
            ):
                raise ValueError(
                    f"Natural source binding changed: {graph_name}")
        else:
            graph_path = (
                graph_root / graph_name / f"{graph_name}.sg")
            source_manifest = source_graphs[graph_name]
        if source_manifest.get("graph_path") != str(graph_path):
            raise ValueError(
                f"Source manifest graph path changed: {graph_name}")
        if row.get("graph_path") not in {None, str(graph_path)}:
            raise ValueError(
                f"Budget graph path changed: {phase}/{graph_name}")
        if not graph_path.is_file():
            raise FileNotFoundError(f"Pilot graph is missing: {graph_path}")
        graph_metadata = _current_graph_metadata(
            graph_path,
            graph_name=graph_name,
            natural=labeling == "natural",
            verify_content=True,
            expected_nodes=int(
                next(
                    graph["nodes"] for graph in EVAL_GRAPHS
                    if graph["name"] == graph_name
                )
            ),
            expected_undirected_edges=int(
                next(
                    graph["undirected_edges"]
                    for graph in EVAL_GRAPHS
                    if graph["name"] == graph_name
                )
            ),
        )

        common = {
            "budget_row_index": row_index,
            "phase": phase,
            "graph": graph_name,
            "graph_path": str(graph_path),
            "graph_output_bytes": graph_path.stat().st_size,
            "graph_mtime_ns": graph_path.stat().st_mtime_ns,
            "graph_crc32": graph_metadata["output_crc32"],
            "labeling": labeling,
            "arm": row.get("arm"),
            "kernel": row.get("kernel"),
            "claim_eligible": False,
            "pilot_only": True,
            "wall_clock_cap_seconds":
                float(row["wall_clock_cap_seconds"]),
            "timeout_interpretation":
                row["timeout_interpretation"],
            "result_graph_name": graph_name,
            "order_spec": (
                "14" if phase == "feature-cost-pilot"
                else str(row.get("arm"))
            ),
            "cap_floor_raw_seconds":
                row.get("cap_floor_raw_seconds"),
            "cpu_list": cpu_list,
            "host_state_requirements": {
                "cpu_governor": "performance",
                "turbo": "disabled",
                "transparent_hugepage": "madvise",
            },
        }

        if phase in {"randomized-pilot", "natural-label-pilot"}:
            process_blocks = int(row["process_blocks"])
            for process_id in range(process_blocks):
                benchmark = str(row["kernel"])
                command = [
                    str(PROJECT_ROOT / "bench" / "bin" / benchmark),
                    "-f", str(graph_path),
                    "-s",
                    "-n", (
                        str(ADAPTIVE_SOURCE_COUNT)
                        if benchmark in SOURCE_DRIVEN_KERNELS
                        else "1"
                    ),
                    *_kernel_policy_args(benchmark, graph_name),
                    "-o", str(row["arm"]),
                ]
                expected_sources = []
                expected_internals = []
                expected_degrees = []
                if benchmark in SOURCE_DRIVEN_KERNELS:
                    records = source_manifest["sources"]
                    expected_sources = _rotate(
                        [int(source["source_id"]) for source in records],
                        process_id,
                    )
                    if str(row["arm"]) == "0":
                        expected_internals = _rotate(
                            [
                                int(source["source_internal"])
                                for source in records
                            ],
                            process_id,
                        )
                    expected_degrees = _rotate(
                        [int(source["source_out_degree"]) for source in records],
                        process_id,
                    )
                    command.extend([
                        "-r", ",".join(map(str, expected_sources)),
                    ])
                cap = math.ceil(
                    float(row["wall_clock_cap_seconds"])
                    / process_blocks
                )
                commands.append({
                    **common,
                    "command_id":
                        f"{phase}|{graph_name}|{row['kernel']}|"
                        f"{row['arm']}|p{process_id}",
                    "process_id": process_id,
                    "measurement_mode": "cold-process",
                    "source_policy_id": (
                        ADAPTIVE_SOURCE_POLICY_ID
                        if expected_sources else None
                    ),
                    "source_repeats": 1,
                    "expected_sources": expected_sources or None,
                    "expected_source_internals":
                        expected_internals or None,
                    "expected_source_out_degrees":
                        expected_degrees or None,
                    "command": _taskset_command(command, cpu_list),
                    "environment": base_environment,
                    "timeout_seconds": cap,
                    "cap_floor_raw_seconds":
                        (
                            float(row["cap_floor_raw_seconds"])
                            / process_blocks
                            if row.get("cap_floor_raw_seconds")
                            is not None else None
                        ),
                    "cap_read_seconds_per_block":
                        row.get("cap_read_seconds_per_block"),
                    "cap_implied_read_bandwidth_mib_s":
                        row.get("cap_implied_read_bandwidth_mib_s"),
                })
            continue

        if phase == "cache-micro-pilot":
            benchmark = str(row["kernel"])
            capacity = int(row["capacities_mib"]) * 1024 * 1024
            ways = 11 if capacity == 22 * 1024 * 1024 else 16
            cache_environment = {
                **base_environment,
                "OMP_NUM_THREADS": "1",
                "CACHE_L1_SIZE": str(min(capacity, 32 * 1024)),
                "CACHE_L2_SIZE": str(min(capacity, 256 * 1024)),
                "CACHE_L3_SIZE": str(capacity),
                "CACHE_L3_WAYS": str(ways),
                "CACHE_LINE_SIZE": "64",
                "CACHE_POLICY": "CLOCK",
                "CACHE_MULTICORE": "0",
                "CACHE_SAMPLED": "0",
                "CACHE_ULTRAFAST": "1",
                "CACHE_FAST": "0",
            }
            command = [
                str(PROJECT_ROOT / "bench" / "bin_sim" / benchmark),
                "-f", str(graph_path),
                "-s",
                "-n", "1",
                *(
                    [
                        "-F",
                        "-i", str(CACHE_PR_ITERATIONS),
                        "-t", str(PR_TOLERANCE),
                    ]
                    if benchmark in {"pr", "pr_spmv"}
                    else _kernel_policy_args(benchmark, graph_name)
                ),
                "-o", str(row["arm"]),
            ]
            expected_sources = []
            expected_internals = []
            expected_degrees = []
            if row.get("source_index") is not None:
                source = source_manifest["sources"][int(row["source_index"])]
                expected_sources = [int(source["source_id"])]
                expected_internals = []
                expected_degrees = [int(source["source_out_degree"])]
                if int(row["source_vertex_id"]) != expected_sources[0]:
                    raise ValueError("Cache probe source binding changed")
                command.extend(["-r", str(expected_sources[0])])
            commands.append({
                **common,
                "command_id":
                    f"{phase}|{graph_name}|{benchmark}|{row['arm']}|"
                    f"{row['capacities_mib']}MiB|s{row.get('source_index')}",
                "process_id": 0,
                "measurement_mode": "cold-process",
                "source_policy_id": (
                    ADAPTIVE_SOURCE_POLICY_ID
                    if expected_sources else None
                ),
                "source_repeats": 1,
                "expected_sources": expected_sources or None,
                "expected_source_internals": None,
                "expected_source_out_degrees":
                    expected_degrees or None,
                "command": _taskset_command(
                    command,
                    cpu_list.split(",")[0].split("-")[0]
                    if cpu_list else None,
                ),
                "environment": cache_environment,
                "timeout_seconds":
                    math.ceil(float(row["wall_clock_cap_seconds"])),
                "cache_iterations": (
                    CACHE_PR_ITERATIONS
                    if benchmark in {"pr", "pr_spmv"}
                    else 1
                ),
            })
            continue

        if phase == "feature-cost-pilot":
            command = [
                str(PROJECT_ROOT / "bench" / "bin" / "pr"),
                "-f", str(graph_path),
                "-s",
                "-n", "1",
                *_kernel_policy_args("pr", graph_name),
                "-o", "14",
            ]
            commands.append({
                **common,
                "command_id":
                    f"{phase}|{graph_name}|{labeling}",
                "process_id": 0,
                "measurement_mode": "cold-process",
                "source_policy_id": None,
                "source_repeats": 1,
                "expected_sources": None,
                "expected_source_internals": None,
                "expected_source_out_degrees": None,
                "command": _taskset_command(command, cpu_list),
                "environment": {
                    **base_environment,
                    "PERCEPTRON_WEIGHTS_FILE": str(contract_path),
                },
                "timeout_seconds":
                    math.ceil(float(row["wall_clock_cap_seconds"])),
                "tier0_trained": False,
            })
            continue

        if phase == "rss-pilot":
            command = [
                "/usr/bin/time", "-v",
                str(PROJECT_ROOT / "bench" / "bin" / "pr"),
                "-f", str(graph_path),
                "-s",
                "-n", "1",
                "-F",
                "-i", "1",
                "-t", str(PR_TOLERANCE),
                "-o", str(row["arm"]),
            ]
            commands.append({
                **common,
                "command_id":
                    f"{phase}|{graph_name}|{row['arm']}",
                "process_id": 0,
                "measurement_mode": "peak-rss",
                "source_policy_id": None,
                "source_repeats": 1,
                "expected_sources": None,
                "expected_source_internals": None,
                "expected_source_out_degrees": None,
                "command": _taskset_command(command, cpu_list),
                "environment": base_environment,
                "timeout_seconds":
                    math.ceil(float(row["wall_clock_cap_seconds"])),
            })
            continue

        raise ValueError(f"Unsupported authorized pilot phase: {phase}")

    expected_command_count = sum(
        (
            int(row["process_blocks"])
            if row["phase"] in {
                "randomized-pilot",
                "natural-label-pilot",
            }
            else 0
            if row["phase"] == "natural-label-materialization"
            else 1
        )
        for row in all_budget_rows
    )
    if len(commands) != expected_command_count:
        raise ValueError(
            "Pilot command count changed: "
            f"expected {expected_command_count}, got {len(commands)}")
    for command in commands:
        if (
            command["phase"] == "cache-micro-pilot"
            and command["kernel"] in {"pr", "pr_spmv"}
        ):
            args = command["command"]
            iteration_index = args.index("-i")
            if int(args[iteration_index + 1]) != CACHE_PR_ITERATIONS:
                raise ValueError(
                    "Cache PR command iteration count changed")
    _validate_command_binaries(commands)
    for command in commands:
        headroom = (
            command["timeout_seconds"]
            - float(command.get("cap_floor_raw_seconds") or 0.0)
        )
        if headroom <= 0:
            raise ValueError(
                f"Pilot command has no cap headroom: {command['command_id']}")
        command["cap_headroom_seconds"] = headroom
        safe_id = (
            command["command_id"]
            .replace("|", "__")
            .replace("/", "_")
            .replace(":", "_")
        )
        command_dir = output_root / safe_id
        command["attempt"] = 0
        command["retry_attempts"] = list(PILOT_RETRY_ATTEMPTS)
        command["idempotency_key"] = (
            f"{command['command_id']}|a{command['attempt']}"
        )
        attempt_dir = command_dir / "attempt_0"
        command["stdout_path"] = str(attempt_dir / "stdout.log")
        command["stderr_path"] = str(attempt_dir / "stderr.log")
        command["result_path"] = str(attempt_dir / "result.json")
        if command["phase"] == "cache-micro-pilot":
            cache_output_path = attempt_dir / "cache_stats.json"
            command["environment"]["CACHE_OUTPUT_JSON"] = str(
                cache_output_path)
            command["cache_output_path"] = str(cache_output_path)
        command["environment_mode"] = "inherit-then-override"
        command["binary_provenance"] = _command_binary_provenance(
            command["command"])
        command["rss_from_stderr"] = (
            command["phase"] == "rss-pilot"
        )
        command["depends_on"] = [
            f"page-cache-prime|{command['graph']}|"
            f"{command.get('labeling') or 'randomized'}"
        ]

    priming_commands = []
    priming_specs = [
        (
            graph_name,
            "randomized",
            graph_root / graph_name / f"{graph_name}.sg",
        )
        for graph_name in budget["pilot_graphs"]
    ] + [
        (
            graph_name,
            "natural",
            Path(natural_graphs[graph_name]["natural_graph"]),
        )
        for graph_name in NATURAL_PILOT_GRAPHS
    ]
    for graph_name, labeling, graph_path in priming_specs:
        graph_metadata = _current_graph_metadata(
            graph_path,
            graph_name=graph_name,
            natural=labeling == "natural",
            verify_content=True,
            expected_nodes=int(
                next(
                    graph["nodes"] for graph in EVAL_GRAPHS
                    if graph["name"] == graph_name
                )
            ),
            expected_undirected_edges=int(
                next(
                    graph["undirected_edges"]
                    for graph in EVAL_GRAPHS
                    if graph["name"] == graph_name
                )
            ),
        )
        read_evidence = max(
            float(row.get("cap_read_seconds_per_block", 0.0))
            for row in (
                budget["randomized_pilot_rows"]
                + budget["natural_pilot_rows"]
            )
            if row["graph"] == graph_name
        )
        priming_commands.append({
            "command_id": f"page-cache-prime|{graph_name}|{labeling}",
            "phase": "page-cache-prime",
            "graph_path": str(graph_path),
            "timeout_interpretation": "hard-preparation-cap",
            "attempt": 0,
            "retry_attempts": list(PILOT_RETRY_ATTEMPTS),
            "depends_on": [],
            "graph": graph_name,
            "labeling": labeling,
            "graph_output_bytes": graph_path.stat().st_size,
            "graph_mtime_ns": graph_path.stat().st_mtime_ns,
            "graph_crc32": graph_metadata["output_crc32"],
            "command": _taskset_command([
                "dd",
                f"if={graph_path}",
                "of=/dev/null",
                "bs=16M",
                "status=none",
            ], cpu_list),
            "timeout_seconds": math.ceil(
                10.0 * read_evidence + 60.0),
            "environment": {},
            "environment_mode": "inherit-then-override",
            "cpu_list": cpu_list,
            "host_state_requirements": {
                "cpu_governor": "performance",
                "turbo": "disabled",
                "transparent_hugepage": "madvise",
            },
            "cap_floor_raw_seconds": read_evidence,
            "cap_headroom_seconds": (
                math.ceil(10.0 * read_evidence + 60.0)
                - read_evidence
            ),
            "stdout_path": str(
                output_root
                / f"prime__{graph_name}__{labeling}"
                / "attempt_0"
                / "stdout.log"),
            "stderr_path": str(
                output_root
                / f"prime__{graph_name}__{labeling}"
                / "attempt_0"
                / "stderr.log"),
            "result_path": str(
                output_root
                / f"prime__{graph_name}__{labeling}"
                / "attempt_0"
                / "result.json"),
        })
        priming_commands[-1]["idempotency_key"] = (
            f"{priming_commands[-1]['command_id']}|a0"
        )
        priming_commands[-1]["binary_provenance"] = (
            _command_binary_provenance(
                priming_commands[-1]["command"])
        )

    input_artifacts = _execution_input_artifacts(
        budget_path,
        source_path,
        natural_path,
        contract_path,
    )
    manifest = {
        "schema": "adaptive-pilot-execution/v2",
        "dry_run_only": True,
        "budget_projection": str(budget_path),
        "source_manifest": str(source_path),
        "natural_manifest": str(natural_path),
        "contract_weights": str(contract_path),
        "input_artifacts": input_artifacts,
        "threads": threads,
        "cpu_list": cpu_list,
        "authorized_budget_rows": len(all_budget_rows),
        "command_count": len(commands),
        "priming_command_count": len(priming_commands),
        "execution_order": ["priming_commands", "commands"],
        "concurrency": "serial-exclusive",
        "execution_lock_path": str(
            sprint_root / "pilot_execution.lock"),
        "execution_authorization_required": True,
        "completed_preparation": completed_preparation,
        "priming_commands": priming_commands,
        "commands": commands,
        "host_state_requirements": {
            "cpu_governor": "performance",
            "turbo": "disabled",
            "transparent_hugepage": "madvise",
            "graphbrew_db_dir": "",
            "topology_analysis": "disabled",
        },
    }
    manifest_path = sprint_root / "pilot_execution_manifest.json"
    if manifest_path.is_file():
        existing = _load_json(manifest_path)
        comparable = (
            "schema",
            "dry_run_only",
            "budget_projection",
            "source_manifest",
            "natural_manifest",
            "contract_weights",
            "input_artifacts",
            "input_artifacts",
            "threads",
            "cpu_list",
            "authorized_budget_rows",
            "command_count",
            "priming_command_count",
            "execution_order",
            "concurrency",
            "completed_preparation",
            "host_state_requirements",
            "priming_commands",
            "commands",
        )
        changed = any(
            existing.get(field) != manifest.get(field)
            for field in comparable
        )
        if changed and not refreeze:
            raise RuntimeError(
                "Frozen pilot execution manifest changed; "
                "use --refreeze-pilot-manifest after review")
    _atomic_json(manifest, manifest_path)
    return manifest_path


def _parse_cpu_list(cpu_list: str) -> list[int]:
    cpus = []
    for token in cpu_list.split(","):
        if "-" in token:
            start, end = map(int, token.split("-", 1))
            cpus.extend(range(start, end + 1))
        else:
            cpus.append(int(token))
    if not cpus or len(cpus) != len(set(cpus)):
        raise ValueError("CPU list must contain unique CPU IDs")
    return cpus


def _host_state_snapshot(cpu_list: str) -> dict[str, Any]:
    cpus = _parse_cpu_list(cpu_list)
    governors = []
    for cpu in cpus:
        path = Path(
            f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/"
            "scaling_governor"
        )
        if path.is_file():
            governors.append(path.read_text().strip())
    no_turbo_path = Path(
        "/sys/devices/system/cpu/intel_pstate/no_turbo")
    thp_path = Path(
        "/sys/kernel/mm/transparent_hugepage/enabled")
    return {
        "hostname": socket.gethostname(),
        "cpu_list": cpu_list,
        "cpu_governors": sorted(set(governors)),
        "intel_pstate_no_turbo": (
            no_turbo_path.read_text().strip()
            if no_turbo_path.is_file() else None
        ),
        "transparent_hugepage": (
            thp_path.read_text().strip()
            if thp_path.is_file() else None
        ),
    }


def _assert_host_state(
    host: dict[str, Any],
    requirements: dict[str, Any],
) -> None:
    if (
        requirements.get("cpu_governor") == "performance"
        and host["cpu_governors"] != ["performance"]
    ):
        raise RuntimeError("Pilot CPU governor is not performance")
    if requirements.get("turbo") == "disabled":
        if host["intel_pstate_no_turbo"] != "1":
            raise RuntimeError("Pilot turbo state is unavailable or enabled")
    required_thp = requirements.get("transparent_hugepage")
    if (
        required_thp
        and f"[{required_thp}]" not in (
            host.get("transparent_hugepage") or "")
    ):
        raise RuntimeError(
            "Pilot transparent hugepage policy changed")


def _forbidden_ambient_timing_environment(
    overrides: dict[str, str] | None = None,
    *,
    application_surface: bool = True,
) -> list[str]:
    overrides = overrides or {}
    forbidden = [
        name for name in PILOT_FORBIDDEN_AMBIENT_ENV
        if os.environ.get(name) and name not in overrides
    ]
    forbidden.extend(
        name for name in os.environ
        if (
            name not in overrides
            and (
                name.startswith(("KMP_", "GOMP_"))
                or (
                    application_surface
                    and name.startswith(
                        PILOT_FORBIDDEN_APPLICATION_PREFIXES)
                )
            )
        )
    )
    return sorted(set(forbidden))


def _validate_runtime_environment_surface() -> None:
    patterns = (
        re.compile(r'getenv\("([A-Z0-9_]+)"\)'),
        re.compile(
            r'(?:env_bool|env_int|env_double|getEnvSize)'
            r'\(\s*"([A-Z0-9_]+)"'
        ),
    )
    names = set()
    for root in (
        PROJECT_ROOT / "bench" / "include",
        PROJECT_ROOT / "bench" / "src",
        PROJECT_ROOT / "bench" / "src_sim",
    ):
        for path in root.rglob("*"):
            if path.suffix not in {
                ".h", ".hpp", ".hxx", ".cc", ".cpp", ".c", ".inc",
            }:
                continue
            content = path.read_text(errors="ignore")
            for pattern in patterns:
                names.update(pattern.findall(content))
    uncovered = [
        name for name in sorted(names)
        if not (
            name in PILOT_FORBIDDEN_AMBIENT_ENV
            or name.startswith(PILOT_FORBIDDEN_APPLICATION_PREFIXES)
        )
    ]
    if uncovered:
        raise RuntimeError(
            "Runtime getenv surface is not covered by pilot policy: "
            + " ".join(uncovered)
        )


def _validate_execution_manifest(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    if manifest.get("schema") != "adaptive-pilot-execution/v2":
        raise ValueError("Unsupported pilot execution manifest")
    if manifest.get("execution_order") != [
        "priming_commands", "commands"
    ]:
        raise ValueError("Pilot execution order changed")
    if manifest.get("concurrency") != "serial-exclusive":
        raise ValueError("Pilot execution must be serial-exclusive")
    input_artifacts = manifest.get("input_artifacts")
    expected_inputs = {
        "budget_projection",
        "source_manifest",
        "natural_manifest",
        "contract_weights",
    }
    if (
        not isinstance(input_artifacts, dict)
        or set(input_artifacts) != expected_inputs
    ):
        raise ValueError("Pilot input artifact coverage changed")
    for name, binding in input_artifacts.items():
        if binding.get("path") != str(
            Path(manifest[name]).resolve()
        ):
            raise ValueError(
                f"Pilot input artifact path changed: {name}")
        _validate_artifact_binding(binding)
    budget = _load_json(Path(manifest["budget_projection"]))
    source_bundle = _load_json(Path(manifest["source_manifest"]))
    natural_manifest = _load_json(Path(manifest["natural_manifest"]))
    _verify_budget_preconditions(
        budget, source_bundle, natural_manifest)
    if _load_json(Path(manifest["contract_weights"])) != (
        _contract_weight_payload()
    ):
        raise ValueError("Pilot contract weights changed")
    expected_command_ids, expected_priming_ids = (
        _expected_pilot_command_ids(budget)
    )
    _validate_runtime_environment_surface()
    priming = manifest.get("priming_commands", [])
    priming_ids = {
        command["command_id"]
        for command in priming
    }
    if len(priming_ids) != manifest.get("priming_command_count"):
        raise ValueError("Pilot priming command count changed")
    if priming_ids != expected_priming_ids:
        raise ValueError("Pilot priming coverage changed")
    commands = manifest.get("commands", [])
    if len(commands) != manifest.get("command_count"):
        raise ValueError("Pilot timing command count changed")
    if {
        command.get("command_id") for command in commands
    } != set(expected_command_ids):
        raise ValueError("Pilot timing command coverage changed")
    command_ids = set()
    result_paths = set()
    for command in [*priming, *commands]:
        missing = PILOT_CONSUMER_REQUIRED_KEYS - set(command)
        if missing:
            raise ValueError(
                "Pilot command is missing consumer fields: "
                + " ".join(sorted(missing))
            )
        if command.get("environment_mode") != "inherit-then-override":
            raise ValueError("Pilot environment merge policy changed")
        if command["command_id"] in command_ids:
            raise ValueError("Pilot command ID is duplicated")
        command_ids.add(command["command_id"])
        if command["result_path"] in result_paths:
            raise ValueError("Pilot result path is duplicated")
        result_paths.add(command["result_path"])
        if command["idempotency_key"] != (
            f"{command['command_id']}|a{command['attempt']}"
        ):
            raise ValueError("Pilot idempotency key does not match attempt")
        if command["attempt"] not in command["retry_attempts"]:
            raise ValueError("Pilot attempt is outside retry policy")
        if command["phase"] != "page-cache-prime":
            if command.get("budget_row_index") != (
                expected_command_ids[command["command_id"]]
            ):
                raise ValueError(
                    "Pilot command budget-row binding changed")
            dependencies = command.get("depends_on")
            if not dependencies:
                raise ValueError("Pilot timing command lost priming dependency")
            if not set(dependencies).issubset(priming_ids):
                raise ValueError("Pilot command has an unknown dependency")
        if float(command.get("timeout_seconds", 0)) <= 0:
            raise ValueError("Pilot command has no timeout")
        if float(command.get("cap_headroom_seconds", 0)) <= 0:
            raise ValueError("Pilot command has no cap headroom")
        graph_path = Path(command["graph_path"])
        if (
            not graph_path.is_file()
            or graph_path.stat().st_size
                != int(command["graph_output_bytes"])
            or graph_path.stat().st_mtime_ns
                != int(command["graph_mtime_ns"])
            or file_crc32(graph_path) != command["graph_crc32"]
        ):
            raise RuntimeError("Pilot graph file changed after freeze")
        _validate_binary_provenance(command)
        if command["phase"] == "cache-micro-pilot":
            environment = command["environment"]
            required = {
                "CACHE_L3_SIZE",
                "CACHE_L3_WAYS",
                "CACHE_POLICY",
                "CACHE_ULTRAFAST",
                "CACHE_OUTPUT_JSON",
                "OMP_NUM_THREADS",
            }
            if not required.issubset(environment):
                raise ValueError("Cache command environment is incomplete")
            if environment["OMP_NUM_THREADS"] != "1":
                raise ValueError("Cache command must be single-threaded")
    _validate_command_binaries(commands)
    for command in [*priming, *commands]:
        forbidden = _forbidden_ambient_timing_environment(
            command["environment"],
            application_surface=(
                command["phase"] != "page-cache-prime"),
        )
        if forbidden:
            raise RuntimeError(
                "Forbidden ambient timing variables are set for "
                f"{command['command_id']}: "
                + " ".join(forbidden)
            )
    if not manifest.get("cpu_list"):
        raise ValueError("Pilot CPU list is missing")
    host = _host_state_snapshot(manifest["cpu_list"])
    requirements = manifest.get("host_state_requirements", {})
    _assert_host_state(host, requirements)
    return host


def _pilot_command_for_attempt(
    command: dict[str, Any],
    attempt: int,
) -> dict[str, Any]:
    if attempt not in command["retry_attempts"]:
        raise ValueError("Requested pilot retry attempt is not pre-registered")
    updated = dict(command)
    updated["environment"] = dict(command["environment"])
    base_dir = Path(command["result_path"]).parents[1]
    attempt_dir = base_dir / f"attempt_{attempt}"
    updated["attempt"] = attempt
    updated["idempotency_key"] = (
        f"{command['command_id']}|a{attempt}")
    updated["stdout_path"] = str(attempt_dir / "stdout.log")
    updated["stderr_path"] = str(attempt_dir / "stderr.log")
    updated["result_path"] = str(attempt_dir / "result.json")
    if command.get("cache_output_path"):
        cache_output_path = attempt_dir / "cache_stats.json"
        updated["cache_output_path"] = str(cache_output_path)
        updated["environment"]["CACHE_OUTPUT_JSON"] = str(
            cache_output_path)
    return updated


def _priming_command_for_session(
    command: dict[str, Any],
    session_id: str,
) -> dict[str, Any]:
    updated = dict(command)
    base_dir = Path(command["result_path"]).parents[1]
    session_dir = base_dir / "sessions" / session_id
    updated["idempotency_key"] = (
        f"{command['command_id']}|session={session_id}")
    updated["stdout_path"] = str(session_dir / "stdout.log")
    updated["stderr_path"] = str(session_dir / "stderr.log")
    updated["result_path"] = str(session_dir / "result.json")
    return updated


class _PilotExecutionLock:
    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, "a+")
        try:
            fcntl.flock(
                self.handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as error:
            self.handle.close()
            raise RuntimeError(
                "Another pilot executor holds the serial run lock"
            ) from error
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(f"{os.getpid()}\n")
        self.handle.flush()
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
        return False


def _run_literal_pilot_command(
    command: dict[str, Any],
    host_state: dict[str, Any],
) -> dict[str, Any]:
    """Execute one literal manifest command; intentionally not CLI-exposed."""
    required_authorization = (
        "authorization_reference",
        "execution_manifest_sha256",
        "command_contract_sha256",
        "input_artifacts",
    )
    if any(
        field not in command
        or (
            field != "input_artifacts"
            and not command.get(field)
        )
        for field in required_authorization
    ):
        raise RuntimeError("Pilot command lacks execution authorization")
    contract_payload = dict(command)
    recorded_contract = contract_payload.pop(
        "command_contract_sha256")
    if _canonical_json_sha256(contract_payload) != recorded_contract:
        raise RuntimeError("Pilot command authorization digest changed")
    if not isinstance(command["input_artifacts"], dict):
        raise RuntimeError("Pilot command input artifact binding is missing")
    for binding in command["input_artifacts"].values():
        _validate_artifact_binding(binding)
    result_path = Path(command["result_path"])
    if result_path.is_file():
        existing = _load_json(result_path)
        if (
            existing.get("schema") != "adaptive-pilot-result/v2"
            or existing.get("command_contract_sha256")
                != command["command_contract_sha256"]
            or existing.get("authorization_reference")
                != command["authorization_reference"]
            or existing.get("execution_manifest_sha256")
                != command["execution_manifest_sha256"]
        ):
            raise RuntimeError("Pilot result command contract changed")
        if existing.get("error_kind") in {"", "timeout"}:
            return existing
        if existing.get("error_kind") == "contract-violation":
            raise SourceContractError(
                "Recorded pilot contract violation: "
                + str(existing.get("contract_violation"))
            )
        raise RuntimeError(
            "Pilot process failure requires an explicit retry attempt")

    result_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path = Path(command["graph_path"])
    if (
        not graph_path.is_file()
        or graph_path.stat().st_size != int(command["graph_output_bytes"])
        or graph_path.stat().st_mtime_ns != int(command["graph_mtime_ns"])
        or file_crc32(graph_path) != command["graph_crc32"]
    ):
        raise RuntimeError("Pilot graph file changed before execution")
    _validate_binary_provenance(command)
    forbidden = _forbidden_ambient_timing_environment(
        command["environment"],
        application_surface=(
            command["phase"] != "page-cache-prime"),
    )
    if forbidden:
        raise RuntimeError(
            "Forbidden ambient timing variables are set: "
            + " ".join(forbidden)
        )
    environment = {**os.environ, **command["environment"]}
    timing_environment = {
        key: environment[key]
        for key in sorted(environment)
        if (
            key.startswith((
                "OMP_", "GOMP_", "KMP_", "CACHE_",
                "GRAPHBREW_", "ECG_",
            ))
            or key in PILOT_FORBIDDEN_AMBIENT_ENV
            or key == "LD_LIBRARY_PATH"
            or key.startswith(PILOT_FORBIDDEN_APPLICATION_PREFIXES)
            or key.startswith("KMP_")
        )
    }
    if command.get("cpu_list"):
        current_host_state = _host_state_snapshot(command["cpu_list"])
        _assert_host_state(
            current_host_state,
            command.get("host_state_requirements", {}),
        )
    else:
        current_host_state = host_state
    started = time.monotonic()
    stdout = ""
    stderr = ""
    returncode = None
    error_kind = ""
    censored = False
    running_path = result_path.parent / "running.json"
    try:
        marker_fd = os.open(
            running_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o644,
        )
    except FileExistsError as error:
        raise RuntimeError(
            f"Pilot command has a stale running marker: {running_path}"
        ) from error
    os.close(marker_fd)
    process = None
    previous_handlers = {}
    previous_signal_mask = None

    def terminate_active_process(_signum, _frame):
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        raise KeyboardInterrupt

    try:
        blocked_signals = {signal.SIGINT, signal.SIGTERM}
        previous_signal_mask = signal.pthread_sigmask(
            signal.SIG_BLOCK, blocked_signals)
        for signal_number in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signal_number] = signal.getsignal(
                signal_number)
            signal.signal(signal_number, terminate_active_process)
        process = subprocess.Popen(
            command["command"],
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        signal.pthread_sigmask(
            signal.SIG_SETMASK, previous_signal_mask)
        previous_signal_mask = None
        try:
            _atomic_json({
                "schema": "adaptive-pilot-running/v1",
                "executor_pid": os.getpid(),
                "child_pid": process.pid,
                "command_id": command["command_id"],
                "attempt": command["attempt"],
            }, running_path)
            stdout, stderr = process.communicate(
                timeout=float(command["timeout_seconds"])
            )
            returncode = process.returncode
            if returncode != 0:
                error_kind = "process-failure"
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
            error_kind = "timeout"
            censored = True
    finally:
        if previous_signal_mask is not None:
            signal.pthread_sigmask(
                signal.SIG_SETMASK, previous_signal_mask)
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        for signal_number, previous in previous_handlers.items():
            signal.signal(signal_number, previous)
        running_path.unlink(missing_ok=True)

    Path(command["stdout_path"]).write_text(stdout)
    Path(command["stderr_path"]).write_text(stderr)
    contract_violation = None
    try:
        average_time, reorder_time, parsed_extra = (
            _parse_and_validate_pilot_output(
                command,
                stdout,
                stderr,
                error_kind=error_kind,
                censored=censored,
            )
        )
    except (SourceContractError, RuntimeError, ValueError) as error:
        average_time = 0.0
        reorder_time = 0.0
        parsed_extra = {}
        error_kind = "contract-violation"
        contract_violation = str(error)

    payload = {
        "schema": "adaptive-pilot-result/v2",
        "idempotency_key": command["idempotency_key"],
        "command_id": command["command_id"],
        "phase": command["phase"],
        "graph": command["graph"],
        "graph_path": command["graph_path"],
        "labeling": command.get("labeling"),
        "kernel": command.get("kernel"),
        "arm": command.get("arm"),
        "order_spec": command.get("order_spec"),
        "process_id": command.get("process_id"),
        "measurement_mode": command.get("measurement_mode"),
        "attempt": command.get("attempt", 0),
        "command": command["command"],
        "environment": command["environment"],
        "resolved_timing_environment": timing_environment,
        "environment_mode": command["environment_mode"],
        "timeout_seconds": command["timeout_seconds"],
        "timeout_interpretation":
            command["timeout_interpretation"],
        "duration_seconds": time.monotonic() - started,
        "returncode": returncode,
        "error_kind": error_kind,
        "contract_violation": contract_violation,
        "censored": censored,
        "claim_eligible": False,
        "pilot_only": True,
        "average_time": average_time,
        "reorder_time": reorder_time,
        "extra": parsed_extra,
        "host_state": current_host_state,
        "authorization_reference":
            command.get("authorization_reference"),
        "execution_manifest_sha256":
            command.get("execution_manifest_sha256"),
        "command_contract_sha256":
            command.get("command_contract_sha256"),
    }
    _atomic_json(payload, result_path)
    if contract_violation is not None:
        raise SourceContractError(
            f"{contract_violation}; recorded at {result_path}")
    return payload


def _parse_and_validate_pilot_output(
    command: dict[str, Any],
    stdout: str,
    stderr: str,
    *,
    error_kind: str,
    censored: bool,
) -> tuple[float, float, dict[str, Any]]:
    parsed_extra: dict[str, Any] = {}
    average_time = 0.0
    reorder_time = 0.0
    if re.search(
        r"warning:.*unknown.*variant",
        stdout + "\n" + stderr,
        flags=re.IGNORECASE,
    ):
        raise SourceContractError(
            "Pilot command emitted an unknown-variant warning")
    if not error_kind:
        average_time, reorder_time, parsed_extra = (
            parse_benchmark_output(stdout)
        )
        order_spec = str(command.get("order_spec") or "")
        effective, realized = _validate_pilot_realized_order(
            order_spec, stdout, stderr)
        if effective:
            parsed_extra["graphbrew_effective_configs"] = effective
            parsed_extra["graphbrew_realized_configs"] = realized
        if command.get("source_policy_id") is not None:
            attach_source_trial_metadata(
                parsed_extra,
                process_id=int(command["process_id"]),
                measurement_mode=str(command["measurement_mode"]),
                source_policy_id=str(command["source_policy_id"]),
                source_repeats=int(command["source_repeats"]),
                expected_sources=command.get("expected_sources"),
                expected_internals=command.get(
                    "expected_source_internals"),
                expected_out_degrees=command.get(
                    "expected_source_out_degrees"),
            )
        if command["phase"] == "feature-cost-pilot":
            if (
                parsed_extra.get("adaptive_weight_source")
                    != "env-override"
                or parsed_extra.get("adaptive_tier0_trained")
                    != "development-override"
            ):
                raise SourceContractError(
                    "Feature pilot did not use contract Tier-0 weights")
    elif censored and stdout:
        order_spec = str(command.get("order_spec") or "")
        if (
            order_spec.split(":", 1)[0] == "12"
            and "GraphBrew Realized Config:" in stdout
        ):
            _validate_pilot_realized_order(
                order_spec, stdout, stderr)
        try:
            average_time, reorder_time, parsed_extra = (
                parse_benchmark_output(stdout)
            )
            parsed_extra["censored_block"] = True
        except (ValueError, SourceContractError):
            parsed_extra = {"censored_block": True}

    if command.get("rss_from_stderr"):
        match = re.search(
            r"Maximum resident set size \(kbytes\):\s*(\d+)",
            stderr,
        )
        if match:
            parsed_extra["peak_rss_kib"] = int(match.group(1))
    if not error_kind and command["phase"] != "page-cache-prime":
        if average_time <= 0 or not parsed_extra.get("trial_times"):
            raise SourceContractError(
                "Pilot command exited successfully without timing output")
        if (
            command["phase"] == "rss-pilot"
            and "peak_rss_kib" not in parsed_extra
        ):
            raise SourceContractError(
                "RSS pilot exited without peak RSS metadata")
        if command["phase"] == "cache-micro-pilot":
            cache_output_path = Path(command["cache_output_path"])
            if not cache_output_path.is_file():
                raise SourceContractError(
                    "Cache pilot exited without cache JSON sidecar")
            parsed_extra["cache_stats_path"] = str(cache_output_path)
            parsed_extra["cache_stats"] = _load_json(cache_output_path)
    return average_time, reorder_time, parsed_extra


def _validate_pilot_realized_order(
    order_spec: str,
    stdout: str,
    stderr: str,
) -> tuple[list[dict], list[dict]]:
    combined_output = stdout + "\n" + stderr
    if re.search(
        r"warning:.*unknown.*variant",
        combined_output,
        flags=re.IGNORECASE,
    ):
        raise SourceContractError(
            "Pilot command emitted an unknown-variant warning")
    if order_spec.split(":", 1)[0] != "12":
        return [], []
    algo_flags = ["-o", order_spec]
    effective = parse_graphbrew_effective_configs(stdout)
    realized = parse_graphbrew_realized_configs(stdout)
    validate_graphbrew_effective_configs(algo_flags, effective)
    validate_graphbrew_realized_configs(
        algo_flags, effective, realized)
    fallbacks = [
        fallback
        for config in realized
        for fallback in config.get("fallbacks", [])
    ]
    if fallbacks:
        raise SourceContractError(
            "Pilot GraphBrew arm used a runtime fallback: "
            + json.dumps(fallbacks, sort_keys=True)
        )
    return effective, realized


def _run_pilot_command_with_retries(
    base_command: dict[str, Any],
    host_state: dict[str, Any],
    authorization_reference: str,
    execution_manifest_sha256: str,
    input_artifacts: dict[str, Any],
    *,
    runner=_run_literal_pilot_command,
) -> dict[str, Any]:
    for attempt in base_command["retry_attempts"]:
        command = _bind_authorized_command(
            _pilot_command_for_attempt(base_command, attempt),
            authorization_reference,
            execution_manifest_sha256,
            input_artifacts,
        )
        result_path = Path(command["result_path"])
        if result_path.is_file():
            existing = _load_json(result_path)
            if (
                existing.get("schema") != "adaptive-pilot-result/v2"
                or existing.get("command_contract_sha256")
                    != command["command_contract_sha256"]
                or existing.get("authorization_reference")
                    != command["authorization_reference"]
                or existing.get("execution_manifest_sha256")
                    != command["execution_manifest_sha256"]
            ):
                raise RuntimeError(
                    "Pilot retry result contract changed")
            if existing.get("error_kind") == "process-failure":
                continue
        result = runner(command, host_state)
        if result["error_kind"] != "process-failure":
            return result
    raise RuntimeError(
        "Pilot command exhausted pre-registered retry attempts: "
        + base_command["command_id"]
    )


def _bind_authorized_command(
    command: dict[str, Any],
    authorization_reference: str,
    execution_manifest_sha256: str,
    input_artifacts: dict[str, Any],
) -> dict[str, Any]:
    bound = dict(command)
    bound["authorization_reference"] = authorization_reference
    bound["execution_manifest_sha256"] = execution_manifest_sha256
    bound["input_artifacts"] = input_artifacts
    bound["command_contract_sha256"] = _canonical_json_sha256(bound)
    return bound


def _authorization_manifest_contract(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Return the reviewer-visible manifest summary bound by authorization."""
    fields = (
        "schema",
        "dry_run_only",
        "budget_projection",
        "source_manifest",
        "natural_manifest",
        "contract_weights",
        "threads",
        "cpu_list",
        "authorized_budget_rows",
        "command_count",
        "priming_command_count",
        "execution_order",
        "concurrency",
        "execution_lock_path",
        "execution_authorization_required",
        "host_state_requirements",
    )
    return {field: manifest.get(field) for field in fields}


def execute_sprint1_pilot(
    artifact_root: Path,
    *,
    authorization_reference: str,
) -> Path:
    """Execute a separately authorized, content-bound pilot manifest."""
    if not authorization_reference:
        raise ValueError("Pilot execution requires authorization reference")
    sprint_root = artifact_root / "adaptive_selector" / "sprint1"
    manifest_path = sprint_root / "pilot_execution_manifest.json"
    manifest = _load_json(manifest_path)
    if not manifest.get("execution_authorization_required"):
        raise RuntimeError("Pilot authorization gate is missing")
    authorization_path = (
        sprint_root / "pilot_execution_authorization.json")
    authorization = _load_json(authorization_path)
    manifest_sha256 = _canonical_json_sha256(manifest)
    manifest_contract = _authorization_manifest_contract(manifest)
    if (
        authorization.get("schema")
            != "adaptive-pilot-execution-authorization/v2"
        or not authorization.get("execution_enabled")
        or authorization.get("authorization_reference")
            != authorization_reference
        or authorization.get("execution_manifest") != str(manifest_path)
        or authorization.get("command_count")
            != manifest.get("command_count")
        or authorization.get("execution_manifest_sha256")
            != manifest_sha256
        or authorization.get("authorized_manifest_contract")
            != manifest_contract
    ):
        raise RuntimeError("Pilot execution authorization is invalid")
    host_state = _validate_execution_manifest(manifest)

    lock_path = Path(manifest["execution_lock_path"])
    session_id = f"{time.time_ns()}-{os.getpid()}"
    execution_results = []
    with _PilotExecutionLock(lock_path):
        for command in manifest["priming_commands"]:
            priming = _bind_authorized_command(
                _priming_command_for_session(command, session_id),
                authorization_reference,
                manifest_sha256,
                manifest["input_artifacts"],
            )
            result = _run_literal_pilot_command(priming, host_state)
            execution_results.append(result)
            if result["error_kind"]:
                raise RuntimeError(
                    "Page-cache priming failed; timing is blocked: "
                    + command["command_id"])
        for command in manifest["commands"]:
            execution_results.append(_run_pilot_command_with_retries(
                command,
                host_state,
                authorization_reference,
                manifest_sha256,
                manifest["input_artifacts"],
            ))
    error_counts = {}
    for result in execution_results:
        kind = result.get("error_kind") or "success"
        error_counts[kind] = error_counts.get(kind, 0) + 1
    completion = {
        "schema": "adaptive-pilot-execution-complete/v2",
        "execution_manifest": str(manifest_path),
        "authorization_reference": authorization_reference,
        "execution_session_id": session_id,
        "command_count": manifest["command_count"],
        "priming_command_count": manifest["priming_command_count"],
        "execution_manifest_sha256": manifest_sha256,
        "authorized_manifest_contract": manifest_contract,
        "result_count": len(execution_results),
        "result_states": error_counts,
        "retried_result_count": sum(
            int(result.get("attempt", 0)) > 0
            for result in execution_results
        ),
        "status": "complete",
    }
    path = sprint_root / "pilot_execution_complete.json"
    _atomic_json(completion, path)
    return path


def create_sprint1_execution_authorization(
    artifact_root: Path,
    *,
    authorization_reference: str,
    refreeze: bool = False,
) -> Path:
    """Create a separate authorization bound to the full reviewed manifest."""
    if not authorization_reference:
        raise ValueError("Authorization reference is required")
    sprint_root = artifact_root / "adaptive_selector" / "sprint1"
    manifest_path = sprint_root / "pilot_execution_manifest.json"
    manifest = _load_json(manifest_path)
    _validate_execution_manifest(manifest)
    manifest_contract = _authorization_manifest_contract(manifest)
    payload = {
        "schema": "adaptive-pilot-execution-authorization/v2",
        "execution_enabled": True,
        "authorization_reference": authorization_reference,
        "execution_manifest": str(manifest_path),
        "command_count": manifest["command_count"],
        "priming_command_count": manifest["priming_command_count"],
        "execution_manifest_sha256":
            _canonical_json_sha256(manifest),
        "authorized_manifest_contract": manifest_contract,
    }
    path = sprint_root / "pilot_execution_authorization.json"
    if (
        path.is_file()
        and _load_json(path) != payload
        and not refreeze
    ):
        raise RuntimeError(
            "Pilot execution authorization changed; "
            "use --refreeze-authorization after review"
        )
    _atomic_json(payload, path)
    return path


def validate_sprint1_pilot_executor(
    artifact_root: Path,
) -> Path:
    """Validate the literal consumer without executing pilot commands."""
    sprint_root = artifact_root / "adaptive_selector" / "sprint1"
    manifest_path = sprint_root / "pilot_execution_manifest.json"
    manifest = _load_json(manifest_path)
    host_state = _validate_execution_manifest(manifest)
    validation = {
        "schema": "adaptive-pilot-executor-validation/v2",
        "execution_manifest": str(manifest_path),
        "dry_run_only": bool(manifest.get("dry_run_only")),
        "command_count": manifest["command_count"],
        "priming_command_count": manifest["priming_command_count"],
        "execution_order": manifest["execution_order"],
        "concurrency": manifest["concurrency"],
        "execution_manifest_sha256":
            _canonical_json_sha256(manifest),
        "authorization_manifest_contract":
            _authorization_manifest_contract(manifest),
        "host_state": host_state,
        "literal_consumer": "_run_literal_pilot_command",
        "status": "pass",
    }
    path = sprint_root / "pilot_executor_validation.json"
    _atomic_json(validation, path)
    return path


def _print_summary(projection: dict[str, Any]) -> None:
    print("Adaptive Sprint 1 budget projection")
    print(f"Pilot graphs: {', '.join(projection['pilot_graphs'])}")
    print()
    print(
        f"{'Phase':<28}"
        f"{'Low hours':>12}"
        f"{'High hours':>12}"
    )
    print("-" * 52)
    phases = sorted(projection["phase_buffered_node_hours_high"])
    for phase in phases:
        low = projection["phase_buffered_node_hours_low"].get(
            phase, 0.0)
        high = projection["phase_buffered_node_hours_high"][phase]
        print(f"{phase:<28}{low:>12.2f}{high:>12.2f}")
    print("-" * 52)
    print(
        f"{'TOTAL':<28}"
        f"{projection['projected_buffered_node_hours_low']:>12.2f}"
        f"{projection['projected_buffered_node_hours_high']:>12.2f}"
    )
    print(
        f"{'Budget':<28}"
        f"{'':>12}"
        f"{projection['policy']['budget_hours']:>12.2f}"
    )
    print(
        "Pilot high estimate: "
        f"{projection['pilot_buffered_node_hours_high']:.2f} hours"
    )
    print(
        "Pilot projection if all defined caps bind: "
        f"{projection['pilot_projection_if_all_defined_caps_bind']:.2f} hours"
    )
    print(f"Status: {projection['status']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GraphBrew adaptive-selector experiment runner")
    parser.add_argument(
        "--plan-sprint1",
        action="store_true",
        help="Emit the pre-data Sprint-1 node-hour projection",
    )
    parser.add_argument(
        "--generate-sprint1-sources",
        action="store_true",
        help="Generate frozen source manifests for the Sprint-1 pilot",
    )
    parser.add_argument(
        "--materialize-sprint1-natural",
        action="store_true",
        help="Materialize and validate natural-label pilot graphs",
    )
    parser.add_argument(
        "--prepare-sprint1-pilot",
        action="store_true",
        help="Write a dry-run-only pilot execution manifest",
    )
    parser.add_argument(
        "--validate-sprint1-executor",
        action="store_true",
        help="Validate the literal pilot consumer without executing it",
    )
    parser.add_argument(
        "--authorize-sprint1-pilot",
        action="store_true",
        help="Bind explicit authorization to the reviewed pilot manifest",
    )
    parser.add_argument(
        "--execute-sprint1-pilot",
        action="store_true",
        help="Execute the separately authorized pilot manifest",
    )
    parser.add_argument(
        "--force-sources",
        action="store_true",
        help="Regenerate existing Sprint-1 source manifests",
    )
    parser.add_argument(
        "--refreeze-sources",
        action="store_true",
        help="Replace a frozen source bundle after explicit review",
    )
    parser.add_argument(
        "--refreeze-budget",
        action="store_true",
        help="Replace frozen budget authorization after explicit review",
    )
    parser.add_argument(
        "--refreeze-pilot-manifest",
        action="store_true",
        help="Replace a reviewed dry-run pilot manifest",
    )
    parser.add_argument(
        "--refreeze-authorization",
        action="store_true",
        help="Replace pilot authorization after manifest re-review",
    )
    parser.add_argument("--authorization-reference")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--cpu-list")
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=Path("/media/Data/00_GraphDatasets/GraphBrew"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(
            "/media/Data/00_GraphDatasets/GraphBrew/artifacts"),
    )
    parser.add_argument(
        "--budget-hours",
        type=float,
        default=SPRINT1_BUDGET_HOURS,
    )
    parser.add_argument(
        "--safety-factor",
        type=float,
        default=PROJECTION_SAFETY_FACTOR,
    )
    args = parser.parse_args()
    selected_stages = sum((
        bool(args.plan_sprint1),
        bool(args.generate_sprint1_sources),
        bool(args.materialize_sprint1_natural),
        bool(args.prepare_sprint1_pilot),
        bool(args.validate_sprint1_executor),
        bool(args.authorize_sprint1_pilot),
        bool(args.execute_sprint1_pilot),
    ))
    if selected_stages != 1:
        parser.error("No adaptive stage selected")

    if args.plan_sprint1:
        projection = build_sprint1_budget_projection(
            args.artifact_root.resolve(),
            args.graph_dir.resolve(),
            budget_hours=args.budget_hours,
            safety_factor=args.safety_factor,
        )
        paths = write_sprint1_budget_projection(
            projection,
            args.artifact_root.resolve(),
            refreeze=args.refreeze_budget,
        )
        _print_summary(projection)
        for path in paths:
            print(f"Wrote: {path}")
    elif args.generate_sprint1_sources:
        path = generate_sprint1_source_manifests(
            args.artifact_root.resolve(),
            args.graph_dir.resolve(),
            threads=args.threads,
            cpu_list=args.cpu_list,
            force=args.force_sources,
            refreeze=args.refreeze_sources,
        )
        print(f"Wrote: {path}")
    elif args.materialize_sprint1_natural:
        path = materialize_sprint1_natural_graphs(
            args.artifact_root.resolve(),
            args.graph_dir.resolve(),
            threads=args.threads,
            cpu_list=args.cpu_list,
            force=args.force_sources,
        )
        print(f"Wrote: {path}")
    elif args.prepare_sprint1_pilot:
        path = prepare_sprint1_pilot_execution(
            args.artifact_root.resolve(),
            args.graph_dir.resolve(),
            threads=args.threads,
            cpu_list=args.cpu_list,
            refreeze=args.refreeze_pilot_manifest,
        )
        payload = _load_json(path)
        bandwidths = [
            float(command["cap_implied_read_bandwidth_mib_s"])
            for command in payload["commands"]
            if command.get("cap_implied_read_bandwidth_mib_s")
        ]
        headrooms = [
            float(command["cap_headroom_seconds"])
            for command in payload["commands"]
        ]
        print(
            "Dry-run pilot commands: "
            f"{payload['command_count']}; "
            f"minimum cap headroom={min(headrooms):.2f}s"
        )
        if bandwidths:
            print(
                "Implied read bandwidth range: "
                f"{min(bandwidths):.2f}-"
                f"{max(bandwidths):.2f} MiB/s"
            )
        print(f"Wrote: {path}")
    elif args.validate_sprint1_executor:
        path = validate_sprint1_pilot_executor(
            args.artifact_root.resolve())
        print(f"Wrote: {path}")
    elif args.authorize_sprint1_pilot:
        if not args.authorization_reference:
            parser.error(
                "--authorize-sprint1-pilot requires "
                "--authorization-reference"
            )
        path = create_sprint1_execution_authorization(
            args.artifact_root.resolve(),
            authorization_reference=args.authorization_reference,
            refreeze=args.refreeze_authorization,
        )
        print(f"Wrote: {path}")
    else:
        if not args.authorization_reference:
            parser.error(
                "--execute-sprint1-pilot requires "
                "--authorization-reference"
            )
        path = execute_sprint1_pilot(
            args.artifact_root.resolve(),
            authorization_reference=args.authorization_reference,
        )
        print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
