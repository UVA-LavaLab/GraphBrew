"""Adaptive source-driven measurement contract."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping
import statistics

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_POLICY_PATH = (
    PROJECT_ROOT
    / "bench/include/graphbrew/reorder/adaptive_source_policy.def"
)
_SOURCE_POLICY_PATTERN = re.compile(
    r'^GRAPHBREW_ADAPTIVE_SOURCE_POLICY\('
    r'"([^"]+)",\s*(\d+),\s*(\d+),\s*([0-9.]+)\)$'
)


def _load_source_policy() -> tuple[str, int, int, float]:
    lines = [
        line.strip()
        for line in SOURCE_POLICY_PATH.read_text().splitlines()
        if line.strip()
    ]
    if len(lines) != 1:
        raise RuntimeError("Adaptive source policy must have one definition")
    match = _SOURCE_POLICY_PATTERN.fullmatch(lines[0])
    if match is None:
        raise RuntimeError(
            f"Invalid adaptive source policy: {lines[0]}")
    policy_id, count, seed, reachability = match.groups()
    return policy_id, int(count), int(seed), float(reachability)


(
    ADAPTIVE_SOURCE_POLICY_ID,
    ADAPTIVE_SOURCE_COUNT,
    ADAPTIVE_SOURCE_SEED,
    ADAPTIVE_SOURCE_MIN_REACHABILITY,
) = _load_source_policy()
ADAPTIVE_PORTFOLIO_VERIFICATION_GATE_ID = "96954491"
SOURCE_DRIVEN_KERNELS = frozenset({"bfs", "bc", "sssp"})


def adaptive_source_record_eligible(record: Mapping) -> bool:
    """Return true only for post-Sprint-0 source-driven measurements."""
    benchmark = record.get("benchmark")
    if benchmark not in SOURCE_DRIVEN_KERNELS:
        return True
    if record.get("source_policy_id") != ADAPTIVE_SOURCE_POLICY_ID:
        return False
    sources = record.get("source_trials")
    if not isinstance(sources, list) or len(sources) < ADAPTIVE_SOURCE_COUNT:
        return False
    required = {
        "process_id",
        "source_id",
        "source_internal",
        "source_out_degree",
        "repetition_index",
        "measurement_mode",
    }
    return all(
        isinstance(source, Mapping)
        and required.issubset(source)
        and source["source_out_degree"] > 0
        for source in sources
    )


def require_adaptive_source_record(record: Mapping) -> None:
    if not adaptive_source_record_eligible(record):
        raise ValueError(
            "Source-driven adaptive record predates or violates "
            f"{ADAPTIVE_SOURCE_POLICY_ID}"
        )


def require_portfolio_gate_coverage(
    rows,
    graph_names,
    arm_specs,
) -> None:
    passed = {
        (str(row.get("graph")), str(row.get("algo_key")))
        for row in rows
        if row.get("gate_id") == ADAPTIVE_PORTFOLIO_VERIFICATION_GATE_ID
        and row.get("verification_state") == "pass"
    }
    missing = [
        (graph, arm)
        for graph in graph_names
        for arm in arm_specs
        if (graph, arm) not in passed
    ]
    if missing:
        raise ValueError(
            "Adaptive portfolio lacks verification-gate coverage for "
            + ", ".join(f"{graph}/{arm}" for graph, arm in missing[:8])
        )


def aggregate_source_trial_times(
    trial_times,
    source_originals,
    measurement_mode: str,
) -> dict:
    """Aggregate source trials without mixing cold and warm-block samples."""
    if len(trial_times) != len(source_originals) or not trial_times:
        raise ValueError("Source trial vectors must be non-empty and aligned")
    grouped = {}
    for source, value in zip(source_originals, trial_times):
        grouped.setdefault(int(source), []).append(float(value))
    if measurement_mode == "cold-process":
        if any(len(values) != 1 for values in grouped.values()):
            raise ValueError(
                "cold-process mode requires one trial per source/process")
        return {
            "cell_time": statistics.fmean(
                values[0] for values in grouped.values()),
            "cold_first_times": [
                values[0] for values in grouped.values()
            ],
            "warm_times": [],
        }
    if measurement_mode == "warm-block":
        if any(len(values) < 2 for values in grouped.values()):
            raise ValueError(
                "warm-block mode requires a cold first trial and warm repeats")
        cold = [values[0] for values in grouped.values()]
        warm = [
            statistics.median(values[1:])
            for values in grouped.values()
        ]
        return {
            "cell_time": statistics.fmean(warm),
            "cold_first_times": cold,
            "warm_times": warm,
        }
    raise ValueError(f"Unknown measurement mode: {measurement_mode}")
