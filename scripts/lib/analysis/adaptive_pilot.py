"""Content-bound analysis for the adaptive Sprint-1 pilot."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from scripts.lib.core.experiment_policy import ADAPTIVE_REUSE_REGIMES
from scripts.lib.pipeline.adaptive_pilot_contract import (
    bind_authorized_command,
    canonical_json_sha256,
    pilot_command_for_attempt,
    priming_command_for_session,
    validate_result_contract,
)

PILOT_ANALYSIS_SCHEMA = "adaptive-pilot-analysis/v1"
TIMING_PHASES = frozenset({
    "randomized-pilot",
    "natural-label-pilot",
})
ALLOWED_RESULT_STATES = frozenset({
    "",
    "timeout",
    "process-failure",
    "contract-violation",
})
HEADLINE_EXCLUDED_KERNELS = frozenset({"sssp"})


def _load_json(path: Path) -> Any:
    with path.open() as stream:
        return json.load(stream)


def _atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _geometric_mean(values: Iterable[float]) -> float:
    positive = [float(value) for value in values]
    if not positive or any(
        not math.isfinite(value) or value <= 0
        for value in positive
    ):
        raise ValueError("Geometric mean requires finite positive values")
    return math.exp(
        math.fsum(math.log(value) for value in positive)
        / len(positive)
    )


def _finite_nonnegative(value: Any, label: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return numeric


def _load_bound_result(
    command: Mapping[str, Any],
) -> tuple[dict[str, Any], Path] | None:
    path = Path(str(command["result_path"]))
    if not path.is_file():
        return None
    result = _load_json(path)
    validate_result_contract(result, command)
    state = str(result.get("error_kind") or "")
    if state not in ALLOWED_RESULT_STATES:
        raise ValueError(
            f"Unsupported adaptive pilot result state: {state}")
    _finite_nonnegative(
        result.get("duration_seconds", 0.0),
        "Adaptive pilot duration",
    )
    return result, path


def _validate_command_attempts(
    base_command: Mapping[str, Any],
    *,
    authorization_reference: str,
    manifest_sha256: str,
    input_artifacts: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[Path]]:
    attempts = []
    paths = []
    terminal = None
    terminal_seen = False
    missing_seen = False
    for attempt in base_command["retry_attempts"]:
        command = bind_authorized_command(
            pilot_command_for_attempt(base_command, int(attempt)),
            authorization_reference,
            manifest_sha256,
            input_artifacts,
        )
        loaded = _load_bound_result(command)
        if loaded is None:
            missing_seen = True
            continue
        result, path = loaded
        if missing_seen:
            raise ValueError(
                "Adaptive pilot retry attempts are non-contiguous: "
                + str(base_command["command_id"]))
        if terminal_seen:
            raise ValueError(
                "Adaptive pilot contains an attempt after the terminal "
                f"result: {base_command['command_id']}")
        attempts.append(result)
        paths.append(path)
        if result.get("error_kind") == "process-failure":
            continue
        terminal = result
        terminal_seen = True
    return terminal, attempts, paths


def load_selected_pilot_results(
    sprint_root: Path,
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Replay the executor contract and load each terminal result."""
    sprint_root = sprint_root.resolve()
    manifest_path = sprint_root / "pilot_execution_manifest.json"
    authorization_path = (
        sprint_root / "pilot_execution_authorization.json")
    completion_path = sprint_root / "pilot_execution_complete.json"
    manifest = _load_json(manifest_path)
    authorization = _load_json(authorization_path)
    if manifest.get("schema") != "adaptive-pilot-execution/v2":
        raise ValueError("Unsupported adaptive pilot manifest")
    manifest_sha256 = canonical_json_sha256(manifest)
    authorization_reference = str(
        authorization.get("authorization_reference") or "")
    if (
        authorization.get("schema")
            != "adaptive-pilot-execution-authorization/v2"
        or not authorization.get("execution_enabled")
        or not authorization_reference
        or authorization.get("execution_manifest_sha256")
            != manifest_sha256
        or authorization.get("command_count")
            != manifest.get("command_count")
    ):
        raise ValueError("Adaptive pilot authorization is invalid")

    completion = None
    if completion_path.is_file():
        completion = _load_json(completion_path)
        if (
            completion.get("schema")
                != "adaptive-pilot-execution-complete/v2"
            or completion.get("status") != "complete"
            or completion.get("authorization_reference")
                != authorization_reference
            or completion.get("execution_manifest_sha256")
                != manifest_sha256
            or completion.get("command_count")
                != manifest.get("command_count")
            or completion.get("priming_command_count")
                != manifest.get("priming_command_count")
        ):
            raise ValueError("Adaptive pilot completion is invalid")
    elif require_complete:
        raise RuntimeError("Adaptive pilot execution is incomplete")

    selected = {}
    missing = []
    all_attempts = []
    validated_paths: set[Path] = set()
    for base_command in manifest["commands"]:
        terminal, attempts, paths = _validate_command_attempts(
            base_command,
            authorization_reference=authorization_reference,
            manifest_sha256=manifest_sha256,
            input_artifacts=manifest["input_artifacts"],
        )
        all_attempts.extend(attempts)
        validated_paths.update(path.resolve() for path in paths)
        if terminal is None:
            missing.append(str(base_command["command_id"]))
        else:
            selected[str(base_command["command_id"])] = terminal

    priming_results = []
    if completion is not None:
        session_id = str(completion.get("execution_session_id") or "")
        if not session_id:
            raise ValueError("Adaptive pilot completion lacks a session ID")
        for base_command in manifest["priming_commands"]:
            command = bind_authorized_command(
                priming_command_for_session(base_command, session_id),
                authorization_reference,
                manifest_sha256,
                manifest["input_artifacts"],
            )
            loaded = _load_bound_result(command)
            if loaded is None:
                raise ValueError(
                    "Adaptive pilot completion lacks priming result: "
                    + str(base_command["command_id"]))
            result, path = loaded
            priming_results.append(result)
            validated_paths.add(path.resolve())
            if result.get("error_kind"):
                raise ValueError(
                    "Adaptive pilot completion contains failed priming")

    for path in (sprint_root / "pilot_runs").glob("**/result.json"):
        result = _load_json(path)
        if (
            result.get("authorization_reference")
                == authorization_reference
            and result.get("execution_manifest_sha256")
                == manifest_sha256
            and path.resolve() not in validated_paths
        ):
            if (
                completion is None
                and result.get("phase") == "page-cache-prime"
            ):
                continue
            raise ValueError(
                f"Adaptive pilot contains an unauthorized result path: {path}")

    if require_complete and missing:
        raise RuntimeError(
            "Adaptive pilot results are incomplete: "
            + ", ".join(missing[:5]))
    terminal_results = list(selected.values()) + priming_results
    if completion is not None:
        expected_result_count = (
            int(manifest["command_count"])
            + int(manifest["priming_command_count"])
        )
        if (
            completion.get("result_count") != expected_result_count
            or len(terminal_results) != expected_result_count
        ):
            raise ValueError(
                "Adaptive pilot completion/result count mismatch")
        states = Counter(
            str(result.get("error_kind") or "success")
            for result in terminal_results
        )
        if completion.get("result_states") != dict(states):
            raise ValueError(
                "Adaptive pilot completion state counts changed")
        retried = sum(
            int(result.get("attempt", 0)) > 0
            for result in terminal_results
        )
        if completion.get("retried_result_count") != retried:
            raise ValueError(
                "Adaptive pilot completion retry count changed")

    result_bindings = sorted(
        {
            str(path.resolve()): _file_sha256(path)
            for path in validated_paths
        }.items()
    )
    budget = _load_json(Path(str(manifest["budget_projection"])))
    return {
        "sprint_root": str(sprint_root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "authorization_path": str(authorization_path),
        "authorization_reference": authorization_reference,
        "completion_path": (
            str(completion_path) if completion is not None else None
        ),
        "completion": completion,
        "manifest": manifest,
        "budget": budget,
        "selected_results": selected,
        "priming_results": priming_results,
        "all_attempt_results": all_attempts,
        "missing_command_ids": missing,
        "result_bindings": result_bindings,
        "result_set_sha256": canonical_json_sha256(result_bindings),
    }


def aggregate_timing_cells(
    results: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Preserve process-indexed timing samples for every expected cell."""
    expected: dict[
        tuple[str, str, str, str],
        set[int],
    ] = defaultdict(set)
    for command in manifest["commands"]:
        if command.get("phase") not in TIMING_PHASES:
            continue
        key = (
            str(command["graph"]),
            str(command["labeling"]),
            str(command["kernel"]),
            str(command["arm"]),
        )
        expected[key].add(int(command["process_id"]))

    grouped: dict[
        tuple[str, str, str, str],
        list[Mapping[str, Any]],
    ] = defaultdict(list)
    for result in results:
        if result.get("phase") not in TIMING_PHASES:
            continue
        key = (
            str(result["graph"]),
            str(result["labeling"]),
            str(result["kernel"]),
            str(result["arm"]),
        )
        grouped[key].append(result)

    cells = []
    for key in sorted(expected):
        graph, labeling, kernel, arm = key
        rows = grouped.get(key, [])
        samples = []
        error_states = Counter()
        for row in sorted(
            rows, key=lambda item: int(item["process_id"])
        ):
            state = str(row.get("error_kind") or "success")
            error_states[state] += 1
            sample = {
                "process_id": int(row["process_id"]),
                "attempt": int(row.get("attempt", 0)),
                "state": state,
                "censored": bool(row.get("censored")),
            }
            if state == "success" and not row.get("censored"):
                extra = row.get("extra", {})
                if "complete_reorder_time" not in extra:
                    raise ValueError(
                        "Pilot timing result lacks complete reorder time: "
                        + str(row["command_id"]))
                kernel_seconds = _finite_nonnegative(
                    row.get("average_time"),
                    "Kernel time",
                )
                measured_reorder = _finite_nonnegative(
                    extra["complete_reorder_time"],
                    "Complete reorder time",
                )
                trial_times = [
                    _finite_nonnegative(value, "Trial time")
                    for value in extra.get("trial_times", [])
                ]
                if not trial_times:
                    raise ValueError(
                        "Pilot timing result lacks trial times: "
                        + str(row["command_id"]))
                if not math.isclose(
                    statistics.fmean(trial_times),
                    kernel_seconds,
                    rel_tol=5e-4,
                    abs_tol=5e-5,
                ):
                    raise ValueError(
                        "Pilot average time is not the equal-weight trial mean")
                sample.update({
                    "kernel_seconds": kernel_seconds,
                    "measured_complete_reorder_seconds":
                        measured_reorder,
                    "modeled_complete_reorder_seconds": (
                        0.0 if arm == "0" else measured_reorder
                    ),
                    "trial_count": len(trial_times),
                    "mapping_fingerprint":
                        extra.get("composed_mapping_fingerprint")
                        or extra.get("mapping_fingerprint"),
                })
            samples.append(sample)
        expected_ids = sorted(expected[key])
        successful_ids = sorted(
            sample["process_id"] for sample in samples
            if sample["state"] == "success"
            and not sample["censored"]
        )
        cells.append({
            "graph": graph,
            "labeling": labeling,
            "kernel": kernel,
            "arm": arm,
            "expected_process_ids": expected_ids,
            "successful_process_ids": successful_ids,
            "eligible": successful_ids == expected_ids,
            "error_states": dict(sorted(error_states.items())),
            "process_samples": samples,
            "mapping_fingerprint_count": len({
                sample.get("mapping_fingerprint")
                for sample in samples
                if sample.get("mapping_fingerprint")
            }),
        })
    return cells


def _selection_costs(
    results: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for result in results:
        if (
            result.get("phase") != "feature-cost-pilot"
            or result.get("error_kind")
            or result.get("censored")
        ):
            continue
        extra = result.get("extra", {})
        if "adaptive_selection_time" not in extra:
            raise ValueError(
                "Feature pilot lacks adaptive selection time")
        values[(
            str(result["graph"]),
            str(result["labeling"]),
        )].append(_finite_nonnegative(
            extra["adaptive_selection_time"],
            "Adaptive selection time",
        ))
    return {
        key: statistics.median(samples)
        for key, samples in values.items()
    }


def summarize_label_sensitivity(
    timing_cells: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Report natural/randomized differences without using labels as features."""
    grouped: dict[
        tuple[str, str, str],
        dict[str, Mapping[str, Any]],
    ] = defaultdict(dict)
    for cell in timing_cells:
        if not cell["eligible"]:
            continue
        grouped[(
            str(cell["graph"]),
            str(cell["kernel"]),
            str(cell["arm"]),
        )][str(cell["labeling"])] = cell
    rows = []
    for (graph, kernel, arm), labels in sorted(grouped.items()):
        if set(labels) != {"natural", "randomized"}:
            continue
        natural_kernel = _mean_process_cost(
            labels["natural"], "infinity")
        randomized_kernel = _mean_process_cost(
            labels["randomized"], "infinity")
        natural_reorder = statistics.fmean(
            float(sample["modeled_complete_reorder_seconds"])
            for sample in labels["natural"]["process_samples"]
        )
        randomized_reorder = statistics.fmean(
            float(sample["modeled_complete_reorder_seconds"])
            for sample in labels["randomized"]["process_samples"]
        )
        rows.append({
            "graph": graph,
            "kernel": kernel,
            "arm": arm,
            "natural_over_randomized_kernel_ratio":
                natural_kernel / randomized_kernel,
            "natural_over_randomized_reorder_ratio": (
                natural_reorder / randomized_reorder
                if randomized_reorder > 0 else None
            ),
        })
    return rows


def summarize_feature_overhead(
    timing_cells: Iterable[Mapping[str, Any]],
    selection_costs: Mapping[tuple[str, str], float],
) -> dict[str, Any]:
    original_pr = {
        (str(cell["graph"]), str(cell["labeling"])): cell
        for cell in timing_cells
        if (
            cell["kernel"] == "pr"
            and cell["arm"] == "0"
            and cell["eligible"]
        )
    }
    rows = []
    for key, selection_seconds in sorted(selection_costs.items()):
        original = original_pr.get(key)
        if original is None:
            continue
        original_kernel = _mean_process_cost(original, "infinity")
        rows.append({
            "graph": key[0],
            "labeling": key[1],
            "selection_seconds": selection_seconds,
            "original_pr_kernel_seconds": original_kernel,
            "selection_over_original_pr_ratio":
                selection_seconds / original_kernel,
        })
    expected = set(selection_costs)
    observed = {
        (row["graph"], row["labeling"]) for row in rows
    }
    return {
        "eligible": bool(rows) and observed == expected,
        "context_count": len(rows),
        "all_contexts_within_2pct": (
            bool(rows)
            and observed == expected
            and all(
                row["selection_over_original_pr_ratio"] <= 0.02
                for row in rows
            )
        ),
        "rows": rows,
    }


def summarize_peak_rss(
    results: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {
        str(command["command_id"])
        for command in manifest["commands"]
        if command.get("phase") == "rss-pilot"
    }
    rows = []
    seen = set()
    for result in results:
        if result.get("phase") != "rss-pilot":
            continue
        command_id = str(result["command_id"])
        seen.add(command_id)
        row = {
            "command_id": command_id,
            "graph": result["graph"],
            "arm": result["arm"],
            "state": str(result.get("error_kind") or "success"),
            "censored": bool(result.get("censored")),
            "peak_rss_kib": result.get(
                "extra", {}).get("peak_rss_kib"),
        }
        rows.append(row)
    eligible = (
        bool(expected)
        and seen == expected
        and all(
            row["state"] == "success"
            and not row["censored"]
            and isinstance(row["peak_rss_kib"], int)
            and row["peak_rss_kib"] > 0
            for row in rows
        )
    )
    return {
        "eligible": eligible,
        "expected_count": len(expected),
        "observed_count": len(rows),
        "max_peak_rss_kib": (
            max(row["peak_rss_kib"] for row in rows)
            if eligible and rows else None
        ),
        "rows": sorted(
            rows, key=lambda row: (row["graph"], row["arm"])),
    }


def _cache_hierarchy_lookups(stats: Mapping[str, Any]) -> int:
    try:
        values = (
            int(stats["total_accesses"]),
            int(stats["L1"]["misses"]),
            int(stats["L2"]["misses"]),
            int(stats["L3"]["misses"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Cache pilot has invalid hierarchy statistics") from error
    if any(value < 0 for value in values):
        raise ValueError("Cache hierarchy statistics must be non-negative")
    return sum(values)


def _cache_capacity_mib(result: Mapping[str, Any]) -> int:
    match = re.search(
        r"\|(\d+)MiB\|",
        str(result.get("command_id") or ""),
    )
    if match is None:
        raise ValueError("Cache pilot command capacity changed")
    return int(match.group(1))


def summarize_cache_repricing(
    results: Iterable[Mapping[str, Any]],
    budget: Mapping[str, Any],
) -> dict[str, Any]:
    expected_rows = budget.get("cache_micro_pilot_rows", [])
    expected = {
        (
            str(row["graph"]),
            str(row["kernel"]),
            str(row["arm"]),
            int(row["capacities_mib"]),
            row.get("source_index"),
        ): row
        for row in expected_rows
    }
    rows = []
    observed = {}
    for result in results:
        if result.get("phase") != "cache-micro-pilot":
            continue
        source_token = str(result["command_id"]).rsplit("|s", 1)[-1]
        source_index = (
            None if source_token == "None" else int(source_token)
        )
        key = (
            str(result["graph"]),
            str(result["kernel"]),
            str(result["arm"]),
            _cache_capacity_mib(result),
            source_index,
        )
        expected_row = expected.get(key)
        if expected_row is None:
            raise ValueError(
                f"Unexpected cache repricing result: {key}")
        if key in observed:
            raise ValueError(
                f"Duplicate cache repricing result: {key}")
        state = str(result.get("error_kind") or "success")
        stats = result.get("extra", {}).get("cache_stats")
        hierarchy = (
            _cache_hierarchy_lookups(stats)
            if state == "success"
            and not result.get("censored")
            and isinstance(stats, Mapping)
            else None
        )
        row = {
            "graph": key[0],
            "kernel": key[1],
            "arm": key[2],
            "capacity_mib": key[3],
            "source_index": key[4],
            "probe_role": expected_row["probe_role"],
            "state": state,
            "censored": bool(result.get("censored")),
            "duration_seconds": _finite_nonnegative(
                result.get("duration_seconds", 0.0),
                "Cache pilot duration",
            ),
            "hierarchy_lookups": hierarchy,
        }
        observed[key] = row
        rows.append(row)
    eligible = (
        bool(expected)
        and set(observed) == set(expected)
        and all(
            row["state"] == "success"
            and not row["censored"]
            and row["hierarchy_lookups"] is not None
            and row["duration_seconds"] > 0
            for row in rows
        )
    )
    summary: dict[str, Any] = {
        "eligible": eligible,
        "expected_count": len(expected),
        "observed_count": len(rows),
        "rows": sorted(
            rows,
            key=lambda row: (
                row["graph"],
                row["kernel"],
                row["arm"],
                row["capacity_mib"],
                -1 if row["source_index"] is None
                else row["source_index"],
            ),
        ),
        "separable_model_eligible": False,
    }
    if not eligible:
        return summary

    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_role[row["probe_role"]].append(row)
    centers = by_role.get("star-center", [])
    interactions = by_role.get("graph-kernel-interaction", [])
    kernel_rows = by_role.get("kernel-factor", [])
    if len(centers) != 1 or len(interactions) != 1 or len(kernel_rows) != 1:
        raise ValueError("Cache repricing star roles changed")
    center = centers[0]
    center_seconds = center["duration_seconds"]
    graph_factors = {
        row["graph"]: row["duration_seconds"] / center_seconds
        for row in by_role.get("graph-factor", [])
    }
    kernel_factor = (
        kernel_rows[0]["duration_seconds"] / center_seconds)
    interaction = interactions[0]
    predicted_interaction = (
        center_seconds
        * graph_factors[interaction["graph"]]
        * kernel_factor
    )
    interaction_residual = abs(
        interaction["duration_seconds"]
        / predicted_interaction - 1.0
    )
    arm_factors = {
        row["arm"]: row["duration_seconds"] / center_seconds
        for row in by_role.get("arm-factor", [])
    }
    capacity_factors = {
        str(row["capacity_mib"]):
            row["duration_seconds"] / center_seconds
        for row in by_role.get("capacity-factor", [])
    }
    source_rows = by_role.get("source-dispersion", [])
    source_dispersion = (
        source_rows[0]["duration_seconds"]
        / kernel_rows[0]["duration_seconds"]
        if len(source_rows) == 1 else None
    )
    ranking_groups: dict[
        tuple[str, str, int, int | None],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in rows:
        ranking_groups[(
            row["graph"],
            row["kernel"],
            row["capacity_mib"],
            row["source_index"],
        )].append(row)
    ranking_rows = []
    for key, group in sorted(ranking_groups.items()):
        if len(group) < 2:
            continue
        best = min(
            int(row["hierarchy_lookups"]) for row in group)
        ranking_rows.extend({
            "graph": key[0],
            "kernel": key[1],
            "capacity_mib": key[2],
            "source_index": key[3],
            "arm": row["arm"],
            "hierarchy_lookup_ratio_to_best":
                int(row["hierarchy_lookups"]) / best,
        } for row in group)
    max_residual = float(
        budget["policy"][
            "cache_repricing_max_interaction_residual"])
    summary.update({
        "center_duration_seconds": center_seconds,
        "graph_runtime_factors": graph_factors,
        "kernel_runtime_factor": kernel_factor,
        "arm_runtime_factors": arm_factors,
        "capacity_runtime_factors": capacity_factors,
        "source_runtime_ratio": source_dispersion,
        "interaction_residual": interaction_residual,
        "maximum_interaction_residual": max_residual,
        "separable_model_eligible":
            interaction_residual <= max_residual,
        "hierarchy_ranking_rows": ranking_rows,
    })
    return summary


def _sample_cost(
    sample: Mapping[str, Any],
    reuse_regime: int | str,
) -> float:
    if reuse_regime == "infinity":
        return float(sample["kernel_seconds"])
    return (
        float(sample["modeled_complete_reorder_seconds"])
        + int(reuse_regime) * float(sample["kernel_seconds"])
    )


def _mean_process_cost(
    cell: Mapping[str, Any],
    reuse_regime: int | str,
    process_ids: set[int] | None = None,
) -> float:
    samples = [
        sample for sample in cell["process_samples"]
        if (
            sample["state"] == "success"
            and not sample["censored"]
            and (
                process_ids is None
                or int(sample["process_id"]) in process_ids
            )
        )
    ]
    if not samples:
        raise ValueError("Policy evaluation has no eligible process samples")
    return statistics.fmean(
        _sample_cost(sample, reuse_regime)
        for sample in samples
    )


def _context_crossfit_oracle(
    arm_cells: Mapping[str, Mapping[str, Any]],
    *,
    reuse_regime: int | str,
    portfolio_order: tuple[str, ...],
) -> dict[str, Any]:
    process_ids = set.intersection(*(
        set(cell["expected_process_ids"])
        for cell in arm_cells.values()
    ))
    odd = {process_id for process_id in process_ids if process_id % 2 == 1}
    even = process_ids - odd
    if not odd or not even:
        raise ValueError(
            "Oracle cross-fitting requires odd and even process blocks")

    def choose(selection_ids: set[int]) -> str:
        costs = {
            arm: _mean_process_cost(
                arm_cells[arm], reuse_regime, selection_ids)
            for arm in portfolio_order
        }
        return min(
            portfolio_order,
            key=lambda arm: (costs[arm], portfolio_order.index(arm)),
        )

    odd_choice = choose(odd)
    even_choice = choose(even)
    crossfit_cost = statistics.fmean((
        _mean_process_cost(
            arm_cells[odd_choice], reuse_regime, even),
        _mean_process_cost(
            arm_cells[even_choice], reuse_regime, odd),
    ))
    all_costs = {
        arm: _mean_process_cost(cell, reuse_regime)
        for arm, cell in arm_cells.items()
    }
    naive_arm = min(
        portfolio_order,
        key=lambda arm: (
            all_costs[arm], portfolio_order.index(arm)),
    )
    return {
        "crossfit_oracle_seconds": crossfit_cost,
        "odd_selected_arm": odd_choice,
        "even_selected_arm": even_choice,
        "naive_oracle_arm": naive_arm,
        "naive_oracle_seconds": all_costs[naive_arm],
        "arm_seconds": all_costs,
    }


def _logo_static_regret(
    contexts: list[dict[str, Any]],
    portfolio_order: tuple[str, ...],
) -> dict[str, Any]:
    graphs = sorted({row["graph"] for row in contexts})
    heldout_rows = []
    for heldout in graphs:
        training = [row for row in contexts if row["graph"] != heldout]
        testing = [row for row in contexts if row["graph"] == heldout]
        if not training or not testing:
            continue
        training_ratios = {}
        for arm in portfolio_order:
            training_ratios[arm] = _geometric_mean(
                row["arm_seconds"][arm]
                / row["arm_seconds"]["0"]
                for row in training
            )
        selected_arm = min(
            portfolio_order,
            key=lambda arm: (
                training_ratios[arm],
                portfolio_order.index(arm),
            ),
        )
        for row in testing:
            heldout_rows.append({
                "graph": heldout,
                "kernel": row["kernel"],
                "selected_arm": selected_arm,
                "regret_ratio":
                    row["arm_seconds"][selected_arm]
                    / row["crossfit_oracle_seconds"],
            })
    return {
        "fold_count": len(graphs),
        "heldout_context_count": len(heldout_rows),
        "geomean_regret_ratio": (
            _geometric_mean(
                row["regret_ratio"] for row in heldout_rows)
            if heldout_rows else None
        ),
        "heldout_rows": heldout_rows,
    }


def evaluate_policy_headroom(
    timing_cells: Iterable[Mapping[str, Any]],
    *,
    portfolio_order: tuple[str, ...],
    selection_costs: Mapping[tuple[str, str], float],
) -> dict[str, Any]:
    """Evaluate winner's-curse-safe oracle headroom on randomized labels."""
    cells = list(timing_cells)
    expected_primary = [
        cell for cell in cells
        if cell["labeling"] == "randomized"
    ]
    ineligible = [
        {
            "graph": cell["graph"],
            "kernel": cell["kernel"],
            "arm": cell["arm"],
            "error_states": cell["error_states"],
        }
        for cell in expected_primary
        if not cell["eligible"]
    ]
    observed_arms = {
        str(cell["arm"]) for cell in expected_primary
    }
    if observed_arms != set(portfolio_order):
        ineligible.append({
            "reason": "portfolio-coverage",
            "expected": list(portfolio_order),
            "observed": sorted(observed_arms),
        })
    if ineligible:
        return {
            "headroom_eligible": False,
            "ineligible_cells": ineligible,
            "by_reuse": {},
        }

    by_context: dict[
        tuple[str, str],
        dict[str, Mapping[str, Any]],
    ] = defaultdict(dict)
    for cell in expected_primary:
        by_context[(
            str(cell["graph"]),
            str(cell["kernel"]),
        )][str(cell["arm"])] = cell
    incomplete_contexts = [
        context for context, arm_cells in by_context.items()
        if set(arm_cells) != set(portfolio_order)
    ]
    if incomplete_contexts:
        return {
            "headroom_eligible": False,
            "ineligible_cells": [
                {"context": list(context)}
                for context in incomplete_contexts
            ],
            "by_reuse": {},
        }

    summaries = {}
    for reuse_regime in ADAPTIVE_REUSE_REGIMES:
        contexts = []
        for (graph, kernel), arm_cells in sorted(by_context.items()):
            oracle = _context_crossfit_oracle(
                arm_cells,
                reuse_regime=reuse_regime,
                portfolio_order=portfolio_order,
            )
            selection_seconds = (
                0.0
                if reuse_regime == "infinity"
                else selection_costs.get((graph, "randomized"))
            )
            contexts.append({
                "graph": graph,
                "kernel": kernel,
                **oracle,
                "selection_seconds": selection_seconds,
                "net_oracle_seconds": (
                    oracle["crossfit_oracle_seconds"] + selection_seconds
                    if selection_seconds is not None else None
                ),
            })
        headline = [
            row for row in contexts
            if row["kernel"] not in HEADLINE_EXCLUDED_KERNELS
        ]
        logo = _logo_static_regret(headline, portfolio_order)
        net_rows = [
            row for row in headline
            if row["net_oracle_seconds"] is not None
        ]
        net_headroom = None
        if (
            logo["heldout_rows"]
            and len(net_rows) == len(headline)
        ):
            net_by_context = {
                (row["graph"], row["kernel"]):
                    row["net_oracle_seconds"]
                for row in net_rows
            }
            net_headroom = _geometric_mean(
                (
                    next(
                        context["arm_seconds"][heldout["selected_arm"]]
                        for context in headline
                        if context["graph"] == heldout["graph"]
                        and context["kernel"] == heldout["kernel"]
                    )
                    / net_by_context[(
                        heldout["graph"], heldout["kernel"])]
                )
                for heldout in logo["heldout_rows"]
            )
        oracle_counts = Counter(
            arm
            for row in headline
            for arm in (
                row["odd_selected_arm"],
                row["even_selected_arm"],
            )
        )
        summaries[str(reuse_regime)] = {
            "context_count": len(contexts),
            "headline_context_count": len(headline),
            "logo_best_static": logo,
            "crossfit_oracle_arm_counts":
                dict(sorted(oracle_counts.items())),
            "crossfit_oracle_arm_diversity": len(oracle_counts),
            "best_static_vs_net_oracle_geomean_ratio":
                net_headroom,
            "naive_oracle_bias_geomean_ratio": _geometric_mean(
                row["crossfit_oracle_seconds"]
                / row["naive_oracle_seconds"]
                for row in headline
            ),
            "contexts": contexts,
        }
    return {
        "headroom_eligible": True,
        "ineligible_cells": [],
        "reuse_regimes": list(ADAPTIVE_REUSE_REGIMES),
        "headline_excluded_kernels":
            sorted(HEADLINE_EXCLUDED_KERNELS),
        "by_reuse": summaries,
    }


def build_pilot_analysis(
    sprint_root: Path,
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    bundle = load_selected_pilot_results(
        sprint_root, require_complete=require_complete)
    selected = list(bundle["selected_results"].values())
    terminal = selected + bundle["priming_results"]
    state_counts = Counter(
        str(result.get("error_kind") or "success")
        for result in terminal
    )
    phase_counts = Counter(
        str(result.get("phase")) for result in selected)
    timing_cells = aggregate_timing_cells(
        selected, bundle["manifest"])
    selection_costs = _selection_costs(selected)
    label_sensitivity = summarize_label_sensitivity(timing_cells)
    feature_overhead = summarize_feature_overhead(
        timing_cells, selection_costs)
    peak_rss = summarize_peak_rss(
        selected, bundle["manifest"])
    cache_repricing = summarize_cache_repricing(
        selected, bundle["budget"])
    portfolio_order = tuple(
        bundle["budget"]["policy"]["deployable_pilot_arms"])
    headroom = evaluate_policy_headroom(
        timing_cells,
        portfolio_order=portfolio_order,
        selection_costs=selection_costs,
    )
    status = "incomplete"
    if bundle["completion"] is not None:
        if set(state_counts) <= {"success"}:
            status = "complete-clean"
        elif set(state_counts) <= {"success", "timeout"}:
            status = "complete-with-censoring"
        else:
            status = "complete-with-errors"
    pilot_gates = {
        "timing_headroom": bool(headroom["headroom_eligible"]),
        "feature_overhead": bool(
            feature_overhead["all_contexts_within_2pct"]),
        "peak_rss": bool(peak_rss["eligible"]),
        "cache_repricing": bool(
            cache_repricing["separable_model_eligible"]),
    }
    return {
        "schema": PILOT_ANALYSIS_SCHEMA,
        "status": status,
        "measurement_mode": "diagnostic-adaptive",
        "claim_eligible": False,
        "headroom_eligible": bool(
            status == "complete-clean"
            and headroom["headroom_eligible"]
        ),
        "full_collection_gate_eligible": bool(
            status == "complete-clean"
            and all(pilot_gates.values())
        ),
        "pilot_gates": pilot_gates,
        "sprint_root": bundle["sprint_root"],
        "execution_manifest_sha256": bundle["manifest_sha256"],
        "authorization_reference":
            bundle["authorization_reference"],
        "completion_path": bundle["completion_path"],
        "result_set_sha256": bundle["result_set_sha256"],
        "expected_command_count":
            int(bundle["manifest"]["command_count"]),
        "expected_priming_command_count":
            int(bundle["manifest"]["priming_command_count"]),
        "selected_result_count": len(selected),
        "priming_result_count": len(bundle["priming_results"]),
        "validated_attempt_count":
            len(bundle["all_attempt_results"]),
        "missing_command_ids": bundle["missing_command_ids"],
        "state_counts": dict(sorted(state_counts.items())),
        "phase_counts": dict(sorted(phase_counts.items())),
        "terminal_command_hours": math.fsum(
            float(result.get("duration_seconds", 0.0))
            for result in terminal
        ) / 3600.0,
        "total_consumed_hours": math.fsum(
            float(result.get("duration_seconds", 0.0))
            for result in (
                bundle["all_attempt_results"]
                + bundle["priming_results"]
            )
        ) / 3600.0,
        "selection_seconds_by_graph_label": {
            f"{graph}|{labeling}": value
            for (graph, labeling), value
            in sorted(selection_costs.items())
        },
        "label_sensitivity": label_sensitivity,
        "feature_overhead": feature_overhead,
        "peak_rss": peak_rss,
        "cache_repricing": cache_repricing,
        "timing_cells": timing_cells,
        "policy_headroom": headroom,
    }


def write_pilot_analysis(
    sprint_root: Path,
    *,
    require_complete: bool = True,
) -> Path:
    payload = build_pilot_analysis(
        sprint_root, require_complete=require_complete)
    path = sprint_root.resolve() / "pilot_analysis.json"
    _atomic_json(payload, path)
    return path
