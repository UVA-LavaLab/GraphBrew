#!/usr/bin/env python3
"""Build a reproducible certificate for GraphBrew composition diversity.

The certificate separates three questions:

1. Does the best GraphBrew composition vary by graph and kernel?
2. How much post-selected headroom exists over one fixed composition?
3. Does a graph-held-out rule recover that headroom?

The first two are design-space evidence. The third is the deployment gate.
They must not be conflated.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Callable, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.vldb.config import GRAPH_TYPE_GROUPS  # noqa: E402


DEFAULT_ARTIFACT_ROOT = Path(
    os.environ.get(
        "GRAPHBREW_PAPER_ARTIFACT_ROOT",
        "/media/NVMeData/00_GraphDatasets/GraphBrew/artifacts",
    )
)
DEFAULT_ATLAS_ROOT = DEFAULT_ARTIFACT_ROOT / "composition_oracle_atlas_v1"
ORIGINAL_SPEC = "0"

Cell = tuple[str, str]
Assignment = Callable[[str, str], str]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometric_mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized or any(
        not math.isfinite(value) or value <= 0
        for value in materialized
    ):
        raise ValueError("Geometric mean requires finite positive values")
    return math.exp(
        sum(math.log(value) for value in materialized)
        / len(materialized)
    )


def canonical_spec(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"Invalid algorithm identifier: {value!r}")
    return str(value)


def load_protocol(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("schema") not in {
        "graphbrew-composition-oracle-atlas/v2",
        "graphbrew-composition-holdout-protocol/v1",
        "graphbrew-composition-sealed-confirmation-protocol/v1",
        "graphbrew-mechanism-factorial-protocol/v1",
        "graphbrew-multifidelity-screen-protocol/v1",
    }:
        raise ValueError(f"Unsupported atlas protocol: {payload.get('schema')}")
    return payload


def build_matrix(
    rows: list[dict],
    *,
    graphs: list[str],
    kernels: list[str],
    specs: set[str],
) -> dict[tuple[str, str, str], float]:
    matrix: dict[tuple[str, str, str], float] = {}
    for row in rows:
        spec = canonical_spec(row.get("algo_id"))
        if spec not in specs:
            continue
        graph = str(row.get("graph", ""))
        kernel = str(row.get("benchmark", ""))
        if graph not in graphs or kernel not in kernels:
            continue
        value = row.get("median_time")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(
                f"Invalid median time for {graph}/{kernel}/{spec}: {value!r}"
            )
        key = (graph, kernel, spec)
        if key in matrix:
            raise ValueError(f"Duplicate timing cell: {key}")
        matrix[key] = float(value)

    missing = [
        (graph, kernel, spec)
        for graph in graphs
        for kernel in kernels
        for spec in sorted(specs)
        if (graph, kernel, spec) not in matrix
    ]
    if missing:
        raise ValueError(
            f"Incomplete composability matrix: {len(missing)} missing cells; "
            f"first={missing[0]}"
        )
    return matrix


def choose_arm(
    cells: Iterable[Cell],
    candidates: list[str],
    matrix: dict[tuple[str, str, str], float],
) -> str:
    materialized = list(cells)
    if not materialized:
        raise ValueError("Cannot select an arm from an empty cell set")
    return min(
        candidates,
        key=lambda spec: (
            geometric_mean(
                matrix[(graph, kernel, spec)]
                for graph, kernel in materialized
            ),
            spec,
        ),
    )


def evaluate_assignment(
    *,
    cells: list[Cell],
    assignment: Assignment,
    matrix: dict[tuple[str, str, str], float],
    comparator_specs: list[str],
    best_fixed_spec: str,
) -> dict:
    comparator_ratios: list[float] = []
    comparator_original_ratios: list[float] = []
    best_fixed_ratios: list[float] = []
    wins = ties = losses = 0
    selected = Counter()
    baseline_ratios = {
        spec: [] for spec in (*comparator_specs, ORIGINAL_SPEC)
    }
    policy_seconds = 0.0
    baseline_seconds = {
        spec: 0.0 for spec in (*comparator_specs, ORIGINAL_SPEC)
    }

    for graph, kernel in cells:
        spec = assignment(graph, kernel)
        selected[spec] += 1
        policy_time = matrix[(graph, kernel, spec)]
        comparator_time = min(
            matrix[(graph, kernel, baseline)]
            for baseline in comparator_specs
        )
        comparator_original_time = min(
            comparator_time,
            matrix[(graph, kernel, ORIGINAL_SPEC)],
        )
        ratio = comparator_time / policy_time
        comparator_ratios.append(ratio)
        comparator_original_ratios.append(
            comparator_original_time / policy_time
        )
        best_fixed_ratios.append(
            matrix[(graph, kernel, best_fixed_spec)] / policy_time
        )
        policy_seconds += policy_time
        for baseline in baseline_ratios:
            baseline_time = matrix[(graph, kernel, baseline)]
            baseline_ratios[baseline].append(
                baseline_time / policy_time
            )
            baseline_seconds[baseline] += baseline_time
        if ratio > 1.02:
            wins += 1
        elif ratio < 0.98:
            losses += 1
        else:
            ties += 1

    return {
        "fastest_comparator_over_policy_gm": geometric_mean(
            comparator_ratios
        ),
        "fastest_comparator_or_original_over_policy_gm": geometric_mean(
            comparator_original_ratios
        ),
        "best_fixed_graphbrew_over_policy_gm": geometric_mean(
            best_fixed_ratios
        ),
        "win_tie_loss_2pct_vs_fastest_comparator": {
            "win": wins,
            "tie": ties,
            "loss": losses,
        },
        "selected_arm_counts": dict(sorted(selected.items())),
        "baseline_over_policy_kernel_gm": {
            spec: geometric_mean(ratios)
            for spec, ratios in baseline_ratios.items()
        },
        "baseline_over_policy_summed_kernel_seconds": {
            spec: seconds / policy_seconds
            for spec, seconds in baseline_seconds.items()
        },
    }


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Percentile requires at least one value")
    index = round(probability * (len(sorted_values) - 1))
    return sorted_values[index]


def bootstrap_graph_geomean(
    graph_values: dict[str, float],
    *,
    resamples: int,
    seed: int = 20260829,
) -> tuple[float, float]:
    import random

    values = list(graph_values.values())
    if resamples <= 0:
        raise ValueError("Bootstrap resamples must be positive")
    rng = random.Random(seed)
    samples = sorted(
        geometric_mean(
            values[rng.randrange(len(values))]
            for _ in values
        )
        for _ in range(resamples)
    )
    return percentile(samples, 0.025), percentile(samples, 0.975)


def mapping_seconds(
    protocol: dict,
    graph: str,
    spec: str,
) -> float:
    if spec == ORIGINAL_SPEC:
        return 0.0
    execution = protocol.get("execution")
    if not isinstance(execution, dict) or not execution.get("artifact_root"):
        raise ValueError("Confirmation protocol lacks an artifact root")
    meta_path = (
        Path(execution["artifact_root"])
        / "vldb_mappings"
        / graph
        / f"{spec.replace(':', '_')}.json"
    )
    meta = json.loads(meta_path.read_text())
    return sum(
        float(meta[field])
        for field in (
            "reorder_core_time",
            "reorder_validation_time",
            "reorder_apply_time",
        )
    )


def scalar_metric(value: object, *, name: str) -> float:
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"{name} must contain exactly one value")
        value = value[0]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"Invalid {name}: {value!r}")
    return float(value)


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while (
            end < len(order)
            and values[order[end]] == values[order[start]]
        ):
            end += 1
        rank = (start + end - 1) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def pearson(values_x: list[float], values_y: list[float]) -> float | None:
    if len(values_x) != len(values_y) or len(values_x) < 2:
        return None
    mean_x = statistics.fmean(values_x)
    mean_y = statistics.fmean(values_y)
    numerator = sum(
        (x - mean_x) * (y - mean_y)
        for x, y in zip(values_x, values_y)
    )
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in values_x)
        * sum((y - mean_y) ** 2 for y in values_y)
    )
    return numerator / denominator if denominator else None


def spearman(values_x: list[float], values_y: list[float]) -> float | None:
    return pearson(average_ranks(values_x), average_ranks(values_y))


def kendall_tau_b(
    values_x: list[float],
    values_y: list[float],
) -> float | None:
    if len(values_x) != len(values_y) or len(values_x) < 2:
        return None
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for left in range(len(values_x)):
        for right in range(left + 1, len(values_x)):
            delta_x = values_x[left] - values_x[right]
            delta_y = values_y[left] - values_y[right]
            if delta_x == 0 and delta_y == 0:
                continue
            if delta_x == 0:
                ties_x += 1
            elif delta_y == 0:
                ties_y += 1
            elif delta_x * delta_y > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_x)
        * (concordant + discordant + ties_y)
    )
    if denominator == 0:
        return None
    return (concordant - discordant) / denominator


def ranking_fidelity(
    reference: dict[str, float],
    candidate: dict[str, float],
    *,
    shortlist_size: int,
) -> dict:
    if set(reference) != set(candidate) or not reference:
        raise ValueError("Ranking fidelity requires matching non-empty arms")
    specs = sorted(reference)
    if shortlist_size < 1 or shortlist_size > len(specs):
        raise ValueError("Invalid shortlist size")
    for label, values in (
        ("reference", reference),
        ("candidate", candidate),
    ):
        if any(
            not math.isfinite(value) or value <= 0
            for value in values.values()
        ):
            raise ValueError(f"{label} ranking contains invalid values")
    reference_order = sorted(
        specs,
        key=lambda spec: (reference[spec], spec),
    )
    candidate_order = sorted(
        specs,
        key=lambda spec: (candidate[spec], spec),
    )
    reference_best = reference[reference_order[0]]
    shortlist = candidate_order[:shortlist_size]
    best_shortlisted = min(reference[spec] for spec in shortlist)
    return {
        "kendall_tau_b": kendall_tau_b(
            [reference[spec] for spec in specs],
            [candidate[spec] for spec in specs],
        ),
        "spearman": spearman(
            [reference[spec] for spec in specs],
            [candidate[spec] for spec in specs],
        ),
        "top1_match": candidate_order[0] == reference_order[0],
        "reference_winner_in_shortlist":
            reference_order[0] in shortlist,
        "shortlist_overlap": (
            len(
                set(reference_order[:shortlist_size])
                & set(shortlist)
            )
            / shortlist_size
        ),
        "candidate_top1_reference_regret":
            reference[candidate_order[0]] / reference_best - 1.0,
        "shortlist_reference_regret":
            best_shortlisted / reference_best - 1.0,
        "reference_order": reference_order,
        "candidate_order": candidate_order,
    }


def aggregate_ranking_fidelity(records: list[dict]) -> dict:
    if not records:
        raise ValueError("Ranking aggregation requires records")
    taus = [
        record["kendall_tau_b"]
        for record in records
        if record["kendall_tau_b"] is not None
    ]
    spearmans = [
        record["spearman"]
        for record in records
        if record["spearman"] is not None
    ]
    top1_regrets = sorted(
        record["candidate_top1_reference_regret"]
        for record in records
    )
    shortlist_regrets = sorted(
        record["shortlist_reference_regret"]
        for record in records
    )
    return {
        "contexts": len(records),
        "kendall_tau_b_mean": (
            statistics.fmean(taus) if taus else None
        ),
        "kendall_tau_b_median": (
            statistics.median(taus) if taus else None
        ),
        "kendall_tau_b_min": min(taus) if taus else None,
        "spearman_mean": (
            statistics.fmean(spearmans) if spearmans else None
        ),
        "top1_accuracy": statistics.fmean(
            float(record["top1_match"]) for record in records
        ),
        "reference_winner_in_shortlist_rate": statistics.fmean(
            float(record["reference_winner_in_shortlist"])
            for record in records
        ),
        "shortlist_overlap_mean": statistics.fmean(
            record["shortlist_overlap"] for record in records
        ),
        "top1_regret": {
            "mean": statistics.fmean(top1_regrets),
            "p90": percentile(top1_regrets, 0.90),
            "max": max(top1_regrets),
        },
        "shortlist_regret": {
            "mean": statistics.fmean(shortlist_regrets),
            "p90": percentile(shortlist_regrets, 0.90),
            "max": max(shortlist_regrets),
        },
    }


def cache_hierarchy_metric(row: dict) -> float:
    return sum(
        scalar_metric(row.get(field), name=field)
        for field in (
            "total_accesses",
            "l1_misses",
            "l2_misses",
            "l3_misses",
        )
    )


def build_multifidelity_certificate(
    protocol: dict,
    timing_rows: list[dict],
    cache_rows: list[dict],
) -> dict:
    graphs = list(protocol["graphs"])
    kernels = list(protocol["kernels"])
    specs = [
        canonical_spec(record["spec"])
        for record in protocol["arms"]
    ]
    sample_rates = [
        int(rate) for rate in protocol["sample_rates"]
    ]
    shortlist_size = int(protocol["shortlist_size"])
    cache_size = int(protocol["cache"]["cache_size_bytes"])

    timing = build_matrix(
        timing_rows,
        graphs=graphs,
        kernels=kernels,
        specs=set(specs),
    )
    cache: dict[tuple[str, str, str, str, int], dict] = {}
    for row in cache_rows:
        graph = str(row.get("graph", ""))
        kernel = str(row.get("benchmark", ""))
        spec = canonical_spec(row.get("algo_key"))
        if graph not in graphs or kernel not in kernels or spec not in specs:
            continue
        if int(row.get("cache_size_bytes", -1)) != cache_size:
            continue
        mode = str(row.get("cache_mode", ""))
        rate = int(row.get("cache_sample_rate", 1))
        if mode == protocol["cache"]["full_mode"]:
            rate = 1
        elif mode != protocol["cache"]["sampled_mode"]:
            continue
        key = (graph, kernel, spec, mode, rate)
        if key in cache:
            raise ValueError(f"Duplicate multi-fidelity cache row: {key}")
        cache[key] = row

    expected_full = {
        (
            graph,
            kernel,
            spec,
            protocol["cache"]["full_mode"],
            1,
        )
        for graph in graphs
        for kernel in kernels
        for spec in specs
    }
    missing_full = sorted(expected_full - set(cache))
    if missing_full:
        raise ValueError(
            f"Incomplete full-cache matrix; missing={missing_full[:1]}"
        )
    expected_sampled = {
        (
            graph,
            kernel,
            spec,
            protocol["cache"]["sampled_mode"],
            rate,
        )
        for graph in graphs
        for kernel in kernels
        for spec in specs
        for rate in sample_rates
    }
    missing_sampled = sorted(expected_sampled - set(cache))
    if missing_sampled:
        raise ValueError(
            "Incomplete sampled-cache matrix; "
            f"missing={missing_sampled[:1]}"
        )

    cache_rates = {}
    cache_gates = protocol["gates"]["sampled_cache"]
    for rate in sample_rates:
        context_records = []
        speedups = []
        full_seconds = 0.0
        sampled_seconds = 0.0
        for graph in graphs:
            for kernel in kernels:
                reference = {
                    spec: cache_hierarchy_metric(cache[(
                        graph,
                        kernel,
                        spec,
                        protocol["cache"]["full_mode"],
                        1,
                    )])
                    for spec in specs
                }
                candidate = {
                    spec: cache_hierarchy_metric(cache[(
                        graph,
                        kernel,
                        spec,
                        protocol["cache"]["sampled_mode"],
                        rate,
                    )])
                    for spec in specs
                }
                fidelity = ranking_fidelity(
                    reference,
                    candidate,
                    shortlist_size=shortlist_size,
                )
                fidelity.update({"graph": graph, "kernel": kernel})
                context_records.append(fidelity)
                for spec in specs:
                    full_time = scalar_metric(
                        cache[(
                            graph,
                            kernel,
                            spec,
                            protocol["cache"]["full_mode"],
                            1,
                        )].get("average_time"),
                        name="full cache average_time",
                    )
                    sampled_time = scalar_metric(
                        cache[(
                            graph,
                            kernel,
                            spec,
                            protocol["cache"]["sampled_mode"],
                            rate,
                        )].get("average_time"),
                        name="sampled cache average_time",
                    )
                    speedups.append(full_time / sampled_time)
                    full_seconds += full_time
                    sampled_seconds += sampled_time
        aggregate = aggregate_ranking_fidelity(context_records)
        runtime = {
            "full_over_sampled_runtime_gm": geometric_mean(speedups),
            "full_over_sampled_summed_runtime": (
                full_seconds / sampled_seconds
            ),
        }
        gate_results = {
            "runtime": (
                runtime["full_over_sampled_runtime_gm"]
                >= cache_gates["minimum_runtime_speedup"]
            ),
            "rank": (
                aggregate["kendall_tau_b_mean"]
                is not None
                and aggregate["kendall_tau_b_mean"]
                >= cache_gates["minimum_kendall_tau_b"]
            ),
            "winner_recall": (
                aggregate["reference_winner_in_shortlist_rate"]
                >= cache_gates[
                    "minimum_reference_winner_in_shortlist_rate"
                ]
            ),
            "regret": (
                aggregate["shortlist_regret"]["max"]
                <= cache_gates["maximum_shortlist_regret"]
            ),
        }
        cache_rates[str(rate)] = {
            "aggregate": aggregate,
            "runtime": runtime,
            "gate_results": gate_results,
            "passes_all_gates": all(gate_results.values()),
            "contexts_detail": context_records,
        }

    proxy_results = {}
    proxy_gates = protocol["gates"]["proxy_kernel"]
    for pair in protocol["proxy_pairs"]:
        proxy = pair["proxy"]
        target = pair["target"]
        records = []
        runtime_speedups = []
        for graph in graphs:
            reference = {
                spec: timing[(graph, target, spec)]
                for spec in specs
            }
            candidate = {
                spec: timing[(graph, proxy, spec)]
                for spec in specs
            }
            fidelity = ranking_fidelity(
                reference,
                candidate,
                shortlist_size=shortlist_size,
            )
            fidelity["graph"] = graph
            records.append(fidelity)
            runtime_speedups.extend(
                timing[(graph, target, spec)]
                / timing[(graph, proxy, spec)]
                for spec in specs
            )
        aggregate = aggregate_ranking_fidelity(records)
        gate_results = {
            "rank": (
                aggregate["kendall_tau_b_mean"]
                is not None
                and aggregate["kendall_tau_b_mean"]
                >= proxy_gates["minimum_kendall_tau_b"]
            ),
            "winner_recall": (
                aggregate["reference_winner_in_shortlist_rate"]
                >= proxy_gates[
                    "minimum_reference_winner_in_shortlist_rate"
                ]
            ),
            "regret": (
                aggregate["shortlist_regret"]["max"]
                <= proxy_gates["maximum_shortlist_regret"]
            ),
        }
        proxy_results[pair["name"]] = {
            "proxy": proxy,
            "target": target,
            "aggregate": aggregate,
            "target_over_proxy_runtime_gm": geometric_mean(
                runtime_speedups
            ),
            "gate_results": gate_results,
            "passes_all_gates": all(gate_results.values()),
            "graphs_detail": records,
        }

    combined_results = {}
    combined_gates = protocol["gates"]["combined_shortlist"]
    for pair in protocol["proxy_pairs"]:
        proxy = pair["proxy"]
        target = pair["target"]
        rate_results = {}
        for rate in sample_rates:
            records = []
            for graph in graphs:
                reference = {
                    spec: timing[(graph, target, spec)]
                    for spec in specs
                }
                proxy_values = [
                    timing[(graph, proxy, spec)]
                    for spec in specs
                ]
                cache_values = [
                    cache_hierarchy_metric(cache[(
                        graph,
                        proxy,
                        spec,
                        protocol["cache"]["sampled_mode"],
                        rate,
                    )])
                    for spec in specs
                ]
                proxy_ranks = average_ranks(proxy_values)
                cache_ranks = average_ranks(cache_values)
                candidate = {
                    spec: (
                        proxy_ranks[index] + cache_ranks[index]
                    ) / 2.0 + 1.0
                    for index, spec in enumerate(specs)
                }
                fidelity = ranking_fidelity(
                    reference,
                    candidate,
                    shortlist_size=shortlist_size,
                )
                fidelity["graph"] = graph
                records.append(fidelity)
            aggregate = aggregate_ranking_fidelity(records)
            gate_results = {
                "rank": (
                    aggregate["kendall_tau_b_mean"]
                    is not None
                    and aggregate["kendall_tau_b_mean"]
                    >= combined_gates["minimum_kendall_tau_b"]
                ),
                "winner_recall": (
                    aggregate["reference_winner_in_shortlist_rate"]
                    >= combined_gates[
                        "minimum_reference_winner_in_shortlist_rate"
                    ]
                ),
                "regret": (
                    aggregate["shortlist_regret"]["max"]
                    <= combined_gates["maximum_shortlist_regret"]
                ),
            }
            rate_results[str(rate)] = {
                "aggregate": aggregate,
                "gate_results": gate_results,
                "passes_all_gates": all(gate_results.values()),
                "graphs_detail": records,
            }
        combined_results[pair["name"]] = rate_results

    passing_cache_rates = [
        rate for rate in sample_rates
        if cache_rates[str(rate)]["passes_all_gates"]
    ]
    authorized_pairs = {
        pair["name"]: [
            rate for rate in passing_cache_rates
            if proxy_results[pair["name"]]["passes_all_gates"]
            and combined_results[pair["name"]][str(rate)][
                "passes_all_gates"
            ]
        ]
        for pair in protocol["proxy_pairs"]
    }
    return {
        "schema": "graphbrew-multifidelity-screen-analysis/v1",
        "scope": {
            "graphs": graphs,
            "kernels": kernels,
            "arms": specs,
            "sample_rates": sample_rates,
            "shortlist_size": shortlist_size,
        },
        "sampled_cache": cache_rates,
        "proxy_kernel": proxy_results,
        "combined_shortlist": combined_results,
        "decision": {
            "passing_cache_rates": passing_cache_rates,
            "authorized_proxy_rates": authorized_pairs,
            "graph_sampling_authorized": any(
                rates for rates in authorized_pairs.values()
            ),
            "policy": (
                "Proceed to topology-aware graph sampling only for proxy "
                "pairs and access-sampling rates that pass every frozen "
                "fidelity, regret, and runtime gate."
            ),
        },
    }


def summarize_ratio_records(
    records: list[tuple[str, str, float]],
    *,
    resamples: int,
) -> dict:
    by_graph: dict[str, list[float]] = {}
    by_kernel: dict[str, list[float]] = {}
    by_kernel_graph: dict[str, dict[str, list[float]]] = {}
    for graph, kernel, ratio in records:
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError(
                f"Invalid paired ratio for {graph}/{kernel}: {ratio}"
            )
        by_graph.setdefault(graph, []).append(ratio)
        by_kernel.setdefault(kernel, []).append(ratio)
        by_kernel_graph.setdefault(kernel, {}).setdefault(
            graph,
            [],
        ).append(ratio)
    graph_ratios = {
        graph: geometric_mean(values)
        for graph, values in sorted(by_graph.items())
    }
    ci_low, ci_high = bootstrap_graph_geomean(
        graph_ratios,
        resamples=resamples,
    )
    kernel_summaries = {}
    for kernel, graph_values in sorted(by_kernel_graph.items()):
        kernel_graph_ratios = {
            graph: geometric_mean(values)
            for graph, values in sorted(graph_values.items())
        }
        kernel_low, kernel_high = bootstrap_graph_geomean(
            kernel_graph_ratios,
            resamples=resamples,
        )
        kernel_summaries[kernel] = {
            "left_over_right_gm": geometric_mean(
                kernel_graph_ratios.values()
            ),
            "graph_block_95": [kernel_low, kernel_high],
            "graph_ratios": kernel_graph_ratios,
        }
    return {
        "left_over_right_gm": geometric_mean(graph_ratios.values()),
        "graph_block_95": [ci_low, ci_high],
        "right_wins_ties_losses_2pct": {
            "win": sum(value > 1.02 for value in records_value(records)),
            "tie": sum(
                0.98 <= value <= 1.02
                for value in records_value(records)
            ),
            "loss": sum(value < 0.98 for value in records_value(records)),
        },
        "by_kernel": {
            kernel: geometric_mean(values)
            for kernel, values in sorted(by_kernel.items())
        },
        "by_kernel_graph_block": kernel_summaries,
        "graph_ratios": graph_ratios,
    }


def records_value(
    records: list[tuple[str, str, float]],
) -> list[float]:
    return [ratio for _graph, _kernel, ratio in records]


def factor_summaries(
    *,
    graphs: list[str],
    kernels: list[str],
    blocks: list[str],
    intras: list[str],
    arm_by_axes: dict[tuple[str, str], str],
    value: Callable[[str, str, str], float],
    resamples: int,
) -> dict:
    block_records = []
    for graph in graphs:
        for kernel in kernels:
            for intra in intras:
                left = value(
                    graph,
                    kernel,
                    arm_by_axes[(blocks[0], intra)],
                )
                right = value(
                    graph,
                    kernel,
                    arm_by_axes[(blocks[1], intra)],
                )
                block_records.append((graph, kernel, left / right))
    block_by_intra = {}
    for intra in intras:
        records = []
        for graph in graphs:
            for kernel in kernels:
                left = value(
                    graph,
                    kernel,
                    arm_by_axes[(blocks[0], intra)],
                )
                right = value(
                    graph,
                    kernel,
                    arm_by_axes[(blocks[1], intra)],
                )
                records.append((graph, kernel, left / right))
        block_by_intra[intra] = summarize_ratio_records(
            records,
            resamples=resamples,
        )

    intra_pairs = {}
    for left_index, left_intra in enumerate(intras):
        for right_intra in intras[left_index + 1:]:
            records = []
            for graph in graphs:
                for kernel in kernels:
                    for block in blocks:
                        left = value(
                            graph,
                            kernel,
                            arm_by_axes[(block, left_intra)],
                        )
                        right = value(
                            graph,
                            kernel,
                            arm_by_axes[(block, right_intra)],
                        )
                        records.append((graph, kernel, left / right))
            intra_pairs[f"{left_intra}_over_{right_intra}"] = (
                summarize_ratio_records(records, resamples=resamples)
            )
    return {
        "ratio_semantics": (
            "left treatment divided by right treatment; values above one "
            "mean the right treatment is faster or performs less work"
        ),
        "block_order": {
            "left": blocks[0],
            "right": blocks[1],
            "marginal": summarize_ratio_records(
                block_records,
                resamples=resamples,
            ),
            "by_intra_order": block_by_intra,
        },
        "intra_order_pairs": intra_pairs,
    }


def contrast_alignment(
    timing: dict,
    locality: dict,
) -> dict:
    common_graphs = sorted(
        set(timing["graph_ratios"]) & set(locality["graph_ratios"])
    )
    timing_logs = [
        math.log(timing["graph_ratios"][graph])
        for graph in common_graphs
    ]
    locality_logs = [
        math.log(locality["graph_ratios"][graph])
        for graph in common_graphs
    ]
    same_direction = sum(
        timing_log * locality_log > 0
        for timing_log, locality_log in zip(
            timing_logs,
            locality_logs,
        )
    )
    return {
        "graphs": len(common_graphs),
        "same_direction_graphs": same_direction,
        "same_direction_fraction": (
            same_direction / len(common_graphs)
            if common_graphs else None
        ),
        "log_ratio_pearson": pearson(timing_logs, locality_logs),
        "rank_correlation": spearman(timing_logs, locality_logs),
    }


def build_mechanism_certificate(
    protocol: dict,
    timing_rows: list[dict],
    verification_rows: list[dict],
    cache_rows: list[dict],
) -> dict:
    graphs = list(protocol["graphs"])
    kernels = list(protocol["kernels"])
    arms = list(protocol["arms"])
    specs = [canonical_spec(record["spec"]) for record in arms]
    arm_by_axes = {
        (record["block_order"], record["intra_order"]):
            canonical_spec(record["spec"])
        for record in arms
    }
    blocks = list(protocol["factors"]["block_order"])
    intras = list(protocol["factors"]["intra_order"])
    expected_axes = {
        (block, intra)
        for block in blocks
        for intra in intras
    }
    if set(arm_by_axes) != expected_axes:
        raise ValueError("Factorial arms do not cover the declared 2x3 grid")
    resamples = int(protocol["analysis"]["bootstrap_resamples"])

    timing = build_matrix(
        timing_rows,
        graphs=graphs,
        kernels=kernels,
        specs=set(specs),
    )
    timing_factors = factor_summaries(
        graphs=graphs,
        kernels=kernels,
        blocks=blocks,
        intras=intras,
        arm_by_axes=arm_by_axes,
        value=lambda graph, kernel, spec: timing[(graph, kernel, spec)],
        resamples=resamples,
    )

    verification: dict[tuple[str, str, str], dict] = {}
    for row in verification_rows:
        graph = str(row.get("graph", ""))
        kernel = str(row.get("benchmark", ""))
        spec = canonical_spec(row.get("algo_key"))
        if graph not in graphs or kernel not in kernels or spec not in specs:
            continue
        key = (graph, kernel, spec)
        if key in verification:
            raise ValueError(f"Duplicate verification row: {key}")
        if row.get("verification_state") != "pass":
            raise ValueError(f"Verification did not pass: {key}")
        verification[key] = row
    expected_verification = {
        (graph, kernel, spec)
        for graph in graphs
        for kernel in kernels
        for spec in specs
    }
    if set(verification) != expected_verification:
        missing = sorted(expected_verification - set(verification))
        raise ValueError(
            "Incomplete mechanism verification matrix; "
            f"missing={missing[:1]}"
        )

    cache_config = protocol["analysis"]["cache"]
    cache_benchmark = cache_config["benchmark"]
    cache_size = int(cache_config["cache_size_bytes"])
    cache: dict[tuple[str, str], float] = {}
    cache_components: dict[tuple[str, str], dict[str, float]] = {}
    for row in cache_rows:
        graph = str(row.get("graph", ""))
        spec = canonical_spec(row.get("algo_key"))
        if graph not in graphs or spec not in specs:
            continue
        if (
            row.get("benchmark") != cache_benchmark
            or int(row.get("cache_size_bytes", -1)) != cache_size
            or row.get("cache_mode") != cache_config["mode"]
        ):
            continue
        key = (graph, spec)
        if key in cache:
            raise ValueError(f"Duplicate cache row: {key}")
        components = {
            field: scalar_metric(row.get(field), name=field)
            for field in (
                "total_accesses",
                "l1_misses",
                "l2_misses",
                "l3_misses",
                "memory_accesses",
            )
        }
        cache_components[key] = components
        cache[key] = sum(
            components[field]
            for field in (
                "total_accesses",
                "l1_misses",
                "l2_misses",
                "l3_misses",
            )
        )
    expected_cache = {
        (graph, spec) for graph in graphs for spec in specs
    }
    if set(cache) != expected_cache:
        missing = sorted(expected_cache - set(cache))
        raise ValueError(
            f"Incomplete cache matrix; missing={missing[:1]}"
        )
    cache_factors = factor_summaries(
        graphs=graphs,
        kernels=[cache_benchmark],
        blocks=blocks,
        intras=intras,
        arm_by_axes=arm_by_axes,
        value=lambda graph, _kernel, spec: cache[(graph, spec)],
        resamples=resamples,
    )
    cache_timing_factors = factor_summaries(
        graphs=graphs,
        kernels=[cache_benchmark],
        blocks=blocks,
        intras=intras,
        arm_by_axes=arm_by_axes,
        value=lambda graph, kernel, spec: timing[(graph, kernel, spec)],
        resamples=resamples,
    )
    cache_component_factorials = {
        field: factor_summaries(
            graphs=graphs,
            kernels=[cache_benchmark],
            blocks=blocks,
            intras=intras,
            arm_by_axes=arm_by_axes,
            value=lambda graph, _kernel, spec, metric=field: (
                cache_components[(graph, spec)][metric]
            ),
            resamples=resamples,
        )
        for field in (
            "l1_misses",
            "l2_misses",
            "l3_misses",
            "memory_accesses",
        )
    }

    artifact_root = Path(protocol["execution"]["artifact_root"])
    memberships = {}
    edge_spans: dict[tuple[str, str], float] = {}
    allowed_fields = set(
        protocol["analysis"]["allowed_effective_config_differences"]
    )
    for graph in graphs:
        fingerprints = {}
        effective_configs = {}
        realized_configs = {}
        for spec in specs:
            meta_path = (
                artifact_root
                / "vldb_mappings"
                / graph
                / f"{spec.replace(':', '_')}.json"
            )
            meta = json.loads(meta_path.read_text())
            effective = meta.get("graphbrew_effective_configs", [])
            realized = meta.get("graphbrew_realized_configs", [])
            if len(effective) != 1 or len(realized) != 1:
                raise ValueError(
                    f"Missing GraphBrew configuration metadata: {graph}/{spec}"
                )
            effective_configs[spec] = effective[0]
            realized_configs[spec] = realized[0]
            fingerprints[spec] = realized[0]["membership_fingerprint"]
            edge_spans[(graph, spec)] = scalar_metric(
                meta["mapping_draws"][0]["mapping_sampled_edge_span"],
                name="mapping_sampled_edge_span",
            )
        if len(set(fingerprints.values())) != 1:
            raise ValueError(f"Membership drift on {graph}")
        reference_spec = specs[0]
        reference = effective_configs[reference_spec]
        for spec, effective in effective_configs.items():
            unexpected = {
                field
                for field in set(reference) | set(effective)
                if reference.get(field) != effective.get(field)
                and field not in allowed_fields
            }
            if unexpected:
                raise ValueError(
                    f"Unexpected factorial config drift on {graph}/{spec}: "
                    f"{sorted(unexpected)}"
                )
        memberships[graph] = {
            "membership_fingerprint": next(iter(fingerprints.values())),
            "num_communities": sorted({
                config["num_communities"]
                for config in realized_configs.values()
            }),
            "num_passes": sorted({
                config["num_passes"]
                for config in realized_configs.values()
            }),
        }

    cache_rank_correlations = {}
    span_rank_correlations: dict[str, dict[str, float | None]] = {}
    for graph in graphs:
        cache_rank_correlations[graph] = spearman(
            [cache[(graph, spec)] for spec in specs],
            [timing[(graph, cache_benchmark, spec)] for spec in specs],
        )
    for kernel in kernels:
        span_rank_correlations[kernel] = {
            graph: spearman(
                [edge_spans[(graph, spec)] for spec in specs],
                [timing[(graph, kernel, spec)] for spec in specs],
            )
            for graph in graphs
        }

    cache_alignment = {
        "block_order": {
            intra: contrast_alignment(
                cache_timing_factors["block_order"]["by_intra_order"][
                    intra
                ],
                cache_factors["block_order"]["by_intra_order"][intra],
            )
            for intra in intras
        },
        "intra_order_pairs": {
            pair: contrast_alignment(
                cache_timing_factors["intra_order_pairs"][pair],
                cache_factors["intra_order_pairs"][pair],
            )
            for pair in timing_factors["intra_order_pairs"]
        },
    }

    work_explanation = {}
    for kernel, work_config in protocol["analysis"]["dynamic_work"].items():
        primary_metric = work_config["primary_metric"]

        def work_value(graph: str, target_kernel: str, spec: str) -> float:
            metrics = verification[(graph, target_kernel, spec)][
                "work_metrics"
            ]
            value = scalar_metric(
                metrics.get(primary_metric),
                name=primary_metric,
            )
            if value <= 0:
                raise ValueError(
                    f"{primary_metric} must be positive for {graph}/{spec}"
                )
            return value

        work_factors = factor_summaries(
            graphs=graphs,
            kernels=[kernel],
            blocks=blocks,
            intras=intras,
            arm_by_axes=arm_by_axes,
            value=work_value,
            resamples=resamples,
        )
        normalized_factors = factor_summaries(
            graphs=graphs,
            kernels=[kernel],
            blocks=blocks,
            intras=intras,
            arm_by_axes=arm_by_axes,
            value=lambda graph, target_kernel, spec: (
                timing[(graph, target_kernel, spec)]
                / work_value(graph, target_kernel, spec)
            ),
            resamples=resamples,
        )
        graph_correlations = {
            graph: spearman(
                [
                    work_value(graph, kernel, spec)
                    for spec in specs
                ],
                [
                    timing[(graph, kernel, spec)]
                    for spec in specs
                ],
            )
            for graph in graphs
        }
        fastest_is_least_work = sum(
            min(
                specs,
                key=lambda spec: timing[(graph, kernel, spec)],
            )
            == min(
                specs,
                key=lambda spec: work_value(graph, kernel, spec),
            )
            for graph in graphs
        )
        work_explanation[kernel] = {
            "primary_metric": primary_metric,
            "work_factorial": work_factors,
            "seconds_per_work_unit_factorial": normalized_factors,
            "work_time_rank_correlation": {
                "by_graph": graph_correlations,
                "mean": statistics.fmean(
                    value for value in graph_correlations.values()
                    if value is not None
                ),
                "median": statistics.median(
                    value for value in graph_correlations.values()
                    if value is not None
                ),
            },
            "fastest_arm_is_least_work": fastest_is_least_work,
            "graphs": len(graphs),
        }

    dynamic_cache_explanation = {}
    dynamic_cache_config = protocol["analysis"].get("dynamic_cache")
    if dynamic_cache_config is not None:
        for kernel in dynamic_cache_config["benchmarks"]:
            primary_metric = protocol["analysis"]["dynamic_work"][kernel][
                "primary_metric"
            ]
            dynamic_cache: dict[tuple[str, str], float] = {}
            for row in cache_rows:
                graph = str(row.get("graph", ""))
                spec = canonical_spec(row.get("algo_key"))
                if graph not in graphs or spec not in specs:
                    continue
                if (
                    row.get("benchmark") != kernel
                    or int(row.get("cache_size_bytes", -1)) != cache_size
                    or row.get("cache_mode") != cache_config["mode"]
                ):
                    continue
                key = (graph, spec)
                if key in dynamic_cache:
                    raise ValueError(f"Duplicate dynamic cache row: {key}")
                dynamic_cache[key] = sum(
                    scalar_metric(row.get(field), name=field)
                    for field in (
                        "total_accesses",
                        "l1_misses",
                        "l2_misses",
                        "l3_misses",
                    )
                )
            if set(dynamic_cache) != expected_cache:
                missing = sorted(expected_cache - set(dynamic_cache))
                raise ValueError(
                    f"Incomplete {kernel} cache matrix; missing={missing[:1]}"
                )

            def primary_work(graph: str, spec: str) -> float:
                metric = verification[(graph, kernel, spec)][
                    "work_metrics"
                ].get(primary_metric)
                value = scalar_metric(metric, name=primary_metric)
                if value <= 0:
                    raise ValueError(
                        f"{primary_metric} must be positive for {graph}/{spec}"
                    )
                return value

            dynamic_cache_factors = factor_summaries(
                graphs=graphs,
                kernels=[kernel],
                blocks=blocks,
                intras=intras,
                arm_by_axes=arm_by_axes,
                value=lambda graph, _kernel, spec: dynamic_cache[
                    (graph, spec)
                ],
                resamples=resamples,
            )
            normalized_timing_factors = work_explanation[kernel][
                "seconds_per_work_unit_factorial"
            ]
            normalized_alignment = {
                "block_order": {
                    intra: contrast_alignment(
                        normalized_timing_factors["block_order"][
                            "by_intra_order"
                        ][intra],
                        dynamic_cache_factors["block_order"][
                            "by_intra_order"
                        ][intra],
                    )
                    for intra in intras
                },
                "intra_order_pairs": {
                    pair: contrast_alignment(
                        normalized_timing_factors[
                            "intra_order_pairs"
                        ][pair],
                        dynamic_cache_factors["intra_order_pairs"][pair],
                    )
                    for pair in normalized_timing_factors[
                        "intra_order_pairs"
                    ]
                },
            }
            graph_correlations = {
                graph: spearman(
                    [
                        dynamic_cache[(graph, spec)]
                        for spec in specs
                    ],
                    [
                        timing[(graph, kernel, spec)]
                        / primary_work(graph, spec)
                        for spec in specs
                    ],
                )
                for graph in graphs
            }
            finite_correlations = [
                value for value in graph_correlations.values()
                if value is not None
            ]
            dynamic_cache_explanation[kernel] = {
                "primary_work_metric": primary_metric,
                "cache_factorial": dynamic_cache_factors,
                "seconds_per_work_unit_alignment": normalized_alignment,
                "cache_vs_seconds_per_work_rank_correlation": {
                    "by_graph": graph_correlations,
                    "mean": statistics.fmean(finite_correlations),
                    "median": statistics.median(finite_correlations),
                },
            }

    winner_counts = Counter()
    for graph in graphs:
        for kernel in kernels:
            winner_counts[min(
                specs,
                key=lambda spec: timing[(graph, kernel, spec)],
            )] += 1

    cache_correlations = [
        value for value in cache_rank_correlations.values()
        if value is not None
    ]
    span_correlations = [
        value
        for kernel_values in span_rank_correlations.values()
        for value in kernel_values.values()
        if value is not None
    ]
    return {
        "schema": "graphbrew-mechanism-factorial-analysis/v1",
        "scope": {
            "graphs": graphs,
            "kernels": kernels,
            "arms": len(specs),
            "timing_cells": len(graphs) * len(kernels) * len(specs),
            "cache_cells": len(graphs) * len(specs),
            "hardware_counters": protocol["analysis"][
                "hardware_counter_status"
            ],
        },
        "factorial_contract": {
            "block_orders": blocks,
            "intra_orders": intras,
            "allowed_effective_config_differences": sorted(
                allowed_fields
            ),
            "memberships": memberships,
            "membership_equivalent": True,
        },
        "winner_counts": {
            next(
                record["name"] for record in arms
                if canonical_spec(record["spec"]) == spec
            ): count
            for spec, count in sorted(winner_counts.items())
        },
        "timing_factorial": timing_factors,
        "cache_hierarchy": {
            "metric": (
                "total_accesses + l1_misses + l2_misses + l3_misses"
            ),
            "factorial": cache_factors,
            "component_factorials": cache_component_factorials,
            "matching_pr_timing_factorial": cache_timing_factors,
            "pr_rank_correlation": {
                "by_graph": cache_rank_correlations,
                "mean": statistics.fmean(cache_correlations),
                "median": statistics.median(cache_correlations),
            },
            "timing_alignment": cache_alignment,
            **(
                {
                    "dynamic_work_normalized":
                        dynamic_cache_explanation
                }
                if dynamic_cache_explanation else {}
            ),
        },
        "dynamic_work": work_explanation,
        "negative_control": {
            "sampled_edge_span_time_rank_correlation": {
                "by_kernel_graph": span_rank_correlations,
                "mean": statistics.fmean(span_correlations),
                "median": statistics.median(span_correlations),
            },
        },
    }


def build_certificate(protocol: dict, rows: list[dict]) -> dict:
    graphs = list(protocol["graphs"])
    kernels = list(protocol["kernels"])
    candidates = [
        canonical_spec(record["spec"])
        for record in protocol["admissible_complete_rabbit_free_candidates"]
    ]
    candidate_names = {
        canonical_spec(record["spec"]): record["name"]
        for record in protocol["admissible_complete_rabbit_free_candidates"]
    }
    comparator_specs = [
        canonical_spec(record["spec"])
        for record in protocol["competitors"]
    ]
    all_specs = set(candidates + comparator_specs + [ORIGINAL_SPEC])
    matrix = build_matrix(
        rows,
        graphs=graphs,
        kernels=kernels,
        specs=all_specs,
    )
    cells = [
        (graph, kernel)
        for graph in graphs
        for kernel in kernels
    ]
    graph_type = protocol.get("graph_types") or {
        graph: family
        for family, family_graphs in GRAPH_TYPE_GROUPS.items()
        for graph in family_graphs
    }
    missing_types = sorted(set(graphs) - set(graph_type))
    if missing_types:
        raise ValueError(
            "Missing graph-type assignments: " + ", ".join(missing_types)
        )

    best_fixed = choose_arm(cells, candidates, matrix)
    cell_oracle = {
        cell: min(
            candidates,
            key=lambda spec: (matrix[(*cell, spec)], spec),
        )
        for cell in cells
    }
    by_kernel = {
        kernel: choose_arm(
            [(graph, kernel) for graph in graphs],
            candidates,
            matrix,
        )
        for kernel in kernels
    }
    by_graph = {
        graph: choose_arm(
            [(graph, kernel) for kernel in kernels],
            candidates,
            matrix,
        )
        for graph in graphs
    }
    by_type = {
        family: choose_arm(
            [
                (graph, kernel)
                for graph, kernel in cells
                if graph_type[graph] == family
            ],
            candidates,
            matrix,
        )
        for family in sorted(set(graph_type.values()))
    }
    by_type_kernel = {
        (family, kernel): choose_arm(
            [
                (graph, candidate_kernel)
                for graph, candidate_kernel in cells
                if graph_type[graph] == family
                and candidate_kernel == kernel
            ],
            candidates,
            matrix,
        )
        for family in sorted(set(graph_type.values()))
        for kernel in kernels
    }

    def held_out_kernel(graph: str, kernel: str) -> str:
        return choose_arm(
            [
                (other_graph, kernel)
                for other_graph in graphs
                if other_graph != graph
            ],
            candidates,
            matrix,
        )

    def held_out_family_kernel(graph: str, kernel: str) -> str:
        peers = [
            (other_graph, kernel)
            for other_graph in graphs
            if other_graph != graph
            and graph_type[other_graph] == graph_type[graph]
        ]
        if not peers:
            peers = [
                (other_graph, kernel)
                for other_graph in graphs
                if other_graph != graph
            ]
        return choose_arm(peers, candidates, matrix)

    assignments: dict[str, Assignment] = {
        "best_fixed": lambda _graph, _kernel: best_fixed,
        "kernel_conditioned": lambda _graph, kernel: by_kernel[kernel],
        "graph_conditioned": lambda graph, _kernel: by_graph[graph],
        "graph_type_conditioned": (
            lambda graph, _kernel: by_type[graph_type[graph]]
        ),
        "graph_type_kernel_conditioned": (
            lambda graph, kernel: by_type_kernel[(graph_type[graph], kernel)]
        ),
        "cell_oracle": lambda graph, kernel: cell_oracle[(graph, kernel)],
        "leave_one_graph_out_kernel": held_out_kernel,
        "leave_one_graph_out_family_kernel": held_out_family_kernel,
    }
    frozen_policy = protocol.get("frozen_family_kernel_policy")
    frozen_family_kernel_assignment: Assignment | None = None
    if frozen_policy is not None:
        by_family_kernel = frozen_policy["by_family_kernel"]
        fallback_by_kernel = frozen_policy["fallback_by_kernel"]

        def frozen_family_kernel(graph: str, kernel: str) -> str:
            spec = by_family_kernel.get(
                f"{graph_type[graph]}/{kernel}",
                fallback_by_kernel[kernel],
            )
            if spec not in candidates:
                raise ValueError(
                    f"Frozen policy selected an inadmissible arm: {spec}"
                )
            return spec

        frozen_family_kernel_assignment = frozen_family_kernel
        assignments["frozen_family_kernel"] = frozen_family_kernel
    policy_results = {
        name: evaluate_assignment(
            cells=cells,
            assignment=assignment,
            matrix=matrix,
            comparator_specs=comparator_specs,
            best_fixed_spec=best_fixed,
        )
        for name, assignment in assignments.items()
    }

    winner_counts = Counter(cell_oracle.values())
    per_graph_distinct = {
        graph: len({cell_oracle[(graph, kernel)] for kernel in kernels})
        for graph in graphs
    }
    per_kernel_distinct = {
        kernel: len({cell_oracle[(graph, kernel)] for graph in graphs})
        for kernel in kernels
    }
    family_support = {
        family: sum(graph_type[graph] == family for graph in graphs)
        for family in sorted(set(graph_type.values()))
    }
    confirmation = None
    confirmation_gates = protocol.get("confirmation_gates")
    if confirmation_gates is not None:
        if frozen_family_kernel_assignment is None:
            raise ValueError(
                "Confirmation protocol requires a frozen family/kernel policy"
            )
        controls = protocol["predeclared_uniform_controls"]
        for spec in controls.values():
            if spec not in candidates:
                raise ValueError(
                    f"Confirmation control is not a candidate arm: {spec}"
                )
        resamples = int(confirmation_gates["bootstrap_resamples"])
        mapping_cache: dict[tuple[str, str], float] = {}

        def complete_mapping_seconds(graph: str, spec: str) -> float:
            key = (graph, spec)
            if key not in mapping_cache:
                mapping_cache[key] = mapping_seconds(
                    protocol,
                    graph,
                    spec,
                )
            return mapping_cache[key]

        def graph_kernel_ratios(spec: str) -> dict[str, float]:
            return {
                graph: geometric_mean(
                    matrix[(graph, kernel, spec)]
                    / matrix[(
                        graph,
                        kernel,
                        frozen_family_kernel_assignment(graph, kernel),
                    )]
                    for kernel in kernels
                )
                for graph in graphs
            }

        def graph_end_to_end_ratios(
            spec: str,
            reuse: int,
        ) -> dict[str, float]:
            return {
                graph: geometric_mean(
                    (
                        complete_mapping_seconds(graph, spec)
                        + reuse * matrix[(graph, kernel, spec)]
                    )
                    / (
                        complete_mapping_seconds(
                            graph,
                            frozen_family_kernel_assignment(graph, kernel),
                        )
                        + reuse
                        * matrix[(
                            graph,
                            kernel,
                            frozen_family_kernel_assignment(graph, kernel),
                        )]
                    )
                    for kernel in kernels
                )
                for graph in graphs
            }

        control_results = {}
        for name, spec in controls.items():
            graph_ratios = graph_kernel_ratios(spec)
            ci_low, ci_high = bootstrap_graph_geomean(
                graph_ratios,
                resamples=resamples,
            )
            control_results[name] = {
                "spec": spec,
                "control_over_frozen_policy_kernel_gm": geometric_mean(
                    graph_ratios.values()
                ),
                "graph_block_95": [ci_low, ci_high],
                "worst_graph_ratio": min(graph_ratios.values()),
                "winning_graphs": sum(
                    value > 1.0 for value in graph_ratios.values()
                ),
                "graph_ratios": graph_ratios,
            }

        gorder_spec = next(
            spec
            for spec in comparator_specs
            if spec.startswith("9:")
        )
        gorder_graph_ratios = graph_kernel_ratios(gorder_spec)
        gorder_ci_low, gorder_ci_high = bootstrap_graph_geomean(
            gorder_graph_ratios,
            resamples=resamples,
        )
        gorder_reuse = {}
        for reuse in (1, 4, 16, 64, 256, 1000):
            graph_ratios = graph_end_to_end_ratios(gorder_spec, reuse)
            ci_low, ci_high = bootstrap_graph_geomean(
                graph_ratios,
                resamples=resamples,
            )
            gorder_reuse[str(reuse)] = {
                "gorder_over_frozen_policy_gm": geometric_mean(
                    graph_ratios.values()
                ),
                "graph_block_95": [ci_low, ci_high],
                "worst_graph_ratio": min(graph_ratios.values()),
            }
        crossover = None
        for reuse in range(1, 1_000_001):
            if geometric_mean(
                graph_end_to_end_ratios(
                    gorder_spec,
                    reuse,
                ).values()
            ) >= 1.0:
                crossover = reuse
                break

        fixed_result = control_results["rapid_best_fixed"]
        gorder_kernel_gm = geometric_mean(gorder_graph_ratios.values())
        gate_results = {
            "rapid_best_fixed_point": (
                fixed_result["control_over_frozen_policy_kernel_gm"]
                >= confirmation_gates[
                    "minimum_rapid_best_fixed_over_policy_gm"
                ]
            ),
            "rapid_best_fixed_ci": (
                fixed_result["graph_block_95"][0]
                >= confirmation_gates[
                    "minimum_rapid_best_fixed_over_policy_graph_block_95_low"
                ]
            ),
            "gorder_kernel_point": (
                gorder_kernel_gm
                >= confirmation_gates[
                    "minimum_gorder_csr_over_policy_kernel_gm"
                ]
            ),
            "gorder_kernel_ci": (
                gorder_ci_low
                >= confirmation_gates[
                    "minimum_gorder_csr_over_policy_graph_block_95_low"
                ]
            ),
            "gorder_reuse1_end_to_end": (
                gorder_reuse["1"][
                    "gorder_over_frozen_policy_gm"
                ]
                >= confirmation_gates[
                    "minimum_gorder_csr_over_policy_reuse1_end_to_end_gm"
                ]
            ),
            "worst_graph_floor": (
                fixed_result["worst_graph_ratio"]
                >= confirmation_gates[
                    "minimum_worst_graph_rapid_best_fixed_over_policy"
                ]
            ),
        }
        confirmation = {
            "controls": control_results,
            "gorder_csr": {
                "kernel_gm": gorder_kernel_gm,
                "graph_block_95": [gorder_ci_low, gorder_ci_high],
                "end_to_end": gorder_reuse,
                "cell_gm_crossover_reuse": crossover,
            },
            "gate_results": gate_results,
            "passes_all_gates": all(gate_results.values()),
        }

    singleton_types = sum(count == 1 for count in family_support.values())
    classification = {
        "composition_expressiveness": (
            "supported as post-selected design-space evidence"
        ),
        "uniform_graphbrew_policy": (
            "rejected: the cell oracle is materially faster than the "
            "best fixed GraphBrew arm"
        ),
        "graph_type_kernel_generalization": (
            "not established: "
            f"{singleton_types} of {len(family_support)} topology groups are "
            "singletons and the held-out family/kernel policy loses"
        ),
        "automated_selector": (
            "rejected for the current library by graph-held-out evidence"
        ),
    }
    if frozen_policy is not None:
        frozen_result = policy_results["frozen_family_kernel"]
        gates = protocol.get("rapid_gates", {})
        classification["frozen_family_kernel_policy"] = {
            "fastest_comparator_over_policy_gm": frozen_result[
                "fastest_comparator_over_policy_gm"
            ],
            "best_fixed_graphbrew_over_policy_gm": frozen_result[
                "best_fixed_graphbrew_over_policy_gm"
            ],
            "passes_rapid_gates": (
                frozen_result["fastest_comparator_over_policy_gm"]
                >= gates.get(
                    "minimum_fastest_comparator_over_policy_gm",
                    math.inf,
                )
                and frozen_result["best_fixed_graphbrew_over_policy_gm"]
                >= gates.get(
                    "minimum_best_fixed_graphbrew_over_policy_gm",
                    math.inf,
                )
            ),
        }

    return {
        "schema": "graphbrew-composability-certificate/v1",
        "scope": {
            "provenance_class": (
                protocol.get("claim_scope")
                or protocol.get("hypothesis")
                or (
                    "historical design-space evidence; not claim-eligible "
                    "timing or a deployable selector"
                )
            ),
            "graphs": graphs,
            "kernels": kernels,
            "cells": len(cells),
            "candidate_compositions": len(candidates),
            "comparators": comparator_specs,
            "mandatory_no_reorder_control": ORIGINAL_SPEC,
        },
        "winner_diversity": {
            "distinct_winning_compositions": len(winner_counts),
            "winner_counts": {
                candidate_names[spec]: count
                for spec, count in sorted(winner_counts.items())
            },
            "distinct_winners_per_graph": per_graph_distinct,
            "minimum_distinct_winners_per_graph": min(
                per_graph_distinct.values()
            ),
            "maximum_distinct_winners_per_graph": max(
                per_graph_distinct.values()
            ),
            "distinct_winners_per_kernel": per_kernel_distinct,
            "minimum_distinct_winners_per_kernel": min(
                per_kernel_distinct.values()
            ),
            "maximum_distinct_winners_per_kernel": max(
                per_kernel_distinct.values()
            ),
        },
        "graph_type_support": family_support,
        "selected_arms": {
            "best_fixed": candidate_names[best_fixed],
            "kernel_conditioned": {
                kernel: candidate_names[spec]
                for kernel, spec in by_kernel.items()
            },
            "graph_conditioned": {
                graph: candidate_names[spec]
                for graph, spec in by_graph.items()
            },
            "graph_type_conditioned": {
                family: candidate_names[spec]
                for family, spec in by_type.items()
            },
            "graph_type_kernel_conditioned": {
                f"{family}/{kernel}": candidate_names[spec]
                for (family, kernel), spec in by_type_kernel.items()
            },
        },
        "policy_results": policy_results,
        **({"confirmation": confirmation} if confirmation else {}),
        "classification": classification,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_ATLAS_ROOT / "protocol_v2.json",
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Override the timing source recorded by the protocol.",
    )
    parser.add_argument(
        "--verification-source",
        type=Path,
        help="Override the verification/work source for a mechanism factorial.",
    )
    parser.add_argument(
        "--cache-source",
        type=Path,
        help="Override the cache source for a mechanism factorial.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ATLAS_ROOT / "composability_certificate_v1.json",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=DEFAULT_ATLAS_ROOT / "composability_receipt_v1.json",
    )
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    source_record = protocol["source"]
    source = args.source or Path(source_record["path"])
    if (
        source_record.get("sha256")
        and sha256(source) != source_record["sha256"]
    ):
        raise ValueError("Timing source hash does not match the atlas protocol")
    rows = json.loads(source.read_text())
    if not isinstance(rows, list):
        raise ValueError("Timing source must contain a JSON row list")

    source_bindings = {
        "protocol": {
            "path": str(args.protocol),
            "sha256": sha256(args.protocol),
        },
        "timing": {
            "path": str(source),
            "sha256": sha256(source),
        },
        "generator": {
            "path": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "sha256": sha256(Path(__file__)),
        },
    }
    if protocol["schema"] == "graphbrew-mechanism-factorial-protocol/v1":
        verification_record = protocol["verification_source"]
        verification_source = (
            args.verification_source
            or Path(verification_record["path"])
        )
        if (
            verification_record.get("sha256")
            and sha256(verification_source)
            != verification_record["sha256"]
        ):
            raise ValueError(
                "Verification source hash does not match the protocol"
            )
        verification_rows = json.loads(verification_source.read_text())
        if not isinstance(verification_rows, list):
            raise ValueError("Verification source must contain a row list")

        cache_record = protocol["cache_source"]
        cache_source = args.cache_source or Path(cache_record["path"])
        if (
            cache_record.get("sha256")
            and sha256(cache_source) != cache_record["sha256"]
        ):
            raise ValueError("Cache source hash does not match the protocol")
        cache_rows = json.loads(cache_source.read_text())
        if not isinstance(cache_rows, list):
            raise ValueError("Cache source must contain a row list")
        certificate = build_mechanism_certificate(
            protocol,
            rows,
            verification_rows,
            cache_rows,
        )
        source_bindings.update({
            "verification": {
                "path": str(verification_source),
                "sha256": sha256(verification_source),
            },
            "cache": {
                "path": str(cache_source),
                "sha256": sha256(cache_source),
            },
        })
    elif protocol["schema"] == "graphbrew-multifidelity-screen-protocol/v1":
        cache_record = protocol["cache_source"]
        cache_source = args.cache_source or Path(cache_record["path"])
        if (
            cache_record.get("sha256")
            and sha256(cache_source) != cache_record["sha256"]
        ):
            raise ValueError("Cache source hash does not match the protocol")
        cache_rows = json.loads(cache_source.read_text())
        if not isinstance(cache_rows, list):
            raise ValueError("Cache source must contain a row list")
        certificate = build_multifidelity_certificate(
            protocol,
            rows,
            cache_rows,
        )
        source_bindings["cache"] = {
            "path": str(cache_source),
            "sha256": sha256(cache_source),
        }
    else:
        certificate = build_certificate(protocol, rows)
    certificate["sources"] = source_bindings
    write_json(args.output, certificate)
    conclusion = certificate.get("classification")
    if conclusion is None:
        if "factorial_contract" in certificate:
            conclusion = {
                "membership_equivalent":
                    certificate["factorial_contract"][
                        "membership_equivalent"
                    ],
                "hardware_counters":
                    certificate["scope"]["hardware_counters"],
            }
        else:
            conclusion = certificate["decision"]
    receipt = {
        "schema": "graphbrew-composability-receipt/v1",
        "certificate": {
            "path": str(args.output),
            "sha256": sha256(args.output),
        },
        "conclusion": conclusion,
    }
    write_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
