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
                "historical design-space evidence; not claim-eligible "
                "timing or a deployable selector"
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

    certificate = build_certificate(protocol, rows)
    certificate["sources"] = {
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
    write_json(args.output, certificate)
    receipt = {
        "schema": "graphbrew-composability-receipt/v1",
        "certificate": {
            "path": str(args.output),
            "sha256": sha256(args.output),
        },
        "conclusion": certificate["classification"],
    }
    write_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
