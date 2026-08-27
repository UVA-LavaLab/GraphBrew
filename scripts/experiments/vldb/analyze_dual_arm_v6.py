#!/usr/bin/env python3
"""Analyze the one-pass native intra-order quality-potential screen."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import statistics
import struct
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.vldb.analyze_dual_arm_s0 import (  # noqa: E402
    geometric_mean,
    positive_finite,
    sha256,
    short_id,
)
from scripts.experiments.vldb.analyze_dual_arm_v3 import crc32  # noqa: E402
from scripts.experiments.vldb.analyze_dual_arm_v5 import (  # noqa: E402
    ORIGINAL,
    RABBIT_BOOST,
    RABBIT_CSR,
    _mapping_sidecars,
    _validate_mapping,
)


EXPECTED_PROTOCOL_SHA256 = (
    "ef9df89a7796a524cf91f1eff3bbefc1"
    "6ea6b25e3c8cd55d04ba29284dfac761"
)


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def analyze(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = root / "protocol.json"
    if sha256(protocol_path) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("V6 protocol hash changed")
    protocol = json.loads(protocol_path.read_text())
    graphs = list(protocol["graphs"])
    kernels = list(protocol["kernels"])
    primary_kernels = list(protocol["primary_kernels"])
    candidates = list(protocol["candidates"])
    reordered_arms = [*candidates, RABBIT_CSR, RABBIT_BOOST]
    timing_arms = [ORIGINAL, *reordered_arms]
    if (
        protocol["schema"]
        != "graphbrew-dual-arm-v6-fast-intra-quality-screen/v1"
        or protocol["repository_clean"] is not True
        or len(graphs) != 5
        or len(kernels) != 7
        or len(primary_kernels) != 5
        or len(candidates) != 4
        or protocol["anchors"]
        != [ORIGINAL, RABBIT_CSR, RABBIT_BOOST]
        or protocol["mapping_draws_per_reordered_arm"] != 3
        or protocol["timing_processes"] != 1
        or protocol["trials_per_process"] != 5
    ):
        raise ValueError("Invalid V6 frozen contract")

    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    if dirty:
        raise ValueError("Repository is dirty during V6 analysis")
    subprocess.run(
        [
            "git", "cat-file", "-e",
            f"{protocol['repository_commit']}^{{commit}}",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    graph_records = {
        str(record["graph"]): record
        for record in protocol["graph_records"]
    }
    if set(graph_records) != set(graphs):
        raise ValueError("V6 graph records changed")
    for graph, record in graph_records.items():
        graph_path = Path(record["path"])
        meta_path = Path(record["meta_path"])
        with graph_path.open("rb") as stream:
            directed, edges, nodes = struct.unpack("<?qq", stream.read(17))
        if (
            graph_path.stat().st_size != record["bytes"]
            or sha256(graph_path) != record["sha256"]
            or crc32(graph_path) != record["crc32"]
            or sha256(meta_path) != record["meta_sha256"]
            or directed is not False
            or nodes != record["nodes"]
            or edges != record["directed_edges"]
        ):
            raise ValueError(f"V6 graph provenance changed: {graph}")

    sidecars = _mapping_sidecars(root)
    expected_mapping_keys = {
        (graph, algorithm)
        for graph in graphs
        for algorithm in reordered_arms
    }
    if not expected_mapping_keys.issubset(sidecars):
        raise ValueError("V6 mapping matrix is incomplete")
    mapping_seconds = {}
    mapping_identities = {}
    mapping_phases = {}
    for key in expected_mapping_keys:
        mapping_seconds[key], mapping_identities[key] = _validate_mapping(
            sidecars[key],
            graph_records[key[0]]["nodes"],
            protocol["mapping_draws_per_reordered_arm"],
            False,
        )
        sidecar = json.loads(sidecars[key].read_text())
        draws = sidecar["mapping_draws"]
        mapping_phases[key] = {
            phase: statistics.median([
                float(draw[phase]) for draw in draws
            ])
            for phase in (
                "compose_grouping_time",
                "compose_community_order_time",
                "compose_intra_order_time",
                "compose_final_assign_time",
            )
            if all(
                isinstance(draw.get(phase), (int, float))
                for draw in draws
            )
        }

    gate_dir = root / "vldb_paper" / "verification_gate"
    manifests = list(gate_dir.glob("manifest-*.json"))
    if len(manifests) != 1:
        raise ValueError("V6 requires exactly one verification manifest")
    gate_manifest_path = manifests[0]
    gate_results_path = gate_dir / "verification_results.json"
    gate_manifest = json.loads(gate_manifest_path.read_text())
    gate_rows = json.loads(gate_results_path.read_text())
    expected_cells = {
        (graph, kernel, algorithm)
        for graph in graphs
        for kernel in kernels
        for algorithm in timing_arms
    }
    gate_index = {}
    for row in gate_rows:
        key = (
            str(row["graph"]),
            str(row["benchmark"]),
            str(row["algo_key"]),
        )
        expected_identity = (
            {"source": "direct", "algo_flags": ["-o", "0"]}
            if key[2] == ORIGINAL
            else mapping_identities[(key[0], key[2])]
        )
        if (
            key in gate_index
            or row.get("gate_id") != gate_manifest["gate_id"]
            or row.get("verification_state") != "pass"
            or row.get("mapping") != expected_identity
        ):
            raise ValueError(f"Invalid V6 verification cell: {key}")
        gate_index[key] = row
    if (
        set(gate_index) != expected_cells
        or gate_manifest["expected_cells"] != len(expected_cells)
        or gate_manifest["completed_cells"] != len(expected_cells)
        or gate_manifest["adjudication"]["total_passes"]
        != len(expected_cells)
    ):
        raise ValueError("V6 verification gate is incomplete")

    timing_path = (
        root / "vldb_paper" / "exp2_speedup" / "speedup_results.json"
    )
    timing_rows = json.loads(timing_path.read_text())
    timing_index = {}
    cohorts = set()
    for row in timing_rows:
        key = (
            str(row["graph"]),
            str(row["benchmark"]),
            str(row["algo_id"]),
        )
        expected_identity = (
            {"source": "direct", "algo_flags": ["-o", "0"]}
            if key[2] == ORIGINAL
            else mapping_identities[(key[0], key[2])]
        )
        trials = row["trial_times"]
        if (
            key in timing_index
            or len(trials) != protocol["trials_per_process"]
            or any(not positive_finite(value) for value in trials)
            or row["verification_gate_id"] != gate_manifest["gate_id"]
            or row["verification_gate_status"] != "pass"
            or row["mapping_identity"] != expected_identity
            or row["mapping_identity_id"] != short_id(expected_identity)
            or row["timing_machine"]["cpu_governors"]
            != ["performance"]
            or row["timing_machine"]["intel_pstate_no_turbo"] != "1"
        ):
            raise ValueError(f"Invalid V6 timing cell: {key}")
        timing_index[key] = statistics.median(
            float(value) for value in trials
        )
        cohorts.add(str(row["cohort_id"]))
    if set(timing_index) != expected_cells or len(cohorts) != 1:
        raise ValueError("V6 timing matrix is incomplete")

    campaign_paths = list(
        (root / "vldb_paper" / "campaigns").glob("*.json")
    )
    timing_campaigns = []
    for path in campaign_paths:
        campaign = json.loads(path.read_text())
        if (
            set(campaign.get("algorithm_filter", [])) == set(timing_arms)
            and campaign.get("trials") == protocol["trials_per_process"]
            and campaign.get("measurement_cohort_id") in cohorts
        ):
            timing_campaigns.append(path)
    if len(timing_campaigns) != 1:
        raise ValueError("V6 timing campaign receipt is missing")
    timing_campaign_path = timing_campaigns[0]

    gates = protocol["kernel_quality_gates"]
    candidate_summaries = {}
    promoted = []
    for candidate in candidates:
        per_graph = {
            graph: geometric_mean([
                timing_index[(graph, kernel, ORIGINAL)]
                / timing_index[(graph, kernel, candidate)]
                for kernel in primary_kernels
            ])
            for graph in graphs
        }
        per_kernel = {
            kernel: geometric_mean([
                timing_index[(graph, kernel, ORIGINAL)]
                / timing_index[(graph, kernel, candidate)]
                for graph in graphs
            ])
            for kernel in kernels
        }
        mapping_ratios = {
            graph: (
                mapping_seconds[(graph, candidate)]
                / min(
                    mapping_seconds[(graph, RABBIT_CSR)],
                    mapping_seconds[(graph, RABBIT_BOOST)],
                )
            )
            for graph in graphs
        }
        gate_results = {
            "primary_kernel_gm": {
                "value": geometric_mean(list(per_graph.values())),
                "minimum":
                    gates["original_over_candidate_primary_gm_min"],
            },
            "pr": {
                "value": per_kernel["pr"],
                "minimum":
                    gates["original_over_candidate_pr_graph_gm_min"],
            },
            "pr_spmv": {
                "value": per_kernel["pr_spmv"],
                "minimum":
                    gates["original_over_candidate_pr_spmv_graph_gm_min"],
            },
            "worst_graph": {
                "value": min(per_graph.values()),
                "minimum": gates["per_graph_primary_gm_min"],
            },
            "graph_wins": {
                "value": sum(value >= 1 for value in per_graph.values()),
                "minimum": gates["graphs_with_primary_point_win_min"],
            },
        }
        for record in gate_results.values():
            record["pass"] = record["value"] >= record["minimum"]
        passed = all(record["pass"] for record in gate_results.values())
        if passed:
            promoted.append(candidate)
        candidate_summaries[candidate] = {
            "original_over_candidate_primary_gm": per_graph,
            "original_over_candidate_by_kernel": per_kernel,
            "candidate_over_min_rabbit_mapping": mapping_ratios,
            "mapping_gm": geometric_mean(list(mapping_ratios.values())),
            "mapping_worst": max(mapping_ratios.values()),
            "mapping_seconds": {
                graph: mapping_seconds[(graph, candidate)]
                for graph in graphs
            },
            "mapping_phases": {
                graph: mapping_phases[(graph, candidate)]
                for graph in graphs
            },
            "winning_graphs": [
                graph for graph, value in per_graph.items()
                if value >= 1
            ],
            "losing_graphs": [
                graph for graph, value in per_graph.items()
                if value < 1
            ],
            "gate_results": gate_results,
            "pass": passed,
        }

    created_at = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()
    analysis = {
        "schema": "graphbrew-dual-arm-v6-fast-intra-analysis/v1",
        "created_at_utc": created_at,
        "inputs": {
            "protocol_sha256": sha256(protocol_path),
            "verification_manifest_sha256": sha256(gate_manifest_path),
            "verification_results_sha256": sha256(gate_results_path),
            "timing_results_sha256": sha256(timing_path),
            "timing_campaign_sha256": sha256(timing_campaign_path),
            "analysis_program_sha256": sha256(Path(__file__).resolve()),
            "execution_repository_commit": protocol["repository_commit"],
            "analysis_repository_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                text=True,
            ).strip(),
        },
        "policy": {
            "timing_estimator": "median of five trials in one process",
            "mapping_estimator": "median complete reorder time over three draws",
            "mapping_role": "descriptive, not a promotion gate",
            "primary_kernels": primary_kernels,
            "claim_boundary": protocol["claim_boundary"],
        },
        "verification": {
            "gate_id": gate_manifest["gate_id"],
            "passed_cells": len(gate_index),
        },
        "candidate_summaries": candidate_summaries,
        "promoted": promoted,
        "pass": bool(promoted),
    }
    winning_sets = {
        tuple(record["winning_graphs"])
        for record in candidate_summaries.values()
    }
    decision = {
        "schema": "graphbrew-dual-arm-v6-fast-intra-decision/v1",
        "created_at_utc": created_at,
        "status": (
            "generic-compact-emit-authorized"
            if promoted
            else "universal-fast-intra-route-closed"
        ),
        "promoted": promoted,
        "shared_regime_split": (
            list(next(iter(winning_sets)))
            if len(winning_sets) == 1 else None
        ),
        "shared_regime_split_observed":
            not promoted and len(winning_sets) == 1,
        "stop_rule": protocol["stop_rule"],
        "claim_boundary": protocol["claim_boundary"],
    }
    return analysis, decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    analysis, decision = analyze(root)
    analysis_path = root / "analysis.json"
    decision_path = root / "decision.json"
    _atomic_write(analysis_path, analysis)
    _atomic_write(decision_path, decision)
    receipt = {
        "schema": "graphbrew-dual-arm-v6-fast-intra-final/v1",
        "frozen_at_utc": analysis["created_at_utc"],
        "repository_clean": True,
        "files": {
            path.name: {
                "path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in (
                root / "protocol.json",
                analysis_path,
                decision_path,
                Path(__file__).resolve(),
            )
        },
        "decision": decision,
    }
    receipt_path = root / "final_receipt.json"
    _atomic_write(receipt_path, receipt)
    print(json.dumps({
        "analysis": str(analysis_path),
        "decision": str(decision_path),
        "final_receipt": str(receipt_path),
        "final_receipt_sha256": sha256(receipt_path),
        "status": decision["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
