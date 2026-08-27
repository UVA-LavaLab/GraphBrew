#!/usr/bin/env python3
"""Analyze the native mid-reuse development rule."""

from __future__ import annotations

import argparse
import datetime
import json
import math
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
    "31bd240f3787b5c89625d2027620821e"
    "102f32c612af7960551b7a672435270d"
)
EXPECTED_ANALYSIS_AMENDMENT_SHA256 = (
    "c13cfe1ed7111d6783ae618af90280af"
    "9ac41aec5edd8accb819237b7834ce01"
)
EXPECTED_CORRECTION_SHA256 = (
    "a544198abd09cd9f060f908892134d3d"
    "1501200cfce51b739e94892ea1b0b9d4"
)
EXPECTED_RULE_SHA256 = (
    "738365f8dc00b39d1a6bee806fe6851b"
    "85f23b7eeac6a1e7e69ac4d4a680af3d"
)


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _summarize_policy(
    *,
    graphs: list[str],
    kernels: list[str],
    reuse: int,
    action,
    mapping_seconds: dict[tuple[str, str], float],
    cell_seconds: dict[tuple[str, str, str], float],
) -> dict[str, Any]:
    per_graph = {}
    for graph in graphs:
        ratios = {
            "original_over_policy": [],
            "rabbit_csr_over_policy": [],
            "rabbit_boost_over_policy": [],
            "min_rabbit_over_policy": [],
        }
        actions = {}
        for kernel in kernels:
            algorithm = action(graph, kernel)
            actions[kernel] = algorithm
            policy_total = (
                (0.0 if algorithm == ORIGINAL
                 else mapping_seconds[(graph, algorithm)])
                + reuse * cell_seconds[(graph, kernel, algorithm)]
            )
            original_total = reuse * cell_seconds[
                (graph, kernel, ORIGINAL)
            ]
            csr_total = (
                mapping_seconds[(graph, RABBIT_CSR)]
                + reuse * cell_seconds[(graph, kernel, RABBIT_CSR)]
            )
            boost_total = (
                mapping_seconds[(graph, RABBIT_BOOST)]
                + reuse * cell_seconds[(graph, kernel, RABBIT_BOOST)]
            )
            ratios["original_over_policy"].append(
                original_total / policy_total
            )
            ratios["rabbit_csr_over_policy"].append(
                csr_total / policy_total
            )
            ratios["rabbit_boost_over_policy"].append(
                boost_total / policy_total
            )
            ratios["min_rabbit_over_policy"].append(
                min(csr_total, boost_total) / policy_total
            )
        per_graph[graph] = {
            "actions": actions,
            **{
                metric: geometric_mean(values)
                for metric, values in ratios.items()
            },
        }
    metrics = (
        "original_over_policy",
        "rabbit_csr_over_policy",
        "rabbit_boost_over_policy",
        "min_rabbit_over_policy",
    )
    return {
        "reuse": reuse,
        "graph_gm": {
            metric: geometric_mean([
                record[metric] for record in per_graph.values()
            ])
            for metric in metrics
        },
        "worst_graph": {
            metric: min(record[metric] for record in per_graph.values())
            for metric in metrics
        },
        "per_graph": per_graph,
    }


def analyze(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = root / "protocol.json"
    analysis_amendment_path = root / "protocol_amendment_pre_analysis.json"
    correction_path = root / "protocol_correction.json"
    rule_path = root / "rule_amendment_v2.json"
    expected_hashes = {
        protocol_path: EXPECTED_PROTOCOL_SHA256,
        analysis_amendment_path: EXPECTED_ANALYSIS_AMENDMENT_SHA256,
        correction_path: EXPECTED_CORRECTION_SHA256,
        rule_path: EXPECTED_RULE_SHA256,
    }
    for path, expected in expected_hashes.items():
        if sha256(path) != expected:
            raise ValueError(f"V8 frozen input changed: {path.name}")

    protocol = json.loads(protocol_path.read_text())
    correction = json.loads(correction_path.read_text())
    rule = json.loads(rule_path.read_text())
    graphs = list(protocol["graphs"])
    kernels = list(protocol["kernels"])
    mapping_arms = list(protocol["mapping_arms"])
    timing_arms = [ORIGINAL, *mapping_arms]
    if (
        protocol["schema"]
        != "graphbrew-dual-arm-v8-native-workload-router/v1"
        or correction["schema"]
        != "graphbrew-dual-arm-v8-protocol-correction/v1"
        or rule["schema"]
        != "graphbrew-dual-arm-v8-native-midreuse-rule/v2"
        or protocol["repository_clean"] is not True
        or rule["repository_clean"] is not True
        or len(graphs) != 12
        or kernels != ["pr", "pr_spmv", "bfs", "bc"]
        or rule["predicate"]["minimum_nodes"] != 1 << 17
        or rule["predicate"]["exact_reuse"] != 40
        or rule["actions"] != {"eligible": "7", "fallback": "0"}
    ):
        raise ValueError("Invalid V8 frozen contract")

    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    if dirty:
        raise ValueError("Repository is dirty during V8 analysis")
    for commit in (
        protocol["repository_commit"],
        rule["repository_commit"],
    ):
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=PROJECT_ROOT,
            check=True,
        )

    graph_records = {
        str(record["graph"]): record
        for record in protocol["graph_records"]
    }
    if set(graph_records) != set(graphs):
        raise ValueError("V8 graph records changed")
    for graph, record in graph_records.items():
        graph_path = Path(record["path"])
        meta_path = Path(record["meta_path"])
        with graph_path.open("rb") as stream:
            directed, edges, nodes = struct.unpack("<?qq", stream.read(17))
        if (
            graph_path.stat().st_size != record["bytes"]
            or sha256(graph_path) != record["sha256"]
            or sha256(meta_path) != record["meta_sha256"]
            or directed is not False
            or nodes != record["nodes"]
            or edges != record["directed_edges"]
        ):
            raise ValueError(f"V8 graph provenance changed: {graph}")

    expected_draws = correction["correction"]["mapping_draws"]
    sidecars = _mapping_sidecars(root)
    expected_mapping_keys = {
        (graph, algorithm)
        for graph in graphs
        for algorithm in mapping_arms
    }
    if set(sidecars) != expected_mapping_keys:
        raise ValueError("V8 mapping matrix changed")
    mapping_seconds = {}
    mapping_identities = {}
    for graph, algorithm in expected_mapping_keys:
        mapping_seconds[(graph, algorithm)], mapping_identities[
            (graph, algorithm)
        ] = _validate_mapping(
            sidecars[(graph, algorithm)],
            graph_records[graph]["nodes"],
            int(expected_draws[algorithm]),
            False,
        )

    gate_dir = root / "vldb_paper" / "verification_gate"
    manifests = list(gate_dir.glob("manifest-*.json"))
    if len(manifests) != 1:
        raise ValueError("V8 requires one verification manifest")
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
            raise ValueError(f"Invalid V8 verification cell: {key}")
        gate_index[key] = row
    if (
        set(gate_index) != expected_cells
        or gate_manifest["expected_cells"] != len(expected_cells)
        or gate_manifest["completed_cells"] != len(expected_cells)
        or gate_manifest["adjudication"]["total_passes"]
        != len(expected_cells)
    ):
        raise ValueError("V8 verification gate is incomplete")

    timing_path = (
        root / "vldb_paper" / "exp2_speedup" / "speedup_results.json"
    )
    timing_rows = json.loads(timing_path.read_text())
    cell_seconds = {}
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
            key in cell_seconds
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
            raise ValueError(f"Invalid V8 timing cell: {key}")
        cell_seconds[key] = statistics.median(
            float(value) for value in trials
        )
        cohorts.add(str(row["cohort_id"]))
    if set(cell_seconds) != expected_cells or len(cohorts) != 1:
        raise ValueError("V8 timing matrix is incomplete")

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
        raise ValueError("V8 timing campaign receipt is missing")
    timing_campaign_path = timing_campaigns[0]

    fixed_router = protocol["fixed_router"]
    fixed = {
        str(reuse): _summarize_policy(
            graphs=graphs,
            kernels=kernels,
            reuse=reuse,
            action=lambda _graph, kernel: fixed_router[kernel],
            mapping_seconds=mapping_seconds,
            cell_seconds=cell_seconds,
        )
        for reuse in protocol["reported_reuse"]
    }

    min_nodes = int(rule["predicate"]["minimum_nodes"])
    candidate = rule["actions"]["eligible"]
    fallback = rule["actions"]["fallback"]
    guarded_all = {
        reuse: _summarize_policy(
            graphs=graphs,
            kernels=kernels,
            reuse=reuse,
            action=lambda graph, _kernel: (
                candidate
                if graph_records[graph]["nodes"] >= min_nodes
                else fallback
            ),
            mapping_seconds=mapping_seconds,
            cell_seconds=cell_seconds,
        )
        for reuse in protocol["reuse_candidates"]
    }
    safe_reuses = [
        reuse for reuse, record in guarded_all.items()
        if (
            record["graph_gm"]["original_over_policy"] > 1
            and record["graph_gm"]["min_rabbit_over_policy"] > 1
            and record["worst_graph"]["original_over_policy"] >= 0.9
            and record["worst_graph"]["min_rabbit_over_policy"] >= 0.9
        )
    ]
    primary = guarded_all[int(rule["predicate"]["exact_reuse"])]
    gates = {
        "original": {
            "value": primary["graph_gm"]["original_over_policy"],
            "minimum": 1.0,
        },
        "min_rabbit": {
            "value": primary["graph_gm"]["min_rabbit_over_policy"],
            "minimum": 1.0,
        },
        "original_worst_graph": {
            "value": primary["worst_graph"]["original_over_policy"],
            "minimum": 0.9,
        },
        "min_rabbit_worst_graph": {
            "value": primary["worst_graph"]["min_rabbit_over_policy"],
            "minimum": 0.9,
        },
    }
    for record in gates.values():
        record["pass"] = record["value"] >= record["minimum"]
    passed = all(record["pass"] for record in gates.values())

    native_arms = [ORIGINAL, "2", "4", "5", "7", "11"]
    oracle = {}
    for reuse in protocol["reported_reuse"]:
        oracle[str(reuse)] = _summarize_policy(
            graphs=graphs,
            kernels=kernels,
            reuse=reuse,
            action=lambda graph, kernel, r=reuse: min(
                native_arms,
                key=lambda algorithm: (
                    (0.0 if algorithm == ORIGINAL
                     else mapping_seconds[(graph, algorithm)])
                    + r * cell_seconds[(graph, kernel, algorithm)]
                ),
            ),
            mapping_seconds=mapping_seconds,
            cell_seconds=cell_seconds,
        )

    created_at = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()
    analysis = {
        "schema": "graphbrew-dual-arm-v8-native-midreuse-analysis/v1",
        "created_at_utc": created_at,
        "inputs": {
            "protocol_sha256": sha256(protocol_path),
            "analysis_amendment_sha256": sha256(
                analysis_amendment_path
            ),
            "protocol_correction_sha256": sha256(correction_path),
            "rule_amendment_v2_sha256": sha256(rule_path),
            "verification_manifest_sha256": sha256(gate_manifest_path),
            "verification_results_sha256": sha256(gate_results_path),
            "timing_results_sha256": sha256(timing_path),
            "timing_campaign_sha256": sha256(timing_campaign_path),
            "analysis_program_sha256": sha256(Path(__file__).resolve()),
            "execution_repository_commit": protocol["repository_commit"],
            "rule_repository_commit": rule["repository_commit"],
            "analysis_repository_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                text=True,
            ).strip(),
        },
        "policy": {
            "timing_estimator": "median of five trials in one process",
            "mapping_estimator":
                "single deterministic draw for native arms; "
                "median of three draws for Rabbit",
            "fixed_router": fixed_router,
            "derived_rule": rule,
            "claim_boundary": protocol["claim_boundary"],
        },
        "verification": {
            "gate_id": gate_manifest["gate_id"],
            "passed_cells": len(gate_index),
        },
        "fixed_router": fixed,
        "native_oracle": oracle,
        "native_midreuse_rule": {
            "primary": primary,
            "safe_reuses": safe_reuses,
            "safe_reuse_interval": (
                [min(safe_reuses), max(safe_reuses)]
                if safe_reuses else None
            ),
            "gates": gates,
            "pass": passed,
        },
        "pass": passed,
    }
    decision = {
        "schema": "graphbrew-dual-arm-v8-native-midreuse-decision/v1",
        "created_at_utc": created_at,
        "status": (
            "native-midreuse-rule-development-passed"
            if passed
            else "native-midreuse-rule-development-failed"
        ),
        "cli": rule["cli"],
        "safe_reuse_interval": (
            [min(safe_reuses), max(safe_reuses)]
            if safe_reuses else None
        ),
        "gates": gates,
        "next_gate":
            "Untouched clustered terminal confirmation is mandatory.",
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
        "schema": "graphbrew-dual-arm-v8-native-midreuse-final/v1",
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
                root / "protocol_amendment_pre_analysis.json",
                root / "protocol_correction.json",
                root / "rule_amendment_v2.json",
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
