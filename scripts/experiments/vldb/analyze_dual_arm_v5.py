#!/usr/bin/env python3
"""Analyze the fixed community-order Compact-and-Emit screen."""

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
from scripts.experiments.vldb.analyze_dual_arm_v3 import (  # noqa: E402
    crc32,
    mapping_entries,
)
from scripts.experiments.vldb.analyze_dual_arm_v4 import (  # noqa: E402
    expected_mapping_identity,
)
from scripts.lib.pipeline.benchmark import (  # noqa: E402
    mapping_permutation_fingerprint,
)


EXPECTED_PROTOCOL_SHA256 = (
    "2bda010f89b0ac827686bc290d83c99b"
    "582f58fb75aebeee5c6cc7c2f2d5a7c8"
)
EXPECTED_DRAW_AMENDMENT_SHA256 = (
    "653981ff91329102d0b5fbe91ec82f1"
    "f91f11394455acd1c2c3ca0d892b3c935"
)
EXPECTED_ANALYSIS_AMENDMENT_SHA256 = (
    "6d077f901d3e08e6c83b57650b12e5a"
    "39bd97b0428aed80bfd1dc1b9e8e04b49"
)
ORIGINAL = "0"
RABBIT_CSR = "8:csr"
RABBIT_BOOST = "8:boost"


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _mapping_sidecars(root: Path) -> dict[tuple[str, str], Path]:
    sidecars: dict[tuple[str, str], Path] = {}
    for path in (root / "vldb_mappings").glob("*/*.json"):
        record = json.loads(path.read_text())
        if record.get("schema") != "reorder_meta/v6":
            continue
        key = (str(record["graph"]), str(record["algo_key"]))
        if key in sidecars:
            raise ValueError(f"Duplicate V5 mapping sidecar: {key}")
        sidecars[key] = path
    return sidecars


def _validate_mapping(
    sidecar_path: Path,
    graph_nodes: int,
    expected_draws: int,
    require_compact_telemetry: bool,
) -> tuple[float, dict[str, Any]]:
    sidecar = json.loads(sidecar_path.read_text())
    mapping_path = sidecar_path.parent / sidecar["lo_path"]
    draws = sidecar["mapping_draws"]
    phase_keys = (
        "compose_grouping_time_passes",
        "compose_community_order_time_passes",
        "compose_vertex_map_time_passes",
        "compose_intra_order_time_passes",
        "compose_final_assign_time_passes",
    )
    if (
        sidecar["selected_draw"] != 0
        or sidecar["mapping_draw_count"] != expected_draws
        or len(draws) != expected_draws
        or {int(draw["draw"]) for draw in draws}
        != set(range(expected_draws))
        or mapping_entries(mapping_path) != graph_nodes
        or mapping_path.stat().st_size != sidecar["lo_bytes"]
        or mapping_permutation_fingerprint(mapping_path)
        != sidecar["mapping_fingerprint"]
    ):
        raise ValueError(f"Invalid V5 mapping sidecar: {sidecar_path}")

    draw_times = []
    identity_draws = []
    for draw in draws:
        draw_path = sidecar_path.parent / draw["path"]
        fingerprint = mapping_permutation_fingerprint(draw_path)
        if (
            mapping_entries(draw_path) != graph_nodes
            or fingerprint != draw["mapping_fingerprint"]
            or not positive_finite(draw["reorder_time"])
            or (
                require_compact_telemetry
                and (
                    any(
                        len(draw.get(key, []))
                        != len(draw.get("reorder_core_time_passes", []))
                        for key in phase_keys
                    )
                    or not isinstance(
                        draw.get("membership_empty_fraction"),
                        (int, float),
                    )
                )
            )
        ):
            raise ValueError(f"Invalid V5 mapping draw: {draw_path}")
        draw_times.append(float(draw["reorder_time"]))
        identity_draws.append({
            "draw": int(draw["draw"]),
            "path": str(draw_path),
            "bytes": draw_path.stat().st_size,
            "mapping_fingerprint": fingerprint,
        })

    validated = {
        "mapping_path": str(mapping_path),
        "mapping_bytes": mapping_path.stat().st_size,
        "mapping_fingerprint": sidecar["mapping_fingerprint"],
        "generation_policy_id": sidecar["generation_policy_id"],
        "selected_draw": sidecar["selected_draw"],
        "draws": identity_draws,
    }
    return statistics.median(draw_times), expected_mapping_identity(validated)


def _ratio_summary(
    *,
    graphs: list[str],
    kernels: list[str],
    candidate: str,
    reuse: int,
    mapping_seconds: dict[tuple[str, str], float],
    cell_seconds: dict[tuple[str, str, str], float],
) -> dict[str, Any]:
    per_graph = {}
    for graph in graphs:
        candidate_mapping = mapping_seconds[(graph, candidate)]
        original_ratios = []
        csr_ratios = []
        boost_ratios = []
        oracle_ratios = []
        for kernel in kernels:
            candidate_total = (
                candidate_mapping
                + reuse * cell_seconds[(graph, kernel, candidate)]
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
            original_ratios.append(original_total / candidate_total)
            csr_ratios.append(csr_total / candidate_total)
            boost_ratios.append(boost_total / candidate_total)
            oracle_ratios.append(min(csr_total, boost_total) / candidate_total)
        per_graph[graph] = {
            "original_over_candidate": geometric_mean(original_ratios),
            "rabbit_csr_over_candidate": geometric_mean(csr_ratios),
            "rabbit_boost_over_candidate": geometric_mean(boost_ratios),
            "min_rabbit_over_candidate": geometric_mean(oracle_ratios),
        }
    return {
        "reuse": reuse,
        "graph_gm": {
            metric: geometric_mean([
                record[metric] for record in per_graph.values()
            ])
            for metric in next(iter(per_graph.values()))
        },
        "worst_graph": {
            metric: min(record[metric] for record in per_graph.values())
            for metric in next(iter(per_graph.values()))
        },
        "per_graph": per_graph,
    }


def analyze(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = root / "protocol.json"
    draw_amendment_path = root / "protocol_amendment_pre_timing.json"
    analysis_amendment_path = root / "protocol_amendment_pre_analysis.json"
    if sha256(protocol_path) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("V5 protocol hash changed")
    if sha256(draw_amendment_path) != EXPECTED_DRAW_AMENDMENT_SHA256:
        raise ValueError("V5 draw amendment hash changed")
    if (
        sha256(analysis_amendment_path)
        != EXPECTED_ANALYSIS_AMENDMENT_SHA256
    ):
        raise ValueError("V5 analysis amendment hash changed")

    protocol = json.loads(protocol_path.read_text())
    analysis_amendment = json.loads(analysis_amendment_path.read_text())
    candidates = [
        protocol["candidate_control"],
        *protocol["candidate_variants"],
    ]
    reordered_arms = [*candidates, RABBIT_CSR, RABBIT_BOOST]
    timing_arms = [ORIGINAL, *reordered_arms]
    graphs = list(protocol["graphs"])
    kernels = list(protocol["kernels"])
    primary_kernels = list(protocol["primary_kernel_families"])
    if (
        protocol["schema"]
        != "graphbrew-dual-arm-v5-community-order-screen/v1"
        or protocol["repository_clean"] is not True
        or len(graphs) != 5
        or len(kernels) != 7
        or len(primary_kernels) != 5
        or len(candidates) != 5
        or protocol["fixed_anchors"]
        != [ORIGINAL, RABBIT_CSR, RABBIT_BOOST]
        or protocol["mapping_draws_per_reordered_arm"] != 3
        or protocol["timing_processes"] != 1
        or protocol["trials_per_process"] != 5
        or analysis_amendment["reuse_candidates"] != list(range(1, 257))
    ):
        raise ValueError("Invalid V5 frozen contract")

    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    if dirty:
        raise ValueError("Repository is dirty during V5 analysis")
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
        raise ValueError("V5 graph records changed")
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
            raise ValueError(f"V5 graph provenance changed: {graph}")

    sidecars = _mapping_sidecars(root)
    expected_mapping_keys = {
        (graph, algorithm)
        for graph in graphs
        for algorithm in reordered_arms
    }
    if not expected_mapping_keys.issubset(sidecars):
        raise ValueError("V5 mapping matrix is incomplete")
    mapping_seconds = {}
    mapping_identities = {}
    for key in expected_mapping_keys:
        mapping_seconds[key], mapping_identities[key] = _validate_mapping(
            sidecars[key],
            graph_records[key[0]]["nodes"],
            protocol["mapping_draws_per_reordered_arm"],
            key[1] in candidates,
        )

    gate_dir = root / "vldb_paper" / "verification_gate"
    manifests = list(gate_dir.glob("manifest-*.json"))
    if len(manifests) != 1:
        raise ValueError("V5 requires exactly one verification manifest")
    gate_manifest_path = manifests[0]
    gate_results_path = gate_dir / "verification_results.json"
    gate_manifest = json.loads(gate_manifest_path.read_text())
    gate_rows = json.loads(gate_results_path.read_text())
    expected_gate_keys = {
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
        if key in gate_index:
            raise ValueError(f"Duplicate V5 verification cell: {key}")
        expected_identity = (
            {"source": "direct", "algo_flags": ["-o", "0"]}
            if key[2] == ORIGINAL
            else mapping_identities[(key[0], key[2])]
        )
        if (
            row.get("gate_id") != gate_manifest["gate_id"]
            or row.get("verification_state") != "pass"
            or row.get("mapping") != expected_identity
        ):
            raise ValueError(f"Invalid V5 verification cell: {key}")
        gate_index[key] = row
    if (
        set(gate_index) != expected_gate_keys
        or gate_manifest["expected_cells"] != len(expected_gate_keys)
        or gate_manifest["completed_cells"] != len(expected_gate_keys)
        or gate_manifest["adjudication"]["total_passes"]
        != len(expected_gate_keys)
    ):
        raise ValueError("V5 verification gate is incomplete")

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
        if key in timing_index:
            raise ValueError(f"Duplicate V5 timing cell: {key}")
        expected_identity = (
            {"source": "direct", "algo_flags": ["-o", "0"]}
            if key[2] == ORIGINAL
            else mapping_identities[(key[0], key[2])]
        )
        trials = row["trial_times"]
        if (
            len(trials) != protocol["trials_per_process"]
            or any(not positive_finite(value) for value in trials)
            or row["verification_gate_id"] != gate_manifest["gate_id"]
            or row["verification_gate_status"] != "pass"
            or row["mapping_identity"] != expected_identity
            or row["mapping_identity_id"] != short_id(expected_identity)
            or row["timing_machine"]["cpu_governors"]
            != ["performance"]
            or row["timing_machine"]["intel_pstate_no_turbo"] != "1"
        ):
            raise ValueError(f"Invalid V5 timing cell: {key}")
        timing_index[key] = statistics.median(
            float(value) for value in trials
        )
        cohorts.add(str(row["cohort_id"]))
    if set(timing_index) != expected_gate_keys or len(cohorts) != 1:
        raise ValueError("V5 timing matrix is incomplete")

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
        raise ValueError("V5 timing campaign receipt is missing")
    timing_campaign_path = timing_campaigns[0]

    gates = protocol["screen_gates"]
    candidate_summaries = {}
    promoted = []
    for candidate in candidates:
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
        kernel_quality_per_graph = {
            graph: geometric_mean([
                timing_index[(graph, kernel, ORIGINAL)]
                / timing_index[(graph, kernel, candidate)]
                for kernel in primary_kernels
            ])
            for graph in graphs
        }
        kernel_only = {
            kernel: {
                "original_over_candidate_graph_gm": geometric_mean([
                    timing_index[(graph, kernel, ORIGINAL)]
                    / timing_index[(graph, kernel, candidate)]
                    for graph in graphs
                ]),
                "rabbit_csr_over_candidate_graph_gm": geometric_mean([
                    timing_index[(graph, kernel, RABBIT_CSR)]
                    / timing_index[(graph, kernel, candidate)]
                    for graph in graphs
                ]),
                "rabbit_boost_over_candidate_graph_gm": geometric_mean([
                    timing_index[(graph, kernel, RABBIT_BOOST)]
                    / timing_index[(graph, kernel, candidate)]
                    for graph in graphs
                ]),
            }
            for kernel in kernels
        }
        reuse_records = {
            reuse: _ratio_summary(
                graphs=graphs,
                kernels=primary_kernels,
                candidate=candidate,
                reuse=reuse,
                mapping_seconds=mapping_seconds,
                cell_seconds=timing_index,
            )
            for reuse in analysis_amendment["reuse_candidates"]
        }
        feasible = [
            reuse for reuse, record in reuse_records.items()
            if (
                record["graph_gm"]["original_over_candidate"] > 1
                and record["graph_gm"]["rabbit_csr_over_candidate"] > 1
                and record["worst_graph"]["original_over_candidate"]
                >= gates["bounded_harm_at_selected_reuse_min"]
                and record["worst_graph"]["rabbit_csr_over_candidate"]
                >= gates["bounded_harm_at_selected_reuse_min"]
            )
        ]
        selected_reuse = min(feasible) if feasible else None
        gate_results = {
            "mapping_gm": {
                "value": geometric_mean(list(mapping_ratios.values())),
                "maximum":
                    gates["candidate_over_min_rabbit_mapping_gm_max"],
            },
            "mapping_worst": {
                "value": max(mapping_ratios.values()),
                "maximum":
                    gates["candidate_over_min_rabbit_mapping_per_graph_max"],
            },
            "kernel_quality": {
                "value":
                    geometric_mean(list(kernel_quality_per_graph.values())),
                "minimum":
                    gates["original_over_candidate_kernel_gm_min"],
            },
            "nonempty_overlap": {
                "feasible_reuses": feasible,
                "selected_reuse": selected_reuse,
            },
        }
        gate_results["mapping_gm"]["pass"] = (
            gate_results["mapping_gm"]["value"]
            <= gate_results["mapping_gm"]["maximum"]
        )
        gate_results["mapping_worst"]["pass"] = (
            gate_results["mapping_worst"]["value"]
            <= gate_results["mapping_worst"]["maximum"]
        )
        gate_results["kernel_quality"]["pass"] = (
            gate_results["kernel_quality"]["value"]
            >= gate_results["kernel_quality"]["minimum"]
        )
        gate_results["nonempty_overlap"]["pass"] = bool(feasible)
        passed = all(record["pass"] for record in gate_results.values())
        if passed:
            promoted.append(candidate)
        candidate_summaries[candidate] = {
            "mapping_seconds": {
                graph: mapping_seconds[(graph, candidate)]
                for graph in graphs
            },
            "candidate_over_min_rabbit_mapping": mapping_ratios,
            "original_over_candidate_kernel_gm":
                kernel_quality_per_graph,
            "kernel_only": kernel_only,
            "reuse_grid": {
                str(reuse): reuse_records[reuse]
                for reuse in protocol["reuse_grid"]
            },
            "selected_reuse": (
                reuse_records[selected_reuse]
                if selected_reuse is not None else None
            ),
            "gate_results": gate_results,
            "pass": passed,
        }

    created_at = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()
    analysis = {
        "schema": "graphbrew-dual-arm-v5-community-order-analysis/v1",
        "created_at_utc": created_at,
        "inputs": {
            "protocol_sha256": sha256(protocol_path),
            "draw_amendment_sha256": sha256(draw_amendment_path),
            "analysis_amendment_sha256": sha256(
                analysis_amendment_path
            ),
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
            "original_mapping_seconds": 0,
            "primary_kernels": primary_kernels,
            "reuse_selection":
                analysis_amendment["selection_rule"],
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
    decision = {
        "schema": "graphbrew-dual-arm-v5-community-order-decision/v1",
        "created_at_utc": created_at,
        "status": (
            "fixed-community-order-promoted"
            if promoted
            else "fixed-community-order-screen-closed"
        ),
        "promoted": promoted,
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
        "schema": "graphbrew-dual-arm-v5-community-order-final/v1",
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
                root / "protocol_amendment_pre_timing.json",
                root / "protocol_amendment_pre_analysis.json",
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
