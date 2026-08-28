#!/usr/bin/env python3
"""Analyze the scheduler-safe native portfolio feasibility screen."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import statistics
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


EXPECTED_PROTOCOL_SHA256 = (
    "6077b379d919515b251efac86f332c92"
    "1432fa977d25a4fb9d81d9b41e522a1c"
)
ORIGINAL = "0"
RABBIT_CSR = "8:csr"
RABBIT_BOOST = "8:boost"


def _atomic_write(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _expected_mapping_identity(
    sidecar: dict[str, Any],
    sidecar_path: Path,
) -> dict[str, Any]:
    mapping_path = sidecar_path.parent / sidecar["lo_path"]
    draws = sidecar["mapping_draws"]
    return {
        "source": "map",
        "path": mapping_path.name,
        "bytes": mapping_path.stat().st_size,
        "mapping_fingerprint": sidecar["mapping_fingerprint"],
        "generation_policy_id": sidecar["generation_policy_id"],
        "selected_draw": sidecar["selected_draw"],
        "mapping_draws": [
            {
                "path": Path(draw["path"]).name,
                "bytes":
                    (sidecar_path.parent / draw["path"]).stat().st_size,
                "mapping_fingerprint": draw["mapping_fingerprint"],
            }
            for draw in draws
        ],
    }


def analyze(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = root / "protocol.json"
    if sha256(protocol_path) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("Native portfolio protocol hash changed")
    protocol = json.loads(protocol_path.read_text())
    if (
        protocol["schema"] != "graphbrew-native-portfolio-v3/v1"
        or protocol["repository_clean"] is not True
        or protocol["native_portfolio"] != ["0", "2", "4", "5", "7"]
        or protocol["anchors"] != [RABBIT_CSR, RABBIT_BOOST]
        or protocol["kernels"] != ["pr", "pr_spmv", "bfs", "bc"]
        or protocol["primary_reuse"] != 16
        or protocol["trials"] != 5
        or protocol["processes"] != 1
        or protocol["scheduler_contract"] != {
            "process_scheduler": "SCHED_OTHER",
            "process_nice": 0,
            "launcher":
                "systemd-run --user --wait --collect --pipe",
            "runner_fail_closed": True,
        }
    ):
        raise ValueError("Invalid native portfolio frozen contract")

    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    if dirty:
        raise ValueError("Repository is dirty during portfolio analysis")
    subprocess.run(
        [
            "git", "cat-file", "-e",
            f"{protocol['repository_commit']}^{{commit}}",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    graph_records = {
        record["graph"]: record
        for record in protocol["graph_records"]
    }
    if len(graph_records) != 15:
        raise ValueError("Native portfolio graph matrix changed")
    for graph, record in graph_records.items():
        graph_path = Path(record["graph_path"])
        meta_path = graph_path.with_suffix(".sg.meta.json")
        meta = json.loads(meta_path.read_text())
        if (
            graph_path.stat().st_size != record["graph_bytes"]
            or sha256(graph_path) != record["graph_sha256"]
            or crc32(graph_path) != record["graph_crc32"]
            or meta["nodes"] != record["nodes"]
            or meta["directed_edges"] != record["directed_edges"]
        ):
            raise ValueError(f"Portfolio graph provenance changed: {graph}")

    mapping_arms = protocol["mapping_arms"]
    mapping_times = {}
    mapping_identities = {}
    mapping_draw_counts = {}
    for graph in protocol["graphs"]:
        mapping_dir = root / "vldb_mappings" / graph
        for algorithm in mapping_arms:
            safe = algorithm.replace(":", "_")
            sidecar_path = mapping_dir / f"{safe}.json"
            if not sidecar_path.is_file():
                raise FileNotFoundError(
                    f"Missing portfolio mapping: {graph}/{algorithm}"
                )
            sidecar = json.loads(sidecar_path.read_text())
            draws = sidecar["mapping_draws"]
            expected_draws = (
                3 if algorithm in {RABBIT_CSR, RABBIT_BOOST} else 1
            )
            if (
                sidecar["graph"] != graph
                or str(sidecar["algo_key"]) != algorithm
                or sidecar["selected_draw"] != 0
                or sidecar["mapping_draw_count"] != expected_draws
                or len(draws) != expected_draws
                or any(
                    not positive_finite(draw["reorder_time"])
                    for draw in draws
                )
            ):
                raise ValueError(
                    f"Invalid portfolio mapping: {graph}/{algorithm}"
                )
            mapping_path = mapping_dir / sidecar["lo_path"]
            if (
                not mapping_path.is_file()
                or sidecar["lo_bytes"] != mapping_path.stat().st_size
            ):
                raise ValueError(
                    f"Invalid portfolio mapping file: "
                    f"{graph}/{algorithm}"
                )
            mapping_times[(graph, algorithm)] = statistics.median([
                float(draw["reorder_time"]) for draw in draws
            ])
            mapping_identities[(graph, algorithm)] = (
                _expected_mapping_identity(sidecar, sidecar_path)
            )
            mapping_draw_counts[(graph, algorithm)] = expected_draws

    gate_dir = root / "vldb_paper" / "verification_gate"
    manifests = list(gate_dir.glob("manifest-*.json"))
    if len(manifests) != 1:
        raise ValueError("Portfolio requires one verification manifest")
    gate_manifest_path = manifests[0]
    gate_results_path = gate_dir / "verification_results.json"
    gate_manifest = json.loads(gate_manifest_path.read_text())
    gate_rows = json.loads(gate_results_path.read_text())
    algorithms = [
        *protocol["native_portfolio"],
        *protocol["anchors"],
    ]
    expected_cells = {
        (graph, kernel, algorithm)
        for graph in protocol["graphs"]
        for kernel in protocol["kernels"]
        for algorithm in algorithms
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
            or key not in expected_cells
            or row.get("gate_id") != gate_manifest["gate_id"]
            or row.get("verification_state") != "pass"
            or row.get("mapping") != expected_identity
        ):
            raise ValueError(f"Invalid portfolio verification cell: {key}")
        gate_index[key] = row
    if (
        set(gate_index) != expected_cells
        or gate_manifest["expected_cells"] != len(expected_cells)
        or gate_manifest["completed_cells"] != len(expected_cells)
        or gate_manifest["adjudication"]["total_passes"]
        != len(expected_cells)
    ):
        raise ValueError("Portfolio verification gate is incomplete")

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
        trials = [float(value) for value in row["trial_times"]]
        machine = row["timing_machine"]
        if (
            key in cell_seconds
            or key not in expected_cells
            or len(trials) != protocol["trials"]
            or any(not positive_finite(value) for value in trials)
            or row["verification_gate_id"] != gate_manifest["gate_id"]
            or row["verification_gate_status"] != "pass"
            or row["mapping_identity"] != expected_identity
            or row["mapping_identity_id"] != short_id(expected_identity)
            or machine["cpu_governors"] != ["performance"]
            or machine["intel_pstate_no_turbo"] != "1"
            or machine["process_scheduler"] != "SCHED_OTHER"
            or machine["process_nice"] != 0
        ):
            raise ValueError(f"Invalid portfolio timing cell: {key}")
        cell_seconds[key] = statistics.median(trials)
        cohorts.add(str(row["cohort_id"]))
    if set(cell_seconds) != expected_cells or len(cohorts) != 1:
        raise ValueError("Portfolio timing matrix is incomplete")

    campaign_paths = list(
        (root / "vldb_paper" / "campaigns").glob("*.json")
    )
    if len(campaign_paths) != 1:
        raise ValueError("Portfolio campaign receipt is missing")
    campaign_path = campaign_paths[0]
    campaign = json.loads(campaign_path.read_text())
    if (
        campaign["git_revision"]
        != protocol["repository_commit"][:8]
        or campaign["algorithm_filter"] != algorithms
        or campaign["benchmarks"] != protocol["kernels"]
        or campaign["trials"] != protocol["trials"]
    ):
        raise ValueError("Portfolio campaign policy changed")

    native = protocol["native_portfolio"]
    summaries = {}
    action_counts = {}
    for reuse in protocol["reported_reuse"]:
        per_graph = {}
        counts = {
            kernel: {algorithm: 0 for algorithm in native}
            for kernel in protocol["kernels"]
        }
        for graph in protocol["graphs"]:
            original_ratios = []
            rabbit_ratios = []
            actions = {}
            for kernel in protocol["kernels"]:
                totals = {
                    algorithm: (
                        0.0 if algorithm == ORIGINAL
                        else mapping_times[(graph, algorithm)]
                    )
                    + reuse * cell_seconds[(graph, kernel, algorithm)]
                    for algorithm in native
                }
                selected = min(totals, key=totals.get)
                selected_total = totals[selected]
                rabbit_total = min(
                    mapping_times[(graph, anchor)]
                    + reuse * cell_seconds[(graph, kernel, anchor)]
                    for anchor in protocol["anchors"]
                )
                original_total = (
                    reuse * cell_seconds[(graph, kernel, ORIGINAL)]
                )
                actions[kernel] = selected
                counts[kernel][selected] += 1
                original_ratios.append(
                    original_total / selected_total
                )
                rabbit_ratios.append(rabbit_total / selected_total)
            per_graph[graph] = {
                "family": protocol["families"][graph],
                "actions": actions,
                "original_over_oracle":
                    geometric_mean(original_ratios),
                "min_rabbit_over_oracle":
                    geometric_mean(rabbit_ratios),
            }
        summaries[str(reuse)] = {
            "original_over_oracle_gm": geometric_mean([
                record["original_over_oracle"]
                for record in per_graph.values()
            ]),
            "min_rabbit_over_oracle_gm": geometric_mean([
                record["min_rabbit_over_oracle"]
                for record in per_graph.values()
            ]),
            "min_rabbit_over_oracle_worst_graph": min(
                record["min_rabbit_over_oracle"]
                for record in per_graph.values()
            ),
            "per_graph": per_graph,
        }
        action_counts[str(reuse)] = counts

    primary = summaries[str(protocol["primary_reuse"])]
    static = {}
    for algorithm in native:
        per_graph = []
        rabbit = []
        for graph in protocol["graphs"]:
            original_ratios = []
            rabbit_ratios = []
            for kernel in protocol["kernels"]:
                total = (
                    0.0 if algorithm == ORIGINAL
                    else mapping_times[(graph, algorithm)]
                ) + (
                    protocol["primary_reuse"]
                    * cell_seconds[(graph, kernel, algorithm)]
                )
                original_total = (
                    protocol["primary_reuse"]
                    * cell_seconds[(graph, kernel, ORIGINAL)]
                )
                rabbit_total = min(
                    mapping_times[(graph, anchor)]
                    + protocol["primary_reuse"]
                    * cell_seconds[(graph, kernel, anchor)]
                    for anchor in protocol["anchors"]
                )
                original_ratios.append(original_total / total)
                rabbit_ratios.append(rabbit_total / total)
            per_graph.append(geometric_mean(original_ratios))
            rabbit.append(geometric_mean(rabbit_ratios))
        static[algorithm] = {
            "original_over_static_gm": geometric_mean(per_graph),
            "min_rabbit_over_static_gm": geometric_mean(rabbit),
        }

    mapping_ratios = {}
    for algorithm in native:
        if algorithm == ORIGINAL:
            continue
        values = [
            mapping_times[(graph, algorithm)]
            / min(
                mapping_times[(graph, RABBIT_CSR)],
                mapping_times[(graph, RABBIT_BOOST)],
            )
            for graph in protocol["graphs"]
        ]
        mapping_ratios[algorithm] = {
            "gm": geometric_mean(values),
            "worst": max(values),
        }

    thresholds = protocol["feasibility_gates"]
    gates = {
        "oracle_vs_rabbit": {
            "value": primary["min_rabbit_over_oracle_gm"],
            "minimum":
                thresholds["native_oracle_over_min_rabbit_gm_min"],
        },
        "oracle_vs_original": {
            "value": primary["original_over_oracle_gm"],
            "minimum":
                thresholds["native_oracle_over_original_gm_min"],
        },
        "worst_graph_vs_rabbit": {
            "value":
                primary["min_rabbit_over_oracle_worst_graph"],
            "minimum":
                thresholds[
                    "native_oracle_worst_graph_vs_min_rabbit_min"
                ],
        },
        "mapping": {
            "value": max(
                record["gm"] for record in mapping_ratios.values()
            ),
            "maximum":
                thresholds["native_mapping_over_min_rabbit_gm_max"],
        },
    }
    for name in (
        "oracle_vs_rabbit",
        "oracle_vs_original",
        "worst_graph_vs_rabbit",
    ):
        gates[name]["pass"] = (
            gates[name]["value"] >= gates[name]["minimum"]
        )
    gates["mapping"]["pass"] = (
        gates["mapping"]["value"] <= gates["mapping"]["maximum"]
    )
    passed = all(record["pass"] for record in gates.values())

    created_at = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()
    analysis = {
        "schema": "graphbrew-native-portfolio-v3-analysis/v1",
        "created_at_utc": created_at,
        "inputs": {
            "protocol_sha256": sha256(protocol_path),
            "verification_manifest_sha256":
                sha256(gate_manifest_path),
            "verification_results_sha256": sha256(gate_results_path),
            "timing_results_sha256": sha256(timing_path),
            "campaign_sha256": sha256(campaign_path),
            "analysis_program_sha256": sha256(
                Path(__file__).resolve()
            ),
            "execution_repository_commit":
                protocol["repository_commit"],
            "analysis_repository_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                text=True,
            ).strip(),
        },
        "policy": {
            "cell_estimator": "median of five trials",
            "mapping_estimator":
                "single deterministic native draw; "
                "median of three Rabbit draws",
            "primary_reuse": protocol["primary_reuse"],
            "native_portfolio": native,
            "oracle_policy": protocol["oracle_policy"],
            "claim_boundary": protocol["claim_boundary"],
        },
        "verification_cells": len(gate_index),
        "mapping_draw_counts": {
            f"{graph}|{algorithm}": count
            for (graph, algorithm), count
            in mapping_draw_counts.items()
        },
        "mapping_ratios": mapping_ratios,
        "static": static,
        "reuse": summaries,
        "action_counts": action_counts,
        "gates": gates,
        "pass": passed,
    }
    decision = {
        "schema": "graphbrew-native-portfolio-v3-decision/v1",
        "created_at_utc": created_at,
        "status": (
            "native-portfolio-oracle-passed"
            if passed else "native-portfolio-oracle-failed"
        ),
        "primary_reuse": protocol["primary_reuse"],
        "gates": gates,
        "next_step": (
            protocol["selection_followup"]
            if passed else protocol["stop_rule"]
        ),
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
        "schema": "graphbrew-native-portfolio-v3-final/v1",
        "frozen_at_utc": analysis["created_at_utc"],
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
