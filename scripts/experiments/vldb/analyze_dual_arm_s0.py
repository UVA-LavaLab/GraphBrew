#!/usr/bin/env python3
"""Analyze the preregistered dual-arm S0 budget-ladder screen."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import statistics
import sys
import zlib
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def geometric_mean(values: list[float]) -> float:
    if not values or any(
        not math.isfinite(value) or value <= 0
        for value in values
    ):
        raise ValueError("Geometric mean requires positive values")
    return math.exp(
        math.fsum(math.log(value) for value in values) / len(values)
    )


def short_id(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return f"{zlib.crc32(encoded) & 0xffffffff:08x}"


def positive_finite(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0
    )


def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_vector = [
        *(-value for value in left["mapping_ratio_vector"]),
        *left["reuse_ratio_vector"],
    ]
    right_vector = [
        *(-value for value in right["mapping_ratio_vector"]),
        *right["reuse_ratio_vector"],
    ]
    return (
        all(a >= b for a, b in zip(left_vector, right_vector))
        and any(a > b for a, b in zip(left_vector, right_vector))
    )


def analyze(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = root / "protocol.json"
    mapping_path = root / "mapping_receipt.json"
    context_path = root / "context_receipt.json"
    timing_path = root / "vldb_paper/exp2_speedup/speedup_results.json"
    gate_path = root / "vldb_paper/verification_gate/verification_results.json"
    manifest_paths = sorted(
        (root / "vldb_paper/verification_gate").glob("manifest-*.json")
    )
    if len(manifest_paths) != 1:
        raise ValueError(
            f"Expected one verification manifest, found {len(manifest_paths)}"
        )

    protocol = json.loads(protocol_path.read_text())
    mapping_receipt = json.loads(mapping_path.read_text())
    context_receipt = json.loads(context_path.read_text())
    timing_rows = json.loads(timing_path.read_text())
    gate_rows = json.loads(gate_path.read_text())
    gate_manifest = json.loads(manifest_paths[0].read_text())

    if protocol.get("schema") != "graphbrew-dual-arm-s0/v1":
        raise ValueError("Unsupported S0 protocol")
    os.environ["GRAPHBREW_SSSP_POLICY_PATH"] = str(
        root / "sssp_policy.json"
    )
    os.environ["GRAPHBREW_SSSP_TUNING_SNAPSHOT_PATH"] = str(
        root / "sssp_delta_tuning.json"
    )
    from scripts.experiments.vldb import runner
    from scripts.lib.pipeline.benchmark import (
        mapping_permutation_fingerprint,
    )

    runner.configure_artifact_root(root)
    runner.configure_measurement_generation(
        protocol["measurement_generation"],
    )
    runner.configure_runtime_policy(
        int(protocol["threads"]),
        str(protocol["cpu_list"]),
    )
    runner.configure_cache_policy(
        preview=False,
        mode="ultrafast",
        sample_rate=64,
        all_algorithms=False,
        sizes_kib=None,
    )
    runner.configure_execution_mode(dry_run=False)
    if (
        context_receipt.get("schema")
        != "graphbrew-dual-arm-s0-context/v1"
        or context_receipt.get("protocol_sha256")
        != sha256(protocol_path)
        or context_receipt.get("mapping_receipt_sha256")
        != sha256(mapping_path)
        or context_receipt.get("timing_results_sha256")
        != sha256(timing_path)
        or context_receipt.get("host") != protocol.get("host")
        or context_receipt.get("cpu_list") != protocol.get("cpu_list")
        or context_receipt.get("omp_env") != protocol.get("omp_env")
        or context_receipt.get("cpu_governor")
        != protocol.get("cpu_governor")
        or context_receipt.get("intel_pstate_no_turbo")
        != protocol.get("intel_pstate_no_turbo")
    ):
        raise ValueError("S0 execution context receipt is invalid")
    promotion_limit = protocol.get("screening_gates", {}).get(
        "promotion_limit",
    )
    if (
        set(protocol.get("graphs", []))
        != {"soc-Slashdot0811", "wiki-topcats"}
        or len(protocol.get("candidates", [])) != 4
        or set(protocol.get("anchors", [])) != {"8:csr", "8:boost"}
        or protocol.get("screening_trials") != 1
        or protocol.get("mapping_draws_per_arm") != 3
        or protocol.get("reuse") != [1, 2]
        or isinstance(promotion_limit, bool)
        or not isinstance(promotion_limit, int)
        or not 0 <= promotion_limit <= 2
    ):
        raise ValueError("S0 protocol violates fixed campaign constraints")
    expected_arms = (
        len(protocol["candidates"]) + len(protocol["anchors"])
    )
    expected_timing = (
        len(protocol["graphs"])
        * len(protocol["kernels"])
        * expected_arms
    )
    expected_gate = (
        len(protocol["graphs"])
        * len(protocol["kernels"])
        * (expected_arms + 1)
    )
    expected_mappings = len(protocol["graphs"]) * expected_arms
    if (
        protocol.get("expected_timing_cells") != expected_timing
        or protocol.get("expected_verification_cells") != expected_gate
        or protocol.get("expected_mapping_records") != expected_mappings
        or protocol.get("expected_mapping_executions")
        != expected_mappings * protocol["mapping_draws_per_arm"]
    ):
        raise ValueError("S0 protocol counts are not self-consistent")
    if len(timing_rows) != protocol["expected_timing_cells"]:
        raise ValueError(
            f"Expected {protocol['expected_timing_cells']} timing cells, "
            f"found {len(timing_rows)}"
        )
    if len(gate_rows) != protocol["expected_verification_cells"]:
        raise ValueError(
            f"Expected {protocol['expected_verification_cells']} gate cells, "
            f"found {len(gate_rows)}"
        )
    if (
        gate_manifest.get("gate_id") is None
        or gate_manifest.get("schema") != "verification_gate/v1"
        or
        gate_manifest.get("completed_cells")
        != protocol["expected_verification_cells"]
        or gate_manifest.get("adjudication", {}).get("total_passes")
        != protocol["expected_verification_cells"]
    ):
        raise ValueError("S0 verification manifest is incomplete")

    timing_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    cohorts = set()
    for row in timing_rows:
        key = (
            str(row["graph"]),
            str(row["benchmark"]),
            str(row["algo_id"]),
        )
        if key in timing_index:
            raise ValueError(f"Duplicate timing cell: {key}")
        trials = row.get("trial_times", [])
        if (
            len(trials) != protocol["screening_trials"]
            or any(not positive_finite(value) for value in trials)
            or row.get("verification_gate_status") != "pass"
            or row.get("verification_gate_id")
            != gate_manifest["gate_id"]
            or row.get("e2e_join_context_id")
            != context_receipt["timing_e2e_join_context_id"]
        ):
            raise ValueError(f"Invalid timing row: {key}")
        cohorts.add(str(row.get("cohort_id")))
        timing_index[key] = {
            "seconds": statistics.median(float(value) for value in trials),
            "mapping_identity_id": row.get("mapping_identity_id"),
            "mapping_identity": row.get("mapping_identity"),
            "policy_id": row.get("policy_id"),
            "e2e_join_context_id":
                row.get("e2e_join_context_id"),
        }
    if len(cohorts) != 1 or "None" in cohorts:
        raise ValueError(f"Expected one timing cohort: {cohorts}")

    gate_keys = set()
    for row in gate_rows:
        key = (
            str(row["graph"]),
            str(row["benchmark"]),
            str(row["algo_key"]),
        )
        if key in gate_keys:
            raise ValueError(f"Duplicate verification cell: {key}")
        if (
            row.get("verification_state") != "pass"
            or row.get("gate_id") != gate_manifest["gate_id"]
        ):
            raise ValueError(f"Failed verification cell: {key}")
        gate_keys.add(key)

    mapping_index: dict[tuple[str, str], dict[str, Any]] = {}
    for record in mapping_receipt["records"]:
        key = (str(record["graph"]), str(record["algorithm"]))
        if key in mapping_index:
            raise ValueError(f"Duplicate mapping record: {key}")
        raw_draw_times = [
            draw["reorder_time"] for draw in record["draws"]
        ]
        if (
            len(raw_draw_times) != protocol["mapping_draws_per_arm"]
            or any(
                not positive_finite(value)
                for value in raw_draw_times
            )
        ):
            raise ValueError(f"Wrong mapping draw count: {key}")
        draw_times = [float(value) for value in raw_draw_times]
        expected_identity = {
            "source": "map",
            "path": Path(record["mapping_path"]).name,
            "bytes": int(record["mapping_bytes"]),
            "mapping_fingerprint":
                record["selected_mapping_fingerprint"],
            "generation_policy_id": record["generation_policy_id"],
            "selected_draw": 0,
            "mapping_draws": [
                {
                    "path": Path(draw["path"]).name,
                    "bytes": int(draw["bytes"]),
                    "mapping_fingerprint":
                        draw["mapping_fingerprint"],
                }
                for draw in record["draws"]
            ],
        }
        mapping_index[key] = {
            "complete_seconds": statistics.median(draw_times),
            "draw_times": draw_times,
            "mapping_fingerprint":
                record["selected_mapping_fingerprint"],
            "identity": expected_identity,
            "identity_id": short_id(expected_identity),
        }
    if len(mapping_index) != protocol["expected_mapping_records"]:
        raise ValueError("S0 mapping receipt is incomplete")
    context_mappings = {
        (str(record["graph"]), str(record["algorithm"])): record
        for record in context_receipt.get("mapping_records", [])
    }
    if set(context_mappings) != set(mapping_index):
        raise ValueError("Context receipt mapping set is incomplete")
    for key, record in context_mappings.items():
        sidecar_path = Path(record["sidecar_path"])
        sidecar = (
            json.loads(sidecar_path.read_text())
            if sidecar_path.is_file() else {}
        )
        sidecar_draws = sidecar.get("mapping_draws", [])
        if (
            not sidecar_path.is_file()
            or sha256(sidecar_path) != record["sidecar_sha256"]
            or record["cpu_list"] != protocol["cpu_list"]
            or record["omp_env"] != protocol["omp_env"]
            or sidecar.get("graph") != key[0]
            or sidecar.get("algo_key") != key[1]
            or sidecar.get("generation_policy_id")
            != record["generation_policy_id"]
            or record["generation_policy_id"]
            != mapping_index[key]["identity"]["generation_policy_id"]
            or sidecar.get("mapping_fingerprint")
            != mapping_index[key]["mapping_fingerprint"]
            or len(sidecar_draws)
            != protocol["mapping_draws_per_arm"]
            or [
                float(draw.get("reorder_time", -1))
                for draw in sidecar_draws
            ] != mapping_index[key]["draw_times"]
            or [
                {
                    "path": Path(draw.get("path", "")).name,
                    "bytes": int(
                        (
                            sidecar_path.parent
                            / draw.get("path", "")
                        ).stat().st_size
                    ),
                    "mapping_fingerprint":
                        draw.get("mapping_fingerprint"),
                }
                for draw in sidecar_draws
            ] != mapping_index[key]["identity"]["mapping_draws"]
            or any(
                mapping_permutation_fingerprint(
                    sidecar_path.parent / draw.get("path", ""),
                ) != draw.get("mapping_fingerprint")
                for draw in sidecar_draws
            )
            or mapping_permutation_fingerprint(
                Path(
                    next(
                        item["mapping_path"]
                        for item in mapping_receipt["records"]
                        if (
                            str(item["graph"]),
                            str(item["algorithm"]),
                        ) == key
                    )
                )
            ) != mapping_index[key]["mapping_fingerprint"]
        ):
            raise ValueError(f"Mapping execution context mismatch: {key}")

    candidates = list(protocol["candidates"])
    anchors = list(protocol["anchors"])
    graphs = list(protocol["graphs"])
    kernels = list(protocol["kernels"])
    reuse_counts = [int(value) for value in protocol["reuse"]]
    gates = protocol["screening_gates"]
    expected_timing_keys = {
        (graph, kernel, algorithm)
        for graph in graphs
        for kernel in kernels
        for algorithm in (*candidates, *anchors)
    }
    if set(timing_index) != expected_timing_keys:
        raise ValueError("Timing cells do not match the fixed S0 matrix")
    expected_gate_keys = {
        (graph, kernel, algorithm)
        for graph in graphs
        for kernel in kernels
        for algorithm in ("0", *candidates, *anchors)
    }
    if gate_keys != expected_gate_keys:
        raise ValueError("Verification cells do not match the fixed S0 matrix")
    for graph, kernel, algorithm in expected_timing_keys:
        expected_mapping = mapping_index[(graph, algorithm)]
        timing = timing_index[(graph, kernel, algorithm)]
        graph_path = (
            Path("/media/Data/00_GraphDatasets/GraphBrew")
            / graph
            / f"{graph}.sg"
        )
        expected_cohort, expected_policy = runner._kernel_policy_ids(
            graph_path=str(graph_path),
            kind="kernel",
            trials=protocol["screening_trials"],
            executable=runner.BIN_DIR / kernel,
            extra={"mapping": expected_mapping["identity"]},
        )
        if (
            timing["mapping_identity"] != expected_mapping["identity"]
            or timing["mapping_identity_id"]
            != expected_mapping["identity_id"]
            or timing["policy_id"] != expected_policy
            or timing["e2e_join_context_id"]
            != context_receipt["timing_e2e_join_context_id"]
            or next(iter(cohorts)) != expected_cohort
        ):
            raise ValueError(
                f"Timing mapping identity mismatch: "
                f"{graph}/{kernel}/{algorithm}"
            )
        gate_row = next(
            row for row in gate_rows
            if (
                str(row["graph"]),
                str(row["benchmark"]),
                str(row["algo_key"]),
            ) == (graph, kernel, algorithm)
        )
        if gate_row.get("mapping") != expected_mapping["identity"]:
            raise ValueError(
                f"Verification mapping identity mismatch: "
                f"{graph}/{kernel}/{algorithm}"
            )
    for graph in graphs:
        for kernel in kernels:
            gate_row = next(
                row for row in gate_rows
                if (
                    str(row["graph"]),
                    str(row["benchmark"]),
                    str(row["algo_key"]),
                ) == (graph, kernel, "0")
            )
            if gate_row.get("mapping") != {
                "source": "direct",
                "algo_flags": ["-o", "0"],
            }:
                raise ValueError(
                    f"SHUFFLED verification identity mismatch: "
                    f"{graph}/{kernel}"
                )

    candidate_records = []
    for candidate in candidates:
        graph_records = {}
        mapping_ratio_vector = []
        reuse_ratio_vector = []
        for graph in graphs:
            candidate_mapping = mapping_index[
                (graph, candidate)
            ]["complete_seconds"]
            anchor_mappings = {
                anchor: mapping_index[
                    (graph, anchor)
                ]["complete_seconds"]
                for anchor in anchors
            }
            best_mapping = min(anchor_mappings.values())
            mapping_ratio = candidate_mapping / best_mapping
            mapping_ratio_vector.append(mapping_ratio)

            kernel_ratios = {}
            per_reuse = {}
            for kernel in kernels:
                candidate_kernel = timing_index[
                    (graph, kernel, candidate)
                ]["seconds"]
                anchor_kernels = {
                    anchor: timing_index[
                        (graph, kernel, anchor)
                    ]["seconds"]
                    for anchor in anchors
                }
                best_anchor = min(
                    anchors,
                    key=lambda anchor: anchor_kernels[anchor],
                )
                kernel_ratios[kernel] = {
                    "best_rabbit": best_anchor,
                    "best_rabbit_over_candidate":
                        anchor_kernels[best_anchor] / candidate_kernel,
                    "candidate_seconds": candidate_kernel,
                    "rabbit_seconds": anchor_kernels,
                }
            for reuse in reuse_counts:
                ratios = []
                per_kernel = {}
                for kernel in kernels:
                    candidate_total = (
                        candidate_mapping
                        + reuse
                        * timing_index[
                            (graph, kernel, candidate)
                        ]["seconds"]
                    )
                    anchor_totals = {
                        anchor: (
                            anchor_mappings[anchor]
                            + reuse
                            * timing_index[
                                (graph, kernel, anchor)
                            ]["seconds"]
                        )
                        for anchor in anchors
                    }
                    best_anchor = min(anchor_totals, key=anchor_totals.get)
                    ratio = anchor_totals[best_anchor] / candidate_total
                    ratios.append(ratio)
                    per_kernel[kernel] = {
                        "best_rabbit": best_anchor,
                        "best_rabbit_over_candidate": ratio,
                    }
                graph_gm = geometric_mean(ratios)
                per_reuse[str(reuse)] = {
                    "best_rabbit_over_candidate_gm": graph_gm,
                    "per_kernel": per_kernel,
                }
                reuse_ratio_vector.append(graph_gm)

            graph_records[graph] = {
                "candidate_mapping_seconds": candidate_mapping,
                "rabbit_mapping_seconds": anchor_mappings,
                "candidate_over_min_rabbit_mapping": mapping_ratio,
                "best_rabbit_over_candidate_kernel_gm":
                    geometric_mean([
                        kernel_ratios[kernel][
                            "best_rabbit_over_candidate"
                        ]
                        for kernel in kernels
                    ]),
                "kernel": kernel_ratios,
                "reuse": per_reuse,
            }

        hard_gate = (
            max(mapping_ratio_vector)
            <= gates["mapping_candidate_over_min_rabbit_per_graph_max"]
            and min(reuse_ratio_vector)
            >= gates["reuse_min_rabbit_over_candidate_per_graph_min"]
        )
        candidate_records.append({
            "algorithm": candidate,
            "graphs": graph_records,
            "mapping_ratio_vector": mapping_ratio_vector,
            "reuse_ratio_vector": reuse_ratio_vector,
            "mapping_gm": geometric_mean(mapping_ratio_vector),
            "reuse_gm": {
                str(reuse): geometric_mean([
                    graph_records[graph]["reuse"][str(reuse)][
                        "best_rabbit_over_candidate_gm"
                    ]
                    for graph in graphs
                ])
                for reuse in reuse_counts
            },
            "worst_mapping_ratio": max(mapping_ratio_vector),
            "worst_reuse_ratio": min(reuse_ratio_vector),
            "hard_gate_pass": hard_gate,
        })

    eligible = [
        record for record in candidate_records
        if record["hard_gate_pass"]
    ]
    nondominated = [
        record for record in eligible
        if not any(
            dominates(other, record)
            for other in eligible
            if other is not record
        )
    ]
    nondominated.sort(
        key=lambda record: (
            -record["worst_reuse_ratio"],
            record["worst_mapping_ratio"],
            record["mapping_gm"],
        )
    )
    promoted = nondominated[:int(gates["promotion_limit"])]

    analysis = {
        "schema": "graphbrew-dual-arm-s0-analysis/v1",
        "created_at_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "inputs": {
            "protocol": {
                "path": str(protocol_path),
                "sha256": sha256(protocol_path),
            },
            "mapping_receipt": {
                "path": str(mapping_path),
                "sha256": sha256(mapping_path),
            },
            "context_receipt": {
                "path": str(context_path),
                "sha256": sha256(context_path),
            },
            "timing_results": {
                "path": str(timing_path),
                "sha256": sha256(timing_path),
            },
            "verification_results": {
                "path": str(gate_path),
                "sha256": sha256(gate_path),
            },
            "verification_manifest": {
                "path": str(manifest_paths[0]),
                "sha256": sha256(manifest_paths[0]),
            },
            "analysis_program": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
        },
        "policy": {
            "scope": "development-only single-process screen",
            "kernel_estimator": "median of one screening trial",
            "mapping_estimator":
                "median complete reorder time over three draws",
            "rabbit_policy":
                "minimum end-to-end time of CSR and Boost per kernel",
            "ratio_orientation":
                "best Rabbit / GraphBrew; above one favors GraphBrew",
        },
        "cohort_id": next(iter(cohorts)),
        "verification": {
            "expected_cells": protocol["expected_verification_cells"],
            "passing_cells": len(gate_rows),
        },
        "candidates": candidate_records,
        "eligible": [record["algorithm"] for record in eligible],
        "nondominated": [
            record["algorithm"] for record in nondominated
        ],
        "promoted": [record["algorithm"] for record in promoted],
    }
    decision = {
        "schema": "graphbrew-dual-arm-s0-decision/v1",
        "created_at_utc": analysis["created_at_utc"],
        "status": (
            "promote-budget-arm"
            if promoted else "authorize-native-layout-screen"
        ),
        "promoted": analysis["promoted"],
        "eligible": analysis["eligible"],
        "nondominated": analysis["nondominated"],
        "conditional_native_layout_screen_authorized": not bool(promoted),
        "claim_boundary":
            "Development-only screen; promoted arms require independent "
            "process replication and broader development evaluation.",
    }
    return analysis, decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="S0 campaign artifact root",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    analysis, decision = analyze(root)
    outputs = {
        root / "analysis.json": analysis,
        root / "decision.json": decision,
    }
    for path, payload in outputs.items():
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, path)
        print(path.name, sha256(path))
    print("status", decision["status"])
    print("promoted", decision["promoted"])


if __name__ == "__main__":
    main()
