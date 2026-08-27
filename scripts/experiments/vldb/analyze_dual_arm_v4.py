#!/usr/bin/env python3
"""Analyze the replicated Compact-and-Emit balanced-arm matrix."""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import random
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
from scripts.lib.pipeline.benchmark import (  # noqa: E402
    mapping_permutation_fingerprint,
)


EXPECTED_PROTOCOL_SHA256 = (
    "3bfa6f0ced5b7bcc20b014f81136a962"
    "e82edd2ebca9103f2e7b120d2c0eb4d7"
)
EXPECTED_EXECUTION_MANIFEST_SHA256 = (
    "6d81286460de8d16ad3a1579217ac094"
    "7dc13373464de1ac17d856e0c276b4ef"
)
BOOTSTRAP_SAMPLES = 50_000
BOOTSTRAP_SEED = 0xC0A6E47


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def expected_mapping_identity(
    record: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source": "map",
        "path": Path(record["mapping_path"]).name,
        "bytes": int(record["mapping_bytes"]),
        "mapping_fingerprint": record["mapping_fingerprint"],
        "generation_policy_id": record["generation_policy_id"],
        "selected_draw": int(record["selected_draw"]),
        "mapping_draws": [
            {
                "path": Path(draw["path"]).name,
                "bytes": int(draw["bytes"]),
                "mapping_fingerprint": draw["mapping_fingerprint"],
            }
            for draw in record["draws"]
        ],
    }


def analyze(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = root / "protocol.json"
    execution_path = root / "execution_manifest.json"
    if sha256(protocol_path) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("V4 protocol hash changed")
    if sha256(execution_path) != EXPECTED_EXECUTION_MANIFEST_SHA256:
        raise ValueError("V4 execution manifest hash changed")
    protocol = json.loads(protocol_path.read_text())
    execution = json.loads(execution_path.read_text())
    if (
        protocol.get("schema")
        != "graphbrew-dual-arm-v4-balanced/v1"
        or execution.get("schema")
        != "graphbrew-dual-arm-v4-execution/v1"
        or execution.get("protocol_sha256")
        != EXPECTED_PROTOCOL_SHA256
        or execution.get("repository_commit")
        != protocol["repository_commit"]
        or execution.get("repository_clean_at_freeze") is not True
        or protocol.get("repository_clean") is not True
        or len(protocol.get("graphs", [])) != 5
        or len(protocol.get("kernels", [])) != 7
        or len(protocol.get("arms", [])) != 3
        or protocol.get("mapping_draws_per_arm") != 3
        or protocol.get("trials_per_process") != 7
        or protocol.get("expected_timing_cells") != 105
    ):
        raise ValueError("Invalid V4 frozen contract")

    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    if dirty:
        raise ValueError("Repository is dirty during V4 analysis")
    if dirty:
        raise ValueError("Repository is dirty during V4 analysis")
    subprocess.run(
        [
            "git", "cat-file", "-e",
            f"{protocol['repository_commit']}^{{commit}}",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    converter = Path(execution["converter_path"])
    if sha256(converter) != execution["converter_sha256"]:
        raise ValueError("V4 converter binary changed")

    manifest_graphs = {
        str(record["graph"]): record
        for record in execution["graphs"]
    }
    graph_info = {}
    for graph in protocol["graphs"]:
        frozen = manifest_graphs[graph]
        graph_path = Path(frozen["path"])
        meta_path = Path(frozen["meta_path"])
        meta = json.loads(meta_path.read_text())
        with graph_path.open("rb") as stream:
            header = stream.read(17)
        directed, edges, nodes = struct.unpack("<?qq", header)
        if (
            graph_path.stat().st_size != frozen["bytes"]
            or sha256(graph_path) != frozen["sha256"]
            or crc32(graph_path) != frozen["crc32"]
            or sha256(meta_path) != frozen["meta_sha256"]
            or directed is not False
            or nodes != frozen["nodes"]
            or edges != frozen["directed_edges"]
            or meta["output_crc32"] != frozen["crc32"]
        ):
            raise ValueError(f"V4 graph provenance changed: {graph}")
        graph_info[graph] = frozen

    manifest_mappings = {
        (str(record["graph"]), str(record["algorithm"])): record
        for record in execution["mapping_records"]
    }
    expected_mapping_keys = {
        (graph, algorithm)
        for graph in protocol["graphs"]
        for algorithm in protocol["mapping_arms"]
    }
    if set(manifest_mappings) != expected_mapping_keys:
        raise ValueError("V4 mapping matrix changed")
    mapping_times = {}
    mapping_identities = {}
    serial_fingerprints = {}
    for key, frozen in manifest_mappings.items():
        graph, algorithm = key
        sidecar_path = Path(frozen["sidecar_path"])
        mapping_path = Path(frozen["mapping_path"])
        sidecar = json.loads(sidecar_path.read_text())
        draws = sidecar["mapping_draws"]
        if (
            sha256(sidecar_path) != frozen["sidecar_sha256"]
            or sha256(mapping_path) != frozen["mapping_sha256"]
            or mapping_entries(mapping_path) != graph_info[graph]["nodes"]
            or sidecar["selected_draw"] != 0
            or len(draws) != 3
            or len(frozen["draws"]) != 3
            or sidecar["mapping_draw_count"] != 3
            or set(draw["draw"] for draw in draws) != {0, 1, 2}
            or len({int(draw["draw"]) for draw in draws}) != 3
            or len({
                int(draw["draw"]) for draw in frozen["draws"]
            }) != 3
        ):
            raise ValueError(f"Invalid V4 mapping record: {key}")
        draw_times = []
        fingerprints = set()
        for draw in draws:
            draw_path = sidecar_path.parent / draw["path"]
            frozen_draw = next(
                item for item in frozen["draws"]
                if int(item["draw"]) == int(draw["draw"])
            )
            actual_fingerprint = mapping_permutation_fingerprint(
                draw_path
            )
            if (
                sha256(draw_path) != frozen_draw["sha256"]
                or mapping_entries(draw_path)
                != graph_info[graph]["nodes"]
                or actual_fingerprint != draw["mapping_fingerprint"]
                or not positive_finite(draw["reorder_time"])
            ):
                raise ValueError(f"Invalid V4 mapping draw: {key}")
            draw_times.append(float(draw["reorder_time"]))
            fingerprints.add(actual_fingerprint)
        validated_record = {
            "mapping_path": str(mapping_path),
            "mapping_bytes": mapping_path.stat().st_size,
            "mapping_fingerprint": sidecar["mapping_fingerprint"],
            "generation_policy_id": sidecar["generation_policy_id"],
            "selected_draw": sidecar["selected_draw"],
            "draws": [
                {
                    "draw": int(draw["draw"]),
                    "path": str(sidecar_path.parent / draw["path"]),
                    "bytes":
                        (sidecar_path.parent / draw["path"]).stat().st_size,
                    "mapping_fingerprint":
                        draw["mapping_fingerprint"],
                }
                for draw in draws
            ],
        }
        validated_identity = expected_mapping_identity(
            validated_record
        )
        frozen_identity = expected_mapping_identity(frozen)
        if validated_identity != frozen_identity:
            raise ValueError(f"V4 frozen mapping identity mismatch: {key}")
        mapping_times[key] = statistics.median(draw_times)
        mapping_identities[key] = validated_identity
        draw_zero = next(
            draw for draw in frozen["draws"]
            if int(draw["draw"]) == 0
        )
        if (
            sha256(mapping_path) != draw_zero["sha256"]
            or mapping_permutation_fingerprint(mapping_path)
            != sidecar["mapping_fingerprint"]
            or sidecar["mapping_fingerprint"]
            != frozen["mapping_fingerprint"]
        ):
            raise ValueError(
                f"V4 selected mapping does not match draw 0: {key}"
            )
        if algorithm in protocol["serial_equivalence_arms"]:
            serial_fingerprints.setdefault(graph, set()).update(
                fingerprints
            )
    if any(len(values) != 1 for values in serial_fingerprints.values()):
        raise ValueError("V4 serial equivalence controls diverged")

    gate_manifest_path = Path(execution["verification_manifest_path"])
    gate_results_path = Path(execution["verification_results_path"])
    if (
        sha256(gate_manifest_path)
        != execution["verification_manifest_sha256"]
        or sha256(gate_results_path)
        != execution["verification_results_sha256"]
    ):
        raise ValueError("V4 verification evidence changed")
    gate_manifest = json.loads(gate_manifest_path.read_text())
    gate_rows = json.loads(gate_results_path.read_text())
    if (
        gate_manifest["completed_cells"]
        != protocol["expected_verification_cells"]
        or gate_manifest["adjudication"]["total_passes"]
        != protocol["expected_verification_cells"]
        or len(gate_rows) != protocol["expected_verification_cells"]
    ):
        raise ValueError("V4 verification gate is incomplete")
    gate_id = gate_manifest["gate_id"]
    gate_index = {}
    for row in gate_rows:
        key = (
            str(row["graph"]),
            str(row["benchmark"]),
            str(row["algo_key"]),
        )
        if key in gate_index:
            raise ValueError(f"Duplicate V4 verification cell: {key}")
        if (
            row.get("gate_id") != gate_id
            or row.get("verification_state") != "pass"
        ):
            raise ValueError("Invalid V4 verification row")
        algorithm = str(row["algo_key"])
        if algorithm == "0":
            if row.get("mapping") != {
                "source": "direct",
                "algo_flags": ["-o", "0"],
            }:
                raise ValueError("V4 SHUFFLED verification mismatch")
        elif (
            row.get("mapping")
            != mapping_identities[(str(row["graph"]), algorithm)]
        ):
            raise ValueError("V4 verification mapping mismatch")
        gate_index[key] = row
    expected_gate_keys = {
        (graph, kernel, algorithm)
        for graph in protocol["graphs"]
        for kernel in protocol["kernels"]
        for algorithm in ("0", *protocol["arms"])
    }
    if set(gate_index) != expected_gate_keys:
        raise ValueError("V4 verification matrix is incomplete")

    os.environ["GRAPHBREW_SSSP_POLICY_PATH"] = str(
        root / "sssp_policy.json"
    )
    os.environ["GRAPHBREW_SSSP_TUNING_SNAPSHOT_PATH"] = str(
        root / "sssp_delta_tuning.json"
    )
    from scripts.experiments.vldb import runner

    runner.configure_artifact_root(root)
    runner.configure_runtime_policy(
        int(protocol["threads"]), str(protocol["cpu_list"])
    )
    runner.configure_cache_policy(
        preview=False,
        mode="ultrafast",
        sample_rate=64,
        all_algorithms=False,
        sizes_kib=None,
    )
    runner.configure_execution_mode(dry_run=False)

    replicate_rows = []
    process_cells = []
    replicate_records = execution.get("replicates", [])
    if (
        len(replicate_records) != 3
        or {int(record["replicate"]) for record in replicate_records}
        != {0, 1, 2}
        or len({
            str(record["timing_path"])
            for record in replicate_records
        }) != 3
        or len({
            str(record["timing_sha256"])
            for record in replicate_records
        }) != 3
        or len({
            str(record["cohort_id"])
            for record in replicate_records
        }) != 3
    ):
        raise ValueError("V4 requires three distinct process replicates")
    expected_cells = {
        (graph, kernel, algorithm)
        for graph in protocol["graphs"]
        for kernel in protocol["kernels"]
        for algorithm in protocol["arms"]
    }
    for replicate in replicate_records:
        timing_path = Path(replicate["timing_path"])
        if sha256(timing_path) != replicate["timing_sha256"]:
            raise ValueError("V4 timing replicate changed")
        rows = json.loads(timing_path.read_text())
        index = {}
        for row in rows:
            key = (
                str(row["graph"]),
                str(row["benchmark"]),
                str(row["algo_id"]),
            )
            if key in index:
                raise ValueError(f"Duplicate V4 timing cell: {key}")
            trials = row["trial_times"]
            if (
                len(trials) != protocol["trials_per_process"]
                or any(not positive_finite(value) for value in trials)
                or row["verification_gate_id"] != gate_id
                or row["verification_gate_status"] != "pass"
                or row["cohort_id"] != replicate["cohort_id"]
                or row["e2e_join_context_id"]
                != replicate["e2e_join_context_id"]
                or row["mapping_identity"]
                != mapping_identities[(key[0], key[2])]
                or row["mapping_identity_id"]
                != short_id(mapping_identities[(key[0], key[2])])
            ):
                raise ValueError(f"Invalid V4 timing cell: {key}")
            graph_path = Path(graph_info[key[0]]["path"])
            _cohort, expected_policy = runner._kernel_policy_ids(
                graph_path=str(graph_path),
                kind="kernel",
                trials=protocol["trials_per_process"],
                executable=runner.BIN_DIR / key[1],
                extra={"mapping": mapping_identities[(key[0], key[2])]},
            )
            if row["policy_id"] != expected_policy:
                raise ValueError(f"V4 policy mismatch: {key}")
            process_median = statistics.median(
                float(value) for value in trials
            )
            index[key] = process_median
            process_cells.append({
                "replicate": int(replicate["replicate"]),
                "graph": key[0],
                "kernel": key[1],
                "algorithm": key[2],
                "process_median_seconds": process_median,
            })
        if set(index) != expected_cells:
            raise ValueError("V4 timing replicate is incomplete")
        replicate_rows.append(index)

    cell_seconds = {
        key: statistics.median(
            replicate[key] for replicate in replicate_rows
        )
        for key in expected_cells
    }
    candidate = protocol["candidate"]
    anchors = protocol["anchors"]
    per_graph = {}
    for graph in protocol["graphs"]:
        candidate_mapping = mapping_times[(graph, candidate)]
        anchor_mapping = {
            anchor: mapping_times[(graph, anchor)]
            for anchor in anchors
        }
        record = {
            "candidate_over_min_rabbit_mapping":
                candidate_mapping / min(anchor_mapping.values()),
            "best_rabbit_over_candidate_kernel_gm":
                geometric_mean([
                    min(
                        cell_seconds[(graph, kernel, anchor)]
                        for anchor in anchors
                    )
                    / cell_seconds[(graph, kernel, candidate)]
                    for kernel in protocol["kernels"]
                ]),
            "reuse": {},
        }
        for reuse in protocol["reuse"]:
            ratios = []
            for kernel in protocol["kernels"]:
                candidate_total = (
                    candidate_mapping
                    + reuse
                    * cell_seconds[(graph, kernel, candidate)]
                )
                rabbit_total = min(
                    anchor_mapping[anchor]
                    + reuse
                    * cell_seconds[(graph, kernel, anchor)]
                    for anchor in anchors
                )
                ratios.append(rabbit_total / candidate_total)
            record["reuse"][str(reuse)] = geometric_mean(ratios)
        per_graph[graph] = record

    reuse_summary = {}
    rng = random.Random(BOOTSTRAP_SEED)
    graph_draws = [
        [
            rng.choice(protocol["graphs"])
            for _ in protocol["graphs"]
        ]
        for _ in range(BOOTSTRAP_SAMPLES)
    ]
    for reuse in protocol["reuse"]:
        values = [
            per_graph[graph]["reuse"][str(reuse)]
            for graph in protocol["graphs"]
        ]
        bootstrap = [
            geometric_mean([
                per_graph[graph]["reuse"][str(reuse)]
                for graph in draw
            ])
            for draw in graph_draws
        ]
        reuse_summary[str(reuse)] = {
            "point_gm": geometric_mean(values),
            "graph_block_bootstrap_95": [
                percentile(bootstrap, 0.025),
                percentile(bootstrap, 0.975),
            ],
            "worst_graph_ratio": min(values),
            "all_graphs_win": all(value >= 1 for value in values),
        }

    mapping_values = [
        per_graph[graph]["candidate_over_min_rabbit_mapping"]
        for graph in protocol["graphs"]
    ]
    gates = protocol["development_gates"]
    gate_results = {
        "mapping_gm": {
            "value": geometric_mean(mapping_values),
            "maximum":
                gates["mapping_candidate_over_min_rabbit_gm_max"],
            "pass": geometric_mean(mapping_values)
            <= gates["mapping_candidate_over_min_rabbit_gm_max"],
        },
        "mapping_worst": {
            "value": max(mapping_values),
            "maximum":
                gates["mapping_candidate_over_min_rabbit_per_graph_max"],
            "pass": max(mapping_values)
            <= gates[
                "mapping_candidate_over_min_rabbit_per_graph_max"
            ],
        },
    }
    for reuse in protocol["reuse"]:
        summary = reuse_summary[str(reuse)]
        gate_results[f"reuse_{reuse}"] = {
            **summary,
            "minimum_lower_95":
                gates["reuse_graph_block_lower_95_min"],
            "minimum_per_graph":
                gates["reuse_min_rabbit_over_candidate_per_graph_min"],
            "pass": (
                summary["graph_block_bootstrap_95"][0]
                > gates["reuse_graph_block_lower_95_min"]
                and summary["worst_graph_ratio"]
                >= gates[
                    "reuse_min_rabbit_over_candidate_per_graph_min"
                ]
            ),
        }
    passed = all(record["pass"] for record in gate_results.values())

    spreads = []
    for key in expected_cells:
        values = [replicate[key] for replicate in replicate_rows]
        spread = max(values) / min(values)
        if spread > 1.3:
            spreads.append({
                "graph": key[0],
                "kernel": key[1],
                "algorithm": key[2],
                "process_medians": values,
                "maximum_over_minimum": spread,
            })

    analysis = {
        "schema": "graphbrew-dual-arm-v4-analysis/v1",
        "created_at_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "inputs": {
            "protocol": {
                "path": str(protocol_path),
                "sha256": sha256(protocol_path),
            },
            "execution_manifest": {
                "path": str(execution_path),
                "sha256": sha256(execution_path),
            },
            "analysis_program": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "execution_repository_commit":
                protocol["repository_commit"],
            "analysis_repository_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                text=True,
            ).strip(),
        },
        "policy": {
            "cell_estimator":
                "median of three independent process medians; "
                "seven trials per process",
            "mapping_estimator":
                "median complete reorder time over three draws",
            "rabbit_policy":
                "minimum of CSR and Boost end-to-end per kernel",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "serial_mapping_equivalence": True,
        "per_graph": per_graph,
        "reuse": reuse_summary,
        "gate_results": gate_results,
        "process_spreads_over_1_3": spreads,
        "process_cells": process_cells,
        "pass": passed,
        "claim_boundary": protocol["claim_boundary"],
    }
    decision = {
        "schema": "graphbrew-dual-arm-v4-decision/v1",
        "created_at_utc": analysis["created_at_utc"],
        "status": (
            "balanced-development-gate-passed"
            if passed else "balanced-development-gate-failed"
        ),
        "candidate": candidate,
        "gate_results": gate_results,
        "claim_boundary":
            "Development result on burned graphs; terminal validation "
            "requires a sealed new corpus and frozen policy.",
    }
    return analysis, decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    analysis, decision = analyze(root)
    for name, payload in (
        ("analysis.json", analysis),
        ("decision.json", decision),
    ):
        path = root / name
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, path)
        print(name, sha256(path))
    print("status", decision["status"])


if __name__ == "__main__":
    main()
