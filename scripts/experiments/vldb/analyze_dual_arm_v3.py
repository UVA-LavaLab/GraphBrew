#!/usr/bin/env python3
"""Analyze the Compact-and-Emit V3 mapping-only campaign."""

from __future__ import annotations

import argparse
import binascii
import datetime
import json
import math
import os
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
)
from scripts.lib.pipeline.benchmark import (  # noqa: E402
    mapping_permutation_fingerprint,
)


PHASE_KEYS = (
    "compose_grouping_time",
    "compose_community_order_time",
    "compose_vertex_map_time",
    "compose_intra_order_time",
    "compose_final_assign_time",
)
EXPECTED_PROTOCOL_SHA256 = (
    "990ee7acdcc265baa6e4abeebdf8d63ca"
    "52557d620d66e32e0ab761cb3bd9aa0"
)
EXPECTED_EXECUTION_MANIFEST_SHA256 = (
    "423c1a9bfcafc2753d3dbf54c97ebea4"
    "bb5fddc63453a786c89db93366858798"
)


def median(values: list[float]) -> float:
    if not values or any(not positive_finite(value) for value in values):
        raise ValueError("Expected finite positive measurements")
    return statistics.median(values)


def range_record(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "maximum": ordered[-1],
    }


def crc32(path: Path) -> str:
    value = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value = binascii.crc32(chunk, value)
    return f"{value & 0xffffffff:08x}"


def mapping_entries(path: Path) -> int:
    count = 0
    last = b""
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            count += chunk.count(b"\n")
            last = chunk[-1:]
    return count + (1 if last and last != b"\n" else 0)


def finite_nonnegative(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0
    )


def analyze(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = root / "protocol.json"
    execution_manifest_path = root / "execution_manifest.json"
    if sha256(protocol_path) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("V3 protocol hash changed after preregistration")
    if (
        sha256(execution_manifest_path)
        != EXPECTED_EXECUTION_MANIFEST_SHA256
    ):
        raise ValueError("V3 execution manifest hash changed")
    protocol = json.loads(protocol_path.read_text())
    execution_manifest = json.loads(
        execution_manifest_path.read_text()
    )
    if protocol.get("schema") != "graphbrew-dual-arm-v3-mapping/v1":
        raise ValueError("Unsupported V3 protocol")
    if (
        protocol.get("repository_clean") is not True
        or len(protocol.get("graphs", [])) != 3
        or len(protocol.get("serial_equivalence_arms", [])) != 4
        or len(protocol.get("parallel_arms", [])) != 4
        or protocol.get("mapping_draws_per_arm") != 3
        or protocol.get("expected_mapping_records") != 27
        or protocol.get("expected_mapping_executions") != 81
        or protocol.get("promotion_gates", {}).get("promotion_limit") != 1
    ):
        raise ValueError("V3 protocol violates fixed campaign constraints")
    if (
        execution_manifest.get("schema")
        != "graphbrew-dual-arm-v3-execution/v1"
        or execution_manifest.get("protocol_sha256")
        != EXPECTED_PROTOCOL_SHA256
        or execution_manifest.get("repository_commit")
        != protocol["repository_commit"]
        or execution_manifest.get("repository_clean_at_freeze")
        is not True
        or execution_manifest.get("expected_records")
        != protocol["expected_mapping_records"]
        or execution_manifest.get("expected_executions")
        != protocol["expected_mapping_executions"]
    ):
        raise ValueError("Invalid V3 execution manifest")

    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    if dirty:
        raise ValueError("Repository is dirty during V3 analysis")
    subprocess.run(
        [
            "git",
            "cat-file",
            "-e",
            f"{protocol['repository_commit']}^{{commit}}",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    converter_path = Path(protocol["binaries"]["converter"]["path"])
    if (
        sha256(converter_path) != protocol["binaries"]["converter"]["sha256"]
        or execution_manifest.get("converter_sha256")
        != protocol["binaries"]["converter"]["sha256"]
    ):
        raise ValueError("V3 converter binary changed after freeze")

    graph_provenance = {}
    manifest_graphs = {
        str(record["graph"]): record
        for record in execution_manifest.get("graphs", [])
    }
    if set(manifest_graphs) != set(protocol["graphs"]):
        raise ValueError("V3 execution graph set changed")
    for graph in protocol["graphs"]:
        path = (
            Path("/media/Data/00_GraphDatasets/GraphBrew")
            / graph
            / f"{graph}.sg.meta.json"
        )
        payload = json.loads(path.read_text())
        graph_path = Path(manifest_graphs[graph]["path"])
        with graph_path.open("rb") as stream:
            header = stream.read(17)
        if len(header) != 17:
            raise ValueError(f"Truncated V3 graph header: {graph}")
        directed, directed_edges, nodes = struct.unpack(
            "<?qq", header
        )
        if (
            payload.get("schema") != "graph_source/v2"
            or payload.get("reorder_semantics_version")
            != "graphbrew-reorder/v4"
            or payload.get("converter_sha256")
            != protocol["binaries"]["converter"]["sha256"]
            or graph_path.stat().st_size
            != manifest_graphs[graph]["bytes"]
            or sha256(graph_path)
            != manifest_graphs[graph]["sha256"]
            or crc32(graph_path)
            != manifest_graphs[graph]["crc32"]
            or graph_path.stat().st_size != payload["output_bytes"]
            or crc32(graph_path) != payload["output_crc32"]
            or sha256(path) != manifest_graphs[graph]["meta_sha256"]
            or payload["nodes"] != manifest_graphs[graph]["nodes"]
            or payload["directed_edges"]
            != manifest_graphs[graph]["directed_edges"]
            or directed is not False
            or nodes != payload["nodes"]
            or directed_edges != payload["directed_edges"]
        ):
            raise ValueError(f"Stale V3 graph provenance: {graph}")
        graph_provenance[graph] = {
            "path": str(path),
            "sha256": sha256(path),
            "crc32": payload["output_crc32"],
            "nodes": payload["nodes"],
            "directed_edges": payload["directed_edges"],
        }

    mapping_root = root / "vldb_mappings"
    manifest_records = {
        (str(record["graph"]), str(record["algorithm"])): record
        for record in execution_manifest.get("mapping_records", [])
    }
    expected_mapping_keys = {
        (graph, algorithm)
        for graph in protocol["graphs"]
        for algorithm in protocol["arms"]
    }
    if set(manifest_records) != expected_mapping_keys:
        raise ValueError("V3 execution mapping set changed")
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for graph in protocol["graphs"]:
        for algorithm in protocol["arms"]:
            safe = algorithm.replace(":", "_").replace("/", "_")
            sidecar_path = mapping_root / graph / f"{safe}.json"
            mapping_path = mapping_root / graph / f"{safe}.lo"
            sidecar = json.loads(sidecar_path.read_text())
            draws = sidecar.get("mapping_draws", [])
            manifest_record = manifest_records[(graph, algorithm)]
            if (
                sidecar.get("schema") != "reorder_meta/v6"
                or sidecar.get("reorder_semantics_version")
                != "graphbrew-reorder/v4"
                or sidecar.get("graph") != graph
                or sidecar.get("algo_key") != algorithm
                or sidecar.get("graph_crc32")
                != graph_provenance[graph]["crc32"]
                or len(draws) != protocol["mapping_draws_per_arm"]
                or sidecar.get("mapping_draw_count")
                != protocol["mapping_draws_per_arm"]
                or sidecar.get("selected_draw") != 0
                or not mapping_path.is_file()
                or sha256(sidecar_path)
                != manifest_record["sidecar_sha256"]
                or sha256(mapping_path)
                != manifest_record["mapping_sha256"]
                or mapping_path.stat().st_size
                != manifest_record["mapping_bytes"]
                or sidecar.get("generation_policy_id")
                != manifest_record["generation_policy_id"]
            ):
                raise ValueError(f"Invalid V3 mapping sidecar: {graph}/{algorithm}")

            draw_ids = [draw.get("draw") for draw in draws]
            draw_paths = [str(draw.get("path")) for draw in draws]
            if (
                set(draw_ids) != {0, 1, 2}
                or len(set(draw_paths)) != len(draw_paths)
            ):
                raise ValueError(f"Invalid V3 draw cohort: {graph}/{algorithm}")
            manifest_draws = {
                int(draw["draw"]): draw
                for draw in manifest_record["draws"]
            }
            draw_records = []
            graphbrew_arm = algorithm != protocol["anchor"]
            for draw in draws:
                draw_path = sidecar_path.parent / draw["path"]
                manifest_draw = manifest_draws[int(draw["draw"])]
                actual_fingerprint = mapping_permutation_fingerprint(
                    draw_path
                )
                if (
                    actual_fingerprint != draw.get("mapping_fingerprint")
                    or sha256(draw_path) != manifest_draw["sha256"]
                    or draw_path.stat().st_size != manifest_draw["bytes"]
                    or mapping_entries(draw_path)
                    != graph_provenance[graph]["nodes"]
                ):
                    raise ValueError(
                        f"V3 mapping fingerprint mismatch: "
                        f"{graph}/{algorithm}/draw{draw.get('draw')}"
                    )
                core_values = draw.get("reorder_core_time_passes", [])
                complete_values = draw.get("reorder_time_passes", [])
                if (
                    len(core_values) != 1
                    or not positive_finite(core_values[0])
                    or not positive_finite(draw.get("reorder_core_time"))
                    or abs(
                        float(draw["reorder_core_time"])
                        - float(core_values[0])
                    ) > 1e-12
                    or len(complete_values) != 1
                    or not positive_finite(complete_values[0])
                    or not positive_finite(draw.get("reorder_time"))
                    or abs(
                        float(draw["reorder_time"])
                        - float(complete_values[0])
                    ) > 1e-12
                ):
                    raise ValueError("V3 draw must contain one reorder pass")
                if graphbrew_arm:
                    for phase in PHASE_KEYS:
                        phase_values = draw.get(f"{phase}_passes", [])
                        if (
                            len(phase_values) != 1
                            or not finite_nonnegative(phase_values[0])
                            or not finite_nonnegative(draw.get(phase))
                            or abs(
                                float(draw[phase])
                                - float(phase_values[0])
                            ) > 1e-12
                        ):
                            raise ValueError(
                                f"Missing V3 phase telemetry: "
                                f"{graph}/{algorithm}/{phase}"
                            )
                compact = "compact" in algorithm
                if compact and (
                    len(draw.get(
                        "membership_compaction_time_passes", [],
                    )) != 1
                    or not isinstance(
                        draw.get("membership_id_slots"), (int, float)
                    )
                    or not isinstance(
                        draw.get("membership_active_communities"),
                        (int, float),
                    )
                    or not isinstance(
                        draw.get("membership_empty_fraction"),
                        (int, float),
                    )
                ):
                    raise ValueError(
                        f"Missing V3 compaction telemetry: "
                        f"{graph}/{algorithm}"
                    )
                if compact:
                    compaction_values = draw.get(
                        "membership_compaction_time_passes", [],
                    )
                    slots = float(draw["membership_id_slots"])
                    active = float(
                        draw["membership_active_communities"]
                    )
                    empty = float(draw["membership_empty_fraction"])
                    if (
                        len(compaction_values) != 1
                        or not finite_nonnegative(compaction_values[0])
                        or not finite_nonnegative(
                            draw["membership_compaction_time"]
                        )
                        or abs(
                            float(draw["membership_compaction_time"])
                            - float(compaction_values[0])
                        ) > 1e-12
                        or not slots >= active > 0
                        or not 0 <= empty <= 1
                        or abs(empty - (1.0 - active / slots)) > 1e-6
                        or abs(
                            float(draw["compose_community_slots"])
                            - active
                        ) > 0.5
                    ):
                        raise ValueError(
                            f"Invalid V3 compaction telemetry: "
                            f"{graph}/{algorithm}"
                        )
                draw_records.append({
                    "draw": int(draw["draw"]),
                    "path": str(draw_path),
                    "bytes": draw_path.stat().st_size,
                    "mapping_fingerprint": actual_fingerprint,
                    "membership_fingerprint": (
                        draw.get("graphbrew_realized_configs", [{}])[0]
                        .get("membership_fingerprint")
                        if graphbrew_arm else None
                    ),
                    "reorder_time": float(complete_values[0]),
                    "reorder_core_time": float(core_values[0]),
                    **(
                        {
                            phase: float(
                                draw[f"{phase}_passes"][0]
                            )
                            for phase in PHASE_KEYS
                        }
                        if graphbrew_arm
                        else {phase: None for phase in PHASE_KEYS}
                    ),
                    "compose_community_slots": (
                        float(draw["compose_community_slots"])
                        if graphbrew_arm else None
                    ),
                    "membership_compaction_time": (
                        float(draw["membership_compaction_time"])
                        if compact else 0.0
                    ),
                    "membership_id_slots": (
                        float(draw["membership_id_slots"])
                        if compact else None
                    ),
                    "membership_active_communities": (
                        float(draw["membership_active_communities"])
                        if compact else None
                    ),
                    "membership_empty_fraction": (
                        float(draw["membership_empty_fraction"])
                        if compact else None
                    ),
                })
            if (
                mapping_permutation_fingerprint(mapping_path)
                != sidecar["mapping_fingerprint"]
                or mapping_entries(mapping_path)
                != graph_provenance[graph]["nodes"]
                or sha256(mapping_path)
                != manifest_draws[0]["sha256"]
            ):
                raise ValueError(
                    f"Selected V3 mapping mismatch: {graph}/{algorithm}"
                )
            records[(graph, algorithm)] = {
                "sidecar_path": str(sidecar_path),
                "sidecar_sha256": sha256(sidecar_path),
                "mapping_path": str(mapping_path),
                "mapping_fingerprint": sidecar["mapping_fingerprint"],
                "draws": draw_records,
            }

    if len(records) != protocol["expected_mapping_records"]:
        raise ValueError("V3 mapping matrix is incomplete")

    serial_equivalence = {}
    for graph in protocol["graphs"]:
        fingerprints = {
            algorithm: {
                draw["mapping_fingerprint"]
                for draw in records[(graph, algorithm)]["draws"]
            }
            for algorithm in protocol["serial_equivalence_arms"]
        }
        if (
            any(len(values) != 1 for values in fingerprints.values())
            or len({
                next(iter(values))
                for values in fingerprints.values()
            }) != 1
        ):
            raise ValueError(f"V3 serial controls diverged: {graph}")
        mapping_fingerprints = {
            next(iter(values))
            for values in fingerprints.values()
        }
        serial_equivalence[graph] = {
            "mapping_fingerprint":
                next(iter(mapping_fingerprints)),
            "membership_fingerprints": {
                algorithm: sorted({
                    draw["membership_fingerprint"]
                    for draw in records[(graph, algorithm)]["draws"]
                })
                for algorithm in protocol["serial_equivalence_arms"]
            },
            "membership_fingerprints_cross_arm_comparable": False,
        }

    parallel_summary = {}
    for algorithm in protocol["parallel_arms"]:
        per_graph = {}
        for graph in protocol["graphs"]:
            draws = records[(graph, algorithm)]["draws"]
            per_graph[graph] = {
                "complete_seconds": range_record([
                    draw["reorder_time"] for draw in draws
                ]),
                "core_seconds": range_record([
                    draw["reorder_core_time"] for draw in draws
                ]),
                "phases": {
                    phase: range_record([
                        draw[phase] for draw in draws
                    ])
                    for phase in PHASE_KEYS
                },
                "compose_community_slots": range_record([
                    draw["compose_community_slots"] for draw in draws
                ]),
            }
            if "compact" in algorithm:
                per_graph[graph]["compaction"] = {
                    "seconds": range_record([
                        draw["membership_compaction_time"]
                        for draw in draws
                    ]),
                    "id_slots": range_record([
                        draw["membership_id_slots"] for draw in draws
                    ]),
                    "active_communities": range_record([
                        draw["membership_active_communities"]
                        for draw in draws
                    ]),
                    "empty_fraction": range_record([
                        draw["membership_empty_fraction"]
                        for draw in draws
                    ]),
                }
        parallel_summary[algorithm] = per_graph

    baseline = protocol["baseline"]
    direct = protocol["direct"]
    compact = protocol["compact"]
    compact_direct = protocol["compact_direct"]
    anchor = protocol["anchor"]
    graphs = protocol["graphs"]
    wiki = "wiki-Talk"

    def complete(graph: str, algorithm: str) -> float:
        if algorithm == anchor:
            return median([
                draw["reorder_time"]
                for draw in records[(graph, algorithm)]["draws"]
            ])
        return parallel_summary[algorithm][graph][
            "complete_seconds"
        ]["median"]

    selection_reduction = 1.0 - geometric_mean([
        complete(graph, compact_direct)
        / complete(graph, baseline)
        for graph in graphs
    ])
    baseline_combined = median([
        draw["compose_intra_order_time"]
        + draw["compose_final_assign_time"]
        for draw in records[(wiki, baseline)]["draws"]
    ])
    direct_combined = median([
        draw["compose_intra_order_time"]
        + draw["compose_final_assign_time"]
        for draw in records[(wiki, direct)]["draws"]
    ])
    compact_intra = parallel_summary[compact][wiki][
        "phases"
    ]["compose_intra_order_time"]["median"]
    baseline_intra = parallel_summary[baseline][wiki][
        "phases"
    ]["compose_intra_order_time"]["median"]

    gates = protocol["promotion_gates"]
    gate_results = {
        "serial_mapping_fingerprints_equal": True,
        "wiki_talk_compact_direct_complete_seconds": {
            "value": complete(wiki, compact_direct),
            "maximum":
                gates["wiki_talk_compact_direct_complete_seconds_max"],
            "pass": (
                complete(wiki, compact_direct)
                <= gates[
                    "wiki_talk_compact_direct_complete_seconds_max"
                ]
            ),
        },
        "selection_graph_reduction_gm": {
            "value": selection_reduction,
            "minimum": gates["selection_graph_reduction_gm_min"],
            "pass": selection_reduction
            >= gates["selection_graph_reduction_gm_min"],
        },
        "wiki_talk_compact_direct_over_csr": {
            "value": (
                complete(wiki, compact_direct)
                / complete(wiki, anchor)
            ),
            "maximum": gates["wiki_talk_compact_direct_over_csr_max"],
            "pass": (
                complete(wiki, compact_direct)
                / complete(wiki, anchor)
                <= gates["wiki_talk_compact_direct_over_csr_max"]
            ),
        },
        "direct_combined_intra_final_over_baseline": {
            "value": direct_combined / baseline_combined,
            "maximum":
                gates["direct_combined_intra_final_over_baseline_max"],
            "pass": (
                direct_combined / baseline_combined
                <= gates[
                    "direct_combined_intra_final_over_baseline_max"
                ]
            ),
        },
        "compact_intra_over_baseline": {
            "value": compact_intra / baseline_intra,
            "maximum": gates["compact_intra_over_baseline_max"],
            "pass": (
                compact_intra / baseline_intra
                <= gates["compact_intra_over_baseline_max"]
            ),
        },
        "telemetry_complete": True,
    }
    passed = all(
        result is True
        or (
            isinstance(result, dict)
            and result.get("pass") is True
        )
        for result in gate_results.values()
    )

    analysis = {
        "schema": "graphbrew-dual-arm-v3-analysis/v1",
        "created_at_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "inputs": {
            "protocol": {
                "path": str(protocol_path),
                "sha256": sha256(protocol_path),
            },
            "analysis_program": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "execution_manifest": {
                "path": str(execution_manifest_path),
                "sha256": sha256(execution_manifest_path),
            },
            "execution_repository_commit":
                protocol["repository_commit"],
            "analysis_repository_commit": current_commit,
        },
        "graph_provenance": graph_provenance,
        "serial_equivalence": serial_equivalence,
        "parallel_summary": parallel_summary,
        "gate_results": gate_results,
        "pass": passed,
        "claim_boundary": protocol["claim_boundary"],
    }
    decision = {
        "schema": "graphbrew-dual-arm-v3-decision/v1",
        "created_at_utc": analysis["created_at_utc"],
        "status": (
            "promote-compact-and-emit"
            if passed else "compact-and-emit-gate-failed"
        ),
        "promoted": [compact_direct] if passed else [],
        "wiki_talk_complete_seconds":
            complete(wiki, compact_direct),
        "wiki_talk_rabbit_csr_seconds": complete(wiki, anchor),
        "selection_graph_reduction_gm": selection_reduction,
        "serial_mapping_equivalence": True,
        "claim_boundary":
            "Mapping-only engineering result with serial functional "
            "equivalence. Parallel-arm kernel behavior is not claimed.",
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
    print("promoted", decision["promoted"])


if __name__ == "__main__":
    main()
