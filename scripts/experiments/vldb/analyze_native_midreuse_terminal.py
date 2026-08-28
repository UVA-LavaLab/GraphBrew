#!/usr/bin/env python3
"""Validate and analyze the frozen native mid-reuse terminal campaign."""

from __future__ import annotations

import argparse
import datetime
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve()
while PROJECT_ROOT != PROJECT_ROOT.parent:
    if (PROJECT_ROOT / "scripts" / "experiments" / "vldb").is_dir():
        break
    PROJECT_ROOT = PROJECT_ROOT.parent
if not (PROJECT_ROOT / "scripts").is_dir():
    PROJECT_ROOT = Path(
        "/home/cmv6ru/Documents/00_github_repos/00_GraphBrew"
    )
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.vldb.analyze_dual_arm_s0 import (  # noqa: E402
    geometric_mean,
    positive_finite,
    sha256,
    short_id,
)


EXPECTED_PROTOCOL_SHA256 = (
    "de6edf9e9ab4d1d2ee0cac7d0bdb65af"
    "2b4778fef39d38578ebcde2d6cccbdb7"
)
EXPECTED_CORPUS_SHA256 = (
    "05b69d98801a998be5c1fcf0cfc1b015"
    "fcbf7e38eb3996602d16feb548b6b204"
)
EXPECTED_MAPPING_SHA256 = (
    "8ddb2250ed0f049498879df294698b249"
    "9dc0d0af90efcc2a32cc2ede3f8136d"
)
EXPECTED_VERIFICATION_SHA256 = (
    "ceabd27a0dbc7261552573cf38a89e2b"
    "2246266a1ed4b8b71c12f31b5ad9e49f"
)
EXPECTED_TIMING_AMENDMENT_SHA256 = (
    "bdf6fd346fe6a31a20d94aa693ac1ca0"
    "61aae53cf79a4654d1aeacfedfa5c1a5"
)
EXPECTED_CONTROL_AMENDMENT_SHA256 = (
    "c1c34b7cf3df1716ae7fc5a62742d543"
    "86ba2936ab6720c00870a91c0de2b1b8"
)
EXPECTED_RESOURCE_AMENDMENT_SHA256 = (
    "5f1f9e9b1d58d837c476e8dadf97ee74"
    "395b56981ebe3b337fd8f48c421666f3"
)
EXPECTED_ESCALATION_PLAN_SHA256 = (
    "c9fe5b2d3a9f341d3a384194c3bc5cb8"
    "36d562f62982e9f26351d4a20c44f1f4"
)
EXECUTION_COMMIT = "bb9023ada3ac70c9f86d415ef4db8efbbe22b6ad"
ORIGINAL = "0"
RABBIT_CSR = "8:csr"
RABBIT_BOOST = "8:boost"
BOOTSTRAP_SAMPLES = 50_000
BOOTSTRAP_SEED = 15_032_391


def _atomic_write(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _percentile(values: list[float], probability: float) -> float:
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


def _trial_cv(values: list[float]) -> float:
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean > 0 else math.inf


def _verification_resource_state_valid(
    state: dict[str, Any],
    protocol: dict[str, Any],
) -> bool:
    resources = protocol["resource_gates"]
    temperatures = [
        float(record["celsius"])
        for record in state.get("temperatures", [])
    ]
    return (
        bool(temperatures)
        and len(state.get("loadavg", [])) >= 3
        and float(state["loadavg"][0])
        <= resources["one_minute_load_start_end_max"]
        and int(state["memory_bytes"]["MemAvailable"])
        >= resources["mem_available_start_end_min_gib"] * 1024**3
        and max(temperatures)
        <= resources["cpu_package_temperature_max_c"]
    )


def _timing_resource_state_valid(
    state: dict[str, Any],
    resource_amendment: dict[str, Any],
) -> bool:
    policy = resource_amendment["timing_resource_gate"]
    temperatures = [
        float(record["celsius"])
        for record in state.get("temperatures", [])
    ]
    idle = state.get("cpu_idle_sample", {})
    return (
        bool(temperatures)
        and idle.get("cpus") == list(range(15))
        and idle.get("interval_seconds") == 5
        and float(idle["average_busy_fraction"])
        <= policy["average_busy_fraction_max"]
        and float(idle["maximum_busy_fraction"])
        <= policy["per_cpu_busy_fraction_max"]
        and int(state["memory_bytes"]["MemAvailable"])
        >= policy["mem_available_min_gib"] * 1024**3
        and max(temperatures) <= policy["maximum_temperature_c"]
    )


def _resource_receipt_valid(
    receipt: dict[str, Any],
    protocol: dict[str, Any],
    resource_amendment: dict[str, Any],
    *,
    kind: str,
    replicate: int,
    graphs: list[str] | None = None,
    kernel: str | None = None,
) -> bool:
    smt = receipt.get("smt_policy", {})
    control = receipt.get("control_policy", {})
    return (
        receipt.get("schema")
        == "graphbrew-terminal-process-resource/v1"
        and receipt.get("kind") == kind
        and int(receipt.get("replicate", -1)) == replicate
        and receipt.get("returncode") == 0
        and (
            graphs is None or receipt.get("graphs") == graphs
        )
        and (
            kernel is None or receipt.get("kernel") == kernel
        )
        and smt.get("benchmark_cpu_list") == "0-15"
        and smt.get("sibling_cpu_list") == "16-31"
        and smt.get("siblings_assigned_to_benchmark") is False
        and smt.get("affinity_enforced_by") == "taskset"
        and smt.get("omp_threads") == 16
        and control.get("all_threads_affinity") == [15]
        and control.get("scheduling_policy") == "SCHED_IDLE"
        and control.get("nice") == 19
        and control.get("shares_logical_cpu_with_benchmark") is True
        and control.get("smt_siblings_excluded") == "16-31"
        and _timing_resource_state_valid(
            receipt["start"], resource_amendment
        )
        and _timing_resource_state_valid(
            receipt["end"], resource_amendment
        )
    )


def _expected_mapping_identity(
    record: dict[str, Any],
    draw_index: int,
) -> dict[str, Any]:
    selected = next(
        draw for draw in record["draws"]
        if int(draw["draw"]) == draw_index
    )
    return {
        "source": "map",
        "path": Path(selected["path"]).name,
        "bytes": int(selected["bytes"]),
        "mapping_fingerprint": selected["mapping_fingerprint"],
        "generation_policy_id": record["generation_policy_id"],
        "selected_draw": draw_index,
        "mapping_draws": [
            {
                "path": Path(draw["path"]).name,
                "bytes": int(draw["bytes"]),
                "mapping_fingerprint": draw["mapping_fingerprint"],
            }
            for draw in record["draws"]
        ],
    }


def _load_frozen_inputs(
    root: Path,
    *,
    require_escalation_plan: bool,
) -> dict[str, Any]:
    paths = {
        "protocol": root / "protocol.json",
        "corpus": root / "corpus_manifest.json",
        "mappings": root / "mapping_manifest.json",
        "verification": root / "verification_manifest.json",
        "timing_amendment": root / "timing_execution_amendment.json",
        "control_amendment": root / "control_isolation_amendment.json",
        "resource_amendment": root / "resource_gate_amendment_v2.json",
    }
    expected = {
        "protocol": EXPECTED_PROTOCOL_SHA256,
        "corpus": EXPECTED_CORPUS_SHA256,
        "mappings": EXPECTED_MAPPING_SHA256,
        "verification": EXPECTED_VERIFICATION_SHA256,
        "timing_amendment": EXPECTED_TIMING_AMENDMENT_SHA256,
        "control_amendment": EXPECTED_CONTROL_AMENDMENT_SHA256,
        "resource_amendment": EXPECTED_RESOURCE_AMENDMENT_SHA256,
    }
    if require_escalation_plan:
        paths["escalation_plan"] = (
            root / "escalation_execution_plan.json"
        )
        expected["escalation_plan"] = EXPECTED_ESCALATION_PLAN_SHA256
    for name, path in paths.items():
        if sha256(path) != expected[name]:
            raise ValueError(f"Terminal frozen input changed: {path}")
    payload = {
        name: json.loads(path.read_text())
        for name, path in paths.items()
    }
    payload["paths"] = paths
    return payload


def _validate_frozen_inputs(
    root: Path,
    frozen: dict[str, Any],
) -> dict[str, Any]:
    protocol = frozen["protocol"]
    corpus = frozen["corpus"]
    mappings = frozen["mappings"]
    verification = frozen["verification"]
    amendment = frozen["timing_amendment"]
    control_amendment = frozen["control_amendment"]
    resource_amendment = frozen["resource_amendment"]
    escalation_plan = frozen.get("escalation_plan")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    if dirty:
        raise ValueError("Repository is dirty during terminal analysis")
    subprocess.run(
        ["git", "cat-file", "-e", f"{EXECUTION_COMMIT}^{{commit}}"],
        cwd=PROJECT_ROOT,
        check=True,
    )
    if (
        protocol["schema"] != "graphbrew-native-midreuse-terminal/v1"
        or corpus["schema"]
        != "graphbrew-native-midreuse-terminal-corpus/v1"
        or mappings["schema"]
        != "graphbrew-native-midreuse-terminal-mappings/v1"
        or verification["schema"]
        != "graphbrew-native-midreuse-terminal-verification/v1"
        or amendment["schema"]
        != "graphbrew-native-midreuse-terminal-timing-amendment/v1"
        or control_amendment["schema"]
        != "graphbrew-native-midreuse-terminal-control-isolation/v1"
        or resource_amendment["schema"]
        != "graphbrew-native-midreuse-terminal-resource-gate/v2"
        or protocol["repository_commit"] != EXECUTION_COMMIT
        or protocol["repository_clean"] is not True
        or corpus["protocol_sha256"] != EXPECTED_PROTOCOL_SHA256
        or mappings["protocol_sha256"] != EXPECTED_PROTOCOL_SHA256
        or verification["protocol_sha256"] != EXPECTED_PROTOCOL_SHA256
        or amendment["protocol_sha256"] != EXPECTED_PROTOCOL_SHA256
        or control_amendment["protocol_sha256"]
        != EXPECTED_PROTOCOL_SHA256
        or resource_amendment["protocol_sha256"]
        != EXPECTED_PROTOCOL_SHA256
        or corpus["graph_content_unchanged"] is not True
        or verification["all_passed"] is not True
        or verification["cells"] != 960
        or protocol["execution"]["process_replicates"] != 4
        or protocol["execution"]["trials_per_process"] != 7
        or protocol["execution"]["mapping_draws"] != 4
        or protocol["primary_reuse"] != 40
        or protocol["primary_kernels"]
        != ["pr", "pr_spmv", "bfs", "bc"]
    ):
        raise ValueError("Invalid terminal frozen contract")
    if escalation_plan is not None and (
        escalation_plan["schema"]
        != "graphbrew-native-midreuse-terminal-escalation-plan/v1"
        or escalation_plan["protocol_sha256"]
        != EXPECTED_PROTOCOL_SHA256
    ):
        raise ValueError("Invalid terminal escalation plan")

    graph_records = {
        record["graph"]: record for record in corpus["records"]
    }
    protocol_graphs = {
        record["graph"]: record for record in protocol["graphs"]
    }
    if (
        len(graph_records) != 15
        or set(graph_records) != set(protocol_graphs)
    ):
        raise ValueError("Terminal graph matrix changed")
    for graph, record in graph_records.items():
        graph_path = Path(record["graph_path"])
        meta_path = Path(record["meta_path"])
        if (
            graph_path.stat().st_size != record["graph_bytes"]
            or sha256(graph_path) != record["graph_sha256"]
            or sha256(meta_path) != record["meta_sha256"]
            or record["graph_sha256"]
            != protocol_graphs[graph]["graph_sha256"]
            or record["nodes"] != protocol_graphs[graph]["nodes"]
            or record["directed_edges"]
            != protocol_graphs[graph]["directed_edges"]
        ):
            raise ValueError(f"Terminal graph provenance changed: {graph}")

    mapping_records = {
        (record["graph"], record["algorithm"]): record
        for record in mappings["records"]
    }
    candidate = protocol["candidate"]
    mapping_arms = [candidate, RABBIT_CSR, RABBIT_BOOST]
    expected_mapping_keys = {
        (graph, algorithm)
        for graph in graph_records
        for algorithm in mapping_arms
    }
    if (
        len(mapping_records) != 45
        or set(mapping_records) != expected_mapping_keys
    ):
        raise ValueError("Terminal mapping matrix changed")
    for key, record in mapping_records.items():
        sidecar_path = Path(record["sidecar_path"])
        mapping_path = Path(record["mapping_path"])
        if (
            sha256(sidecar_path) != record["sidecar_sha256"]
            or sha256(mapping_path) != record["mapping_sha256"]
            or len(record["draws"]) != 4
            or {int(draw["draw"]) for draw in record["draws"]}
            != {0, 1, 2, 3}
        ):
            raise ValueError(f"Terminal mapping changed: {key}")
        for draw in record["draws"]:
            draw_path = Path(draw["path"])
            if (
                draw_path.stat().st_size != draw["bytes"]
                or sha256(draw_path) != draw["sha256"]
                or not positive_finite(draw["reorder_time"])
            ):
                raise ValueError(
                    f"Terminal mapping draw changed: {key}/"
                    f"{draw['draw']}"
                )
        if key[1] == candidate and len({
            draw["mapping_fingerprint"] for draw in record["draws"]
        }) != 1:
            raise ValueError(
                f"Native candidate mapping is not deterministic: {key[0]}"
            )

    gate_ids = {
        int(manifest["draws"][0]): manifest["gate_id"]
        for manifest in verification["manifests"]
    }
    if set(gate_ids) != {0, 1, 2, 3}:
        raise ValueError("Terminal verification draws changed")
    verification_results_path = Path(
        verification["verification_results_path"]
    )
    if (
        sha256(verification_results_path)
        != verification["verification_results_sha256"]
    ):
        raise ValueError("Terminal verification results changed")
    rows = json.loads(verification_results_path.read_text())
    selected_rows = [
        row for row in rows if row.get("gate_id") in set(gate_ids.values())
    ]
    if (
        len(selected_rows) != 960
        or any(
            row.get("verification_state") != "pass"
            for row in selected_rows
        )
    ):
        raise ValueError("Terminal verification matrix is incomplete")
    for replicate, resource_record in enumerate(
        verification["resource_receipts"]
    ):
        resource_path = Path(resource_record["path"])
        resource = json.loads(resource_path.read_text())
        command = [str(value) for value in resource.get("command", [])]
        if (
            sha256(resource_path) != resource_record["sha256"]
            or resource.get("schema")
            != "graphbrew-terminal-process-resource/v1"
            or resource.get("kind") != "verification"
            or resource.get("replicate") != replicate
            or resource.get("returncode") != 0
            or not _verification_resource_state_valid(
                resource["start"], protocol
            )
            or not _verification_resource_state_valid(
                resource["end"], protocol
            )
            or "--cpu-list" not in command
            or command[command.index("--cpu-list") + 1] != "0-15"
            or "--threads" not in command
            or command[command.index("--threads") + 1] != "16"
            or "--mapping-draw-index" not in command
            or command[command.index("--mapping-draw-index") + 1]
            != str(replicate)
        ):
            raise ValueError(
                f"Terminal verification resource gate failed: "
                f"replicate {replicate}"
            )

    return {
        "graphs": list(graph_records),
        "graph_records": graph_records,
        "protocol_graphs": protocol_graphs,
        "mapping_records": mapping_records,
        "mapping_arms": mapping_arms,
        "candidate": candidate,
        "gate_ids": gate_ids,
    }


def _load_base_timing(
    root: Path,
    frozen: dict[str, Any],
    validated: dict[str, Any],
) -> dict[str, Any]:
    protocol = frozen["protocol"]
    graphs = validated["graphs"]
    kernels = protocol["primary_kernels"]
    algorithms = [
        ORIGINAL,
        validated["candidate"],
        RABBIT_CSR,
        RABBIT_BOOST,
    ]
    expected_cells = {
        (graph, kernel, algorithm)
        for graph in graphs
        for kernel in kernels
        for algorithm in algorithms
    }
    process_values: dict[
        tuple[str, str, str], list[dict[str, Any]]
    ] = {key: [] for key in expected_cells}
    timing_inputs = []
    cohorts = set()

    for replicate in range(4):
        rep_root = root / f"timing_final_rep{replicate}"
        timing_path = (
            rep_root
            / "vldb_paper"
            / "exp2_speedup"
            / "speedup_results.json"
        )
        resource_path = root / f"final_timing_resource_rep{replicate}.json"
        if not timing_path.is_file() or not resource_path.is_file():
            raise FileNotFoundError(
                f"Clean terminal replicate {replicate} is incomplete"
            )
        resource = json.loads(resource_path.read_text())
        if (
            not _resource_receipt_valid(
                resource,
                protocol,
                frozen["resource_amendment"],
                kind="timing-final",
                replicate=replicate,
            )
        ):
            raise ValueError(
                f"Terminal resource gate failed: replicate {replicate}"
            )

        campaign_paths = list(
            (rep_root / "vldb_paper" / "campaigns").glob("*.json")
        )
        if len(campaign_paths) != 1:
            raise ValueError(
                f"Expected one campaign receipt for replicate {replicate}"
            )
        campaign_path = campaign_paths[0]
        campaign = json.loads(campaign_path.read_text())
        expected_order = protocol["execution"]["arm_orders"][replicate]
        expected_kernels = protocol["execution"]["kernel_orders"][replicate]
        expected_graphs = protocol["execution"]["graph_orders"][replicate]
        if (
            campaign.get("git_revision")
            != EXECUTION_COMMIT[:8]
            or campaign.get("algorithm_filter") != expected_order
            or campaign.get("benchmarks") != expected_kernels
            or campaign.get("trials")
            != protocol["execution"]["trials_per_process"]
            or campaign.get("mapping_draw_count_override") != 4
            or campaign.get("mapping_draw_index") != replicate
            or list(campaign.get("graphs", {})) != expected_graphs
        ):
            raise ValueError(
                f"Terminal campaign policy changed: replicate {replicate}"
            )

        rows = json.loads(timing_path.read_text())
        if len(rows) != len(expected_cells):
            raise ValueError(
                f"Terminal timing replicate {replicate} has "
                f"{len(rows)} rows, expected {len(expected_cells)}"
            )
        cohort = str(campaign["measurement_cohort_id"])
        cohorts.add(cohort)
        seen = set()
        for row in rows:
            key = (
                str(row["graph"]),
                str(row["benchmark"]),
                str(row["algo_id"]),
            )
            if key in seen or key not in expected_cells:
                raise ValueError(
                    f"Invalid terminal timing cell: {replicate}/{key}"
                )
            seen.add(key)
            trials = [float(value) for value in row["trial_times"]]
            if (
                len(trials)
                != protocol["execution"]["trials_per_process"]
                or any(not positive_finite(value) for value in trials)
                or row["cohort_id"] != cohort
                or row["verification_gate_id"]
                != validated["gate_ids"][replicate]
                or row["verification_gate_status"] != "pass"
                or row["timing_machine"]["cpu_governors"]
                != ["performance"]
                or row["timing_machine"]["intel_pstate_no_turbo"] != "1"
            ):
                raise ValueError(
                    f"Invalid terminal timing row: {replicate}/{key}"
                )
            expected_identity = (
                {"source": "direct", "algo_flags": ["-o", "0"]}
                if key[2] == ORIGINAL
                else _expected_mapping_identity(
                    validated["mapping_records"][(key[0], key[2])],
                    replicate,
                )
            )
            if (
                row["mapping_identity"] != expected_identity
                or row["mapping_identity_id"] != short_id(
                    expected_identity
                )
            ):
                raise ValueError(
                    f"Terminal timing mapping mismatch: "
                    f"{replicate}/{key}"
                )
            process_values[key].append({
                "replicate": replicate,
                "cohort_id": cohort,
                "process_median": statistics.median(trials),
                "trial_cv": _trial_cv(trials),
                "trials": trials,
            })
        if seen != expected_cells:
            raise ValueError(
                f"Terminal timing matrix incomplete: replicate {replicate}"
            )
        timing_inputs.append({
            "replicate": replicate,
            "root": str(rep_root),
            "timing_path": str(timing_path),
            "timing_sha256": sha256(timing_path),
            "resource_path": str(resource_path),
            "resource_sha256": sha256(resource_path),
            "campaign_path": str(campaign_path),
            "campaign_sha256": sha256(campaign_path),
            "cohort_id": cohort,
        })

    if len(cohorts) != 4:
        raise ValueError("Terminal process cohorts are not distinct")
    if any(len(records) != 4 for records in process_values.values()):
        raise ValueError("Terminal cells do not have four process replicates")
    return {
        "process_values": process_values,
        "timing_inputs": timing_inputs,
    }


def detect_escalation(
    root: Path,
    frozen: dict[str, Any],
    validated: dict[str, Any],
    base: dict[str, Any],
) -> dict[str, Any]:
    protocol = frozen["protocol"]
    spread_limit = 1.20
    cv_limit = 0.10
    blocks: dict[tuple[str, str], dict[str, Any]] = {}
    cell_records = []
    for key, records in sorted(base["process_values"].items()):
        medians = [record["process_median"] for record in records]
        cvs = [record["trial_cv"] for record in records]
        spread = max(medians) / min(medians)
        triggered = spread > spread_limit or max(cvs) > cv_limit
        cell = {
            "graph": key[0],
            "kernel": key[1],
            "algorithm": key[2],
            "process_medians": medians,
            "maximum_over_minimum": spread,
            "trial_cvs": cvs,
            "triggered": triggered,
        }
        cell_records.append(cell)
        if triggered:
            block = blocks.setdefault(
                (key[0], key[1]),
                {
                    "graph": key[0],
                    "kernel": key[1],
                    "triggering_algorithms": [],
                    "triggering_cells": [],
                },
            )
            block["triggering_algorithms"].append(key[2])
            block["triggering_cells"].append(cell)

    decision = {
        "schema": "graphbrew-native-midreuse-terminal-escalation/v1",
        "created_at_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "timing_amendment_sha256":
            EXPECTED_TIMING_AMENDMENT_SHA256,
        "control_amendment_sha256":
            EXPECTED_CONTROL_AMENDMENT_SHA256,
        "resource_amendment_sha256":
            EXPECTED_RESOURCE_AMENDMENT_SHA256,
        "timing_inputs": base["timing_inputs"],
        "policy": protocol["escalation"],
        "thresholds": {
            "maximum_over_minimum": spread_limit,
            "trial_cv": cv_limit,
        },
        "triggered_blocks": list(blocks.values()),
        "triggered_block_count": len(blocks),
        "triggered_cell_count": sum(
            record["triggered"] for record in cell_records
        ),
        "cell_diagnostics": cell_records,
        "status": (
            "escalation-required" if blocks else "no-escalation"
        ),
        "effect_sizes_inspected": False,
    }
    return decision


def _load_escalated_timing(
    root: Path,
    frozen: dict[str, Any],
    validated: dict[str, Any],
    decision: dict[str, Any],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    if not decision["triggered_blocks"]:
        return {}
    protocol = frozen["protocol"]
    algorithms = [
        ORIGINAL,
        validated["candidate"],
        RABBIT_CSR,
        RABBIT_BOOST,
    ]
    plan = frozen["escalation_plan"]
    expected_blocks = {
        (block["graph"], block["kernel"])
        for block in decision["triggered_blocks"]
    }
    if (
        plan["escalation_decision_sha256"]
        != sha256(root / "escalation_decision.json")
        or {
            (block["graph"], block["kernel"])
            for block in plan["triggered_blocks"]
        } != expected_blocks
        or plan["effect_sizes_inspected"] is not False
    ):
        raise ValueError("Terminal escalation plan changed")

    output: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    kernel_cohorts: dict[str, set[str]] = {}
    for run in plan["runs"]:
        graph_list = list(run["graphs"])
        kernel = str(run["kernel"])
        replicate = int(run["replicate"])
        rep_root = Path(run["root"])
        if (
            run["algorithms"]
            != protocol["execution"]["arm_orders"][replicate]
            or run["mapping_draw_index"] != replicate
            or run["trials"] != 7
            or any(
                (graph, kernel) not in expected_blocks
                for graph in graph_list
            )
        ):
            raise ValueError(
                f"Invalid escalation plan run: {kernel}/rep{replicate}"
            )
        timing_path = (
            rep_root
            / "vldb_paper"
            / "exp2_speedup"
            / "speedup_results.json"
        )
        resource_path = (
            root
            / "escalation"
            / kernel
            / f"resource_rep{replicate}.json"
        )
        if not timing_path.is_file() or not resource_path.is_file():
            raise FileNotFoundError(
                f"Escalation is incomplete: {kernel}/rep{replicate}"
            )
        resource = json.loads(resource_path.read_text())
        if not _resource_receipt_valid(
            resource,
            protocol,
            frozen["resource_amendment"],
            kind="timing-escalation",
            replicate=replicate,
            graphs=graph_list,
            kernel=kernel,
        ):
            raise ValueError(
                f"Escalation resource gate failed: "
                f"{kernel}/rep{replicate}"
            )
        campaign_paths = list(
            (rep_root / "vldb_paper" / "campaigns").glob("*.json")
        )
        if len(campaign_paths) != 1:
            raise ValueError(
                f"Escalation campaign receipt is missing: "
                f"{kernel}/rep{replicate}"
            )
        campaign = json.loads(campaign_paths[0].read_text())
        expected_algorithms = protocol["execution"][
            "arm_orders"
        ][replicate]
        if (
            campaign.get("git_revision") != EXECUTION_COMMIT[:8]
            or campaign.get("algorithm_filter")
            != expected_algorithms
            or campaign.get("benchmarks") != [kernel]
            or campaign.get("trials") != 7
            or campaign.get("mapping_draw_count_override") != 4
            or campaign.get("mapping_draw_index") != replicate
            or list(campaign.get("graphs", {})) != graph_list
        ):
            raise ValueError(
                f"Escalation campaign policy changed: "
                f"{kernel}/rep{replicate}"
            )
        cohort = str(campaign["measurement_cohort_id"])
        kernel_cohorts.setdefault(kernel, set()).add(cohort)
        rows = json.loads(timing_path.read_text())
        expected_keys = {
            (graph, kernel, algorithm)
            for graph in graph_list
            for algorithm in algorithms
        }
        row_index = {}
        if len(rows) != len(expected_keys):
            raise ValueError(
                f"Escalation run has {len(rows)} rows: "
                f"{kernel}/rep{replicate}"
            )
        for row in rows:
            key = (
                str(row["graph"]),
                str(row["benchmark"]),
                str(row["algo_id"]),
            )
            if key in row_index or key not in expected_keys:
                raise ValueError(f"Invalid escalation cell: {key}")
            row_index[key] = row
            trials = [float(value) for value in row["trial_times"]]
            expected_identity = (
                {"source": "direct", "algo_flags": ["-o", "0"]}
                if key[2] == ORIGINAL
                else _expected_mapping_identity(
                    validated["mapping_records"][(key[0], key[2])],
                    replicate,
                )
            )
            if (
                len(trials) != 7
                or any(not positive_finite(value) for value in trials)
                or row["cohort_id"] != cohort
                or row["verification_gate_id"]
                != validated["gate_ids"][replicate]
                or row["verification_gate_status"] != "pass"
                or row["mapping_identity"] != expected_identity
                or row["mapping_identity_id"]
                != short_id(expected_identity)
                or row["timing_machine"]["cpu_governors"]
                != ["performance"]
                or row["timing_machine"]["intel_pstate_no_turbo"]
                != "1"
            ):
                raise ValueError(
                    f"Invalid escalation timing row: {key}"
                )
            output.setdefault(key, []).append({
                "replicate": replicate + 4,
                "cohort_id": cohort,
                "process_median": statistics.median(trials),
                "trial_cv": _trial_cv(trials),
                "trials": trials,
                "timing_path": str(timing_path),
                "timing_sha256": sha256(timing_path),
                "resource_path": str(resource_path),
                "resource_sha256": sha256(resource_path),
                "campaign_path": str(campaign_paths[0]),
                "campaign_sha256": sha256(campaign_paths[0]),
            })
        if set(row_index) != expected_keys:
            raise ValueError(
                f"Escalation arm matrix is incomplete: "
                f"{kernel}/rep{replicate}"
            )
    if (
        set(kernel_cohorts) != set(protocol["primary_kernels"])
        or any(len(cohorts) != 4 for cohorts in kernel_cohorts.values())
    ):
        raise ValueError("Escalation cohorts are not distinct")
    expected_keys = {
        (graph, kernel, algorithm)
        for graph, kernel in expected_blocks
        for algorithm in algorithms
    }
    if set(output) != expected_keys:
        raise ValueError("Escalation timing matrix is incomplete")
    if any(len(records) != 4 for records in output.values()):
        raise ValueError("Escalation cells are incomplete")
    return output


def _mapping_seconds(
    validated: dict[str, Any],
) -> dict[tuple[str, str], float]:
    return {
        key: statistics.median([
            float(draw["reorder_time"]) for draw in record["draws"]
        ])
        for key, record in validated["mapping_records"].items()
    }


def _cell_estimates(
    base: dict[str, Any],
    escalated: dict[
        tuple[str, str, str], list[dict[str, Any]]
    ],
) -> tuple[
    dict[tuple[str, str, str], float],
    dict[tuple[str, str, str], list[float]],
]:
    estimates = {}
    process_samples = {}
    for key, records in base["process_values"].items():
        combined = records + escalated.get(key, [])
        values = [record["process_median"] for record in combined]
        estimates[key] = statistics.median(values)
        process_samples[key] = values
    return estimates, process_samples


def _ratio_summary(
    *,
    protocol: dict[str, Any],
    validated: dict[str, Any],
    cell_seconds: dict[tuple[str, str, str], float],
    mapping_seconds: dict[tuple[str, str], float],
    reuse: int,
) -> dict[str, Any]:
    candidate = validated["candidate"]
    kernels = protocol["primary_kernels"]
    per_graph = {}
    for graph_record in protocol["graphs"]:
        graph = graph_record["graph"]
        candidate_mapping = mapping_seconds[(graph, candidate)]
        values = {
            "original_over_candidate": [],
            "rabbit_csr_over_candidate": [],
            "rabbit_boost_over_candidate": [],
            "min_rabbit_over_candidate": [],
        }
        kernel_only = {
            "original_over_candidate": [],
            "rabbit_csr_over_candidate": [],
            "rabbit_boost_over_candidate": [],
            "min_rabbit_over_candidate": [],
        }
        per_kernel = {}
        for kernel in kernels:
            candidate_kernel = cell_seconds[(graph, kernel, candidate)]
            candidate_total = candidate_mapping + reuse * candidate_kernel
            original_kernel = cell_seconds[(graph, kernel, ORIGINAL)]
            csr_kernel = cell_seconds[(graph, kernel, RABBIT_CSR)]
            boost_kernel = cell_seconds[(graph, kernel, RABBIT_BOOST)]
            original_total = reuse * original_kernel
            csr_total = (
                mapping_seconds[(graph, RABBIT_CSR)]
                + reuse * csr_kernel
            )
            boost_total = (
                mapping_seconds[(graph, RABBIT_BOOST)]
                + reuse * boost_kernel
            )
            ratios = {
                "original_over_candidate":
                    original_total / candidate_total,
                "rabbit_csr_over_candidate":
                    csr_total / candidate_total,
                "rabbit_boost_over_candidate":
                    boost_total / candidate_total,
                "min_rabbit_over_candidate":
                    min(csr_total, boost_total) / candidate_total,
            }
            kernel_ratios = {
                "original_over_candidate":
                    original_kernel / candidate_kernel,
                "rabbit_csr_over_candidate":
                    csr_kernel / candidate_kernel,
                "rabbit_boost_over_candidate":
                    boost_kernel / candidate_kernel,
                "min_rabbit_over_candidate":
                    min(csr_kernel, boost_kernel) / candidate_kernel,
            }
            for metric in values:
                values[metric].append(ratios[metric])
                kernel_only[metric].append(kernel_ratios[metric])
            per_kernel[kernel] = {
                "candidate_kernel_seconds": candidate_kernel,
                "candidate_total_seconds": candidate_total,
                "original_total_seconds": original_total,
                "rabbit_csr_total_seconds": csr_total,
                "rabbit_boost_total_seconds": boost_total,
                "end_to_end_ratios": ratios,
                "kernel_only_ratios": kernel_ratios,
            }
        per_graph[graph] = {
            "family": graph_record["family"],
            "nodes": graph_record["nodes"],
            "candidate_action": (
                "7" if graph_record["nodes"] >= 131072 else "0"
            ),
            "candidate_mapping_seconds": candidate_mapping,
            "end_to_end": {
                metric: geometric_mean(metric_values)
                for metric, metric_values in values.items()
            },
            "kernel_only": {
                metric: geometric_mean(metric_values)
                for metric, metric_values in kernel_only.items()
            },
            "per_kernel": per_kernel,
        }

    metrics = (
        "original_over_candidate",
        "rabbit_csr_over_candidate",
        "rabbit_boost_over_candidate",
        "min_rabbit_over_candidate",
    )
    family_names = sorted({
        record["family"] for record in protocol["graphs"]
    })
    families = {
        family: {
            metric: geometric_mean([
                record["end_to_end"][metric]
                for record in per_graph.values()
                if record["family"] == family
            ])
            for metric in metrics
        }
        for family in family_names
    }
    return {
        "reuse": reuse,
        "corpus_gm": {
            metric: geometric_mean([
                record["end_to_end"][metric]
                for record in per_graph.values()
            ])
            for metric in metrics
        },
        "worst_graph": {
            metric: min(
                record["end_to_end"][metric]
                for record in per_graph.values()
            )
            for metric in metrics
        },
        "graphs_below_one": {
            metric: [
                graph for graph, record in per_graph.items()
                if record["end_to_end"][metric] < 1.0
            ]
            for metric in metrics
        },
        "families": families,
        "per_graph": per_graph,
    }


def _bootstrap(
    *,
    protocol: dict[str, Any],
    validated: dict[str, Any],
    process_samples: dict[
        tuple[str, str, str], list[float]
    ],
    mapping_seconds: dict[tuple[str, str], float],
    reuse: int,
) -> dict[str, Any]:
    candidate = validated["candidate"]
    kernels = protocol["primary_kernels"]
    graph_family = {
        record["graph"]: record["family"]
        for record in protocol["graphs"]
    }
    family_graphs: dict[str, list[str]] = {}
    for graph, family in graph_family.items():
        family_graphs.setdefault(family, []).append(graph)
    families = sorted(family_graphs)
    rng = random.Random(BOOTSTRAP_SEED)
    distributions = {
        "original_over_candidate": [],
        "rabbit_csr_over_candidate": [],
        "rabbit_boost_over_candidate": [],
        "min_rabbit_over_candidate": [],
    }

    def sampled_cell(graph: str, kernel: str, algorithm: str) -> float:
        values = process_samples[(graph, kernel, algorithm)]
        return statistics.median([
            rng.choice(values) for _ in values
        ])

    for _ in range(BOOTSTRAP_SAMPLES):
        graph_ratios = {
            metric: [] for metric in distributions
        }
        sampled_families = [
            rng.choice(families) for _ in families
        ]
        for family in sampled_families:
            sampled_graphs = [
                rng.choice(family_graphs[family])
                for _ in family_graphs[family]
            ]
            for graph in sampled_graphs:
                kernel_ratios = {
                    metric: [] for metric in distributions
                }
                for kernel in kernels:
                    candidate_kernel = sampled_cell(
                        graph, kernel, candidate
                    )
                    candidate_total = (
                        mapping_seconds[(graph, candidate)]
                        + reuse * candidate_kernel
                    )
                    original_total = reuse * sampled_cell(
                        graph, kernel, ORIGINAL
                    )
                    csr_total = (
                        mapping_seconds[(graph, RABBIT_CSR)]
                        + reuse * sampled_cell(
                            graph, kernel, RABBIT_CSR
                        )
                    )
                    boost_total = (
                        mapping_seconds[(graph, RABBIT_BOOST)]
                        + reuse * sampled_cell(
                            graph, kernel, RABBIT_BOOST
                        )
                    )
                    ratios = {
                        "original_over_candidate":
                            original_total / candidate_total,
                        "rabbit_csr_over_candidate":
                            csr_total / candidate_total,
                        "rabbit_boost_over_candidate":
                            boost_total / candidate_total,
                        "min_rabbit_over_candidate":
                            min(csr_total, boost_total)
                            / candidate_total,
                    }
                    for metric, value in ratios.items():
                        kernel_ratios[metric].append(value)
                for metric in distributions:
                    graph_ratios[metric].append(
                        geometric_mean(kernel_ratios[metric])
                    )
        for metric in distributions:
            distributions[metric].append(
                geometric_mean(graph_ratios[metric])
            )

    return {
        metric: {
            "lower_95": _percentile(values, 0.025),
            "upper_95": _percentile(values, 0.975),
        }
        for metric, values in distributions.items()
    }


def final_analysis(
    root: Path,
    frozen: dict[str, Any],
    validated: dict[str, Any],
    base: dict[str, Any],
    decision: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = frozen["protocol"]
    escalated = _load_escalated_timing(
        root, frozen, validated, decision
    )
    cell_seconds, process_samples = _cell_estimates(base, escalated)
    mapping_seconds = _mapping_seconds(validated)
    primary = _ratio_summary(
        protocol=protocol,
        validated=validated,
        cell_seconds=cell_seconds,
        mapping_seconds=mapping_seconds,
        reuse=protocol["primary_reuse"],
    )
    sensitivity = {
        str(reuse): _ratio_summary(
            protocol=protocol,
            validated=validated,
            cell_seconds=cell_seconds,
            mapping_seconds=mapping_seconds,
            reuse=reuse,
        )
        for reuse in protocol["secondary_reuse"]
    }
    bootstrap = _bootstrap(
        protocol=protocol,
        validated=validated,
        process_samples=process_samples,
        mapping_seconds=mapping_seconds,
        reuse=protocol["primary_reuse"],
    )

    candidate = validated["candidate"]
    eligible_graphs = [
        record["graph"] for record in protocol["graphs"]
        if record["nodes"] >= 131072
    ]
    fallback_graphs = [
        record["graph"] for record in protocol["graphs"]
        if record["nodes"] < 131072
    ]
    mapping_ratios = {
        graph: (
            mapping_seconds[(graph, candidate)]
            / min(
                mapping_seconds[(graph, RABBIT_CSR)],
                mapping_seconds[(graph, RABBIT_BOOST)],
            )
        )
        for graph in eligible_graphs
    }
    fallback_mapping_seconds = {
        graph: mapping_seconds[(graph, candidate)]
        for graph in fallback_graphs
    }
    correctness_pass = (
        frozen["verification"]["all_passed"] is True
        and frozen["verification"]["cells"] == 960
    )
    gates = {
        "T1_correctness": {
            "passed_cells": frozen["verification"]["cells"],
            "pass": correctness_pass,
        },
        "T2_original": {
            "point_gm":
                primary["corpus_gm"]["original_over_candidate"],
            "bootstrap_95": [
                bootstrap["original_over_candidate"]["lower_95"],
                bootstrap["original_over_candidate"]["upper_95"],
            ],
        },
        "T3_rabbit_csr": {
            "point_gm":
                primary["corpus_gm"]["rabbit_csr_over_candidate"],
            "bootstrap_95": [
                bootstrap["rabbit_csr_over_candidate"]["lower_95"],
                bootstrap["rabbit_csr_over_candidate"]["upper_95"],
            ],
        },
        "T4_rabbit_boost": {
            "point_gm":
                primary["corpus_gm"]["rabbit_boost_over_candidate"],
            "bootstrap_95": [
                bootstrap["rabbit_boost_over_candidate"]["lower_95"],
                bootstrap["rabbit_boost_over_candidate"]["upper_95"],
            ],
        },
        "T5_bounded_harm": {
            "original_worst":
                primary["worst_graph"]["original_over_candidate"],
            "min_rabbit_worst":
                primary["worst_graph"]["min_rabbit_over_candidate"],
            "original_graphs_below_one":
                primary["graphs_below_one"]["original_over_candidate"],
            "min_rabbit_graphs_below_one":
                primary["graphs_below_one"]["min_rabbit_over_candidate"],
        },
        "T6_mapping": {
            "eligible_graphs": eligible_graphs,
            "fallback_graphs": fallback_graphs,
            "candidate_over_min_rabbit_per_graph": mapping_ratios,
            "candidate_over_min_rabbit_gm":
                geometric_mean(list(mapping_ratios.values())),
            "candidate_over_min_rabbit_worst": max(
                mapping_ratios.values()
            ),
            "fallback_candidate_mapping_seconds":
                fallback_mapping_seconds,
        },
    }
    for name in ("T2_original", "T3_rabbit_csr", "T4_rabbit_boost"):
        gates[name]["pass"] = (
            gates[name]["point_gm"] > 1.0
            and gates[name]["bootstrap_95"][0] > 1.0
        )
    gates["T5_bounded_harm"]["pass"] = (
        gates["T5_bounded_harm"]["original_worst"] >= 0.90
        and gates["T5_bounded_harm"]["min_rabbit_worst"] >= 0.90
        and len(
            gates["T5_bounded_harm"]["original_graphs_below_one"]
        ) <= 2
        and len(
            gates["T5_bounded_harm"]["min_rabbit_graphs_below_one"]
        ) <= 2
    )
    gates["T6_mapping"]["pass"] = (
        gates["T6_mapping"]["candidate_over_min_rabbit_gm"] <= 0.20
        and gates["T6_mapping"][
            "candidate_over_min_rabbit_worst"
        ] <= 0.30
    )
    passed = all(record["pass"] for record in gates.values())

    created_at = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()
    analysis = {
        "schema": "graphbrew-native-midreuse-terminal-analysis/v1",
        "created_at_utc": created_at,
        "inputs": {
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "corpus_manifest_sha256": EXPECTED_CORPUS_SHA256,
            "mapping_manifest_sha256": EXPECTED_MAPPING_SHA256,
            "verification_manifest_sha256":
                EXPECTED_VERIFICATION_SHA256,
            "timing_amendment_sha256":
                EXPECTED_TIMING_AMENDMENT_SHA256,
            "control_amendment_sha256":
                EXPECTED_CONTROL_AMENDMENT_SHA256,
            "resource_amendment_sha256":
                EXPECTED_RESOURCE_AMENDMENT_SHA256,
            "escalation_plan_sha256":
                EXPECTED_ESCALATION_PLAN_SHA256,
            "escalation_decision_sha256": sha256(
                root / "escalation_decision.json"
            ),
            "analysis_program_sha256": sha256(
                Path(__file__).resolve()
            ),
            "execution_repository_commit": EXECUTION_COMMIT,
            "analysis_repository_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                text=True,
            ).strip(),
            "timing_inputs": base["timing_inputs"],
        },
        "policy": {
            "cell_estimator":
                "median of four process medians; median of eight for "
                "predeclared escalated blocks",
            "trials_per_process": 7,
            "mapping_estimator": "median of four mapping draws",
            "primary_reuse": protocol["primary_reuse"],
            "primary_kernels": protocol["primary_kernels"],
            "pr_iterations_per_invocation": 20,
            "pr_iterations_at_primary_reuse":
                20 * protocol["primary_reuse"],
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "inference_unit": protocol["exposure_audit"][
                "inference_unit"
            ],
        },
        "escalation": {
            "triggered_blocks": decision["triggered_blocks"],
            "escalated_cell_count": len(escalated),
        },
        "primary": primary,
        "bootstrap": bootstrap,
        "sensitivity": sensitivity,
        "mapping": gates["T6_mapping"],
        "gates": gates,
        "pass": passed,
        "claim_boundary": (
            protocol["paper_claim_if_passed"]
            if passed else protocol["paper_claim_if_failed"]
        ),
    }
    terminal_decision = {
        "schema": "graphbrew-native-midreuse-terminal-decision/v1",
        "created_at_utc": created_at,
        "status": (
            "terminal-gates-passed"
            if passed else "terminal-gates-failed"
        ),
        "candidate": candidate,
        "primary_reuse": protocol["primary_reuse"],
        "gates": gates,
        "paper_claim": analysis["claim_boundary"],
        "development_effect_is_not_terminal":
            protocol["paper_claim_if_failed"],
    }
    return analysis, terminal_decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--detect-escalation",
        action="store_true",
        help="Write the preregistered variability decision without effects",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    frozen = _load_frozen_inputs(
        root,
        require_escalation_plan=not args.detect_escalation,
    )
    validated = _validate_frozen_inputs(root, frozen)
    base = _load_base_timing(root, frozen, validated)

    decision_path = root / "escalation_decision.json"
    if args.detect_escalation:
        decision = detect_escalation(
            root, frozen, validated, base
        )
        _atomic_write(decision_path, decision)
        print(json.dumps({
            "decision": str(decision_path),
            "sha256": sha256(decision_path),
            "status": decision["status"],
            "triggered_blocks": decision["triggered_block_count"],
        }, indent=2))
        return

    if not decision_path.is_file():
        raise FileNotFoundError(
            "Run --detect-escalation before final analysis"
        )
    decision = json.loads(decision_path.read_text())
    expected_decision = detect_escalation(
        root, frozen, validated, base
    )
    comparable_decision = {
        key: value for key, value in decision.items()
        if key != "created_at_utc"
    }
    comparable_expected = {
        key: value for key, value in expected_decision.items()
        if key != "created_at_utc"
    }
    if (
        decision.get("schema")
        != "graphbrew-native-midreuse-terminal-escalation/v1"
        or decision.get("effect_sizes_inspected") is not False
        or comparable_decision != comparable_expected
    ):
        raise ValueError("Invalid terminal escalation decision")

    analysis, terminal_decision = final_analysis(
        root, frozen, validated, base, decision
    )
    analysis_path = root / "analysis.json"
    terminal_decision_path = root / "decision.json"
    _atomic_write(analysis_path, analysis)
    _atomic_write(terminal_decision_path, terminal_decision)
    receipt = {
        "schema": "graphbrew-native-midreuse-terminal-final/v1",
        "frozen_at_utc": analysis["created_at_utc"],
        "files": {
            path.name: {
                "path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in (
                root / "protocol.json",
                root / "corpus_manifest.json",
                root / "mapping_manifest.json",
                root / "verification_manifest.json",
                root / "timing_execution_amendment.json",
                root / "control_isolation_amendment.json",
                root / "resource_gate_amendment_v2.json",
                root / "escalation_execution_plan.json",
                decision_path,
                analysis_path,
                terminal_decision_path,
                Path(__file__).resolve(),
            )
        },
        "decision": terminal_decision,
    }
    receipt_path = root / "final_receipt.json"
    _atomic_write(receipt_path, receipt)
    print(json.dumps({
        "analysis": str(analysis_path),
        "decision": str(terminal_decision_path),
        "final_receipt": str(receipt_path),
        "final_receipt_sha256": sha256(receipt_path),
        "status": terminal_decision["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
