#!/usr/bin/env python3
"""Content-bound orchestration for the amortized CPU selector sprint."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.adaptive.runner import (
    _artifact_binding,
    _atomic_json,
    _canonical_json_sha256,
    _load_json,
    _source_graph_provenance,
    _taskset_command,
    _validate_source_manifest,
)
from scripts.experiments.vldb.config import (
    ADAPTIVE_CPU_EXPANSION_GRAPHS,
    EVAL_GRAPHS,
)
from scripts.lib.ml.portfolio import DEPLOYABLE_ARM_SPECS
from scripts.lib.ml.source_policy import (
    ADAPTIVE_SOURCE_COUNT,
    ADAPTIVE_SOURCE_MIN_REACHABILITY,
    ADAPTIVE_SOURCE_POLICY_ID,
    ADAPTIVE_SOURCE_SEED,
)
from scripts.lib.pipeline.benchmark import file_sha256


CPU_SPRINT_SCHEMA = "adaptive-cpu-sprint-scope/v1"
CPU_SPRINT_GRAPHS = (
    *EVAL_GRAPHS,
    *ADAPTIVE_CPU_EXPANSION_GRAPHS,
)
CPU_HEADLINE_KERNELS = (
    "bc",
    "bfs",
    "cc",
    "cc_sv",
    "pr",
    "pr_spmv",
)
CPU_DIAGNOSTIC_KERNELS = ("sssp",)
CPU_REUSE_REGIMES = (20, 50, 100)
CPU_PRIMARY_REUSE = 50
CPU_MAPPING_DRAWS = 3
CPU_BUDGET_HOURS = 168.0
CPU_RESERVE_HOURS = 16.8
CPU_PRIOR_CONSUMED_HOURS = (
    8.048300127879484 + 3.075632533386686
)
CPU_EXPANSION_TARGET_GRAPHS = 30
CPU_MDE_LIMIT = 0.05
CPU_RAPID_KERNELS = ("pr_spmv", "cc", "cc_sv")
CPU_RAPID_PROJECTED_HIGH_HOURS = 12.0


def _sprint_root(artifact_root: Path) -> Path:
    return artifact_root / "adaptive_selector" / "cpu_sprint"


def _graph_records(graph_root: Path) -> list[dict[str, Any]]:
    records = []
    frozen = {
        str(graph["name"]): graph
        for graph in EVAL_GRAPHS
    }
    for configured in CPU_SPRINT_GRAPHS:
        graph_name = str(configured["name"])
        graph_path = graph_root / graph_name / f"{graph_name}.sg"
        provenance = _source_graph_provenance(graph_path)
        nodes = int(provenance["nodes"])
        directed_edges = int(provenance["directed_edges"])
        if graph_name in frozen:
            expected = frozen[graph_name]
            if (
                nodes != int(expected["nodes"])
                or directed_edges // 2
                    != int(expected["undirected_edges"])
            ):
                raise RuntimeError(
                    f"Frozen graph dimensions changed: {graph_name}")
        records.append({
            "name": graph_name,
            "short": configured["short"],
            "type": configured["type"],
            "nodes": nodes,
            "directed_edges": directed_edges,
            "undirected_edges": directed_edges // 2,
            "graph_path": str(graph_path),
            "graph_provenance": provenance,
            "graph_provenance_sha256":
                _canonical_json_sha256(provenance),
        })
    if (
        len(records) != CPU_EXPANSION_TARGET_GRAPHS
        or len({record["name"] for record in records})
            != CPU_EXPANSION_TARGET_GRAPHS
    ):
        raise RuntimeError("Adaptive CPU graph scope changed")
    return records


def write_scope_plan(
    artifact_root: Path,
    graph_root: Path,
    *,
    refreeze: bool = False,
) -> Path:
    sprint_root = _sprint_root(artifact_root)
    sprint_root.mkdir(parents=True, exist_ok=True)
    predata_path = sprint_root / "predata_diagnostic.json"
    predata = _load_json(predata_path)
    if (
        predata.get("schema")
            != "adaptive-cpu-predata-diagnostic/v1"
        or predata.get("decision") != "expand-to-30-graphs"
    ):
        raise RuntimeError(
            "CPU sprint pre-data diagnostic is missing")
    prior_root = (
        artifact_root / "adaptive_selector" / "sprint1")
    input_artifacts = {
        "predata_diagnostic": _artifact_binding(predata_path),
        "sprint1_analysis": _artifact_binding(
            prior_root / "pilot_analysis.json"),
        "cache_amendment_analysis": _artifact_binding(
            prior_root / "cache_amendment2" / "analysis.json"),
        "sprint1_budget": _artifact_binding(
            prior_root / "budget_projection.json"),
    }
    records = _graph_records(graph_root)
    plan = {
        "schema": CPU_SPRINT_SCHEMA,
        "status": "source-generation-authorized",
        "claim_scope": (
            "CPU-only amortization-aware ordering selection for "
            "randomized or unknown-label inputs; no novel base-ordering, "
            "cache, hardware-cache, or natural-label profitability claim"
        ),
        "contribution_delta": (
            "reuse-aware complete-cost selection with lightweight topology "
            "features, graph-held-out evaluation, and OOD abstention"
        ),
        "claim_eligible_portfolio":
            list(DEPLOYABLE_ARM_SPECS),
        "headline_kernels": list(CPU_HEADLINE_KERNELS),
        "diagnostic_kernels": list(CPU_DIAGNOSTIC_KERNELS),
        "reuse_regimes": list(CPU_REUSE_REGIMES),
        "primary_reuse": CPU_PRIMARY_REUSE,
        "graph_count": len(records),
        "graphs": records,
        "source_policy": {
            "policy_id": ADAPTIVE_SOURCE_POLICY_ID,
            "seed": ADAPTIVE_SOURCE_SEED,
            "source_count": ADAPTIVE_SOURCE_COUNT,
            "minimum_reachability_fraction":
                ADAPTIVE_SOURCE_MIN_REACHABILITY,
        },
        "mapping_draw_policy": {
            "randomized_arm_draws": CPU_MAPPING_DRAWS,
            "oracle_cost": "draw-averaged-never-best-of-draws",
            "h4_supergraph_ablation": "not-claimed",
        },
        "evaluation_policy": {
            "outer_split": "leave-one-base-graph-out",
            "inner_split": "leave-one-training-graph-out",
            "static_baselines": [
                "global-static-selected-on-training-graphs",
                "per-kernel-static-selected-on-training-graphs",
                "stronger-training-selected-static",
            ],
            "ood_policy": "training-fold-calibrated-abstention",
            "no_abstention_twin_required": True,
            "sssp_headline_eligible": False,
        },
        "predata_gates": {
            "mde_limit": CPU_MDE_LIMIT,
            "n11_mde_failed": True,
            "n30_projected_mde": {
                "20": 0.0450755006334953,
                "50": 0.04885902600717773,
                "100": 0.04320079998316295,
            },
            "stale_oracle_headroom": {
                "20": 1.102050594199041,
                "50": 1.1324066646038842,
                "100": 1.0850806054126005,
            },
            "stale_deployable_linear_selector": 0.9867717820991159,
            "stale_shallow_tree_selector": 0.9869701554809942,
            "fresh_rapid_diagnostic_required": True,
        },
        "budget_policy": {
            "budget_hours": CPU_BUDGET_HOURS,
            "reserve_hours": CPU_RESERVE_HOURS,
            "prior_consumed_hours": CPU_PRIOR_CONSUMED_HOURS,
            "cache_collection_hours": 0.0,
            "hardware_cache_validation_hours": 0.0,
            "full_collection_authorized": False,
        },
        "authorized_actions": {
            "source_generation": True,
            "rapid_diagnostic": False,
            "full_collection": False,
            "training": False,
            "deployment": False,
        },
        "input_artifacts": input_artifacts,
    }
    plan_path = sprint_root / "scope_plan.json"
    if (
        plan_path.is_file()
        and _load_json(plan_path) != plan
        and not refreeze
    ):
        raise RuntimeError(
            "Frozen CPU sprint scope changed; refreeze after review")
    _atomic_json(plan, plan_path)
    return plan_path


def generate_source_manifests(
    artifact_root: Path,
    graph_root: Path,
    *,
    threads: int,
    cpu_list: str | None,
    force: bool = False,
    refreeze: bool = False,
) -> Path:
    if threads <= 0:
        raise ValueError("CPU source generation threads must be positive")
    sprint_root = _sprint_root(artifact_root)
    plan_path = sprint_root / "scope_plan.json"
    plan = _load_json(plan_path)
    if (
        plan.get("schema") != CPU_SPRINT_SCHEMA
        or not plan.get("authorized_actions", {}).get(
            "source_generation")
    ):
        raise RuntimeError(
            "CPU sprint source generation is not authorized")
    if plan["input_artifacts"]["predata_diagnostic"] != (
        _artifact_binding(
            sprint_root / "predata_diagnostic.json")
    ):
        raise RuntimeError(
            "CPU sprint pre-data diagnostic changed")

    binary = PROJECT_ROOT / "bench" / "bin" / "cc"
    subprocess.run(
        ["make", f"-j{min(threads, 4)}", "bench/bin/cc"],
        cwd=PROJECT_ROOT,
        check=True,
    )
    out_dir = sprint_root / "sources"
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_manifests = {}
    commands = {}
    graph_records = {
        record["name"]: record
        for record in _graph_records(graph_root)
    }
    for graph_name in sorted(graph_records):
        record = graph_records[graph_name]
        graph_path = Path(record["graph_path"])
        graph_provenance = record["graph_provenance"]
        output_path = out_dir / f"{graph_name}.json"
        log_path = out_dir / f"{graph_name}.log"
        base_command = [
            str(binary),
            "-f", str(graph_path),
            "-Y", str(output_path),
        ]
        command = _taskset_command(base_command, cpu_list)
        graph = {
            "name": graph_name,
            "nodes": record["nodes"],
            "undirected_edges": record["undirected_edges"],
        }
        if output_path.is_file() and not force:
            payload = _load_json(output_path)
            if (
                payload.get("graph_provenance")
                    != graph_provenance
                or payload.get("generator_command") != command
                or payload.get("omp_num_threads") != threads
                or payload.get("cpu_list") != cpu_list
            ):
                raise RuntimeError(
                    "Existing CPU source manifest is stale; "
                    "use --force-sources and --refreeze-sources")
            _validate_source_manifest(
                payload,
                graph,
                expected_graph_path=graph_path,
                graph_provenance=graph_provenance,
            )
        else:
            environment = {
                **os.environ,
                "OMP_NUM_THREADS": str(threads),
                "GRAPHBREW_DB_DIR": "",
                "GRAPHBREW_TOPOLOGY_ANALYSIS": "0",
            }
            with open(log_path, "w") as log:
                result = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=12 * 60 * 60,
                )
            if result.returncode != 0:
                raise RuntimeError(
                    f"CPU source generation failed for {graph_name}; "
                    f"see {log_path}")
            payload = _load_json(output_path)
            payload["generator_command"] = command
            payload["omp_num_threads"] = threads
            payload["cpu_list"] = cpu_list
            payload["graph_provenance"] = graph_provenance
            _atomic_json(payload, output_path)
            _validate_source_manifest(
                payload,
                graph,
                expected_graph_path=graph_path,
                graph_provenance=graph_provenance,
            )
        graph_manifests[graph_name] = payload
        commands[graph_name] = command

    bundle = {
        "schema": "adaptive-cpu-source-bundle/v1",
        "scope_plan": str(plan_path),
        "scope_plan_sha256": file_sha256(
            plan_path, use_cache=False),
        "policy_id": ADAPTIVE_SOURCE_POLICY_ID,
        "seed": ADAPTIVE_SOURCE_SEED,
        "source_count": ADAPTIVE_SOURCE_COUNT,
        "minimum_reachability_fraction":
            ADAPTIVE_SOURCE_MIN_REACHABILITY,
        "graph_count": len(graph_manifests),
        "threads": threads,
        "cpu_list": cpu_list,
        "commands": commands,
        "graphs": graph_manifests,
        "graph_provenance": {
            graph_name: payload["graph_provenance"]
            for graph_name, payload in graph_manifests.items()
        },
        "source_lists": {
            graph_name: [
                int(source["source_id"])
                for source in payload["sources"]
            ]
            for graph_name, payload in graph_manifests.items()
        },
    }
    bundle["graph_provenance_sha256"] = (
        _canonical_json_sha256(bundle["graph_provenance"])
    )
    bundle_path = sprint_root / "source_manifest.json"
    if bundle_path.is_file():
        existing = _load_json(bundle_path)
        frozen_fields = (
            "scope_plan_sha256",
            "policy_id",
            "seed",
            "source_lists",
            "graph_provenance",
            "graph_provenance_sha256",
        )
        if (
            any(
                existing.get(field) != bundle.get(field)
                for field in frozen_fields
            )
            and not refreeze
        ):
            raise RuntimeError(
                "Frozen CPU source bundle changed; refreeze after review")
    _atomic_json(bundle, bundle_path)
    return bundle_path


def write_rapid_plan(
    artifact_root: Path,
    graph_root: Path,
    *,
    cpu_list: str,
    threads: int,
    refreeze: bool = False,
) -> Path:
    if not cpu_list or threads <= 0:
        raise ValueError(
            "Rapid diagnostic requires CPU list and positive threads")
    sprint_root = _sprint_root(artifact_root)
    scope_path = sprint_root / "scope_plan.json"
    source_path = sprint_root / "source_manifest.json"
    scope = _load_json(scope_path)
    sources = _load_json(source_path)
    if (
        scope.get("schema") != CPU_SPRINT_SCHEMA
        or sources.get("schema")
            != "adaptive-cpu-source-bundle/v1"
        or sources.get("graph_count")
            != CPU_EXPANSION_TARGET_GRAPHS
        or sources.get("scope_plan_sha256")
            != file_sha256(scope_path, use_cache=False)
    ):
        raise RuntimeError(
            "CPU rapid diagnostic inputs are invalid")
    graph_names = sorted(sources["graphs"])
    rapid_root = sprint_root / "rapid"
    common = [
        "python3",
        "scripts/graphbrew_experiment.py",
        "--paper-graphs", *graph_names,
        "--paper-algorithms", *DEPLOYABLE_ARM_SPECS,
        "--paper-benchmarks", *CPU_RAPID_KERNELS,
        "--paper-trials", "1",
        "--paper-graph-dir", str(graph_root),
        "--paper-artifact-root", str(rapid_root),
        "--paper-threads", str(threads),
        "--paper-cpu-list", cpu_list,
    ]
    commands = [
        [
            *common[:2],
            "--paper-verify-gate",
            *common[2:],
        ],
        [
            *common[:2],
            "--vldb", "2",
            *common[2:],
        ],
    ]
    available_hours = (
        CPU_BUDGET_HOURS
        - CPU_RESERVE_HOURS
        - CPU_PRIOR_CONSUMED_HOURS
    )
    if CPU_RAPID_PROJECTED_HIGH_HOURS > available_hours:
        raise RuntimeError(
            "CPU rapid diagnostic exceeds remaining budget")
    plan = {
        "schema": "adaptive-cpu-rapid-plan/v1",
        "claim_eligible": False,
        "purpose": (
            "fresh 30-graph learnability and headroom falsifier "
            "before full six-kernel collection"
        ),
        "scope_plan": str(scope_path),
        "source_manifest": str(source_path),
        "input_artifacts": {
            "scope_plan": _artifact_binding(scope_path),
            "source_manifest": _artifact_binding(source_path),
        },
        "graph_count": len(graph_names),
        "graphs": graph_names,
        "algorithms": list(DEPLOYABLE_ARM_SPECS),
        "benchmarks": list(CPU_RAPID_KERNELS),
        "trials": 1,
        "command_count": len(commands),
        "commands": commands,
        "cpu_list": cpu_list,
        "threads": threads,
        "artifact_root": str(rapid_root),
        "projected_high_hours":
            CPU_RAPID_PROJECTED_HIGH_HOURS,
        "budget_hours": CPU_BUDGET_HOURS,
        "reserve_hours": CPU_RESERVE_HOURS,
        "prior_consumed_hours": CPU_PRIOR_CONSUMED_HOURS,
        "full_collection_authorized": False,
        "execution_authorization_required": True,
    }
    path = sprint_root / "rapid_plan.json"
    if (
        path.is_file()
        and _load_json(path) != plan
        and not refreeze
    ):
        raise RuntimeError(
            "Frozen CPU rapid plan changed; refreeze after review")
    _atomic_json(plan, path)
    return path


def authorize_rapid_plan(
    artifact_root: Path,
    *,
    authorization_reference: str,
    refreeze: bool = False,
) -> Path:
    if not authorization_reference:
        raise ValueError(
            "CPU rapid authorization reference is required")
    sprint_root = _sprint_root(artifact_root)
    plan_path = sprint_root / "rapid_plan.json"
    plan = _load_json(plan_path)
    for binding in plan.get("input_artifacts", {}).values():
        path = Path(binding["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(binding["bytes"])
            or file_sha256(path, use_cache=False)
                != binding["sha256"]
        ):
            raise RuntimeError(
                f"CPU rapid input changed: {path}")
    payload = {
        "schema": "adaptive-cpu-rapid-authorization/v1",
        "execution_enabled": True,
        "authorization_reference": authorization_reference,
        "plan": str(plan_path),
        "plan_sha256": file_sha256(
            plan_path, use_cache=False),
        "command_count": plan["command_count"],
        "projected_high_hours":
            plan["projected_high_hours"],
    }
    path = sprint_root / "rapid_authorization.json"
    if (
        path.is_file()
        and _load_json(path) != payload
        and not refreeze
    ):
        raise RuntimeError(
            "CPU rapid authorization changed; refreeze after review")
    _atomic_json(payload, path)
    return path


def execute_rapid_plan(
    artifact_root: Path,
    *,
    authorization_reference: str,
) -> Path:
    sprint_root = _sprint_root(artifact_root)
    plan_path = sprint_root / "rapid_plan.json"
    authorization_path = sprint_root / "rapid_authorization.json"
    plan = _load_json(plan_path)
    authorization = _load_json(authorization_path)
    if (
        authorization.get("schema")
            != "adaptive-cpu-rapid-authorization/v1"
        or not authorization.get("execution_enabled")
        or authorization.get("authorization_reference")
            != authorization_reference
        or authorization.get("plan") != str(plan_path)
        or authorization.get("plan_sha256")
            != file_sha256(plan_path, use_cache=False)
        or authorization.get("command_count")
            != plan.get("command_count")
    ):
        raise RuntimeError(
            "CPU rapid execution authorization is invalid")
    for binding in plan["input_artifacts"].values():
        path = Path(binding["path"])
        if (
            path.stat().st_size != int(binding["bytes"])
            or file_sha256(path, use_cache=False)
                != binding["sha256"]
        ):
            raise RuntimeError(
                f"CPU rapid input changed: {path}")
    results = []
    for index, command in enumerate(plan["commands"]):
        started = time.monotonic()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
        )
        results.append({
            "index": index,
            "command": command,
            "returncode": completed.returncode,
            "duration_seconds":
                time.monotonic() - started,
        })
        if completed.returncode != 0:
            raise RuntimeError(
                f"CPU rapid command failed at index {index}")
    payload = {
        "schema": "adaptive-cpu-rapid-complete/v1",
        "authorization_reference": authorization_reference,
        "plan": str(plan_path),
        "plan_sha256": file_sha256(
            plan_path, use_cache=False),
        "command_count": len(results),
        "results": results,
        "status": "complete",
    }
    path = sprint_root / "rapid_complete.json"
    _atomic_json(payload, path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GraphBrew amortized CPU selector sprint")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--generate-sources", action="store_true")
    parser.add_argument("--plan-rapid", action="store_true")
    parser.add_argument("--authorize-rapid", action="store_true")
    parser.add_argument("--execute-rapid", action="store_true")
    parser.add_argument("--force-sources", action="store_true")
    parser.add_argument("--refreeze-scope", action="store_true")
    parser.add_argument("--refreeze-sources", action="store_true")
    parser.add_argument("--refreeze-rapid", action="store_true")
    parser.add_argument("--authorization-reference")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--cpu-list")
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=Path(
            "/media/Data/00_GraphDatasets/GraphBrew"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(
            "/media/Data/00_GraphDatasets/GraphBrew/artifacts"),
    )
    args = parser.parse_args()
    if sum((
        int(args.plan),
        int(args.generate_sources),
        int(args.plan_rapid),
        int(args.authorize_rapid),
        int(args.execute_rapid),
    )) != 1:
        parser.error("Select exactly one CPU sprint stage")
    if args.plan:
        path = write_scope_plan(
            args.artifact_root.resolve(),
            args.graph_dir.resolve(),
            refreeze=args.refreeze_scope,
        )
    elif args.generate_sources:
        path = generate_source_manifests(
            args.artifact_root.resolve(),
            args.graph_dir.resolve(),
            threads=args.threads,
            cpu_list=args.cpu_list,
            force=args.force_sources,
            refreeze=args.refreeze_sources,
        )
    elif args.plan_rapid:
        path = write_rapid_plan(
            args.artifact_root.resolve(),
            args.graph_dir.resolve(),
            cpu_list=args.cpu_list or "",
            threads=args.threads,
            refreeze=args.refreeze_rapid,
        )
    elif args.authorize_rapid:
        path = authorize_rapid_plan(
            args.artifact_root.resolve(),
            authorization_reference=(
                args.authorization_reference or ""),
            refreeze=args.refreeze_rapid,
        )
    else:
        path = execute_rapid_plan(
            args.artifact_root.resolve(),
            authorization_reference=(
                args.authorization_reference or ""),
        )
    print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
