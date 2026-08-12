"""Adaptive Sprint-1 budget planner tests."""

from __future__ import annotations

import copy
import inspect
import json
import os
import shutil
from pathlib import Path

import pytest

from scripts.experiments.adaptive import runner as adaptive_runner
from scripts.experiments.adaptive.runner import (
    ALL_TIMING_ARMS,
    CACHE_KERNELS,
    H4_DENDROGRAM_ANCHOR,
    _kernel_evidence_arm,
    _mapping_evidence_arm,
    _validate_source_manifest,
    _forbidden_ambient_timing_environment,
    _validate_pilot_realized_order,
    _parse_and_validate_pilot_output,
    _validate_runtime_environment_surface,
    _run_literal_pilot_command,
    _PilotExecutionLock,
    _pilot_command_for_attempt,
    _priming_command_for_session,
    _run_pilot_command_with_retries,
    build_sprint1_budget_projection,
    reduced_capacity_set_mib,
    select_medium_pilot_graph,
    write_sprint1_budget_projection,
)
from scripts.lib.core.experiment_policy import REORDER_SEMANTICS_VERSION
from scripts.experiments.vldb import runner as vldb_runner
from scripts.lib.ml.source_policy import (
    ADAPTIVE_SOURCE_COUNT,
    ADAPTIVE_SOURCE_MIN_REACHABILITY,
    ADAPTIVE_SOURCE_POLICY_ID,
    ADAPTIVE_SOURCE_SEED,
)
from scripts.lib.pipeline.benchmark import (
    SourceContractError,
    file_crc32,
    file_sha256,
)
from scripts.experiments.vldb.config import BENCHMARKS, EVAL_GRAPHS


def test_medium_pilot_graph_is_frozen_median():
    assert select_medium_pilot_graph(EVAL_GRAPHS) == "hollywood-2009"


def test_reduced_capacity_set_keeps_boundary_and_endpoint():
    capacities = reduced_capacity_set_mib(
        "pr",
        nodes=1_000_000,
        directed_edges=10_000_000,
    )
    assert 22 in capacities
    assert len(capacities) <= 5
    assert 512 in capacities


def test_source_manifest_validation_uses_shared_policy():
    graph = EVAL_GRAPHS[0]
    payload = {
        "schema": "adaptive-source-manifest/v1",
        "policy_id": ADAPTIVE_SOURCE_POLICY_ID,
        "seed": ADAPTIVE_SOURCE_SEED,
        "source_count": ADAPTIVE_SOURCE_COUNT,
        "component_verification": "pass",
        "component_verifier": "CCVerifier/v1",
        "candidate_order": "seeded-cyclic-octile-scan/v1",
        "labeling_features": {
            "normalized_edge_span": 0.1,
            "window_neighbor_overlap": 0.2,
            "sample_size": 1024,
            "sample_policy": "sqrt-clamped-1024-8192/v1",
        },
        "minimum_reachability_fraction":
            ADAPTIVE_SOURCE_MIN_REACHABILITY,
        "graph": graph["name"],
        "nodes": graph["nodes"],
        "undirected_edges": graph["undirected_edges"],
        "largest_component_size": graph["nodes"],
        "largest_component_min_original": 0,
        "second_largest_component_size": 1,
        "sources": [
            {
                "source_index": index,
                "source_id": index,
                "source_internal": index + 10,
                "requested_octile": index,
                "realized_octile": index,
                "source_out_degree": 1,
                "octile_start":
                    index * graph["nodes"] // ADAPTIVE_SOURCE_COUNT,
                "octile_end":
                    (index + 1) * graph["nodes"]
                    // ADAPTIVE_SOURCE_COUNT,
                "rank":
                    index * graph["nodes"] // ADAPTIVE_SOURCE_COUNT,
                "reachable_vertices": graph["nodes"],
                "reachable_fraction": 1.0,
                "replacement_path": ["primary"],
            }
            for index in range(ADAPTIVE_SOURCE_COUNT)
        ],
    }
    _validate_source_manifest(payload, graph)


def test_runtime_environment_surface_and_unknown_variant_fail_closed(
    monkeypatch,
):
    _validate_runtime_environment_surface()
    monkeypatch.setenv("RABBIT_RESOLUTION", "9")
    assert "RABBIT_RESOLUTION" in _forbidden_ambient_timing_environment()
    with pytest.raises(SourceContractError, match="unknown-variant"):
        _validate_pilot_realized_order(
            "8:csr",
            "",
            "Warning: unknown Rabbit variant",
        )


def test_realized_fallback_and_success_postconditions_fail_closed():
    effective = {
        "schema": "graphbrew_config/v1",
        **vldb_runner._expected_graphbrew_config(
            H4_DENDROGRAM_ANCHOR),
    }
    realized = {
        "schema": "graphbrew_realized/v1",
        "algorithm": "rabbit",
        "aggregation": "rabbit-incremental",
        "ordering": "compose",
        "super_graph": "none",
        "community_order": "identity",
        "intra_community_order": "bfs",
        "refinement_pass": "none",
        "resolution": None,
        "recursive_depth": None,
        "schedule_sensitive": True,
        "final_algo_id": -1,
        "sub_algo_id": 8,
        "num_passes": 1,
        "num_communities": 12,
        "fallbacks": [{
            "reason": "rabbit-dendrogram-unavailable",
            "requested": "dendrogram",
            "realized": "bfs",
        }],
        "block_algorithms": {},
    }
    output = "\n".join([
        "GraphBrew Effective Config: "
        + json.dumps(effective, separators=(",", ":")),
        "GraphBrew Realized Config: "
        + json.dumps(realized, separators=(",", ":")),
    ])
    with pytest.raises(SourceContractError, match="runtime fallback"):
        _validate_pilot_realized_order(
            H4_DENDROGRAM_ANCHOR, output, "")

    with pytest.raises(SourceContractError, match="without timing"):
        _parse_and_validate_pilot_output(
            {
                "phase": "randomized-pilot",
                "order_spec": "0",
                "source_policy_id": None,
            },
            "Average Time: 0.0",
            "",
            error_kind="",
            censored=False,
        )
    with pytest.raises(SourceContractError, match="peak RSS"):
        _parse_and_validate_pilot_output(
            {
                "phase": "rss-pilot",
                "order_spec": "0",
                "source_policy_id": None,
                "rss_from_stderr": True,
            },
            "Trial Time: 0.1\nAverage Time: 0.1",
            "",
            error_kind="",
            censored=False,
        )
    with pytest.raises(SourceContractError, match="contract Tier-0"):
        _parse_and_validate_pilot_output(
            {
                "phase": "feature-cost-pilot",
                "order_spec": "14",
                "source_policy_id": None,
            },
            "Trial Time: 0.1\nAverage Time: 0.1",
            "",
            error_kind="",
            censored=False,
        )


def test_literal_pilot_consumer_honors_command_and_paths(
    tmp_path, monkeypatch,
):
    for name in list(os.environ):
        if name.startswith((
            "ADAPTIVE_", "CACHE_", "ECG_", "GORDER_",
            "GRAPHBREW_", "MODEL_TREE_", "PERCEPTRON_", "RABBIT_",
        )):
            monkeypatch.delenv(name, raising=False)
    graph_path = tmp_path / "fixture.sg"
    graph_path.write_text("fixture")
    python = Path(shutil.which("python3"))
    command = {
        "command_id": "fixture",
        "idempotency_key": "fixture|a0",
        "phase": "randomized-pilot",
        "graph": "fixture",
        "graph_path": str(graph_path),
        "graph_output_bytes": graph_path.stat().st_size,
        "graph_mtime_ns": graph_path.stat().st_mtime_ns,
        "graph_crc32": file_crc32(graph_path),
        "binary_provenance": [{
            "path": str(python),
            "bytes": python.stat().st_size,
            "mtime_ns": python.stat().st_mtime_ns,
            "sha256": file_sha256(python),
        }],
        "labeling": "randomized",
        "kernel": "pr",
        "arm": "14:perceptron-contract-original",
        "order_spec": "14",
        "process_id": 0,
        "measurement_mode": "cold-process",
        "attempt": 0,
        "retry_attempts": [0, 1],
        "command": [
            "python3",
            "-c",
            "print('Trial Time: 0.1\\nAverage Time: 0.1')",
        ],
        "environment": {"GRAPHBREW_TEST_VALUE": "1"},
        "environment_mode": "inherit-then-override",
        "timeout_seconds": 10,
        "timeout_interpretation": "right-censored-lower-bound",
        "source_policy_id": None,
        "source_repeats": 1,
        "expected_sources": None,
        "expected_source_internals": None,
        "expected_source_out_degrees": None,
        "rss_from_stderr": False,
        "stdout_path": str(tmp_path / "stdout.log"),
        "stderr_path": str(tmp_path / "stderr.log"),
        "result_path": str(tmp_path / "result.json"),
    }
    command = adaptive_runner._bind_authorized_command(
        command, "test-authorization", "test-manifest", {})
    result = _run_literal_pilot_command(
        command,
        {"cpu_governors": ["performance"]},
    )
    assert result["error_kind"] == ""
    assert result["average_time"] == 0.1
    assert (tmp_path / "stdout.log").is_file()
    assert _run_literal_pilot_command(
        command,
        {"cpu_governors": ["performance"]},
    ) == result
    unbound = {
        key: value
        for key, value in command.items()
        if key not in {
            "authorization_reference",
            "execution_manifest_sha256",
            "input_artifacts",
            "command_contract_sha256",
        }
    }
    rebound = adaptive_runner._bind_authorized_command(
        unbound, "different-authorization", "test-manifest", {})
    with pytest.raises(RuntimeError, match="command contract changed"):
        _run_literal_pilot_command(
            rebound,
            {"cpu_governors": ["performance"]},
        )


def test_literal_pilot_consumer_accepts_priming_shape(
    tmp_path, monkeypatch,
):
    graph_path = tmp_path / "fixture.sg"
    graph_path.write_text("fixture")
    python = Path(shutil.which("python3"))
    command = {
        "command_id": "prime",
        "idempotency_key": "prime|a0",
        "attempt": 0,
        "retry_attempts": [0, 1],
        "phase": "page-cache-prime",
        "graph": "fixture",
        "graph_path": str(graph_path),
        "graph_output_bytes": graph_path.stat().st_size,
        "graph_mtime_ns": graph_path.stat().st_mtime_ns,
        "graph_crc32": file_crc32(graph_path),
        "binary_provenance": [{
            "path": str(python),
            "bytes": python.stat().st_size,
            "mtime_ns": python.stat().st_mtime_ns,
            "sha256": file_sha256(python),
        }],
        "labeling": "randomized",
        "command": ["python3", "-c", "print('primed')"],
        "environment": {},
        "environment_mode": "inherit-then-override",
        "timeout_seconds": 10,
        "timeout_interpretation": "hard-preparation-cap",
        "stdout_path": str(tmp_path / "prime.stdout"),
        "stderr_path": str(tmp_path / "prime.stderr"),
        "result_path": str(tmp_path / "prime.result.json"),
    }
    command = adaptive_runner._bind_authorized_command(
        command, "test-authorization", "test-manifest", {})
    result = _run_literal_pilot_command(
        command,
        {"cpu_governors": ["performance"]},
    )
    assert result["phase"] == "page-cache-prime"
    assert result["error_kind"] == ""
    assert (tmp_path / "prime.result.json").is_file()


def test_pilot_retry_paths_and_serial_lock(tmp_path):
    command = {
        "command_id": "retry",
        "attempt": 0,
        "retry_attempts": [0, 1],
        "idempotency_key": "retry|a0",
        "environment": {},
        "stdout_path": str(
            tmp_path / "retry" / "attempt_0" / "stdout.log"),
        "stderr_path": str(
            tmp_path / "retry" / "attempt_0" / "stderr.log"),
        "result_path": str(
            tmp_path / "retry" / "attempt_0" / "result.json"),
    }
    retry = _pilot_command_for_attempt(command, 1)
    assert retry["idempotency_key"] == "retry|a1"
    assert "attempt_1" in retry["result_path"]
    priming = _priming_command_for_session(command, "session")
    assert "sessions/session" in priming["result_path"]
    attempts = []

    def fake_runner(attempt_command, _host):
        attempts.append(attempt_command["attempt"])
        return {
            "error_kind": (
                "process-failure"
                if attempt_command["attempt"] == 0 else ""
            )
        }

    result = _run_pilot_command_with_retries(
        command,
        {},
        "approved",
        "manifest-sha",
        {},
        runner=fake_runner,
    )
    assert result["error_kind"] == ""
    assert attempts == [0, 1]
    lock_path = tmp_path / "pilot.lock"
    with _PilotExecutionLock(lock_path):
        with pytest.raises(RuntimeError, match="serial run lock"):
            with _PilotExecutionLock(lock_path):
                pass


def test_execution_authorization_binds_full_manifest(
    tmp_path, monkeypatch,
):
    artifact_root = tmp_path / "artifacts"
    sprint_root = artifact_root / "adaptive_selector" / "sprint1"
    sprint_root.mkdir(parents=True)
    manifest_path = sprint_root / "pilot_execution_manifest.json"
    manifest = {
        "schema": "adaptive-pilot-execution/v2",
        "dry_run_only": True,
        "command_count": 1,
        "priming_command_count": 0,
        "execution_order": ["priming_commands", "commands"],
        "concurrency": "serial-exclusive",
        "execution_authorization_required": True,
        "priming_commands": [],
        "commands": [{"command_id": "one", "value": 1}],
    }
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        adaptive_runner,
        "_validate_execution_manifest",
        lambda payload: {},
    )

    authorization_path = (
        adaptive_runner.create_sprint1_execution_authorization(
            artifact_root,
            authorization_reference="reviewed-characterization-v1",
        )
    )
    authorization = json.loads(authorization_path.read_text())
    assert authorization["schema"] == (
        "adaptive-pilot-execution-authorization/v2"
    )
    assert authorization["execution_manifest_sha256"] == (
        adaptive_runner._canonical_json_sha256(manifest)
    )
    assert authorization["authorized_manifest_contract"] == (
        adaptive_runner._authorization_manifest_contract(manifest)
    )

    manifest["commands"][0]["value"] = 2
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="authorization changed"):
        adaptive_runner.create_sprint1_execution_authorization(
            artifact_root,
            authorization_reference="reviewed-characterization-v1",
        )


def test_execution_input_artifacts_are_built_in_prepare_scope(tmp_path):
    paths = []
    for name in ("budget", "source", "natural", "weights"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"name": name}))
        paths.append(path)
    bindings = adaptive_runner._execution_input_artifacts(*paths)
    assert set(bindings) == {
        "budget_projection",
        "source_manifest",
        "natural_manifest",
        "contract_weights",
    }
    for binding in bindings.values():
        adaptive_runner._validate_artifact_binding(binding)
    prepare_source = inspect.getsource(
        adaptive_runner.prepare_sprint1_pilot_execution)
    assert "_execution_input_artifacts(" in prepare_source


def test_source_bundle_cannot_retrofit_missing_graph_provenance():
    with pytest.raises(ValueError, match="provenance binding"):
        adaptive_runner._validate_source_bundle_graph_provenance({
            "graphs": {},
        })


def test_budget_projection_covers_frozen_matrix(tmp_path):
    graph_root = tmp_path / "graphs"
    artifact_root = tmp_path / "artifacts"
    paper_root = artifact_root / "vldb_paper"
    for graph in EVAL_GRAPHS:
        graph_dir = graph_root / graph["name"]
        graph_dir.mkdir(parents=True)
        graph_path = graph_dir / f"{graph['name']}.sg"
        graph_path.write_bytes(b"fixture")
        (graph_dir / f"{graph['name']}.sg.meta.json").write_text(
            json.dumps({
                "schema": "graph_source/v2",
                "reorder_semantics_version":
                    REORDER_SEMANTICS_VERSION,
                "graph": graph["name"],
                "output_path": str(graph_path.resolve()),
                "output_bytes": graph_path.stat().st_size,
                "output_crc32": file_crc32(graph_path),
                "converter_sha256": file_sha256(
                    adaptive_runner.PROJECT_ROOT
                    / "bench" / "bin" / "converter"
                ),
                "conversion_repository_state":
                    adaptive_runner._conversion_repository_state(),
                "expected_nodes": graph["nodes"],
                "expected_undirected_edges":
                    graph["undirected_edges"],
                "nodes": graph["nodes"],
                "undirected_edges": graph["undirected_edges"],
            })
        )

    exp2 = []
    kernel_evidence_arms = sorted({
        _kernel_evidence_arm(arm)
        for arm in ALL_TIMING_ARMS
    })
    for graph in EVAL_GRAPHS:
        for arm in kernel_evidence_arms:
            for benchmark in BENCHMARKS:
                exp2.append({
                    "graph": graph["name"],
                    "algo_id": arm,
                    "benchmark": benchmark,
                    "read_time": 1.0,
                    "trial_times": [1.0, 1.1],
                })
    exp3 = []
    mapping_evidence_arms = sorted({
        _mapping_evidence_arm(arm)
        for arm in ALL_TIMING_ARMS
        if arm != "0"
    })
    for graph in EVAL_GRAPHS:
        for arm in mapping_evidence_arms:
            exp3.append({
                "graph": graph["name"],
                "algo_id": arm,
                "reorder_time": 2.0,
            })
    exp3 = [
        row for row in exp3
        if not (
            row["graph"] == EVAL_GRAPHS[0]["name"]
            and row["algo_id"] == "5"
        )
    ]
    exp3.extend([
        {
            "graph": EVAL_GRAPHS[0]["name"],
            "algo_id": "5",
            "reorder_time": 1.0,
            "timing_source": "stage02-sidecar",
        },
        {
            "graph": EVAL_GRAPHS[0]["name"],
            "algo_id": "5",
            "reorder_time": 3.0,
            "timing_source": "live-final",
        },
    ])
    cache = [
        {
            "graph": EVAL_GRAPHS[0]["name"],
            "algo_key": arm,
            "benchmark": "pr",
            "average_time": 10.0,
            "directed_edges_processed": [100_000_000],
        }
        for arm in ("0", "5", "8:csr")
    ]
    for relative, payload in (
        ("exp2_speedup/speedup_results.json", exp2),
        ("exp3_overhead/overhead_results.json", exp3),
        ("exp1_cache/cache_results.json", cache),
    ):
        path = paper_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))

    projection = build_sprint1_budget_projection(
        artifact_root,
        graph_root,
    )
    assert len(projection["kernel_rows"]) == (
        len(EVAL_GRAPHS) * len(ALL_TIMING_ARMS) * len(BENCHMARKS)
    )
    assert 0 < len(projection["cache_rows"]) <= (
        len(EVAL_GRAPHS)
        * len(ALL_TIMING_ARMS)
        * len(CACHE_KERNELS)
    )
    assert projection["pilot_graphs"] == [
        "twitter7",
        "webbase-2001",
        "hollywood-2009",
    ]
    assert projection["projected_buffered_node_hours_low"] > 0
    assert (
        projection["projected_buffered_node_hours_high"]
        >= projection["projected_buffered_node_hours_low"]
    )
    assert projection["collection_allowed"]["three_graph_pilot"]
    assert projection["collection_allowed"]["cache_micro_pilot"]
    assert not projection["collection_allowed"]["hardware_validation"]
    assert not projection["collection_allowed"][
        "randomized_kernel_corpus"
    ]
    assert not projection["collection_allowed"][
        "randomized_cache_corpus"
    ]
    assert len(projection["randomized_pilot_rows"]) == (
        3 * len(ALL_TIMING_ARMS) * len(BENCHMARKS)
    )
    assert len(projection["cache_micro_pilot_rows"]) == 9
    assert len(projection["feature_pilot_rows"]) == 4
    assert len(projection["natural_pilot_rows"]) == (
        len(ALL_TIMING_ARMS) * len(BENCHMARKS)
    )
    assert len(projection["materialization_rows"]) == 1
    assert len(projection["rss_rows"]) == 2 * len(ALL_TIMING_ARMS)
    for graph_name in ("twitter7", "webbase-2001"):
        assert {
            row["arm"]
            for row in projection["rss_rows"]
            if row["graph"] == graph_name
        } == set(ALL_TIMING_ARMS)
    assert projection["policy"]["natural_pilot_graphs"] == [
        "hollywood-2009"
    ]
    bfs_probes = [
        row for row in projection["cache_micro_pilot_rows"]
        if row["kernel"] == "bfs"
    ]
    assert {row["source_index"] for row in bfs_probes} == {0, 7}
    assert all(
        row["wall_clock_cap_multiplier"] == 10.0
        for row in bfs_probes
    )
    assert any(
        row["probe_role"] == "graph-kernel-interaction"
        for row in bfs_probes
    )
    assert all(
        row["tier0_trained"] is False
        for row in projection["feature_pilot_rows"]
    )
    assert projection[
        "pilot_projection_if_all_defined_caps_bind"
    ] >= (
        projection["pilot_buffered_node_hours_high"]
    )
    assert projection["program_total_if_full_after_pilot_high"] >= (
        projection["projected_buffered_node_hours_high"]
    )
    assert all(
        row["authorized_for_collection"]
        for phase in projection["authorized_phases"]
        for row in (
            projection["randomized_pilot_rows"]
            + projection["cache_micro_pilot_rows"]
            + projection["feature_pilot_rows"]
            + projection["natural_pilot_rows"]
            + projection["materialization_rows"]
            + projection["rss_rows"]
        )
        if row["phase"] == phase
    )
    assert not any(
        row["authorized_for_collection"]
        for row in projection["kernel_rows"]
    )
    authorized_rows = [
        row
        for rows in (
            projection["randomized_pilot_rows"],
            projection["natural_pilot_rows"],
            projection["materialization_rows"],
            projection["rss_rows"],
            projection["cache_micro_pilot_rows"],
            projection["feature_pilot_rows"],
        )
        for row in rows
    ]
    expected_command_ids, expected_priming_ids = (
        adaptive_runner._expected_pilot_command_ids(projection)
    )
    assert len(expected_priming_ids) == 4
    assert len(expected_command_ids) == (
        4 * len(ALL_TIMING_ARMS) * len(BENCHMARKS)
        * adaptive_runner.PILOT_PROCESS_BLOCKS
        + 2 * len(ALL_TIMING_ARMS)
        + 9
        + 4
    )
    assert all(not row["claim_eligible"] for row in authorized_rows)
    assert all(row["pilot_only"] for row in authorized_rows)
    expected_defined_caps = sum(
        float(row["wall_clock_cap_seconds"])
        * (
            1
            if row["phase"] == "natural-label-materialization"
            else len(adaptive_runner.PILOT_RETRY_ATTEMPTS)
        )
        / 3600.0
        for row in authorized_rows
    )
    assert projection[
        "pilot_projection_if_all_defined_caps_bind"
    ] == pytest.approx(expected_defined_caps)
    for row in (
        projection["randomized_pilot_rows"]
        + projection["natural_pilot_rows"]
    ):
        assert row["wall_clock_cap_seconds"] >= (
            4 * row["cap_floor_raw_seconds"]
        )

    write_sprint1_budget_projection(projection, artifact_root)
    write_sprint1_budget_projection(projection, artifact_root)
    changed = copy.deepcopy(projection)
    changed["collection_allowed"]["randomized_kernel_corpus"] = True
    with pytest.raises(RuntimeError, match="authorization changed"):
        write_sprint1_budget_projection(changed, artifact_root)
    inflated = build_sprint1_budget_projection(
        artifact_root,
        graph_root,
        budget_hours=10_000,
    )
    assert not inflated["collection_allowed"][
        "randomized_kernel_corpus"
    ]
    assert not inflated["collection_allowed"][
        "randomized_cache_corpus"
    ]
    assert not inflated["collection_allowed"][
        "hardware_validation"
    ]
    assert not inflated["collection_allowed"][
        "natural_label_extension"
    ]
    assert inflated["status"] == "pilot-approved-repricing-required"
    first_dbg = next(
        row for row in projection["kernel_rows"]
        if row["graph"] == EVAL_GRAPHS[0]["name"]
        and row["arm"] == "5"
    )
    assert first_dbg["map_seconds_per_block"] == 3.0
