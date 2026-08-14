"""Adaptive Sprint-1 pilot analysis contracts."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.lib.analysis.adaptive_pilot import (
    build_pilot_analysis,
    summarize_cache_repricing,
    summarize_feature_overhead,
    summarize_peak_rss,
    write_pilot_analysis,
)
from scripts.lib.pipeline.adaptive_pilot_contract import (
    bind_authorized_command,
    canonical_json_sha256,
    pilot_command_for_attempt,
    priming_command_for_session,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _base_command(
    sprint_root: Path,
    *,
    command_id: str,
    graph: str,
    arm: str | None,
    process_id: int,
    phase: str = "randomized-pilot",
) -> dict:
    output = (
        sprint_root / "pilot_runs"
        / command_id.replace("|", "__")
        / "attempt_0"
    )
    return {
        "command_id": command_id,
        "idempotency_key": f"{command_id}|a0",
        "attempt": 0,
        "retry_attempts": [0, 1],
        "phase": phase,
        "graph": graph,
        "graph_path": f"/graphs/{graph}.sg",
        "labeling": "randomized",
        "kernel": "pr" if phase != "page-cache-prime" else None,
        "arm": arm,
        "order_spec": arm,
        "process_id": process_id,
        "measurement_mode": (
            "cold-process"
            if phase != "page-cache-prime" else "page-cache-prime"
        ),
        "command": ["fixture", graph, str(arm)],
        "environment": {},
        "environment_mode": "inherit-then-override",
        "timeout_seconds": 10,
        "timeout_interpretation": "right-censored-lower-bound",
        "stdout_path": str(output / "stdout.log"),
        "stderr_path": str(output / "stderr.log"),
        "result_path": str(output / "result.json"),
    }


def _result(
    command: dict,
    *,
    kernel_seconds: float = 0.0,
    reorder_seconds: float = 0.0,
    error_kind: str = "",
) -> dict:
    return {
        "schema": "adaptive-pilot-result/v2",
        "idempotency_key": command["idempotency_key"],
        "command_id": command["command_id"],
        "phase": command["phase"],
        "graph": command["graph"],
        "graph_path": command["graph_path"],
        "labeling": command.get("labeling"),
        "kernel": command.get("kernel"),
        "arm": command.get("arm"),
        "order_spec": command.get("order_spec"),
        "process_id": command.get("process_id"),
        "measurement_mode": command.get("measurement_mode"),
        "attempt": command.get("attempt", 0),
        "command": command["command"],
        "environment": command["environment"],
        "environment_mode": command["environment_mode"],
        "timeout_seconds": command["timeout_seconds"],
        "timeout_interpretation": command["timeout_interpretation"],
        "duration_seconds": 1.0,
        "returncode": 1 if error_kind == "process-failure" else 0,
        "error_kind": error_kind,
        "contract_violation": None,
        "censored": error_kind == "timeout",
        "claim_eligible": False,
        "pilot_only": True,
        "average_time": kernel_seconds,
        "reorder_time": reorder_seconds,
        "extra": (
            {}
            if command["phase"] == "page-cache-prime"
            or error_kind
            else {
                "complete_reorder_time": reorder_seconds,
                "trial_times": [kernel_seconds],
                "composed_mapping_fingerprint":
                    f"{command['graph']}-{command.get('arm')}-"
                    f"{command.get('process_id')}",
            }
        ),
        "host_state": {},
        "authorization_reference":
            command["authorization_reference"],
        "execution_manifest_sha256":
            command["execution_manifest_sha256"],
        "command_contract_sha256":
            command["command_contract_sha256"],
    }


def _fixture(
    tmp_path: Path,
    *,
    complete: bool = True,
) -> Path:
    sprint_root = tmp_path / "sprint1"
    commands = []
    costs = {
        ("g1", "0"): (1.0, 0.0),
        ("g1", "5"): (1.8, 4.0),
        ("g2", "0"): (2.0, 0.0),
        ("g2", "5"): (0.5, 4.0),
    }
    for graph in ("g1", "g2"):
        for arm in ("0", "5"):
            for process_id in range(3):
                commands.append(_base_command(
                    sprint_root,
                    command_id=(
                        f"randomized-pilot|{graph}|pr|{arm}|p{process_id}"
                    ),
                    graph=graph,
                    arm=arm,
                    process_id=process_id,
                ))
    priming = _base_command(
        sprint_root,
        command_id="page-cache-prime|g1|randomized",
        graph="g1",
        arm=None,
        process_id=0,
        phase="page-cache-prime",
    )
    budget_path = sprint_root / "budget_projection.json"
    _write_json(budget_path, {
        "policy": {"deployable_pilot_arms": ["0", "5"]},
    })
    manifest = {
        "schema": "adaptive-pilot-execution/v2",
        "budget_projection": str(budget_path),
        "input_artifacts": {},
        "command_count": len(commands),
        "priming_command_count": 1,
        "commands": commands,
        "priming_commands": [priming],
    }
    manifest_sha256 = canonical_json_sha256(manifest)
    authorization_reference = "fixture-authorization"
    _write_json(
        sprint_root / "pilot_execution_manifest.json",
        manifest,
    )
    _write_json(
        sprint_root / "pilot_execution_authorization.json",
        {
            "schema": "adaptive-pilot-execution-authorization/v2",
            "execution_enabled": True,
            "authorization_reference": authorization_reference,
            "command_count": len(commands),
            "execution_manifest_sha256": manifest_sha256,
        },
    )

    selected_results = []
    all_attempts = []
    for base in commands:
        kernel_seconds, reorder_seconds = costs[(
            base["graph"], base["arm"])]
        if (
            base["graph"] == "g2"
            and base["arm"] == "5"
            and base["process_id"] == 0
        ):
            failed_command = bind_authorized_command(
                pilot_command_for_attempt(base, 0),
                authorization_reference,
                manifest_sha256,
                {},
            )
            failed = _result(
                failed_command, error_kind="process-failure")
            _write_json(
                Path(failed_command["result_path"]), failed)
            all_attempts.append(failed)
            attempt = 1
        else:
            attempt = 0
        command = bind_authorized_command(
            pilot_command_for_attempt(base, attempt),
            authorization_reference,
            manifest_sha256,
            {},
        )
        result = _result(
            command,
            kernel_seconds=kernel_seconds,
            reorder_seconds=reorder_seconds,
        )
        _write_json(Path(command["result_path"]), result)
        selected_results.append(result)
        all_attempts.append(result)

    session_id = "fixture-session"
    priming_command = bind_authorized_command(
        priming_command_for_session(priming, session_id),
        authorization_reference,
        manifest_sha256,
        {},
    )
    priming_result = _result(priming_command)
    _write_json(Path(priming_command["result_path"]), priming_result)
    if complete:
        _write_json(
            sprint_root / "pilot_execution_complete.json",
            {
                "schema": "adaptive-pilot-execution-complete/v2",
                "status": "complete",
                "authorization_reference": authorization_reference,
                "execution_session_id": session_id,
                "execution_manifest_sha256": manifest_sha256,
                "command_count": len(commands),
                "priming_command_count": 1,
                "result_count": len(commands) + 1,
                "result_states": {
                    "success": len(commands) + 1,
                },
                "retried_result_count": 1,
            },
        )
    return sprint_root


def test_pilot_analysis_replays_retries_and_crossfits_headroom(tmp_path):
    sprint_root = _fixture(tmp_path)
    analysis = build_pilot_analysis(sprint_root)
    assert analysis["status"] == "complete-clean"
    assert analysis["headroom_eligible"] is True
    assert analysis["selected_result_count"] == 12
    assert analysis["priming_result_count"] == 1
    assert analysis["validated_attempt_count"] == 13
    assert analysis["total_consumed_hours"] > (
        analysis["terminal_command_hours"])
    reuse_1 = analysis["policy_headroom"]["by_reuse"]["1"]
    assert reuse_1["crossfit_oracle_arm_diversity"] == 1
    reuse_10 = analysis["policy_headroom"]["by_reuse"]["10"]
    assert reuse_10["crossfit_oracle_arm_diversity"] == 2
    assert (
        reuse_10["logo_best_static"]["geomean_regret_ratio"] > 1
    )
    assert "infinity" in analysis["policy_headroom"]["by_reuse"]
    original_cell = next(
        cell for cell in analysis["timing_cells"]
        if cell["graph"] == "g1" and cell["arm"] == "0"
    )
    assert all(
        sample["modeled_complete_reorder_seconds"] == 0
        for sample in original_cell["process_samples"]
    )
    output = write_pilot_analysis(sprint_root)
    assert output.is_file()


def test_pilot_analysis_rejects_incomplete_execution(tmp_path):
    sprint_root = _fixture(tmp_path, complete=False)
    with pytest.raises(RuntimeError, match="incomplete"):
        build_pilot_analysis(sprint_root)
    analysis = build_pilot_analysis(
        sprint_root, require_complete=False)
    assert analysis["status"] == "incomplete"
    assert analysis["headroom_eligible"] is False


def test_pilot_analysis_rejects_authorization_mismatch(tmp_path):
    sprint_root = _fixture(tmp_path)
    authorization_path = (
        sprint_root / "pilot_execution_authorization.json")
    authorization = json.loads(authorization_path.read_text())
    authorization["execution_manifest_sha256"] = "changed"
    _write_json(authorization_path, authorization)
    with pytest.raises(ValueError, match="authorization"):
        build_pilot_analysis(sprint_root)


def test_pilot_analysis_rejects_result_digest_mismatch(tmp_path):
    sprint_root = _fixture(tmp_path)
    result_path = next(
        (sprint_root / "pilot_runs").glob(
            "randomized-pilot*/attempt_0/result.json")
    )
    result = json.loads(result_path.read_text())
    result["command_contract_sha256"] = "changed"
    _write_json(result_path, result)
    with pytest.raises(ValueError, match="digest"):
        build_pilot_analysis(sprint_root)


def test_censoring_suppresses_headroom(tmp_path):
    sprint_root = _fixture(tmp_path)
    result_path = next(
        (sprint_root / "pilot_runs").glob(
            "randomized-pilot__g1__pr__0__p0/attempt_0/result.json")
    )
    result = json.loads(result_path.read_text())
    result["error_kind"] = "timeout"
    result["censored"] = True
    result["extra"] = {}
    _write_json(result_path, result)
    completion_path = sprint_root / "pilot_execution_complete.json"
    completion = json.loads(completion_path.read_text())
    completion["result_states"] = {"success": 12, "timeout": 1}
    _write_json(completion_path, completion)
    analysis = build_pilot_analysis(sprint_root)
    assert analysis["status"] == "complete-with-censoring"
    assert analysis["headroom_eligible"] is False
    assert analysis["policy_headroom"]["headroom_eligible"] is False


def test_feature_rss_and_cache_pilot_gates():
    original_cell = {
        "graph": "g",
        "labeling": "randomized",
        "kernel": "pr",
        "arm": "0",
        "eligible": True,
        "process_samples": [{
            "process_id": process_id,
            "state": "success",
            "censored": False,
            "kernel_seconds": 10.0,
            "modeled_complete_reorder_seconds": 0.0,
        } for process_id in range(3)],
    }
    feature = summarize_feature_overhead(
        [original_cell], {("g", "randomized"): 0.1})
    assert feature["eligible"]
    assert feature["all_contexts_within_2pct"]

    rss_commands = [{
        "command_id": f"rss|{arm}",
        "phase": "rss-pilot",
    } for arm in ("0", "5")]
    rss_results = [{
        "command_id": f"rss|{arm}",
        "phase": "rss-pilot",
        "graph": "g",
        "arm": arm,
        "error_kind": "",
        "censored": False,
        "extra": {"peak_rss_kib": peak},
    } for arm, peak in (("0", 100), ("5", 120))]
    rss = summarize_peak_rss(
        rss_results, {"commands": rss_commands})
    assert rss["eligible"]
    assert rss["max_peak_rss_kib"] == 120

    probes = (
        ("twitter", "pr", "8:csr", 22, None, "graph-factor", 20.0),
        ("webbase", "pr", "8:csr", 22, None, "graph-factor", 30.0),
        ("hollywood", "pr", "8:csr", 22, None, "star-center", 10.0),
        ("hollywood", "bfs", "8:csr", 22, 0, "kernel-factor", 40.0),
        ("hollywood", "pr", "0", 22, None, "arm-factor", 5.0),
        ("hollywood", "pr", "5", 22, None, "arm-factor", 8.0),
        ("hollywood", "pr", "8:csr", 512, None, "capacity-factor", 20.0),
        (
            "twitter", "bfs", "8:csr", 22, 0,
            "graph-kernel-interaction", 80.0,
        ),
        (
            "hollywood", "bfs", "8:csr", 22, 7,
            "source-dispersion", 44.0,
        ),
    )
    budget_rows = [{
        "graph": graph,
        "kernel": kernel,
        "arm": arm,
        "capacities_mib": capacity,
        "source_index": source,
        "probe_role": role,
    } for graph, kernel, arm, capacity, source, role, _ in probes]
    cache_results = []
    for graph, kernel, arm, capacity, source, _, duration in probes:
        cache_results.append({
            "phase": "cache-micro-pilot",
            "command_id": (
                f"cache|{graph}|{kernel}|{arm}|{capacity}MiB|s{source}"
            ),
            "graph": graph,
            "kernel": kernel,
            "arm": arm,
            "measurement_mode": f"cache-simulator-{capacity}mib",
            "error_kind": "",
            "censored": False,
            "duration_seconds": duration,
            "extra": {
                "cache_stats": {
                    "total_accesses": 100,
                    "L1": {"misses": 50},
                    "L2": {"misses": 25},
                    "L3": {"misses": 10},
                },
            },
        })
    cache = summarize_cache_repricing(cache_results, {
        "cache_micro_pilot_rows": budget_rows,
        "policy": {
            "cache_repricing_max_interaction_residual": 0.25,
        },
    })
    assert cache["eligible"]
    assert cache["interaction_residual"] == pytest.approx(0.0)
    assert cache["separable_model_eligible"]


def test_top_level_exposes_adaptive_pilot_analysis():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/graphbrew_experiment.py",
            "--help",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--adaptive-sprint1-analyze" in result.stdout


def test_top_level_rejects_conflicting_adaptive_actions():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/graphbrew_experiment.py",
            "--adaptive-sprint1-executor-check",
            "--adaptive-sprint1-analyze",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "exactly one adaptive Sprint-1 action" in result.stderr
