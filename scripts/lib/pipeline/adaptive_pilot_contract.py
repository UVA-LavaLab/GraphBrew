"""Shared content-binding helpers for adaptive pilot execution and analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

PILOT_RESULT_SCHEMA = "adaptive-pilot-result/v2"


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def pilot_command_for_attempt(
    command: Mapping[str, Any],
    attempt: int,
) -> dict[str, Any]:
    if attempt not in command["retry_attempts"]:
        raise ValueError("Requested pilot retry attempt is not pre-registered")
    updated = dict(command)
    updated["environment"] = dict(command["environment"])
    base_dir = Path(command["result_path"]).parents[1]
    attempt_dir = base_dir / f"attempt_{attempt}"
    updated["attempt"] = attempt
    updated["idempotency_key"] = (
        f"{command['command_id']}|a{attempt}")
    updated["stdout_path"] = str(attempt_dir / "stdout.log")
    updated["stderr_path"] = str(attempt_dir / "stderr.log")
    updated["result_path"] = str(attempt_dir / "result.json")
    if command.get("cache_output_path"):
        cache_output_path = attempt_dir / "cache_stats.json"
        updated["cache_output_path"] = str(cache_output_path)
        updated["environment"]["CACHE_OUTPUT_JSON"] = str(
            cache_output_path)
    return updated


def priming_command_for_session(
    command: Mapping[str, Any],
    session_id: str,
) -> dict[str, Any]:
    updated = dict(command)
    base_dir = Path(command["result_path"]).parents[1]
    session_dir = base_dir / "sessions" / session_id
    updated["idempotency_key"] = (
        f"{command['command_id']}|session={session_id}")
    updated["stdout_path"] = str(session_dir / "stdout.log")
    updated["stderr_path"] = str(session_dir / "stderr.log")
    updated["result_path"] = str(session_dir / "result.json")
    return updated


def bind_authorized_command(
    command: Mapping[str, Any],
    authorization_reference: str,
    execution_manifest_sha256: str,
    input_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    bound = dict(command)
    bound["authorization_reference"] = authorization_reference
    bound["execution_manifest_sha256"] = execution_manifest_sha256
    bound["input_artifacts"] = dict(input_artifacts)
    bound["command_contract_sha256"] = canonical_json_sha256(bound)
    return bound


def validate_result_contract(
    result: Mapping[str, Any],
    command: Mapping[str, Any],
) -> None:
    if result.get("schema") != PILOT_RESULT_SCHEMA:
        raise ValueError("Unsupported adaptive pilot result")
    contract_payload = dict(command)
    recorded_contract = contract_payload.pop(
        "command_contract_sha256", None)
    if (
        not recorded_contract
        or canonical_json_sha256(contract_payload) != recorded_contract
        or result.get("command_contract_sha256") != recorded_contract
    ):
        raise ValueError("Adaptive pilot command digest changed")
    for result_field, command_field in (
        ("idempotency_key", "idempotency_key"),
        ("command_id", "command_id"),
        ("phase", "phase"),
        ("graph", "graph"),
        ("graph_path", "graph_path"),
        ("labeling", "labeling"),
        ("kernel", "kernel"),
        ("arm", "arm"),
        ("order_spec", "order_spec"),
        ("process_id", "process_id"),
        ("measurement_mode", "measurement_mode"),
        ("attempt", "attempt"),
        ("command", "command"),
        ("environment", "environment"),
        ("environment_mode", "environment_mode"),
        ("timeout_seconds", "timeout_seconds"),
        ("timeout_interpretation", "timeout_interpretation"),
        ("authorization_reference", "authorization_reference"),
        ("execution_manifest_sha256", "execution_manifest_sha256"),
    ):
        if result.get(result_field) != command.get(command_field):
            raise ValueError(
                "Adaptive pilot result changed "
                f"{command['command_id']}/{result_field}")
    if (
        result.get("claim_eligible") is not False
        or result.get("pilot_only") is not True
    ):
        raise ValueError("Adaptive pilot result eligibility changed")
