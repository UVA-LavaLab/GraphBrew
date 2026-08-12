"""Versioned reorder-time sidecars."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from scripts.lib.core.experiment_policy import (
    REORDER_SEMANTICS_VERSION,
)


REORDER_TIME_SCHEMA = "reorder-time/v2"


def metadata_path(path: str | os.PathLike) -> Path:
    candidate = Path(path)
    if candidate.name.endswith(".time.json"):
        return candidate
    if candidate.suffix == ".time":
        return candidate.with_suffix(".time.json")
    return candidate.with_name(candidate.name + ".time.json")


def write_reorder_time(
    path: str | os.PathLike,
    *,
    complete_reorder_time: float,
    mapping_fingerprint: str,
    algorithm_spec: str,
) -> Path:
    if complete_reorder_time < 0:
        raise ValueError("Complete reorder time must be non-negative")
    if not mapping_fingerprint:
        raise ValueError("Mapping fingerprint is required")
    if not algorithm_spec:
        raise ValueError("Resolved algorithm spec is required")
    target = metadata_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": REORDER_TIME_SCHEMA,
        "reorder_semantics_version": REORDER_SEMANTICS_VERSION,
        "timing_boundary": "core+validation+apply",
        "complete_reorder_time": float(complete_reorder_time),
        "mapping_fingerprint": mapping_fingerprint,
        "algorithm_spec": algorithm_spec,
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".tmp",
        dir=target.parent,
        delete=False,
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, target)
    return target


def read_reorder_time(
    path: str | os.PathLike,
    *,
    expected_mapping_fingerprint: str | None = None,
    allow_legacy: bool = False,
) -> float | None:
    target = metadata_path(path)
    if target.is_file():
        payload = json.loads(target.read_text())
        if payload.get("schema") != REORDER_TIME_SCHEMA:
            raise ValueError(f"Unsupported reorder-time schema: {target}")
        if (
            payload.get("reorder_semantics_version")
            != REORDER_SEMANTICS_VERSION
        ):
            raise ValueError(f"Stale reorder semantics: {target}")
        if payload.get("timing_boundary") != "core+validation+apply":
            raise ValueError(f"Unknown reorder timing boundary: {target}")
        if (
            expected_mapping_fingerprint is not None
            and payload.get("mapping_fingerprint")
            != expected_mapping_fingerprint
        ):
            raise ValueError(f"Reorder-time mapping mismatch: {target}")
        value = payload.get("complete_reorder_time")
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"Invalid complete reorder time: {target}")
        return float(value)

    legacy = Path(path)
    if legacy.suffix != ".time":
        legacy = legacy.with_suffix(".time")
    if allow_legacy and legacy.is_file():
        return float(legacy.read_text().strip())
    return None
