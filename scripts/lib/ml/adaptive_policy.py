"""Shared deployable adaptive-selection policy constants."""

from __future__ import annotations

import math
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = (
    PROJECT_ROOT / "bench/include/graphbrew/reorder/adaptive_policy.def"
)
_POLICY_PATTERN = re.compile(
    r"^GRAPHBREW_ADAPTIVE_POLICY\(([A-Z0-9_]+),\s*([0-9.]+)\)$"
)


def _load_policy() -> dict[str, float]:
    values = {}
    for line in POLICY_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        match = _POLICY_PATTERN.fullmatch(line)
        if match is None:
            raise RuntimeError(f"Invalid adaptive policy line: {line}")
        name, value = match.groups()
        if name in values:
            raise RuntimeError(f"Duplicate adaptive policy value: {name}")
        values[name] = float(value)
    return values


_POLICY = _load_policy()
ORIGINAL_MARGIN_THRESHOLD = _POLICY["ORIGINAL_MARGIN_THRESHOLD"]
REORDER_WEIGHT_BOOST = _POLICY["REORDER_WEIGHT_BOOST"]


def iterations_to_amortize(
    avg_speedup: float,
    avg_reorder_time: float,
) -> float:
    """Return baseline-time-equivalent runs needed to repay reordering."""
    if (
        not math.isfinite(avg_speedup)
        or not math.isfinite(avg_reorder_time)
        or avg_speedup <= 1.0
        or avg_reorder_time < 0.0
    ):
        return math.inf
    return avg_reorder_time * avg_speedup / (avg_speedup - 1.0)
