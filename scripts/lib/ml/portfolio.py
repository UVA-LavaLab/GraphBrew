"""Frozen deployable adaptive-arm registry shared with C++."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PORTFOLIO_PATH = (
    PROJECT_ROOT / "bench/include/graphbrew/reorder/adaptive_portfolio.def"
)
_ARM_PATTERN = re.compile(
    r'^GRAPHBREW_ADAPTIVE_ARM\('
    r'([A-Z0-9_]+),\s*"([^"]+)",\s*"([^"]+)"\)$'
)


def _load_portfolio() -> tuple[tuple[str, str, str], ...]:
    arms = []
    for line in PORTFOLIO_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        match = _ARM_PATTERN.fullmatch(line)
        if match is None:
            raise RuntimeError(f"Invalid adaptive portfolio line: {line}")
        arms.append(match.groups())
    if not arms:
        raise RuntimeError("Adaptive portfolio is empty")
    specs = [arm[1] for arm in arms]
    canonical = [arm[2] for arm in arms]
    if len(specs) != len(set(specs)) or len(canonical) != len(set(canonical)):
        raise RuntimeError("Adaptive portfolio labels must be unique")
    return tuple(arms)


DEPLOYABLE_ARMS = _load_portfolio()
DEPLOYABLE_ARM_SPECS = tuple(arm[1] for arm in DEPLOYABLE_ARMS)
DEPLOYABLE_ARM_CANONICAL_NAMES = tuple(arm[2] for arm in DEPLOYABLE_ARMS)
_LABEL_TO_SPEC = {
    label: spec
    for _symbol, spec, canonical in DEPLOYABLE_ARMS
    for label in (spec, canonical)
}
_SPEC_TO_CANONICAL = {
    spec: canonical
    for _symbol, spec, canonical in DEPLOYABLE_ARMS
}


def normalize_deployable_arm(label: str) -> str:
    """Normalize an exact spec or migration-safe canonical key to the spec."""
    try:
        return _LABEL_TO_SPEC[label]
    except KeyError as error:
        raise ValueError(
            f"Adaptive model emitted a non-portfolio label: {label}"
        ) from error


def canonical_deployable_arm(label: str) -> str:
    """Return the migration-safe canonical name for a portfolio label."""
    return _SPEC_TO_CANONICAL[normalize_deployable_arm(label)]


def normalize_deployable_portfolio(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize a complete portfolio and reject conflicting aliases."""
    if not isinstance(payload, Mapping):
        raise ValueError("Adaptive portfolio must be an object")
    normalized = {}
    source_labels = {}
    for label, entry in payload.items():
        try:
            spec = normalize_deployable_arm(label)
        except ValueError:
            continue
        if spec in normalized and normalized[spec] != entry:
            raise ValueError(
                "Adaptive model contains conflicting aliases for "
                f"{spec}: {source_labels[spec]} and {label}"
            )
        normalized[spec] = entry
        source_labels[spec] = label

    missing = [
        spec for spec in DEPLOYABLE_ARM_SPECS
        if spec not in normalized
    ]
    if missing:
        raise ValueError(
            "Adaptive model is missing deployable portfolio arms: "
            + " ".join(missing)
        )
    return {
        spec: normalized[spec]
        for spec in DEPLOYABLE_ARM_SPECS
    }


def apply_portfolio_guard(label: str) -> tuple[str, str | None]:
    """Validate a model label before deployment."""
    return normalize_deployable_arm(label), None
