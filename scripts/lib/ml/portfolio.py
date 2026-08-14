"""Legacy adaptive-arm registry shared with C++.

This executable compatibility contract is not the headline portfolio.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from scripts.lib.core.utils import canonical_name_from_converter_opt

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
        if not line or line.startswith("//"):
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
CHARACTERIZATION_DENDROGRAM_ANCHOR = (
    "12:rabbit:compose:sg_none:"
    "comm_identity:intra_dendrogram"
)
CHARACTERIZATION_BASELINE_ARM_SPECS = (
    "0",
    "5",
    "8:csr",
    "8:boost",
    "9:csr",
    "10:canonical",
    "11:mind",
    "11:bnf",
    "15:1.0:10:10:hierarchy-degree",
    "15:1.0:10:10:final-stable",
    "15:1.0:10:10:final-degree",
    "12:rabbit:compose:sg_none:comm_identity:intra_hubsort",
    "12:rabbit:compose:sg_super_rabbit:comm_identity:intra_hubsort",
    CHARACTERIZATION_DENDROGRAM_ANCHOR,
)
_LEGACY_PIPELINE_ALIASES = {
    "12:rabbit:compose:sg_none:comm_identity:intra_hubsort":
        "GraphBrewOrder_rabbit_compose_sg_none_comm_identity_intra_hubsort",
    "12:rabbit:compose:sg_super_rabbit:comm_identity:intra_hubsort":
        "GraphBrewOrder_rabbit_compose_sg_super_rabbit_comm_identity_intra_hubsort",
}
_LABEL_TO_SPEC = {}
for _symbol, spec, canonical in DEPLOYABLE_ARMS:
    labels = {
        spec,
        canonical,
        canonical_name_from_converter_opt(spec),
    }
    legacy_alias = _LEGACY_PIPELINE_ALIASES.get(spec)
    if legacy_alias:
        labels.add(legacy_alias)
    for label in labels:
        previous = _LABEL_TO_SPEC.setdefault(label, spec)
        if previous != spec:
            raise RuntimeError(
                f"Adaptive portfolio alias collision: {label}"
            )
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
            if (
                isinstance(label, str)
                and (
                    label.startswith("RabbitCommunities_")
                    or (
                        "intra_hubsort" in label
                        and (
                            label.startswith(
                                "GraphBrewOrder_rabbit_compose_"
                            )
                            or label.startswith(
                                "12:rabbit:compose:"
                            )
                        )
                    )
                )
            ):
                raise ValueError(
                    "Adaptive model contains an unrecognized Rabbit "
                    f"portfolio alias: {label}"
                )
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
