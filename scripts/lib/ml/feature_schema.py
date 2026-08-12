"""Shared Tier-0 adaptive feature schema and transforms."""

from __future__ import annotations

import math
import re
import statistics
from pathlib import Path
from typing import Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TIER0_SCHEMA_PATH = (
    PROJECT_ROOT
    / "bench/include/graphbrew/reorder/adaptive_feature_schema.def"
)
_SCHEMA_PATTERN = re.compile(
    r'^GRAPHBREW_TIER0_FEATURE\([A-Z0-9_]+,\s*"([^"]+)"\)$'
)


def _load_tier0_feature_names() -> tuple[str, ...]:
    names = []
    for line in TIER0_SCHEMA_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        match = _SCHEMA_PATTERN.fullmatch(line)
        if match is None:
            raise RuntimeError(f"Invalid Tier-0 schema line: {line}")
        names.append(match.group(1))
    if not names or len(names) != len(set(names)):
        raise RuntimeError("Tier-0 feature names must be non-empty and unique")
    return tuple(names)


TIER0_FEATURE_NAMES = _load_tier0_feature_names()
TIER0_FEATURE_INDEX = {
    name: index for index, name in enumerate(TIER0_FEATURE_NAMES)
}
TIER0_FEATURE_COUNT = len(TIER0_FEATURE_NAMES)
TIER0_PROPERTY_INPUT_NAMES = (
    "nodes",
    "edges",
    "avg_degree",
    "degree_cv",
    "hub_concentration",
    "normalized_edge_span",
    "window_neighbor_overlap",
)
TIER0_WEIGHT_NAMES = (
    "bias",
    *(f"w_t0_{name}" for name in TIER0_FEATURE_NAMES),
)


def reuse_bucket(reuse_count: float) -> float:
    """Map the frozen reuse sweep to an ordinal feature bucket."""
    if reuse_count <= 1:
        return 0.0
    if reuse_count <= 5:
        return 1.0
    if reuse_count <= 10:
        return 2.0
    if reuse_count <= 20:
        return 3.0
    if reuse_count <= 50:
        return 4.0
    if reuse_count <= 100:
        return 5.0
    return 6.0


def extract_tier0_features(
    properties: Mapping[str, float],
    *,
    property_wsr_llc: float,
    kernel_class: int,
    reuse_count: float,
) -> list[float]:
    """Extract values in the shared C++/Python Tier-0 schema order."""
    missing = [
        name for name in TIER0_PROPERTY_INPUT_NAMES
        if name not in properties
    ]
    if missing:
        raise ValueError(
            "Tier-0 extraction is missing properties: "
            + ", ".join(missing)
        )
    if not math.isfinite(property_wsr_llc) or property_wsr_llc < 0:
        raise ValueError(
            "Tier-0 extraction requires kernel-specific property_wsr_llc"
        )
    if math.isnan(reuse_count) or reuse_count <= 0:
        raise ValueError("Tier-0 extraction requires a positive reuse count")
    nodes = float(properties["nodes"])
    edges = float(properties["edges"])
    raw_values = {
        name: float(properties[name])
        for name in TIER0_PROPERTY_INPUT_NAMES
    }
    if any(not math.isfinite(value) for value in raw_values.values()):
        raise ValueError("Tier-0 properties must be finite")
    if nodes < 0 or edges < 0:
        raise ValueError("Tier-0 graph dimensions must be non-negative")
    values = {
        "log10_nodes": math.log10(max(0.0, nodes) + 1.0),
        "log10_edges": math.log10(max(0.0, edges) + 1.0),
        "avg_degree": raw_values["avg_degree"],
        "degree_cv": raw_values["degree_cv"],
        "hub_concentration": raw_values["hub_concentration"],
        "normalized_edge_span": raw_values["normalized_edge_span"],
        "window_neighbor_overlap": raw_values[
            "window_neighbor_overlap"
        ],
        "property_wsr_llc": float(property_wsr_llc),
        "kernel_class": float(kernel_class),
        "reuse_bucket": reuse_bucket(reuse_count),
    }
    return [values[name] for name in TIER0_FEATURE_NAMES]


def validate_tier0_weight_entry(
    entry: Mapping[str, object],
    label: str,
) -> None:
    """Validate one strict adaptive-tier0/v1 arm entry."""
    missing = sorted(set(TIER0_WEIGHT_NAMES) - set(entry))
    if missing:
        raise ValueError(
            f"Adaptive arm {label} is missing: " + ", ".join(missing)
        )
    for name in TIER0_WEIGHT_NAMES:
        value = entry[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(
                f"Adaptive arm {label} has invalid numeric field {name}"
            )


def tier0_feature_record(values: Sequence[float]) -> dict[str, float]:
    if len(values) != TIER0_FEATURE_COUNT:
        raise ValueError(
            f"Expected {TIER0_FEATURE_COUNT} Tier-0 values, got {len(values)}"
        )
    return dict(zip(TIER0_FEATURE_NAMES, map(float, values)))


def informativeness_ratio(
        graph_seed_values: Mapping[str, Sequence[float]],
) -> float:
        """Return across-graph variation divided by pooled seed variation."""
        usable = {
            graph: [float(value) for value in values]
            for graph, values in graph_seed_values.items()
            if values
        }
        if len(usable) < 2:
            raise ValueError("Informativeness requires at least two graphs")
        graph_means = [
            statistics.fmean(values) for values in usable.values()
        ]
        across_graph = statistics.pstdev(graph_means)
        within_variances = [
            statistics.pvariance(values) if len(values) > 1 else 0.0
            for values in usable.values()
        ]
        within_seed = math.sqrt(statistics.fmean(within_variances))
        if within_seed == 0:
            return math.inf if across_graph > 0 else 0.0
        return across_graph / within_seed


def passes_informativeness_gate(
        graph_seed_values: Mapping[str, Sequence[float]],
) -> bool:
        return informativeness_ratio(graph_seed_values) > 1.0


def residual_informativeness_ratio(
        graph_seed_values: Mapping[str, Sequence[float]],
        graph_log_sizes: Mapping[str, tuple[float, float]],
) -> float:
        """Measure feature variation after removing log-node/log-edge effects."""
        usable = {
            graph: [float(value) for value in values]
            for graph, values in graph_seed_values.items()
            if values and graph in graph_log_sizes
        }
        if len(usable) < 4:
            raise ValueError(
                "Residual informativeness requires at least four graphs"
            )
        rows = []
        targets = []
        for graph, values in usable.items():
            log_nodes, log_edges = graph_log_sizes[graph]
            rows.append([1.0, float(log_nodes), float(log_edges)])
            targets.append(statistics.fmean(values))

        matrix = [[0.0] * 4 for _ in range(3)]
        for row, target in zip(rows, targets):
            for i in range(3):
                for j in range(3):
                    matrix[i][j] += row[i] * row[j]
                matrix[i][3] += row[i] * target
        for pivot in range(3):
            best = max(range(pivot, 3), key=lambda i: abs(matrix[i][pivot]))
            matrix[pivot], matrix[best] = matrix[best], matrix[pivot]
            if abs(matrix[pivot][pivot]) < 1e-12:
                return 0.0
            scale = matrix[pivot][pivot]
            matrix[pivot] = [value / scale for value in matrix[pivot]]
            for row_index in range(3):
                if row_index == pivot:
                    continue
                factor = matrix[row_index][pivot]
                matrix[row_index] = [
                    left - factor * right
                    for left, right in zip(
                        matrix[row_index],
                        matrix[pivot],
                    )
                ]
        coefficients = [matrix[i][3] for i in range(3)]
        residuals = [
            target - sum(c * value for c, value in zip(coefficients, row))
            for row, target in zip(rows, targets)
        ]
        residual_variation = statistics.pstdev(residuals)
        within_variances = [
            statistics.pvariance(values) if len(values) > 1 else 0.0
            for values in usable.values()
        ]
        within_seed = math.sqrt(statistics.fmean(within_variances))
        if within_seed == 0:
            return math.inf if residual_variation > 0 else 0.0
        return residual_variation / within_seed


def feature_passes_acceptance_gate(
        feature_name: str,
        graph_seed_values: Mapping[str, Sequence[float]],
        graph_log_sizes: Mapping[str, tuple[float, float]],
) -> bool:
        """Binding Sprint-1 feature gate, including size-residual checks."""
        if informativeness_ratio(graph_seed_values) <= 1.0:
            return False
        if feature_name in {
            "normalized_edge_span",
            "window_neighbor_overlap",
        }:
            return residual_informativeness_ratio(
                graph_seed_values, graph_log_sizes,
            ) > 1.0
        return True
