"""Ordering mechanism-discovery planning, execution, and mapping analysis."""

from __future__ import annotations

import json
import hashlib
import math
import os
import statistics
import subprocess
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from scripts.lib.core.utils import (
    canonical_name_from_converter_opt,
)
from scripts.lib.ml.portfolio import (
    CHARACTERIZATION_BASELINE_ARM_SPECS,
)
from scripts.lib.pipeline.benchmark import (
    file_sha256,
    mapping_permutation_fingerprint,
    repository_scope_state,
)
from scripts.lib.pipeline.reorder import (
    parse_reorder_time_from_converter,
)
from scripts.lib.pipeline.synthetic_graphs import (
    SCREEN_NODE_COUNTS,
    SCREEN_SEEDS,
    SYNTHETIC_POLICY_ID,
    SyntheticGraphArtifact,
    SyntheticGraphSpec,
    generate_screen_graphs,
    mechanism_discovery_screen_specs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MECHANISM_DISCOVERY_PLAN_SCHEMA = "mechanism-discovery-plan/v1"
MECHANISM_DISCOVERY_RESULT_SCHEMA = "mechanism-discovery-mapping/v1"
MEASUREMENT_MODE = "diagnostic-synthetic"
SCREEN_BASELINE_SPECS = tuple(
    spec for spec in CHARACTERIZATION_BASELINE_ARM_SPECS
    if spec in {"0", "5", "8:csr", "9:csr"}
)
_REQUIRED_SCREEN_BASELINES = {"0", "5", "8:csr", "9:csr"}
if set(SCREEN_BASELINE_SPECS) != _REQUIRED_SCREEN_BASELINES:
    raise RuntimeError(
        "Mechanism-discovery baseline SSOT changed: "
        + " ".join(SCREEN_BASELINE_SPECS)
    )
SCREEN_REORDER_SPECS = tuple(
    spec for spec in SCREEN_BASELINE_SPECS if spec != "0"
)
SCREEN_TIMEOUT_SECONDS = {
    "5": 30,
    "8:csr": 120,
    "9:csr": 120,
}
SCREEN_REPEATS = {
    "5": 1,
    "8:csr": 3,
    "9:csr": 1,
}
SCREEN_RESOLVED_SPECS = {
    "5": "5:degree=out",
    "8:csr": "8:csr:degree-sort=out-in:resolution=1",
    "9:csr": "9:csr:window=5",
}
SCREEN_NODE_HOUR_CAP = 24.0
SCREEN_CONFIGURATION_CAP = 48
SCREEN_SCALE_RESERVE = 6


def _atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_text(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


def _canonical_digest(payload: Any) -> str:
    import hashlib

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _screen_environment(threads: int) -> dict[str, str]:
    return {
        "PATH": (
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:"
            "/usr/bin:/sbin:/bin"
        ),
        "LD_LIBRARY_PATH": "",
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "GRAPHBREW_DB_DIR": "",
        "GRAPHBREW_TOPOLOGY_ANALYSIS": "0",
        "GRAPHBREW_MAPPING_QUALITY": "0",
        "GRAPHBREW_ALLOW_LEGACY_TIME": "0",
        "OMP_NUM_THREADS": str(threads),
        "OMP_THREAD_LIMIT": str(threads),
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
        "OMP_DYNAMIC": "FALSE",
        "RABBIT_RESOLUTION": "1",
        "GORDER_WINDOW": "5",
        "GORDER_FAST_BATCH": "256",
        "ADAPTIVE_DON_TIEBREAK": "0",
        "ADAPTIVE_SKIP_MODULARITY": "0",
    }


def _parse_resolved_reorder_spec(output: str) -> str:
    import re

    matches = re.findall(
        r"^Resolved Reorder Spec:\s*(\S+)\s*$",
        output,
        flags=re.MULTILINE,
    )
    if len(matches) != 1:
        raise ValueError(
            "Converter output must contain one resolved reorder spec")
    return matches[0]


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, math.ceil(fraction * len(ordered)) - 1),
    )
    return float(ordered[index])


def load_inverse_mapping(
    mapping_path: str | os.PathLike,
    *,
    expected_nodes: int | None = None,
) -> list[int]:
    path = Path(mapping_path)
    try:
        values = [int(value) for value in path.read_text().split()]
    except ValueError as error:
        raise ValueError(f"Malformed mapping: {path}") from error
    if expected_nodes is not None and len(values) != expected_nodes:
        raise ValueError(
            f"Mapping length changed: {path} "
            f"({len(values)} != {expected_nodes})"
        )
    if sorted(values) != list(range(len(values))):
        raise ValueError(f"Mapping is not a permutation: {path}")
    return values


def invert_mapping(new_to_source: Iterable[int]) -> list[int]:
    values = list(new_to_source)
    source_to_new = [-1] * len(values)
    for new_id, source_id in enumerate(values):
        if (
            source_id < 0
            or source_id >= len(values)
            or source_to_new[source_id] != -1
        ):
            raise ValueError("Mapping is not a permutation")
        source_to_new[source_id] = new_id
    return source_to_new


def _load_undirected_adjacency(
    graph_path: Path,
    nodes: int,
) -> tuple[list[list[int]], list[tuple[int, int]]]:
    adjacency = [[] for _ in range(nodes)]
    edges = []
    for line_number, line in enumerate(
        graph_path.read_text().splitlines(), 1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 2:
            raise ValueError(
                f"Malformed edge at {graph_path}:{line_number}")
        source, destination = int(fields[0]), int(fields[1])
        if (
            source < 0
            or destination < 0
            or source >= nodes
            or destination >= nodes
            or source == destination
        ):
            raise ValueError(
                f"Invalid edge at {graph_path}:{line_number}")
        low, high = sorted((source, destination))
        edges.append((low, high))
        adjacency[source].append(destination)
        adjacency[destination].append(source)
    if len(edges) != len(set(edges)):
        raise ValueError(f"Synthetic graph contains duplicate edges: {graph_path}")
    return adjacency, edges


def _mapping_divergence(
    source_to_new: list[int],
    reference_source_to_new: list[int],
    vertex_metadata: dict[str, Any],
) -> dict[str, Any]:
    nodes = len(source_to_new)
    denominator = max(1.0, nodes * nodes / 2.0)
    direct = [
        abs(source_to_new[source] - reference_source_to_new[source])
        for source in range(nodes)
    ]
    reversed_reference = [
        nodes - 1 - reference_source_to_new[source]
        for source in range(nodes)
    ]
    reversed_delta = [
        abs(source_to_new[source] - reversed_reference[source])
        for source in range(nodes)
    ]
    use_reversed = sum(reversed_delta) < sum(direct)
    absolute = reversed_delta if use_reversed else direct
    by_role: dict[str, list[int]] = defaultdict(list)
    roles = vertex_metadata["role_by_source"]
    for source, difference in enumerate(absolute):
        by_role[str(roles[source])].append(difference)
    groups: dict[int, list[int]] = defaultdict(list)
    for source, group in enumerate(vertex_metadata["group_by_source"]):
        groups[int(group)].append(source)
    group_rank_deltas = []
    group_contiguity = []
    for sources in groups.values():
        if len(sources) <= 1:
            continue
        mapping_order = sorted(
            sources, key=lambda source: source_to_new[source])
        reference_order = sorted(
            sources, key=lambda source: reference_source_to_new[source])
        mapping_rank = {
            source: index for index, source in enumerate(mapping_order)
        }
        reference_rank = {
            source: index for index, source in enumerate(reference_order)
        }
        reverse_rank = {
            source: len(sources) - 1 - rank
            for source, rank in reference_rank.items()
        }
        direct_group = sum(
            abs(mapping_rank[source] - reference_rank[source])
            for source in sources
        )
        reverse_group = sum(
            abs(mapping_rank[source] - reverse_rank[source])
            for source in sources
        )
        group_rank_deltas.append(
            min(direct_group, reverse_group)
            / max(1.0, len(sources) * len(sources) / 2.0)
        )
        positions = sorted(source_to_new[source] for source in sources)
        group_contiguity.append(
            (positions[-1] - positions[0] + 1) / len(sources)
        )
    return {
        "frame_alignment": "reversed" if use_reversed else "direct",
        "normalized_footrule_frame_invariant":
            sum(absolute) / denominator,
        "mean_absolute_position_delta": sum(absolute) / max(1, nodes),
        "fraction_moved_over_10pct": sum(
            difference > 0.1 * nodes for difference in absolute
        ) / max(1, nodes),
        "by_role": {
            role: {
                "vertices": len(values),
                "mean_absolute_position_delta":
                    sum(values) / max(1, len(values)),
                "fraction_moved_over_10pct": sum(
                    value > 0.1 * nodes for value in values
                ) / max(1, len(values)),
            }
            for role, values in sorted(by_role.items())
        },
        "mean_within_group_rank_divergence":
            sum(group_rank_deltas) / max(1, len(group_rank_deltas)),
        "mean_group_contiguity_ratio":
            sum(group_contiguity) / max(1, len(group_contiguity)),
    }


def compare_mappings(
    artifact: SyntheticGraphArtifact,
    left_path: Path,
    right_path: Path,
    *,
    left_spec: str,
    right_spec: str,
    property_bytes: int = 8,
    cache_line_bytes: int = 64,
) -> dict[str, Any]:
    metadata = json.loads(artifact.metadata_path.read_text())
    if file_sha256(artifact.graph_path) != metadata["graph_sha256"]:
        raise RuntimeError(
            f"Synthetic graph changed before analysis: {artifact.spec.name}")
    nodes = int(metadata["nodes"])
    if file_sha256(artifact.graph_path) != metadata["graph_sha256"]:
        raise RuntimeError(
            f"Synthetic graph changed before comparison: {artifact.spec.name}")
    left = invert_mapping(load_inverse_mapping(
        left_path, expected_nodes=nodes))
    right = invert_mapping(load_inverse_mapping(
        right_path, expected_nodes=nodes))
    vertex_metadata = json.loads(
        artifact.vertex_metadata_path.read_text())
    _adjacency, edges = _load_undirected_adjacency(
        artifact.graph_path, nodes)
    edge_set = set(edges)
    line_vertices = max(1, cache_line_bytes // property_bytes)
    left_colocated = {
        edge for edge in edges
        if left[edge[0]] // line_vertices
        == left[edge[1]] // line_vertices
    }
    right_colocated = {
        edge for edge in edges
        if right[edge[0]] // line_vertices
        == right[edge[1]] // line_vertices
    }
    return {
        "left_spec": left_spec,
        "right_spec": right_spec,
        "left_mapping_fingerprint":
            mapping_permutation_fingerprint(left_path),
        "right_mapping_fingerprint":
            mapping_permutation_fingerprint(right_path),
        "placement_divergence": _mapping_divergence(
            left, right, vertex_metadata),
        "edge_colocation_agreement_fraction": (
            len(edge_set - (left_colocated ^ right_colocated))
            / max(1, len(edges))
        ),
        "left_only_colocated_fraction":
            len(left_colocated - right_colocated) / max(1, len(edges)),
        "right_only_colocated_fraction":
            len(right_colocated - left_colocated) / max(1, len(edges)),
        "both_colocated_fraction":
            len(left_colocated & right_colocated) / max(1, len(edges)),
    }


def analyze_mapping(
    artifact: SyntheticGraphArtifact,
    mapping_path: Path,
    *,
    algorithm_spec: str,
    reference_mapping_path: Path | None = None,
    property_bytes: int = 8,
    cache_line_bytes: int = 64,
) -> dict[str, Any]:
    metadata = json.loads(artifact.metadata_path.read_text())
    nodes = int(metadata["nodes"])
    new_to_source = load_inverse_mapping(
        mapping_path, expected_nodes=nodes)
    source_to_new = invert_mapping(new_to_source)
    adjacency, edges = _load_undirected_adjacency(
        artifact.graph_path, nodes)
    line_vertices = max(1, cache_line_bytes // property_bytes)

    edge_gaps = [
        abs(source_to_new[source] - source_to_new[destination])
        for source, destination in edges
    ]
    positive_bit_mloga = sum(
        1 + int(math.log2(max(1, gap)))
        for gap in edge_gaps
    )
    same_line_edges = sum(
        source_to_new[source] // line_vertices
        == source_to_new[destination] // line_vertices
        for source, destination in edges
    )

    line_counts = []
    row_spans = []
    neighbor_gaps = []
    group_lines: dict[int, set[int]] = defaultdict(set)
    vertex_metadata = json.loads(
        artifact.vertex_metadata_path.read_text())
    groups = vertex_metadata["group_by_source"]
    group_sizes: dict[int, int] = defaultdict(int)
    for source, neighbors in enumerate(adjacency):
        group = int(groups[source])
        group_sizes[group] += 1
        group_lines[group].add(
            source_to_new[source] // line_vertices)
        if not neighbors:
            continue
        positions = sorted(source_to_new[neighbor] for neighbor in neighbors)
        line_counts.append(len({
            position // line_vertices for position in positions
        }))
        row_spans.append(positions[-1] - positions[0] + 1)
        neighbor_gaps.extend(
            positions[index] - positions[index - 1]
            for index in range(1, len(positions))
        )

    result = {
        "schema": MECHANISM_DISCOVERY_RESULT_SCHEMA,
        "policy_id": SYNTHETIC_POLICY_ID,
        "measurement_mode": MEASUREMENT_MODE,
        "claim_eligible": False,
        "graph": artifact.spec.name,
        "family": artifact.spec.family,
        "nodes": nodes,
        "undirected_edges": len(edges),
        "seed": artifact.spec.seed,
        "reference_kind": metadata["reference_kind"],
        "algorithm_spec": algorithm_spec,
        "algorithm_name": (
            algorithm_spec
            if algorithm_spec == "REFERENCE"
            else canonical_name_from_converter_opt(algorithm_spec)
        ),
        "mapping_path": str(mapping_path.resolve()),
        "mapping_sha256": file_sha256(mapping_path),
        "mapping_fingerprint":
            mapping_permutation_fingerprint(mapping_path),
        "property_bytes": property_bytes,
        "cache_line_bytes": cache_line_bytes,
        "vertices_per_line": line_vertices,
        "positive_bit_mloga": positive_bit_mloga,
        "positive_bit_mloga_per_edge":
            positive_bit_mloga / max(1, len(edges)),
        "mean_edge_gap": sum(edge_gaps) / max(1, len(edge_gaps)),
        "p95_edge_gap": _percentile(edge_gaps, 0.95),
        "max_edge_gap": max(edge_gaps, default=0),
        "same_line_edge_fraction":
            same_line_edges / max(1, len(edges)),
        "mean_lines_per_nonempty_row":
            sum(line_counts) / max(1, len(line_counts)),
        "mean_row_span": sum(row_spans) / max(1, len(row_spans)),
        "mean_normalized_row_span":
            sum(row_spans) / max(1, len(row_spans) * nodes),
        "mean_consecutive_neighbor_gap":
            sum(neighbor_gaps) / max(1, len(neighbor_gaps)),
        "p95_consecutive_neighbor_gap":
            _percentile(neighbor_gaps, 0.95),
        "mean_group_property_lines":
            sum(map(len, group_lines.values()))
            / max(1, len(group_lines)),
        "mean_group_line_overhead_ratio": sum(
            len(group_lines[group])
            / max(1, math.ceil(size / line_vertices))
            for group, size in group_sizes.items()
        ) / max(1, len(group_sizes)),
    }
    if reference_mapping_path is not None:
        reference = invert_mapping(load_inverse_mapping(
            reference_mapping_path,
            expected_nodes=nodes,
        ))
        result["decision_divergence"] = _mapping_divergence(
            source_to_new,
            reference,
            vertex_metadata,
        )
    return result


def _write_identity_mapping(path: Path, nodes: int) -> None:
    content = "\n".join(map(str, range(nodes))) + "\n"
    if path.is_file() and path.read_text() != content:
        raise RuntimeError(f"Frozen identity mapping changed: {path}")
    if not path.is_file():
        _atomic_text(content, path)


def classify_reference(
    aggregate: dict[str, Any],
    reference_kind: str,
) -> dict[str, Any]:
    reference_value = aggregate["REFERENCE"][
        "positive_bit_mloga_per_edge"]["median"]
    improvements = {}
    reference_defects = []
    control_baseline_advantage = {}
    for baseline in ("0", "5", "8:csr", "9:csr"):
        baseline_value = aggregate[baseline][
            "positive_bit_mloga_per_edge"]["median"]
        improvements[baseline] = (
            baseline_value - reference_value
        ) / max(baseline_value, 1e-12)
        if (
            reference_kind.startswith("control-")
            and baseline_value < reference_value
        ):
            control_baseline_advantage[baseline] = (
                reference_value - baseline_value
            ) / max(reference_value, 1e-12)
        elif baseline_value < reference_value:
            reference_defects.append(baseline)
    rabbit_spread = aggregate["8:csr"][
        "positive_bit_mloga_per_edge"]["relative_spread"]
    qualifies = (
        not reference_kind.startswith("control-")
        and not reference_defects
        and improvements["8:csr"] >= 0.05
        and improvements["9:csr"] >= 0.05
        and min(
            improvements["8:csr"],
            improvements["9:csr"],
        ) > rabbit_spread
    )
    if reference_kind.startswith("control-") and qualifies:
        raise RuntimeError("Control reference qualified unexpectedly")
    return {
        "reference_improvement": improvements,
        "reference_defects": reference_defects,
        "control_baseline_advantage":
            control_baseline_advantage,
        "rabbit_relative_spread": rabbit_spread,
        "rabbit_draws_identical": len(set(
            aggregate["8:csr"]["mapping_fingerprints"]
        )) == 1,
        "mapping_screen_qualifies": qualifies,
    }


def build_mapping_screen_plan(
    graph_root: Path,
    artifact_root: Path,
    *,
    threads: int = 4,
    cpu_list: str | None = None,
    refreeze_graphs: bool = False,
    specs: Iterable[SyntheticGraphSpec] | None = None,
    require_clean_implementation: bool = True,
    require_full_screen: bool = True,
) -> dict[str, Any]:
    if threads <= 0:
        raise ValueError("Mechanism-discovery threads must be positive")
    graphs = generate_screen_graphs(
        graph_root,
        refreeze=refreeze_graphs,
        specs=specs,
    )
    full_screen = {
        artifact.spec.name for artifact in graphs
    } == {
        spec.name for spec in mechanism_discovery_screen_specs()
    }
    if require_full_screen and not full_screen:
        raise ValueError(
            "Mechanism-discovery production plan requires the full screen")
    if len(graphs) > SCREEN_CONFIGURATION_CAP:
        raise ValueError("Mechanism-discovery configuration cap exceeded")
    if len(graphs) + SCREEN_SCALE_RESERVE > SCREEN_CONFIGURATION_CAP:
        raise ValueError(
            "Mechanism-discovery screen leaves no scale-stage reserve")

    converter = PROJECT_ROOT / "bench" / "bin" / "converter"
    if not converter.is_file():
        raise FileNotFoundError(f"Converter binary is missing: {converter}")
    output_root = Path(artifact_root)
    commands = []
    graph_records = []
    for artifact in graphs:
        artifact_metadata = json.loads(
            artifact.metadata_path.read_text())
        graph_output = output_root / artifact.spec.name
        original_mapping = graph_output / "ORIGINAL.lo"
        _write_identity_mapping(original_mapping, artifact.spec.nodes)
        graph_records.append({
            "graph": artifact.spec.name,
            "family": artifact.spec.family,
            "nodes": artifact.spec.nodes,
            "seed": artifact.spec.seed,
            "graph_path": str(artifact.graph_path.resolve()),
            "graph_sha256": file_sha256(artifact.graph_path),
            "metadata_path": str(artifact.metadata_path.resolve()),
            "metadata_sha256": file_sha256(artifact.metadata_path),
            "vertex_metadata_path":
                str(artifact.vertex_metadata_path.resolve()),
            "vertex_metadata_sha256":
                file_sha256(artifact.vertex_metadata_path),
            "reference_mapping_path":
                str(artifact.reference_mapping_path.resolve()),
            "reference_mapping_sha256":
                file_sha256(artifact.reference_mapping_path),
            "reference_kind": artifact_metadata["reference_kind"],
            "parameters": artifact_metadata["spec"]["parameters"],
            "original_mapping_path": str(original_mapping.resolve()),
            "original_mapping_sha256": file_sha256(original_mapping),
        })
        for spec in SCREEN_REORDER_SPECS:
            for repeat in range(SCREEN_REPEATS[spec]):
                suffix = (
                    f"__draw{repeat}"
                    if SCREEN_REPEATS[spec] > 1 else ""
                )
                mapping_path = graph_output / (
                    canonical_name_from_converter_opt(spec)
                    + suffix + ".lo"
                )
                command = [
                    str(converter),
                    "-f", str(artifact.graph_path.resolve()),
                    "-s",
                    "-o", spec,
                    "-q", str(mapping_path.resolve()),
                ]
                if cpu_list:
                    command = ["taskset", "-c", cpu_list, *command]
                command_id = (
                    f"{artifact.spec.name}|{spec}|r{repeat}"
                )
                environment = _screen_environment(threads)
                commands.append({
                    "command_id": command_id,
                    "graph": artifact.spec.name,
                    "family": artifact.spec.family,
                    "nodes": artifact.spec.nodes,
                    "seed": artifact.spec.seed,
                    "algorithm_spec": spec,
                    "algorithm_name":
                        canonical_name_from_converter_opt(spec),
                    "repeat": repeat,
                    "mapping_path": str(mapping_path.resolve()),
                    "graph_sha256": file_sha256(
                        artifact.graph_path),
                    "command": command,
                    "timeout_seconds": SCREEN_TIMEOUT_SECONDS[spec],
                    "threads": threads,
                    "cpu_list": cpu_list,
                    "environment_mode": "clean-allowlist/v1",
                    "environment": environment,
                    "expected_resolved_algorithm_spec":
                        SCREEN_RESOLVED_SPECS[spec],
                    "measurement_mode": MEASUREMENT_MODE,
                    "claim_eligible": False,
                })

    cap_hours = sum(
        command["timeout_seconds"] for command in commands
    ) / 3600.0
    if cap_hours > SCREEN_NODE_HOUR_CAP:
        raise ValueError(
            f"Mechanism-discovery cap exceeds {SCREEN_NODE_HOUR_CAP} hours")
    repository_state = repository_scope_state(
        PROJECT_ROOT,
        (
            "scripts/lib/pipeline/synthetic_graphs.py",
            "scripts/lib/analysis/mechanism_discovery.py",
        ),
    )
    if require_clean_implementation and (
        repository_state["relevant_untracked"]
        or repository_state["relevant_diff_sha256"]
        != hashlib.sha256(b"").hexdigest()
    ):
        raise RuntimeError(
            "Commit the reviewed mechanism-discovery implementation "
            "before freezing its plan"
        )
    plan = {
        "schema": MECHANISM_DISCOVERY_PLAN_SCHEMA,
        "policy_id": SYNTHETIC_POLICY_ID,
        "stage": "small-instance-mapping-screen",
        "full_screen": full_screen,
        "measurement_mode": MEASUREMENT_MODE,
        "claim_eligible": False,
        "configuration_count": len(graphs),
        "configuration_cap": SCREEN_CONFIGURATION_CAP,
        "reserved_scale_configurations": SCREEN_SCALE_RESERVE,
        "planned_total_configurations":
            len(graphs) + SCREEN_SCALE_RESERVE,
        "command_count": len(commands),
        "defined_cap_hours": cap_hours,
        "node_hour_cap": SCREEN_NODE_HOUR_CAP,
        "threads": threads,
        "cpu_list": cpu_list,
        "converter": {
            "path": str(converter),
            "sha256": file_sha256(converter),
        },
        "baselines": list(SCREEN_BASELINE_SPECS),
        "reference": "mixed-proven-heuristic-control-layouts",
        "rabbit_repeats_per_graph": SCREEN_REPEATS["8:csr"],
        "generator_repository_state": repository_state,
        "promotion_rule": {
            "primary_metric": "positive_bit_mloga_per_edge",
            "minimum_reference_improvement": 0.05,
            "minimum_improvement_baselines": ["8:csr", "9:csr"],
            "defect_scan_baselines": ["0", "5", "8:csr", "9:csr"],
            "control_reference_prefix": "control-",
            "must_hold_at_both_screen_sizes": True,
            "must_hold_for_all_label_seeds": True,
            "rabbit_signal_must_exceed_draw_spread": True,
            "maximum_promoted_families": 1,
            "scale_configurations_per_family": SCREEN_SCALE_RESERVE,
            "reference_defect_if_baseline_dominates": True,
        },
        "graphs": graph_records,
        "commands": commands,
    }
    plan["plan_sha256"] = _canonical_digest(plan)
    return plan


def write_mapping_screen_plan(
    plan: dict[str, Any],
    artifact_root: Path,
    *,
    refreeze: bool = False,
) -> Path:
    path = Path(artifact_root) / "mapping_screen_plan.json"
    if path.is_file() and json.loads(path.read_text()) != plan and not refreeze:
        raise RuntimeError(
            "Frozen mechanism-discovery plan changed; "
            "use explicit refreeze after review"
        )
    _atomic_json(plan, path)
    return path


def execute_mapping_screen(
    plan_path: Path,
    *,
    resume: bool = True,
) -> Path:
    plan = json.loads(Path(plan_path).read_text())
    recorded_digest = plan.pop("plan_sha256", None)
    if (
        plan.get("schema") != MECHANISM_DISCOVERY_PLAN_SCHEMA
        or recorded_digest != _canonical_digest(plan)
    ):
        raise ValueError("Mechanism-discovery plan binding is invalid")
    plan["plan_sha256"] = recorded_digest
    if plan["defined_cap_hours"] > plan["node_hour_cap"]:
        raise ValueError("Mechanism-discovery plan exceeds its cap")
    current_repository_state = repository_scope_state(
        PROJECT_ROOT,
        (
            "scripts/lib/pipeline/synthetic_graphs.py",
            "scripts/lib/analysis/mechanism_discovery.py",
        ),
    )
    if current_repository_state != plan["generator_repository_state"]:
        raise RuntimeError(
            "Mechanism-discovery implementation changed after review")
    converter_path = Path(plan["converter"]["path"])
    if (
        not converter_path.is_file()
        or file_sha256(converter_path)
            != plan["converter"]["sha256"]
    ):
        raise RuntimeError(
            "Mechanism-discovery converter changed after review")
    for record in plan["graphs"]:
        bindings = (
            ("graph_path", "graph_sha256"),
            ("metadata_path", "metadata_sha256"),
            ("vertex_metadata_path", "vertex_metadata_sha256"),
            ("reference_mapping_path", "reference_mapping_sha256"),
            ("original_mapping_path", "original_mapping_sha256"),
        )
        for path_key, digest_key in bindings:
            path = Path(record[path_key])
            if (
                not path.is_file()
                or file_sha256(path) != record[digest_key]
            ):
                raise RuntimeError(
                    "Mechanism-discovery input changed after review: "
                    f"{record['graph']}/{path_key}"
                )

    plan_root = Path(plan_path).parent
    result_root = plan_root / "mapping_screen_results"
    result_rows = []
    graph_records = {
        record["graph"]: record for record in plan["graphs"]
    }
    for command in plan["commands"]:
        result_path = result_root / (
            command["command_id"].replace("|", "__") + ".json")
        command_digest = _canonical_digest(command)
        if result_path.is_file() and resume:
            existing = json.loads(result_path.read_text())
            if (
                existing.get("schema")
                    != "mechanism-discovery-command/v2"
                or existing.get("command_sha256") != command_digest
                or existing.get("plan_sha256") != recorded_digest
                or existing.get("graph_sha256")
                    != command["graph_sha256"]
                or existing.get("measurement_mode") != MEASUREMENT_MODE
                or existing.get("claim_eligible") is not False
            ):
                raise RuntimeError(
                    "Mechanism-discovery result belongs to a different "
                    "plan or command contract; use --mechanism-discovery-"
                    "no-resume after review"
                )
            mapping_path = Path(command["mapping_path"])
            if (
                existing.get("success") is True
                and mapping_path.is_file()
                and file_sha256(mapping_path)
                    == existing.get("mapping_sha256")
            ):
                result_rows.append(existing)
                continue

        if command.get("environment_mode") != "clean-allowlist/v1":
            raise RuntimeError(
                "Mechanism-discovery environment policy changed")
        expected_environment = _screen_environment(
            int(command["threads"]))
        if command["environment"] != expected_environment:
            raise RuntimeError(
                "Mechanism-discovery environment contract changed")
        environment = dict(command["environment"])
        started = time.monotonic()
        mapping_path = Path(command["mapping_path"])
        mapping_path.unlink(missing_ok=True)
        timeout = False
        try:
            completed = subprocess.run(
                command["command"],
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=command["timeout_seconds"],
            )
            stdout = completed.stdout
            stderr = completed.stderr
            returncode = completed.returncode
        except subprocess.TimeoutExpired as error:
            timeout = True
            stdout = (
                error.stdout.decode()
                if isinstance(error.stdout, bytes)
                else error.stdout or ""
            )
            stderr = (
                error.stderr.decode()
                if isinstance(error.stderr, bytes)
                else error.stderr or ""
            )
            returncode = None
        success = (
            not timeout
            and returncode == 0
            and mapping_path.is_file()
        )
        reorder_time = (
            parse_reorder_time_from_converter(
                stdout + "\n" + stderr
            ) if success else None
        )
        resolved_spec = None
        contract_violation = None
        if success:
            try:
                resolved_spec = _parse_resolved_reorder_spec(
                    stdout + "\n" + stderr)
            except ValueError as error:
                contract_violation = str(error)
                success = False
            if (
                success
                and resolved_spec
                != command["expected_resolved_algorithm_spec"]
            ):
                contract_violation = (
                    "Resolved reorder specification mismatch: "
                    f"{resolved_spec} != "
                    f"{command['expected_resolved_algorithm_spec']}"
                )
                success = False
        result = {
            "schema": "mechanism-discovery-command/v2",
            "plan_sha256": recorded_digest,
            "command_sha256": command_digest,
            "command_id": command["command_id"],
            "graph": command["graph"],
            "algorithm_spec": command["algorithm_spec"],
            "resolved_algorithm_spec": resolved_spec,
            "repeat": command["repeat"],
            "graph_sha256": command["graph_sha256"],
            "mapping_path": command["mapping_path"],
            "mapping_sha256":
                file_sha256(mapping_path) if success else None,
            "mapping_fingerprint":
                mapping_permutation_fingerprint(mapping_path)
                if success else None,
            "reorder_time": reorder_time,
            "duration_seconds": time.monotonic() - started,
            "returncode": returncode,
            "error_kind": "timeout" if timeout else (
                "contract-violation"
                if contract_violation else (
                    "" if success else "process-failure"
                )
            ),
            "contract_violation": contract_violation,
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
            "measurement_mode": MEASUREMENT_MODE,
            "claim_eligible": False,
        }
        _atomic_json(result, result_path)
        result_rows.append(result)
        if not success:
            raise RuntimeError(
                f"Mechanism-discovery command failed: "
                f"{command['command_id']} ({result['error_kind']})"
            )

    metrics = []
    pairwise_divergence = []
    graph_summaries = []
    command_results: dict[
        tuple[str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in result_rows:
        command_results[
            (row["graph"], row["algorithm_spec"])
        ].append(row)
    for graph_name, record in graph_records.items():
        metadata = json.loads(Path(record["metadata_path"]).read_text())
        spec_payload = metadata["spec"]
        artifact = SyntheticGraphArtifact(
            spec=SyntheticGraphSpec(
                family=spec_payload["family"],
                nodes=int(spec_payload["nodes"]),
                seed=int(spec_payload["seed"]),
                parameters=dict(spec_payload.get("parameters", {})),
            ),
            graph_path=Path(record["graph_path"]),
            reference_mapping_path=Path(
                record["reference_mapping_path"]),
            vertex_metadata_path=Path(
                metadata["vertex_metadata_path"]
            ),
            metadata_path=Path(record["metadata_path"]),
            undirected_edges=int(metadata["undirected_edges"]),
        )
        mapping_entries = [
            ("0", 0, Path(record["original_mapping_path"])),
            ("REFERENCE", 0, artifact.reference_mapping_path),
        ]
        for spec in SCREEN_REORDER_SPECS:
            for row in sorted(
                command_results[(graph_name, spec)],
                key=lambda value: int(value["repeat"]),
            ):
                mapping_entries.append((
                    spec,
                    int(row["repeat"]),
                    Path(row["mapping_path"]),
                ))

        graph_metrics = []
        for spec, repeat, mapping_path in mapping_entries:
            row = analyze_mapping(
                artifact,
                mapping_path,
                algorithm_spec=spec,
                reference_mapping_path=artifact.reference_mapping_path,
            )
            row["mapping_draw"] = repeat
            graph_metrics.append(row)
            metrics.append(row)

        pair_entries = [
            (
                spec if SCREEN_REPEATS.get(spec, 1) == 1
                else f"{spec}#draw{repeat}",
                mapping_path,
            )
            for spec, repeat, mapping_path in mapping_entries
        ]
        for (left_spec, left_path), (right_spec, right_path) in combinations(
            pair_entries, 2
        ):
            pairwise_divergence.append({
                "graph": graph_name,
                **compare_mappings(
                    artifact,
                    left_path,
                    right_path,
                    left_spec=left_spec,
                    right_spec=right_spec,
                ),
            })

        by_spec: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in graph_metrics:
            by_spec[row["algorithm_spec"]].append(row)
        aggregate = {}
        for spec, rows in sorted(by_spec.items()):
            primary = [
                float(row["positive_bit_mloga_per_edge"])
                for row in rows
            ]
            same_line = [
                float(row["same_line_edge_fraction"])
                for row in rows
            ]
            primary_median = statistics.median(primary)
            aggregate[spec] = {
                "draws": len(rows),
                "mapping_fingerprints": [
                    row["mapping_fingerprint"] for row in rows
                ],
                "positive_bit_mloga_per_edge": {
                    "median": primary_median,
                    "min": min(primary),
                    "max": max(primary),
                    "relative_spread": (
                        (max(primary) - min(primary))
                        / max(primary_median, 1e-12)
                    ),
                },
                "same_line_edge_fraction": {
                    "median": statistics.median(same_line),
                    "min": min(same_line),
                    "max": max(same_line),
                },
            }
        reference_kind = record["reference_kind"]
        classification = classify_reference(
            aggregate, reference_kind)
        graph_summaries.append({
            "graph": graph_name,
            "family": record["family"],
            "nodes": record["nodes"],
            "seed": record["seed"],
            "reference_kind": record["reference_kind"],
            "aggregate": aggregate,
            **classification,
        })

    family_summaries = []
    for family in sorted({
        summary["family"] for summary in graph_summaries
    }):
        rows = [
            summary for summary in graph_summaries
            if summary["family"] == family
        ]
        reference_kinds = {
            row["reference_kind"] for row in rows
        }
        if len(reference_kinds) != 1:
            raise RuntimeError(
                f"Synthetic family reference kind changed: {family}")
        sizes = sorted({row["nodes"] for row in rows})
        seeds = sorted({row["seed"] for row in rows})
        if plan["full_screen"] and (
            sizes != list(SCREEN_NODE_COUNTS)
            or seeds != list(SCREEN_SEEDS)
        ):
            raise RuntimeError(
                f"Mechanism-discovery family coverage changed: {family}")
        minimum_improvement = min(
            min(
                row["reference_improvement"]["8:csr"],
                row["reference_improvement"]["9:csr"],
            )
            for row in rows
        )
        family_summaries.append({
            "family": family,
            "reference_kind": next(iter(reference_kinds)),
            "configurations": len(rows),
            "sizes": sizes,
            "seeds": seeds,
            "all_configurations_qualify": all(
                row["mapping_screen_qualifies"] for row in rows
            ),
            "minimum_reference_improvement":
                minimum_improvement,
            "reference_defect_graphs": [
                row["graph"] for row in rows
                if row["reference_defects"]
            ],
            "control_advantage_graphs": [
                row["graph"] for row in rows
                if row["control_baseline_advantage"]
            ],
        })
    promotion_candidates = sorted(
        (
            row for row in family_summaries
            if row["all_configurations_qualify"]
        ),
        key=lambda row: row["minimum_reference_improvement"],
        reverse=True,
    )

    output = {
        "schema": "mechanism-discovery-mapping-screen/v1",
        "plan": str(Path(plan_path).resolve()),
        "plan_sha256": recorded_digest,
        "measurement_mode": MEASUREMENT_MODE,
        "claim_eligible": False,
        "measured_hours": sum(
            float(row["duration_seconds"]) for row in result_rows
        ) / 3600.0,
        "command_results": result_rows,
        "mapping_metrics": metrics,
        "pairwise_decision_divergence": pairwise_divergence,
        "graph_summaries": graph_summaries,
        "family_summaries": family_summaries,
        "promotion_candidates": promotion_candidates,
        "promoted_family_pending_review": (
            promotion_candidates[0]["family"]
            if promotion_candidates else None
        ),
    }
    output_path = plan_root / "mapping_screen_results.json"
    _atomic_json(output, output_path)
    return output_path


__all__ = [
    "MEASUREMENT_MODE",
    "MECHANISM_DISCOVERY_PLAN_SCHEMA",
    "SCREEN_BASELINE_SPECS",
    "SCREEN_CONFIGURATION_CAP",
    "SCREEN_NODE_HOUR_CAP",
    "analyze_mapping",
    "build_mapping_screen_plan",
    "classify_reference",
    "compare_mappings",
    "execute_mapping_screen",
    "invert_mapping",
    "load_inverse_mapping",
    "write_mapping_screen_plan",
]
