"""Deterministic synthetic graphs for ordering mechanism discovery."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from scripts.lib.pipeline.benchmark import (
    file_crc32,
    file_sha256,
    mapping_permutation_fingerprint,
)

SYNTHETIC_GRAPH_SCHEMA = "graphbrew-synthetic-graph/v1"
SYNTHETIC_POLICY_ID = "mechanism-discovery-screen/v1"
SCREEN_NODE_COUNTS = (4096, 16384)
SCREEN_SEEDS = (0, 1, 2)
SCREEN_FAMILIES = (
    "chain",
    "grid",
    "hub-spoke",
    "block-biclique",
    "copied-neighborhood",
    "community-bridge",
    "expander-control",
)
REFERENCE_KINDS = {
    "chain": "proven-optimum-positive-bit-mloga",
    "grid": "heuristic-morton-space-filling",
    "hub-spoke": "heuristic-median-hub-packing",
    "block-biclique": "heuristic-alternating-bipartition",
    "copied-neighborhood": "heuristic-proportional-anchor-copy",
    "community-bridge": "heuristic-contiguous-community",
    "expander-control": "control-frame-not-oracle",
}
FAMILY_DEFAULT_PARAMETERS: dict[
    str, dict[str, int | float | str]
] = {
    "chain": {},
    "grid": {"morton_tile": 8},
    "hub-spoke": {"group_size": 64},
    "block-biclique": {"block_size": 64},
    "copied-neighborhood": {
        "block_size": 64,
        "anchor_count": 16,
        "copied_degree": 8,
    },
    "community-bridge": {
        "block_size": 64,
        "ring_offsets": "1,2,5",
        "bridge_width": 2,
    },
    "expander-control": {"circulant_degree": 6},
}

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class SyntheticGraphSpec:
    family: str
    nodes: int
    seed: int
    parameters: dict[str, int | float | str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.family}-n{self.nodes}-s{self.seed}"


@dataclass(frozen=True)
class SyntheticGraphArtifact:
    spec: SyntheticGraphSpec
    graph_path: Path
    reference_mapping_path: Path
    vertex_metadata_path: Path
    metadata_path: Path
    undirected_edges: int


class _SplitMix64:
    def __init__(self, seed: int):
        self.state = seed & ((1 << 64) - 1)

    def next(self) -> int:
        self.state = (
            self.state + 0x9E3779B97F4A7C15
        ) & ((1 << 64) - 1)
        value = self.state
        value = (
            (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9
        ) & ((1 << 64) - 1)
        value = (
            (value ^ (value >> 27)) * 0x94D049BB133111EB
        ) & ((1 << 64) - 1)
        return value ^ (value >> 31)


def _fisher_yates(size: int, seed: int) -> list[int]:
    values = list(range(size))
    random = _SplitMix64(seed)
    for index in range(size - 1, 0, -1):
        selected = random.next() % (index + 1)
        values[index], values[selected] = (
            values[selected],
            values[index],
        )
    return values


def mechanism_discovery_screen_specs() -> tuple[SyntheticGraphSpec, ...]:
    return tuple(
        SyntheticGraphSpec(family, nodes, seed)
        for family in SCREEN_FAMILIES
        for nodes in SCREEN_NODE_COUNTS
        for seed in SCREEN_SEEDS
    )


def _effective_parameters(
    spec: SyntheticGraphSpec,
) -> dict[str, int | float | str]:
    parameters = dict(FAMILY_DEFAULT_PARAMETERS[spec.family])
    parameters.update(spec.parameters)
    if spec.family == "expander-control":
        parameters["circulant_offsets"] = ",".join(
            map(str, _expander_offsets(spec))
        )
    return parameters


def _morton_key(row: int, column: int) -> int:
    value = 0
    bit = 0
    while row or column:
        value |= (column & 1) << (2 * bit)
        value |= (row & 1) << (2 * bit + 1)
        column >>= 1
        row >>= 1
        bit += 1
    return value


def _chain(spec: SyntheticGraphSpec):
    edges = {(vertex, vertex + 1) for vertex in range(spec.nodes - 1)}
    roles = ["endpoint"] + ["interior"] * (spec.nodes - 2) + ["endpoint"]
    groups = [0] * spec.nodes
    return edges, list(range(spec.nodes)), roles, groups


def _grid(spec: SyntheticGraphSpec):
    side = math.isqrt(spec.nodes)
    if side * side != spec.nodes:
        raise ValueError("Grid screen sizes must be perfect squares")
    edges = set()
    for row in range(side):
        for column in range(side):
            vertex = row * side + column
            if row + 1 < side:
                edges.add((vertex, vertex + side))
            if column + 1 < side:
                edges.add((vertex, vertex + 1))
    reference = sorted(
        range(spec.nodes),
        key=lambda vertex: _morton_key(
            vertex // side, vertex % side),
    )
    roles = ["boundary" if (
        vertex // side in {0, side - 1}
        or vertex % side in {0, side - 1}
    ) else "interior" for vertex in range(spec.nodes)]
    tile = int(_effective_parameters(spec)["morton_tile"])
    groups = [
        (vertex // side // tile) * math.ceil(side / tile)
        + (vertex % side // tile)
        for vertex in range(spec.nodes)
    ]
    return edges, reference, roles, groups


def _hub_spoke(spec: SyntheticGraphSpec):
    group_size = int(_effective_parameters(spec)["group_size"])
    group_count = math.ceil(spec.nodes / group_size)
    edges = set()
    roles = ["leaf"] * spec.nodes
    groups = [0] * spec.nodes
    reference = []
    hubs = []
    for group in range(group_count):
        start = group * group_size
        stop = min(spec.nodes, start + group_size)
        hub = start
        hubs.append(hub)
        roles[hub] = "hub"
        for vertex in range(start, stop):
            groups[vertex] = group
            if vertex != hub:
                edges.add((hub, vertex))
        leaves = list(range(start + 1, stop))
        middle = len(leaves) // 2
        reference.extend(leaves[:middle])
        reference.append(hub)
        reference.extend(leaves[middle:])
    for index, hub in enumerate(hubs):
        other = hubs[(index + 1) % len(hubs)]
        if hub != other:
            edges.add(tuple(sorted((hub, other))))
    return edges, reference, roles, groups


def _block_biclique(spec: SyntheticGraphSpec):
    block_size = int(_effective_parameters(spec)["block_size"])
    half = block_size // 2
    edges = set()
    roles = ["right"] * spec.nodes
    groups = [0] * spec.nodes
    reference = []
    for group, start in enumerate(range(0, spec.nodes, block_size)):
        stop = min(spec.nodes, start + block_size)
        middle = min(stop, start + half)
        left = range(start, middle)
        right = range(middle, stop)
        for vertex in left:
            roles[vertex] = "left"
        for vertex in range(start, stop):
            groups[vertex] = group
        for left_vertex, right_vertex in zip(left, right):
            reference.extend((left_vertex, right_vertex))
        for source in left:
            for destination in right:
                edges.add((source, destination))
    return edges, reference, roles, groups


def _copied_neighborhood(spec: SyntheticGraphSpec):
    parameters = _effective_parameters(spec)
    block_size = int(parameters["block_size"])
    anchor_count = int(parameters["anchor_count"])
    copied_degree = int(parameters["copied_degree"])
    edges = set()
    roles = ["copy"] * spec.nodes
    groups = [0] * spec.nodes
    reference = []
    for group, start in enumerate(range(0, spec.nodes, block_size)):
        stop = min(spec.nodes, start + block_size)
        anchors = list(range(start, min(stop, start + anchor_count)))
        copies = range(start + len(anchors), stop)
        for anchor in anchors:
            roles[anchor] = "anchor"
        for vertex in range(start, stop):
            groups[vertex] = group
        for index, anchor in enumerate(anchors):
            if len(anchors) > 1:
                edges.add(tuple(sorted((
                    anchor, anchors[(index + 1) % len(anchors)]
                ))))
        selected = anchors[:copied_degree]
        for copy in copies:
            for anchor in selected:
                edges.add(tuple(sorted((copy, anchor))))
        copy_vertices = list(copies)
        for index, copy in enumerate(copy_vertices):
            if index < len(selected):
                reference.append(selected[index])
            reference.append(copy)
        reference.extend(selected[len(copy_vertices):])
        reference.extend(
            anchor for anchor in anchors if anchor not in selected)
    return edges, reference, roles, groups


def _community_bridge(spec: SyntheticGraphSpec):
    parameters = _effective_parameters(spec)
    block_size = int(parameters["block_size"])
    offsets = tuple(
        int(value)
        for value in str(parameters["ring_offsets"]).split(",")
    )
    bridge_width = int(parameters["bridge_width"])
    edges = set()
    roles = ["interior"] * spec.nodes
    groups = [0] * spec.nodes
    reference = list(range(spec.nodes))
    starts = list(range(0, spec.nodes, block_size))
    for group, start in enumerate(starts):
        stop = min(spec.nodes, start + block_size)
        size = stop - start
        for vertex in range(start, stop):
            groups[vertex] = group
            for offset in offsets:
                other = start + ((vertex - start + offset) % size)
                if vertex != other:
                    edges.add(tuple(sorted((vertex, other))))
    for group in range(len(starts) - 1):
        left = starts[group]
        right = starts[group + 1]
        for offset in range(bridge_width):
            source = min(spec.nodes - 1, left + offset)
            destination = min(spec.nodes - 1, right + offset)
            edges.add((source, destination))
            roles[source] = "bridge"
            roles[destination] = "bridge"
    return edges, reference, roles, groups


def _expander_offsets(spec: SyntheticGraphSpec) -> tuple[int, ...]:
    parameters = dict(FAMILY_DEFAULT_PARAMETERS["expander-control"])
    parameters.update(spec.parameters)
    degree = int(parameters["circulant_degree"])
    if degree <= 0 or degree % 2:
        raise ValueError(
            "Expander circulant degree must be a positive even integer")
    offsets_needed = degree // 2
    available_offsets = spec.nodes // 2 - 1
    if available_offsets < offsets_needed:
        raise ValueError(
            "Expander graph is too small for the requested degree")
    random = _SplitMix64(spec.seed ^ 0x455850414E444552)
    offsets = set()
    while len(offsets) < offsets_needed:
        offset = 1 + random.next() % available_offsets
        offsets.add(int(offset))
    return tuple(sorted(offsets))


def _expander_control(spec: SyntheticGraphSpec):
    offsets = _expander_offsets(spec)
    edges = set()
    for vertex in range(spec.nodes):
        for offset in offsets:
            other = (vertex + offset) % spec.nodes
            if vertex != other:
                edges.add(tuple(sorted((vertex, other))))
    roles = ["uniform"] * spec.nodes
    groups = [0] * spec.nodes
    return edges, list(range(spec.nodes)), roles, groups


_FAMILY_BUILDERS = {
    "chain": _chain,
    "grid": _grid,
    "hub-spoke": _hub_spoke,
    "block-biclique": _block_biclique,
    "copied-neighborhood": _copied_neighborhood,
    "community-bridge": _community_bridge,
    "expander-control": _expander_control,
}


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


def _metadata_matches(metadata_path: Path, expected: dict) -> bool:
    if not metadata_path.is_file():
        return False
    try:
        existing = json.loads(metadata_path.read_text())
    except (OSError, ValueError):
        return False
    return existing == expected


def _artifact_metadata(
    spec: SyntheticGraphSpec,
    graph_path: Path,
    mapping_path: Path,
    vertex_path: Path,
    undirected_edges: int,
) -> dict:
    return {
        "schema": SYNTHETIC_GRAPH_SCHEMA,
        "policy_id": SYNTHETIC_POLICY_ID,
        "spec": {
            **asdict(spec),
            "parameters": _effective_parameters(spec),
        },
        "graph_path": str(graph_path.resolve()),
        "graph_bytes": graph_path.stat().st_size,
        "graph_crc32": file_crc32(graph_path, use_cache=False),
        "graph_sha256": file_sha256(graph_path, use_cache=False),
        "nodes": spec.nodes,
        "undirected_edges": undirected_edges,
        "symmetric": True,
        "labeling": "splitmix64-fisher-yates/v1",
        "reference_mapping_path": str(mapping_path.resolve()),
        "reference_mapping_sha256":
            file_sha256(mapping_path, use_cache=False),
        "reference_mapping_fingerprint":
            mapping_permutation_fingerprint(mapping_path),
        "reference_kind": REFERENCE_KINDS[spec.family],
        "vertex_metadata_path": str(vertex_path.resolve()),
        "vertex_metadata_sha256":
            file_sha256(vertex_path, use_cache=False),
        "converter_identity": "not-applied-edge-list-screen",
        "measurement_mode": "diagnostic-synthetic",
        "claim_eligible": False,
    }


def generate_synthetic_graph(
    spec: SyntheticGraphSpec,
    graph_root: Path,
    *,
    refreeze: bool = False,
) -> SyntheticGraphArtifact:
    if spec.family not in _FAMILY_BUILDERS:
        raise ValueError(f"Unknown synthetic graph family: {spec.family}")
    if spec.nodes < 4:
        raise ValueError("Synthetic graphs require at least four vertices")

    output_dir = Path(graph_root) / spec.name
    graph_path = output_dir / f"{spec.name}.el"
    mapping_path = output_dir / "REFERENCE.lo"
    vertex_path = output_dir / "vertices.json"
    metadata_path = output_dir / "metadata.json"

    edges, reference_order, roles, logical_groups = (
        _FAMILY_BUILDERS[spec.family](spec)
    )
    if (
        len(reference_order) != spec.nodes
        or len(set(reference_order)) != spec.nodes
        or len(roles) != spec.nodes
        or len(logical_groups) != spec.nodes
    ):
        raise RuntimeError(
            f"Synthetic family emitted an invalid contract: {spec.name}")

    logical_to_source = _fisher_yates(
        spec.nodes,
        spec.seed ^ 0x4752415048425245,
    )
    source_to_logical = [-1] * spec.nodes
    for logical, source in enumerate(logical_to_source):
        source_to_logical[source] = logical

    source_edges = sorted({
        tuple(sorted((
            logical_to_source[source],
            logical_to_source[destination],
        )))
        for source, destination in edges
        if source != destination
    })
    graph_content = "".join(
        f"{source} {destination}\n"
        for source, destination in source_edges
    )
    mapping_content = "\n".join(
        str(logical_to_source[logical])
        for logical in reference_order
    ) + "\n"
    role_by_source = [
        roles[source_to_logical[source]]
        for source in range(spec.nodes)
    ]
    group_by_source = [
        logical_groups[source_to_logical[source]]
        for source in range(spec.nodes)
    ]
    vertex_content = json.dumps({
        "schema": "graphbrew-synthetic-vertices/v1",
        "graph": spec.name,
        "source_to_logical": source_to_logical,
        "role_by_source": role_by_source,
        "group_by_source": group_by_source,
    }, separators=(",", ":")) + "\n"

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_exist = (
        graph_path.is_file()
        and mapping_path.is_file()
        and vertex_path.is_file()
    )
    if artifacts_exist and not refreeze:
        expected = (
            (graph_path, graph_content),
            (mapping_path, mapping_content),
            (vertex_path, vertex_content),
        )
        if any(path.read_text() != content for path, content in expected):
            raise RuntimeError(
                f"Frozen synthetic artifact changed: {spec.name}")
    else:
        _atomic_text(graph_path, graph_content)
        _atomic_text(mapping_path, mapping_content)
        _atomic_text(vertex_path, vertex_content)

    metadata = _artifact_metadata(
        spec,
        graph_path,
        mapping_path,
        vertex_path,
        len(source_edges),
    )
    if metadata_path.is_file() and not refreeze:
        if not _metadata_matches(metadata_path, metadata):
            raise RuntimeError(
                f"Frozen synthetic artifact changed: {spec.name}")
    else:
        _atomic_text(
            metadata_path,
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        )

    return SyntheticGraphArtifact(
        spec=spec,
        graph_path=graph_path,
        reference_mapping_path=mapping_path,
        vertex_metadata_path=vertex_path,
        metadata_path=metadata_path,
        undirected_edges=len(source_edges),
    )


def generate_screen_graphs(
    graph_root: Path,
    *,
    refreeze: bool = False,
    specs: Iterable[SyntheticGraphSpec] | None = None,
) -> tuple[SyntheticGraphArtifact, ...]:
    selected = tuple(specs or mechanism_discovery_screen_specs())
    if len(selected) > 48:
        raise ValueError("Mechanism-discovery configuration cap exceeded")
    return tuple(
        generate_synthetic_graph(
            spec,
            graph_root,
            refreeze=refreeze,
        )
        for spec in selected
    )


__all__ = [
    "SCREEN_FAMILIES",
    "SCREEN_NODE_COUNTS",
    "SCREEN_SEEDS",
    "SYNTHETIC_GRAPH_SCHEMA",
    "SYNTHETIC_POLICY_ID",
    "SyntheticGraphArtifact",
    "SyntheticGraphSpec",
    "generate_screen_graphs",
    "generate_synthetic_graph",
    "mechanism_discovery_screen_specs",
]
