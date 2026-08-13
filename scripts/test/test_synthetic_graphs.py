"""Synthetic mechanism-discovery graph contracts."""

from __future__ import annotations

import json

import pytest

from scripts.lib.pipeline.benchmark import (
    mapping_permutation_fingerprint,
)
from scripts.lib.pipeline.synthetic_graphs import (
    _FAMILY_BUILDERS,
    SCREEN_FAMILIES,
    SyntheticGraphSpec,
    generate_synthetic_graph,
    mechanism_discovery_screen_specs,
)


def _mapping(path):
    return [int(value) for value in path.read_text().split()]


def test_screen_matrix_is_bounded_and_complete():
    specs = mechanism_discovery_screen_specs()
    assert len(specs) == 42
    assert {spec.family for spec in specs} == set(SCREEN_FAMILIES)
    assert len({spec.name for spec in specs}) == len(specs)


@pytest.mark.parametrize("family", SCREEN_FAMILIES)
def test_synthetic_family_is_deterministic_and_permuted(
    tmp_path, family,
):
    nodes = 64
    spec = SyntheticGraphSpec(family, nodes, 1)
    artifact = generate_synthetic_graph(spec, tmp_path)
    repeated = generate_synthetic_graph(spec, tmp_path)

    assert artifact == repeated
    mapping = _mapping(artifact.reference_mapping_path)
    assert sorted(mapping) == list(range(nodes))
    assert mapping != list(range(nodes))
    assert artifact.undirected_edges > 0
    assert mapping_permutation_fingerprint(
        artifact.reference_mapping_path
    )

    metadata = json.loads(artifact.metadata_path.read_text())
    vertices = json.loads(artifact.vertex_metadata_path.read_text())
    assert metadata["claim_eligible"] is False
    assert metadata["measurement_mode"] == "diagnostic-synthetic"
    assert metadata["nodes"] == nodes
    assert len(vertices["source_to_logical"]) == nodes
    assert len(vertices["role_by_source"]) == nodes
    assert len(vertices["group_by_source"]) == nodes


def test_grid_requires_square_size(tmp_path):
    with pytest.raises(ValueError, match="perfect squares"):
        generate_synthetic_graph(
            SyntheticGraphSpec("grid", 63, 0),
            tmp_path,
        )


def test_expander_rejects_unrealizable_degree(tmp_path):
    with pytest.raises(ValueError, match="too small"):
        generate_synthetic_graph(
            SyntheticGraphSpec("expander-control", 7, 0),
            tmp_path,
        )


def test_frozen_artifact_detects_content_change(tmp_path):
    spec = SyntheticGraphSpec("chain", 64, 0)
    artifact = generate_synthetic_graph(spec, tmp_path)
    artifact.graph_path.write_text("0 1\n")
    with pytest.raises(RuntimeError, match="Frozen synthetic"):
        generate_synthetic_graph(spec, tmp_path)


def test_frozen_artifact_detects_generator_drift(
    tmp_path, monkeypatch,
):
    spec = SyntheticGraphSpec("chain", 64, 0)
    generate_synthetic_graph(spec, tmp_path)
    original_builder = _FAMILY_BUILDERS["chain"]

    def changed_builder(changed_spec):
        edges, reference, roles, groups = original_builder(changed_spec)
        return edges, list(reversed(reference)), roles, groups

    monkeypatch.setitem(
        _FAMILY_BUILDERS, "chain", changed_builder)
    with pytest.raises(RuntimeError, match="Frozen synthetic"):
        generate_synthetic_graph(spec, tmp_path)
