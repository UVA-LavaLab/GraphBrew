"""Route-F mapping-forensics binary and composition contracts."""

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest

from scripts.lib.analysis.mapping_forensics import (
    CLASS_BANK_SHA256,
    CLASS_PREDICATES,
    LAYOUT_GORDER,
    LAYOUT_INPUT,
    LAYOUT_RABBIT_DRAWS,
    GraphArtifactSet,
    MappingArtifact,
    SerializedGraphMMap,
    analyze_graph_artifacts,
    compute_vertex_feature_codes,
    dbg_bucket_codes,
    load_text_mapping_positions,
    nominate_class,
    positive_bit_cost,
    sampled_distinct_lines_per_degree,
    freeze_artifact_manifest,
    validate_dbg_semantics,
)


def _write_sg(
    path: Path,
    *,
    nodes: int,
    source_edges: list[tuple[int, int]],
    org_ids: list[int],
) -> None:
    source_to_sg = [-1] * nodes
    for sg_id, source_id in enumerate(org_ids):
        source_to_sg[source_id] = sg_id
    adjacency = [[] for _ in range(nodes)]
    for source, destination in source_edges:
        sg_source = source_to_sg[source]
        sg_destination = source_to_sg[destination]
        adjacency[sg_source].append(sg_destination)
        adjacency[sg_destination].append(sg_source)
    offsets = [0]
    neighbors = []
    for row in adjacency:
        neighbors.extend(sorted(row))
        offsets.append(len(neighbors))
    payload = bytearray()
    payload.extend(struct.pack("<?qq", False, len(neighbors), nodes))
    payload.extend(np.asarray(offsets, dtype="<i8").tobytes())
    payload.extend(np.asarray(neighbors, dtype="<i4").tobytes())
    payload.extend(np.asarray(org_ids, dtype="<i4").tobytes())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_mapping(path: Path, new_to_source: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(map(str, new_to_source)) + "\n")


def _sidecar(
    path: Path,
    *,
    graph: str,
    algorithm: str,
    nodes: int,
    directed_edges: int,
    draws: int,
    origin: str | None = None,
) -> None:
    path_by_algorithm = {
        "5": "5.lo",
        "8:csr": "8_csr.lo",
        "9:csr": "9_csr.lo",
    }
    payload = {
        "schema": "reorder_meta/v4",
        "graph": graph,
        "algo_key": algorithm,
        "graph_info": {
            "directed": False,
            "edges": directed_edges,
            "nodes": nodes,
        },
        "mapping_draw_count": draws,
        "mapping_draws": [
            {
                "draw": draw,
                "path": (
                    f"8_csr.draw{draw}.lo"
                    if algorithm == "8:csr"
                    else path_by_algorithm[algorithm]
                ),
            }
            for draw in range(draws)
        ],
        "selected_draw": 0,
        "lo_path": path_by_algorithm[algorithm],
    }
    if origin is not None:
        payload["mapping_origin"] = origin
    path.write_text(json.dumps(payload))


def _fixture_artifacts(tmp_path: Path) -> GraphArtifactSet:
    graph = "hollywood-2009"
    nodes = 16
    org_ids = [7, 3, 14, 0, 12, 5, 9, 1, 15, 6, 2, 10, 4, 13, 8, 11]
    edges = [(vertex, vertex + 1) for vertex in range(nodes - 1)]
    edges.extend((0, vertex) for vertex in range(2, nodes))
    sg_path = tmp_path / "graphs" / graph / f"{graph}.sg"
    _write_sg(
        sg_path,
        nodes=nodes,
        source_edges=edges,
        org_ids=org_ids,
    )
    mapping_dir = tmp_path / "mappings" / graph
    mapping_dir.mkdir(parents=True)
    with SerializedGraphMMap(sg_path) as sg:
        degrees = sg.degrees()
        buckets = dbg_bucket_codes(degrees)
        sg_order = sorted(
            range(nodes),
            key=lambda vertex: (-int(buckets[vertex]), vertex),
        )
        dbg_new_to_source = [org_ids[vertex] for vertex in sg_order]
    _write_mapping(mapping_dir / "5.lo", dbg_new_to_source)
    source_order = list(range(nodes))
    for draw in range(3):
        _write_mapping(
            mapping_dir / f"8_csr.draw{draw}.lo",
            source_order if draw != 1 else list(reversed(source_order)),
        )
    shutil.copyfile(
        mapping_dir / "8_csr.draw0.lo",
        mapping_dir / "8_csr.lo",
    )
    _write_mapping(mapping_dir / "9_csr.lo", source_order)
    directed_edges = 2 * len(edges)
    _sidecar(
        mapping_dir / "5.json",
        graph=graph,
        algorithm="5",
        nodes=nodes,
        directed_edges=directed_edges,
        draws=1,
    )
    _sidecar(
        mapping_dir / "8_csr.json",
        graph=graph,
        algorithm="8:csr",
        nodes=nodes,
        directed_edges=directed_edges,
        draws=3,
    )
    _sidecar(
        mapping_dir / "9_csr.json",
        graph=graph,
        algorithm="9:csr",
        nodes=nodes,
        directed_edges=directed_edges,
        draws=1,
        origin="promoted-mapping-equivalent-legacy-gorder",
    )
    equivalence = tmp_path / "equivalence" / graph / "9_csr.equivalence.json"
    equivalence.parent.mkdir(parents=True)
    live_equivalent = equivalence.parent / "9_csr.live.lo"
    shutil.copyfile(mapping_dir / "9_csr.lo", live_equivalent)
    equivalence.write_text(json.dumps({
        "schema": "mapping_equivalence/v1",
        "graph": graph,
        "algorithm": "9:csr",
        "live_path": str(live_equivalent.resolve()),
        "promoted_path": str((mapping_dir / "9_csr.lo").resolve()),
        "live_bytes": live_equivalent.stat().st_size,
        "promoted_bytes": (mapping_dir / "9_csr.lo").stat().st_size,
        "equal": True,
    }))
    return GraphArtifactSet(
        graph=graph,
        graph_type="collaboration",
        sg_path=sg_path,
        dbg=MappingArtifact(
            "5", mapping_dir / "5.lo", mapping_dir / "5.json", 0),
        rabbit_draws=tuple(
            MappingArtifact(
                f"8:csr#draw{draw}",
                mapping_dir / f"8_csr.draw{draw}.lo",
                mapping_dir / "8_csr.json",
                draw,
            )
            for draw in range(3)
        ),
        rabbit_alias=mapping_dir / "8_csr.lo",
        rabbit_sidecar=mapping_dir / "8_csr.json",
        gorder=MappingArtifact(
            "9:csr",
            mapping_dir / "9_csr.lo",
            mapping_dir / "9_csr.json",
            0,
        ),
        gorder_equivalence=equivalence,
        gorder_live_equivalent=live_equivalent,
    )


def test_sg_mmap_layout_and_edge_iteration(tmp_path):
    path = tmp_path / "tiny.sg"
    _write_sg(
        path,
        nodes=4,
        source_edges=[(0, 1), (1, 2), (2, 3)],
        org_ids=[2, 0, 3, 1],
    )
    with SerializedGraphMMap(path) as graph:
        assert graph.nodes == 4
        assert graph.directed_edges == 6
        assert graph.undirected_edges == 3
        assert graph.org_ids.tolist() == [2, 0, 3, 1]
        chunks = list(graph.iter_edge_chunks(target_edges=2))
        edges = {
            tuple(sorted((int(source), int(destination))))
            for sources, destinations in chunks
            for source, destination in zip(sources, destinations)
        }
        assert len(edges) == 3


def test_mapping_composes_source_ids_through_org_ids(tmp_path):
    path = tmp_path / "tiny.sg"
    org_ids = [2, 0, 3, 1]
    _write_sg(
        path,
        nodes=4,
        source_edges=[(0, 1), (1, 2), (2, 3)],
        org_ids=org_ids,
    )
    mapping = tmp_path / "source-order.lo"
    _write_mapping(mapping, [0, 1, 2, 3])
    with SerializedGraphMMap(path) as graph:
        positions, metadata = load_text_mapping_positions(
            mapping,
            nodes=4,
            org_ids=graph.org_ids,
        )
    assert positions.tolist() == org_ids
    assert positions.tolist() != [0, 1, 2, 3]
    assert metadata["composed_sg_to_new_fingerprint"].startswith(
        "forensic-int32-sha256:")


def test_dbg_semantics_rejects_bucket_interleaving():
    degrees = np.array([100, 1, 50, 2], dtype=np.int32)
    valid = np.array([0, 3, 1, 2], dtype=np.int32)
    assert validate_dbg_semantics(valid, degrees)["valid"]
    invalid = np.array([0, 1, 2, 3], dtype=np.int32)
    with pytest.raises(ValueError, match="bucket order"):
        validate_dbg_semantics(invalid, degrees)


def test_positive_bit_cost():
    gaps = np.array([0, 1, 2, 3, 4, 8, 9], dtype=np.int64)
    assert positive_bit_cost(gaps).tolist() == [1, 1, 2, 2, 3, 4, 4]


def test_class_bank_is_frozen():
    assert len(CLASS_PREDICATES) == 53
    assert len(CLASS_BANK_SHA256) == 64
    assert len({predicate.name for predicate in CLASS_PREDICATES}) == 53
    assert any(
        predicate.scheme == "clustering"
        for predicate in CLASS_PREDICATES
    )
    assert any(
        predicate.scheme == "core"
        and predicate.detector_work == "O(m)"
        for predicate in CLASS_PREDICATES
    )


def test_analyze_graph_artifacts_end_to_end(tmp_path):
    artifacts = _fixture_artifacts(tmp_path)
    result = analyze_graph_artifacts(artifacts)
    assert result["measurement_mode"] == "diagnostic-forensic"
    assert result["claim_eligible"] is False
    assert result["artifact_identity"]["legacy_forensic"] is True
    assert result["dbg_validation"]["valid"] is True
    assert set(result["layout_metrics"]) == {
        LAYOUT_INPUT,
        "SOURCE-ID-DIAGNOSTIC",
        "5",
        *LAYOUT_RABBIT_DRAWS,
        LAYOUT_GORDER,
    }
    assert len(result["class_metrics"]) == 53
    assert result["m3_distinct_lines"]["diagnostic_only"] is True
    assert result["m3_distinct_lines"]["bootstrap_replicates"] == 1024
    assert len(
        result["m3_distinct_lines"]["layouts"][LAYOUT_GORDER][0][
            "bucket_sums"]
    ) == 256
    assert result["post_input_verification"] == "pass"
    assert set(result["post_layout_fingerprints"]) == set(
        result["layout_metrics"])
    assert result["feature_metadata"]["self_loop_entries_observed"] == 0
    assert all(
        len(row["rabbit_u64_draws"]) == 3
        for row in result["class_metrics"]
    )
    assert all(
        len(row["rabbit_gorder_disagreement_draws"]) == 3
        and len(row["rabbit_pair_disagreement_rates"]) == 3
        for row in result["class_metrics"]
    )
    assert result["artifact_identity"]["rabbit_draws_distinct"] is False
    assert result["rabbit_pair_disagreement_max"] == 0.0
    assert result["layout_metrics"][LAYOUT_GORDER][
        "mean_positive_bit_mloga"
    ] < result["layout_metrics"][LAYOUT_INPUT][
        "mean_positive_bit_mloga"
    ]
    assert result["layout_metrics"][LAYOUT_GORDER][
        "mean_positive_bit_mloga"
    ] == pytest.approx(63 / 29)
    assert result["layout_metrics"][LAYOUT_GORDER][
        "same_line_fraction"
    ] == pytest.approx(20 / 29)


def test_feature_pass_reports_and_excludes_self_loops(tmp_path):
    path = tmp_path / "loops.sg"
    _write_sg(
        path,
        nodes=4,
        source_edges=[(0, 1), (1, 1), (1, 2), (2, 3)],
        org_ids=[0, 1, 2, 3],
    )
    with SerializedGraphMMap(path) as graph:
        degrees = graph.degrees()
        _codes, metadata, _edges, feature_degrees = (
            compute_vertex_feature_codes(graph, degrees)
        )
    assert metadata["self_loop_entries_observed"] == 2
    assert metadata["undirected_edges_scanned"] == 3
    assert metadata["directed_edge_identity"] == "2*3+2=8"
    assert feature_degrees.tolist() == [1, 2, 2, 1]


def test_m3_uses_loop_clean_degree(tmp_path):
    path = tmp_path / "loop-cycle.sg"
    _write_sg(
        path,
        nodes=4,
        source_edges=[
            (0, 1), (1, 2), (2, 3), (3, 0),
            (0, 0), (1, 1), (2, 2), (3, 3),
        ],
        org_ids=[0, 1, 2, 3],
    )
    with SerializedGraphMMap(path) as graph:
        raw_degrees = graph.degrees()
        codes, _metadata, _edges, feature_degrees = (
            compute_vertex_feature_codes(graph, raw_degrees)
        )
        m3 = sampled_distinct_lines_per_degree(
            graph,
            feature_degrees,
            {"identity": np.arange(4, dtype=np.int32)},
            codes["degree"],
            sample_limit=4,
        )
    populated = [
        row for row in m3["layouts"]["identity"]
        if row["sample_count"]
    ]
    assert len(populated) == 1
    assert populated[0]["mean_distinct_lines_per_degree"] == 0.5


def test_alias_mismatch_fails_closed(tmp_path):
    artifacts = _fixture_artifacts(tmp_path)
    artifacts.rabbit_alias.write_text(
        "\n".join(map(str, reversed(range(16)))) + "\n")
    with pytest.raises(ValueError, match="alias differs"):
        analyze_graph_artifacts(artifacts)


def test_duplicate_mapping_fails_closed(tmp_path):
    artifacts = _fixture_artifacts(tmp_path)
    artifacts.rabbit_draws[1].path.write_text(
        "\n".join(["0"] * 16) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        analyze_graph_artifacts(artifacts)


def test_false_gorder_equivalence_fails_closed(tmp_path):
    artifacts = _fixture_artifacts(tmp_path)
    payload = json.loads(artifacts.gorder_equivalence.read_text())
    payload["equal"] = False
    artifacts.gorder_equivalence.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="equivalence evidence"):
        analyze_graph_artifacts(artifacts)


def test_manifest_tripwire_precedes_parse(tmp_path):
    artifacts = _fixture_artifacts(tmp_path)
    manifest = freeze_artifact_manifest(artifacts)
    artifacts.sg_path.write_bytes(b"broken")
    with pytest.raises(RuntimeError, match="changed after freeze"):
        analyze_graph_artifacts(
            artifacts, input_manifest=manifest)


def test_asymmetric_multiplicity_fails_closed(tmp_path):
    path = tmp_path / "asymmetric.sg"
    offsets = np.array([0, 1, 2, 4], dtype="<i8")
    neighbors = np.array([1, 2, 0, 1], dtype="<i4")
    payload = bytearray(struct.pack("<?qq", False, 4, 3))
    payload.extend(offsets.tobytes())
    payload.extend(neighbors.tobytes())
    payload.extend(np.arange(3, dtype="<i4").tobytes())
    path.write_bytes(payload)
    with SerializedGraphMMap(path) as graph:
        with pytest.raises(ValueError, match="reciprocal"):
            compute_vertex_feature_codes(
                graph, graph.degrees())


def test_nomination_requires_exact_discovery_cohort():
    with pytest.raises(ValueError, match="exactly the discovery"):
        nominate_class([{
            "graph": "cit-Patents",
            "graph_type": "citation",
            "layout_metrics": {},
            "class_metrics": [],
        }])


def test_zero_rabbit_pair_control_is_valid():
    class_row = {
        "class_id": 0,
        "class_name": "degree:q0",
        "support_fraction": 0.1,
        "rabbit_gorder_disagreement_range": {
            "min": 0.5, "median": 0.5, "max": 0.5,
        },
        "rabbit_pair_disagreement_max": 0.0,
        "rabbit_beyond_gap64_bit_fraction_range": {
            "min": 0.5, "median": 0.5, "max": 0.5,
        },
        "gorder_beyond_gap64_bit_fraction": 0.5,
        "rabbit_u64_range": {
            "min": 0.2, "median": 0.2, "max": 0.2,
        },
        "gorder_u64": 0.2,
        "rabbit_excess64_per_class_edge_range": {
            "min": 1.0, "median": 1.0, "max": 1.0,
        },
        "gorder_excess64_per_class_edge": 1.0,
    }
    rows = []
    for graph in (
        "cit-Patents",
        "soc-pokec",
        "USA-road-d.USA",
        "soc-LiveJournal1",
        "delaunay_n24",
        "com-Orkut",
        "wikipedia_link_en",
        "Gong-gplus",
    ):
        rows.append({
            "graph": graph,
            "graph_type": {
                "USA-road-d.USA": "road",
                "delaunay_n24": "mesh",
            }.get(graph, "social"),
            "layout_metrics": {
                LAYOUT_INPUT: {"mean_positive_bit_mloga": 3.0},
                "8:csr#draw0": {"mean_positive_bit_mloga": 2.0},
                "8:csr#draw1": {"mean_positive_bit_mloga": 2.0},
                "8:csr#draw2": {"mean_positive_bit_mloga": 2.0},
                LAYOUT_GORDER: {"mean_positive_bit_mloga": 2.0},
            },
            "class_metrics": [dict(class_row)],
        })
    result = nominate_class(rows)
    assert result["status"] == "nominee"
    assert result["nominee"]["class_name"] == "degree:q0"
    json.dumps(result)
