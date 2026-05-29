"""Pytest gate for scripts/experiments/ecg/corpus_diversity.py.

Verifies that:
  * the GAPBS topology block can be parsed from a synthetic log fixture
  * the markdown / csv / json writers produce the expected schema
  * the helper survives logs that lack the topology block
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ECG_DIR = REPO_ROOT / "scripts" / "experiments" / "ecg"


def _load(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ECG_DIR / file_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


cd = _load("corpus_diversity", "corpus_diversity.py")

FAKE_LOG = textwrap.dedent(
    """\
    Read Time:           0.20
    === Graph Topology Features ===
    Clustering Coefficient:0.12345
    Avg Path Length:     2.50000
    Diameter Estimate:   4.00000
    Community Count Estimate:42
    Degree Variance:     1.5
    Hub Concentration:   0.55
    Avg Degree:          20.0
    Graph Density:       0.0001
    Modularity:          0.6
    Forward Edge Fraction:0.5
    Working Set Ratio:   3.14
    Vertex Significance Skewness:2.0
    Window Neighbor Overlap:0.07
    Sampled Locality Score:0.05
    ===============================
    Graph has 12345 nodes and 67890 undirected edges for degree: 5 Estimated size: 1 MB
    """
)


def _make_sweep_tree(tmp_path: Path, graph: str, log_text: str = FAKE_LOG) -> Path:
    base = tmp_path / "sweep" / f"{graph}-pr" / "lit" / "logs"
    base.mkdir(parents=True)
    (base / "cache_sim_pr_LRU_L31MB.log").write_text(log_text)
    return tmp_path / "sweep"


def test_parse_log_extracts_topology_and_graph_card(tmp_path: Path) -> None:
    root = _make_sweep_tree(tmp_path, "test-graph")
    log = root / "test-graph-pr" / "lit" / "logs" / "cache_sim_pr_LRU_L31MB.log"
    profile = cd.parse_log(log)

    assert profile.nodes == 12345
    assert profile.edges == 67890
    assert profile.edges_directed is False
    assert profile.features["clustering_coeff"] == pytest.approx(0.12345)
    assert profile.features["community_count"] == 42
    assert profile.features["hub_concentration"] == pytest.approx(0.55)
    assert profile.features["avg_degree"] == pytest.approx(20.0)


def test_collect_finds_log_per_graph(tmp_path: Path) -> None:
    sweep_root = tmp_path / "sweep"
    sweep_root.mkdir()
    for g in ("alpha", "beta"):
        _make_sweep_tree(tmp_path, g)
    profiles = cd.collect(sweep_root, "lit", ["alpha", "beta", "missing-graph"])
    assert [p.graph for p in profiles] == ["alpha", "beta"]


def test_writers_emit_consistent_csv_and_markdown(tmp_path: Path) -> None:
    sweep_root = tmp_path / "sweep"
    sweep_root.mkdir()
    _make_sweep_tree(tmp_path, "alpha")
    profiles = cd.collect(sweep_root, "lit", ["alpha"])

    csv_path = tmp_path / "out.csv"
    md_path = tmp_path / "out.md"
    json_path = tmp_path / "out.json"
    cd.write_csv(profiles, csv_path)
    cd.write_markdown(profiles, md_path)
    json_path.write_text(json.dumps([p.__dict__ for p in profiles], default=str))

    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["graph"] == "alpha"
    assert rows[0]["nodes"] == "12345"
    assert rows[0]["clustering_coeff"] == "0.12345"

    text = md_path.read_text()
    assert "Corpus Diversity Profile" in text
    assert "`alpha`" in text
    assert "Hub Concentration" in text


def test_missing_block_does_not_crash(tmp_path: Path) -> None:
    sweep_root = _make_sweep_tree(tmp_path, "no-features", log_text="hello world")
    profile = cd.parse_log(
        sweep_root / "no-features-pr" / "lit" / "logs" / "cache_sim_pr_LRU_L31MB.log"
    )
    assert profile.features == {}
    assert profile.nodes == 0


def test_real_sweep_when_present() -> None:
    """If the literature sweep is on disk, verify we can parse it end-to-end."""

    sweep_root = Path("/tmp/graphbrew-lit-baseline")
    if not sweep_root.is_dir():
        pytest.skip("literature sweep not on this host")
    profiles = cd.collect(sweep_root, "lit", cd.GRAPH_ORDER)
    if not profiles:
        pytest.skip("no graphs from literature corpus present")
    for p in profiles:
        assert p.graph in cd.GRAPH_ORDER
        assert p.nodes > 0, f"{p.graph} did not parse a node count"
        # Every literature sweep PR / LRU log carries the topology block.
        assert "avg_degree" in p.features, f"{p.graph} missing avg_degree"
