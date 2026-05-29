"""Confidence gate 102 — corpus_diversity ↔ winning_regime_taxonomy
feature-level and family-classification parity.

The regime taxonomy is downstream of corpus_diversity in two distinct
ways: it copies per-graph structural features (avg_degree,
hub_concentration, clustering_coeff) into every cell as decorative
columns, AND it assigns each cell a ``family`` label that must come
from the GRAPH_FAMILY map in policy_winner_table.py. If either link
drifts — copied features become stale, family map gets a new graph
without re-baking the taxonomy, corpus_diversity gains a graph the
taxonomy doesn't cover — the paper's "regime determines policy" claim
starts riding on inconsistent feature values.

This gate locks 13 invariants that keep the two artifacts aligned.

Invariants (4 / 4 / 3 / 2):

  corpus_diversity internal (4):
    1. exactly len(EXPECTED_CORPUS_GRAPHS) entries (8)
    2. every entry has the 6 required top-level keys
       (graph/log_path/nodes/edges/edges_directed/features) and the
       14-key features schema
    3. nodes/edges are positive ints, all 14 features are finite floats
    4. graph names are unique across the list

  WRT cell ↔ corpus feature parity (4):
    5. WRT cell avg_degree (string-parsed) matches
       corpus_diversity[graph].features.avg_degree within 1e-3 for
       all 114 cells
    6. same for hub_concentration
    7. same for clustering_coeff
    8. every WRT cell.family equals GRAPH_FAMILY[cell.graph] (loaded
       from scripts/experiments/ecg/policy_winner_table.py)

  Graph universe agreement (3):
    9. corpus_diversity graph set == WRT graph set (both 8 graphs)
   10. WRT.cells apps ⊆ EXPECTED_APPS ({bc, bfs, cc, pr, sssp})
   11. WRT.cells l3_size values ⊆ EXPECTED_L3_SIZES (the 7-point
       cache-sim axis: 4kB / 16kB / 64kB / 256kB / 1MB / 4MB / 8MB)

  Family math (2):
   12. counting cells per family in WRT.cells reproduces the same
       totals as summary.by_family_regime aggregated over regime
   13. all corpus_diversity graphs are listed in GRAPH_FAMILY (no
       graph in the corpus is missing a family classification)
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COR_PATH = PROJECT_ROOT / "wiki" / "data" / "corpus_diversity.json"
WRT_PATH = PROJECT_ROOT / "wiki" / "data" / "winning_regime_taxonomy.json"

EXPECTED_CORPUS_GRAPHS = frozenset({
    "email-Eu-core", "web-Google", "cit-Patents", "soc-pokec",
    "soc-LiveJournal1", "com-orkut", "roadNet-CA", "delaunay_n19",
})
COR_ENTRY_KEYS = frozenset({
    "graph", "log_path", "nodes", "edges", "edges_directed", "features",
})
COR_FEATURE_KEYS = frozenset({
    "avg_degree", "avg_path_len", "clustering_coeff", "community_count",
    "degree_variance", "diameter_estimate", "forward_edge_fraction",
    "graph_density", "hub_concentration", "modularity",
    "sampled_locality_score", "vertex_sig_skew",
    "window_neighbor_overlap", "working_set_ratio",
})
EXPECTED_APPS = frozenset({"bc", "bfs", "cc", "pr", "sssp"})
EXPECTED_L3_SIZES = frozenset({"4kB", "16kB", "64kB", "256kB", "1MB", "4MB", "8MB"})
FEATURE_TOL = 1e-3


@pytest.fixture(scope="module")
def corpus() -> list[dict]:
    assert COR_PATH.exists(), f"missing corpus_diversity.json at {COR_PATH}"
    return json.loads(COR_PATH.read_text())


@pytest.fixture(scope="module")
def wrt() -> dict:
    assert WRT_PATH.exists(), f"missing winning_regime_taxonomy.json at {WRT_PATH}"
    return json.loads(WRT_PATH.read_text())


@pytest.fixture(scope="module")
def graph_family() -> dict:
    ecg_dir = PROJECT_ROOT / "scripts" / "experiments" / "ecg"
    if str(ecg_dir) not in sys.path:
        sys.path.insert(0, str(ecg_dir))
    from policy_winner_table import GRAPH_FAMILY  # type: ignore[import-not-found]
    return GRAPH_FAMILY


# ---------------------------------------------------------------------------
# corpus_diversity internal (4)
# ---------------------------------------------------------------------------


def test_corpus_count_matches_expected(corpus: list[dict]) -> None:
    n = len(corpus)
    expected = len(EXPECTED_CORPUS_GRAPHS)
    assert n == expected, f"corpus_diversity entry count {n} != expected {expected}"
    graphs = {c["graph"] for c in corpus}
    assert graphs == EXPECTED_CORPUS_GRAPHS, (
        f"corpus_diversity graph set != expected; "
        f"missing={EXPECTED_CORPUS_GRAPHS - graphs}, extra={graphs - EXPECTED_CORPUS_GRAPHS}"
    )


def test_corpus_entry_schema(corpus: list[dict]) -> None:
    bad: list[tuple[str, list[str], list[str]]] = []
    for c in corpus:
        top_keys = set(c.keys())
        if top_keys != COR_ENTRY_KEYS:
            bad.append((c.get("graph", "?"),
                        sorted(top_keys - COR_ENTRY_KEYS),
                        sorted(COR_ENTRY_KEYS - top_keys)))
            continue
        feat_keys = set(c["features"].keys())
        if feat_keys != COR_FEATURE_KEYS:
            bad.append((c["graph"],
                        sorted(feat_keys - COR_FEATURE_KEYS),
                        sorted(COR_FEATURE_KEYS - feat_keys)))
    assert not bad, f"corpus_diversity schema violations (graph, extra, missing): {bad}"


def test_corpus_values_well_typed(corpus: list[dict]) -> None:
    bad: list[tuple[str, str, object]] = []
    for c in corpus:
        if not (isinstance(c["nodes"], int) and c["nodes"] > 0):
            bad.append((c["graph"], "nodes", c["nodes"]))
        if not (isinstance(c["edges"], int) and c["edges"] > 0):
            bad.append((c["graph"], "edges", c["edges"]))
        for k, v in c["features"].items():
            if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                bad.append((c["graph"], f"features.{k}", v))
    assert not bad, f"corpus_diversity ill-typed values: {bad}"


def test_corpus_graph_names_unique(corpus: list[dict]) -> None:
    names = [c["graph"] for c in corpus]
    seen: dict[str, int] = {}
    for n in names:
        seen[n] = seen.get(n, 0) + 1
    dupes = [(k, v) for k, v in seen.items() if v > 1]
    assert not dupes, f"corpus_diversity has duplicate graph names: {dupes}"


# ---------------------------------------------------------------------------
# WRT cell ↔ corpus feature parity (4)
# ---------------------------------------------------------------------------


def _feature_map(corpus: list[dict]) -> dict[str, dict]:
    return {c["graph"]: c["features"] for c in corpus}


def test_wrt_avg_degree_matches_corpus(corpus: list[dict], wrt: dict) -> None:
    fm = _feature_map(corpus)
    bad: list[tuple[str, float, float]] = []
    for cell in wrt["cells"]:
        g = cell["graph"]
        if g not in fm:
            continue
        cell_v = float(cell["avg_degree"])
        cor_v = float(fm[g]["avg_degree"])
        if not math.isclose(cell_v, cor_v, abs_tol=FEATURE_TOL):
            bad.append((g, cell_v, cor_v))
    assert not bad, f"WRT cell.avg_degree drift from corpus_diversity: {bad}"


def test_wrt_hub_concentration_matches_corpus(corpus: list[dict], wrt: dict) -> None:
    fm = _feature_map(corpus)
    bad: list[tuple[str, float, float]] = []
    for cell in wrt["cells"]:
        g = cell["graph"]
        if g not in fm:
            continue
        cell_v = float(cell["hub_concentration"])
        cor_v = float(fm[g]["hub_concentration"])
        if not math.isclose(cell_v, cor_v, abs_tol=FEATURE_TOL):
            bad.append((g, cell_v, cor_v))
    assert not bad, f"WRT cell.hub_concentration drift from corpus_diversity: {bad}"


def test_wrt_clustering_coeff_matches_corpus(corpus: list[dict], wrt: dict) -> None:
    fm = _feature_map(corpus)
    bad: list[tuple[str, float, float]] = []
    for cell in wrt["cells"]:
        g = cell["graph"]
        if g not in fm:
            continue
        cell_v = float(cell["clustering_coeff"])
        cor_v = float(fm[g]["clustering_coeff"])
        if not math.isclose(cell_v, cor_v, abs_tol=FEATURE_TOL):
            bad.append((g, cell_v, cor_v))
    assert not bad, f"WRT cell.clustering_coeff drift from corpus_diversity: {bad}"


def test_wrt_family_matches_classifier(wrt: dict, graph_family: dict) -> None:
    bad: list[tuple[str, str, str | None]] = []
    for cell in wrt["cells"]:
        expected = graph_family.get(cell["graph"])
        if cell["family"] != expected:
            bad.append((cell["graph"], cell["family"], expected))
    assert not bad, (
        f"WRT cell.family does not match GRAPH_FAMILY classifier: {bad}"
    )


# ---------------------------------------------------------------------------
# Graph universe agreement (3)
# ---------------------------------------------------------------------------


def test_corpus_and_wrt_share_same_graph_universe(
    corpus: list[dict], wrt: dict
) -> None:
    cor_graphs = {c["graph"] for c in corpus}
    wrt_graphs = {c["graph"] for c in wrt["cells"]}
    assert cor_graphs == wrt_graphs, (
        f"graph universe mismatch: only_in_corpus={cor_graphs - wrt_graphs}, "
        f"only_in_wrt={wrt_graphs - cor_graphs}"
    )


def test_wrt_apps_subset_of_expected(wrt: dict) -> None:
    seen = {c["app"] for c in wrt["cells"]}
    extra = seen - EXPECTED_APPS
    assert not extra, f"WRT.cells contains unexpected apps: {extra} (expected {sorted(EXPECTED_APPS)})"


def test_wrt_l3_sizes_subset_of_expected(wrt: dict) -> None:
    seen = {c["l3_size"] for c in wrt["cells"]}
    extra = seen - EXPECTED_L3_SIZES
    assert not extra, (
        f"WRT.cells contains unexpected l3_size: {extra} (expected {sorted(EXPECTED_L3_SIZES)})"
    )


# ---------------------------------------------------------------------------
# Family math (2)
# ---------------------------------------------------------------------------


def test_wrt_family_cell_count_matches_summary(wrt: dict) -> None:
    cells_per_family = Counter(c["family"] for c in wrt["cells"])
    summary_per_family: Counter = Counter()
    for fr in wrt["summary"]["by_family_regime"]:
        summary_per_family[fr["family"]] += fr["total"]
    bad = [
        (f, cells_per_family[f], summary_per_family[f])
        for f in set(cells_per_family) | set(summary_per_family)
        if cells_per_family[f] != summary_per_family[f]
    ]
    assert not bad, f"per-family cell count != by_family_regime summary totals: {bad}"


def test_every_corpus_graph_has_family_classification(
    corpus: list[dict], graph_family: dict
) -> None:
    missing = [c["graph"] for c in corpus if c["graph"] not in graph_family]
    assert not missing, (
        f"corpus_diversity graphs without GRAPH_FAMILY entry: {missing}"
    )
