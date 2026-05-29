"""Derivation-parity gate for ``wiki/data/claim_density.json``.

The claim-density artifact is the reviewer-facing breakdown of how many
literature claims register per graph, and what fraction of them are OK
today. It is the artifact reviewers ask "why does email-Eu-core only
have 19 claims?" against, so a silent change to the per-graph
aggregation (claim counts, OK fraction, status mix), the de-duplication
of cells/apps/policies/citations, or the summary roll-up
(total_claims / total_ok / total_ok_pct) would let coverage drift
without tripping any other gate.

This module re-derives the artifact end-to-end from
``literature_reproduction_summary.csv`` and asserts byte-for-byte
equivalence with the committed JSON, plus the load-bearing invariants
the generator enforces.

5 groups, 21 tests total.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = REPO_ROOT / "wiki" / "data" / "claim_density.json"
SOURCE = REPO_ROOT / "wiki" / "data" / "literature_reproduction_summary.csv"
GENERATOR = REPO_ROOT / "scripts" / "experiments" / "ecg" / "claim_density_report.py"


def _load_generator() -> Any:
    spec = importlib.util.spec_from_file_location("claim_density_local", GENERATOR)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_source_rows() -> list[dict[str, Any]]:
    with SOURCE.open(newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def artifact() -> dict[str, Any]:
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def regenerated() -> dict[str, Any]:
    gen = _load_generator()
    src_rows = _load_source_rows()
    rows = gen._per_graph(src_rows)
    summary = gen._summarise(rows)
    return {"schema_version": 1, "summary": summary, "graphs": rows}


# ----------------------------------------------------------------------
# Group 1: cross-source byte equivalence + scaffolding
# ----------------------------------------------------------------------

def test_regenerated_matches_committed_artifact(regenerated: dict[str, Any], artifact: dict[str, Any]) -> None:
    """Re-running _per_graph + _summarise on the same source must yield
    byte-identical JSON."""
    assert json.dumps(regenerated, sort_keys=True) == json.dumps(artifact, sort_keys=True)


def test_top_level_keys_exact(artifact: dict[str, Any]) -> None:
    assert set(artifact.keys()) == {"schema_version", "summary", "graphs"}
    assert artifact["schema_version"] == 1


def test_summary_keys_exact(artifact: dict[str, Any]) -> None:
    assert set(artifact["summary"].keys()) == {
        "n_graphs", "total_claims", "total_ok", "total_ok_pct",
    }


def test_graphs_sorted_alphabetically_by_graph(artifact: dict[str, Any]) -> None:
    names = [g["graph"] for g in artifact["graphs"]]
    assert names == sorted(names), f"graphs not sorted: {names}"


def test_every_graph_row_has_required_fields(artifact: dict[str, Any]) -> None:
    required = {
        "graph", "n_claims", "n_ok", "ok_pct", "status_counts",
        "n_cells", "apps", "n_apps", "policies", "n_policies",
        "citations", "n_citations",
    }
    for g in artifact["graphs"]:
        missing = required - set(g.keys())
        assert not missing, f"row missing fields {missing}: {g['graph']}"


# ----------------------------------------------------------------------
# Group 2: per-graph counts from source
# ----------------------------------------------------------------------

def test_n_claims_matches_source_row_count(artifact: dict[str, Any]) -> None:
    src_rows = _load_source_rows()
    counts: dict[str, int] = {}
    for r in src_rows:
        counts[r["graph"]] = counts.get(r["graph"], 0) + 1
    for g in artifact["graphs"]:
        assert g["n_claims"] == counts.get(g["graph"], 0), g["graph"]


def test_n_ok_matches_source_status_ok_count(artifact: dict[str, Any]) -> None:
    src_rows = _load_source_rows()
    for g in artifact["graphs"]:
        expected = sum(
            1 for r in src_rows if r["graph"] == g["graph"] and r["status"] == "ok"
        )
        assert g["n_ok"] == expected, g["graph"]


def test_ok_pct_matches_n_ok_over_n_claims(artifact: dict[str, Any]) -> None:
    for g in artifact["graphs"]:
        expected = (g["n_ok"] / g["n_claims"] * 100.0) if g["n_claims"] else 0.0
        assert g["ok_pct"] == expected, g["graph"]


def test_status_counts_matches_source_counter(artifact: dict[str, Any]) -> None:
    src_rows = _load_source_rows()
    for g in artifact["graphs"]:
        expected = dict(
            Counter(r["status"] for r in src_rows if r["graph"] == g["graph"])
        )
        assert g["status_counts"] == expected, g["graph"]


def test_status_counts_sums_to_n_claims(artifact: dict[str, Any]) -> None:
    for g in artifact["graphs"]:
        assert sum(g["status_counts"].values()) == g["n_claims"], g["graph"]


# ----------------------------------------------------------------------
# Group 3: per-graph dedup-+-sort fields (apps / policies / citations / cells)
# ----------------------------------------------------------------------

def test_apps_sorted_unique_and_n_apps_matches(artifact: dict[str, Any]) -> None:
    for g in artifact["graphs"]:
        assert g["apps"] == sorted(set(g["apps"])), g["graph"]
        assert g["n_apps"] == len(g["apps"]), g["graph"]


def test_policies_sorted_unique_and_n_policies_matches(artifact: dict[str, Any]) -> None:
    for g in artifact["graphs"]:
        assert g["policies"] == sorted(set(g["policies"])), g["graph"]
        assert g["n_policies"] == len(g["policies"]), g["graph"]


def test_citations_sorted_unique_and_n_citations_matches(artifact: dict[str, Any]) -> None:
    for g in artifact["graphs"]:
        assert g["citations"] == sorted(set(g["citations"])), g["graph"]
        assert g["n_citations"] == len(g["citations"]), g["graph"]


def test_n_cells_matches_source_unique_app_l3_pairs(artifact: dict[str, Any]) -> None:
    src_rows = _load_source_rows()
    for g in artifact["graphs"]:
        expected = len({
            (r["app"], r["l3_size"])
            for r in src_rows if r["graph"] == g["graph"]
        })
        assert g["n_cells"] == expected, g["graph"]


def test_apps_matches_source_unique_apps(artifact: dict[str, Any]) -> None:
    src_rows = _load_source_rows()
    for g in artifact["graphs"]:
        expected = sorted({r["app"] for r in src_rows if r["graph"] == g["graph"]})
        assert g["apps"] == expected, g["graph"]


def test_policies_matches_source_unique_policies(artifact: dict[str, Any]) -> None:
    src_rows = _load_source_rows()
    for g in artifact["graphs"]:
        expected = sorted({r["policy"] for r in src_rows if r["graph"] == g["graph"]})
        assert g["policies"] == expected, g["graph"]


def test_citations_dropna_filter_matches_source(artifact: dict[str, Any]) -> None:
    """Generator drops empty-string citations via the `if r.get('citation')`
    guard. Verify a row with empty citation in source would not appear."""
    src_rows = _load_source_rows()
    for g in artifact["graphs"]:
        expected = sorted({
            r["citation"]
            for r in src_rows
            if r["graph"] == g["graph"] and r.get("citation")
        })
        assert g["citations"] == expected, g["graph"]


# ----------------------------------------------------------------------
# Group 4: summary roll-up arithmetic
# ----------------------------------------------------------------------

def test_summary_n_graphs_matches_graphs_length(artifact: dict[str, Any]) -> None:
    assert artifact["summary"]["n_graphs"] == len(artifact["graphs"])


def test_summary_total_claims_matches_sum_of_per_graph_n_claims(artifact: dict[str, Any]) -> None:
    expected = sum(g["n_claims"] for g in artifact["graphs"])
    assert artifact["summary"]["total_claims"] == expected


def test_summary_total_ok_matches_sum_of_per_graph_n_ok(artifact: dict[str, Any]) -> None:
    expected = sum(g["n_ok"] for g in artifact["graphs"])
    assert artifact["summary"]["total_ok"] == expected


def test_summary_total_ok_pct_matches_total_ok_over_total_claims(artifact: dict[str, Any]) -> None:
    s = artifact["summary"]
    expected = (s["total_ok"] / s["total_claims"] * 100.0) if s["total_claims"] else 0.0
    assert s["total_ok_pct"] == expected


def test_summary_n_graphs_matches_source_distinct_graphs(artifact: dict[str, Any]) -> None:
    src_rows = _load_source_rows()
    expected = len({r["graph"] for r in src_rows})
    assert artifact["summary"]["n_graphs"] == expected
