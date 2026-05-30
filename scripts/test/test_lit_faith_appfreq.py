"""Lock the LIT-AppFreq invariants (gate 235).

Per-app axis-coverage on per_observation. Each app must touch enough
graphs / L3 sizes / policies / rows for the downstream comparators
(oracle-gap, cache-sensitivity slope, monotonicity, per-app stability)
to remain well-defined, and the anchor app (pr) must cover the full
corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT  = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "wiki" / "data" / "lit_faith_appfreq.json"
MD    = ROOT / "wiki" / "data" / "lit_faith_appfreq.md"
CSV   = ROOT / "wiki" / "data" / "lit_faith_appfreq.csv"


CANONICAL_ROSTER = {"LRU", "GRASP", "POPT"}
ANCHOR_APP       = "pr"


@pytest.fixture(scope="module")
def audit() -> dict:
    if not AUDIT.exists():
        pytest.skip("lit_faith_appfreq.json missing; run `make lit-appfreq`")
    return json.loads(AUDIT.read_text(encoding="utf-8"))


# --- Artifact presence + schema --------------------------------------------

def test_artifacts_present():
    assert AUDIT.exists(), f"missing {AUDIT}"
    assert MD.exists(),    f"missing {MD}"
    assert CSV.exists(),   f"missing {CSV}"


def test_schema_version(audit):
    assert audit["schema_version"] == 1


def test_top_level_keys(audit):
    for k in ("summary", "per_app", "per_app_graph", "violations"):
        assert k in audit


# --- Tolerance pinning ------------------------------------------------------

def test_floors_pinned(audit):
    s = audit["summary"]
    assert s["min_graphs_per_app"]   == 6
    assert s["min_l3s_per_app"]      == 3
    assert s["min_policies_per_app"] == 3
    assert s["min_l3_per_ag"]        == 3
    assert s["min_cells_per_app"]    == 60
    assert s["anchor_app"]           == ANCHOR_APP
    assert set(s["canonical_roster"]) == CANONICAL_ROSTER


# --- Zero-violation invariants ---------------------------------------------

def test_zero_violations(audit):
    assert audit["violations"] == [], (
        f"LIT-AppFreq violations: {audit['violations'][:5]}"
    )


def test_summary_violation_count_matches(audit):
    assert audit["summary"]["violations"] == len(audit["violations"])


def test_violations_by_rule_all_zero(audit):
    by_rule = audit["summary"]["violations_by_rule"]
    for rule in ("F1_graph_coverage_floor",
                 "F2_l3_coverage_floor",
                 "F3_policy_coverage_floor",
                 "F3_canonical_roster_missing",
                 "F4_per_app_graph_l3_sweep",
                 "F5_app_row_count_floor",
                 "F6_anchor_app_full_sweep"):
        assert by_rule.get(rule, 0) == 0, (
            f"{rule}: {by_rule.get(rule)} violations"
        )


# --- Corpus floors ----------------------------------------------------------

def test_app_count_floor(audit):
    assert len(audit["summary"]["apps"]) >= 5, (
        f"only {len(audit['summary']['apps'])} apps observed"
    )


def test_corpus_graph_count_floor(audit):
    assert audit["summary"]["corpus_graph_count"] >= 6, (
        f"corpus has only {audit['summary']['corpus_graph_count']} graphs"
    )


def test_total_row_floor(audit):
    assert audit["summary"]["total_rows"] >= 400


# --- Per-app invariants -----------------------------------------------------

def test_every_app_meets_graph_floor(audit):
    floor = audit["summary"]["min_graphs_per_app"]
    for r in audit["per_app"]:
        assert r["graph_count"] >= floor, (
            f"app {r['app']} has only {r['graph_count']} graphs"
        )


def test_every_app_meets_l3_floor(audit):
    floor = audit["summary"]["min_l3s_per_app"]
    for r in audit["per_app"]:
        assert r["l3_count"] >= floor, (
            f"app {r['app']} has only {r['l3_count']} L3 sizes"
        )


def test_every_app_meets_policy_floor(audit):
    floor = audit["summary"]["min_policies_per_app"]
    for r in audit["per_app"]:
        assert r["policy_count"] >= floor, (
            f"app {r['app']} has only {r['policy_count']} policies"
        )


def test_every_app_carries_canonical_roster(audit):
    for r in audit["per_app"]:
        missing = CANONICAL_ROSTER - set(r["policies"])
        assert not missing, (
            f"app {r['app']} missing canonical policies {missing}"
        )


def test_every_app_meets_row_floor(audit):
    floor = audit["summary"]["min_cells_per_app"]
    for r in audit["per_app"]:
        assert r["row_count"] >= floor, (
            f"app {r['app']} has only {r['row_count']} rows"
        )


# --- Anchor app -------------------------------------------------------------

def test_anchor_app_present(audit):
    apps = [r["app"] for r in audit["per_app"]]
    assert ANCHOR_APP in apps, f"anchor app {ANCHOR_APP} missing"


def test_anchor_app_covers_full_corpus(audit):
    anchor = next((r for r in audit["per_app"]
                   if r["app"] == ANCHOR_APP), None)
    assert anchor is not None
    corpus = audit["summary"]["corpus_graphs"]
    missing = set(corpus) - set(anchor["graphs"])
    assert not missing, (
        f"anchor app {ANCHOR_APP} missing graphs {missing}"
    )


# --- Per (app, graph) invariants -------------------------------------------

def test_every_app_graph_meets_l3_sweep(audit):
    floor = audit["summary"]["min_l3_per_ag"]
    for r in audit["per_app_graph"]:
        assert r["l3_count"] >= floor, (
            f"({r['app']}, {r['graph']}) has only {r['l3_count']} L3 sizes"
        )


# --- CSV / MD parity --------------------------------------------------------

def test_csv_row_count_matches(audit):
    import csv as csvm
    with CSV.open(encoding="utf-8") as fh:
        rows = list(csvm.reader(fh))
    assert len(rows) - 1 == len(audit["per_app"]), (
        f"CSV has {len(rows)-1} rows, expected {len(audit['per_app'])}"
    )


def test_markdown_artifact_nonempty():
    txt = MD.read_text(encoding="utf-8")
    assert "LIT-AppFreq" in txt
    assert "Per-app coverage" in txt
    assert ANCHOR_APP in txt
