"""Lock the LIT-CellComp invariants (gate 234).

Per (graph, app, l3) cell in the per_observation table: canonical
policy roster present, LRU baseline present, delta arithmetic matches
underlying miss rates, L3 sweep covers >= 3 sizes per non-LRU policy,
and every present policy shares the same L3 axis within (graph, app).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT  = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "wiki" / "data" / "lit_faith_cellcomp.json"
MD    = ROOT / "wiki" / "data" / "lit_faith_cellcomp.md"
CSV   = ROOT / "wiki" / "data" / "lit_faith_cellcomp.csv"


CANONICAL_ROSTER = {"LRU", "GRASP", "POPT"}


@pytest.fixture(scope="module")
def audit() -> dict:
    if not AUDIT.exists():
        pytest.skip("lit_faith_cellcomp.json missing; run `make lit-cellcomp`")
    return json.loads(AUDIT.read_text(encoding="utf-8"))


# --- Artifact presence + schema --------------------------------------------

def test_artifacts_present():
    assert AUDIT.exists(), f"missing {AUDIT}"
    assert MD.exists(),    f"missing {MD}"
    assert CSV.exists(),   f"missing {CSV}"


def test_schema_version(audit):
    assert audit["schema_version"] == 1


def test_top_level_keys(audit):
    for k in ("summary", "cells", "sweeps", "violations"):
        assert k in audit


# --- Tolerance pinning ------------------------------------------------------

def test_tolerances_pinned(audit):
    s = audit["summary"]
    assert s["min_l3_sweep"]       == 3
    assert s["delta_arith_tol_pp"] == pytest.approx(0.001)
    assert set(s["canonical_roster"]) == CANONICAL_ROSTER


# --- Zero-violation invariants ---------------------------------------------

def test_zero_violations(audit):
    assert audit["violations"] == [], (
        f"LIT-CellComp violations: {audit['violations'][:5]}"
    )


def test_summary_violation_count_matches(audit):
    assert audit["summary"]["violations"] == len(audit["violations"])


def test_zero_duplicate_rows(audit):
    assert audit["summary"]["duplicate_rows"] == 0


def test_violations_by_rule_all_zero(audit):
    by_rule = audit["summary"]["violations_by_rule"]
    for rule in ("C1_cell_roster_floor",
                 "C2_lru_baseline_missing",
                 "C3_l3_sweep_coverage",
                 "C4_l3_axis_parity",
                 "C5_delta_arithmetic",
                 "C6_miss_rate_bounds",
                 "C7_unique_row"):
        assert by_rule.get(rule, 0) == 0, (
            f"{rule}: {by_rule.get(rule)} violations"
        )


# --- Headline coverage floors ----------------------------------------------

def test_total_row_floor(audit):
    assert audit["summary"]["total_rows"] >= 400, (
        f"per_observation row count {audit['summary']['total_rows']} too low"
    )


def test_cell_count_floor(audit):
    assert audit["summary"]["cell_count"] >= 100, (
        f"cell count {audit['summary']['cell_count']} too low"
    )


def test_graph_count_floor(audit):
    assert len(audit["summary"]["graphs"]) >= 6, (
        f"only {len(audit['summary']['graphs'])} graphs in per_observation"
    )


def test_app_count_floor(audit):
    assert len(audit["summary"]["apps"]) >= 4, (
        f"only {len(audit['summary']['apps'])} apps in per_observation"
    )


def test_policy_count_floor(audit):
    assert len(audit["summary"]["policies"]) >= 3, (
        f"only {len(audit['summary']['policies'])} policies in per_observation"
    )


def test_canonical_roster_present_in_corpus(audit):
    seen = set(audit["summary"]["policies"])
    missing = CANONICAL_ROSTER - seen
    assert not missing, f"canonical roster missing from corpus: {missing}"


# --- Per-cell invariants ----------------------------------------------------

def test_every_cell_has_canonical_roster(audit):
    for c in audit["cells"]:
        missing = CANONICAL_ROSTER - set(c["policies"])
        assert not missing, (
            f"cell {c['graph']}/{c['app']}/{c['l3_size']} missing "
            f"policies {missing}"
        )


def test_every_cell_has_lru_baseline(audit):
    for c in audit["cells"]:
        assert c["lru_miss_rate"] is not None, (
            f"cell {c['graph']}/{c['app']}/{c['l3_size']} has no LRU row"
        )
        assert 0.0 <= c["lru_miss_rate"] <= 1.0, (
            f"cell {c['graph']}/{c['app']}/{c['l3_size']} LRU miss "
            f"{c['lru_miss_rate']} out of [0,1]"
        )


def test_every_cell_min_policy_count(audit):
    for c in audit["cells"]:
        assert c["policy_count"] >= len(CANONICAL_ROSTER), (
            f"cell {c['graph']}/{c['app']}/{c['l3_size']} has only "
            f"{c['policy_count']} policies"
        )


# --- L3 sweep invariants ----------------------------------------------------

def test_every_non_lru_sweep_meets_floor(audit):
    floor = audit["summary"]["min_l3_sweep"]
    for s in audit["sweeps"]:
        if s["policy"] == "LRU":
            continue
        assert s["l3_count"] >= floor, (
            f"sweep {s['graph']}/{s['app']}/{s['policy']} has "
            f"only {s['l3_count']} L3 sizes (floor {floor})"
        )


# --- CSV / MD parity --------------------------------------------------------

def test_csv_row_count_matches(audit):
    import csv as csvm
    with CSV.open(encoding="utf-8") as fh:
        rows = list(csvm.reader(fh))
    assert len(rows) - 1 == len(audit["cells"]), (
        f"CSV {len(rows)-1} != JSON cells {len(audit['cells'])}"
    )


def test_markdown_artifact_nonempty():
    txt = MD.read_text(encoding="utf-8")
    assert "LIT-CellComp" in txt
    assert "Per-cell policy roster" in txt
