"""Lock the LIT-RatGrid invariants (gate 233).

Per (policy, graph, app) cell: rationale text must be unique within the
cell (theorem-class policies <= 1 rationale, point policies <= 2 to
accommodate L3-regime variants). Theorem-class policies must carry the
same rationale across all graphs within (policy, app). Point-policy
rationales must include a citation token.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT  = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "wiki" / "data" / "lit_faith_ratgrid.json"
MD    = ROOT / "wiki" / "data" / "lit_faith_ratgrid.md"
CSV   = ROOT / "wiki" / "data" / "lit_faith_ratgrid.csv"


THEOREM_POLICIES = {"POPT_GE_GRASP", "POPT_NEAR_GRASP_IF_BIG_GAP", "SRRIP"}
POINT_POLICIES   = {"GRASP", "POPT", "LRU"}


@pytest.fixture(scope="module")
def audit() -> dict:
    if not AUDIT.exists():
        pytest.skip("lit_faith_ratgrid.json missing; run `make lit-ratgrid`")
    return json.loads(AUDIT.read_text(encoding="utf-8"))


# --- Artifacts present ------------------------------------------------------

def test_artifacts_present():
    assert AUDIT.exists(), f"missing {AUDIT}"
    assert MD.exists(),    f"missing {MD}"
    assert CSV.exists(),   f"missing {CSV}"


def test_schema_version(audit):
    assert audit["schema_version"] == 1


def test_top_level_keys(audit):
    for k in ("summary", "rows", "violations"):
        assert k in audit


# --- Tolerance pinning ------------------------------------------------------

def test_tolerances_pinned(audit):
    s = audit["summary"]
    assert s["min_rationale_len"]         == 40
    assert s["min_rationales_per_policy"] == 1


def test_policy_class_sets_pinned(audit):
    s = audit["summary"]
    assert set(s["theorem_policies"]) == THEOREM_POLICIES
    assert set(s["point_policies"])   == POINT_POLICIES


# --- Zero-violation invariants ---------------------------------------------

def test_zero_violations(audit):
    assert audit["violations"] == [], (
        f"LIT-RatGrid violations: {audit['violations'][:5]}"
    )


def test_summary_violation_count_matches(audit):
    assert audit["summary"]["violations"] == len(audit["violations"])


def test_zero_cell_uniqueness_violations(audit):
    assert audit["summary"]["cell_uniqueness_violations"] == 0


def test_zero_theorem_invariance_violations(audit):
    assert audit["summary"]["theorem_invariance_violations"] == 0


# --- Headline coverage floors ----------------------------------------------

def test_total_row_floor(audit):
    assert audit["summary"]["total_rows"] >= 250, (
        f"per_claim row count {audit['summary']['total_rows']} too low"
    )


def test_cell_count_floor(audit):
    assert audit["summary"]["cell_count"] >= 100, (
        f"cell count {audit['summary']['cell_count']} too low — "
        f"corpus must cover >= 100 (policy, graph, app) cells"
    )


# --- Theorem-class invariance: per-(policy, app) must have <= 1 rationale --

def test_theorem_policies_one_rationale_per_pol_app(audit):
    s = audit["summary"]
    counts = s["rationale_counts_per_pol_app"]
    for key, n in counts.items():
        policy, _ = key.split("|")
        if policy in THEOREM_POLICIES:
            assert n == 1, (
                f"theorem policy {key} has {n} rationales; must be exactly 1"
            )


def test_point_policies_have_multiple_rationales(audit):
    """Point policies (GRASP, POPT) carry per-graph Fig refs, so
    aggregating across an app should see multiple distinct rationales
    when the app is exercised on >= 2 graphs."""
    s = audit["summary"]
    counts = s["rationale_counts_per_pol_app"]
    saw_multi = False
    for key, n in counts.items():
        policy, _ = key.split("|")
        if policy in POINT_POLICIES and n >= 2:
            saw_multi = True
            break
    assert saw_multi, (
        "no point-policy (policy, app) bucket carries >= 2 rationales — "
        "corpus has collapsed to a single graph per app"
    )


# --- Per-policy coverage floor ---------------------------------------------

def test_per_policy_coverage_floor(audit):
    s = audit["summary"]
    for policy, n in s["unique_rationales_per_policy"].items():
        assert n >= s["min_rationales_per_policy"], (
            f"policy {policy} has only {n} rationales "
            f"(floor {s['min_rationales_per_policy']})"
        )


def test_theorem_class_coverage(audit):
    """Each theorem-class policy must appear in at least one row."""
    s = audit["summary"]
    seen = set(s["unique_rationales_per_policy"])
    missing = THEOREM_POLICIES - seen
    assert not missing, f"missing theorem-class policies: {missing}"


def test_point_policy_coverage(audit):
    """At least 2 of the 3 point policies (GRASP, POPT, LRU) must
    appear so the per-graph rationale grid is meaningful."""
    s = audit["summary"]
    seen = set(s["unique_rationales_per_policy"])
    intersection = POINT_POLICIES & seen
    assert len(intersection) >= 2, (
        f"only {sorted(intersection)} point policies in registry"
    )


# --- Row-schema integrity ---------------------------------------------------

def test_row_keys_present(audit):
    required = {"policy", "graph", "app", "l3_size", "rationale_len",
                "has_citation_token", "rationale_excerpt"}
    for r in audit["rows"]:
        missing = required - set(r)
        assert not missing, f"row missing keys {missing}: {r}"


def test_every_row_has_minimum_length(audit):
    floor = audit["summary"]["min_rationale_len"]
    for r in audit["rows"]:
        assert r["rationale_len"] >= floor, (
            f"row {r['policy']}/{r['graph']}/{r['app']}/{r['l3_size']} "
            f"rationale length {r['rationale_len']} < {floor}"
        )


def test_point_policy_rows_have_citation_token(audit):
    for r in audit["rows"]:
        if r["policy"] in POINT_POLICIES:
            assert r["has_citation_token"], (
                f"point-policy row {r['policy']}/{r['graph']}/{r['app']}/"
                f"{r['l3_size']} rationale lacks citation token "
                f"(HPCA/MICRO/Fig/§)"
            )


# --- CSV / MD parity --------------------------------------------------------

def test_csv_row_count_matches(audit):
    import csv as csvm
    with CSV.open(encoding="utf-8") as fh:
        rows = list(csvm.reader(fh))
    assert len(rows) - 1 == len(audit["rows"]), (
        f"CSV {len(rows)-1} != JSON {len(audit['rows'])}"
    )


def test_markdown_artifact_nonempty():
    txt = MD.read_text(encoding="utf-8")
    assert "LIT-RatGrid" in txt
    assert "Unique rationales per policy" in txt
    assert "Rationale count per (policy, app)" in txt
