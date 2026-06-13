"""Lock the LIT-RegimeSign invariants (gate 236).

Regime-aware sign-tally + extreme magnitude ceiling on the
per_observation table. Complements LIT-PolyOrd (per-bucket median
*magnitude* bounds) with per-bucket *direction* tallies and a
per-cell magnitude cap.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT  = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "wiki" / "data" / "lit_faith_regimesign.json"
MD    = ROOT / "wiki" / "data" / "lit_faith_regimesign.md"
CSV   = ROOT / "wiki" / "data" / "lit_faith_regimesign.csv"


HUB_FAMILIES    = {"social", "citation", "web"}
NO_HUB_FAMILIES = {"road",   "mesh"}
ADVICE_POLICIES = {"GRASP", "POPT"}
# Documented frontier-kernel exceptions to the hub no-regression / median
# ceiling rules (see generator + docs/findings/grasp_road_anti_thrashing.md).
FRONTIER_HUB_EXCEPTIONS = {("web", "bc"), ("web", "sssp"), ("web", "cc")}


@pytest.fixture(scope="module")
def audit() -> dict:
    if not AUDIT.exists():
        pytest.skip(
            "lit_faith_regimesign.json missing; run `make lit-regimesign`")
    return json.loads(AUDIT.read_text(encoding="utf-8"))


# --- Artifact presence + schema --------------------------------------------

def test_artifacts_present():
    assert AUDIT.exists(), f"missing {AUDIT}"
    assert MD.exists(),    f"missing {MD}"
    assert CSV.exists(),   f"missing {CSV}"


def test_schema_version(audit):
    assert audit["schema_version"] == 1


def test_top_level_keys(audit):
    for k in ("summary", "buckets", "violations"):
        assert k in audit


# --- Tolerance pinning ------------------------------------------------------

def test_tolerances_pinned(audit):
    s = audit["summary"]
    assert s["sign_deadband_pp"]        == pytest.approx(1.0)
    assert s["hub_median_ceil_pp"]      == pytest.approx(0.5)
    assert s["no_hub_median_radius_pp"] == pytest.approx(8.0)
    assert s["extreme_delta_cap_pp"]    == pytest.approx(80.0)
    assert set(s["hub_families"])    == HUB_FAMILIES
    assert set(s["no_hub_families"]) == NO_HUB_FAMILIES
    assert set(s["advice_policies"]) == ADVICE_POLICIES


# --- Zero-violation invariants ---------------------------------------------

def test_zero_violations(audit):
    assert audit["violations"] == [], (
        f"LIT-RegimeSign violations: {audit['violations'][:5]}"
    )


def test_summary_violation_count_matches(audit):
    assert audit["summary"]["violations"] == len(audit["violations"])


def test_violations_by_rule_all_zero(audit):
    by_rule = audit["summary"]["violations_by_rule"]
    for rule in ("R1_hub_majority_no_regression",
                 "R2_hub_median_ceiling",
                 "R3_extreme_magnitude_ceiling",
                 "R4_no_hub_median_radius"):
        assert by_rule.get(rule, 0) == 0, (
            f"{rule}: {by_rule.get(rule)} violations"
        )


# --- Coverage floors --------------------------------------------------------

def test_bucket_count_floor(audit):
    assert audit["summary"]["bucket_count"] >= 60, (
        f"only {audit['summary']['bucket_count']} buckets"
    )


def test_hub_bucket_floor(audit):
    assert audit["summary"]["hub_buckets"] >= 30


def test_no_hub_bucket_floor(audit):
    assert audit["summary"]["no_hub_buckets"] >= 10


def test_total_row_floor(audit):
    assert audit["summary"]["total_rows"] >= 300


def test_no_extreme_cells(audit):
    assert audit["summary"]["extreme_cells"] == [], (
        f"unexpected extreme cells: {audit['summary']['extreme_cells'][:3]}"
    )


# --- Per-bucket invariants -------------------------------------------------

def test_hub_advice_buckets_pass_R1(audit):
    """Every hub advice-policy bucket must NOT trip the R1
    co-occurrence (pos_cells > neg_cells AND median > +0.5 pp)."""
    ceil = audit["summary"]["hub_median_ceil_pp"]
    for r in audit["buckets"]:
        if r["family"] not in HUB_FAMILIES:    continue
        if r["policy"] not in ADVICE_POLICIES: continue
        if (r["family"], r["app"]) in FRONTIER_HUB_EXCEPTIONS: continue
        regressive = (r["pos_cells"] > r["neg_cells"]
                      and r["median_delta_pp"] > ceil)
        assert not regressive, (
            f"bucket {r['family']}/{r['app']}/{r['policy']} is "
            f"regressive: pos={r['pos_cells']} neg={r['neg_cells']} "
            f"median={r['median_delta_pp']:+.3f} pp"
        )


def test_hub_advice_buckets_pass_R2(audit):
    ceil = audit["summary"]["hub_median_ceil_pp"]
    for r in audit["buckets"]:
        if r["family"] not in HUB_FAMILIES:    continue
        if r["policy"] not in ADVICE_POLICIES: continue
        if (r["family"], r["app"]) in FRONTIER_HUB_EXCEPTIONS: continue
        assert r["median_delta_pp"] <= ceil, (
            f"bucket {r['family']}/{r['app']}/{r['policy']} median "
            f"{r['median_delta_pp']:+.3f} > ceiling {ceil}"
        )


def test_no_hub_advice_buckets_pass_R4(audit):
    rad = audit["summary"]["no_hub_median_radius_pp"]
    for r in audit["buckets"]:
        if r["family"] not in NO_HUB_FAMILIES: continue
        if r["policy"] not in ADVICE_POLICIES: continue
        assert abs(r["median_delta_pp"]) <= rad, (
            f"bucket {r['family']}/{r['app']}/{r['policy']} median "
            f"{r['median_delta_pp']:+.3f} outside ±{rad} pp radius"
        )


def test_all_buckets_pass_R3(audit):
    cap = audit["summary"]["extreme_delta_cap_pp"]
    for r in audit["buckets"]:
        assert r["max_abs_delta_pp"] <= cap, (
            f"bucket {r['family']}/{r['app']}/{r['policy']} has "
            f"|Δ| {r['max_abs_delta_pp']} > cap {cap}"
        )


# --- Specific phenomenology -------------------------------------------------

def test_documented_road_l_curve_present(audit):
    """road family with sssp or bfs should show at least one +Δ cell
    near the working-set knee — this is the documented L-curve hump
    on roadNet-CA at L3=1MB. If this disappears, either the corpus
    no longer ships roadNet-CA or the L-curve has shifted."""
    found = False
    for r in audit["buckets"]:
        if r["family"] == "road" and r["app"] in ("sssp", "bfs"):
            if r["pos_cells"] >= 1 and r["max_abs_delta_pp"] >= 5.0:
                found = True
                break
    assert found, "expected road L-curve hump on sssp/bfs missing"


def test_pr_app_hub_improvements_dominate(audit):
    """pr on hub families is THE load-bearing app for GRASP/POPT
    claims — every (hub, pr, advice_policy) bucket must show
    majority-negative deltas."""
    for r in audit["buckets"]:
        if r["family"] in HUB_FAMILIES and r["app"] == "pr" \
                and r["policy"] in ADVICE_POLICIES:
            assert r["neg_cells"] > r["pos_cells"], (
                f"hub pr bucket {r['family']}/pr/{r['policy']}: "
                f"neg={r['neg_cells']} pos={r['pos_cells']}"
            )


# --- CSV / MD parity --------------------------------------------------------

def test_csv_row_count_matches(audit):
    import csv as csvm
    with CSV.open(encoding="utf-8") as fh:
        rows = list(csvm.reader(fh))
    assert len(rows) - 1 == len(audit["buckets"]), (
        f"CSV {len(rows)-1} != JSON {len(audit['buckets'])} buckets"
    )


def test_markdown_artifact_nonempty():
    txt = MD.read_text(encoding="utf-8")
    assert "LIT-RegimeSign" in txt
    assert "Per-bucket sign tally" in txt
