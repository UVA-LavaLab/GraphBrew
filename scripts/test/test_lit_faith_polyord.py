"""Lock the LIT-PolyOrd invariants (gate 231).

Audits `wiki/data/lit_faith_polyord.json`: per (graph_family × app) the
hub-bearing families (social, citation, web) must respect the literature
ordering POPT/GRASP <= LRU within tolerance, hub-less families (road,
mesh) are documented exceptions, and the per-app global frame must stay
above the global improve-frac floor.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT      = Path(__file__).resolve().parents[2]
AUDIT     = ROOT / "wiki" / "data" / "lit_faith_polyord.json"
CSV_PATH  = ROOT / "wiki" / "data" / "lit_faith_polyord.csv"
MD_PATH   = ROOT / "wiki" / "data" / "lit_faith_polyord.md"

HUB_FAMILIES    = {"social", "citation", "web"}
NO_HUB_FAMILIES = {"road", "mesh"}
ALL_APPS        = {"bfs", "bc", "cc", "pr", "sssp"}


@pytest.fixture(scope="module")
def audit() -> dict:
    if not AUDIT.exists():
        pytest.skip("lit_faith_polyord.json missing; run `make lit-polyord`")
    return json.loads(AUDIT.read_text(encoding="utf-8"))


# --- Artifact presence + schema --------------------------------------------

def test_artifacts_present():
    assert AUDIT.exists(),    f"missing {AUDIT}"
    assert CSV_PATH.exists(), f"missing {CSV_PATH}"
    assert MD_PATH.exists(),  f"missing {MD_PATH}"


def test_schema_version(audit):
    assert audit["schema_version"] == 1


def test_top_level_keys(audit):
    for k in ("summary", "buckets", "per_app_global", "violations"):
        assert k in audit, f"missing top-level key {k!r}"


# --- Tolerance pinning ------------------------------------------------------

def test_tolerances_pinned(audit):
    s = audit["summary"]
    assert s["popt_hub_bound_pp"]  == pytest.approx(0.5)
    assert s["grasp_hub_bound_pp"] == pytest.approx(1.0)
    assert s["improve_frac_floor"] == pytest.approx(0.50)
    assert s["per_app_global_improve_frac_floor"] == pytest.approx(0.55)
    assert s["cell_count_floor"]   == 2


# --- Headline coverage floors ----------------------------------------------

def test_total_cells_floor(audit):
    assert audit["summary"]["total_cells"] >= 100, (
        "LIT-PolyOrd should audit ≥100 (graph,app,L3) cells; "
        "lower counts indicate corpus regression"
    )


def test_bucket_count_floor(audit):
    assert audit["summary"]["bucket_count"] >= 18, (
        "≥18 (family,app) buckets expected (≥3 hub families × 5 apps + "
        "≥3 no-hub buckets)"
    )


def test_hub_bucket_count_floor(audit):
    assert audit["summary"]["hub_buckets"] >= 13, (
        "hub-bearing buckets must cover ≥13 (family,app) combinations"
    )


def test_no_hub_bucket_count_floor(audit):
    assert audit["summary"]["no_hub_buckets"] >= 2, (
        "at least 2 no-hub buckets (road or mesh) must be present so "
        "the no-hub regime is exercised"
    )


# --- Zero-violation invariants ---------------------------------------------

def test_zero_violations(audit):
    assert audit["violations"] == [], (
        f"LIT-PolyOrd violations: {audit['violations'][:5]}"
    )


def test_summary_violation_count_matches_list(audit):
    assert audit["summary"]["violations"] == len(audit["violations"])


# --- Unknown-family guard ---------------------------------------------------

def test_no_unknown_family_graphs(audit):
    assert audit["summary"]["unknown_family_graphs"] == [], (
        "every audited graph must have a family entry in GRAPH_FAMILY; "
        f"missing: {audit['summary']['unknown_family_graphs']}"
    )


# --- Regime / row schema ----------------------------------------------------

def test_regime_field_values(audit):
    for r in audit["buckets"]:
        assert r["regime"] in {"hub", "no_hub"}, (
            f"bad regime {r['regime']!r} on row {r}"
        )
        if r["graph_family"] in HUB_FAMILIES:
            assert r["regime"] == "hub"
        if r["graph_family"] in NO_HUB_FAMILIES:
            assert r["regime"] == "no_hub"


def test_bucket_row_keys(audit):
    required = {
        "graph_family", "regime", "app", "cell_count", "graph_count",
        "popt_median_pp", "grasp_median_pp", "srrip_median_pp",
        "popt_improve_frac", "grasp_improve_frac", "srrip_improve_frac",
    }
    for r in audit["buckets"]:
        missing = required - set(r)
        assert not missing, f"bucket row missing keys {missing}: {r}"


def test_no_negative_cell_counts(audit):
    for r in audit["buckets"]:
        assert r["cell_count"] >= 1, f"non-positive cell_count: {r}"
        assert r["graph_count"] >= 1, f"non-positive graph_count: {r}"


# --- Hub-family per-bucket ordering invariants ------------------------------

def test_hub_buckets_respect_popt_median_bound(audit):
    bound = audit["summary"]["popt_hub_bound_pp"]
    for r in audit["buckets"]:
        if r["graph_family"] not in HUB_FAMILIES:
            continue
        if r["popt_median_pp"] is None:
            continue
        assert r["popt_median_pp"] <= bound, (
            f"hub bucket {r['graph_family']}/{r['app']} POPT median "
            f"{r['popt_median_pp']} pp exceeds bound {bound} pp"
        )


def test_hub_buckets_respect_grasp_median_bound(audit):
    bound = audit["summary"]["grasp_hub_bound_pp"]
    for r in audit["buckets"]:
        if r["graph_family"] not in HUB_FAMILIES:
            continue
        if r["grasp_median_pp"] is None:
            continue
        assert r["grasp_median_pp"] <= bound, (
            f"hub bucket {r['graph_family']}/{r['app']} GRASP median "
            f"{r['grasp_median_pp']} pp exceeds bound {bound} pp"
        )


# --- Per-app global hub-aggregate invariants --------------------------------

def test_per_app_global_apps_covered(audit):
    seen = set(audit["per_app_global"])
    missing = ALL_APPS - seen
    assert not missing, (
        f"per_app_global missing apps {missing}; ALL_APPS={ALL_APPS}"
    )


def test_per_app_global_frac_floor(audit):
    floor = audit["summary"]["per_app_global_improve_frac_floor"]
    for app, row in audit["per_app_global"].items():
        frac = row.get("popt_improve_frac")
        if frac is None:
            continue
        assert frac >= floor, (
            f"app {app} per-app-global POPT improve frac {frac} < {floor}"
        )


def test_per_app_global_popt_median_nonpositive(audit):
    for app, row in audit["per_app_global"].items():
        med = row.get("popt_median_pp")
        if med is None:
            continue
        assert med <= 0.0, (
            f"hub-aggregated POPT median for {app} must be <= 0 pp "
            f"(POPT not worse than LRU on the median cell); got {med}"
        )


# --- CSV schema parity ------------------------------------------------------

def test_csv_header_matches_bucket_keys(audit):
    import csv
    with CSV_PATH.open(encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows   = list(reader)
    # All bucket rows must be present in the CSV exactly once.
    assert len(rows) == len(audit["buckets"]), (
        f"CSV row count {len(rows)} != JSON bucket count "
        f"{len(audit['buckets'])}"
    )
    for col in ("graph_family", "regime", "app", "cell_count",
                "popt_median_pp", "grasp_median_pp",
                "popt_improve_frac", "grasp_improve_frac"):
        assert col in header, f"CSV missing column {col}"


def test_markdown_artifact_nonempty():
    txt = MD_PATH.read_text(encoding="utf-8")
    assert "LIT-PolyOrd" in txt
    assert "Per-bucket detail" in txt
    assert "Per-app global" in txt
