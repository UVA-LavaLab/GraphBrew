"""Pytest gate: winning-regime taxonomy invariants.

The taxonomy joins the policy-winner table with corpus features and
buckets each cell into a (family, regime) bin. This gate pins the
schema and the *load-bearing* paper narrative:

* the matrix covers every family present in the corpus
  (citation, mesh, road, social, web) at the `large` regime — the
  paper's headline regime;
* mesh and road families exist in the corpus (catches a future
  corpus prune that would silently drop the paper's two adversarial
  families);
* on the `road` family at the `large` regime LRU wins at least one
  cell — the GRASP-cannot-help-on-road claim must show LRU as a
  competitor and not just POPT vs GRASP;
* extracted rules use the closed policy vocabulary and reference
  only known (family, regime) bins.

Wider invariants (e.g. the popt-crushes-grasp claim on roads) are
already pinned by other gates; this one focuses on the structure of
the taxonomy itself so a regression in the bucketing code is caught.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT = REPO_ROOT / "wiki" / "data" / "winning_regime_taxonomy.json"

KNOWN_POLICIES = {"LRU", "SRRIP", "GRASP", "POPT"}
KNOWN_FAMILIES = {"citation", "mesh", "road", "social", "web"}
KNOWN_REGIMES = {"tiny", "small", "medium", "large"}


@pytest.fixture(scope="module")
def report() -> dict:
    if not REPORT.exists():
        pytest.skip(f"{REPORT} not generated; run `make lit-regime-taxonomy`")
    return json.loads(REPORT.read_text())


def test_top_level_schema(report):
    assert {"summary", "cells"}.issubset(report.keys())
    s = report["summary"]
    for k in (
        "n_cells", "n_family_regime_bins",
        "by_family_regime", "rules",
        "overall_winner_counts", "rule_threshold",
    ):
        assert k in s, f"missing summary key {k!r}"


def test_n_cells_matches(report):
    assert report["summary"]["n_cells"] == len(report["cells"])


def test_at_least_seven_bins(report):
    assert report["summary"]["n_family_regime_bins"] >= 7, (
        "matrix collapsed below 7 (family, regime) bins — corpus or "
        "regime bucketing may have regressed"
    )


def test_overall_policies_in_known_set(report):
    extra = set(report["summary"]["overall_winner_counts"]) - KNOWN_POLICIES
    assert not extra, f"unexpected winners in tally: {sorted(extra)}"


def test_each_bin_uses_known_family_and_regime(report):
    for row in report["summary"]["by_family_regime"]:
        assert row["family"] in KNOWN_FAMILIES, row["family"]
        assert row["regime"] in KNOWN_REGIMES, row["regime"]


def test_large_regime_covers_every_family(report):
    """At the `large` regime (paper's headline) every corpus family
    must be represented. Catches a future corpus change that drops
    a family entirely from the largest-cache slice."""
    fams_at_large = {
        row["family"] for row in report["summary"]["by_family_regime"]
        if row["regime"] == "large"
    }
    missing = KNOWN_FAMILIES - fams_at_large
    assert not missing, (
        f"families missing from `large` regime: {sorted(missing)} — "
        "regenerate winner table after re-running missing graphs"
    )


def test_mesh_and_road_families_present(report):
    """Mesh + road are the two adversarial-to-GRASP families that
    drive the paper's negative claims. If either disappears the
    central narrative collapses."""
    fams = {row["family"] for row in report["summary"]["by_family_regime"]}
    assert "road" in fams, "road family missing from taxonomy"
    assert "mesh" in fams, "mesh family missing from taxonomy"


def test_road_large_has_at_least_one_lru_win(report):
    """At large L3 on road graphs LRU must beat GRASP somewhere.
    This is the load-bearing 'GRASP cannot help on uniform-degree
    graphs' finding the paper quotes."""
    row = next(
        (
            r for r in report["summary"]["by_family_regime"]
            if r["family"] == "road" and r["regime"] == "large"
        ),
        None,
    )
    assert row is not None, "no (road, large) bin found"
    assert row["LRU_wins"] >= 1, (
        f"road/large should have ≥1 LRU win, got {row['LRU_wins']}"
    )


def test_rules_reference_known_combinations(report):
    """Every extracted rule must point at a (family, regime) bin
    that exists in the matrix and use a known policy."""
    bins = {
        (r["family"], r["regime"])
        for r in report["summary"]["by_family_regime"]
    }
    for rule in report["summary"]["rules"]:
        assert (rule["family"], rule["regime"]) in bins, rule
        assert rule["winner"] in KNOWN_POLICIES, rule
        assert 0.0 < rule["fraction"] <= 1.0, rule
        assert rule["wins"] <= rule["sample_size"], rule
