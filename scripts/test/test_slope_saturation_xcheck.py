"""Gate 69 — saturation distance vs capacity-sensitivity slope invariants."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN = REPO_ROOT / "scripts" / "experiments" / "ecg" / "slope_saturation_xcheck.py"
JSON_PATH = REPO_ROOT / "wiki" / "data" / "slope_saturation_xcheck.json"


@pytest.fixture(scope="module")
def payload() -> dict:
    if not JSON_PATH.exists():
        subprocess.check_call([sys.executable, str(GEN)], cwd=str(REPO_ROOT))
    return json.loads(JSON_PATH.read_text())


def test_payload_well_formed(payload):
    assert "meta" in payload and "per_cell" in payload and "flat_cells" in payload
    meta = payload["meta"]
    for k in (
        "cells_matched",
        "cells_flat_excluded",
        "slope_epsilon_pp",
        "pearson_r",
        "spearman_rho",
        "median_distance_pp",
        "median_abs_slope_pp",
        "median_ratio_distance_to_slope",
        "min_matched_cells",
        "pearson_floor",
        "spearman_floor",
        "ratio_band",
        "verdict",
    ):
        assert k in meta, f"missing meta.{k}"


def test_enough_cells_matched(payload):
    meta = payload["meta"]
    assert meta["cells_matched"] >= meta["min_matched_cells"], (
        f"only {meta['cells_matched']} cells matched; need >= "
        f"{meta['min_matched_cells']}"
    )


def test_pearson_above_floor(payload):
    meta = payload["meta"]
    assert meta["pearson_r"] >= meta["pearson_floor"], (
        f"Pearson r {meta['pearson_r']!r} below floor {meta['pearson_floor']!r}"
    )


def test_spearman_above_floor(payload):
    meta = payload["meta"]
    assert meta["spearman_rho"] >= meta["spearman_floor"], (
        f"Spearman rho {meta['spearman_rho']!r} below floor "
        f"{meta['spearman_floor']!r}"
    )


def test_ratio_in_band(payload):
    meta = payload["meta"]
    lo, hi = meta["ratio_band"]
    r = meta["median_ratio_distance_to_slope"]
    assert lo <= r <= hi, (
        f"median ratio {r!r} outside band [{lo!r}, {hi!r}]"
    )


def test_verdict_pass(payload):
    assert payload["meta"]["verdict"] == "PASS", payload["meta"]


def test_per_cell_records_complete(payload):
    """Every per_cell record exposes the fields used in correlations."""
    required = {
        "app", "graph", "policy", "distance_pp", "slope_pp",
        "abs_slope_pp", "ratio_dist_slope",
    }
    for r in payload["per_cell"]:
        missing = required - set(r.keys())
        assert not missing, f"per_cell record missing keys {missing}: {r}"
        assert r["abs_slope_pp"] >= payload["meta"]["slope_epsilon_pp"], (
            f"non-flat per_cell has |slope| below epsilon: {r}"
        )


def test_flat_cells_have_small_slopes(payload):
    """Every flat_cells record has |slope| below the epsilon — they were
    excluded for exactly that reason."""
    eps = payload["meta"]["slope_epsilon_pp"]
    for r in payload["flat_cells"]:
        assert r["abs_slope_pp"] < eps, (
            f"flat_cells record |slope| {r['abs_slope_pp']!r} >= eps {eps!r}: {r}"
        )


def test_correlations_strictly_positive(payload):
    """Both correlation coefficients must be > 0 (catches a sign flip)."""
    meta = payload["meta"]
    assert meta["pearson_r"] > 0, f"Pearson is non-positive: {meta['pearson_r']!r}"
    assert meta["spearman_rho"] > 0, (
        f"Spearman is non-positive: {meta['spearman_rho']!r}"
    )


def test_median_distance_and_slope_positive(payload):
    """Medians on the absolute-value scale must be positive."""
    meta = payload["meta"]
    assert meta["median_distance_pp"] > 0, meta["median_distance_pp"]
    assert meta["median_abs_slope_pp"] > 0, meta["median_abs_slope_pp"]


def test_no_negative_distance(payload):
    """Distance is best4 - best8, which must be non-negative (cache
    monotonicity, already enforced in gate 65). Slope view here uses
    the policy's own miss-rate, so distance may very rarely be slightly
    negative due to within-policy non-monotonicity. Allow up to 0.5 pp
    of slack but flag if exceeded."""
    bad = [r for r in payload["per_cell"] if r["distance_pp"] < -0.5]
    assert not bad, f"large negative distances detected: {bad}"


def test_corpus_coverage_apps_and_policies(payload):
    """The matched set must cover all 4 policies and at least 4 apps."""
    apps = {r["app"] for r in payload["per_cell"]}
    pols = {r["policy"] for r in payload["per_cell"]}
    assert len(apps) >= 4, f"only {len(apps)} apps matched: {apps}"
    assert pols == {"GRASP", "LRU", "POPT", "SRRIP"}, (
        f"policies missing: expected all 4, got {pols}"
    )
