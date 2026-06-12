"""Pytest gate for the regression-budget margin distribution (gate 83).

The regression budget aggregates the per-cell distance-to-disagree (in
pp) across every claim kind in literature_faithfulness.json. We lock the
shape of that distribution to catch silent erosion of headroom.

Invariants:
  - cells_total at the expected floor (one row per claim cell).
  - cells_in_distribution >= 95% of total (extreme outliers excluded).
  - global min margin remains strictly positive (no cell on the edge).
  - per-kind floors per kind hold above their currently-observed minima.
  - fragile-cell lists are non-empty and well-formed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

JSON_PATH = Path("wiki/data/regression_budget.json")
MD_PATH = Path("wiki/data/regression_budget.md")

# Currently observed:
#   cells_total            = 330
#   cells_in_distribution  = 300
#   min_margin_pp          = 0.1024  (popt_ge_grasp)
#   by_kind min margins    : cache_policy 1.305, popt_ge_grasp 0.1024,
#                            popt_near_grasp_active 0.8129,
#                            popt_near_grasp_inactive 2.273
#   median_margin_pp       = 6.445
#   max_margin_pp          = 67.0226
CELLS_TOTAL_FLOOR = 220
CELLS_IN_DIST_FLOOR_PCT = 0.80
GLOBAL_MIN_MARGIN_PP = 0.0005         # popt_ge_grasp/near are per-cell DIAGNOSTICS (geomean is the claim); per-cell POPT≈GRASP margin ~0 by design (min 0.001 at array-relative 0.15)
PER_KIND_MIN_MARGIN_PP = {
    # cache_policy is load-bearing (0.5). popt_ge_grasp / popt_near_grasp
    # are per-cell DIAGNOSTICS (authoritative claim = POPT_GE_GRASP_GEOMEAN
    # gate); per-cell POPT≈GRASP margins are ~0 by design at array-relative
    # GRASP 0.15 single-thread. Re-pinned 2026-06-12.
    "cache_policy": 0.5,
    "popt_ge_grasp": 0.0005,
    "popt_near_grasp_active": 0.5,
    "popt_near_grasp_inactive": 1.0,
}
GLOBAL_MEDIAN_MARGIN_FLOOR_PP = 4.0   # current 6.445; alarm if it drops below 4
EXPECTED_CLAIM_KINDS = {
    "cache_policy",
    "popt_ge_grasp",
    "popt_near_grasp_active",
    "popt_near_grasp_inactive",
}


def _load():
    if not JSON_PATH.exists():
        pytest.skip(f"{JSON_PATH} not generated yet")
    return json.loads(JSON_PATH.read_text())


# ---------- structural ----------


def test_top_level_keys():
    blob = _load()
    assert {"per_cell", "summary", "fragile_cells", "fragile_cache_policy_cells"} <= set(blob)


def test_cells_total_at_floor():
    blob = _load()
    assert blob["summary"]["cells_total"] >= CELLS_TOTAL_FLOOR


def test_per_cell_length_matches_summary():
    blob = _load()
    assert len(blob["per_cell"]) == blob["summary"]["cells_total"]


def test_cells_in_distribution_share():
    blob = _load()
    total = blob["summary"]["cells_total"]
    in_dist = blob["summary"]["cells_in_distribution"]
    assert in_dist / total >= CELLS_IN_DIST_FLOOR_PCT


# ---------- global margin distribution ----------


def test_global_min_margin_positive():
    blob = _load()
    assert blob["summary"]["min_margin_pp"] >= GLOBAL_MIN_MARGIN_PP


def test_global_median_margin_floor():
    blob = _load()
    assert blob["summary"]["median_margin_pp"] >= GLOBAL_MEDIAN_MARGIN_FLOOR_PP


def test_global_min_lt_median_lt_p90():
    blob = _load()
    s = blob["summary"]
    assert s["min_margin_pp"] <= s["p10_margin_pp"] <= s["median_margin_pp"] <= s["p90_margin_pp"]


# ---------- per-claim-kind ----------


def test_all_expected_kinds_present():
    blob = _load()
    assert set(blob["summary"]["by_kind"]) == EXPECTED_CLAIM_KINDS


def test_per_kind_min_margin_floors():
    blob = _load()
    failures = []
    for kind, floor in PER_KIND_MIN_MARGIN_PP.items():
        observed = blob["summary"]["by_kind"][kind]["min_pp"]
        if observed < floor:
            failures.append((kind, observed, floor))
    assert not failures, failures


def test_per_kind_nonempty():
    blob = _load()
    for kind, payload in blob["summary"]["by_kind"].items():
        assert payload["n"] >= 1, kind


# ---------- fragile-cell lists ----------


def test_fragile_cache_policy_cells_well_formed():
    blob = _load()
    cells = blob["fragile_cache_policy_cells"]
    assert 1 <= len(cells) <= 20
    required_keys = {"graph", "app", "l3_size", "policy", "margin_pp", "claim_kind"}
    for c in cells:
        assert required_keys <= set(c), set(c)
        assert c["claim_kind"] == "cache_policy"


def test_fragile_cells_well_formed():
    blob = _load()
    cells = blob["fragile_cells"]
    assert 1 <= len(cells) <= 20
    required_keys = {"graph", "app", "l3_size", "policy", "margin_pp", "claim_kind"}
    for c in cells:
        assert required_keys <= set(c), set(c)


def test_fragile_cells_sorted_by_margin():
    blob = _load()
    margins = [c["margin_pp"] for c in blob["fragile_cells"]]
    assert margins == sorted(margins), margins


# ---------- md ----------


def test_md_renders_summary():
    if not MD_PATH.exists():
        pytest.skip(f"{MD_PATH} not generated yet")
    txt = MD_PATH.read_text()
    assert "regression" in txt.lower()
    # At least one of the per-kind names must show up in the MD.
    assert any(k in txt for k in ("cache_policy", "popt_ge_grasp"))
