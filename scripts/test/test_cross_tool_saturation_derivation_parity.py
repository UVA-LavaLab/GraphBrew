"""Derivation parity gate for ``wiki/data/cross_tool_saturation.json``.

Locks the cross-tool saturation soundness report against the
internal formulas of ``cross_tool_saturation_report.py``: per-cell
saturation classification (doubly/single/neither), GRASP−LRU
agreement tolerance, regime counts, doubly-saturated agreement
counter, and the disagreement extractor.

The full upstream reconstruction would require loading three
upstreams (lit-faith CSV + gem5_anchor.json + sniper_anchor.json),
which other gates already lock. This gate instead pins:

  1. Per-cell shape + value-range invariants for the structural
     contract (regime ∈ {doubly, single, neither, incomplete},
     tool ∈ {gem5, sniper}, spread and delta are non-negative or
     None, etc).
  2. The classifier output matches `_classify` byte-exact when
     re-run on each cell's recorded inputs.
  3. The summary reducers (regime_counts, doubly_saturated_total,
     doubly_saturated_agree, disagreements) match recomputation
     from the per-cell records.

If any of those drifts (the classifier flips a threshold, the
disagreement filter changes regime gating, the summary stops
counting "agreed"), the gate trips before the dashboard re-
publishes the "gem5 and Sniper agree with cache_sim at saturation"
story.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

ARTIFACT_PATH = WIKI_DATA / "cross_tool_saturation.json"

KNOWN_TOOLS = ("gem5", "sniper")
KNOWN_REGIMES = ("doubly_saturated", "single_saturated",
                 "neither_saturated", "incomplete")
KNOWN_L3_SIZES = {
    "4kB", "8kB", "16kB", "32kB", "64kB", "128kB", "256kB",
    "512kB", "1MB", "2MB", "4MB", "8MB",
}


def _classify(c, sat_floor, headline_tol):
    """Verbatim mirror of generator `_classify`."""
    cs_s = c["cache_sim_spread_pp"]
    a_s = c["anchor_spread_pp"]
    cs_d = c["cache_sim_grasp_minus_lru_pp"]
    a_d = c["anchor_grasp_minus_lru_pp"]
    if cs_s is None or a_s is None:
        return {"regime": "incomplete", "agree": None, "delta_pp": None}
    cs_sat = cs_s < sat_floor
    a_sat = a_s < sat_floor
    if cs_sat and a_sat:
        regime = "doubly_saturated"
    elif cs_sat or a_sat:
        regime = "single_saturated"
    else:
        regime = "neither_saturated"
    delta = None
    agree = None
    if cs_d is not None and a_d is not None:
        delta = abs(cs_d - a_d)
        agree = delta <= headline_tol
    return {"regime": regime, "agree": agree, "delta_pp": delta}


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def artifact() -> dict:
    if not ARTIFACT_PATH.exists():
        pytest.skip(f"missing {ARTIFACT_PATH}")
    return json.loads(ARTIFACT_PATH.read_text())


# ----------------------------------------------------------------------
# Group A: shape & schema
# ----------------------------------------------------------------------

def test_top_level_keys(artifact):
    assert set(artifact.keys()) == {"cells", "schema_version", "summary"}


def test_schema_version(artifact):
    assert artifact["schema_version"] == 1


def test_summary_fields(artifact):
    expected = {
        "sat_floor_pp", "headline_tol_pp", "n_cells",
        "regime_counts", "doubly_saturated_agree",
        "doubly_saturated_total", "disagreements",
    }
    missing = expected - set(artifact["summary"].keys())
    assert not missing, f"summary missing fields: {missing}"


def test_summary_thresholds_positive(artifact):
    s = artifact["summary"]
    assert s["sat_floor_pp"] > 0, "sat_floor_pp must be positive"
    assert s["headline_tol_pp"] > 0, "headline_tol_pp must be positive"


def test_per_cell_shape(artifact):
    expected = {
        "graph", "app", "tool",
        "cache_sim_l3", "anchor_l3",
        "cache_sim_spread_pp", "anchor_spread_pp",
        "cache_sim_grasp_minus_lru_pp", "anchor_grasp_minus_lru_pp",
        "regime", "agree", "delta_pp",
    }
    for c in artifact["cells"]:
        missing = expected - set(c.keys())
        assert not missing, f"cell missing fields: {missing}"


def test_cell_tool_only_known(artifact):
    for c in artifact["cells"]:
        assert c["tool"] in KNOWN_TOOLS, (
            f"cell {c}: unknown tool {c['tool']!r}"
        )


def test_cell_regime_only_known(artifact):
    for c in artifact["cells"]:
        assert c["regime"] in KNOWN_REGIMES, (
            f"cell {c}: unknown regime {c['regime']!r}"
        )


def test_cell_l3_only_known(artifact):
    for c in artifact["cells"]:
        if c["cache_sim_l3"] is not None:
            assert c["cache_sim_l3"] in KNOWN_L3_SIZES, (
                f"cell {c}: unknown cache_sim_l3 {c['cache_sim_l3']!r}"
            )
        if c["anchor_l3"] is not None:
            assert c["anchor_l3"] in KNOWN_L3_SIZES, (
                f"cell {c}: unknown anchor_l3 {c['anchor_l3']!r}"
            )


def test_cell_spreads_non_negative_or_none(artifact):
    """Spread is max-min so must be ≥ 0 when present."""
    for c in artifact["cells"]:
        for fld in ("cache_sim_spread_pp", "anchor_spread_pp"):
            v = c[fld]
            assert v is None or v >= 0.0, (
                f"cell {c}: {fld} {v} must be ≥ 0 or None"
            )


def test_cell_delta_non_negative_or_none(artifact):
    """delta_pp is abs(...) so must be ≥ 0 when present."""
    for c in artifact["cells"]:
        v = c["delta_pp"]
        assert v is None or v >= 0.0, (
            f"cell {c}: delta_pp {v} must be ≥ 0 or None"
        )


# ----------------------------------------------------------------------
# Group B: per-cell classifier parity
# ----------------------------------------------------------------------

def test_cell_regime_matches_classifier(artifact):
    sat_floor = artifact["summary"]["sat_floor_pp"]
    headline_tol = artifact["summary"]["headline_tol_pp"]
    for c in artifact["cells"]:
        result = _classify(c, sat_floor, headline_tol)
        assert c["regime"] == result["regime"], (
            f"cell {c}: regime {c['regime']!r} disagrees with "
            f"_classify({sat_floor}, {headline_tol}) = {result['regime']!r}"
        )


def test_cell_delta_matches_abs_difference(artifact):
    """delta_pp = abs(cs_grasp_minus_lru − anchor_grasp_minus_lru)."""
    for c in artifact["cells"]:
        cs_d = c["cache_sim_grasp_minus_lru_pp"]
        a_d = c["anchor_grasp_minus_lru_pp"]
        if cs_d is None or a_d is None:
            assert c["delta_pp"] is None, (
                f"cell {c}: delta_pp must be None when an input is None"
            )
        else:
            expected = abs(cs_d - a_d)
            assert c["delta_pp"] == pytest.approx(expected, abs=1e-9), (
                f"cell {c}: delta_pp {c['delta_pp']} ≠ "
                f"abs({cs_d} − {a_d}) = {expected}"
            )


def test_cell_agree_matches_tolerance(artifact):
    headline_tol = artifact["summary"]["headline_tol_pp"]
    for c in artifact["cells"]:
        if c["delta_pp"] is None:
            assert c["agree"] is None, (
                f"cell {c}: agree must be None when delta_pp is None"
            )
        else:
            expected = c["delta_pp"] <= headline_tol
            assert c["agree"] == expected, (
                f"cell {c}: agree {c['agree']!r} disagrees with "
                f"({c['delta_pp']} <= {headline_tol})"
            )


def test_doubly_saturated_means_both_below_floor(artifact):
    sat_floor = artifact["summary"]["sat_floor_pp"]
    for c in artifact["cells"]:
        if c["regime"] == "doubly_saturated":
            assert c["cache_sim_spread_pp"] < sat_floor
            assert c["anchor_spread_pp"] < sat_floor


def test_neither_saturated_means_both_above_floor(artifact):
    sat_floor = artifact["summary"]["sat_floor_pp"]
    for c in artifact["cells"]:
        if c["regime"] == "neither_saturated":
            assert c["cache_sim_spread_pp"] >= sat_floor
            assert c["anchor_spread_pp"] >= sat_floor


def test_single_saturated_means_exactly_one_below_floor(artifact):
    sat_floor = artifact["summary"]["sat_floor_pp"]
    for c in artifact["cells"]:
        if c["regime"] == "single_saturated":
            cs_sat = c["cache_sim_spread_pp"] < sat_floor
            a_sat = c["anchor_spread_pp"] < sat_floor
            assert cs_sat ^ a_sat, (
                f"cell {c}: single_saturated requires exactly one side "
                f"below floor — cs_sat={cs_sat} a_sat={a_sat}"
            )


def test_incomplete_means_a_spread_is_none(artifact):
    for c in artifact["cells"]:
        if c["regime"] == "incomplete":
            assert (c["cache_sim_spread_pp"] is None
                    or c["anchor_spread_pp"] is None), (
                f"cell {c}: incomplete requires at least one spread to be None"
            )


# ----------------------------------------------------------------------
# Group C: summary reducer parity
# ----------------------------------------------------------------------

def test_summary_n_cells_matches_cells_length(artifact):
    assert artifact["summary"]["n_cells"] == len(artifact["cells"])


def test_summary_regime_counts_match_per_cell(artifact):
    expected = defaultdict(int)
    for c in artifact["cells"]:
        expected[c["regime"]] += 1
    expected = dict(expected)
    assert artifact["summary"]["regime_counts"] == expected, (
        f"regime_counts drift — got {artifact['summary']['regime_counts']}, "
        f"expected {expected}"
    )


def test_summary_doubly_saturated_total_matches_per_cell(artifact):
    expected = sum(
        1 for c in artifact["cells"] if c["regime"] == "doubly_saturated"
    )
    assert artifact["summary"]["doubly_saturated_total"] == expected


def test_summary_doubly_saturated_agree_counts_only_doubly(artifact):
    """`agreed` only counts cells with regime=doubly_saturated AND agree=True."""
    expected = sum(
        1 for c in artifact["cells"]
        if c["regime"] == "doubly_saturated" and c["agree"] is True
    )
    assert artifact["summary"]["doubly_saturated_agree"] == expected


def test_summary_disagreements_only_doubly_with_agree_false(artifact):
    """Disagreements list contains exactly doubly-saturated cells with
    agree==False, with their (graph, app, tool, regime, delta_pp)."""
    expected = []
    for c in artifact["cells"]:
        if c["regime"] == "doubly_saturated" and c["agree"] is False:
            expected.append({
                "graph": c["graph"], "app": c["app"], "tool": c["tool"],
                "regime": c["regime"], "delta_pp": c["delta_pp"],
            })
    assert artifact["summary"]["disagreements"] == expected, (
        "disagreements list drift — must contain exactly doubly-saturated "
        "cells with agree==False"
    )


def test_summary_agreed_le_doubly_total(artifact):
    s = artifact["summary"]
    assert s["doubly_saturated_agree"] <= s["doubly_saturated_total"], (
        "agreed cannot exceed total doubly-saturated"
    )


def test_summary_doubly_total_le_n_cells(artifact):
    s = artifact["summary"]
    assert s["doubly_saturated_total"] <= s["n_cells"], (
        "doubly_saturated_total cannot exceed n_cells"
    )


def test_summary_regime_counts_sum_to_n_cells(artifact):
    s = artifact["summary"]
    assert sum(s["regime_counts"].values()) == s["n_cells"], (
        "regime_counts sum must equal n_cells"
    )


def test_summary_disagreements_count_consistent(artifact):
    """If agreed == total, disagreements must be empty."""
    s = artifact["summary"]
    if s["doubly_saturated_agree"] == s["doubly_saturated_total"]:
        assert not s["disagreements"], (
            "all doubly-saturated agreed but disagreements non-empty"
        )
