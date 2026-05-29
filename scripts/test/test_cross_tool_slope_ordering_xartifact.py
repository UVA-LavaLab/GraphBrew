"""Gate 89 — cross-tool slope-ordering universality cross-artifact gate.

Locks the universality of the SRRIP-steeper-than-GRASP slope claim and
the LRU regime-inversion claim across all three simulators (cache-sim,
gem5, Sniper) by cross-checking four sibling cross-tool artifacts:

  * wiki/data/cross_tool_slope_ordering.json
  * wiki/data/cross_tool_lru_regime.json
  * wiki/data/anchor_cross_tool_agreement.json
  * wiki/data/cross_tool_saturation.json

This is distinct from gate 76 (test_cross_tool_slope_universality.py),
which locks per-(tool, policy) median-slope band invariants on
cross_tool_slope_universality.json. Gate 89 instead locks the
*ordering* artifact (SRRIP vs GRASP gap on every tool) and the LRU
regime-inversion claim, and cross-checks the two against the anchor-
cell agreement summary and the doubly-saturated agreement summary.

The paper makes three load-bearing cross-tool claims that must hold
together: (a) SRRIP is strictly steeper than GRASP on every tool,
(b) LRU flips from not-strictly-steeper sub-WSS to strictly steeper
post-WSS, and (c) the anchor cells agree across tools on both sign
and on doubly-saturated winners. If any of these silently drifts on
a single tool, the universality narrative breaks.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI = REPO_ROOT / "wiki" / "data"

SLOPE_JSON = WIKI / "cross_tool_slope_ordering.json"
LRU_REGIME_JSON = WIKI / "cross_tool_lru_regime.json"
ANCHOR_AGREE_JSON = WIKI / "anchor_cross_tool_agreement.json"
SAT_JSON = WIKI / "cross_tool_saturation.json"

EXPECTED_TOOLS = ("cache_sim", "gem5", "sniper")

# The slope-ordering artifact reads its medians from these three
# specific upstream JSONs. Locking the mapping catches a silent change
# in upstream source.
EXPECTED_SOURCE_FOR_TOOL = {
    "cache_sim": "capacity_sensitivity.json",
    "gem5":      "gem5_slope_replay.json",
    "sniper":    "sniper_slope_replay.json",
}

PER_TOOL_REQUIRED_FIELDS = {
    "present",
    "source",
    "grasp_median",
    "srrip_median",
    "lru_median",
    "srrip_minus_grasp_pp_oct",
    "lru_minus_grasp_pp_oct",
    "srrip_steeper",
    "srrip_strictly_steeper",
}


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _slope() -> dict:
    return _load(SLOPE_JSON)


def _lru_regime() -> dict:
    return _load(LRU_REGIME_JSON)


def _anchor() -> dict:
    return _load(ANCHOR_AGREE_JSON)


def _sat() -> dict:
    return _load(SAT_JSON)


# ---------------------------------------------------------------------------
# Slope-ordering: per-tool coverage and shape
# ---------------------------------------------------------------------------

def test_slope_ordering_has_all_three_tools():
    per_tool = _slope()["per_tool"]
    assert set(per_tool.keys()) == set(EXPECTED_TOOLS), (
        f"cross_tool_slope_ordering per_tool keyset diverged: "
        f"got {sorted(per_tool.keys())}, expected {sorted(EXPECTED_TOOLS)}"
    )


def test_every_tool_carries_the_full_required_field_set():
    per_tool = _slope()["per_tool"]
    missing = {
        tool: PER_TOOL_REQUIRED_FIELDS - set(body.keys())
        for tool, body in per_tool.items()
        if PER_TOOL_REQUIRED_FIELDS - set(body.keys())
    }
    assert not missing, f"per_tool entries missing required fields: {missing}"


def test_every_tool_sources_from_the_expected_upstream_json():
    per_tool = _slope()["per_tool"]
    bad = []
    for tool, want_source in EXPECTED_SOURCE_FOR_TOOL.items():
        got = per_tool[tool]["source"]
        if got != want_source:
            bad.append((tool, got, want_source))
    assert not bad, (
        f"upstream source filename drift in cross_tool_slope_ordering: "
        f"{bad} — slope-ordering reads from a different file than expected."
    )


# ---------------------------------------------------------------------------
# Slope-ordering: paper-claim invariants on every tool
# ---------------------------------------------------------------------------

def test_srrip_strictly_steeper_than_grasp_on_every_tool():
    """Paper claim: SRRIP is strictly steeper than GRASP on every
    simulator. This is the universality claim that, if it weakens to
    only 2/3 tools, falls back to a quorum claim instead. Lock the
    universal version."""
    per_tool = _slope()["per_tool"]
    failures = [
        tool for tool, body in per_tool.items()
        if not body.get("srrip_strictly_steeper")
    ]
    assert not failures, (
        "srrip_strictly_steeper regressed on tools: "
        f"{failures}; was previously True for all 3 tools."
    )


def test_srrip_minus_grasp_pp_oct_clears_the_gap_floor_on_every_tool():
    """The slope-ordering gate has a configurable gap floor (default
    0.05 pp/oct). Every tool's srrip_minus_grasp slope difference must
    be more-negative-than -floor (i.e., a real, beyond-noise gap)."""
    d = _slope()
    floor = d["meta"]["gap_floor_pp_octave"]
    failures = []
    for tool, body in d["per_tool"].items():
        gap = body["srrip_minus_grasp_pp_oct"]
        if not (gap < -floor):
            failures.append((tool, gap, -floor))
    assert not failures, (
        f"srrip-minus-grasp slope gap does not clear the -{floor} pp/oct "
        f"floor for tools: {failures}"
    )


def test_grasp_median_slope_is_downward_on_every_tool():
    """Curves slope down with cache capacity (more cache → lower miss
    rate). GRASP's median slope must therefore be negative on every
    tool, or our axis interpretation has flipped."""
    per_tool = _slope()["per_tool"]
    failures = [
        (tool, body["grasp_median"])
        for tool, body in per_tool.items()
        if not (body["grasp_median"] < 0)
    ]
    assert not failures, (
        f"grasp_median is not strictly downward on tools: {failures}"
    )


def test_slope_ordering_aggregate_verdict_is_pass():
    d = _slope()
    assert d["meta"]["verdict"] == "PASS", (
        f"slope-ordering aggregate verdict regressed: {d['meta']['verdict']}"
    )
    for check, ok in d["meta"]["verdict_checks"].items():
        assert ok, f"slope-ordering verdict_check {check!r} regressed to {ok}"


def test_slope_ordering_strict_tool_count_meets_required_quorum():
    meta = _slope()["meta"]
    assert meta["n_strict_tools"] >= meta["required_strict_tools"], (
        f"only {meta['n_strict_tools']} tools strictly steeper "
        f"(required {meta['required_strict_tools']})"
    )
    assert meta["n_strict_tools"] == 3, (
        "n_strict_tools regressed from full universality (3) to "
        f"{meta['n_strict_tools']}"
    )


# ---------------------------------------------------------------------------
# LRU regime-inversion sister claim
# ---------------------------------------------------------------------------

def test_cross_tool_lru_regime_verdict_is_pass():
    d = _lru_regime()
    assert d["meta"]["verdict"] == "PASS", (
        f"cross_tool_lru_regime verdict regressed: {d['meta']['verdict']}"
    )


def test_every_lru_regime_check_holds():
    """All 5 LRU regime-inversion checks must hold (cache_sim post-WSS
    LRU steeper, gem5/Sniper sub-WSS LRU not strictly steeper, regime-
    inversion sign holds, regime labels correct)."""
    checks = _lru_regime()["meta"]["verdict_checks"]
    failed = [k for k, v in checks.items() if not v]
    assert not failed, f"cross_tool_lru_regime checks regressed: {failed}"
    assert len(checks) >= 5, (
        f"expected ≥5 regime checks, got {len(checks)}: {sorted(checks)}"
    )


def test_lru_regime_inversion_sign_holds_explicitly():
    """The sign-inversion check is the single load-bearing aggregate;
    pin it by name so a refactor that drops it from verdict_checks
    fails this gate explicitly."""
    d = _lru_regime()
    assert d["meta"]["regime_inversion_holds"] is True, (
        f"regime_inversion_holds regressed to "
        f"{d['meta']['regime_inversion_holds']}"
    )
    assert d["meta"]["verdict_checks"]["regime_inversion_sign_holds"] is True


# ---------------------------------------------------------------------------
# Anchor cross-tool agreement (per-cell, not per-tool-median)
# ---------------------------------------------------------------------------

def test_anchor_cross_tool_agreement_verdict_ok():
    d = _anchor()
    assert d["verdict_ok"] is True, (
        f"anchor_cross_tool_agreement verdict_ok regressed: {d['verdict_ok']}"
    )


def test_anchor_sign_agree_count_at_least_both_negative_count():
    """Every anchor cell where both tools' slopes are negative must
    also be a sign-agreement cell (you cannot disagree on sign when
    both values are <0). This catches a counting bug in the comparator."""
    s = _anchor()["summary"]
    assert s["sign_agree_count"] >= s["both_negative_count"], (
        f"sign_agree_count ({s['sign_agree_count']}) < "
        f"both_negative_count ({s['both_negative_count']}) — "
        "comparator counted disagreement where signs must agree"
    )


# ---------------------------------------------------------------------------
# Saturation sister claim
# ---------------------------------------------------------------------------

def test_cross_tool_saturation_doubly_saturated_cells_all_agree():
    s = _sat()["summary"]
    assert s["doubly_saturated_agree"] == s["doubly_saturated_total"], (
        "cross_tool_saturation has a doubly-saturated cell where the "
        "two tools disagree on the winner; broke universality. "
        f"({s['doubly_saturated_agree']}/{s['doubly_saturated_total']})"
    )
    assert s["doubly_saturated_total"] >= 1, (
        "no doubly-saturated cells found — saturation comparator may "
        "have lost its anchor coverage"
    )
