"""Derivation parity gate for ``wiki/data/cross_tool_slope_universality.json``.

Locks the cross-tool slope-sign universality artifact (gate 76)
against its three upstream slope sources so any silent shift in the
slope reducers, the physical-band thresholds, or the verdict
predicates trips a test before the dashboard re-publishes:

    capacity_sensitivity.json     → cache-sim per-policy median pp/oct
    gem5_slope_replay.json        → gem5 per-policy median pp/oct
    sniper_slope_replay.json      → sniper per-policy median pp/oct
                  │
        cross_tool_slope_universality.py:build()
                  │
                  ▼
    wiki/data/cross_tool_slope_universality.json   ← gate target

Why this matters: this is the only artifact that gathers every (tool,
policy) slope in one place and asserts they are all strictly negative
and within a documented physical band. A regression that flips a
single (tool, policy) median positive — or collapses it to near zero
— would silently break a core "extra cache helps every policy" claim
of the paper.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

ARTIFACT_PATH = WIKI_DATA / "cross_tool_slope_universality.json"
CACHE_SIM_PATH = WIKI_DATA / "capacity_sensitivity.json"
GEM5_PATH = WIKI_DATA / "gem5_slope_replay.json"
SNIPER_PATH = WIKI_DATA / "sniper_slope_replay.json"

# Pinned mirror of the generator constants.
MIN_SLOPE_PP_OCT = -25.0
MAX_SLOPE_PP_OCT = -0.5
STEEPNESS_SPAN_CEILING_PP_OCT = 5.0
EXPECTED_TOOLS = ["cache-sim", "gem5", "sniper"]


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def artifact() -> dict:
    if not ARTIFACT_PATH.exists():
        pytest.skip(f"missing {ARTIFACT_PATH}")
    return json.loads(ARTIFACT_PATH.read_text())


@pytest.fixture(scope="module")
def meta(artifact) -> dict:
    return artifact["meta"]


@pytest.fixture(scope="module")
def upstream_medians() -> dict[str, dict[str, float]]:
    """Re-extract per-tool, per-policy median pp/octave from the three
    upstream JSONs using the exact same accessor the generator uses.
    """
    out: dict[str, dict[str, float]] = {}
    if not CACHE_SIM_PATH.exists():
        pytest.skip(f"missing {CACHE_SIM_PATH}")
    cs_doc = json.loads(CACHE_SIM_PATH.read_text())
    out["cache-sim"] = {
        pol: float(block["median_pp"])
        for pol, block in cs_doc["meta"]["policy_summary"].items()
    }
    if not GEM5_PATH.exists():
        pytest.skip(f"missing {GEM5_PATH}")
    gem5_doc = json.loads(GEM5_PATH.read_text())
    out["gem5"] = {
        pol: float(block["median"])
        for pol, block in gem5_doc["meta"]["per_policy"].items()
    }
    if not SNIPER_PATH.exists():
        pytest.skip(f"missing {SNIPER_PATH}")
    sn_doc = json.loads(SNIPER_PATH.read_text())
    out["sniper"] = {
        pol: float(block["median"])
        for pol, block in sn_doc["meta"]["per_policy"].items()
    }
    return out


# ----------------------------------------------------------------------
# Group A: shape & schema
# ----------------------------------------------------------------------

def test_artifact_has_meta_only(artifact):
    assert set(artifact.keys()) == {"meta"}


def test_meta_carries_canonical_fields(meta):
    expected = {
        "tools", "tool_policies", "medians", "steepness_spans",
        "in_band_count", "expected_in_band_count", "violations",
        "min_slope_pp_oct", "max_slope_pp_oct",
        "steepness_span_ceiling_pp_oct",
        "verdict_checks", "verdict",
    }
    missing = expected - set(meta.keys())
    assert not missing, f"missing meta fields: {missing}"


def test_tools_list_matches_expected(meta):
    assert meta["tools"] == EXPECTED_TOOLS, (
        f"meta.tools drift: expected {EXPECTED_TOOLS}, got {meta['tools']}"
    )


def test_thresholds_pinned(meta):
    assert meta["min_slope_pp_oct"] == MIN_SLOPE_PP_OCT, (
        "MIN_SLOPE_PP_OCT drift — tightening this band could mask "
        "runaway-steep regressions."
    )
    assert meta["max_slope_pp_oct"] == MAX_SLOPE_PP_OCT, (
        "MAX_SLOPE_PP_OCT drift — loosening this ceiling could allow "
        "a policy to collapse to near-zero slope without firing a "
        "violation."
    )
    assert meta["steepness_span_ceiling_pp_oct"] == STEEPNESS_SPAN_CEILING_PP_OCT, (
        "STEEPNESS_SPAN_CEILING_PP_OCT drift — loosening this ceiling "
        "could hide partial regressions where one policy collapses "
        "while siblings stay steep."
    )


# ----------------------------------------------------------------------
# Group B: per-(tool, policy) median cross-source parity
# ----------------------------------------------------------------------

def test_tool_policies_match_upstream_sorted_keys(meta, upstream_medians):
    """``tool_policies[t]`` is ``sorted(medians.keys())`` from the
    upstream JSON — verify the per-tool policy list exactly.
    """
    for tool in EXPECTED_TOOLS:
        expected = sorted(upstream_medians[tool].keys())
        assert meta["tool_policies"][tool] == expected, (
            f"{tool}: tool_policies drift — expected {expected}, "
            f"got {meta['tool_policies'][tool]}"
        )


def test_medians_match_upstream_rounded_to_4dp(meta, upstream_medians):
    for tool in EXPECTED_TOOLS:
        for pol, raw in upstream_medians[tool].items():
            stored = meta["medians"][tool].get(pol)
            assert stored is not None, (
                f"{tool}: policy {pol} missing from medians"
            )
            assert stored == round(raw, 4), (
                f"{tool}/{pol}: median drift — upstream={raw!r} "
                f"round4={round(raw, 4)!r} artifact={stored!r}"
            )


def test_medians_have_no_extra_policies(meta, upstream_medians):
    for tool in EXPECTED_TOOLS:
        assert set(meta["medians"][tool].keys()) == set(upstream_medians[tool].keys())


def test_cache_sim_carries_all_four_policies(meta):
    assert sorted(meta["medians"]["cache-sim"].keys()) == ["GRASP", "LRU", "POPT", "SRRIP"]


def test_anchor_tools_carry_three_policies(meta):
    for tool in ("gem5", "sniper"):
        assert sorted(meta["medians"][tool].keys()) == ["GRASP", "LRU", "SRRIP"], (
            f"{tool}: anchor tools should have GRASP/LRU/SRRIP only "
            "(POPT is cache-sim only)"
        )


# ----------------------------------------------------------------------
# Group C: steepness span
# ----------------------------------------------------------------------

def test_steepness_span_equals_max_minus_min_round_4dp(meta, upstream_medians):
    for tool in EXPECTED_TOOLS:
        vals = list(upstream_medians[tool].values())
        expected = round(max(vals) - min(vals), 4)
        assert meta["steepness_spans"][tool] == expected, (
            f"{tool}: steepness_span drift — expected {expected!r} "
            f"(max-min on raw upstream), got {meta['steepness_spans'][tool]!r}"
        )


def test_steepness_span_is_nonnegative(meta):
    for tool, span in meta["steepness_spans"].items():
        assert span >= 0.0, f"{tool}: negative span {span}"


# ----------------------------------------------------------------------
# Group D: band membership & violations
# ----------------------------------------------------------------------

def _recompute_violations(medians: dict[str, dict[str, float]],
                          spans: dict[str, float]) -> tuple[list[dict], int, int]:
    """Mirror of generator's violation-collection loop."""
    violations: list[dict] = []
    in_band = 0
    expected = sum(len(p) for p in medians.values())
    for t in medians:
        for pol, val in medians[t].items():
            if val >= 0.0:
                violations.append({
                    "type": "positive_slope", "tool": t, "policy": pol,
                    "value": round(val, 4),
                })
            if val < MIN_SLOPE_PP_OCT or val > MAX_SLOPE_PP_OCT:
                violations.append({
                    "type": "out_of_band", "tool": t, "policy": pol,
                    "value": round(val, 4),
                    "band": [MIN_SLOPE_PP_OCT, MAX_SLOPE_PP_OCT],
                })
            else:
                in_band += 1
    for t, span in spans.items():
        if span > STEEPNESS_SPAN_CEILING_PP_OCT:
            violations.append({
                "type": "steepness_span_exceeded", "tool": t,
                "span": span, "ceiling": STEEPNESS_SPAN_CEILING_PP_OCT,
            })
    return violations, in_band, expected


def test_in_band_count_matches_recomputation(meta, upstream_medians):
    _, in_band, expected = _recompute_violations(
        upstream_medians, meta["steepness_spans"]
    )
    assert meta["in_band_count"] == in_band, (
        f"in_band_count drift: recomputed={in_band}, artifact={meta['in_band_count']}"
    )
    assert meta["expected_in_band_count"] == expected, (
        f"expected_in_band_count drift: recomputed={expected}, "
        f"artifact={meta['expected_in_band_count']}"
    )


def test_violations_match_recomputation(meta, upstream_medians):
    expected_violations, _, _ = _recompute_violations(
        upstream_medians, meta["steepness_spans"]
    )
    # Both lists are computed in the same iteration order; the artifact
    # preserves dict insertion order for medians, so reconstruction
    # matches element-for-element.
    assert meta["violations"] == expected_violations, (
        f"violations drift:\n  expected={expected_violations}\n  artifact={meta['violations']}"
    )


def test_no_violations_in_current_run(meta):
    assert meta["violations"] == [], (
        f"slope universality violations detected: {meta['violations']}"
    )


def test_in_band_equals_expected(meta):
    assert meta["in_band_count"] == meta["expected_in_band_count"], (
        f"{meta['in_band_count']}/{meta['expected_in_band_count']} "
        "in-band — at least one (tool, policy) slope median is outside "
        "the physical band [-25, -0.5] pp/oct"
    )


# ----------------------------------------------------------------------
# Group E: verdict predicates
# ----------------------------------------------------------------------

def _recompute_verdict_checks(medians: dict[str, dict[str, float]],
                              spans: dict[str, float]) -> dict[str, bool]:
    all_negative = all(
        v < 0.0 for med in medians.values() for v in med.values()
    )
    # in_band derives from the violation loop; reuse helper
    _, in_band, expected = _recompute_violations(medians, spans)
    all_in_band = (in_band == expected)
    no_span_exceeded = all(
        s <= STEEPNESS_SPAN_CEILING_PP_OCT for s in spans.values()
    )
    return {
        "all_tool_policy_medians_negative":       all_negative,
        "all_tool_policy_medians_in_band":        all_in_band,
        "no_tool_exceeds_steepness_span_ceiling": no_span_exceeded,
    }


def test_verdict_checks_keys(meta):
    expected = {
        "all_tool_policy_medians_negative",
        "all_tool_policy_medians_in_band",
        "no_tool_exceeds_steepness_span_ceiling",
    }
    assert set(meta["verdict_checks"].keys()) == expected


def test_verdict_checks_match_recomputation(meta, upstream_medians):
    recomputed = _recompute_verdict_checks(upstream_medians, meta["steepness_spans"])
    for k, expected in recomputed.items():
        assert meta["verdict_checks"][k] == expected, (
            f"verdict_checks[{k}] drift: recomputed={expected}, "
            f"artifact={meta['verdict_checks'][k]}"
        )


def test_verdict_is_and_of_all_checks(meta):
    expected = "PASS" if all(meta["verdict_checks"].values()) else "FAIL"
    assert meta["verdict"] == expected


def test_current_verdict_is_pass(meta):
    assert meta["verdict"] == "PASS", (
        f"slope universality regressed to {meta['verdict']!r}"
    )


def test_every_median_is_strictly_negative(meta):
    """Physical sanity: 'more cache → fewer misses on average' must
    hold for every (tool, policy) cell."""
    for tool, med in meta["medians"].items():
        for pol, val in med.items():
            assert val < 0.0, (
                f"{tool}/{pol} median {val} pp/oct >= 0 — extra cache "
                "is HURTING this policy on average"
            )
