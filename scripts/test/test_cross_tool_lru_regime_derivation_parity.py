"""Derivation parity gate for ``wiki/data/cross_tool_lru_regime.json``.

Locks the cross-tool LRU-vs-GRASP regime-inversion artifact (gate 74)
against its three upstream slope sources so any silent change in the
slope reducers, the regime-classifier thresholds, or the verdict
predicates trips a test before the dashboard re-publishes:

    capacity_sensitivity.json      → cache-sim per-policy median pp/oct
    gem5_slope_replay.json         → gem5 per-policy median pp/oct
    sniper_slope_replay.json       → sniper per-policy median pp/oct
                  │
        cross_tool_lru_regime.py:build()
                  │
                  ▼
        wiki/data/cross_tool_lru_regime.json   ← gate target

Cross-source ground truth:

* For each tool we re-extract GRASP and LRU median pp/octave from the
  upstream JSON and recompute ``lru_minus_grasp = lru - grasp``; the
  artifact's stored ``lru_minus_grasp_pp_oct`` must match to 4dp.
* Regime classification ``_classify_regime`` is a pure function of
  ``(l3_min_kb, l3_max_kb)`` from ``TOOL_L3_RANGE_KB``; mirroring those
  thresholds here lets us re-derive each tool's regime label.
* Verdict is the AND of five well-defined boolean predicates; we
  reconstruct each predicate from first principles and compare against
  the stored ``verdict_checks`` map plus the headline ``verdict``.

If anyone silently relaxes the post-WSS floor, the sub-WSS tolerance,
or the regime-inversion sign predicate, this gate breaks immediately.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

ARTIFACT_PATH = WIKI_DATA / "cross_tool_lru_regime.json"
CACHE_SIM_PATH = WIKI_DATA / "capacity_sensitivity.json"
GEM5_PATH = WIKI_DATA / "gem5_slope_replay.json"
SNIPER_PATH = WIKI_DATA / "sniper_slope_replay.json"

# Pinned mirror of the generator constants in cross_tool_lru_regime.py.
POSTWSS_GAP_FLOOR_PP_OCT = 0.30
SUBWSS_TOLERANCE_PP = 0.20
TOOL_L3_RANGE_KB = {
    "cache-sim": (1024.0, 8192.0),
    "gem5":      (4.0,    2048.0),
    "sniper":    (4.0,    2048.0),
}

TOOL_PATHS = {
    "cache-sim": CACHE_SIM_PATH,
    "gem5":      GEM5_PATH,
    "sniper":    SNIPER_PATH,
}


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
    """For each tool, return a dict policy -> median pp/octave parsed
    from the upstream slope JSON using the exact same accessor the
    generator uses (``policy_summary[p].median_pp`` for cache-sim,
    ``per_policy[p].median`` for gem5/sniper).
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
    assert set(artifact.keys()) == {"meta"}, (
        f"unexpected top-level keys: {sorted(artifact.keys())}"
    )


def test_meta_carries_canonical_fields(meta):
    expected = {
        "tools", "postwss_gap_floor_pp_oct", "subwss_tolerance_pp",
        "tool_results", "regime_inversion_holds", "verdict_checks",
        "verdict",
    }
    missing = expected - set(meta.keys())
    assert not missing, f"missing meta fields: {missing}"


def test_tools_list_matches_tool_l3_range(meta):
    expected = [
        {"name": t, "l3_min_kb": lo, "l3_max_kb": hi}
        for t, (lo, hi) in TOOL_L3_RANGE_KB.items()
    ]
    assert meta["tools"] == expected, (
        f"meta.tools drift from TOOL_L3_RANGE_KB:\n  expected={expected}\n  actual={meta['tools']}"
    )


def test_tool_results_keys_match_tool_l3_range(meta):
    assert sorted(meta["tool_results"].keys()) == sorted(TOOL_L3_RANGE_KB.keys())


def test_per_tool_result_has_canonical_fields(meta):
    expected = {
        "grasp_pp_oct", "lru_pp_oct", "lru_minus_grasp_pp_oct",
        "l3_min_kb", "l3_max_kb", "regime",
    }
    for tool, t in meta["tool_results"].items():
        missing = expected - set(t.keys())
        assert not missing, f"{tool} missing fields: {missing}"


def test_constants_pinned(meta):
    assert meta["postwss_gap_floor_pp_oct"] == POSTWSS_GAP_FLOOR_PP_OCT, (
        "POSTWSS_GAP_FLOOR_PP_OCT drifted from 0.30 — "
        "tightening this threshold may silently flip the post-WSS verdict."
    )
    assert meta["subwss_tolerance_pp"] == SUBWSS_TOLERANCE_PP, (
        "SUBWSS_TOLERANCE_PP drifted from 0.20 — "
        "loosening this threshold may silently mask a sub-WSS inversion."
    )


# ----------------------------------------------------------------------
# Group B: per-tool median cross-source parity
# ----------------------------------------------------------------------

def test_per_tool_grasp_matches_upstream_to_4dp(meta, upstream_medians):
    for tool, block in meta["tool_results"].items():
        upstream = upstream_medians[tool].get("GRASP")
        assert upstream is not None, f"{tool} upstream missing GRASP"
        assert block["grasp_pp_oct"] == round(upstream, 4), (
            f"{tool}: GRASP median drift — upstream={upstream!r} round4={round(upstream,4)!r}, "
            f"artifact={block['grasp_pp_oct']!r}"
        )


def test_per_tool_lru_matches_upstream_to_4dp(meta, upstream_medians):
    for tool, block in meta["tool_results"].items():
        upstream = upstream_medians[tool].get("LRU")
        assert upstream is not None, f"{tool} upstream missing LRU"
        assert block["lru_pp_oct"] == round(upstream, 4), (
            f"{tool}: LRU median drift — upstream={upstream!r} round4={round(upstream,4)!r}, "
            f"artifact={block['lru_pp_oct']!r}"
        )


def test_per_tool_lru_minus_grasp_matches_upstream_subtraction(meta, upstream_medians):
    """Generator computes ``delta = (l - g)`` on the RAW upstream values
    then rounds to 4dp; verify byte-exact via the same recipe.
    """
    for tool, block in meta["tool_results"].items():
        g_raw = upstream_medians[tool].get("GRASP")
        l_raw = upstream_medians[tool].get("LRU")
        expected = round(l_raw - g_raw, 4)
        assert block["lru_minus_grasp_pp_oct"] == expected, (
            f"{tool}: lru_minus_grasp_pp_oct drift — expected (l-g, round4)={expected!r}, "
            f"got {block['lru_minus_grasp_pp_oct']!r}"
        )


def test_per_tool_l3_range_matches_pinned(meta):
    for tool, block in meta["tool_results"].items():
        lo, hi = TOOL_L3_RANGE_KB[tool]
        assert block["l3_min_kb"] == lo
        assert block["l3_max_kb"] == hi


# ----------------------------------------------------------------------
# Group C: regime classifier
# ----------------------------------------------------------------------

def _classify_regime(lo_kb: float, hi_kb: float) -> str:
    """Mirror of generator's ``_classify_regime``."""
    if hi_kb <= 4096.0:
        return "sub-WSS"
    if lo_kb >= 1024.0:
        return "post-WSS"
    return "mixed"


def test_regime_label_matches_classifier(meta):
    for tool, block in meta["tool_results"].items():
        expected = _classify_regime(block["l3_min_kb"], block["l3_max_kb"])
        assert block["regime"] == expected, (
            f"{tool} regime drift: range=({block['l3_min_kb']}, {block['l3_max_kb']}) "
            f"→ expected {expected!r}, got {block['regime']!r}"
        )


def test_cache_sim_is_post_wss(meta):
    assert meta["tool_results"]["cache-sim"]["regime"] == "post-WSS", (
        "cache-sim L3 sweep (1MB–8MB) must classify as post-WSS for our corpus; "
        "any drift here invalidates the whole regime-inversion claim."
    )


def test_gem5_and_sniper_are_sub_wss(meta):
    for tool in ("gem5", "sniper"):
        assert meta["tool_results"][tool]["regime"] == "sub-WSS", (
            f"{tool} L3 sweep (4kB–2MB) must classify as sub-WSS; otherwise the "
            "cross-tool inversion check is comparing apples to oranges."
        )


# ----------------------------------------------------------------------
# Group D: verdict predicates
# ----------------------------------------------------------------------

def _recompute_verdict_checks(meta) -> dict[str, bool]:
    """Mirror the five verdict predicates from cross_tool_lru_regime.py."""
    cs = meta["tool_results"]["cache-sim"]
    g5 = meta["tool_results"]["gem5"]
    sn = meta["tool_results"]["sniper"]
    cs_d = cs["lru_minus_grasp_pp_oct"]
    g5_d = g5["lru_minus_grasp_pp_oct"]
    sn_d = sn["lru_minus_grasp_pp_oct"]
    return {
        "cache_sim_postwss_LRU_steeper":
            cs_d is not None and cs_d <= -POSTWSS_GAP_FLOOR_PP_OCT,
        "gem5_subwss_LRU_not_strictly_steeper":
            g5_d is not None and g5_d >= -SUBWSS_TOLERANCE_PP,
        "sniper_subwss_LRU_not_strictly_steeper":
            sn_d is not None and sn_d >= -SUBWSS_TOLERANCE_PP,
        "regime_inversion_sign_holds":
            cs_d is not None and g5_d is not None and sn_d is not None
            and cs_d < 0.0 and g5_d >= 0.0 and sn_d >= 0.0,
        "regime_labels_correct":
            cs["regime"] == "post-WSS"
            and g5["regime"] == "sub-WSS"
            and sn["regime"] == "sub-WSS",
    }


def test_verdict_checks_have_canonical_keys(meta):
    expected = {
        "cache_sim_postwss_LRU_steeper",
        "gem5_subwss_LRU_not_strictly_steeper",
        "sniper_subwss_LRU_not_strictly_steeper",
        "regime_inversion_sign_holds",
        "regime_labels_correct",
    }
    assert set(meta["verdict_checks"].keys()) == expected, (
        f"verdict_checks key set drift: missing {expected - set(meta['verdict_checks'].keys())}, "
        f"extra {set(meta['verdict_checks'].keys()) - expected}"
    )


def test_each_verdict_check_matches_recomputation(meta):
    recomputed = _recompute_verdict_checks(meta)
    for k, expected in recomputed.items():
        actual = meta["verdict_checks"][k]
        assert actual == expected, (
            f"verdict_checks[{k}] drift: recomputed={expected}, artifact={actual}"
        )


def test_regime_inversion_holds_matches_sign_predicate(meta):
    recomputed = _recompute_verdict_checks(meta)
    assert meta["regime_inversion_holds"] == recomputed["regime_inversion_sign_holds"], (
        "regime_inversion_holds and regime_inversion_sign_holds must agree — "
        "they are two views of the same predicate (cs<0 AND g5>=0 AND sn>=0)."
    )


def test_verdict_is_and_of_all_checks(meta):
    expected = "PASS" if all(meta["verdict_checks"].values()) else "FAIL"
    assert meta["verdict"] == expected, (
        f"verdict drift: AND-of-checks = {expected}, artifact = {meta['verdict']!r}"
    )


def test_current_run_is_pass(meta):
    """Pin the current PASS state so any future drift in the slope
    sources surfaces as a confidence-gate red flag rather than a silent
    sign flip in the artifact.
    """
    assert meta["verdict"] == "PASS", (
        f"cross_tool_lru_regime verdict regressed to {meta['verdict']!r}; "
        "check upstream slope JSONs for sign drift before relaxing this gate."
    )
    assert meta["regime_inversion_holds"] is True


# ----------------------------------------------------------------------
# Group E: physical-direction sanity (regime story)
# ----------------------------------------------------------------------

def test_cache_sim_lru_strictly_steeper_than_grasp(meta):
    cs = meta["tool_results"]["cache-sim"]
    # Both should be negative slopes; LRU more negative.
    assert cs["lru_pp_oct"] < cs["grasp_pp_oct"], (
        "cache-sim must show LRU's slope more negative than GRASP's "
        "(post-WSS LRU-steeper invariant)"
    )


def test_anchor_tools_lru_not_strictly_steeper_than_grasp(meta):
    """Within the SUBWSS_TOLERANCE_PP band, neither anchor tool may
    have LRU strictly steeper than GRASP — that would collapse the
    regime story.
    """
    for tool in ("gem5", "sniper"):
        d = meta["tool_results"][tool]["lru_minus_grasp_pp_oct"]
        assert d >= -SUBWSS_TOLERANCE_PP, (
            f"{tool} sub-WSS delta {d} pp/oct breaches tolerance "
            f"-{SUBWSS_TOLERANCE_PP}; LRU is becoming strictly steeper "
            "than GRASP at sub-WSS scales, which contradicts the gate's claim."
        )
