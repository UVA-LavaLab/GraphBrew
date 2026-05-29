"""Derivation parity gate for ``wiki/data/cross_tool_slope_ordering.json``.

Locks the cross-tool SRRIP-vs-GRASP slope ordering artifact (gate 72)
against its three upstream slope sources so any silent drift in the
slope reducers, the gap-floor threshold, or the verdict predicates
trips a test before the dashboard re-publishes:

    capacity_sensitivity.json     → cache-sim per-policy median pp/oct
    gem5_slope_replay.json        → gem5 per-policy median pp/oct
    sniper_slope_replay.json      → sniper per-policy median pp/oct
                  │
        cross_tool_slope_ordering.py:compute()
                  │
                  ▼
    wiki/data/cross_tool_slope_ordering.json   ← gate target

The gated claim: GRASP (oracle-aware) has a shallower capacity-
sensitivity slope than SRRIP (non-oracle-aware), so SRRIP-GRASP is
non-positive on every tool and STRICTLY negative on at least two of
three tools. The LRU-vs-GRASP delta is reported but NOT gated (it's
regime-dependent; cross_tool_lru_regime owns that check).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

ARTIFACT_PATH = WIKI_DATA / "cross_tool_slope_ordering.json"
CACHE_SIM_PATH = WIKI_DATA / "capacity_sensitivity.json"
GEM5_PATH = WIKI_DATA / "gem5_slope_replay.json"
SNIPER_PATH = WIKI_DATA / "sniper_slope_replay.json"

# Pinned mirror of generator constants.
GAP_FLOOR_PP_OCTAVE = 0.05
REQUIRED_STRICT_TOOLS = 2
EXPECTED_TOOLS = ["cache_sim", "gem5", "sniper"]
SOURCE_NAMES = {
    "cache_sim": "capacity_sensitivity.json",
    "gem5": "gem5_slope_replay.json",
    "sniper": "sniper_slope_replay.json",
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
def per_tool(artifact) -> dict[str, dict]:
    return artifact["per_tool"]


@pytest.fixture(scope="module")
def upstream_medians() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if not CACHE_SIM_PATH.exists():
        pytest.skip(f"missing {CACHE_SIM_PATH}")
    cs_doc = json.loads(CACHE_SIM_PATH.read_text())
    out["cache_sim"] = {
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

def test_top_level_keys(artifact):
    assert set(artifact.keys()) == {"meta", "per_tool"}


def test_meta_carries_canonical_fields(meta):
    expected = {
        "tools", "gap_floor_pp_octave", "required_strict_tools",
        "n_strict_tools", "verdict_checks", "verdict",
    }
    missing = expected - set(meta.keys())
    assert not missing, f"missing meta fields: {missing}"


def test_tools_list_is_canonical(meta):
    assert meta["tools"] == EXPECTED_TOOLS, (
        f"meta.tools drift: expected {EXPECTED_TOOLS}, got {meta['tools']}"
    )


def test_per_tool_keys_match_tools(meta, per_tool):
    assert sorted(per_tool.keys()) == sorted(meta["tools"])


def test_per_tool_carries_canonical_fields(per_tool):
    expected = {
        "present", "source", "grasp_median", "srrip_median",
        "lru_median", "srrip_minus_grasp_pp_oct",
        "lru_minus_grasp_pp_oct", "srrip_steeper",
        "srrip_strictly_steeper",
    }
    for tool, e in per_tool.items():
        missing = expected - set(e.keys())
        assert not missing, f"{tool}: per_tool missing fields {missing}"


def test_thresholds_pinned(meta):
    assert meta["gap_floor_pp_octave"] == GAP_FLOOR_PP_OCTAVE, (
        "GAP_FLOOR_PP_OCTAVE drifted from 0.05 — loosening this "
        "threshold could let a near-zero gap masquerade as agreement."
    )
    assert meta["required_strict_tools"] == REQUIRED_STRICT_TOOLS, (
        "REQUIRED_STRICT_TOOLS drifted from 2 — relaxing this could "
        "mask a one-tool-only result as cross-tool universality."
    )


def test_source_names_match_pinned(per_tool):
    for tool, name in SOURCE_NAMES.items():
        assert per_tool[tool]["source"] == name, (
            f"{tool}: source name drift — expected {name!r}, "
            f"got {per_tool[tool]['source']!r}"
        )


# ----------------------------------------------------------------------
# Group B: per-tool median cross-source parity
# ----------------------------------------------------------------------

def test_grasp_median_matches_upstream(per_tool, upstream_medians):
    for tool in EXPECTED_TOOLS:
        upstream = upstream_medians[tool].get("GRASP")
        assert upstream is not None, f"{tool}: upstream missing GRASP"
        assert per_tool[tool]["grasp_median"] == round(upstream, 4), (
            f"{tool}: GRASP median drift — upstream={upstream!r} "
            f"round4={round(upstream, 4)!r} artifact={per_tool[tool]['grasp_median']!r}"
        )


def test_srrip_median_matches_upstream(per_tool, upstream_medians):
    for tool in EXPECTED_TOOLS:
        upstream = upstream_medians[tool].get("SRRIP")
        assert upstream is not None, f"{tool}: upstream missing SRRIP"
        assert per_tool[tool]["srrip_median"] == round(upstream, 4), (
            f"{tool}: SRRIP median drift — upstream={upstream!r} "
            f"round4={round(upstream, 4)!r} artifact={per_tool[tool]['srrip_median']!r}"
        )


def test_lru_median_matches_upstream(per_tool, upstream_medians):
    for tool in EXPECTED_TOOLS:
        upstream = upstream_medians[tool].get("LRU")
        if upstream is None:
            assert per_tool[tool]["lru_median"] is None
        else:
            assert per_tool[tool]["lru_median"] == round(upstream, 4), (
                f"{tool}: LRU median drift — upstream={upstream!r} "
                f"round4={round(upstream, 4)!r} artifact={per_tool[tool]['lru_median']!r}"
            )


def test_srrip_minus_grasp_equals_raw_subtraction(per_tool, upstream_medians):
    """Generator computes ``srrip - grasp`` on raw upstream values then
    rounds to 4dp; verify byte-exact via the same recipe.
    """
    for tool in EXPECTED_TOOLS:
        s_raw = upstream_medians[tool]["SRRIP"]
        g_raw = upstream_medians[tool]["GRASP"]
        expected = round(s_raw - g_raw, 4)
        assert per_tool[tool]["srrip_minus_grasp_pp_oct"] == expected, (
            f"{tool}: srrip_minus_grasp drift — expected {expected!r}, "
            f"got {per_tool[tool]['srrip_minus_grasp_pp_oct']!r}"
        )


def test_lru_minus_grasp_equals_raw_subtraction(per_tool, upstream_medians):
    for tool in EXPECTED_TOOLS:
        l_raw = upstream_medians[tool].get("LRU")
        g_raw = upstream_medians[tool]["GRASP"]
        if l_raw is None:
            assert per_tool[tool]["lru_minus_grasp_pp_oct"] is None
            continue
        expected = round(l_raw - g_raw, 4)
        assert per_tool[tool]["lru_minus_grasp_pp_oct"] == expected, (
            f"{tool}: lru_minus_grasp drift — expected {expected!r}, "
            f"got {per_tool[tool]['lru_minus_grasp_pp_oct']!r}"
        )


def test_all_tools_present(per_tool):
    for tool, e in per_tool.items():
        assert e["present"] is True, f"{tool}: upstream JSON missing"


# ----------------------------------------------------------------------
# Group C: ordering predicates
# ----------------------------------------------------------------------

def test_srrip_steeper_matches_raw_comparison(per_tool, upstream_medians):
    """Generator uses ``srrip <= grasp`` on raw upstream values."""
    for tool in EXPECTED_TOOLS:
        s_raw = upstream_medians[tool]["SRRIP"]
        g_raw = upstream_medians[tool]["GRASP"]
        expected = (s_raw <= g_raw)
        assert per_tool[tool]["srrip_steeper"] == expected, (
            f"{tool}: srrip_steeper drift — raw srrip={s_raw!r} grasp={g_raw!r} "
            f"expected {expected}, got {per_tool[tool]['srrip_steeper']}"
        )


def test_srrip_strictly_steeper_matches_gap_floor(per_tool, upstream_medians):
    """Generator uses ``srrip < grasp - gap_floor`` on raw upstream values."""
    for tool in EXPECTED_TOOLS:
        s_raw = upstream_medians[tool]["SRRIP"]
        g_raw = upstream_medians[tool]["GRASP"]
        expected = (s_raw < g_raw - GAP_FLOOR_PP_OCTAVE)
        assert per_tool[tool]["srrip_strictly_steeper"] == expected, (
            f"{tool}: srrip_strictly_steeper drift — raw srrip={s_raw!r} "
            f"grasp={g_raw!r} floor={GAP_FLOOR_PP_OCTAVE} "
            f"expected {expected}, got {per_tool[tool]['srrip_strictly_steeper']}"
        )


def test_n_strict_tools_matches_recomputation(meta, per_tool):
    expected = sum(
        1 for tool in meta["tools"]
        if per_tool[tool].get("srrip_strictly_steeper") is True
    )
    assert meta["n_strict_tools"] == expected, (
        f"n_strict_tools drift: recomputed={expected}, "
        f"artifact={meta['n_strict_tools']}"
    )


# ----------------------------------------------------------------------
# Group D: verdict predicates
# ----------------------------------------------------------------------

def test_verdict_checks_keys(meta):
    expected = {
        "all_tools_present_and_valid",
        "all_tools_srrip_le_grasp",
        "enough_tools_strictly_steeper",
    }
    assert set(meta["verdict_checks"].keys()) == expected


def test_verdict_check_all_tools_present_and_valid(meta, per_tool):
    expected = all(
        per_tool[t].get("present") and "error" not in per_tool[t]
        for t in meta["tools"]
    )
    assert meta["verdict_checks"]["all_tools_present_and_valid"] == expected


def test_verdict_check_all_tools_srrip_le_grasp(meta, per_tool):
    """Mirror of generator's all-tools accumulator:
    ``all_le = AND over tools of srrip_steeper``.
    """
    expected = all(
        per_tool[t].get("srrip_steeper") is True for t in meta["tools"]
    )
    assert meta["verdict_checks"]["all_tools_srrip_le_grasp"] == expected


def test_verdict_check_enough_tools_strictly_steeper(meta):
    expected = meta["n_strict_tools"] >= REQUIRED_STRICT_TOOLS
    assert meta["verdict_checks"]["enough_tools_strictly_steeper"] == expected


def test_verdict_is_and_of_all_checks(meta):
    expected = "PASS" if all(meta["verdict_checks"].values()) else "FAIL"
    assert meta["verdict"] == expected


def test_current_verdict_is_pass(meta):
    assert meta["verdict"] == "PASS", (
        f"cross_tool_slope_ordering regressed to {meta['verdict']!r}; "
        "an oracle-aware policy should not be MORE cache-hungry than "
        "a non-oracle baseline across all three tools."
    )


# ----------------------------------------------------------------------
# Group E: physical-direction sanity
# ----------------------------------------------------------------------

def test_every_tool_shows_srrip_at_least_as_steep_as_grasp(per_tool):
    for tool, e in per_tool.items():
        assert e["srrip_steeper"] is True, (
            f"{tool}: SRRIP {e['srrip_median']} pp/oct shallower than "
            f"GRASP {e['grasp_median']} pp/oct — oracle-aware policy "
            "is somehow MORE cache-hungry than non-oracle, contradicting "
            "the gate's claim."
        )


def test_at_least_two_tools_strictly_steeper_with_real_gap(per_tool):
    """The gate's load-bearing test is exactly this: the strict count
    must clear ``REQUIRED_STRICT_TOOLS`` with a gap of at least
    ``GAP_FLOOR_PP_OCTAVE`` pp/oct.
    """
    strict = [
        t for t, e in per_tool.items()
        if e.get("srrip_strictly_steeper") is True
    ]
    assert len(strict) >= REQUIRED_STRICT_TOOLS, (
        f"only {len(strict)} tools show strictly steeper SRRIP "
        f"(needed {REQUIRED_STRICT_TOOLS} with >= {GAP_FLOOR_PP_OCTAVE} "
        f"pp/oct gap): strict tools = {strict}"
    )
