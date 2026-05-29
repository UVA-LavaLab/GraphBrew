"""Gate 90 — gem5/sniper anchor cell parity cross-artifact gate.

Locks the cell-set parity between the two timing-simulator anchor
artifacts (``wiki/data/gem5_anchor.json`` and ``wiki/data/
sniper_anchor.json``) and the anchor-census roll-up artifact
(``wiki/data/anchor_cell_census.json``).

The shared anchor cell ``(email-Eu-core, pr)`` is the single load-
bearing cross-tool comparison point that every later cross-tool gate
(slope-ordering, lru-regime, cross-tool-saturation, cross-tool-winners)
ultimately resolves to. If gem5 or Sniper silently drops it, or if the
L3 axis or policy set diverges between the two tools, the cross-tool
agreement claim cannot be verified at all.

This gate makes that class of regression impossible by locking:

  * the per-tool anchor scope (graphs, apps),
  * per-tool cell-record counts (cells × L3-axis × policies match the
    expected matrix size),
  * the structural invariants the census says hold (verdict PASS,
    every verdict_check True),
  * the existence of the shared (graph, app) anchor cell,
  * full population of every cell's miss_rate_by_policy across all
    three policies, with values in (0, 1),
  * sanity counters (no missing or disagreeing invariant rows).
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI = REPO_ROOT / "wiki" / "data"

GEM5_JSON = WIKI / "gem5_anchor.json"
SNIPER_JSON = WIKI / "sniper_anchor.json"
CENSUS_JSON = WIKI / "anchor_cell_census.json"

EXPECTED_POLICIES = {"GRASP", "LRU", "SRRIP"}
EXPECTED_L3_AXIS = ("4kB", "32kB", "256kB", "2MB")
EXPECTED_SHARED_CELL = ("email-Eu-core", "pr")

# gem5 anchors the small-graph regime on email-Eu-core only, two apps.
EXPECTED_GEM5_GRAPHS = {"email-Eu-core"}
EXPECTED_GEM5_APPS = {"bc", "pr"}

# Sniper anchors a wider scope (two graphs × at least three apps).
EXPECTED_SNIPER_GRAPHS = {"cit-Patents", "email-Eu-core"}
EXPECTED_SNIPER_APPS_FLOOR = {"pr"}  # must at least cover the shared cell

# Cell counts: per-tool (graph, app) cell count × L3-axis length.
GEM5_CELL_COUNT_FLOOR = 2 * len(EXPECTED_L3_AXIS)         # 8
SNIPER_CELL_COUNT_FLOOR = 6 * len(EXPECTED_L3_AXIS)       # 24


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _gem5() -> dict:
    return _load(GEM5_JSON)


def _sniper() -> dict:
    return _load(SNIPER_JSON)


def _census() -> dict:
    return _load(CENSUS_JSON)


# ---------------------------------------------------------------------------
# Per-tool scope locks
# ---------------------------------------------------------------------------

def test_gem5_anchor_scope_locked():
    g = _gem5()
    assert set(g["graphs_scope"]) == EXPECTED_GEM5_GRAPHS, (
        f"gem5 graphs_scope drifted: got {sorted(g['graphs_scope'])}, "
        f"expected {sorted(EXPECTED_GEM5_GRAPHS)}"
    )
    assert set(g["apps_scope"]) == EXPECTED_GEM5_APPS, (
        f"gem5 apps_scope drifted: got {sorted(g['apps_scope'])}, "
        f"expected {sorted(EXPECTED_GEM5_APPS)}"
    )


def test_sniper_anchor_scope_locked():
    s = _sniper()
    assert set(s["graphs_scope"]) == EXPECTED_SNIPER_GRAPHS, (
        f"sniper graphs_scope drifted: got {sorted(s['graphs_scope'])}, "
        f"expected {sorted(EXPECTED_SNIPER_GRAPHS)}"
    )
    assert EXPECTED_SNIPER_APPS_FLOOR.issubset(set(s["apps_scope"])), (
        f"sniper apps_scope dropped a required app: "
        f"missing={EXPECTED_SNIPER_APPS_FLOOR - set(s['apps_scope'])}"
    )


# ---------------------------------------------------------------------------
# Per-tool cell counts and invariant-status sanity
# ---------------------------------------------------------------------------

def test_gem5_cell_count_meets_baseline():
    g = _gem5()
    n = g["counts"]["cells"]
    assert n >= GEM5_CELL_COUNT_FLOOR, (
        f"gem5 cell count {n} below floor {GEM5_CELL_COUNT_FLOOR} "
        f"(scope = {EXPECTED_GEM5_GRAPHS} × {EXPECTED_GEM5_APPS} × "
        f"{EXPECTED_L3_AXIS})"
    )


def test_sniper_cell_count_meets_baseline():
    s = _sniper()
    n = s["counts"]["cells"]
    assert n >= SNIPER_CELL_COUNT_FLOOR, (
        f"sniper cell count {n} below floor {SNIPER_CELL_COUNT_FLOOR}"
    )


def test_gem5_invariants_no_disagree_no_missing():
    c = _gem5()["counts"]
    assert c["invariants_disagree"] == 0, (
        f"gem5 anchor invariants_disagree regressed to "
        f"{c['invariants_disagree']}"
    )
    assert c["invariants_missing"] == 0, (
        f"gem5 anchor invariants_missing regressed to "
        f"{c['invariants_missing']}"
    )
    assert c["invariants_ok"] >= 1, (
        f"gem5 anchor invariants_ok dropped to {c['invariants_ok']}"
    )


def test_sniper_invariants_no_disagree_no_missing():
    c = _sniper()["counts"]
    assert c["invariants_disagree"] == 0, (
        f"sniper anchor invariants_disagree regressed to "
        f"{c['invariants_disagree']}"
    )
    assert c["invariants_missing"] == 0, (
        f"sniper anchor invariants_missing regressed to "
        f"{c['invariants_missing']}"
    )
    assert c["invariants_ok"] >= 1, (
        f"sniper anchor invariants_ok dropped to {c['invariants_ok']}"
    )


# ---------------------------------------------------------------------------
# Cell-content sanity: every cell carries all 3 policies in (0, 1)
# ---------------------------------------------------------------------------

def _cells_have_all_policies_and_valid_miss_rates(cells: list, tool: str):
    bad = []
    for c in cells:
        mrp = c.get("miss_rate_by_policy", {})
        if set(mrp.keys()) != EXPECTED_POLICIES:
            bad.append(
                (tool, c.get("graph"), c.get("app"), c.get("l3_size"),
                 "policy set", sorted(mrp.keys()))
            )
            continue
        for pol, mr in mrp.items():
            if not (0.0 < mr < 1.0):
                bad.append(
                    (tool, c.get("graph"), c.get("app"), c.get("l3_size"),
                     pol, mr)
                )
    return bad


def test_every_gem5_cell_has_all_three_policies_with_valid_miss_rates():
    bad = _cells_have_all_policies_and_valid_miss_rates(_gem5()["cells"], "gem5")
    assert not bad, f"gem5 cells failed policy/miss-rate sanity: {bad[:5]}"


def test_every_sniper_cell_has_all_three_policies_with_valid_miss_rates():
    bad = _cells_have_all_policies_and_valid_miss_rates(
        _sniper()["cells"], "sniper"
    )
    assert not bad, f"sniper cells failed policy/miss-rate sanity: {bad[:5]}"


# ---------------------------------------------------------------------------
# Census roll-up: cross-tool agreement
# ---------------------------------------------------------------------------

def test_anchor_cell_census_verdict_pass():
    m = _census()["meta"]
    assert m["verdict"] == "PASS", (
        f"anchor_cell_census verdict regressed: {m['verdict']}"
    )


def test_anchor_cell_census_every_verdict_check_holds():
    checks = _census()["meta"]["verdict_checks"]
    failed = [k for k, v in checks.items() if not v]
    assert not failed, f"anchor_cell_census verdict_checks regressed: {failed}"
    assert len(checks) >= 13, (
        f"verdict_checks dropped below the 13-check floor: have {len(checks)}"
    )


def test_anchor_cell_census_shared_l3_axis_and_policies():
    m = _census()["meta"]
    assert m["shared_l3_axis"] is True, (
        "gem5 and sniper no longer share their L3 axis — cross-tool "
        "per-policy comparison is meaningless until this is restored."
    )
    assert m["shared_policies"] is True, (
        "gem5 and sniper no longer share their policy set — cross-tool "
        "per-policy comparison is meaningless until this is restored."
    )


def test_anchor_cell_census_shares_the_load_bearing_anchor_cell():
    """The (email-Eu-core, pr) anchor is the cross-tool reduction point
    for the slope-ordering, lru-regime, cross-tool-saturation, and
    cross-tool-winners gates. If it ever drops out of the shared set,
    universality claims have to be rebuilt from scratch."""
    m = _census()["meta"]
    shared = {tuple(c) for c in m["shared_cells"]}
    assert EXPECTED_SHARED_CELL in shared, (
        f"load-bearing shared anchor {EXPECTED_SHARED_CELL} missing from "
        f"shared_cells: {sorted(shared)}"
    )
    assert m["shared_cell_count"] >= 1, (
        f"shared_cell_count collapsed to {m['shared_cell_count']}"
    )


# ---------------------------------------------------------------------------
# Cross-artifact axis parity
# ---------------------------------------------------------------------------

def test_expected_l3_axis_subset_of_both_tool_axes():
    """The four canonical L3 sizes must appear in both tools' anchor
    axes; if either tool drops a size, the slope-replay artifacts can
    no longer evaluate per-octave."""
    g_axis = set(_census()["meta"]["gem5"]["l3_axis"])
    s_axis = set(_census()["meta"]["sniper"]["l3_axis"])
    missing_g = set(EXPECTED_L3_AXIS) - g_axis
    missing_s = set(EXPECTED_L3_AXIS) - s_axis
    assert not missing_g, f"gem5 anchor lost L3 sizes: {missing_g}"
    assert not missing_s, f"sniper anchor lost L3 sizes: {missing_s}"


def test_expected_policies_subset_of_both_tool_policy_sets():
    g_pol = set(_census()["meta"]["gem5"]["policies"])
    s_pol = set(_census()["meta"]["sniper"]["policies"])
    missing_g = EXPECTED_POLICIES - g_pol
    missing_s = EXPECTED_POLICIES - s_pol
    assert not missing_g, f"gem5 anchor lost policies: {missing_g}"
    assert not missing_s, f"sniper anchor lost policies: {missing_s}"
