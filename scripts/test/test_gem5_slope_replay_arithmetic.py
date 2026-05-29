"""Gate 120 — gem5_slope_replay.json OLS arithmetic + verdict.

Locks the gem5 anchor slope replay that converts gem5_anchor.json
(four L3 sizes: 4kB, 32kB, 256kB, 2MB) into per-(app, graph, policy)
OLS slopes of miss% vs log2(L3_kB) and aggregates them into per-policy
medians + cross-policy comparisons used by the cross-tool roll-ups.

This gate is the load-bearing arithmetic gate for the gem5 column of
gate 119 (cross-tool slope universality) and for several anchor-cross-
tool checks. A bug in the OLS slope computation here would silently
poison every downstream cross-tool invariant.

Invariants (12 tests, 4 groups):
- meta structure (3): anchor_source='gem5_anchor.json', the L3 axis
  log2(kB) table, EXPECTED_SIZES, POLICIES, help_floor_pp_octave=-1.0;
  n_cells and n_cell_policy_records derived from per_cell.
- per_cell OLS arithmetic (3): slope_pp_per_octave is the OLS slope of
  miss_pp(size) vs log2(kB(size)) for the four expected sizes;
  miss_pp_by_size cross-link to gem5_anchor.cells (with %→pp ×100
  conversion); every per_cell record has all four sizes populated.
- per_policy aggregation (3): median, mean, n recomputed from
  per_cell slopes grouped by policy; lru_minus_grasp_pp_oct and
  srrip_minus_grasp_pp_oct are exactly per_policy[LRU/SRRIP].median
  − per_policy[GRASP].median.
- monotonic + verdict (3): monotonic_violations contains every cell
  where miss(smallest) ≤ miss(largest); verdict_checks recomputed
  from the four invariants (cache monotonic, all medians negative,
  SRRIP ≥ GRASP steepness, GRASP below help floor); verdict='PASS'
  iff all checks pass.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import pytest

ARTIFACT = Path("wiki/data/gem5_slope_replay.json")
SOURCE = Path("wiki/data/gem5_anchor.json")

EXPECTED_SIZES = ("4kB", "32kB", "256kB", "2MB")
L3_LOG2_KB = {"4kB": 2.0, "32kB": 5.0, "256kB": 8.0, "2MB": 11.0}
POLICIES = ("GRASP", "LRU", "SRRIP")
HELP_FLOOR = -1.0

SLOPE_TOL = 1e-3
MISS_TOL = 1e-3
DIFF_TOL = 1e-3


@pytest.fixture(scope="module")
def data():
    assert ARTIFACT.exists(), f"missing artifact: {ARTIFACT}"
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def source():
    assert SOURCE.exists(), f"missing source artifact: {SOURCE}"
    return json.loads(SOURCE.read_text())


def _ols_slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den


# ── group 1: meta structure ──────────────────────────────────────────────


def test_meta_constants(data):
    meta = data["meta"]
    assert meta["anchor_source"] == "gem5_anchor.json"
    assert meta["l3_axis_log2_kb"] == L3_LOG2_KB
    assert meta["expected_sizes"] == list(EXPECTED_SIZES)
    assert meta["policies"] == list(POLICIES)
    assert meta["help_floor_pp_octave"] == HELP_FLOOR


def test_meta_n_cells_matches_per_cell(data):
    cells = {(r["app"], r["graph"]) for r in data["per_cell"]}
    assert data["meta"]["n_cells"] == len(cells)


def test_meta_n_cell_policy_records(data):
    assert data["meta"]["n_cell_policy_records"] == len(data["per_cell"])


# ── group 2: per_cell OLS + cross-link ───────────────────────────────────


def test_per_cell_all_sizes_present(data):
    for r in data["per_cell"]:
        sizes = set(r["miss_pp_by_size"].keys())
        assert sizes == set(EXPECTED_SIZES), (
            f"{r['app']}/{r['graph']}/{r['policy']}: sizes={sorted(sizes)}"
        )


def test_per_cell_ols_slope_correct(data):
    for r in data["per_cell"]:
        xs = [L3_LOG2_KB[s] for s in EXPECTED_SIZES]
        ys = [r["miss_pp_by_size"][s] for s in EXPECTED_SIZES]
        expected = _ols_slope(xs, ys)
        assert math.isclose(r["slope_pp_per_octave"], expected, abs_tol=SLOPE_TOL), (
            f"{r['app']}/{r['graph']}/{r['policy']}: "
            f"slope={r['slope_pp_per_octave']} expected={expected:.4f}"
        )


def test_per_cell_miss_pp_cross_links_anchor_source(data, source):
    by_cell: dict = {}
    for c in source["cells"]:
        key = (c["app"], c["graph"], c["l3_size"])
        by_cell[key] = c["miss_rate_by_policy"]
    for r in data["per_cell"]:
        for size, pp in r["miss_pp_by_size"].items():
            mb = by_cell[(r["app"], r["graph"], size)]
            expected_pp = float(mb[r["policy"]]) * 100.0
            assert math.isclose(pp, expected_pp, abs_tol=MISS_TOL), (
                f"{r['app']}/{r['graph']}/{r['policy']}/{size}: "
                f"pp={pp} expected={expected_pp:.4f}"
            )


# ── group 3: per_policy aggregation + cross-policy deltas ────────────────


def test_per_policy_median_mean_count(data):
    slopes_by_pol: dict[str, list[float]] = {p: [] for p in POLICIES}
    for r in data["per_cell"]:
        slopes_by_pol[r["policy"]].append(r["slope_pp_per_octave"])
    for pol in POLICIES:
        entry = data["meta"]["per_policy"][pol]
        vals = slopes_by_pol[pol]
        assert entry["n"] == len(vals), f"{pol}: n mismatch"
        if vals:
            assert math.isclose(entry["median"], statistics.median(vals), abs_tol=SLOPE_TOL), (
                f"{pol}: median mismatch"
            )
            assert math.isclose(entry["mean"], sum(vals) / len(vals), abs_tol=SLOPE_TOL), (
                f"{pol}: mean mismatch"
            )


def test_lru_minus_grasp_pp_oct(data):
    pp = data["meta"]["per_policy"]
    expected = pp["LRU"]["median"] - pp["GRASP"]["median"]
    assert math.isclose(
        data["meta"]["lru_minus_grasp_pp_oct"], expected, abs_tol=DIFF_TOL
    )


def test_srrip_minus_grasp_pp_oct(data):
    pp = data["meta"]["per_policy"]
    expected = pp["SRRIP"]["median"] - pp["GRASP"]["median"]
    assert math.isclose(
        data["meta"]["srrip_minus_grasp_pp_oct"], expected, abs_tol=DIFF_TOL
    )


# ── group 4: monotonic violations + verdict ──────────────────────────────


def test_monotonic_violations_reproduce(data):
    expected = []
    for r in data["per_cell"]:
        miss_small = r["miss_pp_by_size"][EXPECTED_SIZES[0]]
        miss_large = r["miss_pp_by_size"][EXPECTED_SIZES[-1]]
        if miss_small <= miss_large:
            expected.append((r["app"], r["graph"], r["policy"]))
    artifact_keys = [
        (v["app"], v["graph"], v["policy"]) for v in data["meta"]["monotonic_violations"]
    ]
    assert artifact_keys == expected, (
        f"violations artifact={artifact_keys} expected={expected}"
    )


def test_verdict_checks_reproduce(data):
    meta = data["meta"]
    pp = meta["per_policy"]
    expected = {
        "cache_monotonic_every_cell": len(meta["monotonic_violations"]) == 0,
        "all_per_policy_medians_negative": all(
            pp[p]["median"] is not None and pp[p]["median"] < 0 for p in POLICIES
        ),
        "srrip_at_least_as_steep_as_grasp": (
            pp["SRRIP"]["median"] is not None
            and pp["GRASP"]["median"] is not None
            and pp["SRRIP"]["median"] <= pp["GRASP"]["median"]
        ),
        "grasp_below_help_floor": (
            pp["GRASP"]["median"] is not None and pp["GRASP"]["median"] < HELP_FLOOR
        ),
    }
    for key, want in expected.items():
        assert meta["verdict_checks"][key] is want, (
            f"verdict_checks[{key}]={meta['verdict_checks'][key]} expected={want}"
        )


def test_verdict_pass_iff_all_checks_pass(data):
    expected = "PASS" if all(data["meta"]["verdict_checks"].values()) else "FAIL"
    assert data["meta"]["verdict"] == expected
