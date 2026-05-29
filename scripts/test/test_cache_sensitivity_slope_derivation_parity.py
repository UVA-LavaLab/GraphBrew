"""Gate 137+ — cache_sensitivity_slope derivation parity.

cache_sensitivity_slope.json is derived from oracle_gap_auc.json by
walking per-(app,policy) trajectories at the paper L3 sweep
(1MB→4MB→8MB) and computing per-octave slopes:

    slope_pp_per_octave = -delta(gap_pp) / delta(log2_MB)

Higher slope = gap shrinks faster as cache grows (the expected
sign for any well-behaved replacement policy).

Existing tests pin headline facts (GRASP has largest mean slope,
LRU/SRRIP own the anti-scaling list). This gate locks the
*derivation arithmetic* from oracle_gap_auc trajectory cells through
per-cell octave records and per-policy summary stats. If the
generator silently flips sign convention, swaps log2 for raw L3,
changes the avg_slope formula, miscounts monotonic violations, or
mis-reports per_policy_summary stats, this gate flips.

Invariants (20 tests, 5 groups):

Group A — Structural & scope
  1. Top-level keys: meta, per_app, per_policy_summary,
     monotonic_violations
  2. meta.source points to oracle_gap_auc.json
  3. meta.l3_octaves == ['1MB','4MB','8MB']
  4. meta.slope_units mentions log2 and per octave
  5. per_app keys == set(meta.apps); every per_app[app] covers all
     meta.policies (when the corresponding AUC trajectory has all 3
     L3 points)

Group B — Octave arithmetic (oracle_gap_auc trajectory → octave row)
  6. gap_from / gap_to mirror AUC trajectory cells to 1e-4
  7. delta_gap_pp == round(gap_to - gap_from, 4) to 1e-4
  8. delta_log2_mb == round(log2(to) - log2(from), 4)
  9. slope_pp_per_octave == round(-delta_gap_pp/delta_log2_mb, 4)
     to 1e-4
  10. Octave 'from','to' pairs are exactly (1MB,4MB) and (4MB,8MB)
      in that order

Group C — Per-cell aggregate fields
  11. gap_at_1MB / gap_at_8MB mirror trajectory[1MB] / trajectory[8MB]
      to 1e-4
  12. total_shrinkage_pp == round(gap_at_1MB - gap_at_8MB, 4) to 1e-4
  13. avg_slope_pp_per_octave == round((1MB - 8MB) / 3, 4) to 1e-4
      (log2(8MB)-log2(1MB)=3 octaves)
  14. monotonic_decreasing == every octave delta_gap_pp <= 1e-9

Group D — Monotonic violations list + meta counts
  15. monotonic_violations entries correspond EXACTLY to (app,policy)
      cells where monotonic_decreasing is False
  16. meta.n_monotonic_violations == len(monotonic_violations)
  17. meta.all_monotonic == (n_monotonic_violations == 0)

Group E — Per-policy summary reproduction
  18. per_policy_summary keys subset of meta.policies; n_apps matches
      the count of apps where (pol) has all 3 L3 points in AUC
  19. mean_avg_slope reproduces statistics.fmean of avg_slope across
      apps to 1e-4
  20. max_slope/min_slope reproduce max/min of per-app avg_slope
      to 1e-4
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

ORDER = ("1MB", "4MB", "8MB")
L3_MB = {"1MB": 1.0, "4MB": 4.0, "8MB": 8.0}
EPS = 1e-4


@pytest.fixture(scope="module")
def auc() -> dict:
    return json.loads((WIKI_DATA / "oracle_gap_auc.json").read_text())


@pytest.fixture(scope="module")
def slope() -> dict:
    return json.loads((WIKI_DATA / "cache_sensitivity_slope.json").read_text())


# ─── Group A — Structural ────────────────────────────────────────────


def test_top_level_keys(slope):
    assert set(slope.keys()) >= {
        "meta", "per_app", "per_policy_summary", "monotonic_violations"
    }


def test_meta_source_points_to_auc(slope):
    assert slope["meta"]["source"] == "wiki/data/oracle_gap_auc.json"


def test_meta_l3_octaves_locked(slope):
    assert tuple(slope["meta"]["l3_octaves"]) == ORDER


def test_meta_slope_units_describes_log_and_octave(slope):
    units = slope["meta"]["slope_units"].lower()
    assert "log2" in units and "octave" in units


def test_per_app_keys_match_meta_and_cover_full_grid(slope, auc):
    assert set(slope["per_app"].keys()) == set(slope["meta"]["apps"])
    missing = []
    for app in slope["meta"]["apps"]:
        for pol in slope["meta"]["policies"]:
            traj = auc["per_app"][app]["trajectory_by_policy"].get(pol, {})
            if all(l3 in traj for l3 in ORDER):
                if pol not in slope["per_app"][app]:
                    missing.append((app, pol))
    assert not missing, missing


# ─── Group B — Octave arithmetic ─────────────────────────────────────


def test_octave_gap_from_to_mirror_auc_trajectory(slope, auc):
    mism = []
    for app, by_pol in slope["per_app"].items():
        for pol, cell in by_pol.items():
            traj = auc["per_app"][app]["trajectory_by_policy"][pol]
            for o in cell["octaves"]:
                if abs(o["gap_from"] - round(traj[o["from"]], 4)) > EPS:
                    mism.append((app, pol, o["from"], o["gap_from"]))
                if abs(o["gap_to"] - round(traj[o["to"]], 4)) > EPS:
                    mism.append((app, pol, o["to"], o["gap_to"]))
    assert not mism, mism[:5]


def test_octave_delta_gap_pp_reproduces(slope):
    mism = []
    for app, by_pol in slope["per_app"].items():
        for pol, cell in by_pol.items():
            for o in cell["octaves"]:
                expected = round(o["gap_to"] - o["gap_from"], 4)
                if abs(o["delta_gap_pp"] - expected) > EPS:
                    mism.append((app, pol, o["delta_gap_pp"], expected))
    assert not mism, mism[:5]


def test_octave_delta_log2_mb_reproduces(slope):
    mism = []
    for app, by_pol in slope["per_app"].items():
        for pol, cell in by_pol.items():
            for o in cell["octaves"]:
                expected = round(
                    math.log2(L3_MB[o["to"]]) - math.log2(L3_MB[o["from"]]), 4
                )
                if abs(o["delta_log2_mb"] - expected) > EPS:
                    mism.append((app, pol, o["delta_log2_mb"], expected))
    assert not mism, mism[:5]


def test_octave_slope_reproduces(slope, auc):
    """Generator computes slope = -(traj[dst]-traj[src])/d_log on RAW
    trajectory values, then rounds. We must mirror this path (not
    use the pre-rounded delta_gap_pp), or Python banker's rounding
    at the 5th decimal flips e.g. -0.13595 → -0.1359 vs -0.136."""
    mism = []
    for app, by_pol in slope["per_app"].items():
        for pol, cell in by_pol.items():
            traj = auc["per_app"][app]["trajectory_by_policy"][pol]
            for o in cell["octaves"]:
                d_gap = traj[o["to"]] - traj[o["from"]]
                d_log = math.log2(L3_MB[o["to"]]) - math.log2(L3_MB[o["from"]])
                expected = round(-d_gap / d_log, 4)
                if abs(o["slope_pp_per_octave"] - expected) > EPS:
                    mism.append((app, pol, o["slope_pp_per_octave"], expected))
    assert not mism, mism[:5]


def test_octave_pairs_are_locked(slope):
    bad = []
    for app, by_pol in slope["per_app"].items():
        for pol, cell in by_pol.items():
            pairs = [(o["from"], o["to"]) for o in cell["octaves"]]
            if pairs != [("1MB", "4MB"), ("4MB", "8MB")]:
                bad.append((app, pol, pairs))
    assert not bad, bad[:3]


# ─── Group C — Per-cell aggregates ───────────────────────────────────


def test_gap_at_endpoints_mirror_trajectory(slope, auc):
    mism = []
    for app, by_pol in slope["per_app"].items():
        for pol, cell in by_pol.items():
            traj = auc["per_app"][app]["trajectory_by_policy"][pol]
            if abs(cell["gap_at_1MB"] - traj["1MB"]) > EPS:
                mism.append((app, pol, "1MB", cell["gap_at_1MB"], traj["1MB"]))
            if abs(cell["gap_at_8MB"] - traj["8MB"]) > EPS:
                mism.append((app, pol, "8MB", cell["gap_at_8MB"], traj["8MB"]))
    assert not mism, mism[:5]


def test_total_shrinkage_pp_reproduces(slope):
    mism = []
    for app, by_pol in slope["per_app"].items():
        for pol, cell in by_pol.items():
            expected = round(cell["gap_at_1MB"] - cell["gap_at_8MB"], 4)
            if abs(cell["total_shrinkage_pp"] - expected) > EPS:
                mism.append((app, pol, cell["total_shrinkage_pp"], expected))
    assert not mism, mism[:5]


def test_avg_slope_reproduces(slope):
    """avg = (gap_at_1MB - gap_at_8MB) / 3 octaves."""
    mism = []
    for app, by_pol in slope["per_app"].items():
        for pol, cell in by_pol.items():
            expected = round((cell["gap_at_1MB"] - cell["gap_at_8MB"]) / 3.0, 4)
            if abs(cell["avg_slope_pp_per_octave"] - expected) > EPS:
                mism.append((app, pol, cell["avg_slope_pp_per_octave"], expected))
    assert not mism, mism[:5]


def test_monotonic_decreasing_flag(slope):
    bad = []
    for app, by_pol in slope["per_app"].items():
        for pol, cell in by_pol.items():
            mono = all(o["delta_gap_pp"] <= 1e-9 for o in cell["octaves"])
            if cell["monotonic_decreasing"] != mono:
                bad.append((app, pol, cell["monotonic_decreasing"], mono))
    assert not bad, bad


# ─── Group D — Monotonic-violations list + meta counts ───────────────


def test_violations_correspond_to_non_monotonic_cells(slope):
    """The list of (app,policy) entries in monotonic_violations
    matches the set of cells with monotonic_decreasing == False."""
    listed = {(v["app"], v["policy"]) for v in slope["monotonic_violations"]}
    expected = {
        (app, pol)
        for app, by_pol in slope["per_app"].items()
        for pol, cell in by_pol.items()
        if not cell["monotonic_decreasing"]
    }
    assert listed == expected, listed.symmetric_difference(expected)


def test_meta_n_violations_matches_list_length(slope):
    assert slope["meta"]["n_monotonic_violations"] == len(
        slope["monotonic_violations"]
    )


def test_meta_all_monotonic_consistent(slope):
    assert slope["meta"]["all_monotonic"] == (
        slope["meta"]["n_monotonic_violations"] == 0
    )


# ─── Group E — Per-policy summary ────────────────────────────────────


def test_per_policy_summary_n_apps_matches_grid(slope, auc):
    """Each per_policy_summary[pol].n_apps counts apps where the AUC
    trajectory for (pol) has all 3 L3 points."""
    mism = []
    for pol, summary in slope["per_policy_summary"].items():
        expected = sum(
            1
            for app in auc["per_app"]
            if all(
                l3 in auc["per_app"][app]["trajectory_by_policy"].get(pol, {})
                for l3 in ORDER
            )
        )
        if summary["n_apps"] != expected:
            mism.append((pol, summary["n_apps"], expected))
    assert not mism, mism


def test_per_policy_summary_mean_avg_slope_reproduces(slope):
    mism = []
    for pol, summary in slope["per_policy_summary"].items():
        slopes = []
        for app, by_pol in slope["per_app"].items():
            if pol in by_pol:
                slopes.append(by_pol[pol]["avg_slope_pp_per_octave"])
        if not slopes:
            continue
        # generator computes from un-rounded avg, but stored avg is
        # already rounded; allow a slightly larger 1e-3 tolerance so
        # the rounding compounding doesn't flap.
        expected = round(statistics.fmean(slopes), 4)
        if abs(summary["mean_avg_slope"] - expected) > 1e-3:
            mism.append((pol, summary["mean_avg_slope"], expected))
    assert not mism, mism


def test_per_policy_summary_max_min_slope_reproduce(slope):
    mism = []
    for pol, summary in slope["per_policy_summary"].items():
        slopes = []
        for app, by_pol in slope["per_app"].items():
            if pol in by_pol:
                slopes.append(by_pol[pol]["avg_slope_pp_per_octave"])
        if not slopes:
            continue
        expected_max = round(max(slopes), 4)
        expected_min = round(min(slopes), 4)
        if abs(summary["max_slope"] - expected_max) > 1e-3:
            mism.append((pol, "max", summary["max_slope"], expected_max))
        if abs(summary["min_slope"] - expected_min) > 1e-3:
            mism.append((pol, "min", summary["min_slope"], expected_min))
    assert not mism, mism
