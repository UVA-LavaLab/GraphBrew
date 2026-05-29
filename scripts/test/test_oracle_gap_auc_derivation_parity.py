"""Gate 136+ — oracle_gap_auc derivation parity (cross-artifact).

The AUC artifact (`wiki/data/oracle_gap_auc.json`) is *derived* from
`oracle_gap.json` by:

  1. filter rows to scope_l3_sizes (paper L3 only: 1MB / 4MB / 8MB)
  2. group gap_pp by (app, policy, l3)
  3. per (app, policy) build trajectory: l3 -> mean(gap_pp)
  4. trapezoidal AUC on (log2(L3 in MB), mean_gap_pp)
  5. summarize: ranking, winner (min auc), runner_up, ratios, savings

Until now, only the *facts* of the AUC artifact had gates (winner
identity, structural shape — see `test_oracle_gap_auc.py`). This gate
locks the derivation math itself: if the generator math drifts (e.g.
swaps trapezoidal-on-log2 for raw-x, switches mean → median, includes
non-paper L3, rounds at a different decimal), this gate flips before
any downstream paper claim quietly moves.

Invariants (20 tests, 5 groups):

Group A — Structural & scope
  1. Top-level keys: meta, per_app
  2. meta.source == 'wiki/data/oracle_gap.json'
  3. meta.scope_l3_sizes == ['1MB','4MB','8MB']
  4. meta.x_axis describes log2(L3) and y_axis describes mean gap_pp
  5. per_app keys == set(meta.apps) == set(apps observed in oracle_gap)

Group B — Trajectory parity (oracle_gap → trajectory_by_policy)
  6. For every (app, policy, l3) cell in scope: trajectory[l3] ==
     mean(gap_pp) for that cell, rounded to 4 decimals (1e-4 tol).
  7. Trajectory only contains entries where >= 1 source row exists.
  8. Every reported trajectory has >= 2 L3 points (generator skips
     policies with < 2 points before computing AUC).
  9. Every L3 key in any trajectory is in scope_l3_sizes.

Group C — AUC derivation (trapezoidal on log2(MB))
  10. For every (app, policy) with a trajectory, auc_by_policy
      reproduces trapezoidal_log_auc to 1e-4.
  11. AUC values are non-negative (gap_pp >= 0 ⇒ AUC >= 0).
  12. AUC values are finite (no NaN / inf).

Group D — Ranking + winner + runner-up + ratios
  13. ranking is sorted ascending by auc (with policy as tie-break),
      matching generator ordering.
  14. winner == ranking[0].policy; winner_auc == ranking[0].auc.
  15. runner_up == ranking[1].policy; runner_up_auc == ranking[1].auc.
  16. auc_ratio_winner_over_runner_up reproduces round(winner/runner, 4)
      when runner_auc > 0.
  17. auc_ratio_winner_over_lru reproduces round(winner/lru, 4) when
      lru_auc > 0.
  18. auc_pp_savings_winner_vs_lru reproduces round(lru - winner, 4).

Group E — Cross-summary (meta.auc_winner_by_app vs per_app.winner)
  19. meta.auc_winner_by_app[app] == per_app[app].winner for every app.
  20. LRU is NEVER the winner in any app (lit-faithful behavior;
      already in old test, re-asserted here to keep the gate
      self-contained as a regression locker).
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

PAPER_L3_SIZES = ("1MB", "4MB", "8MB")
L3_MB = {"1MB": 1.0, "4MB": 4.0, "8MB": 8.0}

EPS = 1e-4


@pytest.fixture(scope="module")
def og() -> dict:
    return json.loads((WIKI_DATA / "oracle_gap.json").read_text())


@pytest.fixture(scope="module")
def auc() -> dict:
    return json.loads((WIKI_DATA / "oracle_gap_auc.json").read_text())


@pytest.fixture(scope="module")
def grouped_means(og) -> dict:
    """(app, policy, l3) -> mean(gap_pp), restricted to PAPER_L3_SIZES."""
    g: dict[tuple, list[float]] = defaultdict(list)
    for r in og["rows"]:
        if r["l3_size"] in PAPER_L3_SIZES:
            g[(r["app"], r["policy"], r["l3_size"])].append(
                float(r["gap_pp"])
            )
    return {k: statistics.mean(v) for k, v in g.items()}


def _trap_log_auc(points: dict[float, float]) -> float:
    pts = sorted(points.items())
    a = 0.0
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        dx = math.log2(x1) - math.log2(x0)
        a += 0.5 * (y0 + y1) * dx
    return a


# ─── Group A — Structural ────────────────────────────────────────────


def test_top_level_keys(auc):
    assert set(auc.keys()) >= {"meta", "per_app"}


def test_meta_source_points_to_oracle_gap(auc):
    assert auc["meta"]["source"] == "wiki/data/oracle_gap.json"


def test_meta_scope_is_paper_l3(auc):
    assert tuple(auc["meta"]["scope_l3_sizes"]) == PAPER_L3_SIZES


def test_meta_axis_labels_describe_log_and_mean(auc):
    assert "log2" in auc["meta"]["x_axis"].lower()
    assert "mean" in auc["meta"]["y_axis"].lower() or \
           "gap_pp" in auc["meta"]["y_axis"].lower()


def test_per_app_keys_match_meta_and_oracle_gap(auc, og):
    apps_oracle = {r["app"] for r in og["rows"] if r["l3_size"] in PAPER_L3_SIZES}
    assert set(auc["per_app"].keys()) == apps_oracle
    assert set(auc["meta"]["apps"]) == apps_oracle


# ─── Group B — Trajectory parity ─────────────────────────────────────


def test_trajectory_reproduces_mean_gap_pp(auc, grouped_means):
    mism = []
    for app, app_obj in auc["per_app"].items():
        for pol, traj in app_obj["trajectory_by_policy"].items():
            for l3, val in traj.items():
                key = (app, pol, l3)
                if key not in grouped_means:
                    mism.append(("missing-source", key, val))
                    continue
                expected = round(grouped_means[key], 4)
                if abs(val - expected) > EPS:
                    mism.append((key, val, expected))
    assert not mism, mism[:5]


def test_trajectory_only_has_observed_cells(auc, grouped_means):
    """No invented points: every trajectory entry must correspond to a
    real source cell."""
    extras = []
    for app, app_obj in auc["per_app"].items():
        for pol, traj in app_obj["trajectory_by_policy"].items():
            for l3 in traj.keys():
                if (app, pol, l3) not in grouped_means:
                    extras.append((app, pol, l3))
    assert not extras, extras[:5]


def test_trajectory_has_at_least_two_points(auc):
    """Generator skips < 2 points before computing AUC."""
    bad = []
    for app, app_obj in auc["per_app"].items():
        for pol, traj in app_obj["trajectory_by_policy"].items():
            if len(traj) < 2:
                bad.append((app, pol, len(traj)))
    assert not bad, bad


def test_trajectory_l3_keys_in_scope(auc):
    bad = []
    for app, app_obj in auc["per_app"].items():
        for pol, traj in app_obj["trajectory_by_policy"].items():
            for l3 in traj.keys():
                if l3 not in PAPER_L3_SIZES:
                    bad.append((app, pol, l3))
    assert not bad, bad


# ─── Group C — AUC derivation (trapezoidal on log2) ──────────────────


def test_auc_by_policy_reproduces_trapezoidal_log(auc, grouped_means):
    """AUC is computed from RAW means (un-rounded), then the result is
    rounded. The displayed trajectory_by_policy rounds the means
    separately, so reconstruct the raw trajectory from oracle_gap."""
    mism = []
    for app, app_obj in auc["per_app"].items():
        for pol, reported in app_obj["auc_by_policy"].items():
            traj_mb = {}
            for l3 in app_obj["trajectory_by_policy"][pol].keys():
                traj_mb[L3_MB[l3]] = grouped_means[(app, pol, l3)]
            expected = round(_trap_log_auc(traj_mb), 4)
            if abs(reported - expected) > EPS:
                mism.append((app, pol, reported, expected))
    assert not mism, mism[:5]


def test_auc_values_nonnegative(auc):
    bad = []
    for app, app_obj in auc["per_app"].items():
        for pol, v in app_obj["auc_by_policy"].items():
            if v < -EPS:
                bad.append((app, pol, v))
    assert not bad, bad


def test_auc_values_finite(auc):
    bad = []
    for app, app_obj in auc["per_app"].items():
        for pol, v in app_obj["auc_by_policy"].items():
            if not math.isfinite(v):
                bad.append((app, pol, v))
    assert not bad, bad


# ─── Group D — Ranking + winner + ratios ─────────────────────────────


def test_ranking_sorted_ascending_with_policy_tiebreak(auc):
    bad = []
    for app, app_obj in auc["per_app"].items():
        rk = app_obj["ranking"]
        expected = sorted(
            ({"policy": p, "auc": a} for p, a in app_obj["auc_by_policy"].items()),
            key=lambda kv: (kv["auc"], kv["policy"]),
        )
        if rk != expected:
            bad.append((app, rk, expected))
    assert not bad, bad


def test_winner_matches_ranking_head(auc):
    bad = []
    for app, app_obj in auc["per_app"].items():
        rk = app_obj["ranking"]
        if app_obj["winner"] != rk[0]["policy"]:
            bad.append((app, app_obj["winner"], rk[0]))
        if abs(app_obj["winner_auc"] - rk[0]["auc"]) > EPS:
            bad.append((app, app_obj["winner_auc"], rk[0]))
    assert not bad, bad


def test_runner_up_matches_ranking_second(auc):
    bad = []
    for app, app_obj in auc["per_app"].items():
        rk = app_obj["ranking"]
        if len(rk) < 2:
            continue
        if app_obj["runner_up"] != rk[1]["policy"]:
            bad.append((app, app_obj["runner_up"], rk[1]))
        if abs(app_obj["runner_up_auc"] - rk[1]["auc"]) > EPS:
            bad.append((app, app_obj["runner_up_auc"], rk[1]))
    assert not bad, bad


def test_ratio_winner_over_runner_up_reproduces(auc):
    mism = []
    for app, app_obj in auc["per_app"].items():
        ru = app_obj["runner_up_auc"]
        if ru is None or ru <= 0:
            continue
        expected = round(app_obj["winner_auc"] / ru, 4)
        got = app_obj["auc_ratio_winner_over_runner_up"]
        if abs(got - expected) > EPS:
            mism.append((app, got, expected))
    assert not mism, mism


def test_ratio_winner_over_lru_reproduces(auc):
    mism = []
    for app, app_obj in auc["per_app"].items():
        lru = app_obj["auc_by_policy"].get("LRU")
        if lru is None or lru <= 0:
            continue
        expected = round(app_obj["winner_auc"] / lru, 4)
        got = app_obj["auc_ratio_winner_over_lru"]
        if abs(got - expected) > EPS:
            mism.append((app, got, expected))
    assert not mism, mism


def test_pp_savings_winner_vs_lru_reproduces(auc):
    mism = []
    for app, app_obj in auc["per_app"].items():
        lru = app_obj["auc_by_policy"].get("LRU")
        if lru is None:
            continue
        expected = round(lru - app_obj["winner_auc"], 4)
        got = app_obj["auc_pp_savings_winner_vs_lru"]
        if abs(got - expected) > EPS:
            mism.append((app, got, expected))
    assert not mism, mism


# ─── Group E — Cross-summary ─────────────────────────────────────────


def test_meta_winner_by_app_matches_per_app_winner(auc):
    bad = []
    for app, winner in auc["meta"]["auc_winner_by_app"].items():
        if winner != auc["per_app"][app]["winner"]:
            bad.append((app, winner, auc["per_app"][app]["winner"]))
    assert not bad, bad


def test_lru_never_wins(auc):
    bad = [
        (app, app_obj["winner"])
        for app, app_obj in auc["per_app"].items()
        if app_obj["winner"] == "LRU"
    ]
    assert not bad, bad
