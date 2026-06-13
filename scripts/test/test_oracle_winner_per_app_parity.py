"""Per-app oracle-gap rank-1 vs winner-table top-policy parity (gate 86).

Two independent artifacts compute per-app policy quality:

  * oracle_gap_by_app.by_app_ranking[app][0]: the policy with the
    smallest mean distance-to-optimal across that app's cells.
  * policy_winner_table.summary.wins_by_app[app]: the policy that
    *wins outright* most often across that app's cells.

These measure different things — average closeness vs. point-wins —
so they need not agree perfectly. But on a healthy corpus they should
mostly agree, and any disagreement should be a *structural* one we
have already characterized. Under the faithful 1-way-charged P-OPT
corpus, pr (POPT) and bc (GRASP) agree; the frontier kernels (bfs, cc,
sssp) disagree because GRASP wins the most individual cells while a
different policy (POPT on bfs/cc, SRRIP on sssp) holds the smallest
mean oracle-gap.

We lock:

  - For every app the oracle rank-1 policy is among the *top-two*
    policies by win count, except the documented sssp:SRRIP bimodal
    case (smallest mean gap, rare outright winner).
  - At least two of five apps have full agreement (rank-1 == top-1).
  - The three allowed frontier-kernel disagreements are encoded.
  - Every app has a paper-grade win count > 0 for its rank-1 policy
    (the oracle-best policy is not zero-win).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

OGBA_PATH = Path("wiki/data/oracle_gap_by_app.json")
PWT_PATH = Path("wiki/data/policy_winner_table.json")

EXPECTED_APPS = {"bc", "bfs", "cc", "pr", "sssp"}
ALLOWED_DISAGREEMENTS = {
    # (app, oracle_rank1, winner_top1)
    # array-relative GRASP 0.15 + faithful 1-way-charged P-OPT (single-
    # thread): pr (POPT) and bc (GRASP) AGREE. The frontier kernels
    # disagree — POPT keeps the smallest mean oracle-gap (rank1) on bfs
    # and cc but GRASP wins the most individual cells (POPT's wins are
    # concentrated in a few large-gap cells while GRASP wins broadly).
    # On sssp the 1-way RRM charge pushes P-OPT below SRRIP, so SRRIP
    # has the smallest mean gap while GRASP still wins the most cells.
    ("bfs", "POPT", "GRASP"),
    ("cc", "POPT", "GRASP"),
    ("sssp", "SRRIP", "GRASP"),
}
# On sssp the smallest-mean-gap policy (SRRIP) is a rare outright winner
# (GRASP and LRU win more cells), so it falls outside the winner top-two.
# This is the bimodal-frontier signature: GRASP wins many sssp cells by
# small margins but loses a few by large margins (largest mean gap), while
# SRRIP is consistently mid (smallest mean gap) yet seldom the winner.
RANK1_NOT_TOP2_EXCEPTIONS = {("sssp", "SRRIP")}
FULL_AGREEMENT_FLOOR = 2   # pr + bc agree; 3 frontier kernels disagree under the charge
RANK1_WINS_FLOOR = 1       # oracle rank-1 must have >= 1 outright win


def _load(p: Path):
    if not p.exists():
        pytest.skip(f"{p} not generated yet")
    return json.loads(p.read_text())


def _rank1_by_app():
    blob = _load(OGBA_PATH)
    return {app: ranking[0]["policy"] for app, ranking in blob["by_app_ranking"].items()}


def _winners_by_app():
    blob = _load(PWT_PATH)
    return {app: dict(pol_wins) for app, pol_wins in blob["summary"]["wins_by_app"].items()}


def _top1(pol_wins: dict) -> str:
    # Tie-break by policy name to be deterministic.
    return sorted(pol_wins.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _top_k(pol_wins: dict, k: int = 2) -> set[str]:
    """Return policies in the top-k tier, expanding ties at the
    boundary. So for wins {A:15, B:4, C:4} top_k=2 returns {A, B, C}
    because B and C are tied for 2nd and we don't want to penalize a
    rank-1 oracle policy that happens to tie another policy on point-
    wins."""
    if not pol_wins:
        return set()
    ordered = sorted(pol_wins.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(ordered) <= k:
        return {p for p, _ in ordered}
    boundary = ordered[k - 1][1]
    return {p for p, c in ordered if c >= boundary}


def _top2(pol_wins: dict) -> set[str]:
    return _top_k(pol_wins, 2)


# ---------- structural ----------


def test_apps_present_in_both_artifacts():
    r1 = _rank1_by_app()
    wins = _winners_by_app()
    assert set(r1) == EXPECTED_APPS, set(r1) ^ EXPECTED_APPS
    assert set(wins) == EXPECTED_APPS, set(wins) ^ EXPECTED_APPS


def test_oracle_rank1_in_winner_top_two_for_every_app():
    r1 = _rank1_by_app()
    wins = _winners_by_app()
    bad = []
    for app, oracle_pol in r1.items():
        if (app, oracle_pol) in RANK1_NOT_TOP2_EXCEPTIONS:
            continue
        top2 = _top2(wins[app])
        if oracle_pol not in top2:
            bad.append((app, oracle_pol, top2))
    assert not bad, bad


def test_full_agreement_floor():
    r1 = _rank1_by_app()
    wins = _winners_by_app()
    agree = sum(1 for app in EXPECTED_APPS if r1[app] == _top1(wins[app]))
    assert agree >= FULL_AGREEMENT_FLOOR, agree


def test_disagreements_are_known():
    r1 = _rank1_by_app()
    wins = _winners_by_app()
    observed = []
    for app in EXPECTED_APPS:
        oracle = r1[app]
        winner = _top1(wins[app])
        if oracle != winner:
            observed.append((app, oracle, winner))
    unknown = set(observed) - ALLOWED_DISAGREEMENTS
    assert not unknown, unknown


def test_oracle_rank1_has_nonzero_wins_per_app():
    r1 = _rank1_by_app()
    wins = _winners_by_app()
    bad = []
    for app, oracle_pol in r1.items():
        if wins[app].get(oracle_pol, 0) < RANK1_WINS_FLOOR:
            bad.append((app, oracle_pol, wins[app]))
    assert not bad, bad


# ---------- spot-checks anchoring the headlines ----------


def test_pr_oracle_rank1_is_popt():
    r1 = _rank1_by_app()
    assert r1["pr"] == "POPT", r1["pr"]


def test_pr_winner_is_popt():
    wins = _winners_by_app()
    assert _top1(wins["pr"]) == "POPT", wins["pr"]


def test_cc_oracle_rank1_is_popt():
    """Under the 1-way charge, P-OPT still has the smallest mean
    oracle-gap on cc (rank1), even though GRASP wins the most cc cells
    outright (see test_cc_winner_is_grasp)."""
    r1 = _rank1_by_app()
    assert r1["cc"] == "POPT", r1["cc"]


def test_cc_winner_is_grasp():
    wins = _winners_by_app()
    assert _top1(wins["cc"]) == "GRASP", wins["cc"]


def test_sssp_oracle_rank1_is_srrip():
    """The faithful 1-way RRM charge pushes P-OPT below SRRIP on sssp's
    saturated small cells, so SRRIP has the smallest mean oracle-gap on
    sssp (was POPT in the uncharged corpus)."""
    r1 = _rank1_by_app()
    assert r1["sssp"] == "SRRIP", r1["sssp"]


def test_sssp_disagreement_documented():
    """Sanity: the documented sssp disagreement still holds (SRRIP has the
    smallest mean oracle-gap but GRASP wins the most cells), otherwise the
    ALLOWED_DISAGREEMENTS set may be hiding a healthy state. pr and bc are
    the two agreements (POPT and GRASP respectively)."""
    r1 = _rank1_by_app()
    wins = _winners_by_app()
    assert r1["sssp"] == "SRRIP", r1["sssp"]
    assert _top1(wins["sssp"]) == "GRASP", wins["sssp"]
    # pr and bc are agreements (oracle rank1 == winner top1):
    assert r1["pr"] == _top1(wins["pr"]) == "POPT", (r1["pr"], wins["pr"])
    assert r1["bc"] == _top1(wins["bc"]) == "GRASP", (r1["bc"], wins["bc"])


def test_no_app_has_lru_as_oracle_rank1():
    r1 = _rank1_by_app()
    bad = [app for app, pol in r1.items() if pol == "LRU"]
    assert not bad, bad


def test_no_app_has_lru_as_winner_top1():
    wins = _winners_by_app()
    bad = [app for app in EXPECTED_APPS if _top1(wins[app]) == "LRU"]
    assert not bad, bad
