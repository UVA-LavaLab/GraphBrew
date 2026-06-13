"""Pytest gate: per-app oracle gap rankings.

The per-kernel oracle-gap report assigns a per-app rank-1 policy.
This gate enforces that the paper's "no one-size-fits-all" claim is
defensible: different apps have different mean-gap winners, and the
load-bearing app-specific findings (POPT on pr, GRASP on cc) hold.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_JSON = REPO_ROOT / "wiki" / "data" / "oracle_gap_by_app.json"

REQUIRED_APPS = {"bc", "bfs", "cc", "pr", "sssp"}
REQUIRED_POLICIES = {"LRU", "SRRIP", "GRASP", "POPT"}


@pytest.fixture(scope="module")
def doc() -> dict:
    if not DOC_JSON.exists():
        pytest.skip(f"{DOC_JSON} missing; run `make lit-oracle-gap-by-app`")
    return json.loads(DOC_JSON.read_text())


def test_top_level_schema(doc):
    assert {"by_policy_app", "by_app_ranking"}.issubset(doc.keys())


def test_required_apps_present(doc):
    missing = REQUIRED_APPS - set(doc["by_app_ranking"].keys())
    assert not missing, f"per-app ranking missing apps: {missing}"


def test_every_app_has_all_policies(doc):
    for app, ranking in doc["by_app_ranking"].items():
        policies = {r["policy"] for r in ranking}
        missing = REQUIRED_POLICIES - policies
        assert not missing, f"app `{app}` missing policies: {missing}"


def test_ranking_is_sorted_by_mean_gap(doc):
    for app, ranking in doc["by_app_ranking"].items():
        for a, b in zip(ranking, ranking[1:]):
            assert a["mean_gap_pp"] <= b["mean_gap_pp"], (
                f"app `{app}` ranking not sorted: "
                f"{a['policy']}={a['mean_gap_pp']} > {b['policy']}={b['mean_gap_pp']}"
            )


def test_pr_winner_is_popt(doc):
    """Load-bearing: POPT crushes on pr (mean gap 0.100 pp << GRASP 2.251)."""
    rank1 = doc["by_app_ranking"]["pr"][0]
    assert rank1["policy"] == "POPT", (
        f"pr rank-1 policy = {rank1['policy']!r}, expected POPT — "
        "the paper's PR-is-POPT-territory claim no longer holds"
    )
    assert rank1["mean_gap_pp"] <= 0.5, (
        f"POPT/pr mean gap {rank1['mean_gap_pp']:.3f} pp exceeds 0.5 pp floor — "
        "POPT is no longer near-oracle on PageRank"
    )


def test_cc_winner_is_popt_charged_corpus(doc):
    """Charged corpus: POPT is cc rank-1 by mean gap; GRASP no longer wins cc."""
    rank1 = doc["by_app_ranking"]["cc"][0]
    assert rank1["policy"] == "POPT", (
        f"cc rank-1 policy = {rank1['policy']!r}, expected POPT — "
        "the charged-corpus cc mean-gap ordering drifted"
    )


def test_grasp_sssp_is_not_winner(doc):
    """Counter-narrative: GRASP must NOT win sssp (it's typically the
    worst of the four on sssp). If GRASP suddenly wins, something
    structural changed."""
    rank1 = doc["by_app_ranking"]["sssp"][0]
    assert rank1["policy"] != "GRASP", (
        f"sssp rank-1 policy = GRASP — this contradicts the established "
        "narrative that GRASP struggles on SSSP. Did POPT regress?"
    )


def test_no_app_has_lru_as_rank_1(doc):
    """If LRU is ever the per-app rank-1 mean-gap winner, our policy
    portfolio has a serious problem — paper-grade policies should
    always beat baseline on mean gap for at least one cell."""
    lru_winners = [
        app for app, ranking in doc["by_app_ranking"].items()
        if ranking[0]["policy"] == "LRU"
    ]
    assert not lru_winners, (
        f"LRU is rank-1 mean-gap winner on apps {lru_winners} — "
        "no paper-grade policy beats baseline on these kernels"
    )


def test_minimum_sample_per_bucket(doc):
    """Every (policy, app) bucket should have ≥ 15 cells so per-bucket
    means are not noise. roadNet-CA was added explicitly to push the
    pr/bc/cc/sssp/bfs cohorts above this floor."""
    for k, b in doc["by_policy_app"].items():
        assert b["n"] >= 15, (
            f"(policy, app) bucket {k} has only {b['n']} cells — "
            "below the 15-cell floor for stable means"
        )


def test_wins_are_non_negative(doc):
    for k, b in doc["by_policy_app"].items():
        assert b["wins"] >= 0, f"{k} wins={b['wins']} < 0"
        assert b["wins"] <= b["n"], (
            f"{k} wins={b['wins']} > n={b['n']} — impossible"
        )


def test_mean_gap_is_non_negative(doc):
    for k, b in doc["by_policy_app"].items():
        assert b["mean"] >= -1e-6, (
            f"{k} mean gap {b['mean']} < 0 — oracle gap cannot be negative"
        )
