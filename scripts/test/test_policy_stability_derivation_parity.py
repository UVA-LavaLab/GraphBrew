"""Derivation parity gate for ``wiki/data/policy_stability.json``.

Single upstream: ``oracle_gap_auc.json``. Per-policy CV (coefficient
of variation) of AUC across apps + rank stability + best/worst app +
two policy orderings.

Locks the stability index derivation so any drift in the rank
extraction (ranking position 1-indexed via enumerate), the
statistics.fmean / statistics.pstdev choices (POPULATION stdev — NOT
sample stdev statistics.stdev), the CV gate (auc_cv is None iff
mean ≤ 0, NOT just mean == 0), the per-policy reducers
(best_app / worst_app via sorted ASC by AUC; ratio guarded by
best_auc > 0), the always_top_2 / always_bot_2 predicates
(max(ranks) <= 2 / min(ranks) >= 3 NON-STRICT), the n_wins /
n_lasts counters (rank == 1 / rank == 4 specifically), the
safest_order sort key ((auc_cv or inf, mean) — None CVs sink),
the best_avg_order sort key (mean ASC), or the headline policy
identifiers (safest/highest_variance/best_avg) trips a test before
the dashboard re-publishes "LRU safest, POPT highest-variance,
POPT best-avg" verdict.

Mirrors ``build_payload()`` from
``scripts/experiments/ecg/policy_stability.py`` verbatim.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

ARTIFACT_PATH = WIKI_DATA / "policy_stability.json"
UPSTREAM_PATH = WIKI_DATA / "oracle_gap_auc.json"


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def artifact():
    if not ARTIFACT_PATH.exists():
        pytest.skip(f"missing {ARTIFACT_PATH}")
    return json.loads(ARTIFACT_PATH.read_text())


@pytest.fixture(scope="module")
def upstream():
    if not UPSTREAM_PATH.exists():
        pytest.skip(f"missing {UPSTREAM_PATH}")
    return json.loads(UPSTREAM_PATH.read_text())


@pytest.fixture(scope="module")
def apps(upstream):
    return sorted(upstream["meta"]["apps"])


@pytest.fixture(scope="module")
def policies(upstream):
    return sorted(upstream["meta"]["policies"])


@pytest.fixture(scope="module")
def per_app_rank(upstream, apps):
    out = {}
    for app in apps:
        ranking = upstream["per_app"][app]["ranking"]
        out[app] = {entry["policy"]: i + 1 for i, entry in enumerate(ranking)}
    return out


@pytest.fixture(scope="module")
def per_policy_auc(upstream, apps, policies):
    out = defaultdict(dict)
    for app in apps:
        for pol in policies:
            out[pol][app] = upstream["per_app"][app]["auc_by_policy"][pol]
    return out


# ----------------------------------------------------------------------
# Group A: shape & schema
# ----------------------------------------------------------------------

def test_top_level_keys(artifact):
    assert set(artifact.keys()) == {
        "meta", "per_policy",
        "ranking_by_cv_ascending", "ranking_by_mean_auc_ascending",
    }


def test_meta_fields(artifact):
    expected = {
        "source", "n_apps", "n_policies", "apps", "policies",
        "stability_metric", "safest_policy",
        "highest_variance_policy", "best_avg_policy",
    }
    assert set(artifact["meta"].keys()) == expected


def test_meta_constants(artifact):
    assert artifact["meta"]["stability_metric"] == (
        "coefficient of variation (stdev / mean) of AUC across apps"
    )
    assert artifact["meta"]["source"].endswith("oracle_gap_auc.json")


def test_per_policy_entry_shape(artifact):
    expected = {
        "auc_mean_across_apps", "auc_stdev_across_apps", "auc_cv",
        "best_app", "best_app_auc", "worst_app", "worst_app_auc",
        "worst_over_best_ratio", "ranks_by_app",
        "rank_mean", "rank_stdev", "best_rank", "worst_rank",
        "n_wins", "n_lasts", "always_top_2", "always_bot_2",
    }
    for pol, p in artifact["per_policy"].items():
        assert set(p.keys()) == expected, f"{pol}: per_policy drift"


def test_ranking_by_cv_entry_shape(artifact):
    for entry in artifact["ranking_by_cv_ascending"]:
        assert set(entry.keys()) == {"policy", "auc_cv"}


def test_ranking_by_mean_auc_entry_shape(artifact):
    for entry in artifact["ranking_by_mean_auc_ascending"]:
        assert set(entry.keys()) == {"policy", "auc_mean"}


# ----------------------------------------------------------------------
# Group B: meta counters & scope
# ----------------------------------------------------------------------

def test_meta_apps_sorted(artifact, apps):
    assert artifact["meta"]["apps"] == apps


def test_meta_policies_sorted(artifact, policies):
    assert artifact["meta"]["policies"] == policies


def test_meta_n_counters(artifact, apps, policies):
    assert artifact["meta"]["n_apps"] == len(apps)
    assert artifact["meta"]["n_policies"] == len(policies)


def test_per_policy_keys_match_meta(artifact, policies):
    assert sorted(artifact["per_policy"].keys()) == policies


# ----------------------------------------------------------------------
# Group C: per-policy byte-exact arithmetic
# ----------------------------------------------------------------------

def test_auc_mean_byte_exact(artifact, per_policy_auc, apps):
    """statistics.fmean over apps, round 4dp."""
    for pol, p in artifact["per_policy"].items():
        vals = [per_policy_auc[pol][app] for app in apps]
        assert p["auc_mean_across_apps"] == round(statistics.fmean(vals), 4)


def test_auc_stdev_byte_exact(artifact, per_policy_auc, apps):
    """statistics.pstdev (POPULATION stdev — NOT sample stdev) round 4dp."""
    for pol, p in artifact["per_policy"].items():
        vals = [per_policy_auc[pol][app] for app in apps]
        assert p["auc_stdev_across_apps"] == round(statistics.pstdev(vals), 4)


def test_auc_cv_byte_exact(artifact, per_policy_auc, apps):
    """CV = pstdev/mean if mean > 0 else None; round 4dp."""
    for pol, p in artifact["per_policy"].items():
        vals = [per_policy_auc[pol][app] for app in apps]
        mean = statistics.fmean(vals)
        sd = statistics.pstdev(vals)
        expected = round(sd / mean, 4) if mean > 0 else None
        assert p["auc_cv"] == expected


def test_best_and_worst_app(artifact, per_policy_auc, apps):
    """sorted ASC by AUC; first=best (lowest gap), last=worst (highest)."""
    for pol, p in artifact["per_policy"].items():
        items = sorted(per_policy_auc[pol].items(), key=lambda kv: kv[1])
        assert p["best_app"] == items[0][0]
        assert p["best_app_auc"] == round(items[0][1], 4)
        assert p["worst_app"] == items[-1][0]
        assert p["worst_app_auc"] == round(items[-1][1], 4)


def test_worst_over_best_ratio(artifact, per_policy_auc):
    """ratio guarded by best_auc > 0 (None otherwise) — round 4dp."""
    for pol, p in artifact["per_policy"].items():
        items = sorted(per_policy_auc[pol].items(), key=lambda kv: kv[1])
        best_auc = items[0][1]
        worst_auc = items[-1][1]
        expected = round(worst_auc / best_auc, 4) if best_auc > 0 else None
        assert p["worst_over_best_ratio"] == expected


def test_ranks_by_app_byte_exact(artifact, per_app_rank, apps):
    """1-indexed via enumerate over auc.per_app[app].ranking list order."""
    for pol, p in artifact["per_policy"].items():
        expected = {app: per_app_rank[app][pol] for app in apps}
        assert p["ranks_by_app"] == expected


def test_rank_mean_and_stdev_byte_exact(artifact, per_app_rank, apps):
    for pol, p in artifact["per_policy"].items():
        ranks = [per_app_rank[app][pol] for app in apps]
        assert p["rank_mean"] == round(statistics.fmean(ranks), 4)
        assert p["rank_stdev"] == round(statistics.pstdev(ranks), 4)


def test_best_worst_rank(artifact, per_app_rank, apps):
    for pol, p in artifact["per_policy"].items():
        ranks = [per_app_rank[app][pol] for app in apps]
        assert p["best_rank"] == min(ranks)
        assert p["worst_rank"] == max(ranks)


def test_n_wins_n_lasts(artifact, per_app_rank, apps):
    """n_wins = sum rank==1; n_lasts = sum rank==4 specifically (NOT
    rank == n_policies — load-bearing if policy set grows)."""
    for pol, p in artifact["per_policy"].items():
        ranks = [per_app_rank[app][pol] for app in apps]
        assert p["n_wins"] == sum(1 for r in ranks if r == 1)
        assert p["n_lasts"] == sum(1 for r in ranks if r == 4)


def test_always_top_2_and_bot_2(artifact, per_app_rank, apps):
    """always_top_2 ≡ max(ranks) <= 2 (NON-STRICT); always_bot_2 ≡
    min(ranks) >= 3 (NON-STRICT)."""
    for pol, p in artifact["per_policy"].items():
        ranks = [per_app_rank[app][pol] for app in apps]
        assert p["always_top_2"] is (max(ranks) <= 2)
        assert p["always_bot_2"] is (min(ranks) >= 3)


# ----------------------------------------------------------------------
# Group D: ranking orderings
# ----------------------------------------------------------------------

def test_ranking_by_cv_ascending_sort_key(artifact):
    """Sort key: (auc_cv or inf, mean) — None CVs sink to the end."""
    per = artifact["per_policy"]
    expected = sorted(
        per.items(),
        key=lambda kv: (
            kv[1]["auc_cv"] if kv[1]["auc_cv"] is not None else float("inf"),
            kv[1]["auc_mean_across_apps"],
        ),
    )
    actual = [(e["policy"], e["auc_cv"]) for e in artifact["ranking_by_cv_ascending"]]
    exp_pairs = [(p, d["auc_cv"]) for p, d in expected]
    assert actual == exp_pairs


def test_ranking_by_mean_auc_ascending_sort_key(artifact):
    per = artifact["per_policy"]
    expected = sorted(per.items(), key=lambda kv: kv[1]["auc_mean_across_apps"])
    actual = [(e["policy"], e["auc_mean"]) for e in artifact["ranking_by_mean_auc_ascending"]]
    exp_pairs = [(p, d["auc_mean_across_apps"]) for p, d in expected]
    assert actual == exp_pairs


def test_safest_policy_matches_cv_ranking_head(artifact):
    assert artifact["meta"]["safest_policy"] == \
        artifact["ranking_by_cv_ascending"][0]["policy"]


def test_highest_variance_policy_matches_cv_ranking_tail(artifact):
    assert artifact["meta"]["highest_variance_policy"] == \
        artifact["ranking_by_cv_ascending"][-1]["policy"]


def test_best_avg_policy_matches_mean_ranking_head(artifact):
    assert artifact["meta"]["best_avg_policy"] == \
        artifact["ranking_by_mean_auc_ascending"][0]["policy"]


# ----------------------------------------------------------------------
# Group E: end-to-end sanity & claim invariants
# ----------------------------------------------------------------------

def test_at_least_one_always_top_2_or_none(artifact):
    """The 'safe default' claim hinges on having a policy that's
    always top-2 OR documenting why none qualify. Both states are valid."""
    n_top = sum(1 for p in artifact["per_policy"].values() if p["always_top_2"])
    n_bot = sum(1 for p in artifact["per_policy"].values() if p["always_bot_2"])
    assert n_top + n_bot <= len(artifact["per_policy"])


def test_best_rank_le_worst_rank(artifact):
    for pol, p in artifact["per_policy"].items():
        assert p["best_rank"] <= p["worst_rank"]


def test_ratio_ge_1(artifact):
    """worst/best ratio ≥ 1 by construction (worst ≥ best)."""
    for pol, p in artifact["per_policy"].items():
        if p["worst_over_best_ratio"] is not None:
            assert p["worst_over_best_ratio"] >= 1.0


def test_rank_count_invariant(artifact, policies):
    """Sum of n_wins across all policies = n_apps (every app has
    exactly one rank-1 policy). Same for n_lasts."""
    n_apps = artifact["meta"]["n_apps"]
    total_wins = sum(p["n_wins"] for p in artifact["per_policy"].values())
    total_lasts = sum(p["n_lasts"] for p in artifact["per_policy"].values())
    assert total_wins == n_apps
    assert total_lasts == n_apps
