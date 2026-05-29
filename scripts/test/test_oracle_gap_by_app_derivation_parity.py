"""Derivation parity gate for ``wiki/data/oracle_gap_by_app.json``.

Locks the per-kernel oracle-gap breakdown (gate file
``oracle_gap_by_app.json``) against its single upstream —
``oracle_gap.json#rows`` — so any silent drift in the
(policy, app) bucketer, the statistics.fmean/median reducers,
the linear-rank `_p90` percentile, the wins counter, or the
per-app ranking sort key trips a test before the dashboard re-
publishes the "policy-vs-kernel matrix" reviewer narrative.

    oracle_gap.json#rows
              │
        oracle_gap_by_app.py:_per_bucket()
              │
              ▼
    wiki/data/oracle_gap_by_app.json    ← gate target

The gated claim: every (policy, app) bucket aggregates the
expected per-cell gaps, the per-app ranking sorts ASC by
mean_gap_pp (so position 0 in the ranking is always the
per-app "winner" — smallest mean gap to the empirical oracle),
and the wins counter sums per-cell is_winner==1 flags. The
matrix is the corpus's strongest counter to any one-size-fits-
all policy story.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

ARTIFACT_PATH = WIKI_DATA / "oracle_gap_by_app.json"
ORACLE_PATH = WIKI_DATA / "oracle_gap.json"

POLICIES = ("LRU", "SRRIP", "GRASP", "POPT")


def _p90(vals):
    if not vals:
        return 0.0
    sv = sorted(vals)
    idx = min(len(sv) - 1, max(0, int(0.90 * len(sv))))
    return sv[idx]


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def artifact() -> dict:
    if not ARTIFACT_PATH.exists():
        pytest.skip(f"missing {ARTIFACT_PATH}")
    return json.loads(ARTIFACT_PATH.read_text())


@pytest.fixture(scope="module")
def oracle_doc() -> dict:
    if not ORACLE_PATH.exists():
        pytest.skip(f"missing {ORACLE_PATH}")
    return json.loads(ORACLE_PATH.read_text())


@pytest.fixture(scope="module")
def reconstructed(oracle_doc) -> dict:
    """Mirror _per_bucket end-to-end against the same upstream rows."""
    by_pa = defaultdict(list)
    for r in oracle_doc.get("rows", []):
        p, a = r.get("policy"), r.get("app")
        if p not in POLICIES or not a:
            continue
        try:
            gap = float(r["gap_pp"])
        except (ValueError, KeyError, TypeError):
            continue
        by_pa[(p, a)].append({
            "gap_pp": gap,
            "is_winner": int(r.get("is_winner", 0)),
        })

    apps = sorted({a for (_, a) in by_pa})
    by_policy_app = {}
    for (p, a), entries in sorted(by_pa.items()):
        vals = [e["gap_pp"] for e in entries]
        by_policy_app[f"{p}/{a}"] = {
            "n": len(vals),
            "mean": round(statistics.fmean(vals), 4),
            "median": round(statistics.median(vals), 4),
            "p90": round(_p90(vals), 4),
            "max": round(max(vals), 4),
            "wins": sum(e["is_winner"] for e in entries),
        }
    by_app_ranking = {}
    for a in apps:
        ranking = []
        for p in POLICIES:
            b = by_policy_app.get(f"{p}/{a}")
            if b:
                ranking.append({
                    "policy": p,
                    "mean_gap_pp": b["mean"],
                    "median_gap_pp": b["median"],
                    "p90_gap_pp": b["p90"],
                    "max_gap_pp": b["max"],
                    "n": b["n"],
                    "wins": b["wins"],
                })
        ranking.sort(key=lambda x: x["mean_gap_pp"])
        by_app_ranking[a] = ranking
    return {
        "by_policy_app": by_policy_app,
        "by_app_ranking": by_app_ranking,
        "apps": apps,
    }


# ----------------------------------------------------------------------
# Group A: shape & schema
# ----------------------------------------------------------------------

def test_top_level_keys(artifact):
    assert set(artifact.keys()) == {"by_policy_app", "by_app_ranking"}


def test_by_policy_app_bucket_key_format(artifact):
    for key in artifact["by_policy_app"].keys():
        assert "/" in key, f"bucket key {key!r} missing '/'"
        p, a = key.split("/", 1)
        assert p in POLICIES, f"bucket key {key!r}: unknown policy {p!r}"
        assert a, f"bucket key {key!r}: empty app"


def test_by_policy_app_bucket_shape(artifact):
    expected = {"n", "mean", "median", "p90", "max", "wins"}
    for key, b in artifact["by_policy_app"].items():
        missing = expected - set(b.keys())
        assert not missing, f"by_policy_app[{key}] missing fields: {missing}"


def test_by_app_ranking_entry_shape(artifact):
    expected = {
        "policy", "mean_gap_pp", "median_gap_pp", "p90_gap_pp",
        "max_gap_pp", "n", "wins",
    }
    for app, ranking in artifact["by_app_ranking"].items():
        for r in ranking:
            missing = expected - set(r.keys())
            assert not missing, (
                f"by_app_ranking[{app}] entry missing fields: {missing}"
            )
            assert r["policy"] in POLICIES


# ----------------------------------------------------------------------
# Group B: by_policy_app bucket cross-source parity
# ----------------------------------------------------------------------

def test_by_policy_app_keyset_matches_recomputation(artifact, reconstructed):
    assert set(artifact["by_policy_app"].keys()) == (
        set(reconstructed["by_policy_app"].keys())
    )


def test_by_policy_app_records_match_recomputation(artifact, reconstructed):
    expected = reconstructed["by_policy_app"]
    for key, b in artifact["by_policy_app"].items():
        e = expected[key]
        for k in ("n", "mean", "median", "p90", "max", "wins"):
            assert b[k] == e[k], (
                f"by_policy_app[{key}].{k} drift — {b[k]!r} vs {e[k]!r}"
            )


def test_by_policy_app_mean_uses_fmean(artifact, oracle_doc):
    """`mean` must be statistics.fmean rounded to 4dp."""
    rows = oracle_doc.get("rows", [])
    vals_per_bucket = defaultdict(list)
    for r in rows:
        p, a = r.get("policy"), r.get("app")
        if p not in POLICIES or not a:
            continue
        try:
            gap = float(r["gap_pp"])
        except (ValueError, KeyError, TypeError):
            continue
        vals_per_bucket[f"{p}/{a}"].append(gap)
    for key, b in artifact["by_policy_app"].items():
        expected = round(statistics.fmean(vals_per_bucket[key]), 4)
        assert b["mean"] == expected, (
            f"by_policy_app[{key}].mean ≠ round(fmean(values), 4)"
        )


def test_by_policy_app_p90_uses_linear_rank(artifact, oracle_doc):
    """`p90` uses idx = min(n-1, max(0, int(0.90 * n)))."""
    rows = oracle_doc.get("rows", [])
    vals_per_bucket = defaultdict(list)
    for r in rows:
        p, a = r.get("policy"), r.get("app")
        if p not in POLICIES or not a:
            continue
        try:
            gap = float(r["gap_pp"])
        except (ValueError, KeyError, TypeError):
            continue
        vals_per_bucket[f"{p}/{a}"].append(gap)
    for key, b in artifact["by_policy_app"].items():
        vals = vals_per_bucket[key]
        expected = round(_p90(vals), 4)
        assert b["p90"] == expected, (
            f"by_policy_app[{key}].p90 ≠ linear-rank percentile"
        )


def test_by_policy_app_max_matches_max_of_values(artifact, oracle_doc):
    rows = oracle_doc.get("rows", [])
    vals_per_bucket = defaultdict(list)
    for r in rows:
        p, a = r.get("policy"), r.get("app")
        if p not in POLICIES or not a:
            continue
        try:
            gap = float(r["gap_pp"])
        except (ValueError, KeyError, TypeError):
            continue
        vals_per_bucket[f"{p}/{a}"].append(gap)
    for key, b in artifact["by_policy_app"].items():
        expected = round(max(vals_per_bucket[key]), 4)
        assert b["max"] == expected, (
            f"by_policy_app[{key}].max ≠ round(max(values), 4)"
        )


def test_by_policy_app_n_matches_value_count(artifact, oracle_doc):
    rows = oracle_doc.get("rows", [])
    counts = defaultdict(int)
    for r in rows:
        p, a = r.get("policy"), r.get("app")
        if p not in POLICIES or not a:
            continue
        try:
            float(r["gap_pp"])
        except (ValueError, KeyError, TypeError):
            continue
        counts[f"{p}/{a}"] += 1
    for key, b in artifact["by_policy_app"].items():
        assert b["n"] == counts[key], (
            f"by_policy_app[{key}].n ≠ count of valid upstream rows"
        )


def test_by_policy_app_wins_matches_upstream_sum(artifact, oracle_doc):
    rows = oracle_doc.get("rows", [])
    wins = defaultdict(int)
    for r in rows:
        p, a = r.get("policy"), r.get("app")
        if p not in POLICIES or not a:
            continue
        try:
            float(r["gap_pp"])
        except (ValueError, KeyError, TypeError):
            continue
        wins[f"{p}/{a}"] += int(r.get("is_winner", 0))
    for key, b in artifact["by_policy_app"].items():
        assert b["wins"] == wins[key], (
            f"by_policy_app[{key}].wins ≠ sum of upstream is_winner"
        )


def test_by_policy_app_wins_le_n(artifact):
    for key, b in artifact["by_policy_app"].items():
        assert b["wins"] <= b["n"], (
            f"by_policy_app[{key}]: wins {b['wins']} > n {b['n']}"
        )


# ----------------------------------------------------------------------
# Group C: by_app_ranking cross-source parity
# ----------------------------------------------------------------------

def test_by_app_ranking_apps_match_recomputation(artifact, reconstructed):
    assert sorted(artifact["by_app_ranking"].keys()) == reconstructed["apps"]


def test_by_app_ranking_records_match_recomputation(artifact, reconstructed):
    for app, ranking in artifact["by_app_ranking"].items():
        e_ranking = reconstructed["by_app_ranking"][app]
        assert len(ranking) == len(e_ranking), (
            f"by_app_ranking[{app}]: list length drift"
        )
        for r, e in zip(ranking, e_ranking):
            for k in ("policy", "mean_gap_pp", "median_gap_pp", "p90_gap_pp",
                      "max_gap_pp", "n", "wins"):
                assert r[k] == e[k], (
                    f"by_app_ranking[{app}].{k} drift — {r[k]!r} vs {e[k]!r}"
                )


def test_ranking_sorted_asc_by_mean(artifact):
    for app, ranking in artifact["by_app_ranking"].items():
        means = [r["mean_gap_pp"] for r in ranking]
        assert means == sorted(means), (
            f"by_app_ranking[{app}] not sorted ASC by mean_gap_pp: {means}"
        )


def test_ranking_position_zero_is_winner(artifact):
    """Position 0 is the per-app policy with smallest mean gap."""
    for app, ranking in artifact["by_app_ranking"].items():
        if not ranking:
            continue
        assert ranking[0]["mean_gap_pp"] == min(
            r["mean_gap_pp"] for r in ranking
        ), f"by_app_ranking[{app}]: position 0 not the smallest-mean winner"


def test_ranking_policies_unique_per_app(artifact):
    for app, ranking in artifact["by_app_ranking"].items():
        pols = [r["policy"] for r in ranking]
        assert len(pols) == len(set(pols)), (
            f"by_app_ranking[{app}]: duplicate policy entries — {pols}"
        )


def test_ranking_size_matches_bucket_coverage(artifact):
    """Per-app ranking length = number of distinct (policy, app) buckets
    whose app matches."""
    bucket_apps = defaultdict(set)
    for key in artifact["by_policy_app"].keys():
        p, a = key.split("/", 1)
        bucket_apps[a].add(p)
    for app, ranking in artifact["by_app_ranking"].items():
        assert len(ranking) == len(bucket_apps[app]), (
            f"by_app_ranking[{app}] size ≠ bucket coverage for that app"
        )


def test_ranking_aggregates_match_bucket(artifact):
    """Each ranking entry's aggregates must match the bucket they come from."""
    for app, ranking in artifact["by_app_ranking"].items():
        for r in ranking:
            key = f"{r['policy']}/{app}"
            b = artifact["by_policy_app"][key]
            assert r["mean_gap_pp"] == b["mean"]
            assert r["median_gap_pp"] == b["median"]
            assert r["p90_gap_pp"] == b["p90"]
            assert r["max_gap_pp"] == b["max"]
            assert r["n"] == b["n"]
            assert r["wins"] == b["wins"]


# ----------------------------------------------------------------------
# Group D: end-to-end sanity
# ----------------------------------------------------------------------

def test_all_four_policies_observed_per_app(artifact):
    """All 5 apps in the corpus must have all 4 policies — the matrix
    is what makes the report meaningful."""
    for app, ranking in artifact["by_app_ranking"].items():
        assert {r["policy"] for r in ranking} == set(POLICIES), (
            f"by_app_ranking[{app}] missing policies — coverage drift"
        )


def test_total_buckets_equals_apps_times_policies(artifact):
    n_apps = len(artifact["by_app_ranking"])
    assert len(artifact["by_policy_app"]) == n_apps * len(POLICIES), (
        "by_policy_app count != n_apps × 4 — bucket coverage drift"
    )
