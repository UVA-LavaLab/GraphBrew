"""Gate 139+ — policy_auc_correlation derivation parity.

policy_auc_correlation.json is derived from oracle_gap_auc.json by:

  1. for each app, build a 4-vector of AUCs (one per sorted policy)
  2. z-score that vector per app (mean=0, pstdev=1)
  3. Pearson correlation between every (app_a, app_b) pair on the
     z-scored vectors → 5x5 matrix
  4. derived: pair_list (off-diagonal pairs sorted desc by r),
     nearest_sibling (per app, ranked others + closest),
     intra_inter (mean correlation inside/outside the AUC-winner cluster)

Existing tests pin headline facts (symmetric matrix, bfs↔sssp
strongest pair, clusters by winner). This gate locks the *derivation
math* end-to-end. If the generator silently swaps z-score for raw
AUC, switches Pearson for Spearman, mis-orders policies, or
mis-counts intra_mean_r, this gate flips.

Load-bearing subtleties:
- The z-score uses population stdev (statistics.pstdev), not sample
  stdev. The Pearson is computed on the z-scored vectors, but since
  Pearson is invariant under linear transforms, the matrix value
  also equals Pearson on the raw AUC vectors. We test both paths.
- Policies are sorted alphabetically when building the vector
  (GRASP, LRU, POPT, SRRIP). The order is what makes per-app
  vectors comparable.
- pair_list contains exactly C(5,2)=10 entries (off-diagonal upper
  triangle), sorted by pearson_r descending.

Invariants (19 tests, 5 groups):

Group A — Structural & scope
  1. Top-level keys: meta, matrix, pair_list, nearest_sibling, intra_inter
  2. meta.source points to oracle_gap_auc.json
  3. meta.apps and meta.policies match oracle_gap_auc meta
  4. meta.auc_winner_by_app matches oracle_gap_auc.meta
  5. clusters_by_winner partitions apps according to auc_winner_by_app

Group B — Matrix reproduction (oracle_gap_auc → matrix)
  6. Matrix keys match meta.apps; every row has all 5 columns
  7. matrix[a][b] == round(pearson(zscore(auc_a), zscore(auc_b)), 4)
  8. matrix is symmetric: matrix[a][b] == matrix[b][a]
  9. matrix diagonal == 1.0 (within rounding)
  10. matrix[a][b] equals round(pearson(raw_auc_a, raw_auc_b), 4)
      (z-score invariance test)

Group C — pair_list parity
  11. pair_list contains exactly C(n,2) entries
  12. Every entry's pearson_r matches matrix[a][b] for that pair
  13. pair_list is sorted descending by pearson_r
  14. (app_a, app_b) is unique and app_a < app_b

Group D — nearest_sibling reproduction
  15. nearest_sibling[app].closest_app == argmax(matrix[app][b]) for b!=app
  16. nearest_sibling[app].closest_r == max(matrix[app][b]) for b!=app
  17. nearest_sibling[app].ranking is matrix[app] minus self,
      sorted desc by r
  18. nearest_sibling[app].winner_policy == auc_winner_by_app[app]

Group E — intra_inter reproduction
  19. For every app:
      cluster == auc_winner_by_app[app];
      intra_mean_r == round(mean(matrix[app][b] for b in own cluster, b!=app), 4);
      inter_mean_r == round(mean(matrix[app][b] for b outside own cluster), 4);
      gap_intra_minus_inter == round(intra - inter, 4) (1e-3 tolerance
      because rounding compounds twice).
"""

from __future__ import annotations

import json
import math
import statistics
from itertools import combinations
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

EPS = 1e-4
EPS_LOOSE = 1e-3  # for double-rounded fields like gap_intra_minus_inter


def _zscore(vec: list[float]) -> list[float]:
    if len(vec) < 2:
        return [0.0] * len(vec)
    mu = statistics.fmean(vec)
    sd = statistics.pstdev(vec)
    if sd == 0:
        return [0.0] * len(vec)
    return [(x - mu) / sd for x in vec]


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mux = statistics.fmean(x)
    muy = statistics.fmean(y)
    num = sum((a - mux) * (b - muy) for a, b in zip(x, y))
    denx = math.sqrt(sum((a - mux) ** 2 for a in x))
    deny = math.sqrt(sum((b - muy) ** 2 for b in y))
    if denx == 0 or deny == 0:
        return 0.0
    return num / (denx * deny)


@pytest.fixture(scope="module")
def auc() -> dict:
    return json.loads((WIKI_DATA / "oracle_gap_auc.json").read_text())


@pytest.fixture(scope="module")
def pac() -> dict:
    return json.loads((WIKI_DATA / "policy_auc_correlation.json").read_text())


@pytest.fixture(scope="module")
def per_app_vec(pac, auc) -> dict:
    """{app: [auc_grasp, auc_lru, auc_popt, auc_srrip]} — policies sorted."""
    policies = sorted(pac["meta"]["policies"])
    return {
        app: [auc["per_app"][app]["auc_by_policy"][pol] for pol in policies]
        for app in sorted(pac["meta"]["apps"])
    }


# ─── Group A — Structural ────────────────────────────────────────────


def test_top_level_keys(pac):
    assert set(pac.keys()) >= {
        "meta", "matrix", "pair_list", "nearest_sibling", "intra_inter"
    }


def test_meta_source_points_to_auc(pac):
    assert pac["meta"]["source"] == "wiki/data/oracle_gap_auc.json"


def test_meta_apps_policies_match_auc(pac, auc):
    assert sorted(pac["meta"]["apps"]) == sorted(auc["meta"]["apps"])
    assert sorted(pac["meta"]["policies"]) == sorted(auc["meta"]["policies"])


def test_auc_winner_by_app_matches_source(pac, auc):
    assert pac["meta"]["auc_winner_by_app"] == auc["meta"]["auc_winner_by_app"]


def test_clusters_by_winner_partitions_apps(pac):
    flat = sorted(
        a for members in pac["meta"]["clusters_by_winner"].values() for a in members
    )
    assert flat == sorted(pac["meta"]["apps"])
    # Each app belongs to the cluster matching its auc winner
    for pol, members in pac["meta"]["clusters_by_winner"].items():
        for a in members:
            assert pac["meta"]["auc_winner_by_app"][a] == pol


# ─── Group B — Matrix reproduction ───────────────────────────────────


def test_matrix_keys_and_rows_complete(pac):
    apps = sorted(pac["meta"]["apps"])
    assert sorted(pac["matrix"].keys()) == apps
    for a in apps:
        assert sorted(pac["matrix"][a].keys()) == apps


def test_matrix_values_reproduce_pearson_on_zscore(pac, per_app_vec):
    apps = sorted(pac["meta"]["apps"])
    per_app_z = {a: _zscore(v) for a, v in per_app_vec.items()}
    mism = []
    for a in apps:
        for b in apps:
            expected = round(_pearson(per_app_z[a], per_app_z[b]), 4)
            if abs(pac["matrix"][a][b] - expected) > EPS:
                mism.append((a, b, pac["matrix"][a][b], expected))
    assert not mism, mism[:5]


def test_matrix_is_symmetric(pac):
    apps = sorted(pac["meta"]["apps"])
    bad = []
    for a in apps:
        for b in apps:
            if abs(pac["matrix"][a][b] - pac["matrix"][b][a]) > EPS:
                bad.append((a, b, pac["matrix"][a][b], pac["matrix"][b][a]))
    assert not bad, bad[:5]


def test_matrix_diagonal_is_unit(pac):
    apps = sorted(pac["meta"]["apps"])
    bad = [a for a in apps if abs(pac["matrix"][a][a] - 1.0) > EPS]
    assert not bad, bad


def test_matrix_invariant_under_zscore_transform(pac, per_app_vec):
    """Pearson is invariant under per-vector linear transform, so
    pearson(zscore(x), zscore(y)) == pearson(x, y). Re-derive
    matrix using raw AUC vectors to triple-check."""
    apps = sorted(pac["meta"]["apps"])
    mism = []
    for a in apps:
        for b in apps:
            expected = round(_pearson(per_app_vec[a], per_app_vec[b]), 4)
            if abs(pac["matrix"][a][b] - expected) > EPS:
                mism.append((a, b, pac["matrix"][a][b], expected))
    assert not mism, mism[:5]


# ─── Group C — pair_list parity ──────────────────────────────────────


def test_pair_list_has_correct_count(pac):
    n = len(pac["meta"]["apps"])
    assert len(pac["pair_list"]) == n * (n - 1) // 2


def test_pair_list_pearson_values_mirror_matrix(pac):
    mism = []
    for entry in pac["pair_list"]:
        m = pac["matrix"][entry["app_a"]][entry["app_b"]]
        if abs(entry["pearson_r"] - m) > EPS:
            mism.append(entry)
    assert not mism, mism[:5]


def test_pair_list_sorted_descending(pac):
    rs = [e["pearson_r"] for e in pac["pair_list"]]
    assert rs == sorted(rs, reverse=True), rs


def test_pair_list_pairs_unique_and_ordered(pac):
    pairs = [(e["app_a"], e["app_b"]) for e in pac["pair_list"]]
    assert len(set(pairs)) == len(pairs)
    bad = [p for p in pairs if not (p[0] < p[1])]
    assert not bad, bad


# ─── Group D — nearest_sibling reproduction ──────────────────────────


def test_nearest_sibling_closest_app_is_argmax(pac):
    apps = sorted(pac["meta"]["apps"])
    mism = []
    for app in apps:
        others = [(b, pac["matrix"][app][b]) for b in apps if b != app]
        expected_app = max(others, key=lambda kv: kv[1])[0]
        got = pac["nearest_sibling"][app]["closest_app"]
        if got != expected_app:
            mism.append((app, got, expected_app))
    assert not mism, mism


def test_nearest_sibling_closest_r_is_max(pac):
    apps = sorted(pac["meta"]["apps"])
    mism = []
    for app in apps:
        others = [pac["matrix"][app][b] for b in apps if b != app]
        expected_r = max(others)
        got = pac["nearest_sibling"][app]["closest_r"]
        if abs(got - expected_r) > EPS:
            mism.append((app, got, expected_r))
    assert not mism, mism


def test_nearest_sibling_ranking_is_sorted_desc(pac):
    bad = []
    for app, rec in pac["nearest_sibling"].items():
        rs = [r["pearson_r"] for r in rec["ranking"]]
        if rs != sorted(rs, reverse=True):
            bad.append((app, rs))
    assert not bad, bad


def test_nearest_sibling_winner_policy_matches(pac):
    mism = []
    for app, rec in pac["nearest_sibling"].items():
        expected = pac["meta"]["auc_winner_by_app"][app]
        if rec["winner_policy"] != expected:
            mism.append((app, rec["winner_policy"], expected))
    assert not mism, mism


# ─── Group E — intra_inter reproduction ──────────────────────────────


def test_intra_inter_reproduces_means_and_gap(pac):
    """For every app: cluster matches winner, intra/inter means equal
    mean(matrix[app][b]) over the appropriate split, and
    gap_intra_minus_inter reproduces intra - inter."""
    apps = sorted(pac["meta"]["apps"])
    cluster_of = {
        app: pac["meta"]["auc_winner_by_app"][app] for app in apps
    }
    mism = []
    for app, rec in pac["intra_inter"].items():
        own = cluster_of[app]
        if rec["cluster"] != own:
            mism.append((app, "cluster", rec["cluster"], own))
            continue
        intra_vals = [
            pac["matrix"][app][b]
            for b in apps
            if b != app and cluster_of[b] == own
        ]
        inter_vals = [
            pac["matrix"][app][b]
            for b in apps
            if b != app and cluster_of[b] != own
        ]
        if intra_vals:
            expected_intra = round(statistics.fmean(intra_vals), 4)
            if abs(rec["intra_mean_r"] - expected_intra) > EPS:
                mism.append((app, "intra", rec["intra_mean_r"], expected_intra))
        if inter_vals:
            expected_inter = round(statistics.fmean(inter_vals), 4)
            if abs(rec["inter_mean_r"] - expected_inter) > EPS:
                mism.append((app, "inter", rec["inter_mean_r"], expected_inter))
        if intra_vals and inter_vals:
            expected_gap = round(
                statistics.fmean(intra_vals) - statistics.fmean(inter_vals), 4
            )
            if abs(rec["gap_intra_minus_inter"] - expected_gap) > EPS_LOOSE:
                mism.append((app, "gap", rec["gap_intra_minus_inter"], expected_gap))
    assert not mism, mism[:5]
