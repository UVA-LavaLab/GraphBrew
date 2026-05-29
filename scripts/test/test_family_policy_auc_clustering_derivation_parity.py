"""Gate 140+ — family_policy_auc_clustering derivation parity.

family_policy_auc_clustering.json is derived from oracle_gap.json by:

  1. pool gap_pp by (family, app, policy, l3) — mean across graphs
  2. for each family with full L3 coverage on every (app,pol) cell:
     a. compute per-(app,pol) AUC via trapezoidal on log2(MB) of
        the pooled means
     b. winner_by_app = argmin AUC per app
     c. correlation matrix via z-score + Pearson across apps
     d. intra/inter cluster means using GLOBAL_CLUSTERS (the global
        AUC-winner partition: GRASP={bc,cc}, POPT={bfs,pr,sssp})

Existing tests pin facts (qualifying families full L3 coverage,
intra dominates inter, deviations cap on citation). This gate locks
the *derivation arithmetic* from raw oracle_gap rows through pooled
means → AUC → winners → correlation → intra/inter cluster stats.

Load-bearing subtleties:
- AUC uses statistics.fmean over per-(family,app,pol,l3) raw rows
  (mean across graphs in the family), then trapezoidal on log2 MB.
- Family qualification REQUIRES full L3 coverage on every (app,pol)
  cell. If even one cell is missing an L3 size, the family is
  marked qualified=False and skipped from per_family math.
- The intra/inter means use the GLOBAL_CLUSTERS partition (not
  per-family winners). This is a deliberate design choice: it
  asks 'do the per-family AUC ratings cluster the apps the SAME
  way the global pool does?' rather than self-validating.

Invariants (18 tests, 5 groups):

Group A — Structural & scope
  1. Top-level keys: meta, per_family
  2. meta.source points to oracle_gap.json
  3. meta.global_clusters partitions meta.apps exactly
  4. per_family keys match meta.families
  5. qualifying_families subset of meta.families; every entry
     in per_family is qualified=True iff in qualifying_families

Group B — AUC derivation (oracle_gap → auc_by_app_policy)
  6. For every qualifying family + every (app, pol):
     auc_by_app_policy[app][pol] == round(trapezoidal_log_auc(
       per_(family,app,pol,l3) means across graphs), 4) to 1e-3
     (rounding compounds twice: mean → AUC → round)
  7. auc_by_app_policy values are non-negative
  8. n_graphs == count of distinct graphs in family observed in
     paper-L3 rows

Group C — Winner reproduction (auc → winner_by_app)
  9. winner_by_app[app] == argmin(auc_by_app_policy[app])
  10. winner_matches_global[app] == (winner_by_app[app] == GLOBAL_WINNER[app])
  11. winners_matching == sum(winner_matches_global.values())
  12. n_apps == len(meta.apps)

Group D — Correlation matrix reproduction
  13. matrix[a][b] == round(pearson(zscore(auc_a), zscore(auc_b)), 4)
      to 1e-3 (z-score of AUC vector at the policy axis)
  14. Matrix is symmetric; diagonal == 1.0
  15. Matrix equals pearson on raw AUC vectors (z-score invariance)

Group E — intra/inter cluster reproduction
  16. intra_cluster_mean_r reproduces fmean(per-app intra means) to 1e-3
      using GLOBAL_CLUSTERS partition
  17. inter_cluster_mean_r reproduces fmean(per-app inter means) to 1e-3
  18. intra_minus_inter == round(intra - inter, 4) at 1e-3;
      intra_dominates == (intra_minus_inter > 0)
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
L3_OCTAVES_MB = {"1MB": 0.0, "4MB": 2.0, "8MB": 3.0}

GLOBAL_CLUSTERS = {
    "GRASP": ("bc", "cc"),
    "POPT": ("bfs", "pr", "sssp"),
}
GLOBAL_WINNER = {
    "bc": "GRASP", "cc": "GRASP",
    "bfs": "POPT", "pr": "POPT", "sssp": "POPT",
}

EPS = 1e-3  # AUC compounding: mean → AUC → round to 4dp


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


def _trap_auc(values_by_l3: dict[str, float]) -> float:
    xs = [L3_OCTAVES_MB[s] for s in PAPER_L3_SIZES]
    ys = [values_by_l3[s] for s in PAPER_L3_SIZES]
    auc = 0.0
    for i in range(len(xs) - 1):
        auc += 0.5 * (xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1])
    return auc


@pytest.fixture(scope="module")
def og() -> dict:
    return json.loads((WIKI_DATA / "oracle_gap.json").read_text())


@pytest.fixture(scope="module")
def fpac() -> dict:
    return json.loads(
        (WIKI_DATA / "family_policy_auc_clustering.json").read_text()
    )


@pytest.fixture(scope="module")
def pooled_means(og) -> dict:
    """{(family, app, pol, l3): mean(gap_pp across graphs in family)}"""
    pooled: dict[tuple, list[float]] = defaultdict(list)
    for r in og["rows"]:
        if r["l3_size"] in PAPER_L3_SIZES:
            pooled[(r["family"], r["app"], r["policy"], r["l3_size"])].append(
                float(r["gap_pp"])
            )
    return {k: statistics.fmean(v) for k, v in pooled.items()}


# ─── Group A — Structural ────────────────────────────────────────────


def test_top_level_keys(fpac):
    assert set(fpac.keys()) >= {"meta", "per_family"}


def test_meta_source_points_to_oracle_gap(fpac):
    assert fpac["meta"]["source"] == "wiki/data/oracle_gap.json"


def test_global_clusters_partition_apps(fpac):
    flat = sorted(
        a for members in fpac["meta"]["global_clusters"].values() for a in members
    )
    assert flat == sorted(fpac["meta"]["apps"])


def test_per_family_keys_match_meta(fpac):
    assert sorted(fpac["per_family"].keys()) == sorted(fpac["meta"]["families"])


def test_qualified_subset_matches_qualifying_families(fpac):
    listed = set(fpac["meta"]["qualifying_families"])
    actually = {
        f for f, info in fpac["per_family"].items() if info.get("qualified")
    }
    assert listed == actually


# ─── Group B — AUC derivation ────────────────────────────────────────


def test_auc_by_app_policy_reproduces_trapezoidal_on_pooled_means(
    fpac, pooled_means
):
    apps = sorted(fpac["meta"]["apps"])
    policies = sorted(fpac["meta"]["policies"])
    mism = []
    for fam in fpac["meta"]["qualifying_families"]:
        info = fpac["per_family"][fam]
        for app in apps:
            for pol in policies:
                vals = {
                    l3: pooled_means[(fam, app, pol, l3)]
                    for l3 in PAPER_L3_SIZES
                    if (fam, app, pol, l3) in pooled_means
                }
                if not all(l3 in vals for l3 in PAPER_L3_SIZES):
                    continue
                expected = round(_trap_auc(vals), 4)
                got = info["auc_by_app_policy"][app][pol]
                if abs(got - expected) > EPS:
                    mism.append((fam, app, pol, got, expected))
    assert not mism, mism[:5]


def test_auc_by_app_policy_values_nonnegative(fpac):
    bad = []
    for fam in fpac["meta"]["qualifying_families"]:
        info = fpac["per_family"][fam]
        for app, by_pol in info["auc_by_app_policy"].items():
            for pol, v in by_pol.items():
                if v < -EPS:
                    bad.append((fam, app, pol, v))
    assert not bad, bad


def test_n_graphs_matches_oracle_gap_count(fpac, og):
    """n_graphs counts distinct graphs in the family appearing in paper-L3 rows."""
    by_fam = defaultdict(set)
    for r in og["rows"]:
        if r["l3_size"] in PAPER_L3_SIZES:
            by_fam[r["family"]].add(r["graph"])
    mism = []
    for fam, info in fpac["per_family"].items():
        expected = len(by_fam.get(fam, set()))
        if info["n_graphs"] != expected:
            mism.append((fam, info["n_graphs"], expected))
    assert not mism, mism


# ─── Group C — Winner reproduction ───────────────────────────────────


def test_winner_by_app_is_argmin_auc(fpac):
    mism = []
    for fam in fpac["meta"]["qualifying_families"]:
        info = fpac["per_family"][fam]
        for app, by_pol in info["auc_by_app_policy"].items():
            expected = min(by_pol.items(), key=lambda kv: (kv[1], kv[0]))[0]
            got = info["winner_by_app"][app]
            if got != expected:
                mism.append((fam, app, got, expected))
    assert not mism, mism


def test_winner_matches_global_flags(fpac):
    mism = []
    for fam in fpac["meta"]["qualifying_families"]:
        info = fpac["per_family"][fam]
        for app, got in info["winner_matches_global"].items():
            expected = info["winner_by_app"][app] == GLOBAL_WINNER[app]
            if got != expected:
                mism.append((fam, app, got, expected))
    assert not mism, mism


def test_winners_matching_count_reproduces(fpac):
    mism = []
    for fam in fpac["meta"]["qualifying_families"]:
        info = fpac["per_family"][fam]
        expected = sum(1 for v in info["winner_matches_global"].values() if v)
        if info["winners_matching"] != expected:
            mism.append((fam, info["winners_matching"], expected))
    assert not mism, mism


def test_n_apps_matches_meta(fpac):
    bad = []
    for fam in fpac["meta"]["qualifying_families"]:
        info = fpac["per_family"][fam]
        if info["n_apps"] != len(fpac["meta"]["apps"]):
            bad.append((fam, info["n_apps"]))
    assert not bad, bad


# ─── Group D — Correlation matrix ────────────────────────────────────


def test_correlation_matrix_reproduces_pearson_on_zscore(fpac):
    apps = sorted(fpac["meta"]["apps"])
    policies = sorted(fpac["meta"]["policies"])
    mism = []
    for fam in fpac["meta"]["qualifying_families"]:
        info = fpac["per_family"][fam]
        per_app_z = {
            app: _zscore([info["auc_by_app_policy"][app][pol] for pol in policies])
            for app in apps
        }
        for a in apps:
            for b in apps:
                expected = round(_pearson(per_app_z[a], per_app_z[b]), 4)
                if abs(info["correlation_matrix"][a][b] - expected) > EPS:
                    mism.append((fam, a, b, info["correlation_matrix"][a][b], expected))
    assert not mism, mism[:5]


def test_correlation_matrix_symmetric_and_unit_diagonal(fpac):
    apps = sorted(fpac["meta"]["apps"])
    bad = []
    for fam in fpac["meta"]["qualifying_families"]:
        m = fpac["per_family"][fam]["correlation_matrix"]
        for a in apps:
            if abs(m[a][a] - 1.0) > EPS:
                bad.append((fam, "diag", a, m[a][a]))
            for b in apps:
                if abs(m[a][b] - m[b][a]) > EPS:
                    bad.append((fam, "sym", a, b))
    assert not bad, bad[:5]


def test_correlation_matrix_invariant_under_raw_auc(fpac):
    """Pearson invariance: corr on raw AUC equals corr on z-scored AUC."""
    apps = sorted(fpac["meta"]["apps"])
    policies = sorted(fpac["meta"]["policies"])
    mism = []
    for fam in fpac["meta"]["qualifying_families"]:
        info = fpac["per_family"][fam]
        per_app_raw = {
            app: [info["auc_by_app_policy"][app][pol] for pol in policies]
            for app in apps
        }
        for a in apps:
            for b in apps:
                expected = round(_pearson(per_app_raw[a], per_app_raw[b]), 4)
                if abs(info["correlation_matrix"][a][b] - expected) > EPS:
                    mism.append((fam, a, b))
    assert not mism, mism[:5]


# ─── Group E — intra/inter cluster ──────────────────────────────────


def test_intra_cluster_mean_r_reproduces(fpac):
    apps = sorted(fpac["meta"]["apps"])
    cluster_of = {
        app: pol for pol, members in GLOBAL_CLUSTERS.items() for app in members
    }
    mism = []
    for fam in fpac["meta"]["qualifying_families"]:
        info = fpac["per_family"][fam]
        m = info["correlation_matrix"]
        intra_means = []
        for app in apps:
            own = cluster_of[app]
            intra_vals = [
                m[app][b] for b in apps if b != app and cluster_of[b] == own
            ]
            if intra_vals:
                intra_means.append(statistics.fmean(intra_vals))
        if intra_means:
            expected = round(statistics.fmean(intra_means), 4)
            if abs(info["intra_cluster_mean_r"] - expected) > EPS:
                mism.append((fam, info["intra_cluster_mean_r"], expected))
    assert not mism, mism


def test_inter_cluster_mean_r_reproduces(fpac):
    apps = sorted(fpac["meta"]["apps"])
    cluster_of = {
        app: pol for pol, members in GLOBAL_CLUSTERS.items() for app in members
    }
    mism = []
    for fam in fpac["meta"]["qualifying_families"]:
        info = fpac["per_family"][fam]
        m = info["correlation_matrix"]
        inter_means = []
        for app in apps:
            own = cluster_of[app]
            inter_vals = [
                m[app][b] for b in apps if b != app and cluster_of[b] != own
            ]
            if inter_vals:
                inter_means.append(statistics.fmean(inter_vals))
        if inter_means:
            expected = round(statistics.fmean(inter_means), 4)
            if abs(info["inter_cluster_mean_r"] - expected) > EPS:
                mism.append((fam, info["inter_cluster_mean_r"], expected))
    assert not mism, mism


def test_intra_minus_inter_and_dominates_flag(fpac):
    mism = []
    for fam in fpac["meta"]["qualifying_families"]:
        info = fpac["per_family"][fam]
        intra = info["intra_cluster_mean_r"]
        inter = info["inter_cluster_mean_r"]
        if intra is None or inter is None:
            continue
        expected_gap = round(intra - inter, 4)
        if abs(info["intra_minus_inter"] - expected_gap) > EPS:
            mism.append((fam, "gap", info["intra_minus_inter"], expected_gap))
        expected_dom = info["intra_minus_inter"] > 0.0
        if info["intra_dominates"] != expected_dom:
            mism.append((fam, "dom", info["intra_dominates"], expected_dom))
    assert not mism, mism
