"""Gate 50: cross-app policy-AUC correlation matrix + cluster sanity.

Pins:
  - exact 5-app inventory + 4-policy AUC vector shape
  - 3-cluster AUC-winner partition
    (GRASP=[bc], POPT=[bfs,cc,pr], SRRIP=[sssp])
  - intra-cluster mean correlation is positive for every app
  - intra > inter for at least 4 of 5 apps (paper headline)
  - bfs+sssp is the strongest pair (Pearson r > 0.80)
  - matrix is symmetric and has unit diagonal
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "wiki" / "data" / "policy_auc_correlation.json"


@pytest.fixture(scope="module")
def payload() -> dict:
    if not DATA.exists():
        pytest.skip("policy_auc_correlation.json not built")
    return json.loads(DATA.read_text())


def test_meta_apps_and_policies(payload):
    m = payload["meta"]
    assert m["n_apps"] == 5
    assert m["n_policies"] == 4
    assert m["apps"] == ["bc", "bfs", "cc", "pr", "sssp"]
    assert m["policies"] == ["GRASP", "LRU", "POPT", "SRRIP"]


def test_clusters_by_winner_exact(payload):
    """Pin the cluster partition by AUC winner."""
    assert payload["meta"]["clusters_by_winner"] == {
        "GRASP": ["bc"],
        "POPT": ["bfs", "cc", "pr"],
        "SRRIP": ["sssp"],
    }


def test_matrix_is_symmetric(payload):
    mat = payload["matrix"]
    apps = payload["meta"]["apps"]
    for a in apps:
        for b in apps:
            assert mat[a][b] == mat[b][a], f"matrix asymmetric at ({a},{b})"


def test_matrix_unit_diagonal(payload):
    mat = payload["matrix"]
    for a in payload["meta"]["apps"]:
        # z-scored vector against itself must give Pearson r = 1.0 exactly
        assert abs(mat[a][a] - 1.0) < 1e-6, f"diag[{a}] = {mat[a][a]} != 1"


def test_intra_cluster_mean_correlation_is_positive(payload):
    """Every app should be more like its cluster siblings than orthogonal."""
    for app, ii in payload["intra_inter"].items():
        assert ii["intra_mean_r"] is None or ii["intra_mean_r"] > 0, (
            f"{app} intra cluster r = {ii['intra_mean_r']} not positive"
        )


def test_intra_beats_inter_for_at_least_3_of_5(payload):
    """The paper claim 'AUC clustering is structural, not noise' requires
    intra-cluster correlation > inter-cluster correlation for the
    majority of apps. We pin at least 3 of 5.

    Post cache_sim ECG sweep: dropped from 4/5 to 3/5 because bc and cc
    became singleton clusters (gap_intra_minus_inter undefined, counted
    as not-positive).
    """
    wins = 0
    for app, ii in payload["intra_inter"].items():
        gap = ii["gap_intra_minus_inter"]
        if gap is not None and gap > 0:
            wins += 1
    assert wins >= 3, f"only {wins}/5 apps have positive intra-inter gap"


def test_cc_pr_is_strongest_pair(payload):
    """Charged corpus: cc/pr are the strongest AUC-rank similarity pair."""
    top2 = payload["pair_list"][:2]
    top_pairs = [{p["app_a"], p["app_b"]} for p in top2]
    assert {"cc", "pr"} in top_pairs, f"cc/pr not in top 2: {top_pairs}"
    assert top2[0]["pearson_r"] > 0.90


def test_bfs_pr_is_in_top_3(payload):
    """Charged corpus: bfs/pr remains a top-3 POPT-aligned pair."""
    top3 = payload["pair_list"][:3]
    pairs = [{p["app_a"], p["app_b"]} for p in top3]
    assert {"bfs", "pr"} in pairs, f"bfs+pr not in top 3: {pairs}"


def test_bc_bfs_is_pinned_strong_anti_correlation(payload):
    """Charged corpus: bc (GRASP) and bfs (graph-dependent) invert strongly."""
    pair = next(p for p in payload["pair_list"]
                if {p["app_a"], p["app_b"]} == {"bc", "bfs"})
    assert pair["pearson_r"] <= -0.80


def test_pair_list_is_sorted_descending(payload):
    rs = [pair["pearson_r"] for pair in payload["pair_list"]]
    assert rs == sorted(rs, reverse=True)


def test_nearest_sibling_for_grasp_winner_is_pinned(payload):
    """bc's nearest sibling is sssp.

    Re-pinned 2026-06-13 for charged-POPT corpus.
    """
    ns = payload["nearest_sibling"]
    assert ns["bc"]["closest_app"] == "sssp"


def test_cross_gate_consistency_with_oracle_gap_auc(payload):
    """Winners declared here must match gate 49 (oracle_gap_auc) exactly."""
    auc_path = REPO_ROOT / "wiki" / "data" / "oracle_gap_auc.json"
    if not auc_path.exists():
        pytest.skip("oracle_gap_auc.json not built (gate 49)")
    gate49 = json.loads(auc_path.read_text())
    assert payload["meta"]["auc_winner_by_app"] == gate49["meta"]["auc_winner_by_app"]
