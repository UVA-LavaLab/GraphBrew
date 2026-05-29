"""Gate 50: cross-app policy-AUC correlation matrix + cluster sanity.

Pins:
  - exact 5-app inventory + 4-policy AUC vector shape
  - 2-cluster AUC-winner partition (GRASP=[bc,cc], POPT=[bfs,pr,sssp])
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
    """Pin the 2-cluster partition: GRASP wins for bc+cc; POPT for bfs+pr+sssp.
    If a new gate49 result flips a winner, this gate fires."""
    assert payload["meta"]["clusters_by_winner"] == {
        "GRASP": ["bc", "cc"],
        "POPT": ["bfs", "pr", "sssp"],
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


def test_intra_beats_inter_for_at_least_4_of_5(payload):
    """The paper claim 'AUC clustering is structural, not noise' requires
    intra-cluster correlation > inter-cluster correlation for the
    majority of apps. We pin at least 4 of 5."""
    wins = 0
    for app, ii in payload["intra_inter"].items():
        gap = ii["gap_intra_minus_inter"]
        if gap is not None and gap > 0:
            wins += 1
    assert wins >= 4, f"only {wins}/5 apps have positive intra-inter gap"


def test_bfs_sssp_is_strongest_pair(payload):
    """Empirical headline: bfs and sssp (both POPT-winners, both
    frontier-bound traversals) have the strongest policy-ranking
    similarity. Pin r > 0.80 and as rank #1."""
    pair = payload["pair_list"][0]
    assert {pair["app_a"], pair["app_b"]} == {"bfs", "sssp"}
    assert pair["pearson_r"] > 0.80


def test_bc_cc_is_in_top_3(payload):
    """bc and cc share GRASP-winner status; their pair must be in the
    top 3 most-similar pairs."""
    top3 = payload["pair_list"][:3]
    pairs = [{p["app_a"], p["app_b"]} for p in top3]
    assert {"bc", "cc"} in pairs, f"bc+cc not in top 3: {pairs}"


def test_no_pair_has_strong_anti_correlation(payload):
    """If any pair has r < -0.50 that'd suggest the policy ordering
    inverts between two apps — paper text would need to address it.
    Currently the most negative pair is cc+sssp at -0.350. Pin a -0.50
    floor so this test fires if a future run produces real anti-corr."""
    for pair in payload["pair_list"]:
        assert pair["pearson_r"] >= -0.50, (
            f"pair {pair['app_a']}+{pair['app_b']} has r={pair['pearson_r']}"
        )


def test_pair_list_is_sorted_descending(payload):
    rs = [pair["pearson_r"] for pair in payload["pair_list"]]
    assert rs == sorted(rs, reverse=True)


def test_nearest_sibling_for_grasp_winners_is_other_grasp_winner(payload):
    """bc's nearest sibling should be cc (its only other GRASP-winner)
    and cc's nearest sibling should be bc. Pins the cluster structure
    from a per-app point of view."""
    ns = payload["nearest_sibling"]
    assert ns["bc"]["closest_app"] == "cc"
    assert ns["cc"]["closest_app"] == "bc"


def test_cross_gate_consistency_with_oracle_gap_auc(payload):
    """Winners declared here must match gate 49 (oracle_gap_auc) exactly."""
    auc_path = REPO_ROOT / "wiki" / "data" / "oracle_gap_auc.json"
    if not auc_path.exists():
        pytest.skip("oracle_gap_auc.json not built (gate 49)")
    gate49 = json.loads(auc_path.read_text())
    assert payload["meta"]["auc_winner_by_app"] == gate49["meta"]["auc_winner_by_app"]
