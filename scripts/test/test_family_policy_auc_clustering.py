"""Tests for the per-family policy-AUC clustering replay (gate 57)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "wiki" / "data" / "family_policy_auc_clustering.json"
GENERATOR = (
    REPO_ROOT
    / "scripts"
    / "experiments"
    / "ecg"
    / "family_policy_auc_clustering.py"
)


def _payload() -> dict:
    if not DATA.exists():
        subprocess.check_call([sys.executable, str(GENERATOR)])
    return json.loads(DATA.read_text())


def test_payload_present_and_well_formed():
    p = _payload()
    assert "meta" in p
    assert "per_family" in p
    m = p["meta"]
    assert set(m["apps"]) == {"bc", "bfs", "cc", "pr", "sssp"}
    assert set(m["policies"]) == {"GRASP", "LRU", "POPT", "SRRIP"}
    assert set(m["families"]) >= {"citation", "mesh", "road", "social", "web"}


def test_qualifying_families_have_full_l3_coverage():
    p = _payload()
    qualifying = p["meta"]["qualifying_families"]
    # Today: citation, social, web (each has cit-Patents / 4 social /
    # web-Google with full 1MB+4MB+8MB coverage).
    assert {"social"} <= set(qualifying), (
        f"social family must always qualify; got {qualifying}"
    )
    assert len(qualifying) >= 3, (
        f"expected >=3 qualifying families; got {len(qualifying)} = {qualifying}"
    )
    for fam in qualifying:
        info = p["per_family"][fam]
        assert info["qualified"] is True
        assert "auc_by_app_policy" in info
        assert info["n_apps"] == 5


def test_intra_cluster_dominates_inter_in_every_qualifying_family():
    p = _payload()
    for fam in p["meta"]["qualifying_families"]:
        info = p["per_family"][fam]
        assert info["intra_minus_inter"] > 0.0, (
            f"family {fam} fails intra > inter:"
            f" intra={info['intra_cluster_mean_r']},"
            f" inter={info['inter_cluster_mean_r']}"
        )
        assert info["intra_dominates"]


def test_social_family_replays_global_winners_perfectly():
    # social is the strongest replay because it has the largest n_graphs
    # (com-orkut, email-Eu-core, soc-LiveJournal1, soc-pokec).
    p = _payload()
    info = p["per_family"]["social"]
    assert info["qualified"]
    assert info["n_graphs"] >= 4
    assert info["winners_matching"] == 5, (
        f"social family winners do not match global: {info['winner_by_app']}"
    )
    # Mechanism: social graphs are POPT/GRASP-distinguishable.
    assert info["intra_minus_inter"] > 0.25


def test_no_new_deviations_vs_pin():
    p = _payload()
    dev = p["meta"]["deviation_set"]
    assert dev["new_vs_pin"] == [], (
        f"NEW (family, app) deviations from global winner: {dev['new_vs_pin']}."
        " Update PINNED_DEVIATIONS only after manual review."
    )


def test_observed_deviations_inside_pinned_cap():
    p = _payload()
    dev = p["meta"]["deviation_set"]
    assert len(dev["observed"]) <= dev["pinned_max"]


def test_pinned_deviations_concentrate_in_citation_family():
    # All current pin members live in citation (single-graph family).
    p = _payload()
    for d in p["meta"]["deviation_set"]["pinned"]:
        assert d["family"] == "citation", (
            f"pinned deviation {d} unexpectedly outside citation;"
            " rationale only covers cit-Patents's lower out-degree skew"
        )


def test_cluster_invariance_verdict_is_pass():
    p = _payload()
    assert p["meta"]["cluster_invariance_verdict"] == "PASS"


def test_correlation_matrix_diagonal_is_one_per_family():
    p = _payload()
    for fam in p["meta"]["qualifying_families"]:
        info = p["per_family"][fam]
        for app in info["correlation_matrix"]:
            assert info["correlation_matrix"][app][app] == 1.0


def test_correlation_matrix_is_symmetric_per_family():
    p = _payload()
    for fam in p["meta"]["qualifying_families"]:
        info = p["per_family"][fam]
        apps = list(info["correlation_matrix"].keys())
        for a in apps:
            for b in apps:
                if a == b:
                    continue
                assert (
                    abs(
                        info["correlation_matrix"][a][b]
                        - info["correlation_matrix"][b][a]
                    )
                    < 1e-6
                ), f"{fam}/{a},{b} correlation is asymmetric"


def test_cross_gate_50_winners_match_global():
    # Sanity: the global winner map embedded here matches gate 50's
    # auc_winner_by_app exactly. Drift would mean someone changed gate
    # 50 without re-pinning gate 57.
    g50_path = REPO_ROOT / "wiki" / "data" / "policy_auc_correlation.json"
    if not g50_path.exists():
        import pytest

        pytest.skip(
            "policy_auc_correlation.json missing; run gate 50 first"
        )
    g50 = json.loads(g50_path.read_text())
    p = _payload()
    assert (
        g50["meta"]["auc_winner_by_app"]
        == p["meta"]["global_winner_by_app"]
    ), (
        "gate 50 auc_winner_by_app diverges from gate 57's GLOBAL_WINNER;"
        " re-pin gate 57 to match"
    )


def test_unqualified_families_carry_reason():
    p = _payload()
    qualifying = set(p["meta"]["qualifying_families"])
    for fam, info in p["per_family"].items():
        if fam in qualifying:
            continue
        assert info["qualified"] is False
        assert "reason" in info
