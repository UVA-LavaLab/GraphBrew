"""Derivation parity gate (WKL-Der) for `wss_knee_location.json`.

The generator at ``scripts/experiments/ecg/wss_knee_location.py`` walks
the regime ladder ``(under_wss, near_wss, over_wss)`` per policy and
records the first regime whose median_gap_pp drops at-or-below
``KNEE_THRESHOLD_PP = 0.5``. The verdict invariant — oracle-aware
policies (GRASP, POPT) must plateau STRICTLY earlier than non-oracle
policies (LRU, SRRIP) — is the headline result of gate 60 in the
running narrative.

This parity gate locks every load-bearing rule:

* Group A — pinned constants (REGIME_LADDER tuple, POLICIES tuple,
  ORACLE_AWARE/NON_ORACLE sets, KNEE_THRESHOLD_PP).
* Group B — _find_knee_regime walk semantics (first-match left→right,
  INCLUSIVE ≤ threshold, sentinel rank when policy never plateaus).
* Group C — per-policy block construction (per_regime entry shape,
  knee_regime + knee_rank pair, is_oracle_aware membership).
* Group D — verdict block (max/min aggregator, STRICT < invariant,
  knee_rank/regime_by_policy projection blocks).
* Group E — byte parity with the committed JSON.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN_PATH = REPO_ROOT / "scripts" / "experiments" / "ecg" / "wss_knee_location.py"
JSON_PATH = REPO_ROOT / "wiki" / "data" / "wss_knee_location.json"
WSS_PATH = REPO_ROOT / "wiki" / "data" / "wss_relative_l3.json"


def _load_gen():
    spec = importlib.util.spec_from_file_location("wss_knee_location", GEN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen():
    return _load_gen()


@pytest.fixture(scope="module")
def artifact():
    return json.loads(JSON_PATH.read_text())


@pytest.fixture(scope="module")
def wss():
    return json.loads(WSS_PATH.read_text())


# ---------------------------------------------------------------- Group A
def test_regime_ladder_pinned(gen):
    assert gen.REGIME_LADDER == ("under_wss", "near_wss", "over_wss")
    assert isinstance(gen.REGIME_LADDER, tuple)


def test_policies_tuple_pinned(gen):
    assert gen.POLICIES == ("GRASP", "LRU", "POPT", "SRRIP")
    assert isinstance(gen.POLICIES, tuple)


def test_oracle_aware_membership_pinned(gen):
    assert gen.ORACLE_AWARE == {"GRASP", "POPT"}
    assert gen.NON_ORACLE == {"LRU", "SRRIP"}
    assert gen.ORACLE_AWARE.isdisjoint(gen.NON_ORACLE)
    assert gen.ORACLE_AWARE | gen.NON_ORACLE == set(gen.POLICIES)


def test_knee_threshold_pinned(gen):
    assert gen.KNEE_THRESHOLD_PP == 0.5


def test_meta_constants_mirror_pinned(gen, artifact):
    m = artifact["meta"]
    assert m["knee_threshold_pp"] == gen.KNEE_THRESHOLD_PP
    assert m["regime_ladder"] == list(gen.REGIME_LADDER)
    assert m["policies"] == list(gen.POLICIES)
    assert m["oracle_aware_policies"] == sorted(gen.ORACLE_AWARE)
    assert m["non_oracle_policies"] == sorted(gen.NON_ORACLE)


def test_top_level_keys(artifact):
    assert set(artifact.keys()) == {"meta", "per_policy"}


# ---------------------------------------------------------------- Group B
def test_find_knee_returns_first_at_or_below_threshold(gen):
    per = {
        "under_wss": {"median_gap_pp": 5.0},
        "near_wss": {"median_gap_pp": 0.5},   # at threshold (INCLUSIVE)
        "over_wss": {"median_gap_pp": 0.0},
    }
    regime, rank = gen._find_knee_regime(per, 0.5)
    assert (regime, rank) == ("near_wss", 1)


def test_find_knee_threshold_is_inclusive(gen):
    per = {"under_wss": {"median_gap_pp": 0.5}}
    regime, rank = gen._find_knee_regime(per, 0.5)
    assert (regime, rank) == ("under_wss", 0)


def test_find_knee_strict_greater_than_threshold_not_a_knee(gen):
    per = {"under_wss": {"median_gap_pp": 0.500001},
           "near_wss": {"median_gap_pp": 0.4}}
    regime, rank = gen._find_knee_regime(per, 0.5)
    assert (regime, rank) == ("near_wss", 1)


def test_find_knee_sentinel_when_no_regime_plateaus(gen):
    per = {
        "under_wss": {"median_gap_pp": 5.0},
        "near_wss": {"median_gap_pp": 4.0},
        "over_wss": {"median_gap_pp": 3.0},
    }
    regime, rank = gen._find_knee_regime(per, 0.5)
    assert regime is None
    assert rank == len(gen.REGIME_LADDER) == 3


def test_find_knee_skips_missing_regimes(gen):
    per = {"over_wss": {"median_gap_pp": 0.1}}
    regime, rank = gen._find_knee_regime(per, 0.5)
    assert (regime, rank) == ("over_wss", 2)


def test_find_knee_walks_in_ladder_order(gen):
    """First match returned even when later regimes also plateau."""
    per = {
        "under_wss": {"median_gap_pp": 0.1},
        "near_wss": {"median_gap_pp": 0.0},
        "over_wss": {"median_gap_pp": 0.0},
    }
    regime, rank = gen._find_knee_regime(per, 0.5)
    assert (regime, rank) == ("under_wss", 0)


# ---------------------------------------------------------------- Group C
def test_per_policy_keys_match_policies_tuple(gen, artifact):
    assert set(artifact["per_policy"].keys()) == set(gen.POLICIES)


def test_per_policy_record_keys(artifact):
    for pol, info in artifact["per_policy"].items():
        assert set(info.keys()) == {"per_regime", "knee_regime", "knee_rank", "is_oracle_aware"}


def test_per_policy_is_oracle_aware_matches_membership(gen, artifact):
    for pol, info in artifact["per_policy"].items():
        assert info["is_oracle_aware"] == (pol in gen.ORACLE_AWARE)


def test_per_regime_entry_shape(artifact):
    for pol, info in artifact["per_policy"].items():
        for reg, cell in info["per_regime"].items():
            assert set(cell.keys()) == {"n", "mean_gap_pp", "median_gap_pp", "p90_gap_pp", "win_rate"}


def test_per_regime_mirrors_wss_upstream(artifact, wss):
    """Each per_regime cell carries the exact 5-field projection from
    wss_relative_l3.by_policy_regime[policy/regime]."""
    by_pr = wss["by_policy_regime"]
    for pol, info in artifact["per_policy"].items():
        for reg, cell in info["per_regime"].items():
            upstream = by_pr[f"{pol}/{reg}"]
            for k in ("n", "mean_gap_pp", "median_gap_pp", "p90_gap_pp", "win_rate"):
                assert cell[k] == upstream[k]


def test_per_regime_only_existing_upstream_keys(artifact, wss):
    """Cells absent in upstream are not fabricated."""
    by_pr = wss["by_policy_regime"]
    for pol, info in artifact["per_policy"].items():
        for reg in ("under_wss", "near_wss", "over_wss"):
            present_upstream = f"{pol}/{reg}" in by_pr
            present_artifact = reg in info["per_regime"]
            assert present_artifact == present_upstream


def test_per_policy_knee_rank_matches_find_knee(artifact, gen):
    for pol, info in artifact["per_policy"].items():
        per_regime = info["per_regime"]
        expected_regime, expected_rank = gen._find_knee_regime(per_regime, gen.KNEE_THRESHOLD_PP)
        assert info["knee_regime"] == expected_regime
        assert info["knee_rank"] == expected_rank


# ---------------------------------------------------------------- Group D
def test_knee_rank_by_policy_keys_match_policies(gen, artifact):
    assert set(artifact["meta"]["knee_rank_by_policy"].keys()) == set(gen.POLICIES)
    assert set(artifact["meta"]["knee_regime_by_policy"].keys()) == set(gen.POLICIES)


def test_knee_rank_by_policy_mirrors_per_policy(artifact):
    for pol, rank in artifact["meta"]["knee_rank_by_policy"].items():
        assert rank == artifact["per_policy"][pol]["knee_rank"]


def test_knee_regime_by_policy_mirrors_per_policy(artifact):
    for pol, reg in artifact["meta"]["knee_regime_by_policy"].items():
        assert reg == artifact["per_policy"][pol]["knee_regime"]


def test_max_oracle_aware_knee_rank_matches_max(artifact, gen):
    oracle_ranks = [artifact["per_policy"][p]["knee_rank"] for p in gen.ORACLE_AWARE]
    assert artifact["meta"]["max_oracle_aware_knee_rank"] == max(oracle_ranks)


def test_min_non_oracle_knee_rank_matches_min(artifact, gen):
    non_ranks = [artifact["per_policy"][p]["knee_rank"] for p in gen.NON_ORACLE]
    assert artifact["meta"]["min_non_oracle_knee_rank"] == min(non_ranks)


def test_verdict_pass_iff_strict_lt_invariant(artifact):
    expected = (
        "PASS"
        if artifact["meta"]["max_oracle_aware_knee_rank"]
            < artifact["meta"]["min_non_oracle_knee_rank"]
        else "FAIL"
    )
    assert artifact["meta"]["verdict"] == expected


def test_verdict_invariant_text_pinned(artifact):
    assert artifact["meta"]["verdict_invariant"] == (
        "PASS iff max(knee_rank of oracle-aware policies) "
        "< min(knee_rank of non-oracle policies)"
    )


def test_verdict_strict_lt_polarity_via_build_fixture(gen):
    """If max_oracle == min_non_oracle, verdict must FAIL (STRICT <)."""
    fake = {
        "by_policy_regime": {
            "GRASP/under_wss": {"n": 1, "mean_gap_pp": 0, "median_gap_pp": 5,
                                "p90_gap_pp": 0, "win_rate": 0},
            "GRASP/near_wss": {"n": 1, "mean_gap_pp": 0, "median_gap_pp": 0.1,
                               "p90_gap_pp": 0, "win_rate": 0},
            "POPT/under_wss": {"n": 1, "mean_gap_pp": 0, "median_gap_pp": 0.1,
                               "p90_gap_pp": 0, "win_rate": 0},
            "LRU/under_wss": {"n": 1, "mean_gap_pp": 0, "median_gap_pp": 5,
                              "p90_gap_pp": 0, "win_rate": 0},
            "LRU/near_wss": {"n": 1, "mean_gap_pp": 0, "median_gap_pp": 0.1,
                             "p90_gap_pp": 0, "win_rate": 0},  # rank 1 — ties GRASP
            "SRRIP/under_wss": {"n": 1, "mean_gap_pp": 0, "median_gap_pp": 5,
                                "p90_gap_pp": 0, "win_rate": 0},
            "SRRIP/over_wss": {"n": 1, "mean_gap_pp": 0, "median_gap_pp": 0.0,
                               "p90_gap_pp": 0, "win_rate": 0},
        }
    }
    result = gen.build(fake)
    # max oracle = 1 (GRASP=1, POPT=0); min non-oracle = 1 (LRU=1, SRRIP=2)
    assert result["meta"]["max_oracle_aware_knee_rank"] == 1
    assert result["meta"]["min_non_oracle_knee_rank"] == 1
    assert result["meta"]["verdict"] == "FAIL"


# ---------------------------------------------------------------- Group E
def test_full_artifact_byte_parity(tmp_path):
    out_json = tmp_path / "wss_knee_location.json"
    out_md = tmp_path / "wss_knee_location.md"
    res = subprocess.run(
        [
            sys.executable,
            str(GEN_PATH),
            "--wss-json", str(WSS_PATH),
            "--json-out", str(out_json),
            "--md-out", str(out_md),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "verdict=PASS" in res.stdout
    assert out_json.read_text() == JSON_PATH.read_text()
