"""Tests for gate 61 — per-family oracle-gap curvature replay."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "wiki" / "data" / "family_curvature_replay.json"
GENERATOR = (
    REPO_ROOT / "scripts" / "experiments" / "ecg" / "family_curvature_replay.py"
)


def _payload() -> dict:
    if not DATA.exists():
        subprocess.check_call([sys.executable, str(GENERATOR)])
    return json.loads(DATA.read_text())


def test_payload_present_and_well_formed():
    p = _payload()
    assert "meta" in p and "per_family" in p
    assert p["meta"]["policies"] == ["GRASP", "LRU", "POPT", "SRRIP"]
    assert set(p["meta"]["oracle_aware_policies"]) == {"GRASP", "POPT"}
    assert set(p["meta"]["non_oracle_policies"]) == {"LRU", "SRRIP"}


def test_qualifying_families_include_known_set():
    p = _payload()
    # At minimum, citation, social, web — the three with full 1MB/4MB/8MB
    # coverage. mesh and road only have sub-MB sizes.
    assert set(p["meta"]["qualifying_families"]) >= {"citation", "social", "web"}


def test_every_qualifying_family_has_all_four_policies():
    p = _payload()
    for fam in p["meta"]["qualifying_families"]:
        per_pol = p["per_family"][fam]["per_policy"]
        assert set(per_pol.keys()) == {"GRASP", "LRU", "POPT", "SRRIP"}


def test_no_new_deviating_family_beyond_pin():
    p = _payload()
    assert not p["meta"]["new_deviating_families"], (
        "New family deviates from oracle-aware curvature pattern: "
        f"{p['meta']['new_deviating_families']}"
    )


def test_at_least_one_family_replays():
    p = _payload()
    assert p["meta"]["replay_count"] >= 1


def test_verdict_is_pass():
    p = _payload()
    assert p["meta"]["verdict"] == "PASS"


def test_all_three_known_families_currently_replay():
    # citation, social, web are the three families with full L3 cov; all
    # three should currently replay the global GRASP-positive,
    # LRU/SRRIP-non-positive curvature pattern.
    p = _payload()
    for fam in ("citation", "social", "web"):
        info = p["per_family"][fam]
        assert info["replays_pattern"], (
            f"family {fam} no longer replays the global pattern: "
            f"per_policy={info['per_policy']}"
        )


def test_grasp_curvature_positive_in_social_and_web():
    # The two families with the cleanest oracle-aware signal:
    # GRASP curvature should be strictly positive.
    p = _payload()
    for fam in ("social", "web"):
        c = p["per_family"][fam]["per_policy"]["GRASP"]["mean_curvature"]
        assert c > 0, (
            f"GRASP curvature must be positive in {fam}; got {c}"
        )


def test_non_oracle_curvature_non_positive_everywhere():
    p = _payload()
    for fam in p["meta"]["qualifying_families"]:
        for pol in ("LRU", "SRRIP"):
            c = p["per_family"][fam]["per_policy"][pol]["mean_curvature"]
            assert c <= 0, (
                f"{pol} curvature must be <= 0 in family {fam}; got {c}"
            )


def test_lru_curvature_more_negative_than_grasp_everywhere():
    # The whole point: LRU is still accelerating its descent harder than
    # GRASP across every family.
    p = _payload()
    for fam in p["meta"]["qualifying_families"]:
        lru = p["per_family"][fam]["per_policy"]["LRU"]["mean_curvature"]
        grasp = p["per_family"][fam]["per_policy"]["GRASP"]["mean_curvature"]
        assert lru < grasp, (
            f"family {fam}: LRU curvature ({lru}) must be < GRASP "
            f"curvature ({grasp})"
        )


def test_sample_counts_recorded():
    p = _payload()
    for fam in p["meta"]["qualifying_families"]:
        for pol in ("GRASP", "LRU", "POPT", "SRRIP"):
            n = p["per_family"][fam]["per_policy"][pol]["n_app_graph_cells"]
            assert n >= 1, (
                f"{fam}/{pol} must have at least one (app, graph) cell"
            )


def test_curvature_threshold_is_zero_for_sign_test():
    p = _payload()
    assert p["meta"]["curvature_threshold_pp_oct2"] == 0.0
