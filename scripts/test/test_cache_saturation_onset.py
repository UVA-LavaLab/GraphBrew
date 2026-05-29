"""Gate 55: cache-saturation onset detection invariants.

For each (app, policy) trajectory, identifies the L3 size beyond
which extra cache buys negligible gap improvement. Paper-grade
mechanism story: POPT and GRASP saturate earlier than LRU and SRRIP
because they are already close to oracle.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = REPO_ROOT / "wiki" / "data" / "cache_saturation_onset.json"

EXPECTED_APPS = {"bc", "bfs", "cc", "pr", "sssp"}
EXPECTED_POLICIES = {"LRU", "SRRIP", "GRASP", "POPT"}


@pytest.fixture(scope="module")
def payload():
    if not PAYLOAD.exists():
        pytest.skip(f"missing {PAYLOAD}; run `make lit-cache-saturation-onset`")
    return json.loads(PAYLOAD.read_text())


def test_meta_paper_scope(payload):
    assert payload["meta"]["scope_l3_sizes"] == ["1MB", "4MB", "8MB"]


def test_meta_threshold_reasonable(payload):
    """Threshold should be ~0.5 pp/octave (literature-grade noise floor)."""
    t = payload["meta"]["saturation_threshold_pp_per_octave"]
    assert 0.1 <= t <= 1.0


def test_all_apps_and_policies_present(payload):
    assert set(payload["meta"]["apps"]) == EXPECTED_APPS
    assert set(payload["meta"]["policies"]) == EXPECTED_POLICIES
    for app in EXPECTED_APPS:
        assert set(payload["per_app"][app].keys()) == EXPECTED_POLICIES


def test_popt_saturates_earliest_overall(payload):
    """POPT should be ranked first or tied-first for saturation."""
    rank = payload["meta"]["saturation_rank_by_policy"]
    assert rank[0] == "POPT", f"saturation rank should start with POPT; got {rank}"


def test_oracle_aware_outsaturates_lru(payload):
    """POPT + GRASP combined should have more saturated cells than LRU + SRRIP."""
    pv = payload["per_policy"]
    oracle = pv["POPT"]["n_saturated"] + pv["GRASP"]["n_saturated"]
    non_oracle = pv["LRU"]["n_saturated"] + pv["SRRIP"]["n_saturated"]
    assert oracle > non_oracle, (
        f"oracle-aware should out-saturate non-oracle; "
        f"oracle={oracle}, non_oracle={non_oracle}"
    )


def test_popt_saturates_majority_of_apps(payload):
    """POPT should saturate at paper L3 on >=3 of 5 apps."""
    n = payload["per_policy"]["POPT"]["n_saturated"]
    assert n >= 3, f"POPT only saturated on {n} apps; expected >=3"


def test_lru_rarely_saturates(payload):
    """LRU should saturate on at most 2 apps (it almost always benefits from more cache)."""
    n = payload["per_policy"]["LRU"]["n_saturated"]
    assert n <= 2, f"LRU saturated on {n} apps; expected <=2"


def test_pr_popt_saturates_early(payload):
    """pr/POPT is the canonical 'near-oracle from 1MB' cell.

    Should saturate at 4MB or 1MB (final octave slope effectively 0).
    """
    blob = payload["per_app"]["pr"]["POPT"]
    assert blob["saturation_onset"] in {"1MB", "4MB"}, (
        f"pr/POPT onset should be early; got {blob['saturation_onset']}"
    )
    assert abs(blob["final_octave_slope_pp"]) < 0.5


def test_pr_lru_never_saturates(payload):
    """pr/LRU should still be benefiting strongly from cache at 8MB."""
    blob = payload["per_app"]["pr"]["LRU"]
    assert blob["saturation_onset"] == "never"
    # final slope should be > 1.0 pp/octave (still falling steeply)
    assert blob["final_octave_slope_pp"] > 1.0


def test_octaves_well_formed(payload):
    """Schema invariant: each trajectory has at least 1 octave with all fields."""
    for app in EXPECTED_APPS:
        for pol in EXPECTED_POLICIES:
            blob = payload["per_app"][app][pol]
            assert blob["octaves"], f"no octaves for {app}/{pol}"
            for o in blob["octaves"]:
                assert {"from", "to", "gap_from", "gap_to", "delta_gap_pp", "slope_pp_per_octave"} <= o.keys()
                assert o["from"] in {"1MB", "4MB", "8MB"}


def test_cross_consistency_with_slope_gate(payload):
    """Cross-gate anchor with gate 52 (cache_sensitivity_slope).

    The final octave's delta_gap_pp here should match the corresponding
    octave's delta_gap_pp in cache_sensitivity_slope, since both come
    from the same trajectory data.
    """
    sibling = REPO_ROOT / "wiki" / "data" / "cache_sensitivity_slope.json"
    if not sibling.exists():
        pytest.skip("gate 52 sibling missing")
    slope = json.loads(sibling.read_text())
    mismatches = []
    for app in EXPECTED_APPS:
        for pol in EXPECTED_POLICIES:
            here = payload["per_app"][app][pol]["octaves"][-1]
            there = slope["per_app"][app][pol]["octaves"][-1]
            if abs(here["delta_gap_pp"] - there["delta_gap_pp"]) > 1e-3:
                mismatches.append((app, pol, here["delta_gap_pp"], there["delta_gap_pp"]))
    assert not mismatches, f"final-octave delta drift: {mismatches[:3]}"


def test_bfs_universally_unsaturated(payload):
    """bfs is the structural 'cache never helps enough' app.

    Every policy should be unsaturated at paper L3 on bfs (working
    set far exceeds 8MB on the large-corpus graphs).
    """
    for pol in EXPECTED_POLICIES:
        blob = payload["per_app"]["bfs"][pol]
        assert blob["saturation_onset"] == "never", (
            f"bfs/{pol} unexpectedly saturated at {blob['saturation_onset']}"
        )
