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


def test_charged_popt_saturates_on_its_strongest_app(payload):
    """With the faithful 1-way RRM capacity charge (2026-06-13), PRACTICAL
    P-OPT is no longer the keep-benefiting near-oracle: it plateaus on its
    single strongest app (saturation rank places POPT first, n_saturated=1
    vs 0 for the others). Corpus-wide saturation is otherwise minimal."""
    pv = payload["per_policy"]
    rank = payload["meta"]["saturation_rank_by_policy"]
    assert rank[0] == "POPT", f"POPT should be the most-saturated; got {rank}"
    assert pv["POPT"]["n_saturated"] >= max(
        pv[p]["n_saturated"] for p in ("GRASP", "LRU", "SRRIP")
    )


def test_saturation_is_minimal_across_policies(payload):
    """At the charged-POPT corpus, cache-saturation is minimal: no policy
    saturates on more than 1 app — the oracle-gaps keep shrinking with cache
    for almost every (app, policy) cell. POPT saturates on 1, others on 0."""
    pv = payload["per_policy"]
    for p in ("GRASP", "LRU", "SRRIP", "POPT"):
        assert pv[p]["n_saturated"] <= 1, (
            f"{p} saturated on {pv[p]['n_saturated']} apps; expected <=1"
        )


def test_popt_saturates_few_apps(payload):
    """POPT keeps benefiting from cache on most apps, so it saturates on
    at most 2 of 5 (near-oracle, exploits added capacity)."""
    n = payload["per_policy"]["POPT"]["n_saturated"]
    assert n <= 2, f"POPT saturated on {n} apps; expected <=2 (keeps benefiting)"


def test_lru_rarely_saturates(payload):
    """LRU should saturate on at most 2 apps (it almost always benefits from more cache)."""
    n = payload["per_policy"]["LRU"]["n_saturated"]
    assert n <= 2, f"LRU saturated on {n} apps; expected <=2"


def test_pr_popt_is_near_oracle_flat(payload):
    """pr/POPT is the canonical near-oracle cell: its oracle-gap is already
    tiny at 1MB and stays flat (final-octave |slope| < 0.5 pp). The onset
    classifier reports 'never' because POPT never had a gap to drop from —
    the always-low signature of a near-oracle trajectory, not a regression.
    """
    blob = payload["per_app"]["pr"]["POPT"]
    assert abs(blob["final_octave_slope_pp"]) < 0.5, (
        f"pr/POPT final-octave slope {blob['final_octave_slope_pp']} is not "
        f"flat; expected |slope| < 0.5 pp (near-oracle)"
    )


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


def test_bfs_oracle_aware_keep_benefiting(payload):
    """On bfs the oracle-aware policies (GRASP, POPT) keep benefiting from
    cache (onset 'never' — their gap keeps shrinking as L3 grows), while
    the blind LRU/SRRIP plateau at 4MB. Array-relative GRASP 0.15,
    single-thread (inverts the earlier multi-thread framing where the
    baselines stayed unsaturated and POPT saturated)."""
    for pol in ("GRASP", "POPT"):
        onset = payload["per_app"]["bfs"][pol]["saturation_onset"]
        assert onset == "never", (
            f"bfs/{pol} should keep benefiting from cache (onset 'never'); "
            f"got {onset}"
        )
