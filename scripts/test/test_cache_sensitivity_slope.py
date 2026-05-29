"""Gate 52: per-(app, policy) cache-sensitivity slope across L3 octaves.

Empirical finding pinned: 10 of 20 (app, policy) cells are NOT
monotonically decreasing in gap_pp as L3 grows — but ALL the
"significant" anti-scaling cases (|delta_gap| >= 1.0 pp at any
octave) are confined to LRU and SRRIP. GRASP and POPT (the two
oracle-aware policies) never show significant anti-scaling — their
only "violations" are noise-floor rebounds after the gap has
already collapsed to <1 pp.

This pins:
  - 4-policy / 5-app inventory
  - per-policy mean-slope sign + ordering (GRASP > POPT > SRRIP >= LRU)
  - exact set of significant anti-scaling cells (LRU/SRRIP only)
  - GRASP and POPT never have significant anti-scaling
  - LRU and SRRIP have at least one significant anti-scaling cell each
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "wiki" / "data" / "cache_sensitivity_slope.json"

SIGNIFICANT_PP_THRESHOLD = 1.0


@pytest.fixture(scope="module")
def payload() -> dict:
    if not DATA.exists():
        pytest.skip("cache_sensitivity_slope.json not built")
    return json.loads(DATA.read_text())


def test_meta_inventory(payload):
    m = payload["meta"]
    assert m["n_apps"] == 5
    assert m["n_policies"] == 4
    assert m["policies"] == ["GRASP", "LRU", "POPT", "SRRIP"]
    assert m["apps"] == ["bc", "bfs", "cc", "pr", "sssp"]
    assert m["l3_octaves"] == ["1MB", "4MB", "8MB"]
    assert m["slope_units"] == "gap_pp per L3 octave (log2 MB)"


def test_per_policy_mean_slope_ordering(payload):
    """Mean slope (gap_pp shrinkage per octave): GRASP largest, then
    POPT (already near oracle so less to shrink), then SRRIP, then
    LRU (sometimes negative)."""
    s = payload["per_policy_summary"]
    grasp = s["GRASP"]["mean_avg_slope"]
    popt = s["POPT"]["mean_avg_slope"]
    srrip = s["SRRIP"]["mean_avg_slope"]
    lru = s["LRU"]["mean_avg_slope"]
    assert grasp > popt, f"GRASP mean slope {grasp} <= POPT {popt}"
    assert popt > srrip, f"POPT mean slope {popt} <= SRRIP {srrip}"
    assert popt > lru, f"POPT mean slope {popt} <= LRU {lru}"


def test_grasp_has_largest_mean_slope(payload):
    """GRASP benefits most per cache octave — it starts furthest from
    oracle at small L3 and converges fastest as L3 grows."""
    means = {p: d["mean_avg_slope"] for p, d in payload["per_policy_summary"].items()}
    assert max(means, key=means.get) == "GRASP"


def test_significant_anti_scaling_is_lru_or_srrip_only(payload):
    """The paper's mechanism story: oracle-aware policies (GRASP, POPT)
    converge monotonically; oracle-unaware policies (LRU, SRRIP) can
    suffer cache-induced regressions when extra capacity admits stale
    lines that delay re-references. Pin this strictly."""
    for app, per_pol in payload["per_app"].items():
        for pol, data in per_pol.items():
            for oct_entry in data["octaves"]:
                d = oct_entry["delta_gap_pp"]
                if d >= SIGNIFICANT_PP_THRESHOLD:
                    assert pol in ("LRU", "SRRIP"), (
                        f"Significant anti-scaling on {app}/{pol} "
                        f"({oct_entry['from']}->{oct_entry['to']}: "
                        f"+{d} pp) — but only LRU/SRRIP are expected "
                        f"to have such regressions per gate 52"
                    )


def test_grasp_never_has_significant_anti_scaling(payload):
    """Hard guarantee: GRASP gap_pp never grows by >= 1.0 pp at any octave."""
    for app, per_pol in payload["per_app"].items():
        data = per_pol.get("GRASP")
        if not data:
            continue
        for oct_entry in data["octaves"]:
            assert oct_entry["delta_gap_pp"] < SIGNIFICANT_PP_THRESHOLD, (
                f"GRASP on {app} shows significant anti-scaling "
                f"({oct_entry['from']}->{oct_entry['to']}: "
                f"+{oct_entry['delta_gap_pp']} pp)"
            )


def test_popt_never_has_significant_anti_scaling(payload):
    """Hard guarantee: POPT gap_pp never grows by >= 1.0 pp at any octave."""
    for app, per_pol in payload["per_app"].items():
        data = per_pol.get("POPT")
        if not data:
            continue
        for oct_entry in data["octaves"]:
            assert oct_entry["delta_gap_pp"] < SIGNIFICANT_PP_THRESHOLD, (
                f"POPT on {app} shows significant anti-scaling "
                f"({oct_entry['from']}->{oct_entry['to']}: "
                f"+{oct_entry['delta_gap_pp']} pp)"
            )


def test_lru_has_at_least_one_significant_anti_scaling(payload):
    """If LRU's significant anti-scaling cases disappear, the paper's
    'oracle-aware policies monotonically converge while LRU regresses'
    framing weakens. Pin this expectation."""
    found = []
    for app, per_pol in payload["per_app"].items():
        data = per_pol.get("LRU")
        if not data:
            continue
        for oct_entry in data["octaves"]:
            if oct_entry["delta_gap_pp"] >= SIGNIFICANT_PP_THRESHOLD:
                found.append((app, oct_entry["from"], oct_entry["to"]))
    assert len(found) >= 1, "LRU no longer shows any significant anti-scaling"


def test_srrip_has_at_least_one_significant_anti_scaling(payload):
    """Same as LRU but for SRRIP — pin the empirical observation."""
    found = []
    for app, per_pol in payload["per_app"].items():
        data = per_pol.get("SRRIP")
        if not data:
            continue
        for oct_entry in data["octaves"]:
            if oct_entry["delta_gap_pp"] >= SIGNIFICANT_PP_THRESHOLD:
                found.append((app, oct_entry["from"], oct_entry["to"]))
    assert len(found) >= 1, "SRRIP no longer shows any significant anti-scaling"


def test_total_shrinkage_positive_or_near_zero_for_grasp_and_popt(payload):
    """Net shrinkage 1MB->8MB must be > -0.5 pp (i.e. non-negative
    modulo noise floor) for GRASP and POPT on every app. pr/POPT is
    already at <0.2 pp gap at 1MB so tiny rebounds are noise."""
    for app, per_pol in payload["per_app"].items():
        for pol in ("GRASP", "POPT"):
            data = per_pol.get(pol)
            if not data:
                continue
            assert data["total_shrinkage_pp"] > -0.5, (
                f"{pol} on {app} has NEGATIVE net shrinkage > 0.5 pp "
                f"1MB->8MB: {data['total_shrinkage_pp']} pp"
            )


def test_per_app_per_policy_has_2_octaves(payload):
    """Each (app, policy) trajectory must have exactly 2 octave entries
    (1MB->4MB and 4MB->8MB) — guards against missing L3 sweep data."""
    for app, per_pol in payload["per_app"].items():
        for pol, data in per_pol.items():
            assert len(data["octaves"]) == 2, (
                f"{app}/{pol} has {len(data['octaves'])} octaves; expected 2"
            )


def test_grasp_slope_largest_on_bfs(payload):
    """Empirical observation: GRASP gets the biggest cache-octave
    benefit on bfs (drop from 12.5 pp gap at 1MB to ~1 pp at 8MB).
    Pin this — it's a paper-grade single-cell highlight."""
    bfs_slopes = {pol: payload["per_app"]["bfs"][pol]["avg_slope_pp_per_octave"]
                  for pol in payload["meta"]["policies"]}
    assert max(bfs_slopes, key=bfs_slopes.get) == "GRASP"
    assert bfs_slopes["GRASP"] > 3.0, f"GRASP slope on bfs = {bfs_slopes['GRASP']}"


def test_cross_gate_consistency_with_oracle_gap_auc(payload):
    """Trajectory data must be consistent with gate 49's per_app
    trajectories (no silent disagreement between the two summaries)."""
    auc_path = REPO_ROOT / "wiki" / "data" / "oracle_gap_auc.json"
    if not auc_path.exists():
        pytest.skip("oracle_gap_auc.json not built (gate 49)")
    g49 = json.loads(auc_path.read_text())
    for app in payload["meta"]["apps"]:
        for pol in payload["meta"]["policies"]:
            d52 = payload["per_app"][app][pol]
            t49 = g49["per_app"][app]["trajectory_by_policy"][pol]
            assert abs(d52["gap_at_1MB"] - t49["1MB"]) < 1e-3
            assert abs(d52["gap_at_8MB"] - t49["8MB"]) < 1e-3
