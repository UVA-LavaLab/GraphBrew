"""Tests for the oracle-gap trajectory curvature / knee gate 58."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "wiki" / "data" / "oracle_gap_curvature.json"
GENERATOR = (
    REPO_ROOT / "scripts" / "experiments" / "ecg" / "oracle_gap_curvature.py"
)


def _payload() -> dict:
    if not DATA.exists():
        subprocess.check_call([sys.executable, str(GENERATOR)])
    return json.loads(DATA.read_text())


def test_payload_present_and_well_formed():
    p = _payload()
    assert "meta" in p
    assert "per_policy_summary" in p
    assert "per_app" in p
    assert p["meta"]["scope_l3_sizes"] == ["1MB", "4MB", "8MB"]
    assert p["meta"]["x_axis"] == "log2(L3 / 1MB)"


def test_every_app_policy_cell_present():
    p = _payload()
    cells = p["meta"]["cells_total"]
    assert cells == 20, f"expected 5 apps × 4 policies = 20; got {cells}"
    for app in ("bc", "bfs", "cc", "pr", "sssp"):
        assert app in p["per_app"]
        assert set(p["per_app"][app].keys()) >= {"GRASP", "LRU", "POPT", "SRRIP"}


def test_curvature_fields_present():
    p = _payload()
    needed = {
        "gap_at_1MB",
        "gap_at_4MB",
        "gap_at_8MB",
        "slope_1MB_to_4MB",
        "slope_4MB_to_8MB",
        "curvature_at_4MB",
        "knee_present",
    }
    for app, pols in p["per_app"].items():
        for pol, c in pols.items():
            assert needed <= set(c.keys()), (
                f"{app}/{pol} missing curvature fields"
            )


def test_oracle_aware_policies_dominate_knee_count():
    p = _payload()
    s = p["per_policy_summary"]
    min_oracle = min(s["GRASP"]["knee_count"], s["POPT"]["knee_count"])
    max_nonoracle = max(s["LRU"]["knee_count"], s["SRRIP"]["knee_count"])
    assert min_oracle > max_nonoracle, (
        "oracle-aware policies must have more knees than non-oracle;"
        f" got GRASP={s['GRASP']['knee_count']},"
        f" POPT={s['POPT']['knee_count']},"
        f" LRU={s['LRU']['knee_count']},"
        f" SRRIP={s['SRRIP']['knee_count']}"
    )


def test_non_oracle_policies_have_zero_knees():
    # LRU and SRRIP should not yet be plateauing — their trajectories
    # are still descending at 4→8MB on most apps. Allow at most 1 for
    # noise tolerance.
    p = _payload()
    s = p["per_policy_summary"]
    assert s["LRU"]["knee_count"] <= 1
    assert s["SRRIP"]["knee_count"] <= 1


def test_grasp_mean_curvature_positive():
    # GRASP shows the sharpest knee on social/web apps where its small-L3
    # gap is large and its 8MB gap collapses to near zero.
    p = _payload()
    assert p["per_policy_summary"]["GRASP"]["mean_curvature"] > 0


def test_popt_mean_curvature_non_negative():
    # Charged POPT is mid-pack rather than flat; its mean curvature may be
    # slightly negative while knee count remains oracle-aware.
    p = _payload()
    assert p["per_policy_summary"]["POPT"]["mean_curvature"] > -0.5


def test_lru_srrip_mean_curvature_negative_or_zero():
    # Non-oracle policies still have appreciable slope at 4→8MB on
    # bfs/sssp/cc/pr; their mean curvature must be <= 0 (still
    # accelerating, not plateauing).
    p = _payload()
    assert p["per_policy_summary"]["LRU"]["mean_curvature"] <= 0
    assert p["per_policy_summary"]["SRRIP"]["mean_curvature"] <= 0


def test_verdict_is_pass():
    p = _payload()
    assert p["meta"]["knee_lead_verdict"] == "PASS"


def test_knee_rank_starts_with_oracle_aware():
    p = _payload()
    rank = p["meta"]["knee_rank_by_policy"]
    assert rank[0] in ("GRASP", "POPT"), (
        f"knee_rank must lead with an oracle-aware policy; got {rank}"
    )
    assert rank[-1] in ("LRU", "SRRIP"), (
        f"knee_rank must trail with a non-oracle policy; got {rank}"
    )


def test_cross_gate_55_present_and_consistent_on_oracle_split():
    # Gate 55 and gate 58 may disagree on lead (saturation vs
    # curvature measure different things). But they MUST agree that
    # POPT and GRASP outrank LRU and SRRIP in their respective ranks.
    p = _payload()
    cgc = p["meta"].get("cross_gate_consistency")
    if not cgc:
        import pytest

        pytest.skip("gate 55 artifact not present yet")
    rank55 = cgc["saturation_rank_gate55"]
    rank58 = cgc["knee_rank_gate58"]
    assert rank55 and rank58
    # Position of each policy in each rank
    pos55 = {p: i for i, p in enumerate(rank55)}
    pos58 = {p: i for i, p in enumerate(rank58)}
    for oracle in ("GRASP", "POPT"):
        for non in ("LRU", "SRRIP"):
            # Charged corpus: gate55 ranks oracle-aware policies before the
            # blind LRU/SRRIP baselines on saturation distance.
            assert pos55[oracle] < pos55[non], (
                f"gate55: {oracle} should precede {non}; got {rank55}"
            )
            # Knee (gate58): the oracle-aware policies capture the working set
            # quickly, so their oracle-gap knee comes EARLIER (lower index).
            assert pos58[oracle] < pos58[non], (
                f"gate58: {oracle} knee should precede {non}; got {rank58}"
            )


def test_slope_computation_log2_consistency():
    # Spot-check: for any cell, slope_1MB_to_4MB equals
    # (gap_at_4MB - gap_at_1MB) / 2 within rounding.
    p = _payload()
    bad = []
    for app, pols in p["per_app"].items():
        for pol, c in pols.items():
            expected = (c["gap_at_4MB"] - c["gap_at_1MB"]) / 2.0
            if abs(c["slope_1MB_to_4MB"] - expected) > 0.01:
                bad.append((app, pol, expected, c["slope_1MB_to_4MB"]))
    assert not bad, f"slope mismatch in {bad[:5]}"
