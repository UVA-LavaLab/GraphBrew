"""Tests for gate 66 — per-policy capacity-sensitivity slope."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "wiki" / "data" / "capacity_sensitivity.json"
GENERATOR = (
    REPO_ROOT / "scripts" / "experiments" / "ecg" / "capacity_sensitivity.py"
)


def _payload() -> dict:
    if not DATA.exists():
        subprocess.check_call([sys.executable, str(GENERATOR)])
    return json.loads(DATA.read_text())


def test_payload_well_formed():
    p = _payload()
    assert "meta" in p and "per_cell" in p
    m = p["meta"]
    assert m["policies"] == ["GRASP", "LRU", "POPT", "SRRIP"]
    assert m["l3_axis"] == ["1MB", "4MB", "8MB"]


def test_all_four_policies_scored():
    p = _payload()
    s = p["meta"]["policy_summary"]
    assert set(s.keys()) == {"GRASP", "LRU", "POPT", "SRRIP"}


def test_equal_cell_counts_per_policy():
    # Each policy must be scored on the same population of cells
    # (the corpus has full policy coverage on the 4 / 8 MB L3 axis).
    p = _payload()
    counts = {pol: s["n_cells"] for pol, s in p["meta"]["policy_summary"].items()}
    assert len(set(counts.values())) == 1, (
        f"unequal per-policy cell counts: {counts}"
    )
    # And there must be at least 4 graphs * 5 apps = 20 cells per policy
    assert all(c >= 20 for c in counts.values())


def test_total_cells_matches_sum_of_per_policy():
    p = _payload()
    total = sum(s["n_cells"] for s in p["meta"]["policy_summary"].values())
    assert p["meta"]["cell_count"] == total


def test_every_policy_median_helps():
    # All policies must have median slope < -5 pp/octave (cache helps).
    p = _payload()
    for pol, s in p["meta"]["policy_summary"].items():
        assert s["median_pp"] < -5.0, (
            f"{pol} median slope {s['median_pp']} fails help floor"
        )


def test_lru_is_steepest():
    p = _payload()
    assert p["meta"]["steepest_policy"] == "LRU"


def test_grasp_is_shallowest():
    p = _payload()
    assert p["meta"]["shallowest_policy"] == "GRASP"


def test_grasp_median_strictly_shallower_than_lru():
    p = _payload()
    s = p["meta"]["policy_summary"]
    assert s["GRASP"]["median_pp"] > s["LRU"]["median_pp"], (
        f"GRASP median ({s['GRASP']['median_pp']}) must be shallower "
        f"(less negative) than LRU median ({s['LRU']['median_pp']})"
    )


def test_median_steepness_gap_positive():
    p = _payload()
    assert p["meta"]["median_steepness_gap_pp"] > 0.0


def test_steepness_gap_within_reasonable_bound():
    # The gap between steepest and shallowest must be small (<5pp);
    # otherwise the corpus is wildly noisy. Today it's ~0.97pp.
    p = _payload()
    assert p["meta"]["median_steepness_gap_pp"] < 5.0


def test_no_cell_has_strongly_positive_slope():
    # Any cell whose slope is significantly positive would imply 8MB
    # is much worse than 1MB, which violates cache monotonicity.
    # Allow small numerical noise up to 0.05pp/octave.
    p = _payload()
    for c in p["per_cell"]:
        assert c["slope_pp"] < 0.05, (
            f"{c['app']}/{c['graph']}/{c['policy']} has strongly "
            f"positive slope {c['slope_pp']}"
        )


def test_verdict_is_pass():
    p = _payload()
    assert p["meta"]["verdict"] == "PASS"
