"""Tests for gate 63 — per-family winner-margin replay."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "wiki" / "data" / "family_margin_replay.json"
GENERATOR = (
    REPO_ROOT / "scripts" / "experiments" / "ecg" / "family_margin_replay.py"
)
GLOBAL_DATA = REPO_ROOT / "wiki" / "data" / "winner_margin_by_regime.json"


def _payload() -> dict:
    if not DATA.exists():
        subprocess.check_call([sys.executable, str(GENERATOR)])
    return json.loads(DATA.read_text())


def test_payload_well_formed():
    p = _payload()
    assert "meta" in p and "per_family" in p
    m = p["meta"]
    assert m["policies"] == ["GRASP", "LRU", "POPT", "SRRIP"]
    assert m["regimes"] == ["under_wss", "near_wss", "over_wss"]


def test_all_corpus_families_present():
    p = _payload()
    expected = {"citation", "mesh", "road", "social", "web"}
    assert set(p["per_family"].keys()) == expected


def test_at_least_one_qualifying_family():
    p = _payload()
    assert len(p["meta"]["qualifying_families"]) >= 1


def test_at_least_one_replaying_family():
    p = _payload()
    assert p["meta"]["replay_count"] >= 1


def test_no_new_deviations_beyond_pin():
    p = _payload()
    assert p["meta"]["new_deviating_families"] == [], (
        f"unexpected new family deviations: "
        f"{p['meta']['new_deviating_families']}"
    )


def test_social_family_qualifies_and_replays():
    # social has 4 graphs spanning a wide WSS range and is the
    # diversity backbone of the corpus; this is the family that
    # MUST keep working.
    p = _payload()
    s = p["per_family"]["social"]
    assert s["qualifying"], "social family must qualify for the replay test"
    assert s["replays"], (
        "social family must replay the global margin-shrink pattern"
    )


def test_social_shrink_evidence_for_both_oracle_policies():
    p = _payload()
    s = p["per_family"]["social"]
    pols = {e["policy"] for e in s["shrink_evidence"]}
    assert {"GRASP", "POPT"}.issubset(pols), (
        f"social family must show shrink evidence for both GRASP and POPT;"
        f" got {pols}"
    )


def test_every_family_has_non_zero_classified_cells():
    p = _payload()
    for fam, s in p["per_family"].items():
        assert s["cells_classified"] > 0, (
            f"family {fam} has zero classified cells (data missing?)"
        )


def test_no_cells_skipped_in_any_family():
    p = _payload()
    for fam, s in p["per_family"].items():
        assert s["cells_skipped"] == 0, (
            f"family {fam} had {s['cells_skipped']} skipped cells"
        )


def test_qualifying_families_subset_of_replaying_plus_deviating():
    p = _payload()
    m = p["meta"]
    assert set(m["replaying_families"]).isdisjoint(m["deviating_families"])
    assert set(m["qualifying_families"]) == (
        set(m["replaying_families"]) | set(m["deviating_families"])
    )


def test_replay_directionality_matches_global_gate_62():
    # If the social family shows GRASP under > over, the global gate 62
    # must also report GRASP as shrink evidence — and vice versa for
    # POPT. This catches drift between family-level and global signals.
    if not GLOBAL_DATA.exists():
        import pytest
        pytest.skip("gate 62 artifact missing")
    g62 = json.loads(GLOBAL_DATA.read_text())
    g62_pols = {ev["policy"] for ev in g62["meta"]["shrink_evidence"]}
    p = _payload()
    fam_pols = {ev["policy"] for ev in p["per_family"]["social"]["shrink_evidence"]}
    assert fam_pols.issubset(g62_pols), (
        f"family-level shrink policies {fam_pols} must be a subset of "
        f"global shrink policies {g62_pols}"
    )


def test_verdict_is_pass():
    p = _payload()
    assert p["meta"]["verdict"] == "PASS"
