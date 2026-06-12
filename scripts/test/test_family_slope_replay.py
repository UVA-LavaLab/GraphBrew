"""Gate 67 — per-family capacity-sensitivity slope replay invariants."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN = REPO_ROOT / "scripts" / "experiments" / "ecg" / "family_slope_replay.py"
JSON_PATH = REPO_ROOT / "wiki" / "data" / "family_slope_replay.json"
GATE66_JSON = REPO_ROOT / "wiki" / "data" / "capacity_sensitivity.json"

ORACLE_AWARE = ("GRASP", "POPT")
NON_ORACLE = ("LRU", "SRRIP")
HELP_FLOOR_PP_OCTAVE = -5.0


@pytest.fixture(scope="module")
def payload() -> dict:
    if not JSON_PATH.exists():
        subprocess.check_call([sys.executable, str(GEN)], cwd=str(REPO_ROOT))
    return json.loads(JSON_PATH.read_text())


def test_payload_well_formed(payload):
    assert "meta" in payload and "per_family" in payload
    meta = payload["meta"]
    for k in (
        "qualifying_families",
        "replay_count",
        "deviating_families",
        "pinned_deviating_families",
        "new_deviating_families",
        "verdict",
        "help_floor_pp_octave",
    ):
        assert k in meta, f"missing meta.{k}"
    assert meta["help_floor_pp_octave"] == HELP_FLOOR_PP_OCTAVE


def test_at_least_one_qualifying_family(payload):
    assert len(payload["meta"]["qualifying_families"]) >= 1


def test_replay_count_nonzero(payload):
    assert payload["meta"]["replay_count"] >= 1, (
        "at least one family must replay the global slope ordering"
    )


def test_no_new_deviating_families(payload):
    new = payload["meta"]["new_deviating_families"]
    assert new == [], f"NEW deviating families appeared: {new}"


def test_verdict_pass(payload):
    assert payload["meta"]["verdict"] == "PASS", payload["meta"]


def test_pinned_deviations_match_actual(payload):
    meta = payload["meta"]
    pinned = set(meta["pinned_deviating_families"])
    deviating = set(meta["deviating_families"])
    qual = set(meta["qualifying_families"])
    # every pinned family must STILL be qualifying AND actually deviating
    for fam in pinned:
        assert fam in qual, (
            f"pinned family {fam!r} no longer qualifies — remove from pin"
        )
        assert fam in deviating, (
            f"pinned family {fam!r} no longer deviates — remove from pin"
        )


def test_every_family_has_all_four_policies(payload):
    for fam, info in payload["per_family"].items():
        pp = info["per_policy"]
        for pol in ORACLE_AWARE + NON_ORACLE:
            assert pol in pp, f"family {fam!r} missing policy {pol!r}"


def test_every_family_has_negative_medians(payload):
    """Every per-family per-policy median slope must be < 0."""
    for fam, info in payload["per_family"].items():
        for pol, stats in info["per_policy"].items():
            assert stats["median_pp"] < 0.0, (
                f"family {fam!r} policy {pol!r} has non-negative median "
                f"slope {stats['median_pp']!r}"
            )


def test_replaying_families_obey_help_floor(payload):
    """If a family replays, every policy median is below the help floor."""
    for fam, info in payload["per_family"].items():
        if not info["replays_pattern"]:
            continue
        for pol, stats in info["per_policy"].items():
            assert stats["median_pp"] < HELP_FLOOR_PP_OCTAVE, (
                f"replaying family {fam!r} policy {pol!r} median "
                f"{stats['median_pp']!r} is not below help floor"
            )


def test_replaying_families_lru_strictly_steeper_than_grasp(payload):
    for fam, info in payload["per_family"].items():
        if not info["replays_pattern"]:
            continue
        med = {p: info["per_policy"][p]["median_pp"] for p in info["per_policy"]}
        assert med["LRU"] < med["GRASP"], (
            f"replaying family {fam!r}: LRU median {med['LRU']!r} not strictly "
            f"steeper than GRASP {med['GRASP']!r}"
        )


def test_consistent_with_gate66_global(payload):
    """If gate 66 sibling is present, its global ordering must match the
    family-level prediction: at least one family replays."""
    if not GATE66_JSON.exists():
        pytest.skip("gate 66 sibling JSON missing")
    g66 = json.loads(GATE66_JSON.read_text())
    assert g66["meta"]["verdict"] == "PASS"
    assert g66["meta"]["steepest_policy"] == "LRU"
    assert g66["meta"]["shallowest_policy"] == "POPT"
    # Gate 67 must therefore have at least one family replay.
    assert payload["meta"]["replay_count"] >= 1


def test_cell_counts_consistent_per_family(payload):
    """Within a family, every policy must score the same number of cells
    (same set of (graph, app) cells satisfied the coverage gate)."""
    for fam, info in payload["per_family"].items():
        counts = {p: s["n_cells"] for p, s in info["per_policy"].items()}
        unique = set(counts.values())
        assert len(unique) == 1, (
            f"family {fam!r} has inconsistent per-policy cell counts: {counts}"
        )
