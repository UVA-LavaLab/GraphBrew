"""Gate 75 — per-app saturation-vs-slope extremum corroboration invariants."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN = REPO_ROOT / "scripts" / "experiments" / "ecg" / "saturation_slope_extremum.py"
JSON_PATH = REPO_ROOT / "wiki" / "data" / "saturation_slope_extremum.json"
DISTANCE_JSON = REPO_ROOT / "wiki" / "data" / "saturation_distance.json"
SLOPE_JSON    = REPO_ROOT / "wiki" / "data" / "per_app_capacity_slope.json"

EXPECTED_APPS = {"bc", "bfs", "cc", "pr", "sssp"}
EXPECTED_BFS = "bfs"
SLOPE_STEEPNESS_RATIO_FLOOR = 3.0
DISTANCE_RATIO_FLOOR = 2.5


@pytest.fixture(scope="module")
def payload() -> dict:
    for sib, name in (
        (DISTANCE_JSON, "saturation_distance.json (gate 65)"),
        (SLOPE_JSON,    "per_app_capacity_slope.json (gate 68)"),
    ):
        if not sib.exists():
            pytest.skip(f"{name} missing — run upstream gate first")
    if not JSON_PATH.exists():
        subprocess.check_call([sys.executable, str(GEN)], cwd=str(REPO_ROOT))
    return json.loads(JSON_PATH.read_text())


def test_payload_well_formed(payload):
    assert "meta" in payload
    meta = payload["meta"]
    for k in (
        "distance_source",
        "slope_source",
        "apps",
        "per_app",
        "least_cache_sensitive_app_by_distance",
        "least_cache_sensitive_app_by_slope",
        "most_cache_hungry_app_by_distance",
        "most_cache_hungry_app_by_slope",
        "most_hungry_app_disagreement_note",
        "slope_steepness_ratio_floor",
        "distance_ratio_floor",
        "verdict_checks",
        "verdict",
    ):
        assert k in meta, f"missing meta.{k}"


def test_all_expected_apps_present(payload):
    apps = set(payload["meta"]["apps"])
    missing = EXPECTED_APPS - apps
    assert not missing, f"missing apps: {missing}"


def test_every_app_has_full_block(payload):
    for app in payload["meta"]["apps"]:
        e = payload["meta"]["per_app"][app]
        for k in (
            "distance_pp",
            "slope_pp_oct",
            "slope_steepness",
            "distance_rank",
            "slope_rank",
        ):
            assert k in e, f"app {app!r}: missing {k}"


def test_bfs_is_argmin_distance(payload):
    assert payload["meta"]["least_cache_sensitive_app_by_distance"] == EXPECTED_BFS, (
        f"argmin(distance) is "
        f"{payload['meta']['least_cache_sensitive_app_by_distance']!r}, "
        f"expected {EXPECTED_BFS!r}"
    )


def test_bfs_is_shallowest_slope(payload):
    assert payload["meta"]["least_cache_sensitive_app_by_slope"] == EXPECTED_BFS, (
        f"argmin(|slope|) is "
        f"{payload['meta']['least_cache_sensitive_app_by_slope']!r}, "
        f"expected {EXPECTED_BFS!r}"
    )


def test_bfs_has_rank_1_on_both_metrics(payload):
    bfs = payload["meta"]["per_app"][EXPECTED_BFS]
    assert bfs["distance_rank"] == 1, (
        f"bfs distance_rank={bfs['distance_rank']}, expected 1"
    )
    assert bfs["slope_rank"] == 1, (
        f"bfs slope_rank={bfs['slope_rank']}, expected 1"
    )


def test_bfs_is_unique_extremum(payload):
    bfs = payload["meta"]["per_app"][EXPECTED_BFS]
    for app, e in payload["meta"]["per_app"].items():
        if app == EXPECTED_BFS:
            continue
        assert e["distance_pp"] > bfs["distance_pp"], (
            f"app {app!r} distance {e['distance_pp']:.4f} <= "
            f"bfs distance {bfs['distance_pp']:.4f} (bfs not unique)"
        )
        assert e["slope_steepness"] > bfs["slope_steepness"], (
            f"app {app!r} slope_steepness {e['slope_steepness']:.4f} <= "
            f"bfs slope_steepness {bfs['slope_steepness']:.4f} (bfs not unique)"
        )


def test_corpus_has_sensitive_kernel_by_slope(payload):
    bfs = payload["meta"]["per_app"][EXPECTED_BFS]
    max_ratio = max(
        e["slope_steepness"] / bfs["slope_steepness"]
        for a, e in payload["meta"]["per_app"].items() if a != EXPECTED_BFS
    )
    assert max_ratio >= SLOPE_STEEPNESS_RATIO_FLOOR, (
        f"max slope-steepness ratio {max_ratio:.3f} below "
        f"{SLOPE_STEEPNESS_RATIO_FLOOR} — corpus may lack genuinely "
        f"cache-sensitive kernels"
    )


def test_corpus_has_sensitive_kernel_by_distance(payload):
    bfs = payload["meta"]["per_app"][EXPECTED_BFS]
    max_ratio = max(
        e["distance_pp"] / bfs["distance_pp"]
        for a, e in payload["meta"]["per_app"].items() if a != EXPECTED_BFS
    )
    assert max_ratio >= DISTANCE_RATIO_FLOOR, (
        f"max distance ratio {max_ratio:.3f} below "
        f"{DISTANCE_RATIO_FLOOR} — corpus may lack genuinely "
        f"cache-sensitive kernels"
    )


def test_most_hungry_disagreement_note_informational(payload):
    note = payload["meta"]["most_hungry_app_disagreement_note"]
    assert isinstance(note, str), "note must be a string"
    assert "INFORMATIONAL" in note, (
        "note must contain 'INFORMATIONAL' marker so future maintainers "
        "know the most-hungry disagreement is NOT gated"
    )


def test_all_verdict_checks_green(payload):
    failed = [k for k, v in payload["meta"]["verdict_checks"].items() if not v]
    assert not failed, f"failed verdict checks: {failed}"


def test_verdict_pass(payload):
    assert payload["meta"]["verdict"] == "PASS", (
        f"verdict={payload['meta']['verdict']}, "
        f"checks={payload['meta']['verdict_checks']}"
    )
