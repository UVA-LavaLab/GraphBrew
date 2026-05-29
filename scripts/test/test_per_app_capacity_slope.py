"""Gate 68 — per-app capacity-sensitivity slope invariants."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN = REPO_ROOT / "scripts" / "experiments" / "ecg" / "per_app_capacity_slope.py"
JSON_PATH = REPO_ROOT / "wiki" / "data" / "per_app_capacity_slope.json"
GATE65_JSON = REPO_ROOT / "wiki" / "data" / "saturation_distance.json"

EXPECTED_APPS = {"bc", "bfs", "cc", "pr", "sssp"}
HELP_FLOOR_PP_OCTAVE = -5.0


@pytest.fixture(scope="module")
def payload() -> dict:
    if not JSON_PATH.exists():
        subprocess.check_call([sys.executable, str(GEN)], cwd=str(REPO_ROOT))
    return json.loads(JSON_PATH.read_text())


def test_payload_well_formed(payload):
    assert "meta" in payload and "per_cell" in payload
    meta = payload["meta"]
    for k in (
        "apps",
        "policies",
        "per_app",
        "per_app_median_of_medians_pp",
        "most_cache_hungry_app",
        "least_cache_hungry_app",
        "per_app_median_range_pp",
        "deviating_apps",
        "pinned_deviating_apps",
        "new_deviating_apps",
        "verdict",
        "help_floor_pp_octave",
        "allow_lru_shallower_by_pp",
    ):
        assert k in meta, f"missing meta.{k}"


def test_all_expected_apps_present(payload):
    apps = set(payload["meta"]["apps"])
    missing = EXPECTED_APPS - apps
    assert not missing, f"missing apps: {missing}"


def test_every_app_has_four_policies(payload):
    for app, block in payload["meta"]["per_app"].items():
        assert set(block.keys()) == {"GRASP", "LRU", "POPT", "SRRIP"}, (
            f"app {app!r} has policies {set(block.keys())}"
        )


def test_every_median_strictly_negative(payload):
    for app, block in payload["meta"]["per_app"].items():
        for pol, stats in block.items():
            assert stats["median_pp"] < 0.0, (
                f"({app!r}, {pol!r}) has non-negative median {stats['median_pp']!r}"
            )


def test_no_new_deviating_apps(payload):
    new = payload["meta"]["new_deviating_apps"]
    assert new == [], f"NEW deviating apps appeared: {new}"


def test_at_least_one_cache_sensitive_app(payload):
    assert payload["meta"]["invariant_at_least_one_cache_sensitive_app"], (
        "no app has every policy median below the help floor"
    )


def test_verdict_pass(payload):
    assert payload["meta"]["verdict"] == "PASS", payload["meta"]


def test_pinned_deviations_match_actual(payload):
    meta = payload["meta"]
    pinned = set(meta["pinned_deviating_apps"])
    deviating = set(meta["deviating_apps"])
    apps = set(meta["apps"])
    for app in pinned:
        assert app in apps, (
            f"pinned app {app!r} no longer observed — remove from pin"
        )
        assert app in deviating, (
            f"pinned app {app!r} no longer deviates — remove from pin"
        )


def test_per_app_diversity_range(payload):
    """At least 3 pp/octave spread between most- and least- cache-hungry
    apps — guards against the corpus collapsing to a single saturation
    regime."""
    assert payload["meta"]["per_app_median_range_pp"] >= 3.0, (
        f"insufficient per-app range: {payload['meta']['per_app_median_range_pp']!r}"
    )


def test_per_cell_count_matches_expected(payload):
    """Every (app, policy) block reports a cell count that fits within
    the corpus shape (5 apps x 8 graphs = 40 max, but many have <8)."""
    for app, block in payload["meta"]["per_app"].items():
        for pol, stats in block.items():
            assert 1 <= stats["n_cells"] <= 8, (
                f"({app!r}, {pol!r}) cell count {stats['n_cells']!r} out of range"
            )


def test_consistent_with_gate65_saturation(payload):
    """If gate 65 sibling is present, the per-app saturation ordering
    should agree with the per-app slope ordering at least directionally:
    the most-saturated app (smallest median 4MB->8MB distance) should
    NOT also be the most-cache-hungry app."""
    if not GATE65_JSON.exists():
        pytest.skip("gate 65 sibling JSON missing")
    g65 = json.loads(GATE65_JSON.read_text())
    # gate 65 exposes per-app summary at top-level under 'per_app',
    # with median_pp as the per-app median distance.
    if "per_app" not in g65:
        pytest.skip("gate 65 schema does not expose per_app")
    g65_app = g65["per_app"]
    # smallest median distance == most saturated
    saturated_app = min(
        g65_app,
        key=lambda a: abs(g65_app[a].get("median_pp", 0.0)),
    )
    hungry_app = payload["meta"]["most_cache_hungry_app"]
    assert saturated_app != hungry_app, (
        f"most-saturated app ({saturated_app!r}) cannot also be the "
        f"most-cache-hungry app ({hungry_app!r})"
    )


def test_most_hungry_app_below_help_floor(payload):
    """The 'most cache-hungry app' must have all-policy median slopes
    strictly below the help floor (else the ranking is meaningless)."""
    meta = payload["meta"]
    app = meta["most_cache_hungry_app"]
    block = meta["per_app"][app]
    for pol, stats in block.items():
        assert stats["median_pp"] < HELP_FLOOR_PP_OCTAVE, (
            f"most-hungry app {app!r} policy {pol!r} median "
            f"{stats['median_pp']!r} not below help floor"
        )
