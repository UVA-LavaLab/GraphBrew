"""Gate 73 — per-app SRRIP-vs-GRASP slope ordering invariants."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN = REPO_ROOT / "scripts" / "experiments" / "ecg" / "per_app_srrip_vs_grasp.py"
JSON_PATH = REPO_ROOT / "wiki" / "data" / "per_app_srrip_vs_grasp.json"
PER_APP_JSON = REPO_ROOT / "wiki" / "data" / "per_app_capacity_slope.json"

EXPECTED_APPS = {"bc", "bfs", "cc", "pr", "sssp"}
PINNED_DEVIATING_APPS: set[str] = set()
ALLOW_SRRIP_SHALLOWER_BY_PP = 1.0


@pytest.fixture(scope="module")
def payload() -> dict:
    if not PER_APP_JSON.exists():
        pytest.skip("per_app_capacity_slope.json missing — run gate 68 first")
    if not JSON_PATH.exists():
        subprocess.check_call([sys.executable, str(GEN)], cwd=str(REPO_ROOT))
    return json.loads(JSON_PATH.read_text())


def test_payload_well_formed(payload):
    assert "meta" in payload
    meta = payload["meta"]
    for k in (
        "source",
        "apps",
        "allow_srrip_shallower_by_pp",
        "pinned_deviating_apps",
        "deviating_apps",
        "new_deviating_apps",
        "missing_apps",
        "per_app",
        "verdict_checks",
        "verdict",
    ):
        assert k in meta, f"missing meta.{k}"


def test_all_expected_apps_present(payload):
    apps = set(payload["meta"]["apps"])
    missing = EXPECTED_APPS - apps
    assert not missing, f"missing apps: {missing}"


def test_no_missing_grasp_or_srrip(payload):
    missing = payload["meta"]["missing_apps"]
    assert missing == [], f"apps with missing GRASP or SRRIP medians: {missing}"


def test_every_app_has_delta(payload):
    for app, e in payload["meta"]["per_app"].items():
        assert e["srrip_minus_grasp_pp_oct"] is not None, (
            f"app {app!r} missing SRRIP-GRASP delta"
        )


def test_pinned_deviating_set_is_only_bfs(payload):
    pinned = set(payload["meta"]["pinned_deviating_apps"])
    assert pinned == PINNED_DEVIATING_APPS, (
        f"pinned set drift: {pinned} != {PINNED_DEVIATING_APPS}"
    )


def test_no_new_deviating_apps(payload):
    new_dev = payload["meta"]["new_deviating_apps"]
    assert new_dev == [], (
        f"NEW per-app SRRIP shallower than GRASP by >{ALLOW_SRRIP_SHALLOWER_BY_PP} "
        f"pp/oct beyond pinned set: {new_dev}"
    )


def test_all_apps_obey_srrip_vs_grasp_ordering(payload):
    """At array-relative GRASP 0.15 (single-thread) NO app deviates from the
    SRRIP-vs-GRASP ordering — bfs (previously pinned as a frontier-streaming
    deviation under the multi-thread corpus) is now well-behaved, so the
    pinned set is empty."""
    deviating = set(payload["meta"]["deviating_apps"])
    assert deviating == set(), (
        f"unexpected SRRIP-vs-GRASP deviations: {deviating} "
        "(none expected at array-relative GRASP 0.15)"
    )


def test_non_pinned_apps_obey_ordering(payload):
    per_app = payload["meta"]["per_app"]
    for app, e in per_app.items():
        if app in PINNED_DEVIATING_APPS:
            continue
        d = e["srrip_minus_grasp_pp_oct"]
        assert d is not None and d <= ALLOW_SRRIP_SHALLOWER_BY_PP, (
            f"app {app!r}: SRRIP-GRASP delta {d:+.4f} > "
            f"{ALLOW_SRRIP_SHALLOWER_BY_PP} (SRRIP shallower than allowed)"
        )


def test_at_least_one_app_with_strictly_steeper_srrip(payload):
    per_app = payload["meta"]["per_app"]
    strict = [
        a for a, e in per_app.items()
        if e["srrip_minus_grasp_pp_oct"] is not None
        and e["srrip_minus_grasp_pp_oct"] < 0.0
    ]
    assert strict, (
        "no app has SRRIP strictly steeper than GRASP — global gate 72 "
        "would also be at risk; investigate"
    )


def test_verdict_pass(payload):
    assert payload["meta"]["verdict"] == "PASS", (
        f"verdict={payload['meta']['verdict']}, "
        f"checks={payload['meta']['verdict_checks']}"
    )


def test_source_points_to_gate68(payload):
    src = payload["meta"]["source"]
    assert src == "per_app_capacity_slope.json", (
        f"unexpected source: {src!r}"
    )


def test_apps_with_negative_delta_match_global_winner(payload):
    """Apps where SRRIP is steeper than GRASP should form the majority
    of the corpus (gate 72 reported strict steeper at GLOBAL median)."""
    per_app = payload["meta"]["per_app"]
    strict = sum(
        1 for e in per_app.values()
        if e["srrip_minus_grasp_pp_oct"] is not None
        and e["srrip_minus_grasp_pp_oct"] < 0.0
    )
    total = sum(
        1 for e in per_app.values()
        if e["srrip_minus_grasp_pp_oct"] is not None
    )
    assert strict * 2 >= total, (
        f"only {strict}/{total} apps have strictly steeper SRRIP; "
        f"global gate 72 ordering may not be representative"
    )
