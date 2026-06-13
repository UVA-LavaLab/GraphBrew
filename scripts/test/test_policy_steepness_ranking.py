"""Pytest gate for per-policy final-octave steepness ranking (gate 81).

Locks the charge-invariant steepness ordering: GRASP (degree heuristic)
is the FLATTEST policy (cache-insensitive hot-set pinning) and LRU
(blind recency) is the STEEPEST (most cache-sensitive), with a material
spread between them. Charged P-OPT is mid-pack -- no longer flat -- so
the gate no longer asserts POPT-is-flat (that was a multi-thread +
uncharged-P-OPT artifact).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

JSON_PATH = Path("wiki/data/policy_steepness_ranking.json")
MD_PATH = Path("wiki/data/policy_steepness_ranking.md")


def _load():
    if not JSON_PATH.exists():
        pytest.skip(f"{JSON_PATH} not generated yet")
    return json.loads(JSON_PATH.read_text())


def test_schema_tag():
    blob = _load()
    assert blob["schema"] == "policy_steepness_ranking/v1"


def test_apps_and_policies_present():
    blob = _load()
    assert set(blob["meta"]["apps"]) == {"bc", "bfs", "cc", "pr", "sssp"}
    assert set(blob["meta"]["policies"]) == {"POPT", "GRASP", "LRU", "SRRIP"}


def test_per_policy_has_all_5_apps():
    blob = _load()
    for pol in blob["meta"]["policies"]:
        assert blob["per_policy"][pol]["n"] == 5


def test_ranking_starts_with_grasp():
    blob = _load()
    assert blob["ranking_by_median"][0] == "GRASP"


def test_ranking_ends_with_lru():
    blob = _load()
    assert blob["ranking_by_median"][-1] == "LRU"


def test_grasp_is_flattest():
    blob = _load()
    assert blob["checks"]["grasp_is_flattest"]["ok"]


def test_lru_is_steepest():
    blob = _load()
    assert blob["checks"]["lru_is_steepest"]["ok"]


def test_grasp_le_lru_median():
    blob = _load()
    assert blob["checks"]["grasp_le_lru_median"]["ok"]


def test_steepness_spread():
    blob = _load()
    assert blob["checks"]["steepness_spread"]["ok"]


def test_popt_min_saturates():
    blob = _load()
    assert blob["checks"]["popt_min_saturates"]["ok"]


def test_verdict_pass():
    blob = _load()
    assert blob["verdict_ok"] is True


def test_md_renders_verdict():
    if not MD_PATH.exists():
        pytest.skip(f"{MD_PATH} not generated yet")
    txt = MD_PATH.read_text()
    assert "Per-policy final-octave steepness ranking" in txt
    assert "**Verdict:** PASS" in txt
