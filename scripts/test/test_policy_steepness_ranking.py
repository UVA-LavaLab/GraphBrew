"""Pytest gate for per-policy final-octave steepness ranking (gate 81).

Locks the saturation-rank inversion at the absolute-magnitude level:
POPT and GRASP (oracle-aware) keep |final octave slope| small while
LRU and SRRIP (non-oracle) stay steep, with a strict ordering of
medians and a 2x oracle-to-non-oracle ratio gate.
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


def test_ranking_starts_with_popt():
    blob = _load()
    assert blob["ranking_by_median"][0] == "POPT"


def test_ranking_ends_with_non_oracle():
    blob = _load()
    assert blob["ranking_by_median"][-1] in {"LRU", "SRRIP"}


def test_popt_le_grasp_median():
    blob = _load()
    assert blob["checks"]["popt_le_grasp_median"]["ok"]


def test_grasp_le_lru_median():
    blob = _load()
    assert blob["checks"]["grasp_le_lru_median"]["ok"]


def test_popt_lt_srrip_median():
    blob = _load()
    assert blob["checks"]["popt_lt_srrip_median"]["ok"]


def test_oracle_aware_ceiling():
    blob = _load()
    assert blob["checks"]["oracle_aware_ceiling"]["ok"]


def test_non_oracle_floor():
    blob = _load()
    assert blob["checks"]["non_oracle_floor"]["ok"]


def test_oracle_half_of_non_oracle():
    blob = _load()
    assert blob["checks"]["oracle_half_of_non_oracle"]["ok"]


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
