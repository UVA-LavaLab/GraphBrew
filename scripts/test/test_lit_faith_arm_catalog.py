"""Pytest for gate 261 — ECG arm catalog registry."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN = REPO_ROOT / "scripts" / "experiments" / "ecg" / "lit_faith_arm_catalog.py"


def _load():
    spec = importlib.util.spec_from_file_location("gate261", GEN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gate261"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gate():
    return _load()


@pytest.fixture(scope="module")
def data(gate):
    return gate.audit()


def test_module_loads(gate):
    assert hasattr(gate, "audit")
    assert hasattr(gate, "CANONICAL_BASELINES")
    assert hasattr(gate, "main")


def test_baselines(gate):
    assert gate.CANONICAL_BASELINES == ("LRU", "SRRIP", "GRASP", "POPT")


def test_active(data):
    assert data["status"] == "active"


def test_no_violations(data):
    assert data["violations"] == [], data["violations"]


def test_top_level_keys(data):
    for k in (
        "status", "n_paper_policies", "n_paper_charged",
        "n_registry_arms", "n_ablations", "n_adaptive_selectors",
        "paper_to_registry", "policy_order", "ablations",
        "adaptive_selectors", "registry_arms", "rules", "violations",
    ):
        assert k in data, f"missing key {k}"


# --- paper-policy roster -------------------------------------------------

def test_nine_paper_policies(data):
    assert data["n_paper_policies"] == 9


def test_baselines_in_policy_order(data):
    for b in ("LRU", "SRRIP", "GRASP", "POPT"):
        assert b in data["policy_order"]


def test_paper_charged_count(data):
    # POPT_CHARGED + ECG_DBG_PRIMARY_CHARGED
    assert data["n_paper_charged"] == 2


def test_charged_paper_policies(data):
    charged = [p for p in data["policy_order"] if p.endswith("_CHARGED")]
    assert sorted(charged) == ["ECG_DBG_PRIMARY_CHARGED", "POPT_CHARGED"]


# --- A1 paper→registry mapping ------------------------------------------

def test_paper_to_registry_keys(data):
    m = data["paper_to_registry"]
    # POPT_CHARGED stays itself; ECG_* gets the colon namespace
    assert m["POPT_CHARGED"] == "POPT_CHARGED"
    assert m["ECG_DBG_ONLY"] == "ECG:DBG_ONLY"
    assert m["ECG_DBG_PRIMARY"] == "ECG:DBG_PRIMARY"
    assert m["ECG_DBG_PRIMARY_CHARGED"] == "ECG:DBG_PRIMARY_CHARGED"
    assert m["ECG_POPT_PRIMARY"] == "ECG:POPT_PRIMARY"


def test_paper_to_registry_no_baselines(data):
    # baselines should NOT appear in the mapping
    for b in ("LRU", "SRRIP", "GRASP", "POPT"):
        assert b not in data["paper_to_registry"]


def test_paper_to_registry_keys_all_exist(data):
    arm_set = set(data["registry_arms"].keys())
    for paper, reg in data["paper_to_registry"].items():
        assert reg in arm_set, f"{paper}→{reg} missing from registry"


# --- A4 ablation policies -----------------------------------------------

def test_ablation_count(data):
    # 4 cache_alone + 7 ecg_replacement + 2 pfx_only + 3 combined
    assert data["n_ablations"] == 16


def test_ablation_policies_all_known(data):
    arm_set = set(data["registry_arms"].keys())
    baselines = {"LRU", "SRRIP", "GRASP", "POPT"}
    for a in data["ablations"]:
        assert a["policy"] in baselines | arm_set


def test_ablation_groups(data):
    groups = {a["group"] for a in data["ablations"]}
    assert groups == {"cache_alone", "ecg_replacement", "pfx_only",
                      "combined"}


@pytest.mark.parametrize("label,policy", [
    ("LRU_cache_only", "LRU"),
    ("SRRIP_cache_only", "SRRIP"),
    ("GRASP_DBG_only", "GRASP"),
    ("POPT_only", "POPT"),
    ("ECG_DBG_only", "ECG:DBG_ONLY"),
    ("ECG_POPT_primary", "ECG:POPT_PRIMARY"),
    ("ECG_DBG_POPT", "ECG:DBG_PRIMARY"),
    ("ECG_POPT_TIE", "ECG:POPT_TIE"),
    ("ECG_EMBEDDED", "ECG:ECG_EMBEDDED"),
    ("ECG_EPOCH_EMBEDDED", "ECG:ECG_EPOCH_EMBEDDED"),
    ("ECG_COMBINED", "ECG:ECG_COMBINED"),
])
def test_specific_ablation(data, label, policy):
    by_label = {a["label"]: a for a in data["ablations"]}
    assert by_label[label]["policy"] == policy


# --- A5 selector candidates ---------------------------------------------

def test_two_adaptive_selectors(data):
    assert data["n_adaptive_selectors"] == 2


def test_selector_candidates_real(data):
    labels = {a["label"] for a in data["ablations"]}
    for sel in data["adaptive_selectors"]:
        for cand in sel["candidates"]:
            assert cand in labels


def test_selector_names(data):
    names = sorted(s["label"] for s in data["adaptive_selectors"])
    assert names == ["ECG_ADAPTIVE_NO_FULL_POPT", "ECG_ADAPTIVE_ORACLE"]


# --- A6 no duplicate labels ---------------------------------------------

def test_no_duplicate_ablation_labels(data):
    labels = [a["label"] for a in data["ablations"]]
    assert len(labels) == len(set(labels))


# --- A7 registry arm coverage -------------------------------------------

def test_nine_registry_arms(data):
    assert data["n_registry_arms"] == 9


def test_charged_arm_via_parent(data):
    arms = data["registry_arms"]
    # ECG:DBG_PRIMARY_CHARGED itself has 0 direct ablations but is
    # satisfied by parent ECG:DBG_PRIMARY having ≥1 ablation.
    assert arms["ECG:DBG_PRIMARY_CHARGED"]["ablation_count"] == 0
    assert arms["ECG:DBG_PRIMARY"]["ablation_count"] >= 1


# --- artifact round-trip ------------------------------------------------

def test_round_trip(gate, tmp_path: Path):
    j = tmp_path / "x.json"
    m = tmp_path / "x.md"
    c = tmp_path / "x.csv"
    rc = subprocess.run(
        [sys.executable, str(GEN),
         "--json-out", str(j),
         "--md-out", str(m),
         "--csv-out", str(c)],
        capture_output=True, text=True,
    )
    assert rc.returncode == 0, rc.stderr
    p = json.loads(j.read_text())
    assert p["violations"] == []
    txt = m.read_text()
    assert "Gate 261" in txt
    assert txt.endswith("\n") and not txt.endswith("\n\n")
    csv_txt = c.read_text()
    assert "paper_policy" in csv_txt
    assert "ablation" in csv_txt
    assert "selector" in csv_txt
