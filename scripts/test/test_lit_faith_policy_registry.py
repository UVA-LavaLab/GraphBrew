"""Tests for gate 255 — cache-policy vocabulary registry."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GEN_PATH = ROOT / "scripts" / "experiments" / "ecg" / "lit_faith_policy_registry.py"
JSON_OUT = ROOT / "wiki" / "data" / "lit_faith_policy_registry.json"
WIKI_DATA = ROOT / "wiki" / "data"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load("lit_faith_policy_registry_dyn", GEN_PATH)


@pytest.fixture(scope="module")
def audit(gen):
    return gen.audit()


# --- shape -----------------------------------------------------------

def test_generator_imports(gen):
    assert hasattr(gen, "audit")
    assert hasattr(gen, "CANONICAL_POLICY_NAMES")
    assert hasattr(gen, "CANONICAL_ECG_ARMS")
    assert hasattr(gen, "CANONICAL_FOUR_TUPLE")
    assert hasattr(gen, "ANCHOR_TRIPLET")
    assert hasattr(gen, "VALID_FAMILIES")


def test_audit_returns_active(audit):
    assert audit["status"] == "active"


def test_audit_zero_violations(audit):
    assert audit["violations"] == [], (
        f"violations: {audit['violations'][:5]}")


def test_audit_advertises_9_rules(audit):
    assert set(audit["rules"].keys()) == {
        "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"}


def test_canonical_includes_baselines(gen):
    for name in ("LRU", "FIFO", "RANDOM", "LFU", "SRRIP"):
        assert name in gen.CANONICAL_POLICY_NAMES
        assert gen.CANONICAL_POLICY_NAMES[name]["family"] == "baseline"


def test_canonical_includes_graph_aware(gen):
    for name in ("GRASP", "POPT", "ECG"):
        assert name in gen.CANONICAL_POLICY_NAMES
        assert gen.CANONICAL_POLICY_NAMES[name]["family"] == "graph_aware"


def test_canonical_count_is_8(gen):
    assert len(gen.CANONICAL_POLICY_NAMES) == 8


def test_four_tuple_locked(gen):
    assert gen.CANONICAL_FOUR_TUPLE == ("LRU", "SRRIP", "GRASP", "POPT")


def test_anchor_triplet_locked(gen):
    assert gen.ANCHOR_TRIPLET == ("GRASP", "LRU", "SRRIP")


def test_ecg_arms_count(gen):
    assert len(gen.CANONICAL_ECG_ARMS) >= 8


def test_every_ecg_arm_has_parent_and_purpose(gen):
    for arm, meta in gen.CANONICAL_ECG_ARMS.items():
        assert "parent" in meta
        assert "purpose" in meta and len(meta["purpose"]) >= 20


def test_every_canonical_policy_has_paper_label(gen):
    for name, meta in gen.CANONICAL_POLICY_NAMES.items():
        assert "paper_label" in meta
        assert isinstance(meta["paper_label"], str)
        assert meta["paper_label"].strip()


def test_aliases_are_lists_of_strings(gen):
    for name, meta in gen.CANONICAL_POLICY_NAMES.items():
        assert isinstance(meta["aliases"], list)
        for a in meta["aliases"]:
            assert isinstance(a, str)


def test_aliases_dont_collide_with_canonical(gen):
    canonical = set(gen.CANONICAL_POLICY_NAMES)
    for name, meta in gen.CANONICAL_POLICY_NAMES.items():
        for a in meta["aliases"]:
            assert a not in canonical, (
                f"alias {a!r} of {name} collides with another canonical token")


# --- per-rule live checks --------------------------------------------

def test_p1_no_rogue_tokens(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "P1"]
    assert not bad, f"P1: {bad}"


def test_p2_no_duplicates(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "P2"]
    assert not bad, f"P2: {bad}"


def test_p3_all_policies_decomposes(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "P3"]
    assert not bad, f"P3: {bad}"


def test_p4_canonical_well_formed(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "P4"]
    assert not bad, f"P4: {bad}"


def test_p5_paper_labels_unique(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "P5"]
    assert not bad, f"P5: {bad}"


def test_p6_four_tuples_are_permutations(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "P6"]
    assert not bad, f"P6: {bad}"


def test_p7_three_tuples_subset(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "P7"]
    assert not bad, f"P7: {bad}"


def test_p8_no_aliases_harvested(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "P8"]
    assert not bad, f"P8: {bad}"


def test_p9_ecg_arms_well_formed(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "P9"]
    assert not bad, f"P9: {bad}"


# --- totals + content ------------------------------------------------

def test_totals_present(audit):
    for k in ("canonical_tokens", "ecg_arms", "harvested_POLICIES",
              "harvested_ALL", "violations"):
        assert k in audit["totals"]
        assert isinstance(audit["totals"][k], int)


def test_harvested_policies_nonempty(audit):
    assert audit["totals"]["harvested_POLICIES"] >= 10


def test_canonical_tokens_in_audit_match_module(audit, gen):
    assert set(audit["canonical_tokens"]) == set(gen.CANONICAL_POLICY_NAMES.keys())


def test_ecg_arms_in_audit_match_module(audit, gen):
    assert set(audit["ecg_arms"]) == set(gen.CANONICAL_ECG_ARMS.keys())


# --- generated artifact parity --------------------------------------

def test_json_artifact_exists():
    assert JSON_OUT.is_file()


def test_json_artifact_ends_with_newline():
    text = JSON_OUT.read_text()
    assert text.endswith("\n"), (
        "Final-newline JSON requirement; gen must write json.dumps + '\\n'")


def test_json_artifact_is_valid():
    json.loads(JSON_OUT.read_text())


def test_json_artifact_matches_live_audit(gen):
    on_disk = json.loads(JSON_OUT.read_text())
    live = gen.audit()
    assert on_disk["totals"] == live["totals"]
    assert on_disk["canonical_tokens"] == live["canonical_tokens"]
    assert on_disk["ecg_arms"] == live["ecg_arms"]


def test_md_artifact_present():
    md = WIKI_DATA / "lit_faith_policy_registry.md"
    assert md.is_file()


def test_csv_artifact_present():
    csv_path = WIKI_DATA / "lit_faith_policy_registry.csv"
    assert csv_path.is_file()
