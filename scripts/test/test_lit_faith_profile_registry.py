"""Tests for gate 256 — ECG profile registry."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GEN_PATH = ROOT / "scripts" / "experiments" / "ecg" / "lit_faith_profile_registry.py"
JSON_OUT = ROOT / "wiki" / "data" / "lit_faith_profile_registry.json"
WIKI_DATA = ROOT / "wiki" / "data"
MANIFEST = ROOT / "scripts" / "experiments" / "ecg" / "final_paper_manifest.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load("lit_faith_profile_registry_dyn", GEN_PATH)


@pytest.fixture(scope="module")
def audit(gen):
    return gen.audit()


# --- shape -----------------------------------------------------------

def test_generator_imports(gen):
    assert hasattr(gen, "audit")
    assert hasattr(gen, "PROFILE_NAME_RE")
    assert hasattr(gen, "STAGE_NAME_RE")
    assert hasattr(gen, "PLACEHOLDER_PROFILE_HINT")


def test_audit_active(audit):
    assert audit["status"] == "active"


def test_audit_zero_violations(audit):
    assert audit["violations"] == [], (
        f"violations: {audit['violations'][:5]}")


def test_audit_advertises_7_rules(audit):
    assert set(audit["rules"].keys()) == {
        "R1", "R2", "R3", "R4", "R5", "R6", "R7"}


# --- module invariants ----------------------------------------------

def test_profile_name_regex_strict(gen):
    assert gen.PROFILE_NAME_RE.match("rehearsal")
    assert gen.PROFILE_NAME_RE.match("final_replacement_v2")
    assert not gen.PROFILE_NAME_RE.match("Rehearsal")
    assert not gen.PROFILE_NAME_RE.match("final-replacement")
    assert not gen.PROFILE_NAME_RE.match("_rehearsal")
    assert not gen.PROFILE_NAME_RE.match("1_rehearsal")


def test_stage_name_regex_strict(gen):
    assert gen.STAGE_NAME_RE.match("01_cache_sim")
    assert gen.STAGE_NAME_RE.match("09b_sniper_smoke")
    assert gen.STAGE_NAME_RE.match("09g1_sniper_smoke")
    assert not gen.STAGE_NAME_RE.match("cache_sim_01")
    assert not gen.STAGE_NAME_RE.match("01-cache-sim")
    assert not gen.STAGE_NAME_RE.match("01_Cache_Sim")


def test_placeholder_hint_is_proper_word(gen):
    assert gen.PLACEHOLDER_PROFILE_HINT == "Placeholder"


# --- manifest integrity ---------------------------------------------

def test_manifest_exists():
    assert MANIFEST.is_file()


def test_manifest_declares_profiles():
    m = json.loads(MANIFEST.read_text())
    assert "profiles" in m
    assert isinstance(m["profiles"], dict)
    assert len(m["profiles"]) >= 10


def test_manifest_declares_stages():
    m = json.loads(MANIFEST.read_text())
    assert "stages" in m
    assert isinstance(m["stages"], list)
    assert len(m["stages"]) >= 10


def test_every_manifest_profile_is_snake_case(gen):
    m = json.loads(MANIFEST.read_text())
    for key in m["profiles"]:
        assert gen.PROFILE_NAME_RE.match(key), key


def test_every_manifest_stage_name_is_well_formed(gen):
    m = json.loads(MANIFEST.read_text())
    for stage in m["stages"]:
        assert gen.STAGE_NAME_RE.match(stage["name"]), stage["name"]


# --- per-rule live checks --------------------------------------------

def test_r1_stage_tokens_resolve(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "R1"]
    assert not bad, f"R1: {bad}"


def test_r2_descriptions_present(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "R2"]
    assert not bad, f"R2: {bad}"


def test_r3_no_dead_profiles(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "R3"]
    assert not bad, f"R3: {bad}"


def test_r4_no_typos_in_citations(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "R4"]
    assert not bad, f"R4: {bad}"


def test_r5_profile_names_snake_case(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "R5"]
    assert not bad, f"R5: {bad}"


def test_r6_stage_names_well_formed(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "R6"]
    assert not bad, f"R6: {bad}"


def test_r7_stage_profiles_non_empty_unique(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "R7"]
    assert not bad, f"R7: {bad}"


# --- totals + content ------------------------------------------------

def test_totals_present(audit):
    for k in ("manifest_profiles", "stages", "citations_total",
              "distinct_citations", "violations"):
        assert k in audit["totals"]
        assert isinstance(audit["totals"][k], int)


def test_stages_count_matches_audit(audit):
    m = json.loads(MANIFEST.read_text())
    assert audit["totals"]["stages"] == len(m["stages"])


def test_profiles_count_matches_audit(audit):
    m = json.loads(MANIFEST.read_text())
    assert audit["totals"]["manifest_profiles"] == len(m["profiles"])


def test_canonical_rehearsal_profile_present(audit):
    assert "rehearsal" in audit["profiles"]


def test_canonical_final_replacement_present(audit):
    assert "final_replacement" in audit["profiles"]


# --- generated artifact parity --------------------------------------

def test_json_artifact_exists():
    assert JSON_OUT.is_file()


def test_json_artifact_ends_with_newline():
    assert JSON_OUT.read_text().endswith("\n")


def test_json_artifact_matches_live_audit(gen):
    on_disk = json.loads(JSON_OUT.read_text())
    live = gen.audit()
    assert on_disk["totals"] == live["totals"]
    assert on_disk["profiles"] == live["profiles"]
    assert on_disk["stages"] == live["stages"]


def test_md_artifact_present():
    assert (WIKI_DATA / "lit_faith_profile_registry.md").is_file()


def test_csv_artifact_present():
    assert (WIKI_DATA / "lit_faith_profile_registry.csv").is_file()


# --- explicit sniffer for the snake-eating tail ----------------------

def test_rehearsal_profile_used_by_first_stages(audit):
    m = json.loads(MANIFEST.read_text())
    first_stage = m["stages"][0]
    assert "rehearsal" in first_stage["profiles"]


def test_no_profile_collision_with_python_keywords(audit):
    # ensure our snake_case is not collision-prone
    py_keywords = {"def", "class", "for", "while", "lambda", "import",
                   "from", "yield", "async", "await", "return"}
    for p in audit["profiles"]:
        assert p not in py_keywords


def test_citation_tokens_subset_of_profiles_when_not_violating(audit):
    # With 0 violations, every citation token must be a known profile.
    profiles = set(audit["profiles"])
    for tok in audit["citation_tokens"]:
        assert tok in profiles, f"citation token {tok!r} unresolved"
