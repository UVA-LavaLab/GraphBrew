#!/usr/bin/env python3
"""Pytest gate 249 — graph-family map full-coverage audit."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GEN_PATH = ROOT / "scripts" / "experiments" / "ecg" / "lit_faith_graph_family.py"
WIKI_DATA = ROOT / "wiki" / "data"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "lit_faith_graph_family", GEN_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lit_faith_graph_family"] = mod
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module()


# ------------------------------------------------------------- audit --

@pytest.fixture(scope="module")
def audit():
    return MOD.audit()


def test_audit_runs(audit):
    assert audit["status"] == "active"


def test_no_violations(audit):
    assert audit["violations"] == [], audit["violations"]


def test_copies_harvested(audit):
    # We expect at least the 5 in-universe + 7 out-of-universe copies
    # that are present today. New harvested copies are encouraged,
    # so we use a floor, not an exact match.
    assert audit["copy_count"] >= 12, audit["copy_count"]
    assert audit["files_with_copies"] >= 12, audit["files_with_copies"]


def test_canonical_size_today(audit):
    # Today's currently-shipped corpus: 8 graphs across 5 families.
    assert audit["canonical_size"] == 8
    assert set(audit["allowed_families"]) == {
        "social", "web", "citation", "road", "mesh"}


# --------------------------------------------------- registry contract --

def test_canonical_keys_today():
    assert set(MOD.CANONICAL_GRAPH_FAMILY.keys()) == {
        "email-Eu-core", "soc-pokec", "soc-LiveJournal1", "com-orkut",
        "web-Google", "cit-Patents", "roadNet-CA", "delaunay_n19",
    }


def test_canonical_family_distribution():
    from collections import Counter
    c = Counter(MOD.CANONICAL_GRAPH_FAMILY.values())
    # 4 social, 1 web, 1 citation, 1 road, 1 mesh
    assert c == {"social": 4, "web": 1, "citation": 1, "road": 1, "mesh": 1}


def test_allowed_families_set():
    assert MOD.ALLOWED_FAMILIES == {
        "social", "web", "citation", "road", "mesh"}


def test_reserved_future_keys_disjoint_from_canonical():
    assert (set(MOD.RESERVED_FUTURE_KEYS) & set(MOD.CANONICAL_GRAPH_FAMILY)
            == set())


def test_known_graph_names_is_union():
    assert MOD.KNOWN_GRAPH_NAMES == (
        set(MOD.CANONICAL_GRAPH_FAMILY) | set(MOD.RESERVED_FUTURE_KEYS))


# ---------------------------------------------------- gate-107 universe --

def test_gate_107_universe_files_exist():
    for rel in MOD.GATE_107_UNIVERSE:
        assert (ROOT.parent / rel).exists() if False else (ROOT / rel).exists(), rel


def test_gate_107_universe_size_today():
    assert len(MOD.GATE_107_UNIVERSE) == 7


# ------------------------------------------------------------- rules --

# F2/F3 — every harvested copy is consistent with canonical
def test_f2_no_unknown_graph_in_any_copy(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "F2"]
    assert bad == [], bad


def test_f3_no_value_drift_in_any_copy(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "F3"]
    assert bad == [], bad


# F4 — canonical map sanity
def test_f4_canonical_passes(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "F4"]
    assert bad == [], bad


# F5 — no unguarded FULL copy
def test_f5_no_unguarded_full_copy(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "F5"]
    assert bad == [], bad


# --------------------------------------------------- helper invariants --

def test_self_skip_paths_exist():
    # The generator and its pytest must exist (so the SELF_SKIP set
    # actually skips them and doesn't silently miss real audits).
    for rel in MOD.SELF_SKIP:
        assert (ROOT / rel).exists(), rel


def test_harvester_rejects_random_str_to_str_dict():
    import ast
    tree = ast.parse('X = {"foo": "bar", "baz": "qux"}')
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Dict)]
    assert MOD._is_graph_family_dict(nodes[0]) is None


def test_harvester_rejects_unknown_family_value():
    import ast
    tree = ast.parse('X = {"email-Eu-core": "WAT", "web-Google": "WAT"}')
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Dict)]
    assert MOD._is_graph_family_dict(nodes[0]) is None


def test_harvester_accepts_canonical_shaped_dict():
    import ast
    tree = ast.parse(
        'X = {"email-Eu-core": "social", "web-Google": "web"}')
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Dict)]
    out = MOD._is_graph_family_dict(nodes[0])
    assert out == {"email-Eu-core": "social", "web-Google": "web"}


def test_harvester_rejects_single_entry():
    import ast
    tree = ast.parse('X = {"email-Eu-core": "social"}')
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Dict)]
    assert MOD._is_graph_family_dict(nodes[0]) is None


# --------------------------------------------------------- artifact parity --

def test_artifact_parity_when_present():
    art = WIKI_DATA / "lit_faith_graph_family.json"
    if not art.exists():
        pytest.skip("artifact not generated yet")
    on_disk = json.loads(art.read_text())
    live = MOD.audit()
    assert on_disk["canonical_size"] == live["canonical_size"]
    assert on_disk["copy_count"] == live["copy_count"]
    assert on_disk["files_with_copies"] == live["files_with_copies"]
    assert len(on_disk["violations"]) == len(live["violations"])
