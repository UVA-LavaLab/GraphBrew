"""Tests for gate 254 — wiki/data bidirectional registry."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GEN_PATH = ROOT / "scripts" / "experiments" / "ecg" / "lit_faith_wiki_registry.py"
JSON_OUT = ROOT / "wiki" / "data" / "lit_faith_wiki_registry.json"
WIKI_DATA = ROOT / "wiki" / "data"
CATALOG_JSON = WIKI_DATA / "artifact_catalog.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load("lit_faith_wiki_registry_dyn", GEN_PATH)


@pytest.fixture(scope="module")
def audit(gen):
    return gen.audit()


# --- shape -----------------------------------------------------------

def test_generator_imports(gen):
    assert hasattr(gen, "audit")
    assert hasattr(gen, "SELF_REFERENTIAL")
    assert hasattr(gen, "ALLOWED_AUXILIARY")
    assert hasattr(gen, "WIKI_DATA")
    assert hasattr(gen, "CATALOG_JSON")


def test_audit_returns_active(audit):
    assert audit["status"] == "active"


def test_audit_zero_violations(audit):
    assert audit["violations"] == [], (
        f"violations: {audit['violations'][:5]}")


def test_audit_advertises_8_rules(audit):
    assert set(audit["rules"].keys()) == {
        "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8"}


def test_self_referential_includes_catalog(gen):
    assert "wiki/data/artifact_catalog.json" in gen.SELF_REFERENTIAL


def test_self_referential_is_minimal(gen):
    # the catalog cannot catalog itself reflectively; everything else
    # must be in the catalog or the auxiliary allow-list. Lock the
    # exception count low.
    assert len(gen.SELF_REFERENTIAL) <= 2, (
        f"self-referential set should be minimal; got "
        f"{sorted(gen.SELF_REFERENTIAL)}")


def test_auxiliary_entries_have_required_fields(gen):
    for art, meta in gen.ALLOWED_AUXILIARY.items():
        assert art.startswith("wiki/data/") and art.endswith(".json")
        assert "parent_id" in meta and meta["parent_id"], (
            f"{art}: missing parent_id")
        assert "purpose" in meta and len(meta["purpose"]) >= 20, (
            f"{art}: purpose must be a non-trivial documented string")


# --- per-rule live checks --------------------------------------------

def test_w1_every_json_accounted(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "W1"]
    assert not bad, f"W1: {bad}"


def test_w2_no_ghost_entries(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "W2"]
    assert not bad, f"W2: {bad}"


def test_w3_entries_have_non_empty_fields(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "W3"]
    assert not bad, f"W3: {bad}"


def test_w4_paths_exist(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "W4"]
    assert not bad, f"W4: {bad}"


def test_w5_sibling_md_exists(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "W5"]
    assert not bad, f"W5: {bad}"


def test_w6_ids_unique(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "W6"]
    assert not bad, f"W6: {bad}"


def test_w7_artifact_paths_unique(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "W7"]
    assert not bad, f"W7: {bad}"


def test_w8_auxiliary_parent_ids_valid(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "W8"]
    assert not bad, f"W8: {bad}"


# --- totals + content ------------------------------------------------

def test_totals_present(audit):
    t = audit["totals"]
    for k in ("wiki_json_files", "catalog_entries", "auxiliary_entries",
              "self_referential", "violations"):
        assert k in t
        assert isinstance(t[k], int)


def test_wiki_json_files_nonempty(audit):
    assert audit["totals"]["wiki_json_files"] > 0


def test_catalog_entries_nonempty(audit):
    assert audit["totals"]["catalog_entries"] > 0


def test_wiki_files_are_paths(audit):
    for f in audit["wiki_json_files"]:
        assert f.startswith("wiki/data/")
        assert f.endswith(".json")


def test_auxiliary_list_matches_module_constant(audit, gen):
    aux_arts = {a["artifact"] for a in audit["auxiliary"]}
    assert aux_arts == set(gen.ALLOWED_AUXILIARY.keys())


def test_violations_are_dicts_with_rule(audit):
    for v in audit["violations"]:
        assert isinstance(v, dict)
        assert "rule" in v and "msg" in v


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
    assert on_disk["status"] == live["status"]
    assert on_disk["totals"] == live["totals"]


# --- catalog consistency cross-checks --------------------------------

def test_catalog_includes_self(audit):
    """The gate-254 generator should appear in the catalog (registry
    must register itself, since W1 demands all wiki/data/*.json are
    catalogued)."""
    cat = json.loads(CATALOG_JSON.read_text())["entries"]
    arts = {e["artifact"] for e in cat}
    assert "wiki/data/lit_faith_wiki_registry.json" in arts


def test_catalog_self_entry_has_correct_paths(audit):
    cat = json.loads(CATALOG_JSON.read_text())["entries"]
    by_id = {e["id"]: e for e in cat}
    entry = by_id.get("lit_faith_wiki_registry")
    assert entry is not None
    assert entry["generator"] == (
        "scripts/experiments/ecg/lit_faith_wiki_registry.py")
    assert entry["gate"] == "scripts/test/test_lit_faith_wiki_registry.py"
    assert entry["artifact"] == "wiki/data/lit_faith_wiki_registry.json"


def test_auxiliary_files_actually_exist(gen):
    """The auxiliary allow-list documents real files, not hypothetical
    ones — auxiliary entries are escape hatches for *existing* postfix
    artifacts, not promises of future ones."""
    for art in gen.ALLOWED_AUXILIARY:
        assert (ROOT / art).is_file(), (
            f"auxiliary allow-list entry refers to missing file: {art}")


def test_auxiliary_parent_ids_in_catalog(gen):
    cat = json.loads(CATALOG_JSON.read_text())["entries"]
    ids = {e["id"] for e in cat}
    for art, meta in gen.ALLOWED_AUXILIARY.items():
        assert meta["parent_id"] in ids, (
            f"{art}: parent_id {meta['parent_id']!r} not in catalog")


def test_self_referential_files_exist(gen):
    for art in gen.SELF_REFERENTIAL:
        assert (ROOT / art).is_file()


def test_md_artifact_present():
    md = WIKI_DATA / "lit_faith_wiki_registry.md"
    assert md.is_file()


def test_csv_artifact_present():
    csv_path = WIKI_DATA / "lit_faith_wiki_registry.csv"
    assert csv_path.is_file()
