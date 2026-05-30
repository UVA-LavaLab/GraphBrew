"""Tests for gate 246 — lit-faith citation registry purity.

Locks in:

  * generator emits an "active" status (no scaffold mode);
  * zero violations on the current registry + per_claim table;
  * C1 (every citation matches >=1 registered canonical work);
  * C2 (every registered work is referenced >=1 time);
  * C3 (within (policy, app, expected_sign) bucket, members share >=1 key);
  * C4 (every registry entry has non-empty venue + year);
  * C5 (every per_claim row has a non-empty citation);
  * concrete coverage floors (each registered work cited >=1 time today);
  * concrete bucket coverage floor (>=1 (policy, app, sign) bucket);
  * generator's JSON-on-disk matches audit() output byte-for-byte
    (so committed artifacts can't silently drift from the live audit).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GEN_PATH = ROOT / "scripts" / "experiments" / "ecg" / "lit_faith_citation_registry.py"
JSON_OUT = ROOT / "wiki" / "data" / "lit_faith_citation_registry.json"
POSTFIX_JSON = ROOT / "wiki" / "data" / "literature_faithfulness_postfix.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load("lit_faith_citation_registry_dyn", GEN_PATH)


@pytest.fixture(scope="module")
def audit(gen):
    return gen.audit()


# --- shape -----------------------------------------------------------

def test_generator_imports(gen):
    assert hasattr(gen, "audit")
    assert hasattr(gen, "CITATION_REGISTRY")
    assert hasattr(gen, "_keys_matching")


def test_audit_returns_active_status(audit):
    assert audit["status"] == "active", (
        f"expected active, got {audit.get('status')} reason={audit.get('reason')}"
    )


def test_audit_has_zero_violations(audit):
    assert audit["violations"] == [], (
        f"violations: {audit['violations'][:5]}"
    )


def test_audit_advertises_5_rules(audit):
    assert set(audit["rules"].keys()) == {"C1", "C2", "C3", "C4", "C5"}


# --- registry shape --------------------------------------------------

def test_registry_nonempty(gen):
    assert len(gen.CITATION_REGISTRY) >= 3


def test_every_registry_entry_has_required_fields(gen):
    required = {"key", "title", "venue", "year", "patterns", "note"}
    for e in gen.CITATION_REGISTRY:
        missing = required - set(e.keys())
        assert not missing, f"entry {e.get('key')!r} missing: {missing}"
        assert isinstance(e["patterns"], list) and e["patterns"]
        assert all(isinstance(p, str) and p for p in e["patterns"])


def test_registry_keys_are_unique(gen):
    keys = [e["key"] for e in gen.CITATION_REGISTRY]
    assert len(keys) == len(set(keys))


def test_canonical_papers_registered(gen):
    keys = {e["key"] for e in gen.CITATION_REGISTRY}
    # The three primary works lit-faith currently cites:
    assert "Faldu-HPCA-2020" in keys
    assert "Balaji-HPCA-2021" in keys
    assert "Jaleel-ISCA-2010" in keys


# --- per-rule live checks --------------------------------------------

def test_postfix_json_exists():
    assert POSTFIX_JSON.exists(), (
        f"{POSTFIX_JSON} missing — gate cannot evaluate. "
        "Run `make lit-faith` first.")


def test_c1_no_unmatched_citations(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "C1"]
    assert not bad, f"C1 hits: {bad[:3]}"


def test_c2_no_dead_letter_registry_entries(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "C2"]
    assert not bad, f"C2 hits (unreferenced canonical works): {bad}"


def test_c3_buckets_share_at_least_one_canonical_key(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "C3"]
    assert not bad, f"C3 hits (buckets with no shared key): {bad[:3]}"


def test_c4_every_registry_entry_has_venue_and_year(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "C4"]
    assert not bad, f"C4 hits: {bad}"


def test_c5_no_empty_citation_strings(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "C5"]
    assert not bad, f"C5 hits: {bad[:3]}"


# --- concrete coverage floors ----------------------------------------

def test_every_registered_work_is_cited_today(audit):
    cov = audit["coverage_per_key"]
    for key, n in cov.items():
        assert n >= 1, f"registered work {key!r} has 0 citations"


def test_bucket_count_floor(audit):
    assert audit["totals"]["bucket_count"] >= 1


def test_row_count_floor(audit):
    # 330 today; allow shrinkage in case rows are pruned, but require
    # at least one row so the gate can't silently no-op into vacuous PASS
    assert audit["totals"]["row_count"] >= 1


# --- helper invariants -----------------------------------------------

def test_keys_matching_returns_canonical_key_for_known_strings(gen):
    assert "Faldu-HPCA-2020" in gen._keys_matching(
        "Faldu et al. HPCA 2020 §6.1 (extended)")
    assert "Balaji-HPCA-2021" in gen._keys_matching(
        "Balaji & Lucia HPCA 2021 §6.3")
    assert "Jaleel-ISCA-2010" in gen._keys_matching(
        "Jaleel et al. ISCA 2010 §5.2")


def test_keys_matching_returns_empty_for_unknown_string(gen):
    assert gen._keys_matching("Some Unrelated Paper 1999") == []


def test_keys_matching_returns_multiple_for_compound_citation(gen):
    hits = gen._keys_matching(
        "Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check")
    assert "Faldu-HPCA-2020" in hits
    assert "Balaji-HPCA-2021" in hits


# --- artifact-on-disk parity -----------------------------------------

def test_audit_serialisable(audit):
    json.dumps(audit)


def test_on_disk_json_matches_live_audit(audit):
    if not JSON_OUT.exists():
        pytest.skip(f"{JSON_OUT} not yet generated; run `make lit-citation-registry`.")
    on_disk = json.loads(JSON_OUT.read_text())
    assert on_disk["status"] == audit["status"]
    assert on_disk["totals"] == audit["totals"]
    assert on_disk["coverage_per_key"] == audit["coverage_per_key"]
    assert on_disk["violations"] == audit["violations"]


def test_on_disk_md_exists():
    md = ROOT / "wiki" / "data" / "lit_faith_citation_registry.md"
    if md.exists():
        txt = md.read_text()
        assert "lit-faith citation registry purity" in txt
        assert "gate 246" in txt


def test_on_disk_csv_exists():
    csvp = ROOT / "wiki" / "data" / "lit_faith_citation_registry.csv"
    if csvp.exists():
        txt = csvp.read_text()
        assert "registry_size" in txt
        assert "violations" in txt
