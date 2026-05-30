"""Tests for gate 251 — L3 cache-size registry.

Locks in:

  * generator emits an "active" status (no skip on a healthy tree);
  * zero violations on the current registry + AST-harvested constants;
  * L1 (every harvested token is canonical);
  * L2 (every PAPER_L3 tuple == ANCHOR_TRIPLET);
  * L3 (every L3_MB dict matches canonical MB);
  * L4 (every L3_BYTES dict matches canonical bytes);
  * L5 (canonical role + sub_tier vocabularies are valid);
  * L6 (every anchor token appears in some harvested PAPER_L3);
  * L7 (harvested PAPER_L3-shaped constants agree across files);
  * concrete floors (>=11 canonical entries, >=30 PAPER_L3-shaped
    tuples harvested, >=3 L3_MB dicts, >=2 L3_BYTES dicts, >=10
    files harvested);
  * specific contract: ANCHOR_TRIPLET == ("1MB", "4MB", "8MB");
  * canonical byte arithmetic spot-checks (1MB=1048576, 4kB=4096);
  * AST helper spot-checks (constant-fold of ``4 * 1024``);
  * generator's JSON-on-disk matches audit() output.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GEN_PATH = ROOT / "scripts" / "experiments" / "ecg" / "lit_faith_l3_registry.py"
JSON_OUT = ROOT / "wiki" / "data" / "lit_faith_l3_registry.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load("lit_faith_l3_registry_dyn", GEN_PATH)


@pytest.fixture(scope="module")
def audit(gen):
    return gen.audit()


# --- shape -----------------------------------------------------------

def test_generator_imports(gen):
    assert hasattr(gen, "audit")
    assert hasattr(gen, "CANONICAL_L3_TIERS")
    assert hasattr(gen, "ANCHOR_TRIPLET")
    assert hasattr(gen, "VALID_ROLES")
    assert hasattr(gen, "VALID_SUB_TIERS")
    assert hasattr(gen, "_harvest_file")
    assert hasattr(gen, "_fold_int")
    assert hasattr(gen, "_fold_float")


def test_audit_returns_active_status(audit):
    assert audit["status"] == "active", (
        f"expected active, got {audit.get('status')}")


def test_audit_has_zero_violations(audit):
    assert audit["violations"] == [], (
        f"violations: {audit['violations'][:5]}")


def test_audit_advertises_7_rules(audit):
    assert set(audit["rules"].keys()) == {
        "L1", "L2", "L3", "L4", "L5", "L6", "L7"}


# --- canonical shape -------------------------------------------------

def test_anchor_triplet_is_1_4_8_mb(gen):
    assert gen.ANCHOR_TRIPLET == ("1MB", "4MB", "8MB")


def test_canonical_contains_anchor_tokens(gen):
    for tok in gen.ANCHOR_TRIPLET:
        assert tok in gen.CANONICAL_L3_TIERS, (
            f"anchor token {tok!r} missing from canonical")


def test_canonical_byte_arithmetic_spotcheck(gen):
    c = gen.CANONICAL_L3_TIERS
    assert c["1MB"]["bytes"] == 1024 * 1024
    assert c["4MB"]["bytes"] == 4 * 1024 * 1024
    assert c["8MB"]["bytes"] == 8 * 1024 * 1024
    assert c["4kB"]["bytes"] == 4 * 1024
    assert c["256kB"]["bytes"] == 256 * 1024


def test_canonical_mb_scaling_spotcheck(gen):
    c = gen.CANONICAL_L3_TIERS
    assert c["1MB"]["mb"] == 1.0
    assert c["4MB"]["mb"] == 4.0
    assert c["8MB"]["mb"] == 8.0


def test_canonical_every_entry_has_required_fields(gen):
    required = {"bytes", "mb", "role", "sub_tier"}
    for tok, info in gen.CANONICAL_L3_TIERS.items():
        missing = required - set(info.keys())
        assert not missing, (
            f"{tok}: missing canonical fields: {missing}")
        assert info["role"] in gen.VALID_ROLES, (
            f"{tok}: bad role {info['role']!r}")
        assert info["sub_tier"] in gen.VALID_SUB_TIERS, (
            f"{tok}: bad sub_tier {info['sub_tier']!r}")
        assert isinstance(info["bytes"], int) and info["bytes"] > 0


def test_canonical_size_floor(gen):
    assert len(gen.CANONICAL_L3_TIERS) >= 11


# --- per-rule live checks --------------------------------------------

def test_l1_no_unknown_tokens(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "L1"]
    assert not bad, f"L1 hits: {bad[:5]}"


def test_l2_no_anchor_tuple_drift(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "L2"]
    assert not bad, f"L2 hits: {bad[:5]}"


def test_l3_no_mb_dict_drift(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "L3"]
    assert not bad, f"L3 hits: {bad[:5]}"


def test_l4_no_bytes_dict_drift(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "L4"]
    assert not bad, f"L4 hits: {bad[:5]}"


def test_l5_canonical_vocab_is_valid(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "L5"]
    assert not bad, f"L5 hits: {bad}"


def test_l6_every_anchor_token_appears(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "L6"]
    assert not bad, f"L6 hits: {bad}"


def test_l7_no_cross_file_anchor_disagreement(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "L7"]
    assert not bad, f"L7 hits: {bad}"


# --- concrete floors -------------------------------------------------

def test_at_least_10_files_harvested(audit):
    assert audit["totals"]["files_with_l3"] >= 10


def test_at_least_30_paper_l3_tuples_harvested(audit):
    # Today: 43.  >= 30 leaves headroom for refactors.
    assert audit["totals"]["anchor_tuples"] >= 30


def test_at_least_3_mb_dicts_harvested(audit):
    assert audit["totals"]["mb_dicts"] >= 3


def test_at_least_2_bytes_dicts_harvested(audit):
    assert audit["totals"]["bytes_dicts"] >= 2


def test_every_canonical_token_was_seen(audit):
    # Tokens we shipped in the canonical registry but that don't
    # appear anywhere in the codebase are dead weight; flag if any.
    canonical = set(audit["canonical"].keys())
    seen = set(audit["tokens_seen"])
    missing = canonical - seen
    assert not missing, (
        f"canonical tokens never used in any harvested constant: "
        f"{sorted(missing)}")


# --- ast helper invariants -------------------------------------------

def test_fold_int_plain_constant(gen):
    n = ast.parse("4096", mode="eval").body
    assert gen._fold_int(n) == 4096


def test_fold_int_mult_expression(gen):
    n = ast.parse("4 * 1024", mode="eval").body
    assert gen._fold_int(n) == 4096


def test_fold_int_nested_mult(gen):
    n = ast.parse("8 * 1024 * 1024", mode="eval").body
    assert gen._fold_int(n) == 8 * 1024 * 1024


def test_fold_int_rejects_non_int(gen):
    n = ast.parse("'string'", mode="eval").body
    assert gen._fold_int(n) is None


def test_fold_float_handles_int_constant(gen):
    n = ast.parse("4", mode="eval").body
    assert gen._fold_float(n) == 4.0


def test_fold_float_handles_div(gen):
    n = ast.parse("4 / 1024", mode="eval").body
    assert gen._fold_float(n) == 4 / 1024


# --- artifact-on-disk parity -----------------------------------------

def test_audit_serialisable(audit):
    json.dumps(audit)


def test_on_disk_json_matches_live_audit(audit):
    if not JSON_OUT.exists():
        pytest.skip(
            f"{JSON_OUT} not yet generated; run `make lit-l3-registry`.")
    on_disk = json.loads(JSON_OUT.read_text())
    assert on_disk["status"] == audit["status"]
    assert on_disk["totals"] == audit["totals"]
    assert on_disk["anchor_triplet"] == audit["anchor_triplet"]
    assert on_disk["violations"] == audit["violations"]


def test_on_disk_md_exists():
    md = ROOT / "wiki" / "data" / "lit_faith_l3_registry.md"
    if md.exists():
        txt = md.read_text()
        assert "L3 cache-size registry" in txt
        assert "gate 251" in txt


def test_on_disk_csv_exists():
    csvp = ROOT / "wiki" / "data" / "lit_faith_l3_registry.csv"
    if csvp.exists():
        txt = csvp.read_text()
        assert "canonical_size" in txt
        assert "anchor_triplet" in txt
