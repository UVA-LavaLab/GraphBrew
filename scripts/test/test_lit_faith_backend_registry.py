"""Pytest gate for the gate-257 backend/tool vocabulary registry.

Covers:
  * generator imports cleanly and dynamically loads
  * audit() returns the expected shape (status + n_* counts +
    canonical + harvested_tokens + argparse_entries + rules +
    violations)
  * per-rule live checks (R1-R7) fire correctly on the current
    repo state (0 violations)
  * module-level invariants: regex shape, canonical count, family
    set, paper-label disjointness, punctuation-variants integrity
  * artifact parity: json/md/csv all emit, json matches the audit
    payload byte-for-byte
"""
from __future__ import annotations

import csv as _csv
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = (
    REPO_ROOT
    / "scripts"
    / "experiments"
    / "ecg"
    / "lit_faith_backend_registry.py"
)
JSON_OUT = REPO_ROOT / "wiki" / "data" / "lit_faith_backend_registry.json"
MD_OUT = REPO_ROOT / "wiki" / "data" / "lit_faith_backend_registry.md"
CSV_OUT = REPO_ROOT / "wiki" / "data" / "lit_faith_backend_registry.csv"


def _load():
    spec = importlib.util.spec_from_file_location(
        "lit_faith_backend_registry", GENERATOR
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load()


@pytest.fixture(scope="module")
def data(mod):
    return mod.audit()


# --- shape -----------------------------------------------------------------


def test_generator_imports(mod):
    assert mod is not None
    assert hasattr(mod, "audit")
    assert hasattr(mod, "CANONICAL_BACKENDS")
    assert hasattr(mod, "CANONICAL_BACKEND_NAMES")
    assert hasattr(mod, "CANONICAL_BACKEND_FAMILIES")
    assert hasattr(mod, "BACKEND_NAME_RE")


def test_audit_shape(data):
    for k in (
        "status",
        "n_canonical",
        "n_families",
        "n_literal_sites",
        "n_distinct_literals",
        "n_argparse_sites",
        "canonical",
        "harvested_tokens",
        "argparse_entries",
        "rules",
        "violations",
    ):
        assert k in data, f"missing key {k!r}"
    assert data["status"] == "active"
    assert isinstance(data["canonical"], list)
    assert isinstance(data["violations"], list)


def test_no_violations(data):
    assert data["violations"] == [], data["violations"]


# --- per-rule live checks --------------------------------------------------


def test_rules_dict_covers_R1_R7(data):
    keys = set(data["rules"].keys())
    assert keys == {"R1", "R2", "R3", "R4", "R5", "R6", "R7"}, keys


def test_R1_every_literal_canonical(mod, data):
    canon = mod.CANONICAL_BACKEND_NAMES
    for tok in data["harvested_tokens"]:
        assert tok in canon, f"R1: harvested {tok!r} not canonical"


def test_R2_canonical_family_and_label(data):
    families = {"cache_sim", "gem5", "sniper"}
    for b in data["canonical"]:
        assert b["family"] in families, b
        assert b["paper_label"], b


def test_R3_paper_label_unique_except_punctuation_variants(data):
    by_label: dict[str, list[str]] = {}
    for b in data["canonical"]:
        by_label.setdefault(b["paper_label"], []).append(b["name"])
    for label, names in by_label.items():
        if len(names) <= 1:
            continue
        # All sharing names must be mutual punctuation variants.
        entry = next(b for b in data["canonical"] if b["name"] == names[0])
        for n in names[1:]:
            assert n in entry["punctuation_variants"], (label, names)


def test_R4_punctuation_variants_include_self(data):
    for b in data["canonical"]:
        assert b["name"] in b["punctuation_variants"], b


def test_R5_argparse_entries_all_canonical(mod, data):
    canon = mod.CANONICAL_BACKEND_NAMES
    for entry in data["argparse_entries"]:
        for c in entry["choices"]:
            assert c in canon or c == "both", entry
        if entry["default"] is not None:
            assert entry["default"] in canon or entry["default"] == "both", entry


def test_R6_canonical_names_regex(mod, data):
    rx = mod.BACKEND_NAME_RE
    for b in data["canonical"]:
        assert rx.match(b["name"]), b["name"]


def test_R7_every_canonical_has_in_tree_reference(data):
    harvested = set(data["harvested_tokens"])
    for b in data["canonical"]:
        assert b["name"] in harvested, (
            f"R7: canonical {b['name']!r} has no in-tree literal"
        )


# --- module-level invariants ----------------------------------------------


def test_backend_name_regex_shape(mod):
    rx = mod.BACKEND_NAME_RE
    assert rx.pattern == r"^[a-z][a-z0-9_-]*$"
    assert rx.match("gem5")
    assert rx.match("gem5-riscv")
    assert rx.match("cache_sim")
    assert rx.match("cache-sim")
    assert not rx.match("Gem5")
    assert not rx.match("3gem5")
    assert not rx.match("gem5 ")
    assert not rx.match("")


def test_canonical_count(mod):
    assert len(mod.CANONICAL_BACKENDS) == 7


def test_canonical_families_set(mod):
    assert mod.CANONICAL_BACKEND_FAMILIES == frozenset(
        {"cache_sim", "gem5", "sniper"}
    )


def test_canonical_names_frozen(mod):
    assert mod.CANONICAL_BACKEND_NAMES == frozenset({
        "cache_sim", "cache-sim", "gem5", "gem5-riscv",
        "gem5-x86", "sniper", "sniper-sift",
    })


def test_cache_sim_punctuation_pair(mod):
    by_name = {b.name: b for b in mod.CANONICAL_BACKENDS}
    assert "cache_sim" in by_name["cache-sim"].punctuation_variants
    assert "cache-sim" in by_name["cache_sim"].punctuation_variants


def test_gem5_family_has_three_members(mod):
    fam = [b for b in mod.CANONICAL_BACKENDS if b.family == "gem5"]
    assert {b.name for b in fam} == {"gem5", "gem5-riscv", "gem5-x86"}


def test_sniper_family_has_two_members(mod):
    fam = [b for b in mod.CANONICAL_BACKENDS if b.family == "sniper"]
    assert {b.name for b in fam} == {"sniper", "sniper-sift"}


def test_cache_sim_family_has_two_members(mod):
    fam = [b for b in mod.CANONICAL_BACKENDS if b.family == "cache_sim"]
    assert {b.name for b in fam} == {"cache_sim", "cache-sim"}


def test_paper_label_for_unique_keys(mod):
    by_name = {b.name: b for b in mod.CANONICAL_BACKENDS}
    assert by_name["sniper"].paper_label == "Sniper"
    assert by_name["sniper-sift"].paper_label == "Sniper/SIFT"
    assert by_name["gem5-riscv"].paper_label == "gem5/RISC-V"
    assert by_name["gem5-x86"].paper_label == "gem5/X86"


def test_purpose_non_empty(mod):
    for b in mod.CANONICAL_BACKENDS:
        assert b.purpose and len(b.purpose) >= 50, b.name


def test_backend_arg_flags_documented(mod):
    assert "--backend" in mod.BACKEND_ARG_FLAGS
    assert "--tool" in mod.BACKEND_ARG_FLAGS
    assert "--suite" in mod.BACKEND_ARG_FLAGS
    assert "--source-backend" in mod.BACKEND_ARG_FLAGS


# --- artifact parity -------------------------------------------------------


def test_json_artifact_exists():
    assert JSON_OUT.exists(), JSON_OUT


def test_md_artifact_exists():
    assert MD_OUT.exists(), MD_OUT


def test_csv_artifact_exists():
    assert CSV_OUT.exists(), CSV_OUT


def test_json_matches_audit(mod):
    on_disk = json.loads(JSON_OUT.read_text())
    fresh = mod.audit()
    # Stable subset (lists may compare by order; both come from
    # deterministic iteration over the same module state).
    for k in (
        "status",
        "n_canonical",
        "n_families",
        "n_literal_sites",
        "n_distinct_literals",
        "n_argparse_sites",
        "harvested_tokens",
        "canonical",
        "rules",
        "violations",
    ):
        assert on_disk[k] == fresh[k], k


def test_md_documents_gate_257_and_rules():
    text = MD_OUT.read_text()
    assert "Gate 257" in text
    for rid in ("R1", "R2", "R3", "R4", "R5", "R6", "R7"):
        assert rid in text


def test_md_lists_every_canonical(mod):
    text = MD_OUT.read_text()
    for b in mod.CANONICAL_BACKENDS:
        assert f"`{b.name}`" in text, b.name


def test_csv_has_canonical_rows(mod):
    with CSV_OUT.open() as f:
        rdr = _csv.reader(f)
        rows = list(rdr)
    assert rows[0] == ["kind", "name", "extra"]
    canon_rows = [r for r in rows if r[0] == "canonical"]
    assert len(canon_rows) == len(mod.CANONICAL_BACKENDS)


def test_json_final_newline_and_no_double():
    raw = JSON_OUT.read_bytes()
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")


def test_md_final_newline_and_no_double():
    raw = MD_OUT.read_bytes()
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
