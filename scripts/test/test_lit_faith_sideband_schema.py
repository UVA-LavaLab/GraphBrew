#!/usr/bin/env python3
"""Pytest gate 248 — gem5/Sniper/cache_sim sideband-schema registry."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GEN_PATH = ROOT / "scripts" / "experiments" / "ecg" / "lit_faith_sideband_schema.py"
WIKI_DATA = ROOT / "wiki" / "data"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "lit_faith_sideband_schema", GEN_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lit_faith_sideband_schema"] = mod
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


def test_schema_field_count_today(audit):
    assert audit["schema_field_count"] == 6


def test_emit_site_count_today(audit):
    assert audit["emit_site_count"] == 3


# --------------------------------------------------- registry contract --

def test_canonical_field_order():
    names = [f["name"] for f in MOD.SCHEMA_REGISTRY]
    assert names == ["source", "name", "base", "upper", "hot_pct", "grasp_region"]


def test_canonical_param_types():
    types = [f["cpp_type"] for f in MOD.SCHEMA_REGISTRY]
    assert types == ["const char*", "const char*", "uint64_t",
                     "uint64_t", "uint32_t", "bool"]


def test_canonical_specifiers():
    fmts = [f["fmt"] for f in MOD.SCHEMA_REGISTRY]
    assert fmts == ["%s", "%s", "0x%lx", "0x%lx", "%u", "%d"]


def test_emit_prefix_literal():
    assert MOD.EMIT_PREFIX == "[graphctx] register region"


def test_canonical_format_string():
    expected = (
        "[graphctx] register region "
        "source=%s name=%s base=0x%lx upper=0x%lx "
        "hot_pct=%u grasp_region=%d\\n"
    )
    assert MOD._canonical_format_string() == expected


# ---------------------------------------------------------- emit sites --

def test_every_emit_site_exists():
    for rel in MOD.EMIT_SITES:
        assert (ROOT / rel).exists(), rel


def test_every_emit_site_contains_canonical_prefix():
    for rel in MOD.EMIT_SITES:
        src = (ROOT / rel).read_text()
        assert MOD.EMIT_PREFIX in src, rel


# ------------------------------------------------------------ rules --
# S1 — file existence (covered by test_every_emit_site_exists)
# S2 — format string parity per site

@pytest.mark.parametrize("rel", MOD.EMIT_SITES)
def test_s2_format_string_matches(rel):
    src = (ROOT / rel).read_text()
    fmt = MOD._extract_format_string(src)
    assert fmt is not None, rel
    assert fmt == MOD._canonical_format_string(), rel


# S3 — function signature parity per site
@pytest.mark.parametrize("rel", MOD.EMIT_SITES)
def test_s3_function_signature_matches(rel):
    src = (ROOT / rel).read_text()
    params = MOD._extract_param_types(src)
    assert params is not None, rel
    assert params == [f["cpp_type"] for f in MOD.SCHEMA_REGISTRY], rel


# S4 — every emit site uses canonical literal prefix
@pytest.mark.parametrize("rel", MOD.EMIT_SITES)
def test_s4_prefix_in_format_string(rel):
    src = (ROOT / rel).read_text()
    fmt = MOD._extract_format_string(src)
    assert fmt is not None and MOD.EMIT_PREFIX in fmt, rel


# S5 — every schema field has an allowed specifier
def test_s5_specifiers_in_allowed_set():
    for entry in MOD.SCHEMA_REGISTRY:
        assert entry["fmt"] in MOD._ALLOWED_SPECIFIERS, entry


# S6 — Tier-A regex compiles and round-trips
def test_s6_regex_round_trip(audit):
    assert audit["tier_a_regex_round_trip_ok"] is True


def test_s6_regex_groups_match_schema():
    pat = re.compile(MOD.TIER_A_REGEX)
    groups = set(pat.groupindex.keys())
    expected = {f["name"] for f in MOD.SCHEMA_REGISTRY}
    assert groups == expected


def test_s6_regex_extracts_real_values():
    line = (
        "[graphctx] register region source=gem5 name=property "
        "base=0xdeadbeef upper=0xfeedface hot_pct=15 grasp_region=1"
    )
    m = re.search(MOD.TIER_A_REGEX, line)
    assert m is not None
    assert m.group("source") == "gem5"
    assert m.group("name") == "property"
    assert m.group("base") == "deadbeef"
    assert m.group("upper") == "feedface"
    assert m.group("hot_pct") == "15"
    assert m.group("grasp_region") == "1"


# S7 — exactly one register-region fprintf per file
@pytest.mark.parametrize("rel", MOD.EMIT_SITES)
def test_s7_single_emit_call(rel):
    src = (ROOT / rel).read_text()
    assert MOD._count_emit_calls(src) == 1, rel


# ---------------------------------------------------------- helpers --

def test_normalize_concat_literals():
    raw = '"hello " "world\\n"'
    assert MOD._normalize_format_literal(raw) == "hello world\\n"


def test_extract_format_string_handles_multiline():
    src = '''
    std::fprintf(stderr,
                 "[graphctx] register region "
                 "source=%s\\n",
                 src);
    '''
    fmt = MOD._extract_format_string(src)
    assert fmt == "[graphctx] register region source=%s\\n"


def test_extract_param_types_handles_inline_qualifier():
    src = "inline void logGraphCtxRegistration(const char* a, uint64_t b)"
    types = MOD._extract_param_types(src)
    assert types == ["const char*", "uint64_t"]


# --------------------------------------------------------- artifact parity --

def test_artifact_parity_when_present():
    art = WIKI_DATA / "lit_faith_sideband_schema.json"
    if not art.exists():
        pytest.skip("artifact not generated yet")
    on_disk = json.loads(art.read_text())
    live = MOD.audit()
    assert on_disk["schema_field_count"] == live["schema_field_count"]
    assert on_disk["emit_site_count"] == live["emit_site_count"]
    assert on_disk["canonical_format_string"] == live["canonical_format_string"]
    assert len(on_disk["violations"]) == len(live["violations"])
