"""Pytest gate for ``lit_faith_graph_registry`` (gate 258)."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
GEN_PATH = REPO_ROOT / "scripts" / "experiments" / "ecg" / "lit_faith_graph_registry.py"
JSON_PATH = REPO_ROOT / "wiki" / "data" / "lit_faith_graph_registry.json"
MD_PATH = REPO_ROOT / "wiki" / "data" / "lit_faith_graph_registry.md"
CSV_PATH = REPO_ROOT / "wiki" / "data" / "lit_faith_graph_registry.csv"


def _load():
    spec = importlib.util.spec_from_file_location(
        "lit_faith_graph_registry", GEN_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lit_faith_graph_registry"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load()


@pytest.fixture(scope="module")
def audit(gen):
    return gen.audit()


# --- module / shape ---------------------------------------------------------


def test_generator_exists():
    assert GEN_PATH.exists()


def test_audit_returns_dict(audit):
    assert isinstance(audit, dict)


def test_status_active(audit):
    assert audit["status"] == "active"


def test_required_keys(audit):
    for k in (
        "n_canonical", "n_families", "n_literal_sites",
        "n_distinct_literals", "n_family_dicts", "n_eval_graphs",
        "canonical", "harvested_tokens", "family_dicts",
        "eval_graphs", "rules", "violations",
    ):
        assert k in audit, k


def test_n_canonical_positive(audit):
    assert audit["n_canonical"] >= 10


def test_n_families_at_least_six(audit):
    # social, web, road, mesh, citation, kronecker, p2p, content
    assert audit["n_families"] >= 6


# --- R1-R8 live checks ------------------------------------------------------


def test_no_violations(audit):
    assert audit["violations"] == [], audit["violations"]


def test_rules_all_present(audit):
    assert set(audit["rules"]) == {"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"}


def test_r6_canonical_name_regex(gen):
    for g in gen.CANONICAL_GRAPHS:
        assert gen.GRAPH_NAME_RE.match(g.name), g.name


def test_r8_family_regex(gen):
    for g in gen.CANONICAL_GRAPHS:
        assert gen.FAMILY_RE.match(g.family), (g.name, g.family)


def test_r3_source_in_canonical_set(gen):
    allowed = {"SNAP", "GAP", "DIMACS", "KONECT", "WebGraph", "synthetic", "test"}
    for g in gen.CANONICAL_GRAPHS:
        assert g.source in allowed, (g.name, g.source)


def test_r2_non_empty_family_and_label(gen):
    for g in gen.CANONICAL_GRAPHS:
        assert g.family
        assert g.paper_label


# --- core canonical entries -------------------------------------------------


CORE_GRAPHS = {
    "email-Eu-core",
    "soc-pokec",
    "soc-LiveJournal1",
    "com-orkut",
    "cit-Patents",
    "web-Google",
    "roadNet-CA",
    "delaunay_n19",
}


def test_core_corpus_present(gen):
    names = {g.name for g in gen.CANONICAL_GRAPHS}
    missing = CORE_GRAPHS - names
    assert not missing, missing


def test_email_eu_core_is_social(gen):
    [g] = [x for x in gen.CANONICAL_GRAPHS if x.name == "email-Eu-core"]
    assert g.family == "social"
    assert g.source == "SNAP"


def test_road_family_has_kebab_alias(gen):
    [g] = [x for x in gen.CANONICAL_GRAPHS if x.name == "roadNet-CA"]
    assert "road-CA" in g.aliases


# --- harvest behaviour ------------------------------------------------------


def test_n_literal_sites_positive(audit):
    assert audit["n_literal_sites"] >= 100


def test_n_family_dicts_positive(audit):
    assert audit["n_family_dicts"] >= 5


def test_eval_graphs_all_canonical(audit, gen):
    names = {g.name for g in gen.CANONICAL_GRAPHS}
    for entry in audit["eval_graphs"]:
        assert entry["name"] in names, entry


# --- emission round-trip ----------------------------------------------------


def test_json_emit_round_trip(gen, tmp_path):
    out = tmp_path / "out.json"
    gen._emit_json(gen.audit(), out)
    raw = out.read_bytes()
    assert raw.endswith(b"\n"), "json must end with newline"
    parsed = json.loads(raw)
    assert parsed["status"] == "active"


def test_md_emit_round_trip(gen, tmp_path):
    out = tmp_path / "out.md"
    gen._emit_md(gen.audit(), out)
    raw = out.read_bytes()
    assert raw.endswith(b"\n"), "md must end with newline"
    assert not raw.endswith(b"\n\n"), "md must end with exactly one newline"


def test_csv_emit_round_trip(gen, tmp_path):
    out = tmp_path / "out.csv"
    gen._emit_csv(gen.audit(), out)
    text = out.read_text()
    assert text.startswith("kind,name,extra1,extra2")


# --- artifact parity with committed copies ---------------------------------


def test_committed_json_matches_audit(audit, gen):
    if not JSON_PATH.exists():
        pytest.skip("artifact not yet emitted")
    committed = json.loads(JSON_PATH.read_text())
    assert committed["n_canonical"] == audit["n_canonical"]
    assert committed["n_families"] == audit["n_families"]
    assert sorted(committed["harvested_tokens"]) == sorted(audit["harvested_tokens"])


def test_committed_md_exists():
    assert MD_PATH.exists()


def test_committed_csv_exists():
    assert CSV_PATH.exists()


# --- per-source dict cross-checks (R4 spot-checks) -------------------------


def test_family_dicts_use_canonical_keys(audit, gen):
    names = {g.name for g in gen.CANONICAL_GRAPHS}
    for fd in audit["family_dicts"]:
        for k in fd["mapping"]:
            assert k in names, f"{fd['path']}::{fd['dict']} key {k!r} not in canonical"


def test_family_dicts_agree_on_family(audit, gen):
    canon = {g.name: g.family for g in gen.CANONICAL_GRAPHS}
    for fd in audit["family_dicts"]:
        for k, v in fd["mapping"].items():
            assert v == canon[k], (fd["path"], fd["dict"], k, v, canon[k])


# --- regex sanity ------------------------------------------------------------


def test_graph_name_re_matches_known():
    pat = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
    for s in ("email-Eu-core", "soc-LiveJournal1", "delaunay_n19", "kron21", "uk-2005"):
        assert pat.match(s)
    assert not pat.match("3kron")
    assert not pat.match("graph with space")


def test_family_re_matches_known():
    pat = re.compile(r"^[a-z][a-z0-9_]*$")
    for s in ("social", "web", "road", "mesh", "citation", "kronecker", "p2p", "content"):
        assert pat.match(s)
    assert not pat.match("Social")
    assert not pat.match("3road")
