#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for gate 280 — receiver dataclass schema audit."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.ecg import lit_faith_receiver_dataclass_schema as mod  # noqa: E402


# ---------------------------------------------------------------------------
# Registry integrity
# ---------------------------------------------------------------------------

def test_registry_nonempty():
    assert len(mod.REGISTRY) >= 1


def test_registry_entries_have_required_attrs():
    for e in mod.REGISTRY:
        assert isinstance(e.module, str) and e.module.startswith("scripts/")
        assert isinstance(e.class_name, str) and e.class_name
        assert isinstance(e.decorator_kwargs, dict)
        assert isinstance(e.fields, tuple)
        assert len(e.fields) >= 1


def test_registry_fields_unique_within_class():
    for e in mod.REGISTRY:
        names = [f.name for f in e.fields]
        assert len(names) == len(set(names)), \
            f"duplicate field name in {e.module}::{e.class_name}"


def test_registry_default_state_grammar():
    allowed_prefixes = ("none", "literal:", "factory:")
    for e in mod.REGISTRY:
        for f in e.fields:
            assert any(f.default == "none" or f.default.startswith(p)
                       for p in allowed_prefixes), \
                f"bad default grammar {e.class_name}.{f.name}={f.default!r}"


def test_registry_modules_exist():
    for e in mod.REGISTRY:
        assert (REPO_ROOT / e.module).exists(), f"missing: {e.module}"


# ---------------------------------------------------------------------------
# Live audit — must be 0 violations on the checked-in tree
# ---------------------------------------------------------------------------

def test_audit_returns_zero_violations():
    classes, fields_total, n_viol, violations = mod._run_audit()
    assert classes == len(mod.REGISTRY)
    assert fields_total == sum(len(e.fields) for e in mod.REGISTRY)
    assert n_viol == 0, f"violations: {violations}"


def test_audit_field_count_matches_sum():
    _, fields_total, _, _ = mod._run_audit()
    assert fields_total == sum(len(e.fields) for e in mod.REGISTRY)


# ---------------------------------------------------------------------------
# Helpers — _default_descriptor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("expr,expected", [
    (None, "none"),
    (ast.parse("0", mode="eval").body, "literal:0"),
    (ast.parse("'hello'", mode="eval").body, "literal:'hello'"),
    (ast.parse("None", mode="eval").body, "literal:None"),
    (ast.parse("False", mode="eval").body, "literal:False"),
])
def test_default_descriptor_literals(expr, expected):
    assert mod._default_descriptor(expr) == expected


def test_default_descriptor_factory():
    expr = ast.parse("field(default_factory=dict)", mode="eval").body
    out = mod._default_descriptor(expr)
    assert out.startswith("factory:")
    assert "default_factory" in out


# ---------------------------------------------------------------------------
# Helpers — _dataclass_kwargs (three decorator forms)
# ---------------------------------------------------------------------------

def _parse_one_class(src: str) -> ast.ClassDef:
    tree = ast.parse(src)
    for n in tree.body:
        if isinstance(n, ast.ClassDef):
            return n
    raise RuntimeError("no class")


def test_dataclass_kwargs_with_call_kwargs():
    cls = _parse_one_class("@dataclass(frozen=True, eq=False)\nclass C:\n    pass\n")
    assert mod._dataclass_kwargs(cls) == {"frozen": "True", "eq": "False"}


def test_dataclass_kwargs_bare():
    cls = _parse_one_class("@dataclass\nclass C:\n    pass\n")
    assert mod._dataclass_kwargs(cls) == {}


def test_dataclass_kwargs_module_qualified():
    cls = _parse_one_class("@dataclasses.dataclass(frozen=True)\nclass C:\n    pass\n")
    assert mod._dataclass_kwargs(cls) == {"frozen": "True"}


def test_dataclass_kwargs_not_a_dataclass():
    cls = _parse_one_class("class C:\n    pass\n")
    assert mod._dataclass_kwargs(cls) is None


# ---------------------------------------------------------------------------
# Helpers — _live_fields
# ---------------------------------------------------------------------------

def test_live_fields_basic():
    cls = _parse_one_class(
        "@dataclass(frozen=True)\n"
        "class C:\n"
        "    a: int\n"
        "    b: str = 'x'\n"
        "    c: dict = field(default_factory=dict)\n"
    )
    fields = mod._live_fields(cls)
    assert len(fields) == 3
    assert fields[0].name == "a" and fields[0].default == "none"
    assert fields[1].name == "b" and fields[1].default == "literal:'x'"
    assert fields[2].name == "c" and fields[2].default.startswith("factory:")


# ---------------------------------------------------------------------------
# Injection — temporarily mutate REGISTRY to verify each rule fires
# ---------------------------------------------------------------------------

@pytest.fixture
def saved_registry():
    saved = mod.REGISTRY
    yield
    mod.REGISTRY = saved


def _override_registry(new_entries):
    mod.REGISTRY = tuple(new_entries)


def test_g1_missing_module_fires(saved_registry):
    bad = mod.DataclassEntry(module="scripts/experiments/ecg/__does_not_exist.py",
                             class_name="Foo", decorator_kwargs={},
                             fields=(mod.FieldSpec("x", "int", "none"),))
    _override_registry([bad])
    _, _, n, violations = mod._run_audit()
    assert n >= 1
    assert any(v["rule"] == "G1" for v in violations)


def test_g2_missing_class_fires(saved_registry):
    bad = mod.DataclassEntry(module="scripts/experiments/ecg/roi_matrix.py",
                             class_name="ThisClassDoesNotExist",
                             decorator_kwargs={"frozen": "True"},
                             fields=(mod.FieldSpec("x", "int", "none"),))
    _override_registry([bad])
    _, _, n, violations = mod._run_audit()
    assert n >= 1
    assert any(v["rule"] == "G2" for v in violations)


def test_g3_missing_field_fires(saved_registry):
    orig = next(e for e in mod.REGISTRY if e.class_name == "PolicySpec")
    bad = mod.DataclassEntry(
        module=orig.module, class_name=orig.class_name,
        decorator_kwargs=dict(orig.decorator_kwargs),
        fields=orig.fields + (mod.FieldSpec("nonexistent_field", "int", "none"),),
    )
    _override_registry([bad])
    _, _, n, violations = mod._run_audit()
    assert any(v["rule"] == "G3" for v in violations)


def test_g4_annotation_drift_fires(saved_registry):
    orig = next(e for e in mod.REGISTRY if e.class_name == "PolicySpec")
    new_fields = tuple(
        mod.FieldSpec(f.name, "WrongAnnot" if f.name == "label" else f.annotation, f.default)
        for f in orig.fields
    )
    _override_registry([mod.DataclassEntry(orig.module, orig.class_name,
                                           dict(orig.decorator_kwargs), new_fields)])
    _, _, n, violations = mod._run_audit()
    assert any(v["rule"] == "G4" for v in violations)


def test_g5_default_drift_fires(saved_registry):
    orig = next(e for e in mod.REGISTRY if e.class_name == "PolicySpec")
    new_fields = tuple(
        mod.FieldSpec(f.name, f.annotation, "literal:42" if f.name == "label" else f.default)
        for f in orig.fields
    )
    _override_registry([mod.DataclassEntry(orig.module, orig.class_name,
                                           dict(orig.decorator_kwargs), new_fields)])
    _, _, n, violations = mod._run_audit()
    assert any(v["rule"] == "G5" for v in violations)


def test_g6_surprise_field_fires(saved_registry):
    orig = next(e for e in mod.REGISTRY if e.class_name == "PolicySpec")
    truncated = mod.DataclassEntry(orig.module, orig.class_name,
                                   dict(orig.decorator_kwargs),
                                   orig.fields[:-1])
    _override_registry([truncated])
    _, _, n, violations = mod._run_audit()
    assert any(v["rule"] == "G6" for v in violations)


def test_g7_decorator_kwarg_drift_fires(saved_registry):
    orig = next(e for e in mod.REGISTRY if e.class_name == "PolicySpec")
    _override_registry([mod.DataclassEntry(orig.module, orig.class_name,
                                           {"frozen": "False"}, orig.fields)])
    _, _, n, violations = mod._run_audit()
    assert any(v["rule"] == "G7" for v in violations)


def test_g7_surprise_kwarg_fires(saved_registry):
    orig = next(e for e in mod.REGISTRY if e.class_name == "PolicySpec")
    _override_registry([mod.DataclassEntry(orig.module, orig.class_name,
                                           {}, orig.fields)])
    _, _, n, violations = mod._run_audit()
    # frozen=True is on live class, not in registry → G7 surprise
    assert any(v["rule"] == "G7" for v in violations)


# ---------------------------------------------------------------------------
# Cross-gate parity — gate 280's Job-style structure mirrors gate 279
# ---------------------------------------------------------------------------

def test_all_entries_frozen():
    # All three receiver dataclasses must remain frozen — mirrors gate 279.
    for e in mod.REGISTRY:
        assert e.decorator_kwargs.get("frozen") == "True", \
            f"{e.class_name} must be frozen=True for replay parity"


def test_registry_covers_both_receivers():
    """Receivers from gate 278 (roi_matrix, proof_matrix) both appear."""
    modules = {e.module for e in mod.REGISTRY}
    assert "scripts/experiments/ecg/roi_matrix.py" in modules
    assert "scripts/experiments/ecg/proof_matrix.py" in modules


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------

def test_main_writes_artifacts(tmp_path):
    rc = mod.main([
        "--json-out", str(tmp_path / "g.json"),
        "--md-out",   str(tmp_path / "g.md"),
        "--csv-out",  str(tmp_path / "g.csv"),
    ])
    assert rc == 0
    payload = json.loads((tmp_path / "g.json").read_text())
    assert payload["gate"] == 280
    assert payload["status"] == "active"
    assert payload["violations"] == []
    assert payload["classes"] == len(mod.REGISTRY)


def test_json_has_per_class_registry(tmp_path):
    mod.write_json(tmp_path / "g.json", 3, 13, [])
    payload = json.loads((tmp_path / "g.json").read_text())
    assert isinstance(payload["registry"], list)
    assert len(payload["registry"]) == len(mod.REGISTRY)
    for entry in payload["registry"]:
        assert "module" in entry and "class" in entry
        assert "fields" in entry and isinstance(entry["fields"], list)


def test_md_renders_no_violations_marker(tmp_path):
    mod.write_md(tmp_path / "g.md", 3, 13, [])
    text = (tmp_path / "g.md").read_text()
    assert "✅ No violations" in text
    assert "PolicySpec" in text
    assert "Ablation" in text
    assert "AdaptiveSelector" in text


def test_csv_row_per_field(tmp_path):
    mod.write_csv(tmp_path / "g.csv")
    rows = (tmp_path / "g.csv").read_text().splitlines()
    # header + 13 rows
    assert len(rows) == 1 + 13
