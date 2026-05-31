"""Tests for gate 279 — Job dataclass schema registry."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from scripts.experiments.ecg import lit_faith_job_dataclass_schema as M


# --------------------------------------------------------------------
# Structural
# --------------------------------------------------------------------


def test_registry_target_module():
    assert M.JOB_MODULE == "scripts/experiments/ecg/final_paper_run.py"


def test_registry_target_class():
    assert M.JOB_CLASS == "Job"


def test_field_count():
    assert len(M.JOB_FIELDS) == 8


def test_field_names():
    names = [f["name"] for f in M.JOB_FIELDS]
    assert names == [
        "job_id", "stage", "kind", "command",
        "out_dir", "log_path", "metadata", "env",
    ]


def test_decorator_kwargs_present():
    assert M.DATACLASS_DECORATOR_KWARGS == {"frozen": "True"}


# --------------------------------------------------------------------
# Live audit
# --------------------------------------------------------------------


def test_live_audit_zero_violations():
    doc = M.audit()
    assert doc["violations"] == [], doc["violations"]


def test_status_active():
    assert M.audit()["status"] == "active"


def test_schema_string():
    assert M.audit()["schema"] == "lit-faith-job-dataclass-schema/1"


def test_counts_match():
    doc = M.audit()
    assert doc["counts"]["fields"] == 8
    assert doc["counts"]["decorator_kwargs"] == 1
    assert doc["counts"]["modules"] == 1
    assert doc["counts"]["classes"] == 1


def test_rules_present():
    rules = M.audit()["rules"]
    for r in ("F1", "F2", "F3", "F4", "F5", "F6", "F7"):
        assert r in rules


# --------------------------------------------------------------------
# Injection helpers
# --------------------------------------------------------------------


def _clone_fields():
    return [dict(e) for e in M.JOB_FIELDS]


# --------------------------------------------------------------------
# F1 — module ast parses
# --------------------------------------------------------------------


def test_f1_missing_module(monkeypatch):
    monkeypatch.setattr(M, "JOB_MODULE",
                        "scripts/experiments/ecg/__never_exists__.py")
    viols = [v for v in M.audit()["violations"] if v["rule"] == "F1"]
    assert any(v["issue"] == "missing" for v in viols)


def test_f1_syntax_error(monkeypatch, tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("def x(:\n")
    monkeypatch.setattr(M, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(M, "JOB_MODULE", "bad.py")
    viols = [v for v in M.audit()["violations"] if v["rule"] == "F1"]
    assert any("syntax error" in v["issue"] for v in viols)


# --------------------------------------------------------------------
# F2 — class present
# --------------------------------------------------------------------


def test_f2_class_missing(monkeypatch, tmp_path):
    p = tmp_path / "no_job.py"
    p.write_text("def x(): pass\n")
    monkeypatch.setattr(M, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(M, "JOB_MODULE", "no_job.py")
    viols = [v for v in M.audit()["violations"] if v["rule"] == "F2"]
    assert any("not present" in v["issue"] for v in viols)


# --------------------------------------------------------------------
# F3 — registered field missing in live
# --------------------------------------------------------------------


def test_f3_missing_field(monkeypatch):
    fields = _clone_fields() + [
        {"name": "never_defined_field_279",
         "annot": "int", "default": "none"},
    ]
    monkeypatch.setattr(M, "JOB_FIELDS", fields)
    viols = [v for v in M.audit()["violations"] if v["rule"] == "F3"]
    assert any(v["field"] == "never_defined_field_279" for v in viols)


# --------------------------------------------------------------------
# F4 — annotation match
# --------------------------------------------------------------------


def test_f4_wrong_annot(monkeypatch):
    fields = _clone_fields()
    # change log_path to str
    for i, e in enumerate(fields):
        if e["name"] == "log_path":
            fields[i] = {**e, "annot": "str"}
    monkeypatch.setattr(M, "JOB_FIELDS", fields)
    viols = [v for v in M.audit()["violations"] if v["rule"] == "F4"]
    assert any(v["field"] == "log_path" for v in viols)


# --------------------------------------------------------------------
# F5 — default state match
# --------------------------------------------------------------------


def test_f5_wrong_default(monkeypatch):
    fields = _clone_fields()
    for i, e in enumerate(fields):
        if e["name"] == "metadata":
            fields[i] = {**e, "default": "literal:None"}
    monkeypatch.setattr(M, "JOB_FIELDS", fields)
    viols = [v for v in M.audit()["violations"] if v["rule"] == "F5"]
    assert any(v["field"] == "metadata" for v in viols)


def test_f5_missing_factory(monkeypatch):
    fields = _clone_fields()
    for i, e in enumerate(fields):
        if e["name"] == "env":
            fields[i] = {**e, "default": "none"}
    monkeypatch.setattr(M, "JOB_FIELDS", fields)
    viols = [v for v in M.audit()["violations"] if v["rule"] == "F5"]
    assert any(v["field"] == "env" for v in viols)


# --------------------------------------------------------------------
# F6 — exhaustive
# --------------------------------------------------------------------


def test_f6_drops_existing_field(monkeypatch):
    fields = [e for e in _clone_fields() if e["name"] != "out_dir"]
    monkeypatch.setattr(M, "JOB_FIELDS", fields)
    viols = [v for v in M.audit()["violations"] if v["rule"] == "F6"]
    assert any(v["field"] == "out_dir" for v in viols)


# --------------------------------------------------------------------
# F7 — decorator kwargs
# --------------------------------------------------------------------


def test_f7_missing_frozen(monkeypatch):
    monkeypatch.setattr(M, "DATACLASS_DECORATOR_KWARGS",
                        {"frozen": "True", "slots": "True"})
    viols = [v for v in M.audit()["violations"] if v["rule"] == "F7"]
    assert any(v.get("kwarg") == "slots" for v in viols)


def test_f7_no_decorator(monkeypatch, tmp_path):
    p = tmp_path / "plain_class.py"
    p.write_text(
        "class Job:\n"
        "    job_id: str\n"
        "    stage: str\n"
        "    kind: str\n"
        "    command: list[str]\n"
        "    out_dir: 'Path'\n"
        "    log_path: 'Path'\n"
        "    metadata: 'dict[str, Any]'\n"
        "    env: 'dict[str, str]'\n"
    )
    monkeypatch.setattr(M, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(M, "JOB_MODULE", "plain_class.py")
    viols = [v for v in M.audit()["violations"] if v["rule"] == "F7"]
    assert any("not @dataclass" in v["issue"] for v in viols)


# --------------------------------------------------------------------
# Helper unit tests
# --------------------------------------------------------------------


def test_default_descriptor_none():
    assert M._default_descriptor(None) == "none"


def test_default_descriptor_literal():
    val = ast.parse("42").body[0].value
    assert M._default_descriptor(val) == "literal:42"


def test_default_descriptor_factory():
    val = ast.parse("field(default_factory=dict)").body[0].value
    assert M._default_descriptor(val) == "factory:field(default_factory=dict)"


def test_top_level_class_returns_none_for_missing():
    mod = ast.parse("x = 1\n")
    assert M._top_level_class(mod, "Job") is None


def test_dataclass_kwargs_with_args():
    mod = ast.parse(
        "from dataclasses import dataclass\n"
        "@dataclass(frozen=True)\n"
        "class Job:\n"
        "    x: int\n"
    )
    cls = M._top_level_class(mod, "Job")
    kwargs = M._dataclass_kwargs(cls)
    assert kwargs == {"frozen": "True"}


def test_dataclass_kwargs_no_args():
    mod = ast.parse(
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Job:\n"
        "    x: int\n"
    )
    cls = M._top_level_class(mod, "Job")
    assert M._dataclass_kwargs(cls) == {}


def test_dataclass_kwargs_not_dataclass():
    mod = ast.parse(
        "class Job:\n"
        "    x: int\n"
    )
    cls = M._top_level_class(mod, "Job")
    assert M._dataclass_kwargs(cls) is None


def test_dataclass_kwargs_attribute_form():
    mod = ast.parse(
        "import dataclasses\n"
        "@dataclasses.dataclass(frozen=True)\n"
        "class Job:\n"
        "    x: int\n"
    )
    cls = M._top_level_class(mod, "Job")
    assert M._dataclass_kwargs(cls) == {"frozen": "True"}


def test_live_fields_parses_annotations_and_defaults():
    mod = ast.parse(
        "from dataclasses import dataclass, field\n"
        "@dataclass\n"
        "class C:\n"
        "    x: int\n"
        "    y: str = 'z'\n"
        "    z: dict = field(default_factory=dict)\n"
    )
    cls = M._top_level_class(mod, "C")
    fields = M._live_fields(cls)
    by_name = {f["name"]: f for f in fields}
    assert by_name["x"]["default"] == "none"
    assert by_name["y"]["default"] == "literal:'z'"
    assert "factory:" in by_name["z"]["default"]


# --------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------


def test_write_json(tmp_path):
    p = tmp_path / "out.json"
    M.write_json(M.audit(), p)
    d = json.loads(p.read_text())
    assert d["schema"] == "lit-faith-job-dataclass-schema/1"


def test_write_md(tmp_path):
    p = tmp_path / "out.md"
    M.write_md(M.audit(), p)
    txt = p.read_text()
    assert "Job dataclass schema (gate 279)" in txt
    assert "## Fields" in txt
    assert "## Decorator kwargs" in txt
    assert "✅ No violations" in txt
    assert txt.endswith("\n")


def test_write_csv(tmp_path):
    p = tmp_path / "out.csv"
    M.write_csv(M.audit(), p)
    txt = p.read_text()
    assert txt.startswith("kind,name,annot,default")
    assert "frozen" in txt


# --------------------------------------------------------------------
# Main CLI
# --------------------------------------------------------------------


def test_main_clean(tmp_path, capsys):
    rc = M.main([
        "--json-out", str(tmp_path / "x.json"),
        "--md-out",   str(tmp_path / "x.md"),
        "--csv-out",  str(tmp_path / "x.csv"),
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "[lit-faith-job-dataclass-schema]" in out
    assert "violations=0" in out


def test_main_no_args():
    assert M.main([]) == 0


# --------------------------------------------------------------------
# Sanity
# --------------------------------------------------------------------


def test_module_file_exists():
    assert (M.REPO_ROOT / M.JOB_MODULE).is_file()


def test_each_registered_field_has_required_keys():
    for f in M.JOB_FIELDS:
        assert set(f) >= {"name", "annot", "default"}
