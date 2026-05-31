"""Pytest for gate 277 — subprocess argv registry."""

from __future__ import annotations

import ast
import copy
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts/experiments/ecg/lit_faith_subprocess_argv_registry.py"

sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/ecg"))
import lit_faith_subprocess_argv_registry as mod  # noqa: E402


# --------------------------------------------------------------------
# Module presence and surface
# --------------------------------------------------------------------


def test_module_file_exists():
    assert MODULE_PATH.is_file()


def test_audit_callable():
    assert callable(mod.audit)


def test_main_callable():
    assert callable(mod.main)


def test_stages_constant():
    assert hasattr(mod, "STAGES")
    assert isinstance(mod.STAGES, list)
    assert len(mod.STAGES) == 3


def test_helper_constructor_present():
    assert callable(mod._stage)


def test_extract_helpers_present():
    for name in ["_extract_flags", "_extract_uppercase_refs",
                 "_load_ast", "_find_fn"]:
        assert callable(getattr(mod, name))


def test_flag_pattern_regex_present():
    assert mod.FLAG_PATTERN.pattern.startswith("^--")


# --------------------------------------------------------------------
# Live audit — no violations
# --------------------------------------------------------------------


def test_live_audit_no_violations():
    doc = mod.audit()
    assert doc["violations"] == [], doc["violations"]
    assert doc["status"] == "active"
    assert doc["counts"]["n_stages"] == 3
    assert doc["counts"]["n_locked_flags_total"] >= 50


def test_audit_doc_shape():
    doc = mod.audit()
    for k in ["schema", "status", "stages", "counts", "rules", "violations"]:
        assert k in doc
    assert doc["schema"].startswith("lit-faith-subprocess-argv-registry/")
    assert set(doc["rules"].keys()) == {"A1", "A2", "A3", "A4", "A5", "A6"}


def test_locked_targets_match_three_known_constants():
    targets = {s["target_const"] for s in mod.STAGES}
    assert targets == {"FINAL_RUN", "PROOF_MATRIX", "ROI_MATRIX"}


def test_locked_fn_names_match_three_known_builders():
    names = {s["fn_name"] for s in mod.STAGES}
    assert names == {"run_profile", "make_proof_job", "make_roi_job"}


def test_every_stage_flags_match_pattern():
    for s in mod.STAGES:
        for f in s["flags"]:
            assert mod.FLAG_PATTERN.match(f), f"bad flag in locked: {f}"


def test_every_stage_has_required_keys():
    for s in mod.STAGES:
        for k in ["fn_name", "module_path", "target_const", "flags"]:
            assert k in s


def test_modules_exist_on_disk():
    for s in mod.STAGES:
        assert (REPO_ROOT / s["module_path"]).is_file()


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------


def _parse(snippet: str) -> ast.Module:
    return ast.parse(textwrap.dedent(snippet))


def test_extract_flags_basic():
    src = '''
        def f():
            cmd = ["--alpha", "--beta", "x"]
            cmd.append("--gamma")
    '''
    fn = _parse(src).body[0]
    flags = mod._extract_flags(fn)
    assert flags == {"--alpha", "--beta", "--gamma"}


def test_extract_uppercase_refs():
    src = '''
        def f():
            x = SOMETHING
            y = lower_case
            z = ANOTHER_ONE
    '''
    fn = _parse(src).body[0]
    refs = mod._extract_uppercase_refs(fn)
    assert "SOMETHING" in refs
    assert "ANOTHER_ONE" in refs
    assert "lower_case" not in refs


def test_extract_uppercase_skips_underscore_prefixed():
    src = "def f(): return _PRIVATE"
    fn = _parse(src).body[0]
    refs = mod._extract_uppercase_refs(fn)
    assert "_PRIVATE" not in refs


def test_find_fn_returns_target():
    src = "def a(): pass\ndef b(): pass\n"
    m = _parse(src)
    assert mod._find_fn(m, "a").name == "a"
    assert mod._find_fn(m, "b").name == "b"
    assert mod._find_fn(m, "c") is None


def test_load_ast_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    assert mod._load_ast("nope.py") is None


def test_load_ast_syntax_error(monkeypatch, tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("def f(:\n")
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    assert mod._load_ast("bad.py") is None


# --------------------------------------------------------------------
# Injection tests — swap STAGES with a stage that violates one rule
# --------------------------------------------------------------------


@pytest.fixture
def swap_stages(monkeypatch):
    """Replace mod.STAGES with a mutated single-stage list."""
    def _swap(mutator):
        base = copy.deepcopy(mod.STAGES[0])
        mutator(base)
        monkeypatch.setattr(mod, "STAGES", [base])
    return _swap


def _violations_for_rule(doc: dict, rule: str) -> list[dict]:
    return [v for v in doc["violations"] if v.get("rule") == rule]


def test_a1_missing_fn(swap_stages):
    swap_stages(lambda s: s.update(fn_name="this_fn_does_not_exist_anywhere"))
    viols = _violations_for_rule(mod.audit(), "A1")
    assert any("fn not found" in v.get("issue", "") for v in viols)


def test_a1_missing_module(swap_stages, monkeypatch):
    swap_stages(lambda s: s.update(module_path="scripts/experiments/ecg/nope.py"))
    viols = _violations_for_rule(mod.audit(), "A1")
    assert any("module missing" in v.get("issue", "") for v in viols)


def test_a2_wrong_target_const(swap_stages):
    swap_stages(lambda s: s.update(target_const="DEFINITELY_NOT_REFERENCED"))
    viols = _violations_for_rule(mod.audit(), "A2")
    assert any(v.get("want_const") == "DEFINITELY_NOT_REFERENCED"
               for v in viols)


def test_a3_locked_flag_missing(swap_stages):
    swap_stages(lambda s: s["flags"].append("--this-flag-does-not-exist-anywhere"))
    viols = _violations_for_rule(mod.audit(), "A3")
    assert any(v.get("missing_flag") == "--this-flag-does-not-exist-anywhere"
               for v in viols)


def test_a4_unknown_live_flag(swap_stages):
    swap_stages(lambda s: s["flags"].remove("--profile"))
    viols = _violations_for_rule(mod.audit(), "A4")
    assert any(v.get("extra_flag") == "--profile" for v in viols)


def test_a5_empty_flags(swap_stages):
    swap_stages(lambda s: s.update(flags=[]))
    viols = _violations_for_rule(mod.audit(), "A5")
    assert any(v.get("issue") == "registry entry has empty flags list"
               for v in viols)


def test_a5_empty_target_const(swap_stages):
    swap_stages(lambda s: s.update(target_const=""))
    viols = _violations_for_rule(mod.audit(), "A5")
    assert any(v.get("issue") == "registry entry has empty target_const"
               for v in viols)


def test_a6_bad_flag_pattern(swap_stages):
    swap_stages(lambda s: s["flags"].append("--BadCASE"))
    viols = _violations_for_rule(mod.audit(), "A6")
    assert any(v.get("bad_locked_flag") == "--BadCASE" for v in viols)


def test_a6_single_dash_flag(swap_stages):
    swap_stages(lambda s: s["flags"].append("-x"))
    # A6 only checks --flags, so -x would also fail A3 (not in source)
    # and FLAG_PATTERN doesn't match.
    viols = mod.audit()["violations"]
    assert any(v.get("bad_locked_flag") == "-x" for v in viols)


# --------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------


def test_write_json_round_trip(tmp_path):
    doc = mod.audit()
    p = tmp_path / "x.json"
    mod.write_json(doc, p)
    loaded = json.loads(p.read_text("utf-8"))
    assert loaded["schema"] == doc["schema"]


def test_write_md_contains_headline(tmp_path):
    doc = mod.audit()
    p = tmp_path / "x.md"
    mod.write_md(doc, p)
    text = p.read_text("utf-8")
    assert "gate 277" in text.lower()
    assert "subprocess argv registry" in text.lower()
    # The stages table must reference all three builders.
    for fn in ["run_profile", "make_proof_job", "make_roi_job"]:
        assert fn in text


def test_write_csv_rows(tmp_path):
    doc = mod.audit()
    p = tmp_path / "x.csv"
    mod.write_csv(doc, p)
    text = p.read_text("utf-8")
    assert text.startswith("stage,module_path,target_const,locked_flag,index")
    assert "--profile" in text


# --------------------------------------------------------------------
# CLI / main()
# --------------------------------------------------------------------


def test_main_exit_zero_in_subprocess():
    r = subprocess.run([sys.executable, str(MODULE_PATH)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "[lit-faith-subprocess-argv-registry]" in r.stdout


def test_main_writes_files(tmp_path):
    j = tmp_path / "out.json"
    m = tmp_path / "out.md"
    c = tmp_path / "out.csv"
    rc = mod.main(["--json-out", str(j),
                   "--md-out", str(m),
                   "--csv-out", str(c)])
    assert rc == 0
    assert j.exists() and m.exists() and c.exists()


def test_main_returns_nonzero_on_violation(monkeypatch):
    orig_audit = mod.audit

    def _fake_audit():
        doc = orig_audit()
        doc["violations"].append({"rule": "A1", "issue": "test injection"})
        return doc

    monkeypatch.setattr(mod, "audit", _fake_audit)
    rc = mod.main([])
    assert rc == 1
