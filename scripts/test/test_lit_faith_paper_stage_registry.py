"""Tests for ``lit_faith_paper_stage_registry`` (gate 275)."""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pytest

from scripts.experiments.ecg import lit_faith_paper_stage_registry as G

REPO_ROOT = Path(G.__file__).resolve().parents[3]


# ---------------------------------------------------------------- shape


def test_module_importable():
    importlib.reload(G)


def test_registry_is_dict():
    assert isinstance(G.STAGE_REGISTRY, dict)


def test_registry_has_two_orchestrators():
    assert len(G.STAGE_REGISTRY) == 2


def test_paper_pipeline_path_present():
    assert G.PAPER_PIPELINE_PATH in G.STAGE_REGISTRY


def test_final_paper_run_path_present():
    assert G.FINAL_PAPER_RUN_PATH in G.STAGE_REGISTRY


def test_paper_pipeline_path_is_relative():
    assert not Path(G.PAPER_PIPELINE_PATH).is_absolute()


def test_final_paper_run_path_is_relative():
    assert not Path(G.FINAL_PAPER_RUN_PATH).is_absolute()


def test_paper_pipeline_fn_count():
    assert len(G.PAPER_PIPELINE_STAGES) == 50


def test_final_paper_run_fn_count():
    assert len(G.FINAL_PAPER_RUN_STAGES) == 37


def test_total_fn_count():
    total = len(G.PAPER_PIPELINE_STAGES) + len(G.FINAL_PAPER_RUN_STAGES)
    assert total == 87


def test_each_entry_has_required_keys():
    required = {"name", "args", "args_has_default", "kwonly",
                "kwonly_has_default", "vararg", "kwarg", "returns", "is_async"}
    for stages in G.STAGE_REGISTRY.values():
        for e in stages:
            assert required <= set(e.keys()), e


def test_args_and_defaults_length_match():
    for stages in G.STAGE_REGISTRY.values():
        for e in stages:
            assert len(e["args"]) == len(e["args_has_default"]), e["name"]


def test_kwonly_and_defaults_length_match():
    for stages in G.STAGE_REGISTRY.values():
        for e in stages:
            assert len(e["kwonly"]) == len(e["kwonly_has_default"]), e["name"]


def test_no_duplicate_fn_names_per_orchestrator():
    for stages in G.STAGE_REGISTRY.values():
        names = [e["name"] for e in stages]
        assert len(names) == len(set(names))


def test_no_public_fn_starts_with_underscore():
    for stages in G.STAGE_REGISTRY.values():
        for e in stages:
            assert not e["name"].startswith("_"), e["name"]


# ---------------------------------------------------------------- key fns

# paper_pipeline backbone

def test_paper_pipeline_has_run_profile():
    names = {e["name"] for e in G.PAPER_PIPELINE_STAGES}
    assert "run_profile" in names


def test_paper_pipeline_has_collect_csvs():
    names = {e["name"] for e in G.PAPER_PIPELINE_STAGES}
    assert "collect_csvs" in names


def test_paper_pipeline_has_summarize_roi():
    names = {e["name"] for e in G.PAPER_PIPELINE_STAGES}
    assert "summarize_roi" in names


def test_paper_pipeline_has_generate_outputs():
    names = {e["name"] for e in G.PAPER_PIPELINE_STAGES}
    assert "generate_outputs" in names


def test_paper_pipeline_has_main():
    names = {e["name"] for e in G.PAPER_PIPELINE_STAGES}
    assert "main" in names


def test_paper_pipeline_main_returns_int():
    entry = next(e for e in G.PAPER_PIPELINE_STAGES if e["name"] == "main")
    assert entry["returns"] == "int"


def test_paper_pipeline_parse_args_returns_namespace():
    entry = next(e for e in G.PAPER_PIPELINE_STAGES if e["name"] == "parse_args")
    assert entry["returns"] == "argparse.Namespace"


def test_paper_pipeline_generate_outputs_args():
    entry = next(e for e in G.PAPER_PIPELINE_STAGES if e["name"] == "generate_outputs")
    assert entry["args"] == ["out_dir", "roi_rows", "proof_rows", "copy_to_paper"]


# final_paper_run backbone

def test_final_paper_run_has_load_manifest():
    names = {e["name"] for e in G.FINAL_PAPER_RUN_STAGES}
    assert "load_manifest" in names


def test_final_paper_run_has_expand_jobs():
    names = {e["name"] for e in G.FINAL_PAPER_RUN_STAGES}
    assert "expand_jobs" in names


def test_final_paper_run_has_run_job():
    names = {e["name"] for e in G.FINAL_PAPER_RUN_STAGES}
    assert "run_job" in names


def test_final_paper_run_has_run_lock():
    names = {e["name"] for e in G.FINAL_PAPER_RUN_STAGES}
    assert "run_lock" in names


def test_final_paper_run_has_validate_gate():
    names = {e["name"] for e in G.FINAL_PAPER_RUN_STAGES}
    assert "validate_gate" in names


def test_final_paper_run_has_main():
    names = {e["name"] for e in G.FINAL_PAPER_RUN_STAGES}
    assert "main" in names


def test_final_paper_run_main_returns_int():
    entry = next(e for e in G.FINAL_PAPER_RUN_STAGES if e["name"] == "main")
    assert entry["returns"] == "int"


def test_final_paper_run_run_job_returns_int():
    entry = next(e for e in G.FINAL_PAPER_RUN_STAGES if e["name"] == "run_job")
    assert entry["returns"] == "int"


def test_final_paper_run_expand_jobs_returns_list_job():
    entry = next(e for e in G.FINAL_PAPER_RUN_STAGES if e["name"] == "expand_jobs")
    assert entry["returns"] == "list[Job]"


# no async anywhere (sync pipeline)

def test_no_async_anywhere():
    for stages in G.STAGE_REGISTRY.values():
        for e in stages:
            assert e["is_async"] is False, e["name"]


# ---------------------------------------------------------------- live


def test_live_paper_pipeline_clean():
    viols = [v for v in G.audit()["violations"]
             if v.get("path") == G.PAPER_PIPELINE_PATH]
    assert viols == []


def test_live_final_paper_run_clean():
    viols = [v for v in G.audit()["violations"]
             if v.get("path") == G.FINAL_PAPER_RUN_PATH]
    assert viols == []


def test_audit_no_violations_globally():
    assert G.audit()["violations"] == []


def test_audit_counts_shape():
    counts = G.audit()["counts"]
    assert counts["orchestrators"] == 2
    assert counts["paper_pipeline_fns"] == 50
    assert counts["final_paper_run_fns"] == 37
    assert counts["total_fns"] == 87


def test_orchestrator_files_exist_on_disk():
    for rel in G.STAGE_REGISTRY:
        assert (REPO_ROOT / rel).is_file()


def test_audit_schema_field():
    assert G.audit()["schema"] == "lit-faith-paper-stage-registry/1"


def test_audit_rules_present():
    rules = G.audit()["rules"]
    assert set(rules.keys()) == {"S1", "S2", "S3", "S4", "S5", "S6", "S7"}


# ---------------------------------------------------------------- helpers


def test_live_signature_extracts_basic_fn():
    mod = ast.parse("def foo(a, b=1) -> int: return 0")
    fn = mod.body[0]
    sig = G._live_signature(fn)
    assert sig["args"] == ["a", "b"]
    assert sig["args_has_default"] == [False, True]
    assert sig["returns"] == "int"
    assert sig["is_async"] is False


def test_live_signature_recognises_async():
    mod = ast.parse("async def foo() -> None: pass")
    fn = mod.body[0]
    sig = G._live_signature(fn)
    assert sig["is_async"] is True


def test_live_signature_recognises_vararg_kwarg():
    mod = ast.parse("def foo(*args, **kwargs): pass")
    fn = mod.body[0]
    sig = G._live_signature(fn)
    assert sig["vararg"] == "args"
    assert sig["kwarg"] == "kwargs"


def test_live_signature_recognises_kwonly():
    mod = ast.parse("def foo(*, x, y=2): pass")
    fn = mod.body[0]
    sig = G._live_signature(fn)
    assert sig["kwonly"] == ["x", "y"]
    assert sig["kwonly_has_default"] == [False, True]


def test_live_public_signatures_skips_private():
    src = "def foo(): pass\ndef _bar(): pass\n"
    fake = REPO_ROOT / "scripts" / "experiments" / "ecg" / "_fake_for_test.py"
    try:
        fake.write_text(src, "utf-8")
        sigs = G._live_public_signatures(
            str(fake.relative_to(REPO_ROOT)))
        assert "foo" in sigs
        assert "_bar" not in sigs
    finally:
        fake.unlink(missing_ok=True)


# ---------------------------------------------------------------- emitters


def test_write_json(tmp_path):
    p = tmp_path / "x.json"
    G.write_json(G.audit(), p)
    json.loads(p.read_text("utf-8"))


def test_write_md_single_trailing_newline(tmp_path):
    p = tmp_path / "x.md"
    G.write_md(G.audit(), p)
    text = p.read_text("utf-8")
    assert text.endswith("\n") and not text.endswith("\n\n")


def test_write_csv_header(tmp_path):
    p = tmp_path / "x.csv"
    G.write_csv(G.audit(), p)
    head = p.read_text("utf-8").splitlines()[0]
    assert head == "path,fn,args,defaults,returns,is_async,vararg,kwarg,kwonly"


# ---------------------------------------------------------------- injection


def _swap_registry(monkeypatch, mutator):
    new_reg = {rel: [dict(e) for e in stages]
               for rel, stages in G.STAGE_REGISTRY.items()}
    new_reg = mutator(new_reg)
    monkeypatch.setattr(G, "STAGE_REGISTRY", new_reg)


def test_s2_missing_fn_violation(monkeypatch):
    def mutate(reg):
        reg[G.PAPER_PIPELINE_PATH].append(
            G._entry("totally_bogus_xyz", [], [], "None"))
        return reg
    _swap_registry(monkeypatch, mutate)
    rules = {v["rule"] for v in G.audit()["violations"]}
    assert "S2" in rules


def test_s2_async_mismatch_violation(monkeypatch):
    def mutate(reg):
        for e in reg[G.PAPER_PIPELINE_PATH]:
            if e["name"] == "main":
                e["is_async"] = True
                break
        return reg
    _swap_registry(monkeypatch, mutate)
    rules = {v["rule"] for v in G.audit()["violations"]}
    assert "S2" in rules


def test_s3_args_reorder_violation(monkeypatch):
    def mutate(reg):
        for e in reg[G.PAPER_PIPELINE_PATH]:
            if e["name"] == "run_profile":
                e["args"] = ["profile", "run_root", "args"]
                break
        return reg
    _swap_registry(monkeypatch, mutate)
    rules = {v["rule"] for v in G.audit()["violations"]}
    assert "S3" in rules


def test_s4_defaults_mismatch_violation(monkeypatch):
    def mutate(reg):
        for e in reg[G.PAPER_PIPELINE_PATH]:
            if e["name"] == "metric_direction":
                e["args_has_default"] = [True, True, True]
                break
        return reg
    _swap_registry(monkeypatch, mutate)
    rules = {v["rule"] for v in G.audit()["violations"]}
    assert "S4" in rules


def test_s5_return_annotation_violation(monkeypatch):
    def mutate(reg):
        for e in reg[G.PAPER_PIPELINE_PATH]:
            if e["name"] == "main":
                e["returns"] = "bool"
                break
        return reg
    _swap_registry(monkeypatch, mutate)
    rules = {v["rule"] for v in G.audit()["violations"]}
    assert "S5" in rules


def test_s6_kwonly_drift_violation(monkeypatch):
    def mutate(reg):
        for e in reg[G.PAPER_PIPELINE_PATH]:
            if e["name"] == "main":
                e["kwonly"] = ["fake_kw"]
                e["kwonly_has_default"] = [False]
                break
        return reg
    _swap_registry(monkeypatch, mutate)
    rules = {v["rule"] for v in G.audit()["violations"]}
    assert "S6" in rules


def test_s7_exhaustive_violation(monkeypatch):
    def mutate(reg):
        reg[G.PAPER_PIPELINE_PATH] = [
            e for e in reg[G.PAPER_PIPELINE_PATH] if e["name"] != "main"
        ]
        return reg
    _swap_registry(monkeypatch, mutate)
    rules = {v["rule"] for v in G.audit()["violations"]}
    assert "S7" in rules


def test_main_cli_exit_zero(tmp_path):
    rc = G.main([
        "--json-out", str(tmp_path / "a.json"),
        "--md-out", str(tmp_path / "a.md"),
        "--csv-out", str(tmp_path / "a.csv"),
    ])
    assert rc == 0
