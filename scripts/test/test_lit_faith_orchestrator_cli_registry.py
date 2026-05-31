"""Tests for ``lit_faith_orchestrator_cli_registry`` (gate 274)."""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pytest

from scripts.experiments.ecg import lit_faith_orchestrator_cli_registry as G

REPO_ROOT = Path(G.__file__).resolve().parents[3]


# ---------------------------------------------------------------- shape


def test_module_importable():
    importlib.reload(G)


def test_registry_is_dict():
    assert isinstance(G.ORCHESTRATOR_CLI_REGISTRY, dict)


def test_registry_has_two_orchestrators():
    assert len(G.ORCHESTRATOR_CLI_REGISTRY) == 2


def test_paper_pipeline_path_present():
    assert G.PAPER_PIPELINE_PATH in G.ORCHESTRATOR_CLI_REGISTRY


def test_final_paper_run_path_present():
    assert G.FINAL_PAPER_RUN_PATH in G.ORCHESTRATOR_CLI_REGISTRY


def test_paper_pipeline_path_is_relative():
    assert not Path(G.PAPER_PIPELINE_PATH).is_absolute()


def test_final_paper_run_path_is_relative():
    assert not Path(G.FINAL_PAPER_RUN_PATH).is_absolute()


def test_paper_pipeline_flag_count():
    assert len(G.PAPER_PIPELINE_FLAGS) == 14


def test_final_paper_run_flag_count():
    assert len(G.FINAL_PAPER_RUN_FLAGS) == 29


def test_total_flag_count():
    assert (len(G.PAPER_PIPELINE_FLAGS)
            + len(G.FINAL_PAPER_RUN_FLAGS)) == 43


def test_fn_name_locked_to_parse_args():
    for spec in G.ORCHESTRATOR_CLI_REGISTRY.values():
        assert spec["fn_name"] == "parse_args"


def test_each_flag_entry_has_flag_and_action():
    for spec in G.ORCHESTRATOR_CLI_REGISTRY.values():
        for e in spec["flags"]:
            assert "flag" in e
            assert "action" in e


def test_each_flag_string_starts_with_double_dash():
    for spec in G.ORCHESTRATOR_CLI_REGISTRY.values():
        for e in spec["flags"]:
            assert e["flag"].startswith("--")


def test_no_duplicate_flags_per_orchestrator():
    for spec in G.ORCHESTRATOR_CLI_REGISTRY.values():
        names = [e["flag"] for e in spec["flags"]]
        assert len(names) == len(set(names))


def test_action_values_known():
    known = {"store", "store_true", "store_false", "append",
             "count", "BooleanOptionalAction"}
    for spec in G.ORCHESTRATOR_CLI_REGISTRY.values():
        for e in spec["flags"]:
            assert e["action"] in known


# ---------------------------------------------------------------- key flags


def test_paper_pipeline_has_profiles_flag():
    names = {e["flag"] for e in G.PAPER_PIPELINE_FLAGS}
    assert "--profiles" in names


def test_paper_pipeline_has_skip_literature_gate_flag():
    names = {e["flag"] for e in G.PAPER_PIPELINE_FLAGS}
    assert "--skip-literature-gate" in names


def test_paper_pipeline_has_copy_to_paper_flag():
    names = {e["flag"] for e in G.PAPER_PIPELINE_FLAGS}
    assert "--copy-to-paper" in names


def test_paper_pipeline_has_no_stop_on_error_store_false():
    entry = next(e for e in G.PAPER_PIPELINE_FLAGS
                 if e["flag"] == "--no-stop-on-error")
    assert entry["action"] == "store_false"


def test_final_paper_run_has_manifest_flag():
    names = {e["flag"] for e in G.FINAL_PAPER_RUN_FLAGS}
    assert "--manifest" in names


def test_final_paper_run_has_profile_flag():
    names = {e["flag"] for e in G.FINAL_PAPER_RUN_FLAGS}
    assert "--profile" in names


def test_final_paper_run_resume_is_boolean_optional_action():
    entry = next(e for e in G.FINAL_PAPER_RUN_FLAGS
                 if e["flag"] == "--resume")
    assert entry["action"] == "BooleanOptionalAction"


def test_final_paper_run_stop_on_error_is_boolean_optional_action():
    entry = next(e for e in G.FINAL_PAPER_RUN_FLAGS
                 if e["flag"] == "--stop-on-error")
    assert entry["action"] == "BooleanOptionalAction"


def test_final_paper_run_lock_path_present():
    names = {e["flag"] for e in G.FINAL_PAPER_RUN_FLAGS}
    assert "--lock-path" in names


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
    assert counts["paper_pipeline_flags"] == 14
    assert counts["final_paper_run_flags"] == 29
    assert counts["total_flags"] == 43


def test_orchestrator_files_exist_on_disk():
    for rel in G.ORCHESTRATOR_CLI_REGISTRY:
        assert (REPO_ROOT / rel).is_file()


def test_parse_args_present_in_both_orchestrators():
    for rel, spec in G.ORCHESTRATOR_CLI_REGISTRY.items():
        mod = ast.parse((REPO_ROOT / rel).read_text("utf-8"))
        names = [n.name for n in mod.body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        assert spec["fn_name"] in names


# ---------------------------------------------------------------- helpers


def test_action_text_recognises_constant():
    node = ast.parse('"store_true"', mode="eval").body
    assert G._action_text(node) == "store_true"


def test_action_text_recognises_attribute():
    node = ast.parse("argparse.BooleanOptionalAction", mode="eval").body
    assert G._action_text(node) == "BooleanOptionalAction"


def test_action_text_recognises_bare_name():
    node = ast.parse("BooleanOptionalAction", mode="eval").body
    assert G._action_text(node) == "BooleanOptionalAction"


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
    assert head == "path,fn_name,flag,action,nargs,required"


# ---------------------------------------------------------------- injection


def _swap_registry(monkeypatch, mutator):
    new_reg = {rel: {"fn_name": spec["fn_name"],
                     "flags": list(spec["flags"])}
               for rel, spec in G.ORCHESTRATOR_CLI_REGISTRY.items()}
    new_reg = mutator(new_reg)
    monkeypatch.setattr(G, "ORCHESTRATOR_CLI_REGISTRY", new_reg)


def test_o2_wrong_fn_name_violation(monkeypatch):
    def mutate(reg):
        reg[G.PAPER_PIPELINE_PATH]["fn_name"] = "totally_bogus_fn_xyz"
        return reg
    _swap_registry(monkeypatch, mutate)
    rules = {v["rule"] for v in G.audit()["violations"]}
    assert "O2" in rules


def test_o3_missing_flag_violation(monkeypatch):
    def mutate(reg):
        reg[G.PAPER_PIPELINE_PATH]["flags"].append(
            {"flag": "--bogus-not-real", "action": "store_true"})
        return reg
    _swap_registry(monkeypatch, mutate)
    rules = {v["rule"] for v in G.audit()["violations"]}
    assert "O3" in rules


def test_o4_wrong_action_violation(monkeypatch):
    def mutate(reg):
        flags = reg[G.PAPER_PIPELINE_PATH]["flags"]
        reg[G.PAPER_PIPELINE_PATH]["flags"] = [
            {**e, "action": "store"} if e["flag"] == "--skip-run" else e
            for e in flags
        ]
        return reg
    _swap_registry(monkeypatch, mutate)
    rules = {v["rule"] for v in G.audit()["violations"]}
    assert "O4" in rules


def test_o5_wrong_nargs_violation(monkeypatch):
    def mutate(reg):
        flags = reg[G.PAPER_PIPELINE_PATH]["flags"]
        reg[G.PAPER_PIPELINE_PATH]["flags"] = [
            {**e, "nargs": "*"} if e["flag"] == "--profiles" else e
            for e in flags
        ]
        return reg
    _swap_registry(monkeypatch, mutate)
    rules = {v["rule"] for v in G.audit()["violations"]}
    assert "O5" in rules


def test_o6_exhaustive_violation(monkeypatch):
    def mutate(reg):
        reg[G.PAPER_PIPELINE_PATH]["flags"] = [
            e for e in reg[G.PAPER_PIPELINE_PATH]["flags"]
            if e["flag"] != "--profiles"
        ]
        return reg
    _swap_registry(monkeypatch, mutate)
    rules = {v["rule"] for v in G.audit()["violations"]}
    assert "O6" in rules


def test_o7_required_violation(monkeypatch):
    def mutate(reg):
        flags = reg[G.PAPER_PIPELINE_PATH]["flags"]
        reg[G.PAPER_PIPELINE_PATH]["flags"] = [
            {**e, "required": True} if e["flag"] == "--profiles" else e
            for e in flags
        ]
        return reg
    _swap_registry(monkeypatch, mutate)
    rules = {v["rule"] for v in G.audit()["violations"]}
    assert "O7" in rules


def test_main_cli_exit_zero(tmp_path):
    rc = G.main([
        "--json-out", str(tmp_path / "a.json"),
        "--md-out", str(tmp_path / "a.md"),
        "--csv-out", str(tmp_path / "a.csv"),
    ])
    assert rc == 0
