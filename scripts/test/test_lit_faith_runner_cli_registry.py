"""Tests for ``lit_faith_runner_cli_registry`` (gate 273).

These tests act as the literature-faithfulness gate around the two
paper-experiment runners' argparse CLI surfaces (ECG + VLDB).

5 injection tests prove the audit's bite (R3-R6 actively fail).
"""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pytest

from scripts.experiments.ecg import lit_faith_runner_cli_registry as G

REPO_ROOT = Path(G.__file__).resolve().parents[3]


# ---------------------------------------------------------------- shape


def test_module_importable():
    importlib.reload(G)


def test_registry_is_dict():
    assert isinstance(G.RUNNER_CLI_REGISTRY, dict)


def test_registry_has_two_runners():
    assert len(G.RUNNER_CLI_REGISTRY) == 2


def test_ecg_runner_path_present():
    assert G.ECG_RUNNER_PATH in G.RUNNER_CLI_REGISTRY


def test_vldb_runner_path_present():
    assert G.VLDB_RUNNER_PATH in G.RUNNER_CLI_REGISTRY


def test_ecg_runner_path_is_relative():
    assert not Path(G.ECG_RUNNER_PATH).is_absolute()


def test_vldb_runner_path_is_relative():
    assert not Path(G.VLDB_RUNNER_PATH).is_absolute()


def test_ecg_flag_count():
    assert len(G.ECG_RUNNER_FLAGS) == 6


def test_vldb_flag_count():
    assert len(G.VLDB_RUNNER_FLAGS) == 12


def test_each_flag_entry_has_flag_and_action():
    for flags in G.RUNNER_CLI_REGISTRY.values():
        for e in flags:
            assert "flag" in e
            assert "action" in e


def test_each_flag_string_starts_with_double_dash():
    for flags in G.RUNNER_CLI_REGISTRY.values():
        for e in flags:
            assert e["flag"].startswith("--")


def test_no_duplicate_flags_per_runner():
    for flags in G.RUNNER_CLI_REGISTRY.values():
        names = [e["flag"] for e in flags]
        assert len(names) == len(set(names))


def test_action_values_known():
    known = {"store", "store_true", "store_false", "append", "count"}
    for flags in G.RUNNER_CLI_REGISTRY.values():
        for e in flags:
            assert e["action"] in known


# ---------------------------------------------------------------- key flags


def test_ecg_has_all_flag():
    names = {e["flag"] for e in G.ECG_RUNNER_FLAGS}
    assert "--all" in names


def test_ecg_has_exp_flag():
    names = {e["flag"] for e in G.ECG_RUNNER_FLAGS}
    assert "--exp" in names


def test_ecg_has_preview_flag():
    names = {e["flag"] for e in G.ECG_RUNNER_FLAGS}
    assert "--preview" in names


def test_ecg_has_dry_run_flag():
    names = {e["flag"] for e in G.ECG_RUNNER_FLAGS}
    assert "--dry-run" in names


def test_ecg_has_graph_dir_flag():
    names = {e["flag"] for e in G.ECG_RUNNER_FLAGS}
    assert "--graph-dir" in names


def test_vldb_has_64gb_flag():
    names = {e["flag"] for e in G.VLDB_RUNNER_FLAGS}
    assert "--64gb" in names


def test_vldb_has_local_flag():
    names = {e["flag"] for e in G.VLDB_RUNNER_FLAGS}
    assert "--local" in names


def test_vldb_has_figures_only_flag():
    names = {e["flag"] for e in G.VLDB_RUNNER_FLAGS}
    assert "--figures-only" in names


def test_ecg_exp_is_nargs_plus():
    entry = next(e for e in G.ECG_RUNNER_FLAGS if e["flag"] == "--exp")
    assert entry.get("nargs") == "+"


def test_vldb_exp_is_nargs_plus():
    entry = next(e for e in G.VLDB_RUNNER_FLAGS if e["flag"] == "--exp")
    assert entry.get("nargs") == "+"


def test_vldb_graphs_is_nargs_plus():
    entry = next(e for e in G.VLDB_RUNNER_FLAGS if e["flag"] == "--graphs")
    assert entry.get("nargs") == "+"


# ---------------------------------------------------------------- live


def test_live_ecg_clean():
    viols = [v for v in G.audit()["violations"]
             if v.get("path") == G.ECG_RUNNER_PATH]
    assert viols == []


def test_live_vldb_clean():
    viols = [v for v in G.audit()["violations"]
             if v.get("path") == G.VLDB_RUNNER_PATH]
    assert viols == []


def test_audit_no_violations_globally():
    assert G.audit()["violations"] == []


def test_audit_counts_shape():
    counts = G.audit()["counts"]
    assert counts["runners"] == 2
    assert counts["ecg_flags"] == 6
    assert counts["vldb_flags"] == 12
    assert counts["total_flags"] == 18


def test_runner_files_exist_on_disk():
    for rel in G.RUNNER_CLI_REGISTRY:
        assert (REPO_ROOT / rel).is_file()


def test_main_fn_present_in_both_runners():
    for rel in G.RUNNER_CLI_REGISTRY:
        mod = ast.parse((REPO_ROOT / rel).read_text("utf-8"))
        names = [n.name for n in mod.body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        assert "main" in names


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
    assert p.read_text("utf-8").splitlines()[0] == "path,flag,action,nargs"


# ---------------------------------------------------------------- injection


def _swap_registry(monkeypatch, mutator):
    new_reg = {rel: list(flags)
               for rel, flags in G.RUNNER_CLI_REGISTRY.items()}
    new_reg = mutator(new_reg)
    monkeypatch.setattr(G, "RUNNER_CLI_REGISTRY", new_reg)


def test_r3_missing_flag_violation(monkeypatch):
    def mutate(reg):
        reg[G.ECG_RUNNER_PATH] = reg[G.ECG_RUNNER_PATH] + [
            {"flag": "--bogus-not-real", "action": "store_true"},
        ]
        return reg
    _swap_registry(monkeypatch, mutate)
    rules = {v["rule"] for v in G.audit()["violations"]}
    assert "R3" in rules


def test_r4_wrong_action_violation(monkeypatch):
    def mutate(reg):
        reg[G.ECG_RUNNER_PATH] = [
            {**e, "action": "store"} if e["flag"] == "--all" else e
            for e in reg[G.ECG_RUNNER_PATH]
        ]
        return reg
    _swap_registry(monkeypatch, mutate)
    rules = {v["rule"] for v in G.audit()["violations"]}
    assert "R4" in rules


def test_r5_wrong_nargs_violation(monkeypatch):
    def mutate(reg):
        reg[G.ECG_RUNNER_PATH] = [
            {**e, "nargs": "*"} if e["flag"] == "--exp" else e
            for e in reg[G.ECG_RUNNER_PATH]
        ]
        return reg
    _swap_registry(monkeypatch, mutate)
    rules = {v["rule"] for v in G.audit()["violations"]}
    assert "R5" in rules


def test_r6_exhaustive_violation(monkeypatch):
    def mutate(reg):
        reg[G.ECG_RUNNER_PATH] = [e for e in reg[G.ECG_RUNNER_PATH]
                                  if e["flag"] != "--all"]
        return reg
    _swap_registry(monkeypatch, mutate)
    rules = {v["rule"] for v in G.audit()["violations"]}
    assert "R6" in rules


def test_main_cli_exit_zero(tmp_path):
    rc = G.main([
        "--json-out", str(tmp_path / "a.json"),
        "--md-out", str(tmp_path / "a.md"),
        "--csv-out", str(tmp_path / "a.csv"),
    ])
    assert rc == 0
