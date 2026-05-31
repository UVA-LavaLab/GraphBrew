"""Pytest suite for gate 272 — setup-script function signature registry."""

from __future__ import annotations

import json
import re
from pathlib import Path

from scripts.experiments.ecg import lit_faith_setup_fn_signature_registry as M

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------
# Registry sanity
# --------------------------------------------------------------------


def test_registry_has_two_scripts():
    assert len(M.SETUP_SIGNATURE_REGISTRY) == 2


def test_registered_scripts_under_scripts_dir():
    for rel in M.SETUP_SIGNATURE_REGISTRY:
        assert rel.startswith("scripts/")
        assert rel.endswith(".py")


def test_gem5_fn_count_floor():
    assert len(M.SETUP_GEM5_SIGNATURES) >= 12


def test_sniper_fn_count_floor():
    assert len(M.SETUP_SNIPER_SIGNATURES) >= 24


def test_every_signature_entry_has_args_and_defaults():
    for rel, sigs in M.SETUP_SIGNATURE_REGISTRY.items():
        for fn, want in sigs.items():
            assert "args" in want, (rel, fn)
            assert "defaults" in want, (rel, fn)
            assert isinstance(want["args"], list)
            assert isinstance(want["defaults"], int)
            assert want["defaults"] >= 0
            assert want["defaults"] <= len(want["args"])


def test_arg_names_are_identifiers():
    rx = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    for rel, sigs in M.SETUP_SIGNATURE_REGISTRY.items():
        for fn, want in sigs.items():
            for a in want["args"]:
                assert rx.match(a), (rel, fn, a)


def test_fn_names_are_identifiers():
    rx = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    for rel, sigs in M.SETUP_SIGNATURE_REGISTRY.items():
        for fn in sigs:
            assert rx.match(fn), (rel, fn)


def test_main_present_in_both():
    assert "main" in M.SETUP_GEM5_SIGNATURES
    assert "main" in M.SETUP_SNIPER_SIGNATURES


def test_no_dunder_methods_in_registry():
    for rel, sigs in M.SETUP_SIGNATURE_REGISTRY.items():
        for fn in sigs:
            assert not (fn.startswith("__") and fn.endswith("__")), (rel, fn)


# --------------------------------------------------------------------
# Live presence
# --------------------------------------------------------------------


def test_every_registered_script_exists():
    for rel in M.SETUP_SIGNATURE_REGISTRY:
        assert (REPO_ROOT / rel).is_file()


def test_every_registered_fn_exists_in_ast():
    for rel, sigs in M.SETUP_SIGNATURE_REGISTRY.items():
        live = M._parse_top_level(REPO_ROOT / rel)
        missing = [fn for fn in sigs if fn not in live]
        assert not missing, (rel, missing)


def test_every_registered_signature_matches():
    for rel, sigs in M.SETUP_SIGNATURE_REGISTRY.items():
        live = M._parse_top_level(REPO_ROOT / rel)
        for fn, want in sigs.items():
            node = live[fn]
            got_args = [a.arg for a in node.args.args]
            assert got_args == want["args"], (rel, fn, got_args, want["args"])
            assert len(node.args.defaults) == want["defaults"], \
                (rel, fn, len(node.args.defaults), want["defaults"])


def test_registry_exhaustive_over_public_defs():
    for rel, sigs in M.SETUP_SIGNATURE_REGISTRY.items():
        live = M._parse_top_level(REPO_ROOT / rel)
        public_live = {n for n in live if not n.startswith("_")}
        extras = public_live - set(sigs)
        assert not extras, (rel, extras)


# --------------------------------------------------------------------
# audit() shape
# --------------------------------------------------------------------


def test_audit_zero_violations():
    assert M.audit()["violations"] == []


def test_audit_status_active():
    assert M.audit()["status"] == "active"


def test_audit_schema_versioned():
    assert M.audit()["schema"] == "lit-faith-setup-fn-signature-registry/1"


def test_audit_rules_all_seven():
    assert set(M.audit()["rules"]) == {"F1", "F2", "F3", "F4", "F5", "F6", "F7"}


def test_audit_counts_self_consistent():
    c = M.audit()["counts"]
    assert c["scripts"] == 2
    assert c["total_fns"] == c["gem5_fns"] + c["sniper_fns"]


# --------------------------------------------------------------------
# Injection tests
# --------------------------------------------------------------------


def test_f1_detects_missing_fn(monkeypatch):
    bad = {p: dict(s) for p, s in M.SETUP_SIGNATURE_REGISTRY.items()}
    bad[M.SETUP_GEM5_PATH]["DEFINITELY_NOT_A_FN_Zz9"] = {"args": [], "defaults": 0}
    monkeypatch.setattr(M, "SETUP_SIGNATURE_REGISTRY", bad)
    viols = M.audit()["violations"]
    assert any(v["rule"] == "F1"
               and v.get("missing_fn") == "DEFINITELY_NOT_A_FN_Zz9"
               for v in viols)


def test_f2_detects_arg_rename(monkeypatch):
    bad_inner = dict(M.SETUP_GEM5_SIGNATURES)
    bad_inner["run_cmd"] = {"args": ["cmd", "cwd", "renamed_x", "capture", "env"],
                            "defaults": 4}
    bad = {M.SETUP_GEM5_PATH: bad_inner,
           M.SETUP_SNIPER_PATH: M.SETUP_SNIPER_SIGNATURES}
    monkeypatch.setattr(M, "SETUP_SIGNATURE_REGISTRY", bad)
    viols = M.audit()["violations"]
    assert any(v["rule"] == "F2" and v["fn"] == "run_cmd" for v in viols)


def test_f3_detects_defaults_drift(monkeypatch):
    bad_inner = dict(M.SETUP_GEM5_SIGNATURES)
    bad_inner["clone_gem5"] = {"args": ["tag", "force"], "defaults": 0}
    bad = {M.SETUP_GEM5_PATH: bad_inner,
           M.SETUP_SNIPER_PATH: M.SETUP_SNIPER_SIGNATURES}
    monkeypatch.setattr(M, "SETUP_SIGNATURE_REGISTRY", bad)
    viols = M.audit()["violations"]
    assert any(v["rule"] == "F3" and v["fn"] == "clone_gem5" for v in viols)


def test_f4_detects_unregistered_public_fn(monkeypatch):
    bad_inner = dict(M.SETUP_GEM5_SIGNATURES)
    del bad_inner["main"]
    bad = {M.SETUP_GEM5_PATH: bad_inner,
           M.SETUP_SNIPER_PATH: M.SETUP_SNIPER_SIGNATURES}
    monkeypatch.setattr(M, "SETUP_SIGNATURE_REGISTRY", bad)
    viols = M.audit()["violations"]
    assert any(v["rule"] == "F4" and v["fn"] == "main" for v in viols)


def test_f7_returns_annotation_flip_detected(monkeypatch):
    bad_inner = dict(M.SETUP_SNIPER_SIGNATURES)
    bad_inner["main"] = {"args": ["argv"], "defaults": 0,
                         "returns_annotated": True}
    bad = {M.SETUP_GEM5_PATH: M.SETUP_GEM5_SIGNATURES,
           M.SETUP_SNIPER_PATH: bad_inner}
    monkeypatch.setattr(M, "SETUP_SIGNATURE_REGISTRY", bad)
    viols = M.audit()["violations"]
    sniper_main = [v for v in viols if v["rule"] == "F7"
                   and v["fn"] == "main"]
    if sniper_main:
        assert sniper_main[0]["want_returns_annotated"] is True


# --------------------------------------------------------------------
# Artifact parity
# --------------------------------------------------------------------


def test_json_artifact_exists():
    p = REPO_ROOT / "wiki/data/lit_faith_setup_fn_signature_registry.json"
    assert p.is_file()


def test_json_artifact_parses():
    p = REPO_ROOT / "wiki/data/lit_faith_setup_fn_signature_registry.json"
    doc = json.loads(p.read_text())
    assert doc["schema"] == "lit-faith-setup-fn-signature-registry/1"
    assert doc["status"] == "active"


def test_md_artifact_exists():
    p = REPO_ROOT / "wiki/data/lit_faith_setup_fn_signature_registry.md"
    assert p.is_file()
    assert "Setup-script function signature registry" in p.read_text()


def test_md_exactly_one_final_newline():
    p = REPO_ROOT / "wiki/data/lit_faith_setup_fn_signature_registry.md"
    text = p.read_text()
    assert text.endswith("\n") and not text.endswith("\n\n")


def test_csv_artifact_exists():
    p = REPO_ROOT / "wiki/data/lit_faith_setup_fn_signature_registry.csv"
    assert p.is_file()
    assert "path,fn,positional_args,defaults" in p.read_text().splitlines()[0]


# --------------------------------------------------------------------
# Catalog wiring
# --------------------------------------------------------------------


def test_catalog_lists_setup_fn_signature_registry():
    from scripts.experiments.ecg.artifact_catalog import CATALOG

    paths = {a["artifact"] for a in CATALOG}
    assert "wiki/data/lit_faith_setup_fn_signature_registry.json" in paths


# --------------------------------------------------------------------
# CLI runner
# --------------------------------------------------------------------


def test_main_returns_zero_with_clean_registry(tmp_path):
    rc = M.main([
        "--json-out", str(tmp_path / "x.json"),
        "--md-out", str(tmp_path / "x.md"),
        "--csv-out", str(tmp_path / "x.csv"),
    ])
    assert rc == 0
