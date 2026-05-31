"""Tests for gate 278 — receiver CLI registry."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from scripts.experiments.ecg import lit_faith_receiver_cli_registry as M


# --------------------------------------------------------------------
# Structural
# --------------------------------------------------------------------


def test_registry_has_two_receivers():
    assert len(M.RECEIVER_CLI_REGISTRY) == 2
    assert M.PROOF_PATH in M.RECEIVER_CLI_REGISTRY
    assert M.ROI_PATH in M.RECEIVER_CLI_REGISTRY


def test_proof_flag_count():
    assert len(M.PROOF_FLAGS) == 17


def test_roi_flag_count():
    assert len(M.ROI_FLAGS) == 48


def test_total_flag_count():
    assert sum(len(v) for v in M.RECEIVER_CLI_REGISTRY.values()) == 65


def test_cross_pairs_present():
    assert len(M.SENDER_RECEIVER_PAIRS) == 2
    senders = {(fn, recv) for _, fn, recv in M.SENDER_RECEIVER_PAIRS}
    assert ("make_proof_job", M.PROOF_PATH) in senders
    assert ("make_roi_job", M.ROI_PATH) in senders


def test_every_flag_has_action():
    for rel, flags in M.RECEIVER_CLI_REGISTRY.items():
        for e in flags:
            assert "flag" in e
            assert "action" in e
            assert e["flag"].startswith("--")


# --------------------------------------------------------------------
# Live audit
# --------------------------------------------------------------------


def test_live_audit_zero_violations():
    doc = M.audit()
    assert doc["violations"] == [], doc["violations"]


def test_status_active():
    assert M.audit()["status"] == "active"


def test_schema():
    assert M.audit()["schema"] == "lit-faith-receiver-cli-registry/1"


def test_counts_match_registry():
    doc = M.audit()
    assert doc["counts"]["receivers"] == 2
    assert doc["counts"]["proof_flags"] == 17
    assert doc["counts"]["roi_flags"] == 48
    assert doc["counts"]["total_flags"] == 65
    assert doc["counts"]["cross_pairs"] == 2


def test_rules_present():
    rules = M.audit()["rules"]
    for r in ("E1", "E2", "E3", "E4", "E5", "E6", "E7"):
        assert r in rules


# --------------------------------------------------------------------
# Injection helpers
# --------------------------------------------------------------------


@pytest.fixture
def swap_registry(monkeypatch):
    def _swap(reg, pairs=None):
        monkeypatch.setattr(M, "RECEIVER_CLI_REGISTRY", reg)
        if pairs is not None:
            monkeypatch.setattr(M, "SENDER_RECEIVER_PAIRS", pairs)
    return _swap


def _proof_clone():
    return [dict(e) for e in M.PROOF_FLAGS]


def _roi_clone():
    return [dict(e) for e in M.ROI_FLAGS]


# --------------------------------------------------------------------
# E1 — module importable
# --------------------------------------------------------------------


def test_e1_missing_path(swap_registry):
    swap_registry({"scripts/experiments/ecg/__nope__.py": [
        {"flag": "--x", "action": "store"}]})
    viols = [v for v in M.audit()["violations"] if v["rule"] == "E1"]
    assert any(v["issue"] == "missing" for v in viols)


# --------------------------------------------------------------------
# E2 — parse_args present
# --------------------------------------------------------------------


def test_e2_no_parse_args(swap_registry, tmp_path, monkeypatch):
    fake = tmp_path / "no_parse_args.py"
    fake.write_text("def other(): pass\n")
    rel = str(fake.relative_to(tmp_path))
    monkeypatch.setattr(M, "REPO_ROOT", tmp_path)
    swap_registry({rel: [{"flag": "--x", "action": "store"}]})
    viols = [v for v in M.audit()["violations"] if v["rule"] == "E2"]
    assert any(v["issue"] == "no parse_args() fn" for v in viols)


# --------------------------------------------------------------------
# E3 — flag presence
# --------------------------------------------------------------------


def test_e3_missing_flag(swap_registry):
    flags = _proof_clone() + [{"flag": "--never-defined", "action": "store"}]
    swap_registry({M.PROOF_PATH: flags, M.ROI_PATH: _roi_clone()})
    viols = [v for v in M.audit()["violations"] if v["rule"] == "E3"]
    assert any(v["missing_flag"] == "--never-defined" for v in viols)


# --------------------------------------------------------------------
# E4 — action match
# --------------------------------------------------------------------


def test_e4_wrong_action(swap_registry):
    flags = _proof_clone()
    flags[0] = {**flags[0], "action": "store_true"}
    swap_registry({M.PROOF_PATH: flags, M.ROI_PATH: _roi_clone()})
    viols = [v for v in M.audit()["violations"] if v["rule"] == "E4"]
    assert any(v["flag"] == flags[0]["flag"] for v in viols)


# --------------------------------------------------------------------
# E5 — nargs match
# --------------------------------------------------------------------


def test_e5_wrong_nargs(swap_registry):
    flags = _proof_clone()
    # --ablations is nargs='+'; claim it's None
    for i, e in enumerate(flags):
        if e["flag"] == "--ablations":
            flags[i] = {"flag": e["flag"], "action": e["action"]}
    swap_registry({M.PROOF_PATH: flags, M.ROI_PATH: _roi_clone()})
    viols = [v for v in M.audit()["violations"] if v["rule"] == "E5"]
    assert any(v["flag"] == "--ablations" for v in viols)


def test_e5_extra_nargs(swap_registry):
    flags = _proof_clone()
    for i, e in enumerate(flags):
        if e["flag"] == "--out-dir":
            flags[i] = {**e, "nargs": "+"}
    swap_registry({M.PROOF_PATH: flags, M.ROI_PATH: _roi_clone()})
    viols = [v for v in M.audit()["violations"] if v["rule"] == "E5"]
    assert any(v["flag"] == "--out-dir" for v in viols)


# --------------------------------------------------------------------
# E6 — exhaustive
# --------------------------------------------------------------------


def test_e6_drops_existing_flag(swap_registry):
    flags = [e for e in _proof_clone() if e["flag"] != "--out-dir"]
    swap_registry({M.PROOF_PATH: flags, M.ROI_PATH: _roi_clone()})
    viols = [v for v in M.audit()["violations"] if v["rule"] == "E6"]
    assert any(v["flag"] == "--out-dir" for v in viols)


# --------------------------------------------------------------------
# E7 — cross-side parity
# --------------------------------------------------------------------


def test_e7_sender_flag_missing_in_receiver(swap_registry):
    # Drop --out-dir from proof receiver registry while keeping it on sender side.
    # E6 will also fire (live add_argument has it) but E7 should fire because
    # make_proof_job builds --out-dir into argv.
    flags = [e for e in _proof_clone() if e["flag"] != "--out-dir"]
    swap_registry({M.PROOF_PATH: flags, M.ROI_PATH: _roi_clone()})
    viols = [v for v in M.audit()["violations"] if v["rule"] == "E7"]
    assert any(v.get("missing_in_receiver") == "--out-dir" for v in viols)


def test_e7_missing_sender_fn(swap_registry):
    pairs = [("scripts/experiments/ecg/final_paper_run.py",
              "no_such_fn_278", M.PROOF_PATH)]
    swap_registry({M.PROOF_PATH: _proof_clone(),
                   M.ROI_PATH: _roi_clone()},
                  pairs=pairs + list(M.SENDER_RECEIVER_PAIRS))
    viols = [v for v in M.audit()["violations"] if v["rule"] == "E7"]
    assert any(v.get("issue") == "sender fn missing" for v in viols)


def test_e7_missing_sender_path(swap_registry):
    pairs = [("scripts/experiments/ecg/__missing__.py",
              "make_proof_job", M.PROOF_PATH)]
    swap_registry({M.PROOF_PATH: _proof_clone(),
                   M.ROI_PATH: _roi_clone()},
                  pairs=pairs + list(M.SENDER_RECEIVER_PAIRS))
    viols = [v for v in M.audit()["violations"] if v["rule"] == "E7"]
    assert any(v.get("issue") == "sender missing" for v in viols)


# --------------------------------------------------------------------
# Helper unit tests
# --------------------------------------------------------------------


def test_collect_add_argument_calls_parses_action_and_nargs():
    src = """
def parse_args():
    ap = ArgumentParser()
    ap.add_argument('--a')
    ap.add_argument('--b', action='store_true')
    ap.add_argument('--c', nargs='+')
    return ap
"""
    mod = ast.parse(src)
    fn = M._top_level_fn(mod, "parse_args")
    flags = M._collect_add_argument_calls(fn)
    by_flag = {e["flag"]: e for e in flags}
    assert by_flag["--a"]["action"] == "store"
    assert by_flag["--b"]["action"] == "store_true"
    assert by_flag["--c"]["nargs"] == "+"


def test_collect_add_argument_skips_non_flag_first_arg():
    src = """
def parse_args():
    ap = ArgumentParser()
    ap.add_argument('positional')
    ap.add_argument('--real')
    return ap
"""
    mod = ast.parse(src)
    fn = M._top_level_fn(mod, "parse_args")
    flags = M._collect_add_argument_calls(fn)
    assert [e["flag"] for e in flags] == ["--real"]


def test_collect_sender_flag_literals():
    src = """
def make_proof_job():
    return ['--out-dir', '/x', '--ablations', 'a', '--no-build']
"""
    mod = ast.parse(src)
    fn = M._top_level_fn(mod, "make_proof_job")
    got = M._collect_sender_flag_literals(fn)
    assert got == {"--out-dir", "--ablations", "--no-build"}


def test_top_level_fn_returns_none_for_missing():
    mod = ast.parse("x = 1\n")
    assert M._top_level_fn(mod, "main") is None


def test_parse_module_raises_on_syntax_error(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("def x(:\n")
    with pytest.raises(SyntaxError):
        M._parse_module(bad)


# --------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------


def test_write_json(tmp_path):
    p = tmp_path / "out.json"
    M.write_json(M.audit(), p)
    d = json.loads(p.read_text())
    assert d["schema"] == "lit-faith-receiver-cli-registry/1"
    assert d["counts"]["total_flags"] == 65


def test_write_md(tmp_path):
    p = tmp_path / "out.md"
    M.write_md(M.audit(), p)
    txt = p.read_text()
    assert "Receiver CLI registry (gate 278)" in txt
    assert "Cross-side parity pairs (E7)" in txt
    assert "✅ No violations" in txt
    assert txt.endswith("\n")


def test_write_csv(tmp_path):
    p = tmp_path / "out.csv"
    M.write_csv(M.audit(), p)
    txt = p.read_text()
    assert txt.startswith("path,flag,action,nargs")
    assert "--out-dir" in txt


# --------------------------------------------------------------------
# Main CLI
# --------------------------------------------------------------------


def test_main_returns_zero_when_clean(tmp_path, capsys):
    rc = M.main([
        "--json-out", str(tmp_path / "x.json"),
        "--md-out",   str(tmp_path / "x.md"),
        "--csv-out",  str(tmp_path / "x.csv"),
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "[lit-faith-receiver-cli-registry]" in out
    assert "violations=0" in out


def test_main_runs_with_no_args(capsys):
    rc = M.main([])
    assert rc == 0


# --------------------------------------------------------------------
# Sanity — registries reflect real receivers
# --------------------------------------------------------------------


def test_proof_path_exists():
    assert (M.REPO_ROOT / M.PROOF_PATH).is_file()


def test_roi_path_exists():
    assert (M.REPO_ROOT / M.ROI_PATH).is_file()


def test_every_receiver_has_at_least_one_flag():
    for rel, flags in M.RECEIVER_CLI_REGISTRY.items():
        assert len(flags) > 0, rel
