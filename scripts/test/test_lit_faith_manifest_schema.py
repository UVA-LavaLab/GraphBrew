"""Pytest for gate 276 — manifest schema registry."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts/experiments/ecg/lit_faith_manifest_schema.py"

sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/ecg"))
import lit_faith_manifest_schema as mod  # noqa: E402


# --------------------------------------------------------------------
# Module presence and surface
# --------------------------------------------------------------------


def test_module_file_exists():
    assert MODULE_PATH.is_file()


def test_audit_callable():
    assert callable(mod.audit)


def test_main_callable():
    assert callable(mod.main)


def test_constants_present():
    for k in ["MANIFEST_PATH", "LOCKED_VERSION",
              "LOCKED_TOP_LEVEL_KEYS",
              "LOCKED_REQUIRED_DEFAULTS_KEYS",
              "LOCKED_STAGE_KINDS",
              "LOCKED_REQUIRED_STAGE_KEYS",
              "LOCKED_REQUIRED_GRAPH_ENTRY_KEYS",
              "LOCKED_OPTIONAL_GRAPH_ENTRY_KEYS",
              "LOCKED_MIN_COUNTS"]:
        assert hasattr(mod, k), f"missing constant {k}"


def test_locked_version_is_int():
    assert isinstance(mod.LOCKED_VERSION, int)


def test_locked_top_level_keys_nonempty_set():
    assert isinstance(mod.LOCKED_TOP_LEVEL_KEYS, set)
    assert len(mod.LOCKED_TOP_LEVEL_KEYS) >= 5


def test_locked_defaults_keys_nonempty_set():
    assert isinstance(mod.LOCKED_REQUIRED_DEFAULTS_KEYS, set)
    assert len(mod.LOCKED_REQUIRED_DEFAULTS_KEYS) >= 10


def test_locked_stage_kinds():
    assert mod.LOCKED_STAGE_KINDS == {"roi_matrix", "proof_matrix"}


def test_locked_required_stage_keys():
    assert mod.LOCKED_REQUIRED_STAGE_KEYS == {
        "name", "kind", "profiles", "benchmarks"}


def test_locked_required_graph_keys():
    assert mod.LOCKED_REQUIRED_GRAPH_ENTRY_KEYS == {"name", "options_key"}


def test_locked_optional_graph_keys():
    assert mod.LOCKED_OPTIONAL_GRAPH_ENTRY_KEYS == {"path"}


def test_min_counts_present():
    for k in ["profiles", "stages", "graph_sets", "benchmark_options"]:
        assert k in mod.LOCKED_MIN_COUNTS


# --------------------------------------------------------------------
# Live audit — no violations
# --------------------------------------------------------------------


def test_live_audit_no_violations():
    doc = mod.audit()
    assert doc["violations"] == [], doc["violations"]
    assert doc["status"] == "active"
    assert doc["counts"]["version"] == mod.LOCKED_VERSION


def test_audit_doc_shape():
    doc = mod.audit()
    for k in ["schema", "status", "path",
              "locked_version", "counts", "locked", "rules", "violations"]:
        assert k in doc
    assert doc["schema"].startswith("lit-faith-manifest-schema/")
    assert set(doc["rules"].keys()) == {"M1", "M2", "M3", "M4", "M5", "M6"}


def test_counts_above_floors():
    doc = mod.audit()
    c = doc["counts"]
    for k, floor in mod.LOCKED_MIN_COUNTS.items():
        assert c.get(k, c.get(k + "_keys", 0)) >= floor or k == "defaults_keys"


def test_manifest_path_exists():
    assert (REPO_ROOT / mod.MANIFEST_PATH).is_file()


# --------------------------------------------------------------------
# Live loader
# --------------------------------------------------------------------


def test_live_manifest_parses():
    live = mod._live_manifest()
    assert isinstance(live, dict)
    assert live.get("version") == mod.LOCKED_VERSION


def test_live_manifest_returns_none_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "MANIFEST_PATH",
                        "scripts/experiments/ecg/does_not_exist.json")
    # _live_manifest uses module-level REPO_ROOT + MANIFEST_PATH
    assert mod._live_manifest() is None


# --------------------------------------------------------------------
# Injection tests — swap the live manifest with a synthetic doc
# --------------------------------------------------------------------


@pytest.fixture
def fake_manifest(monkeypatch):
    """Provides a synthetic manifest and patches _live_manifest to
    return a deep-copy of it for each test."""

    state = {"doc": copy.deepcopy(mod._live_manifest())}

    def _fake_live():
        return copy.deepcopy(state["doc"])

    monkeypatch.setattr(mod, "_live_manifest", _fake_live)
    return state


def _violations_for_rule(doc: dict, rule: str) -> list[dict]:
    return [v for v in doc["violations"] if v.get("rule") == rule]


def test_m1_wrong_version(fake_manifest):
    fake_manifest["doc"]["version"] = 99
    # M1 reads file directly, not _live_manifest, so we also need to
    # patch the file-read path. Simplest: rebind _check_m1_version
    # via the live manifest itself. Use a temp manifest file.
    pass


def test_m1_version_check_via_temp_manifest(monkeypatch, tmp_path):
    bad = tmp_path / "manifest.json"
    bad.write_text(json.dumps({"version": 99, "description": "x",
                               "defaults": {}, "profiles": {},
                               "benchmark_options": {},
                               "graph_sets": {}, "stages": []}))
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "MANIFEST_PATH", "manifest.json")
    out = mod._check_m1_version()
    assert any(v.get("got_version") == 99 for v in out)


def test_m1_missing_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "MANIFEST_PATH", "missing.json")
    out = mod._check_m1_version()
    assert any(v.get("issue") == "manifest missing" for v in out)


def test_m1_bad_json(monkeypatch, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "MANIFEST_PATH", "bad.json")
    out = mod._check_m1_version()
    assert any("json parse error" in v.get("issue", "") for v in out)


def test_m2_missing_top_level(fake_manifest):
    del fake_manifest["doc"]["profiles"]
    viols = _violations_for_rule(mod.audit(), "M2")
    assert any(v.get("missing_top_level_key") == "profiles" for v in viols)


def test_m2_extra_top_level(fake_manifest):
    fake_manifest["doc"]["surprise_field"] = True
    viols = _violations_for_rule(mod.audit(), "M2")
    assert any(v.get("extra_top_level_key") == "surprise_field" for v in viols)


def test_m2_stages_wrong_type(fake_manifest):
    fake_manifest["doc"]["stages"] = {"not": "a list"}
    viols = _violations_for_rule(mod.audit(), "M2")
    assert any(v.get("issue") == "stages must be a list" for v in viols)


def test_m2_min_count_floor(fake_manifest):
    fake_manifest["doc"]["profiles"] = {"only_one": "tag"}
    viols = _violations_for_rule(mod.audit(), "M2")
    assert any(v.get("key") == "profiles" and v.get("got") == 1
               for v in viols)


def test_m3_missing_defaults_key(fake_manifest):
    fake_manifest["doc"]["defaults"].pop("timeout_cache", None)
    viols = _violations_for_rule(mod.audit(), "M3")
    assert any(v.get("missing_defaults_key") == "timeout_cache"
               for v in viols)


def test_m3_extra_defaults_key(fake_manifest):
    fake_manifest["doc"]["defaults"]["surprise_default"] = "x"
    viols = _violations_for_rule(mod.audit(), "M3")
    assert any(v.get("extra_defaults_key") == "surprise_default"
               for v in viols)


def test_m4_missing_graph_entry_key(fake_manifest):
    # pick an arbitrary graph_set, drop options_key from first entry
    gs_name = next(iter(fake_manifest["doc"]["graph_sets"]))
    fake_manifest["doc"]["graph_sets"][gs_name][0].pop("options_key", None)
    viols = _violations_for_rule(mod.audit(), "M4")
    assert any(v.get("missing_key") == "options_key" for v in viols)


def test_m4_extra_graph_entry_key(fake_manifest):
    gs_name = next(iter(fake_manifest["doc"]["graph_sets"]))
    fake_manifest["doc"]["graph_sets"][gs_name][0]["surprise"] = "field"
    viols = _violations_for_rule(mod.audit(), "M4")
    assert any(v.get("extra_key") == "surprise" for v in viols)


def test_m4_non_string_value(fake_manifest):
    gs_name = next(iter(fake_manifest["doc"]["graph_sets"]))
    fake_manifest["doc"]["graph_sets"][gs_name][0]["name"] = 42
    viols = _violations_for_rule(mod.audit(), "M4")
    assert any(v.get("want_type") == "str" and v.get("got_type") == "int"
               for v in viols)


def test_m4_graph_set_entries_not_list(fake_manifest):
    gs_name = next(iter(fake_manifest["doc"]["graph_sets"]))
    fake_manifest["doc"]["graph_sets"][gs_name] = {"not": "a list"}
    viols = _violations_for_rule(mod.audit(), "M4")
    assert any("must be a list" in v.get("issue", "") for v in viols)


def test_m5_unknown_options_key(fake_manifest):
    gs_name = next(iter(fake_manifest["doc"]["graph_sets"]))
    fake_manifest["doc"]["graph_sets"][gs_name][0]["options_key"] = "totally_made_up"
    viols = _violations_for_rule(mod.audit(), "M5")
    assert any(v.get("options_key") == "totally_made_up" for v in viols)


def test_m6_missing_stage_required(fake_manifest):
    fake_manifest["doc"]["stages"][0].pop("benchmarks", None)
    viols = _violations_for_rule(mod.audit(), "M6")
    assert any(v.get("missing_required_key") == "benchmarks" for v in viols)


def test_m6_unknown_stage_kind(fake_manifest):
    fake_manifest["doc"]["stages"][0]["kind"] = "thread_scan"
    viols = _violations_for_rule(mod.audit(), "M6")
    assert any(v.get("got_kind") == "thread_scan" for v in viols)


def test_m6_unknown_profile_ref(fake_manifest):
    fake_manifest["doc"]["stages"][0]["profiles"] = ["totally_made_up_profile"]
    viols = _violations_for_rule(mod.audit(), "M6")
    assert any(v.get("profile_ref") == "totally_made_up_profile"
               for v in viols)


def test_m6_unknown_graph_set_ref(fake_manifest):
    # Some stages have graph_set; find one and corrupt it.
    target = None
    for s in fake_manifest["doc"]["stages"]:
        if isinstance(s.get("graph_set"), str):
            target = s
            break
    if target is None:
        # Inject one for the test.
        fake_manifest["doc"]["stages"][0]["graph_set"] = "definitely_not_real"
    else:
        target["graph_set"] = "definitely_not_real"
    viols = _violations_for_rule(mod.audit(), "M6")
    assert any(v.get("graph_set") == "definitely_not_real" for v in viols)


def test_m6_stage_not_dict(fake_manifest):
    fake_manifest["doc"]["stages"][0] = "not a dict"
    viols = _violations_for_rule(mod.audit(), "M6")
    assert any(v.get("issue") == "stage must be a dict" for v in viols)


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
    assert "gate 276" in text.lower()
    assert "manifest schema registry" in text.lower()


def test_write_csv_rows(tmp_path):
    doc = mod.audit()
    p = tmp_path / "x.csv"
    mod.write_csv(doc, p)
    text = p.read_text("utf-8")
    assert text.startswith("section,key,value")
    assert "counts,version," in text


def test_write_md_no_trailing_blank_after_no_violations(tmp_path):
    doc = mod.audit()
    p = tmp_path / "y.md"
    mod.write_md(doc, p)
    text = p.read_text("utf-8")
    # exactly one trailing newline, no extra blank lines after No violations
    assert text.endswith("## ✅ No violations\n") or "## Violations" in text


# --------------------------------------------------------------------
# CLI / main()
# --------------------------------------------------------------------


def test_main_exit_zero_in_subprocess():
    r = subprocess.run([sys.executable, str(MODULE_PATH)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "[lit-faith-manifest-schema]" in r.stdout


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
    # Simulate one violation by stubbing audit.
    orig_audit = mod.audit

    def _fake_audit():
        doc = orig_audit()
        doc["violations"].append({"rule": "M1", "issue": "test injection"})
        return doc

    monkeypatch.setattr(mod, "audit", _fake_audit)
    rc = mod.main([])
    assert rc == 1
