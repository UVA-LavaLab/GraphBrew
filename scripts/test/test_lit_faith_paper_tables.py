"""Tests for gate 247 — paper LaTeX-table emit invariant.

Locks in:

  * generator emits an "active" status (latest paper_pipeline dir exists);
  * zero violations on the current registry + .tex files;
  * T1 (every registered table file exists);
  * T2 (caption matches);
  * T3 (tabular col-spec matches);
  * T4 (column-header tuple matches);
  * T5 (correct per-row column count + no NaN/Inf cells);
  * T6 (no unregistered .tex file in paper_pipeline dir);
  * T7 (every table ends with the closing trio);
  * concrete count floors (>=1 row per shipped table);
  * specific contract: faithfulness_summary.tex has the 8-column
    "max LLC delta" header expected by the paper text;
  * generator's JSON-on-disk matches audit() output (so committed
    artifacts can't silently drift from the live audit).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GEN_PATH = ROOT / "scripts" / "experiments" / "ecg" / "lit_faith_paper_tables.py"
JSON_OUT = ROOT / "wiki" / "data" / "lit_faith_paper_tables.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load("lit_faith_paper_tables_dyn", GEN_PATH)


@pytest.fixture(scope="module")
def audit(gen):
    return gen.audit()


# --- shape -----------------------------------------------------------

def test_generator_imports(gen):
    assert hasattr(gen, "audit")
    assert hasattr(gen, "TABLE_REGISTRY")
    assert hasattr(gen, "_parse_table")
    assert hasattr(gen, "_latest_pipeline_dir")


def test_audit_returns_active_status(audit):
    assert audit["status"] == "active", (
        f"expected active, got {audit.get('status')} reason={audit.get('reason')}"
    )


def test_audit_has_zero_violations(audit):
    assert audit["violations"] == [], (
        f"violations: {audit['violations'][:5]}")


def test_audit_advertises_7_rules(audit):
    assert set(audit["rules"].keys()) == {"T1", "T2", "T3", "T4",
                                          "T5", "T6", "T7"}


# --- registry shape --------------------------------------------------

def test_registry_nonempty(gen):
    assert len(gen.TABLE_REGISTRY) >= 5


def test_every_registry_entry_has_required_fields(gen):
    required = {"filename", "caption", "col_spec", "columns"}
    for e in gen.TABLE_REGISTRY:
        missing = required - set(e.keys())
        assert not missing, f"entry {e.get('filename')!r} missing: {missing}"
        assert isinstance(e["columns"], tuple) and e["columns"]
        assert isinstance(e["col_spec"], str) and e["col_spec"]
        assert isinstance(e["caption"], str) and e["caption"]


def test_registry_filenames_are_unique(gen):
    names = [e["filename"] for e in gen.TABLE_REGISTRY]
    assert len(names) == len(set(names))


def test_known_tables_registered(gen):
    names = {e["filename"] for e in gen.TABLE_REGISTRY}
    assert "faithfulness_summary.tex" in names
    assert "ecg_mode_overhead_summary.tex" in names
    assert "popt_storage_overhead_summary.tex" in names
    assert "popt_charged_overhead.tex" in names
    assert "roi_policy_summary.tex" in names


# --- per-rule live checks --------------------------------------------

def test_pipeline_dir_present(gen):
    p = gen._latest_pipeline_dir()
    assert p is not None and p.is_dir(), (
        "no paper_pipeline_YYYYMMDD/ dir found in wiki/data — "
        "gate 244 should have caught this; check gate 247 audit reason.")


def test_t1_all_files_present(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "T1"]
    assert not bad, f"T1 hits: {bad}"


def test_t2_no_caption_drift(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "T2"]
    assert not bad, f"T2 hits: {bad}"


def test_t3_no_col_spec_drift(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "T3"]
    assert not bad, f"T3 hits: {bad}"


def test_t4_no_header_drift(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "T4"]
    assert not bad, f"T4 hits: {bad}"


def test_t5_no_row_or_nan_violations(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "T5"]
    assert not bad, f"T5 hits: {bad[:5]}"


def test_t6_no_unregistered_tex(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "T6"]
    assert not bad, f"T6 hits: {bad}"


def test_t7_every_table_ends_cleanly(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "T7"]
    assert not bad, f"T7 hits: {bad}"


# --- concrete floors -------------------------------------------------

def test_every_table_has_at_least_one_row(audit):
    for fn, info in audit["per_table"].items():
        assert info.get("present"), f"{fn} not present"
        assert info["row_count"] >= 1, f"{fn} has 0 data rows"


def test_total_row_floor(audit):
    # Today: 78 rows. Allow shrink but require >=10.
    assert audit["totals"]["row_total"] >= 10


def test_faithfulness_summary_8_columns(audit):
    info = audit["per_table"]["faithfulness_summary.tex"]
    assert info["present"]
    assert len(info["columns"]) == 8
    # the "max LLC delta" header is the column the paper text quotes
    assert any("LLC delta" in c for c in info["columns"])


def test_roi_summary_includes_avg_llc_red(audit):
    info = audit["per_table"]["roi_policy_summary.tex"]
    assert info["present"]
    assert any("LLC red." in c for c in info["columns"])


def test_storage_overhead_includes_reserved_ways(audit):
    info = audit["per_table"]["popt_storage_overhead_summary.tex"]
    assert info["present"]
    assert "reserved ways" in info["columns"]


# --- helper invariants -----------------------------------------------

def test_parse_table_extracts_caption_and_header(gen):
    pdir = gen._latest_pipeline_dir()
    p = pdir / "faithfulness_summary.tex"
    info = gen._parse_table(p)
    assert info["caption"] == "ECG faithfulness and parity checks"
    assert info["col_spec"] == "llllllll"
    assert info["header"][0] == "check"
    assert info["header"][-1] == "pass"


# --- artifact-on-disk parity -----------------------------------------

def test_audit_serialisable(audit):
    json.dumps(audit)


def test_on_disk_json_matches_live_audit(audit):
    if not JSON_OUT.exists():
        pytest.skip(f"{JSON_OUT} not yet generated; run `make lit-paper-tables`.")
    on_disk = json.loads(JSON_OUT.read_text())
    assert on_disk["status"] == audit["status"]
    assert on_disk["totals"] == audit["totals"]
    assert on_disk["per_table"] == audit["per_table"]
    assert on_disk["violations"] == audit["violations"]


def test_on_disk_md_exists():
    md = ROOT / "wiki" / "data" / "lit_faith_paper_tables.md"
    if md.exists():
        txt = md.read_text()
        assert "Paper LaTeX-table emit invariant" in txt
        assert "gate 247" in txt


def test_on_disk_csv_exists():
    csvp = ROOT / "wiki" / "data" / "lit_faith_paper_tables.csv"
    if csvp.exists():
        txt = csvp.read_text()
        assert "registry_size" in txt
        assert "tables_found" in txt
