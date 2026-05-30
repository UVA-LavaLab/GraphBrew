"""Tests for gate 250 — paper-table CSV provenance.

Locks in:

  * generator emits an "active" status (latest paper_pipeline dir exists);
  * zero violations on the current registry + .tex/.csv pairs;
  * P1 (every registered .tex AND .csv file exists);
  * P2 (subset row-count: tex_rows <= csv_rows);
  * P3 (subset key-column parity: every tex key value traces to csv);
  * P4 (every declared key column exists in tex/csv header);
  * P5 (no empty value in tracked CSV key columns);
  * P6 (no unregistered CSV sibling of a registered .tex);
  * P7 (every registered CSV has a non-empty header row);
  * concrete count floors (>=1 pair per shipped table, >=2 key columns
    for at least one entry, >=10 total tex rows traced to CSV);
  * normalizer round-trip (latex normalizer un-escapes ``\\_``);
  * generator's JSON-on-disk matches audit() output (so committed
    artifacts can't silently drift from the live audit).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GEN_PATH = ROOT / "scripts" / "experiments" / "ecg" / "lit_faith_paper_provenance.py"
JSON_OUT = ROOT / "wiki" / "data" / "lit_faith_paper_provenance.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load("lit_faith_paper_provenance_dyn", GEN_PATH)


@pytest.fixture(scope="module")
def audit(gen):
    return gen.audit()


# --- shape -----------------------------------------------------------

def test_generator_imports(gen):
    assert hasattr(gen, "audit")
    assert hasattr(gen, "PROVENANCE_REGISTRY")
    assert hasattr(gen, "NORMALIZERS")
    assert hasattr(gen, "_parse_tex")
    assert hasattr(gen, "_parse_csv")
    assert hasattr(gen, "_latest_pipeline_dir")


def test_audit_returns_active_status(audit):
    assert audit["status"] == "active", (
        f"expected active, got {audit.get('status')} "
        f"reason={audit.get('reason')}"
    )


def test_audit_has_zero_violations(audit):
    assert audit["violations"] == [], (
        f"violations: {audit['violations'][:5]}")


def test_audit_advertises_7_rules(audit):
    assert set(audit["rules"].keys()) == {
        "P1", "P2", "P3", "P4", "P5", "P6", "P7"}


# --- registry shape --------------------------------------------------

def test_registry_nonempty(gen):
    assert len(gen.PROVENANCE_REGISTRY) >= 5


def test_every_registry_entry_has_required_fields(gen):
    required = {"tex_file", "csv_file", "key_columns"}
    for e in gen.PROVENANCE_REGISTRY:
        missing = required - set(e.keys())
        assert not missing, (
            f"entry {e.get('tex_file')!r} missing: {missing}")
        assert isinstance(e["key_columns"], list) and e["key_columns"]
        for pair in e["key_columns"]:
            assert isinstance(pair, tuple) and len(pair) == 2
            assert all(isinstance(x, str) and x for x in pair)


def test_registry_pairs_are_unique(gen):
    tex_names = [e["tex_file"] for e in gen.PROVENANCE_REGISTRY]
    csv_names = [e["csv_file"] for e in gen.PROVENANCE_REGISTRY]
    assert len(tex_names) == len(set(tex_names))
    assert len(csv_names) == len(set(csv_names))


def test_registry_normalizer_is_known(gen):
    for e in gen.PROVENANCE_REGISTRY:
        norm = e.get("normalizer", "strip")
        assert norm in gen.NORMALIZERS, (
            f"unknown normalizer {norm!r} in entry {e['tex_file']!r}")


def test_known_tables_registered(gen):
    names = {e["tex_file"] for e in gen.PROVENANCE_REGISTRY}
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
        "gate 244 should have caught this.")


def test_p1_all_files_present(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "P1"]
    assert not bad, f"P1 hits: {bad}"


def test_p2_no_orphan_tex_rows(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "P2"]
    assert not bad, f"P2 hits: {bad}"


def test_p3_no_orphan_key_values(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "P3"]
    assert not bad, f"P3 hits: {bad[:5]}"


def test_p4_every_key_column_present(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "P4"]
    assert not bad, f"P4 hits: {bad}"


def test_p5_no_empty_csv_key_values(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "P5"]
    assert not bad, f"P5 hits: {bad[:5]}"


def test_p6_no_unregistered_paired_csv(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "P6"]
    assert not bad, f"P6 hits: {bad}"


def test_p7_every_csv_has_header(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "P7"]
    assert not bad, f"P7 hits: {bad}"


# --- concrete floors -------------------------------------------------

def test_every_pair_present(audit):
    for fn, info in audit["per_pair"].items():
        assert info.get("tex_present"), f"{fn} .tex missing"
        assert info.get("csv_present"), f"{fn} .csv missing"


def test_tex_rows_floor(audit):
    # Today: 78 tex rows. Require >= 10.
    assert audit["totals"]["tex_rows_total"] >= 10


def test_csv_rows_floor(audit):
    # Today: 85 csv rows. Require >= tex_rows (subset invariant).
    assert audit["totals"]["csv_rows_total"] >= \
        audit["totals"]["tex_rows_total"]


def test_at_least_one_entry_has_multiple_key_columns(gen):
    multi = [e for e in gen.PROVENANCE_REGISTRY
             if len(e["key_columns"]) >= 2]
    assert multi, ("expected at least one registry entry to track "
                   "multiple key columns (compound provenance)")


def test_roi_summary_tracks_policy_benchmark_prefetcher(gen):
    e = next(x for x in gen.PROVENANCE_REGISTRY
             if x["tex_file"] == "roi_policy_summary.tex")
    keys = {pair[0] for pair in e["key_columns"]}
    assert {"policy", "benchmark", "prefetcher"} <= keys


def test_faithfulness_tracks_check(gen):
    e = next(x for x in gen.PROVENANCE_REGISTRY
             if x["tex_file"] == "faithfulness_summary.tex")
    keys = {pair[0] for pair in e["key_columns"]}
    assert "check" in keys


# --- normalizer round-trip -------------------------------------------

def test_latex_normalizer_unescapes_underscore(gen):
    assert gen._norm_latex("ECG\\_DBG\\_ONLY") == "ECG_DBG_ONLY"
    assert gen._norm_latex("POPT\\_CHARGED") == "POPT_CHARGED"
    assert gen._norm_latex("plain") == "plain"


def test_latex_normalizer_unescapes_percent(gen):
    assert gen._norm_latex("100\\%") == "100%"


def test_strip_normalizer_only_strips_whitespace(gen):
    assert gen._norm_strip("  ECG_DBG_ONLY  ") == "ECG_DBG_ONLY"
    # does NOT un-escape (it's the conservative normalizer)
    assert gen._norm_strip("ECG\\_DBG") == "ECG\\_DBG"


# --- helper invariants -----------------------------------------------

def test_parse_tex_extracts_header_and_rows(gen):
    pdir = gen._latest_pipeline_dir()
    p = pdir / "ecg_mode_overhead_summary.tex"
    info = gen._parse_tex(p)
    assert info["header"][0] == "policy"
    assert len(info["rows"]) >= 1


def test_parse_csv_extracts_header_and_rows(gen):
    pdir = gen._latest_pipeline_dir()
    p = pdir / "ecg_mode_overhead_summary.csv"
    info = gen._parse_csv(p)
    assert "policy_short" in info["header"]
    assert len(info["rows"]) >= 1


# --- artifact-on-disk parity -----------------------------------------

def test_audit_serialisable(audit):
    json.dumps(audit)


def test_on_disk_json_matches_live_audit(audit):
    if not JSON_OUT.exists():
        pytest.skip(
            f"{JSON_OUT} not yet generated; run `make lit-paper-provenance`.")
    on_disk = json.loads(JSON_OUT.read_text())
    assert on_disk["status"] == audit["status"]
    assert on_disk["totals"] == audit["totals"]
    assert on_disk["per_pair"] == audit["per_pair"]
    assert on_disk["violations"] == audit["violations"]


def test_on_disk_md_exists():
    md = ROOT / "wiki" / "data" / "lit_faith_paper_provenance.md"
    if md.exists():
        txt = md.read_text()
        assert "Paper-table CSV provenance" in txt
        assert "gate 250" in txt


def test_on_disk_csv_exists():
    csvp = ROOT / "wiki" / "data" / "lit_faith_paper_provenance.csv"
    if csvp.exists():
        txt = csvp.read_text()
        assert "registry_size" in txt
        assert "pairs_found" in txt
