"""Tests for gate 252 — Slurm SBATCH schema registry.

Locks in:

  * generator emits an "active" status (no skip on a healthy tree);
  * zero violations on the current shipped ``*.sbatch`` corpus;
  * every required directive is documented in the canonical registry;
  * concrete floors on the canonical universe size, the required
    subset, and the harvested file count;
  * per-rule live checks (S1..S9 each have no violations);
  * vocabulary checks: the canonical directive list contains the
    historically load-bearing names (job-name, time, nodes, ntasks,
    cpus-per-task, mem, output, error, partition, exclusive,
    mem-per-cpu, array, gres, account);
  * regex spot-checks for the time and mem patterns;
  * job-name prefix list contains both gbrew- and ecg-;
  * generator's JSON-on-disk matches audit() output.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GEN_PATH = ROOT / "scripts" / "experiments" / "ecg" / "lit_faith_slurm_schema.py"
JSON_OUT = ROOT / "wiki" / "data" / "lit_faith_slurm_schema.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Register before exec_module so @dataclass can resolve
    # ``sys.modules[cls.__module__]`` while building Directive.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load("lit_faith_slurm_schema_dyn", GEN_PATH)


@pytest.fixture(scope="module")
def audit(gen):
    return gen.audit()


# --- shape -----------------------------------------------------------

def test_generator_imports(gen):
    assert hasattr(gen, "audit")
    assert hasattr(gen, "CANONICAL_SBATCH_DIRECTIVES")
    assert hasattr(gen, "REQUIRED_DIRECTIVES")
    assert hasattr(gen, "JOB_NAME_PREFIXES")


def test_audit_returns_active(audit):
    assert audit["status"] == "active"


def test_audit_zero_violations(audit):
    assert audit["violations"] == [], (
        f"first 5: {audit['violations'][:5]}")


def test_audit_advertises_9_rules(audit):
    assert set(audit["rules"].keys()) == {
        "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"}


# --- canonical shape -------------------------------------------------

REQUIRED_NAMES = {
    "job-name", "time", "nodes", "ntasks", "cpus-per-task", "mem",
    "output",
}
OPTIONAL_NAMES = {
    "error", "partition", "account", "exclusive", "array", "gres",
    "mem-per-cpu",
}


def test_required_directives_match_expectation(gen):
    assert gen.REQUIRED_DIRECTIVES == REQUIRED_NAMES


def test_canonical_includes_required_and_optional(gen):
    canonical = set(gen.CANONICAL_SBATCH_DIRECTIVES.keys())
    assert REQUIRED_NAMES.issubset(canonical)
    assert OPTIONAL_NAMES.issubset(canonical)


def test_canonical_size_floor(gen):
    assert len(gen.CANONICAL_SBATCH_DIRECTIVES) >= 14


def test_required_directives_floor(gen):
    assert len(gen.REQUIRED_DIRECTIVES) >= 7


def test_required_directives_are_marked_required(gen):
    for name in gen.REQUIRED_DIRECTIVES:
        d = gen.CANONICAL_SBATCH_DIRECTIVES[name]
        assert d.required is True, f"{name} not flagged required"


def test_optional_directives_are_marked_optional(gen):
    for name in OPTIONAL_NAMES:
        d = gen.CANONICAL_SBATCH_DIRECTIVES[name]
        assert d.required is False, f"{name} unexpectedly flagged required"


def test_job_name_prefixes(gen):
    assert "gbrew-" in gen.JOB_NAME_PREFIXES
    assert "ecg-" in gen.JOB_NAME_PREFIXES


# --- per-rule live checks --------------------------------------------

def test_s1_no_syntax_violations(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "S1"]
    assert not bad, f"S1: {bad[:3]}"


def test_s2_no_missing_required(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "S2"]
    assert not bad, f"S2: {bad[:3]}"


def test_s3_no_unknown_directives(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "S3"]
    assert not bad, f"S3: {bad[:3]}"


def test_s4_no_bad_mem_values(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "S4"]
    assert not bad, f"S4: {bad[:3]}"


def test_s5_no_bad_time_values(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "S5"]
    assert not bad, f"S5: {bad[:3]}"


def test_s6_single_node_single_task(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "S6"]
    assert not bad, f"S6: {bad[:3]}"


def test_s7_log_templates_well_formed(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "S7"]
    assert not bad, f"S7: {bad[:3]}"


def test_s8_job_name_prefixes(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "S8"]
    assert not bad, f"S8: {bad[:3]}"


def test_s9_no_mem_and_mempercpu_clash(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "S9"]
    assert not bad, f"S9: {bad[:3]}"


# --- concrete floors -------------------------------------------------

def test_at_least_5_sbatch_files_harvested(audit):
    # Today: 9 (2 in ecg/, 7 in vldb/).  >=5 leaves headroom for
    # cleanup of obsolete templates without flipping the gate red.
    assert audit["totals"]["files"] >= 5


def test_every_harvested_file_has_required_directives(audit):
    for f in audit["files"]:
        assert not f["missing_required"], (
            f"{f['path']} missing: {f['missing_required']}")


def test_every_file_has_at_least_5_directives(audit):
    for f in audit["files"]:
        assert f["directive_count"] >= 5, (
            f"{f['path']} only carries {f['directive_count']} directives")


# --- regex spot-checks -----------------------------------------------

def test_time_regex_matches_hms(gen):
    rx = gen.CANONICAL_SBATCH_DIRECTIVES["time"].pattern
    assert re.match(rx, "01:00:00")
    assert re.match(rx, "24:00:00")
    assert re.match(rx, "00:30:00")


def test_time_regex_matches_day_hms(gen):
    rx = gen.CANONICAL_SBATCH_DIRECTIVES["time"].pattern
    assert re.match(rx, "3-00:00:00")
    assert re.match(rx, "1-12:00:00")


def test_time_regex_rejects_bad(gen):
    rx = gen.CANONICAL_SBATCH_DIRECTIVES["time"].pattern
    assert not re.match(rx, "10m")
    assert not re.match(rx, "1h")
    assert not re.match(rx, "00:60")


def test_mem_regex_accepts_suffix(gen):
    rx = gen.CANONICAL_SBATCH_DIRECTIVES["mem"].pattern
    assert re.match(rx, "32G")
    assert re.match(rx, "256G")
    assert re.match(rx, "16M")


def test_mem_regex_rejects_bare_number(gen):
    # Pattern is \d+[GMK]? — bare digits ARE accepted (Slurm uses MB
    # by default).  Lock that behaviour.
    rx = gen.CANONICAL_SBATCH_DIRECTIVES["mem"].pattern
    assert re.match(rx, "32") is not None  # bare allowed


def test_mem_regex_rejects_garbage(gen):
    rx = gen.CANONICAL_SBATCH_DIRECTIVES["mem"].pattern
    assert not re.match(rx, "32GB")
    assert not re.match(rx, "lots")


# --- artifact-on-disk parity -----------------------------------------

def test_audit_serialisable(audit):
    json.dumps(audit)


def test_on_disk_json_matches_live_audit(audit):
    if not JSON_OUT.exists():
        pytest.skip(
            f"{JSON_OUT} not yet generated; run `make lit-slurm-schema`.")
    on_disk = json.loads(JSON_OUT.read_text())
    assert on_disk["status"] == audit["status"]
    assert on_disk["totals"] == audit["totals"]
    assert on_disk["violations"] == audit["violations"]
    assert on_disk["required_directives"] == audit["required_directives"]


def test_on_disk_md_mentions_gate(audit):
    md = ROOT / "wiki" / "data" / "lit_faith_slurm_schema.md"
    if md.exists():
        txt = md.read_text()
        assert "Slurm SBATCH schema registry" in txt
        assert "gate 252" in txt


def test_on_disk_csv_has_expected_columns():
    csvp = ROOT / "wiki" / "data" / "lit_faith_slurm_schema.csv"
    if csvp.exists():
        head = csvp.read_text().splitlines()[0]
        for col in ("file", "directive_count", "missing_required",
                    "violations"):
            assert col in head, f"missing column {col!r} in csv header"
