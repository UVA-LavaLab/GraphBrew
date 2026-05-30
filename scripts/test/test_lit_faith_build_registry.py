"""Pytest gate for the gate-259 SCons/Make build target registry.

Covers:
  * generator imports cleanly + audit() returns expected shape
  * R1-R8 live checks on the current repo state (0 violations)
  * canonical (backend,kernel,variant) coverage of the documented
    Makefile KERNELS_* lists
  * CXXFLAGS spec sanity (required_tokens / opt_level present)
  * SRC_DIRS / BIN_DIRS map cleanly into the on-disk tree
  * ROI mechanism is documented per backend
  * artifact parity: json / md / csv emit deterministically and
    JSON matches the audit payload
"""
from __future__ import annotations

import csv as _csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = (
    REPO_ROOT / "scripts" / "experiments" / "ecg" / "lit_faith_build_registry.py"
)
JSON_OUT = REPO_ROOT / "wiki" / "data" / "lit_faith_build_registry.json"
MD_OUT = REPO_ROOT / "wiki" / "data" / "lit_faith_build_registry.md"
CSV_OUT = REPO_ROOT / "wiki" / "data" / "lit_faith_build_registry.csv"


def _load():
    spec = importlib.util.spec_from_file_location(
        "lit_faith_build_registry", GENERATOR
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load()


@pytest.fixture(scope="module")
def data(mod):
    return mod.audit()


# --- shape -----------------------------------------------------------------

def test_audit_status_active(data):
    assert data["status"] == "active"


def test_audit_shape_keys(data):
    for k in (
        "n_backends", "n_canonical_targets",
        "n_native_kernels", "n_cache_sim_kernels", "n_gem5_kernels",
        "n_sniper_targets", "n_makefile_harvested_kernels",
        "n_orphan_sources", "canonical_targets",
        "makefile_harvested_kernels", "cxxflags",
        "src_dirs", "bin_dirs", "roi_mechanisms",
        "rules", "violations",
    ):
        assert k in data, f"missing key {k}"


def test_n_backends_is_four(data):
    assert data["n_backends"] == 4


def test_n_canonical_targets_positive(data):
    assert data["n_canonical_targets"] > 0


# --- R1: source files exist ------------------------------------------------

def test_r1_no_missing_sources(data):
    bad = [v for v in data["violations"] if v.get("rule") == "R1"]
    assert bad == [], f"R1 violations: {bad}"


# --- R2: Makefile kernels canonical ----------------------------------------

def test_r2_no_makefile_kernel_drift(data):
    bad = [v for v in data["violations"] if v.get("rule") == "R2"]
    assert bad == [], f"R2 violations: {bad}"


def test_r2_native_makefile_matches_canonical(data, mod):
    declared = set(data["makefile_harvested_kernels"]["native"])
    canonical = set(mod._NATIVE_KERNELS)
    # Every declared kernel must be canonical (R2). Converter is
    # built via the same pattern rule but lives in the SUITE not
    # KERNELS, so it should not be in the declared set.
    assert declared <= canonical


def test_r2_cache_sim_makefile_matches_canonical(data, mod):
    declared = set(data["makefile_harvested_kernels"]["cache_sim"])
    canonical = set(mod._CACHE_SIM_KERNELS)
    assert declared == canonical


def test_r2_gem5_makefile_matches_canonical(data, mod):
    declared = set(data["makefile_harvested_kernels"]["gem5"])
    canonical = set(mod._GEM5_KERNELS)
    assert declared == canonical


# --- R3: required CXXFLAGS tokens present ----------------------------------

def test_r3_no_missing_cxxflag_tokens(data):
    bad = [v for v in data["violations"] if v.get("rule") == "R3"]
    assert bad == [], f"R3 violations: {bad}"


def test_r3_gem5_has_no_m5ops_default(data):
    # The default gem5 CXXFLAGS must include -DNO_M5OPS so the base
    # (no-ROI) gem5 binaries don't try to call m5 ops at runtime.
    assert "-DNO_M5OPS" in data["cxxflags"]["gem5"]["live_flags"]


def test_r3_sniper_has_sniper_include(data):
    assert "-I$(SNIPER_INCLUDE)" in data["cxxflags"]["sniper"]["live_flags"]


# --- R4: SRC / BIN dirs ----------------------------------------------------

def test_r4_no_dir_violations(data):
    bad = [v for v in data["violations"] if v.get("rule") == "R4"]
    assert bad == [], f"R4 violations: {bad}"


def test_r4_all_src_dirs_exist(data):
    for backend, rel in data["src_dirs"].items():
        p = REPO_ROOT / rel
        assert p.exists() and p.is_dir(), f"{backend}: {rel} missing"


# --- R5: kernel family classifier ------------------------------------------

def test_r5_every_kernel_has_family(data):
    bad = [v for v in data["violations"] if v.get("rule") == "R5"]
    assert bad == [], f"R5 violations: {bad}"


def test_r5_families_in_canonical_set(data):
    canonical_families = {
        "pagerank", "traversal", "shortest-path", "connected-component",
        "centrality", "triangle", "preprocess", "smoke",
    }
    for t in data["canonical_targets"]:
        assert t["family"] in canonical_families, (
            f"kernel {t['kernel']} has non-canonical family {t['family']!r}"
        )


# --- R6: opt level per backend ---------------------------------------------

def test_r6_native_o3(data):
    assert "-O3" in data["cxxflags"]["native"]["live_flags"]


def test_r6_cache_sim_o3(data):
    assert "-O3" in data["cxxflags"]["cache_sim"]["live_flags"]


def test_r6_gem5_o1(data):
    assert "-O1" in data["cxxflags"]["gem5"]["live_flags"]


def test_r6_sniper_o2(data):
    assert "-O2" in data["cxxflags"]["sniper"]["live_flags"]


def test_r6_no_violations(data):
    bad = [v for v in data["violations"] if v.get("rule") == "R6"]
    assert bad == [], f"R6 violations: {bad}"


# --- R7: no orphan sources --------------------------------------------------

def test_r7_no_orphan_sources(data):
    bad = [v for v in data["violations"] if v.get("rule") == "R7"]
    assert bad == [], f"R7 violations: {bad}"


def test_r7_n_orphan_sources_zero(data):
    assert data["n_orphan_sources"] == 0


# --- R8: ROI mechanism documented ------------------------------------------

def test_r8_no_violations(data):
    bad = [v for v in data["violations"] if v.get("rule") == "R8"]
    assert bad == [], f"R8 violations: {bad}"


def test_r8_roi_mechanism_per_backend(data):
    expected = {"native": "none", "cache_sim": "sim-callback",
                "gem5": "m5ops", "sniper": "sift"}
    assert data["roi_mechanisms"] == expected


# --- aggregate: zero violations -------------------------------------------

def test_zero_total_violations(data):
    assert data["violations"] == []


# --- canonical coverage ----------------------------------------------------

def test_native_target_count(data):
    n = sum(1 for t in data["canonical_targets"] if t["backend"] == "native")
    # 9 production kernels + converter
    assert n == 10


def test_gem5_has_three_variants_per_kernel(data, mod):
    by_kernel = {}
    for t in data["canonical_targets"]:
        if t["backend"] == "gem5":
            by_kernel.setdefault(t["kernel"], set()).add(t["variant"])
    expected_variants = {"base", "m5ops", "riscv_m5ops"}
    for k in mod._GEM5_KERNELS:
        assert by_kernel[k] == expected_variants, (
            f"gem5 kernel {k} has variants {by_kernel.get(k)}, expected {expected_variants}"
        )


def test_sniper_phase0_smokes_present(data, mod):
    snipers = {t["kernel"] for t in data["canonical_targets"] if t["backend"] == "sniper"}
    for k in mod._SNIPER_PHASE0:
        assert k in snipers, f"sniper Phase-0 smoke {k} missing"


# --- artifact parity --------------------------------------------------------

def test_json_artifact_exists():
    assert JSON_OUT.exists()


def test_md_artifact_exists():
    assert MD_OUT.exists()


def test_csv_artifact_exists():
    assert CSV_OUT.exists()


def test_json_matches_audit(data):
    on_disk = json.loads(JSON_OUT.read_text())
    # n_canonical_targets is deterministic
    assert on_disk["n_canonical_targets"] == data["n_canonical_targets"]
    assert on_disk["n_backends"] == data["n_backends"]
    assert on_disk["violations"] == data["violations"]


def test_md_ends_with_single_newline():
    txt = MD_OUT.read_text()
    assert txt.endswith("\n"), "md must end with newline"
    assert not txt.endswith("\n\n"), "md must end with exactly one newline"


def test_csv_round_trip():
    rows = list(_csv.reader(CSV_OUT.open()))
    assert rows[0] == ["kind", "name", "extra"]
    kinds = {r[0] for r in rows[1:]}
    assert "target" in kinds
    assert "cxxflags" in kinds
    assert "dirs" in kinds


def test_rules_has_eight_entries(data):
    assert set(data["rules"].keys()) == {f"R{i}" for i in range(1, 9)}
