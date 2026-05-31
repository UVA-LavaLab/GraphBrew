"""Pytest gate for ``lit_faith_gem5_overlay_hash_registry`` (gate 270)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "scripts" / "experiments" / "ecg" / "lit_faith_gem5_overlay_hash_registry.py"
WIKI_DATA = ROOT / "wiki" / "data"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "lit_faith_gem5_overlay_hash_registry", str(GEN)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lit_faith_gem5_overlay_hash_registry"] = mod
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module()


# --------------------------------------------------------------------
# Registry sanity
# --------------------------------------------------------------------


def test_registry_non_empty():
    assert len(MOD.OVERLAY_HASH_REGISTRY) >= 15


def test_registry_paths_relative():
    for rel in MOD.OVERLAY_HASH_REGISTRY:
        assert not rel.startswith("/")
        assert not rel.startswith(".")


def test_registry_hashes_are_sha256_hex():
    for rel, h in MOD.OVERLAY_HASH_REGISTRY.items():
        assert isinstance(h, str)
        assert len(h) == 64, f"{rel} hash not 64 chars"
        assert all(c in "0123456789abcdef" for c in h)


def test_tracked_extensions():
    assert ".cc" in MOD.OVERLAY_TRACKED_EXTS
    assert ".hh" in MOD.OVERLAY_TRACKED_EXTS
    assert ".py" in MOD.OVERLAY_TRACKED_EXTS
    assert ".isa" in MOD.OVERLAY_TRACKED_EXTS
    assert ".patch" in MOD.OVERLAY_TRACKED_EXTS


def test_size_bounds_sane():
    assert MOD.OVERLAY_MIN_SIZE >= 1
    assert MOD.OVERLAY_MAX_SIZE > MOD.OVERLAY_MIN_SIZE
    assert MOD.OVERLAY_MAX_SIZE <= 10_000_000


def test_required_markers_non_empty():
    assert len(MOD.OVERLAY_REQUIRED_MARKERS) >= 10


def test_simobject_classes_non_empty():
    assert len(MOD.SIMOBJECT_PY_CLASSES) >= 2
    for _, tokens in MOD.SIMOBJECT_PY_CLASSES.items():
        assert tokens


def test_expected_patches_present_in_constant():
    assert len(MOD.EXPECTED_PATCHES) == 2


# --------------------------------------------------------------------
# Live tree presence
# --------------------------------------------------------------------


def test_overlays_root_exists():
    assert MOD.OVERLAYS_ROOT.is_dir(), f"missing {MOD.OVERLAYS_ROOT}"


def test_every_registered_file_exists_on_disk():
    for rel in MOD.OVERLAY_HASH_REGISTRY:
        p = MOD.OVERLAYS_ROOT / rel
        assert p.is_file(), f"missing {p}"


def test_expected_patches_exist_on_disk():
    for rel in MOD.EXPECTED_PATCHES:
        p = MOD.OVERLAYS_ROOT / rel
        assert p.is_file(), f"missing {p}"


# --------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------


def test_audit_runs_and_returns_dict():
    doc = MOD.audit()
    assert isinstance(doc, dict)
    assert "violations" in doc
    assert "rules" in doc


def test_audit_clean_in_repo():
    doc = MOD.audit()
    assert doc["violations"] == [], f"unexpected violations: {doc['violations']}"


def test_audit_status_active():
    doc = MOD.audit()
    assert doc["status"] == "active"


def test_audit_rule_ids_complete():
    doc = MOD.audit()
    for rid in ["M1", "M2", "M3", "M4", "M5", "M6", "M7"]:
        assert rid in doc["rules"]


def test_audit_counts_meaningful():
    doc = MOD.audit()
    assert doc["counts"]["registered"] >= 15
    assert doc["counts"]["on_disk"] >= doc["counts"]["registered"]


# --------------------------------------------------------------------
# Rule-level injection
# --------------------------------------------------------------------


def test_m1_detects_hash_drift():
    orig = dict(MOD.OVERLAY_HASH_REGISTRY)
    try:
        rel = next(iter(MOD.OVERLAY_HASH_REGISTRY))
        MOD.OVERLAY_HASH_REGISTRY[rel] = "0" * 64
        doc = MOD.audit()
        m1 = [v for v in doc["violations"] if v["rule"] == "M1"]
        assert m1, "M1 did not detect hash drift"
    finally:
        MOD.OVERLAY_HASH_REGISTRY.clear()
        MOD.OVERLAY_HASH_REGISTRY.update(orig)


def test_m2_detects_size_out_of_bounds():
    orig_min = MOD.OVERLAY_MIN_SIZE
    try:
        MOD.OVERLAY_MIN_SIZE = 10**9
        doc = MOD.audit()
        m2 = [v for v in doc["violations"] if v["rule"] == "M2"]
        assert m2, "M2 did not detect oversize bounds"
    finally:
        MOD.OVERLAY_MIN_SIZE = orig_min


def test_m3_detects_missing_marker():
    orig = dict(MOD.OVERLAY_REQUIRED_MARKERS)
    try:
        MOD.OVERLAY_REQUIRED_MARKERS["mem/cache/replacement_policies/grasp_rp.cc"] = (
            "NeverGonnaAppearXYZ",
        )
        doc = MOD.audit()
        m3 = [v for v in doc["violations"] if v["rule"] == "M3"]
        assert m3, "M3 did not detect missing marker"
    finally:
        MOD.OVERLAY_REQUIRED_MARKERS.clear()
        MOD.OVERLAY_REQUIRED_MARKERS.update(orig)


def test_m4_detects_missing_from_registry():
    orig = dict(MOD.OVERLAY_HASH_REGISTRY)
    try:
        rel = next(iter(MOD.OVERLAY_HASH_REGISTRY))
        del MOD.OVERLAY_HASH_REGISTRY[rel]
        doc = MOD.audit()
        m4 = [v for v in doc["violations"] if v["rule"] == "M4"]
        assert m4, "M4 did not detect missing-from-registry"
    finally:
        MOD.OVERLAY_HASH_REGISTRY.clear()
        MOD.OVERLAY_HASH_REGISTRY.update(orig)


def test_m4_detects_registered_but_absent():
    orig = dict(MOD.OVERLAY_HASH_REGISTRY)
    try:
        MOD.OVERLAY_HASH_REGISTRY["does/not/exist.cc"] = "a" * 64
        doc = MOD.audit()
        m4 = [v for v in doc["violations"] if v["rule"] == "M4"]
        assert any("missing on disk" in v.get("issue", "") for v in m4), (
            "M4 did not detect registered-but-absent"
        )
    finally:
        MOD.OVERLAY_HASH_REGISTRY.clear()
        MOD.OVERLAY_HASH_REGISTRY.update(orig)


def test_m6_detects_missing_simobject_class():
    orig = dict(MOD.SIMOBJECT_PY_CLASSES)
    try:
        MOD.SIMOBJECT_PY_CLASSES["mem/cache/replacement_policies/GraphReplacementPolicies.py"] = (
            "NeverGonnaAppear",
        )
        doc = MOD.audit()
        m6 = [v for v in doc["violations"] if v["rule"] == "M6"]
        assert m6, "M6 did not detect missing SimObject class"
    finally:
        MOD.SIMOBJECT_PY_CLASSES.clear()
        MOD.SIMOBJECT_PY_CLASSES.update(orig)


# --------------------------------------------------------------------
# Artifact parity
# --------------------------------------------------------------------


def test_json_artifact_exists():
    assert (WIKI_DATA / "lit_faith_gem5_overlay_hash_registry.json").is_file()


def test_md_artifact_exists():
    assert (WIKI_DATA / "lit_faith_gem5_overlay_hash_registry.md").is_file()


def test_csv_artifact_exists():
    assert (WIKI_DATA / "lit_faith_gem5_overlay_hash_registry.csv").is_file()


def test_json_matches_live_audit():
    on_disk = json.loads(
        (WIKI_DATA / "lit_faith_gem5_overlay_hash_registry.json").read_text("utf-8")
    )
    live = MOD.audit()
    assert on_disk["counts"]["registered"] == live["counts"]["registered"]
    assert on_disk["violations"] == live["violations"]


def test_md_no_violations_marker():
    md = (WIKI_DATA / "lit_faith_gem5_overlay_hash_registry.md").read_text("utf-8")
    if not MOD.audit()["violations"]:
        assert "✅ No violations" in md


def test_md_exactly_one_final_newline():
    md = (WIKI_DATA / "lit_faith_gem5_overlay_hash_registry.md").read_text("utf-8")
    assert md.endswith("\n")
    assert not md.endswith("\n\n")


def test_csv_has_hash_rows():
    csv_text = (WIKI_DATA / "lit_faith_gem5_overlay_hash_registry.csv").read_text("utf-8")
    assert "hash," in csv_text
    for rel in list(MOD.OVERLAY_HASH_REGISTRY)[:3]:
        assert rel in csv_text


# --------------------------------------------------------------------
# Catalog wiring
# --------------------------------------------------------------------


def test_catalog_has_gate_270_entry():
    catalog = (WIKI_DATA / "artifact_catalog.json")
    assert catalog.is_file()
    text = catalog.read_text("utf-8")
    assert "lit_faith_gem5_overlay_hash_registry" in text, (
        "gate 270 entry not in artifact_catalog.json"
    )
