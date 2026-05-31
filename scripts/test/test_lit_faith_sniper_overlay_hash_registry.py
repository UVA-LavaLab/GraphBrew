"""Pytest suite for gate 271 — Sniper overlay-file hash registry."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from scripts.experiments.ecg import lit_faith_sniper_overlay_hash_registry as M

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------
# Registry sanity
# --------------------------------------------------------------------


def test_registry_non_empty():
    assert len(M.SNIPER_OVERLAY_HASH_REGISTRY) >= 10


def test_registry_keys_relative_and_normalized():
    for rel in M.SNIPER_OVERLAY_HASH_REGISTRY:
        assert not rel.startswith("/"), rel
        assert "\\" not in rel, rel
        assert ".." not in rel.split("/"), rel


def test_registry_sha256_format():
    rx = re.compile(r"^[0-9a-f]{64}$")
    for rel, h in M.SNIPER_OVERLAY_HASH_REGISTRY.items():
        assert rx.match(h), f"{rel}: {h}"


def test_tracked_exts_subset():
    assert ".cc" in M.SNIPER_OVERLAY_TRACKED_EXTS
    assert ".h" in M.SNIPER_OVERLAY_TRACKED_EXTS


def test_ignored_exts_disjoint_from_tracked():
    assert not (set(M.SNIPER_OVERLAY_TRACKED_EXTS)
                & set(M.SNIPER_OVERLAY_IGNORED_EXTS))


def test_size_bounds_sane():
    assert M.SNIPER_OVERLAY_MIN_SIZE < M.SNIPER_OVERLAY_MAX_SIZE
    assert M.SNIPER_OVERLAY_MIN_SIZE >= 1
    assert M.SNIPER_OVERLAY_MAX_SIZE <= 10_000_000


def test_required_markers_subset_of_registry():
    for rel in M.SNIPER_OVERLAY_REQUIRED_MARKERS:
        assert rel in M.SNIPER_OVERLAY_HASH_REGISTRY, rel


def test_required_markers_non_empty():
    for rel, markers in M.SNIPER_OVERLAY_REQUIRED_MARKERS.items():
        assert markers, rel
        for m in markers:
            assert m and isinstance(m, str), rel


def test_class_declarations_subset_of_registry():
    for rel in M.SNIPER_OVERLAY_CLASS_DECLARATIONS:
        assert rel in M.SNIPER_OVERLAY_HASH_REGISTRY, rel
        assert rel.endswith(".h"), rel


def test_class_declarations_token_format():
    rx = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    for cls in M.SNIPER_OVERLAY_CLASS_DECLARATIONS.values():
        assert rx.match(cls), cls


# --------------------------------------------------------------------
# Live presence
# --------------------------------------------------------------------


def test_overlays_root_exists():
    assert M.OVERLAYS_ROOT.is_dir()


def test_every_registered_file_exists():
    missing = [r for r in M.SNIPER_OVERLAY_HASH_REGISTRY
               if not (M.OVERLAYS_ROOT / r).is_file()]
    assert not missing, missing


def test_every_registered_file_hash_matches():
    bad = []
    for rel, want in M.SNIPER_OVERLAY_HASH_REGISTRY.items():
        p = M.OVERLAYS_ROOT / rel
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != want:
            bad.append((rel, want, got))
    assert not bad, bad


# --------------------------------------------------------------------
# audit() shape
# --------------------------------------------------------------------


def test_audit_zero_violations():
    assert M.audit()["violations"] == []


def test_audit_status_active():
    assert M.audit()["status"] == "active"


def test_audit_schema_versioned():
    assert M.audit()["schema"] == "lit-faith-sniper-overlay-hash-registry/1"


def test_audit_rules_all_six():
    rules = M.audit()["rules"]
    assert set(rules) == {"N1", "N2", "N3", "N4", "N5", "N6"}


def test_audit_counts_consistent():
    c = M.audit()["counts"]
    assert c["registered"] == len(M.SNIPER_OVERLAY_HASH_REGISTRY)
    assert c["on_disk"] >= c["registered"]


# --------------------------------------------------------------------
# Injection tests
# --------------------------------------------------------------------


def test_n1_detects_hash_mutation(monkeypatch):
    bad = dict(M.SNIPER_OVERLAY_HASH_REGISTRY)
    first = next(iter(bad))
    bad[first] = "0" * 64
    monkeypatch.setattr(M, "SNIPER_OVERLAY_HASH_REGISTRY", bad)
    viols = M.audit()["violations"]
    assert any(v["rule"] == "N1" and v["path"] == first for v in viols)


def test_n2_detects_oversize_bound(monkeypatch):
    monkeypatch.setattr(M, "SNIPER_OVERLAY_MAX_SIZE", 10)
    viols = M.audit()["violations"]
    assert any(v["rule"] == "N2" for v in viols)


def test_n3_detects_missing_marker(monkeypatch):
    bad = dict(M.SNIPER_OVERLAY_REQUIRED_MARKERS)
    first = next(iter(bad))
    bad[first] = ("DEFINITELY_NOT_IN_FILE_zZ9",)
    monkeypatch.setattr(M, "SNIPER_OVERLAY_REQUIRED_MARKERS", bad)
    viols = M.audit()["violations"]
    assert any(v["rule"] == "N3" and v["path"] == first for v in viols)


def test_n4_detects_missing_on_disk(monkeypatch):
    bad = dict(M.SNIPER_OVERLAY_HASH_REGISTRY)
    bad["common/core/memory_subsystem/cache/ghost_file.cc"] = "f" * 64
    monkeypatch.setattr(M, "SNIPER_OVERLAY_HASH_REGISTRY", bad)
    viols = M.audit()["violations"]
    assert any(v["rule"] == "N4"
               and "ghost_file" in v["path"]
               and "missing on disk" in v["issue"] for v in viols)


def test_n4_detects_extra_on_disk(monkeypatch):
    bad = dict(M.SNIPER_OVERLAY_HASH_REGISTRY)
    first = next(iter(bad))
    del bad[first]
    monkeypatch.setattr(M, "SNIPER_OVERLAY_HASH_REGISTRY", bad)
    viols = M.audit()["violations"]
    assert any(v["rule"] == "N4" and v["path"] == first
               and "not in registry" in v["issue"] for v in viols)


def test_n6_detects_missing_class(monkeypatch):
    bad = dict(M.SNIPER_OVERLAY_CLASS_DECLARATIONS)
    first = next(iter(bad))
    bad[first] = "NeverDeclaredClassZz9"
    monkeypatch.setattr(M, "SNIPER_OVERLAY_CLASS_DECLARATIONS", bad)
    viols = M.audit()["violations"]
    assert any(v["rule"] == "N6" and v["path"] == first for v in viols)


# --------------------------------------------------------------------
# Artifact parity
# --------------------------------------------------------------------


def test_json_artifact_exists():
    p = REPO_ROOT / "wiki/data/lit_faith_sniper_overlay_hash_registry.json"
    assert p.is_file()


def test_json_artifact_parses():
    p = REPO_ROOT / "wiki/data/lit_faith_sniper_overlay_hash_registry.json"
    doc = json.loads(p.read_text())
    assert doc["schema"] == "lit-faith-sniper-overlay-hash-registry/1"
    assert doc["status"] == "active"


def test_md_artifact_exists():
    p = REPO_ROOT / "wiki/data/lit_faith_sniper_overlay_hash_registry.md"
    assert p.is_file()
    assert "Sniper overlay-file hash registry" in p.read_text()


def test_csv_artifact_exists():
    p = REPO_ROOT / "wiki/data/lit_faith_sniper_overlay_hash_registry.csv"
    assert p.is_file()
    assert "kind,path,sha256_or_meta" in p.read_text().splitlines()[0]


def test_md_exactly_one_final_newline():
    p = REPO_ROOT / "wiki/data/lit_faith_sniper_overlay_hash_registry.md"
    text = p.read_text()
    assert text.endswith("\n") and not text.endswith("\n\n")


# --------------------------------------------------------------------
# Catalog wiring
# --------------------------------------------------------------------


def test_catalog_lists_sniper_overlay_hash_registry():
    from scripts.experiments.ecg.artifact_catalog import CATALOG

    paths = {a["artifact"] for a in CATALOG}
    assert "wiki/data/lit_faith_sniper_overlay_hash_registry.json" in paths


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


def test_main_update_returns_zero(capsys):
    rc = M.main(["--update"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "SNIPER_OVERLAY_HASH_REGISTRY" in out
