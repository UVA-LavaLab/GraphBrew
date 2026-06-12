"""Pytest gate for ``lit_faith_gem5_overlay_tracker`` (gate 267)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "scripts" / "experiments" / "ecg" / "lit_faith_gem5_overlay_tracker.py"
WIKI_DATA = ROOT / "wiki" / "data"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "lit_faith_gem5_overlay_tracker", str(GEN)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lit_faith_gem5_overlay_tracker"] = mod
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module()


# --------------------------------------------------------------------------
# Registry sanity
# --------------------------------------------------------------------------


def test_overlay_file_map_keys_non_empty():
    assert len(MOD.OVERLAY_FILE_MAP_KEYS) >= 14


def test_policies_non_empty():
    assert len(MOD.OVERLAY_POLICIES) >= 3
    assert "grasp" in MOD.OVERLAY_POLICIES
    assert "popt" in MOD.OVERLAY_POLICIES
    assert "ecg" in MOD.OVERLAY_POLICIES


def test_prefetchers_non_empty():
    assert len(MOD.OVERLAY_PREFETCHERS) >= 2
    assert "droplet" in MOD.OVERLAY_PREFETCHERS
    assert "ecg_pfx" in MOD.OVERLAY_PREFETCHERS


def test_patches_non_empty():
    assert len(MOD.OVERLAY_PATCHES) >= 2


def test_overlay_file_map_keys_unique():
    assert len(set(MOD.OVERLAY_FILE_MAP_KEYS)) == len(MOD.OVERLAY_FILE_MAP_KEYS)


def test_policies_unique():
    assert len(set(MOD.OVERLAY_POLICIES)) == len(MOD.OVERLAY_POLICIES)


def test_prefetchers_unique():
    assert len(set(MOD.OVERLAY_PREFETCHERS)) == len(MOD.OVERLAY_PREFETCHERS)


def test_patches_unique():
    assert len(set(MOD.OVERLAY_PATCHES)) == len(MOD.OVERLAY_PATCHES)


@pytest.mark.parametrize("f", MOD.OVERLAY_FILE_MAP_KEYS, ids=lambda x: x.replace("/", "_"))
def test_overlay_source_grammar(f):
    assert MOD.OVERLAY_PATH_RE.match(f), \
        f"OVERLAY_FILE_MAP source {f!r} fails grammar"


@pytest.mark.parametrize("p", MOD.OVERLAY_POLICIES)
def test_policy_grammar(p):
    assert MOD.POLICY_RE.match(p), f"policy {p!r} fails grammar"


@pytest.mark.parametrize("pf", MOD.OVERLAY_PREFETCHERS)
def test_prefetcher_grammar(pf):
    assert MOD.PREFETCHER_RE.match(pf), f"prefetcher {pf!r} fails grammar"


@pytest.mark.parametrize("pt", MOD.OVERLAY_PATCHES, ids=lambda x: x.replace("/", "_"))
def test_patch_grammar(pt):
    assert MOD.PATCH_PATH_RE.match(pt), f"patch {pt!r} fails grammar"


# --------------------------------------------------------------------------
# On-disk cross-checks
# --------------------------------------------------------------------------


def test_overlay_source_dir_exists():
    assert MOD.GEM5_OVERLAY_DIR.is_dir(), \
        f"gem5 overlay dir missing: {MOD.GEM5_OVERLAY_DIR}"


def test_setup_gem5_exists():
    assert MOD.SETUP_GEM5.is_file(), \
        f"setup_gem5.py missing: {MOD.SETUP_GEM5}"


@pytest.mark.parametrize("f", MOD.OVERLAY_FILE_MAP_KEYS, ids=lambda x: x.replace("/", "_"))
def test_overlay_source_on_disk(f):
    p = MOD.GEM5_OVERLAY_DIR / f
    assert p.is_file(), f"OVERLAY_FILE_MAP source not present on disk: {p}"


@pytest.mark.parametrize("pt", MOD.OVERLAY_PATCHES, ids=lambda x: x.replace("/", "_"))
def test_patch_on_disk(pt):
    p = MOD.GEM5_OVERLAY_DIR / pt
    assert p.is_file(), f"PATCH_FILES entry not present on disk: {p}"


@pytest.mark.parametrize("p", MOD.OVERLAY_POLICIES)
def test_policy_has_cc_and_hh(p):
    cc = f"mem/cache/replacement_policies/{p}_rp.cc"
    hh = f"mem/cache/replacement_policies/{p}_rp.hh"
    assert cc in MOD.OVERLAY_FILE_MAP_KEYS, f"policy {p} missing {cc}"
    assert hh in MOD.OVERLAY_FILE_MAP_KEYS, f"policy {p} missing {hh}"


@pytest.mark.parametrize("pf", MOD.OVERLAY_PREFETCHERS)
def test_prefetcher_has_cc_and_hh(pf):
    cc = f"mem/cache/prefetch/{pf}.cc"
    hh = f"mem/cache/prefetch/{pf}.hh"
    assert cc in MOD.OVERLAY_FILE_MAP_KEYS, f"prefetcher {pf} missing {cc}"
    assert hh in MOD.OVERLAY_FILE_MAP_KEYS, f"prefetcher {pf} missing {hh}"


def test_live_setup_gem5_matches_canonical_map():
    """The OVERLAY_FILE_MAP keys in setup_gem5.py must match the
    canonical registry exactly."""
    spec = importlib.util.spec_from_file_location("setup_gem5_test", MOD.SETUP_GEM5)
    sg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sg)
    live = sorted(sg.OVERLAY_FILE_MAP.keys())
    canon = sorted(MOD.OVERLAY_FILE_MAP_KEYS)
    assert live == canon, (
        f"live setup_gem5.OVERLAY_FILE_MAP drifted from canonical:\n"
        f"  extra in live:  {sorted(set(live)-set(canon))}\n"
        f"  missing in live:{sorted(set(canon)-set(live))}"
    )


def test_live_setup_gem5_matches_canonical_patches():
    spec = importlib.util.spec_from_file_location("setup_gem5_test_p", MOD.SETUP_GEM5)
    sg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sg)
    live = sorted(sg.PATCH_FILES)
    canon = sorted(MOD.OVERLAY_PATCHES)
    assert live == canon, (
        f"live setup_gem5.PATCH_FILES drifted from canonical:\n"
        f"  extra:  {sorted(set(live)-set(canon))}\n"
        f"  missing:{sorted(set(canon)-set(live))}"
    )


def test_live_setup_gem5_file_map_is_identity():
    """OVERLAY_FILE_MAP convention is src==dst.  Deviations require
    explicit audit."""
    spec = importlib.util.spec_from_file_location("setup_gem5_test_i", MOD.SETUP_GEM5)
    sg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sg)
    bad = [(s, d) for s, d in sg.OVERLAY_FILE_MAP.items() if s != d]
    assert not bad, f"non-identity src→dst mappings: {bad}"


def test_no_orphan_overlay_files():
    on_disk = set()
    for p in MOD.GEM5_OVERLAY_DIR.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(MOD.GEM5_OVERLAY_DIR)).replace("\\", "/")
        if rel in MOD.OVERLAY_EXTRA_ALLOW:
            continue
        if Path(rel).suffix not in MOD.TRACKED_EXT and not rel.endswith(".patch"):
            continue
        on_disk.add(rel)
    canonical = set(MOD.OVERLAY_FILE_MAP_KEYS) | set(MOD.OVERLAY_PATCHES) | set(MOD.UNIFIED_DIFF_PATCHES)
    extra = on_disk - canonical
    assert not extra, f"orphan overlay files: {sorted(extra)}"


# --------------------------------------------------------------------------
# Setup-source literal cross-checks
# --------------------------------------------------------------------------


def test_setup_gem5_mentions_every_policy_literal():
    text = MOD.SETUP_GEM5.read_text(encoding="utf-8")
    for pol in MOD.OVERLAY_POLICIES:
        # via _rp.cc / _rp.hh patterns
        assert f"{pol}_rp" in text, f"policy {pol!r} not referenced in setup_gem5.py"


def test_setup_gem5_mentions_every_prefetcher_literal():
    text = MOD.SETUP_GEM5.read_text(encoding="utf-8")
    for pf in MOD.OVERLAY_PREFETCHERS:
        assert pf in text, f"prefetcher {pf!r} not referenced in setup_gem5.py"


# --------------------------------------------------------------------------
# End-to-end audit
# --------------------------------------------------------------------------


def test_audit_runs_clean():
    data = MOD.audit()
    assert data["status"] == "active"
    assert not data["violations"], (
        "audit reports violations: "
        + "; ".join(f"{v['rule']} @ {v['where']}: {v['msg']}" for v in data["violations"])
    )


def test_artifact_on_disk():
    art = WIKI_DATA / "lit_faith_gem5_overlay_tracker.json"
    assert art.is_file()
    data = json.loads(art.read_text(encoding="utf-8"))
    assert not data["violations"]


def test_artifact_md_exists():
    md = WIKI_DATA / "lit_faith_gem5_overlay_tracker.md"
    assert md.is_file()
    text = md.read_text(encoding="utf-8")
    assert "Gate 267" in text


def test_artifact_csv_exists():
    csv_p = WIKI_DATA / "lit_faith_gem5_overlay_tracker.csv"
    assert csv_p.is_file()
    lines = csv_p.read_text(encoding="utf-8").strip().splitlines()
    # header + 14 sources + 3 policies + 2 prefetchers + 2 patches = 22
    assert len(lines) == 22, f"unexpected csv row count: {len(lines)}"


def test_catalog_has_gate_267_entry():
    """Gate 267 must be registered in artifact_catalog.py."""
    cat = ROOT / "scripts" / "experiments" / "ecg" / "artifact_catalog.py"
    text = cat.read_text(encoding="utf-8")
    assert "lit_faith_gem5_overlay_tracker" in text, \
        "gate 267 missing from artifact_catalog.py"
