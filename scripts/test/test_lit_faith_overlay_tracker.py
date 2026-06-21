"""Pytest gate for ``lit_faith_overlay_tracker`` (gate 266)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "scripts" / "experiments" / "ecg" / "lit_faith_overlay_tracker.py"
WIKI_DATA = ROOT / "wiki" / "data"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "lit_faith_overlay_tracker", str(GEN)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lit_faith_overlay_tracker"] = mod
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module()


# --------------------------------------------------------------------------
# Registry sanity
# --------------------------------------------------------------------------


def test_copied_files_non_empty():
    assert len(MOD.OVERLAY_COPIED_FILES) >= 12


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
    assert len(MOD.OVERLAY_PATCHES) >= 5


def test_copied_files_unique():
    assert len(set(MOD.OVERLAY_COPIED_FILES)) == len(MOD.OVERLAY_COPIED_FILES)


def test_policies_unique():
    assert len(set(MOD.OVERLAY_POLICIES)) == len(MOD.OVERLAY_POLICIES)


def test_prefetchers_unique():
    assert len(set(MOD.OVERLAY_PREFETCHERS)) == len(MOD.OVERLAY_PREFETCHERS)


def test_patches_unique():
    assert len(set(MOD.OVERLAY_PATCHES)) == len(MOD.OVERLAY_PATCHES)


@pytest.mark.parametrize("f", MOD.OVERLAY_COPIED_FILES, ids=lambda x: x.replace("/", "_"))
def test_copied_file_grammar(f):
    assert MOD.COPIED_FILE_RE.match(f), \
        f"copied_files entry {f!r} fails grammar"


@pytest.mark.parametrize("p", MOD.OVERLAY_POLICIES)
def test_policy_grammar(p):
    assert MOD.POLICY_RE.match(p), f"policy {p!r} fails grammar"


@pytest.mark.parametrize("pf", MOD.OVERLAY_PREFETCHERS)
def test_prefetcher_grammar(pf):
    assert MOD.PREFETCHER_RE.match(pf), f"prefetcher {pf!r} fails grammar"


@pytest.mark.parametrize("pt", MOD.OVERLAY_PATCHES)
def test_patch_grammar(pt):
    assert MOD.PATCH_RE.match(pt), f"patch {pt!r} fails grammar"


# --------------------------------------------------------------------------
# On-disk cross-checks
# --------------------------------------------------------------------------


def test_overlay_source_dir_exists():
    assert MOD.SNIPER_OVERLAY_DIR.is_dir(), \
        f"overlay source dir missing: {MOD.SNIPER_OVERLAY_DIR}"


def test_overlay_status_file_exists():
    assert MOD.OVERLAY_STATUS_FILE.is_file(), \
        f"overlay status file missing: {MOD.OVERLAY_STATUS_FILE}"


def test_setup_sniper_exists():
    assert MOD.SETUP_SNIPER.is_file(), \
        f"setup_sniper.py missing: {MOD.SETUP_SNIPER}"


@pytest.mark.parametrize("f", MOD.OVERLAY_COPIED_FILES, ids=lambda x: x.replace("/", "_"))
def test_copied_file_on_disk(f):
    p = MOD.SNIPER_OVERLAY_DIR / f
    assert p.is_file(), f"copied_files entry not present on disk: {p}"


@pytest.mark.parametrize("p", MOD.OVERLAY_POLICIES)
def test_policy_has_cc_and_h(p):
    cc = f"common/core/memory_subsystem/cache/cache_set_{p}.cc"
    hh = f"common/core/memory_subsystem/cache/cache_set_{p}.h"
    assert cc in MOD.OVERLAY_COPIED_FILES, f"policy {p} missing {cc}"
    assert hh in MOD.OVERLAY_COPIED_FILES, f"policy {p} missing {hh}"


@pytest.mark.parametrize("pf", MOD.OVERLAY_PREFETCHERS)
def test_prefetcher_has_cc_and_h(pf):
    cc = f"common/core/memory_subsystem/parametric_dram_directory_msi/{pf}_prefetcher.cc"
    hh = f"common/core/memory_subsystem/parametric_dram_directory_msi/{pf}_prefetcher.h"
    assert cc in MOD.OVERLAY_COPIED_FILES, f"prefetcher {pf} missing {cc}"
    assert hh in MOD.OVERLAY_COPIED_FILES, f"prefetcher {pf} missing {hh}"


def test_setup_sniper_has_canonical_patch_fns():
    """Every patch_*_overlay fn in setup_sniper.py is either a registered
    patch token OR known to be a private/internal helper."""
    text = MOD.SETUP_SNIPER.read_text(encoding="utf-8")
    fn_names = set(MOD.PATCH_FN_RE.findall(text))
    # We expect at least 5 patch_*_overlay functions (grasp, popt, ecg,
    # droplet, graphbrew_simuser, ecg_pfx_prefetcher).
    assert len(fn_names) >= 5, f"too few patch_*_overlay fns: {fn_names}"


def test_on_disk_overlay_status_matches_canonical():
    data = json.loads(MOD.OVERLAY_STATUS_FILE.read_text(encoding="utf-8"))
    assert sorted(data.get("copied_files", [])) == sorted(MOD.OVERLAY_COPIED_FILES)
    assert sorted(data.get("policies", [])) == sorted(MOD.OVERLAY_POLICIES)
    assert sorted(data.get("prefetchers", [])) == sorted(MOD.OVERLAY_PREFETCHERS)
    assert sorted(data.get("patches", [])) == sorted(MOD.OVERLAY_PATCHES)


def test_setup_sniper_write_overlay_status_uses_canonical():
    """The write_overlay_status function in setup_sniper.py must
    declare the same canonical policies/prefetchers/patches lists."""
    text = MOD.SETUP_SNIPER.read_text(encoding="utf-8")
    for pol in MOD.OVERLAY_POLICIES:
        assert f'"{pol}"' in text, f"policy {pol!r} not literal in setup_sniper.py"
    for pf in MOD.OVERLAY_PREFETCHERS:
        assert f'"{pf}"' in text, f"prefetcher {pf!r} not literal in setup_sniper.py"
    for pt in MOD.OVERLAY_PATCHES:
        assert f'"{pt}"' in text, f"patch {pt!r} not literal in setup_sniper.py"


def test_no_orphan_overlay_files():
    on_disk = {
        str(p.relative_to(MOD.SNIPER_OVERLAY_DIR)).replace("\\", "/")
        for p in MOD.SNIPER_OVERLAY_DIR.rglob("*")
        if p.is_file() and p.name not in MOD.OVERLAY_README_ALLOW
    }
    canonical = set(MOD.OVERLAY_COPIED_FILES)
    extra = on_disk - canonical
    assert not extra, f"orphan overlay files: {sorted(extra)}"


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
    art = WIKI_DATA / "lit_faith_overlay_tracker.json"
    assert art.is_file()
    data = json.loads(art.read_text(encoding="utf-8"))
    assert not data["violations"]


def test_artifact_md_exists():
    md = WIKI_DATA / "lit_faith_overlay_tracker.md"
    assert md.is_file()
    text = md.read_text(encoding="utf-8")
    assert "Gate 266" in text


def test_artifact_csv_exists():
    csv_p = WIKI_DATA / "lit_faith_overlay_tracker.csv"
    assert csv_p.is_file()
    lines = csv_p.read_text(encoding="utf-8").strip().splitlines()
    # header + 13 copied + 3 policies + 2 prefetchers + 5 patches = 24
    assert len(lines) == 24, f"unexpected csv row count: {len(lines)}"
