"""Pytest gate for ``lit_faith_sideband_grammar`` (gate 265).

Cross-checks the audit output AND directly re-reads the source files
to verify the registry's claims.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "scripts" / "experiments" / "ecg" / "lit_faith_sideband_grammar.py"
WIKI_DATA = ROOT / "wiki" / "data"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "lit_faith_sideband_grammar", str(GEN)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lit_faith_sideband_grammar"] = mod
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module()


# --------------------------------------------------------------------------
# Registry sanity
# --------------------------------------------------------------------------


def test_registry_non_empty():
    assert len(MOD.SIDEBAND_REGISTRY) >= 8


def test_registry_has_all_tool_role_pairs():
    pairs = {(e["tool"], e["role"]) for e in MOD.SIDEBAND_REGISTRY}
    expected = {(t, r) for t in MOD.CANONICAL_TOOLS for r in MOD.CANONICAL_ROLES}
    missing = expected - pairs
    assert not missing, f"missing tool/role pairs: {missing}"


def test_registry_filename_unique():
    fns = [e["filename"] for e in MOD.SIDEBAND_REGISTRY]
    assert len(set(fns)) == len(fns), "duplicate filename in registry"


def test_registry_env_var_unique():
    envs = [e["env_var"] for e in MOD.SIDEBAND_REGISTRY]
    assert len(set(envs)) == len(envs), "duplicate env_var in registry"


def test_registry_default_path_unique():
    paths = [e["default_path"] for e in MOD.SIDEBAND_REGISTRY]
    assert len(set(paths)) == len(paths), "duplicate default_path in registry"


@pytest.mark.parametrize("entry", MOD.SIDEBAND_REGISTRY,
                         ids=lambda e: f"{e['tool']}-{e['role']}")
def test_registry_filename_grammar(entry):
    assert re.match(r"^(gem5|sniper)_[a-z0-9_]+\.(json|bin)$", entry["filename"]), \
        f"filename {entry['filename']!r} does not match canonical grammar"


@pytest.mark.parametrize("entry", MOD.SIDEBAND_REGISTRY,
                         ids=lambda e: f"{e['tool']}-{e['role']}")
def test_registry_tool_prefix(entry):
    assert entry["filename"].startswith(f"{entry['tool']}_"), \
        f"filename {entry['filename']!r} does not have tool-prefix {entry['tool']}_"


@pytest.mark.parametrize("entry", MOD.SIDEBAND_REGISTRY,
                         ids=lambda e: f"{e['tool']}-{e['role']}")
def test_registry_role_extension(entry):
    if entry["role"] == "context":
        assert entry["filename"].endswith(".json"), \
            f"role=context but filename does not end .json"
    else:
        assert entry["filename"].endswith(".bin"), \
            f"role={entry['role']} but filename does not end .bin"


@pytest.mark.parametrize("entry", MOD.SIDEBAND_REGISTRY,
                         ids=lambda e: f"{e['tool']}-{e['role']}")
def test_registry_env_var_bijection(entry):
    stem = entry["filename"].rsplit(".", 1)[0]
    expected = stem.upper()
    assert entry["env_var"] == expected, \
        f"env_var {entry['env_var']!r} != expected {expected!r}"


@pytest.mark.parametrize("entry", MOD.SIDEBAND_REGISTRY,
                         ids=lambda e: f"{e['tool']}-{e['role']}")
def test_registry_default_path_bijection(entry):
    expected = f"/tmp/{entry['filename']}"
    assert entry["default_path"] == expected, \
        f"default_path {entry['default_path']!r} != expected {expected!r}"


# --------------------------------------------------------------------------
# Source-file cross-checks (S4 / S5 / S6 / S7 sanity)
# --------------------------------------------------------------------------


def test_gem5_harness_exists():
    assert MOD.GEM5_HARNESS.is_file(), \
        f"gem5_harness.h not found at {MOD.GEM5_HARNESS}"


def test_sniper_cache_sets_exist():
    for p in MOD.SNIPER_CACHE_SETS:
        assert p.is_file(), f"Sniper cache-set source missing: {p}"


def test_sniper_prefetchers_exist():
    for p in MOD.SNIPER_PREFETCHERS:
        assert p.is_file(), f"Sniper prefetcher source missing: {p}"


def test_roi_matrix_exists():
    assert MOD.ROI_MATRIX.is_file(), f"roi_matrix.py not found at {MOD.ROI_MATRIX}"


def test_gem5_harness_uses_canonical_calls():
    text = MOD.GEM5_HARNESS.read_text(encoding="utf-8")
    calls = MOD.GEM5_CALL_RE.findall(text)
    # Filter to gem5-prefixed env vars only (other calls may exist for
    # non-sideband config like timing thresholds).
    sideband_calls = [(e, p) for (e, p) in calls if e.startswith("GEM5_")]
    assert sideband_calls, "no gem5_env_or_default sideband calls found"
    by_env = {e["env_var"]: e for e in MOD.SIDEBAND_REGISTRY if e["tool"] == "gem5"}
    seen = set()
    for env, path in sideband_calls:
        if env in MOD.ENV_VAR_NON_FILE_ALLOW:
            continue
        if env not in by_env:
            continue
        seen.add(env)
        assert path == by_env[env]["default_path"], \
            f"gem5 harness {env!r} default_path mismatch: {path!r}"
    # Every gem5 registry entry should be referenced exactly once.
    expected = set(by_env.keys())
    missing = expected - seen
    assert not missing, f"gem5_harness.h does not reference {missing}"


@pytest.mark.parametrize("srcfile", MOD.SNIPER_CACHE_SETS + MOD.SNIPER_PREFETCHERS,
                         ids=lambda p: p.name)
def test_sniper_source_uses_canonical_calls(srcfile):
    text = srcfile.read_text(encoding="utf-8")
    calls = MOD.SNIPER_CALL_RE.findall(text)
    by_env = {e["env_var"]: e for e in MOD.SIDEBAND_REGISTRY if e["tool"] == "sniper"}
    for env, path in calls:
        if env in MOD.ENV_VAR_NON_FILE_ALLOW:
            continue
        if env not in by_env:
            continue
        assert path == by_env[env]["default_path"], \
            f"{srcfile.name} env_var {env!r} default_path mismatch: {path!r}"


def test_roi_matrix_paths_use_canonical_filenames():
    text = MOD.ROI_MATRIX.read_text(encoding="utf-8")
    by_filename = {e["filename"]: e for e in MOD.SIDEBAND_REGISTRY}
    for tool in MOD.CANONICAL_TOOLS:
        func_name = f"{tool}_sideband_paths"
        start_m = re.search(
            rf"^def {re.escape(func_name)}\b", text, re.MULTILINE
        )
        assert start_m, f"function {func_name} not found"
        tail = text[start_m.start():]
        end_m = re.search(r"^def \w+\b", tail[1:], re.MULTILINE)
        body = tail if not end_m else tail[: end_m.start() + 1]
        # All registered filenames for this tool must appear in the body.
        for e in MOD.SIDEBAND_REGISTRY:
            if e["tool"] != tool:
                continue
            assert f'"{e["filename"]}"' in body, \
                f"{func_name} missing filename literal {e['filename']!r}"


def test_roi_matrix_uses_canonical_sideband_subdir():
    text = MOD.ROI_MATRIX.read_text(encoding="utf-8")
    assert f'"{MOD.SIDEBAND_SUBDIR}"' in text, \
        f"roi_matrix.py does not reference {MOD.SIDEBAND_SUBDIR!r}"


# --------------------------------------------------------------------------
# End-to-end audit run
# --------------------------------------------------------------------------


def test_audit_runs_clean():
    data = MOD.audit()
    assert data["status"] == "active"
    assert data["registry_n"] == len(MOD.SIDEBAND_REGISTRY)
    assert not data["violations"], (
        "audit reports violations: "
        + "; ".join(f"{v['rule']} @ {v['where']}: {v['msg']}" for v in data["violations"])
    )


def test_artifact_on_disk():
    art = WIKI_DATA / "lit_faith_sideband_grammar.json"
    assert art.is_file(), f"artifact missing: {art}"
    data = json.loads(art.read_text(encoding="utf-8"))
    assert data["status"] == "active"
    assert not data["violations"], \
        f"on-disk artifact reports violations: {data['violations']}"


def test_artifact_md_exists():
    md = WIKI_DATA / "lit_faith_sideband_grammar.md"
    assert md.is_file()
    text = md.read_text(encoding="utf-8")
    assert "Gate 265" in text
    assert "registry entries:" in text


def test_artifact_csv_exists():
    csv_p = WIKI_DATA / "lit_faith_sideband_grammar.csv"
    assert csv_p.is_file()
    lines = csv_p.read_text(encoding="utf-8").strip().splitlines()
    # header + 8 entries
    assert len(lines) == 9, f"unexpected csv row count: {len(lines)}"


def test_self_artifact_in_catalog():
    """Gate 265's own stem must be declared in artifact_catalog (not in
    IMPLICIT_PAPER_PIPELINE_STEMS) — otherwise gate 264 would treat it
    as implicit and miss the wiring."""
    from importlib.util import spec_from_file_location, module_from_spec
    cat_path = ROOT / "scripts" / "experiments" / "ecg" / "artifact_catalog.py"
    spec = spec_from_file_location("artifact_catalog", str(cat_path))
    cat = module_from_spec(spec)
    sys.modules["artifact_catalog"] = cat
    spec.loader.exec_module(cat)
    stems = {Path(e["artifact"]).stem for e in cat.CATALOG}
    assert "lit_faith_sideband_grammar" in stems, \
        "gate 265 artifact stem not in artifact_catalog"
