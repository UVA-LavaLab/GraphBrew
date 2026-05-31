"""Pytest gate for ``lit_faith_setup_script_registry`` (gate 268)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "scripts" / "experiments" / "ecg" / "lit_faith_setup_script_registry.py"
WIKI_DATA = ROOT / "wiki" / "data"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "lit_faith_setup_script_registry", str(GEN)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lit_faith_setup_script_registry"] = mod
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module()


# --------------------------------------------------------------------------
# Registry sanity
# --------------------------------------------------------------------------


def test_repo_urls_non_empty():
    assert len(MOD.SETUP_REPO_URLS) >= 2
    assert "GEM5_REPO_URL" in MOD.SETUP_REPO_URLS
    assert "SNIPER_REPO_URL" in MOD.SETUP_REPO_URLS


def test_repo_urls_canonical_values():
    assert MOD.SETUP_REPO_URLS["GEM5_REPO_URL"] == "https://github.com/gem5/gem5.git"
    assert MOD.SETUP_REPO_URLS["SNIPER_REPO_URL"] == "https://github.com/snipersim/snipersim.git"


def test_gem5_dir_constants_non_empty():
    assert len(MOD.SETUP_GEM5_DIR_CONSTANTS) >= 6
    for c in ["SCRIPT_DIR", "PROJECT_ROOT", "GEM5_SIM_DIR", "GEM5_DIR", "OVERLAYS_DIR"]:
        assert c in MOD.SETUP_GEM5_DIR_CONSTANTS, f"missing {c}"


def test_sniper_dir_constants_non_empty():
    assert len(MOD.SETUP_SNIPER_DIR_CONSTANTS) >= 6
    for c in ["SCRIPT_DIR", "PROJECT_ROOT", "SNIPER_SIM_DIR", "SNIPER_DIR",
              "SNIPER_OVERLAY_DIR", "SNIPER_CONFIG_DIR"]:
        assert c in MOD.SETUP_SNIPER_DIR_CONSTANTS, f"missing {c}"


def test_gem5_functions_non_empty():
    assert len(MOD.SETUP_GEM5_FUNCTIONS) >= 14
    for fn in ["main", "apply_overlays", "apply_patches", "build_gem5", "clone_gem5"]:
        assert fn in MOD.SETUP_GEM5_FUNCTIONS, f"missing {fn}"


def test_sniper_functions_non_empty():
    assert len(MOD.SETUP_SNIPER_FUNCTIONS) >= 27
    for fn in ["main", "apply_overlays", "build_sniper", "clone_or_update",
               "copy_overlay_sources", "write_overlay_status"]:
        assert fn in MOD.SETUP_SNIPER_FUNCTIONS, f"missing {fn}"


def test_gem5_functions_unique():
    assert len(set(MOD.SETUP_GEM5_FUNCTIONS)) == len(MOD.SETUP_GEM5_FUNCTIONS)


def test_sniper_functions_unique():
    assert len(set(MOD.SETUP_SNIPER_FUNCTIONS)) == len(MOD.SETUP_SNIPER_FUNCTIONS)


def test_gem5_dir_constants_unique():
    assert len(set(MOD.SETUP_GEM5_DIR_CONSTANTS)) == len(MOD.SETUP_GEM5_DIR_CONSTANTS)


def test_sniper_dir_constants_unique():
    assert len(set(MOD.SETUP_SNIPER_DIR_CONSTANTS)) == len(MOD.SETUP_SNIPER_DIR_CONSTANTS)


def test_extra_allow_lists_are_sets():
    assert isinstance(MOD.SETUP_GEM5_EXTRA_ALLOW, set)
    assert isinstance(MOD.SETUP_SNIPER_EXTRA_ALLOW, set)


# --------------------------------------------------------------------------
# Grammar
# --------------------------------------------------------------------------


def test_repo_urls_are_https():
    for name, url in MOD.SETUP_REPO_URLS.items():
        assert url.startswith("https://"), f"{name} not https: {url}"
        assert url.endswith(".git"), f"{name} not .git: {url}"


def test_dir_constants_uppercase():
    for c in MOD.SETUP_GEM5_DIR_CONSTANTS + MOD.SETUP_SNIPER_DIR_CONSTANTS:
        assert c == c.upper(), f"dir constant {c!r} not uppercase"


def test_function_names_snake_case():
    import re
    rx = re.compile(r"^[a-z][a-z0-9_]*$")
    for fn in MOD.SETUP_GEM5_FUNCTIONS + MOD.SETUP_SNIPER_FUNCTIONS:
        assert rx.match(fn), f"function {fn!r} not snake_case"


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------


def test_audit_runs_and_returns_dict():
    data = MOD.audit()
    assert isinstance(data, dict)
    assert "violations" in data
    assert "rules" in data
    assert "registry" in data


def test_audit_clean_in_repo():
    """The repo's setup scripts must satisfy the gate."""
    data = MOD.audit()
    assert data["violations"] == [], (
        f"unexpected violations: {data['violations']}"
    )


def test_audit_status_active():
    data = MOD.audit()
    assert data["status"] == "active"


def test_audit_counts():
    data = MOD.audit()
    assert data["repo_urls_n"] == len(MOD.SETUP_REPO_URLS)
    assert data["gem5_functions_n"] == len(MOD.SETUP_GEM5_FUNCTIONS)
    assert data["sniper_functions_n"] == len(MOD.SETUP_SNIPER_FUNCTIONS)


def test_rule_ids_complete():
    data = MOD.audit()
    for rid in ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]:
        assert rid in data["rules"], f"missing rule {rid}"


# --------------------------------------------------------------------------
# Rule-level injection tests
# --------------------------------------------------------------------------


def test_s1_detects_missing_repo_url():
    orig = dict(MOD.SETUP_REPO_URLS)
    try:
        MOD.SETUP_REPO_URLS["GEM5_REPO_URL"] = "https://example.com/fake.git"
        data = MOD.audit()
        s1 = [v for v in data["violations"] if v["rule"] == "S1"]
        assert s1, "S1 did not detect drift"
    finally:
        MOD.SETUP_REPO_URLS.clear()
        MOD.SETUP_REPO_URLS.update(orig)


def test_s2_detects_missing_constant():
    orig = list(MOD.SETUP_GEM5_DIR_CONSTANTS)
    try:
        MOD.SETUP_GEM5_DIR_CONSTANTS.append("NONEXISTENT_BOGUS_DIR")
        data = MOD.audit()
        s2 = [v for v in data["violations"] if v["rule"] == "S2"]
        assert s2, "S2 did not detect missing constant"
    finally:
        MOD.SETUP_GEM5_DIR_CONSTANTS[:] = orig


def test_s3_detects_missing_gem5_function():
    orig = list(MOD.SETUP_GEM5_FUNCTIONS)
    try:
        MOD.SETUP_GEM5_FUNCTIONS.append("nonexistent_bogus_function")
        data = MOD.audit()
        s3 = [v for v in data["violations"] if v["rule"] == "S3"]
        assert s3, "S3 did not detect missing function"
    finally:
        MOD.SETUP_GEM5_FUNCTIONS[:] = orig


def test_s4_detects_missing_sniper_function():
    orig = list(MOD.SETUP_SNIPER_FUNCTIONS)
    try:
        MOD.SETUP_SNIPER_FUNCTIONS.append("nonexistent_bogus_function")
        data = MOD.audit()
        s4 = [v for v in data["violations"] if v["rule"] == "S4"]
        assert s4, "S4 did not detect missing function"
    finally:
        MOD.SETUP_SNIPER_FUNCTIONS[:] = orig


def test_s7_detects_unregistered_function():
    """Removing 'main' from canonical → S7 must flag main as unregistered."""
    orig = list(MOD.SETUP_GEM5_FUNCTIONS)
    try:
        MOD.SETUP_GEM5_FUNCTIONS[:] = [f for f in orig if f != "main"]
        data = MOD.audit()
        # main becomes "extra" (on disk but not in canonical)
        s7 = [v for v in data["violations"] if v["rule"] == "S7"]
        assert s7, "S7 did not detect unregistered function"
        assert any("main" in v["msg"] for v in s7), (
            "S7 did not name 'main' as unregistered"
        )
    finally:
        MOD.SETUP_GEM5_FUNCTIONS[:] = orig


def test_s7_detects_missing_canonical():
    """Adding a fake function to canonical → S7 must flag it as missing."""
    orig = list(MOD.SETUP_SNIPER_FUNCTIONS)
    try:
        MOD.SETUP_SNIPER_FUNCTIONS.append("zzz_fake_func_for_test")
        data = MOD.audit()
        # Both S4 (required missing) and S7 (canonical missing) flag this.
        s7 = [v for v in data["violations"] if v["rule"] == "S7"]
        assert s7, "S7 did not detect canonical-missing function"
    finally:
        MOD.SETUP_SNIPER_FUNCTIONS[:] = orig


# --------------------------------------------------------------------------
# Live cross-check against on-disk setup scripts
# --------------------------------------------------------------------------


def test_setup_gem5_script_exists():
    assert MOD.SETUP_GEM5.is_file(), f"missing {MOD.SETUP_GEM5}"


def test_setup_sniper_script_exists():
    assert MOD.SETUP_SNIPER.is_file(), f"missing {MOD.SETUP_SNIPER}"


def test_setup_gem5_actually_contains_canonical_functions():
    text = MOD.SETUP_GEM5.read_text(encoding="utf-8")
    found = set(MOD.DEF_RE.findall(text))
    for fn in MOD.SETUP_GEM5_FUNCTIONS:
        assert fn in found, f"canonical fn {fn!r} not in setup_gem5.py"


def test_setup_sniper_actually_contains_canonical_functions():
    text = MOD.SETUP_SNIPER.read_text(encoding="utf-8")
    found = set(MOD.DEF_RE.findall(text))
    for fn in MOD.SETUP_SNIPER_FUNCTIONS:
        assert fn in found, f"canonical fn {fn!r} not in setup_sniper.py"


def test_setup_gem5_actually_contains_canonical_constants():
    import re
    text = MOD.SETUP_GEM5.read_text(encoding="utf-8")
    for c in MOD.SETUP_GEM5_DIR_CONSTANTS:
        assert re.search(rf"^{c}\s*=", text, re.MULTILINE), (
            f"canonical const {c!r} not in setup_gem5.py"
        )


def test_setup_sniper_actually_contains_canonical_constants():
    import re
    text = MOD.SETUP_SNIPER.read_text(encoding="utf-8")
    for c in MOD.SETUP_SNIPER_DIR_CONSTANTS:
        assert re.search(rf"^{c}\s*=", text, re.MULTILINE), (
            f"canonical const {c!r} not in setup_sniper.py"
        )


# --------------------------------------------------------------------------
# Artifact (json/md/csv) on-disk parity
# --------------------------------------------------------------------------


def test_json_artifact_exists():
    assert (WIKI_DATA / "lit_faith_setup_script_registry.json").is_file()


def test_md_artifact_exists():
    assert (WIKI_DATA / "lit_faith_setup_script_registry.md").is_file()


def test_csv_artifact_exists():
    assert (WIKI_DATA / "lit_faith_setup_script_registry.csv").is_file()


def test_json_artifact_matches_live_audit():
    on_disk = json.loads(
        (WIKI_DATA / "lit_faith_setup_script_registry.json").read_text("utf-8")
    )
    live = MOD.audit()
    assert on_disk["repo_urls_n"] == live["repo_urls_n"]
    assert on_disk["gem5_functions_n"] == live["gem5_functions_n"]
    assert on_disk["sniper_functions_n"] == live["sniper_functions_n"]
    assert on_disk["violations"] == live["violations"]


def test_md_artifact_has_no_violations():
    md = (WIKI_DATA / "lit_faith_setup_script_registry.md").read_text("utf-8")
    if not MOD.audit()["violations"]:
        assert "✅ No violations" in md


def test_md_exactly_one_final_newline():
    md = (WIKI_DATA / "lit_faith_setup_script_registry.md").read_text("utf-8")
    assert md.endswith("\n")
    assert not md.endswith("\n\n")


def test_csv_artifact_has_repo_url_rows():
    csv_text = (WIKI_DATA / "lit_faith_setup_script_registry.csv").read_text("utf-8")
    assert "repo_url,GEM5_REPO_URL" in csv_text
    assert "repo_url,SNIPER_REPO_URL" in csv_text


# --------------------------------------------------------------------------
# Catalog wiring
# --------------------------------------------------------------------------


def test_catalog_has_gate_268_entry():
    catalog = (WIKI_DATA / "artifact_catalog.json")
    assert catalog.is_file()
    text = catalog.read_text("utf-8")
    assert "lit_faith_setup_script_registry" in text, (
        "gate 268 entry not in artifact_catalog.json"
    )
