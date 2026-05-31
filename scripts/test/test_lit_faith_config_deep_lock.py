"""Pytest gate for ``lit_faith_config_deep_lock`` (gate 269)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "scripts" / "experiments" / "ecg" / "lit_faith_config_deep_lock.py"
WIKI_DATA = ROOT / "wiki" / "data"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "lit_faith_config_deep_lock", str(GEN)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lit_faith_config_deep_lock"] = mod
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module()


# --------------------------------------------------------------------------
# Registry sanity
# --------------------------------------------------------------------------


def test_cache_anchors_non_empty():
    assert len(MOD.CONFIG_CACHE_ANCHORS) == 7
    for k in ["CACHE_L1_SIZE", "CACHE_L2_SIZE", "CACHE_L3_SIZE",
              "CACHE_L1_WAYS", "CACHE_L2_WAYS", "CACHE_L3_WAYS",
              "CACHE_LINE_SIZE"]:
        assert k in MOD.CONFIG_CACHE_ANCHORS


def test_cache_anchors_canonical_values():
    a = MOD.CONFIG_CACHE_ANCHORS
    assert a["CACHE_L1_SIZE"] == "32768"
    assert a["CACHE_L2_SIZE"] == "262144"
    assert a["CACHE_L3_SIZE"] == "8388608"
    assert a["CACHE_L1_WAYS"] == "8"
    assert a["CACHE_L2_WAYS"] == "4"
    assert a["CACHE_L3_WAYS"] == "16"
    assert a["CACHE_LINE_SIZE"] == "64"


def test_cache_sweep_range():
    assert MOD.CONFIG_CACHE_SWEEP_MIN == 32 * 1024
    assert MOD.CONFIG_CACHE_SWEEP_MAX == 64 * 1024 * 1024
    assert MOD.CONFIG_CACHE_SWEEP_N == 12


def test_ecg_modes_canonical():
    assert MOD.CONFIG_ECG_MODES == {"DBG_PRIMARY", "POPT_PRIMARY", "DBG_ONLY", "ECG_EMBEDDED"}


def test_graph_types_canonical():
    expected = {"Social", "Citation", "Road", "Web", "Content", "Mesh"}
    assert MOD.CONFIG_CANONICAL_GRAPH_TYPES == expected


def test_required_graph_keys():
    assert MOD.CONFIG_REQUIRED_GRAPH_KEYS == {"name", "short", "type", "vertices_m", "edges_m"}


def test_recognized_reorder_non_empty():
    assert "-o 0" in MOD.CONFIG_RECOGNIZED_REORDER
    assert "-o 5" in MOD.CONFIG_RECOGNIZED_REORDER


def test_canonical_relations_non_empty():
    assert "baseline" in MOD.CONFIG_CANONICAL_RELATIONS
    assert "grasp_beats_srrip" in MOD.CONFIG_CANONICAL_RELATIONS


def test_extra_allow_lists_are_sets():
    assert isinstance(MOD.CONFIG_CACHE_EXTRA_ALLOW, set)
    assert isinstance(MOD.CONFIG_RELATION_EXTRA_ALLOW, set)


# --------------------------------------------------------------------------
# Grammar
# --------------------------------------------------------------------------


def test_cache_anchor_values_are_str():
    for v in MOD.CONFIG_CACHE_ANCHORS.values():
        assert isinstance(v, str)
        assert v.isdigit()


def test_cache_anchor_values_powers_of_two():
    """All size/line anchors must be powers of 2."""
    for k, v in MOD.CONFIG_CACHE_ANCHORS.items():
        n = int(v)
        if "SIZE" in k or "LINE" in k:
            assert n > 0 and (n & (n - 1)) == 0, f"{k}={n} not power of 2"


def test_reorder_flags_format():
    for r in MOD.CONFIG_RECOGNIZED_REORDER:
        assert r.startswith("-o "), f"reorder {r!r} missing -o prefix"


def test_ecg_modes_uppercase():
    for m in MOD.CONFIG_ECG_MODES:
        assert m == m.upper(), f"mode {m!r} not uppercase"


def test_graph_types_titlecase():
    for t in MOD.CONFIG_CANONICAL_GRAPH_TYPES:
        assert t[0].isupper(), f"type {t!r} not title-case"


def test_relations_snake_case():
    import re
    rx = re.compile(r"^[a-z][a-z0-9_]*$")
    for rel in MOD.CONFIG_CANONICAL_RELATIONS:
        assert rx.match(rel), f"relation {rel!r} not snake_case"


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
    """The repo's ecg/config.py must satisfy the gate."""
    data = MOD.audit()
    assert data["violations"] == [], (
        f"unexpected violations: {data['violations']}"
    )


def test_audit_status_active():
    data = MOD.audit()
    assert data["status"] == "active"


def test_audit_counts():
    data = MOD.audit()
    assert data["anchors_n"] == 7
    assert data["sweep_n"] == 12
    assert data["benchmarks_n"] >= 5
    assert data["policies_n"] >= 8
    assert data["ecg_modes_n"] == 4
    assert data["graphs_n"] >= 6


def test_rule_ids_complete():
    data = MOD.audit()
    for rid in ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]:
        assert rid in data["rules"], f"missing rule {rid}"


# --------------------------------------------------------------------------
# Live cross-check against ecg/config.py
# --------------------------------------------------------------------------


def test_ecg_config_exists():
    assert MOD.ECG_CONFIG.is_file(), f"missing {MOD.ECG_CONFIG}"


def test_live_config_loads():
    cfg = MOD._load_config()
    assert cfg is not None
    assert hasattr(cfg, "DEFAULT_CACHE")
    assert hasattr(cfg, "BENCHMARKS")
    assert hasattr(cfg, "ALL_POLICIES")


def test_live_default_cache_matches_anchors():
    cfg = MOD._load_config()
    for k, v in MOD.CONFIG_CACHE_ANCHORS.items():
        assert cfg.DEFAULT_CACHE[k] == v, f"{k} mismatch"


def test_live_cache_sweep_is_ascending_pow2():
    cfg = MOD._load_config()
    sweep = cfg.CACHE_SIZES_SWEEP
    assert sweep == sorted(sweep)
    for v in sweep:
        assert (v & (v - 1)) == 0


def test_live_policies_partition():
    cfg = MOD._load_config()
    union = set(cfg.BASELINE_POLICIES) | set(cfg.GRAPH_AWARE_POLICIES)
    assert union == set(cfg.ALL_POLICIES)
    assert not (set(cfg.BASELINE_POLICIES) & set(cfg.GRAPH_AWARE_POLICIES))


def test_live_benchmarks_partition():
    cfg = MOD._load_config()
    iter_b = set(cfg.ITERATIVE_BENCHMARKS)
    trav_b = set(cfg.TRAVERSAL_BENCHMARKS)
    assert iter_b.issubset(cfg.BENCHMARKS)
    assert trav_b.issubset(cfg.BENCHMARKS)
    assert not (iter_b & trav_b)


def test_live_ecg_modes_matches_canonical():
    cfg = MOD._load_config()
    assert set(cfg.ECG_MODES) == MOD.CONFIG_ECG_MODES


def test_live_eval_graphs_well_formed():
    cfg = MOD._load_config()
    for g in cfg.EVAL_GRAPHS:
        for k in MOD.CONFIG_REQUIRED_GRAPH_KEYS:
            assert k in g
        assert g["type"] in MOD.CONFIG_CANONICAL_GRAPH_TYPES


# --------------------------------------------------------------------------
# Rule-level injection tests
# --------------------------------------------------------------------------


def test_c1_detects_cache_anchor_drift():
    orig = dict(MOD.CONFIG_CACHE_ANCHORS)
    try:
        MOD.CONFIG_CACHE_ANCHORS["CACHE_L1_SIZE"] = "999"
        data = MOD.audit()
        c1 = [v for v in data["violations"] if v["rule"] == "C1"]
        assert c1, "C1 did not detect anchor drift"
    finally:
        MOD.CONFIG_CACHE_ANCHORS.clear()
        MOD.CONFIG_CACHE_ANCHORS.update(orig)


def test_c5_detects_unknown_ecg_mode():
    orig = set(MOD.CONFIG_ECG_MODES)
    try:
        MOD.CONFIG_ECG_MODES.clear()
        MOD.CONFIG_ECG_MODES.update({"FAKE_MODE"})
        data = MOD.audit()
        c5 = [v for v in data["violations"] if v["rule"] == "C5"]
        assert c5, "C5 did not detect unknown mode"
    finally:
        MOD.CONFIG_ECG_MODES.clear()
        MOD.CONFIG_ECG_MODES.update(orig)


def test_c6_detects_unknown_graph_type():
    orig = set(MOD.CONFIG_CANONICAL_GRAPH_TYPES)
    try:
        MOD.CONFIG_CANONICAL_GRAPH_TYPES.clear()
        MOD.CONFIG_CANONICAL_GRAPH_TYPES.add("Bogus")
        data = MOD.audit()
        c6 = [v for v in data["violations"] if v["rule"] == "C6"]
        assert c6, "C6 did not flag non-canonical graph types"
    finally:
        MOD.CONFIG_CANONICAL_GRAPH_TYPES.clear()
        MOD.CONFIG_CANONICAL_GRAPH_TYPES.update(orig)


def test_c7_detects_unknown_reorder_flag():
    orig = set(MOD.CONFIG_RECOGNIZED_REORDER)
    try:
        MOD.CONFIG_RECOGNIZED_REORDER.clear()
        MOD.CONFIG_RECOGNIZED_REORDER.add("-o 99")
        data = MOD.audit()
        c7 = [v for v in data["violations"] if v["rule"] == "C7"]
        assert c7, "C7 did not detect unknown reorder flag"
    finally:
        MOD.CONFIG_RECOGNIZED_REORDER.clear()
        MOD.CONFIG_RECOGNIZED_REORDER.update(orig)


def test_c8_detects_unknown_relation():
    orig = set(MOD.CONFIG_CANONICAL_RELATIONS)
    try:
        MOD.CONFIG_CANONICAL_RELATIONS.clear()
        data = MOD.audit()
        c8 = [v for v in data["violations"] if v["rule"] == "C8"]
        assert c8, "C8 did not detect unknown relation"
    finally:
        MOD.CONFIG_CANONICAL_RELATIONS.clear()
        MOD.CONFIG_CANONICAL_RELATIONS.update(orig)


# --------------------------------------------------------------------------
# Artifact (json/md/csv) on-disk parity
# --------------------------------------------------------------------------


def test_json_artifact_exists():
    assert (WIKI_DATA / "lit_faith_config_deep_lock.json").is_file()


def test_md_artifact_exists():
    assert (WIKI_DATA / "lit_faith_config_deep_lock.md").is_file()


def test_csv_artifact_exists():
    assert (WIKI_DATA / "lit_faith_config_deep_lock.csv").is_file()


def test_json_artifact_matches_live_audit():
    on_disk = json.loads(
        (WIKI_DATA / "lit_faith_config_deep_lock.json").read_text("utf-8")
    )
    live = MOD.audit()
    for key in ["anchors_n", "sweep_n", "benchmarks_n", "policies_n",
                "ecg_modes_n", "graphs_n", "reorder_variants_n",
                "accuracy_pairs_n"]:
        assert on_disk[key] == live[key], f"{key} mismatch"
    assert on_disk["violations"] == live["violations"]


def test_md_artifact_has_no_violations():
    md = (WIKI_DATA / "lit_faith_config_deep_lock.md").read_text("utf-8")
    if not MOD.audit()["violations"]:
        assert "✅ No violations" in md


def test_md_exactly_one_final_newline():
    md = (WIKI_DATA / "lit_faith_config_deep_lock.md").read_text("utf-8")
    assert md.endswith("\n")
    assert not md.endswith("\n\n")


def test_csv_has_cache_anchor_rows():
    csv_text = (WIKI_DATA / "lit_faith_config_deep_lock.csv").read_text("utf-8")
    assert "cache_anchor,CACHE_L1_SIZE,32768" in csv_text
    assert "cache_anchor,CACHE_L3_SIZE,8388608" in csv_text


# --------------------------------------------------------------------------
# Catalog wiring
# --------------------------------------------------------------------------


def test_catalog_has_gate_269_entry():
    catalog = (WIKI_DATA / "artifact_catalog.json")
    assert catalog.is_file()
    text = catalog.read_text("utf-8")
    assert "lit_faith_config_deep_lock" in text, (
        "gate 269 entry not in artifact_catalog.json"
    )
