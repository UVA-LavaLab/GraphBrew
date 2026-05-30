"""Tests for gate 263 — ECG configuration matrix registry.

Locks the invariant that the central
``scripts/experiments/ecg/config.py`` module remains in sync
with the canonical registries (policies, graphs, kernels, L3
sizes) and is internally self-consistent (no orphan ECG modes,
no pair with a non-existent policy, sweep brackets the anchor).

Cases pass today; any regression (e.g. someone adds an unregistered
policy to BASELINE_POLICIES, or renames an EVAL_GRAPHS entry
without registering an alias, or bumps DEFAULT_CACHE.CACHE_L3_SIZE
to a non-canonical byte count) fires a precise assertion that
points at the exact field and the exact registry that's out of
sync.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN = REPO_ROOT / "scripts" / "experiments" / "ecg" / "lit_faith_config_matrix.py"


def _load_gen():
    spec = importlib.util.spec_from_file_location("gen_gate263_test", GEN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gen_gate263_test"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load_gen()


@pytest.fixture(scope="module")
def data(gen):
    return gen.audit()


# --- shape ----------------------------------------------------------------

def test_status_active(data):
    assert data["status"] == "active"


def test_no_violations(data):
    assert data["violations"] == [], (
        "Config matrix has drifted relative to canonical registries:\n"
        + "\n".join(f"  - {v['rule']} {v['subject']}: {v['reason']}"
                    for v in data["violations"])
    )


def test_rule_keys_present(data):
    expected = {"C1", "C2", "C3", "C4", "C5", "C6", "C7"}
    assert set(data["rules"].keys()) == expected


def test_totals_positive(data):
    assert data["n_policies_checked"] >= 5
    assert data["n_graphs_checked"] >= 3
    assert data["n_kernels_checked"] >= 3
    assert data["n_sweep_entries"] >= 6
    assert data["n_pairs_checked"] >= 10
    assert data["n_ecg_modes"] >= 2


# --- C1 vocab -------------------------------------------------------------

def test_baseline_policies_all_canonical(data):
    canonical = set(data["all_policies"])
    for p in data["baseline_policies"]:
        assert p in canonical, f"baseline policy {p!r} dropped out of ALL_POLICIES"


def test_graph_aware_policies_all_canonical(data):
    canonical = set(data["all_policies"])
    for p in data["graph_aware_policies"]:
        assert p in canonical, f"graph-aware policy {p!r} dropped out of ALL_POLICIES"


def test_preview_subset_of_all(data):
    all_pol = set(data["all_policies"])
    for p in data["preview_policies"]:
        assert p in all_pol, f"PREVIEW_POLICIES contains {p!r} not in ALL_POLICIES"


# --- C2 graphs ------------------------------------------------------------

def test_eval_graphs_nonempty(data):
    assert data["eval_graphs"], "EVAL_GRAPHS is empty"


def test_eval_graphs_unique(data):
    names = data["eval_graphs"]
    assert len(set(names)) == len(names), f"duplicate EVAL_GRAPHS name: {names}"


# --- C3 benchmarks --------------------------------------------------------

def test_benchmarks_nonempty(data):
    assert data["benchmarks"], "BENCHMARKS is empty"


def test_pr_is_canonical_benchmark(data):
    assert "pr" in data["benchmarks"], "PR (the paper anchor kernel) missing from BENCHMARKS"


# --- C4 L3 anchor ---------------------------------------------------------

def test_default_l3_is_paper_anchor(data):
    assert data["default_l3_bytes"] == 8 * 1024 * 1024, (
        f"DEFAULT_CACHE.CACHE_L3_SIZE drifted from 8 MB anchor: "
        f"got {data['default_l3_bytes']}"
    )
    assert data["default_l3_tier"] == "8MB", (
        f"default_l3_tier mismatch: got {data['default_l3_tier']!r}"
    )


# --- C5 sweep -------------------------------------------------------------

def test_sweep_strictly_increasing(data):
    s = data["cache_sweep_bytes"]
    for i in range(1, len(s)):
        assert s[i] > s[i - 1], f"CACHE_SIZES_SWEEP not increasing at {i}: {s}"


def test_sweep_all_power_of_two(data):
    for b in data["cache_sweep_bytes"]:
        assert b > 0 and (b & (b - 1)) == 0, (
            f"CACHE_SIZES_SWEEP entry {b} not a power-of-2"
        )


def test_sweep_brackets_anchor(data):
    s = data["cache_sweep_bytes"]
    a = data["default_l3_bytes"]
    assert s[0] <= a <= s[-1], (
        f"sweep [{s[0]}..{s[-1]}] doesn't bracket anchor {a}"
    )


# --- C6 pairs -------------------------------------------------------------

def test_all_policies_is_union(gen):
    cfg_path = REPO_ROOT / "scripts" / "experiments" / "ecg" / "config.py"
    cfg = gen._load(  # type: ignore[attr-defined]
        "ecg_config_check_union", cfg_path
    )
    expected = set(cfg.BASELINE_POLICIES) | set(cfg.GRAPH_AWARE_POLICIES)
    assert set(cfg.ALL_POLICIES) == expected, (
        f"ALL_POLICIES not the union of BASELINE + GRAPH_AWARE: "
        f"got {set(cfg.ALL_POLICIES)}, expected {expected}"
    )


# --- C7 ECG modes ---------------------------------------------------------

def test_observed_modes_subset_of_declared(data):
    declared = set(data["ecg_modes"])
    observed = set(data["observed_ecg_modes"])
    missing = observed - declared
    assert not missing, (
        f"ECG modes observed in ACCURACY_PAIRS but undeclared: {missing}"
    )


def test_dbg_primary_declared(data):
    assert "DBG_PRIMARY" in data["ecg_modes"], (
        "DBG_PRIMARY (the ECG paper-shipping mode) missing from ECG_MODES"
    )


# --- cross-registry sanity -------------------------------------------------

def test_baseline_policies_in_canonical_policy_names(gen):
    pol = gen._load_policy_registry()  # type: ignore[attr-defined]
    canonical = set(pol.CANONICAL_POLICY_NAMES)
    cfg = gen._load_config()  # type: ignore[attr-defined]
    for p in cfg.BASELINE_POLICIES:
        assert p in canonical, (
            f"BASELINE_POLICIES contains {p!r} not in gate 255 CANONICAL_POLICY_NAMES"
        )


def test_graph_aware_policies_in_canonical_policy_names(gen):
    pol = gen._load_policy_registry()  # type: ignore[attr-defined]
    canonical = set(pol.CANONICAL_POLICY_NAMES)
    cfg = gen._load_config()  # type: ignore[attr-defined]
    for p in cfg.GRAPH_AWARE_POLICIES:
        assert p in canonical, (
            f"GRAPH_AWARE_POLICIES contains {p!r} not in gate 255 CANONICAL_POLICY_NAMES"
        )


def test_eval_graph_names_in_canonical_graph_registry(gen):
    gr = gen._load_graph_registry()  # type: ignore[attr-defined]
    canonical = {g.name for g in gr.CANONICAL_GRAPHS}
    cfg = gen._load_config()  # type: ignore[attr-defined]
    for g in cfg.EVAL_GRAPHS:
        assert g["name"] in canonical, (
            f"EVAL_GRAPHS contains {g['name']!r} not in gate 258 CANONICAL_GRAPHS"
        )


def test_benchmarks_are_known_kernels(gen):
    cli = gen._load_cli_registry()  # type: ignore[attr-defined]
    canonical = set(cli.KERNEL_CL_CLASS.keys())
    cfg = gen._load_config()  # type: ignore[attr-defined]
    for b in cfg.BENCHMARKS:
        assert b in canonical, (
            f"BENCHMARKS contains {b!r} not in gate 260 KERNEL_CL_CLASS"
        )


def test_default_l3_matches_canonical_tier(gen):
    l3 = gen._load_l3_registry()  # type: ignore[attr-defined]
    canonical_bytes = {t["bytes"] for t in l3.CANONICAL_L3_TIERS.values()}
    cfg = gen._load_config()  # type: ignore[attr-defined]
    assert int(cfg.DEFAULT_CACHE["CACHE_L3_SIZE"]) in canonical_bytes, (
        f"DEFAULT_CACHE.CACHE_L3_SIZE={cfg.DEFAULT_CACHE['CACHE_L3_SIZE']} "
        f"matches no CANONICAL_L3_TIERS bytes count"
    )


# --- pair structural integrity --------------------------------------------

def test_accuracy_pair_arity(gen):
    cfg = gen._load_config()  # type: ignore[attr-defined]
    for i, p in enumerate(cfg.ACCURACY_PAIRS):
        assert isinstance(p, tuple), f"ACCURACY_PAIRS[{i}] not a tuple: {type(p)}"
        assert len(p) >= 2, f"ACCURACY_PAIRS[{i}] arity < 2: {p}"
        assert isinstance(p[0], str), f"ACCURACY_PAIRS[{i}][0] not str: {p[0]!r}"
        assert isinstance(p[1], str), f"ACCURACY_PAIRS[{i}][1] not str: {p[1]!r}"


def test_reorder_policy_pair_arity(gen):
    cfg = gen._load_config()  # type: ignore[attr-defined]
    for i, p in enumerate(cfg.REORDER_POLICY_PAIRS):
        assert isinstance(p, tuple), f"REORDER_POLICY_PAIRS[{i}] not a tuple: {type(p)}"
        assert len(p) >= 2, f"REORDER_POLICY_PAIRS[{i}] arity < 2: {p}"
        assert isinstance(p[0], str)
        assert isinstance(p[1], str)


def test_accuracy_pairs_env_is_dict(gen):
    cfg = gen._load_config()  # type: ignore[attr-defined]
    for i, p in enumerate(cfg.ACCURACY_PAIRS):
        if len(p) >= 3 and p[2] is not None:
            assert isinstance(p[2], dict), (
                f"ACCURACY_PAIRS[{i}].env not dict-or-None: {p[2]!r}"
            )


# --- sweep boundary anchors -----------------------------------------------

def test_sweep_min_at_or_below_32kb(data):
    assert data["cache_sweep_bytes"][0] <= 32 * 1024, (
        "sweep doesn't start at or below 32 kB (L1 floor for ECG paper figs)"
    )


def test_sweep_max_at_or_above_8mb(data):
    assert data["cache_sweep_bytes"][-1] >= 8 * 1024 * 1024, (
        "sweep doesn't reach the 8 MB L3 anchor"
    )


# --- emit shape -----------------------------------------------------------

def test_emit_json(tmp_path, gen, data):
    out = tmp_path / "out.json"
    gen._emit_json(data, out)  # type: ignore[attr-defined]
    assert out.exists()
    import json
    j = json.loads(out.read_text())
    assert j["status"] == "active"
    assert j["default_l3_tier"] == "8MB"


def test_emit_md(tmp_path, gen, data):
    out = tmp_path / "out.md"
    gen._emit_md(data, out)  # type: ignore[attr-defined]
    txt = out.read_text()
    assert "Gate 263" in txt
    assert "C1" in txt and "C7" in txt
    assert txt.endswith("\n")


def test_emit_csv(tmp_path, gen, data):
    out = tmp_path / "out.csv"
    gen._emit_csv(data, out)  # type: ignore[attr-defined]
    rows = out.read_text().splitlines()
    assert rows[0] == "kind,name,extra"
    assert any("benchmark," in r for r in rows)
    assert any("eval_graph," in r for r in rows)


def test_main_zero_exit(gen, tmp_path):
    rc = gen.main([
        "--json-out", str(tmp_path / "j.json"),
        "--md-out", str(tmp_path / "m.md"),
        "--csv-out", str(tmp_path / "c.csv"),
    ])
    assert rc == 0


# --- regression bait: snapshot key invariants -----------------------------

def test_n_policies_checked_at_least_5(data):
    assert data["n_policies_checked"] >= 5


def test_n_graphs_checked_at_least_3(data):
    assert data["n_graphs_checked"] >= 3


def test_n_kernels_checked_at_least_3(data):
    assert data["n_kernels_checked"] >= 3


def test_n_sweep_entries_at_least_6(data):
    assert data["n_sweep_entries"] >= 6


def test_n_pairs_checked_at_least_10(data):
    assert data["n_pairs_checked"] >= 10


def test_n_ecg_modes_at_least_2(data):
    assert data["n_ecg_modes"] >= 2


def test_ecg_private_modes_documented(data):
    assert data["ecg_private_modes"], (
        "ECG_PRIVATE_MODES is empty; either populate it or remove rule C7's allowlist"
    )


def test_paper_anchor_tier_matches_l3_anchor(data):
    assert data["default_l3_tier"] in ("1MB", "4MB", "8MB"), (
        f"default_l3_tier {data['default_l3_tier']!r} not in gate 251 "
        f"ANCHOR_TRIPLET ('1MB','4MB','8MB')"
    )
