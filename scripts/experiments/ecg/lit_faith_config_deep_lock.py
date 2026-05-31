#!/usr/bin/env python3
"""Generate the gate 269 ECG config deep-lock registry.

Locks ``scripts/experiments/ecg/config.py`` — the central configuration
module that fixes which policies / benchmarks / graphs / cache anchors
the ECG paper experiments use — against silent drift in:

  - the cache-anchor numeric triplet (L1=32K, L2=256K, L3=8M; line=64)
  - the cache-sweep grid (powers-of-2, ascending, 32K..64MB, 12 points)
  - the benchmark partition (ITERATIVE ∪ TRAVERSAL ⊆ BENCHMARKS; no overlap)
  - the policy partition (BASELINE ∪ GRAPH_AWARE == ALL; PREVIEW ⊆ ALL)
  - the ECG mode set (4 enum tokens; subset of canonical)
  - the EVAL_GRAPHS schema (required keys, recognized types)
  - the reorder-flag vocabulary used by REORDER_VARIANTS and PAIRS
  - the ACCURACY_PAIRS expected-relation vocabulary

Where gate 256 covers profile-NAMES, gate 269 covers profile CONTENT.

8 rules C1-C8:

  C1 DEFAULT_CACHE contains canonical cache anchors (L1=32768, L2=262144,
     L3=8388608, ways 8/4/16, line=64) — exact byte values.
  C2 CACHE_SIZES_SWEEP is a 12-point ascending power-of-2 grid spanning
     32 KiB .. 64 MiB inclusive.
  C3 ITERATIVE_BENCHMARKS ∪ TRAVERSAL_BENCHMARKS ⊆ BENCHMARKS; no overlap;
     BENCHMARKS_PREVIEW ⊆ BENCHMARKS.
  C4 BASELINE_POLICIES ∪ GRAPH_AWARE_POLICIES == ALL_POLICIES; no overlap;
     PREVIEW_POLICIES ⊆ ALL_POLICIES.
  C5 ECG_MODES is the canonical 4-token set (DBG_PRIMARY, POPT_PRIMARY,
     DBG_ONLY, ECG_EMBEDDED).
  C6 EVAL_GRAPHS entries have required keys (name, short, type,
     vertices_m, edges_m); type from CANONICAL_GRAPH_TYPES; counts > 0.
  C7 REORDER_VARIANTS uses recognized reorder flags ("-o N" or "-o N:tag")
     and labels are non-empty title-case identifiers.
  C8 ACCURACY_PAIRS entries' expected_relation token belongs to
     CANONICAL_RELATIONS; policy ∈ ALL_POLICIES; reorder ∈ recognized.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ECG_CONFIG = PROJECT_ROOT / "scripts" / "experiments" / "ecg" / "config.py"

# Canonical anchor values.
CONFIG_CACHE_ANCHORS = {
    "CACHE_L1_SIZE": "32768",
    "CACHE_L1_WAYS": "8",
    "CACHE_L2_SIZE": "262144",
    "CACHE_L2_WAYS": "4",
    "CACHE_L3_SIZE": "8388608",
    "CACHE_L3_WAYS": "16",
    "CACHE_LINE_SIZE": "64",
}

# Cache sweep grid: 12 powers-of-2 ascending, 32 KiB .. 64 MiB.
CONFIG_CACHE_SWEEP_MIN = 32 * 1024            # 32 KiB
CONFIG_CACHE_SWEEP_MAX = 64 * 1024 * 1024     # 64 MiB
CONFIG_CACHE_SWEEP_N = 12

# Canonical ECG modes (must match the ECGMode enum in the C++ layer).
CONFIG_ECG_MODES = {"DBG_PRIMARY", "POPT_PRIMARY", "DBG_ONLY", "ECG_EMBEDDED"}

# Canonical graph-family taxonomy (must match gate 250 graph-family registry).
CONFIG_CANONICAL_GRAPH_TYPES = {"Social", "Citation", "Road", "Web", "Content", "Mesh"}

CONFIG_REQUIRED_GRAPH_KEYS = {"name", "short", "type", "vertices_m", "edges_m"}

# Recognized reorder flag values (subset that ECG ships with).
CONFIG_RECOGNIZED_REORDER = {
    "-o 0",          # Original
    "-o 5",          # DBG (Faldu et al.)
    "-o 8:csr",      # RabbitOrder (csr backend)
    "-o 12:leiden",  # GraphBrew (leiden community)
}

# Recognized expected-relation tokens in ACCURACY_PAIRS.
CONFIG_CANONICAL_RELATIONS = {
    "baseline",
    "grasp_beats_srrip",
    "grasp_no_dbg_eq_srrip",
    "popt_best",
    "popt_reorder_agnostic",
    "ecg_dbg_eq_grasp",
    "ecg_popt_approach",
    "ecg_sweet_spot",
}

# Allow-list extension hooks (empty by default — any drift demands review).
CONFIG_CACHE_EXTRA_ALLOW: set[str] = set()
CONFIG_RELATION_EXTRA_ALLOW: set[str] = set()


@dataclass
class Violation:
    rule: str
    where: str
    msg: str


@dataclass
class AuditResult:
    status: str = "active"
    anchors_n: int = 0
    sweep_n: int = 0
    benchmarks_n: int = 0
    policies_n: int = 0
    ecg_modes_n: int = 0
    graphs_n: int = 0
    reorder_variants_n: int = 0
    accuracy_pairs_n: int = 0
    violations: list[Violation] = field(default_factory=list)


def _add(out: AuditResult, rule: str, where: str, msg: str) -> None:
    out.violations.append(Violation(rule, where, msg))


def _load_config():
    if not ECG_CONFIG.is_file():
        return None
    spec = importlib.util.spec_from_file_location("_ecg_config_live", str(ECG_CONFIG))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------


def rule_c1(out: AuditResult, cfg) -> None:
    if cfg is None:
        _add(out, "C1", str(ECG_CONFIG), "ecg/config.py missing")
        return
    live = getattr(cfg, "DEFAULT_CACHE", None)
    if not isinstance(live, dict):
        _add(out, "C1", "DEFAULT_CACHE", "missing or not a dict")
        return
    for key, expected in CONFIG_CACHE_ANCHORS.items():
        actual = live.get(key)
        if actual != expected:
            _add(out, "C1", key,
                 f"cache anchor drift: expected={expected!r} actual={actual!r}")


def rule_c2(out: AuditResult, cfg) -> None:
    if cfg is None:
        return
    sweep = getattr(cfg, "CACHE_SIZES_SWEEP", None)
    if not isinstance(sweep, list):
        _add(out, "C2", "CACHE_SIZES_SWEEP", "missing or not a list")
        return
    if len(sweep) != CONFIG_CACHE_SWEEP_N:
        _add(out, "C2", "CACHE_SIZES_SWEEP",
             f"expected {CONFIG_CACHE_SWEEP_N} points, got {len(sweep)}")
    if sweep != sorted(sweep):
        _add(out, "C2", "CACHE_SIZES_SWEEP", "grid not ascending")
    if sweep:
        if sweep[0] != CONFIG_CACHE_SWEEP_MIN:
            _add(out, "C2", "CACHE_SIZES_SWEEP[0]",
                 f"min expected {CONFIG_CACHE_SWEEP_MIN}, got {sweep[0]}")
        if sweep[-1] != CONFIG_CACHE_SWEEP_MAX:
            _add(out, "C2", "CACHE_SIZES_SWEEP[-1]",
                 f"max expected {CONFIG_CACHE_SWEEP_MAX}, got {sweep[-1]}")
    for v in sweep:
        if not isinstance(v, int) or v <= 0 or (v & (v - 1)) != 0:
            _add(out, "C2", f"sweep_value_{v}",
                 "sweep values must be positive powers of 2")


def rule_c3(out: AuditResult, cfg) -> None:
    if cfg is None:
        return
    benches = set(getattr(cfg, "BENCHMARKS", []) or [])
    iter_b = set(getattr(cfg, "ITERATIVE_BENCHMARKS", []) or [])
    trav_b = set(getattr(cfg, "TRAVERSAL_BENCHMARKS", []) or [])
    prev_b = set(getattr(cfg, "BENCHMARKS_PREVIEW", []) or [])
    if not benches:
        _add(out, "C3", "BENCHMARKS", "empty or missing")
        return
    overlap = iter_b & trav_b
    if overlap:
        _add(out, "C3", "ITERATIVE∩TRAVERSAL",
             f"benchmarks classified twice: {sorted(overlap)}")
    extra_iter = iter_b - benches
    if extra_iter:
        _add(out, "C3", "ITERATIVE_BENCHMARKS",
             f"benchmarks not in BENCHMARKS: {sorted(extra_iter)}")
    extra_trav = trav_b - benches
    if extra_trav:
        _add(out, "C3", "TRAVERSAL_BENCHMARKS",
             f"benchmarks not in BENCHMARKS: {sorted(extra_trav)}")
    extra_prev = prev_b - benches
    if extra_prev:
        _add(out, "C3", "BENCHMARKS_PREVIEW",
             f"benchmarks not in BENCHMARKS: {sorted(extra_prev)}")


def rule_c4(out: AuditResult, cfg) -> None:
    if cfg is None:
        return
    baseline = set(getattr(cfg, "BASELINE_POLICIES", []) or [])
    graph_aw = set(getattr(cfg, "GRAPH_AWARE_POLICIES", []) or [])
    all_pol = set(getattr(cfg, "ALL_POLICIES", []) or [])
    preview = set(getattr(cfg, "PREVIEW_POLICIES", []) or [])
    if not all_pol:
        _add(out, "C4", "ALL_POLICIES", "empty or missing")
        return
    overlap = baseline & graph_aw
    if overlap:
        _add(out, "C4", "BASELINE∩GRAPH_AWARE",
             f"policies classified twice: {sorted(overlap)}")
    union = baseline | graph_aw
    if union != all_pol:
        missing = all_pol - union
        extra = union - all_pol
        _add(out, "C4", "BASELINE∪GRAPH_AWARE",
             f"partition != ALL_POLICIES: missing={sorted(missing)} extra={sorted(extra)}")
    extra_prev = preview - all_pol
    if extra_prev:
        _add(out, "C4", "PREVIEW_POLICIES",
             f"policies not in ALL_POLICIES: {sorted(extra_prev)}")


def rule_c5(out: AuditResult, cfg) -> None:
    if cfg is None:
        return
    modes = set(getattr(cfg, "ECG_MODES", []) or [])
    if not modes:
        _add(out, "C5", "ECG_MODES", "empty or missing")
        return
    extra = modes - CONFIG_ECG_MODES
    missing = CONFIG_ECG_MODES - modes
    if extra:
        _add(out, "C5", "ECG_MODES",
             f"unknown ECG modes: {sorted(extra)} (canonical={sorted(CONFIG_ECG_MODES)})")
    if missing:
        _add(out, "C5", "ECG_MODES",
             f"missing canonical ECG modes: {sorted(missing)}")


def rule_c6(out: AuditResult, cfg) -> None:
    if cfg is None:
        return
    graphs = getattr(cfg, "EVAL_GRAPHS", []) or []
    if not graphs:
        _add(out, "C6", "EVAL_GRAPHS", "empty or missing")
        return
    for i, g in enumerate(graphs):
        if not isinstance(g, dict):
            _add(out, "C6", f"EVAL_GRAPHS[{i}]",
                 f"entry is not a dict: {type(g).__name__}")
            continue
        missing = CONFIG_REQUIRED_GRAPH_KEYS - set(g.keys())
        if missing:
            _add(out, "C6", f"EVAL_GRAPHS[{i}]({g.get('name', '?')})",
                 f"missing required keys: {sorted(missing)}")
            continue
        if g["type"] not in CONFIG_CANONICAL_GRAPH_TYPES:
            _add(out, "C6", f"EVAL_GRAPHS[{i}].type",
                 f"unrecognized type {g['type']!r} (canonical={sorted(CONFIG_CANONICAL_GRAPH_TYPES)})")
        if not isinstance(g["vertices_m"], (int, float)) or g["vertices_m"] <= 0:
            _add(out, "C6", f"EVAL_GRAPHS[{i}].vertices_m",
                 f"vertices_m must be positive number, got {g['vertices_m']!r}")
        if not isinstance(g["edges_m"], (int, float)) or g["edges_m"] <= 0:
            _add(out, "C6", f"EVAL_GRAPHS[{i}].edges_m",
                 f"edges_m must be positive number, got {g['edges_m']!r}")


def rule_c7(out: AuditResult, cfg) -> None:
    if cfg is None:
        return
    variants = getattr(cfg, "REORDER_VARIANTS", []) or []
    if not variants:
        _add(out, "C7", "REORDER_VARIANTS", "empty or missing")
        return
    seen_labels: set[str] = set()
    for i, entry in enumerate(variants):
        if not isinstance(entry, tuple) or len(entry) != 2:
            _add(out, "C7", f"REORDER_VARIANTS[{i}]",
                 "entry must be (reorder_flag, label) tuple")
            continue
        flag, label = entry
        if flag not in CONFIG_RECOGNIZED_REORDER:
            _add(out, "C7", f"REORDER_VARIANTS[{i}]({label})",
                 f"unrecognized reorder flag {flag!r}")
        if not (isinstance(label, str) and label and label[0].isupper()):
            _add(out, "C7", f"REORDER_VARIANTS[{i}]",
                 f"label must be non-empty title-case identifier, got {label!r}")
        if label in seen_labels:
            _add(out, "C7", f"REORDER_VARIANTS[{i}]",
                 f"duplicate label {label!r}")
        seen_labels.add(label)


def rule_c8(out: AuditResult, cfg) -> None:
    if cfg is None:
        return
    pairs = getattr(cfg, "ACCURACY_PAIRS", []) or []
    all_pol = set(getattr(cfg, "ALL_POLICIES", []) or [])
    if not pairs:
        _add(out, "C8", "ACCURACY_PAIRS", "empty or missing")
        return
    for i, entry in enumerate(pairs):
        if not isinstance(entry, tuple) or len(entry) != 5:
            _add(out, "C8", f"ACCURACY_PAIRS[{i}]",
                 "entry must be 5-tuple (reorder, policy, env, label, relation)")
            continue
        reorder, policy, env, label, relation = entry
        if reorder not in CONFIG_RECOGNIZED_REORDER:
            _add(out, "C8", f"ACCURACY_PAIRS[{i}]({label})",
                 f"unrecognized reorder flag {reorder!r}")
        if policy not in all_pol:
            _add(out, "C8", f"ACCURACY_PAIRS[{i}]({label})",
                 f"policy {policy!r} not in ALL_POLICIES")
        if relation not in CONFIG_CANONICAL_RELATIONS and relation not in CONFIG_RELATION_EXTRA_ALLOW:
            _add(out, "C8", f"ACCURACY_PAIRS[{i}]({label})",
                 f"unrecognized expected-relation token {relation!r}")
        if not isinstance(env, dict):
            _add(out, "C8", f"ACCURACY_PAIRS[{i}]({label})",
                 f"env must be dict, got {type(env).__name__}")


def audit() -> dict:
    out = AuditResult()
    cfg = _load_config()
    rule_c1(out, cfg)
    rule_c2(out, cfg)
    rule_c3(out, cfg)
    rule_c4(out, cfg)
    rule_c5(out, cfg)
    rule_c6(out, cfg)
    rule_c7(out, cfg)
    rule_c8(out, cfg)
    if cfg is not None:
        out.anchors_n = len(getattr(cfg, "DEFAULT_CACHE", {}) or {})
        out.sweep_n = len(getattr(cfg, "CACHE_SIZES_SWEEP", []) or [])
        out.benchmarks_n = len(getattr(cfg, "BENCHMARKS", []) or [])
        out.policies_n = len(getattr(cfg, "ALL_POLICIES", []) or [])
        out.ecg_modes_n = len(getattr(cfg, "ECG_MODES", []) or [])
        out.graphs_n = len(getattr(cfg, "EVAL_GRAPHS", []) or [])
        out.reorder_variants_n = len(getattr(cfg, "REORDER_VARIANTS", []) or [])
        out.accuracy_pairs_n = len(getattr(cfg, "ACCURACY_PAIRS", []) or [])
    return {
        "status":             out.status,
        "anchors_n":          out.anchors_n,
        "sweep_n":            out.sweep_n,
        "benchmarks_n":       out.benchmarks_n,
        "policies_n":         out.policies_n,
        "ecg_modes_n":        out.ecg_modes_n,
        "graphs_n":           out.graphs_n,
        "reorder_variants_n": out.reorder_variants_n,
        "accuracy_pairs_n":   out.accuracy_pairs_n,
        "rules": {
            "C1": "DEFAULT_CACHE has canonical L1/L2/L3 size+ways and line-size anchors",
            "C2": "CACHE_SIZES_SWEEP is 12-point ascending power-of-2 grid 32KiB..64MiB",
            "C3": "BENCHMARKS partition: ITERATIVE∪TRAVERSAL no overlap; PREVIEW⊆BENCHMARKS",
            "C4": "POLICIES partition: BASELINE∪GRAPH_AWARE==ALL no overlap; PREVIEW⊆ALL",
            "C5": "ECG_MODES is canonical 4-token set (DBG_PRIMARY, POPT_PRIMARY, DBG_ONLY, ECG_EMBEDDED)",
            "C6": "EVAL_GRAPHS has required keys + canonical type + positive counts",
            "C7": "REORDER_VARIANTS uses recognized reorder flag + non-empty title-case unique labels",
            "C8": "ACCURACY_PAIRS uses recognized reorder + policy∈ALL + relation∈canonical",
        },
        "registry": {
            "cache_anchors":          CONFIG_CACHE_ANCHORS,
            "cache_sweep_n":          CONFIG_CACHE_SWEEP_N,
            "cache_sweep_range":      [CONFIG_CACHE_SWEEP_MIN, CONFIG_CACHE_SWEEP_MAX],
            "ecg_modes":              sorted(CONFIG_ECG_MODES),
            "canonical_graph_types":  sorted(CONFIG_CANONICAL_GRAPH_TYPES),
            "required_graph_keys":    sorted(CONFIG_REQUIRED_GRAPH_KEYS),
            "recognized_reorder":     sorted(CONFIG_RECOGNIZED_REORDER),
            "canonical_relations":    sorted(CONFIG_CANONICAL_RELATIONS),
        },
        "allow_lists": {
            "CONFIG_CACHE_EXTRA_ALLOW":    sorted(CONFIG_CACHE_EXTRA_ALLOW),
            "CONFIG_RELATION_EXTRA_ALLOW": sorted(CONFIG_RELATION_EXTRA_ALLOW),
        },
        "violations": [
            {"rule": v.rule, "where": v.where, "msg": v.msg}
            for v in out.violations
        ],
    }


def write_md(data: dict, path: Path) -> None:
    lines: list[str] = []
    lines.append("# Gate 269 — ECG config deep-lock registry\n")
    lines.append(
        "Locks `scripts/experiments/ecg/config.py` against silent drift in "
        "the cache-anchor numeric triplet, cache-sweep grid, benchmark "
        "partition, policy partition, ECG-mode set, EVAL_GRAPHS schema, "
        "reorder-flag vocabulary, and ACCURACY_PAIRS relation tokens. Where "
        "gate 256 covers profile NAMES, gate 269 covers profile CONTENT.\n"
    )
    lines.append(
        "registry: %d cache anchors; %d cache sweep points; %d benchmarks; "
        "%d policies; %d ECG modes; %d eval graphs; %d reorder variants; "
        "%d accuracy pairs.\n" % (
            data["anchors_n"], data["sweep_n"], data["benchmarks_n"],
            data["policies_n"], data["ecg_modes_n"], data["graphs_n"],
            data["reorder_variants_n"], data["accuracy_pairs_n"],
        )
    )
    lines.append("## Rules\n")
    for rid, txt in data["rules"].items():
        lines.append(f"- **{rid}** — {txt}")
    lines.append("")
    lines.append("## Allow-lists\n")
    for name, vals in data["allow_lists"].items():
        lines.append(f"- `{name}` = {vals}")
    lines.append("")
    if data["violations"]:
        lines.append("## ⛔ Violations\n")
        for v in data["violations"]:
            lines.append(f"- **{v['rule']}** at `{v['where']}`: {v['msg']}")
    else:
        lines.append("## ✅ No violations")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(data: dict, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "key", "value"])
        for name, val in CONFIG_CACHE_ANCHORS.items():
            w.writerow(["cache_anchor", name, val])
        w.writerow(["cache_sweep_n", "CACHE_SIZES_SWEEP", str(CONFIG_CACHE_SWEEP_N)])
        w.writerow(["cache_sweep_min", "CACHE_SIZES_SWEEP[0]", str(CONFIG_CACHE_SWEEP_MIN)])
        w.writerow(["cache_sweep_max", "CACHE_SIZES_SWEEP[-1]", str(CONFIG_CACHE_SWEEP_MAX)])
        for m in sorted(CONFIG_ECG_MODES):
            w.writerow(["ecg_mode", m, "canonical"])
        for t in sorted(CONFIG_CANONICAL_GRAPH_TYPES):
            w.writerow(["graph_type", t, "canonical"])
        for r in sorted(CONFIG_RECOGNIZED_REORDER):
            w.writerow(["reorder_flag", r, "recognized"])
        for rel in sorted(CONFIG_CANONICAL_RELATIONS):
            w.writerow(["relation", rel, "canonical"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json-out", type=Path, required=True)
    ap.add_argument("--md-out", type=Path, required=True)
    ap.add_argument("--csv-out", type=Path, required=True)
    args = ap.parse_args()

    data = audit()

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    write_md(data, args.md_out)
    write_csv(data, args.csv_out)

    n = len(data["violations"])
    print(f"[lit-faith-config-deep-lock] status={data['status']} "
          f"anchors={data['anchors_n']} sweep={data['sweep_n']} "
          f"benches={data['benchmarks_n']} policies={data['policies_n']} "
          f"modes={data['ecg_modes_n']} graphs={data['graphs_n']} "
          f"violations={n}")
    return 0 if n == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
