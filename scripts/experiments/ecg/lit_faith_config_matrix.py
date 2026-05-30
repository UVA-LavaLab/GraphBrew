"""Gate 263 — ECG configuration matrix registry.

Tenth in the vocabulary-lock series (252 SBATCH, 255 policy, 256
profile, 257 backend, 258 graph, 259 build, 260 CLI, 261
arm-catalog, 262 cross-tool aggregator schema, 263 config
matrix). Locks the central
``scripts/experiments/ecg/config.py`` module against silent
vocabulary drift relative to the canonical registries (policies,
graphs, kernels, L3 sizes) and against internal self-consistency
(every (reorder, policy) pair must use a policy that's actually
in ``ALL_POLICIES``, every ECG_MODE referenced in pairs is in
``ECG_MODES``, the L3 anchor in ``DEFAULT_CACHE`` matches a
canonical L3 tier).

Catches the silent-drift cases:

* a contributor adds ``"BIP"`` to ``BASELINE_POLICIES`` but
  forgets to register it in gate 255 ``CANONICAL_POLICY_NAMES``
  — downstream parsers reject BIP rows silently;
* a graph is renamed ``com-orkut`` → ``orkut`` in
  ``EVAL_GRAPHS`` without a corresponding alias entry in gate
  258 ``CANONICAL_GRAPHS`` — sweep runs land in
  ``results/orkut/`` but no plot script knows to look there;
* ``DEFAULT_CACHE["CACHE_L3_SIZE"]`` is bumped to ``"10485760"``
  (10 MB) which doesn't match any tier in gate 251
  ``CANONICAL_L3_TIERS`` — every L3-anchored plot picks an
  arbitrary closest tier and the "8 MB anchor" claim in the
  paper silently breaks;
* a benchmark is renamed ``cc_sv`` → ``conn-comp-sv`` in
  ``BENCHMARKS`` without a corresponding KERNEL_CL_CLASS edit
  in gate 260 — every sweep invokes a non-existent binary and
  records a 100% timeout count as "valid data";
* an ``("-o 99", "MAGIC")`` pair is added to
  ``REORDER_POLICY_PAIRS`` where ``MAGIC`` isn't in
  ``ALL_POLICIES`` (typo of ``MAGI`` or invented mode) — the
  bench runner emits zero rows but the pipeline reports OK.

7 rules C1-C7:

  C1: every policy in
      ``BASELINE_POLICIES ∪ GRAPH_AWARE_POLICIES ∪ PREVIEW_POLICIES``
      is in gate 255 ``CANONICAL_POLICY_NAMES``.
  C2: every ``EVAL_GRAPHS[i].name`` is the canonical name of some
      gate 258 ``CANONICAL_GRAPHS`` entry.
  C3: every ``BENCHMARKS`` entry is a key of gate 260
      ``KERNEL_CL_CLASS``.
  C4: ``DEFAULT_CACHE["CACHE_L3_SIZE"]`` parses to a byte count
      that equals the ``bytes`` field of exactly one tier in
      gate 251 ``CANONICAL_L3_TIERS``.
  C5: every ``CACHE_SIZES_SWEEP`` entry is a power-of-2 byte
      count; the sweep brackets the L3 anchor
      (``min ≤ DEFAULT_CACHE_L3 ≤ max``); entries are strictly
      increasing.
  C6: every ``(reorder, policy, ...)`` tuple across
      ``ACCURACY_PAIRS ∪ REORDER_POLICY_PAIRS`` uses a policy
      that is in ``ALL_POLICIES`` (which is the union of
      ``BASELINE_POLICIES`` and ``GRAPH_AWARE_POLICIES``).
  C7: every ``ECG_MODE`` value observed inside an env dict of
      ``ACCURACY_PAIRS`` is also a member of ``ECG_MODES``
      (no orphan modes); every ``ECG_MODES`` value is either
      a paper-pipeline-recognised mode (after ``ECG_X``
      translation) or documented here as "ECG-private,
      pre-paper".

Today: 8 policies (5 baseline + 3 graph-aware), 7 benchmarks, 6
eval graphs, 4 ECG modes, 12-entry cache-sweep, 18 (reorder,
policy) pairs across the two pair-tables; 0 violations.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PY = REPO_ROOT / "scripts" / "experiments" / "ecg" / "config.py"
ECG_DIR = REPO_ROOT / "scripts" / "experiments" / "ecg"


# --- Documented "ECG-private, pre-paper" mode allow-list ------------------
# Modes that are valid ECG runtime modes but are not currently emitted as
# a separate bar in the paper figures. They are still allowed in
# ECG_MODES (because they can show up in ACCURACY_PAIRS), but they are
# not required to appear in paper_pipeline.POLICY_ORDER.
ECG_PRIVATE_MODES = ("ECG_EMBEDDED",)


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_config():
    return _load("ecg_config_gate263", CONFIG_PY)


def _load_policy_registry():
    return _load("gate255", ECG_DIR / "lit_faith_policy_registry.py")


def _load_graph_registry():
    return _load("gate258", ECG_DIR / "lit_faith_graph_registry.py")


def _load_cli_registry():
    return _load("gate260", ECG_DIR / "lit_faith_cli_registry.py")


def _load_l3_registry():
    return _load("gate251", ECG_DIR / "lit_faith_l3_registry.py")


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def audit() -> dict[str, Any]:
    """Run all 7 rules and return the registry shape."""

    violations: list[dict[str, Any]] = []

    cfg = _load_config()
    pol = _load_policy_registry()
    gr = _load_graph_registry()
    cli = _load_cli_registry()
    l3 = _load_l3_registry()

    canonical_policies = set(pol.CANONICAL_POLICY_NAMES)
    canonical_graphs = {g.name for g in gr.CANONICAL_GRAPHS}
    canonical_kernels = set(cli.KERNEL_CL_CLASS.keys())
    canonical_l3_bytes = {tier_name: t["bytes"] for tier_name, t in l3.CANONICAL_L3_TIERS.items()}

    # --- C1: policy vocab ---------------------------------------------------
    all_policies_seen = (
        list(cfg.BASELINE_POLICIES)
        + list(cfg.GRAPH_AWARE_POLICIES)
        + list(cfg.PREVIEW_POLICIES)
    )
    n_policies_checked = 0
    for p in all_policies_seen:
        n_policies_checked += 1
        if p not in canonical_policies:
            violations.append(
                {"rule": "C1", "subject": p,
                 "reason": f"policy {p!r} not in CANONICAL_POLICY_NAMES "
                           f"({sorted(canonical_policies)})"}
            )

    # --- C2: graph vocab ----------------------------------------------------
    n_graphs_checked = 0
    for g in cfg.EVAL_GRAPHS:
        n_graphs_checked += 1
        name = g["name"]
        if name not in canonical_graphs:
            violations.append(
                {"rule": "C2", "subject": name,
                 "reason": f"EVAL_GRAPHS.name {name!r} not in CANONICAL_GRAPHS"}
            )

    # --- C3: kernel vocab ---------------------------------------------------
    n_kernels_checked = 0
    for b in cfg.BENCHMARKS:
        n_kernels_checked += 1
        if b not in canonical_kernels:
            violations.append(
                {"rule": "C3", "subject": b,
                 "reason": f"BENCHMARKS entry {b!r} not in KERNEL_CL_CLASS "
                           f"({sorted(canonical_kernels)})"}
            )

    # --- C4: L3 anchor ------------------------------------------------------
    try:
        default_l3 = int(cfg.DEFAULT_CACHE["CACHE_L3_SIZE"])
    except (KeyError, ValueError, TypeError) as exc:
        default_l3 = None
        violations.append(
            {"rule": "C4", "subject": "DEFAULT_CACHE",
             "reason": f"CACHE_L3_SIZE not a parseable int: {exc}"}
        )
    matched_tier: str | None = None
    if default_l3 is not None:
        for tier_name, b in canonical_l3_bytes.items():
            if b == default_l3:
                matched_tier = tier_name
                break
        if matched_tier is None:
            violations.append(
                {"rule": "C4", "subject": str(default_l3),
                 "reason": f"DEFAULT_CACHE.CACHE_L3_SIZE={default_l3} bytes "
                           f"matches no CANONICAL_L3_TIERS "
                           f"(canonical bytes={sorted(canonical_l3_bytes.values())})"}
            )

    # --- C5: cache sweep shape ----------------------------------------------
    sweep = list(cfg.CACHE_SIZES_SWEEP)
    n_sweep = len(sweep)
    for i, s in enumerate(sweep):
        if not _is_power_of_two(s):
            violations.append(
                {"rule": "C5", "subject": str(s),
                 "reason": f"CACHE_SIZES_SWEEP[{i}]={s} is not a power-of-2"}
            )
    for i in range(1, n_sweep):
        if sweep[i] <= sweep[i - 1]:
            violations.append(
                {"rule": "C5", "subject": f"{sweep[i-1]}→{sweep[i]}",
                 "reason": f"CACHE_SIZES_SWEEP not strictly increasing at index {i}"}
            )
    if default_l3 is not None and sweep:
        if not (sweep[0] <= default_l3 <= sweep[-1]):
            violations.append(
                {"rule": "C5", "subject": str(default_l3),
                 "reason": (f"DEFAULT_CACHE.CACHE_L3_SIZE={default_l3} not bracketed "
                            f"by CACHE_SIZES_SWEEP [{sweep[0]}..{sweep[-1]}]")}
            )

    # --- C6: pair policies --------------------------------------------------
    all_policies = set(cfg.ALL_POLICIES)
    n_pairs_checked = 0
    for src, table in (("ACCURACY_PAIRS", cfg.ACCURACY_PAIRS),
                       ("REORDER_POLICY_PAIRS", cfg.REORDER_POLICY_PAIRS)):
        for i, pair in enumerate(table):
            n_pairs_checked += 1
            # both tables have policy at index 1
            if len(pair) < 2:
                violations.append(
                    {"rule": "C6", "subject": f"{src}[{i}]",
                     "reason": f"pair tuple too short: {pair}"}
                )
                continue
            policy = pair[1]
            if policy not in all_policies:
                violations.append(
                    {"rule": "C6", "subject": f"{src}[{i}].policy={policy}",
                     "reason": f"policy {policy!r} not in ALL_POLICIES ({sorted(all_policies)})"}
                )

    # --- C7: ECG mode exhaustiveness ---------------------------------------
    declared_modes = set(cfg.ECG_MODES)
    observed_modes: set[str] = set()
    for pair in cfg.ACCURACY_PAIRS:
        if len(pair) >= 3 and isinstance(pair[2], dict):
            mode = pair[2].get("ECG_MODE")
            if mode is not None:
                observed_modes.add(mode)
    for m in observed_modes:
        if m not in declared_modes:
            violations.append(
                {"rule": "C7", "subject": m,
                 "reason": f"ECG_MODE {m!r} observed in ACCURACY_PAIRS but "
                           f"not declared in ECG_MODES ({sorted(declared_modes)})"}
            )
    # Every declared mode is either paper-shipping (POLICY_ORDER recognises
    # ``ECG_<mode>``) or in the private allow-list.
    try:
        paper_mod = _load("paper_pipeline_for_gate263",
                          ECG_DIR / "paper_pipeline.py")
        paper_policy_order = set(getattr(paper_mod, "POLICY_ORDER", ()))
    except Exception:
        paper_policy_order = set()
    private_modes = set(ECG_PRIVATE_MODES)
    for m in declared_modes:
        paper_form = f"ECG_{m}"
        if paper_form in paper_policy_order:
            continue
        if m in private_modes:
            continue
        violations.append(
            {"rule": "C7", "subject": m,
             "reason": (f"ECG_MODE {m!r} declared but not in paper_pipeline.POLICY_ORDER "
                        f"as {paper_form!r} and not in ECG_PRIVATE_MODES "
                        f"({sorted(private_modes)})")}
        )

    return {
        "status": "active",
        "n_policies_checked": n_policies_checked,
        "n_graphs_checked": n_graphs_checked,
        "n_kernels_checked": n_kernels_checked,
        "n_sweep_entries": n_sweep,
        "n_pairs_checked": n_pairs_checked,
        "n_ecg_modes": len(declared_modes),
        "default_l3_bytes": default_l3,
        "default_l3_tier": matched_tier,
        "baseline_policies": list(cfg.BASELINE_POLICIES),
        "graph_aware_policies": list(cfg.GRAPH_AWARE_POLICIES),
        "preview_policies": list(cfg.PREVIEW_POLICIES),
        "all_policies": list(cfg.ALL_POLICIES),
        "ecg_modes": list(cfg.ECG_MODES),
        "ecg_private_modes": list(ECG_PRIVATE_MODES),
        "benchmarks": list(cfg.BENCHMARKS),
        "eval_graphs": [g["name"] for g in cfg.EVAL_GRAPHS],
        "cache_sweep_bytes": sweep,
        "observed_ecg_modes": sorted(observed_modes),
        "rules": {
            "C1": ("every BASELINE_POLICIES + GRAPH_AWARE_POLICIES + PREVIEW_POLICIES entry "
                   "is in gate 255 CANONICAL_POLICY_NAMES"),
            "C2": "every EVAL_GRAPHS.name is in gate 258 CANONICAL_GRAPHS",
            "C3": "every BENCHMARKS entry is a key of gate 260 KERNEL_CL_CLASS",
            "C4": ("DEFAULT_CACHE.CACHE_L3_SIZE parses to bytes matching exactly one tier "
                   "in gate 251 CANONICAL_L3_TIERS"),
            "C5": ("every CACHE_SIZES_SWEEP entry is a power-of-2; the sweep brackets the "
                   "L3 anchor (min ≤ DEFAULT_L3 ≤ max); strictly increasing"),
            "C6": ("every (reorder, policy, ...) pair across ACCURACY_PAIRS + "
                   "REORDER_POLICY_PAIRS uses a policy in ALL_POLICIES"),
            "C7": ("every ECG_MODE observed in ACCURACY_PAIRS env dicts is in ECG_MODES; "
                   "every ECG_MODES entry is paper-pipeline-recognised as ECG_<mode> "
                   "OR in ECG_PRIVATE_MODES"),
        },
        "violations": violations,
    }


def _emit_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _emit_md(data: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# Gate 263 — ECG configuration matrix registry")
    lines.append("")
    lines.append(f"Status: **{data['status']}**")
    lines.append("")
    lines.append("## Totals")
    lines.append("")
    for k in (
        "n_policies_checked", "n_graphs_checked", "n_kernels_checked",
        "n_sweep_entries", "n_pairs_checked", "n_ecg_modes",
    ):
        lines.append(f"- {k}: {data[k]}")
    lines.append(f"- default_l3_bytes: {data['default_l3_bytes']} "
                 f"(tier: {data['default_l3_tier']})")
    lines.append("")
    lines.append("## Rules")
    lines.append("")
    for rid, desc in data["rules"].items():
        lines.append(f"- **{rid}** — {desc}")
    lines.append("")
    lines.append("## Policy vocab")
    lines.append("")
    lines.append(f"- BASELINE_POLICIES: `{data['baseline_policies']}`")
    lines.append(f"- GRAPH_AWARE_POLICIES: `{data['graph_aware_policies']}`")
    lines.append(f"- PREVIEW_POLICIES: `{data['preview_policies']}`")
    lines.append(f"- ALL_POLICIES: `{data['all_policies']}`")
    lines.append("")
    lines.append("## ECG modes")
    lines.append("")
    lines.append(f"- ECG_MODES (declared): `{data['ecg_modes']}`")
    lines.append(f"- ECG_PRIVATE_MODES: `{data['ecg_private_modes']}`")
    lines.append(f"- observed in ACCURACY_PAIRS: `{data['observed_ecg_modes']}`")
    lines.append("")
    lines.append("## Benchmarks + graphs")
    lines.append("")
    lines.append(f"- BENCHMARKS: `{data['benchmarks']}`")
    lines.append(f"- EVAL_GRAPHS: `{data['eval_graphs']}`")
    lines.append("")
    lines.append("## Cache sweep")
    lines.append("")
    lines.append("| index | bytes | size |")
    lines.append("|---:|---:|---|")
    for i, b in enumerate(data["cache_sweep_bytes"]):
        if b >= 1024 * 1024:
            label = f"{b // (1024 * 1024)}MB"
        elif b >= 1024:
            label = f"{b // 1024}kB"
        else:
            label = f"{b}B"
        lines.append(f"| {i} | {b} | {label} |")
    lines.append("")
    if data["violations"]:
        lines.append("## Violations")
        lines.append("")
        for v in data["violations"]:
            lines.append(f"- **{v['rule']}** `{v['subject']}` — {v['reason']}")
    else:
        lines.append("## Violations")
        lines.append("")
        lines.append("None.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _emit_csv(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("kind", "name", "extra"))
        for p in data["baseline_policies"]:
            w.writerow(("baseline_policy", p, ""))
        for p in data["graph_aware_policies"]:
            w.writerow(("graph_aware_policy", p, ""))
        for b in data["benchmarks"]:
            w.writerow(("benchmark", b, ""))
        for g in data["eval_graphs"]:
            w.writerow(("eval_graph", g, ""))
        for m in data["ecg_modes"]:
            w.writerow(("ecg_mode", m, ""))
        for i, b in enumerate(data["cache_sweep_bytes"]):
            w.writerow(("sweep_bytes", str(i), str(b)))
        for v in data["violations"]:
            w.writerow(("violation", v["subject"], f"{v['rule']}: {v['reason']}"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--md-out", type=Path)
    ap.add_argument("--csv-out", type=Path)
    args = ap.parse_args(argv)
    data = audit()
    if args.json_out:
        _emit_json(data, args.json_out)
    if args.md_out:
        _emit_md(data, args.md_out)
    if args.csv_out:
        _emit_csv(data, args.csv_out)
    print(
        f"[lit-faith-config-matrix] status={data['status']} "
        f"policies={data['n_policies_checked']} "
        f"graphs={data['n_graphs_checked']} "
        f"kernels={data['n_kernels_checked']} "
        f"sweep={data['n_sweep_entries']} "
        f"pairs={data['n_pairs_checked']} "
        f"violations={len(data['violations'])}"
    )
    return 1 if data["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
