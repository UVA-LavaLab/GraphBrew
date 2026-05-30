"""ECG substrate-parity audit on gem5 (gate 239 — ECG-Gem5-Parity).

This is the gem5-backend sibling of gate 238 (cache_sim ECG-Parity). It
locks the **POPT-arm faithfulness** of the ECG substrate under cycle-
accurate gem5 timing — i.e. that ``ECG_POPT_PRIMARY`` and stock ``POPT``
agree on L3 miss-rate to within tolerance when the same workload is
replayed through gem5's full memory hierarchy.

Scope today (gem5 POPT-arm only):
  G1 — Roster completeness on gem5. Every benchmark in ``BENCH_FLOOR``
       must have every required policy ∈ ``REQUIRED_POLICIES`` present
       with ``status == "ok"`` and at least one observation per (section, L3).
  G2 — POPT substrate parity in gem5. For every matched (section, L3)
       pair, ``|miss_rate(ECG_POPT_PRIMARY) - miss_rate(POPT)| <=
       EPS_POPT_PARITY``. We deliberately keep the tolerance looser than
       cache_sim's 5e-4 because gem5 introduces real timing noise
       (warmup, MSHR interactions, OoO scheduling); the largest drift
       observed in the bracket sweep was 1.09e-3 so ε=2e-3 gives 2×
       safety.
  G3 — Backend identity. Every per-observation row must have
       ``backend == "gem5"`` and ``simulator == "gem5"``. This prevents
       silent ingestion of cache_sim data into the gem5 gate.
  G4 — Sim-tick non-zero floor. Every row must have ``sim_ticks > 0``
       and ``ipc > 0``. A zero would mean the gem5 simulation didn't
       actually execute the ROI.
  G5 — Baseline non-zero floor. LRU must have strictly positive
       ``l3_accesses`` and ``l3_misses`` on every (section, L3) pair —
       the benchmark must actually touch the L3.
  G6 — L3 hierarchy sanity. ``l3_misses <= l3_accesses`` and miss-rate
       ∈ [0, 1] on every row.
  G7 — Section coverage floor. ≥ 2 distinct sections present (the
       bracket sweep emits section=1 cold-start + section=2 re-warmed,
       so requiring both catches truncated runs).

Out of scope (explicitly queued):
  - DBG arm (``ECG_DBG_ONLY`` ≡ ``GRASP``) in gem5. No ECG_DBG gem5 run
    is available yet; the grasp-gem5-sweep has GRASP rows but not
    ECG_DBG. Queued for future expansion of this gate (or a sibling
    gate) when a matched-proof DBG gem5 run lands.
  - PFX activation in gem5. The bracket sweep used
    ``ecg_pfx_delivery=explicit-hint`` and ``prefetcher=none``; the
    prefetcher leg itself was never exercised. Queued.
  - DROPLET comparison. ``droplet_*`` columns are config metadata only;
    queued for gate 241+ when a DROPLET-active sweep lands.

The audit consumes a curated postfix
(``wiki/data/ecg_gem5_parity_postfix.json``) which is extracted
verbatim from the bracket sweep's ``roi_matrix.csv``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

REQUIRED_POLICIES = {"LRU", "POPT", "ECG_POPT_PRIMARY"}
BENCH_FLOOR = {"pr"}                # gem5 bracket today: PR only
BASELINE_POLICIES = {"LRU"}         # rows that must always have data
EPS_POPT_PARITY = 0.002             # 2e-3 ≈ 0.2 pp; 2× observed max 1.09e-3
SECTION_FLOOR = 2                   # cold + re-warmed
SIM_TICK_FLOOR = 1                  # any nonzero tick count
IPC_FLOOR = 0.0                     # must be strictly > 0


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return abs(a - b)


def audit(postfix: dict[str, Any]) -> dict[str, Any]:
    rows = postfix.get("per_observation", [])
    violations: list[dict[str, Any]] = []

    # Group by (benchmark, section, l3_size) -> {policy_label: row}
    by_cell: dict[tuple, dict[str, dict[str, Any]]] = defaultdict(dict)
    sections_present: set[int] = set()
    backends_present: set[str] = set()
    benches_present: set[str] = set()
    for r in rows:
        bench = r.get("benchmark")
        sect = r.get("section")
        l3 = r.get("l3_size")
        pol = r.get("policy_label")
        backends_present.add(r.get("backend", "unknown"))
        if bench:
            benches_present.add(bench)
        if sect is not None:
            sections_present.add(int(sect))
        if bench and sect is not None and l3 and pol:
            by_cell[(bench, int(sect), l3)][pol] = r

    # G1 — Roster completeness per cell
    for cell, table in sorted(by_cell.items()):
        bench, sect, l3 = cell
        if bench not in BENCH_FLOOR:
            continue
        for pol in sorted(REQUIRED_POLICIES):
            row = table.get(pol)
            if row is None:
                violations.append({
                    "rule": "G1-missing-policy", "benchmark": bench,
                    "section": sect, "l3_size": l3, "policy": pol,
                })
            elif row.get("status") != "ok":
                violations.append({
                    "rule": "G1-non-ok-status", "benchmark": bench,
                    "section": sect, "l3_size": l3, "policy": pol,
                    "status": row.get("status"),
                })

    # G2 — POPT substrate parity per cell
    parity_popt: list[dict[str, Any]] = []
    for cell, table in sorted(by_cell.items()):
        bench, sect, l3 = cell
        popt = table.get("POPT", {}).get("l3_miss_rate")
        ecg = table.get("ECG_POPT_PRIMARY", {}).get("l3_miss_rate")
        delta = _delta(popt, ecg)
        parity_popt.append({
            "benchmark": bench, "section": sect, "l3_size": l3,
            "popt": popt, "ecg_popt_primary": ecg, "abs_delta": delta,
        })
        if delta is None or delta > EPS_POPT_PARITY:
            violations.append({
                "rule": "G2-popt-parity-drift", "benchmark": bench,
                "section": sect, "l3_size": l3,
                "popt": popt, "ecg_popt_primary": ecg,
                "abs_delta": delta, "epsilon": EPS_POPT_PARITY,
            })

    # G3 — Backend identity
    for r in rows:
        b = r.get("backend")
        s = r.get("simulator")
        if b != "gem5" or s != "gem5":
            violations.append({
                "rule": "G3-backend-mismatch", "backend": b, "simulator": s,
                "policy": r.get("policy_label"), "section": r.get("section"),
            })

    # G4 — Sim-tick + IPC floors
    for r in rows:
        ticks = r.get("sim_ticks")
        ipc = r.get("ipc")
        if ticks is None or ticks < SIM_TICK_FLOOR:
            violations.append({
                "rule": "G4-sim-tick-zero", "policy": r.get("policy_label"),
                "section": r.get("section"), "l3_size": r.get("l3_size"),
                "sim_ticks": ticks,
            })
        if ipc is None or ipc <= IPC_FLOOR:
            violations.append({
                "rule": "G4-ipc-zero", "policy": r.get("policy_label"),
                "section": r.get("section"), "l3_size": r.get("l3_size"),
                "ipc": ipc,
            })

    # G5 — Baseline non-zero floor on LRU
    for cell, table in sorted(by_cell.items()):
        bench, sect, l3 = cell
        for pol in BASELINE_POLICIES:
            row = table.get(pol, {})
            if not row:
                continue
            accs = row.get("l3_accesses") or 0
            misses = row.get("l3_misses") or 0
            if accs <= 0 or misses <= 0:
                violations.append({
                    "rule": "G5-baseline-zero", "benchmark": bench,
                    "section": sect, "l3_size": l3, "policy": pol,
                    "l3_accesses": accs, "l3_misses": misses,
                })

    # G6 — L3 hierarchy sanity
    for r in rows:
        misses = r.get("l3_misses")
        accs = r.get("l3_accesses")
        mr = r.get("l3_miss_rate")
        if misses is not None and accs is not None and misses > accs:
            violations.append({
                "rule": "G6-misses-exceed-accesses",
                "policy": r.get("policy_label"), "section": r.get("section"),
                "l3_misses": misses, "l3_accesses": accs,
            })
        if mr is not None and (mr < 0.0 or mr > 1.0):
            violations.append({
                "rule": "G6-miss-rate-out-of-range",
                "policy": r.get("policy_label"), "section": r.get("section"),
                "l3_miss_rate": mr,
            })

    # G7 — Section coverage floor
    if len(sections_present) < SECTION_FLOOR:
        violations.append({
            "rule": "G7-section-coverage-floor",
            "sections_present": sorted(sections_present),
            "floor": SECTION_FLOOR,
        })

    return {
        "rules": {
            "G1": "every required policy present with status=ok per (benchmark, section, l3_size) cell",
            "G2": f"|miss_rate(ECG_POPT_PRIMARY) - miss_rate(POPT)| <= {EPS_POPT_PARITY}",
            "G3": "every row has backend=gem5 AND simulator=gem5 (no silent cache_sim ingestion)",
            "G4": f"sim_ticks >= {SIM_TICK_FLOOR} AND ipc > {IPC_FLOOR} on every row",
            "G5": "LRU baseline has strictly positive l3_accesses and l3_misses on every cell",
            "G6": "l3_misses <= l3_accesses and l3_miss_rate in [0,1] on every row",
            "G7": f"distinct sections present >= {SECTION_FLOOR} (cold + re-warmed)",
        },
        "constants": {
            "eps_popt_parity":     EPS_POPT_PARITY,
            "section_floor":       SECTION_FLOOR,
            "sim_tick_floor":      SIM_TICK_FLOOR,
            "ipc_floor":           IPC_FLOOR,
            "required_policies":   sorted(REQUIRED_POLICIES),
            "baseline_policies":   sorted(BASELINE_POLICIES),
            "bench_floor":         sorted(BENCH_FLOOR),
        },
        "totals": {
            "rows":              len(rows),
            "cells":             len(by_cell),
            "benchmarks":        sorted(benches_present),
            "backends":          sorted(backends_present),
            "sections":          sorted(sections_present),
            "policies_present":  sorted({r.get("policy_label") for r in rows if r.get("policy_label")}),
        },
        "parity_popt": parity_popt,
        "violations":  violations,
    }


def render_markdown(audit_obj: dict[str, Any]) -> str:
    t = audit_obj["totals"]
    c = audit_obj["constants"]
    lines = [
        "# ECG substrate-parity audit (gem5)",
        "",
        "Gate 239 — ECG-Gem5-Parity. Locks the POPT-arm faithfulness of",
        "the ECG substrate under cycle-accurate gem5 timing. Sibling to",
        "gate 238 (cache_sim ECG-Parity). DBG arm + PFX activation are",
        "explicitly out of scope today (see generator docstring).",
        "",
        "## Rules",
    ]
    for rid, desc in audit_obj["rules"].items():
        lines.append(f"- **{rid}** — {desc}")

    lines.extend([
        "",
        "## Constants",
        f"- ε(POPT parity): `{c['eps_popt_parity']}`",
        f"- section floor: `{c['section_floor']}`",
        f"- sim-tick floor: `{c['sim_tick_floor']}`",
        f"- IPC floor: `> {c['ipc_floor']}`",
        f"- required policies: `{', '.join(c['required_policies'])}`",
        f"- baseline policies: `{', '.join(c['baseline_policies'])}`",
        "",
        "## Totals",
        f"- observations: **{t['rows']}**",
        f"- cells (benchmark × section × L3): **{t['cells']}**",
        f"- benchmarks: `{', '.join(t['benchmarks'])}`",
        f"- backends: `{', '.join(t['backends'])}`",
        f"- sections: `{', '.join(map(str, t['sections']))}`",
        f"- policies present: `{', '.join(t['policies_present'])}`",
        "",
        "## POPT parity (gem5)",
        "",
        "| benchmark | section | L3 | POPT | ECG_POPT_PRIMARY | |Δ| |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ])
    for row in audit_obj["parity_popt"]:
        lines.append(
            f"| {row['benchmark']} | {row['section']} | {row['l3_size']} "
            f"| {row['popt']} | {row['ecg_popt_primary']} | {row['abs_delta']} |"
        )

    lines.extend(["", "## Violations", ""])
    if not audit_obj["violations"]:
        lines.append("_None._")
    else:
        lines.append("| rule | benchmark | section | L3 | detail |")
        lines.append("| --- | --- | ---: | --- | --- |")
        for v in audit_obj["violations"]:
            detail = {k: v[k] for k in v
                      if k not in ("rule", "benchmark", "section", "l3_size")}
            lines.append(
                f"| {v.get('rule','')} | {v.get('benchmark','')} | "
                f"{v.get('section','')} | {v.get('l3_size','')} | {detail} |"
            )

    lines.append("")
    return "\n".join(lines)


def render_csv(audit_obj: dict[str, Any]) -> str:
    head = "benchmark,section,l3_size,popt,ecg_popt_primary,abs_delta\n"
    body = "\n".join(
        f"{r['benchmark']},{r['section']},{r['l3_size']},"
        f"{r['popt']},{r['ecg_popt_primary']},{r['abs_delta']}"
        for r in audit_obj["parity_popt"]
    )
    return head + body + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--postfix-json", type=Path, required=True)
    ap.add_argument("--json-out",     type=Path, required=True)
    ap.add_argument("--md-out",       type=Path, required=True)
    ap.add_argument("--csv-out",      type=Path, required=True)
    args = ap.parse_args()

    postfix = json.loads(args.postfix_json.read_text())
    out = audit(postfix)
    args.json_out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(render_markdown(out))
    args.csv_out.write_text(render_csv(out))
    print(
        f"[lit-faith-ecg-gem5-parity] rows={out['totals']['rows']} "
        f"cells={out['totals']['cells']} "
        f"benches={out['totals']['benchmarks']} "
        f"sections={out['totals']['sections']} "
        f"violations={len(out['violations'])}"
    )
    return 1 if out["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
