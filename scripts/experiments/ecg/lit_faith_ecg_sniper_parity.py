"""ECG substrate-parity audit on Sniper (gate 240 — ECG-Sniper-Parity).

This is the Sniper-backend sibling of gate 238 (cache_sim ECG-Parity) and
gate 239 (gem5 ECG-Parity). Today it operates in **SCAFFOLD/DEFERRED**
mode: no matched-proof Sniper ECG run is available yet. The audit logic
is implemented end-to-end and will activate the moment a Sniper ECG
sweep populates ``per_observation`` in the curated postfix
(``wiki/data/ecg_sniper_parity_postfix.json``).

Scope when activated (Sniper backend, POPT+DBG arms):
  G1  — Roster completeness on Sniper. Every benchmark in the
        observed bench-floor must have every policy ∈
        ``REQUIRED_POPT_POLICIES`` (LRU/POPT/ECG_POPT_PRIMARY) present
        with ``status == "ok"`` per (section, L3) cell.
  G1b — DBG roster, OPTIONAL. When any ``ECG_DBG_ONLY`` row is present,
        the cell must also carry a ``GRASP`` row (so a DBG parity check
        can be made). If no DBG rows are present, this rule no-ops.
  G2  — POPT substrate parity on Sniper. For every matched (section,
        L3) pair, ``|miss_rate(ECG_POPT_PRIMARY) - miss_rate(POPT)|
        <= EPS_POPT_PARITY``. Tolerance matches gem5's 2e-3 (Sniper
        also has timing/MSHR noise; we treat it as gem5-class for ε
        until empirical evidence forces a tighter or looser value).
  G2b — DBG substrate parity, OPTIONAL. When both ECG_DBG_ONLY and
        GRASP are present on a cell, ``|miss_rate(ECG_DBG_ONLY) -
        miss_rate(GRASP)| <= EPS_DBG_PARITY``. No-ops when DBG arm is
        absent (i.e. matched-proof DBG sweep not yet curated).
  G3  — Backend identity. Every row must have ``backend == "sniper"``
        and ``simulator == "sniper"``. Prevents silent ingestion of
        cache_sim or gem5 data into the Sniper gate.
  G4  — IPC + instructions floors. ``ipc > 0`` and ``instructions > 0``
        on every row (ROI was actually executed).
  G5  — LRU baseline non-zero floor. ``l3_accesses > 0`` and
        ``l3_misses > 0`` on every LRU cell.
  G6  — L3 hierarchy sanity. ``l3_misses <= l3_accesses`` and
        ``l3_miss_rate in [0, 1]`` on every row.
  G7  — Observation floor. ``len(per_observation) >=
        EXPECTED_MINIMUM_OBSERVATIONS`` from the postfix declaration.

Deferred-stub mode (today): ``per_observation == []`` AND
``status == "deferred"``. In this mode, the audit emits zero
violations but stamps ``audit.status = "deferred"`` and echoes the
defer reason and the expected source pattern from the postfix. This
preserves the green-gate invariant *without* fabricating ECG numbers
that don't yet exist. The pytest gate asserts the deferred shape and
locks the schema so that whoever curates the real fixture cannot
silently drop required keys.

Out of scope when activated (queued):
  - PFX activation in Sniper — needs ``prefetcher=ecg_pfx`` enabled.
    The current Sniper sweeps run with prefetching off.
  - DROPLET comparison — needs a DROPLET-active Sniper sweep
    (queued for gate 241+).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

REQUIRED_POPT_POLICIES = {"LRU", "POPT", "ECG_POPT_PRIMARY"}
OPTIONAL_DBG_POLICIES = {"GRASP", "ECG_DBG_ONLY"}
BASELINE_POLICIES = {"LRU"}
EPS_POPT_PARITY = 0.002             # mirror gem5 tolerance (2e-3, ~0.2 pp)
EPS_DBG_PARITY = 0.002              # mirror gem5 tolerance for DBG arm
IPC_FLOOR = 0.0                     # strictly > 0
INSTR_FLOOR = 1                     # at least 1 retired instruction


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return abs(a - b)


def _build_cells(rows: list[dict[str, Any]]):
    by_cell: dict[tuple, dict[str, dict[str, Any]]] = defaultdict(dict)
    sections: set[int] = set()
    backends: set[str] = set()
    benches: set[str] = set()
    for r in rows:
        bench = r.get("benchmark")
        sect = r.get("section")
        l3 = r.get("l3_size")
        pol = r.get("policy_label")
        backends.add(r.get("backend", "unknown"))
        if bench:
            benches.add(bench)
        if sect is not None:
            sections.add(int(sect))
        if bench and sect is not None and l3 and pol:
            by_cell[(bench, int(sect), l3)][pol] = r
    return by_cell, sections, backends, benches


def audit(postfix: dict[str, Any]) -> dict[str, Any]:
    rows = postfix.get("per_observation", [])
    status_decl = postfix.get("status", "active")
    deferred = (status_decl == "deferred") and not rows

    if deferred:
        return {
            "status": "deferred",
            "defer_reason": postfix.get("defer_reason", ""),
            "expected_source_pattern": postfix.get("expected_source_pattern", ""),
            "expected_minimum_observations":
                postfix.get("expected_minimum_observations", 0),
            "expected_required_policies":
                postfix.get("expected_required_policies",
                            sorted(REQUIRED_POPT_POLICIES)),
            "expected_required_dbg_policies":
                postfix.get("expected_required_dbg_policies",
                            sorted(OPTIONAL_DBG_POLICIES)),
            "rules": _rule_descriptions(),
            "constants": _constants(),
            "totals": {
                "rows": 0, "cells": 0,
                "benchmarks": [], "backends": [],
                "sections": [], "policies_present": [],
            },
            "parity_popt": [],
            "parity_dbg": [],
            "violations": [],
        }

    by_cell, sections, backends, benches = _build_cells(rows)
    bench_floor = set(benches)            # observed benches become the floor
    violations: list[dict[str, Any]] = []

    expected_min = int(postfix.get("expected_minimum_observations", 0))
    if expected_min and len(rows) < expected_min:
        violations.append({
            "rule": "G7-observation-floor",
            "rows": len(rows), "expected_minimum": expected_min,
        })

    for cell, table in sorted(by_cell.items()):
        bench, sect, l3 = cell
        if bench not in bench_floor:
            continue
        for pol in sorted(REQUIRED_POPT_POLICIES):
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
        if "ECG_DBG_ONLY" in table and "GRASP" not in table:
            violations.append({
                "rule": "G1b-missing-grasp-for-dbg",
                "benchmark": bench, "section": sect, "l3_size": l3,
            })

    parity_popt: list[dict[str, Any]] = []
    parity_dbg: list[dict[str, Any]] = []
    for cell, table in sorted(by_cell.items()):
        bench, sect, l3 = cell
        popt = table.get("POPT", {}).get("l3_miss_rate")
        ecg = table.get("ECG_POPT_PRIMARY", {}).get("l3_miss_rate")
        d = _delta(popt, ecg)
        parity_popt.append({
            "benchmark": bench, "section": sect, "l3_size": l3,
            "popt": popt, "ecg_popt_primary": ecg, "abs_delta": d,
        })
        if d is None or d > EPS_POPT_PARITY:
            violations.append({
                "rule": "G2-popt-parity-drift", "benchmark": bench,
                "section": sect, "l3_size": l3,
                "popt": popt, "ecg_popt_primary": ecg,
                "abs_delta": d, "epsilon": EPS_POPT_PARITY,
            })

        grasp = table.get("GRASP", {}).get("l3_miss_rate")
        dbg = table.get("ECG_DBG_ONLY", {}).get("l3_miss_rate")
        if grasp is not None and dbg is not None:
            dd = _delta(grasp, dbg)
            parity_dbg.append({
                "benchmark": bench, "section": sect, "l3_size": l3,
                "grasp": grasp, "ecg_dbg_only": dbg, "abs_delta": dd,
            })
            if dd is None or dd > EPS_DBG_PARITY:
                violations.append({
                    "rule": "G2b-dbg-parity-drift", "benchmark": bench,
                    "section": sect, "l3_size": l3,
                    "grasp": grasp, "ecg_dbg_only": dbg,
                    "abs_delta": dd, "epsilon": EPS_DBG_PARITY,
                })

    for r in rows:
        b = r.get("backend")
        s = r.get("simulator")
        if b != "sniper" or s != "sniper":
            violations.append({
                "rule": "G3-backend-mismatch", "backend": b, "simulator": s,
                "policy": r.get("policy_label"), "section": r.get("section"),
            })
        ipc = r.get("ipc")
        instr = r.get("instructions")
        if ipc is None or ipc <= IPC_FLOOR:
            violations.append({
                "rule": "G4-ipc-zero", "policy": r.get("policy_label"),
                "section": r.get("section"), "l3_size": r.get("l3_size"),
                "ipc": ipc,
            })
        if instr is None or instr < INSTR_FLOOR:
            violations.append({
                "rule": "G4-instructions-zero",
                "policy": r.get("policy_label"),
                "section": r.get("section"), "l3_size": r.get("l3_size"),
                "instructions": instr,
            })
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

    return {
        "status": "active",
        "rules": _rule_descriptions(),
        "constants": _constants(),
        "totals": {
            "rows": len(rows),
            "cells": len(by_cell),
            "benchmarks": sorted(benches),
            "backends": sorted(backends),
            "sections": sorted(sections),
            "policies_present":
                sorted({r.get("policy_label") for r in rows
                        if r.get("policy_label")}),
        },
        "parity_popt": parity_popt,
        "parity_dbg": parity_dbg,
        "violations": violations,
    }


def _rule_descriptions() -> dict[str, str]:
    return {
        "G1": "every required POPT-arm policy present with status=ok per (benchmark, section, l3_size) cell",
        "G1b": "if ECG_DBG_ONLY present on a cell, GRASP must also be present",
        "G2": f"|miss_rate(ECG_POPT_PRIMARY) - miss_rate(POPT)| <= {EPS_POPT_PARITY}",
        "G2b": f"if both GRASP and ECG_DBG_ONLY present, |miss_rate(ECG_DBG_ONLY) - miss_rate(GRASP)| <= {EPS_DBG_PARITY}",
        "G3": "every row has backend=sniper AND simulator=sniper (no silent cache_sim/gem5 ingestion)",
        "G4": f"ipc > {IPC_FLOOR} AND instructions >= {INSTR_FLOOR} on every row",
        "G5": "LRU baseline has strictly positive l3_accesses and l3_misses on every cell",
        "G6": "l3_misses <= l3_accesses and l3_miss_rate in [0,1] on every row",
        "G7": "len(per_observation) >= postfix.expected_minimum_observations",
    }


def _constants() -> dict[str, Any]:
    return {
        "eps_popt_parity":           EPS_POPT_PARITY,
        "eps_dbg_parity":            EPS_DBG_PARITY,
        "ipc_floor":                 IPC_FLOOR,
        "instructions_floor":        INSTR_FLOOR,
        "required_popt_policies":    sorted(REQUIRED_POPT_POLICIES),
        "optional_dbg_policies":     sorted(OPTIONAL_DBG_POLICIES),
        "baseline_policies":         sorted(BASELINE_POLICIES),
    }


def render_markdown(audit_obj: dict[str, Any]) -> str:
    t = audit_obj["totals"]
    c = audit_obj["constants"]
    lines = [
        "# ECG substrate-parity audit (Sniper)",
        "",
        "Gate 240 — ECG-Sniper-Parity. Sibling to gate 238 (cache_sim)",
        "and gate 239 (gem5). Locks the POPT-arm (and optionally DBG-arm)",
        "substrate faithfulness under Sniper. PFX activation and DROPLET",
        "comparison remain out of scope here.",
        "",
        f"**Status:** `{audit_obj.get('status','active')}`",
    ]
    if audit_obj.get("status") == "deferred":
        lines.extend([
            "",
            "## Deferred",
            "",
            f"- defer reason: {audit_obj.get('defer_reason','')}",
            f"- expected source pattern: `{audit_obj.get('expected_source_pattern','')}`",
            f"- expected minimum observations: {audit_obj.get('expected_minimum_observations','')}",
            f"- expected POPT-arm policies: `{', '.join(audit_obj.get('expected_required_policies', []))}`",
            f"- expected DBG-arm policies: `{', '.join(audit_obj.get('expected_required_dbg_policies', []))}`",
        ])

    lines.extend(["", "## Rules"])
    for rid, desc in audit_obj["rules"].items():
        lines.append(f"- **{rid}** — {desc}")

    lines.extend([
        "",
        "## Constants",
        f"- ε(POPT parity): `{c['eps_popt_parity']}`",
        f"- ε(DBG parity): `{c['eps_dbg_parity']}`",
        f"- IPC floor: `> {c['ipc_floor']}`",
        f"- instructions floor: `>= {c['instructions_floor']}`",
        f"- POPT-arm required: `{', '.join(c['required_popt_policies'])}`",
        f"- DBG-arm optional: `{', '.join(c['optional_dbg_policies'])}`",
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
        "## POPT parity (Sniper)",
        "",
    ])
    if audit_obj.get("parity_popt"):
        lines.append("| benchmark | section | L3 | POPT | ECG_POPT_PRIMARY | |Δ| |")
        lines.append("| --- | ---: | --- | ---: | ---: | ---: |")
        for row in audit_obj["parity_popt"]:
            lines.append(
                f"| {row['benchmark']} | {row['section']} | {row['l3_size']} "
                f"| {row['popt']} | {row['ecg_popt_primary']} | {row['abs_delta']} |"
            )
    else:
        lines.append("_No POPT-arm rows present (deferred or empty)._")

    lines.extend(["", "## DBG parity (Sniper)", ""])
    if audit_obj.get("parity_dbg"):
        lines.append("| benchmark | section | L3 | GRASP | ECG_DBG_ONLY | |Δ| |")
        lines.append("| --- | ---: | --- | ---: | ---: | ---: |")
        for row in audit_obj["parity_dbg"]:
            lines.append(
                f"| {row['benchmark']} | {row['section']} | {row['l3_size']} "
                f"| {row['grasp']} | {row['ecg_dbg_only']} | {row['abs_delta']} |"
            )
    else:
        lines.append("_No DBG-arm rows present (deferred or DBG arm not yet curated)._")

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
    head = "arm,benchmark,section,l3_size,baseline,proposed,abs_delta\n"
    body_rows: list[str] = []
    for r in audit_obj.get("parity_popt", []):
        body_rows.append(
            f"popt,{r['benchmark']},{r['section']},{r['l3_size']},"
            f"{r['popt']},{r['ecg_popt_primary']},{r['abs_delta']}"
        )
    for r in audit_obj.get("parity_dbg", []):
        body_rows.append(
            f"dbg,{r['benchmark']},{r['section']},{r['l3_size']},"
            f"{r['grasp']},{r['ecg_dbg_only']},{r['abs_delta']}"
        )
    return head + ("\n".join(body_rows) + "\n" if body_rows else "")


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
        f"[lit-faith-ecg-sniper-parity] status={out.get('status','active')} "
        f"rows={out['totals']['rows']} cells={out['totals']['cells']} "
        f"violations={len(out['violations'])}"
    )
    return 1 if out["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
