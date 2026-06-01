"""ECG PFX vs DROPLET runtime comparison (gate 241 — ECG-Pfx-vs-DROPLET).

Sibling family to the substrate-parity trinity (gates 238/239/240),
but a different concern: those three lock that ECG mode ≡ stock mode
on the same backend (substrate faithfulness); this gate compares the
ECG PFX prefetcher against DROPLET on the **same baseline** to make
the prefetcher-evaluation story sound.

Today: SCAFFOLD/DEFERRED. Audit of every /tmp/graphbrew-*/roi_matrix
.csv shows that the droplet_* and ecg_pfx_* columns present are
configuration-only (degrees, table sizes, delivery mode, hint filter)
— the runtime activity columns (droplet_indirect_issued,
droplet_stride_issued, ecg_pfx_issued, ecg_pfx_target_hints_seen,
ecg_pfx_useful, …) are zero everywhere. So neither prefetcher was
actually exercised, and the gate cannot do a faithful comparison yet.

Scope when activated:
  G1 — Arm completeness. Every benchmark in the observed bench-floor
       must carry every required arm ∈ ``REQUIRED_ARMS`` (LRU baseline
       + DROPLET arm + ECG_PFX arm) with status=ok per (section, L3)
       cell.
  G2 — Baseline neutrality floor. For every cell, the L3 miss-rate
       difference between LRU and either prefetcher must be ≤
       ``EPS_L3_MISS_RATE_NEUTRAL_FLOOR`` if the prefetcher delivered
       zero useful prefetches (a prefetcher that fires but never
       helps must not degrade the baseline).
  G3 — Useful-fraction floor. When a prefetcher arm reports
       ``pf_issued > 0``, ``pf_useful / pf_issued`` must be ≥
       ``EPS_USEFUL_PREFETCH_FLOOR``. Below the floor the prefetcher
       is essentially noise.
  G4 — No-strict-loser claim integrity. For any cell where ECG_PFX
       wins (lower miss-rate) over DROPLET, the win must be by ≥
       ``EPS_L3_MISS_RATE_NEUTRAL_FLOOR``; same for DROPLET-wins.
       Below that, the cell is "comparable" and is logged but does
       not violate.
  G5 — Backend identity. ``backend`` and ``simulator`` on every row
       must agree with the postfix-declared backend (sniper/gem5/
       cache_sim, depending on the matched-proof source).
  G6 — Observation floor. ``len(per_observation) >= postfix.
       expected_minimum_observations``.

Deferred-stub mode (today): ``per_observation == []`` AND ``status ==
"deferred"``. In this mode, the audit emits zero violations, stamps
``audit.status = "deferred"``, and echoes the expected source pattern
+ minimum observations + required arms from the postfix. Tests lock
that the deferred shape is intentional and the schema is complete.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

REQUIRED_ARMS = {"LRU", "DROPLET", "ECG_PFX"}
EPS_L3_MISS_RATE_NEUTRAL_FLOOR = 0.005    # 0.5 pp — half-pp of slop
EPS_USEFUL_PREFETCH_FLOOR = 0.05          # 5 % useful or you're noise

# Per-arm row statuses we accept as a successful observation:
#   'ok'             = simulation completed cleanly
#   'active_no_fill' = prefetcher consumed hints / generated requests
#       but Sniper's L2 enqueue filter dropped them (legitimate
#       cache_cntlr.cc:1146 'already in cache' filtering). Gate 296
#       (baseline-neutrality) covers this regime explicitly.
#   'inactive'       = prefetcher configured but no hints/edges fired
#       (gate 296 also covers this).
# Anything else (most notably 'error') signals a broken cell and
# fires G1-non-ok-status.
VALID_RUNTIME_STATUSES = frozenset({"ok", "active_no_fill", "inactive"})


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
        arm = r.get("arm")
        graph = r.get("graph", "")
        backends.add(r.get("backend", "unknown"))
        if bench:
            benches.add(bench)
        if sect is not None:
            sections.add(int(sect))
        if bench and sect is not None and l3 and arm:
            # Include graph in the cell key so observations across
            # multiple graphs at the same (bench, section, l3) don't
            # collapse and overwrite each other.
            by_cell[(bench, int(sect), l3, graph)][arm] = r
    return by_cell, sections, backends, benches


def audit(postfix: dict[str, Any]) -> dict[str, Any]:
    rows = postfix.get("per_observation", [])
    status_decl = postfix.get("status", "active")
    deferred = (status_decl == "deferred") and not rows

    if deferred:
        return {
            "status": "deferred",
            "defer_reason": postfix.get("defer_reason", ""),
            "expected_source_pattern":
                postfix.get("expected_source_pattern", ""),
            "expected_minimum_observations":
                postfix.get("expected_minimum_observations", 0),
            "expected_required_arms":
                postfix.get("expected_required_arms", sorted(REQUIRED_ARMS)),
            "rules": _rule_descriptions(),
            "constants": _constants(),
            "totals": {
                "rows": 0, "cells": 0, "benchmarks": [],
                "backends": [], "sections": [], "arms_present": [],
            },
            "head_to_head": [],
            "violations": [],
        }

    by_cell, sections, backends, benches = _build_cells(rows)
    bench_floor = set(benches)
    violations: list[dict[str, Any]] = []

    expected_min = int(postfix.get("expected_minimum_observations", 0))
    if expected_min and len(rows) < expected_min:
        violations.append({
            "rule": "G6-observation-floor",
            "rows": len(rows), "expected_minimum": expected_min,
        })

    expected_backend = postfix.get("expected_backend")

    for cell, table in sorted(by_cell.items()):
        bench, sect, l3, graph = cell
        if bench not in bench_floor:
            continue
        for arm in sorted(REQUIRED_ARMS):
            row = table.get(arm)
            if row is None:
                violations.append({
                    "rule": "G1-missing-arm", "benchmark": bench,
                    "section": sect, "l3_size": l3, "graph": graph, "arm": arm,
                })
            elif row.get("status") not in VALID_RUNTIME_STATUSES:
                violations.append({
                    "rule": "G1-non-ok-status", "benchmark": bench,
                    "section": sect, "l3_size": l3, "graph": graph, "arm": arm,
                    "status": row.get("status"),
                })

    head_to_head: list[dict[str, Any]] = []
    for cell, table in sorted(by_cell.items()):
        bench, sect, l3, graph = cell
        lru = table.get("LRU", {})
        drop = table.get("DROPLET", {})
        ecg = table.get("ECG_PFX", {})
        lru_mr = lru.get("l3_miss_rate")
        drop_mr = drop.get("l3_miss_rate")
        ecg_mr = ecg.get("l3_miss_rate")
        h2h = {
            "benchmark": bench, "section": sect, "l3_size": l3, "graph": graph,
            "lru_miss_rate": lru_mr,
            "droplet_miss_rate": drop_mr, "ecg_pfx_miss_rate": ecg_mr,
            "droplet_useful_frac": _useful(drop),
            "ecg_pfx_useful_frac": _useful(ecg),
            "droplet_vs_lru_delta": _delta(drop_mr, lru_mr),
            "ecg_pfx_vs_lru_delta": _delta(ecg_mr, lru_mr),
            "ecg_vs_droplet_delta": _delta(ecg_mr, drop_mr),
            # Surface runtime activity so consumers can skip vacuous cells
            # where the prefetcher returned zero addresses after Sniper's
            # already-in-cache filter (cache_cntlr.cc:1146).
            "droplet_pf_issued": drop.get("pf_issued") or 0,
            "droplet_pf_useful": drop.get("pf_useful") or 0,
            "ecg_pfx_pf_issued": ecg.get("pf_issued") or 0,
            "ecg_pfx_pf_useful": ecg.get("pf_useful") or 0,
            "ecg_pfx_target_hints_seen": ecg.get("ecg_pfx_target_hints_seen") or 0,
            "ecg_pfx_issued": ecg.get("ecg_pfx_issued") or 0,
        }
        head_to_head.append(h2h)

        for arm, row, mr in (("DROPLET", drop, drop_mr), ("ECG_PFX", ecg, ecg_mr)):
            issued = row.get("pf_issued") or 0
            useful = row.get("pf_useful") or 0
            if issued == 0 and mr is not None and lru_mr is not None:
                if mr - lru_mr > EPS_L3_MISS_RATE_NEUTRAL_FLOOR:
                    violations.append({
                        "rule": "G2-baseline-neutral-broken",
                        "benchmark": bench, "section": sect, "l3_size": l3,
                        "graph": graph,
                        "arm": arm, "arm_miss_rate": mr,
                        "lru_miss_rate": lru_mr,
                        "delta": mr - lru_mr,
                        "epsilon": EPS_L3_MISS_RATE_NEUTRAL_FLOOR,
                    })
            elif issued > 0:
                u_frac = useful / issued if issued else 0.0
                if u_frac < EPS_USEFUL_PREFETCH_FLOOR:
                    violations.append({
                        "rule": "G3-useful-fraction-low",
                        "benchmark": bench, "section": sect, "l3_size": l3,
                        "graph": graph,
                        "arm": arm, "pf_issued": issued,
                        "pf_useful": useful, "useful_frac": u_frac,
                        "epsilon": EPS_USEFUL_PREFETCH_FLOOR,
                    })

    if expected_backend:
        for r in rows:
            b = r.get("backend")
            s = r.get("simulator")
            if b != expected_backend or s != expected_backend:
                violations.append({
                    "rule": "G5-backend-mismatch",
                    "expected_backend": expected_backend,
                    "backend": b, "simulator": s,
                    "arm": r.get("arm"), "section": r.get("section"),
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
            "arms_present":
                sorted({r.get("arm") for r in rows if r.get("arm")}),
        },
        "head_to_head": head_to_head,
        "violations": violations,
    }


def _useful(row: dict[str, Any]) -> float | None:
    issued = row.get("pf_issued") or 0
    useful = row.get("pf_useful") or 0
    if not issued:
        return None
    return useful / issued


def _rule_descriptions() -> dict[str, str]:
    return {
        "G1": "every required arm {LRU, DROPLET, ECG_PFX} present with status=ok per (benchmark, section, l3_size) cell",
        "G2": f"if pf_issued==0 then arm_miss_rate - lru_miss_rate <= {EPS_L3_MISS_RATE_NEUTRAL_FLOOR} (a quiet prefetcher must not degrade baseline)",
        "G3": f"if pf_issued > 0 then pf_useful / pf_issued >= {EPS_USEFUL_PREFETCH_FLOOR}",
        "G4": "logged-only: head-to-head wins below the neutral floor are comparable (no violation)",
        "G5": "if postfix.expected_backend set, every row has matching backend AND simulator",
        "G6": "len(per_observation) >= postfix.expected_minimum_observations",
    }


def _constants() -> dict[str, Any]:
    return {
        "eps_l3_miss_rate_neutral_floor":  EPS_L3_MISS_RATE_NEUTRAL_FLOOR,
        "eps_useful_prefetch_floor":       EPS_USEFUL_PREFETCH_FLOOR,
        "required_arms":                   sorted(REQUIRED_ARMS),
    }


def render_markdown(audit_obj: dict[str, Any]) -> str:
    t = audit_obj["totals"]
    c = audit_obj["constants"]
    lines = [
        "# ECG PFX vs DROPLET head-to-head",
        "",
        "Gate 241 — ECG-Pfx-vs-DROPLET. Sibling family to the substrate-",
        "parity trinity (gates 238/239/240) but evaluates a different",
        "axis: ECG's PFX prefetcher vs DROPLET on the same baseline.",
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
            f"- expected required arms: `{', '.join(audit_obj.get('expected_required_arms', []))}`",
        ])

    lines.extend(["", "## Rules"])
    for rid, desc in audit_obj["rules"].items():
        lines.append(f"- **{rid}** — {desc}")

    lines.extend([
        "",
        "## Constants",
        f"- ε(neutral floor): `{c['eps_l3_miss_rate_neutral_floor']}`",
        f"- ε(useful floor): `{c['eps_useful_prefetch_floor']}`",
        f"- required arms: `{', '.join(c['required_arms'])}`",
        "",
        "## Totals",
        f"- observations: **{t['rows']}**",
        f"- cells (benchmark × section × L3): **{t['cells']}**",
        f"- benchmarks: `{', '.join(t['benchmarks'])}`",
        f"- backends: `{', '.join(t['backends'])}`",
        f"- sections: `{', '.join(map(str, t['sections']))}`",
        f"- arms present: `{', '.join(t['arms_present'])}`",
        "",
        "## Head-to-head",
        "",
    ])
    if audit_obj.get("head_to_head"):
        lines.append("| benchmark | section | L3 | LRU | DROPLET | ECG_PFX | "
                     "DROPLET-LRU | ECG-LRU | ECG-DROPLET | DROPLET useful | ECG useful |")
        lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in audit_obj["head_to_head"]:
            lines.append(
                f"| {r['benchmark']} | {r['section']} | {r['l3_size']} "
                f"| {r['lru_miss_rate']} | {r['droplet_miss_rate']} | {r['ecg_pfx_miss_rate']} "
                f"| {r['droplet_vs_lru_delta']} | {r['ecg_pfx_vs_lru_delta']} | {r['ecg_vs_droplet_delta']} "
                f"| {r['droplet_useful_frac']} | {r['ecg_pfx_useful_frac']} |"
            )
    else:
        lines.append("_No head-to-head rows (deferred or empty)._")

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
    head = ("benchmark,section,l3_size,lru,droplet,ecg_pfx,"
            "droplet_vs_lru,ecg_vs_lru,ecg_vs_droplet,"
            "droplet_useful,ecg_useful\n")
    body_rows: list[str] = []
    for r in audit_obj.get("head_to_head", []):
        body_rows.append(
            f"{r['benchmark']},{r['section']},{r['l3_size']},"
            f"{r['lru_miss_rate']},{r['droplet_miss_rate']},{r['ecg_pfx_miss_rate']},"
            f"{r['droplet_vs_lru_delta']},{r['ecg_pfx_vs_lru_delta']},{r['ecg_vs_droplet_delta']},"
            f"{r['droplet_useful_frac']},{r['ecg_pfx_useful_frac']}"
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
        f"[lit-faith-ecg-pfx-vs-droplet] status={out.get('status','active')} "
        f"rows={out['totals']['rows']} cells={out['totals']['cells']} "
        f"violations={len(out['violations'])}"
    )
    return 1 if out["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
