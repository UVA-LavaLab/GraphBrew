"""ECG substrate-parity audit (gate 238 — ECG-Parity).

This gate locks the **ECG implementation faithfulness invariants** that the
component-proof matrix is supposed to establish before any cluster-scale
run can be trusted. The audit consumes a curated snapshot of
``proof_matrix.py``'s output (``wiki/data/ecg_substrate_parity_postfix.json``)
and asserts:

  E1 — Roster completeness. Every benchmark ∈ ``BENCH_FLOOR`` must have
       every ablation ∈ ``REQUIRED_ABLATIONS`` present with ``status == "ok"``.
  E2 — DBG substrate parity. ``ECG_DBG_only`` and ``GRASP_DBG_only`` must
       agree on L3 miss-rate to within ``EPS_DBG_PARITY`` for every
       benchmark. ECG_DBG is a re-implementation of GRASP's DBG-only mode
       and any drift here means the ECG substrate has silently diverged.
  E3 — POPT substrate parity. ``ECG_POPT_primary`` must agree with stock
       ``POPT_only`` on L3 miss-rate to within ``EPS_POPT_PARITY`` for every
       benchmark. This is the load-bearing parity for the POPT-side of the
       ECG family.
  E4 — PFX activation floor. Every ablation in ``PFX_ABLATIONS`` must show
       ``ecg_runtime_issued ≥ PFX_ISSUED_FLOOR`` on every benchmark — i.e.
       the prefetch path is actually delivering hints, not silently no-op'ing.
  E5 — PFX usefulness floor. Same set of ablations must show
       ``prefetch_useful > 0`` on PR (the dense-hub anchor) and the
       useful-rate must be a positive fraction of issued
       (``prefetch_useful ≤ prefetch_requests``).
  E6 — Encoding hygiene. For every ablation that emitted any candidates,
       ``ecg_pfx_encoded ≤ ecg_pfx_candidates`` (encoding cannot manufacture
       ghost entries) and every PFX counter is non-negative. ``dedup_skips``
       is deliberately NOT bounded by ``issued`` — dedup is a runtime
       per-access counter that fires every time a cache lookup would
       re-issue a hint already in the prefetch queue, so on sparse
       traversals ``dedup_skips >> issued`` is the expected phenomenology.
  E7 — Baseline non-zero floor. ``LRU_cache_only``, ``SRRIP_cache_only``,
       ``GRASP_DBG_only``, ``POPT_only`` must all have strictly positive
       ``memory_accesses`` and ``l3_misses``. A zero would indicate the
       benchmark didn't actually execute.
  E8 — Backend coverage floor. ``backend`` distinct count ≥ 1 (cache_sim
       baseline today; ratchets up to 3 once gem5/Sniper data lands in
       gates 239/240).

Tolerance values are deliberately tight (5×10⁻⁴ ≈ 0.05 pp) — the parity
should be *near-bitwise* when both implementations honour the same advice
policy. We use ``≤`` rather than ``<`` so that bit-exact 0.0 drift passes
trivially.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

REQUIRED_ABLATIONS = {
    "LRU_cache_only",
    "SRRIP_cache_only",
    "GRASP_DBG_only",
    "POPT_only",
    "ECG_DBG_only",
    "ECG_POPT_primary",
    "PFX_degree_only",
    "PFX_POPT_only",
    "DBG_PFX",
    "POPT_PFX",
}

PFX_ABLATIONS = {
    "PFX_degree_only",
    "PFX_POPT_only",
    "DBG_PFX",
    "POPT_PFX",
}

BENCH_FLOOR = {"pr", "bfs", "sssp"}

BASELINE_NONZERO = {
    "LRU_cache_only",
    "SRRIP_cache_only",
    "GRASP_DBG_only",
    "POPT_only",
}

EPS_DBG_PARITY = 0.0005
EPS_POPT_PARITY = 0.0005
PFX_ISSUED_FLOOR = 1
BACKEND_FLOOR = 1


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return abs(a - b)


def audit(postfix: dict[str, Any]) -> dict[str, Any]:
    rows = postfix.get("per_observation", [])
    violations: list[dict[str, Any]] = []

    by_bench: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    backends: set[str] = set()
    for r in rows:
        backends.add(r.get("backend", "unknown"))
        bench = r.get("benchmark")
        abl = r.get("ablation")
        if bench and abl:
            by_bench[bench][abl] = r

    # E1 — Roster completeness
    for bench in sorted(BENCH_FLOOR):
        table = by_bench.get(bench, {})
        for abl in sorted(REQUIRED_ABLATIONS):
            row = table.get(abl)
            if row is None:
                violations.append({
                    "rule": "E1-missing-ablation", "benchmark": bench, "ablation": abl,
                })
            elif row.get("status") != "ok":
                violations.append({
                    "rule": "E1-non-ok-status", "benchmark": bench, "ablation": abl,
                    "status": row.get("status"),
                })

    # E2 — DBG substrate parity
    parity_dbg: list[dict[str, Any]] = []
    for bench in sorted(BENCH_FLOOR):
        table = by_bench.get(bench, {})
        grasp = table.get("GRASP_DBG_only", {}).get("l3_miss_rate")
        ecg_dbg = table.get("ECG_DBG_only", {}).get("l3_miss_rate")
        delta = _delta(grasp, ecg_dbg)
        parity_dbg.append({
            "benchmark": bench, "grasp_dbg_only": grasp,
            "ecg_dbg_only": ecg_dbg, "abs_delta": delta,
        })
        if delta is None or delta > EPS_DBG_PARITY:
            violations.append({
                "rule": "E2-dbg-parity-drift", "benchmark": bench,
                "grasp_dbg_only": grasp, "ecg_dbg_only": ecg_dbg,
                "abs_delta": delta, "epsilon": EPS_DBG_PARITY,
            })

    # E3 — POPT substrate parity
    parity_popt: list[dict[str, Any]] = []
    for bench in sorted(BENCH_FLOOR):
        table = by_bench.get(bench, {})
        popt = table.get("POPT_only", {}).get("l3_miss_rate")
        ecg_popt = table.get("ECG_POPT_primary", {}).get("l3_miss_rate")
        delta = _delta(popt, ecg_popt)
        parity_popt.append({
            "benchmark": bench, "popt_only": popt,
            "ecg_popt_primary": ecg_popt, "abs_delta": delta,
        })
        if delta is None or delta > EPS_POPT_PARITY:
            violations.append({
                "rule": "E3-popt-parity-drift", "benchmark": bench,
                "popt_only": popt, "ecg_popt_primary": ecg_popt,
                "abs_delta": delta, "epsilon": EPS_POPT_PARITY,
            })

    # E4 / E5 — PFX activation + usefulness
    pfx_activation: list[dict[str, Any]] = []
    for bench in sorted(BENCH_FLOOR):
        table = by_bench.get(bench, {})
        for abl in sorted(PFX_ABLATIONS):
            row = table.get(abl, {})
            issued = row.get("ecg_runtime_issued")
            useful = row.get("prefetch_useful")
            requests = row.get("prefetch_requests")
            pfx_activation.append({
                "benchmark": bench, "ablation": abl,
                "ecg_runtime_issued": issued, "prefetch_useful": useful,
                "prefetch_requests": requests,
            })

            # E4 — issued floor
            if issued is None or issued < PFX_ISSUED_FLOOR:
                violations.append({
                    "rule": "E4-pfx-issued-floor", "benchmark": bench, "ablation": abl,
                    "ecg_runtime_issued": issued, "floor": PFX_ISSUED_FLOOR,
                })
            # E5 — usefulness on PR (hub anchor) must be > 0;
            #      on every benchmark useful ≤ requests
            if bench == "pr" and (useful is None or useful <= 0):
                violations.append({
                    "rule": "E5-pfx-useful-zero-on-pr", "benchmark": bench, "ablation": abl,
                    "prefetch_useful": useful,
                })
            if useful is not None and requests is not None and useful > requests:
                violations.append({
                    "rule": "E5-pfx-useful-exceeds-requests", "benchmark": bench, "ablation": abl,
                    "prefetch_useful": useful, "prefetch_requests": requests,
                })

    # E6 — Encoding hygiene
    dedup_audit: list[dict[str, Any]] = []
    PFX_COUNTERS = (
        "ecg_pfx_candidates", "ecg_pfx_encoded", "ecg_pfx_dedup_skips",
        "ecg_pfx_table_miss", "ecg_pfx_no_candidate", "ecg_pfx_hint_filter",
        "ecg_runtime_issued", "ecg_runtime_duplicate", "ecg_runtime_no_target",
        "prefetch_useful", "prefetch_requests", "prefetch_fills",
    )
    for r in rows:
        candidates = r.get("ecg_pfx_candidates") or 0
        encoded = r.get("ecg_pfx_encoded") or 0
        dedup = r.get("ecg_pfx_dedup_skips") or 0
        issued = r.get("ecg_runtime_issued") or 0
        if candidates or encoded:
            dedup_audit.append({
                "benchmark": r.get("benchmark"), "ablation": r.get("ablation"),
                "ecg_pfx_candidates": candidates, "ecg_pfx_encoded": encoded,
                "ecg_pfx_dedup_skips": dedup, "ecg_runtime_issued": issued,
            })
            if encoded > candidates:
                violations.append({
                    "rule": "E6-encoded-exceeds-candidates",
                    "benchmark": r.get("benchmark"), "ablation": r.get("ablation"),
                    "ecg_pfx_encoded": encoded, "ecg_pfx_candidates": candidates,
                })
        for field in PFX_COUNTERS:
            v = r.get(field)
            if v is not None and v < 0:
                violations.append({
                    "rule": "E6-negative-counter",
                    "benchmark": r.get("benchmark"), "ablation": r.get("ablation"),
                    "field": field, "value": v,
                })

    # E7 — Baseline non-zero floor
    for bench in sorted(BENCH_FLOOR):
        table = by_bench.get(bench, {})
        for abl in sorted(BASELINE_NONZERO):
            row = table.get(abl, {})
            ma = row.get("memory_accesses") or 0
            misses = row.get("l3_misses") or 0
            if ma <= 0 or misses <= 0:
                violations.append({
                    "rule": "E7-baseline-zero", "benchmark": bench, "ablation": abl,
                    "memory_accesses": ma, "l3_misses": misses,
                })

    # E8 — Backend coverage floor
    if len(backends) < BACKEND_FLOOR:
        violations.append({
            "rule": "E8-backend-coverage-floor",
            "backends_present": sorted(backends),
            "floor": BACKEND_FLOOR,
        })

    return {
        "rules": {
            "E1": "every required ablation present with status=ok per benchmark",
            "E2": f"|miss_rate(ECG_DBG_only) - miss_rate(GRASP_DBG_only)| <= {EPS_DBG_PARITY}",
            "E3": f"|miss_rate(ECG_POPT_primary) - miss_rate(POPT_only)| <= {EPS_POPT_PARITY}",
            "E4": f"every PFX ablation has ecg_runtime_issued >= {PFX_ISSUED_FLOOR} per benchmark",
            "E5": "PFX ablation has prefetch_useful > 0 on PR AND prefetch_useful <= prefetch_requests on all benchmarks",
            "E6": "ecg_pfx_encoded <= ecg_pfx_candidates for any row with candidates; all PFX counters are non-negative (dedup_skips is intentionally unbounded vs issued — dedup is a runtime per-access counter)",
            "E7": "baselines (LRU/SRRIP/GRASP/POPT) have strictly positive memory_accesses and l3_misses",
            "E8": f"distinct backend count >= {BACKEND_FLOOR}",
        },
        "constants": {
            "eps_dbg_parity":     EPS_DBG_PARITY,
            "eps_popt_parity":    EPS_POPT_PARITY,
            "pfx_issued_floor":   PFX_ISSUED_FLOOR,
            "backend_floor":      BACKEND_FLOOR,
            "required_ablations": sorted(REQUIRED_ABLATIONS),
            "pfx_ablations":      sorted(PFX_ABLATIONS),
            "baseline_ablations": sorted(BASELINE_NONZERO),
            "bench_floor":        sorted(BENCH_FLOOR),
        },
        "totals": {
            "rows":              len(rows),
            "benchmarks":        sorted(by_bench.keys()),
            "backends":          sorted(backends),
            "ablations_present": sorted({r["ablation"] for r in rows if r.get("ablation")}),
        },
        "parity_dbg":     parity_dbg,
        "parity_popt":    parity_popt,
        "pfx_activation": pfx_activation,
        "dedup_audit":    dedup_audit,
        "violations":     violations,
    }


def render_markdown(audit_obj: dict[str, Any]) -> str:
    t = audit_obj["totals"]
    c = audit_obj["constants"]
    lines = [
        "# ECG substrate-parity audit",
        "",
        "Gate 238 — ECG-Parity. Locks the cache_sim component-proof matrix's",
        "load-bearing invariants: ECG re-implementations match the policies",
        "they shadow, the prefetch path is actually firing, and the dedup",
        "bookkeeping is consistent. This is the confidence-floor that must",
        "hold before any cluster-scale ECG sweep is launched.",
        "",
        "## Rules",
    ]
    for rid, desc in audit_obj["rules"].items():
        lines.append(f"- **{rid}** — {desc}")

    lines.extend([
        "",
        "## Constants",
        f"- ε(DBG parity): `{c['eps_dbg_parity']}`",
        f"- ε(POPT parity): `{c['eps_popt_parity']}`",
        f"- PFX issued floor: `{c['pfx_issued_floor']}`",
        f"- backend floor: `{c['backend_floor']}`",
        f"- required ablations: `{', '.join(c['required_ablations'])}`",
        f"- PFX ablations: `{', '.join(c['pfx_ablations'])}`",
        "",
        "## Totals",
        f"- observations: **{t['rows']}**",
        f"- benchmarks: `{', '.join(t['benchmarks'])}`",
        f"- backends: `{', '.join(t['backends'])}`",
        f"- ablations present: `{', '.join(t['ablations_present'])}`",
        "",
        "## DBG parity",
        "",
        "| benchmark | GRASP_DBG_only | ECG_DBG_only | |Δ| |",
        "| --- | ---: | ---: | ---: |",
    ])
    for row in audit_obj["parity_dbg"]:
        lines.append(
            f"| {row['benchmark']} | {row['grasp_dbg_only']} | "
            f"{row['ecg_dbg_only']} | {row['abs_delta']} |"
        )

    lines.extend([
        "",
        "## POPT parity",
        "",
        "| benchmark | POPT_only | ECG_POPT_primary | |Δ| |",
        "| --- | ---: | ---: | ---: |",
    ])
    for row in audit_obj["parity_popt"]:
        lines.append(
            f"| {row['benchmark']} | {row['popt_only']} | "
            f"{row['ecg_popt_primary']} | {row['abs_delta']} |"
        )

    lines.extend([
        "",
        "## PFX activation",
        "",
        "| benchmark | ablation | issued | useful | requests |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for row in audit_obj["pfx_activation"]:
        lines.append(
            f"| {row['benchmark']} | {row['ablation']} | "
            f"{row['ecg_runtime_issued']} | {row['prefetch_useful']} | "
            f"{row['prefetch_requests']} |"
        )

    lines.extend(["", "## Violations", ""])
    if not audit_obj["violations"]:
        lines.append("_None._")
    else:
        lines.append("| rule | benchmark | ablation | detail |")
        lines.append("| --- | --- | --- | --- |")
        for v in audit_obj["violations"]:
            detail = {k: v[k] for k in v if k not in ("rule", "benchmark", "ablation")}
            lines.append(
                f"| {v.get('rule','')} | {v.get('benchmark','')} | "
                f"{v.get('ablation','')} | {detail} |"
            )

    lines.append("")
    return "\n".join(lines)


def render_csv(audit_obj: dict[str, Any]) -> str:
    head = "benchmark,ablation,issued,useful,requests\n"
    body = "\n".join(
        f"{r['benchmark']},{r['ablation']},{r['ecg_runtime_issued']},"
        f"{r['prefetch_useful']},{r['prefetch_requests']}"
        for r in audit_obj["pfx_activation"]
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
        f"[lit-faith-ecg-parity] rows={out['totals']['rows']} "
        f"benches={out['totals']['benchmarks']} "
        f"backends={out['totals']['backends']} "
        f"violations={len(out['violations'])}"
    )
    return 1 if out["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
