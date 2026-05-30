#!/usr/bin/env python3
"""LIT-CellComp (gate 234): per_observation cell-completeness + arithmetic audit.

Operates on the per_observation table (not per_claim). Catches corpus
regressions where a policy row is silently dropped, an L3 sweep is
truncated, or the `delta_vs_lru_pct` column drifts from the underlying
miss-rate arithmetic.

Per-cell rules:

  C1 cell roster floor      every (graph, app, l3) cell must include
                            the canonical policy roster {LRU, GRASP,
                            POPT}
  C2 LRU baseline present   every cell must carry an LRU row (otherwise
                            delta_vs_lru_pct is undefined)
  C3 L3 sweep coverage      every (graph, app) must cover >= 3 distinct
                            L3 sizes per non-LRU policy
  C4 L3 axis parity         within (graph, app), every present policy
                            must share the same set of L3 sizes
                            (no policy may be sampled at fewer L3
                            points than another)
  C5 delta_vs_lru arithmetic delta_vs_lru_pct ~= (miss_rate -
                            lru_miss_rate) * 100 within 0.001 pp
                            (excluding LRU itself where the delta is 0)
  C6 miss-rate bounds       0 <= miss_rate <= 1, no NaN/inf
  C7 unique row per
     (graph, app, l3, policy) the per_observation table must not have
                            duplicate rows for the same combo

Emits wiki/data/lit_faith_cellcomp.{json,md,csv}.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


CANONICAL_ROSTER     = {"LRU", "GRASP", "POPT"}
MIN_L3_SWEEP         = 3
DELTA_ARITH_TOL_PP   = 0.001


def _is_finite_unit(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    if math.isnan(v) or math.isinf(v):
        return False
    return 0.0 <= v <= 1.0


def audit(per_obs: list[dict[str, Any]]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []

    # 1. duplicate-row check (C7)
    seen: dict[tuple, int] = defaultdict(int)
    for r in per_obs:
        key = (r.get("graph"), r.get("app"), r.get("l3_size"),
               r.get("policy"))
        seen[key] += 1
    for key, n in seen.items():
        if n > 1:
            violations.append({
                "rule": "C7_unique_row", "row_id": list(key),
                "observed": n,
            })

    # 2. cell-level checks (C1, C2, C5, C6)
    cells: dict[tuple, dict[str, float]] = defaultdict(dict)
    for r in per_obs:
        if not _is_finite_unit(r.get("miss_rate")):
            violations.append({
                "rule": "C6_miss_rate_bounds",
                "row_id": [r.get("graph"), r.get("app"),
                           r.get("l3_size"), r.get("policy")],
                "miss_rate": r.get("miss_rate"),
            })
            continue
        cells[(r.get("graph"), r.get("app"), r.get("l3_size"))][
            r.get("policy")] = float(r["miss_rate"])

    cell_records: list[dict[str, Any]] = []
    for (graph, app, l3), pmap in sorted(cells.items()):
        row = {
            "graph": graph, "app": app, "l3_size": l3,
            "policy_count": len(pmap),
            "policies": sorted(pmap.keys()),
            "lru_miss_rate": pmap.get("LRU"),
        }
        cell_records.append(row)

        if "LRU" not in pmap:
            violations.append({
                "rule": "C2_lru_baseline_missing",
                "row_id": [graph, app, l3, None],
            })
        missing = CANONICAL_ROSTER - set(pmap.keys())
        if missing:
            violations.append({
                "rule": "C1_cell_roster_floor",
                "row_id": [graph, app, l3, None],
                "missing": sorted(missing),
            })

    # 3. delta arithmetic (C5)
    by_cell_lru: dict[tuple, float] = {}
    for r in per_obs:
        if r.get("policy") == "LRU":
            by_cell_lru[(r["graph"], r["app"], r["l3_size"])] = float(
                r["miss_rate"])

    for r in per_obs:
        if r.get("policy") == "LRU":
            continue
        delta = r.get("delta_vs_lru_pct")
        if delta is None:
            continue
        lru = by_cell_lru.get((r["graph"], r["app"], r["l3_size"]))
        if lru is None:
            continue
        expected = (float(r["miss_rate"]) - lru) * 100.0
        if abs(expected - float(delta)) > DELTA_ARITH_TOL_PP:
            violations.append({
                "rule": "C5_delta_arithmetic",
                "row_id": [r["graph"], r["app"], r["l3_size"],
                           r.get("policy")],
                "stored_delta_pp": delta,
                "recomputed_delta_pp": round(expected, 4),
                "diff_pp": round(abs(expected - float(delta)), 5),
            })

    # 4. L3 sweep coverage + axis parity (C3, C4)
    by_pol_app: dict[tuple, set[str]] = defaultdict(set)
    for r in per_obs:
        by_pol_app[(r["graph"], r["app"], r["policy"])].add(r["l3_size"])

    # C3 per (graph, app, non-LRU policy)
    sweep_records: list[dict[str, Any]] = []
    for (graph, app, policy), l3s in sorted(by_pol_app.items()):
        sweep_records.append({
            "graph": graph, "app": app, "policy": policy,
            "l3_count": len(l3s),
            "l3_sizes": sorted(l3s),
        })
        if policy != "LRU" and len(l3s) < MIN_L3_SWEEP:
            violations.append({
                "rule": "C3_l3_sweep_coverage",
                "row_id": [graph, app, None, policy],
                "l3_count": len(l3s),
                "floor": MIN_L3_SWEEP,
            })

    # C4 axis parity per (graph, app)
    by_graph_app: dict[tuple, dict[str, set[str]]] = defaultdict(dict)
    for (graph, app, policy), l3s in by_pol_app.items():
        by_graph_app[(graph, app)][policy] = l3s
    for (graph, app), pmap in sorted(by_graph_app.items()):
        l3_sets = list(pmap.values())
        if not l3_sets:
            continue
        union = set().union(*l3_sets)
        for policy, l3s in pmap.items():
            if l3s != union:
                violations.append({
                    "rule": "C4_l3_axis_parity",
                    "row_id": [graph, app, None, policy],
                    "missing_l3": sorted(union - l3s),
                })

    summary = {
        "min_l3_sweep":          MIN_L3_SWEEP,
        "delta_arith_tol_pp":    DELTA_ARITH_TOL_PP,
        "canonical_roster":      sorted(CANONICAL_ROSTER),
        "total_rows":            len(per_obs),
        "duplicate_rows":        sum(1 for c in seen.values() if c > 1),
        "cell_count":            len(cells),
        "violations":            len(violations),
        "violations_by_rule": {},
        "graphs":                sorted({r["graph"]   for r in per_obs}),
        "apps":                  sorted({r["app"]     for r in per_obs}),
        "policies":              sorted({r["policy"]  for r in per_obs}),
    }
    from collections import Counter
    summary["violations_by_rule"] = dict(Counter(v["rule"] for v in violations))

    return {
        "schema_version": 1,
        "summary":        summary,
        "cells":          cell_records,
        "sweeps":         sweep_records,
        "violations":     violations,
    }


def _to_markdown(a: dict[str, Any]) -> str:
    s = a["summary"]
    lines = ["# Literature-faithfulness cell-completeness audit (LIT-CellComp)",
             "",
             "Per (graph, app, l3) cell in the per_observation table: "
             "canonical roster {LRU, GRASP, POPT} present + LRU "
             "baseline + delta arithmetic + L3 sweep coverage + axis "
             "parity + miss-rate bounds + uniqueness.",
             "",
             "## Summary", "",
             "| Metric | Value |", "|---|---|",
             f"| Total per_observation rows | {s['total_rows']} |",
             f"| Distinct (graph, app, l3) cells | {s['cell_count']} |",
             f"| Distinct graphs | {len(s['graphs'])} |",
             f"| Distinct apps | {len(s['apps'])} |",
             f"| Distinct policies | {len(s['policies'])} |",
             f"| Duplicate rows | {s['duplicate_rows']} |",
             f"| Min L3 sweep | {s['min_l3_sweep']} |",
             f"| Delta arithmetic tolerance | {s['delta_arith_tol_pp']} pp |",
             f"| Violations | {s['violations']} |",
             ""]

    if s["violations_by_rule"]:
        lines += ["## Violations by rule", "", "| rule | count |", "|---|---|"]
        for k, v in sorted(s["violations_by_rule"].items()):
            lines.append(f"| {k} | {v} |")
        lines.append("")

    lines += ["## Per-cell policy roster (first 30)", "",
              "| graph | app | l3 | n | policies | LRU miss |",
              "|---|---|---|---|---|---|"]
    for r in a["cells"][:30]:
        lines.append(
            f"| {r['graph']} | {r['app']} | {r['l3_size']} | "
            f"{r['policy_count']} | {','.join(r['policies'])} | "
            f"{r['lru_miss_rate']} |"
        )

    if a["violations"]:
        lines += ["", f"## First 30 violations", "",
                  "| rule | row_id | detail |",
                  "|---|---|---|"]
        for v in a["violations"][:30]:
            detail = ", ".join(f"{k}={v[k]}" for k in v
                                if k not in ("rule", "row_id"))
            lines.append(f"| {v['rule']} | {v.get('row_id')} | {detail} |")
    else:
        lines += ["", "_No violations._"]
    return "\n".join(lines) + "\n"


def _to_csv(a: dict[str, Any], path: Path) -> None:
    fields = ["graph", "app", "l3_size", "policy_count", "policies",
              "lru_miss_rate"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in a["cells"]:
            row = {k: r.get(k) for k in fields}
            row["policies"] = ",".join(r["policies"])
            w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lit-faith-json", required=True, type=Path)
    ap.add_argument("--json-out",       required=True, type=Path)
    ap.add_argument("--md-out",         required=True, type=Path)
    ap.add_argument("--csv-out",        required=True, type=Path)
    args = ap.parse_args()

    payload = json.loads(args.lit_faith_json.read_text(encoding="utf-8"))
    a = audit(payload["per_observation"])

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(a, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    args.md_out.write_text(_to_markdown(a), encoding="utf-8")
    _to_csv(a, args.csv_out)

    s = a["summary"]
    print(f"[lit-faith-cellcomp] {s['total_rows']} rows / "
          f"{s['cell_count']} cells; violations={s['violations']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
