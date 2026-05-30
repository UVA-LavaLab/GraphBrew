#!/usr/bin/env python3
"""LIT-PolyOrd (gate 231): per (graph_family × app) policy-ordering audit.

The literature claims that hub-bearing graphs (social, citation, web)
gain from hot-vertex retention — so GRASP and POPT should improve over
LRU on the majority of cells in those families. On hub-less graphs
(road, mesh) the same policies typically regress against LRU because
there is no power-law degree distribution to exploit.

This gate locks the *direction* of those orderings per family × app
bucket without imposing point estimates:

  Hub-bearing families (social, citation, web):
    median(POPT - LRU)  must be <= POPT_HUB_BOUND_PP  (0.5 pp)
    median(GRASP - LRU) must be <= GRASP_HUB_BOUND_PP (1.0 pp)
    fraction of cells where POPT improves over LRU >= IMPROVE_FRAC (0.50)

  Hub-less families (road, mesh):
    No ordering constraint, but the regime is flagged so a future
    family addition cannot accidentally be left unclassified.

Also computes the per-(graph_family × app) cell counts so a silent
corpus shrink (an axis going from 6 → 2 cells) trips its floor.

Emits .json / .md / .csv to wiki/data/.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


# Reuse the canonical map (must stay in sync with policy_winner_table.GRAPH_FAMILY)
GRAPH_FAMILY: dict[str, str] = {
    "email-Eu-core":    "social",
    "web-Google":       "web",
    "cit-Patents":      "citation",
    "soc-pokec":        "social",
    "soc-LiveJournal1": "social",
    "com-orkut":        "social",
    "roadNet-CA":       "road",
    "delaunay_n19":     "mesh",
    "road-CA":          "road",
    "twitter-2010":     "social",
    "uk-2005":          "web",
}

HUB_FAMILIES    = {"social", "citation", "web"}
NO_HUB_FAMILIES = {"road", "mesh"}

POPT_HUB_BOUND_PP   = 0.5
GRASP_HUB_BOUND_PP  = 1.0
IMPROVE_FRAC_FLOOR  = 0.50
CELL_COUNT_FLOOR    = 2
IMPROVE_FRAC_MIN_N  = 5  # tiny buckets (n<5) only enforce the median

PER_APP_GLOBAL_FRAC_FLOOR = 0.55   # hub-bearing only, all families combined


def _family(graph: str) -> str:
    return GRAPH_FAMILY.get(graph, "unknown")


def build_audit(lit_path: Path) -> dict[str, Any]:
    payload = json.loads(lit_path.read_text(encoding="utf-8"))
    per_obs = payload["per_observation"]

    # cell_key -> {policy: miss_rate}
    cells: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for r in per_obs:
        cells[(r["graph"], r["app"], r["l3_size"])][r["policy"]] = float(
            r["miss_rate"])

    # bucket: (family, app) -> list of dicts {graph, l3, popt_minus_lru,
    #                                          grasp_minus_lru, srrip_minus_lru}
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    unknown_family_graphs: set[str] = set()

    for (graph, app, l3), pmap in cells.items():
        fam = _family(graph)
        if fam == "unknown":
            unknown_family_graphs.add(graph)
            continue
        lru = pmap.get("LRU")
        if lru is None:
            continue
        entry = {"graph": graph, "l3_size": l3, "lru": lru}
        for p_key, p_name in [("popt", "POPT"), ("grasp", "GRASP"),
                              ("srrip", "SRRIP")]:
            v = pmap.get(p_name)
            entry[p_key] = v
            entry[f"{p_key}_minus_lru_pp"] = (
                (v - lru) * 100.0 if v is not None else None
            )
        buckets[(fam, app)].append(entry)

    bucket_rows: list[dict[str, Any]] = []
    violations:  list[dict[str, Any]] = []

    for (fam, app), cells_list in sorted(buckets.items()):
        n = len(cells_list)
        popt_deltas  = [c["popt_minus_lru_pp"]  for c in cells_list
                        if c["popt_minus_lru_pp"] is not None]
        grasp_deltas = [c["grasp_minus_lru_pp"] for c in cells_list
                        if c["grasp_minus_lru_pp"] is not None]
        srrip_deltas = [c["srrip_minus_lru_pp"] for c in cells_list
                        if c["srrip_minus_lru_pp"] is not None]

        def _med(xs: list[float]) -> float | None:
            return round(statistics.median(xs), 4) if xs else None

        def _frac_improve(xs: list[float]) -> float | None:
            if not xs:
                return None
            return round(sum(1 for x in xs if x < 0) / len(xs), 4)

        row = {
            "graph_family":      fam,
            "regime":            "hub" if fam in HUB_FAMILIES else "no_hub",
            "app":               app,
            "cell_count":        n,
            "graph_count":       len(sorted({c["graph"] for c in cells_list})),
            "popt_median_pp":    _med(popt_deltas),
            "grasp_median_pp":   _med(grasp_deltas),
            "srrip_median_pp":   _med(srrip_deltas),
            "popt_improve_frac": _frac_improve(popt_deltas),
            "grasp_improve_frac":_frac_improve(grasp_deltas),
            "srrip_improve_frac":_frac_improve(srrip_deltas),
            "popt_max_pp":       round(max(popt_deltas), 4)  if popt_deltas  else None,
            "grasp_max_pp":      round(max(grasp_deltas), 4) if grasp_deltas else None,
            "srrip_max_pp":      round(max(srrip_deltas), 4) if srrip_deltas else None,
        }
        bucket_rows.append(row)

        if fam in HUB_FAMILIES:
            if n < CELL_COUNT_FLOOR:
                violations.append({
                    "graph_family": fam, "app": app,
                    "rule": "cell_count_floor",
                    "observed": n, "floor": CELL_COUNT_FLOOR,
                })
            if (row["popt_median_pp"] is not None and
                    row["popt_median_pp"] > POPT_HUB_BOUND_PP):
                violations.append({
                    "graph_family": fam, "app": app,
                    "rule": "popt_median_le_hub_bound",
                    "observed_pp": row["popt_median_pp"],
                    "bound_pp": POPT_HUB_BOUND_PP,
                })
            if (row["grasp_median_pp"] is not None and
                    row["grasp_median_pp"] > GRASP_HUB_BOUND_PP):
                violations.append({
                    "graph_family": fam, "app": app,
                    "rule": "grasp_median_le_hub_bound",
                    "observed_pp": row["grasp_median_pp"],
                    "bound_pp": GRASP_HUB_BOUND_PP,
                })
            if (row["popt_improve_frac"] is not None and
                    n >= IMPROVE_FRAC_MIN_N and
                    row["popt_improve_frac"] < IMPROVE_FRAC_FLOOR):
                violations.append({
                    "graph_family": fam, "app": app,
                    "rule": "popt_improve_frac_floor",
                    "observed": row["popt_improve_frac"],
                    "floor": IMPROVE_FRAC_FLOOR,
                    "n": n,
                })

    # Per-app global aggregates restricted to hub-bearing cells.
    per_app_global: dict[str, dict[str, Any]] = {}
    by_app_hub: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (fam, app), cells_list in buckets.items():
        if fam in HUB_FAMILIES:
            by_app_hub[app].extend(cells_list)

    for app, cells_list in sorted(by_app_hub.items()):
        popt_deltas = [c["popt_minus_lru_pp"] for c in cells_list
                       if c["popt_minus_lru_pp"] is not None]
        frac = (round(sum(1 for x in popt_deltas if x < 0) / len(popt_deltas), 4)
                if popt_deltas else None)
        per_app_global[app] = {
            "hub_cells": len(cells_list),
            "popt_median_pp": (round(statistics.median(popt_deltas), 4)
                               if popt_deltas else None),
            "popt_improve_frac": frac,
        }
        if frac is not None and frac < PER_APP_GLOBAL_FRAC_FLOOR:
            violations.append({
                "graph_family": "(all hub)",
                "app": app,
                "rule": "per_app_global_improve_frac",
                "observed": frac,
                "floor": PER_APP_GLOBAL_FRAC_FLOOR,
            })

    summary = {
        "popt_hub_bound_pp":   POPT_HUB_BOUND_PP,
        "grasp_hub_bound_pp":  GRASP_HUB_BOUND_PP,
        "improve_frac_floor":  IMPROVE_FRAC_FLOOR,
        "cell_count_floor":    CELL_COUNT_FLOOR,
        "per_app_global_improve_frac_floor": PER_APP_GLOBAL_FRAC_FLOOR,
        "total_cells":         sum(1 for _ in cells),
        "bucket_count":        len(buckets),
        "violations":          len(violations),
        "hub_buckets":         sum(1 for k in buckets
                                   if _family(k[0]) in HUB_FAMILIES
                                   or k[0] in HUB_FAMILIES),
        "no_hub_buckets":      sum(1 for k in buckets if k[0] in NO_HUB_FAMILIES),
        "unknown_family_graphs": sorted(unknown_family_graphs),
    }
    # `hub_buckets` above used the wrong key (k[0] is family already).
    summary["hub_buckets"]    = sum(1 for (fam, _) in buckets
                                    if fam in HUB_FAMILIES)
    summary["no_hub_buckets"] = sum(1 for (fam, _) in buckets
                                    if fam in NO_HUB_FAMILIES)

    return {
        "schema_version": 1,
        "summary":        summary,
        "buckets":        bucket_rows,
        "per_app_global": per_app_global,
        "violations":     violations,
    }


def _to_markdown(audit: dict[str, Any]) -> str:
    s = audit["summary"]
    lines = ["# Literature-faithfulness policy-ordering audit (LIT-PolyOrd)",
             "",
             "Per (graph_family × app) bucket: locks the direction of the "
             "POPT-vs-LRU and GRASP-vs-LRU orderings on hub-bearing families "
             "(social / citation / web) while letting hub-less families "
             "(road / mesh) regress as the literature documents.",
             "",
             "## Summary",
             "",
             "| Metric | Value |",
             "|---|---|",
             f"| Bucket count | {s['bucket_count']} |",
             f"| Hub-bearing buckets | {s['hub_buckets']} |",
             f"| Hub-less buckets | {s['no_hub_buckets']} |",
             f"| POPT median bound (hub) | ≤ {s['popt_hub_bound_pp']} pp |",
             f"| GRASP median bound (hub) | ≤ {s['grasp_hub_bound_pp']} pp |",
             f"| POPT improve-frac floor (hub, n≥{IMPROVE_FRAC_MIN_N}) | ≥ {s['improve_frac_floor']} |",
             f"| Per-app global POPT improve-frac floor (hub) | ≥ {s['per_app_global_improve_frac_floor']} |",
             f"| Cell-count floor (hub) | ≥ {s['cell_count_floor']} |",
             f"| Violations | {s['violations']} |",
             ""]

    if s["unknown_family_graphs"]:
        lines += [f"**Unknown-family graphs:** {s['unknown_family_graphs']}",
                  ""]

    lines += ["## Per-bucket detail",
              "",
              "| family | regime | app | n | POPT med (pp) | "
              "POPT improve frac | GRASP med (pp) | GRASP improve frac |",
              "|---|---|---|---|---|---|---|---|"]
    for r in audit["buckets"]:
        lines.append(
            f"| {r['graph_family']} | {r['regime']} | {r['app']} | "
            f"{r['cell_count']} | {r['popt_median_pp']} | "
            f"{r['popt_improve_frac']} | {r['grasp_median_pp']} | "
            f"{r['grasp_improve_frac']} |"
        )

    lines += ["", "## Per-app global (hub-bearing only)", "",
              "| app | hub cells | POPT median (pp) | POPT improve frac |",
              "|---|---|---|---|"]
    for app, row in audit["per_app_global"].items():
        lines.append(
            f"| {app} | {row['hub_cells']} | {row['popt_median_pp']} | "
            f"{row['popt_improve_frac']} |"
        )

    if audit["violations"]:
        lines += ["", f"## Violations ({len(audit['violations'])})", "",
                  "| family | app | rule | observed | bound |",
                  "|---|---|---|---|---|"]
        for v in audit["violations"]:
            lines.append(
                f"| {v.get('graph_family')} | {v.get('app')} | {v['rule']} "
                f"| {v.get('observed') or v.get('observed_pp')} | "
                f"{v.get('floor') or v.get('bound_pp')} |"
            )
    else:
        lines += ["", "_No violations._"]

    return "\n".join(lines) + "\n"


def _to_csv(audit: dict[str, Any], csv_path: Path) -> None:
    fields = ["graph_family", "regime", "app", "cell_count", "graph_count",
              "popt_median_pp", "grasp_median_pp", "srrip_median_pp",
              "popt_improve_frac", "grasp_improve_frac", "srrip_improve_frac",
              "popt_max_pp", "grasp_max_pp", "srrip_max_pp"]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in audit["buckets"]:
            w.writerow({k: r.get(k) for k in fields})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lit-faith-json", required=True, type=Path)
    ap.add_argument("--json-out",       required=True, type=Path)
    ap.add_argument("--md-out",         required=True, type=Path)
    ap.add_argument("--csv-out",        required=True, type=Path)
    args = ap.parse_args()

    audit = build_audit(args.lit_faith_json)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    args.md_out.write_text(_to_markdown(audit), encoding="utf-8")
    _to_csv(audit, args.csv_out)

    s = audit["summary"]
    print(f"[lit-faith-polyord] {s['bucket_count']} buckets "
          f"(hub={s['hub_buckets']} no_hub={s['no_hub_buckets']}); "
          f"violations={s['violations']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
