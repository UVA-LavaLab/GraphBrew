#!/usr/bin/env python3
"""LIT-AppFreq (gate 235): per-app axis-coverage audit on per_observation.

Catches "axis collapse" — a corpus regression where one app silently
drops one or more graphs / L3 sizes / policies. Distinct from
LIT-CellComp (which guarantees individual cells are complete given they
exist) and from LIT-Mono (which checks miss-rate monotonicity given an
L3 sweep exists): this gate guarantees the per_observation grid spans
enough axis points per app for the downstream comparators (oracle gap,
margin, monotonicity, ...) to remain well-defined.

Per-app rules:

  F1 graph coverage floor    every app must touch >= MIN_GRAPHS_PER_APP
                             distinct graphs (so claim density per
                             app is not graph-bounded)
  F2 L3 coverage floor       every app must touch >= MIN_L3S_PER_APP
                             distinct L3 sizes (so cache-sensitivity
                             slopes have a fitting basis)
  F3 policy coverage floor   every app must touch >= MIN_POLICIES_PER_APP
                             distinct policies (canonical roster
                             {LRU, GRASP, POPT} must be present per
                             app)
  F4 per-(app,graph) sweep   every (app, graph) pair seen must cover
                             >= MIN_L3_PER_AG L3 sizes
  F5 cell-count floor        every app must contribute >= MIN_CELLS_PER_APP
                             observation rows (so per-app statistics
                             are stable)
  F6 anchor-app full sweep   the load-bearing app (pr) must cover
                             every graph in the corpus (no graph
                             allowed to drop from the pr sweep)

Emits wiki/data/lit_faith_appfreq.{json,md,csv}.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


MIN_GRAPHS_PER_APP    = 6
MIN_L3S_PER_APP       = 3
MIN_POLICIES_PER_APP  = 3
MIN_L3_PER_AG         = 3
MIN_CELLS_PER_APP     = 60      # 6 graphs * 3 L3 * 3 policies
ANCHOR_APP            = "pr"
CANONICAL_ROSTER      = {"LRU", "GRASP", "POPT"}


def audit(per_obs: list[dict[str, Any]]) -> dict[str, Any]:
    apps = sorted({r["app"] for r in per_obs})
    graphs_corpus = sorted({r["graph"] for r in per_obs})

    by_app: dict[str, dict[str, Any]] = {}
    for app in apps:
        rows_app = [r for r in per_obs if r["app"] == app]
        graphs = sorted({r["graph"]   for r in rows_app})
        l3s    = sorted({r["l3_size"] for r in rows_app})
        pols   = sorted({r["policy"]  for r in rows_app})
        by_app[app] = {
            "app":          app,
            "graphs":       graphs,
            "l3_sizes":     l3s,
            "policies":     pols,
            "graph_count":  len(graphs),
            "l3_count":     len(l3s),
            "policy_count": len(pols),
            "row_count":    len(rows_app),
        }

    # Per-(app, graph) L3 sweep
    by_ag: dict[tuple, set[str]] = defaultdict(set)
    for r in per_obs:
        by_ag[(r["app"], r["graph"])].add(r["l3_size"])
    ag_records = [
        {"app": a, "graph": g,
         "l3_count": len(s), "l3_sizes": sorted(s)}
        for (a, g), s in sorted(by_ag.items())
    ]

    violations: list[dict[str, Any]] = []

    # F1: graph coverage floor
    for app, r in sorted(by_app.items()):
        if r["graph_count"] < MIN_GRAPHS_PER_APP:
            violations.append({
                "rule": "F1_graph_coverage_floor",
                "app": app, "observed": r["graph_count"],
                "floor": MIN_GRAPHS_PER_APP,
                "missing_graphs": sorted(set(graphs_corpus) - set(r["graphs"])),
            })

    # F2: L3 coverage floor
    for app, r in sorted(by_app.items()):
        if r["l3_count"] < MIN_L3S_PER_APP:
            violations.append({
                "rule": "F2_l3_coverage_floor", "app": app,
                "observed": r["l3_count"], "floor": MIN_L3S_PER_APP,
            })

    # F3: policy coverage floor + canonical roster
    for app, r in sorted(by_app.items()):
        if r["policy_count"] < MIN_POLICIES_PER_APP:
            violations.append({
                "rule": "F3_policy_coverage_floor", "app": app,
                "observed": r["policy_count"],
                "floor": MIN_POLICIES_PER_APP,
            })
        missing = CANONICAL_ROSTER - set(r["policies"])
        if missing:
            violations.append({
                "rule": "F3_canonical_roster_missing", "app": app,
                "missing_policies": sorted(missing),
            })

    # F4: per-(app, graph) sweep floor
    for (a, g), s in sorted(by_ag.items()):
        if len(s) < MIN_L3_PER_AG:
            violations.append({
                "rule": "F4_per_app_graph_l3_sweep",
                "app": a, "graph": g,
                "observed": len(s), "floor": MIN_L3_PER_AG,
            })

    # F5: row count floor
    for app, r in sorted(by_app.items()):
        if r["row_count"] < MIN_CELLS_PER_APP:
            violations.append({
                "rule": "F5_app_row_count_floor", "app": app,
                "observed": r["row_count"], "floor": MIN_CELLS_PER_APP,
            })

    # F6: anchor-app full sweep
    if ANCHOR_APP in by_app:
        missing_g = sorted(set(graphs_corpus)
                           - set(by_app[ANCHOR_APP]["graphs"]))
        if missing_g:
            violations.append({
                "rule": "F6_anchor_app_full_sweep",
                "app": ANCHOR_APP,
                "missing_graphs": missing_g,
                "corpus_graphs":  graphs_corpus,
            })
    else:
        violations.append({
            "rule": "F6_anchor_app_full_sweep",
            "app": ANCHOR_APP,
            "missing_graphs": graphs_corpus,
            "note": "anchor app not present in per_observation",
        })

    from collections import Counter
    summary = {
        "min_graphs_per_app":   MIN_GRAPHS_PER_APP,
        "min_l3s_per_app":      MIN_L3S_PER_APP,
        "min_policies_per_app": MIN_POLICIES_PER_APP,
        "min_l3_per_ag":        MIN_L3_PER_AG,
        "min_cells_per_app":    MIN_CELLS_PER_APP,
        "anchor_app":           ANCHOR_APP,
        "canonical_roster":     sorted(CANONICAL_ROSTER),
        "apps":                 apps,
        "corpus_graphs":        graphs_corpus,
        "corpus_graph_count":   len(graphs_corpus),
        "total_rows":           len(per_obs),
        "violations":           len(violations),
        "violations_by_rule":   dict(Counter(v["rule"] for v in violations)),
    }
    return {
        "schema_version":   1,
        "summary":          summary,
        "per_app":          [by_app[a] for a in apps],
        "per_app_graph":    ag_records,
        "violations":       violations,
    }


def _to_markdown(a: dict[str, Any]) -> str:
    s = a["summary"]
    lines = ["# Literature-faithfulness app-frequency audit (LIT-AppFreq)",
             "",
             "Per-app axis-coverage check: each app must touch enough "
             "distinct graphs / L3 sizes / policies / observation rows "
             "for the downstream per-app comparators to remain well-"
             "defined, and the anchor app (pr) must cover the full "
             "corpus.",
             "",
             "## Summary", "",
             "| Metric | Value |", "|---|---|",
             f"| Total per_observation rows | {s['total_rows']} |",
             f"| Distinct apps | {len(s['apps'])} |",
             f"| Corpus graph count | {s['corpus_graph_count']} |",
             f"| Min graphs per app | {s['min_graphs_per_app']} |",
             f"| Min L3 sizes per app | {s['min_l3s_per_app']} |",
             f"| Min policies per app | {s['min_policies_per_app']} |",
             f"| Min L3 per (app, graph) | {s['min_l3_per_ag']} |",
             f"| Min rows per app | {s['min_cells_per_app']} |",
             f"| Anchor app | {s['anchor_app']} |",
             f"| Violations | {s['violations']} |",
             ""]

    if s["violations_by_rule"]:
        lines += ["## Violations by rule", "", "| rule | count |", "|---|---|"]
        for k, v in sorted(s["violations_by_rule"].items()):
            lines.append(f"| {k} | {v} |")
        lines.append("")

    lines += ["## Per-app coverage", "",
              "| app | graphs | L3 sizes | policies | rows |",
              "|---|---|---|---|---:|"]
    for r in a["per_app"]:
        lines.append(
            f"| {r['app']} | {len(r['graphs'])} ({','.join(r['graphs'])}) "
            f"| {len(r['l3_sizes'])} ({','.join(r['l3_sizes'])}) "
            f"| {len(r['policies'])} ({','.join(r['policies'])}) "
            f"| {r['row_count']} |"
        )

    lines += ["", "## Per-(app, graph) L3 sweep", "",
              "| app | graph | L3 count | L3 sizes |",
              "|---|---|---:|---|"]
    for r in a["per_app_graph"]:
        lines.append(
            f"| {r['app']} | {r['graph']} | {r['l3_count']} | "
            f"{','.join(r['l3_sizes'])} |"
        )

    if a["violations"]:
        lines += ["", f"## First 30 violations", "",
                  "| rule | detail |", "|---|---|"]
        for v in a["violations"][:30]:
            detail = ", ".join(f"{k}={v[k]}" for k in v if k != "rule")
            lines.append(f"| {v['rule']} | {detail} |")
    else:
        lines += ["", "_No violations._"]
    return "\n".join(lines) + "\n"


def _to_csv(a: dict[str, Any], path: Path) -> None:
    fields = ["app", "graph_count", "l3_count", "policy_count", "row_count",
              "graphs", "l3_sizes", "policies"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in a["per_app"]:
            w.writerow({
                "app":          r["app"],
                "graph_count":  r["graph_count"],
                "l3_count":     r["l3_count"],
                "policy_count": r["policy_count"],
                "row_count":    r["row_count"],
                "graphs":       ",".join(r["graphs"]),
                "l3_sizes":     ",".join(r["l3_sizes"]),
                "policies":     ",".join(r["policies"]),
            })


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
    print(f"[lit-faith-appfreq] {len(s['apps'])} apps over "
          f"{s['corpus_graph_count']} graphs; violations={s['violations']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
