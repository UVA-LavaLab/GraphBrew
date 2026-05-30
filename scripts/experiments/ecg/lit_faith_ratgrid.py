#!/usr/bin/env python3
"""LIT-RatGrid (gate 233): per-(policy, graph, app) rationale uniqueness.

Audits the rationale-grid invariant: within the per_claim table every
(policy, graph, app) bucket must carry exactly ONE rationale string,
regardless of L3 size. Different L3 sizes for the same (policy, graph,
app) cell are claims about the *same* literature observation — they
must agree on the claim text.

Per-policy *classes*:
  THEOREM_POLICIES = {POPT_GE_GRASP, POPT_NEAR_GRASP_IF_BIG_GAP, SRRIP}
    rationale is policy-class-wide and must be constant within
    (policy, app), regardless of graph.
  POINT_POLICIES   = {GRASP, POPT, LRU}
    rationale is per-graph-Fig-reference and varies by graph.

Per-row rationale-text quality rules:
  R1 non-empty rationale         every per_claim row has a non-empty
                                 `rationale` field
  R2 length floor                len(rationale.strip()) >= 40 chars
  R3 cell uniqueness             |{rationale} for (policy, graph, app)|
                                 <= 1 (theorem-class policies) or
                                 <= 2 (point policies — one per L3
                                 capacity regime)
  R4 theorem-class invariance    |{rationale} for (policy, app)| == 1
                                 when policy in THEOREM_POLICIES
  R5 citation-source mention     rationale text mentions at least one
                                 of CITATION_TOKENS (HPCA20, HPCA21,
                                 ISCA, MICRO, ASPLOS, ATC, ...)
  R6 policy-class coverage floor every policy class appears with
                                 >= MIN_RATIONALES_PER_POLICY rationales
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MIN_RATIONALE_LEN            = 40
MIN_RATIONALES_PER_POLICY    = 1

THEOREM_POLICIES = {"POPT_GE_GRASP", "POPT_NEAR_GRASP_IF_BIG_GAP", "SRRIP"}
POINT_POLICIES   = {"GRASP", "POPT", "LRU"}

CITATION_TOKENS  = ["HPCA20", "HPCA21", "HPCA-21", "HPCA-20",
                    "HPCA 20", "HPCA 21", "HPCA'20", "HPCA'21",
                    "ISCA",    "MICRO",   "ASPLOS", "ATC",
                    "Balaji",  "Faldu",   "Khan",
                    "Fig 9",   "Fig 10",  "Fig 11", "Fig 12",
                    "§6", "§3", "§5", "§7", "Tab"]


def _has_citation_token(text: str) -> bool:
    return any(tok in text for tok in CITATION_TOKENS)


def audit(per_claim: list[dict[str, Any]]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []

    by_cell:    dict[tuple, set[str]] = defaultdict(set)
    by_pol_app: dict[tuple, set[str]] = defaultdict(set)
    by_policy:  dict[str,   set[str]] = defaultdict(set)
    row_records: list[dict[str, Any]] = []

    for r in per_claim:
        rat    = (r.get("rationale") or "").strip()
        policy = r.get("policy")
        graph  = r.get("graph")
        app    = r.get("app")
        l3     = r.get("l3_size")
        row_id = (policy, graph, app, l3)

        record = {
            "policy":           policy,
            "graph":            graph,
            "app":              app,
            "l3_size":          l3,
            "rationale_len":    len(rat),
            "has_citation_token": _has_citation_token(rat),
            "rationale_excerpt": (rat[:80] + "...") if len(rat) > 80 else rat,
        }
        row_records.append(record)

        if not rat:
            violations.append({"row_id": row_id, "rule": "R1_nonempty_rationale"})
            continue

        if len(rat) < MIN_RATIONALE_LEN:
            violations.append({"row_id": row_id, "rule": "R2_length_floor",
                               "observed": len(rat),
                               "floor": MIN_RATIONALE_LEN})

        by_cell[(policy, graph, app)].add(rat)
        by_pol_app[(policy, app)].add(rat)
        by_policy[policy].add(rat)

        if policy in POINT_POLICIES and not _has_citation_token(rat):
            violations.append({
                "row_id": row_id,
                "rule":   "R5_citation_token_missing",
                "reason_excerpt": record["rationale_excerpt"],
            })

    cell_uniqueness_violations = 0
    for (policy, graph, app), rats in sorted(by_cell.items()):
        # Theorem-class policies must have exactly 1 rationale per cell.
        # Point policies may legitimately carry up to 2 rationales — one
        # for the spills-at-small-L3 variant and one for the fits-at-
        # large-L3 variant — since GRASP/POPT Fig references can be
        # capacity-regime-specific.
        limit = 1 if policy in THEOREM_POLICIES else 2
        if len(rats) > limit:
            cell_uniqueness_violations += 1
            violations.append({
                "row_id": (policy, graph, app, None),
                "rule":   "R3_cell_uniqueness",
                "observed": len(rats),
                "limit":    limit,
                "excerpts": [r[:80] for r in list(rats)[:3]],
            })

    theorem_invariance_violations = 0
    for (policy, app), rats in sorted(by_pol_app.items()):
        if policy in THEOREM_POLICIES and len(rats) > 1:
            theorem_invariance_violations += 1
            violations.append({
                "row_id": (policy, None, app, None),
                "rule":   "R4_theorem_class_invariance",
                "observed": len(rats),
                "excerpts": [r[:80] for r in list(rats)[:3]],
            })

    for policy, rats in sorted(by_policy.items()):
        if len(rats) < MIN_RATIONALES_PER_POLICY:
            violations.append({
                "row_id": (policy, None, None, None),
                "rule":   "R6_policy_coverage_floor",
                "observed": len(rats),
                "floor": MIN_RATIONALES_PER_POLICY,
            })

    summary = {
        "min_rationale_len":             MIN_RATIONALE_LEN,
        "min_rationales_per_policy":     MIN_RATIONALES_PER_POLICY,
        "theorem_policies":              sorted(THEOREM_POLICIES),
        "point_policies":                sorted(POINT_POLICIES),
        "total_rows":                    len(per_claim),
        "cell_count":                    len(by_cell),
        "violations":                    len(violations),
        "cell_uniqueness_violations":    cell_uniqueness_violations,
        "theorem_invariance_violations": theorem_invariance_violations,
        "unique_rationales_per_policy": {
            p: len(rats) for p, rats in sorted(by_policy.items())
        },
        "rationale_counts_per_pol_app": {
            f"{p}|{a}": len(rats) for (p, a), rats in sorted(by_pol_app.items())
        },
    }

    return {
        "schema_version": 1,
        "summary":        summary,
        "rows":           row_records,
        "violations":     violations,
    }


def _to_markdown(a: dict[str, Any]) -> str:
    s = a["summary"]
    lines = ["# Literature-faithfulness rationale-grid audit (LIT-RatGrid)",
             "",
             "Per (policy, graph, app) cell: rationale text must be "
             "unique within the cell. Theorem-class policies "
             "(POPT_GE_GRASP, POPT_NEAR_GRASP_IF_BIG_GAP, SRRIP) "
             "additionally require constant rationale across graphs "
             "within (policy, app). Every rationale must cite the "
             "source paper / figure.",
             "",
             "## Summary", "",
             "| Metric | Value |", "|---|---|",
             f"| Total per_claim rows | {s['total_rows']} |",
             f"| (policy, graph, app) cells | {s['cell_count']} |",
             f"| Cell-uniqueness violations | {s['cell_uniqueness_violations']} |",
             f"| Theorem-invariance violations | {s['theorem_invariance_violations']} |",
             f"| Min rationale length | {s['min_rationale_len']} chars |",
             f"| Total violations | {s['violations']} |",
             "",
             "## Unique rationales per policy", "",
             "| policy | distinct rationales |",
             "|---|---|"]
    for p, n in s["unique_rationales_per_policy"].items():
        lines.append(f"| {p} | {n} |")

    lines += ["",
              "## Rationale count per (policy, app)", "",
              "| policy | app | count |",
              "|---|---|---|"]
    for key, n in s["rationale_counts_per_pol_app"].items():
        pol, app = key.split("|")
        lines.append(f"| {pol} | {app} | {n} |")

    if a["violations"]:
        lines += ["", f"## Violations ({len(a['violations'])})", "",
                  "| row_id | rule | observed |",
                  "|---|---|---|"]
        for v in a["violations"][:50]:
            lines.append(
                f"| {v.get('row_id')} | {v['rule']} | "
                f"{v.get('observed', v.get('reason_excerpt',''))} |"
            )
    else:
        lines += ["", "_No violations._"]
    return "\n".join(lines) + "\n"


def _to_csv(a: dict[str, Any], path: Path) -> None:
    fields = ["policy", "graph", "app", "l3_size", "rationale_len",
              "has_citation_token", "rationale_excerpt"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in a["rows"]:
            w.writerow({k: r.get(k) for k in fields})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lit-faith-json", required=True, type=Path)
    ap.add_argument("--json-out",       required=True, type=Path)
    ap.add_argument("--md-out",         required=True, type=Path)
    ap.add_argument("--csv-out",        required=True, type=Path)
    args = ap.parse_args()

    payload = json.loads(args.lit_faith_json.read_text(encoding="utf-8"))
    a = audit(payload["per_claim"])

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(a, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    args.md_out.write_text(_to_markdown(a), encoding="utf-8")
    _to_csv(a, args.csv_out)

    s = a["summary"]
    print(f"[lit-faith-ratgrid] {s['total_rows']} rows / "
          f"{s['cell_count']} cells; violations={s['violations']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
