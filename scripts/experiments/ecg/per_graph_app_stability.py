#!/usr/bin/env python3
"""Per-(graph × app) winner stability across paper L3 sizes (gate 44).

Where gate 39 reports stability per (app), this gate drills down to
(graph × app): does the winner at 1MB survive to 4MB and 8MB on each
specific graph? This pins which cells the paper may quote without a
per-L3 disclaimer and exposes which (graph, app) pairs need a per-L3
breakdown (regime-change cells).

Classification per (graph, app):
  - stable_unique:   exactly one winner that appears at every L3 size
                     present for this (graph, app).
  - stable_partial:  winner sets at each L3 share a non-empty intersection
                     but the intersection is not a single policy (ties).
  - regime_change:   winner sets at different L3 sizes have empty
                     intersection (winner flips between L3 sizes).
  - insufficient:    only one paper L3 size present (cannot establish
                     stability); reported but excluded from headline counts.

Scope: paper L3 sizes (1MB, 4MB, 8MB) per project convention.
Output: wiki/data/per_graph_app_stability.{json,md}
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

PAPER_L3_SIZES = ("1MB", "4MB", "8MB")


def cell_winners(rows: list[dict], scope_l3: set[str]) -> dict[tuple, list[str]]:
    """Per (graph, app, l3) return ALL policies tied for min gap_pp.
    Returns a dict mapping (graph, app, l3) -> sorted list of winner policies."""
    by_cell: dict[tuple, list[tuple[float, str]]] = defaultdict(list)
    for r in rows:
        l3 = r["l3_size"]
        if l3 not in scope_l3:
            continue
        gap = float(r["gap_pp"])
        by_cell[(r["graph"], r["app"], l3)].append((gap, r["policy"]))
    winners: dict[tuple, list[str]] = {}
    for k, pairs in by_cell.items():
        min_gap = min(g for g, _ in pairs)
        # tie tolerance: same as oracle_gap's printed precision (3 decimals)
        tied = sorted(p for g, p in pairs if abs(g - min_gap) < 1e-6)
        winners[k] = tied
    return winners


def classify(winners_per_l3: dict[str, list[str]]) -> dict:
    """winners_per_l3 maps l3_size -> winner-policy list. Returns a
    record describing stability across the L3 sizes present."""
    l3s = sorted(winners_per_l3.keys())
    if len(l3s) < 2:
        return {
            "l3_sizes_present": l3s,
            "winners_by_l3": winners_per_l3,
            "classification": "insufficient_l3",
            "intersection": [],
            "union": sorted({p for ws in winners_per_l3.values() for p in ws}),
            "unique_in_intersection": False,
        }
    intersection = set(winners_per_l3[l3s[0]])
    for l3 in l3s[1:]:
        intersection &= set(winners_per_l3[l3])
    intersection_sorted = sorted(intersection)
    union = sorted({p for ws in winners_per_l3.values() for p in ws})

    if not intersection:
        classification = "regime_change"
    elif len(intersection) == 1 and set(union) == intersection:
        # exactly one policy wins at every L3, no other policy ever ties
        classification = "stable_unique"
    elif len(intersection) == 1:
        # intersection is one policy but other policies tied at some L3
        classification = "stable_unique_with_ties"
    else:
        classification = "stable_partial"

    return {
        "l3_sizes_present": l3s,
        "winners_by_l3": winners_per_l3,
        "classification": classification,
        "intersection": intersection_sorted,
        "union": union,
        "unique_in_intersection": len(intersection) == 1,
    }


def build_payload(oracle_path: Path) -> dict:
    raw = json.loads(oracle_path.read_text())
    rows = raw["rows"]
    winners = cell_winners(rows, set(PAPER_L3_SIZES))

    by_ga: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(dict)
    for (g, a, l), wlist in winners.items():
        by_ga[(g, a)][l] = wlist

    per_ga: list[dict] = []
    for (graph, app) in sorted(by_ga.keys()):
        rec = classify(by_ga[(graph, app)])
        rec["graph"] = graph
        rec["app"] = app
        per_ga.append(rec)

    cls_counts: dict[str, int] = defaultdict(int)
    for r in per_ga:
        cls_counts[r["classification"]] += 1

    # headline lists
    stable_cells = [
        f"{r['graph']}/{r['app']} -> {r['intersection'][0]}"
        for r in per_ga
        if r["classification"] in ("stable_unique", "stable_unique_with_ties")
    ]
    regime_change_cells = [
        f"{r['graph']}/{r['app']}"
        for r in per_ga
        if r["classification"] == "regime_change"
    ]
    partial_cells = [
        f"{r['graph']}/{r['app']} -> {','.join(r['intersection'])}"
        for r in per_ga
        if r["classification"] == "stable_partial"
    ]
    insufficient_cells = [
        f"{r['graph']}/{r['app']}"
        for r in per_ga
        if r["classification"] == "insufficient_l3"
    ]

    # per-graph rollup
    per_graph: dict[str, dict] = defaultdict(
        lambda: {"n_apps": 0, "stable_unique": 0, "regime_change": 0, "partial": 0}
    )
    for r in per_ga:
        rollup = per_graph[r["graph"]]
        rollup["n_apps"] += 1
        if r["classification"] in ("stable_unique", "stable_unique_with_ties"):
            rollup["stable_unique"] += 1
        elif r["classification"] == "regime_change":
            rollup["regime_change"] += 1
        elif r["classification"] == "stable_partial":
            rollup["partial"] += 1

    try:
        src_label = str(oracle_path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        src_label = str(oracle_path)

    payload = {
        "meta": {
            "source": src_label,
            "scope_l3_sizes": list(PAPER_L3_SIZES),
            "n_graph_app_pairs": len(per_ga),
            "n_stable_unique": cls_counts.get("stable_unique", 0)
            + cls_counts.get("stable_unique_with_ties", 0),
            "n_stable_partial": cls_counts.get("stable_partial", 0),
            "n_regime_change": cls_counts.get("regime_change", 0),
            "n_insufficient_l3": cls_counts.get("insufficient_l3", 0),
            "stability_fraction_excl_insufficient": (
                (cls_counts.get("stable_unique", 0)
                 + cls_counts.get("stable_unique_with_ties", 0))
                / max(
                    1,
                    len(per_ga) - cls_counts.get("insufficient_l3", 0),
                )
            ),
        },
        "stable_unique_cells": stable_cells,
        "regime_change_cells": regime_change_cells,
        "stable_partial_cells": partial_cells,
        "insufficient_cells": insufficient_cells,
        "per_graph_rollup": dict(per_graph),
        "per_graph_app": per_ga,
    }
    return payload


def emit_md(payload: dict) -> str:
    meta = payload["meta"]
    out = []
    out.append("# Per-(graph × app) winner stability across L3")
    out.append("")
    out.append(
        f"Source: `{meta['source']}`  •  Scope: {', '.join(meta['scope_l3_sizes'])}"
    )
    out.append("")
    out.append(
        f"Cells: **{meta['n_graph_app_pairs']}** "
        f"(stable-unique {meta['n_stable_unique']}, "
        f"stable-partial {meta['n_stable_partial']}, "
        f"regime-change {meta['n_regime_change']}, "
        f"insufficient-L3 {meta['n_insufficient_l3']})."
    )
    stab_frac = meta["stability_fraction_excl_insufficient"]
    out.append(
        f"Stability fraction (excluding insufficient-L3): "
        f"**{stab_frac * 100:.1f}%**."
    )
    out.append("")
    out.append("## Per-graph rollup")
    out.append("")
    out.append("| graph | n_apps | stable_unique | partial | regime_change |")
    out.append("|---|---:|---:|---:|---:|")
    for graph in sorted(payload["per_graph_rollup"].keys()):
        r = payload["per_graph_rollup"][graph]
        out.append(
            f"| {graph} | {r['n_apps']} | {r['stable_unique']}"
            f" | {r['partial']} | {r['regime_change']} |"
        )
    out.append("")
    out.append("## Stable-unique cells (paper-quotable without per-L3 disclaimer)")
    out.append("")
    if not payload["stable_unique_cells"]:
        out.append("_None._")
    else:
        for line in payload["stable_unique_cells"]:
            out.append(f"- {line}")
    out.append("")
    out.append("## Stable-partial cells (tied across L3)")
    out.append("")
    if not payload["stable_partial_cells"]:
        out.append("_None._")
    else:
        for line in payload["stable_partial_cells"]:
            out.append(f"- {line}")
    out.append("")
    out.append("## Regime-change cells (paper MUST break out per L3)")
    out.append("")
    if not payload["regime_change_cells"]:
        out.append("_None — every cell with multiple L3 sizes has a stable winner._")
    else:
        for line in payload["regime_change_cells"]:
            out.append(f"- {line}")
    out.append("")
    out.append("## Insufficient-L3 cells (only one paper L3 size present)")
    out.append("")
    if not payload["insufficient_cells"]:
        out.append("_None — every cell has ≥2 paper-L3 sizes._")
    else:
        for line in payload["insufficient_cells"]:
            out.append(f"- {line}")
    out.append("")
    out.append("## Full per-(graph, app) table")
    out.append("")
    out.append("| graph | app | L3 sizes | winners per L3 | intersection | classification |")
    out.append("|---|---|---|---|---|---|")
    for r in payload["per_graph_app"]:
        l3s = ",".join(r["l3_sizes_present"])
        winners_str = "; ".join(
            f"{l3}={','.join(r['winners_by_l3'][l3])}" for l3 in r["l3_sizes_present"]
        )
        inter = ",".join(r["intersection"]) if r["intersection"] else "∅"
        out.append(
            f"| {r['graph']} | {r['app']} | {l3s} | {winners_str}"
            f" | {inter} | {r['classification']} |"
        )
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle-json", type=Path, default=WIKI_DATA / "oracle_gap.json"
    )
    parser.add_argument(
        "--json-out", type=Path, default=WIKI_DATA / "per_graph_app_stability.json"
    )
    parser.add_argument(
        "--md-out", type=Path, default=WIKI_DATA / "per_graph_app_stability.md"
    )
    args = parser.parse_args()

    payload = build_payload(args.oracle_json)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(emit_md(payload).rstrip("\n") + "\n")
    meta = payload["meta"]
    print(
        f"per-graph-app-stability: n={meta['n_graph_app_pairs']} "
        f"stable={meta['n_stable_unique']} partial={meta['n_stable_partial']} "
        f"regime={meta['n_regime_change']} insuff={meta['n_insufficient_l3']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
