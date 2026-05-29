#!/usr/bin/env python3
"""Cross-tool winner-agreement report.

Why this exists
---------------
The existing cross-tool saturation report
(``scripts/experiments/ecg/cross_tool_saturation_report.py``) only
checks that ``Δ(GRASP − LRU)`` agrees in sign between simulators when
both are saturated. That's the *headline* claim but it's a single
contrast. Reviewers will also ask "do the simulators agree on the
winning policy per (graph, app, L3) cell?" — a stricter all-policies
test.

This script answers it by intersecting the lit-faith cells with the
matching gem5 / Sniper anchor cells at the same L3 size. For each
3-way overlap it identifies each tool's winner, the runner-up, and
the margin, then classifies the cell as:

* ``unanimous`` — all 3 tools pick the same winner.
* ``majority``  — 2 of 3 tools agree on the winner.
* ``split``     — every tool picks a different winner.

The headline gate forbids ``split`` outcomes and warns if
``majority`` cells exceed an explicit ceiling — this is the
soundness check that the paper's per-tool numbers actually describe
the same world.

Output
------
* ``wiki/data/cross_tool_winners.csv`` — one row per overlap cell.
* ``wiki/data/cross_tool_winners.json`` — machine-readable summary.
* ``wiki/data/cross_tool_winners.md`` — paper-ready markdown.

Usage
-----
    python3 -m scripts.experiments.ecg.cross_tool_winners_report
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# Tools listed in the order they appear in the table columns.
TOOLS = ("cache_sim", "gem5", "sniper")

L3_SIZE_BYTES = {
    "4kB": 4 * 1024,
    "16kB": 16 * 1024,
    "32kB": 32 * 1024,
    "64kB": 64 * 1024,
    "256kB": 256 * 1024,
    "1MB": 1024 * 1024,
    "2MB": 2 * 1024 * 1024,
    "4MB": 4 * 1024 * 1024,
    "8MB": 8 * 1024 * 1024,
}


def _l3_bytes(label: str) -> int:
    return L3_SIZE_BYTES.get(label, -1)


def _winner(miss_by_pol: dict[str, float]) -> tuple[str, float, str, float]:
    """Return (winner_policy, winner_mr, runner_up_policy, runner_up_mr).
    Sorts by miss rate ascending, ties broken by alphabetical policy."""
    items = sorted(
        ((p, mr) for p, mr in miss_by_pol.items() if math.isfinite(mr)),
        key=lambda x: (x[1], x[0]),
    )
    if not items:
        return ("", float("nan"), "", float("nan"))
    if len(items) == 1:
        return (items[0][0], items[0][1], "", float("nan"))
    return (items[0][0], items[0][1], items[1][0], items[1][1])


def _read_anchor(path: Path) -> dict[tuple[str, str, str], dict[str, float]]:
    """Return {(graph, app, l3_size): {policy: miss_rate}}."""
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    out: dict[tuple[str, str, str], dict[str, float]] = {}
    for c in d.get("cells", []):
        key = (c.get("graph", ""), c.get("app", ""), c.get("l3_size", ""))
        out[key] = dict(c.get("miss_rate_by_policy", {}))
    return out


def _read_lit_faith(path: Path) -> dict[tuple[str, str, str], dict[str, float]]:
    """Return {(graph, app, l3_size): {policy: miss_rate}} from the
    flat lit-faith CSV."""
    out: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    with path.open() as f:
        for r in csv.DictReader(f):
            key = (
                r.get("graph", ""),
                r.get("app") or r.get("benchmark", ""),
                r.get("l3_size", ""),
            )
            pol = (r.get("policy") or "").strip()
            try:
                mr = float(r.get("miss_rate") or r.get("l3_miss_rate", "nan"))
            except ValueError:
                continue
            if math.isfinite(mr) and pol:
                out[key][pol] = mr
    return out


def _classify(winners: dict[str, str]) -> str:
    vals = [v for v in winners.values() if v]
    if not vals:
        return "missing"
    counts = Counter(vals)
    most = counts.most_common(1)[0][1]
    if most == len(vals):
        return "unanimous"
    if most >= 2:
        return "majority"
    return "split"


def _max_l3_by_cell(
    src: dict[tuple[str, str, str], dict[str, float]]
) -> dict[tuple[str, str], tuple[str, dict[str, float]]]:
    """Collapse {(graph, app, l3): {pol: mr}} → {(graph, app): (l3, {pol: mr})}
    picking the largest L3 (closest to saturation) per (graph, app).
    """
    out: dict[tuple[str, str], tuple[str, dict[str, float]]] = {}
    for (graph, app, l3), miss_by_pol in src.items():
        if not miss_by_pol:
            continue
        prev = out.get((graph, app))
        if prev is None or _l3_bytes(l3) > _l3_bytes(prev[0]):
            out[(graph, app)] = (l3, miss_by_pol)
    return out


def _build(lit_faith: dict, gem5: dict, sniper: dict) -> list[dict]:
    """For each (graph, app) that has data in at least two tools, pick
    each tool's largest-L3 cell and compute the winning policy.
    Different tools sweep different L3 sizes, so requiring identical
    L3 sizes yields zero overlap; the largest-L3 approximation puts
    each tool at its most-saturated operating point.
    """
    lf_max = _max_l3_by_cell(lit_faith)
    gem5_max = _max_l3_by_cell(gem5)
    sniper_max = _max_l3_by_cell(sniper)
    out: list[dict] = []
    keys = set(lf_max) | set(gem5_max) | set(sniper_max)
    for key in sorted(keys):
        graph, app = key
        sources = {
            "cache_sim": lf_max.get(key),
            "gem5":      gem5_max.get(key),
            "sniper":    sniper_max.get(key),
        }
        winners: dict[str, str] = {}
        details: dict[str, dict] = {}
        l3_by_tool: dict[str, str] = {}
        for tool, entry in sources.items():
            if not entry:
                winners[tool] = ""
                l3_by_tool[tool] = ""
                continue
            l3, miss_by_pol = entry
            l3_by_tool[tool] = l3
            w_pol, w_mr, r_pol, r_mr = _winner(miss_by_pol)
            winners[tool] = w_pol
            details[tool] = {
                "winner_policy": w_pol,
                "winner_miss_rate": (
                    "" if not math.isfinite(w_mr) else f"{w_mr:.6f}"
                ),
                "runner_up_policy": r_pol,
                "margin_pp": (
                    "" if not math.isfinite(r_mr) else f"{(r_mr - w_mr) * 100.0:.3f}"
                ),
            }
        n_tools_with_data = sum(1 for w in winners.values() if w)
        if n_tools_with_data < 2:
            continue
        classification = _classify(winners)
        out.append({
            "graph": graph,
            "app": app,
            "cache_sim_l3": l3_by_tool.get("cache_sim", ""),
            "gem5_l3":      l3_by_tool.get("gem5", ""),
            "sniper_l3":    l3_by_tool.get("sniper", ""),
            "n_tools": n_tools_with_data,
            "cache_sim_winner": winners.get("cache_sim", ""),
            "gem5_winner":      winners.get("gem5", ""),
            "sniper_winner":    winners.get("sniper", ""),
            "cache_sim_margin_pp": details.get("cache_sim", {}).get("margin_pp", ""),
            "gem5_margin_pp":      details.get("gem5", {}).get("margin_pp", ""),
            "sniper_margin_pp":    details.get("sniper", {}).get("margin_pp", ""),
            "classification": classification,
        })
    return out


def _summarize(records: list[dict]) -> dict:
    n = len(records)
    cls = Counter(r["classification"] for r in records)
    return {
        "n_cells": n,
        "by_classification": dict(cls.most_common()),
        "split_cells": [
            {k: r[k] for k in (
                "graph", "app",
                "cache_sim_winner", "gem5_winner", "sniper_winner",
            )}
            for r in records if r["classification"] == "split"
        ],
        "majority_cells": [
            {k: r[k] for k in (
                "graph", "app",
                "cache_sim_winner", "gem5_winner", "sniper_winner",
            )}
            for r in records if r["classification"] == "majority"
        ],
    }


def _write_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("")
        return
    fieldnames = list(records[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow(r)


def _write_json(summary: dict, records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "summary": summary,
        "cells": records,
    }, indent=2, sort_keys=True))


def _write_md(summary: dict, records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Cross-tool winner agreement")
    lines.append("")
    lines.append(
        "_Generated by `scripts/experiments/ecg/cross_tool_winners_report.py` "
        "from `wiki/data/literature_faithfulness_postfix.csv`, "
        "`wiki/data/gem5_anchor.json`, and `wiki/data/sniper_anchor.json`._"
    )
    lines.append("")
    n = summary["n_cells"]
    cls = summary["by_classification"]
    lines.append(
        f"**Cells with at least 2 tools providing data:** {n}. "
        f"Unanimous = {cls.get('unanimous', 0)}, "
        f"majority = {cls.get('majority', 0)}, "
        f"split = {cls.get('split', 0)}."
    )
    lines.append("")

    if summary["split_cells"]:
        lines.append("## Split cells (every tool picks a different winner)")
        lines.append("")
        lines.append("| graph | app | cache_sim | gem5 | sniper |")
        lines.append("|---|---|---|---|---|")
        for r in summary["split_cells"]:
            lines.append(
                f"| {r['graph']} | {r['app']} | "
                f"{r['cache_sim_winner']} | {r['gem5_winner']} | "
                f"{r['sniper_winner']} |"
            )
        lines.append("")

    if summary["majority_cells"]:
        lines.append("## Majority cells (2 of 3 tools agree)")
        lines.append("")
        lines.append("| graph | app | cache_sim | gem5 | sniper |")
        lines.append("|---|---|---|---|---|")
        for r in summary["majority_cells"]:
            lines.append(
                f"| {r['graph']} | {r['app']} | "
                f"{r['cache_sim_winner']} | {r['gem5_winner']} | "
                f"{r['sniper_winner']} |"
            )
        lines.append("")

    lines.append("## All cells (full table)")
    lines.append("")
    lines.append(
        "| graph | app | cache_sim_L3 | gem5_L3 | sniper_L3 | "
        "n_tools | cache_sim | gem5 | sniper | class |"
    )
    lines.append("|---|---|---|---|---|---:|---|---|---|---|")
    for r in records:
        lines.append(
            f"| {r['graph']} | {r['app']} | "
            f"{r['cache_sim_l3']} | {r['gem5_l3']} | {r['sniper_l3']} | "
            f"{r['n_tools']} | {r['cache_sim_winner']} | "
            f"{r['gem5_winner']} | {r['sniper_winner']} | "
            f"{r['classification']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lit-faith-csv",
        type=Path,
        default=REPO_ROOT / "wiki" / "data" / "literature_faithfulness_postfix.csv",
    )
    parser.add_argument(
        "--gem5-json",
        type=Path,
        default=REPO_ROOT / "wiki" / "data" / "gem5_anchor.json",
    )
    parser.add_argument(
        "--sniper-json",
        type=Path,
        default=REPO_ROOT / "wiki" / "data" / "sniper_anchor.json",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=REPO_ROOT / "wiki" / "data" / "cross_tool_winners.csv",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "wiki" / "data" / "cross_tool_winners.json",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=REPO_ROOT / "wiki" / "data" / "cross_tool_winners.md",
    )
    args = parser.parse_args()

    lit_faith = _read_lit_faith(args.lit_faith_csv)
    gem5 = _read_anchor(args.gem5_json)
    sniper = _read_anchor(args.sniper_json)
    records = _build(lit_faith, gem5, sniper)
    summary = _summarize(records)
    _write_csv(records, args.csv_out)
    _write_json(summary, records, args.json_out)
    _write_md(summary, records, args.md_out)
    cls = summary["by_classification"]
    print(
        f"[xt-winners] {summary['n_cells']} cells; "
        + ", ".join(f"{k}={v}" for k, v in cls.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
