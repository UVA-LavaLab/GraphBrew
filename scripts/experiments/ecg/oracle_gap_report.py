#!/usr/bin/env python3
"""Per-policy oracle-gap report.

For each (graph, app, L3) cell with data for at least two of the
four policies {LRU, SRRIP, GRASP, POPT}, compute an empirical
"oracle" = the minimum miss rate any of the four policies achieved
on that cell. Then project, per policy, the gap

    gap_pp(policy) = (miss_rate(policy) - oracle) * 100

in percentage points. Aggregate gaps by graph family and L3 regime
to answer the paper-grade question: "how much performance is left
on the table relative to choosing the best policy per cell?"

Why an *empirical* oracle (not BELADY):

* BELADY is not currently swept in the lit-faith corpus, and
  re-running every cell with BELADY would add hours of work for
  marginal extra signal.
* The empirical min across 4 production-grade policies is a
  conservative oracle — any future policy that beats it would
  shift the floor.
* The gap therefore quantifies the *headroom* remaining for new
  policies in our current corpus.

Outputs
-------
* ``wiki/data/oracle_gap.csv``  — one row per cell × policy.
* ``wiki/data/oracle_gap.json`` — machine-readable with summary.
* ``wiki/data/oracle_gap.md``   — paper-ready table + breakdown.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

POLICIES = ("LRU", "SRRIP", "GRASP", "POPT")

# Same family mapping used elsewhere in the aggregator suite.
GRAPH_FAMILY = {
    "email-Eu-core":   "social",
    "soc-pokec":       "social",
    "soc-LiveJournal1":"social",
    "com-orkut":       "social",
    "cit-Patents":     "citation",
    "web-Google":      "web",
    "roadNet-CA":      "road",
    "delaunay_n19":    "mesh",
}

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

# Same regime bucketing the policy_winner_table uses.
def _regime(l3: str) -> str:
    b = L3_SIZE_BYTES.get(l3, -1)
    if b < 0:
        return "unknown"
    if b <= 64 * 1024:
        return "tiny"
    if b <= 256 * 1024:
        return "small"
    return "large"


def _load_cells(path: Path) -> dict[tuple[str, str, str], dict[str, float]]:
    """{(graph, app, l3): {policy: miss_rate}}"""
    out: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    with path.open() as f:
        for r in csv.DictReader(f):
            try:
                mr = float(r["miss_rate"])
            except (KeyError, ValueError):
                continue
            if not math.isfinite(mr):
                continue
            pol = (r.get("policy") or "").strip()
            if pol not in POLICIES:
                continue
            key = (r.get("graph", ""), r.get("app", ""), r.get("l3_size", ""))
            out[key][pol] = mr
    return out


def _per_cell(cells: dict) -> list[dict]:
    rows: list[dict] = []
    for key, miss_by_pol in cells.items():
        # Need at least 2 policies for "min across policies" to be
        # meaningful — otherwise gap is vacuously zero.
        if len(miss_by_pol) < 2:
            continue
        graph, app, l3 = key
        fam = GRAPH_FAMILY.get(graph, "unknown")
        if fam == "unknown":
            continue
        oracle = min(miss_by_pol.values())
        for pol in POLICIES:
            mr = miss_by_pol.get(pol)
            if mr is None:
                continue
            gap_pp = (mr - oracle) * 100.0
            rows.append({
                "graph": graph,
                "app":   app,
                "l3_size": l3,
                "family":  fam,
                "regime":  _regime(l3),
                "policy":  pol,
                "miss_rate": f"{mr:.6f}",
                "oracle":    f"{oracle:.6f}",
                "gap_pp":    f"{gap_pp:.3f}",
                "is_winner": "1" if abs(mr - oracle) < 1e-9 else "0",
                "n_policies_in_cell": len(miss_by_pol),
            })
    rows.sort(key=lambda r: (
        r["family"], r["graph"], r["app"],
        L3_SIZE_BYTES.get(r["l3_size"], 0), r["policy"],
    ))
    return rows


def _summarize(rows: list[dict]) -> dict:
    by_policy: dict[str, list[float]] = defaultdict(list)
    by_policy_family: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_policy_regime: dict[tuple[str, str], list[float]] = defaultdict(list)
    win_count: dict[str, int] = defaultdict(int)
    cells_seen: set[tuple[str, str, str]] = set()
    for r in rows:
        gap = float(r["gap_pp"])
        pol = r["policy"]
        by_policy[pol].append(gap)
        by_policy_family[(pol, r["family"])].append(gap)
        by_policy_regime[(pol, r["regime"])].append(gap)
        if r["is_winner"] == "1":
            win_count[pol] += 1
        cells_seen.add((r["graph"], r["app"], r["l3_size"]))

    def _stats(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
        ys = sorted(xs)
        p90 = ys[min(len(ys) - 1, int(round(0.9 * (len(ys) - 1))))]
        return {
            "n": len(xs),
            "mean":   round(statistics.fmean(xs), 4),
            "median": round(statistics.median(xs), 4),
            "p90":    round(p90, 4),
            "max":    round(max(xs), 4),
        }

    return {
        "n_cells": len(cells_seen),
        "n_rows":  len(rows),
        "overall_by_policy": {
            p: {**_stats(by_policy[p]), "wins": win_count[p]}
            for p in POLICIES
        },
        "by_policy_family": {
            f"{p}/{fam}": _stats(xs)
            for (p, fam), xs in sorted(by_policy_family.items())
        },
        "by_policy_regime": {
            f"{p}/{r}": _stats(xs)
            for (p, r), xs in sorted(by_policy_regime.items())
        },
    }


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_json(summary: dict, rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "summary": summary,
        "rows": rows,
    }, indent=2, sort_keys=True) + "\n")


def _write_md(summary: dict, rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Per-policy oracle gap")
    lines.append("")
    lines.append(
        "_Generated by `scripts/experiments/ecg/oracle_gap_report.py` from "
        "`wiki/data/literature_faithfulness_postfix.csv`. Oracle = min "
        "miss rate any of {LRU, SRRIP, GRASP, POPT} achieved on the cell._"
    )
    lines.append("")
    lines.append(
        f"**Cells with ≥2 policies:** {summary['n_cells']}; "
        f"**(cell × policy) rows:** {summary['n_rows']}."
    )
    lines.append("")

    lines.append("## Overall gap to empirical oracle, by policy")
    lines.append("")
    lines.append(
        "| policy | wins | n | mean pp | median pp | p90 pp | max pp |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for pol, s in summary["overall_by_policy"].items():
        lines.append(
            f"| {pol} | {s.get('wins', 0)} | {s['n']} | "
            f"{s['mean']} | {s['median']} | {s['p90']} | {s['max']} |"
        )
    lines.append("")

    lines.append("## Gap by policy × family")
    lines.append("")
    lines.append("| policy / family | n | mean pp | p90 pp | max pp |")
    lines.append("|---|---:|---:|---:|---:|")
    for k, s in summary["by_policy_family"].items():
        lines.append(
            f"| {k} | {s['n']} | {s['mean']} | {s['p90']} | {s['max']} |"
        )
    lines.append("")

    lines.append("## Gap by policy × L3 regime")
    lines.append("")
    lines.append("| policy / regime | n | mean pp | p90 pp | max pp |")
    lines.append("|---|---:|---:|---:|---:|")
    for k, s in summary["by_policy_regime"].items():
        lines.append(
            f"| {k} | {s['n']} | {s['mean']} | {s['p90']} | {s['max']} |"
        )
    lines.append("")

    # Worst gaps per policy (where each loses the most ground).
    lines.append("## Top-10 worst gaps per policy")
    lines.append("")
    for pol in POLICIES:
        worst = sorted(
            (r for r in rows if r["policy"] == pol),
            key=lambda r: -float(r["gap_pp"]),
        )[:10]
        if not worst:
            continue
        lines.append(f"### {pol}")
        lines.append("")
        lines.append("| graph | app | L3 | miss_rate | oracle | gap pp |")
        lines.append("|---|---|---|---:|---:|---:|")
        for r in worst:
            lines.append(
                f"| {r['graph']} | {r['app']} | {r['l3_size']} | "
                f"{r['miss_rate']} | {r['oracle']} | {r['gap_pp']} |"
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
        "--csv-out",
        type=Path,
        default=REPO_ROOT / "wiki" / "data" / "oracle_gap.csv",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "wiki" / "data" / "oracle_gap.json",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=REPO_ROOT / "wiki" / "data" / "oracle_gap.md",
    )
    args = parser.parse_args()

    cells = _load_cells(args.lit_faith_csv)
    rows = _per_cell(cells)
    summary = _summarize(rows)
    _write_csv(rows, args.csv_out)
    _write_json(summary, rows, args.json_out)
    _write_md(summary, rows, args.md_out)
    print(
        f"[oracle-gap] n_cells={summary['n_cells']}, n_rows={summary['n_rows']}, "
        + ", ".join(
            f"{p}_mean={summary['overall_by_policy'][p]['mean']}"
            for p in POLICIES
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
