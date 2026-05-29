#!/usr/bin/env python3
"""Build a paper-grade *Policy Winner Table* from the literature-
faithfulness CSV.

Why this exists
---------------
Reviewers want a single artifact that answers "for which (graph, app,
L3 size) does each replacement policy win, and by how much?". The
literature-faithfulness comparator records every cell's per-policy
miss rate; this script projects that onto a winner-per-cell view and
counts wins per policy and per (graph_family, L3 regime).

Output
------
* ``wiki/data/policy_winner_table.csv`` — one row per (graph, app,
  l3_size) with the winning policy, runner-up, and margin.
* ``wiki/data/policy_winner_table.json`` — machine-readable summary
  including per-policy win counts and per-family/regime tallies.
* ``wiki/data/policy_winner_table.md`` — paper-ready markdown with the
  full table and two summary tables.

Usage
-----
    python3 -m scripts.experiments.ecg.policy_winner_table \\
        --lit-faith-csv wiki/data/literature_faithfulness_postfix.csv \\
        --corpus-json   wiki/data/corpus_diversity.json \\
        --csv-out       wiki/data/policy_winner_table.csv \\
        --json-out      wiki/data/policy_winner_table.json \\
        --md-out        wiki/data/policy_winner_table.md
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]

L3_SIZE_BYTES = {
    "4kB": 4 * 1024,
    "16kB": 16 * 1024,
    "64kB": 64 * 1024,
    "256kB": 256 * 1024,
    "1MB": 1024 * 1024,
    "2MB": 2 * 1024 * 1024,
    "4MB": 4 * 1024 * 1024,
    "8MB": 8 * 1024 * 1024,
}


def _l3_bytes(label: str) -> int:
    return L3_SIZE_BYTES.get(label, -1)


def _l3_regime(label: str) -> str:
    """Bucket an L3 size into a coarse regime.

    * tiny   = < 64 kB  (working set always overflows)
    * small  = [64 kB, 1 MB)  (transition zone)
    * large  = >= 1 MB  (large enough for headline curves)
    """
    b = _l3_bytes(label)
    if b < 0:
        return "unknown"
    if b < 64 * 1024:
        return "tiny"
    if b < 1024 * 1024:
        return "small"
    return "large"


def _load_corpus(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    # corpus_diversity.json is currently a top-level list of per-graph
    # objects, but accept the older {"graphs": [...]} shape too.
    graphs: list[dict]
    if isinstance(data, list):
        graphs = data
    elif isinstance(data, dict):
        graphs = data.get("graphs", []) or []
    else:
        graphs = []
    by_name: dict[str, dict] = {}
    for c in graphs:
        g = c.get("graph") or c.get("name")
        if g:
            by_name[g] = c
    return by_name


# Graph family is a corpus-level attribute that drives the per-family
# breakdown. Kept in sync with scripts/test/test_corpus_diversity_floor.py
# (GRAPH_FAMILY) so the row labels mean the same thing across the
# corpus-diversity gate and the policy-winner artifacts.
GRAPH_FAMILY: dict[str, str] = {
    "email-Eu-core": "social",
    "web-Google": "web",
    "cit-Patents": "citation",
    "soc-pokec": "social",
    "soc-LiveJournal1": "social",
    "com-orkut": "social",
    "roadNet-CA": "road",
    "delaunay_n19": "mesh",
    "road-CA": "road",
    "twitter-2010": "social",
    "uk-2005": "web",
}


def _graph_family(graph: str, corpus_entry: dict | None) -> str:
    # Prefer an explicit corpus-level tag if it appears one day, but
    # fall back to the hard-coded map so the table stays sensible.
    if corpus_entry:
        for key in ("family", "category"):
            v = corpus_entry.get(key)
            if v:
                return v
    return GRAPH_FAMILY.get(graph, "unknown")


def _read_lit_faith(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _winner_rows(
    rows: Iterable[dict], corpus: dict[str, dict]
) -> list[dict]:
    """Group the lit-faith rows by (graph, app, l3_size); for each group
    return one record with the winning policy (= lowest miss_rate),
    the runner-up policy, the margin in pp, and a coarse L3 regime.
    """
    grouped: dict[tuple[str, str, str], list[tuple[str, float]]] = defaultdict(list)
    for r in rows:
        graph = r.get("graph") or ""
        app = r.get("app") or r.get("benchmark") or ""
        l3 = r.get("l3_size") or ""
        pol = r.get("policy") or ""
        try:
            mr = float(r.get("miss_rate") or r.get("l3_miss_rate") or "nan")
        except ValueError:
            continue
        if mr != mr:  # NaN
            continue
        grouped[(graph, app, l3)].append((pol, mr))

    out: list[dict] = []
    for (graph, app, l3), pol_mr in sorted(grouped.items(), key=lambda kv: (
        kv[0][0], kv[0][1], _l3_bytes(kv[0][2]), kv[0][2],
    )):
        # Stable sort by miss-rate (lowest first); break ties by policy
        # name so the output is deterministic.
        pol_mr_sorted = sorted(pol_mr, key=lambda x: (x[1], x[0]))
        if not pol_mr_sorted:
            continue
        winner_pol, winner_mr = pol_mr_sorted[0]
        if len(pol_mr_sorted) >= 2:
            runner_pol, runner_mr = pol_mr_sorted[1]
            margin_pp = (runner_mr - winner_mr) * 100.0
        else:
            runner_pol, runner_mr, margin_pp = "", float("nan"), float("nan")
        out.append({
            "graph": graph,
            "graph_family": _graph_family(graph, corpus.get(graph)),
            "app": app,
            "l3_size": l3,
            "l3_regime": _l3_regime(l3),
            "winner_policy": winner_pol,
            "winner_miss_rate": f"{winner_mr:.6f}",
            "runner_up_policy": runner_pol,
            "runner_up_miss_rate": "" if runner_mr != runner_mr else f"{runner_mr:.6f}",
            "margin_pp": "" if margin_pp != margin_pp else f"{margin_pp:.3f}",
            "n_policies": len(pol_mr_sorted),
        })
    return out


def _summarize(records: list[dict]) -> dict:
    by_policy = Counter(r["winner_policy"] for r in records)
    by_family = defaultdict(Counter)
    by_regime = defaultdict(Counter)
    by_app = defaultdict(Counter)
    fragile = []  # winner-by-less-than-0.5pp cells
    for r in records:
        by_family[r["graph_family"]][r["winner_policy"]] += 1
        by_regime[r["l3_regime"]][r["winner_policy"]] += 1
        by_app[r["app"]][r["winner_policy"]] += 1
        try:
            margin = float(r["margin_pp"])
        except (TypeError, ValueError):
            margin = float("nan")
        if margin == margin and margin < 0.5:
            fragile.append((margin, r))
    fragile_top = sorted(fragile, key=lambda x: x[0])[:5]
    return {
        "n_cells": len(records),
        "wins_by_policy": dict(by_policy.most_common()),
        "wins_by_family": {
            fam: dict(c.most_common()) for fam, c in sorted(by_family.items())
        },
        "wins_by_regime": {
            reg: dict(c.most_common()) for reg, c in sorted(by_regime.items())
        },
        "wins_by_app": {
            app: dict(c.most_common()) for app, c in sorted(by_app.items())
        },
        "fragile_top_5": [
            {"margin_pp": f"{m:.3f}", **{k: r[k] for k in (
                "graph", "app", "l3_size", "winner_policy", "runner_up_policy",
            )}} for m, r in fragile_top
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
    lines = ["# Policy winner table", ""]
    lines.append(
        "_Generated by `scripts/experiments/ecg/policy_winner_table.py` from "
        "`wiki/data/literature_faithfulness_postfix.csv`._"
    )
    lines.append("")
    lines.append(
        f"**Total (graph, app, L3) cells with ≥2 policies:** "
        f"{summary['n_cells']}"
    )
    lines.append("")

    lines.append("## Wins by policy")
    lines.append("")
    lines.append("| policy | wins | share |")
    lines.append("|---|---:|---:|")
    n = max(1, summary["n_cells"])
    for pol, count in summary["wins_by_policy"].items():
        lines.append(f"| {pol} | {count} | {count/n*100:.1f}% |")
    lines.append("")

    lines.append("## Wins by graph family")
    lines.append("")
    lines.append("| family | LRU | SRRIP | GRASP | POPT | POPT_CHARGED | other |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    known = ("LRU", "SRRIP", "GRASP", "POPT", "POPT_CHARGED")
    for fam, counts in summary["wins_by_family"].items():
        cells = [str(counts.get(p, 0)) for p in known]
        other = sum(v for k, v in counts.items() if k not in known)
        lines.append(f"| {fam} | " + " | ".join(cells) + f" | {other} |")
    lines.append("")

    lines.append("## Wins by L3 regime")
    lines.append("")
    lines.append("| regime | LRU | SRRIP | GRASP | POPT | POPT_CHARGED | other |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for reg in ("tiny", "small", "large", "unknown"):
        counts = summary["wins_by_regime"].get(reg, {})
        if not counts:
            continue
        cells = [str(counts.get(p, 0)) for p in known]
        other = sum(v for k, v in counts.items() if k not in known)
        lines.append(f"| {reg} | " + " | ".join(cells) + f" | {other} |")
    lines.append("")

    if summary["fragile_top_5"]:
        lines.append("## 5 most fragile cells (winner margin < 0.5pp)")
        lines.append("")
        lines.append(
            "| graph | app | L3 | winner | runner-up | margin pp |"
        )
        lines.append("|---|---|---|---|---|---:|")
        for r in summary["fragile_top_5"]:
            lines.append(
                f"| {r['graph']} | {r['app']} | {r['l3_size']} | "
                f"{r['winner_policy']} | {r['runner_up_policy']} | "
                f"{r['margin_pp']} |"
            )
        lines.append("")

    lines.append("## Per-cell winners (full table)")
    lines.append("")
    lines.append(
        "| graph | family | app | L3 | regime | winner | "
        "miss rate | runner-up | margin pp |"
    )
    lines.append("|---|---|---|---|---|---|---:|---|---:|")
    for r in records:
        lines.append(
            f"| {r['graph']} | {r['graph_family']} | {r['app']} | "
            f"{r['l3_size']} | {r['l3_regime']} | {r['winner_policy']} | "
            f"{r['winner_miss_rate']} | {r['runner_up_policy']} | "
            f"{r['margin_pp']} |"
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
        "--corpus-json",
        type=Path,
        default=REPO_ROOT / "wiki" / "data" / "corpus_diversity.json",
    )
    parser.add_argument("--csv-out", type=Path,
                        default=REPO_ROOT / "wiki" / "data" / "policy_winner_table.csv")
    parser.add_argument("--json-out", type=Path,
                        default=REPO_ROOT / "wiki" / "data" / "policy_winner_table.json")
    parser.add_argument("--md-out", type=Path,
                        default=REPO_ROOT / "wiki" / "data" / "policy_winner_table.md")
    args = parser.parse_args()

    if not args.lit_faith_csv.exists():
        raise SystemExit(
            f"missing lit-faith CSV at {args.lit_faith_csv}; "
            "run `make lit-faith` first."
        )

    rows = _read_lit_faith(args.lit_faith_csv)
    corpus = _load_corpus(args.corpus_json)
    records = _winner_rows(rows, corpus)
    summary = _summarize(records)
    _write_csv(records, args.csv_out)
    _write_json(summary, records, args.json_out)
    _write_md(summary, records, args.md_out)
    print(
        f"[policy-winner] {summary['n_cells']} cells; "
        f"top winner = {next(iter(summary['wins_by_policy']), 'N/A')} "
        f"({next(iter(summary['wins_by_policy'].values()), 0)} wins)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
