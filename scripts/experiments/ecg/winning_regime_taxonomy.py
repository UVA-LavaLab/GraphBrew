#!/usr/bin/env python3
"""Winning-regime taxonomy: empirical rules for "when does each policy win?".

This is the paper's headline-figure aggregator. It joins three
sources of truth:

* ``wiki/data/policy_winner_table.json`` — one row per
  (graph, app, L3) cell with the winning policy and margin.
* ``wiki/data/corpus_diversity.json`` — per-graph structural
  features (hub_concentration, clustering_coeff, avg_degree,
  working_set_ratio, …).
* The L3 regime label already attached to each winner cell
  (``small`` / ``medium`` / ``large`` — bucketed by the winner-table
  script).

It then *bins* graphs by structural family (matching the corpus
diversity scheme: road / mesh / social / web / citation), buckets
each cell into a (family × regime) cell, and reports the empirical
winner distribution + sample size. The deliverable answers — in one
table — questions the reviewer will ask:

* "When does GRASP help? When does it regress?"
* "Is POPT's road-graph dominance an artifact of one graph or a
  family-wide rule?"
* "Do small caches always favor LRU?"

It also extracts *minimal rules*: for each (family, regime) cell
where one policy wins ≥80 % of the time, it emits a textual rule
the paper can quote ("on road-family graphs at L3 ≥ 1 MB, POPT
wins 100 % of cells (n=20)").

Output
------
* ``wiki/data/winning_regime_taxonomy.csv``  — one row per
  (family, regime, policy) tally.
* ``wiki/data/winning_regime_taxonomy.json`` — machine-readable.
* ``wiki/data/winning_regime_taxonomy.md``   — paper-ready figure
  (matrix table + extracted rules).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# Same mapping the corpus_diversity / claim_density reports use.
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

# Regime ordering for stable rendering (tiny → large). The winner-
# table script buckets L3 sizes into {tiny, small, large}; `medium`
# is reserved for future use but currently has zero rows.
REGIME_ORDER = ("tiny", "small", "medium", "large")

# Closed policy set. Anything else surfaces as "OTHER" and trips
# the test gate (catches a typo'd policy column upstream).
KNOWN_POLICIES = ("LRU", "SRRIP", "GRASP", "POPT")

# Confidence threshold (fraction) above which a (family, regime)
# cell yields a quotable "X wins" rule.
RULE_THRESHOLD = 0.80


def _load_winners(path: Path) -> list[dict]:
    d = json.loads(path.read_text())
    return list(d.get("cells", []))


def _load_features(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in json.loads(path.read_text()):
        g = row.get("graph", "")
        feats = dict(row.get("features", {}))
        # Promote nodes/edges into features dict for convenience.
        feats["nodes"] = row.get("nodes")
        feats["edges"] = row.get("edges")
        out[g] = feats
    return out


def _family(graph: str) -> str:
    return GRAPH_FAMILY.get(graph, "unknown")


def _bucket_winners(cells: list[dict]) -> tuple[dict, dict]:
    """Return (matrix, totals) where:

    matrix[(family, regime)] = Counter({policy: count})
    totals[(family, regime)] = total_count
    """
    matrix: dict[tuple[str, str], Counter] = defaultdict(Counter)
    totals: dict[tuple[str, str], int] = defaultdict(int)
    for c in cells:
        fam = _family(c.get("graph", ""))
        if fam == "unknown":
            continue
        regime = c.get("l3_regime", "")
        if regime not in REGIME_ORDER:
            continue
        pol = c.get("winner_policy", "")
        if not pol:
            continue
        bucket = pol if pol in KNOWN_POLICIES else "OTHER"
        matrix[(fam, regime)][bucket] += 1
        totals[(fam, regime)] += 1
    return matrix, totals


def _extract_rules(matrix: dict, totals: dict, threshold: float) -> list[dict]:
    rules: list[dict] = []
    for key, counts in matrix.items():
        fam, regime = key
        total = totals[key]
        if total == 0:
            continue
        for pol, n in counts.most_common():
            frac = n / total
            if frac >= threshold:
                rules.append({
                    "family": fam,
                    "regime": regime,
                    "winner": pol,
                    "wins": n,
                    "sample_size": total,
                    "fraction": round(frac, 6),
                    "rule_text": (
                        f"on {fam}-family graphs at L3 regime "
                        f"\"{regime}\", {pol} wins {n}/{total} cells "
                        f"({frac * 100.0:.1f}%)"
                    ),
                })
                break  # only emit the dominant winner per cell
    rules.sort(key=lambda r: (r["family"], REGIME_ORDER.index(r["regime"])))
    return rules


def _per_cell_table(cells: list[dict], feats: dict) -> list[dict]:
    """Flat (family, regime, graph, app, L3, winner, margin, features...)
    table — useful as the raw csv backing the figure."""
    out: list[dict] = []
    for c in cells:
        g = c.get("graph", "")
        fam = _family(g)
        f = feats.get(g, {})
        out.append({
            "family":    fam,
            "regime":    c.get("l3_regime", ""),
            "graph":     g,
            "app":       c.get("app", ""),
            "l3_size":   c.get("l3_size", ""),
            "winner":    c.get("winner_policy", ""),
            "runner_up": c.get("runner_up_policy", ""),
            "margin_pp": c.get("margin_pp", ""),
            "hub_concentration": (
                f"{f.get('hub_concentration', float('nan')):.6f}"
                if f.get("hub_concentration") is not None
                else ""
            ),
            "clustering_coeff": (
                f"{f.get('clustering_coeff', float('nan')):.6f}"
                if f.get("clustering_coeff") is not None
                else ""
            ),
            "avg_degree": (
                f"{f.get('avg_degree', float('nan')):.6f}"
                if f.get("avg_degree") is not None
                else ""
            ),
        })
    out.sort(key=lambda r: (r["family"], r["graph"], r["app"], r["l3_size"]))
    return out


def _summarize(
    matrix: dict,
    totals: dict,
    rules: list[dict],
    flat: list[dict],
) -> dict:
    by_family_regime: list[dict] = []
    for fam in sorted({k[0] for k in matrix}):
        for regime in REGIME_ORDER:
            key = (fam, regime)
            if key not in matrix:
                continue
            counts = matrix[key]
            total = totals[key]
            entry = {
                "family": fam,
                "regime": regime,
                "total": total,
            }
            for pol in KNOWN_POLICIES:
                entry[f"{pol}_wins"] = counts.get(pol, 0)
                entry[f"{pol}_share"] = (
                    round(counts.get(pol, 0) / total, 6)
                    if total else 0.0
                )
            entry["OTHER_wins"] = counts.get("OTHER", 0)
            by_family_regime.append(entry)
    overall_winner_counts = Counter(r["winner"] for r in flat if r["winner"])
    return {
        "n_cells": len(flat),
        "n_family_regime_bins": len(by_family_regime),
        "by_family_regime": by_family_regime,
        "rules": rules,
        "overall_winner_counts": dict(overall_winner_counts.most_common()),
        "rule_threshold": RULE_THRESHOLD,
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


def _write_json(summary: dict, flat: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "summary": summary,
        "cells": flat,
    }, indent=2, sort_keys=True))


def _write_md(summary: dict, flat: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Winning-regime taxonomy")
    lines.append("")
    lines.append(
        "_Generated by `scripts/experiments/ecg/winning_regime_taxonomy.py` "
        "from `wiki/data/policy_winner_table.json` and "
        "`wiki/data/corpus_diversity.json`._"
    )
    lines.append("")
    lines.append(
        f"**Cells classified:** {summary['n_cells']}; "
        f"**(family, regime) bins:** {summary['n_family_regime_bins']}; "
        f"**rules extracted at ≥{RULE_THRESHOLD * 100:.0f}% threshold:** "
        f"{len(summary['rules'])}."
    )
    lines.append("")
    lines.append("## Overall winner distribution")
    lines.append("")
    lines.append("| policy | wins |")
    lines.append("|---|---:|")
    for pol, n in summary["overall_winner_counts"].items():
        lines.append(f"| {pol} | {n} |")
    lines.append("")

    lines.append("## Family × regime matrix")
    lines.append("")
    lines.append(
        "| family | regime | n | "
        + " | ".join(KNOWN_POLICIES)
        + " | OTHER |"
    )
    lines.append("|---|---|---:|" + "---:|" * len(KNOWN_POLICIES) + "---:|")
    for row in summary["by_family_regime"]:
        cells = " | ".join(
            f"{row[f'{p}_wins']} ({row[f'{p}_share'] * 100:.0f}%)"
            for p in KNOWN_POLICIES
        )
        lines.append(
            f"| {row['family']} | {row['regime']} | {row['total']} | "
            f"{cells} | {row['OTHER_wins']} |"
        )
    lines.append("")

    lines.append("## Extracted rules (≥ 80% dominance)")
    lines.append("")
    if not summary["rules"]:
        lines.append("_No (family, regime) bin reached the 80% threshold._")
        lines.append("")
    else:
        for r in summary["rules"]:
            lines.append(f"- {r['rule_text']}")
        lines.append("")

    lines.append("## All cells (raw)")
    lines.append("")
    lines.append(
        "| family | regime | graph | app | L3 | winner | margin pp | "
        "hub | clust | avg_deg |"
    )
    lines.append("|---|---|---|---|---|---|---:|---:|---:|---:|")
    for r in flat:
        lines.append(
            f"| {r['family']} | {r['regime']} | {r['graph']} | "
            f"{r['app']} | {r['l3_size']} | {r['winner']} | "
            f"{r['margin_pp']} | {r['hub_concentration']} | "
            f"{r['clustering_coeff']} | {r['avg_degree']} |"
        )
    path.write_text("\n".join(lines).rstrip("\n") + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--winners-json",
        type=Path,
        default=REPO_ROOT / "wiki" / "data" / "policy_winner_table.json",
    )
    parser.add_argument(
        "--corpus-json",
        type=Path,
        default=REPO_ROOT / "wiki" / "data" / "corpus_diversity.json",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=REPO_ROOT / "wiki" / "data" / "winning_regime_taxonomy.csv",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "wiki" / "data" / "winning_regime_taxonomy.json",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=REPO_ROOT / "wiki" / "data" / "winning_regime_taxonomy.md",
    )
    args = parser.parse_args()

    cells = _load_winners(args.winners_json)
    feats = _load_features(args.corpus_json)
    matrix, totals = _bucket_winners(cells)
    rules = _extract_rules(matrix, totals, RULE_THRESHOLD)
    flat = _per_cell_table(cells, feats)
    summary = _summarize(matrix, totals, rules, flat)
    _write_csv(flat, args.csv_out)
    _write_json(summary, flat, args.json_out)
    _write_md(summary, flat, args.md_out)
    print(
        f"[regime-tax] n_cells={summary['n_cells']}, "
        f"bins={summary['n_family_regime_bins']}, "
        f"rules={len(summary['rules'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
