#!/usr/bin/env python3
"""WSS-relative L3 axis aggregator.

The paper currently reports L3 sizes in absolute bytes (4kB ... 8MB)
and bins them into three coarse regimes (tiny/small/large). That makes
cross-graph comparisons unfair: 1 MB is "tiny" for cit-Patents
(working set ~5.7 MB) and "huge" for email-Eu-core (working set
~4 KB).

This aggregator joins each oracle-gap cell with the graph's
working-set proxy (`working_set_ratio` from corpus_diversity.json,
which itself is WSS_at_1MB_PR / 1MB), computes the L3/WSS ratio, and
re-bins cells into three WSS-relative regimes:

  under_wss  L3/WSS < 0.25   ("cache much smaller than working set")
  near_wss   0.25 ≤ L3/WSS ≤ 4    ("inflection zone")
  over_wss   L3/WSS > 4      ("cache comfortably holds working set")

For each (policy, wss_regime) it tabulates {n_cells, n_wins, mean
oracle gap}.  This is what tells the paper "POPT wins X% of cells
when the cache is small relative to working set; GRASP wins Y% when
the cache fits comfortably" -- a far stronger statement than the
absolute-byte-axis equivalent.

Caveat: `working_set_ratio` is computed from the PR run only. We use
it as a per-graph proxy; per-kernel refinement is future work.

Inputs:
  wiki/data/oracle_gap.json
  wiki/data/corpus_diversity.json

Outputs:
  wiki/data/wss_relative_l3.{json,md}
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WIKI = REPO_ROOT / "wiki" / "data"

DEFAULT_ORACLE_JSON = WIKI / "oracle_gap.json"
DEFAULT_CORPUS_JSON = WIKI / "corpus_diversity.json"
DEFAULT_JSON_OUT = WIKI / "wss_relative_l3.json"
DEFAULT_MD_OUT = WIKI / "wss_relative_l3.md"

# Same byte map used by the policy_winner_table / popt_vs_grasp_report.
L3_SIZE_BYTES = {
    "4kB":   4 * 1024,
    "8kB":   8 * 1024,
    "16kB":  16 * 1024,
    "32kB":  32 * 1024,
    "64kB":  64 * 1024,
    "128kB": 128 * 1024,
    "256kB": 256 * 1024,
    "512kB": 512 * 1024,
    "1MB":   1 * 1024 * 1024,
    "2MB":   2 * 1024 * 1024,
    "4MB":   4 * 1024 * 1024,
    "8MB":   8 * 1024 * 1024,
    "16MB": 16 * 1024 * 1024,
}

WSS_REFERENCE_L3 = 1 * 1024 * 1024  # working_set_ratio is WSS / 1MB

POLICIES = ("LRU", "SRRIP", "GRASP", "POPT")


def _wss_regime(ratio: float) -> str:
    if ratio < 0.25:
        return "under_wss"
    if ratio > 4.0:
        return "over_wss"
    return "near_wss"


def _load_wss_map(corpus_json: Path) -> dict[str, float]:
    """Returns graph -> WSS bytes proxy."""
    data = json.loads(corpus_json.read_text())
    if isinstance(data, dict):
        data = data.get("graphs", []) or data.get("rows", [])
    out = {}
    for entry in data:
        graph = entry.get("graph")
        wsr = (entry.get("features") or {}).get("working_set_ratio")
        if graph is not None and wsr is not None:
            out[graph] = float(wsr) * WSS_REFERENCE_L3
    return out


def _aggregate(rows, wss_map: dict[str, float]):
    by_policy_regime: dict[tuple[str, str], list[float]] = defaultdict(list)
    cells_with_winner = defaultdict(set)
    wins_by_policy_regime: dict[tuple[str, str], int] = defaultdict(int)
    unknown_graphs = set()
    skipped = 0

    # First: assemble per-cell (graph,app,l3_size) -> oracle (min gap row)
    cells: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["graph"], r["app"], r["l3_size"])
        cells[key].append(r)

    for (graph, app, l3_size), pol_rows in cells.items():
        wss_b = wss_map.get(graph)
        if wss_b is None:
            unknown_graphs.add(graph)
            skipped += len(pol_rows)
            continue
        l3_b = L3_SIZE_BYTES.get(l3_size)
        if l3_b is None:
            skipped += len(pol_rows)
            continue
        ratio = l3_b / wss_b if wss_b > 0 else None
        regime = _wss_regime(ratio) if ratio is not None else None
        if regime is None:
            skipped += len(pol_rows)
            continue
        # the winner is the policy with the smallest gap_pp in this cell
        try:
            winner = min(pol_rows, key=lambda r: float(r["gap_pp"]))["policy"]
        except Exception:
            winner = None
        cells_with_winner[regime].add((graph, app, l3_size))
        if winner:
            wins_by_policy_regime[(winner, regime)] += 1
        for r in pol_rows:
            pol = r["policy"]
            if pol not in POLICIES:
                continue
            try:
                gap = float(r["gap_pp"])
            except (TypeError, ValueError):
                continue
            by_policy_regime[(pol, regime)].append(gap)

    summary = {}
    for pol in POLICIES:
        for regime in ("under_wss", "near_wss", "over_wss"):
            vals = by_policy_regime.get((pol, regime), [])
            n_cells = len(cells_with_winner.get(regime, set()))
            wins = wins_by_policy_regime.get((pol, regime), 0)
            summary[f"{pol}/{regime}"] = {
                "policy":     pol,
                "wss_regime": regime,
                "n":          len(vals),
                "mean_gap_pp": round(statistics.fmean(vals), 4) if vals else None,
                "median_gap_pp": round(statistics.median(vals), 4) if vals else None,
                "p90_gap_pp": round(statistics.quantiles(vals, n=10)[-1], 4) if len(vals) >= 10 else None,
                "n_cells_in_regime": n_cells,
                "wins":       wins,
                "win_rate":   round(wins / n_cells, 4) if n_cells else None,
            }
    return summary, sorted(unknown_graphs), skipped


def _per_regime_ranking(summary: dict) -> dict:
    out = {}
    for regime in ("under_wss", "near_wss", "over_wss"):
        ranked = []
        for pol in POLICIES:
            r = summary.get(f"{pol}/{regime}", {})
            ranked.append({
                "policy":      pol,
                "n":           r.get("n", 0),
                "mean_gap_pp": r.get("mean_gap_pp"),
                "win_rate":    r.get("win_rate"),
                "wins":        r.get("wins", 0),
            })
        # rank by mean_gap_pp ascending (None last)
        ranked.sort(key=lambda x: (x["mean_gap_pp"] is None, x["mean_gap_pp"] or 0))
        out[regime] = ranked
    return out


def _emit_md(doc, path: Path):
    lines = [
        "# WSS-relative L3 axis",
        "",
        "_Generated by `scripts/experiments/ecg/wss_relative_l3.py`. "
        "Bins each (graph, app, L3) cell by L3 / WSS ratio (WSS proxy = "
        "`working_set_ratio` from corpus_diversity × 1 MB) and re-aggregates "
        "winners + oracle gaps per (policy, wss_regime). This is the "
        "paper's defense against \"absolute L3 bytes obscure cross-graph "
        "comparisons\" pushback._",
        "",
        "Regime cuts: `under_wss` (L3/WSS < 0.25), `near_wss` (0.25 ≤ L3/WSS ≤ 4), `over_wss` (L3/WSS > 4).",
        "",
        f"**Cells classified:** {doc['meta']['n_cells_classified']} "
        f"(skipped {doc['meta']['n_cells_skipped']} due to missing WSS or L3 lookup).",
        "",
        "## Per-regime winner ranking (mean gap to per-cell empirical oracle)",
        "",
    ]
    for regime, ranking in doc["per_regime_ranking"].items():
        cells = doc["per_regime_cell_count"].get(regime, 0)
        lines += [
            f"### `{regime}` ({cells} cells)",
            "",
            "| rank | policy | n | mean gap (pp) | wins | win rate |",
            "|---:|---|---:|---:|---:|---:|",
        ]
        for i, row in enumerate(ranking, 1):
            mg = f"{row['mean_gap_pp']:.3f}" if row["mean_gap_pp"] is not None else "n/a"
            wr = f"{row['win_rate']:.3f}" if row["win_rate"] is not None else "n/a"
            crown = " 🥇" if i == 1 else ""
            lines.append(
                f"| {i}{crown} | `{row['policy']}` | {row['n']} | {mg} | {row['wins']} | {wr} |"
            )
        lines.append("")
    lines += [
        "## Full (policy, wss_regime) summary",
        "",
        "| policy | wss_regime | n | mean gap (pp) | wins | win rate |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, v in doc["by_policy_regime"].items():
        mg = f"{v['mean_gap_pp']:.3f}" if v["mean_gap_pp"] is not None else "n/a"
        wr = f"{v['win_rate']:.3f}" if v["win_rate"] is not None else "n/a"
        lines.append(
            f"| `{v['policy']}` | `{v['wss_regime']}` | {v['n']} | {mg} | {v['wins']} | {wr} |"
        )
    lines += ["", "## Per-graph WSS proxies used", ""]
    lines += ["| graph | WSS proxy (bytes) | WSS proxy (MB) |"]
    lines.append("|---|---:|---:|")
    for g, wss in sorted(doc["meta"]["wss_proxies"].items()):
        lines.append(f"| `{g}` | {wss:,.0f} | {wss/1048576:.3f} |")
    lines.append("")
    if doc["meta"]["unknown_graphs"]:
        lines += [
            "## Graphs without WSS proxy (skipped)",
            "",
            *(f"- `{g}`" for g in doc["meta"]["unknown_graphs"]),
            "",
        ]
    path.write_text("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-json", type=Path, default=DEFAULT_ORACLE_JSON)
    ap.add_argument("--corpus-json", type=Path, default=DEFAULT_CORPUS_JSON)
    ap.add_argument("--json-out",    type=Path, default=DEFAULT_JSON_OUT)
    ap.add_argument("--md-out",      type=Path, default=DEFAULT_MD_OUT)
    ns = ap.parse_args(argv)

    rows = json.loads(ns.oracle_json.read_text()).get("rows", [])
    wss_map = _load_wss_map(ns.corpus_json)
    if not wss_map:
        raise SystemExit(
            f"[wss-l3] {ns.corpus_json} has no working_set_ratio entries"
        )

    summary, unknown_graphs, n_skipped = _aggregate(rows, wss_map)
    ranking = _per_regime_ranking(summary)
    # cell counts per regime
    cell_counts = {}
    for regime in ("under_wss", "near_wss", "over_wss"):
        # any non-None summary entry will have the same cell count
        cell_counts[regime] = next(
            (summary[k]["n_cells_in_regime"] for k in summary
             if k.endswith("/" + regime)), 0,
        )

    n_cells_total = sum(cell_counts.values())

    doc = {
        "meta": {
            "n_cells_classified": n_cells_total,
            "n_cells_skipped":    n_skipped,
            "wss_reference_bytes": WSS_REFERENCE_L3,
            "wss_proxies":         {g: round(b, 2) for g, b in wss_map.items()},
            "unknown_graphs":      unknown_graphs,
        },
        "by_policy_regime":      summary,
        "per_regime_ranking":    ranking,
        "per_regime_cell_count": cell_counts,
    }

    ns.json_out.parent.mkdir(parents=True, exist_ok=True)
    ns.json_out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    _emit_md(doc, ns.md_out)

    md_path = Path(ns.md_out).resolve()
    try:
        md_display = md_path.relative_to(REPO_ROOT)
    except ValueError:
        md_display = md_path
    print(
        f"[wss-l3] {n_cells_total} cells classified "
        f"(under={cell_counts['under_wss']}, near={cell_counts['near_wss']}, "
        f"over={cell_counts['over_wss']}, skipped={n_skipped}) → {md_display}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
