#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Headline parity proof tool (gate 283).

PROOF GATE — paper-grade cross-simulator headline table.

For every literature cell (graph, app, l3) where ≥2 simulators report
results, compute the per-sim winner policy (lowest miss_rate) and the
cross-sim winner-agreement verdict:

  agree     — all reporting sims pick the same winner policy.
  disagree  — at least 2 sims pick different winner policies.
  single    — only 1 sim reports (no comparison possible).
  empty     — no sim reports for this cell.

Emits the paper-table preview joining the 3 sims side-by-side:

  | graph | app | L3 | sim | LRU | SRRIP | GRASP | POPT | ECG | winner |

Gate semantics (RATCHET-COMPATIBLE):
  - The test asserts (disagreements / cells_with_overlap) <=
    max_disagreement_ratio, defaulting to 0.0 (no disagreements allowed).
  - When only 1 sim covers a cell, no disagreement is logged — the
    cell waits for a second sim's measurement.
  - When 0 sims cover a cell, the cell is reported as missing (gate
    282's concern, not gate 283's).

This is the paper headline table generator — even sparse coverage
produces useful output for the paper draft.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
LIT_BASELINES = REPO_ROOT / "scripts/experiments/ecg/literature_baselines.py"


# ---------------------------------------------------------------------------
# Sim roster + policy order (consistent with gate 282)
# ---------------------------------------------------------------------------

SIMS = ("cache_sim", "gem5", "sniper")
POLICY_ORDER = ("LRU", "SRRIP", "GRASP", "POPT", "ECG_DBG_PRIMARY")


# ---------------------------------------------------------------------------
# Literature scope (mirrors gate 282 — derived from gate 281 registry)
# ---------------------------------------------------------------------------

def _load_literature_baselines():
    if "literature_baselines" in sys.modules:
        return sys.modules["literature_baselines"]
    spec = importlib.util.spec_from_file_location(
        "literature_baselines", LIT_BASELINES
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["literature_baselines"] = mod
    spec.loader.exec_module(mod)
    return mod


def literature_concrete_cells(l3_filter: tuple[str, ...] | None = None
                              ) -> set[tuple[str, str, str]]:
    lit = _load_literature_baselines()
    all_claims = list(lit.INVARIANT_CLAIMS)
    if hasattr(lit, "PER_GRAPH_CLAIMS"):
        all_claims.extend(lit.PER_GRAPH_CLAIMS)
    cells = set()
    for c in all_claims:
        if c.graph.startswith("*"):
            continue
        if l3_filter is not None and c.l3_size not in l3_filter:
            continue
        cells.add((c.graph, c.app, c.l3_size))
    return cells


# ---------------------------------------------------------------------------
# Per-cell measurement view
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellMeasurement:
    """Per (sim, graph, app, l3) cell: miss-rate by policy + winner."""
    sim: str
    graph: str
    app: str
    l3_size: str
    miss_by_policy: dict[str, float]

    @property
    def winner(self) -> str | None:
        if not self.miss_by_policy:
            return None
        # Lowest miss rate wins; deterministic tiebreak by policy order.
        sorted_pols = sorted(
            self.miss_by_policy.items(),
            key=lambda kv: (kv[1], POLICY_ORDER.index(kv[0])
                            if kv[0] in POLICY_ORDER else 999, kv[0])
        )
        return sorted_pols[0][0]


def load_cache_sim_measurements(csv_path: Path) -> list[CellMeasurement]:
    """Load cache_sim lit-faith CSV; group by (graph, app, l3_size)."""
    out: list[CellMeasurement] = []
    if not csv_path.exists():
        return out
    grouped: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    with csv_path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            graph = (r.get("graph") or "").strip()
            app = (r.get("app") or "").strip()
            l3 = (r.get("l3_size") or "").strip()
            pol = (r.get("policy") or "").strip()
            miss = (r.get("miss_rate") or "").strip()
            if not (graph and app and l3 and pol and miss):
                continue
            try:
                grouped[(graph, app, l3)][pol] = float(miss)
            except ValueError:
                continue
    for (g, a, l), miss_by in grouped.items():
        out.append(CellMeasurement("cache_sim", g, a, l, dict(miss_by)))
    return out


def load_anchor_measurements(json_path: Path,
                              sim_name: str) -> list[CellMeasurement]:
    """Load gem5/sniper anchor JSON cells."""
    out: list[CellMeasurement] = []
    if not json_path.exists():
        return out
    try:
        d = json.loads(json_path.read_text())
    except Exception:
        return out
    for c in d.get("cells", []):
        graph = (c.get("graph") or "").strip()
        app = (c.get("app") or "").strip()
        l3 = (c.get("l3_size") or "").strip()
        miss_by = {}
        for pol, miss in (c.get("miss_rate_by_policy") or {}).items():
            try:
                miss_by[pol] = float(miss)
            except (ValueError, TypeError):
                continue
        if graph and app and l3 and miss_by:
            out.append(CellMeasurement(sim_name, graph, app, l3, miss_by))
    return out


def load_all_measurements(cache_sim_csv: Path, gem5_json: Path,
                           sniper_json: Path) -> list[CellMeasurement]:
    return (load_cache_sim_measurements(cache_sim_csv)
            + load_anchor_measurements(gem5_json, "gem5")
            + load_anchor_measurements(sniper_json, "sniper"))


# ---------------------------------------------------------------------------
# Headline table & parity verdict
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HeadlineRow:
    graph: str
    app: str
    l3_size: str
    per_sim_winner: dict[str, str]  # sim -> winner policy (or "—")
    per_sim_miss: dict[str, dict[str, float]]  # sim -> policy -> miss_rate
    verdict: str  # agree / disagree / single / empty


def compute_headline_table(measurements: list[CellMeasurement],
                            scope_l3: tuple[str, ...] | None = ("1MB",)
                            ) -> list[HeadlineRow]:
    """Join measurements across sims into per-cell headline rows."""
    lit_cells = literature_concrete_cells(l3_filter=scope_l3)
    by_cell: dict[tuple[str, str, str], dict[str, CellMeasurement]] = defaultdict(dict)
    for m in measurements:
        key = (m.graph, m.app, m.l3_size)
        if key in lit_cells:
            by_cell[key][m.sim] = m

    rows: list[HeadlineRow] = []
    for cell in sorted(lit_cells):
        sim_meas = by_cell.get(cell, {})
        per_sim_winner: dict[str, str] = {}
        per_sim_miss: dict[str, dict[str, float]] = {}
        for sim in SIMS:
            if sim in sim_meas:
                per_sim_winner[sim] = sim_meas[sim].winner or "—"
                per_sim_miss[sim] = dict(sim_meas[sim].miss_by_policy)
            else:
                per_sim_winner[sim] = "—"
                per_sim_miss[sim] = {}

        reporting = [s for s in SIMS if per_sim_winner[s] != "—"]
        if not reporting:
            verdict = "empty"
        elif len(reporting) == 1:
            verdict = "single"
        else:
            winners = {per_sim_winner[s] for s in reporting}
            verdict = "agree" if len(winners) == 1 else "disagree"
        rows.append(HeadlineRow(cell[0], cell[1], cell[2],
                                  per_sim_winner, per_sim_miss, verdict))
    return rows


def summarise(rows: list[HeadlineRow]) -> dict[str, Any]:
    by_verdict = defaultdict(int)
    for r in rows:
        by_verdict[r.verdict] += 1
    overlap_total = by_verdict.get("agree", 0) + by_verdict.get("disagree", 0)
    return {
        "cells_total": len(rows),
        "cells_with_overlap": overlap_total,
        "cells_agree": by_verdict.get("agree", 0),
        "cells_disagree": by_verdict.get("disagree", 0),
        "cells_single_sim": by_verdict.get("single", 0),
        "cells_empty": by_verdict.get("empty", 0),
        "winner_agreement_pct": (
            round(100.0 * by_verdict.get("agree", 0) / overlap_total, 2)
            if overlap_total else None
        ),
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_json(path: Path, *, scope_l3: tuple[str, ...] | None,
               rows: list[HeadlineRow], summary: dict) -> None:
    payload = {
        "gate": 283,
        "scope_l3": list(scope_l3) if scope_l3 else None,
        "status": "active",
        "summary": summary,
        "rows": [
            {
                "graph": r.graph, "app": r.app, "l3_size": r.l3_size,
                "per_sim_winner": dict(r.per_sim_winner),
                "per_sim_miss": {s: dict(mv) for s, mv in r.per_sim_miss.items()},
                "verdict": r.verdict,
            }
            for r in rows
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _miss_cell(d: dict[str, float], pol: str) -> str:
    v = d.get(pol)
    return f"{v:.4f}" if v is not None else "—"


def write_md(path: Path, *, scope_l3: tuple[str, ...] | None,
             rows: list[HeadlineRow], summary: dict) -> None:
    lines = []
    scope_str = "+".join(scope_l3) if scope_l3 else "all_L3"
    lines.append(f"# Headline parity proof (gate 283)")
    lines.append("")
    lines.append(f"Cross-simulator headline table at literature L3 scope "
                 f"**{scope_str}**, joining cache_sim + gem5 + Sniper.")
    lines.append("")
    lines.append("## Verdict summary")
    lines.append("")
    lines.append(f"- Cells total: **{summary['cells_total']}**")
    lines.append(f"- Cells with ≥2 sims reporting: **{summary['cells_with_overlap']}**")
    lines.append(f"- ✅ Agree: **{summary['cells_agree']}**")
    lines.append(f"- ❌ Disagree: **{summary['cells_disagree']}**")
    lines.append(f"- 🟡 Single sim only: **{summary['cells_single_sim']}**")
    lines.append(f"- ⚪ Empty: **{summary['cells_empty']}**")
    agree_pct = summary.get("winner_agreement_pct")
    if agree_pct is not None:
        lines.append(f"- Winner agreement (where comparable): "
                     f"**{agree_pct:.1f} %**")
    else:
        lines.append(f"- Winner agreement: **not yet measurable** "
                     f"(0 cells with multi-sim coverage)")
    lines.append("")
    lines.append("## Per-cell winner across sims")
    lines.append("")
    lines.append("| graph | app | L3 | cache_sim | gem5 | Sniper | verdict |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        v_icon = {"agree": "✅", "disagree": "❌",
                  "single": "🟡", "empty": "⚪"}[r.verdict]
        lines.append(f"| `{r.graph}` | {r.app} | {r.l3_size} | "
                     f"{r.per_sim_winner['cache_sim']} | "
                     f"{r.per_sim_winner['gem5']} | "
                     f"{r.per_sim_winner['sniper']} | "
                     f"{v_icon} {r.verdict} |")
    lines.append("")
    lines.append("## Per-cell per-sim miss-rate (paper-table preview)")
    lines.append("")
    pols = list(POLICY_ORDER)
    for r in rows:
        if r.verdict == "empty":
            continue
        lines.append(f"### `{r.graph}` / {r.app} / {r.l3_size}")
        lines.append("")
        header = "| sim | " + " | ".join(pols) + " | winner |"
        sep = "|---|" + "|".join(["---:"] * len(pols)) + "|---|"
        lines.append(header)
        lines.append(sep)
        for sim in SIMS:
            miss = r.per_sim_miss.get(sim, {})
            if not miss:
                continue
            cells = " | ".join(_miss_cell(miss, p) for p in pols)
            lines.append(f"| `{sim}` | {cells} | "
                         f"**{r.per_sim_winner[sim]}** |")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    while lines and lines[-1] == "":
        lines.pop()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, *, rows: list[HeadlineRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        header = ["graph", "app", "l3_size", "sim"] + list(POLICY_ORDER) + \
                 ["winner", "verdict"]
        w.writerow(header)
        for r in rows:
            for sim in SIMS:
                miss = r.per_sim_miss.get(sim, {})
                if not miss:
                    continue
                row = [r.graph, r.app, r.l3_size, sim]
                row.extend(_miss_cell(miss, p) for p in POLICY_ORDER)
                row.append(r.per_sim_winner[sim])
                row.append(r.verdict)
                w.writerow(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cache-sim-csv", type=Path,
                   default=REPO_ROOT / "wiki/data/literature_faithfulness_postfix.csv")
    p.add_argument("--gem5-anchor", type=Path,
                   default=REPO_ROOT / "wiki/data/gem5_anchor.json")
    p.add_argument("--sniper-anchor", type=Path,
                   default=REPO_ROOT / "wiki/data/sniper_anchor.json")
    p.add_argument("--json-out", type=Path,
                   default=REPO_ROOT / "wiki/data/headline_parity.json")
    p.add_argument("--md-out", type=Path,
                   default=REPO_ROOT / "wiki/data/headline_parity.md")
    p.add_argument("--csv-out", type=Path,
                   default=REPO_ROOT / "wiki/data/headline_parity.csv")
    p.add_argument("--scope-l3", default="1MB",
                   help="Comma-separated L3 sizes to include "
                        "(default: 1MB; use 'all' for no filter)")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.scope_l3.lower() == "all":
        scope_l3 = None
    else:
        scope_l3 = tuple(s.strip() for s in args.scope_l3.split(",") if s.strip())
    measurements = load_all_measurements(args.cache_sim_csv, args.gem5_anchor,
                                           args.sniper_anchor)
    rows = compute_headline_table(measurements, scope_l3=scope_l3)
    summary = summarise(rows)
    write_json(args.json_out, scope_l3=scope_l3, rows=rows, summary=summary)
    write_md(args.md_out, scope_l3=scope_l3, rows=rows, summary=summary)
    write_csv(args.csv_out, rows=rows)
    if not args.quiet:
        scope_str = "+".join(scope_l3) if scope_l3 else "all_L3"
        print(f"[headline-parity] scope_l3={scope_str} "
              f"cells={summary['cells_total']} "
              f"overlap={summary['cells_with_overlap']} "
              f"agree={summary['cells_agree']} "
              f"disagree={summary['cells_disagree']} "
              f"single={summary['cells_single_sim']} "
              f"empty={summary['cells_empty']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
