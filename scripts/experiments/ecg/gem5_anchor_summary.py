#!/usr/bin/env python3
"""gem5 literature-anchor summary.

The GraphBrew cache_sim sweep ("``lit-faith``") drives every confidence
gate, but cache_sim is a functional model — it tracks only address
streams and replacement. The companion gem5 sweep on
``email-Eu-core`` is the **gem5 anchor**: it confirms that the policy
ranking we see in cache_sim is also visible in a cycle-accurate
simulator, on a graph small enough that gem5 can finish in minutes.

This script reads the gem5 sweep directory and emits a compact
machine-readable summary of:

* status counts per (graph, app, l3_size)
* the GRASP / LRU / SRRIP / POPT L3 miss rate at each L3 size
* the headline invariants from the GRASP paper:
    - GRASP <= LRU at L3 = 256kB (PR and BC headline regime)
    - all three policies within 1pp at L3 = 2MB (asymptote)
    - all three policies diverge by >= 2pp at L3 = 4kB
      (L-shape companion: policies must NOT converge below capacity)

The output is consumed by
:mod:`scripts.test.test_gem5_anchor` and surfaced on the confidence
dashboard. The script never re-runs gem5; it only consumes already
materialised ``roi_matrix.csv`` files.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

DEFAULT_SWEEP_ROOT = Path("/tmp/graphbrew-grasp-gem5-sweep")
DEFAULT_SUBDIR = "DBG"

HEADLINE_L3 = "256kB"
ASYMPTOTE_L3 = "2MB"
ASYMPTOTE_MAX_SPREAD_PCT = 1.0
HEADLINE_MAX_GRASP_OVER_LRU_PP = 0.5

# The L-shape companion to the asymptote invariant: at the smallest
# L3 (4kB << working-set), policies must NOT have converged into the
# asymptote regime. We require ≥ 2 × ASYMPTOTE_MAX_SPREAD_PCT of spread
# across {LRU, SRRIP, GRASP} for at least one app at this size.
# Rationale: cache_sim, gem5 and Sniper all show the GRASP-paper
# L-shape (divergent at small L3, convergent at large L3); a flat-
# at-4kB result would indicate a behavioural regression where policies
# stop differentiating in the high-pressure regime.
SMALL_CACHE_L3 = "4kB"
SMALL_CACHE_MIN_SPREAD_PP = 2.0


@dataclass
class CellSummary:
    graph: str
    app: str
    l3_size: str
    miss_rate_by_policy: dict[str, float] = field(default_factory=dict)
    ok_rows: int = 0
    error_rows: int = 0


def _pick_canonical_section(rows: list[dict]) -> dict | None:
    """Mirror sign_consistency / literature_faithfulness: smallest non-zero section if any."""
    if not rows:
        return None
    non_zero = [r for r in rows if int(r.get("section") or 0) != 0]
    if non_zero:
        return min(non_zero, key=lambda r: int(r.get("section") or 0))
    return rows[0]


def load_cells(sweep_root: Path, subdir: str, graphs: set[str] | None = None) -> list[CellSummary]:
    cells: dict[tuple[str, str, str], CellSummary] = {}
    for csv_path in sorted(sweep_root.glob(f"*/{subdir}/roi_matrix.csv")):
        graph_app = csv_path.parent.parent.name
        if "-" not in graph_app:
            continue
        graph, _, app = graph_app.rpartition("-")
        if graphs is not None and graph not in graphs:
            continue
        rows_per_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
        status_per_key: dict[tuple[str, str], list[str]] = defaultdict(list)
        with csv_path.open() as f:
            for r in csv.DictReader(f):
                key = (r.get("l3_size") or "", r.get("policy") or "")
                status_per_key[key].append(r.get("status") or "")
                if (r.get("status") or "") == "ok" and r.get("l3_miss_rate"):
                    rows_per_key[key].append(r)
        for (l3, policy), rows in rows_per_key.items():
            cell_key = (graph, app, l3)
            cell = cells.setdefault(cell_key, CellSummary(graph=graph, app=app, l3_size=l3))
            chosen = _pick_canonical_section(rows)
            if chosen is not None:
                cell.miss_rate_by_policy[policy] = float(chosen["l3_miss_rate"])
        for (l3, _policy), statuses in status_per_key.items():
            cell_key = (graph, app, l3)
            cell = cells.setdefault(cell_key, CellSummary(graph=graph, app=app, l3_size=l3))
            cell.ok_rows += sum(1 for s in statuses if s == "ok")
            cell.error_rows += sum(1 for s in statuses if s and s != "ok")
    return sorted(cells.values(), key=lambda c: (c.graph, c.app, c.l3_size))


def _l3_sort_key(label: str) -> int:
    """Lexicographic sort on L3 labels works poorly; convert to bytes."""
    units = {"kB": 1024, "MB": 1024 * 1024, "B": 1, "GB": 1024 ** 3}
    for unit, mult in units.items():
        if label.endswith(unit):
            try:
                return int(float(label[: -len(unit)]) * mult)
            except ValueError:
                return 0
    return 0


@dataclass
class AnchorInvariant:
    name: str
    status: str        # "ok" / "disagree" / "missing"
    detail: str


def evaluate_invariants(
    cells: list[CellSummary],
    apps: tuple[str, ...] = ("pr", "bc"),
    graphs: tuple[str, ...] = ("email-Eu-core",),
) -> list[AnchorInvariant]:
    results: list[AnchorInvariant] = []
    by_key = {(c.graph, c.app, c.l3_size): c for c in cells}

    for graph in graphs:
        for app in apps:
            c = by_key.get((graph, app, HEADLINE_L3))
            name = f"GRASP_LE_LRU_headline:{graph}/{app}@{HEADLINE_L3}"
            if c is None:
                results.append(AnchorInvariant(name, "missing", "cell not in sweep"))
                continue
            grasp = c.miss_rate_by_policy.get("GRASP")
            lru = c.miss_rate_by_policy.get("LRU")
            if grasp is None or lru is None:
                results.append(AnchorInvariant(name, "missing", f"grasp={grasp} lru={lru}"))
                continue
            delta_pp = (grasp - lru) * 100.0
            if delta_pp <= HEADLINE_MAX_GRASP_OVER_LRU_PP:
                results.append(AnchorInvariant(
                    name, "ok",
                    f"grasp={grasp:.4f} lru={lru:.4f} Δ={delta_pp:+.3f}pp "
                    f"(tolerance ≤ {HEADLINE_MAX_GRASP_OVER_LRU_PP:+.2f}pp)",
                ))
            else:
                results.append(AnchorInvariant(
                    name, "disagree",
                    f"grasp={grasp:.4f} lru={lru:.4f} Δ={delta_pp:+.3f}pp exceeds "
                    f"tolerance ({HEADLINE_MAX_GRASP_OVER_LRU_PP:+.2f}pp)",
                ))

    for graph in graphs:
        for app in apps:
            c = by_key.get((graph, app, ASYMPTOTE_L3))
            name = f"asymptote_within_{ASYMPTOTE_MAX_SPREAD_PCT}pp:{graph}/{app}@{ASYMPTOTE_L3}"
            if c is None:
                results.append(AnchorInvariant(name, "missing", "cell not in sweep"))
                continue
            mrates = {p: r for p, r in c.miss_rate_by_policy.items() if p in {"GRASP", "LRU", "SRRIP"}}
            if len(mrates) < 3:
                results.append(AnchorInvariant(name, "missing", f"have policies={sorted(mrates)}"))
                continue
            spread_pp = (max(mrates.values()) - min(mrates.values())) * 100.0
            if spread_pp <= ASYMPTOTE_MAX_SPREAD_PCT:
                results.append(AnchorInvariant(
                    name, "ok",
                    f"spread={spread_pp:.3f}pp across {sorted(mrates)} "
                    f"(tolerance ≤ {ASYMPTOTE_MAX_SPREAD_PCT:.2f}pp)",
                ))
            else:
                results.append(AnchorInvariant(
                    name, "disagree",
                    f"spread={spread_pp:.3f}pp across {sorted(mrates)} exceeds "
                    f"tolerance ({ASYMPTOTE_MAX_SPREAD_PCT:.2f}pp)",
                ))

    for graph in graphs:
        for app in apps:
            c = by_key.get((graph, app, SMALL_CACHE_L3))
            name = f"small_cache_divergence:{graph}/{app}@{SMALL_CACHE_L3}"
            if c is None:
                results.append(AnchorInvariant(name, "missing", "cell not in sweep"))
                continue
            mrates = {p: r for p, r in c.miss_rate_by_policy.items() if p in {"GRASP", "LRU", "SRRIP"}}
            if len(mrates) < 3:
                results.append(AnchorInvariant(name, "missing", f"have policies={sorted(mrates)}"))
                continue
            spread_pp = (max(mrates.values()) - min(mrates.values())) * 100.0
            if spread_pp >= SMALL_CACHE_MIN_SPREAD_PP:
                results.append(AnchorInvariant(
                    name, "ok",
                    f"spread={spread_pp:.3f}pp across {sorted(mrates)} "
                    f"(min ≥ {SMALL_CACHE_MIN_SPREAD_PP:.2f}pp; L-shape holds)",
                ))
            else:
                results.append(AnchorInvariant(
                    name, "disagree",
                    f"spread={spread_pp:.3f}pp across {sorted(mrates)} below "
                    f"min {SMALL_CACHE_MIN_SPREAD_PP:.2f}pp (policies converged "
                    f"at {SMALL_CACHE_L3}; L-shape broken)",
                ))

    name = "no_error_rows"
    total_err = sum(c.error_rows for c in cells)
    if total_err == 0:
        results.append(AnchorInvariant(
            name, "ok",
            f"{sum(c.ok_rows for c in cells)} ok rows across {len(cells)} cells",
        ))
    else:
        results.append(AnchorInvariant(
            name, "disagree",
            f"{total_err} error rows across {len(cells)} cells",
        ))

    return results


def render_markdown(
    cells: list[CellSummary],
    invariants: list[AnchorInvariant],
    title: str = "gem5 literature anchor",
    sweep_root: Path | None = None,
    sweep_subdir: str | None = None,
) -> str:
    out = [f"# {title}", ""]
    if sweep_root is not None and sweep_subdir is not None:
        out.append(f"Source sweep: `{sweep_root}/<graph>-<app>/{sweep_subdir}`")
    else:
        out.append(f"Source sweep: `{DEFAULT_SWEEP_ROOT}/<graph>-<app>/{DEFAULT_SUBDIR}`")
    out.append("")
    out.append("## Invariants")
    out.append("")
    out.append("| invariant | status | detail |")
    out.append("|---|:---:|---|")
    for inv in invariants:
        icon = {"ok": "✅", "disagree": "❌", "missing": "⚠️"}.get(inv.status, "?")
        out.append(f"| `{inv.name}` | {icon} | {inv.detail} |")
    out.append("")
    out.append("## Per-cell summary")
    out.append("")
    out.append("| graph | app | L3 | LRU | SRRIP | GRASP | POPT | ok | err |")
    out.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for c in sorted(cells, key=lambda c: (c.graph, c.app, _l3_sort_key(c.l3_size))):
        def fmt(p: str) -> str:
            v = c.miss_rate_by_policy.get(p)
            return f"{v:.4f}" if v is not None else "—"
        out.append(
            f"| {c.graph} | {c.app} | {c.l3_size} | "
            f"{fmt('LRU')} | {fmt('SRRIP')} | {fmt('GRASP')} | {fmt('POPT')} | "
            f"{c.ok_rows} | {c.error_rows} |"
        )
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    p.add_argument("--sweep-subdir", default=DEFAULT_SUBDIR)
    p.add_argument(
        "--graphs",
        nargs="+",
        default=["email-Eu-core"],
        help=(
            "Restrict the anchor to these graph names (default: email-Eu-core, "
            "the small graph that gem5 can finish in minutes). Pass additional "
            "graph names once their gem5 sweeps complete."
        ),
    )
    p.add_argument(
        "--apps",
        nargs="+",
        default=["pr", "bc"],
        help=(
            "Apps for which to assert headline + asymptote invariants on "
            "email-Eu-core. Default: pr bc. Pass only 'pr' when running "
            "against a sweep that lacks bc coverage (e.g. the current "
            "Sniper sweep)."
        ),
    )
    p.add_argument("--json-out", type=Path, required=True)
    p.add_argument("--md-out", type=Path)
    p.add_argument(
        "--title",
        default="gem5 literature anchor",
        help="Markdown title for the anchor (override for other simulators).",
    )
    p.add_argument("--exit-on-disagree", action="store_true",
                   help="exit 2 if any invariant is in disagree state")
    args = p.parse_args(argv)

    if not args.sweep_root.exists():
        sys.stderr.write(
            f"gem5 anchor sweep root not found: {args.sweep_root}\n"
            f"  hint: rerun the gem5 sweep for email-Eu-core to materialise it.\n"
        )
        return 2

    cells = load_cells(args.sweep_root, args.sweep_subdir, graphs=set(args.graphs))
    invariants = evaluate_invariants(cells, apps=tuple(args.apps), graphs=tuple(args.graphs))

    payload = {
        "sweep_root": str(args.sweep_root),
        "sweep_subdir": args.sweep_subdir,
        "graphs_scope": sorted(args.graphs),
        "apps_scope": sorted(args.apps),
        "cells": [
            {
                "graph": c.graph, "app": c.app, "l3_size": c.l3_size,
                "miss_rate_by_policy": c.miss_rate_by_policy,
                "ok_rows": c.ok_rows, "error_rows": c.error_rows,
            }
            for c in cells
        ],
        "invariants": [asdict(i) for i in invariants],
        "counts": {
            "cells": len(cells),
            "invariants_ok": sum(1 for i in invariants if i.status == "ok"),
            "invariants_disagree": sum(1 for i in invariants if i.status == "disagree"),
            "invariants_missing": sum(1 for i in invariants if i.status == "missing"),
        },
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(
            cells, invariants,
            title=args.title,
            sweep_root=args.sweep_root,
            sweep_subdir=args.sweep_subdir,
        ))

    print(f"wrote {args.json_out}")
    if args.md_out:
        print(f"wrote {args.md_out}")
    print(
        f"  cells={len(cells)} invariants ok={payload['counts']['invariants_ok']} "
        f"disagree={payload['counts']['invariants_disagree']} "
        f"missing={payload['counts']['invariants_missing']}"
    )

    if args.exit_on_disagree and payload["counts"]["invariants_disagree"] > 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
