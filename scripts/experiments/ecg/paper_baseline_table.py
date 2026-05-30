#!/usr/bin/env python3
"""Paper-ready baseline table generator.

Reads ``roi_matrix.csv`` files from a sweep root (the same layout consumed
by :mod:`literature_faithfulness`) and produces a cross-tabulated table
suitable for direct paste into the paper baselines section:

    rows    = (graph, app, L3 size)
    columns = LRU miss-rate, SRRIP Δ, GRASP Δ, POPT Δ
              + literature claim verdict for GRASP and POPT when available

Δ is reported in percentage points (lower is better) versus the LRU baseline
measured at the same (graph, app, L3) tuple.

Outputs:
    --markdown OUT.md  -- a Markdown table with one row per
                          (graph, app, L3) tuple.
    --csv OUT.csv      -- the same data in CSV form.
    --json OUT.json    -- machine-readable per-row dump.

The Δ vs LRU values mirror those used by ``literature_faithfulness`` so the
paper table and the lit-gate verdict cannot disagree on miss-rate values.
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
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent

# Reuse the comparator's loader so we agree on section-picking, status
# filtering, and column names.
_lf_spec = importlib.util.spec_from_file_location(
    "literature_faithfulness", HERE / "literature_faithfulness.py"
)
_lf = importlib.util.module_from_spec(_lf_spec)
sys.modules["literature_faithfulness"] = _lf
_lf_spec.loader.exec_module(_lf)  # type: ignore[union-attr]

_lit_spec = importlib.util.spec_from_file_location(
    "literature_baselines", HERE / "literature_baselines.py"
)
_lit = importlib.util.module_from_spec(_lit_spec)
sys.modules["literature_baselines"] = _lit
_lit_spec.loader.exec_module(_lit)  # type: ignore[union-attr]


POLICY_COLS = ["LRU", "SRRIP", "GRASP", "POPT"]


@dataclass(frozen=True)
class Row:
    graph: str
    app: str
    l3_size: str
    miss: dict[str, float]              # policy -> miss_rate
    delta: dict[str, float]             # policy -> Δ vs LRU pp (LRU=0)
    verdict: dict[str, str]             # policy -> verdict label (or "")
    accesses: int


def build_rows(obs_idx: dict, claims_idx: dict,
               min_accesses: int) -> list[Row]:
    """Group observations by (graph, app, l3); attach claim verdicts."""
    grouped: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(dict)
    for (g, a, l, p), o in obs_idx.items():
        grouped[(g, a, l)][p] = o
    out: list[Row] = []
    for (g, a, l), policy_map in sorted(grouped.items()):
        lru = policy_map.get("LRU")
        miss: dict[str, float] = {}
        delta: dict[str, float] = {}
        verdict: dict[str, str] = {}
        for pol in POLICY_COLS:
            o = policy_map.get(pol)
            if o is None:
                continue
            miss[pol] = o.miss_rate
            if lru is not None and pol != "LRU":
                delta[pol] = (o.miss_rate - lru.miss_rate) * 100.0
            elif pol == "LRU":
                delta[pol] = 0.0
            # Attach claim verdict if any literature claim applies.
            for claim in claims_idx.get((g, a, l), []):
                if claim.policy != pol:
                    continue
                v = _verdict_for(o, lru, claim, min_accesses)
                verdict[pol] = v
                break
        accesses = max((policy_map[p].accesses for p in policy_map), default=0)
        out.append(Row(g, a, l, miss, delta, verdict, accesses))
    return out


def _verdict_for(o, lru, claim, min_accesses: int) -> str:
    """Re-derive the per-claim verdict label using the comparator helpers."""
    if lru is None:
        return "no_lru"
    if o.accesses < min_accesses:
        return "insufficient"
    delta_pp = (o.miss_rate - lru.miss_rate) * 100.0
    lo = claim.min_abs_delta_pct if claim.min_abs_delta_pct is not None else 0.0
    hi = claim.max_abs_delta_pct if claim.max_abs_delta_pct is not None else float("inf")
    tol = claim.tolerance_pct
    sign = claim.expected_sign
    sign_ok = (sign == "~") or (
        (sign == "-" and delta_pp <= 0)
        or (sign == "+" and delta_pp >= 0)
    )
    if not sign_ok and abs(delta_pp) > tol:
        return "DISAGREE"
    mag = abs(delta_pp)
    if mag < lo - tol:
        return "DISAGREE"
    if mag > hi + tol:
        return "DISAGREE"
    if mag < lo or mag > hi:
        return "within_tol"
    return "ok"


def _index_claims() -> dict[tuple[str, str, str], list[Any]]:
    out: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    for c in _lit.PER_GRAPH_CLAIMS:
        # Skip pseudo-policy / cross-policy invariants - they don't apply to
        # a single observed (graph, app, L3, policy) row.
        if c.policy in {"POPT_GE_GRASP", "POPT_NEAR_GRASP_IF_BIG_GAP"}:
            continue
        out[(c.graph, c.app, c.l3_size)].append(c)
    return out


def format_markdown(rows: list[Row]) -> str:
    lines = []
    lines.append("# Paper baseline table")
    lines.append("")
    lines.append("All Δ values are percentage-points of L3 miss-rate vs LRU "
                 "at the same (graph, app, L3) tuple (negative = better).")
    lines.append("")
    lines.append("Verdict suffix encodes the literature-claim outcome where applicable: "
                 "`✓` ok, `~` within tolerance, `✗` DISAGREE, `?` insufficient data.")
    lines.append("")
    header = ("| Graph | App | L3 | LRU miss | SRRIP Δ | GRASP Δ | POPT Δ |"
              " GRASP claim | POPT claim |")
    sep = "|---|---|---|---:|---:|---:|---:|---|---|"
    lines.append(header)
    lines.append(sep)
    for r in rows:
        def fmt_pct(p):
            return f"{r.miss.get(p, float('nan')) * 100:.2f}%" \
                if p in r.miss else "—"

        def fmt_d(p):
            if p == "LRU":
                return "0.00"
            if p not in r.delta:
                return "—"
            return f"{r.delta[p]:+.2f}"
        gv = _verdict_glyph(r.verdict.get("GRASP", ""))
        pv = _verdict_glyph(r.verdict.get("POPT", ""))
        lines.append(
            f"| {r.graph} | {r.app} | {r.l3_size} | "
            f"{fmt_pct('LRU')} | {fmt_d('SRRIP')} | {fmt_d('GRASP')} | {fmt_d('POPT')} | "
            f"{gv} | {pv} |"
        )
    lines.append("")
    return "\n".join(lines)


def _verdict_glyph(v: str) -> str:
    return {
        "ok": "✓ ok",
        "within_tol": "~ within_tol",
        "DISAGREE": "✗ DISAGREE",
        "insufficient": "? insufficient",
        "no_lru": "— no LRU",
        "": "—",
    }.get(v, v)


def format_csv(rows: list[Row]) -> str:
    import io
    buf = io.StringIO()
    fields = ["graph", "app", "l3_size",
              "lru_miss_rate", "srrip_miss_rate",
              "grasp_miss_rate", "popt_miss_rate",
              "srrip_delta_pp", "grasp_delta_pp", "popt_delta_pp",
              "grasp_verdict", "popt_verdict",
              "min_accesses"]
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({
            "graph": r.graph,
            "app": r.app,
            "l3_size": r.l3_size,
            "lru_miss_rate": f"{r.miss.get('LRU', float('nan')):.6f}" if "LRU" in r.miss else "",
            "srrip_miss_rate": f"{r.miss.get('SRRIP', float('nan')):.6f}" if "SRRIP" in r.miss else "",
            "grasp_miss_rate": f"{r.miss.get('GRASP', float('nan')):.6f}" if "GRASP" in r.miss else "",
            "popt_miss_rate": f"{r.miss.get('POPT', float('nan')):.6f}" if "POPT" in r.miss else "",
            "srrip_delta_pp": f"{r.delta.get('SRRIP', 0.0):+.3f}" if "SRRIP" in r.delta else "",
            "grasp_delta_pp": f"{r.delta.get('GRASP', 0.0):+.3f}" if "GRASP" in r.delta else "",
            "popt_delta_pp": f"{r.delta.get('POPT', 0.0):+.3f}" if "POPT" in r.delta else "",
            "grasp_verdict": r.verdict.get("GRASP", ""),
            "popt_verdict": r.verdict.get("POPT", ""),
            "min_accesses": r.accesses,
        })
    return buf.getvalue()


def format_json(rows: list[Row]) -> str:
    return json.dumps([
        {
            "graph": r.graph, "app": r.app, "l3_size": r.l3_size,
            "miss_rate": r.miss, "delta_pp_vs_lru": r.delta,
            "verdict": r.verdict, "accesses": r.accesses,
        } for r in rows
    ], indent=2, sort_keys=True)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-root", type=Path, required=True)
    p.add_argument("--sweep-subdir", default="lit")
    p.add_argument("--graphs", nargs="*", default=None)
    p.add_argument("--apps", nargs="*", default=None)
    p.add_argument("--l3-sizes", nargs="*", default=None,
                   help="Restrict to these L3 sizes, e.g. 1MB 4MB.")
    p.add_argument("--min-accesses", type=int, default=10_000)
    p.add_argument("--markdown", type=Path, default=None)
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)

    obs = _lf.load_observations(args.sweep_root, args.sweep_subdir)
    if not obs:
        print(f"[paper-table] no observations under {args.sweep_root}/*/{args.sweep_subdir}/",
              file=sys.stderr)
        return 1

    if args.graphs:
        obs = [o for o in obs if o.graph in args.graphs]
    if args.apps:
        obs = [o for o in obs if o.app in args.apps]
    if args.l3_sizes:
        obs = [o for o in obs if o.l3_size in args.l3_sizes]

    obs_idx = _lf.index(obs)
    claims_idx = _index_claims()
    rows = build_rows(obs_idx, claims_idx, args.min_accesses)
    if not rows:
        print("[paper-table] no rows after filters; nothing to emit.",
              file=sys.stderr)
        return 1

    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(format_markdown(rows).rstrip("\n") + "\n")
        print(f"[paper-table] markdown: {args.markdown}", file=sys.stderr)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        args.csv.write_text(format_csv(rows))
        print(f"[paper-table] csv: {args.csv}", file=sys.stderr)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(format_json(rows).rstrip("\n") + "\n")
        print(f"[paper-table] json: {args.json}", file=sys.stderr)

    if not (args.markdown or args.csv or args.json):
        # Default to stdout markdown if no output requested.
        sys.stdout.write(format_markdown(rows))

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
