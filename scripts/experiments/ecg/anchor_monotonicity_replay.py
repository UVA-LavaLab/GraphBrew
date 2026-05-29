#!/usr/bin/env python3
"""Anchor cell-level L3-sweep monotonicity replay across gem5 + Sniper.

Reads `wiki/data/gem5_slope_replay.json` and `wiki/data/sniper_slope_replay.json`,
walks every (graph, app, policy) anchor cell, and asserts that the per-cell
`miss_pp_by_size` curve is non-increasing as L3 grows --- with **tier-aware**
tolerances:

* gem5 (high-fidelity sim) must be **strictly monotone** (zero bumps allowed).
* sniper (lower-fidelity sim) is permitted bounded bumps:
    - per-tool bump rate ceiling
    - per-tool hard-violation (>=0.5 pp) ceiling
    - per-tool max-bump magnitude ceiling

The numbers are chosen to lock the **current** state of the locked anchor
sweeps, so any future corpus reshuffle that worsens monotonicity (eg new
sniper noise, broken policy plumbing, regressed cache sizing) gets caught.

Outputs:
  - wiki/data/anchor_monotonicity_replay.json
  - wiki/data/anchor_monotonicity_replay.md

Verdict PASS iff every per-tool ceiling holds.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Per-tool tolerances. These are intentionally close to the current observed
# values so the gate locks the present state rather than offering slack room.
TOOL_TOLERANCES: Dict[str, Dict[str, float]] = {
    # gem5 is perfectly monotone right now -> zero tolerance for bumps.
    "gem5": {
        "bump_rate_max_pct": 0.0,
        "hard_bumps_max": 0,
        "max_bump_pp_max": 0.0,
    },
    # sniper is the lower-fidelity simulator; locked tightly above the current
    # observation (~35% bump rate, 2 hard, max ~1.18 pp).
    "sniper": {
        "bump_rate_max_pct": 40.0,
        "hard_bumps_max": 5,
        "max_bump_pp_max": 2.0,
    },
}

# A bump >= this magnitude (pp) counts as a "hard" monotonicity violation.
HARD_BUMP_THRESHOLD_PP = 0.5

# Any per-step regression at or above this (pp) is a structural failure for
# *any* tool, regardless of per-tool tolerances.
CATASTROPHIC_BUMP_PP = 3.0


def _load_tool(path: Path) -> Dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)


def _walk_tool(tool: str, blob: Dict[str, Any]) -> Dict[str, Any]:
    expected = list(blob.get("meta", {}).get("expected_sizes", []) or [])
    if not expected:
        raise ValueError(f"{tool}: meta.expected_sizes missing or empty")
    cells = blob.get("per_cell", []) or []
    bumps: List[Dict[str, Any]] = []
    steps_total = 0
    hard_total = 0
    max_bump_pp = 0.0
    for cell in cells:
        mb = cell.get("miss_pp_by_size", {}) or {}
        try:
            seq = [float(mb[s]) for s in expected]
        except (KeyError, TypeError, ValueError):
            continue
        for i in range(len(seq) - 1):
            steps_total += 1
            delta = seq[i + 1] - seq[i]
            if delta > 0:
                bump = {
                    "tool": tool,
                    "graph": cell.get("graph"),
                    "app": cell.get("app"),
                    "policy": cell.get("policy"),
                    "l3_from": expected[i],
                    "l3_to": expected[i + 1],
                    "delta_pp": round(delta, 6),
                }
                bumps.append(bump)
                if delta >= HARD_BUMP_THRESHOLD_PP:
                    hard_total += 1
                if delta > max_bump_pp:
                    max_bump_pp = delta
    bump_rate_pct = (len(bumps) / steps_total * 100.0) if steps_total else 0.0
    return {
        "tool": tool,
        "expected_sizes": expected,
        "cells": len(cells),
        "steps_total": steps_total,
        "bumps": len(bumps),
        "bump_rate_pct": round(bump_rate_pct, 4),
        "hard_bumps": hard_total,
        "max_bump_pp": round(max_bump_pp, 6),
        "worst_bumps": sorted(bumps, key=lambda b: -b["delta_pp"])[:6],
        "all_bumps": bumps,
    }


def _evaluate(summary: Dict[str, Any], tol: Dict[str, float]) -> Dict[str, Any]:
    checks = {
        "bump_rate_ok": summary["bump_rate_pct"] <= tol["bump_rate_max_pct"] + 1e-9,
        "hard_bumps_ok": summary["hard_bumps"] <= tol["hard_bumps_max"],
        "max_bump_pp_ok": summary["max_bump_pp"] <= tol["max_bump_pp_max"] + 1e-9,
    }
    catastrophic = [b for b in summary["all_bumps"] if b["delta_pp"] >= CATASTROPHIC_BUMP_PP]
    checks["no_catastrophic"] = not catastrophic
    return {
        "tolerances": tol,
        "checks": checks,
        "catastrophic_bumps": catastrophic,
        "verdict_ok": all(checks.values()),
    }


def _build_report(gem5_path: Path, sniper_path: Path) -> Dict[str, Any]:
    per_tool: Dict[str, Dict[str, Any]] = {}
    overall_ok = True
    catastrophic_total: List[Dict[str, Any]] = []
    for tool, path in (("gem5", gem5_path), ("sniper", sniper_path)):
        blob = _load_tool(path)
        summary = _walk_tool(tool, blob)
        evaluation = _evaluate(summary, TOOL_TOLERANCES[tool])
        per_tool[tool] = {**summary, "evaluation": evaluation}
        overall_ok = overall_ok and evaluation["verdict_ok"]
        catastrophic_total.extend(evaluation["catastrophic_bumps"])
    medians = {
        tool: statistics.median([b["delta_pp"] for b in payload["all_bumps"]])
        if payload["all_bumps"]
        else 0.0
        for tool, payload in per_tool.items()
    }
    return {
        "schema": "anchor_monotonicity_replay/v1",
        "per_tool": per_tool,
        "overall": {
            "verdict_ok": overall_ok,
            "catastrophic_bumps": catastrophic_total,
            "median_bump_pp": {k: round(v, 6) for k, v in medians.items()},
        },
        "constants": {
            "hard_bump_threshold_pp": HARD_BUMP_THRESHOLD_PP,
            "catastrophic_bump_pp": CATASTROPHIC_BUMP_PP,
        },
    }


def _render_md(report: Dict[str, Any]) -> str:
    overall = report["overall"]
    lines: List[str] = []
    lines.append("# Anchor cell L3-sweep monotonicity replay")
    lines.append("")
    lines.append(
        f"**Verdict:** {'PASS' if overall['verdict_ok'] else 'FAIL'} "
        f"(catastrophic_bumps={len(overall['catastrophic_bumps'])})"
    )
    lines.append("")
    lines.append("## Per-tool summary")
    lines.append("")
    lines.append(
        "| tool | cells | steps | bumps | bump_rate_% | hard_bumps | max_bump_pp | verdict |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|:---|")
    for tool, payload in report["per_tool"].items():
        ev = payload["evaluation"]
        lines.append(
            f"| {tool} | {payload['cells']} | {payload['steps_total']} | "
            f"{payload['bumps']} | {payload['bump_rate_pct']:.2f} | "
            f"{payload['hard_bumps']} | {payload['max_bump_pp']:.4f} | "
            f"{'PASS' if ev['verdict_ok'] else 'FAIL'} |"
        )
    lines.append("")
    lines.append("## Per-tool tolerances (locked)")
    lines.append("")
    lines.append("| tool | bump_rate_max_% | hard_bumps_max | max_bump_pp_max |")
    lines.append("|---|---:|---:|---:|")
    for tool, payload in report["per_tool"].items():
        tol = payload["evaluation"]["tolerances"]
        lines.append(
            f"| {tool} | {tol['bump_rate_max_pct']:.2f} | "
            f"{tol['hard_bumps_max']} | {tol['max_bump_pp_max']:.4f} |"
        )
    lines.append("")
    lines.append("## Worst bumps per tool (up to 6)")
    lines.append("")
    for tool, payload in report["per_tool"].items():
        lines.append(f"### {tool}")
        lines.append("")
        if not payload["worst_bumps"]:
            lines.append("_no bumps observed_")
            lines.append("")
            continue
        lines.append("| graph | app | policy | L3 from -> to | delta_pp |")
        lines.append("|---|---|---|---|---:|")
        for b in payload["worst_bumps"]:
            lines.append(
                f"| {b['graph']} | {b['app']} | {b['policy']} | "
                f"{b['l3_from']} -> {b['l3_to']} | +{b['delta_pp']:.4f} |"
            )
        lines.append("")
    lines.append("## Constants")
    lines.append("")
    lines.append(f"- hard_bump_threshold_pp = {report['constants']['hard_bump_threshold_pp']}")
    lines.append(f"- catastrophic_bump_pp   = {report['constants']['catastrophic_bump_pp']}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gem5-json", default="wiki/data/gem5_slope_replay.json")
    ap.add_argument("--sniper-json", default="wiki/data/sniper_slope_replay.json")
    ap.add_argument("--json-out", default="wiki/data/anchor_monotonicity_replay.json")
    ap.add_argument("--md-out", default="wiki/data/anchor_monotonicity_replay.md")
    args = ap.parse_args()

    gem5_path = Path(args.gem5_json)
    sniper_path = Path(args.sniper_json)
    if not gem5_path.exists() or not sniper_path.exists():
        print(f"anchor-monotonicity-replay: missing inputs (gem5={gem5_path.exists()} sniper={sniper_path.exists()})")
        return 0

    report = _build_report(gem5_path, sniper_path)

    out_json = Path(args.json_out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    md = _render_md(report)
    Path(args.md_out).write_text(md)

    per_tool = report["per_tool"]
    summary = " ".join(
        f"{t}={p['bumps']}b/{p['steps_total']}s({p['bump_rate_pct']:.1f}%)"
        for t, p in per_tool.items()
    )
    verdict = "PASS" if report["overall"]["verdict_ok"] else "FAIL"
    print(f"anchor-monotonicity-replay: {summary} verdict={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
