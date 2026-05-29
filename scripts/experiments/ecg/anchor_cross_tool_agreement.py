#!/usr/bin/env python3
"""Cross-tool slope-sign agreement on shared anchor cells (gate 82).

Reads `wiki/data/gem5_slope_replay.json` and `wiki/data/sniper_slope_replay.json`,
finds (graph, app, policy) cells present in BOTH tools, and asserts:

  1. Every shared cell has matching slope sign across the two simulators
     (physical replication).
  2. Every shared cell has *negative* slope (miss% decreases as L3 grows,
     the basic "bigger cache helps" invariant).
  3. Sniper slopes are uniformly steeper in magnitude than gem5 slopes
     (sniper's wider L3 sweep + lower fidelity = larger |slope|).
  4. The per-cell |slope| difference is bounded.

Currently shared cells: (email-Eu-core, pr) x {GRASP, LRU, SRRIP} = 3 cells,
all negative on both tools, sniper steeper everywhere.

Outputs:
  - wiki/data/anchor_cross_tool_agreement.json
  - wiki/data/anchor_cross_tool_agreement.md

Verdict PASS iff every invariant holds.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

# Locked thresholds.
MAX_ABS_SLOPE_DIFF_PP = 8.0          # |sniper_slope - gem5_slope| ceiling per cell
SHARED_CELLS_FLOOR = 3               # need at least this many shared cells
SIGN_AGREEMENT_FLOOR = 1.0           # require 100% sign agreement
SNIPER_STEEPER_FLOOR = 1.0           # require 100% sniper-steeper cells


def _index(blob: Dict[str, Any]) -> Dict[Any, Dict[str, Any]]:
    return {
        (c["graph"], c["app"], c["policy"]): c
        for c in blob.get("per_cell", [])
    }


def _build_report(gem5_path: Path, sniper_path: Path) -> Dict[str, Any]:
    g5_blob = json.loads(gem5_path.read_text())
    sn_blob = json.loads(sniper_path.read_text())
    g5 = _index(g5_blob)
    sn = _index(sn_blob)

    shared_keys = sorted(set(g5) & set(sn))
    shared_cells: List[Dict[str, Any]] = []
    sign_agree_count = 0
    both_negative_count = 0
    sniper_steeper_count = 0
    abs_diffs: List[float] = []

    for key in shared_keys:
        g_slope = float(g5[key]["slope_pp_per_octave"])
        s_slope = float(sn[key]["slope_pp_per_octave"])
        sign_match = (g_slope > 0) == (s_slope > 0) or (g_slope == 0 and s_slope == 0)
        both_neg = g_slope < 0 and s_slope < 0
        sniper_steeper = abs(s_slope) >= abs(g_slope)
        abs_diff = abs(abs(s_slope) - abs(g_slope))
        if sign_match:
            sign_agree_count += 1
        if both_neg:
            both_negative_count += 1
        if sniper_steeper:
            sniper_steeper_count += 1
        abs_diffs.append(abs_diff)
        shared_cells.append(
            {
                "graph": key[0],
                "app": key[1],
                "policy": key[2],
                "gem5_slope_pp": round(g_slope, 6),
                "sniper_slope_pp": round(s_slope, 6),
                "sign_match": sign_match,
                "both_negative": both_neg,
                "sniper_steeper": sniper_steeper,
                "abs_diff_pp": round(abs_diff, 6),
            }
        )

    n_shared = len(shared_keys)
    sign_rate = (sign_agree_count / n_shared) if n_shared else 0.0
    both_neg_rate = (both_negative_count / n_shared) if n_shared else 0.0
    sniper_steeper_rate = (sniper_steeper_count / n_shared) if n_shared else 0.0
    max_abs_diff = max(abs_diffs) if abs_diffs else 0.0
    median_abs_diff = statistics.median(abs_diffs) if abs_diffs else 0.0

    checks = {
        "shared_floor": {
            "n_shared": n_shared,
            "floor": SHARED_CELLS_FLOOR,
            "ok": n_shared >= SHARED_CELLS_FLOOR,
        },
        "sign_agreement": {
            "rate": round(sign_rate, 6),
            "floor": SIGN_AGREEMENT_FLOOR,
            "ok": sign_rate >= SIGN_AGREEMENT_FLOOR - 1e-9,
        },
        "both_negative": {
            "rate": round(both_neg_rate, 6),
            "floor": 1.0,
            "ok": both_neg_rate >= 1.0 - 1e-9,
        },
        "sniper_steeper": {
            "rate": round(sniper_steeper_rate, 6),
            "floor": SNIPER_STEEPER_FLOOR,
            "ok": sniper_steeper_rate >= SNIPER_STEEPER_FLOOR - 1e-9,
        },
        "abs_diff_ceiling": {
            "max_abs_diff_pp": round(max_abs_diff, 6),
            "ceiling": MAX_ABS_SLOPE_DIFF_PP,
            "ok": max_abs_diff <= MAX_ABS_SLOPE_DIFF_PP + 1e-9,
        },
    }
    verdict_ok = all(c["ok"] for c in checks.values())

    return {
        "schema": "anchor_cross_tool_agreement/v1",
        "meta": {
            "gem5_cells": len(g5),
            "sniper_cells": len(sn),
            "shared_cells": n_shared,
            "thresholds": {
                "shared_cells_floor": SHARED_CELLS_FLOOR,
                "sign_agreement_floor": SIGN_AGREEMENT_FLOOR,
                "sniper_steeper_floor": SNIPER_STEEPER_FLOOR,
                "max_abs_slope_diff_pp": MAX_ABS_SLOPE_DIFF_PP,
            },
        },
        "shared_cells": shared_cells,
        "summary": {
            "sign_agree_count": sign_agree_count,
            "both_negative_count": both_negative_count,
            "sniper_steeper_count": sniper_steeper_count,
            "max_abs_diff_pp": round(max_abs_diff, 6),
            "median_abs_diff_pp": round(median_abs_diff, 6),
        },
        "checks": checks,
        "verdict_ok": verdict_ok,
    }


def _render_md(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Cross-tool shared-anchor slope-sign agreement")
    lines.append("")
    lines.append(
        f"**Verdict:** {'PASS' if report['verdict_ok'] else 'FAIL'} "
        f"(shared={report['meta']['shared_cells']}, "
        f"sign_agree={report['summary']['sign_agree_count']}/{report['meta']['shared_cells']})"
    )
    lines.append("")
    lines.append("## Shared cells (gem5 ∩ sniper)")
    lines.append("")
    lines.append(
        "| graph | app | policy | gem5_slope | sniper_slope | "
        "sign_match | both_neg | sniper_steeper | |Δ| |"
    )
    lines.append("|---|---|---|---:|---:|:---:|:---:|:---:|---:|")
    for c in report["shared_cells"]:
        lines.append(
            f"| {c['graph']} | {c['app']} | {c['policy']} | "
            f"{c['gem5_slope_pp']:+.4f} | {c['sniper_slope_pp']:+.4f} | "
            f"{'OK' if c['sign_match'] else 'FAIL'} | "
            f"{'OK' if c['both_negative'] else 'FAIL'} | "
            f"{'OK' if c['sniper_steeper'] else 'FAIL'} | "
            f"{c['abs_diff_pp']:.4f} |"
        )
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| check | ok |")
    lines.append("|---|:---:|")
    for name, payload in report["checks"].items():
        lines.append(f"| {name} | {'OK' if payload['ok'] else 'FAIL'} |")
    lines.append("")
    lines.append("## Thresholds (locked)")
    lines.append("")
    for k, v in report["meta"]["thresholds"].items():
        lines.append(f"- {k} = {v}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gem5-json", default="wiki/data/gem5_slope_replay.json")
    ap.add_argument("--sniper-json", default="wiki/data/sniper_slope_replay.json")
    ap.add_argument("--json-out", default="wiki/data/anchor_cross_tool_agreement.json")
    ap.add_argument("--md-out", default="wiki/data/anchor_cross_tool_agreement.md")
    args = ap.parse_args()

    g5 = Path(args.gem5_json)
    sn = Path(args.sniper_json)
    if not g5.exists() or not sn.exists():
        print(
            f"anchor-cross-tool-agreement: missing inputs "
            f"(gem5={g5.exists()} sniper={sn.exists()})"
        )
        return 0

    report = _build_report(g5, sn)
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    Path(args.md_out).write_text(_render_md(report))

    s = report["summary"]
    n = report["meta"]["shared_cells"]
    verdict = "PASS" if report["verdict_ok"] else "FAIL"
    print(
        f"anchor-cross-tool-agreement: shared={n} sign_agree={s['sign_agree_count']}/{n} "
        f"both_neg={s['both_negative_count']}/{n} sniper_steeper={s['sniper_steeper_count']}/{n} "
        f"max|diff|={s['max_abs_diff_pp']:.3f}pp verdict={verdict}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
