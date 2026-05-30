#!/usr/bin/env python3
"""Per-L3-size policy stability: does the winner persist across cache sizes?

Why this exists
---------------
The most common paper claim shape is "policy X is the best across
cache sizes". This is precisely the property that fails first when
methodology drifts (e.g. an evaluation that only used one L3 size,
or one that averaged across sizes and hid a regime change).

This gate decomposes every (app, l3_size) cell, counts which policy
wins how many cells at that size, and pins the *stability* pattern:

* cc/GRASP must win at every "paper L3 size" (1MB, 4MB, 8MB).
* pr/POPT must win at every "paper L3 size".
* bfs must exhibit a GRASP-at-1MB → POPT-at-≥4MB regime change.
* sssp must NOT have a stable cross-L3 winner (honest negative).

Definitions
-----------
For each (app, l3_size):

* For each policy, count cells (= graphs) at that size where the
  policy is a winner. With multiple-winner tie cells this can sum
  to more than the number of graphs.
* ``top_policy`` is the argmax. ``top_share`` = top_wins / n_cells.
* ``runner_up`` is the second-place policy. ``margin`` = top_wins −
  runner_up_wins.
* ``unique_winner`` is True iff exactly one policy is at top_wins.

Paper-grade L3 sizes are 1MB, 4MB, 8MB — every other size in the
corpus has only 1–2 cells and is too sparse for stability claims.

Output
------
* ``wiki/data/l3_policy_stability.json``
* ``wiki/data/l3_policy_stability.md``
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "wiki" / "data"

POLICIES = ("LRU", "SRRIP", "GRASP", "POPT")
APPS = ("pr", "bc", "cc", "bfs", "sssp")
PAPER_L3 = ("1MB", "4MB", "8MB")
# L3 sizes ordered small→large for stable rendering
L3_ORDER = ("4kB", "16kB", "64kB", "256kB", "1MB", "4MB", "8MB")


def _is_winner(row: dict) -> bool:
    val = row.get("is_winner")
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return int(val) == 1
    if isinstance(val, str):
        return val.strip() in {"1", "true", "True"}
    return False


def load_oracle_rows(path: Path) -> list[dict]:
    blob = json.loads(path.read_text())
    if isinstance(blob, dict) and "rows" in blob:
        return blob["rows"]
    return blob


def aggregate(rows: list[dict]) -> dict:
    by_app_l3_pol = defaultdict(lambda: defaultdict(int))  # (app,l3) → pol → wins
    cells_by_app_l3 = defaultdict(set)  # (app,l3) → set of graphs
    for r in rows:
        key = (r["app"], r["l3_size"])
        cells_by_app_l3[key].add(r["graph"])
        if _is_winner(r):
            by_app_l3_pol[key][r["policy"]] += 1

    per_app = {}
    for app in sorted({r["app"] for r in rows}):
        l3_payload = {}
        l3_sizes_present = sorted(
            {l3 for (a, l3) in cells_by_app_l3 if a == app},
            key=lambda s: L3_ORDER.index(s) if s in L3_ORDER else 99,
        )
        for l3 in l3_sizes_present:
            n_cells = len(cells_by_app_l3[(app, l3)])
            wins = dict(by_app_l3_pol[(app, l3)])
            # Sort policy entries by wins desc, then by POLICIES order
            ranked = sorted(
                ((p, wins.get(p, 0)) for p in POLICIES),
                key=lambda x: (-x[1], POLICIES.index(x[0])),
            )
            top_policy, top_wins = ranked[0]
            runner_up, runner_wins = ranked[1] if len(ranked) > 1 else (None, 0)
            unique_winner = top_wins > runner_wins
            l3_payload[l3] = {
                "n_cells": n_cells,
                "wins": wins,
                "top_policy": top_policy,
                "top_wins": top_wins,
                "top_share": round(top_wins / n_cells, 4) if n_cells else 0.0,
                "runner_up": runner_up,
                "runner_up_wins": runner_wins,
                "margin": top_wins - runner_wins,
                "unique_winner": unique_winner,
            }

        # Cross-L3 stability summary across PAPER_L3 sizes
        paper_tops = [
            l3_payload[l3]["top_policy"]
            for l3 in PAPER_L3
            if l3 in l3_payload and l3_payload[l3]["unique_winner"]
        ]
        unique_tops = sorted(set(paper_tops))
        stability = {
            "paper_l3_tops": paper_tops,
            "unique_top_policies_at_paper_l3": unique_tops,
            "n_unique_top_policies": len(unique_tops),
            "is_stable_single_winner": len(unique_tops) == 1
            and len(paper_tops) == len(PAPER_L3),
            "has_regime_change": len(unique_tops) >= 2,
        }

        per_app[app] = {"l3": l3_payload, "stability": stability}

    return {"per_app": per_app}


def render_md(payload: dict) -> str:
    meta = payload["meta"]
    lines = [
        "# Per-L3-size policy stability",
        "",
        f"Source: `{meta['source']}` ({meta['n_rows']} rows).",
        "",
        "Paper-grade L3 sizes: " + ", ".join(meta["paper_l3"]) + ".",
        "",
        "## Stability summary across paper L3 sizes (1MB, 4MB, 8MB)",
        "",
        "| App | Tops (1MB / 4MB / 8MB) | Unique tops | Stable? | Regime change? |",
        "| --- | --- | --- | --- | --- |",
    ]
    for app in APPS:
        if app not in payload["per_app"]:
            continue
        s = payload["per_app"][app]["stability"]
        tops_str = " / ".join(s["paper_l3_tops"]) if s["paper_l3_tops"] else "-"
        lines.append(
            f"| {app} | {tops_str} | {s['n_unique_top_policies']} "
            f"| {'YES' if s['is_stable_single_winner'] else 'no'} "
            f"| {'YES' if s['has_regime_change'] else 'no'} |"
        )

    lines += ["", "## Per-app per-L3 winner tables", ""]
    for app in APPS:
        if app not in payload["per_app"]:
            continue
        lines += [
            f"### {app}",
            "",
            "| L3 | n_cells | Top | wins | share | Runner-up | margin | Unique? |",
            "| --- | ---: | --- | ---: | ---: | --- | ---: | --- |",
        ]
        for l3 in L3_ORDER:
            row = payload["per_app"][app]["l3"].get(l3)
            if not row:
                continue
            ru = row["runner_up"] or "-"
            lines.append(
                f"| {l3} | {row['n_cells']} | **{row['top_policy']}** "
                f"| {row['top_wins']} | {row['top_share']:.2f} | {ru} "
                f"| {row['margin']} | {'yes' if row['unique_winner'] else 'tie'} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle-json", default=str(DATA_DIR / "oracle_gap.json")
    )
    parser.add_argument(
        "--json-out", default=str(DATA_DIR / "l3_policy_stability.json")
    )
    parser.add_argument(
        "--md-out", default=str(DATA_DIR / "l3_policy_stability.md")
    )
    args = parser.parse_args()

    rows = load_oracle_rows(Path(args.oracle_json))
    agg = aggregate(rows)

    src_path = Path(args.oracle_json)
    try:
        src_label = str(src_path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        src_label = str(src_path)

    payload = {
        "meta": {
            "source": src_label,
            "n_rows": len(rows),
            "policies": list(POLICIES),
            "apps": list(APPS),
            "paper_l3": list(PAPER_L3),
        },
        **agg,
    }

    Path(args.json_out).write_text(json.dumps(payload, indent=2, sort_keys=True))
    Path(args.md_out).write_text(render_md(payload).rstrip("\n") + "\n")

    stable = [
        a for a, p in payload["per_app"].items()
        if p["stability"]["is_stable_single_winner"]
    ]
    regime = [
        a for a, p in payload["per_app"].items()
        if p["stability"]["has_regime_change"]
    ]
    print(
        f"[l3-stability] n_rows={len(rows)} | "
        f"stable single-winner kernels: {sorted(stable)} | "
        f"regime-change kernels: {sorted(regime)} → {args.md_out}"
    )


if __name__ == "__main__":
    main()
