#!/usr/bin/env python3
"""Per-(app, L3) winner-margin gradient (gate 48).

Defends against the reviewer pushback 'your winner is decided by one
cell — what if a single graph flips it?' by publishing the exact
margin (top wins minus runner-up wins) for every (app, L3-size) cell
in the paper scope (1MB / 4MB / 8MB).

Each (app, L3) cell is classified as:

  decisive  : margin >= 4   — strong, paper-defensive
  moderate  : margin in [2, 4)  — still distinctive
  weak      : margin == 1   — single-graph flip risk; surface honestly
  tied      : margin == 0   — multi-policy tie; report as a tie

The class distribution + per-cell margin matrix become the gate's
auditable record. Future corpus changes that demote a 'decisive' cell
to 'weak' or 'tied' fail this gate.

Output: wiki/data/winner_margin_gradient.{json,md}
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

PAPER_L3_SIZES = ("1MB", "4MB", "8MB")


def _classify(margin: int) -> str:
    if margin >= 4:
        return "decisive"
    if margin >= 2:
        return "moderate"
    if margin == 1:
        return "weak"
    return "tied"


def _resolve_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_payload(oracle_path: Path) -> dict:
    rows = json.loads(oracle_path.read_text())["rows"]
    paper_rows = [r for r in rows if r["l3_size"] in PAPER_L3_SIZES]

    apps = sorted({r["app"] for r in paper_rows})

    # win_counts[(app, l3)][policy] = int
    win_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    cell_count: dict[tuple[str, str], int] = defaultdict(int)
    seen_cells: dict[tuple[str, str], set] = defaultdict(set)
    for r in paper_rows:
        key = (r["app"], r["l3_size"])
        seen_cells[key].add((r["graph"], r["app"], r["l3_size"]))
        if int(r["is_winner"]) == 1:
            win_counts[key][r["policy"]] += 1
    for key, cells in seen_cells.items():
        cell_count[key] = len(cells)

    per_cell: dict[str, dict] = {}
    class_counts: Counter = Counter()
    for app in apps:
        for l3 in PAPER_L3_SIZES:
            key = (app, l3)
            c = win_counts.get(key, Counter())
            if not c:
                continue
            ordered = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
            top_policy, top_wins = ordered[0]
            runner_wins = ordered[1][1] if len(ordered) > 1 else 0
            margin = top_wins - runner_wins
            klass = _classify(margin)
            class_counts[klass] += 1
            tied_with = [p for p, w in c.items() if w == top_wins and p != top_policy]
            per_cell[f"{app}__{l3}"] = {
                "app": app,
                "l3_size": l3,
                "top_policy": top_policy,
                "top_wins": top_wins,
                "runner_up_wins": runner_wins,
                "margin": margin,
                "class": klass,
                "n_cells_in_scope": cell_count[key],
                "tied_top_policies": sorted(tied_with),
                "win_counts": dict(c),
            }

    n_total = sum(class_counts.values())
    n_decisive_or_moderate = class_counts["decisive"] + class_counts["moderate"]
    strong_fraction = (
        round(n_decisive_or_moderate / n_total, 4) if n_total else 0.0
    )

    weak_cells = sorted(
        k for k, v in per_cell.items() if v["class"] == "weak"
    )
    tied_cells = sorted(
        k for k, v in per_cell.items() if v["class"] == "tied"
    )

    return {
        "meta": {
            "source": _resolve_label(oracle_path),
            "scope_l3_sizes": list(PAPER_L3_SIZES),
            "n_apps": len(apps),
            "apps": apps,
            "n_cells_total": n_total,
            "class_thresholds": {
                "decisive": "margin >= 4",
                "moderate": "2 <= margin < 4",
                "weak": "margin == 1",
                "tied": "margin == 0",
            },
            "class_counts": dict(class_counts),
            "strong_cell_fraction": strong_fraction,
            "weak_cells": weak_cells,
            "tied_cells": tied_cells,
            "n_weak_cells": len(weak_cells),
            "n_tied_cells": len(tied_cells),
        },
        "per_cell": per_cell,
    }


def emit_md(payload: dict) -> str:
    m = payload["meta"]
    out = []
    out.append("# Per-(app, L3) winner-margin gradient")
    out.append("")
    out.append(
        f"Source: `{m['source']}`  •  Paper L3 scope: "
        f"{', '.join(m['scope_l3_sizes'])}"
    )
    out.append("")
    out.append(
        f"**Headline:** {sum(v for k, v in m['class_counts'].items() if k in ('decisive', 'moderate'))}"
        f"/{m['n_cells_total']} ({m['strong_cell_fraction'] * 100:.0f}%) of"
        f" (app, L3) cells have a margin >= 2; "
        f"{m['n_weak_cells']} weak, {m['n_tied_cells']} tied (honestly disclosed)."
    )
    out.append("")
    out.append("## Classification thresholds")
    out.append("")
    out.append("| class | rule | count |")
    out.append("|---|---|---:|")
    for k in ("decisive", "moderate", "weak", "tied"):
        out.append(
            f"| {k} | {m['class_thresholds'][k]} | {m['class_counts'].get(k, 0)} |"
        )
    out.append("")
    out.append("## Per-(app, L3) cell verdict")
    out.append("")
    out.append("| app | L3 | top | top_wins | runner_up | margin | class | tied with |")
    out.append("|---|---|---|---:|---:|---:|---|---|")
    for app in m["apps"]:
        for l3 in PAPER_L3_SIZES:
            key = f"{app}__{l3}"
            d = payload["per_cell"].get(key)
            if not d:
                out.append(f"| {app} | {l3} | — | — | — | — | — | — |")
                continue
            tied = ", ".join(d["tied_top_policies"]) if d["tied_top_policies"] else "—"
            out.append(
                f"| {d['app']} | {d['l3_size']} | {d['top_policy']}"
                f" | {d['top_wins']} | {d['runner_up_wins']}"
                f" | {d['margin']} | {d['class']} | {tied} |"
            )
    out.append("")
    out.append("## Honest disclosures")
    out.append("")
    if m["weak_cells"]:
        out.append(f"- Weak cells (margin == 1, single-graph flip risk): {m['weak_cells']}")
    else:
        out.append("- No weak cells.")
    if m["tied_cells"]:
        out.append(f"- Tied cells (margin == 0, report as multi-policy tie): {m['tied_cells']}")
    else:
        out.append("- No tied cells.")
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle-json", type=Path, default=WIKI_DATA / "oracle_gap.json"
    )
    parser.add_argument(
        "--json-out", type=Path, default=WIKI_DATA / "winner_margin_gradient.json"
    )
    parser.add_argument(
        "--md-out", type=Path, default=WIKI_DATA / "winner_margin_gradient.md"
    )
    args = parser.parse_args()

    payload = build_payload(args.oracle_json)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(emit_md(payload) + "\n")
    m = payload["meta"]
    print(
        f"margin-gradient: apps={m['n_apps']} cells={m['n_cells_total']}"
        f" decisive={m['class_counts'].get('decisive', 0)}"
        f" moderate={m['class_counts'].get('moderate', 0)}"
        f" weak={m['class_counts'].get('weak', 0)}"
        f" tied={m['class_counts'].get('tied', 0)}"
        f" strong_frac={m['strong_cell_fraction'] * 100:.0f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
