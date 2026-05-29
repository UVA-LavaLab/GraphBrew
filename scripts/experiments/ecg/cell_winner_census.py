#!/usr/bin/env python3
"""Cell-winner census: how decisive is the corpus?

For each (graph, app, l3_size) cell, classify by winner status:

  * unique_winner  : exactly one policy has is_winner=1
  * tied_winners   : two or more policies tied at top
  * no_winner      : no policy flagged (degenerate cell)

Reports per-app breakdown and lists the tied cells explicitly
(they are the corpus's 'unwinnable' cases and any paper claim
that includes them must be qualified).

This gate pins the corpus decisiveness — a number that MUST appear
in the paper text (something like '97% of cells have a unique winner;
the remaining 3% are tied between policies and are reported
separately'). If we ever silently include tied cells in win-rate
counts, that's a credibility-killing methodology bug.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WIKI_DATA = REPO_ROOT / "wiki" / "data"


def _resolve_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def classify_cells(rows: list[dict]) -> dict:
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["graph"], r["app"], r["l3_size"])
        cells[key].append(r)

    per_app: dict[str, dict] = defaultdict(lambda: {
        "n_cells": 0, "unique_winner": 0, "tied_winners": 0,
        "no_winner": 0, "tied_cells": [], "no_winner_cells": [],
    })
    tied_cells_all: list[dict] = []
    no_winner_cells_all: list[dict] = []
    tied_breakdown: Counter = Counter()

    for (graph, app, l3), rs in cells.items():
        winners = [r for r in rs if r.get("is_winner") == "1"]
        winner_policies = sorted({r["policy"] for r in winners})
        ap = per_app[app]
        ap["n_cells"] += 1
        if len(winners) == 0:
            ap["no_winner"] += 1
            cell_payload = {"graph": graph, "app": app, "l3": l3,
                            "policies_present": sorted({r["policy"] for r in rs})}
            ap["no_winner_cells"].append(cell_payload)
            no_winner_cells_all.append(cell_payload)
        elif len(winner_policies) == 1:
            ap["unique_winner"] += 1
        else:
            ap["tied_winners"] += 1
            tied_breakdown[len(winner_policies)] += 1
            cell_payload = {"graph": graph, "app": app, "l3": l3,
                            "tied_policies": winner_policies,
                            "tied_count": len(winner_policies)}
            ap["tied_cells"].append(cell_payload)
            tied_cells_all.append(cell_payload)

    n_total = sum(p["n_cells"] for p in per_app.values())
    n_unique = sum(p["unique_winner"] for p in per_app.values())
    n_tied = sum(p["tied_winners"] for p in per_app.values())
    n_none = sum(p["no_winner"] for p in per_app.values())

    return {
        "meta": {
            "n_cells_total": n_total,
            "n_unique_winner": n_unique,
            "n_tied_winners": n_tied,
            "n_no_winner": n_none,
            "pct_unique_winner": round(100.0 * n_unique / n_total, 2) if n_total else 0,
            "pct_tied_winners": round(100.0 * n_tied / n_total, 2) if n_total else 0,
            "pct_no_winner": round(100.0 * n_none / n_total, 2) if n_total else 0,
            "tied_breakdown_by_count": dict(tied_breakdown),
        },
        "per_app": {app: dict(p) for app, p in per_app.items()},
        "all_tied_cells": tied_cells_all,
        "all_no_winner_cells": no_winner_cells_all,
    }


def write_md(payload: dict, md_path: Path) -> None:
    m = payload["meta"]
    lines = [
        "# Cell-winner census: corpus decisiveness",
        "",
        "Classification of every `(graph, app, l3_size)` cell by winner status.",
        "Tied/no-winner cells must be excluded or qualified in any per-cell",
        "win-rate claim — they are the corpus's 'unwinnable' cases.",
        "",
        f"- Total cells: **{m['n_cells_total']}**",
        f"- Cells with **unique winner**: **{m['n_unique_winner']}** "
        f"({m['pct_unique_winner']}%)",
        f"- Cells with **tied winners**: **{m['n_tied_winners']}** "
        f"({m['pct_tied_winners']}%)",
        f"- Cells with **no winner**: **{m['n_no_winner']}** "
        f"({m['pct_no_winner']}%)",
        "",
    ]
    if m["tied_breakdown_by_count"]:
        lines += [
            "Tied cells by tie-count:",
            "",
        ]
        for k, v in sorted(m["tied_breakdown_by_count"].items()):
            lines.append(f"- {k}-way tie: **{v}** cells")
        lines.append("")

    lines += [
        "## Per-app census",
        "",
        "| App | n cells | Unique | Tied | None | % decisive |",
        "| :-- | ------: | -----: | ---: | ---: | ---------: |",
    ]
    for app, p in sorted(payload["per_app"].items()):
        pct = round(100.0 * p["unique_winner"] / p["n_cells"], 1) \
            if p["n_cells"] else 0
        lines.append(
            f"| `{app}` | {p['n_cells']} | {p['unique_winner']} "
            f"| {p['tied_winners']} | {p['no_winner']} | {pct}% |"
        )

    if payload["all_tied_cells"]:
        lines += [
            "",
            f"## Tied cells (n={len(payload['all_tied_cells'])})",
            "",
            "These cells have ≥2 policies tied at top of the cell.",
            "",
            "| Graph | App | L3 | Tied policies |",
            "| :---- | :-- | :- | :------------ |",
        ]
        for c in payload["all_tied_cells"]:
            tied = ", ".join(f"`{p}`" for p in c["tied_policies"])
            lines.append(
                f"| `{c['graph']}` | `{c['app']}` | `{c['l3']}` | {tied} |"
            )

    if payload["all_no_winner_cells"]:
        lines += [
            "",
            f"## No-winner cells (n={len(payload['all_no_winner_cells'])})",
            "",
            "Degenerate cells with no policy flagged as winner — must be",
            "investigated, never silently dropped.",
            "",
        ]
        for c in payload["all_no_winner_cells"]:
            lines.append(
                f"- `{c['graph']}` / `{c['app']}` / `{c['l3']}`: "
                f"policies present = {c['policies_present']}"
            )

    md_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oracle-json", default=str(WIKI_DATA / "oracle_gap.json"))
    ap.add_argument("--json-out", default=str(WIKI_DATA / "cell_winner_census.json"))
    ap.add_argument("--md-out", default=str(WIKI_DATA / "cell_winner_census.md"))
    args = ap.parse_args()

    oracle = json.loads(Path(args.oracle_json).read_text())
    payload = classify_cells(oracle["rows"])

    json_path = Path(args.json_out)
    md_path = Path(args.md_out)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_md(payload, md_path)

    m = payload["meta"]
    print(
        f"[cell-census] n_cells={m['n_cells_total']} "
        f"unique={m['n_unique_winner']} ({m['pct_unique_winner']}%) "
        f"tied={m['n_tied_winners']} no_winner={m['n_no_winner']} "
        f"→ {_resolve_label(md_path)}"
    )


if __name__ == "__main__":
    main()
