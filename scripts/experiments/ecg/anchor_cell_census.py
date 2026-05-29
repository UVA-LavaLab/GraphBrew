"""Gate 78 — anchor cell-pair census.

Pins the gem5 and Sniper cell coverage against silent shrinkage that
would break every downstream cross-tool gate (70, 71, 72, 74, 76)
without triggering an obvious failure.

Documented baseline coverage as of the 8-graph corpus:

  gem5 anchor:
    cells: (email-Eu-core, bc), (email-Eu-core, pr)
    policies: GRASP, LRU, SRRIP
    L3 axis: 4kB, 32kB, 256kB, 2MB
    cell-policy records: 6

  Sniper anchor:
    cells: (cit-Patents, bfs/pr/sssp), (email-Eu-core, bfs/pr/sssp)
    policies: GRASP, LRU, SRRIP
    L3 axis: 4kB, 32kB, 256kB, 2MB
    cell-policy records: 18

Cross-tool invariants:
  - Both anchors must share the same L3 axis (so cross-tool slope
    comparisons in gates 70/71/74/76 are apples-to-apples).
  - Both anchors must share the same policy set.
  - At least one shared (graph, app) cell must be present in both
    anchors (currently (email-Eu-core, pr)) so the per-cell parity
    spot-check used by gate 74 has a foothold.

Output schema:
  meta.gem5.cells, meta.gem5.policies, meta.gem5.l3_axis,
       meta.gem5.cell_policy_records, meta.gem5.n_cells_minimum
  meta.sniper.cells, meta.sniper.policies, meta.sniper.l3_axis,
       meta.sniper.cell_policy_records, meta.sniper.n_cells_minimum
  meta.shared_l3_axis : bool
  meta.shared_policies : bool
  meta.shared_cells : list[(graph, app)]
  meta.shared_cell_count : int
  meta.verdict_checks
  meta.verdict
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GEM5_JSON   = REPO_ROOT / "wiki" / "data" / "gem5_slope_replay.json"
DEFAULT_SNIPER_JSON = REPO_ROOT / "wiki" / "data" / "sniper_slope_replay.json"
DEFAULT_JSON_OUT    = REPO_ROOT / "wiki" / "data" / "anchor_cell_census.json"
DEFAULT_MD_OUT      = REPO_ROOT / "wiki" / "data" / "anchor_cell_census.md"

EXPECTED_L3_AXIS = ["4kB", "32kB", "256kB", "2MB"]
EXPECTED_POLICIES = ["GRASP", "LRU", "SRRIP"]
EXPECTED_GEM5_CELLS = [("email-Eu-core", "bc"), ("email-Eu-core", "pr")]
EXPECTED_SNIPER_CELLS = [
    ("cit-Patents",   "bfs"),
    ("cit-Patents",   "pr"),
    ("cit-Patents",   "sssp"),
    ("email-Eu-core", "bfs"),
    ("email-Eu-core", "pr"),
    ("email-Eu-core", "sssp"),
]
GEM5_MIN_CELLS = len(EXPECTED_GEM5_CELLS)
SNIPER_MIN_CELLS = len(EXPECTED_SNIPER_CELLS)


def _summarize(path: Path) -> dict:
    doc = json.loads(path.read_text())
    meta = doc["meta"]
    per_cell = doc["per_cell"]
    cells = sorted({(c["graph"], c["app"]) for c in per_cell})
    policies = sorted({c["policy"] for c in per_cell})
    l3_axis = list(meta.get("expected_sizes", []))
    return {
        "anchor_source":        meta.get("anchor_source"),
        "cells":                cells,
        "n_cells":              len(cells),
        "policies":             policies,
        "l3_axis":              l3_axis,
        "cell_policy_records":  len(per_cell),
    }


def build(gem5_path: Path, sniper_path: Path) -> dict:
    gem5   = _summarize(gem5_path)
    sniper = _summarize(sniper_path)

    gem5_cells_set = {tuple(c) for c in gem5["cells"]}
    sniper_cells_set = {tuple(c) for c in sniper["cells"]}
    expected_gem5_set = set(EXPECTED_GEM5_CELLS)
    expected_sniper_set = set(EXPECTED_SNIPER_CELLS)

    shared_cells = sorted(gem5_cells_set & sniper_cells_set)

    inv_gem5_min_count    = gem5["n_cells"] >= GEM5_MIN_CELLS
    inv_sniper_min_count  = sniper["n_cells"] >= SNIPER_MIN_CELLS
    inv_gem5_has_expected = expected_gem5_set.issubset(gem5_cells_set)
    inv_sniper_has_expected = expected_sniper_set.issubset(sniper_cells_set)
    inv_gem5_l3_matches   = gem5["l3_axis"]   == EXPECTED_L3_AXIS
    inv_sniper_l3_matches = sniper["l3_axis"] == EXPECTED_L3_AXIS
    inv_gem5_policies     = gem5["policies"]   == EXPECTED_POLICIES
    inv_sniper_policies   = sniper["policies"] == EXPECTED_POLICIES
    inv_shared_l3         = gem5["l3_axis"]   == sniper["l3_axis"]
    inv_shared_policies   = gem5["policies"]   == sniper["policies"]
    inv_at_least_one_shared = len(shared_cells) >= 1
    inv_gem5_record_count = gem5["cell_policy_records"]   == len(EXPECTED_GEM5_CELLS)   * len(EXPECTED_POLICIES)
    inv_sniper_record_count = sniper["cell_policy_records"] == len(EXPECTED_SNIPER_CELLS) * len(EXPECTED_POLICIES)

    verdict_checks = {
        "gem5_cell_count_at_or_above_baseline":   inv_gem5_min_count,
        "sniper_cell_count_at_or_above_baseline": inv_sniper_min_count,
        "gem5_has_expected_cells":                inv_gem5_has_expected,
        "sniper_has_expected_cells":              inv_sniper_has_expected,
        "gem5_l3_axis_matches":                   inv_gem5_l3_matches,
        "sniper_l3_axis_matches":                 inv_sniper_l3_matches,
        "gem5_policy_set_matches":                inv_gem5_policies,
        "sniper_policy_set_matches":              inv_sniper_policies,
        "anchors_share_l3_axis":                  inv_shared_l3,
        "anchors_share_policy_set":               inv_shared_policies,
        "anchors_share_at_least_one_cell":        inv_at_least_one_shared,
        "gem5_cell_policy_records_match":         inv_gem5_record_count,
        "sniper_cell_policy_records_match":       inv_sniper_record_count,
    }
    verdict = "PASS" if all(verdict_checks.values()) else "FAIL"

    return {
        "meta": {
            "gem5":                {**gem5, "n_cells_minimum": GEM5_MIN_CELLS},
            "sniper":              {**sniper, "n_cells_minimum": SNIPER_MIN_CELLS},
            "expected_l3_axis":    EXPECTED_L3_AXIS,
            "expected_policies":   EXPECTED_POLICIES,
            "expected_gem5_cells": [list(c) for c in EXPECTED_GEM5_CELLS],
            "expected_sniper_cells": [list(c) for c in EXPECTED_SNIPER_CELLS],
            "shared_l3_axis":      inv_shared_l3,
            "shared_policies":     inv_shared_policies,
            "shared_cells":        [list(c) for c in shared_cells],
            "shared_cell_count":   len(shared_cells),
            "verdict_checks":      verdict_checks,
            "verdict":             verdict,
        },
    }


def render_md(payload: dict) -> str:
    m = payload["meta"]
    g = m["gem5"]
    s = m["sniper"]
    lines = [
        "# Anchor cell-pair census",
        "",
        f"**Verdict:** {m['verdict']}  ",
        f"**Expected L3 axis:** {', '.join(m['expected_l3_axis'])}  ",
        f"**Expected policies:** {', '.join(m['expected_policies'])}  ",
        f"**Shared L3 axis:** {'✅' if m['shared_l3_axis'] else '❌'}  ",
        f"**Shared policy set:** {'✅' if m['shared_policies'] else '❌'}  ",
        f"**Shared (graph, app) cells:** {m['shared_cell_count']}  ",
        "",
        "## Anchor coverage",
        "",
        "| anchor | n_cells | min | records | l3_axis | policies |",
        "|---|---:|---:|---:|---|---|",
        f"| gem5   | {g['n_cells']} | {g['n_cells_minimum']} | "
        f"{g['cell_policy_records']} | "
        f"{', '.join(g['l3_axis'])} | "
        f"{', '.join(g['policies'])} |",
        f"| sniper | {s['n_cells']} | {s['n_cells_minimum']} | "
        f"{s['cell_policy_records']} | "
        f"{', '.join(s['l3_axis'])} | "
        f"{', '.join(s['policies'])} |",
        "",
        "## gem5 cells",
        "",
        "| graph | app |",
        "|---|---|",
    ]
    for graph, app in g["cells"]:
        lines.append(f"| {graph} | {app} |")

    lines += ["", "## Sniper cells", "", "| graph | app |", "|---|---|"]
    for graph, app in s["cells"]:
        lines.append(f"| {graph} | {app} |")

    lines += ["", "## Shared cells (both anchors)", ""]
    if m["shared_cells"]:
        lines += ["| graph | app |", "|---|---|"]
        for graph, app in m["shared_cells"]:
            lines.append(f"| {graph} | {app} |")
    else:
        lines.append("_None._")

    lines += [
        "",
        "## Verdict checks",
        "",
        "| check | result |",
        "|---|---|",
    ]
    for k, v in m["verdict_checks"].items():
        lines.append(f"| {k} | {'✅' if v else '❌'} |")

    lines += [
        "",
        "## Interpretation",
        "",
        "Pins gem5/Sniper anchor coverage so that downstream cross-tool "
        "gates (70, 71, 72, 74, 76) cannot silently lose explanatory "
        "power if anchor sweeps shrink. Shared L3 axis and shared "
        "policy set guarantee apples-to-apples cross-tool slope "
        "comparisons; at least one shared (graph, app) cell ensures "
        "per-cell parity spot-checks have a foothold.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gem5-json",   type=Path, default=DEFAULT_GEM5_JSON)
    ap.add_argument("--sniper-json", type=Path, default=DEFAULT_SNIPER_JSON)
    ap.add_argument("--json-out",    type=Path, default=DEFAULT_JSON_OUT)
    ap.add_argument("--md-out",      type=Path, default=DEFAULT_MD_OUT)
    args = ap.parse_args()

    payload = build(args.gem5_json, args.sniper_json)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    args.md_out.write_text(render_md(payload))

    m = payload["meta"]
    print(
        f"anchor-cell-census: "
        f"gem5_cells={m['gem5']['n_cells']}/{m['gem5']['n_cells_minimum']} "
        f"sniper_cells={m['sniper']['n_cells']}/{m['sniper']['n_cells_minimum']} "
        f"shared_cells={m['shared_cell_count']} "
        f"verdict={m['verdict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
