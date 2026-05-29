#!/usr/bin/env python3
"""Per-graph oracle-gap cache-sensitivity slope (gate 53).

Like gate 52 but operating on the raw per-graph oracle-gap data
(wiki/data/oracle_gap.json) instead of the corpus-averaged
trajectories in gate 49. Asks the stronger question:

    'Does the GRASP/POPT-never-regress invariant hold at the
     per-graph level, not just the corpus-averaged level?'

If the answer is yes, the paper can claim the anti-scaling story
without footnotes — every individual graph confirms it. If the
answer is no, we surface which graphs break it.

For each (graph, app, policy) trajectory with full 1MB/4MB/8MB
coverage we compute:
  - per-octave delta_gap_pp
  - per-octave slope (gap shrinkage per log2(MB))
  - significant_anti_scaling flag (delta_gap_pp >= 1.0)

Aggregates:
  - per-policy: how many (graph, app) cells have significant
    anti-scaling, and on which (graph, app, octave)
  - per-(graph, policy): how many apps have anti-scaling
  - per-graph: total anti-scaling count

Output: wiki/data/per_graph_cache_slope.{json,md}
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

PAPER_L3 = ("1MB", "4MB", "8MB")
L3_MB = {"1MB": 1.0, "4MB": 4.0, "8MB": 8.0}
SIGNIFICANT_PP = 1.0


def _resolve_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_payload(oracle_path: Path) -> dict:
    raw = json.loads(oracle_path.read_text())
    # Index per (graph, app, policy, l3) -> gap_pp
    grid: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    families: dict[str, str] = {}
    for row in raw["rows"]:
        if row["l3_size"] not in PAPER_L3:
            continue
        key = (row["graph"], row["app"], row["policy"])
        grid[key][row["l3_size"]] = float(row["gap_pp"])
        families[row["graph"]] = row["family"]

    full_trajectories: dict[tuple[str, str, str], dict[str, float]] = {
        k: v for k, v in grid.items() if set(v.keys()) == set(PAPER_L3)
    }

    anti_scaling_cells: list[dict] = []
    per_policy_counts: dict[str, int] = defaultdict(int)
    per_graph_counts: dict[str, int] = defaultdict(int)
    per_graph_policy_counts: dict[tuple[str, str], int] = defaultdict(int)

    cell_records = []
    for (graph, app, policy), traj in full_trajectories.items():
        octaves = []
        for src, dst in (("1MB", "4MB"), ("4MB", "8MB")):
            d_log = math.log2(L3_MB[dst]) - math.log2(L3_MB[src])
            d_gap = traj[dst] - traj[src]
            slope = -d_gap / d_log if d_log > 0 else 0.0
            octaves.append({
                "from": src,
                "to": dst,
                "gap_from": round(traj[src], 4),
                "gap_to": round(traj[dst], 4),
                "delta_gap_pp": round(d_gap, 4),
                "slope_pp_per_octave": round(slope, 4),
                "significant_anti_scaling": d_gap >= SIGNIFICANT_PP,
            })
        any_anti = any(o["significant_anti_scaling"] for o in octaves)
        if any_anti:
            per_policy_counts[policy] += 1
            per_graph_counts[graph] += 1
            per_graph_policy_counts[(graph, policy)] += 1
            anti_scaling_cells.append({
                "graph": graph,
                "family": families.get(graph, "unknown"),
                "app": app,
                "policy": policy,
                "octaves": octaves,
                "max_pp_growth": round(
                    max(o["delta_gap_pp"] for o in octaves), 4
                ),
            })
        cell_records.append({
            "graph": graph,
            "family": families.get(graph, "unknown"),
            "app": app,
            "policy": policy,
            "octaves": octaves,
            "any_significant_anti_scaling": any_anti,
            "total_shrinkage_pp": round(traj["1MB"] - traj["8MB"], 4),
        })

    anti_scaling_cells.sort(key=lambda d: d["max_pp_growth"], reverse=True)

    policies = sorted({p for (_, _, p) in full_trajectories.keys()})
    apps = sorted({a for (_, a, _) in full_trajectories.keys()})
    graphs = sorted({g for (g, _, _) in full_trajectories.keys()})

    # The paper-grade headline metric:
    grasp_anti = per_policy_counts.get("GRASP", 0)
    popt_anti = per_policy_counts.get("POPT", 0)
    lru_anti = per_policy_counts.get("LRU", 0)
    srrip_anti = per_policy_counts.get("SRRIP", 0)
    total_oracle_aware_anti = grasp_anti + popt_anti

    return {
        "meta": {
            "source": _resolve_label(oracle_path),
            "scope_l3_sizes": list(PAPER_L3),
            "n_apps": len(apps),
            "n_policies": len(policies),
            "n_graphs_with_full_trajectory": len(graphs),
            "apps": apps,
            "policies": policies,
            "graphs": graphs,
            "n_full_trajectories": len(full_trajectories),
            "significant_pp_threshold": SIGNIFICANT_PP,
            "n_cells_with_significant_anti_scaling": len(anti_scaling_cells),
            "n_oracle_aware_anti_scaling": total_oracle_aware_anti,
        },
        "per_policy_anti_scaling_count": dict(per_policy_counts),
        "per_graph_anti_scaling_count": dict(per_graph_counts),
        "per_graph_policy_anti_scaling_count": {
            f"{g}|{p}": n for (g, p), n in per_graph_policy_counts.items()
        },
        "anti_scaling_cells": anti_scaling_cells,
        "all_trajectories": cell_records,
    }


def emit_md(payload: dict) -> str:
    m = payload["meta"]
    out = []
    out.append("# Per-graph oracle-gap cache-sensitivity slope")
    out.append("")
    out.append(
        f"Source: `{m['source']}`  •  paper L3: {', '.join(m['scope_l3_sizes'])}  "
        f"•  graphs with full trajectory: {m['n_graphs_with_full_trajectory']}  "
        f"•  full (graph, app, policy) trajectories: {m['n_full_trajectories']}"
    )
    out.append("")
    out.append(
        f"Significant anti-scaling threshold: **+{m['significant_pp_threshold']} pp** per octave. "
        f"A cell that grows its gap by that much (or more) at any single octave is flagged."
    )
    out.append("")
    out.append("## Headline")
    out.append("")
    out.append(
        f"- Cells with significant anti-scaling: "
        f"**{m['n_cells_with_significant_anti_scaling']} / {m['n_full_trajectories']}**"
    )
    out.append(
        f"- Of those, **{m['n_oracle_aware_anti_scaling']}** belong to "
        f"the oracle-aware policies (GRASP + POPT). The remainder are LRU + SRRIP."
    )
    out.append("")
    out.append("## Per-policy anti-scaling cell count")
    out.append("")
    out.append("| policy | n_cells_with_anti_scaling |")
    out.append("|---|---:|")
    for pol in m["policies"]:
        out.append(
            f"| **{pol}** | {payload['per_policy_anti_scaling_count'].get(pol, 0)} |"
        )
    out.append("")
    out.append("## Per-graph anti-scaling cell count")
    out.append("")
    out.append("| graph | family | n_cells_with_anti_scaling |")
    out.append("|---|---|---:|")
    fam_index = {c["graph"]: c["family"] for c in payload["all_trajectories"]}
    for g in m["graphs"]:
        out.append(
            f"| {g} | {fam_index.get(g, '?')} | "
            f"{payload['per_graph_anti_scaling_count'].get(g, 0)} |"
        )
    out.append("")
    out.append("## Top anti-scaling cells (largest single-octave gap growth)")
    out.append("")
    out.append(
        "| graph | family | app | policy | max octave growth | "
        "1MB → 4MB → 8MB gap |"
    )
    out.append("|---|---|---|---|---:|---|")
    for c in payload["anti_scaling_cells"][:25]:
        traj = (
            f"{c['octaves'][0]['gap_from']:.2f} → "
            f"{c['octaves'][0]['gap_to']:.2f} → "
            f"{c['octaves'][1]['gap_to']:.2f}"
        )
        out.append(
            f"| {c['graph']} | {c['family']} | {c['app']} | {c['policy']} "
            f"| +{c['max_pp_growth']:.2f} pp | {traj} |"
        )
    out.append("")
    out.append("## Interpretation")
    out.append("")
    out.append(
        "- If `n_oracle_aware_anti_scaling` is 0, the paper can claim:"
        " 'no individual graph shows GRASP or POPT regressing as L3 grows'"
        " — strictly stronger than the corpus-averaged finding in gate 52."
    )
    out.append(
        "- If a small handful of (graph, app) cells flag GRASP/POPT anti-scaling,"
        " those should be called out specifically in the paper as known"
        " exceptions worth disclosing."
    )
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle-json", type=Path, default=WIKI_DATA / "oracle_gap.json"
    )
    parser.add_argument(
        "--json-out", type=Path, default=WIKI_DATA / "per_graph_cache_slope.json"
    )
    parser.add_argument(
        "--md-out", type=Path, default=WIKI_DATA / "per_graph_cache_slope.md"
    )
    args = parser.parse_args()

    payload = build_payload(args.oracle_json)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    args.md_out.write_text(emit_md(payload))

    m = payload["meta"]
    pol_counts = payload["per_policy_anti_scaling_count"]
    bits = " ".join(f"{p}={pol_counts.get(p, 0)}" for p in m["policies"])
    print(
        f"per-graph-cache-slope: n_full_trajectories={m['n_full_trajectories']} "
        f"| anti_scaling_cells={m['n_cells_with_significant_anti_scaling']} "
        f"| per_policy: {bits} "
        f"| oracle_aware_anti={m['n_oracle_aware_anti_scaling']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
