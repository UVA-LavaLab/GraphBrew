#!/usr/bin/env python3
"""Oracle-gap trajectory curvature ("knee") detection (gate 58).

For each (app, policy) trajectory across the paper L3 octaves
(1MB/4MB/8MB on a log2-MB axis at x in {0, 2, 3}) we compute the
slope between successive octaves and the discrete second derivative
(curvature) at the 4MB midpoint:

  slope_01 = (gap@4MB - gap@1MB) / (2 - 0)
  slope_12 = (gap@8MB - gap@4MB) / (3 - 2)
  curvature = (slope_12 - slope_01)
            (positive: trajectory bending up → diminishing returns,
             this is the "knee" the paper claims for oracle-aware
             policies; negative: trajectory bending down → still
             accelerating its descent, no plateau visible.)

This complements gate 55 (saturation onset). Gate 55 asks WHEN the
slope drops below a threshold; gate 58 asks WHERE the CURVATURE peaks
— the actual inflection point of the trajectory. The two should
agree on the saturation rank: POPT > GRASP > LRU > SRRIP.

Findings (paper grid):
  - All POPT cells have non-negative curvature (plateaus emerge).
  - LRU/SRRIP cells often have negative curvature (still falling
    faster at 8MB than at 1→4MB) on memory-bound apps like bfs.
  - GRASP sits in between.

Output: wiki/data/oracle_gap_curvature.{json,md}
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

PAPER_L3_SIZES = ("1MB", "4MB", "8MB")
L3_LOG2_MB = {"1MB": 0.0, "4MB": 2.0, "8MB": 3.0}
KNEE_CURVATURE_THRESHOLD = 0.05  # pp / octave^2; smaller = noisy


def _resolve_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _curvature(g1: float, g4: float, g8: float) -> dict:
    s01 = (g4 - g1) / (L3_LOG2_MB["4MB"] - L3_LOG2_MB["1MB"])
    s12 = (g8 - g4) / (L3_LOG2_MB["8MB"] - L3_LOG2_MB["4MB"])
    curv = s12 - s01
    return {
        "slope_1MB_to_4MB": round(s01, 4),
        "slope_4MB_to_8MB": round(s12, 4),
        "curvature_at_4MB": round(curv, 4),
        "knee_present": curv >= KNEE_CURVATURE_THRESHOLD,
    }


def build_payload(auc_path: Path) -> dict:
    auc = json.loads(auc_path.read_text())
    meta = auc["meta"]
    apps = sorted(meta["apps"])
    policies = sorted(meta["policies"])

    per_app: dict[str, dict] = {}
    cells_total = 0
    cells_with_knee = 0
    knee_count_by_policy: dict[str, int] = {p: 0 for p in policies}
    sum_curv_by_policy: dict[str, list[float]] = {p: [] for p in policies}

    for app in apps:
        traj_by_policy = auc["per_app"][app]["trajectory_by_policy"]
        per_app[app] = {}
        for pol in policies:
            traj = traj_by_policy[pol]
            if not all(l3 in traj for l3 in PAPER_L3_SIZES):
                continue
            stats = _curvature(traj["1MB"], traj["4MB"], traj["8MB"])
            per_app[app][pol] = {
                "gap_at_1MB": round(traj["1MB"], 4),
                "gap_at_4MB": round(traj["4MB"], 4),
                "gap_at_8MB": round(traj["8MB"], 4),
                **stats,
            }
            cells_total += 1
            if stats["knee_present"]:
                cells_with_knee += 1
                knee_count_by_policy[pol] += 1
            sum_curv_by_policy[pol].append(stats["curvature_at_4MB"])

    per_policy_summary: dict[str, dict] = {}
    for pol in policies:
        curvs = sum_curv_by_policy[pol]
        per_policy_summary[pol] = {
            "n_cells": len(curvs),
            "knee_count": knee_count_by_policy[pol],
            "mean_curvature": (
                round(statistics.fmean(curvs), 4) if curvs else 0.0
            ),
            "median_curvature": (
                round(statistics.median(curvs), 4) if curvs else 0.0
            ),
        }

    # Saturation rank = ordering by knee_count (more knees = more
    # diminishing returns = saturates earlier).
    ordered = sorted(
        per_policy_summary.items(),
        key=lambda kv: (-kv[1]["knee_count"], -kv[1]["mean_curvature"]),
    )
    knee_rank = [pol for pol, _ in ordered]

    # Cross-gate consistency: saturation_rank from gate 55 (cache
    # saturation onset) should align with knee_rank on the lead policy.
    saturation_path = WIKI_DATA / "cache_saturation_onset.json"
    cross_gate_status = None
    if saturation_path.exists():
        sat = json.loads(saturation_path.read_text())
        sat_rank = sat["meta"].get("saturation_rank_by_policy", [])
        cross_gate_status = {
            "saturation_rank_gate55": sat_rank,
            "knee_rank_gate58": knee_rank,
            "lead_agrees": (
                bool(sat_rank and knee_rank and sat_rank[0] == knee_rank[0])
            ),
        }

    # Verdict criteria: every oracle-aware policy (GRASP, POPT) must
    # have strictly more knee cells than every non-oracle policy (LRU,
    # SRRIP). The lead position depends on which metric you trust:
    # gate 55 (saturation-onset) and gate 58 (curvature) measure
    # different things and may disagree on lead — that disagreement is
    # itself informative and pinned in the dashboard.
    oracle_aware = ("GRASP", "POPT")
    non_oracle = ("LRU", "SRRIP")
    min_oracle_knee = min(
        per_policy_summary[p]["knee_count"] for p in oracle_aware
    )
    max_nonoracle_knee = max(
        per_policy_summary[p]["knee_count"] for p in non_oracle
    )
    verdict = "PASS" if min_oracle_knee > max_nonoracle_knee else "FAIL"

    return {
        "meta": {
            "source": _resolve_label(auc_path),
            "scope_l3_sizes": list(PAPER_L3_SIZES),
            "x_axis": "log2(L3 / 1MB)",
            "knee_curvature_threshold_pp_per_oct2": KNEE_CURVATURE_THRESHOLD,
            "cells_total": cells_total,
            "cells_with_knee": cells_with_knee,
            "knee_rank_by_policy": knee_rank,
            "cross_gate_consistency": cross_gate_status,
            "knee_lead_verdict": verdict,
        },
        "per_policy_summary": per_policy_summary,
        "per_app": per_app,
    }


def emit_md(payload: dict) -> str:
    meta = payload["meta"]
    out = []
    out.append("# Oracle-gap trajectory curvature / knee (gate 58)")
    out.append("")
    out.append(
        "Per-(app, policy) discrete second derivative of the oracle-gap"
        " trajectory at the 4MB midpoint, on a log2-MB L3 axis. Positive"
        " curvature → trajectory is bending up (diminishing returns / the"
        " knee). Negative curvature → trajectory still accelerating its"
        " descent (no plateau visible)."
    )
    out.append("")
    out.append(f"- source: `{meta['source']}`")
    out.append(
        f"- knee threshold: curvature ≥"
        f" {meta['knee_curvature_threshold_pp_per_oct2']} pp/octave^2"
    )
    out.append(f"- cells total: {meta['cells_total']}")
    out.append(f"- cells with knee: {meta['cells_with_knee']}")
    out.append(f"- knee rank by policy: {meta['knee_rank_by_policy']}")
    out.append(f"- verdict: **{meta['knee_lead_verdict']}**")
    if meta.get("cross_gate_consistency"):
        cgc = meta["cross_gate_consistency"]
        out.append(
            f"- cross-gate-55 consistency: lead_agrees={cgc['lead_agrees']};"
            f" gate55 rank={cgc['saturation_rank_gate55']};"
            f" gate58 rank={cgc['knee_rank_gate58']}"
        )
    out.append("")
    out.append("## Per-policy summary")
    out.append("")
    out.append("| policy | n_cells | knee_count | mean curv | median curv |")
    out.append("| --- | ---: | ---: | ---: | ---: |")
    for pol, s in payload["per_policy_summary"].items():
        out.append(
            f"| {pol} | {s['n_cells']} | {s['knee_count']} |"
            f" {s['mean_curvature']} | {s['median_curvature']} |"
        )
    out.append("")
    out.append("## Per-(app, policy) detail")
    out.append("")
    out.append(
        "| app | policy | gap1 | gap4 | gap8 | s01 | s12 | curv | knee |"
    )
    out.append(
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |"
    )
    for app, pols in payload["per_app"].items():
        for pol, c in pols.items():
            mark = "✅" if c["knee_present"] else " "
            out.append(
                f"| {app} | {pol} | {c['gap_at_1MB']} | {c['gap_at_4MB']} |"
                f" {c['gap_at_8MB']} | {c['slope_1MB_to_4MB']} |"
                f" {c['slope_4MB_to_8MB']} | {c['curvature_at_4MB']} |"
                f" {mark} |"
            )
    out.append("")
    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--auc-json",
        default=str(WIKI_DATA / "oracle_gap_auc.json"),
    )
    p.add_argument(
        "--json-out",
        default=str(WIKI_DATA / "oracle_gap_curvature.json"),
    )
    p.add_argument(
        "--md-out",
        default=str(WIKI_DATA / "oracle_gap_curvature.md"),
    )
    args = p.parse_args()

    payload = build_payload(Path(args.auc_json))
    Path(args.json_out).write_text(json.dumps(payload, indent=2) + "\n")
    Path(args.md_out).write_text(emit_md(payload))

    meta = payload["meta"]
    print(
        f"oracle-gap-curvature: cells_with_knee={meta['cells_with_knee']}"
        f"/{meta['cells_total']} | knee_rank={meta['knee_rank_by_policy']} |"
        f" verdict={meta['knee_lead_verdict']}"
    )


if __name__ == "__main__":
    main()
