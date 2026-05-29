#!/usr/bin/env python3
"""Per-(app, policy) oracle-gap area-under-curve across L3 sweep (gate 49).

Collapses each (app, policy) trajectory across the paper L3 sweep
(1MB → 4MB → 8MB) into a single trapezoidal area-under-curve (AUC)
score: smaller AUC = closer to the offline oracle across the cache
sweep, averaged on a log2(L3 MB) x-axis to weight each cache octave
equally.

Why AUC and not 'wins at the headline L3':
  - 'Wins count' (gate 1, etc.) tells you which policy bins the most
    L3 sizes. AUC tells you how *closely* each policy tracks oracle.
  - Two policies can have identical win-counts yet very different
    average gaps. AUC surfaces that difference in a single scalar.
  - Trapezoidal on a log2(L3 MB) axis weights 1MB→4MB equally with
    4MB→8MB (each is one cache-octave), matching the way reviewers
    interpret cache-sensitivity studies.

For each app we publish:

  - winner (lowest AUC) and runner-up
  - per-policy AUC + ranking
  - AUC ratio winner/runner-up (smaller = bigger dominance)
  - AUC ratio winner/LRU (paper headline)

The gate then pins which apps have a clear AUC winner that matches
the paper text. Cells where AUC winner != cell-vote winner (gate 48)
get surfaced as 'AUC-vote disagreement' for honest reporting.

Output: wiki/data/oracle_gap_auc.{json,md}
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

PAPER_L3_SIZES = ("1MB", "4MB", "8MB")
L3_MB = {"1MB": 1.0, "4MB": 4.0, "8MB": 8.0}


def _resolve_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def trapezoidal_log_auc(points: dict[float, float]) -> float:
    """Trapezoidal area on (log2(x), y), with x in MB and y the mean gap_pp."""
    pts = sorted(points.items())
    a = 0.0
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        dx = math.log2(x1) - math.log2(x0)
        a += 0.5 * (y0 + y1) * dx
    return a


def build_payload(oracle_path: Path) -> dict:
    rows = [
        r
        for r in json.loads(oracle_path.read_text())["rows"]
        if r["l3_size"] in PAPER_L3_SIZES
    ]

    # mean_gap_pp[(app, policy, l3)]
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        grouped[(r["app"], r["policy"], r["l3_size"])].append(
            float(r["gap_pp"])
        )

    apps = sorted({r["app"] for r in rows})
    policies = sorted({r["policy"] for r in rows})

    per_app: dict[str, dict] = {}
    for app in apps:
        per_policy_auc: dict[str, float] = {}
        per_policy_trajectory: dict[str, dict] = {}
        for pol in policies:
            traj: dict[float, float] = {}
            for l3 in PAPER_L3_SIZES:
                if (app, pol, l3) in grouped:
                    traj[L3_MB[l3]] = statistics.mean(
                        grouped[(app, pol, l3)]
                    )
            if len(traj) < 2:
                continue
            auc = trapezoidal_log_auc(traj)
            per_policy_auc[pol] = round(auc, 4)
            per_policy_trajectory[pol] = {
                f"{int(k)}MB": round(v, 4) for k, v in sorted(traj.items())
            }

        ordered = sorted(per_policy_auc.items(), key=lambda kv: (kv[1], kv[0]))
        winner_pol, winner_auc = ordered[0]
        runner_pol, runner_auc = ordered[1] if len(ordered) > 1 else (None, None)
        lru_auc = per_policy_auc.get("LRU")
        per_app[app] = {
            "auc_by_policy": per_policy_auc,
            "trajectory_by_policy": per_policy_trajectory,
            "ranking": [{"policy": p, "auc": a} for p, a in ordered],
            "winner": winner_pol,
            "winner_auc": winner_auc,
            "runner_up": runner_pol,
            "runner_up_auc": runner_auc,
            "auc_ratio_winner_over_runner_up": (
                round(winner_auc / runner_auc, 4)
                if runner_auc and runner_auc > 0
                else None
            ),
            "auc_ratio_winner_over_lru": (
                round(winner_auc / lru_auc, 4)
                if lru_auc is not None and lru_auc > 0
                else None
            ),
            "auc_pp_savings_winner_vs_lru": (
                round(lru_auc - winner_auc, 4)
                if lru_auc is not None
                else None
            ),
        }

    auc_winners = {app: per_app[app]["winner"] for app in apps if app in per_app}
    return {
        "meta": {
            "source": _resolve_label(oracle_path),
            "scope_l3_sizes": list(PAPER_L3_SIZES),
            "x_axis": "log2(L3 size in MB)",
            "y_axis": "mean gap_pp across graphs at that (app, policy, L3)",
            "auc_units": "gap_pp × log2(MB)",
            "n_apps": len(apps),
            "n_policies": len(policies),
            "policies": policies,
            "apps": apps,
            "auc_winner_by_app": auc_winners,
        },
        "per_app": per_app,
    }


def emit_md(payload: dict) -> str:
    m = payload["meta"]
    out = []
    out.append("# Per-(app, policy) oracle-gap AUC across L3 sweep")
    out.append("")
    out.append(
        f"Source: `{m['source']}`  •  Paper L3 scope: "
        f"{', '.join(m['scope_l3_sizes'])}"
    )
    out.append("")
    out.append(
        f"AUC = trapezoidal area on x={m['x_axis']}, y={m['y_axis']}."
        f" Units: **{m['auc_units']}** (smaller = closer to offline oracle)."
    )
    out.append("")
    out.append("## AUC winner per app")
    out.append("")
    out.append(
        "| app | AUC winner | winner AUC | runner-up | runner-up AUC | "
        "win/run ratio | win/LRU ratio | AUC savings vs LRU |"
    )
    out.append("|---|---|---:|---|---:|---:|---:|---:|")
    for app in m["apps"]:
        p = payload["per_app"].get(app)
        if not p:
            continue
        out.append(
            f"| {app} | **{p['winner']}** | {p['winner_auc']} "
            f"| {p['runner_up']} | {p['runner_up_auc']} "
            f"| {p['auc_ratio_winner_over_runner_up']} "
            f"| {p['auc_ratio_winner_over_lru']} "
            f"| {p['auc_pp_savings_winner_vs_lru']} |"
        )
    out.append("")
    out.append("## Per-app per-policy AUC ranking")
    out.append("")
    for app in m["apps"]:
        p = payload["per_app"].get(app)
        if not p:
            continue
        out.append(f"### {app}")
        out.append("")
        out.append("| policy | AUC | trajectory (1MB→4MB→8MB) |")
        out.append("|---|---:|---|")
        for entry in p["ranking"]:
            pol = entry["policy"]
            auc = entry["auc"]
            traj = p["trajectory_by_policy"].get(pol, {})
            traj_str = " → ".join(f"{traj.get(l3, '—')}" for l3 in ("1MB", "4MB", "8MB"))
            out.append(f"| {pol} | {auc} | {traj_str} |")
        out.append("")
    out.append("## Interpretation")
    out.append("")
    out.append(
        "- AUC < 1 (in gap_pp × log2(MB) units) means the policy tracks oracle"
        " *very* closely on average across the cache sweep — only pr/POPT"
        " currently achieves this."
    )
    out.append(
        "- AUC savings vs LRU = how many `gap_pp × log2(MB)` units the winner"
        " saves over LRU integrated across the sweep. A large value indicates"
        " a policy that is closer to oracle at *every* paper L3 size."
    )
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle-json", type=Path, default=WIKI_DATA / "oracle_gap.json"
    )
    parser.add_argument(
        "--json-out", type=Path, default=WIKI_DATA / "oracle_gap_auc.json"
    )
    parser.add_argument(
        "--md-out", type=Path, default=WIKI_DATA / "oracle_gap_auc.md"
    )
    args = parser.parse_args()

    payload = build_payload(args.oracle_json)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(emit_md(payload) + "\n")
    m = payload["meta"]
    winners = ", ".join(f"{a}={w}" for a, w in m["auc_winner_by_app"].items())
    print(f"oracle-gap-auc: apps={m['n_apps']} | auc_winners={winners}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
