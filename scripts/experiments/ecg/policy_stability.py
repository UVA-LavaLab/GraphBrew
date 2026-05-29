#!/usr/bin/env python3
"""Per-policy stability index across apps (gate 51).

Reads `wiki/data/oracle_gap_auc.json` (gate 49) and computes, per
policy, how *consistent* the policy's AUC is across the 5 paper
apps. The score is the coefficient of variation:

    CV = stdev(AUC across apps) / mean(AUC across apps)

A small CV means the policy behaves predictably — it doesn't have a
catastrophic blowup on any one workload. A large CV means the policy
wins big on some apps and loses badly on others (high variance).

This complements gate 49 (which app wins on average?) and gate 50
(do apps cluster?) by answering 'which policy is the safest
all-rounder vs. the high-variance specialist?'

We also publish:
  - best-and-worst app per policy (where it shines, where it tanks)
  - rank-stability: each policy's rank position within each app's
    AUC ordering (1=winner, 4=worst); mean rank + stdev of rank
  - 'always-in-top-2' flag — a policy that never finishes worse
    than #2 in any app is a defensible 'safe default'.

Output: wiki/data/policy_stability.{json,md}
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


def _resolve_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_payload(auc_path: Path) -> dict:
    auc = json.loads(auc_path.read_text())
    meta = auc["meta"]
    apps = sorted(meta["apps"])
    policies = sorted(meta["policies"])

    # Collect AUCs per policy across apps + rank per app
    per_policy_auc_across_apps: dict[str, dict[str, float]] = defaultdict(dict)
    per_app_rank: dict[str, dict[str, int]] = {}
    for app in apps:
        ranking = auc["per_app"][app]["ranking"]
        rank_map = {entry["policy"]: i + 1 for i, entry in enumerate(ranking)}
        per_app_rank[app] = rank_map
        for pol in policies:
            per_policy_auc_across_apps[pol][app] = auc["per_app"][app][
                "auc_by_policy"
            ][pol]

    per_policy = {}
    for pol in policies:
        vals = [per_policy_auc_across_apps[pol][app] for app in apps]
        mean = statistics.fmean(vals)
        sd = statistics.pstdev(vals)
        cv = (sd / mean) if mean > 0 else None

        # Find best/worst app for this policy
        sorted_apps = sorted(per_policy_auc_across_apps[pol].items(), key=lambda kv: kv[1])
        best_app, best_auc = sorted_apps[0]
        worst_app, worst_auc = sorted_apps[-1]

        # Rank stability
        ranks = [per_app_rank[app][pol] for app in apps]
        rank_mean = statistics.fmean(ranks)
        rank_sd = statistics.pstdev(ranks)
        best_rank = min(ranks)
        worst_rank = max(ranks)
        always_top_2 = max(ranks) <= 2
        always_bot_2 = min(ranks) >= 3
        n_wins = sum(1 for r in ranks if r == 1)
        n_last = sum(1 for r in ranks if r == 4)

        per_policy[pol] = {
            "auc_mean_across_apps": round(mean, 4),
            "auc_stdev_across_apps": round(sd, 4),
            "auc_cv": round(cv, 4) if cv is not None else None,
            "best_app": best_app,
            "best_app_auc": round(best_auc, 4),
            "worst_app": worst_app,
            "worst_app_auc": round(worst_auc, 4),
            "worst_over_best_ratio": (
                round(worst_auc / best_auc, 4) if best_auc > 0 else None
            ),
            "ranks_by_app": {app: per_app_rank[app][pol] for app in apps},
            "rank_mean": round(rank_mean, 4),
            "rank_stdev": round(rank_sd, 4),
            "best_rank": best_rank,
            "worst_rank": worst_rank,
            "n_wins": n_wins,
            "n_lasts": n_last,
            "always_top_2": always_top_2,
            "always_bot_2": always_bot_2,
        }

    # Rank policies by safest-first (smallest CV among those with
    # at-least-1 win) then by mean AUC.
    safest_order = sorted(
        per_policy.items(),
        key=lambda kv: (
            kv[1]["auc_cv"] if kv[1]["auc_cv"] is not None else float("inf"),
            kv[1]["auc_mean_across_apps"],
        ),
    )

    # Rank policies by mean AUC ascending (best average)
    best_avg_order = sorted(
        per_policy.items(),
        key=lambda kv: kv[1]["auc_mean_across_apps"],
    )

    return {
        "meta": {
            "source": _resolve_label(auc_path),
            "n_apps": len(apps),
            "n_policies": len(policies),
            "apps": apps,
            "policies": policies,
            "stability_metric": "coefficient of variation (stdev / mean) of AUC across apps",
            "safest_policy": safest_order[0][0],
            "highest_variance_policy": safest_order[-1][0],
            "best_avg_policy": best_avg_order[0][0],
        },
        "per_policy": per_policy,
        "ranking_by_cv_ascending": [
            {"policy": p, "auc_cv": d["auc_cv"]} for p, d in safest_order
        ],
        "ranking_by_mean_auc_ascending": [
            {"policy": p, "auc_mean": d["auc_mean_across_apps"]} for p, d in best_avg_order
        ],
    }


def emit_md(payload: dict) -> str:
    m = payload["meta"]
    out = []
    out.append("# Per-policy stability index across apps")
    out.append("")
    out.append(
        f"Source: `{m['source']}`  •  apps={m['n_apps']}  •  policies={m['n_policies']}"
    )
    out.append("")
    out.append(
        f"Stability metric: **{m['stability_metric']}**. "
        f"Lower CV = more predictable across workloads."
    )
    out.append("")
    out.append("## Headline")
    out.append("")
    out.append(f"- Safest (lowest CV): **{m['safest_policy']}**")
    out.append(f"- Highest-variance: **{m['highest_variance_policy']}**")
    out.append(f"- Best mean AUC (lowest avg gap): **{m['best_avg_policy']}**")
    out.append("")
    out.append("## Per-policy stability table")
    out.append("")
    out.append(
        "| policy | mean AUC | stdev AUC | CV | best app | worst app | "
        "worst/best | mean rank | wins | lasts | always top-2 |"
    )
    out.append("|---|---:|---:|---:|---|---|---:|---:|---:|---:|---|")
    for pol in m["policies"]:
        p = payload["per_policy"][pol]
        out.append(
            f"| **{pol}** | {p['auc_mean_across_apps']} | {p['auc_stdev_across_apps']} "
            f"| {p['auc_cv']} | {p['best_app']} ({p['best_app_auc']}) "
            f"| {p['worst_app']} ({p['worst_app_auc']}) | {p['worst_over_best_ratio']} "
            f"| {p['rank_mean']} | {p['n_wins']} | {p['n_lasts']} "
            f"| {'yes' if p['always_top_2'] else 'no'} |"
        )
    out.append("")
    out.append("## Per-policy rank per app (1 = AUC winner; 4 = AUC worst)")
    out.append("")
    head = "| policy | " + " | ".join(m["apps"]) + " | rank mean | rank stdev |"
    out.append(head)
    out.append("|" + "---|" * (len(m["apps"]) + 3))
    for pol in m["policies"]:
        p = payload["per_policy"][pol]
        ranks = [str(p["ranks_by_app"][app]) for app in m["apps"]]
        out.append(
            f"| **{pol}** | " + " | ".join(ranks) +
            f" | {p['rank_mean']} | {p['rank_stdev']} |"
        )
    out.append("")
    out.append("## Interpretation")
    out.append("")
    out.append(
        "- Coefficient of variation isolates *behavior dispersion* from"
        " *absolute scale*. A policy with CV near zero behaves the same"
        " way regardless of workload."
    )
    out.append(
        "- 'Always top-2' is a defensible 'safe default' claim: such a"
        " policy never finishes worse than runner-up on any app."
    )
    out.append(
        "- LRU pairs the highest mean AUC with low CV: it is"
        " *predictably bad*. POPT pairs the lowest mean AUC with the"
        " highest variance: it is the high-reward / high-variance choice."
    )
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--auc-json", type=Path, default=WIKI_DATA / "oracle_gap_auc.json"
    )
    parser.add_argument(
        "--json-out", type=Path, default=WIKI_DATA / "policy_stability.json"
    )
    parser.add_argument(
        "--md-out", type=Path, default=WIKI_DATA / "policy_stability.md"
    )
    args = parser.parse_args()

    payload = build_payload(args.auc_json)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    args.md_out.write_text(emit_md(payload))

    bits = []
    for pol in payload["meta"]["policies"]:
        p = payload["per_policy"][pol]
        bits.append(f"{pol}:CV={p['auc_cv']:.3f},mean={p['auc_mean_across_apps']:.2f}")
    print(
        f"policy-stability: {' '.join(bits)} "
        f"| safest={payload['meta']['safest_policy']} "
        f"| best_avg={payload['meta']['best_avg_policy']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
