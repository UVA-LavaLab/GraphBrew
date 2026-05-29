#!/usr/bin/env python3
"""Cross-app policy-AUC correlation matrix (gate 50).

Reads `wiki/data/oracle_gap_auc.json` (gate 49) and asks:

    'Do the apps cluster into POPT-friendly and GRASP-friendly classes,
     or does every app have its own idiosyncratic policy ordering?'

For each app we have a 4-element AUC vector
(GRASP, LRU, POPT, SRRIP). We then:

    1. z-score-normalize each vector within its app so absolute AUC
       magnitude does not dominate (cc's LRU AUC is 24.67, pr's only
       27.36 — but BFS sits around 13 — so apps live on very different
       scales).
    2. Compute pairwise Pearson correlation across the 4 policy
       dimensions for every (appA, appB) pair.
    3. Hierarchical-style report: per-app similarity ranking + a
       cluster summary that names which apps belong to which family.

A high correlation between two apps means 'these two apps rank
policies in the same order' — strong evidence for a shared
preference class (e.g., 'POPT-friendly').

Used by the paper to defend the claim that the AUC winners are not
random per-app coincidences but reflect two underlying classes
(pull-style traversal vs push-style frontier-bound).

Output: wiki/data/policy_auc_correlation.{json,md}
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WIKI_DATA = REPO_ROOT / "wiki" / "data"


def _resolve_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _zscore(vec: list[float]) -> list[float]:
    if len(vec) < 2:
        return [0.0] * len(vec)
    mu = statistics.fmean(vec)
    sd = statistics.pstdev(vec)
    if sd == 0:
        return [0.0] * len(vec)
    return [(x - mu) / sd for x in vec]


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mux = statistics.fmean(x)
    muy = statistics.fmean(y)
    num = sum((a - mux) * (b - muy) for a, b in zip(x, y))
    denx = math.sqrt(sum((a - mux) ** 2 for a in x))
    deny = math.sqrt(sum((b - muy) ** 2 for b in y))
    if denx == 0 or deny == 0:
        return 0.0
    return num / (denx * deny)


def build_payload(auc_path: Path) -> dict:
    auc = json.loads(auc_path.read_text())
    meta = auc["meta"]
    apps = sorted(meta["apps"])
    policies = sorted(meta["policies"])

    # Build per-app AUC vector, indexed identically across apps.
    per_app_vec: dict[str, list[float]] = {}
    for app in apps:
        per_app_vec[app] = [
            auc["per_app"][app]["auc_by_policy"][pol] for pol in policies
        ]

    per_app_z = {app: _zscore(vec) for app, vec in per_app_vec.items()}

    matrix: dict[str, dict[str, float]] = {}
    pair_list = []
    for a in apps:
        matrix[a] = {}
        for b in apps:
            r = _pearson(per_app_z[a], per_app_z[b])
            matrix[a][b] = round(r, 4)
            if a < b:
                pair_list.append({"app_a": a, "app_b": b, "pearson_r": round(r, 4)})

    pair_list.sort(key=lambda d: d["pearson_r"], reverse=True)

    # Cluster: for each app find its closest sibling and the closest
    # member that does not share its top policy.
    auc_winner_by_app = meta["auc_winner_by_app"]
    nearest_sibling = {}
    for app in apps:
        ranking = sorted(
            ((b, matrix[app][b]) for b in apps if b != app),
            key=lambda kv: kv[1],
            reverse=True,
        )
        nearest_sibling[app] = {
            "ranking": [{"app": b, "pearson_r": r} for b, r in ranking],
            "closest_app": ranking[0][0] if ranking else None,
            "closest_r": ranking[0][1] if ranking else None,
            "winner_policy": auc_winner_by_app.get(app),
        }

    # Auto-cluster by AUC-winner agreement (the paper's headline framing).
    clusters: dict[str, list[str]] = {}
    for app, pol in auc_winner_by_app.items():
        clusters.setdefault(pol, []).append(app)
    clusters_sorted = {
        pol: sorted(members) for pol, members in sorted(clusters.items())
    }

    # In-cluster vs out-cluster average correlation per app.
    cluster_of = {
        app: pol for pol, members in clusters_sorted.items() for app in members
    }
    intra_inter = {}
    for app in apps:
        own = cluster_of.get(app)
        intra_vals = [
            matrix[app][b]
            for b in apps
            if b != app and cluster_of.get(b) == own
        ]
        inter_vals = [
            matrix[app][b]
            for b in apps
            if b != app and cluster_of.get(b) != own
        ]
        intra_inter[app] = {
            "cluster": own,
            "intra_mean_r": round(statistics.fmean(intra_vals), 4) if intra_vals else None,
            "inter_mean_r": round(statistics.fmean(inter_vals), 4) if inter_vals else None,
            "gap_intra_minus_inter": (
                round(statistics.fmean(intra_vals) - statistics.fmean(inter_vals), 4)
                if intra_vals and inter_vals
                else None
            ),
        }

    return {
        "meta": {
            "source": _resolve_label(auc_path),
            "n_apps": len(apps),
            "n_policies": len(policies),
            "apps": apps,
            "policies": policies,
            "z_norm": "per-app z-score across the 4 policy AUC values",
            "auc_winner_by_app": auc_winner_by_app,
            "clusters_by_winner": clusters_sorted,
        },
        "matrix": matrix,
        "pair_list": pair_list,
        "nearest_sibling": nearest_sibling,
        "intra_inter": intra_inter,
    }


def emit_md(payload: dict) -> str:
    m = payload["meta"]
    out = []
    out.append("# Cross-app policy-AUC correlation matrix")
    out.append("")
    out.append(
        f"Source: `{m['source']}`  •  apps={m['n_apps']}  •  policies={m['n_policies']}"
    )
    out.append("")
    out.append(
        "Each app's AUC vector across 4 policies is z-normalized within "
        "the app, then pairwise Pearson correlations are taken across "
        "the 4 policy dimensions. r=+1 ⇒ two apps rank policies in the "
        "same order; r=-1 ⇒ exact opposite ordering."
    )
    out.append("")
    out.append("## Clusters by AUC winner")
    out.append("")
    out.append("| winner policy | apps |")
    out.append("|---|---|")
    for pol, members in m["clusters_by_winner"].items():
        out.append(f"| **{pol}** | {', '.join(members)} |")
    out.append("")
    out.append("## Correlation matrix (Pearson r on z-normalized AUC vectors)")
    out.append("")
    head = "| app | " + " | ".join(m["apps"]) + " |"
    out.append(head)
    out.append("|" + "---|" * (len(m["apps"]) + 1))
    for a in m["apps"]:
        row = [f"**{a}**"] + [f"{payload['matrix'][a][b]:+.3f}" for b in m["apps"]]
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    out.append("## In-cluster vs out-cluster mean correlation")
    out.append("")
    out.append("| app | cluster | intra mean r | inter mean r | intra-inter gap |")
    out.append("|---|---|---:|---:|---:|")
    for app in m["apps"]:
        ii = payload["intra_inter"][app]
        out.append(
            f"| {app} | {ii['cluster']} | {ii['intra_mean_r']} "
            f"| {ii['inter_mean_r']} | {ii['gap_intra_minus_inter']} |"
        )
    out.append("")
    out.append("## Top pairs by similarity")
    out.append("")
    out.append("| rank | app A | app B | Pearson r |")
    out.append("|---:|---|---|---:|")
    for i, pair in enumerate(payload["pair_list"], 1):
        out.append(f"| {i} | {pair['app_a']} | {pair['app_b']} | {pair['pearson_r']:+.3f} |")
    out.append("")
    out.append("## Interpretation")
    out.append("")
    out.append(
        "- A positive intra-inter gap means apps inside the same"
        " 'AUC-winner' cluster are more correlated than apps across"
        " clusters — strong evidence that AUC winners are not"
        " idiosyncratic per-app artifacts."
    )
    out.append(
        "- A negative gap would mean the AUC clustering is noise and"
        " the per-app winners do not reflect a shared structural"
        " preference."
    )
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--auc-json", type=Path, default=WIKI_DATA / "oracle_gap_auc.json"
    )
    parser.add_argument(
        "--json-out", type=Path, default=WIKI_DATA / "policy_auc_correlation.json"
    )
    parser.add_argument(
        "--md-out", type=Path, default=WIKI_DATA / "policy_auc_correlation.md"
    )
    args = parser.parse_args()

    payload = build_payload(args.auc_json)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    args.md_out.write_text(emit_md(payload))

    summary_bits = []
    for pol, members in payload["meta"]["clusters_by_winner"].items():
        summary_bits.append(f"{pol}:[{','.join(members)}]")
    top = payload["pair_list"][0] if payload["pair_list"] else None
    bot = payload["pair_list"][-1] if payload["pair_list"] else None
    print(
        f"policy-auc-correlation: clusters={' '.join(summary_bits)} "
        f"| top_pair={top['app_a']}+{top['app_b']}={top['pearson_r']:+.3f} "
        f"| bot_pair={bot['app_a']}+{bot['app_b']}={bot['pearson_r']:+.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
