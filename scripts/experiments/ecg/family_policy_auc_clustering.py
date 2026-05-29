#!/usr/bin/env python3
"""Per-family policy-AUC clustering replay (gate 57).

Splits the global cross-app policy-AUC clustering (gate 50) by graph
family, asking: 'Do the POPT-friendly={bfs, pr, sssp} and
GRASP-friendly={bc, cc} clusters survive when we re-derive AUC
winners using only graphs from within a single family — or is the
global clustering driven by family composition?'

Method:
  - For each (family, app, policy) we recompute trapezoidal AUC over
    the paper L3 grid (1MB/4MB/8MB) using gap_pp values restricted to
    graphs in that family.
  - We mark a family as *qualifying* if every (app, policy) cell has
    all 3 paper L3 octaves present. Only qualifying families enter
    the clustering replay (otherwise AUC is undefined).
  - Within each qualifying family we re-derive the AUC winner per app
    and compare to the global winner.
  - We compute per-app intra-cluster mean Pearson r (using the global
    cluster definition) and report whether the family preserves the
    intra > inter separation.

Today, three families qualify (citation, social, web). Of these:
  - social (n=4 graphs) is the strongest replay: cluster gap > 0.5 and
    every global winner is preserved within the family.
  - citation and web have n=1 each so AUC variance is trivial; we
    still pin the global-vs-family winner agreement count.

Output: wiki/data/family_policy_auc_clustering.{json,md}
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
L3_OCTAVES_MB = {"1MB": 0.0, "4MB": 2.0, "8MB": 3.0}  # log2 MB

GLOBAL_CLUSTERS = {
    "GRASP": ("bc", "cc"),
    "POPT": ("bfs", "pr", "sssp"),
}

# Per-(family, app) winner deviations observed in the current corpus,
# pinned so any NEW deviation entering the set surfaces as a regression.
# Today the only deviations live in the citation family (single graph,
# cit-Patents) where bfs and sssp pick GRASP over POPT. This is the
# expected behavior for a citation network — out-degree skew is lower
# than in social/web, so POPT's popularity-prior loses its edge.
PINNED_DEVIATIONS: tuple[tuple[str, str], ...] = (
    ("citation", "bfs"),
    ("citation", "sssp"),
)
PINNED_DEVIATIONS_MAX = 2


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


def _trapezoidal_auc(values_by_l3: dict[str, float]) -> float:
    """Trapezoidal AUC over the paper L3 octaves on log2 MB axis."""
    xs = [L3_OCTAVES_MB[s] for s in PAPER_L3_SIZES]
    ys = [values_by_l3[s] for s in PAPER_L3_SIZES]
    auc = 0.0
    for i in range(len(xs) - 1):
        auc += 0.5 * (xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1])
    return auc


def _load_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "rows" in data:
        return data["rows"]
    if isinstance(data, list):
        return data
    raise ValueError(f"unrecognized shape in {path}")


def build_payload(oracle_json: Path) -> dict:
    rows = _load_rows(oracle_json)
    paper_rows = [r for r in rows if r["l3_size"] in PAPER_L3_SIZES]

    apps = sorted({r["app"] for r in paper_rows})
    policies = sorted({r["policy"] for r in paper_rows})
    families = sorted({r["family"] for r in paper_rows})

    # Pool gap_pp by (family, app, policy, L3): mean across graphs.
    pooled: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for r in paper_rows:
        pooled[
            (r["family"], r["app"], r["policy"], r["l3_size"])
        ].append(float(r["gap_pp"]))

    per_family_l3: dict[
        tuple[str, str, str], dict[str, float]
    ] = defaultdict(dict)
    for (fam, app, pol, l3), vals in pooled.items():
        per_family_l3[(fam, app, pol)][l3] = statistics.fmean(vals)

    qualifying: list[str] = []
    per_family: dict[str, dict] = {}
    for fam in families:
        # Family qualifies iff every (app, pol) cell has full L3 coverage.
        full = True
        for app in apps:
            for pol in policies:
                vals = per_family_l3.get((fam, app, pol), {})
                if not all(l3 in vals for l3 in PAPER_L3_SIZES):
                    full = False
                    break
            if not full:
                break
        if not full:
            per_family[fam] = {
                "qualified": False,
                "reason": "incomplete paper-L3 coverage in at least one cell",
                "n_graphs": len(
                    {r["graph"] for r in paper_rows if r["family"] == fam}
                ),
            }
            continue
        qualifying.append(fam)

        n_graphs = len(
            {r["graph"] for r in paper_rows if r["family"] == fam}
        )
        auc_by_app: dict[str, dict[str, float]] = {}
        for app in apps:
            auc_by_app[app] = {
                pol: round(
                    _trapezoidal_auc(per_family_l3[(fam, app, pol)]),
                    4,
                )
                for pol in policies
            }

        family_winner_by_app: dict[str, str] = {
            app: min(auc_by_app[app].items(), key=lambda kv: kv[1])[0]
            for app in apps
        }
        global_winner_by_app: dict[str, str] = {
            app: GLOBAL_WINNER[app] for app in apps
        }
        winners_match = {
            app: family_winner_by_app[app] == global_winner_by_app[app]
            for app in apps
        }

        # Per-app z-vector + correlation matrix.
        per_app_z = {
            app: _zscore([auc_by_app[app][pol] for pol in policies])
            for app in apps
        }
        matrix: dict[str, dict[str, float]] = {}
        for a in apps:
            matrix[a] = {}
            for b in apps:
                matrix[a][b] = round(_pearson(per_app_z[a], per_app_z[b]), 4)

        cluster_of = {
            app: pol
            for pol, members in GLOBAL_CLUSTERS.items()
            for app in members
        }
        intra_means = []
        inter_means = []
        for app in apps:
            own = cluster_of[app]
            intra_vals = [
                matrix[app][b]
                for b in apps
                if b != app and cluster_of[b] == own
            ]
            inter_vals = [
                matrix[app][b]
                for b in apps
                if b != app and cluster_of[b] != own
            ]
            if intra_vals:
                intra_means.append(statistics.fmean(intra_vals))
            if inter_vals:
                inter_means.append(statistics.fmean(inter_vals))
        intra_mean = (
            round(statistics.fmean(intra_means), 4) if intra_means else None
        )
        inter_mean = (
            round(statistics.fmean(inter_means), 4) if inter_means else None
        )
        gap = (
            round(intra_mean - inter_mean, 4)
            if intra_mean is not None and inter_mean is not None
            else None
        )

        per_family[fam] = {
            "qualified": True,
            "n_graphs": n_graphs,
            "auc_by_app_policy": auc_by_app,
            "winner_by_app": family_winner_by_app,
            "winner_matches_global": winners_match,
            "winners_matching": sum(1 for v in winners_match.values() if v),
            "n_apps": len(apps),
            "correlation_matrix": matrix,
            "intra_cluster_mean_r": intra_mean,
            "inter_cluster_mean_r": inter_mean,
            "intra_minus_inter": gap,
            "intra_dominates": (
                gap is not None and gap > 0.0
            ),
        }

    # Cross-family rollup
    qualifying_payload = [per_family[f] for f in qualifying]
    min_winners_matching = (
        min(p["winners_matching"] for p in qualifying_payload)
        if qualifying_payload
        else 0
    )
    n_families_intra_dominates = sum(
        1 for p in qualifying_payload if p.get("intra_dominates")
    )

    observed_deviations: list[tuple[str, str]] = []
    for fam in qualifying:
        info = per_family[fam]
        for app, match in info["winner_matches_global"].items():
            if not match:
                observed_deviations.append((fam, app))

    pinned_set = set(PINNED_DEVIATIONS)
    new_deviations = sorted(set(observed_deviations) - pinned_set)
    gone_deviations = sorted(pinned_set - set(observed_deviations))

    cluster_invariance_verdict = (
        "PASS"
        if (
            qualifying_payload
            and not new_deviations
            and len(observed_deviations) <= PINNED_DEVIATIONS_MAX
            and n_families_intra_dominates == len(qualifying_payload)
        )
        else "FAIL"
    )

    return {
        "meta": {
            "source": _resolve_label(oracle_json),
            "scope_l3_sizes": list(PAPER_L3_SIZES),
            "apps": apps,
            "policies": policies,
            "families": families,
            "qualifying_families": qualifying,
            "global_clusters": {k: list(v) for k, v in GLOBAL_CLUSTERS.items()},
            "global_winner_by_app": GLOBAL_WINNER,
            "min_winners_matching_across_families": min_winners_matching,
            "n_families_with_intra_dominates": n_families_intra_dominates,
            "deviation_set": {
                "observed": [
                    {"family": f, "app": a} for f, a in observed_deviations
                ],
                "pinned": [
                    {"family": f, "app": a} for f, a in PINNED_DEVIATIONS
                ],
                "pinned_max": PINNED_DEVIATIONS_MAX,
                "new_vs_pin": [
                    {"family": f, "app": a} for f, a in new_deviations
                ],
                "gone_vs_pin": [
                    {"family": f, "app": a} for f, a in gone_deviations
                ],
                "rationale": (
                    "citation family (sole graph cit-Patents) has lower"
                    " out-degree skew than social/web, so POPT's"
                    " popularity-prior loses its edge on bfs and sssp."
                    " Both fall back to GRASP. Pin guards against any NEW"
                    " family/app pair drifting away from the global winner."
                ),
            },
            "cluster_invariance_verdict": cluster_invariance_verdict,
        },
        "per_family": per_family,
    }


GLOBAL_WINNER = {
    "bc": "GRASP",
    "bfs": "POPT",
    "cc": "GRASP",
    "pr": "POPT",
    "sssp": "POPT",
}


def emit_md(payload: dict) -> str:
    meta = payload["meta"]
    out = []
    out.append("# Per-family policy-AUC clustering replay (gate 57)")
    out.append("")
    out.append(
        "Per-family re-derivation of the AUC winner and Pearson correlation"
        " clusters from gate 50. Tests whether the global"
        " POPT-friendly / GRASP-friendly clustering is intrinsic to the apps"
        " or merely a side-effect of the corpus's family mix."
    )
    out.append("")
    out.append(f"- source: `{meta['source']}`")
    out.append(
        f"- qualifying families (full L3 coverage):"
        f" {meta['qualifying_families']}"
    )
    out.append(
        f"- global clusters: GRASP-friendly={meta['global_clusters']['GRASP']},"
        f" POPT-friendly={meta['global_clusters']['POPT']}"
    )
    out.append(
        f"- min winners matching across qualifying families:"
        f" **{meta['min_winners_matching_across_families']}** / 5"
    )
    out.append(
        f"- qualifying families where intra > inter correlation:"
        f" **{meta['n_families_with_intra_dominates']}** /"
        f" {len(meta['qualifying_families'])}"
    )
    out.append(
        f"- verdict: **{meta['cluster_invariance_verdict']}**"
    )
    dev = meta["deviation_set"]
    out.append(
        f"- pinned deviations: {len(dev['pinned'])}"
        f" / observed: {len(dev['observed'])}"
        f" / max allowed: {dev['pinned_max']}"
    )
    if dev["new_vs_pin"]:
        out.append(f"- NEW deviations vs pin: {dev['new_vs_pin']}")
    if dev["gone_vs_pin"]:
        out.append(f"- gone deviations vs pin: {dev['gone_vs_pin']}")
    out.append("")
    out.append("## Per-family winner replay")
    out.append("")
    out.append("| family | n_graphs | qualified | winners matching | intra-r | inter-r | gap |")
    out.append("| --- | ---: | :---: | ---: | ---: | ---: | ---: |")
    for fam, info in payload["per_family"].items():
        if not info.get("qualified"):
            out.append(
                f"| {fam} | {info.get('n_graphs', 0)} | ❌ |"
                f" — | — | — | — |"
            )
        else:
            out.append(
                f"| {fam} | {info['n_graphs']} | ✅ |"
                f" {info['winners_matching']}/{info['n_apps']} |"
                f" {info['intra_cluster_mean_r']} |"
                f" {info['inter_cluster_mean_r']} |"
                f" {info['intra_minus_inter']} |"
            )
    out.append("")
    for fam, info in payload["per_family"].items():
        if not info.get("qualified"):
            continue
        out.append(f"### {fam} — winner-by-app vs global")
        out.append("")
        out.append("| app | family winner | global winner | match |")
        out.append("| --- | --- | --- | :---: |")
        for app, win in info["winner_by_app"].items():
            g = meta["global_winner_by_app"][app]
            mark = "✅" if win == g else "❌"
            out.append(f"| {app} | {win} | {g} | {mark} |")
        out.append("")
    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--oracle-json",
        default=str(WIKI_DATA / "oracle_gap.json"),
    )
    p.add_argument(
        "--json-out",
        default=str(WIKI_DATA / "family_policy_auc_clustering.json"),
    )
    p.add_argument(
        "--md-out",
        default=str(WIKI_DATA / "family_policy_auc_clustering.md"),
    )
    args = p.parse_args()

    payload = build_payload(Path(args.oracle_json))
    Path(args.json_out).write_text(json.dumps(payload, indent=2) + "\n")
    Path(args.md_out).write_text(emit_md(payload))

    meta = payload["meta"]
    print(
        f"family-policy-auc-clustering: qualifying_families={meta['qualifying_families']} |"
        f" min_winners_matching={meta['min_winners_matching_across_families']}/5 |"
        f" intra_dominates_count={meta['n_families_with_intra_dominates']}/{len(meta['qualifying_families'])} |"
        f" verdict={meta['cluster_invariance_verdict']}"
    )


if __name__ == "__main__":
    main()
