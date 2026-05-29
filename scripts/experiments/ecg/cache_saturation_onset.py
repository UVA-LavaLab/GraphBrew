#!/usr/bin/env python3
"""Cache-saturation onset detection (gate 55).

For each (app, policy) trajectory, identify the L3 size beyond which
additional cache buys negligible gap_pp improvement — the 'saturation
point'. POPT should saturate earliest (it's already near-oracle even
at small caches); LRU should rarely or never saturate at the paper
L3 scope (still benefits from every doubling).

Concretely, for each octave (X -> 2X) compute the gap shrinkage rate
in pp/octave. A cell is 'saturated at L3=Y' if all octaves Y..end
have shrinkage rate below SATURATION_THRESHOLD_PP.

Output: wiki/data/cache_saturation_onset.{json,md}
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
SATURATION_THRESHOLD_PP = 0.5  # pp/octave below this counts as saturated


def _resolve_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _onset(octaves: list[dict]) -> str:
    """Smallest L3 from which all remaining octaves are below threshold.

    Returns the L3 label where saturation begins, or 'never' if even
    the last octave still has appreciable shrinkage. Anti-scaling
    octaves (positive delta_gap_pp) count as 'not saturated'.
    """
    for i, oct_ in enumerate(octaves):
        # require all octaves from this index onward to be flat
        if all(
            o["delta_gap_pp"] > -SATURATION_THRESHOLD_PP
            and o["delta_gap_pp"] <= 0
            for o in octaves[i:]
        ):
            return oct_["from"]
    # check if even the final octave is flat
    last = octaves[-1]
    if last["delta_gap_pp"] > -SATURATION_THRESHOLD_PP and last["delta_gap_pp"] <= 0:
        return last["from"]
    return "never"


def build_payload(auc_path: Path) -> dict:
    auc = json.loads(auc_path.read_text())
    per_app: dict[str, dict[str, dict]] = {}
    onset_summary: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_policy_apps: dict[str, list[dict]] = defaultdict(list)

    for app, app_blob in auc["per_app"].items():
        per_app[app] = {}
        for pol, traj in app_blob["trajectory_by_policy"].items():
            sizes = [s for s in PAPER_L3 if s in traj]
            if len(sizes) < 2:
                continue
            octaves = []
            for src, dst in zip(sizes, sizes[1:]):
                d_log = math.log2(L3_MB[dst]) - math.log2(L3_MB[src])
                d_gap = traj[dst] - traj[src]
                octaves.append({
                    "from": src,
                    "to": dst,
                    "gap_from": round(traj[src], 4),
                    "gap_to": round(traj[dst], 4),
                    "delta_gap_pp": round(d_gap, 4),
                    "slope_pp_per_octave": (
                        round(-d_gap / d_log, 4) if d_log > 0 else 0.0
                    ),
                })
            onset = _onset(octaves)
            saturated = onset != "never"
            per_app[app][pol] = {
                "octaves": octaves,
                "saturation_onset": onset,
                "saturated_within_paper_l3": saturated,
                "final_octave_slope_pp": octaves[-1]["slope_pp_per_octave"],
                "final_octave_delta_pp": octaves[-1]["delta_gap_pp"],
            }
            onset_summary[pol][onset] += 1
            per_policy_apps[pol].append({
                "app": app,
                "onset": onset,
                "final_slope": octaves[-1]["slope_pp_per_octave"],
            })

    per_policy_view = {
        pol: {
            "onset_counts": dict(onsets),
            "apps": sorted(per_policy_apps[pol], key=lambda d: d["app"]),
            "n_saturated": sum(
                v for k, v in onsets.items() if k != "never"
            ),
            "n_never_saturated": onsets.get("never", 0),
        }
        for pol, onsets in onset_summary.items()
    }

    # Rank policies by how early they saturate (more 1MB-saturated wins).
    saturation_rank = sorted(
        per_policy_view.items(),
        key=lambda kv: (
            -kv[1]["onset_counts"].get("1MB", 0),
            -kv[1]["onset_counts"].get("4MB", 0),
            kv[1]["n_never_saturated"],
        ),
    )

    return {
        "meta": {
            "source": _resolve_label(auc_path),
            "scope_l3_sizes": list(PAPER_L3),
            "saturation_threshold_pp_per_octave": SATURATION_THRESHOLD_PP,
            "n_apps": len(per_app),
            "n_policies": max((len(b) for b in per_app.values()), default=0),
            "apps": sorted(per_app.keys()),
            "policies": sorted(per_policy_view.keys()),
            "saturation_rank_by_policy": [k for k, _ in saturation_rank],
        },
        "per_app": per_app,
        "per_policy": per_policy_view,
    }


def emit_md(payload: dict) -> str:
    m = payload["meta"]
    out = []
    out.append("# Cache-saturation onset detection")
    out.append("")
    out.append(
        f"Source: `{m['source']}`  •  paper L3: {', '.join(m['scope_l3_sizes'])}  "
        f"•  saturation threshold: **{m['saturation_threshold_pp_per_octave']} pp/octave**"
    )
    out.append("")
    out.append(
        "A cell is 'saturated at L3=Y' if every L3 octave at Y or larger "
        "shows a shrinkage rate below the threshold (positive deltas, "
        "i.e. anti-scaling, also disqualify)."
    )
    out.append("")
    out.append("## Per-policy saturation summary")
    out.append("")
    out.append(
        "| policy | saturated at 1MB | at 4MB | at 8MB | never | n apps |"
    )
    out.append("|---|---:|---:|---:|---:|---:|")
    for pol in m["policies"]:
        bucket = payload["per_policy"][pol]["onset_counts"]
        n_apps = sum(bucket.values())
        out.append(
            f"| **{pol}** | {bucket.get('1MB', 0)} | {bucket.get('4MB', 0)} "
            f"| {bucket.get('8MB', 0)} | {bucket.get('never', 0)} | {n_apps} |"
        )
    out.append("")
    out.append(
        "**Saturation ordering (earliest → latest):** "
        + " > ".join(m["saturation_rank_by_policy"])
    )
    out.append("")
    out.append("## Per-(app, policy) onset")
    out.append("")
    out.append(
        "| app | policy | onset | final-octave slope | final-octave delta_pp |"
    )
    out.append("|---|---|---|---:|---:|")
    for app in m["apps"]:
        for pol in m["policies"]:
            blob = payload["per_app"][app].get(pol)
            if not blob:
                continue
            out.append(
                f"| {app} | {pol} | {blob['saturation_onset']} "
                f"| {blob['final_octave_slope_pp']:.4f} "
                f"| {blob['final_octave_delta_pp']:+.4f} |"
            )
    out.append("")
    out.append("## Interpretation")
    out.append("")
    out.append(
        "- POPT should saturate earliest — it's near-oracle at every L3 — "
        "if not, the oracle-popularity hint is being wasted."
    )
    out.append(
        "- LRU and SRRIP rarely saturate at paper L3 — additional cache "
        "almost always helps them. This is the mechanism story: oracle-aware "
        "policies hit diminishing returns sooner because they're already "
        "close to ideal."
    )
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--auc-json", type=Path, default=WIKI_DATA / "oracle_gap_auc.json"
    )
    parser.add_argument(
        "--json-out", type=Path, default=WIKI_DATA / "cache_saturation_onset.json"
    )
    parser.add_argument(
        "--md-out", type=Path, default=WIKI_DATA / "cache_saturation_onset.md"
    )
    args = parser.parse_args()

    payload = build_payload(args.auc_json)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    args.md_out.write_text(emit_md(payload))

    m = payload["meta"]
    rank = " > ".join(m["saturation_rank_by_policy"])
    print(
        f"cache-saturation-onset: apps={m['n_apps']} policies={m['n_policies']} "
        f"| saturation rank: {rank} "
        f"| threshold={m['saturation_threshold_pp_per_octave']} pp/octave"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
