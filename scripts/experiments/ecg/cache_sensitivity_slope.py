#!/usr/bin/env python3
"""Per-(app, policy) cache-sensitivity slope across L3 octaves (gate 52).

Reads `wiki/data/oracle_gap_auc.json` (gate 49) and asks:

    'How fast does each policy's oracle gap shrink as L3 doubles?'

For each (app, policy) we have a trajectory of mean gap_pp at
{1MB, 4MB, 8MB}. This gate computes:

    - per-octave delta: gap_pp shrinkage from one L3 step to the next
      * 1MB -> 4MB is +2 octaves (log2)
      * 4MB -> 8MB is +1 octave
    - per-octave slope = -delta(gap_pp) / delta(log2_MB)
      (positive slope = gap shrinks as L3 grows — the expected sign)
    - monotonic_decreasing flag (gap should never grow as L3 grows)
    - average slope across the sweep (single-number cache-sensitivity)

Why this matters for the paper:

    1. Validates the implicit 'bigger cache = closer to oracle'
       assumption that underlies the L3 sweep design. If any (app,
       policy) violates monotonicity, that's a real anomaly worth
       discussing (or an oracle-trace bug worth fixing).

    2. Surfaces 'cache-saturating' policies — ones whose slope
       collapses near the headline L3, meaning extra cache buys you
       nothing. Conversely, 'cache-hungry' policies keep shrinking
       even at 8MB, suggesting they'd benefit from larger L3.

Output: wiki/data/cache_sensitivity_slope.{json,md}
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

L3_MB = {"1MB": 1.0, "4MB": 4.0, "8MB": 8.0}
ORDER = ("1MB", "4MB", "8MB")


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

    per_app = {}
    monotonic_violations = []
    all_avg_slopes = {pol: [] for pol in policies}

    for app in apps:
        per_policy = {}
        for pol in policies:
            traj = auc["per_app"][app]["trajectory_by_policy"][pol]
            # Guarantee all 3 L3 points present
            if not all(l3 in traj for l3 in ORDER):
                continue
            gap_at = {l3: traj[l3] for l3 in ORDER}

            octaves = []
            # 1MB -> 4MB: delta_log2 = 2
            # 4MB -> 8MB: delta_log2 = 1
            for src, dst in (("1MB", "4MB"), ("4MB", "8MB")):
                src_log = math.log2(L3_MB[src])
                dst_log = math.log2(L3_MB[dst])
                d_log = dst_log - src_log
                d_gap = gap_at[dst] - gap_at[src]
                slope = -d_gap / d_log if d_log > 0 else 0.0
                octaves.append({
                    "from": src,
                    "to": dst,
                    "gap_from": round(gap_at[src], 4),
                    "gap_to": round(gap_at[dst], 4),
                    "delta_gap_pp": round(d_gap, 4),
                    "delta_log2_mb": round(d_log, 4),
                    "slope_pp_per_octave": round(slope, 4),
                })

            mono = all(o["delta_gap_pp"] <= 1e-9 for o in octaves)
            if not mono:
                monotonic_violations.append({
                    "app": app,
                    "policy": pol,
                    "octaves": octaves,
                })

            total_d_log = math.log2(L3_MB["8MB"]) - math.log2(L3_MB["1MB"])
            avg_slope = (gap_at["1MB"] - gap_at["8MB"]) / total_d_log if total_d_log > 0 else 0.0
            all_avg_slopes[pol].append(avg_slope)

            per_policy[pol] = {
                "octaves": octaves,
                "avg_slope_pp_per_octave": round(avg_slope, 4),
                "monotonic_decreasing": mono,
                "gap_at_1MB": gap_at["1MB"],
                "gap_at_8MB": gap_at["8MB"],
                "total_shrinkage_pp": round(gap_at["1MB"] - gap_at["8MB"], 4),
            }
        per_app[app] = per_policy

    per_policy_summary = {}
    for pol in policies:
        slopes = all_avg_slopes[pol]
        if not slopes:
            continue
        per_policy_summary[pol] = {
            "n_apps": len(slopes),
            "mean_avg_slope": round(statistics.fmean(slopes), 4),
            "stdev_avg_slope": round(statistics.pstdev(slopes), 4),
            "max_slope": round(max(slopes), 4),
            "min_slope": round(min(slopes), 4),
        }

    return {
        "meta": {
            "source": _resolve_label(auc_path),
            "n_apps": len(apps),
            "n_policies": len(policies),
            "apps": apps,
            "policies": policies,
            "l3_octaves": list(ORDER),
            "slope_units": "gap_pp per L3 octave (log2 MB)",
            "n_monotonic_violations": len(monotonic_violations),
            "all_monotonic": len(monotonic_violations) == 0,
        },
        "per_app": per_app,
        "per_policy_summary": per_policy_summary,
        "monotonic_violations": monotonic_violations,
    }


def emit_md(payload: dict) -> str:
    m = payload["meta"]
    out = []
    out.append("# Per-(app, policy) cache-sensitivity slope across L3 octaves")
    out.append("")
    out.append(
        f"Source: `{m['source']}`  •  apps={m['n_apps']}  •  policies={m['n_policies']}"
        f"  •  L3 octaves: {' → '.join(m['l3_octaves'])}"
    )
    out.append("")
    out.append(
        f"Slope units: **{m['slope_units']}**. Positive slope means the"
        f" oracle gap shrinks as L3 grows (the expected sign for any"
        f" sensible policy)."
    )
    out.append("")
    out.append("## Headline")
    out.append("")
    if m["all_monotonic"]:
        out.append("- ✅ Every (app, policy) trajectory is monotonically non-increasing"
                   " in gap_pp as L3 grows (no inversion anomalies).")
    else:
        out.append(f"- ⚠️ {m['n_monotonic_violations']} (app, policy) cells violate"
                   f" monotonicity. See 'Monotonic violations' below.")
    out.append("")
    out.append("## Per-policy slope summary (mean across apps)")
    out.append("")
    out.append("| policy | mean slope | stdev slope | min slope | max slope | n_apps |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for pol in m["policies"]:
        s = payload["per_policy_summary"].get(pol)
        if not s:
            continue
        out.append(
            f"| **{pol}** | {s['mean_avg_slope']} | {s['stdev_avg_slope']} "
            f"| {s['min_slope']} | {s['max_slope']} | {s['n_apps']} |"
        )
    out.append("")
    out.append("## Per-(app, policy) avg slope (gap_pp shrinkage per L3 octave)")
    out.append("")
    head = "| app | " + " | ".join(m["policies"]) + " |"
    out.append(head)
    out.append("|" + "---|" * (len(m["policies"]) + 1))
    for app in m["apps"]:
        row = [f"**{app}**"]
        for pol in m["policies"]:
            d = payload["per_app"][app].get(pol)
            if d is None:
                row.append("—")
            else:
                row.append(f"{d['avg_slope_pp_per_octave']:.3f}")
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    if not m["all_monotonic"]:
        out.append("## Monotonic violations")
        out.append("")
        out.append("| app | policy | 1MB→4MB Δgap | 4MB→8MB Δgap |")
        out.append("|---|---|---:|---:|")
        for v in payload["monotonic_violations"]:
            d1, d2 = v["octaves"][0]["delta_gap_pp"], v["octaves"][1]["delta_gap_pp"]
            out.append(f"| {v['app']} | {v['policy']} | {d1} | {d2} |")
        out.append("")
    out.append("## Interpretation")
    out.append("")
    out.append(
        "- A high mean slope means the policy benefits a lot from each"
        " extra cache octave — *cache-hungry*."
    )
    out.append(
        "- A near-zero mean slope means extra cache buys little"
        " improvement — *cache-saturating*. POPT typically saturates"
        " fast because it's already close to the oracle ceiling."
    )
    out.append(
        "- Per-octave slope drop (1MB→4MB vs 4MB→8MB) reveals where a"
        " policy hits diminishing returns. The headline policy choice"
        " can vary depending on which L3 octave the design targets."
    )
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--auc-json", type=Path, default=WIKI_DATA / "oracle_gap_auc.json"
    )
    parser.add_argument(
        "--json-out", type=Path, default=WIKI_DATA / "cache_sensitivity_slope.json"
    )
    parser.add_argument(
        "--md-out", type=Path, default=WIKI_DATA / "cache_sensitivity_slope.md"
    )
    args = parser.parse_args()

    payload = build_payload(args.auc_json)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    args.md_out.write_text(emit_md(payload))

    bits = []
    for pol in payload["meta"]["policies"]:
        s = payload["per_policy_summary"].get(pol, {})
        bits.append(f"{pol}:slope={s.get('mean_avg_slope', '?')}")
    print(
        f"cache-sensitivity-slope: {' '.join(bits)} "
        f"| all_monotonic={payload['meta']['all_monotonic']} "
        f"| violations={payload['meta']['n_monotonic_violations']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
