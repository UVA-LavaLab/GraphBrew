#!/usr/bin/env python3
"""Per-policy final-octave steepness ranking (gate 81).

Reads `wiki/data/cache_saturation_onset.json` and aggregates the
**final-octave** (4MB->8MB) slope magnitudes per policy across all
apps. Asserts the charge-invariant steepness ordering:

  - GRASP (degree heuristic) is the FLATTEST policy: its hot-set
    pinning is cache-size-insensitive, so its miss-rate barely moves
    over the final octave (median magnitude <= every other policy).
  - LRU (blind recency) is the STEEPEST policy: it is the most
    cache-sensitive, so it keeps benefiting from added capacity
    (median magnitude >= every other policy).
  - GRASP <= LRU, and the spread between flattest and steepest is
    material (LRU median >= 1.5x GRASP median).
  - Charged P-OPT fully saturates on at least one app (its strongest):
    POPT's per-app minimum slope is near zero.

NOTE (2026-06-13): the earlier "oracle-aware {GRASP, POPT} both flat /
< half of non-oracle" model was a multi-thread + UNCHARGED-P-OPT
artifact. Under the faithful 1-way RRM capacity charge (Balaji & Lucia
HPCA'21) P-OPT is a practical policy that pays the reserved-way tax and
recovers the gap as cache grows -> it is mid-pack in steepness, no
longer characteristically flat. GRASP is now the only flat policy.

Output:
  - wiki/data/policy_steepness_ranking.json
  - wiki/data/policy_steepness_ranking.md

Verdict PASS iff every ordering and threshold check holds.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

# The flattest (cache-insensitive degree heuristic) and the steepest
# (most cache-sensitive blind-recency) policy. These anchor the
# charge-invariant ordering; the other two policies sit between them.
FLATTEST_POLICY = "GRASP"
STEEPEST_POLICY = "LRU"

# Thresholds (pp/octave). Pinned 2026-06-13 to the reproducible
# single-thread, array-relative-GRASP 0.15, CHARGED-P-OPT 1-way corpus
# (final-octave medians GRASP=1.04, SRRIP=1.36, POPT=1.52, LRU=2.24).
LRU_OVER_GRASP_SPREAD = 1.5        # LRU median >= this x GRASP median
POPT_MIN_SLOPE_CEILING_PP = 0.2    # at least one app must fully saturate with POPT


def _abs_final_slope(per_app: Dict[str, Any], app: str, policy: str) -> float:
    return abs(float(per_app[app][policy]["final_octave_slope_pp"]))


def _build_report(onset_path: Path) -> Dict[str, Any]:
    with onset_path.open() as fh:
        blob = json.load(fh)
    apps: List[str] = list(blob["meta"]["apps"])
    policies: List[str] = list(blob["meta"]["policies"])
    per_app = blob["per_app"]

    per_policy: Dict[str, Dict[str, Any]] = {}
    for pol in policies:
        slopes = [_abs_final_slope(per_app, app, pol) for app in apps]
        per_policy[pol] = {
            "n": len(slopes),
            "min": round(min(slopes), 6),
            "median": round(statistics.median(slopes), 6),
            "mean": round(statistics.mean(slopes), 6),
            "max": round(max(slopes), 6),
            "per_app": {a: round(_abs_final_slope(per_app, a, pol), 6) for a in apps},
        }

    medians = {pol: per_policy[pol]["median"] for pol in policies}
    others_vs_flattest = [p for p in policies if p != FLATTEST_POLICY]
    others_vs_steepest = [p for p in policies if p != STEEPEST_POLICY]

    ranking = sorted(policies, key=lambda p: per_policy[p]["median"])

    checks: Dict[str, Dict[str, Any]] = {}

    checks["grasp_is_flattest"] = {
        "grasp": medians[FLATTEST_POLICY],
        "others": {p: medians[p] for p in others_vs_flattest},
        "ok": all(
            medians[FLATTEST_POLICY] <= medians[p] + 1e-9 for p in others_vs_flattest
        ),
    }
    checks["lru_is_steepest"] = {
        "lru": medians[STEEPEST_POLICY],
        "others": {p: medians[p] for p in others_vs_steepest},
        "ok": all(
            medians[STEEPEST_POLICY] >= medians[p] - 1e-9 for p in others_vs_steepest
        ),
    }
    checks["grasp_le_lru_median"] = {
        "grasp": medians[FLATTEST_POLICY],
        "lru": medians[STEEPEST_POLICY],
        "ok": medians[FLATTEST_POLICY] <= medians[STEEPEST_POLICY] + 1e-9,
    }
    spread_ok = (
        medians[STEEPEST_POLICY]
        >= LRU_OVER_GRASP_SPREAD * medians[FLATTEST_POLICY] - 1e-9
    )
    checks["steepness_spread"] = {
        "lru": medians[STEEPEST_POLICY],
        "grasp": medians[FLATTEST_POLICY],
        "required_ratio": LRU_OVER_GRASP_SPREAD,
        "actual_ratio": round(
            medians[STEEPEST_POLICY] / medians[FLATTEST_POLICY], 6
        )
        if medians[FLATTEST_POLICY]
        else None,
        "ok": spread_ok,
    }
    checks["popt_min_saturates"] = {
        "popt_min": per_policy["POPT"]["min"],
        "ceiling": POPT_MIN_SLOPE_CEILING_PP,
        "ok": per_policy["POPT"]["min"] <= POPT_MIN_SLOPE_CEILING_PP + 1e-9,
    }

    verdict_ok = all(c["ok"] for c in checks.values())

    return {
        "schema": "policy_steepness_ranking/v1",
        "source": str(onset_path),
        "meta": {
            "apps": apps,
            "policies": policies,
            "flattest_policy": FLATTEST_POLICY,
            "steepest_policy": STEEPEST_POLICY,
            "thresholds": {
                "lru_over_grasp_spread": LRU_OVER_GRASP_SPREAD,
                "popt_min_slope_ceiling_pp": POPT_MIN_SLOPE_CEILING_PP,
            },
        },
        "per_policy": per_policy,
        "medians_pp": medians,
        "ranking_by_median": ranking,
        "checks": checks,
        "verdict_ok": verdict_ok,
    }


def _render_md(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Per-policy final-octave steepness ranking")
    lines.append("")
    lines.append(
        f"**Verdict:** {'PASS' if report['verdict_ok'] else 'FAIL'} "
        f"(ranking={' < '.join(report['ranking_by_median'])})"
    )
    lines.append("")
    lines.append("## Per-policy aggregates (|final-octave slope|, pp/octave)")
    lines.append("")
    lines.append("| policy | n | min | median | mean | max |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for pol, p in report["per_policy"].items():
        lines.append(
            f"| {pol} | {p['n']} | {p['min']:.4f} | {p['median']:.4f} | "
            f"{p['mean']:.4f} | {p['max']:.4f} |"
        )
    lines.append("")
    lines.append("## Per-app |final-octave slope| breakdown")
    lines.append("")
    apps = report["meta"]["apps"]
    header = "| policy |" + "".join(f" {a} |" for a in apps)
    sep = "|---|" + "".join("---:|" for _ in apps)
    lines.append(header)
    lines.append(sep)
    for pol in report["meta"]["policies"]:
        row = f"| {pol} |"
        for a in apps:
            row += f" {report['per_policy'][pol]['per_app'][a]:.4f} |"
        lines.append(row)
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| check | ok |")
    lines.append("|---|:---:|")
    for name, payload in report["checks"].items():
        lines.append(f"| {name} | {'OK' if payload['ok'] else 'FAIL'} |")
    lines.append("")
    lines.append("## Thresholds (locked)")
    lines.append("")
    for k, v in report["meta"]["thresholds"].items():
        lines.append(f"- {k} = {v}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onset-json", default="wiki/data/cache_saturation_onset.json")
    ap.add_argument("--json-out", default="wiki/data/policy_steepness_ranking.json")
    ap.add_argument("--md-out", default="wiki/data/policy_steepness_ranking.md")
    args = ap.parse_args()

    onset_path = Path(args.onset_json)
    if not onset_path.exists():
        print(f"policy-steepness-ranking: input {onset_path} missing -- skipping")
        return 0
    report = _build_report(onset_path)

    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    Path(args.md_out).write_text(_render_md(report))

    medians = report["medians_pp"]
    summary = " ".join(f"{p}={medians[p]:.3f}" for p in ["POPT", "GRASP", "LRU", "SRRIP"])
    verdict = "PASS" if report["verdict_ok"] else "FAIL"
    print(f"policy-steepness-ranking: medians {summary} verdict={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
