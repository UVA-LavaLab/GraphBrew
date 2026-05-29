#!/usr/bin/env python3
"""Per-policy final-octave steepness ranking (gate 81).

Reads `wiki/data/cache_saturation_onset.json` and aggregates the
**final-octave** (4MB->8MB) slope magnitudes per policy across all
apps. Asserts the headline policy ordering that mirrors the
saturation-rank story from gate 64 but in absolute steepness terms:

  - POPT and GRASP (oracle-aware) hold flat at small magnitudes.
  - LRU and SRRIP (non-oracle) stay steep at large magnitudes.
  - Strict ordering of medians: POPT <= GRASP <= LRU, and POPT < SRRIP.
  - Oracle-aware median steepness must be < half of non-oracle median.

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

ORACLE_AWARE = ("POPT", "GRASP")
NON_ORACLE = ("LRU", "SRRIP")

# Thresholds (pp/octave). Chosen tight against current observation
# (POPT=0.10, GRASP=0.23, LRU=1.06, SRRIP=1.09) so a real regression
# in the saturation story is caught.
ORACLE_AWARE_CEILING_PP = 0.5      # median |final-octave slope|
NON_ORACLE_FLOOR_PP = 0.5          # median |final-octave slope|
ORACLE_AWARE_HALF_OF_NON_ORACLE = 0.5  # oracle median must be < this fraction
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
    oracle_med = statistics.median([medians[p] for p in ORACLE_AWARE])
    non_oracle_med = statistics.median([medians[p] for p in NON_ORACLE])

    ranking = sorted(policies, key=lambda p: per_policy[p]["median"])

    checks: Dict[str, Dict[str, Any]] = {}

    checks["popt_le_grasp_median"] = {
        "popt": medians["POPT"],
        "grasp": medians["GRASP"],
        "ok": medians["POPT"] <= medians["GRASP"] + 1e-9,
    }
    checks["grasp_le_lru_median"] = {
        "grasp": medians["GRASP"],
        "lru": medians["LRU"],
        "ok": medians["GRASP"] <= medians["LRU"] + 1e-9,
    }
    checks["popt_lt_srrip_median"] = {
        "popt": medians["POPT"],
        "srrip": medians["SRRIP"],
        "ok": medians["POPT"] < medians["SRRIP"] - 1e-9,
    }
    checks["oracle_aware_ceiling"] = {
        "ceiling": ORACLE_AWARE_CEILING_PP,
        "popt": medians["POPT"],
        "grasp": medians["GRASP"],
        "ok": all(medians[p] <= ORACLE_AWARE_CEILING_PP + 1e-9 for p in ORACLE_AWARE),
    }
    checks["non_oracle_floor"] = {
        "floor": NON_ORACLE_FLOOR_PP,
        "lru": medians["LRU"],
        "srrip": medians["SRRIP"],
        "ok": all(medians[p] >= NON_ORACLE_FLOOR_PP - 1e-9 for p in NON_ORACLE),
    }
    half_check = oracle_med < non_oracle_med * ORACLE_AWARE_HALF_OF_NON_ORACLE + 1e-9
    checks["oracle_half_of_non_oracle"] = {
        "oracle_median": round(oracle_med, 6),
        "non_oracle_median": round(non_oracle_med, 6),
        "required_ratio": ORACLE_AWARE_HALF_OF_NON_ORACLE,
        "actual_ratio": round(oracle_med / non_oracle_med, 6) if non_oracle_med else None,
        "ok": half_check,
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
            "oracle_aware": list(ORACLE_AWARE),
            "non_oracle": list(NON_ORACLE),
            "thresholds": {
                "oracle_aware_ceiling_pp": ORACLE_AWARE_CEILING_PP,
                "non_oracle_floor_pp": NON_ORACLE_FLOOR_PP,
                "oracle_aware_half_of_non_oracle": ORACLE_AWARE_HALF_OF_NON_ORACLE,
                "popt_min_slope_ceiling_pp": POPT_MIN_SLOPE_CEILING_PP,
            },
        },
        "per_policy": per_policy,
        "medians_pp": medians,
        "oracle_median_pp": round(oracle_med, 6),
        "non_oracle_median_pp": round(non_oracle_med, 6),
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
