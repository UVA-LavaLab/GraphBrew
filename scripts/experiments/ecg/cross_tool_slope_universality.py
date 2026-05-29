"""Gate 76 — cross-tool slope-sign universality.

A roll-up invariant ensuring that EVERY (tool, policy) median capacity-
sensitivity slope is negative: extra cache must, on average, REDUCE
miss rate for every policy on every tool. A positive median on any
(tool, policy) cell would indicate that policy gets WORSE with more
cache, which would be either a measurement bug or a fundamentally
broken policy.

Three individual gates already check this within their own tool:

  - gate 66 (capacity_sensitivity): cache-sim, all 4 policies.
  - gate 70 (gem5_slope_replay): gem5, 3 policies (no POPT).
  - gate 71 (sniper_slope_replay): sniper, 3 policies (no POPT).

This gate centralizes those checks into a single artifact and adds
two cross-tool roll-up invariants that individual gates cannot:

  (1) Every (tool, policy) pair has a negative median slope (sign-
      universality).
  (2) No tool exhibits a steepness span (max - min across policies)
      larger than STEEPNESS_SPAN_CEILING_PP_OCT — catches a regression
      where one policy collapses to near-zero slope while others
      stay steep.
  (3) Every (tool, policy) slope lies within the documented physical
      band [MIN_SLOPE_PP_OCT, MAX_SLOPE_PP_OCT] — catches both
      runaway-steep regressions and near-zero collapse.

Output schema:
  meta.tools                          : ["cache-sim", "gem5", "sniper"]
  meta.tool_policies[tool]            : list of policies observed
  meta.medians[tool][policy]          : float pp/oct
  meta.steepness_spans[tool]          : float pp/oct
  meta.in_band_count                  : int
  meta.expected_in_band_count         : int
  meta.violations                     : list of dicts
  meta.verdict                        : PASS / FAIL
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_SIM_JSON = REPO_ROOT / "wiki" / "data" / "capacity_sensitivity.json"
DEFAULT_GEM5_JSON      = REPO_ROOT / "wiki" / "data" / "gem5_slope_replay.json"
DEFAULT_SNIPER_JSON    = REPO_ROOT / "wiki" / "data" / "sniper_slope_replay.json"
DEFAULT_JSON_OUT       = REPO_ROOT / "wiki" / "data" / "cross_tool_slope_universality.json"
DEFAULT_MD_OUT         = REPO_ROOT / "wiki" / "data" / "cross_tool_slope_universality.md"

# Physical band: a per-policy median outside this range is suspect.
# Lower bound -25 pp/oct: anchor sweeps at sub-WSS scales rarely exceed
# -20 (gem5 max steepness ~-7); cache-sim post-WSS ~-19. -25 is well
# below any observed median.
# Upper bound -0.5 pp/oct: a near-zero median means the policy stopped
# benefiting from cache entirely, which is itself a sign of either
# saturation collapse or a measurement bug.
MIN_SLOPE_PP_OCT = -25.0
MAX_SLOPE_PP_OCT = -0.5

# A tool's steepness span (max steepness - min steepness across its
# policies) larger than this ceiling means one policy is collapsing
# while others stay steep — catches partial regressions.
STEEPNESS_SPAN_CEILING_PP_OCT = 5.0


def _cache_sim_medians(path: Path) -> dict[str, float]:
    doc = json.loads(path.read_text())
    return {
        pol: float(block["median_pp"])
        for pol, block in doc["meta"]["policy_summary"].items()
    }


def _anchor_medians(path: Path) -> dict[str, float]:
    doc = json.loads(path.read_text())
    return {
        pol: float(block["median"])
        for pol, block in doc["meta"]["per_policy"].items()
    }


def build(cache_sim_path: Path, gem5_path: Path, sniper_path: Path) -> dict:
    medians = {
        "cache-sim": _cache_sim_medians(cache_sim_path),
        "gem5":      _anchor_medians(gem5_path),
        "sniper":    _anchor_medians(sniper_path),
    }

    tool_policies = {t: sorted(med.keys()) for t, med in medians.items()}
    steepness_spans: dict[str, float] = {}
    for t, med in medians.items():
        vals = list(med.values())
        if vals:
            steepness_spans[t] = round(max(vals) - min(vals), 4)
        else:
            steepness_spans[t] = 0.0

    violations: list[dict] = []
    in_band_count = 0
    expected_in_band_count = sum(len(p) for p in tool_policies.values())

    for t, med in medians.items():
        for pol, val in med.items():
            if val >= 0.0:
                violations.append({
                    "type":   "positive_slope",
                    "tool":   t,
                    "policy": pol,
                    "value":  round(val, 4),
                })
            if val < MIN_SLOPE_PP_OCT or val > MAX_SLOPE_PP_OCT:
                violations.append({
                    "type":   "out_of_band",
                    "tool":   t,
                    "policy": pol,
                    "value":  round(val, 4),
                    "band":   [MIN_SLOPE_PP_OCT, MAX_SLOPE_PP_OCT],
                })
            else:
                in_band_count += 1

    for t, span in steepness_spans.items():
        if span > STEEPNESS_SPAN_CEILING_PP_OCT:
            violations.append({
                "type":    "steepness_span_exceeded",
                "tool":    t,
                "span":    span,
                "ceiling": STEEPNESS_SPAN_CEILING_PP_OCT,
            })

    inv_all_negative = all(
        v < 0.0
        for med in medians.values() for v in med.values()
    )
    inv_all_in_band = (in_band_count == expected_in_band_count)
    inv_no_span_exceeded = all(
        s <= STEEPNESS_SPAN_CEILING_PP_OCT
        for s in steepness_spans.values()
    )

    verdict_checks = {
        "all_tool_policy_medians_negative":      inv_all_negative,
        "all_tool_policy_medians_in_band":       inv_all_in_band,
        "no_tool_exceeds_steepness_span_ceiling": inv_no_span_exceeded,
    }
    verdict = "PASS" if all(verdict_checks.values()) else "FAIL"

    return {
        "meta": {
            "tools":                          list(medians.keys()),
            "tool_policies":                  tool_policies,
            "medians":                        {t: {p: round(v, 4) for p, v in m.items()} for t, m in medians.items()},
            "steepness_spans":                steepness_spans,
            "in_band_count":                  in_band_count,
            "expected_in_band_count":         expected_in_band_count,
            "violations":                     violations,
            "min_slope_pp_oct":               MIN_SLOPE_PP_OCT,
            "max_slope_pp_oct":               MAX_SLOPE_PP_OCT,
            "steepness_span_ceiling_pp_oct":  STEEPNESS_SPAN_CEILING_PP_OCT,
            "verdict_checks":                 verdict_checks,
            "verdict":                        verdict,
        },
    }


def render_md(payload: dict) -> str:
    m = payload["meta"]
    lines = [
        "# Cross-tool slope-sign universality",
        "",
        f"**Verdict:** {m['verdict']}  ",
        f"**Slope band:** [{m['min_slope_pp_oct']}, {m['max_slope_pp_oct']}] pp/oct  ",
        f"**Steepness span ceiling:** {m['steepness_span_ceiling_pp_oct']} pp/oct  ",
        f"**(tool, policy) medians in band:** "
        f"{m['in_band_count']}/{m['expected_in_band_count']}  ",
        "",
        "## Median slope per (tool, policy)",
        "",
        "| tool | policy | median pp/oct |",
        "|---|---|---:|",
    ]
    for tool in m["tools"]:
        for pol in m["tool_policies"][tool]:
            v = m["medians"][tool][pol]
            lines.append(f"| {tool} | {pol} | {v:+.4f} |")

    lines += [
        "",
        "## Per-tool steepness span (max - min across policies)",
        "",
        "| tool | span pp/oct |",
        "|---|---:|",
    ]
    for tool, span in m["steepness_spans"].items():
        lines.append(f"| {tool} | {span:.4f} |")

    lines += [
        "",
        "## Verdict checks",
        "",
        "| check | result |",
        "|---|---|",
    ]
    for k, v in m["verdict_checks"].items():
        lines.append(f"| {k} | {'✅' if v else '❌'} |")

    if m["violations"]:
        lines += [
            "",
            "## Violations",
            "",
            "```json",
            json.dumps(m["violations"], indent=2),
            "```",
        ]
    else:
        lines += [
            "",
            "## Violations",
            "",
            "_None._",
        ]

    lines += [
        "",
        "## Interpretation",
        "",
        "This is a roll-up invariant ensuring no (tool, policy) slope "
        "median ever turns positive (extra cache must not hurt on "
        "average) or collapses to near-zero (a policy that stops "
        "responding to cache scaling is suspect). The cross-tool span "
        "check catches partial regressions where one policy on one "
        "tool stops scaling while siblings stay healthy.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-sim-json", type=Path, default=DEFAULT_CACHE_SIM_JSON)
    ap.add_argument("--gem5-json",      type=Path, default=DEFAULT_GEM5_JSON)
    ap.add_argument("--sniper-json",    type=Path, default=DEFAULT_SNIPER_JSON)
    ap.add_argument("--json-out",       type=Path, default=DEFAULT_JSON_OUT)
    ap.add_argument("--md-out",         type=Path, default=DEFAULT_MD_OUT)
    args = ap.parse_args()

    payload = build(args.cache_sim_json, args.gem5_json, args.sniper_json)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    args.md_out.write_text(render_md(payload))

    m = payload["meta"]
    print(
        f"cross-tool-slope-universality: "
        f"in_band={m['in_band_count']}/{m['expected_in_band_count']} "
        f"violations={len(m['violations'])} "
        f"verdict={m['verdict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
