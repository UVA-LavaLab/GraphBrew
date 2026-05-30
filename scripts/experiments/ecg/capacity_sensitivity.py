"""Gate 66 — per-policy capacity-sensitivity slope.

For each (app, graph, policy) cell with at least two L3 measurements
on the {1MB, 4MB, 8MB} axis, compute the least-squares slope of
miss_rate (in pp) versus log2(L3_MB):

    slope_pp_per_octave = OLS slope of (log2(L3 MB), miss_rate_pp)

Per policy, we aggregate across all cells and report the distribution
of slopes (median, mean, p10, p90). A very-negative slope means the
policy benefits a lot from doubling cache; a near-zero slope means the
policy is already saturated.

This is informative for two reasons:

  1. Sanity: every policy MUST have median slope strictly < 0 because
     larger L3 cannot make best-policy miss-rate worse (cache
     monotonicity, also enforced cell-by-cell in gate 65).

  2. Policy character: in this corpus oracle-aware policies (GRASP,
     POPT) tend to have SHALLOWER slopes than non-oracle policies
     (LRU, SRRIP). They extract more value at small caches, so they
     have less marginal value to gain as cache grows. This is the
     dual of gate 62's central finding: oracle-aware payoff shrinks
     when capacity loosens.

Verdict (structural sanity, not a paper claim):
  PASS iff every policy has median slope < -5 pp/octave (cache helps)
       AND LRU has the steepest (most negative) median slope
       AND GRASP has shallower median slope than LRU (oracle-aware
           extracts more value at small caches).

Output schema:
  meta.policies                : list of 4 policies
  meta.cell_count              : number of (app, graph, policy) cells
                                 scored (each policy has the same total)
  meta.policy_summary          : policy -> {median, mean, p10, p90,
                                            min, max, n_cells}
  meta.steepest_policy         : policy with the most-negative median
  meta.shallowest_policy       : policy with the least-negative median
  meta.median_steepness_gap_pp : |steepest_median| - |shallowest_median|
  meta.verdict                 : PASS iff structural invariants hold
  per_cell                     : list of {app, graph, policy, slope_pp}
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ORACLE_JSON = REPO_ROOT / "wiki" / "data" / "oracle_gap.json"
DEFAULT_JSON_OUT = REPO_ROOT / "wiki" / "data" / "capacity_sensitivity.json"
DEFAULT_MD_OUT = REPO_ROOT / "wiki" / "data" / "capacity_sensitivity.md"

POLICIES = ("GRASP", "LRU", "POPT", "SRRIP")
L3_LOG2_MB = {"1MB": 0.0, "4MB": 2.0, "8MB": 3.0}

# Sanity floor: every policy must benefit by at least this many
# pp/octave on the median cell (otherwise the corpus has saturated
# globally).
HELP_FLOOR_PP_OCTAVE = -5.0


def _ols_slope(pts: list[tuple[float, float]]) -> float | None:
    n = len(pts)
    if n < 2:
        return None
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    den = n * sxx - sx * sx
    if den == 0:
        return None
    return (n * sxy - sx * sy) / den


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _pct(xs: list[float], p: float) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    k = max(0, min(n - 1, int(round(p * (n - 1)))))
    return s[k]


def build(payload: dict) -> dict:
    rows = payload["rows"]
    cells: dict = defaultdict(list)
    for r in rows:
        if r["l3_size"] not in L3_LOG2_MB:
            continue
        cells[(r["app"], r["graph"], r["policy"])].append(
            (L3_LOG2_MB[r["l3_size"]], float(r["miss_rate"]) * 100.0)
        )

    per_cell = []
    slopes_by_policy: dict = defaultdict(list)
    for (app, graph, pol), pts in sorted(cells.items()):
        slope = _ols_slope(pts)
        if slope is None:
            continue
        slope = round(slope, 4)
        per_cell.append({
            "app":      app,
            "graph":    graph,
            "policy":   pol,
            "slope_pp": slope,
            "n_points": len(pts),
        })
        slopes_by_policy[pol].append(slope)

    policy_summary: dict[str, dict] = {}
    medians: dict[str, float] = {}
    for pol in POLICIES:
        xs = slopes_by_policy.get(pol, [])
        if not xs:
            continue
        med = round(_median(xs), 4)
        policy_summary[pol] = {
            "n_cells":   len(xs),
            "median_pp": med,
            "mean_pp":   round(sum(xs) / len(xs), 4),
            "p10_pp":    round(_pct(xs, 0.10), 4),
            "p90_pp":    round(_pct(xs, 0.90), 4),
            "min_pp":    round(min(xs), 4),
            "max_pp":    round(max(xs), 4),
        }
        medians[pol] = med

    steepest = min(medians, key=lambda p: medians[p]) if medians else None
    shallowest = max(medians, key=lambda p: medians[p]) if medians else None
    gap = (
        round(abs(medians[steepest]) - abs(medians[shallowest]), 4)
        if steepest and shallowest
        else 0.0
    )

    inv_all_help = all(
        s["median_pp"] < HELP_FLOOR_PP_OCTAVE for s in policy_summary.values()
    )
    inv_lru_steepest = (steepest == "LRU")
    inv_grasp_shallower_than_lru = (
        "GRASP" in medians
        and "LRU" in medians
        and medians["GRASP"] > medians["LRU"]
    )
    verdict = "PASS" if (
        inv_all_help and inv_lru_steepest and inv_grasp_shallower_than_lru
    ) else "FAIL"

    return {
        "meta": {
            "policies":                       list(POLICIES),
            "cell_count":                     sum(
                s["n_cells"] for s in policy_summary.values()
            ),
            "l3_axis":                        list(L3_LOG2_MB.keys()),
            "help_floor_pp_octave":           HELP_FLOOR_PP_OCTAVE,
            "policy_summary":                 policy_summary,
            "steepest_policy":                steepest,
            "shallowest_policy":              shallowest,
            "median_steepness_gap_pp":        gap,
            "invariant_all_help":             inv_all_help,
            "invariant_lru_steepest":         inv_lru_steepest,
            "invariant_grasp_shallower_lru": inv_grasp_shallower_than_lru,
            "verdict":                        verdict,
            "verdict_invariant": (
                "PASS iff (1) every policy's median slope is strictly less "
                f"than {HELP_FLOOR_PP_OCTAVE} pp/octave, (2) LRU has the "
                "steepest (most-negative) median slope, and (3) GRASP has "
                "a strictly shallower (less-negative) median slope than LRU."
            ),
        },
        "per_cell": per_cell,
    }


def render_md(result: dict, src_label: str) -> str:
    m = result["meta"]
    out = [
        "# Gate 66 — Per-policy capacity-sensitivity slope",
        "",
        f"source: `{src_label}`",
        "",
        f"verdict: **{m['verdict']}**",
        "",
        f"  invariant: {m['verdict_invariant']}",
        "",
        f"cells scored: {m['cell_count']}; "
        f"L3 axis: {', '.join(m['l3_axis'])}",
        "",
        f"steepest median (most cache-hungry policy): "
        f"**{m['steepest_policy']}**",
        "",
        f"shallowest median (least cache-hungry policy): "
        f"**{m['shallowest_policy']}**",
        "",
        f"median steepness gap: {m['median_steepness_gap_pp']:.3f} pp/octave",
        "",
        "## Per-policy slope distribution (pp of miss-rate per log2(L3 MB))",
        "",
        "| policy | n cells | median | mean | p10 | p90 | min | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pol in POLICIES:
        if pol not in m["policy_summary"]:
            continue
        s = m["policy_summary"][pol]
        out.append(
            f"| {pol} | {s['n_cells']} | {s['median_pp']:.3f} "
            f"| {s['mean_pp']:.3f} | {s['p10_pp']:.3f} | {s['p90_pp']:.3f} "
            f"| {s['min_pp']:.3f} | {s['max_pp']:.3f} |"
        )
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oracle-json", type=Path, default=DEFAULT_ORACLE_JSON)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    ap.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = ap.parse_args()

    src_path = args.oracle_json
    try:
        src_label = str(src_path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        src_label = str(src_path)

    payload = json.loads(args.oracle_json.read_text())
    result = build(payload)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(render_md(result, src_label))
    m = result["meta"]
    print(
        f"capacity-sensitivity: cells={m['cell_count']} "
        f"steepest={m['steepest_policy']} "
        f"shallowest={m['shallowest_policy']} "
        f"gap={m['median_steepness_gap_pp']} "
        f"verdict={m['verdict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
