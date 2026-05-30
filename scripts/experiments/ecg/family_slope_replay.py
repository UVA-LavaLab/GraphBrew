"""Gate 67 — per-family replay of the capacity-sensitivity slope ordering.

Gate 66 computes a per-policy OLS slope of miss_rate (pp) versus
log2(L3_MB) across the whole corpus and ranks policies: LRU steepest
(most cache-hungry), GRASP shallowest (least cache-hungry; extracts
most at small caches). Gate 67 asks the same question one graph
family at a time: does each qualifying family independently replay
the global slope ordering?

Computation (mirrors gate 66):
    log2(MB) axis: 1MB->0, 4MB->2, 8MB->3 (non-uniform).
    slope_pp_per_octave = OLS slope of (log2(L3 MB), miss_rate_pp)
    over the three L3 points of each (app, graph, policy) cell.

For each family we collect, per policy, the list of slopes across all
(app, graph) cells in that family with full 1MB / 4MB / 8MB coverage
and at least all 4 policies present. We then summarise per policy
(median, mean, n_cells). A family REPLAYS the global pattern iff:
    (1) median(LRU)   <  median(GRASP)  (LRU strictly steeper)
    (2) median(SRRIP) <  median(GRASP)  (SRRIP also steeper than GRASP)
    (3) every policy median is strictly < HELP_FLOOR_PP_OCTAVE
        (cache helps every policy in this family).

Single-graph families with five (app) cells still produce a useful
median; small-n families with deviations should be PINNED rather than
allowed to silently fail.

Output schema:
  meta.qualifying_families        : families with full 1MB+4MB+8MB
  meta.help_floor_pp_octave       : -5.0 (see gate 66)
  meta.replay_count               : families that replay the pattern
  meta.deviating_families         : families that do NOT replay
  meta.pinned_deviating_families  : known/accepted deviations
  meta.verdict                    : PASS iff no NEW deviation beyond pin
                                    AND replay_count >= 1
  per_family.<F>.per_policy.<P>   : median/mean/n_cells/min/max
  per_family.<F>.replays_pattern  : bool
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ORACLE_JSON = REPO_ROOT / "wiki" / "data" / "oracle_gap.json"
DEFAULT_JSON_OUT = REPO_ROOT / "wiki" / "data" / "family_slope_replay.json"
DEFAULT_MD_OUT = REPO_ROOT / "wiki" / "data" / "family_slope_replay.md"

L3_SIZES = ("1MB", "4MB", "8MB")
L3_LOG2_MB = {"1MB": 0.0, "4MB": 2.0, "8MB": 3.0}
POLICIES = ("GRASP", "LRU", "POPT", "SRRIP")
ORACLE_AWARE = ("GRASP", "POPT")
NON_ORACLE = ("LRU", "SRRIP")

HELP_FLOOR_PP_OCTAVE = -5.0

# The social family is a known deviation: it contains email-Eu-core
# (WSS ~4.5 kB) whose 5 cells are fully saturated at every L3 point,
# so their slopes are near zero. Mixing those 5 saturated cells with
# the other 13 social cells washes out the per-family GRASP-vs-LRU
# ordering. Pinned explicitly so the verdict tracks NEW deviations
# only.
PINNED_DEVIATING_FAMILIES: tuple[str, ...] = ("social",)


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


def build(payload: dict) -> dict:
    rows = payload["rows"]
    # family -> graph -> app -> policy -> l3 -> miss_rate(pp)
    by = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(dict))
        )
    )
    for r in rows:
        if r["l3_size"] not in L3_LOG2_MB:
            continue
        by[r["family"]][r["graph"]][r["app"]][r["policy"]][r["l3_size"]] = (
            float(r["miss_rate"]) * 100.0
        )

    per_family: dict[str, dict] = {}
    qualifying: list[str] = []
    for fam, by_graph in sorted(by.items()):
        # collect per-policy slopes across every (graph, app) cell that
        # has the three L3 points for every policy.
        per_pol_slopes: dict[str, list[float]] = defaultdict(list)
        any_full = False
        for graph, by_app in by_graph.items():
            for app, by_pol in by_app.items():
                if not all(p in by_pol for p in POLICIES):
                    continue
                if not all(
                    all(l in by_pol[p] for l in L3_SIZES) for p in POLICIES
                ):
                    continue
                any_full = True
                for pol in POLICIES:
                    pts = [
                        (L3_LOG2_MB[l], by_pol[pol][l]) for l in L3_SIZES
                    ]
                    slope = _ols_slope(pts)
                    if slope is None:
                        continue
                    per_pol_slopes[pol].append(slope)
        if not any_full:
            continue
        qualifying.append(fam)

        per_policy: dict[str, dict] = {}
        for pol in POLICIES:
            xs = per_pol_slopes.get(pol, [])
            if not xs:
                continue
            per_policy[pol] = {
                "n_cells":   len(xs),
                "median_pp": round(_median(xs), 4),
                "mean_pp":   round(sum(xs) / len(xs), 4),
                "min_pp":    round(min(xs), 4),
                "max_pp":    round(max(xs), 4),
                "is_oracle_aware": pol in ORACLE_AWARE,
            }
        medians = {p: per_policy[p]["median_pp"] for p in per_policy}
        inv_lru_steeper_grasp = (
            "LRU" in medians
            and "GRASP" in medians
            and medians["LRU"] < medians["GRASP"]
        )
        inv_srrip_steeper_grasp = (
            "SRRIP" in medians
            and "GRASP" in medians
            and medians["SRRIP"] < medians["GRASP"]
        )
        inv_all_help = all(
            s["median_pp"] < HELP_FLOOR_PP_OCTAVE for s in per_policy.values()
        )
        replays = (
            inv_lru_steeper_grasp
            and inv_srrip_steeper_grasp
            and inv_all_help
        )
        per_family[fam] = {
            "per_policy":              per_policy,
            "replays_pattern":         replays,
            "lru_steeper_than_grasp":  inv_lru_steeper_grasp,
            "srrip_steeper_than_grasp": inv_srrip_steeper_grasp,
            "all_policies_helped":     inv_all_help,
        }

    deviating = [f for f in qualifying if not per_family[f]["replays_pattern"]]
    new_dev = [f for f in deviating if f not in PINNED_DEVIATING_FAMILIES]
    replay_count = sum(1 for f in qualifying if per_family[f]["replays_pattern"])
    verdict = "PASS" if (replay_count >= 1 and not new_dev) else "FAIL"

    return {
        "meta": {
            "qualifying_families":       qualifying,
            "help_floor_pp_octave":      HELP_FLOOR_PP_OCTAVE,
            "replay_count":              replay_count,
            "deviating_families":        deviating,
            "pinned_deviating_families": list(PINNED_DEVIATING_FAMILIES),
            "new_deviating_families":    new_dev,
            "policies":                  list(POLICIES),
            "oracle_aware_policies":     list(ORACLE_AWARE),
            "non_oracle_policies":       list(NON_ORACLE),
            "l3_axis":                   list(L3_LOG2_MB.keys()),
            "verdict":                   verdict,
            "verdict_invariant": (
                "PASS iff at least one family replays the pattern (LRU and "
                "SRRIP both strictly steeper than GRASP, every policy "
                f"median < {HELP_FLOOR_PP_OCTAVE} pp/octave) AND no NEW "
                "deviating family appears beyond the pinned set."
            ),
        },
        "per_family": per_family,
    }


def render_md(result: dict, src_label: str) -> str:
    m = result["meta"]
    out = [
        "# Gate 67 — Per-family capacity-sensitivity slope replay",
        "",
        f"source: `{src_label}`",
        "",
        f"verdict: **{m['verdict']}**",
        "",
        f"  invariant: {m['verdict_invariant']}",
        "",
        "qualifying families (full 1MB/4MB/8MB coverage, "
        "all 4 policies, at least one app): "
        + ", ".join(m["qualifying_families"]),
        "",
        f"replay count: **{m['replay_count']} / {len(m['qualifying_families'])}**",
        f"deviating families (pinned): {m['pinned_deviating_families']}",
        f"deviating families (new):    {m['new_deviating_families']}",
        "",
        "## Per-family median slope by policy (pp / log2(L3 MB))",
        "",
        "Smaller (more negative) = more cache-hungry. "
        f"Help floor: {m['help_floor_pp_octave']} pp/octave.",
        "",
        "| family | GRASP | POPT | LRU | SRRIP | n cells | replays? |",
        "| --- | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for fam in m["qualifying_families"]:
        info = result["per_family"][fam]
        pp = info["per_policy"]

        def _cell(pol: str) -> str:
            if pol not in pp:
                return "—"
            return f"{pp[pol]['median_pp']:.3f}"

        n = max((pp[p]["n_cells"] for p in pp), default=0)
        out.append(
            f"| {fam} | {_cell('GRASP')} | {_cell('POPT')} "
            f"| {_cell('LRU')} | {_cell('SRRIP')} | {n} "
            f"| {'yes' if info['replays_pattern'] else 'no'} |"
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

    payload = json.loads(src_path.read_text())
    result = build(payload)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(render_md(result, src_label))
    m = result["meta"]
    print(
        f"family-slope-replay: families={m['qualifying_families']} | "
        f"replays={m['replay_count']}/{len(m['qualifying_families'])} | "
        f"dev_new={len(m['new_deviating_families'])} "
        f"dev_pinned={len(m['pinned_deviating_families'])} | "
        f"verdict={m['verdict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
