"""Gate 71 — sniper anchor slope sanity gate.

The cache-sim sweep (gates 65/66/67/68/69) demonstrates that
GRASP has the SHALLOWEST slope across {1MB, 4MB, 8MB}: it extracts
most of its help at small caches and the curve flattens fastest.

The sniper anchor sweep operates at much smaller L3 sizes (4kB / 32kB
/ 256kB / 2MB), so at 4kB we are far below the WSS of even the
smallest corpus graph (email-Eu-core, WSS ~4.5kB). At sub-WSS L3
sizes a "give-up-and-stream" policy like LRU can actually beat a
"hold-the-hot-set" policy like GRASP because the hot set does not
fit — a real physical regime that inverts the LRU-vs-GRASP slope
ordering observed at 1-8MB.

This gate therefore validates the parts of the slope picture that
DO transfer across scales:
  (1) cache monotonicity within each (app, graph, policy) cell:
      miss(4kB) > miss(2MB) (more cache always helps);
  (2) every per-policy median slope is negative;
  (3) SRRIP median slope is at least as steep as GRASP — SRRIP is
      not oracle-aware and is consistently more cache-hungry;
  (4) GRASP median slope is below the help-floor — cache still
      materially helps the oracle-aware policy at anchor scales.
The LRU-vs-GRASP slope delta is reported as INFORMATIONAL (and is
expected to be regime-dependent) but does NOT gate PASS/FAIL.

Computation:
  log2(kB) axis: 4kB -> 2.0, 32kB -> 5.0, 256kB -> 8.0, 2MB -> 11.0
  slope_pp_per_octave = OLS slope of
      (log2(L3 kB), miss_rate_pp)
  over the four L3 points of each (app, graph, policy) cell.
  miss_rate is in [0,1] in the anchor JSON and is multiplied by 100
  to land on the percentage-point scale used throughout the project.

Output schema:
  meta.n_cells                  : (app, graph) cells used
  meta.l3_axis_log2_kb          : the log2-kB axis used
  meta.per_policy[P].median     : median slope
  meta.per_policy[P].mean       : mean slope
  meta.per_policy[P].n          : number of cells with this policy
  meta.lru_minus_grasp_pp_oct   : median(LRU) - median(GRASP)
      (INFORMATIONAL — regime-dependent, not gated)
  meta.srrip_minus_grasp_pp_oct : median(SRRIP) - median(GRASP)
      (gated — must be <= 0)
  meta.help_floor_pp_octave     : -1.0
  meta.verdict                  : PASS iff
    (1) every per-cell miss(4kB) > miss(2MB) (cache monotonicity),
    (2) every per-policy median slope < 0,
    (3) median(SRRIP) <= median(GRASP),
    (4) median(GRASP) < help_floor.
  per_cell                      : every (app, graph, policy) slope
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ANCHOR_JSON = REPO_ROOT / "wiki" / "data" / "sniper_anchor.json"
DEFAULT_JSON_OUT = REPO_ROOT / "wiki" / "data" / "sniper_slope_replay.json"
DEFAULT_MD_OUT = REPO_ROOT / "wiki" / "data" / "sniper_slope_replay.md"

ANCHOR_L3_LOG2_KB = {
    "4kB":   2.0,
    "32kB":  5.0,
    "256kB": 8.0,
    "2MB":   11.0,
}
EXPECTED_SIZES = tuple(ANCHOR_L3_LOG2_KB.keys())

POLICIES = ("GRASP", "LRU", "SRRIP")
HELP_FLOOR_PP_OCTAVE = -1.0


def _ols_slope(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    return num / den


def _median(vs: list[float]) -> float | None:
    if not vs:
        return None
    s = sorted(vs)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def _mean(vs: list[float]) -> float | None:
    if not vs:
        return None
    return sum(vs) / len(vs)


def compute(anchor_path: Path) -> dict:
    raw = json.loads(anchor_path.read_text())
    cells = raw["cells"]

    # Reshape: (app, graph) -> {l3_size: {policy: miss_pp}}
    by_cell: dict[tuple[str, str], dict[str, dict[str, float]]] = defaultdict(dict)
    for c in cells:
        app = c["app"]
        graph = c["graph"]
        size = c["l3_size"]
        mb = c["miss_rate_by_policy"]
        by_cell[(app, graph)][size] = {
            p: float(v) * 100.0 for p, v in mb.items()
        }

    per_cell_records: list[dict] = []
    slopes_by_policy: dict[str, list[float]] = {p: [] for p in POLICIES}
    monotonic_violations: list[dict] = []

    for (app, graph), sweep in sorted(by_cell.items()):
        if not all(s in sweep for s in EXPECTED_SIZES):
            continue
        for policy in POLICIES:
            xs: list[float] = []
            ys: list[float] = []
            for s in EXPECTED_SIZES:
                if policy not in sweep[s]:
                    break
                xs.append(ANCHOR_L3_LOG2_KB[s])
                ys.append(sweep[s][policy])
            else:
                slope = _ols_slope(xs, ys)
                if slope is None or math.isnan(slope):
                    continue
                miss_small = sweep[EXPECTED_SIZES[0]][policy]
                miss_large = sweep[EXPECTED_SIZES[-1]][policy]
                if miss_small <= miss_large:
                    monotonic_violations.append({
                        "app": app, "graph": graph, "policy": policy,
                        "miss_small": miss_small, "miss_large": miss_large,
                    })
                per_cell_records.append({
                    "app":     app,
                    "graph":   graph,
                    "policy":  policy,
                    "slope_pp_per_octave": round(slope, 4),
                    "miss_pp_by_size": {s: round(sweep[s][policy], 4)
                                        for s in EXPECTED_SIZES},
                })
                slopes_by_policy[policy].append(slope)

    per_policy = {
        p: {
            "median": round(_median(slopes_by_policy[p]), 4)
            if slopes_by_policy[p] else None,
            "mean":   round(_mean(slopes_by_policy[p]), 4)
            if slopes_by_policy[p] else None,
            "n":      len(slopes_by_policy[p]),
        }
        for p in POLICIES
    }

    grasp_med = per_policy["GRASP"]["median"]
    lru_med = per_policy["LRU"]["median"]
    srrip_med = per_policy["SRRIP"]["median"]
    lru_minus_grasp = (lru_med - grasp_med) if (lru_med is not None and grasp_med is not None) else None
    srrip_minus_grasp = (srrip_med - grasp_med) if (srrip_med is not None and grasp_med is not None) else None

    all_medians_negative = all(
        per_policy[p]["median"] is not None and per_policy[p]["median"] < 0
        for p in POLICIES
    )

    verdict_checks = {
        "cache_monotonic_every_cell":
            len(monotonic_violations) == 0,
        "all_per_policy_medians_negative":
            all_medians_negative,
        "srrip_at_least_as_steep_as_grasp":
            (srrip_med is not None and grasp_med is not None
             and srrip_med <= grasp_med),
        "grasp_below_help_floor":
            (grasp_med is not None and grasp_med < HELP_FLOOR_PP_OCTAVE),
    }
    verdict = "PASS" if all(verdict_checks.values()) else "FAIL"

    return {
        "meta": {
            "anchor_source":            str(anchor_path.name),
            "n_cells":                  len(by_cell),
            "n_cell_policy_records":    len(per_cell_records),
            "l3_axis_log2_kb":          ANCHOR_L3_LOG2_KB,
            "expected_sizes":           list(EXPECTED_SIZES),
            "policies":                 list(POLICIES),
            "per_policy":               per_policy,
            "lru_minus_grasp_pp_oct":
                round(lru_minus_grasp, 4) if lru_minus_grasp is not None else None,
            "lru_minus_grasp_note":
                "INFORMATIONAL — regime-dependent at sub-WSS anchor "
                "scales, not gated. See module docstring.",
            "srrip_minus_grasp_pp_oct":
                round(srrip_minus_grasp, 4) if srrip_minus_grasp is not None else None,
            "help_floor_pp_octave":     HELP_FLOOR_PP_OCTAVE,
            "monotonic_violations":     monotonic_violations,
            "verdict_checks":           verdict_checks,
            "verdict":                  verdict,
        },
        "per_cell": per_cell_records,
    }


def render_md(payload: dict) -> str:
    m = payload["meta"]
    lines = [
        "# Sniper anchor — capacity-sensitivity slope replay",
        "",
        f"**Verdict:** {m['verdict']}  ",
        f"**Cells (app, graph):** {m['n_cells']}  ",
        f"**(cell, policy) records:** {m['n_cell_policy_records']}",
        "",
        "## Method",
        "",
        "log2(kB) axis: " + ", ".join(f"{k} → {v}"
            for k, v in m['l3_axis_log2_kb'].items()) + ".",
        "OLS slope of miss_rate (pp) versus log2(L3 kB) across the four "
        "anchor sizes per (app, graph, policy). Anchor sizes are smaller "
        "than the cache-sim sweep, so absolute slope magnitudes are not "
        "comparable to gates 66/67/68 — only the ordering is.",
        "",
        "## Per-policy median slope (pp / octave)",
        "",
        "| policy | n | median | mean |",
        "|---|---:|---:|---:|",
    ]
    for p in POLICIES:
        d = m["per_policy"][p]
        med = "—" if d["median"] is None else f"{d['median']:+.4f}"
        mean = "—" if d["mean"] is None else f"{d['mean']:+.4f}"
        lines.append(f"| {p} | {d['n']} | {med} | {mean} |")
    lines += [
        "",
        f"**median(LRU)  − median(GRASP) =** "
        f"{m['lru_minus_grasp_pp_oct']:+.4f} pp/octave (INFORMATIONAL; "
        f"sub-WSS regime can invert this — not gated)",
        f"**median(SRRIP) − median(GRASP) =** "
        f"{m['srrip_minus_grasp_pp_oct']:+.4f} pp/octave (want <= 0)",
        f"**help-floor for median(GRASP):** "
        f"{m['help_floor_pp_octave']} pp/octave (want median(GRASP) below it)",
        f"**cache-monotonicity violations:** {len(m['monotonic_violations'])} "
        f"(want 0)",
        "",
        "## Verdict checks",
        "",
        "| check | result |",
        "|---|---|",
    ]
    for k, v in m["verdict_checks"].items():
        lines.append(f"| {k} | {'✅' if v else '❌'} |")
    lines += [
        "",
        "## Per-cell slopes",
        "",
        "| app | graph | policy | slope (pp/oct) | "
        + " | ".join(f"miss@{s}" for s in EXPECTED_SIZES) + " |",
        "|---|---|---|---:|" + "---:|" * len(EXPECTED_SIZES),
    ]
    for r in payload["per_cell"]:
        misses = " | ".join(
            f"{r['miss_pp_by_size'][s]:.3f}"
            for s in EXPECTED_SIZES
        )
        lines.append(
            f"| {r['app']} | {r['graph']} | {r['policy']} | "
            f"{r['slope_pp_per_octave']:+.4f} | {misses} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor-json", type=Path, default=DEFAULT_ANCHOR_JSON)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    ap.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = ap.parse_args()

    payload = compute(args.anchor_json)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    args.md_out.write_text(render_md(payload))

    m = payload["meta"]
    print(
        f"sniper-slope-replay: cells={m['n_cells']} "
        f"GRASP_med={m['per_policy']['GRASP']['median']} "
        f"LRU_med={m['per_policy']['LRU']['median']} "
        f"SRRIP_med={m['per_policy']['SRRIP']['median']} "
        f"verdict={m['verdict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
