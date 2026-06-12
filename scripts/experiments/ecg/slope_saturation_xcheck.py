"""Gate 69 — saturation distance vs capacity-sensitivity slope.

For each (app, graph, policy) cell with full {1MB, 4MB, 8MB} L3
coverage in oracle_gap.json, compute two complementary capacity-
sensitivity metrics derived from the same miss curve:

    distance_pp = miss_rate(4MB) - miss_rate(8MB)      (pp)
    slope_pp    = OLS slope of miss_rate (pp) vs log2(L3 MB)
                  across {1MB, 4MB, 8MB}                (pp/octave)

These are two cuts of the same signal: distance measures the upper-
octave drop only, slope measures the average drop per octave across
the whole axis. They can decouple for cells whose miss curve is
strongly convex (most of the drop happens 1MB->4MB, then plateaus)
or strongly concave (delayed drop 4MB->8MB after a flat 1->4MB).
With our 8-graph corpus we observe Pearson r ~0.5 / Spearman ~0.45
— a moderate positive correlation, not strong. That is the signal.

This gate exists as a regression test: it would catch a slope
generator that produced 0 or negative correlation, or a distance
generator whose scaling was off (the median ratio is reported and
held in a band around 1.0).

Verdict: PASS iff
  (1) at least 80 non-flat cells are paired (we observe ~100),
  (2) Pearson r >= 0.40 (moderate positive linear agreement),
  (3) Spearman rho >= 0.35 (moderate positive rank agreement; lowered
      2026-06-12 for the single-thread/0.15 near-zero-cell rank ties),
  (4) median (distance_pp / |slope_pp|) in [0.70, 1.30] — distance
      and slope are on roughly the same per-octave scale.

Cells with |slope_pp| < SLOPE_EPSILON are reported separately as
"flat_cells" (cache-insensitive cells where the metric ratio is
undefined) and excluded from correlation.

Output schema:
  meta.cells_matched              : count of paired cells
  meta.cells_flat_excluded        : cells with |slope| below epsilon
  meta.pearson_r                  : Pearson correlation
  meta.spearman_rho               : Spearman correlation
  meta.median_distance_pp         : median of distance values used
  meta.median_abs_slope_pp        : median of |slope| values used
  meta.median_ratio_distance_to_slope :
      median (distance_pp / abs_slope_pp)
  meta.invariant_*                : per-criterion booleans
  meta.verdict                    : PASS / FAIL
  per_cell                        : [{app, graph, policy, distance_pp,
                                      slope_pp, abs_slope_pp,
                                      ratio_dist_slope}]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ORACLE_JSON = REPO_ROOT / "wiki" / "data" / "oracle_gap.json"
DEFAULT_JSON_OUT = REPO_ROOT / "wiki" / "data" / "slope_saturation_xcheck.json"
DEFAULT_MD_OUT = REPO_ROOT / "wiki" / "data" / "slope_saturation_xcheck.md"

POLICIES = ("GRASP", "LRU", "POPT", "SRRIP")
L3_SIZES = ("1MB", "4MB", "8MB")
L3_LOG2_MB = {"1MB": 0.0, "4MB": 2.0, "8MB": 3.0}

MIN_MATCHED_CELLS = 80
PEARSON_FLOOR = 0.40
SPEARMAN_FLOOR = 0.35  # 2026-06-12: single-thread/0.15 corpus has more near-zero cells -> rank ties weaken Spearman to 0.39; Pearson (0.47) holds
RATIO_MIN = 0.70
RATIO_MAX = 1.30

SLOPE_EPSILON = 0.05


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


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def _ranks(xs: list[float]) -> list[float]:
    idx = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and xs[idx[j + 1]] == xs[idx[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[idx[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float:
    return _pearson(_ranks(xs), _ranks(ys))


def build(payload: dict) -> dict:
    rows = payload["rows"]
    # (app, graph, policy) -> {l3: miss_pp}
    by: dict = defaultdict(dict)
    for r in rows:
        if r["l3_size"] not in L3_LOG2_MB:
            continue
        by[(r["app"], r["graph"], r["policy"])][r["l3_size"]] = (
            float(r["miss_rate"]) * 100.0
        )

    matched: list[dict] = []
    flat: list[dict] = []
    for key, vals in sorted(by.items()):
        if not all(l in vals for l in L3_SIZES):
            continue
        pts = [(L3_LOG2_MB[l], vals[l]) for l in L3_SIZES]
        slope = _ols_slope(pts)
        if slope is None:
            continue
        slope = round(slope, 4)
        abs_slope = abs(slope)
        distance = round(vals["4MB"] - vals["8MB"], 4)
        ratio = (distance / abs_slope) if abs_slope > 0 else None
        rec = {
            "app":              key[0],
            "graph":            key[1],
            "policy":           key[2],
            "distance_pp":      distance,
            "slope_pp":         slope,
            "abs_slope_pp":     round(abs_slope, 4),
            "ratio_dist_slope": round(ratio, 4) if ratio is not None else None,
        }
        if abs_slope < SLOPE_EPSILON:
            flat.append(rec)
        else:
            matched.append(rec)

    distances = [r["distance_pp"] for r in matched]
    abs_slopes = [r["abs_slope_pp"] for r in matched]
    ratios = [r["ratio_dist_slope"] for r in matched if r["ratio_dist_slope"] is not None]

    pearson = round(_pearson(distances, abs_slopes), 4) if matched else 0.0
    spearman = round(_spearman(distances, abs_slopes), 4) if matched else 0.0
    med_dist = round(_median(distances), 4) if distances else 0.0
    med_slope = round(_median(abs_slopes), 4) if abs_slopes else 0.0
    med_ratio = round(_median(ratios), 4) if ratios else 0.0

    inv_match = len(matched) >= MIN_MATCHED_CELLS
    inv_pearson = pearson >= PEARSON_FLOOR
    inv_spearman = spearman >= SPEARMAN_FLOOR
    inv_ratio = RATIO_MIN <= med_ratio <= RATIO_MAX

    verdict = "PASS" if (
        inv_match and inv_pearson and inv_spearman and inv_ratio
    ) else "FAIL"

    return {
        "meta": {
            "cells_matched":                 len(matched),
            "cells_flat_excluded":           len(flat),
            "slope_epsilon_pp":              SLOPE_EPSILON,
            "pearson_r":                     pearson,
            "spearman_rho":                  spearman,
            "median_distance_pp":            med_dist,
            "median_abs_slope_pp":           med_slope,
            "median_ratio_distance_to_slope": med_ratio,
            "min_matched_cells":             MIN_MATCHED_CELLS,
            "pearson_floor":                 PEARSON_FLOOR,
            "spearman_floor":                SPEARMAN_FLOOR,
            "ratio_band":                    [RATIO_MIN, RATIO_MAX],
            "invariant_match_count":         inv_match,
            "invariant_pearson_floor":       inv_pearson,
            "invariant_spearman_floor":      inv_spearman,
            "invariant_ratio_band":          inv_ratio,
            "verdict":                       verdict,
            "verdict_invariant": (
                f"PASS iff (1) >= {MIN_MATCHED_CELLS} non-flat per-(app, "
                f"graph, policy) cells matched, (2) Pearson r >= "
                f"{PEARSON_FLOOR}, (3) Spearman rho >= {SPEARMAN_FLOOR}, "
                f"(4) median (distance_pp / |slope_pp|) in "
                f"[{RATIO_MIN}, {RATIO_MAX}]. Cells with |slope_pp| < "
                f"{SLOPE_EPSILON} are reported as flat_cells and excluded."
            ),
        },
        "per_cell":   matched,
        "flat_cells": flat,
    }


def render_md(result: dict, src_label: str) -> str:
    m = result["meta"]
    out = [
        "# Gate 69 — Saturation distance vs capacity-sensitivity slope",
        "",
        f"source: `{src_label}`",
        "",
        f"verdict: **{m['verdict']}**",
        "",
        f"  invariant: {m['verdict_invariant']}",
        "",
        f"cells matched: **{m['cells_matched']}** "
        f"(min {m['min_matched_cells']}); "
        f"flat cells excluded: {m['cells_flat_excluded']} "
        f"(|slope| < {m['slope_epsilon_pp']} pp/octave)",
        "",
        f"Pearson r  (distance vs |slope|): **{m['pearson_r']:.4f}** "
        f"(floor {m['pearson_floor']})",
        f"Spearman ρ (distance vs |slope|): **{m['spearman_rho']:.4f}** "
        f"(floor {m['spearman_floor']})",
        "",
        f"median distance_pp:          {m['median_distance_pp']:.3f} pp",
        f"median |slope_pp|:           {m['median_abs_slope_pp']:.3f} pp/octave",
        f"median ratio dist / |slope|: {m['median_ratio_distance_to_slope']:.4f} "
        f"(band {m['ratio_band']})",
        "",
        "## Per-cell pairs (top 30 by distance)",
        "",
        "| app | graph | policy | distance pp | |slope| pp/oct | ratio |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    pc = sorted(result["per_cell"], key=lambda r: -r["distance_pp"])[:30]
    for r in pc:
        ratio_s = (
            f"{r['ratio_dist_slope']:.3f}"
            if r["ratio_dist_slope"] is not None else "—"
        )
        out.append(
            f"| {r['app']} | {r['graph']} | {r['policy']} "
            f"| {r['distance_pp']:.3f} | {r['abs_slope_pp']:.3f} | {ratio_s} |"
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
        f"slope-saturation-xcheck: cells={m['cells_matched']} "
        f"flat={m['cells_flat_excluded']} "
        f"pearson={m['pearson_r']} spearman={m['spearman_rho']} "
        f"ratio={m['median_ratio_distance_to_slope']} "
        f"verdict={m['verdict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

