"""Gate 75 — per-app saturation-vs-slope extremum corroboration.

Gate 65 (saturation_distance) measures the per-app mean miss-rate
drop from 4MB to 8MB across all policies and graphs — the residual
cache-sensitivity at large caches.

Gate 68 (per_app_capacity_slope) measures the per-app median OLS
slope of miss-rate vs log2(L3 MB) across the 1MB-8MB sweep — the
average cache-sensitivity across the whole sweep.

Both metrics independently measure "how much does extra cache help
this kernel?" but they aggregate differently:

  * distance = ONE point (4MB->8MB drop), captures upper-octave
    residual headroom.
  * slope    = OLS over THREE points, captures average per-octave
    drop across the whole sweep.

These two metrics typically disagree on the per-app ranking (gate 69
already documented that the per-cell correlation is moderate, not
strong, because heterogeneous curve shapes — convex vs concave —
decouple upper-octave drop from average per-octave drop). However,
both metrics MUST agree at the EXTREMUM: bfs is unambiguously the
least cache-sensitive kernel by both measures, which is a real
corpus property (frontier-driven streaming pattern — gate 65 flags
bfs as most-saturated, gates 68/73 pin bfs for LRU-vs-GRASP and
SRRIP-vs-GRASP slope deviations).

The most cache-hungry app DOES NOT agree across metrics:

  * slope-steepest    : sssp (-19.4 pp/oct median across policies)
                        — its full sweep shows uniform large drops
  * distance-largest  : bc   (+15.6 pp mean 4MB->8MB drop)
                        — its upper octave has the largest residual

This INFORMATIONAL disagreement IS the regime-vs-aggregate
distinction in action and is reported but explicitly NOT gated.

PASS iff:
  (1) bfs is argmin(distance) — smallest 4MB->8MB drop,
  (2) bfs is argmin(|slope|) — shallowest OLS slope (closest to 0),
  (3) every other app has BOTH steeper slope AND larger distance
      than bfs (bfs is the unique extremum on both metrics),
  (4) at least one app has slope steepness > 3x bfs's
      (corpus contains genuinely cache-sensitive kernels by slope),
  (5) at least one app has distance > 2.5x bfs's
      (corpus contains genuinely cache-sensitive kernels by distance).

Output schema:
  meta.distance_source            : "saturation_distance.json"
  meta.slope_source               : "per_app_capacity_slope.json"
  meta.per_app[A]                 : distance_pp, slope_pp_oct,
                                    slope_steepness, distance_rank,
                                    slope_rank
  meta.least_cache_sensitive_app_by_distance : argmin(distance)
  meta.least_cache_sensitive_app_by_slope    : argmin(|slope|)
  meta.most_cache_hungry_app_by_distance     : argmax(distance) [INFO]
  meta.most_cache_hungry_app_by_slope        : argmin(slope) [INFO]
  meta.most_hungry_app_disagreement_note     : INFORMATIONAL note
  meta.verdict                    : PASS / FAIL
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DISTANCE_JSON = REPO_ROOT / "wiki" / "data" / "saturation_distance.json"
DEFAULT_SLOPE_JSON    = REPO_ROOT / "wiki" / "data" / "per_app_capacity_slope.json"
DEFAULT_JSON_OUT      = REPO_ROOT / "wiki" / "data" / "saturation_slope_extremum.json"
DEFAULT_MD_OUT        = REPO_ROOT / "wiki" / "data" / "saturation_slope_extremum.md"

EXPECTED_BFS = "bfs"
SLOPE_STEEPNESS_RATIO_FLOOR = 3.0
DISTANCE_RATIO_FLOOR = 2.5


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def build(distance_path: Path, slope_path: Path) -> dict:
    dist_doc  = json.loads(distance_path.read_text())
    slope_doc = json.loads(slope_path.read_text())
    per_app_dist  = dist_doc["per_app"]
    per_app_slope = slope_doc["meta"]["per_app"]

    apps_in_dist  = set(per_app_dist.keys())
    apps_in_slope = set(per_app_slope.keys())
    common = sorted(apps_in_dist & apps_in_slope)

    rows: list[dict] = []
    for app in common:
        dist = float(per_app_dist[app]["mean_pp"])
        pol_slopes = [
            b["median_pp"] for b in per_app_slope[app].values()
        ]
        if not pol_slopes:
            continue
        slope = _median(pol_slopes)
        rows.append({
            "app":              app,
            "distance_pp":      round(dist, 4),
            "slope_pp_oct":     round(slope, 4),
            "slope_steepness":  round(abs(slope), 4),
        })

    rows_by_dist  = sorted(rows, key=lambda r: r["distance_pp"])
    rows_by_slope = sorted(rows, key=lambda r: r["slope_steepness"])

    dist_rank  = {r["app"]: i + 1 for i, r in enumerate(rows_by_dist)}
    slope_rank = {r["app"]: i + 1 for i, r in enumerate(rows_by_slope)}

    per_app: dict[str, dict] = {}
    for r in rows:
        per_app[r["app"]] = {
            "distance_pp":     r["distance_pp"],
            "slope_pp_oct":    r["slope_pp_oct"],
            "slope_steepness": r["slope_steepness"],
            "distance_rank":   dist_rank[r["app"]],
            "slope_rank":      slope_rank[r["app"]],
        }

    least_by_dist  = rows_by_dist[0]["app"]  if rows_by_dist  else None
    least_by_slope = rows_by_slope[0]["app"] if rows_by_slope else None
    most_by_dist   = rows_by_dist[-1]["app"]  if rows_by_dist  else None
    most_by_slope  = sorted(rows, key=lambda r: r["slope_pp_oct"])[0]["app"] if rows else None

    bfs_block = per_app.get(EXPECTED_BFS)

    inv_bfs_is_least_by_dist  = (least_by_dist  == EXPECTED_BFS)
    inv_bfs_is_least_by_slope = (least_by_slope == EXPECTED_BFS)

    inv_bfs_is_unique_extremum = bool(bfs_block) and all(
        e["distance_pp"]     >  bfs_block["distance_pp"]
        and e["slope_steepness"] > bfs_block["slope_steepness"]
        for a, e in per_app.items() if a != EXPECTED_BFS
    )

    slope_ratios = [
        e["slope_steepness"] / bfs_block["slope_steepness"]
        for a, e in per_app.items() if a != EXPECTED_BFS
    ] if bfs_block and bfs_block["slope_steepness"] > 0 else []
    dist_ratios = [
        e["distance_pp"] / bfs_block["distance_pp"]
        for a, e in per_app.items() if a != EXPECTED_BFS
    ] if bfs_block and bfs_block["distance_pp"] > 0 else []

    inv_slope_corpus_sensitive = any(
        r >= SLOPE_STEEPNESS_RATIO_FLOOR for r in slope_ratios
    ) if slope_ratios else False
    inv_dist_corpus_sensitive = any(
        r >= DISTANCE_RATIO_FLOOR for r in dist_ratios
    ) if dist_ratios else False

    verdict_checks = {
        "bfs_is_argmin_distance":   inv_bfs_is_least_by_dist,
        "bfs_is_shallowest_slope":  inv_bfs_is_least_by_slope,
        "bfs_unique_extremum_on_both_metrics": inv_bfs_is_unique_extremum,
        "corpus_has_slope_steeper_than_3x_bfs":     inv_slope_corpus_sensitive,
        "corpus_has_distance_larger_than_2_5x_bfs": inv_dist_corpus_sensitive,
    }
    verdict = "PASS" if all(verdict_checks.values()) else "FAIL"

    note = (
        "INFORMATIONAL: the most cache-hungry app DISAGREES across "
        "metrics. By distance (4MB->8MB upper-octave drop), the most "
        f"cache-hungry app is {most_by_dist!r}. By slope (OLS over "
        f"the 1MB-8MB sweep), the most cache-hungry app is "
        f"{most_by_slope!r}. This is not a fault — it is the regime-"
        "vs-aggregate distinction at play: distance captures upper-"
        "octave residual headroom while slope averages per-octave "
        "drop across the whole sweep. Apps with convex miss curves "
        "rank differently from apps with concave miss curves. This "
        "gate explicitly does NOT enforce agreement on the most-"
        "hungry extremum — only on the least-hungry extremum (bfs)."
    )

    return {
        "meta": {
            "distance_source":  str(distance_path.name),
            "slope_source":     str(slope_path.name),
            "apps":             common,
            "per_app":          per_app,
            "least_cache_sensitive_app_by_distance":  least_by_dist,
            "least_cache_sensitive_app_by_slope":     least_by_slope,
            "most_cache_hungry_app_by_distance":      most_by_dist,
            "most_cache_hungry_app_by_slope":         most_by_slope,
            "most_hungry_app_disagreement_note":      note,
            "slope_steepness_ratio_floor":            SLOPE_STEEPNESS_RATIO_FLOOR,
            "distance_ratio_floor":                   DISTANCE_RATIO_FLOOR,
            "verdict_checks":                         verdict_checks,
            "verdict":                                verdict,
        },
    }


def render_md(payload: dict) -> str:
    m = payload["meta"]
    lines = [
        "# Per-app saturation-vs-slope extremum corroboration",
        "",
        f"**Verdict:** {m['verdict']}  ",
        f"**Distance source:** `{m['distance_source']}`  ",
        f"**Slope source:** `{m['slope_source']}`  ",
        f"**Least cache-sensitive by distance:** "
        f"`{m['least_cache_sensitive_app_by_distance']}`  ",
        f"**Least cache-sensitive by slope:** "
        f"`{m['least_cache_sensitive_app_by_slope']}`  ",
        f"**Most cache-hungry by distance:** "
        f"`{m['most_cache_hungry_app_by_distance']}` (INFO)  ",
        f"**Most cache-hungry by slope:** "
        f"`{m['most_cache_hungry_app_by_slope']}` (INFO)  ",
        "",
        "## Per-app rankings",
        "",
        "| app | distance pp | dist rank | slope pp/oct | slope rank |",
        "|---|---:|:---:|---:|:---:|",
    ]
    for app in m["apps"]:
        e = m["per_app"][app]
        lines.append(
            f"| {app} | {e['distance_pp']:+.4f} | {e['distance_rank']} | "
            f"{e['slope_pp_oct']:+.4f} | {e['slope_rank']} |"
        )

    lines += [
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
        "## INFORMATIONAL note",
        "",
        m["most_hungry_app_disagreement_note"],
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance-json", type=Path, default=DEFAULT_DISTANCE_JSON)
    ap.add_argument("--slope-json",    type=Path, default=DEFAULT_SLOPE_JSON)
    ap.add_argument("--json-out",      type=Path, default=DEFAULT_JSON_OUT)
    ap.add_argument("--md-out",        type=Path, default=DEFAULT_MD_OUT)
    args = ap.parse_args()

    payload = build(args.distance_json, args.slope_json)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    args.md_out.write_text(render_md(payload))

    m = payload["meta"]
    print(
        f"saturation-slope-extremum: apps={len(m['apps'])} "
        f"least_dist={m['least_cache_sensitive_app_by_distance']} "
        f"least_slope={m['least_cache_sensitive_app_by_slope']} "
        f"verdict={m['verdict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
