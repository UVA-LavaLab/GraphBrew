"""Gate 79 — per-family saturation-distance replay.

Mirrors gate 67 (family_curvature_replay) for the saturation-distance
metric (gate 65). For each graph family present in the high-WSS
qualifying subset, computes median 4MB->8MB miss-rate drop in pp and
locks the invariants:

  1. Every family has non-negative median distance (cache helps on
     average; this is the family-level dual of gate 77's cell-level
     monotonicity guard).
  2. Every family has min distance >= 0 (no individual cell shows a
     net miss-rate INCREASE going from 4MB to 8MB once pico sentinels
     are dropped).
  3. Citation and social families have median distance >= 5 pp
     (these families dominate the high-WSS regime and consistently
     show meaningful headroom).
  4. Web family is pinned as the low-headroom outlier: median
     distance < 5 pp (web-Google is the only web graph and is much
     closer to saturation at 4MB).
  5. Family ordering by median: citation >= social >= web (within a
     1.0 pp slack to absorb cell-count noise). Currently citation
     +15.69, social +12.50, web +2.15.

This makes the regime story explicit at the family level: dense
hub-heavy graphs (citation/social) still have meaningful upper-octave
headroom; web-Google is much closer to its saturation point. Pins
the asymmetry against silent regressions where, e.g., a new social
graph pulls the median below web's.

Output schema:
  meta.families                 : list of family labels observed
  meta.per_family[F]            : {n_cells, min_pp, median_pp,
                                   p90_pp, max_pp, graphs}
  meta.pinned_low_headroom_families : list of families with median < 5 pp
  meta.high_headroom_floor_pp   : threshold (5 pp)
  meta.low_headroom_ceiling_pp  : threshold (5 pp; same number,
                                   different role)
  meta.ordering_slack_pp        : slack for the family-ordering check
  meta.verdict_checks
  meta.verdict
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DISTANCE_JSON = REPO_ROOT / "wiki" / "data" / "saturation_distance.json"
DEFAULT_JSON_OUT      = REPO_ROOT / "wiki" / "data" / "family_saturation_distance.json"
DEFAULT_MD_OUT        = REPO_ROOT / "wiki" / "data" / "family_saturation_distance.md"

GRAPH_FAMILY: dict[str, str] = {
    "email-Eu-core":    "social",
    "web-Google":       "web",
    "cit-Patents":      "citation",
    "soc-pokec":        "social",
    "soc-LiveJournal1": "social",
    "com-orkut":        "social",
    "roadNet-CA":       "road",
    "delaunay_n19":     "mesh",
}

HIGH_HEADROOM_FLOOR_PP = 5.0
LOW_HEADROOM_CEILING_PP = 5.0
ORDERING_SLACK_PP = 1.0
PINNED_LOW_HEADROOM = ("web",)
HIGH_HEADROOM_FAMILIES = ("citation", "social")


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _p90(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    idx = max(0, min(n - 1, int(round(0.9 * (n - 1)))))
    return s[idx]


def build(distance_path: Path) -> dict:
    doc = json.loads(distance_path.read_text())
    rows = [c for c in doc["per_cell"] if not c.get("is_pico_sentinel")]

    by_family: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for r in rows:
        fam = GRAPH_FAMILY.get(r["graph"], "unknown")
        by_family[fam].append((r["graph"], float(r["distance_pp"])))

    families = sorted(by_family.keys())
    per_family: dict[str, dict] = {}
    for fam in families:
        entries = by_family[fam]
        vals = [v for _, v in entries]
        graphs = sorted({g for g, _ in entries})
        per_family[fam] = {
            "n_cells":    len(entries),
            "min_pp":     round(min(vals), 4),
            "median_pp":  round(_median(vals), 4),
            "p90_pp":     round(_p90(vals), 4),
            "max_pp":     round(max(vals), 4),
            "graphs":     graphs,
        }

    inv_all_medians_nonneg = all(per_family[f]["median_pp"] >= 0.0 for f in families)
    inv_all_mins_nonneg    = all(per_family[f]["min_pp"]    >= 0.0 for f in families)

    high_present = [f for f in HIGH_HEADROOM_FAMILIES if f in per_family]
    low_present  = [f for f in PINNED_LOW_HEADROOM    if f in per_family]

    inv_high_floor = all(per_family[f]["median_pp"] >= HIGH_HEADROOM_FLOOR_PP for f in high_present)
    inv_low_ceiling = all(per_family[f]["median_pp"] < LOW_HEADROOM_CEILING_PP for f in low_present)

    # Ordering: citation >= social >= web (within slack)
    ordering_ok = True
    if all(f in per_family for f in ("citation", "social", "web")):
        cit = per_family["citation"]["median_pp"]
        soc = per_family["social"]["median_pp"]
        web = per_family["web"]["median_pp"]
        ordering_ok = (cit + ORDERING_SLACK_PP >= soc) and (soc + ORDERING_SLACK_PP >= web)

    inv_at_least_three_families = len([f for f in families if per_family[f]["n_cells"] >= 1]) >= 3

    verdict_checks = {
        "all_family_medians_nonneg":              inv_all_medians_nonneg,
        "all_family_mins_nonneg":                 inv_all_mins_nonneg,
        "high_headroom_families_meet_floor":      inv_high_floor,
        "pinned_low_headroom_under_ceiling":      inv_low_ceiling,
        "family_ordering_citation_social_web":    ordering_ok,
        "at_least_three_families_present":        inv_at_least_three_families,
    }
    verdict = "PASS" if all(verdict_checks.values()) else "FAIL"

    return {
        "meta": {
            "families":                       families,
            "per_family":                     per_family,
            "high_headroom_families":         list(HIGH_HEADROOM_FAMILIES),
            "pinned_low_headroom_families":   list(PINNED_LOW_HEADROOM),
            "high_headroom_floor_pp":         HIGH_HEADROOM_FLOOR_PP,
            "low_headroom_ceiling_pp":        LOW_HEADROOM_CEILING_PP,
            "ordering_slack_pp":              ORDERING_SLACK_PP,
            "source_artifact":                str(distance_path)
                                              if not distance_path.is_absolute()
                                              else str(distance_path.relative_to(REPO_ROOT)),
            "verdict_checks":                 verdict_checks,
            "verdict":                        verdict,
        },
    }


def render_md(payload: dict) -> str:
    m = payload["meta"]
    lines = [
        "# Per-family saturation-distance replay",
        "",
        f"**Verdict:** {m['verdict']}  ",
        f"**Source:** `{m['source_artifact']}`  ",
        f"**High-headroom families:** "
        f"{', '.join(m['high_headroom_families'])} "
        f"(floor {m['high_headroom_floor_pp']} pp)  ",
        f"**Pinned low-headroom families:** "
        f"{', '.join(m['pinned_low_headroom_families'])} "
        f"(ceiling {m['low_headroom_ceiling_pp']} pp)  ",
        f"**Ordering slack:** {m['ordering_slack_pp']} pp",
        "",
        "## Per-family medians",
        "",
        "| family | n_cells | min pp | median pp | p90 pp | max pp | graphs |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for fam in m["families"]:
        e = m["per_family"][fam]
        lines.append(
            f"| {fam} | {e['n_cells']} | "
            f"{e['min_pp']:+.4f} | {e['median_pp']:+.4f} | "
            f"{e['p90_pp']:+.4f} | {e['max_pp']:+.4f} | "
            f"{', '.join(e['graphs'])} |"
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
        "## Interpretation",
        "",
        "Mirror of gate 67 for the saturation-distance metric. "
        "Hub-heavy families (citation, social) still have meaningful "
        "upper-octave headroom (median >= 5 pp 4MB->8MB drop); the "
        "single web graph (web-Google) is pinned as the low-headroom "
        "exemplar (median < 5 pp). Family ordering "
        "citation >= social >= web is locked within a 1 pp slack so "
        "small cell-count shifts don't break the gate but a real "
        "regime flip would.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance-json", type=Path, default=DEFAULT_DISTANCE_JSON)
    ap.add_argument("--json-out",      type=Path, default=DEFAULT_JSON_OUT)
    ap.add_argument("--md-out",        type=Path, default=DEFAULT_MD_OUT)
    args = ap.parse_args()

    payload = build(args.distance_json)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    args.md_out.write_text(render_md(payload))

    m = payload["meta"]
    parts = " ".join(
        f"{f}={m['per_family'][f]['median_pp']:+.2f}"
        for f in m["families"]
    )
    print(f"family-saturation-distance: {parts} verdict={m['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
