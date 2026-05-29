"""Gate 65 — per-app saturation distance at the 4MB->8MB step.

For each (app, graph) cell with both 4MB and 8MB measurements, the
"saturation distance" is the gap between the best-policy miss rate at
4MB and the best-policy miss rate at 8MB:

    distance_pp(app, graph) = 100 * (best_miss_at_4MB - best_miss_at_8MB)

A small distance means the app+graph has saturated its working set
within the cache; going from 4MB to 8MB barely helps. A large distance
means cache is still the bottleneck even at 8MB.

The per-app median over all graphs quantifies how compute-bound vs
memory-bound each application is in our corpus.

This gate pins three structural facts:

  1. Saturation distance is non-negative for every cell where the
     graph's WSS exceeds 4MB (i.e., where 4MB cannot already fit the
     full working set). Negative distance would imply 8MB is *worse*
     than 4MB on the best-policy miss rate, which never happens in a
     monotone cache hierarchy.

  2. email-Eu-core is saturated for every app (distance < 0.05 pp).
     Its WSS proxy is ~4.5 kB, so even a 4 kB cache nearly fits the
     full working set; this is the corpus's pico-sentinel.

  3. App-level diversity: the per-app median distance varies by at
     least 3 pp between the most-saturated app and the least-saturated
     app. If every app saturated identically the corpus would have
     no app-level signal left.

Output schema:
  meta.app_count                : number of apps measured
  meta.graph_count              : number of graphs measured per app
                                  (must be uniform; the corpus is
                                  fully populated at 4/8 MB today)
  meta.per_app                  : app -> {median, mean, p90, max, min}
  meta.app_diversity_range_pp   : max(median) - min(median) over apps
  meta.app_diversity_threshold  : 3.0
  meta.verdict                  : PASS iff all three invariants hold
  per_cell                      : list of {app, graph, best4_miss_pp,
                                  best8_miss_pp, distance_pp,
                                  is_pico_sentinel}
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ORACLE_JSON = REPO_ROOT / "wiki" / "data" / "oracle_gap.json"
DEFAULT_WSS_JSON = REPO_ROOT / "wiki" / "data" / "wss_relative_l3.json"
DEFAULT_JSON_OUT = REPO_ROOT / "wiki" / "data" / "saturation_distance.json"
DEFAULT_MD_OUT = REPO_ROOT / "wiki" / "data" / "saturation_distance.md"

# Pico-sentinel: very small graph whose WSS fits comfortably in any
# studied L3 size.
PICO_GRAPH = "email-Eu-core"
PICO_SATURATION_PP = 0.05

# Per-app median saturation distance must vary by at least this many
# pp across apps (otherwise the corpus loses app-level signal).
APP_DIVERSITY_THRESHOLD_PP = 3.0

# WSS floor (bytes) for the non-negative-distance invariant. A graph
# whose WSS is below 4 MB might already fit the full working set at
# 4 MB, so an "8 MB is worse" anomaly there is noise rather than a
# physics violation; we only enforce non-negativity on graphs with
# WSS > 4 MB.
WSS_FLOOR_BYTES = 4 * 1024 * 1024


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[k]


def build(oracle_payload: dict, wss_meta: dict) -> dict:
    wss_proxies = wss_meta["wss_proxies"]
    rows = oracle_payload["rows"]
    cells: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        cells[(r["app"], r["graph"])][r["l3_size"]].append(float(r["miss_rate"]))

    per_cell = []
    per_app_distances: dict = defaultdict(list)
    non_negative_violations = []
    for (app, graph), by_l3 in sorted(cells.items()):
        if "4MB" not in by_l3 or "8MB" not in by_l3:
            continue
        best4 = min(by_l3["4MB"]) * 100.0
        best8 = min(by_l3["8MB"]) * 100.0
        dist_pp = round(best4 - best8, 4)
        is_pico = graph == PICO_GRAPH
        wss = wss_proxies.get(graph, 0)
        per_cell.append({
            "app":              app,
            "graph":            graph,
            "wss_bytes":        wss,
            "best4_miss_pp":    round(best4, 4),
            "best8_miss_pp":    round(best8, 4),
            "distance_pp":      dist_pp,
            "is_pico_sentinel": is_pico,
        })
        per_app_distances[app].append(dist_pp)
        if wss > WSS_FLOOR_BYTES and dist_pp < 0:
            non_negative_violations.append({
                "app": app, "graph": graph, "distance_pp": dist_pp,
            })

    per_app: dict[str, dict] = {}
    medians = []
    for app, ds in sorted(per_app_distances.items()):
        if not ds:
            continue
        med = round(_median(ds), 4)
        per_app[app] = {
            "n_graphs":      len(ds),
            "median_pp":     med,
            "mean_pp":       round(sum(ds) / len(ds), 4),
            "p90_pp":        round(_pct(ds, 0.9), 4),
            "max_pp":        round(max(ds), 4),
            "min_pp":        round(min(ds), 4),
        }
        medians.append(med)
    diversity_range = round(max(medians) - min(medians), 4) if medians else 0.0

    pico_cells = [c for c in per_cell if c["is_pico_sentinel"]]
    pico_violations = [
        c for c in pico_cells if c["distance_pp"] > PICO_SATURATION_PP
    ]

    inv_nonneg = (len(non_negative_violations) == 0)
    inv_pico = (len(pico_violations) == 0)
    inv_div = (diversity_range >= APP_DIVERSITY_THRESHOLD_PP)
    verdict = "PASS" if (inv_nonneg and inv_pico and inv_div) else "FAIL"

    return {
        "meta": {
            "app_count":                 len(per_app),
            "cell_count":                len(per_cell),
            "pico_graph":                PICO_GRAPH,
            "pico_saturation_pp":        PICO_SATURATION_PP,
            "wss_floor_bytes":           WSS_FLOOR_BYTES,
            "app_diversity_range_pp":    diversity_range,
            "app_diversity_threshold":   APP_DIVERSITY_THRESHOLD_PP,
            "non_negative_violations":   non_negative_violations,
            "pico_violations":           pico_violations,
            "invariant_non_negative":    inv_nonneg,
            "invariant_pico_saturated":  inv_pico,
            "invariant_app_diversity":   inv_div,
            "verdict":                   verdict,
            "verdict_invariant": (
                "PASS iff (1) every cell with WSS > 4 MB has non-negative "
                "4MB->8MB best-policy improvement, (2) email-Eu-core "
                "(pico-sentinel) is saturated for every app within "
                f"{PICO_SATURATION_PP} pp, and (3) per-app median "
                "saturation distance varies by at least "
                f"{APP_DIVERSITY_THRESHOLD_PP} pp across apps."
            ),
        },
        "per_app":  per_app,
        "per_cell": per_cell,
    }


def render_md(result: dict, src_label: str) -> str:
    m = result["meta"]
    out = [
        "# Gate 65 — Per-app saturation distance at 4MB->8MB",
        "",
        f"source: `{src_label}`",
        "",
        f"verdict: **{m['verdict']}**",
        "",
        f"  invariant: {m['verdict_invariant']}",
        "",
        f"cells measured: {m['cell_count']} across {m['app_count']} apps",
        "",
        f"app-level diversity range: {m['app_diversity_range_pp']} pp "
        f"(threshold {m['app_diversity_threshold']})",
        "",
        "## Per-app saturation distance (pp of best-policy miss rate)",
        "",
        "| app | n graphs | median pp | mean pp | p90 pp | max pp | min pp |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for app in sorted(result["per_app"].keys()):
        s = result["per_app"][app]
        out.append(
            f"| {app} | {s['n_graphs']} | {s['median_pp']:.3f} "
            f"| {s['mean_pp']:.3f} | {s['p90_pp']:.3f} "
            f"| {s['max_pp']:.3f} | {s['min_pp']:.3f} |"
        )
    out.extend([
        "",
        "## All cells (4MB -> 8MB best-policy improvement)",
        "",
        "| app | graph | best4 pp | best8 pp | distance pp | sentinel |",
        "| --- | --- | ---: | ---: | ---: | :---: |",
    ])
    for c in result["per_cell"]:
        sent = "pico" if c["is_pico_sentinel"] else ""
        out.append(
            f"| {c['app']} | {c['graph']} "
            f"| {c['best4_miss_pp']:.3f} | {c['best8_miss_pp']:.3f} "
            f"| {c['distance_pp']:.3f} | {sent} |"
        )
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oracle-json", type=Path, default=DEFAULT_ORACLE_JSON)
    ap.add_argument("--wss-json", type=Path, default=DEFAULT_WSS_JSON)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    ap.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = ap.parse_args()

    src_path = args.oracle_json
    try:
        src_label = str(src_path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        src_label = str(src_path)

    oracle = json.loads(args.oracle_json.read_text())
    wss = json.loads(args.wss_json.read_text())
    result = build(oracle, wss["meta"])
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True))
    args.md_out.write_text(render_md(result, src_label))
    m = result["meta"]
    print(
        f"saturation-distance: cells={m['cell_count']} "
        f"apps={m['app_count']} "
        f"diversity_pp={m['app_diversity_range_pp']} "
        f"nonneg_viol={len(m['non_negative_violations'])} "
        f"pico_viol={len(m['pico_violations'])} "
        f"verdict={m['verdict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
