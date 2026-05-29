#!/usr/bin/env python3
"""Per-(family × app × policy) geomean miss-rate improvement vs LRU.

For every (family, app, policy != LRU) tuple, computes the geometric mean
of (miss_rate(policy) / miss_rate(LRU)) over the (graph, L3) cells in that
slice, plus a 95% percentile bootstrap CI (B=2000, seed=1729) on the geomean
ratio. Reports the *size* of the improvement that the significance gates
(34/35/36/37/38/40) only show direction for.

Scope is restricted to paper-realistic L3 sizes (>= 1MB) so the geomean
isn't dominated by 4kB regime data that has no LRU baseline reuse to begin
with. Families whose graphs cap below 1MB (road, mesh) get caught by the
n >= 2 floor and surface honestly with whatever paper-L3 cells they have.

Output: wiki/data/family_geomean_improvement.{json,md}
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

PAPER_L3_SIZES = ("1MB", "4MB", "8MB")
BOOTSTRAP_ITERS = 2000
BOOTSTRAP_SEED = 1729
ALPHA = 0.05


def geomean(xs: Iterable[float]) -> float:
    xs = [x for x in xs if x > 0]
    if not xs:
        raise ValueError("geomean of empty / non-positive sample")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def bootstrap_geomean_ci(
    sample: list[float],
    iters: int = BOOTSTRAP_ITERS,
    alpha: float = ALPHA,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Returns (geomean_point, ci_lo, ci_hi)."""
    rng = random.Random(seed)
    n = len(sample)
    if n < 2:
        raise ValueError(f"need at least 2 paired cells to bootstrap, got {n}")
    point = geomean(sample)
    boot = []
    for _ in range(iters):
        resample = [sample[rng.randrange(n)] for _ in range(n)]
        boot.append(geomean(resample))
    boot.sort()
    lo_idx = int(math.floor((alpha / 2.0) * iters))
    hi_idx = int(math.ceil((1.0 - alpha / 2.0) * iters)) - 1
    lo_idx = max(0, min(iters - 1, lo_idx))
    hi_idx = max(0, min(iters - 1, hi_idx))
    return point, boot[lo_idx], boot[hi_idx]


def collect_paired_ratios(
    rows: list[dict], scope_l3: set[str]
) -> dict[tuple[str, str, str], list[dict]]:
    """For each (family, app, policy != LRU), return list of cell dicts with
    keys: graph, l3_size, mr_policy, mr_lru, ratio. Only includes cells where
    LRU baseline exists and both miss-rates are > 0."""
    # group by (family, app, graph, l3_size) -> policy -> mr
    by_cell: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(dict)
    for r in rows:
        l3 = r["l3_size"]
        if scope_l3 and l3 not in scope_l3:
            continue
        family = r.get("family")
        if not family:
            continue
        mr = float(r["miss_rate"])
        by_cell[(family, r["app"], r["graph"], l3)][r["policy"]] = mr

    result: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for (family, app, graph, l3), pmap in by_cell.items():
        lru = pmap.get("LRU")
        if lru is None or lru <= 0:
            continue
        for pol, mr in pmap.items():
            if pol == "LRU" or mr <= 0:
                continue
            result[(family, app, pol)].append(
                {
                    "graph": graph,
                    "l3_size": l3,
                    "mr_policy": mr,
                    "mr_lru": lru,
                    "ratio": mr / lru,
                }
            )
    return result


def build_record(family: str, app: str, policy: str, cells: list[dict]) -> dict:
    ratios = [c["ratio"] for c in cells]
    if len(ratios) < 2:
        return {
            "family": family,
            "app": app,
            "policy": policy,
            "n_cells": len(ratios),
            "skipped_reason": "insufficient_cells_for_bootstrap_min_2",
            "cells": cells,
        }
    point, ci_lo, ci_hi = bootstrap_geomean_ci(ratios)
    # convert ratio to improvement-vs-LRU pct (positive = improvement)
    improve_pct_point = (1.0 - point) * 100.0
    improve_pct_lo = (1.0 - ci_hi) * 100.0
    improve_pct_hi = (1.0 - ci_lo) * 100.0
    ci_strict_improvement = ci_hi < 1.0
    ci_strict_regression = ci_lo > 1.0
    return {
        "family": family,
        "app": app,
        "policy": policy,
        "n_cells": len(ratios),
        "geomean_ratio": round(point, 6),
        "ci_lo_ratio": round(ci_lo, 6),
        "ci_hi_ratio": round(ci_hi, 6),
        "geomean_improve_pct": round(improve_pct_point, 3),
        "ci_lo_improve_pct": round(improve_pct_lo, 3),
        "ci_hi_improve_pct": round(improve_pct_hi, 3),
        "ci_strict_improvement_vs_lru": bool(ci_strict_improvement),
        "ci_strict_regression_vs_lru": bool(ci_strict_regression),
        "cells": cells,
    }


def build_payload(oracle_path: Path) -> dict:
    raw = json.loads(oracle_path.read_text())
    rows = raw["rows"]
    paired = collect_paired_ratios(rows, set(PAPER_L3_SIZES))

    records = []
    for key in sorted(paired.keys()):
        family, app, policy = key
        records.append(build_record(family, app, policy, paired[key]))

    # headlines: CI-strict improvements with |improvement| >= 10pp, ranked
    headline_improvements = sorted(
        (
            r
            for r in records
            if r.get("ci_strict_improvement_vs_lru")
            and r.get("geomean_improve_pct", 0.0) >= 10.0
        ),
        key=lambda r: -r["geomean_improve_pct"],
    )
    headline_regressions = sorted(
        (r for r in records if r.get("ci_strict_regression_vs_lru")),
        key=lambda r: r["geomean_improve_pct"],  # most-negative first
    )

    try:
        src_label = str(oracle_path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        src_label = str(oracle_path)

    payload = {
        "meta": {
            "source": src_label,
            "scope_l3_sizes": list(PAPER_L3_SIZES),
            "bootstrap_iters": BOOTSTRAP_ITERS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "alpha": ALPHA,
            "n_records": len(records),
            "n_ci_strict_improvements": sum(
                1 for r in records if r.get("ci_strict_improvement_vs_lru")
            ),
            "n_ci_strict_regressions": sum(
                1 for r in records if r.get("ci_strict_regression_vs_lru")
            ),
        },
        "headline_improvements_ge_10pct": [
            {k: v for k, v in r.items() if k != "cells"}
            for r in headline_improvements
        ],
        "headline_regressions_ci_strict": [
            {k: v for k, v in r.items() if k != "cells"}
            for r in headline_regressions
        ],
        "records": records,
    }
    return payload


def emit_md(payload: dict) -> str:
    meta = payload["meta"]
    out = []
    out.append("# Family × App × Policy: geomean improvement vs LRU")
    out.append("")
    out.append(
        f"Source: `{meta['source']}`  •  L3 scope: {', '.join(meta['scope_l3_sizes'])}"
    )
    out.append(
        f"Bootstrap: B={meta['bootstrap_iters']}, seed={meta['bootstrap_seed']},"
        f" α={meta['alpha']} (percentile CI)"
    )
    out.append("")
    out.append(
        f"Records: **{meta['n_records']}** "
        f"({meta['n_ci_strict_improvements']} CI-strict improvements vs LRU, "
        f"{meta['n_ci_strict_regressions']} CI-strict regressions)."
    )
    out.append("")
    out.append("## Headline CI-strict improvements (≥10% miss-rate reduction)")
    out.append("")
    out.append("| Family | App | Policy | n | Geomean | CI ratio | Improve%  (CI) |")
    out.append("|---|---|---|---|---|---|---|")
    for r in payload["headline_improvements_ge_10pct"]:
        ci_ratio = f"[{r['ci_lo_ratio']:.3f}, {r['ci_hi_ratio']:.3f}]"
        ci_pct = (
            f"+{r['geomean_improve_pct']:.2f}% "
            f"[+{r['ci_lo_improve_pct']:.2f}, +{r['ci_hi_improve_pct']:.2f}]"
        )
        out.append(
            f"| {r['family']} | {r['app']} | {r['policy']} | {r['n_cells']}"
            f" | {r['geomean_ratio']:.4f} | {ci_ratio} | {ci_pct} |"
        )
    out.append("")
    out.append("## CI-strict regressions (geomean ratio > 1.0, CI lo > 1.0)")
    out.append("")
    if not payload["headline_regressions_ci_strict"]:
        out.append("_None — no policy CI-strictly regresses on any (family, app)._")
    else:
        out.append("| Family | App | Policy | n | Geomean | CI ratio | Improve%  (CI) |")
        out.append("|---|---|---|---|---|---|---|")
        for r in payload["headline_regressions_ci_strict"]:
            ci_ratio = f"[{r['ci_lo_ratio']:.3f}, {r['ci_hi_ratio']:.3f}]"
            ci_pct = (
                f"{r['geomean_improve_pct']:+.2f}% "
                f"[{r['ci_lo_improve_pct']:+.2f}, {r['ci_hi_improve_pct']:+.2f}]"
            )
            out.append(
                f"| {r['family']} | {r['app']} | {r['policy']} | {r['n_cells']}"
                f" | {r['geomean_ratio']:.4f} | {ci_ratio} | {ci_pct} |"
            )
    out.append("")
    out.append("## All records (sorted by family, app, policy)")
    out.append("")
    out.append("| Family | App | Policy | n | Geomean ratio | CI ratio | Improve% | CI-strict |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in payload["records"]:
        if r.get("skipped_reason"):
            out.append(
                f"| {r['family']} | {r['app']} | {r['policy']} | {r['n_cells']}"
                f" | _skipped_ | _skipped_ | _skipped_ | _{r['skipped_reason']}_ |"
            )
            continue
        ci_ratio = f"[{r['ci_lo_ratio']:.3f}, {r['ci_hi_ratio']:.3f}]"
        ci_marker = (
            "improvement"
            if r["ci_strict_improvement_vs_lru"]
            else ("regression" if r["ci_strict_regression_vs_lru"] else "—")
        )
        out.append(
            f"| {r['family']} | {r['app']} | {r['policy']} | {r['n_cells']}"
            f" | {r['geomean_ratio']:.4f} | {ci_ratio}"
            f" | {r['geomean_improve_pct']:+.2f}% | {ci_marker} |"
        )
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle-json",
        type=Path,
        default=WIKI_DATA / "oracle_gap.json",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=WIKI_DATA / "family_geomean_improvement.json",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=WIKI_DATA / "family_geomean_improvement.md",
    )
    args = parser.parse_args()

    payload = build_payload(args.oracle_json)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(emit_md(payload) + "\n")
    print(
        f"family_geomean_improvement: {payload['meta']['n_records']} records,"
        f" {payload['meta']['n_ci_strict_improvements']} CI-strict improvements,"
        f" {payload['meta']['n_ci_strict_regressions']} CI-strict regressions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
