#!/usr/bin/env python3
"""Wilson 95% CIs on policy win-counts.

Why this exists
---------------
Throughout the dashboard the phrase "policy X wins Y of N cells" is
used as an evidence-by-counting argument. With small N (per-app
totals are 20–28), point estimates like "GRASP wins 17/20 cc cells"
need confidence intervals before they survive review.

Wilson score intervals are the right tool: they handle small n,
they handle the p̂=0/1 edge cases (jackknife/normal-approx break
down there), and they are the canonical method for proportions.

What we compute
---------------
For every (scope, policy) where scope ∈ {overall, per-app, per-family},
we compute:

* win_count  — Σ ``is_winner`` rows for that (scope, policy)
* total      — number of cells that policy appears in within scope
* p_hat      — win_count / total
* ci_lo, ci_hi — Wilson 95% CI bounds

Test gates pin:

* POPT-on-pr CI lower bound > 0.5 (CI-strict majority winner).
* GRASP-on-cc CI lower bound > 0.5 and excludes 0.25 (chance baseline
  for 4 policies).
* GRASP-on-bc CI excludes 0.25 (above random).
* LRU CI upper bound < 0.5 on every scope where it appears (no scope
  in which LRU is plausibly a majority winner).
* The narrow ties — sssp POPT vs GRASP — are NOT CI-distinguishable;
  this is pinned as an explicit "no claim" gate.

Output
------
* ``wiki/data/wilson_win_rates.json``
* ``wiki/data/wilson_win_rates.md``
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "wiki" / "data"

POLICIES = ("LRU", "SRRIP", "GRASP", "POPT")
APPS = ("pr", "bc", "cc", "bfs", "sssp")
Z_95 = 1.959963984540054  # two-sided 95% normal quantile


def wilson_ci(wins: int, total: int, z: float = Z_95) -> tuple[float, float, float]:
    """Wilson score interval for binomial proportion.

    Returns ``(p_hat, ci_lo, ci_hi)``. For ``total == 0`` returns
    ``(0.0, 0.0, 1.0)`` (uninformative wide interval) so callers do
    not have to special-case empty scopes.
    """
    if total <= 0:
        return 0.0, 0.0, 1.0
    p_hat = wins / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p_hat + z2 / (2.0 * total)) / denom
    margin = (
        z
        * math.sqrt(p_hat * (1.0 - p_hat) / total + z2 / (4.0 * total * total))
        / denom
    )
    return p_hat, max(0.0, center - margin), min(1.0, center + margin)


def load_oracle_rows(path: Path) -> list[dict]:
    blob = json.loads(path.read_text())
    if isinstance(blob, dict) and "rows" in blob:
        return blob["rows"]
    return blob  # tolerate raw-list shape


def _is_winner(row: dict) -> bool:
    val = row.get("is_winner")
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return int(val) == 1
    if isinstance(val, str):
        return val.strip() in {"1", "true", "True"}
    return False


def aggregate(rows: list[dict]) -> dict:
    overall = defaultdict(lambda: [0, 0])  # policy -> [wins, total]
    by_app = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    by_family = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    for r in rows:
        pol = r["policy"]
        app = r["app"]
        fam = r["family"]
        w = 1 if _is_winner(r) else 0
        overall[pol][0] += w
        overall[pol][1] += 1
        by_app[app][pol][0] += w
        by_app[app][pol][1] += 1
        by_family[fam][pol][0] += w
        by_family[fam][pol][1] += 1

    def _emit(d):
        out = {}
        for pol in sorted(d):
            wins, total = d[pol]
            p_hat, lo, hi = wilson_ci(wins, total)
            out[pol] = {
                "wins": wins,
                "total": total,
                "p_hat": round(p_hat, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
            }
        return out

    return {
        "overall": _emit(overall),
        "per_app": {a: _emit(by_app[a]) for a in sorted(by_app)},
        "per_family": {f: _emit(by_family[f]) for f in sorted(by_family)},
    }


def render_md(payload: dict) -> str:
    meta = payload["meta"]
    lines = [
        "# Wilson 95% CIs on policy win-counts",
        "",
        f"Source: `{meta['source']}`.",
        f"Z = {meta['z']:.6f} (two-sided 95%).",
        "",
        "## Overall (all cells)",
        "",
        "| Policy | Wins / N | p̂ | 95% Wilson CI |",
        "| --- | ---: | ---: | --- |",
    ]
    for pol in POLICIES:
        row = payload["overall"].get(pol)
        if not row:
            continue
        lines.append(
            f"| {pol} | {row['wins']} / {row['total']} | {row['p_hat']:.3f} "
            f"| [{row['ci_lo']:.3f}, {row['ci_hi']:.3f}] |"
        )

    lines += ["", "## Per-app", ""]
    for app in APPS:
        if app not in payload["per_app"]:
            continue
        lines += [
            f"### {app}",
            "",
            "| Policy | Wins / N | p̂ | 95% Wilson CI | CI-strict |",
            "| --- | ---: | ---: | --- | --- |",
        ]
        for pol in POLICIES:
            row = payload["per_app"][app].get(pol)
            if not row:
                continue
            strict = []
            if row["ci_lo"] > 0.5:
                strict.append("majority")
            if row["ci_lo"] > 0.25:
                strict.append("above-chance")
            if row["ci_hi"] < 0.25:
                strict.append("below-chance")
            if row["ci_hi"] < 0.5:
                strict.append("minority")
            lines.append(
                f"| {pol} | {row['wins']} / {row['total']} | {row['p_hat']:.3f} "
                f"| [{row['ci_lo']:.3f}, {row['ci_hi']:.3f}] "
                f"| {', '.join(strict) or '-'} |"
            )
        lines.append("")

    lines += ["## Per-family", ""]
    for fam in sorted(payload["per_family"]):
        lines += [
            f"### {fam}",
            "",
            "| Policy | Wins / N | p̂ | 95% Wilson CI |",
            "| --- | ---: | ---: | --- |",
        ]
        for pol in POLICIES:
            row = payload["per_family"][fam].get(pol)
            if not row:
                continue
            lines.append(
                f"| {pol} | {row['wins']} / {row['total']} | {row['p_hat']:.3f} "
                f"| [{row['ci_lo']:.3f}, {row['ci_hi']:.3f}] |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle-json",
        default=str(DATA_DIR / "oracle_gap.json"),
        help="oracle_gap.json source (rows list).",
    )
    parser.add_argument(
        "--json-out", default=str(DATA_DIR / "wilson_win_rates.json")
    )
    parser.add_argument("--md-out", default=str(DATA_DIR / "wilson_win_rates.md"))
    args = parser.parse_args()

    rows = load_oracle_rows(Path(args.oracle_json))
    agg = aggregate(rows)

    src_path = Path(args.oracle_json)
    try:
        src_label = str(src_path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        src_label = str(src_path)
    payload = {
        "meta": {
            "source": src_label,
            "z": Z_95,
            "ci_level": 0.95,
            "method": "wilson_score",
            "policies": list(POLICIES),
            "apps": list(APPS),
            "n_rows": len(rows),
        },
        **agg,
    }

    Path(args.json_out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    Path(args.md_out).write_text(render_md(payload).rstrip("\n") + "\n")

    overall_grasp = payload["overall"].get("GRASP", {})
    overall_popt = payload["overall"].get("POPT", {})
    print(
        f"[wilson-wins] n_rows={len(rows)} | GRASP {overall_grasp.get('wins')}/{overall_grasp.get('total')} "
        f"CI=[{overall_grasp.get('ci_lo')}, {overall_grasp.get('ci_hi')}] | "
        f"POPT {overall_popt.get('wins')}/{overall_popt.get('total')} "
        f"CI=[{overall_popt.get('ci_lo')}, {overall_popt.get('ci_hi')}] → "
        f"{args.md_out}"
    )


if __name__ == "__main__":
    main()
