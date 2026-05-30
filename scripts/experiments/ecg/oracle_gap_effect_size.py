#!/usr/bin/env python3
"""Cliff's delta + Mann-Whitney U on raw oracle-gap distributions.

Why this exists
---------------
Wilson CIs (gate 36) and Cohen's h (gate 37) defend WIN-COUNT claims
("policy X wins Y/N cells"). But the actual *magnitude* of how much
worse one policy is per cell — captured by gap_pp = miss_rate −
oracle_miss_rate — is the thing readers actually care about.

This gate runs the two standard nonparametric tools on those raw
distributions:

* **Cliff's delta**: a robust, ranking-based effect size on
  ``P(X_a < X_b) − P(X_a > X_b)`` (range −1..+1).
* **Mann-Whitney U** + asymptotic p-value: a distribution-free
  test of stochastic dominance.

These complement Wilson/Cohen by being insensitive to outliers and
not requiring binomial collapse of the data.

Cliff's delta thresholds (Romano et al. 2006, Vargha-Delaney):

* |d| < 0.147 — negligible
* |d| >= 0.147 — small
* |d| >= 0.33  — medium
* |d| >= 0.474 — large

A negative ``d_a_vs_b`` means policy a has *smaller* gaps (better)
than policy b. We expect the headline POPT-on-pr and GRASP-on-cc
claims to show large NEGATIVE Cliff's delta vs the dumb policies
plus a Mann-Whitney p well below 0.05.

What we compute
---------------
For each app and ordered pair (a, b) with a != b:

* n_a, n_b
* median_a, median_b
* cliffs_delta_a_minus_b — negative ↔ a has smaller gaps
* magnitude — negligible | small | medium | large
* mannwhitney_u
* mannwhitney_p (two-sided asymptotic, no ties correction needed
  for our floor checks)

Output
------
* ``wiki/data/oracle_gap_effect_size.json``
* ``wiki/data/oracle_gap_effect_size.md``
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

NEGLIGIBLE = 0.147
SMALL = 0.147
MEDIUM = 0.33
LARGE = 0.474


def cliffs_delta(xs: list[float], ys: list[float]) -> float:
    """Cliff's delta = (#{x>y} − #{x<y}) / (n_x * n_y).

    Range: [−1, +1]. Negative ↔ xs stochastically smaller than ys.
    O(n_x * n_y) — fine for our cell counts (≤28 per group).
    """
    if not xs or not ys:
        return 0.0
    gt = 0
    lt = 0
    for x in xs:
        for y in ys:
            if x > y:
                gt += 1
            elif x < y:
                lt += 1
    return (gt - lt) / (len(xs) * len(ys))


def magnitude(d: float) -> str:
    ad = abs(d)
    if ad >= LARGE:
        return "large"
    if ad >= MEDIUM:
        return "medium"
    if ad >= SMALL:
        return "small"
    return "negligible"


def _erfc(x: float) -> float:
    return math.erfc(x)


def _normal_sf(z: float) -> float:
    """1 − Φ(z) using complementary error function."""
    return 0.5 * _erfc(z / math.sqrt(2.0))


def mannwhitney_u(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Two-sided Mann-Whitney U with asymptotic normal p-value.

    Returns (U_xy, p). Handles tied ranks by averaging. For our
    cell counts (~20–28 per group) the normal approximation is
    fine; we use it without ties correction since the only floors
    the gate checks are ``p < 0.05`` and ``p < 0.001`` (both
    extremely conservative for the effect sizes we expect to see).
    """
    if not xs or not ys:
        return 0.0, 1.0
    combined = sorted([(v, "x") for v in xs] + [(v, "y") for v in ys])
    # Average-rank assignment with tie handling
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-indexed
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1

    rank_x = sum(r for r, (_, lab) in zip(ranks, combined) if lab == "x")
    n_x, n_y = len(xs), len(ys)
    u_x = rank_x - n_x * (n_x + 1) / 2.0

    mu = n_x * n_y / 2.0
    sigma = math.sqrt(n_x * n_y * (n_x + n_y + 1) / 12.0)
    if sigma == 0.0:
        return u_x, 1.0
    z = abs(u_x - mu) / sigma
    p = 2.0 * _normal_sf(z)
    return u_x, p


def _gap(row: dict) -> float:
    return float(row["gap_pp"])


def load_oracle_rows(path: Path) -> list[dict]:
    blob = json.loads(path.read_text())
    if isinstance(blob, dict) and "rows" in blob:
        return blob["rows"]
    return blob


def aggregate(rows: list[dict]) -> dict:
    by_app_pol = defaultdict(list)
    for r in rows:
        by_app_pol[(r["app"], r["policy"])].append(_gap(r))

    per_app = {}
    for app in sorted({r["app"] for r in rows}):
        groups = {pol: by_app_pol.get((app, pol), []) for pol in POLICIES}
        groups = {pol: vals for pol, vals in groups.items() if vals}

        per_policy = {}
        for pol, vals in groups.items():
            vs = sorted(vals)
            n = len(vs)
            per_policy[pol] = {
                "n": n,
                "median": round(vs[n // 2], 4),
                "mean": round(sum(vs) / n, 4),
                "min": round(vs[0], 4),
                "max": round(vs[-1], 4),
            }

        comparisons = []
        for a in POLICIES:
            for b in POLICIES:
                if a == b or a not in groups or b not in groups:
                    continue
                xs, ys = groups[a], groups[b]
                d = cliffs_delta(xs, ys)
                u, p = mannwhitney_u(xs, ys)
                comparisons.append(
                    {
                        "a": a,
                        "b": b,
                        "n_a": len(xs),
                        "n_b": len(ys),
                        "cliffs_delta_a_minus_b": round(d, 4),
                        "magnitude": magnitude(d),
                        "mannwhitney_u": round(u, 2),
                        "mannwhitney_p": round(p, 6),
                        "stochastically_smaller": (
                            a if d < 0 else (b if d > 0 else "tie")
                        ),
                    }
                )
        per_app[app] = {"per_policy": per_policy, "comparisons": comparisons}

    large_neg = []
    for app, payload in per_app.items():
        for c in payload["comparisons"]:
            if c["magnitude"] == "large" and c["cliffs_delta_a_minus_b"] < 0:
                large_neg.append({"app": app, **c})
    large_neg.sort(key=lambda r: r["cliffs_delta_a_minus_b"])

    return {"per_app": per_app, "large_negative_deltas": large_neg}


def render_md(payload: dict) -> str:
    meta = payload["meta"]
    lines = [
        "# Cliff's delta + Mann-Whitney U on oracle-gap distributions",
        "",
        f"Source: `{meta['source']}` ({meta['n_rows']} rows).",
        "",
        "Cliff's delta thresholds (Romano et al. 2006): small ≥ 0.147, "
        "medium ≥ 0.33, large ≥ 0.474.",
        "",
        "Negative ``cliffs_delta_a_minus_b`` ↔ policy *a* has stochastically "
        "smaller gaps (i.e. is the better policy).",
        "",
        "## All large-effect dominance pairs (|d| ≥ 0.474, sorted by d asc)",
        "",
        "| App | Better (a) | Worse (b) | d (a−b) | MW p | n_a | n_b |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in payload["large_negative_deltas"]:
        lines.append(
            f"| {r['app']} | {r['a']} | {r['b']} | {r['cliffs_delta_a_minus_b']:.3f} "
            f"| {r['mannwhitney_p']:.2e} | {r['n_a']} | {r['n_b']} |"
        )

    lines += ["", "## Per-app distributions and pairwise tests", ""]
    for app in APPS:
        if app not in payload["per_app"]:
            continue
        lines += [
            f"### {app}",
            "",
            "Distribution (gap_pp):",
            "",
            "| Policy | n | median | mean | min | max |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for pol, stats in payload["per_app"][app]["per_policy"].items():
            lines.append(
                f"| {pol} | {stats['n']} | {stats['median']:.3f} "
                f"| {stats['mean']:.3f} | {stats['min']:.3f} | {stats['max']:.3f} |"
            )
        lines += [
            "",
            "Pairwise tests (Cliff's d, MW-U two-sided p):",
            "",
            "| a | b | d (a−b) | magnitude | MW p |",
            "| --- | --- | ---: | --- | ---: |",
        ]
        for c in payload["per_app"][app]["comparisons"]:
            lines.append(
                f"| {c['a']} | {c['b']} | {c['cliffs_delta_a_minus_b']:+.3f} "
                f"| {c['magnitude']} | {c['mannwhitney_p']:.2e} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle-json",
        default=str(DATA_DIR / "oracle_gap.json"),
    )
    parser.add_argument(
        "--json-out", default=str(DATA_DIR / "oracle_gap_effect_size.json")
    )
    parser.add_argument(
        "--md-out", default=str(DATA_DIR / "oracle_gap_effect_size.md")
    )
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
            "n_rows": len(rows),
            "policies": list(POLICIES),
            "apps": list(APPS),
            "cliffs_thresholds": {"small": SMALL, "medium": MEDIUM, "large": LARGE},
        },
        **agg,
    }

    Path(args.json_out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    Path(args.md_out).write_text(render_md(payload).rstrip("\n") + "\n")

    print(
        f"[gap-effect-size] n_rows={len(rows)} | "
        f"{len(payload['large_negative_deltas'])} large-effect dominance pairs → "
        f"{args.md_out}"
    )


if __name__ == "__main__":
    main()
