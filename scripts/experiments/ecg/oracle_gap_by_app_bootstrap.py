#!/usr/bin/env python3
"""Bootstrap CIs on per-kernel oracle-gap rank claims.

Why this exists
---------------
``oracle_gap_by_app`` reports per-(policy, app) means as point
estimates: pr→POPT 0.100 pp, cc→GRASP 0.640 pp, sssp catastrophic
for GRASP at 7.106 pp. Reviewers asking "is that statistically
real?" deserve confidence intervals.

This script bootstraps **paired ΔPOPT−other** across the cells of
each app and reports, per app and per comparison policy, the
fraction of resamples in which POPT beats the other policy
(``P(Δ < 0)``). The same is computed for GRASP-as-baseline so the
cc→GRASP and bc→SRRIP-but-GRASP-by-win-count subtleties get
explicit CIs.

Paired bootstrap
----------------
For each app, we identify cells (graph × L3-size) for which BOTH
policies in a comparison have a gap row, compute Δ per cell, then
resample those Δs with replacement. The fraction with mean Δ < 0
gives ``P(policy_a < policy_b)``.

Output
------
* ``wiki/data/oracle_gap_by_app_bootstrap.json``
* ``wiki/data/oracle_gap_by_app_bootstrap.md``
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "wiki" / "data"

DEFAULT_ORACLE_JSON = DATA_DIR / "oracle_gap.json"
DEFAULT_JSON_OUT = DATA_DIR / "oracle_gap_by_app_bootstrap.json"
DEFAULT_MD_OUT = DATA_DIR / "oracle_gap_by_app_bootstrap.md"

POLICIES = ("LRU", "SRRIP", "GRASP", "POPT")
APPS = ("pr", "bc", "cc", "bfs", "sssp")

DEFAULT_N_RESAMPLES = 2000
DEFAULT_SEED = 1729
CI_LEVEL = 0.95


def _load_rows(path: Path) -> list[dict]:
    raw = json.loads(path.read_text())
    rows = raw["rows"]
    out = []
    for r in rows:
        try:
            r = {
                **r,
                "gap_pp": float(r["gap_pp"]),
            }
            out.append(r)
        except (ValueError, KeyError):
            continue
    return out


def _paired_deltas(rows: list[dict], app: str, a: str, b: str) -> list[float]:
    """Return list of per-cell Δ = gap_a − gap_b for cells where both
    policies appear. Cells are keyed by (graph, l3_size)."""
    by_cell_a, by_cell_b = {}, {}
    for r in rows:
        if r["app"] != app:
            continue
        cell = (r["graph"], r["l3_size"])
        if r["policy"] == a:
            by_cell_a[cell] = r["gap_pp"]
        elif r["policy"] == b:
            by_cell_b[cell] = r["gap_pp"]
    return [by_cell_a[c] - by_cell_b[c] for c in by_cell_a if c in by_cell_b]


def _bootstrap_mean_negative_prob(
    deltas: list[float], rng: random.Random, n_resamples: int
) -> dict:
    """Return P(mean Δ < 0) and the (lo, hi) CI on the mean across
    n_resamples paired bootstrap samples."""
    if not deltas:
        return {
            "n_paired": 0,
            "p_a_lt_b": None,
            "mean_delta": None,
            "ci_lo": None,
            "ci_hi": None,
        }
    n = len(deltas)
    means = []
    n_neg = 0
    for _ in range(n_resamples):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        m = sum(sample) / n
        means.append(m)
        if m < 0:
            n_neg += 1
    means.sort()
    alpha = (1.0 - CI_LEVEL) / 2.0
    lo_i = int(alpha * n_resamples)
    hi_i = int((1.0 - alpha) * n_resamples) - 1
    return {
        "n_paired":   n,
        "p_a_lt_b":   round(n_neg / n_resamples, 4),
        "mean_delta": round(sum(deltas) / n, 4),
        "ci_lo":      round(means[lo_i], 4),
        "ci_hi":      round(means[hi_i], 4),
    }


def _aggregate(rows: list[dict], n_resamples: int, seed: int) -> dict:
    rng = random.Random(seed)
    per_app_pairs: dict[str, dict[str, dict]] = {}
    apps_seen = sorted({r["app"] for r in rows}) or list(APPS)

    for app in apps_seen:
        pairs: dict[str, dict] = {}
        for a in POLICIES:
            for b in POLICIES:
                if a == b:
                    continue
                deltas = _paired_deltas(rows, app, a, b)
                pairs[f"{a}_vs_{b}"] = _bootstrap_mean_negative_prob(
                    deltas, rng, n_resamples,
                )
        per_app_pairs[app] = pairs

    return {
        "meta": {
            "n_resamples":  n_resamples,
            "seed":         seed,
            "ci_level":     CI_LEVEL,
            "apps":         apps_seen,
            "policies":     list(POLICIES),
            "n_total_rows": len(rows),
        },
        "per_app_pairs": per_app_pairs,
    }


def _write_md(doc: dict, path: Path) -> None:
    lines = [
        "# Per-kernel oracle-gap bootstrap CIs",
        "",
        "_Bootstrapped paired Δ = gap(policy_a) − gap(policy_b) per app._",
        f"_Resamples: {doc['meta']['n_resamples']}, seed: "
        f"{doc['meta']['seed']}, CI level: {doc['meta']['ci_level']}._",
        "",
        "`P(a<b)` = fraction of bootstrap means in which policy *a* "
        "has a smaller (better) oracle gap than *b*.",
        "",
    ]
    for app in doc["meta"]["apps"]:
        lines += [
            f"## App: `{app}`",
            "",
            "| comparison | n paired | mean Δ (pp) | 95% CI | P(a<b) |",
            "|---|---:|---:|:---|---:|",
        ]
        pairs = doc["per_app_pairs"][app]
        for key in sorted(pairs.keys()):
            r = pairs[key]
            ci = "—"
            if r["ci_lo"] is not None:
                ci = f"[{r['ci_lo']}, {r['ci_hi']}]"
            mean = "—" if r["mean_delta"] is None else f"{r['mean_delta']}"
            p = "—" if r["p_a_lt_b"] is None else f"{r['p_a_lt_b']}"
            lines.append(
                f"| {key.replace('_vs_', ' < ')} | "
                f"{r['n_paired']} | {mean} | {ci} | {p} |"
            )
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-json", type=Path, default=DEFAULT_ORACLE_JSON)
    ap.add_argument("--json-out",    type=Path, default=DEFAULT_JSON_OUT)
    ap.add_argument("--md-out",      type=Path, default=DEFAULT_MD_OUT)
    ap.add_argument("--n-resamples", type=int, default=DEFAULT_N_RESAMPLES)
    ap.add_argument("--seed",        type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    rows = _load_rows(args.oracle_json)
    doc = _aggregate(rows, args.n_resamples, args.seed)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    _write_md(doc, args.md_out)
    n_apps = len(doc["meta"]["apps"])
    n_pairs = sum(len(v) for v in doc["per_app_pairs"].values())
    print(
        f"[oracle-by-app-boot] {n_apps} apps, {n_pairs} comparisons, "
        f"resamples={args.n_resamples} → {args.md_out}"
    )


if __name__ == "__main__":
    main()
