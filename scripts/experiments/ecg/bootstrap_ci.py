#!/usr/bin/env python3
"""Bootstrap confidence intervals on the paper's load-bearing claims.

The paper makes several numerical assertions of the form "POPT mean
oracle gap on road family is 3.47 pp, GRASP is 11.82 pp". With a
sample of n = 25 cells per (policy, family) bucket, a reviewer is
entitled to ask: what is the 95 % CI on those means, and does the
POPT < GRASP ordering survive resampling?

This script computes percentile bootstrap CIs (5000 resamples) on
every (policy, family) and (policy, regime) bucket in the oracle-gap
report, plus the per-family ΔPOPT = POPT − GRASP delta. It also runs
a sign-stability check: in what fraction of bootstrap resamples does
mean(POPT_road) < mean(GRASP_road)? If it is not ≈ 1.0, the claim is
not stable.

Inputs
------
* ``wiki/data/oracle_gap.json``       — per-cell rows used as source of truth.
* ``wiki/data/popt_vs_grasp_delta.json`` — per-cell ΔPOPT rows for the
  paired bootstrap.

Outputs
-------
* ``wiki/data/bootstrap_ci.json``  — CI tables + sign-stability fractions.
* ``wiki/data/bootstrap_ci.md``    — paper-ready markdown.

Determinism
-----------
Bootstrap RNG is seeded with ``--seed`` (default 1729). Same input +
same seed = byte-identical output.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WIKI = REPO_ROOT / "wiki" / "data"

DEFAULT_ORACLE_JSON = WIKI / "oracle_gap.json"
DEFAULT_DELTA_JSON = WIKI / "popt_vs_grasp_delta.json"

N_RESAMPLES_DEFAULT = 5000
CI_LEVEL_DEFAULT = 0.95


def _bootstrap_mean(values: list[float], n_resamples: int, rng: random.Random) -> list[float]:
    n = len(values)
    if n == 0:
        return []
    return [
        sum(rng.choice(values) for _ in range(n)) / n
        for _ in range(n_resamples)
    ]


def _ci(boot: list[float], level: float) -> tuple[float, float]:
    if not boot:
        return (float("nan"), float("nan"))
    sb = sorted(boot)
    alpha = (1.0 - level) / 2.0
    lo_idx = max(0, int(alpha * len(sb)))
    hi_idx = min(len(sb) - 1, int((1.0 - alpha) * len(sb)))
    return (sb[lo_idx], sb[hi_idx])


def _bucket(
    rows: list[dict],
    key_fn,
    n_resamples: int,
    level: float,
    rng: random.Random,
) -> dict:
    by_bucket: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        key = key_fn(r)
        if key is None:
            continue
        try:
            by_bucket[key].append(float(r["gap_pp"]))
        except (KeyError, ValueError, TypeError):
            continue
    out: dict[str, dict] = {}
    for k, vals in sorted(by_bucket.items()):
        if not vals:
            continue
        boot = _bootstrap_mean(vals, n_resamples, rng)
        lo, hi = _ci(boot, level)
        out[k] = {
            "n":         len(vals),
            "mean":      round(statistics.fmean(vals), 4),
            "median":    round(statistics.median(vals), 4),
            "ci_lo":     round(lo, 4),
            "ci_hi":     round(hi, 4),
            "ci_width":  round(hi - lo, 4),
            "ci_level":  level,
        }
    return out


def _paired_delta(
    delta_rows: list[dict],
    key_fn,
    n_resamples: int,
    level: float,
    rng: random.Random,
) -> dict:
    """Paired bootstrap on Δ = POPT − GRASP cells already produced by
    popt_vs_grasp_report. Negative mean = POPT wins."""
    by_bucket: dict[str, list[float]] = defaultdict(list)
    for r in delta_rows:
        key = key_fn(r)
        if key is None:
            continue
        try:
            by_bucket[key].append(float(r["delta_pp"]))
        except (KeyError, ValueError, TypeError):
            continue
    out: dict[str, dict] = {}
    for k, vals in sorted(by_bucket.items()):
        if not vals:
            continue
        boot = _bootstrap_mean(vals, n_resamples, rng)
        lo, hi = _ci(boot, level)
        sig = "+" if lo > 0 else ("-" if hi < 0 else "0")
        out[k] = {
            "n":              len(vals),
            "mean_delta":     round(statistics.fmean(vals), 4),
            "median_delta":   round(statistics.median(vals), 4),
            "ci_lo":          round(lo, 4),
            "ci_hi":          round(hi, 4),
            "ci_width":       round(hi - lo, 4),
            "ci_excludes_zero": lo > 0 or hi < 0,
            "sign":            sig,
        }
    return out


def _sign_stability(
    rows: list[dict],
    pol_a: str,
    pol_b: str,
    family: str,
    n_resamples: int,
    rng: random.Random,
) -> dict:
    a_vals = [
        float(r["gap_pp"]) for r in rows
        if r.get("policy") == pol_a and r.get("family") == family
    ]
    b_vals = [
        float(r["gap_pp"]) for r in rows
        if r.get("policy") == pol_b and r.get("family") == family
    ]
    if not a_vals or not b_vals:
        return {
            "policy_a": pol_a, "policy_b": pol_b, "family": family,
            "n_a": len(a_vals), "n_b": len(b_vals),
            "frac_a_lt_b": None,
        }
    n_lt = 0
    for _ in range(n_resamples):
        a_mean = sum(rng.choice(a_vals) for _ in range(len(a_vals))) / len(a_vals)
        b_mean = sum(rng.choice(b_vals) for _ in range(len(b_vals))) / len(b_vals)
        if a_mean < b_mean:
            n_lt += 1
    return {
        "policy_a":   pol_a,
        "policy_b":   pol_b,
        "family":     family,
        "n_a":        len(a_vals),
        "n_b":        len(b_vals),
        "mean_a":     round(statistics.fmean(a_vals), 4),
        "mean_b":     round(statistics.fmean(b_vals), 4),
        "frac_a_lt_b": round(n_lt / n_resamples, 4),
    }


def _emit_md(out: dict, md_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Bootstrap confidence intervals (load-bearing claims)\n")
    lines.append(
        "_Generated by `scripts/experiments/ecg/bootstrap_ci.py`. "
        f"Each CI is computed from {out['meta']['n_resamples']} percentile "
        f"bootstrap resamples at level {out['meta']['ci_level']:.2f}. "
        "Seed-deterministic — same inputs + same seed = byte-identical output.\n"
    )

    lines.append("## Per (policy × family) oracle-gap mean CIs\n")
    lines.append("| policy / family | n | mean (pp) | 95 % CI | width (pp) |")
    lines.append("|---|---:|---:|---|---:|")
    for k, v in out["oracle_gap_by_policy_family"].items():
        lines.append(
            f"| `{k}` | {v['n']} | {v['mean']:.3f} | "
            f"[{v['ci_lo']:.3f}, {v['ci_hi']:.3f}] | {v['ci_width']:.3f} |"
        )

    lines.append("\n## Per (policy × L3 regime) oracle-gap mean CIs\n")
    lines.append("| policy / regime | n | mean (pp) | 95 % CI | width (pp) |")
    lines.append("|---|---:|---:|---|---:|")
    for k, v in out["oracle_gap_by_policy_regime"].items():
        lines.append(
            f"| `{k}` | {v['n']} | {v['mean']:.3f} | "
            f"[{v['ci_lo']:.3f}, {v['ci_hi']:.3f}] | {v['ci_width']:.3f} |"
        )

    lines.append("\n## ΔPOPT−GRASP per family (paired bootstrap)\n")
    lines.append(
        "Negative mean = POPT beats GRASP. "
        "`ci_excludes_zero` flags claims that survive resampling.\n"
    )
    lines.append("| family | n | mean Δ (pp) | 95 % CI | excludes 0 | sign |")
    lines.append("|---|---:|---:|---|:---:|:---:|")
    for k, v in out["popt_minus_grasp_by_family"].items():
        excl = "✅" if v["ci_excludes_zero"] else "❌"
        lines.append(
            f"| `{k}` | {v['n']} | {v['mean_delta']:+.3f} | "
            f"[{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}] | {excl} | `{v['sign']}` |"
        )

    lines.append("\n## Sign-stability of headline ordering claims\n")
    lines.append(
        "How often does the per-bootstrap-sample mean ordering "
        "match the headline claim? 1.0 = always; 0.5 = no signal.\n"
    )
    lines.append("| claim | n_a | n_b | mean_a (pp) | mean_b (pp) | P(a < b) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for s in out["sign_stability"]:
        if s["frac_a_lt_b"] is None:
            continue
        lines.append(
            f"| `{s['policy_a']} < {s['policy_b']} on {s['family']}` | "
            f"{s['n_a']} | {s['n_b']} | "
            f"{s['mean_a']:.3f} | {s['mean_b']:.3f} | {s['frac_a_lt_b']:.4f} |"
        )

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--oracle-json", type=Path, default=DEFAULT_ORACLE_JSON)
    p.add_argument("--delta-json",  type=Path, default=DEFAULT_DELTA_JSON)
    p.add_argument("--json-out",    type=Path, default=WIKI / "bootstrap_ci.json")
    p.add_argument("--md-out",      type=Path, default=WIKI / "bootstrap_ci.md")
    p.add_argument("--seed",        type=int, default=1729)
    p.add_argument("--n-resamples", type=int, default=N_RESAMPLES_DEFAULT)
    p.add_argument("--ci-level",    type=float, default=CI_LEVEL_DEFAULT)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = _parse_args(argv)

    oracle = json.loads(ns.oracle_json.read_text())
    rows = oracle.get("rows", [])
    if not rows:
        raise SystemExit(
            f"[bootstrap-ci] {ns.oracle_json} has no `rows` — regenerate with `make lit-oracle-gap`"
        )

    delta_doc = json.loads(ns.delta_json.read_text())
    delta_rows = delta_doc.get("cells", delta_doc.get("rows", []))
    if not delta_rows:
        raise SystemExit(
            f"[bootstrap-ci] {ns.delta_json} has no `cells` — regenerate with `make lit-popt-vs-grasp`"
        )

    rng = random.Random(ns.seed)

    by_pf = _bucket(
        rows,
        key_fn=lambda r: f"{r.get('policy')}/{r.get('family')}" if r.get('policy') and r.get('family') else None,
        n_resamples=ns.n_resamples,
        level=ns.ci_level,
        rng=rng,
    )

    by_pr = _bucket(
        rows,
        key_fn=lambda r: f"{r.get('policy')}/{r.get('regime')}" if r.get('policy') and r.get('regime') else None,
        n_resamples=ns.n_resamples,
        level=ns.ci_level,
        rng=rng,
    )

    delta_by_family = _paired_delta(
        delta_rows,
        key_fn=lambda r: r.get("graph_family") or r.get("family"),
        n_resamples=ns.n_resamples,
        level=ns.ci_level,
        rng=rng,
    )

    sign_claims = [
        ("POPT", "GRASP", "road"),
        ("POPT", "GRASP", "social"),
        ("POPT", "GRASP", "mesh"),
        ("POPT", "GRASP", "citation"),
        ("POPT", "GRASP", "web"),
        ("POPT", "LRU",   "social"),
        ("GRASP", "LRU",  "social"),
    ]
    sign_stability = [
        _sign_stability(rows, a, b, f, ns.n_resamples, rng)
        for (a, b, f) in sign_claims
    ]

    out = {
        "meta": {
            "n_resamples": ns.n_resamples,
            "ci_level":    ns.ci_level,
            "seed":        ns.seed,
        },
        "oracle_gap_by_policy_family": by_pf,
        "oracle_gap_by_policy_regime": by_pr,
        "popt_minus_grasp_by_family":  delta_by_family,
        "sign_stability":              sign_stability,
    }

    ns.json_out.parent.mkdir(parents=True, exist_ok=True)
    ns.json_out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    _emit_md(out, ns.md_out)

    md_path = Path(ns.md_out).resolve()
    try:
        md_display = md_path.relative_to(REPO_ROOT)
    except ValueError:
        md_display = md_path
    print(
        f"[bootstrap-ci] {len(by_pf)} (policy, family) buckets; "
        f"{len(by_pr)} (policy, regime) buckets; "
        f"{len(delta_by_family)} family Δ-buckets; "
        f"{len(sign_stability)} sign-stability claims → {md_display}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
