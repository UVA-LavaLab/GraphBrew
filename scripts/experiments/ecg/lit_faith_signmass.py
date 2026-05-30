#!/usr/bin/env python3
"""Literature-faithfulness sign-mass concentration audit.

The lit-faith comparator tells us whether each cell's observed
``delta_pct`` is *inside the claim envelope*. The LIT-Mar gate then
measures the *distance* from the observation to the disagree
boundary. But neither tells us whether the literature's *sign claims*
actually show measurable signal in our corpus — a row with
``expected_sign="-"`` is "correctly signed" if observed
``delta_pct`` < 0, and we want most of those rows to in fact be
on the negative side with non-trivial magnitude (real signal, not
just statistical wobble at zero).

This module slices the lit-faith corpus by
(expected_sign, policy) bucket and computes, per bucket:

* **correctly_signed_fraction** — for ``-`` claims the fraction of
  ok rows with ``delta_pct < 0`` (or ≤ 0 for POPT_GE_GRASP which can
  legitimately tie); for ``+`` claims fraction with ``delta_pct > 0``;
  ``~`` claims are not sign claims and are reported but not locked.
* **median_delta_pct / mean_delta_pct** — central tendency of observed
  effect.
* **binomial sign-test p-value** — probability of observing this many
  correctly-signed cells under the null "true sign is 50/50". A
  low p-value (e.g. < 0.001) is evidence the sign claim isn't noise.
* **wilson 95 % lower bound** on correctly_signed_fraction — a
  small-sample-safe lower bound so a single graph contributing a
  handful of cells doesn't fake a "100 %" signal.

LIT-Sig locks per-bucket invariants so:

* every (sign, policy) bucket has a minimum count of usable ok rows.
* the correctly-signed fraction stays above a per-bucket floor.
* the per-bucket median ``delta_pct`` keeps its expected sign and a
  minimum magnitude.
* the binomial sign test stays below an alpha threshold (sign-mass
  is statistically distinguishable from a coin flip).

Emits ``wiki/data/lit_faith_signmass.{json,md,csv}``.

CLI::

    python3 -m scripts.experiments.ecg.lit_faith_signmass \\
        --lit-faith-json wiki/data/literature_faithfulness_postfix.json \\
        --json-out wiki/data/lit_faith_signmass.json \\
        --md-out   wiki/data/lit_faith_signmass.md \\
        --csv-out  wiki/data/lit_faith_signmass.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LIT_FAITH = REPO_ROOT / "wiki" / "data" / "literature_faithfulness_postfix.json"
DEFAULT_JSON_OUT = REPO_ROOT / "wiki" / "data" / "lit_faith_signmass.json"
DEFAULT_MD_OUT = REPO_ROOT / "wiki" / "data" / "lit_faith_signmass.md"
DEFAULT_CSV_OUT = REPO_ROOT / "wiki" / "data" / "lit_faith_signmass.csv"


def _safe(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _binom_log_choose(n: int, k: int) -> float:
    """log(C(n, k)) via lgamma — avoids overflow for large n."""
    if k < 0 or k > n:
        return -math.inf
    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
    )


def _binom_pmf(n: int, k: int, p: float) -> float:
    if k < 0 or k > n:
        return 0.0
    if p == 0.0:
        return 1.0 if k == 0 else 0.0
    if p == 1.0:
        return 1.0 if k == n else 0.0
    log_pmf = (
        _binom_log_choose(n, k)
        + k * math.log(p)
        + (n - k) * math.log1p(-p)
    )
    return math.exp(log_pmf)


def _binom_two_sided_p(n: int, k: int, p0: float = 0.5) -> float:
    """Two-sided exact binomial p-value (Mead's method).

    Sums PMF values <= PMF(k) under H0. With p0=0.5 this is the
    standard symmetric sign-test p-value.
    """
    if n == 0:
        return 1.0
    pmf_k = _binom_pmf(n, k, p0)
    total = 0.0
    for j in range(n + 1):
        pmf_j = _binom_pmf(n, j, p0)
        if pmf_j <= pmf_k + 1e-12:
            total += pmf_j
    return min(1.0, total)


def _wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    """Wilson 95 % lower bound on a binomial proportion."""
    if n == 0:
        return 0.0
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(
        phat * (1.0 - phat) / n + z * z / (4 * n * n)
    )
    return max(0.0, centre - half)


def _classify_sign(delta_pct: float, expected_sign: str, policy: str) -> str:
    """Return 'correct' / 'tie' / 'wrong' for a single observation."""
    if expected_sign == "-":
        # POPT_GE_GRASP allows ties (delta == 0); others must be strictly negative.
        if policy == "POPT_GE_GRASP":
            if delta_pct < 0:
                return "correct"
            if delta_pct == 0:
                return "tie"
            return "wrong"
        if delta_pct < 0:
            return "correct"
        if delta_pct == 0:
            return "tie"
        return "wrong"
    if expected_sign == "+":
        if delta_pct > 0:
            return "correct"
        if delta_pct == 0:
            return "tie"
        return "wrong"
    # sign == "~"
    return "tie"  # magnitude-only; sign is irrelevant


def build_audit(lit_faith: dict[str, Any]) -> dict[str, Any]:
    per_claim = lit_faith.get("per_claim", [])
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for claim in per_claim:
        sign = str(claim.get("expected_sign", ""))
        policy = str(claim.get("policy", ""))
        delta = _safe(claim.get("delta_pct"))
        if delta is None:
            continue
        status = str(claim.get("status", ""))
        # For POPT_NEAR_GRASP_IF_BIG_GAP, the delta_pct field stores |signed|;
        # use signed_delta_pct when present so we look at the genuine sign.
        if policy == "POPT_NEAR_GRASP_IF_BIG_GAP":
            signed = _safe(claim.get("signed_delta_pct"))
            if signed is not None:
                delta = signed
        cell = {
            "graph": claim.get("graph"),
            "app": claim.get("app"),
            "l3_size": claim.get("l3_size"),
            "policy": policy,
            "expected_sign": sign,
            "delta_pct": delta,
            "status": status,
        }
        cell["sign_class"] = _classify_sign(delta, sign, policy)
        buckets[(sign, policy)].append(cell)

    bucket_summaries: list[dict[str, Any]] = []
    for (sign, policy), cells in sorted(buckets.items()):
        ok_cells = [c for c in cells if c["status"] == "ok"]
        all_n = len(cells)
        ok_n = len(ok_cells)
        if ok_n == 0:
            bucket_summaries.append({
                "expected_sign": sign,
                "policy": policy,
                "n_total": all_n,
                "n_ok": 0,
                "correctly_signed": 0,
                "tie": 0,
                "wrong": 0,
                "correctly_signed_fraction": None,
                "wilson_95_lower_bound": None,
                "binomial_sign_test_p": None,
                "median_delta_pct": None,
                "mean_delta_pct": None,
                "min_delta_pct": None,
                "max_delta_pct": None,
                "median_abs_delta_pct": None,
            })
            continue
        correct = sum(1 for c in ok_cells if c["sign_class"] == "correct")
        ties = sum(1 for c in ok_cells if c["sign_class"] == "tie")
        wrong = sum(1 for c in ok_cells if c["sign_class"] == "wrong")
        deltas = [c["delta_pct"] for c in ok_cells]
        # Treat ties as half-credit for sign concentration (standard for
        # POPT_GE_GRASP ties that legitimately satisfy the claim).
        directional = correct + 0.5 * ties
        directional_n = correct + ties + wrong
        fraction = (
            directional / directional_n if directional_n > 0 else None
        )
        # Binomial sign test using strict-sign counts only (ties excluded),
        # which is the standard non-parametric formulation.
        strict_n = correct + wrong
        sign_test_p = (
            _binom_two_sided_p(strict_n, correct) if strict_n > 0 else None
        )
        wilson_lb = (
            _wilson_lower(round(directional), directional_n)
            if directional_n > 0
            else None
        )
        bucket_summaries.append({
            "expected_sign": sign,
            "policy": policy,
            "n_total": all_n,
            "n_ok": ok_n,
            "correctly_signed": correct,
            "tie": ties,
            "wrong": wrong,
            "correctly_signed_fraction": (
                round(fraction, 6) if fraction is not None else None
            ),
            "wilson_95_lower_bound": (
                round(wilson_lb, 6) if wilson_lb is not None else None
            ),
            "binomial_sign_test_p": (
                round(sign_test_p, 6) if sign_test_p is not None else None
            ),
            "median_delta_pct": round(statistics.median(deltas), 4),
            "mean_delta_pct": round(statistics.fmean(deltas), 4),
            "min_delta_pct": round(min(deltas), 4),
            "max_delta_pct": round(max(deltas), 4),
            "median_abs_delta_pct": round(
                statistics.median(abs(d) for d in deltas), 4
            ),
        })

    payload = {
        "schema_version": 1,
        "summary": {
            "buckets_total": len(bucket_summaries),
            "buckets_with_ok_rows": sum(
                1 for b in bucket_summaries if b["n_ok"] > 0
            ),
            "claims_total": sum(b["n_total"] for b in bucket_summaries),
            "ok_rows_total": sum(b["n_ok"] for b in bucket_summaries),
        },
        "buckets": bucket_summaries,
        "rows": [
            {
                **c,
            }
            for cells in buckets.values()
            for c in cells
        ],
    }
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    s = payload["summary"]
    lines: list[str] = [
        "# Literature-faithfulness sign-mass audit",
        "",
        "For each (expected_sign × policy) bucket: how often the observed "
        "delta_pct has the literature's claimed sign, with binomial "
        "sign-test p-value and Wilson 95 % lower bound on the "
        "correctly-signed fraction.",
        "",
        "## Summary",
        "",
        f"- Total claims (with computable delta_pct): "
        f"**{s['claims_total']}**",
        f"- ok-status rows across all buckets: **{s['ok_rows_total']}**",
        f"- Buckets (sign × policy): **{s['buckets_total']}**",
        f"- Buckets with at least one ok row: "
        f"**{s['buckets_with_ok_rows']}**",
        "",
        "## Per-bucket sign-mass",
        "",
        "Sign-classes: correct (delta has the expected sign), tie "
        "(delta == 0 or expected_sign is `~`), wrong (delta has the "
        "opposite sign). The binomial p-value uses only strict-sign "
        "counts (ties excluded).",
        "",
        "| sign | policy | n_ok | correct | ties | wrong | frac | "
        "wilson 95 LB | binom p | median Δpp | mean Δpp |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for b in payload["buckets"]:
        frac = (
            f"{b['correctly_signed_fraction']:.3f}"
            if b["correctly_signed_fraction"] is not None
            else "—"
        )
        wlb = (
            f"{b['wilson_95_lower_bound']:.3f}"
            if b["wilson_95_lower_bound"] is not None
            else "—"
        )
        pv = (
            f"{b['binomial_sign_test_p']:.4f}"
            if b["binomial_sign_test_p"] is not None
            else "—"
        )
        lines.append(
            f"| {b['expected_sign']} | {b['policy']} | {b['n_ok']} | "
            f"{b['correctly_signed']} | {b['tie']} | {b['wrong']} | "
            f"{frac} | {wlb} | {pv} | {b['median_delta_pct']} | "
            f"{b['mean_delta_pct']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "* `-` × GRASP / `-` × POPT — these are the *load-bearing* "
            "sign claims (the policy beats LRU). They should show "
            "strongly negative median delta_pct and a binomial sign-test "
            "p-value below 0.05 (often well below). The Wilson 95 % "
            "lower bound floor is sample-size dependent: 13 cells caps "
            "the lower bound around 0.77; 7 cells around 0.65.",
            "* `-` × POPT_GE_GRASP — claim that POPT ≥ GRASP (delta "
            "is POPT − GRASP). Many cells legitimately tie at 0 "
            "(both policies saturate at the same value); the gate "
            "treats ties as half-credit. We still expect Wilson LB "
            "≥ 0.50 (better than coin-flip).",
            "* `~` × SRRIP / `~` × POPT_NEAR_GRASP_IF_BIG_GAP — magnitude "
            "claims; sign is not asserted and these rows are not locked "
            "by the LIT-Sig gate (only reported for reference).",
        ]
    )
    return "\n".join(lines)


def render_csv(payload: dict[str, Any], path: Path) -> None:
    fields = [
        "expected_sign",
        "policy",
        "n_total",
        "n_ok",
        "correctly_signed",
        "tie",
        "wrong",
        "correctly_signed_fraction",
        "wilson_95_lower_bound",
        "binomial_sign_test_p",
        "median_delta_pct",
        "mean_delta_pct",
        "min_delta_pct",
        "max_delta_pct",
        "median_abs_delta_pct",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for b in payload["buckets"]:
            writer.writerow({k: b.get(k) for k in fields})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lit-faith-json", type=Path, default=DEFAULT_LIT_FAITH
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV_OUT)
    args = parser.parse_args(argv)

    lit_faith = json.loads(args.lit_faith_json.read_text())
    payload = build_audit(lit_faith)

    args.json_out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    args.md_out.write_text(render_markdown(payload).rstrip("\n") + "\n")
    render_csv(payload, args.csv_out)

    print(
        f"[lit-faith-signmass] {payload['summary']['claims_total']} claims; "
        f"{payload['summary']['buckets_with_ok_rows']}/"
        f"{payload['summary']['buckets_total']} buckets with ok rows."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
