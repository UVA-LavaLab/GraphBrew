#!/usr/bin/env python3
"""Regression budget: how much can each lit-faith cell drift before it
flips from ``ok`` / ``within_tolerance`` to ``disagree``?

The dashboard tells us *whether* we agree with the literature today. It
does not tell us *how close to the boundary* we are. A cell at margin
0.05 pp is one rebuild away from flipping the headline, whereas a cell
at margin 20 pp is genuinely robust.

This module loads the post-fix lit-faith JSON, re-derives the per-cell
margin by perturbing the observed delta in the adverse direction and
asking ``literature_faithfulness._classify`` where the status flips, and
emits aggregates so we can:

* Track the worst-case margin over time (Have we made the headline more
  fragile or more robust?)
* Identify the K most fragile cells for follow-up.
* Wire a pytest gate that fails if the minimum margin falls below a
  floor (default 0.5 pp).

The JSON shape is intentionally similar to the lit-faith summary so
``confidence_dashboard.py`` can pick it up without bespoke parsing.

CLI::

    python3 scripts/experiments/ecg/regression_budget.py \
        --lit-faith-json wiki/data/literature_faithfulness_postfix.json \
        --json-out wiki/data/regression_budget.json \
        --md-out   wiki/data/regression_budget.md \
        --csv-out  wiki/data/regression_budget.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

# Cells classified as "disagree" or worse get budget 0. "ok" and
# "within_tolerance" both get a positive budget. "insufficient_data" /
# "missing" / "known_deviation" are excluded from the headline
# distribution (their budget isn't meaningful).
INCLUDED_STATUSES = {"ok", "within_tolerance"}

# How far to step in the adverse direction during the binary search.
_PERTURB_MAX_PCT = 50.0
_PERTURB_STEP_PCT = 0.005  # 0.005pp resolution


def _load_lit_baselines() -> Any:
    """Dynamic-load literature_baselines (it's not on PYTHONPATH)."""
    here = REPO_ROOT / "scripts" / "experiments" / "ecg" / "literature_baselines.py"
    spec = importlib.util.spec_from_file_location("lit_baselines_local", here)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_lit_faith_classifier() -> Any:
    here = REPO_ROOT / "scripts" / "experiments" / "ecg" / "literature_faithfulness.py"
    spec = importlib.util.spec_from_file_location("lit_faith_local", here)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _adverse_direction(sign: str) -> float:
    """Return the sign of the perturbation that moves toward disagree.

    For sign='-' (must reduce miss_rate), the adverse direction is
    *increasing* delta (delta climbs toward 0 and beyond). For sign='+'
    it's the opposite. For sign='~' both directions are adverse so we
    explore both and take the smaller margin.
    """
    if sign == "-":
        return +1.0
    if sign == "+":
        return -1.0
    return +1.0  # sign='~' handled separately


def _margin_for(claim: Any, entry: dict[str, Any], classify) -> tuple[float, str]:
    """Return (margin_pp, kind) for an entry from the lit-faith JSON.

    Kind is one of {"cache_policy", "popt_ge_grasp",
    "popt_near_grasp_active", "popt_near_grasp_inactive"} to let the
    caller bucket cells by their gate type when reporting fragility.
    """
    policy = claim.policy
    tol = claim.tolerance_pct or 0.0

    if policy == "POPT_GE_GRASP":
        # Classification: ok iff diff_pct <= tolerance_pct where
        # diff_pct = (popt.miss_rate - grasp.miss_rate)*100 (positive
        # means POPT is worse). The JSON's delta_pct field for these
        # rows already carries that signed diff. Margin (adverse =
        # POPT gets worse) = tolerance - diff_pct.
        diff_pct = float(entry.get("delta_pct") or 0.0)
        margin = max(0.0, tol - diff_pct)
        return margin, "popt_ge_grasp"

    if policy == "POPT_GE_GRASP_GEOMEAN":
        # Corpus geomean gate (the authoritative POPT-vs-GRASP claim): ok iff
        # (popt_geomean - grasp_geomean)*100 <= tol. delta_pct carries that
        # signed geomean diff (positive = POPT geomean worse). Margin = how many
        # pp the POPT geomean can rise before it would flip to disagree.
        diff_pct = float(entry.get("delta_pct") or 0.0)
        return max(0.0, tol - diff_pct), "popt_ge_grasp_geomean"

    if policy == "POPT_NEAR_GRASP_IF_BIG_GAP":
        max_abs = claim.max_abs_delta_pct or 0.0
        # signed_delta is the signed gap (positive = POPT worse than
        # GRASP). When the GRASP-vs-LRU gain exceeds 10 pp the gate
        # fires; otherwise it's inactive.
        signed = float(entry.get("signed_delta_pct") or entry.get("delta_pct") or 0.0)
        grasp_gain = float(entry.get("grasp_gain_vs_lru_pct") or 0.0)
        active = grasp_gain > 10.0
        # Conservative budget: how close is signed_pp to the
        # disagree boundary (max_abs + tol). For inactive cells this
        # is a *what-if* lower bound on the margin should the gate
        # fire later.
        margin = max(0.0, (max_abs + tol) - signed)
        return margin, "popt_near_grasp_active" if active else "popt_near_grasp_inactive"

    # Cache-policy claim (LRU/SRRIP/GRASP/POPT).
    delta_pct = float(entry.get("delta_pct") or 0.0)
    sign = claim.expected_sign
    base_status = classify(claim, delta_pct)
    if base_status not in INCLUDED_STATUSES:
        return 0.0, "cache_policy"
    candidates: list[float] = []
    directions = (+1.0, -1.0) if sign == "~" else (_adverse_direction(sign),)
    for direction in directions:
        step = direction * _PERTURB_STEP_PCT
        cur = delta_pct
        margin = 0.0
        max_steps = int(_PERTURB_MAX_PCT / _PERTURB_STEP_PCT)
        for _ in range(max_steps):
            cur += step
            margin += abs(step)
            if classify(claim, cur) == "disagree":
                candidates.append(margin)
                break
        else:
            candidates.append(_PERTURB_MAX_PCT)
    return min(candidates), "cache_policy"


def compute(lit_faith_json: Path) -> dict[str, Any]:
    lit = _load_lit_baselines()
    faith_mod = _load_lit_faith_classifier()
    classify = faith_mod._classify

    data = json.loads(lit_faith_json.read_text())
    per_claim = data.get("per_claim") or data.get("claims") or []

    rows: list[dict[str, Any]] = []
    for entry in per_claim:
        status = entry.get("claim_status") or entry.get("status") or ""
        # Skip POPT_GE_GRASP-style relative rows — they live in a
        # different list in the JSON and need their own treatment. Their
        # absolute claim is bounded by tolerance_pct only.
        graph = entry["graph"]
        app = entry["app"]
        l3 = entry["l3_size"]
        policy = entry["policy"]
        delta_pct = entry.get("delta_pct")

        # Find the underlying claim object so we can re-evaluate.
        claim = None
        if policy == "POPT_GE_GRASP_GEOMEAN":
            claim = getattr(lit, "POPT_GE_GRASP_GEOMEAN_CLAIM", None)
        else:
            for c in lit.claims_for(graph, app, l3):
                if c.policy == policy:
                    claim = c
                    break
        if claim is None or delta_pct is None:
            continue

        if status in INCLUDED_STATUSES:
            margin, kind = _margin_for(claim, entry, classify)
        else:
            margin, kind = 0.0, "cache_policy" if policy in {"LRU", "SRRIP", "GRASP", "POPT"} else policy.lower()

        rows.append({
            "graph": graph,
            "app": app,
            "l3_size": l3,
            "policy": policy,
            "status": status,
            "delta_pct": float(delta_pct),
            "expected_sign": claim.expected_sign,
            "min_abs_delta_pct": claim.min_abs_delta_pct,
            "max_abs_delta_pct": claim.max_abs_delta_pct,
            "tolerance_pct": claim.tolerance_pct,
            "margin_pp": round(margin, 4),
            "claim_kind": kind,
            "citation": claim.citation,
        })

    included = [r for r in rows if r["status"] in INCLUDED_STATUSES]
    margins = [r["margin_pp"] for r in included]
    margins.sort()

    def _pct(p: float) -> float:
        if not margins:
            return 0.0
        idx = max(0, min(len(margins) - 1, int(math.floor(p * (len(margins) - 1)))))
        return margins[idx]

    # Bucket the headline summary by claim kind, so a future regression
    # tells us whether the brittleness is in a cache-policy claim
    # (real concern) or a not-yet-active POPT_NEAR_GRASP gate (less
    # concerning).
    by_kind: dict[str, list[float]] = {}
    for r in included:
        by_kind.setdefault(r["claim_kind"], []).append(r["margin_pp"])
    kind_summary = {
        k: {
            "n": len(vs),
            "min_pp": round(min(vs), 4),
            "median_pp": round(sorted(vs)[len(vs) // 2], 4),
        }
        for k, vs in sorted(by_kind.items())
    }

    summary = {
        "cells_total": len(rows),
        "cells_in_distribution": len(included),
        "min_margin_pp": round(min(margins), 4) if margins else 0.0,
        "p10_margin_pp": round(_pct(0.10), 4),
        "median_margin_pp": round(_pct(0.50), 4),
        "p90_margin_pp": round(_pct(0.90), 4),
        "max_margin_pp": round(max(margins), 4) if margins else 0.0,
        "by_kind": kind_summary,
    }
    # Identify the K most fragile cells globally and per-kind.
    fragile = sorted(included, key=lambda r: r["margin_pp"])[:10]
    cache_only = [r for r in included if r["claim_kind"] == "cache_policy"]
    fragile_cache = sorted(cache_only, key=lambda r: r["margin_pp"])[:10]
    return {
        "summary": summary,
        "fragile_cells": fragile,
        "fragile_cache_policy_cells": fragile_cache,
        "per_cell": rows,
    }


def emit_markdown(out: dict[str, Any]) -> str:
    s = out["summary"]
    md: list[str] = []
    md.append("# Regression budget — distance to disagree")
    md.append("")
    md.append("Per-cell distance (pp) the observed Δ must drift in the")
    md.append("adverse direction before the lit-faith status flips to")
    md.append("`disagree`. Larger margins = more robust headline.")
    md.append("")
    md.append("## Headline")
    md.append("")
    md.append(f"- Cells in distribution: **{s['cells_in_distribution']}** "
              f"(of {s['cells_total']} total claim rows)")
    md.append(f"- Minimum margin: **{s['min_margin_pp']:.3f} pp**")
    md.append(f"- p10 margin: {s['p10_margin_pp']:.3f} pp")
    md.append(f"- Median margin: {s['median_margin_pp']:.3f} pp")
    md.append(f"- p90 margin: {s['p90_margin_pp']:.3f} pp")
    md.append(f"- Max margin: {s['max_margin_pp']:.3f} pp")
    md.append("")
    md.append("### By claim kind")
    md.append("")
    md.append("| kind | n cells | min margin (pp) | median margin (pp) |")
    md.append("|---|---:|---:|---:|")
    for k, v in s["by_kind"].items():
        md.append(f"| {k} | {v['n']} | {v['min_pp']:.3f} | {v['median_pp']:.3f} |")
    md.append("")
    md.append("Cache-policy claims (LRU/SRRIP/GRASP/POPT individually) are")
    md.append("the **primary** load-bearing checks. POPT_GE_GRASP enforces")
    md.append("the P-OPT-vs-GRASP ordering and POPT_NEAR_GRASP_IF_BIG_GAP")
    md.append("only fires when GRASP outperforms LRU by >10 pp; the latter")
    md.append("is reported in this table as a *what-if* margin should the")
    md.append("gate fire later.")
    md.append("")
    md.append("## 10 most fragile cache-policy cells")
    md.append("")
    md.append("| graph | app | l3 | policy | status | Δ (pp) | margin (pp) | citation |")
    md.append("|---|---|---|---|---|---:|---:|---|")
    for r in out["fragile_cache_policy_cells"]:
        md.append(
            f"| {r['graph']} | {r['app']} | {r['l3_size']} | {r['policy']} "
            f"| {r['status']} | {r['delta_pct']:+.3f} | {r['margin_pp']:.3f} "
            f"| {r['citation']} |"
        )
    md.append("")
    md.append("## 10 most fragile cells (any kind)")
    md.append("")
    md.append("| graph | app | l3 | policy | kind | status | Δ (pp) | margin (pp) |")
    md.append("|---|---|---|---|---|---|---:|---:|")
    for r in out["fragile_cells"]:
        md.append(
            f"| {r['graph']} | {r['app']} | {r['l3_size']} | {r['policy']} "
            f"| {r['claim_kind']} | {r['status']} | {r['delta_pct']:+.3f} "
            f"| {r['margin_pp']:.3f} |"
        )
    md.append("")
    return "\n".join(md) + "\n"


def emit_csv(out: dict[str, Any], path: Path) -> None:
    rows = out["per_cell"]
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    default_json = REPO_ROOT / "wiki" / "data" / "literature_faithfulness_postfix.json"
    p.add_argument("--lit-faith-json", type=Path, default=default_json)
    p.add_argument("--json-out", type=Path,
                   default=REPO_ROOT / "wiki" / "data" / "regression_budget.json")
    p.add_argument("--md-out", type=Path,
                   default=REPO_ROOT / "wiki" / "data" / "regression_budget.md")
    p.add_argument("--csv-out", type=Path,
                   default=REPO_ROOT / "wiki" / "data" / "regression_budget.csv")
    args = p.parse_args()

    out = compute(args.lit_faith_json)
    args.json_out.write_text(json.dumps(out, indent=2) + "\n")
    args.md_out.write_text(emit_markdown(out).rstrip("\n") + "\n")
    emit_csv(out, args.csv_out)
    s = out["summary"]
    print(
        f"regression budget: {s['cells_in_distribution']} cells, "
        f"min={s['min_margin_pp']:.3f}pp, "
        f"median={s['median_margin_pp']:.3f}pp, "
        f"max={s['max_margin_pp']:.3f}pp"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
