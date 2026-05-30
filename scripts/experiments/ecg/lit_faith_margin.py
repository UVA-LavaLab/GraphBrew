#!/usr/bin/env python3
"""Literature-faithfulness margin audit.

Each row of ``literature_faithfulness_postfix.json`` carries an
``expected_sign`` and one or more numerical bounds (``tolerance_pct``,
``min_abs_delta_pct``, ``max_abs_delta_pct``). A row is classified
``ok`` / ``within_tolerance`` / ``known_deviation`` / ``disagree`` by
comparing observed ``delta_pct`` against these bounds.

This module computes the *distance to the nearest disagree boundary*
for every claim — the **margin** — and locks distribution invariants so
the corpus cannot silently drift into the fragile-cell regime where
small jitter would flip ``ok`` rows into ``disagree`` rows.

For each claim we compute ``margin_pp`` = the smallest signed distance
in percentage points from observed ``delta_pct`` to any disagree
boundary implied by the claim's bounds. Positive margins mean the
claim is in the comfortable region; negative margins mean the row is
already in ``disagree`` (should not happen for ``ok`` rows).

Per-sign margin formulas (mirrors ``literature_faithfulness._classify``):

* ``-`` (POPT ≤ GRASP):
    - upper boundary: ``delta_pct`` must be ≤ ``-min_abs + tol`` (if
      ``min_abs`` is set) else ``≤ tol``.
    - magnitude boundary: ``|delta_pct|`` ≤ ``max_abs + tol`` (if set).
* ``+`` (POPT > GRASP):
    - lower boundary: ``delta_pct ≥ min_abs - tol`` (if set) else
      ``≥ -tol``.
    - upper magnitude boundary: ``≤ max_abs + tol`` (if set).
* ``~`` (POPT ≈ GRASP):
    - magnitude boundary: ``|delta_pct| ≤ max_abs + tol`` (if set);
      otherwise the claim is unbounded and margin is +inf.

Emits ``wiki/data/lit_faith_margin.{json,md,csv}`` and powers the
``LIT-Mar`` confidence gate that locks:

* ``median_margin_pp`` ≥ a comfortable floor.
* fragile-cell count (margin < 1pp) ≤ a small ceiling.
* zero negative margins among ``ok`` rows (invariant: classifier and
  audit must agree on which rows are inside the claim envelope).

CLI::

    python3 -m scripts.experiments.ecg.lit_faith_margin \\
        --lit-faith-json wiki/data/literature_faithfulness_postfix.json \\
        --json-out wiki/data/lit_faith_margin.json \\
        --md-out   wiki/data/lit_faith_margin.md \\
        --csv-out  wiki/data/lit_faith_margin.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LIT_FAITH = REPO_ROOT / "wiki" / "data" / "literature_faithfulness_postfix.json"
DEFAULT_JSON_OUT = REPO_ROOT / "wiki" / "data" / "lit_faith_margin.json"
DEFAULT_MD_OUT = REPO_ROOT / "wiki" / "data" / "lit_faith_margin.md"
DEFAULT_CSV_OUT = REPO_ROOT / "wiki" / "data" / "lit_faith_margin.csv"

# Fragility threshold — cells with margin below this are "fragile":
# observed delta_pct is within FRAGILE_THRESHOLD_PP of flipping
# the claim's classification into ``disagree``.
FRAGILE_THRESHOLD_PP = 1.0


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


def _margins_for_claim(claim: dict[str, Any]) -> list[float]:
    """Return list of signed distances from delta_pct to each disagree boundary.

    A positive value means the observed delta_pct is *inside* the
    allowed region with that much room to spare. A negative value
    means it is past the boundary (i.e. classifier should call
    ``disagree``).
    """
    delta = _safe(claim.get("delta_pct"))
    if delta is None:
        return []
    sign = claim.get("expected_sign")
    tol = _safe(claim.get("tolerance_pct")) or 0.0
    min_abs = _safe(claim.get("min_abs_delta_pct"))
    max_abs = _safe(claim.get("max_abs_delta_pct"))
    margins: list[float] = []

    if sign == "-":
        # Boundary: delta_pct <= -(min_abs - tol)  or  delta_pct <= tol
        if min_abs is not None:
            upper = -(min_abs - tol)
            margins.append(upper - delta)
        else:
            margins.append(tol - delta)
        # Magnitude ceiling: |delta| <= max_abs + tol
        if max_abs is not None:
            margins.append((max_abs + tol) - abs(delta))
    elif sign == "+":
        # Boundary: delta_pct >= min_abs - tol  (or >= -tol)
        if min_abs is not None:
            lower = min_abs - tol
            margins.append(delta - lower)
        else:
            margins.append(delta - (-tol))
        if max_abs is not None:
            margins.append((max_abs + tol) - delta)
    else:  # sign == "~"
        if max_abs is not None:
            margins.append((max_abs + tol) - abs(delta))
        # If no max_abs, magnitude-only claim is unbounded; return empty.
    return margins


def _cell_margin(claim: dict[str, Any]) -> tuple[float | None, str | None]:
    """Return (margin_pp, binding_boundary_label) for a claim.

    ``margin_pp`` is the *minimum* of all per-boundary distances (the
    closest boundary wins). ``binding_boundary_label`` is a short tag
    like ``"sign_lower"`` / ``"magnitude_upper"`` describing which
    boundary is currently binding.

    Special case: the ``POPT_NEAR_GRASP_IF_BIG_GAP`` policy carries a
    *conditional* assertion that only fires when GRASP outperforms LRU
    by > 10 pp (the phase-transition regime). When the trigger is not
    met (the row carries a ``note`` field), the magnitude bound is
    vacuous and the margin is the distance to the trigger threshold
    itself (positive = how far below the 10 pp trigger we are).
    """
    delta = _safe(claim.get("delta_pct"))
    if delta is None:
        return None, None
    policy = claim.get("policy")
    grasp_gain = _safe(claim.get("grasp_gain_vs_lru_pct"))
    if policy == "POPT_NEAR_GRASP_IF_BIG_GAP":
        # Trigger margin: distance from grasp_gain_pp to the 10 pp threshold.
        # When grasp_gain <= 10, claim is dormant; report (10 - grasp_gain)
        # as the "headroom-before-claim-fires" margin and tag accordingly.
        if grasp_gain is not None and grasp_gain <= 10.0:
            return 10.0 - grasp_gain, "trigger_headroom"
        # In-regime: the classifier's actual rule is one-sided:
        # signed_pp <= max_abs + tol (POPT may outperform GRASP arbitrarily
        # without violating; it may only underperform by up to max_abs + tol).
        # Margin = (max_abs + tol) - signed_pp.
        signed = _safe(claim.get("signed_delta_pct"))
        if signed is None:
            signed = delta
        tol = _safe(claim.get("tolerance_pct")) or 0.0
        max_abs = _safe(claim.get("max_abs_delta_pct"))
        if max_abs is None:
            return None, "unbounded"
        margin = (max_abs + tol) - signed
        return margin, "near_grasp_upper"

    margins = _margins_for_claim(claim)
    if not margins:
        return None, "unbounded"
    # Build label/value pairs to identify binding boundary
    pairs: list[tuple[float, str]] = []
    sign = claim.get("expected_sign")
    tol = _safe(claim.get("tolerance_pct")) or 0.0
    min_abs = _safe(claim.get("min_abs_delta_pct"))
    max_abs = _safe(claim.get("max_abs_delta_pct"))

    if sign == "-":
        if min_abs is not None:
            pairs.append((-(min_abs - tol) - delta, "sign_upper_min_abs"))
        else:
            pairs.append((tol - delta, "sign_upper_tol"))
        if max_abs is not None:
            pairs.append(((max_abs + tol) - abs(delta), "magnitude_max_abs"))
    elif sign == "+":
        if min_abs is not None:
            pairs.append((delta - (min_abs - tol), "sign_lower_min_abs"))
        else:
            pairs.append((delta - (-tol), "sign_lower_tol"))
        if max_abs is not None:
            pairs.append(((max_abs + tol) - delta, "magnitude_max_abs"))
    else:
        if max_abs is not None:
            pairs.append(((max_abs + tol) - abs(delta), "magnitude_max_abs"))
    binding = min(pairs, key=lambda p: p[0])
    return binding[0], binding[1]


def _cell_key(claim: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(claim.get("graph", "")),
        str(claim.get("app", "")),
        str(claim.get("l3_size", "")),
        str(claim.get("policy", "")),
    )


def build_audit(lit_faith: dict[str, Any]) -> dict[str, Any]:
    per_claim = lit_faith.get("per_claim", [])
    rows: list[dict[str, Any]] = []
    margin_by_status: dict[str, list[float]] = defaultdict(list)
    fragile_rows: list[dict[str, Any]] = []
    negative_ok_rows: list[dict[str, Any]] = []
    unbounded_rows: list[dict[str, Any]] = []

    for claim in per_claim:
        margin, binding = _cell_margin(claim)
        graph, app, l3, policy = _cell_key(claim)
        status = str(claim.get("status", ""))
        row = {
            "graph": graph,
            "app": app,
            "l3_size": l3,
            "policy": policy,
            "expected_sign": claim.get("expected_sign"),
            "delta_pct": _safe(claim.get("delta_pct")),
            "tolerance_pct": _safe(claim.get("tolerance_pct")),
            "min_abs_delta_pct": _safe(claim.get("min_abs_delta_pct")),
            "max_abs_delta_pct": _safe(claim.get("max_abs_delta_pct")),
            "status": status,
            "margin_pp": margin,
            "binding_boundary": binding,
            "fragile": bool(
                margin is not None and margin < FRAGILE_THRESHOLD_PP
            ),
            "citation": claim.get("citation"),
        }
        rows.append(row)
        if margin is None:
            unbounded_rows.append(row)
            continue
        margin_by_status[status].append(margin)
        if row["fragile"]:
            fragile_rows.append(row)
        if status == "ok" and margin < 0:
            negative_ok_rows.append(row)

    all_margins = [
        r["margin_pp"]
        for r in rows
        if r["margin_pp"] is not None
    ]

    def _stats(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {
                "count": 0,
                "min": None,
                "median": None,
                "mean": None,
                "max": None,
                "p10": None,
                "p25": None,
                "p75": None,
                "p90": None,
            }
        sorted_v = sorted(values)
        def _q(p: float) -> float:
            if len(sorted_v) == 1:
                return sorted_v[0]
            k = (len(sorted_v) - 1) * p
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return sorted_v[int(k)]
            return sorted_v[f] * (c - k) + sorted_v[c] * (k - f)
        return {
            "count": len(values),
            "min": round(sorted_v[0], 4),
            "median": round(statistics.median(values), 4),
            "mean": round(statistics.fmean(values), 4),
            "max": round(sorted_v[-1], 4),
            "p10": round(_q(0.10), 4),
            "p25": round(_q(0.25), 4),
            "p75": round(_q(0.75), 4),
            "p90": round(_q(0.90), 4),
        }

    per_status_stats = {
        status: _stats(values) for status, values in margin_by_status.items()
    }

    # Bucket per family (heuristic: lookup from graph)
    GRAPH_FAMILY = {
        "soc-LiveJournal1": "social",
        "soc-pokec": "social",
        "com-orkut": "social",
        "email-Eu-core": "social",
        "cit-Patents": "citation",
        "web-Google": "web",
        "roadNet-CA": "road",
        "roadNet-PA": "road",
        "roadNet-TX": "road",
        "delaunay_n18": "mesh",
        "delaunay_n19": "mesh",
        "delaunay_n20": "mesh",
    }
    per_family: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        fam = GRAPH_FAMILY.get(r["graph"], "unknown")
        if r["margin_pp"] is not None:
            per_family[fam].append(r["margin_pp"])
    per_family_stats = {
        fam: _stats(values) for fam, values in per_family.items()
    }

    payload = {
        "schema_version": 1,
        "fragile_threshold_pp": FRAGILE_THRESHOLD_PP,
        "summary": {
            "claims_total": len(per_claim),
            "claims_with_margin": len(all_margins),
            "claims_unbounded": len(unbounded_rows),
            "fragile_count": len(fragile_rows),
            "negative_ok_count": len(negative_ok_rows),
            **_stats(all_margins),
        },
        "per_status_stats": per_status_stats,
        "per_family_stats": per_family_stats,
        "binding_boundary_counts": dict(
            Counter(
                r["binding_boundary"]
                for r in rows
                if r["binding_boundary"] is not None
            )
        ),
        "fragile_rows": sorted(
            fragile_rows,
            key=lambda r: (r["margin_pp"] if r["margin_pp"] is not None else 0.0),
        ),
        "negative_ok_rows": sorted(
            negative_ok_rows,
            key=lambda r: (r["margin_pp"] if r["margin_pp"] is not None else 0.0),
        ),
        "unbounded_rows_count_by_sign": dict(
            Counter(r["expected_sign"] for r in unbounded_rows)
        ),
        "rows": rows,
    }
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    s = payload["summary"]
    lines: list[str] = [
        "# Literature-faithfulness margin audit",
        "",
        "Per-claim distance to the nearest disagree boundary "
        f"(threshold for *fragile*: < {payload['fragile_threshold_pp']} pp).",
        "",
        "## Summary",
        "",
        f"- Total claims: **{s['claims_total']}**",
        f"- Claims with a bounded margin: **{s['claims_with_margin']}**",
        f"- Unbounded magnitude-only (`~` with no `max_abs`): "
        f"**{s['claims_unbounded']}**",
        f"- **Fragile** cells (< {payload['fragile_threshold_pp']} pp from "
        f"disagree boundary): **{s['fragile_count']}**",
        f"- `ok`-status cells with negative margin "
        "(classifier/audit disagreement): "
        f"**{s['negative_ok_count']}**",
        "",
        "### Margin distribution (pp)",
        "",
        "| stat | value |",
        "|---|---|",
        f"| min | {s.get('min')} |",
        f"| p10 | {s.get('p10')} |",
        f"| p25 | {s.get('p25')} |",
        f"| median | {s.get('median')} |",
        f"| mean | {s.get('mean')} |",
        f"| p75 | {s.get('p75')} |",
        f"| p90 | {s.get('p90')} |",
        f"| max | {s.get('max')} |",
        "",
        "## Per-status margin",
        "",
        "| status | count | min | median | mean | max |",
        "|---|---|---|---|---|---|",
    ]
    for status in sorted(payload["per_status_stats"]):
        st = payload["per_status_stats"][status]
        lines.append(
            f"| {status} | {st['count']} | {st['min']} | {st['median']} | "
            f"{st['mean']} | {st['max']} |"
        )

    lines.extend(
        [
            "",
            "## Per-family margin",
            "",
            "| family | count | min | median | mean | max |",
            "|---|---|---|---|---|---|",
        ]
    )
    for fam in sorted(payload["per_family_stats"]):
        fs = payload["per_family_stats"][fam]
        lines.append(
            f"| {fam} | {fs['count']} | {fs['min']} | {fs['median']} | "
            f"{fs['mean']} | {fs['max']} |"
        )

    lines.extend(
        [
            "",
            "## Binding-boundary breakdown",
            "",
            "Which side of the claim envelope is currently nearest "
            "(the boundary that defines each cell's margin):",
            "",
            "| boundary | count |",
            "|---|---|",
        ]
    )
    for boundary, count in sorted(
        payload["binding_boundary_counts"].items(),
        key=lambda kv: -kv[1],
    ):
        lines.append(f"| {boundary} | {count} |")

    fragile = payload["fragile_rows"]
    if fragile:
        lines.extend(
            [
                "",
                f"## Fragile cells "
                f"(margin < {payload['fragile_threshold_pp']} pp)",
                "",
                "| graph | app | L3 | policy | sign | Δ pp | margin pp | "
                "status | binding |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
        )
        for r in fragile:
            lines.append(
                f"| {r['graph']} | {r['app']} | {r['l3_size']} | "
                f"{r['policy']} | {r['expected_sign']} | {r['delta_pct']} | "
                f"{round(r['margin_pp'], 3)} | {r['status']} | "
                f"{r['binding_boundary']} |"
            )

    if payload["negative_ok_rows"]:
        lines.extend(
            [
                "",
                "## ⚠ classifier/audit disagreement "
                "(ok-status row with negative margin)",
                "",
                "| graph | app | L3 | policy | sign | Δ pp | margin pp | "
                "binding |",
                "|---|---|---|---|---|---|---|---|",
            ]
        )
        for r in payload["negative_ok_rows"]:
            lines.append(
                f"| {r['graph']} | {r['app']} | {r['l3_size']} | "
                f"{r['policy']} | {r['expected_sign']} | {r['delta_pct']} | "
                f"{round(r['margin_pp'], 3)} | {r['binding_boundary']} |"
            )

    return "\n".join(lines)


def render_csv(payload: dict[str, Any], path: Path) -> None:
    rows = payload["rows"]
    fields = [
        "graph",
        "app",
        "l3_size",
        "policy",
        "expected_sign",
        "delta_pct",
        "tolerance_pct",
        "min_abs_delta_pct",
        "max_abs_delta_pct",
        "status",
        "margin_pp",
        "binding_boundary",
        "fragile",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fields})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lit-faith-json",
        type=Path,
        default=DEFAULT_LIT_FAITH,
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

    s = payload["summary"]
    print(
        f"[lit-faith-margin] {s['claims_total']} claims; "
        f"{s['claims_with_margin']} bounded; "
        f"median margin = {s.get('median')} pp; "
        f"{s['fragile_count']} fragile (< "
        f"{payload['fragile_threshold_pp']} pp); "
        f"{s['negative_ok_count']} ok-with-negative-margin."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
