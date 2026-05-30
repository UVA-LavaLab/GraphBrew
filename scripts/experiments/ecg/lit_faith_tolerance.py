#!/usr/bin/env python3
"""Literature-faithfulness tolerance-calibration audit.

Each ``LiteratureClaim`` in ``literature_baselines.py`` carries a
``tolerance_pct`` slack — additive pp budget that widens the comparator
bounds. That budget is a calibration knob, and like any knob it can
drift:

* **Over-permissive** — bounds so wide that no realistic regression
  could flip the verdict. The gate then offers false confidence.
* **Fragile** — the observed |delta_pct| is sitting right at the
  disagree boundary, so a single noisy regen can flip the verdict
  and trigger a CI red. These are the cells most likely to break
  next.

This audit answers, for every claim that the comparator actually
asserts (i.e. fires a real bound), the question:

    "How many pp could |delta_pct| move in the wrong direction
     before the verdict flips from ok → disagree?"

We call that value the **slack** (pp). Smaller = more fragile.
For each "ok" or "within_tolerance" row that fires a real assertion
(POPT_NEAR_GRASP_IF_BIG_GAP non-triggered rows are excluded since
they carry no budget), we compute:

    slack_pp        = distance to the disagree boundary in pp
    fragile         = slack_pp < FRAGILE_SLACK_PP
    fragile_status  = "fragile" / "comfortable" / "very_comfortable"

and roll up to (policy × expected_sign × app) buckets, plus a
corpus-wide histogram and a top-N most-fragile list.

The LIT-Tol gate (gate 226) then locks:
* corpus-wide median slack ≥ floor
* fragile fraction ≤ ceiling
* every audited row has a non-NaN, finite slack
* no audited row has negative slack (would mean the comparator's
  classify branch and our slack formula disagree → bug)
* per-policy minimum-slack floors

Emits ``wiki/data/lit_faith_tolerance.{json,md,csv}``.
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
DEFAULT_JSON_OUT = REPO_ROOT / "wiki" / "data" / "lit_faith_tolerance.json"
DEFAULT_MD_OUT = REPO_ROOT / "wiki" / "data" / "lit_faith_tolerance.md"
DEFAULT_CSV_OUT = REPO_ROOT / "wiki" / "data" / "lit_faith_tolerance.csv"

FRAGILE_SLACK_PP = 1.0  # slack < 1.0pp → "fragile"
COMFORTABLE_SLACK_PP = 5.0  # slack >= 5.0pp → "very_comfortable"
NOT_TRIGGERED_NOTE = "not in phase-transition regime; assertion not triggered"


def _compute_slack(row: dict[str, Any]) -> tuple[float | None, str]:
    """Return (slack_pp, audit_status).

    audit_status ∈ {audited, not_triggered, missing_data, deviation,
                     disagree}; only `audited` rows contribute to the
    budget distribution.
    """
    status = row.get("status")
    policy = row.get("policy")
    sign = row.get("expected_sign")
    tol = row.get("tolerance_pct")
    delta = row.get("delta_pct")
    min_abs = row.get("min_abs_delta_pct")
    max_abs = row.get("max_abs_delta_pct")
    note = row.get("note") or ""

    if status in ("missing", "insufficient_data"):
        return None, "missing_data"
    if status == "known_deviation":
        return None, "deviation"
    if status == "disagree":
        return None, "disagree"

    if policy == "POPT_NEAR_GRASP_IF_BIG_GAP":
        # Only contributes if the assertion actually fired.
        if NOT_TRIGGERED_NOTE in note:
            return None, "not_triggered"
        signed = row.get("signed_delta_pct")
        if signed is None or max_abs is None or tol is None:
            return None, "missing_data"
        # Failure mode: POPT too much worse than GRASP, i.e.
        # signed_pp > max_abs + tol. Slack = max_abs + tol - signed_pp.
        return float(max_abs + tol - signed), "audited"

    if tol is None or delta is None:
        return None, "missing_data"

    if sign == "-":
        if min_abs is not None:
            # boundary: delta > -(min_abs - tol)  →  disagree
            return float(-(min_abs - tol) - delta), "audited"
        # boundary: delta > tol  →  disagree
        return float(tol - delta), "audited"
    if sign == "+":
        if min_abs is not None:
            # boundary: delta < min_abs - tol  →  disagree
            return float(delta - (min_abs - tol)), "audited"
        # boundary: delta < -tol  →  disagree
        return float(delta - (-tol)), "audited"
    # sign == "~" : magnitude only
    if max_abs is not None:
        return float((max_abs + tol) - abs(delta)), "audited"
    return float(tol - abs(delta)), "audited"


def _fragile_bucket(slack: float) -> str:
    if slack < FRAGILE_SLACK_PP:
        return "fragile"
    if slack >= COMFORTABLE_SLACK_PP:
        return "very_comfortable"
    return "comfortable"


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)


def build_audit(lit_faith: dict[str, Any]) -> dict[str, Any]:
    rows = lit_faith.get("per_claim", [])

    per_row: list[dict[str, Any]] = []
    for r in rows:
        slack, audit_status = _compute_slack(r)
        rec: dict[str, Any] = {
            "graph": r["graph"],
            "app": r["app"],
            "l3_size": r["l3_size"],
            "policy": r["policy"],
            "expected_sign": r["expected_sign"],
            "citation": r["citation"],
            "status": r["status"],
            "tolerance_pct": r.get("tolerance_pct"),
            "delta_pct": r.get("delta_pct"),
            "signed_delta_pct": r.get("signed_delta_pct"),
            "min_abs_delta_pct": r.get("min_abs_delta_pct"),
            "max_abs_delta_pct": r.get("max_abs_delta_pct"),
            "audit_status": audit_status,
            "slack_pp": (
                round(slack, 4) if slack is not None and not math.isnan(slack)
                else None
            ),
            "fragile_bucket": (
                _fragile_bucket(slack) if slack is not None else None
            ),
        }
        per_row.append(rec)

    audited = [r for r in per_row if r["audit_status"] == "audited"]
    slacks = [r["slack_pp"] for r in audited]

    fragile_rows = [r for r in audited if r["fragile_bucket"] == "fragile"]
    comfortable_rows = [
        r for r in audited if r["fragile_bucket"] == "comfortable"
    ]
    very_comfortable_rows = [
        r for r in audited if r["fragile_bucket"] == "very_comfortable"
    ]
    negative_slack = [r for r in audited if r["slack_pp"] < 0]

    # Per-policy aggregates
    by_policy: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[float]] = defaultdict(list)
    for r in audited:
        grouped[r["policy"]].append(r["slack_pp"])
    for pol, vals in grouped.items():
        by_policy[pol] = {
            "n": len(vals),
            "min_slack_pp": round(min(vals), 4),
            "median_slack_pp": round(statistics.median(vals), 4),
            "p10_slack_pp": round(_percentile(vals, 0.10), 4),
            "p90_slack_pp": round(_percentile(vals, 0.90), 4),
            "max_slack_pp": round(max(vals), 4),
            "fragile_count": sum(1 for v in vals if v < FRAGILE_SLACK_PP),
        }

    # Per-app aggregates
    by_app: dict[str, dict[str, Any]] = {}
    app_grouped: dict[str, list[float]] = defaultdict(list)
    for r in audited:
        app_grouped[r["app"]].append(r["slack_pp"])
    for app, vals in app_grouped.items():
        by_app[app] = {
            "n": len(vals),
            "min_slack_pp": round(min(vals), 4),
            "median_slack_pp": round(statistics.median(vals), 4),
            "fragile_count": sum(1 for v in vals if v < FRAGILE_SLACK_PP),
        }

    # Histogram across the corpus
    histogram_edges = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    histogram_counts = [0] * (len(histogram_edges) + 1)
    for v in slacks:
        placed = False
        for i, edge in enumerate(histogram_edges):
            if v < edge:
                histogram_counts[i] += 1
                placed = True
                break
        if not placed:
            histogram_counts[-1] += 1
    histogram = []
    prev = "-∞"
    for edge, cnt in zip(histogram_edges, histogram_counts[:-1]):
        histogram.append({"bin": f"[{prev}, {edge})", "count": cnt})
        prev = str(edge)
    histogram.append({"bin": f"[{prev}, +∞)", "count": histogram_counts[-1]})

    # Top-N most fragile (sorted by slack ascending)
    top_fragile = sorted(
        audited, key=lambda r: r["slack_pp"]
    )[:15]

    # Coverage by audit_status
    audit_status_counts = Counter(r["audit_status"] for r in per_row)

    payload = {
        "schema_version": 1,
        "summary": {
            "total_rows": len(per_row),
            "audited_rows": len(audited),
            "fragile_rows": len(fragile_rows),
            "comfortable_rows": len(comfortable_rows),
            "very_comfortable_rows": len(very_comfortable_rows),
            "negative_slack_rows": len(negative_slack),
            "median_slack_pp": (
                round(statistics.median(slacks), 4) if slacks else None
            ),
            "min_slack_pp": (
                round(min(slacks), 4) if slacks else None
            ),
            "max_slack_pp": (
                round(max(slacks), 4) if slacks else None
            ),
            "p10_slack_pp": (
                round(_percentile(slacks, 0.10), 4) if slacks else None
            ),
            "p90_slack_pp": (
                round(_percentile(slacks, 0.90), 4) if slacks else None
            ),
            "fragile_fraction": (
                round(len(fragile_rows) / len(audited), 4) if audited else None
            ),
            "fragile_threshold_pp": FRAGILE_SLACK_PP,
            "comfortable_threshold_pp": COMFORTABLE_SLACK_PP,
            "audit_status_counts": dict(audit_status_counts),
        },
        "by_policy": by_policy,
        "by_app": by_app,
        "histogram": histogram,
        "top_fragile": top_fragile,
        "negative_slack": negative_slack,
        "per_row": per_row,
    }
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    s = payload["summary"]
    lines = [
        "# Literature-faithfulness tolerance-calibration audit",
        "",
        "For every literature claim whose comparator-asserted bound is "
        "actually exercised, this report computes the **slack** — how "
        "many pp the observed `|delta_pct|` could move in the wrong "
        "direction before the verdict flips from `ok` → `disagree`. "
        "Small slack means a future regen could easily break the gate; "
        "large slack means the bound is over-permissive.",
        "",
        "## Summary",
        "",
        f"- Total per_claim rows: **{s['total_rows']}**",
        f"- Audited rows (real assertion fired): **{s['audited_rows']}**",
        f"- Median slack: **{s['median_slack_pp']} pp**",
        f"- p10 slack: **{s['p10_slack_pp']} pp** · "
        f"p90 slack: **{s['p90_slack_pp']} pp**",
        f"- Min / max slack: **{s['min_slack_pp']}** / "
        f"**{s['max_slack_pp']}** pp",
        f"- Fragile rows "
        f"(slack < {s['fragile_threshold_pp']} pp): "
        f"**{s['fragile_rows']}** "
        f"({(s['fragile_fraction'] or 0) * 100:.1f}%)",
        f"- Very comfortable rows "
        f"(slack ≥ {s['comfortable_threshold_pp']} pp): "
        f"**{s['very_comfortable_rows']}**",
        f"- Negative-slack rows (audit bug if non-zero): "
        f"**{s['negative_slack_rows']}**",
        "",
        "## Audit-status breakdown",
        "",
        "| audit_status | count |",
        "|---|---:|",
    ]
    for k, v in sorted(s["audit_status_counts"].items()):
        lines.append(f"| `{k}` | {v} |")

    lines.extend(["", "## Slack histogram (audited rows)", ""])
    lines.append("| bin (pp) | count |")
    lines.append("|---|---:|")
    for h in payload["histogram"]:
        lines.append(f"| {h['bin']} | {h['count']} |")

    lines.extend(["", "## Per-policy slack distribution", ""])
    lines.append(
        "| policy | n | min | p10 | median | p90 | max | fragile |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for pol in sorted(payload["by_policy"]):
        b = payload["by_policy"][pol]
        lines.append(
            f"| {pol} | {b['n']} | {b['min_slack_pp']} | "
            f"{b['p10_slack_pp']} | {b['median_slack_pp']} | "
            f"{b['p90_slack_pp']} | {b['max_slack_pp']} | "
            f"{b['fragile_count']} |"
        )

    lines.extend(["", "## Per-app slack distribution", ""])
    lines.append("| app | n | min | median | fragile |")
    lines.append("|---|---:|---:|---:|---:|")
    for app in sorted(payload["by_app"]):
        b = payload["by_app"][app]
        lines.append(
            f"| {app} | {b['n']} | {b['min_slack_pp']} | "
            f"{b['median_slack_pp']} | {b['fragile_count']} |"
        )

    lines.extend([
        "",
        f"## Top-{len(payload['top_fragile'])} most fragile rows",
        "",
    ])
    lines.append(
        "| graph | app | L3 | policy | sign | tol | slack pp | status |"
    )
    lines.append("|---|---|---|---|---|---:|---:|---|")
    for r in payload["top_fragile"]:
        lines.append(
            f"| {r['graph']} | {r['app']} | {r['l3_size']} | "
            f"{r['policy']} | {r['expected_sign']} | "
            f"{r['tolerance_pct']} | {r['slack_pp']} | {r['status']} |"
        )

    if payload["negative_slack"]:
        lines.extend(["", "## ⚠ Negative-slack rows (audit bug)", ""])
        for r in payload["negative_slack"]:
            lines.append(
                f"- `{r['graph']}` / `{r['app']}` / `{r['l3_size']}` "
                f"/ `{r['policy']}` slack={r['slack_pp']} "
                f"(status={r['status']})"
            )

    return "\n".join(lines)


def render_csv(payload: dict[str, Any], path: Path) -> None:
    fields = [
        "graph", "app", "l3_size", "policy", "expected_sign",
        "status", "audit_status", "tolerance_pct", "delta_pct",
        "signed_delta_pct", "slack_pp", "fragile_bucket",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in payload["per_row"]:
            writer.writerow(r)


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

    s = payload["summary"]
    print(
        f"[lit-faith-tolerance] {s['audited_rows']} audited rows, "
        f"median slack {s['median_slack_pp']} pp, "
        f"fragile {s['fragile_rows']} "
        f"({(s['fragile_fraction'] or 0)*100:.1f}%), "
        f"negative-slack {s['negative_slack_rows']}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
