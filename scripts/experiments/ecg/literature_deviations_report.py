#!/usr/bin/env python3
"""Literature-deviation inventory.

Why this exists
---------------
30 cells in ``wiki/data/literature_reproduction_summary.csv`` carry
``status=known_deviation``. They are *allowed* — the literature claim
they encode is known not to hold under our measurement setup — but
they are not *explained*. The paper has to defend each one. Today
they all happen to be flavours of ``POPT_GE_GRASP`` derived from
Balaji & Lucia HPCA 2021 §6.3, but as the corpus grows other claim
families will start failing too.

This script consolidates the deviations and assigns a **mechanism
label** to each by cross-referencing the lit-faith CSV (where the
actual GRASP/POPT/LRU/SRRIP miss rates live). The four mechanism
buckets are:

* ``popt_overhead_dominates`` — claim was ``POPT_GE_GRASP`` but
  measured POPT miss rate is strictly *higher* than GRASP by more
  than the claim tolerance. The most common GRASP-paper failure
  mode.
* ``within_extended_tolerance`` — claim's measured delta has the
  expected sign and magnitude < ``2 × tolerance_pct``. These are
  cells the reproduction marked as a deviation only because the
  tolerance was tight; loosening to 2× would re-classify them OK.
* ``policy_data_missing`` — the policy referenced by the claim has
  no matching row in lit-faith (e.g., synthetic policies like
  ``POPT_GE_GRASP`` that are computed claims, not real
  ``policy=*`` rows).
* ``unclassified`` — everything else; flagged for manual review.

Output
------
* ``wiki/data/literature_deviations.csv`` — one row per deviation
  with the citation, observed delta, popt_vs_grasp delta in the
  same cell (if available), and assigned mechanism label.
* ``wiki/data/literature_deviations.json`` — machine-readable
  summary: counts by mechanism, by graph, by app, and the full
  per-row table.
* ``wiki/data/literature_deviations.md`` — paper-ready markdown.

Usage
-----
    python3 -m scripts.experiments.ecg.literature_deviations_report
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]

GRAPH_FAMILY: dict[str, str] = {
    "email-Eu-core": "social",
    "web-Google": "web",
    "cit-Patents": "citation",
    "soc-pokec": "social",
    "soc-LiveJournal1": "social",
    "com-orkut": "social",
    "roadNet-CA": "road",
    "delaunay_n19": "mesh",
}

MECHANISM_ORDER = (
    "popt_overhead_dominates",
    "within_extended_tolerance",
    "policy_data_missing",
    "unclassified",
)


def _read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _build_miss_rate_index(lit_faith_rows: Iterable[dict]) -> dict:
    """Index miss_rate by (graph, app, l3_size, policy) for fast lookup."""
    idx: dict[tuple[str, str, str, str], float] = {}
    for r in lit_faith_rows:
        graph = r.get("graph", "")
        app = r.get("app") or r.get("benchmark", "")
        l3 = r.get("l3_size", "")
        pol = (r.get("policy") or "").strip()
        try:
            mr = float(r.get("miss_rate") or r.get("l3_miss_rate", "nan"))
        except ValueError:
            continue
        if not math.isfinite(mr):
            continue
        idx[(graph, app, l3, pol)] = mr
    return idx


def _classify(
    row: dict, mr_index: dict[tuple[str, str, str, str], float]
) -> tuple[str, float | None]:
    """Return (mechanism_label, popt_vs_grasp_delta_pp)."""
    graph = row.get("graph", "")
    app = row.get("app", "")
    l3 = row.get("l3_size", "")
    policy = (row.get("policy") or "").strip()
    try:
        delta_pct = float(row.get("delta_pct") or "nan")
    except ValueError:
        delta_pct = float("nan")
    try:
        tol = float(row.get("tolerance_pct") or "nan")
    except ValueError:
        tol = float("nan")

    grasp_mr = mr_index.get((graph, app, l3, "GRASP"))
    popt_mr = mr_index.get((graph, app, l3, "POPT"))
    popt_vs_grasp_pp: float | None = None
    if grasp_mr is not None and popt_mr is not None:
        popt_vs_grasp_pp = (popt_mr - grasp_mr) * 100.0

    # Computed/synthetic policy names that won't appear in lit-faith.
    if policy in {"POPT_GE_GRASP", "POPT_NEAR_GRASP_IF_BIG_GAP"}:
        # The claim is "POPT should be <= GRASP". We deviate when
        # measured POPT > GRASP by more than the tolerance.
        if popt_vs_grasp_pp is not None and popt_vs_grasp_pp > tol:
            return "popt_overhead_dominates", popt_vs_grasp_pp
        # If popt is actually <= grasp, the only reason this is flagged
        # must be the tightness of the test; bucket as extended tol.
        if (
            popt_vs_grasp_pp is not None
            and math.isfinite(tol)
            and abs(popt_vs_grasp_pp) <= 2.0 * tol
        ):
            return "within_extended_tolerance", popt_vs_grasp_pp
        if popt_vs_grasp_pp is None:
            return "policy_data_missing", None
        return "unclassified", popt_vs_grasp_pp

    # Real policy referenced (GRASP, POPT, etc.) but data missing.
    if (graph, app, l3, policy) not in mr_index:
        return "policy_data_missing", popt_vs_grasp_pp

    if math.isfinite(tol) and math.isfinite(delta_pct):
        if abs(delta_pct) <= 2.0 * tol:
            return "within_extended_tolerance", popt_vs_grasp_pp
    return "unclassified", popt_vs_grasp_pp


def _inventory(
    repro_rows: Iterable[dict], lit_faith_rows: Iterable[dict]
) -> list[dict]:
    mr_index = _build_miss_rate_index(lit_faith_rows)
    out: list[dict] = []
    for r in repro_rows:
        if (r.get("status") or "") != "known_deviation":
            continue
        graph = r.get("graph", "")
        mechanism, popt_vs_grasp_pp = _classify(r, mr_index)
        out.append({
            "citation": r.get("citation", ""),
            "graph": graph,
            "graph_family": GRAPH_FAMILY.get(graph, "unknown"),
            "app": r.get("app", ""),
            "l3_size": r.get("l3_size", ""),
            "policy": r.get("policy", ""),
            "expected_sign": r.get("expected_sign", ""),
            "tolerance_pct": r.get("tolerance_pct", ""),
            "delta_pct": r.get("delta_pct", ""),
            "popt_vs_grasp_pp": (
                "" if popt_vs_grasp_pp is None
                else f"{popt_vs_grasp_pp:.3f}"
            ),
            "mechanism": mechanism,
        })
    # Stable sort by (mechanism, graph, app, l3_size) so the table reads
    # by-mechanism.
    out.sort(key=lambda r: (
        MECHANISM_ORDER.index(r["mechanism"])
        if r["mechanism"] in MECHANISM_ORDER else len(MECHANISM_ORDER),
        r["graph"], r["app"], r["l3_size"],
    ))
    return out


def _summarize(records: list[dict]) -> dict:
    by_mech = Counter(r["mechanism"] for r in records)
    by_graph = Counter(r["graph"] for r in records)
    by_family = Counter(r["graph_family"] for r in records)
    by_app = Counter(r["app"] for r in records)
    by_policy = Counter(r["policy"] for r in records)
    # Cross-tab: mechanism x family
    cross: dict[tuple[str, str], int] = defaultdict(int)
    for r in records:
        cross[(r["mechanism"], r["graph_family"])] += 1
    return {
        "n_deviations": len(records),
        "by_mechanism": dict(by_mech.most_common()),
        "by_graph": dict(by_graph.most_common()),
        "by_family": dict(by_family.most_common()),
        "by_app": dict(by_app.most_common()),
        "by_policy": dict(by_policy.most_common()),
        "mechanism_family_cross_tab": {
            f"{mech}|{fam}": n for (mech, fam), n in sorted(cross.items())
        },
    }


def _write_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("")
        return
    fieldnames = list(records[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow(r)


def _write_json(summary: dict, records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "summary": summary,
        "deviations": records,
    }, indent=2, sort_keys=True) + "\n")


def _write_md(summary: dict, records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Literature deviations inventory")
    lines.append("")
    lines.append(
        "_Generated by `scripts/experiments/ecg/literature_deviations_report.py` "
        "from `wiki/data/literature_reproduction_summary.csv` "
        "and `wiki/data/literature_faithfulness_postfix.csv`._"
    )
    lines.append("")
    n = summary["n_deviations"]
    lines.append(
        f"**Total deviations:** {n} cells. Each is explained by a "
        "mechanism label so the paper can defend the KNOWN_DEVIATIONS "
        "table point-by-point."
    )
    lines.append("")

    lines.append("## Counts by mechanism")
    lines.append("")
    lines.append("| mechanism | count | share |")
    lines.append("|---|---:|---:|")
    den = max(1, n)
    for mech in MECHANISM_ORDER:
        c = summary["by_mechanism"].get(mech, 0)
        if c == 0:
            continue
        lines.append(f"| {mech} | {c} | {c/den*100:.1f}% |")
    # Any unknown buckets (defensive).
    for mech, c in summary["by_mechanism"].items():
        if mech not in MECHANISM_ORDER and c > 0:
            lines.append(f"| {mech} | {c} | {c/den*100:.1f}% |")
    lines.append("")

    lines.append("## Counts by graph family")
    lines.append("")
    lines.append("| family | count |")
    lines.append("|---|---:|")
    for fam, c in summary["by_family"].items():
        lines.append(f"| {fam} | {c} |")
    lines.append("")

    lines.append("## Counts by application")
    lines.append("")
    lines.append("| app | count |")
    lines.append("|---|---:|")
    for app, c in summary["by_app"].items():
        lines.append(f"| {app} | {c} |")
    lines.append("")

    lines.append("## Mechanism × family cross-tab")
    lines.append("")
    lines.append("| mechanism | family | count |")
    lines.append("|---|---|---:|")
    for key, c in summary["mechanism_family_cross_tab"].items():
        mech, fam = key.split("|", 1)
        if c == 0:
            continue
        lines.append(f"| {mech} | {fam} | {c} |")
    lines.append("")

    lines.append("## Per-cell deviations (full table)")
    lines.append("")
    lines.append(
        "| mechanism | citation | graph | app | L3 | policy | "
        "expected sign | tol % | observed Δ% | POPT−GRASP pp |"
    )
    lines.append("|---|---|---|---|---|---|---|---:|---:|---:|")
    for r in records:
        cite = r["citation"]
        if len(cite) > 50:
            cite = cite[:47] + "..."
        lines.append(
            f"| {r['mechanism']} | {cite} | {r['graph']} | {r['app']} | "
            f"{r['l3_size']} | {r['policy']} | {r['expected_sign']} | "
            f"{r['tolerance_pct']} | {r['delta_pct']} | "
            f"{r['popt_vs_grasp_pp']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repro-csv",
        type=Path,
        default=REPO_ROOT / "wiki" / "data" / "literature_reproduction_summary.csv",
    )
    parser.add_argument(
        "--lit-faith-csv",
        type=Path,
        default=REPO_ROOT / "wiki" / "data" / "literature_faithfulness_postfix.csv",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=REPO_ROOT / "wiki" / "data" / "literature_deviations.csv",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "wiki" / "data" / "literature_deviations.json",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=REPO_ROOT / "wiki" / "data" / "literature_deviations.md",
    )
    args = parser.parse_args()

    if not args.repro_csv.exists():
        raise SystemExit(
            f"missing repro CSV {args.repro_csv}; run `make lit-repro` first."
        )
    if not args.lit_faith_csv.exists():
        raise SystemExit(
            f"missing lit-faith CSV {args.lit_faith_csv}; "
            "run `make lit-faith` first."
        )

    repro_rows = _read_csv(args.repro_csv)
    lit_faith_rows = _read_csv(args.lit_faith_csv)
    records = _inventory(repro_rows, lit_faith_rows)
    summary = _summarize(records)
    _write_csv(records, args.csv_out)
    _write_json(summary, records, args.json_out)
    _write_md(summary, records, args.md_out)
    print(
        f"[lit-deviations] {summary['n_deviations']} deviations classified: "
        + ", ".join(
            f"{m}={c}" for m, c in summary["by_mechanism"].items() if c > 0
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
