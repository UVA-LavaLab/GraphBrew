#!/usr/bin/env python3
"""Per-(family × app) POPT-vs-GRASP bootstrap CIs.

Why this exists
---------------
The family-level `bootstrap_ci.py` reports things like
"POPT < GRASP on road  P = 0.976" but does not break this down
by app. A reviewer asking "is POPT's road win carried by every
kernel or driven by sssp alone?" cannot be answered without a
(family × app) cut.

This script paired-bootstraps Δ = gap(POPT) − gap(GRASP) within
each (family, app) cell that has BOTH policies present, and
reports per-cell `P(POPT < GRASP)` plus a 95% CI on the mean Δ.

Output
------
* ``wiki/data/popt_vs_grasp_by_family_app.json``
* ``wiki/data/popt_vs_grasp_by_family_app.md``
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "wiki" / "data"

DEFAULT_ORACLE_JSON = DATA_DIR / "oracle_gap.json"
DEFAULT_JSON_OUT = DATA_DIR / "popt_vs_grasp_by_family_app.json"
DEFAULT_MD_OUT = DATA_DIR / "popt_vs_grasp_by_family_app.md"

DEFAULT_N_RESAMPLES = 2000
DEFAULT_SEED = 1729
CI_LEVEL = 0.95
N_PAIRED_FLOOR = 3


def _load_rows(path: Path) -> list[dict]:
    raw = json.loads(path.read_text())
    out = []
    for r in raw.get("rows", []):
        try:
            r = {**r, "gap_pp": float(r["gap_pp"])}
            out.append(r)
        except (ValueError, KeyError):
            continue
    return out


def _paired_deltas(rows: list[dict], family: str, app: str) -> list[float]:
    """Per-cell ΔPOPT−GRASP for matched (graph, l3_size) keys."""
    a_by_cell, b_by_cell = {}, {}
    for r in rows:
        if r["family"] != family or r["app"] != app:
            continue
        cell = (r["graph"], r["l3_size"])
        if r["policy"] == "POPT":
            a_by_cell[cell] = r["gap_pp"]
        elif r["policy"] == "GRASP":
            b_by_cell[cell] = r["gap_pp"]
    return [a_by_cell[c] - b_by_cell[c] for c in a_by_cell if c in b_by_cell]


def _bootstrap(deltas: list[float], rng: random.Random, n_res: int) -> dict:
    if not deltas:
        return {
            "n_paired":  0, "p_popt_lt_grasp": None, "mean_delta": None,
            "ci_lo":     None, "ci_hi": None,
        }
    n = len(deltas)
    means: list[float] = []
    n_neg = 0
    for _ in range(n_res):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        m = sum(sample) / n
        means.append(m)
        if m < 0:
            n_neg += 1
    means.sort()
    alpha = (1.0 - CI_LEVEL) / 2.0
    return {
        "n_paired":         n,
        "p_popt_lt_grasp":  round(n_neg / n_res, 4),
        "mean_delta":       round(sum(deltas) / n, 4),
        "ci_lo":            round(means[int(alpha * n_res)], 4),
        "ci_hi":            round(means[int((1.0 - alpha) * n_res) - 1], 4),
    }


def _aggregate(rows: list[dict], n_res: int, seed: int) -> dict:
    rng = random.Random(seed)
    families = sorted({r["family"] for r in rows})
    apps = sorted({r["app"] for r in rows})

    per_cell: dict[str, dict] = {}
    coverage = {"cells_with_data": 0, "cells_skipped_insufficient": 0}
    for f in families:
        for a in apps:
            deltas = _paired_deltas(rows, f, a)
            r = _bootstrap(deltas, rng, n_res)
            per_cell[f"{f}/{a}"] = r
            if r["n_paired"] >= N_PAIRED_FLOOR:
                coverage["cells_with_data"] += 1
            elif r["n_paired"] > 0:
                coverage["cells_skipped_insufficient"] += 1

    return {
        "meta": {
            "n_resamples":      n_res,
            "seed":             seed,
            "ci_level":         CI_LEVEL,
            "families":         families,
            "apps":             apps,
            "n_paired_floor":   N_PAIRED_FLOOR,
            **coverage,
        },
        "per_family_app": per_cell,
    }


def _write_md(doc: dict, path: Path) -> None:
    m = doc["meta"]
    lines = [
        "# POPT vs GRASP — per-(family × app) bootstrap CIs",
        "",
        f"_Resamples: {m['n_resamples']}, seed: {m['seed']}, "
        f"CI level: {m['ci_level']}._",
        f"_Cells with data: {m['cells_with_data']}; "
        f"skipped (n < {m['n_paired_floor']}): "
        f"{m['cells_skipped_insufficient']}._",
        "",
        "`P(POPT < GRASP)` = bootstrap fraction with mean Δ < 0.",
        "",
        "| family | app | n paired | mean Δ (pp) | 95% CI | P(POPT<GRASP) |",
        "|---|---|---:|---:|:---|---:|",
    ]
    for f in m["families"]:
        for a in m["apps"]:
            r = doc["per_family_app"][f"{f}/{a}"]
            n = r["n_paired"]
            if n == 0:
                continue
            mean = f"{r['mean_delta']}" if r["mean_delta"] is not None else "—"
            ci = (
                f"[{r['ci_lo']}, {r['ci_hi']}]"
                if r["ci_lo"] is not None else "—"
            )
            p = f"{r['p_popt_lt_grasp']}" if r["p_popt_lt_grasp"] is not None else "—"
            lines.append(f"| {f} | {a} | {n} | {mean} | {ci} | {p} |")
    path.write_text("\n".join(lines) + "\n")


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
    n_cells = doc["meta"]["cells_with_data"]
    print(
        f"[popt-vs-grasp-by-fa] {n_cells} (family,app) cells with paired "
        f"data; resamples={args.n_resamples} → {args.md_out}"
    )


if __name__ == "__main__":
    main()
