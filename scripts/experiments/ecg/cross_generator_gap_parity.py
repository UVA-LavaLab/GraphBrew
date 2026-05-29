#!/usr/bin/env python3
"""Cross-generator gap_pp parity (gate 54).

Three load-bearing aggregators expose the same underlying per-(app,
policy, L3) gap_pp data through different schemas:

  - wiki/data/oracle_gap.json
      raw per (graph, app, policy, L3) rows. Source of truth.
  - wiki/data/oracle_gap_auc.json
      per (app, policy) trajectory averaged across graphs.
  - wiki/data/cache_sensitivity_slope.json
      same trajectory, decorated with per-octave slope/delta.

If any of these three drift apart — different averaging window,
different rounding, swapped sign, stale cache — the paper's
narrative breaks silently. This gate runs a value-by-value parity
check across all three.

Output: wiki/data/cross_generator_gap_parity.{json,md}
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WIKI_DATA = REPO_ROOT / "wiki" / "data"
PAPER_L3 = ("1MB", "4MB", "8MB")
TOLERANCE = 1e-3  # 1 thousandth of a percentage point


def _resolve_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_payload(
    oracle_path: Path, auc_path: Path, slope_path: Path
) -> dict:
    raw = json.loads(oracle_path.read_text())
    auc = json.loads(auc_path.read_text())
    slope = json.loads(slope_path.read_text())

    # 1. raw mean per (app, policy, L3) across all graphs in cell.
    raw_acc: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in raw["rows"]:
        if r["l3_size"] not in PAPER_L3:
            continue
        raw_acc[(r["app"], r["policy"], r["l3_size"])].append(
            float(r["gap_pp"])
        )
    raw_means = {
        k: round(statistics.mean(vs), 4) for k, vs in raw_acc.items()
    }
    raw_counts = {k: len(vs) for k, vs in raw_acc.items()}

    # 2. AUC trajectory values
    auc_vals: dict[tuple[str, str, str], float] = {}
    for app, app_blob in auc["per_app"].items():
        for pol, traj in app_blob["trajectory_by_policy"].items():
            for l3, val in traj.items():
                if l3 in PAPER_L3:
                    auc_vals[(app, pol, l3)] = float(val)

    # 3. cache-sensitivity-slope values
    slope_vals: dict[tuple[str, str, str], float] = {}
    for app, app_blob in slope["per_app"].items():
        for pol, blob in app_blob.items():
            for l3 in PAPER_L3:
                key = f"gap_at_{l3}"
                if key in blob:
                    slope_vals[(app, pol, l3)] = float(blob[key])
            # 4MB lives inside the octave records, not as a top-level key.
            for oct_ in blob.get("octaves", []):
                if oct_.get("from") in PAPER_L3:
                    slope_vals.setdefault(
                        (app, pol, oct_["from"]), float(oct_["gap_from"])
                    )
                if oct_.get("to") in PAPER_L3:
                    slope_vals.setdefault(
                        (app, pol, oct_["to"]), float(oct_["gap_to"])
                    )

    # 4. parity check: for every triple referenced by any source, all
    #    three values must agree to within TOLERANCE.
    keys = set(raw_means) | set(auc_vals) | set(slope_vals)
    mismatches = []
    cells = []
    for k in sorted(keys):
        raw_v = raw_means.get(k)
        auc_v = auc_vals.get(k)
        slope_v = slope_vals.get(k)
        present = [v for v in (raw_v, auc_v, slope_v) if v is not None]
        if not present:
            continue
        rng = max(present) - min(present)
        agree = rng <= TOLERANCE and all(
            v is not None for v in (raw_v, auc_v, slope_v)
        )
        rec = {
            "app": k[0],
            "policy": k[1],
            "l3_size": k[2],
            "raw_mean_gap_pp": raw_v,
            "auc_trajectory_gap_pp": auc_v,
            "slope_gap_pp": slope_v,
            "spread_pp": round(rng, 6),
            "n_graphs_in_raw": raw_counts.get(k),
            "all_three_present": all(
                v is not None for v in (raw_v, auc_v, slope_v)
            ),
            "agree": agree,
        }
        cells.append(rec)
        if not agree:
            mismatches.append(rec)

    return {
        "meta": {
            "sources": {
                "oracle_gap":           _resolve_label(oracle_path),
                "oracle_gap_auc":       _resolve_label(auc_path),
                "cache_sensitivity":    _resolve_label(slope_path),
            },
            "scope_l3_sizes": list(PAPER_L3),
            "tolerance_pp": TOLERANCE,
            "n_cells_checked": len(cells),
            "n_mismatches": len(mismatches),
            "n_full_triple_cells": sum(
                1 for c in cells if c["all_three_present"]
            ),
        },
        "mismatches": mismatches,
        "cells": cells,
    }


def emit_md(payload: dict) -> str:
    m = payload["meta"]
    out = []
    out.append("# Cross-generator gap_pp parity")
    out.append("")
    out.append(
        f"Tolerance: **{m['tolerance_pp']} pp**  •  cells checked: "
        f"{m['n_cells_checked']}  •  full triple cells "
        f"{m['n_full_triple_cells']}  •  mismatches: **{m['n_mismatches']}**"
    )
    out.append("")
    out.append("## Sources reconciled")
    out.append("")
    for label, src in m["sources"].items():
        out.append(f"- {label}: `{src}`")
    out.append("")
    out.append("## Headline")
    out.append("")
    if m["n_mismatches"] == 0:
        out.append(
            "✅ All three aggregators report identical gap_pp values "
            "(within tolerance) for every (app, policy, L3) triple "
            "they share."
        )
    else:
        out.append(
            f"❌ {m['n_mismatches']} (app, policy, L3) triples disagree "
            f"across the three generators. See table below."
        )
    out.append("")
    if payload["mismatches"]:
        out.append("## Mismatches")
        out.append("")
        out.append(
            "| app | policy | L3 | raw mean | AUC traj | slope | spread |"
        )
        out.append("|---|---|---|---:|---:|---:|---:|")
        for m_ in payload["mismatches"]:
            out.append(
                f"| {m_['app']} | {m_['policy']} | {m_['l3_size']} "
                f"| {m_['raw_mean_gap_pp']} | {m_['auc_trajectory_gap_pp']} "
                f"| {m_['slope_gap_pp']} | {m_['spread_pp']:.6f} |"
            )
        out.append("")
    out.append("## Sample of reconciled cells")
    out.append("")
    out.append(
        "| app | policy | L3 | gap_pp (raw) | "
        "n graphs | spread |"
    )
    out.append("|---|---|---|---:|---:|---:|")
    for c in payload["cells"][:20]:
        out.append(
            f"| {c['app']} | {c['policy']} | {c['l3_size']} "
            f"| {c['raw_mean_gap_pp']} | {c['n_graphs_in_raw']} "
            f"| {c['spread_pp']:.6f} |"
        )
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle-json", type=Path, default=WIKI_DATA / "oracle_gap.json"
    )
    parser.add_argument(
        "--auc-json", type=Path, default=WIKI_DATA / "oracle_gap_auc.json"
    )
    parser.add_argument(
        "--slope-json",
        type=Path,
        default=WIKI_DATA / "cache_sensitivity_slope.json",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=WIKI_DATA / "cross_generator_gap_parity.json",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=WIKI_DATA / "cross_generator_gap_parity.md",
    )
    args = parser.parse_args()

    payload = build_payload(args.oracle_json, args.auc_json, args.slope_json)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    args.md_out.write_text(emit_md(payload))

    m = payload["meta"]
    print(
        f"cross-generator-gap-parity: cells={m['n_cells_checked']} "
        f"| full_triples={m['n_full_triple_cells']} "
        f"| mismatches={m['n_mismatches']} "
        f"| tolerance={m['tolerance_pp']} pp"
    )
    return 0 if m["n_mismatches"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
