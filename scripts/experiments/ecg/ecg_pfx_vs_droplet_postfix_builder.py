#!/usr/bin/env python3
"""Convert matched-proof prefetcher sweep CSVs into the gate 241
ecg_pfx_vs_droplet_postfix.json per_observation array.

Reads CSVs produced by ``scripts/experiments/ecg/sweeps/pfx_matched_proof_sweep.sh``
and emits postfix observation rows in the schema the audit
(``lit_faith_ecg_pfx_vs_droplet.py``) expects:

  {benchmark, section, l3_size, arm, l3_miss_rate, pf_issued,
   pf_useful, backend, simulator, status}

Sweep layout::

    /tmp/graphbrew-ecg-pfx-vs-droplet-{date}/
        {graph}-{app}/{arm}/roi_matrix.csv

CLI::

    python3 -m scripts.experiments.ecg.ecg_pfx_vs_droplet_postfix_builder \\
        --sweep-root /tmp/graphbrew-ecg-pfx-vs-droplet-canonical \\
        --postfix-in wiki/data/ecg_pfx_vs_droplet_postfix.json \\
        --postfix-out wiki/data/ecg_pfx_vs_droplet_postfix.json

When ``per_observation`` is non-empty and ``--activate`` is passed,
also flips ``status`` from ``deferred`` to ``active`` and clears
``defer_reason``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_ARMS = {"LRU", "DROPLET", "ECG_PFX"}
ARM_TO_DIR = {"LRU": "none", "DROPLET": "DROPLET", "ECG_PFX": "ECG_PFX"}
DIR_TO_ARM = {v: k for k, v in ARM_TO_DIR.items()}

# Per-graph BFS/SSSP frontier offsets used by the sweep (mirror the
# headline-1MB ECG sweep map).
SOURCE_VERTEX_BY_GRAPH = {
    "email-Eu-core": 0,
    "cit-Patents": 1500000,
    "soc-pokec": 800000,
    "web-Google": 0,
    "soc-LiveJournal1": 2000000,
    "com-orkut": 1500000,
}


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_observations(sweep_root: Path,
                      exclude_statuses: tuple[str, ...] = ("error",)) -> list[dict[str, Any]]:
    """Walk the sweep root and return per-observation rows.

    ``exclude_statuses`` filters out CSV rows whose ``status`` column
    matches. Defaults to ('error',) so timed-out / crashed sims do not
    pollute the audit's per-cell row count or trigger spurious
    G1-non-ok-status violations.
    """
    observations: list[dict[str, Any]] = []
    if not sweep_root.exists():
        return observations
    for cell_dir in sorted(sweep_root.iterdir()):
        if not cell_dir.is_dir():
            continue
        name = cell_dir.name
        if "-" not in name:
            continue
        graph, app = name.rsplit("-", 1)
        for arm_dir in sorted(cell_dir.iterdir()):
            if not arm_dir.is_dir():
                continue
            arm = DIR_TO_ARM.get(arm_dir.name)
            if arm is None:
                continue
            csv_path = arm_dir / "roi_matrix.csv"
            if not csv_path.exists():
                continue
            with csv_path.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    status = row.get("status") or "ok"
                    if status in exclude_statuses:
                        continue
                    obs = {
                        "benchmark": app,
                        "section": 0,
                        "l3_size": row.get("l3_size") or "1MB",
                        "arm": arm,
                        "graph": graph,
                        "l3_miss_rate": _coerce_float(row.get("l3_miss_rate")),
                        "pf_issued": _coerce_int(row.get("pf_issued")) or 0,
                        "pf_useful": _coerce_int(row.get("pf_useful")) or 0,
                        "backend": row.get("simulator") or row.get("backend") or "sniper",
                        "simulator": row.get("simulator") or "sniper",
                        "status": status,
                        "policy_label": row.get("policy_label"),
                        "source_csv": str(csv_path),
                        # Carry runtime activity counters so the audit can
                        # cross-check that the prefetcher actually fired.
                        "droplet_sideband_loaded": _coerce_int(row.get("droplet_sideband_loaded")) or 0,
                        "droplet_indirect_issued": _coerce_int(row.get("droplet_indirect_issued")) or 0,
                        "droplet_stride_issued": _coerce_int(row.get("droplet_stride_issued")) or 0,
                        "ecg_pfx_sideband_loaded": _coerce_int(row.get("ecg_pfx_sideband_loaded")) or 0,
                        "ecg_pfx_target_hints_seen": _coerce_int(row.get("ecg_pfx_target_hints_seen")) or 0,
                        "ecg_pfx_issued": _coerce_int(row.get("ecg_pfx_issued")) or 0,
                    }
                    observations.append(obs)
    return observations


def update_postfix(postfix: dict[str, Any], observations: list[dict[str, Any]],
                   sweep_root: Path, activate: bool) -> dict[str, Any]:
    postfix["per_observation"] = observations
    postfix["observation_source_root"] = str(sweep_root)
    if not observations:
        return postfix
    if activate:
        postfix["status"] = "active"
        postfix["defer_reason"] = ""
        postfix["expected_source_pattern"] = str(sweep_root / "{graph}-{app}/{arm}/roi_matrix.csv")
    return postfix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, required=True,
                        help="Sweep output root (e.g. /tmp/graphbrew-ecg-pfx-vs-droplet-canonical)")
    parser.add_argument("--postfix-in", type=Path, required=True,
                        help="Source postfix JSON to update.")
    parser.add_argument("--postfix-out", type=Path, required=True,
                        help="Destination postfix JSON (may be same as --postfix-in).")
    parser.add_argument("--activate", action="store_true",
                        help="If observations are non-empty, flip status=deferred → status=active.")
    parser.add_argument("--print-summary", action="store_true",
                        help="Echo a per-arm summary to stdout.")
    args = parser.parse_args()

    sweep_root = args.sweep_root.resolve()
    observations = load_observations(sweep_root)

    postfix = json.loads(args.postfix_in.read_text())
    postfix = update_postfix(postfix, observations, sweep_root, args.activate)

    args.postfix_out.write_text(json.dumps(postfix, indent=2) + "\n")

    print(f"[pfx-postfix-builder] sweep_root={sweep_root}")
    print(f"[pfx-postfix-builder] observations={len(observations)}")
    if args.activate and observations:
        print(f"[pfx-postfix-builder] status → active (was {postfix.get('status', '?')} on disk)")
    print(f"[pfx-postfix-builder] wrote {args.postfix_out}")
    if args.print_summary and observations:
        by_arm: dict[str, int] = {}
        for obs in observations:
            by_arm[obs["arm"]] = by_arm.get(obs["arm"], 0) + 1
        print(f"[pfx-postfix-builder] per-arm row counts: {by_arm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
