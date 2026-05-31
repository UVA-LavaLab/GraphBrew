#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ECG variants companion to literature_faithfulness_postfix.

Emits the ECG-policy rows that are filtered out of the canonical
literature_faithfulness comparator (which restricts to LRU/SRRIP/
GRASP/POPT for cross-tool parity-gate stability) into a SEPARATE
artifact that gate 282 (headline_coverage) and gate 283
(headline_parity) consume.

Mirrors the gem5_anchor.json -> gem5_anchor_headline_1mb.json
separation pattern: stable canonical artifact for parity gates,
companion artifact for proof-gate consumption.

Output shape:
  graph, app, l3_size, policy, miss_rate, l3_accesses

Only ECG variants and POPT_CHARGED are emitted (anything OUTSIDE
literature_faithfulness.CANONICAL_POLICY_ROSTER).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LIT_FAITH = REPO_ROOT / "scripts/experiments/ecg/literature_faithfulness.py"


def _load_lit_faith():
    if "literature_faithfulness" in sys.modules:
        return sys.modules["literature_faithfulness"]
    spec = importlib.util.spec_from_file_location(
        "literature_faithfulness", LIT_FAITH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["literature_faithfulness"] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sweep-root", type=Path,
                   default=Path("/tmp/graphbrew-lit-baseline"))
    p.add_argument("--sweep-subdir", default="lit")
    p.add_argument("--json-out", type=Path,
                   default=REPO_ROOT / "wiki/data/literature_faithfulness_ecg.json")
    p.add_argument("--md-out", type=Path,
                   default=REPO_ROOT / "wiki/data/literature_faithfulness_ecg.md")
    p.add_argument("--csv-out", type=Path,
                   default=REPO_ROOT / "wiki/data/literature_faithfulness_ecg.csv")
    args = p.parse_args(argv)

    lit = _load_lit_faith()
    canonical = set(lit.CANONICAL_POLICY_ROSTER)

    # Load ALL policies (no filter), then drop canonical baselines.
    observations = lit.load_observations(args.sweep_root, args.sweep_subdir,
                                           policy_filter=None)
    extension = [o for o in observations if o.policy not in canonical]

    # Index canonical LRU rows for delta-vs-LRU computation.
    lru_by_key: dict[tuple[str, str, str], float] = {}
    for o in observations:
        if o.policy == "LRU":
            lru_by_key[(o.graph, o.app, o.l3_size)] = o.miss_rate

    rows = []
    by_policy = defaultdict(int)
    for o in sorted(extension, key=lambda x: (x.graph, x.app, x.l3_size, x.policy)):
        by_policy[o.policy] += 1
        lru = lru_by_key.get((o.graph, o.app, o.l3_size))
        delta_pp = ((o.miss_rate - lru) * 100.0) if lru is not None else None
        rows.append({
            "graph": o.graph,
            "app": o.app,
            "l3_size": o.l3_size,
            "policy": o.policy,
            "miss_rate": round(o.miss_rate, 6),
            "delta_vs_lru_pp": round(delta_pp, 4) if delta_pp is not None else None,
            "l3_accesses": o.accesses,
        })

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps({
        "sweep_root": str(args.sweep_root),
        "sweep_subdir": args.sweep_subdir,
        "policies_emitted": sorted(by_policy.keys()),
        "rows_per_policy": dict(by_policy),
        "rows": rows,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md_lines = []
    md_lines.append("# Literature faithfulness — ECG variants companion")
    md_lines.append("")
    md_lines.append(f"**Sweep root:** `{args.sweep_root}`")
    md_lines.append(f"**Total ECG-variant rows:** {len(rows)}")
    md_lines.append(f"**Policies emitted:** {sorted(by_policy.keys())}")
    md_lines.append("")
    md_lines.append("## Rows-per-policy")
    md_lines.append("")
    md_lines.append("| policy | rows |")
    md_lines.append("|---|---:|")
    for pol, n in sorted(by_policy.items()):
        md_lines.append(f"| `{pol}` | {n} |")
    md_lines.append("")
    md_lines.append("## All rows")
    md_lines.append("")
    md_lines.append("| graph | app | L3 | policy | miss_rate | ΔvsLRU(pp) |")
    md_lines.append("|---|---|---|---|---:|---:|")
    for r in rows:
        dvl = f"{r['delta_vs_lru_pp']:+.4f}" if r["delta_vs_lru_pp"] is not None else "—"
        md_lines.append(f"| {r['graph']} | {r['app']} | {r['l3_size']} | "
                        f"`{r['policy']}` | {r['miss_rate']:.6f} | {dvl} |")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["graph", "app", "l3_size", "policy",
                    "miss_rate", "delta_vs_lru_pp", "l3_accesses"])
        for r in rows:
            w.writerow([r["graph"], r["app"], r["l3_size"], r["policy"],
                        r["miss_rate"],
                        r["delta_vs_lru_pp"] if r["delta_vs_lru_pp"] is not None else "",
                        r["l3_accesses"]])

    print(f"[lit-faith-ecg] rows={len(rows)} policies={sorted(by_policy.keys())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
