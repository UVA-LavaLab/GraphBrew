#!/usr/bin/env python3
"""Compare GRASP-vs-LRU delta sign across simulators.

For each (graph, app, L3 size) we compute
``delta_miss = miss_rate(GRASP) - miss_rate(LRU)`` from every available
``roi_matrix.csv`` (cache_sim is the reference, gem5 / Sniper are tested).

The script reports per-row sign agreement and exits non-zero whenever a
*mandatory* L3 size (default: 4 kB and 32 kB) disagrees in sign between the
cache_sim reference and a timing simulator. Larger caches are reported as
warnings only because GRASP and LRU often converge there.

Usage::

    python3 scripts/experiments/ecg/sign_consistency.py \\
        --cache-root /tmp/graphbrew-grasp-cache-sweep \\
        --gem5-root  /tmp/graphbrew-grasp-gem5-sweep \\
        --sniper-root /tmp/graphbrew-grasp-sniper-sweep \\
        --pairs email-Eu-core/pr email-Eu-core/bc cit-Patents/pr cit-Patents/bc
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

MANDATORY_L3_SIZES = ("4kB", "32kB")


@dataclass(frozen=True)
class PolicyRow:
    simulator: str
    graph: str
    app: str
    l3_size: str
    policy: str
    miss_rate: float | None
    section: int = 0


def _coerce_float(value: object) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: object) -> int | None:
    if value in (None, "", "None"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def load_roi_matrix(path: Path, simulator_filter: str, graph: str, app: str) -> list[PolicyRow]:
    if not path.exists():
        return []
    rows: list[PolicyRow] = []
    with path.open(newline="") as fh:
        for raw in csv.DictReader(fh):
            if raw.get("simulator") != simulator_filter:
                continue
            if raw.get("status") not in (None, "", "ok"):
                continue
            miss = _coerce_float(raw.get("l3_miss_rate"))
            if miss is None:
                continue
            rows.append(
                PolicyRow(
                    simulator=simulator_filter,
                    graph=graph,
                    app=app,
                    l3_size=str(raw.get("l3_size")),
                    policy=str(raw.get("policy")),
                    miss_rate=miss,
                    section=_coerce_int(raw.get("section")) or 0,
                )
            )
    return rows


def _pick(rows: Iterable[PolicyRow], policy: str, l3_size: str) -> PolicyRow | None:
    matches = [r for r in rows if r.policy == policy and r.l3_size == l3_size]
    if not matches:
        return None
    # Prefer the ROI section. cache_sim emits section 0 (single value);
    # gem5 emits section 1 at m5_work_end (= ROI), section 2 at m5_dump_stats
    # (post-ROI cumulative which includes teardown noise on tiny graphs).
    # We deterministically pick the smallest non-zero section if any exists,
    # falling back to section 0 for cache_sim / sniper rows.
    nonzero = [r for r in matches if r.section]
    if nonzero:
        nonzero.sort(key=lambda r: r.section)
        return nonzero[0]
    return matches[0]


def _sign(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < -1e-9:
        return "-"
    if value > 1e-9:
        return "+"
    return "0"


def compute_deltas(rows: list[PolicyRow]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    sizes = sorted({r.l3_size for r in rows})
    for size in sizes:
        lru = _pick(rows, "LRU", size)
        grasp = _pick(rows, "GRASP", size)
        if lru is None or grasp is None or lru.miss_rate is None or grasp.miss_rate is None:
            out[size] = None
        else:
            out[size] = grasp.miss_rate - lru.miss_rate
    return out


def evaluate(
    cache_root: Path,
    gem5_root: Path | None,
    sniper_root: Path | None,
    pairs: list[tuple[str, str]],
    mandatory_sizes: tuple[str, ...] = MANDATORY_L3_SIZES,
) -> dict:
    summary: dict = {"pairs": [], "mandatory_violations": [], "warnings": []}
    for graph, app in pairs:
        slug = f"{graph}-{app}"
        ref_csv = cache_root / slug / "DBG" / "roi_matrix.csv"
        ref_rows = load_roi_matrix(ref_csv, "cache_sim", graph, app)
        ref_delta = compute_deltas(ref_rows)

        block: dict = {
            "graph": graph,
            "app": app,
            "cache_sim_csv": str(ref_csv),
            "deltas": {"cache_sim": ref_delta},
        }

        for simulator, root in (("gem5", gem5_root), ("sniper", sniper_root)):
            if root is None:
                continue
            csv_path = root / slug / "DBG" / "roi_matrix.csv"
            sim_rows = load_roi_matrix(csv_path, simulator, graph, app)
            sim_delta = compute_deltas(sim_rows)
            block["deltas"][simulator] = sim_delta
            block.setdefault(f"{simulator}_csv", str(csv_path))

            for size, ref_value in ref_delta.items():
                sim_value = sim_delta.get(size)
                ref_sign = _sign(ref_value)
                sim_sign = _sign(sim_value)
                record = {
                    "graph": graph,
                    "app": app,
                    "l3_size": size,
                    "simulator": simulator,
                    "cache_sim_delta": ref_value,
                    f"{simulator}_delta": sim_value,
                    "cache_sim_sign": ref_sign,
                    f"{simulator}_sign": sim_sign,
                }
                if ref_value is None or sim_value is None:
                    record["status"] = "missing"
                    summary["warnings"].append(record)
                elif ref_sign == sim_sign or "0" in (ref_sign, sim_sign):
                    record["status"] = "ok"
                else:
                    record["status"] = "disagree"
                    if size in mandatory_sizes:
                        summary["mandatory_violations"].append(record)
                    else:
                        summary["warnings"].append(record)

        summary["pairs"].append(block)
    return summary


def format_human(summary: dict) -> str:
    out: list[str] = []
    for block in summary["pairs"]:
        out.append(f"\n=== {block['graph']} / {block['app']} ===")
        sims = list(block["deltas"].keys())
        sizes = sorted({s for d in block["deltas"].values() for s in d.keys()})
        header = "  L3       | " + " | ".join(f"{s:>12}" for s in sims)
        out.append(header)
        out.append("  " + "-" * (len(header) - 2))
        for size in sizes:
            cells = []
            for sim in sims:
                value = block["deltas"][sim].get(size)
                cells.append(f"{value:+.5f}" if value is not None else "      n/a")
            out.append(f"  {size:<8} | " + " | ".join(f"{cell:>12}" for cell in cells))
    if summary["mandatory_violations"]:
        out.append("\nMANDATORY sign disagreements:")
        for rec in summary["mandatory_violations"]:
            out.append(f"  - {rec}")
    if summary["warnings"]:
        out.append("\nWarnings (non-mandatory or missing data):")
        for rec in summary["warnings"]:
            out.append(f"  - {rec}")
    return "\n".join(out) if out else "(no pairs)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--gem5-root", type=Path, default=None)
    parser.add_argument("--sniper-root", type=Path, default=None)
    parser.add_argument(
        "--pairs",
        nargs="+",
        required=True,
        help="Graph/app pairs as graph/app (e.g. email-Eu-core/pr)",
    )
    parser.add_argument(
        "--mandatory-sizes",
        nargs="+",
        default=list(MANDATORY_L3_SIZES),
        help="L3 sizes that MUST agree in sign with cache_sim",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--exit-on-disagree", action="store_true", default=True)
    parser.add_argument("--no-exit-on-disagree", dest="exit_on_disagree", action="store_false")
    args = parser.parse_args(argv)

    pairs: list[tuple[str, str]] = []
    for entry in args.pairs:
        if "/" not in entry:
            parser.error(f"--pairs entry {entry!r} is not graph/app")
        graph, app = entry.split("/", 1)
        pairs.append((graph, app))

    summary = evaluate(
        cache_root=args.cache_root,
        gem5_root=args.gem5_root,
        sniper_root=args.sniper_root,
        pairs=pairs,
        mandatory_sizes=tuple(args.mandatory_sizes),
    )

    print(format_human(summary))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, default=str))

    if args.exit_on_disagree and summary["mandatory_violations"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
