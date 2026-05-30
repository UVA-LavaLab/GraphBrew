"""Literature-faithfulness cache-size monotonicity audit (LIT-Mono,
gate 229).

For every (graph, app, policy) triple in the lit-faith corpus that
spans ≥ 2 L3 sizes and has both `lru_miss_rate` and `policy_miss_rate`
populated, this gate enforces the physical invariant that miss rate is
a monotonically non-increasing function of cache size: a larger LLC
cannot cause *more* misses on the same workload (modulo a small noise
tolerance for re-runs).

If this ever fires it almost always means one of three things:
  * the comparator merged trace runs from different commits;
  * a workload's input size was silently bumped between L3 points;
  * an L3 label was misencoded (e.g., `4MB` row swapped with `1MB`).

The gate also surfaces the per-triple **slope** (miss-rate reduction
per doubling of L3) so reviewers can spot saturated workloads (slope
≈ 0, the corpus is not exercising cache pressure) and degenerate
ones (slope ≈ 100 %, the workload's working set is impossibly small).
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
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]

# L3 size labels → bytes. The lit-faith comparator emits these labels;
# they are derived from `cache_sim` config strings, not free-form.
L3_BYTES: Dict[str, int] = {
    "4kB":   4 * 1024,
    "16kB":  16 * 1024,
    "64kB":  64 * 1024,
    "256kB": 256 * 1024,
    "1MB":   1 * 1024 * 1024,
    "4MB":   4 * 1024 * 1024,
    "8MB":   8 * 1024 * 1024,
    "16MB": 16 * 1024 * 1024,
}

# Noise tolerance: miss-rate may *increase* by up to this many absolute
# fraction-points between adjacent L3 sizes before we call it a real
# monotonicity violation. 0.5 pp is well above any deterministic
# cache_sim run-to-run jitter today.
MONOTONICITY_TOLERANCE = 0.005   # 0.5 pp


def _l3_bytes(label: str) -> int:
    if label not in L3_BYTES:
        raise SystemExit(f"[lit-faith-monotonicity] unknown L3 label {label!r}")
    return L3_BYTES[label]


def _slope_per_doubling(samples: List[Tuple[int, float]]) -> float:
    """Average miss-rate drop per *doubling* of L3 size.

    samples = sorted list of (l3_bytes, miss_rate). Returns the average
    delta_miss_rate per log2(L3) step. Positive values mean miss rate
    decreases as L3 grows (expected); negative values mean miss rate
    increases (anomalous)."""
    if len(samples) < 2:
        return 0.0
    drops: List[float] = []
    for i in range(1, len(samples)):
        prev_b, prev_mr = samples[i - 1]
        curr_b, curr_mr = samples[i]
        if prev_b <= 0 or curr_b <= 0:
            continue
        log_step = math.log2(curr_b) - math.log2(prev_b)
        if log_step <= 0:
            continue
        drops.append((prev_mr - curr_mr) / log_step)
    if not drops:
        return 0.0
    return statistics.mean(drops)


def build_audit(lit_faith: Dict[str, Any]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = list(lit_faith.get("per_claim") or [])
    if not rows:
        raise SystemExit("[lit-faith-monotonicity] empty per_claim table — run `make lit-faith` first")

    triples: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    rows_with_rates = 0
    for r in rows:
        if r.get("lru_miss_rate") is None or r.get("policy_miss_rate") is None:
            continue
        rows_with_rates += 1
        triples[(r["graph"], r["app"], r["policy"])].append(r)

    triple_audits: List[Dict[str, Any]] = []
    violations_lru: List[Dict[str, Any]] = []
    violations_policy: List[Dict[str, Any]] = []
    saturated: List[Dict[str, Any]] = []
    slopes_lru: List[float] = []
    slopes_policy: List[float] = []
    audited_count = 0
    skipped_singleton = 0

    for (graph, app, policy), members in sorted(triples.items()):
        members.sort(key=lambda x: _l3_bytes(x["l3_size"]))
        if len(members) < 2:
            skipped_singleton += 1
            continue
        audited_count += 1

        lru_samples = [(_l3_bytes(m["l3_size"]), float(m["lru_miss_rate"]))
                       for m in members]
        pol_samples = [(_l3_bytes(m["l3_size"]), float(m["policy_miss_rate"]))
                       for m in members]

        # Detect monotonicity violations: bigger L3 with higher miss
        # rate (beyond tolerance).
        for i in range(1, len(lru_samples)):
            prev_b, prev_mr = lru_samples[i - 1]
            curr_b, curr_mr = lru_samples[i]
            if curr_mr > prev_mr + MONOTONICITY_TOLERANCE:
                violations_lru.append({
                    "graph": graph, "app": app, "policy": policy,
                    "l3_smaller": members[i - 1]["l3_size"],
                    "l3_larger":  members[i]["l3_size"],
                    "miss_rate_smaller": prev_mr,
                    "miss_rate_larger":  curr_mr,
                    "delta": round(curr_mr - prev_mr, 6),
                })
        for i in range(1, len(pol_samples)):
            prev_b, prev_mr = pol_samples[i - 1]
            curr_b, curr_mr = pol_samples[i]
            if curr_mr > prev_mr + MONOTONICITY_TOLERANCE:
                violations_policy.append({
                    "graph": graph, "app": app, "policy": policy,
                    "l3_smaller": members[i - 1]["l3_size"],
                    "l3_larger":  members[i]["l3_size"],
                    "miss_rate_smaller": prev_mr,
                    "miss_rate_larger":  curr_mr,
                    "delta": round(curr_mr - prev_mr, 6),
                })

        slope_lru = _slope_per_doubling(lru_samples)
        slope_pol = _slope_per_doubling(pol_samples)
        slopes_lru.append(slope_lru)
        slopes_policy.append(slope_pol)

        # Saturation: total miss-rate change over the full sweep is
        # tiny — workload is below noise floor of cache pressure.
        total_lru_drop = lru_samples[0][1] - lru_samples[-1][1]
        total_pol_drop = pol_samples[0][1] - pol_samples[-1][1]
        if total_lru_drop < 0.01 and total_pol_drop < 0.01:
            saturated.append({
                "graph": graph, "app": app, "policy": policy,
                "l3_range":     f"{members[0]['l3_size']}→{members[-1]['l3_size']}",
                "total_lru_drop":    round(total_lru_drop, 6),
                "total_policy_drop": round(total_pol_drop, 6),
            })

        triple_audits.append({
            "graph":           graph,
            "app":             app,
            "policy":          policy,
            "l3_count":        len(members),
            "l3_min":          members[0]["l3_size"],
            "l3_max":          members[-1]["l3_size"],
            "lru_miss_min":    round(min(s[1] for s in lru_samples), 6),
            "lru_miss_max":    round(max(s[1] for s in lru_samples), 6),
            "policy_miss_min": round(min(s[1] for s in pol_samples), 6),
            "policy_miss_max": round(max(s[1] for s in pol_samples), 6),
            "lru_total_drop":    round(total_lru_drop, 6),
            "policy_total_drop": round(total_pol_drop, 6),
            "slope_lru":         round(slope_lru, 6),
            "slope_policy":      round(slope_pol, 6),
            "samples": [
                {
                    "l3_size":         m["l3_size"],
                    "lru_miss_rate":   round(float(m["lru_miss_rate"]), 6),
                    "policy_miss_rate": round(float(m["policy_miss_rate"]), 6),
                }
                for m in members
            ],
        })

    return {
        "schema_version": 1,
        "tolerance":      MONOTONICITY_TOLERANCE,
        "summary": {
            "total_rows":           len(rows),
            "rows_with_rates":      rows_with_rates,
            "triple_count":         len(triples),
            "triples_audited":      audited_count,
            "triples_singleton":    skipped_singleton,
            "violations_lru":       len(violations_lru),
            "violations_policy":    len(violations_policy),
            "saturated_count":      len(saturated),
            "median_slope_lru":     round(statistics.median(slopes_lru), 6) if slopes_lru else 0.0,
            "median_slope_policy":  round(statistics.median(slopes_policy), 6) if slopes_policy else 0.0,
            "max_slope_lru":        round(max(slopes_lru), 6) if slopes_lru else 0.0,
            "min_slope_lru":        round(min(slopes_lru), 6) if slopes_lru else 0.0,
            "max_slope_policy":     round(max(slopes_policy), 6) if slopes_policy else 0.0,
            "min_slope_policy":     round(min(slopes_policy), 6) if slopes_policy else 0.0,
        },
        "violations_lru":    violations_lru,
        "violations_policy": violations_policy,
        "saturated_triples": saturated,
        "triples":           triple_audits,
    }


def write_markdown(audit: Dict[str, Any], path: Path) -> None:
    s = audit["summary"]
    lines: List[str] = []
    lines.append("# Literature-faithfulness cache-size monotonicity audit\n")
    lines.append(f"Generated by `make lit-monotonicity`. Tolerance: "
                 f"{audit['tolerance']:.3f} (i.e., miss rate may increase by "
                 f"≤ {audit['tolerance']*100:.1f} pp between adjacent L3 "
                 f"sizes before counting as a violation).\n")
    lines.append("## Summary\n")
    lines.append(f"- Total per_claim rows: **{s['total_rows']}** "
                 f"(rows with both miss-rates: {s['rows_with_rates']})")
    lines.append(f"- Triples (graph × app × policy) with miss-rates: "
                 f"**{s['triple_count']}**")
    lines.append(f"- Triples audited (≥ 2 L3 sizes): **{s['triples_audited']}** "
                 f"(singleton: {s['triples_singleton']})")
    lines.append(f"- LRU monotonicity violations:    **{s['violations_lru']}**")
    lines.append(f"- Policy monotonicity violations: **{s['violations_policy']}**")
    lines.append(f"- Saturated triples (< 1 pp total drop): "
                 f"**{s['saturated_count']}**")
    lines.append(f"- Median slope per log2(L3): "
                 f"LRU {s['median_slope_lru']:.4f}, "
                 f"Policy {s['median_slope_policy']:.4f}\n")

    lines.append("## Per-triple table\n")
    lines.append("| graph | app | policy | n | l3 range | lru drop | pol drop | slope_lru | slope_pol |")
    lines.append("|-------|-----|--------|--:|---------:|---------:|---------:|---------:|---------:|")
    for t in audit["triples"]:
        l3_range = f"{t['l3_min']}→{t['l3_max']}"
        lines.append(
            f"| {t['graph']} | {t['app']} | {t['policy']} | {t['l3_count']} | "
            f"{l3_range} | {t['lru_total_drop']:.4f} | "
            f"{t['policy_total_drop']:.4f} | {t['slope_lru']:.4f} | "
            f"{t['slope_policy']:.4f} |"
        )
    lines.append("")

    if audit["violations_lru"] or audit["violations_policy"]:
        lines.append("## Violations\n")
        for label, vs in [("LRU", audit["violations_lru"]),
                           ("Policy", audit["violations_policy"])]:
            for v in vs:
                lines.append(
                    f"- **{label}** {v['graph']}/{v['app']}/{v['policy']}: "
                    f"{v['l3_smaller']}→{v['l3_larger']} "
                    f"mr {v['miss_rate_smaller']:.4f}→{v['miss_rate_larger']:.4f} "
                    f"(Δ {v['delta']:+.4f})"
                )
        lines.append("")
    else:
        lines.append("_No monotonicity violations — every triple's miss rate "
                     "is non-increasing in L3 size._")

    if audit["saturated_triples"]:
        lines.append("\n## Saturated triples\n")
        for t in audit["saturated_triples"]:
            lines.append(
                f"- {t['graph']}/{t['app']}/{t['policy']} "
                f"({t['l3_range']}): lru drop {t['total_lru_drop']:.4f}, "
                f"policy drop {t['total_policy_drop']:.4f}"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(audit: Dict[str, Any], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "graph", "app", "policy", "l3_count", "l3_min", "l3_max",
            "lru_miss_min", "lru_miss_max", "lru_total_drop",
            "policy_miss_min", "policy_miss_max", "policy_total_drop",
            "slope_lru", "slope_policy",
        ])
        w.writeheader()
        for t in audit["triples"]:
            row = {k: t[k] for k in w.fieldnames}
            w.writerow(row)


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="LIT-Mono cache-size monotonicity audit (gate 229)")
    p.add_argument("--lit-faith-json", type=Path,
                   default=REPO_ROOT / "wiki" / "data" / "literature_faithfulness_postfix.json")
    p.add_argument("--json-out", type=Path,
                   default=REPO_ROOT / "wiki" / "data" / "lit_faith_monotonicity.json")
    p.add_argument("--md-out", type=Path,
                   default=REPO_ROOT / "wiki" / "data" / "lit_faith_monotonicity.md")
    p.add_argument("--csv-out", type=Path,
                   default=REPO_ROOT / "wiki" / "data" / "lit_faith_monotonicity.csv")
    args = p.parse_args(list(argv) if argv is not None else None)

    if not args.lit_faith_json.exists():
        print(f"[lit-faith-monotonicity] missing {args.lit_faith_json}; "
              "run `make lit-faith` first", file=sys.stderr)
        return 1

    lit_faith = json.loads(args.lit_faith_json.read_text(encoding="utf-8"))
    audit = build_audit(lit_faith)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")
    write_markdown(audit, args.md_out)
    write_csv(audit, args.csv_out)

    s = audit["summary"]
    print(f"[lit-faith-monotonicity] {s['triples_audited']} triples audited; "
          f"LRU violations {s['violations_lru']}, policy violations "
          f"{s['violations_policy']}, saturated {s['saturated_count']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
