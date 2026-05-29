"""Gate 64 — cross-policy mean-margin asymmetry.

For every unordered pair (A, B) of distinct policies, look at every
cache cell (app, graph, L3) and partition cells by who wins between A
and B (head-to-head, ignoring the other two policies):

    cells where A is better (A_miss < B_miss):
        record (B_miss - A_miss) pp  -> "A_win_margin"
    cells where B is better (B_miss < A_miss):
        record (A_miss - B_miss) pp  -> "B_win_margin"

Two policies are "symmetric" when, conditional on losing, both lose by
roughly the same amount. Many oracle-aware vs oracle-aware comparisons
in this corpus are NOT symmetric: when GRASP loses head-to-head to
POPT, it loses by a small amount; when POPT loses head-to-head to
GRASP, it also loses by a small amount — but the magnitude can be
asymmetric across pairs. Gate 64 measures and pins this.

For each unordered pair (A, B) we record:
    a_wins                  : count of cells where A wins H2H
    b_wins                  : count of cells where B wins H2H
    ties                    : count of cells where A_miss == B_miss
    a_mean_margin_pp        : mean of (B - A) in pp over A_wins
    b_mean_margin_pp        : mean of (A - B) in pp over B_wins
    asymmetry_ratio         : max(a_mean, b_mean) / min(a_mean, b_mean)
                              (1.0 means perfectly symmetric loss profiles)

The verdict is a structural / sanity invariant rather than a paper
claim:
  PASS iff every pair has at least one cell where each side wins
       AND the most asymmetric pair's ratio is below a generous
           sanity bound (today 20x) — pinning the observed
           asymmetry ceiling so a future regression that explodes
           it is caught.

Today: ORACLE_AWARE (GRASP vs POPT) is the most-balanced pair; the
ratio bound mainly catches regressions where one policy's mean-loss
spirals (e.g., a buggy POPT update producing 100x worse losses on
the cells it does lose).
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ORACLE_JSON = REPO_ROOT / "wiki" / "data" / "oracle_gap.json"
DEFAULT_JSON_OUT = REPO_ROOT / "wiki" / "data" / "cross_policy_asymmetry.json"
DEFAULT_MD_OUT = REPO_ROOT / "wiki" / "data" / "cross_policy_asymmetry.md"

POLICIES = ("GRASP", "LRU", "POPT", "SRRIP")

# Sanity ceiling for the asymmetry ratio. The most-asymmetric pair
# observed today is well below this; raising it requires explicit
# review.
ASYMMETRY_RATIO_CEILING = 20.0


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def build(payload: dict) -> dict:
    rows = payload["rows"]
    by_cell: dict = defaultdict(dict)  # (app, graph, l3) -> {policy: miss}
    for r in rows:
        by_cell[(r["app"], r["graph"], r["l3_size"])][r["policy"]] = float(
            r["miss_rate"]
        )

    per_pair: dict[str, dict] = {}
    pair_summary = []
    for a, b in itertools.combinations(POLICIES, 2):
        a_wins = []
        b_wins = []
        ties = 0
        for misses in by_cell.values():
            if a not in misses or b not in misses:
                continue
            ma = misses[a]
            mb = misses[b]
            if ma < mb:
                a_wins.append((mb - ma) * 100.0)
            elif mb < ma:
                b_wins.append((ma - mb) * 100.0)
            else:
                ties += 1
        a_mean = _mean(a_wins)
        b_mean = _mean(b_wins)
        ratio = (
            max(a_mean, b_mean) / min(a_mean, b_mean)
            if a_mean > 0 and b_mean > 0
            else float("inf")
        )
        entry = {
            "a_policy":          a,
            "b_policy":          b,
            "a_wins":            len(a_wins),
            "b_wins":            len(b_wins),
            "ties":              ties,
            "a_mean_margin_pp":  round(a_mean, 4),
            "b_mean_margin_pp":  round(b_mean, 4),
            "asymmetry_ratio":   round(ratio, 4)
                                  if ratio != float("inf") else None,
        }
        per_pair[f"{a}_vs_{b}"] = entry
        pair_summary.append(entry)

    # Verdict invariants.
    #   1. every pair has at least one a-win and at least one b-win
    #   2. observed max asymmetry ratio <= ceiling
    both_win = all(p["a_wins"] >= 1 and p["b_wins"] >= 1 for p in pair_summary)
    finite_ratios = [
        p["asymmetry_ratio"]
        for p in pair_summary
        if p["asymmetry_ratio"] is not None
    ]
    max_ratio = max(finite_ratios) if finite_ratios else 0.0
    under_ceiling = max_ratio < ASYMMETRY_RATIO_CEILING
    verdict = "PASS" if (both_win and under_ceiling) else "FAIL"

    return {
        "meta": {
            "policies":              list(POLICIES),
            "pair_count":            len(pair_summary),
            "every_pair_both_win":   both_win,
            "max_asymmetry_ratio":   round(max_ratio, 4),
            "ratio_ceiling":         ASYMMETRY_RATIO_CEILING,
            "max_ratio_under_ceiling": under_ceiling,
            "verdict":               verdict,
            "verdict_invariant":     (
                "PASS iff every (A,B) policy pair has at least one cell "
                "where each side wins head-to-head AND the largest "
                "observed asymmetry ratio (max(meanA,meanB) / "
                "min(meanA,meanB)) is strictly less than the sanity "
                f"ceiling of {ASYMMETRY_RATIO_CEILING}"
            ),
        },
        "per_pair": per_pair,
    }


def render_md(result: dict, src_label: str) -> str:
    m = result["meta"]
    out = [
        "# Gate 64 — Cross-policy mean-margin asymmetry",
        "",
        f"source: `{src_label}`",
        "",
        f"verdict: **{m['verdict']}**",
        "",
        f"  invariant: {m['verdict_invariant']}",
        "",
        f"max asymmetry ratio observed: {m['max_asymmetry_ratio']} "
        f"(ceiling {m['ratio_ceiling']})",
        "",
        "## Head-to-head per policy pair (margin in pp of miss-rate)",
        "",
        "| pair | A wins | B wins | ties | A mean pp | B mean pp "
        "| asymmetry |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in sorted(result["per_pair"].keys()):
        p = result["per_pair"][key]
        ratio = (
            f"{p['asymmetry_ratio']:.3f}"
            if p["asymmetry_ratio"] is not None
            else "∞"
        )
        out.append(
            f"| {p['a_policy']} vs {p['b_policy']} "
            f"| {p['a_wins']} | {p['b_wins']} | {p['ties']} "
            f"| {p['a_mean_margin_pp']:.3f} | {p['b_mean_margin_pp']:.3f} "
            f"| {ratio} |"
        )
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oracle-json", type=Path, default=DEFAULT_ORACLE_JSON)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    ap.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = ap.parse_args()

    src_path = args.oracle_json
    try:
        src_label = str(src_path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        src_label = str(src_path)

    payload = json.loads(args.oracle_json.read_text())
    result = build(payload)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True))
    args.md_out.write_text(render_md(result, src_label))
    m = result["meta"]
    print(
        f"cross-policy-asymmetry: pairs={m['pair_count']} "
        f"every_pair_both_win={m['every_pair_both_win']} "
        f"max_ratio={m['max_asymmetry_ratio']} "
        f"verdict={m['verdict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
