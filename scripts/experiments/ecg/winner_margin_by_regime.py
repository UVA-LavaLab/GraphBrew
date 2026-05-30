"""Gate 62 — winner-margin distribution per WSS-relative regime.

For each (app, graph, L3) cell with all four policies present, the
"winner margin" is the gap (in percentage points of miss-rate) between
the winning policy and the second-best policy. A larger margin means
the winner is decisively better; a smaller margin means the policies
are roughly interchangeable for that cell.

This gate asks: when an oracle-aware policy wins, is its margin over
second-place larger in tight-capacity regimes than in loose-capacity
regimes? If yes, oracle-aware replacement is most valuable precisely
when capacity pressure is highest — a key narrative claim of the paper.

For each (policy, wss_regime) we record:
  cells_won            : how many cells this policy wins in this regime
  median_margin_pp     : median over those wins
  mean_margin_pp       : mean over those wins
  p90_margin_pp        : 90th percentile

Verdict PASS iff:
  * Every regime has at least one win recorded.
  * For at least one oracle-aware policy (GRASP, POPT), the median
    winner margin at under_wss is STRICTLY GREATER than the median at
    over_wss (margins shrink as capacity loosens — oracle pinning pays
    off most under heavy pressure).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ORACLE_JSON = REPO_ROOT / "wiki" / "data" / "oracle_gap.json"
DEFAULT_WSS_JSON = REPO_ROOT / "wiki" / "data" / "wss_relative_l3.json"
DEFAULT_JSON_OUT = REPO_ROOT / "wiki" / "data" / "winner_margin_by_regime.json"
DEFAULT_MD_OUT = REPO_ROOT / "wiki" / "data" / "winner_margin_by_regime.md"

REGIMES = ("under_wss", "near_wss", "over_wss")
POLICIES = ("GRASP", "LRU", "POPT", "SRRIP")
ORACLE_AWARE = ("GRASP", "POPT")

# L3 size labels -> bytes (must match the corpus exactly).
L3_BYTES = {
    "4kB":   4 * 1024,
    "16kB":  16 * 1024,
    "64kB":  64 * 1024,
    "256kB": 256 * 1024,
    "1MB":   1024 * 1024,
    "4MB":   4 * 1024 * 1024,
    "8MB":   8 * 1024 * 1024,
}


def _classify(l3_bytes: float, wss_bytes: float) -> str:
    # Must match scripts/experiments/ecg/wss_relative_l3.py exactly.
    ratio = l3_bytes / wss_bytes
    if ratio < 0.25:
        return "under_wss"
    if ratio > 4.0:
        return "over_wss"
    return "near_wss"


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[k]


def build(oracle_payload: dict, wss_meta: dict) -> dict:
    wss_proxies = wss_meta["wss_proxies"]
    rows = oracle_payload["rows"]
    cells = defaultdict(list)  # (app, graph, l3) -> list[row]
    for r in rows:
        cells[(r["app"], r["graph"], r["l3_size"])].append(r)

    margins = defaultdict(list)  # (policy, regime) -> list[float]
    cells_classified = 0
    cells_skipped = 0
    for (app, graph, l3), rs in cells.items():
        if len(rs) != len(POLICIES):
            cells_skipped += 1
            continue
        if graph not in wss_proxies or l3 not in L3_BYTES:
            cells_skipped += 1
            continue
        regime = _classify(L3_BYTES[l3], wss_proxies[graph])
        miss = sorted([(float(r["miss_rate"]), r["policy"]) for r in rs])
        best_miss, best_pol = miss[0]
        second_miss, _ = miss[1]
        margin_pp = (second_miss - best_miss) * 100.0  # to pp
        margins[(best_pol, regime)].append(margin_pp)
        cells_classified += 1

    per_policy_regime: dict[str, dict] = {}
    for pol in POLICIES:
        for regime in REGIMES:
            xs = margins.get((pol, regime), [])
            per_policy_regime[f"{pol}/{regime}"] = {
                "policy":      pol,
                "wss_regime":  regime,
                "cells_won":   len(xs),
                "median_margin_pp": round(_median(xs), 4),
                "mean_margin_pp":   round(sum(xs) / len(xs), 4) if xs else 0.0,
                "p90_margin_pp":    round(_pct(xs, 0.9), 4),
                "max_margin_pp":    round(max(xs), 4) if xs else 0.0,
            }

    # Verdict: every regime has at least one win recorded
    # AND at least one oracle-aware policy has under_wss median >
    # over_wss median (margins shrink as capacity loosens).
    regime_has_wins = {
        regime: any(margins.get((pol, regime)) for pol in POLICIES)
        for regime in REGIMES
    }
    shrink_evidence = []
    for pol in ORACLE_AWARE:
        u = _median(margins.get((pol, "under_wss"), []))
        o = _median(margins.get((pol, "over_wss"), []))
        if u > 0 and u > o:
            shrink_evidence.append({"policy": pol, "under_median": u, "over_median": o})

    verdict = "PASS" if (
        all(regime_has_wins.values()) and len(shrink_evidence) >= 1
    ) else "FAIL"

    return {
        "meta": {
            "policies":            list(POLICIES),
            "regimes":             list(REGIMES),
            "cells_classified":    cells_classified,
            "cells_skipped":       cells_skipped,
            "regime_has_wins":     regime_has_wins,
            "shrink_evidence":     shrink_evidence,
            "verdict":             verdict,
            "verdict_invariant":   (
                "PASS iff every regime has at least one win AND at least one "
                "oracle-aware policy has median winner-margin strictly larger "
                "at under_wss than at over_wss"
            ),
        },
        "per_policy_regime": per_policy_regime,
    }


def render_md(result: dict, src_label: str) -> str:
    m = result["meta"]
    out = [
        "# Gate 62 — Winner-margin distribution per WSS regime",
        "",
        f"source: `{src_label}`",
        "",
        f"verdict: **{m['verdict']}**",
        "",
        f"  invariant: {m['verdict_invariant']}",
        "",
        f"cells classified: {m['cells_classified']} "
        f"(skipped {m['cells_skipped']})",
        "",
        "## Per-(policy, regime) winner margin in pp of miss-rate",
        "",
        "| policy | regime | cells won | median pp "
        "| mean pp | p90 pp | max pp |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pol in POLICIES:
        for regime in REGIMES:
            c = result["per_policy_regime"][f"{pol}/{regime}"]
            out.append(
                f"| {pol} | {regime} | {c['cells_won']} "
                f"| {c['median_margin_pp']:.3f} "
                f"| {c['mean_margin_pp']:.3f} "
                f"| {c['p90_margin_pp']:.3f} "
                f"| {c['max_margin_pp']:.3f} |"
            )
    out.extend([
        "",
        "## Margin-shrink evidence",
        "",
        "Oracle-aware policies whose under_wss median margin exceeds "
        "their over_wss median margin (the paper's central claim).",
        "",
        "| policy | under median pp | over median pp |",
        "| --- | ---: | ---: |",
    ])
    for ev in m["shrink_evidence"]:
        out.append(
            f"| {ev['policy']} | {ev['under_median']:.3f} "
            f"| {ev['over_median']:.3f} |"
        )
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oracle-json", type=Path, default=DEFAULT_ORACLE_JSON)
    ap.add_argument("--wss-json", type=Path, default=DEFAULT_WSS_JSON)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    ap.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = ap.parse_args()

    src_path = args.oracle_json
    try:
        src_label = str(src_path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        src_label = str(src_path)

    oracle = json.loads(args.oracle_json.read_text())
    wss = json.loads(args.wss_json.read_text())
    result = build(oracle, wss["meta"])
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(render_md(result, src_label))
    m = result["meta"]
    shrinks = ",".join(ev["policy"] for ev in m["shrink_evidence"]) or "-"
    print(
        f"winner-margin-by-regime: classified={m['cells_classified']} "
        f"skipped={m['cells_skipped']} | "
        f"shrink_evidence=[{shrinks}] | verdict={m['verdict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
