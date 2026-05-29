"""Gate 60 — per-policy knee location in WSS-relative L3 capacity.

Combines gate 55 (saturation onset), gate 41 (WSS-relative L3), and gate
58 (curvature). Asks "at what WSS-relative capacity regime does each
policy first reach its plateau (median gap-to-oracle <= 0.5pp)?".

The three regimes from gate 41:
  under_wss  : L3 capacity well below the working set (heavy pressure)
  near_wss   : L3 capacity comparable to the working set (transition)
  over_wss   : L3 capacity well above the working set (no pressure)

For each policy we walk the regime ladder under_wss → near_wss → over_wss
and record the first regime whose median_gap_pp is at-or-below the knee
threshold.  Oracle-aware policies (POPT, GRASP) should plateau at the
left of the ladder; non-oracle policies (LRU, SRRIP) only at the right.

Output schema:
  meta.knee_threshold_pp        : 0.5 pp by default
  meta.regime_ladder            : ["under_wss", "near_wss", "over_wss"]
  meta.knee_rank_by_policy      : ladder-index where each policy plateaus
  meta.verdict                  : PASS iff oracle-aware policies plateau
                                  STRICTLY earlier than non-oracle
  per_policy.<P>                : per-regime mean/median/win_rate +
                                  knee_regime + knee_rank
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WSS_JSON = REPO_ROOT / "wiki" / "data" / "wss_relative_l3.json"
DEFAULT_JSON_OUT = REPO_ROOT / "wiki" / "data" / "wss_knee_location.json"
DEFAULT_MD_OUT = REPO_ROOT / "wiki" / "data" / "wss_knee_location.md"

# The regime ladder: increasing L3-to-WSS capacity.
REGIME_LADDER = ("under_wss", "near_wss", "over_wss")
POLICIES = ("GRASP", "LRU", "POPT", "SRRIP")
ORACLE_AWARE = {"GRASP", "POPT"}
NON_ORACLE = {"LRU", "SRRIP"}

# A policy "plateaus" when its median gap to oracle falls at-or-below
# this threshold (in percentage points of miss-rate).
KNEE_THRESHOLD_PP = 0.5


def _find_knee_regime(per_regime: dict[str, dict], thr: float) -> tuple[str | None, int]:
    """Walk the ladder left→right; return the first regime under thr."""
    for idx, regime in enumerate(REGIME_LADDER):
        cell = per_regime.get(regime)
        if cell is None:
            continue
        if cell["median_gap_pp"] <= thr:
            return regime, idx
    # Never plateaus across the corpus — sentinel rank above ladder end.
    return None, len(REGIME_LADDER)


def build(payload: dict, threshold: float = KNEE_THRESHOLD_PP) -> dict:
    by_pr = payload["by_policy_regime"]
    per_policy: dict[str, dict] = {}
    for pol in POLICIES:
        per_regime = {}
        for regime in REGIME_LADDER:
            key = f"{pol}/{regime}"
            cell = by_pr.get(key)
            if cell is None:
                continue
            per_regime[regime] = {
                "n":              cell["n"],
                "mean_gap_pp":    cell["mean_gap_pp"],
                "median_gap_pp":  cell["median_gap_pp"],
                "p90_gap_pp":     cell["p90_gap_pp"],
                "win_rate":       cell["win_rate"],
            }
        knee_regime, knee_rank = _find_knee_regime(per_regime, threshold)
        per_policy[pol] = {
            "per_regime":   per_regime,
            "knee_regime":  knee_regime,
            "knee_rank":    knee_rank,  # 0=under,1=near,2=over,3=never
            "is_oracle_aware": pol in ORACLE_AWARE,
        }

    # Verdict: every oracle-aware policy must have a strictly smaller
    # knee_rank than every non-oracle policy.
    oracle_ranks = [per_policy[p]["knee_rank"] for p in ORACLE_AWARE]
    non_ranks    = [per_policy[p]["knee_rank"] for p in NON_ORACLE]
    verdict = "PASS" if (
        oracle_ranks and non_ranks
        and max(oracle_ranks) < min(non_ranks)
    ) else "FAIL"

    knee_rank_by_policy = {p: per_policy[p]["knee_rank"] for p in POLICIES}
    knee_regime_by_policy = {p: per_policy[p]["knee_regime"] for p in POLICIES}
    return {
        "meta": {
            "knee_threshold_pp":        threshold,
            "regime_ladder":            list(REGIME_LADDER),
            "policies":                 list(POLICIES),
            "oracle_aware_policies":    sorted(ORACLE_AWARE),
            "non_oracle_policies":      sorted(NON_ORACLE),
            "knee_rank_by_policy":      knee_rank_by_policy,
            "knee_regime_by_policy":    knee_regime_by_policy,
            "max_oracle_aware_knee_rank": max(oracle_ranks),
            "min_non_oracle_knee_rank":   min(non_ranks),
            "verdict":                  verdict,
            "verdict_invariant":        (
                "PASS iff max(knee_rank of oracle-aware policies) "
                "< min(knee_rank of non-oracle policies)"
            ),
        },
        "per_policy": per_policy,
    }


def render_md(result: dict, src_label: str) -> str:
    m = result["meta"]
    out = [
        "# Gate 60 — WSS-relative knee location",
        "",
        f"source: `{src_label}`",
        "",
        f"verdict: **{m['verdict']}**",
        "",
        f"  invariant: {m['verdict_invariant']}",
        "",
        f"knee threshold: median gap-to-oracle ≤ "
        f"**{m['knee_threshold_pp']:.2f} pp**",
        "",
        "regime ladder (increasing capacity): "
        + " → ".join(m["regime_ladder"]),
        "",
        "## Per-policy knee location",
        "",
        "| policy | type | knee regime | knee rank "
        "| median@under | median@near | median@over |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for pol in POLICIES:
        info = result["per_policy"][pol]
        typ = "oracle-aware" if info["is_oracle_aware"] else "non-oracle"
        kr = info["knee_regime"] or "(never)"
        pr = info["per_regime"]
        mu = pr.get("under_wss", {}).get("median_gap_pp", float("nan"))
        mn = pr.get("near_wss", {}).get("median_gap_pp", float("nan"))
        mo = pr.get("over_wss", {}).get("median_gap_pp", float("nan"))
        out.append(
            f"| {pol} | {typ} | {kr} | {info['knee_rank']} "
            f"| {mu:.3f} | {mn:.3f} | {mo:.3f} |"
        )
    out.extend([
        "",
        "## Per-policy win rate by regime",
        "",
        "| policy | win@under | win@near | win@over |",
        "| --- | ---: | ---: | ---: |",
    ])
    for pol in POLICIES:
        pr = result["per_policy"][pol]["per_regime"]
        wu = pr.get("under_wss", {}).get("win_rate", float("nan"))
        wn = pr.get("near_wss", {}).get("win_rate", float("nan"))
        wo = pr.get("over_wss", {}).get("win_rate", float("nan"))
        out.append(
            f"| {pol} | {wu:.3f} | {wn:.3f} | {wo:.3f} |"
        )
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wss-json", type=Path, default=DEFAULT_WSS_JSON)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    ap.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    ap.add_argument("--threshold-pp", type=float, default=KNEE_THRESHOLD_PP)
    args = ap.parse_args()

    src_path = args.wss_json
    try:
        src_label = str(src_path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        src_label = str(src_path)

    payload = json.loads(src_path.read_text())
    result = build(payload, threshold=args.threshold_pp)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True))
    args.md_out.write_text(render_md(result, src_label))
    m = result["meta"]
    rows = " ".join(
        f"{p}={m['knee_regime_by_policy'][p] or 'never'}({m['knee_rank_by_policy'][p]})"
        for p in POLICIES
    )
    print(
        f"wss-knee-location: knee_threshold={m['knee_threshold_pp']:.2f}pp | "
        f"{rows} | "
        f"max_oracle={m['max_oracle_aware_knee_rank']} < "
        f"min_nonoracle={m['min_non_oracle_knee_rank']} | "
        f"verdict={m['verdict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
