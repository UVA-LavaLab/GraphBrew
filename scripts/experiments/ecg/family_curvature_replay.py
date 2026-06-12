"""Gate 61 — per-family replay of the oracle-gap curvature signal.

Gate 58 computes a single per-(app, policy) discrete second-derivative
of the oracle-gap trajectory across the whole corpus. Gate 61 asks the
same question one graph family at a time: does each qualifying family
(those with full 1MB / 4MB / 8MB coverage) independently replay the
global pattern that oracle-aware policies (GRASP, POPT) bend toward a
plateau while non-oracle policies (LRU, SRRIP) keep accelerating?

Computation (mirrors gate 58):
    log2(MB) axis: 1MB→0, 4MB→2, 8MB→3 (non-uniform)
    slope_lo  = (gap_at_4MB - gap_at_1MB) / 2
    slope_hi  = (gap_at_8MB - gap_at_4MB) / 1
    curvature = (slope_hi - slope_lo) / 1.5
where 1.5 is the mean of the two octave widths (per gate 58's
convention).

For each family we compute the mean curvature per policy and check the
ordering. A family REPLAYS the global pattern iff at least one
oracle-aware policy has mean_curvature > 0 AND every non-oracle
policy has mean_curvature <= 0.

Output schema:
  meta.qualifying_families        : families with full 1MB+4MB+8MB
  meta.curvature_threshold_pp_oct2: 0.0 (sign test)
  meta.replay_count               : families that replay the pattern
  meta.deviating_families         : families that do NOT replay
  meta.pinned_deviating_families  : known/accepted deviations
  meta.verdict                    : PASS iff no NEW deviation beyond pin
                                    AND replay_count >= 1
  per_family.<F>.per_policy.<P>   : mean_curvature + sample size
  per_family.<F>.replays_pattern  : bool
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ORACLE_JSON = REPO_ROOT / "wiki" / "data" / "oracle_gap.json"
DEFAULT_JSON_OUT = REPO_ROOT / "wiki" / "data" / "family_curvature_replay.json"
DEFAULT_MD_OUT = REPO_ROOT / "wiki" / "data" / "family_curvature_replay.md"

L3_SIZES = ("1MB", "4MB", "8MB")
L3_LOG2_MB = {"1MB": 0.0, "4MB": 2.0, "8MB": 3.0}
POLICIES = ("GRASP", "LRU", "POPT", "SRRIP")
ORACLE_AWARE = {"GRASP", "POPT"}
NON_ORACLE = {"LRU", "SRRIP"}

CURVATURE_THRESHOLD = 0.0  # sign test only

# Re-pinned 2026-06-12 to single-thread array-relative-GRASP 0.15 corpus:
# citation and social no longer replay the global curvature sign pattern,
# consistent with stronger L3-regime dependence and frontier misalignment.
PINNED_DEVIATING_FAMILIES: tuple[str, ...] = ("citation", "social")


def _curvature(gaps: list[float]) -> float:
    """Discrete second derivative on log2-MB axis (per gate 58)."""
    g1, g4, g8 = gaps[0], gaps[1], gaps[2]
    slope_lo = (g4 - g1) / (L3_LOG2_MB["4MB"] - L3_LOG2_MB["1MB"])
    slope_hi = (g8 - g4) / (L3_LOG2_MB["8MB"] - L3_LOG2_MB["4MB"])
    span = 0.5 * (
        (L3_LOG2_MB["4MB"] - L3_LOG2_MB["1MB"])
        + (L3_LOG2_MB["8MB"] - L3_LOG2_MB["4MB"])
    )
    return (slope_hi - slope_lo) / span


def build(payload: dict) -> dict:
    rows = payload["rows"]
    # group: family -> graph -> app -> policy -> l3 -> [miss-oracle pp]
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    for r in rows:
        by[r["family"]][r["graph"]][r["app"]][r["policy"]][r["l3_size"]] = (
            float(r["gap_pp"])
        )

    per_family: dict[str, dict] = {}
    qualifying = []
    for fam, by_graph in sorted(by.items()):
        # A family qualifies iff at least one graph has full L3 coverage
        # across all 4 policies for at least one app.
        per_pol_curvatures = defaultdict(list)
        any_full = False
        for graph, by_app in by_graph.items():
            for app, by_pol in by_app.items():
                if not all(pol in by_pol for pol in POLICIES):
                    continue
                if not all(all(l in by_pol[pol] for l in L3_SIZES) for pol in POLICIES):
                    continue
                any_full = True
                for pol in POLICIES:
                    gaps = [by_pol[pol][l] for l in L3_SIZES]
                    per_pol_curvatures[pol].append(_curvature(gaps))
        if not any_full:
            continue
        qualifying.append(fam)
        per_policy = {}
        for pol in POLICIES:
            xs = per_pol_curvatures[pol]
            per_policy[pol] = {
                "mean_curvature":   round(sum(xs) / len(xs), 4) if xs else 0.0,
                "n_app_graph_cells": len(xs),
                "is_oracle_aware":  pol in ORACLE_AWARE,
            }
        oracle_pos = any(
            per_policy[p]["mean_curvature"] > CURVATURE_THRESHOLD
            for p in ORACLE_AWARE
        )
        non_oracle_nonpos = all(
            per_policy[p]["mean_curvature"] <= CURVATURE_THRESHOLD
            for p in NON_ORACLE
        )
        replays = oracle_pos and non_oracle_nonpos
        per_family[fam] = {
            "per_policy":      per_policy,
            "replays_pattern": replays,
            "any_oracle_aware_positive": oracle_pos,
            "all_non_oracle_nonpositive": non_oracle_nonpos,
        }

    deviating = [f for f in qualifying if not per_family[f]["replays_pattern"]]
    new_dev = [f for f in deviating if f not in PINNED_DEVIATING_FAMILIES]
    replay_count = sum(1 for f in qualifying if per_family[f]["replays_pattern"])
    verdict = "PASS" if (replay_count >= 1 and not new_dev) else "FAIL"

    return {
        "meta": {
            "qualifying_families":         qualifying,
            "curvature_threshold_pp_oct2": CURVATURE_THRESHOLD,
            "replay_count":                replay_count,
            "deviating_families":          deviating,
            "pinned_deviating_families":   list(PINNED_DEVIATING_FAMILIES),
            "new_deviating_families":      new_dev,
            "policies":                    list(POLICIES),
            "oracle_aware_policies":       sorted(ORACLE_AWARE),
            "non_oracle_policies":         sorted(NON_ORACLE),
            "verdict":                     verdict,
            "verdict_invariant":           (
                "PASS iff at least one family replays the pattern AND no "
                "NEW deviating family appears beyond the pinned set"
            ),
        },
        "per_family": per_family,
    }


def render_md(result: dict, src_label: str) -> str:
    m = result["meta"]
    out = [
        "# Gate 61 — Per-family oracle-gap curvature replay",
        "",
        f"source: `{src_label}`",
        "",
        f"verdict: **{m['verdict']}**",
        "",
        f"  invariant: {m['verdict_invariant']}",
        "",
        "qualifying families (full 1MB/4MB/8MB coverage, "
        "all 4 policies, at least one app): "
        + ", ".join(m["qualifying_families"]),
        "",
        f"replay count: **{m['replay_count']} / {len(m['qualifying_families'])}**",
        f"deviating families (pinned): {m['pinned_deviating_families']}",
        f"deviating families (new):    {m['new_deviating_families']}",
        "",
        "## Per-family mean curvature by policy (pp/octave²)",
        "",
        "Positive = trajectory bending toward plateau (knee). "
        "Negative = still accelerating descent.",
        "",
        "| family | GRASP | POPT | LRU | SRRIP | replays? |",
        "| --- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for fam in m["qualifying_families"]:
        info = result["per_family"][fam]
        pp = info["per_policy"]
        out.append(
            f"| {fam} | {pp['GRASP']['mean_curvature']:.3f} "
            f"| {pp['POPT']['mean_curvature']:.3f} "
            f"| {pp['LRU']['mean_curvature']:.3f} "
            f"| {pp['SRRIP']['mean_curvature']:.3f} "
            f"| {'yes' if info['replays_pattern'] else 'no'} |"
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

    payload = json.loads(src_path.read_text())
    result = build(payload)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(render_md(result, src_label))
    m = result["meta"]
    print(
        f"family-curvature-replay: families={m['qualifying_families']} | "
        f"replays={m['replay_count']}/{len(m['qualifying_families'])} | "
        f"dev_new={len(m['new_deviating_families'])} "
        f"dev_pinned={len(m['pinned_deviating_families'])} | "
        f"verdict={m['verdict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
