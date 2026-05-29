"""Gate 63 — per-family replay of gate 62 (winner-margin shrinkage).

Gate 62 establishes the global claim that oracle-aware policies' winner
margins shrink as capacity loosens (under_wss → over_wss). Gate 63 asks
the same question one graph family at a time: does each qualifying
family independently exhibit the same margin-shrink pattern that the
global corpus exhibits?

For each family we compute the per-(policy, regime) winner-margin
distribution exactly as gate 62 does, but restricted to that family's
rows. A family is "qualifying" if it contributes ≥ 1 win in both the
under_wss and over_wss regimes for at least one oracle-aware policy
(otherwise the under-vs-over comparison is undefined for that family).

A qualifying family REPLAYS the global pattern iff at least one
oracle-aware policy has:
    median(margin | regime=under_wss) > median(margin | regime=over_wss)

Output schema:
  meta.qualifying_families        : families with both under+over wins
                                    for at least one oracle-aware policy
  meta.replay_count               : qualifying families that replay
  meta.deviating_families         : qualifying families that do NOT
  meta.pinned_deviating_families  : known/accepted deviations (empty today)
  meta.verdict                    : PASS iff no NEW deviation beyond pin
                                    AND replay_count ≥ 1
  per_family.<F>.qualifying       : bool
  per_family.<F>.replays          : bool
  per_family.<F>.per_policy_regime: same shape as gate 62 per-cell entries
  per_family.<F>.shrink_evidence  : list[{policy, under_median, over_median}]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ORACLE_JSON = REPO_ROOT / "wiki" / "data" / "oracle_gap.json"
DEFAULT_WSS_JSON = REPO_ROOT / "wiki" / "data" / "wss_relative_l3.json"
DEFAULT_JSON_OUT = REPO_ROOT / "wiki" / "data" / "family_margin_replay.json"
DEFAULT_MD_OUT = REPO_ROOT / "wiki" / "data" / "family_margin_replay.md"

REGIMES = ("under_wss", "near_wss", "over_wss")
POLICIES = ("GRASP", "LRU", "POPT", "SRRIP")
ORACLE_AWARE = ("GRASP", "POPT")

# Must match wss_relative_l3.py.
L3_BYTES = {
    "4kB":   4 * 1024,
    "16kB":  16 * 1024,
    "64kB":  64 * 1024,
    "256kB": 256 * 1024,
    "1MB":   1024 * 1024,
    "4MB":   4 * 1024 * 1024,
    "8MB":   8 * 1024 * 1024,
}

# Families that legitimately deviate from the global pattern. Today no
# qualifying family deviates; if a future corpus shift introduces one,
# pin it here with a short rationale comment.
PINNED_DEVIATING_FAMILIES: tuple[str, ...] = ()


def _classify(l3_bytes: float, wss_bytes: float) -> str:
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


def _family_stats(family_rows: list[dict], wss_proxies: dict) -> dict:
    """Reproduce gate 62 logic for a single family slice."""
    cells: dict = defaultdict(list)
    for r in family_rows:
        cells[(r["app"], r["graph"], r["l3_size"])].append(r)

    margins: dict = defaultdict(list)
    classified = 0
    skipped = 0
    for (app, graph, l3), rs in cells.items():
        if len(rs) != len(POLICIES):
            skipped += 1
            continue
        if graph not in wss_proxies or l3 not in L3_BYTES:
            skipped += 1
            continue
        regime = _classify(L3_BYTES[l3], wss_proxies[graph])
        miss = sorted([(float(r["miss_rate"]), r["policy"]) for r in rs])
        best_miss, best_pol = miss[0]
        second_miss, _ = miss[1]
        margins[(best_pol, regime)].append((second_miss - best_miss) * 100.0)
        classified += 1

    per_policy_regime: dict[str, dict] = {}
    for pol in POLICIES:
        for regime in REGIMES:
            xs = margins.get((pol, regime), [])
            per_policy_regime[f"{pol}/{regime}"] = {
                "policy":            pol,
                "wss_regime":        regime,
                "cells_won":         len(xs),
                "median_margin_pp":  round(_median(xs), 4),
                "mean_margin_pp":    round(sum(xs) / len(xs), 4) if xs else 0.0,
                "p90_margin_pp":     round(_pct(xs, 0.9), 4),
                "max_margin_pp":     round(max(xs), 4) if xs else 0.0,
            }

    # qualifying iff some oracle-aware policy has wins in BOTH under+over
    qualifies = False
    shrink_evidence = []
    for pol in ORACLE_AWARE:
        u_xs = margins.get((pol, "under_wss"), [])
        o_xs = margins.get((pol, "over_wss"), [])
        if u_xs and o_xs:
            qualifies = True
            u_med = _median(u_xs)
            o_med = _median(o_xs)
            if u_med > o_med:
                shrink_evidence.append({
                    "policy":        pol,
                    "under_median":  round(u_med, 4),
                    "over_median":   round(o_med, 4),
                })

    return {
        "cells_classified":   classified,
        "cells_skipped":      skipped,
        "qualifying":         qualifies,
        "replays":            qualifies and len(shrink_evidence) >= 1,
        "shrink_evidence":    shrink_evidence,
        "per_policy_regime":  per_policy_regime,
    }


def build(oracle_payload: dict, wss_meta: dict) -> dict:
    wss_proxies = wss_meta["wss_proxies"]
    by_family: dict = defaultdict(list)
    for r in oracle_payload["rows"]:
        by_family[r["family"]].append(r)

    per_family: dict[str, dict] = {}
    qualifying = []
    replays = []
    deviating = []
    for fam in sorted(by_family.keys()):
        stats = _family_stats(by_family[fam], wss_proxies)
        per_family[fam] = stats
        if stats["qualifying"]:
            qualifying.append(fam)
            if stats["replays"]:
                replays.append(fam)
            else:
                deviating.append(fam)

    new_deviating = [f for f in deviating if f not in PINNED_DEVIATING_FAMILIES]
    verdict = "PASS" if (len(replays) >= 1 and not new_deviating) else "FAIL"

    return {
        "meta": {
            "policies":                 list(POLICIES),
            "regimes":                  list(REGIMES),
            "oracle_aware":             list(ORACLE_AWARE),
            "qualifying_families":      qualifying,
            "replay_count":             len(replays),
            "replaying_families":       replays,
            "deviating_families":       deviating,
            "pinned_deviating_families": list(PINNED_DEVIATING_FAMILIES),
            "new_deviating_families":   new_deviating,
            "verdict":                  verdict,
            "verdict_invariant":        (
                "PASS iff at least one qualifying family replays the global "
                "margin-shrink pattern (some oracle-aware policy median "
                "under_wss > over_wss) AND no NEW family deviates beyond "
                "the pinned set"
            ),
        },
        "per_family": per_family,
    }


def render_md(result: dict, src_label: str) -> str:
    m = result["meta"]
    out = [
        "# Gate 63 — Per-family winner-margin replay",
        "",
        f"source: `{src_label}`",
        "",
        f"verdict: **{m['verdict']}**",
        "",
        f"  invariant: {m['verdict_invariant']}",
        "",
        f"qualifying families: {len(m['qualifying_families'])}"
        f" ({', '.join(m['qualifying_families']) or '-'})",
        "",
        f"replaying families: {m['replay_count']}"
        f" ({', '.join(m['replaying_families']) or '-'})",
        "",
        f"deviating families: {len(m['deviating_families'])}"
        f" ({', '.join(m['deviating_families']) or '-'})",
        "",
        f"pinned deviating: {len(m['pinned_deviating_families'])}"
        f" ({', '.join(m['pinned_deviating_families']) or '-'})",
        "",
        "## Per-family margin-shrink evidence",
        "",
        "| family | qualifying | replays | oracle-aware shrink summary |",
        "| --- | :---: | :---: | --- |",
    ]
    for fam in sorted(result["per_family"].keys()):
        s = result["per_family"][fam]
        ev = s["shrink_evidence"]
        if ev:
            ev_str = "; ".join(
                f"{e['policy']}: {e['under_median']:.3f}→{e['over_median']:.3f} pp"
                for e in ev
            )
        else:
            ev_str = "—"
        out.append(
            f"| {fam} | {'yes' if s['qualifying'] else 'no'} "
            f"| {'yes' if s['replays'] else 'no'} | {ev_str} |"
        )
    out.extend([
        "",
        "## Per-family cell counts",
        "",
        "| family | classified | skipped |",
        "| --- | ---: | ---: |",
    ])
    for fam in sorted(result["per_family"].keys()):
        s = result["per_family"][fam]
        out.append(
            f"| {fam} | {s['cells_classified']} | {s['cells_skipped']} |"
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
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True))
    args.md_out.write_text(render_md(result, src_label))
    m = result["meta"]
    print(
        f"family-margin-replay: qualifying={len(m['qualifying_families'])} "
        f"replays={m['replay_count']} "
        f"new_deviating={len(m['new_deviating_families'])} "
        f"verdict={m['verdict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
