"""Gate 59 — per-(app, graph) Kendall-tau policy-rank correlation across L3 sizes.

For each (app, graph) cell with full 1MB / 4MB / 8MB coverage, rank
the four cache policies (GRASP, LRU, POPT, SRRIP) by miss_rate at each
L3 size. Then compute the Kendall-tau rank correlation between the rank
vectors at each L3 pair: (1MB ↔ 4MB), (4MB ↔ 8MB), (1MB ↔ 8MB).

A positive median tau says: the policy ranking at a small cache predicts
the ranking at a large cache. A negative or zero median says the
ranking flips with capacity — making 'best policy' a capacity-dependent
question rather than a global one. The paper's policy-winner table
(gate 30) implicitly assumes rank-stability across the L3 octave; this
gate puts that assumption under a number.

Output schema:
  meta.cells_total                : full-coverage (app, graph) cells
  meta.cell_pairs                 : list of (l3_a, l3_b) pairs scored
  meta.median_tau_by_pair         : median Kendall-tau per pair
  meta.flip_cells                 : cells where the 1MB↔8MB tau < 0
  meta.verdict                    : PASS iff median 1MB↔8MB tau > 0
  per_cell                        : per (app, graph) — three taus + ranks
  per_pair_summary                : mean/median/min tau per L3-pair
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ORACLE_JSON = REPO_ROOT / "wiki" / "data" / "oracle_gap.json"
DEFAULT_JSON_OUT = REPO_ROOT / "wiki" / "data" / "policy_rank_kendall.json"
DEFAULT_MD_OUT = REPO_ROOT / "wiki" / "data" / "policy_rank_kendall.md"

L3_SIZES = ("1MB", "4MB", "8MB")
POLICIES = ("GRASP", "LRU", "POPT", "SRRIP")
PAIRS = list(itertools.combinations(L3_SIZES, 2))  # 3 pairs

# Six cells flip the rank between 1MB and 8MB in today's corpus. These
# are not bugs — they encode three real phenomena that the gate should
# acknowledge rather than hide:
#
#   1) Small-cache GRASP thrash. At 1MB, GRASP ranks worst because its
#      hot-region pin costs cold misses; at 4MB+ the hot region fits
#      comfortably and the pin pays off. Symbol: rank@1MB=4 (worst),
#      rank@4MB=1 (best).  Cells: bc/cit-Patents, bc/web-Google,
#      pr/email-Eu-core.
#   2) Pico-corpus fit-in-cache. email-Eu-core is ~1k nodes; at 8MB the
#      working set fits entirely and the rank becomes noise.
#      Cells: bfs/email-Eu-core, pr/email-Eu-core (also in 1).
#   3) Large-cache oracle counter-productivity. When the WSS fits at 8MB,
#      oracle-pinning hurts because there is no replacement pressure.
#      Cells: cc/web-Google, sssp/soc-pokec.
#
# Verdict tracks "no NEW flip cells beyond this pin set" so any new graph
# that introduces additional rank-flip behavior is flagged.
PINNED_FLIP_CELLS: tuple[tuple[str, str], ...] = (
    ("bc",   "cit-Patents"),
    ("bc",   "web-Google"),
    ("bfs",  "email-Eu-core"),
    ("cc",   "web-Google"),
    ("pr",   "email-Eu-core"),
    ("sssp", "soc-pokec"),
)
PINNED_FLIP_CELLS_MAX = 6


def _kendall_tau(rank_a: list[int], rank_b: list[int]) -> float:
    """Standard tau-b for short discrete rank vectors with ties."""
    n = len(rank_a)
    assert n == len(rank_b)
    concordant = 0
    discordant = 0
    tie_a = 0
    tie_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = rank_a[i] - rank_a[j]
            db = rank_b[i] - rank_b[j]
            if da == 0 and db == 0:
                tie_a += 1
                tie_b += 1
            elif da == 0:
                tie_a += 1
            elif db == 0:
                tie_b += 1
            elif da * db > 0:
                concordant += 1
            else:
                discordant += 1
    n_pairs = n * (n - 1) / 2
    denom_a = n_pairs - tie_a
    denom_b = n_pairs - tie_b
    if denom_a <= 0 or denom_b <= 0:
        return 0.0
    return (concordant - discordant) / math.sqrt(denom_a * denom_b)


def _rank_vector(rows: list[dict]) -> list[int]:
    """Rank policies by miss_rate (lower = better = rank 1).

    Returns a 4-vector aligned with POLICIES order.
    """
    by_pol = {r["policy"]: float(r["miss_rate"]) for r in rows}
    # Get values in POLICIES order
    vals = [by_pol.get(p, math.inf) for p in POLICIES]
    # Sort ascending and assign ranks, breaking ties by mean rank
    indexed = sorted(range(len(vals)), key=lambda i: vals[i])
    rank = [0] * len(vals)
    i = 0
    r = 1
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and vals[indexed[j + 1]] == vals[indexed[i]]:
            j += 1
        mean_rank = (r + (r + (j - i))) / 2.0
        for k in range(i, j + 1):
            rank[indexed[k]] = mean_rank
        r += (j - i + 1)
        i = j + 1
    return rank


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def build(payload: dict) -> dict:
    rows = payload["rows"]
    # Bucket by (app, graph, l3)
    by_cell = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_cell[(r["app"], r["graph"])][r["l3_size"]].append(r)

    per_cell: dict[str, dict] = {}
    pair_to_taus: dict[tuple[str, str], list[float]] = {p: [] for p in PAIRS}
    flip_cells: list[tuple[str, str]] = []

    for (app, graph), by_l3 in sorted(by_cell.items()):
        if not all(l in by_l3 for l in L3_SIZES):
            continue
        ranks = {l3: _rank_vector(by_l3[l3]) for l3 in L3_SIZES}
        taus = {}
        for a, b in PAIRS:
            t = _kendall_tau(ranks[a], ranks[b])
            taus[f"{a}_vs_{b}"] = round(t, 4)
            pair_to_taus[(a, b)].append(t)
        if taus[f"{L3_SIZES[0]}_vs_{L3_SIZES[-1]}"] < 0:
            flip_cells.append((app, graph))
        per_cell.setdefault(app, {})[graph] = {
            "ranks_by_l3": {l3: [round(x, 2) for x in ranks[l3]] for l3 in L3_SIZES},
            "policies_order": list(POLICIES),
            "kendall_tau": taus,
        }

    per_pair_summary = {}
    for (a, b), taus in pair_to_taus.items():
        per_pair_summary[f"{a}_vs_{b}"] = {
            "n_cells":   len(taus),
            "mean_tau":  round(sum(taus) / len(taus), 4) if taus else 0.0,
            "median_tau": round(_median(taus), 4),
            "min_tau":   round(min(taus), 4) if taus else 0.0,
            "max_tau":   round(max(taus), 4) if taus else 0.0,
        }

    extremes_pair = f"{L3_SIZES[0]}_vs_{L3_SIZES[-1]}"
    median_extreme = per_pair_summary[extremes_pair]["median_tau"]
    flip_set = {tuple(c) for c in flip_cells}
    pinned_set = set(PINNED_FLIP_CELLS)
    new_flips = sorted(flip_set - pinned_set)
    verdict = "PASS" if (
        median_extreme > 0
        and not new_flips
        and len(flip_cells) <= PINNED_FLIP_CELLS_MAX
    ) else "FAIL"

    cells_total = sum(len(v) for v in per_cell.values())
    return {
        "meta": {
            "cells_total":          cells_total,
            "cell_pairs":           [f"{a}_vs_{b}" for a, b in PAIRS],
            "median_tau_by_pair":   {f"{a}_vs_{b}": per_pair_summary[f"{a}_vs_{b}"]["median_tau"] for a, b in PAIRS},
            "flip_cells":           flip_cells,
            "pinned_flip_cells":    list(PINNED_FLIP_CELLS),
            "new_flip_cells":       new_flips,
            "verdict":              verdict,
            "verdict_invariant":    (
                f"PASS iff median {extremes_pair} tau > 0 AND no NEW flip "
                f"cells beyond the {PINNED_FLIP_CELLS_MAX} pinned cells"
            ),
            "policy_order":         list(POLICIES),
            "l3_sizes":             list(L3_SIZES),
        },
        "per_pair_summary": per_pair_summary,
        "per_cell":         per_cell,
    }


def render_md(result: dict, src_label: str) -> str:
    m = result["meta"]
    out = [
        "# Gate 59 — Policy-rank Kendall-tau across L3 octave",
        "",
        f"source: `{src_label}`",
        "",
        f"verdict: **{m['verdict']}**",
        "",
        f"  invariant: {m['verdict_invariant']}",
        "",
        f"cells with full L3 coverage: **{m['cells_total']}**",
        "",
        "## Median Kendall-tau by L3 pair",
        "",
        "| L3 pair | n cells | mean τ | median τ | min τ | max τ |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pair, s in result["per_pair_summary"].items():
        out.append(
            f"| {pair} | {s['n_cells']} | {s['mean_tau']:.3f} | "
            f"{s['median_tau']:.3f} | {s['min_tau']:.3f} | {s['max_tau']:.3f} |"
        )
    out.extend([
        "",
        f"flip cells (1MB↔8MB τ<0): **{len(m['flip_cells'])}**"
        f"  pinned: {len(m['pinned_flip_cells'])}"
        f"  new: {len(m['new_flip_cells'])}",
        "",
        "## Per-cell rank vectors and tau",
        "",
        "Ranks are 1 = best (lowest miss-rate). Policy order: "
        + ", ".join(POLICIES),
        "",
        "| app | graph | rank@1MB | rank@4MB | rank@8MB "
        "| τ 1↔4 | τ 4↔8 | τ 1↔8 |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ])
    for app in sorted(result["per_cell"].keys()):
        for graph in sorted(result["per_cell"][app].keys()):
            c = result["per_cell"][app][graph]
            r = c["ranks_by_l3"]
            t = c["kendall_tau"]
            out.append(
                f"| {app} | {graph} "
                f"| {r['1MB']} | {r['4MB']} | {r['8MB']} "
                f"| {t['1MB_vs_4MB']:.2f} | {t['4MB_vs_8MB']:.2f} "
                f"| {t['1MB_vs_8MB']:.2f} |"
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
        f"policy-rank-kendall: cells={m['cells_total']} | "
        f"median τ {L3_SIZES[0]}↔{L3_SIZES[-1]}="
        f"{m['median_tau_by_pair'][f'{L3_SIZES[0]}_vs_{L3_SIZES[-1]}']:.3f} | "
        f"flips new={len(m['new_flip_cells'])} pinned="
        f"{len(m['pinned_flip_cells'])} | "
        f"verdict={m['verdict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
