"""Gate 77 — per-tool L3 sweep monotonicity (cache-sim).

Invariant: for every (graph, app, policy) cell with multiple L3
measurements, miss_rate must be monotone-non-increasing as L3 grows
within a documented measurement-noise tolerance.

Reads wiki/data/oracle_gap.json (the load-bearing raw cache-sim
sweep), groups rows by (graph, app, policy), sorts by L3 bytes, and
checks every consecutive (L_i, L_{i+1}) pair. A step is a "bump" iff
miss_rate(L_{i+1}) > miss_rate(L_i).

Bumps below MAX_NOISE_BUMP_PP are measurement noise; bumps at or
above the threshold are real non-monotone behaviour and cause FAIL.

Why this gate matters:

  - Cache monotonicity (more cache cannot hurt for the optimal
    policy) is a foundational soundness check. A real-policy run
    can in principle exhibit small non-monotone steps due to
    sampling, warmup, or replacement-policy edge effects, but a
    large step (>= 0.5 pp) would indicate either a bug in the
    simulator, a corrupted sweep, or a fundamentally broken policy.
  - Gates 65-68, 70-76 all assume the L3 sweep is monotone in
    aggregate (slopes, distances, sensitivity). This gate locks
    that assumption at the cell level.

Current state on the 8-graph corpus (cache-sim, 4 policies, 5 apps):

  136 cells with >=2 L3 points.
  320 (L_i, L_{i+1}) steps total.
  14 steps are bumps (4.4%); largest bump 0.035 pp (well below
  the 0.5 pp noise tolerance).
  Zero hard violations.

Output schema:
  meta.source_artifact
  meta.l3_axis_bytes              : sorted list of L3 sizes observed (bytes)
  meta.l3_axis_labels
  meta.max_noise_bump_pp          : tolerance threshold
  meta.cell_count                 : cells with >= 2 L3 points
  meta.total_step_count           : total consecutive-pair steps
  meta.bump_count                 : steps with delta > 0 (any size)
  meta.bump_pct                   : bump_count / total_step_count
  meta.hard_violation_count       : steps with delta >= MAX_NOISE_BUMP_PP
  meta.largest_bump_pp
  meta.largest_bump_cell          : {graph, app, policy, l3_from, l3_to, delta_pp}
  meta.bump_pct_ceiling           : max allowed bump_pct
  meta.bumps                      : all bumps (any size), sorted by magnitude
  meta.verdict_checks             : dict[str -> bool]
  meta.verdict                    : PASS / FAIL
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ORACLE_JSON = REPO_ROOT / "wiki" / "data" / "oracle_gap.json"
DEFAULT_JSON_OUT    = REPO_ROOT / "wiki" / "data" / "monotonicity_universality.json"
DEFAULT_MD_OUT      = REPO_ROOT / "wiki" / "data" / "monotonicity_universality.md"

# A bump (miss_rate increase across an L3 step) at or above this
# magnitude in percentage points indicates a real non-monotone
# behaviour. Below this is treated as measurement noise. The current
# worst-observed bump is 0.035 pp on a small graph (email-Eu-core),
# leaving ~14x margin against this 0.5 pp ceiling.
MAX_NOISE_BUMP_PP = 0.5

# Even noise-level bumps shouldn't dominate the sweep. Current
# observed bump rate is 4.4%; gate trips above 10%.
BUMP_PCT_CEILING = 0.10


def _l3_bytes(label: str) -> int:
    n = int(label[:-2])
    unit = label[-2:]
    if unit == "kB":
        return n * 1024
    if unit == "MB":
        return n * 1024 * 1024
    raise ValueError(f"unknown L3 unit in label: {label!r}")


def build(oracle_path: Path) -> dict:
    doc = json.loads(oracle_path.read_text())
    rows = doc["rows"]

    cells: dict[tuple, list[tuple[int, str, float]]] = defaultdict(list)
    l3_labels_seen: set[str] = set()
    for r in rows:
        key = (r["graph"], r["app"], r["policy"])
        label = r["l3_size"]
        bytes_ = _l3_bytes(label)
        miss = float(r["miss_rate"])
        cells[key].append((bytes_, label, miss))
        l3_labels_seen.add(label)

    l3_axis_labels = sorted(l3_labels_seen, key=_l3_bytes)
    l3_axis_bytes  = [_l3_bytes(l) for l in l3_axis_labels]

    bumps: list[dict] = []
    total_steps = 0
    cell_count_ok = 0
    largest_bump_pp = 0.0
    largest_bump_cell: dict | None = None

    for (graph, app, policy), pts in cells.items():
        if len(pts) < 2:
            continue
        cell_count_ok += 1
        s = sorted(pts, key=lambda x: x[0])
        for i in range(len(s) - 1):
            total_steps += 1
            delta_fraction = s[i + 1][2] - s[i][2]
            delta_pp = delta_fraction * 100.0
            if delta_pp > 0.0:
                entry = {
                    "graph":    graph,
                    "app":      app,
                    "policy":   policy,
                    "l3_from":  s[i][1],
                    "l3_to":    s[i + 1][1],
                    "delta_pp": round(delta_pp, 6),
                }
                bumps.append(entry)
                if delta_pp > largest_bump_pp:
                    largest_bump_pp = delta_pp
                    largest_bump_cell = entry

    bumps.sort(key=lambda e: -e["delta_pp"])
    bump_count = len(bumps)
    bump_pct = (bump_count / total_steps) if total_steps else 0.0
    hard_violation_count = sum(1 for b in bumps if b["delta_pp"] >= MAX_NOISE_BUMP_PP)

    inv_no_hard_violations  = (hard_violation_count == 0)
    inv_bump_pct_under_ceiling = (bump_pct <= BUMP_PCT_CEILING)
    inv_largest_below_tolerance = (largest_bump_pp < MAX_NOISE_BUMP_PP)

    verdict_checks = {
        "no_hard_violations":        inv_no_hard_violations,
        "bump_pct_under_ceiling":    inv_bump_pct_under_ceiling,
        "largest_bump_within_noise": inv_largest_below_tolerance,
    }
    verdict = "PASS" if all(verdict_checks.values()) else "FAIL"

    return {
        "meta": {
            "source_artifact":      str(oracle_path.relative_to(REPO_ROOT))
                                     if oracle_path.is_absolute() else str(oracle_path),
            "l3_axis_bytes":        l3_axis_bytes,
            "l3_axis_labels":       l3_axis_labels,
            "max_noise_bump_pp":    MAX_NOISE_BUMP_PP,
            "bump_pct_ceiling":     BUMP_PCT_CEILING,
            "cell_count":           cell_count_ok,
            "total_step_count":     total_steps,
            "bump_count":           bump_count,
            "bump_pct":             round(bump_pct, 6),
            "hard_violation_count": hard_violation_count,
            "largest_bump_pp":      round(largest_bump_pp, 6),
            "largest_bump_cell":    largest_bump_cell,
            "bumps":                bumps,
            "verdict_checks":       verdict_checks,
            "verdict":              verdict,
        },
    }


def render_md(payload: dict) -> str:
    m = payload["meta"]
    lines = [
        "# L3-sweep monotonicity universality (cache-sim)",
        "",
        f"**Verdict:** {m['verdict']}  ",
        f"**Source:** `{m['source_artifact']}`  ",
        f"**L3 axis:** {', '.join(m['l3_axis_labels'])}  ",
        f"**Cells (>=2 L3 points):** {m['cell_count']}  ",
        f"**Total steps:** {m['total_step_count']}  ",
        f"**Bumps:** {m['bump_count']} ({m['bump_pct']*100:.2f}%) "
        f"ceiling {m['bump_pct_ceiling']*100:.0f}%  ",
        f"**Largest bump:** {m['largest_bump_pp']:.4f} pp "
        f"(noise tolerance {m['max_noise_bump_pp']:.2f} pp)  ",
        f"**Hard violations (>= {m['max_noise_bump_pp']} pp):** {m['hard_violation_count']}",
        "",
        "## Verdict checks",
        "",
        "| check | result |",
        "|---|---|",
    ]
    for k, v in m["verdict_checks"].items():
        lines.append(f"| {k} | {'✅' if v else '❌'} |")

    lines += ["", "## Largest bump (worst-case cell)", ""]
    if m["largest_bump_cell"]:
        c = m["largest_bump_cell"]
        lines.append(
            f"`{c['graph']}` / `{c['app']}` / `{c['policy']}`: "
            f"{c['l3_from']} -> {c['l3_to']} delta = {c['delta_pp']:+.4f} pp"
        )
    else:
        lines.append("_None — every step is monotone non-increasing._")

    lines += ["", "## All bumps (sorted by magnitude)", ""]
    if m["bumps"]:
        lines += [
            "| graph | app | policy | from | to | delta pp |",
            "|---|---|---|---|---|---:|",
        ]
        for b in m["bumps"]:
            lines.append(
                f"| {b['graph']} | {b['app']} | {b['policy']} | "
                f"{b['l3_from']} | {b['l3_to']} | {b['delta_pp']:+.4f} |"
            )
    else:
        lines.append("_None._")

    lines += [
        "",
        "## Interpretation",
        "",
        "Cache monotonicity (more cache cannot hurt) is a foundational "
        "soundness check that downstream slope/distance/sensitivity "
        "gates rely on. Small noise-level bumps (<0.5 pp) are expected "
        "from sampling and warmup; any larger bump would indicate a "
        "simulator bug, corrupted sweep, or pathological policy.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-json", type=Path, default=DEFAULT_ORACLE_JSON)
    ap.add_argument("--json-out",    type=Path, default=DEFAULT_JSON_OUT)
    ap.add_argument("--md-out",      type=Path, default=DEFAULT_MD_OUT)
    args = ap.parse_args()

    payload = build(args.oracle_json)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    args.md_out.write_text(render_md(payload))

    m = payload["meta"]
    print(
        f"monotonicity-universality: "
        f"cells={m['cell_count']} steps={m['total_step_count']} "
        f"bumps={m['bump_count']} ({m['bump_pct']*100:.2f}%) "
        f"largest={m['largest_bump_pp']:.4f}pp "
        f"hard_violations={m['hard_violation_count']} "
        f"verdict={m['verdict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
