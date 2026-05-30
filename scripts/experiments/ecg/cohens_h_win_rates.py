#!/usr/bin/env python3
"""Cohen's h effect sizes on policy win-rate differences.

Why this exists
---------------
Wilson CIs (gate 36) tell you *whether* a win-rate difference is
statistically separable. Cohen's h tells you how *big* that
difference is.

Effect-size thresholds (Cohen 1988):

* |h| < 0.2  — negligible
* |h| >= 0.2 — small
* |h| >= 0.5 — medium
* |h| >= 0.8 — large

A small CI-strict gap is not the same kind of evidence as a large
CI-strict gap, and papers should not present them with equal voice.
This gate pins which (app, policy_a, policy_b) comparisons carry
LARGE effect size so future regressions that flatten the policy
landscape will trip.

Why Cohen's h vs raw Δp
-----------------------
Cohen's h is variance-stabilizing through the arcsine transform.
A jump from 0.95→0.99 (Δp=0.04) is more discriminating than
0.50→0.54 (also Δp=0.04). h handles this correctly; Δp does not.

What we compute
---------------
For each app and each ordered pair (a, b) with a, b ∈ POLICIES
and a != b:

* p_a, p_b  — win rates from win-counts
* h         — 2*|arcsin(√p_a) − arcsin(√p_b)|  (Cohen 1988)
* magnitude — "negligible" | "small" | "medium" | "large"

Output
------
* ``wiki/data/cohens_h_win_rates.json``
* ``wiki/data/cohens_h_win_rates.md``
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "wiki" / "data"

POLICIES = ("LRU", "SRRIP", "GRASP", "POPT")
APPS = ("pr", "bc", "cc", "bfs", "sssp")

NEGLIGIBLE_FLOOR = 0.2
SMALL_FLOOR = 0.2
MEDIUM_FLOOR = 0.5
LARGE_FLOOR = 0.8


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h for two proportions; arcsine-transformed delta."""
    # Clamp to [0, 1] to avoid math domain issues on rounded inputs
    p1 = max(0.0, min(1.0, p1))
    p2 = max(0.0, min(1.0, p2))
    phi1 = 2.0 * math.asin(math.sqrt(p1))
    phi2 = 2.0 * math.asin(math.sqrt(p2))
    return abs(phi1 - phi2)


def magnitude(h: float) -> str:
    if h >= LARGE_FLOOR:
        return "large"
    if h >= MEDIUM_FLOOR:
        return "medium"
    if h >= SMALL_FLOOR:
        return "small"
    return "negligible"


def _is_winner(row: dict) -> bool:
    val = row.get("is_winner")
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return int(val) == 1
    if isinstance(val, str):
        return val.strip() in {"1", "true", "True"}
    return False


def load_oracle_rows(path: Path) -> list[dict]:
    blob = json.loads(path.read_text())
    if isinstance(blob, dict) and "rows" in blob:
        return blob["rows"]
    return blob


def aggregate(rows: list[dict]) -> dict:
    by_app = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        pol = r["policy"]
        app = r["app"]
        w = 1 if _is_winner(r) else 0
        by_app[app][pol][0] += w
        by_app[app][pol][1] += 1

    per_app = {}
    for app in sorted(by_app):
        rates = {}
        for pol in POLICIES:
            if pol not in by_app[app]:
                continue
            wins, total = by_app[app][pol]
            rates[pol] = {"wins": wins, "total": total, "p_hat": wins / total}

        comparisons = []
        for a in POLICIES:
            for b in POLICIES:
                if a == b or a not in rates or b not in rates:
                    continue
                p_a = rates[a]["p_hat"]
                p_b = rates[b]["p_hat"]
                h = cohens_h(p_a, p_b)
                comparisons.append(
                    {
                        "a": a,
                        "b": b,
                        "p_a": round(p_a, 4),
                        "p_b": round(p_b, 4),
                        "delta_p": round(p_a - p_b, 4),
                        "h": round(h, 4),
                        "magnitude": magnitude(h),
                        "favors": a if p_a > p_b else (b if p_b > p_a else "tie"),
                    }
                )

        per_app[app] = {
            "rates": {pol: {**v, "p_hat": round(v["p_hat"], 4)} for pol, v in rates.items()},
            "comparisons": comparisons,
        }

    # Headline collections
    largest_per_app = {}
    for app, payload in per_app.items():
        if not payload["comparisons"]:
            continue
        best = max(payload["comparisons"], key=lambda c: c["h"])
        largest_per_app[app] = best

    large_effects = [
        {"app": app, **c}
        for app, payload in per_app.items()
        for c in payload["comparisons"]
        if c["magnitude"] == "large" and c["p_a"] > c["p_b"]
    ]
    large_effects.sort(key=lambda r: -r["h"])

    return {
        "per_app": per_app,
        "largest_per_app": largest_per_app,
        "large_effects": large_effects,
    }


def render_md(payload: dict) -> str:
    meta = payload["meta"]
    lines = [
        "# Cohen's h on policy win-rate gaps",
        "",
        f"Source: `{meta['source']}` ({meta['n_rows']} rows).",
        "",
        "Thresholds (Cohen 1988): small ≥ 0.2, medium ≥ 0.5, large ≥ 0.8.",
        "",
        "## Largest effect per kernel",
        "",
        "| App | Favors | Comparison | p_a | p_b | h | magnitude |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for app in APPS:
        c = payload["largest_per_app"].get(app)
        if not c:
            continue
        lines.append(
            f"| {app} | {c['favors']} | {c['a']} vs {c['b']} | {c['p_a']:.3f} "
            f"| {c['p_b']:.3f} | {c['h']:.3f} | **{c['magnitude']}** |"
        )

    lines += [
        "",
        "## All large-effect dominance pairs (h ≥ 0.8, p_a > p_b)",
        "",
        "| App | Winner | Loser | p_winner | p_loser | h |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for r in payload["large_effects"]:
        lines.append(
            f"| {r['app']} | {r['a']} | {r['b']} | {r['p_a']:.3f} "
            f"| {r['p_b']:.3f} | {r['h']:.3f} |"
        )

    lines += ["", "## Per-app full table", ""]
    for app in APPS:
        if app not in payload["per_app"]:
            continue
        lines += [f"### {app}", "", "Win rates:", "", "| Policy | Wins / N | p̂ |", "| --- | ---: | ---: |"]
        for pol, stats in payload["per_app"][app]["rates"].items():
            lines.append(
                f"| {pol} | {stats['wins']} / {stats['total']} | {stats['p_hat']:.3f} |"
            )
        lines += ["", "Comparisons (h on ordered pairs):", "", "| a | b | Δp | h | magnitude |", "| --- | --- | ---: | ---: | --- |"]
        for c in payload["per_app"][app]["comparisons"]:
            lines.append(
                f"| {c['a']} | {c['b']} | {c['delta_p']:+.3f} | {c['h']:.3f} | {c['magnitude']} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle-json",
        default=str(DATA_DIR / "oracle_gap.json"),
    )
    parser.add_argument(
        "--json-out", default=str(DATA_DIR / "cohens_h_win_rates.json")
    )
    parser.add_argument("--md-out", default=str(DATA_DIR / "cohens_h_win_rates.md"))
    args = parser.parse_args()

    rows = load_oracle_rows(Path(args.oracle_json))
    agg = aggregate(rows)

    src_path = Path(args.oracle_json)
    try:
        src_label = str(src_path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        src_label = str(src_path)

    payload = {
        "meta": {
            "source": src_label,
            "n_rows": len(rows),
            "policies": list(POLICIES),
            "apps": list(APPS),
            "thresholds": {
                "small": SMALL_FLOOR,
                "medium": MEDIUM_FLOOR,
                "large": LARGE_FLOOR,
            },
        },
        **agg,
    }

    Path(args.json_out).write_text(json.dumps(payload, indent=2, sort_keys=True))
    Path(args.md_out).write_text(render_md(payload).rstrip("\n") + "\n")

    n_large = len(payload["large_effects"])
    print(
        f"[cohens-h] n_rows={len(rows)} | {n_large} large-effect dominance pairs "
        f"across {len(payload['per_app'])} apps → {args.md_out}"
    )


if __name__ == "__main__":
    main()
