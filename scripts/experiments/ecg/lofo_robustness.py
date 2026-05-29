#!/usr/bin/env python3
"""Leave-one-family-out (LOFO) winner robustness (gate 47).

The stronger sibling of gate 41 (LOGO). Instead of dropping one graph
at a time and re-ranking, LOFO drops an entire family (all social
graphs, all citation graphs, etc.) and asks whether the top-line
policy per app survives.

A LOFO-robust claim is structurally stronger than a LOGO-robust claim:
it says the result is not just 'not driven by one graph' but 'not even
driven by one whole family of graphs'. Reviewers can ask 'what if your
corpus over-indexes on social graphs?' — this gate gives an exact,
auditable answer.

Reads wiki/data/oracle_gap.json (paper L3 scope only: 1MB/4MB/8MB) and
computes, per app:

  * full-corpus top policy + win count
  * for each family, the top policy + win count AFTER excluding that family
  * fragile_family_drops: families whose removal flips the top policy
  * is_lofo_robust: True iff fragile_family_drops is empty

Output: wiki/data/lofo_robustness.{json,md}
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

PAPER_L3_SIZES = ("1MB", "4MB", "8MB")


def _resolve_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _top_policy_by_app(rows: list[dict]) -> dict[str, dict]:
    """Returns {app: {win_counts, top_policy, top_wins, runner_wins, unique_top, margin}}."""
    win_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r.get("is_winner") == "1":
            win_counts[r["app"]][r["policy"]] += 1

    out: dict[str, dict] = {}
    for app, c in win_counts.items():
        ordered = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
        top_policy, top_wins = ordered[0]
        runner_wins = ordered[1][1] if len(ordered) > 1 else 0
        out[app] = {
            "win_counts": dict(c),
            "top_policy": top_policy,
            "top_wins": top_wins,
            "runner_up_wins": runner_wins,
            "unique_top": top_wins > runner_wins,
            "margin": top_wins - runner_wins,
        }
    return out


def build_payload(oracle_path: Path) -> dict:
    oracle = json.loads(oracle_path.read_text())
    rows = [r for r in oracle["rows"] if r["l3_size"] in PAPER_L3_SIZES]

    families = sorted({r["family"] for r in rows})
    apps = sorted({r["app"] for r in rows})

    full = _top_policy_by_app(rows)

    per_app: dict[str, dict] = {}
    for app in apps:
        full_top = full[app]["top_policy"] if app in full else None
        drops_payload: dict[str, dict] = {}
        fragile_drops: list[str] = []
        for f in families:
            drop_rows = [r for r in rows if r["family"] != f]
            after = _top_policy_by_app(drop_rows)
            after_app = after.get(app)
            if after_app is None:
                drops_payload[f] = {"missing": True}
                continue
            same_winner = after_app["top_policy"] == full_top
            drops_payload[f] = {
                "top_policy": after_app["top_policy"],
                "top_wins":   after_app["top_wins"],
                "runner_up_wins": after_app["runner_up_wins"],
                "margin":     after_app["margin"],
                "unique_top": after_app["unique_top"],
                "same_winner_as_full": same_winner,
            }
            if not same_winner:
                fragile_drops.append(f)

        per_app[app] = {
            "full_corpus": full.get(app, {}),
            "drops": drops_payload,
            "n_drops": len(families),
            "n_robust_drops": len(families) - len(fragile_drops),
            "fragile_family_drops": fragile_drops,
            "is_lofo_robust": len(fragile_drops) == 0,
        }

    robust_apps = [a for a, p in per_app.items() if p["is_lofo_robust"]]
    fragile_apps = [a for a, p in per_app.items() if not p["is_lofo_robust"]]

    return {
        "meta": {
            "source": _resolve_label(oracle_path),
            "scope_l3_sizes": list(PAPER_L3_SIZES),
            "n_rows_in_scope": len(rows),
            "n_families": len(families),
            "families": families,
            "n_apps": len(apps),
            "apps": apps,
            "robust_apps": sorted(robust_apps),
            "fragile_apps": sorted(fragile_apps),
            "n_robust_apps": len(robust_apps),
            "n_fragile_apps": len(fragile_apps),
            "robustness_fraction": (
                round(len(robust_apps) / len(apps), 4) if apps else 0.0
            ),
        },
        "per_app": per_app,
    }


def emit_md(payload: dict) -> str:
    m = payload["meta"]
    out = []
    out.append("# Leave-one-family-out (LOFO) winner robustness")
    out.append("")
    out.append(
        f"Source: `{m['source']}`  •  Paper L3 scope: "
        f"{', '.join(m['scope_l3_sizes'])}"
    )
    out.append("")
    out.append(
        f"For each app, drop each of the {m['n_families']} families "
        f"({', '.join(m['families'])}) in turn and re-rank policy winners "
        "by cell count. A LOFO-robust claim has the same top policy after "
        "every family drop — no single family is driving the headline."
    )
    out.append("")
    out.append(
        f"**Headline:** {m['n_robust_apps']}/{m['n_apps']} apps "
        f"({m['robustness_fraction'] * 100:.0f}%) are LOFO-robust.  "
        f"Robust apps: {sorted(m['robust_apps'])}.  "
        f"Fragile apps: {sorted(m['fragile_apps'])}."
    )
    out.append("")
    out.append("## Per-app verdict")
    out.append("")
    out.append("| app | full top | full wins | LOFO-robust | fragile family drops |")
    out.append("|---|---|---:|---|---|")
    for app in sorted(payload["per_app"]):
        p = payload["per_app"][app]
        full = p["full_corpus"]
        fr = (
            ", ".join(p["fragile_family_drops"])
            if p["fragile_family_drops"]
            else "—"
        )
        out.append(
            f"| {app} | {full.get('top_policy', '?')} "
            f"| {full.get('top_wins', '?')} "
            f"| {'✅' if p['is_lofo_robust'] else '❌'} | {fr} |"
        )
    out.append("")
    out.append("## Per-app drop matrix")
    out.append("")
    fams = m["families"]
    out.append("| app | full top |" + "".join(f" drop-{f} |" for f in fams))
    out.append("|---|---|" + "".join("---|" for _ in fams))
    for app in sorted(payload["per_app"]):
        p = payload["per_app"][app]
        full = p["full_corpus"]
        cells = []
        for f in fams:
            d = p["drops"].get(f, {})
            if d.get("missing"):
                cells.append("MISSING")
            else:
                same = d.get("same_winner_as_full", False)
                cells.append(f"{d['top_policy']} ({d['top_wins']})" + ("" if same else " ⚠"))
        out.append(
            f"| {app} | {full.get('top_policy', '?')} ({full.get('top_wins', '?')}) | "
            + " | ".join(cells)
            + " |"
        )
    out.append("")
    out.append("## Interpretation")
    out.append("")
    out.append(
        "- LOFO is strictly stronger than LOGO (gate 41). Where LOGO drops"
        " one graph, LOFO drops 1–4 graphs (a whole family). Surviving LOFO"
        " is therefore a higher robustness bar."
    )
    out.append(
        "- Fragile apps under LOFO are honestly disclosed: the paper must"
        " qualify those headline policies as 'family-sensitive' or report"
        " family-stratified winners instead."
    )
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle-json", type=Path, default=WIKI_DATA / "oracle_gap.json"
    )
    parser.add_argument(
        "--json-out", type=Path, default=WIKI_DATA / "lofo_robustness.json"
    )
    parser.add_argument(
        "--md-out", type=Path, default=WIKI_DATA / "lofo_robustness.md"
    )
    args = parser.parse_args()

    payload = build_payload(args.oracle_json)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(emit_md(payload) + "\n")
    m = payload["meta"]
    print(
        f"lofo: apps={m['n_apps']} families={m['n_families']}"
        f" robust={m['n_robust_apps']}/{m['n_apps']}"
        f" ({m['robustness_fraction'] * 100:.0f}%)"
        f" fragile={m['fragile_apps']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
