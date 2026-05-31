#!/usr/bin/env python3
"""Leave-one-graph-out (LOGO) winner robustness.

The single most paper-defensive sensitivity analysis: drop each graph
in turn and re-rank winners. A claim that survives EVERY drop is
'LOGO-robust' — no single graph is driving the headline. A claim that
flips winner under any drop is 'LOGO-fragile' and must be qualified
(or restated) in the paper text.

Reads wiki/data/oracle_gap.json (the same authoritative cell-level
oracle-gap matrix the rest of the gates use) and computes:

  * For each app: full-corpus top policy and win count.
  * For each (app, dropped_graph): top policy and win count after
    excluding that graph's cells.
  * Robustness flag: True iff top policy is preserved across all drops.
  * Per-app list of 'fragile drops' that change the top.

Anchors the paper claim: 'all top-line winners survive LOGO except
sssp, which is honestly fragile and reported as such.'
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WIKI_DATA = REPO_ROOT / "wiki" / "data"


def _resolve_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _top_policy_by_app(rows: list[dict]) -> dict[str, dict]:
    """Returns {app: {policy: wins, top_policy, top_wins, unique_top, margin}}."""
    win_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r.get("is_winner") == "1":
            win_counts[r["app"]][r["policy"]] += 1

    out: dict[str, dict] = {}
    for app, c in win_counts.items():
        ordered = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
        top_policy, top_wins = ordered[0]
        runner_wins = ordered[1][1] if len(ordered) > 1 else 0
        unique_top = top_wins > runner_wins
        out[app] = {
            "win_counts": dict(c),
            "top_policy": top_policy,
            "top_wins": top_wins,
            "runner_up_wins": runner_wins,
            "unique_top": unique_top,
            "margin": top_wins - runner_wins,
        }
    return out


def build_payload(oracle_path: Path) -> dict:
    oracle = json.loads(oracle_path.read_text())
    rows = oracle["rows"]
    graphs = sorted({r["graph"] for r in rows})
    apps = sorted({r["app"] for r in rows})

    full = _top_policy_by_app(rows)

    per_app: dict[str, dict] = {}
    for app in apps:
        full_top = full[app]["top_policy"] if app in full else None
        drops_payload: dict[str, dict] = {}
        fragile_drops: list[str] = []
        for g in graphs:
            drop_rows = [r for r in rows if r["graph"] != g]
            after = _top_policy_by_app(drop_rows)
            after_app = after.get(app)
            if after_app is None:
                drops_payload[g] = {"missing": True}
                continue
            same_winner = (
                after_app["top_policy"] == full_top
                and after_app["unique_top"]
            )
            drops_payload[g] = {
                "top_policy": after_app["top_policy"],
                "top_wins":   after_app["top_wins"],
                "margin":     after_app["margin"],
                "unique_top": after_app["unique_top"],
                "same_winner_as_full": same_winner,
            }
            if not same_winner:
                fragile_drops.append(g)

        per_app[app] = {
            "full_corpus":   full.get(app, {}),
            "drops":         drops_payload,
            "n_drops":       len(graphs),
            "n_robust_drops": len(graphs) - len(fragile_drops),
            "fragile_drops": fragile_drops,
            "is_logo_robust": len(fragile_drops) == 0,
        }

    robust_apps = [a for a, p in per_app.items() if p["is_logo_robust"]]
    fragile_apps = [a for a, p in per_app.items() if not p["is_logo_robust"]]

    return {
        "meta": {
            "n_rows":         len(rows),
            "n_graphs":       len(graphs),
            "graphs":         graphs,
            "apps":           apps,
            "robust_apps":    sorted(robust_apps),
            "fragile_apps":   sorted(fragile_apps),
            "n_robust_apps":  len(robust_apps),
            "n_fragile_apps": len(fragile_apps),
        },
        "per_app": per_app,
    }


def write_md(payload: dict, md_path: Path) -> None:
    m = payload["meta"]
    lines = [
        "# Leave-one-graph-out (LOGO) winner robustness",
        "",
        "For each application, drop each graph in turn and re-rank policy",
        "winners by cell count. A LOGO-robust claim has the same top",
        "policy after every drop — no single graph is driving the headline.",
        "",
        f"- Graphs in corpus: **{m['n_graphs']}** "
        f"({', '.join(f'`{g}`' for g in m['graphs'])})",
        f"- LOGO-robust applications: **{m['n_robust_apps']}** "
        f"(`{', '.join(m['robust_apps']) or '—'}`)",
        f"- LOGO-fragile applications: **{m['n_fragile_apps']}** "
        f"(`{', '.join(m['fragile_apps']) or '—'}`)",
        "",
        "## Per-app summary",
        "",
        "| App | Full-corpus top | Wins | Robust? | Fragile drops |",
        "| :-- | :-------------- | ---: | :------ | :------------ |",
    ]
    for app, p in sorted(payload["per_app"].items()):
        full = p["full_corpus"]
        robust = "✓ ROBUST" if p["is_logo_robust"] else "✗ FRAGILE"
        fragile_str = ", ".join(f"`{g}`" for g in p["fragile_drops"]) or "—"
        lines.append(
            f"| `{app}` | `{full.get('top_policy', '—')}` | "
            f"{full.get('top_wins', 0)} | {robust} | {fragile_str} |"
        )

    lines += ["", "## Per-(app, drop) detail", ""]
    for app, p in sorted(payload["per_app"].items()):
        full = p["full_corpus"]
        lines += [
            f"### `{app}` — full top: `{full.get('top_policy', '—')}` "
            f"({full.get('top_wins', 0)} wins)",
            "",
            "| Dropped graph | Top after drop | Wins | Margin | Same? |",
            "| :------------ | :------------- | ---: | -----: | :---- |",
        ]
        for g, d in sorted(p["drops"].items()):
            if d.get("missing"):
                lines.append(f"| `{g}` | (no rows for {app}) | — | — | — |")
                continue
            mark = "✓" if d["same_winner_as_full"] else "✗"
            lines.append(
                f"| `{g}` | `{d['top_policy']}` | {d['top_wins']} "
                f"| {d['margin']} | {mark} |"
            )
        lines.append("")

    md_path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oracle-json", default=str(WIKI_DATA / "oracle_gap.json"))
    ap.add_argument("--json-out",
                    default=str(WIKI_DATA / "leave_one_graph_out.json"))
    ap.add_argument("--md-out",
                    default=str(WIKI_DATA / "leave_one_graph_out.md"))
    args = ap.parse_args()

    payload = build_payload(Path(args.oracle_json))

    json_path = Path(args.json_out)
    md_path = Path(args.md_out)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_md(payload, md_path)

    m = payload["meta"]
    print(
        f"[logo-robust] n_rows={m['n_rows']} n_graphs={m['n_graphs']} "
        f"robust={m['n_robust_apps']}/{len(m['apps'])} "
        f"fragile={m['fragile_apps']} → {_resolve_label(md_path)}"
    )


if __name__ == "__main__":
    main()
