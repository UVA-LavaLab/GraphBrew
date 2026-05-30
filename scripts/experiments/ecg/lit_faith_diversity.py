#!/usr/bin/env python3
"""Literature-faithfulness diversity audit.

The lit-faith comparator at ``wiki/data/literature_faithfulness_postfix.json``
tells us *whether* each cell agrees with the literature. It does not
report how well the corpus *spans* the literature: how many graph
families are exercised, how many cache sizes per claim, how many
distinct papers are cited, and where the literature corpus is thin.

This module produces a diversity audit:

* (family × app × L3 × policy) coverage matrix with counts.
* Per-paper claim counts (which paper is best/worst represented).
* Cross-paper triangulation cells: (graph, app, l3, policy) entries
  where ≥2 papers issue claims; we verify those claims share the same
  expected_sign so the literature itself doesn't internally disagree.
* Coverage gaps: cells below the per-axis floor that the pytest gate
  guards.

Emits ``wiki/data/lit_faith_diversity.{json,md,csv}``. Designed for the
``LIT-Cov`` confidence gate that locks coverage floors so future
literature additions can't silently drop a family or paper below the
diversity threshold.

CLI::

    python3 -m scripts.experiments.ecg.lit_faith_diversity \\
        --lit-faith-json wiki/data/literature_faithfulness_postfix.json \\
        --json-out wiki/data/lit_faith_diversity.json \\
        --md-out   wiki/data/lit_faith_diversity.md \\
        --csv-out  wiki/data/lit_faith_diversity.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LIT_FAITH = REPO_ROOT / "wiki" / "data" / "literature_faithfulness_postfix.json"
DEFAULT_JSON_OUT = REPO_ROOT / "wiki" / "data" / "lit_faith_diversity.json"
DEFAULT_MD_OUT = REPO_ROOT / "wiki" / "data" / "lit_faith_diversity.md"
DEFAULT_CSV_OUT = REPO_ROOT / "wiki" / "data" / "lit_faith_diversity.csv"

# Graph-family map mirrors the project-wide convention; kept here
# vendored so the audit doesn't tightly couple to the literature
# baselines module (which is dynamic-loaded elsewhere).
GRAPH_FAMILY: dict[str, str] = {
    "cit-Patents": "citation",
    "soc-pokec": "social",
    "soc-LiveJournal1": "social",
    "com-orkut": "social",
    "com-orkut-undir": "social",
    "web-Google": "web",
    "web-BerkStan": "web",
    "roadNet-CA": "road",
    "roadNet-TX": "road",
    "roadNet-PA": "road",
    "delaunay_n19": "mesh",
    "delaunay_n20": "mesh",
    "email-Eu-core": "social",
    "p2p-Gnutella31": "p2p",
}


def _paper_key(citation: str) -> str:
    """Reduce a free-form citation to a canonical paper tag.

    Matches the citation conventions used in ``literature_baselines.py``
    (Faldu HPCA20, Balaji HPCA21, Jaleel ISCA10). Cross-check citations
    of the form ``Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check``
    explode into both papers; this lets the diversity audit correctly
    credit shared-cell coverage to both papers.
    """
    if not citation:
        return "unknown"
    keys: list[str] = []
    low = citation.lower()
    if "faldu" in low:
        keys.append("Faldu HPCA20")
    if "balaji" in low:
        keys.append("Balaji HPCA21")
    if "jaleel" in low:
        keys.append("Jaleel ISCA10")
    if not keys:
        # Truncate to the first ~40 chars as a fallback bucket.
        return citation.split(";")[0].split(",")[0].strip()[:40]
    return " + ".join(keys)


def _papers_of(citation: str) -> list[str]:
    """Return the list of distinct paper tags credited by ``citation``."""
    if not citation:
        return ["unknown"]
    out: list[str] = []
    low = citation.lower()
    if "faldu" in low:
        out.append("Faldu HPCA20")
    if "balaji" in low:
        out.append("Balaji HPCA21")
    if "jaleel" in low:
        out.append("Jaleel ISCA10")
    if not out:
        out.append(citation.split(";")[0].split(",")[0].strip()[:40])
    return out


def build_audit(claims: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the diversity audit dict from the per_claim list."""
    by_family: Counter[str] = Counter()
    by_app: Counter[str] = Counter()
    by_l3: Counter[str] = Counter()
    by_policy: Counter[str] = Counter()
    by_graph: Counter[str] = Counter()
    by_paper: Counter[str] = Counter()
    by_status: Counter[str] = Counter()
    by_expected_sign: Counter[str] = Counter()

    cell_to_papers: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    cell_to_signs: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    family_app: Counter[tuple[str, str]] = Counter()
    family_l3: Counter[tuple[str, str]] = Counter()

    for c in claims:
        fam = GRAPH_FAMILY.get(c.get("graph", ""), "unknown")
        app = c.get("app", "?")
        l3 = c.get("l3_size", "?")
        pol = c.get("policy", "?")
        graph = c.get("graph", "?")
        status = c.get("status", "?")
        sign = c.get("expected_sign", "?")
        by_family[fam] += 1
        by_app[app] += 1
        by_l3[l3] += 1
        by_policy[pol] += 1
        by_graph[graph] += 1
        by_status[status] += 1
        by_expected_sign[sign] += 1
        family_app[(fam, app)] += 1
        family_l3[(fam, l3)] += 1
        cell = (graph, app, l3, pol)
        for paper in _papers_of(c.get("citation", "")):
            by_paper[paper] += 1
            cell_to_papers[cell].add(paper)
        cell_to_signs[cell].add(sign)

    triangulated = [
        {
            "graph": g,
            "app": a,
            "l3_size": l,
            "policy": p,
            "papers": sorted(papers),
            "n_papers": len(papers),
            "expected_signs": sorted(cell_to_signs[(g, a, l, p)]),
            "sign_consistent": len(cell_to_signs[(g, a, l, p)]) == 1,
        }
        for (g, a, l, p), papers in cell_to_papers.items()
        if len(papers) >= 2
    ]
    triangulated.sort(key=lambda x: (-x["n_papers"], x["graph"], x["app"], x["l3_size"], x["policy"]))

    sign_inconsistent = [t for t in triangulated if not t["sign_consistent"]]

    summary = {
        "claims_total": len(claims),
        "n_families": len(by_family),
        "n_apps": len(by_app),
        "n_l3_sizes": len(by_l3),
        "n_policies": len(by_policy),
        "n_graphs": len(by_graph),
        "n_papers": len(by_paper),
        "n_triangulated_cells": len(triangulated),
        "n_sign_inconsistent_cells": len(sign_inconsistent),
        "min_family_count": min(by_family.values()) if by_family else 0,
        "min_app_count": min(by_app.values()) if by_app else 0,
        "min_l3_count": min(by_l3.values()) if by_l3 else 0,
        "min_paper_count": min(by_paper.values()) if by_paper else 0,
        "min_graph_count": min(by_graph.values()) if by_graph else 0,
    }
    return {
        "summary": summary,
        "by_family": dict(sorted(by_family.items())),
        "by_app": dict(sorted(by_app.items())),
        "by_l3_size": dict(sorted(by_l3.items())),
        "by_policy": dict(sorted(by_policy.items())),
        "by_graph": dict(sorted(by_graph.items())),
        "by_paper": dict(sorted(by_paper.items())),
        "by_status": dict(sorted(by_status.items())),
        "by_expected_sign": dict(sorted(by_expected_sign.items())),
        "family_x_app": {
            f"{f}|{a}": n for (f, a), n in sorted(family_app.items())
        },
        "family_x_l3": {
            f"{f}|{l}": n for (f, l), n in sorted(family_l3.items())
        },
        "triangulated_cells": triangulated,
        "sign_inconsistent_cells": sign_inconsistent,
    }


def render_markdown(audit: dict[str, Any]) -> str:
    s = audit["summary"]
    out: list[str] = []
    out.append("# Literature-faithfulness diversity audit")
    out.append("")
    out.append(
        f"Audit of {s['claims_total']} per-claim entries from "
        "`wiki/data/literature_faithfulness_postfix.json`. The headline "
        "diversity numerics are produced by "
        "[`scripts/experiments/ecg/lit_faith_diversity.py`]"
        "(../../scripts/experiments/ecg/lit_faith_diversity.py) and "
        "locked by the `LIT-Cov` confidence gate."
    )
    out.append("")
    out.append("## Summary")
    out.append("")
    out.append("| Field | Value |")
    out.append("|---|---|")
    for k, v in s.items():
        out.append(f"| `{k}` | {v} |")
    out.append("")

    def _table(title: str, key: str) -> None:
        out.append(f"## {title}")
        out.append("")
        out.append("| Value | Count |")
        out.append("|---|---|")
        for k, v in audit[key].items():
            out.append(f"| `{k}` | {v} |")
        out.append("")

    _table("Claims by graph family", "by_family")
    _table("Claims by application", "by_app")
    _table("Claims by L3 size", "by_l3_size")
    _table("Claims by policy", "by_policy")
    _table("Claims by paper", "by_paper")
    _table("Claims by status", "by_status")
    _table("Claims by expected_sign", "by_expected_sign")

    out.append("## Cross-paper triangulation cells")
    out.append("")
    out.append(
        f"{s['n_triangulated_cells']} cells receive claims from "
        "≥ 2 distinct papers; "
        f"{s['n_sign_inconsistent_cells']} of them carry "
        "inconsistent `expected_sign` (a paper-vs-paper disagreement "
        "the corpus surfaces)."
    )
    out.append("")
    if audit["triangulated_cells"]:
        out.append("| graph | app | l3_size | policy | n_papers | papers | signs | consistent |")
        out.append("|---|---|---|---|---|---|---|---|")
        for t in audit["triangulated_cells"][:30]:
            out.append(
                f"| `{t['graph']}` | `{t['app']}` | `{t['l3_size']}` | "
                f"`{t['policy']}` | {t['n_papers']} | "
                f"{', '.join(t['papers'])} | {', '.join(t['expected_signs'])} | "
                f"{'✅' if t['sign_consistent'] else '❌'} |"
            )
        if len(audit["triangulated_cells"]) > 30:
            out.append(f"| _… {len(audit['triangulated_cells']) - 30} more_ |")
        out.append("")

    out.append("## Family × app density")
    out.append("")
    out.append("| family\\app | " + " | ".join(audit["by_app"].keys()) + " |")
    out.append("|---|" + "|".join(["---"] * len(audit["by_app"])) + "|")
    for fam in audit["by_family"]:
        row = [f"`{fam}`"]
        for app in audit["by_app"]:
            row.append(str(audit["family_x_app"].get(f"{fam}|{app}", 0)))
        out.append("| " + " | ".join(row) + " |")
    out.append("")

    return "\n".join(out).rstrip("\n") + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lit-faith-json", type=Path, default=DEFAULT_LIT_FAITH)
    p.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    p.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    p.add_argument("--csv-out", type=Path, default=DEFAULT_CSV_OUT)
    args = p.parse_args(argv)

    lf = json.loads(args.lit_faith_json.read_text(encoding="utf-8"))
    claims = lf.get("per_claim", [])
    audit = build_audit(claims)

    args.json_out.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.md_out.write_text(render_markdown(audit), encoding="utf-8")

    with args.csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["axis", "value", "count"])
        for axis in ("by_family", "by_app", "by_l3_size", "by_policy",
                     "by_paper", "by_graph", "by_status", "by_expected_sign"):
            for k, v in audit[axis].items():
                w.writerow([axis.removeprefix("by_"), k, v])

    print(
        f"[lit-faith-diversity] {audit['summary']['claims_total']} claims; "
        f"{audit['summary']['n_families']} families, "
        f"{audit['summary']['n_papers']} papers; "
        f"{audit['summary']['n_triangulated_cells']} triangulated; "
        f"{audit['summary']['n_sign_inconsistent_cells']} sign-inconsistent."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
