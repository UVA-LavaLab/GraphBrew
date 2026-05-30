#!/usr/bin/env python3
"""Corpus tier/family balance audit (gate 45).

Defends against the reviewer claim 'your corpus is unbalanced toward X
family' by pinning the actual composition and computing simple balance
metrics. Surfaces:

  - n_graphs per family (8 graphs across 5 families)
  - n_paper_l3_cells per family (1MB/4MB/8MB)
  - n_paper_l3_cells per app
  - per-family L3 coverage matrix (which families reach 4MB/8MB)
  - Shannon entropy + Simpson's index on (family) and (app) distributions
  - dominant family + dominance fraction
  - honest tier disclosures: small graphs (email-Eu-core), road/mesh
    L3 caps, etc.

Output: wiki/data/corpus_balance.{json,md}
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

PAPER_L3_SIZES = ("1MB", "4MB", "8MB")


def shannon_entropy_bits(counts: dict) -> float:
    """H = -sum(p_i * log2(p_i)) in bits."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for n in counts.values():
        if n == 0:
            continue
        p = n / total
        h -= p * math.log2(p)
    return h


def simpsons_index(counts: dict) -> float:
    """D = 1 - sum(p_i^2). 0 = no diversity, 1 = max diversity."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return 1.0 - sum((n / total) ** 2 for n in counts.values())


def build_payload(oracle_path: Path) -> dict:
    raw = json.loads(oracle_path.read_text())
    rows = raw["rows"]

    # graph -> family map
    g2f: dict[str, str] = {}
    for r in rows:
        g = r["graph"]
        fam = r.get("family")
        if fam:
            g2f[g] = fam

    # per-family graphs
    fam_graphs: dict[str, set[str]] = defaultdict(set)
    for g, f in g2f.items():
        fam_graphs[f].add(g)

    # paper-L3 cells per family
    fam_l3_cells: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        if r["l3_size"] in PAPER_L3_SIZES:
            fam_l3_cells[(r.get("family"), r["l3_size"])] += 1

    # paper-L3 cells per app
    app_l3_cells: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        if r["l3_size"] in PAPER_L3_SIZES:
            app_l3_cells[(r["app"], r["l3_size"])] += 1

    # per-family L3 coverage (which paper L3 sizes reached)
    fam_l3_coverage: dict[str, list[str]] = defaultdict(list)
    for (fam, l3) in fam_l3_cells:
        fam_l3_coverage[fam].append(l3)
    fam_l3_coverage = {k: sorted(set(v)) for k, v in fam_l3_coverage.items()}

    # totals
    n_graphs_per_family = {f: len(gs) for f, gs in fam_graphs.items()}
    n_cells_per_family = {
        f: sum(c for (ff, _), c in fam_l3_cells.items() if ff == f)
        for f in fam_graphs
    }
    n_cells_per_app = {
        a: sum(c for (aa, _), c in app_l3_cells.items() if aa == a)
        for a in {r["app"] for r in rows}
    }

    # diversity metrics
    fam_entropy_bits = shannon_entropy_bits(n_graphs_per_family)
    fam_entropy_max = math.log2(len(n_graphs_per_family)) if n_graphs_per_family else 0
    fam_evenness = (fam_entropy_bits / fam_entropy_max) if fam_entropy_max > 0 else 0.0
    fam_simpson = simpsons_index(n_graphs_per_family)

    app_entropy_bits = shannon_entropy_bits(n_cells_per_app)
    app_entropy_max = math.log2(len(n_cells_per_app)) if n_cells_per_app else 0
    app_evenness = (
        (app_entropy_bits / app_entropy_max) if app_entropy_max > 0 else 0.0
    )
    app_simpson = simpsons_index(n_cells_per_app)

    # dominance
    total_graphs = sum(n_graphs_per_family.values())
    dom_family, dom_n = max(n_graphs_per_family.items(), key=lambda kv: kv[1])
    dom_family_fraction = dom_n / total_graphs if total_graphs else 0.0

    total_cells = sum(n_cells_per_family.values())
    dom_family_by_cells, dom_n_cells = max(
        n_cells_per_family.items(), key=lambda kv: kv[1]
    )
    dom_family_cell_fraction = dom_n_cells / total_cells if total_cells else 0.0

    # honest disclosures
    families_capped_below_4mb = sorted(
        f for f, l3s in fam_l3_coverage.items() if "4MB" not in l3s
    )
    families_capped_below_8mb = sorted(
        f for f, l3s in fam_l3_coverage.items() if "8MB" not in l3s
    )
    families_reaching_8mb = sorted(
        f for f, l3s in fam_l3_coverage.items() if "8MB" in l3s
    )

    try:
        src_label = str(oracle_path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        src_label = str(oracle_path)

    payload = {
        "meta": {
            "source": src_label,
            "scope_l3_sizes": list(PAPER_L3_SIZES),
            "n_graphs": total_graphs,
            "n_families": len(fam_graphs),
            "n_apps": len(n_cells_per_app),
            "n_paper_l3_cells_total": total_cells,
            "shannon_entropy_graphs_per_family_bits": round(fam_entropy_bits, 4),
            "shannon_entropy_graphs_per_family_max_bits": round(fam_entropy_max, 4),
            "evenness_graphs_per_family": round(fam_evenness, 4),
            "simpsons_diversity_graphs_per_family": round(fam_simpson, 4),
            "shannon_entropy_cells_per_app_bits": round(app_entropy_bits, 4),
            "evenness_cells_per_app": round(app_evenness, 4),
            "simpsons_diversity_cells_per_app": round(app_simpson, 4),
        },
        "dominance": {
            "dominant_family_by_graph_count": dom_family,
            "dominant_family_graph_count": dom_n,
            "dominant_family_graph_fraction": round(dom_family_fraction, 4),
            "dominant_family_by_paper_l3_cells": dom_family_by_cells,
            "dominant_family_paper_l3_cell_count": dom_n_cells,
            "dominant_family_paper_l3_cell_fraction": round(
                dom_family_cell_fraction, 4
            ),
        },
        "per_family": {
            f: {
                "graphs": sorted(fam_graphs[f]),
                "n_graphs": n_graphs_per_family[f],
                "n_paper_l3_cells": n_cells_per_family[f],
                "paper_l3_sizes_reached": fam_l3_coverage[f],
                "reaches_4mb": "4MB" in fam_l3_coverage[f],
                "reaches_8mb": "8MB" in fam_l3_coverage[f],
            }
            for f in sorted(fam_graphs)
        },
        "per_app": {
            a: {
                "n_paper_l3_cells": n_cells_per_app[a],
            }
            for a in sorted(n_cells_per_app)
        },
        "per_family_per_l3_cells": {
            f: {
                l3: fam_l3_cells.get((f, l3), 0) for l3 in PAPER_L3_SIZES
            }
            for f in sorted(fam_graphs)
        },
        "per_app_per_l3_cells": {
            a: {
                l3: app_l3_cells.get((a, l3), 0) for l3 in PAPER_L3_SIZES
            }
            for a in sorted(n_cells_per_app)
        },
        "honest_disclosures": {
            "families_capped_below_4MB": families_capped_below_4mb,
            "families_capped_below_8MB": families_capped_below_8mb,
            "families_reaching_8MB": families_reaching_8mb,
            "note": (
                "Families lacking 4MB/8MB cells have graphs whose WSS-relative L3"
                " classification lands them in 'over' regime before 4MB. Reviewer"
                " comparisons that include 4MB+ L3 will exclude these families."
            ),
        },
    }
    return payload


def emit_md(payload: dict) -> str:
    meta = payload["meta"]
    dom = payload["dominance"]
    out = []
    out.append("# Corpus tier / family balance audit")
    out.append("")
    out.append(
        f"Source: `{meta['source']}`  •  Paper L3 scope: "
        f"{', '.join(meta['scope_l3_sizes'])}"
    )
    out.append("")
    out.append(
        f"Corpus: **{meta['n_graphs']} graphs** across **{meta['n_families']} "
        f"families** × **{meta['n_apps']} apps**; "
        f"{meta['n_paper_l3_cells_total']} paper-L3 cells in total."
    )
    out.append("")
    out.append("## Dominance disclosures")
    out.append("")
    out.append(
        f"- Dominant family by graph count: **{dom['dominant_family_by_graph_count']}** "
        f"with {dom['dominant_family_graph_count']} graphs "
        f"({dom['dominant_family_graph_fraction'] * 100:.1f}% of the corpus)"
    )
    out.append(
        f"- Dominant family by paper-L3 cells: "
        f"**{dom['dominant_family_by_paper_l3_cells']}** "
        f"with {dom['dominant_family_paper_l3_cell_count']} cells "
        f"({dom['dominant_family_paper_l3_cell_fraction'] * 100:.1f}% of paper-L3 cells)"
    )
    out.append("")
    out.append("## Diversity metrics (higher = more balanced)")
    out.append("")
    out.append("| metric | value | max |")
    out.append("|---|---:|---:|")
    out.append(
        f"| Shannon H (graphs/family, bits) "
        f"| {meta['shannon_entropy_graphs_per_family_bits']:.3f} "
        f"| {meta['shannon_entropy_graphs_per_family_max_bits']:.3f} |"
    )
    out.append(
        f"| Pielou evenness (graphs/family) "
        f"| {meta['evenness_graphs_per_family']:.3f} | 1.000 |"
    )
    out.append(
        f"| Simpson's D (graphs/family) "
        f"| {meta['simpsons_diversity_graphs_per_family']:.3f} | "
        f"{1 - 1/meta['n_families']:.3f} |"
    )
    out.append(
        f"| Pielou evenness (cells/app) "
        f"| {meta['evenness_cells_per_app']:.3f} | 1.000 |"
    )
    out.append(
        f"| Simpson's D (cells/app) "
        f"| {meta['simpsons_diversity_cells_per_app']:.3f} | "
        f"{1 - 1/meta['n_apps']:.3f} |"
    )
    out.append("")
    out.append("## Per-family composition")
    out.append("")
    out.append(
        "| family | n_graphs | n_paper_l3_cells | reaches_4MB | reaches_8MB | graphs |"
    )
    out.append("|---|---:|---:|:---:|:---:|---|")
    for fam in sorted(payload["per_family"].keys()):
        r = payload["per_family"][fam]
        gs = ", ".join(r["graphs"])
        check4 = "✅" if r["reaches_4mb"] else "❌"
        check8 = "✅" if r["reaches_8mb"] else "❌"
        out.append(
            f"| {fam} | {r['n_graphs']} | {r['n_paper_l3_cells']} "
            f"| {check4} | {check8} | {gs} |"
        )
    out.append("")
    out.append("## Per-(family, L3) cell counts")
    out.append("")
    out.append("| family | 1MB | 4MB | 8MB |")
    out.append("|---|---:|---:|---:|")
    for fam in sorted(payload["per_family_per_l3_cells"].keys()):
        r = payload["per_family_per_l3_cells"][fam]
        out.append(
            f"| {fam} | {r['1MB']} | {r['4MB']} | {r['8MB']} |"
        )
    out.append("")
    out.append("## Per-(app, L3) cell counts")
    out.append("")
    out.append("| app | 1MB | 4MB | 8MB |")
    out.append("|---|---:|---:|---:|")
    for app in sorted(payload["per_app_per_l3_cells"].keys()):
        r = payload["per_app_per_l3_cells"][app]
        out.append(f"| {app} | {r['1MB']} | {r['4MB']} | {r['8MB']} |")
    out.append("")
    out.append("## Honest disclosures")
    out.append("")
    for line in payload["honest_disclosures"]["note"].split(". "):
        if line.strip():
            out.append(f"- {line.strip()}")
    out.append(
        f"- Families capped below 4MB: "
        f"{payload['honest_disclosures']['families_capped_below_4MB']}"
    )
    out.append(
        f"- Families capped below 8MB: "
        f"{payload['honest_disclosures']['families_capped_below_8MB']}"
    )
    out.append(
        f"- Families reaching 8MB: "
        f"{payload['honest_disclosures']['families_reaching_8MB']}"
    )
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle-json", type=Path, default=WIKI_DATA / "oracle_gap.json"
    )
    parser.add_argument(
        "--json-out", type=Path, default=WIKI_DATA / "corpus_balance.json"
    )
    parser.add_argument(
        "--md-out", type=Path, default=WIKI_DATA / "corpus_balance.md"
    )
    args = parser.parse_args()

    payload = build_payload(args.oracle_json)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(emit_md(payload).rstrip("\n") + "\n")
    meta = payload["meta"]
    dom = payload["dominance"]
    print(
        f"corpus-balance: graphs={meta['n_graphs']} families={meta['n_families']}"
        f" apps={meta['n_apps']} paper_l3_cells={meta['n_paper_l3_cells_total']}"
        f" | dom_family_by_graphs={dom['dominant_family_by_graph_count']}"
        f" ({dom['dominant_family_graph_fraction'] * 100:.1f}%)"
        f" | family_evenness={meta['evenness_graphs_per_family']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
