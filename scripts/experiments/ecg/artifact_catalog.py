#!/usr/bin/env python3
"""Paper-artifact catalog — single authoritative index of every
paper-grade aggregator in ``wiki/data/`` plus its governing pytest
gate, source generator, citation purpose, and current headline
finding.

Why this exists
---------------
The paper-grade evidence chain now has 27 confidence gates spread
across 17 JSON aggregators. Onboarding a co-author (or a reviewer
asking "where does claim X come from?") requires walking the
Makefile + gate dashboard + per-aggregator scripts to find the
canonical chain. This script collapses that walk into one
machine-readable + paper-ready index.

The catalog answers, per artifact:

* What it computes (one-line summary).
* The script that generates it.
* The pytest module that guards it.
* The headline number(s) it contributes to the paper.

It is intentionally self-contained — no joins with the data files,
just metadata + a sanity check that each listed source script /
gate / artifact file exists on disk.

Outputs
-------
* ``wiki/data/artifact_catalog.json`` — machine-readable index.
* ``wiki/data/artifact_catalog.md``   — paper-ready table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


# Each entry: (key, label, generator script, pytest gate path,
# json artifact path relative to repo root, one-line headline).
# Keep this sorted by canonical evidence-chain ordering so readers
# see corpus -> reproduction -> per-policy -> per-graph -> meta.
CATALOG = [
    # ---------- Corpus + reproduction foundation ----------
    {
        "id":        "corpus_diversity",
        "label":     "Corpus structural diversity",
        "generator": "scripts/experiments/ecg/corpus_diversity.py",
        "gate":      "scripts/test/test_corpus_diversity_floor.py",
        "artifact":  "wiki/data/corpus_diversity.json",
        "summary":   "Per-graph structural features (hub_concentration, clustering_coeff, avg_degree, working_set_ratio) for every corpus graph; underpins family classification used by every other report.",
    },
    {
        "id":        "literature_reproduction",
        "label":     "Per-paper reproduction summary",
        "generator": "scripts/experiments/ecg/literature_reproduction_summary.py",
        "gate":      "scripts/test/test_baselines_match_literature.py",
        "artifact":  "wiki/data/literature_reproduction_summary.csv",
        "summary":   "Per-paper grouped reproduction map (Faldu HPCA20, Balaji HPCA21, Jaleel ISCA10) classifying every claim into ok/within_tolerance/disagree/known_deviation/missing.",
    },
    {
        "id":        "literature_faithfulness",
        "label":     "Literature faithfulness comparator",
        "generator": "scripts/experiments/ecg/literature_faithfulness.py",
        "gate":      "scripts/test/test_lit_faith_no_disagree.py",
        "artifact":  "wiki/data/literature_faithfulness_postfix.json",
        "summary":   "Cell-level comparator: 288/320 ok (90.0 %), 0 disagree, 30 known_deviation. The single load-bearing aggregate behind every other paper-grade finding.",
    },
    {
        "id":        "regression_budget",
        "label":     "Regression budget floor",
        "generator": "scripts/experiments/ecg/regression_budget.py",
        "gate":      "scripts/test/test_regression_budget_floor.py",
        "artifact":  "wiki/data/regression_budget.json",
        "summary":   "Per-cell distance-to-disagree in pp; the smallest margin defines the corpus-wide regression budget. Fails if any cell's budget collapses below the floor.",
    },
    # ---------- Cross-tool simulator soundness ----------
    {
        "id":        "gem5_anchor",
        "label":     "gem5 literature anchor",
        "generator": "scripts/experiments/ecg/gem5_anchor_summary.py",
        "gate":      "scripts/test/test_gem5_anchor.py",
        "artifact":  "wiki/data/gem5_anchor.json",
        "summary":   "GRASP-paper L-shape invariants codified per (graph, app) at 4 kB / 256 kB / 2 MB. 16 invariants today, all ok.",
    },
    {
        "id":        "sniper_anchor",
        "label":     "Sniper literature anchor",
        "generator": "scripts/experiments/ecg/gem5_anchor_summary.py",
        "gate":      "scripts/test/test_sniper_anchor.py",
        "artifact":  "wiki/data/sniper_anchor.json",
        "summary":   "Sniper L-shape mirror for PR + SSSP on email-Eu-core + cit-Patents (16 invariants ok; max small-cache spread 6.36 pp). Uses the shared gem5_anchor_summary.py generator.",
    },
    {
        "id":        "cross_tool_saturation",
        "label":     "Cross-tool saturation soundness",
        "generator": "scripts/experiments/ecg/cross_tool_saturation_report.py",
        "gate":      "scripts/test/test_cross_tool_saturation.py",
        "artifact":  "wiki/data/cross_tool_saturation.json",
        "summary":   "Pairs each lit-faith cell with its gem5/Sniper anchor at each tool's largest L3 and verifies Δ(GRASP−LRU) sign agreement when doubly saturated.",
    },
    {
        "id":        "cross_tool_winners",
        "label":     "Cross-tool winner agreement",
        "generator": "scripts/experiments/ecg/cross_tool_winners_report.py",
        "gate":      "scripts/test/test_cross_tool_winners.py",
        "artifact":  "wiki/data/cross_tool_winners.json",
        "summary":   "At each tool's largest L3 per (graph, app), do simulators pick the same winning policy? Surfaces 6 split cells (expected; tools sweep disjoint L3 ranges).",
    },
    # ---------- Per-policy / per-cell analyses ----------
    {
        "id":        "policy_winner_table",
        "label":     "Policy winner table",
        "generator": "scripts/experiments/ecg/policy_winner_table.py",
        "gate":      "scripts/test/test_policy_winner_table.py",
        "artifact":  "wiki/data/policy_winner_table.json",
        "summary":   "Per-cell winner projection: GRASP 56 wins, POPT 41, LRU 6, SRRIP 6 (n=109). Top-5 fragile cells flagged when margin ≤ 1 pp.",
    },
    {
        "id":        "popt_vs_grasp_delta",
        "label":     "POPT-vs-GRASP delta",
        "generator": "scripts/experiments/ecg/popt_vs_grasp_report.py",
        "gate":      "scripts/test/test_popt_vs_grasp_delta.py",
        "artifact":  "wiki/data/popt_vs_grasp_delta.json",
        "summary":   "Per-cell Δ(POPT − GRASP) in pp by family/regime. Road family mean −9.276 pp (POPT crushes GRASP); social family mean +0.360 pp (tie).",
    },
    {
        "id":        "oracle_gap",
        "label":     "Per-policy oracle gap",
        "generator": "scripts/experiments/ecg/oracle_gap_report.py",
        "gate":      "scripts/test/test_oracle_gap.py",
        "artifact":  "wiki/data/oracle_gap.json",
        "summary":   "Each policy's gap to per-cell empirical oracle (min across 4 policies). Mean gaps: POPT 1.65 pp, GRASP 3.10 pp, SRRIP 3.60 pp, LRU 4.93 pp.",
    },
    {
        "id":        "winning_regime_taxonomy",
        "label":     "Winning-regime taxonomy",
        "generator": "scripts/experiments/ecg/winning_regime_taxonomy.py",
        "gate":      "scripts/test/test_winning_regime_taxonomy.py",
        "artifact":  "wiki/data/winning_regime_taxonomy.json",
        "summary":   "(graph_family × L3 regime) winner matrix with auto-extracted ≥80 % dominance rules. Headline rules: mesh/all → POPT 100 %; road/large → LRU 75 %.",
    },
    # ---------- Edge regime + diagnostics ----------
    {
        "id":        "small_l3_thrash",
        "label":     "Small-L3 thrash report",
        "generator": "scripts/experiments/ecg/small_l3_thrash_report.py",
        "gate":      "scripts/test/test_small_l3_thrash.py",
        "artifact":  "wiki/data/small_l3_thrash.json",
        "summary":   "4 kB-L3 sweep (9 (graph, app) cells × 9 policy variants). LRU wins 5/9 cells; GRASP regresses up to +35.857 pp vs LRU on soc-LiveJournal1/bfs.",
    },
    {
        "id":        "literature_deviations",
        "label":     "Literature deviations inventory",
        "generator": "scripts/experiments/ecg/literature_deviations_report.py",
        "gate":      "scripts/test/test_literature_deviations.py",
        "artifact":  "wiki/data/literature_deviations.json",
        "summary":   "Closed-vocab mechanism classifier for known_deviation rows: 30/30 classify as popt_overhead_dominates — the exact inverse of road-graph finding.",
    },
    {
        "id":        "claim_density",
        "label":     "Per-graph claim density",
        "generator": "scripts/experiments/ecg/claim_density_report.py",
        "gate":      "scripts/test/test_claim_density.py",
        "artifact":  "wiki/data/claim_density.json",
        "summary":   "Per-graph literature claim density (8 graphs, 320 claims, 288 OK = 90.0 %). Density per graph: 2 (delaunay_n19) → 12 (cit-Patents).",
    },
    {
        "id":        "bootstrap_ci",
        "label":     "Bootstrap CIs on load-bearing claims",
        "generator": "scripts/experiments/ecg/bootstrap_ci.py",
        "gate":      "scripts/test/test_bootstrap_ci.py",
        "artifact":  "wiki/data/bootstrap_ci.json",
        "summary":   "Percentile bootstrap (5000 resamples, seed 1729) on every (policy, family) and (policy, regime) oracle-gap bucket + paired ΔPOPT−GRASP per family + sign-stability fractions. Road POPT < GRASP survives 97.6 % of resamples; social/citation/web do not.",
    },
    # ---------- Meta artifacts ----------
    {
        "id":        "paper_claims",
        "label":     "Paper claims registry",
        "generator": "scripts/experiments/ecg/paper_claims_registry.py",
        "gate":      "scripts/test/test_paper_claims_registry.py",
        "artifact":  "wiki/data/paper_claims.json",
        "summary":   "Single source of truth for every numerical claim the paper makes (14 claims across 8 categories), each linked to its source artifact + governing gate.",
    },
    {
        "id":        "confidence_dashboard",
        "label":     "Confidence dashboard",
        "generator": "scripts/experiments/ecg/confidence_dashboard.py",
        "gate":      "scripts/test/test_confidence_dashboard.py",
        "artifact":  "wiki/data/confidence_dashboard.json",
        "summary":   "Single-screen verdict (29 gates today, all GREEN). The dashboard this catalog sits next to.",
    },
]


def _audit(entries: list[dict]) -> list[dict]:
    """Annotate each entry with on-disk presence flags so the gate
    can verify nothing has gone stale."""
    out: list[dict] = []
    for e in entries:
        gen = REPO_ROOT / e["generator"]
        gate = REPO_ROOT / e["gate"]
        art = REPO_ROOT / e["artifact"]
        out.append({
            **e,
            "generator_exists": gen.exists(),
            "gate_exists":      gate.exists(),
            "artifact_exists":  art.exists(),
        })
    return out


def _write_json(entries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_entries":             len(entries),
        "missing_generators":    [e["id"] for e in entries if not e["generator_exists"]],
        "missing_gates":         [e["id"] for e in entries if not e["gate_exists"]],
        "missing_artifacts":     [e["id"] for e in entries if not e["artifact_exists"]],
    }
    path.write_text(json.dumps({
        "summary":  summary,
        "entries":  entries,
    }, indent=2, sort_keys=True))


def _write_md(entries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Paper-artifact catalog")
    lines.append("")
    lines.append(
        "_Generated by `scripts/experiments/ecg/artifact_catalog.py`. "
        "This is the single canonical index reviewers should consult "
        "to trace every paper claim back to its source artifact + "
        "governing pytest gate._"
    )
    lines.append("")
    n = len(entries)
    missing_gen = sum(1 for e in entries if not e["generator_exists"])
    missing_gate = sum(1 for e in entries if not e["gate_exists"])
    missing_art = sum(1 for e in entries if not e["artifact_exists"])
    lines.append(
        f"**{n} entries.** Missing generators: {missing_gen}; "
        f"missing gates: {missing_gate}; "
        f"missing artifacts: {missing_art}."
    )
    lines.append("")
    lines.append("| # | id | label | summary |")
    lines.append("|---:|---|---|---|")
    for i, e in enumerate(entries, 1):
        lines.append(f"| {i} | `{e['id']}` | {e['label']} | {e['summary']} |")
    lines.append("")
    lines.append("## Source chain per entry")
    lines.append("")
    for e in entries:
        lines.append(f"### `{e['id']}` — {e['label']}")
        lines.append("")
        lines.append(f"- **Generator:** `{e['generator']}`"
                     + ("" if e["generator_exists"] else "  **❌ MISSING**"))
        lines.append(f"- **Gate:** `{e['gate']}`"
                     + ("" if e["gate_exists"] else "  **❌ MISSING**"))
        lines.append(f"- **Artifact:** `{e['artifact']}`"
                     + ("" if e["artifact_exists"] else "  **❌ MISSING**"))
        lines.append(f"- **Headline:** {e['summary']}")
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "wiki" / "data" / "artifact_catalog.json",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=REPO_ROOT / "wiki" / "data" / "artifact_catalog.md",
    )
    args = parser.parse_args()

    entries = _audit(CATALOG)
    _write_json(entries, args.json_out)
    _write_md(entries, args.md_out)
    missing_gen = [e["id"] for e in entries if not e["generator_exists"]]
    missing_gate = [e["id"] for e in entries if not e["gate_exists"]]
    missing_art = [e["id"] for e in entries if not e["artifact_exists"]]
    print(
        f"[catalog] {len(entries)} entries; "
        f"missing gen={missing_gen} gate={missing_gate} art={missing_art}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
