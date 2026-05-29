#!/usr/bin/env python3
"""Confidence dashboard for the GraphBrew literature-faithfulness pipeline.

Why this exists
---------------
At any moment a reviewer (or the maintainer) needs to know "are we still
green?" without re-reading 6 separate reports. This script aggregates the
existing gates into one screen so the answer is unambiguous.

What it reports
---------------
* Tier A — sideband registration sanity         (test_grasp_sideband_registration)
* Tier B — POPT permutation equivalence         (test_popt_permutation_equivalence)
* Tier C — GRASP-vs-LRU sign test               (test_grasp_sign_consistency)
* Structural lit-baseline test                  (test_literature_baselines_structure)
* Data-driven lit-baseline test                 (test_baselines_match_literature)
* Corpus diversity profile parity test          (test_corpus_diversity)
* Literature-faithfulness comparator headline   (literature_faithfulness JSON summary)
* Corpus diversity coverage                     (corpus_diversity JSON)

Usage
-----
    python -m scripts.experiments.ecg.confidence_dashboard \\
        --lit-faith-json wiki/data/literature_faithfulness_postfix.json \\
        --corpus-diversity-json wiki/data/corpus_diversity.json \\
        --markdown wiki/data/confidence_dashboard.md

Tests can run in `--fast` mode (default) which skips heavyweight suites
(fill_weights_variants) — the dashboard only needs the literature /
sideband / sign tests.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]

PYTEST_SUITES: dict[str, tuple[str, str]] = {
    "Tier A — GRASP sideband registration":
        ("scripts/test/test_grasp_sideband_registration.py", "Tier A"),
    "Tier B — POPT permutation equivalence":
        ("scripts/test/test_popt_permutation_equivalence.py", "Tier B"),
    "Tier C — GRASP vs LRU sign test":
        ("scripts/test/test_grasp_sign_consistency.py", "Tier C"),
    "Lit-baseline data-driven gate":
        ("scripts/test/test_baselines_match_literature.py", "Lit-data"),
    "Lit-baseline structural gate":
        ("scripts/test/test_literature_baselines_structure.py", "Lit-struct"),
    "Lit-faith no-disagree gate":
        ("scripts/test/test_lit_faith_no_disagree.py", "Lit-faith"),
    "ECG validation gate catalog":
        ("scripts/test/test_ecg_validation_gates_catalog.py", "ECG-cat"),
    "Corpus diversity floor":
        ("scripts/test/test_corpus_diversity_floor.py", "Corpus-floor"),
    "Cross-tool report parity":
        ("scripts/test/test_cross_tool_parity.py", "Parity"),
    "Regression budget floor":
        ("scripts/test/test_regression_budget_floor.py", "Budget"),
    "Corpus diversity profile parity":
        ("scripts/test/test_corpus_diversity.py", "Corpus"),
    "Paper-pipeline literature pre-flight gate":
        ("scripts/test/test_paper_pipeline_lit_gate.py", "Preflight"),
    "gem5 literature anchor":
        ("scripts/test/test_gem5_anchor.py", "Gem5-anchor"),
    "Sniper literature anchor":
        ("scripts/test/test_sniper_anchor.py", "Sniper-anchor"),
    "Lit-preflight shared helper":
        ("scripts/test/test_literature_preflight.py", "Lit-helper"),
    "GRASP road-like graph invariant":
        ("scripts/test/test_road_like_graph_invariant.py", "Road-like"),
    "L-curve monotonicity gate":
        ("scripts/test/test_l_curve_monotonicity.py", "L-mono"),
    "Policy-winner table sanity":
        ("scripts/test/test_policy_winner_table.py", "Winner"),
    "Small-L3 thrash sanity":
        ("scripts/test/test_small_l3_thrash.py", "Thrash"),
    "Cross-tool saturation soundness":
        ("scripts/test/test_cross_tool_saturation.py", "X-tool"),
    "Cross-tool winner agreement":
        ("scripts/test/test_cross_tool_winners.py", "X-win"),
    "Per-graph claim density":
        ("scripts/test/test_claim_density.py", "Density"),
    "POPT-vs-GRASP delta":
        ("scripts/test/test_popt_vs_grasp_delta.py", "P-vs-G"),
    "Literature deviations inventory":
        ("scripts/test/test_literature_deviations.py", "Lit-dev"),
    "Paper claims registry":
        ("scripts/test/test_paper_claims_registry.py", "Claims"),
    "Winning-regime taxonomy":
        ("scripts/test/test_winning_regime_taxonomy.py", "Regime"),
    "Oracle gap":
        ("scripts/test/test_oracle_gap.py", "Oracle"),
}


@dataclass
class SuiteResult:
    label: str
    short: str
    path: str
    passed: int
    failed: int
    skipped: int
    xfailed: int
    xpassed: int
    errors: int
    runtime_s: float
    raw_tail: str

    @property
    def is_green(self) -> bool:
        return self.failed == 0 and self.errors == 0


SUMMARY_RE = re.compile(
    r"(?P<passed>\d+)\s+passed"
    r"(?:,\s+(?P<skipped>\d+)\s+skipped)?"
    r"(?:,\s+(?P<xfailed>\d+)\s+xfailed)?"
    r"(?:,\s+(?P<xpassed>\d+)\s+xpassed)?"
    r"(?:,\s+(?P<failed>\d+)\s+failed)?"
    r"(?:,\s+(?P<errors>\d+)\s+errors?)?"
)


def _parse_pytest_summary(text: str) -> dict[str, int]:
    """Pull pass/fail/skipped/xfailed/xpassed/errors from a pytest summary line."""
    out = {k: 0 for k in ("passed", "failed", "skipped", "xfailed", "xpassed", "errors")}
    for line in text.splitlines()[::-1]:
        # The summary line looks like:  "== 6 passed, 1 skipped in 0.12s =="
        # or "== 6 passed, 1 failed in 0.12s =="; numbers may appear in any
        # order so we scan for individual keywords.
        if " passed" not in line and " failed" not in line and " error" not in line:
            continue
        for key in out:
            m = re.search(rf"(\d+)\s+{key}", line)
            if m:
                out[key] = int(m.group(1))
        if any(out.values()):
            break
    return out


def _run_suite(label: str, short: str, path: str, pytest_args: Sequence[str]) -> SuiteResult:
    cmd = [sys.executable, "-m", "pytest", path, "-q", "--no-header", "--tb=no"] + list(pytest_args)
    started = time.time()
    completed = subprocess.run(  # noqa: S603 — fixed argv
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    elapsed = time.time() - started
    text = completed.stdout + completed.stderr
    counts = _parse_pytest_summary(text)
    tail = "\n".join(text.splitlines()[-15:])
    return SuiteResult(
        label=label, short=short, path=path,
        passed=counts["passed"], failed=counts["failed"], skipped=counts["skipped"],
        xfailed=counts["xfailed"], xpassed=counts["xpassed"], errors=counts["errors"],
        runtime_s=elapsed, raw_tail=tail,
    )


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _lit_faith_section(lit) -> list[str]:
    out = ["## Literature-faithfulness comparator", ""]
    if not isinstance(lit, dict):
        out += ["_No literature_faithfulness JSON found — run "
                "`python -m scripts.experiments.ecg.literature_faithfulness ...` first._", ""]
        return out
    s = lit.get("summary", {})
    total = s.get("claims_total", 0)
    ok = s.get("ok", 0)
    disagree = s.get("disagree", 0)
    known = s.get("known_deviation", 0)
    within = s.get("within_tolerance", 0)
    insuf = s.get("insufficient_data", 0)
    missing = s.get("missing", 0)
    ok_pct = (100.0 * ok / total) if total else 0.0
    verdict = "✅ green" if disagree == 0 else f"⛔ {disagree} unexplained disagreements"
    out += [
        f"**Verdict:** {verdict}  ({ok}/{total} ok = {ok_pct:.1f}%)",
        "",
        "| status | count |",
        "|---|---:|",
        f"| ok | {ok} |",
        f"| within_tolerance | {within} |",
        f"| **DISAGREE** | **{disagree}** |",
        f"| known_deviation | {known} |",
        f"| insufficient_data | {insuf} |",
        f"| missing | {missing} |",
        f"| **total claims** | **{total}** |",
        "",
    ]
    return out


def _budget_section(budget) -> list[str]:
    out = ["## Regression budget — distance to disagree", ""]
    if not isinstance(budget, dict):
        out += ["_No regression_budget JSON found — run "
                "`make lit-budget` to generate one._", ""]
        return out
    s = budget.get("summary", {})
    by_kind = s.get("by_kind", {})
    out += [
        f"- Cells in distribution: **{s.get('cells_in_distribution', 0)}**",
        f"- Min margin (any kind): **{s.get('min_margin_pp', 0):.3f} pp**",
        f"- Median margin: {s.get('median_margin_pp', 0):.3f} pp",
        f"- p90 margin: {s.get('p90_margin_pp', 0):.3f} pp",
        "",
        "| claim kind | n | min margin (pp) | median margin (pp) |",
        "|---|---:|---:|---:|",
    ]
    for k, v in by_kind.items():
        out.append(
            f"| {k} | {v.get('n', 0)} | {v.get('min_pp', 0):.3f} "
            f"| {v.get('median_pp', 0):.3f} |"
        )
    out.append("")
    fragile = (budget.get("fragile_cache_policy_cells") or [])[:5]
    if fragile:
        out += ["**5 most fragile cache-policy cells:**", ""]
        out += ["| graph | app | l3 | policy | Δ (pp) | margin (pp) |"]
        out += ["|---|---|---|---|---:|---:|"]
        for r in fragile:
            out.append(
                f"| {r['graph']} | {r['app']} | {r['l3_size']} "
                f"| {r['policy']} | {r['delta_pct']:+.3f} "
                f"| {r['margin_pp']:.3f} |"
            )
        out.append("")
    return out


def _corpus_section(corpus) -> list[str]:
    out = ["## Corpus diversity coverage", ""]
    if corpus is None:
        out += ["_No corpus_diversity JSON found — run "
                "`python -m scripts.experiments.ecg.corpus_diversity ...` first._", ""]
        return out
    # corpus_diversity emits a top-level list; older callers may wrap it.
    if isinstance(corpus, dict):
        cards = corpus.get("graphs") or corpus.get("rows") or []
    else:
        cards = corpus or []
    if not cards:
        out += ["_corpus_diversity JSON has no graph cards._", ""]
        return out
    out += [f"**Graphs profiled:** {len(cards)}", ""]
    out += ["| graph | nodes | edges | hub_conc | avg_deg | clustering_sampled | working_set_ratio |"]
    out += ["|---|---:|---:|---:|---:|---:|---:|"]
    for card in cards:
        name = card.get("graph", "?")
        feats = card.get("features", card)

        def _g(k: str, src=feats, top=card):
            v = src.get(k) if isinstance(src, dict) else None
            if v is None and isinstance(top, dict):
                v = top.get(k)
            if v is None:
                return "—"
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)

        out.append(
            "| " + " | ".join([
                name,
                _g("nodes", src=card),
                _g("edges", src=card),
                _g("hub_concentration"),
                _g("avg_degree"),
                _g("clustering_coeff"),
                _g("working_set_ratio"),
            ]) + " |"
        )
    out.append("")
    return out


def _headline_verdict(results: list[SuiteResult], lit) -> str:
    red = [r.label for r in results if not r.is_green]
    if red:
        return "⛔ RED — " + ", ".join(red) + " failing"
    if isinstance(lit, dict) and lit.get("summary", {}).get("disagree", 0) > 0:
        n = lit["summary"]["disagree"]
        return f"⛔ RED — {n} unexplained disagreements in lit-faith comparator"
    return "✅ GREEN — every tier + gate + comparator is within tolerance"


def _pytest_section(results: list[SuiteResult]) -> list[str]:
    out = ["## Tier & gate pytest results", ""]
    out += ["| gate | pass | skip | xfail | fail | err | runtime | verdict |"]
    out += ["|---|---:|---:|---:|---:|---:|---:|:---:|"]
    for r in results:
        verdict = "✅" if r.is_green else "⛔"
        out.append(
            f"| {r.label} | {r.passed} | {r.skipped} | {r.xfailed} | "
            f"{r.failed} | {r.errors} | {r.runtime_s:.1f}s | {verdict} |"
        )
    out.append("")
    failing = [r for r in results if not r.is_green]
    if failing:
        out += ["### ⛔ Failing tail (last 15 lines per failing gate)", ""]
        for r in failing:
            out += [f"#### {r.label}", "```", r.raw_tail, "```", ""]
    return out


def render(results: list[SuiteResult], lit, corpus, budget=None) -> str:
    out = [
        "# GraphBrew literature-faithfulness confidence dashboard",
        "",
        "_Generated by `scripts/experiments/ecg/confidence_dashboard.py`._",
        "",
        f"## Headline: {_headline_verdict(results, lit)}",
        "",
    ]
    out += _pytest_section(results)
    out += _lit_faith_section(lit)
    out += _budget_section(budget)
    out += _corpus_section(corpus)
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--lit-faith-json",
        default=str(REPO_ROOT / "wiki" / "data" / "literature_faithfulness_postfix.json"),
        help="Path to literature_faithfulness JSON output.",
    )
    parser.add_argument(
        "--corpus-diversity-json",
        default=str(REPO_ROOT / "wiki" / "data" / "corpus_diversity.json"),
        help="Path to corpus_diversity JSON output.",
    )
    parser.add_argument(
        "--regression-budget-json",
        default=str(REPO_ROOT / "wiki" / "data" / "regression_budget.json"),
        help="Path to regression_budget JSON output.",
    )
    parser.add_argument(
        "--markdown", default=None,
        help="Optional path to write the rendered dashboard markdown.",
    )
    parser.add_argument(
        "--skip-pytest", action="store_true",
        help="Skip running pytest; only render the data sections.",
    )
    parser.add_argument(
        "--include-slow", action="store_true",
        help="Include slow tier suites (none currently; reserved for future).",
    )
    parser.add_argument(
        "--pytest-arg", action="append", default=[],
        help="Extra arg to forward to pytest (repeatable).",
    )
    parser.add_argument(
        "--json-out", default=None,
        help="Optional path to dump a machine-readable summary.",
    )
    args = parser.parse_args()

    results: list[SuiteResult] = []
    if not args.skip_pytest:
        for label, (path, short) in PYTEST_SUITES.items():
            results.append(_run_suite(label, short, path, args.pytest_arg))

    lit = _read_json(Path(args.lit_faith_json))
    corpus = _read_json(Path(args.corpus_diversity_json))
    budget = _read_json(Path(args.regression_budget_json))

    rendered = render(results, lit, corpus, budget)
    print(rendered)
    if args.markdown:
        Path(args.markdown).write_text(rendered.rstrip("\n") + "\n")
        print(f"[dashboard] markdown -> {args.markdown}", file=sys.stderr)

    if args.json_out:
        graph_count = 0
        if isinstance(corpus, dict):
            graph_count = len(corpus.get("graphs", []) or corpus.get("rows", []))
        elif isinstance(corpus, list):
            graph_count = len(corpus)
        Path(args.json_out).write_text(json.dumps({
            "headline": _headline_verdict(results, lit),
            "suites": [r.__dict__ for r in results],
            "lit_faith_summary": (lit or {}).get("summary") if isinstance(lit, dict) else None,
            "regression_budget_summary": (budget or {}).get("summary") if isinstance(budget, dict) else None,
            "corpus_graph_count": graph_count,
        }, indent=2) + "\n")
        print(f"[dashboard] json -> {args.json_out}", file=sys.stderr)

    return 0 if "GREEN" in _headline_verdict(results, lit) else 1


if __name__ == "__main__":
    sys.exit(main())
