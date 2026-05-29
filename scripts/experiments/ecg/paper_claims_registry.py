#!/usr/bin/env python3
"""Paper claims registry.

Why this exists
---------------
The paper contains dozens of concrete numerical claims (e.g.
"GRASP wins 51 % of cells", "POPT improves GRASP by 9.3 pp on road
graphs", "30 documented literature deviations are all POPT-overhead
artefacts"). Reviewers must be able to *check every number*. This
script crawls every paper-grade aggregator we ship and emits a
single registry that links each claim to:

* the value (with units),
* the artifact file that hosts the raw data,
* the gate (pytest module) that guards the invariant the claim
  depends on,
* a one-line provenance string for the paper bibliography.

The registry is meant to be the **single source of truth** the paper
quotes from. If the paper text ever drifts from the registry, the
matching gate will catch it.

Output
------
* ``wiki/data/paper_claims.json`` — machine-readable list of
  claims with values + provenance + governing gate.
* ``wiki/data/paper_claims.md`` — paper-ready markdown.

Usage
-----
    python3 -m scripts.experiments.ecg.paper_claims_registry
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA = REPO_ROOT / "wiki" / "data"


def _safe_load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _maybe_round(v, ndigits: int = 3):
    if isinstance(v, float):
        return round(v, ndigits)
    return v


def build_claims() -> list[dict]:
    """Produce the canonical claims list. Order matters — the paper's
    narrative arc follows this sequence."""
    claims: list[dict] = []

    confidence = _safe_load(DATA / "confidence_dashboard.json") or {}
    lit_faith = _safe_load(DATA / "literature_faithfulness_postfix.json") or {}
    winner = _safe_load(DATA / "policy_winner_table.json") or {}
    thrash = _safe_load(DATA / "small_l3_thrash.json") or {}
    crosstool = _safe_load(DATA / "cross_tool_saturation.json") or {}
    density = _safe_load(DATA / "claim_density.json") or {}
    deviations = _safe_load(DATA / "literature_deviations.json") or {}
    pvg = _safe_load(DATA / "popt_vs_grasp_delta.json") or {}
    corpus = _safe_load(DATA / "corpus_diversity.json") or []

    # ------------------------------------------------------------------
    # 1) Corpus / scale claims
    # ------------------------------------------------------------------
    if isinstance(corpus, list):
        graphs = corpus
    else:
        graphs = corpus.get("graphs", []) if isinstance(corpus, dict) else []
    if graphs:
        claims.append({
            "id": "corpus.graph_count",
            "category": "corpus",
            "text": "Corpus spans %d graphs across web/citation/social/road/mesh families." % len(graphs),
            "value": len(graphs),
            "units": "graphs",
            "source": "wiki/data/corpus_diversity.json",
            "gate": "scripts/test/test_corpus_diversity_floor.py",
        })

    # ------------------------------------------------------------------
    # 2) Reproduction claims
    # ------------------------------------------------------------------
    density_summary = density.get("summary", {}) if isinstance(density, dict) else {}
    if density_summary:
        n_claims = density_summary.get("total_claims", 0)
        n_ok = density_summary.get("total_ok", 0)
        claims.append({
            "id": "reproduction.ok_ratio",
            "category": "reproduction",
            "text": (
                "Literature reproduction matches %d / %d claims (%.1f %%) "
                "across the corpus." % (
                    n_ok, n_claims,
                    100.0 * n_ok / n_claims if n_claims else 0.0,
                )
            ),
            "value": _maybe_round(100.0 * n_ok / n_claims if n_claims else 0.0, 1),
            "units": "percent",
            "source": "wiki/data/claim_density.json",
            "gate": "scripts/test/test_claim_density.py",
        })
        claims.append({
            "id": "reproduction.n_graphs_with_claims",
            "category": "reproduction",
            "text": (
                "Reproduction summary spans %d graphs and %d total "
                "literature claims." % (
                    density_summary.get("n_graphs", 0), n_claims,
                )
            ),
            "value": density_summary.get("n_graphs", 0),
            "units": "graphs",
            "source": "wiki/data/claim_density.json",
            "gate": "scripts/test/test_claim_density.py",
        })

    # ------------------------------------------------------------------
    # 3) Lit-faith no-disagreement
    # ------------------------------------------------------------------
    lf_summary = lit_faith.get("summary", {}) if isinstance(lit_faith, dict) else {}
    if lf_summary:
        n_total = lf_summary.get("claims_total", 0)
        n_disagree = lf_summary.get("disagree", 0)
        if n_total:
            claims.append({
                "id": "lit_faith.disagreement_rate",
                "category": "lit_faith",
                "text": (
                    "Across %d (graph, app, L3, policy) claims the lit-faith "
                    "comparator records %d disagreement(s)." % (
                        n_total, n_disagree,
                    )
                ),
                "value": n_disagree,
                "units": "claims",
                "source": "wiki/data/literature_faithfulness_postfix.json",
                "gate": "scripts/test/test_lit_faith_no_disagree.py",
            })

    # ------------------------------------------------------------------
    # 4) Policy winner table
    # ------------------------------------------------------------------
    w_summary = winner.get("summary", {}) if isinstance(winner, dict) else {}
    if w_summary:
        n_cells = w_summary.get("n_cells", 0)
        wins_by_policy = w_summary.get("wins_by_policy", {})
        for pol in ("GRASP", "POPT", "SRRIP", "LRU"):
            wins = wins_by_policy.get(pol, 0)
            if n_cells:
                claims.append({
                    "id": f"winner.{pol.lower()}_share",
                    "category": "winner_table",
                    "text": (
                        "%s wins %d / %d cells (%.1f %%) in the "
                        "lit-faith corpus." % (
                            pol, wins, n_cells, 100.0 * wins / n_cells,
                        )
                    ),
                    "value": _maybe_round(100.0 * wins / n_cells, 1),
                    "units": "percent",
                    "source": "wiki/data/policy_winner_table.json",
                    "gate": "scripts/test/test_policy_winner_table.py",
                })

    # ------------------------------------------------------------------
    # 5) Small-L3 thrash
    # ------------------------------------------------------------------
    t_summary = thrash.get("summary", {}) if isinstance(thrash, dict) else {}
    if t_summary:
        winners = t_summary.get("win_counts", {})
        lru_w = winners.get("LRU", 0)
        n = t_summary.get("n_cells", 0)
        if n:
            claims.append({
                "id": "thrash.lru_wins_at_4kb",
                "category": "thrash",
                "text": (
                    "At a 4 kB L3 LRU wins %d / %d cells, beating both "
                    "GRASP and POPT - the canonical thrash regime where "
                    "policy overhead dominates." % (lru_w, n)
                ),
                "value": lru_w,
                "units": "cells",
                "source": "wiki/data/small_l3_thrash.json",
                "gate": "scripts/test/test_small_l3_thrash.py",
            })

    # ------------------------------------------------------------------
    # 6) POPT vs GRASP delta
    # ------------------------------------------------------------------
    p_summary = pvg.get("summary", {}) if isinstance(pvg, dict) else {}
    if p_summary:
        by_family = p_summary.get("by_family", {})
        road = by_family.get("road", {})
        social = by_family.get("social", {})
        if road.get("n", 0):
            claims.append({
                "id": "popt_vs_grasp.road_family_mean",
                "category": "popt_vs_grasp",
                "text": (
                    "On road-family graphs POPT improves on GRASP by a "
                    "mean of %+.3f pp (min %+.3f, max %+.3f) across %d "
                    "cells - POPT's offline lookahead is decisive where "
                    "GRASP's hub-protection has nothing to grip." % (
                        road.get("mean_pp", 0.0),
                        road.get("min_pp", 0.0),
                        road.get("max_pp", 0.0),
                        road.get("n", 0),
                    )
                ),
                "value": _maybe_round(road.get("mean_pp", 0.0), 3),
                "units": "pp",
                "source": "wiki/data/popt_vs_grasp_delta.json",
                "gate": "scripts/test/test_popt_vs_grasp_delta.py",
            })
        if social.get("n", 0):
            claims.append({
                "id": "popt_vs_grasp.social_family_mean",
                "category": "popt_vs_grasp",
                "text": (
                    "On social-family graphs POPT's mean lift over GRASP "
                    "is only %+.3f pp across %d cells; the permutation "
                    "overhead is not always recovered, contradicting the "
                    "literature's blanket POPT_GE_GRASP claim." % (
                        social.get("mean_pp", 0.0), social.get("n", 0),
                    )
                ),
                "value": _maybe_round(social.get("mean_pp", 0.0), 3),
                "units": "pp",
                "source": "wiki/data/popt_vs_grasp_delta.json",
                "gate": "scripts/test/test_popt_vs_grasp_delta.py",
            })

    # ------------------------------------------------------------------
    # 7) Literature deviations
    # ------------------------------------------------------------------
    d_summary = deviations.get("summary", {}) if isinstance(deviations, dict) else {}
    if d_summary:
        n_dev = d_summary.get("n_deviations", 0)
        by_mech = d_summary.get("by_mechanism", {})
        popt_over = by_mech.get("popt_overhead_dominates", 0)
        if n_dev:
            claims.append({
                "id": "deviations.popt_overhead_share",
                "category": "deviations",
                "text": (
                    "%d / %d documented literature deviations (%.1f %%) "
                    "classify as `popt_overhead_dominates` - measured "
                    "POPT miss rate exceeds GRASP by more than the "
                    "claim tolerance." % (
                        popt_over, n_dev, 100.0 * popt_over / n_dev,
                    )
                ),
                "value": _maybe_round(100.0 * popt_over / n_dev, 1),
                "units": "percent",
                "source": "wiki/data/literature_deviations.json",
                "gate": "scripts/test/test_literature_deviations.py",
            })

    # ------------------------------------------------------------------
    # 8) Cross-tool saturation soundness
    # ------------------------------------------------------------------
    x_summary = crosstool.get("summary", {}) if isinstance(crosstool, dict) else {}
    if x_summary:
        doubly = x_summary.get("doubly_saturated_total", 0)
        disagrees_field = x_summary.get("disagreements", 0)
        # disagreements may be a list (per-cell records) or a count.
        disagrees = (
            len(disagrees_field) if isinstance(disagrees_field, list)
            else int(disagrees_field)
        )
        n = x_summary.get("n_cells", 0)
        if n:
            claims.append({
                "id": "cross_tool.doubly_saturated_agreement",
                "category": "cross_tool",
                "text": (
                    "%d / %d cross-tool cells reach simultaneous saturation "
                    "on cache_sim and either gem5 or Sniper; all %d "
                    "doubly-saturated cells agree on the GRASP-vs-LRU sign "
                    "within the 2 pp headline tolerance (%d disagreement(s))."
                    % (doubly, n, doubly, disagrees)
                ),
                "value": disagrees,
                "units": "disagreements",
                "source": "wiki/data/cross_tool_saturation.json",
                "gate": "scripts/test/test_cross_tool_saturation.py",
            })

    # ------------------------------------------------------------------
    # 9) Confidence dashboard rollup
    # ------------------------------------------------------------------
    suites = confidence.get("suites", []) if isinstance(confidence, dict) else []
    if suites:
        n_total = len(suites)
        # Each suite is a dict with passed_all bool (newer) OR failed count (older)
        def _green(s):
            if isinstance(s, dict):
                if "passed_all" in s:
                    return bool(s["passed_all"])
                return int(s.get("failed", 0)) == 0
            return False
        n_green = sum(1 for s in suites if _green(s))
        claims.append({
            "id": "confidence.green_gate_count",
            "category": "meta",
            "text": (
                "%d / %d confidence gates pass in the current run." % (
                    n_green, n_total,
                )
            ),
            "value": n_green,
            "units": "gates",
            "source": "wiki/data/confidence_dashboard.json",
            "gate": "scripts/experiments/ecg/confidence_dashboard.py",
        })
    return claims


def _write_json(claims: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "n_claims": len(claims),
        "claims": claims,
    }, indent=2, sort_keys=True))


def _write_md(claims: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Paper claims registry")
    lines.append("")
    lines.append(
        "_Generated by `scripts/experiments/ecg/paper_claims_registry.py` from "
        "the paper-grade aggregator JSONs. **Single source of truth** for "
        "every numerical claim the paper makes; if the paper text drifts "
        "from a row here, the listed gate will fail._"
    )
    lines.append("")
    by_cat: dict[str, list[dict]] = {}
    for c in claims:
        by_cat.setdefault(c.get("category", "other"), []).append(c)
    for cat in sorted(by_cat):
        lines.append(f"## {cat}")
        lines.append("")
        lines.append("| id | claim | value | source | gate |")
        lines.append("|---|---|---:|---|---|")
        for c in by_cat[cat]:
            val = c.get("value")
            units = c.get("units", "")
            val_str = (
                f"{val} {units}".strip() if val is not None else "—"
            )
            lines.append(
                f"| `{c['id']}` | {c['text']} | {val_str} | "
                f"`{c['source']}` | `{c.get('gate', '—')}` |"
            )
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DATA / "paper_claims.json",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=DATA / "paper_claims.md",
    )
    args = parser.parse_args()

    claims = build_claims()
    _write_json(claims, args.json_out)
    _write_md(claims, args.md_out)
    print(f"[paper-claims] registry has {len(claims)} claims")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
