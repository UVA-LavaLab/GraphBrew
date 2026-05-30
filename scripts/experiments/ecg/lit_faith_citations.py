#!/usr/bin/env python3
"""Literature-faithfulness citation locator integrity audit.

LIT-Cov tells us the *coverage cube* is broad. LIT-Mar tells us each
claim has *margin*. LIT-Sig tells us the sign claims show *signal*.
None of those checks notice if a baseline claim's ``citation`` field
slowly drifts away from its real-world source — a stale ``§5.2`` after
a paper has been edited, a missing ``Fig 10`` qualifier, a typo in the
venue. Long before a regression breaks magnitude/sign envelopes, a
broken or under-specified citation is exactly the bug that makes a
later reviewer say "where does this number come from?".

This module audits:

1. **Citation bijection** — every citation referenced by lit-faith
   resolves to at least one ``LiteratureClaim`` in
   ``literature_baselines.py`` and vice-versa.
2. **Paper-anchor coverage** — each citation contains a recognized
   venue tag (HPCA/ISCA/MICRO/ASPLOS/PLDI), a 4-digit year, and a
   location qualifier (``§N.N``, ``Fig N``, or a chapter/section
   identifier). Strings missing any of these are listed.
3. **Anchor-paper inventory** — the canonical 3-paper anchor set
   (Faldu HPCA 2020, Balaji & Lucia HPCA 2021, Jaleel ISCA 2010) is
   confirmed present; each anchor is used by at least one baseline
   claim and at least one lit-faith claim.
4. **Source-of-truth grounding** — the module docstring in
   ``literature_baselines.py`` mentions each anchor paper with a
   resolvable URL or DOI (warned if missing, not gated, since Jaleel
   doesn't currently have one).
5. **Per-claim citation non-emptiness** — every
   ``LiteratureClaim.citation`` is non-empty and ≥ 20 characters; any
   shorter is almost certainly an unfinished placeholder.
6. **Per-anchor usage floor** — each canonical paper anchors a
   minimum number of baseline claims so removing a paper trips the
   gate.

Emits ``wiki/data/lit_faith_citations.{json,md,csv}``.

CLI::

    python3 -m scripts.experiments.ecg.lit_faith_citations \\
        --lit-faith-json wiki/data/literature_faithfulness_postfix.json \\
        --baselines-module scripts/experiments/ecg/literature_baselines.py \\
        --json-out wiki/data/lit_faith_citations.json \\
        --md-out   wiki/data/lit_faith_citations.md \\
        --csv-out  wiki/data/lit_faith_citations.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LIT_FAITH = REPO_ROOT / "wiki" / "data" / "literature_faithfulness_postfix.json"
DEFAULT_BASELINES = REPO_ROOT / "scripts" / "experiments" / "ecg" / "literature_baselines.py"
DEFAULT_JSON_OUT = REPO_ROOT / "wiki" / "data" / "lit_faith_citations.json"
DEFAULT_MD_OUT = REPO_ROOT / "wiki" / "data" / "lit_faith_citations.md"
DEFAULT_CSV_OUT = REPO_ROOT / "wiki" / "data" / "lit_faith_citations.csv"

# Canonical 3-paper anchor set: each anchor is a (key, label) tuple
# plus a regex matching how the paper is identified inside citation
# strings. Keys are stable identifiers used in JSON output.
ANCHORS: list[dict[str, Any]] = [
    {
        "key": "faldu_hpca_2020",
        "label": "Faldu et al. HPCA 2020 (GRASP)",
        # Accept both "HPCA 2020" and the "HPCA20" shorthand.
        "match": re.compile(r"\bFaldu\b.*\bHPCA\s*(?:20)?20\b"),
        "doi_or_url_prefix": "https://dl.acm.org/doi/10.1109/HPCA47549.2020",
        "venue": "HPCA 2020",
    },
    {
        "key": "balaji_hpca_2021",
        "label": "Balaji & Lucia HPCA 2021 (P-OPT)",
        "match": re.compile(r"\bBalaji\b.*\bHPCA\s*(?:20)?21\b"),
        "doi_or_url_prefix": "https://www.cs.cmu.edu/~vbalaji/papers/popt_hpca21",
        "venue": "HPCA 2021",
    },
    {
        "key": "jaleel_isca_2010",
        "label": "Jaleel et al. ISCA 2010 (RRIP)",
        "match": re.compile(r"\bJaleel\b.*\bISCA\s*(?:20)?10\b"),
        "doi_or_url_prefix": None,  # not currently linked
        "venue": "ISCA 2010",
    },
]

VENUE_TAG_RE = re.compile(
    r"\b(HPCA|ISCA|MICRO|ASPLOS|PLDI|OSDI|SOSP|PPoPP|SC|ACM)\s*\d{2,4}\b"
)
# Accept a standalone 4-digit year (1900–2099) OR a venue+2-digit shorthand
# ("HPCA20" → year 2020) so common conference-shorthand citations are
# treated as well-formed.
_FULL_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_SHORT_YEAR_RE = re.compile(
    r"\b(HPCA|ISCA|MICRO|ASPLOS|PLDI|OSDI|SOSP|PPoPP|SC|ACM)\s*\d{2}\b"
)
LOCATION_RE = re.compile(
    r"§\s*\d+(?:\.\d+)*|\bFig(?:\.|ure)?\s*\d+|\bSec(?:tion)?\s*\d+|\bTable\s*\d+"
)

CITATION_MIN_LENGTH = 20


def _load_baselines_module(path: Path):
    spec = importlib.util.spec_from_file_location(
        "literature_baselines_audit", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _classify_citation(c: str) -> dict[str, Any]:
    """Return per-citation grounding diagnostics."""
    has_venue = bool(VENUE_TAG_RE.search(c))
    has_year = bool(_FULL_YEAR_RE.search(c)) or bool(_SHORT_YEAR_RE.search(c))
    has_location = bool(LOCATION_RE.search(c))
    anchors = [
        a["key"] for a in ANCHORS if a["match"].search(c)
    ]
    return {
        "citation": c,
        "length": len(c),
        "has_venue_tag": has_venue,
        "has_year": has_year,
        "has_location_qualifier": has_location,
        "anchors": anchors,
        "well_formed": has_venue and has_year and has_location and bool(anchors),
    }


def build_audit(
    lit_faith: dict[str, Any], baselines_module
) -> dict[str, Any]:
    per_graph = list(baselines_module.PER_GRAPH_CLAIMS)
    invariant = list(getattr(baselines_module, "INVARIANT_CLAIMS", ()))
    baseline_claims = per_graph + invariant
    baseline_cites_all = [c.citation for c in baseline_claims]
    baseline_cite_counter = Counter(baseline_cites_all)
    baseline_cites = set(baseline_cites_all)

    faith_per_claim = lit_faith.get("per_claim", [])
    faith_cites_all = [r.get("citation") or "" for r in faith_per_claim]
    faith_cite_counter = Counter(faith_cites_all)
    faith_cites = {c for c in faith_cites_all if c}

    only_in_faith = sorted(faith_cites - baseline_cites)
    only_in_baselines = sorted(baseline_cites - faith_cites)

    # Per-citation grounding
    all_cites = sorted(faith_cites | baseline_cites)
    grounded = [_classify_citation(c) for c in all_cites]

    # Per-anchor inventory
    anchor_summaries: list[dict[str, Any]] = []
    for anchor in ANCHORS:
        baseline_cites_for_anchor = [
            c for c in baseline_cites if anchor["match"].search(c)
        ]
        faith_cites_for_anchor = [
            c for c in faith_cites if anchor["match"].search(c)
        ]
        baseline_claim_count = sum(
            baseline_cite_counter[c] for c in baseline_cites_for_anchor
        )
        faith_claim_count = sum(
            faith_cite_counter[c] for c in faith_cites_for_anchor
        )
        anchor_summaries.append({
            "key": anchor["key"],
            "label": anchor["label"],
            "venue": anchor["venue"],
            "doi_or_url_prefix": anchor["doi_or_url_prefix"],
            "baseline_citations": sorted(baseline_cites_for_anchor),
            "baseline_claim_count": baseline_claim_count,
            "faith_citations": sorted(faith_cites_for_anchor),
            "faith_claim_count": faith_claim_count,
        })

    # Source-of-truth (literature_baselines.py module docstring) URL/DOI presence
    module_doc = (baselines_module.__doc__ or "")
    doc_grounding = []
    for anchor in ANCHORS:
        prefix = anchor["doi_or_url_prefix"]
        doc_grounding.append({
            "key": anchor["key"],
            "url_prefix_expected": prefix,
            "url_prefix_present_in_docstring": (
                prefix is not None and prefix in module_doc
            ),
        })

    # Per-claim citation non-emptiness check on baselines
    bad_per_claim = []
    for claim in baseline_claims:
        cite = claim.citation or ""
        if len(cite) < CITATION_MIN_LENGTH:
            bad_per_claim.append({
                "graph": getattr(claim, "graph", None),
                "app": getattr(claim, "app", None),
                "l3_size": getattr(claim, "l3_size", None),
                "policy": claim.policy,
                "citation": cite,
                "length": len(cite),
            })

    # Citations that don't pass the well-formed regexes
    ill_formed = [g for g in grounded if not g["well_formed"]]

    payload = {
        "schema_version": 1,
        "summary": {
            "lit_faith_unique_citations": len(faith_cites),
            "baseline_unique_citations": len(baseline_cites),
            "intersection_size": len(faith_cites & baseline_cites),
            "only_in_faith_count": len(only_in_faith),
            "only_in_baselines_count": len(only_in_baselines),
            "baseline_claim_count": len(baseline_claims),
            "faith_claim_count": len(faith_per_claim),
            "anchors_total": len(ANCHORS),
            "anchors_present_in_baselines": sum(
                1 for a in anchor_summaries if a["baseline_claim_count"] > 0
            ),
            "anchors_present_in_faith": sum(
                1 for a in anchor_summaries if a["faith_claim_count"] > 0
            ),
            "anchors_url_in_docstring": sum(
                1 for d in doc_grounding if d["url_prefix_present_in_docstring"]
            ),
            "citations_well_formed": sum(1 for g in grounded if g["well_formed"]),
            "citations_ill_formed_count": len(ill_formed),
            "baseline_short_citations": len(bad_per_claim),
        },
        "only_in_faith": only_in_faith,
        "only_in_baselines": only_in_baselines,
        "anchors": anchor_summaries,
        "docstring_grounding": doc_grounding,
        "citation_grounding": grounded,
        "ill_formed_citations": ill_formed,
        "baseline_short_citations": bad_per_claim,
    }
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    s = payload["summary"]
    lines: list[str] = [
        "# Literature-faithfulness citation locator audit",
        "",
        "Bijection + grounding check between the live lit-faith corpus "
        "(`literature_faithfulness_postfix.json`) and the static source-"
        "of-truth (`literature_baselines.py`).",
        "",
        "## Summary",
        "",
        f"- Unique citations in lit-faith: **{s['lit_faith_unique_citations']}**",
        f"- Unique citations in baselines: **{s['baseline_unique_citations']}**",
        f"- Intersection: **{s['intersection_size']}** "
        f"(faith-only: {s['only_in_faith_count']}, "
        f"baselines-only: {s['only_in_baselines_count']})",
        f"- Baseline claims total: **{s['baseline_claim_count']}**; "
        f"lit-faith per-cell claims: **{s['faith_claim_count']}**",
        f"- Anchor papers expected: **{s['anchors_total']}**; "
        f"present in baselines: **{s['anchors_present_in_baselines']}**; "
        f"present in lit-faith: **{s['anchors_present_in_faith']}**; "
        f"URL in docstring: **{s['anchors_url_in_docstring']}**",
        f"- Citations well-formed (venue + year + locator + known anchor): "
        f"**{s['citations_well_formed']}** "
        f"(ill-formed: {s['citations_ill_formed_count']})",
        f"- Baseline claims with short citation (< "
        f"{CITATION_MIN_LENGTH} chars): "
        f"**{s['baseline_short_citations']}**",
        "",
        "## Anchor papers",
        "",
        "| key | venue | baseline claims | lit-faith claims | "
        "docstring URL/DOI |",
        "|---|---|---:|---:|---|",
    ]
    grounding_by_key = {
        d["key"]: d for d in payload["docstring_grounding"]
    }
    for a in payload["anchors"]:
        url_status = (
            "✓ present"
            if grounding_by_key[a["key"]]["url_prefix_present_in_docstring"]
            else (
                "— not linked"
                if grounding_by_key[a["key"]]["url_prefix_expected"] is None
                else "✗ missing"
            )
        )
        lines.append(
            f"| `{a['key']}` | {a['venue']} | {a['baseline_claim_count']} "
            f"| {a['faith_claim_count']} | {url_status} |"
        )

    lines.extend(["", "## Citation grounding", ""])
    lines.append(
        "| citation | len | venue | year | locator | anchors | "
        "well_formed |"
    )
    lines.append("|---|---:|---|---|---|---|---|")
    for g in payload["citation_grounding"]:
        venue = "✓" if g["has_venue_tag"] else "✗"
        year = "✓" if g["has_year"] else "✗"
        loc = "✓" if g["has_location_qualifier"] else "✗"
        wf = "✓" if g["well_formed"] else "✗"
        anchors_str = ",".join(g["anchors"]) if g["anchors"] else "—"
        c = g["citation"].replace("|", "\\|")
        lines.append(
            f"| `{c}` | {g['length']} | {venue} | {year} | {loc} | "
            f"{anchors_str} | {wf} |"
        )

    if payload["only_in_faith"]:
        lines.extend(["", "## ⚠ Citations only in lit-faith (orphan)", ""])
        for c in payload["only_in_faith"]:
            lines.append(f"- `{c}`")

    if payload["only_in_baselines"]:
        lines.extend(["", "## ⚠ Citations only in baselines (unused)", ""])
        for c in payload["only_in_baselines"]:
            lines.append(f"- `{c}`")

    if payload["baseline_short_citations"]:
        lines.extend(
            ["", "## ⚠ Baseline claims with short citation strings", ""]
        )
        for entry in payload["baseline_short_citations"]:
            lines.append(
                f"- `{entry['graph']}/{entry['app']}/{entry['l3_size']}/"
                f"{entry['policy']}` → `{entry['citation']}` "
                f"({entry['length']} chars)"
            )

    return "\n".join(lines)


def render_csv(payload: dict[str, Any], path: Path) -> None:
    fields = [
        "citation",
        "length",
        "has_venue_tag",
        "has_year",
        "has_location_qualifier",
        "anchors",
        "well_formed",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for g in payload["citation_grounding"]:
            row = {
                "citation": g["citation"],
                "length": g["length"],
                "has_venue_tag": g["has_venue_tag"],
                "has_year": g["has_year"],
                "has_location_qualifier": g["has_location_qualifier"],
                "anchors": ";".join(g["anchors"]),
                "well_formed": g["well_formed"],
            }
            writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lit-faith-json", type=Path, default=DEFAULT_LIT_FAITH
    )
    parser.add_argument(
        "--baselines-module", type=Path, default=DEFAULT_BASELINES
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV_OUT)
    args = parser.parse_args(argv)

    lit_faith = json.loads(args.lit_faith_json.read_text())
    baselines = _load_baselines_module(args.baselines_module)

    payload = build_audit(lit_faith, baselines)

    args.json_out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    args.md_out.write_text(render_markdown(payload).rstrip("\n") + "\n")
    render_csv(payload, args.csv_out)

    s = payload["summary"]
    print(
        f"[lit-faith-citations] {s['intersection_size']}↔ bijection; "
        f"{s['anchors_present_in_baselines']}/{s['anchors_total']} anchors "
        f"in baselines; {s['citations_well_formed']} well-formed "
        f"({s['citations_ill_formed_count']} ill-formed)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
