"""Literature-faithfulness citation/date audit (gate 237 — LIT-CitDate).

Per_claim citation-field structural audit. For every row in
`per_claim`, the `citation` string must:

  D1 — be non-empty (schema floor)
  D2 — parse cleanly to (primary_author, venue, year), handling both the
       canonical "Faldu et al. HPCA 2020" form and the compact
       "Faldu HPCA20" form
  D3 — name a top-tier architecture venue ∈ VENUE_WHITELIST
  D4 — name a year inside YEAR_RANGE
  D5 — for the originator policies (GRASP/POPT/SRRIP), the citation must
       reference the policy's *originating* publication (Faldu HPCA 2020 /
       Balaji & Lucia HPCA 2021 / Jaleel ISCA 2010 respectively). Derived
       policies (POPT_GE_GRASP, POPT_NEAR_GRASP_IF_BIG_GAP) may cite either
       Balaji 2021 or a cross-citation that mentions Balaji 2021.
  D6 — contain at least one locator (§N, Fig N, Tab N, Table N, Section N)
  D7 — the corpus as a whole must use at least DISTINCT_CITATION_FLOOR
       different citation strings (anti-degeneration)

This complements LIT-Cite (gate 224, citation *locator* integrity) by adding
*venue/year/author* parseability and per-policy publication match: a row
citing GRASP's claim with a Balaji-only reference now fails the build.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

VENUE_WHITELIST = {"HPCA", "ISCA", "MICRO", "ASPLOS", "SC"}
YEAR_RANGE = (2005, 2026)
DISTINCT_CITATION_FLOOR = 10

# Originating publication per policy. For derived policies the value is a
# *cross-citation expectation*: the citation may name Balaji-2021 OR
# include a cross-validation against the named cross-source.
POLICY_ORIGIN = {
    "GRASP":                       {"author": "Faldu",   "venue": "HPCA", "year": 2020},
    "POPT":                        {"author": "Balaji",  "venue": "HPCA", "year": 2021},
    "POPT_GE_GRASP":               {"author": "Balaji",  "venue": "HPCA", "year": 2021},
    "POPT_NEAR_GRASP_IF_BIG_GAP":  {"author": "Balaji",  "venue": "HPCA", "year": 2021},
    "SRRIP":                       {"author": "Jaleel",  "venue": "ISCA", "year": 2010},
}

# Canonical form: "Faldu et al. HPCA 2020 ..." or "Balaji & Lucia HPCA 2021 ..."
# Compact form:   "Faldu HPCA20 ..."  (no "et al.", glued year)
_AUTHOR_TOK = r"(?P<author>Faldu|Balaji|Jaleel|Khan|Wu|Beckmann|Khan & Lucia)"
_VENUE_TOK = r"(?P<venue>HPCA|ISCA|MICRO|ASPLOS|SC)"
_CANON_PAT = re.compile(
    rf"{_AUTHOR_TOK}(?:\s*&\s*\w+)?(?:\s+et\s+al\.)?\s+{_VENUE_TOK}\s*(?P<year>(?:20\d{{2}}|\d{{2}}))"
)
_LOCATOR_PAT = re.compile(r"(§\d|Fig\s*\d|Tab\s*\d|Table\s*\d|Section\s*\d)")


def _normalize_year(raw: str) -> int:
    if len(raw) == 2:
        return 2000 + int(raw)
    return int(raw)


def _parse_citation(citation: str) -> list[dict[str, Any]]:
    """Extract every (author, venue, year) tuple mentioned in the string.

    Cross-citations may include multiple tuples (e.g. "Jaleel ISCA 2010;
    Faldu HPCA 2020"). All are returned.
    """
    return [
        {
            "author": m.group("author").split("&")[0].split()[0].strip(),
            "venue":  m.group("venue"),
            "year":   _normalize_year(m.group("year")),
        }
        for m in _CANON_PAT.finditer(citation)
    ]


def audit(lit_faith: dict[str, Any]) -> dict[str, Any]:
    rows = lit_faith.get("per_claim", [])
    violations: list[dict[str, Any]] = []
    per_row: list[dict[str, Any]] = []
    distinct_citations: set[str] = set()

    for idx, row in enumerate(rows):
        policy = row.get("policy", "")
        citation = (row.get("citation") or "").strip()

        # D1 — schema
        if not citation:
            violations.append({
                "rule": "D1-empty-citation", "idx": idx,
                "graph": row.get("graph"), "app": row.get("app"),
                "policy": policy, "citation": citation,
            })
            per_row.append({"idx": idx, "policy": policy, "citation": citation, "parsed_ok": False})
            continue

        distinct_citations.add(citation)
        parses = _parse_citation(citation)

        # D2 — parseability
        if not parses:
            violations.append({
                "rule": "D2-unparseable-citation", "idx": idx,
                "graph": row.get("graph"), "app": row.get("app"),
                "policy": policy, "citation": citation,
            })
            per_row.append({"idx": idx, "policy": policy, "citation": citation, "parsed_ok": False})
            continue

        # D3 / D4 — venue whitelist + year range
        for p in parses:
            if p["venue"] not in VENUE_WHITELIST:
                violations.append({
                    "rule": "D3-bad-venue", "idx": idx, "policy": policy,
                    "citation": citation, "bad_venue": p["venue"],
                })
            if not (YEAR_RANGE[0] <= p["year"] <= YEAR_RANGE[1]):
                violations.append({
                    "rule": "D4-year-out-of-range", "idx": idx, "policy": policy,
                    "citation": citation, "bad_year": p["year"],
                })

        # D5 — per-policy origin match (or explicit cross-attribution)
        expected = POLICY_ORIGIN.get(policy)
        if expected is not None:
            origin_matched = any(
                p["author"] == expected["author"]
                and p["venue"]  == expected["venue"]
                and p["year"]   == expected["year"]
                for p in parses
            )
            # Cross-attribution exception: a citation may reference the
            # originator's data via a comparison paper if the policy name is
            # *explicitly* called out in the citation string (e.g.
            # "Balaji & Lucia HPCA 2021 Fig 10 (GRASP bar)").
            policy_token = policy.split("_")[0]  # "POPT_GE_GRASP" -> "POPT"
            cross_attributed = (
                not origin_matched
                and re.search(rf"\b{re.escape(policy_token)}\b", citation) is not None
                and any(p["venue"] in VENUE_WHITELIST for p in parses)
            )
            if not (origin_matched or cross_attributed):
                violations.append({
                    "rule": "D5-policy-origin-mismatch", "idx": idx, "policy": policy,
                    "citation": citation, "expected": expected, "found": parses,
                })

        # D6 — locator present
        if not _LOCATOR_PAT.search(citation):
            violations.append({
                "rule": "D6-no-locator", "idx": idx, "policy": policy,
                "citation": citation,
            })

        per_row.append({
            "idx": idx, "policy": policy, "citation": citation,
            "parsed_ok": True, "parses": parses,
        })

    # D7 — distinct citation floor
    if len(distinct_citations) < DISTINCT_CITATION_FLOOR:
        violations.append({
            "rule": "D7-distinct-citation-floor",
            "distinct_citations": len(distinct_citations),
            "floor": DISTINCT_CITATION_FLOOR,
        })

    # Per-policy roll-up
    by_policy: dict[str, dict[str, Any]] = defaultdict(lambda: {"rows": 0, "distinct_citations": set(), "parses_ok": 0})
    for r in per_row:
        bp = by_policy[r["policy"]]
        bp["rows"] += 1
        bp["distinct_citations"].add(r["citation"])
        if r["parsed_ok"]:
            bp["parses_ok"] += 1
    by_policy_out = {
        p: {
            "rows": v["rows"],
            "distinct_citations": sorted(v["distinct_citations"]),
            "distinct_citation_count": len(v["distinct_citations"]),
            "parses_ok": v["parses_ok"],
        }
        for p, v in sorted(by_policy.items())
    }

    venue_tally  = Counter(p["venue"] for r in per_row if r["parsed_ok"] for p in r.get("parses", []))
    year_tally   = Counter(p["year"]  for r in per_row if r["parsed_ok"] for p in r.get("parses", []))
    author_tally = Counter(p["author"] for r in per_row if r["parsed_ok"] for p in r.get("parses", []))

    return {
        "rules": {
            "D1": "every per_claim row has non-empty citation",
            "D2": "citation parses to at least one (author, venue, year) tuple",
            "D3": f"every parsed venue ∈ {sorted(VENUE_WHITELIST)}",
            "D4": f"every parsed year ∈ [{YEAR_RANGE[0]}, {YEAR_RANGE[1]}]",
            "D5": "originator policy citations match the originating publication "
                  "(GRASP→Faldu HPCA 2020, POPT[+derived]→Balaji HPCA 2021, "
                  "SRRIP→Jaleel ISCA 2010), OR the citation is an explicit "
                  "cross-attribution where the policy name appears in the "
                  "citation string AND the parsed venue is in the whitelist",
            "D6": "every citation contains a locator (§N | Fig N | Tab N | Table N | Section N)",
            "D7": f"corpus uses at least {DISTINCT_CITATION_FLOOR} distinct citation strings",
        },
        "constants": {
            "venue_whitelist": sorted(VENUE_WHITELIST),
            "year_range": list(YEAR_RANGE),
            "distinct_citation_floor": DISTINCT_CITATION_FLOOR,
            "policy_origin": POLICY_ORIGIN,
        },
        "totals": {
            "rows": len(rows),
            "rows_parsed_ok": sum(1 for r in per_row if r["parsed_ok"]),
            "distinct_citations": sorted(distinct_citations),
            "distinct_citation_count": len(distinct_citations),
            "venue_tally":  dict(sorted(venue_tally.items())),
            "year_tally":   {str(k): v for k, v in sorted(year_tally.items())},
            "author_tally": dict(sorted(author_tally.items())),
        },
        "by_policy": by_policy_out,
        "violations": violations,
    }


def render_markdown(audit_obj: dict[str, Any]) -> str:
    t = audit_obj["totals"]
    c = audit_obj["constants"]
    lines = [
        "# Literature-faithfulness citation/date audit",
        "",
        "Per_claim citation-field structural audit (gate 237 — LIT-CitDate).",
        "",
        "## Rules",
    ]
    for rid, desc in audit_obj["rules"].items():
        lines.append(f"- **{rid}** — {desc}")
    lines.extend([
        "",
        "## Constants",
        f"- venue whitelist: `{', '.join(c['venue_whitelist'])}`",
        f"- year range: `{c['year_range'][0]}..{c['year_range'][1]}`",
        f"- distinct-citation floor: `{c['distinct_citation_floor']}`",
        "",
        "## Totals",
        f"- per_claim rows: **{t['rows']}**",
        f"- rows parsed cleanly: **{t['rows_parsed_ok']}**",
        f"- distinct citation strings: **{t['distinct_citation_count']}**",
        f"- venue tally: `{t['venue_tally']}`",
        f"- year tally: `{t['year_tally']}`",
        f"- author tally: `{t['author_tally']}`",
        "",
        "## Per policy",
        "",
        "| policy | rows | parses ok | distinct citations |",
        "| --- | ---: | ---: | ---: |",
    ])
    for pol, v in audit_obj["by_policy"].items():
        lines.append(f"| {pol} | {v['rows']} | {v['parses_ok']} | {v['distinct_citation_count']} |")

    lines.extend([
        "",
        "## Violations",
        "",
    ])
    if not audit_obj["violations"]:
        lines.append("_None._")
    else:
        lines.append("| rule | policy | citation | detail |")
        lines.append("| --- | --- | --- | --- |")
        for v in audit_obj["violations"]:
            lines.append(
                f"| {v.get('rule','')} | {v.get('policy','')} | "
                f"`{v.get('citation','')}` | "
                f"{ {k: v[k] for k in v if k not in ('rule','policy','citation','idx')} } |"
            )

    lines.append("")
    return "\n".join(lines)


def render_csv(audit_obj: dict[str, Any]) -> str:
    head = "policy,rows,parses_ok,distinct_citation_count\n"
    body = "\n".join(
        f"{pol},{v['rows']},{v['parses_ok']},{v['distinct_citation_count']}"
        for pol, v in audit_obj["by_policy"].items()
    )
    return head + body + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lit-faith-json", type=Path, required=True)
    ap.add_argument("--json-out",       type=Path, required=True)
    ap.add_argument("--md-out",         type=Path, required=True)
    ap.add_argument("--csv-out",        type=Path, required=True)
    args = ap.parse_args()

    lit = json.loads(args.lit_faith_json.read_text())
    out = audit(lit)
    args.json_out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(render_markdown(out))
    args.csv_out.write_text(render_csv(out))
    print(
        f"[lit-faith-citdate] rows={out['totals']['rows']} "
        f"distinct_cit={out['totals']['distinct_citation_count']} "
        f"venues={list(out['totals']['venue_tally'])} "
        f"violations={len(out['violations'])}"
    )
    return 1 if out["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
