#!/usr/bin/env python3
"""Gate 246 — lit-faith citation registry purity.

`wiki/data/literature_faithfulness_postfix.json` carries a `per_claim`
array where every row describes one (policy, app, graph, l3_size)
literature-comparison cell. Each row has a free-form `citation` string
identifying which prior work the *expected sign* was lifted from
(e.g. ``Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check``).

Today nothing checks those citation strings: an author can hand-edit a
row to point at a paper that does not actually carry the claim, or
silently rewrite a citation while refining a row, and no test fails.

Worse, within a (policy, app, expected_sign) bucket — which is the
unit the paper actually quotes — different rows can drift apart in
which canonical paper they claim as the source, without any of them
becoming "wrong" enough to fail a numerical test.

This gate codifies the situation with a hand-curated registry:

  * every canonical literature work the lit-faith table cites is
    registered in ``CITATION_REGISTRY`` with a ``key``, a list of
    ``patterns`` (substrings that must appear in a `per_claim`
    citation string to count as "references this work"), and
    structured ``venue/year/note`` metadata;
  * every `per_claim` citation string must match the patterns of
    at least one registered canonical work (C1);
  * every registered canonical work must be referenced by at least
    one `per_claim` row (C2 — no dead-letter registry entries);
  * within each (policy, app, expected_sign) bucket, all rows must
    share at least one canonical citation key (C3 — the paper quote
    "policy X behaves like Y on app Z per [cite]" stays anchored);
  * every registered work has a non-empty ``venue`` and ``year``
    (C4 — keeps the registry mineable for later bibliography
    generation);
  * every `per_claim` row carries a non-empty ``citation`` (C5).

Source-of-truth: `wiki/data/literature_faithfulness_postfix.json`
loaded as JSON. The registry itself is the second source-of-truth and
lives in this file.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2].parent
WIKI_DATA = ROOT / "wiki" / "data"
POSTFIX_JSON = WIKI_DATA / "literature_faithfulness_postfix.json"


# ----------------------------------------------------------- registry --

# Each entry declares one canonical literature work the lit-faith table
# is allowed to cite. ``patterns`` are case-sensitive substrings; a
# citation string matches the entry if ANY pattern appears in it.
CITATION_REGISTRY: list[dict] = [
    {
        "key":      "Faldu-HPCA-2020",
        "title":    "Domain-Specialized Cache Management for Graph Analytics (GRASP)",
        "venue":    "HPCA",
        "year":     2020,
        "patterns": ["Faldu HPCA20", "Faldu et al. HPCA 2020"],
        "note":     "GRASP source paper. Cited for hub-prioritization and "
                    "scan-resistance behavior on PR/BC/Radii kernels.",
    },
    {
        "key":      "Balaji-HPCA-2021",
        "title":    "P-OPT: Practical Optimal Cache Replacement for Graph Analytics",
        "venue":    "HPCA",
        "year":     2021,
        "patterns": ["Balaji HPCA21", "Balaji & Lucia HPCA 2021"],
        "note":     "POPT source paper. Cited for transpose-driven priorities "
                    "and the practical-optimal upper-bound comparison.",
    },
    {
        "key":      "Jaleel-ISCA-2010",
        "title":    "High Performance Cache Replacement Using Re-Reference "
                    "Interval Prediction (RRIP)",
        "venue":    "ISCA",
        "year":     2010,
        "patterns": ["Jaleel et al. ISCA 2010", "Jaleel ISCA 2010",
                     "Jaleel ISCA10"],
        "note":     "SRRIP/DRRIP source paper. Cited for the scan-resistance "
                    "argument that we extend to BC/CC.",
    },
]


# ----------------------------------------------------------- helpers --

def _load_per_claim() -> list[dict]:
    raw = json.loads(POSTFIX_JSON.read_text())
    return list(raw.get("per_claim", []))


def _keys_matching(citation: str) -> list[str]:
    """Return the list of registered canonical-work keys whose patterns
    appear inside the citation string."""
    hits: list[str] = []
    for e in CITATION_REGISTRY:
        if any(p in citation for p in e["patterns"]):
            hits.append(e["key"])
    return hits


# ----------------------------------------------------------- rules --

def _rule_c1(rows: list[dict]) -> list[dict]:
    """Every per_claim citation must match ≥1 registered canonical key."""
    out: list[dict] = []
    for r in rows:
        cite = r.get("citation", "") or ""
        if not _keys_matching(cite):
            out.append({"rule": "C1",
                        "policy": r.get("policy"),
                        "app": r.get("app"),
                        "graph": r.get("graph"),
                        "l3_size": r.get("l3_size"),
                        "citation": cite[:160],
                        "issue": "citation matches no registered canonical work"})
    return out


def _rule_c2(rows: list[dict]) -> list[dict]:
    """Every registered canonical work must be referenced ≥1 time."""
    out: list[dict] = []
    seen: set[str] = set()
    for r in rows:
        for k in _keys_matching(r.get("citation", "") or ""):
            seen.add(k)
    for e in CITATION_REGISTRY:
        if e["key"] not in seen:
            out.append({"rule": "C2", "key": e["key"],
                        "issue": "registered canonical work has zero "
                                 "per_claim references — dead-letter entry"})
    return out


def _rule_c3(rows: list[dict]) -> list[dict]:
    """Within each (policy, app, expected_sign) bucket, all rows must
    share ≥1 canonical citation key."""
    out: list[dict] = []
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        k = (r.get("policy"), r.get("app"), r.get("expected_sign"))
        buckets[k].append(r)
    for key, members in sorted(buckets.items()):
        if len(members) < 2:
            continue
        per_member = [set(_keys_matching(r.get("citation", "") or ""))
                      for r in members]
        shared = set.intersection(*per_member) if per_member else set()
        if not shared:
            out.append({"rule": "C3",
                        "policy": key[0], "app": key[1],
                        "expected_sign": key[2],
                        "member_count": len(members),
                        "per_member_keys": [sorted(s) for s in per_member],
                        "issue": "bucket members do not share any "
                                 "canonical citation key"})
    return out


def _rule_c4(_rows: list[dict]) -> list[dict]:
    """Every registry entry has non-empty venue + year."""
    out: list[dict] = []
    for e in CITATION_REGISTRY:
        missing = [f for f in ("venue", "year") if not e.get(f)]
        if missing:
            out.append({"rule": "C4", "key": e["key"],
                        "missing": missing,
                        "issue": "registry entry missing bibliographic field(s)"})
    return out


def _rule_c5(rows: list[dict]) -> list[dict]:
    """Every per_claim row carries a non-empty citation."""
    out: list[dict] = []
    for r in rows:
        cite = (r.get("citation") or "").strip()
        if not cite:
            out.append({"rule": "C5",
                        "policy": r.get("policy"),
                        "app": r.get("app"),
                        "graph": r.get("graph"),
                        "l3_size": r.get("l3_size"),
                        "issue": "empty citation string"})
    return out


# ----------------------------------------------------------- audit --

def audit() -> dict:
    if not POSTFIX_JSON.exists():
        return {
            "status": "skip",
            "reason": f"{POSTFIX_JSON.relative_to(ROOT)} not on disk",
            "registry_size": len(CITATION_REGISTRY),
            "rules": {},
            "totals": {"row_count": 0, "registry_size": len(CITATION_REGISTRY),
                       "violations": 0},
            "violations": [],
        }

    rows = _load_per_claim()
    violations: list[dict] = []
    violations.extend(_rule_c1(rows))
    violations.extend(_rule_c2(rows))
    violations.extend(_rule_c3(rows))
    violations.extend(_rule_c4(rows))
    violations.extend(_rule_c5(rows))

    # coverage table: how many per_claim rows reference each key
    coverage: dict[str, int] = {e["key"]: 0 for e in CITATION_REGISTRY}
    for r in rows:
        for k in _keys_matching(r.get("citation", "") or ""):
            coverage[k] = coverage.get(k, 0) + 1

    # bucket diagnostics
    bucket_keys: dict[str, list[str]] = {}
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[(r.get("policy"), r.get("app"),
                 r.get("expected_sign"))].append(r)
    for key, members in sorted(buckets.items()):
        per = [set(_keys_matching(r.get("citation", "") or ""))
               for r in members]
        shared = sorted(set.intersection(*per)) if per else []
        bucket_keys[f"{key[0]}|{key[1]}|{key[2]}"] = shared

    return {
        "status": "active",
        "rules": {
            "C1": "every per_claim citation matches >=1 registered canonical work",
            "C2": "every registered canonical work is referenced >=1 time",
            "C3": "within (policy, app, expected_sign) bucket, all rows share >=1 canonical key",
            "C4": "every registry entry has non-empty venue + year",
            "C5": "every per_claim row carries a non-empty citation",
        },
        "registry":           CITATION_REGISTRY,
        "registry_size":      len(CITATION_REGISTRY),
        "row_count":          len(rows),
        "coverage_per_key":   coverage,
        "shared_keys_per_bucket": bucket_keys,
        "totals": {
            "registry_size": len(CITATION_REGISTRY),
            "row_count":     len(rows),
            "bucket_count":  len(buckets),
            "violations":    len(violations),
        },
        "violations": violations,
    }


# ----------------------------------------------------------- writers --

def _render_md(audit: dict) -> str:
    L: list[str] = []
    L.append("# lit-faith citation registry purity (gate 246)")
    L.append("")
    t = audit["totals"]
    L.append(f"**Status:** {audit['status']}  •  "
             f"registered works: {t['registry_size']}  •  "
             f"per_claim rows: {t['row_count']}  •  "
             f"violations: {t['violations']}")
    L.append("")
    L.append("## Rules")
    for k, v in audit.get("rules", {}).items():
        L.append(f"- **{k}** — {v}")
    L.append("")
    L.append("## Registered canonical works")
    L.append("")
    L.append("| key | venue | year | per_claim references |")
    L.append("|---|---|---:|---:|")
    cov = audit.get("coverage_per_key", {})
    for e in audit.get("registry", []):
        L.append(f"| `{e['key']}` | {e['venue']} | {e['year']} | "
                 f"{cov.get(e['key'], 0)} |")
    L.append("")
    bk = audit.get("shared_keys_per_bucket", {})
    if bk:
        L.append("## Shared canonical keys per (policy, app, sign) bucket")
        L.append("")
        L.append("| bucket | shared canonical keys |")
        L.append("|---|---|")
        for k in sorted(bk):
            shared = ", ".join(f"`{x}`" for x in bk[k]) or "_(none)_"
            L.append(f"| `{k}` | {shared} |")
        L.append("")
    if audit.get("violations"):
        L.append("## Violations")
        for v in audit["violations"]:
            L.append(f"- {v}")
    else:
        L.append("**0 violations** — every citation maps to a registered "
                "canonical work, every registered work is referenced, "
                "every (policy, app, sign) bucket is internally consistent.")
    return "\n".join(L) + "\n"


def _render_csv(audit: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["field", "value"])
    t = audit["totals"]
    for k in ("registry_size", "row_count", "bucket_count", "violations"):
        w.writerow([k, t[k]])
    cov = audit.get("coverage_per_key", {})
    for k in sorted(cov):
        w.writerow([f"coverage::{k}", cov[k]])
    return buf.getvalue()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json-out", required=True)
    p.add_argument("--md-out",   required=True)
    p.add_argument("--csv-out",  required=True)
    args = p.parse_args()
    a = audit()
    Path(args.json_out).write_text(json.dumps(a, indent=2, sort_keys=True) + "\n")
    Path(args.md_out).write_text(_render_md(a))
    Path(args.csv_out).write_text(_render_csv(a))
    print(f"[lit-faith-citation-registry] status={a['status']} "
          f"works={a['totals']['registry_size']} "
          f"rows={a['totals']['row_count']} "
          f"buckets={a['totals'].get('bucket_count', 0)} "
          f"violations={a['totals']['violations']}")


if __name__ == "__main__":
    main()
