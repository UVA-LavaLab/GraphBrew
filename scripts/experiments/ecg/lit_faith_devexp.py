#!/usr/bin/env python3
"""LIT-DevExp (gate 232): deviation-explanation depth audit.

Every `status == 'known_deviation'` row in the literature-faithfulness
per_claim table must carry a `known_deviation_reason` that names at
least one *algorithmic* mechanism — not just paraphrase the magnitude.

This sister-audit to LIT-Dev (gate 225) operates at the per-row level:

  R1 length-floor
        len(reason.strip()) >= MIN_REASON_LEN

  R2 mechanism-vocabulary floor
        reason must contain >= MIN_MECHANISM_HITS keywords drawn from
        MECHANISM_VOCAB (PR-rank, frontier, hub, union-find, ordering,
        capacity, static schedule, look-ahead, ...)

  R3 magnitude-only ban
        reason must NOT consist entirely of digit/unit tokens

  R4 citation present
        non-empty `citation` field

  R5 cross-reference resolvability
        if the reason says "same as ... entry" / "as the ... above",
        another known_deviation row with the same (app) or (graph) key
        must exist

  R6 reuse ceiling
        the same reason text may not cover > REUSE_CEILING_FRAC of all
        known_deviation rows (catches lazy "ditto" patterns)

  R7 known_deviation row-count floor
        >= MIN_KNOWN_DEVIATIONS rows so the audit isn't trivially empty

Emits wiki/data/lit_faith_devexp.{json,md,csv}.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


MIN_REASON_LEN          = 60     # characters after strip
MIN_MECHANISM_HITS      = 1
MIN_KNOWN_DEVIATIONS    = 15
REUSE_CEILING_FRAC      = 0.50

MECHANISM_VOCAB: list[str] = [
    # graph-structural mechanisms
    "PR-rank",   "pr-rank",   "page-rank",   "pagerank",   "page rank",
    "hub",       "hub-spoke", "no-hub",      "power-law",  "skew",
    # access-pattern mechanisms
    "frontier",  "traversal", "union-find",  "static schedule",
    "ordering",  "mis-alignment", "misalignment",
    "look-ahead","lookahead", "oracle",      "static schedule",
    # cache-mechanism vocabulary
    "capacity",  "compulsory","warm-up",     "warmup",
    "evict",     "reuse",     "phase-transition", "phase transition",
    # algorithmic family identifiers
    "BC frontier", "BFS frontier", "PR-rank schedule", "edge-driven",
    "source-rooted", "depth-first", "breadth-first",
]


CROSS_REF_RE = re.compile(
    r"(same as.*?(entry|entries)|"
    r"as the .*?(entry|entries)|"
    r"see above|"
    r"see KNOWN_DEVIATIONS|"
    r"ditto)",
    re.IGNORECASE,
)


def _count_mechanism_hits(text: str) -> int:
    lower = text.lower()
    hits  = 0
    for kw in MECHANISM_VOCAB:
        if kw.lower() in lower:
            hits += 1
    return hits


def _looks_magnitude_only(text: str) -> bool:
    """A reason is magnitude-only if every token is digits, units, or
    common stopwords."""
    stop = {"gap", "is", "at", "the", "of", "a", "an", "to", "and",
            "pp", "%", "~", "~", ".", ","}
    tokens = re.findall(r"[A-Za-z]+", text)
    content_words = [
        t for t in tokens
        if t.lower() not in stop and len(t) >= 4
    ]
    return len(content_words) == 0


def audit(per_claim: list[dict[str, Any]]) -> dict[str, Any]:
    known = [r for r in per_claim if r.get("status") == "known_deviation"]
    n = len(known)

    violations: list[dict[str, Any]] = []
    per_row: list[dict[str, Any]] = []

    # Pre-build index for cross-reference resolvability.
    by_app:   dict[str, list[dict[str, Any]]] = {}
    by_graph: dict[str, list[dict[str, Any]]] = {}
    for r in known:
        by_app.setdefault(r.get("app", ""), []).append(r)
        by_graph.setdefault(r.get("graph", ""), []).append(r)

    reason_counter: Counter[str] = Counter(
        (r.get("known_deviation_reason", "") or "").strip()
        for r in known
    )

    for r in known:
        reason = (r.get("known_deviation_reason") or "").strip()
        row_id = (r.get("graph"), r.get("app"), r.get("l3_size"),
                  r.get("policy"))
        mech_hits = _count_mechanism_hits(reason)
        row_record: dict[str, Any] = {
            "graph":         r.get("graph"),
            "app":           r.get("app"),
            "l3_size":       r.get("l3_size"),
            "policy":        r.get("policy"),
            "citation":      r.get("citation"),
            "reason_len":    len(reason),
            "mechanism_hits":mech_hits,
            "cross_ref":     bool(CROSS_REF_RE.search(reason)),
            "reason_excerpt": (reason[:80] + "...") if len(reason) > 80 else reason,
        }
        per_row.append(row_record)

        if len(reason) < MIN_REASON_LEN:
            violations.append({"row_id": row_id, "rule": "R1_length_floor",
                               "observed": len(reason),
                               "floor": MIN_REASON_LEN})
        if mech_hits < MIN_MECHANISM_HITS:
            violations.append({"row_id": row_id, "rule": "R2_mechanism_vocab",
                               "observed": mech_hits,
                               "floor": MIN_MECHANISM_HITS})
        if _looks_magnitude_only(reason):
            violations.append({"row_id": row_id, "rule": "R3_magnitude_only",
                               "reason_excerpt": row_record["reason_excerpt"]})
        if not (r.get("citation") or "").strip():
            violations.append({"row_id": row_id, "rule": "R4_citation_present"})
        if row_record["cross_ref"]:
            # Find at least one sibling known_deviation row by app or graph
            siblings = [
                s for s in (by_app.get(r.get("app"), [])
                            + by_graph.get(r.get("graph"), []))
                if (s.get("graph"), s.get("app"), s.get("l3_size"),
                    s.get("policy")) != row_id
            ]
            if not siblings:
                violations.append({"row_id": row_id,
                                   "rule": "R5_cross_ref_unresolved"})

    # R6 reuse ceiling
    ceiling = int(REUSE_CEILING_FRAC * n) if n else 0
    for text, count in reason_counter.most_common():
        if count > ceiling and n >= MIN_KNOWN_DEVIATIONS:
            violations.append({"row_id": None,
                               "rule": "R6_reuse_ceiling",
                               "observed": count,
                               "ceiling": ceiling,
                               "reason_excerpt":
                                   text[:80] + ("..." if len(text) > 80 else "")})

    # R7 row-count floor
    if n < MIN_KNOWN_DEVIATIONS:
        violations.append({"row_id": None,
                           "rule": "R7_known_deviations_floor",
                           "observed": n,
                           "floor": MIN_KNOWN_DEVIATIONS})

    summary = {
        "min_reason_len":         MIN_REASON_LEN,
        "min_mechanism_hits":     MIN_MECHANISM_HITS,
        "min_known_deviations":   MIN_KNOWN_DEVIATIONS,
        "reuse_ceiling_frac":     REUSE_CEILING_FRAC,
        "known_deviation_rows":   n,
        "violations":             len(violations),
        "median_reason_len":      (sorted([r["reason_len"] for r in per_row])
                                   [n // 2] if n else None),
        "median_mechanism_hits":  (sorted([r["mechanism_hits"] for r in per_row])
                                   [n // 2] if n else None),
        "rows_with_cross_ref":    sum(1 for r in per_row if r["cross_ref"]),
        "unique_reason_texts":    len(reason_counter),
        "max_reuse_count":        max(reason_counter.values()) if reason_counter else 0,
    }

    return {
        "schema_version": 1,
        "summary":        summary,
        "rows":           per_row,
        "violations":     violations,
    }


def _to_markdown(a: dict[str, Any]) -> str:
    s = a["summary"]
    lines = ["# Literature-faithfulness deviation-explanation audit (LIT-DevExp)",
             "",
             "Per `status == 'known_deviation'` row: the "
             "`known_deviation_reason` text must name at least one "
             "algorithmic mechanism, exceed a length floor, carry a "
             "non-empty citation, resolve any cross-references, and the "
             "same reason text may not cover more than half the rows.",
             "",
             "## Summary", "",
             "| Metric | Value |", "|---|---|",
             f"| Known-deviation rows | {s['known_deviation_rows']} |",
             f"| Min reason length | {s['min_reason_len']} chars |",
             f"| Min mechanism hits | {s['min_mechanism_hits']} |",
             f"| Reuse ceiling fraction | {s['reuse_ceiling_frac']} |",
             f"| Median reason length | {s['median_reason_len']} chars |",
             f"| Median mechanism hits | {s['median_mechanism_hits']} |",
             f"| Unique reason texts | {s['unique_reason_texts']} |",
             f"| Max reuse count (single text) | {s['max_reuse_count']} |",
             f"| Rows with cross-reference | {s['rows_with_cross_ref']} |",
             f"| Violations | {s['violations']} |",
             "",
             "## Per-row detail", "",
             "| graph | app | l3 | policy | len | mech hits | xref | excerpt |",
             "|---|---|---|---|---|---|---|---|"]
    for r in a["rows"]:
        lines.append(
            f"| {r['graph']} | {r['app']} | {r['l3_size']} | {r['policy']} "
            f"| {r['reason_len']} | {r['mechanism_hits']} "
            f"| {'Y' if r['cross_ref'] else 'N'} | {r['reason_excerpt']} |"
        )
    if a["violations"]:
        lines += ["", f"## Violations ({len(a['violations'])})", "",
                  "| row_id | rule | observed | floor/ceiling |",
                  "|---|---|---|---|"]
        for v in a["violations"]:
            lines.append(
                f"| {v.get('row_id')} | {v['rule']} | "
                f"{v.get('observed', v.get('reason_excerpt',''))} | "
                f"{v.get('floor') or v.get('ceiling') or ''} |"
            )
    else:
        lines += ["", "_No violations._"]
    return "\n".join(lines) + "\n"


def _to_csv(a: dict[str, Any], path: Path) -> None:
    fields = ["graph", "app", "l3_size", "policy", "citation",
              "reason_len", "mechanism_hits", "cross_ref",
              "reason_excerpt"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in a["rows"]:
            w.writerow({k: r.get(k) for k in fields})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lit-faith-json", required=True, type=Path)
    ap.add_argument("--json-out",       required=True, type=Path)
    ap.add_argument("--md-out",         required=True, type=Path)
    ap.add_argument("--csv-out",        required=True, type=Path)
    args = ap.parse_args()

    payload = json.loads(args.lit_faith_json.read_text(encoding="utf-8"))
    a = audit(payload["per_claim"])

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(a, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    args.md_out.write_text(_to_markdown(a), encoding="utf-8")
    _to_csv(a, args.csv_out)

    s = a["summary"]
    print(f"[lit-faith-devexp] {s['known_deviation_rows']} "
          f"known-deviation rows; violations={s['violations']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
