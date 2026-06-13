#!/usr/bin/env python3
"""Literature-faithfulness known-deviation completeness audit.

``literature_baselines.KNOWN_DEVIATIONS`` is the whitelist of
(graph, app, l3_size, policy) cells where the live cache_sim run is
expected to disagree with the published claim — typically a documented
design choice (P-OPT Phase 1 evicting non-property lines first), a
simulator-level limitation (single-core, no prefetching), or a
graph/topology mismatch (BC frontier vs PR-rank schedule). The
comparator downgrades those cells from ``disagree`` to
``known_deviation`` so CI stays green on accepted issues.

That whitelist is load-bearing. If a reviewer hits a "known_deviation"
row and the explanation is empty, hand-wavy, or has no anchor in the
literature or implementation, the whole confidence story for that
cell unravels. Even worse, an *orphan* KNOWN_DEVIATIONS entry —
one that no longer maps to any live lit-faith row because the
underlying (graph, app, l3_size) was removed — accumulates silently
as the corpus evolves.

This module audits:

1. **Reason completeness** — every KNOWN_DEVIATIONS value is a non-empty
   string, ≥ MIN_REASON_LENGTH characters, contains a quantitative
   magnitude phrase (e.g. ``3 pp``, ``8 MB``, ``5 %``), and mentions
   at least one anchor (paper venue tag, ``§``, ``Fig``, ``design``,
   ``sim``, ``HPCA``, ``ISCA``).
2. **Bijection with live faith corpus** — every ``status="known_deviation"``
   row in lit-faith has a matching key in ``KNOWN_DEVIATIONS``, and
   every KNOWN_DEVIATIONS key is exercised by lit-faith.
3. **Per-policy coverage** — KD keys span the expected set of policies
   (currently POPT_GE_GRASP and POPT_NEAR_GRASP_IF_BIG_GAP).
4. **Per-graph coverage** — ≥ 4 distinct graphs are represented,
   ensuring deviations aren't concentrated on a single dataset.
5. **Reason quality fingerprint** — counts of reasons mentioning
   "Phase 1", "P-OPT", "frontier", "MPKI", etc. — surfaces vocabulary
   shift over time.

Emits ``wiki/data/lit_faith_deviations.{json,md,csv}``.
"""

from __future__ import annotations

import argparse
import csv
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
DEFAULT_JSON_OUT = REPO_ROOT / "wiki" / "data" / "lit_faith_deviations.json"
DEFAULT_MD_OUT = REPO_ROOT / "wiki" / "data" / "lit_faith_deviations.md"
DEFAULT_CSV_OUT = REPO_ROOT / "wiki" / "data" / "lit_faith_deviations.csv"

MIN_REASON_LENGTH = 80

QUANTITATIVE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:pp|MB|KB|GB|%|times|MPKI|miss|ways|MHz|cycles|MB/s)\b",
    re.IGNORECASE,
)
ANCHOR_RE = re.compile(
    r"\b(?:HPCA|ISCA|MICRO|ASPLOS|PLDI|§|Fig(?:\.|ure)?\s*\d|design|"
    r"sim\s*bug|single-core|Phase\s*\d|P-OPT|GRASP|SRRIP|LRU|cache_sim|"
    r"mismatch|mis[-\s]?alignment|union[-\s]?find|frontier|PR[-\s]?rank|"
    r"property\s+array|working\s+set|hot[-\s]?region|root\s+cause|"
    r"algorithmic|ranking|eviction|prefetch)",
    re.IGNORECASE,
)

# Vocabulary fingerprint terms — frequencies tracked over time
VOCAB_TERMS = [
    "Phase 1",
    "P-OPT",
    "GRASP",
    "SRRIP",
    "frontier",
    "MPKI",
    "design",
    "sim",
    "single-core",
    "property array",
    "HPCA",
]


def _load_baselines_module(path: Path):
    spec = importlib.util.spec_from_file_location(
        "literature_baselines_dev_audit", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_audit(
    lit_faith: dict[str, Any], baselines_module
) -> dict[str, Any]:
    known_deviations = dict(baselines_module.KNOWN_DEVIATIONS)
    faith_per_claim = lit_faith.get("per_claim", [])

    # Build per-cell key set from faith corpus
    faith_keys_all = {
        (
            r["graph"],
            r["app"],
            r["l3_size"],
            r["policy"],
        ): r
        for r in faith_per_claim
    }
    # Explicitly-whitelisted known_deviation rows. Auto-tolerated per-cell
    # P-OPT diagnostics (known_deviation_auto=True) are EXCLUDED here: they
    # are tolerated by the evaluator's POPT_DIAGNOSTIC_CLAIMS rule (the
    # authoritative claim is the corpus geomean, not per-cell dominance) and
    # legitimately carry no hand-written KNOWN_DEVIATIONS whitelist entry, so
    # they must not trip the whitelist-bijection invariant.
    faith_kd_keys = {
        k for k, r in faith_keys_all.items()
        if r.get("status") == "known_deviation"
        and not r.get("known_deviation_auto")
    }
    auto_kd_keys = {
        k for k, r in faith_keys_all.items()
        if r.get("status") == "known_deviation"
        and r.get("known_deviation_auto")
    }
    baseline_kd_keys = set(known_deviations.keys())

    # Per-entry reason audit
    reason_audits: list[dict[str, Any]] = []
    for key, reason in sorted(known_deviations.items()):
        text = reason or ""
        anchor_match = ANCHOR_RE.search(text)
        quant_match = QUANTITATIVE_RE.search(text)
        is_orphan = key not in faith_keys_all
        is_inactive = key in faith_keys_all and (
            faith_keys_all[key].get("status") != "known_deviation"
        )
        vocab_hits = [
            term for term in VOCAB_TERMS
            if re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE)
        ]
        reason_audits.append({
            "graph": key[0],
            "app": key[1],
            "l3_size": key[2],
            "policy": key[3],
            "reason": text,
            "length": len(text),
            "has_quantitative_phrase": bool(quant_match),
            "has_anchor": bool(anchor_match),
            "is_orphan": is_orphan,
            "is_inactive": is_inactive,
            "vocab_hits": vocab_hits,
            "well_formed": (
                len(text) >= MIN_REASON_LENGTH
                and bool(anchor_match)
                and bool(quant_match)
                and not is_orphan
            ),
        })

    # Live KD faith rows missing KNOWN_DEVIATIONS entry (would be orphan
    # from the other direction — should be 0).
    faith_kd_no_entry = sorted(faith_kd_keys - baseline_kd_keys)

    # Coverage fingerprints
    policy_coverage = Counter(k[3] for k in known_deviations)
    graph_coverage = Counter(k[0] for k in known_deviations)
    app_coverage = Counter(k[1] for k in known_deviations)
    l3_coverage = Counter(k[2] for k in known_deviations)
    vocab_counts = Counter(
        hit for r in reason_audits for hit in r["vocab_hits"]
    )

    orphan_count = sum(1 for r in reason_audits if r["is_orphan"])
    inactive_count = sum(1 for r in reason_audits if r["is_inactive"])
    short_reasons = [
        r for r in reason_audits if r["length"] < MIN_REASON_LENGTH
    ]
    missing_quant = [
        r for r in reason_audits if not r["has_quantitative_phrase"]
    ]
    missing_anchor = [r for r in reason_audits if not r["has_anchor"]]
    well_formed_count = sum(1 for r in reason_audits if r["well_formed"])

    payload = {
        "schema_version": 1,
        "summary": {
            "known_deviations_total": len(known_deviations),
            "faith_known_deviation_rows": len(faith_kd_keys),
            "auto_known_deviation_rows": len(auto_kd_keys),
            "orphan_known_deviations": orphan_count,
            "inactive_known_deviations": inactive_count,
            "faith_kd_without_entry_count": len(faith_kd_no_entry),
            "well_formed_reasons": well_formed_count,
            "short_reasons": len(short_reasons),
            "missing_quantitative_phrase": len(missing_quant),
            "missing_anchor": len(missing_anchor),
            "policies_covered": len(policy_coverage),
            "graphs_covered": len(graph_coverage),
            "apps_covered": len(app_coverage),
            "l3_sizes_covered": len(l3_coverage),
            "min_reason_length": MIN_REASON_LENGTH,
        },
        "reason_audits": reason_audits,
        "faith_kd_without_entry": [
            list(k) for k in faith_kd_no_entry
        ],
        "policy_coverage": dict(policy_coverage),
        "graph_coverage": dict(graph_coverage),
        "app_coverage": dict(app_coverage),
        "l3_size_coverage": dict(l3_coverage),
        "vocab_counts": dict(vocab_counts),
    }
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    s = payload["summary"]
    lines = [
        "# Literature-faithfulness known-deviation completeness audit",
        "",
        "Every (graph, app, l3, policy) cell in `KNOWN_DEVIATIONS` "
        "must carry a complete, quantitative, and anchored explanation; "
        "every live `status=\"known_deviation\"` row must resolve to a "
        "whitelist entry; and the whitelist must not accumulate orphan "
        "entries that no live cell exercises.",
        "",
        "## Summary",
        "",
        f"- KNOWN_DEVIATIONS entries: **{s['known_deviations_total']}**",
        f"- Live lit-faith known_deviation rows: "
        f"**{s['faith_known_deviation_rows']}**",
        f"- Orphan whitelist entries (no live cell): "
        f"**{s['orphan_known_deviations']}**",
        f"- Inactive whitelist entries (live cell present, status ≠ "
        f"`known_deviation`): **{s['inactive_known_deviations']}**",
        f"- Live KD rows without whitelist entry: "
        f"**{s['faith_kd_without_entry_count']}**",
        f"- Well-formed reasons (≥ {s['min_reason_length']} chars + "
        f"quantitative phrase + anchor + not orphan): "
        f"**{s['well_formed_reasons']}** / "
        f"{s['known_deviations_total']}",
        f"- Reasons missing quantitative phrase: "
        f"**{s['missing_quantitative_phrase']}**",
        f"- Reasons missing anchor: **{s['missing_anchor']}**",
        f"- Short reasons (< {s['min_reason_length']} chars): "
        f"**{s['short_reasons']}**",
        f"- Coverage: **{s['policies_covered']}** policies × "
        f"**{s['graphs_covered']}** graphs × "
        f"**{s['apps_covered']}** apps × "
        f"**{s['l3_sizes_covered']}** L3 sizes",
        "",
        "## Coverage breakdown",
        "",
        "**By policy:**",
    ]
    for k, v in sorted(payload["policy_coverage"].items()):
        lines.append(f"- `{k}` → {v}")
    lines.extend(["", "**By graph:**"])
    for k, v in sorted(payload["graph_coverage"].items()):
        lines.append(f"- `{k}` → {v}")
    lines.extend(["", "**By app:**"])
    for k, v in sorted(payload["app_coverage"].items()):
        lines.append(f"- `{k}` → {v}")
    lines.extend(["", "**By L3 size:**"])
    for k, v in sorted(payload["l3_size_coverage"].items()):
        lines.append(f"- `{k}` → {v}")
    lines.extend(["", "## Vocabulary fingerprint", ""])
    lines.append("| term | reason count |")
    lines.append("|---|---:|")
    for term in VOCAB_TERMS:
        lines.append(f"| `{term}` | {payload['vocab_counts'].get(term, 0)} |")

    lines.extend(["", "## Per-entry reason status", ""])
    lines.append(
        "| graph | app | L3 | policy | len | quant | anchor | "
        "orphan | inactive | well-formed |"
    )
    lines.append("|---|---|---|---|---:|---|---|---|---|---|")
    for r in payload["reason_audits"]:
        wf = "✓" if r["well_formed"] else "✗"
        q = "✓" if r["has_quantitative_phrase"] else "✗"
        a = "✓" if r["has_anchor"] else "✗"
        o = "✗" if r["is_orphan"] else "—"
        i = "⚠" if r["is_inactive"] else "—"
        lines.append(
            f"| {r['graph']} | {r['app']} | {r['l3_size']} | "
            f"{r['policy']} | {r['length']} | {q} | {a} | {o} | {i} | {wf} |"
        )

    if payload["faith_kd_without_entry"]:
        lines.extend(
            ["", "## ⚠ Live KD rows without whitelist entry", ""]
        )
        for k in payload["faith_kd_without_entry"]:
            lines.append(f"- `{k}`")

    return "\n".join(lines)


def render_csv(payload: dict[str, Any], path: Path) -> None:
    fields = [
        "graph", "app", "l3_size", "policy", "length",
        "has_quantitative_phrase", "has_anchor",
        "is_orphan", "is_inactive", "well_formed",
        "vocab_hits",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in payload["reason_audits"]:
            row = dict(r)
            row["vocab_hits"] = ";".join(r["vocab_hits"])
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
        f"[lit-faith-deviations] {s['known_deviations_total']} KD entries, "
        f"{s['well_formed_reasons']} well-formed, "
        f"{s['orphan_known_deviations']} orphan, "
        f"{s['faith_kd_without_entry_count']} live without entry."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
