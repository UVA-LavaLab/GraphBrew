#!/usr/bin/env python3
"""Gate 242 — paper-label-map integrity audit.

Verifies the canonical policy_label → paper-figure-label mapping in
``scripts/experiments/ecg/paper_pipeline.py`` stays consistent with:

  G1 — every POLICY_LABELS key has a POLICY_DESCRIPTIONS and
       POLICY_COLORS entry (no partial additions).
  G2 — every figure_label is unique across the map (no two internal
       policy_labels map to the same paper-figure label).
  G3 — every policy_label that appears in any tracked source artifact
       (lit-faith per_observation, gate-238/239/240/241 postfixes,
       winner_table, etc.) is in POLICY_LABELS.
  G4 — the latest committed paper_pipeline_*/policy_label_map.csv
       matches the code's POLICY_LABELS + POLICY_DESCRIPTIONS exactly
       (rebuild from code, byte-compare against committed CSV).
  G5 — every POLICY_LABELS figure_label appears as policy_short
       somewhere in the tracked source artifacts (no orphan paper
       labels — every paper label corresponds to a real policy in
       a real audit).

Unlike the substrate-parity gates (238/239/240) and the prefetcher
head-to-head gate (241), this gate has no scaffold-deferred mode:
the source-of-truth is the code itself, not a hand-curated data
fixture, so it is always active.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import importlib.util


# ------------------------------------------------------------------ paths --

ROOT = Path(__file__).resolve().parents[2].parent
WIKI_DATA = ROOT / "wiki" / "data"
PAPER_PIPELINE = ROOT / "scripts" / "experiments" / "ecg" / "paper_pipeline.py"


# Tracked source artifacts to scan for policy_label occurrences.
# Every entry must be a JSON file in wiki/data. Each value is a JSON
# path expression understood by ``_collect_labels``: a list of keys
# describing how to walk into the document looking for policy_label
# fields.
TRACKED_SOURCES: dict[str, list[list[str]]] = {
    "ecg_substrate_parity_postfix.json":      [["per_observation", "*", "policy_label"]],
    "ecg_gem5_parity_postfix.json":           [["per_observation", "*", "policy_label"]],
    "ecg_sniper_parity_postfix.json":         [["per_observation", "*", "policy_label"]],
    "ecg_pfx_vs_droplet_postfix.json":        [["per_observation", "*", "policy_label"]],
    "lit_faith_ecg_parity.json":              [["per_observation", "*", "policy_label"]],
    "lit_faith_ecg_gem5_parity.json":         [["per_observation", "*", "policy_label"]],
    "literature_faithfulness_postfix.json":   [["per_observation", "*", "policy"],
                                                ["per_claim", "*", "policy"]],
    "policy_winner_table.json":               [["cells", "*", "winner_policy"],
                                                ["cells", "*", "runner_up_policy"]],
}


# Latest paper_pipeline_*/ CSVs are also authoritative sources for
# what labels actually appear in paper figures — scanned in addition
# to the JSON sources above.
TRACKED_PAPER_PIPELINE_CSVS: list[str] = [
    "roi_matrix_all.csv",
    "roi_policy_summary.csv",
    "ecg_mode_overhead_summary.csv",
    "popt_storage_overhead_summary.csv",
    "popt_charged_overhead.csv",
]


# Theorem-class virtual policy labels — they appear in the per_claim
# table as theorem-statements (e.g. "POPT >= GRASP under conditions
# X") and are NOT replacement policies. Per gate 233 (LIT-RatGrid),
# they exist for rationale rows.
THEOREM_CLASS_LABELS = {
    "POPT_GE_GRASP",
    "POPT_NEAR_GRASP_IF_BIG_GAP",
}

# Runtime arm names that are not replacement policies and therefore
# legitimately absent from POLICY_LABELS:
NON_PAPER_LABELS = {
    # cache-level rollups, not a replacement policy
    "CACHE",
    # gate 241 prefetcher arm names, not replacement policies
    "DROPLET", "ECG_PFX",
} | THEOREM_CLASS_LABELS


# ------------------------------------------------------------------ helpers --

def _load_paper_pipeline_constants() -> dict:
    """Import paper_pipeline.py and pull POLICY_LABELS/DESCRIPTIONS/COLORS."""
    spec = importlib.util.spec_from_file_location("paper_pipeline_dyn",
                                                  PAPER_PIPELINE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return {
        "POLICY_LABELS":       dict(getattr(mod, "POLICY_LABELS", {})),
        "POLICY_DESCRIPTIONS": dict(getattr(mod, "POLICY_DESCRIPTIONS", {})),
        "POLICY_COLORS":       dict(getattr(mod, "POLICY_COLORS", {})),
    }


def _walk_json(obj, path: list[str]):
    """Yield every leaf value reached by ``path`` ('*' is wildcard)."""
    if not path:
        yield obj
        return
    head, *rest = path
    if isinstance(obj, dict):
        if head == "*":
            for v in obj.values():
                yield from _walk_json(v, rest)
        elif head in obj:
            yield from _walk_json(obj[head], rest)
    elif isinstance(obj, list):
        if head == "*":
            for v in obj:
                yield from _walk_json(v, rest)


def _collect_labels(constants: dict) -> tuple[dict[str, list[str]], list[dict]]:
    """Walk every TRACKED_SOURCES file; return (label -> sources, audit_rows)."""
    found: dict[str, set[str]] = {}
    audit_rows: list[dict] = []
    for name, jpaths in sorted(TRACKED_SOURCES.items()):
        p = WIKI_DATA / name
        if not p.exists():
            audit_rows.append({"source": name, "status": "missing",
                               "label_count": 0, "labels": []})
            continue
        try:
            doc = json.loads(p.read_text())
        except json.JSONDecodeError:
            audit_rows.append({"source": name, "status": "unreadable",
                               "label_count": 0, "labels": []})
            continue
        seen_in_file: set[str] = set()
        for jp in jpaths:
            for v in _walk_json(doc, jp):
                if isinstance(v, str) and v:
                    seen_in_file.add(v)
        for lab in seen_in_file:
            found.setdefault(lab, set()).add(name)
        audit_rows.append({
            "source":      name,
            "status":      "ok",
            "label_count": len(seen_in_file),
            "labels":      sorted(seen_in_file),
        })
    # Also scan latest paper_pipeline_*/CSVs for policy_label / policy_short
    # columns. These are the actual paper-figure data files; the JSON
    # sources above cover lit-faith/audit corpora.
    dirp = _find_latest_paper_pipeline_dir()
    if dirp is not None:
        for csv_name in TRACKED_PAPER_PIPELINE_CSVS:
            cp = dirp / csv_name
            if not cp.exists():
                audit_rows.append({"source": f"{dirp.name}/{csv_name}",
                                   "status": "missing",
                                   "label_count": 0, "labels": []})
                continue
            seen_in_file: set[str] = set()
            with cp.open() as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    # paper_pipeline CSVs carry both a "policy_label"
                    # (full canonical name) and a short-form "policy"
                    # column that's a family rollup ("ECG" for all
                    # ECG_* variants). Only the full label is what
                    # POLICY_LABELS keys on, so scan policy_label.
                    v = row.get("policy_label")
                    if v:
                        seen_in_file.add(v)
            for lab in seen_in_file:
                found.setdefault(lab, set()).add(f"{dirp.name}/{csv_name}")
            audit_rows.append({
                "source":      f"{dirp.name}/{csv_name}",
                "status":      "ok",
                "label_count": len(seen_in_file),
                "labels":      sorted(seen_in_file),
            })
    return ({k: sorted(v) for k, v in found.items()}, audit_rows)


def _find_latest_paper_pipeline_dir() -> Path | None:
    candidates = sorted(WIKI_DATA.glob("paper_pipeline_*"))
    return candidates[-1] if candidates else None


def _rebuild_label_map_csv(POLICY_LABELS: dict[str, str],
                           POLICY_DESCRIPTIONS: dict[str, str]) -> str:
    """Rebuild the canonical policy_label_map.csv byte-stream from code."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["description", "figure_label", "policy_label"])
    for k, fig in POLICY_LABELS.items():
        w.writerow([POLICY_DESCRIPTIONS.get(k, ""), fig, k])
    return buf.getvalue()


# ------------------------------------------------------------------ rules --

def _rule_g1(constants: dict) -> list[dict]:
    out = []
    L = constants["POLICY_LABELS"]
    D = constants["POLICY_DESCRIPTIONS"]
    C = constants["POLICY_COLORS"]
    for k in sorted(L.keys()):
        if k not in D:
            out.append({"rule": "G1", "policy_label": k,
                        "missing": "description"})
        if k not in C:
            out.append({"rule": "G1", "policy_label": k,
                        "missing": "color"})
    return out


def _rule_g2(constants: dict) -> list[dict]:
    L = constants["POLICY_LABELS"]
    rev: dict[str, list[str]] = {}
    for k, v in L.items():
        rev.setdefault(v, []).append(k)
    return [
        {"rule": "G2", "figure_label": v, "internal_keys": sorted(ks)}
        for v, ks in sorted(rev.items())
        if len(ks) > 1
    ]


def _rule_g3(constants: dict, found_labels: dict[str, list[str]]) -> list[dict]:
    """Every policy_label in tracked sources must be in POLICY_LABELS,
    or in the NON_PAPER_LABELS allowlist (theorem-class virtual labels
    + cache-level rollups + prefetcher arm names from gate 241)."""
    L = set(constants["POLICY_LABELS"].keys())
    out = []
    for lab in sorted(found_labels.keys()):
        if lab in L or lab in NON_PAPER_LABELS:
            continue
        out.append({"rule": "G3", "policy_label": lab,
                    "found_in": found_labels[lab]})
    return out


def _rule_g4(constants: dict) -> list[dict]:
    """Latest committed policy_label_map.csv must match code byte-for-byte."""
    dirp = _find_latest_paper_pipeline_dir()
    if dirp is None:
        return [{"rule": "G4", "csv_path": "<no paper_pipeline_* dir>"}]
    csv_path = dirp / "policy_label_map.csv"
    if not csv_path.exists():
        return [{"rule": "G4", "csv_path": str(csv_path.relative_to(ROOT)),
                 "issue": "missing"}]
    rebuilt = _rebuild_label_map_csv(constants["POLICY_LABELS"],
                                     constants["POLICY_DESCRIPTIONS"])
    committed = csv_path.read_text()
    if rebuilt != committed:
        return [{"rule": "G4",
                 "csv_path":            str(csv_path.relative_to(ROOT)),
                 "rebuilt_byte_count":  len(rebuilt),
                 "committed_byte_count": len(committed),
                 "issue": "code/committed mismatch — regenerate paper_pipeline"}]
    return []


def _rule_g5(constants: dict, found_labels: dict[str, list[str]]) -> list[dict]:
    """Every POLICY_LABELS key must appear in at least one tracked source."""
    out = []
    for k in sorted(constants["POLICY_LABELS"].keys()):
        if k not in found_labels:
            out.append({"rule": "G5", "policy_label": k,
                        "issue": "orphan — not used in any tracked source"})
    return out


# ------------------------------------------------------------------ audit --

def audit() -> dict:
    constants = _load_paper_pipeline_constants()
    found_labels, source_rows = _collect_labels(constants)
    violations = []
    violations.extend(_rule_g1(constants))
    violations.extend(_rule_g2(constants))
    violations.extend(_rule_g3(constants, found_labels))
    violations.extend(_rule_g4(constants))
    violations.extend(_rule_g5(constants, found_labels))
    return {
        "status":   "active",
        "rules": {
            "G1": "every POLICY_LABELS key has matching description and color",
            "G2": "every figure_label is unique across the map",
            "G3": "every policy_label in tracked sources is in POLICY_LABELS (modulo allowlist)",
            "G4": "latest committed policy_label_map.csv matches code byte-for-byte",
            "G5": "every POLICY_LABELS key appears in at least one tracked source",
        },
        "constants": {
            "policy_labels_count":       len(constants["POLICY_LABELS"]),
            "policy_descriptions_count": len(constants["POLICY_DESCRIPTIONS"]),
            "policy_colors_count":       len(constants["POLICY_COLORS"]),
            "tracked_source_count":      len(TRACKED_SOURCES) + len(TRACKED_PAPER_PIPELINE_CSVS),
            "non_paper_label_allowlist": sorted(NON_PAPER_LABELS),
            "theorem_class_labels":      sorted(THEOREM_CLASS_LABELS),
        },
        "totals": {
            "labels_in_code":   len(constants["POLICY_LABELS"]),
            "sources_scanned":  len(source_rows),
            "sources_ok":       sum(1 for r in source_rows if r["status"] == "ok"),
            "sources_missing":  sum(1 for r in source_rows if r["status"] == "missing"),
            "labels_found":     len(found_labels),
            "violations":       len(violations),
        },
        "policy_label_map": [
            {"policy_label": k,
             "figure_label": constants["POLICY_LABELS"][k],
             "description":  constants["POLICY_DESCRIPTIONS"].get(k, ""),
             "color":        constants["POLICY_COLORS"].get(k, "")}
            for k in sorted(constants["POLICY_LABELS"].keys())
        ],
        "source_scan": source_rows,
        "violations":  violations,
    }


# ------------------------------------------------------------------ writers --

def _render_md(audit: dict) -> str:
    lines = []
    lines.append("# Paper label-map integrity (gate 242)")
    lines.append("")
    t = audit["totals"]
    lines.append(f"**Status:** {audit['status']}  •  "
                 f"labels: {t['labels_in_code']}  •  "
                 f"sources scanned: {t['sources_scanned']} "
                 f"(ok {t['sources_ok']}, missing {t['sources_missing']})  •  "
                 f"violations: {t['violations']}")
    lines.append("")
    lines.append("## Rules")
    for k, v in audit["rules"].items():
        lines.append(f"- **{k}** — {v}")
    lines.append("")
    lines.append("## Canonical policy → figure-label map")
    lines.append("")
    lines.append("| policy_label | figure_label | description |")
    lines.append("|---|---|---|")
    for row in audit["policy_label_map"]:
        lines.append(f"| {row['policy_label']} | {row['figure_label']} | "
                     f"{row['description']} |")
    lines.append("")
    lines.append("## Source-artifact scan")
    lines.append("")
    lines.append("| source | status | label_count | labels |")
    lines.append("|---|---|---:|---|")
    for r in audit["source_scan"]:
        labels = ", ".join(r["labels"]) if r["labels"] else "—"
        lines.append(f"| {r['source']} | {r['status']} | "
                     f"{r['label_count']} | {labels} |")
    lines.append("")
    if audit["violations"]:
        lines.append("## Violations")
        for v in audit["violations"]:
            lines.append(f"- {v}")
    else:
        lines.append("**0 violations** — paper label map is consistent.")
    return "\n".join(lines) + "\n"


def _render_csv(audit: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["policy_label", "figure_label", "description", "color"])
    for row in audit["policy_label_map"]:
        w.writerow([row["policy_label"], row["figure_label"],
                    row["description"], row["color"]])
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
    t = a["totals"]
    print(f"[lit-faith-paper-label-map] status={a['status']} "
          f"labels={t['labels_in_code']} "
          f"sources_ok={t['sources_ok']}/{t['sources_scanned']} "
          f"violations={t['violations']}")


if __name__ == "__main__":
    main()
