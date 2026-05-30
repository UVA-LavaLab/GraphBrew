#!/usr/bin/env python3
"""Gate 245 — L3 regime-classifier consistency.

Across `scripts/experiments/ecg/` there are several functions that
classify an L3 cache size (or L3/WSS ratio, or L3 size range) into a
"regime" label that downstream reports use to bucket policies. Today
those classifiers are written ad hoc — at least two of them
(`policy_winner_table._l3_regime`, `oracle_gap_report._regime`) share
the *vocabulary* {tiny, small, large, unknown} but use *different
boundaries*, and a third (`popt_vs_grasp_report._l3_regime`) is a
copy of the first.

That is a subtle, real footgun: an unaware author can tweak one
boundary, no test fails, and the paper's per-regime bar groupings
silently diverge between two side-by-side figures.

This gate codifies the situation with a hand-curated registry:

  * every regime classifier we know about is listed in
    ``REGIME_REGISTRY`` with its module path, function name, declared
    *taxonomy family*, declared *vocabulary*, and a free-form note
    (e.g. "diverges from v1 at 32 kB and 64 kB — see ...");
  * within each family, every member must agree on every label in
    ``CANONICAL_L3_GRID`` (so duplicating a classifier is fine, but
    drifting one of the copies is not);
  * every classifier must return only labels from its declared
    vocabulary (no silent new labels);
  * every classifier file must still exist and resolve to a callable.

The gate is intentionally tolerant of *honest* divergence: if two
classifiers should be different (`oracle_gap_report._regime` really
does use a different small/large split than the v1 paper table),
they live in *different families* and the gate does not force them
to agree — it just forces a maintainer to *declare* the divergence,
giving reviewers a single place to audit it.

Rules:

  R1 — every entry in REGIME_REGISTRY resolves to an importable
       module + callable function;
  R2 — every "byte-input" classifier (signature == "byte_label")
       returns only labels from its declared vocabulary when fed
       the canonical L3 grid;
  R3 — within each family, all byte-input members agree on every
       label in CANONICAL_L3_GRID;
  R4 — every byte-input regime classifier function we discover in
       ``scripts/experiments/ecg/*.py`` by source-pattern match
       (function name starts with ``_l3_regime`` / ``_regime`` /
       ``_classify_regime``) is registered (defensive — catches new
       drift-prone classifiers);
  R5 — every classifier with a non-default signature (ratio-input,
       range-input) has its ``signature`` field populated AND a
       ``note`` describing what it actually classifies.

Source-of-truth: each classifier in its own module, loaded via
importlib. No paper_pipeline.py dependency.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2].parent
ECG_DIR = ROOT / "scripts" / "experiments" / "ecg"
WIKI_DATA = ROOT / "wiki" / "data"


# ----------------------------------------------------------- registry --

# Each entry declares one regime classifier we know about. Within a
# taxonomy *family*, all members must agree on every label in the
# canonical L3 grid (R3). Cross-family disagreement is allowed but
# must be declared with a ``note`` (R5).
REGIME_REGISTRY: list[dict] = [
    {
        "path":       "scripts/experiments/ecg/policy_winner_table.py",
        "func":       "_l3_regime",
        "family":     "tiny_small_large_v1",
        "vocabulary": ["unknown", "tiny", "small", "large"],
        "signature":  "byte_label",
        "note":       "Paper winner-table boundaries: <64 kB tiny; "
                      "[64 kB, 1 MB) small; >=1 MB large.",
    },
    {
        "path":       "scripts/experiments/ecg/popt_vs_grasp_report.py",
        "func":       "_l3_regime",
        "family":     "tiny_small_large_v1",
        "vocabulary": ["unknown", "tiny", "small", "large"],
        "signature":  "byte_label",
        "note":       "Sibling of policy_winner_table._l3_regime; "
                      "kept identical so POPT-vs-GRASP plots align "
                      "with winner-table groupings.",
    },
    {
        "path":       "scripts/experiments/ecg/oracle_gap_report.py",
        "func":       "_regime",
        "family":     "tiny_small_large_v2_oracle_gap",
        "vocabulary": ["unknown", "tiny", "small", "large"],
        "signature":  "byte_label",
        "note":       "Diverges from v1 at 32-64 kB AND uses 256 kB "
                      "(not 1 MB) as the small/large boundary. "
                      "Intentionally separate family to avoid "
                      "silently re-bucketing the oracle-gap figures.",
    },
    {
        "path":       "scripts/experiments/ecg/cross_tool_lru_regime.py",
        "func":       "_classify_regime",
        "family":     "wss_range",
        "vocabulary": ["sub-WSS", "post-WSS", "mixed"],
        "signature":  "kb_range",
        "note":       "Classifies an L3-size RANGE (lo_kb, hi_kb), "
                      "not a single L3 size. sub-WSS = hi_kb<=4096; "
                      "post-WSS = lo_kb>=1024; mixed otherwise.",
    },
    {
        "path":       "scripts/experiments/ecg/wss_relative_l3.py",
        "func":       "_wss_regime",
        "family":     "wss_ratio",
        "vocabulary": ["under_wss", "near_wss", "over_wss"],
        "signature":  "ratio",
        "note":       "Classifies the L3/WSS RATIO (not a size in "
                      "bytes). under_wss = ratio<0.25; over_wss = "
                      "ratio>4.0; near_wss otherwise.",
    },
]

# Canonical L3-size labels covering the cell census the paper uses,
# extended at both ends so future label additions don't silently slip
# through. Each label must parse identically across all byte-input
# classifiers in the same family.
CANONICAL_L3_GRID = [
    "1kB", "4kB", "16kB", "32kB", "64kB", "128kB", "256kB",
    "512kB", "1MB", "2MB", "4MB", "8MB", "16MB",
]

# Function-name patterns we accept as "this is a regime classifier".
# R4 enforces that every match found by these patterns inside
# scripts/experiments/ecg/ is registered.
REGIME_FUNC_NAME_RE = re.compile(
    r"^\s*def\s+(_l3_regime|_regime|_classify_regime|_wss_regime)\s*\(",
    re.MULTILINE,
)


# ----------------------------------------------------------- helpers --

def _load_module(rel_path: str, name: str):
    p = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _classify_byte_label(entry: dict, label: str) -> str:
    mod = _load_module(entry["path"], f"regime_dyn_{entry['family']}_{entry['func']}")
    fn = getattr(mod, entry["func"])
    return fn(label)


# ----------------------------------------------------------- rules --

def _rule_r1(entries: list[dict]) -> list[dict]:
    out: list[dict] = []
    for e in entries:
        p = ROOT / e["path"]
        if not p.exists():
            out.append({"rule": "R1", "path": e["path"], "func": e["func"],
                        "issue": "module file does not exist"})
            continue
        try:
            mod = _load_module(e["path"], f"r1_{e['func']}")
        except Exception as exc:
            out.append({"rule": "R1", "path": e["path"], "func": e["func"],
                        "issue": f"import failed: {exc!r}"})
            continue
        if not callable(getattr(mod, e["func"], None)):
            out.append({"rule": "R1", "path": e["path"], "func": e["func"],
                        "issue": "function not found or not callable"})
    return out


def _rule_r2(entries: list[dict]) -> list[dict]:
    out: list[dict] = []
    for e in entries:
        if e.get("signature") != "byte_label":
            continue
        vocab = set(e["vocabulary"])
        for label in CANONICAL_L3_GRID:
            try:
                got = _classify_byte_label(e, label)
            except Exception as exc:
                out.append({"rule": "R2", "func": e["func"],
                            "label": label, "issue": f"raised: {exc!r}"})
                continue
            if got not in vocab:
                out.append({"rule": "R2", "func": e["func"],
                            "family": e["family"], "label": label,
                            "got": got, "vocabulary": sorted(vocab),
                            "issue": "label outside declared vocabulary"})
    return out


def _rule_r3(entries: list[dict]) -> list[dict]:
    out: list[dict] = []
    by_family: dict[str, list[dict]] = {}
    for e in entries:
        if e.get("signature") != "byte_label":
            continue
        by_family.setdefault(e["family"], []).append(e)
    for fam, members in by_family.items():
        if len(members) < 2:
            continue
        for label in CANONICAL_L3_GRID:
            labels = {}
            for e in members:
                try:
                    labels[e["func"] + "@" + e["path"]] = \
                        _classify_byte_label(e, label)
                except Exception as exc:
                    labels[e["func"] + "@" + e["path"]] = f"<error:{exc!r}>"
            distinct = set(labels.values())
            if len(distinct) > 1:
                out.append({"rule": "R3", "family": fam,
                            "label": label, "per_member": labels,
                            "issue": "in-family disagreement"})
    return out


def _rule_r4(entries: list[dict]) -> list[dict]:
    """Defensive: scan ECG dir for regime-classifier function definitions
    that match our naming convention and assert each one is registered."""
    out: list[dict] = []
    registered = {(e["path"], e["func"]) for e in entries}
    for py in sorted(ECG_DIR.glob("*.py")):
        try:
            txt = py.read_text()
        except Exception:
            continue
        rel = "scripts/experiments/ecg/" + py.name
        for m in REGIME_FUNC_NAME_RE.finditer(txt):
            fname = m.group(1)
            if (rel, fname) not in registered:
                out.append({"rule":  "R4",
                            "path":  rel,
                            "func":  fname,
                            "issue": "regime-classifier function found "
                                     "by source-pattern scan but not "
                                     "registered in REGIME_REGISTRY"})
    return out


def _rule_r5(entries: list[dict]) -> list[dict]:
    out: list[dict] = []
    for e in entries:
        sig = e.get("signature")
        if sig != "byte_label" and not e.get("note"):
            out.append({"rule": "R5", "func": e["func"],
                        "family": e["family"], "signature": sig,
                        "issue": "non-byte-label signature missing "
                                 "explanatory note"})
        if not sig:
            out.append({"rule": "R5", "func": e["func"],
                        "family": e["family"],
                        "issue": "signature field missing"})
    return out


# ----------------------------------------------------------- audit --

def audit() -> dict:
    violations: list[dict] = []
    violations.extend(_rule_r1(REGIME_REGISTRY))
    violations.extend(_rule_r2(REGIME_REGISTRY))
    violations.extend(_rule_r3(REGIME_REGISTRY))
    violations.extend(_rule_r4(REGIME_REGISTRY))
    violations.extend(_rule_r5(REGIME_REGISTRY))

    # build per-family agreement table for diagnostics
    families: dict[str, dict] = {}
    for e in REGIME_REGISTRY:
        fam = e["family"]
        families.setdefault(fam, {
            "members":    [],
            "vocabulary": e["vocabulary"],
            "signature":  e["signature"],
        })
        families[fam]["members"].append({
            "func": e["func"], "path": e["path"]})

    grid_dump: dict[str, dict[str, str]] = {}
    for e in REGIME_REGISTRY:
        if e.get("signature") != "byte_label":
            continue
        key = f"{e['family']}::{e['func']}@{e['path']}"
        grid_dump[key] = {}
        for label in CANONICAL_L3_GRID:
            try:
                grid_dump[key][label] = _classify_byte_label(e, label)
            except Exception as exc:
                grid_dump[key][label] = f"<error:{exc!r}>"

    return {
        "status": "active",
        "rules": {
            "R1": "every registered classifier resolves to a callable",
            "R2": "byte-input classifiers return only declared-vocabulary labels",
            "R3": "byte-input classifiers within a family agree on the canonical L3 grid",
            "R4": "every regime-classifier function in ECG dir is registered",
            "R5": "non-byte-label classifiers have signature + note",
        },
        "registry_size":      len(REGIME_REGISTRY),
        "family_count":       len({e["family"] for e in REGIME_REGISTRY}),
        "canonical_grid":     CANONICAL_L3_GRID,
        "families":           families,
        "grid_classification": grid_dump,
        "totals": {
            "registry_size":   len(REGIME_REGISTRY),
            "family_count":    len({e["family"] for e in REGIME_REGISTRY}),
            "canonical_labels": len(CANONICAL_L3_GRID),
            "violations":      len(violations),
        },
        "violations": violations,
    }


# ----------------------------------------------------------- writers --

def _render_md(audit: dict) -> str:
    L: list[str] = []
    L.append("# L3 regime-classifier consistency (gate 245)")
    L.append("")
    t = audit["totals"]
    L.append(f"**Status:** {audit['status']}  •  "
             f"classifiers: {t['registry_size']}  •  "
             f"families: {t['family_count']}  •  "
             f"violations: {t['violations']}")
    L.append("")
    L.append("## Rules")
    for k, v in audit["rules"].items():
        L.append(f"- **{k}** — {v}")
    L.append("")
    L.append("## Families")
    for fam, info in sorted(audit["families"].items()):
        L.append(f"### `{fam}`")
        L.append(f"- signature: `{info['signature']}`")
        L.append(f"- vocabulary: {info['vocabulary']}")
        L.append("- members:")
        for m in info["members"]:
            L.append(f"  - `{m['func']}` in `{m['path']}`")
        L.append("")
    L.append("## Canonical-grid classification")
    L.append("")
    headers = ["classifier"] + CANONICAL_L3_GRID
    L.append("| " + " | ".join(headers) + " |")
    L.append("|" + "|".join(["---"] * len(headers)) + "|")
    for key, row in sorted(audit["grid_classification"].items()):
        cells = [f"`{row[label]}`" for label in CANONICAL_L3_GRID]
        L.append(f"| `{key}` | " + " | ".join(cells) + " |")
    L.append("")
    if audit["violations"]:
        L.append("## Violations")
        for v in audit["violations"]:
            L.append(f"- {v}")
    else:
        L.append("**0 violations** — every regime classifier is registered, "
                "vocab-clean, in-family agreement holds, and non-default "
                "signatures are documented.")
    return "\n".join(L) + "\n"


def _render_csv(audit: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["field", "value"])
    t = audit["totals"]
    for k in ("registry_size", "family_count", "canonical_labels",
              "violations"):
        w.writerow([k, t[k]])
    w.writerow(["families", "|".join(sorted(audit["families"].keys()))])
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
    print(f"[lit-faith-regime-classifier] status={a['status']} "
          f"classifiers={a['totals']['registry_size']} "
          f"families={a['totals']['family_count']} "
          f"violations={a['totals']['violations']}")


if __name__ == "__main__":
    main()
