#!/usr/bin/env python3
"""Gate 247 — paper LaTeX-table emit invariant.

`wiki/data/paper_pipeline_YYYYMMDD/` carries auto-generated LaTeX
tables that go straight into the paper. Each table has a stable
caption, column count, and column-header tuple — and if any of those
silently change, the paper text and the table fall out of sync and
no test fails today.

This gate codifies each shipped table with a hand-curated
`TABLE_REGISTRY` declaring its filename + expected caption + expected
``\\begin{tabular}{...}`` column spec + expected header row.

For every registered table the gate asserts:

  T1 — the .tex file exists in the latest paper_pipeline dir;
  T2 — registered caption matches the in-file caption exactly;
  T3 — registered tabular column-spec matches the in-file spec;
  T4 — registered header tuple matches the in-file header tuple;
  T5 — every data row has exactly ``len(columns)`` cells,
       no cell is the literal string ``"nan"`` or ``"NaN"``,
       no cell is the literal string ``"inf"`` or ``"-inf"``;
  T6 — there is no .tex file in the paper_pipeline dir that
       is NOT registered (defensive — catches new tables added
       without registration);
  T7 — every table file ends with the
       ``\\bottomrule\\end{tabular}\\end{table}`` closing trio
       (no truncated LaTeX).

Source-of-truth: ``wiki/data/paper_pipeline_YYYYMMDD/*.tex`` on disk
(the latest snapshot dir, discovered by glob), and the
``TABLE_REGISTRY`` in this file.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2].parent
WIKI_DATA = ROOT / "wiki" / "data"

# Regex helpers
_CAPTION_RE  = re.compile(r"\\caption\{([^}]*)\}")
_TABULAR_RE  = re.compile(r"\\begin\{tabular\}\{([^}]*)\}")
_BOTTOM_RE   = re.compile(
    r"\\bottomrule\s*\\end\{tabular\}\s*\\end\{table\}\s*\Z")


# ----------------------------------------------------------- registry --

# Each entry declares one shipped paper table.
#   filename — file name inside the latest paper_pipeline_YYYYMMDD/ dir
#   caption  — exact text inside \caption{...}
#   col_spec — exact contents of \begin{tabular}{...} (without braces)
#   columns  — header tuple as it appears in the table
TABLE_REGISTRY: list[dict] = [
    {
        "filename": "ecg_mode_overhead_summary.tex",
        "caption":  "ECG mode overhead summary",
        "col_spec": "llll",
        "columns":  ("policy", "P-OPT lookup", "charged\\_alias",
                     "reserved ways"),
    },
    {
        "filename": "faithfulness_summary.tex",
        "caption":  "ECG faithfulness and parity checks",
        "col_spec": "llllllll",
        "columns":  ("check", "benchmark", "prefetcher", "reference",
                     "candidate", "max tick delta (\\%)",
                     "max LLC delta (\\%)", "pass"),
    },
    {
        "filename": "popt_charged_overhead.tex",
        "caption":  "P-OPT charged overhead summary",
        "col_spec": "llllll",
        "columns":  ("charged", "oracle", "benchmark", "section",
                     "tick delta (\\%)", "LLC delta (\\%)"),
    },
    {
        "filename": "popt_storage_overhead_summary.tex",
        "caption":  "P-OPT storage and stream overhead summary",
        "col_spec": "lllllll",
        "columns":  ("policy", "benchmark", "prefetcher",
                     "reserved ways", "reserved B",
                     "reserved LLC (\\%)", "matrix stream lines"),
    },
    {
        "filename": "roi_policy_summary.tex",
        "caption":  "ECG normalized ROI summary",
        "col_spec": "lllll",
        "columns":  ("policy", "benchmark", "prefetcher",
                     "avg speedup", "avg LLC red. (\\%)"),
    },
]


# ----------------------------------------------------------- helpers --

def _latest_pipeline_dir() -> Path | None:
    candidates = sorted(WIKI_DATA.glob("paper_pipeline_*"))
    candidates = [c for c in candidates if c.is_dir()]
    return candidates[-1] if candidates else None


def _strip_header_cells(line: str) -> tuple[str, ...]:
    line = line.rstrip()
    if line.endswith("\\\\"):
        line = line[:-2]
    return tuple(c.strip() for c in line.split("&"))


def _parse_table(path: Path) -> dict:
    """Return {caption, col_spec, header, data_rows, raw_tail}."""
    txt = path.read_text()
    caption = (_CAPTION_RE.search(txt) or [None, None]).group(1)
    tabular = (_TABULAR_RE.search(txt) or [None, None]).group(1)

    # Locate header row: first non-comment, non-control line that has '&'
    # between \toprule and \midrule.
    m_top = re.search(r"\\toprule\s*(.*?)\\midrule", txt, re.DOTALL)
    m_body = re.search(r"\\midrule\s*(.*?)\\bottomrule", txt, re.DOTALL)
    header: tuple[str, ...] = ()
    data_rows: list[tuple[str, ...]] = []
    if m_top:
        for line in m_top.group(1).splitlines():
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            header = _strip_header_cells(s)
            break
    if m_body:
        for line in m_body.group(1).splitlines():
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            data_rows.append(_strip_header_cells(s))

    return {
        "caption":   caption,
        "col_spec":  tabular,
        "header":    header,
        "data_rows": data_rows,
        "raw":       txt,
    }


# ----------------------------------------------------------- rules --

def _rule_t1(entries: list[dict], pdir: Path) -> list[dict]:
    out: list[dict] = []
    for e in entries:
        p = pdir / e["filename"]
        if not p.exists():
            out.append({"rule": "T1", "filename": e["filename"],
                        "issue": "registered table file missing on disk"})
    return out


def _rule_t2(entries: list[dict], pdir: Path) -> list[dict]:
    out: list[dict] = []
    for e in entries:
        p = pdir / e["filename"]
        if not p.exists():
            continue
        info = _parse_table(p)
        if info["caption"] != e["caption"]:
            out.append({"rule": "T2", "filename": e["filename"],
                        "expected": e["caption"], "got": info["caption"],
                        "issue": "caption drift"})
    return out


def _rule_t3(entries: list[dict], pdir: Path) -> list[dict]:
    out: list[dict] = []
    for e in entries:
        p = pdir / e["filename"]
        if not p.exists():
            continue
        info = _parse_table(p)
        if info["col_spec"] != e["col_spec"]:
            out.append({"rule": "T3", "filename": e["filename"],
                        "expected": e["col_spec"], "got": info["col_spec"],
                        "issue": "tabular column-spec drift"})
    return out


def _rule_t4(entries: list[dict], pdir: Path) -> list[dict]:
    out: list[dict] = []
    for e in entries:
        p = pdir / e["filename"]
        if not p.exists():
            continue
        info = _parse_table(p)
        if info["header"] != e["columns"]:
            out.append({"rule": "T4", "filename": e["filename"],
                        "expected": list(e["columns"]),
                        "got":      list(info["header"]),
                        "issue":   "column-header drift"})
    return out


_BAD_VALUES = {"nan", "NaN", "NAN", "inf", "-inf", "Inf", "-Inf"}


def _rule_t5(entries: list[dict], pdir: Path) -> list[dict]:
    out: list[dict] = []
    for e in entries:
        p = pdir / e["filename"]
        if not p.exists():
            continue
        info = _parse_table(p)
        expected_cols = len(e["columns"])
        for idx, row in enumerate(info["data_rows"], start=1):
            if len(row) != expected_cols:
                out.append({"rule": "T5", "filename": e["filename"],
                            "row_index": idx, "row_cells": list(row),
                            "expected_cells": expected_cols,
                            "issue": "row column-count mismatch"})
                continue
            for c_idx, cell in enumerate(row):
                if cell in _BAD_VALUES:
                    out.append({"rule": "T5", "filename": e["filename"],
                                "row_index": idx, "column_index": c_idx,
                                "value": cell,
                                "issue": "row contains NaN/Inf cell"})
    return out


def _rule_t6(entries: list[dict], pdir: Path) -> list[dict]:
    out: list[dict] = []
    registered = {e["filename"] for e in entries}
    for p in sorted(pdir.glob("*.tex")):
        if p.name not in registered:
            out.append({"rule": "T6", "filename": p.name,
                        "issue": "unregistered .tex file in paper_pipeline dir"})
    return out


def _rule_t7(entries: list[dict], pdir: Path) -> list[dict]:
    out: list[dict] = []
    for e in entries:
        p = pdir / e["filename"]
        if not p.exists():
            continue
        txt = p.read_text()
        if not _BOTTOM_RE.search(txt):
            out.append({"rule": "T7", "filename": e["filename"],
                        "issue": "missing \\bottomrule\\end{tabular}\\end{table} "
                                 "closing trio (truncated LaTeX?)"})
    return out


# ----------------------------------------------------------- audit --

def audit() -> dict:
    pdir = _latest_pipeline_dir()
    if pdir is None:
        return {
            "status":  "skip",
            "reason":  "no wiki/data/paper_pipeline_YYYYMMDD/ dir found",
            "rules":   {},
            "totals":  {"registry_size": len(TABLE_REGISTRY),
                        "tables_found": 0, "violations": 0,
                        "row_total": 0},
            "violations": [],
        }

    violations: list[dict] = []
    for fn in (_rule_t1, _rule_t2, _rule_t3, _rule_t4,
               _rule_t5, _rule_t6, _rule_t7):
        violations.extend(fn(TABLE_REGISTRY, pdir))

    # per-table diagnostics
    per_table: dict[str, dict] = {}
    row_total = 0
    for e in TABLE_REGISTRY:
        p = pdir / e["filename"]
        if not p.exists():
            per_table[e["filename"]] = {"present": False}
            continue
        info = _parse_table(p)
        per_table[e["filename"]] = {
            "present":    True,
            "caption":    info["caption"],
            "col_spec":   info["col_spec"],
            "columns":    list(info["header"]),
            "row_count":  len(info["data_rows"]),
        }
        row_total += len(info["data_rows"])

    return {
        "status": "active",
        "rules": {
            "T1": "every registered table file exists",
            "T2": "registered caption matches in-file caption",
            "T3": "registered tabular col-spec matches in-file spec",
            "T4": "registered column-header tuple matches in-file",
            "T5": "every data row has correct column count and no NaN/Inf cells",
            "T6": "no unregistered .tex file in paper_pipeline dir",
            "T7": "every table ends with \\bottomrule\\end{tabular}\\end{table}",
        },
        "pipeline_dir":  str(pdir.relative_to(ROOT)),
        "registry":      TABLE_REGISTRY,
        "registry_size": len(TABLE_REGISTRY),
        "per_table":     per_table,
        "totals": {
            "registry_size": len(TABLE_REGISTRY),
            "tables_found":  sum(1 for v in per_table.values()
                                 if v.get("present")),
            "row_total":     row_total,
            "violations":    len(violations),
        },
        "violations": violations,
    }


# ----------------------------------------------------------- writers --

def _render_md(audit: dict) -> str:
    L: list[str] = []
    L.append("# Paper LaTeX-table emit invariant (gate 247)")
    L.append("")
    t = audit["totals"]
    L.append(f"**Status:** {audit['status']}  •  "
             f"pipeline dir: `{audit.get('pipeline_dir', '?')}`  •  "
             f"tables: {t['tables_found']}/{t['registry_size']}  •  "
             f"rows: {t['row_total']}  •  "
             f"violations: {t['violations']}")
    L.append("")
    L.append("## Rules")
    for k, v in audit.get("rules", {}).items():
        L.append(f"- **{k}** — {v}")
    L.append("")
    L.append("## Registered tables")
    L.append("")
    L.append("| filename | col_spec | columns | rows |")
    L.append("|---|---|---|---:|")
    for e in audit.get("registry", []):
        info = audit["per_table"].get(e["filename"], {})
        if not info.get("present"):
            L.append(f"| `{e['filename']}` | — | — | _MISSING_ |")
            continue
        cols = ", ".join(f"`{c}`" for c in info["columns"])
        L.append(f"| `{e['filename']}` | `{info['col_spec']}` "
                 f"| {cols} | {info['row_count']} |")
    L.append("")
    if audit.get("violations"):
        L.append("## Violations")
        for v in audit["violations"]:
            L.append(f"- {v}")
    else:
        L.append("**0 violations** — every registered paper table "
                "matches its expected caption, column-spec, header, "
                "and emits clean rows.")
    return "\n".join(L) + "\n"


def _render_csv(audit: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["field", "value"])
    t = audit["totals"]
    for k in ("registry_size", "tables_found", "row_total", "violations"):
        w.writerow([k, t[k]])
    for fn, info in sorted(audit.get("per_table", {}).items()):
        w.writerow([f"rows::{fn}", info.get("row_count", 0)])
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
    print(f"[lit-faith-paper-tables] status={a['status']} "
          f"tables={a['totals']['tables_found']}/{a['totals']['registry_size']} "
          f"rows={a['totals']['row_total']} "
          f"violations={a['totals']['violations']}")


if __name__ == "__main__":
    main()
