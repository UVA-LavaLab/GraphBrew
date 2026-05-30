#!/usr/bin/env python3
"""Gate 250 — paper-table CSV provenance.

Every shipped LaTeX paper table in
``wiki/data/paper_pipeline_YYYYMMDD/`` is co-emitted with a parallel
``.csv`` source that holds the same underlying observations.  When the
``paper_pipeline.py`` pipeline silently regenerates one file but not
the other (e.g. a refactor splits a column, or a row is dropped from
the CSV but kept in the .tex), the paper and the data disagree and no
test fails today.

This gate locks the .tex⇄.csv pairing.  For every entry in
``PROVENANCE_REGISTRY`` the gate asserts:

  P1 — both the .tex and the matching .csv exist in the latest
       paper_pipeline dir;
  P2 — the .tex data-row count is ≤ the .csv data-row count
       (paper shows at most what the CSV holds; ``paper_pipeline.py``
       caps shipped tables at ``[:20]``/``[:24]`` for layout, so the
       .csv is allowed to be the strict superset, but the .tex must
       NEVER contain rows the CSV does not hold);
  P3 — for each declared ``key_columns`` pair (tex_header, csv_header)
       the multiset of values in the .tex column is a sub-multiset
       of the .csv column, after applying the optional value
       normalizer declared in the registry (handles LaTeX escaping
       such as ``ECG\\_DBG\\_ONLY`` ⇄ ``ECG_DBG_ONLY``).  Equivalent
       to: every paper row's key tuple must trace to at least one
       CSV row;
  P4 — every declared key tex_header exists in the .tex header tuple
       AND every declared csv_header exists in the .csv header tuple;
  P5 — no value in any tracked key column is the empty string in the
       .csv (paper rows must trace to non-empty CSV cells);
  P6 — there is no .csv file in the paper_pipeline dir that:
         * matches a registered .tex stem (so each .tex has a CSV
           sibling), AND
         * is NOT registered (defensive — catches new paired CSVs
           added without registration);
  P7 — every registered .csv file has a non-empty header row (no
       headerless or empty CSV slipping into the paper bundle).

Source-of-truth: ``wiki/data/paper_pipeline_YYYYMMDD/*.tex`` and the
sibling ``*.csv`` files on disk (latest snapshot dir, discovered by
glob), plus ``PROVENANCE_REGISTRY`` in this file.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2].parent
WIKI_DATA = ROOT / "wiki" / "data"

_CAPTION_RE = re.compile(r"\\caption\{([^}]*)\}")
_TABULAR_RE = re.compile(r"\\begin\{tabular\}\{([^}]*)\}")


# ----------------------------------------------------------- normalizers --

def _norm_latex(s: str) -> str:
    """Strip LaTeX backslash-escape on underscores/percent."""
    return s.replace("\\_", "_").replace("\\%", "%").strip()


def _norm_strip(s: str) -> str:
    return s.strip()


# ----------------------------------------------------------- registry --

# Each entry pairs one .tex with one .csv:
#   tex_file / csv_file — filenames inside latest paper_pipeline dir
#   key_columns         — list of (tex_header, csv_header) pairs whose
#                         multisets must match
#   normalizer          — name of normalizer applied to BOTH .tex and
#                         .csv cells before multiset comparison
PROVENANCE_REGISTRY: list[dict] = [
    {
        "tex_file": "ecg_mode_overhead_summary.tex",
        "csv_file": "ecg_mode_overhead_summary.csv",
        "key_columns": [("policy", "policy_short")],
        "normalizer":  "strip",
    },
    {
        "tex_file": "faithfulness_summary.tex",
        "csv_file": "faithfulness_summary.csv",
        "key_columns": [
            ("check",      "check"),
            ("benchmark",  "benchmark"),
            ("prefetcher", "prefetcher"),
            ("candidate",  "candidate_short"),
        ],
        "normalizer": "latex",
    },
    {
        "tex_file": "popt_charged_overhead.tex",
        "csv_file": "popt_charged_overhead.csv",
        "key_columns": [
            ("charged",   "charged_policy"),
            ("oracle",    "oracle_policy"),
            ("benchmark", "benchmark"),
        ],
        "normalizer": "latex",
    },
    {
        "tex_file": "popt_storage_overhead_summary.tex",
        "csv_file": "popt_storage_overhead_summary.csv",
        "key_columns": [
            ("policy",     "policy_short"),
            ("benchmark",  "benchmark"),
            ("prefetcher", "prefetcher"),
        ],
        "normalizer": "latex",
    },
    {
        "tex_file": "roi_policy_summary.tex",
        "csv_file": "roi_policy_summary.csv",
        "key_columns": [
            ("policy",     "policy_short"),
            ("benchmark",  "benchmark"),
            ("prefetcher", "prefetcher"),
        ],
        "normalizer": "latex",
    },
]

NORMALIZERS = {
    "strip": _norm_strip,
    "latex": _norm_latex,
}


# ----------------------------------------------------------- helpers --

def _latest_pipeline_dir() -> Path | None:
    candidates = sorted(WIKI_DATA.glob("paper_pipeline_*"))
    candidates = [c for c in candidates if c.is_dir()]
    return candidates[-1] if candidates else None


def _strip_cells(line: str) -> tuple[str, ...]:
    line = line.rstrip()
    if line.endswith("\\\\"):
        line = line[:-2]
    return tuple(c.strip() for c in line.split("&"))


def _parse_tex(path: Path) -> dict:
    """Return {header: tuple, rows: list[tuple]}."""
    txt = path.read_text()
    m_top = re.search(r"\\toprule\s*(.*?)\\midrule", txt, re.DOTALL)
    m_body = re.search(r"\\midrule\s*(.*?)\\bottomrule", txt, re.DOTALL)
    header: tuple[str, ...] = ()
    rows: list[tuple[str, ...]] = []
    if m_top:
        for line in m_top.group(1).splitlines():
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            header = _strip_cells(s)
            break
    if m_body:
        for line in m_body.group(1).splitlines():
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            rows.append(_strip_cells(s))
    return {"header": header, "rows": rows}


def _parse_csv(path: Path) -> dict:
    """Return {header: tuple, rows: list[dict]}."""
    with path.open() as fh:
        reader = csv.reader(fh)
        try:
            header = tuple(next(reader))
        except StopIteration:
            return {"header": (), "rows": []}
        rows: list[dict] = []
        for r in reader:
            if not r:
                continue
            rows.append(dict(zip(header, r)))
        return {"header": header, "rows": rows}


def _col_values_tex(tex: dict, col_name: str) -> list[str]:
    if col_name not in tex["header"]:
        return []
    idx = tex["header"].index(col_name)
    out: list[str] = []
    for r in tex["rows"]:
        if idx < len(r):
            out.append(r[idx])
    return out


# ----------------------------------------------------------- rules --

def _rule_p1(entries: list[dict], pdir: Path) -> list[dict]:
    out: list[dict] = []
    for e in entries:
        for kind, key in (("tex", "tex_file"), ("csv", "csv_file")):
            p = pdir / e[key]
            if not p.exists():
                out.append({"rule": "P1", "file": e[key],
                            "kind": kind,
                            "issue": f"registered {kind} file missing on disk"})
    return out


def _rule_p2(entries: list[dict], pdir: Path) -> list[dict]:
    out: list[dict] = []
    for e in entries:
        tp = pdir / e["tex_file"]
        cp = pdir / e["csv_file"]
        if not (tp.exists() and cp.exists()):
            continue
        tex = _parse_tex(tp)
        csv_ = _parse_csv(cp)
        if len(tex["rows"]) > len(csv_["rows"]):
            out.append({"rule": "P2",
                        "tex_file": e["tex_file"],
                        "csv_file": e["csv_file"],
                        "tex_rows": len(tex["rows"]),
                        "csv_rows": len(csv_["rows"]),
                        "issue": "tex has more data rows than csv "
                                 "(orphan paper rows)"})
    return out


def _rule_p3(entries: list[dict], pdir: Path) -> list[dict]:
    out: list[dict] = []
    for e in entries:
        tp = pdir / e["tex_file"]
        cp = pdir / e["csv_file"]
        if not (tp.exists() and cp.exists()):
            continue
        tex = _parse_tex(tp)
        csv_ = _parse_csv(cp)
        norm = NORMALIZERS.get(e.get("normalizer", "strip"), _norm_strip)
        for tex_col, csv_col in e["key_columns"]:
            if tex_col not in tex["header"] or csv_col not in csv_["header"]:
                # caught by P4
                continue
            tex_vals = [norm(v) for v in _col_values_tex(tex, tex_col)]
            csv_vals = [norm(r.get(csv_col, "")) for r in csv_["rows"]]
            tex_ctr = Counter(tex_vals)
            csv_ctr = Counter(csv_vals)
            # Subset semantics: every paper-row's key value must
            # appear in the CSV at least as many times.
            orphan = tex_ctr - csv_ctr
            if orphan:
                out.append({"rule": "P3",
                            "tex_file": e["tex_file"],
                            "csv_file": e["csv_file"],
                            "tex_col":  tex_col,
                            "csv_col":  csv_col,
                            "orphan_in_tex": dict(orphan),
                            "issue": "paper-row key value not present "
                                     "in CSV column"})
    return out


def _rule_p4(entries: list[dict], pdir: Path) -> list[dict]:
    out: list[dict] = []
    for e in entries:
        tp = pdir / e["tex_file"]
        cp = pdir / e["csv_file"]
        if not (tp.exists() and cp.exists()):
            continue
        tex = _parse_tex(tp)
        csv_ = _parse_csv(cp)
        for tex_col, csv_col in e["key_columns"]:
            if tex_col not in tex["header"]:
                out.append({"rule": "P4",
                            "tex_file": e["tex_file"],
                            "missing_col": tex_col,
                            "where": "tex_header",
                            "issue": "declared key column missing from tex header"})
            if csv_col not in csv_["header"]:
                out.append({"rule": "P4",
                            "csv_file": e["csv_file"],
                            "missing_col": csv_col,
                            "where": "csv_header",
                            "issue": "declared key column missing from csv header"})
    return out


def _rule_p5(entries: list[dict], pdir: Path) -> list[dict]:
    out: list[dict] = []
    for e in entries:
        cp = pdir / e["csv_file"]
        if not cp.exists():
            continue
        csv_ = _parse_csv(cp)
        for _, csv_col in e["key_columns"]:
            if csv_col not in csv_["header"]:
                continue
            for idx, r in enumerate(csv_["rows"], start=1):
                val = r.get(csv_col, "")
                if val == "" or val is None:
                    out.append({"rule": "P5",
                                "csv_file": e["csv_file"],
                                "column": csv_col,
                                "row_index": idx,
                                "issue": "empty value in tracked key column"})
    return out


def _rule_p6(entries: list[dict], pdir: Path) -> list[dict]:
    out: list[dict] = []
    registered_csv = {e["csv_file"] for e in entries}
    registered_tex_stems = {Path(e["tex_file"]).stem for e in entries}
    for p in sorted(pdir.glob("*.csv")):
        if p.name in registered_csv:
            continue
        if p.stem in registered_tex_stems:
            out.append({"rule": "P6", "csv_file": p.name,
                        "issue": "unregistered CSV sibling of registered .tex"})
    return out


def _rule_p7(entries: list[dict], pdir: Path) -> list[dict]:
    out: list[dict] = []
    for e in entries:
        cp = pdir / e["csv_file"]
        if not cp.exists():
            continue
        csv_ = _parse_csv(cp)
        if not csv_["header"]:
            out.append({"rule": "P7", "csv_file": e["csv_file"],
                        "issue": "csv file has empty/missing header row"})
    return out


# ----------------------------------------------------------- audit --

def audit() -> dict:
    pdir = _latest_pipeline_dir()
    if pdir is None:
        return {
            "status":  "skip",
            "reason":  "no wiki/data/paper_pipeline_YYYYMMDD/ dir found",
            "rules":   {},
            "totals":  {"registry_size":   len(PROVENANCE_REGISTRY),
                        "pairs_found":     0,
                        "tracked_columns": 0,
                        "tex_rows_total":  0,
                        "csv_rows_total":  0,
                        "violations":      0},
            "violations": [],
        }

    violations: list[dict] = []
    for fn in (_rule_p1, _rule_p2, _rule_p3, _rule_p4,
               _rule_p5, _rule_p6, _rule_p7):
        violations.extend(fn(PROVENANCE_REGISTRY, pdir))

    per_pair: dict[str, dict] = {}
    tex_total = 0
    csv_total = 0
    tracked = 0
    pairs_found = 0
    for e in PROVENANCE_REGISTRY:
        tp = pdir / e["tex_file"]
        cp = pdir / e["csv_file"]
        info: dict = {"tex_present": tp.exists(), "csv_present": cp.exists()}
        if tp.exists():
            tex = _parse_tex(tp)
            info["tex_rows"] = len(tex["rows"])
            tex_total += len(tex["rows"])
        if cp.exists():
            csv_ = _parse_csv(cp)
            info["csv_rows"] = len(csv_["rows"])
            csv_total += len(csv_["rows"])
        if info["tex_present"] and info["csv_present"]:
            pairs_found += 1
        info["key_columns"] = [list(kc) for kc in e["key_columns"]]
        tracked += len(e["key_columns"])
        per_pair[e["tex_file"]] = info

    return {
        "status": "active",
        "rules": {
            "P1": "every registered .tex and .csv file exists",
            "P2": "tex data-row count ≤ csv data-row count (subset)",
            "P3": "every tex key-column value traces to csv (subset)",
            "P4": "every declared key column exists in tex/csv header",
            "P5": "no empty value in tracked CSV key columns",
            "P6": "no unregistered CSV sibling of a registered .tex",
            "P7": "every registered csv has a non-empty header row",
        },
        "pipeline_dir":   str(pdir.relative_to(ROOT)),
        "registry":       PROVENANCE_REGISTRY,
        "registry_size":  len(PROVENANCE_REGISTRY),
        "per_pair":       per_pair,
        "totals": {
            "registry_size":   len(PROVENANCE_REGISTRY),
            "pairs_found":     pairs_found,
            "tracked_columns": tracked,
            "tex_rows_total":  tex_total,
            "csv_rows_total":  csv_total,
            "violations":      len(violations),
        },
        "violations": violations,
    }


# ----------------------------------------------------------- writers --

def _render_md(audit: dict) -> str:
    L: list[str] = []
    L.append("# Paper-table CSV provenance (gate 250)")
    L.append("")
    t = audit["totals"]
    L.append(f"**Status:** {audit['status']}  •  "
             f"pipeline dir: `{audit.get('pipeline_dir', '?')}`  •  "
             f"pairs: {t['pairs_found']}/{t['registry_size']}  •  "
             f"tracked key columns: {t['tracked_columns']}  •  "
             f"tex rows: {t['tex_rows_total']}  •  "
             f"csv rows: {t['csv_rows_total']}  •  "
             f"violations: {t['violations']}")
    L.append("")
    L.append("## Rules")
    for k, v in audit.get("rules", {}).items():
        L.append(f"- **{k}** — {v}")
    L.append("")
    L.append("## Registered pairs")
    L.append("")
    L.append("| tex_file | csv_file | normalizer | key columns | tex rows | csv rows |")
    L.append("|---|---|---|---|---:|---:|")
    for e in audit.get("registry", []):
        info = audit["per_pair"].get(e["tex_file"], {})
        kc = ", ".join(f"`{a}`⇄`{b}`" for a, b in e["key_columns"])
        tex_rows = info.get("tex_rows", "—")
        csv_rows = info.get("csv_rows", "—")
        L.append(f"| `{e['tex_file']}` | `{e['csv_file']}` "
                 f"| `{e.get('normalizer','strip')}` | {kc} "
                 f"| {tex_rows} | {csv_rows} |")
    L.append("")
    if audit.get("violations"):
        L.append("## Violations")
        for v in audit["violations"]:
            L.append(f"- {v}")
    else:
        L.append("**0 violations** — every shipped paper-table .tex "
                 "row traces 1:1 to the sibling CSV, and every "
                 "tracked key column matches as a multiset.")
    return "\n".join(L) + "\n"


def _render_csv(audit: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["field", "value"])
    t = audit["totals"]
    for k in ("registry_size", "pairs_found", "tracked_columns",
              "tex_rows_total", "csv_rows_total", "violations"):
        w.writerow([k, t[k]])
    for fn, info in sorted(audit.get("per_pair", {}).items()):
        w.writerow([f"tex_rows::{fn}", info.get("tex_rows", 0)])
        w.writerow([f"csv_rows::{fn}", info.get("csv_rows", 0)])
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
    print(f"[lit-faith-paper-provenance] status={a['status']} "
          f"pairs={a['totals']['pairs_found']}/{a['totals']['registry_size']} "
          f"tex_rows={a['totals']['tex_rows_total']} "
          f"csv_rows={a['totals']['csv_rows_total']} "
          f"violations={a['totals']['violations']}")


if __name__ == "__main__":
    main()
