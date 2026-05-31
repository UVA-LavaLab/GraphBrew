#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analysis dataclass schema audit (gate 281).

Locks the per-row analysis dataclasses across the four ECG analysis
modules that aggregate sweep observations into paper-ready tables and
invariant verdicts. Extends gate 280's receiver-side registry beyond
the orchestration carriers into the analysis layer.

Today's registry:

    scripts/experiments/ecg/paper_baseline_table.py
      - Row                  (7  fields, frozen=True)

    scripts/experiments/ecg/literature_baselines.py
      - LiteratureClaim      (10 fields, frozen=True)
      - CacheOrg             (9  fields, frozen=True)

    scripts/experiments/ecg/corpus_diversity.py
      - GraphProfile         (6  fields, @dataclass bare)

    scripts/experiments/ecg/gem5_anchor_summary.py
      - CellSummary          (6  fields, @dataclass bare)
      - AnchorInvariant      (3  fields, @dataclass bare)

Six dataclasses across four modules, 41 fields total. Each analysis
module iterates these dataclass instances and treats their field shape
as a stable per-row description. Silent edits to field names,
annotations, defaults, or @dataclass kwargs break paper-table
reproducibility, anchor verdict reproducibility, or corpus profile
serialisation even if every gate-278 receiver CLI flag and every
gate-280 receiver dataclass stays unchanged.

Rules (H1..H7) — uniform across all (module, class) entries:

  H1  module exists and ast.parses
  H2  registered class present at module top level
  H3  every registered field exists in live class body
  H4  every registered field annotation matches (string equality
       after ast.unparse on both sides)
  H5  every registered field default state matches
       (none / literal:<value> / factory:<call>)
  H6  class exhaustive — no surprise live field slips in
  H7  registered @dataclass(...) kwargs preserved on live class
       (both directions — surprise live kwargs also fail)

Twenty-eighth in the vocabulary-lock series (252 SBATCH … 280 receiver
dataclass, 281 analysis dataclass schema).
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Registry — single source of truth.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSpec:
    name: str
    annotation: str
    default: str  # "none" | "literal:<unparsed>" | "factory:<unparsed>"


@dataclass(frozen=True)
class DataclassEntry:
    module: str
    class_name: str
    decorator_kwargs: dict[str, str]
    fields: tuple[FieldSpec, ...]


REGISTRY: tuple[DataclassEntry, ...] = (
    DataclassEntry(
        module="scripts/experiments/ecg/paper_baseline_table.py",
        class_name="Row",
        decorator_kwargs={"frozen": "True"},
        fields=(
            FieldSpec("graph", "str", "none"),
            FieldSpec("app", "str", "none"),
            FieldSpec("l3_size", "str", "none"),
            FieldSpec("miss", "dict[str, float]", "none"),
            FieldSpec("delta", "dict[str, float]", "none"),
            FieldSpec("verdict", "dict[str, str]", "none"),
            FieldSpec("accesses", "int", "none"),
        ),
    ),
    DataclassEntry(
        module="scripts/experiments/ecg/literature_baselines.py",
        class_name="LiteratureClaim",
        decorator_kwargs={"frozen": "True"},
        fields=(
            FieldSpec("graph", "str", "none"),
            FieldSpec("app", "str", "none"),
            FieldSpec("l3_size", "str", "none"),
            FieldSpec("policy", "str", "none"),
            FieldSpec("expected_sign", "str", "none"),
            FieldSpec("min_abs_delta_pct", "float | None", "none"),
            FieldSpec("max_abs_delta_pct", "float | None", "none"),
            FieldSpec("tolerance_pct", "float", "none"),
            FieldSpec("rationale", "str", "none"),
            FieldSpec("citation", "str", "none"),
        ),
    ),
    DataclassEntry(
        module="scripts/experiments/ecg/literature_baselines.py",
        class_name="CacheOrg",
        decorator_kwargs={"frozen": "True"},
        fields=(
            FieldSpec("name", "str", "none"),
            FieldSpec("l1d_size", "str", "none"),
            FieldSpec("l1d_ways", "str", "none"),
            FieldSpec("l2_size", "str", "none"),
            FieldSpec("l2_ways", "str", "none"),
            FieldSpec("l3_size", "str", "none"),
            FieldSpec("l3_ways", "str", "none"),
            FieldSpec("line_size", "str", "none"),
            FieldSpec("rationale", "str", "none"),
        ),
    ),
    DataclassEntry(
        module="scripts/experiments/ecg/corpus_diversity.py",
        class_name="GraphProfile",
        decorator_kwargs={},
        fields=(
            FieldSpec("graph", "str", "none"),
            FieldSpec("log_path", "str", "none"),
            FieldSpec("nodes", "int", "literal:0"),
            FieldSpec("edges", "int", "literal:0"),
            FieldSpec("edges_directed", "bool", "literal:False"),
            FieldSpec("features", "dict", "factory:field(default_factory=dict)"),
        ),
    ),
    DataclassEntry(
        module="scripts/experiments/ecg/gem5_anchor_summary.py",
        class_name="CellSummary",
        decorator_kwargs={},
        fields=(
            FieldSpec("graph", "str", "none"),
            FieldSpec("app", "str", "none"),
            FieldSpec("l3_size", "str", "none"),
            FieldSpec("miss_rate_by_policy", "dict[str, float]",
                      "factory:field(default_factory=dict)"),
            FieldSpec("ok_rows", "int", "literal:0"),
            FieldSpec("error_rows", "int", "literal:0"),
        ),
    ),
    DataclassEntry(
        module="scripts/experiments/ecg/gem5_anchor_summary.py",
        class_name="AnchorInvariant",
        decorator_kwargs={},
        fields=(
            FieldSpec("name", "str", "none"),
            FieldSpec("status", "str", "none"),
            FieldSpec("detail", "str", "none"),
        ),
    ),
)


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _default_descriptor(value: ast.expr | None) -> str:
    if value is None:
        return "none"
    if isinstance(value, ast.Call):
        return "factory:" + ast.unparse(value)
    return "literal:" + ast.unparse(value)


def _live_fields(class_node: ast.ClassDef) -> list[FieldSpec]:
    out: list[FieldSpec] = []
    for stmt in class_node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            out.append(
                FieldSpec(
                    name=stmt.target.id,
                    annotation=ast.unparse(stmt.annotation),
                    default=_default_descriptor(stmt.value),
                )
            )
    return out


def _dataclass_kwargs(class_node: ast.ClassDef) -> dict[str, str] | None:
    """Return decorator kwargs if class has @dataclass / @dataclasses.dataclass.

    Returns {} for bare @dataclass (no call), {kw: unparsed} for
    @dataclass(kw=val, ...), or None if the class isn't dataclass-decorated.
    """
    for dec in class_node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        name = ast.unparse(target)
        if name not in {"dataclass", "dataclasses.dataclass"}:
            continue
        if not isinstance(dec, ast.Call):
            return {}
        return {kw.arg: ast.unparse(kw.value) for kw in dec.keywords if kw.arg}
    return None


def _find_class(tree: ast.AST, class_name: str) -> ast.ClassDef | None:
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


# ---------------------------------------------------------------------------
# Rule checks
# ---------------------------------------------------------------------------

def _check_entry(entry: DataclassEntry) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    module_path = REPO_ROOT / entry.module
    label = f"{entry.module}::{entry.class_name}"

    # H1
    if not module_path.exists():
        violations.append({"rule": "H1", "entry": label,
                           "msg": f"module missing: {entry.module}"})
        return violations
    try:
        src = module_path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(module_path))
    except SyntaxError as exc:
        violations.append({"rule": "H1", "entry": label,
                           "msg": f"ast.parse failed: {exc}"})
        return violations

    # H2
    class_node = _find_class(tree, entry.class_name)
    if class_node is None:
        violations.append({"rule": "H2", "entry": label,
                           "msg": f"class {entry.class_name} missing in {entry.module}"})
        return violations

    live = _live_fields(class_node)
    live_by_name = {f.name: f for f in live}
    reg_by_name = {f.name: f for f in entry.fields}

    # H3
    for reg in entry.fields:
        if reg.name not in live_by_name:
            violations.append({"rule": "H3", "entry": label,
                               "msg": f"registered field '{reg.name}' missing in live class"})

    # H4
    for reg in entry.fields:
        live_f = live_by_name.get(reg.name)
        if live_f is None:
            continue
        if live_f.annotation != reg.annotation:
            violations.append({"rule": "H4", "entry": label,
                               "msg": f"field '{reg.name}' annotation drift: registered={reg.annotation!r} live={live_f.annotation!r}"})

    # H5
    for reg in entry.fields:
        live_f = live_by_name.get(reg.name)
        if live_f is None:
            continue
        if live_f.default != reg.default:
            violations.append({"rule": "H5", "entry": label,
                               "msg": f"field '{reg.name}' default drift: registered={reg.default!r} live={live_f.default!r}"})

    # H6
    for live_f in live:
        if live_f.name not in reg_by_name:
            violations.append({"rule": "H6", "entry": label,
                               "msg": f"surprise live field '{live_f.name}' not in registry"})

    # H7
    live_kwargs = _dataclass_kwargs(class_node)
    if live_kwargs is None:
        violations.append({"rule": "H7", "entry": label,
                           "msg": f"class {entry.class_name} is not @dataclass decorated"})
    else:
        for kw, expected in entry.decorator_kwargs.items():
            got = live_kwargs.get(kw)
            if got != expected:
                violations.append({"rule": "H7", "entry": label,
                                   "msg": f"decorator kwarg drift: registered {kw}={expected} live {kw}={got}"})
        for kw, got in live_kwargs.items():
            if kw not in entry.decorator_kwargs:
                violations.append({"rule": "H7", "entry": label,
                                   "msg": f"surprise live decorator kwarg {kw}={got} not in registry"})

    return violations


def _run_audit() -> tuple[int, int, int, list[dict[str, Any]]]:
    all_violations: list[dict[str, Any]] = []
    classes = 0
    fields_total = 0
    for entry in REGISTRY:
        classes += 1
        fields_total += len(entry.fields)
        all_violations.extend(_check_entry(entry))
    return classes, fields_total, len(all_violations), all_violations


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_json(path: Path, classes: int, fields_total: int,
               violations: list[dict[str, Any]]) -> None:
    payload = {
        "gate": 281,
        "status": "active",
        "registry": [
            {
                "module": e.module,
                "class": e.class_name,
                "decorator_kwargs": dict(e.decorator_kwargs),
                "fields": [
                    {"name": f.name, "annotation": f.annotation, "default": f.default}
                    for f in e.fields
                ],
            }
            for e in REGISTRY
        ],
        "modules": sorted({e.module for e in REGISTRY}),
        "classes": classes,
        "fields": fields_total,
        "violations": violations,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def write_md(path: Path, classes: int, fields_total: int,
             violations: list[dict[str, Any]]) -> None:
    lines = []
    lines.append("# Analysis dataclass schema (gate 281)")
    lines.append("")
    lines.append("Locks the per-row analysis dataclasses across the four ECG "
                 "analysis modules that aggregate sweep observations into "
                 "paper-ready tables and invariant verdicts — extends gate "
                 "280's receiver-side registry beyond the orchestration "
                 "carriers into the analysis layer.")
    lines.append("")
    lines.append(f"**Classes:** {classes}")
    lines.append(f"**Fields:** {fields_total}")
    lines.append(f"**Violations:** {len(violations)}")
    lines.append("")
    lines.append("## Registry")
    lines.append("")
    for e in REGISTRY:
        kw = ", ".join(f"{k}={v}" for k, v in e.decorator_kwargs.items())
        lines.append(f"### `{e.module}` :: `{e.class_name}` — `@dataclass({kw})`")
        lines.append("")
        lines.append("| field | annotation | default |")
        lines.append("|---|---|---|")
        for f in e.fields:
            lines.append(f"| `{f.name}` | `{f.annotation}` | `{f.default}` |")
        lines.append("")
    if violations:
        lines.append("## ⛔ Violations")
        lines.append("")
        lines.append("| rule | entry | message |")
        lines.append("|---|---|---|")
        for v in violations:
            lines.append(f"| {v['rule']} | `{v['entry']}` | {v['msg']} |")
    else:
        lines.append("## ✅ No violations")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["module", "class", "decorator_kwargs",
                    "field", "annotation", "default"])
        for e in REGISTRY:
            kw_str = ";".join(f"{k}={v}" for k, v in sorted(e.decorator_kwargs.items()))
            for f in e.fields:
                w.writerow([e.module, e.class_name, kw_str,
                            f.name, f.annotation, f.default])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--json-out", required=True, type=Path)
    p.add_argument("--md-out", required=True, type=Path)
    p.add_argument("--csv-out", required=True, type=Path)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    classes, fields_total, n_viol, violations = _run_audit()
    write_json(args.json_out, classes, fields_total, violations)
    write_md(args.md_out, classes, fields_total, violations)
    write_csv(args.csv_out)
    print(f"[lit-faith-analysis-dataclass-schema] status=active "
          f"classes={classes} fields={fields_total} violations={n_viol}")
    return 0 if n_viol == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
