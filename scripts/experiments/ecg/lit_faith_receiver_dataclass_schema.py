#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Receiver dataclass schema audit (gate 280).

Locks the orchestration-carrier dataclasses on the RECEIVING side of the
publish-reproduction subprocess contract — the analog of gate 279's
sender-side Job lock.

Today's registry:

    scripts/experiments/ecg/roi_matrix.py
      - PolicySpec     (4 fields, frozen=True)

    scripts/experiments/ecg/proof_matrix.py
      - Ablation       (6 fields, frozen=True)
      - AdaptiveSelector (3 fields, frozen=True)

Each receiver iterates these dataclass instances and treats their field
shape as a stable per-row description. Silent edits to field names,
annotations, defaults, or @dataclass kwargs (frozen, eq, order, ...)
break replay parity even if every gate-278 receiver CLI flag stays
unchanged.

Rules (G1..G7) — uniform across all (module, class) entries:

  G1  module exists and ast.parses
  G2  registered class present at module top level
  G3  every registered field exists in live class body
  G4  every registered field annotation matches (string equality after
       ast.unparse on both sides)
  G5  every registered field default state matches
       (none / literal:<value> / factory:<call>)
  G6  class exhaustive — no surprise live field slips in
  G7  registered @dataclass(...) kwargs preserved on live class

Twenty-seventh in the vocabulary-lock series.
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
        module="scripts/experiments/ecg/roi_matrix.py",
        class_name="PolicySpec",
        decorator_kwargs={"frozen": "True"},
        fields=(
            FieldSpec("label", "str", "none"),
            FieldSpec("policy", "str", "none"),
            FieldSpec("ecg_mode", "str | None", "literal:None"),
            FieldSpec("charge_popt_overhead", "bool", "literal:False"),
        ),
    ),
    DataclassEntry(
        module="scripts/experiments/ecg/proof_matrix.py",
        class_name="Ablation",
        decorator_kwargs={"frozen": "True"},
        fields=(
            FieldSpec("label", "str", "none"),
            FieldSpec("group", "str", "none"),
            FieldSpec("policy", "str", "none"),
            FieldSpec("pfx_mode", "int", "literal:0"),
            FieldSpec("pfx_lookahead", "int", "literal:0"),
            FieldSpec("note", "str", "literal:''"),
        ),
    ),
    DataclassEntry(
        module="scripts/experiments/ecg/proof_matrix.py",
        class_name="AdaptiveSelector",
        decorator_kwargs={"frozen": "True"},
        fields=(
            FieldSpec("label", "str", "none"),
            FieldSpec("candidates", "tuple[str, ...]", "none"),
            FieldSpec("note", "str", "none"),
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

    # G1
    if not module_path.exists():
        violations.append({"rule": "G1", "entry": label,
                           "msg": f"module missing: {entry.module}"})
        return violations
    try:
        src = module_path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(module_path))
    except SyntaxError as exc:
        violations.append({"rule": "G1", "entry": label,
                           "msg": f"ast.parse failed: {exc}"})
        return violations

    # G2
    class_node = _find_class(tree, entry.class_name)
    if class_node is None:
        violations.append({"rule": "G2", "entry": label,
                           "msg": f"class {entry.class_name} missing in {entry.module}"})
        return violations

    live = _live_fields(class_node)
    live_by_name = {f.name: f for f in live}
    reg_by_name = {f.name: f for f in entry.fields}

    # G3
    for reg in entry.fields:
        if reg.name not in live_by_name:
            violations.append({"rule": "G3", "entry": label,
                               "msg": f"registered field '{reg.name}' missing in live class"})

    # G4
    for reg in entry.fields:
        live_f = live_by_name.get(reg.name)
        if live_f is None:
            continue
        if live_f.annotation != reg.annotation:
            violations.append({"rule": "G4", "entry": label,
                               "msg": f"field '{reg.name}' annotation drift: registered={reg.annotation!r} live={live_f.annotation!r}"})

    # G5
    for reg in entry.fields:
        live_f = live_by_name.get(reg.name)
        if live_f is None:
            continue
        if live_f.default != reg.default:
            violations.append({"rule": "G5", "entry": label,
                               "msg": f"field '{reg.name}' default drift: registered={reg.default!r} live={live_f.default!r}"})

    # G6
    for live_f in live:
        if live_f.name not in reg_by_name:
            violations.append({"rule": "G6", "entry": label,
                               "msg": f"surprise live field '{live_f.name}' not in registry"})

    # G7
    live_kwargs = _dataclass_kwargs(class_node)
    if live_kwargs is None:
        violations.append({"rule": "G7", "entry": label,
                           "msg": f"class {entry.class_name} is not @dataclass decorated"})
    else:
        for kw, expected in entry.decorator_kwargs.items():
            got = live_kwargs.get(kw)
            if got != expected:
                violations.append({"rule": "G7", "entry": label,
                                   "msg": f"decorator kwarg drift: registered {kw}={expected} live {kw}={got}"})
        for kw, got in live_kwargs.items():
            if kw not in entry.decorator_kwargs:
                violations.append({"rule": "G7", "entry": label,
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
        "gate": 280,
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
    lines.append("# Receiver dataclass schema (gate 280)")
    lines.append("")
    lines.append("Locks the orchestration-carrier dataclasses on the RECEIVING side of "
                 "the publish-reproduction subprocess contract — analog of gate 279's "
                 "sender-side `Job`.")
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
    print(f"[lit-faith-receiver-dataclass-schema] status=active "
          f"classes={classes} fields={fields_total} violations={n_viol}")
    return 0 if n_viol == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
