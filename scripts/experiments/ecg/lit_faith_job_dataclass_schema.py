"""Gate 279 — Job dataclass schema registry.

Locks the ``Job`` dataclass shape in
``scripts/experiments/ecg/final_paper_run.py``. ``Job`` is the unit
of subprocess orchestration — every gate-277 sender (``run_profile``,
``make_proof_job``, ``make_roi_job``) emits ``Job`` instances and the
orchestrator pipeline serialises them to log paths, persists their
``out_dir`` artefacts, and propagates ``env`` + ``metadata`` into
every downstream stage of the paper-reproduction contract.

Gate 277 locked the SENDER side of subprocess invocations (--flag
literals in argv). Gate 278 locked the RECEIVER side (parse_args
calls in proof_matrix + roi_matrix). Gate 279 locks the
ORCHESTRATION-CARRIER side — the data structure that connects them
to the executor, the log archiver, the manifest writer, the
checkpoint walker, and every downstream stage. Together gates
277+278+279 fully lock the subprocess-orchestration contract from
job-definition through invocation through reception.

Catches the silent-drift cases gates 277+278 can't catch:

* Someone adds a new ``Job`` field (e.g. ``priority: int``) without
  updating the executor / manifest writer — every consumer using
  ``asdict(job)`` silently writes ``priority`` into manifest with no
  schema documentation, and unmarshalled-from-disk Jobs are missing
  the field. F6 catches.
* Someone changes ``log_path: Path`` → ``log_path: str`` (Path-vs-str
  silently mixes through pathlib + os.path operations; downstream
  consumers comparing log paths get False-positive duplicates). F4
  catches.
* Someone drops ``frozen=True`` from ``@dataclass`` (Jobs become
  mutable; long-running pipelines mutate Job.metadata in-place during
  iteration; later replay-from-log produces non-deterministic output).
  F7 catches.
* Someone changes ``metadata: dict[str, Any] = field(default_factory=dict)``
  to ``metadata: dict[str, Any] = {}`` (classic shared-mutable-default
  trap — all Jobs share one metadata dict; the first Job's metadata
  ends up on every subsequent Job). F5 catches.
* Someone renames ``out_dir`` → ``output_dir`` (every consumer
  reading ``job.out_dir`` AttributeErrors at runtime; gate 277's argv
  lock passes because ``--out-dir`` flag is unchanged). F3 catches.

7 rules F1-F7:

* **F1** — target module exists and ast.parses.
* **F2** — ``Job`` class is present at module top-level.
* **F3** — every registered field exists in live class body.
* **F4** — every registered field annotation string matches live.
* **F5** — every registered field default state matches: ``none``
  (no default), ``literal:<value>``, or ``factory:<call>``.
* **F6** — class exhaustive — no surprise live field slips in.
* **F7** — ``@dataclass(frozen=True)`` decorator preserved with
  ``frozen=True`` kwarg.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]


JOB_MODULE = "scripts/experiments/ecg/final_paper_run.py"
JOB_CLASS = "Job"


# Each entry: (field_name, annotation, default_descriptor)
# default_descriptor: "none" | "literal:<unparsed>" | "factory:<unparsed>"
JOB_FIELDS: list[dict] = [
    {"name": "job_id",   "annot": "str",             "default": "none"},
    {"name": "stage",    "annot": "str",             "default": "none"},
    {"name": "kind",     "annot": "str",             "default": "none"},
    {"name": "command",  "annot": "list[str]",       "default": "none"},
    {"name": "out_dir",  "annot": "Path",            "default": "none"},
    {"name": "log_path", "annot": "Path",            "default": "none"},
    {"name": "metadata", "annot": "dict[str, Any]",
        "default": "factory:field(default_factory=dict)"},
    {"name": "env",      "annot": "dict[str, str]",
        "default": "factory:field(default_factory=dict)"},
]


DATACLASS_DECORATOR_KWARGS: dict[str, str] = {
    "frozen": "True",
}


# --------------------------------------------------------------------
# AST helpers
# --------------------------------------------------------------------


def _parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text("utf-8"))


def _top_level_class(module: ast.Module, name: str) -> ast.ClassDef | None:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _default_descriptor(value: ast.expr | None) -> str:
    if value is None:
        return "none"
    src = ast.unparse(value)
    if isinstance(value, ast.Call):
        return f"factory:{src}"
    return f"literal:{src}"


def _live_fields(cls: ast.ClassDef) -> list[dict]:
    out: list[dict] = []
    for stmt in cls.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            out.append({
                "name":    stmt.target.id,
                "annot":   ast.unparse(stmt.annotation) if stmt.annotation else "",
                "default": _default_descriptor(stmt.value),
            })
    return out


def _dataclass_kwargs(cls: ast.ClassDef) -> dict[str, str] | None:
    """Return @dataclass kwargs if class is dataclass-decorated, else None."""
    for dec in cls.decorator_list:
        # @dataclass(frozen=True)
        if isinstance(dec, ast.Call):
            fn = dec.func
            if isinstance(fn, ast.Name) and fn.id == "dataclass":
                return {kw.arg: ast.unparse(kw.value)
                        for kw in dec.keywords if kw.arg}
            if isinstance(fn, ast.Attribute) and fn.attr == "dataclass":
                return {kw.arg: ast.unparse(kw.value)
                        for kw in dec.keywords if kw.arg}
        # @dataclass (no args)
        if isinstance(dec, ast.Name) and dec.id == "dataclass":
            return {}
        if isinstance(dec, ast.Attribute) and dec.attr == "dataclass":
            return {}
    return None


# --------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------


def _check_f1_module_parses() -> list[dict]:
    out = []
    p = REPO_ROOT / JOB_MODULE
    if not p.is_file():
        out.append({"rule": "F1", "path": JOB_MODULE, "issue": "missing"})
        return out
    try:
        _parse_module(p)
    except SyntaxError as exc:
        out.append({"rule": "F1", "path": JOB_MODULE,
                    "issue": f"syntax error: {exc}"})
    return out


def _check_f2_class_present() -> list[dict]:
    out = []
    p = REPO_ROOT / JOB_MODULE
    if not p.is_file():
        return out
    try:
        mod = _parse_module(p)
    except SyntaxError:
        return out
    if _top_level_class(mod, JOB_CLASS) is None:
        out.append({"rule": "F2", "path": JOB_MODULE,
                    "issue": f"class {JOB_CLASS} not present"})
    return out


def _live_class() -> ast.ClassDef | None:
    p = REPO_ROOT / JOB_MODULE
    if not p.is_file():
        return None
    try:
        mod = _parse_module(p)
    except SyntaxError:
        return None
    return _top_level_class(mod, JOB_CLASS)


def _check_f3_field_presence() -> list[dict]:
    out = []
    cls = _live_class()
    if cls is None:
        return out
    live = {f["name"]: f for f in _live_fields(cls)}
    for entry in JOB_FIELDS:
        if entry["name"] not in live:
            out.append({"rule": "F3", "field": entry["name"],
                        "issue": "registered field missing in live class"})
    return out


def _check_f4_annot_match() -> list[dict]:
    out = []
    cls = _live_class()
    if cls is None:
        return out
    live = {f["name"]: f for f in _live_fields(cls)}
    for entry in JOB_FIELDS:
        got = live.get(entry["name"])
        if got is None:
            continue
        if got["annot"] != entry["annot"]:
            out.append({"rule": "F4", "field": entry["name"],
                        "want_annot": entry["annot"],
                        "got_annot": got["annot"]})
    return out


def _check_f5_default_match() -> list[dict]:
    out = []
    cls = _live_class()
    if cls is None:
        return out
    live = {f["name"]: f for f in _live_fields(cls)}
    for entry in JOB_FIELDS:
        got = live.get(entry["name"])
        if got is None:
            continue
        if got["default"] != entry["default"]:
            out.append({"rule": "F5", "field": entry["name"],
                        "want_default": entry["default"],
                        "got_default": got["default"]})
    return out


def _check_f6_exhaustive() -> list[dict]:
    out = []
    cls = _live_class()
    if cls is None:
        return out
    want_set = {f["name"] for f in JOB_FIELDS}
    live_set = {f["name"] for f in _live_fields(cls)}
    for extra in sorted(live_set - want_set):
        out.append({"rule": "F6", "field": extra,
                    "issue": "live field not in registry"})
    return out


def _check_f7_decorator_kwargs() -> list[dict]:
    out = []
    cls = _live_class()
    if cls is None:
        return out
    kwargs = _dataclass_kwargs(cls)
    if kwargs is None:
        out.append({"rule": "F7", "issue": "class is not @dataclass-decorated"})
        return out
    for kw, want in DATACLASS_DECORATOR_KWARGS.items():
        got = kwargs.get(kw)
        if got != want:
            out.append({"rule": "F7", "kwarg": kw,
                        "want": want, "got": got})
    return out


# --------------------------------------------------------------------
# Audit + emit
# --------------------------------------------------------------------


def audit() -> dict:
    viols: list[dict] = []
    viols += _check_f1_module_parses()
    viols += _check_f2_class_present()
    viols += _check_f3_field_presence()
    viols += _check_f4_annot_match()
    viols += _check_f5_default_match()
    viols += _check_f6_exhaustive()
    viols += _check_f7_decorator_kwargs()

    counts = {
        "modules": 1,
        "classes": 1,
        "fields":  len(JOB_FIELDS),
        "decorator_kwargs": len(DATACLASS_DECORATOR_KWARGS),
    }
    return {
        "schema": "lit-faith-job-dataclass-schema/1",
        "status": "active",
        "counts": counts,
        "registry": {
            "module": JOB_MODULE,
            "class":  JOB_CLASS,
            "fields": list(JOB_FIELDS),
            "decorator_kwargs": dict(DATACLASS_DECORATOR_KWARGS),
        },
        "rules": {
            "F1": "module exists and ast-parses",
            "F2": "Job class present at module top level",
            "F3": "every registered field exists in live class body",
            "F4": "every registered field annotation matches live",
            "F5": "every registered field default state matches",
            "F6": "class exhaustive — no surprise live field",
            "F7": "@dataclass(frozen=True) decorator kwargs preserved",
        },
        "violations": viols,
    }


def write_json(doc: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", "utf-8")


def write_md(doc: dict, path: Path) -> None:
    lines: list[str] = [
        "# Job dataclass schema (gate 279)",
        "",
        "_Auto-generated by `lit_faith_job_dataclass_schema.py`._",
        "",
        f"- module: `{doc['registry']['module']}`",
        f"- class: `{doc['registry']['class']}`",
        f"- fields: **{doc['counts']['fields']}**",
        f"- @dataclass kwargs: **{doc['counts']['decorator_kwargs']}**",
        f"- violations: **{len(doc['violations'])}**",
        "",
        "## Fields",
        "",
        "| name | annotation | default |",
        "| --- | --- | --- |",
    ]
    for f in doc["registry"]["fields"]:
        lines.append(f"| `{f['name']}` | `{f['annot']}` | `{f['default']}` |")
    lines.append("")
    lines.append("## Decorator kwargs")
    lines.append("")
    lines.append("| kwarg | value |")
    lines.append("| --- | --- |")
    for k, v in sorted(doc["registry"]["decorator_kwargs"].items()):
        lines.append(f"| `{k}` | `{v}` |")
    lines.append("")
    if doc["violations"]:
        lines.append("## Violations")
        lines.append("")
        for v in doc["violations"]:
            lines.append(f"- {json.dumps(v, sort_keys=True)}")
    else:
        lines.append("## ✅ No violations")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", "utf-8")


def write_csv(doc: dict, path: Path) -> None:
    rows = ["kind,name,annot,default"]
    for f in doc["registry"]["fields"]:
        rows.append(f"field,{f['name']},{f['annot']},{f['default']}")
    for k, v in sorted(doc["registry"]["decorator_kwargs"].items()):
        rows.append(f"decorator_kwarg,{k},,{v}")
    for v in doc["violations"]:
        rows.append(f"violation,{v.get('field', v.get('kwarg', ''))},"
                    f"{v.get('issue', '')},"
                    f"{v.get('rule', '')}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", "utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--md-out", type=Path)
    ap.add_argument("--csv-out", type=Path)
    args = ap.parse_args(list(argv) if argv is not None else None)

    doc = audit()
    if args.json_out:
        write_json(doc, args.json_out)
    if args.md_out:
        write_md(doc, args.md_out)
    if args.csv_out:
        write_csv(doc, args.csv_out)

    print(
        f"[lit-faith-job-dataclass-schema] "
        f"status={doc['status']} "
        f"fields={doc['counts']['fields']} "
        f"violations={len(doc['violations'])}"
    )
    return 1 if doc["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
