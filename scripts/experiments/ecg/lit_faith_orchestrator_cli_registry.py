"""Gate 274 — orchestrator CLI registry.

Locks the argparse surface of the two paper-orchestrators that sit
*above* the per-experiment runners locked in gate 273:

* ``scripts/experiments/ecg/paper_pipeline.py`` — drives full paper
  runs end-to-end (build → profile → aggregate → figures → copy).
* ``scripts/experiments/ecg/final_paper_run.py`` — drives a single
  ``final_*`` profile (multi-stage job expansion + execution).

These are the entry points every SBATCH template and every
documented reproduction recipe shells out to. Their argparse
surface IS the publish-reproduction contract.

Extends gate 273 (which covers ``ecg/runner.py`` + ``vldb/runner.py``)
by:

* Recognising ``argparse.BooleanOptionalAction`` (used by
  ``--resume`` / ``--stop-on-error`` in ``final_paper_run.py``).
* Allowing the locked fn to be ``parse_args`` (both orchestrators
  use a top-level ``parse_args(argv)`` instead of inlining argparse
  into ``main()``).
* Locking ``required=True`` flags as a distinct shape (catches
  drift where a flag silently becomes optional).

7 rules O1-O7 (orchestrator):

* **O1** — every registered orchestrator module ast-parses cleanly.
* **O2** — every orchestrator has the locked ``fn_name`` top-level fn.
* **O3** — every registered flag exists in the live ``add_argument``
  calls with the same option string.
* **O4** — every registered flag has the same ``action``.
* **O5** — every registered flag with ``nargs`` has the same nargs.
* **O6** — registry is exhaustive over all live ``add_argument``
  calls in the locked fn.
* **O7** — every registered flag's ``required`` shape matches live
  (locked flags stay optional, locked-required stays required).
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]


PAPER_PIPELINE_PATH = "scripts/experiments/ecg/paper_pipeline.py"
FINAL_PAPER_RUN_PATH = "scripts/experiments/ecg/final_paper_run.py"


PAPER_PIPELINE_FLAGS: list[dict] = [
    {"flag": "--profiles",              "action": "store", "nargs": "+"},
    {"flag": "--run-root",              "action": "store"},
    {"flag": "--input-run-dirs",        "action": "store", "nargs": "+"},
    {"flag": "--input-run-glob",        "action": "store", "nargs": "+"},
    {"flag": "--input-csv",             "action": "store", "nargs": "+"},
    {"flag": "--input-csv-glob",        "action": "store", "nargs": "+"},
    {"flag": "--skip-run",              "action": "store_true"},
    {"flag": "--dry-run",               "action": "store_true"},
    {"flag": "--no-build",              "action": "store_true"},
    {"flag": "--allow-missing-graphs",  "action": "store_true"},
    {"flag": "--force",                 "action": "store_true"},
    {"flag": "--no-stop-on-error",      "action": "store_false"},
    {"flag": "--copy-to-paper",         "action": "store_true"},
    {"flag": "--skip-literature-gate",  "action": "store_true"},
]


FINAL_PAPER_RUN_FLAGS: list[dict] = [
    {"flag": "--manifest",                  "action": "store"},
    {"flag": "--profile",                   "action": "store", "nargs": "+"},
    {"flag": "--run-dir",                   "action": "store"},
    {"flag": "--graph-dir",                 "action": "store"},
    {"flag": "--only",                      "action": "store", "nargs": "+"},
    {"flag": "--skip",                      "action": "store", "nargs": "+"},
    {"flag": "--graph",                     "action": "store", "nargs": "+"},
    {"flag": "--benchmark",                 "action": "store", "nargs": "+"},
    {"flag": "--policy",                    "action": "store", "nargs": "+"},
    {"flag": "--job",                       "action": "store", "nargs": "+"},
    {"flag": "--from-job",                  "action": "store"},
    {"flag": "--limit",                     "action": "store"},
    {"flag": "--list",                      "action": "store_true"},
    {"flag": "--status",                    "action": "store_true"},
    {"flag": "--check-graphs",              "action": "store_true"},
    {"flag": "--dry-run",                   "action": "store_true"},
    {"flag": "--resume",                    "action": "BooleanOptionalAction"},
    {"flag": "--force",                     "action": "store_true"},
    {"flag": "--skip-failed",               "action": "store_true"},
    {"flag": "--stop-on-error",             "action": "BooleanOptionalAction"},
    {"flag": "--no-build",                  "action": "store_true"},
    {"flag": "--allow-missing-graphs",      "action": "store_true"},
    {"flag": "--skip-validation-gate",      "action": "store_true"},
    {"flag": "--require-validation-gate",   "action": "store_true"},
    {"flag": "--literature-gate-root",      "action": "store"},
    {"flag": "--literature-gate-subdir",    "action": "store"},
    {"flag": "--require-literature-gate",   "action": "store_true"},
    {"flag": "--skip-literature-gate",      "action": "store_true"},
    {"flag": "--lock-path",                 "action": "store"},
]


ORCHESTRATOR_CLI_REGISTRY: dict[str, dict] = {
    PAPER_PIPELINE_PATH: {
        "fn_name": "parse_args",
        "flags":   PAPER_PIPELINE_FLAGS,
    },
    FINAL_PAPER_RUN_PATH: {
        "fn_name": "parse_args",
        "flags":   FINAL_PAPER_RUN_FLAGS,
    },
}


# --------------------------------------------------------------------
# AST helpers
# --------------------------------------------------------------------


def _parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text("utf-8"))


def _top_level_fn(module: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == name:
            return node  # type: ignore[return-value]
    return None


def _ast_const(node: ast.AST) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    return None


def _action_text(node: ast.AST) -> str | None:
    """Recognise both string-action and BooleanOptionalAction attr."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    # argparse.BooleanOptionalAction
    if isinstance(node, ast.Attribute) and node.attr == "BooleanOptionalAction":
        return "BooleanOptionalAction"
    if isinstance(node, ast.Name) and node.id == "BooleanOptionalAction":
        return "BooleanOptionalAction"
    return None


def _collect_add_argument_calls(fn: ast.FunctionDef) -> list[dict]:
    out: list[dict] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        if not node.args:
            continue
        first = node.args[0]
        flag = _ast_const(first)
        if not isinstance(flag, str) or not flag.startswith("-"):
            continue
        action: str = "store"
        nargs: object | None = None
        required: bool = False
        for kw in node.keywords:
            if kw.arg == "action":
                v = _action_text(kw.value)
                if v is not None:
                    action = v
            elif kw.arg == "nargs":
                v = _ast_const(kw.value)
                nargs = v
            elif kw.arg == "required":
                v = _ast_const(kw.value)
                if isinstance(v, bool):
                    required = v
        entry: dict = {"flag": flag, "action": action}
        if nargs is not None:
            entry["nargs"] = nargs
        if required:
            entry["required"] = True
        out.append(entry)
    return out


def _live_entries(rel: str) -> list[dict]:
    p = REPO_ROOT / rel
    if not p.is_file():
        return []
    spec = ORCHESTRATOR_CLI_REGISTRY[rel]
    mod = _parse_module(p)
    fn = _top_level_fn(mod, spec["fn_name"])
    if fn is None:
        return []
    return _collect_add_argument_calls(fn)


# --------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------


def _check_o1_importable() -> list[dict]:
    out = []
    for rel in ORCHESTRATOR_CLI_REGISTRY:
        p = REPO_ROOT / rel
        if not p.is_file():
            out.append({"rule": "O1", "path": rel, "issue": "missing"})
            continue
        try:
            _parse_module(p)
        except SyntaxError as exc:
            out.append({"rule": "O1", "path": rel,
                        "issue": f"syntax error: {exc}"})
    return out


def _check_o2_fn_present() -> list[dict]:
    out = []
    for rel, spec in ORCHESTRATOR_CLI_REGISTRY.items():
        p = REPO_ROOT / rel
        if not p.is_file():
            continue
        mod = _parse_module(p)
        if _top_level_fn(mod, spec["fn_name"]) is None:
            out.append({"rule": "O2", "path": rel,
                        "issue": f"no top-level {spec['fn_name']}() fn"})
    return out


def _check_o3_flag_presence() -> list[dict]:
    out = []
    for rel, spec in ORCHESTRATOR_CLI_REGISTRY.items():
        live_flags = {e["flag"] for e in _live_entries(rel)}
        for entry in spec["flags"]:
            if entry["flag"] not in live_flags:
                out.append({"rule": "O3", "path": rel,
                            "missing_flag": entry["flag"]})
    return out


def _check_o4_action_match() -> list[dict]:
    out = []
    for rel, spec in ORCHESTRATOR_CLI_REGISTRY.items():
        live = {e["flag"]: e for e in _live_entries(rel)}
        for entry in spec["flags"]:
            got = live.get(entry["flag"])
            if got is None:
                continue
            if got["action"] != entry["action"]:
                out.append({"rule": "O4", "path": rel, "flag": entry["flag"],
                            "want_action": entry["action"],
                            "got_action": got["action"]})
    return out


def _check_o5_nargs_match() -> list[dict]:
    out = []
    for rel, spec in ORCHESTRATOR_CLI_REGISTRY.items():
        live = {e["flag"]: e for e in _live_entries(rel)}
        for entry in spec["flags"]:
            got = live.get(entry["flag"])
            if got is None:
                continue
            want_nargs = entry.get("nargs")
            got_nargs = got.get("nargs")
            if want_nargs != got_nargs:
                out.append({"rule": "O5", "path": rel, "flag": entry["flag"],
                            "want_nargs": want_nargs, "got_nargs": got_nargs})
    return out


def _check_o6_exhaustive() -> list[dict]:
    out = []
    for rel, spec in ORCHESTRATOR_CLI_REGISTRY.items():
        want_set = {e["flag"] for e in spec["flags"]}
        live_set = {e["flag"] for e in _live_entries(rel)}
        for extra in sorted(live_set - want_set):
            out.append({"rule": "O6", "path": rel, "flag": extra,
                        "issue": "live flag not in registry"})
    return out


def _check_o7_required_match() -> list[dict]:
    out = []
    for rel, spec in ORCHESTRATOR_CLI_REGISTRY.items():
        live = {e["flag"]: e for e in _live_entries(rel)}
        for entry in spec["flags"]:
            got = live.get(entry["flag"])
            if got is None:
                continue
            want_required = bool(entry.get("required", False))
            got_required = bool(got.get("required", False))
            if want_required != got_required:
                out.append({"rule": "O7", "path": rel, "flag": entry["flag"],
                            "want_required": want_required,
                            "got_required": got_required})
    return out


# --------------------------------------------------------------------
# Audit + emit
# --------------------------------------------------------------------


def audit() -> dict:
    viols: list[dict] = []
    viols += _check_o1_importable()
    viols += _check_o2_fn_present()
    viols += _check_o3_flag_presence()
    viols += _check_o4_action_match()
    viols += _check_o5_nargs_match()
    viols += _check_o6_exhaustive()
    viols += _check_o7_required_match()

    counts = {
        "orchestrators":         len(ORCHESTRATOR_CLI_REGISTRY),
        "paper_pipeline_flags":  len(PAPER_PIPELINE_FLAGS),
        "final_paper_run_flags": len(FINAL_PAPER_RUN_FLAGS),
        "total_flags": sum(len(spec["flags"])
                           for spec in ORCHESTRATOR_CLI_REGISTRY.values()),
    }
    return {
        "schema": "lit-faith-orchestrator-cli-registry/1",
        "status": "active",
        "counts": counts,
        "registry": {
            rel: {"fn_name": spec["fn_name"], "flags": list(spec["flags"])}
            for rel, spec in ORCHESTRATOR_CLI_REGISTRY.items()
        },
        "rules": {
            "O1": "orchestrator module ast-parses cleanly",
            "O2": "locked fn_name top-level fn present",
            "O3": "every registered flag exists in live add_argument calls",
            "O4": "every registered flag action matches live",
            "O5": "every registered flag nargs matches live",
            "O6": "registry exhaustive — no surprise live flag",
            "O7": "every registered flag required-shape matches live",
        },
        "violations": viols,
    }


def write_json(doc: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", "utf-8")


def write_md(doc: dict, path: Path) -> None:
    lines: list[str] = [
        "# Orchestrator CLI registry (gate 274)",
        "",
        "_Auto-generated by `lit_faith_orchestrator_cli_registry.py`._",
        "",
        f"- orchestrators: **{doc['counts']['orchestrators']}**",
        f"- paper_pipeline.py flags: **{doc['counts']['paper_pipeline_flags']}**",
        f"- final_paper_run.py flags: **{doc['counts']['final_paper_run_flags']}**",
        f"- total flags: **{doc['counts']['total_flags']}**",
        f"- violations: **{len(doc['violations'])}**",
        "",
    ]
    for rel, spec in sorted(doc["registry"].items()):
        lines.append(f"## `{rel}` (`{spec['fn_name']}()`)")
        lines.append("")
        lines.append("| flag | action | nargs | required |")
        lines.append("| --- | --- | --- | --- |")
        for e in spec["flags"]:
            lines.append(
                f"| `{e['flag']}` | `{e['action']}` | "
                f"`{e.get('nargs', '—')}` | "
                f"`{e.get('required', False)}` |"
            )
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
    rows = ["path,fn_name,flag,action,nargs,required"]
    for rel, spec in sorted(doc["registry"].items()):
        for e in spec["flags"]:
            rows.append(f"{rel},{spec['fn_name']},{e['flag']},{e['action']},"
                        f"{e.get('nargs','')},{e.get('required', False)}")
    for v in doc["violations"]:
        rows.append(f"violation,,{v.get('path','')},{v.get('flag','')},"
                    f"{v.get('rule','')},")
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
        f"[lit-faith-orchestrator-cli-registry] "
        f"status={doc['status']} flags={doc['counts']['total_flags']} "
        f"violations={len(doc['violations'])}"
    )
    return 1 if doc["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
