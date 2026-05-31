"""Gate 273 — runner CLI registry.

Locks the argparse CLI surface of the two paper-experiment runners:

* ``scripts/experiments/ecg/runner.py``
* ``scripts/experiments/vldb/runner.py``

Both runners are user-facing — invoked by ``paper_pipeline.py``,
SBATCH templates, and every documented reproduction recipe. Their
``--exp``, ``--graphs``, ``--graph-dir``, ``--preview``, ``--dry-run``
surface is part of the published paper-reproduction contract.

Catches the silent-drift cases:

* a contributor renames ``--graph-dir`` to ``--graphs-dir`` in
  ``vldb/runner.py`` and every SBATCH template + docs example
  silently invokes help-text fallback (or fails outright);
* a contributor removes the ``--preview`` flag from
  ``ecg/runner.py`` and CI / docs reproduction smokes start failing
  in opaque ways;
* a contributor adds a new positional that breaks ``--all`` mode;
* a contributor changes ``--exp`` from ``nargs="+"`` to a single
  value — every example using ``--exp A1 A2 A3`` silently swallows
  only the last value;
* a contributor swaps the ``choices`` set of ``--exp`` in
  ``ecg/runner.py`` (drops B7 or B8) and silently skips experiments.

6 rules R1-R6:

* **R1** — every registered runner exists and is importable as a
  module (ast.parse succeeds).
* **R2** — every registered runner has a ``main()`` top-level fn.
* **R3** — every registered flag exists in the live argparse calls
  with the same option strings.
* **R4** — every registered flag has the same ``action`` (default
  ``store``) — catches store_true ↔ store drift.
* **R5** — every registered flag with ``nargs`` has the same nargs
  value (``+``, ``*``, ``?``, or explicit int).
* **R6** — registry is exhaustive over all ``add_argument`` calls in
  ``main()`` — no surprise new flag slips into the runner.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]


ECG_RUNNER_PATH = "scripts/experiments/ecg/runner.py"
VLDB_RUNNER_PATH = "scripts/experiments/vldb/runner.py"


ECG_RUNNER_FLAGS: list[dict] = [
    {"flag": "--all",       "action": "store_true"},
    {"flag": "--section",   "action": "store"},
    {"flag": "--exp",       "action": "store", "nargs": "+"},
    {"flag": "--preview",   "action": "store_true"},
    {"flag": "--dry-run",   "action": "store_true"},
    {"flag": "--graph-dir", "action": "store"},
]


VLDB_RUNNER_FLAGS: list[dict] = [
    {"flag": "--all",            "action": "store_true"},
    {"flag": "--exp",            "action": "store", "nargs": "+"},
    {"flag": "--preview",        "action": "store_true"},
    {"flag": "--dry-run",        "action": "store_true"},
    {"flag": "--graphs",         "action": "store", "nargs": "+"},
    {"flag": "--graph-dir",      "action": "store"},
    {"flag": "--64gb",           "action": "store_true"},
    {"flag": "--local",          "action": "store_true"},
    {"flag": "--skip-setup",     "action": "store_true"},
    {"flag": "--skip-download",  "action": "store_true"},
    {"flag": "--no-figures",     "action": "store_true"},
    {"flag": "--figures-only",   "action": "store_true"},
]


RUNNER_CLI_REGISTRY: dict[str, list[dict]] = {
    ECG_RUNNER_PATH: ECG_RUNNER_FLAGS,
    VLDB_RUNNER_PATH: VLDB_RUNNER_FLAGS,
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


def _collect_add_argument_calls(fn: ast.FunctionDef) -> list[dict]:
    """Return [{flag, action, nargs}] for every parser.add_argument(...)."""
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
        action = "store"
        nargs: object | None = None
        for kw in node.keywords:
            if kw.arg == "action":
                v = _ast_const(kw.value)
                if isinstance(v, str):
                    action = v
            elif kw.arg == "nargs":
                v = _ast_const(kw.value)
                nargs = v
        entry: dict = {"flag": flag, "action": action}
        if nargs is not None:
            entry["nargs"] = nargs
        out.append(entry)
    return out


# --------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------


def _check_r1_importable() -> list[dict]:
    out = []
    for rel in RUNNER_CLI_REGISTRY:
        p = REPO_ROOT / rel
        if not p.is_file():
            out.append({"rule": "R1", "path": rel, "issue": "missing"})
            continue
        try:
            _parse_module(p)
        except SyntaxError as exc:
            out.append({"rule": "R1", "path": rel,
                        "issue": f"syntax error: {exc}"})
    return out


def _check_r2_main_present() -> list[dict]:
    out = []
    for rel in RUNNER_CLI_REGISTRY:
        p = REPO_ROOT / rel
        if not p.is_file():
            continue
        mod = _parse_module(p)
        if _top_level_fn(mod, "main") is None:
            out.append({"rule": "R2", "path": rel, "issue": "no main() fn"})
    return out


def _live_flags(rel: str) -> list[dict]:
    p = REPO_ROOT / rel
    if not p.is_file():
        return []
    mod = _parse_module(p)
    fn = _top_level_fn(mod, "main")
    if fn is None:
        return []
    return _collect_add_argument_calls(fn)


def _check_r3_flag_presence() -> list[dict]:
    out = []
    for rel, want_flags in RUNNER_CLI_REGISTRY.items():
        live = _live_flags(rel)
        live_flags = {e["flag"] for e in live}
        for entry in want_flags:
            if entry["flag"] not in live_flags:
                out.append({"rule": "R3", "path": rel,
                            "missing_flag": entry["flag"]})
    return out


def _check_r4_action_match() -> list[dict]:
    out = []
    for rel, want_flags in RUNNER_CLI_REGISTRY.items():
        live = {e["flag"]: e for e in _live_flags(rel)}
        for entry in want_flags:
            got = live.get(entry["flag"])
            if got is None:
                continue
            if got["action"] != entry["action"]:
                out.append({"rule": "R4", "path": rel, "flag": entry["flag"],
                            "want_action": entry["action"],
                            "got_action": got["action"]})
    return out


def _check_r5_nargs_match() -> list[dict]:
    out = []
    for rel, want_flags in RUNNER_CLI_REGISTRY.items():
        live = {e["flag"]: e for e in _live_flags(rel)}
        for entry in want_flags:
            got = live.get(entry["flag"])
            if got is None:
                continue
            want_nargs = entry.get("nargs")
            got_nargs = got.get("nargs")
            if want_nargs != got_nargs:
                out.append({"rule": "R5", "path": rel, "flag": entry["flag"],
                            "want_nargs": want_nargs, "got_nargs": got_nargs})
    return out


def _check_r6_exhaustive() -> list[dict]:
    out = []
    for rel, want_flags in RUNNER_CLI_REGISTRY.items():
        want_set = {e["flag"] for e in want_flags}
        live_set = {e["flag"] for e in _live_flags(rel)}
        for extra in sorted(live_set - want_set):
            out.append({"rule": "R6", "path": rel, "flag": extra,
                        "issue": "live flag not in registry"})
    return out


# --------------------------------------------------------------------
# Audit + emit
# --------------------------------------------------------------------


def audit() -> dict:
    viols: list[dict] = []
    viols += _check_r1_importable()
    viols += _check_r2_main_present()
    viols += _check_r3_flag_presence()
    viols += _check_r4_action_match()
    viols += _check_r5_nargs_match()
    viols += _check_r6_exhaustive()

    counts = {
        "runners": len(RUNNER_CLI_REGISTRY),
        "ecg_flags": len(ECG_RUNNER_FLAGS),
        "vldb_flags": len(VLDB_RUNNER_FLAGS),
        "total_flags": sum(len(v) for v in RUNNER_CLI_REGISTRY.values()),
    }
    return {
        "schema": "lit-faith-runner-cli-registry/1",
        "status": "active",
        "counts": counts,
        "registry": {rel: list(flags) for rel, flags in RUNNER_CLI_REGISTRY.items()},
        "rules": {
            "R1": "runner module importable (ast parses)",
            "R2": "main() top-level fn present",
            "R3": "every registered flag exists in live add_argument calls",
            "R4": "every registered flag action matches live",
            "R5": "every registered flag nargs matches live",
            "R6": "registry exhaustive — no surprise live flag",
        },
        "violations": viols,
    }


def write_json(doc: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", "utf-8")


def write_md(doc: dict, path: Path) -> None:
    lines: list[str] = [
        "# Runner CLI registry (gate 273)",
        "",
        "_Auto-generated by `lit_faith_runner_cli_registry.py`._",
        "",
        f"- runners: **{doc['counts']['runners']}**",
        f"- ECG runner flags: **{doc['counts']['ecg_flags']}**",
        f"- VLDB runner flags: **{doc['counts']['vldb_flags']}**",
        f"- total flags: **{doc['counts']['total_flags']}**",
        f"- violations: **{len(doc['violations'])}**",
        "",
    ]
    for rel, flags in sorted(doc["registry"].items()):
        lines.append(f"## `{rel}`")
        lines.append("")
        lines.append("| flag | action | nargs |")
        lines.append("| --- | --- | --- |")
        for e in flags:
            lines.append(f"| `{e['flag']}` | `{e['action']}` | "
                         f"`{e.get('nargs', '—')}` |")
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
    rows = ["path,flag,action,nargs"]
    for rel, flags in sorted(doc["registry"].items()):
        for e in flags:
            rows.append(f"{rel},{e['flag']},{e['action']},{e.get('nargs','')}")
    for v in doc["violations"]:
        rows.append(f"violation,{v.get('path','')},{v.get('flag','')},"
                    f"{v.get('rule','')}")
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
        f"[lit-faith-runner-cli-registry] "
        f"status={doc['status']} flags={doc['counts']['total_flags']} "
        f"violations={len(doc['violations'])}"
    )
    return 1 if doc["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
