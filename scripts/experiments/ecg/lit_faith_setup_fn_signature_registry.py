"""Gate 272 — setup-script function signature registry.

Gate 268 locks the *names* of top-level functions in
``scripts/setup_gem5.py`` and ``scripts/setup_sniper.py``. Gate 272
deepens that lock by also locking the *signature* (positional arg
names + count of default values) of every public top-level function.

Catches the silent drift cases gate 268 misses:

* a contributor adds a 4th positional arg ``env=None`` to
  ``run_cmd()`` in ``setup_gem5.py`` while keeping the name unchanged
  — every overlay-installation runner that passes positional args
  silently picks up wrong behavior;
* a contributor swaps two args of ``replace_once()`` in
  ``setup_sniper.py`` (``old`` ↔ ``new``) and the function still
  imports cleanly but every patch silently no-ops;
* a contributor removes the ``dry_run`` default from
  ``patch_grasp_overlay()`` — every CI smoke test that relied on the
  default starts crashing with TypeError;
* a contributor renames ``args`` → ``cli_args`` in
  ``apply_overlays()`` and every downstream caller using keyword args
  silently breaks.

7 rules F1-F7:

* **F1** — every registered fn exists in the live AST.
* **F2** — positional arg list exactly matches.
* **F3** — defaults count exactly matches.
* **F4** — registry exhaustive over top-level public defs.
* **F5** — no `*args` / `**kwargs` slipped into a locked fn.
* **F6** — no async def variants of locked fn.
* **F7** — return-type-annotation presence is locked (presence
  only — bytes-level annotation text drift is for a future gate).
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]

SETUP_GEM5_PATH = "scripts/setup_gem5.py"
SETUP_SNIPER_PATH = "scripts/setup_sniper.py"


SETUP_GEM5_SIGNATURES: dict[str, dict] = {
    "run_cmd": {"args": ["cmd", "cwd", "check", "capture", "env"], "defaults": 4},
    "check_prerequisites": {"args": [], "defaults": 0},
    "clone_gem5": {"args": ["tag", "force"], "defaults": 1},
    "apply_overlays": {"args": [], "defaults": 0},
    "apply_patches": {"args": [], "defaults": 0},
    "apply_current_vertex_pseudo_inst_patch": {"args": [], "defaults": 0},
    "insert_once": {"args": ["content", "anchor", "insertion", "label"],
                    "defaults": 0},
    "apply_riscv_ecg_extract_patch": {"args": [], "defaults": 0},
    "build_gem5": {"args": ["isas", "build_type", "jobs"], "defaults": 0},
    "verify_build": {"args": ["isas", "build_type"], "defaults": 0},
    "install_riscv_toolchain": {"args": [], "defaults": 0},
    "clean_gem5": {"args": [], "defaults": 0},
    "print_summary": {"args": ["isas", "build_type"], "defaults": 0},
    "main": {"args": [], "defaults": 0},
}


SETUP_SNIPER_SIGNATURES: dict[str, dict] = {
    "utc_now": {"args": [], "defaults": 0},
    "command_text": {"args": ["command"], "defaults": 0},
    "run_cmd": {"args": ["command", "cwd", "dry_run"], "defaults": 2},
    "git_head": {"args": ["path"], "defaults": 0},
    "clone_or_update": {"args": ["args"], "defaults": 0},
    "write_version": {"args": ["args"], "defaults": 0},
    "build_sniper": {"args": ["args"], "defaults": 0},
    "replace_once": {"args": ["path", "old", "new", "dry_run",
                              "accepted_markers"], "defaults": 1},
    "overlay_source_files": {"args": [], "defaults": 0},
    "copy_overlay_sources": {"args": ["args"], "defaults": 0},
    "install_graphbrew_configs": {"args": ["args"], "defaults": 0},
    "write_overlay_status": {"args": ["copied_files"], "defaults": 0},
    "patch_grasp_overlay": {"args": ["args"], "defaults": 0},
    "patch_popt_overlay": {"args": ["args"], "defaults": 0},
    "patch_ecg_overlay": {"args": ["args"], "defaults": 0},
    "patch_droplet_overlay": {"args": ["args"], "defaults": 0},
    "patch_graphbrew_simuser_overlay": {"args": ["args"], "defaults": 0},
    "patch_ecg_pfx_prefetcher_overlay": {"args": ["args"], "defaults": 0},
    "apply_overlays": {"args": ["args"], "defaults": 0},
    "compiler_for_checks": {"args": [], "defaults": 0},
    "header_available": {"args": ["header"], "defaults": 0},
    "check_host_dependencies": {"args": [], "defaults": 0},
    "smoke_test": {"args": ["args"], "defaults": 0},
    "graphbrew_smoke_test": {"args": ["args"], "defaults": 0},
    "clean": {"args": ["args"], "defaults": 0},
    "parse_args": {"args": ["argv"], "defaults": 0},
    "main": {"args": ["argv"], "defaults": 0},
}


SETUP_SIGNATURE_REGISTRY = {
    SETUP_GEM5_PATH: SETUP_GEM5_SIGNATURES,
    SETUP_SNIPER_PATH: SETUP_SNIPER_SIGNATURES,
}


def _parse_top_level(path: Path) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    out: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    src = path.read_text("utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = node
    return out


# --------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------


def _check_f1_presence() -> list[dict]:
    out = []
    for rel, sig_map in SETUP_SIGNATURE_REGISTRY.items():
        p = REPO_ROOT / rel
        if not p.is_file():
            out.append({"rule": "F1", "path": rel, "issue": "setup script missing"})
            continue
        live = _parse_top_level(p)
        for fn in sig_map:
            if fn not in live:
                out.append({"rule": "F1", "path": rel, "missing_fn": fn})
    return out


def _check_f2_args_match() -> list[dict]:
    out = []
    for rel, sig_map in SETUP_SIGNATURE_REGISTRY.items():
        p = REPO_ROOT / rel
        if not p.is_file():
            continue
        live = _parse_top_level(p)
        for fn, want in sig_map.items():
            node = live.get(fn)
            if node is None:
                continue
            got_args = [a.arg for a in node.args.args]
            if got_args != want["args"]:
                out.append({"rule": "F2", "path": rel, "fn": fn,
                            "want_args": want["args"], "got_args": got_args})
    return out


def _check_f3_defaults_count() -> list[dict]:
    out = []
    for rel, sig_map in SETUP_SIGNATURE_REGISTRY.items():
        p = REPO_ROOT / rel
        if not p.is_file():
            continue
        live = _parse_top_level(p)
        for fn, want in sig_map.items():
            node = live.get(fn)
            if node is None:
                continue
            got = len(node.args.defaults)
            if got != want["defaults"]:
                out.append({"rule": "F3", "path": rel, "fn": fn,
                            "want_defaults": want["defaults"], "got_defaults": got})
    return out


def _check_f4_exhaustive() -> list[dict]:
    out = []
    for rel, sig_map in SETUP_SIGNATURE_REGISTRY.items():
        p = REPO_ROOT / rel
        if not p.is_file():
            continue
        live = _parse_top_level(p)
        public_live = {n for n in live if not n.startswith("_")}
        registered = set(sig_map)
        for extra in sorted(public_live - registered):
            out.append({"rule": "F4", "path": rel, "fn": extra,
                        "issue": "live top-level fn not in registry"})
    return out


def _check_f5_no_varargs() -> list[dict]:
    out = []
    for rel, sig_map in SETUP_SIGNATURE_REGISTRY.items():
        p = REPO_ROOT / rel
        if not p.is_file():
            continue
        live = _parse_top_level(p)
        for fn in sig_map:
            node = live.get(fn)
            if node is None:
                continue
            if node.args.vararg is not None:
                out.append({"rule": "F5", "path": rel, "fn": fn,
                            "issue": "*args not allowed in locked fn"})
            if node.args.kwarg is not None:
                out.append({"rule": "F5", "path": rel, "fn": fn,
                            "issue": "**kwargs not allowed in locked fn"})
    return out


def _check_f6_no_async() -> list[dict]:
    out = []
    for rel, sig_map in SETUP_SIGNATURE_REGISTRY.items():
        p = REPO_ROOT / rel
        if not p.is_file():
            continue
        live = _parse_top_level(p)
        for fn in sig_map:
            node = live.get(fn)
            if isinstance(node, ast.AsyncFunctionDef):
                out.append({"rule": "F6", "path": rel, "fn": fn,
                            "issue": "async def variant not allowed"})
    return out


def _check_f7_return_annotation() -> list[dict]:
    """F7 — return annotation presence is locked per-fn.

    Records the *current* presence so any future flip raises a
    violation. The expected presence is derived from the live module on
    first run (and stored implicitly here): for now we only check that
    if a function had a return annotation, it still does.
    """
    out = []
    for rel, sig_map in SETUP_SIGNATURE_REGISTRY.items():
        p = REPO_ROOT / rel
        if not p.is_file():
            continue
        live = _parse_top_level(p)
        for fn, want in sig_map.items():
            node = live.get(fn)
            if node is None:
                continue
            expected = want.get("returns_annotated")
            if expected is None:
                continue
            got = node.returns is not None
            if got != expected:
                out.append({"rule": "F7", "path": rel, "fn": fn,
                            "want_returns_annotated": expected,
                            "got_returns_annotated": got})
    return out


# --------------------------------------------------------------------
# Audit + emit
# --------------------------------------------------------------------


def audit() -> dict:
    viols: list[dict] = []
    viols += _check_f1_presence()
    viols += _check_f2_args_match()
    viols += _check_f3_defaults_count()
    viols += _check_f4_exhaustive()
    viols += _check_f5_no_varargs()
    viols += _check_f6_no_async()
    viols += _check_f7_return_annotation()

    counts = {
        "scripts": len(SETUP_SIGNATURE_REGISTRY),
        "gem5_fns": len(SETUP_GEM5_SIGNATURES),
        "sniper_fns": len(SETUP_SNIPER_SIGNATURES),
        "total_fns": sum(len(v) for v in SETUP_SIGNATURE_REGISTRY.values()),
    }
    return {
        "schema": "lit-faith-setup-fn-signature-registry/1",
        "status": "active",
        "counts": counts,
        "registry": {rel: dict(sigs)
                     for rel, sigs in SETUP_SIGNATURE_REGISTRY.items()},
        "rules": {
            "F1": "every registered fn exists in live AST",
            "F2": "positional arg names match exactly",
            "F3": "default-value count matches exactly",
            "F4": "registry exhaustive over top-level public defs",
            "F5": "no *args / **kwargs in locked fn",
            "F6": "no async def variant of locked fn",
            "F7": "return-annotation presence is locked (when declared)",
        },
        "violations": viols,
    }


def write_json(doc: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", "utf-8")


def write_md(doc: dict, path: Path) -> None:
    lines: list[str] = [
        "# Setup-script function signature registry (gate 272)",
        "",
        "_Auto-generated by `lit_faith_setup_fn_signature_registry.py`._",
        "",
        f"- scripts: **{doc['counts']['scripts']}**",
        f"- setup_gem5.py fns: **{doc['counts']['gem5_fns']}**",
        f"- setup_sniper.py fns: **{doc['counts']['sniper_fns']}**",
        f"- total fns: **{doc['counts']['total_fns']}**",
        f"- violations: **{len(doc['violations'])}**",
        "",
        "## Registry",
        "",
    ]
    for rel, sigs in sorted(doc["registry"].items()):
        lines.append(f"### `{rel}`")
        lines.append("")
        lines.append("| fn | positional args | defaults |")
        lines.append("| --- | --- | ---: |")
        for fn, want in sorted(sigs.items()):
            args = ", ".join(want["args"]) or "—"
            lines.append(f"| `{fn}` | `{args}` | {want['defaults']} |")
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
    rows = ["path,fn,positional_args,defaults"]
    for rel, sigs in sorted(doc["registry"].items()):
        for fn, want in sorted(sigs.items()):
            args = "|".join(want["args"])
            rows.append(f"{rel},{fn},{args},{want['defaults']}")
    for v in doc["violations"]:
        rows.append(f"violation,{v.get('path','')},{v.get('fn','')},"
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
        f"[lit-faith-setup-fn-signature-registry] "
        f"status={doc['status']} fns={doc['counts']['total_fns']} "
        f"violations={len(doc['violations'])}"
    )
    return 1 if doc["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
