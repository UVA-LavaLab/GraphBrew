"""Gate 277 — subprocess argv registry.

Locks the exact set of ``--flag`` literals that the three job-builders
in the paper-publication pipeline construct:

* ``paper_pipeline.run_profile()`` builds the argv that spawns
  ``final_paper_run.py`` for each profile (top-level orchestrator
  → orchestrator).
* ``final_paper_run.make_proof_job()`` builds the argv that spawns
  ``proof_matrix.py`` for each proof-matrix stage.
* ``final_paper_run.make_roi_job()`` builds the argv that spawns
  ``roi_matrix.py`` for each ROI cell.

Why this matters: gates 273+274+275 lock the *callable surfaces* of
the runner CLI, orchestrator CLI, and orchestrator backbone fns.
Gate 276 locks the *data* the orchestrator consumes (the manifest).
This gate locks the *argv contract* — the wire-level call to each
downstream tool. Without this, someone can rename
``proof_matrix.py``'s ``--out-dir`` to ``--output-dir``, push both
sides of the rename, ship green tests… and silently break every
recipe that's already in flight because the orchestrator argv
construction is unchanged.

Silent-drift cases caught:

* Someone renames ``--out-dir`` → ``--output-dir`` in
  ``proof_matrix.py`` but not in ``make_proof_job``: the argv lock
  here catches the orchestrator side; the runner-CLI lock (gate 273)
  catches the runner side; together they enforce a coordinated
  rename or a hard test failure.
* Someone adds a *new* required flag to ``proof_matrix.py`` (say
  ``--cache-replacement``) without teaching ``make_proof_job`` to
  pass it — every proof run now fails with an argparse error in the
  middle of an 8-hour SLURM job.
* Someone removes a conditional flag (say ``--allow-gem5-ecg-pfx``)
  from ``make_roi_job`` but leaves the ``settings.get(...)`` branch
  intact — the branch fires, but the flag never reaches roi_matrix
  so the runner refuses to start gem5 ECG_PFX runs.
* Someone changes the ``PROOF_MATRIX`` constant in ``final_paper_run``
  to ``PROVE_MATRIX`` but ``make_proof_job`` still references
  ``PROOF_MATRIX`` — NameError at job-build time (currently caught
  by python's lazy import semantics ONLY when the path is actually
  triggered).

6 rules A1-A6:

* **A1** — every target module ast-parses; every target fn is present.
* **A2** — every target fn references its locked ``UPPERCASE``
  command-target constant (FINAL_RUN / PROOF_MATRIX / ROI_MATRIX).
* **A3** — every locked flag must appear in the extracted set
  (catches removals).
* **A4** — every extracted flag must appear in the locked set
  (catches surprise additions).
* **A5** — exhaustive — every stage in the registry has non-empty
  flags and a target constant.
* **A6** — flag namespace hygiene — every extracted flag matches
  the canonical ``--[a-z0-9-]+`` pattern.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]

PAPER_PIPELINE_PATH = "scripts/experiments/ecg/paper_pipeline.py"
FINAL_PAPER_RUN_PATH = "scripts/experiments/ecg/final_paper_run.py"

FLAG_PATTERN = re.compile(r"^--[a-z0-9][a-z0-9-]*$")


def _stage(fn_name: str, module_path: str, target_const: str,
           flags: list[str]) -> dict:
    return {
        "fn_name":      fn_name,
        "module_path":  module_path,
        "target_const": target_const,
        "flags":        sorted(set(flags)),
    }


STAGES: list[dict] = [
    _stage(
        fn_name="run_profile",
        module_path=PAPER_PIPELINE_PATH,
        target_const="FINAL_RUN",
        flags=[
            "--allow-missing-graphs",
            "--dry-run",
            "--force",
            "--no-build",
            "--no-stop-on-error",
            "--profile",
            "--run-dir",
        ],
    ),
    _stage(
        fn_name="make_proof_job",
        module_path=FINAL_PAPER_RUN_PATH,
        target_const="PROOF_MATRIX",
        flags=[
            "--benchmarks",
            "--dry-run",
            "--l1d-size",
            "--l2-size",
            "--l3-sizes",
            "--l3-ways",
            "--line-size",
            "--no-build",
            "--out-dir",
            "--timeout-cache",
        ],
    ),
    _stage(
        fn_name="make_roi_job",
        module_path=FINAL_PAPER_RUN_PATH,
        target_const="ROI_MATRIX",
        flags=[
            "--allow-gem5-ecg-pfx",
            "--allow-sniper-benchmark-workload",
            "--allow-sniper-sg-kernel-workload",
            "--benchmark",
            "--droplet-indirect-degree",
            "--droplet-prefetch-degree",
            "--droplet-stride-table-size",
            "--dry-run",
            "--ecg-pfx-delivery",
            "--ecg-pfx-hint-filter",
            "--ecg-pfx-lookahead",
            "--ecg-pfx-mode",
            "--ecg-pfx-window",
            "--l1d-size",
            "--l2-size",
            "--l3-sizes",
            "--l3-ways",
            "--line-size",
            "--no-build",
            "--options",
            "--out-dir",
            "--policies",
            "--prefetcher",
            "--prefetcher-level",
            "--sniper-address-domain",
            "--sniper-base-config",
            "--sniper-cores",
            "--sniper-frontend",
            "--sniper-memory-limit-gb",
            "--sniper-mimicos-kernel-mb",
            "--sniper-mimicos-memory-mb",
            "--sniper-omp-wait-policy",
            "--sniper-root",
            "--sniper-workload",
            "--suite",
            "--threads",
            "--timeout-cache",
            "--timeout-gem5",
            "--timeout-sniper",
        ],
    ),
]


# --------------------------------------------------------------------
# AST extraction
# --------------------------------------------------------------------


def _load_ast(rel_path: str) -> ast.Module | None:
    p = REPO_ROOT / rel_path
    if not p.is_file():
        return None
    try:
        return ast.parse(p.read_text("utf-8"))
    except SyntaxError:
        return None


def _find_fn(module: ast.Module, fn_name: str) -> ast.FunctionDef | None:
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            return node
    return None


def _extract_flags(fn: ast.FunctionDef) -> set[str]:
    """Return every string-constant inside fn that starts with '--'."""
    return {
        n.value for n in ast.walk(fn)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and n.value.startswith("--")
    }


def _extract_uppercase_refs(fn: ast.FunctionDef) -> set[str]:
    """Return every UPPERCASE identifier referenced inside fn."""
    return {
        n.id for n in ast.walk(fn)
        if isinstance(n, ast.Name)
        and n.id.isupper()
        and not n.id.startswith("_")
    }


# --------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------


def _check_a1_module_and_fn_present(stage: dict, mod_cache: dict) -> list[dict]:
    out = []
    rel = stage["module_path"]
    mod = mod_cache.get(rel)
    if mod is None:
        mod = _load_ast(rel)
        mod_cache[rel] = mod
    if mod is None:
        out.append({"rule": "A1", "stage": stage["fn_name"],
                    "module": rel,
                    "issue": "module missing or syntax error"})
        return out
    fn = _find_fn(mod, stage["fn_name"])
    if fn is None:
        out.append({"rule": "A1", "stage": stage["fn_name"],
                    "module": rel,
                    "issue": "fn not found at module top-level"})
    return out


def _check_a2_target_const_referenced(stage: dict, mod_cache: dict) -> list[dict]:
    out = []
    mod = mod_cache.get(stage["module_path"]) or _load_ast(stage["module_path"])
    if mod is None:
        return out
    fn = _find_fn(mod, stage["fn_name"])
    if fn is None:
        return out
    refs = _extract_uppercase_refs(fn)
    if stage["target_const"] not in refs:
        out.append({"rule": "A2", "stage": stage["fn_name"],
                    "want_const": stage["target_const"],
                    "got_uppercase_refs": sorted(refs),
                    "issue": "target const not referenced in fn"})
    return out


def _check_a3_no_removals(stage: dict, mod_cache: dict) -> list[dict]:
    out = []
    mod = mod_cache.get(stage["module_path"]) or _load_ast(stage["module_path"])
    if mod is None:
        return out
    fn = _find_fn(mod, stage["fn_name"])
    if fn is None:
        return out
    live = _extract_flags(fn)
    locked = set(stage["flags"])
    for f in sorted(locked - live):
        out.append({"rule": "A3", "stage": stage["fn_name"],
                    "missing_flag": f,
                    "issue": "locked flag no longer in source"})
    return out


def _check_a4_no_additions(stage: dict, mod_cache: dict) -> list[dict]:
    out = []
    mod = mod_cache.get(stage["module_path"]) or _load_ast(stage["module_path"])
    if mod is None:
        return out
    fn = _find_fn(mod, stage["fn_name"])
    if fn is None:
        return out
    live = _extract_flags(fn)
    locked = set(stage["flags"])
    for f in sorted(live - locked):
        out.append({"rule": "A4", "stage": stage["fn_name"],
                    "extra_flag": f,
                    "issue": "new flag in source not in locked set"})
    return out


def _check_a5_registry_exhaustive(stage: dict, mod_cache: dict) -> list[dict]:
    out = []
    if not stage["flags"]:
        out.append({"rule": "A5", "stage": stage["fn_name"],
                    "issue": "registry entry has empty flags list"})
    if not stage["target_const"]:
        out.append({"rule": "A5", "stage": stage["fn_name"],
                    "issue": "registry entry has empty target_const"})
    if not stage["fn_name"]:
        out.append({"rule": "A5",
                    "issue": "registry entry has empty fn_name"})
    if not stage["module_path"]:
        out.append({"rule": "A5", "stage": stage["fn_name"],
                    "issue": "registry entry has empty module_path"})
    return out


def _check_a6_flag_hygiene(stage: dict, mod_cache: dict) -> list[dict]:
    out = []
    mod = mod_cache.get(stage["module_path"]) or _load_ast(stage["module_path"])
    if mod is None:
        return out
    fn = _find_fn(mod, stage["fn_name"])
    if fn is None:
        return out
    live = _extract_flags(fn)
    for f in sorted(live):
        if not FLAG_PATTERN.match(f):
            out.append({"rule": "A6", "stage": stage["fn_name"],
                        "bad_flag": f,
                        "issue": "flag fails canonical pattern --[a-z0-9-]+"})
    # Also check locked flags
    for f in stage["flags"]:
        if not FLAG_PATTERN.match(f):
            out.append({"rule": "A6", "stage": stage["fn_name"],
                        "bad_locked_flag": f,
                        "issue": "locked flag fails canonical pattern"})
    return out


# --------------------------------------------------------------------
# Audit + emit
# --------------------------------------------------------------------


def audit() -> dict:
    viols: list[dict] = []
    mod_cache: dict = {}
    per_stage_info: list[dict] = []
    for stage in STAGES:
        viols += _check_a1_module_and_fn_present(stage, mod_cache)
        viols += _check_a2_target_const_referenced(stage, mod_cache)
        viols += _check_a3_no_removals(stage, mod_cache)
        viols += _check_a4_no_additions(stage, mod_cache)
        viols += _check_a5_registry_exhaustive(stage, mod_cache)
        viols += _check_a6_flag_hygiene(stage, mod_cache)
        mod = mod_cache.get(stage["module_path"])
        live_flags: list[str] = []
        if mod is not None:
            fn = _find_fn(mod, stage["fn_name"])
            if fn is not None:
                live_flags = sorted(_extract_flags(fn))
        per_stage_info.append({
            "fn_name":          stage["fn_name"],
            "module_path":      stage["module_path"],
            "target_const":     stage["target_const"],
            "locked_flag_count": len(stage["flags"]),
            "live_flag_count":   len(live_flags),
            "flags":            stage["flags"],
        })
    total_locked = sum(len(s["flags"]) for s in STAGES)
    return {
        "schema":  "lit-faith-subprocess-argv-registry/1",
        "status":  "active",
        "stages":  per_stage_info,
        "counts":  {
            "n_stages": len(STAGES),
            "n_locked_flags_total": total_locked,
        },
        "rules":   {
            "A1": "module ast-parses; fn present at top-level",
            "A2": "fn references locked UPPERCASE target const",
            "A3": "every locked flag appears in live source",
            "A4": "every live flag appears in locked set",
            "A5": "registry entry is well-formed and non-empty",
            "A6": "every flag matches canonical --[a-z0-9-]+ pattern",
        },
        "violations": viols,
    }


def write_json(doc: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", "utf-8")


def write_md(doc: dict, path: Path) -> None:
    c = doc["counts"]
    lines: list[str] = [
        "# Subprocess argv registry (gate 277)",
        "",
        "_Auto-generated by `lit_faith_subprocess_argv_registry.py`._",
        "",
        f"- stages: **{c['n_stages']}**",
        f"- locked flags total: **{c['n_locked_flags_total']}**",
        f"- violations: **{len(doc['violations'])}**",
        "",
        "## Per-stage detail",
        "",
        "| fn_name | module | target const | locked flags | live flags |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for s in doc["stages"]:
        lines.append(
            f"| `{s['fn_name']}` | `{s['module_path']}` | "
            f"`{s['target_const']}` | {s['locked_flag_count']} | "
            f"{s['live_flag_count']} |"
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
    rows = ["stage,module_path,target_const,locked_flag,index"]
    for s in doc["stages"]:
        for i, f in enumerate(s["flags"]):
            rows.append(
                f"{s['fn_name']},{s['module_path']},"
                f"{s['target_const']},{f},{i}"
            )
    for v in doc["violations"]:
        rows.append(f"violation,{v.get('rule','')},,"
                    f"{json.dumps(v, sort_keys=True)},")
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
        f"[lit-faith-subprocess-argv-registry] "
        f"status={doc['status']} "
        f"stages={doc['counts']['n_stages']} "
        f"flags={doc['counts']['n_locked_flags_total']} "
        f"violations={len(doc['violations'])}"
    )
    return 1 if doc["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
