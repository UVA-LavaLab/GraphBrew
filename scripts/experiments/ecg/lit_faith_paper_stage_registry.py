"""Gate 275 — paper-pipeline stage registry.

Locks the *public* top-level function surface of the two paper
orchestrators whose CLI was locked in gates 273 (runners) + 274
(orchestrator CLI):

* ``scripts/experiments/ecg/paper_pipeline.py`` — 50 public top-level
  fns spanning the full build → profile → aggregate → figures → copy
  stages (``run_profile``, ``collect_csvs``, ``summarize_roi``,
  ``charged_overhead``, ``plot_metric_by_policy``, ``write_latex_table``,
  ``generate_outputs``, ``main``).
* ``scripts/experiments/ecg/final_paper_run.py`` — 37 public top-level
  fns covering manifest load, job expansion, gate validation, run-lock,
  execution, status, and per-job persistence (``load_manifest``,
  ``expand_jobs``, ``make_proof_job``, ``make_roi_job``,
  ``validate_gate``, ``validate_literature_gate``, ``run_job``,
  ``run_lock``, ``main``).

Why this matters: the CLI gates (273+274) lock the *entry points* of
the publish-reproduction pipeline. This gate locks the *internal
backbone* — every helper that every documented call path actually
shells through. Silent drift in any of these signatures (rename,
arg reorder, return-type flip from ``int`` to ``bool``, ``main``
becoming ``async``) would let an orchestrator continue to "run" while
silently producing wrong outputs or skipping stages.

7 rules S1-S7 (Stage registry):

* **S1** — every registered orchestrator module ast-parses cleanly.
* **S2** — every registered fn is a top-level fn of the locked kind
  (sync vs async).
* **S3** — every registered fn's positional arg list matches live
  (names AND order).
* **S4** — every registered fn's positional-default presence vector
  matches live (which args have defaults — value irrelevant).
* **S5** — every registered fn's return annotation matches live
  (stringified, ``ast.unparse`` form).
* **S6** — every registered fn's ``*args`` / ``**kwargs`` / keyword-only
  shape matches live.
* **S7** — registry is exhaustive — every public top-level fn (no
  leading underscore) appears in the registry.
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


def _entry(
    name: str,
    args: list[str],
    args_has_default: list[bool],
    returns: str | None,
    *,
    kwonly: list[str] | None = None,
    kwonly_has_default: list[bool] | None = None,
    vararg: str | None = None,
    kwarg: str | None = None,
    is_async: bool = False,
) -> dict:
    return {
        "name": name,
        "args": args,
        "args_has_default": args_has_default,
        "kwonly": kwonly or [],
        "kwonly_has_default": kwonly_has_default or [],
        "vararg": vararg,
        "kwarg": kwarg,
        "returns": returns,
        "is_async": is_async,
    }


PAPER_PIPELINE_STAGES: list[dict] = [
    _entry("utc_now", [], [], "str"),
    _entry("now_tag", [], [], "str"),
    _entry("resolve_path", ["path_text"], [False], "Path"),
    _entry("command_text", ["command"], [False], "str"),
    _entry("read_csv", ["path"], [False], "list[dict[str, Any]]"),
    _entry("write_csv", ["path", "rows"], [False, False], "None"),
    _entry("as_float", ["value"], [False], "float | None"),
    _entry("parse_size_bytes", ["value"], [False], "float | None"),
    _entry("pct_delta", ["value", "baseline"], [False, False], "float | None"),
    _entry("safe_ratio", ["numerator", "denominator"], [False, False], "float | None"),
    _entry("thread_count", ["row"], [False], "int | None"),
    _entry("metric_direction", ["value", "reference", "tolerance"], [False, False, True], "str"),
    _entry("avg", ["values"], [False], "float | None"),
    _entry("geo_mean", ["values"], [False], "float | None"),
    _entry("policy_sort_key", ["policy"], [False], "tuple[int, str]"),
    _entry("benchmark_sort_key", ["benchmark"], [False], "tuple[int, str]"),
    _entry("benchmark_label", ["benchmark"], [False], "str"),
    _entry("policy_label", ["policy"], [False], "str"),
    _entry("policy_label_rows", ["policies"], [False], "list[dict[str, Any]]"),
    _entry("effective_l3_misses", ["row"], [False], "float | None"),
    _entry("timing_valid_for_speedup", ["row"], [False], "bool"),
    _entry("timing_valid_label", ["row"], [False], "str"),
    _entry("timing_model_label", ["row"], [False], "str"),
    _entry("compare_key", ["row"], [False], "tuple[Any, ...]"),
    _entry("run_profile", ["args", "run_root", "profile"], [False, False, False], "Path"),
    _entry("collect_csvs", ["run_dirs", "input_csvs"], [False, False],
           "tuple[list[dict[str, Any]], list[dict[str, Any]]]"),
    _entry("summarize_roi", ["rows"], [False], "list[dict[str, Any]]"),
    _entry("roi_relative_metrics", ["rows"], [False], "list[dict[str, Any]]"),
    _entry("proof_relative_metrics", ["rows"], [False], "list[dict[str, Any]]"),
    _entry("summarize_relative", ["rows", "metrics"], [False, False], "list[dict[str, Any]]"),
    _entry("charged_overhead", ["rows"], [False], "list[dict[str, Any]]"),
    _entry("faithfulness_summary", ["rows"], [False], "list[dict[str, Any]]"),
    _entry("popt_storage_overhead_summary", ["rows"], [False], "list[dict[str, Any]]"),
    _entry("ecg_mode_overhead_rows", [], [], "list[dict[str, Any]]"),
    _entry("prefetch_quality_summary", ["roi_rows", "proof_rows"], [False, False], "list[dict[str, Any]]"),
    _entry("thread_scaling_metrics", ["rows"], [False], "list[dict[str, Any]]"),
    _entry("backend_direction_agreement", ["rows"], [False], "list[dict[str, Any]]"),
    _entry("sniper_cpi_stack_summary", ["rows"], [False], "list[dict[str, Any]]"),
    _entry("write_latex_table", ["path", "rows", "fields", "caption"], [False, False, False, False], "None"),
    _entry("set_paper_plot_style", [], [], "None"),
    _entry("save_figure", ["path"], [False], "None"),
    _entry("plot_metric_by_policy",
           ["path", "rows", "metric", "xlabel", "value_format", "reference"],
           [False, False, False, False, False, False], "None"),
    _entry("plot_grouped_metric_by_benchmark",
           ["path", "rows", "metric", "ylabel", "reference", "summary_label", "summary_mode"],
           [False, False, False, False, False, False, False], "None"),
    _entry("plot_charged_overhead", ["path", "rows"], [False, False], "None"),
    _entry("plot_sniper_thread_scaling", ["path", "rows"], [False, False], "None"),
    _entry("l_curve_rows", ["roi_rows"], [False],
           "dict[tuple[str, str], list[dict[str, Any]]]"),
    _entry("plot_l_curve", ["path", "group_key", "entries"], [False, False, False], "None"),
    _entry("generate_outputs",
           ["out_dir", "roi_rows", "proof_rows", "copy_to_paper"],
           [False, False, False, False], "None"),
    _entry("parse_args", ["argv"], [False], "argparse.Namespace"),
    _entry("main", ["argv"], [False], "int"),
]


FINAL_PAPER_RUN_STAGES: list[dict] = [
    _entry("utc_now", [], [], "str"),
    _entry("now_tag", [], [], "str"),
    _entry("sanitize", ["text"], [False], "str"),
    _entry("normalize_filter_token", ["text"], [False], "str"),
    _entry("token_matches", ["text", "filters"], [False, False], "bool"),
    _entry("filter_policy_specs", ["policies", "filters"], [False, False], "list[str]"),
    _entry("load_manifest", ["path"], [False], "dict[str, Any]"),
    _entry("resolve_path", ["path_text", "base"], [False, True], "Path"),
    _entry("find_graph_path", ["graph", "graph_dir", "allow_missing"], [False, False, False], "Path | None"),
    _entry("graph_uses_synthetic_options", ["graph"], [False], "bool"),
    _entry("options_for", ["manifest", "graph", "graph_path", "benchmark"], [False, False, False, False], "str"),
    _entry("merged_defaults", ["manifest", "stage"], [False, False], "dict[str, Any]"),
    _entry("expand_jobs", ["args", "manifest", "run_dir"], [False, False, False], "list[Job]"),
    _entry("make_proof_job", ["args", "run_dir", "settings"], [False, False, False], "Job"),
    _entry("make_roi_job",
           ["args", "manifest", "run_dir", "settings", "graph", "graph_path", "benchmark"],
           [False, False, False, False, False, False, False], "Job"),
    _entry("csv_status", ["path"], [False], "tuple[str, str]"),
    _entry("final_profiles_requested", ["profiles"], [False], "bool"),
    _entry("validate_gate", ["run_dir", "strict"], [False, False], "bool"),
    _entry("validate_literature_gate",
           ["run_dir", "sweep_root", "sweep_subdir", "strict"],
           [False, False, False, False], "bool"),
    _entry("validate_job_graphs", ["run_dir", "jobs", "strict"], [False, False, False], "bool"),
    _entry("latest_run_dir", [], [], "Path"),
    _entry("latest_job_events", ["run_dir"], [False], "dict[str, dict[str, Any]]"),
    _entry("display_status", ["row", "latest_events"], [False, False], "tuple[str, str]"),
    _entry("print_run_status", ["run_dir"], [False], "int"),
    _entry("command_text", ["command"], [False], "str"),
    _entry("write_run_manifest", ["run_dir", "args", "manifest", "jobs"],
           [False, False, False, False], "None"),
    _entry("write_combined_outputs", ["run_dir", "jobs"], [False, False], "None"),
    _entry("write_preflight", ["run_dir", "args"], [False, False], "None"),
    _entry("run_lock", ["lock_path"], [False], "Iterator[None]"),
    _entry("should_run", ["job", "args"], [False, False], "tuple[bool, str]"),
    _entry("append_status", ["run_dir", "record"], [False, False], "None"),
    _entry("terminate_process_group", ["process", "log", "timeout_s"], [False, False, True], "int"),
    _entry("run_job", ["job", "run_dir", "args"], [False, False, False], "int"),
    _entry("print_job_list", ["jobs"], [False], "None"),
    _entry("filter_jobs", ["jobs", "args"], [False, False], "list[Job]"),
    _entry("parse_args", ["argv"], [False], "argparse.Namespace"),
    _entry("main", ["argv"], [False], "int"),
]


STAGE_REGISTRY: dict[str, list[dict]] = {
    PAPER_PIPELINE_PATH:  PAPER_PIPELINE_STAGES,
    FINAL_PAPER_RUN_PATH: FINAL_PAPER_RUN_STAGES,
}


# --------------------------------------------------------------------
# AST helpers
# --------------------------------------------------------------------


def _parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text("utf-8"))


def _top_level_fns(module: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [n for n in module.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _live_signature(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> dict:
    a = fn.args
    pos = [arg.arg for arg in a.args]
    n_pos = len(pos)
    n_pos_defaults = len(a.defaults or [])
    pos_has_default = [False] * (n_pos - n_pos_defaults) + [True] * n_pos_defaults
    kwonly = [arg.arg for arg in a.kwonlyargs]
    kwonly_has_default = [d is not None for d in (a.kw_defaults or [])]
    return {
        "name": fn.name,
        "args": pos,
        "args_has_default": pos_has_default,
        "kwonly": kwonly,
        "kwonly_has_default": kwonly_has_default,
        "vararg": a.vararg.arg if a.vararg else None,
        "kwarg": a.kwarg.arg if a.kwarg else None,
        "returns": ast.unparse(fn.returns) if fn.returns else None,
        "is_async": isinstance(fn, ast.AsyncFunctionDef),
    }


def _live_public_signatures(rel: str) -> dict[str, dict]:
    p = REPO_ROOT / rel
    if not p.is_file():
        return {}
    mod = _parse_module(p)
    out: dict[str, dict] = {}
    for fn in _top_level_fns(mod):
        if fn.name.startswith("_"):
            continue
        out[fn.name] = _live_signature(fn)
    return out


# --------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------


def _check_s1_importable() -> list[dict]:
    out = []
    for rel in STAGE_REGISTRY:
        p = REPO_ROOT / rel
        if not p.is_file():
            out.append({"rule": "S1", "path": rel, "issue": "missing"})
            continue
        try:
            _parse_module(p)
        except SyntaxError as exc:
            out.append({"rule": "S1", "path": rel,
                        "issue": f"syntax error: {exc}"})
    return out


def _check_s2_fn_kind() -> list[dict]:
    out = []
    for rel, stages in STAGE_REGISTRY.items():
        live = _live_public_signatures(rel)
        for entry in stages:
            got = live.get(entry["name"])
            if got is None:
                out.append({"rule": "S2", "path": rel, "fn": entry["name"],
                            "issue": "missing top-level fn"})
                continue
            if got["is_async"] != entry["is_async"]:
                out.append({"rule": "S2", "path": rel, "fn": entry["name"],
                            "want_is_async": entry["is_async"],
                            "got_is_async": got["is_async"]})
    return out


def _check_s3_args_match() -> list[dict]:
    out = []
    for rel, stages in STAGE_REGISTRY.items():
        live = _live_public_signatures(rel)
        for entry in stages:
            got = live.get(entry["name"])
            if got is None:
                continue
            if got["args"] != entry["args"]:
                out.append({"rule": "S3", "path": rel, "fn": entry["name"],
                            "want_args": entry["args"],
                            "got_args": got["args"]})
    return out


def _check_s4_defaults_match() -> list[dict]:
    out = []
    for rel, stages in STAGE_REGISTRY.items():
        live = _live_public_signatures(rel)
        for entry in stages:
            got = live.get(entry["name"])
            if got is None:
                continue
            if got["args_has_default"] != entry["args_has_default"]:
                out.append({"rule": "S4", "path": rel, "fn": entry["name"],
                            "want_defaults": entry["args_has_default"],
                            "got_defaults": got["args_has_default"]})
    return out


def _check_s5_returns_match() -> list[dict]:
    out = []
    for rel, stages in STAGE_REGISTRY.items():
        live = _live_public_signatures(rel)
        for entry in stages:
            got = live.get(entry["name"])
            if got is None:
                continue
            if got["returns"] != entry["returns"]:
                out.append({"rule": "S5", "path": rel, "fn": entry["name"],
                            "want_returns": entry["returns"],
                            "got_returns": got["returns"]})
    return out


def _check_s6_varargs_kwonly() -> list[dict]:
    out = []
    for rel, stages in STAGE_REGISTRY.items():
        live = _live_public_signatures(rel)
        for entry in stages:
            got = live.get(entry["name"])
            if got is None:
                continue
            for key in ("vararg", "kwarg", "kwonly", "kwonly_has_default"):
                if got.get(key) != entry.get(key):
                    out.append({"rule": "S6", "path": rel, "fn": entry["name"],
                                "key": key,
                                "want": entry.get(key),
                                "got": got.get(key)})
    return out


def _check_s7_exhaustive() -> list[dict]:
    out = []
    for rel, stages in STAGE_REGISTRY.items():
        want_set = {e["name"] for e in stages}
        live = _live_public_signatures(rel)
        for extra in sorted(set(live.keys()) - want_set):
            out.append({"rule": "S7", "path": rel, "fn": extra,
                        "issue": "live public fn not in registry"})
    return out


# --------------------------------------------------------------------
# Audit + emit
# --------------------------------------------------------------------


def audit() -> dict:
    viols: list[dict] = []
    viols += _check_s1_importable()
    viols += _check_s2_fn_kind()
    viols += _check_s3_args_match()
    viols += _check_s4_defaults_match()
    viols += _check_s5_returns_match()
    viols += _check_s6_varargs_kwonly()
    viols += _check_s7_exhaustive()

    counts = {
        "orchestrators":         len(STAGE_REGISTRY),
        "paper_pipeline_fns":    len(PAPER_PIPELINE_STAGES),
        "final_paper_run_fns":   len(FINAL_PAPER_RUN_STAGES),
        "total_fns":             sum(len(v) for v in STAGE_REGISTRY.values()),
    }
    return {
        "schema": "lit-faith-paper-stage-registry/1",
        "status": "active",
        "counts": counts,
        "registry": {rel: list(stages) for rel, stages in STAGE_REGISTRY.items()},
        "rules": {
            "S1": "orchestrator module ast-parses cleanly",
            "S2": "every registered fn is a top-level fn of the locked kind (sync/async)",
            "S3": "every registered fn positional arg list matches live (names+order)",
            "S4": "every registered fn positional defaults-presence vector matches live",
            "S5": "every registered fn return annotation matches live (ast.unparse form)",
            "S6": "every registered fn vararg/kwarg/kwonly shape matches live",
            "S7": "registry exhaustive — every public top-level fn appears",
        },
        "violations": viols,
    }


def write_json(doc: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", "utf-8")


def write_md(doc: dict, path: Path) -> None:
    lines: list[str] = [
        "# Paper-pipeline stage registry (gate 275)",
        "",
        "_Auto-generated by `lit_faith_paper_stage_registry.py`._",
        "",
        f"- orchestrators: **{doc['counts']['orchestrators']}**",
        f"- paper_pipeline.py public fns: **{doc['counts']['paper_pipeline_fns']}**",
        f"- final_paper_run.py public fns: **{doc['counts']['final_paper_run_fns']}**",
        f"- total locked fns: **{doc['counts']['total_fns']}**",
        f"- violations: **{len(doc['violations'])}**",
        "",
    ]
    for rel, stages in sorted(doc["registry"].items()):
        lines.append(f"## `{rel}` ({len(stages)} public fns)")
        lines.append("")
        lines.append("| fn | args | defaults | returns | async |")
        lines.append("| --- | --- | --- | --- | --- |")
        for e in stages:
            args_txt = ", ".join(e["args"]) if e["args"] else "—"
            defaults_txt = "".join("D" if d else "." for d in e["args_has_default"]) or "—"
            ret = e["returns"] or "—"
            lines.append(
                f"| `{e['name']}` | `{args_txt}` | `{defaults_txt}` | "
                f"`{ret}` | `{e['is_async']}` |"
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
    rows = ["path,fn,args,defaults,returns,is_async,vararg,kwarg,kwonly"]
    for rel, stages in sorted(doc["registry"].items()):
        for e in stages:
            args_txt = "|".join(e["args"])
            defaults_txt = "".join("D" if d else "." for d in e["args_has_default"])
            kwonly_txt = "|".join(e["kwonly"])
            rows.append(
                f"{rel},{e['name']},{args_txt},{defaults_txt},"
                f"{e['returns'] or ''},{e['is_async']},"
                f"{e['vararg'] or ''},{e['kwarg'] or ''},{kwonly_txt}"
            )
    for v in doc["violations"]:
        rows.append(f"violation,{v.get('rule','')},{v.get('path','')},"
                    f"{v.get('fn','')},,,,,")
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
        f"[lit-faith-paper-stage-registry] "
        f"status={doc['status']} fns={doc['counts']['total_fns']} "
        f"violations={len(doc['violations'])}"
    )
    return 1 if doc["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
