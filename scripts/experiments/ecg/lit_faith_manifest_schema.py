"""Gate 276 — final-paper-run manifest schema registry.

Locks the JSON shape of ``scripts/experiments/ecg/final_paper_manifest.json``,
the single source of truth for every recipe ``final_paper_run.py``
expands into jobs. The manifest declares:

* the ``version`` integer (currently 1),
* the ``defaults`` dict (per-tool timeouts, cache sizes, sniper
  config knobs),
* the ``profiles`` dict (recipe-name → command-line tag),
* the ``benchmark_options`` dict (graph-set → benchmark → CLI options),
* the ``graph_sets`` dict (graph-set → list of graph entries),
* the ``stages`` list (recipe expansion plans).

Why this matters: every published reproduction recipe shells through
``final_paper_run --manifest scripts/experiments/ecg/final_paper_manifest.json
--profile <name>``. The manifest's shape IS the publish-reproduction
contract from a *data* angle (gates 274+275 lock the *code* angle).

Silent-drift cases this gate catches:

* Someone bumps ``"version": 1`` → ``"version": 2`` to add a new
  field, but every documented recipe still loads it as ``v1``.
* Someone removes ``timeout_cache`` from ``defaults`` and every
  ``cache_sim`` stage silently falls back to a hard-coded 600 s
  timeout that's too low for cit-Patents.
* Someone adds a new stage ``kind: "thread_scan"`` not in the locked
  set; ``expand_jobs()`` silently no-ops instead of raising.
* A ``graph_set`` entry references ``options_key: "syntehtic_g12"``
  (typo) — silently re-uses the default options.
* A stage references ``profiles: ["finall_cache_sim"]`` (typo) and
  the SBATCH wrapper silently expands to zero jobs.
* A graph entry drops the ``options_key`` field and the per-graph
  benchmark line becomes empty.

6 rules M1-M6:

* **M1** — manifest file exists, parses as JSON, ``version`` equals
  the locked value.
* **M2** — every locked top-level key is present (``version``,
  ``description``, ``defaults``, ``profiles``, ``benchmark_options``,
  ``graph_sets``, ``stages``); no surprise top-level keys.
* **M3** — every locked ``defaults`` key is present; every live
  ``defaults`` key is in the locked vocabulary.
* **M4** — every graph entry in ``graph_sets`` has the required
  ``{name, options_key}`` keys (``path`` optional); every value is
  a string.
* **M5** — every graph entry's ``options_key`` references a real
  ``benchmark_options`` entry.
* **M6** — every stage has required ``{name, kind, profiles,
  benchmarks}``; ``kind`` ∈ locked set; every stage ``profiles[i]``
  references a real ``profiles`` entry; every stage ``graph_set``
  (when present) references a real ``graph_sets`` entry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]

MANIFEST_PATH = "scripts/experiments/ecg/final_paper_manifest.json"

LOCKED_VERSION = 1

LOCKED_TOP_LEVEL_KEYS = {
    "version", "description", "defaults",
    "profiles", "benchmark_options", "graph_sets", "stages",
}

LOCKED_REQUIRED_DEFAULTS_KEYS = {
    # cache geometry
    "l1d_size", "l2_size", "l3_sizes", "l3_ways", "line_size",
    # timeouts
    "timeout_cache", "timeout_gem5", "timeout_sniper",
    # sniper knobs
    "sniper_root", "sniper_frontend", "sniper_omp_wait_policy",
    "sniper_memory_limit_gb", "sniper_base_config",
    "sniper_mimicos_memory_mb", "sniper_mimicos_kernel_mb",
    # build
    "no_build",
}

LOCKED_STAGE_KINDS = {"roi_matrix", "proof_matrix"}

LOCKED_REQUIRED_STAGE_KEYS = {"name", "kind", "profiles", "benchmarks"}

LOCKED_REQUIRED_GRAPH_ENTRY_KEYS = {"name", "options_key"}

LOCKED_OPTIONAL_GRAPH_ENTRY_KEYS = {"path"}

# Floors so someone can't silently strip the manifest down to a stub.
LOCKED_MIN_COUNTS = {
    "profiles":   30,
    "stages":     30,
    "graph_sets":  8,
    "benchmark_options": 4,
}


# --------------------------------------------------------------------
# Live loaders
# --------------------------------------------------------------------


def _live_manifest() -> dict | None:
    p = REPO_ROOT / MANIFEST_PATH
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text("utf-8"))
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------


def _check_m1_version() -> list[dict]:
    out = []
    p = REPO_ROOT / MANIFEST_PATH
    if not p.is_file():
        out.append({"rule": "M1", "issue": "manifest missing", "path": MANIFEST_PATH})
        return out
    try:
        live = json.loads(p.read_text("utf-8"))
    except json.JSONDecodeError as exc:
        out.append({"rule": "M1", "issue": f"json parse error: {exc}"})
        return out
    if live.get("version") != LOCKED_VERSION:
        out.append({"rule": "M1",
                    "want_version": LOCKED_VERSION,
                    "got_version": live.get("version")})
    return out


def _check_m2_top_level() -> list[dict]:
    out = []
    live = _live_manifest()
    if live is None:
        return out
    live_keys = set(live.keys())
    missing = LOCKED_TOP_LEVEL_KEYS - live_keys
    extra = live_keys - LOCKED_TOP_LEVEL_KEYS
    for k in sorted(missing):
        out.append({"rule": "M2", "missing_top_level_key": k})
    for k in sorted(extra):
        out.append({"rule": "M2", "extra_top_level_key": k})
    if "stages" in live and not isinstance(live["stages"], list):
        out.append({"rule": "M2",
                    "issue": "stages must be a list",
                    "got_type": type(live["stages"]).__name__})
    return out


def _check_m3_defaults() -> list[dict]:
    out = []
    live = _live_manifest()
    if live is None or not isinstance(live.get("defaults"), dict):
        return out
    live_keys = set(live["defaults"].keys())
    missing = LOCKED_REQUIRED_DEFAULTS_KEYS - live_keys
    extra = live_keys - LOCKED_REQUIRED_DEFAULTS_KEYS
    for k in sorted(missing):
        out.append({"rule": "M3", "missing_defaults_key": k})
    for k in sorted(extra):
        out.append({"rule": "M3", "extra_defaults_key": k})
    return out


def _check_m4_graph_entry_shape() -> list[dict]:
    out = []
    live = _live_manifest()
    if live is None or not isinstance(live.get("graph_sets"), dict):
        return out
    allowed = LOCKED_REQUIRED_GRAPH_ENTRY_KEYS | LOCKED_OPTIONAL_GRAPH_ENTRY_KEYS
    for gs_name, entries in live["graph_sets"].items():
        if not isinstance(entries, list):
            out.append({"rule": "M4", "graph_set": gs_name,
                        "issue": "graph_sets[graph_set] must be a list"})
            continue
        for i, e in enumerate(entries):
            if not isinstance(e, dict):
                out.append({"rule": "M4", "graph_set": gs_name, "index": i,
                            "issue": "entry must be a dict"})
                continue
            missing = LOCKED_REQUIRED_GRAPH_ENTRY_KEYS - set(e.keys())
            for k in sorted(missing):
                out.append({"rule": "M4", "graph_set": gs_name, "index": i,
                            "missing_key": k})
            extra = set(e.keys()) - allowed
            for k in sorted(extra):
                out.append({"rule": "M4", "graph_set": gs_name, "index": i,
                            "extra_key": k})
            for k, v in e.items():
                if not isinstance(v, str):
                    out.append({"rule": "M4", "graph_set": gs_name, "index": i,
                                "key": k, "want_type": "str",
                                "got_type": type(v).__name__})
    return out


def _check_m5_options_key_xref() -> list[dict]:
    out = []
    live = _live_manifest()
    if live is None:
        return out
    if not isinstance(live.get("graph_sets"), dict):
        return out
    if not isinstance(live.get("benchmark_options"), dict):
        return out
    bo_keys = set(live["benchmark_options"].keys())
    for gs_name, entries in live["graph_sets"].items():
        if not isinstance(entries, list):
            continue
        for i, e in enumerate(entries):
            if not isinstance(e, dict):
                continue
            ok = e.get("options_key")
            if isinstance(ok, str) and ok not in bo_keys:
                out.append({"rule": "M5", "graph_set": gs_name, "index": i,
                            "options_key": ok,
                            "issue": "options_key not in benchmark_options"})
    return out


def _check_m6_stage_shape() -> list[dict]:
    out = []
    live = _live_manifest()
    if live is None or not isinstance(live.get("stages"), list):
        return out
    profile_names = set(live.get("profiles", {}).keys()) \
        if isinstance(live.get("profiles"), dict) else set()
    graph_set_names = set(live.get("graph_sets", {}).keys()) \
        if isinstance(live.get("graph_sets"), dict) else set()
    for i, s in enumerate(live["stages"]):
        if not isinstance(s, dict):
            out.append({"rule": "M6", "index": i, "issue": "stage must be a dict"})
            continue
        name = s.get("name", f"<stage[{i}]>")
        missing = LOCKED_REQUIRED_STAGE_KEYS - set(s.keys())
        for k in sorted(missing):
            out.append({"rule": "M6", "stage": name,
                        "missing_required_key": k})
        kind = s.get("kind")
        if kind is not None and kind not in LOCKED_STAGE_KINDS:
            out.append({"rule": "M6", "stage": name,
                        "got_kind": kind,
                        "issue": "kind not in locked vocabulary"})
        if isinstance(s.get("profiles"), list):
            for j, p in enumerate(s["profiles"]):
                if isinstance(p, str) and p not in profile_names:
                    out.append({"rule": "M6", "stage": name,
                                "profile_index": j, "profile_ref": p,
                                "issue": "profile not in profiles"})
        gs = s.get("graph_set")
        if isinstance(gs, str) and gs not in graph_set_names:
            out.append({"rule": "M6", "stage": name,
                        "graph_set": gs,
                        "issue": "graph_set not in graph_sets"})
    return out


def _check_min_counts() -> list[dict]:
    """Bundled into M2 violations (no separate rule) — these are
    floor invariants on the top-level dicts/lists."""
    out = []
    live = _live_manifest()
    if live is None:
        return out
    for key, floor in LOCKED_MIN_COUNTS.items():
        container = live.get(key)
        if container is None:
            continue
        try:
            n = len(container)
        except TypeError:
            continue
        if n < floor:
            out.append({"rule": "M2", "key": key,
                        "want_min": floor, "got": n,
                        "issue": "below floor (stripped manifest?)"})
    return out


# --------------------------------------------------------------------
# Audit + emit
# --------------------------------------------------------------------


def audit() -> dict:
    viols: list[dict] = []
    viols += _check_m1_version()
    viols += _check_m2_top_level()
    viols += _check_min_counts()
    viols += _check_m3_defaults()
    viols += _check_m4_graph_entry_shape()
    viols += _check_m5_options_key_xref()
    viols += _check_m6_stage_shape()

    live = _live_manifest()
    counts = {
        "version":            live.get("version") if live else None,
        "profiles":           len(live.get("profiles", {}))            if live else 0,
        "stages":             len(live.get("stages", []))              if live else 0,
        "graph_sets":         len(live.get("graph_sets", {}))          if live else 0,
        "benchmark_options":  len(live.get("benchmark_options", {}))   if live else 0,
        "defaults_keys":      len(live.get("defaults", {}))            if live else 0,
    }
    return {
        "schema": "lit-faith-manifest-schema/1",
        "status": "active",
        "path": MANIFEST_PATH,
        "locked_version": LOCKED_VERSION,
        "counts": counts,
        "locked": {
            "top_level_keys":        sorted(LOCKED_TOP_LEVEL_KEYS),
            "defaults_keys":         sorted(LOCKED_REQUIRED_DEFAULTS_KEYS),
            "stage_kinds":           sorted(LOCKED_STAGE_KINDS),
            "required_stage_keys":   sorted(LOCKED_REQUIRED_STAGE_KEYS),
            "required_graph_keys":   sorted(LOCKED_REQUIRED_GRAPH_ENTRY_KEYS),
            "optional_graph_keys":   sorted(LOCKED_OPTIONAL_GRAPH_ENTRY_KEYS),
            "min_counts":            dict(LOCKED_MIN_COUNTS),
        },
        "rules": {
            "M1": "manifest exists, parses, version equals locked",
            "M2": "top-level keys match locked set; min-counts hold",
            "M3": "defaults keys match locked vocabulary",
            "M4": "graph entry shape — required keys + str types",
            "M5": "graph entry options_key references real benchmark_options",
            "M6": "stage shape, kind, profile/graph_set xref",
        },
        "violations": viols,
    }


def write_json(doc: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", "utf-8")


def write_md(doc: dict, path: Path) -> None:
    c = doc["counts"]
    lines: list[str] = [
        "# Final-paper-run manifest schema registry (gate 276)",
        "",
        "_Auto-generated by `lit_faith_manifest_schema.py`._",
        "",
        f"- manifest: `{doc['path']}`",
        f"- locked version: **{doc['locked_version']}** (live: {c['version']})",
        f"- profiles: **{c['profiles']}** (floor {LOCKED_MIN_COUNTS['profiles']})",
        f"- stages: **{c['stages']}** (floor {LOCKED_MIN_COUNTS['stages']})",
        f"- graph_sets: **{c['graph_sets']}** (floor {LOCKED_MIN_COUNTS['graph_sets']})",
        f"- benchmark_options: **{c['benchmark_options']}** "
        f"(floor {LOCKED_MIN_COUNTS['benchmark_options']})",
        f"- defaults keys: **{c['defaults_keys']}** "
        f"(locked {len(LOCKED_REQUIRED_DEFAULTS_KEYS)})",
        f"- violations: **{len(doc['violations'])}**",
        "",
        "## Locked vocabularies",
        "",
        f"- **top-level keys** ({len(LOCKED_TOP_LEVEL_KEYS)}): "
        f"`{', '.join(sorted(LOCKED_TOP_LEVEL_KEYS))}`",
        f"- **defaults keys** ({len(LOCKED_REQUIRED_DEFAULTS_KEYS)}): "
        f"`{', '.join(sorted(LOCKED_REQUIRED_DEFAULTS_KEYS))}`",
        f"- **stage kinds** ({len(LOCKED_STAGE_KINDS)}): "
        f"`{', '.join(sorted(LOCKED_STAGE_KINDS))}`",
        f"- **required stage keys** ({len(LOCKED_REQUIRED_STAGE_KEYS)}): "
        f"`{', '.join(sorted(LOCKED_REQUIRED_STAGE_KEYS))}`",
        f"- **required graph-entry keys** "
        f"({len(LOCKED_REQUIRED_GRAPH_ENTRY_KEYS)}): "
        f"`{', '.join(sorted(LOCKED_REQUIRED_GRAPH_ENTRY_KEYS))}`",
        f"- **optional graph-entry keys** "
        f"({len(LOCKED_OPTIONAL_GRAPH_ENTRY_KEYS)}): "
        f"`{', '.join(sorted(LOCKED_OPTIONAL_GRAPH_ENTRY_KEYS))}`",
        "",
    ]
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
    rows = ["section,key,value"]
    c = doc["counts"]
    for k in ["version", "profiles", "stages", "graph_sets",
              "benchmark_options", "defaults_keys"]:
        rows.append(f"counts,{k},{c[k]}")
    rows.append(f"locked,version,{doc['locked_version']}")
    for k in sorted(LOCKED_TOP_LEVEL_KEYS):
        rows.append(f"top_level_key,{k},")
    for k in sorted(LOCKED_REQUIRED_DEFAULTS_KEYS):
        rows.append(f"defaults_key,{k},")
    for k in sorted(LOCKED_STAGE_KINDS):
        rows.append(f"stage_kind,{k},")
    for v in doc["violations"]:
        rows.append(f"violation,{v.get('rule','')},"
                    f"{json.dumps(v, sort_keys=True)}")
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
        f"[lit-faith-manifest-schema] "
        f"status={doc['status']} version={doc['counts']['version']} "
        f"profiles={doc['counts']['profiles']} "
        f"stages={doc['counts']['stages']} "
        f"violations={len(doc['violations'])}"
    )
    return 1 if doc["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
