"""Gate 265 — gem5/Sniper sideband filename + env-var grammar registry.

Twelfth in the vocabulary-lock series (252 SBATCH, 255 policy, 256
profile, 257 backend, 258 graph, 259 build, 260 CLI, 261 arm-catalog,
262 cross-tool schema, 263 config matrix, 264 wiki/data filename
grammar, 265 sideband filename grammar). Locks the **filename** +
**env-var** vocabulary that gem5 and Sniper overlays use to refer
to the four runtime sideband artifacts (context JSON, popt matrix
bin, out-edges bin, in-edges bin) — the wire-format between the
benchmark process and the simulator overlay.

Gate 248 already locks the *internal schema* of the ``[graphctx]
register region`` log line (field names + printf specifiers + parser
regex). Gate 265 locks the complementary surface area: the names by
which those artifacts are referred to at every emit-site (C++
overlay default-paths via ``gem5_env_or_default`` / ``envOrDefault``)
and every parse-site (Python ``gem5_sideband_paths`` /
``sniper_sideband_paths`` dicts in ``scripts/experiments/ecg/
roi_matrix.py``).

Catches the silent-drift cases:

* a contributor renames ``gem5_graphbrew_ctx.json`` to
  ``gem5_graphbrew_context.json`` in the C++ default-path string of
  ``gem5_harness.h`` but forgets to update the Python parse-site
  ``gem5_sideband_paths`` dict — runs succeed but Tier-A parsing
  silently sees zero registered regions and pivot tests RED for
  unrelated-looking reasons;
* a contributor flips ``SNIPER_POPT_MATRIX`` to ``SNIPER_PMATRIX``
  in one of three Sniper cache-set sources (popt/grasp/ecg) but
  not the others — runs with the changed cache-set fall back to
  ``/tmp/sniper_popt_matrix.bin`` (the unchanged literal default)
  while the runner sets ``SNIPER_PMATRIX`` (the new env name), and
  the POPT vector never loads;
* a contributor adds a fifth sideband artifact under a path like
  ``/tmp/gem5_popt_pfx.bin`` without registering it here — runs
  succeed but the proof-matrix audit (gate 1's parsing) doesn't
  know to clean it up between runs, and stale state from prior
  runs silently contaminates the next run;
* the ``graphbrew_sidebands/`` subdirectory convention used by
  the Python parsers is changed to ``sidebands/`` (typo) — runs
  succeed but Python parsers look in the wrong dir and silently
  treat all artifacts as missing.

7 rules S1-S7:

  S1: every registry entry filename matches
      ``^(gem5|sniper)_[a-z0-9_]+\\.(json|bin)$`` — tool prefix,
      lower_snake_case stem, approved extension (.json for context,
      .bin for matrix/edges).
  S2: every registry filename has tool-prefix matching its ``tool``
      field; ``role`` ∈ {context, popt_matrix, out_edges, in_edges};
      ``context`` → .json, others → .bin (no mixed up extensions).
  S3: env_var = ``<TOOL>_<STEM>`` (uppercase + underscores, derived
      from filename); default_path = ``/tmp/<filename>`` —
      bijection between filename and (env_var, default_path).
  S4: every ``gem5_env_or_default(NAME, PATH)`` call in
      ``bench/include/gem5_sim/gem5_harness.h`` for a sideband role
      uses a registry-declared (NAME, PATH) pair; no orphan
      sideband-shaped literals in that file.
  S5: every Sniper ``envOrDefault(NAME, PATH)`` call in the
      cache-set ``.cc`` files (cache_set_popt.cc, cache_set_grasp.cc,
      cache_set_ecg.cc) AND the prefetcher ``.cc`` files
      (ecg_pfx_prefetcher.cc, droplet_prefetcher.cc) for a sideband
      role uses a registry-declared (NAME, PATH) pair.
  S6: every Python sideband-path dict entry in
      ``scripts/experiments/ecg/roi_matrix.py``'s
      ``gem5_sideband_paths`` and ``sniper_sideband_paths`` functions
      points at ``<sideband_dir>/<canonical-filename>`` for some
      ``role`` in the registry — keys and filenames bijective with
      registry. ``sideband_dir`` = ``<tool>_out / 'graphbrew_sidebands'``.
  S7: every sideband filename literal found anywhere under
      ``bench/include/{gem5,sniper}_sim/`` (overlays + harness) AND
      under ``scripts/experiments/ecg/roi_matrix.py`` is declared in
      the registry — no orphan ``(gem5|sniper)_[a-z_]+\\.(json|bin)``
      literals.

Today the registry has 8 entries (4 gem5 + 4 sniper). Together with
gate 248 (schema content) and gate 264 (wiki/data filename shape)
this completes the filename/vocabulary lock on every shipping
surface in the pipeline.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2].parent
BENCH_GEM5 = ROOT / "bench" / "include" / "gem5_sim"
BENCH_SNIPER = ROOT / "bench" / "include" / "sniper_sim"
ROI_MATRIX = ROOT / "scripts" / "experiments" / "ecg" / "roi_matrix.py"


# ----------------------------------------------------------- registry --

# Hand-curated source-of-truth. Each entry pins the (tool, role,
# filename, env_var, default_path) tuple. The audit then proves every
# emit-site and parse-site references this canonical tuple.
SIDEBAND_REGISTRY: list[dict[str, str]] = [
    # gem5 — four roles
    {
        "tool":         "gem5",
        "role":         "context",
        "filename":     "gem5_graphbrew_ctx.json",
        "env_var":      "GEM5_GRAPHBREW_CTX",
        "default_path": "/tmp/gem5_graphbrew_ctx.json",
    },
    {
        "tool":         "gem5",
        "role":         "popt_matrix",
        "filename":     "gem5_popt_matrix.bin",
        "env_var":      "GEM5_POPT_MATRIX",
        "default_path": "/tmp/gem5_popt_matrix.bin",
    },
    {
        "tool":         "gem5",
        "role":         "out_edges",
        "filename":     "gem5_graphbrew_out_edges.bin",
        "env_var":      "GEM5_GRAPHBREW_OUT_EDGES",
        "default_path": "/tmp/gem5_graphbrew_out_edges.bin",
    },
    {
        "tool":         "gem5",
        "role":         "in_edges",
        "filename":     "gem5_graphbrew_in_edges.bin",
        "env_var":      "GEM5_GRAPHBREW_IN_EDGES",
        "default_path": "/tmp/gem5_graphbrew_in_edges.bin",
    },
    # Sniper — four roles
    {
        "tool":         "sniper",
        "role":         "context",
        "filename":     "sniper_graphbrew_ctx.json",
        "env_var":      "SNIPER_GRAPHBREW_CTX",
        "default_path": "/tmp/sniper_graphbrew_ctx.json",
    },
    {
        "tool":         "sniper",
        "role":         "popt_matrix",
        "filename":     "sniper_popt_matrix.bin",
        "env_var":      "SNIPER_POPT_MATRIX",
        "default_path": "/tmp/sniper_popt_matrix.bin",
    },
    {
        "tool":         "sniper",
        "role":         "out_edges",
        "filename":     "sniper_graphbrew_out_edges.bin",
        "env_var":      "SNIPER_GRAPHBREW_OUT_EDGES",
        "default_path": "/tmp/sniper_graphbrew_out_edges.bin",
    },
    {
        "tool":         "sniper",
        "role":         "in_edges",
        "filename":     "sniper_graphbrew_in_edges.bin",
        "env_var":      "SNIPER_GRAPHBREW_IN_EDGES",
        "default_path": "/tmp/sniper_graphbrew_in_edges.bin",
    },
]

CANONICAL_ROLES = ("context", "popt_matrix", "out_edges", "in_edges")
CANONICAL_TOOLS = ("gem5", "sniper")
SIDEBAND_SUBDIR = "graphbrew_sidebands"

# Files known to define sideband-shaped string literals. The S7 scanner
# walks every line of these files looking for filename literals; any
# literal matching FILENAME_RE that is NOT in the registry is an orphan.
GEM5_HARNESS = BENCH_GEM5 / "gem5_harness.h"
SNIPER_CACHE_SETS = [
    BENCH_SNIPER / "overlays/common/core/memory_subsystem/cache/cache_set_popt.cc",
    BENCH_SNIPER / "overlays/common/core/memory_subsystem/cache/cache_set_grasp.cc",
    BENCH_SNIPER / "overlays/common/core/memory_subsystem/cache/cache_set_ecg.cc",
]
SNIPER_PREFETCHERS = [
    BENCH_SNIPER / "overlays/common/core/memory_subsystem/parametric_dram_directory_msi/ecg_pfx_prefetcher.cc",
    BENCH_SNIPER / "overlays/common/core/memory_subsystem/parametric_dram_directory_msi/droplet_prefetcher.cc",
]

# Files allowed to reference sideband filenames in a non-canonical
# context (test assertion strings; documentation comments). They must
# still only use *registered* filenames, but they are not emit-sites.
PARSE_SITES = [ROI_MATRIX]


FILENAME_RE = re.compile(r'\b(gem5|sniper)_[a-z0-9_]+\.(?:json|bin)\b')
ENV_VAR_RE = re.compile(r'\b(GEM5|SNIPER)_[A-Z0-9_]+\b')

# Sideband env vars not associated with a *file* — these are allowed
# under that prefix without being in the filename registry. SNIPER_ECG_MODE
# is a config selector, not a path.
ENV_VAR_NON_FILE_ALLOW = {"SNIPER_ECG_MODE"}

# Filename-shaped strings that match `(gem5|sniper)_*\.(json|bin)` but
# are NOT runtime sideband artifacts — typically build-time config files
# or overlay-tracker dotfiles. These must be explicitly allow-listed so
# S6/S7 don't false-positive on them.
FILENAME_NON_SIDEBAND_ALLOW = {
    "sniper_overlays.json",  # bench/include/sniper_sim/.sniper_overlays.json
                             # — overlay-installation tracker, NOT a runtime
                             # sideband. Path is hidden (leading dot) but
                             # the regex doesn't see the dot prefix.
}

GEM5_CALL_RE = re.compile(
    r'gem5_env_or_default\s*\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)'
)
SNIPER_CALL_RE = re.compile(
    r'envOrDefault\s*\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)'
)


# ----------------------------------------------------------- helpers --

def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _by_filename() -> dict[str, dict[str, str]]:
    return {e["filename"]: e for e in SIDEBAND_REGISTRY}


def _by_env_var() -> dict[str, dict[str, str]]:
    return {e["env_var"]: e for e in SIDEBAND_REGISTRY}


# ----------------------------------------------------------- audit --

def audit() -> dict:
    violations: list[dict] = []
    by_filename = _by_filename()
    by_env_var = _by_env_var()

    # S1 — registry filename grammar
    for entry in SIDEBAND_REGISTRY:
        fn = entry["filename"]
        if not re.match(r"^(gem5|sniper)_[a-z0-9_]+\.(json|bin)$", fn):
            violations.append({
                "rule": "S1",
                "where": f"registry[{fn}]",
                "msg":  f"filename does not match canonical grammar: {fn!r}",
            })

    # S2 — tool/role/extension consistency
    for entry in SIDEBAND_REGISTRY:
        fn = entry["filename"]
        tool = entry["tool"]
        role = entry["role"]
        if tool not in CANONICAL_TOOLS:
            violations.append({
                "rule": "S2",
                "where": f"registry[{fn}]",
                "msg":  f"unknown tool {tool!r}; expected one of {CANONICAL_TOOLS}",
            })
        if role not in CANONICAL_ROLES:
            violations.append({
                "rule": "S2",
                "where": f"registry[{fn}]",
                "msg":  f"unknown role {role!r}; expected one of {CANONICAL_ROLES}",
            })
        if not fn.startswith(f"{tool}_"):
            violations.append({
                "rule": "S2",
                "where": f"registry[{fn}]",
                "msg":  f"filename {fn!r} does not have tool-prefix {tool}_",
            })
        if role == "context" and not fn.endswith(".json"):
            violations.append({
                "rule": "S2",
                "where": f"registry[{fn}]",
                "msg":  f"role=context but filename does not end .json: {fn!r}",
            })
        if role != "context" and not fn.endswith(".bin"):
            violations.append({
                "rule": "S2",
                "where": f"registry[{fn}]",
                "msg":  f"role={role} but filename does not end .bin: {fn!r}",
            })

    # S3 — env_var / default_path bijection with filename
    for entry in SIDEBAND_REGISTRY:
        fn = entry["filename"]
        env_var = entry["env_var"]
        default_path = entry["default_path"]
        stem = fn.rsplit(".", 1)[0]
        expected_env = stem.upper()
        if env_var != expected_env:
            violations.append({
                "rule": "S3",
                "where": f"registry[{fn}]",
                "msg":  f"env_var {env_var!r} does not match expected "
                        f"{expected_env!r} (derived from filename stem)",
            })
        expected_path = f"/tmp/{fn}"
        if default_path != expected_path:
            violations.append({
                "rule": "S3",
                "where": f"registry[{fn}]",
                "msg":  f"default_path {default_path!r} does not match expected "
                        f"{expected_path!r}",
            })

    # S4 — gem5_harness.h calls
    gem5_harness_text = _read(GEM5_HARNESS)
    if not gem5_harness_text:
        violations.append({
            "rule": "S4",
            "where": str(GEM5_HARNESS.relative_to(ROOT)),
            "msg":  "gem5_harness.h not readable",
        })
    else:
        for env_var, default_path in GEM5_CALL_RE.findall(gem5_harness_text):
            if not env_var.startswith("GEM5_"):
                continue
            if env_var in ENV_VAR_NON_FILE_ALLOW:
                continue
            if env_var not in by_env_var:
                violations.append({
                    "rule": "S4",
                    "where": str(GEM5_HARNESS.relative_to(ROOT)),
                    "msg":  f"orphan gem5_env_or_default call: env_var {env_var!r} "
                            f"not in registry",
                })
                continue
            entry = by_env_var[env_var]
            if default_path != entry["default_path"]:
                violations.append({
                    "rule": "S4",
                    "where": str(GEM5_HARNESS.relative_to(ROOT)),
                    "msg":  f"env_var {env_var!r} default_path mismatch: "
                            f"file says {default_path!r}, registry says "
                            f"{entry['default_path']!r}",
                })
            if entry["tool"] != "gem5":
                violations.append({
                    "rule": "S4",
                    "where": str(GEM5_HARNESS.relative_to(ROOT)),
                    "msg":  f"env_var {env_var!r} resolves to non-gem5 registry "
                            f"entry tool={entry['tool']!r}",
                })

    # S5 — sniper cache-set + prefetcher calls
    for srcfile in SNIPER_CACHE_SETS + SNIPER_PREFETCHERS:
        text = _read(srcfile)
        if not text:
            violations.append({
                "rule": "S5",
                "where": str(srcfile.relative_to(ROOT)),
                "msg":  "Sniper sideband source not readable",
            })
            continue
        for env_var, default_path in SNIPER_CALL_RE.findall(text):
            if not env_var.startswith("SNIPER_"):
                continue
            if env_var in ENV_VAR_NON_FILE_ALLOW:
                continue
            if env_var not in by_env_var:
                violations.append({
                    "rule": "S5",
                    "where": str(srcfile.relative_to(ROOT)),
                    "msg":  f"orphan envOrDefault call: env_var {env_var!r} "
                            f"not in registry",
                })
                continue
            entry = by_env_var[env_var]
            if default_path != entry["default_path"]:
                violations.append({
                    "rule": "S5",
                    "where": str(srcfile.relative_to(ROOT)),
                    "msg":  f"env_var {env_var!r} default_path mismatch: "
                            f"file says {default_path!r}, registry says "
                            f"{entry['default_path']!r}",
                })
            if entry["tool"] != "sniper":
                violations.append({
                    "rule": "S5",
                    "where": str(srcfile.relative_to(ROOT)),
                    "msg":  f"env_var {env_var!r} resolves to non-sniper registry "
                            f"entry tool={entry['tool']!r}",
                })

    # S6 — roi_matrix.py Python sideband-path dicts
    roi_text = _read(ROI_MATRIX)
    if not roi_text:
        violations.append({
            "rule": "S6",
            "where": str(ROI_MATRIX.relative_to(ROOT)),
            "msg":  "roi_matrix.py not readable",
        })
    else:
        for tool in CANONICAL_TOOLS:
            func_name = f"{tool}_sideband_paths"
            # Find function body — slice between `def {func_name}` and the
            # next top-level `def `.
            start_m = re.search(
                rf"^def {re.escape(func_name)}\b", roi_text, re.MULTILINE
            )
            if not start_m:
                violations.append({
                    "rule": "S6",
                    "where": str(ROI_MATRIX.relative_to(ROOT)),
                    "msg":  f"function {func_name} not found in roi_matrix.py",
                })
                continue
            tail = roi_text[start_m.start():]
            end_m = re.search(r"^def \w+\b", tail[1:], re.MULTILINE)
            body = tail if not end_m else tail[: end_m.start() + 1]

            # S6a — function must declare `sideband_dir = <tool>_out / SIDEBAND_SUBDIR`
            if f'"{SIDEBAND_SUBDIR}"' not in body and f"'{SIDEBAND_SUBDIR}'" not in body:
                violations.append({
                    "rule": "S6",
                    "where": f"{ROI_MATRIX.relative_to(ROOT)}::{func_name}",
                    "msg":  f"function does not reference canonical "
                            f"sideband_dir literal {SIDEBAND_SUBDIR!r}",
                })

            # S6b — function must include each canonical role mapped to the
            # canonical filename for its tool.
            for entry in SIDEBAND_REGISTRY:
                if entry["tool"] != tool:
                    continue
                role = entry["role"]
                fn = entry["filename"]
                # Look for `"<role>":` near `"<fn>"` in body — both literals.
                if f'"{role}"' not in body and f"'{role}'" not in body:
                    violations.append({
                        "rule": "S6",
                        "where": f"{ROI_MATRIX.relative_to(ROOT)}::{func_name}",
                        "msg":  f"function missing role key {role!r} for tool {tool}",
                    })
                if f'"{fn}"' not in body and f"'{fn}'" not in body:
                    violations.append({
                        "rule": "S6",
                        "where": f"{ROI_MATRIX.relative_to(ROOT)}::{func_name}",
                        "msg":  f"function missing filename literal {fn!r} for "
                                f"role {role!r}",
                    })

            # S6c — no orphan sideband filename literals in the function body
            # (catches typos and partial renames).
            literals = set(FILENAME_RE.findall(body))
            # FILENAME_RE returns tool prefix as a single match group; we
            # need full filenames. Re-run with findall on a non-group regex.
            literals = set(re.findall(
                r'(?:gem5|sniper)_[a-z0-9_]+\.(?:json|bin)', body
            ))
            for lit in literals:
                if lit in FILENAME_NON_SIDEBAND_ALLOW:
                    continue
                if lit not in by_filename:
                    violations.append({
                        "rule": "S6",
                        "where": f"{ROI_MATRIX.relative_to(ROOT)}::{func_name}",
                        "msg":  f"orphan sideband filename literal: {lit!r}",
                    })

    # S7 — no orphan sideband filename literals anywhere in the audited
    # emit/parse sites (gem5_harness.h + sniper cache_set + sniper
    # prefetcher + roi_matrix.py). Together S4/S5/S6 cover the call-site
    # *semantics* (env_var ↔ default_path); S7 covers raw filename
    # literals that might appear in comments or other unconventional
    # contexts.
    for srcfile in [GEM5_HARNESS, *SNIPER_CACHE_SETS, *SNIPER_PREFETCHERS, ROI_MATRIX]:
        text = _read(srcfile)
        if not text:
            continue
        literals = set(re.findall(
            r'(?:gem5|sniper)_[a-z0-9_]+\.(?:json|bin)', text
        ))
        for lit in literals:
            if lit in FILENAME_NON_SIDEBAND_ALLOW:
                continue
            if lit not in by_filename:
                violations.append({
                    "rule": "S7",
                    "where": str(srcfile.relative_to(ROOT)),
                    "msg":  f"orphan sideband filename literal: {lit!r}",
                })

    return {
        "status":      "active",
        "registry_n":  len(SIDEBAND_REGISTRY),
        "tools":       list(CANONICAL_TOOLS),
        "roles":       list(CANONICAL_ROLES),
        "sideband_subdir": SIDEBAND_SUBDIR,
        "emit_sites_audited": [
            str(GEM5_HARNESS.relative_to(ROOT)),
            *(str(p.relative_to(ROOT)) for p in SNIPER_CACHE_SETS),
            *(str(p.relative_to(ROOT)) for p in SNIPER_PREFETCHERS),
        ],
        "parse_sites_audited": [str(ROI_MATRIX.relative_to(ROOT))],
        "registry":    SIDEBAND_REGISTRY,
        "violations":  violations,
    }


# ------------------------------------------------------------- IO --

def _md_report(data: dict) -> str:
    buf = io.StringIO()
    buf.write("# Gate 265 — gem5/Sniper sideband filename + env-var grammar\n\n")
    buf.write(f"- status: `{data['status']}`\n")
    buf.write(f"- registry entries: {data['registry_n']}\n")
    buf.write(f"- tools: {', '.join(data['tools'])}\n")
    buf.write(f"- roles: {', '.join(data['roles'])}\n")
    buf.write(f"- sideband subdir: `{data['sideband_subdir']}`\n")
    buf.write(f"- violations: {len(data['violations'])}\n\n")

    buf.write("## Registry\n\n")
    buf.write("| tool | role | filename | env_var | default_path |\n")
    buf.write("|---|---|---|---|---|\n")
    for e in data["registry"]:
        buf.write(
            f"| {e['tool']} | {e['role']} | `{e['filename']}` "
            f"| `{e['env_var']}` | `{e['default_path']}` |\n"
        )

    buf.write("\n## Emit-sites audited\n\n")
    for p in data["emit_sites_audited"]:
        buf.write(f"- `{p}`\n")

    buf.write("\n## Parse-sites audited\n\n")
    for p in data["parse_sites_audited"]:
        buf.write(f"- `{p}`\n")

    if data["violations"]:
        buf.write("\n## Violations\n\n")
        for v in data["violations"]:
            buf.write(f"- **{v['rule']}** @ `{v['where']}`: {v['msg']}\n")

    return buf.getvalue().rstrip() + "\n"


def _csv_report(data: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["tool", "role", "filename", "env_var", "default_path"])
    for e in data["registry"]:
        w.writerow([e["tool"], e["role"], e["filename"], e["env_var"], e["default_path"]])
    return buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json-out", required=True)
    p.add_argument("--md-out",   required=True)
    p.add_argument("--csv-out",  required=True)
    args = p.parse_args(argv)

    data = audit()

    Path(args.json_out).write_text(
        json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    Path(args.md_out).write_text(_md_report(data), encoding="utf-8")
    Path(args.csv_out).write_text(_csv_report(data), encoding="utf-8")

    print(
        f"[lit-faith-sideband-grammar] status={data['status']} "
        f"registry={data['registry_n']} tools={len(data['tools'])} "
        f"roles={len(data['roles'])} violations={len(data['violations'])}"
    )
    return 1 if data["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
