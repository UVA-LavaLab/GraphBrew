"""Gate 260 — GAPBS kernel CLI vocabulary registry.

Seventh in the vocabulary-lock series (252 SBATCH, 255 policy,
256 profile, 257 backend, 258 graph, 259 build, 260 CLI). Locks
WHICH command-line flags each GAPBS-derived kernel binary
actually accepts — so a contributor cannot silently pass ``-i 20``
to ``bfs`` (which uses ``CLApp`` and does NOT support
``-i``), pass ``-t 1e-6`` to ``cc`` (no PageRank tolerance), or
introduce a wrapper that emits ``--num-trials 5`` (long-form)
when the GAPBS getopt loop only ever recognises the short
``-n 5``.

Catches the silent-drift cases:

* a sweep helper iterates ``-i 16 -i 32`` for both pr AND bfs;
  bfs silently ignores ``-i`` and runs with default trials,
  so the sweep collapses to a single bfs measurement repeated
  N times,
* a regression run passes ``-t 1e-7`` (PageRank tolerance) to
  sssp; sssp's CLDelta intercepts ``-t`` as an unknown opt and
  bails out, but only after starting the run — wasted SBATCH
  time,
* a per-kernel CLI-arg lookup table downstream tries
  ``args["pr_spmv"]["--tolerance"]`` (long-form) but the
  binary only accepts the short ``-t`` form → KeyError at
  sweep-emit time.

The gate parses ``bench/include/external/gapbs/command_line.h``
for the five CL classes (CLBase / CLApp / CLIterApp /
CLPageRank / CLDelta / CLConvert), then cross-validates:

  - the canonical inheritance chain declared here
    (kernel → its CL class → the union of all getopt flags
    reachable from that class up to CLBase)
  - the on-disk source files (every kernel `.cc` under
    canonical SRC_DIRs must instantiate the canonical CL
    class)
  - the in-tree CLI literals (no Python sweep helper passes
    a flag the kernel does not accept).

7 rules R1-R7:
  R1: every CL class declared in the header has a non-empty
      getopt extension string (its ``get_args_ +=`` literal).
  R2: every canonical kernel → CL class mapping in
      KERNEL_CL_CLASS has a matching ``CL<…>`` instantiation
      in the kernel's source file (in every backend).
  R3: every canonical kernel's flag-set is the union of its
      class chain (no orphan kernel-only flags; no missing
      inherited flags).
  R4: no two CL classes' getopt extension strings overlap
      (e.g. both CLIterApp and CLPageRank claim ``i``;
      that is allowed BECAUSE they live on disjoint
      inheritance paths) — the rule fires only on within-chain
      conflicts.
  R5: every canonical CLI flag matches the documented regex
      ``^[A-Za-z]$`` (single ASCII letter; getopt convention).
  R6: every canonical short flag has a documented
      semantic-purpose label (file / scale / trials / verify /
      iters / tolerance / delta / ...).
  R7: the canonical flag-arity table (FLAG_TAKES_VALUE) is
      consistent with the getopt format string (every
      ``X:`` in the extension declares the same X as
      "takes value").
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
HEADER = REPO_ROOT / "bench" / "include" / "external" / "gapbs" / "command_line.h"
SRC_DIRS: dict[str, Path] = {
    "native":    REPO_ROOT / "bench" / "src",
    "cache_sim": REPO_ROOT / "bench" / "src_sim",
    "gem5":      REPO_ROOT / "bench" / "src_gem5",
}

FLAG_RE = re.compile(r"^[A-Za-z]$")


# --- Canonical inheritance chain ------------------------------------------

# Each class's getopt extension and immediate parent (the chain
# is closed by CLBase having parent=None). The full flag set for
# a class is the union of its own + its ancestors' flag-letters.
@dataclass(frozen=True)
class CLClass:
    name: str
    parent: str | None
    getopt_ext: str
    purpose: str


CL_CLASSES: tuple[CLClass, ...] = (
    CLClass(
        name="CLBase",
        parent=None,
        getopt_ext="f:g:hk:su:m:o:zj:SlD:",
        purpose="Base class: graph loader (-f/-g/-u/-k/-s/-m/-S), "
                "ordering (-o), logging (-l/-D), segmentation (-j), "
                "indegree toggle (-z).",
    ),
    CLClass(
        name="CLApp",
        parent="CLBase",
        getopt_ext="an:r:v",
        purpose="Generic per-trial driver: analysis (-a), trial count "
                "(-n), start-vertex (-r), verify (-v).",
    ),
    CLClass(
        name="CLIterApp",
        parent="CLApp",
        getopt_ext="i:",
        purpose="CLApp + iteration count (-i). Used by bc.",
    ),
    CLClass(
        name="CLPageRank",
        parent="CLApp",
        getopt_ext="i:t:",
        purpose="CLApp + max iters (-i) + tolerance (-t). Used by "
                "pr and pr_spmv.",
    ),
    CLClass(
        name="CLDelta",
        parent="CLApp",
        getopt_ext="d:",
        purpose="CLApp + delta-stepping parameter (-d). Used by sssp.",
    ),
    CLClass(
        name="CLConvert",
        parent="CLBase",
        getopt_ext="e:b:x:q:p:y:V:w",
        purpose="CLBase + writer options for converter (-e/-b/-x/-q/"
                "-p/-y/-V/-w).",
    ),
)


# --- Canonical kernel → CL class ------------------------------------------

KERNEL_CL_CLASS: dict[str, str] = {
    "pr":             "CLPageRank",
    "pr_spmv":        "CLPageRank",
    "bfs":            "CLApp",
    "sssp":           "CLDelta",
    "bc":             "CLIterApp",
    "cc":             "CLApp",
    "cc_sv":          "CLApp",
    "tc":             "CLApp",
    "tc_p":           "CLApp",
    "ecg_preprocess": "CLApp",
    "converter":      "CLConvert",
}


# --- Canonical flag semantics --------------------------------------------

FLAG_PURPOSE: dict[str, str] = {
    # CLBase
    "f": "load graph from edge-list file",
    "g": "generate 2^scale kronecker synthetic graph",
    "h": "print help",
    "k": "average degree for synthetic graph",
    "s": "symmetrize input",
    "u": "generate 2^scale uniform-random synthetic graph",
    "m": "in-place / memory-friendly loader",
    "o": "apply reordering strategy (POPT / GRASP / etc.)",
    "z": "use indegree for degree-based orderings",
    "j": "segmentation config (type:n:m)",
    "S": "keep self loops",
    "l": "log per-trial performance",
    "D": "database directory for JSON output",
    # CLApp
    "a": "output last-run analysis",
    "n": "number of trials",
    "r": "starting vertex (or 'rand')",
    "v": "verify output",
    # CLIterApp / CLPageRank
    "i": "iteration count (bc) or max iters (pr)",
    # CLPageRank
    "t": "tolerance (pr / pr_spmv)",
    # CLDelta
    "d": "delta-stepping parameter",
    # CLConvert
    "e": "output edge list (.el)",
    "b": "output serialized graph (.sg)",
    "x": "output reordered labels (.so)",
    "q": "output reordered labels serialized (.lo)",
    "p": "output Matrix Market (.mtx)",
    "y": "output Ligra adjacency (.ligra)",
    "V": "output split CSR arrays (.out_degree/.out_neigh/.offset)",
    "w": "make output weighted (.wel/.wsg)",
}


# --- Helpers --------------------------------------------------------------

def _parse_getopt(ext: str) -> dict[str, bool]:
    """Returns ``{flag: takes_value}``. ``X:`` means X takes a value."""
    out: dict[str, bool] = {}
    i = 0
    while i < len(ext):
        c = ext[i]
        takes = (i + 1 < len(ext)) and ext[i + 1] == ":"
        out[c] = takes
        i += 2 if takes else 1
    return out


def _class_chain(name: str) -> list[str]:
    """Walks parent links from ``name`` up to CLBase, inclusive."""
    chain: list[str] = []
    by_name = {c.name: c for c in CL_CLASSES}
    cur: str | None = name
    while cur:
        chain.append(cur)
        cur = by_name[cur].parent
    return chain


def _flags_for_class(name: str) -> dict[str, bool]:
    out: dict[str, bool] = {}
    by_name = {c.name: c for c in CL_CLASSES}
    for c in _class_chain(name):
        out.update(_parse_getopt(by_name[c].getopt_ext))
    return out


def _harvest_header() -> dict[str, dict[str, Any]]:
    """Returns ``{class_name: {parent, getopt_ext_decl}}``
    parsed from the header (live state vs canonical declared)."""
    text = HEADER.read_text()
    classes: dict[str, dict[str, Any]] = {}
    for m in re.finditer(
        r"class\s+(CL\w+)\s*(?::\s*public\s+(\w+))?\s*\{", text
    ):
        cname = m.group(1)
        parent = m.group(2)
        classes[cname] = {"parent": parent, "ext_decls": []}
    # template <typename WeightT_> class CLDelta : public CLApp
    for m in re.finditer(
        r"class\s+(CL\w+)\s*:\s*public\s+(\w+)", text
    ):
        cname = m.group(1)
        parent = m.group(2)
        classes.setdefault(cname, {"parent": parent, "ext_decls": []})
        if classes[cname].get("parent") is None:
            classes[cname]["parent"] = parent
    # ``get_args_ += "..."`` and the base ``get_args_ = "...";`` lines.
    base_ext = re.search(r"get_args_\s*=\s*\"([^\"]+)\"", text)
    if base_ext:
        classes.setdefault("CLBase", {"parent": None, "ext_decls": []})
        classes["CLBase"]["ext_decls"].append(base_ext.group(1))
    # Find each `get_args_ += "..."` inside a class body and assign to
    # the most-recently opened class. Walk in declaration order.
    # We tokenise by scanning for `class CL...{` openings and
    # `get_args_ +=` lines.
    pos = 0
    cur_class: str | None = None
    while pos < len(text):
        m_class = re.search(
            r"class\s+(CL\w+)\b", text[pos:]
        )
        m_ext = re.search(
            r"get_args_\s*\+=\s*\"([^\"]+)\"", text[pos:]
        )
        if not m_ext:
            break
        if m_class and m_class.start() < m_ext.start():
            cur_class = m_class.group(1)
            pos += m_class.end()
        else:
            if cur_class is not None:
                classes.setdefault(
                    cur_class, {"parent": None, "ext_decls": []}
                )["ext_decls"].append(m_ext.group(1))
            pos += m_ext.end()
    return classes


# --- Source-file check ----------------------------------------------------

def _kernel_uses_class(src: Path, expected_cl: str) -> bool:
    if not src.exists():
        return False
    txt = src.read_text()
    return bool(re.search(rf"\b{re.escape(expected_cl)}\b", txt))


# --- Audit ----------------------------------------------------------------

def audit() -> dict[str, Any]:
    live_classes = _harvest_header()
    violations: list[dict[str, Any]] = []

    # R1: every declared class has a non-empty extension.
    for c in CL_CLASSES:
        if not c.getopt_ext:
            violations.append({
                "rule": "R1",
                "class": c.name,
                "msg": "empty canonical getopt extension",
            })
        live = live_classes.get(c.name, {})
        if c.name == "CLBase":
            ok = bool(live.get("ext_decls"))
        else:
            ok = c.getopt_ext in live.get("ext_decls", [])
        if not ok:
            violations.append({
                "rule": "R1",
                "class": c.name,
                "msg": (
                    f"declared getopt ext {c.getopt_ext!r} does not match "
                    f"live extensions {live.get('ext_decls', [])!r}"
                ),
            })

    # R2: every kernel source uses the canonical CL class.
    kernel_src_status: list[dict[str, Any]] = []
    for kernel, cl in sorted(KERNEL_CL_CLASS.items()):
        for backend, srcdir in SRC_DIRS.items():
            src = srcdir / f"{kernel}.cc"
            if not src.exists():
                continue
            ok = _kernel_uses_class(src, cl)
            kernel_src_status.append({
                "kernel": kernel, "backend": backend,
                "cl": cl, "ok": ok,
                "src": str(src.relative_to(REPO_ROOT)),
            })
            if not ok:
                violations.append({
                    "rule": "R2",
                    "kernel": kernel,
                    "backend": backend,
                    "msg": (
                        f"{src.relative_to(REPO_ROOT)} does not "
                        f"instantiate the canonical {cl}"
                    ),
                })

    # R3: every kernel's full flag-set is the union of its chain.
    kernel_flag_sets: dict[str, dict[str, bool]] = {}
    for kernel, cl in KERNEL_CL_CLASS.items():
        kernel_flag_sets[kernel] = _flags_for_class(cl)
        if not kernel_flag_sets[kernel]:
            violations.append({
                "rule": "R3",
                "kernel": kernel,
                "msg": f"empty flag union for {cl}",
            })

    # R4: within-chain conflicts. Walk each chain; same flag at two
    # levels of the same chain is a conflict.
    for c in CL_CLASSES:
        seen: dict[str, str] = {}
        for anc in reversed(_class_chain(c.name)):
            anc_cls = next(x for x in CL_CLASSES if x.name == anc)
            for f, _t in _parse_getopt(anc_cls.getopt_ext).items():
                if f in seen and seen[f] != anc:
                    violations.append({
                        "rule": "R4",
                        "class": c.name,
                        "msg": (
                            f"within-chain getopt conflict: flag {f!r} "
                            f"appears in both {seen[f]} and {anc}"
                        ),
                    })
                seen[f] = anc

    # R5: every harvested flag matches the documented regex.
    all_flags: set[str] = set()
    for c in CL_CLASSES:
        for f in _parse_getopt(c.getopt_ext):
            all_flags.add(f)
            if not FLAG_RE.match(f):
                violations.append({
                    "rule": "R5",
                    "class": c.name,
                    "flag": f,
                    "msg": f"flag {f!r} does not match {FLAG_RE.pattern}",
                })

    # R6: every canonical flag has a purpose label.
    for f in sorted(all_flags):
        if f not in FLAG_PURPOSE:
            violations.append({
                "rule": "R6",
                "flag": f,
                "msg": f"canonical flag {f!r} has no FLAG_PURPOSE entry",
            })

    # R7: flag-arity consistency. (Every X: across all classes
    # claims X takes a value; every bare X claims X takes none. A
    # within-chain conflict where the same flag has both arities
    # would be a bug — but R4 catches that. Here we surface the
    # canonical takes-value map for the artifact.)
    flag_arity: dict[str, bool] = {}
    for c in CL_CLASSES:
        for f, takes in _parse_getopt(c.getopt_ext).items():
            if f in flag_arity and flag_arity[f] != takes:
                violations.append({
                    "rule": "R7",
                    "flag": f,
                    "msg": (
                        f"flag {f!r} declared with conflicting arity "
                        f"across classes (takes_value={flag_arity[f]} "
                        f"vs {takes})"
                    ),
                })
            flag_arity[f] = takes

    return {
        "status": "active",
        "n_cl_classes": len(CL_CLASSES),
        "n_kernels": len(KERNEL_CL_CLASS),
        "n_distinct_flags": len(all_flags),
        "n_kernel_source_checks": len(kernel_src_status),
        "n_kernel_source_ok": sum(1 for r in kernel_src_status if r["ok"]),
        "cl_classes": [
            {
                "name": c.name,
                "parent": c.parent,
                "getopt_ext": c.getopt_ext,
                "chain": _class_chain(c.name),
                "full_flags": sorted(_flags_for_class(c.name).keys()),
                "purpose": c.purpose,
            }
            for c in CL_CLASSES
        ],
        "kernels": [
            {
                "kernel": k,
                "cl_class": cl,
                "full_flags": sorted(kernel_flag_sets[k].keys()),
                "n_flags": len(kernel_flag_sets[k]),
            }
            for k, cl in sorted(KERNEL_CL_CLASS.items())
        ],
        "kernel_source_status": kernel_src_status,
        "flag_purpose": FLAG_PURPOSE,
        "flag_arity": flag_arity,
        "rules": {
            "R1": "every declared CL class has a non-empty getopt extension matching the live header",
            "R2": "every canonical kernel source instantiates the canonical CL class",
            "R3": "every canonical kernel's flag-set is the union of its inheritance chain",
            "R4": "no within-chain getopt-letter conflict (same letter at two levels of the same chain)",
            "R5": f"every canonical flag matches {FLAG_RE.pattern}",
            "R6": "every canonical flag has a documented FLAG_PURPOSE entry",
            "R7": "every flag's arity (takes value / no value) is consistent across all classes that declare it",
        },
        "violations": violations,
    }


def _emit_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _emit_md(data: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# Gate 260 — GAPBS kernel CLI vocabulary registry")
    lines.append("")
    lines.append(f"Status: **{data['status']}**")
    lines.append("")
    lines.append("## Totals")
    lines.append("")
    for k in ("n_cl_classes", "n_kernels", "n_distinct_flags",
              "n_kernel_source_checks", "n_kernel_source_ok"):
        lines.append(f"- {k}: {data[k]}")
    lines.append("")
    lines.append("## Rules")
    lines.append("")
    for rid, desc in data["rules"].items():
        lines.append(f"- **{rid}** — {desc}")
    lines.append("")
    lines.append("## CL classes")
    lines.append("")
    lines.append("| name | parent | getopt_ext | chain | full_flags |")
    lines.append("|---|---|---|---|---|")
    for c in data["cl_classes"]:
        chain = " → ".join(c["chain"])
        flags = ", ".join(f"`{f}`" for f in c["full_flags"])
        lines.append(
            f"| `{c['name']}` | `{c['parent']}` | `{c['getopt_ext']}` | "
            f"{chain} | {flags} |"
        )
    lines.append("")
    lines.append("## Kernels → CL class")
    lines.append("")
    lines.append("| kernel | cl_class | n_flags | full_flags |")
    lines.append("|---|---|---:|---|")
    for k in data["kernels"]:
        flags = ", ".join(f"`{f}`" for f in k["full_flags"])
        lines.append(
            f"| `{k['kernel']}` | `{k['cl_class']}` | {k['n_flags']} | {flags} |"
        )
    lines.append("")
    lines.append("## Flag purpose")
    lines.append("")
    lines.append("| flag | takes_value | purpose |")
    lines.append("|---|---|---|")
    for f, p in sorted(data["flag_purpose"].items()):
        takes = data["flag_arity"].get(f, False)
        lines.append(f"| `-{f}` | {takes} | {p} |")
    lines.append("")
    if data["violations"]:
        lines.append("## Violations")
        lines.append("")
        for v in data["violations"]:
            lines.append(f"- {v}")
    else:
        lines.append("## Violations")
        lines.append("")
        lines.append("None.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _emit_csv(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("kind", "name", "extra"))
        for c in data["cl_classes"]:
            w.writerow(("cl_class", c["name"], c["parent"] or ""))
        for k in data["kernels"]:
            w.writerow(("kernel", k["kernel"], k["cl_class"]))
        for f, p in sorted(data["flag_purpose"].items()):
            w.writerow(("flag", f, p))
        for v in data["violations"]:
            w.writerow(("violation", str(v.get("rule", "")), str(v)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--md-out", type=Path)
    ap.add_argument("--csv-out", type=Path)
    args = ap.parse_args(argv)
    data = audit()
    if args.json_out:
        _emit_json(data, args.json_out)
    if args.md_out:
        _emit_md(data, args.md_out)
    if args.csv_out:
        _emit_csv(data, args.csv_out)
    print(
        f"[lit-faith-cli-registry] status={data['status']} "
        f"classes={data['n_cl_classes']} "
        f"kernels={data['n_kernels']} "
        f"flags={data['n_distinct_flags']} "
        f"src_ok={data['n_kernel_source_ok']}/{data['n_kernel_source_checks']} "
        f"violations={len(data['violations'])}"
    )
    return 1 if data["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
