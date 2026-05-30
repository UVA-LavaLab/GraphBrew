"""Gate 259 — SCons/Make build target registry.

Sixth in the vocabulary-lock series (252 SBATCH, 255 policy,
256 profile, 257 backend, 258 graph, 259 build). Locks WHICH
compile targets actually produce the binaries that the downstream
gates measure — so a contributor cannot silently rename
``bench/bin_sim/pr`` to ``bench/bin_sim/pr_cs``, swap an
``-O3`` for ``-O2`` in ``CXXFLAGS_GAP``, drop the ``-fopenmp``
that ``ecg_preprocess`` requires, or introduce a new gem5
frontend variant without an explicit canonical entry.

Catches the silent-drift cases:

* a per-row simulator binary path is hard-coded as
  ``bench/bin_sim/pr_spmv`` but the Makefile rule was renamed
  ``pr_spmv2`` → silent FileNotFoundError at sweep time,
* the GAP ``-O3`` baseline is downgraded to ``-O2`` for a
  one-off ROI experiment → every later cache-sim number is
  apples-to-oranges against the literature baseline,
* a new ``KERNELS_SNIPER`` member is added (e.g.
  ``bc_kernel_smoke``) but only one of the wrapper scripts
  knows about it → cross-tool aggregator omits it without
  warning,
* the gem5 m5ops variant gets compiled without
  ``-DNO_M5OPS`` filtered out → ROI markers no-op,
* a contributor introduces a ``bench/bin_gem5/pr_x86`` flavour
  that is neither ``base``, ``m5ops``, nor ``riscv_m5ops``.

The gate parses the **root Makefile** for the four ``KERNELS_*``
variables + four ``CXXFLAGS_*`` blocks + four ``SRC_*_DIR`` /
``BIN_*_DIR`` pairs, then cross-validates against:

  - CANONICAL_BUILD_TARGETS (the allow-list defined here),
  - the on-disk ``bench/src*/`` directory contents (every
    canonical source file must exist),
  - the documented per-backend required CXXFLAGS (every
    backend's flag string must contain its required tokens).

8 rules R1-R8:
  R1: every (backend, kernel, variant) in CANONICAL_BUILD_TARGETS
      has a matching source file under the canonical SRC_DIR.
  R2: every kernel listed in the Makefile ``KERNELS_<BACKEND>``
      variable is canonical for that backend (no silent additions).
  R3: every canonical backend has a CXXFLAGS_<BACKEND> block in
      the Makefile that contains all required flag tokens
      (e.g. ``-std=c++17`` and ``-O3`` for GAP; ``-O1``,
      ``-DNO_M5OPS`` for gem5; ``-O2``, ``-I$(SNIPER_INCLUDE)``
      for sniper).
  R4: every canonical backend maps to the canonical SRC_DIR /
      BIN_DIR path (no off-tree binaries).
  R5: every canonical kernel has a non-empty graph-algorithm
      family classification (traversal / centrality /
      shortest-path / triangle / connected-component / pagerank /
      preprocess / smoke).
  R6: every canonical backend has the documented optimisation
      level in its CXXFLAGS (``-O3`` for native+sim, ``-O1`` for
      gem5, ``-O2`` for sniper) — locks the apples-to-apples
      baseline.
  R7: every ``.cc`` file under the canonical SRC_DIRs maps to a
      canonical (backend, kernel) entry — no orphan sources
      except the documented exception list ({``converter``,
      ``ecg_preprocess`` already canonical; ``hello_roi``,
      ``sg_kernel`` documented as auxiliary on the sniper side}).
  R8: every canonical backend has a documented ROI mechanism
      (m5ops / sift / sim-callback / none) — locks the
      measurement boundary so result rows can be matched
      across backends.
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
ROOT_MAKEFILE = REPO_ROOT / "Makefile"

# Allow-list of canonical backends keyed by name.
CANONICAL_BACKENDS: frozenset[str] = frozenset({
    "native", "cache_sim", "gem5", "sniper",
})

# Documented ROI mechanism per backend (R8).
ROI_MECHANISMS: dict[str, str] = {
    "native":    "none",
    "cache_sim": "sim-callback",
    "gem5":      "m5ops",
    "sniper":    "sift",
}

# Source / bin directory each backend builds into (R4).
SRC_DIRS: dict[str, Path] = {
    "native":    REPO_ROOT / "bench" / "src",
    "cache_sim": REPO_ROOT / "bench" / "src_sim",
    "gem5":      REPO_ROOT / "bench" / "src_gem5",
    "sniper":    REPO_ROOT / "bench" / "src_sniper",
}
BIN_DIRS: dict[str, str] = {
    "native":    "bench/bin",
    "cache_sim": "bench/bin_sim",
    "gem5":      "bench/bin_gem5",
    "sniper":    "bench/bin_sniper",
}

# Per-backend Makefile variable that holds the canonical kernel
# roster. Sniper currently has no KERNELS_SNIPER list — every
# target is built on demand by the pattern rule, so we mark it
# explicitly (None) and skip R2 for it.
MAKEFILE_KERNEL_VARS: dict[str, str | None] = {
    "native":    "KERNELS",
    "cache_sim": "KERNELS_SIM",
    "gem5":      "KERNELS_GEM5",
    "sniper":    None,
}

# Per-backend CXXFLAGS variable name + required flag tokens (R3).
# Each backend's documented Makefile block MUST contain every
# string in `required_tokens`. The optimisation level in
# `opt_level` is checked separately by R6.
@dataclass(frozen=True)
class CXXFlagSpec:
    var_name: str
    required_tokens: tuple[str, ...]
    opt_level: str


CXXFLAGS_SPEC: dict[str, CXXFlagSpec] = {
    "native": CXXFlagSpec(
        var_name="CXXFLAGS_GAP",
        required_tokens=("-std=c++17", "-Wall", "-fopenmp", "-DNDEBUG"),
        opt_level="-O3",
    ),
    "cache_sim": CXXFlagSpec(
        # cache_sim re-uses CXXFLAGS = CXXFLAGS_GAP + extras.
        var_name="CXXFLAGS_GAP",
        required_tokens=("-std=c++17", "-Wall", "-fopenmp", "-DNDEBUG"),
        opt_level="-O3",
    ),
    "gem5": CXXFlagSpec(
        var_name="CXXFLAGS_GEM5",
        required_tokens=("-std=c++17", "-Wall", "-fopenmp", "-DNDEBUG", "-DNO_M5OPS"),
        opt_level="-O1",
    ),
    "sniper": CXXFlagSpec(
        var_name="CXXFLAGS_SNIPER",
        required_tokens=("-std=c++17", "-Wall", "-fopenmp", "-DNDEBUG",
                         "-I$(SNIPER_INCLUDE)"),
        opt_level="-O2",
    ),
}

# Documented kernel → graph-algorithm family map (R5). Every
# canonical kernel name MUST be in this dict.
KERNEL_FAMILY: dict[str, str] = {
    "pr":              "pagerank",
    "pr_spmv":         "pagerank",
    "bfs":             "traversal",
    "sssp":            "shortest-path",
    "cc":              "connected-component",
    "cc_sv":           "connected-component",
    "bc":              "centrality",
    "tc":              "triangle",
    "tc_p":            "triangle",
    "ecg_preprocess":  "preprocess",
    "hello_roi":       "smoke",
    "sg_kernel":       "smoke",
    "pr_kernel_smoke": "smoke",
    "bfs_kernel_smoke":"smoke",
    "sssp_kernel_smoke":"smoke",
    "converter":       "preprocess",
}


@dataclass(frozen=True)
class BuildTarget:
    """One canonical compile target.

    ``variant`` is ``"base"`` for the default rule, or
    ``"m5ops"`` / ``"riscv_m5ops"`` for the gem5 ROI variants
    produced by the suffix pattern rules.
    """
    backend: str
    kernel: str
    variant: str = "base"
    documented_aux: bool = False  # True for hello_roi / sg_kernel etc.


# Native (bench/bin/*): the production GAPBS roster.
_NATIVE_KERNELS = ("bc", "bfs", "cc", "cc_sv", "pr", "pr_spmv",
                   "sssp", "tc", "tc_p")
# cache_sim (bench/bin_sim/*): GAPBS + ecg_preprocess.
_CACHE_SIM_KERNELS = ("pr", "pr_spmv", "bfs", "bc", "cc", "cc_sv",
                      "sssp", "tc", "ecg_preprocess")
# gem5 (bench/bin_gem5/*): GAPBS roster minus tc_p (no SE-mode build).
_GEM5_KERNELS = ("pr", "pr_spmv", "bfs", "sssp", "cc", "cc_sv", "bc", "tc")
# Sniper Phase-0 production targets (the README documented smokes).
_SNIPER_PHASE0 = ("hello_roi", "pr_kernel_smoke", "bfs_kernel_smoke",
                  "sssp_kernel_smoke", "sg_kernel")
# Sniper in-flight real kernels (compile path exists; not yet wired
# into the ROI pipeline). Allowed as canonical so R7 doesn't flag
# the .cc files as orphans.
_SNIPER_INFLIGHT = ("pr", "bfs", "sssp")


def _build_canonical() -> tuple[BuildTarget, ...]:
    out: list[BuildTarget] = []
    for k in _NATIVE_KERNELS:
        out.append(BuildTarget("native", k))
    out.append(BuildTarget("native", "converter", documented_aux=True))
    for k in _CACHE_SIM_KERNELS:
        out.append(BuildTarget("cache_sim", k))
    for k in _GEM5_KERNELS:
        out.append(BuildTarget("gem5", k, "base"))
        out.append(BuildTarget("gem5", k, "m5ops"))
        out.append(BuildTarget("gem5", k, "riscv_m5ops"))
    for k in _SNIPER_PHASE0:
        out.append(BuildTarget("sniper", k, documented_aux=(k in {"hello_roi", "sg_kernel"})))
    for k in _SNIPER_INFLIGHT:
        out.append(BuildTarget("sniper", k))
    return tuple(out)


CANONICAL_BUILD_TARGETS: tuple[BuildTarget, ...] = _build_canonical()


def _harvest_makefile_var(text: str, var: str) -> list[str]:
    """Returns the whitespace-split tokens of the first assignment
    of ``var = <rhs>`` (only single-line assignments)."""
    m = re.search(rf"^{re.escape(var)}\s*=\s*(.*)$", text, re.MULTILINE)
    if not m:
        return []
    return m.group(1).strip().split()


def _harvest_cxxflags(text: str, var: str) -> str:
    """Returns the concatenation of every ``var = …`` and ``var += …``
    assignment of the same name (handles +=)."""
    out: list[str] = []
    for m in re.finditer(rf"^{re.escape(var)}\s*[+]?=\s*(.*)$", text, re.MULTILINE):
        out.append(m.group(1).strip())
    return " ".join(out)


def audit() -> dict[str, Any]:
    text = ROOT_MAKEFILE.read_text()
    violations: list[dict[str, Any]] = []

    # R2 + harvested Makefile kernels
    harvested_kernels: dict[str, list[str]] = {}
    for backend, var in MAKEFILE_KERNEL_VARS.items():
        if var is None:
            harvested_kernels[backend] = []
            continue
        harvested_kernels[backend] = _harvest_makefile_var(text, var)

    canonical_by_backend: dict[str, set[str]] = {}
    for t in CANONICAL_BUILD_TARGETS:
        canonical_by_backend.setdefault(t.backend, set()).add(t.kernel)

    # R1: every (backend, kernel) source exists.
    src_present: list[tuple[str, str, bool]] = []
    for t in CANONICAL_BUILD_TARGETS:
        if t.variant != "base":
            # variants reuse the base .cc (gem5 m5ops/riscv_m5ops are
            # the same source compiled with extra flags + libm5)
            continue
        src = SRC_DIRS[t.backend] / f"{t.kernel}.cc"
        ok = src.exists()
        src_present.append((t.backend, t.kernel, ok))
        if not ok:
            violations.append({
                "rule": "R1",
                "backend": t.backend,
                "kernel": t.kernel,
                "msg": f"canonical source {src.relative_to(REPO_ROOT)} not on disk",
            })

    # R2: every Makefile-declared kernel is canonical for its backend.
    for backend, kernels in harvested_kernels.items():
        for k in kernels:
            if k not in canonical_by_backend.get(backend, set()):
                violations.append({
                    "rule": "R2",
                    "backend": backend,
                    "kernel": k,
                    "msg": f"Makefile {MAKEFILE_KERNEL_VARS[backend]} lists {k!r} but it is not canonical for {backend}",
                })

    # R3: every backend has the required CXXFLAGS tokens.
    cxx_strings: dict[str, str] = {}
    for backend, spec in CXXFLAGS_SPEC.items():
        flags = _harvest_cxxflags(text, spec.var_name)
        cxx_strings[backend] = flags
        for tok in spec.required_tokens:
            if tok not in flags:
                violations.append({
                    "rule": "R3",
                    "backend": backend,
                    "msg": f"{spec.var_name} missing required token {tok!r}",
                })

    # R4: SRC_DIR / BIN_DIR canonical paths exist in the Makefile.
    for backend, src in SRC_DIRS.items():
        if not src.exists():
            violations.append({
                "rule": "R4",
                "backend": backend,
                "msg": f"canonical SRC_DIR {src.relative_to(REPO_ROOT)} missing on disk",
            })
        bin_path = BIN_DIRS[backend]
        # Look for the var = <path> assignment that resolves to BIN_DIRS[backend].
        # native uses $(BENCH_DIR)/bin, sim uses $(BENCH_DIR)/bin_sim, etc.
        suffix = bin_path.split("/", 1)[1]  # e.g. "bin", "bin_sim"
        if f"$(BENCH_DIR)/{suffix}" not in text:
            violations.append({
                "rule": "R4",
                "backend": backend,
                "msg": f"Makefile has no $(BENCH_DIR)/{suffix} assignment for {backend}",
            })

    # R5: every canonical kernel has a documented family.
    for t in CANONICAL_BUILD_TARGETS:
        if t.kernel not in KERNEL_FAMILY:
            violations.append({
                "rule": "R5",
                "backend": t.backend,
                "kernel": t.kernel,
                "msg": f"canonical kernel {t.kernel!r} missing entry in KERNEL_FAMILY",
            })

    # R6: optimisation level matches per backend.
    for backend, spec in CXXFLAGS_SPEC.items():
        flags = cxx_strings[backend]
        if spec.opt_level not in flags:
            violations.append({
                "rule": "R6",
                "backend": backend,
                "msg": f"{spec.var_name} missing required opt level {spec.opt_level!r}",
            })

    # R7: every .cc under canonical SRC_DIRs maps to a canonical kernel.
    orphan_sources: list[str] = []
    for backend, src_dir in SRC_DIRS.items():
        if not src_dir.exists():
            continue
        for cc in sorted(src_dir.glob("*.cc")):
            kernel = cc.stem
            if kernel not in canonical_by_backend.get(backend, set()):
                rel = str(cc.relative_to(REPO_ROOT))
                orphan_sources.append(rel)
                violations.append({
                    "rule": "R7",
                    "backend": backend,
                    "kernel": kernel,
                    "msg": f"orphan source {rel} has no canonical (backend,kernel) entry",
                })

    # R8: every canonical backend has a documented ROI mechanism.
    for backend in CANONICAL_BACKENDS:
        m = ROI_MECHANISMS.get(backend)
        if m not in {"m5ops", "sift", "sim-callback", "none"}:
            violations.append({
                "rule": "R8",
                "backend": backend,
                "msg": f"backend {backend} has undocumented ROI mechanism {m!r}",
            })

    return {
        "status": "active",
        "n_backends": len(CANONICAL_BACKENDS),
        "n_canonical_targets": len(CANONICAL_BUILD_TARGETS),
        "n_native_kernels": len(_NATIVE_KERNELS),
        "n_cache_sim_kernels": len(_CACHE_SIM_KERNELS),
        "n_gem5_kernels": len(_GEM5_KERNELS),
        "n_sniper_targets": len(_SNIPER_PHASE0) + len(_SNIPER_INFLIGHT),
        "n_makefile_harvested_kernels": sum(len(v) for v in harvested_kernels.values()),
        "n_orphan_sources": len(orphan_sources),
        "canonical_targets": [
            {
                "backend": t.backend,
                "kernel": t.kernel,
                "variant": t.variant,
                "family": KERNEL_FAMILY.get(t.kernel, ""),
                "documented_aux": t.documented_aux,
                "src": str((SRC_DIRS[t.backend] / f"{t.kernel}.cc").relative_to(REPO_ROOT))
                       if t.variant == "base" else "",
                "bin": f"{BIN_DIRS[t.backend]}/{t.kernel}"
                       + ("" if t.variant == "base" else f"_{t.variant}"),
                "roi_mechanism": ROI_MECHANISMS[t.backend],
            }
            for t in CANONICAL_BUILD_TARGETS
        ],
        "makefile_harvested_kernels": {
            backend: kernels for backend, kernels in harvested_kernels.items()
        },
        "cxxflags": {
            backend: {
                "var_name": spec.var_name,
                "opt_level": spec.opt_level,
                "required_tokens": list(spec.required_tokens),
                "live_flags": cxx_strings[backend],
            }
            for backend, spec in CXXFLAGS_SPEC.items()
        },
        "src_dirs": {b: str(p.relative_to(REPO_ROOT)) for b, p in SRC_DIRS.items()},
        "bin_dirs": dict(BIN_DIRS),
        "roi_mechanisms": dict(ROI_MECHANISMS),
        "rules": {
            "R1": "every (backend,kernel) in CANONICAL_BUILD_TARGETS has a matching source file under SRC_DIRS[backend]",
            "R2": "every Makefile KERNELS_<BACKEND> entry is canonical for that backend",
            "R3": "every CXXFLAGS_<BACKEND> contains all documented required tokens",
            "R4": "every canonical backend maps to the canonical SRC_DIR + BIN_DIR (both must exist in tree and Makefile)",
            "R5": "every canonical kernel has a non-empty graph-algorithm family classification",
            "R6": "every backend's CXXFLAGS carries the documented optimisation level (-O3 native+sim, -O1 gem5, -O2 sniper)",
            "R7": "every .cc under SRC_DIRS[backend] maps to a canonical (backend,kernel) entry — no orphans",
            "R8": "every canonical backend has a documented ROI mechanism ∈ {m5ops, sift, sim-callback, none}",
        },
        "violations": violations,
    }


def _emit_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _emit_md(data: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# Gate 259 — SCons/Make build target registry")
    lines.append("")
    lines.append(f"Status: **{data['status']}**")
    lines.append("")
    lines.append("## Totals")
    lines.append("")
    for k in ("n_backends", "n_canonical_targets", "n_native_kernels",
              "n_cache_sim_kernels", "n_gem5_kernels", "n_sniper_targets",
              "n_makefile_harvested_kernels", "n_orphan_sources"):
        lines.append(f"- {k}: {data[k]}")
    lines.append("")
    lines.append("## Rules")
    lines.append("")
    for rid, desc in data["rules"].items():
        lines.append(f"- **{rid}** — {desc}")
    lines.append("")
    lines.append("## Canonical build targets")
    lines.append("")
    lines.append("| backend | kernel | variant | family | bin |")
    lines.append("|---|---|---|---|---|")
    for t in data["canonical_targets"]:
        lines.append(
            f"| `{t['backend']}` | `{t['kernel']}` | `{t['variant']}` | "
            f"{t['family']} | `{t['bin']}` |"
        )
    lines.append("")
    lines.append("## CXXFLAGS")
    lines.append("")
    lines.append("| backend | var_name | opt_level | required_tokens |")
    lines.append("|---|---|---|---|")
    for b, f in data["cxxflags"].items():
        toks = ", ".join(f"`{t}`" for t in f["required_tokens"])
        lines.append(
            f"| `{b}` | `{f['var_name']}` | `{f['opt_level']}` | {toks} |"
        )
    lines.append("")
    lines.append("## SRC / BIN directories")
    lines.append("")
    lines.append("| backend | src_dir | bin_dir | roi_mechanism |")
    lines.append("|---|---|---|---|")
    for b in sorted(data["src_dirs"]):
        lines.append(
            f"| `{b}` | `{data['src_dirs'][b]}` | `{data['bin_dirs'][b]}` | "
            f"`{data['roi_mechanisms'][b]}` |"
        )
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
        for t in data["canonical_targets"]:
            w.writerow(("target", f"{t['backend']}:{t['kernel']}:{t['variant']}", t["family"]))
        for b, f_spec in data["cxxflags"].items():
            w.writerow(("cxxflags", b, f_spec["var_name"] + " " + f_spec["opt_level"]))
        for b in sorted(data["src_dirs"]):
            w.writerow(("dirs", b, data["src_dirs"][b] + " -> " + data["bin_dirs"][b]))
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
        f"[lit-faith-build-registry] status={data['status']} "
        f"backends={data['n_backends']} "
        f"targets={data['n_canonical_targets']} "
        f"harvested={data['n_makefile_harvested_kernels']} "
        f"orphans={data['n_orphan_sources']} "
        f"violations={len(data['violations'])}"
    )
    return 1 if data["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
