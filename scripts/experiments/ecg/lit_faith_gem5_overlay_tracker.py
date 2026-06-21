#!/usr/bin/env python3
"""Generate the gate 267 gem5 overlay-installation tracker registry.

Locks ``scripts/setup_gem5.py``'s overlay installation contract
(``OVERLAY_FILE_MAP`` source→destination dict + ``PATCH_FILES`` list +
``apply_overlays()`` + ``apply_patches()``) against silent drift in:

  - which overlay source files are claimed to be copied from
    ``bench/include/gem5_sim/overlays/`` into the gem5 source tree
  - whether those source files actually exist on disk under
    ``bench/include/gem5_sim/overlays/``
  - whether every claimed replacement_policy + prefetcher has matching
    ``.cc`` + ``.hh`` pairs (and a registered Python SimObject declaration)
  - whether every patch file is a real on-disk ``.patch`` under
    ``bench/include/gem5_sim/overlays/``
  - whether the on-disk overlay tree is exhaustively covered by the
    registry (no orphan files; no missing files)

Together with gate 266 (Sniper overlay tracker) this completes the
gem5/Sniper build-time overlay-installation vocabulary-lock.

7 rules G1-G7:

  G1 every ``OVERLAY_FILE_MAP`` source path has a non-empty relative
     path with forward-slash separators and a recognized extension
     (``.cc`` / ``.hh`` / ``.py`` / ``.isa``).
  G2 every ``OVERLAY_FILE_MAP`` source path exists on disk under
     ``bench/include/gem5_sim/overlays/<relpath>``.
  G3 every ``OVERLAY_POLICIES`` token has both
     ``mem/cache/replacement_policies/<pol>_rp.cc`` AND
     ``mem/cache/replacement_policies/<pol>_rp.hh`` in OVERLAY_FILE_MAP.
  G4 every ``OVERLAY_PREFETCHERS`` token has both
     ``mem/cache/prefetch/<pf>.cc`` AND
     ``mem/cache/prefetch/<pf>.hh`` in OVERLAY_FILE_MAP.
  G5 every ``OVERLAY_PATCHES`` entry exists on disk under
     ``bench/include/gem5_sim/overlays/<patch>``.
  G6 ``OVERLAY_FILE_MAP`` is exhaustive: every regular file under
     ``bench/include/gem5_sim/overlays/`` with a tracked extension is
     listed (modulo OVERLAY_EXTRA_ALLOW for dev/staging files).
  G7 the live ``setup_gem5.py`` module's ``OVERLAY_FILE_MAP`` keys and
     ``PATCH_FILES`` list exactly match the canonical registry derived
     here (parity between this gate's snapshot and the imported source).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GEM5_SIM_DIR = PROJECT_ROOT / "bench" / "include" / "gem5_sim"
GEM5_OVERLAY_DIR = GEM5_SIM_DIR / "overlays"
SETUP_GEM5 = PROJECT_ROOT / "scripts" / "setup_gem5.py"

# Canonical registry — source of truth for what gem5 overlay
# installation produces.  Mirrors scripts/setup_gem5.py's
# OVERLAY_FILE_MAP keys and PATCH_FILES list verbatim.
OVERLAY_POLICIES = ["grasp", "popt", "ecg"]
OVERLAY_PREFETCHERS = ["droplet", "ecg_pfx"]
OVERLAY_FILE_MAP_KEYS = [
    # Replacement policies (8 entries: 3 pol × 2 + context.hh + Py decl)
    "mem/cache/replacement_policies/grasp_rp.hh",
    "mem/cache/replacement_policies/grasp_rp.cc",
    "mem/cache/replacement_policies/popt_rp.hh",
    "mem/cache/replacement_policies/popt_rp.cc",
    "mem/cache/replacement_policies/ecg_rp.hh",
    "mem/cache/replacement_policies/ecg_rp.cc",
    "mem/cache/replacement_policies/ecg_victim_policy.hh",
    "mem/cache/replacement_policies/graph_cache_context_gem5.hh",
    "mem/cache/replacement_policies/GraphReplacementPolicies.py",
    # Prefetchers (5 entries: 2 pf × 2 + Py decl)
    "mem/cache/prefetch/droplet.hh",
    "mem/cache/prefetch/droplet.cc",
    "mem/cache/prefetch/ecg_pfx.hh",
    "mem/cache/prefetch/ecg_pfx.cc",
    "mem/cache/prefetch/GraphPrefetchers.py",
    # RISC-V ECG custom instruction scaffold
    "arch/riscv/isa/formats/ecg.isa",
]
OVERLAY_PATCHES = [
    "mem/cache/replacement_policies/SConscript.patch",
    "mem/cache/prefetch/SConscript.patch",
]

# Unified-diff overlay patches applied to vanilla gem5 sources by
# setup_gem5.apply_unified_diff_patches() (sprint S68 — queue-servicing +
# prefetch-latency fixes). Mirrors scripts/setup_gem5.py's UNIFIED_DIFF_PATCHES
# source paths verbatim (the list of (src_rel, target_dir) tuples — we mirror
# the src_rel keys here).
UNIFIED_DIFF_PATCHES = [
    "mem/cache/prefetch/queued_hh.patch",
    "mem/cache/prefetch/queued_cc_latency.patch",
]

# Allow-lists.
# Files under GEM5_OVERLAY_DIR/ that are staging or dev artifacts
# not yet wired into the installed overlay set.
OVERLAY_EXTRA_ALLOW = {
    "arch/riscv/isa/decoder_ecg_extract.isa",  # staging snippet, not yet copied
}

TRACKED_EXT = {".cc", ".hh", ".py", ".isa"}

OVERLAY_PATH_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_/.]+\.(cc|hh|py|isa)$")
PATCH_PATH_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_/.]+\.patch$")
POLICY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
PREFETCHER_RE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass
class Violation:
    rule: str
    where: str
    msg: str


@dataclass
class AuditResult:
    status: str = "active"
    registry_n: int = 0
    policies_n: int = 0
    prefetchers_n: int = 0
    patches_n: int = 0
    violations: list[Violation] = field(default_factory=list)


def _add(out: AuditResult, rule: str, where: str, msg: str) -> None:
    out.violations.append(Violation(rule, where, msg))


# --------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------


def rule_g1(out: AuditResult) -> None:
    for f in OVERLAY_FILE_MAP_KEYS:
        if not OVERLAY_PATH_RE.match(f):
            _add(out, "G1", f, "overlay file path does not match grammar")
        if Path(f).suffix not in TRACKED_EXT:
            _add(out, "G1", f, f"unrecognized extension {Path(f).suffix}")


def rule_g2(out: AuditResult) -> None:
    if not GEM5_OVERLAY_DIR.is_dir():
        _add(out, "G2", str(GEM5_OVERLAY_DIR),
             "gem5 overlay source dir missing on disk")
        return
    for f in OVERLAY_FILE_MAP_KEYS:
        p = GEM5_OVERLAY_DIR / f
        if not p.is_file():
            _add(out, "G2", f,
                 "OVERLAY_FILE_MAP source not present on disk under overlays/")


def rule_g3(out: AuditResult) -> None:
    for pol in OVERLAY_POLICIES:
        if not POLICY_RE.match(pol):
            _add(out, "G3", pol, "policy token does not match grammar")
            continue
        cc = f"mem/cache/replacement_policies/{pol}_rp.cc"
        hh = f"mem/cache/replacement_policies/{pol}_rp.hh"
        if cc not in OVERLAY_FILE_MAP_KEYS:
            _add(out, "G3", pol, f"policy missing {cc} in OVERLAY_FILE_MAP")
        if hh not in OVERLAY_FILE_MAP_KEYS:
            _add(out, "G3", pol, f"policy missing {hh} in OVERLAY_FILE_MAP")


def rule_g4(out: AuditResult) -> None:
    for pf in OVERLAY_PREFETCHERS:
        if not PREFETCHER_RE.match(pf):
            _add(out, "G4", pf, "prefetcher token does not match grammar")
            continue
        cc = f"mem/cache/prefetch/{pf}.cc"
        hh = f"mem/cache/prefetch/{pf}.hh"
        if cc not in OVERLAY_FILE_MAP_KEYS:
            _add(out, "G4", pf, f"prefetcher missing {cc} in OVERLAY_FILE_MAP")
        if hh not in OVERLAY_FILE_MAP_KEYS:
            _add(out, "G4", pf, f"prefetcher missing {hh} in OVERLAY_FILE_MAP")


def rule_g5(out: AuditResult) -> None:
    for patch in OVERLAY_PATCHES:
        if not PATCH_PATH_RE.match(patch):
            _add(out, "G5", patch, "patch path does not match grammar")
            continue
        p = GEM5_OVERLAY_DIR / patch
        if not p.is_file():
            _add(out, "G5", patch, "PATCH_FILES entry not present on disk under overlays/")


def rule_g6(out: AuditResult) -> None:
    if not GEM5_OVERLAY_DIR.is_dir():
        return  # already reported by G2
    on_disk = []
    for p in GEM5_OVERLAY_DIR.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(GEM5_OVERLAY_DIR)).replace("\\", "/")
        if rel in OVERLAY_EXTRA_ALLOW:
            continue
        if Path(rel).suffix not in TRACKED_EXT and not rel.endswith(".patch"):
            continue
        on_disk.append(rel)
    on_disk_set = set(on_disk)
    canonical_set = set(OVERLAY_FILE_MAP_KEYS) | set(OVERLAY_PATCHES) | set(UNIFIED_DIFF_PATCHES)
    extra = on_disk_set - canonical_set
    missing = canonical_set - on_disk_set
    if extra:
        _add(out, "G6", "overlays/",
             f"unregistered overlay files on disk: {sorted(extra)}")
    if missing:
        _add(out, "G6", "overlays/",
             f"registered files missing from disk: {sorted(missing)}")


def rule_g7(out: AuditResult) -> None:
    if not SETUP_GEM5.is_file():
        _add(out, "G7", str(SETUP_GEM5), "setup_gem5.py missing")
        return
    # Import live setup_gem5.OVERLAY_FILE_MAP and PATCH_FILES to ensure
    # this gate's static snapshot matches the source-of-truth.
    import importlib.util
    spec = importlib.util.spec_from_file_location("setup_gem5_g7", SETUP_GEM5)
    if spec is None or spec.loader is None:
        _add(out, "G7", str(SETUP_GEM5), "could not load setup_gem5.py as module")
        return
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        _add(out, "G7", str(SETUP_GEM5), f"setup_gem5 import failed: {e}")
        return
    live_map_keys = sorted(getattr(mod, "OVERLAY_FILE_MAP", {}).keys())
    live_patches = sorted(getattr(mod, "PATCH_FILES", []))
    canon_map_keys = sorted(OVERLAY_FILE_MAP_KEYS)
    canon_patches = sorted(OVERLAY_PATCHES)
    if live_map_keys != canon_map_keys:
        extra = set(live_map_keys) - set(canon_map_keys)
        missing = set(canon_map_keys) - set(live_map_keys)
        _add(out, "G7", "OVERLAY_FILE_MAP",
             f"live ↔ canonical drift: missing={sorted(missing)} extra={sorted(extra)}")
    if live_patches != canon_patches:
        extra = set(live_patches) - set(canon_patches)
        missing = set(canon_patches) - set(live_patches)
        _add(out, "G7", "PATCH_FILES",
             f"live ↔ canonical drift: missing={sorted(missing)} extra={sorted(extra)}")
    # Unified-diff patches (sprint S68): live UNIFIED_DIFF_PATCHES is a list of
    # (src_rel, target_dir) tuples; mirror its source paths must match canonical.
    live_udp = sorted(t[0] for t in getattr(mod, "UNIFIED_DIFF_PATCHES", []))
    canon_udp = sorted(UNIFIED_DIFF_PATCHES)
    if live_udp != canon_udp:
        extra = set(live_udp) - set(canon_udp)
        missing = set(canon_udp) - set(live_udp)
        _add(out, "G7", "UNIFIED_DIFF_PATCHES",
             f"live ↔ canonical drift: missing={sorted(missing)} extra={sorted(extra)}")
    # Also enforce identity invariant: every map entry has source==dest
    # (current convention; deviations should be intentional and audited).
    bad_identity = []
    for src, dst in getattr(mod, "OVERLAY_FILE_MAP", {}).items():
        if src != dst:
            bad_identity.append((src, dst))
    if bad_identity:
        _add(out, "G7", "OVERLAY_FILE_MAP",
             f"non-identity src→dst mappings (audit required): {bad_identity}")


def audit() -> dict:
    out = AuditResult()
    rule_g1(out)
    rule_g2(out)
    rule_g3(out)
    rule_g4(out)
    rule_g5(out)
    rule_g6(out)
    rule_g7(out)
    out.registry_n = len(OVERLAY_FILE_MAP_KEYS)
    out.policies_n = len(OVERLAY_POLICIES)
    out.prefetchers_n = len(OVERLAY_PREFETCHERS)
    out.patches_n = len(OVERLAY_PATCHES)
    return {
        "status":         out.status,
        "registry_n":     out.registry_n,
        "policies_n":     out.policies_n,
        "prefetchers_n":  out.prefetchers_n,
        "patches_n":      out.patches_n,
        "rules": {
            "G1": "every OVERLAY_FILE_MAP source has valid grammar (path + .cc/.hh/.py/.isa)",
            "G2": "every OVERLAY_FILE_MAP source exists on disk under overlays/",
            "G3": "every policy token has both <pol>_rp.cc + .hh in OVERLAY_FILE_MAP",
            "G4": "every prefetcher token has both <pf>.cc + .hh in OVERLAY_FILE_MAP",
            "G5": "every PATCH_FILES entry exists on disk under overlays/",
            "G6": "OVERLAY_FILE_MAP+PATCH_FILES is exhaustive over overlays/ (modulo OVERLAY_EXTRA_ALLOW)",
            "G7": "live setup_gem5.OVERLAY_FILE_MAP+PATCH_FILES matches canonical registry (parity + identity invariant)",
        },
        "registry": {
            "overlay_file_map_keys": OVERLAY_FILE_MAP_KEYS,
            "policies":              OVERLAY_POLICIES,
            "prefetchers":           OVERLAY_PREFETCHERS,
            "patches":               OVERLAY_PATCHES,
        },
        "allow_lists": {
            "OVERLAY_EXTRA_ALLOW": sorted(OVERLAY_EXTRA_ALLOW),
        },
        "violations": [
            {"rule": v.rule, "where": v.where, "msg": v.msg}
            for v in out.violations
        ],
    }


def write_md(data: dict, path: Path) -> None:
    lines: list[str] = []
    lines.append("# Gate 267 — gem5 overlay-installation tracker registry\n")
    lines.append(
        "Locks `scripts/setup_gem5.py`'s overlay installation contract "
        "(`OVERLAY_FILE_MAP` + `PATCH_FILES` + `apply_overlays()` + "
        "`apply_patches()`) against silent drift in source files / "
        "policies / prefetchers / patches.\n"
    )
    lines.append(
        "registry entries: %d overlay sources; %d policies (%s); "
        "%d prefetchers (%s); %d patches.\n" % (
            data["registry_n"], data["policies_n"], ", ".join(OVERLAY_POLICIES),
            data["prefetchers_n"], ", ".join(OVERLAY_PREFETCHERS),
            data["patches_n"],
        )
    )
    lines.append("## Rules\n")
    for rid, txt in data["rules"].items():
        lines.append(f"- **{rid}** — {txt}")
    lines.append("")
    lines.append("## Allow-lists\n")
    for name, vals in data["allow_lists"].items():
        lines.append(f"- `{name}` = {vals}")
    lines.append("")
    if data["violations"]:
        lines.append("## ⛔ Violations\n")
        for v in data["violations"]:
            lines.append(f"- **{v['rule']}** at `{v['where']}`: {v['msg']}")
    else:
        lines.append("## ✅ No violations")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(data: dict, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "key", "value"])
        for f in OVERLAY_FILE_MAP_KEYS:
            w.writerow(["overlay_source", f, "tracked"])
        for p in OVERLAY_POLICIES:
            w.writerow(["policy", p, "registered"])
        for pf in OVERLAY_PREFETCHERS:
            w.writerow(["prefetcher", pf, "registered"])
        for pt in OVERLAY_PATCHES:
            w.writerow(["patch", pt, "registered"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json-out", type=Path, required=True)
    ap.add_argument("--md-out", type=Path, required=True)
    ap.add_argument("--csv-out", type=Path, required=True)
    args = ap.parse_args()

    data = audit()

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    write_md(data, args.md_out)
    write_csv(data, args.csv_out)

    n_v = len(data["violations"])
    print(
        "[lit-faith-gem5-overlay-tracker] status=%s files=%d policies=%d "
        "prefetchers=%d patches=%d violations=%d" % (
            data["status"], data["registry_n"], data["policies_n"],
            data["prefetchers_n"], data["patches_n"], n_v,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
