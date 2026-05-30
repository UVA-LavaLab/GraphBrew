#!/usr/bin/env python3
"""Generate the gate 266 Sniper overlay-installation tracker registry.

Locks ``scripts/setup_sniper.py``'s overlay installation contract
(``.sniper_overlays.json`` + ``write_overlay_status`` + ``copy_overlay_sources``
+ patch functions) against silent drift in:

  - which source files are claimed to be copied
  - whether those source files actually exist under
    ``bench/include/sniper_sim/overlays/``
  - whether every claimed policy/prefetcher has matching ``.cc`` + ``.h``
    pairs in the copied_files list
  - whether every claimed patch token corresponds to a real
    ``patch_<name>_overlay`` function in ``setup_sniper.py``
  - whether the on-disk ``.sniper_overlays.json`` matches the canonical
    registry derived from the live source tree.

7 rules O1-O7:

  O1 every ``copied_files`` entry has a non-empty relative path with
     forward-slash separators and a recognized C++ extension (``.cc`` or
     ``.h``).
  O2 every ``copied_files`` entry exists on disk under
     ``bench/include/sniper_sim/overlays/<relpath>``.
  O3 every ``policies`` token (e.g. grasp/popt/ecg) has both
     ``common/core/memory_subsystem/cache/cache_set_<policy>.cc`` AND
     ``cache_set_<policy>.h`` in ``copied_files``.
  O4 every ``prefetchers`` token has both
     ``common/core/memory_subsystem/parametric_dram_directory_msi/<pf>_prefetcher.cc``
     AND ``<pf>_prefetcher.h`` in ``copied_files``.
  O5 every ``patches`` token corresponds to either a function
     ``patch_<token>_overlay`` in setup_sniper.py OR is in the documented
     PATCH_NON_FUNCTION_ALLOW allow-list of bundled patches (callable
     under a different name).
  O6 the on-disk ``.sniper_overlays.json`` file's ``copied_files``,
     ``policies``, ``prefetchers`` and ``patches`` lists match the
     canonical registry exactly (sorted set equality).
  O7 ``copied_files`` is exhaustive: every regular file under
     ``bench/include/sniper_sim/overlays/`` with a tracked extension is
     listed (modulo OVERLAY_README_ALLOW for ``README.md`` files).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SNIPER_SIM_DIR = PROJECT_ROOT / "bench" / "include" / "sniper_sim"
SNIPER_OVERLAY_DIR = SNIPER_SIM_DIR / "overlays"
OVERLAY_STATUS_FILE = SNIPER_SIM_DIR / ".sniper_overlays.json"
SETUP_SNIPER = PROJECT_ROOT / "scripts" / "setup_sniper.py"

# Canonical registry — source of truth for what Sniper overlay
# installation produces.
OVERLAY_POLICIES = ["grasp", "popt", "ecg"]
OVERLAY_PREFETCHERS = ["droplet", "ecg_pfx"]
OVERLAY_PATCHES = [
    "cache_base_replacement_policy_grasp",
    "cache_set_factory_grasp_popt_ecg",
    "cache_insert_prepare_insertion",
    "prefetcher_factory_droplet",
    "magic_user_graphbrew_hints",
]
OVERLAY_COPIED_FILES = [
    "common/core/memory_subsystem/cache/cache_set_ecg.cc",
    "common/core/memory_subsystem/cache/cache_set_ecg.h",
    "common/core/memory_subsystem/cache/cache_set_grasp.cc",
    "common/core/memory_subsystem/cache/cache_set_grasp.h",
    "common/core/memory_subsystem/cache/cache_set_popt.cc",
    "common/core/memory_subsystem/cache/cache_set_popt.h",
    "common/core/memory_subsystem/cache/graph_cache_context_sniper.cc",
    "common/core/memory_subsystem/cache/graph_cache_context_sniper.h",
    "common/core/memory_subsystem/parametric_dram_directory_msi/droplet_prefetcher.cc",
    "common/core/memory_subsystem/parametric_dram_directory_msi/droplet_prefetcher.h",
    "common/core/memory_subsystem/parametric_dram_directory_msi/ecg_pfx_prefetcher.cc",
    "common/core/memory_subsystem/parametric_dram_directory_msi/ecg_pfx_prefetcher.h",
]

# Allow-lists.
# Patches that exist as standalone tokens in the registry but whose
# implementation is bundled in another patch_*_overlay() function or
# applied implicitly via the apply_overlays() driver.
PATCH_NON_FUNCTION_ALLOW = {
    "cache_set_factory_grasp_popt_ecg",  # bundled in patch_grasp_overlay / patch_popt_overlay
    "cache_insert_prepare_insertion",    # bundled in patch_grasp_overlay
    "prefetcher_factory_droplet",        # bundled in patch_droplet_overlay
    "magic_user_graphbrew_hints",        # patch_graphbrew_simuser_overlay
}

# Files under SNIPER_OVERLAY_DIR/ that are documentation, not copied
# overlay source files.
OVERLAY_README_ALLOW = {"README.md"}

TRACKED_EXT = {".cc", ".h"}

COPIED_FILE_RE = re.compile(r"^[a-z][a-z0-9_/.]+\.(cc|h)$")
POLICY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
PREFETCHER_RE = re.compile(r"^[a-z][a-z0-9_]*$")
PATCH_RE = re.compile(r"^[a-z][a-z0-9_]*$")

PATCH_FN_RE = re.compile(r"^def patch_([a-z][a-z0-9_]*)_overlay\(", re.MULTILINE)


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


def rule_o1(out: AuditResult) -> None:
    for f in OVERLAY_COPIED_FILES:
        if not COPIED_FILE_RE.match(f):
            _add(out, "O1", f, "copied_files entry does not match grammar")
        if Path(f).suffix not in TRACKED_EXT:
            _add(out, "O1", f, f"unrecognized extension {Path(f).suffix}")


def rule_o2(out: AuditResult) -> None:
    if not SNIPER_OVERLAY_DIR.is_dir():
        _add(out, "O2", str(SNIPER_OVERLAY_DIR),
             "overlay source dir missing on disk")
        return
    for f in OVERLAY_COPIED_FILES:
        p = SNIPER_OVERLAY_DIR / f
        if not p.is_file():
            _add(out, "O2", f, f"copied_files entry not present on disk under overlays/")


def rule_o3(out: AuditResult) -> None:
    for pol in OVERLAY_POLICIES:
        if not POLICY_RE.match(pol):
            _add(out, "O3", pol, "policy token does not match grammar")
            continue
        cc = f"common/core/memory_subsystem/cache/cache_set_{pol}.cc"
        hh = f"common/core/memory_subsystem/cache/cache_set_{pol}.h"
        if cc not in OVERLAY_COPIED_FILES:
            _add(out, "O3", pol, f"policy missing {cc} in copied_files")
        if hh not in OVERLAY_COPIED_FILES:
            _add(out, "O3", pol, f"policy missing {hh} in copied_files")


def rule_o4(out: AuditResult) -> None:
    for pf in OVERLAY_PREFETCHERS:
        if not PREFETCHER_RE.match(pf):
            _add(out, "O4", pf, "prefetcher token does not match grammar")
            continue
        cc = f"common/core/memory_subsystem/parametric_dram_directory_msi/{pf}_prefetcher.cc"
        hh = f"common/core/memory_subsystem/parametric_dram_directory_msi/{pf}_prefetcher.h"
        if cc not in OVERLAY_COPIED_FILES:
            _add(out, "O4", pf, f"prefetcher missing {cc} in copied_files")
        if hh not in OVERLAY_COPIED_FILES:
            _add(out, "O4", pf, f"prefetcher missing {hh} in copied_files")


def rule_o5(out: AuditResult) -> None:
    if not SETUP_SNIPER.is_file():
        _add(out, "O5", str(SETUP_SNIPER), "setup_sniper.py missing")
        return
    text = SETUP_SNIPER.read_text(encoding="utf-8")
    fn_names = set(PATCH_FN_RE.findall(text))
    for patch in OVERLAY_PATCHES:
        if not PATCH_RE.match(patch):
            _add(out, "O5", patch, "patch token does not match grammar")
            continue
        if patch in PATCH_NON_FUNCTION_ALLOW:
            continue
        # patch token must equal one of the function-name stems OR be a
        # prefix of one (e.g. "cache_base_replacement_policy_grasp"
        # implemented as patch_grasp_overlay).
        if not any(patch == fn or patch.endswith(fn) for fn in fn_names):
            _add(out, "O5", patch,
                 f"patch token has no matching patch_*_overlay function (and not in allow-list)")


def rule_o6(out: AuditResult) -> None:
    if not OVERLAY_STATUS_FILE.is_file():
        _add(out, "O6", str(OVERLAY_STATUS_FILE),
             "overlay status file missing on disk")
        return
    try:
        data = json.loads(OVERLAY_STATUS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        _add(out, "O6", str(OVERLAY_STATUS_FILE), f"invalid JSON: {e}")
        return
    on_disk = {
        "copied_files": sorted(data.get("copied_files", [])),
        "policies":     sorted(data.get("policies", [])),
        "prefetchers":  sorted(data.get("prefetchers", [])),
        "patches":      sorted(data.get("patches", [])),
    }
    canonical = {
        "copied_files": sorted(OVERLAY_COPIED_FILES),
        "policies":     sorted(OVERLAY_POLICIES),
        "prefetchers":  sorted(OVERLAY_PREFETCHERS),
        "patches":      sorted(OVERLAY_PATCHES),
    }
    for key in canonical:
        if on_disk[key] != canonical[key]:
            extra = set(on_disk[key]) - set(canonical[key])
            missing = set(canonical[key]) - set(on_disk[key])
            _add(out, "O6", key,
                 f"on-disk {key} drift: missing={sorted(missing)} extra={sorted(extra)}")


def rule_o7(out: AuditResult) -> None:
    if not SNIPER_OVERLAY_DIR.is_dir():
        return  # already reported by O2
    on_disk = []
    for p in SNIPER_OVERLAY_DIR.rglob("*"):
        if not p.is_file():
            continue
        if p.name in OVERLAY_README_ALLOW:
            continue
        rel = str(p.relative_to(SNIPER_OVERLAY_DIR)).replace("\\", "/")
        on_disk.append(rel)
    on_disk_set = set(on_disk)
    canonical_set = set(OVERLAY_COPIED_FILES)
    extra = on_disk_set - canonical_set
    missing = canonical_set - on_disk_set
    if extra:
        _add(out, "O7", "overlays/",
             f"unregistered overlay files on disk: {sorted(extra)}")
    if missing:
        _add(out, "O7", "overlays/",
             f"registered copied_files missing from disk: {sorted(missing)}")


def audit() -> dict:
    out = AuditResult()
    rule_o1(out)
    rule_o2(out)
    rule_o3(out)
    rule_o4(out)
    rule_o5(out)
    rule_o6(out)
    rule_o7(out)
    out.registry_n = len(OVERLAY_COPIED_FILES)
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
            "O1": "every copied_files entry has valid grammar (lower_snake_case path + .cc/.h)",
            "O2": "every copied_files entry exists on disk under overlays/",
            "O3": "every policy token has both cache_set_<pol>.cc + .h in copied_files",
            "O4": "every prefetcher token has both <pf>_prefetcher.cc + .h in copied_files",
            "O5": "every patches token has a patch_<token>_overlay function OR is in PATCH_NON_FUNCTION_ALLOW",
            "O6": "on-disk .sniper_overlays.json matches canonical registry (copied_files+policies+prefetchers+patches)",
            "O7": "copied_files is exhaustive: every regular file under overlays/ with tracked extension is listed (modulo README allow-list)",
        },
        "registry": {
            "copied_files":   OVERLAY_COPIED_FILES,
            "policies":       OVERLAY_POLICIES,
            "prefetchers":    OVERLAY_PREFETCHERS,
            "patches":        OVERLAY_PATCHES,
        },
        "allow_lists": {
            "PATCH_NON_FUNCTION_ALLOW": sorted(PATCH_NON_FUNCTION_ALLOW),
            "OVERLAY_README_ALLOW":     sorted(OVERLAY_README_ALLOW),
        },
        "violations": [
            {"rule": v.rule, "where": v.where, "msg": v.msg}
            for v in out.violations
        ],
    }


def write_md(data: dict, path: Path) -> None:
    lines: list[str] = []
    lines.append("# Gate 266 — Sniper overlay-installation tracker registry\n")
    lines.append(
        "Locks `scripts/setup_sniper.py`'s overlay installation contract "
        "(`.sniper_overlays.json` + `write_overlay_status` + "
        "`copy_overlay_sources` + `patch_*_overlay` functions) against "
        "silent drift in copied_files / policies / prefetchers / patches.\n"
    )
    lines.append(
        "registry entries: %d copied files; %d policies (%s); "
        "%d prefetchers (%s); %d patches (%s).\n" % (
            data["registry_n"], data["policies_n"], ", ".join(OVERLAY_POLICIES),
            data["prefetchers_n"], ", ".join(OVERLAY_PREFETCHERS),
            data["patches_n"], ", ".join(OVERLAY_PATCHES),
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
        for f in OVERLAY_COPIED_FILES:
            w.writerow(["copied_file", f, "tracked"])
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
        "[lit-faith-overlay-tracker] status=%s files=%d policies=%d "
        "prefetchers=%d patches=%d violations=%d" % (
            data["status"], data["registry_n"], data["policies_n"],
            data["prefetchers_n"], data["patches_n"], n_v,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
