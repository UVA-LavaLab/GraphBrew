#!/usr/bin/env python3
"""Generate the gate 268 setup-script invariant registry.

Locks ``scripts/setup_gem5.py`` and ``scripts/setup_sniper.py`` against
silent drift in the public installation contract:

  - the upstream git URL each script clones from
  - the directory-constant skeleton (SCRIPT_DIR, PROJECT_ROOT, *_SIM_DIR)
  - the canonical top-level function inventory each script exposes
  - the presence of ``def main(`` and ``def apply_overlays(`` in both
  - exhaustive coverage: no unregistered top-level functions in either
    script (modulo SETUP_GEM5_EXTRA_ALLOW / SETUP_SNIPER_EXTRA_ALLOW)

7 rules S1-S7:

  S1 ``GEM5_REPO_URL`` and ``SNIPER_REPO_URL`` constants exist in their
     respective scripts and equal their canonical pinned values.
  S2 every required directory constant
     (SCRIPT_DIR / PROJECT_ROOT / SIM_DIR / *_DIR / OVERLAYS_DIR) is
     present in each script.
  S3 every canonical gem5 entry-point function is present in
     ``setup_gem5.py``.
  S4 every canonical sniper entry-point function is present in
     ``setup_sniper.py``.
  S5 both scripts define ``def main(`` (CLI entry point invariant).
  S6 both scripts define ``def apply_overlays(`` (overlay-install contract).
  S7 each script's actual top-level ``def`` set equals the canonical
     registry exactly (no unregistered helpers; missing ones flagged too).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SETUP_GEM5 = PROJECT_ROOT / "scripts" / "setup_gem5.py"
SETUP_SNIPER = PROJECT_ROOT / "scripts" / "setup_sniper.py"

# Canonical pinned repository URLs.
SETUP_REPO_URLS = {
    "GEM5_REPO_URL":   "https://github.com/gem5/gem5.git",
    "SNIPER_REPO_URL": "https://github.com/snipersim/snipersim.git",
}

# Required directory constants in each script.
SETUP_GEM5_DIR_CONSTANTS = [
    "SCRIPT_DIR",
    "PROJECT_ROOT",
    "BENCH_DIR",
    "GEM5_SIM_DIR",
    "GEM5_DIR",
    "OVERLAYS_DIR",
]
SETUP_SNIPER_DIR_CONSTANTS = [
    "SCRIPT_DIR",
    "PROJECT_ROOT",
    "SNIPER_SIM_DIR",
    "SNIPER_DIR",
    "SNIPER_OVERLAY_DIR",
    "SNIPER_CONFIG_DIR",
]

# Canonical top-level function inventory for each script. Adding or
# removing a function requires registry update (vocabulary lock).
SETUP_GEM5_FUNCTIONS = [
    "run_cmd",
    "check_prerequisites",
    "clone_gem5",
    "apply_overlays",
    "apply_patches",
    "apply_current_vertex_pseudo_inst_patch",
    "insert_once",
    "apply_riscv_ecg_extract_patch",
    "build_gem5",
    "verify_build",
    "install_riscv_toolchain",
    "clean_gem5",
    "print_summary",
    "main",
]
SETUP_SNIPER_FUNCTIONS = [
    "utc_now",
    "command_text",
    "run_cmd",
    "git_head",
    "clone_or_update",
    "write_version",
    "build_sniper",
    "replace_once",
    "overlay_source_files",
    "copy_overlay_sources",
    "install_graphbrew_configs",
    "write_overlay_status",
    "patch_grasp_overlay",
    "patch_popt_overlay",
    "patch_ecg_overlay",
    "patch_droplet_overlay",
    "patch_graphbrew_simuser_overlay",
    "patch_ecg_pfx_prefetcher_overlay",
    "apply_overlays",
    "compiler_for_checks",
    "header_available",
    "check_host_dependencies",
    "smoke_test",
    "graphbrew_smoke_test",
    "clean",
    "parse_args",
    "main",
]

# Allow-lists for top-level helpers added in the future but not yet
# admitted to the canonical function set. Empty by default — any drift
# must be reviewed.
SETUP_GEM5_EXTRA_ALLOW: set[str] = set()
SETUP_SNIPER_EXTRA_ALLOW: set[str] = set()

DEF_RE = re.compile(r"^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", re.MULTILINE)
CONST_ASSIGN_RE_TMPL = r"^{}\s*="
URL_ASSIGN_RE_TMPL = r'^{}\s*=\s*["\']([^"\']+)["\']'


@dataclass
class Violation:
    rule: str
    where: str
    msg: str


@dataclass
class AuditResult:
    status: str = "active"
    gem5_functions_n: int = 0
    sniper_functions_n: int = 0
    gem5_constants_n: int = 0
    sniper_constants_n: int = 0
    repo_urls_n: int = 0
    violations: list[Violation] = field(default_factory=list)


def _add(out: AuditResult, rule: str, where: str, msg: str) -> None:
    out.violations.append(Violation(rule, where, msg))


def _read(path: Path) -> str | None:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------


def rule_s1(out: AuditResult) -> None:
    for path, label, expected_name in (
        (SETUP_GEM5,   "setup_gem5.py",   "GEM5_REPO_URL"),
        (SETUP_SNIPER, "setup_sniper.py", "SNIPER_REPO_URL"),
    ):
        text = _read(path)
        if text is None:
            _add(out, "S1", str(path), f"{label} missing")
            continue
        canonical = SETUP_REPO_URLS[expected_name]
        m = re.search(URL_ASSIGN_RE_TMPL.format(expected_name), text, re.MULTILINE)
        if m is None:
            _add(out, "S1", expected_name,
                 f"{expected_name} constant not found in {label}")
            continue
        actual = m.group(1)
        if actual != canonical:
            _add(out, "S1", expected_name,
                 f"{expected_name} drift: expected={canonical} actual={actual}")


def rule_s2(out: AuditResult) -> None:
    for path, label, consts in (
        (SETUP_GEM5,   "setup_gem5.py",   SETUP_GEM5_DIR_CONSTANTS),
        (SETUP_SNIPER, "setup_sniper.py", SETUP_SNIPER_DIR_CONSTANTS),
    ):
        text = _read(path)
        if text is None:
            _add(out, "S2", str(path), f"{label} missing")
            continue
        for name in consts:
            if not re.search(CONST_ASSIGN_RE_TMPL.format(name), text, re.MULTILINE):
                _add(out, "S2", name,
                     f"required directory constant {name} missing from {label}")


def rule_s3(out: AuditResult) -> None:
    text = _read(SETUP_GEM5)
    if text is None:
        _add(out, "S3", str(SETUP_GEM5), "setup_gem5.py missing")
        return
    found = set(DEF_RE.findall(text))
    for fn in SETUP_GEM5_FUNCTIONS:
        if fn not in found:
            _add(out, "S3", fn,
                 f"required gem5 entry-point function {fn!r} missing from setup_gem5.py")


def rule_s4(out: AuditResult) -> None:
    text = _read(SETUP_SNIPER)
    if text is None:
        _add(out, "S4", str(SETUP_SNIPER), "setup_sniper.py missing")
        return
    found = set(DEF_RE.findall(text))
    for fn in SETUP_SNIPER_FUNCTIONS:
        if fn not in found:
            _add(out, "S4", fn,
                 f"required sniper entry-point function {fn!r} missing from setup_sniper.py")


def rule_s5(out: AuditResult) -> None:
    for path, label in (
        (SETUP_GEM5,   "setup_gem5.py"),
        (SETUP_SNIPER, "setup_sniper.py"),
    ):
        text = _read(path)
        if text is None:
            _add(out, "S5", str(path), f"{label} missing")
            continue
        if "main" not in set(DEF_RE.findall(text)):
            _add(out, "S5", label, f"{label} has no def main( — CLI entry-point invariant broken")


def rule_s6(out: AuditResult) -> None:
    for path, label in (
        (SETUP_GEM5,   "setup_gem5.py"),
        (SETUP_SNIPER, "setup_sniper.py"),
    ):
        text = _read(path)
        if text is None:
            _add(out, "S6", str(path), f"{label} missing")
            continue
        if "apply_overlays" not in set(DEF_RE.findall(text)):
            _add(out, "S6", label,
                 f"{label} has no def apply_overlays( — overlay-install contract broken")


def rule_s7(out: AuditResult) -> None:
    for path, label, canonical, allow in (
        (SETUP_GEM5,   "setup_gem5.py",   SETUP_GEM5_FUNCTIONS,   SETUP_GEM5_EXTRA_ALLOW),
        (SETUP_SNIPER, "setup_sniper.py", SETUP_SNIPER_FUNCTIONS, SETUP_SNIPER_EXTRA_ALLOW),
    ):
        text = _read(path)
        if text is None:
            continue  # already reported by S3/S4
        found = set(DEF_RE.findall(text))
        canonical_set = set(canonical)
        extra = found - canonical_set - allow
        missing = canonical_set - found
        if extra:
            _add(out, "S7", label,
                 f"unregistered top-level functions in {label}: {sorted(extra)}")
        if missing:
            _add(out, "S7", label,
                 f"canonical functions missing from {label}: {sorted(missing)}")


def audit() -> dict:
    out = AuditResult()
    rule_s1(out)
    rule_s2(out)
    rule_s3(out)
    rule_s4(out)
    rule_s5(out)
    rule_s6(out)
    rule_s7(out)
    out.gem5_functions_n = len(SETUP_GEM5_FUNCTIONS)
    out.sniper_functions_n = len(SETUP_SNIPER_FUNCTIONS)
    out.gem5_constants_n = len(SETUP_GEM5_DIR_CONSTANTS)
    out.sniper_constants_n = len(SETUP_SNIPER_DIR_CONSTANTS)
    out.repo_urls_n = len(SETUP_REPO_URLS)
    return {
        "status":             out.status,
        "gem5_functions_n":   out.gem5_functions_n,
        "sniper_functions_n": out.sniper_functions_n,
        "gem5_constants_n":   out.gem5_constants_n,
        "sniper_constants_n": out.sniper_constants_n,
        "repo_urls_n":        out.repo_urls_n,
        "rules": {
            "S1": "GEM5_REPO_URL and SNIPER_REPO_URL constants exist and equal canonical values",
            "S2": "every required directory constant is present in each script",
            "S3": "every canonical gem5 entry-point function is present in setup_gem5.py",
            "S4": "every canonical sniper entry-point function is present in setup_sniper.py",
            "S5": "both scripts define def main( (CLI entry point invariant)",
            "S6": "both scripts define def apply_overlays( (overlay-install contract)",
            "S7": "actual top-level def set equals canonical registry exactly (no unregistered helpers)",
        },
        "registry": {
            "repo_urls":         SETUP_REPO_URLS,
            "gem5_dir_consts":   SETUP_GEM5_DIR_CONSTANTS,
            "sniper_dir_consts": SETUP_SNIPER_DIR_CONSTANTS,
            "gem5_functions":    SETUP_GEM5_FUNCTIONS,
            "sniper_functions":  SETUP_SNIPER_FUNCTIONS,
        },
        "allow_lists": {
            "SETUP_GEM5_EXTRA_ALLOW":   sorted(SETUP_GEM5_EXTRA_ALLOW),
            "SETUP_SNIPER_EXTRA_ALLOW": sorted(SETUP_SNIPER_EXTRA_ALLOW),
        },
        "violations": [
            {"rule": v.rule, "where": v.where, "msg": v.msg}
            for v in out.violations
        ],
    }


def write_md(data: dict, path: Path) -> None:
    lines: list[str] = []
    lines.append("# Gate 268 — Setup-script invariant registry\n")
    lines.append(
        "Locks `scripts/setup_gem5.py` and `scripts/setup_sniper.py` "
        "against silent drift in upstream repo URLs, directory-constant "
        "skeleton, and the canonical top-level function inventory each "
        "script exposes.\n"
    )
    lines.append(
        "registry entries: %d repo URLs; %d gem5 dir-constants + %d functions; "
        "%d sniper dir-constants + %d functions.\n" % (
            data["repo_urls_n"],
            data["gem5_constants_n"], data["gem5_functions_n"],
            data["sniper_constants_n"], data["sniper_functions_n"],
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
        for name, url in SETUP_REPO_URLS.items():
            w.writerow(["repo_url", name, url])
        for c in SETUP_GEM5_DIR_CONSTANTS:
            w.writerow(["gem5_dir_const", c, "registered"])
        for c in SETUP_SNIPER_DIR_CONSTANTS:
            w.writerow(["sniper_dir_const", c, "registered"])
        for fn in SETUP_GEM5_FUNCTIONS:
            w.writerow(["gem5_function", fn, "registered"])
        for fn in SETUP_SNIPER_FUNCTIONS:
            w.writerow(["sniper_function", fn, "registered"])


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

    n = len(data["violations"])
    g5fn = data["gem5_functions_n"]
    snfn = data["sniper_functions_n"]
    print(f"[lit-faith-setup-script-registry] status={data['status']} "
          f"repo_urls={data['repo_urls_n']} "
          f"gem5_fn={g5fn} sniper_fn={snfn} violations={n}")
    return 0 if n == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
