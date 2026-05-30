#!/usr/bin/env python3
"""Gate 252 — Slurm SBATCH directive schema registry.

Locks the schema of every ``*.sbatch`` file under
``scripts/experiments/``:

* declares ``CANONICAL_SBATCH_DIRECTIVES`` with valid directive names,
  whether they are required vs optional, expected pattern, and
  forbidden co-occurrences;
* parses every ``#SBATCH`` line, classifies it, and asserts:

  - S1 syntax: every ``#SBATCH`` line matches the long-form
    ``--key=value`` shape (or two ``--key=value`` pairs on the same
    line, which Slurm accepts);
  - S2 coverage: every shipped sbatch carries the mandatory directives
    (``--job-name``, ``--time``, ``--nodes``, ``--ntasks``,
    ``--cpus-per-task``, ``--mem``, ``--output``);
  - S3 vocabulary: every directive used appears in the canonical
    registry (no typos like ``--mem-per-node``);
  - S4 mem syntax: ``--mem`` values match ``\\d+[GM]``;
  - S5 time syntax: ``--time`` values match ``HH:MM:SS`` or
    ``D-HH:MM:SS``;
  - S6 node shape: every shipped sbatch is single-node single-task
    (``--nodes=1`` and ``--ntasks=1``);
  - S7 log templates: ``--output``/``--error`` templates use
    ``%x``+``%j`` or ``%A``+``%a`` (no anonymous logs);
  - S8 job-name prefix: ``--job-name`` starts with one of the
    documented project prefixes (``gbrew-`` or ``ecg-``);
  - S9 mem co-occurrence: ``--mem`` and ``--mem-per-cpu`` never
    co-occur in the same file (Slurm semantics forbid both).

The artifact is consumed by
``scripts/test/test_lit_faith_slurm_schema.py``.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SBATCH_GLOB_ROOTS = (
    ROOT / "scripts" / "experiments" / "ecg",
    ROOT / "scripts" / "experiments" / "vldb",
)
SELF_PATH = Path(__file__).resolve()


# --- canonical registry ---------------------------------------------

@dataclass(frozen=True)
class Directive:
    name: str
    required: bool
    pattern: str | None  # regex applied to value, or None for free-form
    note: str = ""


CANONICAL_SBATCH_DIRECTIVES: dict[str, Directive] = {
    # --- required everywhere -------------------------------------
    "job-name":      Directive("job-name",      True,  r"^[A-Za-z][\w\-]*$",            "human-readable label"),
    "time":          Directive("time",          True,  r"^(\d{1,2}:\d{2}:\d{2}|\d{1,2}-\d{1,2}:\d{2}:\d{2})$",
                               "HH:MM:SS or D-HH:MM:SS"),
    "nodes":         Directive("nodes",         True,  r"^\d+$",                        "node count"),
    "ntasks":        Directive("ntasks",        True,  r"^\d+$",                        "MPI task count"),
    "cpus-per-task": Directive("cpus-per-task", True,  r"^\d+$",                        "OMP thread count"),
    "mem":           Directive("mem",           True,  r"^\d+[GMK]?$",                  "per-node memory; G|M"),
    "output":        Directive("output",        True,  None,                            "stdout log template"),
    # --- optional but allowed ------------------------------------
    "error":         Directive("error",         False, None,                            "stderr log template"),
    "partition":     Directive("partition",     False, r"^[A-Za-z][\w\-]*$",            "queue / partition"),
    "account":       Directive("account",       False, r"^[A-Za-z][\w\-]*$",            "billing account"),
    "exclusive":     Directive("exclusive",     False, r"^$",                           "flag-style (no value)"),
    "array":         Directive("array",         False, r"^\d+(-\d+)?(:\d+)?(%\d+)?$",   "array spec"),
    "gres":          Directive("gres",          False, None,                            "generic resources e.g. gpu:1"),
    "mem-per-cpu":   Directive("mem-per-cpu",   False, r"^\d+[GMK]?$",                  "per-CPU memory"),
}

REQUIRED_DIRECTIVES = {
    name for name, d in CANONICAL_SBATCH_DIRECTIVES.items() if d.required
}

JOB_NAME_PREFIXES = ("gbrew-", "ecg-")
LOG_TEMPLATE_GROUPS = (
    ("%x", "%j"),
    ("%A", "%a"),
)


# --- parser ----------------------------------------------------------

SBATCH_LINE_RE = re.compile(r"^#SBATCH\s+(.+)$")
KV_RE = re.compile(r"^--([\w\-]+)(?:=(.+))?$")


@dataclass
class SbatchFile:
    path: Path
    directives: list[tuple[str, str | None]] = field(default_factory=list)
    raw_lines: list[str] = field(default_factory=list)

    @property
    def directive_names(self) -> set[str]:
        return {name for name, _ in self.directives}

    def get(self, name: str) -> str | None:
        for n, v in self.directives:
            if n == name:
                return v
        return None


def _parse_sbatch(path: Path) -> SbatchFile:
    sf = SbatchFile(path=path)
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line.startswith("#SBATCH"):
            continue
        sf.raw_lines.append(line)
        m = SBATCH_LINE_RE.match(line)
        if not m:
            continue
        body = m.group(1)
        # Strip trailing comments like '# blah'.
        body = re.split(r"\s+#", body, maxsplit=1)[0].strip()
        # Some files put multiple flags on one line:
        # "--nodes=1 --ntasks=1 --cpus-per-task=16".  Split by whitespace.
        for token in body.split():
            km = KV_RE.match(token)
            if not km:
                # Could not parse this token; record an anomalous
                # entry so S1 can flag it.
                sf.directives.append((token, None))
                continue
            sf.directives.append((km.group(1), km.group(2)))
    return sf


def _collect_files() -> list[Path]:
    seen: set[Path] = set()
    for root in SBATCH_GLOB_ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("*.sbatch"):
            if p.resolve() == SELF_PATH:
                continue
            seen.add(p.resolve())
    return sorted(seen)


# --- rules -----------------------------------------------------------

def _rule_s1(sbatch: SbatchFile) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, _ in sbatch.directives:
        if not re.match(r"^[\w\-]+$", name):
            out.append({
                "rule":     "S1",
                "file":     str(sbatch.path.relative_to(ROOT)),
                "msg":      f"unparseable #SBATCH token: {name!r}",
            })
    return out


def _rule_s2(sbatch: SbatchFile) -> list[dict[str, Any]]:
    missing = REQUIRED_DIRECTIVES - sbatch.directive_names
    if missing:
        return [{
            "rule": "S2",
            "file": str(sbatch.path.relative_to(ROOT)),
            "msg":  f"missing required directives: {sorted(missing)}",
        }]
    return []


def _rule_s3(sbatch: SbatchFile) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, _ in sbatch.directives:
        if name not in CANONICAL_SBATCH_DIRECTIVES:
            out.append({
                "rule":     "S3",
                "file":     str(sbatch.path.relative_to(ROOT)),
                "directive": name,
                "msg":      f"unknown directive --{name}",
            })
    return out


def _rule_s4(sbatch: SbatchFile) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, value in sbatch.directives:
        if name not in ("mem", "mem-per-cpu") or value is None:
            continue
        d = CANONICAL_SBATCH_DIRECTIVES.get(name)
        if d is None or d.pattern is None:
            continue
        if not re.match(d.pattern, value):
            out.append({
                "rule":     "S4",
                "file":     str(sbatch.path.relative_to(ROOT)),
                "directive": name,
                "value":    value,
                "msg":      f"--{name}={value!r} does not match {d.pattern}",
            })
    return out


def _rule_s5(sbatch: SbatchFile) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    value = sbatch.get("time")
    if value is None:
        return out
    d = CANONICAL_SBATCH_DIRECTIVES["time"]
    if d.pattern and not re.match(d.pattern, value):
        out.append({
            "rule":     "S5",
            "file":     str(sbatch.path.relative_to(ROOT)),
            "value":    value,
            "msg":      f"--time={value!r} does not match HH:MM:SS or D-HH:MM:SS",
        })
    return out


def _rule_s6(sbatch: SbatchFile) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for k in ("nodes", "ntasks"):
        v = sbatch.get(k)
        if v not in (None, "1"):
            out.append({
                "rule":     "S6",
                "file":     str(sbatch.path.relative_to(ROOT)),
                "directive": k,
                "value":    v,
                "msg":      f"--{k}={v!r} expected '1' (single-node single-task)",
            })
    return out


def _rule_s7(sbatch: SbatchFile) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for k in ("output", "error"):
        v = sbatch.get(k)
        if v is None:
            continue
        ok = any(all(tok in v for tok in group) for group in LOG_TEMPLATE_GROUPS)
        if not ok:
            out.append({
                "rule":     "S7",
                "file":     str(sbatch.path.relative_to(ROOT)),
                "directive": k,
                "value":    v,
                "msg":      f"--{k}={v!r} must contain %x+%j or %A+%a",
            })
    return out


def _rule_s8(sbatch: SbatchFile) -> list[dict[str, Any]]:
    name = sbatch.get("job-name")
    if name is None:
        return []
    if not any(name.startswith(p) for p in JOB_NAME_PREFIXES):
        return [{
            "rule":     "S8",
            "file":     str(sbatch.path.relative_to(ROOT)),
            "value":    name,
            "msg":      f"--job-name={name!r} must start with one of {JOB_NAME_PREFIXES}",
        }]
    return []


def _rule_s9(sbatch: SbatchFile) -> list[dict[str, Any]]:
    if sbatch.get("mem") is not None and sbatch.get("mem-per-cpu") is not None:
        return [{
            "rule": "S9",
            "file": str(sbatch.path.relative_to(ROOT)),
            "msg":  "--mem and --mem-per-cpu both set (Slurm forbids this combination)",
        }]
    return []


RULES = (
    ("S1", _rule_s1),
    ("S2", _rule_s2),
    ("S3", _rule_s3),
    ("S4", _rule_s4),
    ("S5", _rule_s5),
    ("S6", _rule_s6),
    ("S7", _rule_s7),
    ("S8", _rule_s8),
    ("S9", _rule_s9),
)


# --- driver ----------------------------------------------------------

def audit() -> dict[str, Any]:
    files = _collect_files()
    parsed = [_parse_sbatch(p) for p in files]
    violations: list[dict[str, Any]] = []
    for sf in parsed:
        for _, fn in RULES:
            violations.extend(fn(sf))

    files_summary = []
    for sf in parsed:
        files_summary.append({
            "path":           str(sf.path.relative_to(ROOT)),
            "directives":     sorted(sf.directive_names),
            "directive_count": len(sf.directives),
            "missing_required": sorted(REQUIRED_DIRECTIVES - sf.directive_names),
        })

    return {
        "status":         "active",
        "rules": {
            "S1": "every #SBATCH line parses to --key[=value]",
            "S2": "every required directive present",
            "S3": "every directive in CANONICAL_SBATCH_DIRECTIVES",
            "S4": "mem / mem-per-cpu values match \\d+[GMK]?",
            "S5": "time values match HH:MM:SS or D-HH:MM:SS",
            "S6": "single-node single-task (nodes=1 + ntasks=1)",
            "S7": "log templates contain %x+%j or %A+%a",
            "S8": f"job-name starts with one of {list(JOB_NAME_PREFIXES)}",
            "S9": "mem and mem-per-cpu never co-occur in same file",
        },
        "canonical_directives": {
            name: {
                "required": d.required,
                "pattern":  d.pattern,
                "note":     d.note,
            }
            for name, d in CANONICAL_SBATCH_DIRECTIVES.items()
        },
        "required_directives": sorted(REQUIRED_DIRECTIVES),
        "job_name_prefixes":   list(JOB_NAME_PREFIXES),
        "totals": {
            "files":               len(files),
            "directives_canonical": len(CANONICAL_SBATCH_DIRECTIVES),
            "directives_required":  len(REQUIRED_DIRECTIVES),
            "violations":           len(violations),
        },
        "files":      files_summary,
        "violations": violations,
    }


def write_outputs(audit_data: dict[str, Any],
                  json_out: Path | None,
                  md_out: Path | None,
                  csv_out: Path | None) -> None:
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(audit_data, indent=2, sort_keys=True) + "\n")
    if md_out:
        md_out.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        lines.append("# Slurm SBATCH schema registry — gate 252")
        lines.append("")
        lines.append(f"Status: `{audit_data['status']}`")
        lines.append("")
        t = audit_data["totals"]
        lines.append(
            f"Totals: files={t['files']}  canonical={t['directives_canonical']}  "
            f"required={t['directives_required']}  violations={t['violations']}")
        lines.append("")
        lines.append("## Required directives")
        lines.append("")
        for name in audit_data["required_directives"]:
            d = audit_data["canonical_directives"][name]
            lines.append(f"- `--{name}` — {d['note']}")
        lines.append("")
        lines.append("## Files")
        lines.append("")
        lines.append("| File | directive_count | missing_required |")
        lines.append("|------|-----------------|------------------|")
        for f in audit_data["files"]:
            miss = ",".join(f["missing_required"]) or "—"
            lines.append(f"| {f['path']} | {f['directive_count']} | {miss} |")
        lines.append("")
        if audit_data["violations"]:
            lines.append("## Violations")
            lines.append("")
            for v in audit_data["violations"][:50]:
                lines.append(f"- {v}")
            lines.append("")
        md_out.write_text("\n".join(lines))
    if csv_out:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        with csv_out.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "directive_count", "missing_required",
                        "violations"])
            viol_by_file: dict[str, int] = {}
            for v in audit_data["violations"]:
                viol_by_file[v["file"]] = viol_by_file.get(v["file"], 0) + 1
            for f in audit_data["files"]:
                w.writerow([
                    f["path"],
                    f["directive_count"],
                    ",".join(f["missing_required"]),
                    viol_by_file.get(f["path"], 0),
                ])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--md-out",   type=Path, default=None)
    ap.add_argument("--csv-out",  type=Path, default=None)
    args = ap.parse_args()
    a = audit()
    write_outputs(a, args.json_out, args.md_out, args.csv_out)
    print(
        f"[lit-faith-slurm-schema] status={a['status']} "
        f"files={a['totals']['files']} "
        f"canonical={a['totals']['directives_canonical']} "
        f"violations={a['totals']['violations']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
