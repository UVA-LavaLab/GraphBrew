"""Gate 256 — ECG final-paper-run profile registry.

Locks the cross-reference between:
  - the manifest profiles dict declared in
    ``scripts/experiments/ecg/final_paper_manifest.json``
  - every ``stage.profiles[*]`` token consumed by stages in that
    same manifest
  - every ``--profile <name>`` reference in pytest fixtures, helper
    scripts under ``scripts/experiments/ecg``, and the
    ``scripts/experiments/README.md`` walkthrough

Catches the silent-drift cases:

* a manifest profile is renamed but a stage still references the
  old token (``stage.profiles`` parses, but ``defaults`` merge
  silently degrades),
* a contributor adds a new profile without a description (the
  ``--list-profiles`` UX silently shows an empty descriptor),
* a profile is declared in the manifest but no stage ever references
  it (dead profile) and no helper script names it either,
* a typo in a README example (``--profile fianl_replacement``)
  bit-rots the published walkthrough.

7 rules R1-R7:
  R1: every stage.profiles[*] token resolves to a key in
      manifest.profiles
  R2: every manifest.profiles key has a non-empty description
  R3: every manifest.profiles key is referenced by either a stage,
      a pytest test, or a helper script (no dead profiles)
  R4: every --profile <token> reference outside the manifest
      resolves to a known manifest.profiles key (no typos)
  R5: profile names are snake_case ASCII (regex: ``^[a-z][a-z0-9_]*$``)
  R6: stage names are snake_case-with-leading-digit-prefix ASCII
      (regex: ``^[0-9]+[a-z][a-z0-9_]*$``) so the natural sort matches
      execution order
  R7: each stage's profiles list is unique (no duplicates) and
      non-empty
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
ECG_DIR = REPO_ROOT / "scripts" / "experiments" / "ecg"
MANIFEST = ECG_DIR / "final_paper_manifest.json"
WIKI_DATA = REPO_ROOT / "wiki" / "data"
TEST_DIR = REPO_ROOT / "scripts" / "test"
EXPERIMENTS_README = REPO_ROOT / "scripts" / "experiments" / "README.md"

PROFILE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
STAGE_NAME_RE = re.compile(r"^[0-9]+[a-z0-9]*_[a-z][a-z0-9_]*$")

# Profiles that are intentionally declared in the manifest but not yet
# wired to a stage — they document upcoming work explicitly (matched
# by a substring in the description). Without this allow-list the
# placeholder profiles would trip R3 even though the manifest itself
# documents why they exist.
PLACEHOLDER_PROFILE_HINT = "Placeholder"

# Locations whose `--profile <token>` references must resolve. Limited
# to documented helper scripts + pytest harness fixtures + the
# experiments README to keep the audit fast and meaningful.
PROFILE_CITATION_ROOTS: tuple[Path, ...] = (
    ECG_DIR,
    TEST_DIR,
)


@dataclass(frozen=True)
class ProfileEntry:
    """A single declared manifest profile."""

    key: str
    description: str

    def __post_init__(self) -> None:  # pragma: no cover - sanity guard
        if not isinstance(self.key, str) or not isinstance(self.description, str):
            raise TypeError("ProfileEntry expects string fields")


# ----------------------------- harvest --------------------------------

def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _harvest_manifest_profiles(
    manifest: dict[str, Any],
) -> dict[str, ProfileEntry]:
    raw = manifest.get("profiles", {}) or {}
    out: dict[str, ProfileEntry] = {}
    for key, desc in raw.items():
        out[key] = ProfileEntry(key=str(key), description=str(desc or ""))
    return out


def _harvest_stage_profile_tokens(
    manifest: dict[str, Any],
) -> list[tuple[str, list[str]]]:
    stages = manifest.get("stages", []) or []
    out: list[tuple[str, list[str]]] = []
    for stage in stages:
        name = str(stage.get("name") or "")
        profs = list(stage.get("profiles", []) or [])
        out.append((name, [str(p) for p in profs]))
    return out


def _iter_profile_string_args(node: ast.AST) -> list[str]:
    """Find `--profile <token>` adjacencies inside ast.List / ast.Call
    args+kwargs. Conservative: only literal string constants."""
    out: list[str] = []
    if isinstance(node, ast.List):
        items = node.elts
    elif isinstance(node, ast.Tuple):
        items = node.elts
    elif isinstance(node, ast.Call):
        items = node.args
    else:
        return out
    for i, elt in enumerate(items):
        if (
            isinstance(elt, ast.Constant)
            and isinstance(elt.value, str)
            and elt.value == "--profile"
        ):
            for j in range(i + 1, len(items)):
                nxt = items[j]
                if (
                    isinstance(nxt, ast.Constant)
                    and isinstance(nxt.value, str)
                ):
                    tok = nxt.value
                    if tok.startswith("-"):
                        break
                    out.append(tok)
                else:
                    break
    return out


def _harvest_profile_citations_py(path: Path) -> list[str]:
    try:
        text = path.read_text()
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return []
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple, ast.Call)):
            found.extend(_iter_profile_string_args(node))
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (
                    isinstance(tgt, ast.Name)
                    and tgt.id in {"profile", "profiles"}
                    and isinstance(node.value, ast.List)
                ):
                    for elt in node.value.elts:
                        if (
                            isinstance(elt, ast.Constant)
                            and isinstance(elt.value, str)
                        ):
                            found.append(elt.value)
        if isinstance(node, ast.keyword) and node.arg in {"profile", "profiles"}:
            if isinstance(node.value, ast.List):
                for elt in node.value.elts:
                    if (
                        isinstance(elt, ast.Constant)
                        and isinstance(elt.value, str)
                    ):
                        found.append(elt.value)
    return found


_README_PROFILE_RE = re.compile(
    r"--profiles?[ \t]+([a-z][a-z0-9_]*(?:[ \t]+[a-z][a-z0-9_]*)*)"
)


def _harvest_profile_citations_md(path: Path) -> list[str]:
    try:
        text = path.read_text()
    except OSError:
        return []
    found: list[str] = []
    for m in _README_PROFILE_RE.finditer(text):
        for tok in m.group(1).split():
            found.append(tok)
    return found


def _harvest_all_citations() -> dict[str, list[tuple[str, str]]]:
    """name → list of (relpath, token)."""
    out: dict[str, list[tuple[str, str]]] = {}
    for root in PROFILE_CITATION_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            rel = str(path.relative_to(REPO_ROOT))
            for tok in _harvest_profile_citations_py(path):
                out.setdefault(tok, []).append((rel, tok))
    if EXPERIMENTS_README.is_file():
        rel = str(EXPERIMENTS_README.relative_to(REPO_ROOT))
        for tok in _harvest_profile_citations_md(EXPERIMENTS_README):
            out.setdefault(tok, []).append((rel, tok))
    return out


# ----------------------------- rules ---------------------------------

def _rule_r1(
    profiles: dict[str, ProfileEntry],
    stages: list[tuple[str, list[str]]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    known = set(profiles)
    for stage_name, profs in stages:
        for tok in profs:
            if tok not in known:
                out.append({
                    "rule": "R1",
                    "stage": stage_name,
                    "token": tok,
                    "msg": "stage profile token not in manifest.profiles",
                })
    return out


def _rule_r2(profiles: dict[str, ProfileEntry]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, entry in profiles.items():
        if not entry.description.strip():
            out.append({
                "rule": "R2",
                "profile": key,
                "msg": "manifest profile has empty description",
            })
    return out


def _rule_r3(
    profiles: dict[str, ProfileEntry],
    stages: list[tuple[str, list[str]]],
    citations: dict[str, list[tuple[str, str]]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    referenced = set()
    for _, profs in stages:
        referenced.update(profs)
    referenced.update(citations.keys())
    for key, entry in profiles.items():
        if key in referenced:
            continue
        if PLACEHOLDER_PROFILE_HINT in entry.description:
            continue
        out.append({
            "rule": "R3",
            "profile": key,
            "msg": "manifest profile is declared but never referenced "
                   "(no stage, pytest, helper script, or README "
                   "uses it; description does not flag it as a "
                   "Placeholder)",
        })
    return out


def _rule_r4(
    profiles: dict[str, ProfileEntry],
    citations: dict[str, list[tuple[str, str]]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    known = set(profiles)
    for tok, hits in citations.items():
        if tok in known:
            continue
        for rel, _ in hits:
            out.append({
                "rule": "R4",
                "token": tok,
                "file": rel,
                "msg": "external --profile / profiles= citation does "
                       "not resolve to a manifest.profiles key",
            })
    return out


def _rule_r5(profiles: dict[str, ProfileEntry]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key in profiles:
        if not PROFILE_NAME_RE.match(key):
            out.append({
                "rule": "R5",
                "profile": key,
                "msg": "profile name does not match ^[a-z][a-z0-9_]*$",
            })
    return out


def _rule_r6(stages: list[tuple[str, list[str]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for stage_name, _ in stages:
        if not STAGE_NAME_RE.match(stage_name):
            out.append({
                "rule": "R6",
                "stage": stage_name,
                "msg": "stage name does not match "
                       "^[0-9]+[a-z0-9]*_[a-z][a-z0-9_]*$",
            })
    return out


def _rule_r7(stages: list[tuple[str, list[str]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for stage_name, profs in stages:
        if not profs:
            out.append({
                "rule": "R7",
                "stage": stage_name,
                "msg": "stage has empty profiles list",
            })
        if len(set(profs)) != len(profs):
            out.append({
                "rule": "R7",
                "stage": stage_name,
                "profiles": profs,
                "msg": "stage profiles list has duplicates",
            })
    return out


# ----------------------------- audit ---------------------------------

def audit() -> dict[str, Any]:
    manifest = _load_manifest(MANIFEST)
    profiles = _harvest_manifest_profiles(manifest)
    stages = _harvest_stage_profile_tokens(manifest)
    citations = _harvest_all_citations()

    violations: list[dict[str, Any]] = []
    violations.extend(_rule_r1(profiles, stages))
    violations.extend(_rule_r2(profiles))
    violations.extend(_rule_r3(profiles, stages, citations))
    violations.extend(_rule_r4(profiles, citations))
    violations.extend(_rule_r5(profiles))
    violations.extend(_rule_r6(stages))
    violations.extend(_rule_r7(stages))

    return {
        "status": "active",
        "rules": {
            "R1": "every stage.profiles[*] token resolves to manifest.profiles",
            "R2": "every manifest.profiles key has a non-empty description",
            "R3": "every manifest.profiles key is referenced by stage, pytest, helper, or README (no dead profiles)",
            "R4": "every --profile / profiles= citation outside the manifest resolves",
            "R5": "profile names match ^[a-z][a-z0-9_]*$",
            "R6": "stage names match ^[0-9]+[a-z0-9]*_[a-z][a-z0-9_]*$",
            "R7": "each stage's profiles list is non-empty and duplicate-free",
        },
        "totals": {
            "manifest_profiles":  len(profiles),
            "stages":             len(stages),
            "citations_total":    sum(len(v) for v in citations.values()),
            "distinct_citations": len(citations),
            "violations":         len(violations),
        },
        "profiles":          sorted(profiles.keys()),
        "stages":            [s for s, _ in stages],
        "citation_tokens":   sorted(citations.keys()),
        "violations":        violations,
    }


# ----------------------------- emitters ------------------------------

def _emit_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _emit_md(data: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# Gate 256 — ECG profile registry")
    lines.append("")
    lines.append(f"Status: **{data['status']}**")
    lines.append("")
    lines.append("## Totals")
    lines.append("")
    for k, v in data["totals"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Rules")
    lines.append("")
    for rid, desc in data["rules"].items():
        lines.append(f"- **{rid}** — {desc}")
    lines.append("")
    lines.append("## Manifest profiles")
    lines.append("")
    for p in data["profiles"]:
        lines.append(f"- `{p}`")
    lines.append("")
    lines.append("## Stages (run-order)")
    lines.append("")
    for s in data["stages"]:
        lines.append(f"- `{s}`")
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
    rows: list[tuple[str, str, str]] = []
    for p in data["profiles"]:
        rows.append(("profile", p, ""))
    for s in data["stages"]:
        rows.append(("stage", s, ""))
    for v in data["violations"]:
        rows.append(("violation", str(v.get("rule", "")), str(v)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(("kind", "name", "detail"))
        w.writerows(rows)


# ----------------------------- CLI ----------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--md-out", type=Path, default=None)
    ap.add_argument("--csv-out", type=Path, default=None)
    args = ap.parse_args(argv)

    data = audit()
    if args.json_out:
        _emit_json(data, args.json_out)
    if args.md_out:
        _emit_md(data, args.md_out)
    if args.csv_out:
        _emit_csv(data, args.csv_out)

    print(
        f"[lit-faith-profile-registry] status={data['status']} "
        f"profiles={data['totals']['manifest_profiles']} "
        f"stages={data['totals']['stages']} "
        f"citations={data['totals']['citations_total']} "
        f"violations={data['totals']['violations']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
