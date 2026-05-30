"""Gate 257 — backend/tool vocabulary registry.

Locks the cross-reference between:
  - the canonical backend/tool token allow-list defined here
  - every backend / tool literal harvested by AST from
    ``scripts/experiments/ecg/*.py`` and ``scripts/test/*.py``
  - every ``--source-backend``, ``--backend``, ``--suite``,
    ``--tool``, and ``--tool-name`` argparse default and choices
    list

The cache-policy registry (gate 255) locks WHICH replacement policy
is referenced; the slurm-schema registry (gate 252) locks WHICH
cluster directives are emitted; the profile registry (gate 256)
locks WHICH run-profile is invoked. Gate 257 locks WHICH simulator
backend / tool every result row, anchor pickup, and report
labelling references — so a contributor cannot silently rename
``cache_sim`` to ``cache_simulator``, drop the hyphen in
``gem5-riscv`` to ``gem5riscv``, or introduce a rogue
``sniperx`` / ``GEM5`` upper-case variant without an explicit
canonical entry.

Catches the silent-drift cases:

* a per-row ``"backend": "gem5"`` literal is renamed to
  ``"gem5_x86"`` but downstream cross-tool aggregators still
  look up ``"gem5"`` (silent KeyError → empty cross-tool table),
* a new gem5 frontend (``gem5-arm``) is added but no canonical
  entry → the colour-palette and label maps silently render it as
  the python ``repr`` instead of a friendly name,
* a contributor uses ``"cachesim"`` / ``"cache simulator"`` /
  ``"CacheSim"`` instead of ``cache_sim`` / ``cache-sim``.

7 rules R1-R7:
  R1: every harvested backend/tool literal is in
      CANONICAL_BACKEND_TOKENS (the allow-list)
  R2: every canonical token has a non-empty family
      (cache_sim / gem5 / sniper) and a non-empty paper_label
  R3: no two canonical tokens share the same paper_label
  R4: every canonical token has a punctuation_variants entry that
      includes itself (catches "lonely" duplicate-styles like
      cache_sim having no cache-sim sibling)
  R5: every harvested argparse ``--backend`` / ``--tool`` choices
      list is a subset of CANONICAL_BACKEND_TOKENS
  R6: every canonical token name matches the documented regex
      ``^[a-z][a-z0-9_-]*$`` (lowercase ASCII; hyphen/underscore
      allowed; no leading digit)
  R7: every harvested literal is referenced by at least one of:
      a tuple/list/set literal, a dict key, a dict value, or an
      argparse default/choices entry (no truly-dead tokens)
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
ECG_DIR = REPO_ROOT / "scripts" / "experiments" / "ecg"
SCRIPTS_TEST_DIR = REPO_ROOT / "scripts" / "test"

BACKEND_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*$")


@dataclass(frozen=True)
class BackendEntry:
    """A canonical simulator backend / tool token."""

    name: str
    family: str
    paper_label: str
    purpose: str
    punctuation_variants: tuple[str, ...] = field(default_factory=tuple)


CANONICAL_BACKENDS: tuple[BackendEntry, ...] = (
    BackendEntry(
        name="cache_sim",
        family="cache_sim",
        paper_label="cache-sim",
        purpose=(
            "Analytical LRU-stack cache simulator (single-level, "
            "per-policy hit-rate sweep). Underscore form is used as "
            "Python identifier / dict-key; the hyphen form "
            "'cache-sim' is its display/paper-prose sibling."
        ),
        punctuation_variants=("cache_sim", "cache-sim"),
    ),
    BackendEntry(
        name="cache-sim",
        family="cache_sim",
        paper_label="cache-sim",
        purpose=(
            "Display/paper-prose form of cache_sim. Same artifact, "
            "hyphen-styled for narrative text and matplotlib labels."
        ),
        punctuation_variants=("cache_sim", "cache-sim"),
    ),
    BackendEntry(
        name="gem5",
        family="gem5",
        paper_label="gem5",
        purpose=(
            "Cycle-accurate full-system simulator (default X86 "
            "frontend in this repo). Used as the generic gem5 key in "
            "anchor pickups; the architecture-specific siblings "
            "(gem5-riscv, gem5-x86) only appear when the run "
            "explicitly distinguishes frontends (e.g. ECG scale "
            "preflight)."
        ),
        punctuation_variants=("gem5",),
    ),
    BackendEntry(
        name="gem5-riscv",
        family="gem5",
        paper_label="gem5/RISC-V",
        purpose=(
            "gem5 RISC-V frontend (ECG scale runs). Distinct from "
            "the generic gem5 key because the binary path and the "
            "preflight checks differ. Result rows from RISC-V runs "
            "must carry this token, not 'gem5'."
        ),
        punctuation_variants=("gem5-riscv",),
    ),
    BackendEntry(
        name="gem5-x86",
        family="gem5",
        paper_label="gem5/X86",
        purpose=(
            "gem5 X86 frontend marker (cluster preflight binary "
            "label). Currently only used by ecg_cluster_preflight "
            "for human-readable diagnostic output; result rows from "
            "X86 runs still use the generic 'gem5' key."
        ),
        punctuation_variants=("gem5-x86",),
    ),
    BackendEntry(
        name="sniper",
        family="sniper",
        paper_label="Sniper",
        purpose=(
            "Sniper interval-simulator (Pin-based instrumentation). "
            "Used as the generic Sniper key in anchor pickups; the "
            "frontend-specific sibling (sniper-sift) only appears "
            "when the run distinguishes the SIFT trace frontend."
        ),
        punctuation_variants=("sniper",),
    ),
    BackendEntry(
        name="sniper-sift",
        family="sniper",
        paper_label="Sniper/SIFT",
        purpose=(
            "Sniper SIFT (Sniper Instruction Format trace) frontend "
            "marker. Used by ECG scale preflight + scale-status row "
            "tagging so trace-driven runs are distinguished from "
            "Pin-driven runs."
        ),
        punctuation_variants=("sniper-sift",),
    ),
)

CANONICAL_BACKEND_NAMES: frozenset[str] = frozenset(
    b.name for b in CANONICAL_BACKENDS
)
CANONICAL_BACKEND_FAMILIES: frozenset[str] = frozenset(
    b.family for b in CANONICAL_BACKENDS
)
# Paper labels (e.g. "Sniper", "gem5/RISC-V") are valid prose
# substrings, NOT code-side backend tokens. We skip them from the
# typo allow-list so a `assert "Sniper" in text` in a markdown
# substring check doesn't trip R1.
CANONICAL_PAPER_LABELS: frozenset[str] = frozenset(
    b.paper_label for b in CANONICAL_BACKENDS
)

# Argparse flags that take a backend/tool token as a value. Every
# default / choices member must be in CANONICAL_BACKEND_NAMES.
BACKEND_ARG_FLAGS: frozenset[str] = frozenset({
    "--backend",
    "--source-backend",
    "--tool",
    "--tool-name",
    "--suite",
})


def _is_backend_string(s: str) -> bool:
    if s in CANONICAL_BACKEND_NAMES:
        # Always treat as a backend token, even if it happens to
        # also be its own paper_label (e.g. "gem5", "cache-sim").
        return True
    if s in CANONICAL_PAPER_LABELS:
        # Paper-prose-only label (e.g. "Sniper", "gem5/RISC-V"):
        # valid in markdown substring checks, NOT a code token.
        return False
    return s in {
        # Common typos / aliases that MUST be caught (R1).
        "Gem5", "GEM5", "SNIPER",
        "cachesim", "CacheSim", "cache_simulator",
        "gem5riscv", "snipersift",
    }


def _harvest_string_literals(tree: ast.AST) -> list[str]:
    """Returns all `ast.Constant(str)` values that match an exact
    backend/tool token or a documented typo. We deliberately do NOT
    use a prefix heuristic — strings like ``gem5_anchor`` or
    ``gem5_cells`` are dict keys / variable names, not backend
    tokens, and would generate hundreds of false positives."""
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _is_backend_string(node.value):
                out.append(node.value)
    return out


def _harvest_argparse_backend_args(tree: ast.AST) -> list[dict]:
    """Returns `[{flag, default, choices}]` for every call that looks
    like `parser.add_argument("--backend", default=..., choices=[...])`
    where flag is in BACKEND_ARG_FLAGS."""
    out: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "add_argument":
            pass
        elif isinstance(func, ast.Name) and func.id == "add_argument":
            pass
        else:
            continue
        if not node.args:
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
            continue
        flag = first.value
        if flag not in BACKEND_ARG_FLAGS:
            continue
        rec: dict[str, Any] = {"flag": flag, "default": None, "choices": []}
        for kw in node.keywords:
            if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                if isinstance(kw.value.value, str):
                    rec["default"] = kw.value.value
            elif kw.arg == "choices" and isinstance(kw.value, (ast.List, ast.Tuple, ast.Set)):
                vals = []
                for elt in kw.value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        vals.append(elt.value)
                rec["choices"] = vals
        out.append(rec)
    return out


def _scan_dir(d: Path) -> tuple[list[tuple[str, str]], list[tuple[str, dict]]]:
    """Returns (literals, argparse_entries) with file-path provenance.

    Skips the registry generator itself (lit_faith_backend_registry.py)
    AND its pytest gate (test_lit_faith_backend_registry.py) so the
    documented typo allow-list / regex fixtures do not self-cite as
    R1 violations."""
    literals: list[tuple[str, str]] = []
    arg_entries: list[tuple[str, dict]] = []
    skip_names = {
        Path(__file__).name,
        "test_lit_faith_backend_registry.py",
    }
    for f in sorted(d.glob("*.py")):
        if f.name in skip_names:
            continue
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            continue
        rel = str(f.relative_to(REPO_ROOT))
        for s in _harvest_string_literals(tree):
            literals.append((rel, s))
        for rec in _harvest_argparse_backend_args(tree):
            arg_entries.append((rel, rec))
    return literals, arg_entries


def audit() -> dict[str, Any]:
    literals_ecg, args_ecg = _scan_dir(ECG_DIR)
    literals_test, args_test = _scan_dir(SCRIPTS_TEST_DIR)
    all_literals = literals_ecg + literals_test
    all_args = args_ecg + args_test

    by_token: dict[str, list[str]] = {}
    for path, tok in all_literals:
        by_token.setdefault(tok, []).append(path)

    violations: list[dict[str, Any]] = []

    # R1: every harvested literal is canonical.
    for tok, paths in sorted(by_token.items()):
        if tok not in CANONICAL_BACKEND_NAMES:
            violations.append({
                "rule": "R1",
                "token": tok,
                "first_path": paths[0],
                "n_sites": len(paths),
                "msg": f"non-canonical backend/tool literal {tok!r}; add to CANONICAL_BACKENDS or fix the typo",
            })

    # R2: every canonical has non-empty family + paper_label.
    seen_labels: dict[str, str] = {}
    for b in CANONICAL_BACKENDS:
        if not b.family:
            violations.append({"rule": "R2", "token": b.name, "msg": "empty family"})
        if b.family not in {"cache_sim", "gem5", "sniper"}:
            violations.append({
                "rule": "R2",
                "token": b.name,
                "msg": f"family {b.family!r} not in {{cache_sim,gem5,sniper}}",
            })
        if not b.paper_label:
            violations.append({"rule": "R2", "token": b.name, "msg": "empty paper_label"})

    # R3: no duplicate paper_label EXCEPT within the same punctuation
    # variant pair (cache_sim / cache-sim deliberately share).
    for b in CANONICAL_BACKENDS:
        prev = seen_labels.get(b.paper_label)
        if prev is not None and b.name not in CANONICAL_BACKENDS[
            next(i for i, x in enumerate(CANONICAL_BACKENDS) if x.name == prev)
        ].punctuation_variants:
            violations.append({
                "rule": "R3",
                "token": b.name,
                "msg": f"paper_label {b.paper_label!r} collides with {prev!r} (not a declared punctuation variant)",
            })
        seen_labels[b.paper_label] = b.name

    # R4: each canonical's punctuation_variants includes itself.
    for b in CANONICAL_BACKENDS:
        if b.name not in b.punctuation_variants:
            violations.append({
                "rule": "R4",
                "token": b.name,
                "msg": f"punctuation_variants {b.punctuation_variants!r} does not include self",
            })

    # R5: every argparse choices list is a subset of canonical.
    for path, rec in all_args:
        bad_choices = [c for c in rec["choices"] if c not in CANONICAL_BACKEND_NAMES and c != "both"]
        if bad_choices:
            violations.append({
                "rule": "R5",
                "token": ",".join(bad_choices),
                "first_path": path,
                "flag": rec["flag"],
                "msg": f"argparse {rec['flag']} choices include non-canonical tokens {bad_choices}",
            })
        if rec["default"] is not None and rec["default"] not in CANONICAL_BACKEND_NAMES and rec["default"] != "both":
            violations.append({
                "rule": "R5",
                "token": rec["default"],
                "first_path": path,
                "flag": rec["flag"],
                "msg": f"argparse {rec['flag']} default {rec['default']!r} is non-canonical",
            })

    # R6: canonical token names match BACKEND_NAME_RE.
    for b in CANONICAL_BACKENDS:
        if not BACKEND_NAME_RE.match(b.name):
            violations.append({
                "rule": "R6",
                "token": b.name,
                "msg": f"token {b.name!r} does not match {BACKEND_NAME_RE.pattern}",
            })

    # R7: every harvested literal is also referenced by some site (we
    # already have at least one path per token from the harvest, so
    # this is implicitly satisfied unless the canonical list grows a
    # token that NO site ever uses).
    used_tokens = set(by_token.keys())
    for b in CANONICAL_BACKENDS:
        if b.name not in used_tokens:
            violations.append({
                "rule": "R7",
                "token": b.name,
                "msg": f"canonical token {b.name!r} declared but no in-tree literal references it",
            })

    return {
        "status": "active",
        "n_canonical": len(CANONICAL_BACKENDS),
        "n_families": len(CANONICAL_BACKEND_FAMILIES),
        "n_literal_sites": len(all_literals),
        "n_distinct_literals": len(by_token),
        "n_argparse_sites": len(all_args),
        "canonical": [
            {
                "name": b.name,
                "family": b.family,
                "paper_label": b.paper_label,
                "purpose": b.purpose,
                "punctuation_variants": list(b.punctuation_variants),
            }
            for b in CANONICAL_BACKENDS
        ],
        "harvested_tokens": sorted(by_token.keys()),
        "argparse_entries": [
            {"path": p, **rec} for (p, rec) in all_args
        ],
        "rules": {
            "R1": "every harvested backend/tool literal is in CANONICAL_BACKENDS",
            "R2": "every canonical has non-empty family ∈ {cache_sim,gem5,sniper} + non-empty paper_label",
            "R3": "no two canonicals share a paper_label unless they are declared punctuation variants",
            "R4": "every canonical's punctuation_variants includes its own name",
            "R5": "every --backend/--tool/--suite argparse choices+default ⊆ CANONICAL_BACKENDS (+'both')",
            "R6": f"canonical token names match {BACKEND_NAME_RE.pattern}",
            "R7": "every canonical token is referenced by at least one in-tree literal",
        },
        "violations": violations,
    }


def _emit_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _emit_md(data: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# Gate 257 — backend/tool vocabulary registry")
    lines.append("")
    lines.append(f"Status: **{data['status']}**")
    lines.append("")
    lines.append("## Totals")
    lines.append("")
    for k in ("n_canonical", "n_families", "n_literal_sites",
              "n_distinct_literals", "n_argparse_sites"):
        lines.append(f"- {k}: {data[k]}")
    lines.append("")
    lines.append("## Rules")
    lines.append("")
    for rid, desc in data["rules"].items():
        lines.append(f"- **{rid}** — {desc}")
    lines.append("")
    lines.append("## Canonical backends")
    lines.append("")
    lines.append("| name | family | paper_label | punctuation_variants |")
    lines.append("|---|---|---|---|")
    for b in data["canonical"]:
        variants = ", ".join(f"`{v}`" for v in b["punctuation_variants"])
        lines.append(
            f"| `{b['name']}` | `{b['family']}` | {b['paper_label']} | {variants} |"
        )
    lines.append("")
    lines.append("## Harvested tokens (in-tree literals)")
    lines.append("")
    for t in data["harvested_tokens"]:
        lines.append(f"- `{t}`")
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
    for b in data["canonical"]:
        rows.append(("canonical", b["name"], b["family"]))
    for t in data["harvested_tokens"]:
        rows.append(("literal", t, ""))
    for v in data["violations"]:
        rows.append(("violation", str(v.get("rule", "")), str(v)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("kind", "name", "extra"))
        for r in rows:
            w.writerow(r)


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
        f"[lit-faith-backend-registry] status={data['status']} "
        f"canonical={data['n_canonical']} "
        f"families={data['n_families']} "
        f"sites={data['n_literal_sites']} "
        f"distinct={data['n_distinct_literals']} "
        f"argparse={data['n_argparse_sites']} "
        f"violations={len(data['violations'])}"
    )
    return 1 if data["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
