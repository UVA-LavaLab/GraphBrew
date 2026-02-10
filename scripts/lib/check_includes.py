#!/usr/bin/env python3
"""
check_includes.py

Scan repository for legacy include paths and fail if found.

Legacy → Preferred:
- cache/cache_sim.h         → cache_sim/cache_sim.h
- cache/graph_sim.h         → cache_sim/graph_sim.h
- graphbrew/cache/popt.h    → graphbrew/partition/cagra/popt.h

Usage:
    python3 scripts/graphbrew_experiment.py --check-includes
    python3 -m scripts.lib.check_includes [--root DIR]

Exits non-zero if any legacy include is found.
"""
import argparse
import os
import re
from pathlib import Path

LEGACY_MAP = {
    "cache/cache_sim.h": "cache_sim/cache_sim.h",
    "cache/graph_sim.h": "cache_sim/graph_sim.h",
    "graphbrew/cache/popt.h": "graphbrew/partition/cagra/popt.h",
    "graphbrew/partitioning/popt.h": "graphbrew/partition/cagra/popt.h",
}

INCLUDE_RE = re.compile(r"^\s*#\s*include\s*[<\"]([^\"<>]+)[\"\"]")

CODE_EXTS = {
    ".h", ".hh", ".hpp", ".hxx", ".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh",
}

EXCLUDE_DIRS = {
    ".git", ".venv", "build", "dist", "bench/bin", "bench/bin_sim", "node_modules", "results",
}

def should_exclude(path: Path) -> bool:
    parts = path.parts
    for ex in EXCLUDE_DIRS:
        ex_parts = Path(ex).parts
        # simple prefix match for exclude dirs
        if parts[: len(ex_parts)] == ex_parts:
            return True
    return False

def scan_file(path: Path):
    hits = []
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return hits
    for idx, line in enumerate(text.splitlines(), start=1):
        m = INCLUDE_RE.match(line)
        if not m:
            continue
        inc = m.group(1)
        if inc in LEGACY_MAP:
            hits.append((idx, inc, LEGACY_MAP[inc], line.strip()))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]),
                        help="Repository root (default: repo root)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"error: root {root} is not a directory")
        return 2

    legacy_hits = []
    for path in root.rglob("*"):
        if path.is_dir():
            if should_exclude(path.relative_to(root)):
                # skip subtree
                # Path.rglob doesn't allow prune; rely on skip at file level
                pass
            continue
        if path.suffix not in CODE_EXTS:
            continue
        rel = path.relative_to(root)
        if should_exclude(rel):
            continue
        hits = scan_file(path)
        if hits:
            legacy_hits.append((rel, hits))

    if legacy_hits:
        print("Legacy includes found:\n")
        for rel, hits in legacy_hits:
            print(f"- {rel}")
            for (ln, inc, pref, line) in hits:
                print(f"  L{ln}: {line}")
                print(f"    -> Use: #include \"{pref}\"")
            print()
        return 1
    else:
        print("No legacy includes found.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
