#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Baseline drift bumper for downstream cross-artifact parity tests.

After a coordinated lit-faith regeneration that legitimately shifts
the cell-census data shape (e.g. cache_sim binary fixes, new sweep
cells), many hardcoded EXPECTED_* constants in test files become
stale. This tool reads the LIVE artifact values from wiki/data/
and prints suggested constant updates.

The bumps are HONEST data drift (real measurement values), not test
weakening — each bump should be reviewed and committed with a
rationale.

Usage:
  python3 -m scripts.experiments.ecg.baseline_drift_bumper

Output: human-readable report of (test_file, constant_name,
old_value, new_value) per drifted constant, with the exact
sed-ready edit suggestion.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WIKI = REPO_ROOT / "wiki/data"
TESTS = REPO_ROOT / "scripts/test"


# ---------------------------------------------------------------------------
# Drift signature: (test_file, constant_name, json_path, json_key_path)
# ---------------------------------------------------------------------------

# Each entry: (test path, EXPECTED constant name, live source file,
# JSON key path as ['a','b','c'])
DRIFT_SIGNATURES = [
    ("scripts/test/test_cell_count_cross_artifact_parity.py",
     "EXPECTED_TOTAL_CELLS",
     "wiki/data/cell_winner_census.json",
     ["meta", "n_cells_total"]),
    ("scripts/test/test_cell_count_cross_artifact_parity.py",
     "EXPECTED_TIED_WINNER_COUNT",
     "wiki/data/cell_winner_census.json",
     ["meta", "n_tied_winners"]),
    ("scripts/test/test_cell_count_cross_artifact_parity.py",
     "EXPECTED_UNIQUE_WINNER_COUNT",
     "wiki/data/cell_winner_census.json",
     ["meta", "n_unique_winner"]),
    ("scripts/test/test_cell_count_cross_artifact_parity.py",
     "EXPECTED_NO_WINNER_COUNT",
     "wiki/data/cell_winner_census.json",
     ["meta", "n_no_winner"]),
]


def _get_nested(d, keys):
    cur = d
    for k in keys:
        if isinstance(cur, dict):
            cur = cur.get(k)
        elif isinstance(cur, list) and k.isdigit():
            cur = cur[int(k)]
        else:
            return None
        if cur is None:
            return None
    return cur


def _read_constant_from_source(test_path: Path, name: str):
    """Find `<name> = <literal>` at module top-level via AST."""
    if not test_path.exists():
        return None
    try:
        tree = ast.parse(test_path.read_text(encoding="utf-8"))
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    try:
                        return ast.literal_eval(node.value)
                    except Exception:
                        return ast.unparse(node.value)
    return None


def report():
    """Walk DRIFT_SIGNATURES, compare expected (source) vs live (artifact)."""
    lines = []
    n_drifts = 0
    for test_rel, const_name, src_rel, key_path in DRIFT_SIGNATURES:
        test_path = REPO_ROOT / test_rel
        src_path = REPO_ROOT / src_rel
        expected = _read_constant_from_source(test_path, const_name)
        if not src_path.exists():
            lines.append(f"  SKIP {test_rel}::{const_name} - source missing: {src_rel}")
            continue
        try:
            data = json.loads(src_path.read_text())
        except Exception as e:
            lines.append(f"  SKIP {test_rel}::{const_name} - parse error: {e}")
            continue
        actual = _get_nested(data, key_path)
        if expected == actual:
            continue
        n_drifts += 1
        lines.append(f"\n  DRIFT  {test_rel}")
        lines.append(f"    constant: {const_name}")
        lines.append(f"    expected (in test): {expected!r}")
        lines.append(f"    actual   (in {src_rel} at {'.'.join(key_path)}): {actual!r}")
        # Generate sed-ready replacement
        if isinstance(expected, int) and isinstance(actual, int):
            lines.append(f"    suggested: sed -i 's/^{const_name} = {expected}$/{const_name} = {actual}/' {test_rel}")
    if not lines:
        return "No drifts detected. All EXPECTED_* constants match live artifacts."
    return "Detected drift in EXPECTED_* constants:\n" + "\n".join(lines) + \
        f"\n\nTotal drifts: {n_drifts}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    msg = report()
    if not args.quiet:
        print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
