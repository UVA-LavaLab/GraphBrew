#!/usr/bin/env python3
"""Canonical reader for the GraphBrew ``results/`` directory tree.

The pipeline writes three kinds of JSON sidecars, all of which are
flattened to row-shaped dicts here so analysis and figure-generation
code never has to glob/parse manually.

Layout (canonical; see ``results/README.md``)::

    results/
      graphs/<graph>/<graph>.{mtx,sg,el}        source graphs
      vldb_mappings/<graph>/<algo>.{lo,time,json}
          .lo    — binary permutation                (cached reorder output)
          .time  — single-float legacy sidecar       (back-compat)
          .json  — schema reorder_meta/v1            (rich: cmd, env, timing, stdout_tail)
      vldb_runs/<graph>/<algo>__<benchmark>.json   schema kernel_run/v1
      vldb_paper/exp<N>_<name>/<result>.json       aggregated tables (one row per cell)

Public API:

    walk_kernel_runs(root)      -> List[dict]        # per-cell rows from vldb_runs/
    walk_reorder_meta(root)     -> List[dict]        # per-mapping rows from vldb_mappings/*.json
    walk_aggregates(root)       -> List[dict]        # rows from every list-of-dict file under vldb_paper/

    load_runs_df(root)          -> pandas.DataFrame  # flat frame, one row per kernel run
    load_reorder_df(root)       -> pandas.DataFrame
    load_aggregates_df(root)    -> pandas.DataFrame

    build_index(root)           -> dict              # hierarchical manifest
    write_index(root, path=...) -> Path              # also dumps INDEX.json

CLI::

    python3 -m scripts.lib.analysis.results_index --build-index
    python3 -m scripts.lib.analysis.results_index --print-summary
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Repo root: scripts/lib/analysis/results_index.py -> repo
_REPO = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS = _REPO / "results"


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _load_json(p: Path) -> Optional[Any]:
    try:
        with p.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _flatten_timing(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Promote ``timing`` and ``cache`` sub-dicts to top-level columns."""
    out: Dict[str, Any] = {}
    for k, v in rec.items():
        if k in ("timing", "cache") and isinstance(v, dict):
            out.update(v)
        elif k == "stdout_tail":
            continue
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Walkers
# ---------------------------------------------------------------------------

def walk_kernel_runs(root: Path = DEFAULT_RESULTS) -> List[Dict[str, Any]]:
    """Return one flat dict per ``vldb_runs/<graph>/<algo>__<benchmark>.json``."""
    base = Path(root) / "vldb_runs"
    rows: List[Dict[str, Any]] = []
    if not base.is_dir():
        return rows
    for graph_dir in sorted(base.iterdir()):
        if not graph_dir.is_dir():
            continue
        for jf in sorted(graph_dir.glob("*.json")):
            rec = _load_json(jf)
            if not isinstance(rec, dict):
                continue
            row = _flatten_timing(rec)
            row.setdefault("graph", graph_dir.name)
            row["_path"] = str(jf.relative_to(root))
            rows.append(row)
    return rows


def walk_reorder_meta(root: Path = DEFAULT_RESULTS) -> List[Dict[str, Any]]:
    """Return one flat dict per ``vldb_mappings/<graph>/<algo>.json`` (reorder_meta/v1)."""
    base = Path(root) / "vldb_mappings"
    rows: List[Dict[str, Any]] = []
    if not base.is_dir():
        return rows
    for graph_dir in sorted(base.iterdir()):
        if not graph_dir.is_dir():
            continue
        for jf in sorted(graph_dir.glob("*.json")):
            rec = _load_json(jf)
            if not isinstance(rec, dict):
                continue
            row = _flatten_timing(rec)
            row.setdefault("graph", graph_dir.name)
            row["_path"] = str(jf.relative_to(root))
            rows.append(row)
    return rows


def walk_aggregates(root: Path = DEFAULT_RESULTS) -> List[Dict[str, Any]]:
    """Flatten every ``vldb_paper/exp*/*.json`` whose top-level is a list of dicts.

    Tags each row with ``experiment`` (e.g. ``exp2_speedup``) and ``_path``.
    """
    base = Path(root) / "vldb_paper"
    rows: List[Dict[str, Any]] = []
    if not base.is_dir():
        return rows
    for exp_dir in sorted(base.iterdir()):
        if not exp_dir.is_dir():
            continue
        for jf in sorted(exp_dir.glob("*.json")):
            data = _load_json(jf)
            if not isinstance(data, list):
                continue
            for r in data:
                if not isinstance(r, dict):
                    continue
                row = dict(r)
                row["experiment"] = exp_dir.name
                row["_path"] = str(jf.relative_to(root))
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# DataFrame loaders (pandas is optional)
# ---------------------------------------------------------------------------

def _to_df(rows: List[Dict[str, Any]]):
    try:
        import pandas as pd  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "pandas is required for load_*_df(); install with `pip install pandas`."
        ) from e
    return pd.DataFrame(rows)


def load_runs_df(root: Path = DEFAULT_RESULTS):
    return _to_df(walk_kernel_runs(root))


def load_reorder_df(root: Path = DEFAULT_RESULTS):
    return _to_df(walk_reorder_meta(root))


def load_aggregates_df(root: Path = DEFAULT_RESULTS):
    return _to_df(walk_aggregates(root))


# ---------------------------------------------------------------------------
# Hierarchical manifest (INDEX.json)
# ---------------------------------------------------------------------------

def build_index(root: Path = DEFAULT_RESULTS) -> Dict[str, Any]:
    """Build a compact manifest of everything under ``results/``.

    Returns a dict with keys ``graphs``, ``mappings``, ``runs``,
    ``aggregates``, ``schema_versions``, plus summary counts.
    """
    root = Path(root)
    idx: Dict[str, Any] = {
        "schema": "results_index/v1",
        "root": str(root),
        "graphs": {},        # name -> {files: [...]}
        "mappings": {},      # graph -> [{algo_key, has_lo, has_json, has_time, reorder_time}]
        "runs": {},          # graph -> [{algo_key, benchmark, reorder_source, avg_time, ...}]
        "aggregates": {},    # exp_dir -> [filename, ...]
        "counts": {},
    }

    # ---- graphs/
    g_dir = root / "graphs"
    if g_dir.is_dir():
        for gd in sorted(g_dir.iterdir()):
            if gd.is_dir():
                idx["graphs"][gd.name] = {
                    "files": sorted(p.name for p in gd.iterdir() if p.is_file())
                }

    # ---- vldb_mappings/
    for r in walk_reorder_meta(root):
        g = r.get("graph")
        if not g:
            continue
        idx["mappings"].setdefault(g, []).append({
            "algo_key": r.get("algo_key"),
            "reorder_time": r.get("reorder_time"),
            "schema": r.get("schema"),
            "_path": r.get("_path"),
        })

    # ---- vldb_runs/
    for r in walk_kernel_runs(root):
        g = r.get("graph")
        if not g:
            continue
        idx["runs"].setdefault(g, []).append({
            "algo_key": r.get("algo_key"),
            "benchmark": r.get("benchmark"),
            "reorder_source": r.get("reorder_source"),
            "average_time": r.get("average_time"),
            "reorder_time": r.get("reorder_time"),
            "_path": r.get("_path"),
        })

    # ---- vldb_paper/
    pap = root / "vldb_paper"
    if pap.is_dir():
        for ed in sorted(pap.iterdir()):
            if ed.is_dir():
                idx["aggregates"][ed.name] = sorted(p.name for p in ed.glob("*.json"))

    idx["counts"] = {
        "graphs": len(idx["graphs"]),
        "mappings_total": sum(len(v) for v in idx["mappings"].values()),
        "runs_total": sum(len(v) for v in idx["runs"].values()),
        "aggregate_files": sum(len(v) for v in idx["aggregates"].values()),
    }
    # Collect all distinct schema versions seen in any sidecar.
    schemas: set = {"results_index/v1"}
    schemas.update(m.get("schema") for ms in idx["mappings"].values() for m in ms if m.get("schema"))
    for r in walk_kernel_runs(root):
        s = r.get("schema")
        if s:
            schemas.add(s)
    idx["schema_versions"] = sorted(schemas)
    return idx


def write_index(root: Path = DEFAULT_RESULTS, path: Optional[Path] = None) -> Path:
    """Build the manifest and write it to ``<root>/INDEX.json`` (default)."""
    root = Path(root)
    out = Path(path) if path else (root / "INDEX.json")
    idx = build_index(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(idx, f, indent=2)
    tmp.replace(out)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_RESULTS,
                    help=f"Results root (default: {DEFAULT_RESULTS})")
    ap.add_argument("--build-index", action="store_true",
                    help="Write <root>/INDEX.json.")
    ap.add_argument("--print-summary", action="store_true",
                    help="Print counts of graphs/mappings/runs/aggregates.")
    args = ap.parse_args(argv)

    if args.build_index:
        out = write_index(args.root)
        print(f"wrote {out}")
    if args.print_summary or not args.build_index:
        idx = build_index(args.root)
        c = idx["counts"]
        print(f"results root: {idx['root']}")
        print(f"  graphs:           {c['graphs']}")
        print(f"  reorder mappings: {c['mappings_total']}")
        print(f"  kernel runs:      {c['runs_total']}")
        print(f"  aggregate files:  {c['aggregate_files']}")
        print(f"  schemas seen:     {idx['schema_versions']}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
