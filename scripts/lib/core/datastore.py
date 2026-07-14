#!/usr/bin/env python3
"""
Centralized Data Store for GraphBrew.

Single-file append-only benchmark database with automatic deduplication.
All models (perceptron, decision tree, hybrid) read from this one source.

Data file: results/data/benchmarks.json
    Schema: JSON array of records, each with a unique composite key
            (graph, algorithm, benchmark). Best time is kept on conflict.

Graph properties: results/data/graph_properties.json
    Schema: JSON dict keyed by graph name, values are feature dicts.

Usage:
    from scripts.lib.core.datastore import BenchmarkStore

    store = BenchmarkStore()           # loads results/data/benchmarks.json
    store.append(results)              # append + dedup, auto-saves
    store.append_from_file("run.json") # ingest a legacy file
    data = store.query(benchmark="pr") # filter records
    matrix = store.perf_matrix()       # {graph: {algo: {bench: time}}}
"""

import json
import os
import shutil
from datetime import datetime
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

from .utils import DATA_DIR, RESULTS_DIR, Logger, DISPLAY_TO_CANONICAL, VARIANT_PREFIXES

log = Logger()

# =============================================================================
# Constants
# =============================================================================

BENCHMARKS_FILE = DATA_DIR / "benchmarks.json"
GRAPH_PROPS_FILE = DATA_DIR / "graph_properties.json"

# Composite key for deduplication: (graph, algorithm, benchmark)
# When a duplicate key is found, the record with the lowest time_seconds wins.
_KEY_FIELDS = ("graph", "algorithm", "benchmark")


# =============================================================================
# Benchmark Store
# =============================================================================

class BenchmarkStore:
    """
    Append-only benchmark database backed by a single JSON file.

    Records are deduplicated by (graph, algorithm, benchmark).
    On conflict the record with the lowest time_seconds is kept.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else BENCHMARKS_FILE
        self._records: Dict[tuple, dict] = {}  # key → record
        self._load()

    # ── persistence ──────────────────────────────────────────────────────

    def _load(self):
        """Load existing database from disk."""
        if self.path.exists():
            try:
                with open(self.path) as f:
                    data = json.load(f)
                for r in data:
                    if isinstance(r, dict):
                        self._insert(r)
            except Exception as e:
                log.warning(f"DataStore: failed to load {self.path}: {e}")
        log.info(f"DataStore: {len(self._records)} records loaded from {self.path}")

    def save(self):
        """Write database to disk (atomic via temp file)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.tmp')
        with open(tmp, 'w') as f:
            json.dump(self.to_list(), f, indent=2)
        shutil.move(str(tmp), str(self.path))

    # ── core operations ──────────────────────────────────────────────────

    @staticmethod
    def _normalize_algorithm(name: str) -> str:
        """Normalize algorithm display name → canonical training name.

        Variant-prefixed names (``GraphBrewOrder_*``, ``RABBITORDER_*``,
        ``RCM_*``) pass through unchanged.  Known display names are mapped
        via ``DISPLAY_TO_CANONICAL``.  Unknown names pass through as-is.

        ``MAP`` is explicitly rejected — callers must resolve the real
        algorithm identity from the ``.lo`` filename before inserting.
        """
        if not name:
            return name
        for prefix in VARIANT_PREFIXES:
            if name.startswith(prefix):
                return name
        return DISPLAY_TO_CANONICAL.get(name, name)

    def _key(self, record: dict) -> Optional[tuple]:
        """Extract composite key from a record. Returns None if incomplete."""
        vals = tuple(record.get(f, '') for f in _KEY_FIELDS)
        if all(vals):
            return vals
        return None

    def _insert(self, record: dict):
        """Insert a record, keeping the one with lower time on conflict.

        B4 safeguards:
        - Records with algorithm="MAP" are rejected (MAP is a mechanism,
          not an algorithm — the caller must resolve the real name from
          the .lo filename before storing).
        - Algorithm names are auto-normalized via DISPLAY_TO_CANONICAL
          so legacy display names (e.g. "RabbitOrder") become canonical
          training names (e.g. "RABBITORDER_csr").
        """
        algo = record.get('algorithm', '')
        if algo == 'MAP':
            log.debug(f"DataStore: rejected MAP record for "
                      f"{record.get('graph', '?')}/{record.get('benchmark', '?')} "
                      f"— resolve to real algorithm name before inserting")
            return

        # Auto-normalize algorithm name
        canonical = self._normalize_algorithm(algo)
        if canonical != algo:
            record['algorithm'] = canonical

        key = self._key(record)
        if key is None:
            return
        existing = self._records.get(key)
        if existing is None:
            self._records[key] = record
        else:
            # Keep the faster time (lower is better)
            new_time = record.get('time_seconds', float('inf'))
            old_time = existing.get('time_seconds', float('inf'))
            if new_time < old_time:
                self._records[key] = record

    def append(self, results, save: bool = True):
        """
        Append benchmark results to the store.

        Args:
            results: List of dicts or BenchmarkResult dataclasses.
            save: Auto-save after appending (default True).
        """
        if not results:
            return
        added = 0
        updated = 0
        for r in results:
            rec = asdict(r) if hasattr(r, '__dataclass_fields__') else dict(r)
            if not rec.get('success', True):
                continue  # skip failed runs
            key = self._key(rec)
            if key is None:
                continue
            was_present = key in self._records
            old_time = self._records[key].get('time_seconds', float('inf')) if was_present else float('inf')
            self._insert(rec)
            if not was_present:
                added += 1
            elif rec.get('time_seconds', float('inf')) < old_time:
                updated += 1

        if added or updated:
            log.info(f"DataStore: +{added} new, {updated} updated → "
                     f"{len(self._records)} total records")
            if save:
                self.save()

    def append_from_file(self, filepath: str, save: bool = True):
        """Ingest records from a legacy benchmark JSON file."""
        with open(filepath) as f:
            data = json.load(f)
        if isinstance(data, list):
            self.append(data, save=save)

    # ── queries ──────────────────────────────────────────────────────────

    def get_existing_keys(self) -> set:
        """Return the set of ``(graph, algorithm, benchmark)`` composite keys.

        This is used by the incremental pipeline to skip runs that are
        already present in the database.
        """
        return set(self._records.keys())

    def to_list(self) -> List[dict]:
        """Return all records as a sorted list."""
        records = list(self._records.values())
        records.sort(key=lambda r: (r.get('graph', ''), r.get('benchmark', ''),
                                     r.get('algorithm', '')))
        return records

    def query(self, graph: str = None, algorithm: str = None,
              benchmark: str = None) -> List[dict]:
        """Filter records by any combination of fields."""
        out = []
        for r in self._records.values():
            if graph and r.get('graph') != graph:
                continue
            if algorithm and r.get('algorithm') != algorithm:
                continue
            if benchmark and r.get('benchmark') != benchmark:
                continue
            out.append(r)
        return out

    def graphs(self) -> List[str]:
        """Return sorted list of unique graph names."""
        return sorted(set(r.get('graph', '') for r in self._records.values()))

    def algorithms(self) -> List[str]:
        """Return sorted list of unique algorithm names."""
        return sorted(set(r.get('algorithm', '') for r in self._records.values()))

    def benchmarks(self) -> List[str]:
        """Return sorted list of unique benchmark names."""
        return sorted(set(r.get('benchmark', '') for r in self._records.values()))

    def perf_matrix(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Build performance matrix: {graph: {algo: {bench: best_time}}}.

        This is the standard format consumed by
        perceptron training, evaluation scripts, and C++ runtime.
        """
        matrix: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        for r in self._records.values():
            g = r.get('graph', '')
            a = r.get('algorithm', '')
            b = r.get('benchmark', '')
            t = r.get('time_seconds', float('inf'))
            if g and a and b:
                matrix[g][a][b] = t  # already deduplicated to best time
        return dict(matrix)

    def stats(self) -> Dict[str, Any]:
        """Return summary statistics."""
        return {
            'records': len(self._records),
            'graphs': len(self.graphs()),
            'algorithms': len(self.algorithms()),
            'benchmarks': len(self.benchmarks()),
        }

    def __len__(self):
        return len(self._records)

    def __repr__(self):
        s = self.stats()
        return (f"BenchmarkStore({s['records']} records, "
                f"{s['graphs']} graphs, {s['algorithms']} algos, "
                f"{s['benchmarks']} benchmarks)")


# =============================================================================
# Graph Properties Store
# =============================================================================

class GraphPropsStore:
    """
    Centralized graph properties database.

    One file: results/data/graph_properties.json
    Schema: {graph_name: {feature: value, ...}, ...}
    Auto-append: new features merge into existing entries.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else GRAPH_PROPS_FILE
        self._props: Dict[str, Dict] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path) as f:
                    self._props = json.load(f)
            except Exception:
                self._props = {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.tmp')
        with open(tmp, 'w') as f:
            json.dump(self._props, f, indent=2)
        shutil.move(str(tmp), str(self.path))

    def update(self, graph_name: str, properties: Dict):
        """Merge new properties into existing entry (non-None values only)."""
        if graph_name not in self._props:
            self._props[graph_name] = {}
        for k, v in properties.items():
            if v is not None:
                self._props[graph_name][k] = v

    def get(self, graph_name: str) -> Optional[Dict]:
        return self._props.get(graph_name)

    def all(self) -> Dict[str, Dict]:
        return dict(self._props)

    def graphs(self) -> List[str]:
        return sorted(self._props.keys())

    def __len__(self):
        return len(self._props)

    def __repr__(self):
        return f"GraphPropsStore({len(self._props)} graphs)"


# =============================================================================
# Module-level convenience functions
# =============================================================================

# DEPRECATED: C++ now trains models from DB at runtime (train_all_models).
# adaptive_models.json is no longer the mechanism for model delivery.
ADAPTIVE_MODELS_FILE = DATA_DIR / "adaptive_models.json"

_benchmark_store: Optional[BenchmarkStore] = None
_props_store: Optional[GraphPropsStore] = None


def get_benchmark_store() -> BenchmarkStore:
    """Get or create the global BenchmarkStore singleton."""
    global _benchmark_store
    if _benchmark_store is None:
        _benchmark_store = BenchmarkStore()
    return _benchmark_store


def get_props_store() -> GraphPropsStore:
    """Get or create the global GraphPropsStore singleton."""
    global _props_store
    if _props_store is None:
        _props_store = GraphPropsStore()
    return _props_store


# =============================================================================
# Unified Model Export: single adaptive_models.json
# DEPRECATED: C++ now trains models from raw DB data at runtime.
# This function is retained only for legacy compatibility.
# =============================================================================

def export_unified_models(out_path: Optional[Path] = None) -> Path:
    """
    DEPRECATED: C++ now trains perceptron, DT, and hybrid models from
    benchmarks.json + graph_properties.json at load time.  This function
    is a no-op stub retained for backward compatibility.
    """
    import warnings
    warnings.warn(
        "export_unified_models() is deprecated: C++ trains models at runtime.",
        DeprecationWarning, stacklevel=2,
    )
    out_path = Path(out_path) if out_path else ADAPTIVE_MODELS_FILE
    return out_path


# =============================================================================
# Migration: import old scattered files into the centralized store
# =============================================================================

def migrate_legacy_files(results_dir: str = None, dry_run: bool = False) -> Dict[str, Any]:
    """
    Import all legacy benchmark_*.json files into the centralized store.

    Args:
        results_dir: Directory containing old benchmark files (default: results/)
        dry_run: If True, report what would happen without writing.

    Returns:
        Migration summary dict.
    """
    results_dir = Path(results_dir) if results_dir else RESULTS_DIR
    store = BenchmarkStore()

    old_count = len(store)
    files_imported = []

    for f in sorted(results_dir.glob("benchmark_*.json")):
        # Skip the new centralized file
        if f.parent == DATA_DIR:
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
            if isinstance(data, list) and data:
                store.append(data, save=False)
                files_imported.append(f.name)
        except Exception as e:
            log.warning(f"Skipping {f.name}: {e}")

    new_count = len(store)

    summary = {
        'files_imported': files_imported,
        'records_before': old_count,
        'records_after': new_count,
        'records_added': new_count - old_count,
    }

    if not dry_run:
        store.save()
        log.info(f"Migration: {old_count} → {new_count} records "
                 f"({new_count - old_count} new from {len(files_imported)} files)")

    return summary


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI for data store operations."""
    import argparse
    parser = argparse.ArgumentParser(description="GraphBrew Data Store")
    parser.add_argument('--stats', action='store_true', help="Show database statistics")
    parser.add_argument('--migrate', action='store_true',
                        help="Import legacy benchmark_*.json files into centralized store")
    parser.add_argument('--dry-run', action='store_true',
                        help="Show what migration would do without writing")
    parser.add_argument('--ingest', type=str,
                        help="Ingest a specific JSON file into the store")
    parser.add_argument('--export', type=str,
                        help="Export store to a specific file path")
    parser.add_argument('--export-models', action='store_true',
                        help="Merge all trained models into results/data/adaptive_models.json")
    args = parser.parse_args()

    if args.migrate or args.dry_run:
        summary = migrate_legacy_files(dry_run=args.dry_run)
        print(f"Files: {len(summary['files_imported'])}")
        for f in summary['files_imported']:
            print(f"  {f}")
        print(f"Records: {summary['records_before']} → {summary['records_after']} "
              f"(+{summary['records_added']})")
        return

    if args.ingest:
        store = BenchmarkStore()
        store.append_from_file(args.ingest)
        print(store)
        return

    if args.export:
        store = BenchmarkStore()
        with open(args.export, 'w') as f:
            json.dump(store.to_list(), f, indent=2)
        print(f"Exported {len(store)} records to {args.export}")
        return

    if args.export_models:
        import warnings
        warnings.warn(
            "--export-models is deprecated: C++ now trains models from "
            "raw DB data at runtime (no adaptive_models.json needed).",
            DeprecationWarning, stacklevel=2)
        path = export_unified_models()
        print(f"Exported unified models to {path}")
        return

    # Default: show stats
    store = BenchmarkStore()
    print(store)
    s = store.stats()
    print(f"  Graphs:     {', '.join(store.graphs())}")
    print(f"  Benchmarks: {', '.join(store.benchmarks())}")
    print(f"  Algorithms: {len(store.algorithms())}")


if __name__ == '__main__':
    main()
