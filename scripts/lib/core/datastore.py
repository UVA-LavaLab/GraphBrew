#!/usr/bin/env python3
"""
Centralized Data Store for GraphBrew.

Append-only benchmark database that persists **immutable raw observations**.
Each observation is one measured run identified by a unique ``run_id``; the
store never min-collapses distinct ``(graph, algorithm, benchmark)`` tuples.
Distinct labeling, measurement mode, thread policy, mapping identity and
attempt are preserved as separate observations and never collide.

Failures and timeouts are persisted as observations too, but the
compatibility query/training views (``to_list``/``query``/``perf_matrix``/
``get_existing_keys``) default to successful records.

Data file: results/data/benchmarks.json
    Schema: JSON array of raw observation records.  Aggregation (median over
            repeated same-condition observations) is a *derived view*, not the
            stored evidence.

Graph properties: results/data/graph_properties.json
    Schema: JSON dict keyed by graph name, values are feature dicts.

Usage:
    from scripts.lib.core.datastore import BenchmarkStore

    store = BenchmarkStore()           # loads results/data/benchmarks.json
    store.append(results)              # append raw observations, auto-saves
    store.append_from_file("run.json") # ingest a legacy file
    obs = store.observations()         # every raw observation
    data = store.query(benchmark="pr") # successful records only
    matrix = store.perf_matrix()       # {graph: {algo: {bench: median_time}}}
"""

import json
import os
import fcntl
import copy
import shutil
import statistics
import uuid
from datetime import datetime
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Iterable
from collections import defaultdict

from .utils import (
    DATA_DIR, RESULTS_DIR, WEIGHTS_DIR, Logger,
    DISPLAY_TO_CANONICAL, VARIANT_PREFIXES,
    BENCHMARK_OBSERVATION_SCHEMA,
    CONDITION_FIELDS, CONDITION_DISCRIMINATORS,
    benchmark_condition_key, condition_discriminator,
)

log = Logger()

# =============================================================================
# Constants
# =============================================================================

BENCHMARKS_FILE = DATA_DIR / "benchmarks.json"
GRAPH_PROPS_FILE = DATA_DIR / "graph_properties.json"

# Legacy compatibility: the (graph, algorithm, benchmark) identity tuple.  It is
# no longer a dedup key — raw observations are retained — but some callers still
# reference the field names.  The authoritative condition identity is
# ``benchmark_condition_key`` from utils.
_KEY_FIELDS = ("graph", "algorithm", "benchmark")

# In-memory-only sentinels applied to legacy rows (rows loaded without a
# ``run_id``).  These are NEVER written to disk on load.
_LEGACY_LABELING = "legacy-unspecified"
_LEGACY_MEASUREMENT_MODE = "legacy"
_LEGACY_RUN_ID_PREFIX = "legacy-"


class FileLock:
    """Minimal advisory file lock (fcntl-based), mirroring the C++ flock guard.

    Acquires an exclusive lock on ``<path>.lock`` on entry and releases it on
    exit so concurrent official writers cannot clobber each other's appends.
    """

    def __init__(self, target: Path):
        self._lock_path = Path(str(target) + ".lock")
        self._fd = None

    def __enter__(self):
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(str(self._lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None
        return False


# =============================================================================
# Benchmark Store
# =============================================================================

class BenchmarkStore:
    """
    Append-only benchmark database of immutable raw observations.

    Every observation is retained verbatim and identified by a unique
    ``run_id``.  Distinct labeling / measurement mode / thread policy /
    mapping identity / attempt never collapse.  Failures are kept as
    observations; the compatibility views (``to_list``/``query``/
    ``perf_matrix``/``get_existing_keys``) default to successful records.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else BENCHMARKS_FILE
        self._observations: List[dict] = []       # raw ordered observations
        self._by_run_id: Dict[str, dict] = {}     # run_id → record
        self._persisted_records: Dict[str, dict] = {}
        self._persisted_sequence: List[dict] = []
        self._reserved_run_ids: Dict[str, dict] = {}
        self._legacy_counter: int = 0
        self._load_error: Optional[Exception] = None
        self._load()

    # ── persistence ──────────────────────────────────────────────────────

    def _reset(self):
        self._observations = []
        self._by_run_id = {}
        self._persisted_records = {}
        self._persisted_sequence = []
        self._reserved_run_ids = {}
        self._legacy_counter = 0
        self._load_error = None

    def _load(self):
        """Load existing database from disk (never writes the file on load)."""
        if self.path.exists():
            try:
                with open(self.path) as f:
                    data = json.load(f)
                self._ingest_disk_records(data, strict=True)
            except Exception as e:
                self._load_error = e
                log.warning(f"DataStore: failed to load {self.path}: {e}")
        log.info(
            f"DataStore: {len(self._observations)} observations loaded "
            f"from {self.path}"
        )

    def _ingest_disk_records(self, data, *, strict: bool = False):
        """Populate in-memory state from a list of on-disk dicts.

        Legacy rows (no ``run_id``) receive deterministic in-memory-only
        defaults; the file itself is not rewritten here.
        """
        if not isinstance(data, list):
            if strict:
                raise ValueError(
                    f"Benchmark database must be a JSON array: {self.path}"
                )
            return
        for r in data:
            if not isinstance(r, dict):
                if strict:
                    raise ValueError(
                        f"Benchmark database contains a non-object row: {self.path}"
                    )
                continue
            raw = copy.deepcopy(r)
            self._persisted_sequence.append(raw)
            adapted = self._adapt_loaded_record(r)
            if adapted is None:
                run_id = raw.get("run_id")
                if run_id:
                    existing = self._reserved_run_ids.get(run_id)
                    if existing is not None:
                        raise ValueError(
                            f"duplicate run_id in benchmark database: {run_id!r}"
                        )
                    self._reserved_run_ids[run_id] = raw
                continue
            try:
                added = self._add_observation(adapted)
                if added:
                    self._persisted_records[adapted["run_id"]] = raw
            except ValueError as e:
                if strict:
                    raise
                log.warning(f"DataStore: skipping conflicting loaded row: {e}")

    def save(self):
        """Persist observations added with ``save=False``.

        Existing on-disk observations are re-read under the file lock and
        preserved verbatim.  A malformed or conflicting database aborts the
        save rather than being replaced by a partial in-memory view.
        """
        pending = [
            copy.deepcopy(record)
            for record in self._observations
            if record["run_id"] not in self._persisted_records
        ]
        if pending:
            self.append(pending, save=True)

    def _atomic_write(self, records: List[dict]):
        """Serialize ``records`` to ``self.path`` via a process-unique temp file."""
        tmp = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with open(tmp, 'w') as f:
                json.dump(records, f, indent=2)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(str(tmp), str(self.path))
            directory_fd = os.open(str(self.path.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
            raise

    def _serialized_observations(self) -> List[dict]:
        """Return records exactly as they should appear on disk.

        Rows loaded from an existing file retain their original spelling and
        legacy shape; adapter-only defaults remain in memory.  Newly appended
        observations use the versioned condition-aware representation.
        """
        records = copy.deepcopy(self._persisted_sequence)
        records.extend(
            copy.deepcopy(record)
            for record in self._observations
            if record["run_id"] not in self._persisted_records
        )
        return records

    def _mark_persisted(self, records: List[dict]):
        """Snapshot the representation written for each observation."""
        self._persisted_sequence = copy.deepcopy(records)
        by_run_id = {
            record.get("run_id"): record
            for record in records
            if record.get("run_id")
        }
        for view in self._observations:
            run_id = view["run_id"]
            if run_id in by_run_id:
                self._persisted_records[run_id] = copy.deepcopy(
                    by_run_id[run_id]
                )

    # ── core operations ──────────────────────────────────────────────────

    @staticmethod
    def _normalize_algorithm(name: str) -> str:
        """Normalize algorithm display name → canonical training name.

        Variant-prefixed names (``GraphBrewOrder_*``, ``RABBITORDER_*``,
        ``RCM_*``) pass through unchanged.  Known display names are mapped
        via ``DISPLAY_TO_CANONICAL``.  Unknown names pass through as-is.
        """
        if not name:
            return name
        for prefix in VARIANT_PREFIXES:
            if name.startswith(prefix):
                return name
        return DISPLAY_TO_CANONICAL.get(name, name)

    def _adapt_loaded_record(self, record: dict) -> Optional[dict]:
        """Return an in-memory copy of a loaded row with legacy defaults.

        - ``MAP`` records are rejected (MAP is a loading mechanism, not an
          algorithm — the real name must come from the ``.lo`` filename).
        - Algorithm display names are normalized to canonical names.
        - Rows without a ``run_id`` are treated as legacy: they receive a
          deterministic ``legacy-0000000N`` id and ``legacy-unspecified`` /
          ``legacy`` condition sentinels *in memory only*.
        """
        algo = record.get('algorithm', '')
        if algo == 'MAP':
            log.debug(
                f"DataStore: rejected MAP record for "
                f"{record.get('graph', '?')}/{record.get('benchmark', '?')}"
            )
            return None

        rec = dict(record)
        canonical = self._normalize_algorithm(algo)
        if canonical != algo:
            rec['algorithm'] = canonical

        run_id = rec.get('run_id')
        if not run_id:
            self._legacy_counter += 1
            rec['run_id'] = f"{_LEGACY_RUN_ID_PREFIX}{self._legacy_counter:08d}"
            rec.setdefault('labeling', _LEGACY_LABELING)
            rec.setdefault('measurement_mode', _LEGACY_MEASUREMENT_MODE)
        rec.setdefault('algorithm_spec', rec.get('algorithm', ''))
        rec.setdefault('schema', 'benchmark-observation/legacy')
        return rec

    def _prepare_append_record(self, record) -> Optional[dict]:
        """Normalize a to-be-appended record and ensure it carries a run_id."""
        rec = asdict(record) if hasattr(record, '__dataclass_fields__') else dict(record)
        algo = rec.get('algorithm', '')
        if algo == 'MAP':
            log.debug(
                f"DataStore: rejected MAP record for "
                f"{rec.get('graph', '?')}/{rec.get('benchmark', '?')}"
            )
            return None
        canonical = self._normalize_algorithm(algo)
        if canonical != algo:
            rec['algorithm'] = canonical
        schema = rec.get('schema')
        if schema in (None, '', 'benchmark-observation/v1'):
            rec['schema'] = BENCHMARK_OBSERVATION_SCHEMA
        elif schema != BENCHMARK_OBSERVATION_SCHEMA:
            raise ValueError(
                f"Unsupported benchmark observation schema: {schema!r}"
            )
        rec.setdefault('algorithm_spec', rec.get('algorithm', ''))
        if not rec.get('run_id'):
            rec['run_id'] = uuid.uuid4().hex
        return rec

    @staticmethod
    def _content_equal(a: dict, b: dict) -> bool:
        """Order-insensitive content comparison for idempotency checks."""
        try:
            return json.dumps(a, sort_keys=True, default=str) == \
                   json.dumps(b, sort_keys=True, default=str)
        except TypeError:
            return a == b

    def _add_observation(self, record: dict) -> bool:
        """Insert one raw observation.

        Exact-duplicate ``run_id`` with identical content is idempotent; the
        same ``run_id`` with different content raises ``ValueError``.
        """
        run_id = record.get('run_id')
        if not run_id:
            run_id = uuid.uuid4().hex
            record['run_id'] = run_id
        existing = self._by_run_id.get(run_id)
        if existing is not None:
            if self._content_equal(existing, record):
                return False  # idempotent
            raise ValueError(
                f"run_id collision with differing content: {run_id!r}"
            )
        reserved = self._reserved_run_ids.get(run_id)
        if reserved is not None:
            raise ValueError(
                f"run_id collision with excluded persisted record: {run_id!r}"
            )
        self._by_run_id[run_id] = record
        self._reserved_run_ids[run_id] = record
        self._observations.append(record)
        return True

    def append(self, results, save: bool = True):
        """
        Append raw benchmark observations to the store.

        Failed observations are retained (they are excluded only from the
        default query/perf views).  The write is performed under a file lock
        with a re-read-merge-write cycle so concurrent official writers do not
        clobber each other.

        Args:
            results: List of dicts or BenchmarkResult dataclasses.
            save: Persist to disk after appending (default True).
        """
        if not results:
            return

        prepared: List[dict] = []
        for r in results:
            rec = self._prepare_append_record(r)
            if rec is not None:
                prepared.append(rec)
        if not prepared:
            return

        if not save:
            added = 0
            for rec in prepared:
                if self._add_observation(rec):
                    added += 1
            if added:
                log.info(
                    f"DataStore: +{added} observations (unsaved) → "
                    f"{len(self._observations)} total"
                )
            return

        # Saved path: lock, re-read disk, merge, add new, write raw.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(self.path):
            self._reset()
            if self.path.exists():
                with open(self.path) as f:
                    disk = json.load(f)
                self._ingest_disk_records(disk, strict=True)
            added = 0
            for rec in prepared:
                if self._add_observation(rec):
                    added += 1
            if added:
                serialized = self._serialized_observations()
                self._atomic_write(serialized)
                self._mark_persisted(serialized)

        if added:
            s = self.stats()
            log.info(
                f"DataStore: +{added} observations → {s['observations']} total "
                f"({s['successful']} ok, {s['failures']} failed)"
            )

    def append_from_file(self, filepath: str, save: bool = True):
        """Ingest records from a benchmark JSON file (raw append)."""
        with open(filepath) as f:
            data = json.load(f)
        if isinstance(data, list):
            self.append(data, save=save)

    # ── queries ──────────────────────────────────────────────────────────

    def observations(self) -> List[dict]:
        """Return every raw observation (successful and failed), in order."""
        return copy.deepcopy(self._observations)

    def _successful(self) -> List[dict]:
        return [r for r in self._observations if r.get('success', True)]

    def get_existing_keys(self) -> set:
        """Return the set of successful observation condition keys.

        Used by the incremental pipeline to skip runs already present.  The key
        is the shared :func:`benchmark_condition_key`, so producer (resume
        check) and consumer (this store) never build divergent tuples.
        """
        return {benchmark_condition_key(r) for r in self._successful()}

    def to_list(self, include_failed: bool = False) -> List[dict]:
        """Return records as a sorted list (successful-only by default)."""
        records = self._observations if include_failed else self._successful()
        out = copy.deepcopy(records)
        out.sort(key=lambda r: (r.get('graph', ''), r.get('benchmark', ''),
                                r.get('algorithm', '')))
        return out

    def query(self, graph: str = None, algorithm: str = None,
              benchmark: str = None, algorithm_spec: str = None,
              include_failed: bool = False) -> List[dict]:
        """Filter records by any combination of fields (successful-only default)."""
        source = self._observations if include_failed else self._successful()
        out = []
        for r in source:
            if graph and r.get('graph') != graph:
                continue
            if algorithm and r.get('algorithm') != algorithm:
                continue
            if algorithm_spec and r.get('algorithm_spec') != algorithm_spec:
                continue
            if benchmark and r.get('benchmark') != benchmark:
                continue
            out.append(r)
        return copy.deepcopy(out)

    def graphs(self) -> List[str]:
        """Return sorted list of unique graph names (successful records)."""
        return sorted(set(r.get('graph', '') for r in self._successful()))

    def algorithms(self) -> List[str]:
        """Return sorted list of unique algorithm names (successful records)."""
        return sorted(set(r.get('algorithm', '') for r in self._successful()))

    def benchmarks(self) -> List[str]:
        """Return sorted list of unique benchmark names (successful records)."""
        return sorted(set(r.get('benchmark', '') for r in self._successful()))

    def perf_matrix(
        self,
        algorithm_spec: str = None,
        labeling: str = None,
        measurement_mode: str = None,
        threads: int = None,
        mapping_identity_id: str = None,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Build performance matrix ``{graph: {algo: {bench: median_time}}}``.

        Successful, strictly-positive observations are aggregated by **median**
        over repeated attempts of the *same* measurement condition.  If more
        than one distinct condition exists for the same graph/algorithm/
        benchmark and no explicit condition filter was supplied, this fails
        closed (raises ``ValueError``) rather than silently mixing conditions.

        Optional filters restrict to a single condition; supply them to
        disambiguate an intentionally multi-condition store.
        """
        explicit_filter = any(
            v is not None
            for v in (
                algorithm_spec,
                labeling,
                measurement_mode,
                threads,
                mapping_identity_id,
            )
        )

        # group[(graph, algo, bench)][discriminator] = [times...]
        group: Dict[Tuple[str, str, str],
                    Dict[Tuple, List[float]]] = defaultdict(
                        lambda: defaultdict(list))
        for r in self._observations:
            if not r.get('success', True):
                continue
            if algorithm_spec is not None and \
                    r.get('algorithm_spec') != algorithm_spec:
                continue
            t = r.get('time_seconds', 0.0)
            if not (isinstance(t, (int, float)) and t > 0):
                continue
            if labeling is not None and r.get('labeling') != labeling:
                continue
            if measurement_mode is not None and \
                    r.get('measurement_mode') != measurement_mode:
                continue
            if threads is not None and r.get('threads') != threads:
                continue
            if mapping_identity_id is not None and \
                    r.get('mapping_identity_id') != mapping_identity_id:
                continue
            g = r.get('graph', '')
            a = r.get('algorithm', '')
            b = r.get('benchmark', '')
            if not (g and a and b):
                continue
            group[(g, a, b)][condition_discriminator(r)].append(float(t))

        matrix: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        for (g, a, b), by_condition in group.items():
            if len(by_condition) > 1:
                if not explicit_filter:
                    raise ValueError(
                        "perf_matrix: multiple measurement conditions for "
                        f"{g!r}/{a!r}/{b!r}: {sorted(by_condition.keys())}. "
                        "Supply a condition filter (labeling / measurement_mode "
                        "/ algorithm_spec / threads / mapping_identity_id) "
                        "to disambiguate."
                    )
                # Even with a filter, refuse to mix conditions.
                raise ValueError(
                    "perf_matrix: condition filter still leaves multiple "
                    f"conditions for {g!r}/{a!r}/{b!r}: "
                    f"{sorted(by_condition.keys())}."
                )
            (times,) = by_condition.values()
            matrix[g][a][b] = statistics.median(times)
        return dict(matrix)

    def stats(self) -> Dict[str, Any]:
        """Return summary statistics distinguishing totals and outcomes."""
        successful = self._successful()
        failures = len(self._observations) - len(successful)
        return {
            'observations': len(self._observations),
            'successful': len(successful),
            'failures': failures,
            # Backward-compatible alias (successful record count).
            'records': len(successful),
            'graphs': len(self.graphs()),
            'algorithms': len(self.algorithms()),
            'benchmarks': len(self.benchmarks()),
        }

    def __len__(self):
        return len(self._observations)

    def __repr__(self):
        s = self.stats()
        return (f"BenchmarkStore({s['observations']} observations, "
                f"{s['successful']} ok, {s['failures']} failed, "
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
# Unified Model Export: single load-only adaptive_models.json
# =============================================================================

def export_unified_models(
    out_path: Optional[Path] = None,
    weights_dir: Optional[Path] = None,
    tier0_trained: bool = False,
) -> Path:
    """Merge offline perceptron staging files into adaptive_models.json.

    Existing decision-tree, hybrid, and random-forest sections are preserved.
    """
    out_path = Path(out_path) if out_path else ADAPTIVE_MODELS_FILE
    data = {}
    if out_path.is_file():
        with open(out_path) as stream:
            loaded = json.load(stream)
        if isinstance(loaded, dict):
            data = loaded

    active_dir = Path(weights_dir or WEIGHTS_DIR) / "type_0"
    weights_file = active_dir / "weights.json"
    if not weights_file.is_file():
        raise FileNotFoundError(
            f"Offline perceptron weights not found: {weights_file}"
        )
    with open(weights_file) as stream:
        weights = json.load(stream)
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Offline perceptron weights must be a non-empty object")
    from scripts.lib.ml.portfolio import (
        DEPLOYABLE_ARM_SPECS,
        normalize_deployable_portfolio,
    )
    from scripts.lib.ml.feature_schema import validate_tier0_weight_entry

    def normalize_portfolio(payload: Dict) -> Dict:
        normalized = normalize_deployable_portfolio(payload)
        for spec, entry in normalized.items():
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Offline adaptive arm {spec} must be an object")
            validate_tier0_weight_entry(entry, spec)
        return {
            spec: normalized[spec]
            for spec in DEPLOYABLE_ARM_SPECS
        }

    weights = normalize_portfolio(weights)

    per_benchmark = {}
    for path in sorted(active_dir.glob("*.json")):
        if path.name == "weights.json":
            continue
        with open(path) as stream:
            payload = json.load(stream)
        if isinstance(payload, dict) and payload:
            per_benchmark[path.stem] = normalize_portfolio(payload)
    data["perceptron"] = {
        "schema": "adaptive-tier0/v1",
        "tier0_trained": bool(tier0_trained),
        "weights": weights,
        "per_benchmark": per_benchmark,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(temporary, "w") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, out_path)
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
