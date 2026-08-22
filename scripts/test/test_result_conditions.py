#!/usr/bin/env python3
"""
Condition-aware result-storage tests.
=====================================

Verify the immutable raw-observation model of :class:`BenchmarkStore` and the
shared condition-key SSOT:

  * distinct labeling / measurement mode / thread policy / mapping identity /
    attempt never collapse into one another;
  * failures are persisted as observations but excluded from the default
    query / perf views;
  * duplicate ``run_id`` with differing content is rejected;
  * ``perf_matrix`` medians repeated same-condition observations and fails
    closed on mixed conditions unless disambiguated by a filter;
  * the resume condition key has the correct field order and distinguishes
    direct from pre-generated MAP inputs;
  * the generic ``run_benchmark`` disables inherited C++ self-recording even
    when an ambient ``GRAPHBREW_DB_DIR`` is set;
  * explicit C++ self-recording accumulates raw observations (never
    replace-on-faster);
  * a single C++-written record still loads compatibly.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.core.datastore import BenchmarkStore  # noqa: E402
from scripts.lib.core.experiment_policy import (  # noqa: E402
    REORDER_SEMANTICS_VERSION,
)
from scripts.lib.core.utils import (  # noqa: E402
    BenchmarkResult,
    BENCHMARK_OBSERVATION_SCHEMA,
    CONDITION_FIELDS,
    benchmark_condition_key,
    benchmark_request_key,
    condition_discriminator,
)

BIN_DIR = PROJECT_ROOT / "bench" / "bin"
TINY_GRAPH = PROJECT_ROOT / "scripts" / "test" / "data" / "tiny.el"


def _result(**overrides) -> BenchmarkResult:
    """Build a BenchmarkResult with sensible defaults for the store tests."""
    base = dict(
        graph="g",
        algorithm="ORIGINAL",
        algorithm_id=0,
        benchmark="pr",
        time_seconds=1.0,
        nodes=4,
        edges=6,
        success=True,
        trials=1,
    )
    base.update(overrides)
    return BenchmarkResult(**base)


# =============================================================================
# Raw observation retention
# =============================================================================

def test_store_retains_two_labelings_and_two_attempts(tmp_path):
    """Same graph/algo/bench, two labelings × two attempts → four raw rows."""
    path = tmp_path / "benchmarks.json"
    store = BenchmarkStore(path)
    store.append([
        _result(time_seconds=1.0, labeling="natural", attempt=1),
        _result(time_seconds=2.0, labeling="natural", attempt=2),
        _result(time_seconds=3.0, labeling="shuffled", attempt=1),
        _result(time_seconds=4.0, labeling="shuffled", attempt=2),
    ])
    assert len(store.observations()) == 4

    # Persisted and re-read from disk with all four retained.
    reloaded = BenchmarkStore(path)
    obs = reloaded.observations()
    assert len(obs) == 4
    assert {o["labeling"] for o in obs} == {"natural", "shuffled"}
    assert {o["attempt"] for o in obs} == {1, 2}
    # Four distinct run_ids, four distinct condition keys.
    assert len({o["run_id"] for o in obs}) == 4
    assert len({benchmark_condition_key(o) for o in obs}) == 4


def test_failure_persisted_but_excluded_from_default_views(tmp_path):
    """A failed observation is stored but hidden from query/perf by default."""
    path = tmp_path / "benchmarks.json"
    store = BenchmarkStore(path)
    store.append([
        _result(time_seconds=1.0, success=True),
        _result(time_seconds=0.0, success=False, error="timeout",
                labeling="natural", attempt=2),
    ])

    # Both observations retained on disk.
    assert len(store.observations()) == 2
    stats = store.stats()
    assert stats["observations"] == 2
    assert stats["successful"] == 1
    assert stats["failures"] == 1

    # Default views are successful-only.
    assert len(store.query(graph="g")) == 1
    assert len(store.to_list()) == 1
    # Opt-in view exposes the failure.
    assert len(store.query(graph="g", include_failed=True)) == 2

    # Failure excluded from perf and from resume keys.
    pm = store.perf_matrix()
    assert pm["g"]["ORIGINAL"]["pr"] == 1.0
    assert len(store.get_existing_keys()) == 1


def test_returned_observations_cannot_mutate_store(tmp_path):
    """Public raw/query views are copies despite the immutable store model."""
    store = BenchmarkStore(tmp_path / "benchmarks.json")
    store.append([_result(run_id="immutable-1", time_seconds=1.0)])

    observations = store.observations()
    observations[0]["time_seconds"] = 99.0
    queried = store.query(graph="g")
    queried[0]["time_seconds"] = 88.0

    assert store.observations()[0]["time_seconds"] == 1.0


# =============================================================================
# run_id idempotency / collision
# =============================================================================

def test_duplicate_run_id_identical_content_is_idempotent(tmp_path):
    path = tmp_path / "benchmarks.json"
    store = BenchmarkStore(path)
    store.append([_result(run_id="fixed-1", time_seconds=1.0)])
    # Re-appending the identical observation is a no-op.
    store.append([_result(run_id="fixed-1", time_seconds=1.0)])
    assert len(store.observations()) == 1


def test_duplicate_run_id_mismatch_raises(tmp_path):
    path = tmp_path / "benchmarks.json"
    store = BenchmarkStore(path)
    store.append([_result(run_id="fixed-1", time_seconds=1.0)])
    with pytest.raises(ValueError, match="run_id collision"):
        store.append([_result(run_id="fixed-1", time_seconds=2.0)])


def test_stale_store_instance_reloads_before_append(tmp_path):
    """Independent writers merge under the lock instead of clobbering."""
    path = tmp_path / "benchmarks.json"
    first = BenchmarkStore(path)
    second = BenchmarkStore(path)

    first.append([_result(run_id="writer-1", time_seconds=1.0)])
    second.append([_result(run_id="writer-2", time_seconds=2.0)])

    assert {
        row["run_id"] for row in BenchmarkStore(path).observations()
    } == {"writer-1", "writer-2"}


# =============================================================================
# perf_matrix aggregation semantics
# =============================================================================

def test_perf_matrix_medians_repeated_same_condition(tmp_path):
    """Repeated attempts of one condition aggregate by median, not min."""
    path = tmp_path / "benchmarks.json"
    store = BenchmarkStore(path)
    store.append([
        _result(time_seconds=1.0, attempt=1),
        _result(time_seconds=2.0, attempt=2),
        _result(time_seconds=9.0, attempt=3),
    ])
    pm = store.perf_matrix()
    # median(1, 2, 9) == 2.0 (min would be 1.0)
    assert pm["g"]["ORIGINAL"]["pr"] == 2.0


def test_perf_matrix_fails_closed_on_mixed_conditions(tmp_path):
    """Mixed measurement conditions without a filter fail closed."""
    path = tmp_path / "benchmarks.json"
    store = BenchmarkStore(path)
    store.append([
        _result(time_seconds=1.0, labeling="natural"),
        _result(time_seconds=3.0, labeling="shuffled"),
    ])
    with pytest.raises(ValueError, match="multiple measurement conditions"):
        store.perf_matrix()

    # A filter disambiguates to a single condition.
    assert store.perf_matrix(labeling="natural")["g"]["ORIGINAL"]["pr"] == 1.0
    assert store.perf_matrix(labeling="shuffled")["g"]["ORIGINAL"]["pr"] == 3.0


def test_perf_matrix_filter_still_refuses_to_mix(tmp_path):
    """Even with a filter, remaining multi-condition rows fail closed."""
    path = tmp_path / "benchmarks.json"
    store = BenchmarkStore(path)
    store.append([
        _result(time_seconds=1.0, labeling="natural", threads=8),
        _result(time_seconds=3.0, labeling="natural", threads=16),
    ])
    with pytest.raises(ValueError, match="still leaves multiple"):
        store.perf_matrix(labeling="natural")
    # Fully-qualified filter resolves to one condition.
    pm = store.perf_matrix(labeling="natural", threads=8)
    assert pm["g"]["ORIGINAL"]["pr"] == 1.0


def test_perf_matrix_never_mixes_algorithm_specs(tmp_path):
    store = BenchmarkStore(tmp_path / "benchmarks.json")
    store.append([
        _result(
            algorithm="LeidenOrder",
            algorithm_id=15,
            algorithm_spec="15:0.5",
            time_seconds=1.0,
        ),
        _result(
            algorithm="LeidenOrder",
            algorithm_id=15,
            algorithm_spec="15:2.0",
            time_seconds=2.0,
        ),
    ])

    with pytest.raises(ValueError, match="multiple measurement conditions"):
        store.perf_matrix()
    assert store.perf_matrix(
        algorithm_spec="15:0.5"
    )["g"]["LeidenOrder"]["pr"] == 1.0


def test_perf_matrix_never_mixes_mapping_draws(tmp_path):
    store = BenchmarkStore(tmp_path / "benchmarks.json")
    store.append([
        _result(
            time_seconds=1.0,
            mapping_fingerprint="draw-a",
        ),
        _result(
            time_seconds=2.0,
            mapping_fingerprint="draw-b",
        ),
    ])

    with pytest.raises(ValueError, match="multiple measurement conditions"):
        store.perf_matrix()
    assert store.perf_matrix(
        mapping_fingerprint="draw-a"
    )["g"]["ORIGINAL"]["pr"] == 1.0


# =============================================================================
# Condition key: order, fields, direct vs MAP
# =============================================================================

def test_condition_key_field_order():
    """The key includes exact algorithm spec before benchmark."""
    assert CONDITION_FIELDS[:6] == (
        "graph",
        "algorithm",
        "reorder_semantics_version",
        "requested_algorithm_spec",
        "algorithm_spec",
        "benchmark",
    )
    key = benchmark_condition_key({
        "graph": "gA", "algorithm": "GORDER",
        "reorder_semantics_version": "graphbrew-reorder/v2",
        "requested_algorithm_spec": "9:csr",
        "algorithm_spec": "9:csr", "benchmark": "pr",
        "labeling": "natural", "measurement_mode": "process", "threads": 16,
        "mapping_identity_id": "direct", "attempt": 1,
    })
    # Positional order must match CONDITION_FIELDS, catching any regression to
    # the old (graph, benchmark, algorithm) tuple ordering.
    assert key == (
        "gA", "GORDER", "graphbrew-reorder/v2",
        "9:csr", "9:csr", "pr",
        "natural", "process", 16, "direct", "", 1)


def test_condition_key_distinguishes_direct_vs_map_and_modes():
    common = dict(graph="gA", algorithm="GORDER",
                  reorder_semantics_version="graphbrew-reorder/v2",
                  requested_algorithm_spec="9:csr",
                  algorithm_spec="9:csr", benchmark="pr",
                  labeling="natural", measurement_mode="process", threads=16,
                  attempt=1)
    k_direct = benchmark_condition_key({**common, "mapping_identity_id": "direct"})
    k_map = benchmark_condition_key(
        {**common, "mapping_identity_id": "map:GORDER.lo"})
    assert k_direct != k_map

    # Measurement mode also discriminates.
    k_self = benchmark_condition_key(
        {**common, "mapping_identity_id": "direct",
         "measurement_mode": "self-record"})
    assert k_self != k_direct

    # The perf-aggregation discriminator excludes graph/algo/bench/attempt.
    assert condition_discriminator({**common, "mapping_identity_id": "direct"}) == (
        "graphbrew-reorder/v2", "9:csr",
        "natural", "process", 16, "direct", "")


def test_condition_key_distinguishes_algorithm_specs():
    common = {
        "graph": "g",
        "algorithm": "LeidenOrder",
        "benchmark": "pr",
    }
    assert benchmark_condition_key({
        **common, "algorithm_spec": "15:0.5",
    }) != benchmark_condition_key({
        **common, "algorithm_spec": "15:2.0",
    })


def test_resume_key_matches_store_existing_key(tmp_path):
    """A resume key built from named fields matches get_existing_keys()."""
    path = tmp_path / "benchmarks.json"
    store = BenchmarkStore(path)
    store.append([
        _result(algorithm="GORDER", algorithm_id=8, time_seconds=1.0,
                requested_algorithm_spec="9:csr",
                algorithm_spec="9:csr",
                labeling="natural", measurement_mode="process", threads=16,
                mapping_identity_id="direct", attempt=1),
    ])
    existing = store.get_existing_request_keys()

    resume_direct = benchmark_request_key({
        "graph": "g", "algorithm": "GORDER",
        "reorder_semantics_version": REORDER_SEMANTICS_VERSION,
        "requested_algorithm_spec": "9:csr", "benchmark": "pr",
        "labeling": "natural", "measurement_mode": "process", "threads": 16,
        "mapping_identity_id": "direct", "attempt": 1,
    })
    assert resume_direct in existing

    # A MAP-mode run of the same algorithm is a *different* condition, so a
    # direct-only record must NOT satisfy its resume check.
    resume_map = benchmark_request_key({
        "graph": "g", "algorithm": "GORDER",
        "reorder_semantics_version": REORDER_SEMANTICS_VERSION,
        "requested_algorithm_spec": "13:map:GORDER.lo",
        "benchmark": "pr",
        "labeling": "natural", "measurement_mode": "process", "threads": 16,
        "mapping_identity_id": "map:GORDER.lo", "attempt": 1,
    })
    assert resume_map not in existing


# =============================================================================
# Legacy adapter (in-memory only, no rewrite on load)
# =============================================================================

def test_legacy_row_gets_in_memory_defaults_without_rewriting(tmp_path):
    path = tmp_path / "benchmarks.json"
    legacy = [{
        "graph": "g", "algorithm": "Original", "benchmark": "pr",
        "time_seconds": 1.5, "nodes": 4, "edges": 6, "success": True,
    }]
    path.write_text(json.dumps(legacy))
    mtime_before = path.stat().st_mtime_ns

    store = BenchmarkStore(path)
    obs = store.observations()
    assert len(obs) == 1
    assert obs[0]["run_id"] == "legacy-00000001"
    assert obs[0]["labeling"] == "legacy-unspecified"
    assert obs[0]["measurement_mode"] == "legacy"
    assert obs[0]["algorithm_spec"] == "legacy-derived:ORIGINAL"
    # Display name normalized to the canonical training name.
    assert obs[0]["algorithm"] == "ORIGINAL"

    # Loading must NOT rewrite the file (frozen artifacts stay byte-identical).
    assert path.stat().st_mtime_ns == mtime_before
    assert json.loads(path.read_text()) == legacy


def test_append_preserves_legacy_and_excluded_rows_verbatim(tmp_path):
    """Appending new evidence must not migrate or drop existing legacy rows."""
    path = tmp_path / "benchmarks.json"
    existing = [
        {
            "graph": "g",
            "algorithm": "Original",
            "benchmark": "pr",
            "time_seconds": 1.5,
            "success": True,
        },
        {
            "graph": "g",
            "algorithm": "MAP",
            "benchmark": "pr",
            "time_seconds": 2.0,
            "success": True,
        },
    ]
    path.write_text(json.dumps(existing))

    store = BenchmarkStore(path)
    store.append([_result(run_id="new-1", time_seconds=0.5)])

    persisted = json.loads(path.read_text())
    assert persisted[:2] == existing
    assert persisted[2]["run_id"] == "new-1"
    assert persisted[2]["schema"] == BENCHMARK_OBSERVATION_SCHEMA


def test_appending_v1_row_does_not_restamp_it_as_v2(tmp_path):
    path = tmp_path / "benchmarks.json"
    store = BenchmarkStore(path)
    store.append([{
        "schema": "benchmark-observation/v1",
        "run_id": "v1-row",
        "graph": "g",
        "algorithm": "ORIGINAL",
        "algorithm_id": 0,
        "benchmark": "pr",
        "time_seconds": 1.0,
        "success": True,
    }])

    row = json.loads(path.read_text())[0]
    assert row["schema"] == "benchmark-observation/v1"
    assert row["algorithm_spec"] == "derived-v1:ORIGINAL"


def test_append_fails_closed_on_malformed_database(tmp_path):
    path = tmp_path / "benchmarks.json"
    malformed = b'{"not": "a complete JSON array"'
    path.write_bytes(malformed)

    store = BenchmarkStore(path)
    with pytest.raises(json.JSONDecodeError):
        store.append([_result(run_id="new-1")])
    assert path.read_bytes() == malformed


def test_legacy_migration_is_idempotent(monkeypatch, tmp_path):
    import scripts.lib.core.datastore as datastore

    target = tmp_path / "data" / "benchmarks.json"
    legacy_file = tmp_path / "benchmark_old.json"
    legacy_file.write_text(json.dumps([{
        "graph": "g",
        "algorithm": "ORIGINAL",
        "algorithm_id": 0,
        "benchmark": "pr",
        "time_seconds": 1.0,
        "success": True,
    }]))
    monkeypatch.setattr(datastore, "BENCHMARKS_FILE", target)

    datastore.migrate_legacy_files(tmp_path)
    first = json.loads(target.read_text())
    datastore.migrate_legacy_files(tmp_path)
    second = json.loads(target.read_text())

    assert first == second
    assert len(second) == 1
    assert second[0]["schema"] == "benchmark-observation/legacy"
    assert second[0]["run_id"].startswith("migration-")


def test_single_cpp_style_record_reads_compatibly(tmp_path):
    """A single C++-written record (already carrying condition fields) loads."""
    path = tmp_path / "benchmarks.json"
    cpp_row = [{
        "schema": "benchmark-observation/v1",
        "graph": "tiny", "algorithm": "Original", "algorithm_id": 0,
        "benchmark": "pr", "time_seconds": 0.25, "reorder_time": 0.0,
        "trials": 1, "nodes": 4, "edges": 6, "success": True, "error": "",
        "run_id": "cpp-123-456-0", "labeling": "natural",
        "measurement_mode": "self-record", "threads": 8,
        "mapping_identity_id": "direct", "process_id": 123, "attempt": 1,
        "trial_details": [], "extra": {},
    }]
    path.write_text(json.dumps(cpp_row))

    store = BenchmarkStore(path)
    assert len(store.observations()) == 1
    records = store.query(graph="tiny")
    assert len(records) == 1
    assert records[0]["benchmark"] == "pr"
    assert records[0]["algorithm_spec"] == "derived-v1:ORIGINAL"
    assert records[0]["time_seconds"] == 0.25
    assert store.perf_matrix()["tiny"]["ORIGINAL"]["pr"] == 0.25


def test_mapping_artifact_identity_changes_with_content(tmp_path):
    from scripts.lib.pipeline.benchmark import mapping_artifact_identity

    mapping = tmp_path / "GORDER.lo"
    mapping.write_bytes(b"first mapping")
    first = mapping_artifact_identity(mapping)
    assert first == mapping_artifact_identity(mapping)
    assert first.startswith("map:GORDER.lo:sha256:")

    mapping.write_bytes(b"second mapping with different size")
    second = mapping_artifact_identity(mapping)
    assert second != first


# =============================================================================
# Generic harness disables inherited C++ self-recording
# =============================================================================

def test_generic_run_benchmark_disables_cpp_self_recording(monkeypatch, tmp_path):
    """run_benchmark launches C++ with GRAPHBREW_DB_DIR='' by default even when
    an ambient GRAPHBREW_DB_DIR is exported; self_record=True opts back in."""
    import scripts.lib.pipeline.benchmark as bench

    captured = {}

    class _FakeCompleted:
        returncode = 0
        stdout = "Average Time:      0.5000\n"
        stderr = ""

    def _fake_run_command(cmd, timeout=None, check=False, env=None):
        captured["env"] = dict(env or {})
        return _FakeCompleted()

    monkeypatch.setattr(bench, "run_command", _fake_run_command)

    # Ambient writer present in the parent environment.
    ambient = str(tmp_path / "ambient") + "/"
    monkeypatch.setenv("GRAPHBREW_DB_DIR", ambient)

    # A file only needs to *exist* for the binary-presence check to pass.
    dummy_bin = tmp_path / "pr"
    dummy_bin.write_text("#!/bin/sh\n")

    # Default generic run: inherited self-recording is disabled.
    bench.run_benchmark(
        benchmark="pr", graph_path=str(TINY_GRAPH), algorithm="0",
        trials=1, bin_dir=str(tmp_path),
    )
    assert captured["env"].get("GRAPHBREW_DB_DIR") == ""
    assert captured["env"].get("GRAPHBREW_TOPOLOGY_ANALYSIS") == "0"

    # Explicit opt-in inherits the ambient environment.
    bench.run_benchmark(
        benchmark="pr", graph_path=str(TINY_GRAPH), algorithm="0",
        trials=1, bin_dir=str(tmp_path), self_record=True,
    )
    assert captured["env"].get("GRAPHBREW_DB_DIR") == ambient


def test_generic_run_benchmark_populates_condition(monkeypatch, tmp_path):
    """Every generic result carries the caller's observation condition."""
    import scripts.lib.pipeline.benchmark as bench

    class _FakeCompleted:
        returncode = 0
        stdout = (
            "Representation Build Time: 1.0000\n"
            "Reorder Core Time: 0.2000\n"
            "Reorder Time: 0.2000\n"
            "Mapping Fingerprint: abcdef0123456789\n"
            "Reorder Validation Time: 0.0100\n"
            "Reorder Apply Time: 0.0300\n"
            "Reorder End-to-End Time: 0.2400\n"
            "Resolved Reorder Spec: 0\n"
            "Total Preprocessing Time: 1.3000\n"
            "Average Time: 0.5000\n"
        )
        stderr = ""

    monkeypatch.setattr(
        bench, "run_command",
        lambda cmd, timeout=None, check=False, env=None: _FakeCompleted())

    dummy_bin = tmp_path / "pr"
    dummy_bin.write_text("#!/bin/sh\n")

    res = bench.run_benchmark(
        benchmark="pr", graph_path=str(TINY_GRAPH), algorithm="0", trials=1,
        bin_dir=str(tmp_path), labeling="shuffled", measurement_mode="process",
        threads=16, mapping_identity_id="map:GORDER.lo", attempt=2,
    )
    assert res.labeling == "shuffled"
    assert res.measurement_mode == "process"
    assert res.threads == 16
    assert res.mapping_identity_id == "map:GORDER.lo"
    assert res.requested_algorithm_spec.startswith("0|graph:")
    assert res.algorithm_spec == "0"
    assert res.attempt == 2
    assert res.run_id  # non-empty unique id
    assert res.representation_build_time == pytest.approx(1.0)
    assert res.reorder_core_time == pytest.approx(0.2)
    assert res.reorder_validation_time == pytest.approx(0.01)
    assert res.reorder_apply_time == pytest.approx(0.03)
    assert res.reorder_time == pytest.approx(0.24)
    assert res.total_preprocessing_time == pytest.approx(1.3)
    assert res.mapping_fingerprint == "abcdef0123456789"


def test_generic_run_benchmark_classifies_parse_contract_errors(
    monkeypatch,
    tmp_path,
):
    import scripts.lib.pipeline.benchmark as bench

    class _FakeCompleted:
        returncode = 0
        stdout = (
            "Reorder Core Time: 0.2000\n"
            "Reorder Time: 0.3000\n"
            "Average Time: 0.5000\n"
        )
        stderr = ""

    monkeypatch.setattr(
        bench,
        "run_command",
        lambda cmd, timeout=None, check=False, env=None: _FakeCompleted(),
    )
    (tmp_path / "pr").write_text("#!/bin/sh\n")

    result = bench.run_benchmark(
        benchmark="pr",
        graph_path=str(TINY_GRAPH),
        algorithm="0",
        trials=1,
        bin_dir=str(tmp_path),
    )

    assert result.success is False
    assert result.error_kind == "parse-contract"


def test_map_request_spec_uses_content_identity_not_path(
    monkeypatch,
    tmp_path,
):
    import scripts.lib.pipeline.benchmark as bench

    class _FakeCompleted:
        returncode = 0
        stdout = (
            "Resolved Reorder Spec: "
            "13:fingerprint=abcdef0123456789\n"
            "Average Time: 0.5000\n"
        )
        stderr = ""

    monkeypatch.setattr(
        bench,
        "run_command",
        lambda cmd, timeout=None, check=False, env=None: _FakeCompleted(),
    )
    (tmp_path / "pr").write_text("#!/bin/sh\n")
    result = bench.run_benchmark(
        benchmark="pr",
        graph_path=str(TINY_GRAPH),
        algorithm="13:/absolute/private/path/GORDER.lo",
        trials=1,
        bin_dir=str(tmp_path),
        mapping_identity_id="map:GORDER.lo:sha256:deadbeef",
    )

    assert result.requested_algorithm_spec.startswith(
        "13:map:GORDER.lo:sha256:deadbeef|graph:")
    assert "/absolute/private/path" not in result.requested_algorithm_spec


def test_adaptive_request_spec_tracks_model_content(
    monkeypatch,
    tmp_path,
):
    from scripts.lib.pipeline.benchmark import requested_execution_spec

    model = tmp_path / "weights.json"
    model.write_text('{"version": 1}')
    monkeypatch.setenv("PERCEPTRON_WEIGHTS_FILE", str(model))
    first = requested_execution_spec(
        algorithm="14",
        graph_path=TINY_GRAPH,
    )
    model.write_text('{"version": 2}')
    second = requested_execution_spec(
        algorithm="14",
        graph_path=TINY_GRAPH,
    )

    assert first != second


# =============================================================================
# Explicit C++ self-recording accumulates raw observations
# =============================================================================

def _require_pr() -> Path:
    binary = BIN_DIR / "pr"
    if not binary.exists():
        pytest.skip("Binary pr not built; run 'make bench/bin/pr' first")
    return binary


def _run_pr_self_record(pr: Path, db_dir: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.pop("GRAPHBREW_DB_DIR", None)
    cmd = [str(pr), "-f", str(TINY_GRAPH), "-s", "-n", "1",
           "-D", str(db_dir) + "/"]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                          env=env)


def test_cpp_self_recording_twice_accumulates_two_observations(tmp_path):
    """Two explicit self-recording runs append two raw rows (no replace)."""
    pr = _require_pr()
    db_dir = tmp_path / "db"
    db_dir.mkdir()

    r1 = _run_pr_self_record(pr, db_dir)
    assert r1.returncode == 0, r1.stderr
    r2 = _run_pr_self_record(pr, db_dir)
    assert r2.returncode == 0, r2.stderr

    data = json.loads((db_dir / "benchmarks.json").read_text())
    assert len(data) == 2, f"expected 2 raw observations, got {len(data)}"
    # Distinct run_ids, both successful, same (graph, algorithm, benchmark).
    assert len({row["run_id"] for row in data}) == 2
    for row in data:
        assert row["schema"] == BENCHMARK_OBSERVATION_SCHEMA
        assert row["graph"] == "tiny"
        assert row["benchmark"] == "pr"
        assert row["success"] is True
        assert row["mapping_identity_id"] == "direct"
        assert row["measurement_mode"] == "self-record"

    # The Python store medians the two raw observations (never min-collapses).
    store = BenchmarkStore(db_dir / "benchmarks.json")
    assert len(store.observations()) == 2
    import statistics
    expected = statistics.median([row["time_seconds"] for row in data])
    assert store.perf_matrix()["tiny"]["ORIGINAL"]["pr"] == pytest.approx(expected)


def test_cpp_self_recording_fails_closed_on_malformed_database(tmp_path):
    pr = _require_pr()
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    benchmark_file = db_dir / "benchmarks.json"
    malformed = b'{"truncated":'
    benchmark_file.write_bytes(malformed)

    result = _run_pr_self_record(pr, db_dir)

    assert result.returncode != 0
    assert benchmark_file.read_bytes() == malformed
    assert "Refusing to replace malformed" in result.stderr
