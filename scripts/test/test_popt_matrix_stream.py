#!/usr/bin/env python3
"""cache_sim must charge P-OPT's matrix stream once per sweep, not once per run.

The first version of the simulated column stream only charged forward epoch
progress. PageRank sweeps epochs 0..N-1 once per iteration, so every sweep after
the first was silently free: the stream cost was identical at -i 1, -i 2 and
-i 4. That reproduced the same undercharge as the flat analytic count, which is
also a single sweep, and it undercharged P-OPT by the iteration count.

The residency model replaces it: an epoch whose column is still one of the two
resident columns costs nothing, anything else streams a fresh column. These
tests pin that behaviour against the real binary.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PR = ROOT / "bench/bin_sim/pr"

pytestmark = pytest.mark.skipif(
    not PR.exists(), reason="cache_sim pr binary not built")


def run_pr(iterations: int, extra_env: dict | None = None) -> dict:
    """Run PageRank on a small synthetic graph and return the stats JSON."""
    env = dict(os.environ)
    env.update({
        "OMP_NUM_THREADS": "1",
        "CACHE_ULTRAFAST": "0",
        "CACHE_POLICY": "POPT",
        # Small enough to stay fast, small enough that the property array does
        # not fit the LLC, so the cell is not degenerate.
        "CACHE_L1_SIZE": "1024",
        "CACHE_L2_SIZE": "2048",
        "CACHE_L3_SIZE": "8192",
        "CACHE_L3_WAYS": "16",
        "POPT_MATRIX_STREAM_SIM": "1",
    })
    env.update(extra_env or {})
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "stats.json"
        env["CACHE_OUTPUT_JSON"] = str(out)
        subprocess.run(
            [str(PR), "-g", "12", "-k", "8", "-o", "5", "-n", "1",
             "-i", str(iterations)],
            env=env, capture_output=True, text=True, check=True, timeout=900)
        return json.loads(out.read_text())


def test_stream_is_charged_once_per_sweep():
    """Columns must scale with iterations, not be fixed at one sweep."""
    one = run_pr(1)
    two = run_pr(2)
    c1 = one["popt_matrix_stream_columns_simulated"]
    c2 = two["popt_matrix_stream_columns_simulated"]
    assert c1 > 0, "matrix stream never fired"
    # Exactly one column per epoch per sweep.
    assert c2 == 2 * c1, (
        f"expected two sweeps to stream twice as many columns, got {c1} then {c2}; "
        "a fixed count means later sweeps are silently free")


def test_stream_lines_track_columns():
    stats = run_pr(1)
    columns = stats["popt_matrix_stream_columns_simulated"]
    lines = stats["popt_matrix_stream_lines_simulated"]
    assert columns > 0 and lines > 0
    assert lines % columns == 0, "every column must stream the same line count"


def test_stream_is_off_by_default():
    """The stream must never appear unless explicitly requested."""
    stats = run_pr(1, {"POPT_MATRIX_STREAM_SIM": "0"})
    assert stats["popt_matrix_stream_columns_simulated"] == 0
    assert stats["popt_matrix_stream_lines_simulated"] == 0


def test_stream_adds_traffic_without_a_prefetcher():
    """A cold sequential stream must cost real memory traffic."""
    off = run_pr(1, {"POPT_MATRIX_STREAM_SIM": "0"})
    on = run_pr(1)
    added = on["total_memory_traffic"] - off["total_memory_traffic"]
    lines = on["popt_matrix_stream_lines_simulated"]
    # Nearly every line of a cold stream misses; allow slack for the few served
    # by the private caches.
    assert added >= 0.9 * lines, (
        f"stream of {lines} lines added only {added} traffic")
