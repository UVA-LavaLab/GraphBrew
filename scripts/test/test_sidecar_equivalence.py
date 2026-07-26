#!/usr/bin/env python3
"""S2 (narrow sidecar) must be semantically identical to S1 (packed record).

The paper rests on a 15-cell conformance gate: the eviction decision in
`ecg_victim_policy.h` is kernel-agnostic and byte-identical across cache_sim,
gem5 and Sniper. Introducing a second metadata delivery structure threatens
that gate unless the structure is provably transport-only.

It is. The gate verifies victim decisions given the epochs, not how the epochs
reached the policy. So if S2 delivers the same stamps as S1, every victim
decision is unchanged and conformance is preserved by construction.

These tests prove the antecedent the cheap way: charge nothing for the metadata
in either structure, which removes transport from both, and require the two to
produce byte-identical cache behaviour. Any divergence means S2 is delivering
different stamps and is therefore not a drop-in structure.
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

COMMON = {
    "OMP_NUM_THREADS": "1",
    "CACHE_ULTRAFAST": "0",
    "CACHE_POLICY": "ECG",
    "CACHE_L1_SIZE": "1024",
    "CACHE_L2_SIZE": "2048",
    "CACHE_L3_SIZE": "8192",
    "CACHE_L3_WAYS": "16",
    "ECG_MODE": "ECG_GRASP_POPT",
    "ECG_EDGE_MASKS": "1",
    "ECG_EDGE_MASK_SCHED": "2",
    "ECG_EDGE_MASK_EPOCH": "1",
    "ECG_EDGE_MASK_LEAN": "1",
    "ECG_EDGE_MASK_PACK": "1",
    "ECG_EDGE_MASK_LINEMIN": "1",
    "ECG_EXACT_REREF": "1",
    "ECG_PREFETCH_MODE": "6",
    "ECG_EDGE_MASK_EPOCHS": "32",
    "ECG_VARIANT": "epoch_first",
}


def run(**overrides) -> dict:
    env = dict(os.environ)
    env.update(COMMON)
    env.update({k: str(v) for k, v in overrides.items()})
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "stats.json"
        env["CACHE_OUTPUT_JSON"] = str(out)
        # ASLR must be off: cache_sim tracks real pointers, so address-space
        # randomisation perturbs cache set mapping by ~0.07% run to run, which
        # is larger than the differences these gates assert on.
        subprocess.run(
            ["/usr/bin/setarch", "x86_64", "-R",
             str(PR), "-g", "12", "-k", "8", "-o", "5", "-n", "1", "-i", "2"],
            env=env, capture_output=True, text=True, check=True, timeout=900)
        return json.loads(out.read_text())


def signature(stats: dict) -> dict:
    return {
        "offchip": stats["total_offchip_traffic"],
        "reads": stats["total_memory_traffic"],
        "writebacks": stats["llc_writebacks"],
        "l1_misses": stats["L1"]["misses"],
        "l2_misses": stats["L2"]["misses"],
        "l3_misses": stats["L3"]["misses"],
    }


def test_results_are_deterministic():
    """Everything below assumes exact equality, so prove runs are repeatable."""
    a = signature(run(ECG_EDGE_MASK_CHARGED=1))
    b = signature(run(ECG_EDGE_MASK_CHARGED=1))
    assert a == b, f"cache_sim is not deterministic under setarch -R:\n{a}\n{b}"


def test_sidecar_is_transport_only():
    """Uncharged S1 and S2 must be byte-identical at every cache level."""
    s1 = signature(run(ECG_EDGE_MASK_CHARGED=0))
    s2 = signature(run(ECG_EDGE_MASK_CHARGED=0, ECG_DELIVERY="sidecar"))
    assert s1 == s2, (
        "S2 changed cache behaviour with transport removed, so it is NOT "
        f"delivering the same stamps as S1:\nS1={s1}\nS2={s2}")


def test_sidecar_costs_traffic_when_charged():
    """The gate above must not pass by the sidecar simply doing nothing."""
    free = run(ECG_EDGE_MASK_CHARGED=0, ECG_DELIVERY="sidecar")["total_offchip_traffic"]
    charged = run(ECG_EDGE_MASK_CHARGED=1, ECG_DELIVERY="sidecar")["total_offchip_traffic"]
    assert charged > free, (
        "charging the sidecar did not add traffic, so it is not being "
        "simulated and the equivalence test above is vacuous")


def test_sidecar_payload_width_changes_cost_monotonically():
    """A wider payload must cost more; width must actually reach the model."""
    narrow = run(ECG_EDGE_MASK_CHARGED=1, ECG_DELIVERY="sidecar",
                 ECG_SIDECAR_PAYLOAD_BITS=6)["total_offchip_traffic"]
    wide = run(ECG_EDGE_MASK_CHARGED=1, ECG_DELIVERY="sidecar",
               ECG_SIDECAR_PAYLOAD_BITS=24)["total_offchip_traffic"]
    assert wide > narrow, (
        f"payload width did not affect cost: 6b={narrow} 24b={wide}")


def test_sidecar_width_is_independent_of_graph_size():
    """The whole point of S2: payload must not depend on vertex count.

    The packed record's width is id_bits + stamps*epoch_bits + tier_bits, so it
    grows with the graph. The sidecar carries no destination id, so two graphs
    of different size at the same payload setting must charge the same bits per
    edge.
    """
    header = ROOT / "bench/include/ecg_metadata.h"
    text = header.read_text()
    assert "payload_bits" in text
    start = text.index("const int forced_payload")
    body = text[start:text.index("const int needed", start)]
    assert "num_vertices" not in body, (
        "the sidecar payload consults the vertex count, so its width is not "
        "graph-size independent")
    assert "id_bits" not in body, (
        "the sidecar payload includes destination id bits it does not need; "
        "the CSR edge already carries the destination")
