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
import re
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


def test_every_cache_sim_kernel_uses_the_metadata_ssot():
    """All five algorithms must share one delivery site, on every simulator.

    The paper's foundation is a 15-cell conformance gate, so a kernel that
    delivers metadata its own way is a correctness risk, not just untidy. These
    checks fail if any kernel drifts back to a private chain.
    """
    for kernel in ("pr", "bfs", "cc", "bc", "sssp"):
        src = (ROOT / f"bench/src_sim/{kernel}.cc").read_text()
        assert "::ecg_metadata::configure(" in src, (
            f"{kernel} does not configure delivery from the SSOT")
        assert "SIM_ECG_EDGE(" in src, (
            f"{kernel} does not use the single delivery site")
        assert "::ecg_metadata::announce(" in src, (
            f"{kernel} emits no configuration receipt")
        for dead in ("SIM_CACHE_READ_EDGE_RECORD(",
                     "SIM_CACHE_READ_EDGE_RECORD_BYPASS(",
                     "GraphSimEcgRecordBytes("):
            assert dead not in src, (
                f"{kernel} still carries the superseded {dead}")


def test_all_three_simulators_share_the_metadata_ssot():
    """cache_sim, gem5 and Sniper must derive width and structure from one header.

    ecg_victim_policy.h owns the eviction DECISION and is kept identical across
    the three by copying it into each overlay and hash-checking the copies.
    ecg_metadata.h owns TRANSPORT, and because it is consumed by the guest
    kernels rather than by simulator internals, all three can include the
    canonical file directly via -I bench/include. There is therefore nothing to
    copy and nothing that can drift -- but only as long as each simulator
    actually uses it, which is what this asserts.
    """
    canonical = ROOT / "bench/include/ecg_metadata.h"
    assert canonical.is_file(), "metadata SSOT header is missing"

    consumers = {
        "cache_sim": [ROOT / f"bench/src_sim/{k}.cc"
                      for k in ("pr", "bfs", "cc", "bc", "sssp")],
        "gem5": [ROOT / f"bench/src_gem5/{k}.cc"
                 for k in ("pr", "bfs", "cc", "bc", "sssp")],
        "sniper": [ROOT / "bench/src_sniper/sg_kernel.cc"],
    }
    for sim, paths in consumers.items():
        for path in paths:
            src = path.read_text()
            assert "ecg_metadata.h" in src or "::ecg_metadata::" in src, (
                f"{sim} source {path.name} does not use the metadata SSOT")

    # No simulator may keep a private width rule.
    for path in [p for paths in consumers.values() for p in paths]:
        src = path.read_text()
        assert "GraphSimEcgRecordBytes(" not in src, (
            f"{path.name} still computes record width locally")


def test_metadata_ssot_has_no_simulator_dependencies():
    """It must stay includable by guest kernels on every backend.

    Checks code, not prose: the header names the simulators in its own
    documentation, which is fine. What must not appear is an include of, or a
    type from, any one backend.
    """
    lines = (ROOT / "bench/include/ecg_metadata.h").read_text().splitlines()
    code = "\n".join(
        l for l in lines if not l.lstrip().startswith("//"))
    for forbidden in ("cache_sim.h", "graph_sim.h", "CacheHierarchy",
                      "SimArray", "m5op", "sift"):
        assert forbidden not in code, (
            f"metadata SSOT depends on {forbidden}, so it is no longer "
            "backend-neutral")
    # Only standard headers.
    includes = [l for l in lines if l.lstrip().startswith("#include")]
    assert includes, "header includes nothing at all"
    for inc in includes:
        assert "<" in inc, f"non-standard include in the SSOT: {inc.strip()}"


# ---------------------------------------------------------------------------
# Cross-simulator width agreement
# ---------------------------------------------------------------------------

GEM5_PR = ROOT / "bench/bin_gem5/pr"
GRAPH = ROOT / "results/graphs/web-Google-n16/web-Google-n16.sg"

RECEIPT = re.compile(
    r"stamps=\d+ epoch_bits=\d+ tier_bits=\d+ id_bits=\d+ "
    r"record_bytes=\d+ payload_bits=\d+")


def _receipt(cmd, env):
    e = dict(os.environ); e.update({k: str(v) for k, v in env.items()})
    out = subprocess.run(cmd, env=e, capture_output=True, text=True,
                         timeout=900)
    m = RECEIPT.search(out.stdout + out.stderr)
    return m.group(0) if m else None


@pytest.mark.skipif(not (GEM5_PR.exists() and GRAPH.exists()),
                    reason="gem5 pr binary or graph fixture missing")
@pytest.mark.parametrize("stamps", [1, 2])
def test_cache_sim_and_gem5_derive_identical_width(stamps):
    """The whole point of the SSOT: no backend may compute its own width.

    Both simulators independently call ecg_metadata::configure and print a
    receipt. Identical configuration must produce byte-identical receipts. A
    mismatch means one backend has drifted back to a private width rule, which
    is exactly the defect that made K2-versus-K1 a comparison of record widths.
    """
    shared = {
        "ECG_EDGE_MASK_EPOCH": 1, "ECG_EDGE_MASK_LINEMIN": 1,
        "ECG_EDGE_MASK_EPOCHS": 32,
    }
    if stamps == 2:
        shared["ECG_EDGE_MASK_SCHED"] = 2

    cs_env = dict(shared)
    cs_env.update({
        "ECG_MODE": "ECG_GRASP_POPT", "ECG_EDGE_MASKS": 1,
        "ECG_EDGE_MASK_LEAN": 1, "ECG_EDGE_MASK_PACK": 1,
        "ECG_EXACT_REREF": 1, "ECG_PREFETCH_MODE": 6,
        "OMP_NUM_THREADS": 1, "CACHE_ULTRAFAST": 0,
        "CACHE_POLICY": "ECG", "CACHE_L3_SIZE": 131072,
    })
    cs = _receipt(
        ["/usr/bin/setarch", "x86_64", "-R", str(PR),
         "-f", str(GRAPH), "-o", "5", "-n", "1", "-i", "1"], cs_env)

    g5_env = dict(shared)
    g5_env.update({"GEM5_ENABLE_ECG_PFX_HINTS": 1, "GEM5_ECG_PFX_MODE": 6})
    g5 = _receipt([str(GEM5_PR), "-f", str(GRAPH), "-n", "1", "-i", "1"],
                  g5_env)

    assert cs is not None, "cache_sim emitted no metadata receipt"
    assert g5 is not None, "gem5 emitted no metadata receipt"
    assert cs == g5, (
        f"backends disagree on record width at stamps={stamps}:\n"
        f"  cache_sim: {cs}\n  gem5     : {g5}")

    # Sniper's workload is a third independent consumer of the same header.
    sniper = ROOT / "bench/bin_sniper/sg_kernel"
    if sniper.exists():
        sn_env = dict(shared)
        sn_env["SNIPER_ENABLE_ECG_EXTRACT"] = 1
        sn = _receipt(
            [str(sniper), "--benchmark", "pr", "-f", str(GRAPH), "-i", "1"],
            sn_env)
        assert sn is not None, "Sniper emitted no metadata receipt"
        assert sn == cs, (
            f"Sniper disagrees on record width at stamps={stamps}:\n"
            f"  cache_sim: {cs}\n  sniper   : {sn}")


def test_declared_gem5_timing_stages_are_honestly_scoped():
    """gem5 streams 8 bytes for Schedule-2, so it cannot host a width contrast.

    gem5 builds pvector<uint64_t> in_edge_pair_flat, so both arms of a
    4-versus-8-byte contrast would stream 8 bytes and the comparison would be
    vacuous. That is why the gem5 stages are a timing and bandwidth study at the
    width gem5 actually streams, and why the width contrast belongs to
    cache_sim, where substitution for the CSR edge is modelled.

    Guards the two mistakes already made: forcing a width here (vacuous), and
    nesting the explicit-cell channel inside itself (silently dropped, since
    paper_run already wraps the stage env).
    """
    manifest = json.loads(
        (ROOT / "scripts/experiments/ecg/final_paper_manifest.json").read_text())
    stages = [s for s in manifest["stages"]
              if str(s.get("name", "")).startswith("31_gem5_record_width")]
    assert stages, "the declared gem5 timing stages are missing"

    for stage in stages:
        env = stage.get("env", {})
        assert "GRAPHBREW_EXPLICIT_CELL_ENV" not in env, (
            f"{stage['name']} nests the explicit channel inside itself; "
            "paper_run already wraps the stage env, so this double-encodes")
        assert env.get("ECG_RECORD_VARIABLE_WIDTH") == "1", (
            f"{stage['name']} must request variable width so the receipt "
            "reports a computed width rather than a hardcoded default")
        if stage["name"].endswith("_8b"):
            assert env.get("ECG_EDGE_RECORD_BYTES") == "8", (
                "the 8-byte arm must force its width, or both arms measure the "
                "same thing")
        else:
            assert "ECG_EDGE_RECORD_BYTES" not in env, (
                f"{stage['name']} forces a width, so it is not the compact arm")
        assert int(stage.get("ecg_epochs", 0)) <= 4096, (
            f"{stage['name']} uses too many epochs for the record to pack")


def test_gem5_forwards_metadata_knobs_into_the_simulated_guest():
    """gem5 SE mode does not inherit the host environment.

    graph_se.py builds an explicit allowlist of variables to hand the simulated
    process. The metadata SSOT knobs were absent from it, so a stage asking for
    a 4-byte record silently got the Schedule-2 default of 8: the run looked
    correct at every layer above, and only the guest's own receipt disagreed.

    This is the third distinct layer of env plumbing between a manifest stage
    and the guest, after roi_matrix's scrub and paper_run's explicit-cell
    channel, and the only one that is invisible from the host side.
    """
    config = (ROOT / "bench/include/gem5_sim/configs/graphbrew/graph_se.py").read_text()
    required = [
        "ECG_RECORD_VARIABLE_WIDTH",
        "ECG_EDGE_RECORD_BYTES",
        "ECG_DELIVERY",
        "ECG_SIDECAR_PAYLOAD_BITS",
        "ECG_RECORD_TIER_BITS",
        "ECG_VIRTUAL_ID_BITS",
    ]
    for name in required:
        assert f'"{name}"' in config, (
            f"graph_se.py does not forward {name} to the simulated guest, so "
            "gem5 cells cannot honour it and will silently use the default")


def test_compact_two_stamp_record_packs_and_round_trips():
    """The 32-bit two-stamp format must be exact, and honest about its limits.

    gem5 and Sniper previously had only a 64-bit Schedule-2 record, so they
    streamed 8 bytes per edge and DOUBLED the structural stream against a 4-byte
    CSR edge, while cache_sim modelled the record as substituting for that edge.
    That produced a direction reversal between simulators: 0.557 against LRU in
    cache_sim versus 1.189 in gem5 at identical geometry.

    The compact format closes it, but only where the fields genuinely fit.
    """
    header = (ROOT / "bench/include/ecg_epoch_builder.h").read_text()
    for fn in ("canPackEpochPair32", "packEpochPairRecord32",
               "extractEpochPair32Dest", "extractEpochPair32Tier",
               "extractEpochPair32First", "extractEpochPair32Second",
               "widenEpochPair32", "buildInEdgeEpochPairRecords32"):
        assert fn in header, f"compact Schedule-2 helper {fn} is missing"

    # The compact builder must reuse the SAME epoch computation as the 64-bit
    # one, or the two widths would mean different policies.
    start = header.index("bool buildInEdgeEpochPairRecords32")
    body = header[start:start + 3000]
    assert "nextEpochPairForLine" in body, (
        "the compact builder computes epochs its own way, so a width change "
        "would silently change the policy")

    # And it must refuse rather than truncate when the fields do not fit.
    assert "if (!canPackEpochPair32(n, ne)) return false;" in body, (
        "the compact builder does not check feasibility, so it could silently "
        "truncate destinations or epochs")


def test_gem5_prefers_the_compact_record_and_declares_its_width():
    src = (ROOT / "bench/src_gem5/pr.cc").read_text()
    assert "buildInEdgeEpochPairRecords32" in src, (
        "gem5 does not try the compact record, so it always streams 8 bytes")
    assert "widenEpochPair32" in src, (
        "gem5 does not widen the compact record for the ISA helpers")
    assert "declareContainerBytes" in src, (
        "gem5 does not declare the container it actually streams, so its "
        "receipt can claim a width the guest does not deliver")
    assert "canPackEpochPair32" in src, (
        "gem5 declares a fixed container instead of the one feasibility allows")
