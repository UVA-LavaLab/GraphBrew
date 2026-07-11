from pathlib import Path
import argparse
import importlib.util
import sys


ROOT = Path(__file__).resolve().parents[2]


def read(path: str) -> str:
    return (ROOT / path).read_text()


def test_streamshield_is_explicit_and_default_off():
    pr = read("bench/src_sim/pr.cc")
    assert 'GraphSimEnvIntClamped("ECG_STREAM_BYPASS", 0, 0, 1)' in pr
    assert "SIM_CACHE_READ_EDGE_BYPASS" in pr
    assert "SIM_CACHE_READ_STREAM_BYPASS" in pr


def test_streamshield_bypasses_llc_but_preserves_private_fills():
    cache = read("bench/include/cache_sim/cache_sim.h")
    block = cache.split("void accessStream(", 1)[1].split(
        "// StreamShield prefetch", 1
    )[0]
    assert "if (l3_->access" not in block
    assert "l2_->insert" in block
    assert "l1_->insert" in block


def test_streamshield_prefetch_avoids_false_memory_fills():
    cache = read("bench/include/cache_sim/cache_sim.h")
    block = cache.split("void prefetchStream(", 1)[1].split(
        "// Prefetch:", 1
    )[0]
    assert "l3_->contains" in block
    assert "prefetch_fills_++" in block


def test_gem5_streamshield_suppresses_only_l3_allocation():
    patch = read(
        "bench/include/gem5_sim/overlays/mem/cache/"
        "base_stream_bypass.patch"
    )
    context = read(
        "bench/include/gem5_sim/overlays/mem/cache/replacement_policies/"
        "graph_cache_context_gem5.hh"
    )
    assert 'find("l3cache")' in patch
    assert "getVaddr()" in patch
    assert "allocOnFill(pkt->cmd) && !stream_bypass" in patch
    assert "allow_alloc_on_fill" in patch
    assert "isEcgStreamBypassAddress" in context
    assert "stream_bypass_base" in context


def test_sniper_streamshield_skips_nuca_lookup_and_fill():
    setup = read("scripts/setup_sniper.py")
    context = read(
        "bench/include/sniper_sim/overlays/common/core/memory_subsystem/cache/"
        "graph_cache_context_sniper.cc"
    )
    assert "m_stream_bypass_reads" in setup
    assert "m_stream_bypass_writes" in setup
    assert "SubsecondTime::Zero(), HitWhere::MISS" in setup
    assert "eviction = false" in setup
    assert "isEcgStreamBypassAddress" in context


def test_streamshield_is_policy_isolated_and_verified():
    runner = read("scripts/experiments/ecg/roi_matrix.py")
    verifier = read("scripts/experiments/ecg/verify/equiv_kernels.py")
    assert 'env.pop("ECG_STREAM_BYPASS", None)' in runner
    assert '"ecg_stream_bypass"' in runner
    assert "--stream-bypass" in verifier
    assert "stream-bypass-reads" in verifier
    assert "stream-bypass-writes" in verifier
    assert "cache_sim_ecg_epoch_region_index" in runner
    assert "Sniper StreamShield requires --sniper-workload sg_kernel" in runner
    assert 'env.get("ECG_STREAM_BYPASS") == "1"' in runner
    assert "--stream-bypass requires --schedule-k 2" in verifier


def test_streamshield_is_pr_only_in_detailed_kernels():
    gem5_bfs = read("bench/src_gem5/bfs.cc")
    sniper = read("bench/src_sniper/sg_kernel.cc")
    gem5_export = gem5_bfs.split("gem5_export_context(", 1)[1].split(");", 1)[0]
    assert "pair_flat.data()" not in gem5_export
    bfs_export = sniper.split(
        "sniper_export_context(\n        regions, 1, graph", 1
    )[1].split(");", 1)[0]
    assert "bfs_pair_flat.data()" not in bfs_export


def test_schedule_bits_are_charged_in_record_width():
    pr = read("bench/src_sim/pr.cc")
    assert 'GraphSimEnvIntClamped("ECG_EDGE_MASK_SCHED", 0, 0, 4)' in pr
    assert "epoch_bits * std::max(1, sched_k)" in pr
    assert "static_cast<uint64_t>(edge_pos) * 16" in pr


def test_adaptive_variant_selects_by_kernel(monkeypatch):
    path = ROOT / "scripts/experiments/ecg/roi_matrix.py"
    spec = importlib.util.spec_from_file_location("roi_matrix_adaptive", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    monkeypatch.setenv("ECG_VARIANT", "adaptive")
    expected = {
        "pr": "epoch_first",
        "bfs": "degree_first",
        "sssp": "degree_first",
        "bc": "rrip_first",
        "cc": "rrip_first",
    }
    for benchmark, variant in expected.items():
        args = argparse.Namespace(benchmark=benchmark)
        assert module.effective_ecg_variant(args) == variant
