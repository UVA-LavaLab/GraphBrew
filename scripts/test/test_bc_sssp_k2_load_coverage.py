from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read(path: str) -> str:
    return (ROOT / path).read_text()


def test_sssp_hoists_source_distance_before_k2_delivery():
    cache_sim = read("bench/src_sim/sssp.cc")
    gem5 = read("bench/src_gem5/sssp.cc")
    sniper = read("bench/src_sniper/sg_kernel.cc")

    assert "RelaxEdges_Sim(g, u, delta, source_dist" in cache_sim
    assert "WeightT new_dist = source_dist + wn.w;" in cache_sim
    assert "const WeightT source_dist = dist[u];\n    GEM5_SET_VERTEX(u);" in gem5
    assert "WeightT new_dist = source_dist + wn.w;" in gem5
    assert "const WeightT source_dist = dist[node];\n        SNIPER_SET_VERTEX(node);" in sniper
    assert "const WeightT candidate = source_dist + edge.w;" in sniper
    sidecar = sniper.split(
        "inline uint32_t consume_fused_k2_sidecar", 1)[1].split(
            "\n}", 1)[0]
    assert 'asm volatile("" : : "r"(sidecar));' in sidecar
    assert '"memory"' not in sidecar
    assert "packCompactWeightedEpochPairRecord" in sniper
    assert "[ECG_FUSED_K2_WEIGHTED64]" in sniper


def test_bc_masks_depth_and_path_counts():
    cache_sim = read("bench/src_sim/bc.cc")
    gem5 = read("bench/src_gem5/bc.cc")
    decoder = read(
        "bench/include/gem5_sim/overlays/arch/riscv/isa/"
        "decoder_ecg_extract.isa")
    sniper = read("bench/src_sniper/sg_kernel.cc")
    sniper_context = read(
        "bench/include/sniper_sim/overlays/common/core/memory_subsystem/"
        "cache/graph_cache_context_sniper.cc")

    assert "SIM_CACHE_READ_MASKED(cache, depths.data(), v" in cache_sim
    assert "SIM_CACHE_READ_MASKED(cache, path_counts.data(), v" in cache_sim
    assert "gem5_ecg_load_k2_u64(path_counts.data(), record)" in gem5
    assert "0x04: ecg_load_k2_u64" in decoder
    assert '"path_counts"' in sniper_context
    assert "k2_line8_offsets" in sniper_context
    assert "const int64_t source_paths = path_counts[u];\n                SNIPER_SET_VERTEX(u);" in sniper
    assert "SNIPER_CLEAR_VERTEX();" in sniper
