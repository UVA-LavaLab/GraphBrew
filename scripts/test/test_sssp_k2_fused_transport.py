from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_weighted_sssp_fused_path_moves_validation_before_roi():
    builder = (ROOT / "bench/include/ecg_epoch_builder.h").read_text()
    sniper = (ROOT / "bench/src_sniper/sg_kernel.cc").read_text()
    gem5 = (ROOT / "bench/src_gem5/sssp.cc").read_text()

    assert "validateWeightedEpochPairRecords" in builder
    assert "consume_fused_k2_sidecar" in sniper
    assert "packWeightedEpochPairSidecar" in builder
    assert "pair_sidecars.size() * sizeof(uint32_t)" in sniper
    assert "pair_flat.size() * sizeof(uint64_t)" in sniper
    assert "static_cast<uint32_t>(edge.v)" in sniper
    assert "deliver_k2_record(record, fused_k2_model);" in sniper
    assert "} else if (\n                pair_ok ||" in sniper
    assert "auto relax_edges = [&](" in sniper
    assert sniper.count("relax_edges(") == 4
    assert 'std::getenv("ECG_K2_VALIDATE")' in sniper
    assert 'std::getenv("ECG_K2_VALIDATE")' in gem5
    assert "gem5_ecg_stream_weighted_load2_instruction" in gem5
    assert "gem5_ecg_weighted_load2_instruction" in gem5
    assert "combineWeightedEpochPairRecord" in builder
    assert "gem5_ecg_load_k2(dist.data(), record)" in gem5

    decoder = (
        ROOT
        / "bench/include/gem5_sim/overlays/arch/riscv/isa/"
        "decoder_ecg_extract.isa"
    ).read_text()
    assert decoder.count("uint64_t dest_dependency = Rs2;") == 1
    stream_block = decoder.split("0x3: ecg_stream_load2", 1)[1].split(
        "0x4: ecg_load2", 1)[0]
    weighted_stream_block = decoder.split(
        "0x5: ecg_stream_weighted_load2", 1)[1].split(
            "0x6: ecg_weighted_load2", 1)[0]
    assert "setDecodedEcgExtractHint2" not in stream_block
    assert "setDecodedEcgExtractHint2" not in weighted_stream_block

    sniper_roi = sniper.split("int run_sssp(", 1)[1].split(
        "SNIPER_ROI_BEGIN();", 1)[1].split("SNIPER_ROI_END();", 1)[0]
    assert "SSSP K2 pair index out of range" not in sniper_roi
    assert "SSSP K2 destination mismatch" not in sniper_roi

    gem5_relax = gem5.split("inline void RelaxEdges_Gem5", 1)[1].split(
        "int main(", 1)[0]
    assert "SSSP K2 pair index out of range" not in gem5_relax
    assert "SSSP K2 destination mismatch" not in gem5_relax

    runner = (ROOT / "scripts/experiments/ecg/roi_matrix.py").read_text()
    assert '4 if transport.schedule_k == 2 and args.benchmark == "sssp"' in runner
    assert '"edge_stream_bytes_per_edge"' in runner
    assert '"ecg_record_replaces_edge"' in runner
