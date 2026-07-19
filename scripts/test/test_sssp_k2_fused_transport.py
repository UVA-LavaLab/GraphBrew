from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_weighted_sssp_fused_path_moves_validation_before_roi():
    builder = (ROOT / "bench/include/ecg_epoch_builder.h").read_text()
    sniper = (ROOT / "bench/src_sniper/sg_kernel.cc").read_text()
    gem5 = (ROOT / "bench/src_gem5/sssp.cc").read_text()

    assert "validateWeightedEpochPairRecords" in builder
    assert "consume_fused_k2_record" in sniper
    assert "deliver_k2_record(record, fused_k2_model);" in sniper
    assert "} else if (\n                pair_ok ||" in sniper
    assert "auto relax_edges = [&](" in sniper
    assert sniper.count("relax_edges(") == 4
    assert 'std::getenv("ECG_K2_VALIDATE")' in sniper
    assert 'std::getenv("ECG_K2_VALIDATE")' in gem5

    sniper_roi = sniper.split("int run_sssp(", 1)[1].split(
        "SNIPER_ROI_BEGIN();", 1)[1].split("SNIPER_ROI_END();", 1)[0]
    assert "SSSP K2 pair index out of range" not in sniper_roi
    assert "SSSP K2 destination mismatch" not in sniper_roi

    gem5_relax = gem5.split("inline void RelaxEdges_Gem5", 1)[1].split(
        "int main(", 1)[0]
    assert "SSSP K2 pair index out of range" not in gem5_relax
    assert "SSSP K2 destination mismatch" not in gem5_relax
