from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read(relative: str) -> str:
    return (ROOT / relative).read_text()


def test_grasp_hot_insertion_matches_upstream_priority_rrip():
    cache_sim = read("bench/include/cache_sim/cache_sim.h")
    gem5_grasp = read("bench/include/gem5_sim/overlays/mem/cache/replacement_policies/grasp_rp.cc")
    gem5_ecg = read("bench/include/gem5_sim/overlays/mem/cache/replacement_policies/ecg_rp.cc")
    sniper_grasp = read("bench/include/sniper_sim/overlays/common/core/memory_subsystem/cache/cache_set_grasp.cc")
    sniper_ecg = read("bench/include/sniper_sim/overlays/common/core/memory_subsystem/cache/cache_set_ecg.cc")

    assert "constexpr uint8_t P_RRIP = 1" in cache_sim
    assert "case ReuseTier::HIGH:     return 1;" in gem5_grasp
    assert "constexpr uint8_t pRrip = 1;" in gem5_ecg
    assert "if (tier == 1) return 1;" in sniper_grasp
    assert "if (tier == 1) return 1;" in sniper_ecg


def test_popt_mixed_sets_use_phase_one_not_far_distance_boost():
    cache_sim = read("bench/include/cache_sim/cache_sim.h")
    gem5_popt = read("bench/include/gem5_sim/overlays/mem/cache/replacement_policies/popt_rp.cc")
    gem5_ecg = read("bench/include/gem5_sim/overlays/mem/cache/replacement_policies/ecg_rp.cc")
    sniper_popt = read("bench/include/sniper_sim/overlays/common/core/memory_subsystem/cache/cache_set_popt.cc")
    sniper_ecg = read("bench/include/sniper_sim/overlays/common/core/memory_subsystem/cache/cache_set_ecg.cc")

    assert "if (!is_graph_data) return i;" in cache_sim
    assert "if (!d->is_property_data) return c;" in gem5_popt
    assert "if (!data->is_property_data) return c;" in gem5_ecg
    assert "if (!m_property_lines[way])" in sniper_popt
    assert "if (!m_property_lines[way])" in sniper_ecg

    for text in (gem5_popt, gem5_ecg, sniper_popt, sniper_ecg):
        assert "dist > 64" not in text
        assert "distance > 64" not in text
        assert "far-rereference boost" not in text