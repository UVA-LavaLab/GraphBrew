import subprocess
from pathlib import Path

from scripts.experiments.ecg.policy_specs import parse_policy_spec


ROOT = Path(__file__).resolve().parents[2]


def test_hawkeye_proxy_policy_label_is_explicit():
    spec = parse_policy_spec("HAWKEYE:PROXY")
    assert spec.policy == "HAWKEYE"
    assert spec.label == "HAWKEYE_PROXY"


def test_hawkeye_policy_clean_room_core(tmp_path: Path):
    source = tmp_path / "hawkeye_policy_test.cc"
    binary = tmp_path / "hawkeye_policy_test"
    source.write_text(
        r'''
#include <cassert>
#include <cstdint>

#include "hawkeye_policy.h"

int main()
{
    using namespace hawkeye_policy;

    Predictor predictor;
    const uint64_t pc = 0x1234;
    assert(predictor.friendly(pc));
    for (int i = 0; i < 4; ++i) predictor.decrease(pc);
    assert(!predictor.friendly(pc));
    for (int i = 0; i < 20; ++i) predictor.increase(pc);
    assert(predictor.value(pc) == kPredictorMax);

    Optgen optgen(2);
    optgen.advance(0);
    assert(optgen.addInterval(0, 3, 3));
    assert(optgen.occupancyAt(0) == 1);
    assert(optgen.addInterval(0, 3, 3));
    assert(!optgen.addInterval(0, 3, 3));
    assert(!optgen.addInterval(0, 0, kOptgenQuanta));

    State state(8192, 16);
    std::size_t sampled = 0;
    for (std::size_t set = 0; set < 8192; ++set)
        sampled += state.sampledSet(set) ? 1 : 0;
    assert(sampled == 64);
    assert(state.access(0, 0x100, pc));
    state.access(0, 0x100, pc);

    State lru_state(8192, 16);
    uint64_t blocks[9] = {};
    for (uint64_t i = 0; i < 9; ++i)
        blocks[i] = i * kSamplerSets * 64;
    for (uint64_t i = 0; i < 8; ++i)
        lru_state.access(0, blocks[i], 0x2000 + i);
    lru_state.access(0, blocks[0], 0x2000);
    lru_state.access(0, blocks[8], 0x2008);
    assert(lru_state.samplerContains(blocks[0]));
    assert(!lru_state.samplerContains(blocks[1]));
    for (int i = 2; i < 9; ++i)
        assert(lru_state.samplerContains(blocks[i]));

    uint8_t rrpv[4] = {0, 4, 7, 6};
    assert(selectVictim(rrpv, 4) == 2);
    uint8_t friendly[4] = {0, 1, 2, 7};
    ageFriendlyFill(friendly, 4);
    assert(friendly[0] == 1 && friendly[1] == 2);
    assert(insertionRrpv(true) == 0);
    assert(insertionRrpv(false) == 7);
    return 0;
}
'''
    )
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(ROOT / "bench/include"),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        cwd=ROOT,
    )
    subprocess.run([str(binary)], check=True, cwd=ROOT)


def test_cache_sim_hawkeye_adapter_and_site_proxy(tmp_path: Path):
    source = tmp_path / "cache_sim_hawkeye_test.cc"
    binary = tmp_path / "cache_sim_hawkeye_test"
    source.write_text(
        r'''
#include <cassert>
#include <cstdint>

#include "cache_sim/cache_sim.h"
#include "cache_sim/graph_sim.h"

int main()
{
    using namespace cache_sim;
    assert(StringToPolicy("HAWKEYE") == EvictionPolicy::HAWKEYE);
    assert(PolicyToString(EvictionPolicy::HAWKEYE) == "HAWKEYE");

    CacheHierarchy cache(
        256, 2, 512, 2, 1024, 4, 64,
        EvictionPolicy::LRU, EvictionPolicy::LRU,
        EvictionPolicy::HAWKEYE);
    uint64_t values[64] = {};
    SIM_CACHE_READ(cache, values, 0);
    SIM_CACHE_READ(cache, values, 0);
    SIM_CACHE_READ(cache, values, 16);
    assert(cache.getTotalAccesses() == 3);

    bool rejected_private = false;
    try {
        CacheLevel invalid(
            "L1", 256, 64, 2, EvictionPolicy::HAWKEYE);
    } catch (const std::invalid_argument&) {
        rejected_private = true;
    }
    assert(rejected_private);
    return 0;
}
'''
    )
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-O0",
            "-Wall",
            "-Wextra",
            "-fopenmp",
            "-I",
            str(ROOT / "bench/include"),
            "-I",
            str(ROOT / "bench/include/external/gapbs"),
            "-I",
            str(ROOT / "bench/include/graphbrew"),
            "-I",
            str(ROOT / "bench/include/external"),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        cwd=ROOT,
    )
    subprocess.run([str(binary)], check=True, cwd=ROOT)
