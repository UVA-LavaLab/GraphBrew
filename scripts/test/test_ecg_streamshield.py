from pathlib import Path


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


def test_schedule_bits_are_charged_in_record_width():
    pr = read("bench/src_sim/pr.cc")
    assert 'GraphSimEnvIntClamped("ECG_EDGE_MASK_SCHED", 0, 0, 4)' in pr
    assert "epoch_bits * std::max(1, sched_k)" in pr
    assert "static_cast<uint64_t>(edge_pos) * 16" in pr
