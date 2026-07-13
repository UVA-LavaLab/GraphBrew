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


def test_streamshield_preserves_llc_hits_and_suppresses_miss_fill():
    cache = read("bench/include/cache_sim/cache_sim.h")
    block = cache.split("void accessStream(", 1)[1].split(
        "// StreamShield prefetch", 1
    )[0]
    assert "if (l3_->access" in block
    assert "l3_->insert" not in block
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
    flag_patch = read(
        "bench/include/gem5_sim/overlays/mem/cache/"
        "base_stream_bypass_request_flag.patch"
    )
    context = read(
        "bench/include/gem5_sim/overlays/mem/cache/replacement_policies/"
        "graph_cache_context_gem5.hh"
    )
    request_patch = read(
        "bench/include/gem5_sim/overlays/mem/request_stream_bypass.patch"
    )
    prefetch_patch = read(
        "bench/include/gem5_sim/overlays/mem/cache/"
        "prefetch_stream_bypass.patch"
    )
    decoder = read(
        "bench/include/gem5_sim/overlays/arch/riscv/isa/"
        "decoder_ecg_extract.isa"
    )
    harness = read("bench/include/gem5_sim/gem5_harness.h")
    assert 'find("l3cache")' in patch
    assert "getVaddr()" in patch
    assert "Request::ECG_STREAM_BYPASS" in flag_patch
    assert "GEM5_ECG_STREAM_REQUEST_BOUND" in flag_patch
    assert "allocOnFill(pkt->cmd) && !stream_bypass" in patch
    assert "allow_alloc_on_fill" in patch
    assert "isEcgStreamBypassAddress" in context
    assert "stream_bypass_base" in context
    assert "ECG_STREAM_BYPASS" in request_patch
    assert "isStreamBypass" in prefetch_patch
    assert "req->setFlags(Request::ECG_STREAM_BYPASS)" in prefetch_patch
    assert "ecg_stream_load2" in decoder
    assert "ecg_load2" in decoder
    assert "mem_flags=[ECG_STREAM_BYPASS]" in decoder
    assert ".insn i 0x0b, 0x3" in harness
    assert ".insn i 0x0b, 0x4" in harness


def test_sniper_streamshield_preserves_nuca_lookup_and_skips_miss_fill():
    setup = read("scripts/setup_sniper.py")
    context = read(
        "bench/include/sniper_sim/overlays/common/core/memory_subsystem/cache/"
        "graph_cache_context_sniper.cc"
    )
    assert "m_stream_bypass_reads" in setup
    assert "m_stream_bypass_writes" in setup
    assert "latency, HitWhere::MISS" in setup
    assert "if (stream_bypass) ++m_stream_bypass_reads;" in setup
    assert "eviction = false" in setup
    assert "isEcgStreamBypassAddress" in context
    assert "lookupFusedK2Pair" in context
    assert "k2_offsets_path" in context
    assert "k2_line_offsets" in context
    assert "std::lower_bound" in context
    assert "Sniper fused K2 sideband is missing or incomplete" in context
    harness = read("bench/include/sniper_sim/sniper_harness.h")
    assert "sniper_write_binary_atomic" in harness
    assert '"SNIPER_CACHE_LINE_SIZE"' in harness
    assert "property_alignment()" in harness
    assert "stream_bypass_base -= stream_bypass_base % line_size" in harness
    assert "const uint64_t aligned_upper = raw_upper + padding" in harness
    assert "std::remove(k2_offsets_path.c_str())" in harness
    sniper = read("bench/src_sniper/sg_kernel.cc")
    assert "context/K2 sideband export failed" in sniper


def test_sniper_dry_run_migration_updates_virtual_text(tmp_path):
    path = ROOT / "scripts/setup_sniper.py"
    spec = importlib.util.spec_from_file_location("setup_sniper_dry_run", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    target = tmp_path / "legacy.cc"
    target.write_text("old anchor\n")
    module.SNIPER_DIR = tmp_path
    module._DRY_RUN_OVERLAY_TEXT.clear()
    module.migrate_if_present(target, "old anchor", "migrated anchor", True)
    module.replace_once(target, "migrated anchor", "final content", True)

    assert target.read_text() == "old anchor\n"
    assert module._overlay_text(target, True) == "final content\n"


def test_sniper_build_dry_run_does_not_require_checkout(tmp_path):
    path = ROOT / "scripts/setup_sniper.py"
    spec = importlib.util.spec_from_file_location(
        "setup_sniper_missing_checkout", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.SNIPER_DIR = tmp_path / "missing"
    module.build_sniper(argparse.Namespace(
        skip_build=False,
        build_target="",
        jobs=2,
        dry_run=True,
        skip_deps_check=True,
    ))


def test_sniper_clean_removes_all_capability_markers(tmp_path):
    path = ROOT / "scripts/setup_sniper.py"
    spec = importlib.util.spec_from_file_location(
        "setup_sniper_clean_markers", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.SNIPER_DIR = tmp_path / "snipersim"
    module.VERSION_FILE = tmp_path / ".sniper_version"
    module.OVERLAY_STATUS_FILE = tmp_path / ".sniper_overlays.json"
    module.SNIPER_DIR.mkdir()
    module.VERSION_FILE.write_text("{}")
    module.OVERLAY_STATUS_FILE.write_text("{}")
    module.clean(argparse.Namespace(dry_run=False))
    assert not module.SNIPER_DIR.exists()
    assert not module.VERSION_FILE.exists()
    assert not module.OVERLAY_STATUS_FILE.exists()


def test_streamshield_is_policy_isolated_and_verified():
    runner = read("scripts/experiments/ecg/roi_matrix.py")
    policy_specs = read("scripts/experiments/ecg/policy_specs.py")
    verifier = read("scripts/experiments/ecg/verify/equiv_kernels.py")
    ecg_verifier = read("scripts/experiments/ecg/verify/ecg.py")
    assert "apply_ecg_transport_env" in runner
    assert '"CACHE_FAST": "0"' in runner
    assert '"CACHE_SAMPLED": "0"' in runner
    assert '"CACHE_MULTICORE": "0"' in runner
    assert "StreamShield requested but cache_sim bypass path was inactive" in runner
    assert '"ECG:K2_STREAMSHIELD"' in policy_specs
    assert '"ecg_stream_bypass"' in runner
    assert "--stream-bypass" in verifier
    assert "stream-bypass-reads" in verifier
    assert "stream-bypass-writes" in verifier
    assert "dest // vpl == line_id" in ecg_verifier
    assert "cache_sim_ecg_epoch_region_index" in runner
    assert "Sniper StreamShield requires --sniper-workload sg_kernel" in runner
    assert 'env.get("ECG_STREAM_BYPASS") == "1"' in runner
    assert "--stream-bypass requires --schedule-k 2" in verifier
    assert "SNIPER_ECG_FUSED_K2" in runner
    assert "StreamShield inactive" in runner
    assert 'env.pop("SNIPER_ECG_FUSED_K2", None)' in runner
    assert 'env.pop("SNIPER_ECG_FUSED_VALIDATE", None)' in runner
    assert 'env["SNIPER_CACHE_LINE_SIZE"] = str(args.line_size)' in runner
    assert "GEM5_ECG_STREAM_REQUEST_BOUND" in runner
    assert "previous == record_count" in runner
    assert "mmap.mmap" in runner
    assert ".read_bytes()" not in runner.split(
        "def validate_sniper_fused_receipts", 1
    )[1].split("def ", 1)[0]
    assert "fused_k2 = False" in runner


def test_streamshield_is_pr_only_in_detailed_kernels():
    gem5_bfs = read("bench/src_gem5/bfs.cc")
    sniper = read("bench/src_sniper/sg_kernel.cc")
    gem5_export = gem5_bfs.split("gem5_export_context(", 1)[1].split(");", 1)[0]
    assert "pair_flat.data()" not in gem5_export
    bfs_export = sniper.split(
        "sniper_export_context(\n        regions, 1, graph", 1
    )[1].split(");", 1)[0]
    assert "bfs_pair_flat.data()" not in bfs_export


def test_streamshield_setup_migrates_and_rebuilds():
    gem5_setup = read("scripts/setup_gem5.py")
    sniper_setup = read("scripts/setup_sniper.py")
    assert "Incrementally rebuilding gem5" in gem5_setup
    assert 'GEM5_DEFAULT_COMMIT = "b1a44b89c7bae73fae2dc547bc1f871452075b85"' in gem5_setup
    assert "def verify_installation_postconditions" in gem5_setup
    assert "PATCH_STATE_FILE" in gem5_setup
    assert "PATCH_STATE_FILE.unlink(missing_ok=True)" in gem5_setup
    assert "Tracked gem5 patch changed after installation" in gem5_setup
    assert "S68-QUEUE-SERVICING-PATCH" in gem5_setup
    assert "Required gem5 patch file missing" in gem5_setup
    assert "unsupported --tag" in gem5_setup
    assert "base_stream_bypass_request_flag.patch" in gem5_setup
    assert "prefetch_stream_bypass.patch" in gem5_setup
    assert "def migrate_if_present" in sniper_setup
    assert 'SNIPER_DEFAULT_REF = "56505e42fd98bca863fac181e769bd3c98d2bb33"' in sniper_setup
    main_block = sniper_setup.split("def main(argv:", 1)[1]
    assert main_block.index("OVERLAY_STATUS_FILE.unlink(missing_ok=True)") < \
        main_block.index("install_graphbrew_configs(args)")
    assert main_block.index("graphbrew_smoke_test(args)") < \
        main_block.index("write_overlay_status(copied_files)")
    assert "migrate_if_present(\n        magic_server, old_decode, new_decode" in sniper_setup


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
