#!/usr/bin/env python3
"""Regression tests for the gem5 ECG_PFX scaffold wiring."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETUP_GEM5_PATH = PROJECT_ROOT / "scripts" / "setup_gem5.py"
spec = importlib.util.spec_from_file_location("setup_gem5", SETUP_GEM5_PATH)
setup_gem5 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["setup_gem5"] = setup_gem5
spec.loader.exec_module(setup_gem5)

ROI_MATRIX_PATH = PROJECT_ROOT / "scripts/experiments/ecg/roi_matrix.py"
roi_spec = importlib.util.spec_from_file_location("ecg_roi_matrix", ROI_MATRIX_PATH)
roi_matrix = importlib.util.module_from_spec(roi_spec)
assert roi_spec.loader is not None
sys.modules["ecg_roi_matrix"] = roi_matrix
roi_spec.loader.exec_module(roi_matrix)


def read(path: str) -> str:
    return (PROJECT_ROOT / path).read_text()


def test_setup_gem5_installs_ecg_pfx_overlays():
    overlay_values = set(setup_gem5.OVERLAY_FILE_MAP.values())

    assert "mem/cache/prefetch/ecg_pfx.hh" in overlay_values
    assert "mem/cache/prefetch/ecg_pfx.cc" in overlay_values
    assert "arch/riscv/isa/formats/ecg.isa" in overlay_values


def test_prefetch_sconscript_registers_ecg_pfx():
    text = read("bench/include/gem5_sim/overlays/mem/cache/prefetch/SConscript.patch")

    assert "GraphEcgPfxPrefetcher" in text
    assert "Source('ecg_pfx.cc')" in text


def test_graph_se_accepts_ecg_pfx_prefetcher():
    text = read("bench/include/gem5_sim/configs/graphbrew/graph_se.py")

    assert 'choices=["none", "DROPLET", "ECG_PFX", "STRIDE"]' in text
    assert "GEM5_ENABLE_ECG_PFX_HINTS" in text
    assert "GEM5_ECG_PFX_LOOKAHEAD" in text
    assert "GEM5_ECG_PFX_HINT_FILTER" in text
    assert "GEM5_ECG_PFX_FILTER_ELEM_SIZE" in text
    assert "GEM5_ECG_PFX_FILTER_LINE_SIZE" in text
    assert "GEM5_ENABLE_ECG_EXTRACT" in text
    assert "make_ecg_pfx_prefetcher" in text


def test_graph_se_caps_instructions_relative_to_roi():
    text = read("bench/include/gem5_sim/configs/graphbrew/graph_se.py")
    assert "system.exit_on_work_items = True" in text
    assert "system.cpu.scheduleInstStop(" in text
    assert '"ROI instruction cap reached"' in text
    assert "simulation exited before ROI work-begin" in text


def test_gem5_harness_defines_ecg_pfx_m5ops_macro():
    text = read("bench/include/gem5_sim/gem5_harness.h")

    assert "GEM5_WORK_ECG_PFX_TARGET" in text
    assert "GEM5_ECG_PFX_TARGET" in text
    assert "gem5_should_emit_ecg_pfx_hint" in text
    assert "gem5_ecg_extract_target_instruction" in text
    assert "gem5_ecg_pfx_target_instruction" in text
    assert ".insn r 0x0b" in text


def test_x86_instruction_path_emits_gem5_pseudo_op_bytes():
    harness = read("bench/include/gem5_sim/gem5_harness.h")

    assert 'asm volatile (".byte 0x0F, 0x04' in harness
    assert '"D"(work_id)' in harness
    assert '"S"(argument)' in harness
    assert "M5OP_WORK_BEGIN" in harness


def test_riscv_ecg_extract_overlay_uses_custom0_opcode():
    text = read("bench/include/gem5_sim/overlays/arch/riscv/isa/decoder_ecg_extract.isa")

    # custom-0 opcode space (full opcode 0x0b -> OPCODE5 0x02), FUNCT3 decode.
    assert "0x02: decode FUNCT3" in text
    assert "ecg_extract" in text
    # WIDE (S10.2) mode-6 delivery: next-ref epoch + a widened 24-bit prefetch
    # target (dbg/popt reclaimed; see packMaskEpochWide). Hints are delivered via
    # the per-vertex metadata table and the legacy single-slot mailbox.
    assert "epoch" in text
    assert "pfx_target" in text
    assert "storeEcgMetadataByVertex" in text
    assert "setDecodedEcgExtractHint" in text
    assert "setPrefetchTargetHint" in text


def test_gem5_graph_context_stores_decoded_ecg_extract_hint():
    text = read("bench/include/gem5_sim/overlays/mem/cache/replacement_policies/graph_cache_context_gem5.hh")

    assert "decodedEcgRealVertexStorage" in text
    assert "decodedEcgMetadataStorage" in text
    assert "setDecodedEcgExtractHint" in text
    assert "GRAPHBREW_ECG_EXTRACT_MASK_WORK_ID" in text


def test_gem5_schedule2_delivery_is_pair_aware():
    harness = read("bench/include/gem5_sim/gem5_harness.h")
    decoder = read(
        "bench/include/gem5_sim/overlays/arch/riscv/isa/"
        "decoder_ecg_extract.isa"
    )
    context = read(
        "bench/include/gem5_sim/overlays/mem/cache/replacement_policies/"
        "graph_cache_context_gem5.hh"
    )
    policy = read(
        "bench/include/gem5_sim/overlays/mem/cache/replacement_policies/"
        "ecg_rp.cc"
    )
    setup = read("scripts/setup_gem5.py")
    graph_se = read(
        "bench/include/gem5_sim/configs/graphbrew/graph_se.py")

    assert "GEM5_ECG_EXTRACT2" in harness
    assert "0x01: ecg_extract2" in decoder
    assert "(packed >> 32) & 0x3" in decoder
    assert "(packed >> 34) & 0x7FFF" in decoder
    assert "(packed >> 49) & 0x7FFF" in decoder
    assert "setDecodedEcgExtractHint2" in decoder
    assert "0x03: ecg_load_k2" in decoder
    assert "0x06: ecg_mload_k2_u32" in decoder
    assert "0x07: ecg_mload_k2_s32" in decoder
    assert "0x08: ecg_mload_k2_u64" in decoder
    assert "0x09: ecg_mload_k2_compact_u32" in decoder
    assert "0x0A: ecg_mload_k2_f32" in decoder
    assert "xc->setEcgLoadHint2(" in decoder
    assert "lookupDecodedEcgHint2" in context
    assert "isEcgEpochData" in context
    assert "ecg_epoch2" in policy
    assert "ecg_epoch_count" in policy
    assert "bool valid;" in read(
        "bench/include/gem5_sim/overlays/mem/cache/replacement_policies/"
        "ecg_rp.hh"
    )
    assert "if (!getData(candidate)->valid) return candidate;" in policy
    assert "ctx.isEcgEpochData(getData(c)->line_addr)" in policy
    assert "setDueling && graph::hasCurrentVertexHint()" in policy
    assert "dd->ecg_dbg_tier < 1 || dd->ecg_dbg_tier > 3" in policy
    assert "ctx.classifyGRASP(addr, llcSize, ghf)" in policy
    assert "isa_dbg >= 1 && isa_dbg <= 3" in policy
    assert "epochPairDistance(" in policy
    assert policy.count("readEcgEpochPair(") >= 2
    assert policy.count(
        "!got && !requestBoundEcgProducerEnabled()") >= 2
    assert "GRAPHBREW_ECG_EXTRACT2_WORK_ID" in setup

    request_ext = read(
        "bench/include/gem5_sim/overlays/mem/cache/replacement_policies/"
        "ecg_epoch_request_ext.hh")
    assert "attachEcgEpochPair" in request_ext
    assert "readEcgEpochPair" in request_ext
    assert "epoch2_" in request_ext
    assert "epoch_count_" in request_ext

    exec_patch = read(
        "bench/include/gem5_sim/overlays/cpu/exec_context_ecg_producer.patch")
    dyn_patch = read(
        "bench/include/gem5_sim/overlays/cpu/o3/dyn_inst_ecg_producer.patch")
    lsq_patch = read(
        "bench/include/gem5_sim/overlays/cpu/o3/lsq_ecg_producer.patch")
    assert "setEcgLoadHint2" in exec_patch
    assert "setEcgLoadHint2" in dyn_patch
    assert "attachEcgEpochPair" in lsq_patch
    assert 'schedule_k == "2"' in graph_se
    assert '"GRASP_HOT_FRACTION"' in graph_se


def test_schedule2_runner_selects_adaptive_variants_and_rejects_o3(monkeypatch):
    monkeypatch.delenv("ECG_VARIANT", raising=False)
    monkeypatch.setenv("ECG_EDGE_MASK_SCHED", "2")
    assert roi_matrix.effective_ecg_variant(
        SimpleNamespace(benchmark="pr")) == "epoch_first"
    assert roi_matrix.effective_ecg_variant(
        SimpleNamespace(benchmark="bfs")) == "degree_first"
    assert roi_matrix.effective_ecg_variant(
        SimpleNamespace(benchmark="sssp")) == "degree_first"
    assert roi_matrix.effective_ecg_variant(
        SimpleNamespace(benchmark="bc")) == "rrip_first"
    assert roi_matrix.effective_ecg_variant(
        SimpleNamespace(benchmark="cc")) == "rrip_first"

    monkeypatch.setenv("ECG_VARIANT", "rrip_first")
    assert roi_matrix.effective_ecg_variant(
        SimpleNamespace(benchmark="pr")) == "rrip_first"

    runner = read("scripts/experiments/ecg/roi_matrix.py")
    graph_se = read("bench/include/gem5_sim/configs/graphbrew/graph_se.py")
    assert "Schedule-2 O3 requires the RISC-V masked property-load" in runner
    assert 'args.gem5_cpu_type == "O3"' in runner
    assert "request_bound_k2" in graph_se
    assert "Schedule-2 O3 requires the masked property-load path" in graph_se
    assert "prefetcher none or STRIDE" in runner
    assert "GEM5_ECG_EPOCH_REGION_INDICES" in graph_se
    assert "GEM5_ECG_EPOCH_REGION_INDEX" in graph_se
    assert "GEM5_ECG_ISA_VARIANT" in graph_se
    verifier = read("scripts/experiments/ecg/verify/ecg.py")
    assert "required = set(range(32))" in verifier


def test_gem5_k2_uses_configured_epoch_count_not_packed4_cap():
    for path in (
        "bench/src_gem5/pr.cc",
        "bench/src_gem5/bfs.cc",
        "bench/src_gem5/sssp.cc",
        "bench/src_gem5/bc.cc",
        "bench/src_gem5/cc.cc",
    ):
        text = read(path)
        assert 'gem5_env_int_clamped("ECG_EDGE_MASK_EPOCHS"' in text
        assert "ecg_sched_k != 2" in text
        assert "requested_epoch_count" in text
    pr = read("bench/src_gem5/pr.cc")
    assert "Schedule-2 record ON" in pr
    assert "buildInEdgeEpochPairRecords" in pr
    cache_context = read("bench/include/cache_sim/graph_cache_context.h")
    assert 'std::getenv("ECG_EDGE_MASK_PACK") && sched_k != 2' in cache_context


def test_gem5_k2_mailbox_is_cleared_after_governed_load():
    context = read(
        "bench/include/gem5_sim/overlays/mem/cache/replacement_policies/"
        "graph_cache_context_gem5.hh")
    harness = read("bench/include/gem5_sim/gem5_harness.h")
    assert "clearDecodedEcgExtractHint()" in context
    assert "if (tier == 0)" in context
    assert "GEM5_ECG_CLEAR_EXTRACT2_HINT" in harness
    for path in (
        "bench/src_gem5/pr.cc",
        "bench/src_gem5/bfs.cc",
        "bench/src_gem5/sssp.cc",
        "bench/src_gem5/bc.cc",
        "bench/src_gem5/cc.cc",
    ):
        assert "GEM5_ECG_CLEAR_EXTRACT2_HINT()" in read(path)
    runner = read("scripts/experiments/ecg/roi_matrix.py")
    assert '"prototype_instruction_delivery"' in runner
    assert '"packed8+k2+ecg.extract2"' in runner


def test_gem5_exports_prefetch_and_dram_traffic_metrics():
    runner = read("scripts/experiments/ecg/roi_matrix.py")
    for metric in (
        "l3_prefetch_misses",
        "l3_prefetch_accesses",
        "dram_read_bytes",
        "dram_write_bytes",
        "dram_prefetch_read_bytes",
    ):
        assert f'"{metric}"' in runner
    assert "system.mem_ctrl.dram.bytesRead::total" in runner
    assert "system.l3cache.overallMisses::l2cache.prefetcher" in runner
    assert '"gem5_stats_sections_seen": len(sections)' in runner
    assert "row.update(sections[0])" in runner


def test_gem5_srrip_is_true_three_bit_srrip():
    text = read("bench/include/gem5_sim/configs/graphbrew/graph_cache_config.py")
    assert '"SRRIP": lambda: RRIPRP(num_bits=3)' in text
    assert '"SRRIP": lambda: BRRIPRP(btp=0)' not in text


def test_roi_matrix_auto_selects_riscv_ecg_delivery():
    text = read("scripts/experiments/ecg/roi_matrix.py")
    graph_se = read("bench/include/gem5_sim/configs/graphbrew/graph_se.py")
    assert 'env["GEM5_FORCE_ECG_PLOAD"] = "1"' in text
    assert '"packed4+ecg.extract"' in text
    assert 'os.environ.get("GEM5_FORCE_ECG_LOAD") == "1"' in text
    assert '"ecg.pload-request-bound"' in text
    assert 'env["GEM5_ECG_PRODUCER"] = "1"' in text
    assert '"ECG_EDGE_MASK_PREFETCH"' in text
    assert 'row["gem5_ecg_delivery"] = "ecg.load"' not in text
    assert 'base["gem5_ecg_delivery"] = gem5_ecg_delivery' in text
    assert 'os.environ.get("ECG_FORCE_DELIVERY") == "1"' in graph_se
    assert 'ecg_pfx_enabled = args.prefetcher == "ECG_PFX"' in graph_se
    assert "or ecg_epoch_delivery" not in graph_se


def test_epoch_extract_is_not_gated_by_prefetch_enable():
    harness = read("bench/include/gem5_sim/gem5_harness.h")
    good = (
        "#define GEM5_ECG_EXTRACT_MASK(mask_u64) \\\n"
        "    do { \\\n"
        "        if (gem5_ecg_extract_enabled()) {"
    )
    bad = (
        "#define GEM5_ECG_EXTRACT_MASK(mask_u64) \\\n"
        "    do { \\\n"
        "        if (gem5_ecg_pfx_hints_enabled() && "
        "gem5_ecg_extract_enabled()) {"
    )
    assert harness.count(good) == 2
    assert bad not in harness
    assert "GEM5_WORK_ECG_EXTRACT_MASK" in harness


def test_k2_property_load_clears_mailbox_without_extra_instruction():
    decoder = read(
        "bench/include/gem5_sim/overlays/arch/riscv/isa/"
        "decoder_ecg_extract.isa")
    k2_load = decoder.split("0x03: ecg_load_k2", 1)[1].split(
        "}}, ea_code={{", 1)[0]
    assert "Rd = Mem_uw;" in k2_load
    assert "clearDecodedEcgExtractHint();" in k2_load
    assert "traceExpectedEcgExtractHint2(packed);" in decoder

    harness = read("bench/include/gem5_sim/gem5_harness.h")
    helper = harness.split("inline uint32_t gem5_ecg_load_k2", 1)[1].split(
        "inline uint32_t gem5_ecg_extract2_instruction", 1)[0]
    assert "gem5_trace_ecg_k2_expect" not in helper

    for kernel in ("bfs", "sssp", "bc", "cc"):
        source = read(f"bench/src_gem5/{kernel}.cc")
        for block in source.split("if (ecg_k2_pload_on) {")[1:]:
            canonical = block.split("} else {", 1)[0]
            assert "GEM5_ECG_CLEAR_EXTRACT2_HINT" not in canonical, kernel

    pr = read("bench/src_gem5/pr.cc")
    canonical_pr = pr.split("if (ecg_k2_pload_on) {", 1)[1].split(
        "continue;", 1)[0]
    assert "GEM5_ECG_CLEAR_EXTRACT2_HINT" not in canonical_pr
    assert "GEM5_ECG_CLEAR_EXTRACT2_HINT" in pr


def test_k2_mask_only_variant_is_distinct_from_indexed_load():
    harness = read("bench/include/gem5_sim/gem5_harness.h")
    runner = read("scripts/experiments/ecg/roi_matrix.py")
    decoder = read(
        "bench/include/gem5_sim/overlays/arch/riscv/isa/"
        "decoder_ecg_extract.isa")
    assert "GEM5_ECG_ISA_VARIANT" in harness
    assert '"ecg_isa_variant"' in runner
    assert 'env["GEM5_ECG_ISA_VARIANT"] = args.ecg_isa_variant' in runner
    assert "SNIPER_K2_TRANSPORT_MATCHED" in runner
    assert "matched-k2m-sideband-model" in runner
    assert '"prototype_mask_only_load"' in runner
    assert "prototype current-vertex channel" in runner
    assert "transport.schedule_k == 2" in runner
    assert 'std::strcmp(value, "mask") == 0' in harness
    assert '".insn r 0x0b, 0x2, 0x18' in harness
    assert '".insn r 0x0b, 0x2, 0x1c' in harness
    assert '".insn r 0x0b, 0x2, 0x20' in harness
    assert '".insn r 0x0b, 0x2, 0x24' in harness
    assert '".insn r 0x0b, 0x2, 0x28' in harness

    u32 = decoder.split("0x06: ecg_mload_k2_u32", 1)[1].split(
        "// 0x07 K2-M S32.D32", 1)[0]
    s32 = decoder.split("0x07: ecg_mload_k2_s32", 1)[1].split(
        "// 0x08 K2-M U64.D32", 1)[0]
    u64 = decoder.split("0x08: ecg_mload_k2_u64", 1)[1].split(
        "// 0x09 K2-M U32.CW24", 1)[0]
    compact = decoder.split(
        "0x09: ecg_mload_k2_compact_u32", 1)[1].split(
            "// 0x0A K2-M F32.D32", 1)[0]
    f32 = decoder.split("0x0A: ecg_mload_k2_f32", 1)[1].split(
        "\n                }", 1)[0]
    for block in (u32, s32, u64, compact, f32):
        assert "EA = rvZext(Rs1);" in block
        assert "Rs1 +" not in block
        assert "xc->setEcgLoadHint2(" in block
    assert "Rd = Mem_uw;" in u32
    assert "Rd_sd = Mem_sw;" in s32
    assert "Rd = Mem_ud;" in u64
    assert "packed & 0x00FFFFFFULL" in compact
    assert "Fd_bits = fd.v;" in f32
    assert "FloatMemReadOp" in f32
    assert "Rd =" not in f32

    expected_helpers = {
        "pr": ("gem5_ecg_mload_k2_f32",),
        "bfs": ("gem5_ecg_mload_k2_s32",),
        "sssp": (
            "gem5_ecg_mload_k2_s32",
            "gem5_ecg_mload_k2_compact_u32",
        ),
        "bc": (
            "gem5_ecg_mload_k2_s32",
            "gem5_ecg_mload_k2_u64",
        ),
        "cc": ("gem5_ecg_mload_k2_s32",),
    }
    for kernel, helpers in expected_helpers.items():
        source = read(f"bench/src_gem5/{kernel}.cc")
        assert "gem5_ecg_k2_mask_only_enabled()" in source
        for helper in helpers:
            assert helper in source
        assert "ECG_K2_MLOAD" in source
        assert "ECG_K2_ILOAD" in source


def test_riscv_gem5_build_unswitches_runtime_policy_loops():
    makefile = read("Makefile")
    flags = makefile.split("CXXFLAGS_GEM5_RISCV :=", 1)[1].splitlines()[0]
    assert "-funswitch-loops" in flags


def test_setup_gem5_uses_dedicated_x86_extract_work_id():
    text = read("scripts/setup_gem5.py")
    assert "legacy content-based PFX/mask multiplexing" in text
    assert "GRAPHBREW_ECG_EXTRACT_MASK_WORK_ID" in text