#!/usr/bin/env python3
"""Regression tests for the gem5 ECG_PFX scaffold wiring."""

import importlib.util
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETUP_GEM5_PATH = PROJECT_ROOT / "scripts" / "setup_gem5.py"
spec = importlib.util.spec_from_file_location("setup_gem5", SETUP_GEM5_PATH)
setup_gem5 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["setup_gem5"] = setup_gem5
spec.loader.exec_module(setup_gem5)


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
    x86_m5op = read("bench/include/gem5_sim/gem5/util/m5/src/abi/x86/m5op.S")
    generic_m5ops = read("bench/include/gem5_sim/gem5/include/gem5/asm/generic/m5ops.h")

    assert 'asm volatile (".byte 0x0F, 0x04' in harness
    assert '"D"(work_id)' in harness
    assert '"S"(argument)' in harness
    assert "M5OP_WORK_BEGIN" in harness
    assert ".byte 0x0F, 0x04" in x86_m5op
    assert "#define M5OP_WORK_BEGIN         0x5a" in generic_m5ops


def test_gem5_tiny_smoke_uses_instruction_delivery():
    text = read("scripts/experiments/ecg/final_paper_manifest.json")

    assert '"profiles": ["gem5_ecg_pfx_tiny_smoke"]' in text
    assert '"ecg_pfx_delivery": "instruction"' in text


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


def test_setup_gem5_uses_dedicated_x86_extract_work_id():
    text = read("scripts/setup_gem5.py")
    assert "legacy content-based PFX/mask multiplexing" in text
    assert "GRAPHBREW_ECG_EXTRACT_MASK_WORK_ID" in text


def test_three_sim_showcase_selects_explicit_roi_section():
    text = read("scripts/experiments/ecg/three_sim_showcase.py")
    assert "section == 1" in text
    assert "automatic final section after the ROI" in text