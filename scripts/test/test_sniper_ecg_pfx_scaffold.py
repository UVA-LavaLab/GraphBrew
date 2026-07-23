import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text()


def test_context_handler_normalizes_fresh_indent(tmp_path) -> None:
    path = ROOT / "scripts/setup_sniper.py"
    spec = importlib.util.spec_from_file_location(
        "setup_sniper_context_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = tmp_path / "magic_server.cc"
    target.write_text(
        "        if (arg0 == graphbrew::sniper::GRAPHBREW_SET_VERTEX_WORK_ID)\n"
        "        {\n"
        "           return 0;\n"
        "        }\n"
    )
    module.normalize_context_ready_handler(target, False)
    module.normalize_context_ready_handler(target, False)
    text = target.read_text()
    assert text.count("GRAPHBREW_CONTEXT_READY_WORK_ID") == 1
    assert "        if (arg0 == graphbrew::sniper::GRAPHBREW_CONTEXT_READY_WORK_ID)" in text


def test_sniper_harness_defines_ecg_pfx_hint_surface() -> None:
    text = read("bench/include/sniper_sim/sniper_harness.h")
    assert "GRAPHBREW_SNIPER_USER_ECG_PFX_TARGET" in text
    assert "SNIPER_ENABLE_ECG_PFX_HINTS" in text
    assert "SNIPER_ECG_PFX_HINT_FILTER" in text
    assert "SNIPER_ECG_PFX_FILTER_ELEM_SIZE" in text
    assert "SNIPER_ECG_PFX_FILTER_LINE_SIZE" in text
    assert "should_emit_ecg_pfx_hint" in text
    assert "SNIPER_ECG_PFX_TARGET" in text


def test_sniper_harness_caches_hot_path_environment_controls() -> None:
    text = read("bench/include/sniper_sim/sniper_harness.h")
    for function_name in (
        "hints_enabled",
        "ecg_pfx_hints_enabled",
        "ecg_extract_enabled",
    ):
        body = text.split(f"inline bool {function_name}()", 1)[1].split(
            "\n}", 1)[0]
        assert "static const bool enabled" in body
    pfx_filter = text.split(
        "inline bool should_emit_ecg_pfx_hint", 1)[1].split("\n}", 1)[0]
    assert "static const int capacity" in pfx_filter
    assert "static const uint64_t vertices_per_line" in pfx_filter


def test_sniper_fused_k2_skips_software_only_delivery() -> None:
    text = read("bench/src_sniper/sg_kernel.cc")
    assert text.count("const bool software_k2_delivery =") == 5
    assert text.count("const bool ecg_pfx_hints_on =") == 3
    assert text.count("const bool no_delivery_pair_loop =") == 4
    assert text.count("if (no_delivery_pair_loop)") == 5
    assert text.count("if (software_k2_delivery) {") == 3
    assert text.count("if (!fused_k2_model) {") >= 6
    assert "if (delivered_k2 && !fused_k2_model)" in text
    assert "!graphbrew_sniper::ecg_pfx_hints_enabled()" not in text


def test_sniper_ecg_pfx_prefetcher_overlay_exists() -> None:
    header = read("bench/include/sniper_sim/overlays/common/core/memory_subsystem/parametric_dram_directory_msi/ecg_pfx_prefetcher.h")
    source = read("bench/include/sniper_sim/overlays/common/core/memory_subsystem/parametric_dram_directory_msi/ecg_pfx_prefetcher.cc")
    assert "class EcgPfxPrefetcher" in header
    assert "consumePrefetchTargetHint" in source
    assert "ecg-pfx-prefetcher" in source
    assert "target-hints-seen" in source


def test_sniper_context_tracks_prefetch_target_hint() -> None:
    header = read("bench/include/sniper_sim/overlays/common/core/memory_subsystem/cache/graph_cache_context_sniper.h")
    source = read("bench/include/sniper_sim/overlays/common/core/memory_subsystem/cache/graph_cache_context_sniper.cc")
    assert "GRAPHBREW_ECG_PFX_TARGET_WORK_ID" in header
    for symbol in (
        "setPrefetchTargetHint",
        "hasPrefetchTargetHint",
        "getPrefetchTargetHint",
        "consumePrefetchTargetHint",
        "clearPrefetchTargetHint",
    ):
        assert symbol in header
        assert symbol in source


def test_sniper_benchmarks_emit_ecg_pfx_targets() -> None:
    for relative_path in ("bench/src_sniper/pr.cc", "bench/src_sniper/bfs.cc", "bench/src_sniper/sssp.cc"):
        text = read(relative_path)
        assert "SNIPER_ECG_PFX_TARGET" in text
        assert "SNIPER_ECG_PFX_LOOKAHEAD" in text


def test_sniper_runner_wires_ecg_pfx_prefetcher() -> None:
    text = read("scripts/experiments/ecg/roi_matrix.py")
    assert 'if args.prefetcher == "ECG_PFX":' in text
    assert '"Sniper ECG_PFX requires overlays' in text
    assert 'prefetcher"] = "ecg_pfx"' in text
    assert 'SNIPER_ENABLE_ECG_PFX_HINTS' in text
    assert 'SNIPER_ECG_PFX_HINT_FILTER' in text
    assert 'SNIPER_ECG_PFX_FILTER_ELEM_SIZE' in text
    assert 'SNIPER_ECG_PFX_FILTER_LINE_SIZE' in text
    assert 'ecg_pfx_target_hints_seen' in text
    assert 'ecg_pfx_activity' in text


def test_setup_sniper_patches_simuser_hint_dispatch() -> None:
    text = read("scripts/setup_sniper.py")
    assert "patch_graphbrew_simuser_overlay" in text
    assert "patch_ecg_pfx_prefetcher_overlay" in text
    assert "ecg_pfx_prefetcher.h" in text
    assert "EcgPfxPrefetcher" in text
    assert "core/memory_subsystem/cache/graph_cache_context_sniper.h" in text
    assert "GRAPHBREW_SET_VERTEX_WORK_ID" in text
    assert "GRAPHBREW_ECG_PFX_TARGET_WORK_ID" in text
    assert "setCurrentVertexHint" in text
    assert "setPrefetchTargetHint" in text


def test_sniper_ecg_extract_payload_and_runner_are_faithful() -> None:
    harness = read("bench/include/sniper_sim/sniper_harness.h")
    setup = read("scripts/setup_sniper.py")
    context_h = read(
        "bench/include/sniper_sim/overlays/common/core/memory_subsystem/cache/"
        "graph_cache_context_sniper.h"
    )
    context_cc = read(
        "bench/include/sniper_sim/overlays/common/core/memory_subsystem/cache/"
        "graph_cache_context_sniper.cc"
    )
    cache = read(
        "bench/include/sniper_sim/overlays/common/core/memory_subsystem/cache/"
        "cache_set_ecg.cc"
    )
    runner = read("scripts/experiments/ecg/roi_matrix.py")

    # Keep NodeID[31:0]+epoch[15:0] inside the magic ABI's reliable low 48 bits.
    assert "(vertex & 0xFFFFFFFFULL)" in harness
    assert "static_cast<uint64_t>(epoch) << 32" in harness
    assert "arg1 & 0xFFFFFFFFULL" in setup
    assert "(arg1 >> 32) & 0xFFFFULL" in setup
    assert "epoch) << 48" not in harness
    assert "(arg1 >> 32) & 0x3ULL" in setup
    assert "(arg1 >> 34) & 0x7FFFULL" in setup
    assert "(arg1 >> 49) & 0x7FFFULL" in setup

    # SimMagic inputs must not alias the RAX output and get overwritten by cmd=5.
    assert "early-clobber: inputs cannot alias RAX" in setup
    assert "replace(old_constraint, new_constraint, 3)" in setup
    assert "old_decode" in setup
    assert "new_decode" in setup

    # Shared LLC consumes only the requesting core's newest stable delivery.
    assert "lookupEcgEpochAnyCore" not in context_h
    assert "ecgEpochGlobalSequence" in context_cc
    assert "before != after" in context_cc
    assert "lookupLineEcgEpochPair(" in cache
    assert "lookupEcgEpochPair(" in context_cc
    assert "recordEcgEpochPair(" in context_cc
    assert "GRAPHBREW_CONTEXT_READY_WORK_ID" in context_h
    assert "GRAPHBREW_CONTEXT_READY_WORK_ID" in setup
    assert "ECG-CONTEXT-READY sim=sniper" in setup
    assert "SNIPER_REQUIRE_POPT_MATRIX" in setup
    assert "reref=%d" in setup
    assert "normalize_context_ready_handler" in setup
    assert "text.count(marker) != 1" in setup
    for source_name in (
        "pr_kernel_smoke.cc", "bfs_kernel_smoke.cc",
        "sssp_kernel_smoke.cc",
    ):
        assert "notify_context_ready()" in read(
            f"bench/src_sniper/{source_name}")
    assert "bool epoch_property[64]" in cache
    assert "hasCurrentVertexHint(" in cache
    assert "m_property_lines[accessed_index] = context.isPropertyData" in cache
    assert "m_set_info->increment(accessed_index)" in cache
    assert "hasCurrentVertexHint(requester_core)" in cache
    assert "Sniper graph policy completed without a loaded graph context" in runner
    assert "[ECG-CONTEXT-READY sim=sniper loaded=1" in runner
    assert "m_property_lines[way] =" not in cache.split(
        "CacheSetECG::findECGGraspPoptVictim", 1)[1].split(
            "CacheSetECG::getReplacementIndex", 1)[0]
    assert "line_plus1" in context_cc
    assert "vertex_plus1" not in context_cc
    assert "ecgVerticesPerLine()" in context_cc
    assert "isEcgEpochData" in context_h
    assert "SNIPER_ECG_EPOCH_REGION" in context_cc
    assert "GRAPHBREW_ECG_EXTRACT2_WORK_ID" in setup
    assert "epochPairDistance(" in cache
    assert "currentNucaRequesterCore()" in cache
    assert "address, requester, data_buf" in setup
    assert "NucaCache::read(IntPtr address, core_id_t requester" in setup
    assert "NucaCache::write(IntPtr address, core_id_t requester" in setup

    # Paper runs use the real outer clock + delivered epoch, not the live oracle.
    assert 'env["SNIPER_ENABLE_VERTEX_HINTS"] = "1"' in runner
    assert 'env["SNIPER_ENABLE_ECG_EXTRACT"] = "1"' in runner
    assert 'int(args.line_size) // 4' in runner
    assert "requires --sniper-workload sg_kernel" in runner
    assert 'os.environ.get("ECG_FORCE_DELIVERY") == "1"' in runner
    assert "ws[w].recency = m_last_touch[w];" in cache


def test_sniper_mask_only_uses_transport_matched_loops():
    source = read("bench/src_sniper/sg_kernel.cc")
    runner = read("scripts/experiments/ecg/roi_matrix.py")

    assert "bool k2_transport_matched_enabled()" in source
    assert source.count(
        "const bool k2_transport_matched = "
        "k2_transport_matched_enabled();") == 5
    assert source.count("[K2_TRANSPORT_MATCHED]") == 6
    assert "k2_transport_matched && !k2_trace_on" in source
    assert 'env["SNIPER_K2_TRANSPORT_MATCHED"] = "1"' in runner
    assert 'env["SNIPER_ENABLE_ECG_EXTRACT"] = "1"' in runner
    assert '"sniper_transport_record_bytes"] = 8' in runner
    assert '"matched_mask_only_sideband_model"' in runner


def test_sniper_ecg_host_profile_covers_cache_callbacks():
    cache = read(
        "bench/include/sniper_sim/overlays/common/core/memory_subsystem/cache/"
        "cache_set_ecg.cc")
    context = read(
        "bench/include/sniper_sim/overlays/common/core/memory_subsystem/cache/"
        "graph_cache_context_sniper.cc")

    assert 'std::getenv("SNIPER_ECG_HOST_PROFILE")' in cache
    assert "[ECG-HOST-PROFILE" in cache
    assert "(void)ecgHostProfile();" in cache
    assert "Kind::Replacement" in cache
    assert "Kind::Update" in cache
    assert "Kind::Prepare" in cache
    assert 'std::getenv("SNIPER_K2_LOOKUP_PROFILE")' in context
    assert "[K2-LOOKUP-PROFILE" in context
    assert "k2_profile_classify_ns" in context
    assert "k2_profile_search_ns" in context