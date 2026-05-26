from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text()


def test_sniper_harness_defines_ecg_pfx_hint_surface() -> None:
    text = read("bench/include/sniper_sim/sniper_harness.h")
    assert "GRAPHBREW_SNIPER_USER_ECG_PFX_TARGET" in text
    assert "SNIPER_ENABLE_ECG_PFX_HINTS" in text
    assert "SNIPER_ECG_PFX_HINT_FILTER" in text
    assert "SNIPER_ECG_PFX_FILTER_ELEM_SIZE" in text
    assert "SNIPER_ECG_PFX_FILTER_LINE_SIZE" in text
    assert "should_emit_ecg_pfx_hint" in text
    assert "SNIPER_ECG_PFX_TARGET" in text


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