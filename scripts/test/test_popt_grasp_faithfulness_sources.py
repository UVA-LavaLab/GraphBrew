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


def test_grasp_trace_header_fraction_and_cold_fill_are_upstream_faithful():
    cache_sim = read("bench/include/cache_sim/cache_sim.h")
    graph_ctx = read("bench/include/cache_sim/graph_cache_context.h")
    gem5_context = read("bench/include/gem5_sim/overlays/mem/cache/replacement_policies/graph_cache_context_gem5.hh")
    sniper_context = read("bench/include/sniper_sim/overlays/common/core/memory_subsystem/cache/graph_cache_context_sniper.cc")
    compare_script = read("scripts/experiments/ecg/upstream_policy_compare.py")

    assert "grasp_hot_percent" in graph_ctx
    assert "grasp_hot_percent = 15" in graph_ctx
    assert "bool grasp_region = true" in graph_ctx
    assert "if (!r.grasp_region) continue;" in graph_ctx
    assert "registerGRASPTraceRegion" in graph_ctx
    assert "double hot_fraction = 0.5" in cache_sim
    assert "hot_bound += 8" in graph_ctx
    assert "moderate_bound += 8" in graph_ctx
    # GRASP-faithful array-relative normalization (sim_3way_parity_audit_v2.md
    # §13): classifyGRASP marks the hot region as a fraction of the VERTEX ARRAY
    # (frac x array_bytes), NOT of the LLC; default 0.15 (~ Faldu's vertex-relative
    # "10%"). This auto-scales where the old fixed 0.50-of-LLC under-protected.
    assert "double hot_fraction = 0.15" in gem5_context
    assert "std::atof(e) : 0.15" in sniper_context
    assert "array_bytes = " in graph_ctx
    assert "array_bytes = " in gem5_context
    assert "array_bytes = " in sniper_context
    assert "if (!regions[i].grasp_region) continue;" in gem5_context
    assert "if (!regions[i].grasp_region) continue;" in sniper_context
    assert "policy_ == EvictionPolicy::GRASP" in cache_sim
    assert "mode == ECGMode::DBG_ONLY" in cache_sim
    assert "no valid bit" in cache_sim
    assert "patch_grasp_portability" in compare_script
    assert "popt_pin_toolchain" in compare_script


def test_live_grasp_regions_match_upstream_property_a_scope():
    """All vertex-indexed property arrays in the live bench/src_* sources
    must register with grasp_region=true.

    Prior to commit 65a41df ("Fix multi-property GRASP region
    misclassification across all suites"), pr/pr_spmv/bc had mixed flags
    where only one of several vertex-indexed arrays was marked
    grasp_region=true. The classifier in classifyGRASP() only iterates
    arrays with the flag set, so any vertex-indexed array left as false
    fell into SRRIP's RRPV lane and inverted GRASP's expected behaviour
    (web-Google/BC went from +20 pp regression to +0.02 pp parity after
    the fix). This test pins the multi-property invariant at source
    level: every vertex-indexed property array must opt into the GRASP
    hot-region classifier.
    """
    pr = read("bench/src_sim/pr.cc")
    pr_spmv = read("bench/src_sim/pr_spmv.cc")
    bc = read("bench/src_sim/bc.cc")
    gem5_pr = read("bench/src_gem5/pr.cc")
    gem5_bc = read("bench/src_gem5/bc.cc")
    sniper_pr = read("bench/src_sniper/pr.cc")

    # cache_sim sources: registerPropertyArray(..., grasp_region=true)
    assert "scores_ptr, g.num_nodes(), sizeof(ScoreT), llc_size, -1.0, true" in pr
    assert "contrib_ptr, g.num_nodes(), sizeof(ScoreT), llc_size, -1.0, true" in pr
    assert "scores_ptr, g.num_nodes(), sizeof(ScoreT), llc_size, -1.0, true" in pr_spmv
    assert "contrib_ptr, g.num_nodes(), sizeof(ScoreT), llc_size, -1.0, true" in pr_spmv
    assert "deltas.data(), g.num_nodes(), sizeof(ScoreT), llc_size, -1.0, true" in bc
    assert "path_counts.data(), g.num_nodes(), sizeof(int64_t), llc_size, -1.0, true" in bc
    assert "depths.data(), g.num_nodes(), sizeof(int32_t), llc_size, -1.0, true" in bc
    assert "scores.data(), g.num_nodes(), sizeof(ScoreT), llc_size, -1.0, true" in bc

    # gem5 sources: brace-initialised regions[] must all close with `true`.
    # Look for the per-array signature: sizeof(<T>), true},
    assert "sizeof(ScoreT), true}" in gem5_pr
    assert '"deltas"' in gem5_bc
    assert gem5_bc.count("sizeof(ScoreT), true},") >= 2, gem5_bc.count("sizeof(ScoreT), true},")
    assert "sizeof(int32_t), true}" in gem5_bc
    assert "sizeof(int64_t), true}" in gem5_bc

    # sniper sources: same regions[] pattern.
    assert "sizeof(ScoreT), true}" in sniper_pr

    # And forbid the *pre-fix* state where any of these were marked false.
    # The strings ", false}" or ", false);" must not appear on the
    # registerPropertyArray / Property region lines for vertex-indexed arrays
    # in these files. (Other lines in the source may legitimately use
    # `false` for non-property arguments, so we narrow to the exact pattern.)
    for text, name in [(pr, "src_sim/pr.cc"), (pr_spmv, "src_sim/pr_spmv.cc"),
                       (bc, "src_sim/bc.cc")]:
        assert "llc_size, -1.0, false" not in text, (
            f"{name}: a vertex-indexed property array is still being "
            f"registered with grasp_region=false; multi-property GRASP "
            f"misclassification fix (commit 65a41df) regressed."
        )


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


def test_popt_rereference_encoding_matches_official_artifact_polarity():
    popt_builder = read("bench/include/graphbrew/partition/cagra/popt.h")
    cache_context = read("bench/include/cache_sim/graph_cache_context.h")
    cache_sim = read("bench/include/cache_sim/cache_sim.h")
    gem5_context = read("bench/include/gem5_sim/overlays/mem/cache/replacement_policies/graph_cache_context_gem5.hh")
    sniper_context = read("bench/include/sniper_sim/overlays/common/core/memory_subsystem/cache/graph_cache_context_sniper.cc")

    assert "MSB=0: referenced in this epoch" in popt_builder
    assert "MSB=1: not referenced in this epoch" in popt_builder
    assert "official P-OPT artifact convention" in cache_context
    assert "MSB=0 (bit 7 clear): cache line IS referenced" in cache_sim
    assert "MSB=1 (bit 7 set): cache line is NOT referenced" in cache_sim
    assert "if ((next & MSB) == 0) return 1" in cache_context
    assert "if ((next_entry & OR_MASK) == 0)" in cache_sim
    assert "MSB=0 means referenced in this epoch" in gem5_context
    assert "if ((next_entry & OR_MASK) == 0) return 1" in gem5_context
    assert "if ((next_entry & OR_MASK) == 0) return 1" in sniper_context
    for text in (popt_builder, cache_context, cache_sim, gem5_context, sniper_context):
        assert "inverse bit polarity" not in text