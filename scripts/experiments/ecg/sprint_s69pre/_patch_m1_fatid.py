#!/usr/bin/env python3
"""
M1 helper: gem5 RISCV fat-id wiring — end-to-end.

Sprint S69-pre M1: make gem5 RISCV ecg.extract carry the full mode-6
fat mask (24+2+7+31 bits) instead of just the prefetch_target.
Replacement policy ECG_RP gains an ISA-metadata consumer that
prefers ISA-delivered DBG/POPT over sideband-derived values.

Files edited (all idempotent via S69PRE-M1-MASK markers):
  1. bench/include/gem5_sim/overlays/arch/riscv/isa/decoder_ecg_extract.isa
  2. bench/include/gem5_sim/overlays/mem/cache/replacement_policies/graph_cache_context_gem5.hh
  3. bench/include/gem5_sim/overlays/mem/cache/replacement_policies/ecg_rp.cc
  4. bench/include/gem5_sim/gem5_harness.h
  5. bench/src_gem5/pr.cc

Re-runs detect the marker and skip.
"""
import sys, os, re

REPO = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))))

MARKER = "S69PRE-M1-MASK"

# ============================================================================
# Edit 1: decoder_ecg_extract.isa — use mode-6 mask layout
# ============================================================================
def edit_decoder():
    path = os.path.join(REPO, "bench/include/gem5_sim/overlays/arch/riscv/isa/decoder_ecg_extract.isa")
    src = open(path).read()
    if MARKER in src:
        print(f"[m1] {path}: already patched, skip")
        return False

    # Replace the body of ecg_extract
    OLD = """                    0x00: ecg_extract({{
                        uint64_t fat_id = Rs1;
                        uint32_t real_vertex = static_cast<uint32_t>(fat_id & 0xffffffffULL);
                        uint8_t dbg_hint = static_cast<uint8_t>((fat_id >> 32) & 0xffULL);
                        uint8_t popt_hint = static_cast<uint8_t>((fat_id >> 40) & 0xffULL);
                        uint16_t pfx_hint = static_cast<uint16_t>((fat_id >> 48) & 0xffffULL);
                        uint32_t pfx_target = pfx_hint ? static_cast<uint32_t>(pfx_hint) : real_vertex;
                        gem5::replacement_policy::graph::setDecodedEcgExtractHint(
                            real_vertex, dbg_hint, popt_hint, pfx_hint);
                        gem5::replacement_policy::graph::setPrefetchTargetHint(pfx_target);
                        Rd = rvZext(real_vertex);
                    }});"""
    NEW = """                    0x00: ecg_extract({{
                        // S69PRE-M1-MASK: use mode-6 mask layout (matches
                        // bench/include/ecg_mode6_builder.h packMask):
                        //   [0:24]   dest_id (demand load vertex)
                        //   [24:26]  DBG tier (0..3)
                        //   [26:33]  POPT quant (0..127)
                        //   [33:64]  prefetch target (0 = no prefetch)
                        uint64_t fat_mask = Rs1;
                        uint32_t dest_id     = static_cast<uint32_t>((fat_mask >>  0) & 0xFFFFFFULL);
                        uint8_t  dbg_tier    = static_cast<uint8_t> ((fat_mask >> 24) & 0x3ULL);
                        uint8_t  popt_quant  = static_cast<uint8_t> ((fat_mask >> 26) & 0x7FULL);
                        uint32_t pfx_target  = static_cast<uint32_t>((fat_mask >> 33) & 0x7FFFFFFFULL);
                        // Populate per-vertex metadata table so ECG_RP can
                        // consume ISA-delivered DBG/POPT for paper-faithful
                        // CHARGED=0 replacement.
                        gem5::replacement_policy::graph::storeEcgMetadataByVertex(
                            dest_id, dbg_tier, popt_quant);
                        // Legacy storage (single-slot mailbox; kept for
                        // back-compat with non-ECG_RP consumers).
                        gem5::replacement_policy::graph::setDecodedEcgExtractHint(
                            dest_id, dbg_tier, popt_quant, 0);
                        // Queue the prefetch hint (or fall back to demand
                        // vertex if no prefetch encoded).
                        if (pfx_target != 0) {
                            gem5::replacement_policy::graph::setPrefetchTargetHint(pfx_target);
                        } else {
                            gem5::replacement_policy::graph::setPrefetchTargetHint(dest_id);
                        }
                        Rd = rvZext(dest_id);
                    }});"""
    if OLD not in src:
        print(f"[m1] ERROR: decoder body did not match expected pattern", file=sys.stderr)
        return None
    src = src.replace(OLD, NEW, 1)
    open(path, "w").write(src)
    print(f"[m1] patched {path}")
    return True

# ============================================================================
# Edit 2: graph_cache_context_gem5.hh — add per-vertex metadata table
# ============================================================================
def edit_context_gem5():
    path = os.path.join(REPO, "bench/include/gem5_sim/overlays/mem/cache/replacement_policies/graph_cache_context_gem5.hh")
    src = open(path).read()
    if MARKER in src:
        print(f"[m1] {path}: already patched, skip")
        return False

    # Insert the table + helpers after setDecodedEcgExtractHint definition
    # (line ~180 area in current file). Use stable anchor.
    anchor = """inline void setDecodedEcgExtractHint(uint32_t real_vertex,
                                     uint8_t dbg_hint,
                                     uint8_t popt_hint,
                                     uint16_t pfx_hint) {
    uint32_t metadata = static_cast<uint32_t>(dbg_hint)
        | (static_cast<uint32_t>(popt_hint) << 8)
        | (static_cast<uint32_t>(pfx_hint) << 16);
    decodedEcgRealVertexStorage().store(real_vertex, std::memory_order_release);
    decodedEcgMetadataStorage().store(metadata, std::memory_order_release);
    decodedEcgHintValidStorage().store(true, std::memory_order_release);
}"""

    addition = """

// === """ + MARKER + """: Per-vertex ECG metadata table ===
//
// The legacy setDecodedEcgExtractHint above is a single-slot mailbox.
// For paper-faithful CHARGED=0 the replacement policy needs to look
// up DBG/POPT metadata BY VERTEX when a cache miss for property[v]
// is being resolved. A direct-mapped 4K-entry table provides
// constant-time lookup without dynamic allocation. The kernel emits
// hints in spatial order (PR pull: for u, for v in in_neigh(u))
// matching the cache miss pattern, so direct-mapped collisions are
// rare in practice.

inline constexpr std::size_t kEcgMetadataTableSize = 4096;

struct EcgMetadataEntry {
    std::atomic<uint32_t> vertex{UINT32_MAX};  // sentinel = invalid
    std::atomic<uint8_t>  dbg_tier{0};
    std::atomic<uint8_t>  popt_quant{0};
};

inline std::array<EcgMetadataEntry, kEcgMetadataTableSize>& ecgMetadataTable() {
    static std::array<EcgMetadataEntry, kEcgMetadataTableSize> table;
    return table;
}

inline void storeEcgMetadataByVertex(uint32_t vertex,
                                     uint8_t dbg_tier,
                                     uint8_t popt_quant) {
    auto& entry = ecgMetadataTable()[vertex % kEcgMetadataTableSize];
    entry.dbg_tier.store(dbg_tier, std::memory_order_relaxed);
    entry.popt_quant.store(popt_quant, std::memory_order_relaxed);
    // Store vertex LAST so a concurrent reader sees a coherent
    // (vertex, dbg, popt) triple — happens-before via the release on
    // vertex.
    entry.vertex.store(vertex, std::memory_order_release);
}

inline bool lookupEcgMetadataByVertex(uint32_t vertex,
                                      uint8_t& dbg_tier_out,
                                      uint8_t& popt_quant_out) {
    auto& entry = ecgMetadataTable()[vertex % kEcgMetadataTableSize];
    if (entry.vertex.load(std::memory_order_acquire) != vertex) {
        return false;  // miss (sentinel, evicted, or different vertex hashed to same slot)
    }
    dbg_tier_out  = entry.dbg_tier.load(std::memory_order_relaxed);
    popt_quant_out = entry.popt_quant.load(std::memory_order_relaxed);
    return true;
}

// Address-to-vertex helper for ECG_RP. Property region base + elem_size
// come from the sideband JSON. Returns UINT32_MAX if addr is not in any
// known property region.
inline uint32_t addressToVertex(uint64_t addr,
                                uint64_t property_base,
                                uint64_t property_end,
                                uint32_t elem_size) {
    if (addr < property_base || addr >= property_end || elem_size == 0) {
        return UINT32_MAX;
    }
    return static_cast<uint32_t>((addr - property_base) / elem_size);
}
"""
    if anchor not in src:
        print(f"[m1] ERROR: setDecodedEcgExtractHint anchor not found", file=sys.stderr)
        return None
    src = src.replace(anchor, anchor + addition, 1)
    open(path, "w").write(src)
    print(f"[m1] patched {path}")
    return True

# ============================================================================
# Edit 3: ecg_rp.cc — consume ISA-delivered metadata when available
# ============================================================================
def edit_ecg_rp():
    path = os.path.join(REPO, "bench/include/gem5_sim/overlays/mem/cache/replacement_policies/ecg_rp.cc")
    src = open(path).read()
    if MARKER in src:
        print(f"[m1] {path}: already patched, skip")
        return False

    # Insert after sideband-derived dbg/popt are computed, BEFORE mode-specific
    # rrpv assignment. The hook overrides ecg_dbg_tier and ecg_popt_hint with
    # ISA-delivered values when available.
    anchor = """        data->ecg_popt_hint = 0;
        if (data->is_property_data && ctx.rereference.enabled) {
            uint32_t dist = ctx.findNextRef(data->line_addr);
            data->ecg_popt_hint = static_cast<uint8_t>(
                std::min(dist, uint32_t(127)) >> 3);
        }"""

    addition = """

        // === """ + MARKER + """: prefer ISA-delivered metadata over sideband ===
        // When the kernel has emitted an ecg.extract opcode for the vertex
        // owning this cache line, the per-vertex metadata table holds the
        // CHARGED=0 paper-faithful DBG tier + POPT quant. Prefer those over
        // the sideband-JSON-derived values. Falls back to sideband if the
        // table has no entry for this vertex.
        if (data->is_property_data && ctx.loaded && ctx.num_regions > 0) {
            const auto& region = ctx.regions[0];
            uint32_t vertex = graph::addressToVertex(
                data->line_addr,
                region.base_address, region.upper_bound,
                region.elem_size);
            if (vertex != UINT32_MAX) {
                uint8_t isa_dbg = 0, isa_popt = 0;
                if (graph::lookupEcgMetadataByVertex(vertex, isa_dbg, isa_popt)) {
                    // Use ISA-delivered metadata directly.
                    data->ecg_dbg_tier = isa_dbg;
                    // POPT quant is 7 bits; ECG_RP stores as 8-bit; range OK.
                    data->ecg_popt_hint = isa_popt;
                }
            }
        }"""
    if anchor not in src:
        print(f"[m1] ERROR: ecg_rp.cc anchor not found", file=sys.stderr)
        return None
    src = src.replace(anchor, anchor + addition, 1)
    open(path, "w").write(src)
    print(f"[m1] patched {path}")
    return True

# ============================================================================
# Edit 4: gem5_harness.h — add GEM5_ECG_EXTRACT_MASK macro
# ============================================================================
def edit_harness():
    path = os.path.join(REPO, "bench/include/gem5_sim/gem5_harness.h")
    src = open(path).read()
    if MARKER in src:
        print(f"[m1] {path}: already patched, skip")
        return False

    # Insert a new function + macro RIGHT AFTER gem5_ecg_pfx_target_instruction
    anchor = """inline uint32_t gem5_ecg_pfx_target_instruction(uint32_t target_vertex) {
#if defined(__riscv)
    return gem5_ecg_extract_target_instruction(target_vertex);
#elif defined(__x86_64__)
    gem5_x86_work_begin_instruction(GEM5_WORK_ECG_PFX_TARGET, static_cast<uint64_t>(target_vertex));
    return target_vertex;
#else
    return target_vertex;
#endif
}"""

    addition = """

// === """ + MARKER + """: full mode-6 mask emission via ecg.extract ===
//
// The S69-pre M1 wiring extends RISCV ecg.extract delivery from
// just the prefetch target to the full 64-bit per-edge mode-6 mask
// (dest + DBG + POPT + PFX). The decoder unpacks all 4 fields and
// populates the per-vertex ECG metadata table; ECG_RP consumes that
// table during replacement decisions.
//
// On X86 the work_begin path can only carry one uint64_t threadid
// argument, which happens to be exactly the 64-bit mask. The
// pseudo_inst handler treats the threadid as a packed mask when the
// new work_id is used. (Backward-compat: GEM5_WORK_ECG_PFX_TARGET
// continues to deliver a bare vertex via setPrefetchTargetHint.)

inline uint32_t gem5_ecg_extract_mask_instruction(uint64_t fat_mask) {
#if defined(__riscv)
    uint64_t real_vertex = 0;
    asm volatile (".insn r 0x0b, 0x0, 0x00, %0, %1, x0"
                  : "=r"(real_vertex)
                  : "r"(fat_mask)
                  : "memory");
    return static_cast<uint32_t>(real_vertex);
#elif defined(__x86_64__)
    // X86 fallback: deliver just the prefetch target (high 31 bits of mask).
    // ECG_RP metadata channel is RISCV-only on X86 today.
    uint32_t pfx_target = static_cast<uint32_t>((fat_mask >> 33) & 0x7FFFFFFFULL);
    if (pfx_target == 0) {
        pfx_target = static_cast<uint32_t>(fat_mask & 0xFFFFFFULL);  // dest
    }
    gem5_x86_work_begin_instruction(GEM5_WORK_ECG_PFX_TARGET,
                                    static_cast<uint64_t>(pfx_target));
    return pfx_target;
#else
    return static_cast<uint32_t>(fat_mask & 0xFFFFFFULL);  // dest
#endif
}

// GEM5_ECG_EXTRACT_MASK(mask_u64): emit the full mode-6 mask via the
// RISCV ecg.extract opcode (or the X86 fallback). Bypasses the dedup
// filter — caller is responsible for not over-emitting.
#define GEM5_ECG_EXTRACT_MASK(mask_u64) \\
    do { \\
        if (gem5_ecg_pfx_hints_enabled() && gem5_ecg_extract_enabled()) { \\
            (void)gem5_ecg_extract_mask_instruction(static_cast<uint64_t>(mask_u64)); \\
        } \\
    } while (0)
"""
    if anchor not in src:
        print(f"[m1] ERROR: gem5_ecg_pfx_target_instruction anchor not found", file=sys.stderr)
        return None
    src = src.replace(anchor, anchor + addition, 1)
    open(path, "w").write(src)
    print(f"[m1] patched {path}")
    return True

# ============================================================================
# Edit 5: bench/src_gem5/pr.cc — use full mask in mode-6 path
# ============================================================================
def edit_pr_kernel():
    path = os.path.join(REPO, "bench/src_gem5/pr.cc")
    src = open(path).read()
    if MARKER in src:
        print(f"[m1] {path}: already patched, skip")
        return False

    # Add a new conditional that emits the full mask via GEM5_ECG_EXTRACT_MASK
    # immediately AFTER the existing GEM5_ECG_PFX_TARGET line, gated on a
    # new env var so the legacy path stays the default.
    anchor = """                        if (!in_window) {
                            GEM5_ECG_PFX_TARGET(prefetch_target);
                            pfx_window[pfx_window_pos % PREFETCH_WINDOW] = prefetch_target;
                            pfx_window_pos++;
                        }"""

    new_block = """                        if (!in_window) {
                            // """ + MARKER + """: emit FULL mode-6 mask via ecg.extract
                            // when ISA-delivered metadata channel is enabled.
                            // Else fall back to the legacy prefetch-target-only path.
                            if (gem5_ecg_extract_enabled()) {
                                GEM5_ECG_EXTRACT_MASK(mask);
                            } else {
                                GEM5_ECG_PFX_TARGET(prefetch_target);
                            }
                            pfx_window[pfx_window_pos % PREFETCH_WINDOW] = prefetch_target;
                            pfx_window_pos++;
                        }"""
    if anchor not in src:
        print(f"[m1] ERROR: pr.cc anchor not found", file=sys.stderr)
        return None
    src = src.replace(anchor, new_block, 1)
    open(path, "w").write(src)
    print(f"[m1] patched {path}")
    return True

# ============================================================================
# Main
# ============================================================================
results = {}
for name, fn in [
    ("decoder", edit_decoder),
    ("context_gem5", edit_context_gem5),
    ("ecg_rp", edit_ecg_rp),
    ("harness", edit_harness),
    ("pr_kernel", edit_pr_kernel),
]:
    r = fn()
    if r is None:
        print(f"[m1] FATAL: {name} edit failed", file=sys.stderr)
        sys.exit(2)
    results[name] = "patched" if r else "skip_idempotent"

print()
for k, v in results.items():
    print(f"[m1] {k}: {v}")
print(f"[m1] all 5 edits complete. Rebuild RISCV gem5 next.")
