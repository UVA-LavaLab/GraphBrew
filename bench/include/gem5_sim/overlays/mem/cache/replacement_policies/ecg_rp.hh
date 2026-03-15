// ============================================================================
// ECG Replacement Policy for gem5
// ============================================================================
//
// Expressing Locality and Prefetching for Optimal Caching in Graph Structures
// Mughrabi et al., GrAPL @ IPDPS 2026
//
// 3-level layered eviction with mode-dependent tiebreaker priority:
//
//   Level 1 (all modes): SRRIP aging — find max RRPV, age until found
//   Level 2/3 depend on ECGMode:
//     DBG_PRIMARY:  L2=DBG tier (coldest vertex), L3=dynamic P-OPT
//     POPT_PRIMARY: L2=dynamic P-OPT (furthest future), L3=DBG tier
//     DBG_ONLY:     L2=DBG tier only (fast path, no P-OPT)
//
// Key design:
//   - RRPV set at insert from DBG tier (structural, unchanging)
//   - ecg_dbg_tier stored per-line (constant metadata)
//   - P-OPT consulted dynamically at eviction (avoids stale snapshots)
//   - Per-access hints via custom ECG instruction → CSR → cache controller
//
// Reference implementation: bench/include/cache_sim/cache_sim.h
//   findVictimECG (lines 963-1052)
// ============================================================================

#ifndef __MEM_CACHE_REPLACEMENT_POLICIES_ECG_RP_HH__
#define __MEM_CACHE_REPLACEMENT_POLICIES_ECG_RP_HH__

#include "mem/cache/replacement_policies/base.hh"
#include "mem/cache/replacement_policies/graph_cache_context_gem5.hh"
#include "params/GraphEcgRP.hh"

#include <cstdint>
#include <memory>

namespace gem5 {
namespace replacement_policy {

class GraphEcgRP : public Base
{
  public:
    PARAMS(GraphEcgRP);

    // Per-cache-line replacement data
    struct EcgReplData : public ReplacementData
    {
        uint8_t rrpv;             // RRPV (configurable width: 3-8 bits)
        uint8_t ecg_dbg_tier;     // Stored DBG degree tier (structural, constant)
        bool is_property_data;    // Whether this line contains vertex property data
        uint64_t line_addr;       // Cache-line-aligned address (for P-OPT lookup)

        EcgReplData(uint8_t max_rrpv)
            : rrpv(max_rrpv), ecg_dbg_tier(0),
              is_property_data(false), line_addr(0) {}
    };

    GraphEcgRP(const Params &p);
    ~GraphEcgRP() override = default;

    void invalidate(const std::shared_ptr<ReplacementData>& replacement_data)
        override;
    void touch(const std::shared_ptr<ReplacementData>& replacement_data,
               const PacketPtr pkt) override;
    void touch(const std::shared_ptr<ReplacementData>& replacement_data)
        const override;
    void reset(const std::shared_ptr<ReplacementData>& replacement_data,
               const PacketPtr pkt) override;
    void reset(const std::shared_ptr<ReplacementData>& replacement_data)
        const override;

    ReplaceableEntry* getVictim(
        const ReplacementCandidates& candidates) const override;

    std::shared_ptr<ReplacementData> instantiateEntry() override;

    void setGraphContext(graph::GraphCacheContext* ctx) { graphCtx = ctx; }

    // Update per-access mask hint (from custom instruction / CSR)
    void setCurrentMask(uint8_t mask) {
        if (graphCtx) graphCtx->current_mask = mask;
    }

    void setCurrentVertex(uint32_t v) {
        if (graphCtx) graphCtx->current_dst_vertex = v;
    }

  private:
    const uint8_t rrpvMax;          // Maximum RRPV value
    const uint8_t numBuckets;       // Number of degree buckets
    const graph::ECGMode ecgMode;   // Eviction tiebreaker mode

    graph::GraphCacheContext* graphCtx = nullptr;
};

} // namespace replacement_policy
} // namespace gem5

#endif // __MEM_CACHE_REPLACEMENT_POLICIES_ECG_RP_HH__
