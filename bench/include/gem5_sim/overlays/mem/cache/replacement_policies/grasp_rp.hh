// ============================================================================
// GRASP Replacement Policy for gem5
// ============================================================================
//
// Graph-aware cache Replacement with Software Prefetching (GRASP)
// Faldu et al., HPCA 2020
//
// Extends BRRIP with degree-based insertion and promotion:
//   - High-reuse (hot hubs): insert RRPV=1, hit → RRPV=0
//   - Moderate-reuse:        insert RRPV=M-1, hit → decrement
//   - Low-reuse (cold):      insert RRPV=M, hit → decrement
//
// Eviction: find way with max RRPV, age all if none at max (same as SRRIP).
//
// Requires DBG-reordered graph so hot vertices occupy low addresses, OR
// uses GraphCacheContext for region-based bucket classification.
//
// Reference implementation: bench/include/cache_sim/cache_sim.h
//   GRASPState (lines 218-280), findVictimGRASP (line ~870)
// ============================================================================

#ifndef __MEM_CACHE_REPLACEMENT_POLICIES_GRASP_RP_HH__
#define __MEM_CACHE_REPLACEMENT_POLICIES_GRASP_RP_HH__

#include "mem/cache/replacement_policies/base.hh"
#include "mem/cache/replacement_policies/graph_cache_context_gem5.hh"
#include "params/GraphGraspRP.hh"

#include <cstdint>
#include <memory>

namespace gem5 {
namespace replacement_policy {

class GraphGraspRP : public Base
{
  public:
    PARAMS(GraphGraspRP);

    // Per-cache-line replacement data
    struct GraspReplData : public ReplacementData
    {
        uint8_t rrpv;           // Re-Reference Prediction Value (0 = near, M = distant)
        uint8_t degree_bucket;  // Degree bucket index (0=hub, N-1=cold)
        bool is_property_data;  // Whether this line holds vertex property data

        GraspReplData(uint8_t max_rrpv)
            : rrpv(max_rrpv), degree_bucket(0), is_property_data(false) {}
    };

    GraphGraspRP(const Params &p);
    ~GraphGraspRP() override = default;

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

    // Set the graph cache context (called from Python config)
    void setGraphContext(graph::GraphCacheContext* ctx) { graphCtx = ctx; }

  private:
    // GRASP reuse tier classification
    enum class ReuseTier { HIGH, MODERATE, LOW };

    ReuseTier classifyAddress(uint64_t addr) const;
    uint8_t insertionRRPV(ReuseTier tier) const;
    void promoteOnHit(GraspReplData* data) const;

    // Configuration
    const uint8_t maxRRPV;          // Maximum RRPV value (2^rrpv_bits - 1)
    const uint8_t numBuckets;       // Number of degree buckets
    const double hotFraction;       // Fraction of LLC for hot vertices

    // Graph metadata
    graph::GraphCacheContext* graphCtx = nullptr;

    // Legacy GRASP state (used when no GraphCacheContext)
    uint64_t dataBase = 0;
    uint64_t highReuseBound = 0;
    uint64_t moderateReuseBound = 0;
    uint64_t dataEnd = 0;
};

} // namespace replacement_policy
} // namespace gem5

#endif // __MEM_CACHE_REPLACEMENT_POLICIES_GRASP_RP_HH__
