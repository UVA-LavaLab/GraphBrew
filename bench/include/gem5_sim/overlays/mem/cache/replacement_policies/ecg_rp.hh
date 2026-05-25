// ============================================================================
// ECG Replacement Policy for gem5
// ============================================================================
// Faithful 3-level layered eviction matching cache_sim.h findVictimECG.
// Loads context from sideband JSON written by benchmark at runtime.
// ============================================================================

#ifndef __MEM_CACHE_REPLACEMENT_POLICIES_ECG_RP_HH__
#define __MEM_CACHE_REPLACEMENT_POLICIES_ECG_RP_HH__

#include "mem/cache/replacement_policies/base.hh"
#include "mem/cache/replacement_policies/graph_cache_context_gem5.hh"
#include "params/GraphEcgRP.hh"

#include <cstdint>
#include <memory>
#include <string>

namespace gem5 {
namespace replacement_policy {

class GraphEcgRP : public Base
{
  public:
    PARAMS(GraphEcgRP);

    struct EcgReplData : public ReplacementData
    {
        uint8_t rrpv;
        uint8_t ecg_dbg_tier;
        uint8_t ecg_popt_hint;
        bool is_property_data;
        uint64_t line_addr;

        EcgReplData(uint8_t max_rrpv)
            : rrpv(max_rrpv), ecg_dbg_tier(0), ecg_popt_hint(0),
              is_property_data(false), line_addr(0) {}
    };

    GraphEcgRP(const Params &p);
    ~GraphEcgRP() override = default;

    void invalidate(const std::shared_ptr<ReplacementData>& replacement_data) override;
    void touch(const std::shared_ptr<ReplacementData>& replacement_data,
               const PacketPtr pkt) override;
    void touch(const std::shared_ptr<ReplacementData>& replacement_data) const override;
    void reset(const std::shared_ptr<ReplacementData>& replacement_data,
               const PacketPtr pkt) override;
    void reset(const std::shared_ptr<ReplacementData>& replacement_data) const override;
    ReplaceableEntry* getVictim(const ReplacementCandidates& candidates) const override;
    std::shared_ptr<ReplacementData> instantiateEntry() override;

  private:
    void tryLoadContext() const;

    const uint8_t rrpvMax;
    const uint8_t numBuckets;
    const graph::ECGMode ecgMode;
    const uint64_t llcSize;
    const std::string sidebandPath;
    const std::string poptMatrixPath;

    mutable graph::GraphCacheContext ctx;
    mutable bool loadAttempted = false;
    mutable uint64_t loadAttemptCount = 0;
};

} // namespace replacement_policy
} // namespace gem5

#endif // __MEM_CACHE_REPLACEMENT_POLICIES_ECG_RP_HH__
