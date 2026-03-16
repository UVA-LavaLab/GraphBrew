// ============================================================================
// GRASP Replacement Policy for gem5 — Implementation
// ============================================================================
// Faithful implementation matching bench/include/cache_sim/cache_sim.h.
// Loads property region metadata from sideband JSON (written by benchmark).
// ============================================================================

#include "mem/cache/replacement_policies/grasp_rp.hh"

#include <algorithm>
#include <cassert>

namespace gem5 {
namespace replacement_policy {

GraphGraspRP::GraphGraspRP(const Params &p)
    : Base(p),
      maxRRPV(p.max_rrpv),
      numBuckets(p.num_buckets),
      hotFraction(p.hot_fraction),
      llcSize(p.llc_size_bytes),
      sidebandPath(p.sideband_path)
{
}

void
GraphGraspRP::tryLoadContext() const
{
    if (loadAttempted) return;
    loadAttempted = true;
    if (ctx.loadFromSideband(sidebandPath)) {
        ctx.loaded = true;
    }
}

void
GraphGraspRP::invalidate(
    const std::shared_ptr<ReplacementData>& replacement_data)
{
    auto data = std::static_pointer_cast<GraspReplData>(replacement_data);
    data->rrpv = maxRRPV;
    data->is_property_data = false;
    data->degree_bucket = numBuckets;
}

void
GraphGraspRP::touch(const std::shared_ptr<ReplacementData>& replacement_data,
                    const PacketPtr pkt)
{
    auto data = std::static_pointer_cast<GraspReplData>(replacement_data);
    promoteOnHit(data.get());
}

void
GraphGraspRP::touch(
    const std::shared_ptr<ReplacementData>& replacement_data) const
{
    auto data = std::static_pointer_cast<GraspReplData>(replacement_data);
    if (data->rrpv > 0) data->rrpv--;
}

void
GraphGraspRP::reset(const std::shared_ptr<ReplacementData>& replacement_data,
                    const PacketPtr pkt)
{
    auto data = std::static_pointer_cast<GraspReplData>(replacement_data);

    // Try loading context on first access
    tryLoadContext();

    if (pkt && pkt->req) {
        uint64_t addr = pkt->req->getPaddr();

        // Only apply GRASP classification to known property regions.
        // Non-property data (instructions, CSR edges, stack) gets SRRIP
        // default RRPV=2. This matches standalone cache_sim where GRASP
        // only sees property array accesses via SIM_CACHE_READ.
        if (ctx.loaded && ctx.isPropertyData(addr)) {
            ReuseTier tier = classifyAddress(addr);
            data->rrpv = insertionRRPV(tier);
            data->is_property_data = true;
        } else {
            data->rrpv = 2;  // SRRIP default: long re-reference (M-1 for 3-bit)
            data->is_property_data = false;
        }
    } else {
        data->rrpv = maxRRPV - 1;
        data->is_property_data = false;
    }
}

void
GraphGraspRP::reset(
    const std::shared_ptr<ReplacementData>& replacement_data) const
{
    auto data = std::static_pointer_cast<GraspReplData>(replacement_data);
    data->rrpv = maxRRPV - 1;
}

ReplaceableEntry*
GraphGraspRP::getVictim(const ReplacementCandidates& candidates) const
{
    assert(candidates.size() > 0);

    // SRRIP eviction: find way with max RRPV, age all if none found
    while (true) {
        for (const auto& candidate : candidates) {
            auto data = std::static_pointer_cast<GraspReplData>(
                candidate->replacementData);
            if (data->rrpv >= maxRRPV)
                return candidate;
        }
        for (const auto& candidate : candidates) {
            auto data = std::static_pointer_cast<GraspReplData>(
                candidate->replacementData);
            if (data->rrpv < maxRRPV) data->rrpv++;
        }
    }
}

std::shared_ptr<ReplacementData>
GraphGraspRP::instantiateEntry()
{
    return std::make_shared<GraspReplData>(maxRRPV);
}

// ============================================================================
// Private: faithful GRASP 3-tier classification
// ============================================================================

GraphGraspRP::ReuseTier
GraphGraspRP::classifyAddress(uint64_t addr) const
{
    if (ctx.loaded) {
        uint32_t tier = ctx.classifyGRASP(addr, llcSize);
        if (tier == 1) return ReuseTier::HIGH;
        if (tier == 2) return ReuseTier::MODERATE;
        return ReuseTier::LOW;
    }
    return ReuseTier::LOW;
}

uint8_t
GraphGraspRP::insertionRRPV(ReuseTier tier) const
{
    switch (tier) {
        case ReuseTier::HIGH:     return 1;
        case ReuseTier::MODERATE: return maxRRPV - 1;
        case ReuseTier::LOW:      return maxRRPV;
    }
    return maxRRPV;
}

void
GraphGraspRP::promoteOnHit(GraspReplData* data) const
{
    if (data->is_property_data && data->rrpv <= 1) {
        data->rrpv = 0;
    } else if (data->rrpv > 0) {
        data->rrpv--;
    }
}

} // namespace replacement_policy
} // namespace gem5
