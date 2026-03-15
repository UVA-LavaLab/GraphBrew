// ============================================================================
// GRASP Replacement Policy for gem5 — Implementation
// ============================================================================
// Reference: Faldu et al., HPCA 2020 + bench/include/cache_sim/cache_sim.h
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
      hotFraction(p.hot_fraction)
{
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
    // Without packet, use generic promotion: decrement RRPV
    if (data->rrpv > 0) data->rrpv--;
}

void
GraphGraspRP::reset(const std::shared_ptr<ReplacementData>& replacement_data,
                    const PacketPtr pkt)
{
    auto data = std::static_pointer_cast<GraspReplData>(replacement_data);

    if (pkt && pkt->req) {
        uint64_t addr = pkt->req->getPaddr();
        ReuseTier tier = classifyAddress(addr);
        data->rrpv = insertionRRPV(tier);
        data->is_property_data = (tier != ReuseTier::LOW);

        // Classify degree bucket if context available
        if (graphCtx) {
            uint32_t bucket = graphCtx->classifyBucket(addr);
            data->degree_bucket = static_cast<uint8_t>(
                std::min(bucket, static_cast<uint32_t>(numBuckets - 1)));
        }
    } else {
        data->rrpv = maxRRPV - 1; // Default: long re-reference
        data->is_property_data = false;
    }
}

void
GraphGraspRP::reset(
    const std::shared_ptr<ReplacementData>& replacement_data) const
{
    auto data = std::static_pointer_cast<GraspReplData>(replacement_data);
    data->rrpv = maxRRPV - 1; // SRRIP default insertion
}

ReplaceableEntry*
GraphGraspRP::getVictim(const ReplacementCandidates& candidates) const
{
    assert(candidates.size() > 0);

    // Find line with max RRPV; if none at maxRRPV, age all until found
    // (Matching standalone cache_sim findVictimGRASP)
    while (true) {
        ReplaceableEntry* victim = nullptr;
        uint8_t highestRRPV = 0;

        for (const auto& candidate : candidates) {
            auto data = std::static_pointer_cast<GraspReplData>(
                candidate->replacementData);
            if (data->rrpv >= maxRRPV) {
                return candidate; // Found victim at max RRPV
            }
            if (data->rrpv > highestRRPV) {
                highestRRPV = data->rrpv;
                victim = candidate;
            }
        }

        // Age all lines: increment RRPV by 1
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
// Private helpers
// ============================================================================

GraphGraspRP::ReuseTier
GraphGraspRP::classifyAddress(uint64_t addr) const
{
    // Prefer classifyGRASP (faithful 3-tier per Faldu et al.)
    if (graphCtx) {
        uint32_t tier = graphCtx->classifyGRASP(addr, 8 * 1024 * 1024); // 8MB LLC default
        if (tier == 1) return ReuseTier::HIGH;
        if (tier == 2) return ReuseTier::MODERATE;
        return ReuseTier::LOW;
    }

    // Legacy GRASP state (address-range based, requires DBG ordering)
    if (dataBase == 0 && dataEnd == 0) return ReuseTier::LOW;
    if (addr >= dataBase && addr < highReuseBound) return ReuseTier::HIGH;
    if (addr >= highReuseBound && addr < moderateReuseBound) return ReuseTier::MODERATE;
    return ReuseTier::LOW;
}

uint8_t
GraphGraspRP::insertionRRPV(ReuseTier tier) const
{
    // Matching Faldu et al. HPCA 2020: P_RRIP=1, I_RRIP=6, M_RRIP=7
    switch (tier) {
        case ReuseTier::HIGH:     return 1;              // P_RRIP
        case ReuseTier::MODERATE: return maxRRPV - 1;    // I_RRIP (6)
        case ReuseTier::LOW:      return maxRRPV;        // M_RRIP (7)
    }
    return maxRRPV;
}

void
GraphGraspRP::promoteOnHit(GraspReplData* data) const
{
    // GRASP hit promotion (Faldu et al. HPCA 2020):
    //   Hot region (bucket 0): RRPV → 0 (aggressive reset)
    //   Others: decrement by 1
    if (data->degree_bucket == 0) {
        data->rrpv = 0;
    } else if (data->rrpv > 0) {
        data->rrpv--;
    }
}

} // namespace replacement_policy
} // namespace gem5
