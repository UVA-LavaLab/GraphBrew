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
    // Prefer unified GraphCacheContext over legacy GRASP state
    if (graphCtx) {
        uint32_t bucket = graphCtx->classifyBucket(addr);
        if (bucket >= graphCtx->mask_config.num_buckets) return ReuseTier::LOW;
        // Map bucket index to 3-tier model:
        //   Bucket 0 (highest degree) → HIGH
        //   Buckets 1..(N/3) → MODERATE
        //   Rest → LOW
        uint32_t third = std::max(1u, static_cast<uint32_t>(graphCtx->mask_config.num_buckets / 3));
        if (bucket == 0) return ReuseTier::HIGH;
        if (bucket < third) return ReuseTier::MODERATE;
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
    // Matching GRASP paper: P_RRIP=1 (high), I_RRIP=M-1 (moderate), M_RRIP=M (low)
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
    // Matching standalone cache_sim GRASP hit logic:
    //   High-reuse (bucket 0): RRPV → 0
    //   Others: decrement by 1
    if (data->degree_bucket == 0 || data->rrpv <= 1) {
        data->rrpv = 0;
    } else if (data->rrpv > 0) {
        data->rrpv--;
    }
}

} // namespace replacement_policy
} // namespace gem5
