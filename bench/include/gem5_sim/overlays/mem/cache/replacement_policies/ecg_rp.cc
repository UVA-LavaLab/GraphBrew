// ============================================================================
// ECG Replacement Policy for gem5 — Implementation
// ============================================================================
// Reference: Mughrabi et al., GrAPL 2026 + bench/include/cache_sim/cache_sim.h
//
// Implements the 3-level layered eviction algorithm exactly matching the
// standalone cache_sim implementation (findVictimECG, lines 963-1052).
// ============================================================================

#include "mem/cache/replacement_policies/ecg_rp.hh"

#include <algorithm>
#include <cassert>
#include <vector>

namespace gem5 {
namespace replacement_policy {

GraphEcgRP::GraphEcgRP(const Params &p)
    : Base(p),
      rrpvMax(p.rrpv_max),
      numBuckets(p.num_buckets),
      ecgMode(graph::stringToECGMode(p.ecg_mode))
{
}

void
GraphEcgRP::invalidate(
    const std::shared_ptr<ReplacementData>& replacement_data)
{
    auto data = std::static_pointer_cast<EcgReplData>(replacement_data);
    data->rrpv = rrpvMax;
    data->ecg_dbg_tier = 0;
    data->is_property_data = false;
    data->line_addr = 0;
}

void
GraphEcgRP::touch(
    const std::shared_ptr<ReplacementData>& replacement_data,
    const PacketPtr pkt)
{
    auto data = std::static_pointer_cast<EcgReplData>(replacement_data);

    // DBG-aware hit promotion (matching standalone cache_sim):
    //   Bucket 0 (hubs) → RRPV=0
    //   Others → decrement by 1
    if (data->ecg_dbg_tier == 0) {
        data->rrpv = 0;
    } else if (data->rrpv > 0) {
        data->rrpv--;
    }
}

void
GraphEcgRP::touch(
    const std::shared_ptr<ReplacementData>& replacement_data) const
{
    auto data = std::static_pointer_cast<EcgReplData>(replacement_data);
    if (data->rrpv > 0) data->rrpv--;
}

void
GraphEcgRP::reset(
    const std::shared_ptr<ReplacementData>& replacement_data,
    const PacketPtr pkt)
{
    auto data = std::static_pointer_cast<EcgReplData>(replacement_data);

    if (pkt && pkt->req) {
        uint64_t addr = pkt->req->getPaddr();
        data->line_addr = addr & ~uint64_t(63);

        if (graphCtx) {
            data->is_property_data = graphCtx->isPropertyData(data->line_addr);

            // Classify degree bucket
            uint32_t bucket = graphCtx->classifyBucket(data->line_addr);
            if (bucket < numBuckets) {
                data->ecg_dbg_tier = static_cast<uint8_t>(bucket);
                // RRPV from DBG tier (structural priority, set at insert)
                data->rrpv = graphCtx->mask_config.dbgTierToRRPV(
                    data->ecg_dbg_tier);
            } else {
                data->ecg_dbg_tier = numBuckets - 1; // Unknown → cold
                data->rrpv = rrpvMax;
            }

            // If per-access mask is available (from custom instruction),
            // override with mask-derived tier
            if (graphCtx->current_mask != 0 &&
                graphCtx->mask_config.enabled) {
                uint8_t mask_dbg = graphCtx->mask_config.decodeDBG(
                    graphCtx->current_mask);
                data->ecg_dbg_tier = mask_dbg;
                data->rrpv = graphCtx->mask_config.dbgTierToRRPV(mask_dbg);
            }
        } else {
            data->ecg_dbg_tier = numBuckets - 1;
            data->rrpv = rrpvMax - 1;
            data->is_property_data = false;
        }
    } else {
        data->ecg_dbg_tier = numBuckets - 1;
        data->rrpv = rrpvMax - 1;
        data->is_property_data = false;
    }
}

void
GraphEcgRP::reset(
    const std::shared_ptr<ReplacementData>& replacement_data) const
{
    auto data = std::static_pointer_cast<EcgReplData>(replacement_data);
    data->rrpv = rrpvMax - 1;
    data->ecg_dbg_tier = numBuckets - 1;
}

ReplaceableEntry*
GraphEcgRP::getVictim(const ReplacementCandidates& candidates) const
{
    assert(candidates.size() > 0);

    // ── Level 1: SRRIP aging — find lines at max RRPV ──
    // Age all lines until at least one reaches rrpvMax
    while (true) {
        bool found = false;
        for (const auto& c : candidates) {
            auto d = std::static_pointer_cast<EcgReplData>(c->replacementData);
            if (d->rrpv >= rrpvMax) { found = true; break; }
        }
        if (found) break;
        for (const auto& c : candidates) {
            auto d = std::static_pointer_cast<EcgReplData>(c->replacementData);
            if (d->rrpv < rrpvMax) d->rrpv++;
        }
    }

    // Collect candidates at max RRPV
    std::vector<ReplaceableEntry*> maxRrpvCandidates;
    for (const auto& c : candidates) {
        auto d = std::static_pointer_cast<EcgReplData>(c->replacementData);
        if (d->rrpv >= rrpvMax) maxRrpvCandidates.push_back(c);
    }
    if (maxRrpvCandidates.size() == 1) return maxRrpvCandidates[0];

    // ── Level 2/3 tiebreakers (mode-dependent) ──

    if (ecgMode == graph::ECGMode::DBG_PRIMARY ||
        ecgMode == graph::ECGMode::DBG_ONLY) {
        // ── L2: DBG tier tiebreak (evict highest tier = coldest) ──
        uint8_t maxDBG = 0;
        for (const auto& c : maxRrpvCandidates) {
            auto d = std::static_pointer_cast<EcgReplData>(c->replacementData);
            if (d->ecg_dbg_tier > maxDBG) maxDBG = d->ecg_dbg_tier;
        }

        std::vector<ReplaceableEntry*> dbgTied;
        for (const auto& c : maxRrpvCandidates) {
            auto d = std::static_pointer_cast<EcgReplData>(c->replacementData);
            if (d->ecg_dbg_tier == maxDBG) dbgTied.push_back(c);
        }

        if (dbgTied.size() == 1 || ecgMode == graph::ECGMode::DBG_ONLY)
            return dbgTied[0];

        // ── L3: Dynamic P-OPT tiebreak ──
        if (graphCtx && graphCtx->rereference.enabled) {
            uint32_t maxDist = 0;
            ReplaceableEntry* victim = dbgTied[0];
            for (const auto& c : dbgTied) {
                auto d = std::static_pointer_cast<EcgReplData>(
                    c->replacementData);
                uint32_t dist = graphCtx->findNextRef(d->line_addr);
                if (dist > maxDist) {
                    maxDist = dist;
                    victim = c;
                }
            }
            return victim;
        }
        return dbgTied[0];

    } else {
        // POPT_PRIMARY mode
        // ── L2: Dynamic P-OPT tiebreak (evict furthest next-reference) ──
        if (graphCtx && graphCtx->rereference.enabled) {
            uint32_t maxDist = 0;
            for (const auto& c : maxRrpvCandidates) {
                auto d = std::static_pointer_cast<EcgReplData>(
                    c->replacementData);
                uint32_t dist = graphCtx->findNextRef(d->line_addr);
                if (dist > maxDist) maxDist = dist;
            }

            std::vector<ReplaceableEntry*> poptTied;
            for (const auto& c : maxRrpvCandidates) {
                auto d = std::static_pointer_cast<EcgReplData>(
                    c->replacementData);
                uint32_t dist = graphCtx->findNextRef(d->line_addr);
                if (dist == maxDist) poptTied.push_back(c);
            }

            if (poptTied.size() == 1) return poptTied[0];

            // ── L3: DBG tier tiebreak among P-OPT ties ──
            uint8_t maxDBG = 0;
            ReplaceableEntry* victim = poptTied[0];
            for (const auto& c : poptTied) {
                auto d = std::static_pointer_cast<EcgReplData>(
                    c->replacementData);
                if (d->ecg_dbg_tier > maxDBG) {
                    maxDBG = d->ecg_dbg_tier;
                    victim = c;
                }
            }
            return victim;
        }

        // No P-OPT matrix: fall back to DBG tiebreak
        uint8_t maxDBG = 0;
        ReplaceableEntry* victim = maxRrpvCandidates[0];
        for (const auto& c : maxRrpvCandidates) {
            auto d = std::static_pointer_cast<EcgReplData>(c->replacementData);
            if (d->ecg_dbg_tier > maxDBG) {
                maxDBG = d->ecg_dbg_tier;
                victim = c;
            }
        }
        return victim;
    }
}

std::shared_ptr<ReplacementData>
GraphEcgRP::instantiateEntry()
{
    return std::make_shared<EcgReplData>(rrpvMax);
}

} // namespace replacement_policy
} // namespace gem5
