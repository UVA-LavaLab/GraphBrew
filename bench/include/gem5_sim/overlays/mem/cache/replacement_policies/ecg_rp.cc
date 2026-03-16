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
      ecgMode(graph::stringToECGMode(p.ecg_mode)),
      llcSize(p.llc_size),
      learnRegions(p.learn_regions)
{
}

void
GraphEcgRP::invalidate(
    const std::shared_ptr<ReplacementData>& replacement_data)
{
    auto data = std::static_pointer_cast<EcgReplData>(replacement_data);
    data->rrpv = rrpvMax;
    data->ecg_dbg_tier = 0;
    data->ecg_popt_hint = 0;
    data->is_property_data = false;
    data->line_addr = 0;
}

void
GraphEcgRP::touch(
    const std::shared_ptr<ReplacementData>& replacement_data,
    const PacketPtr pkt)
{
    auto data = std::static_pointer_cast<EcgReplData>(replacement_data);

    // ECG: GRASP-faithful 3-tier hit promotion
    if (pkt && pkt->req) {
        uint64_t addr = pkt->req->getPaddr() & ~uint64_t(63);
        uint32_t tier = classifyGRASPTier(addr);
        if (tier == 1) {
            data->rrpv = 0;            // Hot: aggressive reset
        } else if (data->rrpv > 0) {
            data->rrpv--;              // Others: gradual decrement
        }
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

        // Online region learning: track address range during warmup
        if (learnRegions && !regionsLearned) {
            if (data->line_addr < minAddr) minAddr = data->line_addr;
            if (data->line_addr + 64 > maxAddr) maxAddr = data->line_addr + 64;
            accessCount++;
            if (accessCount >= LEARN_WARMUP && minAddr < maxAddr) {
                learnedBase = minAddr;
                learnedEnd = maxAddr;
                regionsLearned = true;
            }
        }

        // ECG insertion: GRASP-faithful 3-tier RRPV (matching standalone)
        constexpr uint8_t P_RRIP = 1;    // Priority (hot)
        constexpr uint8_t I_RRIP = 6;    // Intermediate (moderate)
        constexpr uint8_t M_RRIP_C = 7;  // Max (cold)

        uint32_t tier = classifyGRASPTier(data->line_addr);
        if (tier == 1)       data->rrpv = P_RRIP;
        else if (tier == 2)  data->rrpv = I_RRIP;
        else                 data->rrpv = M_RRIP_C;

        data->is_property_data = (tier <= 3);

        // Store ECG mask fields for eviction tiebreaking
        if (graphCtx && graphCtx->current_mask != 0 &&
            graphCtx->mask_config.enabled) {
            data->ecg_dbg_tier = graphCtx->mask_config.decodeDBG(
                graphCtx->current_mask);
            data->ecg_popt_hint = graphCtx->mask_config.decodePOPT(
                graphCtx->current_mask);
        } else if (graphCtx) {
            uint32_t bucket = graphCtx->classifyBucket(data->line_addr);
            data->ecg_dbg_tier = (bucket < numBuckets)
                ? static_cast<uint8_t>(bucket) : 0;
            data->ecg_popt_hint = 0;
        } else {
            data->ecg_dbg_tier = (tier == 1) ? 0 : (tier == 2) ? 4 : 10;
            data->ecg_popt_hint = 0;
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
        ecgMode == graph::ECGMode::DBG_ONLY ||
        ecgMode == graph::ECGMode::ECG_EMBEDDED) {
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

        // ── L3 tiebreak ──
        if (ecgMode == graph::ECGMode::ECG_EMBEDDED) {
            // ECG_EMBEDDED: use stored P-OPT hint (zero LLC overhead)
            uint8_t maxHint = 0;
            ReplaceableEntry* victim = dbgTied[0];
            for (const auto& c : dbgTied) {
                auto d = std::static_pointer_cast<EcgReplData>(
                    c->replacementData);
                if (d->ecg_popt_hint > maxHint) {
                    maxHint = d->ecg_popt_hint;
                    victim = c;
                }
            }
            return victim;
        }

        // DBG_PRIMARY: Dynamic P-OPT tiebreak via rereference matrix
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

uint32_t
GraphEcgRP::classifyGRASPTier(uint64_t addr) const
{
    // Use explicit context first
    if (graphCtx) {
        return graphCtx->classifyGRASP(addr, llcSize);
    }

    // Use learned region
    if (learnRegions && regionsLearned) {
        if (addr < learnedBase || addr >= learnedEnd)
            return 3;  // COLD
        uint64_t offset = addr - learnedBase;
        uint64_t hotBytes = static_cast<uint64_t>(0.10 * llcSize);
        if (offset < hotBytes)      return 1;  // HOT
        if (offset < 2 * hotBytes)  return 2;  // MODERATE
        return 3;                               // COLD
    }

    return 3;  // No context, no learning → cold
}

} // namespace replacement_policy
} // namespace gem5
