// ============================================================================
// ECG Replacement Policy for gem5 — Implementation
// ============================================================================
// Faithful 3-level layered eviction matching cache_sim.h findVictimECG.
// Context loaded from sideband JSON written by benchmark.
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
      llcSize(p.llc_size_bytes),
      sidebandPath(p.sideband_path),
      poptMatrixPath(p.popt_matrix_path)
{
}

void
GraphEcgRP::tryLoadContext() const
{
    if (loadAttempted) return;
    loadAttempted = true;
    if (ctx.loadFromSideband(sidebandPath)) {
        ctx.loaded = true;
    }
    // Load P-OPT rereference matrix for Level 3 tiebreaking
    // (DBG_PRIMARY and POPT_PRIMARY modes use dynamic P-OPT)
    if (ctx.rereference.loadFromFile(poptMatrixPath)) {
        if (ctx.num_regions > 0) {
            ctx.rereference.base_address = ctx.regions[0].base_address;
        }
    }
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
    // Mode-dependent hit promotion (matching standalone cache_sim.h)
    if (ecgMode == graph::ECGMode::POPT_PRIMARY) {
        // P-OPT: reset to 0 on hit (same as SRRIP)
        data->rrpv = 0;
    } else {
        // GRASP-faithful 3-tier: re-classify address on hit
        if (data->is_property_data && ctx.loaded) {
            uint32_t tier = ctx.classifyGRASP(data->line_addr, llcSize);
            if (tier == 1) {
                data->rrpv = 0;  // Hot: aggressive reset
            } else if (data->rrpv > 0) {
                data->rrpv--;    // Others: gradual
            }
            ctx.updateVertexFromAddr(data->line_addr);
        } else if (data->rrpv > 0) {
            data->rrpv--;
        }
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

    tryLoadContext();

    if (pkt && pkt->req) {
        uint64_t addr = pkt->req->getPaddr();
        data->line_addr = addr & ~uint64_t(63);

        // ECG insertion: mode-dependent RRPV
        // POPT_PRIMARY: uniform RRPV=6 (matches pure P-OPT aging)
        // DBG modes: GRASP 3-tier classification (1/6/7)
        constexpr uint8_t P_RRIP = 1;
        constexpr uint8_t I_RRIP = 6;
        constexpr uint8_t M_RRIP = 7;

        if (ecgMode == graph::ECGMode::POPT_PRIMARY) {
            // P-OPT-style: uniform insertion RRPV for all
            data->rrpv = 6;
            data->is_property_data = ctx.loaded && ctx.isPropertyData(addr);
            // Set DBG tier from full 11-bucket classification for tiebreaking
            if (ctx.loaded) {
                uint32_t bucket = ctx.classifyBucket(addr);
                data->ecg_dbg_tier = (bucket < numBuckets)
                    ? static_cast<uint8_t>(bucket) : (numBuckets - 1);
            } else {
                data->ecg_dbg_tier = numBuckets - 1;
            }
        } else if (ctx.loaded && ctx.isPropertyData(addr)) {
            // DBG modes: GRASP 3-tier classification for property data
            uint32_t tier = ctx.classifyGRASP(addr, llcSize);
            data->is_property_data = true;
            if (tier == 1)       data->rrpv = P_RRIP;
            else if (tier == 2)  data->rrpv = I_RRIP;
            else                 data->rrpv = M_RRIP;
            // Full 11-bucket classification for L2 tiebreaking
            uint32_t bucket = ctx.classifyBucket(addr);
            data->ecg_dbg_tier = (bucket < numBuckets)
                ? static_cast<uint8_t>(bucket) : (numBuckets - 1);
            // Update P-OPT vertex tracking from property access
            ctx.updateVertexFromAddr(addr);
        } else if (ctx.loaded) {
            // Non-property data: SRRIP default
            data->rrpv = 2;
            data->is_property_data = false;
            data->ecg_dbg_tier = 0;
        } else {
            data->rrpv = rrpvMax - 1;
            data->is_property_data = false;
            data->ecg_dbg_tier = numBuckets - 1;
        }
        // Compute ecg_popt_hint from P-OPT matrix (matching standalone)
        if (data->is_property_data && ctx.loaded && ctx.rereference.enabled) {
            uint32_t dist = ctx.findNextRef(data->line_addr);
            data->ecg_popt_hint = static_cast<uint8_t>(
                std::min(dist, uint32_t(127)) >> 3);
        } else {
            data->ecg_popt_hint = 0;
        }
    } else {
        data->rrpv = rrpvMax - 1;
        data->ecg_dbg_tier = numBuckets - 1;
        data->ecg_popt_hint = 0;
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

    // Phase 0: POPT_PRIMARY — prefer evicting non-property data when
    // there are property lines to protect. In gem5, most L3 lines are
    // non-property (instructions, CSR edges). Only evict non-property
    // if the set contains BOTH property and non-property lines.
    if (ecgMode == graph::ECGMode::POPT_PRIMARY && ctx.loaded) {
        int propCount = 0;
        ReplaceableEntry* nonPropVictim = nullptr;
        for (const auto& c : candidates) {
            auto d = std::static_pointer_cast<EcgReplData>(c->replacementData);
            if (d->is_property_data) {
                propCount++;
            } else if (!nonPropVictim) {
                nonPropVictim = c;
            }
        }
        // Only evict non-property if there ARE property lines to protect
        if (propCount > 0 && nonPropVictim) {
            return nonPropVictim;
        }
    }

    // POPT_PRIMARY: Use exact P-OPT 3-phase algorithm (bypass SRRIP aging)
    if (ecgMode == graph::ECGMode::POPT_PRIMARY && ctx.loaded &&
        ctx.rereference.enabled) {
        // Phase 2: Find max rereference distance across ALL ways
        uint32_t maxDist = 0;
        for (const auto& c : candidates) {
            auto d = std::static_pointer_cast<EcgReplData>(c->replacementData);
            uint32_t dist = ctx.findNextRef(d->line_addr);
            if (dist > maxDist) maxDist = dist;
        }
        // Phase 3: Among max-distance lines, prefer highest DBG tier
        constexpr uint8_t M_RRPV = 7;
        while (true) {
            ReplaceableEntry* best = nullptr;
            uint8_t bestDBG = 0;
            for (const auto& c : candidates) {
                auto d = std::static_pointer_cast<EcgReplData>(c->replacementData);
                if (ctx.findNextRef(d->line_addr) == maxDist && d->rrpv >= M_RRPV) {
                    if (!best || d->ecg_dbg_tier > bestDBG) {
                        best = c;
                        bestDBG = d->ecg_dbg_tier;
                    }
                }
            }
            if (best) return best;
            // Age only tied lines
            for (const auto& c : candidates) {
                auto d = std::static_pointer_cast<EcgReplData>(c->replacementData);
                if (ctx.findNextRef(d->line_addr) == maxDist && d->rrpv < M_RRPV)
                    d->rrpv++;
            }
        }
    }

    // Level 1: SRRIP aging — find lines at max RRPV
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
    std::vector<ReplaceableEntry*> maxCands;
    for (const auto& c : candidates) {
        auto d = std::static_pointer_cast<EcgReplData>(c->replacementData);
        if (d->rrpv >= rrpvMax) maxCands.push_back(c);
    }
    if (maxCands.size() == 1) return maxCands[0];

    // Level 2/3 tiebreakers (mode-dependent)
    if (ecgMode == graph::ECGMode::DBG_PRIMARY ||
        ecgMode == graph::ECGMode::DBG_ONLY ||
        ecgMode == graph::ECGMode::ECG_EMBEDDED) {

        // L2: DBG tier tiebreak (higher tier = colder = evict first)
        uint8_t maxDBG = 0;
        for (const auto& c : maxCands) {
            auto d = std::static_pointer_cast<EcgReplData>(c->replacementData);
            if (d->ecg_dbg_tier > maxDBG) maxDBG = d->ecg_dbg_tier;
        }
        std::vector<ReplaceableEntry*> dbgTied;
        for (const auto& c : maxCands) {
            auto d = std::static_pointer_cast<EcgReplData>(c->replacementData);
            if (d->ecg_dbg_tier == maxDBG) dbgTied.push_back(c);
        }
        if (dbgTied.size() == 1 || ecgMode == graph::ECGMode::DBG_ONLY)
            return dbgTied[0];

        // L3: ECG_EMBEDDED uses stored P-OPT hint
        if (ecgMode == graph::ECGMode::ECG_EMBEDDED) {
            uint8_t maxHint = 0;
            ReplaceableEntry* victim = dbgTied[0];
            for (const auto& c : dbgTied) {
                auto d = std::static_pointer_cast<EcgReplData>(c->replacementData);
                if (d->ecg_popt_hint > maxHint) {
                    maxHint = d->ecg_popt_hint;
                    victim = c;
                }
            }
            return victim;
        }

        // L3: DBG_PRIMARY uses dynamic P-OPT (if rereference matrix loaded)
        if (ctx.loaded && ctx.rereference.enabled) {
            uint32_t maxDist = 0;
            ReplaceableEntry* victim = dbgTied[0];
            for (const auto& c : dbgTied) {
                auto d = std::static_pointer_cast<EcgReplData>(c->replacementData);
                uint32_t dist = ctx.findNextRef(d->line_addr);
                if (dist > maxDist) { maxDist = dist; victim = c; }
            }
            return victim;
        }
        return dbgTied[0];

    } else {
        // POPT_PRIMARY: L2=P-OPT distance, L3=DBG tier
        if (ctx.loaded && ctx.rereference.enabled) {
            uint32_t maxDist = 0;
            for (const auto& c : maxCands) {
                auto d = std::static_pointer_cast<EcgReplData>(c->replacementData);
                uint32_t dist = ctx.findNextRef(d->line_addr);
                if (dist > maxDist) maxDist = dist;
            }
            std::vector<ReplaceableEntry*> poptTied;
            for (const auto& c : maxCands) {
                auto d = std::static_pointer_cast<EcgReplData>(c->replacementData);
                if (ctx.findNextRef(d->line_addr) == maxDist)
                    poptTied.push_back(c);
            }
            if (poptTied.size() == 1) return poptTied[0];
            // L3: DBG tiebreak
            uint8_t maxDBG = 0;
            ReplaceableEntry* victim = poptTied[0];
            for (const auto& c : poptTied) {
                auto d = std::static_pointer_cast<EcgReplData>(c->replacementData);
                if (d->ecg_dbg_tier > maxDBG) { maxDBG = d->ecg_dbg_tier; victim = c; }
            }
            return victim;
        }
        // No P-OPT: fall back to DBG tiebreak
        uint8_t maxDBG = 0;
        ReplaceableEntry* victim = maxCands[0];
        for (const auto& c : maxCands) {
            auto d = std::static_pointer_cast<EcgReplData>(c->replacementData);
            if (d->ecg_dbg_tier > maxDBG) { maxDBG = d->ecg_dbg_tier; victim = c; }
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
