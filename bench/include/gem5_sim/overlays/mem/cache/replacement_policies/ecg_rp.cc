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
      sidebandPath(p.sideband_path)
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
    // Faithful GRASP-style hit promotion:
    //   Hot property data: RRPV -> 0
    //   Others: decrement by 1
    if (data->is_property_data && data->rrpv <= 1) {
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

    tryLoadContext();

    if (pkt && pkt->req) {
        uint64_t addr = pkt->req->getPaddr();
        data->line_addr = addr & ~uint64_t(63);

        // ECG insertion: same as GRASP 3-tier classification
        // (matching standalone cache_sim.h ECG insertion block)
        constexpr uint8_t P_RRIP = 1;
        constexpr uint8_t I_RRIP = 6;
        constexpr uint8_t M_RRIP = 7;

        if (ctx.loaded) {
            uint32_t tier = ctx.classifyGRASP(addr, llcSize);
            data->is_property_data = ctx.isPropertyData(addr);
            if (tier == 1)       data->rrpv = P_RRIP;
            else if (tier == 2)  data->rrpv = I_RRIP;
            else                 data->rrpv = M_RRIP;
            // DBG tier for eviction tiebreaking
            data->ecg_dbg_tier = (tier <= 3) ? static_cast<uint8_t>(tier) : 3;
        } else {
            data->rrpv = rrpvMax - 1;
            data->is_property_data = false;
            data->ecg_dbg_tier = numBuckets - 1;
        }
        data->ecg_popt_hint = 0;
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
