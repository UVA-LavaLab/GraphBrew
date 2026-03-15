// ============================================================================
// P-OPT Replacement Policy for gem5 — Implementation
// ============================================================================
// Reference: Balaji et al., HPCA 2021 + bench/include/cache_sim/cache_sim.h
// ============================================================================

#include "mem/cache/replacement_policies/popt_rp.hh"

#include <algorithm>
#include <cassert>

namespace gem5 {
namespace replacement_policy {

GraphPoptRP::GraphPoptRP(const Params &p)
    : Base(p),
      maxRRPV(p.max_rrpv)
{
}

void
GraphPoptRP::invalidate(
    const std::shared_ptr<ReplacementData>& replacement_data)
{
    auto data = std::static_pointer_cast<PoptReplData>(replacement_data);
    data->rrpv = maxRRPV;
    data->is_property_data = false;
    data->line_addr = 0;
}

void
GraphPoptRP::touch(
    const std::shared_ptr<ReplacementData>& replacement_data,
    const PacketPtr pkt)
{
    auto data = std::static_pointer_cast<PoptReplData>(replacement_data);
    // P-OPT on hit: RRPV = 0 (near-immediate re-reference)
    data->rrpv = 0;
}

void
GraphPoptRP::touch(
    const std::shared_ptr<ReplacementData>& replacement_data) const
{
    auto data = std::static_pointer_cast<PoptReplData>(replacement_data);
    data->rrpv = 0;
}

void
GraphPoptRP::reset(
    const std::shared_ptr<ReplacementData>& replacement_data,
    const PacketPtr pkt)
{
    auto data = std::static_pointer_cast<PoptReplData>(replacement_data);

    // P-OPT insertion: RRPV = M-1 (matching reference llc.cpp)
    data->rrpv = (maxRRPV > 0) ? maxRRPV - 1 : 0;

    if (pkt && pkt->req) {
        uint64_t addr = pkt->req->getPaddr();
        // Align to cache line
        data->line_addr = addr & ~uint64_t(63);

        // Classify if this is graph property data
        data->is_property_data = graphCtx && graphCtx->isPropertyData(data->line_addr);
    } else {
        data->line_addr = 0;
        data->is_property_data = false;
    }
}

void
GraphPoptRP::reset(
    const std::shared_ptr<ReplacementData>& replacement_data) const
{
    auto data = std::static_pointer_cast<PoptReplData>(replacement_data);
    data->rrpv = (maxRRPV > 0) ? maxRRPV - 1 : 0;
}

ReplaceableEntry*
GraphPoptRP::getVictim(const ReplacementCandidates& candidates) const
{
    assert(candidates.size() > 0);

    bool hasPopt = graphCtx && graphCtx->rereference.enabled;

    if (!hasPopt) {
        // Fallback to LRU-like: evict highest RRPV
        ReplaceableEntry* victim = candidates[0];
        uint8_t highest = 0;
        for (const auto& c : candidates) {
            auto d = std::static_pointer_cast<PoptReplData>(c->replacementData);
            if (d->rrpv > highest) {
                highest = d->rrpv;
                victim = c;
            }
        }
        return victim;
    }

    // ── Phase 1: Evict non-graph data first ──
    for (const auto& c : candidates) {
        auto d = std::static_pointer_cast<PoptReplData>(c->replacementData);
        if (!d->is_property_data) return c;
    }

    // ── Phase 2: All ways are graph data → find max rereference distance ──
    uint8_t maxRerefDist = 0;
    std::vector<std::pair<ReplaceableEntry*, uint8_t>> wayDists;
    wayDists.reserve(candidates.size());

    for (const auto& c : candidates) {
        auto d = std::static_pointer_cast<PoptReplData>(c->replacementData);
        uint32_t dist = graphCtx->findNextRef(d->line_addr);
        uint8_t d8 = static_cast<uint8_t>(std::min(dist, uint32_t(127)));
        wayDists.emplace_back(c, d8);
        if (d8 > maxRerefDist) maxRerefDist = d8;
    }

    // ── Phase 3: RRIP tiebreaker among lines with max rereference distance ──
    // Age only tied lines until one reaches maxRRPV
    while (true) {
        for (auto& [entry, dist] : wayDists) {
            if (dist == maxRerefDist) {
                auto d = std::static_pointer_cast<PoptReplData>(
                    entry->replacementData);
                if (d->rrpv >= maxRRPV) return entry;
            }
        }
        for (auto& [entry, dist] : wayDists) {
            if (dist == maxRerefDist) {
                auto d = std::static_pointer_cast<PoptReplData>(
                    entry->replacementData);
                if (d->rrpv < maxRRPV) d->rrpv++;
            }
        }
    }
}

std::shared_ptr<ReplacementData>
GraphPoptRP::instantiateEntry()
{
    return std::make_shared<PoptReplData>(maxRRPV);
}

} // namespace replacement_policy
} // namespace gem5
