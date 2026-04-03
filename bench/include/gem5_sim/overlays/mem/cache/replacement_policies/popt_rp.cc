// ============================================================================
// P-OPT Replacement Policy for gem5 — Implementation
// ============================================================================
// Faithful P-OPT adapted for gem5's full-system address space.
//
// Key difference from standalone cache_sim: gem5 sees ALL memory accesses
// (instructions, stack, CSR edges) not just property arrays. Phase 1
// ("evict non-property first") would thrash in gem5 because most L3 lines
// are non-property. Instead, we use P-OPT rereference distance ONLY for
// property data lines, and SRRIP for everything else.
//
// Reference: Balaji et al., HPCA 2021
// ============================================================================

#include "mem/cache/replacement_policies/popt_rp.hh"

#include <algorithm>
#include <cassert>
#include <vector>

namespace gem5 {
namespace replacement_policy {

GraphPoptRP::GraphPoptRP(const Params &p)
    : Base(p),
      maxRRPV(p.max_rrpv),
      sidebandPath(p.sideband_path),
      poptMatrixPath(p.popt_matrix_path)
{
}

void
GraphPoptRP::tryLoadContext() const
{
    if (loadAttempted) return;
    loadAttempted = true;

    ctx.loadFromSideband(sidebandPath);

    if (ctx.rereference.loadFromFile(poptMatrixPath)) {
        // base_address is no longer hardcoded to region[0].
        // findNextRef() in GraphCacheContext now searches all regions
        // to find which one the address belongs to.
        // Set base_address to region[0] as a fallback for RereferenceMatrix's
        // direct findNextRefByAddr() (used when findNextRef bypasses context).
        if (ctx.num_regions > 0) {
            ctx.rereference.base_address = ctx.regions[0].base_address;
        }
    }
    ctx.loaded = (ctx.num_regions > 0);
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

    tryLoadContext();

    // P-OPT insertion: RRPV = M-1
    data->rrpv = (maxRRPV > 0) ? maxRRPV - 1 : 0;

    if (pkt && pkt->req) {
        uint64_t addr = pkt->req->getPaddr();
        data->line_addr = addr & ~uint64_t(63);
        data->is_property_data = ctx.loaded && ctx.isPropertyData(data->line_addr);
        // Update P-OPT vertex tracking from property accesses
        if (data->is_property_data) {
            ctx.updateVertexFromAddr(addr);
        }
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

    bool hasPopt = ctx.loaded && ctx.rereference.enabled;

    // If no P-OPT context, fall back to SRRIP
    if (!hasPopt) {
        while (true) {
            for (const auto& c : candidates) {
                auto d = std::static_pointer_cast<PoptReplData>(c->replacementData);
                if (d->rrpv >= maxRRPV) return c;
            }
            for (const auto& c : candidates) {
                auto d = std::static_pointer_cast<PoptReplData>(c->replacementData);
                if (d->rrpv < maxRRPV) d->rrpv++;
            }
        }
    }

    // Count property vs non-property lines in the candidate set
    int propCount = 0;
    for (const auto& c : candidates) {
        auto d = std::static_pointer_cast<PoptReplData>(c->replacementData);
        if (d->is_property_data) propCount++;
    }

    // If ALL candidates are property data, use full P-OPT 3-phase
    if (propCount == static_cast<int>(candidates.size())) {
        // Phase 2: find max rereference distance among property lines
        uint8_t maxDist = 0;
        std::vector<std::pair<ReplaceableEntry*, uint8_t>> wayDists;
        wayDists.reserve(candidates.size());

        for (const auto& c : candidates) {
            auto d = std::static_pointer_cast<PoptReplData>(c->replacementData);
            uint32_t dist = ctx.findNextRef(d->line_addr);
            uint8_t d8 = static_cast<uint8_t>(std::min(dist, uint32_t(127)));
            wayDists.emplace_back(c, d8);
            if (d8 > maxDist) maxDist = d8;
        }

        // Phase 3: RRIP tiebreaker among max-distance lines
        while (true) {
            for (auto& [entry, dist] : wayDists) {
                if (dist == maxDist) {
                    auto d = std::static_pointer_cast<PoptReplData>(
                        entry->replacementData);
                    if (d->rrpv >= maxRRPV) return entry;
                }
            }
            for (auto& [entry, dist] : wayDists) {
                if (dist == maxDist) {
                    auto d = std::static_pointer_cast<PoptReplData>(
                        entry->replacementData);
                    if (d->rrpv < maxRRPV) d->rrpv++;
                }
            }
        }
    }

    // Mixed set (property + non-property): use SRRIP with P-OPT boost.
    // Property lines with far rereference get RRPV boosted toward eviction.
    // Non-property lines use standard SRRIP aging.
    // This avoids the Phase 1 problem of blindly evicting non-property data.
    for (const auto& c : candidates) {
        auto d = std::static_pointer_cast<PoptReplData>(c->replacementData);
        if (d->is_property_data) {
            uint32_t dist = ctx.findNextRef(d->line_addr);
            // Far rereference -> boost RRPV toward eviction
            if (dist > 64 && d->rrpv < maxRRPV) {
                d->rrpv = maxRRPV;  // Push far-future property lines to evict
            }
        }
    }

    // Standard SRRIP eviction on the (now-boosted) set
    while (true) {
        for (const auto& c : candidates) {
            auto d = std::static_pointer_cast<PoptReplData>(c->replacementData);
            if (d->rrpv >= maxRRPV) return c;
        }
        for (const auto& c : candidates) {
            auto d = std::static_pointer_cast<PoptReplData>(c->replacementData);
            if (d->rrpv < maxRRPV) d->rrpv++;
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
