// ============================================================================
// ECG Replacement Policy for gem5 - Implementation
// ============================================================================
// Reference: Mughrabi et al., GrAPL 2026 + bench/include/cache_sim/cache_sim.h
//
// Sideband-loading gem5 port of cache_sim.h::findVictimECG. The policy keeps
// gem5-specific context loading, but mode ordering mirrors cache_sim.
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
    if (ctx.loaded &&
        (ecgMode == graph::ECGMode::ECG_GRASP_POPT || ctx.rereference.enabled)) {
        return;
    }
    loadAttempted = true;

    constexpr uint64_t retryInterval = 512;
    if ((loadAttemptCount++ % retryInterval) != 0) return;

    if (!ctx.loaded) {
        ctx.loadFromSideband(sidebandPath);
        ctx.loaded = (ctx.num_regions > 0);
    }

    if (ecgMode == graph::ECGMode::ECG_GRASP_POPT) return;

    if (!ctx.rereference.enabled &&
        ctx.rereference.loadFromFile(poptMatrixPath)) {
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
    data->ecg_epoch = 0;
    data->is_property_data = false;
    data->line_addr = 0;
}

void
GraphEcgRP::touch(
    const std::shared_ptr<ReplacementData>& replacement_data,
    const PacketPtr pkt)
{
    auto data = std::static_pointer_cast<EcgReplData>(replacement_data);

    if (ecgMode == graph::ECGMode::POPT_PRIMARY ||
        ecgMode == graph::ECGMode::ECG_COMBINED) {
        data->rrpv = 0;
        return;
    }

    if (ctx.loaded && ctx.isPropertyData(data->line_addr)) {
        data->is_property_data = true;
        uint32_t vertex = UINT32_MAX;
        if (ctx.num_regions > 0) {
            const auto& region = ctx.regions[0];
            vertex = graph::addressToVertex(
            addr, region.base_address, region.upper_bound,
                region.elem_size);
        }
        if (vertex != UINT32_MAX) {
            uint8_t isa_dbg = 0, isa_popt = 0;
            uint16_t isa_epoch = data->ecg_epoch;
            if (graph::lookupEcgMetadataByVertex(vertex, isa_dbg, isa_popt,
                                                 isa_epoch)) {
                data->ecg_dbg_tier = isa_dbg;
                data->ecg_popt_hint = isa_popt;
                data->ecg_epoch = isa_epoch;
            }
        }
        uint32_t tier = ctx.classifyGRASP(data->line_addr, llcSize);
        if (tier == 1) {
            data->rrpv = 0;
        } else if (data->rrpv > 0) {
            data->rrpv--;
        }
        ctx.updateVertexFromAddr(data->line_addr);
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
        uint64_t addr = pkt->req->hasVaddr() ? pkt->req->getVaddr()
                             : pkt->req->getPaddr();
        data->line_addr = addr & ~uint64_t(63);
        data->is_property_data = ctx.loaded && ctx.isPropertyData(addr);

        constexpr uint8_t pRrip = 1;
        constexpr uint8_t iRrip = 6;
        constexpr uint8_t mRrip = 7;

        uint32_t bucket = ctx.loaded ? ctx.classifyBucket(addr)
                                     : numBuckets;
        data->ecg_dbg_tier = (bucket < numBuckets)
            ? static_cast<uint8_t>(bucket) : (numBuckets - 1);

        data->ecg_popt_hint = 0;
        data->ecg_epoch = 0;
            if (data->is_property_data && ctx.rereference.enabled &&
            ecgMode != graph::ECGMode::ECG_GRASP_POPT) {
            uint32_t dist = ctx.findNextRef(data->line_addr);
            data->ecg_popt_hint = static_cast<uint8_t>(
                std::min(dist, uint32_t(127)) >> 3);
        }

        // === S69PRE-M1-MASK: prefer ISA-delivered metadata over sideband ===
        // When the kernel has emitted an ecg.extract opcode for the vertex
        // owning this cache line, the per-vertex metadata table holds the
        // CHARGED=0 paper-faithful DBG tier + POPT quant. Prefer those over
        // the sideband-JSON-derived values. Falls back to sideband if the
        // table has no entry for this vertex.
        if (data->is_property_data && ctx.loaded && ctx.num_regions > 0) {
            const auto& region = ctx.regions[0];
            uint32_t vertex = graph::addressToVertex(
                addr,
                region.base_address, region.upper_bound,
                region.elem_size);
            if (vertex != UINT32_MAX) {
                uint8_t isa_dbg = 0, isa_popt = 0;
                uint16_t isa_epoch = 0;
                bool got = (ecgMode == graph::ECGMode::ECG_GRASP_POPT)
                    ? graph::lookupDecodedEcgHint(vertex, isa_dbg, isa_popt, isa_epoch)
                    : graph::lookupEcgMetadataByVertex(vertex, isa_dbg, isa_popt, isa_epoch);
                if (got) {
                    // Use ISA-delivered metadata directly.
                    data->ecg_dbg_tier = isa_dbg;
                    // POPT quant is 7 bits; ECG_RP stores as 8-bit; range OK.
                    data->ecg_popt_hint = isa_popt;
                    data->ecg_epoch = isa_epoch;
                }
            }
        }

        if (ecgMode == graph::ECGMode::POPT_PRIMARY) {
            data->rrpv = (rrpvMax > 0) ? rrpvMax - 1 : 0;
        } else if (ecgMode == graph::ECGMode::ECG_COMBINED) {
            uint8_t dbgRrpv = mRrip;
            if (data->is_property_data && ctx.loaded) {
                uint32_t tier = ctx.classifyGRASP(addr, llcSize);
                if (tier == 1) dbgRrpv = pRrip;
                else if (tier == 2) dbgRrpv = iRrip;
            }
            uint8_t poptRrpv = static_cast<uint8_t>(
                (uint32_t(data->ecg_popt_hint) * rrpvMax) / 15u);
            uint8_t combined = static_cast<uint8_t>(
                (uint32_t(dbgRrpv) + uint32_t(poptRrpv)) / 2u);
            if (combined == 0 && dbgRrpv > 0) combined = 1;
            data->rrpv = std::min<uint8_t>(combined, rrpvMax);
        } else if (data->is_property_data && ctx.loaded) {
            uint32_t tier = ctx.classifyGRASP(addr, llcSize);
            if (tier == 1) data->rrpv = pRrip;
            else if (tier == 2) data->rrpv = iRrip;
            else data->rrpv = mRrip;
            ctx.updateVertexFromAddr(addr);
        } else if (ctx.loaded) {
            data->rrpv = 2;
        } else {
            data->rrpv = 2;
        }
    } else {
        data->rrpv = (rrpvMax > 0) ? rrpvMax - 1 : 0;
        data->ecg_dbg_tier = numBuckets - 1;
        data->ecg_popt_hint = 0;
        data->ecg_epoch = 0;
            data->is_property_data = false;
        data->line_addr = 0;
    }
}

void
GraphEcgRP::reset(
    const std::shared_ptr<ReplacementData>& replacement_data) const
{
    auto data = std::static_pointer_cast<EcgReplData>(replacement_data);
    data->rrpv = (rrpvMax > 0) ? rrpvMax - 1 : 0;
    data->ecg_dbg_tier = numBuckets - 1;
    data->ecg_popt_hint = 0;
    data->ecg_epoch = 0;
}

ReplaceableEntry*
GraphEcgRP::getVictim(const ReplacementCandidates& candidates) const
{
    assert(candidates.size() > 0);

    auto getData = [](ReplaceableEntry* c) {
        return std::static_pointer_cast<EcgReplData>(c->replacementData);
    };

    if (ecgMode == graph::ECGMode::ECG_COMBINED) {
        while (true) {
            for (const auto& c : candidates) {
                if (getData(c)->rrpv >= rrpvMax) return c;
            }
            for (const auto& c : candidates) {
                auto data = getData(c);
                if (data->rrpv < rrpvMax) data->rrpv++;
            }
        }
    }

    if (ecgMode == graph::ECGMode::ECG_GRASP_POPT && ctx.loaded) {
        int propCount = 0;
        for (const auto& c : candidates) {
            auto data = getData(c);
            data->is_property_data = ctx.isPropertyData(data->line_addr);
            if (data->is_property_data) propCount++;
        }

        if (propCount != static_cast<int>(candidates.size())) {
            for (const auto& c : candidates) {
                auto data = getData(c);
                if (!data->is_property_data) return c;
            }
        }

        const uint32_t n = std::max<uint32_t>(1u, ctx.topology.num_vertices);
        const uint32_t ne = std::max<uint32_t>(2u, ctx.topology.edge_epoch_count);
        uint32_t cur_epoch = static_cast<uint32_t>(
            (static_cast<uint64_t>(ctx.currentVertexForPopt()) * ne) / n);
        if (cur_epoch >= ne) cur_epoch = ne - 1;

        uint32_t maxDist = 0;
        uint8_t maxDbg = 0;
        ReplaceableEntry* victim = candidates[0];
        bool haveVictim = false;
        for (const auto& c : candidates) {
            auto data = getData(c);
            uint32_t epoch = data->ecg_epoch;
            if (epoch >= ne) epoch = ne - 1;
            uint32_t dist = (epoch + ne - cur_epoch) % ne;
            if (!haveVictim || dist > maxDist ||
                (dist == maxDist && data->ecg_dbg_tier > maxDbg)) {
                victim = c;
                maxDist = dist;
                maxDbg = data->ecg_dbg_tier;
                haveVictim = true;
            }
        }
        return victim;
    }

    if (ecgMode == graph::ECGMode::POPT_PRIMARY && ctx.loaded &&
        ctx.rereference.enabled) {
        int propCount = 0;
        for (const auto& c : candidates) {
            auto data = getData(c);
            data->is_property_data = ctx.isPropertyData(data->line_addr);
            if (data->is_property_data) propCount++;
        }

        if (propCount != static_cast<int>(candidates.size())) {
            for (const auto& c : candidates) {
                auto data = getData(c);
                if (!data->is_property_data) return c;
            }
        }

        uint8_t maxDist = 0;
        std::vector<std::pair<ReplaceableEntry*, uint8_t>> dists;
        dists.reserve(candidates.size());
        for (const auto& c : candidates) {
            uint32_t dist = ctx.findNextRef(getData(c)->line_addr);
            uint8_t d8 = static_cast<uint8_t>(std::min(dist, uint32_t(127)));
            dists.emplace_back(c, d8);
            if (d8 > maxDist) maxDist = d8;
        }

        const uint8_t maxRrpv = rrpvMax;
        while (true) {
            ReplaceableEntry* best = nullptr;
            uint8_t bestDbg = 0;
            for (auto& [c, dist] : dists) {
                auto data = getData(c);
                if (dist == maxDist && data->rrpv >= maxRrpv &&
                    (!best || data->ecg_dbg_tier > bestDbg)) {
                    best = c;
                    bestDbg = data->ecg_dbg_tier;
                }
            }
            if (best) return best;
            for (auto& [c, dist] : dists) {
                auto data = getData(c);
                if (dist == maxDist && data->rrpv < maxRrpv) data->rrpv++;
            }
        }
    }

    if (ecgMode == graph::ECGMode::DBG_ONLY) {
        while (true) {
            for (const auto& c : candidates) {
                if (getData(c)->rrpv >= rrpvMax) return c;
            }
            for (const auto& c : candidates) {
                auto data = getData(c);
                if (data->rrpv < rrpvMax) data->rrpv++;
            }
        }
    }

    while (true) {
        bool found = false;
        for (const auto& c : candidates) {
            if (getData(c)->rrpv >= rrpvMax) { found = true; break; }
        }
        if (found) break;
        for (const auto& c : candidates) {
            auto data = getData(c);
            if (data->rrpv < rrpvMax) data->rrpv++;
        }
    }

    std::vector<ReplaceableEntry*> maxCands;
    maxCands.reserve(candidates.size());
    for (const auto& c : candidates) {
        if (getData(c)->rrpv >= rrpvMax) maxCands.push_back(c);
    }
    if (maxCands.size() == 1) return maxCands[0];

    if (ecgMode == graph::ECGMode::ECG_EMBEDDED) {
        // L2: stored P-OPT hint primary (evict highest hint = furthest future).
        // L3: DBG tier tiebreak among same-hint lines.
        uint8_t maxHint = 0;
        for (const auto& c : maxCands) {
            uint8_t hint = getData(c)->ecg_popt_hint;
            if (hint > maxHint) maxHint = hint;
        }

        std::vector<ReplaceableEntry*> hintTied;
        for (const auto& c : maxCands) {
            if (getData(c)->ecg_popt_hint == maxHint) hintTied.push_back(c);
        }
        if (hintTied.size() == 1) return hintTied[0];

        uint8_t maxDbg = 0;
        ReplaceableEntry* victim = hintTied[0];
        for (const auto& c : hintTied) {
            uint8_t dbg = getData(c)->ecg_dbg_tier;
            if (dbg > maxDbg) { maxDbg = dbg; victim = c; }
        }
        return victim;
    }

    uint8_t maxDbg = 0;
    for (const auto& c : maxCands) {
        uint8_t dbg = getData(c)->ecg_dbg_tier;
        if (dbg > maxDbg) maxDbg = dbg;
    }

    std::vector<ReplaceableEntry*> dbgTied;
    for (const auto& c : maxCands) {
        if (getData(c)->ecg_dbg_tier == maxDbg) dbgTied.push_back(c);
    }
    if (dbgTied.size() == 1) {
        return dbgTied[0];
    }

    if (ctx.loaded && ctx.rereference.enabled) {
        uint32_t maxDist = 0;
        ReplaceableEntry* victim = dbgTied[0];
        for (const auto& c : dbgTied) {
            uint32_t dist = ctx.findNextRef(getData(c)->line_addr);
            if (dist > maxDist) { maxDist = dist; victim = c; }
        }
        return victim;
    }
    return dbgTied[0];
}

std::shared_ptr<ReplacementData>
GraphEcgRP::instantiateEntry()
{
    return std::make_shared<EcgReplData>(rrpvMax);
}

} // namespace replacement_policy
} // namespace gem5
