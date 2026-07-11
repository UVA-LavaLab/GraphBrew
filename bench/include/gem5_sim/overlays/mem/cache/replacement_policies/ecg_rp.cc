// ============================================================================
// ECG Replacement Policy for gem5 - Implementation
// ============================================================================
// Reference: Mughrabi et al., GrAPL 2026 + bench/include/cache_sim/cache_sim.h
//
// Sideband-loading gem5 port of cache_sim.h::findVictimECG. The policy keeps
// gem5-specific context loading, but mode ordering mirrors cache_sim.
// ============================================================================

#include "mem/cache/replacement_policies/ecg_rp.hh"

#include "mem/cache/replacement_policies/ecg_epoch_request_ext.hh"
#include "mem/cache/replacement_policies/ecg_victim_policy.hh"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <iostream>
#include <vector>

namespace gem5 {
namespace replacement_policy {

namespace {
// ECG GRASP-tier SOURCE (ECG_GRASP_SRC), mirrors cache_sim's two variants:
//   mask (0, our ECG): DELIVERED per-vertex graspTierByIndex keyed by the INSERTED
//     LINE's own vertex (graph::addressToVertex) — identical across simulators.
//   region (1, default, original GRASP): classifyGRASP(addr) spatial top-fraction.
// Templated on the context type to avoid spelling graph::GraphCacheContext.
template <typename Ctx>
inline uint32_t ecgGraspTier(const Ctx& ctx, uint64_t addr, uint64_t llcSize) {
    static const int gsrc = [](){
        const char* v = std::getenv("ECG_GRASP_SRC");
        return (v && std::string(v) == "mask") ? 0 : 1;  // default 1 = region
    }();
    static const double ghf = [](){
        const char* v = std::getenv("GRASP_HOT_FRACTION");
        double f = v ? std::atof(v) : 0.15;
        return (f > 0.0 && f <= 1.0) ? f : 0.15;
    }();
    if (gsrc == 0) {  // MASK (ECG): delivered per-vertex tier, byte-exact to region
        return ctx.maskGraspTier(addr, ghf);
    }
    return ctx.classifyGRASP(addr, llcSize);  // REGION (GRASP)
}
}  // namespace

GraphEcgRP::GraphEcgRP(const Params &p)
    : Base(p),
      rrpvMax(p.rrpv_max),
      numBuckets(p.num_buckets),
      ecgMode(graph::stringToECGMode(p.ecg_mode)),
      llcSize(p.llc_size_bytes),
      sidebandPath(p.sideband_path),
      poptMatrixPath(p.popt_matrix_path)
{
    // ECG-CONFIG proof banner (ECG_DEBUG=1): same format as cache_sim/sniper, proving
    // which mode/variant this gem5 ECG replacement policy resolved. p.ecg_mode is the
    // mode string straight from the SimObject Param (no reverse lookup needed).
    const char* dbg = std::getenv("ECG_DEBUG");
    if (dbg && *dbg && std::string(dbg) != "0") {
        const char* var = std::getenv("ECG_VARIANT");
        std::cerr << "[ECG-CONFIG sim=gem5 policy=ECG mode=" << p.ecg_mode
                  << " variant=" << (var ? var : "rrip_first")
                  << " llc=" << llcSize << "B]\n";
    }
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
    data->ecg_epoch2 = 0;
    data->ecg_epoch_count = 0;
    data->ecg_epoch_valid = false;
    data->is_property_data = false;
    data->line_addr = 0;
}

void
GraphEcgRP::touch(
    const std::shared_ptr<ReplacementData>& replacement_data,
    const PacketPtr pkt)
{
    auto data = std::static_pointer_cast<EcgReplData>(replacement_data);

    if (ecgMode == graph::ECGMode::ECG_GRASP_POPT) {
        tryLoadContext();
        uint64_t addr = data->line_addr;
        if (pkt && pkt->req) {
            addr = pkt->req->hasVaddr() ? pkt->req->getVaddr()
                         : pkt->req->getPaddr();
            data->line_addr = addr & ~uint64_t(63);
        }
        data->lastTouchTick = curTick();
        if (ctx.loaded && ctx.isPropertyData(addr)) {
            data->is_property_data = true;
            uint32_t vertex = UINT32_MAX;
            for (uint32_t ri = 0; ri < ctx.num_regions; ++ri) {
                const auto& reg = ctx.regions[ri];
                if (addr >= reg.base_address && addr < reg.upper_bound) {
                    vertex = graph::addressToVertex(addr, reg.base_address,
                                 reg.upper_bound, reg.elem_size);
                    break;
                }
            }
            if (vertex != UINT32_MAX &&
                (ecgMode != graph::ECGMode::ECG_GRASP_POPT ||
                 ctx.isEcgEpochData(addr))) {
                uint8_t isa_dbg = 0, isa_popt = 0;
                uint16_t isa_epoch = data->ecg_epoch;
                uint16_t isa_epoch2 = data->ecg_epoch2;
                uint8_t isa_count = data->ecg_epoch_count;
                if (graph::lookupDecodedEcgHint2(
                        vertex, isa_epoch, isa_epoch2, isa_count)) {
                    // ECG_GRASP_POPT's EVICT format carries no DBG field.
                    // Preserve the address-derived GRASP tier for degree_first.
                    if (ecgMode != graph::ECGMode::ECG_GRASP_POPT)
                        data->ecg_dbg_tier = isa_dbg;
                    data->ecg_popt_hint = isa_popt;
                    data->ecg_epoch = isa_epoch;
                    data->ecg_epoch2 = isa_epoch2;
                    data->ecg_epoch_count = isa_count;
                    data->ecg_epoch_valid = true;
                }
            }
            uint32_t tier = ecgGraspTier(ctx, addr, llcSize);
            data->ecg_dbg_tier = static_cast<uint8_t>(tier);
            if (tier == 1) {
                data->rrpv = 0;
            } else if (data->rrpv > 0) {
                data->rrpv--;
            }
            ctx.updateVertexFromAddr(addr);
        } else if (data->rrpv > 0) {
            data->is_property_data = false;
            data->rrpv--;
        }
        return;
    }

    if (ecgMode == graph::ECGMode::POPT_PRIMARY ||
        ecgMode == graph::ECGMode::ECG_COMBINED) {
        data->rrpv = 0;
        return;
    }

    if (ctx.loaded && ctx.isPropertyData(data->line_addr)) {
        data->is_property_data = true;
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
        data->lastTouchTick = curTick();

        constexpr uint8_t pRrip = 1;
        constexpr uint8_t iRrip = 6;
        constexpr uint8_t mRrip = 7;

        uint32_t bucket = ctx.loaded ? ctx.classifyBucket(addr)
                                     : numBuckets;
        data->ecg_dbg_tier = (bucket < numBuckets)
            ? static_cast<uint8_t>(bucket) : (numBuckets - 1);

        data->ecg_popt_hint = 0;
        data->ecg_epoch = 0;
        data->ecg_epoch2 = 0;
        data->ecg_epoch_count = 0;
        data->ecg_epoch_valid = false;
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
            uint32_t vertex = UINT32_MAX;
            uint64_t reg_base = 0; uint32_t reg_elem = 0;
            for (uint32_t ri = 0; ri < ctx.num_regions; ++ri) {
                const auto& reg = ctx.regions[ri];
                if (addr >= reg.base_address && addr < reg.upper_bound) {
                    vertex = graph::addressToVertex(addr, reg.base_address,
                                 reg.upper_bound, reg.elem_size);
                    reg_base = reg.base_address; reg_elem = reg.elem_size;
                    break;
                }
            }
            if (vertex != UINT32_MAX &&
                (ecgMode != graph::ECGMode::ECG_GRASP_POPT ||
                 ctx.isEcgEpochData(addr))) {
                uint8_t isa_dbg = 0, isa_popt = 0;
                uint16_t isa_epoch = 0;
                uint16_t isa_epoch2 = 0;
                uint8_t isa_count = 0;
                uint32_t isa_dest = 0;
                // OoO request-sideband FIRST (race-free; an O3 ecg.load attaches the
                // epoch to the demand request). Falls back to the in-order single-slot
                // mailbox / per-vertex table, which is equivalent for serialized loads.
                bool got = graph::readEcgEpoch(pkt->req, isa_epoch, isa_dbg,
                                               isa_popt, isa_dest);
                if (got) {
                    isa_epoch2 = isa_epoch;
                    isa_count = 1;
                }
                if (got && reg_elem > 0) {
                    // Dest-guard: accept the sideband epoch only if the ecg.load's dest
                    // maps to the SAME cache line as this fill (defends against MSHR
                    // coalescing delivering a different line's epoch). Within-line by
                    // construction in the validated A/B, so this never rejects there.
                    uint64_t dest_line =
                        (reg_base + static_cast<uint64_t>(isa_dest) * reg_elem) & ~uint64_t(63);
                    if (dest_line != (addr & ~uint64_t(63))) got = false;
                }
                if (!got) {
                    if (ecgMode == graph::ECGMode::ECG_GRASP_POPT) {
                        got = graph::lookupDecodedEcgHint2(
                            vertex, isa_epoch, isa_epoch2, isa_count);
                    } else {
                        got = graph::lookupEcgMetadataByVertex(
                            vertex, isa_dbg, isa_popt, isa_epoch);
                        if (got) {
                            isa_epoch2 = isa_epoch;
                            isa_count = 1;
                        }
                    }
                }
                if (got) {
                    // Use ISA-delivered metadata directly.
                    if (ecgMode != graph::ECGMode::ECG_GRASP_POPT)
                        data->ecg_dbg_tier = isa_dbg;
                    data->ecg_popt_hint = isa_popt;  // 7-bit POPT quant
                    data->ecg_epoch = isa_epoch;
                    data->ecg_epoch2 = isa_epoch2;
                    data->ecg_epoch_count = isa_count;
                    data->ecg_epoch_valid = true;
                } else if (ecgMode == graph::ECGMode::ECG_GRASP_POPT) {
                    // Path A: this is a prefetch FILL (the in-order demand
                    // single-slot holds a different vertex). Recover the
                    // candidate epoch the prefetch carried, from the bounded
                    // in-flight buffer; keep the degree-derived DBG tier.
                    uint16_t pf_epoch = 0;
                    if (graph::consumePendingPrefetchEpoch(vertex, pf_epoch)) {
                        data->ecg_epoch = pf_epoch;
                        data->ecg_epoch2 = pf_epoch;
                        data->ecg_epoch_count = 1;
                        data->ecg_epoch_valid = true;
                    }
                }
            }
        }

        if (ecgMode == graph::ECGMode::POPT_PRIMARY) {
            data->rrpv = (rrpvMax > 0) ? rrpvMax - 1 : 0;
        } else if (ecgMode == graph::ECGMode::ECG_GRASP_POPT) {
            // Non-property insertion RRPV: max (evicted before reused property, but
            // recency-aware via touch) for all variants except legacy shortcircuit,
            // whose eviction ignores rrpv. (ECG_VARIANT read in getVictim.)
            static const bool legacy_sc = [](){
                const char* v = std::getenv("ECG_VARIANT");
                return v && std::string(v) == "shortcircuit";
            }();
            if (data->is_property_data && ctx.loaded) {
                uint32_t tier = ecgGraspTier(ctx, addr, llcSize);
                data->ecg_dbg_tier = static_cast<uint8_t>(tier);
                if (tier == 1) data->rrpv = pRrip;
                else if (tier == 2) data->rrpv = iRrip;
                else data->rrpv = mRrip;

                ctx.updateVertexFromAddr(addr);
            } else if (ctx.loaded) {
                data->rrpv = legacy_sc ? 2 : mRrip;
            } else {
                data->rrpv = 2;
            }
        } else if (ecgMode == graph::ECGMode::ECG_COMBINED) {
            uint8_t dbgRrpv = mRrip;
            if (data->is_property_data && ctx.loaded) {
                uint32_t tier = ecgGraspTier(ctx, addr, llcSize);
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
            uint32_t tier = ecgGraspTier(ctx, addr, llcSize);
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
        data->ecg_epoch2 = 0;
        data->ecg_epoch_count = 0;
        data->ecg_epoch_valid = false;
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
    data->ecg_epoch2 = 0;
    data->ecg_epoch_count = 0;
    data->ecg_epoch_valid = false;
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
        // ECG_VARIANT factorial ablation (mirrors cache_sim findVictimECG).
        // Invariants in ALL variants: epoch is PROPERTY-ONLY; records evicted by
        // recency; unstamped property (epoch==0) -> recency (never "farthest").
        //   grasp_only(0): pure RRIP, no epoch
        //   epoch_first(1): records by recency, then farthest-epoch property (epoch vetoes rrpv)
        //   rrip_first(2,default): max-rrpv set (recency vetoes); records-first by recency,
        //                          then farthest-epoch property
        //   epoch_only(3): same eviction as epoch_first (insertion uniform, set in reset())
        //   shortcircuit(4,legacy): non-property first, then epoch among property
        static const int variant = [](){
            const char* v = std::getenv("ECG_VARIANT");
            if (!v) return 2;
            std::string s(v);
            if (s=="grasp_only")  return 0;
            if (s=="epoch_first") return 1;
            if (s=="rrip_first")  return 2;
            if (s=="epoch_only")  return 3;
            if (s=="shortcircuit"||s=="legacy") return 4;
            if (s=="degree_first"||s=="traversal") return 5;
            return 2;
        }();
        const uint32_t n = std::max<uint32_t>(1u, ctx.topology.num_vertices);
        const uint32_t ne = std::max<uint32_t>(2u, ctx.topology.edge_epoch_count);
        uint32_t curEpoch = static_cast<uint32_t>(
            (static_cast<uint64_t>(ctx.currentVertexForPopt()) * ne) / n);
        if (curEpoch >= ne) curEpoch = ne - 1;
        auto isProp  = [&](ReplaceableEntry* c){ return ctx.isPropertyData(getData(c)->line_addr); };
        auto dist    = [&](ReplaceableEntry* c){
            auto data = getData(c);
            return ecg_policy::epochPairDistance(
                data->ecg_epoch, data->ecg_epoch2,
                data->ecg_epoch_count, curEpoch, ne);
        };
        auto stamped = [&](ReplaceableEntry* c){ return isProp(c) && getData(c)->ecg_epoch_valid; };
        // ECG_EVICT_TRACE=N: emit the first N L3 evictions in cache_sim's
        // [EVICT L3 ...] format so scripts/.../verify_ecg.py asserts each victim
        // obeys the variant spec (one checker across all three simulators).
        // ECG_EVICT_TRACE_ROI=1 restricts the trace to evictions that occur AFTER the
        // kernel has begun its property traversal (hasCurrentVertexHint() — set by the
        // first GEM5_SET_VERTEX). Without it, the first N evictions are PRE-ROI graph
        // build + reorder traffic (no property stamped yet) which makes the trace
        // record/recency-only and the epoch-eviction path appear unexercised.
        static long ecgEvTrace = [](){ const char* e=std::getenv("ECG_EVICT_TRACE"); return e?std::atol(e):0L; }();
        static bool ecgEvRoi = std::getenv("ECG_EVICT_TRACE_ROI") != nullptr;
        const char* epol = (variant==1) ? "ECG:epoch_first" : "ECG:epoch_only";
        auto traced = [&](ReplaceableEntry* victimEntry, const char* pol, const char* reason)->ReplaceableEntry* {
            if (ecgEvTrace > 0 && (!ecgEvRoi || graph::hasCurrentVertexHint())) {
                --ecgEvTrace;
                int vidx = -1;
                for (size_t i=0;i<candidates.size();++i) if (candidates[i]==victimEntry){ vidx=(int)i; break; }
                std::cerr << "[EVICT L3 pol=" << pol << " curEpoch=" << curEpoch
                          << " set_ways=" << candidates.size() << "]\n";
                for (size_t i=0;i<candidates.size();++i){
                    auto dd = getData(candidates[i]);
                    std::cerr << "   way" << i << " valid=1 rrpv=" << (int)dd->rrpv
                              << " epoch=" << dd->ecg_epoch << " dist=" << dist(candidates[i])
                              << " prop=" << (int)isProp(candidates[i])
                              << " stamped=" << (int)(stamped(candidates[i]) ? 1 : 0)
                              << " dbg=" << (int)dd->ecg_dbg_tier
                              << " last=" << dd->lastTouchTick
                              << " epoch2=" << dd->ecg_epoch2
                              << " sched_n=" << (int)dd->ecg_epoch_count
                              << ((int)i==vidx ? "   <== VICTIM" : "") << "\n";
                }
                std::cerr << "   -> victim=way" << vidx << " reason=" << reason << "\n";
            }
            return victimEntry;
        };

        // Build the per-way state and delegate the DECISION to the shared
        // ecg_policy::selectVictim (identical across cache_sim / gem5 / Sniper).
        const size_t nc = candidates.size();
        ecg_policy::WayState ws[64];
        for (size_t i = 0; i < nc; i++) {
            auto dd = getData(candidates[i]);
            ws[i].prop    = isProp(candidates[i]);
            ws[i].rrpv    = dd->rrpv;
            ws[i].recency = dd->lastTouchTick;
            ws[i].dbg     = dd->ecg_dbg_tier;
            ws[i].dist    = dist(candidates[i]);
            ws[i].stamped = stamped(candidates[i]);
        }
        size_t vidx = ecg_policy::selectVictim(ws, nc, variant, rrpvMax);
        for (size_t i = 0; i < nc; i++) getData(candidates[i])->rrpv = ws[i].rrpv;  // persist SRRIP aging
        ReplaceableEntry* victim = candidates[vidx];

        // Reconstruct the trace pol/reason (verify_ecg.py keys on the pol name).
        const char* pol; const char* reason;
        if (variant == 0)      { pol = "ECG:grasp_only";  reason = "RRIP max-rrpv"; }
        else if (variant == 4) {
            if (!isProp(victim)) { pol = "ECG:shortcircuit";       reason = "first non-property"; }
            else                 { pol = "ECG:shortcircuit+epoch"; reason = "all-prop farthest epoch"; }
        } else if (variant == 2) {
            pol = "ECG:rrip_first";
            reason = !isProp(victim) ? "max-rrpv record by recency" : "max-rrpv farthest-epoch property";
        } else if (variant == 5) {
            pol = "ECG:degree_first";
            reason = !isProp(victim) ? "max-rrpv record by recency"
                                     : "max-rrpv coldest-degree then epoch";
        } else {
            pol = epol;
            reason = !isProp(victim) ? "record by recency"
                   : stamped(victim) ? "farthest-epoch property" : "recency fallback";
        }
        return traced(victim, pol, reason);
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
