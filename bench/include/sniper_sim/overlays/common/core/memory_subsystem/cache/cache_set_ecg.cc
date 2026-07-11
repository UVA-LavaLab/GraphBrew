#include "cache_set_ecg.h"
#include "ecg_victim_policy.h"

#include "config.hpp"
#include "log.h"
#include "simulator.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>

namespace {

// ECG GRASP-tier SOURCE (ECG_GRASP_SRC), mirrors cache_sim/gem5 two variants:
//   mask (0, our ECG): DELIVERED per-vertex graspTierByIndex keyed by the INSERTED
//     LINE's own vertex (vertexForAddress) -- identical across simulators.
//   region (1, default, original GRASP): classifyGRASP(addr) spatial top-fraction.
template <typename Ctx>
inline uint32_t ecgGraspTier(const Ctx& ctx, uint64_t addr, uint64_t llc_size) {
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
        uint32_t vtx = ctx.vertexForAddress(addr);
        if (vtx == UINT32_MAX) return 3u;
        uint32_t es = ctx.propertyElemSizeForAddress(addr);
        return static_cast<uint32_t>(ecg_policy::graspTierByIndex(
            vtx, ctx.topology.num_vertices, ghf, es));
    }
    return ctx.classifyGRASP(addr, llc_size);  // REGION (GRASP)
}

const char* envOrDefault(const char* name, const char* fallback)
{
   const char* value = std::getenv(name);
   return value && value[0] ? value : fallback;
}

graphbrew::sniper::ECGMode modeFromEnv()
{
   return graphbrew::sniper::stringToECGMode(envOrDefault("SNIPER_ECG_MODE", "DBG_PRIMARY"));
}

bool sniperEcgExtractEnabled()
{
   static const bool enabled = []() {
      const char* value = std::getenv("SNIPER_ENABLE_ECG_EXTRACT");
      return value && value[0] && std::string(value) != "0";
   }();
   return enabled;
}

uint32_t requesterCoreOr(core_id_t fallback)
{
   uint32_t requester = graphbrew::sniper::currentNucaRequesterCore();
   if (requester < graphbrew::sniper::MAX_TRACKED_CORES) return requester;
   return static_cast<uint32_t>(fallback);
}

}  // namespace

CacheSetECG::CacheSetECG(
      String cfgname, core_id_t core_id,
      CacheBase::cache_t cache_type,
      UInt32 associativity, UInt32 blocksize,
      CacheSetInfoLRU* set_info, UInt8 num_attempts, bool is_tlb_set)
   : CacheSet(cache_type, associativity, blocksize, is_tlb_set)
   , m_cfgname(cfgname)
   , m_core_id(core_id)
   , m_rrip_numbits(Sim()->getCfg()->getIntArray(cfgname + "/srrip/bits", core_id))
   , m_rrip_max((1 << m_rrip_numbits) - 1)
   , m_rrip_insert(m_rrip_max - 1)
   , m_num_attempts(num_attempts)
   , m_mode(modeFromEnv())
   , m_access_tick(0)
   , m_replacement_pointer(0)
   , m_set_info(set_info)
   , m_srrip_tlb_enabled(Sim()->getCfg()->getBoolArray(cfgname + "/srrip/tlb_enabled", core_id))
   , m_context_load_attempted(false)
   , m_has_pending_insert(false)
   , m_pending_insert_addr(0)
   , m_llc_size_bytes(0)
   , m_sideband_path(envOrDefault("SNIPER_GRAPHBREW_CTX", "/tmp/sniper_graphbrew_ctx.json"))
   , m_popt_matrix_path(envOrDefault("SNIPER_POPT_MATRIX", "/tmp/sniper_popt_matrix.bin"))
{
   m_rrip_bits = new UInt8[m_associativity];
   m_dbg_tiers = new UInt8[m_associativity];
   m_popt_hints = new UInt8[m_associativity];
   m_line_addrs = new IntPtr[m_associativity];
   m_property_lines = new bool[m_associativity];
   m_ecg_epoch = new UInt16[m_associativity];
   m_ecg_epoch2 = new UInt16[m_associativity];
   m_ecg_epoch_count = new UInt8[m_associativity];
   m_ecg_epoch_valid = new bool[m_associativity];
   m_last_touch = new UInt64[m_associativity];
   for (UInt32 way = 0; way < m_associativity; way++) {
      m_rrip_bits[way] = m_rrip_insert;
      m_dbg_tiers[way] = 0;
      m_popt_hints[way] = 0;
      m_line_addrs[way] = 0;
      m_property_lines[way] = false;
      m_ecg_epoch[way] = 0;
      m_ecg_epoch2[way] = 0;
      m_ecg_epoch_count[way] = 0;
      m_ecg_epoch_valid[way] = false;
      m_last_touch[way] = 0;
   }
   if (Sim()->getCfg()->hasKey(cfgname + "/cache_size", core_id)) {
      m_llc_size_bytes = UInt64(Sim()->getCfg()->getIntArray(cfgname + "/cache_size", core_id)) * k_KILO;
   }
   // ECG-CONFIG proof banner (ECG_DEBUG=1): same format as cache_sim/gem5. CacheSetECG is
   // per-set, so a static one-shot guard prints exactly once per process.
   static bool ecg_announced = false;
   const char* dbg = std::getenv("ECG_DEBUG");
   if (!ecg_announced && dbg && *dbg && std::string(dbg) != "0") {
      ecg_announced = true;
      const char* m = std::getenv("SNIPER_ECG_MODE");
      const char* var = std::getenv("ECG_VARIANT");
      std::fprintf(stderr,
                   "[ECG-CONFIG sim=sniper policy=ECG mode=%s variant=%s llc=%lluB]\n",
                   m ? m : "DBG_PRIMARY", var ? var : "rrip_first",
                   (unsigned long long)m_llc_size_bytes);
   }
}

CacheSetECG::~CacheSetECG()
{
   delete [] m_rrip_bits;
   delete [] m_dbg_tiers;
   delete [] m_popt_hints;
   delete [] m_line_addrs;
   delete [] m_property_lines;
   delete [] m_ecg_epoch;
   delete [] m_ecg_epoch2;
   delete [] m_ecg_epoch_count;
   delete [] m_ecg_epoch_valid;
   delete [] m_last_touch;
}

void
CacheSetECG::tryLoadContext()
{
   auto& context = graphbrew::sniper::globalContext();
   if (context.loaded &&
       (sniperEcgExtractEnabled() || context.rereference.enabled)) return;
   if (m_context_load_attempted) return;
   m_context_load_attempted = true;
   context.setCacheLineSize(m_blocksize);
   if (!context.loaded) {
      context.loadFromSideband(m_sideband_path);
   }
   // The faithful ECG_GRASP_POPT path consumes the delivered per-edge epoch.
   // Load the live P-OPT oracle only for explicit non-delivery diagnostics.
   if (!sniperEcgExtractEnabled() && !context.rereference.enabled &&
       context.loadRereferenceMatrix(m_popt_matrix_path) &&
       context.num_regions > 0) {
      context.rereference.base_address = context.regions[0].base_address;
   }
}

void
CacheSetECG::prepareInsertion(IntPtr addr)
{
   tryLoadContext();
   m_pending_insert_addr = addr & ~(IntPtr(m_blocksize) - 1);
   m_has_pending_insert = true;
   graphbrew::sniper::globalContext().updateVertexFromAddr(
         m_pending_insert_addr, requesterCoreOr(m_core_id));
}

UInt8
CacheSetECG::graspInsertionRRPV(IntPtr addr) const
{
   const auto& context = graphbrew::sniper::globalContext();
   if (context.loaded && context.isPropertyData(static_cast<uint64_t>(addr))) {
      uint64_t llc_size = m_llc_size_bytes ? m_llc_size_bytes : UInt64(m_associativity) * m_blocksize;
      uint32_t tier = ecgGraspTier(context, static_cast<uint64_t>(addr), llc_size);
      if (tier == 1) return 1;
      if (tier == 2) return m_rrip_max > 0 ? m_rrip_max - 1 : 0;
      return m_rrip_max;
   }
   // ECG's insertion arm is GRASP: outside high/moderate regions -> M_RRIP.
   return m_rrip_max;
}

UInt8
CacheSetECG::dbgTier(IntPtr addr) const
{
   const auto& context = graphbrew::sniper::globalContext();
   uint32_t bucket = context.loaded ? context.classifyBucket(static_cast<uint64_t>(addr))
                                    : context.mask_config.num_buckets;
   if (bucket < context.mask_config.num_buckets) return static_cast<UInt8>(bucket);
   return context.mask_config.num_buckets > 0 ? context.mask_config.num_buckets - 1 : 0;
}

UInt8
CacheSetECG::poptHint(IntPtr addr) const
{
   const auto& context = graphbrew::sniper::globalContext();
   if (!context.loaded || !context.rereference.enabled ||
       !context.isPropertyData(static_cast<uint64_t>(addr))) {
      return 0;
   }
   uint32_t dist = context.findNextRef(
         static_cast<uint64_t>(addr), requesterCoreOr(m_core_id));
   return static_cast<UInt8>(std::min(dist, uint32_t(127)) >> 3);
}

bool
CacheSetECG::lookupLineEcgEpochPair(
      IntPtr line_addr, UInt16& first, UInt16& second, UInt8& count) const
{
   const auto& context = graphbrew::sniper::globalContext();
   if (!context.isEcgEpochData(static_cast<uint64_t>(line_addr)))
      return false;
   uint32_t v0 = context.vertexForAddress(static_cast<uint64_t>(line_addr));
   if (v0 == UINT32_MAX) return false;
   uint32_t requester_core = requesterCoreOr(m_core_id);
   if (requester_core >= graphbrew::sniper::MAX_TRACKED_CORES) return false;
   uint64_t sequence = 0;
   return graphbrew::sniper::lookupEcgEpochPair(
       requester_core, v0, first, second, count, sequence);
}

void
CacheSetECG::applyPendingInsertion(UInt32 way)
{
   if (m_has_pending_insert) {
      m_line_addrs[way] = m_pending_insert_addr;
      m_property_lines[way] = graphbrew::sniper::globalContext().isPropertyData(
            static_cast<uint64_t>(m_pending_insert_addr));
      m_dbg_tiers[way] =
         (m_mode == graphbrew::sniper::ECGMode::ECG_GRASP_POPT)
            ? static_cast<UInt8>(ecgGraspTier(
                 graphbrew::sniper::globalContext(),
                 static_cast<uint64_t>(m_pending_insert_addr),
                 m_llc_size_bytes ? m_llc_size_bytes
                                  : UInt64(m_associativity) * m_blocksize))
            : dbgTier(m_pending_insert_addr);
      m_popt_hints[way] = poptHint(m_pending_insert_addr);

      // SNIPER_ECG_EXTRACT fill-stamp: the demand for this vertex delivered its
      // next-ref epoch (recordEcgEpoch); stamp the line so the eviction can rank
      // it by the delivered (HW-faithful) epoch instead of the findNextRef matrix.
      m_ecg_epoch[way] = 0;
      m_ecg_epoch2[way] = 0;
      m_ecg_epoch_count[way] = 0;
      m_ecg_epoch_valid[way] = false;
      if (m_property_lines[way]) {
         UInt16 first = 0, second = 0;
         UInt8 count = 0;
         if (lookupLineEcgEpochPair(
                 m_pending_insert_addr, first, second, count)) {
            m_ecg_epoch[way] = first;
            m_ecg_epoch2[way] = second;
            m_ecg_epoch_count[way] = count;
            m_ecg_epoch_valid[way] = true;
         }
      }

      if (m_mode == graphbrew::sniper::ECGMode::POPT_PRIMARY) {
         m_rrip_bits[way] = m_rrip_insert;
      } else if (m_mode == graphbrew::sniper::ECGMode::ECG_COMBINED) {
         UInt8 dbg_rrpv = graspInsertionRRPV(m_pending_insert_addr);
         UInt8 popt_rrpv = static_cast<UInt8>((UInt32(m_popt_hints[way]) * m_rrip_max) / 15u);
         UInt8 combined = static_cast<UInt8>((UInt32(dbg_rrpv) + UInt32(popt_rrpv)) / 2u);
         if (combined == 0 && dbg_rrpv > 0) combined = 1;
         m_rrip_bits[way] = std::min<UInt8>(combined, m_rrip_max);
      } else {
         m_rrip_bits[way] = graspInsertionRRPV(m_pending_insert_addr);
      }
      m_last_touch[way] = ++m_access_tick;
      m_has_pending_insert = false;
      return;
   }

   m_rrip_bits[way] = m_rrip_max;
   m_dbg_tiers[way] = graphbrew::sniper::globalContext().mask_config.num_buckets - 1;
   m_popt_hints[way] = 0;
   m_line_addrs[way] = 0;
   m_property_lines[way] = false;
   m_ecg_epoch[way] = 0;
   m_ecg_epoch2[way] = 0;
   m_ecg_epoch_count[way] = 0;
   m_ecg_epoch_valid[way] = false;
   m_last_touch[way] = ++m_access_tick;
}

UInt32
CacheSetECG::findSRRIPVictim(CacheCntlr *cntlr)
{
   UInt8 attempt = 0;
   for (UInt32 age_round = 0; age_round <= m_rrip_max; ++age_round) {
      for (UInt32 probe = 0; probe < m_associativity; probe++) {
         if (m_rrip_bits[m_replacement_pointer] >= m_rrip_max) {
            UInt8 index = m_replacement_pointer;
            bool qbs_reject = false;
            bool attempt_goforit = false;
            if (attempt < m_num_attempts - 1) {
               LOG_ASSERT_ERROR(cntlr != NULL, "CacheCntlr == NULL, QBS can only be used when cntlr is passed in");
               qbs_reject = cntlr->isInLowerLevelCache(m_cache_block_info_array[index]);
               attempt_goforit = true;
            }
            if (qbs_reject) {
               m_rrip_bits[index] = 0;
               cntlr->incrementQBSLookupCost();
               ++attempt;
               continue;
            }
            if (m_cache_block_info_array[index]->isPageTableBlock() &&
                m_srrip_tlb_enabled && attempt_goforit) {
               m_rrip_bits[index] = 0;
               cntlr->incrementQBSLookupCost();
               ++attempt;
               continue;
            }
            m_replacement_pointer = (m_replacement_pointer + 1) % m_associativity;
            applyPendingInsertion(index);
            m_set_info->incrementAttempt(attempt);
            LOG_ASSERT_ERROR(isValidReplacement(index), "ECG selected an invalid replacement candidate");
            return index;
         }
         m_replacement_pointer = (m_replacement_pointer + 1) % m_associativity;
      }
      for (UInt32 way = 0; way < m_associativity; way++) {
         if (m_rrip_bits[way] < m_rrip_max) m_rrip_bits[way]++;
      }
   }
   LOG_PRINT_ERROR("Error finding ECG replacement index");
}

UInt32
CacheSetECG::findPOPTVictim(CacheCntlr *cntlr)
{
   auto& context = graphbrew::sniper::globalContext();
   if (!context.loaded || !context.rereference.enabled) return findSRRIPVictim(cntlr);

   UInt32 property_count = 0;
   for (UInt32 way = 0; way < m_associativity; way++) {
      m_property_lines[way] = context.isPropertyData(static_cast<uint64_t>(m_line_addrs[way]));
      if (m_property_lines[way]) property_count++;
   }

   if (property_count != m_associativity) {
      for (UInt32 way = 0; way < m_associativity; way++) {
         if (!m_property_lines[way]) {
            applyPendingInsertion(way);
            LOG_ASSERT_ERROR(isValidReplacement(way), "ECG POPT selected an invalid replacement candidate");
            return way;
         }
      }
   }

   UInt32 max_distance = 0;
   uint32_t requester_core = requesterCoreOr(m_core_id);
   for (UInt32 way = 0; way < m_associativity; way++) {
      UInt32 distance = context.findNextRef(
            static_cast<uint64_t>(m_line_addrs[way]), requester_core);
      max_distance = std::max(max_distance, std::min(distance, uint32_t(127)));
   }

   while (true) {
      // POPT_PRIMARY (ECG arm): P-OPT max-reref-distance eviction, with the ECG
      // degree (DBG) tiebreak among reref-ties — prefer the highest DBG tier.
      // This is the ECG contribution (P-OPT + degree) that makes POPT_PRIMARY
      // BEAT plain POPT, identical to cache_sim (cache_sim.h findVictimECG
      // POPT_PRIMARY) and gem5 (ecg_rp.cc). Plain CacheSetPOPT is the parity arm.
      UInt32 best = m_associativity;
      UInt8 best_dbg = 0;
      for (UInt32 way = 0; way < m_associativity; way++) {
         UInt32 distance = context.findNextRef(
               static_cast<uint64_t>(m_line_addrs[way]), requester_core);
         if (std::min(distance, uint32_t(127)) == max_distance && m_rrip_bits[way] >= m_rrip_max &&
             (best == m_associativity || m_dbg_tiers[way] > best_dbg)) {
            best = way;
            best_dbg = m_dbg_tiers[way];
         }
      }
      if (best != m_associativity) {
         applyPendingInsertion(best);
         LOG_ASSERT_ERROR(isValidReplacement(best), "ECG POPT selected an invalid replacement candidate");
         return best;
      }
      for (UInt32 way = 0; way < m_associativity; way++) {
         UInt32 distance = context.findNextRef(
               static_cast<uint64_t>(m_line_addrs[way]), requester_core);
         if (std::min(distance, uint32_t(127)) == max_distance && m_rrip_bits[way] < m_rrip_max) {
            m_rrip_bits[way]++;
         }
      }
   }
}

UInt32
CacheSetECG::findDBGPrimaryVictim(CacheCntlr *cntlr)
{
   (void)cntlr;
   while (true) {
      uint32_t requester_core = requesterCoreOr(m_core_id);
      UInt32 max_count = 0;
      UInt8 max_dbg = 0;
      for (UInt32 way = 0; way < m_associativity; way++) {
         if (m_rrip_bits[way] >= m_rrip_max) {
            max_count++;
            max_dbg = std::max(max_dbg, m_dbg_tiers[way]);
         }
      }
      if (max_count > 0) {
         UInt32 best = m_associativity;
         UInt32 best_dist = 0;
         auto& context = graphbrew::sniper::globalContext();
         for (UInt32 way = 0; way < m_associativity; way++) {
            if (m_rrip_bits[way] < m_rrip_max || m_dbg_tiers[way] != max_dbg) continue;
            UInt32 dist = context.rereference.enabled
               ? context.findNextRef(
                    static_cast<uint64_t>(m_line_addrs[way]), requester_core)
               : 0;
            if (best == m_associativity || dist > best_dist) {
               best = way;
               best_dist = dist;
            }
         }
         applyPendingInsertion(best);
         LOG_ASSERT_ERROR(isValidReplacement(best), "ECG DBG selected an invalid replacement candidate");
         return best;
      }
      for (UInt32 way = 0; way < m_associativity; way++) {
         if (m_rrip_bits[way] < m_rrip_max) m_rrip_bits[way]++;
      }
   }
}

UInt32
CacheSetECG::findECGEmbeddedVictim(CacheCntlr *cntlr)
{
   (void)cntlr;
   // ECG_EMBEDDED: among the max-RRPV lines, evict the one with the highest
   // stored P-OPT hint (furthest predicted reuse); tie-break by highest DBG
   // tier. Mirrors gem5 GraphEcgRP ECG_EMBEDDED (ecg_rp.cc:296-317) and
   // cache_sim findVictimECG. Uses the per-line hint captured at insertion
   // (m_popt_hints), so it carries zero extra LLC lookup cost.
   while (true) {
      UInt32 max_count = 0;
      UInt8 max_hint = 0;
      for (UInt32 way = 0; way < m_associativity; way++) {
         if (m_rrip_bits[way] >= m_rrip_max) {
            max_count++;
            max_hint = std::max(max_hint, m_popt_hints[way]);
         }
      }
      if (max_count > 0) {
         UInt32 best = m_associativity;
         UInt8 best_dbg = 0;
         for (UInt32 way = 0; way < m_associativity; way++) {
            if (m_rrip_bits[way] < m_rrip_max || m_popt_hints[way] != max_hint) continue;
            if (best == m_associativity || m_dbg_tiers[way] > best_dbg) {
               best = way;
               best_dbg = m_dbg_tiers[way];
            }
         }
         applyPendingInsertion(best);
         LOG_ASSERT_ERROR(isValidReplacement(best), "ECG EMBEDDED selected an invalid replacement candidate");
         return best;
      }
      for (UInt32 way = 0; way < m_associativity; way++) {
         if (m_rrip_bits[way] < m_rrip_max) m_rrip_bits[way]++;
      }
   }
}

UInt32
CacheSetECG::findECGGraspPoptVictim(CacheCntlr *cntlr)
{
   // ECG_GRASP_POPT factorial ablation (ECG_VARIANT) — Sniper analog of
   // cache_sim findVictimECG / gem5 ecg_rp.cc getVictim. The paper path consumes
   // the delivered per-edge epoch (SNIPER_ENABLE_ECG_EXTRACT=1), stores it per
   // line on fill/L3 hit, and uses true per-set access recency. Native
   // findNextRef remains only as an explicit non-delivery diagnostic fallback.
   // Invariants in ALL variants: epoch is PROPERTY-ONLY; record lines are
   // ordered by recency.
   //   grasp_only(0): pure RRIP, no next-ref
   //   epoch_first(1): records by recency, then farthest-next-ref property
   //   rrip_first(2,default): max-rrpv set; records-first, then farthest property
   //   epoch_only(3): same eviction as epoch_first
   //   shortcircuit(4,legacy): non-property first, then farthest property
   static const int variant = [](){
      const char* v = std::getenv("ECG_VARIANT");
      if (!v) return 2;
      std::string s(v);
      if (s == "grasp_only")   return 0;
      if (s == "epoch_first")  return 1;
      if (s == "rrip_first")   return 2;
      if (s == "epoch_only")   return 3;
      if (s == "shortcircuit" || s == "legacy") return 4;
      if (s == "degree_first" || s == "traversal") return 5;
      return 2;
   }();

   auto& context = graphbrew::sniper::globalContext();
   // SNIPER_ECG_EXTRACT: rank property lines by the DELIVERED per-edge epoch
   // (HW-faithful, matching gem5/cache_sim) instead of the host-side findNextRef
   // matrix. cur_ep formula mirrors ecg_epoch::currentEpoch (snipersim build has
   // no bench/include path; the kernel side uses the shared helper).
   const bool fatLoad = sniperEcgExtractEnabled();

   // grasp_only, or no usable next-ref signal -> pure RRIP. In fat-load mode the
   // signal is the delivered epoch (needs only the property region, not the matrix).
   if (variant == 0 || !context.loaded || (!fatLoad && !context.rereference.enabled))
      return findSRRIPVictim(cntlr);

   const uint32_t ne = context.edge_epoch_count ? context.edge_epoch_count : 256u;
   const uint32_t N = context.topology.num_vertices;
   uint32_t requester_core = requesterCoreOr(m_core_id);
   const uint32_t srcv = context.currentVertexForPopt(requester_core);
   const uint32_t cur_ep = (N > 0)
      ? static_cast<uint32_t>((static_cast<uint64_t>(srcv) * ne) / N) : 0u;

   for (UInt32 way = 0; way < m_associativity; way++) {
      m_property_lines[way] =
         context.isEcgEpochData(static_cast<uint64_t>(m_line_addrs[way]));
   }
   auto isProp = [&](UInt32 w) { return m_property_lines[w]; };
   auto dist   = [&](UInt32 w) -> uint32_t {
      if (fatLoad) {
         return ecg_policy::epochPairDistance(
               m_ecg_epoch[w], m_ecg_epoch2[w],
               m_ecg_epoch_count[w], cur_ep, ne);
      }
      return std::min(context.findNextRef(
                          static_cast<uint64_t>(m_line_addrs[w]), requester_core),
                      uint32_t(127));
   };
   auto stamped = [&](UInt32 w) { return isProp(w) && (!fatLoad || m_ecg_epoch_valid[w]); };
   // ECG_EVICT_TRACE=N: emit the first N L3 evictions in cache_sim's
   // [EVICT L3 ...] format so scripts/.../verify_ecg.py asserts each victim
   // obeys the variant spec. The faithful path prints the real stored epoch,
   // computed current epoch, delivery-valid bit, and true recency.
   static long ecgEvTrace = [](){ const char* e = std::getenv("ECG_EVICT_TRACE"); return e ? std::atol(e) : 0L; }();
   static long ecgEvTraceSkip = [](){
      const char* e = std::getenv("ECG_EVICT_TRACE_SKIP");
      return e ? std::max(0L, std::atol(e)) : 0L;
   }();
   static const bool ecgEvTraceRoi = [](){
      const char* e = std::getenv("ECG_EVICT_TRACE_ROI");
      return e && e[0] && std::string(e) != "0";
   }();
   const char* epol = (variant == 1) ? "ECG:epoch_first" : "ECG:epoch_only";
   auto emitTrace = [&](UInt32 victimWay, const char* pol, const char* reason) {
      if (ecgEvTrace <= 0) return;
      // Shared NUCA sets do not carry a normal application core ID. Gate on
      // whether any application core has entered its vertex loop instead.
      if (ecgEvTraceRoi && !graphbrew::sniper::hasAnyCurrentVertexHint()) return;
      if (ecgEvTraceSkip > 0) {
         --ecgEvTraceSkip;
         return;
      }
      --ecgEvTrace;
      std::fprintf(stderr, "[EVICT L3 pol=%s curEpoch=%u set_ways=%u]\n",
                   pol, cur_ep, m_associativity);
      for (UInt32 w = 0; w < m_associativity; w++) {
         UInt32 d = isProp(w) ? dist(w) : 0;
         UInt32 epoch = (fatLoad && isProp(w)) ? m_ecg_epoch[w] : (isProp(w) ? 1u : 0u);
         std::fprintf(stderr, "   way%u valid=1 rrpv=%d epoch=%d dist=%u prop=%d stamped=%d dbg=%d last=%llu epoch2=%u sched_n=%u%s\n",
                      w, (int)m_rrip_bits[w], (int)epoch, d, (int)isProp(w),
                      (int)(stamped(w) ? 1 : 0), (int)m_dbg_tiers[w],
                      (unsigned long long)m_last_touch[w],
                      (unsigned)m_ecg_epoch2[w],
                      (unsigned)m_ecg_epoch_count[w],
                      w == victimWay ? "   <== VICTIM" : "");
      }
      std::fprintf(stderr, "   -> victim=way%u reason=%s\n", victimWay, reason);
   };

   // Build the per-way state and delegate the DECISION to the shared
   // ecg_policy::selectVictim (identical across cache_sim / gem5 / Sniper).
   // True recency is tracked explicitly so records tied at max RRPV are ordered
   // exactly like cache_sim (last_access) and gem5 (lastTouchTick).
   ecg_policy::WayState ws[64];
   for (UInt32 w = 0; w < m_associativity; w++) {
      ws[w].prop    = isProp(w);
      ws[w].rrpv    = m_rrip_bits[w];
      ws[w].recency = m_last_touch[w];
      ws[w].dbg     = m_dbg_tiers[w];
      ws[w].dist    = dist(w);
      ws[w].stamped = stamped(w);
   }
   size_t vidx = ecg_policy::selectVictim(ws, m_associativity, variant, m_rrip_max);
   for (UInt32 w = 0; w < m_associativity; w++) m_rrip_bits[w] = ws[w].rrpv;  // persist SRRIP aging
   UInt32 victimWay = static_cast<UInt32>(vidx);

   // Reconstruct the trace pol/reason (verify_ecg.py keys on the pol name).
   const char* pol; const char* reason;
   if (variant == 4) {
      if (!isProp(victimWay)) { pol = "ECG:shortcircuit";       reason = "first non-property"; }
      else                    { pol = "ECG:shortcircuit+epoch"; reason = "all-prop farthest epoch"; }
   } else if (variant == 2) {
      pol = "ECG:rrip_first";
      reason = !isProp(victimWay) ? "max-rrpv record by recency" : "max-rrpv farthest-epoch property";
   } else if (variant == 5) {
      pol = "ECG:degree_first";
      reason = !isProp(victimWay) ? "max-rrpv record by recency"
                                  : "max-rrpv coldest-degree then epoch";
   } else {
      pol = epol;
      reason = !isProp(victimWay) ? "record by recency" : "farthest-epoch property";
   }
   emitTrace(victimWay, pol, reason);
   applyPendingInsertion(victimWay);
   LOG_ASSERT_ERROR(isValidReplacement(victimWay), "ECG GRASP_POPT shared-fn invalid");
   return victimWay;
}

UInt32
CacheSetECG::getReplacementIndex(CacheCntlr *cntlr)
{
   for (UInt32 way = 0; way < m_associativity; way++) {
      if (!m_cache_block_info_array[way]->isValid()) {
         applyPendingInsertion(way);
         if (m_cache_block_info_array[way]->isPageTableBlock() && m_srrip_tlb_enabled) {
            m_rrip_bits[way] = 0;
         }
         return way;
      }
   }
   tryLoadContext();
   if (m_mode == graphbrew::sniper::ECGMode::POPT_PRIMARY) return findPOPTVictim(cntlr);
   if (m_mode == graphbrew::sniper::ECGMode::ECG_GRASP_POPT) return findECGGraspPoptVictim(cntlr);
   if (m_mode == graphbrew::sniper::ECGMode::ECG_EMBEDDED) return findECGEmbeddedVictim(cntlr);
   if (m_mode == graphbrew::sniper::ECGMode::DBG_ONLY ||
       m_mode == graphbrew::sniper::ECGMode::ECG_COMBINED) {
      return findSRRIPVictim(cntlr);
   }
   return findDBGPrimaryVictim(cntlr);
}

void
CacheSetECG::updateReplacementIndex(UInt32 accessed_index)
{
   m_set_info->increment(m_rrip_bits[accessed_index]);
   m_last_touch[accessed_index] = ++m_access_tick;
   if (m_cache_block_info_array[accessed_index]->isPageTableBlock() && m_srrip_tlb_enabled) {
      m_rrip_bits[accessed_index] = 0;
      return;
   }
   tryLoadContext();
   if (m_mode == graphbrew::sniper::ECGMode::POPT_PRIMARY ||
       m_mode == graphbrew::sniper::ECGMode::ECG_COMBINED) {
      m_rrip_bits[accessed_index] = 0;
      // Advance the P-OPT vertex pointer on hit, exactly like standalone
      // CacheSetPOPT::updateReplacementIndex — otherwise POPT_PRIMARY and POPT
      // see different current vertices and diverge.
      graphbrew::sniper::globalContext().updateVertexFromAddr(
            static_cast<uint64_t>(m_line_addrs[accessed_index]),
            requesterCoreOr(m_core_id));
      return;
   }
   // Hardware-feasible epoch refresh: the access reached this L3 set, so update
   // only the line that was actually touched. Do not broadcast inner-cache hits
   // to every resident L3 line at eviction time.
   if (m_mode == graphbrew::sniper::ECGMode::ECG_GRASP_POPT &&
       sniperEcgExtractEnabled() && m_property_lines[accessed_index]) {
      UInt16 first = 0, second = 0;
      UInt8 count = 0;
      if (lookupLineEcgEpochPair(
              m_line_addrs[accessed_index], first, second, count)) {
         m_ecg_epoch[accessed_index] = first;
         m_ecg_epoch2[accessed_index] = second;
         m_ecg_epoch_count[accessed_index] = count;
         m_ecg_epoch_valid[accessed_index] = true;
      }
   }
   if (m_property_lines[accessed_index] && graphbrew::sniper::globalContext().loaded) {
      uint64_t llc_size = m_llc_size_bytes ? m_llc_size_bytes : UInt64(m_associativity) * m_blocksize;
      uint32_t tier = ecgGraspTier(graphbrew::sniper::globalContext(), static_cast<uint64_t>(m_line_addrs[accessed_index]), llc_size);
      if (tier == 1) {
         m_rrip_bits[accessed_index] = 0;
         return;
      }
   }
   if (m_rrip_bits[accessed_index] > 0) m_rrip_bits[accessed_index]--;
}