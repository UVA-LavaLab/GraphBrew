#include "cache_set_ecg.h"

#include "config.hpp"
#include "log.h"
#include "simulator.h"

#include <algorithm>
#include <cstdlib>

namespace {

const char* envOrDefault(const char* name, const char* fallback)
{
   const char* value = std::getenv(name);
   return value && value[0] ? value : fallback;
}

graphbrew::sniper::ECGMode modeFromEnv()
{
   return graphbrew::sniper::stringToECGMode(envOrDefault("SNIPER_ECG_MODE", "DBG_PRIMARY"));
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
   for (UInt32 way = 0; way < m_associativity; way++) {
      m_rrip_bits[way] = m_rrip_insert;
      m_dbg_tiers[way] = 0;
      m_popt_hints[way] = 0;
      m_line_addrs[way] = 0;
      m_property_lines[way] = false;
   }
   if (Sim()->getCfg()->hasKey(cfgname + "/cache_size", core_id)) {
      m_llc_size_bytes = UInt64(Sim()->getCfg()->getIntArray(cfgname + "/cache_size", core_id)) * k_KILO;
   }
}

CacheSetECG::~CacheSetECG()
{
   delete [] m_rrip_bits;
   delete [] m_dbg_tiers;
   delete [] m_popt_hints;
   delete [] m_line_addrs;
   delete [] m_property_lines;
}

void
CacheSetECG::tryLoadContext()
{
   auto& context = graphbrew::sniper::globalContext();
   if (context.loaded && context.rereference.enabled) return;
   if (m_context_load_attempted) return;
   m_context_load_attempted = true;
   context.setCacheLineSize(m_blocksize);
   if (!context.loaded) {
      context.loadFromSideband(m_sideband_path);
   }
   if (!context.rereference.enabled && context.loadRereferenceMatrix(m_popt_matrix_path) &&
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
   graphbrew::sniper::globalContext().updateVertexFromAddr(m_pending_insert_addr, m_core_id);
}

UInt8
CacheSetECG::graspInsertionRRPV(IntPtr addr) const
{
   const auto& context = graphbrew::sniper::globalContext();
   if (context.loaded && context.isPropertyData(static_cast<uint64_t>(addr))) {
      uint64_t llc_size = m_llc_size_bytes ? m_llc_size_bytes : UInt64(m_associativity) * m_blocksize;
      uint32_t tier = context.classifyGRASP(static_cast<uint64_t>(addr), llc_size);
      if (tier == 1) return 1;
      if (tier == 2) return m_rrip_max > 0 ? m_rrip_max - 1 : 0;
      return m_rrip_max;
   }
   return m_rrip_insert;
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
   uint32_t dist = context.findNextRef(static_cast<uint64_t>(addr), m_core_id);
   return static_cast<UInt8>(std::min(dist, uint32_t(127)) >> 3);
}

void
CacheSetECG::applyPendingInsertion(UInt32 way)
{
   if (m_has_pending_insert) {
      m_line_addrs[way] = m_pending_insert_addr;
      m_property_lines[way] = graphbrew::sniper::globalContext().isPropertyData(
            static_cast<uint64_t>(m_pending_insert_addr));
      m_dbg_tiers[way] = dbgTier(m_pending_insert_addr);
      m_popt_hints[way] = poptHint(m_pending_insert_addr);

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
      m_has_pending_insert = false;
      return;
   }

   m_rrip_bits[way] = m_rrip_insert;
   m_dbg_tiers[way] = graphbrew::sniper::globalContext().mask_config.num_buckets - 1;
   m_popt_hints[way] = 0;
   m_line_addrs[way] = 0;
   m_property_lines[way] = false;
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
   for (UInt32 way = 0; way < m_associativity; way++) {
      UInt32 distance = context.findNextRef(static_cast<uint64_t>(m_line_addrs[way]), m_core_id);
      max_distance = std::max(max_distance, std::min(distance, uint32_t(127)));
   }

   while (true) {
      UInt32 best = m_associativity;
      UInt8 best_dbg = 0;
      for (UInt32 way = 0; way < m_associativity; way++) {
         UInt32 distance = context.findNextRef(static_cast<uint64_t>(m_line_addrs[way]), m_core_id);
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
         UInt32 distance = context.findNextRef(static_cast<uint64_t>(m_line_addrs[way]), m_core_id);
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
               ? context.findNextRef(static_cast<uint64_t>(m_line_addrs[way]), m_core_id)
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
   if (m_cache_block_info_array[accessed_index]->isPageTableBlock() && m_srrip_tlb_enabled) {
      m_rrip_bits[accessed_index] = 0;
      return;
   }
   tryLoadContext();
   if (m_mode == graphbrew::sniper::ECGMode::POPT_PRIMARY ||
       m_mode == graphbrew::sniper::ECGMode::ECG_COMBINED) {
      m_rrip_bits[accessed_index] = 0;
      return;
   }
   if (m_property_lines[accessed_index] && graphbrew::sniper::globalContext().loaded) {
      uint64_t llc_size = m_llc_size_bytes ? m_llc_size_bytes : UInt64(m_associativity) * m_blocksize;
      uint32_t tier = graphbrew::sniper::globalContext().classifyGRASP(
            static_cast<uint64_t>(m_line_addrs[accessed_index]), llc_size);
      if (tier == 1) {
         m_rrip_bits[accessed_index] = 0;
         return;
      }
   }
   if (m_rrip_bits[accessed_index] > 0) m_rrip_bits[accessed_index]--;
}