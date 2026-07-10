#include "cache_set_popt.h"

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

}  // namespace

CacheSetPOPT::CacheSetPOPT(
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
   , m_replacement_pointer(0)
   , m_set_info(set_info)
   , m_srrip_tlb_enabled(Sim()->getCfg()->getBoolArray(cfgname + "/srrip/tlb_enabled", core_id))
   , m_context_load_attempted(false)
   , m_has_pending_insert(false)
   , m_pending_insert_addr(0)
   , m_sideband_path(envOrDefault("SNIPER_GRAPHBREW_CTX", "/tmp/sniper_graphbrew_ctx.json"))
   , m_popt_matrix_path(envOrDefault("SNIPER_POPT_MATRIX", "/tmp/sniper_popt_matrix.bin"))
{
   m_rrip_bits = new UInt8[m_associativity];
   m_line_addrs = new IntPtr[m_associativity];
   m_property_lines = new bool[m_associativity];
   for (UInt32 way = 0; way < m_associativity; way++) {
      m_rrip_bits[way] = m_rrip_insert;
      m_line_addrs[way] = 0;
      m_property_lines[way] = false;
   }
}

CacheSetPOPT::~CacheSetPOPT()
{
   delete [] m_rrip_bits;
   delete [] m_line_addrs;
   delete [] m_property_lines;
}

void
CacheSetPOPT::tryLoadContext()
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
CacheSetPOPT::prepareInsertion(IntPtr addr)
{
   tryLoadContext();
   m_pending_insert_addr = addr & ~(IntPtr(m_blocksize) - 1);
   m_has_pending_insert = true;
   graphbrew::sniper::globalContext().updateVertexFromAddr(m_pending_insert_addr, m_core_id);
}

void
CacheSetPOPT::applyPendingInsertion(UInt32 way)
{
   m_rrip_bits[way] = m_rrip_insert;
   if (m_has_pending_insert) {
      m_line_addrs[way] = m_pending_insert_addr;
      m_property_lines[way] = graphbrew::sniper::globalContext().isPropertyData(
            static_cast<uint64_t>(m_pending_insert_addr));
      m_has_pending_insert = false;
      return;
   }
   m_line_addrs[way] = 0;
   m_property_lines[way] = false;
}

UInt32
CacheSetPOPT::findSRRIPVictim(CacheCntlr *cntlr)
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
            LOG_ASSERT_ERROR(isValidReplacement(index), "POPT selected an invalid replacement candidate");
            return index;
         }
         m_replacement_pointer = (m_replacement_pointer + 1) % m_associativity;
      }

      for (UInt32 way = 0; way < m_associativity; way++) {
         if (m_rrip_bits[way] < m_rrip_max) m_rrip_bits[way]++;
      }
   }

   LOG_PRINT_ERROR("Error finding POPT replacement index");
}

UInt32
CacheSetPOPT::getReplacementIndex(CacheCntlr *cntlr)
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
   auto& context = graphbrew::sniper::globalContext();
   if (!context.loaded || !context.rereference.enabled) {
      return findSRRIPVictim(cntlr);
   }

   UInt32 property_count = 0;
   for (UInt32 way = 0; way < m_associativity; way++) {
      m_property_lines[way] = context.isPropertyData(static_cast<uint64_t>(m_line_addrs[way]));
      if (m_property_lines[way]) property_count++;
   }

   if (property_count != m_associativity) {
      for (UInt32 way = 0; way < m_associativity; way++) {
         if (!m_property_lines[way]) {
            applyPendingInsertion(way);
            LOG_ASSERT_ERROR(isValidReplacement(way), "POPT selected an invalid replacement candidate");
            return way;
         }
      }
   }

   // P-OPT eviction: evict the property line whose next reference (from the
   // rereference matrix, indexed by the current-vertex epoch clock) is farthest
   // in the future. This is byte-faithful to cache_sim findVictimPOPT
   // (Phase 1 non-property evict -> Phase 2 max next-ref distance -> Phase 3 RRIP
   // tiebreak) and gem5's ecg_rp: same makeOffsetMatrix builder, same
   // epoch_size/sub_epoch_size, same findNextRef, same cline_id mapping.
   //
   // NOTE (validated 2026-07): standalone POPT only produces a signal when the
   // L3 is actually exercised (property_bytes > effective LLC). On a small graph
   // at a tight geometry the property set fits in the inner caches, so the L3
   // sees only the cold-miss stream and EVERY policy (LRU/GRASP/POPT) reports
   // l3_miss_rate == 1.0 -- POPT looks "inert", but this is degenerate geometry,
   // not a matrix/lookup bug. roi_matrix flags these cells (l3_exercised=False,
   // "[warn] L3 inert"). With the L3 exercised (e.g. cit-Patents PR) standalone
   // Sniper POPT beats LRU by 3-15pp, matching cache_sim direction.
   UInt32 max_distance = 0;
   for (UInt32 way = 0; way < m_associativity; way++) {
      UInt32 distance = context.findNextRef(static_cast<uint64_t>(m_line_addrs[way]), m_core_id);
      max_distance = std::max(max_distance, std::min(distance, uint32_t(127)));
   }

   while (true) {
      for (UInt32 way = 0; way < m_associativity; way++) {
         UInt32 distance = context.findNextRef(static_cast<uint64_t>(m_line_addrs[way]), m_core_id);
         if (std::min(distance, uint32_t(127)) == max_distance && m_rrip_bits[way] >= m_rrip_max) {
            applyPendingInsertion(way);
            LOG_ASSERT_ERROR(isValidReplacement(way), "POPT selected an invalid replacement candidate");
            return way;
         }
      }
      for (UInt32 way = 0; way < m_associativity; way++) {
         UInt32 distance = context.findNextRef(static_cast<uint64_t>(m_line_addrs[way]), m_core_id);
         if (std::min(distance, uint32_t(127)) == max_distance && m_rrip_bits[way] < m_rrip_max) {
            m_rrip_bits[way]++;
         }
      }
   }
}

void
CacheSetPOPT::updateReplacementIndex(UInt32 accessed_index)
{
   m_set_info->increment(m_rrip_bits[accessed_index]);

   if (m_cache_block_info_array[accessed_index]->isPageTableBlock() && m_srrip_tlb_enabled) {
      m_rrip_bits[accessed_index] = 0;
      return;
   }

   m_rrip_bits[accessed_index] = 0;
   tryLoadContext();
   graphbrew::sniper::globalContext().updateVertexFromAddr(
         static_cast<uint64_t>(m_line_addrs[accessed_index]), m_core_id);
}