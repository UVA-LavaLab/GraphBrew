#include "ecg_pfx_prefetcher.h"

#if __has_include("../../../config/config.hpp")
#include "../../../config/config.hpp"
#include "simulator.h"
#include "stats.h"
#else
template <class T>
void registerStatsMetric(String, UInt32, String, T*) {}
#endif
#include "core/memory_subsystem/cache/graph_cache_context_sniper.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>

namespace {

const char* envOrDefault(const char* name, const char* fallback)
{
   const char* value = std::getenv(name);
   return value && value[0] ? value : fallback;
}

}  // namespace

EcgPfxPrefetcher::EcgPfxPrefetcher(String configName, core_id_t core_id)
   : m_core_id(core_id)
   , m_cache_line_size(64)
   , m_recent_filter_size(256)
{
   (void)configName;
   registerStatsMetric("ecg-pfx-prefetcher", core_id, "sideband-loaded", &m_stat_sideband_loaded);
   registerStatsMetric("ecg-pfx-prefetcher", core_id, "target-hints-seen", &m_stat_target_hints_seen);
   registerStatsMetric("ecg-pfx-prefetcher", core_id, "issued", &m_stat_issued);
   registerStatsMetric("ecg-pfx-prefetcher", core_id, "duplicate-skips", &m_stat_duplicate_skips);
   registerStatsMetric("ecg-pfx-prefetcher", core_id, "no-sideband", &m_stat_no_sideband);
   registerStatsMetric("ecg-pfx-prefetcher", core_id, "invalid-target", &m_stat_invalid_target);
}

std::vector<IntPtr>
EcgPfxPrefetcher::getNextAddress(IntPtr current_address, core_id_t core_id,
      Core::mem_op_t mem_op_type, bool cache_hit, bool prefetch_hit, IntPtr eip)
{
   (void)current_address;
   (void)core_id;
   (void)mem_op_type;
   (void)cache_hit;
   (void)prefetch_hit;
   (void)eip;

   if (!m_sideband_loaded) {
      m_sideband_probe_count++;
      if (!m_sideband_tried || (m_sideband_probe_count & 0x3ff) == 0) {
         tryLoadSideband();
      }
   }

   std::vector<IntPtr> addresses;
   UInt32 target = 0;
   if (!graphbrew::sniper::consumePrefetchTargetHint(static_cast<UInt32>(m_core_id), target)) {
      return addresses;
   }
   m_stat_target_hints_seen++;

   if (!m_sideband_loaded || !m_property_configured) {
      m_stat_no_sideband++;
      return addresses;
   }

   IntPtr address = propertyAddress(target);
   if (!isPropertyAddress(address)) {
      m_stat_invalid_target++;
      return addresses;
   }

   IntPtr line = lineAddress(address);
   if (!shouldPrefetch(line)) {
      return addresses;
   }

   if (!m_reported_first_hint) {
      std::cerr << "SNIPER_ECG_PFX: first target vertex=" << target
                << " addr=0x" << std::hex << address
                << " property=[0x" << m_property_base << ",0x" << m_property_end
                << ")" << std::dec << std::endl;
      m_reported_first_hint = true;
   }

   addresses.push_back(line);
   m_stat_issued++;
   return addresses;
}

void
EcgPfxPrefetcher::tryLoadSideband()
{
   m_sideband_tried = true;
   const std::string path = envOrDefault("SNIPER_GRAPHBREW_CTX", "/tmp/sniper_graphbrew_ctx.json");
   std::ifstream file(path);
   if (!file.is_open()) return;

   std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

   std::string prop = findPreferredObject(content, "\"property_regions\"", "contrib");
   if (prop.empty()) prop = findPreferredObject(content, "\"property_regions\"", "scores");
   if (prop.empty()) prop = findPreferredObject(content, "\"property_regions\"", "dist");
   if (prop.empty()) prop = findPreferredObject(content, "\"property_regions\"", "parent");
   if (prop.empty()) return;

   UInt64 base = parseJsonUint(prop, "\"base\"");
   UInt64 size = parseJsonUint(prop, "\"size\"");
   UInt32 elem = static_cast<UInt32>(parseJsonUint(prop, "\"elem_size\""));
   if (base == 0 || size == 0 || elem == 0) return;

   m_property_base = static_cast<IntPtr>(base);
   m_property_end = static_cast<IntPtr>(base + size);
   m_property_elem_size = elem;
   m_property_configured = true;
   m_sideband_loaded = true;
   m_stat_sideband_loaded = 1;

   if (!m_reported_loaded) {
      std::cerr << "SNIPER_ECG_PFX: loaded sideband property=[0x" << std::hex
                << m_property_base << ",0x" << m_property_end << ") elem="
                << std::dec << m_property_elem_size << std::endl;
      m_reported_loaded = true;
   }
}

UInt64
EcgPfxPrefetcher::parseJsonUint(const std::string& json, const std::string& key)
{
   size_t pos = json.find(key);
   if (pos == std::string::npos) return 0;
   pos = json.find(':', pos);
   if (pos == std::string::npos) return 0;
   pos++;
   while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
   return std::strtoull(json.c_str() + pos, nullptr, 10);
}

std::string
EcgPfxPrefetcher::findPreferredObject(const std::string& json,
      const std::string& section, const std::string& preferred)
{
   size_t pos = json.find(section);
   if (pos == std::string::npos) return "";
   size_t arr_start = json.find('[', pos);
   size_t arr_end = json.find(']', arr_start);
   if (arr_start == std::string::npos || arr_end == std::string::npos) return "";

   std::string fallback;
   size_t obj_pos = arr_start;
   while ((obj_pos = json.find('{', obj_pos)) != std::string::npos && obj_pos < arr_end) {
      size_t obj_end = json.find('}', obj_pos);
      if (obj_end == std::string::npos || obj_end > arr_end) break;
      std::string obj = json.substr(obj_pos, obj_end - obj_pos + 1);
      if (fallback.empty()) fallback = obj;
      if (obj.find(preferred) != std::string::npos) return obj;
      obj_pos = obj_end + 1;
   }
   return fallback;
}

bool
EcgPfxPrefetcher::isPropertyAddress(IntPtr address) const
{
   return m_property_configured && address >= m_property_base && address < m_property_end;
}

IntPtr
EcgPfxPrefetcher::lineAddress(IntPtr address) const
{
   return address & ~(IntPtr(m_cache_line_size) - 1);
}

IntPtr
EcgPfxPrefetcher::propertyAddress(UInt32 vertex_id) const
{
   return m_property_base + IntPtr(UInt64(vertex_id) * m_property_elem_size);
}

bool
EcgPfxPrefetcher::shouldPrefetch(IntPtr address)
{
   IntPtr line = lineAddress(address);
   if (m_recent_prefetches.count(line)) {
      m_stat_duplicate_skips++;
      return false;
   }
   m_recent_prefetches.insert(line);
   m_recent_order.push_back(line);
   while (m_recent_order.size() > m_recent_filter_size) {
      m_recent_prefetches.erase(m_recent_order.front());
      m_recent_order.pop_front();
   }
   return true;
}