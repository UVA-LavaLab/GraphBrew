#include "graph_cache_context_sniper.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>

namespace graphbrew {
namespace sniper {

namespace {

std::array<std::atomic<uint32_t>, MAX_TRACKED_CORES>& vertexStorage()
{
    static std::array<std::atomic<uint32_t>, MAX_TRACKED_CORES> storage{};
    return storage;
}

std::array<std::atomic<bool>, MAX_TRACKED_CORES>& vertexValidStorage()
{
    static std::array<std::atomic<bool>, MAX_TRACKED_CORES> storage{};
    return storage;
}

std::array<std::atomic<uint32_t>, MAX_TRACKED_CORES>& prefetchTargetStorage()
{
    static std::array<std::atomic<uint32_t>, MAX_TRACKED_CORES> storage{};
    return storage;
}

std::array<std::atomic<bool>, MAX_TRACKED_CORES>& prefetchTargetValidStorage()
{
    static std::array<std::atomic<bool>, MAX_TRACKED_CORES> storage{};
    return storage;
}

uint32_t clampVertex(uint64_t vertex)
{
    return vertex > std::numeric_limits<uint32_t>::max()
        ? std::numeric_limits<uint32_t>::max()
        : static_cast<uint32_t>(vertex);
}

uint64_t parseJsonUint(const std::string& json, const std::string& key)
{
    size_t pos = json.find(key);
    if (pos == std::string::npos) return 0;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return 0;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    return std::strtoull(json.c_str() + pos, nullptr, 10);
}

std::string parseJsonString(const std::string& json, const std::string& key)
{
    size_t pos = json.find(key);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos);
    if (pos == std::string::npos) return "";
    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return "";
    size_t end = json.find('"', pos + 1);
    if (end == std::string::npos) return "";
    return json.substr(pos + 1, end - pos - 1);
}

bool parseJsonBool(const std::string& json, const std::string& key)
{
    size_t pos = json.find(key);
    if (pos == std::string::npos) return false;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return false;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    return json.compare(pos, 4, "true") == 0;
}

}  // namespace

void setCurrentVertexHint(uint32_t core_id, uint64_t vertex)
{
    if (core_id >= MAX_TRACKED_CORES) return;
    vertexStorage()[core_id].store(clampVertex(vertex), std::memory_order_release);
    vertexValidStorage()[core_id].store(true, std::memory_order_release);
}

bool hasCurrentVertexHint(uint32_t core_id)
{
    return core_id < MAX_TRACKED_CORES &&
        vertexValidStorage()[core_id].load(std::memory_order_acquire);
}

uint32_t getCurrentVertexHint(uint32_t core_id)
{
    if (core_id >= MAX_TRACKED_CORES) return 0;
    return vertexStorage()[core_id].load(std::memory_order_acquire);
}

void clearCurrentVertexHint(uint32_t core_id)
{
    if (core_id >= MAX_TRACKED_CORES) return;
    vertexValidStorage()[core_id].store(false, std::memory_order_release);
}

void setPrefetchTargetHint(uint32_t core_id, uint64_t vertex)
{
    if (core_id >= MAX_TRACKED_CORES) return;
    prefetchTargetStorage()[core_id].store(clampVertex(vertex), std::memory_order_release);
    prefetchTargetValidStorage()[core_id].store(true, std::memory_order_release);
}

bool hasPrefetchTargetHint(uint32_t core_id)
{
    return core_id < MAX_TRACKED_CORES &&
        prefetchTargetValidStorage()[core_id].load(std::memory_order_acquire);
}

uint32_t getPrefetchTargetHint(uint32_t core_id)
{
    if (core_id >= MAX_TRACKED_CORES) return 0;
    return prefetchTargetStorage()[core_id].load(std::memory_order_acquire);
}

bool consumePrefetchTargetHint(uint32_t core_id, uint32_t& vertex)
{
    if (core_id >= MAX_TRACKED_CORES) return false;
    if (!prefetchTargetValidStorage()[core_id].exchange(false, std::memory_order_acq_rel)) {
        return false;
    }
    vertex = prefetchTargetStorage()[core_id].load(std::memory_order_acquire);
    return true;
}

void clearPrefetchTargetHint(uint32_t core_id)
{
    if (core_id >= MAX_TRACKED_CORES) return;
    prefetchTargetValidStorage()[core_id].store(false, std::memory_order_release);
}

ECGMode stringToECGMode(const std::string& text)
{
    if (text == "POPT_PRIMARY" || text == "popt_primary" || text == "popt") return ECGMode::POPT_PRIMARY;
    if (text == "DBG_ONLY" || text == "dbg_only" || text == "dbg") return ECGMode::DBG_ONLY;
    if (text == "ECG_EMBEDDED" || text == "ecg_embedded" || text == "embedded") return ECGMode::ECG_EMBEDDED;
    if (text == "ECG_COMBINED" || text == "ecg_combined" || text == "combined") return ECGMode::ECG_COMBINED;
    return ECGMode::DBG_PRIMARY;
}

std::string ecgModeToString(ECGMode mode)
{
    switch (mode) {
      case ECGMode::DBG_PRIMARY:  return "DBG_PRIMARY";
      case ECGMode::POPT_PRIMARY: return "POPT_PRIMARY";
      case ECGMode::DBG_ONLY:     return "DBG_ONLY";
      case ECGMode::ECG_EMBEDDED: return "ECG_EMBEDDED";
      case ECGMode::ECG_COMBINED: return "ECG_COMBINED";
      default:                    return "UNKNOWN";
    }
}

bool PropertyRegion::contains(uint64_t addr) const
{
    return addr >= base_address && addr < upper_bound;
}

uint32_t PropertyRegion::classifyBucket(uint64_t addr) const
{
    if (!contains(addr) || num_buckets == 0) return num_buckets;
    for (uint32_t bucket = 0; bucket < num_buckets; ++bucket) {
        if (addr < bucket_bounds[bucket]) return bucket;
    }
    return num_buckets - 1;
}

bool EdgeRegion::contains(uint64_t addr) const
{
    return addr >= base_address && addr < upper_bound;
}

bool RereferenceMatrix::loadFromFile(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;

    file.read(reinterpret_cast<char*>(&num_epochs), sizeof(num_epochs));
    file.read(reinterpret_cast<char*>(&num_cache_lines), sizeof(num_cache_lines));
    file.read(reinterpret_cast<char*>(&epoch_size), sizeof(epoch_size));
    file.read(reinterpret_cast<char*>(&sub_epoch_size), sizeof(sub_epoch_size));
    if (!file || num_epochs == 0 || num_cache_lines == 0) {
        enabled = false;
        return false;
    }

    size_t matrix_size = static_cast<size_t>(num_epochs) * num_cache_lines;
    if (num_cache_lines != 0 && matrix_size / num_cache_lines != num_epochs) {
        enabled = false;
        return false;
    }
    data.assign(matrix_size, 0);
    file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(matrix_size));
    enabled = static_cast<size_t>(file.gcount()) == matrix_size;
    return enabled;
}

uint32_t RereferenceMatrix::findNextRef(uint32_t cline_id, uint32_t current_vertex) const
{
    if (!enabled || cline_id >= num_cache_lines) return 127;
    uint32_t epoch_id = epoch_size > 0 ? current_vertex / epoch_size : 0;
    if (epoch_id >= num_epochs) return 127;

    uint8_t entry = data[static_cast<size_t>(epoch_id) * num_cache_lines + cline_id];
    constexpr uint8_t OR_MASK = 0x80;
    constexpr uint8_t AND_MASK = 0x7F;

    if ((entry & OR_MASK) != 0) {
        uint8_t last_ref_sub_epoch = entry & AND_MASK;
        uint32_t current_sub_epoch = sub_epoch_size > 0
            ? ((current_vertex % epoch_size) / sub_epoch_size)
            : 0;
        if (current_sub_epoch <= last_ref_sub_epoch) return 0;
        if (epoch_id + 1 < num_epochs) {
            uint8_t next_entry = data[static_cast<size_t>(epoch_id + 1) * num_cache_lines + cline_id];
            if ((next_entry & OR_MASK) != 0) return 1;
            uint8_t reref = next_entry & AND_MASK;
            return reref < 127 ? reref + 1 : 127;
        }
        return 127;
    }
    return entry & AND_MASK;
}

uint32_t RereferenceMatrix::findNextRefByAddr(uint64_t addr, uint32_t current_vertex) const
{
    if (!enabled || addr < base_address) return 127;
    uint32_t cline_id = static_cast<uint32_t>((addr - base_address) / cache_line_size);
    return findNextRef(cline_id, current_vertex);
}

void MaskConfig::computeShifts()
{
    prefetch_shift = 0;
    popt_shift = prefetch_bits;
    dbg_shift = prefetch_bits + popt_bits;
    prefetch_mask_val = prefetch_bits ? ((1U << prefetch_bits) - 1) : 0;
    popt_mask_val = popt_bits ? (((1U << popt_bits) - 1) << popt_shift) : 0;
    dbg_mask_val = dbg_bits ? (((1U << dbg_bits) - 1) << dbg_shift) : 0;
}

uint8_t MaskConfig::decodeDBG(uint32_t mask_entry) const
{
    return dbg_bits ? static_cast<uint8_t>((mask_entry & dbg_mask_val) >> dbg_shift) : 0;
}

uint8_t MaskConfig::decodePOPT(uint32_t mask_entry) const
{
    return popt_bits ? static_cast<uint8_t>((mask_entry & popt_mask_val) >> popt_shift) : 0;
}

uint8_t MaskConfig::dbgTierToRRPV(uint8_t dbg_tier) const
{
    float fraction = static_cast<float>(dbg_tier) / std::max<uint8_t>(1, num_buckets);
    uint8_t result = static_cast<uint8_t>(rrpv_max * fraction);
    if (result > rrpv_max) result = rrpv_max;
    if (result == 0 && fraction > 0.0f) result = 1;
    return result;
}

bool GraphCacheContext::loadFromSideband(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open()) return false;
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    topology.num_vertices = static_cast<uint32_t>(parseJsonUint(content, "\"num_vertices\""));
    topology.num_edges = parseJsonUint(content, "\"num_edges\"");
    topology.max_degree = static_cast<uint32_t>(parseJsonUint(content, "\"max_degree\""));
    topology.avg_degree = topology.num_vertices > 0
        ? static_cast<double>(topology.num_edges) / topology.num_vertices
        : 0.0;
    topology.enabled = topology.num_vertices > 0;

    num_regions = 0;
    size_t pos = content.find("\"property_regions\"");
    if (pos != std::string::npos) {
        size_t arr_start = content.find('[', pos);
        size_t arr_end = content.find(']', arr_start);
        if (arr_start != std::string::npos && arr_end != std::string::npos) {
            std::string array_text = content.substr(arr_start, arr_end - arr_start + 1);
            size_t obj_pos = 0;
            while ((obj_pos = array_text.find('{', obj_pos)) != std::string::npos &&
                   num_regions < MAX_PROPERTY_REGIONS) {
                size_t obj_end = array_text.find('}', obj_pos);
                if (obj_end == std::string::npos) break;
                std::string obj = array_text.substr(obj_pos, obj_end - obj_pos + 1);

                PropertyRegion& region = regions[num_regions];
                region.name = parseJsonString(obj, "\"name\"");
                region.base_address = parseJsonUint(obj, "\"base\"");
                uint64_t size = parseJsonUint(obj, "\"size\"");
                region.upper_bound = region.base_address + size;
                region.num_elements = static_cast<uint32_t>(parseJsonUint(obj, "\"count\""));
                region.elem_size = static_cast<uint32_t>(parseJsonUint(obj, "\"elem_size\""));
                region.region_id = num_regions;
                region.num_buckets = 3;

                uint64_t third = ((size / 3) + rereference.cache_line_size - 1) & ~(rereference.cache_line_size - 1);
                region.bucket_bounds[0] = region.base_address + third;
                region.bucket_bounds[1] = region.base_address + 2 * third;
                region.bucket_bounds[2] = region.upper_bound;

                num_regions++;
                obj_pos = obj_end + 1;
            }
        }
    }

    num_edge_regions = 0;
    pos = content.find("\"edge_regions\"");
    if (pos != std::string::npos) {
        size_t arr_start = content.find('[', pos);
        size_t arr_end = content.find(']', arr_start);
        if (arr_start != std::string::npos && arr_end != std::string::npos) {
            std::string array_text = content.substr(arr_start, arr_end - arr_start + 1);
            size_t obj_pos = 0;
            while ((obj_pos = array_text.find('{', obj_pos)) != std::string::npos &&
                   num_edge_regions < edge_regions.size()) {
                size_t obj_end = array_text.find('}', obj_pos);
                if (obj_end == std::string::npos) break;
                std::string obj = array_text.substr(obj_pos, obj_end - obj_pos + 1);
                EdgeRegion& region = edge_regions[num_edge_regions];
                region.name = parseJsonString(obj, "\"name\"");
                region.base_address = parseJsonUint(obj, "\"base\"");
                uint64_t size = parseJsonUint(obj, "\"size\"");
                region.upper_bound = region.base_address + size;
                region.elem_size = static_cast<uint32_t>(parseJsonUint(obj, "\"elem_size\""));
                region.preferred = parseJsonBool(obj, "\"preferred\"");
                region.data_path = parseJsonString(obj, "\"data_path\"");
                num_edge_regions++;
                obj_pos = obj_end + 1;
            }
        }
    }

    loaded = num_regions > 0;
    mask_config.computeShifts();
    return loaded;
}

bool GraphCacheContext::loadRereferenceMatrix(const std::string& path)
{
    return rereference.loadFromFile(path);
}

void GraphCacheContext::setCacheLineSize(uint64_t line_size)
{
    if (line_size == 0) return;
    rereference.cache_line_size = line_size;
}

uint32_t GraphCacheContext::currentVertexForPopt(uint32_t core_id) const
{
    if (hasCurrentVertexHint(core_id)) {
        uint32_t vertex = getCurrentVertexHint(core_id);
        current_dst_vertex = vertex;
        current_outer_vertex = vertex;
        return vertex;
    }
    return current_dst_vertex;
}

void GraphCacheContext::updateVertexFromAddr(uint64_t addr, uint32_t core_id) const
{
    if (num_regions == 0 || !regions[0].contains(addr) || regions[0].elem_size == 0) return;
    uint32_t vertex = static_cast<uint32_t>((addr - regions[0].base_address) / regions[0].elem_size);
    current_dst_vertex = vertex;
    if (!hasCurrentVertexHint(core_id)) current_outer_vertex = vertex;
}

bool GraphCacheContext::isPropertyData(uint64_t addr) const
{
    for (uint32_t i = 0; i < num_regions; ++i) {
        if (regions[i].contains(addr)) return true;
    }
    return false;
}

bool GraphCacheContext::isEdgeData(uint64_t addr) const
{
    for (uint32_t i = 0; i < num_edge_regions; ++i) {
        if (edge_regions[i].contains(addr)) return true;
    }
    return false;
}

uint32_t GraphCacheContext::classifyBucket(uint64_t addr) const
{
    for (uint32_t i = 0; i < num_regions; ++i) {
        if (regions[i].contains(addr)) return regions[i].classifyBucket(addr);
    }
    return mask_config.num_buckets;
}

uint32_t GraphCacheContext::findNextRef(uint64_t addr, uint32_t core_id) const
{
    if (!rereference.enabled) return 127;
    for (uint32_t i = 0; i < num_regions; ++i) {
        if (regions[i].contains(addr)) {
            uint32_t cline_id = static_cast<uint32_t>(
                (addr - regions[i].base_address) / rereference.cache_line_size);
            return rereference.findNextRef(cline_id, currentVertexForPopt(core_id));
        }
    }
    return 127;
}

uint32_t GraphCacheContext::classifyGRASP(uint64_t addr, uint64_t llc_size) const
{
    constexpr double hot_fraction = 0.10;
    uint64_t hot_bytes = static_cast<uint64_t>(hot_fraction * llc_size);
    for (uint32_t i = 0; i < num_regions; ++i) {
        if (regions[i].contains(addr)) {
            uint64_t offset = addr - regions[i].base_address;
            if (offset < hot_bytes) return 1;
            if (offset < 2 * hot_bytes) return 2;
            return 3;
        }
    }
    return 3;
}

uint8_t GraphCacheContext::getInsertRRPV(uint64_t addr) const
{
    uint32_t bucket = classifyBucket(addr);
    if (bucket >= mask_config.num_buckets) return mask_config.rrpv_max;
    return mask_config.dbgTierToRRPV(static_cast<uint8_t>(bucket));
}

GraphCacheContext& globalContext()
{
    static GraphCacheContext context;
    return context;
}

}  // namespace sniper
}  // namespace graphbrew