// ============================================================================
// GraphCacheContext for gem5: Adapted metadata structures
// ============================================================================
//
// Provides the graph-aware metadata needed by GRASP, P-OPT, and ECG
// replacement policies running inside gem5. Mirrors the structures in
// bench/include/cache_sim/graph_cache_context.h but adapted for gem5's
// SimObject lifecycle and memory model.
//
// Key differences from standalone cache_sim version:
//   - Loaded from JSON sideband file (not inline C++ initialization)
//   - Uses gem5 physical addresses observed by the cache
//   - P-OPT rereference matrix stored host-side (oracle, not simulated memory)
//   - Per-access hints can be delivered by custom ECG instruction / CSR
//
// References:
//   - GRASP: Faldu et al., HPCA 2020
//   - P-OPT: Balaji et al., HPCA 2021
//   - ECG:   Mughrabi et al., GrAPL @ IPDPS 2026
// ============================================================================

#ifndef __MEM_CACHE_REPLACEMENT_POLICIES_GRAPH_CACHE_CONTEXT_GEM5_HH__
#define __MEM_CACHE_REPLACEMENT_POLICIES_GRAPH_CACHE_CONTEXT_GEM5_HH__

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace gem5 {
namespace replacement_policy {
namespace graph {

static constexpr uint32_t MAX_REGION_BUCKETS = 16;
static constexpr uint32_t MAX_PROPERTY_REGIONS = 8;
static constexpr uint64_t GRAPHBREW_SET_VERTEX_WORK_ID = 0x47525654ULL;

inline std::atomic<uint32_t>& currentVertexHintStorage() {
    static std::atomic<uint32_t> vertex{0};
    return vertex;
}

inline std::atomic<bool>& currentVertexHintValidStorage() {
    static std::atomic<bool> valid{false};
    return valid;
}

inline void setCurrentVertexHint(uint64_t vertex) {
    uint32_t clamped = vertex > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(vertex);
    currentVertexHintStorage().store(clamped, std::memory_order_release);
    currentVertexHintValidStorage().store(true, std::memory_order_release);
}

inline bool hasCurrentVertexHint() {
    return currentVertexHintValidStorage().load(std::memory_order_acquire);
}

inline uint32_t getCurrentVertexHint() {
    return currentVertexHintStorage().load(std::memory_order_acquire);
}

// ============================================================================
// ECGMode: Controls eviction tiebreaker priority
// ============================================================================
enum class ECGMode : uint8_t {
    DBG_PRIMARY,    // SRRIP -> DBG tier -> dynamic P-OPT
    POPT_PRIMARY,   // P-OPT exact 3-phase (bypasses SRRIP aging) -> DBG
    DBG_ONLY,       // GRASP-faithful DBG insertion/hit, plain SRRIP victim
    ECG_EMBEDDED,   // SRRIP -> stored P-OPT hint -> DBG (zero LLC overhead)
    ECG_COMBINED    // Pure SRRIP aging; insertion RRPV combines DBG+P-OPT
};

inline ECGMode stringToECGMode(const std::string& s) {
    if (s == "POPT_PRIMARY" || s == "popt_primary" || s == "popt") return ECGMode::POPT_PRIMARY;
    if (s == "DBG_ONLY" || s == "dbg_only" || s == "dbg") return ECGMode::DBG_ONLY;
    if (s == "ECG_EMBEDDED" || s == "ecg_embedded" || s == "embedded") return ECGMode::ECG_EMBEDDED;
    if (s == "ECG_COMBINED" || s == "ecg_combined" || s == "combined") return ECGMode::ECG_COMBINED;
    return ECGMode::DBG_PRIMARY;
}

inline std::string ecgModeToString(ECGMode mode) {
    switch (mode) {
        case ECGMode::DBG_PRIMARY:  return "DBG_PRIMARY";
        case ECGMode::POPT_PRIMARY: return "POPT_PRIMARY";
        case ECGMode::DBG_ONLY:     return "DBG_ONLY";
        case ECGMode::ECG_EMBEDDED: return "ECG_EMBEDDED";
        case ECGMode::ECG_COMBINED: return "ECG_COMBINED";
        default:                    return "UNKNOWN";
    }
}

// ============================================================================
// PropertyRegion: One tracked vertex data array
// ============================================================================
struct PropertyRegion {
    uint64_t base_address = 0;
    uint64_t upper_bound = 0;
    uint32_t num_elements = 0;
    uint32_t elem_size = 0;
    uint32_t region_id = 0;
    uint32_t num_buckets = 0;
    uint64_t bucket_bounds[MAX_REGION_BUCKETS] = {};

    uint32_t classifyBucket(uint64_t addr) const {
        if (addr < base_address || addr >= upper_bound || num_buckets == 0) {
            return num_buckets;
        }
        for (uint32_t bucket = 0; bucket < num_buckets; ++bucket) {
            if (addr < bucket_bounds[bucket]) return bucket;
        }
        return num_buckets - 1;
    }

    bool contains(uint64_t addr) const {
        return addr >= base_address && addr < upper_bound;
    }
};

// ============================================================================
// RereferenceMatrix: P-OPT oracle data (host-side, not in simulated memory)
// ============================================================================
struct RereferenceMatrix {
    std::vector<uint8_t> data;
    uint32_t num_cache_lines = 0;
    uint32_t num_epochs = 256;
    uint32_t epoch_size = 0;
    uint32_t sub_epoch_size = 0;
    uint64_t base_address = 0;
    uint64_t cache_line_size = 64;
    bool enabled = false;

    // P-OPT Algorithm 2 semantics using GraphBrew's paired matrix convention:
    // MSB=1 means referenced in this epoch (final sub-epoch in low bits),
    // MSB=0 means not referenced (distance-to-next in low bits). This is the
    // inverse bit polarity of the HPCA'21 paper text, but matches
    // bench/include/graphbrew/partition/cagra/popt.h::makeOffsetMatrix().
    uint32_t findNextRef(uint32_t cline_id, uint32_t current_vertex) const {
        if (!enabled || cline_id >= num_cache_lines) return 127;
        uint32_t epoch_id = (epoch_size > 0) ? (current_vertex / epoch_size) : 0;
        if (epoch_id >= num_epochs) return 127;

        uint8_t entry = data[epoch_id * num_cache_lines + cline_id];
        constexpr uint8_t OR_MASK = 0x80;
        constexpr uint8_t AND_MASK = 0x7F;

        if ((entry & OR_MASK) != 0) {
            uint8_t last_ref_sub_epoch = entry & AND_MASK;
            uint32_t current_sub_epoch = (sub_epoch_size > 0)
                ? ((current_vertex % epoch_size) / sub_epoch_size) : 0;
            if (current_sub_epoch <= last_ref_sub_epoch) return 0;
            if (epoch_id + 1 < num_epochs) {
                uint8_t next_entry = data[(epoch_id + 1) * num_cache_lines + cline_id];
                if ((next_entry & OR_MASK) != 0) return 1;
                uint8_t reref = next_entry & AND_MASK;
                return (reref < 127) ? reref + 1 : 127;
            }
            return 127;
        }
        return entry & AND_MASK;
    }

    uint32_t findNextRefByAddr(uint64_t addr, uint32_t current_vertex) const {
        if (!enabled || addr < base_address) return 127;
        uint32_t cline_id = static_cast<uint32_t>(
            (addr - base_address) / cache_line_size);
        return findNextRef(cline_id, current_vertex);
    }

    bool loadFromFile(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) return false;

        file.read(reinterpret_cast<char*>(&num_epochs), 4);
        file.read(reinterpret_cast<char*>(&num_cache_lines), 4);
        file.read(reinterpret_cast<char*>(&epoch_size), 4);
        file.read(reinterpret_cast<char*>(&sub_epoch_size), 4);

        size_t matrix_size = static_cast<size_t>(num_epochs) * num_cache_lines;
        data.resize(matrix_size);
        file.read(reinterpret_cast<char*>(data.data()), matrix_size);

        enabled = file.good();
        return enabled;
    }
};

// ============================================================================
// MaskConfig: ECG per-edge mask hint configuration
// ============================================================================
struct MaskConfig {
    uint8_t mask_width = 8;
    uint8_t dbg_bits = 2;
    uint8_t popt_bits = 4;
    uint8_t prefetch_bits = 2;
    uint8_t num_buckets = 11;
    uint8_t rrpv_max = 7;
    ECGMode ecg_mode = ECGMode::DBG_PRIMARY;
    bool enabled = false;

    uint8_t prefetch_shift = 0;
    uint8_t popt_shift = 0;
    uint8_t dbg_shift = 0;
    uint32_t prefetch_mask_val = 0;
    uint32_t popt_mask_val = 0;
    uint32_t dbg_mask_val = 0;

    void computeShifts() {
        prefetch_shift = 0;
        popt_shift = prefetch_bits;
        dbg_shift = prefetch_bits + popt_bits;
        prefetch_mask_val = prefetch_bits ? ((1U << prefetch_bits) - 1) : 0;
        popt_mask_val = popt_bits ? (((1U << popt_bits) - 1) << popt_shift) : 0;
        dbg_mask_val = dbg_bits ? (((1U << dbg_bits) - 1) << dbg_shift) : 0;
    }

    uint8_t decodeDBG(uint32_t mask_entry) const {
        return dbg_bits ? static_cast<uint8_t>((mask_entry & dbg_mask_val) >> dbg_shift) : 0;
    }

    uint8_t decodePOPT(uint32_t mask_entry) const {
        return popt_bits ? static_cast<uint8_t>((mask_entry & popt_mask_val) >> popt_shift) : 0;
    }

    uint8_t dbgTierToRRPV(uint8_t dbg_tier) const {
        float fraction = static_cast<float>(dbg_tier) / std::max(uint8_t(1), num_buckets);
        uint8_t result = static_cast<uint8_t>(rrpv_max * fraction);
        if (result > rrpv_max) result = rrpv_max;
        if (result == 0 && fraction > 0.0f) result = 1;
        return result;
    }
};

// ============================================================================
// GraphTopology: Degree distribution for bucket classification
// ============================================================================
struct GraphTopology {
    uint32_t num_vertices = 0;
    uint64_t num_edges = 0;
    uint32_t num_buckets = 11;
    double avg_degree = 0.0;
    uint32_t bucket_vertex_counts[MAX_REGION_BUCKETS] = {};
    bool enabled = false;
};

// ============================================================================
// GraphCacheContext: Unified metadata for all graph-aware policies
// ============================================================================
struct GraphCacheContext {
    PropertyRegion regions[MAX_PROPERTY_REGIONS];
    uint32_t num_regions = 0;

    GraphTopology topology;
    MaskConfig mask_config;
    RereferenceMatrix rereference;

    uint32_t current_src_vertex = 0;
    mutable uint32_t current_dst_vertex = 0;
    uint8_t current_mask = 0;
    mutable uint32_t current_outer_vertex = 0;
    bool loaded = false;

    uint32_t currentVertexForPopt() const {
        if (hasCurrentVertexHint()) {
            uint32_t vertex = getCurrentVertexHint();
            current_dst_vertex = vertex;
            current_outer_vertex = vertex;
            return vertex;
        }
        return current_dst_vertex;
    }

    void updateVertexFromAddr(uint64_t addr) const {
        if (num_regions > 0 && regions[0].contains(addr) &&
            regions[0].elem_size > 0) {
            uint32_t vertex = static_cast<uint32_t>(
                (addr - regions[0].base_address) / regions[0].elem_size);
            current_dst_vertex = vertex;
            if (!hasCurrentVertexHint()) current_outer_vertex = vertex;
        }
    }

    bool isPropertyData(uint64_t addr) const {
        for (uint32_t i = 0; i < num_regions; ++i) {
            if (regions[i].contains(addr)) return true;
        }
        return false;
    }

    uint32_t classifyBucket(uint64_t addr) const {
        for (uint32_t i = 0; i < num_regions; ++i) {
            if (regions[i].contains(addr)) return regions[i].classifyBucket(addr);
        }
        return mask_config.num_buckets;
    }

    uint32_t findNextRef(uint64_t addr) const {
        if (!rereference.enabled) return 127;
        for (uint32_t i = 0; i < num_regions; ++i) {
            if (regions[i].contains(addr)) {
                uint32_t cline_id = static_cast<uint32_t>(
                    (addr - regions[i].base_address) / rereference.cache_line_size);
                return rereference.findNextRef(cline_id, currentVertexForPopt());
            }
        }
        return 127;
    }

    uint32_t classifyGRASP(uint64_t addr, size_t llc_size) const {
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

    uint8_t getInsertRRPV(uint64_t addr) const {
        uint32_t bucket = classifyBucket(addr);
        if (bucket >= mask_config.num_buckets) return mask_config.rrpv_max;
        return mask_config.dbgTierToRRPV(static_cast<uint8_t>(bucket));
    }

    bool loadFromSideband(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) return false;

        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
        file.close();

        topology.num_vertices = parseJsonUint(content, "\"num_vertices\"");
        topology.num_edges = parseJsonUint(content, "\"num_edges\"");
        topology.avg_degree = (topology.num_vertices > 0)
            ? static_cast<double>(topology.num_edges) / topology.num_vertices : 0.0;
        topology.enabled = true;

        num_regions = 0;
        size_t pos = content.find("\"property_regions\"");
        if (pos != std::string::npos) {
            size_t arr_start = content.find('[', pos);
            size_t arr_end = content.find(']', arr_start);
            if (arr_start != std::string::npos && arr_end != std::string::npos) {
                std::string arr = content.substr(arr_start, arr_end - arr_start + 1);
                size_t obj_pos = 0;
                while ((obj_pos = arr.find('{', obj_pos)) != std::string::npos &&
                       num_regions < MAX_PROPERTY_REGIONS) {
                    size_t obj_end = arr.find('}', obj_pos);
                    if (obj_end == std::string::npos) break;
                    std::string obj = arr.substr(obj_pos, obj_end - obj_pos + 1);

                    regions[num_regions].base_address = parseJsonUint(obj, "\"base\"");
                    uint64_t size = parseJsonUint(obj, "\"size\"");
                    regions[num_regions].upper_bound = regions[num_regions].base_address + size;
                    regions[num_regions].num_elements = static_cast<uint32_t>(
                        parseJsonUint(obj, "\"count\""));
                    regions[num_regions].elem_size = static_cast<uint32_t>(
                        parseJsonUint(obj, "\"elem_size\""));
                    regions[num_regions].region_id = num_regions;

                    uint64_t region_bytes = size;
                    uint64_t third = (region_bytes / 3 + 63) & ~uint64_t(63);
                    regions[num_regions].num_buckets = 3;
                    regions[num_regions].bucket_bounds[0] = regions[num_regions].base_address + third;
                    regions[num_regions].bucket_bounds[1] = regions[num_regions].base_address + 2 * third;
                    regions[num_regions].bucket_bounds[2] = regions[num_regions].upper_bound;

                    num_regions++;
                    obj_pos = obj_end + 1;
                }
            }
        }

        loaded = (num_regions > 0);
        return loaded;
    }

private:
    static uint64_t parseJsonUint(const std::string& json, const std::string& key) {
        size_t pos = json.find(key);
        if (pos == std::string::npos) return 0;
        pos = json.find(':', pos);
        if (pos == std::string::npos) return 0;
        pos++;
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
        return std::strtoull(json.c_str() + pos, nullptr, 10);
    }
};

} // namespace graph
} // namespace replacement_policy
} // namespace gem5

#endif // __MEM_CACHE_REPLACEMENT_POLICIES_GRAPH_CACHE_CONTEXT_GEM5_HH__
