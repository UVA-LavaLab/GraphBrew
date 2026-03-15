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
//   - Uses gem5 Addr type for addresses
//   - P-OPT rereference matrix stored host-side (oracle, not in simulated memory)
//   - Per-access hints delivered via CSR written by custom ECG instruction
//
// References:
//   - GRASP: Faldu et al., HPCA 2020
//   - P-OPT: Balaji et al., HPCA 2021
//   - ECG:   Mughrabi et al., GrAPL @ IPDPS 2026
// ============================================================================

#ifndef __MEM_CACHE_REPLACEMENT_POLICIES_GRAPH_CACHE_CONTEXT_GEM5_HH__
#define __MEM_CACHE_REPLACEMENT_POLICIES_GRAPH_CACHE_CONTEXT_GEM5_HH__

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>

namespace gem5 {
namespace replacement_policy {
namespace graph {

// ============================================================================
// Constants
// ============================================================================
static constexpr uint32_t MAX_REGION_BUCKETS = 16;
static constexpr uint32_t MAX_PROPERTY_REGIONS = 8;

// ============================================================================
// ECGMode: Controls eviction tiebreaker priority
// ============================================================================
enum class ECGMode : uint8_t {
    DBG_PRIMARY,    // SRRIP → DBG tier → dynamic P-OPT
    POPT_PRIMARY,   // SRRIP → dynamic P-OPT → DBG tier
    DBG_ONLY        // SRRIP → DBG tier (no P-OPT, fast path)
};

inline ECGMode stringToECGMode(const std::string& s) {
    if (s == "POPT_PRIMARY" || s == "popt_primary") return ECGMode::POPT_PRIMARY;
    if (s == "DBG_ONLY" || s == "dbg_only") return ECGMode::DBG_ONLY;
    return ECGMode::DBG_PRIMARY;
}

inline std::string ecgModeToString(ECGMode mode) {
    switch (mode) {
        case ECGMode::DBG_PRIMARY:  return "DBG_PRIMARY";
        case ECGMode::POPT_PRIMARY: return "POPT_PRIMARY";
        case ECGMode::DBG_ONLY:     return "DBG_ONLY";
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
        if (addr < base_address || addr >= upper_bound || num_buckets == 0)
            return num_buckets;
        for (uint32_t b = 0; b < num_buckets; ++b) {
            if (addr < bucket_bounds[b]) return b;
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
    std::vector<uint8_t> data;      // Flat [num_epochs × num_cache_lines]
    uint32_t num_cache_lines = 0;
    uint32_t num_epochs = 256;
    uint32_t epoch_size = 0;
    uint32_t sub_epoch_size = 0;
    uint64_t base_address = 0;      // Base address of tracked region
    uint64_t cache_line_size = 64;
    bool enabled = false;

    // Algorithm 2 from P-OPT paper (Balaji et al., HPCA 2021)
    uint32_t findNextRef(uint32_t cline_id, uint32_t current_vertex) const {
        if (!enabled || cline_id >= num_cache_lines) return 127;
        uint32_t epoch_id = (epoch_size > 0) ? (current_vertex / epoch_size) : 0;
        if (epoch_id >= num_epochs) return 127;

        uint8_t entry = data[epoch_id * num_cache_lines + cline_id];
        constexpr uint8_t OR_MASK = 0x80;
        constexpr uint8_t AND_MASK = 0x7F;

        if ((entry & OR_MASK) != 0) {
            uint8_t lastRefSubEpoch = entry & AND_MASK;
            uint32_t currSubEpoch = (sub_epoch_size > 0)
                ? ((current_vertex % epoch_size) / sub_epoch_size) : 0;
            if (currSubEpoch <= lastRefSubEpoch) {
                return 0;
            } else {
                if (epoch_id + 1 < num_epochs) {
                    uint8_t next_entry = data[(epoch_id + 1) * num_cache_lines + cline_id];
                    if ((next_entry & OR_MASK) != 0) return 1;
                    uint8_t reref = next_entry & AND_MASK;
                    return (reref < 127) ? reref + 1 : 127;
                }
                return 127;
            }
        } else {
            return entry & AND_MASK;
        }
    }

    // Find next reference for a physical address
    uint32_t findNextRefByAddr(uint64_t addr, uint32_t current_vertex) const {
        if (!enabled || addr < base_address) return 127;
        uint32_t cline_id = static_cast<uint32_t>(
            (addr - base_address) / cache_line_size);
        return findNextRef(cline_id, current_vertex);
    }

    // Load from binary file (same format as standalone cache_sim)
    bool loadFromFile(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) return false;

        // Read header: num_epochs(4) + num_cache_lines(4) + epoch_size(4) + sub_epoch_size(4)
        f.read(reinterpret_cast<char*>(&num_epochs), 4);
        f.read(reinterpret_cast<char*>(&num_cache_lines), 4);
        f.read(reinterpret_cast<char*>(&epoch_size), 4);
        f.read(reinterpret_cast<char*>(&sub_epoch_size), 4);

        size_t matrix_size = static_cast<size_t>(num_epochs) * num_cache_lines;
        data.resize(matrix_size);
        f.read(reinterpret_cast<char*>(data.data()), matrix_size);

        enabled = f.good();
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

    // Computed shifts/masks
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
    // Degree bucket boundaries (vertex count thresholds)
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

    // Per-access mutable state (set by custom instruction or sideband)
    uint32_t current_src_vertex = 0;
    uint32_t current_dst_vertex = 0;
    uint8_t current_mask = 0;

    // Check if an address belongs to any tracked property region
    bool isPropertyData(uint64_t addr) const {
        for (uint32_t i = 0; i < num_regions; ++i) {
            if (regions[i].contains(addr)) return true;
        }
        return false;
    }

    // Classify address into degree bucket (across all regions)
    uint32_t classifyBucket(uint64_t addr) const {
        for (uint32_t i = 0; i < num_regions; ++i) {
            if (regions[i].contains(addr)) {
                return regions[i].classifyBucket(addr);
            }
        }
        return mask_config.num_buckets; // Outside all regions
    }

    // Find next reference distance for P-OPT
    uint32_t findNextRef(uint64_t addr) const {
        return rereference.findNextRefByAddr(addr, current_dst_vertex);
    }

    // Get RRPV for an address based on its degree bucket
    uint8_t getInsertRRPV(uint64_t addr) const {
        uint32_t bucket = classifyBucket(addr);
        if (bucket >= mask_config.num_buckets) {
            return mask_config.rrpv_max; // Unknown → evict soon
        }
        return mask_config.dbgTierToRRPV(static_cast<uint8_t>(bucket));
    }
};

} // namespace graph
} // namespace replacement_policy
} // namespace gem5

#endif // __MEM_CACHE_REPLACEMENT_POLICIES_GRAPH_CACHE_CONTEXT_GEM5_HH__
