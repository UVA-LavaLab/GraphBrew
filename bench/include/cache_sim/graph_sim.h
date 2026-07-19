// Copyright (c) 2024, UVA LavaLab
// Graph Simulation Helper for Cache Tracking
// Provides helper functions and macros for cache simulation

#ifndef GRAPH_SIM_H_
#define GRAPH_SIM_H_

#include "cache_sim.h"
#include "graph_cache_context.h"
#include <graph.h>
#include <pvector.h>
#include <string>

namespace cache_sim {

static constexpr size_t GRAPH_SIM_PROPERTY_ALIGNMENT = 4096;
static constexpr uint64_t GRAPH_SIM_IN_RECORD_BASE = 0x100000000000ULL;
static constexpr uint64_t GRAPH_SIM_OUT_RECORD_BASE = 0x200000000000ULL;

inline int GraphSimEnvIntClamped(const char* name, int default_value,
                                 int min_value, int max_value) {
    const char* value = std::getenv(name);
    if (!value) return default_value;
    int parsed = std::atoi(value);
    return std::max(min_value, std::min(max_value, parsed));
}

inline EvictionPolicy GraphSimEffectiveL3Policy() {
    EvictionPolicy policy =
        GetEnvPolicy("CACHE_POLICY", EvictionPolicy::LRU);
    return GetEnvPolicy("CACHE_L3_POLICY", policy);
}

inline bool GraphSimEcgGraspPoptPolicy() {
    const EvictionPolicy policy = GraphSimEffectiveL3Policy();
    const char* mode = std::getenv("ECG_MODE");
    return policy == EvictionPolicy::ECG && mode &&
           StringToECGMode(mode) == ECGMode::ECG_GRASP_POPT;
}

inline bool GraphSimMatrixFreeK2() {
    return GraphSimEcgGraspPoptPolicy() &&
           GraphSimEnvIntClamped("ECG_EDGE_MASK_SCHED", 0, 0, 4) == 2;
}

inline bool GraphSimEcgEdgeRecord() {
    const bool masks_enabled =
        std::getenv("ECG_EDGE_MASKS") != nullptr ||
        std::getenv("ECG_BFS_EDGE_MASKS") != nullptr ||
        std::getenv("ECG_SSSP_EDGE_MASKS") != nullptr ||
        std::getenv("ECG_BC_EDGE_MASKS") != nullptr ||
        std::getenv("ECG_CC_EDGE_MASKS") != nullptr;
    return GraphSimEcgGraspPoptPolicy() && masks_enabled;
}

inline int GraphSimEcgRecordBytes(uint64_t num_vertices, int epoch_bits) {
    int forced = GraphSimEnvIntClamped(
        "ECG_EDGE_RECORD_BYTES", 0, 0, 16);
    if (forced == 4 || forced == 8 || forced == 16) return forced;
    int id_bits = 1;
    while (id_bits < 32 &&
           (uint64_t(1) << id_bits) < num_vertices) {
        ++id_bits;
    }

    int tier_bits = GraphSimEnvIntClamped(
        "ECG_RECORD_TIER_BITS", 2, 0, 8);
    int popt_bits = GraphSimEnvIntClamped(
        "ECG_RECORD_POPT_BITS", 0, 0, 8);
    int prefetch_bits = GraphSimEnvIntClamped(
        "ECG_RECORD_PREFETCH_BITS", 0, 0, 32);
    int schedule_k = GraphSimEnvIntClamped(
        "ECG_EDGE_MASK_SCHED", 0, 0, 4);
    if (schedule_k == 2) return 8;
    int epoch_payload_bits = epoch_bits * std::max(1, schedule_k);
    int needed = id_bits + epoch_payload_bits +
                 tier_bits + popt_bits + prefetch_bits;
    if (needed <= 32) return 4;
    if (needed <= 64) return 8;
    return 16;
}

inline int GraphSimEcgWeightedSidecarBytes(
        uint64_t num_vertices, int epoch_bits) {
    if (GraphSimEnvIntClamped("ECG_EDGE_MASK_SCHED", 0, 0, 4) == 2)
        return 4;
    return GraphSimEcgRecordBytes(num_vertices, epoch_bits);
}

// ============================================================================
// SimArray: Wrapper for property arrays with cache tracking
// Works with both single-core CacheHierarchy and MultiCoreCacheHierarchy
// ============================================================================
template<typename T, typename CacheType = CacheHierarchy>
class SimArray {
public:
    SimArray(pvector<T>& arr, CacheType& cache)
        : data_(arr.data()), size_(arr.size()), cache_(cache) {}
    
    SimArray(T* data, size_t size, CacheType& cache)
        : data_(data), size_(size), cache_(cache) {}

    // Read with tracking
    T read(size_t index) const {
        cache_.readArray(data_, index);
        return data_[index];
    }

    // Write with tracking
    void write(size_t index, const T& value) {
        cache_.writeArray(data_, index);
        data_[index] = value;
    }

    // Atomic add with tracking
    void atomicAdd(size_t index, const T& value) {
        cache_.readArray(data_, index);
        cache_.writeArray(data_, index);
        #pragma omp atomic
        data_[index] += value;
    }

    // Get raw pointer
    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return size_; }

private:
    T* data_;
    size_t size_;
    CacheType& cache_;
};

// ============================================================================
// Convenience macros for cache tracking with explicit cache instance
// (Use these in simulation code that passes a specific cache object)
// ============================================================================

// Track reading from array element (with explicit cache instance)
#define SIM_CACHE_READ(cache, arr, idx) \
    (cache).access(reinterpret_cast<uint64_t>(&(arr)[idx]), false)

// Track writing to array element (with explicit cache instance)
#define SIM_CACHE_WRITE(cache, arr, idx) \
    (cache).access(reinterpret_cast<uint64_t>(&(arr)[idx]), true)

// Track reading neighbor iteration (one cache access per neighbor)
#define SIM_CACHE_TRACK_NEIGHBOR(cache, neighbor_ptr) \
    (cache).access(reinterpret_cast<uint64_t>(neighbor_ptr), false)

// P-OPT / GRASP: Update current destination vertex being processed.
// Call this at the top of the outer loop (for each destination vertex)
// so P-OPT can compute next-reference distances from the rereference matrix.
#define SIM_SET_VERTEX(cache, vertex_id) \
    (cache).setCurrentVertex(static_cast<uint32_t>(vertex_id))

// ECG: Read with per-edge mask hint.
// Sets the mask in GraphCacheContext before the access so the ECG policy
// can read DBG tier + P-OPT quant from the mask instead of address-range.
// mask_val = pre-encoded mask entry from the parallel mask array.
#define SIM_CACHE_READ_MASKED(cache, arr, idx, graph_ctx, mask_val) \
    do { \
        (graph_ctx).hints_for_thread().mask = static_cast<uint32_t>(mask_val); \
        (cache).access(reinterpret_cast<uint64_t>(&(arr)[idx]), false); \
    } while(0)

// ECG: Read with mask + prefetch hint.
// After the primary access, resolves the prefetch target from the mask
// and issues a prefetch if the target is not in the runtime dedup window.
// Prefetch uses cache.prefetch() which fills the cache WITHOUT counting
// as a demand access — prefetch misses don't inflate the miss rate.
#define SIM_CACHE_READ_MASKED_PREFETCH(cache, arr, idx, graph_ctx, mask_val) \
    do { \
        (graph_ctx).hints_for_thread().mask = static_cast<uint32_t>(mask_val); \
        (cache).access(reinterpret_cast<uint64_t>(&(arr)[idx]), false); \
        uint32_t _pfx_target = (graph_ctx).resolvePrefetchTarget(mask_val); \
        if (_pfx_target != UINT32_MAX) { \
            auto& _dw = (graph_ctx).dedup_for_thread(); \
            if (!_dw.contains(_pfx_target)) { \
                _dw.push(_pfx_target); \
                (cache).prefetch(reinterpret_cast<uint64_t>(&(arr)[_pfx_target])); \
                (graph_ctx).recordPrefetchIssued(); \
            } else { \
                (graph_ctx).recordPrefetchDuplicate(); \
            } \
        } else { \
            (graph_ctx).recordPrefetchNoTarget(); \
        } \
    } while(0)

// ECG: Prefetch a known future vertex property element.
// This is useful for runtime lookahead paths where the access stream already
// exposes a future vertex ID and the current mask target would be too late.
#define SIM_CACHE_PREFETCH_VERTEX(cache, arr, idx, graph_ctx) \
    do { \
        uint32_t _pfx_target = static_cast<uint32_t>(idx); \
        auto& _dw = (graph_ctx).dedup_for_thread(); \
        if (!_dw.contains(_pfx_target)) { \
            _dw.push(_pfx_target); \
            (cache).prefetch(reinterpret_cast<uint64_t>(&(arr)[_pfx_target])); \
            (graph_ctx).recordPrefetchIssued(); \
        } else { \
            (graph_ctx).recordPrefetchDuplicate(); \
        } \
    } while(0)

// Track CSR edge list traversal (reading neighbor IDs from edge array).
// Call once per edge during neighbor iteration.
#define SIM_CACHE_READ_EDGE(cache, neighbor_ptr) \
    (cache).access(reinterpret_cast<uint64_t>(neighbor_ptr), false)

#define SIM_CACHE_READ_EDGE_RECORD(cache, neighbor_ptr, edge_base, synthetic_base, record_bytes) \
    do { \
        const uint64_t _edge_index = static_cast<uint64_t>( \
            (neighbor_ptr) - (edge_base)); \
        const uint64_t _record_addr = (synthetic_base) + \
            _edge_index * static_cast<uint64_t>(record_bytes); \
        if ((record_bytes) >= 16) { \
            (cache).access(_record_addr, false); \
            (cache).access(_record_addr + 8ULL, false); \
        } else { \
            (cache).access(_record_addr, false); \
        } \
    } while (0)

#define SIM_CACHE_READ_EDGE_RECORD_BYPASS(cache, neighbor_ptr, edge_base, synthetic_base, record_bytes) \
    do { \
        const uint64_t _edge_index = static_cast<uint64_t>( \
            (neighbor_ptr) - (edge_base)); \
        const uint64_t _record_addr = (synthetic_base) + \
            _edge_index * static_cast<uint64_t>(record_bytes); \
        if ((record_bytes) >= 16) { \
            (cache).accessStream(_record_addr, false); \
            (cache).accessStream(_record_addr + 8ULL, false); \
        } else { \
            (cache).accessStream(_record_addr, false); \
        } \
    } while (0)

// ECG StreamShield: one-touch packed edge records can bypass LLC allocation
// while still filling the private caches. Only ECG's explicit stream path uses
// this; baseline CSR accesses remain unchanged.
#define SIM_CACHE_READ_EDGE_BYPASS(cache, neighbor_ptr) \
    (cache).accessStream(reinterpret_cast<uint64_t>(neighbor_ptr), false)
#define SIM_CACHE_READ_STREAM_BYPASS(cache, ptr, idx) \
    (cache).accessStream(reinterpret_cast<uint64_t>(&(ptr)[idx]), false)

// Track CSR offset array access (reading row pointer for vertex u).
// Call once per vertex to track the offset[u] and offset[u+1] lookups.
#define SIM_CACHE_READ_OFFSET(cache, offset_arr, u) \
    do { \
        (cache).access(reinterpret_cast<uint64_t>(&(offset_arr)[u]), false); \
        (cache).access(reinterpret_cast<uint64_t>(&(offset_arr)[(u)+1]), false); \
    } while(0)

} // namespace cache_sim

#endif // GRAPH_SIM_H_
