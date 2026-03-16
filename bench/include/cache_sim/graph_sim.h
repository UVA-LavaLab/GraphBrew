// Copyright (c) 2024, UVA LavaLab
// Graph Simulation Helper for Cache Tracking
// Provides helper functions and macros for cache simulation

#ifndef GRAPH_SIM_H_
#define GRAPH_SIM_H_

#include "cache_sim.h"
#include "graph_cache_context.h"
#include <graph.h>
#include <pvector.h>

namespace cache_sim {

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
        (graph_ctx).hints_for_thread().mask = static_cast<uint8_t>(mask_val); \
        (cache).access(reinterpret_cast<uint64_t>(&(arr)[idx]), false); \
    } while(0)

// ECG: Read with mask + prefetch hint.
// After the primary access, resolves the prefetch target from the mask
// and issues a prefetch (read into cache hierarchy without data use).
#define SIM_CACHE_READ_MASKED_PREFETCH(cache, arr, idx, graph_ctx, mask_val) \
    do { \
        (graph_ctx).hints_for_thread().mask = static_cast<uint8_t>(mask_val); \
        (cache).access(reinterpret_cast<uint64_t>(&(arr)[idx]), false); \
        uint32_t _pfx_target = (graph_ctx).resolvePrefetchTarget(mask_val); \
        if (_pfx_target != UINT32_MAX) { \
            (cache).access(reinterpret_cast<uint64_t>(&(arr)[_pfx_target]), false); \
        } \
    } while(0)

// Track CSR edge list traversal (reading neighbor IDs from edge array).
// Call once per edge during neighbor iteration.
#define SIM_CACHE_READ_EDGE(cache, neighbor_ptr) \
    (cache).access(reinterpret_cast<uint64_t>(neighbor_ptr), false)

// Track CSR offset array access (reading row pointer for vertex u).
// Call once per vertex to track the offset[u] and offset[u+1] lookups.
#define SIM_CACHE_READ_OFFSET(cache, offset_arr, u) \
    do { \
        (cache).access(reinterpret_cast<uint64_t>(&(offset_arr)[u]), false); \
        (cache).access(reinterpret_cast<uint64_t>(&(offset_arr)[(u)+1]), false); \
    } while(0)

} // namespace cache_sim

#endif // GRAPH_SIM_H_
