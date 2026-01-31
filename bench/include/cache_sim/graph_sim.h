// Copyright (c) 2024, UVA LavaLab
// Graph Simulation Helper for Cache Tracking
// Provides helper functions and macros for cache simulation

#ifndef GRAPH_SIM_H_
#define GRAPH_SIM_H_

#include "cache_sim.h"
#include "../gapbs/graph.h"
#include "../gapbs/pvector.h"

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

} // namespace cache_sim

#endif // GRAPH_SIM_H_
