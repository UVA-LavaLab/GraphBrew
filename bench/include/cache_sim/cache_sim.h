// Copyright (c) 2024, UVA LavaLab
// Cache Simulator for Graph Algorithm Analysis
// Tracks L1/L2/L3 cache hits and misses with configurable parameters
// Supports multi-core architecture: private L1/L2 per core, shared L3

#ifndef CACHE_SIM_H_
#define CACHE_SIM_H_

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <list>
#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <mutex>
#include <atomic>
#include <omp.h>

#include "graph_cache_context.h"

namespace cache_sim {

// ============================================================================
// Eviction Policy Enumeration
// ============================================================================
enum class EvictionPolicy {
    LRU,      // Least Recently Used
    FIFO,     // First In First Out
    RANDOM,   // Random eviction
    LFU,      // Least Frequently Used
    PLRU,     // Pseudo-LRU (tree-based)
    SRRIP,    // Static Re-Reference Interval Prediction
    GRASP,    // Graph-aware cache Replacement with Software Prefetching (Faldu et al., HPCA 2020)
    POPT,     // Practical Optimal cache replacement for Graph Analytics (Balaji et al., HPCA 2021)
    ECG       // Expressing Locality for Caching in Graphs — fat-ID encoding (Mughrabi et al., GrAPL)
};

inline std::string PolicyToString(EvictionPolicy policy) {
    switch (policy) {
        case EvictionPolicy::LRU:    return "LRU";
        case EvictionPolicy::FIFO:   return "FIFO";
        case EvictionPolicy::RANDOM: return "RANDOM";
        case EvictionPolicy::LFU:    return "LFU";
        case EvictionPolicy::PLRU:   return "PLRU";
        case EvictionPolicy::SRRIP:  return "SRRIP";
        case EvictionPolicy::GRASP:  return "GRASP";
        case EvictionPolicy::POPT:   return "POPT";
        case EvictionPolicy::ECG:    return "ECG";
        default: return "UNKNOWN";
    }
}

inline EvictionPolicy StringToPolicy(const std::string& s) {
    if (s == "LRU" || s == "lru") return EvictionPolicy::LRU;
    if (s == "FIFO" || s == "fifo") return EvictionPolicy::FIFO;
    if (s == "RANDOM" || s == "random") return EvictionPolicy::RANDOM;
    if (s == "LFU" || s == "lfu") return EvictionPolicy::LFU;
    if (s == "PLRU" || s == "plru") return EvictionPolicy::PLRU;
    if (s == "SRRIP" || s == "srrip") return EvictionPolicy::SRRIP;
    if (s == "GRASP" || s == "grasp") return EvictionPolicy::GRASP;
    if (s == "POPT" || s == "popt" || s == "P-OPT" || s == "p-opt") return EvictionPolicy::POPT;
    if (s == "ECG" || s == "ecg") return EvictionPolicy::ECG;
    return EvictionPolicy::LRU;  // Default
}

// ============================================================================
// Cache Statistics
// ============================================================================
struct CacheStats {
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
    std::atomic<uint64_t> reads{0};
    std::atomic<uint64_t> writes{0};
    std::atomic<uint64_t> evictions{0};
    std::atomic<uint64_t> writebacks{0};
    
    CacheStats() = default;
    
    // Copy constructor (needed for aggregation)
    CacheStats(const CacheStats& other) 
        : hits(other.hits.load())
        , misses(other.misses.load())
        , reads(other.reads.load())
        , writes(other.writes.load())
        , evictions(other.evictions.load())
        , writebacks(other.writebacks.load()) {}
    
    // Copy assignment
    CacheStats& operator=(const CacheStats& other) {
        hits = other.hits.load();
        misses = other.misses.load();
        reads = other.reads.load();
        writes = other.writes.load();
        evictions = other.evictions.load();
        writebacks = other.writebacks.load();
        return *this;
    }

    void reset() {
        hits = 0;
        misses = 0;
        reads = 0;
        writes = 0;
        evictions = 0;
        writebacks = 0;
    }

    double hitRate() const {
        uint64_t total = hits + misses;
        return total > 0 ? (double)hits / total : 0.0;
    }

    double missRate() const {
        return 1.0 - hitRate();
    }

    uint64_t totalAccesses() const {
        return hits + misses;
    }
};

// ============================================================================
// Cache Line
// ============================================================================
struct CacheLine {
    uint64_t tag = 0;
    bool valid = false;
    bool dirty = false;
    uint64_t last_access = 0;    // For LRU
    uint64_t insert_time = 0;    // For FIFO
    uint64_t access_count = 0;   // For LFU
    uint8_t rrpv = 3;            // For SRRIP, GRASP, P-OPT, ECG
    uint64_t line_addr = 0;      // Cache-line-aligned address
    uint8_t ecg_dbg_tier = 0;    // ECG: stored DBG degree tier (structural, for eviction tiebreak)
    uint8_t ecg_popt_hint = 0;   // ECG_EMBEDDED: stored P-OPT quantized rereference hint
};

// ============================================================================
// P-OPT State: Rereference Matrix context for graph-aware replacement
// ============================================================================
struct POPTState {
    const uint8_t* reref_matrix = nullptr;  // Compressed rereference matrix [epochs × cache_lines]
    uint64_t irreg_base = 0;       // Base address of irregular (vertex) data region
    uint64_t irreg_bound = 0;      // Upper bound of irregular data region
    uint32_t num_cache_lines = 0;  // Number of cache lines covering vertex data
    uint32_t num_epochs = 256;     // Number of epochs (default 256)
    uint32_t epoch_size = 0;       // Vertices per epoch
    uint32_t sub_epoch_size = 0;   // Vertices per sub-epoch (epoch_size / 128)
    uint32_t current_vertex = 0;   // Current destination vertex being processed
    bool enabled = false;          // Whether P-OPT state has been initialized

    // Algorithm 2 from P-OPT paper (Balaji et al., HPCA 2021, Section 4.1):
    // Compute next-reference distance for a cache line.
    //
    // Encoding (from makeOffsetMatrix in popt.h, matching reference llc.cpp):
    //   MSB=1 (bit 7 set): cache line IS referenced in this epoch
    //     → bits [6:0] = sub-epoch of LAST access within the epoch
    //   MSB=0 (bit 7 clear): cache line is NOT referenced in this epoch
    //     → bits [6:0] = distance (in epochs) to next epoch with a reference
    //
    // Returns: rereference distance (0 = accessed soon, higher = farther away)
    uint32_t findNextRef(uint32_t cline_id) const {
        if (!enabled || cline_id >= num_cache_lines) return 127;
        uint32_t epoch_id = current_vertex / epoch_size;
        if (epoch_id >= num_epochs) return 127;

        // Look up rereference matrix entry: matrix is transposed as [epoch][cline]
        uint8_t entry = reref_matrix[epoch_id * num_cache_lines + cline_id];
        constexpr uint8_t OR_MASK = 0x80;   // MSB
        constexpr uint8_t AND_MASK = 0x7F;  // lower 7 bits

        if ((entry & OR_MASK) != 0) {
            // MSB=1: Referenced in this epoch — data = sub-epoch of last access
            uint8_t lastRefSubEpoch = entry & AND_MASK;
            uint32_t currSubEpoch = (current_vertex % epoch_size) / sub_epoch_size;
            if (currSubEpoch <= lastRefSubEpoch) {
                return 0;  // Still will be accessed within this epoch
            } else {
                // Past final access in this epoch — check next epoch
                if (epoch_id + 1 < num_epochs) {
                    uint8_t next_entry = reref_matrix[(epoch_id + 1) * num_cache_lines + cline_id];
                    if ((next_entry & OR_MASK) != 0) {
                        return 1;  // Referenced next epoch
                    } else {
                        uint8_t reref = next_entry & AND_MASK;
                        return (reref < 127) ? reref + 1 : 127;
                    }
                }
                return 127;  // No future reference found (max distance)
            }
        } else {
            // MSB=0: NOT referenced in this epoch — data = distance to next epoch
            uint8_t reref = entry & AND_MASK;
            return reref;  // 0 = check next epoch, 127 = very far away
        }
    }
};

// ============================================================================
// GRASP State: Degree-aware retention with 3-tier RRIP insertion
// (Faldu et al., HPCA 2020 — reference: grasp.cpp + common.h)
//
// GRASP uses DBG-reordered vertex data where high-degree vertices are
// placed at the front of the array.  Three reuse tiers:
//   High-reuse:      addr ∈ [data_base, data_base + high_reuse_bound)
//   Moderate-reuse:  addr ∈ [high_reuse_bound, moderate_reuse_bound)
//   Low-reuse:       addr ∉ above ranges
//
// The border between high/moderate is determined by what fraction of
// the LLC capacity should be reserved for high-degree vertices (default
// from paper: the "f" parameter in the trace header, typically 5-20%).
// ============================================================================
struct GRASPState {
    uint64_t data_base = 0;            // Base address of vertex data array (DBG-ordered)
    uint64_t data_end = 0;             // End address of vertex data array
    uint64_t high_reuse_bound = 0;     // Addresses < this are high-reuse (hot hubs)
    uint64_t moderate_reuse_bound = 0; // Addresses < this (and >= high_reuse_bound) are moderate
    bool enabled = false;

    enum class ReuseTier { HIGH, MODERATE, LOW };

    // Classify an address into one of three reuse tiers
    ReuseTier classify(uint64_t address) const {
        if (!enabled) return ReuseTier::LOW;
        if (address >= data_base && address < high_reuse_bound)
            return ReuseTier::HIGH;
        if (address >= high_reuse_bound && address < moderate_reuse_bound)
            return ReuseTier::MODERATE;
        return ReuseTier::LOW;
    }

    // Initialize from graph properties:
    //   data_ptr: base address of the vertex property array (must be DBG-ordered)
    //   num_vertices: total vertices
    //   elem_size: sizeof each element (e.g., sizeof(float) for PageRank scores)
    //   llc_size: LLC size in bytes
    //   hot_fraction: fraction of LLC to reserve for hot vertices (0.0-1.0, default 0.1)
    void init(uint64_t data_ptr, uint32_t num_vertices, size_t elem_size,
              size_t llc_size, double hot_fraction = 0.1) {
        data_base = data_ptr;
        data_end = data_ptr + num_vertices * elem_size;
        // High-reuse border: hot_fraction of LLC capacity worth of vertex data
        // Align to 64-byte cache line boundary for correct line_addr classification
        uint64_t high_bytes = static_cast<uint64_t>(hot_fraction * llc_size);
        constexpr uint64_t LINE_MASK = ~uint64_t(63);
        high_reuse_bound = (data_base + high_bytes + 63) & LINE_MASK;
        if (high_reuse_bound > data_end) high_reuse_bound = data_end;
        // Moderate-reuse border: 2× the high-reuse region (per reference common.h)
        moderate_reuse_bound = (data_base + 2 * high_bytes + 63) & LINE_MASK;
        if (moderate_reuse_bound > data_end) moderate_reuse_bound = data_end;
        enabled = true;
    }
};

// ============================================================================
// Fast Cache Level - NO LOCKS, uses clock algorithm (faster than LRU)
// Use this for private per-thread caches where no locking is needed
// ============================================================================
class FastCacheLevel {
public:
    FastCacheLevel(const std::string& name, size_t size_bytes, size_t line_size,
                   size_t associativity)
        : name_(name), size_bytes_(size_bytes), line_size_(line_size),
          associativity_(associativity) {
        
        num_sets_ = size_bytes / (line_size * associativity);
        if (num_sets_ == 0) num_sets_ = 1;
        
        // Round to power of 2 for fast modulo
        size_t orig_sets = num_sets_;
        num_sets_ = 1;
        while (num_sets_ < orig_sets) num_sets_ <<= 1;
        set_mask_ = num_sets_ - 1;
        
        offset_bits_ = log2i(line_size);
        
        // Flat arrays for better cache locality
        tags_.resize(num_sets_ * associativity, 0);
        valid_.resize(num_sets_ * associativity, false);
        clock_.resize(num_sets_ * associativity, false);
    }

    // Fast access - no locks, inline everything
    __attribute__((always_inline))
    inline bool access(uint64_t address) {
        stats_.reads++;
        
        uint64_t tag = address >> offset_bits_;
        size_t set_base = (tag & set_mask_) * associativity_;
        
        // Check all ways
        for (size_t i = 0; i < associativity_; i++) {
            size_t idx = set_base + i;
            if (valid_[idx] && tags_[idx] == tag) {
                stats_.hits++;
                clock_[idx] = true;  // Mark recently used
                return true;
            }
        }
        
        stats_.misses++;
        return false;
    }

    // Fast insert using clock algorithm
    __attribute__((always_inline))
    inline void insert(uint64_t address) {
        uint64_t tag = address >> offset_bits_;
        size_t set_base = (tag & set_mask_) * associativity_;
        
        // Find invalid slot first
        for (size_t i = 0; i < associativity_; i++) {
            size_t idx = set_base + i;
            if (!valid_[idx]) {
                tags_[idx] = tag;
                valid_[idx] = true;
                clock_[idx] = true;
                return;
            }
        }
        
        // Clock algorithm: find slot with clock=false
        for (size_t i = 0; i < associativity_; i++) {
            size_t idx = set_base + i;
            if (!clock_[idx]) {
                stats_.evictions++;
                tags_[idx] = tag;
                clock_[idx] = true;
                return;
            }
            clock_[idx] = false;  // Second chance
        }
        
        // All had clock=true, use first
        stats_.evictions++;
        tags_[set_base] = tag;
        clock_[set_base] = true;
    }

    const CacheStats& getStats() const { return stats_; }
    void resetStats() { stats_.reset(); }
    const std::string& getName() const { return name_; }
    size_t getSizeBytes() const { return size_bytes_; }
    size_t getAssociativity() const { return associativity_; }
    size_t getNumSets() const { return num_sets_; }

private:
    static size_t log2i(size_t n) {
        size_t r = 0;
        while (n > 1) { n >>= 1; r++; }
        return r;
    }

    std::string name_;
    size_t size_bytes_;
    size_t line_size_;
    size_t associativity_;
    size_t num_sets_;
    size_t set_mask_;
    size_t offset_bits_;
    
    std::vector<uint64_t> tags_;
    std::vector<bool> valid_;
    std::vector<bool> clock_;
    CacheStats stats_;
};

// ============================================================================
// ULTRA-FAST Cache Level - Packed cache line structure for better locality
// ~2-3x faster than FastCacheLevel through better memory layout
// ============================================================================
class UltraFastCacheLevel {
public:
    // Pack tag + valid + clock into single struct for cache-friendly access
    struct alignas(16) CacheEntry {
        uint64_t tag;
        uint8_t valid;
        uint8_t clock;
        uint8_t pad[6];  // Pad to 16 bytes for alignment
    };

    UltraFastCacheLevel(const std::string& name, size_t size_bytes, size_t line_size,
                        size_t associativity)
        : name_(name), size_bytes_(size_bytes), line_size_(line_size),
          associativity_(associativity), hits_(0), misses_(0), evictions_(0) {
        
        num_sets_ = size_bytes / (line_size * associativity);
        if (num_sets_ == 0) num_sets_ = 1;
        
        // Round to power of 2 for fast modulo
        size_t orig_sets = num_sets_;
        num_sets_ = 1;
        while (num_sets_ < orig_sets) num_sets_ <<= 1;
        set_mask_ = num_sets_ - 1;
        
        offset_bits_ = __builtin_ctzll(line_size);  // Fast log2 for power of 2
        
        // Single contiguous array - all data for a set is together
        entries_.resize(num_sets_ * associativity);
        memset(entries_.data(), 0, entries_.size() * sizeof(CacheEntry));
    }

    // Ultra-fast access with packed structure
    __attribute__((always_inline, hot))
    inline bool access(uint64_t address) {
        const uint64_t tag = address >> offset_bits_;
        CacheEntry* __restrict__ set = &entries_[(tag & set_mask_) * associativity_];
        
        // Unrolled check for 8-way (most common)
        if (__builtin_expect(associativity_ == 8, 1)) {
            #pragma GCC unroll 8
            for (size_t i = 0; i < 8; i++) {
                if (set[i].valid && set[i].tag == tag) {
                    hits_++;
                    set[i].clock = 1;
                    return true;
                }
            }
        } else {
            for (size_t i = 0; i < associativity_; i++) {
                if (set[i].valid && set[i].tag == tag) {
                    hits_++;
                    set[i].clock = 1;
                    return true;
                }
            }
        }
        
        misses_++;
        return false;
    }

    // Ultra-fast insert
    __attribute__((always_inline, hot))
    inline void insert(uint64_t address) {
        const uint64_t tag = address >> offset_bits_;
        CacheEntry* __restrict__ set = &entries_[(tag & set_mask_) * associativity_];
        
        // Find invalid slot first
        for (size_t i = 0; i < associativity_; i++) {
            if (!set[i].valid) {
                set[i].tag = tag;
                set[i].valid = 1;
                set[i].clock = 1;
                return;
            }
        }
        
        // Clock algorithm: find slot with clock=0
        for (size_t i = 0; i < associativity_; i++) {
            if (!set[i].clock) {
                evictions_++;
                set[i].tag = tag;
                set[i].clock = 1;
                return;
            }
            set[i].clock = 0;
        }
        
        // All had clock=1, use first
        evictions_++;
        set[0].tag = tag;
        set[0].clock = 1;
    }

    CacheStats getStats() const {
        CacheStats s;
        s.hits = hits_;
        s.misses = misses_;
        s.evictions = evictions_;
        return s;
    }
    
    void resetStats() { hits_ = misses_ = evictions_ = 0; }
    const std::string& getName() const { return name_; }
    size_t getSizeBytes() const { return size_bytes_; }
    size_t getAssociativity() const { return associativity_; }
    size_t getNumSets() const { return num_sets_; }
    uint64_t getHits() const { return hits_; }
    uint64_t getMisses() const { return misses_; }
    double hitRate() const { 
        uint64_t total = hits_ + misses_;
        return total > 0 ? (double)hits_ / total : 0.0;
    }

private:
    std::string name_;
    size_t size_bytes_;
    size_t line_size_;
    size_t associativity_;
    size_t num_sets_;
    size_t set_mask_;
    size_t offset_bits_;
    
    std::vector<CacheEntry> entries_;
    uint64_t hits_;
    uint64_t misses_;
    uint64_t evictions_;
};

// ============================================================================
// Single Cache Level (with locks - for shared caches)
// ============================================================================
class CacheLevel {
public:
    CacheLevel(const std::string& name, size_t size_bytes, size_t line_size,
               size_t associativity, EvictionPolicy policy)
        : name_(name), size_bytes_(size_bytes), line_size_(line_size),
          associativity_(associativity), policy_(policy) {
        
        num_sets_ = size_bytes / (line_size * associativity);
        if (num_sets_ == 0) num_sets_ = 1;
        
        // Calculate bit widths
        offset_bits_ = log2i(line_size);
        index_bits_ = log2i(num_sets_);
        
        // Initialize cache structure
        cache_.resize(num_sets_);
        for (auto& set : cache_) {
            set.resize(associativity);
        }
        
        // Initialize random generator for RANDOM policy
        rng_.seed(42);
    }

    // Access cache (returns true on hit, false on miss)
    bool access(uint64_t address, bool is_write) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (is_write) {
            stats_.writes++;
        } else {
            stats_.reads++;
        }
        
        uint64_t tag = getTag(address);
        size_t set_idx = getSetIndex(address);
        auto& set = cache_[set_idx];
        
        // Check for hit
        for (size_t i = 0; i < associativity_; i++) {
            if (set[i].valid && set[i].tag == tag) {
                // Hit!
                stats_.hits++;
                updateOnHit(set, i);
                if (is_write) {
                    set[i].dirty = true;
                }
                return true;
            }
        }
        
        // Miss
        stats_.misses++;
        return false;
    }

    // Insert a line after a miss (called when lower level provides data)
    void insert(uint64_t address, bool is_write) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        uint64_t tag = getTag(address);
        size_t set_idx = getSetIndex(address);
        auto& set = cache_[set_idx];
        
        // Find victim
        size_t victim_idx = findVictim(set);
        
        // Evict if necessary
        if (set[victim_idx].valid) {
            stats_.evictions++;
            if (set[victim_idx].dirty) {
                stats_.writebacks++;
            }
        }
        
        // Insert new line
        set[victim_idx].tag = tag;
        set[victim_idx].valid = true;
        set[victim_idx].dirty = is_write;
        set[victim_idx].last_access = global_time_++;
        set[victim_idx].insert_time = global_time_;
        set[victim_idx].access_count = 1;
        set[victim_idx].rrpv = 2;  // For SRRIP: long re-reference (M-1 = 2, per Jaleel ISCA'10)
        set[victim_idx].line_addr = address & ~(uint64_t(line_size_ - 1));  // Store line-aligned address

        // GRASP: 3-tier RRIP insertion matching Faldu et al. HPCA 2020.
        // HIGH reuse (hubs):    RRPV = 1 (P_RRIP — protect in cache)
        // MODERATE reuse:       RRPV = 6 (I_RRIP — intermediate)
        // LOW reuse (cold/OOB): RRPV = 7 (M_RRIP — evict sooner)
        //
        // Hot boundary: first hot_fraction (10%) of LLC capacity within
        // each property region. Moderate = next 10%. Cold = rest.
        // After DBG reorder, highest-degree vertices are at front (low addr).
        if (policy_ == EvictionPolicy::GRASP) {
            constexpr uint8_t P_RRIP = 1;   // Priority (hot)
            constexpr uint8_t I_RRIP = 6;   // Intermediate (moderate)
            constexpr uint8_t M_RRIP = 7;   // Max (cold)
            if (graph_ctx_) {
                uint32_t tier = graph_ctx_->classifyGRASP(address, size_bytes_);
                if (tier == 1)       set[victim_idx].rrpv = P_RRIP;
                else if (tier == 2)  set[victim_idx].rrpv = I_RRIP;
                else                 set[victim_idx].rrpv = M_RRIP;
            } else if (grasp_state_.enabled) {
                auto t = grasp_state_.classify(address);
                if (t == GRASPState::ReuseTier::HIGH)           set[victim_idx].rrpv = P_RRIP;
                else if (t == GRASPState::ReuseTier::MODERATE)  set[victim_idx].rrpv = I_RRIP;
                else                                            set[victim_idx].rrpv = M_RRIP;
            }
        }

        // P-OPT: insert with SRRIP-style RRPV (long re-reference = M-1)
        if (policy_ == EvictionPolicy::POPT) {
            set[victim_idx].rrpv = 6;  // M_RRPV - 1 = long re-reference (SRRIP default)
        }

        // ECG: Mode-dependent insertion RRPV.
        // DBG_ONLY / DBG_PRIMARY / ECG_EMBEDDED: use GRASP 3-tier (1/6/7)
        // POPT_PRIMARY: use P-OPT-style RRPV=6 (matches pure P-OPT aging)
        if (policy_ == EvictionPolicy::ECG) {
            ECGMode mode = (graph_ctx_ && graph_ctx_->mask_config.enabled)
                ? graph_ctx_->mask_config.ecg_mode : ECGMode::DBG_PRIMARY;

            if (mode == ECGMode::POPT_PRIMARY) {
                // Match P-OPT insertion: uniform RRPV=6 for all lines
                set[victim_idx].rrpv = 6;
            } else {
                // GRASP-faithful 3-tier for DBG modes
                constexpr uint8_t P_RRIP = 1;
                constexpr uint8_t I_RRIP = 6;
                constexpr uint8_t M_RRIP_C = 7;
                if (graph_ctx_) {
                    uint32_t tier = graph_ctx_->classifyGRASP(address, size_bytes_);
                    if (tier == 1)       set[victim_idx].rrpv = P_RRIP;
                    else if (tier == 2)  set[victim_idx].rrpv = I_RRIP;
                    else                 set[victim_idx].rrpv = M_RRIP_C;
                }
            }

            // Store ECG mask fields for eviction tiebreaking
            if (graph_ctx_ && graph_ctx_->mask_array.enabled) {
                uint32_t mask_entry = graph_ctx_->hints_for_thread().mask;
                set[victim_idx].ecg_dbg_tier = graph_ctx_->mask_config.decodeDBG(mask_entry);
                set[victim_idx].ecg_popt_hint = graph_ctx_->mask_config.decodePOPT(mask_entry);
            } else if (graph_ctx_) {
                uint32_t bucket = graph_ctx_->classifyBucket(address);
                set[victim_idx].ecg_dbg_tier = (bucket < 11) ? static_cast<uint8_t>(bucket) : 0;
                // Compute live P-OPT hint if matrix available
                if (graph_ctx_->rereference.matrix) {
                    uint32_t dist = graph_ctx_->findNextRef(address);
                    // Quantize 0-127 distance to 4-bit (0-15)
                    set[victim_idx].ecg_popt_hint = static_cast<uint8_t>(
                        std::min(dist, uint32_t(127)) >> 3);
                } else {
                    set[victim_idx].ecg_popt_hint = 0;
                }
            }
        }
    }

    const CacheStats& getStats() const { return stats_; }
    void resetStats() { stats_.reset(); }
    
    const std::string& getName() const { return name_; }
    size_t getSizeBytes() const { return size_bytes_; }
    size_t getLineSize() const { return line_size_; }
    size_t getAssociativity() const { return associativity_; }
    size_t getNumSets() const { return num_sets_; }
    EvictionPolicy getPolicy() const { return policy_; }

    // ================================================================
    // P-OPT initialization: call once after building the rereference
    // matrix with makeOffsetMatrix() from popt.h
    // ================================================================
    void initPOPT(const uint8_t* reref_matrix, uint64_t irreg_base,
                  uint64_t irreg_bound, uint32_t num_vertices,
                  uint32_t num_epochs = 256) {
        uint32_t vtx_per_line = static_cast<uint32_t>(line_size_ / sizeof(float));
        if (vtx_per_line == 0) vtx_per_line = 1;
        popt_state_.reref_matrix = reref_matrix;
        popt_state_.irreg_base = irreg_base;
        popt_state_.irreg_bound = irreg_bound;
        popt_state_.num_cache_lines = (num_vertices + vtx_per_line - 1) / vtx_per_line;
        popt_state_.num_epochs = num_epochs;
        popt_state_.epoch_size = (num_vertices + num_epochs - 1) / num_epochs;
        popt_state_.sub_epoch_size = (popt_state_.epoch_size + 127) / 128;
        popt_state_.current_vertex = 0;
        popt_state_.enabled = true;
    }

    // ================================================================
    // GRASP initialization: call once with vertex data address range.
    // Requires DBG-reordered graph (hot vertices at low addresses).
    //   data_ptr: base address of vertex property array
    //   num_vertices: total vertices
    //   elem_size: sizeof each element (e.g. sizeof(float))
    //   llc_size: LLC size in bytes (for computing border regions)
    //   hot_fraction: fraction of LLC reserved for hot vertices (0.0-1.0)
    // ================================================================
    void initGRASP(uint64_t data_ptr, uint32_t num_vertices,
                   size_t elem_size, size_t llc_size,
                   double hot_fraction = 0.1) {
        grasp_state_.init(data_ptr, num_vertices, elem_size, llc_size, hot_fraction);
    }

    // Update current vertex for P-OPT (call at each outer-loop iteration)
    void setCurrentVertex(uint32_t vertex_id) {
        popt_state_.current_vertex = vertex_id;
        // Update unified context (thread-safe via per-thread hints)
        if (graph_ctx_) {
            const_cast<GraphCacheContext*>(graph_ctx_)->setCurrentVertices(vertex_id, vertex_id);
        }
    }

    // ================================================================
    // Unified GraphCacheContext: preferred over legacy init methods.
    // Replaces initPOPT() and initGRASP() with a single context.
    // ================================================================
    void initGraphContext(const GraphCacheContext* ctx) {
        graph_ctx_ = ctx;
    }

private:
    static size_t log2i(size_t n) {
        size_t r = 0;
        while (n > 1) { n >>= 1; r++; }
        return r;
    }

    uint64_t getTag(uint64_t address) const {
        return address >> (offset_bits_ + index_bits_);
    }

    size_t getSetIndex(uint64_t address) const {
        return (address >> offset_bits_) & ((1ULL << index_bits_) - 1);
    }

    void updateOnHit(std::vector<CacheLine>& set, size_t idx) {
        set[idx].last_access = global_time_++;
        set[idx].access_count++;
        
        // SRRIP: reset RRPV to 0 on hit
        if (policy_ == EvictionPolicy::SRRIP) {
            set[idx].rrpv = 0;
        }

        // GRASP: 3-tier hit promotion (Faldu et al. HPCA 2020)
        // Hot region → RRPV=0 (aggressive reset), others → decrement by 1
        if (policy_ == EvictionPolicy::GRASP) {
            uint64_t addr = set[idx].line_addr;
            if (graph_ctx_) {
                uint32_t tier = graph_ctx_->classifyGRASP(addr, size_bytes_);
                if (tier == 1) {
                    set[idx].rrpv = 0;  // Hot (hub): aggressive reset
                } else if (set[idx].rrpv > 0) {
                    set[idx].rrpv--;    // Others: gradual decrement
                }
            } else if (grasp_state_.enabled) {
                auto t = grasp_state_.classify(addr);
                if (t == GRASPState::ReuseTier::HIGH) set[idx].rrpv = 0;
                else if (set[idx].rrpv > 0) set[idx].rrpv--;
            }
        }

        // P-OPT: reset RRPV to 0 on hit (same as SRRIP hit promotion)
        if (policy_ == EvictionPolicy::POPT) {
            set[idx].rrpv = 0;
        }

        // ECG: Mode-dependent hit promotion
        if (policy_ == EvictionPolicy::ECG) {
            ECGMode mode = (graph_ctx_ && graph_ctx_->mask_config.enabled)
                ? graph_ctx_->mask_config.ecg_mode : ECGMode::DBG_PRIMARY;

            if (mode == ECGMode::POPT_PRIMARY) {
                // Match P-OPT: reset to 0 on hit (same as SRRIP)
                set[idx].rrpv = 0;
            } else {
                // GRASP-faithful 3-tier for DBG modes and ECG_EMBEDDED
                if (graph_ctx_) {
                    uint64_t addr = set[idx].line_addr;
                    uint32_t tier = graph_ctx_->classifyGRASP(addr, size_bytes_);
                    if (tier == 1) set[idx].rrpv = 0;           // Hot: aggressive reset
                    else if (set[idx].rrpv > 0) set[idx].rrpv--; // Others: gradual
                }
            }
        }
    }

    size_t findVictim(std::vector<CacheLine>& set) {
        // First, look for invalid line
        for (size_t i = 0; i < associativity_; i++) {
            if (!set[i].valid) return i;
        }
        
        // All lines valid, use eviction policy
        switch (policy_) {
            case EvictionPolicy::LRU:
                return findVictimLRU(set);
            case EvictionPolicy::FIFO:
                return findVictimFIFO(set);
            case EvictionPolicy::RANDOM:
                return findVictimRandom(set);
            case EvictionPolicy::LFU:
                return findVictimLFU(set);
            case EvictionPolicy::PLRU:
                return findVictimPLRU(set);
            case EvictionPolicy::SRRIP:
                return findVictimSRRIP(set);
            case EvictionPolicy::GRASP:
                return findVictimGRASP(set);
            case EvictionPolicy::POPT:
                return findVictimPOPT(set);
            case EvictionPolicy::ECG:
                return findVictimECG(set);
            default:
                return findVictimLRU(set);
        }
    }

    size_t findVictimLRU(std::vector<CacheLine>& set) {
        size_t victim = 0;
        uint64_t oldest = set[0].last_access;
        for (size_t i = 1; i < associativity_; i++) {
            if (set[i].last_access < oldest) {
                oldest = set[i].last_access;
                victim = i;
            }
        }
        return victim;
    }

    size_t findVictimFIFO(std::vector<CacheLine>& set) {
        size_t victim = 0;
        uint64_t oldest = set[0].insert_time;
        for (size_t i = 1; i < associativity_; i++) {
            if (set[i].insert_time < oldest) {
                oldest = set[i].insert_time;
                victim = i;
            }
        }
        return victim;
    }

    size_t findVictimRandom(std::vector<CacheLine>& set) {
        std::uniform_int_distribution<size_t> dist(0, associativity_ - 1);
        return dist(rng_);
    }

    size_t findVictimLFU(std::vector<CacheLine>& set) {
        // LFU aging: periodically halve access counts to prevent stale data
        // from staying cached forever. Triggered every 1024 evictions.
        // (ECG reference ages by decrementing; halving is equivalent and faster.)
        if ((stats_.evictions.load() & 1023) == 0) {
            for (size_t i = 0; i < associativity_; i++) {
                if (set[i].valid) set[i].access_count >>= 1;
            }
        }
        size_t victim = 0;
        uint64_t min_count = set[0].access_count;
        for (size_t i = 1; i < associativity_; i++) {
            if (set[i].access_count < min_count) {
                min_count = set[i].access_count;
                victim = i;
            }
        }
        return victim;
    }

    size_t findVictimPLRU(std::vector<CacheLine>& set) {
        // Simplified PLRU: use LRU for now
        // Full tree-PLRU would need additional state
        return findVictimLRU(set);
    }

    size_t findVictimSRRIP(std::vector<CacheLine>& set) {
        // Find line with RRPV = 3 (distant re-reference)
        while (true) {
            for (size_t i = 0; i < associativity_; i++) {
                if (set[i].rrpv == 3) return i;
            }
            // Increment all RRPVs
            for (size_t i = 0; i < associativity_; i++) {
                if (set[i].rrpv < 3) set[i].rrpv++;
            }
        }
    }

    // ================================================================
    // GRASP: Graph-aware cache Replacement with Software Prefetching
    // (Faldu et al., HPCA 2020 — reference: grasp.cpp)
    //
    // RRIP-based policy with 3-tier insertion depending on address region:
    //   High-reuse (hot hubs):    insert RRPV = 1 (P_RRIP), hit → 0
    //   Moderate-reuse:           insert RRPV = M-1 (I_RRIP), hit → decrement
    //   Low-reuse (cold/other):   insert RRPV = M (M_RRIP), hit → decrement
    // Eviction: find way with max RRPV, age all if none at max.
    // Requires DBG-reordered graph so hot vertices occupy low addresses.
    // ================================================================
    size_t findVictimGRASP(std::vector<CacheLine>& set) {
        // Same eviction as SRRIP: find way with rrpv == max, age until found
        constexpr uint8_t M_RRIP = 7;  // 3-bit RRPV, max = 2^3-1
        while (true) {
            for (size_t i = 0; i < associativity_; i++) {
                if (set[i].rrpv >= M_RRIP) return i;
            }
            for (size_t i = 0; i < associativity_; i++) {
                if (set[i].rrpv < M_RRIP) set[i].rrpv++;
            }
        }
    }

    // ================================================================
    // P-OPT: Practical Optimal cache replacement for Graph Analytics
    // (Balaji et al., HPCA 2021 — reference: llc.cpp)
    //
    // Uses the graph's transpose (encoded in a compressed rereference
    // matrix) to predict exactly when each cache line will be accessed
    // again. Evicts the line with the furthest next-reference distance.
    // When multiple lines tie on rereference distance, uses RRIP aging
    // to break ties (matching the reference implementation).
    //
    // Requires: initPOPT() called before simulation with a precomputed
    // rereference matrix from makeOffsetMatrix() in popt.h.
    // ================================================================
    size_t findVictimPOPT(std::vector<CacheLine>& set) {
        // Check if either unified context or legacy state is available
        bool has_popt = (graph_ctx_ && graph_ctx_->rereference.matrix) || popt_state_.enabled;
        if (!has_popt) {
            return findVictimLRU(set);  // Fallback if not initialized
        }

        // Phase 1: Evict non-graph data first (streaming/CSR metadata)
        for (size_t i = 0; i < associativity_; i++) {
            uint64_t la = set[i].line_addr;
            bool is_graph_data = false;
            if (graph_ctx_) {
                is_graph_data = graph_ctx_->isPropertyData(la);
            } else {
                is_graph_data = (la >= popt_state_.irreg_base && la < popt_state_.irreg_bound);
            }
            if (!is_graph_data) return i;  // Not graph vertex data — evict immediately
        }

        // Phase 2: All ways contain graph vertex data — find max rereference distance
        uint8_t maxRerefDist = 0;
        uint8_t wayRerefDists[64] = {};
        for (size_t i = 0; i < associativity_; i++) {
            uint64_t la = set[i].line_addr;
            uint32_t dist;
            if (graph_ctx_) {
                dist = graph_ctx_->findNextRef(la);
            } else {
                uint32_t cline_id = static_cast<uint32_t>(
                    (la - popt_state_.irreg_base) / line_size_);
                dist = popt_state_.findNextRef(cline_id);
            }
            uint8_t d = static_cast<uint8_t>(std::min(dist, uint32_t(127)));
            wayRerefDists[i] = d;
            if (d > maxRerefDist) maxRerefDist = d;
        }

        // Phase 3: RRIP tiebreaker among lines with max rereference distance
        // (matching reference llc.cpp: age RRPV only for tied lines)
        constexpr uint8_t M_RRPV = 7;
        while (true) {
            for (size_t i = 0; i < associativity_; i++) {
                if (wayRerefDists[i] == maxRerefDist && set[i].rrpv >= M_RRPV) {
                    return i;
                }
            }
            // Age only the tied lines
            for (size_t i = 0; i < associativity_; i++) {
                if (wayRerefDists[i] == maxRerefDist && set[i].rrpv < M_RRPV) {
                    set[i].rrpv++;
                }
            }
        }
    }

    // ================================================================
    // ECG: Graph-aware cache replacement (Mughrabi et al., GrAPL)
    //
    // Layered eviction with mode-dependent tiebreaker priority:
    //   Level 1 (all modes): SRRIP aging — find max RRPV, age until found
    //   Level 2/3 depend on ECGMode:
    //     DBG_PRIMARY:  L2=DBG tier (coldest vertex), L3=dynamic P-OPT
    //     POPT_PRIMARY: L2=dynamic P-OPT (furthest future), L3=DBG tier
    //     DBG_ONLY:     L2=DBG tier, no L3
    //
    // Key design points:
    //  - RRPV set at insert from DBG tier (bucketToRRPV), ages via SRRIP
    //  - ecg_dbg_tier stored per-line (structural, constant)
    //  - P-OPT consulted dynamically via findNextRef() at eviction time
    //    (not cached — avoids stale snapshot problem)
    // ================================================================
    size_t findVictimECG(std::vector<CacheLine>& set) {
        uint8_t rrpv_max = (graph_ctx_ && graph_ctx_->mask_config.enabled)
            ? graph_ctx_->mask_config.rrpv_max : 7;

        // Determine ECG mode
        ECGMode mode = (graph_ctx_ && graph_ctx_->mask_config.enabled)
            ? graph_ctx_->mask_config.ecg_mode : ECGMode::DBG_PRIMARY;

        // ── Phase 0: Evict non-property data first (matching P-OPT Phase 1) ──
        // Non-property data (CSR edges, offsets, streaming) has no oracle
        // prediction and should be evicted before property data.
        // For DBG modes, this is already handled by RRPV=7 at insert (cold).
        // For POPT_PRIMARY, all lines get RRPV=6 at insert, so non-property
        // lines compete equally — we must explicitly prefer evicting them.
        if (graph_ctx_ && mode == ECGMode::POPT_PRIMARY) {
            for (size_t i = 0; i < associativity_; i++) {
                if (!graph_ctx_->isPropertyData(set[i].line_addr)) {
                    return i;  // Evict non-property data immediately
                }
            }
        }

        // ── POPT_PRIMARY: Use P-OPT's exact 3-phase algorithm ──
        // Bypass SRRIP aging entirely — P-OPT operates on ALL ways, not
        // just RRPV-aged candidates. This ensures ECG(POPT_PRIMARY)
        // matches pure P-OPT's eviction behavior exactly.
        if (mode == ECGMode::POPT_PRIMARY && graph_ctx_ && graph_ctx_->rereference.matrix) {
            // Phase 2: Find max rereference distance across ALL ways
            uint8_t maxRerefDist = 0;
            uint8_t wayRerefDists[64] = {};
            for (size_t i = 0; i < associativity_; i++) {
                uint32_t dist = graph_ctx_->findNextRef(set[i].line_addr);
                uint8_t d = static_cast<uint8_t>(std::min(dist, uint32_t(127)));
                wayRerefDists[i] = d;
                if (d > maxRerefDist) maxRerefDist = d;
            }

            // Phase 3: RRIP tiebreak among max-distance lines
            // (age only tied lines, matching P-OPT's Algorithm 2)
            // Level 3 enhancement: among RRIP ties, prefer highest DBG tier
            constexpr uint8_t M_RRPV = 7;
            while (true) {
                size_t best = SIZE_MAX;
                uint8_t best_dbg = 0;
                for (size_t i = 0; i < associativity_; i++) {
                    if (wayRerefDists[i] == maxRerefDist && set[i].rrpv >= M_RRPV) {
                        if (best == SIZE_MAX || set[i].ecg_dbg_tier > best_dbg) {
                            best = i;
                            best_dbg = set[i].ecg_dbg_tier;
                        }
                    }
                }
                if (best != SIZE_MAX) return best;

                // Age only the tied lines
                for (size_t i = 0; i < associativity_; i++) {
                    if (wayRerefDists[i] == maxRerefDist && set[i].rrpv < M_RRPV) {
                        set[i].rrpv++;
                    }
                }
            }
        }

        // ── Level 1: SRRIP aging — find lines at max RRPV ──
        // Age all lines until at least one reaches rrpv_max.
        while (true) {
            bool found = false;
            for (size_t i = 0; i < associativity_; i++) {
                if (set[i].rrpv >= rrpv_max) { found = true; break; }
            }
            if (found) break;
            for (size_t i = 0; i < associativity_; i++) {
                if (set[i].rrpv < rrpv_max) set[i].rrpv++;
            }
        }

        // Collect candidates at max RRPV
        size_t candidates[64];
        size_t num_candidates = 0;
        for (size_t i = 0; i < associativity_; i++) {
            if (set[i].rrpv >= rrpv_max && num_candidates < 64)
                candidates[num_candidates++] = i;
        }
        if (num_candidates == 1) return candidates[0];

        // ── Level 2 tiebreak (mode-dependent) ──
        if (mode == ECGMode::DBG_PRIMARY || mode == ECGMode::DBG_ONLY) {
            // DBG tiebreak: evict highest ecg_dbg_tier (coldest/lowest-degree)
            uint8_t max_dbg = 0;
            for (size_t c = 0; c < num_candidates; c++)
                if (set[candidates[c]].ecg_dbg_tier > max_dbg)
                    max_dbg = set[candidates[c]].ecg_dbg_tier;

            // Narrow candidates to max-DBG lines
            size_t narrowed[64];
            size_t num_narrowed = 0;
            for (size_t c = 0; c < num_candidates; c++)
                if (set[candidates[c]].ecg_dbg_tier == max_dbg)
                    narrowed[num_narrowed++] = candidates[c];

            if (num_narrowed == 1 || mode == ECGMode::DBG_ONLY)
                return narrowed[0];

            // ── Level 3: Dynamic P-OPT tiebreak via rereference matrix ──
            if (graph_ctx_ && graph_ctx_->rereference.matrix) {
                uint32_t max_dist = 0;
                size_t victim = narrowed[0];
                for (size_t c = 0; c < num_narrowed; c++) {
                    uint32_t dist = graph_ctx_->findNextRef(set[narrowed[c]].line_addr);
                    if (dist > max_dist) {
                        max_dist = dist;
                        victim = narrowed[c];
                    }
                }
                return victim;
            }
            return narrowed[0];

        } else if (mode == ECGMode::ECG_EMBEDDED) {
            // ECG_EMBEDDED: stored P-OPT hint as Level 2 (zero LLC overhead).
            // Evict line with highest stored rereference hint (furthest future).
            // Falls back to DBG tier as Level 3 tiebreaker.
            uint8_t max_hint = 0;
            for (size_t c = 0; c < num_candidates; c++)
                if (set[candidates[c]].ecg_popt_hint > max_hint)
                    max_hint = set[candidates[c]].ecg_popt_hint;

            size_t narrowed[64];
            size_t num_narrowed = 0;
            for (size_t c = 0; c < num_candidates; c++)
                if (set[candidates[c]].ecg_popt_hint == max_hint)
                    narrowed[num_narrowed++] = candidates[c];

            if (num_narrowed == 1) return narrowed[0];

            // Level 3: DBG tier tiebreak among same-hint lines
            uint8_t max_dbg = 0;
            size_t victim = narrowed[0];
            for (size_t c = 0; c < num_narrowed; c++) {
                if (set[narrowed[c]].ecg_dbg_tier > max_dbg) {
                    max_dbg = set[narrowed[c]].ecg_dbg_tier;
                    victim = narrowed[c];
                }
            }
            return victim;

        } else {  // POPT_PRIMARY fallback (no matrix available)
            // Fall back to DBG tiebreak if no P-OPT matrix

            // No P-OPT matrix: fall back to DBG tiebreak
            uint8_t max_dbg = 0;
            size_t victim = candidates[0];
            for (size_t c = 0; c < num_candidates; c++) {
                if (set[candidates[c]].ecg_dbg_tier > max_dbg) {
                    max_dbg = set[candidates[c]].ecg_dbg_tier;
                    victim = candidates[c];
                }
            }
            return victim;
        }
    }

    std::string name_;
    size_t size_bytes_;
    size_t line_size_;
    size_t associativity_;
    size_t num_sets_;
    size_t offset_bits_;
    size_t index_bits_;
    EvictionPolicy policy_;
    
    std::vector<std::vector<CacheLine>> cache_;
    CacheStats stats_;
    uint64_t global_time_ = 0;
    std::mt19937 rng_;
    std::mutex mutex_;

    POPTState popt_state_;    // P-OPT rereference matrix state (legacy, used if no GraphCacheContext)
    GRASPState grasp_state_;  // GRASP degree-aware state (legacy, used if no GraphCacheContext)
    const GraphCacheContext* graph_ctx_ = nullptr;  // Unified graph-aware context (preferred)
};

// ============================================================================
// Cache Hierarchy (L1 -> L2 -> L3)
// ============================================================================
class CacheHierarchy {
public:
    // Default: Intel-like hierarchy
    // L1: 32KB, 8-way, 64B lines
    // L2: 256KB, 4-way, 64B lines
    // L3: 8MB, 16-way, 64B lines
    CacheHierarchy(
        size_t l1_size = 32 * 1024,
        size_t l1_ways = 8,
        size_t l2_size = 256 * 1024,
        size_t l2_ways = 4,
        size_t l3_size = 8 * 1024 * 1024,
        size_t l3_ways = 16,
        size_t line_size = 64,
        EvictionPolicy policy = EvictionPolicy::LRU
    ) : line_size_(line_size), enabled_(true) {
        l1_ = std::make_unique<CacheLevel>("L1", l1_size, line_size, l1_ways, policy);
        l2_ = std::make_unique<CacheLevel>("L2", l2_size, line_size, l2_ways, policy);
        l3_ = std::make_unique<CacheLevel>("L3", l3_size, line_size, l3_ways, policy);
    }

    // Configure from environment variables
    static CacheHierarchy fromEnvironment() {
        size_t l1_size = getEnvSize("CACHE_L1_SIZE", 32 * 1024);
        size_t l1_ways = getEnvSize("CACHE_L1_WAYS", 8);
        size_t l2_size = getEnvSize("CACHE_L2_SIZE", 256 * 1024);
        size_t l2_ways = getEnvSize("CACHE_L2_WAYS", 4);
        size_t l3_size = getEnvSize("CACHE_L3_SIZE", 8 * 1024 * 1024);
        size_t l3_ways = getEnvSize("CACHE_L3_WAYS", 16);
        size_t line_size = getEnvSize("CACHE_LINE_SIZE", 64);
        
        const char* policy_str = std::getenv("CACHE_POLICY");
        EvictionPolicy policy = policy_str ? StringToPolicy(policy_str) : EvictionPolicy::LRU;
        
        return CacheHierarchy(l1_size, l1_ways, l2_size, l2_ways,
                             l3_size, l3_ways, line_size, policy);
    }

    // Main access function - simulates hierarchical access
    void access(uint64_t address, bool is_write = false) {
        if (!enabled_) return;
        
        total_accesses_++;
        
        // Try L1
        if (l1_->access(address, is_write)) {
            return;  // L1 hit
        }
        
        // L1 miss, try L2
        if (l2_->access(address, is_write)) {
            l1_->insert(address, is_write);  // Bring to L1
            return;  // L2 hit
        }
        
        // L2 miss, try L3
        if (l3_->access(address, is_write)) {
            l2_->insert(address, is_write);  // Bring to L2
            l1_->insert(address, is_write);  // Bring to L1
            return;  // L3 hit
        }
        
        // L3 miss - fetch from memory
        memory_accesses_++;
        l3_->insert(address, is_write);
        l2_->insert(address, is_write);
        l1_->insert(address, is_write);
    }

    // Convenience methods for common access patterns
    template<typename T>
    void read(const T* ptr) {
        access(reinterpret_cast<uint64_t>(ptr), false);
    }

    template<typename T>
    void write(T* ptr) {
        access(reinterpret_cast<uint64_t>(ptr), true);
    }

    // Read an array element
    template<typename T>
    void readArray(const T* arr, size_t index) {
        access(reinterpret_cast<uint64_t>(&arr[index]), false);
    }

    // Write an array element
    template<typename T>
    void writeArray(T* arr, size_t index) {
        access(reinterpret_cast<uint64_t>(&arr[index]), true);
    }

    // Read a range (e.g., CSR row)
    template<typename T>
    void readRange(const T* arr, size_t start, size_t end) {
        for (size_t i = start; i < end; i++) {
            // Only access each cache line once
            uint64_t addr = reinterpret_cast<uint64_t>(&arr[i]);
            access(addr, false);
        }
    }

    // Reset all statistics
    void resetStats() {
        l1_->resetStats();
        l2_->resetStats();
        l3_->resetStats();
        total_accesses_ = 0;
        memory_accesses_ = 0;
    }

    // Enable/disable simulation
    void enable() { enabled_ = true; }
    void disable() { enabled_ = false; }
    bool isEnabled() const { return enabled_; }

    // Get cache levels
    CacheLevel* L1() { return l1_.get(); }
    CacheLevel* L2() { return l2_.get(); }
    CacheLevel* L3() { return l3_.get(); }

    // ================================================================
    // Unified GraphCacheContext (preferred over legacy init methods)
    // ================================================================
    void initGraphContext(const GraphCacheContext* ctx) {
        l1_->initGraphContext(ctx);
        l2_->initGraphContext(ctx);
        l3_->initGraphContext(ctx);
    }

    // P-OPT: Initialize rereference matrix on LLC (L3) — legacy API
    void initPOPT(const uint8_t* reref_matrix, uint64_t irreg_base,
                  uint64_t irreg_bound, uint32_t num_vertices,
                  uint32_t num_epochs = 256) {
        l3_->initPOPT(reref_matrix, irreg_base, irreg_bound, num_vertices, num_epochs);
    }

    // GRASP: Initialize degree-aware RRIP retention — legacy API
    void initGRASP(uint64_t data_ptr, uint32_t num_vertices,
                   size_t elem_size, double hot_fraction = 0.1) {
        size_t llc_size = l3_->getSizeBytes();
        l1_->initGRASP(data_ptr, num_vertices, elem_size, l1_->getSizeBytes(), hot_fraction);
        l2_->initGRASP(data_ptr, num_vertices, elem_size, l2_->getSizeBytes(), hot_fraction);
        l3_->initGRASP(data_ptr, num_vertices, elem_size, llc_size, hot_fraction);
    }

    // P-OPT: Update current vertex (call at each outer-loop iteration)
    void setCurrentVertex(uint32_t vertex_id) {
        l3_->setCurrentVertex(vertex_id);
    }

    uint64_t getTotalAccesses() const { return total_accesses_; }
    uint64_t getMemoryAccesses() const { return memory_accesses_; }

    // Print statistics
    void printStats(std::ostream& os = std::cout) const {
        os << "\n";
        os << "╔══════════════════════════════════════════════════════════════════╗\n";
        os << "║                    CACHE SIMULATION RESULTS                      ║\n";
        os << "╠══════════════════════════════════════════════════════════════════╣\n";
        
        printLevelStats(os, *l1_);
        printLevelStats(os, *l2_);
        printLevelStats(os, *l3_);
        
        os << "╠══════════════════════════════════════════════════════════════════╣\n";
        os << "║ SUMMARY                                                          ║\n";
        os << "╠══════════════════════════════════════════════════════════════════╣\n";
        os << "║ Total Accesses:      " << std::setw(15) << total_accesses_ 
           << "                          ║\n";
        os << "║ Memory Accesses:     " << std::setw(15) << memory_accesses_
           << "                          ║\n";
        os << "║ Overall Hit Rate:    " << std::setw(14) << std::fixed 
           << std::setprecision(4) << (1.0 - (double)memory_accesses_ / total_accesses_)
           << "%                          ║\n";
        os << "╚══════════════════════════════════════════════════════════════════╝\n";
    }

    // Export statistics as JSON
    std::string toJSON() const {
        std::ostringstream ss;
        ss << "{\n";
        ss << "  \"total_accesses\": " << total_accesses_ << ",\n";
        ss << "  \"memory_accesses\": " << memory_accesses_ << ",\n";
        ss << "  \"L1\": " << levelToJSON(*l1_) << ",\n";
        ss << "  \"L2\": " << levelToJSON(*l2_) << ",\n";
        ss << "  \"L3\": " << levelToJSON(*l3_) << "\n";
        ss << "}";
        return ss.str();
    }

    // Get stats as feature vector for perceptron
    std::vector<double> getFeatures() const {
        const auto& l1s = l1_->getStats();
        const auto& l2s = l2_->getStats();
        const auto& l3s = l3_->getStats();
        
        return {
            l1s.hitRate(),
            l2s.hitRate(),
            l3s.hitRate(),
            (double)memory_accesses_ / total_accesses_,  // DRAM access rate
            (double)l1s.evictions / l1s.totalAccesses(), // L1 eviction rate
            (double)l2s.evictions / l2s.totalAccesses(), // L2 eviction rate
            (double)l3s.evictions / l3s.totalAccesses(), // L3 eviction rate
        };
    }

private:
    static size_t getEnvSize(const char* name, size_t default_val) {
        const char* val = std::getenv(name);
        if (!val) return default_val;
        
        char* end;
        size_t result = std::strtoul(val, &end, 10);
        
        // Handle K, M, G suffixes
        if (*end == 'K' || *end == 'k') result *= 1024;
        else if (*end == 'M' || *end == 'm') result *= 1024 * 1024;
        else if (*end == 'G' || *end == 'g') result *= 1024 * 1024 * 1024;
        
        return result > 0 ? result : default_val;
    }

    void printLevelStats(std::ostream& os, const CacheLevel& level) const {
        const auto& stats = level.getStats();
        os << "╠──────────────────────────────────────────────────────────────────╣\n";
        os << "║ " << level.getName() << " Cache (" 
           << formatSize(level.getSizeBytes()) << ", "
           << level.getAssociativity() << "-way, "
           << PolicyToString(level.getPolicy()) << ")"
           << std::string(40 - level.getName().length() - 
                         formatSize(level.getSizeBytes()).length() -
                         std::to_string(level.getAssociativity()).length() -
                         PolicyToString(level.getPolicy()).length(), ' ')
           << "║\n";
        os << "║   Hits:              " << std::setw(15) << stats.hits.load()
           << "                          ║\n";
        os << "║   Misses:            " << std::setw(15) << stats.misses.load()
           << "                          ║\n";
        os << "║   Hit Rate:          " << std::setw(14) << std::fixed 
           << std::setprecision(4) << (stats.hitRate() * 100) << "%"
           << "                          ║\n";
        os << "║   Evictions:         " << std::setw(15) << stats.evictions.load()
           << "                          ║\n";
    }

    std::string levelToJSON(const CacheLevel& level) const {
        const auto& stats = level.getStats();
        std::ostringstream ss;
        ss << "{\n";
        ss << "    \"size_bytes\": " << level.getSizeBytes() << ",\n";
        ss << "    \"ways\": " << level.getAssociativity() << ",\n";
        ss << "    \"sets\": " << level.getNumSets() << ",\n";
        ss << "    \"line_size\": " << level.getLineSize() << ",\n";
        ss << "    \"policy\": \"" << PolicyToString(level.getPolicy()) << "\",\n";
        ss << "    \"hits\": " << stats.hits.load() << ",\n";
        ss << "    \"misses\": " << stats.misses.load() << ",\n";
        ss << "    \"hit_rate\": " << std::fixed << std::setprecision(6) << stats.hitRate() << ",\n";
        ss << "    \"evictions\": " << stats.evictions.load() << ",\n";
        ss << "    \"writebacks\": " << stats.writebacks.load() << "\n";
        ss << "  }";
        return ss.str();
    }

    static std::string formatSize(size_t bytes) {
        if (bytes >= 1024 * 1024 * 1024) {
            return std::to_string(bytes / (1024 * 1024 * 1024)) + "GB";
        } else if (bytes >= 1024 * 1024) {
            return std::to_string(bytes / (1024 * 1024)) + "MB";
        } else if (bytes >= 1024) {
            return std::to_string(bytes / 1024) + "KB";
        }
        return std::to_string(bytes) + "B";
    }

    std::unique_ptr<CacheLevel> l1_;
    std::unique_ptr<CacheLevel> l2_;
    std::unique_ptr<CacheLevel> l3_;
    size_t line_size_;
    bool enabled_;
    std::atomic<uint64_t> total_accesses_{0};
    std::atomic<uint64_t> memory_accesses_{0};
};

// ============================================================================
// FAST Cache Hierarchy - NO LOCKS, optimized for single-threaded simulation
// ~10-20x faster than CacheHierarchy, exact results
// ============================================================================
class FastCacheHierarchy {
public:
    FastCacheHierarchy(
        size_t l1_size = 32 * 1024,
        size_t l1_ways = 8,
        size_t l2_size = 256 * 1024,
        size_t l2_ways = 8,
        size_t l3_size = 8 * 1024 * 1024,
        size_t l3_ways = 16,
        size_t line_size = 64
    ) : line_size_(line_size), enabled_(true) {
        l1_ = std::make_unique<FastCacheLevel>("L1", l1_size, line_size, l1_ways);
        l2_ = std::make_unique<FastCacheLevel>("L2", l2_size, line_size, l2_ways);
        l3_ = std::make_unique<FastCacheLevel>("L3", l3_size, line_size, l3_ways);
    }

    static FastCacheHierarchy fromEnvironment() {
        size_t l1_size = getEnvSize("CACHE_L1_SIZE", 32 * 1024);
        size_t l1_ways = getEnvSize("CACHE_L1_WAYS", 8);
        size_t l2_size = getEnvSize("CACHE_L2_SIZE", 256 * 1024);
        size_t l2_ways = getEnvSize("CACHE_L2_WAYS", 8);
        size_t l3_size = getEnvSize("CACHE_L3_SIZE", 8 * 1024 * 1024);
        size_t l3_ways = getEnvSize("CACHE_L3_WAYS", 16);
        size_t line_size = getEnvSize("CACHE_LINE_SIZE", 64);
        
        return FastCacheHierarchy(l1_size, l1_ways, l2_size, l2_ways,
                                  l3_size, l3_ways, line_size);
    }

    // FAST access - no locks, inline
    __attribute__((always_inline))
    inline void access(uint64_t address, bool is_write = false) {
        if (!enabled_) return;
        
        total_accesses_++;
        address = address & ~(line_size_ - 1);  // Align to cache line
        
        if (l1_->access(address)) return;
        if (l2_->access(address)) { l1_->insert(address); return; }
        if (l3_->access(address)) { l2_->insert(address); l1_->insert(address); return; }
        
        memory_accesses_++;
        l3_->insert(address);
        l2_->insert(address);
        l1_->insert(address);
    }

    template<typename T>
    inline void read(const T* ptr) { access(reinterpret_cast<uint64_t>(ptr), false); }

    template<typename T>
    inline void write(T* ptr) { access(reinterpret_cast<uint64_t>(ptr), true); }

    template<typename T>
    inline void readArray(const T* arr, size_t index) {
        access(reinterpret_cast<uint64_t>(&arr[index]), false);
    }

    template<typename T>
    inline void writeArray(T* arr, size_t index) {
        access(reinterpret_cast<uint64_t>(&arr[index]), true);
    }

    void resetStats() {
        l1_->resetStats();
        l2_->resetStats();
        l3_->resetStats();
        total_accesses_ = 0;
        memory_accesses_ = 0;
    }

    void enable() { enabled_ = true; }
    void disable() { enabled_ = false; }
    bool isEnabled() const { return enabled_; }

    // No-op: FastCacheHierarchy uses clock algorithm, not policy-based eviction
    void setCurrentVertex(uint32_t) {}
    void initGraphContext(const GraphCacheContext*) {}
    
    uint64_t getTotalAccesses() const { return total_accesses_; }
    uint64_t getMemoryAccesses() const { return memory_accesses_; }

    void printStats(std::ostream& os = std::cout) const {
        os << "\n";
        os << "╔══════════════════════════════════════════════════════════════════╗\n";
        os << "║              FAST CACHE SIMULATION RESULTS                       ║\n";
        os << "╠══════════════════════════════════════════════════════════════════╣\n";
        
        printFastLevelStats(os, *l1_);
        printFastLevelStats(os, *l2_);
        printFastLevelStats(os, *l3_);
        
        os << "╠══════════════════════════════════════════════════════════════════╣\n";
        os << "║ SUMMARY                                                          ║\n";
        os << "╠══════════════════════════════════════════════════════════════════╣\n";
        os << "║ Total Accesses:      " << std::setw(15) << total_accesses_ 
           << "                          ║\n";
        os << "║ Memory Accesses:     " << std::setw(15) << memory_accesses_
           << "                          ║\n";
        double overall = total_accesses_ > 0 ? 
            (1.0 - (double)memory_accesses_ / total_accesses_) : 0.0;
        os << "║ Overall Hit Rate:    " << std::setw(14) << std::fixed 
           << std::setprecision(4) << (overall * 100) << "%"
           << "                          ║\n";
        os << "╚══════════════════════════════════════════════════════════════════╝\n";
    }

    std::string toJSON() const {
        auto& l1 = l1_->getStats();
        auto& l2 = l2_->getStats();
        auto& l3 = l3_->getStats();
        
        std::ostringstream ss;
        ss << "{\n";
        ss << "  \"mode\": \"fast\",\n";
        ss << "  \"total_accesses\": " << total_accesses_ << ",\n";
        ss << "  \"memory_accesses\": " << memory_accesses_ << ",\n";
        ss << "  \"L1\": { \"hits\": " << l1.hits.load() << ", \"misses\": " << l1.misses.load() 
           << ", \"hit_rate\": " << std::fixed << std::setprecision(6) << l1.hitRate() << " },\n";
        ss << "  \"L2\": { \"hits\": " << l2.hits.load() << ", \"misses\": " << l2.misses.load() 
           << ", \"hit_rate\": " << std::fixed << std::setprecision(6) << l2.hitRate() << " },\n";
        ss << "  \"L3\": { \"hits\": " << l3.hits.load() << ", \"misses\": " << l3.misses.load() 
           << ", \"hit_rate\": " << std::fixed << std::setprecision(6) << l3.hitRate() << " }\n";
        ss << "}";
        return ss.str();
    }

    std::vector<double> getFeatures() const {
        const auto& l1s = l1_->getStats();
        const auto& l2s = l2_->getStats();
        const auto& l3s = l3_->getStats();
        
        return {
            l1s.hitRate(),
            l2s.hitRate(),
            l3s.hitRate(),
            total_accesses_ > 0 ? (double)memory_accesses_ / total_accesses_ : 0.0,
            l1s.totalAccesses() > 0 ? (double)l1s.evictions / l1s.totalAccesses() : 0.0,
            l2s.totalAccesses() > 0 ? (double)l2s.evictions / l2s.totalAccesses() : 0.0,
            l3s.totalAccesses() > 0 ? (double)l3s.evictions / l3s.totalAccesses() : 0.0,
        };
    }

private:
    static size_t getEnvSize(const char* name, size_t default_val) {
        const char* val = std::getenv(name);
        if (!val) return default_val;
        char* end;
        size_t result = std::strtoul(val, &end, 10);
        if (*end == 'K' || *end == 'k') result *= 1024;
        else if (*end == 'M' || *end == 'm') result *= 1024 * 1024;
        else if (*end == 'G' || *end == 'g') result *= 1024 * 1024 * 1024;
        return result > 0 ? result : default_val;
    }

    void printFastLevelStats(std::ostream& os, const FastCacheLevel& level) const {
        const auto& stats = level.getStats();
        os << "╠──────────────────────────────────────────────────────────────────╣\n";
        os << "║ " << level.getName() << " Cache (" 
           << formatSize(level.getSizeBytes()) << ", "
           << level.getAssociativity() << "-way, Clock)"
           << std::string(42 - level.getName().length() - 
                         formatSize(level.getSizeBytes()).length() -
                         std::to_string(level.getAssociativity()).length(), ' ')
           << "║\n";
        os << "║   Hits:              " << std::setw(15) << stats.hits.load()
           << "                          ║\n";
        os << "║   Misses:            " << std::setw(15) << stats.misses.load()
           << "                          ║\n";
        os << "║   Hit Rate:          " << std::setw(14) << std::fixed 
           << std::setprecision(4) << (stats.hitRate() * 100) << "%"
           << "                          ║\n";
    }

    static std::string formatSize(size_t bytes) {
        if (bytes >= 1024 * 1024) return std::to_string(bytes / (1024 * 1024)) + "MB";
        if (bytes >= 1024) return std::to_string(bytes / 1024) + "KB";
        return std::to_string(bytes) + "B";
    }

    std::unique_ptr<FastCacheLevel> l1_;
    std::unique_ptr<FastCacheLevel> l2_;
    std::unique_ptr<FastCacheLevel> l3_;
    size_t line_size_;
    bool enabled_;
    uint64_t total_accesses_ = 0;
    uint64_t memory_accesses_ = 0;
};

// ============================================================================
// ULTRA-FAST Cache Hierarchy - Maximum performance with packed structures
// ~2-3x faster than FastCacheHierarchy through better memory layout
// ============================================================================
class UltraFastCacheHierarchy {
public:
    UltraFastCacheHierarchy(
        size_t l1_size = 32 * 1024,
        size_t l1_ways = 8,
        size_t l2_size = 256 * 1024,
        size_t l2_ways = 8,
        size_t l3_size = 8 * 1024 * 1024,
        size_t l3_ways = 16,
        size_t line_size = 64
    ) : line_size_(line_size), line_mask_(~(line_size - 1)), enabled_(true),
        total_accesses_(0), memory_accesses_(0) {
        l1_ = std::make_unique<UltraFastCacheLevel>("L1", l1_size, line_size, l1_ways);
        l2_ = std::make_unique<UltraFastCacheLevel>("L2", l2_size, line_size, l2_ways);
        l3_ = std::make_unique<UltraFastCacheLevel>("L3", l3_size, line_size, l3_ways);
    }

    static UltraFastCacheHierarchy fromEnvironment() {
        size_t l1_size = getEnvSize("CACHE_L1_SIZE", 32 * 1024);
        size_t l1_ways = getEnvSize("CACHE_L1_WAYS", 8);
        size_t l2_size = getEnvSize("CACHE_L2_SIZE", 256 * 1024);
        size_t l2_ways = getEnvSize("CACHE_L2_WAYS", 8);
        size_t l3_size = getEnvSize("CACHE_L3_SIZE", 8 * 1024 * 1024);
        size_t l3_ways = getEnvSize("CACHE_L3_WAYS", 16);
        size_t line_size = getEnvSize("CACHE_LINE_SIZE", 64);
        
        return UltraFastCacheHierarchy(l1_size, l1_ways, l2_size, l2_ways,
                                       l3_size, l3_ways, line_size);
    }

    // ULTRA-FAST access - maximum inlining, minimal branching
    __attribute__((always_inline, hot))
    inline void access(uint64_t address, bool is_write = false) {
        total_accesses_++;
        address &= line_mask_;
        
        if (__builtin_expect(l1_->access(address), 1)) return;
        if (l2_->access(address)) { l1_->insert(address); return; }
        if (l3_->access(address)) { l2_->insert(address); l1_->insert(address); return; }
        
        memory_accesses_++;
        l3_->insert(address);
        l2_->insert(address);
        l1_->insert(address);
    }

    template<typename T>
    __attribute__((always_inline))
    inline void read(const T* ptr) { access(reinterpret_cast<uint64_t>(ptr)); }

    template<typename T>
    __attribute__((always_inline))
    inline void write(T* ptr) { access(reinterpret_cast<uint64_t>(ptr)); }

    template<typename T>
    __attribute__((always_inline))
    inline void readArray(const T* arr, size_t index) {
        access(reinterpret_cast<uint64_t>(&arr[index]));
    }

    template<typename T>
    __attribute__((always_inline))
    inline void writeArray(T* arr, size_t index) {
        access(reinterpret_cast<uint64_t>(&arr[index]));
    }

    void resetStats() {
        l1_->resetStats();
        l2_->resetStats();
        l3_->resetStats();
        total_accesses_ = 0;
        memory_accesses_ = 0;
    }

    void enable() { enabled_ = true; }
    void disable() { enabled_ = false; }
    bool isEnabled() const { return enabled_; }

    // No-op: UltraFastCacheHierarchy uses packed clock algorithm
    void setCurrentVertex(uint32_t) {}
    void initGraphContext(const GraphCacheContext*) {}
    
    uint64_t getTotalAccesses() const { return total_accesses_; }
    uint64_t getMemoryAccesses() const { return memory_accesses_; }

    void printStats(std::ostream& os = std::cout) const {
        os << "\n";
        os << "╔══════════════════════════════════════════════════════════════════╗\n";
        os << "║            ULTRA-FAST CACHE SIMULATION RESULTS                   ║\n";
        os << "╠══════════════════════════════════════════════════════════════════╣\n";
        
        printUltraFastLevelStats(os, *l1_);
        printUltraFastLevelStats(os, *l2_);
        printUltraFastLevelStats(os, *l3_);
        
        os << "╠══════════════════════════════════════════════════════════════════╣\n";
        os << "║ SUMMARY                                                          ║\n";
        os << "╠══════════════════════════════════════════════════════════════════╣\n";
        os << "║ Total Accesses:      " << std::setw(15) << total_accesses_ 
           << "                          ║\n";
        os << "║ Memory Accesses:     " << std::setw(15) << memory_accesses_
           << "                          ║\n";
        double overall = total_accesses_ > 0 ? 
            (1.0 - (double)memory_accesses_ / total_accesses_) : 0.0;
        os << "║ Overall Hit Rate:    " << std::setw(14) << std::fixed 
           << std::setprecision(4) << (overall * 100) << "%"
           << "                          ║\n";
        os << "╚══════════════════════════════════════════════════════════════════╝\n";
    }

    std::string toJSON() const {
        std::ostringstream ss;
        ss << "{\n";
        ss << "  \"mode\": \"ultrafast\",\n";
        ss << "  \"total_accesses\": " << total_accesses_ << ",\n";
        ss << "  \"memory_accesses\": " << memory_accesses_ << ",\n";
        ss << "  \"L1\": { \"hits\": " << l1_->getHits() << ", \"misses\": " << l1_->getMisses() 
           << ", \"hit_rate\": " << std::fixed << std::setprecision(6) << l1_->hitRate() << " },\n";
        ss << "  \"L2\": { \"hits\": " << l2_->getHits() << ", \"misses\": " << l2_->getMisses() 
           << ", \"hit_rate\": " << std::fixed << std::setprecision(6) << l2_->hitRate() << " },\n";
        ss << "  \"L3\": { \"hits\": " << l3_->getHits() << ", \"misses\": " << l3_->getMisses() 
           << ", \"hit_rate\": " << std::fixed << std::setprecision(6) << l3_->hitRate() << " }\n";
        ss << "}";
        return ss.str();
    }

private:
    static size_t getEnvSize(const char* name, size_t default_val) {
        const char* val = std::getenv(name);
        if (!val) return default_val;
        char* end;
        size_t result = std::strtoul(val, &end, 10);
        if (*end == 'K' || *end == 'k') result *= 1024;
        else if (*end == 'M' || *end == 'm') result *= 1024 * 1024;
        else if (*end == 'G' || *end == 'g') result *= 1024 * 1024 * 1024;
        return result > 0 ? result : default_val;
    }

    void printUltraFastLevelStats(std::ostream& os, const UltraFastCacheLevel& level) const {
        os << "╠──────────────────────────────────────────────────────────────────╣\n";
        os << "║ " << level.getName() << " Cache (" 
           << formatSize(level.getSizeBytes()) << ", "
           << level.getAssociativity() << "-way, Clock)"
           << std::string(40 - level.getName().length() - 
                         formatSize(level.getSizeBytes()).length() -
                         std::to_string(level.getAssociativity()).length(), ' ')
           << "║\n";
        os << "║   Hits:              " << std::setw(15) << level.getHits()
           << "                          ║\n";
        os << "║   Misses:            " << std::setw(15) << level.getMisses()
           << "                          ║\n";
        os << "║   Hit Rate:          " << std::setw(14) << std::fixed 
           << std::setprecision(4) << (level.hitRate() * 100) << "%"
           << "                          ║\n";
    }

    static std::string formatSize(size_t bytes) {
        if (bytes >= 1024 * 1024) return std::to_string(bytes / (1024 * 1024)) + "MB";
        if (bytes >= 1024) return std::to_string(bytes / 1024) + "KB";
        return std::to_string(bytes) + "B";
    }

    std::unique_ptr<UltraFastCacheLevel> l1_;
    std::unique_ptr<UltraFastCacheLevel> l2_;
    std::unique_ptr<UltraFastCacheLevel> l3_;
    size_t line_size_;
    uint64_t line_mask_;
    bool enabled_;
    uint64_t total_accesses_;
    uint64_t memory_accesses_;
};

// ============================================================================
// Multi-Core Cache Hierarchy (Private L1/L2 per core, Shared L3)
// Simulates realistic multi-core architecture like Intel/AMD processors
// ============================================================================
class MultiCoreCacheHierarchy {
public:
    static constexpr int MAX_CORES = 64;
    
    // Default: 8-core Intel-like hierarchy
    // Per-core: L1 32KB 8-way, L2 256KB 8-way (private)
    // Shared:   L3 8MB 16-way
    MultiCoreCacheHierarchy(
        int num_cores = 8,
        size_t l1_size = 32 * 1024,
        size_t l1_ways = 8,
        size_t l2_size = 256 * 1024,
        size_t l2_ways = 8,
        size_t l3_size = 8 * 1024 * 1024,
        size_t l3_ways = 16,
        size_t line_size = 64,
        EvictionPolicy policy = EvictionPolicy::LRU
    ) : num_cores_(num_cores), line_size_(line_size), enabled_(true) {
        
        if (num_cores_ > MAX_CORES) num_cores_ = MAX_CORES;
        if (num_cores_ < 1) num_cores_ = 1;
        
        // Create private L1 and L2 for each core
        for (int i = 0; i < num_cores_; i++) {
            l1_caches_.push_back(std::make_unique<CacheLevel>(
                "L1-Core" + std::to_string(i), l1_size, line_size, l1_ways, policy));
            l2_caches_.push_back(std::make_unique<CacheLevel>(
                "L2-Core" + std::to_string(i), l2_size, line_size, l2_ways, policy));
        }
        
        // Create shared L3 (total size, not per-core)
        l3_shared_ = std::make_unique<CacheLevel>("L3-Shared", l3_size, line_size, l3_ways, policy);
        
        // Initialize per-core statistics
        core_accesses_.resize(num_cores_, 0);
        core_memory_accesses_.resize(num_cores_, 0);
    }

    // Configure from environment variables
    static MultiCoreCacheHierarchy fromEnvironment() {
        int num_cores = static_cast<int>(getEnvSize("CACHE_NUM_CORES", 8));
        size_t l1_size = getEnvSize("CACHE_L1_SIZE", 32 * 1024);
        size_t l1_ways = getEnvSize("CACHE_L1_WAYS", 8);
        size_t l2_size = getEnvSize("CACHE_L2_SIZE", 256 * 1024);
        size_t l2_ways = getEnvSize("CACHE_L2_WAYS", 8);
        size_t l3_size = getEnvSize("CACHE_L3_SIZE", 8 * 1024 * 1024);
        size_t l3_ways = getEnvSize("CACHE_L3_WAYS", 16);
        size_t line_size = getEnvSize("CACHE_LINE_SIZE", 64);
        
        const char* policy_str = std::getenv("CACHE_POLICY");
        EvictionPolicy policy = policy_str ? StringToPolicy(policy_str) : EvictionPolicy::LRU;
        
        return MultiCoreCacheHierarchy(num_cores, l1_size, l1_ways, l2_size, l2_ways,
                                       l3_size, l3_ways, line_size, policy);
    }

    // Main access function - uses OMP thread ID to select core
    void access(uint64_t address, bool is_write = false) {
        if (!enabled_) return;
        
        int core_id = omp_get_thread_num() % num_cores_;
        accessCore(core_id, address, is_write);
    }

    // Access from specific core
    void accessCore(int core_id, uint64_t address, bool is_write = false) {
        if (!enabled_) return;
        if (core_id >= num_cores_) core_id = core_id % num_cores_;
        
        total_accesses_++;
        core_accesses_[core_id]++;
        
        CacheLevel* l1 = l1_caches_[core_id].get();
        CacheLevel* l2 = l2_caches_[core_id].get();
        
        // Try L1 (private, no contention)
        if (l1->access(address, is_write)) {
            return;  // L1 hit
        }
        
        // L1 miss, try L2 (private, no contention)
        if (l2->access(address, is_write)) {
            l1->insert(address, is_write);
            return;  // L2 hit
        }
        
        // L2 miss, try shared L3 (may have contention)
        if (l3_shared_->access(address, is_write)) {
            l2->insert(address, is_write);
            l1->insert(address, is_write);
            return;  // L3 hit
        }
        
        // L3 miss - fetch from memory
        memory_accesses_++;
        core_memory_accesses_[core_id]++;
        l3_shared_->insert(address, is_write);
        l2->insert(address, is_write);
        l1->insert(address, is_write);
    }

    // Convenience methods
    template<typename T>
    void read(const T* ptr) {
        access(reinterpret_cast<uint64_t>(ptr), false);
    }

    template<typename T>
    void write(T* ptr) {
        access(reinterpret_cast<uint64_t>(ptr), true);
    }

    template<typename T>
    void readArray(const T* arr, size_t index) {
        access(reinterpret_cast<uint64_t>(&arr[index]), false);
    }

    template<typename T>
    void writeArray(T* arr, size_t index) {
        access(reinterpret_cast<uint64_t>(&arr[index]), true);
    }

    // Reset all statistics
    void resetStats() {
        for (int i = 0; i < num_cores_; i++) {
            l1_caches_[i]->resetStats();
            l2_caches_[i]->resetStats();
            core_accesses_[i] = 0;
            core_memory_accesses_[i] = 0;
        }
        l3_shared_->resetStats();
        total_accesses_ = 0;
        memory_accesses_ = 0;
    }

    // Enable/disable simulation
    void enable() { enabled_ = true; }
    void disable() { enabled_ = false; }
    bool isEnabled() const { return enabled_; }

    // Get cache levels
    CacheLevel* L1(int core) { return l1_caches_[core % num_cores_].get(); }
    CacheLevel* L2(int core) { return l2_caches_[core % num_cores_].get(); }
    CacheLevel* L3() { return l3_shared_.get(); }
    int getNumCores() const { return num_cores_; }

    // Unified GraphCacheContext (preferred over legacy init methods)
    void initGraphContext(const GraphCacheContext* ctx) {
        for (int i = 0; i < num_cores_; i++) {
            l1_caches_[i]->initGraphContext(ctx);
            l2_caches_[i]->initGraphContext(ctx);
        }
        l3_shared_->initGraphContext(ctx);
    }

    // P-OPT: Initialize rereference matrix on shared LLC (L3) — legacy API
    void initPOPT(const uint8_t* reref_matrix, uint64_t irreg_base,
                  uint64_t irreg_bound, uint32_t num_vertices,
                  uint32_t num_epochs = 256) {
        l3_shared_->initPOPT(reref_matrix, irreg_base, irreg_bound, num_vertices, num_epochs);
    }

    // GRASP: Initialize degree-aware RRIP retention — legacy API
    void initGRASP(uint64_t data_ptr, uint32_t num_vertices,
                   size_t elem_size, double hot_fraction = 0.1) {
        size_t llc_size = l3_shared_->getSizeBytes();
        for (int i = 0; i < num_cores_; i++) {
            l1_caches_[i]->initGRASP(data_ptr, num_vertices, elem_size, l1_caches_[i]->getSizeBytes(), hot_fraction);
            l2_caches_[i]->initGRASP(data_ptr, num_vertices, elem_size, l2_caches_[i]->getSizeBytes(), hot_fraction);
        }
        l3_shared_->initGRASP(data_ptr, num_vertices, elem_size, llc_size, hot_fraction);
    }

    // P-OPT: Update current vertex (call at each outer-loop iteration)
    void setCurrentVertex(uint32_t vertex_id) {
        l3_shared_->setCurrentVertex(vertex_id);
    }

    uint64_t getTotalAccesses() const { return total_accesses_; }
    uint64_t getMemoryAccesses() const { return memory_accesses_; }
    
    // Get aggregated L1/L2 stats across all cores
    CacheStats getAggregatedL1Stats() const {
        CacheStats agg;
        for (int i = 0; i < num_cores_; i++) {
            const auto& s = l1_caches_[i]->getStats();
            agg.hits += s.hits.load();
            agg.misses += s.misses.load();
            agg.reads += s.reads.load();
            agg.writes += s.writes.load();
            agg.evictions += s.evictions.load();
            agg.writebacks += s.writebacks.load();
        }
        return agg;
    }
    
    CacheStats getAggregatedL2Stats() const {
        CacheStats agg;
        for (int i = 0; i < num_cores_; i++) {
            const auto& s = l2_caches_[i]->getStats();
            agg.hits += s.hits.load();
            agg.misses += s.misses.load();
            agg.reads += s.reads.load();
            agg.writes += s.writes.load();
            agg.evictions += s.evictions.load();
            agg.writebacks += s.writebacks.load();
        }
        return agg;
    }

    // Print statistics
    void printStats(std::ostream& os = std::cout) const {
        os << "\n";
        os << "╔══════════════════════════════════════════════════════════════════╗\n";
        os << "║          MULTI-CORE CACHE SIMULATION RESULTS (" << num_cores_ << " cores)          ║\n";
        os << "╠══════════════════════════════════════════════════════════════════╣\n";
        
        // Aggregate L1 stats
        CacheStats l1_agg = getAggregatedL1Stats();
        os << "╠──────────────────────────────────────────────────────────────────╣\n";
        os << "║ L1 Cache (Private per core, " << l1_caches_[0]->getSizeBytes()/1024 << "KB each)                       ║\n";
        os << "║   Total Hits:        " << std::setw(15) << l1_agg.hits.load()
           << "                          ║\n";
        os << "║   Total Misses:      " << std::setw(15) << l1_agg.misses.load()
           << "                          ║\n";
        os << "║   Hit Rate:          " << std::setw(14) << std::fixed 
           << std::setprecision(4) << (l1_agg.hitRate() * 100) << "%"
           << "                          ║\n";
        
        // Aggregate L2 stats
        CacheStats l2_agg = getAggregatedL2Stats();
        os << "╠──────────────────────────────────────────────────────────────────╣\n";
        os << "║ L2 Cache (Private per core, " << l2_caches_[0]->getSizeBytes()/1024 << "KB each)                      ║\n";
        os << "║   Total Hits:        " << std::setw(15) << l2_agg.hits.load()
           << "                          ║\n";
        os << "║   Total Misses:      " << std::setw(15) << l2_agg.misses.load()
           << "                          ║\n";
        os << "║   Hit Rate:          " << std::setw(14) << std::fixed 
           << std::setprecision(4) << (l2_agg.hitRate() * 100) << "%"
           << "                          ║\n";
        
        // L3 shared stats
        const auto& l3_stats = l3_shared_->getStats();
        os << "╠──────────────────────────────────────────────────────────────────╣\n";
        os << "║ L3 Cache (Shared, " << l3_shared_->getSizeBytes()/(1024*1024) << "MB)                                      ║\n";
        os << "║   Hits:              " << std::setw(15) << l3_stats.hits.load()
           << "                          ║\n";
        os << "║   Misses:            " << std::setw(15) << l3_stats.misses.load()
           << "                          ║\n";
        os << "║   Hit Rate:          " << std::setw(14) << std::fixed 
           << std::setprecision(4) << (l3_stats.hitRate() * 100) << "%"
           << "                          ║\n";
        
        // Summary
        os << "╠══════════════════════════════════════════════════════════════════╣\n";
        os << "║ SUMMARY                                                          ║\n";
        os << "╠══════════════════════════════════════════════════════════════════╣\n";
        os << "║ Total Accesses:      " << std::setw(15) << total_accesses_.load() 
           << "                          ║\n";
        os << "║ Memory Accesses:     " << std::setw(15) << memory_accesses_.load()
           << "                          ║\n";
        double overall = total_accesses_ > 0 ? 
            (1.0 - (double)memory_accesses_.load() / total_accesses_.load()) : 0.0;
        os << "║ Overall Hit Rate:    " << std::setw(14) << std::fixed 
           << std::setprecision(4) << (overall * 100) << "%"
           << "                          ║\n";
        os << "╚══════════════════════════════════════════════════════════════════╝\n";
        
        // Per-core breakdown (optional, detailed)
        os << "\nPer-Core Statistics:\n";
        os << "  Core │     Accesses │ Memory Acc │ Mem Rate\n";
        os << "───────┼──────────────┼────────────┼─────────\n";
        for (int i = 0; i < num_cores_; i++) {
            double rate = core_accesses_[i] > 0 ? 
                (double)core_memory_accesses_[i] / core_accesses_[i] * 100 : 0;
            os << std::setw(6) << i << " │ " 
               << std::setw(12) << core_accesses_[i] << " │ "
               << std::setw(10) << core_memory_accesses_[i] << " │ "
               << std::setw(6) << std::fixed << std::setprecision(2) << rate << "%\n";
        }
    }

    // Export statistics as JSON
    std::string toJSON() const {
        std::ostringstream ss;
        CacheStats l1_agg = getAggregatedL1Stats();
        CacheStats l2_agg = getAggregatedL2Stats();
        const auto& l3_stats = l3_shared_->getStats();
        
        ss << "{\n";
        ss << "  \"architecture\": \"multi-core\",\n";
        ss << "  \"num_cores\": " << num_cores_ << ",\n";
        ss << "  \"total_accesses\": " << total_accesses_.load() << ",\n";
        ss << "  \"memory_accesses\": " << memory_accesses_.load() << ",\n";
        ss << "  \"L1\": {\n";
        ss << "    \"type\": \"private\",\n";
        ss << "    \"size_per_core\": " << l1_caches_[0]->getSizeBytes() << ",\n";
        ss << "    \"total_hits\": " << l1_agg.hits.load() << ",\n";
        ss << "    \"total_misses\": " << l1_agg.misses.load() << ",\n";
        ss << "    \"hit_rate\": " << std::fixed << std::setprecision(6) << l1_agg.hitRate() << "\n";
        ss << "  },\n";
        ss << "  \"L2\": {\n";
        ss << "    \"type\": \"private\",\n";
        ss << "    \"size_per_core\": " << l2_caches_[0]->getSizeBytes() << ",\n";
        ss << "    \"total_hits\": " << l2_agg.hits.load() << ",\n";
        ss << "    \"total_misses\": " << l2_agg.misses.load() << ",\n";
        ss << "    \"hit_rate\": " << std::fixed << std::setprecision(6) << l2_agg.hitRate() << "\n";
        ss << "  },\n";
        ss << "  \"L3\": {\n";
        ss << "    \"type\": \"shared\",\n";
        ss << "    \"size_total\": " << l3_shared_->getSizeBytes() << ",\n";
        ss << "    \"hits\": " << l3_stats.hits.load() << ",\n";
        ss << "    \"misses\": " << l3_stats.misses.load() << ",\n";
        ss << "    \"hit_rate\": " << std::fixed << std::setprecision(6) << l3_stats.hitRate() << "\n";
        ss << "  },\n";
        ss << "  \"per_core\": [\n";
        for (int i = 0; i < num_cores_; i++) {
            ss << "    {\"core\": " << i 
               << ", \"accesses\": " << core_accesses_[i]
               << ", \"memory_accesses\": " << core_memory_accesses_[i] << "}";
            if (i < num_cores_ - 1) ss << ",";
            ss << "\n";
        }
        ss << "  ]\n";
        ss << "}";
        return ss.str();
    }

    // Get stats as feature vector for perceptron
    std::vector<double> getFeatures() const {
        CacheStats l1_agg = getAggregatedL1Stats();
        CacheStats l2_agg = getAggregatedL2Stats();
        const auto& l3s = l3_shared_->getStats();
        
        double total = total_accesses_.load();
        double mem = memory_accesses_.load();
        
        return {
            l1_agg.hitRate(),
            l2_agg.hitRate(),
            l3s.hitRate(),
            total > 0 ? mem / total : 0.0,  // DRAM access rate
            l1_agg.totalAccesses() > 0 ? (double)l1_agg.evictions / l1_agg.totalAccesses() : 0.0,
            l2_agg.totalAccesses() > 0 ? (double)l2_agg.evictions / l2_agg.totalAccesses() : 0.0,
            l3s.totalAccesses() > 0 ? (double)l3s.evictions / l3s.totalAccesses() : 0.0,
        };
    }

private:
    static size_t getEnvSize(const char* name, size_t default_val) {
        const char* val = std::getenv(name);
        if (!val) return default_val;
        
        char* end;
        size_t result = std::strtoul(val, &end, 10);
        
        // Handle K, M, G suffixes
        if (*end == 'K' || *end == 'k') result *= 1024;
        else if (*end == 'M' || *end == 'm') result *= 1024 * 1024;
        else if (*end == 'G' || *end == 'g') result *= 1024 * 1024 * 1024;
        
        return result > 0 ? result : default_val;
    }

    int num_cores_;
    size_t line_size_;
    bool enabled_;
    
    std::vector<std::unique_ptr<CacheLevel>> l1_caches_;  // Private L1 per core
    std::vector<std::unique_ptr<CacheLevel>> l2_caches_;  // Private L2 per core
    std::unique_ptr<CacheLevel> l3_shared_;               // Shared L3
    
    std::atomic<uint64_t> total_accesses_{0};
    std::atomic<uint64_t> memory_accesses_{0};
    
    // Per-core statistics
    std::vector<uint64_t> core_accesses_;
    std::vector<uint64_t> core_memory_accesses_;
};

// ============================================================================
// SAMPLED Cache Hierarchy - Uses statistical sampling for ~5-20x speedup
// Samples every Nth access and extrapolates results
// ============================================================================
class SampledCacheHierarchy {
public:
    SampledCacheHierarchy(
        size_t sample_rate = 8,  // Sample 1 in N accesses
        size_t l1_size = 32 * 1024,
        size_t l1_ways = 8,
        size_t l2_size = 256 * 1024,
        size_t l2_ways = 8,
        size_t l3_size = 8 * 1024 * 1024,
        size_t l3_ways = 16,
        size_t line_size = 64
    ) : sample_rate_(sample_rate), line_size_(line_size), enabled_(true), counter_(0) {
        l1_ = std::make_unique<FastCacheLevel>("L1", l1_size, line_size, l1_ways);
        l2_ = std::make_unique<FastCacheLevel>("L2", l2_size, line_size, l2_ways);
        l3_ = std::make_unique<FastCacheLevel>("L3", l3_size, line_size, l3_ways);
        
        // Use prime-based sampling to avoid patterns
        sample_mask_ = sample_rate_ - 1;  // Works best when sample_rate is power of 2
    }

    static SampledCacheHierarchy fromEnvironment() {
        size_t sample_rate = getEnvSize("CACHE_SAMPLE_RATE", 8);
        // Round to power of 2
        size_t sr = 1;
        while (sr < sample_rate) sr <<= 1;
        sample_rate = sr;
        
        size_t l1_size = getEnvSize("CACHE_L1_SIZE", 32 * 1024);
        size_t l1_ways = getEnvSize("CACHE_L1_WAYS", 8);
        size_t l2_size = getEnvSize("CACHE_L2_SIZE", 256 * 1024);
        size_t l2_ways = getEnvSize("CACHE_L2_WAYS", 8);
        size_t l3_size = getEnvSize("CACHE_L3_SIZE", 8 * 1024 * 1024);
        size_t l3_ways = getEnvSize("CACHE_L3_WAYS", 16);
        size_t line_size = getEnvSize("CACHE_LINE_SIZE", 64);
        
        return SampledCacheHierarchy(sample_rate, l1_size, l1_ways, l2_size, l2_ways,
                                     l3_size, l3_ways, line_size);
    }

    // Sampled access - only simulates every Nth access
    __attribute__((always_inline))
    inline void access(uint64_t address, bool is_write = false) {
        if (!enabled_) return;
        
        total_accesses_++;
        
        // Fast modulo for power-of-2 sample rate
        if ((counter_++ & sample_mask_) != 0) return;
        
        // This access is sampled - simulate it
        address = address & ~(line_size_ - 1);
        
        if (l1_->access(address)) return;
        if (l2_->access(address)) { l1_->insert(address); return; }
        if (l3_->access(address)) { l2_->insert(address); l1_->insert(address); return; }
        
        sampled_memory_accesses_++;
        l3_->insert(address);
        l2_->insert(address);
        l1_->insert(address);
    }

    template<typename T>
    inline void read(const T* ptr) { access(reinterpret_cast<uint64_t>(ptr), false); }

    template<typename T>
    inline void write(T* ptr) { access(reinterpret_cast<uint64_t>(ptr), true); }

    template<typename T>
    inline void readArray(const T* arr, size_t index) {
        access(reinterpret_cast<uint64_t>(&arr[index]), false);
    }

    template<typename T>
    inline void writeArray(T* arr, size_t index) {
        access(reinterpret_cast<uint64_t>(&arr[index]), true);
    }

    void resetStats() {
        l1_->resetStats();
        l2_->resetStats();
        l3_->resetStats();
        total_accesses_ = 0;
        sampled_memory_accesses_ = 0;
        counter_ = 0;
    }

    void enable() { enabled_ = true; }
    void disable() { enabled_ = false; }
    bool isEnabled() const { return enabled_; }

    // No-op: P-OPT/GRASP require CacheLevel-based hierarchy
    void setCurrentVertex(uint32_t) {}
    void initGraphContext(const GraphCacheContext*) {}
    
    uint64_t getTotalAccesses() const { return total_accesses_; }
    uint64_t getSampleRate() const { return sample_rate_; }
    
    // Extrapolated memory accesses
    uint64_t getMemoryAccesses() const { 
        return sampled_memory_accesses_ * sample_rate_; 
    }

    void printStats(std::ostream& os = std::cout) const {
        auto& l1 = l1_->getStats();
        auto& l2 = l2_->getStats();
        auto& l3 = l3_->getStats();
        
        os << "\n";
        os << "╔══════════════════════════════════════════════════════════════════╗\n";
        os << "║          SAMPLED CACHE SIMULATION RESULTS (1:" << sample_rate_ << " sampling)        ║\n";
        os << "╠══════════════════════════════════════════════════════════════════╣\n";
        
        printSampledLevelStats(os, "L1", l1);
        printSampledLevelStats(os, "L2", l2);
        printSampledLevelStats(os, "L3", l3);
        
        os << "╠══════════════════════════════════════════════════════════════════╣\n";
        os << "║ SUMMARY (Extrapolated from " << std::setw(3) << (100.0/sample_rate_) << "% sample)                       ║\n";
        os << "╠══════════════════════════════════════════════════════════════════╣\n";
        os << "║ Total Accesses:      " << std::setw(15) << total_accesses_ 
           << "                          ║\n";
        os << "║ Sampled Accesses:    " << std::setw(15) << (total_accesses_ / sample_rate_)
           << "                          ║\n";
        uint64_t mem = getMemoryAccesses();
        os << "║ Memory Accesses:     " << std::setw(15) << mem
           << " (extrapolated)           ║\n";
        double overall = total_accesses_ > 0 ? 
            (1.0 - (double)mem / total_accesses_) : 0.0;
        os << "║ Overall Hit Rate:    " << std::setw(14) << std::fixed 
           << std::setprecision(4) << (overall * 100) << "%"
           << "                          ║\n";
        os << "╚══════════════════════════════════════════════════════════════════╝\n";
    }

    std::string toJSON() const {
        auto& l1 = l1_->getStats();
        auto& l2 = l2_->getStats();
        auto& l3 = l3_->getStats();
        
        std::ostringstream ss;
        ss << "{\n";
        ss << "  \"mode\": \"sampled\",\n";
        ss << "  \"sample_rate\": " << sample_rate_ << ",\n";
        ss << "  \"total_accesses\": " << total_accesses_ << ",\n";
        ss << "  \"sampled_accesses\": " << (total_accesses_ / sample_rate_) << ",\n";
        ss << "  \"memory_accesses\": " << getMemoryAccesses() << ",\n";
        ss << "  \"L1\": { \"hits\": " << (l1.hits.load() * sample_rate_) 
           << ", \"misses\": " << (l1.misses.load() * sample_rate_)
           << ", \"hit_rate\": " << std::fixed << std::setprecision(6) << l1.hitRate() << " },\n";
        ss << "  \"L2\": { \"hits\": " << (l2.hits.load() * sample_rate_)
           << ", \"misses\": " << (l2.misses.load() * sample_rate_)
           << ", \"hit_rate\": " << std::fixed << std::setprecision(6) << l2.hitRate() << " },\n";
        ss << "  \"L3\": { \"hits\": " << (l3.hits.load() * sample_rate_)
           << ", \"misses\": " << (l3.misses.load() * sample_rate_)
           << ", \"hit_rate\": " << std::fixed << std::setprecision(6) << l3.hitRate() << " }\n";
        ss << "}";
        return ss.str();
    }

private:
    static size_t getEnvSize(const char* name, size_t default_val) {
        const char* val = std::getenv(name);
        if (!val) return default_val;
        char* end;
        size_t result = std::strtoul(val, &end, 10);
        if (*end == 'K' || *end == 'k') result *= 1024;
        else if (*end == 'M' || *end == 'm') result *= 1024 * 1024;
        else if (*end == 'G' || *end == 'g') result *= 1024 * 1024 * 1024;
        return result > 0 ? result : default_val;
    }

    void printSampledLevelStats(std::ostream& os, const std::string& name, 
                                 const CacheStats& stats) const {
        os << "╠──────────────────────────────────────────────────────────────────╣\n";
        os << "║ " << name << " Cache (sampled)                                           ║\n";
        os << "║   Hits:                " << std::setw(15) << (stats.hits.load() * sample_rate_)
           << " (extrapolated)           ║\n";
        os << "║   Misses:              " << std::setw(15) << (stats.misses.load() * sample_rate_)
           << " (extrapolated)           ║\n";
        os << "║   Hit Rate:            " << std::setw(14) << std::fixed 
           << std::setprecision(4) << (stats.hitRate() * 100) << "%"
           << "                          ║\n";
    }

    std::unique_ptr<FastCacheLevel> l1_;
    std::unique_ptr<FastCacheLevel> l2_;
    std::unique_ptr<FastCacheLevel> l3_;
    size_t sample_rate_;
    size_t sample_mask_;
    size_t line_size_;
    bool enabled_;
    uint64_t counter_ = 0;
    uint64_t total_accesses_ = 0;
    uint64_t sampled_memory_accesses_ = 0;
};

// ============================================================================
// Global Cache Simulator Instances
// ============================================================================

// Single-core cache (original behavior - with locks, slower)
inline CacheHierarchy& GlobalCache() {
    static CacheHierarchy cache = CacheHierarchy::fromEnvironment();
    return cache;
}

// FAST single-core cache (no locks, ~10-20x faster)
inline FastCacheHierarchy& GlobalFastCache() {
    static FastCacheHierarchy cache = FastCacheHierarchy::fromEnvironment();
    return cache;
}

// ULTRA-FAST single-core cache (packed structures, ~2-3x faster than Fast)
inline UltraFastCacheHierarchy& GlobalUltraFastCache() {
    static UltraFastCacheHierarchy cache = UltraFastCacheHierarchy::fromEnvironment();
    return cache;
}

// Multi-core cache (for realistic architecture simulation)
inline MultiCoreCacheHierarchy& GlobalMultiCoreCache() {
    static MultiCoreCacheHierarchy cache = MultiCoreCacheHierarchy::fromEnvironment();
    return cache;
}

// SAMPLED cache (statistical sampling for ~5-20x speedup)
inline SampledCacheHierarchy& GlobalSampledCache() {
    static SampledCacheHierarchy cache = SampledCacheHierarchy::fromEnvironment();
    return cache;
}

// Check if multi-core mode is enabled via environment
inline bool IsMultiCoreMode() {
    static int mode = -1;
    if (mode < 0) {
        const char* val = std::getenv("CACHE_MULTICORE");
        mode = (val && (std::string(val) == "1" || std::string(val) == "true")) ? 1 : 0;
    }
    return mode == 1;
}

// Check if SAMPLED mode is enabled via environment
inline bool IsSampledMode() {
    static int mode = -1;
    if (mode < 0) {
        const char* val = std::getenv("CACHE_SAMPLED");
        mode = (val && (std::string(val) == "1" || std::string(val) == "true")) ? 1 : 0;
    }
    return mode == 1;
}

// Check if ULTRA-FAST mode is enabled via environment (default: true for best performance)
inline bool IsUltraFastMode() {
    static int mode = -1;
    if (mode < 0) {
        const char* val = std::getenv("CACHE_ULTRAFAST");
        // Default to ultrafast mode unless explicitly disabled
        mode = (val && (std::string(val) == "0" || std::string(val) == "false")) ? 0 : 1;
    }
    return mode == 1;
}

// Check if FAST mode is enabled via environment
inline bool IsFastMode() {
    static int mode = -1;
    if (mode < 0) {
        const char* val = std::getenv("CACHE_FAST");
        mode = (val && (std::string(val) == "1" || std::string(val) == "true")) ? 1 : 0;
    }
    return mode == 1;
}

// ============================================================================
// Convenience Macros for Instrumentation
// ============================================================================
#ifdef CACHE_SIM_ENABLED

// Auto-select single-core or multi-core based on CACHE_MULTICORE env var
#define CACHE_READ(ptr) do { \
    if (cache_sim::IsMultiCoreMode()) cache_sim::GlobalMultiCoreCache().read(ptr); \
    else cache_sim::GlobalCache().read(ptr); \
} while(0)

#define CACHE_WRITE(ptr) do { \
    if (cache_sim::IsMultiCoreMode()) cache_sim::GlobalMultiCoreCache().write(ptr); \
    else cache_sim::GlobalCache().write(ptr); \
} while(0)

#define CACHE_READ_ARRAY(arr, idx) do { \
    if (cache_sim::IsMultiCoreMode()) cache_sim::GlobalMultiCoreCache().readArray(arr, idx); \
    else cache_sim::GlobalCache().readArray(arr, idx); \
} while(0)

#define CACHE_WRITE_ARRAY(arr, idx) do { \
    if (cache_sim::IsMultiCoreMode()) cache_sim::GlobalMultiCoreCache().writeArray(arr, idx); \
    else cache_sim::GlobalCache().writeArray(arr, idx); \
} while(0)

#define CACHE_READ_RANGE(arr, start, end) cache_sim::GlobalCache().readRange(arr, start, end)

#define CACHE_ACCESS(addr, is_write) do { \
    if (cache_sim::IsMultiCoreMode()) cache_sim::GlobalMultiCoreCache().access(addr, is_write); \
    else cache_sim::GlobalCache().access(addr, is_write); \
} while(0)

#define CACHE_RESET() do { \
    cache_sim::GlobalCache().resetStats(); \
    cache_sim::GlobalMultiCoreCache().resetStats(); \
} while(0)

#define CACHE_PRINT() do { \
    if (cache_sim::IsMultiCoreMode()) cache_sim::GlobalMultiCoreCache().printStats(); \
    else cache_sim::GlobalCache().printStats(); \
} while(0)

#define CACHE_JSON() (cache_sim::IsMultiCoreMode() ? \
    cache_sim::GlobalMultiCoreCache().toJSON() : cache_sim::GlobalCache().toJSON())

#define CACHE_FEATURES() (cache_sim::IsMultiCoreMode() ? \
    cache_sim::GlobalMultiCoreCache().getFeatures() : cache_sim::GlobalCache().getFeatures())

#else

#define CACHE_READ(ptr) ((void)0)
#define CACHE_WRITE(ptr) ((void)0)
#define CACHE_READ_ARRAY(arr, idx) ((void)0)
#define CACHE_WRITE_ARRAY(arr, idx) ((void)0)
#define CACHE_READ_RANGE(arr, start, end) ((void)0)
#define CACHE_ACCESS(addr, is_write) ((void)0)
#define CACHE_RESET() ((void)0)
#define CACHE_PRINT() ((void)0)
#define CACHE_JSON() std::string("{}")
#define CACHE_FEATURES() std::vector<double>()

#endif

}  // namespace cache_sim

#endif  // CACHE_SIM_H_
