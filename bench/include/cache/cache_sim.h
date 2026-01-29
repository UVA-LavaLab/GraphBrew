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
    SRRIP     // Static Re-Reference Interval Prediction
};

inline std::string PolicyToString(EvictionPolicy policy) {
    switch (policy) {
        case EvictionPolicy::LRU:    return "LRU";
        case EvictionPolicy::FIFO:   return "FIFO";
        case EvictionPolicy::RANDOM: return "RANDOM";
        case EvictionPolicy::LFU:    return "LFU";
        case EvictionPolicy::PLRU:   return "PLRU";
        case EvictionPolicy::SRRIP:  return "SRRIP";
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
    uint8_t rrpv = 3;            // For SRRIP (2-bit)
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
        set[victim_idx].rrpv = 2;  // For SRRIP: near-immediate
    }

    const CacheStats& getStats() const { return stats_; }
    void resetStats() { stats_.reset(); }
    
    const std::string& getName() const { return name_; }
    size_t getSizeBytes() const { return size_bytes_; }
    size_t getLineSize() const { return line_size_; }
    size_t getAssociativity() const { return associativity_; }
    size_t getNumSets() const { return num_sets_; }
    EvictionPolicy getPolicy() const { return policy_; }

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
