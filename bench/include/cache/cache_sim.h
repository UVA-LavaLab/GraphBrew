// Copyright (c) 2024, UVA LavaLab
// Cache Simulator for Graph Algorithm Analysis
// Tracks L1/L2/L3 cache hits and misses with configurable parameters

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
// Single Cache Level
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
// Global Cache Simulator Instance
// ============================================================================
inline CacheHierarchy& GlobalCache() {
    static CacheHierarchy cache = CacheHierarchy::fromEnvironment();
    return cache;
}

// ============================================================================
// Convenience Macros for Instrumentation
// ============================================================================
#ifdef CACHE_SIM_ENABLED

#define CACHE_READ(ptr) cache_sim::GlobalCache().read(ptr)
#define CACHE_WRITE(ptr) cache_sim::GlobalCache().write(ptr)
#define CACHE_READ_ARRAY(arr, idx) cache_sim::GlobalCache().readArray(arr, idx)
#define CACHE_WRITE_ARRAY(arr, idx) cache_sim::GlobalCache().writeArray(arr, idx)
#define CACHE_READ_RANGE(arr, start, end) cache_sim::GlobalCache().readRange(arr, start, end)
#define CACHE_ACCESS(addr, is_write) cache_sim::GlobalCache().access(addr, is_write)
#define CACHE_RESET() cache_sim::GlobalCache().resetStats()
#define CACHE_PRINT() cache_sim::GlobalCache().printStats()
#define CACHE_JSON() cache_sim::GlobalCache().toJSON()
#define CACHE_FEATURES() cache_sim::GlobalCache().getFeatures()

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
