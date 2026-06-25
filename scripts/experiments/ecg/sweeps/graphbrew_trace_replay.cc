#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "cache_sim/cache_sim.h"

namespace {

// Trace-replay-only reference implementations for upstream-parity validation.
// PIN and BELADY mirror upstream faldupriyank/grasp pin.cpp and belady.cpp
// semantics exactly. They are not part of the general cache_sim policy surface
// because PIN needs static graph-aware pinning and BELADY needs a full
// future-access pre-pass; both are only meaningful for offline trace replay.
struct ReplayBlock {
    uint64_t addr = 0;
    int64_t time = -1;  // signed: PIN uses LRU (min). Belady casts via uint64_t.
    bool pin = false;
};

struct BorderRegion {
    uint64_t min = 0;
    uint64_t border_high_reuse = 0;
};

bool in_high_reuse_region(uint64_t addr, const std::vector<BorderRegion>& regs) {
    for (const auto& r : regs) {
        if (addr >= r.min && addr < r.border_high_reuse) return true;
    }
    return false;
}

uint64_t read_u64(std::ifstream& in) {
    uint64_t value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) {
        std::cerr << "failed to read uint64_t from trace\n";
        std::exit(2);
    }
    return value;
}

std::string read_key(std::ifstream& in) {
    char key[25];
    in.read(key, sizeof(key));
    if (!in) {
        std::cerr << "failed to read trace header key\n";
        std::exit(2);
    }
    size_t len = 0;
    while (len < sizeof(key) && key[len] != '\0') ++len;
    return std::string(key, len);
}

uint64_t require(const std::unordered_map<std::string, uint64_t>& fields,
                 const std::string& key) {
    auto it = fields.find(key);
    if (it == fields.end()) return 0;
    return it->second;
}

cache_sim::EvictionPolicy parse_policy(const std::string& text) {
    if (text == "lru" || text == "LRU") return cache_sim::EvictionPolicy::LRU;
    if (text == "grasp" || text == "GRASP") return cache_sim::EvictionPolicy::GRASP;
    if (text == "pin" || text == "PIN") return cache_sim::EvictionPolicy::PIN;
    // BELADY is handled outside cache_sim (offline oracle); return LRU as a placeholder.
    if (text == "belady" || text == "BELADY") return cache_sim::EvictionPolicy::LRU;
    std::cerr << "unsupported policy for trace replay: " << text << "\n";
    std::exit(2);
}

std::vector<BorderRegion> build_border_regions(
    const std::unordered_map<std::string, uint64_t>& fields, size_t cache_size) {
    std::vector<BorderRegion> regions;
    for (const std::string prefix : {"propertyA", "propertyB"}) {
        auto fit = fields.find(prefix + "-f");
        if (fit == fields.end() || fit->second == 0) continue;
        auto minit = fields.find(prefix + "-0");
        auto maxit = fields.find(prefix + "-n");
        if (minit == fields.end() || maxit == fields.end()) continue;
        BorderRegion r;
        r.min = minit->second;
        uint64_t max = maxit->second;
        // Mirror upstream common.h add_border_boundry exactly.
        uint64_t f_bytes = static_cast<uint64_t>(
            static_cast<long long>(fit->second) *
            static_cast<long long>(cache_size) / 100);
        r.border_high_reuse = r.min + f_bytes;
        if (r.border_high_reuse > max) r.border_high_reuse = max;
        r.border_high_reuse += 8;  // upstream "size of a ptr" offset
        regions.push_back(r);
    }
    return regions;
}

// Upstream pin.cpp: LRU among unpinned ways, bypass if all pinned, set pin on
// insert when new line falls in the high-reuse region.
uint64_t replay_pin(const std::vector<uint64_t>& trace, size_t sets, size_t assoc,
                    const std::vector<BorderRegion>& regions) {
    std::vector<std::vector<ReplayBlock>> blocks(sets, std::vector<ReplayBlock>(assoc));
    uint64_t misses = 0;
    for (size_t i = 0; i < trace.size(); ++i) {
        uint64_t tr = trace[i];
        size_t set_idx = (tr >> 6) % sets;
        auto& set = blocks[set_idx];
        bool hit = false;
        size_t hit_way = 0;
        for (size_t w = 0; w < assoc; ++w) {
            if (set[w].addr == tr) { hit = true; hit_way = w; break; }
        }
        if (hit) {
            set[hit_way].time = static_cast<int64_t>(i);
            continue;
        }
        ++misses;
        bool in_high = in_high_reuse_region(tr, regions);
        int64_t min = LLONG_MAX;
        int min_index = -1;
        for (size_t w = 0; w < assoc; ++w) {
            if (!set[w].pin && set[w].time < min) {
                min = set[w].time;
                min_index = static_cast<int>(w);
            }
        }
        if (min_index >= 0) {
            set[min_index].addr = tr;
            set[min_index].time = static_cast<int64_t>(i);
            set[min_index].pin = in_high;
        }
        // else: all ways pinned, bypass (count miss but do not insert).
    }
    return misses;
}

// Upstream belady.cpp: backward pre-pass to compute next-use index per access,
// then forward pass that evicts the way with the largest stored next-use index.
uint64_t replay_belady(const std::vector<uint64_t>& trace, size_t sets, size_t assoc) {
    const size_t n = trace.size();
    std::vector<uint64_t> next_use(n);
    // Match upstream's sentinel: memset(timestamp, 1, ...) -> 0x0101010101010101.
    const uint64_t sentinel = 0x0101010101010101ULL;
    for (size_t i = 0; i < n; ++i) next_use[i] = sentinel;
    std::map<uint64_t, uint64_t> last_pos;
    for (int64_t i = static_cast<int64_t>(n) - 1; i >= 0; --i) {
        auto it = last_pos.find(trace[i]);
        if (it != last_pos.end()) {
            next_use[i] = it->second;
            it->second = static_cast<uint64_t>(i);
        } else {
            last_pos[trace[i]] = static_cast<uint64_t>(i);
        }
    }
    std::vector<std::vector<ReplayBlock>> blocks(sets, std::vector<ReplayBlock>(assoc));
    uint64_t misses = 0;
    for (size_t i = 0; i < n; ++i) {
        uint64_t tr = trace[i];
        size_t set_idx = (tr >> 6) % sets;
        auto& set = blocks[set_idx];
        bool hit = false;
        size_t hit_way = 0;
        for (size_t w = 0; w < assoc; ++w) {
            if (set[w].addr == tr) { hit = true; hit_way = w; break; }
        }
        if (hit) {
            set[hit_way].time = static_cast<int64_t>(next_use[i]);
            continue;
        }
        ++misses;
        // Upstream evicts the way with the largest stored time, comparing as
        // uint64_t so empty ways (time == -1 → UINT64_MAX) get filled first.
        uint64_t max_time = static_cast<uint64_t>(set[0].time);
        size_t max_way = 0;
        for (size_t w = 1; w < assoc; ++w) {
            uint64_t t = static_cast<uint64_t>(set[w].time);
            if (t > max_time) {
                max_time = t;
                max_way = w;
            }
        }
        set[max_way].addr = tr;
        set[max_way].time = static_cast<int64_t>(next_use[i]);
    }
    return misses;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: graphbrew_trace_replay TRACE CACHE_MB POLICY\n";
        return 2;
    }

    const std::string trace_path = argv[1];
    const size_t cache_mb = std::strtoull(argv[2], nullptr, 10);
    const std::string policy_text = argv[3];
    const size_t cache_size = cache_mb * 1024 * 1024;
    const size_t line_size = 64;
    const size_t associativity = 16;

    std::ifstream in(trace_path, std::ios::binary);
    if (!in) {
        std::cerr << "could not open trace: " << trace_path << "\n";
        return 2;
    }

    std::unordered_map<std::string, uint64_t> fields;
    const uint64_t num_regions = read_u64(in);
    for (uint64_t i = 0; i < num_regions; ++i) {
        std::string key = read_key(in);
        uint64_t value = read_u64(in);
        fields[key] = value;
    }
    const uint64_t total_addresses = read_u64(in);

    std::vector<uint64_t> trace(total_addresses);
    if (total_addresses > 0) {
        in.read(reinterpret_cast<char*>(trace.data()),
                static_cast<std::streamsize>(total_addresses * sizeof(uint64_t)));
        if (!in) {
            std::cerr << "could not read all trace addresses\n";
            return 2;
        }
    }

    const auto policy = parse_policy(policy_text);
    const std::string policy_lc = [&]() {
        std::string s = policy_text;
        for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    }();
    const size_t num_sets = (cache_size / line_size) / associativity;

    uint64_t aligned = 0;
    for (uint64_t address : trace) {
        if ((address & (line_size - 1)) == 0) ++aligned;
    }

    uint64_t misses = 0;
    uint64_t total = trace.size();
    uint64_t grasp_regions = 0;

    if (policy_lc == "pin") {
        // PIN is now a first-class cache_sim policy; treat it like GRASP for
        // region registration and exercise the real CacheLevel path.
        cache_sim::CacheLevel cache("L3", cache_size, line_size, associativity,
                                    cache_sim::EvictionPolicy::PIN);
        cache_sim::GraphCacheContext ctx;
        for (const std::string prefix : {"propertyA", "propertyB"}) {
            uint64_t base = require(fields, prefix + "-0");
            uint64_t upper = require(fields, prefix + "-n");
            uint32_t hot_percent = static_cast<uint32_t>(require(fields, prefix + "-f"));
            if (base != 0 && upper > base && hot_percent > 0) {
                ctx.registerGRASPTraceRegion(base, upper, hot_percent);
            }
        }
        cache.initGraphContext(&ctx);
        for (uint64_t address : trace) {
            bool hit = cache.access(address, false);
            if (!hit) cache.insert(address, false);
        }
        const auto& stats = cache.getStats();
        misses = stats.misses.load();
        total = stats.hits.load() + misses;
        grasp_regions = ctx.num_regions;
    } else if (policy_lc == "belady") {
        misses = replay_belady(trace, num_sets, associativity);
    } else {
        cache_sim::CacheLevel cache("L3", cache_size, line_size, associativity, policy);
        cache_sim::GraphCacheContext ctx;
        if (policy == cache_sim::EvictionPolicy::GRASP) {
            for (const std::string prefix : {"propertyA", "propertyB"}) {
                uint64_t base = require(fields, prefix + "-0");
                uint64_t upper = require(fields, prefix + "-n");
                uint32_t hot_percent = static_cast<uint32_t>(require(fields, prefix + "-f"));
                if (base != 0 && upper > base && hot_percent > 0) {
                    ctx.registerGRASPTraceRegion(base, upper, hot_percent);
                }
            }
            cache.initGraphContext(&ctx);
        }
        for (uint64_t address : trace) {
            bool hit = cache.access(address, false);
            if (!hit) cache.insert(address, false);
        }
        const auto& stats = cache.getStats();
        misses = stats.misses.load();
        total = stats.hits.load() + misses;
        grasp_regions = ctx.num_regions;
    }

    const double miss_rate = total > 0 ? static_cast<double>(misses) / static_cast<double>(total) : 0.0;

    std::cout << "policy," << policy_text << "\n";
    std::cout << "cache_mb," << cache_mb << "\n";
    std::cout << "total-accesses," << total << "\n";
    std::cout << "total-misses," << misses << "\n";
    std::cout << "miss-rate," << miss_rate << "\n";
    std::cout << "aligned-addresses," << aligned << "\n";
    std::cout << "grasp-regions," << grasp_regions << "\n";
    return 0;
}