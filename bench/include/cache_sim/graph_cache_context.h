// ============================================================================
// GraphCacheContext: Unified graph-aware cache metadata for all policies
// ============================================================================
//
// Single structure carrying ALL graph metadata needed by graph-aware cache
// replacement policies (GRASP, P-OPT, MASK/ECG, GRASP-OPT, GRASP-XP).
//
// Design goals:
//   1. Unified — one struct serves all current and future graph-aware policies
//   2. Simulator-ready — flat/serializable layout for Sniper/gem5 pass-through
//   3. Multi-region — tracks multiple vertex property arrays (scores, contrib)
//   4. Per-access context — carries source/dest vertex + mask hint per access
//   5. Self-tuning — hot fractions computed from degree distribution, not manual
//
// Simulator integration (Sniper/gem5):
//   - Flat fields (topology, hints, reref config) map to memory-mapped registers
//   - PropertyRegion array is fixed-size (MAX_PROPERTY_REGIONS = 8)
//   - Rereference matrix is a contiguous uint8_t buffer (DMA-able)
//   - setCurrentVertices() maps to a magic instruction writing registers
//
// References:
//   - GRASP: Faldu et al., HPCA 2020 (DBG + 3-tier RRIP)
//   - P-OPT: Balaji et al., HPCA 2021 (transpose-based Belady approximation)
//   - ECG:   Mughrabi et al., GrAPL (MASK encoding + multi-region + prefetch)
//
// Author: GraphBrew Team
// ============================================================================

#ifndef GRAPH_CACHE_CONTEXT_H_
#define GRAPH_CACHE_CONTEXT_H_

#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace cache_sim {

// ============================================================================
// PropertyRegion: One tracked vertex data array
// ============================================================================
// Each graph algorithm accesses multiple vertex property arrays (e.g., scores,
// outgoing_contrib, comp). GRASP/ECG need to know which array an address
// belongs to, and where the hot/warm/cold boundaries are within that array.
//
// Memory layout (requires DBG-reordered graph):
//   [base_address ... hot_bound)       = HOT       (high-degree hubs, RRPV = 1)
//   [hot_bound ... warm_bound)          = WARM      (moderate-degree, RRPV = 3)
//   [warm_bound ... lukewarm_bound)     = LUKEWARM  (low-moderate, RRPV = 5)
//   [lukewarm_bound ... upper_bound)    = COLD      (low-degree, RRPV = 7)
//
// The 4-tier classification matches ECG's MASK encoding:
//   11=HOT, 10=WARM, 01=LUKEWARM, 00=COLD
//
// Region boundaries can be:
//   a) Auto-computed from degree distribution + cache geometry (preferred)
//   b) Manually specified via manual_hot_fraction
struct PropertyRegion {
    uint64_t base_address = 0;     // Start address of the property array
    uint64_t upper_bound = 0;      // End address
    uint64_t hot_bound = 0;        // [base, hot_bound) = HOT
    uint64_t warm_bound = 0;       // [hot_bound, warm_bound) = WARM
    uint64_t lukewarm_bound = 0;   // [warm_bound, lukewarm_bound) = LUKEWARM
    uint32_t num_elements = 0;     // Number of elements
    uint32_t elem_size = 0;        // Size of each element in bytes
    uint32_t region_id = 0;        // ID for per-region statistics
    uint32_t _pad = 0;             // Padding for 8-byte alignment

    // Classify an address within this region
    // Returns: 0=not in region, 1=HOT, 2=WARM, 3=LUKEWARM, 4=COLD
    uint32_t classify(uint64_t addr) const {
        if (addr < base_address || addr >= upper_bound) return 0;
        if (addr < hot_bound)      return 1;  // HOT
        if (addr < warm_bound)     return 2;  // WARM
        if (addr < lukewarm_bound) return 3;  // LUKEWARM
        return 4;                             // COLD
    }
};

// Maximum property regions (compile-time for simulator flat layout)
// ECG uses 5 (irregData, regData, CSR-offsets, CSR-coords, frontier)
// GraphBrew needs at most 4 (scores, contrib, comp, dist)
static constexpr uint32_t MAX_PROPERTY_REGIONS = 8;

// ============================================================================
// RereferenceConfig: P-OPT compressed transpose matrix
// ============================================================================
// Stored as flat uint8_t array of size [num_epochs × num_cache_lines].
// Layout: matrix[epoch * num_cache_lines + cache_line_id]
//
// Entry encoding (8-bit, from makeOffsetMatrix in popt.h):
//   MSB=1: cache line IS referenced in this epoch
//     bits [6:0] = sub-epoch of LAST access within epoch
//   MSB=0: cache line is NOT referenced in this epoch
//     bits [6:0] = distance (in epochs) to next epoch with a reference
struct RereferenceConfig {
    const uint8_t* matrix = nullptr;  // Rereference matrix data (not owned)
    uint32_t num_cache_lines = 0;     // Cache lines covering vertex data
    uint32_t num_epochs = 256;        // Number of epochs (typically 256)
    uint32_t epoch_size = 0;          // Vertices per epoch
    uint32_t sub_epoch_size = 0;      // Vertices per sub-epoch (epoch_size / 128)
    uint32_t line_size = 64;          // Cache line size in bytes
    uint32_t _pad = 0;

    // Algorithm 2: compute next-reference distance for a cache line
    uint32_t findNextRef(uint32_t cline_id, uint32_t current_vertex) const {
        if (matrix == nullptr || cline_id >= num_cache_lines) return 127;
        uint32_t epoch_id = current_vertex / epoch_size;
        if (epoch_id >= num_epochs) return 127;

        uint8_t entry = matrix[epoch_id * num_cache_lines + cline_id];
        constexpr uint8_t MSB = 0x80;
        constexpr uint8_t MASK = 0x7F;

        if ((entry & MSB) != 0) {
            // Referenced in this epoch — check sub-epoch position
            uint8_t last_sub = entry & MASK;
            uint32_t curr_sub = (current_vertex % epoch_size) / sub_epoch_size;
            if (curr_sub <= last_sub) return 0;  // Still upcoming
            // Past final access — check next epoch
            if (epoch_id + 1 < num_epochs) {
                uint8_t next = matrix[(epoch_id + 1) * num_cache_lines + cline_id];
                if ((next & MSB) != 0) return 1;  // Referenced next epoch
                uint8_t dist = next & MASK;
                return (dist < 127) ? dist + 1 : 127;
            }
            return 127;
        } else {
            // NOT referenced — data encodes distance to next epoch
            return entry & MASK;
        }
    }
};

// ============================================================================
// GraphTopology: Degree distribution summary
// ============================================================================
// Used by GRASP-XP for degree-bucket-proportional RRPV,
// and by auto-computation of hot fractions.
struct GraphTopology {
    uint32_t num_vertices = 0;
    uint64_t num_edges = 0;
    uint32_t avg_degree = 0;
    uint32_t max_degree = 0;
    bool     directed = false;

    // ECG-style logarithmic degree buckets
    static constexpr uint32_t NUM_BUCKETS = 11;
    uint32_t bucket_thresholds[NUM_BUCKETS] = {};
    uint64_t bucket_counts[NUM_BUCKETS] = {};
    uint64_t bucket_total_degrees[NUM_BUCKETS] = {};

    void computeFromDegrees(const uint32_t* degrees, uint32_t n,
                            uint64_t m, bool dir) {
        num_vertices = n;
        num_edges = m;
        directed = dir;
        avg_degree = (n > 0) ? static_cast<uint32_t>(m / n) : 0;
        max_degree = 0;

        // Initialize thresholds (ECG-style logarithmic buckets)
        bucket_thresholds[0] = (avg_degree > 1) ? avg_degree / 2 : 1;
        for (uint32_t i = 1; i < NUM_BUCKETS - 1; ++i)
            bucket_thresholds[i] = bucket_thresholds[i - 1] * 2;
        bucket_thresholds[NUM_BUCKETS - 1] = UINT32_MAX;

        std::memset(bucket_counts, 0, sizeof(bucket_counts));
        std::memset(bucket_total_degrees, 0, sizeof(bucket_total_degrees));

        for (uint32_t v = 0; v < n; ++v) {
            uint32_t d = degrees[v];
            if (d > max_degree) max_degree = d;
            for (uint32_t b = 0; b < NUM_BUCKETS; ++b) {
                if (d <= bucket_thresholds[b]) {
                    bucket_counts[b]++;
                    bucket_total_degrees[b] += d;
                    break;
                }
            }
        }
    }
};

// ============================================================================
// AccessHints: Per-access context (maps to simulator registers)
// ============================================================================
// Updated by simulation code on each outer-loop iteration and inner-loop access.
// In Sniper/gem5, these would be written via magic instructions.
struct AccessHints {
    uint32_t current_src = UINT32_MAX;  // Outer-loop vertex (regInd in P-OPT)
    uint32_t current_dst = UINT32_MAX;  // Inner-loop neighbor (irregInd in P-OPT)
    uint8_t  mask = 0;                  // ECG MASK hint (width determined by mask_bits)
    uint8_t  mask_bits = 2;             // Number of mask bits (2=ECG default, 4/8 for finer control)
    uint8_t  _pad[2] = {};

    // ECG mask encoding constants (2-bit, from ECG -M flag graphConfig.h)
    static constexpr uint8_t MASK_HOT      = 0x03;  // 11
    static constexpr uint8_t MASK_WARM     = 0x02;  // 10
    static constexpr uint8_t MASK_LUKEWARM = 0x01;  // 01
    static constexpr uint8_t MASK_COLD     = 0x00;  // 00

    // Compute mask value from tier classification (0-4) using current mask_bits
    // Maps tier 1=HOT → highest mask, tier 4=COLD → 0
    uint8_t tierToMask(uint32_t tier) const {
        if (tier == 0) return MASK_COLD;  // Unknown = cold
        uint8_t max_val = (1 << mask_bits) - 1;
        // Linearly map: tier 1 → max_val, tier 4 → 0
        if (tier >= 4) return 0;
        return static_cast<uint8_t>(max_val * (4 - tier) / 3);
    }
};

// ============================================================================
// VertexStats: Optional per-vertex cache statistics
// ============================================================================
// Enabled via enableVertexStats(). Tracks per-vertex hit/miss/reuse data
// for analysis of which vertices cause the most cache pressure.
struct VertexStats {
    std::vector<uint64_t> misses;
    std::vector<uint64_t> hits;
    std::vector<uint64_t> accesses;
    std::vector<uint64_t> reuse_distance;
    std::vector<uint64_t> last_access_time;
    bool enabled = false;

    void init(uint32_t n) {
        misses.assign(n, 0);
        hits.assign(n, 0);
        accesses.assign(n, 0);
        reuse_distance.assign(n, 0);
        last_access_time.assign(n, 0);
        enabled = true;
    }

    void recordAccess(uint32_t vertex_id, bool is_hit, uint64_t global_time) {
        if (!enabled || vertex_id >= accesses.size()) return;
        accesses[vertex_id]++;
        if (is_hit) hits[vertex_id]++;
        else misses[vertex_id]++;
        if (last_access_time[vertex_id] > 0)
            reuse_distance[vertex_id] += (global_time - last_access_time[vertex_id]);
        last_access_time[vertex_id] = global_time;
    }
};

// ============================================================================
// GraphCacheContext: The unified structure
// ============================================================================
//
// Usage in sim benchmarks:
//
//   GraphCacheContext ctx;
//   ctx.initTopology(degrees, g.num_nodes(), g.num_edges(), g.directed());
//   ctx.registerPropertyArray(scores.data(), g.num_nodes(), sizeof(float),
//                             cache.L3()->getSizeBytes());
//   ctx.registerPropertyArray(contrib.data(), g.num_nodes(), sizeof(float),
//                             cache.L3()->getSizeBytes());
//   ctx.initRereference(reref_data, num_clines, 256, g.num_nodes(), 64);
//   cache.initGraphContext(&ctx);
//
//   // In kernel loop:
//   for (NodeID u = 0; u < g.num_nodes(); u++) {
//       ctx.setCurrentVertices(u, u);
//       for (NodeID v : g.in_neigh(u)) {
//           ctx.setCurrentDst(v);
//           SIM_CACHE_READ(cache, contrib_ptr, v);
//       }
//   }
//
struct GraphCacheContext {
    // --- Property Regions (multiple tracked arrays) ---
    PropertyRegion regions[MAX_PROPERTY_REGIONS];
    uint32_t num_regions = 0;

    // --- Graph Topology (degree distribution) ---
    GraphTopology topology;

    // --- Per-Access Hints (mutable, updated each vertex iteration) ---
    AccessHints hints;

    // --- Rereference Matrix (P-OPT) ---
    RereferenceConfig rereference;

    // --- Prefetch Matrix (ECG-style graph-aware prefetching) ---
    // Maps each vertex to its predicted next-access vertex.
    // When a vertex v is accessed, prefetch data for prefetch_map[v].
    // Built from graph transpose (similar to P-OPT but for prefetching).
    // nullptr = prefetching disabled.
    const uint32_t* prefetch_map = nullptr;
    uint32_t prefetch_num_vertices = 0;

    // --- Per-Vertex Statistics (optional) ---
    VertexStats vertex_stats;

    // ================================================================
    // Registration Functions
    // ================================================================

    // Register graph topology from degree array.
    // Needed by: GRASP-XP (degree-bucket RRPV), auto hot-fraction computation.
    void initTopology(const uint32_t* degrees, uint32_t num_vertices,
                      uint64_t num_edges, bool directed) {
        topology.computeFromDegrees(degrees, num_vertices, num_edges, directed);
    }

    // Register a property array with auto-computed hot/warm boundaries.
    //
    // Hot fraction is auto-computed from degree distribution if topology is
    // initialized (preferred), otherwise falls back to LLC capacity ratio.
    //
    // Auto-computation logic:
    //   1. How many vertex elements fit in this array's share of the LLC?
    //      vtx_per_llc = (llc_size / num_regions) / elem_size
    //   2. If topology available: what fraction of vertices covers 50% of edges?
    //      (power-law: 1% of vertices may hold 50% of edges)
    //   3. hot_fraction = min(vtx_per_llc / num_vertices, degree_cutoff)
    //
    // This makes hot/warm boundaries self-tuning based on graph properties
    // instead of requiring manual parameter tuning (GRASP's key weakness).
    void registerPropertyArray(const void* data_ptr, uint32_t num_elements,
                               uint32_t elem_size, size_t llc_size,
                               double manual_hot_fraction = -1.0) {
        if (num_regions >= MAX_PROPERTY_REGIONS) return;

        PropertyRegion& r = regions[num_regions];
        r.base_address = reinterpret_cast<uint64_t>(data_ptr);
        r.upper_bound = r.base_address + static_cast<uint64_t>(num_elements) * elem_size;
        r.num_elements = num_elements;
        r.elem_size = elem_size;
        r.region_id = num_regions;
        r._pad = 0;

        // Compute hot fraction
        double hot_fraction;
        if (manual_hot_fraction > 0.0 && manual_hot_fraction <= 1.0) {
            // Manual override
            hot_fraction = manual_hot_fraction;
        } else {
            // Auto-compute from LLC capacity and degree distribution
            hot_fraction = autoComputeHotFraction(num_elements, elem_size, llc_size);
        }

        // Compute bounds aligned to 64-byte cache line boundary
        // 4-tier geometric sizing (matching ECG reorder.c):
        //   HOT:      hot_fraction of array
        //   WARM:     4× HOT region (ECG: cache_regions[1] = cache_regions[0] * 4)
        //   LUKEWARM: 4× WARM region (ECG: cache_regions[2] = cache_regions[1] * 4)
        //   COLD:     everything else
        uint64_t array_bytes = static_cast<uint64_t>(num_elements) * elem_size;
        uint64_t hot_bytes = static_cast<uint64_t>(hot_fraction * array_bytes);
        constexpr uint64_t LINE_MASK = ~uint64_t(63);

        // Check if array is small enough to split equally (ECG fallback)
        uint64_t cache_vtx_bytes = static_cast<uint64_t>((llc_size / (num_regions + 1)));
        bool small_graph = (array_bytes < 3 * cache_vtx_bytes);

        if (small_graph) {
            // Small graph: equal thirds (ECG: diff < 2*cache_size)
            uint64_t third = array_bytes / 3;
            r.hot_bound = (r.base_address + third + 63) & LINE_MASK;
            r.warm_bound = (r.base_address + 2 * third + 63) & LINE_MASK;
            r.lukewarm_bound = r.upper_bound;
        } else {
            // Large graph: geometric progression (ECG: ×4 per tier)
            r.hot_bound = (r.base_address + hot_bytes + 63) & LINE_MASK;
            uint64_t warm_bytes = hot_bytes * 4;
            r.warm_bound = (r.base_address + warm_bytes + 63) & LINE_MASK;
            uint64_t lukewarm_bytes = warm_bytes * 4;
            r.lukewarm_bound = (r.base_address + lukewarm_bytes + 63) & LINE_MASK;
        }

        // Clamp all bounds to array end
        if (r.hot_bound > r.upper_bound) r.hot_bound = r.upper_bound;
        if (r.warm_bound > r.upper_bound) r.warm_bound = r.upper_bound;
        if (r.lukewarm_bound > r.upper_bound) r.lukewarm_bound = r.upper_bound;

        num_regions++;
    }

    // Register rereference matrix for P-OPT.
    // Call after makeOffsetMatrix() from popt.h.
    void initRereference(const uint8_t* matrix, uint32_t num_cache_lines,
                         uint32_t num_epochs, uint32_t num_vertices,
                         uint32_t line_size) {
        rereference.matrix = matrix;
        rereference.num_cache_lines = num_cache_lines;
        rereference.num_epochs = num_epochs;
        rereference.epoch_size = (num_vertices + num_epochs - 1) / num_epochs;
        rereference.sub_epoch_size = (rereference.epoch_size + 127) / 128;
        rereference.line_size = line_size;
        rereference._pad = 0;
    }

    // Enable per-vertex statistics tracking.
    void enableVertexStats() {
        if (topology.num_vertices > 0) {
            vertex_stats.init(topology.num_vertices);
        }
    }

    // Register prefetch matrix for ECG-style graph-aware prefetching.
    // The map stores: prefetch_map[v] = vertex whose data to prefetch
    // when vertex v is accessed. Built from graph transpose.
    void initPrefetchMap(const uint32_t* map, uint32_t num_vertices) {
        prefetch_map = map;
        prefetch_num_vertices = num_vertices;
    }

    // Get prefetch target for a vertex (UINT32_MAX if none)
    uint32_t getPrefetchTarget(uint32_t vertex_id) const {
        if (!prefetch_map || vertex_id >= prefetch_num_vertices) return UINT32_MAX;
        return prefetch_map[vertex_id];
    }

    // ================================================================
    // Per-Access Updates (maps to simulator registers/magic instructions)
    // ================================================================

    // Set current source and destination vertices (outer loop).
    // In Sniper: SimMagic1(CMD_SET_VERTICES, pack(src, dst))
    void setCurrentVertices(uint32_t src, uint32_t dst) {
        hints.current_src = src;
        hints.current_dst = dst;
    }

    // Update destination vertex only (inner loop neighbor).
    void setCurrentDst(uint32_t dst) {
        hints.current_dst = dst;
    }

    // Set ECG mask hint for current access.
    // In Sniper: the mask is encoded in the last 2 bits of vertex property data.
    void setMask(uint8_t mask) {
        hints.mask = mask;
    }

    // ================================================================
    // Query Functions (used by cache replacement policies)
    // ================================================================

    // Classify an address across all registered property regions.
    // Returns: 0=unknown/streaming, 1=HOT, 2=WARM, 3=LUKEWARM, 4=COLD
    uint32_t classifyAddress(uint64_t addr) const {
        for (uint32_t i = 0; i < num_regions; ++i) {
            uint32_t tier = regions[i].classify(addr);
            if (tier != 0) return tier;
        }
        return 0;
    }

    // Check if address is in any registered property region.
    bool isPropertyData(uint64_t addr) const {
        return classifyAddress(addr) != 0;
    }

    // Find the property region containing an address.
    const PropertyRegion* findRegion(uint64_t addr) const {
        for (uint32_t i = 0; i < num_regions; ++i) {
            if (addr >= regions[i].base_address && addr < regions[i].upper_bound)
                return &regions[i];
        }
        return nullptr;
    }

    // Compute P-OPT rereference distance for a cache line address.
    uint32_t findNextRef(uint64_t line_addr) const {
        if (rereference.matrix == nullptr) return 127;
        const PropertyRegion* r = findRegion(line_addr);
        if (r == nullptr) return 127;
        uint32_t cline_id = static_cast<uint32_t>(
            (line_addr - r->base_address) / rereference.line_size);
        return rereference.findNextRef(cline_id, hints.current_src);
    }

    // ================================================================
    // Diagnostics
    // ================================================================

    void printSummary(std::ostream& os = std::cout) const {
        os << "\n=== Graph Cache Context ===\n";
        os << "Vertices: " << topology.num_vertices
           << "  Edges: " << topology.num_edges
           << "  AvgDeg: " << topology.avg_degree
           << "  MaxDeg: " << topology.max_degree << "\n";
        os << "Property Regions: " << num_regions << "\n";
        for (uint32_t i = 0; i < num_regions; ++i) {
            const auto& r = regions[i];
            uint64_t total = r.upper_bound - r.base_address;
            uint64_t hot = r.hot_bound - r.base_address;
            uint64_t warm = r.warm_bound - r.base_address;
            uint64_t lukewarm = r.lukewarm_bound - r.base_address;
            os << "  [" << i << "] "
               << total << "B (elem=" << r.elem_size << "B × " << r.num_elements << ")"
               << "  hot=" << std::fixed << std::setprecision(1)
               << (total > 0 ? 100.0 * hot / total : 0) << "%"
               << "  warm=" << (total > 0 ? 100.0 * warm / total : 0) << "%"
               << "  lukewarm=" << (total > 0 ? 100.0 * lukewarm / total : 0) << "%\n";
        }
        if (rereference.matrix) {
            os << "Rereference: " << rereference.num_epochs << " epochs × "
               << rereference.num_cache_lines << " cache lines\n";
        }
        if (vertex_stats.enabled) {
            os << "Per-vertex stats: enabled (" << vertex_stats.accesses.size() << " vertices)\n";
        }
        os << std::defaultfloat << "===========================\n";
    }

private:
    // Auto-compute hot fraction from degree distribution and LLC capacity.
    //
    // Strategy:
    //   1. Capacity-based: what fraction of the array fits in this region's
    //      LLC share? (llc_size / num_regions_so_far / array_bytes)
    //   2. Degree-based (if topology available): what fraction of vertices
    //      covers 50% of total edges? For power-law graphs, this is << 1%.
    //   3. Take the minimum: the tighter constraint wins.
    //
    // This replaces GRASP's manual "f" parameter with a self-tuning approach.
    double autoComputeHotFraction(uint32_t num_elements, uint32_t elem_size,
                                  size_t llc_size) const {
        uint64_t array_bytes = static_cast<uint64_t>(num_elements) * elem_size;
        if (array_bytes == 0 || llc_size == 0) return 0.1;

        // 1. Capacity-based: what share of LLC can this array occupy?
        //    Divide LLC among registered regions (including this new one)
        uint32_t total_regions = num_regions + 1;  // +1 for the one being registered
        uint64_t llc_share = llc_size / total_regions;
        double capacity_fraction = static_cast<double>(llc_share) / array_bytes;
        if (capacity_fraction > 1.0) capacity_fraction = 1.0;

        // 2. Degree-based: what fraction covers 50% of edges?
        //    Only if topology is initialized
        double degree_fraction = 1.0;
        if (topology.num_vertices > 0 && topology.num_edges > 0) {
            uint64_t half_edges = topology.num_edges / 2;
            uint64_t cumulative = 0;
            // Scan from highest-degree bucket downward
            // (DBG-ordered: high-degree vertices are at front of array)
            for (int b = GraphTopology::NUM_BUCKETS - 1; b >= 0; --b) {
                cumulative += topology.bucket_total_degrees[b];
                if (cumulative >= half_edges) {
                    // Count vertices in this and higher buckets
                    uint64_t hot_vertices = 0;
                    for (int bb = b; bb < static_cast<int>(GraphTopology::NUM_BUCKETS); ++bb)
                        hot_vertices += topology.bucket_counts[bb];
                    degree_fraction = static_cast<double>(hot_vertices) / topology.num_vertices;
                    break;
                }
            }
            // Clamp: at least 1% (avoid empty hot region)
            if (degree_fraction < 0.01) degree_fraction = 0.01;
        }

        // Take the tighter constraint
        double result = std::min(capacity_fraction, degree_fraction);

        // Sanity: at least 1%, at most 50%
        if (result < 0.01) result = 0.01;
        if (result > 0.50) result = 0.50;

        return result;
    }
};

} // namespace cache_sim

#endif // GRAPH_CACHE_CONTEXT_H_
