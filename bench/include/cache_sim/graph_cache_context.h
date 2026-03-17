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
#include <unordered_set>
#include <iostream>
#include <iomanip>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace cache_sim {

// ============================================================================
// PropertyRegion: One tracked vertex data array
// ============================================================================
// Each graph algorithm accesses multiple vertex property arrays (e.g., scores,
// outgoing_contrib, comp). GRASP/ECG need to know which array an address
// belongs to, and which degree bucket a vertex falls in.
//
// After DBG reordering, vertices are sorted by descending degree. The array
// is partitioned into N degree buckets (default 11, matching DBG). Each
// bucket maps to a proportional RRPV based on its share of total edges.
//
// The number of buckets is flexible:
//   - GRASP paper uses 3 regions (simplified from the RRPV space)
//   - ECG GRASP-XP uses all 11 DBG buckets with 8-bit RRPV
//   - Our implementation supports 1-16 buckets (configurable)
//
// bucket_bounds[i] = byte offset where bucket i ends in the array
// bucket_bounds[0] = end of highest-degree bucket (smallest, at front)
// bucket_bounds[N-1] = end of lowest-degree bucket (= upper_bound)
//
// Requires DBG-reordered graph for meaningful classification.
static constexpr uint32_t MAX_REGION_BUCKETS = 16;

struct PropertyRegion {
    uint64_t base_address = 0;      // Start address of the property array
    uint64_t upper_bound = 0;       // End address
    uint32_t num_elements = 0;      // Number of elements
    uint32_t elem_size = 0;         // Size of each element in bytes
    uint32_t region_id = 0;         // ID for per-region statistics
    uint32_t num_buckets = 0;       // Active bucket count (0 = uninitialized)

    // Bucket boundaries: bucket_bounds[i] = upper byte address of bucket i
    // Bucket 0 = highest-degree (most important to cache)
    // Bucket num_buckets-1 = lowest-degree (least important)
    uint64_t bucket_bounds[MAX_REGION_BUCKETS] = {};

    // Classify an address into a bucket index.
    // Returns: 0..num_buckets-1 for valid buckets, num_buckets for outside region
    uint32_t classifyBucket(uint64_t addr) const {
        if (addr < base_address || addr >= upper_bound || num_buckets == 0)
            return num_buckets;  // Outside region
        for (uint32_t b = 0; b < num_buckets; ++b) {
            if (addr < bucket_bounds[b]) return b;
        }
        return num_buckets - 1;  // Last bucket
    }

    // Legacy 4-tier classification (for backward compatibility)
    // Maps N buckets to 4 tiers: first quarter = HOT(1), etc.
    uint32_t classify(uint64_t addr) const {
        uint32_t b = classifyBucket(addr);
        if (b >= num_buckets) return 0;  // Outside
        // Map bucket to tier: 0..N/4=HOT(1), N/4..N/2=WARM(2), etc.
        uint32_t quarter = (num_buckets > 0) ? (b * 4 / num_buckets) : 3;
        return quarter + 1;  // 1=HOT, 2=WARM, 3=LUKEWARM, 4=COLD
    }
};

// Maximum property regions (compile-time for simulator flat layout)
// ECG uses 5 (irregData, regData, CSR-offsets, CSR-coords, frontier)
// GraphBrew needs at most 4 (scores, contrib, comp, dist)
static constexpr uint32_t MAX_PROPERTY_REGIONS = 8;

// ============================================================================
// ECGMode: Controls how the ECG policy layers its eviction signals
// ============================================================================
// ECG uses a 3-level layered eviction strategy. The mode determines the
// priority order of the tiebreakers after SRRIP aging (Level 1).
//
// All modes use SRRIP aging as Level 1 (find max RRPV, age until found).
// The mode controls Level 2 and Level 3 tiebreakers:
//
//   DBG_PRIMARY  (default): SRRIP → DBG tier → dynamic P-OPT
//   POPT_PRIMARY:           SRRIP → dynamic P-OPT → DBG tier
//   DBG_ONLY:               SRRIP → DBG tier (no P-OPT, fast path)
//   ECG_EMBEDDED:           Stored P-OPT hint as primary, DBG as secondary
//   ECG_COMBINED:           Both DBG + P-OPT hint → unified insertion RRPV (Hawkeye-inspired)
enum class ECGMode {
    DBG_PRIMARY,   // DBG tier is primary tiebreaker, P-OPT is secondary
    POPT_PRIMARY,  // Dynamic P-OPT is primary tiebreaker, DBG is secondary
    DBG_ONLY,      // DBG tier only, no P-OPT consulted (equivalent to GRASP+mask)
    ECG_EMBEDDED,  // DBG tier primary, stored P-OPT hint (from mask) as secondary — zero LLC overhead
    ECG_COMBINED   // Combined DBG+P-OPT → insertion RRPV (both signals at insert, not evict)
};

inline std::string ECGModeToString(ECGMode mode) {
    switch (mode) {
        case ECGMode::DBG_PRIMARY:  return "DBG_PRIMARY";
        case ECGMode::POPT_PRIMARY: return "POPT_PRIMARY";
        case ECGMode::DBG_ONLY:     return "DBG_ONLY";
        case ECGMode::ECG_EMBEDDED: return "ECG_EMBEDDED";
        case ECGMode::ECG_COMBINED: return "ECG_COMBINED";
        default:                    return "UNKNOWN";
    }
}

inline ECGMode StringToECGMode(const std::string& s) {
    if (s == "POPT_PRIMARY" || s == "popt_primary" || s == "popt") return ECGMode::POPT_PRIMARY;
    if (s == "DBG_ONLY" || s == "dbg_only" || s == "dbg") return ECGMode::DBG_ONLY;
    if (s == "ECG_EMBEDDED" || s == "ecg_embedded" || s == "embedded") return ECGMode::ECG_EMBEDDED;
    if (s == "ECG_COMBINED" || s == "ecg_combined" || s == "combined") return ECGMode::ECG_COMBINED;
    return ECGMode::DBG_PRIMARY;  // Default
}

// ============================================================================
// MaskConfig: Configuration for per-edge mask array
// ============================================================================
// Controls how per-edge cache hints are encoded, how DBG and P-OPT signals
// are combined, and how prefetch targets are resolved.
//
// The mask array is a parallel array alongside the CSR edge list: same size,
// same indexing. Each entry packs DBG tier + P-OPT rereference + prefetch
// target into mask_width bits.
//
// Key insight: mask decouples classification (degree/rereference) from
// ordering (community). You can use Rabbit Order for locality AND get
// GRASP-quality caching via the mask, without DBG reordering.
struct MaskConfig {
    // ── Field widths ──
    uint8_t  mask_width = 8;        // Total bits per mask entry (8,16,32)
    uint8_t  dbg_bits = 2;          // Bits for degree tier (2-4)
    uint8_t  popt_bits = 4;         // Bits for rereference quantization (0-12)
    uint8_t  prefetch_bits = 2;     // Bits for prefetch target (remaining)

    // ── Prefetch ──
    bool     prefetch_direct = false; // true = raw vertex ID, false = hot table index
    uint32_t hot_table_size = 0;     // 2^prefetch_bits (only if !prefetch_direct)
    uint8_t  prefetch_window = 8;    // Dedup window size for prefetch (4,8,16)
    uint8_t  prefetch_mode = 0;      // 0=NONE, 1=DEGREE (highest-deg neighbor), 2=POPT (nearest reref)

    // ── ECG Mode ──
    // Controls eviction tiebreaker priority (see ECGMode enum above).
    // DBG_PRIMARY:  SRRIP → DBG tier → dynamic P-OPT (default)
    // POPT_PRIMARY: SRRIP → dynamic P-OPT → DBG tier
    // DBG_ONLY:     SRRIP → DBG tier (no P-OPT, fast path)
    ECGMode  ecg_mode = ECGMode::DBG_PRIMARY;

    // ── Classification ──
    uint8_t  num_buckets = 11;      // Number of degree buckets (2-16)
    uint8_t  rrpv_max = 7;          // Maximum RRPV value (3-bit=7, 8-bit=255)
    uint8_t  degree_mode = 0;       // 0=OUT, 1=IN, 2=BOTH
    bool     per_vertex = false;    // false=per-edge O(m), true=per-vertex O(n)
    bool     enabled = false;

    // ── Field positions (computed) ──
    uint8_t  prefetch_shift = 0;
    uint8_t  popt_shift = 0;
    uint8_t  dbg_shift = 0;
    uint32_t prefetch_mask_val = 0;
    uint32_t popt_mask_val = 0;
    uint32_t dbg_mask_val = 0;

    // Auto-compute field allocation from mask_width and graph size.
    // Priority: DBG (min 2) → POPT (scales with space) → PFX (remaining).
    // Wider masks give more P-OPT precision and larger prefetch target space.
    //
    // Allocation strategy:
    //   1. DBG always gets 2 bits (4 tiers)
    //   2. P-OPT gets as many bits as practical for oracle quality
    //   3. Prefetch gets remaining — but if < 4 bits, reallocate to P-OPT
    //      (fewer than 16 prefetch targets isn't meaningful for coverage)
    //
    // Results:
    //   8-bit:  DBG=2, POPT=6, PFX=0  (64 reref levels, no prefetch)
    //  16-bit:  DBG=2, POPT=8, PFX=6  (256 levels = matrix precision, 64 targets)
    //  32-bit:  DBG=2, POPT=8, PFX=22 (256 levels, 4M direct IDs)
    void autoAllocate(uint32_t num_vertices) {
        dbg_bits = 2;
        uint8_t remaining = mask_width - dbg_bits;

        // First pass: allocate P-OPT generously
        if (remaining >= 30) popt_bits = 8;        // 32-bit: cap at 8 (matrix precision), save rest for PFX
        else if (remaining >= 14) popt_bits = 8;   // 16-bit+: matrix-quality oracle
        else if (remaining >= 6) popt_bits = 6;    // 8-bit: 64 levels (decent precision)
        else if (remaining >= 2) popt_bits = 2;
        else popt_bits = 0;
        remaining -= popt_bits;

        // Second pass: prefetch only if >= 4 bits (16+ targets)
        // Otherwise give remaining bits to P-OPT for better oracle
        if (remaining >= 4) {
            prefetch_bits = remaining;
        } else {
            popt_bits += remaining;  // Reallocate to P-OPT
            prefetch_bits = 0;
        }

        // Can we encode vertex IDs directly?
        uint8_t id_bits = 1;
        while ((1ULL << id_bits) < num_vertices) id_bits++;
        prefetch_direct = (prefetch_bits >= id_bits);
        hot_table_size = prefetch_direct ? 0 : (prefetch_bits > 0 ? (1U << prefetch_bits) : 0);

        // Compute shifts and masks
        prefetch_shift = 0;
        popt_shift = prefetch_bits;
        dbg_shift = prefetch_bits + popt_bits;

        prefetch_mask_val = prefetch_bits ? ((1U << prefetch_bits) - 1) : 0;
        popt_mask_val = popt_bits ? (((1U << popt_bits) - 1) << popt_shift) : 0;
        dbg_mask_val = dbg_bits ? (((1U << dbg_bits) - 1) << dbg_shift) : 0;

        enabled = true;
    }

    // Initialize from environment variables
    void initFromEnv() {
        const char* v;
        if ((v = std::getenv("ECG_MASK_WIDTH")))    mask_width = static_cast<uint8_t>(std::atoi(v));
        if ((v = std::getenv("ECG_MODE")))          ecg_mode = StringToECGMode(v);
        if ((v = std::getenv("ECG_NUM_BUCKETS")))   num_buckets = static_cast<uint8_t>(std::atoi(v));
        if ((v = std::getenv("ECG_RRPV_BITS"))) {
            int bits = std::atoi(v);
            rrpv_max = (1 << bits) - 1;
        }
        if ((v = std::getenv("ECG_PREFETCH_WINDOW"))) prefetch_window = static_cast<uint8_t>(std::atoi(v));
        if ((v = std::getenv("ECG_PREFETCH_MODE"))) prefetch_mode = static_cast<uint8_t>(std::atoi(v));
        if ((v = std::getenv("ECG_PER_VERTEX")))    per_vertex = (std::atoi(v) != 0);
        if ((v = std::getenv("ECG_DEGREE_MODE"))) {
            if (std::string(v) == "IN") degree_mode = 1;
            else if (std::string(v) == "BOTH") degree_mode = 2;
            else degree_mode = 0;
        }
    }

    // Encode a mask entry from fields
    uint32_t encode(uint8_t dbg_tier, uint8_t popt_quant, uint32_t prefetch_target) const {
        uint32_t entry = 0;
        entry |= (prefetch_target & prefetch_mask_val);
        entry |= ((uint32_t(popt_quant) << popt_shift) & popt_mask_val);
        entry |= ((uint32_t(dbg_tier) << dbg_shift) & dbg_mask_val);
        return entry;
    }

    // Decode fields from a mask entry
    uint8_t decodeDBG(uint32_t entry) const {
        return dbg_bits ? static_cast<uint8_t>((entry & dbg_mask_val) >> dbg_shift) : 0;
    }
    uint8_t decodePOPT(uint32_t entry) const {
        return popt_bits ? static_cast<uint8_t>((entry & popt_mask_val) >> popt_shift) : 0;
    }
    uint32_t decodePrefetch(uint32_t entry) const {
        return prefetch_bits ? (entry & prefetch_mask_val) : 0;
    }

    // Compute RRPV from DBG tier (structural priority).
    // Higher tier = lower degree = higher RRPV (evict sooner).
    // P-OPT is NOT used at insert — consulted dynamically at eviction.
    uint8_t dbgTierToRRPV(uint8_t dbg_tier) const {
        float fraction = static_cast<float>(dbg_tier) / std::max(uint8_t(1), num_buckets);
        uint8_t result = static_cast<uint8_t>(rrpv_max * fraction);
        if (result > rrpv_max) result = rrpv_max;
        if (result == 0 && fraction > 0.0f) result = 1;  // Reserve 0 for hit promotion
        return result;
    }

    void printConfig(std::ostream& os = std::cout) const {
        if (!enabled) { os << "MaskConfig: disabled\n"; return; }
        os << "MaskConfig: " << int(mask_width) << "-bit"
           << " [DBG=" << int(dbg_bits)
           << " POPT=" << int(popt_bits)
           << " PFX=" << int(prefetch_bits) << "]"
           << " mode=" << ECGModeToString(ecg_mode)
           << " buckets=" << int(num_buckets)
           << " rrpv_max=" << int(rrpv_max)
           << " prefetch=" << (prefetch_direct ? "direct" : "table")
           << (per_vertex ? " per-vertex" : " per-edge") << "\n";
    }
};

// ============================================================================
// MaskArray: Per-edge (or per-vertex) mask storage
// ============================================================================
// Parallel array alongside CSR edges. Same indexing: masks[edge_idx].
// Stores encoded ECG hints (DBG tier + P-OPT rereference + prefetch target).
//
// For per-vertex mode: masks[vertex_id] — one entry per vertex, applied to
// all its edges. Cheaper (O(n) vs O(m)) but P-OPT loses per-edge precision.
struct MaskArray {
    const uint8_t*  data8 = nullptr;   // 8-bit mask entries
    const uint16_t* data16 = nullptr;  // 16-bit mask entries
    const uint32_t* data32 = nullptr;  // 32-bit mask entries
    uint64_t        count = 0;         // Number of entries
    uint8_t         entry_width = 0;   // 8, 16, or 32
    bool            enabled = false;

    // Get mask entry by index (auto-width)
    uint32_t get(uint64_t idx) const {
        if (!enabled || idx >= count) return 0;
        switch (entry_width) {
            case 8:  return data8  ? data8[idx]  : 0;
            case 16: return data16 ? data16[idx] : 0;
            case 32: return data32 ? data32[idx] : 0;
            default: return 0;
        }
    }

    // Initialize from raw data pointer
    void init8(const uint8_t* data, uint64_t n) {
        data8 = data; count = n; entry_width = 8; enabled = true;
    }
    void init16(const uint16_t* data, uint64_t n) {
        data16 = data; count = n; entry_width = 16; enabled = true;
    }
    void init32(const uint32_t* data, uint64_t n) {
        data32 = data; count = n; entry_width = 32; enabled = true;
    }
};

// ============================================================================
// FatIDConfig: Adaptive bit layout for encoded neighbor IDs
// ============================================================================
// After DBG reordering, vertex IDs carry inherent degree information (low
// IDs = high degree). ECG encodes cache hints into the top bits of CSR
// neighbor IDs so hints travel with the data through the cache hierarchy.
//
// This struct computes the adaptive bit allocation based on graph size:
//   [container_bits-1 ... real_id_bits]  = metadata fields
//   [real_id_bits-1 ... 0]              = real vertex ID
//
// Metadata fields (allocated from MSB to LSB in priority order):
//   1. DBG tier    (2 bits min) — cache insertion priority (HOT/WARM/LUKEWARM/COLD)
//   2. P-OPT quant (2-4 bits)  — rereference distance (log-quantized from transpose)
//   3. Prefetch Δ  (1-4 bits)  — prefetch delta hint
//
// Unlike P-OPT's rereference matrix (stored in LLC, consuming cache ways),
// fat-ID encoding has ZERO runtime storage overhead — hints are precomputed
// and baked into the CSR edge array during preprocessing.
//
// Quantization: With 2-4 bits instead of P-OPT's 7 bits, we use non-linear
// (logarithmic) mapping — more resolution at short distances where eviction
// decisions matter most.
struct FatIDConfig {
    uint8_t  container_bits = 32; // 32 or 64
    uint8_t  real_id_bits = 32;   // ceil(log2(num_vertices))
    uint8_t  spare_bits = 0;      // container_bits - real_id_bits

    // Metadata field widths (computed from spare_bits)
    uint8_t  dbg_bits = 0;        // Cache tier hint (min 2 when spare >= 2)
    uint8_t  popt_bits = 0;       // Rereference quantization
    uint8_t  prefetch_bits = 0;   // Prefetch delta
    uint8_t  _pad[2] = {};

    // Field positions (bit offset from LSB)
    uint8_t  prefetch_shift = 0;
    uint8_t  popt_shift = 0;
    uint8_t  dbg_shift = 0;

    // Masks for extraction
    uint64_t real_id_mask = 0;
    uint64_t dbg_mask = 0;
    uint64_t popt_mask = 0;
    uint64_t prefetch_mask = 0;

    bool     enabled = false;     // True after computeFromGraph()

    // Compute adaptive bit allocation from graph size.
    // Call once during preprocessing, before encoding any edges.
    void computeFromGraph(uint64_t num_vertices) {
        // Determine container size
        container_bits = (num_vertices > (1ULL << 30)) ? 64 : 32;

        // Real ID bits: minimum to address all vertices
        real_id_bits = 1;
        while ((1ULL << real_id_bits) < num_vertices)
            real_id_bits++;

        spare_bits = container_bits - real_id_bits;

        // If less than 2 spare bits in 32-bit, upgrade to 64-bit
        if (spare_bits < 2 && container_bits == 32) {
            container_bits = 64;
            spare_bits = container_bits - real_id_bits;
        }

        // Adaptive allocation scales with available spare bits.
        // 64-bit containers get much richer metadata than 32-bit.
        // With 8+ P-OPT bits, fat-ID encoding EXCEEDS P-OPT's original
        // 7-bit matrix precision — with zero LLC capacity consumed.
        //
        // Allocation tiers:
        //   spare >= 16: DBG=2, POPT=8, PFX=6 (exceeds P-OPT matrix!)
        //   spare >= 10: DBG=2, POPT=4, PFX=4
        //   spare >= 6:  DBG=2, POPT=2, PFX=2
        //   spare >= 4:  DBG=2, POPT=2, PFX=0
        //   spare >= 2:  DBG=2, POPT=0, PFX=0
        //   spare < 2:   DBG=spare, POPT=0, PFX=0
        if (spare_bits >= 16) {
            dbg_bits = 2;  popt_bits = 8;
            prefetch_bits = (spare_bits - 10 > 6) ? 6 : (spare_bits - 10);
        } else if (spare_bits >= 10) {
            dbg_bits = 2;  popt_bits = 4;
            prefetch_bits = (spare_bits - 6 > 4) ? 4 : (spare_bits - 6);
        } else if (spare_bits >= 6) {
            dbg_bits = 2;  popt_bits = 2;  prefetch_bits = spare_bits - 4;
        } else if (spare_bits >= 4) {
            dbg_bits = 2;  popt_bits = 2;  prefetch_bits = 0;
        } else if (spare_bits >= 2) {
            dbg_bits = 2;  popt_bits = 0;  prefetch_bits = 0;
        } else {
            dbg_bits = spare_bits;  popt_bits = 0;  prefetch_bits = 0;
        }

        // Compute shifts (fields packed from real_id upward)
        prefetch_shift = real_id_bits;
        popt_shift = prefetch_shift + prefetch_bits;
        dbg_shift = popt_shift + popt_bits;

        // Compute masks
        real_id_mask = (1ULL << real_id_bits) - 1;
        prefetch_mask = prefetch_bits ? (((1ULL << prefetch_bits) - 1) << prefetch_shift) : 0;
        popt_mask = popt_bits ? (((1ULL << popt_bits) - 1) << popt_shift) : 0;
        dbg_mask = dbg_bits ? (((1ULL << dbg_bits) - 1) << dbg_shift) : 0;

        enabled = true;
    }

    // ================================================================
    // Encoding (preprocessing time — called per edge)
    // ================================================================

    // Encode a fat neighbor ID from components.
    // dbg_tier: 0-3 (HOT=3, WARM=2, LUKEWARM=1, COLD=0 — high value = high priority)
    // popt_q:   quantized rereference distance (0=imminent, max=distant)
    // pfx_d:    prefetch delta encoding (0=none)
    uint64_t encode(uint64_t real_id, uint8_t dbg_tier,
                    uint8_t popt_q, uint8_t pfx_d) const {
        uint64_t fat = real_id & real_id_mask;
        if (dbg_bits)      fat |= (uint64_t(dbg_tier & ((1 << dbg_bits) - 1)) << dbg_shift);
        if (popt_bits)     fat |= (uint64_t(popt_q & ((1 << popt_bits) - 1)) << popt_shift);
        if (prefetch_bits) fat |= (uint64_t(pfx_d & ((1 << prefetch_bits) - 1)) << prefetch_shift);
        return fat;
    }

    // ================================================================
    // Decoding (runtime — called per neighbor access)
    // ================================================================

    uint64_t extractRealID(uint64_t fat) const {
        return fat & real_id_mask;
    }

    uint8_t extractDBGTier(uint64_t fat) const {
        return dbg_bits ? static_cast<uint8_t>((fat & dbg_mask) >> dbg_shift) : 0;
    }

    uint8_t extractPOPT(uint64_t fat) const {
        return popt_bits ? static_cast<uint8_t>((fat & popt_mask) >> popt_shift) : 0;
    }

    uint8_t extractPrefetch(uint64_t fat) const {
        return prefetch_bits ? static_cast<uint8_t>((fat & prefetch_mask) >> prefetch_shift) : 0;
    }

    // ================================================================
    // Quantization helpers (preprocessing time)
    // ================================================================

    // Quantize a full rereference distance (0..num_epochs) to popt_bits.
    // Uses logarithmic mapping: more resolution at short distances.
    // distance=0 → 0 (imminent), distance=max → max_val (distant)
    uint8_t quantizeRereference(uint32_t full_distance) const {
        if (popt_bits == 0) return 0;
        uint8_t max_val = (1 << popt_bits) - 1;
        if (full_distance == 0) return 0;
        // Log2-based quantization
        uint32_t log_dist = 0;
        uint32_t d = full_distance;
        while (d > 0) { d >>= 1; log_dist++; }
        return (log_dist > max_val) ? max_val : static_cast<uint8_t>(log_dist);
    }

    // Compute DBG tier for a vertex given its degree and graph topology.
    // Returns: 1=HOT, 2=WARM, 3=LUKEWARM, 4=COLD
    // Matches PropertyRegion::classify() numbering so fat-ID decode values
    // can be used directly by cache insertion code without remapping.
    uint8_t computeDBGTier(uint32_t degree, uint32_t avg_degree) const {
        if (dbg_bits == 0) return 4;  // COLD if no bits
        // ECG-style: geometric thresholds from avg_degree
        uint32_t hot_thresh = avg_degree * 4;   // Top hubs
        uint32_t warm_thresh = avg_degree;       // Above average
        uint32_t lukewarm_thresh = avg_degree / 2; // Near average
        if (degree >= hot_thresh)      return 1;  // HOT
        if (degree >= warm_thresh)     return 2;  // WARM
        if (degree >= lukewarm_thresh) return 3;  // LUKEWARM
        return 4;                                 // COLD
        if (degree >= lukewarm_thresh) return 1;  // LUKEWARM
        return 0;                                 // COLD
    }

    // Compute prefetch delta: distance to the next likely access vertex.
    // After DBG ordering, nearby IDs have similar degrees → likely co-accessed.
    // Returns encoded delta (0=none, 1=+1, 2=+2, 3=+4, etc.)
    uint8_t computePrefetchDelta(uint32_t src_id, uint32_t dst_id,
                                  uint32_t num_vertices) const {
        if (prefetch_bits == 0) return 0;
        // Simple heuristic: encode the stride to the next neighbor
        // In DBG-ordered graphs, consecutively-IDed vertices are often
        // co-accessed (same degree bucket). Delta encoding captures this.
        // For now: 0=no prefetch (default)
        // Future: compute from graph transpose neighbor patterns
        (void)src_id; (void)dst_id; (void)num_vertices;
        return 0;  // Placeholder — requires graph-specific analysis
    }

    // ================================================================
    // Diagnostics
    // ================================================================

    void printConfig(std::ostream& os = std::cout) const {
        if (!enabled) { os << "FatID: disabled\n"; return; }
        os << "FatID: " << int(container_bits) << "-bit container, "
           << int(real_id_bits) << "-bit ID, "
           << int(spare_bits) << " spare bits "
           << "[DBG=" << int(dbg_bits)
           << " POPT=" << int(popt_bits)
           << " PFX=" << int(prefetch_bits) << "]\n";
        os << "  Layout: [" << int(dbg_shift) << ":" << int(dbg_shift + dbg_bits - 1) << "]=DBG"
           << " [" << int(popt_shift) << ":" << int(popt_shift + popt_bits - 1) << "]=POPT"
           << " [" << int(prefetch_shift) << ":" << int(prefetch_shift + prefetch_bits - 1) << "]=PFX"
           << " [0:" << int(real_id_bits - 1) << "]=ID\n";
    }
};

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
        if (epoch_size == 0 || sub_epoch_size == 0) return 127;
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

    // --- Per-Access Hints (thread-safe: one per OMP thread) ---
    // Each OMP thread gets its own AccessHints to avoid data races
    // when setting current_src/mask during parallel graph traversal.
    // Mutable because hints are updated via const GraphCacheContext* in cache.
    static constexpr int ECG_MAX_THREADS = 128;
    mutable AccessHints thread_hints[ECG_MAX_THREADS];

    // Convenience accessor: returns hints for the calling thread.
    AccessHints& hints_for_thread() {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        return thread_hints[tid < ECG_MAX_THREADS ? tid : 0];
    }
    const AccessHints& hints_for_thread() const {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        return thread_hints[tid < ECG_MAX_THREADS ? tid : 0];
    }

    // Legacy single-thread alias (for backward compatibility with non-parallel code).
    // Accesses thread_hints[0] directly. In parallel regions, use hints_for_thread().
    AccessHints& hints_thread0() { return thread_hints[0]; }

    // --- Runtime prefetch dedup window (per-thread) ---
    // Tracks recently prefetched vertex IDs to suppress duplicates.
    // When a prefetch target matches any entry in the window, it's skipped.
    static constexpr int PREFETCH_DEDUP_MAX = 16;
    struct PrefetchDedupWindow {
        uint32_t entries[PREFETCH_DEDUP_MAX] = {};
        uint8_t head = 0;
        uint8_t size = 0;
        uint8_t capacity = 8;  // configurable via prefetch_window

        bool contains(uint32_t target) const {
            for (uint8_t i = 0; i < size; i++) {
                if (entries[(head + i) % PREFETCH_DEDUP_MAX] == target) return true;
            }
            return false;
        }

        void push(uint32_t target) {
            if (size < capacity) {
                entries[(head + size) % PREFETCH_DEDUP_MAX] = target;
                size++;
            } else {
                // Overwrite oldest
                entries[head] = target;
                head = (head + 1) % PREFETCH_DEDUP_MAX;
            }
        }
    };
    mutable PrefetchDedupWindow prefetch_dedup[ECG_MAX_THREADS];

    PrefetchDedupWindow& dedup_for_thread() const {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        return prefetch_dedup[tid < ECG_MAX_THREADS ? tid : 0];
    }

    // --- Rereference Matrix (P-OPT) ---
    RereferenceConfig rereference;

    // --- Prefetch Matrix (ECG-style graph-aware prefetching) ---
    // Maps each vertex to its predicted next-access vertex.
    // When a vertex v is accessed, prefetch data for prefetch_map[v].
    // Built from graph transpose (similar to P-OPT but for prefetching).
    // nullptr = prefetching disabled.
    const uint32_t* prefetch_map = nullptr;
    uint32_t prefetch_num_vertices = 0;

    // --- Fat ID Configuration (ECG adaptive encoding) ---
    FatIDConfig fat_id;

    // --- ECG Mask Configuration + Array ---
    // Parallel mask array alongside CSR edges for per-edge cache hints.
    // Decouples degree classification from vertex ordering.
    MaskConfig mask_config;
    MaskArray  mask_array;
    std::vector<uint32_t> hot_table;  // Hot vertex table for indexed prefetch

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
    //
    // Region boundary computation (DBG bucket-aligned):
    //   1. Compute how many vertex elements fit in LLC share for this array
    //      (cache_capacity_vertices = LLC_share / elem_size)
    //   2. Walk DBG degree buckets from highest to lowest, accumulating
    //      vertex counts until cache_capacity_vertices is reached
    //   3. That bucket boundary becomes hot_bound
    //   4. Continue for WARM (2× capacity) and LUKEWARM (4× capacity)
    //   5. Boundaries align with DBG bucket boundaries so region cuts
    //      don't split within a degree group
    //
    // Requires: initTopology() called before registerPropertyArray()
    //           Graph must be DBG-reordered for regions to be meaningful
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

        constexpr uint64_t LINE_MASK = ~uint64_t(63);

        if (topology.num_vertices > 0 && elem_size > 0) {
            // N-bucket boundaries aligned with DBG degree buckets.
            // Each bucket boundary marks where the next degree group starts.
            // DBG order: highest-degree at front (bucket N-1 → position 0).
            uint32_t active_buckets = 0;
            uint64_t cumulative_vtx = 0;

            // Walk from highest-degree bucket to lowest, accumulating vertex counts
            for (int b = GraphTopology::NUM_BUCKETS - 1; b >= 0; --b) {
                uint64_t count = topology.bucket_counts[b];
                if (count == 0) continue;  // Skip empty buckets
                cumulative_vtx += count;
                if (active_buckets < MAX_REGION_BUCKETS) {
                    uint64_t bound_bytes = cumulative_vtx * elem_size;
                    r.bucket_bounds[active_buckets] = (r.base_address + bound_bytes + 63) & LINE_MASK;
                    if (r.bucket_bounds[active_buckets] > r.upper_bound)
                        r.bucket_bounds[active_buckets] = r.upper_bound;
                    active_buckets++;
                }
            }
            r.num_buckets = active_buckets;

            // Ensure last bucket covers entire array
            if (active_buckets > 0)
                r.bucket_bounds[active_buckets - 1] = r.upper_bound;
        } else if (manual_hot_fraction > 0.0 && manual_hot_fraction <= 1.0) {
            // Manual: 4 buckets with geometric sizing
            uint64_t array_bytes = static_cast<uint64_t>(num_elements) * elem_size;
            uint64_t hot = static_cast<uint64_t>(manual_hot_fraction * array_bytes);
            r.bucket_bounds[0] = (r.base_address + hot + 63) & LINE_MASK;
            r.bucket_bounds[1] = (r.base_address + hot * 4 + 63) & LINE_MASK;
            r.bucket_bounds[2] = (r.base_address + hot * 16 + 63) & LINE_MASK;
            r.bucket_bounds[3] = r.upper_bound;
            for (int i = 0; i < 4; ++i)
                if (r.bucket_bounds[i] > r.upper_bound) r.bucket_bounds[i] = r.upper_bound;
            r.num_buckets = 4;
        } else {
            // Fallback: equal thirds (3 buckets)
            uint64_t array_bytes = static_cast<uint64_t>(num_elements) * elem_size;
            r.bucket_bounds[0] = (r.base_address + array_bytes / 3 + 63) & LINE_MASK;
            r.bucket_bounds[1] = (r.base_address + 2 * array_bytes / 3 + 63) & LINE_MASK;
            r.bucket_bounds[2] = r.upper_bound;
            r.num_buckets = 3;
        }

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

    // Initialize fat-ID encoding for ECG-style embedded hints.
    // Call after initTopology(). Computes adaptive bit allocation.
    void initFatID() {
        if (topology.num_vertices > 0) {
            fat_id.computeFromGraph(topology.num_vertices);
        }
    }

    // Encode a neighbor ID with all metadata fields.
    // Called during preprocessing (once per edge).
    uint64_t encodeFatNeighbor(uint64_t real_id, uint32_t degree,
                                uint32_t rereference_distance) const {
        if (!fat_id.enabled) return real_id;
        uint8_t dbg_tier = fat_id.computeDBGTier(degree, topology.avg_degree);
        uint8_t popt_q = fat_id.quantizeRereference(rereference_distance);
        uint8_t pfx_d = 0;  // Prefetch delta: placeholder for future
        return fat_id.encode(real_id, dbg_tier, popt_q, pfx_d);
    }

    // Decode a fat neighbor ID — extract just the real vertex ID.
    // Called at every neighbor access in the algorithm.
    uint64_t decodeFatNeighbor(uint64_t fat) const {
        if (!fat_id.enabled) return fat;
        return fat_id.extractRealID(fat);
    }

    // Extract all fields from a fat neighbor ID at once.
    // Used by the ECG cache policy for making replacement decisions.
    void decodeFatFull(uint64_t fat, uint64_t& real_id, uint8_t& dbg_tier,
                       uint8_t& popt_q, uint8_t& pfx_delta) const {
        real_id = fat_id.extractRealID(fat);
        dbg_tier = fat_id.extractDBGTier(fat);
        popt_q = fat_id.extractPOPT(fat);
        pfx_delta = fat_id.extractPrefetch(fat);
    }

    // ================================================================
    // ECG Mask Array Initialization
    // ================================================================

    // Initialize mask configuration from environment variables and graph size.
    // Call after initTopology().
    void initMaskConfig() {
        mask_config.initFromEnv();
        if (topology.num_vertices > 0) {
            mask_config.autoAllocate(topology.num_vertices);
        }
    }

    // Register a pre-computed mask array (per-edge or per-vertex).
    // The caller owns the data; the context stores a non-owning view.
    void initMaskArray8(const uint8_t* data, uint64_t count) {
        mask_array.init8(data, count);
    }
    void initMaskArray16(const uint16_t* data, uint64_t count) {
        mask_array.init16(data, count);
    }
    void initMaskArray32(const uint32_t* data, uint64_t count) {
        mask_array.init32(data, count);
    }

    // Build hot table for indexed prefetch targets (sorted by degree descending).
    // Only needed when prefetch_bits < id_bits (small mask width, large graph).
    void buildHotTable(const uint32_t* degrees, uint32_t num_vertices) {
        if (mask_config.hot_table_size == 0) return;
        // Collect (degree, vertex_id) pairs, sort descending
        std::vector<std::pair<uint32_t, uint32_t>> dv(num_vertices);
        for (uint32_t i = 0; i < num_vertices; ++i)
            dv[i] = {degrees[i], i};
        std::partial_sort(dv.begin(),
                          dv.begin() + std::min(uint32_t(mask_config.hot_table_size), num_vertices),
                          dv.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        hot_table.resize(std::min(uint32_t(mask_config.hot_table_size), num_vertices));
        for (size_t i = 0; i < hot_table.size(); ++i)
            hot_table[i] = dv[i].second;
    }

    // Get mask entry for an edge (or vertex in per-vertex mode).
    uint32_t getMaskEntry(uint64_t edge_or_vertex_idx) const {
        if (!mask_array.enabled) return 0;
        return mask_array.get(edge_or_vertex_idx);
    }

    // Decode a mask entry into its components.
    void decodeMask(uint32_t entry, uint8_t& dbg_tier, uint8_t& popt_quant,
                    uint32_t& prefetch_target) const {
        dbg_tier = mask_config.decodeDBG(entry);
        popt_quant = mask_config.decodePOPT(entry);
        prefetch_target = mask_config.decodePrefetch(entry);
    }

    // Compute RRPV from a mask entry (for ECG insert).
    // Uses only the DBG tier from the mask — P-OPT is consulted dynamically
    // at eviction time via findNextRef(), not at insert.
    uint8_t maskToRRPV(uint32_t mask_entry) const {
        uint8_t dbg = mask_config.decodeDBG(mask_entry);
        return mask_config.dbgTierToRRPV(dbg);
    }

    // Resolve prefetch target vertex ID from a mask entry.
    uint32_t resolvePrefetchTarget(uint32_t mask_entry) const {
        uint32_t raw = mask_config.decodePrefetch(mask_entry);
        if (raw == 0) return UINT32_MAX;  // No prefetch
        if (mask_config.prefetch_direct) return raw;  // Direct vertex ID
        if (raw < hot_table.size()) return hot_table[raw];  // Table lookup
        return UINT32_MAX;
    }

    // ================================================================
    // Per-Access Updates (maps to simulator registers/magic instructions)
    // ================================================================

    // Set current source and destination vertices (outer loop).
    // In Sniper: SimMagic1(CMD_SET_VERTICES, pack(src, dst))
    void setCurrentVertices(uint32_t src, uint32_t dst) {
        auto& h = hints_for_thread();
        h.current_src = src;
        h.current_dst = dst;
    }

    // Update destination vertex only (inner loop neighbor).
    void setCurrentDst(uint32_t dst) {
        hints_for_thread().current_dst = dst;
    }

    // Set ECG mask hint for current access.
    // In Sniper: the mask is encoded in the last 2 bits of vertex property data.
    void setMask(uint8_t mask) {
        hints_for_thread().mask = mask;
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

    // Classify an address to its degree bucket index (0 = highest-degree).
    // Returns: bucket index (0..N-1), or UINT32_MAX if not in any region.
    uint32_t classifyBucket(uint64_t addr) const {
        for (uint32_t i = 0; i < num_regions; ++i) {
            uint32_t b = regions[i].classifyBucket(addr);
            if (b < regions[i].num_buckets) return b;
        }
        return UINT32_MAX;
    }

    // Classify address into GRASP 3-tier reuse (Faldu et al. HPCA 2020).
    //
    // After DBG reorder, highest-degree vertices are at front (low addresses).
    // The original GRASP uses hot_fraction (default 10%) of LLC capacity to
    // define the hot boundary within each property region:
    //   HOT:      [base, base + f × llc_size)
    //   MODERATE: [base + f × llc_size, base + 2 × f × llc_size)
    //   COLD:     everything else (including non-property data)
    //
    // Returns: 1=HOT, 2=MODERATE, 3=COLD (0=not in any region)
    uint32_t classifyGRASP(uint64_t addr, size_t llc_size) const {
        constexpr double hot_fraction = 0.10;  // 10% of LLC, matches paper "f"
        uint64_t hot_bytes = static_cast<uint64_t>(hot_fraction * llc_size);

        for (uint32_t i = 0; i < num_regions; ++i) {
            if (addr >= regions[i].base_address && addr < regions[i].upper_bound) {
                uint64_t offset = addr - regions[i].base_address;
                if (offset < hot_bytes)          return 1;  // HOT (hubs)
                if (offset < 2 * hot_bytes)      return 2;  // MODERATE
                return 3;                                    // COLD
            }
        }
        return 3;  // Not in any property region → cold
    }

    // Map a bucket index to an RRPV value (GRASP-XP style).
    // Uses degree-proportional mapping: buckets with more total edges
    // get lower RRPV (higher cache priority).
    //
    // max_rrpv: maximum RRPV value (3-bit=7, 8-bit=255)
    // bucket: bucket index from classifyBucket() or extractDBGTier()
    //
    // Mapping: RRPV = max_rrpv × (1 - edge_fraction_of_bucket)
    //   High-degree bucket (many edges) → low RRPV (keep in cache)
    //   Low-degree bucket (few edges) → high RRPV (evict sooner)
    uint8_t bucketToRRPV(uint32_t bucket, uint8_t max_rrpv = 7) const {
        if (topology.num_vertices == 0 || topology.num_edges == 0)
            return max_rrpv;  // No topology: default to cold

        if (bucket >= GraphTopology::NUM_BUCKETS)
            return max_rrpv;  // Unknown bucket: cold

        // classifyBucket() returns: 0 = highest degree (front after DBG reorder),
        //                           N-1 = lowest degree (back).
        // topology.bucket_total_degrees[]: 0 = lowest degree, N-1 = highest.
        // Reverse spatial → topology: dbg_bucket = N-1 - classifier_bucket.
        uint32_t dbg_bucket = GraphTopology::NUM_BUCKETS - 1 - bucket;

        // GRASP RRPV formula: edge_fraction = cumulative share of edges at this
        // degree level and ALL HIGHER degree levels. For the highest-degree bucket
        // (dbg_bucket = N-1), this includes everything from N-1 down... but we want
        // hubs to get LOW RRPV, meaning HIGH edge_fraction.
        //
        // Correct summation: from dbg_bucket UP to N-1 (all degrees >= this level).
        // For hubs (dbg_bucket = N-1): just the top bucket → fraction = hub edge share.
        // For cold (dbg_bucket = 0): all buckets → fraction = 1.0.
        //
        // But this gives cold vertices LOW RRPV (fraction=1.0 → rrpv=0)!
        // The fix: sum from 0 to dbg_bucket (cumulative from bottom), so:
        //   hubs (dbg_bucket = N-1) → sum all → fraction = 1.0 → rrpv = 0 (keep)
        //   cold (dbg_bucket = 0) → sum just bottom → fraction = small → rrpv = high (evict)
        uint64_t edge_sum = 0;
        for (uint32_t b = 0; b <= dbg_bucket; ++b)
            edge_sum += topology.bucket_total_degrees[b];

        double edge_fraction = static_cast<double>(edge_sum) / topology.num_edges;

        // RRPV = max × (1 - edge_fraction)
        // High edge_fraction → low RRPV (important, keep in cache)
        // Low edge_fraction → high RRPV (unimportant, evict)
        uint8_t rrpv = static_cast<uint8_t>(max_rrpv * (1.0 - edge_fraction));
        if (rrpv >= max_rrpv) rrpv = max_rrpv;
        if (rrpv == 0 && edge_fraction < 1.0) rrpv = 1;  // Reserve 0 for hit promotion
        return rrpv;
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
        if (hints_for_thread().current_src == UINT32_MAX) return 127;  // No vertex context set
        const PropertyRegion* r = findRegion(line_addr);
        if (r == nullptr) return 127;
        uint32_t cline_id = static_cast<uint32_t>(
            (line_addr - r->base_address) / rereference.line_size);
        return rereference.findNextRef(cline_id, hints_for_thread().current_src);
    }

    // ================================================================
    // Mask Computation (Phase 2 — preprocessing, called once)
    // ================================================================

    // Compute per-vertex DBG tier classification WITHOUT reordering.
    // Classifies each vertex by degree bucket relative to avg_degree.
    // Works with ANY vertex ordering (Rabbit, Leiden, GraphBrew, etc.).
    //
    // Returns: tier per vertex (0 = highest-degree, NUM_BUCKETS-1 = lowest)
    template<typename GraphT>
    std::vector<uint8_t> computeVertexTiers(const GraphT& g) const {
        uint32_t n = g.num_nodes();
        std::vector<uint8_t> tiers(n, 0);
        if (topology.num_vertices == 0) return tiers;

        #pragma omp parallel for schedule(static)
        for (uint32_t v = 0; v < n; ++v) {
            uint32_t deg = (mask_config.degree_mode == 1)
                ? static_cast<uint32_t>(g.in_degree(v))
                : static_cast<uint32_t>(g.out_degree(v));
            uint8_t tier = static_cast<uint8_t>(topology.NUM_BUCKETS - 1);
            for (uint32_t b = 0; b < topology.NUM_BUCKETS; ++b) {
                if (deg <= topology.bucket_thresholds[b]) {
                    tier = static_cast<uint8_t>(topology.NUM_BUCKETS - 1 - b);
                    break;
                }
            }
            tiers[v] = tier;
        }
        return tiers;
    }

    // Compute per-vertex mask entries with DBG tier, P-OPT hint, and prefetch target.
    //
    // P-OPT hint: quantized rereference distance from matrix (or degree proxy).
    //   Wider popt_bits → less quantization loss → better ECG_EMBEDDED oracle.
    //
    // Prefetch target: neighbor vertex ID to prefetch when this vertex is accessed.
    //   prefetch_mode 0 (NONE): no prefetch target (all zeros)
    //   prefetch_mode 1 (DEGREE): highest-degree neighbor not in dedup window
    //   prefetch_mode 2 (POPT): neighbor with shortest rereference distance
    //
    // Construction-time dedup: sliding window over vertex scan order prevents
    //   encoding the same prefetch target for consecutive vertices.
    //
    // Returns: vector of encoded mask entries (8-bit for mask_width=8)
    template<typename GraphT>
    std::vector<uint8_t> computeVertexMasks8(const GraphT& g) {
        auto masks32 = computeVertexMasks(g);
        std::vector<uint8_t> masks8(masks32.size());
        for (size_t i = 0; i < masks32.size(); i++)
            masks8[i] = static_cast<uint8_t>(masks32[i]);
        return masks8;
    }

    // General mask computation supporting any width (8/16/32 bits).
    // Returns uint32_t entries — caller truncates to mask_width.
    template<typename GraphT>
    std::vector<uint32_t> computeVertexMasks(const GraphT& g) {
        if (!mask_config.enabled) {
            initMaskConfig();
            if (topology.num_vertices > 0)
                mask_config.autoAllocate(topology.num_vertices);
        }
        auto tiers = computeVertexTiers(g);
        uint32_t n = g.num_nodes();
        std::vector<uint32_t> masks(n, 0);

        uint32_t popt_max = mask_config.popt_bits > 0 ? ((1U << mask_config.popt_bits) - 1) : 0;
        uint32_t pfx_max = mask_config.prefetch_bits > 0 ? ((1U << mask_config.prefetch_bits) - 1) : 0;

        // Build hot table if using TABLE mode for prefetch
        if (pfx_max > 0 && !mask_config.prefetch_direct && hot_table.empty()) {
            std::vector<uint32_t> deg_arr(n);
            for (uint32_t v = 0; v < n; v++)
                deg_arr[v] = static_cast<uint32_t>(g.out_degree(v));
            buildHotTable(deg_arr.data(), n);
        }

        // Build hot table set for O(1) membership lookup
        std::unordered_set<uint32_t> hot_set;
        if (!mask_config.prefetch_direct && !hot_table.empty()) {
            for (auto id : hot_table) hot_set.insert(id);
        }

        // Construction-time dedup window (sliding across vertex scan order)
        // Not parallelized — sequential scan to maintain window state
        PrefetchDedupWindow build_dedup;
        build_dedup.capacity = mask_config.prefetch_window;

        for (uint32_t v = 0; v < n; ++v) {
            // --- P-OPT hint ---
            uint32_t popt_hint = 0;
            if (popt_max > 0 && topology.num_vertices > 0) {
                if (rereference.matrix) {
                    // Average rereference distance across ALL epochs for this
                    // vertex's cache line. This gives a representative distance
                    // instead of sampling a single epoch.
                    constexpr int numVtxPerLine = 16;
                    uint32_t cline = v / numVtxPerLine;
                    if (cline < rereference.num_cache_lines) {
                        uint32_t total_dist = 0;
                        uint32_t count = 0;
                        for (uint32_t e = 0; e < rereference.num_epochs; e++) {
                            uint8_t entry = rereference.matrix[e * rereference.num_cache_lines + cline];
                            if ((entry & 0x80) == 0) {  // Not referenced this epoch
                                total_dist += (entry & 0x7F);
                                count++;
                            }
                            // Referenced epochs (MSB=1) have distance 0 — skip
                        }
                        uint8_t avg_dist = count > 0
                            ? static_cast<uint8_t>(std::min(total_dist / count, uint32_t(127)))
                            : 0;
                        popt_hint = (uint32_t(avg_dist) * popt_max) / 127;
                    }
                } else {
                    uint32_t deg = g.out_degree(v);
                    if (deg == 0) popt_hint = popt_max;
                    else {
                        double ratio = static_cast<double>(deg) / topology.max_degree;
                        popt_hint = static_cast<uint32_t>(popt_max * (1.0 - ratio));
                    }
                }
            }

            // --- Prefetch target ---
            uint32_t pfx_value = 0;
            if (pfx_max > 0 && mask_config.prefetch_mode > 0) {
                // Find best neighbor to prefetch
                uint32_t best_target = UINT32_MAX;

                if (mask_config.prefetch_mode == 1) {
                    // DEGREE mode: highest-degree neighbor not in dedup window
                    uint32_t best_deg = 0;
                    for (auto ngh : g.out_neigh(v)) {
                        uint32_t nd = static_cast<uint32_t>(g.out_degree(ngh));
                        if (nd > best_deg && !build_dedup.contains(ngh)) {
                            best_deg = nd;
                            best_target = ngh;
                        }
                    }
                } else if (mask_config.prefetch_mode == 2 && rereference.matrix) {
                    // POPT mode: neighbor with shortest avg rereference distance
                    uint32_t best_dist = 128;
                    constexpr int numVtxPerLine = 16;
                    for (auto ngh : g.out_neigh(v)) {
                        uint32_t ncline = static_cast<uint32_t>(ngh) / numVtxPerLine;
                        if (ncline < rereference.num_cache_lines) {
                            // Average distance across epochs for this neighbor
                            uint32_t td = 0, cnt = 0;
                            for (uint32_t e = 0; e < rereference.num_epochs; e++) {
                                uint8_t entry = rereference.matrix[e * rereference.num_cache_lines + ncline];
                                if ((entry & 0x80) == 0) { td += (entry & 0x7F); cnt++; }
                            }
                            uint32_t dist = cnt > 0 ? td / cnt : 127;
                            if (dist < best_dist && !build_dedup.contains(ngh)) {
                                best_dist = dist;
                                best_target = ngh;
                            }
                        }
                    }
                }

                if (best_target != UINT32_MAX) {
                    build_dedup.push(best_target);
                    if (mask_config.prefetch_direct) {
                        pfx_value = best_target & pfx_max;
                    } else if (!hot_table.empty()) {
                        // Encode as hot table index (+1, 0 = no prefetch)
                        for (uint32_t hi = 0; hi < hot_table.size(); hi++) {
                            if (hot_table[hi] == best_target) {
                                pfx_value = hi + 1;
                                break;
                            }
                        }
                        // If target not in hot table, try to find any hot neighbor
                        if (pfx_value == 0) {
                            for (auto ngh : g.out_neigh(v)) {
                                if (hot_set.count(ngh) && !build_dedup.contains(ngh)) {
                                    for (uint32_t hi = 0; hi < hot_table.size(); hi++) {
                                        if (hot_table[hi] == static_cast<uint32_t>(ngh)) {
                                            pfx_value = hi + 1;
                                            build_dedup.push(ngh);
                                            break;
                                        }
                                    }
                                    if (pfx_value != 0) break;
                                }
                            }
                        }
                    }
                }
            }

            masks[v] = mask_config.encode(tiers[v],
                                          static_cast<uint8_t>(std::min(popt_hint, popt_max)),
                                          pfx_value);
        }
        return masks;
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
            os << "  [" << i << "] "
               << total << "B (elem=" << r.elem_size << "B × " << r.num_elements << ")"
               << "  " << r.num_buckets << " buckets\n";
        }
        if (rereference.matrix) {
            os << "Rereference: " << rereference.num_epochs << " epochs × "
               << rereference.num_cache_lines << " cache lines\n";
        }
        if (vertex_stats.enabled) {
            os << "Per-vertex stats: enabled (" << vertex_stats.accesses.size() << " vertices)\n";
        }
        if (fat_id.enabled) {
            fat_id.printConfig(os);
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
