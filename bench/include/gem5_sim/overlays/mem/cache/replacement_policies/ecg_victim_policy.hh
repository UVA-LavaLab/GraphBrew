// ECG policy (GRASP insertion tier + ECG_GRASP_POPT eviction) — single source of
// truth for all simulators.
//
// cache_sim, gem5 and Sniper all call ecg_policy::selectVictim with a per-way
// WayState built from their native cache-line structures. The DECISION logic is
// therefore identical across the three (nothing is "ported" or "mirrored"), so a
// single unit test of this function verifies the eviction choice for every
// backend; each simulator's thin adapter (native state -> WayState) is covered by
// its live eviction trace. See scripts/experiments/ecg/verify_ecg.py and
// bench/src_sim/test_ecg_victim.cc.
//
// The five variants (selected by ECG_VARIANT) and the invariants are documented
// in wiki/ECG-Policy-Comparison.md. Summary:
//   - epoch is PROPERTY-ONLY; record (non-property) lines never carry a usable
//     epoch and are ranked by recency / set order.
//   - "recency" is normalised so SMALLER == older == evict-first. cache_sim and
//     gem5 pass last_access / lastTouchTick directly; Sniper, which has no
//     per-line timestamp, passes a monotone-decreasing function of its RRIP age
//     so the oldest-by-RRIP line is evicted first (consistent across variants).
//   - rrpv is aged in place (the SRRIP state update); the caller must write the
//     possibly-incremented rrpv back to its native lines.
#ifndef ECG_VICTIM_POLICY_H
#define ECG_VICTIM_POLICY_H

#include <cstddef>
#include <cstdint>

namespace ecg_policy {

enum Variant {
    GRASP_ONLY   = 0,  // pure RRIP, no epoch (== GRASP)
    EPOCH_FIRST  = 1,  // records by recency, then farthest-epoch property (epoch vetoes rrpv)
    RRIP_FIRST   = 2,  // max-rrpv set (recency vetoes); records-first, then farthest-epoch property
    EPOCH_ONLY   = 3,  // same eviction as EPOCH_FIRST (differs only at insertion)
    SHORTCIRCUIT = 4,  // non-property first (set order), then farthest RAW-dist property
};

struct WayState {
    bool     prop;     // property (vertex) line, vs record (edge-stream) line
    uint8_t  rrpv;     // RRIP age (aged in place for variants that age)
    uint64_t recency;  // smaller == older == evict-first
    uint8_t  dbg;      // DBG degree tier (shortcircuit all-property tiebreak)
    uint32_t dist;     // raw circular next-ref distance (stored_epoch + ne - cur_epoch) % ne
    bool     stamped;  // epoch is meaningful here (property line with a live stamp)
};

// Effective epoch distance: rrip_first/epoch_* treat an unstamped line as
// distance 0 (kept), so only genuinely stamped property competes on epoch.
inline uint32_t effDist(const WayState& w) { return w.stamped ? w.dist : 0; }

// Select the victim index among ways[0..n). Ages rrpv in place where the variant
// ages. n must be >= 1.
inline size_t selectVictim(WayState* ways, size_t n, int variant, uint8_t rrpvMax) {
    // grasp_only: pure RRIP — first line at max RRPV, aging until one reaches it.
    if (variant == GRASP_ONLY) {
        for (;;) {
            for (size_t i = 0; i < n; i++) if (ways[i].rrpv >= rrpvMax) return i;
            for (size_t i = 0; i < n; i++) if (ways[i].rrpv < rrpvMax) ways[i].rrpv++;
        }
    }

    // shortcircuit (legacy): evict any non-property line first (set order); if the
    // set is all property, evict the farthest effective-dist line (unstamped property
    // -> dist 0 = kept, so only genuinely stamped property competes; DBG tiebreak).
    if (variant == SHORTCIRCUIT) {
        for (size_t i = 0; i < n; i++) if (!ways[i].prop) return i;
        size_t best = 0; uint32_t bd = 0; uint8_t bdbg = 0;
        for (size_t i = 0; i < n; i++) {
            uint32_t d = effDist(ways[i]);
            if (d > bd || (d == bd && ways[i].dbg > bdbg)) { best = i; bd = d; bdbg = ways[i].dbg; }
        }
        return best;
    }

    // epoch_first / epoch_only: records first by recency (no rrpv gate); else the
    // farthest-next-ref stamped property; else recency fallback (LRU).
    if (variant == EPOCH_FIRST || variant == EPOCH_ONLY) {
        size_t rec = n; uint64_t ro = 0;
        for (size_t i = 0; i < n; i++) if (!ways[i].prop)
            if (rec == n || ways[i].recency < ro) { rec = i; ro = ways[i].recency; }
        if (rec != n) return rec;
        size_t best = n; uint32_t bd = 0;
        for (size_t i = 0; i < n; i++) if (ways[i].stamped) {
            uint32_t d = ways[i].dist;
            if (best == n || d > bd) { best = i; bd = d; }
        }
        if (best != n) return best;
        size_t v = 0; uint64_t o = ways[0].recency;
        for (size_t i = 1; i < n; i++) if (ways[i].recency < o) { o = ways[i].recency; v = i; }
        return v;
    }

    // rrip_first (default): among the max-RRPV set, evict the oldest record by
    // recency; else the farthest effective-epoch property. Age and retry if the
    // max-RRPV set yields no candidate.
    for (;;) {
        size_t recIdx = n; uint64_t ro = 0;
        size_t propIdx = n; uint32_t pb = 0;
        for (size_t i = 0; i < n; i++) {
            if (ways[i].rrpv < rrpvMax) continue;
            if (!ways[i].prop) {
                if (recIdx == n || ways[i].recency < ro) { recIdx = i; ro = ways[i].recency; }
            } else {
                uint32_t d = effDist(ways[i]);
                if (propIdx == n || d > pb) { propIdx = i; pb = d; }
            }
        }
        if (recIdx != n) return recIdx;
        if (propIdx != n) return propIdx;
        for (size_t i = 0; i < n; i++) if (ways[i].rrpv < rrpvMax) ways[i].rrpv++;
    }
}

// ---------------------------------------------------------------------------
// GRASP insertion classification — the OTHER half of the ECG policy SSOT.
// Insertion RRPV is GRASP's whole mechanism (Faldu HPCA'20): high-degree
// property lines are protected, low-degree lines are evicted first. A single
// implementation here guarantees cache_sim / gem5 / Sniper classify and insert
// identically (the eviction DECISION above and the INSERTION tier below are now
// both single-source). cache_sim/gem5/Sniper each iterate their own property
// regions and call classifyGraspTier per region — no logic is mirrored.
// ---------------------------------------------------------------------------

// GRASP degree tier of `addr` within ONE property region [base, upper):
// the top `hot_fraction` of the (DBG-reordered) array is HOT(1), the next
// `hot_fraction` is MODERATE(2), the rest is COLD(3). Returns 0 when `addr`
// is outside [base, upper) so the caller can try the next region. The +8
// boundary nudge matches the upstream GRASP (ligra common.h add_region) rule.
inline uint32_t classifyGraspTier(uint64_t addr, uint64_t base, uint64_t upper,
                                  double hot_fraction) {
    if (addr < base || addr >= upper) return 0;
    const uint64_t array_bytes = upper - base;
    const uint64_t hot_bytes = static_cast<uint64_t>(hot_fraction * array_bytes);
    uint64_t hot_bound = base + hot_bytes;
    uint64_t moderate_bound = base + 2 * hot_bytes;
    if (hot_bound > upper) hot_bound = upper;
    if (moderate_bound > upper) moderate_bound = upper;
    hot_bound += 8;
    moderate_bound += 8;
    if (addr < hot_bound) return 1;       // HOT (hubs)  -> protected insertion
    if (addr < moderate_bound) return 2;  // MODERATE
    return 3;                             // COLD        -> evict-first insertion
}

// GRASP insertion RRPV for a degree tier (1/2/3 from classifyGraspTier; 0 or
// out-of-region maps to cold). P_RRIP=1 (protected), I_RRIP=rrpvMax-1,
// M_RRIP=rrpvMax — i.e. 1 / 6 / 7 for a 3-bit RRPV.
inline uint8_t graspTierRRPV(uint32_t tier, uint8_t rrpvMax) {
    if (tier == 1) return 1;
    if (tier == 2) return (rrpvMax > 1) ? static_cast<uint8_t>(rrpvMax - 1) : rrpvMax;
    return rrpvMax;
}

// GRASP degree tier of vertex v by its POSITION in the (DBG-reordered) property
// array: top hot_fraction = HOT(1), next hot_fraction = MODERATE(2), rest COLD(3).
// With elem_size>0 this is BYTE-EXACT to classifyGraspTier (same floor + the +8
// boundary nudge), so the DELIVERED "ECG mask" variant is byte-identical to the
// region-based "original GRASP". elem_size==0 uses the plain index split (for the
// mask dbg tiebreak, where exactness is not required). The per-vertex form is
// computed offline + delivered, so it is identical across simulators.
inline uint32_t graspTierByIndex(uint64_t v, uint64_t num_vertices,
                                 double hot_fraction, uint32_t elem_size = 0) {
    if (num_vertices == 0) return 3;
    if (elem_size == 0) {  // index split (no +8) — mask dbg tiebreak only
        double pos = static_cast<double>(v) / static_cast<double>(num_vertices);
        if (pos < hot_fraction) return 1;
        if (pos < 2.0 * hot_fraction) return 2;
        return 3;
    }
    // BYTE-EXACT mirror of classifyGraspTier (base=0, addr=v*elem_size).
    const uint64_t array_bytes = num_vertices * elem_size;
    const uint64_t addr_off = v * elem_size;
    const uint64_t hot_bytes = static_cast<uint64_t>(hot_fraction * array_bytes);
    uint64_t hot_bound = hot_bytes;            if (hot_bound > array_bytes) hot_bound = array_bytes;
    uint64_t mod_bound = 2 * hot_bytes;        if (mod_bound > array_bytes) mod_bound = array_bytes;
    hot_bound += 8;
    mod_bound += 8;
    if (addr_off < hot_bound) return 1;        // HOT (hubs at the array front)
    if (addr_off < mod_bound) return 2;        // MODERATE
    return 3;                                  // COLD
}

}  // namespace ecg_policy

#endif  // ECG_VICTIM_POLICY_H
