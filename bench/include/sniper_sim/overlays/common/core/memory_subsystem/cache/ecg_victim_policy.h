// ECG_GRASP_POPT eviction decision — single source of truth for all simulators.
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
    // set is all property, evict the farthest RAW-dist line (DBG tier tiebreak).
    if (variant == SHORTCIRCUIT) {
        for (size_t i = 0; i < n; i++) if (!ways[i].prop) return i;
        size_t best = 0; uint32_t bd = 0; uint8_t bdbg = 0;
        for (size_t i = 0; i < n; i++) {
            uint32_t d = ways[i].dist;
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

}  // namespace ecg_policy

#endif  // ECG_VICTIM_POLICY_H
