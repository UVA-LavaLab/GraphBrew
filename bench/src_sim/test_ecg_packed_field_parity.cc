// Packed-ISA field parity test (3-sim equivalency, Phase B / doc S10).
//
// Two delivery layouts share ecg_mode6_builder.h:
//   - packMask          : cache_sim/Sniper "fat" mask, 31-bit prefetch target at bits [33:64]
//                         (extractPrefetchTarget, mask 0x7FFFFFFF).
//   - packMaskEpoch     : gem5 ISA mask, epoch at [33:49] + 15-bit prefetch target at [49:64]
//                         (extractPrefetchTargetEpoch, mask 0x7FFF).
//
// This pins the field layout against silent repack/shift regressions and PROVES the
// documented divergence: the 31-bit and 15-bit targets AGREE for targets <= 32767
// (in-range parity), but packMaskEpoch TRUNCATES targets > 32767 to a WRONG vertex while
// packMask preserves them -> the gem5 large-graph ECG_PFX limitation (doc S10 validity
// matrix). verify_pfx.py proves target SELECTION parity; this proves field DELIVERY parity.
#include "ecg_mode6_builder.h"
#include <cstdio>
#include <cstdint>

using namespace ecg_mode6;

static int g_pass = 0, g_fail = 0;
static void check(const char* n, uint64_t got, uint64_t exp) {
    bool ok = (got == exp);
    printf("    %-54s got=%-10llu expect=%-10llu [%s]\n", n,
           (unsigned long long)got, (unsigned long long)exp, ok ? "OK" : "FAIL");
    if (ok) g_pass++; else g_fail++;
}

int main() {
    printf("[test_ecg_packed_field_parity] packMask (31-bit) vs packMaskEpoch (15-bit) target\n");

    // (1) Non-target fields round-trip in BOTH layouts.
    {
        uint32_t dest = 0x123456; uint8_t dbg = 2; uint8_t popt = 0x5A; uint16_t epoch = 0xABCD;
        uint64_t m1 = packMask(dest, dbg, popt, 12345);
        check("packMask dest", extractDest(m1), dest);
        check("packMask dbg", extractDbg(m1), dbg);
        check("packMask popt", extractPopt(m1), popt & 0x7F);
        uint64_t m2 = packMaskEpoch(dest, dbg, popt, epoch, 12345);
        check("packMaskEpoch dest", extractDest(m2), dest);
        check("packMaskEpoch dbg", extractDbg(m2), dbg);
        check("packMaskEpoch popt", extractPopt(m2), popt & 0x7F);
        check("packMaskEpoch epoch", extractEpoch(m2), epoch);
    }

    // (2) In-range parity: for targets <= 32767, both layouts deliver the SAME target.
    for (uint32_t t : {0u, 1u, 100u, 1000u, 16384u, 32766u, 32767u}) {
        uint64_t m1 = packMask(7, 0, 0, t);
        uint64_t m2 = packMaskEpoch(7, 0, 0, 0, t);
        char nm[80];
        snprintf(nm, sizeof nm, "in-range t=%u: packMask 31-bit", t);
        check(nm, extractPrefetchTarget(m1), t);
        snprintf(nm, sizeof nm, "in-range t=%u: packMaskEpoch 15-bit", t);
        check(nm, extractPrefetchTargetEpoch(m2), t);
        snprintf(nm, sizeof nm, "in-range t=%u: 31-bit == 15-bit (PARITY)", t);
        check(nm, extractPrefetchTarget(m1), extractPrefetchTargetEpoch(m2));
    }

    // (3) Out-of-range (the gem5 limitation): targets > 32767 survive in packMask (31-bit)
    //     but TRUNCATE in packMaskEpoch (15-bit) -> the 15-bit field is a WRONG vertex.
    for (uint32_t t : {32768u, 65535u, 916428u, 0x7FFFFFFFu}) {
        uint64_t m1 = packMask(7, 0, 0, t);
        uint64_t m2 = packMaskEpoch(7, 0, 0, 0, t);
        char nm[96];
        snprintf(nm, sizeof nm, "out-of-range t=%u: packMask 31-bit preserves", t);
        check(nm, extractPrefetchTarget(m1), t & 0x7FFFFFFFu);
        snprintf(nm, sizeof nm, "out-of-range t=%u: packMaskEpoch truncates to t&0x7FFF", t);
        check(nm, extractPrefetchTargetEpoch(m2), t & 0x7FFFu);
        snprintf(nm, sizeof nm, "out-of-range t=%u: 15-bit field is WRONG (!= true target)", t);
        check(nm, (extractPrefetchTargetEpoch(m2) != t) ? 1u : 0u, 1u);
    }

    // (4) Truncation boundary: 32767 OK, 32768 -> 0 (silent wrong vertex).
    check("boundary 32767 ok in 15-bit",
          extractPrefetchTargetEpoch(packMaskEpoch(0, 0, 0, 0, 32767)), 32767);
    check("boundary 32768 -> 0 in 15-bit (truncated)",
          extractPrefetchTargetEpoch(packMaskEpoch(0, 0, 0, 0, 32768)), 0);

    // (5) WIDE layout (gem5 large-graph fix, doc S10.2): packMaskEpochWide reclaims the
    //     vestigial dbg+popt fields to carry a 24-bit prefetch target -> covers ids
    //     <= 16,777,215 (all headline graphs), fixing what the 15-bit field truncated.
    {
        printf("  -- packMaskEpochWide (24-bit target) --\n");
        uint32_t dest = 0xABCDEF; uint16_t epoch = 0x1234; uint32_t pfx = 916428; // web-Google id
        uint64_t w = packMaskEpochWide(dest, epoch, pfx);
        check("wide dest round-trip", extractDest(w), dest);
        check("wide epoch round-trip", extractEpochWide(w), epoch);
        check("wide pfx web-Google 916428 SURVIVES (15-bit truncated it)",
              extractPrefetchTargetWide(w), pfx);
        check("wide pfx boundary 16777215 (2^24-1) ok",
              extractPrefetchTargetWide(packMaskEpochWide(0, 0, 16777215u)), 16777215u);
        check("wide pfx 16777216 (2^24) truncates to 0",
              extractPrefetchTargetWide(packMaskEpochWide(0, 0, 16777216u)), 0u);
        // Cross-check: the same target that the 15-bit packMaskEpoch mangled (916428 ->
        // 31692) is preserved by the wide layout.
        check("wide preserves what 15-bit mangled (916428 != 31692)",
              (extractPrefetchTargetWide(packMaskEpochWide(0, 0, 916428u)) == 916428u) ? 1u : 0u, 1u);
    }

    // (6) EPOCH-ONLY honest layout (doc S13): NO prefetch-target field — Path A
    //     reads ahead in the CSR (= DROPLET), so nothing is stored for prefetch.
    //     The reclaimed bits widen the epoch (34 bits) so it never starves as ids
    //     grow, and the dest covers 268M verts (twitter/friendster/kron-s27).
    {
        printf("  -- packMaskEpochOnly (no prefetch field, 34-bit epoch) --\n");
        uint32_t dest = 0x0ABCDEF;          // 28-bit dest
        uint64_t epoch = 0x3FFFFFFFFULL;    // 34-bit all-ones
        uint64_t e = packMaskEpochOnly(dest, 2, epoch);
        check("epoch-only dest round-trip", extractDestEpochOnly(e), dest);
        check("epoch-only dbg round-trip", extractDbgEpochOnly(e), 2);
        check("epoch-only epoch round-trip (34-bit)", extractEpochOnly(e), epoch);
        // Scale: dest covers the comparison-paper graphs.
        check("epoch-only dest twitter 41.7M ok",
              extractDestEpochOnly(packMaskEpochOnly(41700000u, 0, 0)), 41700000u);
        check("epoch-only dest friendster 65.6M ok",
              extractDestEpochOnly(packMaskEpochOnly(65600000u, 0, 0)), 65600000u);
        check("epoch-only dest kron-s27 134M ok",
              extractDestEpochOnly(packMaskEpochOnly(134217727u, 0, 0)), 134217727u);
        check("epoch-only dest boundary 2^28-1 ok",
              extractDestEpochOnly(packMaskEpochOnly(268435455u, 0, 0)), 268435455u);
        // Precision: the epoch carries values a 16-bit field would truncate.
        check("epoch-only epoch 100000 (>16-bit) survives (16-bit would truncate)",
              extractEpochOnly(packMaskEpochOnly(7, 0, 100000u)), 100000u);
        check("epoch-only epoch 2^20 survives",
              extractEpochOnly(packMaskEpochOnly(7, 0, 1048576u)), 1048576u);
        // No prefetch field: a large value in the old prefetch position is now
        // part of the EPOCH, not a separate target — dest stays clean.
        check("epoch-only: high bits are epoch (dest unpolluted)",
              extractDestEpochOnly(packMaskEpochOnly(7, 0, 916428u)), 7u);
    }

    printf("  RESULT: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
