// Synthetic deterministic victim-selection test for the ECG_GRASP_POPT variants.
//
// This drives cache_sim's findVictimECG, which delegates the decision to the
// shared ecg_policy::selectVictim (bench/include/ecg_victim_policy.h) — the SAME
// function gem5 and Sniper call. So this test directly verifies the eviction
// decision for ALL THREE simulators (the SSOT test asserts the copies are
// byte-identical). It is mutation-proven: flipping the shared function's
// farthest->nearest epoch pick makes the epoch cases below FAIL.
//
// Unlike verify_ecg.py (which checks the LIVE trace against whatever set states a
// real run happens to produce), this builds CONTROLLED 8-way sets and asserts the
// EXACT victim, computed independently here — guaranteeing the epoch-property
// ranking branch is exercised and pinning the exact choice.
//
// One ECG_VARIANT per process: findVictimECG reads ECG_VARIANT once via a
// function-local static, so the harness runs this binary once per variant.
#include "cache_sim/cache_sim.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace cache_sim;

// Property region [PB,PU): a line whose addr falls inside is "property"; outside
// is a "record" (CSR edge-stream line). nv/ne with current_src=0 give curEpoch=0,
// so dist(i) == ecg_epoch(i) and "farthest next-reference" == "max epoch".
static const uint64_t PB = 0x10000ull, PU = 0x20000ull;
static uint64_t paddr(int k) { return PB + (uint64_t)k * 64; }       // property line
static uint64_t raddr(int k) { return 0x80000ull + (uint64_t)k * 64; } // record line

struct Way { uint64_t addr; int rrpv; int epoch; uint64_t last; int dbg; };

static GraphCacheContext g_ctx;
static void build_ctx() {
    g_ctx.num_regions = 1;
    g_ctx.regions[0].base_address = PB;
    g_ctx.regions[0].upper_bound = PU;
    g_ctx.regions[0].num_buckets = 1;
    g_ctx.regions[0].bucket_bounds[0] = PU;
    g_ctx.regions[0].region_id = 0;
    g_ctx.regions[0].elem_size = 64;
    g_ctx.exact_nv = 1024;            // nv
    g_ctx.edge_epoch_count = 32;      // ne; curEpoch = current_src*ne/nv
    g_ctx.mask_config.enabled = true;
    g_ctx.mask_config.ecg_mode = ECGMode::ECG_GRASP_POPT;
    g_ctx.hints_for_thread().current_src = 0;  // curEpoch = 0
}

static int g_pass = 0, g_fail = 0;
static void check(CacheLevel& L3, const char* name, std::vector<Way> w, int expected) {
    std::vector<CacheLine> set(8);
    for (int i = 0; i < 8; i++) {
        set[i].valid = true;
        set[i].tag = 1000 + i;
        set[i].line_addr = w[i].addr;
        set[i].rrpv = (uint8_t)w[i].rrpv;
        set[i].ecg_epoch = (uint16_t)w[i].epoch;
        set[i].last_access = w[i].last;
        set[i].ecg_dbg_tier = (uint8_t)w[i].dbg;
    }
    size_t v = L3.selectVictimForTest(set);
    bool ok = ((int)v == expected);
    printf("    %-46s expect=way%d got=way%zu  [%s]\n", name, expected, v, ok ? "OK" : "FAIL");
    if (ok) g_pass++; else g_fail++;
}

int main() {
    build_ctx();
    CacheLevel L3("L3", 16 * 1024, 64, 8, EvictionPolicy::ECG);
    L3.initGraphContext(&g_ctx);

    const char* ve = getenv("ECG_VARIANT");
    std::string var = ve ? ve : "rrip_first";
    printf("[test_ecg_victim] ECG_VARIANT=%s\n", var.c_str());

    // P=property line (addr in region), R=record line. epoch only meaningful for P.
    if (var == "rrip_first") {
        // max-RRPV set; records-first by recency, else farthest-epoch property.
        check(L3, "all-prop max-rrpv -> farthest epoch (way3=20)",
              {{paddr(0),7,3,0,0},{paddr(1),7,9,0,0},{paddr(2),7,1,0,0},{paddr(3),7,20,0,0},
               {paddr(4),7,7,0,0},{paddr(5),7,15,0,0},{paddr(6),7,2,0,0},{paddr(7),7,11,0,0}}, 3);
        check(L3, "mixed max-rrpv -> oldest record (way1 last=10)",
              {{raddr(0),7,0,50,0},{raddr(1),7,0,10,0},{paddr(2),7,5,30,0},{paddr(3),7,8,30,0},
               {paddr(4),7,2,30,0},{paddr(5),7,9,30,0},{paddr(6),7,1,30,0},{paddr(7),7,4,30,0}}, 1);
        check(L3, "sub-max records ignored -> farthest prop (way4=9)",
              {{raddr(0),3,0,5,0},{raddr(1),5,0,5,0},{paddr(2),7,4,0,0},{paddr(3),7,1,0,0},
               {paddr(4),7,9,0,0},{paddr(5),7,2,0,0},{paddr(6),7,6,0,0},{paddr(7),7,3,0,0}}, 4);
    } else if (var == "epoch_first" || var == "epoch_only") {
        // records first by recency (no rrpv gating), else farthest-epoch property.
        check(L3, "all-prop stamped -> farthest epoch (way3=20)",
              {{paddr(0),0,3,0,0},{paddr(1),0,9,0,0},{paddr(2),0,1,0,0},{paddr(3),0,20,0,0},
               {paddr(4),0,7,0,0},{paddr(5),0,15,0,0},{paddr(6),0,2,0,0},{paddr(7),0,11,0,0}}, 3);
        check(L3, "mixed -> oldest record by recency (way2 last=5)",
              {{raddr(0),0,0,50,0},{paddr(1),0,9,0,0},{raddr(2),0,0,5,0},{paddr(3),0,20,0,0},
               {paddr(4),0,7,0,0},{paddr(5),0,15,0,0},{paddr(6),0,2,0,0},{paddr(7),0,11,0,0}}, 2);
        check(L3, "unstamped(epoch=0) excluded -> farthest stamped (way3=20)",
              {{paddr(0),0,0,0,0},{paddr(1),0,9,0,0},{paddr(2),0,0,0,0},{paddr(3),0,20,0,0},
               {paddr(4),0,0,0,0},{paddr(5),0,0,0,0},{paddr(6),0,0,0,0},{paddr(7),0,0,0,0}}, 3);
    } else if (var == "shortcircuit") {
        // any non-property first (SET ORDER, not recency), else farthest-epoch + DBG.
        check(L3, "mixed -> FIRST record in set order (way1, not older way2)",
              {{paddr(0),0,3,0,0},{raddr(1),0,0,50,0},{raddr(2),0,0,5,0},{paddr(3),0,20,0,0},
               {paddr(4),0,7,0,0},{paddr(5),0,15,0,0},{paddr(6),0,2,0,0},{paddr(7),0,11,0,0}}, 1);
        check(L3, "all-prop -> farthest epoch (way3=20)",
              {{paddr(0),0,3,0,0},{paddr(1),0,9,0,0},{paddr(2),0,1,0,0},{paddr(3),0,20,0,0},
               {paddr(4),0,7,0,0},{paddr(5),0,15,0,0},{paddr(6),0,2,0,0},{paddr(7),0,11,0,0}}, 3);
        check(L3, "all-prop epoch tie -> DBG tiebreak (way2 dbg=5)",
              {{paddr(0),0,10,0,0},{paddr(1),0,10,0,0},{paddr(2),0,10,0,5},{paddr(3),0,10,0,0},
               {paddr(4),0,10,0,2},{paddr(5),0,10,0,0},{paddr(6),0,10,0,0},{paddr(7),0,10,0,0}}, 2);
    } else if (var == "grasp_only") {
        // pure RRIP: first line at max RRPV (epoch/property irrelevant), aging if none.
        check(L3, "first max-rrpv ignores epoch (way1)",
              {{paddr(0),3,20,0,0},{raddr(1),7,0,0,0},{paddr(2),7,9,0,0},{paddr(3),5,0,0,0},
               {paddr(4),7,1,0,0},{paddr(5),0,0,0,0},{paddr(6),7,0,0,0},{paddr(7),2,0,0,0}}, 1);
        check(L3, "aging to max-rrpv (way6 reaches 7 first)",
              {{paddr(0),3,0,0,0},{paddr(1),5,0,0,0},{paddr(2),2,0,0,0},{paddr(3),4,0,0,0},
               {paddr(4),1,0,0,0},{paddr(5),0,0,0,0},{raddr(6),6,0,0,0},{paddr(7),2,0,0,0}}, 6);
    } else {
        printf("  (no scenarios for variant '%s')\n", var.c_str());
        return 2;
    }

    printf("  RESULT[%s]: %d passed, %d failed\n", var.c_str(), g_pass, g_fail);
    return g_fail ? 1 : 0;
}
