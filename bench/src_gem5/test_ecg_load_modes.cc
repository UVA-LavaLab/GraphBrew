// Runtime gem5-DECODER test for the consolidated ecg.load (custom-0, FUNCT3=0x2).
//
// The field-parity test (test_ecg_packed_field_parity.cc) pins the layout against a C++
// MIRROR of the decoder shifts; the 3-sim verify exercises ECG eviction via the X86 m5op
// path. NEITHER runs the actual gem5-decoded ecg.load. This test does: it issues EVERY
// (mode, width) variant through the REAL RISC-V decoder and checks the decoded dest via
// rd = prop[dest] (prop[i] = i). A wrong ECG_MODE dispatch or wrong ECG_WIDTH (W) extracts
// a different dest -> rd != dest -> caught. Run under gem5 RISCV (the real decoder); the
// host build is a no-op stub (the emitters dereference the record, so it still links).
#include "gem5_sim/gem5_harness.h"
#include "ecg_mode6_builder.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

static int g_fail = 0;
alignas(64) static uint64_t g_k2_record =
    (static_cast<uint64_t>(0xACE0u) << 48) |
    (static_cast<uint64_t>(0x2468u) << 32) |
    0x12345678u;

// TEETH PROOF: ECG_TEST_FORCE_WC forces the EMITTED width class (FUNCT7) to a fixed value
// while the record is still packed with the CORRECT wc. If the gem5 decoder truly reads
// ECG_WIDTH (not a hardcoded W), a forced-wrong emit must decode a DIFFERENT dest -> the
// test FAILs. Unset (the normal run) => emit the correct wc => PASS. This proves the test
// is not vacuous: the decoder's ECG_WIDTH handling is load-bearing.
static int emit_wc(int correct_wc) {
    const char* f = std::getenv("ECG_TEST_FORCE_WC");
    return f ? std::atoi(f) : correct_wc;
}

static void check(const char* mode, int wc, uint32_t dest, uint32_t rd) {
    bool ok = (rd == dest);
    printf("[test_ecg_load_modes] %-10s wc=%2d dest=%-10u rd=%-10u [%s]\n",
           mode, wc, dest, rd, ok ? "OK" : "FAIL");
    if (!ok) g_fail++;
}

int main() {
    // 4M-entry property array (16 MB). prop[dest] = dest, so a correctly decoded dest
    // returns rd == dest; a wrong width/mode lands on a different (unwritten => 0) index.
    const size_t N = (size_t)4u << 20;
    uint32_t* prop = static_cast<uint32_t*>(std::calloc(N, sizeof(uint32_t)));
    if (!prop) { printf("[test_ecg_load_modes] calloc failed\n"); return 2; }

    // dest values: valid for the width class AND large enough that a too-small W would
    // mask off high bits to a DIFFERENT (in-array) index -> clean mismatch.
    struct { int wc; uint32_t dest; uint16_t epoch; } vec[] = {
        {0, 0x000000FEu, 0xBEEF},  // W8  (dest < 256;   wrong W16 -> 0xEFFE)
        {1, 0x0000BEEFu, 0x000D},  // W16 (dest < 65536; wrong W8 -> 0xEF,  wrong W24 -> 0x0DBEEF)
        {2, 0x003ABCDEu, 0x000D},  // W24 (dest < 16.7M; wrong W16 -> 0xBCDE)
        {3, 0x003ABCDEu, 0x000D},  // W32 (representable; full 32-bit range pinned by field-parity)
    };

    for (auto& c : vec) {
        prop[c.dest] = c.dest;
        uint64_t rec = ecg_mode6::packEvict(c.dest, c.epoch, c.wc);
        check("EVICT", c.wc, c.dest, gem5_ecg_load_evict(prop, rec, emit_wc(c.wc)));
    }
    for (auto& c : vec) {
        prop[c.dest] = c.dest;
        uint64_t rec = ecg_mode6::packEvictPfx(c.dest, c.epoch, 0x5A5Au, c.wc);
        check("EVICT+PFX", c.wc, c.dest, gem5_ecg_load_pfx(prop, rec, emit_wc(c.wc)));
    }
    {
        // EMBEDDED (mode 2): NARROW packMaskEpoch, fixed 24-bit dest + dbg/popt/pfx.
        uint32_t dest = 0x003ABCDEu;
        prop[dest] = dest;
        uint64_t rec = ecg_mode6::packMaskEpoch(dest, 2, 0x5A, 0x1234, 0x7F);
        check("EMBEDDED", 24, dest, gem5_ecg_load_embedded(prop, rec));
    }
    {
        constexpr uint64_t record =
            (static_cast<uint64_t>(0xACE0u) << 48) |
            (static_cast<uint64_t>(0x2468u) << 32) |
            0x12345678u;
        uint64_t rd_stream = gem5_ecg_stream_load2_instruction(&g_k2_record);
        uint64_t rd = gem5_ecg_load2_instruction(&g_k2_record);
        bool ok = rd == record && rd_stream == record;
        printf("[test_ecg_load_modes] LOAD2/K2 record=%#llx rd=%#llx "
               "stream=%#llx [%s]\n",
               (unsigned long long)record, (unsigned long long)rd,
               (unsigned long long)rd_stream,
               ok ? "OK" : "FAIL");
        if (!ok) g_fail++;
    }

    printf("[test_ecg_load_modes] RESULT: %s (%d fail)\n",
           g_fail ? "FAIL" : "PASS", g_fail);
    std::free(prop);
    return g_fail ? 1 : 0;
}
