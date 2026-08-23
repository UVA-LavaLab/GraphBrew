// Direct proof: every compact record must decode to exactly the same
// (dest, tier, first, second) as the 64-bit record built from the same graph.
// This is far stronger than comparing kernel outputs, which cannot detect a
// wrong epoch at all.
#include <cstdio>
#include <cstdint>
#include <vector>
#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "ecg_epoch_builder.h"

int main(int argc, char* argv[]) {
    CLBase cli(argc, argv, "epoch-pair-equivalence");
    if (!cli.ParseArgs()) return 1;
    WeightedBuilder bw(cli);
    Builder b(cli);
    Graph g = b.MakeGraph();
    const uint32_t n = static_cast<uint32_t>(g.num_nodes());

    for (uint32_t ne : {8u, 16u, 32u, 64u}) {
        std::vector<uint64_t> off64, rec64;
        std::vector<uint64_t> off32;
        std::vector<uint32_t> rec32;
        ecg_epoch::buildInEdgeEpochPairRecords(g, 16, ne, true, off64, rec64);
        if (!ecg_epoch::epochPairOffsetsMatchInCsr(
                g, off64, static_cast<uint64_t>(rec64.size()))) {
            printf("ne=%u: WIDE CSR OFFSET MISMATCH\n", ne);
            return 1;
        }
        bool ok32 = ecg_epoch::buildInEdgeEpochPairRecords32(
            g, 16, ne, true, off32, rec32);
        if (!ok32) { printf("ne=%u: compact refused (expected when fields do not fit)\n", ne); continue; }
        if (!ecg_epoch::epochPairOffsetsMatchInCsr(
                g, off32, static_cast<uint64_t>(rec32.size()))) {
            printf("ne=%u: COMPACT CSR OFFSET MISMATCH\n", ne);
            return 1;
        }
        if (off64 != off32) { printf("ne=%u: OFFSET MISMATCH\n", ne); return 1; }
        if (rec64.size() != rec32.size()) { printf("ne=%u: SIZE MISMATCH\n", ne); return 1; }
        const uint32_t idb = ecg_epoch::epochPair32IdBits(n);
        const uint32_t epb = ecg_epoch::epochPair32EpochBits(ne);
        uint64_t bad = 0, checked = 0, csr_checked = 0;
        for (uint32_t u = 0; u < n; ++u) {
            uint64_t pos = off64[u];
            const uint64_t end = off64[u + 1];
            for (auto v_raw : g.in_neigh(u)) {
                const uint32_t v = static_cast<uint32_t>(v_raw);
                if (pos >= end ||
                    ecg_epoch::extractEpochPairDest(rec64[pos]) != v ||
                    ecg_epoch::extractEpochPair32Dest(rec32[pos], idb) != v) {
                    printf("ne=%u: CSR DESTINATION MISMATCH row=%u pos=%llu\n",
                           ne, u, (unsigned long long)pos);
                    return 1;
                }
                ++pos;
                ++csr_checked;
            }
            if (pos != end) {
                printf("ne=%u: CSR ROW LENGTH MISMATCH row=%u\n", ne, u);
                return 1;
            }
        }
        if (csr_checked != rec64.size()) {
            printf("ne=%u: CSR RECORD COUNT MISMATCH\n", ne);
            return 1;
        }
        for (size_t i = 0; i < rec64.size(); ++i) {
            const uint64_t w = ecg_epoch::widenEpochPair32(rec32[i], idb, epb);
            if (w != rec64[i]) {
                if (bad < 3)
                    printf("  edge %zu: 64b=%016llx widened=%016llx\n",
                           i, (unsigned long long)rec64[i], (unsigned long long)w);
                bad++;
            }
            checked++;
        }
        printf("ne=%2u: %llu records checked, %llu mismatches  id_bits=%u epoch_bits=%u\n",
               ne, (unsigned long long)checked, (unsigned long long)bad, idb, epb);
        if (bad) return 1;
    }
    printf("CSR OFFSETS AND DESTINATIONS MATCH\n");
    printf("ALL EQUIVALENT\n");
    return 0;
}
