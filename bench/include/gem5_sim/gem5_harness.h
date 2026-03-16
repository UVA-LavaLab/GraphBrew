// ============================================================================
// gem5 Graph Simulation Harness
// ============================================================================
//
// Provides macros and utilities for running graph benchmarks under gem5 SE mode.
// Unlike src_sim/ which uses an in-process cache simulator, gem5 benchmarks
// run natively — gem5's memory subsystem automatically tracks all accesses.
//
// Context passing: The benchmark writes a JSON sideband file with property
// region addresses and degree distribution. The gem5 replacement policy
// SimObjects lazily load this file on first eviction, getting the REAL
// addresses from within the simulated execution. This matches the original
// standalone approach (registerPropertyArray + initTopology) faithfully.
//
// Build WITHOUT m5ops (default — works natively and under gem5):
//   g++ -O1 -static -DNO_M5OPS src_gem5/pr.cc -o bin_gem5/pr
//
// Build WITH m5ops (enables ROI markers):
//   g++ -O1 -static -I$(GEM5)/include src_gem5/pr.cc -lm5 -o bin_gem5/pr
// ============================================================================

#ifndef GEM5_HARNESS_H_
#define GEM5_HARNESS_H_

#include <cstdint>
#include <cstdio>
#include <cstring>

#ifndef NO_M5OPS
#include <gem5/m5ops.h>
#define GEM5_RESET_STATS()  m5_reset_stats(0, 0)
#define GEM5_DUMP_STATS()   m5_dump_stats(0, 0)
#define GEM5_WORK_BEGIN(id) m5_work_begin(id, 0)
#define GEM5_WORK_END(id)   m5_work_end(id, 0)
#else
#define GEM5_RESET_STATS()  do {} while(0)
#define GEM5_DUMP_STATS()   do {} while(0)
#define GEM5_WORK_BEGIN(id) do {} while(0)
#define GEM5_WORK_END(id)   do {} while(0)
#endif

#define GEM5_WORK_INIT    0
#define GEM5_WORK_COMPUTE 1

// Default sideband file path — gem5 SE mode forwards file I/O to host
#define GEM5_SIDEBAND_PATH "/tmp/gem5_graphbrew_ctx.json"
#define GEM5_POPT_MATRIX_PATH "/tmp/gem5_popt_matrix.bin"

// ============================================================================
// GraphCacheContext exporter — writes sideband JSON for gem5 SimObjects
// ============================================================================
// Call this AFTER allocating property arrays and BEFORE the computation ROI.
// The replacement policy SimObjects will lazily load this file on first use.
//
// This mirrors what src_sim/ does with:
//   graph_ctx.registerPropertyArray(ptr, count, elem_size, llc_size);
//   graph_ctx.initTopology(degrees, num_nodes, num_edges, directed);
// ============================================================================

#include <graph.h>
#include <pvector.h>

// Maximum number of property regions we can export
#define GEM5_MAX_REGIONS 8

struct Gem5PropertyRegion {
    const char* name;
    uint64_t base_address;
    uint64_t size_bytes;
    uint32_t num_elements;
    uint32_t elem_size;
};

// Export graph cache context to sideband JSON file.
// Called by the benchmark after allocating all property arrays.
//
// Parameters:
//   regions:     Array of property region descriptors
//   num_regions: Number of regions
//   g:           Graph reference (for degree distribution)
//   path:        Output file path (default: /tmp/gem5_graphbrew_ctx.json)
template<typename GraphType>
inline void gem5_export_context(
    const Gem5PropertyRegion* regions, int num_regions,
    const GraphType& g,
    const char* path = GEM5_SIDEBAND_PATH)
{
    FILE* f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "gem5_harness: cannot write sideband to %s\n", path);
        return;
    }

    fprintf(f, "{\n");
    fprintf(f, "  \"num_vertices\": %ld,\n", (long)g.num_nodes());
    fprintf(f, "  \"num_edges\": %ld,\n", (long)g.num_edges_directed());
    fprintf(f, "  \"directed\": %s,\n", g.directed() ? "true" : "false");

    // Property regions
    fprintf(f, "  \"property_regions\": [\n");
    for (int i = 0; i < num_regions; i++) {
        fprintf(f, "    {\"name\": \"%s\", \"base\": %lu, \"size\": %lu, "
                "\"count\": %u, \"elem_size\": %u}%s\n",
                regions[i].name,
                (unsigned long)regions[i].base_address,
                (unsigned long)regions[i].size_bytes,
                regions[i].num_elements,
                regions[i].elem_size,
                (i < num_regions - 1) ? "," : "");
    }
    fprintf(f, "  ],\n");

    // Degree distribution (for GRASP bucket classification)
    // Compute degree histogram matching GraphTopology::NUM_BUCKETS = 11
    uint64_t total_edges = 0;
    uint32_t max_degree = 0;
    for (int64_t n = 0; n < g.num_nodes(); n++) {
        uint32_t d = static_cast<uint32_t>(g.out_degree(n));
        total_edges += d;
        if (d > max_degree) max_degree = d;
    }
    double avg_degree = (g.num_nodes() > 0) ?
        (double)total_edges / g.num_nodes() : 0.0;

    fprintf(f, "  \"avg_degree\": %.4f,\n", avg_degree);
    fprintf(f, "  \"max_degree\": %u,\n", max_degree);

    // 11-bucket logarithmic degree histogram (matching standalone)
    // Bucket boundaries: [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, inf)
    uint32_t bucket_counts[11] = {};
    uint64_t bucket_degrees[11] = {};
    for (int64_t n = 0; n < g.num_nodes(); n++) {
        uint32_t d = static_cast<uint32_t>(g.out_degree(n));
        int b = 0;
        if (d >= 512) b = 10;
        else if (d >= 256) b = 9;
        else if (d >= 128) b = 8;
        else if (d >= 64)  b = 7;
        else if (d >= 32)  b = 6;
        else if (d >= 16)  b = 5;
        else if (d >= 8)   b = 4;
        else if (d >= 4)   b = 3;
        else if (d >= 2)   b = 2;
        else if (d >= 1)   b = 1;
        else               b = 0;
        bucket_counts[b]++;
        bucket_degrees[b] += d;
    }

    fprintf(f, "  \"degree_buckets\": {\n");
    fprintf(f, "    \"counts\": [");
    for (int i = 0; i < 11; i++)
        fprintf(f, "%u%s", bucket_counts[i], i < 10 ? ", " : "");
    fprintf(f, "],\n");
    fprintf(f, "    \"total_degrees\": [");
    for (int i = 0; i < 11; i++)
        fprintf(f, "%lu%s", (unsigned long)bucket_degrees[i], i < 10 ? ", " : "");
    fprintf(f, "]\n");
    fprintf(f, "  }\n");

    fprintf(f, "}\n");
    fclose(f);

    printf("gem5_harness: exported context to %s "
           "(%ld vertices, %ld edges, %d regions)\n",
           path, (long)g.num_nodes(), (long)g.num_edges_directed(), num_regions);
}

// Print region info to stdout (for debugging)
inline void gem5_report_region(const char* name, const void* base,
                                size_t count, size_t elem_size) {
    printf("GRAPHBREW_REGION:%s:0x%lx:%lu:%lu\n",
           name, reinterpret_cast<uint64_t>(base), count, elem_size);
}

// Export P-OPT rereference matrix to binary file for gem5 SimObjects.
// Binary format: [num_epochs(4B)][num_cache_lines(4B)][epoch_size(4B)]
//                [sub_epoch_size(4B)][matrix data (num_epochs * num_cache_lines bytes)]
// This matches RereferenceMatrix::loadFromFile() in graph_cache_context_gem5.hh.
inline bool gem5_export_popt_matrix(
    const uint8_t* matrix_data,
    uint32_t num_cache_lines,
    uint32_t num_epochs,
    uint32_t num_vertices,
    uint32_t cache_line_size = 64,
    const char* path = GEM5_POPT_MATRIX_PATH)
{
    FILE* f = fopen(path, "wb");
    if (!f) return false;

    uint32_t epoch_size = (num_vertices + num_epochs - 1) / num_epochs;
    uint32_t sub_epoch_size = (epoch_size > 128) ? epoch_size / 128 : 1;

    fwrite(&num_epochs, 4, 1, f);
    fwrite(&num_cache_lines, 4, 1, f);
    fwrite(&epoch_size, 4, 1, f);
    fwrite(&sub_epoch_size, 4, 1, f);
    fwrite(matrix_data, 1, (size_t)num_epochs * num_cache_lines, f);
    fclose(f);

    printf("gem5_harness: exported P-OPT matrix to %s "
           "(%u epochs x %u lines = %lu bytes)\n",
           path, num_epochs, num_cache_lines,
           (unsigned long)num_epochs * num_cache_lines);
    return true;
}

#endif // GEM5_HARNESS_H_
