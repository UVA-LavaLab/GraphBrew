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
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>

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
#define GEM5_WORK_SET_VERTEX 0x47525654ULL  // GraphBrew current vertex hint
#define GEM5_WORK_ECG_PFX_TARGET 0x47504658ULL  // GraphBrew ECG PFX target hint

#ifndef NO_M5OPS
inline bool gem5_vertex_hints_enabled() {
    static int enabled = []() {
        const char* value = std::getenv("GEM5_ENABLE_VERTEX_HINTS");
        return (value && std::strcmp(value, "0") != 0) ? 1 : 0;
    }();
    return enabled != 0;
}

#define GEM5_SET_VERTEX(vertex_id) \
    do { \
        if (gem5_vertex_hints_enabled()) { \
            m5_work_begin(GEM5_WORK_SET_VERTEX, static_cast<uint64_t>(vertex_id)); \
        } \
    } while (0)

inline bool gem5_ecg_pfx_hints_enabled() {
    static int enabled = []() {
        const char* value = std::getenv("GEM5_ENABLE_ECG_PFX_HINTS");
        return (value && std::strcmp(value, "0") != 0) ? 1 : 0;
    }();
    return enabled != 0;
}

inline bool gem5_ecg_extract_enabled() {
    static int enabled = []() {
        const char* value = std::getenv("GEM5_ENABLE_ECG_EXTRACT");
        return (value && std::strcmp(value, "0") != 0) ? 1 : 0;
    }();
    return enabled != 0;
}

inline int gem5_ecg_pfx_hint_filter_capacity() {
    static int capacity = []() {
        const char* value = std::getenv("GEM5_ECG_PFX_HINT_FILTER");
        if (!value || !value[0]) return 16;
        int parsed = std::atoi(value);
        if (parsed < 0) return 0;
        if (parsed > 64) return 64;
        return parsed;
    }();
    return capacity;
}

inline bool gem5_should_emit_ecg_pfx_hint(uint64_t vertex_id) {
    int capacity = gem5_ecg_pfx_hint_filter_capacity();
    if (capacity == 0) return true;
    auto env_int = [](const char* name, int default_value, int min_value, int max_value) {
        const char* value = std::getenv(name);
        if (!value || !value[0]) return default_value;
        int parsed = std::atoi(value);
        if (parsed < min_value) return min_value;
        if (parsed > max_value) return max_value;
        return parsed;
    };
    int elem_size = env_int("GEM5_ECG_PFX_FILTER_ELEM_SIZE", 4, 1, 64);
    int line_size = env_int("GEM5_ECG_PFX_FILTER_LINE_SIZE", 64, 1, 4096);
    uint64_t vertices_per_line = static_cast<uint64_t>(line_size / elem_size);
    if (vertices_per_line == 0) vertices_per_line = 1;
    uint64_t filter_key = vertex_id / vertices_per_line;
    thread_local uint64_t recent[64] = {};
    thread_local int count = 0;
    thread_local int next = 0;
    for (int i = 0; i < count; ++i) {
        if (recent[i] == filter_key) return false;
    }
    recent[next] = filter_key;
    next = (next + 1) % capacity;
    if (count < capacity) ++count;
    return true;
}

inline uint32_t gem5_ecg_extract_target_instruction(uint32_t target_vertex) {
#if defined(__riscv)
    uint64_t fat_id = static_cast<uint64_t>(target_vertex);
    uint64_t real_vertex = 0;
    asm volatile (".insn r 0x0b, 0x0, 0x00, %0, %1, x0"
                  : "=r"(real_vertex)
                  : "r"(fat_id)
                  : "memory");
    return static_cast<uint32_t>(real_vertex);
#else
    return target_vertex;
#endif
}

inline bool gem5_x86_instruction_m5ops_available() {
#if defined(__x86_64__)
    return true;
#else
    return false;
#endif
}

inline void gem5_x86_work_begin_instruction(uint64_t work_id, uint64_t argument) {
#if defined(__x86_64__)
    asm volatile (".byte 0x0F, 0x04\n\t.word %c0"
                  :
                  : "i"(M5OP_WORK_BEGIN), "D"(work_id), "S"(argument)
                  : "rax", "memory");
#else
    m5_work_begin(work_id, argument);
#endif
}

inline uint32_t gem5_ecg_pfx_target_instruction(uint32_t target_vertex) {
#if defined(__riscv)
    return gem5_ecg_extract_target_instruction(target_vertex);
#elif defined(__x86_64__)
    gem5_x86_work_begin_instruction(GEM5_WORK_ECG_PFX_TARGET, static_cast<uint64_t>(target_vertex));
    return target_vertex;
#else
    return target_vertex;
#endif
}

// === S69PRE-M1-MASK: full mode-6 mask emission via ecg.extract ===
//
// The S69-pre M1 wiring extends RISCV ecg.extract delivery from
// just the prefetch target to the full 64-bit per-edge mode-6 mask
// (dest + DBG + POPT + PFX). The decoder unpacks all 4 fields and
// populates the per-vertex ECG metadata table; ECG_RP consumes that
// table during replacement decisions.
//
// On X86 the work_begin path can only carry one uint64_t threadid
// argument, which happens to be exactly the 64-bit mask. The
// pseudo_inst handler treats the threadid as a packed mask when the
// new work_id is used. (Backward-compat: GEM5_WORK_ECG_PFX_TARGET
// continues to deliver a bare vertex via setPrefetchTargetHint.)

inline uint32_t gem5_ecg_extract_mask_instruction(uint64_t fat_mask) {
#if defined(__riscv)
    uint64_t real_vertex = 0;
    asm volatile (".insn r 0x0b, 0x0, 0x00, %0, %1, x0"
                  : "=r"(real_vertex)
                  : "r"(fat_mask)
                  : "memory");
    return static_cast<uint32_t>(real_vertex);
#elif defined(__x86_64__)
    // X86 fallback: deliver just the prefetch target (high 31 bits of mask).
    // ECG_RP metadata channel is RISCV-only on X86 today.
    uint32_t pfx_target = static_cast<uint32_t>((fat_mask >> 33) & 0x7FFFFFFFULL);
    if (pfx_target == 0) {
        pfx_target = static_cast<uint32_t>(fat_mask & 0xFFFFFFULL);  // dest
    }
    gem5_x86_work_begin_instruction(GEM5_WORK_ECG_PFX_TARGET,
                                    static_cast<uint64_t>(pfx_target));
    return pfx_target;
#else
    return static_cast<uint32_t>(fat_mask & 0xFFFFFFULL);  // dest
#endif
}

// GEM5_ECG_EXTRACT_MASK(mask_u64): emit the full mode-6 mask via the
// RISCV ecg.extract opcode (or the X86 fallback). Bypasses the dedup
// filter — caller is responsible for not over-emitting.
#define GEM5_ECG_EXTRACT_MASK(mask_u64) \
    do { \
        if (gem5_ecg_pfx_hints_enabled() && gem5_ecg_extract_enabled()) { \
            (void)gem5_ecg_extract_mask_instruction(static_cast<uint64_t>(mask_u64)); \
        } \
    } while (0)


#if defined(__riscv)
#define GEM5_ECG_PFX_TARGET(vertex_id) \
    do { \
        if (gem5_ecg_pfx_hints_enabled()) { \
            uint64_t _gem5_pfx_vertex = static_cast<uint64_t>(vertex_id); \
            if (gem5_should_emit_ecg_pfx_hint(_gem5_pfx_vertex)) { \
                if (gem5_ecg_extract_enabled()) { \
                    (void)gem5_ecg_pfx_target_instruction(static_cast<uint32_t>(_gem5_pfx_vertex)); \
                } else { \
                    m5_work_begin(GEM5_WORK_ECG_PFX_TARGET, _gem5_pfx_vertex); \
                } \
            } \
        } \
    } while (0)
#else
#define GEM5_ECG_PFX_TARGET(vertex_id) \
    do { \
        if (gem5_ecg_pfx_hints_enabled()) { \
            uint64_t _gem5_pfx_vertex = static_cast<uint64_t>(vertex_id); \
            if (gem5_should_emit_ecg_pfx_hint(_gem5_pfx_vertex)) { \
                if (gem5_ecg_extract_enabled() && gem5_x86_instruction_m5ops_available()) { \
                    (void)gem5_ecg_pfx_target_instruction(static_cast<uint32_t>(_gem5_pfx_vertex)); \
                } else { \
                    m5_work_begin(GEM5_WORK_ECG_PFX_TARGET, _gem5_pfx_vertex); \
                } \
            } \
        } \
    } while (0)
#endif
#else
#define GEM5_SET_VERTEX(vertex_id) do {} while(0)
#if defined(__riscv)
inline bool gem5_ecg_pfx_hints_enabled() {
    static int enabled = []() {
        const char* value = std::getenv("GEM5_ENABLE_ECG_PFX_HINTS");
        return (value && std::strcmp(value, "0") != 0) ? 1 : 0;
    }();
    return enabled != 0;
}

inline bool gem5_ecg_extract_enabled() {
    static int enabled = []() {
        const char* value = std::getenv("GEM5_ENABLE_ECG_EXTRACT");
        return (value && std::strcmp(value, "0") != 0) ? 1 : 0;
    }();
    return enabled != 0;
}

inline int gem5_ecg_pfx_hint_filter_capacity() {
    static int capacity = []() {
        const char* value = std::getenv("GEM5_ECG_PFX_HINT_FILTER");
        if (!value || !value[0]) return 16;
        int parsed = std::atoi(value);
        if (parsed < 0) return 0;
        if (parsed > 64) return 64;
        return parsed;
    }();
    return capacity;
}

inline bool gem5_should_emit_ecg_pfx_hint(uint64_t vertex_id) {
    int capacity = gem5_ecg_pfx_hint_filter_capacity();
    if (capacity == 0) return true;
    auto env_int = [](const char* name, int default_value, int min_value, int max_value) {
        const char* value = std::getenv(name);
        if (!value || !value[0]) return default_value;
        int parsed = std::atoi(value);
        if (parsed < min_value) return min_value;
        if (parsed > max_value) return max_value;
        return parsed;
    };
    int elem_size = env_int("GEM5_ECG_PFX_FILTER_ELEM_SIZE", 4, 1, 64);
    int line_size = env_int("GEM5_ECG_PFX_FILTER_LINE_SIZE", 64, 1, 4096);
    uint64_t vertices_per_line = static_cast<uint64_t>(line_size / elem_size);
    if (vertices_per_line == 0) vertices_per_line = 1;
    uint64_t filter_key = vertex_id / vertices_per_line;
    thread_local uint64_t recent[64] = {};
    thread_local int count = 0;
    thread_local int next = 0;
    for (int i = 0; i < count; ++i) {
        if (recent[i] == filter_key) return false;
    }
    recent[next] = filter_key;
    next = (next + 1) % capacity;
    if (count < capacity) ++count;
    return true;
}

inline uint32_t gem5_ecg_extract_target_instruction(uint32_t target_vertex) {
    uint64_t fat_id = static_cast<uint64_t>(target_vertex);
    uint64_t real_vertex = 0;
    asm volatile (".insn r 0x0b, 0x0, 0x00, %0, %1, x0"
                  : "=r"(real_vertex)
                  : "r"(fat_id)
                  : "memory");
    return static_cast<uint32_t>(real_vertex);
}

inline uint32_t gem5_ecg_pfx_target_instruction(uint32_t target_vertex) {
    return gem5_ecg_extract_target_instruction(target_vertex);
}

#define GEM5_ECG_PFX_TARGET(vertex_id) \
    do { \
        if (gem5_ecg_pfx_hints_enabled() && gem5_ecg_extract_enabled()) { \
            uint64_t _gem5_pfx_vertex = static_cast<uint64_t>(vertex_id); \
            if (gem5_should_emit_ecg_pfx_hint(_gem5_pfx_vertex)) { \
                (void)gem5_ecg_pfx_target_instruction(static_cast<uint32_t>(_gem5_pfx_vertex)); \
            } \
        } \
    } while (0)
#else
#define GEM5_ECG_PFX_TARGET(vertex_id) do {} while(0)
#endif
#endif

inline const char* gem5_env_or_default(const char* name, const char* fallback) {
    const char* value = std::getenv(name);
    return value && value[0] ? value : fallback;
}

inline int gem5_env_int_clamped(const char* name, int default_value,
                                int min_value, int max_value) {
    const char* value = std::getenv(name);
    if (!value || !value[0]) return default_value;
    int parsed = std::atoi(value);
    if (parsed < min_value) return min_value;
    if (parsed > max_value) return max_value;
    return parsed;
}

inline const char* gem5_context_path() {
    return gem5_env_or_default("GEM5_GRAPHBREW_CTX", "/tmp/gem5_graphbrew_ctx.json");
}

inline const char* gem5_popt_matrix_path() {
    return gem5_env_or_default("GEM5_POPT_MATRIX", "/tmp/gem5_popt_matrix.bin");
}

inline const char* gem5_out_edges_path() {
    return gem5_env_or_default("GEM5_GRAPHBREW_OUT_EDGES", "/tmp/gem5_graphbrew_out_edges.bin");
}

inline const char* gem5_in_edges_path() {
    return gem5_env_or_default("GEM5_GRAPHBREW_IN_EDGES", "/tmp/gem5_graphbrew_in_edges.bin");
}

// Default sideband file paths — gem5 SE mode forwards file I/O to host.
// Runner-launched jobs can override these with per-run environment variables.
#define GEM5_SIDEBAND_PATH gem5_context_path()
#define GEM5_POPT_MATRIX_PATH gem5_popt_matrix_path()

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
    bool grasp_region = true;
};

struct Gem5EdgeRegion {
    const char* name;
    uint64_t base_address;
    uint64_t size_bytes;
    uint32_t elem_size;
    const void* data = nullptr;
    const char* data_path = nullptr;
};

template<typename GraphType>
inline int gem5_make_edge_regions(const GraphType& g,
                                  Gem5EdgeRegion* edge_regions,
                                  int max_edge_regions,
                                  bool prefer_in_edges = false)
{
    if (max_edge_regions < 2 || g.num_nodes() == 0 ||
        g.num_edges_directed() == 0) {
        return 0;
    }

    auto out0 = g.out_neigh(0);
    auto in0 = g.in_neigh(0);
    const void* out_base = static_cast<const void*>(out0.begin());
    const void* in_base = static_cast<const void*>(in0.begin());
    const uint64_t edge_count = static_cast<uint64_t>(g.num_edges_directed());
    const uint32_t out_elem_size = static_cast<uint32_t>(sizeof(*out0.begin()));
    const uint32_t in_elem_size = static_cast<uint32_t>(sizeof(*in0.begin()));

    Gem5EdgeRegion out_region = {
        "out_edges", reinterpret_cast<uint64_t>(out_base),
        edge_count * out_elem_size, out_elem_size, out_base
    };
    Gem5EdgeRegion in_region = {
        "in_edges", reinterpret_cast<uint64_t>(in_base),
        edge_count * in_elem_size, in_elem_size, in_base
    };

    edge_regions[0] = prefer_in_edges ? in_region : out_region;
    edge_regions[1] = prefer_in_edges ? out_region : in_region;
    return 2;
}

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
    const char* path = GEM5_SIDEBAND_PATH,
    const Gem5EdgeRegion* edge_regions = nullptr,
    int num_edge_regions = 0)
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
            "\"count\": %u, \"elem_size\": %u, \"grasp\": %s}%s\n",
                regions[i].name,
                (unsigned long)regions[i].base_address,
                (unsigned long)regions[i].size_bytes,
                regions[i].num_elements,
                regions[i].elem_size,
            regions[i].grasp_region ? "true" : "false",
                (i < num_regions - 1) ? "," : "");
    }
    fprintf(f, "  ],\n");

    // Edge regions for graph prefetchers such as DROPLET. These are runtime
    // addresses of CSR neighbor arrays inside the simulated process.
    fprintf(f, "  \"edge_regions\": [\n");
    for (int i = 0; i < num_edge_regions; i++) {
        const char* data_path = edge_regions[i].data_path;
        if (!data_path && edge_regions[i].data && edge_regions[i].size_bytes > 0) {
            if (std::strcmp(edge_regions[i].name, "out_edges") == 0) {
                data_path = gem5_out_edges_path();
            } else if (std::strcmp(edge_regions[i].name, "in_edges") == 0) {
                data_path = gem5_in_edges_path();
            }
        }
        if (data_path && edge_regions[i].data && edge_regions[i].size_bytes > 0) {
            FILE* ef = fopen(data_path, "wb");
            if (ef) {
                fwrite(edge_regions[i].data, 1,
                       static_cast<size_t>(edge_regions[i].size_bytes), ef);
                fclose(ef);
            } else {
                fprintf(stderr, "gem5_harness: cannot write edge data to %s\n",
                        data_path);
                data_path = nullptr;
            }
        }
        fprintf(f, "    {\"name\": \"%s\", \"base\": %lu, \"size\": %lu, "
            "\"elem_size\": %u, \"preferred\": %s",
                edge_regions[i].name,
                (unsigned long)edge_regions[i].base_address,
                (unsigned long)edge_regions[i].size_bytes,
            edge_regions[i].elem_size,
            (i == 0) ? "true" : "false");
        if (data_path) {
            fprintf(f, ", \"data_path\": \"%s\"", data_path);
        }
        fprintf(f, "}%s\n",
                (i < num_edge_regions - 1) ? "," : "");
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
    uint32_t sub_epoch_size = (epoch_size + 127) / 128;
    if (sub_epoch_size == 0) sub_epoch_size = 1;

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
