// ============================================================================
// gem5 Graph Simulation Harness
// ============================================================================
//
// Provides macros for running graph benchmarks under gem5 SE mode.
// Unlike src_sim/ which uses an in-process cache simulator, gem5 benchmarks
// run natively — gem5's memory subsystem automatically tracks all accesses.
//
// This harness provides:
//   1. m5ops integration for ROI (Region of Interest) marking
//   2. Stats reset/dump around the computation phase
//   3. Property region reporting for cache policy debugging
//   4. Single-threaded execution helpers
//
// Build WITH m5ops (enables ROI markers — requires gem5 m5ops library):
//   g++ -O1 -static -I$(GEM5)/include -L$(GEM5)/util/m5/build/x86/out
//       src_gem5/pr.cc -lm5 -o bin_gem5/pr
//
// Build WITHOUT m5ops (works natively and under gem5, just no ROI markers):
//   g++ -O1 -static -DNO_M5OPS src_gem5/pr.cc -o bin_gem5/pr
// ============================================================================

#ifndef GEM5_HARNESS_H_
#define GEM5_HARNESS_H_

#include <cstdint>
#include <cstdio>

#ifndef NO_M5OPS
#include <gem5/m5ops.h>
#define GEM5_RESET_STATS()  m5_reset_stats(0, 0)
#define GEM5_DUMP_STATS()   m5_dump_stats(0, 0)
#define GEM5_WORK_BEGIN(id) m5_work_begin(id, 0)
#define GEM5_WORK_END(id)   m5_work_end(id, 0)
#else
// No-op stubs when building without m5ops (for native testing)
#define GEM5_RESET_STATS()  do {} while(0)
#define GEM5_DUMP_STATS()   do {} while(0)
#define GEM5_WORK_BEGIN(id) do {} while(0)
#define GEM5_WORK_END(id)   do {} while(0)
#endif

// Work IDs for different phases
#define GEM5_WORK_INIT    0
#define GEM5_WORK_COMPUTE 1

// Print property region info for gem5 cache policy debugging.
// The online region-learning in GRASP/ECG SimObjects handles classification
// automatically from access patterns. This output helps with validation.
inline void gem5_report_region(const char* name, const void* base,
                                size_t count, size_t elem_size) {
    printf("GRAPHBREW_REGION:%s:0x%lx:%lu:%lu\n",
           name,
           reinterpret_cast<uint64_t>(base),
           count, elem_size);
}

#endif // GEM5_HARNESS_H_
