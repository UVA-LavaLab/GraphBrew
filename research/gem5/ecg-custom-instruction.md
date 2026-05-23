# ECG Custom Instruction — RISC-V and x86 Integration

## Overview

ECG encodes graph-aware cache hints into the upper bits of CSR neighbor vertex IDs
(fat-ID encoding). A custom instruction is needed to:
1. Extract the real vertex ID (lower bits) for the algorithm to use
2. Write the mask (DBG tier + P-OPT hint + prefetch distance) to a CSR that
   the cache controller reads on subsequent memory accesses

This document covers the RISC-V custom instruction approach (cleaner encoding)
and the x86 pseudo-instruction fallback.

## RISC-V: `ecg.extract` Instruction

### Encoding

Uses RISC-V **custom-0** opcode space (`0x0B`), R-type format:

```
31      25   24  20   19  15   14  12   11   7   6    0
┌────────┬───────┬───────┬──────┬───────┬───────┐
│ funct7 │  rs2  │  rs1  │funct3│  rd   │opcode │
│0000000 │00000  │ src   │ 000  │ dst   │0001011│
└────────┴───────┴───────┴──────┴───────┴───────┘
```

**Semantics**: `ecg.extract rd, rs1`
- `rs1`: Fat-ID encoded neighbor value (loaded from CSR edge array)
- `rd`: Cleaned vertex ID (lower bits, mask stripped)
- **Side effect**: Writes extracted mask to CSR `0x800` (`mecg_mask`)

### Mask Extraction

The mask width and bit layout are configured via a second CSR (`0x801`,
`mecg_config`) that is written once at kernel start:

```
CSR 0x801 (mecg_config):
  bits[7:0]  = mask_width (8, 16, 32)
  bits[15:8] = dbg_bits (2-4)
  bits[23:16] = popt_bits (0-8)
  bits[31:24] = prefetch_bits (remaining)
```

The `ecg.extract` instruction uses this config to determine how many upper bits
to strip and what to write to the mask CSR:

```
vertex_id = rs1 & ((1 << (64 - mask_width)) - 1)
mask      = rs1 >> (64 - mask_width)

CSR[0x800] = mask  // Available to cache controller
rd         = vertex_id
```

### Feasibility in gem5

gem5's RISC-V ISA decoder supports custom instructions via:
1. Adding a decoder entry in `src/arch/riscv/isa/decoder.isa`
2. Defining the execution semantics in a new `.isa` file
3. The custom CSR is added to the RISC-V CSR map

The instruction is decoded when opcode matches `0x0B` and funct3/funct7 match
our encoding. The execution writes the mask to the CSR and the vertex ID to `rd`.

### gem5 Decoder Patch

```
// In decoder.isa, under custom-0 opcode (0x0B):
0x0B: decode FUNCT3 {
    0x0: decode FUNCT7 {
        0x00: EcgExtract::ecg_extract({{
            uint64_t fat_id = Rs1;
            uint64_t mask_width = xc->readMiscReg(MISCREG_MECG_CONFIG) & 0xFF;
            uint64_t vertex_id = fat_id & ((1ULL << (64 - mask_width)) - 1);
            uint64_t mask = fat_id >> (64 - mask_width);
            xc->setMiscReg(MISCREG_MECG_MASK, mask);
            Rd = vertex_id;
        }});
    }
}
```

### Cache Controller Integration

The L3 cache controller reads `CSR[0x800]` when ECG replacement policy is active:

```cpp
// In cache controller, on each access:
if (replacementPolicy->isType<GraphEcgRP>()) {
    uint8_t mask = cpu->readMiscReg(MISCREG_MECG_MASK);
    ecgRP->setCurrentMask(mask);
}
```

This avoids modifying the standard load/store path — the custom instruction
writes hints "ahead" of the actual memory access.

## x86: m5 Pseudo-Instruction Fallback

For x86 ISA (native host compilation, easier development):

### m5ops Approach

gem5 provides a pseudo-instruction mechanism via `m5ops.h`:

```cpp
#include "gem5/m5ops.h"

// In benchmark inner loop:
uint64_t fat_id = edge_list[i];
uint64_t vertex_id = m5_ecg_extract(fat_id);  // Custom pseudo-op

// m5_ecg_extract implementation (in gem5):
//   1. Strips mask from fat_id using configured mask_width
//   2. Stores mask in internal register (accessible to cache controller)
//   3. Returns clean vertex_id
```

### m5ops Registration

Add to gem5's `src/sim/pseudo_inst.cc`:

```cpp
void pseudoInst::ecgExtract(ThreadContext *tc, uint64_t fat_id) {
    uint64_t config = /* read config register */;
    uint64_t mask_width = config & 0xFF;
    uint64_t vertex_id = fat_id & ((1ULL << (64 - mask_width)) - 1);
    uint64_t mask = fat_id >> (64 - mask_width);

    // Store mask for cache controller
    tc->setMiscReg(MISCREG_ECG_MASK, mask);

    // Return vertex_id via register
    tc->setReg(RegId(IntRegClass, ReturnValueReg), vertex_id);
}
```

### Comparison

| Aspect | RISC-V Custom Instruction | x86 Pseudo-Instruction |
|--------|---------------------------|------------------------|
| Encoding | Clean custom-0 opcode | m5ops magic address |
| Assembly | `ecg.extract rd, rs1` | `m5_ecg_extract(fat_id)` |
| Latency | 1 cycle (single-cycle exec) | ~10+ cycles (trap to simulator) |
| Realism | Realistic custom ISA extension | Simulation artifact |
| Cross-compile | Requires RISC-V toolchain | Native compilation |
| Development ease | More setup | Easy, no cross-compile |

**Recommendation**: Use x86 pseudo-instruction for initial development and
validation. Switch to RISC-V custom instruction for publication-quality results
and hardware-realistic latency modeling.

## Benchmark Instrumentation

Add ECG instruction calls where `SIM_CACHE_READ_MASKED` is currently used
in `bench/src_sim/*.cc`:

```cpp
// Current standalone cache_sim:
SIM_CACHE_READ_MASKED(cache, scores, neighbor, graph_ctx, mask_val);

// gem5 equivalent (RISC-V):
uint64_t fat_neighbor = edge_list[i];  // Fat-ID from CSR
uint32_t clean_id;
asm volatile ("ecg.extract %0, %1" : "=r"(clean_id) : "r"(fat_neighbor));
// Cache controller automatically reads mask CSR on next load
float val = scores[clean_id];

// gem5 equivalent (x86):
uint64_t fat_neighbor = edge_list[i];
uint32_t clean_id = m5_ecg_extract(fat_neighbor);
float val = scores[clean_id];
```

## Macro Abstraction

```cpp
// bench/include/gem5_sim/overlays/gem5_ecg_ops.h
#ifdef GEM5_RISCV
  #define GEM5_ECG_EXTRACT(rd, fat_id) \
      asm volatile (".insn r 0x0B, 0, 0, %0, %1, x0" : "=r"(rd) : "r"(fat_id))
#elif defined(GEM5_X86)
  #include "gem5/m5ops.h"
  #define GEM5_ECG_EXTRACT(rd, fat_id) \
      rd = m5_ecg_extract(fat_id)
#else
  // No gem5: just strip mask in software (standalone sim path)
  #define GEM5_ECG_EXTRACT(rd, fat_id) \
      rd = (fat_id) & VERTEX_ID_MASK
#endif
```
