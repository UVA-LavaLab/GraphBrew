# Paper Table 6 — Cache-substrate complexity comparison

Hardware/software complexity for each cache-substrate component
the paper compares against. Sprint 6f-5 rubber-duck recommendation:
the prior "ECG = 2 magic instructions vs DROPLET's 2 prefetch
engines" headline was apples-to-oranges (transparent hardware vs
software-assisted). This table provides a fair comparison along
five axes.

## Comparison axes

- **Storage per vertex**: bytes of per-vertex state in DRAM
- **Fixed state**: bytes of SRAM-resident state (tables, monitors)
- **Hardware datapath**: combinational logic + state machines
- **ISA extensions**: new instructions / magic opcodes
- **Offline preprocessing**: complexity + wall time on representative graphs
- **Per-access runtime cost**: extra cycles + tables traversed per cache access
- **Software kernel changes**: lines of kernel source modified

## Storage summary

| Component | Per-vertex (B) | Fixed (B) | Per-access cycles | ISA |
|---|---:|---:|---:|---|
| ECG (this work) | 8 | 0 | 1 ev / 2 pf | 2 magic |
| DROPLET (Basak HPCA'19) | 0 | 16,384 | 0 ev / 4 pf | none |
| POPT (Balaji HPCA'21) | 16 | 0 | 2 ev / 0 pf | none |
| GRASP (Faldu HPCA'20) | 0 | 20,480 | 0 ev / 0 pf | none |

## Hardware datapath comparison

### ECG (this work)

1 per-access mask decoder: 1 bit-shift + 2 range compares + 1 OR. ~15 gates of combinational logic + 1 4-input MUX for hint hand-off to L2 prefetch port. Mask itself is a uint32_t/uint64_t array in memory (no SRAM-resident table).

### DROPLET (Basak HPCA'19)

2 prefetch engines: (a) stride detector with 64-entry stride table tracking edge-list access pattern; (b) indirect-property engine issuing K=16 prefetches per stride trigger. Both engines snoop L2 access stream + property-region monitors. ~5,000 gates of combinational logic + state machines per engine (estimated from Basak HPCA'19 ASIC synthesis numbers).

### POPT (Balaji HPCA'21)

Per-access re-reference matrix lookup unit. Each LLC access computes cache_line_index = addr / line_size + epoch_index = cycle / epoch_length, then indexes a 2-D rereference matrix (numEpochs × numCacheLines bytes) to get the predicted reuse distance. ~500 gates of address computation + matrix read port. The matrix itself is multi-MB and typically lives in dedicated SRAM next to the LLC (per Balaji HPCA'21 Section 4).

### GRASP (Faldu HPCA'20)

Per-line degree-bucket tag + range-classification monitor. GRASP adds a 1-2-bit tier tag to every L3 cache line + a small region table tracking vertex-property address ranges. The replacement policy reads the tier tag to bias eviction. ~200 gates per tier comparator + small region tag lookup.

## ISA extensions

### ECG (this work)

2 magic instructions wired through Sniper SimMagic / gem5 MAGIC / cache_sim SIM_CACHE_PREFETCH_VERTEX + SIM_CACHE_READ_MASKED. Each instruction is a 1-cycle no-op outside the simulator and a single tagged opcode inside. Mask value is delivered as a register argument; target is a register-resident vertex ID.

### DROPLET (Basak HPCA'19)

Zero ISA changes — transparent hardware. The CPU emits ordinary loads/stores; DROPLET watches the L2 access stream and emits speculative prefetches.

### POPT (Balaji HPCA'21)

Zero ISA changes — transparent hardware. POPT only needs the cache controller to know about the rereference matrix; software does not see it.

### GRASP (Faldu HPCA'20)

Zero ISA changes — transparent hardware. GRASP infers vertex tier from property-array address ranges set up at program start.

## Offline preprocessing

| Component | Complexity | email-Eu-core | cit-Patents | kron-s24 |
|---|---|---:|---:|---:|
| ECG (this work) | O(N · avg_degree) — one pass over CSR + POPT-rank lookup per vertex | 0.001s | 0.079s | 1.550s |
| DROPLET (Basak HPCA'19) | None — fully runtime | 0.000s | 0.000s | 0.000s |
| POPT (Balaji HPCA'21) | O(N · avg_degree · numEpochs) — sliding-window pass building per-(cline, epoch) reuse-distance map | 0.002s | 0.094s | 6.000s |
| GRASP (Faldu HPCA'20) | O(N) — single-pass degree histogram for tier boundaries | 0.001s | 0.017s | 0.400s |

## Software kernel changes

### ECG (this work)

Inner loop adds SIM_CACHE_READ_MASKED(...) before demand load + (optional) SIM_CACHE_PREFETCH_VERTEX(...) for lookahead. ~5-10 lines per kernel function in bench/src_sim/{pr,bfs,sssp}.cc.

### DROPLET (Basak HPCA'19)

None — transparent hardware

### POPT (Balaji HPCA'21)

None at the kernel level (transparent hardware), BUT the matrix must be built offline before the kernel runs — see preprocessing complexity above.

### GRASP (Faldu HPCA'20)

Software must declare property-array address ranges at program start. Otherwise transparent. ~2-5 lines per kernel.

## Citations

- **ECG (this work)**: This work
- **DROPLET (Basak HPCA'19)**: Basak et al., HPCA 2019, "Analysis and Optimization of the Memory Hierarchy for Graph Processing Workloads"
- **POPT (Balaji HPCA'21)**: Balaji and Lustig, HPCA 2021, "P-OPT: Practical Optimal Cache Replacement for Graph Analytics"
- **GRASP (Faldu HPCA'20)**: Faldu et al., HPCA 2020, "A Closer Look at Lightweight Graph Reordering"

## Pareto-frontier interpretation (paper-ready language)

ECG occupies a different point in the (per-vertex storage, fixed
state, ISA complexity, runtime cost) space than DROPLET or POPT:

- vs **DROPLET**: ECG trades 2 ISA instructions + per-vertex mask
  storage for elimination of the 2 prefetch engines + 16 KB SRAM
  state + 4 cycles per prefetch decision. ECG's preprocessing
  cost (~0.08s on cit-Patents) is the price of removing the
  hardware engines.
- vs **POPT**: ECG cuts per-vertex storage from 16 B (POPT
  rereference matrix) to 8 B (ECG mask) — **2x smaller** — at
  comparable preprocessing cost. ECG's mask is read once per
  cache access (1 cycle) vs POPT's 2-cycle matrix index +
  lookup.
- vs **GRASP**: ECG adds POPT-class reuse-distance prediction +
  prefetch hints on top of GRASP-class eviction tiers, at ~8 B
  per vertex vs GRASP's ~20 KB fixed + per-line tier tag.

**Honest framing**: ECG is not strictly Pareto-dominant on any
single axis — POPT's transparent hardware avoids ISA changes;
DROPLET's transparent hardware avoids software preprocessing.
ECG's value is the **unification**: one mask substrate replacing
three separate mechanisms.
