# ECG Successor HPCA Paper

## Working title

**Public title pending**

The public name should be graph-specific and should not reuse the workshop title.
The implementation remains under the `ECG_*` namespace.

## Prior publication gate

The preliminary paper is:

> A. T. Mughrabi, M. Baradaran, A. Samara, and K. Skadron, “ECG:
> Expressing Locality and Prefetching for Optimal Caching in Graph Structures,”
> IEEE IPDPSW 2024, pp. 520–525, DOI 10.1109/IPDPSW59749.2024.00094.

This is an archival IEEE workshop publication. HPCA 2027 does not permit a
substantially similar submission. Before registration:

1. send the PC chairs the workshop paper and a contribution-delta summary;
2. cite the workshop paper in the submission in the third person;
3. ensure the new title and abstract describe the new architecture, not the
   workshop mask/prefetch prototype;
4. retain written chair guidance with the artifact records.

Renaming alone does not make the submission eligible; the technical contribution
must be materially different.

## Required contribution delta

| IPDPSW 2024 ECG | HPCA successor |
|---|---|
| single metadata mask concept | K2 two-future-reference records |
| preliminary replacement/prefetch study | adaptive replacement plus StreamShield placement |
| basic trace-driven evaluation | cache_sim + gem5 + Sniper implementation and mechanism conformance |
| conceptual graph instruction | executable record-load ISA plus request-bound StreamShield placement |
| no complete overhead attribution | K2-vs-bypass factorial, traffic, capacity, timing, and instruction accounting |
| PageRank-focused | First-class PR/BFS/SSSP/BC/CC K2 delivery and online selection |
| instruction-capped detailed simulation | policy-independent static-edge work caps with fail-closed matrix certification |

The authoritative status of each delta and headline claim is generated from
`research/ecg-hpca/claim_gate.json`. Submission text may use only claims marked
`allowed`; currently those are matrix-free operation, scoped mechanism
correctness, the single full-graph cache_sim direction cell, static ISA
decomposition, and semantic-work evaluation infrastructure.

## Thesis

Graph analytics already stream an edge record before accessing irregular vertex
properties. K2 carries a compact future-reuse contract from that edge onto the
exact property Request, exposing graph semantics unavailable to PC/address
predictors without an eviction-time rereference-matrix lookup. The design keeps
the configured LLC data ways, accepts a disclosed side-metadata silicon
overhead, and requires no live P-OPT matrix.

## Mechanism

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the complete request flow,
replacement logic, worked K2 example, ISA table, and baseline comparison.

- **Tiered K2:** one 8-byte record carries
  `dest32 | tier2 | epoch1_15 | epoch2_15`; replacement uses the nearer valid
  rereference and a direction-aware, order-independent property-line tier.
- **Adaptive eviction:** PR uses `epoch_first`; BFS/SSSP use `degree_first`;
  BC/CC use the safe `rrip_first` fallback. BC applies K2 to its forward
  static-edge phase, and CC retains its undirected graph contract.
- **Online eviction:** sampled leader sets choose among RRIP-, GRASP-, epoch-,
  degree-, and LRU-first arms without using the graph or kernel name.
- **StreamShield:** one-touch packed records fill private caches, retain LLC-hit
  behavior, and do not allocate after an LLC miss.
- **ISA:** K2-M is a computed-address masked property load and the canonical
  contribution. K2-I optionally folds destination decoding/address generation
  into the load. StreamShield independently marks the record request.

In the canonical U32.D32 sequence, baseline and K2-M each execute six body
instructions; K2-M replaces the ordinary load one-for-one. K2-I executes two
and removes four extraction/address-generation instructions. Only K2-I may
claim that static instruction reduction.

## Contributions

1. Edge-carried degree and two-epoch reuse guidance bound to the exact property
   Request, with no LLC data-way reservation or live rereference matrix.
2. A typed computed-address semantic load plus an explicit request-carried
   current-epoch/context channel;
   indexed fusion is a separate extension.
3. Online set-dueling that selects among graph-aware victim rules at runtime.
4. One shared eviction-decision SSOT plus independent per-backend delivery and
   decision-conformance gates across cache_sim, gem5, and Sniper.
5. A validation framework that separates cache, transport, and optional indexed
   fusion; the primary 16-way design reports its added silicon cost, while
   equal-area/energy sensitivity remains a submission gate.

## Evaluation structure

1. Mechanism and factorial attribution in cache_sim on real graphs.
2. ISA and request-bound mechanism validation in gem5.
3. Scale and timing confirmation in Sniper using the full policy set.
4. Hardware/storage accounting and sensitivity analysis.

The artifact exposes separate 15-way and conservative 14-way K2 profiles
against unchanged 16-way baselines. These are equal-silicon sensitivities;
the primary implementation remains the full-capacity 16-way design with
disclosed metadata overhead.
Physical overhead values are accepted only from hashed CACTI/synthesis reports;
the artifact does not substitute analytical defaults for those measurements.
The checked-in packet generates a 16-way LLC input and 1RW/1R1W metadata SRAM
inputs with explicit ECC/macro rounding and all 16 ways in each set row.
Standard CACTI cannot represent the
14/15-way associativities, so those remain simulator capacity sensitivities.
The analytical table separately reports edge-stream bytes, added line-metadata
SRAM, and P-OPT reserved data capacity so unlike costs are not conflated.

The learned-policy baseline is Hawkeye. cache_sim's static-site proxy is used
only for development. gem5's real instruction-PC implementation is present;
the synthetic execution gate is complete and negative versus LRU, while the
submission table still requires a real-graph comparison.

The corrected cache_sim factorial is complete: StreamShield is an incremental
demand/traffic improvement over online K2, while the full mechanism still uses
more total traffic than P-OPT. This bounds the detailed-simulator claim before
timing runs.

The earlier sampled PageRank profile isolates the fused-stream tradeoff, but its
pre-surgical timing is superseded by the current all-kernel matrix.

The corrected 120-row sampled Sniper matrix is an idealized packed-record
K2-I-like model, not measured K2-I ISA timing and not the core mask-only load.
The model reaches 1.792x on PR, 1.675x on BFS, 1.145x on SSSP,
1.082x on BC, and 1.115x on CC versus LRU. K2-online+StreamShield reaches
1.329x overall, versus GRASP at 1.100x and charged P-OPT at 1.082x. Because its
0.881x instruction ratio includes indexed/packed-loop savings, this number is
an extension-model result and cannot headline K2-M; its TPI also comes from a
different instruction mix and is not a K2-M estimate. The
ordering remains after excluding the shortest BFS cell (1.276x versus 1.107x).
Compact SSSP wins in geomean but remains graph-sensitive: cit-Patents still
loses substantially. The CPI stack cannot be decomposed beyond total ticks per
instruction.

The full-graph warm-SIFT queue blocker is resolved: matched web-Google LRU and
K2 100K probes reach ROI with normal cache warming. The 600M matrix remains a
data-collection gate before claiming overall detailed-simulator superiority
over P-OPT. The first post-fix K2 cell completes a full PR iteration before the
600M cap. Its current generic STRIDE8
configuration is rejected by the bounded diagnostic and must be replaced by a
traffic-bounded setting before the headline run.
