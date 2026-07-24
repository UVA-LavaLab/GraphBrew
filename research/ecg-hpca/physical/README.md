# K2 Physical Measurement Inputs

This directory defines the physical-cost inputs; it contains no measured result.
Generate the runnable CACTI packet with:

```bash
python3 -m scripts.experiments.ecg.analysis.k2_cacti_packet \
  --out-dir /tmp/k2-cacti-packet
```

The packet derives three configurations from the vendored CACTI 6.5 template:

- 8 MiB, 16-way, 64-byte LLC with one read/write port and CACTI ECC storage;
- K2 metadata SRAM with one read/write port;
- the same metadata SRAM with one read and one write port as a port sensitivity.

The LLC has 131,072 lines. K2 stores 49 logical bits per line before ECC.
The physical input adds seven SECDED bits, rounds the protected 56-bit entry to
a 64-bit per-way field, and packs all 16 ways into one 1,024-bit row for each
of 8,192 sets. The resulting macro is 1 MiB and exposes every candidate way to
victim selection in one read. CACTI ECC is disabled because the row already
includes the check bits; encoder/decoder logic is a mandatory separate
synthesis component.

CACTI 6.5 rejects non-power-of-two associativity, so it cannot directly model
the 14-way and 15-way simulation sensitivities. Those rows remain capacity
sensitivities; they are not mislabeled as measured CACTI equal-area points.

Running CACTI is an explicit later step:

```bash
make -C bench/include/gem5_sim/gem5/ext/mcpat/cacti CXX=g++ CC=gcc
python3 -m scripts.experiments.ecg.analysis.k2_cacti_packet \
  --out-dir /tmp/k2-cacti-packet \
  --run \
  --cacti-binary \
  bench/include/gem5_sim/gem5/ext/mcpat/cacti/cacti
```

The runner isolates each invocation because vendored CACTI appends to
`out.csv`. It records config, report, stdout, template, source, and executable
hashes and writes `physical_input.1rw.partial.json` and
`physical_input.1r1w.partial.json` with the CACTI fields mapped to the
fail-closed physical schema. Report node, capacity, banks, associativity, and
output width must match the selected profile. Final JSON artifacts are
published atomically and stale results are invalidated before a rerun.
Synthesis fields intentionally remain empty.
These measurements still do not satisfy the physical gate until ECC,
replacement, and request-path logic are synthesized with a pinned 32 nm
technology library. The 1,024-bit row is a conservative full-set read/write
model; per-way write masking is not modeled.
The SECDED synthesis component represents 16 parallel per-way codecs: its
latency is one decoder, while its total area scales with LLC ways in the linear
equal-area sensitivity.

Emit the hashed replacement/ECC RTL manifest with:

```bash
python3 -m scripts.experiments.ecg.analysis.k2_rtl_packet \
  --out-dir /tmp/k2-rtl-packet
```

The checked-in RTL and functional tests cover all seven C++ victim-ranking
variants, equivalent collapsed RRIP aging, property-region/context/two-epoch
qualification, exact five-arm online dueling, and 49-bit SECDED.
`k2_replacement_path` is fixed to the 32,768-epoch physical point. Any
non-baseline recency-rank maintenance must still be charged. Technology
synthesis is not performed by this command, and request-state storage/merge RTL
is emitted as per-unit tops rather than an arbitrary machine-wide count.
Its two region descriptors must contain only the benchmark's epoch-governed
arrays (for example PR `contrib`, not `scores`; BC `depth,path_counts`).

The request packet covers one MSHR extension slot, per-hart epoch/context CSRs,
one 95-bit pipeline copy, an optional eight-lane sequence allocator, and
optional registered recency rank state. Final characterization must scale them
using the target machine's actual counts. `k2_physical.py` requires those
instance counts, per-access activations, and the integrated request-path
critical delay; a single per-unit value cannot pass as machine-wide cost.
Existing MSHR address matching, allocation/arbitration, and queue control are
baseline resources and are not recharged to K2.
