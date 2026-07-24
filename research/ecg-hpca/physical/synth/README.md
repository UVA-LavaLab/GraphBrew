# K2 Synthesis Inputs

These RTL files are measurement inputs, not measured results:

- `bench/src_rtl/k2_victim_select.sv` implements the seven
  `ecg_policy::Variant` victim decisions for 16 ways. Uniform RRIP aging is
  collapsed to the equivalent current-max-RRPV candidate set and emitted as
  updated RRPVs.
- `bench/src_rtl/k2_secded_49.sv` implements a 49-bit SECDED codec and a
  16-way parallel encoder top.

The victim selector is a ranking/aging subcomponent. It consumes
baseline-provided replacement state: valid,
property/stamp bits, 3-bit RRPV, 4-bit age rank, 2-bit tier, and 15-bit
effective distance. The 4-bit rank is sufficient to order 16 ways and is not
additional K2 line metadata.

Use `k2_secded_49_parallel16`, which instantiates 16 encoders and 16 decoders,
for total codec area. Use `k2_secded_49_decode` for one-way decode delay. Use
`k2_victim_select` with its default 16-way parameters only for ranking-core
area/delay. `k2_replacement_path` is the replacement-component top: it adds
two prefiltered epoch-region descriptors, context qualification, two-epoch
circular distance at the fixed 32,768-epoch physical point, and exact five-arm
online dueling. With 16 ways this synthesizes 32 parallel range checks, not two
shared comparators. Singleton records must repeat epoch1 in epoch2. Any
recency-rank maintenance not already present in the baseline LLC must still be
added or charged separately. Physical reports require a pinned 32 nm library
and remain pending.

`tb_k2_physical_logic.sv` checks invalid-way priority, all headline selector
orders including `EPOCH_ONLY` and default dispatch, RRIP aging equivalence, and
SECDED no-error/single-error/double-error behavior. It is a functional test,
not technology synthesis.

Run the required generic verification with:

```bash
python3 -m scripts.experiments.ecg.analysis.k2_rtl_verify
```

Unlike the portable pytest wrapper, this command fails if Verilator or Yosys is
missing and records tool versions plus input hashes.
