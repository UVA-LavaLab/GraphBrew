# Reproduction Runbook

## Required graph

The correctness gates and headline profile expect:

```text
results/graphs/email-Eu-core/email-Eu-core.sg
results/graphs/web-Google/web-Google.sg
results/graphs/soc-pokec/soc-pokec.sg
results/graphs/cit-Patents/cit-Patents.sg
```

Graph datasets and converted `.sg` files are ignored. Build the converter with
`make converter` when staging a new graph. One reproducible SNAP staging recipe:

```bash
mkdir -p \
  results/graphs/email-Eu-core \
  results/graphs/web-Google \
  results/graphs/soc-pokec \
  results/graphs/cit-Patents

curl -L https://snap.stanford.edu/data/email-Eu-core.txt.gz |
  gzip -dc > results/graphs/email-Eu-core/email-Eu-core.el
curl -L https://snap.stanford.edu/data/web-Google.txt.gz |
  gzip -dc > results/graphs/web-Google/web-Google.el
curl -L https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz |
  gzip -dc > results/graphs/soc-pokec/soc-pokec.el
curl -L https://snap.stanford.edu/data/cit-Patents.txt.gz |
  gzip -dc > results/graphs/cit-Patents/cit-Patents.el

make converter
bench/bin/converter \
  -f results/graphs/email-Eu-core/email-Eu-core.el \
  -b results/graphs/email-Eu-core/email-Eu-core.sg
bench/bin/converter \
  -f results/graphs/web-Google/web-Google.el \
  -b results/graphs/web-Google/web-Google.sg
bench/bin/converter \
  -f results/graphs/soc-pokec/soc-pokec.el \
  -b results/graphs/soc-pokec/soc-pokec.sg
bench/bin/converter \
  -f results/graphs/cit-Patents/cit-Patents.el \
  -b results/graphs/cit-Patents/cit-Patents.sg
```

## Build correctness-gate binaries

```bash
make setup-gem5
make setup-sniper
make all-sim
make gem5-riscv-m5ops-pr gem5-riscv-m5ops-bfs \
  gem5-riscv-m5ops-sssp gem5-riscv-m5ops-bc gem5-riscv-m5ops-cc
make sniper-sg_kernel
```

## Final run order

1. Run the three correctness gates.
2. Run and validate `ecg_3sim_allalg_smoke`.
3. Close the current-epoch/context and MSHR correctness gates.
4. Quantify the full 16-way metadata overhead and run 15/14-way equal-area
   sensitivities.
5. Add the learned replacement baseline before freezing policy rankings.
6. Run `ecg_replacement_baseline`, `ecg_cache_sim_factorial`, and
   `ecg_streamshield_generality`.
7. Run gem5 and Sniper mechanism profiles.
8. Aggregate only complete, hash-consistent runs.
9. Run one bounded matched Sniper pair before expanding to a headline matrix.

Before editing the abstract, contributions, or conclusions, run:

```bash
python3 -m scripts.experiments.ecg.analysis.claim_gate
```

Do not promote a prohibited claim by prose alone. Update a gate to `passed`
only when its evidence commit/path exists and the underlying result is frozen.

Render the canonical ISA decomposition:

```bash
python3 -m scripts.experiments.ecg.analysis.k2_isa_decomposition
```

Acceptance is baseline=6, K2-M=6, K2-I=2 body instructions for U32.D32, with
raw encodings matching `gem5_harness.h`. Never translate the K2-I `-4` static
delta into a K2-M speedup estimate.

## Full 3-simulator/all-algorithm smoke

```bash
python3 scripts/experiments/ecg/slurm/make_slurm_shards.py \
  --profile ecg_3sim_allalg_smoke \
  --run-tag ecg_3sim_smoke \
  --out results/ecg_experiments/slurm/ecg_3sim_smoke.tsv

python3 scripts/experiments/ecg/flows/run_local_shards.py \
  --shards results/ecg_experiments/slurm/ecg_3sim_smoke.tsv \
  --run-root results/ecg_experiments/final_paper_runs/local \
  --jobs 8 --cache-sim-jobs 5 --gem5-jobs 1 --sniper-jobs 1

python3 scripts/experiments/ecg/flows/paper_pipeline.py \
  --skip-run \
  --input-run-glob \
    "results/ecg_experiments/final_paper_runs/local/ecg_3sim_smoke/*" \
  --run-root results/ecg_experiments/paper_pipeline/ecg_3sim_smoke

python3 scripts/experiments/ecg/verify/smoke_coverage.py \
  --csv results/ecg_experiments/paper_pipeline/ecg_3sim_smoke/aggregate/roi_matrix_all.csv
```

Acceptance is exactly 120 valid rows: 3 simulators x 5 algorithms x 8 policies.

## Three-real-graph cross-simulator matrix

This no-prefetch comparison runs web-Google, soc-pokec, and cit-Patents across
cache_sim, gem5, and Sniper for PR/BFS/SSSP/BC/CC and the eight final policies.

```bash
python3 scripts/experiments/ecg/slurm/make_slurm_shards.py \
  --profile ecg_3sim_realgraph_allalg \
  --run-tag ecg_3sim_realgraph_allalg \
  --out results/ecg_experiments/slurm/ecg_3sim_realgraph_allalg.tsv

python3 scripts/experiments/ecg/flows/paper_run.py \
  --profile ecg_3sim_realgraph_allalg \
  --only 24_sniper --benchmark pr --policy LRU \
  --run-dir results/ecg_experiments/final_paper_runs/sniper_realgraph_calibration \
  --no-build

python3 scripts/experiments/ecg/flows/run_local_shards.py \
  --shards results/ecg_experiments/slurm/ecg_3sim_realgraph_allalg.tsv \
  --run-root results/ecg_experiments/final_paper_runs/local \
  --jobs 9 --cache-sim-jobs 4 --gem5-jobs 4 --sniper-jobs 1

python3 scripts/experiments/ecg/flows/paper_pipeline.py \
  --skip-run \
  --input-run-glob \
    "results/ecg_experiments/final_paper_runs/local/ecg_3sim_realgraph_allalg/*" \
  --run-root results/ecg_experiments/paper_pipeline/ecg_3sim_realgraph_allalg

python3 scripts/experiments/ecg/verify/smoke_coverage.py \
  --csv \
    results/ecg_experiments/paper_pipeline/ecg_3sim_realgraph_allalg/aggregate/roi_matrix_all.csv \
  --graph web-Google soc-pokec cit-Patents
```

Acceptance is exactly 360 valid rows. Use
`aggregate/roi_relative_metrics.csv` for within-simulator LRU-normalized
miss/timing comparisons; do not compare absolute miss rates across simulators.

The calibration command must complete three rows before the full launch. The
single-node defaults above are for this 32-core/62-GiB host: four gem5 jobs are
safe because every shard has isolated sidebands, while Sniper remains at one
job under its 20-GiB address-space cap. Reduce concurrency if memory pressure
appears. BC K2 covers the forward Brandes traversal only; CC retains the
artifact's undirected/symmetric graph contract.

### Quick 1B-instruction diagnostic

Use the already-complete full-work cache_sim rows and rerun only gem5/Sniper
with a one-billion-instruction detailed-ROI cap:

```bash
python3 scripts/experiments/ecg/slurm/make_slurm_shards.py \
  --profile ecg_3sim_realgraph_allalg_1b \
  --run-tag ecg_3sim_realgraph_allalg_1b \
  --out results/ecg_experiments/slurm/ecg_3sim_realgraph_allalg_1b.tsv

awk -F '\t' '$2 ~ /^25_|^26_/' \
  results/ecg_experiments/slurm/ecg_3sim_realgraph_allalg_1b.tsv \
  > results/ecg_experiments/slurm/ecg_3sim_realgraph_detailed_1b.tsv

python3 scripts/experiments/ecg/flows/run_local_shards.py \
  --shards results/ecg_experiments/slurm/ecg_3sim_realgraph_detailed_1b.tsv \
  --run-root results/ecg_experiments/final_paper_runs/local \
  --jobs 8 --gem5-jobs 4 --sniper-jobs 4
```

Gem5 schedules the cap from the compute ROI work-begin marker, so graph loading
does not consume the one-billion-instruction budget. Capped rows set
`timing_valid_for_speedup=0`; compare cache metrics only and label every table
as instruction-capped diagnostic evidence.

### Fast full-work sampled matrix

The reproducible sample sizes are web-Google 65,536 vertices/502,529 edges,
soc-pokec 65,536/1,089,520, and symmetrized cit-Patents
262,144/340,054 undirected edges. Generate samples with
`flows/sample_realgraph.py`, serialize them with `bench/bin/converter` (`-s`
for cit-Patents), then run:

```bash
python3 scripts/experiments/ecg/flows/sample_realgraph.py \
  --input results/graphs/web-Google/web-Google.el \
  --output results/graphs/web-Google-n16/web-Google-n16.el \
  --vertices results/graphs/web-Google-n16/web-Google-n16.vertices.tsv \
  --metadata results/graphs/web-Google-n16/web-Google-n16.sample.json \
  --target-vertices 65536

python3 scripts/experiments/ecg/flows/sample_realgraph.py \
  --input results/graphs/soc-pokec/soc-pokec.el \
  --output results/graphs/soc-pokec-n16/soc-pokec-n16.el \
  --vertices results/graphs/soc-pokec-n16/soc-pokec-n16.vertices.tsv \
  --metadata results/graphs/soc-pokec-n16/soc-pokec-n16.sample.json \
  --target-vertices 65536

python3 scripts/experiments/ecg/flows/sample_realgraph.py \
  --input results/graphs/cit-Patents/cit-Patents.mtx \
  --output results/graphs/cit-Patents-n18/cit-Patents-n18.el \
  --vertices results/graphs/cit-Patents-n18/cit-Patents-n18.vertices.tsv \
  --metadata results/graphs/cit-Patents-n18/cit-Patents-n18.sample.json \
  --target-vertices 262144

bench/bin/converter \
  -f results/graphs/web-Google-n16/web-Google-n16.el \
  -b results/graphs/web-Google-n16/web-Google-n16.sg
bench/bin/converter \
  -f results/graphs/soc-pokec-n16/soc-pokec-n16.el \
  -b results/graphs/soc-pokec-n16/soc-pokec-n16.sg
bench/bin/converter -s \
  -f results/graphs/cit-Patents-n18/cit-Patents-n18.el \
  -b results/graphs/cit-Patents-n18/cit-Patents-n18-sym.sg

python3 scripts/experiments/ecg/slurm/make_slurm_shards.py \
  --profile ecg_3sim_sampled_allalg \
  --run-tag ecg_3sim_sampled_allalg \
  --out results/ecg_experiments/slurm/ecg_3sim_sampled_allalg.tsv

python3 scripts/experiments/ecg/flows/run_local_shards.py \
  --shards results/ecg_experiments/slurm/ecg_3sim_sampled_allalg.tsv \
  --run-root results/ecg_experiments/final_paper_runs/local \
  --jobs 12 --cache-sim-jobs 4 --gem5-jobs 4 --sniper-jobs 4
```

All rows run to semantic completion. The samples are deterministic diagnostic
proxies for the named real graphs, not replacements for full-graph authority.
The soc-pokec sample has 2.0 LLC bytes/vertex versus 1.28 for the full graph,
so its sampled cache pressure is lower. Sample metadata counts pre-converter
directed arcs; the symmetrized cit-Patents `.sg` contains both directions.

### Confirm fused record bandwidth on sampled PageRank

This profile compares equal-work GRASP, charged P-OPT, and
K2-online+StreamShield after removing software-only Sniper delivery and disabled
hint instrumentation from the ROI:

```bash
python3 scripts/experiments/ecg/slurm/make_slurm_shards.py \
  --profile ecg_sniper_sampled_pr_streamengine \
  --run-tag ecg_sniper_sampled_pr_streamengine \
  --out results/ecg_experiments/slurm/ecg_sniper_sampled_pr_streamengine.tsv

python3 scripts/experiments/ecg/flows/run_local_shards.py \
  --shards results/ecg_experiments/slurm/ecg_sniper_sampled_pr_streamengine.tsv \
  --run-root results/ecg_experiments/final_paper_runs/local \
  --jobs 3 --sniper-jobs 3

python3 scripts/experiments/ecg/flows/paper_pipeline.py \
  --skip-run \
  --input-run-glob \
    'results/ecg_experiments/final_paper_runs/local/ecg_sniper_sampled_pr_streamengine/*' \
  --run-root \
    results/ecg_experiments/paper_pipeline/ecg_sniper_sampled_pr_streamengine
```

Require all nine rows, valid fused receipts, and comparable instruction counts.
`l3_misses - sniper_stream_bypass_reads` separates non-record demand misses
because `stream-bypass-reads` increments only on a bypassed LLC miss. Keep total
LLC accesses/misses and instructions beside simulated time. Report
ticks-per-instruction so packed-record traversal savings are separated from the
cache/memory effect.

Sniper P-OPT uses the exact optimized host consultation path by default. Set
`SNIPER_POPT_FAST=0` only for legacy A/B equivalence checks; paper rows record
`sniper_popt_fast=1`.

gem5 P-OPT already performs one matrix lookup per candidate and caches those
distances through RRIP tie aging. Do not attribute gem5's cycle-accurate wall
time to the repeated-consultation issue that affected legacy Sniper.

### K2-M versus K2-I

The canonical paper load is K2-M:

```text
record = load K2 edge record
dest   = extract destination
addr   = property_base + dest * element_size
value  = ecg.k2.mload(addr, record)
```

The existing gem5 path is K2-I; Sniper is only a K2-I-like packed-loop model:

```text
record = load K2 edge record
value  = ecg.k2.iload(property_base, record)
```

Never merge their timing rows. K2-M isolates request-bound cache metadata;
gem5 K2-I additionally credits destination/address fusion. Sniper's model is
not measured K2-I ISA timing. Each result row must gain an `ecg_isa_variant`
field before the next timing matrix.

K2-M mechanism validation:

```bash
python3 scripts/experiments/ecg/verify/equiv_kernels.py \
  --gem5 --kernels pr bfs sssp bc cc \
  --schedule-k 2 --gem5-isa-variant mask
```

Acceptance requires all five gem5 cells to report
`computed-address K2-M property load: [OK]`, zero distance mismatches, and BC
dual-load coverage. Compact SSSP must report `ecg_isa_variant=mask`,
`ecg_record_bytes=8`, `edge_stream_bytes_per_edge=8`, and
`ecg_record_replaces_edge=1`.

Matched Sniper K2-M certification:

```bash
python3 scripts/experiments/ecg/verify/matched_k2m.py \
  --root /tmp/ecg_sniper_k2m_allkernel_final
```

Each kernel directory must contain exactly one LRU and one K2 row from the same
`sg_kernel` binary/configuration. Rows may be uncapped or use the semantic edge
cap, but committed-instruction-capped rows are rejected. Completion hashes,
transport markers, semantic outputs, diagnostic-only timing, and workload hashes
must match.
Instruction ratio must remain within 0.25%; the current five-kernel gate is
exactly 1.000x.
For semantic-capped rows, `sniper_semantic_edge_limit`,
`sniper_semantic_edge_visits`, and `sniper_semantic_truncated` must match and
`semantic_work_matched=1` must be present in both rows.
Policy-filtered shards remain uncertified individually; `paper_pipeline.py`
sets the field only after the complete policy group passes the same checks.
New certifications additionally require `sniper_k2_exact_bind=1` in both rows
and `[K2_EXACT_BIND]` in both logs, plus
`sniper_k2_epoch_context_bound=1`. The marker must cover only edge-governed
destination loads; SSSP source distance, BC source path count/backward work,
and CC pointer chasing/compression remain unmarked.

gem5 architectural epoch/context validation:

```bash
python3 -m pytest \
  scripts/test/test_k2_mshr_state.py \
  scripts/test/test_gem5_ecg_pfx_scaffold.py -q
python3 scripts/setup_gem5.py --isa X86 RISCV --jobs 16 --rebuild
make -j12 gem5-riscv-m5ops-pr gem5-riscv-m5ops-bfs \
  gem5-riscv-m5ops-sssp gem5-riscv-m5ops-bc gem5-riscv-m5ops-cc
```

Acceptance requires CSR addresses `0x800/0x801`, all K2 decoder forms to
snapshot both registers, request-bound O3 sequence propagation, context-tagged
line metadata, and passing MSHR merge mutations. These are implementation gates,
not performance experiments.

For host-cost diagnosis, use `SNIPER_K2_LOOKUP_PROFILE=1` and explicit
`SNIPER_ECG_HOST_PROFILE=1`. Do not reintroduce the rejected direct-mapped K2
lookup memo: its measured hit rate was 0.0059% and it slowed the A/B run 7.4%
with bit-identical target statistics. Live versus explicit-SIFT frontend
selection and a global `-O3` simulator build were also neutral. Do not replace
the warm run with standalone ROI-only trace replay; it starts with cold caches,
while replaying the complete warm prefix retains essentially the direct-run
cost.

### Equal-area acceptance gate

Before any hardware-efficiency headline:

1. enumerate line metadata, context tags, CSR state, request/queue/MSHR bits,
   comparator logic, and ECC;
2. report SRAM area and access energy plus replacement-selection logic area,
   energy, and delay;
3. report the primary full 16-way equal-data-capacity design with its added
   silicon overhead;
4. run equal-silicon-area results for a 15-way K2 LLC versus a 16-way baseline
   and for the conservative 14-way integral sensitivity; 15 ways is
   intentionally undercharged for the contextual 49-bit state;
5. verify metadata lookup runs parallel to tag/data access and that victim
   selection does not extend the cache critical path;
6. reject “lower hardware overhead than P-OPT” unless K2 retains its direction
   after the equal-area gate.

For Sniper mask-mode rows, only P-OPT may set
`sniper_popt_matrix_required=1` or report `sniper_rereference_loaded=1`.
K2-M, LRU, SRRIP, and GRASP use the matched K2 record transport without
constructing or loading the P-OPT matrix.

Generate the analytical floor with:

```bash
python3 -m scripts.experiments.ecg.analysis.k2_area \
  --cache-bytes 8388608 --line-bytes 64 --ways 16
```

Inspect the two no-build sensitivity profiles with:

```bash
python3 scripts/experiments/ecg/flows/paper_run.py \
  --profile ecg_equal_area_15 \
  --run-dir /tmp/ecg-equal-area-15 \
  --list --dry-run --no-build
python3 scripts/experiments/ecg/flows/paper_run.py \
  --profile ecg_equal_area_14 \
  --run-dir /tmp/ecg-equal-area-14 \
  --list --dry-run --no-build
```

Rows must report baseline `l3_ways=16`, the actually configured
`l3_effective_ways`, `k2_area_mode`, and `k2_l3_ways_requested`. Only
Schedule-2 K2 policies receive the override.

The equal-area profiles include `HAWKEYE:PROXY`. Accept it only as a cache_sim
diagnostic and require `hawkeye_pc_source=static_access_site_proxy` plus
`hawkeye_faithfulness=proxy_not_real_instruction_pc`. Do not promote it to the
headline Hawkeye baseline; that requires the gem5 real-PC implementation.

Inspect the faithful real-PC gem5 gate without running it:

```bash
python3 scripts/experiments/ecg/flows/paper_run.py \
  --profile ecg_gem5_hawkeye_gate \
  --run-dir /tmp/ecg-gem5-hawkeye \
  --list --dry-run --no-build
```

Its Hawkeye rows must report
`hawkeye_pc_source=request_instruction_pc` and
`hawkeye_faithfulness=faithful_real_instruction_pc`.

The default v2 contract reports a 49-bit line payload before ECC, 1.531
baseline-way equivalents, 14.602 self-consistent fractional ways, 15 ways as
an intentionally undercharged first sensitivity, 14 ways as the maximum
integral equal-area point, and a 95-bit logical request payload. These are
bit-level lower bounds, not CACTI/synthesized physical costs.

Emit the pinned CACTI 6.5 input packet without running it:

```bash
python3 -m scripts.experiments.ecg.analysis.k2_cacti_packet \
  --out-dir /tmp/k2-cacti-packet
```

The packet hashes the vendored source/template and emits the 8 MiB/16-way LLC,
a 1 MiB 1RW K2 metadata SRAM, and a 1R1W metadata port sensitivity. Each of
8,192 rows contains all 16 rounded 64-bit way fields, exposing a 1,024-bit set
row for victim selection. Standard CACTI rejects 14/15-way associativity, so
those simulator profiles remain capacity sensitivities rather than measured
CACTI points.

Emit and validate the replacement/ECC RTL inputs:

```bash
python3 -m scripts.experiments.ecg.analysis.k2_rtl_packet \
  --out-dir /tmp/k2-rtl-packet
python3 -m scripts.experiments.ecg.analysis.k2_rtl_verify
```

`k2_victim_select` covers all seven SSOT ranking variants and collapses uniform
RRIP aging to the equivalent current-max-RRPV candidate set. It is not the
final replacement component: distance/context/property qualification, online
selection, and non-baseline rank maintenance must be added before physical
measurement. The SECDED area top contains 16 encoders and 16 decoders; one
decoder supplies read delay. Request/CSR/queue/MSHR registered synthesis input
is still pending.

After external CACTI and synthesis runs, create and fill the measured physical
input schema:

```bash
python3 -m scripts.experiments.ecg.analysis.k2_physical \
  --template > /tmp/k2-physical-input.json
# Fill baseline/metadata SRAM, SECDED, replacement logic, request-path logic,
# activation fractions, and every report/input/library hash.
python3 -m scripts.experiments.ecg.analysis.k2_physical \
  --input /tmp/k2-physical-input.json \
  > /tmp/k2-physical-result.json
```

The command fails on missing components, malformed SHA-256 provenance, or
placeholder values. CACTI and synthesis technology nodes must match. The
parallel lookup delay includes metadata SECDED decode but excludes replacement selection;
the output reports request-to-data and eviction-selection delays separately.
Fields named
`linear_equal_area_*` are interpolation sensitivities, not measured 14/15-way
CACTI points.

Generate the three-cost reviewer table:

```bash
python3 -m scripts.experiments.ecg.analysis.three_costs \
  --cache-sizes 2MB 8MB \
  --out-json /tmp/k2-three-costs.json \
  --out-csv /tmp/k2-three-costs.csv
```

`Added SRAM way-eq` and `Reserved data ways` are deliberately different units:
K2 adds side metadata without removing data capacity; P-OPT reserves existing
LLC ways. Edge bytes cover one active traversal-direction stream.

Canonical K2-I timing requires RISC-V guest binaries rebuilt from the
current Makefile with `-funswitch-loops`. Reject rows from the stopped
`ecg_gem5_sampled_allalg_masked_load_20260719` run: they still contain
per-edge clear/trace scaffolding and are not timing authority.

Canonical Sniper packed K2-I-like model timing requires the current `sg_kernel`
with surgically split no-trace loops. It is not measured K2-I ISA timing and
cannot be reused as K2-M timing. The global
`-funswitch-loops` Sniper probe is rejected, and the pre-split 120-row timing
matrix remains historical.

The post-`e1ce2a8e` load-coverage binary additionally masks BC `path_counts`,
isolates the SSSP source load, clears CC hints before compression, and uses the
compact weighted record when eligible. Compact SSSP rows must report
`graph_edge_bytes=8`, `ecg_record_bytes=8`,
`edge_stream_bytes_per_edge=8`, and `ecg_record_replaces_edge=1`. General
fallback rows retain the 8+4=12-byte provenance. Do not merge focused probes
into historical matrices.

Canonical BC rows must additionally report
`property_regions=scores,depth,path_counts,deltas` and
`ecg_epoch_regions=depth,path_counts`. Reject the stopped
`ecg_{sniper,gem5}_sampled_allalg_compact_final_20260721` partial runs: their
completed rows predate the multi-region runner fix and govern only `depth`.

The completed sampled Sniper authority is
`results/ecg_experiments/paper_pipeline/`
`ecg_sniper_sampled_allalg_compact_scope_final_20260721/aggregate/`
with 120/120 valid rows.

The full cit-Patents compact SSSP risk gate is
`results/ecg_experiments/final_paper_runs/`
`ecg_cache_sim_citpatents_sssp_compact_full_20260721/roi_matrix.csv`.
Use it as cache-level evidence against a size-only failure, not as a causal
topology/scale decomposition or a Sniper timing result. The full-graph
K2-online+StreamShield row must retain `ecg_record_replaces_edge=1` and an
8-byte total edge stream.

General-fallback weighted SSSP rows must report `graph_edge_bytes=8`,
`ecg_record_bytes=4`, and `edge_stream_bytes_per_edge=12`. gem5 loads the
sidecar normally or with `ecg.stream.wload2`, then carries the reconstructed K2
mask on the property load. Sniper reports `fused-k2-weighted32-model`.

### Paper-faithful full-graph Sniper ROI

Before another long detailed run, dry-run the policy-independent semantic-work
gate:

```bash
python3 scripts/experiments/ecg/flows/paper_run.py \
  --profile ecg_sniper_semantic_gate \
  --run-dir \
    results/ecg_experiments/final_paper_runs/ecg_sniper_semantic_gate \
  --no-build --dry-run
```

The profile covers PR/BFS/SSSP/BC/CC and counts static graph-edge visits in the
single-core transport-matched mask guest kernel. Every completed row must contain exactly one
`[SEMANTIC-ROI benchmark=... edge_visits=... limit=... truncated=...]` marker.
The runner fails on a missing marker, wrong benchmark, wrong visit count, or
unexpected truncation state. This profile is implemented but has not been run.
For truncated rows, matching semantic outputs certify the same deterministic
edge prefix only; they do not replace the uncapped algorithm-correctness gate.

DROPLET warmed graph loading and collected 600 million ROI instructions.
GRASP simulated one representative high-activity iteration, and P-OPT used one
PageRank iteration or sampled pull iterations. GraphBrew follows that precedent
with full graphs and a detailed ROI capped at 600M instructions:

Run the warm full-graph gate first:

```bash
python3 scripts/experiments/ecg/flows/paper_run.py \
  --profile ecg_sniper_realgraph_warm_probe \
  --run-dir \
    results/ecg_experiments/final_paper_runs/ecg_sniper_realgraph_warm_probe \
  --no-build
```

Acceptance requires both LRU and K2 to report `status=ok`,
`sniper_cache_warming=1`, positive L3 metrics, and K2 context loading. The
CACHE_ONLY setup patch suppresses queue/shared-memory timing only during
warming; cache contents continue to update.

```bash
python3 scripts/experiments/ecg/slurm/make_slurm_shards.py \
  --profile ecg_sniper_realgraph_600m \
  --run-tag ecg_sniper_realgraph_600m \
  --out results/ecg_experiments/slurm/ecg_sniper_realgraph_600m.tsv
```

The profile uses the explicit SIFT frontend because the pinned build disables
its original Pin frontend.
Pre-ROI execution is not part of the 600M detailed budget. Explicit property replay
immediately before ROI supplements Sniper's normal cache-warming pass. Fused
transport is validated separately by the strict 120-row smoke gate, avoiding a
cold-start mechanism-proof mode in the performance profile. Capped rows set
`timing_valid_for_speedup=0` because K2 and the baselines do not execute
identical instruction streams; use miss, traffic, and direction metrics only.
Rows may reach semantic completion before the cap. The post-fix web-Google K2
PR cell completes at 179.4M reported instructions; it must not be treated as
equal-instruction timing against a baseline that reaches 600M.
The sampled full-completion profile remains the equal-work detailed timing
comparison. Under the baseline LRU instruction stream, existing calibration
shows 600M instructions cover about 18% of one web-Google PR iteration and 9%
of one soc-pokec PR iteration. K2's fused packed path is shorter and may finish
the iteration before the cap; report each row's executed scope explicitly.

## Inspect the blocked headline job

This profile remains blocked until a matched Sniper K2-M implementation exists,
the runner emits `ecg_isa_variant`, and the policy set uses explicit variant
labels. Renaming the current packed model is insufficient. Until then, the
legacy labels below denote the packed K2-I-like extension model only.

```bash
python3 scripts/experiments/ecg/flows/paper_run.py \
  --profile streamshield_sniper_realgraph \
  --run-dir /tmp/ecg-successor-webgoogle-dryrun \
  --list --dry-run --no-build
```

The command must contain exactly:

```text
LRU SRRIP GRASP POPT ECG:K2 ECG:K2_ONLINE
ECG:K2_STREAMSHIELD ECG:K2_ONLINE_STREAMSHIELD
```

## Reproduce the real-graph cache_sim factorial

First run the bounded five-algorithm diagnostic matrix:

```bash
python3 scripts/experiments/ecg/flows/paper_run.py \
  --profile ecg_preliminary_5alg_3sim \
  --run-dir results/ecg_experiments/final_paper_runs/ecg_preliminary_5alg \
  --no-build
```

This historical profile runs LRU, SRRIP, GRASP, charged P-OPT, static packed
K2-I-like, and online packed K2-I-like for
PR/BFS/SSSP/BC/CC on the common `kron_s15_k4` cell in cache_sim, gem5, and
Sniper. Compare policy direction and rank **within** each simulator. Do not
compare absolute gem5 and Sniper miss rates. Canonical Schedule-2 reruns use
the masked property-load delivery for all five kernels. Use gem5 O3 only for
tiny instruction-correctness cells; scale runs remain on TimingSimpleCPU.

Then run the matched structure-prefetch sensitivity:

```bash
python3 scripts/experiments/ecg/flows/paper_run.py \
  --profile ecg_preliminary_5alg_stride \
  --run-dir results/ecg_experiments/final_paper_runs/ecg_preliminary_5alg_stride \
  --no-build
```

STRIDE8 is enabled for every policy. A lower demand miss rate does not imply
lower bandwidth: compare `total_memory_traffic_with_overhead` and prefetch fills
alongside demand misses.

The current Sniper simple-prefetcher implementation does not export a
demand/prefetch NUCA miss split and expands total LLC read misses by 9x--596x
on this diagnostic. Treat its output as a rejected prefetch configuration, not
as demand-miss or speedup evidence.

When rerunning only one simulator or stage, use a distinct `--run-dir`.
`paper_run.py` refuses to replace a broader resolved manifest with an
`--only`/filtered subset. Aggregate shard directories together with
`paper_pipeline.py --input-run-dirs ...`.

First isolate replacement quality and online regret:

```bash
python3 scripts/experiments/ecg/flows/paper_run.py \
  --profile ecg_replacement_baseline \
  --run-dir results/ecg_experiments/final_paper_runs/ecg_replacement \
  --no-build
```

This PR/BFS/SSSP/BC/CC stage disables prefetching and uncharges ECG record delivery. It
reports LRU, SRRIP, GRASP, uncharged and charged P-OPT, K1, all five static K2
arms, and `ECG:K2_ONLINE`.

Before launching real graphs, certify all five algorithms:

```bash
python3 scripts/experiments/ecg/verify/equiv_kernels.py \
  --gem5 --sniper \
  --kernels pr bfs sssp bc cc \
  --schedule-k 2
```

BC certification applies K2 to the forward Brandes edge traversal only; its
runtime successor-DAG backward phase is not a static record stream. CC uses the
existing undirected/symmetric graph contract.

Then run the hardware-faithful placement factorial:

```bash
python3 scripts/experiments/ecg/flows/paper_run.py \
  --profile ecg_cache_sim_factorial \
  --run-dir results/ecg_experiments/final_paper_runs/ecg_factorial \
  --no-build
```

The factorial includes uncharged and charged P-OPT, K1/K2, StreamShield, and
online K2 with record traffic charged. Use
`--allow-missing-graphs --list --dry-run` to inspect the complete job set before
staging all three graphs.

To test adaptive placement on reused kernels with the full baseline set:

```bash
python3 scripts/experiments/ecg/flows/paper_run.py \
  --profile ecg_streamshield_generality \
  --run-dir results/ecg_experiments/final_paper_runs/ecg_streamshield_generality \
  --no-build
```

## Reproduce the detailed-simulator mechanism cells

```bash
python3 scripts/experiments/ecg/flows/paper_run.py \
  --profile gem5_streamshield_mechanism \
  --run-dir results/ecg_experiments/final_paper_runs/gem5_mechanism \
  --no-build

python3 scripts/experiments/ecg/flows/paper_run.py \
  --profile sniper_streamshield_mechanism \
  --run-dir results/ecg_experiments/final_paper_runs/sniper_mechanism \
  --no-build
```

## Full-iteration headline matrix (blocked)

```bash
python3 scripts/experiments/ecg/flows/paper_run.py \
  --profile streamshield_sniper_realgraph \
  --run-dir /tmp/ecg-successor-webgoogle-dryrun \
  --list --dry-run --no-build
```

Do not launch this profile until the manifest's `blocked_reason` is removed by
the prefetch-calibration milestone.

## Run local shards in parallel

All binaries must be prebuilt. The launcher gives every shard a unique run
directory and lock; roi_matrix derives isolated fixed-length gem5/Sniper
sideband directories from that output path.

```bash
python3 scripts/experiments/ecg/slurm/make_slurm_shards.py \
  --profile ecg_streamshield_generality \
  --run-tag ecg_generality_parallel \
  --out results/ecg_experiments/slurm/ecg_generality_parallel.tsv

python3 scripts/experiments/ecg/flows/run_local_shards.py \
  --shards results/ecg_experiments/slurm/ecg_generality_parallel.tsv \
  --run-root results/ecg_experiments/final_paper_runs/local \
  --jobs 8 \
  --cache-sim-jobs 8 \
  --gem5-jobs 1 \
  --sniper-jobs 1
```

`--jobs` is the global process cap. Per-simulator caps prevent gem5/Sniper
memory overcommit; raise them only on a machine sized for multiple simulators.
Interrupted shards are resumable because each shard is a normal `paper_run.py`
run with completion and content hashes.

## Generate one-policy Slurm shards after calibration

```bash
python3 -m venv .venv
.venv/bin/pip install -r scripts/requirements.txt
mkdir -p results/slurm_logs results/ecg_experiments/slurm

python3 scripts/experiments/ecg/slurm/make_slurm_shards.py \
  --profile streamshield_sniper_realgraph \
  --run-tag ecg_successor_webgoogle \
  --out results/ecg_experiments/slurm/ecg_successor_webgoogle.tsv \
  --allow-blocked
```

Submit on a configured cluster:

```bash
SHARDS=results/ecg_experiments/slurm/ecg_successor_webgoogle.tsv
COUNT=$(wc -l < "$SHARDS")
export SHARDS
sbatch --array=0-$((COUNT - 1))%16 \
  scripts/experiments/ecg/slurm/slurm_final_shard.sbatch
```

## Aggregate

Local completed runs:

```bash
python3 scripts/experiments/ecg/flows/paper_pipeline.py \
  --skip-run \
  --input-run-dirs \
    results/ecg_experiments/final_paper_runs/ecg_replacement \
    results/ecg_experiments/final_paper_runs/ecg_factorial \
    results/ecg_experiments/final_paper_runs/ecg_streamshield_generality \
  --run-root results/ecg_experiments/paper_pipeline/ecg_final

test -f \
  results/ecg_experiments/paper_pipeline/ecg_final/aggregate/online_dueling_regret.csv
```

Parallel local or Slurm shards:

```bash
python3 scripts/experiments/ecg/flows/paper_pipeline.py \
  --skip-run \
  --input-run-glob \
    "results/ecg_experiments/final_paper_runs/local/ecg_generality_parallel/*" \
  --run-root results/ecg_experiments/paper_pipeline/ecg_generality_parallel
```

The replacement profile emits
`aggregate/online_dueling_regret.csv`, which reports online K2's delta from the
best static arm using total LLC misses, plus a separate property-miss diagnostic
and deltas from uncharged and overhead-aware charged P-OPT.

## Correctness gates

```bash
python3 scripts/experiments/ecg/verify/equiv_kernels.py \
  --gem5 --sniper --kernels pr bfs sssp bc cc --schedule-k 2

python3 scripts/experiments/ecg/verify/equiv_kernels.py \
  --gem5 --sniper --kernels pr bfs sssp bc cc \
  --schedule-k 2 --stream-bypass

python3 scripts/experiments/ecg/verify/equiv_kernels.py \
  --gem5 --sniper --kernels pr bfs sssp bc cc \
  --schedule-k 2 --stream-bypass --adaptive-stream-bypass
```
