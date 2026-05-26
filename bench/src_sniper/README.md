# Sniper Benchmark Wrappers

This directory will hold Sniper-oriented GraphBrew benchmark wrappers.

Current Phase 0 smoke:

```bash
make sniper-hello_roi
bench/bin_sniper/hello_roi
make sniper-pr_kernel_smoke
bench/include/sniper_sim/snipersim/run-sniper --roi --no-cache-warming \
	-n 1 -d /tmp/sniper-graphbrew-pr-kernel-smoke \
	-caddress_translation_schemes/baseline -- bench/bin_sniper/pr_kernel_smoke
python3 bench/include/sniper_sim/scripts/parse_stats.py \
	/tmp/sniper-graphbrew-pr-kernel-smoke
```

`hello_roi.cc` verifies that `bench/include/sniper_sim/sniper_harness.h` can be
included before and after the upstream Sniper checkout exists. It calls ROI
start/end macros and writes a minimal sideband file to
`/tmp/sniper_graphbrew_ctx.json`.

`pr_kernel_smoke.cc`, `bfs_kernel_smoke.cc`, and `sssp_kernel_smoke.cc` are tiny
fixed-graph kernels that avoid the full GraphBrew builder startup cost. They
export minimal context plus P-OPT matrix sidebands, enter a small ROI, and are
the quick Sniper smoke targets until the full wrappers have a tuned SDE/SIFT run
mode.

Initial full wrappers now exist for `pr`, `bfs`, and `sssp`. They start from the
audited `bench/src_gem5` kernels, replace gem5 m5ops with Sniper ROI/hint macros,
and preserve sideband and P-OPT matrix exports. They currently build and run
natively, but full Sniper/SDE startup is too slow for them to be the default
smoke until the runner mode is tuned. Do not use full `pr` as a Sniper smoke in
the current SDE/SIFT mode: a bounded tiny run reached about 53 GiB RSS before it
was killed.

Runner smoke:

```bash
python3 scripts/experiments/ecg/roi_matrix.py \
	--suite sniper --policies LRU SRRIP --l3-sizes 32kB

python3 scripts/experiments/ecg/final_paper_run.py \
	--profile sniper_kernel_smoke \
	--run-dir /tmp/graphbrew-final-sniper-kernel-smoke \
	--no-build --force
```

The `sniper_kernel_smoke` profile currently runs PR/BFS/SSSP with the full final
policy-label surface through the safe kernel-smoke workload.

The `sniper_droplet_smoke` profile runs the safe virtual-address DROPLET smoke
for PR/BFS, where the Sniper counters show issued and useful prefetches.

`sg_kernel.cc` is a native diagnostic target for `.sg` graph loading plus
kernel-only ROI sideband export. Native runs are clean on tiny, email-Eu-core,
and cit-Patents inputs, but running it under Sniper/SDE repeated the high-memory
child-process issue. Keep it native-only unless doing tightly bounded frontend
debugging.

Native validation:

```bash
make sniper-pr sniper-bfs sniper-sssp
bench/bin_sniper/pr   -g 8 -k 16 -o 5 -n 1 -i 1
bench/bin_sniper/bfs  -g 4 -k 2  -o 0 -n 1 -r 0
bench/bin_sniper/sssp -g 4 -k 2  -o 0 -n 1 -r 0 -d 1
```
