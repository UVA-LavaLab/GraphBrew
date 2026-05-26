# Sniper Prefetcher Overlays

This directory documents GraphBrew Sniper graph-prefetcher overlays. Sniper's
active prefetcher factory lives in `parametric_dram_directory_msi`, so tracked
prefetcher implementations follow that upstream path even though this README
keeps the prefetch work grouped conceptually.

Tracked implementations:

```text
common/core/memory_subsystem/parametric_dram_directory_msi/droplet_prefetcher.{h,cc}
common/core/memory_subsystem/parametric_dram_directory_msi/ecg_pfx_prefetcher.{h,cc}
```

The Sniper DROPLET path must match the active gem5 DROPLET semantics:

- use actual CSR edge shadow data,
- detect edge-stream cache lines,
- resolve future neighbor IDs to property-region prefetch addresses,
- expose issued/useful/late/unused counters when possible.

The Sniper ECG_PFX path is separate from DROPLET:

- benchmarks emit `SNIPER_ECG_PFX_TARGET(vertex)` hints,
- `scripts/setup_sniper.py --apply-overlays` patches `magic_server.cc` so the
	`GPFX` SimUser command stores per-core prefetch targets,
- `ecg_pfx_prefetcher` consumes those stored targets,
- targets are resolved to the exported property-region sideband,
- recent cache lines are deduplicated,
- `ecg-pfx-prefetcher.*` counters report sideband load, hints seen, requests
	issued, duplicate skips, and invalid/no-sideband cases.

Do not use this path for final claims until those counters prove the prefetcher
is active on the relevant graph/benchmark rows.

Current validated status:

- `scripts/setup_sniper.py --apply-overlays` wires `prefetcher = droplet` and
	`prefetcher = ecg_pfx` into `Prefetcher::createPrefetcher()`.
- `roi_matrix.py --suite sniper --prefetcher DROPLET` and
	`--prefetcher ECG_PFX` attach the selected prefetcher to `l2` by default or
	`l1d` when requested.
- `sniper_sift_ecg_pfx_smoke` validates synthetic PR/BFS/SSSP ECG_PFX rows.
- `sniper_sift_file_ecg_pfx_smoke` validates email-Eu-core PR/BFS/SSSP ECG_PFX
	rows using lookahead 4 for PR/BFS and lookahead 0 for SSSP.
