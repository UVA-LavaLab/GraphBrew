# Gate 267 — gem5 overlay-installation tracker registry

Locks `scripts/setup_gem5.py`'s overlay installation contract (`OVERLAY_FILE_MAP` + `PATCH_FILES` + `apply_overlays()` + `apply_patches()`) against silent drift in source files / policies / prefetchers / patches.

registry entries: 14 overlay sources; 3 policies (grasp, popt, ecg); 2 prefetchers (droplet, ecg_pfx); 2 patches.

## Rules

- **G1** — every OVERLAY_FILE_MAP source has valid grammar (path + .cc/.hh/.py/.isa)
- **G2** — every OVERLAY_FILE_MAP source exists on disk under overlays/
- **G3** — every policy token has both <pol>_rp.cc + .hh in OVERLAY_FILE_MAP
- **G4** — every prefetcher token has both <pf>.cc + .hh in OVERLAY_FILE_MAP
- **G5** — every PATCH_FILES entry exists on disk under overlays/
- **G6** — OVERLAY_FILE_MAP+PATCH_FILES is exhaustive over overlays/ (modulo OVERLAY_EXTRA_ALLOW)
- **G7** — live setup_gem5.OVERLAY_FILE_MAP+PATCH_FILES matches canonical registry (parity + identity invariant)

## Allow-lists

- `OVERLAY_EXTRA_ALLOW` = ['arch/riscv/isa/decoder_ecg_extract.isa']

## ✅ No violations
