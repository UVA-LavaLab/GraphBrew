# Gate 266 — Sniper overlay-installation tracker registry

Locks `scripts/setup_sniper.py`'s overlay installation contract (`.sniper_overlays.json` + `write_overlay_status` + `copy_overlay_sources` + `patch_*_overlay` functions) against silent drift in copied_files / policies / prefetchers / patches.

registry entries: 12 copied files; 3 policies (grasp, popt, ecg); 2 prefetchers (droplet, ecg_pfx); 5 patches (cache_base_replacement_policy_grasp, cache_set_factory_grasp_popt_ecg, cache_insert_prepare_insertion, prefetcher_factory_droplet, magic_user_graphbrew_hints).

## Rules

- **O1** — every copied_files entry has valid grammar (lower_snake_case path + .cc/.h)
- **O2** — every copied_files entry exists on disk under overlays/
- **O3** — every policy token has both cache_set_<pol>.cc + .h in copied_files
- **O4** — every prefetcher token has both <pf>_prefetcher.cc + .h in copied_files
- **O5** — every patches token has a patch_<token>_overlay function OR is in PATCH_NON_FUNCTION_ALLOW
- **O6** — on-disk .sniper_overlays.json matches canonical registry (copied_files+policies+prefetchers+patches)
- **O7** — copied_files is exhaustive: every regular file under overlays/ with tracked extension is listed (modulo README allow-list)

## Allow-lists

- `PATCH_NON_FUNCTION_ALLOW` = ['cache_insert_prepare_insertion', 'cache_set_factory_grasp_popt_ecg', 'magic_user_graphbrew_hints', 'prefetcher_factory_droplet']
- `OVERLAY_README_ALLOW` = ['README.md']

## ✅ No violations
