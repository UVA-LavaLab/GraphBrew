# Gate 268 — Setup-script invariant registry

Locks `scripts/setup_gem5.py` and `scripts/setup_sniper.py` against silent drift in upstream repo URLs, directory-constant skeleton, and the canonical top-level function inventory each script exposes.

registry entries: 2 repo URLs; 6 gem5 dir-constants + 15 functions; 6 sniper dir-constants + 27 functions.

## Rules

- **S1** — GEM5_REPO_URL and SNIPER_REPO_URL constants exist and equal canonical values
- **S2** — every required directory constant is present in each script
- **S3** — every canonical gem5 entry-point function is present in setup_gem5.py
- **S4** — every canonical sniper entry-point function is present in setup_sniper.py
- **S5** — both scripts define def main( (CLI entry point invariant)
- **S6** — both scripts define def apply_overlays( (overlay-install contract)
- **S7** — actual top-level def set equals canonical registry exactly (no unregistered helpers)

## Allow-lists

- `SETUP_GEM5_EXTRA_ALLOW` = []
- `SETUP_SNIPER_EXTRA_ALLOW` = []

## ✅ No violations
