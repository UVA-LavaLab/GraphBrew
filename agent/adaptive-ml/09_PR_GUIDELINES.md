# PR Guidelines

---

## General Rules

- Keep PRs small and scoped.
- Every PR must include:
  - Reason + evidence
  - Tests or experiment script update
  - How to reproduce
- No refactors without measured benefit.
- Add flags rather than changing defaults first.
- Always preserve ORIGINAL behavior behind a flag.

---

## Adding New Algorithms / Variants

### Adding a new VIBE variant
1. **C++ side** (`reorder_vibe.h`): Add parsing in `reorderVIBE()` dispatch (~L6500+)
2. **Python side** (`scripts/lib/utils.py`): Add to `LEIDEN_CSR_VARIANTS` list
3. **Python side** (`scripts/lib/utils.py`): Add to `VIBE_LEIDEN_VARIANTS` or `VIBE_RABBIT_VARIANTS`
4. **Evaluate**: Follow the protocol in [03_WORKFLOW.md](03_WORKFLOW.md) (build → smoke → test → medium → large)

### Adding a new benchmark algorithm (e.g., new graph kernel)
1. Create `bench/src/mykernel.cc` using the `BenchmarkKernel(cli, g, kernel, print, verify)` pattern
2. Add to `KERNELS` in `Makefile`
3. Add the benchmark name to `BENCHMARKS` list in `scripts/lib/utils.py`
4. Rebuild: `make -j$(nproc)`
5. Use `--benchmarks pr bfs mykernel` to include in experiments
