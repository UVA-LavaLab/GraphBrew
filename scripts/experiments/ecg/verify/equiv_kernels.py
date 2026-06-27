#!/usr/bin/env python3
"""Multi-kernel 3-simulator equivalence + full debug.

The eviction DECISION (`ecg_victim_policy.h`) is kernel-AGNOSTIC and byte-identical across
cache_sim / gem5 / Sniper, so the ECG policy must obey the same eviction spec for EVERY
kernel in EVERY simulator — not just PageRank. This runs PR / BFS / SSSP (the kernels all
three simulators share) on each simulator with the eviction trace on, asserts every L3
eviction obeys the policy spec (reusing `verify_ecg.py`'s `verify_trace`), AND captures the
per-sim `[ECG-CONFIG …]` banner (full debug: each run proves the policy/mode/variant it ran).

This certifies the DECISION equivalence across kernels. NOTE: the per-edge mask DIRECTION is
PR-tuned (out-neigh = PR's in-pull transpose); BFS-top-down/SSSP traverse out-edges and are
direction-UNCERTIFIED on directed graphs (a miss-rate, not a spec, concern — the spec holds
because the policy correctly evicts given whatever masks it has). See wiki/ECG-Policy-Comparison.md.

Usage:
  python3 scripts/experiments/ecg/verify/equiv_kernels.py                 # cache_sim only (fast)
  python3 scripts/experiments/ecg/verify/equiv_kernels.py --gem5 --sniper # full 3-sim (slow)
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ecg  # noqa: E402  (reuse verify_trace, BASE_ENV, ECG_ENV, COV_ENV, GRAPH, GEM5_OPT, ROI_MATRIX, ROOT)

BANNER_RE = re.compile(r"\[ECG-CONFIG[^\]]*\]")
# kernel -> simulators that can run it on the UNWEIGHTED eval graph (available binaries).
# SSSP is omitted: it needs a WEIGHTED (.wsg) graph (on email-Eu-core.sg it sees ~63 accesses
# and never pressures the L3); the eval corpus ships only unweighted .sg, and the converter has
# no weight-generation flag. BC runs unweighted (uses BFS internally) but Sniper's sg_kernel has
# no bc target, so BC is cache_sim + gem5 only.
KERNEL_SIMS = {
    "pr":  ["cache_sim", "gem5", "sniper"],
    "bfs": ["cache_sim", "gem5", "sniper"],
    "bc":  ["cache_sim", "gem5"],
}
GEM5_X86 = ecg.ROOT / "bench" / "include" / "gem5_sim" / "gem5" / "build" / "X86" / "gem5.opt"
GEM5_RISCV = ecg.ROOT / "bench" / "include" / "gem5_sim" / "gem5" / "build" / "RISCV" / "gem5.opt"
# Kernels whose gem5 leg runs on RISC-V via the validated fused ecg.load EVICT delivery
# (GEM5_FORCE_ECG_PLOAD). bc has no RISC-V m5op binary, so it stays on X86 (whose fat-mask
# m5op epoch delivery was fixed to the WIDE layout — see setup_gem5.py).
GEM5_RISCV_KERNELS = {"pr", "bfs"}


def _banner(text):
    m = BANNER_RE.search(text or "")
    return m.group(0) if m else "(no banner)"


def run_cache(kernel):
    """cache_sim <kernel> with ECG_GRASP_POPT + coverage geometry (force property eviction)."""
    binp = ecg.ROOT / "bench" / "bin_sim" / kernel
    if not binp.exists():
        return ("", False), "(binary missing)"
    env = {**os.environ, **ecg.BASE_ENV, **ecg.ECG_ENV, **ecg.COV_ENV,
           "ECG_VARIANT": "rrip_first", "ECG_DEBUG": "1"}
    p = subprocess.run([str(binp), "-f", str(ecg.GRAPH), "-o", "0", "-n", "1"],
                       env=env, capture_output=True, text=True, timeout=300)
    return (p.stderr, p.returncode == 0), _banner(p.stderr)


def _roi_log(out):
    logs = sorted((out / "logs").glob("*.log")) if (out / "logs").exists() else []
    text = logs[0].read_text(errors="ignore") if logs else ""
    return text, bool(text)


def run_gem5(kernel):
    """gem5 <kernel> with ECG_GRASP_POPT + coverage geometry. pr/bfs run on RISC-V via the
    validated fused ecg.load EVICT delivery (GEM5_FORCE_ECG_PLOAD); bc runs on X86 (its m5op
    fat-mask epoch delivery was fixed to the WIDE layout). The eviction DECISION is
    delivery-agnostic, but using the real epoch-delivery path makes the equivalence exercise
    the same stamped-epoch eviction the headline does."""
    out = Path("/tmp") / f"equivk_gem5_{kernel}"
    shutil.rmtree(out, ignore_errors=True)
    if kernel in GEM5_RISCV_KERNELS:
        env = {**os.environ, "GEM5_OPT": str(GEM5_RISCV), "GEM5_KERNEL_SUFFIX": "_riscv_m5ops",
               "GEM5_FORCE_ECG_PLOAD": "1", "GEM5_FORCE_ECG_EXTRACT": "1",
               "GEM5_ECG_PFX_MODE": "6", "ECG_PREFETCH_MODE": "6",
               "ECG_VARIANT": "rrip_first", "ECG_EVICT_TRACE": "4000", "ECG_STORED_REFRESH": "1",
               "ECG_DEBUG": "1"}
    else:
        env = {**os.environ, "GEM5_OPT": str(GEM5_X86), "GEM5_KERNEL_SUFFIX": "_m5ops",
               "GEM5_FORCE_ECG_EXTRACT": "1", "GEM5_ECG_PFX_MODE": "6", "ECG_PREFETCH_MODE": "6",
               "ECG_VARIANT": "rrip_first", "ECG_EVICT_TRACE": "4000", "ECG_STORED_REFRESH": "1",
               "ECG_DEBUG": "1"}
    cmd = [sys.executable, str(ecg.ROI_MATRIX), "--suite", "gem5", "--no-build",
           "--benchmark", kernel, "--policies", "ECG:ECG_GRASP_POPT",
           "--options", f"-f {ecg.GRAPH} -o 5 -n 1", "--l3-sizes", "4kB", "--l3-ways", "8",
           "--l1d-size", "2kB", "--l2-size", "1MB", "--out-dir", str(out)]
    subprocess.run(cmd, env=env, cwd=str(ecg.ROOT), stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, timeout=1200, check=False)
    text, ran = _roi_log(out)
    return (text, ran), _banner(text)


def run_sniper(kernel):
    """Sniper sg_kernel --benchmark <kernel> with ECG_GRASP_POPT (memory-capped, guarded)."""
    out = Path("/tmp") / f"equivk_sniper_{kernel}"
    shutil.rmtree(out, ignore_errors=True)
    env = {**os.environ, "SNIPER_ECG_MODE": "ECG_GRASP_POPT",
           "ECG_VARIANT": "rrip_first", "ECG_EVICT_TRACE": "40", "ECG_DEBUG": "1"}
    cmd = [sys.executable, str(ecg.ROI_MATRIX), "--suite", "sniper",
           "--sniper-workload", "sg_kernel", "--allow-sniper-sg-kernel-workload",
           "--sniper-memory-limit-gb", "20", "--sniper-enable-graph-policies", "--no-build",
           "--benchmark", kernel, "--policies", "ECG:ECG_GRASP_POPT",
           "--options", f"-f {ecg.GRAPH} -o 5 -n 1", "--l3-sizes", "16kB", "--l3-ways", "8",
           "--l1d-size", "2kB", "--l2-size", "4kB", "--timeout-sniper", "540", "--out-dir", str(out)]
    subprocess.run(cmd, env=env, cwd=str(ecg.ROOT), stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, timeout=900, check=False)
    text, ran = _roi_log(out)
    return (text, ran), _banner(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Multi-kernel 3-sim ECG equivalence + debug")
    ap.add_argument("--gem5", action="store_true", help="also run gem5 (X86; slower)")
    ap.add_argument("--sniper", action="store_true", help="also run Sniper (guarded; slowest)")
    ap.add_argument("--kernels", nargs="+", default=list(KERNEL_SIMS),
                    choices=list(KERNEL_SIMS))
    args = ap.parse_args(argv)

    enabled = {"cache_sim"}
    if args.gem5:
        missing = [str(p) for p in (GEM5_RISCV, GEM5_X86) if not p.exists()]
        if missing:
            print("FAIL: build gem5 first: " + ", ".join(missing)); return 2
        enabled.add("gem5")
    if args.sniper:
        enabled.add("sniper")
    RUNNERS = {"cache_sim": run_cache, "gem5": run_gem5, "sniper": run_sniper}
    sims_order = [s for s in ("cache_sim", "gem5", "sniper") if s in enabled]

    print("== Multi-kernel 3-sim ECG equivalence (eviction-spec + debug banner) ==")
    print(f"   graph={ecg.GRAPH.name}  policy=ECG:ECG_GRASP_POPT variant=rrip_first  "
          f"sims={sims_order}")
    print("   (SSSP omitted: needs a weighted .wsg graph; BC has no Sniper sg_kernel target)\n")

    ok_all = True
    results = {}   # (sim, kernel) -> status: 'ok' / 'spec-FAIL' / 'banner-X' / 'n/a'
    for kernel in args.kernels:
        for sim in sims_order:
            if sim not in KERNEL_SIMS[kernel]:
                results[(sim, kernel)] = "n/a"
                continue
            result, banner = RUNNERS[sim](kernel)
            spec_ok = ecg.verify_trace(f"{sim}/{kernel}", result)
            banner_ok = ("policy=ECG" in banner) and ("ECG_GRASP_POPT" in banner)
            print(f"      debug banner: {banner}  [{'OK' if banner_ok else 'MISSING'}]")
            status = "ok" if (spec_ok and banner_ok) else (
                "spec-FAIL" if not spec_ok else "banner-X")
            results[(sim, kernel)] = status
            ok_all &= (status == "ok")

    print("\n## kernel x sim matrix (ok = eviction-spec PASS + debug banner present)")
    hdr = "kernel".ljust(8) + "".join(s.ljust(14) for s in sims_order)
    print(hdr)
    for kernel in args.kernels:
        row = kernel.ljust(8)
        for sim in sims_order:
            row += results[(sim, kernel)].ljust(14)
        print(row)
    print(f"\nRESULT: {'ALL (kernel x sim): eviction-spec + debug banner OK ✓' if ok_all else 'see FAIL above'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
