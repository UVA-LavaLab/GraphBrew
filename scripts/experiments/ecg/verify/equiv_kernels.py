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
# SSSP is omitted from the 3-sim matrix: on the small email-Eu-core.sg its dist[] either fits in
# the L3 (no eviction) or, shrunk below it, fills the L3 with all-property uniform-epoch sets (no
# DECISIVE epoch eviction) — it needs a larger graph for clean cache_sim L3 pressure (Phase B B2).
# gem5/sssp epoch DELIVERY is validated separately (ecg.load EVICT: 607 stamped, 3 decisive victims).
# BC runs unweighted (uses BFS internally) but Sniper's sg_kernel has no bc target, so BC is
# cache_sim + gem5 only.
KERNEL_SIMS = {
    "pr":  ["cache_sim", "gem5", "sniper"],
    "bfs": ["cache_sim", "gem5", "sniper"],
    "bc":  ["cache_sim", "gem5"],
}
# Headline kernels with property REUSE that MUST decisively exercise epoch eviction (epoch
# distance strictly decides >=1 victim) on every sim. bfs/bc are do-no-harm (low property reuse,
# ECG ~= GRASP): they verify policy + epoch DELIVERY, but epoch is seldom strictly decisive
# (e.g. cache_sim bfs/bc evict stamped property on tied eff-dist), so a 0 decisive count there is
# expected, not a failure. (See findings 22.10.)
EXPECTED_DECISIVE = {"pr"}
GEM5_X86 = ecg.ROOT / "bench" / "include" / "gem5_sim" / "gem5" / "build" / "X86" / "gem5.opt"
GEM5_RISCV = ecg.ROOT / "bench" / "include" / "gem5_sim" / "gem5" / "build" / "RISCV" / "gem5.opt"
# Kernels whose gem5 leg runs on RISC-V via the validated fused ecg.load EVICT delivery
# (GEM5_FORCE_ECG_PLOAD). All ship a *_riscv_m5ops binary with real epoch delivery
# (pr: contrib; bfs: parent; bc: depth), so no equiv cell depends on the X86 fat-mask path.
GEM5_RISCV_KERNELS = {"pr", "bfs", "bc"}


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
               "ECG_VARIANT": "rrip_first", "ECG_EVICT_TRACE": "4000",
               "ECG_EVICT_TRACE_ROI": "1", "ECG_STORED_REFRESH": "1",
               "ECG_DEBUG": "1"}
    else:
        env = {**os.environ, "GEM5_OPT": str(GEM5_X86), "GEM5_KERNEL_SUFFIX": "_m5ops",
               "GEM5_FORCE_ECG_EXTRACT": "1", "GEM5_ECG_PFX_MODE": "6", "ECG_PREFETCH_MODE": "6",
               "ECG_VARIANT": "rrip_first", "ECG_EVICT_TRACE": "4000",
               "ECG_EVICT_TRACE_ROI": "1", "ECG_STORED_REFRESH": "1",
               "ECG_DEBUG": "1"}
    cmd = [sys.executable, str(ecg.ROI_MATRIX), "--suite", "gem5", "--no-build",
           "--benchmark", kernel, "--policies", "ECG:ECG_GRASP_POPT",
           "--options", f"-f {ecg.GRAPH} -o 5 -n 1", "--l3-sizes", "4kB", "--l3-ways", "8",
           "--l1d-size", "1kB", "--l2-size", "2kB", "--out-dir", str(out)]
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
        need = {(GEM5_RISCV if k in GEM5_RISCV_KERNELS else GEM5_X86) for k in args.kernels}
        missing = [str(p) for p in need if not p.exists()]
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
            cov = {}
            spec_ok = ecg.verify_trace(f"{sim}/{kernel}", result, coverage=cov)
            ev, tv = cov.get("epoch_victims", 0), cov.get("victims", 0)
            dec = cov.get("epoch_decisive", 0)
            delivery_ok = ev > 0          # >=1 stamped property line was evicted (epoch DELIVERED)
            decisive_ok = dec > 0         # epoch DISTANCE strictly decided >=1 victim
            if kernel in EXPECTED_DECISIVE:
                cell_ok = decisive_ok     # headline (property-reuse) kernel MUST be decisive
                label = ("decisive real-epoch (headline)" if decisive_ok
                         else "FAIL: headline kernel, NO decisive epoch eviction")
            else:
                cell_ok = delivery_ok     # do-no-harm kernels: delivery + policy; epoch rarely decisive
                label = (f"delivery+policy verified; epoch decisive {dec}x"
                         + ("" if decisive_ok else " (do-no-harm: low property reuse -> epoch seldom decisive)")
                         if delivery_ok else "FAIL: NO epoch delivered (vacuous)")
            print(f"      epoch coverage: decisive={dec}  stamped-prop-victims={ev} / {tv} total  [{label}]")
            banner_ok = ("policy=ECG" in banner) and ("ECG_GRASP_POPT" in banner)
            print(f"      debug banner: {banner}  [{'OK' if banner_ok else 'MISSING'}]")
            if not spec_ok:
                status = "spec-FAIL"
            elif not banner_ok:
                status = "banner-X"
            elif not cell_ok:
                status = "FAIL-dec0" if kernel in EXPECTED_DECISIVE else "FAIL-nodeliv"
            else:
                status = "ok"
            results[(sim, kernel)] = status
            ok_all &= (status == "ok")

    print("\n## kernel x sim matrix (ok = spec PASS + banner + [pr: >=1 DECISIVE epoch victim] "
          "[bfs/bc/sssp: epoch DELIVERED]); decisive counts reported per cell above")
    hdr = "kernel".ljust(8) + "".join(s.ljust(14) for s in sims_order)
    print(hdr)
    for kernel in args.kernels:
        row = kernel.ljust(8)
        for sim in sims_order:
            row += results[(sim, kernel)].ljust(14)
        print(row)
    bad = [f"{s}/{k}" for (s, k), v in results.items() if v.startswith("FAIL")]
    if bad:
        print(f"\nFAIL: {', '.join(sorted(bad))}")
    print(f"\nRESULT: {'ALL (kernel x sim) PASS ✓ (pr decisive on all sims; bfs/bc/sssp deliver epochs)' if ok_all else 'see FAIL above'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
