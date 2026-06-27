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
# SSSP runs on the unweighted email-Eu-core.sg via GAPBS WeightedBuilder (it deterministically
# synthesizes edge weights — no .wsg needed) and, at the cc small-cache geometry (L2 1kB/L3 2kB),
# its dist[] spills to the L3 so the epoch eviction is exercised (banner + nonzero stamped epochs).
# BC runs unweighted (uses BFS internally) but Sniper's sg_kernel has no bc target, so BC/CC/SSSP
# are cache_sim + gem5 (gem5 via the validated RISC-V ecg.load EVICT delivery on dist[]).
# tc (triangle counting) is intentionally EXCLUDED: it is a pure set-intersection over CSR neighbour
# lists with NO vertex-indexed property array (gem5 tc registers 0 property regions), so ECG's
# per-edge epoch has nothing to stamp — out of the mechanism's scope (see findings 22.13).
KERNEL_SIMS = {
    "pr":   ["cache_sim", "gem5", "sniper"],
    "bfs":  ["cache_sim", "gem5", "sniper"],
    "bc":   ["cache_sim", "gem5"],
    "cc":   ["cache_sim", "gem5"],
    "sssp": ["cache_sim", "gem5"],
}
# Headline kernels with property REUSE that MUST decisively exercise epoch eviction (epoch distance
# strictly decides >=1 victim) on EVERY sim. With the faithful per-edge OUT-direction masks delivered
# (ECG_EDGE_MASKS=1 on the cache_sim out-traversal legs; the gem5 ecg.load EVICT path already delivers
# them), pr/bfs/bc/sssp are all decisive on both sims (cache_sim 2/1/123/5, gem5 17/14/27/3) -> these
# cells prove the EPOCH ordering (not just record/recency) is what selects victims, cross-sim.
# cc is the sole DO-NO-HARM cell: undirected union-find with low property reuse and no OUT-edge-mask
# consumption in the cache_sim kernel -> decisive=0 on BOTH sims (epoch delivered + policy-compliant,
# but eff-dist ties dominate, consistent with ECG ~= GRASP on that access pattern). (Findings 22.14.)
EXPECTED_DECISIVE = {"pr", "bfs", "bc", "sssp"}
GEM5_X86 = ecg.ROOT / "bench" / "include" / "gem5_sim" / "gem5" / "build" / "X86" / "gem5.opt"
GEM5_RISCV = ecg.ROOT / "bench" / "include" / "gem5_sim" / "gem5" / "build" / "RISCV" / "gem5.opt"
# Kernels whose gem5 leg runs on RISC-V via the validated fused ecg.load EVICT delivery
# (GEM5_FORCE_ECG_PLOAD). All ship a *_riscv_m5ops binary with real epoch delivery
# (pr: contrib; bfs: parent; bc: depth; cc: comp; sssp: dist), so no equiv cell depends on the
# X86 fat-mask (BFS/SSSP/BC/CC have no X86 epoch delivery).
GEM5_RISCV_KERNELS = {"pr", "bfs", "bc", "cc", "sssp"}


def _banner(text):
    m = BANNER_RE.search(text or "")
    return m.group(0) if m else "(no banner)"


def _stale(binp, kernel):
    """Return a [STALE] note if bin_sim/<kernel> predates the ECG headers/policy/kernel source it
    is built from (the cc/sssp banner trap: a binary built before a header change silently runs old
    logic). Empty string when fresh. Guards the equiv against a stale-binary false pass/fail."""
    if not binp.exists():
        return ""
    bmt = binp.stat().st_mtime
    inc = ecg.ROOT / "bench" / "include"
    deps = [inc / "cache_sim" / "cache_sim.h", inc / "ecg_victim_policy.h",
            inc / "ecg_epoch_builder.h", inc / "ecg_mode6_builder.h",
            inc / "cache_sim" / "graph_cache_context.h",
            ecg.ROOT / "bench" / "src_sim" / f"{kernel}.cc"]
    newer = [d.name for d in deps if d.exists() and d.stat().st_mtime > bmt]
    return f"  [STALE] bin_sim/{kernel} older than {', '.join(newer)} — rebuild (make bench/bin_sim/{kernel})\n" if newer else ""


def run_cache(kernel):
    """cache_sim <kernel> with ECG_GRASP_POPT + coverage geometry (force property eviction)."""
    binp = ecg.ROOT / "bench" / "bin_sim" / kernel
    if not binp.exists():
        return ("", False), "(binary missing)"
    stale = _stale(binp, kernel)
    if stale:
        sys.stderr.write(stale)
    env = {**os.environ, **ecg.BASE_ENV, **ecg.ECG_ENV, **ecg.COV_ENV,
           "ECG_VARIANT": "rrip_first", "ECG_DEBUG": "1"}
    if kernel in ("bfs", "bc", "sssp"):
        # Out-traversal kernels read property[dest] over out_neigh(u); deliver the FAITHFUL per-edge
        # OUT-direction next-ref masks (ECG_EDGE_MASKS=1, epoch = next in_neigh(dest) > u) — the same
        # direction the gem5 ecg.load EVICT leg delivers. This makes the epoch STRICTLY DECIDE victims
        # on cache_sim (sssp 5, bfs 1, bc 123), matching the already-decisive gem5 legs, so the cell
        # proves epoch-equivalence (not just delivery). (Default PR mode-6 in-edge env stays for pr.)
        env["ECG_EDGE_MASKS"] = "1"
    if kernel in ("cc", "sssp"):
        # cc-Afforest's comp[] (~4KB) and sssp's dist[] fit the 1MB COV L2 -> never reach L3
        # (PR's contrib is re-read every iteration so it churns; gem5 works because its full-ISA
        # stream churns the L3). Shrink the cache_sim L2+L3 below the property footprint so the
        # epoch eviction is exercised. (cc has no OUT-edge-mask consumption -> stays do-no-harm.)
        env["CACHE_L2_SIZE"] = "1kB"
        env["CACHE_L3_SIZE"] = "2kB"
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
    print("   (BC/CC/SSSP: cache_sim + gem5; SSSP weights auto-synthesized by WeightedBuilder)\n")

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
            nz = cov.get("epoch_victims_nz", 0)   # stamped victims with a NON-ZERO delivered epoch
            delivery_ok = ev > 0          # >=1 stamped property line was evicted (epoch DELIVERED)
            decisive_ok = dec > 0         # epoch DISTANCE strictly decided >=1 victim
            # collapse check: stamped property was evicted but EVERY delivered epoch was 0 -> the
            # epochs collapsed (a delivery-quality regression, not the benign tied-eff-dist case).
            collapsed = delivery_ok and nz == 0
            if kernel in EXPECTED_DECISIVE and sim != "sniper":
                cell_ok = decisive_ok     # headline (property-reuse) kernel MUST be decisive
                label = ("decisive real-epoch (headline)" if decisive_ok
                         else "FAIL: headline kernel, NO decisive epoch eviction")
            elif not delivery_ok:
                cell_ok = False
                label = "FAIL: NO epoch delivered (vacuous)"
            elif collapsed:
                cell_ok = False
                label = f"FAIL: {ev} stamped victims but ALL epoch=0 (delivery COLLAPSED, not do-no-harm)"
            else:
                cell_ok = True            # do-no-harm: delivery + policy verified
                label = (f"delivery+policy verified; epoch decisive {dec}x, nonzero {nz}/{ev}"
                         + ("" if decisive_ok else " (do-no-harm: tied eff-dist -> epoch seldom decisive)"))
            print(f"      epoch coverage: decisive={dec} nonzero={nz} stamped={ev} / {tv} total  [{label}]")
            banner_ok = ("policy=ECG" in banner) and ("ECG_GRASP_POPT" in banner)
            print(f"      debug banner: {banner}  [{'OK' if banner_ok else 'MISSING'}]")
            if not spec_ok:
                status = "spec-FAIL"
            elif not banner_ok:
                status = "banner-X"
            elif not cell_ok:
                if kernel in EXPECTED_DECISIVE and sim != "sniper":
                    status = "FAIL-dec0"
                elif collapsed:
                    status = "FAIL-collapse"
                else:
                    status = "FAIL-nodeliv"
            else:
                status = "ok"
            results[(sim, kernel)] = status
            ok_all &= (status == "ok")

    print("\n## kernel x sim matrix (ok = spec PASS + banner + [pr/bfs/bc/sssp on cache_sim+gem5: "
          ">=1 DECISIVE epoch victim] [cc + any sniper leg: epoch DELIVERED, do-no-harm])")
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
    print(f"\nRESULT: {'ALL (kernel x sim) PASS ✓ (pr/bfs/bc/sssp DECISIVE on cache_sim+gem5; cc do-no-harm)' if ok_all else 'see FAIL above'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
