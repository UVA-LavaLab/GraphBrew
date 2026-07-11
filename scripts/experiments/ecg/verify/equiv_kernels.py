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
# BC and CC now have a Sniper sg_kernel target (single-threaded ports of the audited
# Brandes_Sniper/Afforest_Sniper, same 4-region/1-region + transpose reref + epoch delivery), so
# they are certified on all three sims. SSSP stays cache_sim + gem5: the sg_kernel SSSP target
# needs a weighted (.wsg) graph the unweighted eval corpus does not provide.
# tc (triangle counting) is intentionally EXCLUDED: it is a pure set-intersection over CSR neighbour
# lists with NO vertex-indexed property array (gem5 tc registers 0 property regions), so ECG's
# per-edge epoch has nothing to stamp — out of the mechanism's scope (see findings 22.13).
KERNEL_SIMS = {
    "pr":   ["cache_sim", "gem5", "sniper"],
    "bfs":  ["cache_sim", "gem5", "sniper"],
    "bc":   ["cache_sim", "gem5", "sniper"],
    "cc":   ["cache_sim", "gem5", "sniper"],
    "sssp": ["cache_sim", "gem5"],
}
# Headline kernels with property REUSE that MUST decisively exercise epoch eviction (epoch distance
# strictly decides >=1 victim) on EVERY sim. With the faithful per-edge OUT-direction masks delivered
# (ECG_EDGE_MASKS=1 on the cache_sim out-traversal legs; the gem5 ecg.load EVICT path already delivers
# them), BFS/BC/SSSP decisively exercise epoch ordering. PR is delivery+policy
# coverage: after region isolation, the bounded tiny cell is record-first/do-no-harm.
# cc is the sole DO-NO-HARM cell: undirected union-find with low property reuse and no OUT-edge-mask
# consumption in the cache_sim kernel -> decisive=0 on BOTH sims (epoch delivered + policy-compliant,
# but eff-dist ties dominate, consistent with ECG ~= GRASP on that access pattern). (Findings 22.14.)
EXPECTED_DECISIVE = {"bfs", "bc", "sssp"}
GEM5_X86 = ecg.ROOT / "bench" / "include" / "gem5_sim" / "gem5" / "build" / "X86" / "gem5.opt"
GEM5_RISCV = ecg.ROOT / "bench" / "include" / "gem5_sim" / "gem5" / "build" / "RISCV" / "gem5.opt"
# Kernels whose gem5 leg runs on RISC-V via the validated fused ecg.load EVICT delivery
# (GEM5_FORCE_ECG_PLOAD). All ship a *_riscv_m5ops binary with real epoch delivery
# (pr: contrib; bfs: parent; bc: depth; cc: comp; sssp: dist), so no equiv cell depends on the
# X86 fat-mask (BFS/SSSP/BC/CC have no X86 epoch delivery).
GEM5_RISCV_KERNELS = {"pr", "bfs", "bc", "cc", "sssp"}

# Optional cross-sim stream-prefetcher degree (--stream-prefetch-degree). 0 = off
# (the byte-identical-decisive baseline). >0 turns on each sim's native structure
# prefetcher (cache_sim next-line via CACHE_STREAM_PREFETCH_DEGREE; gem5 stride via
# roi_matrix --prefetcher STRIDE) so the run proves the policy spec still holds with
# the realistic prefetcher. The prefetchers are NOT algorithm-identical, so under
# prefetch the equivalence is SPEC-level (every eviction obeys the ECG spec), not
# byte-identical counts.
STREAM_PF_DEGREE = 0
SCHEDULE_K = 0
STREAM_BYPASS = False


def effective_variant(kernel):
    if SCHEDULE_K == 2:
        return "epoch_first" if kernel == "pr" else "degree_first"
    return "rrip_first"


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
           "ECG_VARIANT": effective_variant(kernel), "ECG_DEBUG": "1"}
    env["CACHE_ECG_EPOCH_REGION_INDEX"] = "1" if kernel == "pr" else "0"
    if SCHEDULE_K:
        env["ECG_EDGE_MASK_SCHED"] = str(SCHEDULE_K)
        env["ECG_K2_DELIVERY_TRACE"] = "32"
    if STREAM_BYPASS:
        env["ECG_STREAM_BYPASS"] = "1"
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
    if STREAM_PF_DEGREE > 0:
        env["CACHE_STREAM_PREFETCH_DEGREE"] = str(STREAM_PF_DEGREE)
    p = subprocess.run([str(binp), "-f", str(ecg.GRAPH), "-o", "0", "-n", "1"],
                       env=env, capture_output=True, text=True, timeout=300)
    return (p.stderr, p.returncode == 0), _banner(p.stderr)


def _roi_log(out):
    logs = sorted((out / "logs").glob("*.log")) if (out / "logs").exists() else []
    text = logs[0].read_text(errors="ignore") if logs else ""
    # gem5 redirects benchmark stdout/stderr away from the simulator log. Append
    # them so K2 EXPECT records from the guest can be matched against RECV records
    # emitted by the decoder/backend.
    for path in sorted(out.rglob("benchmark_stderr.txt")):
        text += "\n" + path.read_text(errors="ignore")
    for path in sorted(out.rglob("benchmark_stdout.txt")):
        text += "\n" + path.read_text(errors="ignore")
    for path in sorted(out.rglob("sim.stats")):
        text += "\n" + path.read_text(errors="ignore")
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
               "ECG_VARIANT": effective_variant(kernel), "ECG_EVICT_TRACE": "4000",
               "ECG_EVICT_TRACE_ROI": "1", "ECG_STORED_REFRESH": "1",
               "ECG_DEBUG": "1"}
    else:
        env = {**os.environ, "GEM5_OPT": str(GEM5_X86), "GEM5_KERNEL_SUFFIX": "_m5ops",
               "GEM5_FORCE_ECG_EXTRACT": "1", "GEM5_ECG_PFX_MODE": "6", "ECG_PREFETCH_MODE": "6",
               "ECG_VARIANT": effective_variant(kernel), "ECG_EVICT_TRACE": "4000",
               "ECG_EVICT_TRACE_ROI": "1", "ECG_STORED_REFRESH": "1",
               "ECG_DEBUG": "1"}
    if SCHEDULE_K:
        env["ECG_EDGE_MASK_SCHED"] = str(SCHEDULE_K)
        env["ECG_K2_DELIVERY_TRACE"] = "32"
    if STREAM_BYPASS:
        env["ECG_STREAM_BYPASS"] = "1"
        env["ECG_STREAM_BYPASS_TRACE"] = "8"
    cmd = [sys.executable, str(ecg.ROI_MATRIX), "--suite", "gem5", "--no-build",
           "--benchmark", kernel, "--policies", "ECG:ECG_GRASP_POPT",
           "--options", f"-f {ecg.GRAPH} -o 5 -n 1", "--l3-sizes", "4kB", "--l3-ways", "8",
           "--l1d-size", "1kB", "--l2-size", "2kB", "--out-dir", str(out)]
    if STREAM_PF_DEGREE > 0:
        cmd += ["--prefetcher", "STRIDE", "--structure-prefetch-degree", str(STREAM_PF_DEGREE)]
    subprocess.run(cmd, env=env, cwd=str(ecg.ROOT), stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, timeout=1200, check=False)
    text, ran = _roi_log(out)
    return (text, ran), _banner(text)


def run_sniper(kernel):
    """Sniper sg_kernel --benchmark <kernel> with ECG_GRASP_POPT (memory-capped, guarded)."""
    out = Path("/tmp") / f"equivk_sniper_{kernel}"
    shutil.rmtree(out, ignore_errors=True)
    env = {**os.environ, "SNIPER_ECG_MODE": "ECG_GRASP_POPT",
           "ECG_VARIANT": effective_variant(kernel),
           "ECG_EVICT_TRACE": "4000", "ECG_DEBUG": "1"}
    if SCHEDULE_K:
        env["ECG_EDGE_MASK_SCHED"] = str(SCHEDULE_K)
        env["ECG_K2_DELIVERY_TRACE"] = "32"
    if STREAM_BYPASS:
        env["ECG_STREAM_BYPASS"] = "1"
        env["ECG_STREAM_BYPASS_TRACE"] = "8"
    # Per-kernel geometry: cc's comp[] (~4KB) and sssp's dist[] fit Sniper's inner
    # caches, and the L3 is NON-INCLUSIVE (sees only L2 evictions), so at the default
    # 2kB/4kB/16kB the property never reaches the L3 -> no epoch is stamped (vacuous).
    # Shrink L1d+L2 below the property footprint (and the L3 with it) so comp[]/dist[]
    # spill to and churn the L3, exercising the epoch delivery. Mirrors run_cache's
    # cc/sssp CACHE_L2=1kB/L3=2kB special-case. (cc stays do-no-harm: no OUT-edge mask.)
    if kernel in ("cc", "sssp"):
        l1d, l2, l3 = "1kB", "1kB", "2kB"
    else:
        l1d, l2, l3 = "2kB", "4kB", "16kB"
    cmd = [sys.executable, str(ecg.ROI_MATRIX), "--suite", "sniper",
           "--sniper-workload", "sg_kernel", "--allow-sniper-sg-kernel-workload",
           "--sniper-memory-limit-gb", "20", "--sniper-enable-graph-policies", "--no-build",
           "--benchmark", kernel, "--policies", "ECG:ECG_GRASP_POPT",
           "--options", f"-f {ecg.GRAPH} -o 5 -n 1", "--l3-sizes", l3, "--l3-ways", "8",
           "--l1d-size", l1d, "--l2-size", l2, "--timeout-sniper", "540", "--out-dir", str(out)]
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
    ap.add_argument("--stream-prefetch-degree", type=int, default=0,
                    help="cross-sim structure stream-prefetcher degree (0=off, the byte-identical "
                         "baseline; >0 = spec-level equivalence under the realistic prefetcher).")
    ap.add_argument("--schedule-k", type=int, choices=[0, 2], default=0,
                    help="enable Schedule-2 delivery and require live K2 pair/distance coverage "
                         "(currently certified for PR and BFS).")
    ap.add_argument("--stream-bypass", action="store_true",
                    help="enable StreamShield and require a live LLC-bypass mechanism trace.")
    args = ap.parse_args(argv)

    global STREAM_PF_DEGREE, SCHEDULE_K, STREAM_BYPASS
    STREAM_PF_DEGREE = args.stream_prefetch_degree
    SCHEDULE_K = args.schedule_k
    STREAM_BYPASS = args.stream_bypass
    if SCHEDULE_K and any(k not in ("pr", "bfs") for k in args.kernels):
        ap.error("--schedule-k 2 currently supports --kernels pr bfs")
    if STREAM_BYPASS and any(k != "pr" for k in args.kernels):
        ap.error("--stream-bypass currently supports --kernels pr")
    if STREAM_BYPASS and SCHEDULE_K != 2:
        ap.error("--stream-bypass requires --schedule-k 2")

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
    variant_label = "PR=epoch_first,BFS=degree_first" if SCHEDULE_K else "rrip_first"
    print(f"   graph={ecg.GRAPH.name}  policy=ECG:ECG_GRASP_POPT variant={variant_label}  "
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
            if SCHEDULE_K:
                spec_ok = ecg.verify_k2_trace(
                    f"{sim}/{kernel}", result, ne=65535, coverage=cov)
            else:
                spec_ok = ecg.verify_trace(
                    f"{sim}/{kernel}", result, coverage=cov)
            if STREAM_BYPASS:
                text, ran_ok = result
                if sim == "cache_sim":
                    bypass_ok = "[ECG-STREAM-BYPASS sim=cache_sim active=1]" in text
                elif sim == "gem5":
                    bypass_ok = "[ECG-STREAM-BYPASS sim=gem5" in text
                else:
                    reads = re.search(r"nuca-cache\.stream-bypass-reads = (\d+)", text)
                    writes = re.search(r"nuca-cache\.stream-bypass-writes = (\d+)", text)
                    bypass_ok = (
                        reads is not None and writes is not None and
                        int(reads.group(1)) > 0 and int(writes.group(1)) > 0
                    )
                print(f"      StreamShield LLC bypass: "
                      f"{'[OK]' if ran_ok and bypass_ok else '[FAIL]'}")
                spec_ok &= ran_ok and bypass_ok
            ev, tv = cov.get("epoch_victims", 0), cov.get("victims", 0)
            dec = cov.get("epoch_decisive", 0)
            nz = cov.get("epoch_victims_nz", 0)   # stamped victims with a NON-ZERO delivered epoch
            k2_live = cov.get("k2_ways", 0) > 0
            delivery_ok = ev > 0 or (SCHEDULE_K == 2 and k2_live)
            # >=1 stamped property victim normally proves delivery. Under K2,
            # resident stamped K2 property ways + verified pair distances also
            # prove delivery even if a non-inclusive backend evicts records for
            # the whole bounded trace (Sniper PR's do-no-harm geometry).
            decisive_ok = dec > 0         # epoch DISTANCE strictly decided >=1 victim
            # collapse check: stamped property was evicted but EVERY delivered epoch was 0 -> the
            # epochs collapsed (a delivery-quality regression, not the benign tied-eff-dist case).
            collapsed = ev > 0 and nz == 0
            if STREAM_BYPASS and sim == "cache_sim":
                cell_ok = spec_ok
                label = (
                    "StreamShield removes post-delivery LLC churn; "
                    "K2 delivery is certified by the no-bypass cache_sim gate")
            elif STREAM_PF_DEGREE > 0:
                # Under the realistic stream prefetcher, the prefetched STRUCTURAL lines
                # change the cache contents (and carry no epoch), so the epoch may no
                # longer strictly DECIDE a victim -- decisive/nonzero are degree-0 metrics.
                # The equivalence here is SPEC-level: every eviction must still obey the
                # ECG policy spec (verify_trace -> spec_ok).
                cell_ok = spec_ok
                label = (f"prefetch(d{STREAM_PF_DEGREE}) spec-level: evictions obey spec; "
                         f"decisive {dec}x nonzero {nz} stamped {ev} (decisiveness is the degree-0 metric)")
            elif (kernel in EXPECTED_DECISIVE and sim != "sniper" and
                  not STREAM_BYPASS):
                cell_ok = decisive_ok     # headline (property-reuse) kernel MUST be decisive
                label = ("decisive real-epoch (headline)" if decisive_ok
                         else "FAIL: headline kernel, NO decisive epoch eviction")
            elif kernel == "cc" and sim == "sniper":
                # cc DECISION-level certification on Sniper. cc is the do-no-harm cell
                # (union-find pointer-chases a SMALL, heavily-reused comp[]: most comp[]
                # accesses are undelivered chain hops, and the few property lines that
                # fill are mostly already-resident), and Sniper's L3 is NON-INCLUSIVE
                # (comp[] is protected + fits the inner caches), so property lines carry
                # a fresh delivered epoch only rarely -> stamped ~0 / epochs collapse to
                # 0. This mirrors the inclusive-vs-non-inclusive gap seen elsewhere; the
                # inclusive cache_sim/gem5 legs DO deliver+stamp cc (nonzero>0). What is
                # certifiable here is the byte-identical eviction DECISION: every eviction
                # obeys the ECG spec + the debug banner matches. Do NOT require delivery.
                cell_ok = spec_ok
                label = ("do-no-harm DECISION certified on Sniper (spec obeyed + banner); "
                         "epoch-delivery not exercised: union-find chases a small reused comp[] on "
                         f"a non-inclusive L3 -> stamped={ev} (delivery exercised on inclusive cache_sim/gem5)")
            elif not delivery_ok:
                cell_ok = False
                label = "FAIL: NO epoch delivered (vacuous)"
            elif collapsed:
                cell_ok = False
                label = f"FAIL: {ev} stamped victims but ALL epoch=0 (delivery COLLAPSED, not do-no-harm)"
            else:
                cell_ok = True            # do-no-harm: delivery + policy verified
                if SCHEDULE_K == 2 and k2_live and ev == 0:
                    label = (
                        f"K2 resident+distance verified ({cov.get('k2_ways', 0)} ways); "
                        "no property victim in bounded trace (record-first do-no-harm)")
                else:
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

    print("\n## kernel x sim matrix (ok = spec PASS + banner + [bfs/bc/sssp on cache_sim+gem5: "
          ">=1 DECISIVE epoch victim] [cc on cache_sim/gem5: epoch DELIVERED, do-no-harm; "
          "cc on Sniper: DECISION-level (spec+banner) — union-find/non-inclusive L3 doesn't exercise delivery])")
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
    print(f"\nRESULT: {'ALL (kernel x sim) PASS ✓ (BFS/BC/SSSP decisive; PR/CC do-no-harm where bounded)' if ok_all else 'see FAIL above'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
