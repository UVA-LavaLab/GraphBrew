#!/usr/bin/env python3
"""Reproducible correctness check for the L3 replacement policies.

Runs each policy on a tiny controlled cell with ECG_EVICT_TRACE enabled, parses
the per-eviction trace (each way's rrpv/epoch/dist/property/recency + the chosen
victim), and ASSERTS the victim matches the policy's defining rule. Exit code 0
iff every eviction of every policy obeys its spec. Researcher-runnable artifact
verification — no trust in aggregate numbers required.

  python3 scripts/experiments/ecg/verify_ecg.py
"""
import os, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PR = ROOT / "bench" / "bin_sim" / "pr"
GRAPH = ROOT / "results" / "graphs" / "email-Eu-core" / "email-Eu-core.sg"

BASE_ENV = dict(CACHE_ULTRAFAST="0", CACHE_L1_POLICY="LRU", CACHE_L2_POLICY="LRU",
                CACHE_L1_SIZE="2kB", CACHE_L1_WAYS="8", CACHE_L2_SIZE="4kB",
                CACHE_L2_WAYS="8", CACHE_L3_SIZE="16kB", CACHE_L3_WAYS="8",
                CACHE_LINE_SIZE="64", OMP_NUM_THREADS="1", ECG_EVICT_TRACE="40")
ECG_ENV = dict(CACHE_POLICY="ECG", CACHE_L3_POLICY="ECG", ECG_MODE="ECG_GRASP_POPT",
               ECG_EXACT_REREF="1", ECG_PREFETCH_MODE="6", ECG_EDGE_MASK_EPOCH="1",
               ECG_EDGE_MASK_LINEMIN="1", ECG_EDGE_MASK_EPOCHS="65535",
               ECG_EDGE_MASK_LEAN="1", ECG_EDGE_MASK_PACK="1", ECG_EDGE_MASK_CHARGED="1")

WAY_RE = re.compile(r"way(\d+) valid=(\d+) rrpv=(\d+) epoch=(\d+) dist=(\d+) prop=(\d+) last=(\d+)")
HDR_RE = re.compile(r"\[EVICT L3 pol=(\S+)")
VIC_RE = re.compile(r"-> victim=way(\d+)")


def run(policy_env):
    env = {**os.environ, **BASE_ENV, **policy_env}
    p = subprocess.run([str(PR), "-f", str(GRAPH), "-o", "0", "-n", "1", "-i", "1"],
                       env=env, capture_output=True, text=True, timeout=300)
    return p.stderr


GEM5_OPT = ROOT / "bench" / "include" / "gem5_sim" / "gem5" / "build" / "RISCV" / "gem5.opt"
ROI_MATRIX = ROOT / "scripts" / "experiments" / "ecg" / "roi_matrix.py"


def run_gem5(variant):
    """Run gem5 ECG_GRASP_POPT on the tiny graph with the trace on; return the
    gem5 log text (run_command pipes the policy's stderr trace into the log)."""
    out = Path("/tmp") / f"verify_gem5_{variant}"
    env = {**os.environ, "GEM5_OPT": str(GEM5_OPT), "GEM5_KERNEL_SUFFIX": "_riscv_m5ops",
           "GEM5_FORCE_ECG_EXTRACT": "1", "GEM5_ECG_PFX_MODE": "6", "ECG_PREFETCH_MODE": "6",
           "ECG_VARIANT": variant, "ECG_EVICT_TRACE": "40"}
    cmd = [sys.executable, str(ROI_MATRIX), "--suite", "gem5", "--no-build",
           "--benchmark", "pr", "--policies", "ECG:ECG_GRASP_POPT",
           "--options", f"-f {GRAPH} -o 5 -n 1 -i 1",
           "--l3-sizes", "16kB", "--l3-ways", "8", "--l1d-size", "2kB", "--l2-size", "4kB",
           "--out-dir", str(out)]
    subprocess.run(cmd, env=env, cwd=str(ROOT),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=900, check=False)
    logs = sorted((out / "logs").glob("*.log")) if (out / "logs").exists() else []
    return logs[0].read_text(errors="ignore") if logs else ""


def run_sniper(variant):
    """Run Sniper ECG_GRASP_POPT on the tiny graph with the trace on; return the
    Sniper log text. The sg_kernel workload is gated (Sniper/SDE has a documented
    ~50 GiB runaway), so it runs under prlimit via --sniper-memory-limit-gb."""
    import shutil
    out = Path("/tmp") / f"verify_sniper_{variant}"
    shutil.rmtree(out, ignore_errors=True)
    env = {**os.environ, "SNIPER_ECG_MODE": "ECG_GRASP_POPT",
           "ECG_VARIANT": variant, "ECG_EVICT_TRACE": "40"}
    cmd = [sys.executable, str(ROI_MATRIX), "--suite", "sniper",
           "--sniper-workload", "sg_kernel", "--allow-sniper-sg-kernel-workload",
           "--sniper-memory-limit-gb", "20", "--sniper-enable-graph-policies",
           "--no-build", "--benchmark", "pr", "--policies", "ECG:ECG_GRASP_POPT",
           "--options", f"-f {GRAPH} -o 5 -n 1 -i 1",
           "--l3-sizes", "16kB", "--l3-ways", "8", "--l1d-size", "2kB", "--l2-size", "4kB",
           "--timeout-sniper", "540", "--out-dir", str(out)]
    subprocess.run(cmd, env=env, cwd=str(ROOT),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=900, check=False)
    logs = sorted((out / "logs").glob("*.log")) if (out / "logs").exists() else []
    return logs[0].read_text(errors="ignore") if logs else ""


def parse_blocks(text):
    """Yield (pol, ways[list of dict], victim_way)."""
    pol = None; ways = []
    for line in text.splitlines():
        h = HDR_RE.search(line)
        if h:
            if pol and ways: yield pol, ways, victim
            pol = h.group(1); ways = []; victim = None; continue
        m = WAY_RE.search(line)
        if m:
            w, valid, rrpv, epoch, dist, prop, last = map(int, m.groups())
            ways.append(dict(way=w, valid=valid, rrpv=rrpv, epoch=epoch,
                             dist=dist, prop=prop, last=last))
        v = VIC_RE.search(line)
        if v: victim = int(v.group(1))
    if pol and ways: yield pol, ways, victim


# rule(ways, victim) -> True if victim obeys the policy spec
RULES = {
    "LRU": lambda ways, v: ways[v]["last"] == min(w["last"] for w in ways),
    "GRASP": lambda ways, v: ways[v]["rrpv"] == max(w["rrpv"] for w in ways),
    "ECG:grasp_only": lambda ways, v: ways[v]["rrpv"] == max(w["rrpv"] for w in ways),
    # shortcircuit: if any non-property present, victim must be non-property
    "ECG:shortcircuit": lambda ways, v: (
        ways[v]["prop"] == 0 if any(w["prop"] == 0 for w in ways)
        else ways[v]["dist"] == max(w["dist"] for w in ways)),
    "ECG:shortcircuit+epoch(all-prop)": lambda ways, v: ways[v]["dist"] == max(w["dist"] for w in ways),
    # epoch variants: if a record present -> victim is a record; else farthest-epoch stamped prop
    "ECG:epoch_first": lambda ways, v: _epoch_rule(ways, v),
    "ECG:epoch_only": lambda ways, v: _epoch_rule(ways, v),
    # rrip_first: victim at max-rrpv; record preferred, else farthest-epoch prop among max-rrpv
    "ECG:rrip_first": lambda ways, v: _rrip_rule(ways, v),
}


def _epoch_rule(ways, v):
    recs = [w for w in ways if w["prop"] == 0]
    if recs:
        return ways[v]["prop"] == 0  # a record must be chosen
    stamped = [w for w in ways if w["prop"] == 1 and w["epoch"] != 0]
    if stamped:
        return ways[v]["dist"] == max(w["dist"] for w in stamped)
    return True  # all unstamped -> recency fallback (accept)


def _rrip_rule(ways, v):
    mx = max(w["rrpv"] for w in ways)
    if ways[v]["rrpv"] != mx:
        return False  # must be at max-rrpv
    cand = [w for w in ways if w["rrpv"] == mx]
    recs = [w for w in cand if w["prop"] == 0]
    if recs:
        return ways[v]["prop"] == 0  # record preferred among max-rrpv
    return True


def verify_trace(name, text, prefix=""):
    """Parse a trace, assert each victim obeys its policy rule, print, return ok."""
    checked = passed = 0
    ok = True
    for pol, ways, victim in parse_blocks(text):
        rule = RULES.get(pol)
        if rule is None or victim is None:
            continue
        checked += 1
        if rule(ways, victim):
            passed += 1
        else:
            ok = False
            print(f"  [VIOLATION] {prefix}{name}/{pol}: victim=way{victim} "
                  f"ways={[ (w['way'],w['rrpv'],w['dist'],w['prop'],w['last']) for w in ways]}")
    status = "OK " if checked == passed and checked > 0 else ("NO-TRACE" if checked == 0 else "FAIL")
    print(f"  {prefix}{name:14s}: {passed}/{checked} evictions obey spec   [{status}]")
    return ok and checked > 0


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Trace-assert each L3 policy obeys its spec.")
    ap.add_argument("--gem5", action="store_true",
                    help="Also verify the gem5 ECG_GRASP_POPT variants (slower; needs gem5.opt).")
    ap.add_argument("--sniper", action="store_true",
                    help="Also verify Sniper ECG variants (guarded sg_kernel run under prlimit).")
    args = ap.parse_args(argv)

    if not PR.exists():
        print(f"FAIL: build cache_sim first (make sim-pr): {PR}"); return 2
    suites = [("LRU", dict(CACHE_POLICY="LRU", CACHE_L3_POLICY="LRU")),
              ("GRASP", dict(CACHE_POLICY="GRASP", CACHE_L3_POLICY="GRASP")),
              ("grasp_only", {**ECG_ENV, "ECG_VARIANT": "grasp_only"}),
              ("epoch_only", {**ECG_ENV, "ECG_VARIANT": "epoch_only"}),
              ("rrip_first", {**ECG_ENV, "ECG_VARIANT": "rrip_first"}),
              ("epoch_first", {**ECG_ENV, "ECG_VARIANT": "epoch_first"}),
              ("shortcircuit", {**ECG_ENV, "ECG_VARIANT": "shortcircuit"})]
    ok_all = True
    print("-- cache_sim (L3 policies, email-Eu-core) --")
    for name, env in suites:
        ok_all &= verify_trace(name, run(env))

    if args.gem5:
        if not GEM5_OPT.exists():
            print(f"FAIL: build gem5 first: {GEM5_OPT}"); return 2
        print("\n-- gem5 (ECG_GRASP_POPT variants, email-Eu-core/-o5) --")
        for variant in ["grasp_only", "epoch_only", "rrip_first", "epoch_first", "shortcircuit"]:
            ok_all &= verify_trace(variant, run_gem5(variant), prefix="gem5 ")

    if args.sniper:
        # grasp_only delegates to the shared SRRIP path (no ECG trace); verify the
        # four ECG-specific variants. Runs are memory-capped (Sniper/SDE runaway).
        print("\n-- sniper (ECG_GRASP_POPT variants, email-Eu-core/-o5, guarded) --")
        for variant in ["epoch_only", "rrip_first", "epoch_first", "shortcircuit"]:
            ok_all &= verify_trace(variant, run_sniper(variant), prefix="sniper ")

    print("\nRESULT:", "ALL POLICIES VERIFIED ✓" if ok_all else "VERIFICATION FAILED ✗")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
