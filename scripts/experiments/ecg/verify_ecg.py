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
# Epoch-coverage geometry: a big L2 absorbs the edge stream so the L3 sees
# property-dominated sets (the default workload only ever evicts records, so the
# epoch-property branch never fires); ECG_STORED_REFRESH broadcasts the next-ref
# epoch to the L3 so resident property lines carry a live stamp and the epoch
# VALUE actually discriminates between competing property lines. This is how the
# core ECG eviction logic is exercised end-to-end on the real simulator code.
COV_ENV = dict(CACHE_L2_SIZE="1MB", CACHE_L3_SIZE="4kB", ECG_STORED_REFRESH="1",
               ECG_EVICT_TRACE="4000")

WAY_RE = re.compile(r"way(\d+) valid=(\d+) rrpv=(\d+) epoch=(\d+) dist=(\d+) prop=(\d+) stamped=(\d+) last=(\d+)")
HDR_RE = re.compile(r"\[EVICT L3 pol=(\S+)")
VIC_RE = re.compile(r"-> victim=way(\d+)(?: reason=(.*))?")


def run(policy_env, extra=None):
    env = {**os.environ, **BASE_ENV, **policy_env, **(extra or {})}
    p = subprocess.run([str(PR), "-f", str(GRAPH), "-o", "0", "-n", "1", "-i", "1"],
                       env=env, capture_output=True, text=True, timeout=300)
    return p.stderr, (p.returncode == 0)


BC = ROOT / "bench" / "bin_sim" / "bc"


def run_bc(policy_env, extra=None):
    """Run BC instead of PR. BC's bottom-up + back-propagation traversal NATURALLY
    evicts PROPERTY lines (farthest-epoch branch), which the PR workload never does
    (PR only ever evicts records) — so this is the live cross-kernel coverage of the
    epoch-eviction path, on a different adapter access pattern than PR."""
    env = {**os.environ, **BASE_ENV, **policy_env, **(extra or {})}
    p = subprocess.run([str(BC), "-f", str(GRAPH), "-o", "0", "-n", "1"],
                       env=env, capture_output=True, text=True, timeout=300)
    return p.stderr, (p.returncode == 0)


GEM5_OPT = ROOT / "bench" / "include" / "gem5_sim" / "gem5" / "build" / "RISCV" / "gem5.opt"
ROI_MATRIX = ROOT / "scripts" / "experiments" / "ecg" / "roi_matrix.py"


def run_gem5(variant, cov=False):
    """Run gem5 ECG_GRASP_POPT on the tiny graph with the trace on; return the
    gem5 log text (run_command pipes the policy's stderr trace into the log).
    cov=True uses the epoch-coverage geometry (big L2 + small L3 + STORED_REFRESH)
    so the property-eviction / epoch branch is exercised."""
    out = Path("/tmp") / f"verify_gem5_{variant}{'_cov' if cov else ''}"
    env = {**os.environ, "GEM5_OPT": str(GEM5_OPT), "GEM5_KERNEL_SUFFIX": "_riscv_m5ops",
           "GEM5_FORCE_ECG_EXTRACT": "1", "GEM5_ECG_PFX_MODE": "6", "ECG_PREFETCH_MODE": "6",
           "ECG_VARIANT": variant, "ECG_EVICT_TRACE": "4000" if cov else "40"}
    l3, l2 = ("4kB", "1MB") if cov else ("16kB", "4kB")
    if cov:
        env["ECG_STORED_REFRESH"] = "1"
    cmd = [sys.executable, str(ROI_MATRIX), "--suite", "gem5", "--no-build",
           "--benchmark", "pr", "--policies", "ECG:ECG_GRASP_POPT",
           "--options", f"-f {GRAPH} -o 5 -n 1 -i 1",
           "--l3-sizes", l3, "--l3-ways", "8", "--l1d-size", "2kB", "--l2-size", l2,
           "--out-dir", str(out)]
    subprocess.run(cmd, env=env, cwd=str(ROOT),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=900, check=False)
    logs = sorted((out / "logs").glob("*.log")) if (out / "logs").exists() else []
    text = logs[0].read_text(errors="ignore") if logs else ""
    return text, bool(text)


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
    text = logs[0].read_text(errors="ignore") if logs else ""
    return text, bool(text)


def parse_blocks(text):
    """Yield (pol, ways[list of dict], victim_way, reason)."""
    pol = None; ways = []; reason = None
    for line in text.splitlines():
        h = HDR_RE.search(line)
        if h:
            if pol and ways: yield pol, ways, victim, reason
            pol = h.group(1); ways = []; victim = None; reason = None; continue
        m = WAY_RE.search(line)
        if m:
            w, valid, rrpv, epoch, dist, prop, stamped, last = map(int, m.groups())
            ways.append(dict(way=w, valid=valid, rrpv=rrpv, epoch=epoch,
                             dist=dist, prop=prop, stamped=stamped, last=last))
        v = VIC_RE.search(line)
        if v:
            victim = int(v.group(1)); reason = (v.group(2) or "").strip()
    if pol and ways: yield pol, ways, victim, reason


# --- Exact-victim rules. The trace `dist` field shows the RAW circular distance
# (epoch+ne-curEpoch)%ne. Stamped-ness is now an EXPLICIT trace bit (a per-edge
# epoch was DELIVERED), NOT "epoch != 0" — a real epoch-0 line (low-ID next-ref)
# IS stamped. rrip_first/epoch_*/shortcircuit all treat an UNSTAMPED property line
# as effective distance 0 (stamped?dist:0).
def _eff_d(w):
    return w["dist"] if (w["prop"] == 1 and w["stamped"]) else 0


# rule(ways, victim) -> True if victim is EXACTLY what the policy must evict
RULES = {
    "LRU": lambda ways, v: ways[v]["last"] == min(w["last"] for w in ways),
    "GRASP": lambda ways, v: ways[v]["rrpv"] == max(w["rrpv"] for w in ways),
    "ECG:grasp_only": lambda ways, v: ways[v]["rrpv"] == max(w["rrpv"] for w in ways),
    # shortcircuit: FIRST record by set index; else farthest EFFECTIVE-dist property
    "ECG:shortcircuit": lambda ways, v: _shortcircuit_rule(ways, v),
    "ECG:shortcircuit+epoch": lambda ways, v: _eff_d(ways[v]) == max(_eff_d(w) for w in ways),
    # epoch_*: oldest record by recency; else farthest dist among STAMPED; else LRU
    "ECG:epoch_first": lambda ways, v: _epoch_rule(ways, v),
    "ECG:epoch_only": lambda ways, v: _epoch_rule(ways, v),
    # rrip_first: among max-rrpv, oldest record by recency; else max effective-epoch property
    "ECG:rrip_first": lambda ways, v: _rrip_rule(ways, v),
}


def _epoch_rule(ways, v):
    recs = [w for w in ways if w["prop"] == 0]
    if recs:  # oldest record by recency
        return ways[v]["prop"] == 0 and ways[v]["last"] == min(w["last"] for w in recs)
    stamped_lines = [w for w in ways if w["prop"] == 1 and w["stamped"]]
    if stamped_lines:  # farthest next-ref among stamped property
        return ways[v]["stamped"] and ways[v]["dist"] == max(w["dist"] for w in stamped_lines)
    return ways[v]["last"] == min(w["last"] for w in ways)  # all unstamped -> LRU fallback


def _rrip_rule(ways, v):
    mx = max(w["rrpv"] for w in ways)
    if ways[v]["rrpv"] != mx:
        return False  # must be at max-rrpv
    cand = [w for w in ways if w["rrpv"] == mx]
    recs = [w for w in cand if w["prop"] == 0]
    if recs:  # oldest record by recency among max-rrpv
        return ways[v]["prop"] == 0 and ways[v]["last"] == min(w["last"] for w in recs)
    # else farthest EFFECTIVE-epoch property among max-rrpv (unstamped -> 0)
    return ways[v]["prop"] == 1 and _eff_d(ways[v]) == max(_eff_d(w) for w in cand)


def _shortcircuit_rule(ways, v):
    recs = [w for w in ways if w["prop"] == 0]
    if recs:  # FIRST record by set order (distinguishes from epoch's recency)
        return ways[v]["prop"] == 0 and ways[v]["way"] == min(w["way"] for w in recs)
    return _eff_d(ways[v]) == max(_eff_d(w) for w in ways)  # all property -> max effective dist


def verify_trace(name, result, prefix="", reasons=None):
    """Assert each victim in a (text, ran_ok) result obeys its policy rule.
    Hard-fails on runner failure (no/empty trace) and on any emitted policy with
    no rule. Tallies eviction `reason=` strings into `reasons` for coverage."""
    text, ran_ok = result
    if not ran_ok:
        print(f"  {prefix}{name:14s}: runner FAILED (crash / no log)   [FAIL]")
        return False
    checked = passed = 0
    ok = True
    unknown = set()
    stamp_violations = 0
    for pol, ways, victim, reason in parse_blocks(text):
        if reasons is not None and reason:
            reasons.add(reason)
        # Stamping-correctness invariant (C): a record (non-property) line must NEVER
        # carry a stamp — stamped=1 implies prop=1. The shared policy's effDist and the
        # rules below all assume this; a backend adapter that stamped a record (or a
        # kernel that failed to clear before a sequential read) would corrupt eviction.
        for w in ways:
            if w["prop"] == 0 and w["stamped"] == 1:
                stamp_violations += 1
                if stamp_violations <= 3:
                    print(f"  [STAMP-INVARIANT] {prefix}{name}/{pol}: way{w['way']} is a "
                          f"record (prop=0) but stamped=1 (records must never be stamped)")
        rule = RULES.get(pol)
        if rule is None:
            unknown.add(pol); continue
        if victim is None:
            continue
        checked += 1
        if rule(ways, victim):
            passed += 1
        else:
            ok = False
            print(f"  [VIOLATION] {prefix}{name}/{pol}: victim=way{victim} "
                  f"ways={[ (w['way'],w['rrpv'],w['dist'],w['prop'],w['last']) for w in ways]}")
    if stamp_violations:  # records must never be stamped -> fail loudly
        ok = False
        print(f"  [STAMP-INVARIANT] {prefix}{name}: {stamp_violations} record(s) stamped (must be 0)")
    if unknown:  # an emitted policy with no checker is a coverage hole -> fail loudly
        ok = False
        print(f"  [UNKNOWN POL] {prefix}{name}: {sorted(unknown)} has no RULES entry")
    status = "OK " if ok and checked > 0 else ("NO-TRACE" if checked == 0 else "FAIL")
    print(f"  {prefix}{name:14s}: {passed}/{checked} evictions obey spec   [{status}]")
    return ok and checked > 0


SYNTH_BIN = ROOT / "bench" / "bin_sim" / "test_ecg_victim"


def verify_unknown_mode_hardfails():
    """Negative test: an unrecognized ECG_MODE must HARD-FAIL (exit!=0 + [FATAL]),
    not silently fall back to DBG_PRIMARY. Silent fallback would run a different
    policy than requested while labelling itself as the requested mode. This is the
    safety gate that must hold before any ECG mode can be deleted/renamed."""
    env = {**os.environ, **BASE_ENV, "CACHE_POLICY": "ECG", "ECG_MODE": "BOGUS_MODE_XYZ"}
    p = subprocess.run([str(PR), "-f", str(GRAPH), "-o", "0", "-n", "1", "-i", "1"],
                       env=env, capture_output=True, text=True, timeout=120)
    hard_failed = (p.returncode != 0) and ("[FATAL]" in p.stderr) and ("BOGUS_MODE_XYZ" in p.stderr)
    print(f"  unknown ECG_MODE hard-fails (exit={p.returncode}, [FATAL] emitted): "
          f"{'[OK ]' if hard_failed else '[FAIL]'}")
    return hard_failed


def run_synthetic():
    """Build + run the synthetic deterministic victim test: controlled 8-way sets
    with hand-computed exact victims. This is the part that actually exercises the
    epoch-property ranking (the live PageRank trace only ever evicts records) and
    pins the EXACT victim (not just necessary conditions), independent of the
    simulator's self-reported state."""
    subprocess.run(["make", "bench/bin_sim/test_ecg_victim"], cwd=str(ROOT),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if not SYNTH_BIN.exists():
        print("  [synthetic] FAIL: could not build test_ecg_victim"); return False
    ok = True
    for variant in ["grasp_only", "epoch_only", "rrip_first", "epoch_first", "shortcircuit"]:
        p = subprocess.run([str(SYNTH_BIN)], env={**os.environ, "ECG_VARIANT": variant},
                           capture_output=True, text=True, timeout=60)
        for line in p.stdout.splitlines():
            if "expect=" in line or line.startswith("[test_ecg_victim]"):
                print("  " + line.rstrip())
        if p.returncode != 0:
            ok = False
    return ok


def _epoch_decided(pol, ways, v):
    """True iff the epoch VALUE selected this victim: the victim is a stamped
    property line chosen as the farthest among >=2 stamped property lines with
    distinct epochs, within the candidate pool that variant ranks. (shortcircuit
    ranks property by RAW dist, so unstamped property—huge raw dist—is evicted
    first and the stamped-epoch comparison is rarely operative; that variant's
    epoch ranking is covered by the synthetic test instead.)"""
    if v is None or ways[v]["prop"] != 1 or not ways[v]["stamped"]:
        return False
    if pol == "ECG:rrip_first":
        mx = max(w["rrpv"] for w in ways)
        pool = [w for w in ways if w["rrpv"] == mx and w["prop"] == 1 and w["stamped"]]
    else:  # epoch_first / epoch_only rank all stamped property
        pool = [w for w in ways if w["prop"] == 1 and w["stamped"]]
    return (len(pool) >= 2 and len({w["dist"] for w in pool}) >= 2
            and ways[v]["dist"] == max(w["dist"] for w in pool))


def _count_epoch_decided(text):
    return sum(1 for pol, ways, v, r in parse_blocks(text) if _epoch_decided(pol, ways, v))


def verify_epoch_coverage(name, result, prefix="", strict=True):
    """Like verify_trace, but ALSO consider whether the epoch VALUE genuinely
    selected the victim (a stamped property line chosen as farthest among >=2
    distinct-epoch competitors). With strict=True, fail if that never happened
    (so the check cannot pass vacuously). With strict=False, report the count but
    do not fail on it — used where the model cannot keep a live L3 epoch stamp
    (gem5 has no ECG_STORED_REFRESH, so property reaches the L3 unstamped; its
    epoch ranking is covered by the cache_sim mirror + the synthetic test)."""
    text, ran_ok = result
    ok = verify_trace(name, result, prefix=prefix)
    if not ran_ok:
        return False
    comp = _count_epoch_decided(text)
    tag = f"{prefix}{name}"
    if comp == 0:
        verdict = "FAIL" if strict else "info: covered by cache_sim mirror"
        print(f"  [COVERAGE] {tag}: epoch value never selected the victim  [{verdict}]")
        return False if strict else ok
    print(f"  [COVERAGE] {tag}: {comp} evictions where the epoch value selected the victim  [OK]")
    return ok


def verify_omp_robustness():
    """OMP>1 robustness (D): cache_sim's L3 is mutex-serialized and the per-edge hints
    are PER-THREAD (hints_for_thread), so the eviction decision must stay correct under
    concurrency. Run the clearEdgeEpoch workload (BC + per-edge masks) with 4 OMP threads
    and assert every eviction still obeys spec AND both the delivered (stamped) and cleared
    (unstamped) paths still fire. A per-thread hint hazard — e.g. a worker's first
    sequential read over-stamping because its thread-local valid bit was never cleared —
    would break spec or collapse the cleared count. (Counts vary run-to-run; only the
    invariants are asserted, so this is a stable gate.)"""
    if not BC.exists():
        print("  [skip] BC binary not built — skipping OMP robustness"); return True
    env = {**ECG_ENV, "ECG_VARIANT": "epoch_only", "ECG_EDGE_MASKS": "1"}
    text, ran = run_bc(env, {**COV_ENV, "OMP_NUM_THREADS": "4"})
    ok = verify_trace("bc+masks OMP=4", (text, ran), prefix="(omp) ")
    sp = up = 0
    for _p, ways, _v, _r in parse_blocks(text):
        for w in ways:
            if w["prop"] == 1 and w["stamped"] == 1: sp += 1
            elif w["prop"] == 1 and w["stamped"] == 0: up += 1
    fired = sp > 0 and up > 0
    print(f"  OMP=4 per-thread hints: delivered={sp} cleared={up} "
          f"(both>0 under concurrency): {'[OK ]' if fired else '[FAIL]'}")
    return ok and fired


def verify_clearedge_path():
    """Cross-kernel clearEdgeEpoch coverage — the exact locus of the over-stamping bug
    that the per-sim spec checks (each sim vs its OWN trace) structurally cannot catch.
    Run BC with per-edge masks ON so clearEdgeEpoch is non-no-op, then assert BOTH:
      (a) the live trace obeys spec (the cleared/unstamped lines follow effDist=0), and
      (b) BOTH stamped AND unstamped property lines appear — i.e. the per-edge DELIVERY
          path (valid=true) and the SEQUENTIAL/cleared path (valid=false) both fire live.
    If clearEdgeEpoch regressed to a no-op, or the fill stamped unconditionally (the bug),
    there would be ZERO unstamped property lines and this FAILS."""
    if not BC.exists():
        print("  [skip] BC binary not built — skipping clearEdgeEpoch coverage"); return True
    env = {**ECG_ENV, "ECG_VARIANT": "epoch_only", "ECG_EDGE_MASKS": "1"}
    text, ran = run_bc(env, COV_ENV)
    ok = verify_trace("bc+masks/epoch_only", (text, ran), prefix="(ce) ")
    stamped_prop = unstamped_prop = 0
    for _pol, ways, _victim, _reason in parse_blocks(text):
        for w in ways:
            if w["prop"] == 1 and w["stamped"] == 1: stamped_prop += 1
            elif w["prop"] == 1 and w["stamped"] == 0: unstamped_prop += 1
    fired = stamped_prop > 0 and unstamped_prop > 0
    print(f"  clearEdgeEpoch live: delivered(stamped)={stamped_prop} cleared(unstamped)="
          f"{unstamped_prop}  (both>0 = delivery AND clear fire): {'[OK ]' if fired else '[FAIL]'}")
    return ok and fired


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Assert each L3 policy obeys its spec.")
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
    live_reasons = set()
    print("== synthetic deterministic victim tests (EXACT victim; exercises the epoch branch) ==")
    ok_all &= run_synthetic()
    print("\n-- negative test: unknown ECG_MODE must hard-fail (not silent DBG_PRIMARY) --")
    ok_all &= verify_unknown_mode_hardfails()
    print("\n-- cache_sim (L3 policies, email-Eu-core; live-trace integration) --")
    for name, env in suites:
        ok_all &= verify_trace(name, run(env), reasons=live_reasons)

    # Epoch-coverage: force the epoch-property branch live (big L2 + small L3 +
    # ECG_STORED_REFRESH) and assert the tightened exact rules hold AND the epoch
    # value genuinely broke property ties. This exercises ECG's core eviction on
    # the REAL simulator end-to-end (not just the synthetic unit test).
    print("\n-- cache_sim epoch-coverage (forced property eviction; tightened exact rules) --")
    for variant in ["rrip_first", "epoch_first", "epoch_only"]:
        ok_all &= verify_epoch_coverage(variant, run({**ECG_ENV, "ECG_VARIANT": variant}, COV_ENV))
    # shortcircuit ranks property by RAW dist (evicts unstamped first), so its
    # stamped-epoch ranking is rarely operative live; verify its exact rule here
    # and rely on the synthetic test for its stamped-epoch + DBG-tiebreak path.
    ok_all &= verify_trace("shortcircuit", run({**ECG_ENV, "ECG_VARIANT": "shortcircuit"}, COV_ENV),
                           prefix="(sc) ")

    # Cross-kernel coverage (B+C): BC's bottom-up traversal NATURALLY evicts PROPERTY
    # lines (the PR workload only ever evicts records), so this is the only LIVE check
    # of the epoch-eviction branch on a real kernel via a DIFFERENT adapter access
    # pattern. verify_trace also enforces the record-never-stamped invariant (C).
    if BC.exists():
        print("\n-- cache_sim BC cross-kernel (BC evicts property -> live epoch branch + stamp invariant) --")
        for variant in ["grasp_only", "epoch_only", "rrip_first", "epoch_first", "shortcircuit"]:
            ok_all &= verify_trace(f"bc/{variant}", run_bc({**ECG_ENV, "ECG_VARIANT": variant}, COV_ENV),
                                   prefix="(bc) ", reasons=live_reasons)
        # BC epoch-coverage is INFORMATIONAL (strict=False): BC's property lines carry
        # uniform/fallback epochs under this geometry (it is not the full per-edge-mask
        # delivery kernel that PR is), so the epoch VALUE rarely discriminates between
        # >=2 candidates. The strict epoch-discrimination is covered by PR's epoch-
        # coverage + the synthetic test; BC's value here is the LIVE property-eviction
        # spec-compliance (above) on a different adapter access pattern.
        for variant in ["rrip_first", "epoch_first", "epoch_only"]:
            ok_all &= verify_epoch_coverage(f"bc/{variant}",
                                            run_bc({**ECG_ENV, "ECG_VARIANT": variant}, COV_ENV),
                                            prefix="(bc) ", strict=False)
        # clearEdgeEpoch live coverage (the over-stamping bug locus PR never reaches).
        print("\n-- cache_sim clearEdgeEpoch path (BC + per-edge masks: delivery vs cleared) --")
        ok_all &= verify_clearedge_path()
        # OMP>1 robustness (D): per-thread hints must stay correct under concurrency.
        print("\n-- cache_sim OMP>1 robustness (BC + masks, 4 threads) --")
        ok_all &= verify_omp_robustness()
    else:
        print("  [skip] BC binary not built (make sim-bc) — skipping cross-kernel coverage")

    if args.gem5:
        if not GEM5_OPT.exists():
            print(f"FAIL: build gem5 first: {GEM5_OPT}"); return 2
        print("\n-- gem5 (ECG_GRASP_POPT variants, email-Eu-core/-o5) --")
        for variant in ["grasp_only", "epoch_only", "rrip_first", "epoch_first", "shortcircuit"]:
            ok_all &= verify_trace(variant, run_gem5(variant), prefix="gem5 ", reasons=live_reasons)
        print("\n-- gem5 epoch-coverage (exact rules on forced geometry; epoch-value gate informational) --")
        for variant in ["rrip_first", "epoch_first"]:
            ok_all &= verify_epoch_coverage(variant, run_gem5(variant, cov=True), prefix="gem5 ", strict=False)

    if args.sniper:
        # grasp_only delegates to the shared SRRIP path (no ECG trace); verify the
        # four ECG-specific variants. Runs are memory-capped (Sniper/SDE runaway).
        print("\n-- sniper (ECG_GRASP_POPT variants, email-Eu-core/-o5, guarded) --")
        for variant in ["epoch_only", "rrip_first", "epoch_first", "shortcircuit"]:
            ok_all &= verify_trace(variant, run_sniper(variant), prefix="sniper ", reasons=live_reasons)

    # Live default-geometry coverage note: that workload only ever evicts records,
    # so the epoch branch is exercised by the synthetic + epoch-coverage runs above.
    epoch_reasons = {r for r in live_reasons if "epoch property" in r or "farthest" in r}
    print("\n-- live-trace branch coverage (default geometry) --")
    print(f"  live eviction reasons seen: {sorted(live_reasons) or '(none)'}")
    print(f"  epoch-property branch fired in default geom: {'yes' if epoch_reasons else 'NO (covered by synthetic + epoch-coverage runs)'}")

    print("\nRESULT:", "ALL POLICIES VERIFIED ✓" if ok_all else "VERIFICATION FAILED ✗")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
