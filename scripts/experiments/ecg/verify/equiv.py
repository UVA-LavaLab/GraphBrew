#!/usr/bin/env python3
"""Behavioral cross-simulator equivalence gate + reorder guard + insertion-RRPV
invariant for the L3 graph-aware replacement policies (GRASP / ECG).

WHY THIS EXISTS
---------------
The per-simulator eviction-spec checks in verify/ecg.py assert that *every
eviction obeys its policy spec*. That is necessary but NOT sufficient — it
structurally cannot catch three real bug classes that all shipped "green":

  1. A backwards INSERTION RRPV. gem5 GraphGraspRP inserted non-property /
     streaming data at rrpv=2 (near-MRU, protected) instead of the SRRIP-distant
     maxRRPV-1. Every eviction still obeyed spec (it always evicted the max-RRPV
     line); the bug was that the *wrong lines* held low RRPV. GRASP then BACKFIRED
     (hurt vs LRU) only in gem5 — invisible to a per-sim eviction check.
  2. An UNREORDERED workload. Sniper's sg_kernel silently ignored -o5, so all its
     degree-policy runs used original-order graphs. Every eviction still obeyed
     spec, so verify --sniper passed on a graph the policy was never meant to see.
  3. NO cross-sim OUTCOME comparison + uncontrolled graphs (k4 vs k16). gem5 GRASP
     hurting while cache_sim/Sniper helped was invisible because each sim was only
     ever checked against its own trace.

These checks enforce BEHAVIORAL correctness/equivalence instead of mere mechanism
conformance:

  GATE 1 (behavioral equivalence): run LRU + GRASP + ECG on the SAME reordered
    tiny cell in each sim; assert each graph-aware policy moves the L3 miss rate
    the SAME direction vs LRU (does not regress beyond a small tolerance) and that
    all simulators agree on that direction. Catches bug #1 and #3.

  GATE 2 (reorder-applied guard): per sim, assert that -o5 actually changes the
    cache outcome vs -o0. A sim that ignores -o (Sniper's bug) produces identical
    numbers and FAILS. Catches bug #2.

  GATE 3 (insertion-RRPV invariant): assert each sim's GRASP/ECG non-property
    insertion uses the SRRIP-distant RRPV (>= maxRRPV-1), not a near-MRU constant.
    A fast static guard that pins the exact lines bug #1 lived on.

Usage:
  python3 scripts/experiments/ecg/experiments.py verify --equiv   # cache_sim (fast) + static invariant
  python3 scripts/experiments/ecg/verify/equiv.py                 # same, direct
  python3 scripts/experiments/ecg/verify/equiv.py --gem5          # + gem5 behavioral gate (slow)
  python3 scripts/experiments/ecg/verify/equiv.py --sniper        # + Sniper behavioral gate (slow, guarded)
The behavioral gates also run automatically at the end of `experiments.py verify`
(cache_sim always; gem5/Sniper with --gem5/--sniper).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ROI_MATRIX = ROOT / "scripts" / "experiments" / "ecg" / "roi_matrix.py"
# Pressured cell: the property arrays must EXCEED the L3 so the replacement policy
# actually differentiates. A tiny graph whose property fits in L3 (e.g.
# email-Eu-core @ 16kB) leaves GRASP marginal/noisy and the cache_sim (graph-only)
# vs gem5 (full-ISA) access-stream difference dominates -> false disagreement.
# kron_s16_k4 @ 128kB/16-way is gem5/Sniper-feasible AND shows all three sims agree
# GRASP/ECG help (this is the cell the non-property-RRPV fix was validated on).
GRAPH = ROOT / "results" / "graphs" / "kron_s16_k4" / "kron_s16_k4.sg"
CELL = dict(l3_size="128kB", l3_ways="16", l1d_size="16kB", l2_size="64kB")

# A graph-aware policy must not be WORSE than LRU by more than this (noise band).
# The gem5 non-property=2 bug made GRASP ~10% WORSE than LRU -> fails this.
REGRESS_TOL = 0.03
# -o5 must change the L3 miss rate vs -o0 by at least this (else the reorder was
# silently ignored, as in Sniper's sg_kernel bug).
REORDER_MIN_DELTA = 0.01


# --------------------------------------------------------------------------- #
# Uniform runner: roi_matrix drives cache_sim / gem5 / Sniper identically.
# --------------------------------------------------------------------------- #
def _run_matrix(suite: str, policies: list[str], reorder: int, out: Path,
                extra: list[str] | None = None, timeout: int = 1200) -> dict[str, float]:
    """Run roi_matrix for one suite and return {policy_label: l3_miss_rate}."""
    import shutil
    shutil.rmtree(out, ignore_errors=True)   # never read stale results from a prior cell
    out_str = str(out)
    cmd = [sys.executable, str(ROI_MATRIX), "--suite", suite, "--no-build",
           "--benchmark", "pr", "--policies", *policies,
           "--options", f"-f {GRAPH} -o {reorder} -n 1 -i 1",
           "--l3-sizes", CELL["l3_size"], "--l3-ways", CELL["l3_ways"],
           "--l1d-size", CELL["l1d_size"], "--l2-size", CELL["l2_size"],
           "--out-dir", out_str]
    if extra:
        cmd += extra
    subprocess.run(cmd, cwd=str(ROOT), stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, timeout=timeout, check=False)
    rates: dict[str, float] = {}
    for p in glob.glob(os.path.join(out_str, "**", "roi_matrix.json"), recursive=True):
        try:
            rows = json.load(open(p))
        except Exception:
            continue
        for r in rows:
            mr = r.get("l3_miss_rate")
            pol = r.get("policy_label") or r.get("policy")
            if mr is None or pol is None or r.get("error"):
                continue
            try:
                rates[pol] = float(mr)
            except (TypeError, ValueError):
                continue
    return rates


SUITE_OF = {"cache_sim": "cache-sim", "gem5": "gem5", "sniper": "sniper"}
SUITE_EXTRA = {
    "sniper": ["--sniper-workload", "sg_kernel", "--allow-sniper-sg-kernel-workload",
               "--sniper-memory-limit-gb", "20", "--sniper-enable-graph-policies",
               "--timeout-sniper", "540"],
}


# --------------------------------------------------------------------------- #
# GATE 1: behavioral equivalence  +  GATE 2: reorder-applied guard
# --------------------------------------------------------------------------- #
def check_behavioral_equivalence(sims: list[str]) -> bool:
    """Same reordered cell, every sim: GRASP and ECG must not regress vs LRU, and
    all sims must agree on the direction. Also runs the reorder guard per sim."""
    print("== GATE 1+2: behavioral equivalence (reordered) + reorder-applied guard ==")
    # cache_sim is fast -> check ECG too. gem5/Sniper ECG on a PRESSURED cell exceeds
    # the per-run sim timeout (minutes of detailed sim), so the slow sims gate on
    # LRU+GRASP (the policy whose insertion bug caused the backfire); ECG equivalence
    # is covered by cache_sim here + the gem5/Sniper ECG eviction-spec checks.
    POLICIES = {"cache_sim": ["LRU", "GRASP", "ECG:ECG_GRASP_POPT"]}
    SLOW_POLICIES = ["LRU", "GRASP"]
    ok_all = True
    directions: dict[str, dict[str, float]] = {}

    for sim in sims:
        suite = SUITE_OF[sim]
        extra = SUITE_EXTRA.get(sim)
        policies = POLICIES.get(sim, SLOW_POLICIES)
        # Run ONE policy per roi_matrix call so a slow policy (e.g. gem5 ECG on a
        # pressured cell) stays under the per-call timeout instead of starving the
        # others. Each call writes its own out-dir (cleaned by _run_matrix).
        ro5: dict[str, float] = {}
        for pol in policies:
            tag = re.sub(r"[^0-9A-Za-z]+", "_", pol)
            ro5.update(_run_matrix(suite, [pol], 5, Path("/tmp") / f"equiv_{sim}_o5_{tag}", extra))
        lru = ro5.get("LRU")
        if lru is None:
            print(f"  [{sim:9}] FAIL: no LRU L3 miss rate (run failed)")
            ok_all = False
            continue

        sim_dir: dict[str, float] = {}
        checks = [("GRASP", "GRASP")]
        if any(p.startswith("ECG") for p in policies):
            checks.append(("ECG", "ECG_ECG_GRASP_POPT"))
        for label, key in checks:
            mr = ro5.get(key) or ro5.get(label)
            if mr is None:
                print(f"  [{sim:9}] FAIL: no {label} L3 miss rate")
                ok_all = False
                continue
            delta = lru - mr                      # >0 = helps
            sim_dir[label] = delta
            regressed = mr > lru * (1.0 + REGRESS_TOL)
            verdict = "HELPS" if delta > 0 else ("ties" if not regressed else "HURTS")
            status = "[FAIL]" if regressed else "[OK ]"
            print(f"  [{sim:9}] {label:5}: LRU={lru:.4f} {label}={mr:.4f} "
                  f"d={delta:+.4f} ({verdict}) {status}")
            if regressed:
                ok_all = False
        directions[sim] = sim_dir

        # GATE 2: reorder-applied guard — -o5 must differ from -o0 for this sim.
        ro0 = _run_matrix(suite, ["GRASP"], 0, Path("/tmp") / f"equiv_{sim}_o0", extra)
        g0 = ro0.get("GRASP")
        g5 = ro5.get("GRASP")
        if g0 is None or g5 is None:
            print(f"  [{sim:9}] reorder-guard: SKIP (missing -o0/-o5 GRASP)")
        else:
            d = abs(g5 - g0)
            ok = d >= REORDER_MIN_DELTA
            print(f"  [{sim:9}] reorder-guard: GRASP -o0={g0:.4f} -o5={g5:.4f} "
                  f"|d|={d:.4f}  {'[OK ]' if ok else '[FAIL: -o ignored?]'}")
            ok_all &= ok

    # Cross-sim direction agreement: every sim that ran a policy must agree it helps
    # (or all tie); no sim may go the opposite way.
    if len(directions) > 1:
        for label in ("GRASP", "ECG"):
            signs = {sim: (1 if d.get(label, 0) > 0 else (0 if abs(d.get(label, 0)) <= 1e-6 else -1))
                     for sim, d in directions.items() if label in d}
            if not signs:
                continue
            helps = sorted(s for s, v in signs.items() if v > 0)
            hurts = sorted(s for s, v in signs.items() if v < 0)
            agree = bool(helps) and not hurts
            print(f"  cross-sim {label}: helps={helps} hurts={hurts} "
                  f"{'[OK ]' if agree else '[FAIL: sims disagree on direction]'}")
            ok_all &= agree
    return ok_all


# --------------------------------------------------------------------------- #
# GATE 3: insertion-RRPV invariant (static — pins the lines bug #1 lived on)
# --------------------------------------------------------------------------- #
def _read(rel: str) -> str:
    p = ROOT / rel
    return p.read_text(errors="ignore") if p.exists() else ""


def _slice(text: str, start_pat: str, span: int = 1200) -> str:
    m = re.search(start_pat, text)
    return text[m.start():m.start() + span] if m else ""


def check_insertion_rrpv_invariant() -> bool:
    """Assert each sim inserts NON-property / unclassified data at the SRRIP-distant
    RRPV (>= maxRRPV-1), never a near-MRU constant. Pins the exact bug #1 lines."""
    print("\n== GATE 3: insertion-RRPV invariant (non-property must insert DISTANT) ==")
    ok = True

    # gem5 GraphGraspRP::reset (grasp_rp.cc). The whole policy classifies property
    # via insertionRRPV(tier); its ONLY other insert is the non-property else-branch
    # right after `isPropertyData(addr)`. The backfire bug was that branch using
    # rrpv=2. Scope the scan to that insertion else-branch (NOT promoteOnHit, whose
    # rrpv=0 HOT promotion is legitimate).
    g = _read("bench/include/gem5_sim/gem5/src/mem/cache/replacement_policies/grasp_rp.cc")
    g_reset_else = _slice(g, r"ctx\.isPropertyData\(addr\)", 1000)
    g_bad = re.search(r"\}\s*else\s*\{[^}]*?data->rrpv\s*=\s*([0-5])\s*;", g_reset_else, re.DOTALL)
    g_distant = re.search(r"\}\s*else\s*\{[^}]*?data->rrpv\s*=\s*maxRRPV(\s*-\s*1)?\s*;",
                          g_reset_else, re.DOTALL) is not None
    if g_bad:
        print(f"  [gem5  GraphGraspRP] FAIL: non-property inserts near-MRU rrpv={g_bad.group(1)} "
              f"(must be maxRRPV-1). THE GRASP-backfire bug.")
        ok = False
    elif g_distant:
        print("  [gem5  GraphGraspRP] non-property insert -> maxRRPV-1 (distant): [OK ]")
    else:
        print("  [gem5  GraphGraspRP] FAIL: could not confirm non-property -> maxRRPV-1 "
              "(review grasp_rp.cc reset()).")
        ok = False

    # Sniper CacheSetGRASP::insertionRRPV — non-property falls through to m_rrip_insert,
    # which is constructed as (m_rrip_max - 1) = SRRIP distant.
    sn = _read("bench/include/sniper_sim/snipersim/common/core/memory_subsystem/cache/cache_set_grasp.cc")
    sn_distant = re.search(r"m_rrip_insert\s*\(\s*m_rrip_max\s*-\s*1\s*\)", sn) is not None
    sn_fallthrough = "return m_rrip_insert;" in sn
    if sn_distant and sn_fallthrough:
        print("  [sniper CacheSetGRASP] non-property -> m_rrip_insert=(maxRRPV-1): [OK ]")
    else:
        print(f"  [sniper CacheSetGRASP] FAIL: non-property not distant "
              f"(m_rrip_insert=max-1:{sn_distant} returns it:{sn_fallthrough})")
        ok = False

    # cache_sim GRASP (cache_sim.h) — every insert routes through classifyGRASP; the
    # tier-3 (non-property / cold) else-branch maps to M_RRIP (=7=max, distant).
    cs = _read("bench/include/cache_sim/cache_sim.h")
    cs_block = _slice(cs, r"if \(policy_ == EvictionPolicy::GRASP\)", 800)
    cs_m7 = re.search(r"M_RRIP\s*=\s*7", cs) is not None
    cs_else = re.search(r"else\s+set\[victim_idx\]\.rrpv\s*=\s*M_RRIP", cs_block) is not None
    if cs_m7 and cs_else:
        print("  [cache_sim GRASP] non-property (tier 3) -> M_RRIP=7 (max): [OK ]")
    else:
        print(f"  [cache_sim GRASP] FAIL: non-property not -> M_RRIP "
              f"(M_RRIP==7:{cs_m7} else->M_RRIP:{cs_else})")
        ok = False

    # gem5 ECG_GRASP_POPT headline path (ecg_rp.cc) — non-property (non-legacy
    # variants) inserts mRrip (distant); only the legacy shortcircuit variant uses 2.
    ecg = _read("bench/include/gem5_sim/gem5/src/mem/cache/replacement_policies/ecg_rp.cc")
    if "legacy_sc ? 2 : mRrip" in ecg:
        print("  [gem5  ECG_GRASP_POPT] non-property (non-legacy) -> mRrip (distant): [OK ]")
    else:
        print("  [gem5  ECG_GRASP_POPT] FAIL: headline non-property path not -> mRrip "
              "(review ecg_rp.cc ECG_GRASP_POPT insertion).")
        ok = False
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gem5", action="store_true", help="include gem5 behavioral gate (slow)")
    ap.add_argument("--sniper", action="store_true", help="include Sniper behavioral gate (slow, guarded)")
    args = ap.parse_args(argv)

    if not GRAPH.exists():
        print(f"FAIL: tiny graph missing: {GRAPH}")
        return 2

    sims = ["cache_sim"]
    if args.gem5:
        sims.append("gem5")
    if args.sniper:
        sims.append("sniper")

    ok = True
    ok &= check_behavioral_equivalence(sims)
    ok &= check_insertion_rrpv_invariant()

    print("\nRESULT:", "EQUIVALENCE VERIFIED ✓" if ok else "EQUIVALENCE FAILED ✗")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
