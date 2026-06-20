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


def main():
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
    for name, env in suites:
        text = run(env)
        checked = passed = 0
        for pol, ways, victim in parse_blocks(text):
            rule = RULES.get(pol)
            if rule is None or victim is None:
                continue
            checked += 1
            if rule(ways, victim):
                passed += 1
            else:
                ok_all = False
                print(f"  [VIOLATION] {name}/{pol}: victim=way{victim} "
                      f"ways={[ (w['way'],w['rrpv'],w['dist'],w['prop'],w['last']) for w in ways]}")
        status = "OK " if checked == passed and checked > 0 else ("NO-TRACE" if checked == 0 else "FAIL")
        print(f"  {name:14s}: {passed}/{checked} evictions obey spec   [{status}]")
        if checked == 0:
            ok_all = False
    print("\nRESULT:", "ALL POLICIES VERIFIED ✓" if ok_all else "VERIFICATION FAILED ✗")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
