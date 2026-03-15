#!/usr/bin/env python3
"""
gem5 SE-mode config: single-core 3-level cache with selectable L3 policy.

Usage:
    gem5.opt policy_sweep.py <POLICY> [--binary PATH] [--ecg-mode MODE]

Example:
    gem5.opt --outdir=m5out_LRU  policy_sweep.py LRU
    gem5.opt --outdir=m5out_GRASP policy_sweep.py GRASP
    gem5.opt --outdir=m5out_ECG   policy_sweep.py ECG --ecg-mode ECG_EMBEDDED
"""

import argparse
import os
import sys

import m5
from m5.objects import *


def make_l3_policy(name, ecg_mode="DBG_PRIMARY"):
    """Create L3 replacement policy SimObject by name."""
    if name == "LRU":
        return LRURP()
    elif name == "FIFO":
        return FIFORP()
    elif name == "SRRIP":
        return BRRIPRP(btp=0)
    elif name == "RANDOM":
        return RandomRP()
    elif name == "GRASP":
        return GraphGraspRP(max_rrpv=7, num_buckets=11, hot_fraction=0.1)
    elif name == "POPT":
        return GraphPoptRP(max_rrpv=7)
    elif name == "ECG":
        return GraphEcgRP(rrpv_max=7, num_buckets=11, ecg_mode=ecg_mode)
    else:
        print(f"Unknown policy '{name}', using LRU")
        return LRURP()


# Parse args (gem5 passes remaining args after config script)
parser = argparse.ArgumentParser()
parser.add_argument("policy", default="LRU",
                    help="L3 cache policy: LRU, SRRIP, GRASP, POPT, ECG")
parser.add_argument("--binary", default="",
                    help="Binary to simulate (default: /tmp/gem5_pr_large)")
parser.add_argument("--ecg-mode", default="DBG_PRIMARY",
                    help="ECG mode: DBG_PRIMARY, POPT_PRIMARY, DBG_ONLY, ECG_EMBEDDED")
parser.add_argument("--binary-args", default="",
                    help="Arguments to pass to the binary")

args = parser.parse_args()

# Find binary
if not args.binary:
    args.binary = "/tmp/gem5_pr_large"
if not os.path.exists(args.binary):
    print(f"Error: binary not found: {args.binary}")
    sys.exit(1)

# System
system = System()
system.clk_domain = SrcClockDomain(clock="2GHz",
                                    voltage_domain=VoltageDomain())
system.mem_mode = "timing"
system.mem_ranges = [AddrRange("4GB")]

# CPU
system.cpu = TimingSimpleCPU()

# L1 caches (LRU — standard for all experiments)
system.cpu.icache = Cache(
    size="32kB", assoc=8,
    tag_latency=2, data_latency=2, response_latency=2,
    mshrs=4, tgts_per_mshr=20)
system.cpu.dcache = Cache(
    size="32kB", assoc=8,
    tag_latency=2, data_latency=2, response_latency=2,
    mshrs=4, tgts_per_mshr=20)

# L2 cache (LRU — standard)
system.l2cache = Cache(
    size="256kB", assoc=4,
    tag_latency=10, data_latency=10, response_latency=10,
    mshrs=20, tgts_per_mshr=12)

# L3 cache with selectable policy
system.l3cache = Cache(
    size="8MB", assoc=16,
    tag_latency=20, data_latency=20, response_latency=20,
    mshrs=32, tgts_per_mshr=16,
    replacement_policy=make_l3_policy(args.policy, args.ecg_mode))

# Buses
system.membus = SystemXBar()
system.l2bus = L2XBar()
system.l3bus = L2XBar()

# L1 → L2
system.cpu.icache.mem_side = system.l2bus.cpu_side_ports
system.cpu.dcache.mem_side = system.l2bus.cpu_side_ports

# L2 → L3
system.l2cache.cpu_side = system.l2bus.mem_side_ports
system.l2cache.mem_side = system.l3bus.cpu_side_ports

# L3 → Memory
system.l3cache.cpu_side = system.l3bus.mem_side_ports
system.l3cache.mem_side = system.membus.cpu_side_ports

# DRAM
system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR4_2400_16x4()
system.mem_ctrl.dram.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports
system.system_port = system.membus.cpu_side_ports

# CPU ports
system.cpu.icache_port = system.cpu.icache.cpu_side
system.cpu.dcache_port = system.cpu.dcache.cpu_side

# X86 interrupt controller
system.cpu.createInterruptController()
system.cpu.interrupts[0].pio = system.membus.mem_side_ports
system.cpu.interrupts[0].int_requestor = system.membus.cpu_side_ports
system.cpu.interrupts[0].int_responder = system.membus.mem_side_ports

# Workload
system.workload = SEWorkload.init_compatible(args.binary)
cmd = [args.binary]
if args.binary_args:
    cmd.extend(args.binary_args.split())
process = Process(cmd=cmd)
system.cpu.workload = process
system.cpu.createThreads()

# Instantiate and run
root = Root(full_system=False, system=system)
m5.instantiate()

print(f"=== gem5 Policy Test ===")
print(f"  Binary:  {args.binary}")
print(f"  Policy:  {args.policy}"
      + (f" ({args.ecg_mode})" if args.policy == "ECG" else ""))
print(f"  L3:      8MB 16-way")
print(f"  CPU:     TimingSimpleCPU @ 2GHz")
print(f"========================")

exit_event = m5.simulate()
print(f"\nDone @ tick {m5.curTick()}: {exit_event.getCause()}")
