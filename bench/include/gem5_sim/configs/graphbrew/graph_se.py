#!/usr/bin/env python3
"""
gem5 SE-mode configuration for GraphBrew graph benchmarks.

Configures a TimingSimpleCPU (or DerivO3CPU) with a 3-level cache hierarchy
matching ECG defaults, running a graph benchmark binary under SE (syscall
emulation) mode.

Usage:
    gem5.opt graph_se.py --binary bench/bin/pr \\
        --options "-f graph.sg -s -n 1 -o 5" \\
        --policy GRASP \\
        --graph-metadata results/gem5_metadata/pokec/context.json

    gem5.opt graph_se.py --binary bench/bin/bfs \\
        --options "-f graph.sg -s -n 1 -o 0" \\
        --policy ECG --ecg-mode DBG_PRIMARY \\
        --cpu-type O3

    gem5.opt graph_se.py --binary bench/bin/pr \\
        --options "-f graph.sg -s -n 1 -o 5" \\
        --policy GRASP --prefetcher DROPLET
"""

import argparse
import os
import sys

import m5
from m5.objects import *
from m5.util import addToPath

# Add configs directory to path for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
addToPath(script_dir)

from graph_cache_config import (
    make_l1d_cache, make_l1i_cache, make_l2_cache, make_l3_cache,
    make_replacement_policy, make_droplet_prefetcher, make_ecg_pfx_prefetcher, DEFAULTS,
)
from graph_metadata_loader import load_graph_metadata, metadata_summary


RUNTIME_SIDEBAND_FILES = (
    ("GEM5_GRAPHBREW_CTX", "/tmp/gem5_graphbrew_ctx.json"),
    ("GEM5_POPT_MATRIX", "/tmp/gem5_popt_matrix.bin"),
    ("GEM5_GRAPHBREW_OUT_EDGES", "/tmp/gem5_graphbrew_out_edges.bin"),
    ("GEM5_GRAPHBREW_IN_EDGES", "/tmp/gem5_graphbrew_in_edges.bin"),
)


def clear_runtime_sideband_files():
    """Remove stale benchmark-written metadata before simulation starts."""
    for env_name, default_path in RUNTIME_SIDEBAND_FILES:
        path = os.environ.get(env_name, default_path)
        if not path:
            continue
        try:
            os.remove(path)
            print(f"  Cleared stale runtime sideband: {path}")
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"Warning: could not clear stale runtime sideband {path}: {exc}")


def needs_vertex_hints(args):
    """Return whether benchmark should emit explicit P-OPT current-vertex hints."""
    return args.policy == "POPT" or (
        args.policy == "ECG" and args.ecg_mode != "DBG_ONLY"
    )


def benchmark_environment(args):
    """Environment variables visible inside the simulated benchmark."""
    ecg_grasp_popt = args.policy == "ECG" and args.ecg_mode == "ECG_GRASP_POPT"
    ecg_pfx_metadata = args.prefetcher == "ECG_PFX" or ecg_grasp_popt
    env = [
        f"GEM5_ENABLE_VERTEX_HINTS={1 if needs_vertex_hints(args) else 0}",
        f"GEM5_ENABLE_ECG_PFX_HINTS={1 if ecg_pfx_metadata else 0}",
        f"GEM5_ECG_PFX_LOOKAHEAD={args.ecg_pfx_lookahead if ecg_pfx_metadata else 0}",
        f"GEM5_ECG_PFX_HINT_FILTER={args.ecg_pfx_hint_filter if args.prefetcher == 'ECG_PFX' else 0}",
        f"GEM5_ECG_PFX_FILTER_ELEM_SIZE=4",
        f"GEM5_ECG_PFX_FILTER_LINE_SIZE=64",
        f"GEM5_ENABLE_ECG_EXTRACT={1 if ecg_grasp_popt or (args.prefetcher == 'ECG_PFX' and args.ecg_pfx_delivery == 'instruction') or os.environ.get('GEM5_FORCE_ECG_EXTRACT') == '1' else 0}",
    ]
    # Propagate ECG mode + per-edge-mask lookahead from the outer harness
    # env so kernel-side `ecg_pfx_mode` reads the value roi_matrix.py set.
    for pass_name in (
        "GEM5_ECG_PFX_MODE",
        "ECG_PREFETCH_MODE",
        "GEM5_ECG_EDGE_MASK_LOOKAHEAD",
        "ECG_EDGE_MASK_LOOKAHEAD",
        "ECG_EDGE_MASK_EPOCH",
        "ECG_EDGE_MASK_LINEMIN",
        "ECG_EDGE_MASK_EPOCHS",
        # Path A (epoch-filtered DROPLET lookahead): the kernel gates the
        # next-K lookahead on these; gem5 SE mode does NOT inherit the host
        # env, so they must be forwarded explicitly or the kernel falls back
        # to Path B (lean_pfx_k=0).
        "ECG_EDGE_MASK_PREFETCH",
        "ECG_PREFETCH_EPOCH_FILTER",
        "ECG_PREFETCH_EPOCH_THRESH_PCT",
    ):
        outer = os.environ.get(pass_name)
        if outer is not None and outer != "":
            env.append(f"{pass_name}={outer}")
    for env_name, default_path in RUNTIME_SIDEBAND_FILES:
        env.append(f"{env_name}={os.environ.get(env_name, default_path)}")
    return tuple(env)


def parse_args():
    """Parse command-line arguments for graph benchmark SE simulation."""
    parser = argparse.ArgumentParser(
        description="gem5 SE-mode config for GraphBrew graph benchmarks")

    parser.add_argument("--binary", required=True,
        help="Path to the benchmark binary (e.g., bench/bin/pr)")
    parser.add_argument("--options", default="",
        help="Command-line arguments for the benchmark binary")

    # Cache policy
    parser.add_argument("--policy", default="LRU",
        choices=["LRU", "FIFO", "SRRIP", "RANDOM", "GRASP", "POPT", "ECG"],
        help="Cache replacement policy for L3 (default: LRU)")
    parser.add_argument("--ecg-mode", default="DBG_PRIMARY",
        choices=["DBG_PRIMARY", "POPT_PRIMARY", "ECG_GRASP_POPT", "DBG_ONLY",
                 "ECG_EMBEDDED", "ECG_COMBINED"],
        help="ECG eviction mode (only used with --policy ECG)")
    parser.add_argument("--l1-policy", default="LRU",
        help="L1 cache replacement policy (default: LRU)")
    parser.add_argument("--l2-policy", default="LRU",
        help="L2 cache replacement policy (default: LRU)")

    # Prefetcher
    parser.add_argument("--prefetcher", default="none",
        choices=["none", "DROPLET", "ECG_PFX"],
        help="Graph prefetcher to attach (default: none)")
    parser.add_argument("--prefetcher-level", default="l2",
        choices=["l1d", "l2"],
        help="Cache level for graph prefetcher attachment (default: l2)")
    parser.add_argument("--droplet-prefetch-degree", type=int, default=1,
        help="DROPLET edge-stream cache lines to prefetch per trigger (artifact default: 1)")
    parser.add_argument("--droplet-indirect-degree", type=int, default=16,
        help="DROPLET neighbor IDs to translate into property prefetches per edge line (artifact default: 16)")
    parser.add_argument("--droplet-stride-table-size", type=int, default=64,
        help="DROPLET stream table entries (artifact config streams default: 64)")
    parser.add_argument("--ecg-pfx-lookahead", default="4",
        help="Future-neighbor lookahead distance for benchmark-emitted ECG_PFX hints")
    parser.add_argument("--ecg-pfx-hint-filter", default="16",
        help="Recent-target filter capacity before emitting ECG_PFX hints; 0 disables filtering")
    parser.add_argument("--ecg-pfx-delivery", default="explicit-hint",
        choices=["explicit-hint", "instruction"],
        help="ECG_PFX delivery path: explicit m5ops hint, RISC-V ecg.extract, or x86 gem5 pseudo-op instruction")

    # CPU model
    parser.add_argument("--cpu-type", default="timing",
        choices=["timing", "O3", "minor"],
        help="CPU model (default: timing = TimingSimpleCPU)")

    # Graph metadata
    parser.add_argument("--graph-metadata", default="",
        help="Path to graph metadata JSON file for graph-aware policies")

    # Cache sizes (override defaults)
    parser.add_argument("--l1d-size", default=DEFAULTS["l1d_size"])
    parser.add_argument("--l1i-size", default=DEFAULTS["l1i_size"])
    parser.add_argument("--l2-size", default=DEFAULTS["l2_size"])
    parser.add_argument("--l3-size", default=DEFAULTS["l3_size"])
    parser.add_argument("--l3-ways", type=int, default=DEFAULTS["l3_assoc"],
        help="L3 associativity / data ways (default: GraphBrew DEFAULTS l3_assoc)")

    return parser.parse_args()


def create_system(args):
    """Create the full gem5 system for graph benchmark simulation."""

    system = System()
    system.clk_domain = SrcClockDomain()
    system.clk_domain.clock = "2GHz"
    system.clk_domain.voltage_domain = VoltageDomain()
    system.mem_mode = "timing"
    system.mem_ranges = [AddrRange("4GB")]

    # ── CPU ──
    if args.cpu_type == "O3":
        system.cpu = DerivO3CPU()
    elif args.cpu_type == "minor":
        system.cpu = MinorCPU()
    else:
        system.cpu = TimingSimpleCPU()

    # ── L3 replacement policy (graph-aware) ──
    l3_policy_kwargs = {}
    if args.policy == "ECG":
        l3_policy_kwargs["ecg_mode"] = args.ecg_mode
    if args.policy in ("GRASP", "POPT", "ECG"):
        l3_policy_kwargs["num_buckets"] = 11

    # ── Cache hierarchy ──
    system.cpu.icache = make_l1i_cache(
        policy=args.l1_policy, size=args.l1i_size)
    system.cpu.dcache = make_l1d_cache(
        policy=args.l1_policy, size=args.l1d_size)
    system.l2cache = make_l2_cache(
        policy=args.l2_policy, size=args.l2_size)
    system.l3cache = make_l3_cache(
        policy=args.policy, size=args.l3_size, assoc=args.l3_ways,
        **l3_policy_kwargs)

    # ── Prefetcher ──
    if args.prefetcher == "DROPLET":
        droplet_kwargs = {
            "prefetch_degree": args.droplet_prefetch_degree,
            "indirect_degree": args.droplet_indirect_degree,
            "stride_table_size": args.droplet_stride_table_size,
        }
        if args.prefetcher_level == "l1d":
            system.cpu.dcache.prefetcher = make_droplet_prefetcher(**droplet_kwargs)
            # S68-MMU-PATCH: gem5 Queued::notify drops cross-page
            # prefetches unless prefetcher.mmu is set. See
            # docs/findings/gem5_implementation_audit_v1.md.
            _pf = system.cpu.dcache.prefetcher
            if hasattr(system.cpu, 'mmu'):
                _pf.registerMMU(system.cpu.mmu)
            elif hasattr(system.cpu, 'dtb'):
                _pf.registerMMU(system.cpu.dtb)
        else:
            system.l2cache.prefetcher = make_droplet_prefetcher(**droplet_kwargs)
            # S68-MMU-PATCH: gem5 Queued::notify drops cross-page
            # prefetches unless prefetcher.mmu is set. See
            # docs/findings/gem5_implementation_audit_v1.md.
            _pf = system.l2cache.prefetcher
            if hasattr(system.cpu, 'mmu'):
                _pf.registerMMU(system.cpu.mmu)
            elif hasattr(system.cpu, 'dtb'):
                _pf.registerMMU(system.cpu.dtb)
    elif args.prefetcher == "ECG_PFX":
        if args.prefetcher_level == "l1d":
            system.cpu.dcache.prefetcher = make_ecg_pfx_prefetcher()
            # S68-MMU-PATCH: gem5 Queued::notify drops cross-page
            # prefetches unless prefetcher.mmu is set. See
            # docs/findings/gem5_implementation_audit_v1.md.
            _pf = system.cpu.dcache.prefetcher
            if hasattr(system.cpu, 'mmu'):
                _pf.registerMMU(system.cpu.mmu)
            elif hasattr(system.cpu, 'dtb'):
                _pf.registerMMU(system.cpu.dtb)
        else:
            system.l2cache.prefetcher = make_ecg_pfx_prefetcher()
            # S68-MMU-PATCH: gem5 Queued::notify drops cross-page
            # prefetches unless prefetcher.mmu is set. See
            # docs/findings/gem5_implementation_audit_v1.md.
            _pf = system.l2cache.prefetcher
            if hasattr(system.cpu, 'mmu'):
                _pf.registerMMU(system.cpu.mmu)
            elif hasattr(system.cpu, 'dtb'):
                _pf.registerMMU(system.cpu.dtb)

    # ── Memory bus connections ──
    system.membus = SystemXBar()
    system.l2bus = L2XBar()

    # L1 → L2
    system.cpu.icache.mem_side = system.l2bus.cpu_side_ports
    system.cpu.dcache.mem_side = system.l2bus.cpu_side_ports

    # L2 → L3
    system.l2cache.cpu_side = system.l2bus.mem_side_ports
    system.l3bus = L2XBar()
    system.l2cache.mem_side = system.l3bus.cpu_side_ports

    # L3 → Memory
    system.l3cache.cpu_side = system.l3bus.mem_side_ports
    system.l3cache.mem_side = system.membus.cpu_side_ports

    # Memory controller
    system.mem_ctrl = MemCtrl()
    system.mem_ctrl.dram = DDR4_2400_16x4()
    system.mem_ctrl.dram.range = system.mem_ranges[0]
    system.mem_ctrl.port = system.membus.mem_side_ports

    # System port
    system.system_port = system.membus.cpu_side_ports

    # Interrupt controller
    system.cpu.createInterruptController()
    if hasattr(system.cpu, "interrupts"):
        for interrupt in system.cpu.interrupts:
            if hasattr(interrupt, "pio"):
                interrupt.pio = system.membus.mem_side_ports
            if hasattr(interrupt, "int_requestor"):
                interrupt.int_requestor = system.membus.cpu_side_ports
            if hasattr(interrupt, "int_responder"):
                interrupt.int_responder = system.membus.mem_side_ports

    # CPU ports
    system.cpu.icache_port = system.cpu.icache.cpu_side
    system.cpu.dcache_port = system.cpu.dcache.cpu_side

    # ── Workload ──
    binary = args.binary
    system.workload = SEWorkload.init_compatible(binary)

    process = Process()
    process.cmd = [binary] + args.options.split()
    process.env = benchmark_environment(args)
    system.cpu.workload = process
    system.cpu.createThreads()

    return system


def main():
    args = parse_args()

    clear_runtime_sideband_files()

    # Load graph metadata if provided
    if args.graph_metadata:
        metadata = load_graph_metadata(args.graph_metadata)
        print(metadata_summary(metadata))
        print()
    else:
        if args.policy in ("GRASP", "POPT", "ECG"):
            print(f"Info: --graph-metadata not provided for {args.policy} policy.")
            print("  C++ graph policies will use benchmark-written runtime sideband.")
            print("  Static Python metadata is only needed for pre-exported contexts.")
            print()

    print(f"Configuring gem5 SE simulation:")
    print(f"  Binary:     {args.binary}")
    print(f"  Options:    {args.options}")
    print(f"  CPU:        {args.cpu_type}")
    print(f"  L3 Policy:  {args.policy}"
          + (f" ({args.ecg_mode})" if args.policy == "ECG" else ""))
    print(f"  Prefetcher: {args.prefetcher} ({args.prefetcher_level})")
    print()

    # Create system
    system = create_system(args)

    # Create root and instantiate
    root = Root(full_system=False, system=system)
    m5.instantiate()

    print("Starting simulation...")
    exit_event = m5.simulate()
    print(f"Exiting @ tick {m5.curTick()} because {exit_event.getCause()}")


if __name__ == "__m5_main__":
    main()
