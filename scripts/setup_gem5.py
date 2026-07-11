#!/usr/bin/env python3
"""
Setup script for gem5 integration with GraphBrew.

Clones gem5 into bench/include/gem5_sim/gem5/, applies GraphBrew overlay patches
(GRASP, P-OPT, ECG replacement policies and DROPLET prefetcher), and builds
gem5 for the requested ISAs.

Usage:
    python scripts/setup_gem5.py                        # Clone + build X86 (default)
    python scripts/setup_gem5.py --isa X86 RISCV        # Build both ISAs
    python scripts/setup_gem5.py --isa RISCV --build-type debug
    python scripts/setup_gem5.py --skip-build            # Clone + patch only
    python scripts/setup_gem5.py --clean                 # Remove cloned gem5
    python scripts/setup_gem5.py --rebuild               # Force rebuild

The script is idempotent: re-running it will skip completed steps.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# =============================================================================
# Constants
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BENCH_DIR = PROJECT_ROOT / "bench"
GEM5_SIM_DIR = BENCH_DIR / "include" / "gem5_sim"
GEM5_DIR = GEM5_SIM_DIR / "gem5"
OVERLAYS_DIR = GEM5_SIM_DIR / "overlays"
VERSION_FILE = GEM5_SIM_DIR / ".gem5_version"

GEM5_REPO_URL = "https://github.com/gem5/gem5.git"
GEM5_DEFAULT_TAG = "v24.0"

VALID_ISAS = ("X86", "RISCV", "ARM")
VALID_BUILD_TYPES = ("opt", "debug", "fast")

# Overlay mappings: source (relative to overlays/) -> destination (relative to gem5/src/)
OVERLAY_FILE_MAP = {
    # Replacement policies
    "mem/cache/replacement_policies/grasp_rp.hh":
        "mem/cache/replacement_policies/grasp_rp.hh",
    "mem/cache/replacement_policies/grasp_rp.cc":
        "mem/cache/replacement_policies/grasp_rp.cc",
    "mem/cache/replacement_policies/popt_rp.hh":
        "mem/cache/replacement_policies/popt_rp.hh",
    "mem/cache/replacement_policies/popt_rp.cc":
        "mem/cache/replacement_policies/popt_rp.cc",
    "mem/cache/replacement_policies/ecg_rp.hh":
        "mem/cache/replacement_policies/ecg_rp.hh",
    "mem/cache/replacement_policies/ecg_rp.cc":
        "mem/cache/replacement_policies/ecg_rp.cc",
    "mem/cache/replacement_policies/ecg_victim_policy.hh":
        "mem/cache/replacement_policies/ecg_victim_policy.hh",
    "mem/cache/replacement_policies/ecg_epoch_request_ext.hh":
        "mem/cache/replacement_policies/ecg_epoch_request_ext.hh",
    "mem/cache/replacement_policies/graph_cache_context_gem5.hh":
        "mem/cache/replacement_policies/graph_cache_context_gem5.hh",
    "mem/cache/replacement_policies/GraphReplacementPolicies.py":
        "mem/cache/replacement_policies/GraphReplacementPolicies.py",
    # Prefetcher
    "mem/cache/prefetch/droplet.hh":
        "mem/cache/prefetch/droplet.hh",
    "mem/cache/prefetch/droplet.cc":
        "mem/cache/prefetch/droplet.cc",
    "mem/cache/prefetch/ecg_pfx.hh":
        "mem/cache/prefetch/ecg_pfx.hh",
    "mem/cache/prefetch/ecg_pfx.cc":
        "mem/cache/prefetch/ecg_pfx.cc",
    "mem/cache/prefetch/GraphPrefetchers.py":
        "mem/cache/prefetch/GraphPrefetchers.py",
    # RISC-V ECG custom instruction scaffold
    "arch/riscv/isa/formats/ecg.isa":
        "arch/riscv/isa/formats/ecg.isa",
}

# Patches to apply (relative to overlays/)
PATCH_FILES = [
    "mem/cache/replacement_policies/SConscript.patch",
    "mem/cache/prefetch/SConscript.patch",
]

# Unified-diff patches to apply via `patch -p1` (relative to overlays/).
# Each entry: (overlay_relpath, target_dir_relative_to_gem5_root)
# These are tracked as patches (not full file copies) so upstream gem5
# changes are easier to merge.
UNIFIED_DIFF_PATCHES = [
    # S68 queue-servicing fix: nextPrefetchReadyTime returns curTick()
    # when pfqMissingTranslation has entries even if pfq is empty.
    # Required for prefetchers like ECG_PFX that emit only cross-page
    # candidates. See docs/findings/gem5_implementation_audit_v1.md.
    ("mem/cache/prefetch/queued_hh.patch", "."),
    # S68 latency-readiness guard: getPacket() returns nullptr if the
    # front-of-queue prefetch's tick is in the future, preserving the
    # prefetcher's latency contract for cycle-accurate parity.
    ("mem/cache/prefetch/queued_cc_latency.patch", "."),
    # ECG ecg.load OoO producer: bind the per-dynamic {dest,epoch} to the
    # ecg.load's own demand Request (EcgEpochExtension) so the epoch reaches
    # the LLC race-free on DerivO3CPU (the single-slot mailbox is overwritten
    # ~100% of the time under OoO — see docs/findings/ooo_ecg_load_propagation_
    # and_design.md). exec_context.hh adds a default-noop setEcgLoadHint hook;
    # o3/dyn_inst.hh overrides it with per-dynamic state; o3/lsq.cc attaches the
    # extension in LSQRequest::addReq (gated on env GEM5_ECG_PRODUCER). The
    # ea_code caller lives in the decoder_ecg_extract.isa snippet.
    ("cpu/exec_context_ecg_producer.patch", "."),
    ("cpu/o3/dyn_inst_ecg_producer.patch", "."),
    ("cpu/o3/lsq_ecg_producer.patch", "."),
]


# =============================================================================
# Utility Functions
# =============================================================================

class Logger:
    """Colored terminal logger."""
    BLUE = "\033[0;34m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[0;33m"
    RED = "\033[0;31m"
    NC = "\033[0m"

    @staticmethod
    def info(msg: str):
        print(f"{Logger.BLUE}[gem5-setup]{Logger.NC} {msg}")

    @staticmethod
    def success(msg: str):
        print(f"{Logger.GREEN}[gem5-setup]{Logger.NC} {msg}")

    @staticmethod
    def warn(msg: str):
        print(f"{Logger.YELLOW}[gem5-setup]{Logger.NC} {msg}")

    @staticmethod
    def error(msg: str):
        print(f"{Logger.RED}[gem5-setup]{Logger.NC} {msg}", file=sys.stderr)

    @staticmethod
    def step(num: int, total: int, msg: str):
        print(f"{Logger.BLUE}[{num}/{total}]{Logger.NC} {msg}")


log = Logger()


def run_cmd(cmd: list, cwd: str = None, check: bool = True,
            capture: bool = False, env: dict = None) -> subprocess.CompletedProcess:
    """Run a command with logging."""
    cmd_str = " ".join(str(c) for c in cmd)
    log.info(f"  $ {cmd_str}")
    merged_env = {**os.environ, **(env or {})}
    return subprocess.run(
        cmd, cwd=cwd, check=check, capture_output=capture,
        text=True, env=merged_env,
    )


def check_prerequisites():
    """Verify required tools are installed."""
    required = {
        "git": "git --version",
        "g++": "g++ --version",
        "python3": "python3 --version",
        "scons": "scons --version",
    }
    missing = []
    for tool, cmd in required.items():
        try:
            subprocess.run(cmd.split(), capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing.append(tool)

    if missing:
        log.error(f"Missing prerequisites: {', '.join(missing)}")
        log.error("Install with:")
        log.error("  sudo apt install build-essential git scons python3-dev")
        log.error("  pip install scons  # if not available via apt")
        sys.exit(1)

    # Check GCC version >= 9
    try:
        result = subprocess.run(
            ["g++", "-dumpversion"], capture_output=True, text=True, check=True
        )
        major = int(result.stdout.strip().split(".")[0])
        if major < 9:
            log.warn(f"g++ version {result.stdout.strip()} detected. gem5 requires >= 9.")
    except (subprocess.CalledProcessError, ValueError):
        pass

    log.success("All prerequisites satisfied.")


# =============================================================================
# Core Steps
# =============================================================================

def clone_gem5(tag: str, force: bool = False):
    """Clone gem5 repository at the specified tag."""
    if GEM5_DIR.exists() and not force:
        log.info(f"gem5 already cloned at {GEM5_DIR}")
        # Check if version matches
        if VERSION_FILE.exists():
            stored = VERSION_FILE.read_text().strip()
            if stored == tag:
                log.success(f"gem5 version matches ({tag}), skipping clone.")
                return
            else:
                log.warn(f"Version mismatch: have {stored}, want {tag}.")
                log.warn("Use --clean first, or --force to re-clone.")
                sys.exit(1)
        return

    log.info(f"Cloning gem5 ({tag}) into {GEM5_DIR}...")
    log.info("This may take a few minutes (~300 MB download).")

    if force and GEM5_DIR.exists():
        shutil.rmtree(GEM5_DIR)

    run_cmd([
        "git", "clone", "--depth", "1", "--branch", tag,
        GEM5_REPO_URL, str(GEM5_DIR),
    ])

    # Record version
    VERSION_FILE.write_text(f"{tag}\n")
    log.success(f"gem5 cloned successfully ({tag}).")


def apply_overlays():
    """Copy overlay source files into the cloned gem5 tree."""
    log.info("Applying GraphBrew overlay files to gem5/src/...")
    applied = 0

    for src_rel, dst_rel in OVERLAY_FILE_MAP.items():
        src = OVERLAYS_DIR / src_rel
        dst = GEM5_DIR / "src" / dst_rel

        if not src.exists():
            log.warn(f"  Overlay file not found (will be created later): {src_rel}")
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        applied += 1

    log.success(f"Applied {applied} overlay files.")


def apply_patches():
    """Apply SConscript patches to register new SimObjects."""
    log.info("Applying SConscript patches...")

    for patch_rel in PATCH_FILES:
        patch_file = OVERLAYS_DIR / patch_rel
        if not patch_file.exists():
            log.warn(f"  Patch file not found (will be created later): {patch_rel}")
            continue

        # Determine target SConscript
        patch_dir = Path(patch_rel).parent
        target_sconscript = GEM5_DIR / "src" / patch_dir / "SConscript"

        if not target_sconscript.exists():
            log.warn(f"  Target SConscript not found: {target_sconscript}")
            continue

        # Read patch content (simple append-style patches)
        patch_content = patch_file.read_text()
        current_content = target_sconscript.read_text()

        if patch_rel == "mem/cache/prefetch/SConscript.patch":
            old_simobject = "SimObject('GraphPrefetchers.py', sim_objects=['GraphDropletPrefetcher'])"
            new_simobject = "SimObject('GraphPrefetchers.py', sim_objects=['GraphDropletPrefetcher', 'GraphEcgPfxPrefetcher'])"
            if old_simobject in current_content and new_simobject not in current_content:
                current_content = current_content.replace(old_simobject, new_simobject)
                target_sconscript.write_text(current_content)

        patch_lines = [line for line in patch_content.splitlines() if line.strip()]
        missing_lines = [line for line in patch_lines if line not in current_content]
        if not missing_lines:
            log.info(f"  Patch already applied: {patch_dir}/SConscript")
            continue

        # Append patch content
        marker = "# --- GraphBrew graph-aware policies ---"
        with open(target_sconscript, "a") as f:
            f.write(f"\n{marker}\n")
            f.write("\n".join(missing_lines) + "\n")

        log.success(f"  Patched: {patch_dir}/SConscript")


def apply_unified_diff_patches():
    """Apply unified-diff patches (via `patch -p1`) under gem5/.

    Each patch is checked for idempotency via `patch --dry-run` before
    actually applying. Re-runs are safe and produce no-op output if the
    patch is already applied.
    """
    import subprocess
    log.info("Applying unified-diff patches under gem5/...")

    for overlay_rel, target_dir in UNIFIED_DIFF_PATCHES:
        patch_file = OVERLAYS_DIR / overlay_rel
        if not patch_file.exists():
            log.warn(f"  Patch file not found: {overlay_rel}")
            continue

        target = GEM5_DIR / target_dir
        if not target.exists():
            log.warn(f"  Target dir not found: {target}")
            continue

        # Dry-run forward to see if already applied
        dry_fwd = subprocess.run(
            ["patch", "-p1", "--dry-run", "--silent", "-i", str(patch_file)],
            cwd=target, capture_output=True, text=True,
        )
        # If dry-run forward says "already applied" or "reversed", skip
        if "previously applied" in (dry_fwd.stdout + dry_fwd.stderr).lower() or \
           "reversed" in (dry_fwd.stdout + dry_fwd.stderr).lower():
            log.info(f"  Patch already applied (skip): {overlay_rel}")
            continue

        if dry_fwd.returncode != 0:
            log.warn(f"  Dry-run failed for {overlay_rel}:")
            log.warn(f"    {dry_fwd.stdout.strip()}")
            log.warn(f"    {dry_fwd.stderr.strip()}")
            continue

        apply = subprocess.run(
            ["patch", "-p1", "-i", str(patch_file)],
            cwd=target, capture_output=True, text=True,
        )
        if apply.returncode == 0:
            log.success(f"  Applied: {overlay_rel}")
        else:
            log.warn(f"  Apply failed for {overlay_rel}:")
            log.warn(f"    {apply.stdout.strip()}")
            log.warn(f"    {apply.stderr.strip()}")


def apply_current_vertex_pseudo_inst_patch():
    """Patch m5_work_begin to carry GraphBrew graph hints."""
    target = GEM5_DIR / "src" / "sim" / "pseudo_inst.cc"
    if not target.exists():
        log.warn(f"  pseudo_inst.cc not found: {target}")
        return

    content = target.read_text()
    include_line = '#include "mem/cache/replacement_policies/graph_cache_context_gem5.hh"\n'
    include_anchor = '#include "sim/pseudo_inst.hh"\n'
    if include_line not in content:
        if include_anchor not in content:
            log.warn("  Could not locate pseudo_inst include anchor")
        else:
            content = content.replace(include_anchor, include_anchor + "\n" + include_line, 1)

    workbegin_anchor = '    DPRINTF(PseudoInst, "pseudo_inst::workbegin(%i, %i)\\n", workid, threadid);\n'
    # Upgrade the legacy content-based PFX/mask multiplexing. Epoch zero is a
    # valid extraction, so payload bits cannot distinguish a bare target from a
    # full mask; use a dedicated work ID for extraction.
    legacy_pfx_start = (
        "    if (workid == replacement_policy::graph::"
        "GRAPHBREW_ECG_PFX_TARGET_WORK_ID) {\n"
    )
    if legacy_pfx_start in content and "if ((threadid >> 24) != 0)" in content:
        start = content.index(legacy_pfx_start)
        end = content.index("    }\n\n", start) + len("    }\n\n")
        content = (
            content[:start]
            + legacy_pfx_start
            + "        replacement_policy::graph::setPrefetchTargetHint(threadid);\n"
            + "        return;\n"
            + "    }\n\n"
            + content[end:]
        )

    hint_blocks = []
    if "GRAPHBREW_SET_VERTEX_WORK_ID" not in content:
        hint_blocks.append(
            "    if (workid == replacement_policy::graph::GRAPHBREW_SET_VERTEX_WORK_ID) {\n"
            "        replacement_policy::graph::setCurrentVertexHint(threadid);\n"
            "        return;\n"
            "    }\n\n"
        )
    if "GRAPHBREW_ECG_PFX_TARGET_WORK_ID" not in content:
        hint_blocks.append(
            "    if (workid == replacement_policy::graph::GRAPHBREW_ECG_PFX_TARGET_WORK_ID) {\n"
            "        replacement_policy::graph::setPrefetchTargetHint(threadid);\n"
            "        return;\n"
            "    }\n\n"
        )
    if "GRAPHBREW_ECG_EXTRACT_MASK_WORK_ID" not in content:
        hint_blocks.append(
            "    if (workid == replacement_policy::graph::GRAPHBREW_ECG_EXTRACT_MASK_WORK_ID) {\n"
            "        uint64_t fat_mask = threadid;\n"
            "        uint32_t dest_id = static_cast<uint32_t>(fat_mask & 0xFFFFFFULL);\n"
            "        uint16_t epoch = static_cast<uint16_t>((fat_mask >> 24) & 0xFFFFULL);\n"
            "        uint32_t pfx_target = static_cast<uint32_t>((fat_mask >> 40) & 0xFFFFFFULL);\n"
            "        replacement_policy::graph::storeEcgMetadataByVertex(dest_id, 0, 0, epoch);\n"
            "        replacement_policy::graph::setDecodedEcgExtractHint(dest_id, 0, 0, 0, epoch);\n"
            "        if (pfx_target != 0) replacement_policy::graph::setPrefetchTargetHint(pfx_target);\n"
            "        return;\n"
            "    }\n\n"
        )
    if "GRAPHBREW_ECG_PFX_TARGET_EPOCH_WORK_ID" not in content:
        # Path A (epoch-filtered DROPLET lookahead): threadid = target | epoch<<32.
        # Record the candidate epoch in the bounded in-flight buffer (so the
        # prefetched line is stamped at fill) and push the bare target. No
        # single-slot touch (no demand-epoch corruption), no 24-bit truncation.
        hint_blocks.append(
            "    if (workid == replacement_policy::graph::GRAPHBREW_ECG_PFX_TARGET_EPOCH_WORK_ID) {\n"
            "        uint32_t pfxa_target = static_cast<uint32_t>(threadid & 0xFFFFFFFFULL);\n"
            "        uint16_t pfxa_epoch = static_cast<uint16_t>((threadid >> 32) & 0xFFFFULL);\n"
            "        replacement_policy::graph::recordPendingPrefetchEpoch(pfxa_target, pfxa_epoch);\n"
            "        replacement_policy::graph::setPrefetchTargetHint(pfxa_target);\n"
            "        return;\n"
            "    }\n\n"
        )
    if hint_blocks:
        insertion = "".join(hint_blocks)
        if workbegin_anchor in content:
            content = content.replace(workbegin_anchor, workbegin_anchor + insertion, 1)
        elif "GRAPHBREW_SET_VERTEX_WORK_ID" in content:
            content = content.replace(
                "        replacement_policy::graph::setCurrentVertexHint(threadid);\n"
                "        return;\n"
                "    }\n\n",
                "        replacement_policy::graph::setCurrentVertexHint(threadid);\n"
                "        return;\n"
                "    }\n\n" + insertion,
                1,
            )
        else:
            log.warn("  Could not locate workbegin patch anchor")

    target.write_text(content)
    log.success("  Patched sim/pseudo_inst.cc for GraphBrew current-vertex hints.")


def insert_once(content: str, anchor: str, insertion: str, label: str) -> tuple[str, bool]:
    if insertion.strip() in content:
        return content, False
    if anchor not in content:
        log.warn(f"  Could not locate {label} patch anchor")
        return content, False
    return content.replace(anchor, anchor + insertion, 1), True


def apply_riscv_ecg_extract_patch():
    """Patch RISC-V ISA description for GraphBrew ecg.extract scaffold."""
    isa_dir = GEM5_DIR / "src" / "arch" / "riscv" / "isa"
    formats_path = isa_dir / "formats" / "formats.isa"
    includes_path = isa_dir / "includes.isa"
    decoder_path = isa_dir / "decoder.isa"
    snippet_path = OVERLAYS_DIR / "arch" / "riscv" / "isa" / "decoder_ecg_extract.isa"

    if not formats_path.exists() or not includes_path.exists() or not decoder_path.exists():
        log.warn("  RISC-V ISA files not found; skipping ECG extract patch")
        return

    formats = formats_path.read_text()
    formats, changed = insert_once(
        formats,
        '##include "m5ops.isa"\n',
        '##include "ecg.isa"\n',
        "RISC-V formats include",
    )
    if changed:
        formats_path.write_text(formats)

    includes = includes_path.read_text()
    includes, changed = insert_once(
        includes,
        '#include "sim/pseudo_inst.hh"\n',
        '#include "mem/cache/replacement_policies/graph_cache_context_gem5.hh"\n',
        "RISC-V exec include",
    )
    if changed:
        includes_path.write_text(includes)

    if not snippet_path.exists():
        log.warn(f"  RISC-V ECG decoder snippet not found: {snippet_path}")
        return
    snippet = snippet_path.read_text()
    decoder = decoder_path.read_text()
    marker = "        // GraphBrew ECG custom-0 instruction space.\n"
    load_anchor = "        0x00: decode FUNCT3 {\n"
    marker_pos = decoder.find(marker)
    load_pos = decoder.find(load_anchor, marker_pos if marker_pos >= 0 else 0)
    if marker_pos >= 0 and load_pos > marker_pos:
        decoder = decoder[:marker_pos] + decoder[load_pos:]
    decoder, changed = insert_once(
        decoder,
        '    0x3: decode OPCODE5 {\n',
        snippet,
        "RISC-V custom-0 decoder",
    )
    if changed:
        decoder_path.write_text(decoder)
    log.success("  Patched RISC-V ecg.extract custom-0 scaffold.")


def build_gem5(isas: list, build_type: str, jobs: int):
    """Build gem5 for the specified ISAs."""
    for isa in isas:
        target = f"build/{isa}/gem5.{build_type}"
        binary = GEM5_DIR / target

        if binary.exists():
            log.info(f"gem5 binary already exists: {target}")
            continue

        log.info(f"Building gem5 for {isa} ({build_type})...")
        log.info(f"This will take 10-30 minutes with {jobs} jobs.")

        run_cmd(
            ["scons", f"-j{jobs}", target],
            cwd=str(GEM5_DIR),
        )

        if binary.exists():
            log.success(f"Built: {target}")
        else:
            log.error(f"Build failed: {target} not found after scons.")
            sys.exit(1)


def verify_build(isas: list, build_type: str):
    """Verify gem5 binaries are functional."""
    for isa in isas:
        binary = GEM5_DIR / f"build/{isa}/gem5.{build_type}"
        if not binary.exists():
            log.warn(f"Binary not found: {binary}")
            continue

        result = run_cmd(
            [str(binary), "--help"],
            check=False, capture=True,
        )
        if result.returncode == 0 and "gem5" in result.stdout.lower():
            log.success(f"Verified: {isa}/gem5.{build_type} is functional.")
        else:
            log.warn(f"Verification issue with {isa}/gem5.{build_type}")


def install_riscv_toolchain():
    """Optionally install RISC-V cross-compilation toolchain."""
    try:
        subprocess.run(
            ["riscv64-linux-gnu-gcc", "--version"],
            capture_output=True, check=True,
        )
        log.info("RISC-V cross-compiler already installed.")
        return
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    log.info("Installing RISC-V cross-compilation toolchain...")
    log.info("  sudo apt install gcc-riscv64-linux-gnu g++-riscv64-linux-gnu")
    log.warn("Run manually if needed — skipping automatic install for safety.")


def clean_gem5():
    """Remove cloned gem5 directory."""
    if GEM5_DIR.exists():
        log.info(f"Removing {GEM5_DIR}...")
        shutil.rmtree(GEM5_DIR)
        log.success("gem5 directory removed.")
    if VERSION_FILE.exists():
        VERSION_FILE.unlink()
    log.success("Clean complete.")


def print_summary(isas: list, build_type: str):
    """Print setup summary."""
    print()
    print("=" * 70)
    print("  gem5 Setup Summary for GraphBrew")
    print("=" * 70)
    print(f"  gem5 location:   {GEM5_DIR}")
    print(f"  Version tag:     {VERSION_FILE.read_text().strip() if VERSION_FILE.exists() else 'unknown'}")
    print(f"  ISAs built:      {', '.join(isas)}")
    print(f"  Build type:      {build_type}")
    print()
    print("  Binaries:")
    for isa in isas:
        binary = GEM5_DIR / f"build/{isa}/gem5.{build_type}"
        status = "OK" if binary.exists() else "MISSING"
        print(f"    {isa}: {binary} [{status}]")
    print()
    print("  Overlay files installed:")
    for src_rel in OVERLAY_FILE_MAP:
        dst = GEM5_DIR / "src" / OVERLAY_FILE_MAP[src_rel]
        status = "OK" if dst.exists() else "pending"
        print(f"    {src_rel} [{status}]")
    print()
    print("  Next steps:")
    print("    1. Run benchmarks:")
    print(f"       {GEM5_DIR}/build/X86/gem5.{build_type} \\")
    print("           gem5_sim/configs/graphbrew/graph_se.py \\")
    print("           --binary bench/bin/pr --policy LRU")
    print()
    print("    2. Or use pipeline integration:")
    print("       python scripts/graphbrew_experiment.py --phase cache --simulator gem5")
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Setup gem5 for GraphBrew graph-aware cache simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--isa", nargs="+", default=["X86"],
        choices=VALID_ISAS,
        help="ISA targets to build (default: X86)",
    )
    parser.add_argument(
        "--build-type", default="opt",
        choices=VALID_BUILD_TYPES,
        help="gem5 build type (default: opt)",
    )
    parser.add_argument(
        "--tag", default=GEM5_DEFAULT_TAG,
        help=f"gem5 git tag to clone (default: {GEM5_DEFAULT_TAG})",
    )
    parser.add_argument(
        "--jobs", type=int, default=max(1, os.cpu_count() - 2),
        help="Parallel build jobs (default: nproc-2)",
    )
    parser.add_argument(
        "--skip-build", action="store_true",
        help="Clone and patch only, skip building",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force rebuild even if binaries exist",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Remove cloned gem5 directory and exit",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-clone even if gem5 directory exists",
    )

    args = parser.parse_args()

    # Clean mode
    if args.clean:
        clean_gem5()
        return

    total_steps = 6 if not args.skip_build else 4

    # Step 1: Prerequisites
    log.step(1, total_steps, "Checking prerequisites...")
    check_prerequisites()

    # Step 2: Clone
    log.step(2, total_steps, "Cloning gem5...")
    clone_gem5(args.tag, force=args.force)

    # Step 3: Apply overlays
    log.step(3, total_steps, "Applying overlay files...")
    apply_overlays()

    # Step 4: Apply patches
    log.step(4, total_steps, "Applying SConscript patches...")
    apply_patches()
    apply_current_vertex_pseudo_inst_patch()
    apply_riscv_ecg_extract_patch()
    apply_unified_diff_patches()

    if not args.skip_build:
        # Step 5: Build
        log.step(5, total_steps, f"Building gem5 ({', '.join(args.isa)})...")

        if args.rebuild:
            # Force rebuild by removing existing binaries
            for isa in args.isa:
                binary = GEM5_DIR / f"build/{isa}/gem5.{args.build_type}"
                if binary.exists():
                    binary.unlink()

        build_gem5(args.isa, args.build_type, args.jobs)

        # Step 6: Verify
        log.step(6, total_steps, "Verifying build...")
        verify_build(args.isa, args.build_type)

        # RISC-V toolchain hint
        if "RISCV" in args.isa:
            install_riscv_toolchain()

    # Summary
    print_summary(args.isa, args.build_type)
    log.success("gem5 setup complete!")


if __name__ == "__main__":
    main()
