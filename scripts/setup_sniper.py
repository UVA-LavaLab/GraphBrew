#!/usr/bin/env python3
"""Setup script for Sniper integration with GraphBrew.

This script mirrors the gem5 setup flow at a lighter-weight level:
- clone Sniper into bench/include/sniper_sim/snipersim,
- optionally checkout a pinned ref,
- leave GraphBrew overlay/config files in bench/include/sniper_sim/,
- optionally build Sniper once outside long experiment jobs.

Usage:
    python3 scripts/setup_sniper.py --dry-run
    python3 scripts/setup_sniper.py --skip-build
    python3 scripts/setup_sniper.py --ref main --jobs 8
    python3 scripts/setup_sniper.py --clean

The script is intentionally conservative: it does not copy overlays yet because
Sniper policy integration has not started. Overlay application will be added once
we identify the exact Sniper cache/prefetch extension points for the pinned ref.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SNIPER_SIM_DIR = PROJECT_ROOT / "bench" / "include" / "sniper_sim"
SNIPER_DIR = SNIPER_SIM_DIR / "snipersim"
SNIPER_OVERLAY_DIR = SNIPER_SIM_DIR / "overlays"
SNIPER_CONFIG_DIR = SNIPER_SIM_DIR / "configs"
VERSION_FILE = SNIPER_SIM_DIR / ".sniper_version"
OVERLAY_STATUS_FILE = SNIPER_SIM_DIR / ".sniper_overlays.json"
SNIPER_REPO_URL = "https://github.com/snipersim/snipersim.git"
SNIPER_DEFAULT_REF = "main"


class Logger:
    BLUE = "\033[0;34m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[0;33m"
    RED = "\033[0;31m"
    NC = "\033[0m"

    @staticmethod
    def info(message: str) -> None:
        print(f"{Logger.BLUE}[sniper-setup]{Logger.NC} {message}")

    @staticmethod
    def success(message: str) -> None:
        print(f"{Logger.GREEN}[sniper-setup]{Logger.NC} {message}")

    @staticmethod
    def warn(message: str) -> None:
        print(f"{Logger.YELLOW}[sniper-setup]{Logger.NC} {message}")

    @staticmethod
    def error(message: str) -> None:
        print(f"{Logger.RED}[sniper-setup]{Logger.NC} {message}", file=sys.stderr)


log = Logger()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def command_text(command: list[str]) -> str:
    return " ".join(command)


def run_cmd(command: list[str], cwd: Path | None = None, dry_run: bool = False) -> subprocess.CompletedProcess[str] | None:
    prefix = f"cd {cwd} && " if cwd else ""
    log.info(prefix + command_text(command))
    if dry_run:
        return None
    return subprocess.run(command, cwd=str(cwd) if cwd else None, text=True, check=True)


def git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(path),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def clone_or_update(args: argparse.Namespace) -> None:
    SNIPER_SIM_DIR.mkdir(parents=True, exist_ok=True)
    if SNIPER_DIR.exists():
        log.info(f"Sniper checkout already exists: {SNIPER_DIR}")
        if args.update:
            run_cmd(["git", "fetch", "--tags", "origin"], cwd=SNIPER_DIR, dry_run=args.dry_run)
    else:
        run_cmd(
            ["git", "clone", args.repo, str(SNIPER_DIR)],
            dry_run=args.dry_run,
        )

    if args.ref:
        run_cmd(["git", "checkout", args.ref], cwd=SNIPER_DIR, dry_run=args.dry_run)


def write_version(args: argparse.Namespace) -> None:
    if args.dry_run or not SNIPER_DIR.exists():
        return
    data = {
        "created_utc": utc_now(),
        "repo": args.repo,
        "requested_ref": args.ref,
        "head": git_head(SNIPER_DIR),
        "path": str(SNIPER_DIR),
    }
    VERSION_FILE.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    log.success(f"Wrote {VERSION_FILE}")


def build_sniper(args: argparse.Namespace) -> None:
    if args.skip_build:
        log.info("Skipping Sniper build (--skip-build).")
        return
    if not SNIPER_DIR.exists():
        raise SystemExit(f"Sniper checkout missing: {SNIPER_DIR}")
    command = ["make", f"-j{args.jobs}"]
    if args.build_target:
        command.append(args.build_target)
    if args.dry_run:
        log.info(f"Would build Sniper with: {command_text(command)}")
        return
    if not args.skip_deps_check:
        check_host_dependencies()
    run_cmd(command, cwd=SNIPER_DIR)


def replace_once(
    path: Path,
    old: str,
    new: str,
    dry_run: bool,
    accepted_markers: list[str] | None = None,
) -> None:
    if not path.exists():
        raise SystemExit(f"Sniper overlay patch target missing: {path}")
    text = path.read_text()
    if new in text:
        log.info(f"Overlay patch already present in {path.relative_to(SNIPER_DIR)}")
        return
    if accepted_markers and any(marker in text for marker in accepted_markers):
        log.info(f"Overlay patch already superseded in {path.relative_to(SNIPER_DIR)}")
        return
    if old not in text:
        raise SystemExit(
            f"Could not apply overlay patch to {path}; expected anchor not found. "
            "The Sniper checkout may have changed."
        )
    log.info(f"Patch {path.relative_to(SNIPER_DIR)}")
    if not dry_run:
        path.write_text(text.replace(old, new, 1))


def overlay_source_files() -> list[Path]:
    if not SNIPER_OVERLAY_DIR.exists():
        raise SystemExit(f"Sniper overlay directory missing: {SNIPER_OVERLAY_DIR}")
    return [
        source for source in sorted(SNIPER_OVERLAY_DIR.rglob("*"))
        if source.is_file() and source.suffix.lower() in {".h", ".hh", ".cc", ".cpp"}
    ]


def copy_overlay_sources(args: argparse.Namespace) -> list[str]:
    copied: list[str] = []
    for source in overlay_source_files():
        relative = source.relative_to(SNIPER_OVERLAY_DIR)
        target = SNIPER_DIR / relative
        log.info(f"Overlay copy {relative}")
        if not args.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        copied.append(str(relative))
    if not copied:
        log.warn(f"No overlay source files found under {SNIPER_OVERLAY_DIR}")
    return copied


def install_graphbrew_configs(args: argparse.Namespace) -> list[str]:
    if not SNIPER_CONFIG_DIR.exists():
        log.warn(f"No tracked Sniper config directory found: {SNIPER_CONFIG_DIR}")
        return []
    if not SNIPER_DIR.exists():
        if args.dry_run:
            log.info(f"Would install GraphBrew Sniper configs after cloning {SNIPER_DIR}")
            return []
        raise SystemExit(f"Sniper checkout missing: {SNIPER_DIR}")

    installed: list[str] = []
    for source in sorted(SNIPER_CONFIG_DIR.rglob("*.cfg")):
        relative = source.relative_to(SNIPER_CONFIG_DIR)
        target = SNIPER_DIR / "config" / relative
        log.info(f"Config copy {relative}")
        if not args.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        installed.append(str(relative))
    if not installed:
        log.warn(f"No tracked Sniper config files found under {SNIPER_CONFIG_DIR}")
    return installed


def write_overlay_status(copied_files: list[str]) -> None:
    data = {
        "created_utc": utc_now(),
        "sniper_head": git_head(SNIPER_DIR) if SNIPER_DIR.exists() else "",
        "policies": ["grasp", "popt", "ecg"],
        "prefetchers": ["droplet", "ecg_pfx"],
        "copied_files": copied_files,
        "patches": [
            "cache_base_replacement_policy_grasp",
            "cache_set_factory_grasp_popt_ecg",
            "cache_insert_prepare_insertion",
            "prefetcher_factory_droplet",
            "magic_user_graphbrew_hints",
        ],
    }
    OVERLAY_STATUS_FILE.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    log.success(f"Wrote {OVERLAY_STATUS_FILE}")


def patch_grasp_overlay(args: argparse.Namespace) -> None:
    cache_dir = SNIPER_DIR / "common" / "core" / "memory_subsystem" / "cache"
    cache_base = cache_dir / "cache_base.h"
    cache_set = cache_dir / "cache_set.cc"
    cache = cache_dir / "cache.cc"

    replace_once(
        cache_base,
        """         SRRIP,
         SRRIP_QBS,
         RANDOM,
""",
        """         SRRIP,
         SRRIP_QBS,
         GRASP,      // GraphBrew graph-aware SRRIP/GRASP replacement
         RANDOM,
""",
        args.dry_run,
        ["POPT,       // GraphBrew P-OPT oracle replacement"],
    )

    replace_once(
        cache_set,
        """#include "cache_set_srrip.h"
#include "cache_set_mplru.h"
""",
        """#include "cache_set_srrip.h"
#include "cache_set_grasp.h"
#include "cache_set_mplru.h"
""",
        args.dry_run,
        ['#include "cache_set_popt.h"'],
    )
    replace_once(
        cache_set,
        """      case CacheBase::SRRIP:
      case CacheBase::SRRIP_QBS:
         return new CacheSetSRRIP(cfgname, core_id, cache_type, associativity, blocksize, dynamic_cast<CacheSetInfoLRU*>(set_info), getNumQBSAttempts(policy, cfgname, core_id), is_tlb_set);

      case CacheBase::RANDOM:
""",
        """      case CacheBase::SRRIP:
      case CacheBase::SRRIP_QBS:
         return new CacheSetSRRIP(cfgname, core_id, cache_type, associativity, blocksize, dynamic_cast<CacheSetInfoLRU*>(set_info), getNumQBSAttempts(policy, cfgname, core_id), is_tlb_set);

      case CacheBase::GRASP:
         return new CacheSetGRASP(cfgname, core_id, cache_type, associativity, blocksize, dynamic_cast<CacheSetInfoLRU*>(set_info), getNumQBSAttempts(policy, cfgname, core_id), is_tlb_set);

      case CacheBase::RANDOM:
""",
        args.dry_run,
        ["case CacheBase::POPT:"],
    )
    replace_once(
        cache_set,
        """      case CacheBase::SRRIP:
      case CacheBase::SRRIP_QBS:
         return new CacheSetInfoLRU(name, cfgname, core_id, associativity, getNumQBSAttempts(policy, cfgname, core_id));
      case CacheBase::MPLRU:
""",
        """      case CacheBase::SRRIP:
      case CacheBase::SRRIP_QBS:
      case CacheBase::GRASP:
         return new CacheSetInfoLRU(name, cfgname, core_id, associativity, getNumQBSAttempts(policy, cfgname, core_id));
      case CacheBase::MPLRU:
""",
        args.dry_run,
        ["case CacheBase::POPT:"],
    )
    replace_once(
        cache_set,
        """   if (policy == "srrip_qbs")
      return CacheBase::SRRIP_QBS;
   if (policy == "random")
""",
        """   if (policy == "srrip_qbs")
      return CacheBase::SRRIP_QBS;
   if (policy == "grasp")
      return CacheBase::GRASP;
   if (policy == "random")
""",
        args.dry_run,
        ['if (policy == "popt")'],
    )

    replace_once(
        cache,
        """#include "simulator.h"
#include "cache.h"
#include "log.h"
""",
        """#include "simulator.h"
#include "cache.h"
#include "cache_set_grasp.h"
#include "log.h"
""",
        args.dry_run,
        ['#include "cache_set_popt.h"'],
    )
    replace_once(
        cache,
        """\tm_fake_sets[0]->insert(cache_block_info, fill_buff,
							   eviction, evict_block_info, evict_buff, cntlr);
""",
        """\t\tif (auto grasp_set = dynamic_cast<CacheSetGRASP*>(m_fake_sets[0]))
		{
			grasp_set->prepareInsertion(addr);
		}
		m_fake_sets[0]->insert(cache_block_info, fill_buff,
							   eviction, evict_block_info, evict_buff, cntlr);
""",
        args.dry_run,
        ["dynamic_cast<CacheSetPOPT*>(m_fake_sets[0])"],
    )
    replace_once(
        cache,
        """\tm_sets[set_index]->insert(cache_block_info, fill_buff,
								  eviction, evict_block_info, evict_buff, cntlr);
""",
        """\t\tif (auto grasp_set = dynamic_cast<CacheSetGRASP*>(m_sets[set_index]))
		{
			grasp_set->prepareInsertion(addr);
		}
		m_sets[set_index]->insert(cache_block_info, fill_buff,
								  eviction, evict_block_info, evict_buff, cntlr);
""",
        args.dry_run,
        ["dynamic_cast<CacheSetPOPT*>(m_sets[set_index])"],
    )
def patch_popt_overlay(args: argparse.Namespace) -> None:
    cache_dir = SNIPER_DIR / "common" / "core" / "memory_subsystem" / "cache"
    cache_base = cache_dir / "cache_base.h"
    cache_set = cache_dir / "cache_set.cc"
    cache = cache_dir / "cache.cc"

    replace_once(
        cache_base,
        """         GRASP,      // GraphBrew graph-aware SRRIP/GRASP replacement
         RANDOM,
""",
        """         GRASP,      // GraphBrew graph-aware SRRIP/GRASP replacement
         POPT,       // GraphBrew P-OPT oracle replacement
         RANDOM,
""",
        args.dry_run,
        ["ECG,        // GraphBrew ECG hybrid replacement"],
    )
    replace_once(
        cache_set,
        """#include "cache_set_grasp.h"
#include "cache_set_mplru.h"
""",
        """#include "cache_set_grasp.h"
#include "cache_set_popt.h"
#include "cache_set_mplru.h"
""",
        args.dry_run,
        ['#include "cache_set_ecg.h"'],
    )
    replace_once(
        cache_set,
        """      case CacheBase::GRASP:
         return new CacheSetGRASP(cfgname, core_id, cache_type, associativity, blocksize, dynamic_cast<CacheSetInfoLRU*>(set_info), getNumQBSAttempts(policy, cfgname, core_id), is_tlb_set);

      case CacheBase::RANDOM:
""",
        """      case CacheBase::GRASP:
         return new CacheSetGRASP(cfgname, core_id, cache_type, associativity, blocksize, dynamic_cast<CacheSetInfoLRU*>(set_info), getNumQBSAttempts(policy, cfgname, core_id), is_tlb_set);

      case CacheBase::POPT:
         return new CacheSetPOPT(cfgname, core_id, cache_type, associativity, blocksize, dynamic_cast<CacheSetInfoLRU*>(set_info), getNumQBSAttempts(policy, cfgname, core_id), is_tlb_set);

      case CacheBase::RANDOM:
""",
        args.dry_run,
        ["case CacheBase::ECG:"],
    )
    replace_once(
        cache_set,
        """      case CacheBase::SRRIP_QBS:
      case CacheBase::GRASP:
         return new CacheSetInfoLRU(name, cfgname, core_id, associativity, getNumQBSAttempts(policy, cfgname, core_id));
      case CacheBase::MPLRU:
""",
        """      case CacheBase::SRRIP_QBS:
      case CacheBase::GRASP:
      case CacheBase::POPT:
         return new CacheSetInfoLRU(name, cfgname, core_id, associativity, getNumQBSAttempts(policy, cfgname, core_id));
      case CacheBase::MPLRU:
""",
        args.dry_run,
        ["case CacheBase::ECG:"],
    )
    replace_once(
        cache_set,
        """   if (policy == "grasp")
      return CacheBase::GRASP;
   if (policy == "random")
""",
        """   if (policy == "grasp")
      return CacheBase::GRASP;
   if (policy == "popt")
      return CacheBase::POPT;
   if (policy == "random")
""",
        args.dry_run,
        ['if (policy == "ecg")'],
    )

    replace_once(
        cache,
        """#include "cache_set_grasp.h"
#include "log.h"
""",
        """#include "cache_set_grasp.h"
#include "cache_set_popt.h"
#include "log.h"
""",
        args.dry_run,
        ['#include "cache_set_ecg.h"'],
    )
    replace_once(
        cache,
        """\t\tif (auto grasp_set = dynamic_cast<CacheSetGRASP*>(m_fake_sets[0]))
		{
			grasp_set->prepareInsertion(addr);
		}
		m_fake_sets[0]->insert(cache_block_info, fill_buff,
""",
        """\t\tif (auto grasp_set = dynamic_cast<CacheSetGRASP*>(m_fake_sets[0]))
		{
			grasp_set->prepareInsertion(addr);
		}
		if (auto popt_set = dynamic_cast<CacheSetPOPT*>(m_fake_sets[0]))
		{
			popt_set->prepareInsertion(addr);
		}
		m_fake_sets[0]->insert(cache_block_info, fill_buff,
""",
        args.dry_run,
        ["dynamic_cast<CacheSetECG*>(m_fake_sets[0])"],
    )
    replace_once(
        cache,
        """\t\tif (auto grasp_set = dynamic_cast<CacheSetGRASP*>(m_sets[set_index]))
		{
			grasp_set->prepareInsertion(addr);
		}
		m_sets[set_index]->insert(cache_block_info, fill_buff,
""",
        """\t\tif (auto grasp_set = dynamic_cast<CacheSetGRASP*>(m_sets[set_index]))
		{
			grasp_set->prepareInsertion(addr);
		}
		if (auto popt_set = dynamic_cast<CacheSetPOPT*>(m_sets[set_index]))
		{
			popt_set->prepareInsertion(addr);
		}
		m_sets[set_index]->insert(cache_block_info, fill_buff,
""",
        args.dry_run,
        ["dynamic_cast<CacheSetECG*>(m_sets[set_index])"],
    )
def patch_ecg_overlay(args: argparse.Namespace) -> None:
    cache_dir = SNIPER_DIR / "common" / "core" / "memory_subsystem" / "cache"
    nuca_dir = (
        SNIPER_DIR / "common" / "core" / "memory_subsystem" /
        "parametric_dram_directory_msi"
    )
    directory_dir = (
        SNIPER_DIR / "common" / "core" / "memory_subsystem" /
        "pr_l1_pr_l2_dram_directory_msi"
    )
    cache_base = cache_dir / "cache_base.h"
    cache_set = cache_dir / "cache_set.cc"
    cache = cache_dir / "cache.cc"
    nuca_header = nuca_dir / "nuca_cache.h"
    nuca_source = nuca_dir / "nuca_cache.cc"
    directory_source = directory_dir / "dram_directory_cntlr.cc"

    replace_once(
        cache_base,
        """         POPT,       // GraphBrew P-OPT oracle replacement
         RANDOM,
""",
        """         POPT,       // GraphBrew P-OPT oracle replacement
         ECG,        // GraphBrew ECG hybrid replacement
         RANDOM,
""",
        args.dry_run,
        ['return new DropletPrefetcher(configName, core_id);'],
    )
    replace_once(
        cache_set,
        """#include "cache_set_popt.h"
#include "cache_set_mplru.h"
""",
        """#include "cache_set_popt.h"
#include "cache_set_ecg.h"
#include "cache_set_mplru.h"
""",
        args.dry_run,
    ['return new DropletPrefetcher(configName, core_id);'],
    )
    replace_once(
        cache_set,
        """      case CacheBase::POPT:
         return new CacheSetPOPT(cfgname, core_id, cache_type, associativity, blocksize, dynamic_cast<CacheSetInfoLRU*>(set_info), getNumQBSAttempts(policy, cfgname, core_id), is_tlb_set);

      case CacheBase::RANDOM:
""",
        """      case CacheBase::POPT:
         return new CacheSetPOPT(cfgname, core_id, cache_type, associativity, blocksize, dynamic_cast<CacheSetInfoLRU*>(set_info), getNumQBSAttempts(policy, cfgname, core_id), is_tlb_set);

      case CacheBase::ECG:
         return new CacheSetECG(cfgname, core_id, cache_type, associativity, blocksize, dynamic_cast<CacheSetInfoLRU*>(set_info), getNumQBSAttempts(policy, cfgname, core_id), is_tlb_set);

      case CacheBase::RANDOM:
""",
        args.dry_run,
    ["EcgPfxPrefetcher"],
    )
    replace_once(
        cache_set,
        """      case CacheBase::GRASP:
      case CacheBase::POPT:
         return new CacheSetInfoLRU(name, cfgname, core_id, associativity, getNumQBSAttempts(policy, cfgname, core_id));
      case CacheBase::MPLRU:
""",
        """      case CacheBase::GRASP:
      case CacheBase::POPT:
      case CacheBase::ECG:
         return new CacheSetInfoLRU(name, cfgname, core_id, associativity, getNumQBSAttempts(policy, cfgname, core_id));
      case CacheBase::MPLRU:
""",
        args.dry_run,
    )
    replace_once(
        cache_set,
        """   if (policy == "popt")
      return CacheBase::POPT;
   if (policy == "random")
""",
        """   if (policy == "popt")
      return CacheBase::POPT;
   if (policy == "ecg")
      return CacheBase::ECG;
   if (policy == "random")
""",
        args.dry_run,
    )

    replace_once(
        cache,
        """#include "cache_set_popt.h"
#include "log.h"
""",
        """#include "cache_set_popt.h"
#include "cache_set_ecg.h"
#include "log.h"
""",
        args.dry_run,
    )
    replace_once(
        nuca_header,
        """      boost::tuple<SubsecondTime, HitWhere::where_t> read(IntPtr address, Byte* data_buf, SubsecondTime now, ShmemPerf *perf, bool count, bool is_metadata = false);
""",
        """      boost::tuple<SubsecondTime, HitWhere::where_t> read(IntPtr address, core_id_t requester, Byte* data_buf, SubsecondTime now, ShmemPerf *perf, bool count, bool is_metadata = false);
""",
        args.dry_run,
    )
    replace_once(
        nuca_header,
        """      boost::tuple<SubsecondTime, HitWhere::where_t> write(IntPtr address, Byte* data_buf, bool& eviction, IntPtr& evict_address, Byte* evict_buf, SubsecondTime now, bool count, bool is_metadata = false);
""",
        """      boost::tuple<SubsecondTime, HitWhere::where_t> write(IntPtr address, core_id_t requester, Byte* data_buf, bool& eviction, IntPtr& evict_address, Byte* evict_buf, SubsecondTime now, bool count, bool is_metadata = false);
""",
        args.dry_run,
    )
    replace_once(
        nuca_source,
        """#include "shmem_perf.h"
""",
        """#include "shmem_perf.h"
#include "core/memory_subsystem/cache/graph_cache_context_sniper.h"
""",
        args.dry_run,
    )
    replace_once(
        nuca_source,
        """NucaCache::read(IntPtr address, Byte* data_buf, SubsecondTime now, ShmemPerf *perf, bool count, bool is_metadata)
{
   HitWhere::where_t hit_where = HitWhere::MISS;
""",
        """NucaCache::read(IntPtr address, core_id_t requester, Byte* data_buf, SubsecondTime now, ShmemPerf *perf, bool count, bool is_metadata)
{
   graphbrew::sniper::setCurrentNucaRequesterCore(
      static_cast<uint32_t>(requester));
   HitWhere::where_t hit_where = HitWhere::MISS;
""",
        args.dry_run,
        ["NucaCache::read(IntPtr address, core_id_t requester"],
    )
    replace_once(
        nuca_source,
        """NucaCache::write(IntPtr address, Byte* data_buf, bool& eviction, IntPtr& evict_address, Byte* evict_buf, SubsecondTime now, bool count, bool is_metadata)
{
   HitWhere::where_t hit_where = HitWhere::MISS;
""",
        """NucaCache::write(IntPtr address, core_id_t requester, Byte* data_buf, bool& eviction, IntPtr& evict_address, Byte* evict_buf, SubsecondTime now, bool count, bool is_metadata)
{
   graphbrew::sniper::setCurrentNucaRequesterCore(
      static_cast<uint32_t>(requester));
   HitWhere::where_t hit_where = HitWhere::MISS;
""",
        args.dry_run,
["NucaCache::write(IntPtr address, core_id_t requester"],
    )
    replace_once(
        directory_source,
        """boost::tie(nuca_latency, hit_where) = m_nuca_cache->read(address, nuca_data_buf, getShmemPerfModel()->getElapsedTime(ShmemPerfModel::_SIM_THREAD), orig_shmem_msg->getPerf(), true,orig_shmem_msg->getBlockType());
""",
        """boost::tie(nuca_latency, hit_where) = m_nuca_cache->read(address, orig_shmem_msg->getRequester(), nuca_data_buf, getShmemPerfModel()->getElapsedTime(ShmemPerfModel::_SIM_THREAD), orig_shmem_msg->getPerf(), true,orig_shmem_msg->getBlockType());
""",
        args.dry_run,
        ["m_nuca_cache->read(address, orig_shmem_msg->getRequester()"],
    )
    replace_once(
        directory_source,
        """      m_nuca_cache->write(
         address, data_buf,
""",
        """      m_nuca_cache->write(
         address, requester, data_buf,
""",
        args.dry_run,
        ["m_nuca_cache->write(\n         address, requester, data_buf,"],
    )
    replace_once(
        cache,
        """\t\tif (auto popt_set = dynamic_cast<CacheSetPOPT*>(m_fake_sets[0]))
		{
			popt_set->prepareInsertion(addr);
		}
		m_fake_sets[0]->insert(cache_block_info, fill_buff,
""",
        """\t\tif (auto popt_set = dynamic_cast<CacheSetPOPT*>(m_fake_sets[0]))
		{
			popt_set->prepareInsertion(addr);
		}
		if (auto ecg_set = dynamic_cast<CacheSetECG*>(m_fake_sets[0]))
		{
			ecg_set->prepareInsertion(addr);
		}
		m_fake_sets[0]->insert(cache_block_info, fill_buff,
""",
        args.dry_run,
    )
    replace_once(
        cache,
        """\t\tif (auto popt_set = dynamic_cast<CacheSetPOPT*>(m_sets[set_index]))
		{
			popt_set->prepareInsertion(addr);
		}
		m_sets[set_index]->insert(cache_block_info, fill_buff,
""",
        """\t\tif (auto popt_set = dynamic_cast<CacheSetPOPT*>(m_sets[set_index]))
		{
			popt_set->prepareInsertion(addr);
		}
		if (auto ecg_set = dynamic_cast<CacheSetECG*>(m_sets[set_index]))
		{
			ecg_set->prepareInsertion(addr);
		}
		m_sets[set_index]->insert(cache_block_info, fill_buff,
""",
        args.dry_run,
    )


def patch_droplet_overlay(args: argparse.Namespace) -> None:
    prefetcher = SNIPER_DIR / "common" / "core" / "memory_subsystem" / "parametric_dram_directory_msi" / "prefetcher.cc"
    replace_once(
        prefetcher,
        """#include "a53prefetcher.h"
""",
        """#include "a53prefetcher.h"
#include "droplet_prefetcher.h"
""",
        args.dry_run,
    )
    replace_once(
        prefetcher,
        """   else if (type == "a53prefetcher")
       return new A53Prefetcher(configName, core_id);

   LOG_PRINT_ERROR("Invalid prefetcher type %s", type.c_str());
""",
        """   else if (type == "a53prefetcher")
       return new A53Prefetcher(configName, core_id);
   else if (type == "droplet")
       return new DropletPrefetcher(configName, core_id);

   LOG_PRINT_ERROR("Invalid prefetcher type %s", type.c_str());
""",
        args.dry_run,
        ['return new DropletPrefetcher(configName, core_id);'],
    )


def patch_graphbrew_simuser_overlay(args: argparse.Namespace) -> None:
    magic_server = SNIPER_DIR / "common" / "system" / "magic_server.cc"
    sim_api = SNIPER_DIR / "include" / "sim_api.h"
    sim_api_text = sim_api.read_text()
    old_constraint = ': "=a"(_res) /* output    */'
    new_constraint = ': "=&a"(_res) /* early-clobber: inputs cannot alias RAX */'
    if sim_api_text.count(new_constraint) >= 3:
        log.info("Overlay patch already present in include/sim_api.h")
    elif sim_api_text.count(old_constraint) >= 3:
        log.info("Patch include/sim_api.h (SimMagic0/1/2 early-clobber)")
        if not args.dry_run:
            sim_api.write_text(
                sim_api_text.replace(old_constraint, new_constraint, 3)
            )
    else:
        raise SystemExit(
            "Could not patch include/sim_api.h SimMagic1/2 constraints; "
            "expected three '=a' outputs or three GraphBrew early-clobber outputs."
        )
    magic_text = magic_server.read_text()
    old_decode = (
        "uint32_t fl_vertex = static_cast<uint32_t>(arg1 & 0xFFFFFFFFFFFFULL);\n"
        "             uint16_t fl_epoch = static_cast<uint16_t>((arg1 >> 48) & 0xFFFFULL);"
    )
    new_decode = (
        "uint32_t fl_vertex = static_cast<uint32_t>(arg1 & 0xFFFFFFFFULL);\n"
        "             uint16_t fl_epoch = static_cast<uint16_t>((arg1 >> 32) & 0xFFFFULL);"
    )
    if old_decode in magic_text:
        log.info("Upgrade common/system/magic_server.cc ECG extract payload decode")
        magic_text = magic_text.replace(old_decode, new_decode, 1)
        if not args.dry_run:
            magic_server.write_text(magic_text)
    replace_once(
        magic_server,
        """#include "magic_server.h"
#include "sim_api.h"
""",
        """#include "magic_server.h"
#include "sim_api.h"
#include "core/memory_subsystem/cache/graph_cache_context_sniper.h"
""",
        args.dry_run,
    )
    if new_decode in magic_text:
       log.info("Overlay patch already present in common/system/magic_server.cc")
    else:
       replace_once(
           magic_server,
           """      case SIM_CMD_USER:
      {
         MagicMarkerType args = { thread_id: thread_id, core_id: core_id, arg0: arg0, arg1: arg1, str: NULL };
         return Sim()->getHooksManager()->callHooks(HookType::HOOK_MAGIC_USER, (UInt64)&args, true /* expect return value */);
      }
""",
           """      case SIM_CMD_USER:
      {
         if (arg0 == graphbrew::sniper::GRAPHBREW_SET_VERTEX_WORK_ID)
         {
            graphbrew::sniper::setCurrentVertexHint(static_cast<uint32_t>(core_id), arg1);
            return 0;
         }
         if (arg0 == graphbrew::sniper::GRAPHBREW_ECG_PFX_TARGET_WORK_ID)
         {
            graphbrew::sniper::setPrefetchTargetHint(static_cast<uint32_t>(core_id), arg1);
            return 0;
         }
         if (arg0 == graphbrew::sniper::GRAPHBREW_ECG_EXTRACT_WORK_ID)
         {
            uint32_t fl_vertex = static_cast<uint32_t>(arg1 & 0xFFFFFFFFULL);
            uint16_t fl_epoch = static_cast<uint16_t>((arg1 >> 32) & 0xFFFFULL);
            graphbrew::sniper::recordEcgEpoch(static_cast<uint32_t>(core_id), fl_vertex, fl_epoch);
            return 0;
         }
         MagicMarkerType args = { thread_id: thread_id, core_id: core_id, arg0: arg0, arg1: arg1, str: NULL };
         return Sim()->getHooksManager()->callHooks(HookType::HOOK_MAGIC_USER, (UInt64)&args, true /* expect return value */);
      }
""",
           args.dry_run,
)


def patch_ecg_pfx_prefetcher_overlay(args: argparse.Namespace) -> None:
    prefetcher = SNIPER_DIR / "common" / "core" / "memory_subsystem" / "parametric_dram_directory_msi" / "prefetcher.cc"
    replace_once(
        prefetcher,
        """#include "droplet_prefetcher.h"
""",
        """#include "droplet_prefetcher.h"
#include "ecg_pfx_prefetcher.h"
""",
        args.dry_run,
    )
    replace_once(
        prefetcher,
        """   else if (type == "droplet")
       return new DropletPrefetcher(configName, core_id);

   LOG_PRINT_ERROR("Invalid prefetcher type %s", type.c_str());
""",
        """   else if (type == "droplet")
       return new DropletPrefetcher(configName, core_id);
   else if (type == "ecg_pfx")
       return new EcgPfxPrefetcher(configName, core_id);

   LOG_PRINT_ERROR("Invalid prefetcher type %s", type.c_str());
""",
        args.dry_run,
        ["EcgPfxPrefetcher"],
    )


def apply_overlays(args: argparse.Namespace) -> None:
    if not args.apply_overlays:
        return
    if not SNIPER_DIR.exists():
        raise SystemExit(f"Sniper checkout missing: {SNIPER_DIR}")
    log.info("Applying GraphBrew Sniper overlays")
    copied_files = copy_overlay_sources(args)
    patch_grasp_overlay(args)
    patch_popt_overlay(args)
    patch_ecg_overlay(args)
    patch_droplet_overlay(args)
    patch_graphbrew_simuser_overlay(args)
    patch_ecg_pfx_prefetcher_overlay(args)
    if args.dry_run:
        log.info("Overlay application dry-run completed.")
    else:
        write_overlay_status(copied_files)
        log.success("Applied GraphBrew Sniper overlays.")


def compiler_for_checks() -> str:
    return os.environ.get("CC") or shutil.which("gcc") or shutil.which("cc") or "cc"


def header_available(header: str) -> bool:
    compiler = compiler_for_checks()
    result = subprocess.run(
        [compiler, "-x", "c", "-E", "-"],
        input=f"#include <{header}>\n",
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def check_host_dependencies() -> None:
    missing_headers = [header for header in ("sqlite3.h",) if not header_available(header)]
    if not missing_headers:
        return
    raise SystemExit(
        "Missing Sniper build dependency headers: "
        + ", ".join(missing_headers)
        + "\nInstall the matching OS packages, e.g. Ubuntu/Debian: "
        + "sudo apt-get install libsqlite3-dev; RHEL/CentOS/Fedora: "
        + "sudo dnf install sqlite-devel. On UVA Slurm, load/use a toolchain "
        + "environment that provides sqlite development headers."
    )


def smoke_test(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    run_sniper = SNIPER_DIR / "run-sniper"
    if args.dry_run:
        log.info("Would run Sniper smoke test.")
        return
    if not run_sniper.exists():
        raise SystemExit(f"run-sniper not found: {run_sniper}")
    out_dir = Path(args.smoke_dir)
    command = [
        str(run_sniper),
        "-n", "1",
        "--fast-forward",
        "-d", str(out_dir),
        "-cgraphbrew/graph_sniper",
        "--", "/bin/true",
    ]
    run_cmd(command)


def graphbrew_smoke_test(args: argparse.Namespace) -> None:
    if not args.graphbrew_smoke:
        return
    run_sniper = SNIPER_DIR / "run-sniper"
    if args.dry_run:
        log.info("Would build and run GraphBrew Sniper smoke binaries.")
        return
    if not run_sniper.exists():
        raise SystemExit(f"run-sniper not found: {run_sniper}")
    run_cmd(["make", "sniper-hello_roi", "sniper-pr_kernel_smoke", "sniper-bfs_kernel_smoke", "sniper-sssp_kernel_smoke"], cwd=PROJECT_ROOT)
    out_dir = Path(args.graphbrew_smoke_dir)
    command = [
        str(run_sniper),
        "--roi",
        "--no-cache-warming",
        "-n", "1",
        "-d", str(out_dir),
        "-cgraphbrew/graph_sniper",
        "--", str(PROJECT_ROOT / "bench" / "bin_sniper" / "pr_kernel_smoke"),
    ]
    run_cmd(command, cwd=PROJECT_ROOT)
    parser = PROJECT_ROOT / "bench" / "include" / "sniper_sim" / "scripts" / "parse_stats.py"
    if parser.exists():
        run_cmd([sys.executable, str(parser), str(out_dir)], cwd=PROJECT_ROOT)


def clean(args: argparse.Namespace) -> int:
    if not SNIPER_DIR.exists():
        log.info(f"No Sniper checkout to remove: {SNIPER_DIR}")
        return 0
    if args.dry_run:
        log.info(f"Would remove {SNIPER_DIR}")
        return 0
    shutil.rmtree(SNIPER_DIR)
    if VERSION_FILE.exists():
        VERSION_FILE.unlink()
    log.success(f"Removed {SNIPER_DIR}")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Setup Sniper for GraphBrew.")
    parser.add_argument("--repo", default=SNIPER_REPO_URL, help="Sniper git repository URL.")
    parser.add_argument("--ref", default=SNIPER_DEFAULT_REF, help="Sniper branch/tag/commit to checkout.")
    parser.add_argument("--jobs", type=int, default=8, help="Parallel build jobs.")
    parser.add_argument("--build-target", default="", help="Optional Sniper make target, e.g. configscripts or standalone.")
    parser.add_argument("--skip-build", action="store_true", help="Clone/checkout only; do not build.")
    parser.add_argument("--skip-deps-check", action="store_true", help="Skip GraphBrew host dependency preflight before building Sniper.")
    parser.add_argument("--update", action="store_true", help="Fetch updates if checkout already exists.")
    parser.add_argument("--apply-overlays", action="store_true", help="Copy tracked GraphBrew overlay files into the Sniper checkout and apply wiring patches.")
    parser.add_argument("--smoke", action="store_true", help="Run a minimal /bin/true Sniper smoke test after build.")
    parser.add_argument("--smoke-dir", default="/tmp/sniper-graphbrew-smoke", help="Smoke-test output directory.")
    parser.add_argument("--graphbrew-smoke", action="store_true", help="Run GraphBrew hello/pr_kernel Sniper smoke tests after build.")
    parser.add_argument("--graphbrew-smoke-dir", default="/tmp/sniper-graphbrew-pr-kernel", help="GraphBrew Sniper smoke output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing them.")
    parser.add_argument("--clean", action="store_true", help="Remove the Sniper checkout and version file.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.clean:
        return clean(args)
    clone_or_update(args)
    write_version(args)
    install_graphbrew_configs(args)
    apply_overlays(args)
    build_sniper(args)
    smoke_test(args)
    graphbrew_smoke_test(args)
    log.success("Sniper setup step completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
