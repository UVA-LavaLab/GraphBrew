#!/usr/bin/env python3
"""Build a reproducible Pin-4 port of the public P-OPT cache simulators.

The cache/replacement source remains copied from the pinned artifacts. Two
compatibility adaptations are applied to the build copy:

1. restore the C++ names that old Pin exported globally;
2. terminate after the application's explicit PIN_DumpStats hook, avoiding
   Pin-4 C++ teardown of application-owned graph pointers.

An optional GRASP arm copies P-OPT's DRRIP hierarchy and substitutes the exact
3-bit insertion/promotion rules from faldupriyank/grasp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


POPT_COMMIT = "53b5021846690d0f3445428c6380e877ecf7a10e"
GRASP_COMMIT = "6e3814430265fc4f2513c95ef131a6522bc9d389"
PUBLIC_POLICIES = ("lru", "drrip", "popt-8b", "opt-ideal")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path,
        capture_output=True, text=True, check=True).stdout.strip()


def copy_sources(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.is_file() and item.suffix in {".cpp", ".h"}:
            shutil.copy2(item, target / item.name)
    pinsim = target / "cache_pinsim.cpp"
    text = pinsim.read_text()
    needle = """void printStats()
{
    cache.reportTotalStats();
}"""
    replacement = """void printStats()
{
    cache.reportTotalStats();
    std::cout.flush();
    std::cerr.flush();
    PIN_ExitProcess(0);
}"""
    if needle not in text:
        raise SystemExit(f"printStats hook not found in {pinsim}")
    pinsim.write_text(text.replace(needle, replacement, 1))


def make_grasp(source: Path, target: Path) -> None:
    copy_sources(source, target)
    header = target / "llc.h"
    text = header.read_text().replace(
        "#include <cstdlib>", "#include <algorithm>\n#include <cstdlib>", 1)
    text = text.replace(
        """        void updateReplacementState(int setID, int wayID);
        void moveToMRU(int setID, int wayID);""",
        """        void updateReplacementState(int setID, int wayID);
        void setInsertionState(intptr_t addr, int setID, int wayID);
        bool isHighReuse(intptr_t addr);
        bool isModerateReuse(intptr_t addr);
        void moveToMRU(int setID, int wayID);""",
        1)
    header.write_text(text)

    cpp = target / "llc.cpp"
    text = cpp.read_text().replace(
        """    m_tagArray[setID][index] = addr; //new line inserted 
    //m_dirty[setID][index]    = (isWrite == true) ? 1 : 0;""",
        """    m_tagArray[setID][index] = addr; //new line inserted
    setInsertionState(addr, setID, index);
    //m_dirty[setID][index]    = (isWrite == true) ? 1 : 0;""",
        1)
    begin = text.index("int LLC::getReplacementIndex(")
    end = text.index("void LLC::moveToMRU(", begin)
    policy = r"""bool LLC::isHighReuse(intptr_t addr)
{
    const uint64_t capacity =
        static_cast<uint64_t>(m_numSets) * m_numWays * m_lineSz;
    const uint64_t high_bytes = capacity / 2;
    for (int dTypeID : {IRREGDATA, REGDATA}) {
        for (size_t i = 0; i < m_dType_addrStart[dTypeID].size(); ++i) {
            const intptr_t start = m_dType_addrStart[dTypeID][i];
            const intptr_t end = m_dType_addrEnd[dTypeID][i];
            const intptr_t high =
                std::min<intptr_t>(end, start + high_bytes) + 8;
            if (addr >= start && addr < high) return true;
        }
    }
    return false;
}

bool LLC::isModerateReuse(intptr_t addr)
{
    const uint64_t capacity =
        static_cast<uint64_t>(m_numSets) * m_numWays * m_lineSz;
    const uint64_t high_bytes = capacity / 2;
    for (int dTypeID : {IRREGDATA, REGDATA}) {
        for (size_t i = 0; i < m_dType_addrStart[dTypeID].size(); ++i) {
            const intptr_t start = m_dType_addrStart[dTypeID][i];
            const intptr_t end = m_dType_addrEnd[dTypeID][i];
            const intptr_t high =
                std::min<intptr_t>(end, start + high_bytes) + 8;
            const intptr_t moderate =
                std::min<intptr_t>(end, start + 2 * high_bytes) + 8;
            if (addr >= high && addr < moderate) return true;
        }
    }
    return false;
}

int LLC::getReplacementIndex(int setID, int setType, int tid)
{
    (void)setType; (void)tid;
    for (int way = 0; way < m_numWays; ++way)
        if (m_tagArray[setID][way] == -1) return way;
    int victim = 0;
    int max_rrpv = m_rrpv[setID][0];
    for (int way = 1; way < m_numWays; ++way) {
        if (m_rrpv[setID][way] > max_rrpv) {
            max_rrpv = m_rrpv[setID][way];
            victim = way;
        }
    }
    if (max_rrpv < m_MAX_rrpv) {
        const int diff = m_MAX_rrpv - max_rrpv;
        for (int way = 0; way < m_numWays; ++way)
            m_rrpv[setID][way] += diff;
    }
    return victim;
}

void LLC::setInsertionState(intptr_t addr, int setID, int wayID)
{
    if (isHighReuse(addr)) m_rrpv[setID][wayID] = 1;
    else if (isModerateReuse(addr))
        m_rrpv[setID][wayID] = m_MAX_rrpv - 1;
    else m_rrpv[setID][wayID] = m_MAX_rrpv;
}

void LLC::updateReplacementState(int setID, int wayID)
{
    if (isHighReuse(m_tagArray[setID][wayID]))
        m_rrpv[setID][wayID] = 0;
    else if (m_rrpv[setID][wayID] > 0)
        --m_rrpv[setID][wayID];
}

"""
    cpp.write_text(text[:begin] + policy + text[end:])


def write_build_files(root: Path, pin_root: Path) -> None:
    (root / "compat.hpp").write_text(
        '#include "pin.H"\n#include <iostream>\n#include <string>\n'
        "using std::cerr;\nusing std::endl;\nusing std::string;\n")
    (root / "Makefile").write_text(f"""PIN_ROOT ?= {pin_root}
SRC_DIR ?= ../../simulators/lru
CONFIG_ROOT := $(PIN_ROOT)/source/tools/Config
include $(CONFIG_ROOT)/makefile.config
TOOL_ROOTS := cache_pinsim
OBJECT_ROOTS := cache_backend l1 l2 llc
VPATH := $(SRC_DIR)
TOOL_CXXFLAGS += -std=c++11 -DBIGARRAY_MULTIPLIER=1 -I$(SRC_DIR) \\
\t-include $(CURDIR)/compat.hpp
include $(TOOLS_ROOT)/Config/makefile.default.rules
$(OBJDIR)cache_pinsim$(PINTOOL_SUFFIX): \\
\t$(OBJDIR)cache_pinsim$(OBJ_SUFFIX) $(OBJECTS)
\t$(LINKER) $(TOOL_LDFLAGS) $(LINK_EXE)$@ $^ \\
\t\t$(TOOL_LPATHS) $(TOOL_LIBS)
""")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--pin-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--grasp-source-root", type=Path)
    args = parser.parse_args(argv)
    artifact = args.artifact_root.resolve()
    pin_root = args.pin_root.resolve()
    out = args.out_dir.resolve()
    if git_head(artifact) != POPT_COMMIT:
        raise SystemExit("unexpected P-OPT artifact commit")
    grasp = args.grasp_source_root.resolve() if args.grasp_source_root else None
    if grasp and git_head(grasp) != GRASP_COMMIT:
        raise SystemExit("unexpected GRASP artifact commit")
    out.mkdir(parents=True, exist_ok=True)
    write_build_files(out, pin_root)

    policies = list(PUBLIC_POLICIES)
    for policy in policies:
        copy_sources(
            artifact / "simulators" / policy, out / "src" / policy)
    if grasp:
        make_grasp(
            artifact / "simulators" / "drrip", out / "src" / "grasp")
        policies.append("grasp")

    binaries = {}
    for policy in policies:
        subprocess.run(
            ["make", f"PIN_ROOT={pin_root}", "clean"],
            cwd=out, stdout=subprocess.DEVNULL, check=False)
        subprocess.run(
            ["make", f"PIN_ROOT={pin_root}",
             f"SRC_DIR={out / 'src' / policy}", "-j4"],
            cwd=out, check=True)
        target = out / "bin" / policy / "cache_pinsim.so"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out / "obj-intel64/cache_pinsim.so", target)
        binaries[policy] = sha256(target)

    manifest = {
        "popt_commit": POPT_COMMIT,
        "grasp_commit": GRASP_COMMIT if grasp else "",
        "pin_version": subprocess.run(
            [str(pin_root / "pin"), "-version"],
            capture_output=True, text=True, check=True).stdout.splitlines()[0],
        "policies": policies,
        "binaries": binaries,
        "compatibility_changes": [
            "restore old Pin global C++ names",
            "PIN_ExitProcess(0) after explicit PIN_DumpStats",
            "optional GRASP arm copies official 3-bit insertion/hit rules",
        ],
    }
    (out / "port_build_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
