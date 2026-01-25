#!/usr/bin/env python3
"""
Build utilities for GraphBrew.

Handles compilation of benchmark binaries from source.
Can be used standalone or as a library.

Standalone usage:
    python -m scripts.lib.build --check          # Check if binaries exist
    python -m scripts.lib.build --build          # Build all binaries
    python -m scripts.lib.build --build --sim    # Build simulation binaries
    python -m scripts.lib.build --clean          # Clean build artifacts

Library usage:
    from scripts.lib.build import build_binaries, check_binaries
    
    if not check_binaries():
        build_binaries()
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from .utils import (
    PROJECT_ROOT, BENCH_DIR, BIN_DIR, BIN_SIM_DIR,
    BENCHMARKS, Logger, get_timestamp,
)

# Import dependency checker (optional - graceful fallback if not available)
try:
    from .dependencies import check_dependencies, install_dependencies, print_install_instructions
    HAS_DEPENDENCY_CHECKER = True
except ImportError:
    HAS_DEPENDENCY_CHECKER = False

# Initialize logger
log = Logger()

# =============================================================================
# Constants
# =============================================================================

# Benchmark binary names (match BENCHMARKS from utils)
BENCHMARK_BINARIES = ["pr", "bfs", "cc", "sssp", "bc", "tc"]

# Makefile targets
MAKE_TARGET_ALL = "all"
MAKE_TARGET_SIM = "sim"
MAKE_TARGET_CLEAN = "clean"


# =============================================================================
# Build Functions
# =============================================================================

def get_cpu_count() -> int:
    """Get number of CPUs for parallel builds."""
    try:
        return os.cpu_count() or 4
    except:
        return 4


def check_binary_exists(binary_name: str, sim: bool = False) -> bool:
    """
    Check if a specific benchmark binary exists.
    
    Args:
        binary_name: Name of the binary (e.g., 'pr', 'bfs')
        sim: If True, check in bin_sim/ instead of bin/
        
    Returns:
        True if binary exists and is executable
    """
    bin_dir = BIN_SIM_DIR if sim else BIN_DIR
    binary_path = bin_dir / binary_name
    return binary_path.exists() and os.access(binary_path, os.X_OK)


def check_binaries(sim: bool = False) -> Tuple[bool, List[str], List[str]]:
    """
    Check which benchmark binaries exist.
    
    Args:
        sim: If True, check simulation binaries
        
    Returns:
        Tuple of (all_exist, found_list, missing_list)
    """
    found = []
    missing = []
    
    for binary in BENCHMARK_BINARIES:
        if check_binary_exists(binary, sim=sim):
            found.append(binary)
        else:
            missing.append(binary)
    
    return len(missing) == 0, found, missing


def check_build_requirements() -> Tuple[bool, List[str]]:
    """
    Check if build requirements are available.
    
    Returns:
        Tuple of (can_build, missing_requirements)
    """
    missing = []
    
    # Use comprehensive dependency checker if available
    if HAS_DEPENDENCY_CHECKER:
        all_ok, status = check_dependencies(verbose=False)
        if not all_ok:
            for name, (ok, msg) in status.items():
                if not ok and name not in ("tcmalloc",):  # tcmalloc is optional
                    missing.append(f"{name}: {msg}")
            return False, missing
        return True, []
    
    # Fallback to basic checks
    # Check for make
    try:
        result = subprocess.run(["make", "--version"], capture_output=True, timeout=5)
        if result.returncode != 0:
            missing.append("make")
    except:
        missing.append("make")
    
    # Check for C++ compiler
    for compiler in ["g++", "clang++"]:
        try:
            result = subprocess.run([compiler, "--version"], capture_output=True, timeout=5)
            if result.returncode == 0:
                break
        except:
            pass
    else:
        missing.append("g++ or clang++")
    
    # Check for Makefile
    makefile = BENCH_DIR / "Makefile"
    if not makefile.exists():
        missing.append(f"Makefile at {makefile}")
    
    return len(missing) == 0, missing


def ensure_dependencies(auto_install: bool = False, verbose: bool = False) -> Tuple[bool, str]:
    """
    Ensure all build dependencies are installed.
    
    Args:
        auto_install: Automatically install missing dependencies (needs sudo)
        verbose: Print detailed status
        
    Returns:
        Tuple of (success, message)
    """
    if not HAS_DEPENDENCY_CHECKER:
        can_build, missing = check_build_requirements()
        if can_build:
            return True, "Basic requirements met"
        return False, f"Missing: {', '.join(missing)}"
    
    all_ok, status = check_dependencies(verbose=verbose)
    
    if all_ok:
        return True, "All dependencies satisfied"
    
    if auto_install:
        log.info("Attempting to install missing dependencies...")
        success, msg = install_dependencies(verbose=verbose)
        if success:
            # Re-check
            all_ok, _ = check_dependencies(verbose=False)
            if all_ok:
                return True, "Dependencies installed successfully"
        return False, msg
    
    # Build manual install message
    missing_names = [name for name, (ok, _) in status.items() if not ok]
    return False, f"Missing dependencies: {', '.join(missing_names)}. Run: python -m scripts.lib.dependencies --install"


def build_binaries(
    sim: bool = False,
    clean_first: bool = False,
    parallel: int = None,
    verbose: bool = False
) -> Tuple[bool, str]:
    """
    Build benchmark binaries using make.
    
    Args:
        sim: If True, build simulation binaries
        clean_first: If True, run 'make clean' before building
        parallel: Number of parallel jobs (default: CPU count)
        verbose: Show full build output
        
    Returns:
        Tuple of (success, message)
    """
    # Check requirements
    can_build, missing = check_build_requirements()
    if not can_build:
        return False, f"Missing build requirements: {', '.join(missing)}"
    
    if parallel is None:
        parallel = get_cpu_count()
    
    cwd = BENCH_DIR
    env = os.environ.copy()
    
    # Clean if requested
    if clean_first:
        log.info("Cleaning build artifacts...")
        try:
            result = subprocess.run(
                ["make", "clean"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode != 0 and verbose:
                log.warning(f"Clean warning: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            log.warning("Clean timed out")
        except Exception as e:
            log.warning(f"Clean failed: {e}")
    
    # Determine make target
    target = MAKE_TARGET_SIM if sim else MAKE_TARGET_ALL
    target_desc = "simulation binaries" if sim else "benchmark binaries"
    
    log.info(f"Building {target_desc} with {parallel} parallel jobs...")
    
    try:
        cmd = ["make", f"-j{parallel}", target]
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )
        
        if result.returncode != 0:
            error_msg = result.stderr[-1000:] if result.stderr else "Unknown error"
            log.error(f"Build failed: {error_msg}")
            return False, f"Build failed: {error_msg}"
        
        if verbose and result.stdout:
            print(result.stdout[-2000:])
        
        # Verify binaries were created
        all_exist, found, missing = check_binaries(sim=sim)
        if not all_exist:
            return False, f"Build completed but missing binaries: {', '.join(missing)}"
        
        log.success(f"Successfully built {len(found)} binaries: {', '.join(found)}")
        return True, f"Built {len(found)} binaries"
        
    except subprocess.TimeoutExpired:
        return False, "Build timed out after 30 minutes"
    except Exception as e:
        return False, f"Build error: {e}"


def clean_build() -> Tuple[bool, str]:
    """
    Clean build artifacts.
    
    Returns:
        Tuple of (success, message)
    """
    try:
        result = subprocess.run(
            ["make", "clean"],
            cwd=BENCH_DIR,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            log.success("Build cleaned successfully")
            return True, "Cleaned"
        else:
            return False, result.stderr[:500]
            
    except Exception as e:
        return False, str(e)


def ensure_binaries(sim: bool = False) -> bool:
    """
    Ensure binaries exist, building if necessary.
    
    Args:
        sim: If True, check/build simulation binaries
        
    Returns:
        True if binaries are available
    """
    all_exist, found, missing = check_binaries(sim=sim)
    
    if all_exist:
        log.info(f"All {len(found)} binaries found")
        return True
    
    log.info(f"Missing {len(missing)} binaries: {', '.join(missing)}")
    log.info("Building...")
    
    success, msg = build_binaries(sim=sim)
    return success


# =============================================================================
# Standalone CLI
# =============================================================================

def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="GraphBrew Build Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m scripts.lib.build --check              # Check if binaries exist
    python -m scripts.lib.build --check --sim        # Check simulation binaries
    python -m scripts.lib.build --build              # Build benchmark binaries
    python -m scripts.lib.build --build --sim        # Build simulation binaries
    python -m scripts.lib.build --build --clean      # Clean and rebuild
    python -m scripts.lib.build --clean              # Just clean
"""
    )
    
    parser.add_argument("--check", action="store_true",
                        help="Check if binaries exist")
    parser.add_argument("--build", action="store_true",
                        help="Build binaries")
    parser.add_argument("--clean", action="store_true",
                        help="Clean build artifacts")
    parser.add_argument("--sim", action="store_true",
                        help="Target simulation binaries")
    parser.add_argument("-j", "--jobs", type=int, default=None,
                        help="Parallel build jobs (default: CPU count)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    
    args = parser.parse_args()
    
    if args.clean:
        success, msg = clean_build()
        sys.exit(0 if success else 1)
    
    if args.check:
        all_exist, found, missing = check_binaries(sim=args.sim)
        bin_type = "simulation" if args.sim else "benchmark"
        
        print(f"\n{bin_type.title()} Binary Status:")
        print(f"  Found ({len(found)}): {', '.join(found) or 'none'}")
        print(f"  Missing ({len(missing)}): {', '.join(missing) or 'none'}")
        
        if all_exist:
            print(f"\n✓ All {bin_type} binaries available")
            sys.exit(0)
        else:
            print(f"\n✗ Missing {len(missing)} binaries")
            sys.exit(1)
    
    if args.build:
        success, msg = build_binaries(
            sim=args.sim,
            clean_first=args.clean,
            parallel=args.jobs,
            verbose=args.verbose
        )
        print(msg)
        sys.exit(0 if success else 1)
    
    # Default: show status
    parser.print_help()


if __name__ == "__main__":
    main()
