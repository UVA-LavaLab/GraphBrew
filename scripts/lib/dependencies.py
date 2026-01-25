#!/usr/bin/env python3
"""
Dependency management for GraphBrew.

Checks and optionally installs system dependencies required for building
and running GraphBrew benchmarks.

Standalone usage:
    python -m scripts.lib.dependencies --check           # Check all dependencies
    python -m scripts.lib.dependencies --install         # Install missing (needs sudo)
    python -m scripts.lib.dependencies --check --verbose # Detailed status

Library usage:
    from scripts.lib.dependencies import check_dependencies, install_dependencies
    
    ok, missing = check_dependencies()
    if not ok:
        install_dependencies(missing)
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .utils import Logger

# Initialize logger
log = Logger()

# =============================================================================
# Dependency Definitions
# =============================================================================

# System packages required for GraphBrew
# Format: {package_name: {distro: package_name, ...}}
SYSTEM_PACKAGES = {
    "build-essential": {
        "ubuntu": "build-essential",
        "debian": "build-essential", 
        "fedora": "gcc-c++ make",
        "rhel": "gcc-c++ make",
        "centos": "gcc-c++ make",
        "arch": "base-devel",
        "macos": "xcode-select --install",  # Special case
    },
    "g++": {
        "ubuntu": "g++",
        "debian": "g++",
        "fedora": "gcc-c++",
        "rhel": "gcc-c++",
        "centos": "gcc-c++",
        "arch": "gcc",
        "macos": "gcc",  # via brew
    },
    "boost": {
        "ubuntu": "libboost-all-dev",
        "debian": "libboost-all-dev",
        "fedora": "boost-devel",
        "rhel": "boost-devel",
        "centos": "boost-devel",
        "arch": "boost",
        "macos": "boost",  # via brew
    },
    "numa": {
        "ubuntu": "libnuma-dev",
        "debian": "libnuma-dev",
        "fedora": "numactl-devel",
        "rhel": "numactl-devel",
        "centos": "numactl-devel",
        "arch": "numactl",
        "macos": None,  # Not available on macOS
    },
    "tcmalloc": {
        "ubuntu": "google-perftools",
        "debian": "google-perftools",
        "fedora": "gperftools",
        "rhel": "gperftools",
        "centos": "gperftools",
        "arch": "gperftools",
        "macos": "google-perftools",  # via brew
    },
}

# Minimum versions
MIN_VERSIONS = {
    "g++": "7.0.0",
    "boost": "1.58.0",
    "python": "3.8.0",
}

# Commands to check if a tool/library is available
CHECK_COMMANDS = {
    "make": ["make", "--version"],
    "g++": ["g++", "--version"],
    "clang++": ["clang++", "--version"],
    "boost": None,  # Special check via header file
    "numa": None,   # Special check via header file
    "tcmalloc": None,  # Special check via library
}


# =============================================================================
# Platform Detection
# =============================================================================

def detect_platform() -> Tuple[str, str]:
    """
    Detect the current operating system and distribution.
    
    Returns:
        Tuple of (os_type, distro) where os_type is 'linux', 'macos', or 'windows'
        and distro is the specific distribution (e.g., 'ubuntu', 'fedora')
    """
    system = platform.system().lower()
    
    if system == "darwin":
        return "macos", "macos"
    elif system == "windows":
        return "windows", "windows"
    elif system == "linux":
        # Try to detect Linux distribution
        distro = "linux"  # Generic fallback
        
        # Check /etc/os-release (modern standard)
        if Path("/etc/os-release").exists():
            with open("/etc/os-release") as f:
                content = f.read().lower()
                if "ubuntu" in content:
                    distro = "ubuntu"
                elif "debian" in content:
                    distro = "debian"
                elif "fedora" in content:
                    distro = "fedora"
                elif "rhel" in content or "red hat" in content:
                    distro = "rhel"
                elif "centos" in content:
                    distro = "centos"
                elif "arch" in content:
                    distro = "arch"
        
        return "linux", distro
    else:
        return "unknown", "unknown"


def get_package_manager() -> Optional[str]:
    """
    Detect the system package manager.
    
    Returns:
        Package manager command or None if not detected
    """
    os_type, distro = detect_platform()
    
    if os_type == "macos":
        if shutil.which("brew"):
            return "brew"
        return None
    
    if os_type == "linux":
        # Check for package managers in order of preference
        if shutil.which("apt-get"):
            return "apt-get"
        elif shutil.which("apt"):
            return "apt"
        elif shutil.which("dnf"):
            return "dnf"
        elif shutil.which("yum"):
            return "yum"
        elif shutil.which("pacman"):
            return "pacman"
    
    return None


# =============================================================================
# Dependency Checking
# =============================================================================

def check_command(cmd: List[str], timeout: int = 10) -> Tuple[bool, str]:
    """
    Check if a command runs successfully.
    
    Args:
        cmd: Command and arguments to run
        timeout: Timeout in seconds
        
    Returns:
        Tuple of (success, version_string)
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode == 0:
            # Extract version from first line
            output = result.stdout.strip() or result.stderr.strip()
            first_line = output.split('\n')[0] if output else ""
            return True, first_line
        return False, ""
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False, ""


def check_header_exists(header: str) -> bool:
    """
    Check if a C++ header file exists in standard include paths.
    
    Args:
        header: Header file path (e.g., 'boost/version.hpp')
        
    Returns:
        True if header exists
    """
    include_paths = [
        "/usr/include",
        "/usr/local/include",
        "/opt/homebrew/include",  # macOS ARM
        "/usr/local/opt/boost/include",  # macOS Intel
    ]
    
    for base in include_paths:
        if Path(base, header).exists():
            return True
    return False


def get_boost_version() -> Optional[str]:
    """
    Get the installed Boost version.
    
    Returns:
        Version string or None if not found
    """
    version_headers = [
        "/usr/include/boost/version.hpp",
        "/usr/local/include/boost/version.hpp",
        "/opt/homebrew/include/boost/version.hpp",
    ]
    
    for header_path in version_headers:
        if Path(header_path).exists():
            try:
                with open(header_path) as f:
                    content = f.read()
                    # Look for: #define BOOST_LIB_VERSION "1_74"
                    for line in content.split('\n'):
                        if 'BOOST_LIB_VERSION' in line and '"' in line:
                            version = line.split('"')[1].replace('_', '.')
                            return version
            except:
                pass
    return None


def check_library_exists(lib_name: str) -> bool:
    """
    Check if a shared library exists.
    
    Args:
        lib_name: Library name (e.g., 'tcmalloc')
        
    Returns:
        True if library can be found
    """
    # Try using ldconfig on Linux
    try:
        result = subprocess.run(
            ["ldconfig", "-p"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if lib_name in result.stdout:
            return True
    except:
        pass
    
    # Check common library paths
    lib_paths = [
        "/usr/lib",
        "/usr/lib64",
        "/usr/local/lib",
        "/usr/local/lib64",
        "/opt/homebrew/lib",
    ]
    
    patterns = [f"lib{lib_name}.so", f"lib{lib_name}.dylib", f"lib{lib_name}.a"]
    
    for base in lib_paths:
        base_path = Path(base)
        if base_path.exists():
            for pattern in patterns:
                if list(base_path.glob(f"{pattern}*")):
                    return True
    
    return False


def check_compiler_version(compiler: str = "g++") -> Tuple[bool, str]:
    """
    Check compiler version meets minimum requirements.
    
    Args:
        compiler: Compiler command ('g++' or 'clang++')
        
    Returns:
        Tuple of (meets_minimum, version_string)
    """
    ok, version_line = check_command([compiler, "--version"])
    if not ok:
        return False, "not found"
    
    # Extract version number
    import re
    version_match = re.search(r'(\d+\.\d+\.?\d*)', version_line)
    if version_match:
        version = version_match.group(1)
        # Compare with minimum
        min_version = MIN_VERSIONS.get(compiler, "0")
        try:
            from packaging.version import Version
            if Version(version) >= Version(min_version):
                return True, version
            return False, f"{version} (need {min_version}+)"
        except ImportError:
            # Simple comparison if packaging not available
            return True, version
    
    return True, version_line


def check_dependencies(verbose: bool = False) -> Tuple[bool, Dict[str, Tuple[bool, str]]]:
    """
    Check all GraphBrew dependencies.
    
    Args:
        verbose: Print detailed status
        
    Returns:
        Tuple of (all_ok, status_dict) where status_dict maps
        dependency name to (is_ok, status_message)
    """
    status = {}
    
    # Check make
    ok, ver = check_command(["make", "--version"])
    status["make"] = (ok, ver if ok else "not found")
    
    # Check C++ compiler
    compiler_ok = False
    for compiler in ["g++", "clang++"]:
        ok, ver = check_compiler_version(compiler)
        if ok:
            status["c++ compiler"] = (True, f"{compiler} {ver}")
            compiler_ok = True
            break
    if not compiler_ok:
        status["c++ compiler"] = (False, "g++ or clang++ not found (need g++ 7+)")
    
    # Check Boost
    boost_ver = get_boost_version()
    if boost_ver:
        # Check minimum version
        try:
            from packaging.version import Version
            min_boost = MIN_VERSIONS.get("boost", "1.58.0")
            if Version(boost_ver) >= Version(min_boost):
                status["boost"] = (True, f"v{boost_ver}")
            else:
                status["boost"] = (False, f"v{boost_ver} (need {min_boost}+)")
        except ImportError:
            status["boost"] = (True, f"v{boost_ver}")
    elif check_header_exists("boost/version.hpp"):
        status["boost"] = (True, "found (version unknown)")
    else:
        status["boost"] = (False, "not found (required for RabbitOrder)")
    
    # Check numa (optional on macOS)
    os_type, _ = detect_platform()
    if os_type == "macos":
        status["numa"] = (True, "not needed on macOS")
    elif check_header_exists("numa.h") or check_header_exists("numaif.h"):
        status["numa"] = (True, "found")
    else:
        status["numa"] = (False, "not found (libnuma-dev)")
    
    # Check tcmalloc (optional but recommended)
    if check_library_exists("tcmalloc") or check_library_exists("tcmalloc_minimal"):
        status["tcmalloc"] = (True, "found")
    else:
        status["tcmalloc"] = (False, "not found (google-perftools, optional)")
    
    # Check Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    min_py = MIN_VERSIONS.get("python", "3.8.0")
    try:
        from packaging.version import Version
        py_ok = Version(py_ver) >= Version(min_py)
    except ImportError:
        py_ok = sys.version_info >= (3, 8)
    status["python"] = (py_ok, f"v{py_ver}" if py_ok else f"v{py_ver} (need {min_py}+)")
    
    # Overall status
    # Required: make, c++ compiler, boost
    # Optional: numa (Linux only), tcmalloc
    required_ok = all([
        status.get("make", (False,))[0],
        status.get("c++ compiler", (False,))[0],
        status.get("boost", (False,))[0],
        status.get("python", (False,))[0],
    ])
    
    if verbose:
        print("\nGraphBrew Dependency Status:")
        print("=" * 50)
        for name, (ok, msg) in status.items():
            symbol = "✓" if ok else "✗"
            print(f"  {symbol} {name:15} {msg}")
        print("=" * 50)
        if required_ok:
            print("  All required dependencies satisfied!")
        else:
            print("  ⚠ Missing required dependencies")
    
    return required_ok, status


# =============================================================================
# Dependency Installation
# =============================================================================

def get_install_command(packages: List[str]) -> Optional[List[str]]:
    """
    Get the command to install packages on this system.
    
    Args:
        packages: List of package names to install
        
    Returns:
        Command list or None if not supported
    """
    os_type, distro = detect_platform()
    pkg_mgr = get_package_manager()
    
    if not pkg_mgr:
        return None
    
    # Map generic package names to distro-specific names
    distro_packages = []
    for pkg in packages:
        if pkg in SYSTEM_PACKAGES:
            distro_pkg = SYSTEM_PACKAGES[pkg].get(distro)
            if distro_pkg:
                distro_packages.extend(distro_pkg.split())
        else:
            distro_packages.append(pkg)
    
    if not distro_packages:
        return None
    
    # Build command based on package manager
    if pkg_mgr in ("apt-get", "apt"):
        return ["sudo", pkg_mgr, "install", "-y"] + distro_packages
    elif pkg_mgr == "dnf":
        return ["sudo", "dnf", "install", "-y"] + distro_packages
    elif pkg_mgr == "yum":
        return ["sudo", "yum", "install", "-y"] + distro_packages
    elif pkg_mgr == "pacman":
        return ["sudo", "pacman", "-S", "--noconfirm"] + distro_packages
    elif pkg_mgr == "brew":
        return ["brew", "install"] + distro_packages
    
    return None


def install_dependencies(
    missing: Optional[List[str]] = None,
    dry_run: bool = False,
    verbose: bool = False
) -> Tuple[bool, str]:
    """
    Install missing dependencies.
    
    Args:
        missing: List of missing dependencies (if None, auto-detect)
        dry_run: Just print commands without executing
        verbose: Print detailed output
        
    Returns:
        Tuple of (success, message)
    """
    os_type, distro = detect_platform()
    pkg_mgr = get_package_manager()
    
    if not pkg_mgr:
        return False, f"No supported package manager found on {distro}"
    
    # Detect missing if not provided
    if missing is None:
        _, status = check_dependencies(verbose=False)
        missing = [name for name, (ok, _) in status.items() if not ok]
    
    if not missing:
        return True, "All dependencies already installed"
    
    # Map missing deps to packages
    packages_to_install = []
    for dep in missing:
        if dep == "c++ compiler":
            packages_to_install.append("g++")
        elif dep in SYSTEM_PACKAGES:
            packages_to_install.append(dep)
    
    if not packages_to_install:
        return True, "No packages to install"
    
    # Get install command
    cmd = get_install_command(packages_to_install)
    if not cmd:
        # Provide manual instructions
        _, distro = detect_platform()
        if distro in ("ubuntu", "debian"):
            manual = f"sudo apt-get install -y {' '.join(packages_to_install)}"
        elif distro in ("fedora", "rhel", "centos"):
            manual = f"sudo dnf install -y {' '.join(packages_to_install)}"
        elif distro == "macos":
            manual = f"brew install {' '.join(packages_to_install)}"
        else:
            manual = f"Install: {', '.join(packages_to_install)}"
        
        return False, f"Manual install required:\n  {manual}"
    
    if dry_run:
        return True, f"Would run: {' '.join(cmd)}"
    
    log.info(f"Installing: {', '.join(packages_to_install)}")
    log.info(f"Running: {' '.join(cmd)}")
    
    try:
        # First update package lists on apt systems
        if pkg_mgr in ("apt-get", "apt"):
            log.info("Updating package lists...")
            subprocess.run(
                ["sudo", pkg_mgr, "update"],
                check=False,
                timeout=300
            )
        
        # Install packages
        result = subprocess.run(
            cmd,
            timeout=600,  # 10 minute timeout
            capture_output=not verbose
        )
        
        if result.returncode == 0:
            log.success(f"Successfully installed: {', '.join(packages_to_install)}")
            return True, "Installation complete"
        else:
            error = result.stderr.decode() if result.stderr else "Unknown error"
            return False, f"Installation failed: {error[:500]}"
            
    except subprocess.TimeoutExpired:
        return False, "Installation timed out"
    except Exception as e:
        return False, f"Installation error: {e}"


def print_install_instructions():
    """Print manual installation instructions for all platforms."""
    print("""
GraphBrew System Dependencies
=============================

Ubuntu/Debian:
    sudo apt-get update
    sudo apt-get install -y \\
        build-essential \\
        g++ \\
        libboost-all-dev \\
        libnuma-dev \\
        google-perftools

Fedora/RHEL/CentOS:
    sudo dnf install -y \\
        gcc-c++ \\
        make \\
        boost-devel \\
        numactl-devel \\
        gperftools

Arch Linux:
    sudo pacman -S \\
        base-devel \\
        boost \\
        numactl \\
        gperftools

macOS (with Homebrew):
    xcode-select --install
    brew install gcc boost google-perftools

After installing dependencies:
    make clean && make all
""")


# =============================================================================
# Standalone CLI
# =============================================================================

def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="GraphBrew Dependency Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m scripts.lib.dependencies --check           # Check dependencies
    python -m scripts.lib.dependencies --check --verbose # Detailed status
    python -m scripts.lib.dependencies --install         # Install missing
    python -m scripts.lib.dependencies --install --dry-run  # Show what would install
    python -m scripts.lib.dependencies --instructions    # Print install guide
"""
    )
    
    parser.add_argument("--check", action="store_true",
                        help="Check if dependencies are installed")
    parser.add_argument("--install", action="store_true",
                        help="Install missing dependencies (may need sudo)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be installed without doing it")
    parser.add_argument("--instructions", action="store_true",
                        help="Print manual installation instructions")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    
    args = parser.parse_args()
    
    if args.instructions:
        print_install_instructions()
        sys.exit(0)
    
    if args.check or (not args.install):
        all_ok, status = check_dependencies(verbose=True)
        if not all_ok and not args.install:
            print("\nTo install missing dependencies:")
            print("  python -m scripts.lib.dependencies --install")
            print("\nOr see manual instructions:")
            print("  python -m scripts.lib.dependencies --instructions")
        sys.exit(0 if all_ok else 1)
    
    if args.install:
        success, msg = install_dependencies(dry_run=args.dry_run, verbose=args.verbose)
        print(msg)
        if success and not args.dry_run:
            print("\nVerifying installation...")
            check_dependencies(verbose=True)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
