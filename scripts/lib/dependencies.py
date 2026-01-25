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
        # GraphBrew recommended path - source tarball structure
        "/opt/boost_1_58_0",
        # GraphBrew recommended path - compiled structure
        "/opt/boost_1_58_0/include",
        # Standard system paths
        "/usr/include",
        "/usr/local/include",
        # macOS paths
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
    
    Checks multiple locations:
    1. /opt/boost_1_58_0 (GraphBrew recommended install path from README)
    2. Standard system paths (/usr/include, /usr/local/include)
    3. Homebrew paths (macOS)
    
    Returns:
        Version string or None if not found
    """
    version_headers = [
        # GraphBrew recommended path - source tarball structure
        "/opt/boost_1_58_0/boost/version.hpp",
        # GraphBrew recommended path - compiled structure  
        "/opt/boost_1_58_0/include/boost/version.hpp",
        # Standard system paths
        "/usr/include/boost/version.hpp",
        "/usr/local/include/boost/version.hpp",
        # macOS Homebrew paths
        "/opt/homebrew/include/boost/version.hpp",
        "/usr/local/opt/boost/include/boost/version.hpp",
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


def install_boost_158(
    install_path: str = "/opt/boost_1_58_0",
    dry_run: bool = False,
    verbose: bool = False
) -> Tuple[bool, str]:
    """
    Download, compile, and install Boost 1.58.0 for RabbitOrder compatibility.
    
    This downloads Boost 1.58.0 source, compiles it with bootstrap.sh and b2,
    and installs to /opt/boost_1_58_0 with proper include/ and lib/ structure
    as expected by the GraphBrew Makefile.
    
    Args:
        install_path: Where to install Boost (default: /opt/boost_1_58_0)
        dry_run: Just print commands without executing
        verbose: Print detailed output
        
    Returns:
        Tuple of (success, message)
    """
    # Check if already installed (compiled structure with include/)
    if os.path.exists(os.path.join(install_path, "include", "boost", "version.hpp")):
        return True, f"Boost 1.58.0 already installed at {install_path}"
    
    boost_url = "https://archives.boost.io/release/1.58.0/source/boost_1_58_0.tar.gz"
    tmp_dir = "/tmp/boost_install"
    
    # Get CPU cores for parallel compilation
    try:
        cpu_cores = os.cpu_count() or 4
    except:
        cpu_cores = 4
    
    if dry_run:
        commands = [
            f"mkdir -p {tmp_dir}",
            f"cd {tmp_dir} && wget {boost_url}",
            f"cd {tmp_dir} && tar -xzf boost_1_58_0.tar.gz",
            f"cd {tmp_dir}/boost_1_58_0 && ./bootstrap.sh --prefix={install_path}",
            f"cd {tmp_dir}/boost_1_58_0 && sudo ./b2 --with=all -j {cpu_cores} install",
            f"rm -rf {tmp_dir}",
        ]
        return True, "Would run:\n  " + "\n  ".join(commands)
    
    log.info("Downloading, compiling, and installing Boost 1.58.0 for RabbitOrder...")
    log.info(f"Install path: {install_path}")
    log.info(f"Using {cpu_cores} CPU cores for compilation")
    log.warning("This may take 10-30 minutes depending on your system...")
    
    try:
        # Check for required tools
        if shutil.which("wget") is None and shutil.which("curl") is None:
            return False, "Neither wget nor curl found. Please install one of them."
        
        # Create temp directory
        os.makedirs(tmp_dir, exist_ok=True)
        
        # Download
        log.info("Step 1/4: Downloading Boost 1.58.0 (approx 73MB)...")
        if shutil.which("wget"):
            download_cmd = ["wget", "-q", "--show-progress", boost_url]
        else:
            download_cmd = ["curl", "-L", "-o", "boost_1_58_0.tar.gz", boost_url]
        
        result = subprocess.run(
            download_cmd,
            cwd=tmp_dir,
            timeout=600,  # 10 min timeout for download
        )
        if result.returncode != 0:
            return False, "Download failed. Check your internet connection."
        
        # Extract
        log.info("Step 2/4: Extracting...")
        result = subprocess.run(
            ["tar", "-xzf", "boost_1_58_0.tar.gz"],
            cwd=tmp_dir,
            timeout=120,
        )
        if result.returncode != 0:
            return False, "Failed to extract archive"
        
        boost_src = os.path.join(tmp_dir, "boost_1_58_0")
        
        # Bootstrap
        log.info("Step 3/4: Running bootstrap.sh...")
        result = subprocess.run(
            ["./bootstrap.sh", f"--prefix={install_path}"],
            cwd=boost_src,
            timeout=300,  # 5 min timeout
            capture_output=not verbose,
        )
        if result.returncode != 0:
            error_msg = ""
            if result.stderr:
                error_msg = result.stderr.decode()[:500]
            return False, f"bootstrap.sh failed: {error_msg}"
        
        # Compile and install with b2
        log.info(f"Step 4/4: Compiling and installing (this takes a while)...")
        log.info(f"Running: sudo ./b2 --with=all -j {cpu_cores} install")
        result = subprocess.run(
            ["sudo", "./b2", "--with=all", "-j", str(cpu_cores), "install"],
            cwd=boost_src,
            timeout=3600,  # 1 hour timeout for compilation
        )
        if result.returncode != 0:
            # b2 may return non-zero but still succeed partially
            # Check if essential files were created
            if not os.path.exists(os.path.join(install_path, "include", "boost", "version.hpp")):
                return False, "b2 compilation failed. Check build output."
        
        # Cleanup
        log.info("Cleaning up temporary files...")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        
        # Verify
        if os.path.exists(os.path.join(install_path, "include", "boost", "version.hpp")):
            log.success(f"Boost 1.58.0 compiled and installed successfully at {install_path}")
            return True, f"Boost 1.58.0 installed at {install_path}"
        else:
            return False, "Installation completed but verification failed"
            
    except subprocess.TimeoutExpired:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False, "Installation timed out (compilation can take 30+ minutes)"
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False, f"Installation error: {e}"


def check_boost_158() -> Tuple[bool, str]:
    """
    Check if Boost 1.58.0 is installed at the expected path for RabbitOrder.
    
    Returns:
        Tuple of (is_installed, message)
    """
    install_path = "/opt/boost_1_58_0"
    # Check both source tarball structure (boost/) and compiled structure (include/boost/)
    version_file = os.path.join(install_path, "boost", "version.hpp")
    if not os.path.exists(version_file):
        version_file = os.path.join(install_path, "include", "boost", "version.hpp")
    
    if not os.path.exists(version_file):
        return False, f"Boost 1.58.0 not found at {install_path}"
    
    # Verify it's actually 1.58
    try:
        with open(version_file, "r") as f:
            content = f.read()
            if "105800" in content:  # BOOST_VERSION for 1.58.0
                return True, f"Boost 1.58.0 found at {install_path}"
            else:
                return False, f"Boost at {install_path} is not version 1.58.0"
    except Exception as e:
        return False, f"Error reading version: {e}"


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
        libnuma-dev \\
        google-perftools

Fedora/RHEL/CentOS:
    sudo dnf install -y \\
        gcc-c++ \\
        make \\
        numactl-devel \\
        gperftools

Arch Linux:
    sudo pacman -S \\
        base-devel \\
        numactl \\
        gperftools

macOS (with Homebrew):
    xcode-select --install
    brew install gcc google-perftools

Boost 1.58.0 for RabbitOrder (Required):
----------------------------------------
RabbitOrder requires Boost 1.58.0 specifically. System package managers
typically install newer versions which may cause compatibility issues.

Automatic installation (recommended):
    python3 scripts/graphbrew_experiment.py --install-boost
    
    This downloads, compiles, and installs Boost 1.58.0 automatically.
    Note: Compilation takes 10-30 minutes.

Manual installation:
    wget https://archives.boost.io/release/1.58.0/source/boost_1_58_0.tar.gz
    tar -xzf boost_1_58_0.tar.gz
    cd boost_1_58_0
    ./bootstrap.sh --prefix=/opt/boost_1_58_0
    sudo ./b2 --with=all -j $(nproc) install
    cd .. && rm -rf boost_1_58_0 boost_1_58_0.tar.gz

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
    python -m scripts.lib.dependencies --install-boost   # Install Boost 1.58
    python -m scripts.lib.dependencies --install --dry-run  # Show what would install
    python -m scripts.lib.dependencies --instructions    # Print install guide
"""
    )
    
    parser.add_argument("--check", action="store_true",
                        help="Check if dependencies are installed")
    parser.add_argument("--install", action="store_true",
                        help="Install missing dependencies (may need sudo)")
    parser.add_argument("--install-boost", action="store_true",
                        help="Download and install Boost 1.58.0 for RabbitOrder")
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
    
    if args.install_boost:
        # Check current status first
        is_installed, msg = check_boost_158()
        if is_installed:
            print(f"✓ {msg}")
            sys.exit(0)
        
        # Install Boost 1.58
        success, msg = install_boost_158(dry_run=args.dry_run, verbose=args.verbose)
        print(msg)
        if success and not args.dry_run:
            print("\nVerifying installation...")
            is_ok, verify_msg = check_boost_158()
            print(f"  {'✓' if is_ok else '✗'} {verify_msg}")
        sys.exit(0 if success else 1)
    
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
