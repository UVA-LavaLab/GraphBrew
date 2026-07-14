# =========================================================
# GraphBrew WSL Setup Script
# =========================================================
# Run this script in PowerShell (Admin recommended for WSL install)
# Usage: .\setup_wsl.ps1
# =========================================================

$ErrorActionPreference = "Stop"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host " GraphBrew WSL Setup" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# Step 1: Check if Ubuntu-24.04 is installed
$distros = wsl --list --quiet 2>&1
if ($distros -match "Ubuntu-24.04") {
    Write-Host "[OK] Ubuntu-24.04 already installed" -ForegroundColor Green
} else {
    Write-Host "[INSTALL] Installing Ubuntu-24.04 via WSL..." -ForegroundColor Yellow
    Write-Host "  This will download ~700MB and take a few minutes." -ForegroundColor Yellow
    Write-Host "  You will be asked to create a Unix username and password." -ForegroundColor Yellow
    wsl --install -d Ubuntu-24.04
    Write-Host ""
    Write-Host "[NOTE] If WSL just installed for the first time, restart your PC then re-run this script." -ForegroundColor Magenta
    exit 0
}

# Step 2: Install build dependencies inside WSL
Write-Host ""
Write-Host "[SETUP] Installing build dependencies inside WSL..." -ForegroundColor Yellow

$setupScript = @'
set -e
echo ">>> Updating package lists..."
sudo apt-get update -qq

echo ">>> Installing build tools and dependencies..."
sudo apt-get install -y -qq \
    build-essential \
    g++ \
    make \
    libomp-dev \
    libboost-all-dev \
    libnuma-dev \
    libgoogle-perftools-dev \
    python3 \
    python3-pip \
    2>&1 | tail -5

echo ""
echo ">>> Verifying installation..."
echo "  g++ version: $(g++ --version | head -1)"
echo "  make version: $(make --version | head -1)"
echo "  OpenMP: $(dpkg -s libomp-dev 2>/dev/null | grep Version || echo 'installed via g++')"
echo ""
echo ">>> Dependencies installed successfully!"
'@

wsl -d Ubuntu-24.04 -- bash -c $setupScript

# Step 3: Print usage instructions
Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host " Setup Complete!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""
Write-Host "You can now build and run GraphBrew with:" -ForegroundColor White
Write-Host ""
Write-Host "  Option A: Use the build helper script" -ForegroundColor Cyan
Write-Host "    .\build_wsl.ps1 all                  # Build everything" -ForegroundColor Gray
Write-Host "    .\build_wsl.ps1 bfs                  # Build single kernel" -ForegroundColor Gray
Write-Host "    .\build_wsl.ps1 run-bfs              # Run BFS benchmark" -ForegroundColor Gray
Write-Host "    .\build_wsl.ps1 clean                # Clean build" -ForegroundColor Gray
Write-Host ""
Write-Host "  Option B: Drop into WSL directly" -ForegroundColor Cyan
Write-Host "    wsl -d Ubuntu-24.04" -ForegroundColor Gray
Write-Host "    cd /mnt/c/Users/amughrabi/Documents/00_github_repos/GraphBrew" -ForegroundColor Gray
Write-Host "    make all" -ForegroundColor Gray
Write-Host ""
Write-Host "  Option C: One-liner from PowerShell" -ForegroundColor Cyan
Write-Host "    wsl -d Ubuntu-24.04 -- bash -c 'cd /mnt/c/Users/amughrabi/Documents/00_github_repos/GraphBrew && make all'" -ForegroundColor Gray
Write-Host ""
