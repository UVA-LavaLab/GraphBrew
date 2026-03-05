# =========================================================
# GraphBrew WSL Build Helper
# =========================================================
# Wraps make commands to run inside WSL from PowerShell.
#
# Usage:
#   .\build_wsl.ps1 all                    # Build all kernels
#   .\build_wsl.ps1 clean                  # Clean build artifacts
#   .\build_wsl.ps1 run-bfs               # Run BFS benchmark
#   .\build_wsl.ps1 run-pr                # Run PageRank benchmark
#   .\build_wsl.ps1 all RABBIT_ENABLE=0   # Build without RabbitOrder
#   .\build_wsl.ps1 help                  # Show make help
#   .\build_wsl.ps1 shell                 # Open WSL shell in project dir
# =========================================================

param(
    [Parameter(Position=0)]
    [string]$Target = "all",

    [Parameter(Position=1, ValueFromRemainingArguments=$true)]
    [string[]]$ExtraArgs
)

$distro = "Ubuntu-24.04"

# Convert Windows path to WSL path
$winPath = (Get-Location).Path
$wslPath = wsl -d $distro -- wslpath -u "$winPath" 2>&1
if ($LASTEXITCODE -ne 0) {
    # Fallback: manual conversion
    $wslPath = $winPath -replace '\\', '/'
    $wslPath = $wslPath -replace '^([A-Za-z]):', { "/mnt/" + $_.Groups[1].Value.ToLower() }
}

# Special case: open an interactive shell
if ($Target -eq "shell") {
    Write-Host "Opening WSL shell in $wslPath ..." -ForegroundColor Cyan
    wsl -d $distro -- bash -c "cd '$wslPath' && exec bash"
    exit $LASTEXITCODE
}

# Build the make command
$makeArgs = @($Target) + $ExtraArgs
$makeCmd = "make " + ($makeArgs -join " ")

Write-Host "[GraphBrew] Running: $makeCmd" -ForegroundColor Cyan
Write-Host "[GraphBrew] Directory: $wslPath" -ForegroundColor DarkGray
Write-Host ""

wsl -d $distro -- bash -c "cd '$wslPath' && $makeCmd"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "[GraphBrew] Success!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "[GraphBrew] Failed (exit code: $LASTEXITCODE)" -ForegroundColor Red
    exit $LASTEXITCODE
}
