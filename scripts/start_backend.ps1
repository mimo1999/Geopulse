# =============================================================
# GLDT Backend Startup Script (PowerShell — Windows / Local PG18)
# =============================================================
# Usage: .\scripts\start_backend.ps1
#        .\scripts\start_backend.ps1 -Port 8000 -LogLevel info
#
# Requires:
#   - PostgreSQL 18 service running (postgresql-x64-18)
#   - Python 3.10+ with project dependencies installed
#   - models/checkpoints/forecaster_v1_best.pt (train first if missing)
# =============================================================

param(
    [int]   $Port       = 8000,
    [string]$LogLevel   = "info",
    [string]$ModelPath  = "models\checkpoints\run_phase1_best.pt",
    [string]$ForecasterPath = "models\checkpoints\forecaster_v1_best.pt"
)

# --- Verify PostgreSQL is running ---
$svc = Get-Service "postgresql-x64-18" -ErrorAction SilentlyContinue
if ($svc -and $svc.Status -ne "Running") {
    Write-Host "Starting PostgreSQL service..." -ForegroundColor Yellow
    Start-Service "postgresql-x64-18"
    Start-Sleep -Seconds 3
}

# --- Set environment variables ---
$env:DATABASE_SYNC_URL = "postgresql://gldt:gldt_secret@localhost:5432/gdelt_risk"
$env:MODEL_PATH        = $ModelPath
$env:FORECASTER_PATH   = $ForecasterPath

Write-Host ""
Write-Host "=== GLDT Backend ===" -ForegroundColor Cyan
Write-Host "  DB  : $env:DATABASE_SYNC_URL"
Write-Host "  Port: $Port"
Write-Host "  Risk model      : $ModelPath"
Write-Host "  Forecaster model: $ForecasterPath"
Write-Host ""

# --- Launch uvicorn ---
python -m uvicorn backend.main:app `
    --host 0.0.0.0 `
    --port $Port `
    --log-level $LogLevel `
    --reload
