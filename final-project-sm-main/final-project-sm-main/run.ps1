# Creates virtual environment, installs dependencies, runs model pipeline.
# Usage: powershell -ExecutionPolicy Bypass -File .\run.ps1

$ErrorActionPreference = "Stop"

Write-Host "[1/4] Creating virtual environment (.venv) if missing..." -ForegroundColor Cyan
if (!(Test-Path .venv)) {
    python -m venv .venv
}

Write-Host "[2/4] Activating virtual environment..." -ForegroundColor Cyan
.\.venv\Scripts\Activate.ps1

Write-Host "[3/4] Installing dependencies..." -ForegroundColor Cyan
pip install --upgrade pip
pip install -r requirements.txt

Write-Host "[4/4] Running model pipeline..." -ForegroundColor Cyan

if ($Env:TF_USE_LEGACY_KERAS) {
    Write-Host "[INFO] Unsetting TF_USE_LEGACY_KERAS for bundled keras." -ForegroundColor Yellow
    Remove-Item Env:TF_USE_LEGACY_KERAS
}
python model.py

Write-Host "Done. Artifacts in .\models" -ForegroundColor Green
