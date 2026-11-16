# HartsOfflineToolkit (Windows 10/11) - Automated PowerShell Setup
$ErrorActionPreference = "Stop"

Write-Host "=== HartsOfflineToolkit Windows Setup ===" -ForegroundColor Cyan

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python 3.10-3.12 was not found in PATH. Install Python from python.org and retry."
}

if (-not (Test-Path ".venv")) {
    Write-Host "[1/4] Creating isolated virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
}

$VenvPython = ".\.venv\Scripts\python.exe"
$VenvPip = ".\.venv\Scripts\pip.exe"

Write-Host "[2/4] Installing CPU-only PyTorch and Torchvision..." -ForegroundColor Yellow
& $VenvPython -m pip install --upgrade pip setuptools wheel
& $VenvPip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

Write-Host "[3/4] Installing HartsOfflineToolkit dependencies..." -ForegroundColor Yellow
& $VenvPip install -r requirements.txt

Write-Host "[4/4] Initializing runtime directories, downloading EmbeddingGemma 2, and locking weights..." -ForegroundColor Yellow
& $VenvPython cli.py init

Write-Host "`nSetup complete! Activate your environment with:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "Then drop documents into .\data\raw and run:" -ForegroundColor Green
Write-Host "  python cli.py index" -ForegroundColor White
