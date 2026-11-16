# HartsOfflineToolkit (Windows 10/11) - Native SHA256SUMS Verification Script
$ErrorActionPreference = "Stop"

if (-not (Test-Path "SHA256SUMS")) {
    Write-Error "SHA256SUMS manifest not found in current directory."
}

$Failed = 0
Get-Content "SHA256SUMS" | ForEach-Object {
    if ($_ -match "^([0-9a-fA-F]{64})\s+\*?(.+)$") {
        $Expected = $Matches[1].ToLower()
        $RelPath = $Matches[2].Trim() -replace "/", "\"
        if (-not (Test-Path $RelPath)) {
            Write-Host "${RelPath}: MISSING" -ForegroundColor Red
            $Failed++
        } else {
            $Actual = (Get-FileHash -Path $RelPath -Algorithm SHA256).Hash.ToLower()
            if ($Actual -eq $Expected) {
                Write-Host "${RelPath}: OK" -ForegroundColor Green
            } else {
                Write-Host "${RelPath}: FAILED (hash mismatch)" -ForegroundColor Red
                $Failed++
            }
        }
    }
}

if ($Failed -gt 0) {
    Write-Error "$Failed file(s) failed SHA-256 verification."
}
Write-Host "`nAll tracked files verified successfully." -ForegroundColor Cyan
