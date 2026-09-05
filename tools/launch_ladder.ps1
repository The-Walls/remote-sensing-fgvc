# Launch the full ladder as a detached process that survives the terminal /
# Claude session closing. Progress: Get-Content runs\_ladder_v2.log -Wait
#
#   powershell -ExecutionPolicy Bypass -File tools\launch_ladder.ps1 [-Seeds "0 1 2"]
param([string]$Seeds = "0 1 2")

$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path (Split-Path -Parent $root) "venv\Scripts\python.exe"
$log = Join-Path $root "runs\_ladder_v2.log"
$err = Join-Path $root "runs\_ladder_v2.err"

$p = Start-Process -FilePath $py `
    -ArgumentList "tools/run_ladder.py configs/fgscr42/L*.yaml --seeds $Seeds" `
    -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $log -RedirectStandardError $err
"launched pid $($p.Id) at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
"log: $log"
