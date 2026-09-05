# Launch the full ladder as a process that survives the launching terminal or
# app closing, with no visible window.
#
# Why this shape: a plain Start-Process child inherits the caller's Job Object
# (Claude Desktop, some terminals) and dies with it. A process created through
# WMI does not belong to that job. It still needs a console for the `>>` log
# redirection to work (DETACHED_PROCESS silently loses stdout), but closing a
# visible console window sends CTRL_CLOSE to the whole tree (MKL then aborts
# with "forrtl: error (200)") -- so the console is created hidden (SW_HIDE).
#
#   powershell -ExecutionPolicy Bypass -File tools\launch_ladder.ps1 [-Seeds "0 1 2"]
#   progress:  Get-Content runs\_ladder_v2.log -Wait -Tail 20
#   stop:      Get-Process python | Stop-Process     (resumable from latest.pt)
param([string]$Seeds = "0 1 2")

$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path (Split-Path -Parent $root) "venv\Scripts\python.exe"
$log = Join-Path $root "runs\_ladder_v2.log"

Add-Content $log "`n##### launch $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') seeds=$Seeds #####"
$cmd = "cmd.exe /c `"cd /d $root && $py tools\run_ladder.py configs/fgscr42/L*.yaml --seeds $Seeds >> $log 2>&1`""
$startup = ([wmiclass]"Win32_ProcessStartup").CreateInstance()
$startup.ShowWindow = [uint16]0          # SW_HIDE
$r = ([wmiclass]"Win32_Process").Create($cmd, $root, $startup)
if ($r.ReturnValue -ne 0) { throw "Win32_Process.Create failed: $($r.ReturnValue)" }
"launched (cmd pid $($r.ProcessId), hidden console) at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
"log: $log"
