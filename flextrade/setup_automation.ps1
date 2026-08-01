# Registers FlexTrade's background jobs as Windows Scheduled Tasks.
#
#   .\setup_automation.ps1            # register (or refresh) both tasks
#   .\setup_automation.ps1 -Remove    # unregister them
#
# Two jobs:
#   FlexTrade-DailyPipeline  11:00 daily — refresh live data, self-heal any
#                            gaps, forecast tomorrow, optimize, emit bids.
#                            11:00 because the IEX DAM gate closes ~12:00.
#   FlexTrade-BessPoller     at logon  — samples the BRPL battery every
#                            2 min. SLDC serves no history, so telemetry
#                            only exists if we are sampling continuously.
#
# Both run under the current user (no admin needed) and log to logs/.

param([switch]$Remove)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$logs = Join-Path $root "logs"
$python = (Get-Command python).Source

$tasks = @("FlexTrade-DailyPipeline", "FlexTrade-BessPoller", "FlexTrade-StatesPoller")

if ($Remove) {
    foreach ($t in $tasks) {
        try {
            Unregister-ScheduledTask -TaskName $t -Confirm:$false -ErrorAction Stop
            Write-Host "removed $t"
        } catch { Write-Host "$t not registered" }
    }
    return
}

if (-not (Test-Path $logs)) { New-Item -ItemType Directory $logs | Out-Null }

# wrapper scripts keep the task definitions simple and give us log files
$pipelineCmd = @"
@echo off
cd /d "$root"
set PYTHONIOENCODING=utf-8
"$python" run_pipeline.py >> "$logs\pipeline.log" 2>&1
"@
$pollerCmd = @"
@echo off
cd /d "$root"
set PYTHONIOENCODING=utf-8
"$python" poll_bess.py 120 >> "$logs\poller.log" 2>&1
"@
Set-Content -Path (Join-Path $root "_run_pipeline.cmd") -Value $pipelineCmd -Encoding ascii
Set-Content -Path (Join-Path $root "_run_poller.cmd") -Value $pollerCmd -Encoding ascii

# Invisible launcher: Task Scheduler runs cmd files in a visible console
# window, which flashes on screen every 5 minutes when the poller fires.
# Routing through wscript with window-style 0 makes every run silent.
$vbs = @"
Set sh = CreateObject("WScript.Shell")
sh.Run """" & WScript.Arguments(0) & """", 0, False
"@
Set-Content -Path (Join-Path $root "_run_hidden.vbs") -Value $vbs -Encoding ascii
$wscript = Join-Path $env:SystemRoot "System32\wscript.exe"

# --- daily pipeline, 11:00 ---
$action = New-ScheduledTaskAction -Execute $wscript `
    -Argument "`"$(Join-Path $root '_run_hidden.vbs')`" `"$(Join-Path $root '_run_pipeline.cmd')`""
$trigger = New-ScheduledTaskTrigger -Daily -At 11:00
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
    -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1)
Register-ScheduledTask -TaskName "FlexTrade-DailyPipeline" -Action $action `
    -Trigger $trigger -Settings $settings -Force `
    -Description "FlexTrade daily forecast + dispatch cycle (pre-DAM gate)" | Out-Null
Write-Host "registered FlexTrade-DailyPipeline (daily 11:00)"

# --- BESS sampler ---
# An AtLogOn trigger with auto-restart needs elevation, which we do not
# assume. A repeating time trigger achieves the same coverage without it:
# fire every 5 minutes, each run takes a single sample and exits. That is
# also more robust than one long-lived process — nothing to crash and stay
# dead between checks.
$action2 = New-ScheduledTaskAction -Execute $wscript `
    -Argument "`"$(Join-Path $root '_run_hidden.vbs')`" `"$(Join-Path $root '_run_sample.cmd')`""
$sampleCmd = @"
@echo off
cd /d "$root"
set PYTHONIOENCODING=utf-8
"$python" -c "from ingest import bess; r,m=bess.poll_once(); print(r['ts'], r['soc_pct'], r['discharge_mw'], 'live' if m['live'] else 'CACHED')" >> "$logs\poller.log" 2>&1
"@
Set-Content -Path (Join-Path $root "_run_sample.cmd") -Value $sampleCmd -Encoding ascii

$trigger2 = New-ScheduledTaskTrigger -Once -At (Get-Date).Date `
    -RepetitionInterval (New-TimeSpan -Minutes 5)
$settings2 = New-ScheduledTaskSettingsSet -StartWhenAvailable `
    -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 4)
Register-ScheduledTask -TaskName "FlexTrade-BessPoller" -Action $action2 `
    -Trigger $trigger2 -Settings $settings2 -Force `
    -Description "Samples BRPL Kilokari BESS telemetry (no history endpoint)" | Out-Null
Write-Host "registered FlexTrade-BessPoller (every 5 min)"

# --- 23-state MERIT sampler ---
# Every 15 minutes (matches the market's block length; MERIT itself
# refreshes ~30 s, so this is far below one open browser tab's request
# rate). Each sample = one row per state in state_live + one national
# row — this table is the training data for per-state forecast models,
# so history depth is the whole point.
$statesCmd = @"
@echo off
cd /d "$root"
set PYTHONIOENCODING=utf-8
"$python" -c "from ingest import states; snap,m=states.get_india_snapshot(); n=snap['national']; print(m['asof'], 'live' if m['live'] else 'CACHED', len(snap['states']), 'states,', n.get('demand_met_mw'), 'MW national')" >> "$logs\states.log" 2>&1
"@
Set-Content -Path (Join-Path $root "_run_states.cmd") -Value $statesCmd -Encoding ascii

$action3 = New-ScheduledTaskAction -Execute $wscript `
    -Argument "`"$(Join-Path $root '_run_hidden.vbs')`" `"$(Join-Path $root '_run_states.cmd')`""
$trigger3 = New-ScheduledTaskTrigger -Once -At (Get-Date).Date `
    -RepetitionInterval (New-TimeSpan -Minutes 15)
$settings3 = New-ScheduledTaskSettingsSet -StartWhenAvailable `
    -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
Register-ScheduledTask -TaskName "FlexTrade-StatesPoller" -Action $action3 `
    -Trigger $trigger3 -Settings $settings3 -Force `
    -Description "Samples 23-state + national demand from MERIT (builds per-state forecast training data)" | Out-Null
Write-Host "registered FlexTrade-StatesPoller (every 15 min)"

Write-Host ""
Write-Host "Check status:  Get-ScheduledTask FlexTrade-*"
Write-Host "Run pipeline now:  Start-ScheduledTask FlexTrade-DailyPipeline"
Write-Host "Logs:  $logs"
