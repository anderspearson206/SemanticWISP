# Laptop side of the Hyak pipeline. Run from anywhere:  cluster\hyak.ps1 <command> [args]
#
#   push                      copy the code to Hyak (everything except data, results, logs, .venv, papers)
#   submit [K=V ..] <script.py> [args]
#                             push, then queue the script as a Slurm job (see cluster/submit.sh for K=V overrides)
#   status                    squeue for your jobs
#   pull                      copy results/ and logs/ from Hyak back here (never overwrites a newer local file)
#   data                      one-time: upload conv_2B.zip and unzip it into the same relative path as here
#   run <command>             run a shell command in the project directory on Hyak
#   ssh                       open a shell in the project directory on Hyak
#
# Every command is ONE ssh connection, so one password + Duo prompt (Hyak does not accept ssh keys).
# Settings: cluster/config.env.

$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
$Tar = "$env:SystemRoot\System32\tar.exe"
$Ssh = "$env:SystemRoot\System32\OpenSSH\ssh.exe"

$cfg = @{}
Get-Content "$PSScriptRoot\config.env" | Where-Object { $_ -match '^\s*([A-Z_]+)=(.*)$' } |
    ForEach-Object { $cfg[$Matches[1]] = $Matches[2].Trim() }
$RemoteHost = $cfg.HYAK_HOST
$Dir = $cfg.HYAK_DIR

$Command = if ($args.Count) { $args[0] } else { 'help' }
$Rest = @(); if ($args.Count -gt 1) { $Rest = @($args[1..($args.Count - 1)]) }

if ($Command -eq 'help') {
    Get-Content $PSCommandPath | Select-Object -Skip 1 -First 10 | ForEach-Object { $_ -replace '^# ?', '' }
    exit 0
}
if ($Dir -match 'CHANGEME') { throw "Set HYAK_DIR in cluster\config.env first (your /gscratch/<group> path)." }
if ($Dir -match "[\s'`"]") { throw "HYAK_DIR must not contain spaces or quotes." }

# Runs one pipeline line through cmd.exe: PowerShell 5.1 corrupts binary data in its own pipes.
function Invoke-Cmd([string]$Line) {
    if ($env:HYAK_DRYRUN) { Write-Host $Line; return }     # $env:HYAK_DRYRUN = 1 prints the command instead
    $f = Join-Path $env:TEMP "hyak_$PID.cmd"
    Set-Content -Path $f -Value "@echo off`r`n$Line" -Encoding ascii
    try { cmd.exe /c "`"$f`""; if ($LASTEXITCODE -ne 0) { throw "failed (exit code $LASTEXITCODE)" } }
    finally { Remove-Item $f -ErrorAction SilentlyContinue }
}

function Remote-Words([string[]]$Words) {
    foreach ($w in $Words) {
        if ($w -notmatch '^[A-Za-z0-9_./:=,+@-]+$') { throw "Unsupported character in '$w' (allowed: letters, digits, _ . / : = , + @ -)." }
    }
    $Words -join ' '
}

# Code only. Line endings are fixed on arrival, in case a Windows tool saved a shell script with CRLF.
$Excludes = '.venv', 'data', 'results', 'logs', 'papers', '.git', '.claude' | ForEach-Object { "--exclude=./$_" }
$Excludes += '--exclude=__pycache__', '--exclude=.ipynb_checkpoints'
$Pack = "`"$Tar`" -C `"$Repo`" $($Excludes -join ' ') -cf - ."
$Unpack = "mkdir -p '$Dir' && tar -xf - -C '$Dir' && cd '$Dir' && sed -i 's/\r`$//' cluster/*.sh cluster/*.sbatch cluster/config.env"

switch ($Command) {
    'push' {
        Invoke-Cmd "$Pack | `"$Ssh`" $RemoteHost `"$Unpack && echo pushed to ${RemoteHost}:$Dir`""
    }
    'submit' {
        if (-not $Rest.Count) { throw 'usage: hyak.ps1 submit [KEY=VALUE ...] <script.py> [script args]' }
        Invoke-Cmd "$Pack | `"$Ssh`" $RemoteHost `"$Unpack && bash cluster/submit.sh $(Remote-Words $Rest)`""
    }
    'status' {
        & $Ssh $RemoteHost 'squeue --me --format=''%.10i %.9P %.24j %.2t %.10M %.10l %R'''
    }
    'pull' {
        Invoke-Cmd "`"$Ssh`" $RemoteHost `"cd '$Dir' && mkdir -p results logs && tar -cf - results logs`" | `"$Tar`" -xf - -C `"$Repo`" --keep-newer-files"
        Write-Host "pulled results/ and logs/ into $Repo"
    }
    'data' {
        $rel = 'data/roomb_piropo/Room B'
        $zip = Join-Path $Repo "$rel\conv_2B.zip"
        if (-not (Test-Path $zip)) { throw "not found: $zip" }
        # the zip holds conv_2B/..., and the local layout is 'Room B/conv_2B/conv_2B/...'
        $remote = "mkdir -p '$Dir/$rel/conv_2B' && cd '$Dir/$rel' && cat > conv_2B.zip && unzip -q -n conv_2B.zip -d conv_2B && find conv_2B -name '*.jpg' | wc -l"
        Write-Host 'uploading conv_2B.zip (1.3 GB, no progress bar); prints the JPEG count when done (expect 9332)'
        Invoke-Cmd "`"$Ssh`" $RemoteHost `"$remote`" < `"$zip`""
    }
    'run' {
        if (-not $Rest.Count) { throw 'usage: hyak.ps1 run <command>' }
        & $Ssh -t $RemoteHost "cd '$Dir' && $($Rest -join ' ')"
    }
    'ssh' {
        & $Ssh -t $RemoteHost "cd '$Dir' && exec bash -l"
    }
    default { throw "unknown command '$Command' (try: cluster\hyak.ps1 help)" }
}
