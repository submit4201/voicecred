<#
nm.ps1 — helper script for configuring a Windows build environment and
running CMake with the "NMake Makefiles JOM" generator.

Why this file exists
- Avoid using `setx PATH` to add a new folder to PATH. `setx` historically
  truncates values (1024 char limit) and writing machine-level PATH with it
  will fail without elevation ("Access to registry path is denied").

This script:
- Shows safe diagnostics for current PATH values (process/user/machine)
- Provides a helper to append a path to the User OR Machine PATH using
  [Environment]::Get/SetEnvironmentVariable (no 1024-char truncation)
- Does not automatically try to write Machine PATH unless you're elevated.
- Launches Developer tools and runs CMake with JOM in a safe, repeatable way.
#>

function Show-PathInfo {
	Write-Host "--- PATH diagnostics (Process / User / Machine) ---" -ForegroundColor Cyan
	$proc = [Environment]::GetEnvironmentVariable("PATH","Process")
	$user = [Environment]::GetEnvironmentVariable("PATH","User")
	$machine = [Environment]::GetEnvironmentVariable("PATH","Machine")

	$getLen = { param($s) if ($s) { $s.Length } else { 0 } }
	Write-Host "Process PATH length: $(& $getLen $proc)"
	Write-Host "User PATH length:    $(& $getLen $user)"
	Write-Host "Machine PATH length: $(& $getLen $machine)"
	Write-Host "(Search paths containing 'jom' below)" -ForegroundColor Yellow
	$env:PATH -split ';' | Where-Object { $_ -match 'jom' } | ForEach-Object { Write-Host "  -> $_" }
	Write-Host "-------------------------------------------------" -ForegroundColor Cyan
}

function Add-ToPath {
	[CmdletBinding()]
	param(
		[Parameter(Mandatory=$true)][string]$NewPath,
		[ValidateSet('User','Machine')][string]$Scope = 'User'
	)

	if (-not (Test-Path -Path $NewPath)) {
		Write-Host "WARNING: path '$NewPath' does not exist; creating it for convenience." -ForegroundColor Yellow
		New-Item -ItemType Directory -Path $NewPath -Force | Out-Null
	}

	$cur = [Environment]::GetEnvironmentVariable('PATH',$Scope)
	if ($cur -and $cur -match [regex]::Escape($NewPath)) {
		Write-Host "Already present in $Scope PATH: $NewPath" -ForegroundColor Green
		return
	}

	# If we are trying to modify Machine PATH but are not elevated, refuse.
	$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)
	if ($Scope -eq 'Machine' -and -not $isAdmin) {
		Write-Host "ERROR: writing Machine PATH requires elevation. Re-run PowerShell as Administrator to update Machine PATH." -ForegroundColor Red
		return
	}

	# Compose and set without using setx (prevents truncation and avoids registry truncation issues)
	$newVal = if ([string]::IsNullOrEmpty($cur)) { $NewPath } else { "$cur;$NewPath" }
	[Environment]::SetEnvironmentVariable('PATH',$newVal,$Scope)
	Write-Host "Added $NewPath to $Scope PATH. New length: $($newVal.Length)" -ForegroundColor Green
	Write-Host "NOTE: open a new shell to pick up persistent PATH changes." -ForegroundColor Yellow
}

function Repair-IfTruncated {
	Write-Host "Diagnostic: check for obvious truncation issues (setx 1024-char legacy)." -ForegroundColor Cyan
	$user = [Environment]::GetEnvironmentVariable('PATH','User')
	$machine = [Environment]::GetEnvironmentVariable('PATH','Machine')

	# Simple heuristic: setx historically truncated to 1024; detect short length
	if ($user -and $user.Length -lt 1024) { Write-Host "User PATH length: $($user.Length) chars - could be truncated by a prior setx call." -ForegroundColor Yellow }
	if ($machine -and $machine.Length -lt 1024) { Write-Host "Machine PATH length: $($machine.Length) chars - could be truncated by a prior setx call." -ForegroundColor Yellow }

	Write-Host "If your PATH looks corrupted, restore from backups or recompose PATH by reading valid parts from the registry keys:", -ForegroundColor Yellow
	Write-Host " - HKCU: HKCU:\Environment\PATH (User)"; Write-Host " - HKLM: HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment\Path (Machine)"
}

<# ===== Usage examples ===== #>
Write-Host 'nm.ps1 helper — examples & recommended steps' -ForegroundColor Cyan
Show-PathInfo

Write-Host 'EXAMPLE: Add jom to your USER PATH (no elevation required):' -ForegroundColor Green
Write-Host "    Add-ToPath -NewPath 'C:\tools\jom' -Scope User" -ForegroundColor Green

Write-Host 'EXAMPLE: Add jom to MACHINE PATH (requires elevation):' -ForegroundColor Yellow
Write-Host "  * Open an elevated PowerShell (Run as Administrator)" -ForegroundColor Yellow
Write-Host "  * Then call (inside elevated shell): Add-ToPath -NewPath 'C:\tools\jom' -Scope Machine" -ForegroundColor Yellow

# Optionally run developer environment and CMake
Write-Host ''
Write-Host 'To configure and build (recommended to run from Visual Studio dev prompt or run vcvars64 first):' -ForegroundColor Cyan
Write-Host "    & 'C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat'" -ForegroundColor Yellow
Write-Host "    cmake -S . -B build -G 'NMake Makefiles JOM' -T v141" -ForegroundColor Yellow
Write-Host "    cmake --build build --config Release  # or: Push-Location build; jom; Pop-Location" -ForegroundColor Yellow

Write-Host ''
Write-Host 'If you previously used setx and suspect truncation or access errors, run Show-PathInfo and Repair-IfTruncated to inspect and then fix using Add-ToPath.' -ForegroundColor Cyan

# End of script