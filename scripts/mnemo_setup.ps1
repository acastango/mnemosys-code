#
# mnemo_setup.ps1 - add mnemo project memory to a Claude Code project
#
# Usage:
#   cd C:\path\to\your\project
#   powershell -ExecutionPolicy Bypass -File C:\path\to\monet-code\scripts\mnemo_setup.ps1
#
# Options:
#   -Remove    Remove mnemo from the current project
#   -Status    Show current mnemo configuration

param(
    [switch]$Remove,
    [switch]$Status,
    [switch]$Help
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$MnemoSrc = Split-Path -Parent $ScriptDir
$ProjectDir = (Get-Location).Path
$ProjectName = Split-Path -Leaf $ProjectDir

if ($Help) {
    Write-Host "Usage: mnemo_setup.ps1 [-Remove] [-Status]"
    Write-Host ""
    Write-Host "Run from your project directory to add mnemo project memory."
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Remove   Remove mnemo from this project"
    Write-Host "  -Status   Show current configuration"
    Write-Host ""
    Write-Host "Store goes in: <project>\.mnemo\ (auto-added to .gitignore)"
    exit 0
}

# --- Remove ---
if ($Remove) {
    Write-Host "Removing mnemo from $ProjectName..."
    claude mcp remove mnemo 2>$null
    Write-Host "Done."
    exit 0
}

# --- Status ---
if ($Status) {
    Write-Host "Project: $ProjectDir"
    $StorePath = Join-Path $ProjectDir ".mnemo"
    if (Test-Path (Join-Path $StorePath "nodes")) {
        $NodeCount = (Get-ChildItem (Join-Path $StorePath "nodes") -File -ErrorAction SilentlyContinue | Measure-Object).Count
        Write-Host "Store:   $StorePath ($NodeCount nodes)"
    } else {
        Write-Host "Store:   not configured"
    }
    Write-Host "Source:  $MnemoSrc"
    exit 0
}

# --- Add ---
Write-Host "Setting up mnemo for: $ProjectName"
Write-Host "  Project root: $ProjectDir"
Write-Host "  Mnemo source: $MnemoSrc"

$StorePath = Join-Path $ProjectDir ".mnemo"
Write-Host "  Store: $StorePath"
Write-Host ""

# Create store directory
New-Item -ItemType Directory -Force -Path (Join-Path $StorePath "nodes") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $StorePath "logs") | Out-Null

# Add to .gitignore if not already there
$GitIgnore = Join-Path $ProjectDir ".gitignore"
$GitDir = Join-Path $ProjectDir ".git"
if (Test-Path $GitIgnore) {
    $Content = Get-Content $GitIgnore -Raw -ErrorAction SilentlyContinue
    if ($Content -notmatch '\.mnemo') {
        Add-Content $GitIgnore "`n# mnemo project memory`n.mnemo/"
        Write-Host "  Added .mnemo/ to .gitignore"
    }
} elseif (Test-Path $GitDir) {
    Set-Content $GitIgnore "# mnemo project memory`n.mnemo/"
    Write-Host "  Created .gitignore with .mnemo/"
}

$McpScript = Join-Path $MnemoSrc "mnemo_mcp.py"

claude mcp add mnemo `
    -e "MNEMO_STORE=$StorePath" `
    -e "MNEMO_PROJECT_ROOT=$ProjectDir" `
    -- uv run --with fastmcp --directory "$MnemoSrc" fastmcp run mnemo_mcp.py

# Write .monet (monet-code instructions for Claude)
$MnemoInstructions = Join-Path $MnemoSrc "CLAUDE_MNEMO.md"
$DotMonet = Join-Path $ProjectDir ".monet"
Copy-Item $MnemoInstructions $DotMonet
Write-Host "  Created .monet with monet-code instructions"

# Add @.monet import to CLAUDE.md
$TargetClaude = Join-Path $ProjectDir "CLAUDE.md"
if (Test-Path $TargetClaude) {
    $Existing = Get-Content $TargetClaude -Raw -ErrorAction SilentlyContinue
    if ($Existing -notmatch '@\.monet') {
        Add-Content $TargetClaude "`n@.monet"
        Write-Host "  Added @.monet to CLAUDE.md"
    } else {
        Write-Host "  CLAUDE.md already imports .monet - skipped"
    }
} else {
    Set-Content $TargetClaude "@.monet"
    Write-Host "  Created CLAUDE.md with @.monet import"
}

# Bootstrap tree from codebase
Write-Host "  Scanning codebase to bootstrap tree..."
$TempPy = Join-Path $env:TEMP "mnemo_bootstrap.py"
$PyLines = [System.Collections.Generic.List[string]]::new()
$PyLines.Add('import sys')
$PyLines.Add('sys.path.insert(0, r"' + $MnemoSrc + '")')
$PyLines.Add('from pathlib import Path')
$PyLines.Add('from mnemo import Store')
$PyLines.Add('from mnemo_scan import scan')
$PyLines.Add('store = Store(r"' + $StorePath + '")')
$PyLines.Add('result = scan(".", store, project_root=Path(r"' + $ProjectDir + '"))')
$PyLines.Add('n_files = result["files_scanned"]')
$PyLines.Add('n_claims = result["claims_created"]')
$PyLines.Add('print("  Scanned " + str(n_files) + " files, " + str(n_claims) + " claims created.")')
[System.IO.File]::WriteAllLines($TempPy, $PyLines)
try {
    $env:MNEMO_STORE = $StorePath
    $env:MNEMO_PROJECT_ROOT = $ProjectDir
    uv run --with fastmcp --directory "$MnemoSrc" python $TempPy 2>&1
} catch {
    Write-Host "  (scan skipped - run memory_scan('.') from Claude to bootstrap later)"
} finally {
    Remove-Item $TempPy -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "Done. mnemo is now active for $ProjectName."
Write-Host ""
Write-Host "What happens next:"
Write-Host "  - Every turn, Claude calls memory_recall automatically"
Write-Host "  - Project knowledge accumulates in .mnemo/"
Write-Host "  - Session handoffs preserve continuity across restarts"
Write-Host "  - Tree is pre-seeded with codebase structure from AST scan"
