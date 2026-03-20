#
# mnemo_setup.ps1 — add mnemo project memory to a Claude Code project
#
# Usage:
#   cd C:\path\to\your\project
#   powershell -ExecutionPolicy Bypass -File C:\path\to\mnemo-code\scripts\mnemo_setup.ps1
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

# Inject mnemo instructions into the project's CLAUDE.md
$MnemoInstructions = Join-Path $MnemoSrc "CLAUDE_MNEMO.md"
$TargetClaude = Join-Path $ProjectDir "CLAUDE.md"

if (Test-Path $TargetClaude) {
    $Existing = Get-Content $TargetClaude -Raw -ErrorAction SilentlyContinue
    if ($Existing -notmatch "mnemo instructions") {
        Add-Content $TargetClaude "`n"
        Get-Content $MnemoInstructions | Add-Content $TargetClaude
        Write-Host "  Appended mnemo instructions to CLAUDE.md"
    } else {
        Write-Host "  CLAUDE.md already contains mnemo instructions — skipped"
    }
} else {
    Copy-Item $MnemoInstructions $TargetClaude
    Write-Host "  Created CLAUDE.md with mnemo instructions"
}

Write-Host ""
Write-Host "Done. mnemo is now active for $ProjectName."
Write-Host ""
Write-Host "What happens next:"
Write-Host "  - Every turn, Claude calls memory_recall automatically"
Write-Host "  - Project knowledge accumulates in .mnemo/"
Write-Host "  - Session handoffs preserve continuity across restarts"
