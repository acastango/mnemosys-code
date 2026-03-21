"""
mnemo_cli.py - Command-line interface for mnemo

Commands:
  mnemo serve    Start the MCP server (stdio transport, used by Claude Code)
  mnemo install  Register mnemo as a global Claude Code MCP server + hook
  mnemo init     Initialize mnemo for the current project
  mnemo hook     Run the proactive recall hook (called by Claude Code hooks)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def cmd_serve(_args) -> None:
    """Start the MCP server on stdio."""
    from mnemo_mcp import mcp
    mcp.run(transport="stdio")


def cmd_hook(_args) -> None:
    """Run the proactive recall hook (reads hook event from stdin)."""
    from mnemo_hook import main as hook_main
    hook_main()


def _install_hook() -> bool:
    """
    Add PreToolUse and PostToolUse hooks to ~/.claude/settings.json.
    Merges safely — does not overwrite existing hooks.
    Returns True if settings were changed.
    """
    settings_path = Path.home() / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
    except (json.JSONDecodeError, OSError):
        settings = {}

    hooks = settings.setdefault("hooks", {})

    pre_hook  = {"matcher": "Read|Edit|Write", "hooks": [{"type": "command", "command": "mnemo hook"}]}
    post_hook = {"matcher": "Edit|Write",      "hooks": [{"type": "command", "command": "mnemo hook"}]}

    def _already_present(hook_list: list, command: str) -> bool:
        return any(
            any(h.get("command") == command for h in entry.get("hooks", []))
            for entry in hook_list
        )

    changed = False

    pre_list = hooks.setdefault("PreToolUse", [])
    if not _already_present(pre_list, "mnemo hook"):
        pre_list.append(pre_hook)
        changed = True

    post_list = hooks.setdefault("PostToolUse", [])
    if not _already_present(post_list, "mnemo hook"):
        post_list.append(post_hook)
        changed = True

    if changed:
        tmp = settings_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        tmp.replace(settings_path)

    return changed


def cmd_install(_args) -> None:
    """Register mnemo as a global Claude Code MCP server and install hooks."""
    # MCP server
    result = subprocess.run(
        ["claude", "mcp", "add", "--scope", "user", "mnemo", "--", "mnemo", "serve"],
    )
    if result.returncode == 0:
        print("mnemo MCP server registered globally.")
    else:
        print("MCP registration failed. Try manually:")
        print("  claude mcp add --scope user mnemo -- mnemo serve")
        sys.exit(1)

    # Proactive recall hook
    changed = _install_hook()
    if changed:
        print("mnemo hook installed in ~/.claude/settings.json.")
    else:
        print("mnemo hook already configured - skipped.")

    print()
    print("It will auto-detect .mnemo/ in any project.")
    print("To initialize a project:  cd <project> && mnemo init")


def cmd_init(_args) -> None:
    """Initialize mnemo for the current project."""
    project_dir = Path.cwd()
    project_name = project_dir.name
    store_path = project_dir / ".mnemo"

    print(f"Initializing mnemo for: {project_name}")

    # Create store dirs
    (store_path / "nodes").mkdir(parents=True, exist_ok=True)
    (store_path / "logs").mkdir(parents=True, exist_ok=True)

    # .gitignore
    gitignore = project_dir / ".gitignore"
    if gitignore.exists():
        content = gitignore.read_text(encoding="utf-8")
        if ".mnemo" not in content:
            with open(gitignore, "a", encoding="utf-8") as f:
                f.write("\n# mnemo project memory\n.mnemo/\n")
            print("  Added .mnemo/ to .gitignore")
    elif (project_dir / ".git").is_dir():
        gitignore.write_text("# mnemo project memory\n.mnemo/\n", encoding="utf-8")
        print("  Created .gitignore with .mnemo/")

    # Write .monet (monet-code instructions for Claude)
    claude_mnemo = Path(__file__).parent / "CLAUDE_MNEMO.md"
    dot_monet = project_dir / ".monet"
    if claude_mnemo.exists():
        instructions = claude_mnemo.read_text(encoding="utf-8")
        dot_monet.write_text(instructions, encoding="utf-8")
        print("  Created .monet with monet-code instructions")

    # Add @.monet import to CLAUDE.md
    target_claude = project_dir / "CLAUDE.md"
    if target_claude.exists():
        existing = target_claude.read_text(encoding="utf-8")
        if "@.monet" not in existing:
            with open(target_claude, "a", encoding="utf-8") as f:
                f.write("\n@.monet\n")
            print("  Added @.monet to CLAUDE.md")
        else:
            print("  CLAUDE.md already imports .monet - skipped")
    else:
        target_claude.write_text("@.monet\n", encoding="utf-8")
        print("  Created CLAUDE.md with @.monet import")

    # Bootstrap tree from codebase
    print("  Scanning codebase to bootstrap tree...")
    try:
        from mnemo import Store
        from mnemo_scan import scan
        store = Store(str(store_path))
        result = scan(".", store, project_root=project_dir)
        print(f"  Scanned {result['files_scanned']} files, {result['claims_created']} claims created.")
    except Exception as e:
        print(f"  (scan skipped: {e})")
        print("  Run memory_scan('.') from Claude to bootstrap later.")

    print()
    print(f"Done. Open {project_name} in Claude Code.")
    print("mnemo recalls context automatically every turn.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mnemo",
        description="Content-addressed project memory for Claude Code",
    )
    sub = parser.add_subparsers(dest="command", metavar="command")
    sub.required = True

    sub.add_parser("serve",   help="Start the MCP server (used internally by Claude Code)")
    sub.add_parser("install", help="Register mnemo globally with Claude Code")
    sub.add_parser("init",    help="Initialize mnemo for the current project")
    sub.add_parser("hook",    help="Run the proactive recall hook (stdin → additionalContext)")

    args = parser.parse_args()
    {"serve": cmd_serve, "install": cmd_install, "init": cmd_init, "hook": cmd_hook}[args.command](args)


if __name__ == "__main__":
    main()
