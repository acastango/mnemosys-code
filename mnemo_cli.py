"""
mnemo_cli.py - Command-line interface for mnemo

Commands:
  mnemo serve    Start the MCP server (stdio transport, used by Claude Code)
  mnemo install  Register mnemo as a global Claude Code MCP server
  mnemo init     Initialize mnemo for the current project
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def cmd_serve(_args) -> None:
    """Start the MCP server on stdio."""
    from mnemo_mcp import mcp
    mcp.run(transport="stdio")


def cmd_install(_args) -> None:
    """Register mnemo as a global Claude Code MCP server."""
    result = subprocess.run(
        ["claude", "mcp", "add", "--scope", "user", "mnemo", "--", "mnemo", "serve"],
    )
    if result.returncode == 0:
        print("mnemo registered globally.")
        print("It will auto-detect .mnemo/ in any project.")
        print()
        print("To initialize a project:  cd <project> && mnemo init")
    else:
        print("Registration failed. Try manually:")
        print("  claude mcp add --scope user mnemo -- mnemo serve")
        sys.exit(1)


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

    # Inject mnemo instructions into CLAUDE.md
    claude_mnemo = Path(__file__).parent / "CLAUDE_MNEMO.md"
    target_claude = project_dir / "CLAUDE.md"
    if claude_mnemo.exists():
        instructions = claude_mnemo.read_text(encoding="utf-8")
        if target_claude.exists():
            existing = target_claude.read_text(encoding="utf-8")
            if "mnemo instructions" not in existing:
                with open(target_claude, "a", encoding="utf-8") as f:
                    f.write("\n" + instructions)
                print("  Appended mnemo instructions to CLAUDE.md")
            else:
                print("  CLAUDE.md already has mnemo instructions - skipped")
        else:
            target_claude.write_text(instructions, encoding="utf-8")
            print("  Created CLAUDE.md with mnemo instructions")

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

    sub.add_parser("serve", help="Start the MCP server (used internally by Claude Code)")
    sub.add_parser("install", help="Register mnemo globally with Claude Code")
    sub.add_parser("init", help="Initialize mnemo for the current project")

    args = parser.parse_args()
    {"serve": cmd_serve, "install": cmd_install, "init": cmd_init}[args.command](args)


if __name__ == "__main__":
    main()
