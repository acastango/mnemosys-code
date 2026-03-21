#!/usr/bin/env bash
#
# mnemo_setup.sh — add mnemo project memory to a Claude Code project
#
# Usage:
#   cd /path/to/your/project
#   bash /path/to/monet-code/scripts/mnemo_setup.sh
#
# Options:
#   --remove    Remove mnemo from the current project
#   --status    Show current mnemo configuration

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MNEMO_SRC="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(pwd)"
PROJECT_NAME="$(basename "$PROJECT_DIR")"

# --- Argument parsing ---
ACTION="add"

for arg in "$@"; do
    case "$arg" in
        --remove)  ACTION="remove" ;;
        --status)  ACTION="status" ;;
        -h|--help)
            echo "Usage: bash mnemo_setup.sh [--remove] [--status]"
            echo ""
            echo "Run from your project directory to add mnemo project memory."
            echo ""
            echo "Options:"
            echo "  --remove   Remove mnemo from this project"
            echo "  --status   Show current configuration"
            echo ""
            echo "Store goes in: <project>/.mnemo/ (auto-added to .gitignore)"
            exit 0
            ;;
    esac
done

# --- Remove ---
if [ "$ACTION" = "remove" ]; then
    echo "Removing mnemo from $PROJECT_NAME..."
    claude mcp remove mnemo 2>/dev/null && echo "Done." || echo "mnemo was not configured."
    exit 0
fi

# --- Status ---
if [ "$ACTION" = "status" ]; then
    echo "Project: $PROJECT_DIR"
    if [ -d "$PROJECT_DIR/.mnemo" ]; then
        NODE_COUNT=$(ls "$PROJECT_DIR/.mnemo/nodes/" 2>/dev/null | wc -l)
        echo "Store:   $PROJECT_DIR/.mnemo ($NODE_COUNT nodes)"
    else
        echo "Store:   not configured"
    fi
    echo "Source:  $MNEMO_SRC"
    exit 0
fi

# --- Add ---
echo "Setting up mnemo for: $PROJECT_NAME"
echo "  Project root: $PROJECT_DIR"
echo "  Mnemo source: $MNEMO_SRC"

STORE_PATH="$PROJECT_DIR/.mnemo"
echo "  Store: $STORE_PATH"
echo ""

# Create store directory
mkdir -p "$STORE_PATH/nodes" "$STORE_PATH/logs"

# Add to .gitignore if not already there
if [ -f "$PROJECT_DIR/.gitignore" ]; then
    if ! grep -q "^\.mnemo" "$PROJECT_DIR/.gitignore" 2>/dev/null; then
        echo "" >> "$PROJECT_DIR/.gitignore"
        echo "# mnemo project memory" >> "$PROJECT_DIR/.gitignore"
        echo ".mnemo/" >> "$PROJECT_DIR/.gitignore"
        echo "  Added .mnemo/ to .gitignore"
    fi
elif [ -d "$PROJECT_DIR/.git" ]; then
    echo "# mnemo project memory" > "$PROJECT_DIR/.gitignore"
    echo ".mnemo/" >> "$PROJECT_DIR/.gitignore"
    echo "  Created .gitignore with .mnemo/"
fi

claude mcp add mnemo \
    -e MNEMO_STORE="$STORE_PATH" \
    -e MNEMO_PROJECT_ROOT="$PROJECT_DIR" \
    -- uv run --with fastmcp --directory "$MNEMO_SRC" fastmcp run mnemo_mcp.py

# Inject mnemo instructions into the project's CLAUDE.md
MNEMO_INSTRUCTIONS="$MNEMO_SRC/CLAUDE_MNEMO.md"
TARGET_CLAUDE="$PROJECT_DIR/CLAUDE.md"

if [ -f "$TARGET_CLAUDE" ]; then
    if ! grep -q "mnemo instructions" "$TARGET_CLAUDE" 2>/dev/null; then
        echo "" >> "$TARGET_CLAUDE"
        cat "$MNEMO_INSTRUCTIONS" >> "$TARGET_CLAUDE"
        echo "  Appended mnemo instructions to CLAUDE.md"
    else
        echo "  CLAUDE.md already contains mnemo instructions — skipped"
    fi
else
    cp "$MNEMO_INSTRUCTIONS" "$TARGET_CLAUDE"
    echo "  Created CLAUDE.md with mnemo instructions"
fi

# Bootstrap tree from codebase
echo "  Scanning codebase to bootstrap tree..."
uv run --with fastmcp --directory "$MNEMO_SRC" python -c "
import sys; sys.path.insert(0, '$MNEMO_SRC')
from pathlib import Path
from mnemo import Store
from mnemo_scan import scan
import os
os.environ.setdefault('MNEMO_STORE', '$STORE_PATH')
os.environ.setdefault('MNEMO_PROJECT_ROOT', '$PROJECT_DIR')
store = Store('$STORE_PATH')
result = scan('.', store, project_root=Path('$PROJECT_DIR'))
print(f'  Scanned {result[\"files_scanned\"]} files, {result[\"claims_created\"]} claims created.')
" 2>&1 || echo "  (scan skipped — run memory_scan(\".\") from Claude to bootstrap later)"

echo ""
echo "Done. mnemo is now active for $PROJECT_NAME."
echo ""
echo "What happens next:"
echo "  - Every turn, Claude calls memory_recall automatically"
echo "  - Project knowledge accumulates in .mnemo/"
echo "  - Session handoffs preserve continuity across restarts"
echo "  - Tree is pre-seeded with codebase structure from AST scan"
