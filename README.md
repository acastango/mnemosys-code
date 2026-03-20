# mnemo

Persistent memory for Claude Code. Mnemo gives Claude a content-addressed knowledge tree that survives context resets — so it stops rediscovering why your code is structured the way it is.

---

## The problem

Every time Claude Code starts a new session, it starts from scratch. It re-reads the same files, re-infers the same patterns, and re-makes the same mistakes. Architecture decisions, conventions, known gotchas, current work state — all of it evaporates at context end.

Mnemo captures this knowledge in a persistent tree and surfaces it automatically, every turn. Claude walks into each session already oriented.

---

## How it works

Mnemo runs as an MCP server alongside Claude Code. On every turn:

1. `memory_recall` fires automatically — surfaces relevant nodes from the project tree
2. Claude reads, writes, and edits through mnemo's filesystem tools — every file operation auto-claims itself in the tree
3. Explicitly learned facts (architecture decisions, gotchas, conventions) get stored with `memory_claim`

Knowledge is content-addressed: nodes are SHA-256 hashed, superseded nodes stay addressable, and the active set is always the current source of truth. Sessions build a chain of reasoning that future instances can reconstruct.

---

## Installation

**Prerequisites:** Python 3.10+, [uv](https://github.com/astral-sh/uv), Claude Code CLI

```bash
git clone https://github.com/acastango/mnemo-code
cd mnemo-code

# Add to a project
cd /path/to/your/project
bash /path/to/mnemo-code/scripts/mnemo_setup.sh

# Or on Windows
powershell -ExecutionPolicy Bypass -File \path\to\mnemo-code\scripts\mnemo_setup.ps1
```

The setup script:
- Creates `.mnemo/` in your project (added to `.gitignore`)
- Registers the MCP server with Claude Code for that project
- Writes mnemo usage instructions into your project's `CLAUDE.md` (creates it if it doesn't exist, appends if it does)

### Manual setup

```bash
claude mcp add mnemo \
    -e MNEMO_STORE="/path/to/your/project/.mnemo" \
    -e MNEMO_PROJECT_ROOT="/path/to/your/project" \
    -- uv run --with fastmcp fastmcp run /path/to/mnemo-code/mnemo_mcp.py
```

### Optional: proactive recall hook

Injects memory context before every Edit/Write — zero API calls, ~10ms.

Add to `~/.claude/settings.json` under `hooks.PreToolUse`:
```json
{
  "matcher": "Edit|Write",
  "hooks": [{ "type": "command", "command": "python /path/to/mnemo-code/mnemo_hook.py" }]
}
```

---

## Usage

Once the MCP server is running, Claude Code uses mnemo automatically. The key rules (in `CLAUDE_MNEMO.md`) tell Claude:

- Call `memory_recall` every turn
- Use `memory_read` / `memory_grep` / `memory_glob` instead of native tools when exploring
- Use `memory_write` / `memory_edit` instead of native tools for all writes
- Call `memory_claim` when something worth preserving is discovered

You can also call tools directly in conversation:

```
# What does the tree know about this project?
memory_soul()

# What changed since the last session?
memory_diff()

# Store a decision
memory_claim(content="We use optimistic locking throughout — see db/lock.py", domain="decisions")

# Find something
memory_search("authentication flow")

# See all active knowledge
memory_active()
```

---

## Optional tools

### Sidecar UI

Live terminal view of mnemo activity — tails the event log and renders it with Rich.

```bash
uv run --with rich python mnemo_sidecar.py
```

### Web dashboard

REST API + WebSocket log tail. Local only, no auth.

```bash
uv run --with fastapi --with uvicorn uvicorn mnemo_web:app --reload
# http://localhost:8000
```

---

## Store layout

```
<project>/.mnemo/
├── nodes/              Content-addressed JSON files
├── active.json         Currently active node addresses
├── chains.json         Chain metadata (ordered reasoning sequences)
├── roots.json          Project knowledge snapshots
├── index/              TF-IDF and embedding indices
└── logs/               Session event logs (JSON Lines)

~/.mnemo/global/        Cross-project store — user preferences, general conventions
```

The `.mnemo/` directory is gitignored by default. Knowledge stays local to your machine.

---

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `MNEMO_STORE` | `<project>/.mnemo` | Project store path |
| `MNEMO_GLOBAL` | `~/.mnemo/global` | Global store path |
| `MNEMO_RETRIEVAL` | `tfidf` | Retrieval backend: `tfidf` or `embedding` |
| `MNEMO_EMBEDDING_PROVIDER` | `auto` | `voyage`, `openai`, or `auto` |
| `MNEMO_SMALL_MODEL` | `claude-haiku-4-5-20251001` | Model for background extraction |
| `MNEMO_PROJECT_ROOT` | git root / CWD | Root for file anchor resolution |
| `ANTHROPIC_API_KEY` | — | Required for extraction sidecar |
| `VOYAGE_API_KEY` | — | Voyage AI embeddings (no SDK required) |
| `OPENAI_API_KEY` | — | OpenAI embeddings |

---

## License

MIT
