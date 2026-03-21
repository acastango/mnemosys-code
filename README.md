# monet-code

Persistent memory for Claude Code. Gives Claude a content-addressed knowledge tree that survives context resets — so it stops rediscovering why your code is structured the way it is.

---

## The problem

Every time Claude Code starts a new session, it starts from scratch. It re-reads the same files, re-infers the same patterns, and re-makes the same mistakes. Architecture decisions, conventions, known gotchas, current work state — all of it evaporates at context end.

monet-code captures this knowledge in a persistent tree and surfaces it automatically, every turn. Claude walks into each session already oriented.

---

## Installation

**Prerequisites:** Python 3.10+, Claude Code CLI

```bash
pip install monet-code

# Register with Claude Code (once, globally)
mnemo install

# Initialize a project
cd /path/to/your/project
mnemo init
```

`mnemo init`:
- Creates `.mnemo/` in your project (gitignored automatically)
- Writes mnemo instructions into `CLAUDE.md`
- Runs a static AST scan to bootstrap the tree from your existing codebase

The MCP server auto-detects the store by walking up from the working directory — no per-project configuration needed after `mnemo init`.

---

## How it works

monet-code runs as an MCP server alongside Claude Code. On every turn:

1. `memory_recall` fires automatically — surfaces relevant nodes from the project tree
2. Claude reads, writes, and edits through mnemo's filesystem tools — every file operation auto-claims itself in the tree
3. Explicitly learned facts get stored with `memory_claim`
4. Sessions compress into handoff nodes — the next instance picks up where the last one stopped

Knowledge is content-addressed: nodes are SHA-256 hashed, superseded nodes stay addressable, and the active set is always the current source of truth.

---

## Usage

Once running, monet-code works automatically. Key tools:

```
# What does the tree know about this project?
memory_soul()

# What changed since the last session?
memory_diff()

# Store a decision
memory_claim("We use optimistic locking — see db/lock.py", domain="decisions")

# Find something
memory_search("authentication flow")

# Bootstrap tree from existing codebase
memory_scan(".")

# Run a pipeline
memory_run("session-orient", params={"input": "collision detection"})
memory_run("file-context",   params={"input": "src/physics.c"})
```

---

## Pipelines

Pipelines are composable, reusable sequences of memory operations — stored as nodes in the tree, addressable and supersedable like any other knowledge.

```
memory_pipelines()           # list available pipelines
memory_run("name", params)   # run a pipeline
memory_pipeline("name", steps, description)  # define a custom pipeline
```

Built-in pipelines:

| Pipeline | Steps | Use |
|----------|-------|-----|
| `session-orient` | recall → traverse → filter → compress | Orient a new session to a topic |
| `file-context` | spatial → traverse → dedupe | Surface all tree knowledge for a file |
| `issue-cluster` | active(issues) → dedupe → compress | Cluster known bugs into a summary |
| `drift-check` | active → filter(anchored) → dedupe | Find potentially stale claims |

Custom pipelines can be defined and stored in the tree:

```python
memory_pipeline("my-workflow", [
    {"op": "recall",   "query": "{input}"},
    {"op": "traverse", "depth": 2},
    {"op": "filter",   "domain": "architecture"},
    {"op": "compress", "label": "context: {input}"}
])
memory_run("my-workflow", params={"input": "auth system"})
```

---

## Optional tools

### Proactive recall hook

Injects memory context before every Edit/Write — zero API calls, ~10ms.

Add to `~/.claude/settings.json` under `hooks.PreToolUse`:

```json
{
  "matcher": "Edit|Write",
  "hooks": [{ "type": "command", "command": "python -m mnemo_hook" }]
}
```

### Sidecar UI

Live terminal view of monet-code activity — tails the event log and renders with Rich.

```bash
pip install monet-code[sidecar]
python -m mnemo_sidecar
```

### Web dashboard

REST API + WebSocket log tail. Local only, no auth.

```bash
pip install monet-code[web]
uvicorn mnemo_web:app --reload
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

`.mnemo/` is gitignored by default. Knowledge stays local to your machine.

---

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `MNEMO_STORE` | auto-detected | Override project store path |
| `MNEMO_GLOBAL` | `~/.mnemo/global` | Global store path |
| `MNEMO_RETRIEVAL` | `tfidf` | Retrieval backend: `tfidf` or `embedding` |
| `MNEMO_EMBEDDING_PROVIDER` | `auto` | `voyage`, `openai`, or `auto` |
| `MNEMO_SMALL_MODEL` | `claude-haiku-4-5-20251001` | Model for background operations |
| `MNEMO_PROJECT_ROOT` | git root / CWD | Root for file anchor resolution |
| `ANTHROPIC_API_KEY` | — | Required for Haiku-based operations |
| `VOYAGE_API_KEY` | — | Voyage AI embeddings |
| `OPENAI_API_KEY` | — | OpenAI embeddings |

---

## License

MIT
