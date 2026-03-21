# mnemo — content-addressed project memory

## Overview

A Merkle-structured memory tree for software development projects. Runs as an MCP server for native Claude Code integration.

The store lives at `<project>/.mnemo/` by default (`MNEMO_STORE` env to override).

---

## Problem this solves

When an AI instance starts fresh on a big project, it has to rediscover:
- **Why is the code structured this way?** (architecture decisions get lost)
- **What patterns should I follow?** (conventions are inferred inconsistently)
- **What's the current state of work?** (half-finished work, blockers)
- **What breaks / known gotchas?** (workarounds get "fixed", fragile areas hit blind)
- **What was already tried?** (dead-end paths get re-explored)

mnemo captures this knowledge in a content-addressed tree that persists across sessions.

---

## File map

```
mnemo.py           Core engine — Node/Store, compress, supersede, reroot, project knowledge
mnemo_associate.py Subconscious layer — signal extraction, TF-IDF scoring, domain boosts, link traversal, priority scoring, adaptive depth
mnemo_retrieval.py Retrieval backends — RetrievalBackend protocol, TfIdfBackend, EmbeddingBackend, compute_coverage_score()
mnemo_explore.py   Tree-aware codebase exploration — recall + grep + gap/tension detection reasoning trace
mnemo_grep.py      Tree-aware pattern search — intent-driven, annotates results with tree knowledge per file
mnemo_plan.py      Tree-aware planning context — categorizes nodes by planning role, extracts affected files/blockers
mnemo_read.py      Tree-annotated file reading — inline annotations from tree at line/section level
mnemo_infer.py     Passive pattern inference — 5-layer analysis of session logs (co-occurrence, recall, corrections, sequences, workflow)
mnemo_handoff.py   Session continuity — structured handoff nodes at session end, first-recall orientation priming
mnemo_arc.py       Work arcs — multi-session goal tracking with trajectory, auto-update at session compress
mnemo_fs.py        Filesystem integration — read/write/edit/grep/glob with bidirectional tree linkage, auto-claim, stale anchor detection
mnemo_anchor.py    Verification anchor index — content-hash tracking, file index, drift detection
mnemo_chains.py    Chain operations — create, extend, list, promote, stash/pop, render
mnemo_session.py   Session store — ephemeral per-session chain workspace, promote to project
mnemo_coverage.py  Coverage scoring — how well the tree describes the codebase
mnemo_map.py       Codebase cartography — section detection, file walk, structure mapping
mnemo_graph.py     Explicit link-graph traversal — BFS from a node, renders subgraph; used by memory_graph tool
mnemo_scan.py      Static codebase scanner — AST docstring extraction → tree claims, no LLM, idempotent via scan_index.json
mnemo_pipeline.py  Composable memory pipelines and vectors — pipelines (type="pipeline") are linear step sequences; vectors (type="vector") compose N pipelines with merge strategies: dedupe/union/intersect/ranked/sequential; sequential threads output between components; built-ins: session-orient, file-context, issue-cluster, drift-check
mnemo_cli.py       CLI entry point — mnemo install (global MCP), mnemo init (per-project), mnemo serve (stdio server)
mnemo_verify.py    Verification anchors — pins claims to code via file/grep/dependency checks
mnemo_log.py       Structured event emitter — writes JSON Lines to memory.log
mnemo_sidecar.py   Live sidecar UI — tails memory.log, renders in terminal via Rich
mnemo_hook.py      Proactive recall hook — injects memory context before Edit/Write via PreToolUse
mnemo_web.py       FastAPI observability dashboard — REST API + WebSocket log tail
MNEMO_GUIDE.md     Usage guide — when to call what. Served by memory_help.
mnemo_mcp.py       MCP server — exposes tools to Claude via fastmcp, session tracking, recall metadata
```

---

## Running

### MCP mode (active)
```bash
# Already configured. Claude Code loads this automatically.
# To re-add:
claude mcp add mnemo -- uv run --with fastmcp fastmcp run mnemo_mcp.py
```

### Proactive recall hook
```bash
# Add to ~/.claude/settings.json under hooks.PreToolUse:
{
  "matcher": "Edit|Write",
  "hooks": [{ "type": "command", "command": "python PATH/TO/mnemo_hook.py" }]
}
# Zero API calls — fast text matching (~10ms), injects relevant memory as additionalContext
```

### Sidecar UI (open a second terminal)
```bash
uv run --with rich python mnemo_sidecar.py

# Options
python mnemo_sidecar.py --tail 50          # show last 50 events
python mnemo_sidecar.py --log .mnemo/logs/current.log
```

### Web dashboard
```bash
uv run --with fastapi --with uvicorn uvicorn mnemo_web:app --reload
# Endpoints: /api/status, /api/nodes, /api/nodes/{addr}, /api/provenance/{addr},
#            /api/graph, /api/roots, /api/logs, /ws/logs, /
# Note: no authentication — intended for local use only
```

---

## Architecture

### Memory layers

```
MESSAGE IN
   │
   ├─► subconscious (ambient)
   │     memory_recall   — surfaces relevant project knowledge before thinking
   │     auto-extraction — after each turn, detects facts to store/update
   │     source: "subconscious" in node meta
   │
   └─► conscious (deliberate)
         memory_claim      — explicitly store project knowledge
         memory_update     — supersede a stale claim
         memory_reinforce  — confirm something still holds
         memory_link       — connect related nodes
         memory_verify     — check anchored claims against codebase
         source: "conscious" / "live" in node meta
```

### Node types

| Type | Description |
|------|-------------|
| `leaf` | Raw claim — ground truth, never modified |
| `compress` | Summary of N inputs — lossy content, lossless provenance |
| `supersede` | Replaces a prior claim — old stays addressable |
| `root` | Project knowledge hash — snapshot of entire active set |

### Store layout

```
<project>/.mnemo/
├── nodes/              Content-addressed JSON files (addr = 12-char SHA256)
├── active.json         Set of currently active addresses
├── chains.json         Chain metadata — stable ch_<12hex> IDs
├── roots.json          Linear chain of root hashes
├── index/
│   ├── tfidf.json      Persisted TF-IDF index
│   └── embeddings.json Persisted embedding vectors (optional)
├── session_state.json  Session cycle state (turns, recalled addrs)
└── logs/               Session event logs (JSON Lines)
    └── current.txt     Pointer to active session log

~/.mnemo/global/        Cross-project global store (user preferences, general conventions)
```

---

## Event log format

Each line is a JSON object:

```json
{
  "ts": "2026-03-10T06:12:01.234Z",
  "layer": "subconscious | conscious | system",
  "event": "recall | claim | update | reinforce | link | compress | verify | search | ...",
  "summary": "human-readable one-liner",
  "addresses": ["abc12345"],
  "domain": "architecture",
  "detail": {}
}
```

Layers:
- `subconscious` — ambient ops: recall, auto-extracted claims
- `conscious`    — deliberate tool calls
- `system`       — status checks, compression, reroot, dream sessions

---

## MCP tools

| Tool | Layer | Description |
|------|-------|-------------|
| `memory_recall` | subconscious | **Called every turn.** Adaptive associative recall. Tracks `recall_count`/`last_recalled`. |
| `memory_write` | conscious | Write a file — auto-claim change in tree |
| `memory_edit` | conscious | Edit a file — stale anchor warnings, auto-claim |
| `memory_glob` | conscious | Glob with per-file tree coverage annotation |
| `memory_read` | conscious | Tree-annotated file reading — inline annotations at line/section level; stale anchor detection |
| `memory_grep` | conscious | Tree-aware pattern search — intent-driven, annotates results with tree knowledge |
| `memory_claim` | conscious | Store new fact(s) — supports batch, project/global scope, priority, anchors |
| `memory_update` | conscious | Supersede existing claim — inherits domain/confidence/priority |
| `memory_reinforce` | conscious | Bump freshness, no new node |
| `memory_link` | conscious | Create directional relationship between nodes |
| `memory_verify` | conscious | Verify anchored claims against codebase (file/grep/dependency checks) |
| `memory_search` | conscious | TF-IDF search across project + global stores |
| `memory_query` | conscious | Look up node by address (prefix ok) |
| `memory_graph` | conscious | Traverse link graph from a node — renders subgraph N hops deep |
| `memory_provenance` | conscious | Trace derivation chain to leaves |
| `memory_gap` | conscious | Flag something you don't know — creates a gap node for future resolution |
| `memory_ask` | conscious | Flag a pending decision or question |
| `memory_checkpoint` | conscious | Save a resume point mid-session |
| `memory_compress` | conscious/system | Compress N nodes into summary. Reports coverage score. |
| `memory_session_compress` | system | Compress current work cycle, reset counter |
| `memory_reroot` | system | Recompute project knowledge hash from active set |
| `memory_status` | system | Active count, health signals, domain breakdown |
| `memory_diff` | system | Delta since last root |
| `memory_active` | system | Full active context dump |
| `memory_soul` | system | Generate project knowledge document |
| `memory_explore` | conscious | Tree-aware codebase exploration — reasoning trace: recall → locate → search → gaps → tensions |
| `memory_plan` | conscious | Tree-aware planning context — architecture, constraints, risks, state, affected files, blockers |
| `memory_map` | conscious | Codebase cartography — structure overview with tree coverage |
| `memory_scan` | conscious | Static AST scan — extracts docstrings/signatures into tree claims, no LLM, idempotent |
| `memory_pipeline` | conscious | Define a reusable pipeline as a node — stored, addressed, supersedable |
| `memory_vector` | conscious | Define a vector: N pipelines composed with a merge strategy (dedupe/union/intersect/ranked/sequential) + optional post steps |
| `memory_run` | conscious | Run a named pipeline or vector — dispatches on node type; built-ins + stored |
| `memory_pipelines` | conscious | List all available pipelines — built-ins and stored |
| `memory_vectors` | conscious | List all stored vectors |
| `memory_learn` | conscious | Extract a reusable pipeline from a successful chain — learns methodology from what worked |
| `memory_survey` | conscious | Rate session memory quality — recall precision/coverage, compression loss, orientation speed, notes, requests. Accumulates in tree as domain="feedback" |
| `memory_coverage` | conscious | Coverage report — how well the tree describes the codebase |
| `memory_infer` | system | Passive pattern inference from session logs — 5 layers: co-occurrence, recall, corrections, sequences, workflow |
| `memory_arc` | conscious | Work arcs — create, update, complete, pause, list, detect multi-session goals |
| `memory_prune_candidates` | system | Find supersession candidates via Jaccard similarity |
| `memory_cat` | conscious | Render a chain or node as a coherent narrative |
| `memory_chains` | conscious | List and filter chains in the project and session stores |
| `memory_promote` | conscious | Promote a preliminary chain from session store to project store |
| `memory_session_status` | conscious | Show what's in the session store — preliminary chains, promoted work |
| `memory_stash` | conscious | Shelve a chain without losing it |
| `memory_stash_pop` | conscious | Restore a stashed chain |
| `memory_blame` | conscious | Attribution decomposition — who said what, when, why (chain/file/node) |
| `memory_log` | conscious | Chronological tree history including superseded nodes |
| `memory_rebase` | conscious | Find nodes that may have become stale when a key node changed |
| `memory_spatial` | conscious | Spatial retrieval — query by file and line range |
| `memory_init` | conscious | Initialize a project memory store |
| `memory_projects` | conscious | List all registered project stores |
| `memory_switch` | conscious | Switch the active store |
| `memory_import` | conscious | Import a snapshot into the store |
| `memory_help` | conscious | Return usage guide for specified context: `claude-code`, `desktop`, `quick`, `all` |

---

## Retrieval pipeline

```
signal extraction → density classification → backend scoring → domain boosts
→ priority boost → temporal signals → utility feedback → confidence decay
→ link traversal → adaptive depth cap
```

### Priority scoring

Nodes carry an optional `priority` float in meta (default 0). Applied as a flat additive boost, independent of context.

| Priority | Use case |
|----------|----------|
| 0 | Normal facts (default) |
| 0.5 | Moderate importance — critical gotchas, invariants |
| 1.0 | High importance — working agreements, design principles, user preferences |

Set via `priority` param on `memory_claim`. Inherited during supersession.

### Verification anchors

Claims can carry anchors that pin them to the codebase:

```json
{"type": "file", "path": "mnemo.py"}
{"type": "grep", "pattern": "def supersede", "path": "mnemo.py"}
{"type": "dependency", "name": "fastmcp"}
```

`memory_verify` checks anchors against the filesystem. Zero LLM calls. Set via `anchors` param on `memory_claim`.

---

## Key configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `MNEMO_STORE` | `<project>/.mnemo` | Project store directory |
| `MNEMO_GLOBAL` | `~/.mnemo/global` | Cross-project global store |
| `MNEMO_RETRIEVAL` | `tfidf` | Retrieval backend (`tfidf` or `embedding`) |
| `MNEMO_EMBEDDING_PROVIDER` | `auto` | `voyage`, `openai`, or `auto` |
| `MNEMO_EMBEDDING_MODEL` | varies | Override embedding model name |
| `MNEMO_SIDECAR_CAP` | `15000` | Sidecar context budget (chars) |
| `MNEMO_COMPRESS_INTERVAL` | `15` | Turns between compression nudges |
| `MNEMO_SMALL_MODEL` | `claude-haiku-4-5-20251001` | Model for background operations |
| `MNEMO_PROJECT_ROOT` | git root / CWD | Project root for verification anchors |
| `VOYAGE_API_KEY` | — | Voyage AI embedding provider (no SDK needed) |
| `OPENAI_API_KEY` | — | OpenAI embedding provider (uses openai SDK) |
| `ANTHROPIC_API_KEY` | — | Required for Haiku-based operations (memory_infer) |

---

## Domains

`architecture` · `decisions` · `patterns` · `tasks` · `issues` · `dependencies` · `history` · `context`

| Domain | What goes here |
|--------|---------------|
| `architecture` | System structure, module responsibilities, data flow, API design |
| `decisions` | Why choices were made, trade-offs considered, rejected alternatives |
| `patterns` | Coding conventions, naming rules, file structure, test patterns |
| `tasks` | Current work state, TODOs, blockers, priorities, what's in progress |
| `issues` | Known bugs, gotchas, fragile areas, workarounds, regressions |
| `dependencies` | Libraries, versions, constraints, upgrade notes, compatibility |
| `history` | What was tried, what changed, failed approaches, evolution notes |
| `context` | General project context, environment setup, deployment, team info |

---

## Development notes

- `mnemo_log.emit()` never raises — swallows errors silently
- `mnemo_log.configure(path)` should be called when `STORE_PATH` is set from args (not env)
- The `active.json` file is the source of truth for what's "live"
- Addresses are 12-char hex prefixes of SHA-256(type + content + sorted inputs)
- Meta is **not** part of the address — safe to mutate via `store.put()`
- Prefix matching is supported in most tools (min 6 chars recommended)
- Priority, links, recall stats, reinforcement counts are all meta — mutating them doesn't change the node's address
- `supersede()` inherits domain, confidence, source, priority, links, utility counters, and timestamps from the old node by default
