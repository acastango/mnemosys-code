# monet-code 

Persistent, content-addressed memory for Claude Code. Every decision, convention, and discovered fact gets stored in a Merkle-structured tree that survives context resets — so Claude stops rediscovering why your code is structured the way it is.

Named after Moneta, Roman goddess of memory. Runs entirely local.

---

## The problem

Every time Claude Code starts a new session, it starts from scratch. It re-reads the same files, re-infers the same patterns, re-makes the same mistakes. Architecture decisions, conventions, known bugs, current work state — all of it evaporates at context end.

monet-code captures this knowledge in a persistent tree and surfaces it automatically, every turn. Claude walks into each session already oriented.

---

## Installation

**Prerequisites:** Python 3.10+, Claude Code CLI

```bash
pip install monet-code

# Register with Claude Code once, globally
mnemo install

# Initialize a project
cd /path/to/your/project
mnemo init
```

`mnemo init`:
- Creates `.mnemo/` in your project (gitignored automatically)
- Creates `.monet` with monet-code instructions for Claude
- Adds `@.monet` to `CLAUDE.md` (one clean import line)
- Runs a static AST scan to bootstrap the tree from your existing codebase — new sessions start with structural knowledge already in place

The MCP server auto-detects the store by walking up from the working directory. No per-project configuration needed after `mnemo init`.

```
mnemo install   Register MCP server globally + install recall hooks
mnemo init      Initialize a project (run once per repo)
mnemo serve     Start the MCP server (called internally by Claude Code)
mnemo hook      Run the proactive recall hook (called by Claude Code hooks)
```

---

## Architecture

### Content-addressed Merkle tree

monet-code is not a vector database or a summary buffer. It's a filesystem for memory.

Every node is addressed by `SHA-256(type + content + inputs)` — immutable, deduplicated, and verifiable. Nothing is ever deleted. Superseded claims stay addressable. The active set is the current source of truth, but the full provenance chain is always intact.

```
leaf        Raw claim — ground truth, never modified
compress    Summary of N inputs — lossy content, lossless provenance
supersede   Replaces a prior claim — old stays addressable
root        Snapshot of the entire active set at a point in time
```

### Memory layers

```
MESSAGE IN
   │
   ├─► subconscious   memory_recall fires every turn — ambient surfacing
   │
   └─► conscious      memory_claim, memory_update, memory_link — deliberate storage
```

### Chains

Chains are ordered sequences of nodes — reasoning trails. A chain records not just *what* was concluded but *how*: what was recalled, what was discovered, what was compressed. Sessions build chains automatically. Chains can be promoted, stashed, rendered as narratives, and used as the basis for reusable pipelines.

### Pipelines

Pipelines are composable sequences of memory operations, stored as first-class nodes in the tree. The runner is pure Python — no LLM in the loop. Built-ins:

| Pipeline | Steps | Use |
|----------|-------|-----|
| `session-orient` | recall → traverse → filter → compress | Orient a new session |
| `file-context` | spatial → traverse → dedupe | Surface knowledge for a file |
| `issue-cluster` | active(issues) → dedupe → compress | Cluster known bugs |
| `drift-check` | active → filter(anchored) → dedupe | Find stale claims |

When a methodology works, `memory_learn` extracts it from a successful chain and stores it as a reusable pipeline — so the approach that solved the problem becomes available for next time.

### Vectors

Vectors are compositions of multiple pipelines — an abstraction above the pipeline abstraction. Each component pipeline contributes a dimension; the vector merges them into a single result.

```
memory_vector("full-orient", [
    {"pipeline": "session-orient", "params": {"input": "{input}"}},
    {"pipeline": "issue-cluster"},
    {"pipeline": "drift-check"},
], merge="ranked")
```

Merge strategies:

| Merge | Behavior |
|-------|----------|
| `dedupe` | Union of all components, deduplicated (default) |
| `union` | Alias for dedupe |
| `intersect` | Only nodes present in every component |
| `ranked` | Round-robin interleave — one from each component in turn |
| `sequential` | Chain: output of component i feeds into component i+1 |

Vectors can also have a `post` field — pipeline steps applied to the merged result after combining. `memory_run` dispatches on type, so pipelines and vectors share the same invocation interface.

---

## How it works

On every turn:

1. `memory_recall` fires — surfaces relevant nodes via TF-IDF + domain boosts + link traversal
2. The proactive hook injects tree context before every file edit — zero API calls, ~10ms
3. File operations go through monet-code's filesystem tools — reads, writes, and edits auto-claim themselves in the tree with content-hash anchors
4. Discoveries get stored explicitly with `memory_claim`
5. At session end, work compresses into a handoff node — the next instance picks up where the last stopped

---

## Key tools

```
# Orientation
memory_soul()                          # full project knowledge document
memory_diff()                          # what changed since the last root
memory_status()                        # active node count, domain breakdown

# Knowledge
memory_claim("...", domain="decisions")
memory_update(addr, "new content")     # supersede a stale claim
memory_search("authentication flow")
memory_graph(addr, depth=2)            # traverse the link graph

# Codebase
memory_scan(".")                       # bootstrap tree from AST (no LLM)
memory_explore(".")                    # reasoning trace: recall → locate → gaps
memory_read("src/file.c")             # tree-annotated file reading
memory_coverage(".")                   # how much of the codebase has tree coverage

# Pipelines & Vectors
memory_pipelines()
memory_vectors()
memory_run("session-orient", params={"input": "collision system"})
memory_run("full-orient", params={"input": "collision system"})  # vector
memory_pipeline("name", steps)         # define and store a custom pipeline
memory_vector("name", components, merge="ranked")  # compose pipelines
memory_learn(chain_id, "name")         # extract a reusable pipeline from a successful chain

# Chains
memory_chains()                        # list chains
memory_cat(chain_id)                   # render chain as narrative
memory_arc("goal description")         # track a multi-session work arc
```

---

## Extras

### Sidecar UI

Live terminal view of monet-code activity.

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
├── nodes/              Content-addressed JSON files (addr = 12-char SHA-256)
├── active.json         Currently active node addresses
├── chains.json         Chain metadata — stable ch_<12hex> IDs
├── roots.json          Project knowledge snapshots
├── index/              TF-IDF and embedding indices
├── session_state.json  Session cycle state
└── logs/               Session event logs (JSON Lines)

~/.mnemo/global/        Cross-project store — preferences, general conventions
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
| `MNEMO_COMPRESS_INTERVAL` | `15` | Turns between auto-compression |
| `MNEMO_SMALL_MODEL` | `claude-haiku-4-5-20251001` | Model for background operations |
| `MNEMO_PROJECT_ROOT` | git root / CWD | Root for file anchor resolution |
| `ANTHROPIC_API_KEY` | — | Required for Haiku-based operations |
| `VOYAGE_API_KEY` | — | Voyage AI embeddings |
| `OPENAI_API_KEY` | — | OpenAI embeddings |

---

## License

MIT
