# mnemo — Architecture

## What it is

mnemo is a content-addressed memory system for AI coding sessions. It runs as an MCP server alongside Claude Code and gives the model persistent, structured access to project knowledge that survives across sessions.

The problem it solves: every AI session starts cold. Architecture decisions, naming conventions, known gotchas, half-finished work, failed approaches — none of it is available unless it's in the files. mnemo captures that layer of knowledge in a queryable tree that's automatically surfaced on every turn.

---

## The core idea: chains, not notes

The fundamental unit is the **chain** — an ordered sequence of reasoning nodes. Not a flat bag of facts, but a line of thought: how a decision developed, what was tried, what was learned, where things stand.

When you recall memory, you get back the 2–3 most relevant reasoning paths, not 8 disconnected facts. The structure carries information that flat retrieval loses.

A **node** is a single fact or observation — immutable, content-addressed. A chain is an ordered list of node addresses with metadata (domain, summary, creation time). Nodes can belong to multiple chains.

---

## Store architecture

Three layers, each with distinct scope and lifetime:

```
~/.mnemo/                    Global store
  User preferences, cross-project conventions, tool workflows.
  Follows the user, not any project.

<project>/.mnemo/            Project store  ← primary
  Architecture decisions, patterns, issues, dependencies —
  everything specific to this codebase. Co-located like .git/.
  Gitignored by default (private to the developer).

.mnemo/sessions/<id>/        Session store (ephemeral)
  Current reasoning chains, working notes, scratch thoughts.
  Promoted to the project store at session end, or discarded.
  Linear scan — no index needed (<50 nodes typical).
```

Store discovery follows the same convention as git: walk up from CWD, stop at the first `.mnemo/`. The closest ancestor wins.

---

## Data model

### Nodes

Every node is a JSON file named by a 12-character hex prefix of `SHA-256(type + content + sorted_inputs)`. Content-addressed means:
- No duplication — the same fact written twice produces one node
- Immutable — modifying a node's content produces a new address
- Provenance is structural — you can trace any summary back to its sources

Four node types:

| Type | Description |
|------|-------------|
| `leaf` | A raw claim. Ground truth. Never modified. |
| `compress` | Summary of N inputs. Lossy content, lossless provenance. |
| `supersede` | Replaces a prior claim. Old node stays addressable. |
| `root` | Snapshot of the entire active set — a project knowledge hash. |

**Meta is not part of the address.** Recall counts, timestamps, priority boosts, and chain membership are stored in node metadata and can be updated without changing the address.

### Active set

`active.json` is the source of truth for what's live. It's a set of node addresses. Superseded nodes are removed from the active set but remain in `nodes/` — they're still addressable and traversable via provenance.

### Chains

`chains.json` maps stable chain IDs (`ch_<12hex>`) to mutable metadata: member list, head, tail, domain, summary, agent attribution, status. Chain IDs are randomly assigned at creation and never change. Chains are organizational metadata — not content-addressed, not immutable.

```
chains.json
  ch_a7f3c2d1 → {
    summary: "auth refactor decision thread",
    domain: "decisions",
    members: ["abc123", "def456", "789ghi"],
    head: "abc123",
    tail: "789ghi",
    status: "active",
    last_extended: 1710000000
  }
```

---

## Retrieval pipeline

`memory_recall` runs on every turn. It takes the user's message, extracts signals, scores the active node set, and returns the top chains and standalone nodes within a ~25,000 char budget.

```
message
  │
  ▼
signal extraction          keywords, code symbols, domain hints, session affinity
  │
  ▼
TF-IDF scoring             bag-of-words similarity across active nodes
  │
  ▼
domain boosts              +weight for nodes matching detected domain
  │
  ▼
priority boost             flat additive boost from node.meta["priority"]
  │
  ▼
temporal signals           recently created/reinforced nodes rank higher
  │
  ▼
utility feedback           nodes that led to good outcomes rank higher
  │
  ▼
confidence decay           stale nodes decay slightly over time
  │
  ▼
chain assembly             group top nodes into their chains; return full chains
  │
  ▼
recall budget              top 3 chains (~20k chars) + standalone nodes (~3k)
```

The chain assembly step is what makes v2 different from a flat retrieval system. A node's score promotes its entire chain — the model gets a coherent reasoning path, not an isolated fact.

An optional embedding backend (Voyage AI or OpenAI) can replace TF-IDF for semantic similarity. Controlled by `MNEMO_RETRIEVAL=embedding`.

---

## Extraction sidecar

After each turn, a background process (`mnemo_extract.py`) proposes facts worth storing. It runs a small model (Haiku by default) against the conversation context and suggests `memory_claim` or `memory_update` calls. The main model decides whether to accept them.

This is the "subconscious" layer — passive knowledge capture that doesn't interrupt the main flow.

---

## Verification anchors

Claims can be pinned to specific code via anchors:

```python
memory_claim(
    content="auth token validation is in validate_token() in auth.py",
    anchors=[
        {"type": "grep", "pattern": "def validate_token", "path": "auth.py"},
        {"type": "file", "path": "auth.py"},
    ]
)
```

`memory_verify` checks anchors against the filesystem with zero LLM calls. If the code has moved or changed, the anchor fails and the claim is flagged stale.

Content-hash anchors (from `memory_map`) bind to the actual bytes of a function body, not line numbers — they survive line number shifts and flag genuine code changes.

---

## MCP integration

mnemo runs as an MCP server via [fastmcp](https://github.com/jlowin/fastmcp):

```bash
claude mcp add mnemo -- uv run --with fastmcp fastmcp run mnemo_mcp.py
```

Claude Code calls the tools natively. The key behavioral contract: `memory_recall` is called on every turn before the model responds. Everything else is deliberate — the model decides when to claim, update, or explore.

The server maintains session state across turns (turn count, recently recalled addresses, session chain IDs) in `session_state.json`. This state persists across brief restarts but expires after 2 hours of inactivity.

---

## File map

```
Core engine
  mnemo.py              Store/Node, content addressing, compress/supersede/reroot
  mnemo_chains.py       Chain CRUD, scoring, chain-first retrieval assembly
  mnemo_associate.py    Subconscious layer — TF-IDF, signal extraction, recall pipeline
  mnemo_retrieval.py    Retrieval backends — TfIdfBackend, EmbeddingBackend (optional)
  mnemo_session.py      Session store lifecycle — create, promote, archive, gc

Filesystem integration
  mnemo_fs.py           Read/write/edit/glob with auto-claim and stale anchor detection
  mnemo_anchor.py       Content-hash anchors — generation, file index, lookup
  mnemo_verify.py       Anchor validation against the codebase (zero LLM calls)
  mnemo_map.py          Cartographer — walk a file/dir and generate anchor coverage
  mnemo_coverage.py     Coverage report — what's annotated vs unexplored

Tree-aware tools
  mnemo_explore.py      Recall + grep + gap/tension reasoning trace
  mnemo_grep.py         Pattern search annotated with per-file tree knowledge
  mnemo_plan.py         Planning context — constraints, risks, affected files, blockers
  mnemo_read.py         File reading with inline tree annotations at line/section level

Knowledge management
  mnemo_extract.py      Haiku sidecar — background fact extraction proposals
  mnemo_infer.py        Passive pattern inference from session logs (5 analysis layers)
  mnemo_handoff.py      Session continuity — handoff nodes, first-recall orientation
  mnemo_arc.py          Work arcs — multi-session goal tracking and momentum
  mnemo_init.py         Project store initialization and registry

Observability
  mnemo_log.py          Structured event emitter (JSON Lines)
  mnemo_hook.py         Proactive recall hook — injects context before Edit/Write
  mnemo_sidecar.py      Live sidecar UI — tails memory.log in a second terminal
  mnemo_web.py          FastAPI dashboard — REST API + WebSocket log tail

Entry point
  mnemo_mcp.py          MCP server — exposes all tools to Claude Code via fastmcp
```

---

## Store layout

```
<project>/
└── .mnemo/
    ├── nodes/                  Content-addressed node files (<addr>.json)
    ├── active.json             Current active set (non-superseded addresses)
    ├── chains.json             Chain metadata (ch_<id> → members/summary/domain)
    ├── roots.json              Linear chain of root hashes (project knowledge snapshots)
    ├── index/
    │   ├── tfidf.json          Persisted TF-IDF index
    │   └── by_file.json        File → anchor mapping for fast lookup
    ├── session_state.json      Session cycle state (turns, recalled addrs)
    ├── sessions/               Ephemeral session stores
    │   └── <session-id>/
    │       ├── nodes/
    │       └── active.json
    └── logs/
        └── <session>.jsonl     Structured event log

~/.mnemo/
├── nodes/                      Global store (cross-project knowledge)
├── active.json
├── config.json                 User preferences
└── projects.json               Registry of known project stores
```

---

## Key design decisions

**Content addressing over line numbers.** Code moves. Line numbers drift. Anchoring to content hashes (SHA-256 of function bodies) means annotations survive refactors and flag genuine semantic changes.

**Chains as the retrieval unit.** Returning a coherent reasoning path is more useful than returning N highest-scoring isolated facts. The path shows how understanding developed — context that a flat bag of facts loses entirely.

**Supersession, not mutation.** When a fact changes, the old node is never modified. A new `supersede` node replaces it in the active set. The old node stays addressable. The full history of how knowledge evolved is always recoverable.

**Project store co-located with the project.** `.mnemo/` lives next to `.git/`. It's gitignored by default — the memory tree is private to the developer. Sharing is opt-in via `memory_export`.

**No automatic compression.** The system nudges toward compression when the active set grows large, but never compresses automatically. Compression is lossy — the model decides when the trade-off is worth it.
