# mnemo usage guide

This document tells you **when to reach for each tool** depending on your role.
No need to reason about it — follow the triggers below and mnemo works.

Two audiences:
- **[CLAUDE-CODE]** — Claude Code CLI agent doing implementation work
- **[DESKTOP]** — Claude.ai desktop conversation with mnemo MCP connected

---

## [CLAUDE-CODE] — Implementation agent

### Your mandatory behaviors

| When | What | Why |
|------|------|-----|
| **Every single turn** | `memory_recall(message)` | Surfaces what the project knows. Non-negotiable. |
| **You learn something architectural** | `memory_claim(content, domain)` | Preserves it for future sessions. If in doubt, claim it. |
| **You find a stale or wrong node** | `memory_update(old_addr, content)` | Fixes the tree. Never leave known-wrong nodes. |
| **You confirm something still holds** | `memory_reinforce(addr)` | Bumps freshness without creating noise. |
| **memory_status shows pressure** | `memory_compress(addrs)` | Keeps the tree healthy. Do it when nudged. |

### Decision tree — should I claim this?

```
Did I just discover something that a future instance would want to know?
├── Yes → memory_claim()
│         domain guide:
│           architecture   why the code is structured this way
│           decisions      why we chose X over Y
│           patterns       naming rules, file layout, test conventions
│           issues         bugs, gotchas, fragile areas, workarounds
│           dependencies   library versions, constraints, upgrade notes
│           tasks          current work state, blockers, what's in progress
│           history        what was tried, what failed, dead ends
│           context        environment, deployment, team info
│
└── No (it's ephemeral, obvious from code, or already in the tree)
    → skip
```

### When to use each tool

**Reading and exploration:**

| Instead of | Use | When |
|---|---|---|
| `Read(file)` | `memory_read(file_path)` | Learning — want inline tree annotations + stale anchor warnings |
| `Glob(pattern)` | `memory_glob(pattern)` | Learning — want per-file tree coverage |
| `Grep(pattern)` | `memory_grep(pattern, intent)` | Learning — want results annotated with tree context |
| "how does X work?" | `memory_explore(question)` | Full reasoning trace: recall → locate → search → gaps |
| "what should I change?" | `memory_plan(task)` | Planning context: constraints, risks, blockers, affected files |

Use native Read/Grep/Glob only when fetching a specific known value — not discovering it.

**Writing and editing — always use mnemo:**

| Instead of | Use |
|---|---|
| `Write(file, content)` | `memory_write(path, content)` |
| `Edit(file, old, new)` | `memory_edit(path, old_string, new_string)` |

Every write auto-claims the change. Native writes produce no trail.

**Lookup:**

| Trigger | Tool |
|---|---|
| "What do we know about X?" | `memory_search(query)` |
| Have a node address, want details | `memory_query(addr)` |
| Want to know where a node came from | `memory_provenance(addr)` |

**Multi-session goals:**

| Trigger | Tool |
|---|---|
| Starting work spanning multiple sessions | `memory_arc(action="create", goal=...)` |
| Resuming — what was I doing? | `memory_arc(action="list")` |
| Done with a goal | `memory_arc(action="complete", arc_id=...)` |

**Chains:**

| Trigger | Tool |
|---|---|
| Review a reasoning chain | `memory_cat(chain_id)` |
| List chains in the project | `memory_chains()` |
| Who said what in a chain | `memory_blame(chain_id)` |
| What's anchored to a file | `memory_spatial(file)` |

**Housekeeping:**

| Trigger | Tool |
|---|---|
| End of a work cycle | `memory_session_compress()` |
| Told memory is under pressure | `memory_compress(addrs=[...])` |
| Want full project picture | `memory_soul()` |
| Suspect node is stale | `memory_verify(addr)` |
| Passive pattern review | `memory_infer()` |

### Anti-patterns — don't do these

- **Don't skip recall.** Even on trivial turns. Recall is how context gets delivered.
- **Don't store code.** The codebase is the codebase. Store *why*, not *what*.
- **Don't store ephemeral state.** "Currently on step 3" is not a claim — use tasks.
- **Don't leave stale nodes.** If you know something is wrong, call `memory_update`.
- **Don't over-compress.** Compress when the tool nudges you, not preemptively.

### Typical session flow

```
Turn 1:  memory_recall("what the user said")
         → orientation + handoff from prior session

Turns 2–N:
         memory_recall("what the user said")     ← every turn
         ... do work ...
         memory_claim(...)                        ← when you learn something
         memory_update(...)                       ← when you find stale info

End of cycle:
         memory_session_compress()               ← when nudged or wrapping up
```

---

## [DESKTOP] — Claude.ai desktop conversation

You have mnemo MCP connected. You are typically having a focused conversation
about a project — not doing implementation. Your job is to use the tree as a
knowledge base and to capture decisions from the conversation.

### Your mandatory behaviors

| When | What | Why |
|------|------|-----|
| **Start of any project conversation** | `memory_recall(topic)` | Orient on what's already known. |
| **You make a design decision** | `memory_claim(decision, domain="decisions")` | Preserves it for Claude Code sessions. |
| **User explains something important** | `memory_claim(content, domain=...)` | Don't let it get lost at context end. |
| **You find a node is wrong** | `memory_update(addr, content)` | Fix it now, while you know. |

### When to use each tool

| Trigger | Tool |
|---|---|
| "What does this project do?" | `memory_soul()` |
| "What's the current state?" | `memory_status()` |
| "What changed recently?" | `memory_diff()` |
| "Tell me about X" | `memory_search(query)` |
| "What's in the tree?" | `memory_active()` |
| Reviewing a specific node | `memory_query(addr)` |
| Cross-project lookup | `memory_recall(query, project="other-project")` |
| List registered projects | `memory_projects()` |

### Typical desktop session flow

```
Start:   memory_recall("topic of this conversation")
         → surfaces relevant project knowledge

During:  ... discuss, reason, decide ...
         memory_claim(...)    ← when a decision is reached
         memory_search(...)   ← when you need to look something up
         memory_update(...)   ← when you find something stale

End:     nothing required — claims are already committed
```

### Capturing decisions — what's worth claiming

Claim it if it answers one of:
- "Why is the code structured this way?"
- "Why did we choose X over Y?"
- "What should I watch out for in this area?"
- "What are the current priorities?"
- "What was tried and failed?"

Skip it if it's derivable from reading the code, or it's just conversation context.

---

## Quick reference — all tools by trigger

| I want to... | Tool |
|---|---|
| Orient on the current turn | `memory_recall(message)` |
| Store a fact | `memory_claim(content, domain)` |
| Fix a stale fact | `memory_update(old_addr, content)` |
| Confirm a fact still holds | `memory_reinforce(addr)` |
| Connect two facts | `memory_link(from_addr, to_addr, rel)` |
| Search the tree | `memory_search(query)` |
| Look up a node | `memory_query(addr)` |
| Trace a node's history | `memory_provenance(addr)` |
| Check a claim against codebase | `memory_verify(addr)` |
| Read a file with tree context | `memory_read(file_path)` |
| Write a file + claim | `memory_write(path, content)` |
| Edit a file + claim | `memory_edit(path, old, new)` |
| Explore a question | `memory_explore(question)` |
| Search code with tree context | `memory_grep(pattern, intent)` |
| Plan a task | `memory_plan(task)` |
| File listing with coverage | `memory_glob(pattern)` |
| Codebase structure overview | `memory_map(path)` |
| Coverage report | `memory_coverage()` |
| Compress a cluster | `memory_compress(addrs)` |
| End-of-cycle compression | `memory_session_compress()` |
| Full project document | `memory_soul()` |
| Health check | `memory_status()` |
| What changed since last root | `memory_diff()` |
| Multi-session goal tracking | `memory_arc(action, ...)` |
| Infer patterns from logs | `memory_infer()` |
| Health review | `memory_dream()` |
| Render a chain as narrative | `memory_cat(chain_id)` |
| List chains | `memory_chains(...)` |
| Review chain attribution | `memory_blame(chain_id)` |
| Query by file location | `memory_spatial(file, start_line, end_line)` |
| Full tree history | `memory_log(topic)` |
| Stale-dependency scan | `memory_rebase(node_addr)` |
| List projects | `memory_projects()` |
| Get this guide | `memory_help(context=...)` |
