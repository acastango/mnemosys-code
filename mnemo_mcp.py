"""
mnemo MCP server — content-addressed project memory for Claude Code

Exposes mnemo operations as MCP tools that Claude can call natively.
Run with: claude mcp add mnemo -- uv run --with fastmcp fastmcp run mnemo_mcp.py

Or for development:
    uv run --with fastmcp fastmcp dev mnemo_mcp.py

Tools:
    memory_claim     — commit a project fact (or batch of facts) to the memory tree
    memory_update    — supersede an existing claim with new info
    memory_reinforce — mark an existing claim as still current
    memory_link      — create a directional relationship between two nodes
    memory_query     — look up a node by address (prefix ok)
    memory_verify    — verify anchored claims against the actual codebase
    memory_search    — search active memory by keyword
    memory_provenance — trace a claim back to its origins
    memory_compress  — compress a set of nodes into a summary
    memory_session_compress — compress current work cycle into a summary
    memory_status    — active set size, pressure level, last root
    memory_diff      — what changed since last root
    memory_soul      — generate the current project knowledge document
    memory_reroot    — recompute the root from active nodes
    memory_infer     — passive pattern inference from session logs
    memory_help      — return usage guide for claude-code / quick / all
    memory_write     — write a file + auto-claim the change
    memory_edit      — edit a file + stale warnings + auto-claim
    memory_glob      — glob with tree coverage annotation per file

Session continuity:
    mnemo_handoff.py generates structured handoff nodes at session compress time
    and provides first-recall orientation priming on session start (turns 1-2).
"""

import json
import os
import time
from fastmcp import FastMCP
from mnemo import (
    Store, Node, GENESIS,
    supersede, compress, reroot,
    build_active_context,
    propose_supersessions,
    generate_soul_doc,
    discover_store,
)
from mnemo_associate import associate, retrieve_relevant
from mnemo_log import emit, configure as log_configure
from mnemo_explore import explore as _explore
from mnemo_grep import grep as _grep
from mnemo_plan import plan as _plan
from mnemo_read import read as _read
from mnemo_infer import infer as _infer
from mnemo_handoff import generate_handoff, build_orientation
from mnemo_arc import (
    create_arc, update_arc, complete_arc, pause_arc,
    find_active_arcs, match_session_to_arcs, detect_arc_candidates,
)
from mnemo_anchor import compute_content_hash, update_file_index
from mnemo_map import map_path as _map_path
from mnemo_coverage import coverage as _coverage, format_report as _format_coverage
from mnemo_session import (
    load_or_create_session, get_session_store,
    promote_chain, promote_nodes, promote_all_preliminary,
    archive_session, session_summary, gc_sessions,
    list_preliminary_chains,
)
from mnemo_fs import (
    get_project_root as _get_project_root,
    normalize_path as _normalize_path,
    nodes_for_file as _nodes_for_file,
    check_stale_anchors as _check_stale_anchors,
    auto_claim as _auto_claim,
    fs_write as _fs_write,
    fs_edit as _fs_edit,
    fs_glob as _fs_glob,
    format_glob_with_coverage as _format_glob_with_coverage,
)

# --- Project registry ---
REGISTRY_DIR = os.path.expanduser("~/.mnemo")
REGISTRY_PATH = os.path.join(REGISTRY_DIR, "projects.json")


def _load_registry() -> dict[str, str]:
    """Load project name -> store path mapping. Canonical format: {name: path}."""
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if isinstance(v, str)}
        if isinstance(data, list):
            # Recover from legacy list format
            result = {}
            for entry in data:
                if isinstance(entry, dict) and "path" in entry:
                    import os as _os
                    name = entry.get("name") or _os.path.basename(_os.path.dirname(entry["path"]))
                    result[name] = entry["path"]
            return result
        return {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_registry(registry: dict[str, str]) -> None:
    """Persist registry. Atomic write."""
    os.makedirs(REGISTRY_DIR, exist_ok=True)
    tmp = REGISTRY_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    os.replace(tmp, REGISTRY_PATH)


def _register_project(name: str, store_path: str) -> None:
    """Add/update a project in the registry."""
    registry = _load_registry()
    registry[name] = os.path.abspath(store_path)
    _save_registry(registry)


def _resolve_project(name: str) -> Store | None:
    """Look up a project by name, return a Store instance."""
    registry = _load_registry()
    path = registry.get(name)
    if path and os.path.isdir(path):
        return Store(path)
    return None


# --- Configuration ---
def _detect_store_path() -> tuple[str, bool]:
    """
    Auto-detect project store using v2 discover_store() semantics.

    Returns (store_path_str, is_v2).
    v2: found .mnemo/ by walking up from CWD.
    v1-compat: fell back to MNEMO_STORE env or ~/mnemo.

    MNEMO_PROJECT_DIR overrides the CWD used for discovery — useful when
    the MCP server is launched from a different working directory (e.g.
    from Claude Desktop, which doesn't support the cwd config field).
    """
    project_dir_override = os.environ.get("MNEMO_PROJECT_DIR", "")
    store_path, is_v2 = discover_store(cwd=project_dir_override or None)
    store_path_str = str(store_path)
    if is_v2:
        # Auto-register v2 project store by directory name
        from pathlib import Path as _Path
        project_dir = _Path(store_path_str).parent
        dir_name = project_dir.name
        _register_project(dir_name, store_path_str)
    return store_path_str, is_v2

_store_path_result = _detect_store_path()
STORE_PATH = _store_path_result[0]
STORE_IS_V2 = _store_path_result[1]
GLOBAL_PATH = os.environ.get("MNEMO_GLOBAL", os.path.expanduser("~/.mnemo/global"))
COMPRESS_INTERVAL = int(os.environ.get("MNEMO_COMPRESS_INTERVAL", "15"))
store = Store(STORE_PATH)
global_store = Store(GLOBAL_PATH)
log_configure(STORE_PATH)
emit("status", "system",
     f"mnemo MCP session started — project: {STORE_PATH}, global: {GLOBAL_PATH}")

# --- Session tracking ---
_session_turns = 0
_session_addrs: list[str] = []  # node addresses created/modified this cycle
_recalled_recent: list[set[str]] = []  # addresses surfaced on recent turns (last 5)
_file_visits: dict[str, int] = {}  # basename -> read count this session

# --- Session store (v2 only) ---
_session_id: str = ""
_session_store: Store | None = None  # ephemeral working memory

_recently_extended_chain_ids: set[str] = set()  # chains extended this session (continuity boost)

def _init_session_store():
    """Initialize the session store if running against a v2 project store."""
    global _session_id, _session_store
    if STORE_IS_V2:
        from pathlib import Path as _Path
        state_path = _Path(STORE_PATH) / "session_state.json"
        _session_id, _session_store = load_or_create_session(store, state_path)

_init_session_store()

def _is_recently_recalled(addr: str) -> bool:
    """Check if an address was surfaced in any of the last 5 recall turns."""
    return any(addr in turn_set for turn_set in _recalled_recent)


def _get_store(project: str = "") -> Store:
    """Get a store by project name, or the active store if empty."""
    if not project:
        return store
    resolved = _resolve_project(project)
    if not resolved:
        raise ValueError(f"Unknown project '{project}'. Use memory_projects() to list registered projects.")
    return resolved


def _save_session_state():
    """Persist session cycle state to disk so it survives MCP restarts."""
    state = {
        "session_turns": _session_turns,
        "session_addrs": _session_addrs,
        "recalled_recent": [sorted(s) for s in _recalled_recent],
        "file_visits": _file_visits,
        "session_id": _session_id,
        "recently_extended_chain_ids": sorted(_recently_extended_chain_ids),
        "saved_at": time.time(),
    }
    path = os.path.join(STORE_PATH, "session_state.json")
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp_path, path)  # atomic on both POSIX and Windows
    except Exception:
        # Clean up temp file if rename failed
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _load_session_state():
    """Restore session state from disk. Discards if >2 hours stale."""
    global _session_turns, _session_addrs, _recalled_recent, _file_visits
    global _recently_extended_chain_ids
    path = os.path.join(STORE_PATH, "session_state.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        # Discard if stale (>2 hours)
        if time.time() - state.get("saved_at", 0) > 7200:
            return
        _session_turns = state.get("session_turns", 0)
        _session_addrs = state.get("session_addrs", [])
        _recalled_recent = [set(s) for s in state.get("recalled_recent", [])]
        _file_visits = state.get("file_visits", {})
        _recently_extended_chain_ids = set(state.get("recently_extended_chain_ids", []))
    except (FileNotFoundError, json.JSONDecodeError, Exception):
        pass  # no state to restore, start fresh


_load_session_state()


def _build_session_context() -> dict:
    """Build the session context dict passed to retrieval and read tools."""
    return {
        "session_addrs": set(_session_addrs),
        "recalled_recent": _recalled_recent,
        "recently_extended_chain_ids": _recently_extended_chain_ids,
    }


mcp = FastMCP("mnemo", instructions=f"""You have a project memory. It persists across sessions.

This is the project's knowledge base — what's been learned about the
codebase, what decisions were made and why, what conventions to follow,
what's known to break, what's been tried before. When you start a fresh
session, this is how you get oriented fast instead of rediscovering
everything from scratch.

EVERY TURN, call memory_recall with what the user just said. Every
single turn. This is how you access project context. Don't think about
whether to do it — just do it. The result tells you what the project
already knows. Let it inform your response naturally.

memory_claim — when something worth preserving happens. Architecture
decisions, conventions established, bugs discovered, dependency
constraints, module responsibilities, approaches that worked or failed.
Don't store noise. Store what would make a future instance productive
on this project faster. The bar: "would a fresh instance need this?"

memory_update — when project knowledge has changed. A dependency was
upgraded, an approach was abandoned, a module was restructured, a bug
was fixed. Always reference what it replaces. The old knowledge stays
addressable but the active path moves forward.

memory_reinforce — when existing knowledge gets confirmed as still
current. Doesn't create new nodes, just marks existing ones as fresh.

memory_link — when two pieces of knowledge are related. "This decision
was caused_by this constraint." "This pattern depends_on this dependency."
Links turn the tree into a graph — recall follows them automatically.
Relationship types: relates_to, caused_by, depends_on, blocks, enables, contradicts.

memory_compress — when memory_status shows pressure, or when a cluster
of related facts can be summarized. Compression is lossy in content
but lossless in provenance.

Domains: architecture, decisions, patterns, tasks, issues, dependencies, history, context

The memory store is at: {STORE_PATH}
""")


# ===================================================================
# Associative recall — direct retrieval, no LLM intermediary
# ===================================================================

@mcp.tool()
def memory_recall(message: str, project: str = "") -> str:
    """
    Associative recall. Call this every turn with what the user said.
    Surfaces what the project already knows that's relevant.

    Depth adapts automatically — brief/trivial messages get minimal context,
    substantive questions get full context. You don't need to manage this.

    When project is specified, queries that project's store instead of the
    active one. Cross-project recall is lightweight — no extraction or
    session tracking.

    Args:
        message: What the user said (or a brief summary of the topic)
        project: Optional project name to query instead of the active store
    """
    # Cross-project recall — lightweight, no session tracking or extraction
    if project:
        try:
            target = _get_store(project)
        except ValueError as e:
            return str(e)
        relevant = retrieve_relevant(message, target, max_nodes=5)
        if not relevant:
            return f"(nothing found in {project})"
        lines = [f"From {project}:"]
        for item in relevant:
            domain = item["node"].meta.get("domain", "?")
            content = item["node"].content[:120]
            addr = item["node"].addr[:8]
            lines.append(f"  [{domain}] {content} [{addr}]")
        return "\n".join(lines)

    global _session_turns
    _session_turns += 1
    is_session_start = _session_turns <= 2 and not _session_addrs

    # Build session context for temporal relevance scoring
    session_context = _build_session_context()

    # Adaptive retrieval — max_nodes=0 lets signal density control depth
    result = associate(message, store, narrative=True, max_nodes=0,
                       session_context=session_context)
    density = result.get("signal_density", "medium")

    # First-recall priming — inject orientation context on session start
    orientation = ""
    if is_session_start:
        orientation_text = build_orientation(store, global_store)
        if orientation_text:
            orientation = orientation_text + "\n\n"
            emit("recall", "subconscious", "session orientation injected",
                 detail={"session_start": True})

    if not result["preload"]:
        emit("recall", "subconscious", "nothing surfaced",
             detail={"message": message[:100], "relevant_count": 0,
                     "session_turns": _session_turns, "density": density})
        recall_body = orientation or "(nothing comes to mind)"
    else:
        emit("recall", "subconscious", result["preload"],
             addresses=result["relevant_addrs"],
             detail={"message": message[:100],
                     "relevant_count": result["relevant_count"],
                     "tension_count": result["tension_count"],
                     "session_turns": _session_turns, "density": density})
        preload = result["preload"]
        # If only always-active pinned nodes surfaced, signal it — no domain hits
        if result["relevant_count"] > 0 and "── Chain" not in preload and not is_session_start:
            preload += "\n\n(no chain hits — always-active context only)"
        recall_body = orientation + preload

    response = recall_body

    # Track what was recalled for session affinity on future turns
    _recalled_recent.append(set(result["relevant_addrs"]))
    if len(_recalled_recent) > 5:
        _recalled_recent.pop(0)

    _save_session_state()

    # Update recall metadata on surfaced nodes
    for addr in result["relevant_addrs"]:
        node = store.get(addr)
        if node:
            node.meta["recall_count"] = node.meta.get("recall_count", 0) + 1
            node.meta["last_recalled"] = time.time()
            store.put(node)

    # Query global store for cross-project knowledge (user prefs, general patterns)
    if global_store.get_active():
        global_relevant = retrieve_relevant(message, global_store, max_nodes=3)
        if global_relevant:
            fragments = []
            for item in global_relevant:
                # Apply penalty — project context takes priority
                if item["score"] * 0.7 > 0.5:
                    content = item["node"].content
                    addr = item["node"].addr
                    fragments.append(f"{content} [{addr[:8]}]")
            if fragments:
                response += "\n\nGlobal context: " + " — ".join(fragments)

    # Cross-project fallback — when active store has sparse hits, check other
    # registered projects. Capped at 3 total hits to avoid noise.
    # Only fires when relevant_count < 2 (active store nearly empty on this topic).
    if result["relevant_count"] < 2:
        registry = _load_registry()
        cross_hits = []
        active_store_path = os.path.abspath(STORE_PATH)

        for proj_name, proj_path in registry.items():
            if not os.path.isdir(proj_path):
                continue
            if os.path.abspath(proj_path) == active_store_path:
                continue  # skip the active project
            if len(cross_hits) >= 3:
                break
            try:
                from mnemo import Store as _Store
                proj_store = _Store(proj_path)
                if not proj_store.get_active():
                    continue
                proj_relevant = retrieve_relevant(message, proj_store, max_nodes=2)
                for item in proj_relevant[:2]:
                    if item["score"] > 0.6 and len(cross_hits) < 3:
                        cross_hits.append((proj_name, item["node"]))
            except Exception:
                continue

        if cross_hits:
            fragments = []
            for proj_name, node in cross_hits:
                domain = node.meta.get("domain", "?")
                fragments.append(
                    f"  [{proj_name}/{domain}] {node.content[:120]} [{node.addr[:8]}]"
                )
            response += "\n\nCross-project context:\n" + "\n".join(fragments)
            emit("recall", "subconscious",
                 f"cross-project: {len(cross_hits)} hit(s) from other projects",
                 detail={"projects": list({p for p, _ in cross_hits}),
                         "count": len(cross_hits)})

    # Auto-compress when threshold reached — don't nudge, just do it
    if _session_turns >= COMPRESS_INTERVAL and _session_addrs:
        # Derive a summary from the cycle nodes (domain breakdown + snippets)
        active = store.get_active()
        cycle_nodes = [store.get(a) for a in _session_addrs if a in active]
        cycle_nodes = [n for n in cycle_nodes if n]
        domain_counts: dict[str, int] = {}
        for n in cycle_nodes:
            d = n.meta.get("domain", "context")
            domain_counts[d] = domain_counts.get(d, 0) + 1
        domain_str = ", ".join(
            f"{d}({c})" for d, c in
            sorted(domain_counts.items(), key=lambda x: -x[1])
        )
        snippets = "; ".join(n.content[:60] for n in cycle_nodes[:3])
        captured_turns = _session_turns
        captured_count = len(_session_addrs)
        auto_summary = f"auto-compress turn {captured_turns}: [{domain_str}] {snippets}"
        memory_session_compress(auto_summary)
        response += (
            f"\n\n[Auto-compressed {captured_count} nodes at turn {captured_turns}.]"
        )

    # Nudge for claiming when significant work has happened without claims
    elif _session_turns >= 10 and not _session_addrs:
        response += (
            "\n\n[Claim nudge: 10+ turns with no claims stored. "
            "If decisions were made or reasoning discussed, "
            "capture the 'why' with memory_claim before it fades.]"
        )

    # Queue extraction for this turn (runs in background thread)


    return response


# ===================================================================
# Core tools
# ===================================================================

@mcp.tool()
def memory_claim(content: str = "", domain: str = "", confidence: float = 0.8,
                 batch: list[dict] = None,
                 scope: str = "project",
                 anchors: list[dict] = None,
                 priority: float = 0,
                 project: str = "",
                 chain_id: str = "",
                 chain_name: str = "",
                 preliminary: bool = False,
                 agent_id: str = "",
                 ttl_days: float = 0) -> str:
    """
    Commit facts to the memory tree.

    Can store a single claim via the content/domain params, or multiple
    claims at once via the batch param. Use batch when you have several
    facts to store — it's one tool call instead of many.

    Scope controls where the claim is stored:
    - "project" (default): stored in the project-specific memory tree
    - "global": stored in the cross-project global memory (~/.mnemo/global/)
      Use global for user preferences, general conventions, workflow patterns,
      and knowledge that transcends any single codebase.

    Chain assignment (v2 stores):
    - chain_id: append this node to an existing chain (use memory_chains to list)
    - chain_name: create a new chain with this node as its seed (sets the summary)
    - Neither: node is standalone; extraction sidecar may propose chain membership later

    Args:
        content: The claim as a standalone fact (ignored if batch is provided)
        domain: Category — architecture, decisions, patterns, tasks, issues, dependencies, history, context (ignored if batch is provided)
        confidence: 0.0-1.0 how established this fact is (ignored if batch is provided)
        batch: List of claims, each with keys: content (str), domain (str), confidence (float, optional, default 0.8), scope (str, optional, default "project"), anchors (list, optional), priority (float, optional, default 0)
        scope: "project" or "global" — where to store the claim (ignored if batch is provided)
        anchors: Optional verification anchors — list of dicts like {"type": "file", "path": "..."}, {"type": "grep", "pattern": "...", "path": "..."}, {"type": "dependency", "name": "..."} (ignored if batch is provided)
        priority: Score boost for high-importance nodes (0 = normal, 0.5 = moderate, 1.0 = high). Use for user preferences, working agreements, critical invariants. (ignored if batch is provided)
        project: Optional project name — store the claim in that project's tree instead of the active one
        chain_id: Append this node to an existing chain (v2 stores only)
        chain_name: Create a new chain with this node as seed; chain_name becomes the summary (v2 stores only)
        preliminary: If True (v2 only), store in the session store as a preliminary node.
                     Preliminary nodes won't appear in project recall until promoted via memory_promote.
                     Use for work-in-progress reasoning you're not ready to commit.
        agent_id: Agent attribution (v2 multi-agent). Tags the node with the owning agent.
                  Used for retrieval boosting and chain-diff output. Optional in single-agent sessions.
        ttl_days: Time-to-live in days. Node is silently skipped in retrieval after this
                  many days from creation. Use for ephemeral working notes, task state,
                  and anything that's only relevant for the current sprint. 0 = no expiry.
    """
    claims = []
    if batch:
        claims = batch
    elif not content or not domain:
        return "Either provide content + domain for a single claim, or batch for multiple."
    else:
        claims = [{"content": content, "domain": domain,
                    "confidence": confidence, "scope": scope,
                    "anchors": anchors, "priority": priority}]

    results = []
    claimed_addrs = []

    for claim in claims:
        c = claim.get("content", "")
        d = claim.get("domain", "")
        conf = claim.get("confidence", 0.8)
        claim_scope = claim.get("scope", "project")
        claim_priority = claim.get("priority", 0)
        if not c or not d:
            results.append(f"Skipped (missing content or domain): {c[:40]}")
            continue

        # Resolve target store: preliminary > project param > scope > active store
        use_session = (preliminary and STORE_IS_V2
                       and _session_store is not None
                       and claim_scope == "project"
                       and not project)
        if use_session:
            target_store = _session_store
        elif project:
            try:
                target_store = _get_store(project)
            except ValueError as e:
                return str(e)
        elif claim_scope == "global":
            target_store = global_store
        else:
            target_store = store

        meta = {
            "domain": d,
            "confidence": conf,
            "source": "live",
            "scope": claim_scope,
        }
        if claim_priority:
            meta["priority"] = claim_priority
        if agent_id:
            meta["agent_id"] = agent_id
        claim_ttl = claim.get("ttl_days", ttl_days)
        if claim_ttl:
            meta["ttl_days"] = claim_ttl

        # Attach verified anchors if provided
        claim_anchors = claim.get("anchors") or anchors
        if claim_anchors:
            from mnemo_verify import validate_anchor
            processed = []
            for a in claim_anchors:
                if not validate_anchor(a) and a.get("type") != "content_hash":
                    continue
                # Auto-compute content_hash when context_lines provided but hash missing
                if (a.get("type") == "content_hash"
                        and a.get("context_lines")
                        and not a.get("content_hash")):
                    a = dict(a)  # don't mutate caller's dict
                    a["content_hash"] = compute_content_hash(a["context_lines"])
                processed.append(a)
            if processed:
                meta["anchors"] = processed

        node = Node(
            type="leaf",
            content=c,
            meta=meta,
        )
        target_store.put(node)
        active = target_store.get_active()
        active.add(node.addr)
        target_store.set_active(active)

        # Register content_hash anchors in the file index
        if meta.get("anchors"):
            update_file_index(target_store, node)

        # Only track project-scope claims in session (global is user-level)
        if claim_scope == "project" and not use_session:
            _session_addrs.append(node.addr)

        # Chain assignment (v2 stores only, project scope only)
        chain_tag = ""
        if claim_scope == "project" and STORE_IS_V2:
            try:
                from mnemo_chains import extend_chain, create_chain
                if chain_id:
                    ok = extend_chain(target_store, chain_id, node.addr)
                    if ok:
                        chain_tag = f" (→ chain {chain_id[:10]})"
                        _recently_extended_chain_ids.add(chain_id)
                    else:
                        chain_tag = f" (chain {chain_id[:10]} not found)"
                elif chain_name:
                    chain_authority = 0.0
                    new_chain_id = create_chain(
                        target_store, node.addr,
                        domain=d, summary=chain_name,
                        agent_id=agent_id or None,
                        authority=chain_authority,
                    )
                    chain_tag = f" (new chain {new_chain_id})"
                    _recently_extended_chain_ids.add(new_chain_id)
            except Exception:
                pass  # chain ops are non-fatal

        claimed_addrs.append(node.addr)
        scope_tag = " [global]" if claim_scope == "global" else ""
        prelim_tag = " [preliminary]" if use_session else ""
        emit("claim", "conscious", f"[{d}]{scope_tag}{prelim_tag} {c}",
             addresses=[node.addr], domain=d,
             detail={"confidence": conf, "scope": claim_scope,
                     "preliminary": use_session,
                     "chain_id": chain_id or None,
                     "chain_name": chain_name or None})
        results.append(
            f"Stored [{d}]{scope_tag}{prelim_tag} {node.addr}: {c}{chain_tag}"
        )

    _save_session_state()

    return "\n".join(results)


@mcp.tool()
def memory_update(old_address: str, new_content: str, reason: str = "",
                  domain: str = "") -> str:
    """
    Supersede an existing claim with updated information.
    The old claim remains addressable but the active path routes through the new one.
    Domain and confidence are inherited from the old node unless overridden.

    Args:
        old_address: Address (or prefix) of the claim being replaced
        new_content: The updated claim text
        reason: Why this changed (e.g. "dependency upgraded", "module restructured")
        domain: Override domain (leave empty to inherit from old node)
    """
    old = store.get(old_address)
    if not old:
        emit("update", "conscious", f"NOT FOUND: {old_address}")
        return f"Not found: {old_address}"

    new_addr = supersede(old.addr, new_content, store, reason=reason, domain=domain)
    if _is_recently_recalled(old.addr):
        new_node = store.get(new_addr)
        if new_node:
            new_node.meta["recall_hits"] = new_node.meta.get("recall_hits", 0) + 1
            store.put(new_node)
    _session_addrs.append(new_addr)
    _save_session_state()
    emit("update", "conscious",
         f"{old.addr[:8]} -> {new_addr[:8]}: {new_content}",
         addresses=[old.addr, new_addr],
         domain=old.meta.get("domain"),
         detail={"reason": reason, "old_content": old.content})


    return f"Updated {old.addr[:8]} -> {new_addr}: {new_content}"


@mcp.tool()
def memory_reinforce(address: str) -> str:
    """
    Mark an existing claim as still current. Bumps its freshness
    without creating a new node. Use when you verify something still holds.

    Args:
        address: Address (or prefix) of the claim to reinforce
    """
    node = store.get(address)
    if not node:
        emit("reinforce", "conscious", f"NOT FOUND: {address}")
        return f"Not found: {address}"

    node.meta["last_reinforced"] = time.time()
    count = node.meta.get("reinforcement_count", 0) + 1
    node.meta["reinforcement_count"] = count
    if _is_recently_recalled(node.addr):
        node.meta["recall_hits"] = node.meta.get("recall_hits", 0) + 1
    store.put(node)
    _save_session_state()
    emit("reinforce", "conscious",
         f"{node.addr[:8]}: {node.content}",
         addresses=[node.addr],
         domain=node.meta.get("domain"),
         detail={"reinforcement_count": count})


    return f"Reinforced {node.addr}: {node.content[:60]}"


@mcp.tool()
def memory_link(source: str, target: str, rel: str = "relates_to") -> str:
    """
    Create a directional link between two nodes, turning the tree into a graph.
    Links are metadata — they don't change node addresses or provenance.

    When recall surfaces the source node, linked nodes get a relevance
    boost proportional to the source's score. Causal links propagate stronger.

    Args:
        source: Address of the node to link FROM
        target: Address of the node to link TO
        rel: Relationship type — relates_to, caused_by, depends_on, blocks, enables, contradicts
    """
    valid_rels = {"relates_to", "caused_by", "depends_on", "blocks", "enables", "contradicts"}
    if rel not in valid_rels:
        return f"Invalid rel '{rel}'. Valid: {', '.join(sorted(valid_rels))}"

    src = store.get(source)
    if not src:
        return f"Source not found: {source}"
    tgt = store.get(target)
    if not tgt:
        return f"Target not found: {target}"

    # Add link to source node's meta
    links = src.meta.get("links", [])

    # Don't duplicate
    for existing in links:
        if existing.get("addr") == tgt.addr and existing.get("rel") == rel:
            return f"Link already exists: {src.addr[:8]} --{rel}--> {tgt.addr[:8]}"

    links.append({"addr": tgt.addr, "rel": rel})
    src.meta["links"] = links
    if _is_recently_recalled(src.addr):
        src.meta["recall_hits"] = src.meta.get("recall_hits", 0) + 1
    store.put(src)
    _save_session_state()

    emit("link", "conscious",
         f"{src.addr[:8]} --{rel}--> {tgt.addr[:8]}",
         addresses=[src.addr, tgt.addr],
         detail={"rel": rel,
                 "source_content": src.content[:60],
                 "target_content": tgt.content[:60]})
    return (
        f"Linked: {src.addr[:8]} --{rel}--> {tgt.addr[:8]}\n"
        f"  from: {src.content[:60]}\n"
        f"  to:   {tgt.content[:60]}"
    )


# ===================================================================
# Query tools
# ===================================================================

@mcp.tool()
def memory_query(address: str) -> str:
    """
    Look up a node by its address (prefix matching supported).

    Args:
        address: Full address or prefix (e.g. "a7f3c2")
    """
    node = store.get(address)
    if not node:
        return f"Not found: {address}"

    age_days = int((time.time() - node.created) / 86400)
    reinforced = node.meta.get("last_reinforced")
    r_info = ""
    if reinforced:
        r_days = int((time.time() - reinforced) / 86400)
        r_count = node.meta.get("reinforcement_count", 0)
        r_info = f"\nReinforced: {r_count}x, last {r_days}d ago"

    # Preserved values from compression
    pv_info = ""
    preserved = node.meta.get("preserved_values")
    if preserved:
        pv_lines = [p["fragment"] for p in preserved[:20]]
        pv_info = "\nPreserved values:\n  " + "\n  ".join(pv_lines)

    # Coverage score for compress nodes
    coverage = node.meta.get("coverage_score")
    c_info = f"\nCoverage: {coverage:.0%}" if coverage is not None else ""

    # Reverse links — what links TO this node
    rev_links = store.get_reverse_links(node.addr)
    rl_info = ""
    if rev_links:
        rl_lines = []
        for rl in rev_links:
            src = store.get(rl["source_addr"])
            preview = src.content[:50] if src else "?"
            rl_lines.append(f"  {rl['source_addr'][:8]} --{rl['rel']}--> here: {preview}")
        rl_info = "\nLinked FROM:\n" + "\n".join(rl_lines)

    # Anchors display
    anchor_info = ""
    node_anchors = node.meta.get("anchors")
    if node_anchors:
        anchor_lines = []
        for a in node_anchors:
            atype = a.get("type", "?")
            if atype == "file":
                anchor_lines.append(f"  file: {a.get('path', '?')}")
            elif atype == "grep":
                path_part = f" in {a['path']}" if a.get("path") else ""
                anchor_lines.append(f"  grep: '{a.get('pattern', '?')}'{path_part}")
            elif atype == "dependency":
                anchor_lines.append(f"  dependency: {a.get('name', '?')}")
        if anchor_lines:
            anchor_info = "\nAnchors:\n" + "\n".join(anchor_lines)

    emit("query", "conscious",
         f"{node.addr[:8]} [{node.meta.get('domain', '?')}] {age_days}d: {node.content}",
         addresses=[node.addr], domain=node.meta.get("domain"))
    return (
        f"addr: {node.addr}\n"
        f"type: {node.type}\n"
        f"domain: {node.meta.get('domain', 'n/a')}\n"
        f"age: {age_days}d\n"
        f"inputs: {node.inputs}{r_info}{c_info}{pv_info}{rl_info}{anchor_info}\n"
        f"content: {node.content}"
    )


@mcp.tool()
def memory_graph(
    address: str,
    depth: int = 2,
    rel_types: str = "",
    direction: str = "both",
) -> str:
    """
    Traverse the link graph from a node and render the subgraph.

    Unlike recall (which uses links to boost scores implicitly), this tool
    makes the graph structure explicit — useful for understanding how a decision
    connects to architecture, or how a bug relates to a dependency chain.

    Args:
        address:   Node address to start from (prefix matching supported)
        depth:     Hops to traverse (default 2; max useful is 3-4)
        rel_types: Comma-separated relationship types to follow
                   (default: all — caused_by, depends_on, blocks, enables,
                   relates_to, contradicts)
        direction: "forward" (this node → others), "reverse" (others → this),
                   or "both" (default)
    """
    from mnemo_graph import traverse_graph, render_graph

    node = store.get(address)
    if not node:
        return f"Not found: {address}"

    rels = [r.strip() for r in rel_types.split(",") if r.strip()] or None
    depth = max(1, min(depth, 5))  # clamp to reasonable range

    result = traverse_graph(store, node.addr, depth=depth,
                            rel_types=rels, direction=direction)
    rendered = render_graph(result)

    emit("graph", "conscious",
         f"graph from {node.addr[:8]} depth={depth} nodes={len(result['nodes'])}",
         addresses=[node.addr],
         detail={"depth": depth, "nodes": len(result["nodes"]),
                 "edges": len(result["edges"])})
    return rendered


@mcp.tool()
def memory_gap(topic: str, context: str = "") -> str:
    """
    Record a knowledge gap — something you're uncertain about that's
    potentially significant.

    Gaps surface in future recall like any issues node, so they can be
    answered by a future instance, another agent, or a conscious claim.
    Resolve a gap with memory_update(old_addr=<gap_addr>, content=<answer>).

    Args:
        topic:   What you don't know (e.g. "how _find_relevant_nodes ranks results")
        context: Why it matters / where you encountered it (optional)
    """
    content = f"[GAP] {topic}"
    if context:
        content += f" — {context}"

    node = Node(
        type="leaf",
        content=content,
        meta={
            "domain": "issues",
            "source": "conscious",
            "gap": True,
            "session_id": _session_id or "",
        },
    )
    store.put(node)
    active = store.get_active()
    active.add(node.addr)
    store.set_active(active)
    _save_session_state()

    emit("claim", "conscious", f"gap recorded: {topic[:80]}",
         addresses=[node.addr],
         detail={"topic": topic, "context": context})

    return (
        f"Gap recorded: [{node.addr[:8]}]\n"
        f"  {content}\n"
        f"Resolve with: memory_update(old_addr='{node.addr[:8]}', content='<answer>')"
    )


@mcp.tool()
def memory_ask(question: str, context: str = "", target_agent: str = "") -> str:
    """
    Record a pending decision or question that needs ratification.

    Unlike memory_gap (I don't know this), memory_ask is for choices you've
    made that need confirmation, or questions directed at another agent.
    Surfaces in recall with [ASK] marker until resolved.

    Args:
        question:     The question or decision to ratify
        context:      Why it matters / what's at stake (optional)
        target_agent: Auto-ping this agent with the question (optional)
    """
    content = f"[ASK] {question}"
    if context:
        content += f" — {context}"

    node = Node(
        type="leaf",
        content=content,
        meta={
            "domain": "issues",
            "source": "conscious",
            "ask": True,
            "session_id": _session_id or "",
        },
    )
    store.put(node)
    active = store.get_active()
    active.add(node.addr)
    store.set_active(active)
    _save_session_state()

    emit("claim", "conscious", f"ask recorded: {question[:80]}",
         addresses=[node.addr],
         detail={"question": question, "target_agent": target_agent})

    result = (
        f"Ask recorded: [{node.addr[:8]}]\n"
        f"  {content}\n"
        f"Resolve with: memory_update(old_addr='{node.addr[:8]}', content='<answer>')"
    )

    return result


@mcp.tool()
def memory_checkpoint(label: str, done: list[str] = None,
                      remaining: list[str] = None) -> str:
    """
    Mark current work state as a named checkpoint for session continuity.

    Checkpoints surface on the next session's first recall so you can
    resume from exactly where you left off. Stored in the session store
    and auto-promoted to the project store at session compress.

    Args:
        label:     Short description of where you are (e.g. "wired self-banner, need memory_where")
        done:      List of completed items (optional)
        remaining: List of items still to do (optional)
    """
    parts = [f"Checkpoint: {label}"]
    if done:
        parts.append("Done: " + "; ".join(done))
    if remaining:
        parts.append("Remaining: " + "; ".join(remaining))
    content = " | ".join(parts)

    target = _session_store if (STORE_IS_V2 and _session_store is not None) else store

    node = Node(
        type="leaf",
        content=content,
        meta={
            "domain": "tasks",
            "source": "conscious",
            "checkpoint": True,
            "priority": 0.5,
            "session_id": _session_id or "",
        },
    )
    target.put(node)
    active = target.get_active()
    active.add(node.addr)
    target.set_active(active)
    _save_session_state()

    emit("claim", "conscious", f"checkpoint: {label[:80]}",
         addresses=[node.addr],
         detail={"label": label, "done": done, "remaining": remaining})

    store_label = "session store (auto-promotes at compress)" if target is _session_store else "project store"
    return (
        f"Checkpoint saved: [{node.addr[:8]}] → {store_label}\n"
        f"  {content}"
    )


@mcp.tool()
def memory_verify(address: str = "") -> str:
    """
    Verify anchored claims against the actual codebase.
    If address given, verify that node. Otherwise verify all anchored active nodes.
    Returns pass/fail for each anchor.

    Args:
        address: Address of a specific node to verify (optional — omit to verify all)
    """
    from mnemo_verify import (
        verify_node, verify_active, _resolve_project_root,
    )

    project_root = _resolve_project_root()
    if not project_root:
        return "Could not resolve project root."

    if address:
        node = store.get(address)
        if not node:
            return f"Not found: {address}"
        anchors = node.meta.get("anchors", [])
        if not anchors:
            return f"Node {node.addr[:8]} has no anchors."

        result = verify_node(node, project_root)
        lines = [f"Verify {node.addr[:8]}: {result['passed']}/{result['total']} passed"]
        for r in result["results"]:
            mark = "\u2713" if r["passed"] else "\u2717"
            lines.append(f"  {mark} {r['detail']}")

        emit("verify", "conscious",
             f"{node.addr[:8]}: {result['passed']}/{result['total']} anchors passed",
             addresses=[node.addr],
             detail={"passed": result["passed"], "failed": result["failed"]})
        return "\n".join(lines)

    else:
        failures = verify_active(store, project_root)
        # Also count passing nodes for the summary
        anchored_count = 0
        for addr in store.get_active():
            n = store.get(addr)
            if n and n.meta.get("anchors"):
                anchored_count += 1

        if not failures:
            emit("verify", "conscious",
                 f"all {anchored_count} anchored node(s) passed",
                 detail={"anchored": anchored_count, "failures": 0})
            if anchored_count == 0:
                return "No anchored claims found."
            return f"All {anchored_count} anchored claim(s) verified successfully."

        lines = [f"Anchor failures ({len(failures)} of {anchored_count} anchored nodes):"]
        for result in failures:
            lines.append(f"\n  {result['addr'][:8]}: {result['content'][:70]}")
            for r in result["results"]:
                lines.append(f"    \u2717 {r['detail']}")

        emit("verify", "conscious",
             f"{len(failures)} failure(s) out of {anchored_count} anchored nodes",
             addresses=[r["addr"] for r in failures[:5]],
             detail={"anchored": anchored_count, "failures": len(failures)})
        return "\n".join(lines)


@mcp.tool()
def memory_search(query: str, max_results: int = 10,
                  scope: str = "all", project: str = "") -> str:
    """
    Search active memory using TF-IDF similarity.
    Finds conceptual matches, not just exact keywords.
    Searches both project and global memory by default.

    Args:
        query: What to search for (e.g. "how does retrieval work")
        max_results: Maximum results to return (per-project when project="all")
        scope: "all" (both stores), "project", or "global"
        project: Named project to search, or "all" to fan out across every
                 registered project and merge results ranked by score
    """
    from mnemo_associate import _get_backend, extract_signals, STOP_WORDS
    import re

    lower = query.lower()
    words = set(re.findall(r'[a-z_]+', lower)) - STOP_WORDS
    keywords = {w for w in words if len(w) >= 3}

    # Fan-out: search every registered project
    if project == "all":
        registry = _load_registry()
        if not registry:
            return "No registered projects. Projects are registered when you open a directory with a .mnemo/ store."

        all_results: list[tuple] = []  # (score, node, project_name)
        searched = []

        for proj_name, proj_path in registry.items():
            if not os.path.isdir(proj_path):
                continue
            try:
                from mnemo import Store as _Store
                proj_store = _Store(proj_path)
                if not proj_store.get_active():
                    continue
                backend = _get_backend(proj_store)
                backend.ensure_fresh(proj_store)
                backend.prepare_query(query)
                for addr in proj_store.get_active():
                    node = proj_store.get(addr)
                    if not node:
                        continue
                    score = backend.score(keywords, addr)
                    if score > 0:
                        all_results.append((score, node, proj_name))
                searched.append(proj_name)
            except Exception:
                continue

        if not searched:
            return "No projects with active nodes found."

        all_results.sort(key=lambda x: -x[0])
        results = all_results[:max_results]

        if not results:
            return f"No matches for '{query}' across {len(searched)} project(s): {', '.join(searched)}"

        emit("search", "conscious",
             f"'{query}' -> {len(results)} match(es) across {len(searched)} projects",
             detail={"query": query, "count": len(results), "projects": searched})

        lines = [f"Found {len(results)} match(es) across {len(searched)} project(s):"]
        for score, node, proj_name in results:
            domain = node.meta.get("domain", "?")
            lines.append(
                f"  [{proj_name}] {node.addr[:8]} [{domain}] ({score:.3f}): {node.content[:80]}"
            )
        return "\n".join(lines)

    # Single named project
    if project:
        try:
            target = _get_store(project)
        except ValueError as e:
            return str(e)
        backend = _get_backend(target)
        backend.ensure_fresh(target)
        backend.prepare_query(query)
        scored = []
        for addr in target.get_active():
            node = target.get(addr)
            if not node:
                continue
            score = backend.score(keywords, addr)
            if score > 0:
                scored.append((score, node))
        scored.sort(key=lambda x: -x[0])
        results = scored[:max_results]
        if not results:
            return f"No matches for '{query}' in {project}."
        lines = [f"Found {len(results)} match(es) in {project}:"]
        for score, node in results:
            domain = node.meta.get("domain", "?")
            lines.append(f"  {node.addr} [{domain}] ({score:.3f}): {node.content[:80]}")
        return "\n".join(lines)

    scored = []

    # Search project store
    if scope in ("all", "project"):
        backend = _get_backend(store)
        backend.ensure_fresh(store)
        backend.prepare_query(query)
        for addr in store.get_active():
            node = store.get(addr)
            if not node:
                continue
            score = backend.score(keywords, addr)
            if score > 0:
                scored.append((score, node, "project"))

    # Search global store
    if scope in ("all", "global") and global_store.get_active():
        g_backend = _get_backend(global_store)
        g_backend.ensure_fresh(global_store)
        g_backend.prepare_query(query)
        for addr in global_store.get_active():
            node = global_store.get(addr)
            if not node:
                continue
            score = g_backend.score(keywords, addr)
            if score > 0:
                scored.append((score, node, "global"))

    scored.sort(key=lambda x: -x[0])
    results = scored[:max_results]

    total_active = len(store.get_active()) + len(global_store.get_active())
    if not results:
        emit("search", "conscious", f"'{query}' -> no matches",
             detail={"query": query, "count": 0})
        return f"No matches for '{query}' in {total_active} active nodes."

    emit("search", "conscious",
         f"'{query}' -> {len(results)} match(es)",
         addresses=[n.addr for _, n, _ in results[:5]],
         detail={"query": query, "count": len(results)})
    lines = [f"Found {len(results)} match(es):"]
    for score, node, src in results:
        domain = node.meta.get("domain", "?")
        tag = " [G]" if src == "global" else ""
        lines.append(f"  {node.addr} [{domain}]{tag} ({score:.3f}): {node.content[:80]}")
    return "\n".join(lines)


@mcp.tool()
def memory_provenance(address: str) -> str:
    """
    Trace a claim back through its inputs to the original leaves.
    Shows the full derivation chain — useful for understanding
    how a piece of project knowledge evolved.

    Args:
        address: Address of the node to trace
    """
    chain = store.provenance(address)
    if not chain:
        emit("provenance", "conscious", f"NOT FOUND: {address}")
        return f"No provenance found for {address}"

    emit("provenance", "conscious",
         f"{address[:8]}: chain of {len(chain)} nodes",
         addresses=[n.addr for n in chain])
    lines = [f"Provenance ({len(chain)} nodes):"]
    for node in chain:
        preview = node.content[:70].replace("\n", " ")
        inputs = f"<- {len(node.inputs)} inputs" if node.inputs else "<- (leaf)"
        coverage = node.meta.get("coverage_score")
        coverage_tag = f" [coverage: {coverage:.0%}]" if coverage is not None else ""
        preserved = node.meta.get("preserved_values")
        pv_tag = f" [{len(preserved)} preserved values]" if preserved else ""
        lines.append(f"  {node.addr} [{node.type}]{coverage_tag}{pv_tag} {inputs}: {preview}")
    return "\n".join(lines)


# ===================================================================
# Compression tools
# ===================================================================

@mcp.tool()
def memory_compress(summary: str, addresses: list[str] = None,
                    domain: str = "") -> str:
    """
    Compress a set of nodes into a single summary node.
    The input nodes become inactive; the summary replaces them in the active set.
    Their content is still addressable via provenance.

    If no addresses given, compresses ALL active nodes in the specified domain.

    Args:
        summary: The compressed summary text
        addresses: List of node addresses to compress (optional)
        domain: If no addresses given, compress all active nodes in this domain
    """
    if not addresses and domain:
        active = store.get_active()
        addresses = []
        for addr in active:
            node = store.get(addr)
            if node and node.meta.get("domain") == domain:
                addresses.append(addr)
        if not addresses:
            return f"No active nodes found in domain '{domain}'"

    if not addresses:
        return "Provide either addresses or a domain to compress"

    addr = compress(addresses, summary, store, domain=domain)
    compress_node = store.get(addr)
    coverage = compress_node.meta.get("coverage_score") if compress_node else None
    coverage_str = f" [coverage: {coverage:.0%}]" if coverage is not None else ""
    emit("compress", "conscious",
         f"{len(addresses)} nodes -> {addr[:8]}: {summary}",
         addresses=[addr] + addresses[:5],
         domain=domain or None,
         detail={"count": len(addresses),
                 "coverage": coverage})
    return f"Compressed {len(addresses)} nodes -> {addr}{coverage_str}: {summary[:60]}"


@mcp.tool()
def memory_session_compress(summary: str) -> str:
    """
    Compress the current work cycle into a summary node.
    Call this when prompted by a session checkpoint, or at natural
    breakpoints in a long session.

    Captures the reasoning and decisions from this cycle — the "why"
    behind what was done — and compresses it into a single node whose
    inputs are all the nodes created during this cycle.

    The turn counter and session address list reset after compression.

    Args:
        summary: What was worked on and why. Focus on decisions made,
                 reasoning behind changes, and context that would help
                 a fresh instance understand this work.
    """
    global _session_turns, _session_addrs, _recalled_recent

    if not _session_addrs:
        return "Nothing to compress — no nodes created this cycle."

    # Filter to addresses that are still in the active set
    active = store.get_active()
    cycle_addrs = [a for a in _session_addrs if a in active]

    if not cycle_addrs:
        _session_turns = 0
        _session_addrs = []
        return "No active nodes from this cycle to compress (all may have been superseded)."

    # Create the compression node
    addr = compress(cycle_addrs, summary, store, domain="history")

    compress_node = store.get(addr)
    coverage = compress_node.meta.get("coverage_score") if compress_node else None
    coverage_str = f" [coverage: {coverage:.0%}]" if coverage is not None else ""

    turns = _session_turns
    node_count = len(cycle_addrs)

    emit("session_compress", "system",
         f"cycle: {turns} turns, {node_count} nodes -> {addr[:8]}: {summary}",
         addresses=[addr] + cycle_addrs[:5],
         domain="history",
         detail={"turns": turns, "node_count": node_count,
                 "cycle_addrs": cycle_addrs, "coverage": coverage})

    # Generate handoff node for next session's orientation
    logs_dir = os.path.join(STORE_PATH, "logs")
    handoff_addr = generate_handoff(
        store, _session_addrs, turns, logs_dir=logs_dir,
    )
    handoff_str = ""
    if handoff_addr:
        emit("handoff", "system",
             f"handoff node created: {handoff_addr[:8]}",
             addresses=[handoff_addr], domain="tasks")
        handoff_str = f"\nHandoff node: {handoff_addr} (will prime next session's first recall)"

    # Auto-update matching work arcs
    arc_str = ""
    if handoff_addr:
        handoff_node = store.get(handoff_addr)
        domains_touched = handoff_node.meta.get("domains_touched", []) if handoff_node else []
        from mnemo_arc import _extract_keywords
        work_keywords = _extract_keywords(summary)
        arc_matches = match_session_to_arcs(store, domains_touched, work_keywords)
        for arc_node, score in arc_matches[:2]:  # update top 2 matching arcs
            new_arc_addr = update_arc(
                store, arc_node.addr, summary[:120],
                domains_touched=domains_touched,
            )
            if new_arc_addr:
                arc_name = arc_node.meta.get("arc_name", "unnamed")
                emit("arc", "system",
                     f"arc auto-updated: {arc_name} (overlap {score:.0%})",
                     addresses=[new_arc_addr], domain="tasks",
                     detail={"action": "auto_update", "overlap": score})
                arc_str += f"\nArc updated: {arc_name} ({score:.0%} overlap) -> {new_arc_addr[:8]}"

    # Archive session store and start a new one (v2 only)
    global _session_id, _session_store
    session_str = ""
    if STORE_IS_V2 and _session_store is not None:
        # Promote any remaining preliminary chains before archiving
        remaining = list_preliminary_chains(_session_store)
        if remaining:
            auto_promoted = promote_all_preliminary(_session_store, store)
            if auto_promoted:
                session_str = f"\nAuto-promoted {len(auto_promoted)} preliminary chain(s) at compress time."

        archive_path = archive_session(store, _session_id)
        archive_note = f" (archived: {archive_path.name})" if archive_path else ""

        # Start a fresh session store for the next cycle
        from mnemo_session import new_session_id
        _session_id = new_session_id()
        _session_store = get_session_store(store, _session_id)
        session_str += f"\nSession store archived{archive_note}. New session: {_session_id}"

    # Reset for next cycle
    _session_turns = 0
    _session_addrs = []
    _recalled_recent = []

    return (
        f"Session compressed: {node_count} nodes from {turns} turns -> {addr}{coverage_str}\n"
        f"Summary: {summary[:120]}{handoff_str}{arc_str}{session_str}\n"
        f"Cycle reset. Next checkpoint in {COMPRESS_INTERVAL} turns."
    )


@mcp.tool()
def memory_reroot(domain_summaries: str = "") -> str:
    """
    Recompute the root (project knowledge hash) from the current active set.
    Optionally provide domain summaries as the root content.

    Args:
        domain_summaries: Optional structured summary for the root node
    """
    ds = {}
    if domain_summaries:
        for line in domain_summaries.strip().split("\n"):
            if ":" in line:
                d, s = line.split(":", 1)
                ds[d.strip()] = s.strip()

    addr = reroot(store, domain_summaries=ds if ds else None)
    emit("reroot", "system", f"new root: {addr[:12]}",
         addresses=[addr])
    return f"New root: {addr}"


# ===================================================================
# Status tools
# ===================================================================

@mcp.tool()
def memory_import(address: str, from_project: str) -> str:
    """
    Import a node from another registered project into the active tree.

    Found something relevant via memory_search(project="all") or
    memory_recall cross-project context? Pull it in with full provenance.
    The node is cloned into the active store — original stays untouched.

    Args:
        address: Address (or prefix) of the node to import
        from_project: Name of the source project (from memory_projects())
    """
    try:
        source_store = _get_store(from_project)
    except ValueError as e:
        return str(e)

    node = source_store.get(address)
    if not node:
        return f"Not found: {address} in {from_project}"

    # Clone into active store — preserve content, domain, confidence,
    # anchors, and priority. Tag provenance clearly.
    meta = {
        "domain": node.meta.get("domain", "context"),
        "confidence": node.meta.get("confidence", 0.7),
        "source": "import",
        "imported_from": node.addr,
        "source_project": from_project,
    }
    if node.meta.get("priority"):
        meta["priority"] = node.meta["priority"]
    if node.meta.get("anchors"):
        meta["anchors"] = node.meta["anchors"]

    new_node = Node(type="leaf", content=node.content, meta=meta)
    store.put(new_node)
    active = store.get_active()
    active.add(new_node.addr)
    store.set_active(active)

    if meta.get("anchors"):
        update_file_index(store, new_node)

    _session_addrs.append(new_node.addr)
    _save_session_state()

    emit("import", "conscious",
         f"imported {node.addr[:8]} from {from_project} -> {new_node.addr[:8]}",
         addresses=[new_node.addr],
         domain=meta["domain"],
         detail={"source_project": from_project, "original_addr": node.addr})

    return (
        f"Imported from {from_project}:\n"
        f"  Original: {node.addr[:8]} -> New: {new_node.addr}\n"
        f"  [{meta['domain']}] {node.content[:120]}"
    )


@mcp.tool()
def memory_status() -> str:
    """
    Show the current state of the memory system:
    project + global stores, health signals, domain breakdown.
    """
    active = store.get_active()
    ctx = build_active_context(store)
    ctx_len = len(ctx)

    # Health signals — actual tree quality, not raw size
    domains = {}
    for addr in active:
        node = store.get(addr)
        if node:
            d = node.meta.get("domain", "uncategorized")
            domains[d] = domains.get(d, 0) + 1

    health_notes = []
    # Domain imbalance
    for d, count in domains.items():
        if count >= 4:
            health_notes.append(f"{d} has {count} nodes — consider compressing")
    # Redundancy
    try:
        redundant = propose_supersessions(store, threshold=0.5)
        if redundant:
            health_notes.append(f"{len(redundant)} redundant pair(s) detected")
    except Exception:
        pass
    # Confidence decay — surface heavily decayed nodes
    decayed_count = 0
    drifted_count = 0
    for addr in active:
        node = store.get(addr)
        if not node:
            continue
        anchors = node.meta.get("anchors", [])
        if any(a.get("drifted") for a in anchors):
            drifted_count += 1
        last_fresh = node.meta.get("last_reinforced", node.created)
        days = (time.time() - last_fresh) / 86400
        base_conf = node.meta.get("confidence", 0.5)
        effective = base_conf * max(0.3, 1.0 - days * 0.02)
        if effective < 0.4:
            decayed_count += 1
    if drifted_count:
        health_notes.append(f"{drifted_count} node(s) with drifted anchors — run memory_verify")
    if decayed_count:
        health_notes.append(f"{decayed_count} node(s) confidence below 0.4 — consider reinforcing or updating")

    health = "healthy" if not health_notes else "; ".join(health_notes)

    root = store.current_root()

    # Chain info (v2 only)
    chain_line = ""
    if STORE_IS_V2:
        try:
            from mnemo_chains import get_chains
            all_chains = get_chains(store)
            active_chains = sum(1 for c in all_chains.values() if c.get("status") == "active")
            stashed_chains = sum(1 for c in all_chains.values() if c.get("status") == "stashed")
            chain_line = f"  Chains: {active_chains} active, {stashed_chains} stashed ({len(all_chains)} total)"
        except Exception:
            chain_line = "  Chains: (unavailable)"

    mode_line = "  Mode: v2 (chain-first)" if STORE_IS_V2 else "  Mode: v1-compat (flat nodes)"

    lines = [
        "── Project store ──",
        mode_line,
        f"  Active nodes: {len(active)}",
        f"  Context size: {ctx_len} chars",
        f"  Health: {health}",
        f"  Current root: {root or 'none'}",
        f"  Root history: {len(store.get_roots())} versions",
        f"  Path: {STORE_PATH}",
    ]
    if chain_line:
        lines.append(chain_line)
    lines += ["", "  Domains:"]
    for d, count in sorted(domains.items(), key=lambda x: -x[1]):
        lines.append(f"    {d}: {count}")

    # Global store info
    g_active = global_store.get_active()
    lines.append("")
    lines.append("── Global store ──")
    lines.append(f"  Active nodes: {len(g_active)}")
    if g_active:
        g_domains = {}
        for addr in g_active:
            node = global_store.get(addr)
            if node:
                d = node.meta.get("domain", "uncategorized")
                g_domains[d] = g_domains.get(d, 0) + 1
        g_root = global_store.current_root()
        lines.append(f"  Current root: {g_root or 'none'}")
        lines.append(f"  Domains:")
        for d, count in sorted(g_domains.items(), key=lambda x: -x[1]):
            lines.append(f"    {d}: {count}")
    lines.append(f"  Path: {GLOBAL_PATH}")

    emit("status", "system",
         f"project: {len(active)} nodes/{ctx_len} chars, health: {health}, "
         f"global: {len(g_active)} nodes",
         detail={"count": len(active), "ctx_len": ctx_len,
                 "health": health, "domains": domains,
                 "global_count": len(g_active)})
    return "\n".join(lines)


@mcp.tool()
def memory_diff() -> str:
    """
    Show what changed since the last root was computed.
    Useful at session start to quickly see what's new, modified, or removed
    without re-reading everything.

    Returns new nodes (added since last root), removed nodes (in root but
    no longer active), and unchanged count.
    """
    root_addr = store.current_root()
    active = store.get_active()

    if not root_addr:
        # No root yet — everything is "new"
        lines = [f"No root exists yet. All {len(active)} active nodes are unrooted."]
        for addr in sorted(active):
            node = store.get(addr)
            if node:
                domain = node.meta.get("domain", "?")
                lines.append(f"  + {addr[:8]} [{domain}]: {node.content[:70]}")
        emit("diff", "system", f"no root, {len(active)} unrooted nodes")
        return "\n".join(lines)

    root = store.get(root_addr)
    if not root:
        return f"Root {root_addr} not found in store."

    root_set = set(root.inputs)
    added = active - root_set
    removed = root_set - active
    unchanged = active & root_set

    lines = [f"Diff since root {root_addr[:8]} ({len(root_set)} nodes at root time):"]

    if added:
        lines.append(f"\n  Added ({len(added)}):")
        for addr in sorted(added):
            node = store.get(addr)
            if node:
                domain = node.meta.get("domain", "?")
                age_days = int((time.time() - node.created) / 86400)
                age = f"{age_days}d" if age_days > 0 else "today"
                lines.append(f"    + {addr[:8]} [{domain}] ({age}): {node.content[:65]}")

    if removed:
        lines.append(f"\n  Removed ({len(removed)}):")
        for addr in sorted(removed):
            node = store.get(addr)
            if node:
                domain = node.meta.get("domain", "?")
                # Check if it was superseded
                reason = ""
                descendants = store.descendants(addr)
                for d in descendants:
                    if d.type == "supersede":
                        reason = f" -> superseded by {d.addr[:8]}"
                        break
                lines.append(f"    - {addr[:8]} [{domain}]{reason}: {node.content[:60]}")

    lines.append(f"\n  Unchanged: {len(unchanged)}")

    emit("diff", "system",
         f"+{len(added)} -{len(removed)} ={len(unchanged)} since root {root_addr[:8]}",
         detail={"added": len(added), "removed": len(removed),
                 "unchanged": len(unchanged)})
    return "\n".join(lines)


@mcp.tool()
def memory_soul() -> str:
    """Generate the current project knowledge document — the compressed state of all project memory."""
    doc = generate_soul_doc(store)
    active = store.get_active()
    emit("soul", "system", f"generated project knowledge doc ({len(active)} nodes)",
         detail={"node_count": len(active)})
    return doc


@mcp.tool()
def memory_active() -> str:
    """Show the full active project memory context (what the model sees as current knowledge)."""
    ctx = build_active_context(store)
    return ctx if ctx else "(empty)"


@mcp.tool()
def memory_prune_candidates(threshold: float = 0.5) -> str:
    """
    Find pairs of similar active claims where the newer one
    likely supersedes the older one. Returns candidates for review.

    Args:
        threshold: Similarity threshold (0-1). Lower = more candidates.
    """
    proposals = propose_supersessions(store, threshold)
    emit("prune", "system",
         f"{len(proposals)} supersession candidate(s) at threshold={threshold}",
         detail={"count": len(proposals), "threshold": threshold})
    if not proposals:
        return "No supersession candidates found."

    lines = [f"Found {len(proposals)} candidate(s):"]
    for p in proposals:
        lines.append(f"\n  similarity: {p['similarity']}")
        lines.append(f"  old ({p['old'][:8]}): {p['old_content']}")
        lines.append(f"  new ({p['new'][:8]}): {p['new_content']}")
    return "\n".join(lines)


@mcp.tool()
def memory_explore(topic: str, deep: bool = False) -> str:
    """
    Tree-aware codebase exploration. Produces a reasoning trace:

    1. Recall — what does the tree already know about this topic?
    2. Locate — extract file references and code symbols from recalled nodes
    3. Search — grep the codebase for topic keywords + discovered symbols
    4. Gaps — what's in the code that the tree doesn't cover?
    5. Tensions — do anchored claims still hold against the code?

    Use this instead of raw grep/glob when you want the tree's perspective
    on a topic. The trace shows what's known, what's in the code, and
    where the two diverge.

    Args:
        topic: What to explore (e.g. "retrieval pipeline", "session tracking", "compression")
        deep: If true, includes targeted file reads of regions around matches
    """
    from pathlib import Path

    project_root = Path(os.environ.get(
        "MNEMO_PROJECT_ROOT",
        os.path.dirname(os.path.abspath(__file__)) or os.getcwd(),
    ))

    session_context = _build_session_context()

    result = _explore(topic, store,
                      session_context=session_context,
                      project_root=project_root,
                      deep=deep)

    emit("explore", "conscious",
         f"explored: {topic}",
         detail={"deep": deep, "topic": topic})

    return result


@mcp.tool()
def memory_grep(pattern: str, intent: str,
                glob: str = "", path: str = "") -> str:
    """
    Tree-aware pattern search. Like grep, but checks the tree first
    and annotates results with what the project already knows.

    The intent parameter is key — it tells the tool WHY you're searching,
    which lets it:
    1. Check if the tree already answers the question
    2. Use architecture knowledge to suggest which files to search first
    3. Annotate results with tree context for each matched file
    4. Prioritize tree-suggested files in the output

    Use this instead of raw grep when the search benefits from project
    context. Use raw grep for simple string lookups where context doesn't matter.

    Args:
        pattern: Regex pattern to search for (e.g. "def compress", "recall_hits")
        intent: Why you're searching — what you want to understand or find
        glob: File extension filter (e.g. "*.py", "*.ts") — empty for all code files
        path: Subdirectory to scope the search (e.g. "src/") — empty for project root
    """
    from pathlib import Path as P

    project_root = P(os.environ.get(
        "MNEMO_PROJECT_ROOT",
        os.path.dirname(os.path.abspath(__file__)) or os.getcwd(),
    ))

    session_context = _build_session_context()

    result = _grep(pattern, intent, store,
                   session_context=session_context,
                   project_root=project_root,
                   glob_filter=glob or None,
                   path=path or None)

    emit("grep", "conscious",
         f"grep: /{pattern}/ ({intent})",
         detail={"pattern": pattern, "intent": intent,
                 "glob": glob, "path": path})

    return result


@mcp.tool()
def memory_plan(task: str) -> str:
    """
    Tree-aware planning context. Retrieves and organizes everything
    the tree knows that's relevant to a task, structured by planning role:

    1. Architecture — which modules/systems are involved
    2. Constraints — decisions, patterns, and dependencies to respect
    3. Risks — known issues, gotchas, and fragile areas
    4. Current state — what's in progress, what's blocked
    5. History — what was tried before
    6. Affected files — where changes need to happen
    7. Blockers — issues that overlap with the task + graph "blocks" links

    This is NOT the plan — it's the tree-informed context the plan should
    be built on. Use this before planning any non-trivial change.

    Args:
        task: What you're planning to do (e.g. "fix the recall_hits bug on supersede")
    """
    from pathlib import Path as P

    project_root = P(os.environ.get(
        "MNEMO_PROJECT_ROOT",
        os.path.dirname(os.path.abspath(__file__)) or os.getcwd(),
    ))

    session_context = _build_session_context()

    result = _plan(task, store,
                   session_context=session_context,
                   project_root=project_root)

    emit("plan", "conscious",
         f"plan context: {task}",
         detail={"task": task})

    return result


@mcp.tool()
def memory_read(file: str, offset: int = 0, limit: int = 0) -> str:
    """
    Tree-annotated file reading. Reads a file and overlays it with
    everything the tree knows about it:

    1. Header — tree nodes that reference this file/module
    2. Section annotations — tree context at class/function boundaries
    3. Line annotations — known issues, decisions at specific lines
       (e.g. "mnemo_mcp.py:417 — recall_hits lost on supersede")
    4. Summary — tree coverage stats and suggestions

    Like reading code with a senior dev's annotations already in place.
    Use this instead of raw file reads when you want the tree's
    accumulated knowledge about a file.

    Args:
        file: Path to read (relative to project root or absolute)
        offset: Start from this line number (0-based, default: beginning)
        limit: Max lines to show (default: up to 300)
    """
    from pathlib import Path as P

    project_root = P(os.environ.get(
        "MNEMO_PROJECT_ROOT",
        os.path.dirname(os.path.abspath(__file__)) or os.getcwd(),
    ))

    session_context = _build_session_context()

    basename = os.path.basename(file)
    _file_visits[basename] = _file_visits.get(basename, 0) + 1
    visit = _file_visits[basename]
    _save_session_state()

    # Stale anchor check
    stale_warning = ""
    try:
        project_root_path = _get_project_root()
        stale = _check_stale_anchors(store, file, project_root_path)
        if stale:
            sw = [f"⚠ {len(stale)} stale anchor{'s' if len(stale) != 1 else ''} — nodes may describe outdated code:"]
            for s in stale[:3]:
                n = s["node"]
                sw.append(f"  [{n.addr[:8]}] {n.content[:70]} ({s['status']})")
            if len(stale) > 3:
                sw.append(f"  ... {len(stale) - 3} more — run memory_verify for full check")
            stale_warning = "\n".join(sw) + "\n\n"
    except Exception:
        pass

    result = _read(file, store,
                   session_context=session_context,
                   project_root=project_root,
                   offset=offset,
                   limit=limit,
                   visit=visit)

    emit("read", "conscious",
         f"read: {file} (visit {visit})",
         detail={"file": file, "offset": offset, "limit": limit,
                 "visit": visit})

    return stale_warning + result


@mcp.tool()
def memory_map(path: str, extensions: str = "") -> str:
    """
    Cartographer: map an existing file or directory to content-hash anchors.

    Walks the target, detects every function/class/struct, and generates
    comprehension nodes bound to those sections via content hash. Nodes
    inject automatically on future reads — no manual recall needed.

    Use on existing codebases that have no coverage yet, or on a specific
    file after significant refactoring.

    Args:
        path: File or directory to map (relative to project root or absolute)
        extensions: Comma-separated file extensions to process, e.g. ".py,.ts"
                    (default: .py .js .ts .tsx .jsx .rs .go .c .h .cpp .cs)
    """
    from pathlib import Path as P
    from mnemo_log import _get_log_path

    project_root = P(os.environ.get(
        "MNEMO_PROJECT_ROOT",
        os.path.dirname(os.path.abspath(__file__)) or os.getcwd(),
    ))

    ext_set = None
    if extensions.strip():
        ext_set = {e.strip() for e in extensions.split(",") if e.strip()}

    try:
        import anthropic
        client = anthropic.Anthropic()
    except Exception as e:
        return f"Error: anthropic SDK unavailable — {e}"

    emit("map_start", "conscious",
         f"mapping: {path}",
         detail={"path": path, "extensions": extensions or "default"})

    result = _map_path(
        target=path,
        store=store,
        project_root=project_root,
        extensions=ext_set,
        log_path=_get_log_path(),
        client=client,
    )

    if "error" in result:
        return f"Map failed: {result['error']}"

    lines = [
        f"Map complete: {path}",
        f"  Files processed:  {result['files_processed']}",
        f"  Sections found:   {result['sections_found']}",
        f"  Nodes created:    {result['nodes_created']}",
        f"  Sections skipped: {result['nodes_skipped']}",
    ]
    if result["unmapped"]:
        shown = result["unmapped"][:5]
        more = len(result["unmapped"]) - len(shown)
        lines.append(f"  Unmapped files:   {', '.join(shown)}"
                     + (f" (+{more} more)" if more else ""))
    lines.append(
        f"\nComprehension nodes are now bound to code sections in {path}. "
        f"They will inject automatically on future reads."
    )

    _save_session_state()

    return "\n".join(lines)


@mcp.tool()
def memory_scan(path: str = ".", extensions: str = "",
                force: bool = False) -> str:
    """
    Scan a file or directory and commit structural claims to the tree.

    Extracts module docstrings, class docstrings, and public function
    signatures + docstrings via static analysis (no LLM). Stores each
    as a claim with a file anchor. Idempotent — unchanged files are
    skipped on subsequent calls.

    Use this to bootstrap the tree on an existing codebase, or to
    refresh after significant changes.

    Args:
        path:       File or directory to scan (default: project root)
        extensions: Comma-separated extensions to include, e.g. ".py,.ts"
                    (default: .py .js .ts .tsx .jsx .rs .go .c .h .cpp .cs)
        force:      Re-scan even unchanged files (default: False)
    """
    from pathlib import Path as P
    from mnemo_scan import scan as _scan

    project_root = P(os.environ.get(
        "MNEMO_PROJECT_ROOT",
        os.path.dirname(os.path.abspath(__file__)) or os.getcwd(),
    ))

    ext_set = None
    if extensions.strip():
        ext_set = {e.strip() for e in extensions.split(",") if e.strip()}

    result = _scan(path, store, project_root=project_root,
                   extensions=ext_set, force=force)

    if "error" in result:
        return f"Scan failed: {result['error']}"

    lines = [
        f"Scan complete: {path}",
        f"  Files scanned:  {result['files_scanned']}",
        f"  Files skipped:  {result['files_skipped']} (unchanged)",
        f"  Claims created: {result['claims_created']}",
    ]

    if result.get("per_file"):
        lines.append("\nPer file:")
        for rel, count in sorted(result["per_file"].items()):
            lines.append(f"  {rel}: {count} claim(s)")

    _save_session_state()
    return "\n".join(lines)


@mcp.tool()
def memory_pipeline(name: str, steps: list, description: str = "") -> str:
    """
    Define a reusable memory pipeline and store it as a node in the tree.

    A pipeline is a sequence of steps that transform a node set. Each step
    has an 'op' field and op-specific parameters. Variables in string
    parameters ({varname}) are resolved at run time from the params passed
    to memory_run.

    Available ops:
      Sources (ignore current set):
        recall(query, max_nodes=8)       - associative recall
        search(query, max_nodes=8)       - TF-IDF search
        active(domain?)                  - all active nodes
        spatial(file)                    - nodes anchored to a file

      Transforms (node set -> node set):
        traverse(depth=2, rel_types?, direction="both")
        filter(domain?, min_priority?, min_confidence?, has_anchors?, type?)
        sort(by="created"|"priority"|"confidence", reverse=True)
        limit(n=10)
        dedupe()

      Sinks (side effects, pass through):
        compress(label, domain="context") - compress set into summary node
        claim(content, domain="context", priority=0)
        link(target, rel="relates_to")

    Example:
        memory_pipeline("my-orient", [
            {"op": "recall", "query": "{input}"},
            {"op": "traverse", "depth": 1},
            {"op": "compress", "label": "context: {input}"}
        ])
    """
    from mnemo_pipeline import define_pipeline
    addr = define_pipeline(name, steps, store, description=description)
    emit("pipeline", "conscious", f"defined pipeline '{name}' ({len(steps)} steps)",
         detail={"name": name, "addr": addr})
    return f"Pipeline '{name}' stored at {addr} ({len(steps)} steps)."


@mcp.tool()
def memory_run(name: str, params: dict = None) -> str:
    """
    Run a named pipeline against the project store.

    Built-in pipelines:
      session-orient   recall + traverse + compress for a topic
      file-context     surface all tree knowledge for a file
      issue-cluster    compress all known issues into a summary
      drift-check      find nodes with potentially stale anchors

    Args:
        name:   Pipeline name or address prefix
        params: Variables to substitute into {varname} step parameters
                e.g. {"input": "authentication"}
    """
    from mnemo_pipeline import get_pipeline, run_pipeline, render_result
    pipeline_def = get_pipeline(name, store)
    if not pipeline_def:
        return f"Pipeline '{name}' not found. Use memory_pipelines() to list available."

    p = params or {}
    result = run_pipeline(pipeline_def, store, params=p)

    emit("pipeline_run", "conscious",
         f"ran pipeline '{name}': {result['steps_run']} steps, {len(result['nodes'])} nodes",
         detail={"name": name, "params": p, "errors": result["errors"]})

    return render_result(result)


@mcp.tool()
def memory_pipelines() -> str:
    """
    List all available pipelines: built-ins and stored.
    """
    from mnemo_pipeline import list_pipelines
    pipelines = list_pipelines(store)
    if not pipelines:
        return "No pipelines defined."

    lines = [f"{len(pipelines)} pipeline(s) available:\n"]
    for p in pipelines:
        src = p["source"]
        addr = f"  {p['addr'][:8]}" if p.get("addr") else ""
        lines.append(f"  {p['name']}  [{src}]{addr}")
        if p.get("description"):
            lines.append(f"    {p['description']}")
        lines.append(f"    {p['steps_count']} step(s)")
    return "\n".join(lines)


@mcp.tool()
def memory_learn(chain_id: str, name: str = "", store_pipeline: bool = True) -> str:
    """
    Extract a reusable pipeline from a successful chain.

    Analyzes the sequence of nodes in the chain and infers what operations
    produced them — recall patterns, traversal, domain clustering, compression.
    Produces a pipeline that captures the methodology, not the content.

    The learned pipeline can be run immediately with memory_run() using
    {input} as the variable for the topic or file.

    Args:
        chain_id:       Chain ID (ch_...) or address prefix
        name:           Name for the extracted pipeline (default: learned-<chain_id>)
        store_pipeline: Store the pipeline as a node in the tree (default: True)
    """
    from mnemo_pipeline import learn_from_chain, define_pipeline, render_learned

    pipeline_def = learn_from_chain(chain_id, store, name=name)
    if not pipeline_def:
        return f"Chain '{chain_id}' not found or too short to extract a pattern."

    addr = None
    if store_pipeline:
        addr = define_pipeline(
            pipeline_def["name"],
            pipeline_def["steps"],
            store,
            description=pipeline_def.get("description", ""),
        )
        # Link pipeline node back to the chain's head node
        from mnemo_chains import get_chain
        chain = get_chain(store, chain_id)
        if chain and chain.get("head") and addr:
            head_node = store.get(chain["head"])
            if head_node:
                links = head_node.meta.setdefault("links", [])
                links.append({"addr": addr, "rel": "enables"})
                store.put(head_node)

    emit("learn", "conscious",
         f"learned pipeline '{pipeline_def['name']}' from chain {chain_id[:8]}",
         detail={"chain_id": chain_id, "steps": len(pipeline_def["steps"]), "addr": addr})

    result = render_learned(pipeline_def)
    if addr:
        result += f"\n\nStored at {addr[:8]}."
    return result


@mcp.tool()
def memory_coverage(path: str = ".", extensions: str = "") -> str:
    """
    Anchor coverage report: what percentage of this codebase has
    comprehension nodes bound to it via content-hash anchors?

    Shows covered files (with section density), unmapped files, and an
    overall coverage percentage. Sorted worst-first so you know exactly
    where to run memory_map next.

    Zero API calls — reads the file index and detects sections locally.

    Args:
        path: File or directory to audit (default: current directory)
        extensions: Comma-separated extensions to check, e.g. ".py,.ts"
                    (default: .py .js .ts .tsx .jsx .rs .go .c .h .cpp .cs)
    """
    from pathlib import Path as P

    project_root = P(os.environ.get(
        "MNEMO_PROJECT_ROOT",
        os.path.dirname(os.path.abspath(__file__)) or os.getcwd(),
    ))

    ext_set = None
    if extensions.strip():
        ext_set = {e.strip() for e in extensions.split(",") if e.strip()}

    result = _coverage(
        target=path,
        store=store,
        project_root=project_root,
        extensions=ext_set,
    )

    emit("coverage", "conscious",
         f"coverage: {result['covered_files']}/{result['total_files']} files, "
         f"{result['covered_sections']}/{result['total_sections']} sections",
         detail={"path": path,
                 "covered_files": result["covered_files"],
                 "total_files": result["total_files"]})

    return _format_coverage(result, path)


@mcp.tool()
def memory_infer(layers: str = "all") -> str:
    """
    Passive pattern inference from session logs. Analyzes behavioral
    patterns across all sessions to discover implicit knowledge:

    1. Co-occurrence — files consistently edited/discussed together
       (proposes relates_to links)
    2. Recall patterns — which nodes are noise vs high-value, which
       are always recalled together, which are never surfaced
    3. Corrections — behavioral patterns from update history: correction
       types (user feedback, staleness, evolution, recategorization),
       domain stability, rapid corrections, supersession chains
    4. Sequences — implicit knowledge from action patterns: volatile
       nodes (recalled then corrected), discovery sequences, core
       knowledge cluster, domain cascades, recall trigger mapping
    5. Workflow — session structure, domain transitions, event distribution

    No LLM calls. Pure statistics over session logs.

    Args:
        layers: Which layers to run. "all" (default), or comma-separated:
                "cooccurrence", "recall", "corrections", "sequences", "workflow"
    """
    if layers == "all":
        layer_list = ["cooccurrence", "recall", "workflow"]
    else:
        layer_list = [l.strip() for l in layers.split(",")]

    logs_dir = os.path.join(STORE_PATH, "logs")

    result = _infer(store, logs_dir=logs_dir, layers=layer_list)

    emit("infer", "system",
         f"pattern inference: {', '.join(layer_list)}",
         detail={"layers": layer_list})

    return result


@mcp.tool()
def memory_arc(
    action: str,
    name: str = "",
    goal: str = "",
    arc_address: str = "",
    progress: str = "",
    next_step: str = "",
    outcome: str = "",
    reason: str = "",
    domains: list[str] = None,
) -> str:
    """
    Manage work arcs — multi-session goals that track momentum and trajectory.

    A work arc spans multiple sessions. It captures not just where you left off,
    but where you're heading. Active arcs surface automatically on session start.

    Actions:
      create  — start a new arc (requires name + goal)
      update  — append progress to an arc (requires arc_address + progress)
      complete — mark an arc as done (requires arc_address, optional outcome)
      pause   — pause without completing (requires arc_address, optional reason)
      list    — show all active/paused arcs
      detect  — scan handoff chain for potential arcs to create

    Args:
        action: create, update, complete, pause, list, or detect
        name: Arc name for create (e.g. "auth refactor", "session continuity")
        goal: What this arc aims to achieve (for create)
        arc_address: Address of existing arc (for update/complete/pause)
        progress: What happened this session (for update)
        next_step: Where we're heading next (for update)
        outcome: Final result (for complete)
        reason: Why pausing (for pause)
        domains: Relevant domains (for create)
    """
    if action == "create":
        if not name or not goal:
            return "Create requires name and goal."
        addr = create_arc(store, name, goal, domains=domains)
        emit("arc", "conscious", f"arc created: {name}",
             addresses=[addr], domain="tasks",
             detail={"action": "create", "name": name})
        node = store.get(addr)
        return f"Arc created: {addr}\n\n{node.content}" if node else f"Arc created: {addr}"

    elif action == "update":
        if not arc_address or not progress:
            return "Update requires arc_address and progress."
        new_addr = update_arc(store, arc_address, progress, next_step=next_step)
        if not new_addr:
            return f"Arc not found: {arc_address}"
        _session_addrs.append(new_addr)
        emit("arc", "conscious", f"arc updated: {progress[:60]}",
             addresses=[new_addr], domain="tasks",
             detail={"action": "update", "old": arc_address[:8]})
        node = store.get(new_addr)
        return f"Arc updated: {arc_address[:8]} -> {new_addr}\n\n{node.content}" if node else f"Arc updated: {new_addr}"

    elif action == "complete":
        if not arc_address:
            return "Complete requires arc_address."
        new_addr = complete_arc(store, arc_address, outcome=outcome)
        if not new_addr:
            return f"Arc not found: {arc_address}"
        emit("arc", "conscious", f"arc completed: {outcome[:60]}",
             addresses=[new_addr], domain="tasks",
             detail={"action": "complete", "old": arc_address[:8]})
        node = store.get(new_addr)
        return f"Arc completed: {new_addr}\n\n{node.content}" if node else f"Arc completed: {new_addr}"

    elif action == "pause":
        if not arc_address:
            return "Pause requires arc_address."
        new_addr = pause_arc(store, arc_address, reason=reason)
        if not new_addr:
            return f"Arc not found: {arc_address}"
        emit("arc", "conscious", f"arc paused: {reason[:60]}",
             addresses=[new_addr], domain="tasks",
             detail={"action": "pause", "old": arc_address[:8]})
        return f"Arc paused: {new_addr}"

    elif action == "list":
        arcs = find_active_arcs(store)
        if not arcs:
            return "No active or paused arcs."
        lines = [f"Active arcs ({len(arcs)}):"]
        for arc in arcs:
            status = arc.meta.get("arc_status", "?")
            sessions = arc.meta.get("arc_sessions", 0)
            name = arc.meta.get("arc_name", "unnamed")
            lines.append(f"\n  {arc.addr[:8]} [{status}] {name} ({sessions} sessions)")
            # Show first line of goal
            for content_line in arc.content.split("\n"):
                if content_line.strip().startswith("Goal:"):
                    lines.append(f"    {content_line.strip()}")
                    break
        return "\n".join(lines)

    elif action == "detect":
        candidates = detect_arc_candidates(store)
        if not candidates:
            return "No arc candidates detected from handoff history."
        lines = [f"Detected {len(candidates)} potential arc(s):\n"]
        for i, c in enumerate(candidates, 1):
            lines.append(f"  {i}. \"{c['name']}\" (overlap: {c['overlap_score']:.0%})")
            lines.append(f"     Keywords: {', '.join(c['keywords'][:8])}")
            lines.append(f"     Evidence: handoffs {', '.join(c['evidence'])}")
            if c['domains']:
                lines.append(f"     Domains: {', '.join(c['domains'])}")
            lines.append("")
        lines.append("Create with memory_arc(action='create', name='...', goal='...')")
        return "\n".join(lines)

    else:
        return f"Unknown action: {action}. Valid: create, update, complete, pause, list, detect"


@mcp.tool()
def memory_switch(
    project_path: str,
) -> str:
    """Switch mnemo to a different project's memory store.

    Saves current session state, then points mnemo at the new project.
    All subsequent memory operations (recall, claim, etc.) will use the
    new store. The global store is unaffected.

    The path can be:
    - A project root containing a .mnemo/ directory
    - A direct path to a .mnemo store directory

    Args:
        project_path: Path to project root or .mnemo store directory
    """
    global STORE_PATH, STORE_IS_V2, store
    global _session_turns, _session_addrs, _recalled_recent
    global _session_id, _session_store

    resolved = os.path.expanduser(project_path)
    resolved = os.path.abspath(resolved)

    # Accept either a project root (with .mnemo/) or a direct store path
    if os.path.isdir(os.path.join(resolved, ".mnemo")):
        new_store_path = os.path.join(resolved, ".mnemo")
    elif os.path.isdir(resolved) and os.path.basename(resolved) == ".mnemo":
        new_store_path = resolved
    elif os.path.isdir(os.path.join(resolved, "nodes")):
        new_store_path = resolved
    else:
        new_store_path = os.path.join(resolved, ".mnemo")

    old_store_path = STORE_PATH
    old_active_count = len(store.get_active())

    # Save current session state before switching
    _save_session_state()
    emit("switch", "system",
         f"switching store: {old_store_path} -> {new_store_path}",
         detail={"from": old_store_path, "to": new_store_path})

    # Switch store
    STORE_PATH = new_store_path
    STORE_IS_V2 = os.path.exists(os.path.join(new_store_path, "chains.json"))
    store = Store(STORE_PATH)
    log_configure(STORE_PATH)

    # Reset session state and load from new store
    _session_turns = 0
    _session_addrs = []
    _recalled_recent = []
    _session_id = ""
    _session_store = None
    _load_session_state()
    _init_session_store()

    new_active_count = len(store.get_active())

    emit("status", "system",
         f"mnemo switched to: {STORE_PATH} ({new_active_count} active nodes)")

    # Auto-register the new store
    parent = os.path.dirname(new_store_path)
    dir_name = os.path.basename(parent)
    _register_project(dir_name, new_store_path)

    return (
        f"Switched mnemo store.\n"
        f"  From: {old_store_path} ({old_active_count} nodes)\n"
        f"  To:   {STORE_PATH} ({new_active_count} nodes)\n"
        f"  Registered as: {dir_name}\n\n"
        f"All memory operations now target the new store."
    )


@mcp.tool()
def memory_init(name: str = "") -> str:
    """Create a new .mnemo memory store in the current working directory.

    Initializes the store, registers it by name, and switches to it.
    If .mnemo/ already exists, just registers the name and switches.

    Args:
        name: Project name for the registry. Defaults to the directory name.
    """
    global STORE_PATH, STORE_IS_V2, store
    global _session_turns, _session_addrs, _recalled_recent
    global _session_id, _session_store

    cwd = os.getcwd()
    store_path = os.path.join(cwd, ".mnemo")
    project_name = name or os.path.basename(cwd)

    already_existed = os.path.isdir(store_path)

    # Save current session state before switching
    _save_session_state()

    # Create/open the store (Store.__init__ creates dirs)
    STORE_PATH = store_path
    STORE_IS_V2 = os.path.exists(os.path.join(store_path, "chains.json"))
    store = Store(STORE_PATH)
    log_configure(STORE_PATH)

    # Register
    _register_project(project_name, store_path)

    # Reset session
    _session_turns = 0
    _session_addrs = []
    _recalled_recent = []
    _session_id = ""
    _session_store = None
    _load_session_state()
    _init_session_store()

    active_count = len(store.get_active())

    emit("init", "system",
         f"memory_init: {project_name} at {store_path}",
         detail={"name": project_name, "path": store_path,
                 "existed": already_existed})

    if already_existed:
        return (
            f"Store already exists at {store_path}\n"
            f"  Registered as: {project_name}\n"
            f"  Active nodes: {active_count}\n"
            f"  Now active — all memory operations target this store."
        )
    return (
        f"Created new memory store: {store_path}\n"
        f"  Registered as: {project_name}\n"
        f"  Now active — use memory_claim to start building knowledge."
    )


@mcp.tool()
def memory_projects() -> str:
    """List all registered project memory stores.

    Shows project names, node counts, and paths.
    The active store is marked.
    """
    registry = _load_registry()
    if not registry:
        return "No projects registered. Use memory_init() to create one."

    lines = ["Registered projects:"]
    for name, path in sorted(registry.items()):
        is_active = os.path.abspath(path) == os.path.abspath(STORE_PATH)
        tag = " (active)" if is_active else ""

        if os.path.isdir(path):
            try:
                s = Store(path)
                count = len(s.get_active())
                lines.append(f"  {name}{tag}: {count} nodes — {path}")
            except Exception:
                lines.append(f"  {name}{tag}: (error reading) — {path}")
        else:
            lines.append(f"  {name}{tag}: (path missing) — {path}")

    return "\n".join(lines)


# ===================================================================
# Chain tools (v2)
# ===================================================================

@mcp.tool()
def memory_cat(chain_or_node: str, code: bool = False) -> str:
    """
    Render a chain or node as a coherent narrative (tail → head).

    For a chain_id (ch_...): renders the full reasoning path as a story,
    showing each node's content in order with agent attribution and code
    context where available.

    For a node address: renders the node and its provenance chain
    depth-first, reconstructing the reasoning path from ancestors.

    Args:
        chain_or_node: A chain ID (ch_...) or node address (or prefix)
        code: If true, emphasize code context — shows full snapshots at each step
    """
    if not STORE_IS_V2:
        return "memory_cat requires a v2 store (.mnemo/). Current store is v1-compat."

    from mnemo_chains import get_chain, render_chain

    # Chain ID path
    if chain_or_node.startswith("ch_"):
        chain = get_chain(store, chain_or_node)
        if not chain:
            return f"Chain not found: {chain_or_node}"
        rendered = render_chain(chain, store, max_chars=20000)
        emit("cat", "conscious", f"cat chain {chain_or_node}",
             detail={"chain_id": chain_or_node, "members": len(chain.get("members", []))})
        return rendered

    # Node address path — render node + provenance ancestors
    node = store.get(chain_or_node)
    if not node:
        return f"Not found: {chain_or_node}"

    ancestors = store.ancestors(node.addr)
    all_nodes = [node] + [n for n in ancestors if n.addr != node.addr]

    lines = [f"── Node {node.addr[:8]} [{node.type}] + {len(ancestors)} ancestor(s) ──"]
    chars = len(lines[0])
    budget = 20000

    for n in all_nodes:
        if chars >= budget:
            break
        agent = n.meta.get("agent_id")
        attr = f"[{agent}] " if agent else ""
        domain = n.meta.get("domain", "")
        domain_tag = f"[{domain}] " if domain else ""
        content = n.content

        code_ctx = n.meta.get("code_context") or {}
        if code_ctx and code:
            snapshot = code_ctx.get("snapshot", "")
            file_ref = code_ctx.get("file", "")
            lines_range = code_ctx.get("lines", [])
            if snapshot and file_ref:
                content += f"\n    ({file_ref}:{lines_range})\n    " + "\n    ".join(snapshot.splitlines()[:8])

        anchors = n.meta.get("anchors", [])
        drift_tag = " ⚠ drifted" if any(a.get("drifted") for a in anchors) else ""
        line = f"  {attr}{domain_tag}{content} [{n.addr[:8]}]{drift_tag}"
        lines.append(line)
        chars += len(line)

    emit("cat", "conscious", f"cat node {node.addr[:8]}",
         addresses=[node.addr],
         detail={"ancestors": len(ancestors)})
    return "\n".join(lines)




@mcp.tool()
def memory_chains(
    status: str = "",
    domain: str = "",
    agent_id: str = "",
    plan_root: str = "",
    store_scope: str = "both",
    limit: int = 20,
) -> str:
    """
    List and filter chains across the project and session stores.

    Arachne's primary read path. Shows plan chains first, then work chains,
    sorted by last-extended. Use memory_cat <chain_id> for full chain narrative.

    Args:
        status:      Filter by status: active | preliminary | stashed | archived |
                     superseded | all. Default: all.
        domain:      Filter by domain (e.g. "architecture"). Empty = all.
        agent_id:    Filter by owning agent (e.g. "opus-lead"). Empty = all.
        plan_root:   "true" → only planning chains. "false" → only work chains.
                     Empty = all.
        store_scope: Which store(s): "project" | "session" | "both". Default "both".
        limit:       Maximum chains to return (default 20).
    """
    if not STORE_IS_V2:
        return "memory_chains requires a v2 store (.mnemo/). Current store is v1-compat."

    from mnemo_chains import list_chains

    status_filter = None if (not status or status == "all") else {status}
    plan_root_filter = None
    if plan_root.lower() == "true":
        plan_root_filter = True
    elif plan_root.lower() == "false":
        plan_root_filter = False

    all_chains: list[dict] = []
    sources: list[tuple[str, object]] = []

    if store_scope in ("project", "both"):
        sources.append(("project", store))
    if store_scope in ("session", "both") and _session_store is not None:
        sources.append(("session", _session_store))

    for label, s in sources:
        try:
            c_list = list_chains(
                s,
                status_filter=status_filter,
                agent_id=agent_id or None,
                domain=domain or None,
                plan_root=plan_root_filter,
            )
            for c in c_list:
                c["_source"] = label
            all_chains.extend(c_list)
        except Exception:
            pass

    # Plan chains first, then by recency
    all_chains.sort(key=lambda c: (
        0 if c.get("plan_root") else 1,
        -(c.get("last_extended") or 0),
    ))
    shown = all_chains[:limit]

    if not shown:
        filters = [f"scope={store_scope}"]
        if status and status != "all":
            filters.append(f"status={status}")
        if domain:
            filters.append(f"domain={domain}")
        if agent_id:
            filters.append(f"agent={agent_id}")
        if plan_root:
            filters.append(f"plan_root={plan_root}")
        return f"No chains found ({', '.join(filters)})."

    lines = [f"Chains ({len(shown)} of {len(all_chains)}, scope={store_scope}):"]

    for chain in shown:
        chain_id = chain["chain_id"]
        ch_domain = chain.get("domain", "?")
        summary = chain.get("summary", "")
        n_members = len(chain.get("members", []))
        ch_status = chain.get("status", "active")
        ch_agent = chain.get("agent_id")
        ch_authority = chain.get("authority", 0.0)
        source = chain.get("_source", "?")
        is_plan = chain.get("plan_root", False)

        last_ext = chain.get("last_extended", 0)
        age_days = int((time.time() - last_ext) / 86400) if last_ext else 0
        age_str = f"{age_days}d ago" if age_days > 0 else "today"

        plan_marker = " [PLAN]" if is_plan else ""
        status_tag = f" [{ch_status}]" if ch_status != "active" else ""
        agent_tag = f" by {ch_agent} (auth={ch_authority:.1f})" if ch_agent else ""
        summary_tag = f" — {summary[:60]}" if summary else ""

        lines.append(
            f"  {chain_id}{plan_marker} [{ch_domain}]{status_tag}{agent_tag} "
            f"({n_members} nodes, {age_str}, {source}){summary_tag}"
        )

        if is_plan:
            tc = chain.get("team_config", [])
            sc = chain.get("success_criteria", {})
            fp = chain.get("friction_points", [])
            if tc:
                team_str = ", ".join(
                    f"{a.get('agent_id', '?')} ({a.get('role', '?')})" for a in tc
                )
                lines.append(f"    team: {team_str}")
            if sc.get("expected_outputs"):
                lines.append(
                    f"    target: {sc['expected_outputs']} {sc.get('output_type', '')}"
                )
            if fp:
                lines.append(f"    friction: {len(fp)} anticipated file overlap(s)")

    if len(all_chains) > limit:
        lines.append(f"  ... {len(all_chains) - limit} more (increase limit to see all)")

    emit("chains", "conscious",
         f"listed {len(shown)} chain(s)",
         detail={"total": len(all_chains), "status": status, "domain": domain,
                 "agent_id": agent_id, "plan_root": plan_root})
    return "\n".join(lines)


@mcp.tool()
def memory_promote(chain_id: str = "", all_preliminary: bool = False) -> str:
    """
    Promote a preliminary chain (or all preliminary chains) from the session
    store to the project store.

    Promoted chains become active project knowledge — they appear in recall,
    are indexed by TF-IDF, and persist across sessions. Before promoting,
    you're just thinking out loud. After promoting, it's committed.

    Args:
        chain_id: Specific chain ID to promote (ch_...). Use memory_chains
                  with status="preliminary" to see what's available.
        all_preliminary: If True, promote all preliminary chains at once.
                         Useful at natural breakpoints when your working
                         reasoning is solid enough to commit.
    """
    if not STORE_IS_V2:
        return "memory_promote requires a v2 store (.mnemo/)."
    if _session_store is None:
        return "No active session store."
    if not chain_id and not all_preliminary:
        return "Provide a chain_id or set all_preliminary=True."

    if all_preliminary:
        promoted = promote_all_preliminary(_session_store, store)
        if not promoted:
            return "No preliminary chains to promote."
        emit("promote", "conscious",
             f"promoted {len(promoted)} chain(s): {', '.join(promoted)}",
             detail={"chains": promoted, "all": True})
        lines = [f"Promoted {len(promoted)} chain(s) to project store:"]
        for cid in promoted:
            from mnemo_chains import get_chain
            chain = get_chain(store, cid)
            summary = chain.get("summary", "") if chain else ""
            members = len(chain.get("members", [])) if chain else 0
            lines.append(f"  {cid} ({members} nodes){f' — {summary}' if summary else ''}")
        return "\n".join(lines)

    # Single chain promotion
    from mnemo_chains import get_chain
    session_chain = get_chain(_session_store, chain_id)
    if not session_chain:
        return f"Chain not found in session store: {chain_id}\nUse memory_chains(status='preliminary') to list available chains."
    if session_chain.get("status") != "preliminary":
        return f"Chain {chain_id} is not preliminary (status: {session_chain.get('status')})."

    result = promote_chain(_session_store, store, chain_id)
    if not result:
        return f"Failed to promote chain {chain_id}."

    promoted_chain = get_chain(store, chain_id)
    members = len(promoted_chain.get("members", [])) if promoted_chain else 0
    summary = promoted_chain.get("summary", "") if promoted_chain else ""

    emit("promote", "conscious",
         f"promoted chain {chain_id}: {summary or '(no summary)'}",
         detail={"chain_id": chain_id, "members": members})
    return (
        f"Promoted {chain_id} to project store.\n"
        f"  Nodes: {members}\n"
        + (f"  Summary: {summary}\n" if summary else "")
        + f"Chain is now active project knowledge and will appear in recall."
    )


@mcp.tool()
def memory_session_status() -> str:
    """
    Show what's currently in the session store — preliminary chains,
    promoted work, and unpromoted node count.

    Use this to decide what to promote, stash, or discard before a
    session compress.
    """
    if not STORE_IS_V2:
        return "Session store requires a v2 store (.mnemo/)."
    if _session_store is None:
        return "No active session store."

    summary = session_summary(_session_store)
    lines = [
        f"Session: {_session_id}",
        f"  Nodes in session store: {summary['active_nodes']}",
        f"  Preliminary chains: {summary['preliminary_chains']}",
        f"  Already promoted: {summary['promoted_chains']}",
    ]
    if summary["preliminary_chain_list"]:
        lines.append("\n  Preliminary chains (not yet in project store):")
        for c in summary["preliminary_chain_list"]:
            lines.append(
                f"    {c['chain_id']} [{c['domain']}] ({c['members']} nodes)"
                + (f" — {c['summary']}" if c["summary"] else "")
            )
        lines.append(
            "\n  Use memory_promote to commit, memory_stash to shelve, "
            "or let them discard at session compress."
        )
    else:
        lines.append("  No preliminary chains. All session work has been promoted or the session store is empty.")

    emit("session_status", "conscious",
         f"session {_session_id}: {summary['preliminary_chains']} preliminary, "
         f"{summary['promoted_chains']} promoted",
         detail=summary)
    return "\n".join(lines)


@mcp.tool()
def memory_stash(chain_id: str, reason: str = "") -> str:
    """
    Shelve a reasoning chain without losing it.

    Marks the chain as stashed and removes its member nodes from the
    active set (they remain addressable). Useful for dead-end explorations
    or work that's not relevant right now but might be later.

    Use memory_chains(status="stashed") to see stashed chains.
    Use memory_stash_pop to restore.

    Args:
        chain_id: Chain ID to stash (ch_...)
        reason: Why you're stashing this (stored for future context)
    """
    if not STORE_IS_V2:
        return "memory_stash requires a v2 store (.mnemo/)."

    from mnemo_chains import stash_chain, get_chain

    chain = get_chain(store, chain_id)
    if not chain:
        return f"Chain not found: {chain_id}"

    ok = stash_chain(store, chain_id, reason=reason)
    if not ok:
        return f"Chain {chain_id} is already stashed."

    members = len(chain.get("members", []))
    emit("stash", "conscious", f"stashed chain {chain_id}: {reason or '(no reason)'}",
         detail={"chain_id": chain_id, "members": members, "reason": reason})
    return (
        f"Stashed chain {chain_id} ({members} nodes removed from active set).\n"
        f"Reason: {reason or '(none)'}\n"
        f"Restore with memory_stash_pop('{chain_id}')."
    )


@mcp.tool()
def memory_stash_pop(chain_id: str) -> str:
    """
    Restore a stashed chain to active status.

    Re-adds the chain's member nodes to the active set and sets
    chain status back to active.

    Args:
        chain_id: Chain ID to restore (ch_...)
    """
    if not STORE_IS_V2:
        return "memory_stash_pop requires a v2 store (.mnemo/)."

    from mnemo_chains import pop_chain, get_chain

    chain = get_chain(store, chain_id)
    if not chain:
        return f"Chain not found: {chain_id}"
    if chain.get("status") != "stashed":
        return f"Chain {chain_id} is not stashed (status: {chain.get('status')})."

    ok = pop_chain(store, chain_id)
    if not ok:
        return f"Failed to restore chain {chain_id}."

    members = len(chain.get("members", []))
    stash_reason = chain.get("stash_reason", "")
    emit("stash_pop", "conscious", f"restored chain {chain_id}",
         detail={"chain_id": chain_id, "members": members})
    return (
        f"Restored chain {chain_id} ({members} nodes back in active set).\n"
        + (f"Was stashed for: {stash_reason}" if stash_reason else "")
    )










@mcp.tool()
def memory_help(context: str = "claude-code") -> str:
    """
    Return the mnemo usage guide for the specified context.

    Tells you exactly when to call each tool, what triggers what,
    and what to avoid — no reasoning required, just follow the guide.

    Args:
        context: Which section to return.
                 "claude-code" — implementation agent (Claude Code CLI)
                 "desktop"     — Claude.ai desktop conversation
                 "cowork"      — Cowork / Arachne orchestration layer
                 "quick"       — quick reference table (all tools by trigger)
                 "all"         — full guide (all sections)
    """
    valid = set(_GUIDE_SECTIONS) | {"all"}
    if context not in valid:
        opts = ", ".join(f'"{v}"' for v in sorted(valid))
        return f"Unknown context '{context}'. Valid options: {opts}"

    result = _read_guide_section(context)
    emit("status", "conscious", f"memory_help: guide returned ({context})",
         detail={"context": context})
    return result


@mcp.tool()
def memory_write(path: str, content: str, claim: bool = True) -> str:
    """
    Write a file and update the memory tree.

    Replaces the native Write tool. Before writing:
    - Checks for agent conflicts on this file (auto-pings conflicting agents)
    - Surfaces what the tree knows about the file

    After writing:
    - Auto-claims the change as a history node (opt-out via claim=False)

    Args:
        path:    File path (relative to project root, or absolute)
        content: Full file content to write
        claim:   Auto-claim this change in the tree (default True)
    """
    project_root = _get_project_root()
    rel_path = _normalize_path(path, project_root)
    output = []

    # Write
    try:
        bytes_written = _fs_write(path, content, project_root)
    except Exception as e:
        return f"Write failed: {e}"

    output.insert(0, f"Wrote: {rel_path} ({bytes_written:,} bytes)")

    # Auto-claim
    if claim:
        summary_preview = content[:120].replace("\n", " ").strip()
        summary = f"Wrote {rel_path}: {summary_preview}{'...' if len(content) > 120 else ''}"
        try:
            addr = _auto_claim(
                store, _session_store, path, summary,
                session_id=_session_id,
            )
            output.append(f"Claimed: [{addr[:8]}]")
        except Exception:
            pass

    emit("write", "conscious", f"write: {rel_path} ({bytes_written}b)",
         detail={"path": rel_path, "bytes": bytes_written, "claimed": claim})

    return "\n".join(output)


@mcp.tool()
def memory_edit(
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    claim: bool = True,
) -> str:
    """
    Edit a file and update the memory tree.

    Replaces the native Edit tool. Before editing:
    - Checks for agent conflicts on this file (auto-pings conflicting agents)
    - Surfaces stale anchors that may be affected by the change

    After editing:
    - Auto-claims the change as a history node (opt-out via claim=False)

    Args:
        path:        File path (relative to project root, or absolute)
        old_string:  Exact text to replace (must be unique unless replace_all=True)
        new_string:  Replacement text
        replace_all: Replace all occurrences (default False)
        claim:       Auto-claim this change in the tree (default True)
    """
    project_root = _get_project_root()
    rel_path = _normalize_path(path, project_root)
    output = []

    # Apply edit
    try:
        _, replacements = _fs_edit(path, old_string, new_string, replace_all, project_root)
    except ValueError as e:
        return f"Edit failed: {e}"
    except Exception as e:
        return f"Edit failed: {e}"

    old_p = old_string[:50].replace("\n", "↵")
    new_p = new_string[:50].replace("\n", "↵")
    output.insert(0, f"Edited: {rel_path} ({replacements} replacement{'s' if replacements != 1 else ''})")
    output.append(f"  - {old_p}{'…' if len(old_string) > 50 else ''}")
    output.append(f"  + {new_p}{'…' if len(new_string) > 50 else ''}")

    # Stale anchor check
    try:
        stale = _check_stale_anchors(store, path, project_root)
        if stale:
            output.append(f"⚠ {len(stale)} stale anchor{'s' if len(stale) != 1 else ''} — nodes may describe code that changed:")
            for s in stale[:3]:
                n = s["node"]
                output.append(f"  [{n.addr[:8]}] {n.content[:60]} ({s['status']})")
            if len(stale) > 3:
                output.append(f"  … {len(stale) - 3} more — run memory_verify for full details")
    except Exception:
        pass

    # Auto-claim
    if claim:
        summary = f"Edited {rel_path}: {old_p!r} → {new_p!r}"
        try:
            addr = _auto_claim(
                store, _session_store, path, summary,
                session_id=_session_id,
            )
            output.append(f"Claimed: [{addr[:8]}]")
        except Exception:
            pass

    emit("edit", "conscious", f"edit: {rel_path} ({replacements} replacements)",
         detail={"path": rel_path, "replacements": replacements, "claimed": claim})

    return "\n".join(output)


@mcp.tool()
def memory_glob(pattern: str, path: str = ".") -> str:
    """
    Glob for files with tree coverage annotation.

    Replaces the native Glob tool. Shows how many mnemo nodes reference
    each matched file — immediately visible what's known vs unexplored.

    Args:
        pattern: Glob pattern (e.g. "**/*.py", "src/**/*.ts")
        path:    Directory to search in (default: project root)
    """
    project_root = _get_project_root()

    try:
        paths = _fs_glob(pattern, path, project_root)
    except Exception as e:
        return f"Glob failed: {e}"

    result = _format_glob_with_coverage(paths, store, project_root)

    emit("glob", "conscious", f"glob: {pattern} ({len(paths)} files)",
         detail={"pattern": pattern, "path": path, "count": len(paths)})

    return result




@mcp.tool()
def memory_blame(target: str, line: int = 0) -> str:
    """
    Attribution decomposition — who said what, when, and why.

    Like "git blame" for the reasoning layer. Three modes:

    - Chain ID (ch_...): for each node in the chain — agent, timestamp,
      confidence at creation vs. now (after decay), and any anchors.

    - File path (e.g. "auth.py"): all reasoning nodes anchored to this file,
      grouped by agent, with chain context.

    - File path + line (e.g. "auth.py" with line=47): all reasoning nodes
      anchored to that specific line (±10 line window), showing the full
      multi-agent annotation for that location.

    - Node address: if it's a compress node, decomposes back to constituent
      leaves — "which original claims contributed to this summary."

    Args:
        target: Chain ID (ch_...), file path, or node address
        line:   Line number for file blame (0 = whole file)
    """
    if not STORE_IS_V2:
        return "memory_blame requires a v2 store (.mnemo/)."

    from mnemo_chains import get_chain

    # ── Chain blame ──
    if target.startswith("ch_"):
        chain = get_chain(store, target)
        if not chain:
            return f"Chain not found: {target}"

        members = chain.get("members", [])
        lines = [
            f"── blame: chain {target} ({len(members)} nodes) ──",
            f"Domain: {chain.get('domain', '?')}  Status: {chain.get('status', '?')}",
        ]
        if chain.get("agent_id"):
            lines.append(f"Owner: {chain['agent_id']}  Authority: {chain.get('authority', '?')}")

        for i, addr in enumerate(members):
            node = store.get(addr)
            if not node:
                lines.append(f"  [{i+1}] {addr[:8]} (missing)")
                continue

            agent = node.meta.get("agent_id", "unattributed")
            created = node.meta.get("created") or node.created
            ts = time.strftime("%Y-%m-%dT%H:%M", time.localtime(created)) if created else "?"
            conf = node.meta.get("confidence", "?")
            domain = node.meta.get("domain", "")
            anchors = node.meta.get("anchors", [])
            anchor_tag = ""
            if anchors:
                files = list({a.get("file") or a.get("path", "") for a in anchors if a.get("file") or a.get("path")})
                anchor_tag = f"  @ {', '.join(f for f in files if f)}"
            is_superseded = addr not in store.get_active()
            stale_tag = " [superseded]" if is_superseded else ""

            lines.append(
                f"  [{i+1}] {addr[:8]} {ts}  [{agent}]{stale_tag}  conf={conf}"
                f"  {domain}{anchor_tag}"
            )
            lines.append(f"       {node.content[:120]}")

        emit("blame", "conscious", f"blame chain {target}",
             detail={"chain_id": target, "members": len(members)})
        return "\n".join(lines)

    # ── File blame (with optional line filter) ──
    if "." in target and not target.startswith("0") and len(target) != 12:
        # Looks like a file path
        from mnemo_anchor import get_anchors_for_file
        LINE_WINDOW = 10  # lines of proximity for blame filtering

        filepath = target
        anchored = get_anchors_for_file(filepath, store)

        # Filter by line if requested
        if line > 0:
            filtered = []
            for item in anchored:
                hint = item["anchor"].get("line_hint")
                if hint is None or abs(hint - line) <= LINE_WINDOW:
                    filtered.append(item)
            anchored = filtered
            header = f"── blame: {filepath}:{line} ({len(anchored)} node(s) within ±{LINE_WINDOW} lines) ──"
        else:
            header = f"── blame: {filepath} ({len(anchored)} anchored node(s)) ──"

        if not anchored:
            return f"{header}\nNo reasoning nodes anchored to this {'line' if line else 'file'}."

        lines_out = [header]
        # Group by agent
        by_agent: dict[str, list] = {}
        for item in anchored:
            node = item["node"]
            agent = node.meta.get("agent_id", "unattributed")
            by_agent.setdefault(agent, []).append(item)

        for agent, items in sorted(by_agent.items()):
            lines_out.append(f"\n  [{agent}]")
            for item in items:
                node = item["node"]
                anchor = item["anchor"]
                chain_ids = node.meta.get("chains", [])
                chain_tag = f" chain={chain_ids[0]}" if chain_ids else ""
                hint = anchor.get("line_hint")
                line_tag = f":{hint}" if hint else ""
                scope = anchor.get("scope", "")
                scope_tag = f" ({scope})" if scope else ""
                lines_out.append(
                    f"    {node.addr[:8]}{line_tag}{scope_tag}{chain_tag}"
                )
                lines_out.append(f"    {node.content[:120]}")

        emit("blame", "conscious", f"blame file {filepath}" + (f":{line}" if line else ""),
             detail={"file": filepath, "line": line, "nodes": len(anchored)})
        return "\n".join(lines_out)

    # ── Node blame (compress decomposition) ──
    node = store.get(target)
    if not node:
        return f"Not found: {target}"

    if node.type == "compress":
        # Decompose compress node to leaves
        lines_out = [f"── blame: compress {node.addr[:8]} ──"]
        lines_out.append(f"Summary: {node.content[:200]}")
        lines_out.append(f"Inputs ({len(node.inputs)}):")
        for inp_addr in node.inputs:
            inp = store.get(inp_addr)
            if not inp:
                lines_out.append(f"  {inp_addr[:8]} (missing — may have been superseded)")
                continue
            agent = inp.meta.get("agent_id", "unattributed")
            conf = inp.meta.get("confidence", "?")
            lines_out.append(f"  {inp_addr[:8]} [{agent}] conf={conf}: {inp.content[:100]}")
        emit("blame", "conscious", f"blame compress {node.addr[:8]}",
             addresses=[node.addr], detail={"inputs": len(node.inputs)})
        return "\n".join(lines_out)

    # Plain leaf/supersede
    agent = node.meta.get("agent_id", "unattributed")
    created = node.meta.get("created") or node.created
    ts = time.strftime("%Y-%m-%dT%H:%M", time.localtime(created)) if created else "?"
    chain_ids = node.meta.get("chains", [])
    lines_out = [
        f"── blame: {node.addr[:8]} [{node.type}] ──",
        f"Agent:   {agent}",
        f"Created: {ts}",
        f"Domain:  {node.meta.get('domain', '?')}",
        f"Conf:    {node.meta.get('confidence', '?')}",
    ]
    if chain_ids:
        lines_out.append(f"Chains:  {', '.join(chain_ids)}")
    if node.inputs:
        lines_out.append(f"Inputs:  {', '.join(a[:8] for a in node.inputs)}")
    anchors = node.meta.get("anchors", [])
    if anchors:
        for a in anchors:
            atype = a.get("type", "?")
            ref = a.get("file") or a.get("path") or a.get("name", "")
            extra = f":{a['line_hint']}" if a.get("line_hint") else ""
            lines_out.append(f"Anchor:  [{atype}] {ref}{extra}")
    lines_out.append(f"\n{node.content}")
    emit("blame", "conscious", f"blame node {node.addr[:8]}",
         addresses=[node.addr])
    return "\n".join(lines_out)


@mcp.tool()
def memory_log(
    topic: str = "",
    agent_filter: str = "",
    domain: str = "",
    by_chains: bool = False,
    reconciliations: bool = False,
    pings: bool = False,
    pending_only: bool = False,
    limit: int = 50,
) -> str:
    """
    Chronological history of the project tree, including superseded nodes.

    Unlike memory_search (which searches active nodes only), memory_log
    shows the full history — what was believed, when it changed, and who
    changed it. Dead ends, corrections, and reconciliations are all visible.

    Filters:
    - topic:          Keyword filter on node content (substring match)
    - agent_filter:   Show only nodes from this agent (e.g. "opus-lead")
    - domain:         Filter by domain (e.g. "architecture")
    - by_chains:      Group output by chain rather than chronology
    - reconciliations: Show only reconciliation nodes (multi-agent merges)
    - pings:          Show ping nodes (inter-agent messages)
    - pending_only:   With pings=True, show only unacknowledged pings
    - limit:          Max entries (default 50)

    Args:
        topic:          Keyword to filter node content
        agent_filter:   Only show nodes from this agent
        domain:         Only show nodes in this domain
        by_chains:      Group by chain instead of chronology
        reconciliations: Show only reconciliation events
        pings:          Show ping nodes
        pending_only:   Combined with pings — only unacknowledged
        limit:          Max nodes to show
    """
    # Collect all nodes (including non-active = superseded history)
    all_nodes = store.all_nodes()

    # Apply filters
    filtered = []
    for node in all_nodes:
        meta = node.meta

        if reconciliations:
            # Reconciliation nodes have source="reconciliation" or content mentions it
            src = meta.get("source", "")
            if src != "reconciliation" and "reconcil" not in node.content.lower():
                continue

        if pings:
            ping_meta = meta.get("ping")
            if not ping_meta:
                continue
            if pending_only and ping_meta.get("acked_at") is not None:
                continue

        if agent_filter and meta.get("agent_id") != agent_filter:
            continue
        if domain and meta.get("domain") != domain:
            continue
        if topic and topic.lower() not in node.content.lower():
            continue

        filtered.append(node)

    if not filtered:
        desc = " ".join(filter(None, [
            topic, agent_filter and f"agent={agent_filter}",
            domain and f"domain={domain}",
            "reconciliations" if reconciliations else "",
            "pings" if pings else "",
        ]))
        return f"No entries found{(' for: ' + desc) if desc else ''}."

    # Sort chronologically
    filtered.sort(key=lambda n: n.created)

    active_set = store.get_active()

    if by_chains:
        # Group by chain membership
        from mnemo_chains import get_chain, list_chains as _list_chains
        chain_map: dict[str, list] = {}
        unchained = []
        for node in filtered[:limit]:
            chain_ids = node.meta.get("chains", [])
            if chain_ids:
                chain_map.setdefault(chain_ids[0], []).append(node)
            else:
                unchained.append(node)

        parts = []
        for chain_id, nodes in chain_map.items():
            chain = get_chain(store, chain_id)
            summary = chain.get("summary", chain_id) if chain else chain_id
            agent_tag = f" [{chain.get('agent_id')}]" if chain and chain.get("agent_id") else ""
            part = [f"── {summary}{agent_tag} ({len(nodes)} nodes) ──"]
            for node in nodes:
                _append_log_entry(part, node, active_set)
            parts.append("\n".join(part))

        if unchained:
            part = [f"── Unchained ({len(unchained)} nodes) ──"]
            for node in unchained:
                _append_log_entry(part, node, active_set)
            parts.append("\n".join(part))

        emit("log", "conscious",
             f"log by_chains: {len(filtered)} entries",
             detail={"topic": topic, "agent": agent_filter, "domain": domain})
        return "\n\n".join(parts) or "(empty)"

    # Chronological output
    total = len(filtered)
    shown = filtered[:limit]
    lines = [
        f"── memory log: {total} entries"
        + (f" (showing {limit})" if total > limit else "")
        + " ──"
    ]
    for node in shown:
        _append_log_entry(lines, node, active_set)

    if total > limit:
        lines.append(f"... {total - limit} more entries. Increase limit or add filters.")

    emit("log", "conscious",
         f"log: {total} entries, topic={topic!r}",
         detail={"topic": topic, "agent": agent_filter, "domain": domain,
                 "reconciliations": reconciliations, "pings": pings})
    return "\n".join(lines)


def _append_log_entry(lines: list, node, active_set: set) -> None:
    """Append a single node as a log line."""
    ts = time.strftime("%m-%d %H:%M", time.localtime(node.created))
    agent = node.meta.get("agent_id", "")
    agent_tag = f"[{agent}] " if agent else ""
    domain = node.meta.get("domain", "")
    domain_tag = f"[{domain}] " if domain else ""
    is_active = node.addr in active_set
    status = "" if is_active else " [superseded]"
    node_type = "" if node.type == "leaf" else f" [{node.type}]"

    # Ping special display
    ping_meta = node.meta.get("ping")
    if ping_meta:
        urgency = ping_meta.get("urgency", "low")
        target = ping_meta.get("target_agent", "?")
        acked = " ✓" if ping_meta.get("acked_at") else " ○"
        lines.append(
            f"  {ts}  ping→{target} [{urgency}]{acked}{status}"
            f"  {node.content[:100]}"
        )
        return

    lines.append(
        f"  {ts}  {agent_tag}{domain_tag}{node.addr[:8]}{node_type}{status}"
        f"  {node.content[:120]}"
    )


@mcp.tool()
def memory_rebase(node_addr: str, dry_run: bool = False) -> str:
    """
    Propagate implications when foundational knowledge changes.

    When a key node is updated (superseded), other nodes that depend on
    it — through links, chain membership, file references, or content
    similarity — may have become stale. memory_rebase finds them.

    Does NOT auto-update anything. Returns a review list for agents/human
    to act on. Each dependent node gets a staleness reason so you know
    why it was flagged.

    Use dry_run=True to see what would be flagged without writing anything.
    When dry_run=False (default), flagged nodes get a "potentially_stale"
    meta tag so they surface in future recall.

    Args:
        node_addr: Address (or prefix) of the node that changed
        dry_run:   If True, report without modifying any nodes
    """
    node = store.get(node_addr)
    if not node:
        return f"Not found: {node_addr}"

    active_set = store.get_active()
    dependents: dict[str, dict] = {}  # addr → {node, reasons}

    # Gather facts about the target node for matching
    target_files = set()
    for a in node.meta.get("anchors", []):
        f = a.get("file") or a.get("path", "")
        if f:
            target_files.add(f)
    target_chains = set(node.meta.get("chains", []))
    target_domain = node.meta.get("domain", "")

    for addr in active_set:
        if addr == node.addr:
            continue
        candidate = store.get(addr)
        if not candidate:
            continue

        reasons = []

        # 1. Direct input dependency (candidate derived from this node)
        if node.addr in candidate.inputs:
            reasons.append("direct input — derived from this node")

        # 2. Explicit links
        cand_links = candidate.meta.get("links", [])
        if any(lnk.get("target") == node.addr for lnk in cand_links):
            reasons.append("linked (depends_on / caused_by / relates_to)")

        # 3. Same chain — chain members may rely on shared reasoning context
        cand_chains = set(candidate.meta.get("chains", []))
        shared_chains = target_chains & cand_chains
        if shared_chains:
            reasons.append(f"same chain(s): {', '.join(shared_chains)}")

        # 4. Same file anchors — co-located reasoning may be affected
        cand_files = set()
        for a in candidate.meta.get("anchors", []):
            f = a.get("file") or a.get("path", "")
            if f:
                cand_files.add(f)
        shared_files = target_files & cand_files
        if shared_files and not shared_chains:  # only if not already flagged by chain
            reasons.append(f"anchored to same file(s): {', '.join(shared_files)}")

        # 5. Content reference — candidate mentions the target's content keyword
        # (lightweight: check if the target's first 5 words appear in candidate)
        target_words = node.content.split()[:5]
        if (len(target_words) >= 3
                and all(w.lower() in candidate.content.lower() for w in target_words[:3])):
            if not reasons:  # only flag as content ref if no stronger signal
                reasons.append("content reference — mentions similar concept")

        if reasons:
            dependents[addr] = {"node": candidate, "reasons": reasons}

    if not dependents:
        return (
            f"No dependents found for {node.addr[:8]}.\n"
            f"Node: {node.content[:120]}"
        )

    # Flag dependents (unless dry_run)
    flagged_count = 0
    if not dry_run:
        for addr, info in dependents.items():
            dep_node = info["node"]
            if not dep_node.meta.get("potentially_stale"):
                dep_node.meta["potentially_stale"] = {
                    "flagged_by": node.addr,
                    "flagged_at": time.time(),
                    "reasons": info["reasons"],
                }
                store.put(dep_node)
                flagged_count += 1

    # Format report
    lines = [
        f"── rebase: {node.addr[:8]}"
        + (" (dry run)" if dry_run else f" ({flagged_count} nodes flagged)")
        + " ──",
        f"Changed node: {node.content[:120]}",
        f"Found {len(dependents)} dependent(s):\n",
    ]

    # Group by reason type for readability
    for addr, info in sorted(dependents.items(),
                              key=lambda x: len(x[1]["reasons"]), reverse=True):
        dep = info["node"]
        agent = dep.meta.get("agent_id", "")
        agent_tag = f"[{agent}] " if agent else ""
        chains = dep.meta.get("chains", [])
        chain_tag = f" chain={chains[0]}" if chains else ""
        lines.append(
            f"  {dep.addr[:8]} {agent_tag}{dep.content[:100]}{chain_tag}"
        )
        for r in info["reasons"]:
            lines.append(f"    ↳ {r}")

    if not dry_run and flagged_count:
        lines.append(
            f"\n{flagged_count} node(s) tagged 'potentially_stale'. "
            "They will surface in future recall with a staleness signal. "
            "Use memory_update to resolve each one, or memory_reinforce to confirm still valid."
        )
    elif dry_run:
        lines.append(
            f"\nDry run — nothing modified. "
            f"Run without dry_run=True to flag {len(dependents)} node(s)."
        )

    emit("rebase", "conscious",
         f"rebase {node.addr[:8]}: {len(dependents)} dependents",
         addresses=[node.addr],
         detail={"dependents": len(dependents), "flagged": flagged_count,
                 "dry_run": dry_run})
    return "\n".join(lines)


@mcp.tool()
def memory_spatial(
    file: str,
    start_line: int = 0,
    end_line: int = 0,
    agent_filter: str = "",
) -> str:
    """
    Spatial retrieval — query by file location rather than topic.

    Complements semantic recall: instead of asking "what do we know about
    auth?", ask "what has been said about lines 40-60 of auth.py?"

    Returns all reasoning nodes anchored to the specified file and line
    range, from all agents, with chain context. Shows the multi-agent
    annotation history for a specific piece of code.

    Both modes:
    - Whole file: memory_spatial("auth.py") — everything anchored to this file
    - Line range: memory_spatial("auth.py", start_line=40, end_line=60)
      → all nodes with line_hint within that range (±10 tolerance)

    Args:
        file:         File path (relative or basename, e.g. "auth.py")
        start_line:   Start of line range (0 = no lower bound)
        end_line:     End of line range (0 = no upper bound)
        agent_filter: Only show nodes from this agent
    """
    if not STORE_IS_V2:
        return "memory_spatial requires a v2 store (.mnemo/)."

    from mnemo_anchor import get_anchors_for_file
    LINE_WINDOW = 10  # lines of proximity for spatial filtering
    _file_anchors = lambda n: [a for a in n.meta.get("anchors", []) if a.get("type") == "file"]
    _basename = os.path.basename

    # Primary lookup via file index (content_hash anchors)
    anchored = get_anchors_for_file(file, store)

    # Supplement: scan active set for grep/file anchor types not in the index
    import os as _os
    target_basename = _os.path.basename(file)
    active = store.get_active()
    indexed_addrs = {item["node"].addr for item in anchored}
    for addr in active:
        if addr in indexed_addrs:
            continue
        node = store.get(addr)
        if not node:
            continue
        for fa in _file_anchors(node):
            if _basename(fa["file"]) == target_basename:
                anchored.append({
                    "node": node,
                    "anchor": fa,
                    "anchor_idx": 0,
                })
                indexed_addrs.add(addr)
                break  # one entry per node per file

    if not anchored:
        return f"No reasoning nodes anchored to '{file}'."

    # Filter by line range
    if start_line > 0 or end_line > 0:
        filtered = []
        for item in anchored:
            hint = item["anchor"].get("line_hint")
            if hint is None:
                continue  # no line info — can't place in range
            in_range = True
            if start_line > 0 and hint < (start_line - LINE_WINDOW):
                in_range = False
            if end_line > 0 and hint > (end_line + LINE_WINDOW):
                in_range = False
            if in_range:
                filtered.append(item)
        anchored = filtered

    # Filter by agent
    if agent_filter:
        anchored = [i for i in anchored
                    if i["node"].meta.get("agent_id") == agent_filter]

    if not anchored:
        range_desc = ""
        if start_line or end_line:
            range_desc = f":{start_line}-{end_line}"
        return f"No nodes found for '{file}{range_desc}'" + (
            f" (agent={agent_filter})" if agent_filter else ""
        )

    # Sort by line hint
    anchored.sort(key=lambda i: i["anchor"].get("line_hint") or 0)

    range_str = ""
    if start_line or end_line:
        range_str = f":{start_line or '?'}-{end_line or '?'}"
    header = f"── spatial: {file}{range_str} ({len(anchored)} node(s)) ──"
    lines = [header]

    # Group by agent for multi-agent view
    from collections import defaultdict
    by_agent: dict[str, list] = defaultdict(list)
    for item in anchored:
        agent = item["node"].meta.get("agent_id", "unattributed")
        by_agent[agent].append(item)

    for agent, items in sorted(by_agent.items()):
        if len(by_agent) > 1:
            lines.append(f"\n  [{agent}]")
        for item in items:
            node = item["node"]
            anchor = item["anchor"]
            hint = anchor.get("line_hint")
            scope = anchor.get("scope", "")
            chain_ids = node.meta.get("chains", [])

            line_tag = f":{hint}" if hint else ""
            scope_tag = f" ({scope})" if scope else ""
            chain_tag = f"  chain={chain_ids[0]}" if chain_ids else ""
            ts = time.strftime("%m-%d", time.localtime(node.created))

            lines.append(
                f"    {ts} {node.addr[:8]}{line_tag}{scope_tag}{chain_tag}"
            )
            lines.append(f"    {node.content[:140]}")

    emit("spatial", "conscious",
         f"spatial {file}{range_str}: {len(anchored)} node(s)",
         detail={"file": file, "start": start_line, "end": end_line,
                 "nodes": len(anchored)})
    return "\n".join(lines)

if __name__ == "__main__":
    mcp.run(transport="stdio")
