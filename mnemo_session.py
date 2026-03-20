"""
mnemo_session.py — Session store lifecycle

A session store is ephemeral working memory for a single agent session.
It holds reasoning chains that haven't been committed to the project store
yet — preliminary work that may be promoted, stashed, or discarded.

== Why this exists ==

In v1, every memory_claim immediately became a permanent project node.
This created pressure to only store "finished" thoughts, losing the
intermediate reasoning that's often the most valuable part. The session
store lets an agent think out loud without polluting the project tree.

== Store layout ==

    .mnemo/
    ├── sessions/
    │   ├── s_a7f3c2d1/          ← current session
    │   │   ├── nodes/
    │   │   ├── active.json
    │   │   └── chains.json      ← preliminary chains only
    │   └── ...
    └── session_archive/
        ├── s_b91e4d3f/          ← archived sessions (recoverable)
        └── ...

== Lifecycle ==

    session start
      └── new_session_id() → get_session_store() → session Store object

    work happens
      └── claims with preliminary=True go to session store
          chains build with status="preliminary"

    promote
      └── promote_chain() copies nodes + chain to project store
          chain status: preliminary → active

    session compress
      └── unpromoted chains: archive_session() moves dir to session_archive/
          promoted chains: already in project store, session copy is redundant

    discard (noise)
      └── session dir is archived (not deleted) — content-addressed files
          are cheap; recovery is valuable

== Node format ==

Identical to project store nodes. Promotion is a file copy + active.json
update. Content-addressing means the same node address in both stores
points to identical content — no reconciliation needed.

Preliminary chains carry status="preliminary" in chains.json. After
promotion, the project store chain gets status="active".
"""

import json
import secrets
import shutil
import time
from pathlib import Path
from typing import Optional

from mnemo import Store, Node


# ===================================================================
# Session ID
# ===================================================================

def new_session_id() -> str:
    """Generate a unique session ID: s_ + 8 hex chars."""
    return "s_" + secrets.token_hex(4)


# ===================================================================
# Session store access
# ===================================================================

def session_store_path(base_store: Store, session_id: str,
                       agent_id: Optional[str] = None) -> Path:
    """
    Return the path for a session store.

    Single-agent: .mnemo/sessions/<session_id>/
    Multi-agent:  .mnemo/sessions/<session_id>/<agent_id>/
    """
    sessions_dir = base_store.root / "sessions"
    if agent_id:
        return sessions_dir / session_id / agent_id
    return sessions_dir / session_id


def get_session_store(base_store: Store, session_id: str,
                      agent_id: Optional[str] = None) -> Store:
    """
    Return a Store for the given session. Creates the directory if needed.

    The session store is a full Store object — same format as the project
    store, but scoped to this session. No TF-IDF index is built (linear
    scan is fast enough for <50 nodes per session).
    """
    path = session_store_path(base_store, session_id, agent_id)
    path.mkdir(parents=True, exist_ok=True)

    # Initialize chains.json if not present
    chains_path = path / "chains.json"
    if not chains_path.exists():
        chains_path.write_text("{}\n", encoding="utf-8")

    return Store(path)


def load_or_create_session(base_store: Store,
                           session_state_path: Optional[Path] = None
                           ) -> tuple[str, Store]:
    """
    Load the current session ID from session_state.json, or create a new one.

    Returns (session_id, session_store).
    """
    state_path = session_state_path or (base_store.root / "session_state.json")

    session_id = None
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            saved_at = state.get("saved_at", 0)
            # Reuse session if less than 2 hours old
            if time.time() - saved_at < 7200:
                session_id = state.get("session_id")
        except (json.JSONDecodeError, OSError):
            pass

    if not session_id:
        session_id = new_session_id()

    return session_id, get_session_store(base_store, session_id)


# ===================================================================
# Preliminary chain management
# ===================================================================

def create_preliminary_chain(session_store: Store, head_addr: str,
                             domain: str = "", summary: str = "",
                             agent_id: Optional[str] = None) -> str:
    """
    Create a chain with status="preliminary" in the session store.
    Wraps mnemo_chains.create_chain with the preliminary status.
    """
    from mnemo_chains import create_chain
    return create_chain(
        session_store, head_addr,
        domain=domain, summary=summary,
        agent_id=agent_id, status="preliminary",
    )


def list_preliminary_chains(session_store: Store) -> list[dict]:
    """Return all preliminary chains in the session store."""
    from mnemo_chains import list_chains
    return list_chains(session_store, status_filter={"preliminary"})


# ===================================================================
# Promotion: session store → project store
# ===================================================================

def promote_nodes(session_store: Store, project_store: Store,
                  addrs: list[str]) -> list[str]:
    """
    Copy nodes from session store to project store and add them to active.

    Since nodes are content-addressed, copying a node that already exists
    in the project store is a no-op (same address = same file). Returns the
    list of addresses that were promoted (some may have already existed).

    Does NOT modify the session store — session copy remains until archived.
    """
    promoted = []
    project_active = project_store.get_active()

    for addr in addrs:
        node = session_store.get(addr)
        if not node:
            continue

        # Copy node file to project store
        src = session_store.nodes_dir / f"{addr}.json"
        dst = project_store.nodes_dir / f"{addr}.json"
        if not dst.exists():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

        project_active.add(addr)
        promoted.append(addr)

    project_store.set_active(project_active)

    # Re-register any content_hash anchors in the project store's file index
    if promoted:
        try:
            from mnemo_anchor import update_file_index
            for addr in promoted:
                node = project_store.get(addr)
                if node and node.meta.get("anchors"):
                    update_file_index(project_store, node)
        except Exception:
            pass

    return promoted


def promote_chain(session_store: Store, project_store: Store,
                  chain_id: str) -> Optional[str]:
    """
    Promote a preliminary chain from the session store to the project store.

    Steps:
    1. Copy all member nodes to the project store
    2. Add them to project active set
    3. Write the chain to project store's chains.json with status="active"
    4. Mark the session chain as status="archived" (it stays in session store)

    Returns the chain_id on success, None if chain not found.
    """
    from mnemo_chains import get_chain, get_chains, _set_chains

    # Get chain from session store
    chain = get_chain(session_store, chain_id)
    if not chain:
        return None

    # Promote member nodes
    members = chain.get("members", [])
    promote_nodes(session_store, project_store, members)

    # Write chain to project store with status=active
    project_chains_path = project_store.root / "chains.json"
    if project_chains_path.exists():
        try:
            project_chains = json.loads(
                project_chains_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            project_chains = {}
    else:
        project_chains = {}

    promoted_chain = dict(chain)
    promoted_chain["status"] = "active"
    promoted_chain["promoted_at"] = time.time()
    promoted_chain["promoted_from_session"] = str(session_store.root)
    project_chains[chain_id] = promoted_chain
    project_chains_path.write_text(
        json.dumps(project_chains, indent=2), encoding="utf-8"
    )

    # Mark session chain as archived
    session_chains = get_chains(session_store)
    if chain_id in session_chains:
        session_chains[chain_id]["status"] = "archived"
        _set_chains(session_store, session_chains)

    # Update node meta in project store to reflect active chain membership
    for addr in members:
        node = project_store.get(addr)
        if node:
            existing_chains = node.meta.get("chains", [])
            if chain_id not in existing_chains:
                node.meta["chains"] = existing_chains + [chain_id]
                project_store.put(node)

    return chain_id


def promote_all_preliminary(session_store: Store,
                            project_store: Store) -> list[str]:
    """
    Promote all preliminary chains in the session store to the project store.
    Returns list of promoted chain IDs.
    """
    preliminary = list_preliminary_chains(session_store)
    promoted = []
    for chain in preliminary:
        chain_id = chain["chain_id"]
        result = promote_chain(session_store, project_store, chain_id)
        if result:
            promoted.append(chain_id)
    return promoted


# ===================================================================
# Session archiving and cleanup
# ===================================================================

def archive_session(base_store: Store, session_id: str,
                    agent_id: Optional[str] = None) -> Optional[Path]:
    """
    Move the session directory to session_archive/ for recovery.

    Returns the archive path, or None if the session dir doesn't exist.
    Does NOT delete the source — the move IS the cleanup.
    """
    src = session_store_path(base_store, session_id, agent_id)
    if not src.exists():
        return None

    archive_dir = base_store.root / "session_archive"
    archive_dir.mkdir(exist_ok=True)

    # Add timestamp to avoid collisions if same session_id reused
    ts = int(time.time())
    dest_name = f"{session_id}_{ts}" if agent_id is None else f"{session_id}_{agent_id}_{ts}"
    dest = archive_dir / dest_name

    shutil.move(str(src), str(dest))
    return dest


def session_summary(session_store: Store) -> dict:
    """
    Return a summary of the current session store state.

    Useful for the session compress prompt — shows what's preliminary,
    what's already been promoted, and what's available to promote.
    """
    from mnemo_chains import list_chains

    active = session_store.get_active()
    all_chains = list_chains(session_store)

    preliminary = [c for c in all_chains if c.get("status") == "preliminary"]
    archived = [c for c in all_chains if c.get("status") == "archived"]

    return {
        "session_path": str(session_store.root),
        "active_nodes": len(active),
        "total_chains": len(all_chains),
        "preliminary_chains": len(preliminary),
        "promoted_chains": len(archived),  # archived in session = promoted to project
        "preliminary_chain_list": [
            {
                "chain_id": c["chain_id"],
                "domain": c.get("domain", ""),
                "summary": c.get("summary", ""),
                "members": len(c.get("members", [])),
            }
            for c in preliminary
        ],
    }


# ===================================================================
# Session GC
# ===================================================================

def gc_sessions(base_store: Store, max_age_days: int = 30) -> int:
    """
    Purge session archives older than max_age_days.

    Only removes archived sessions (in session_archive/), not active ones.
    Returns the number of session archives purged.
    """
    archive_dir = base_store.root / "session_archive"
    if not archive_dir.exists():
        return 0

    cutoff = time.time() - (max_age_days * 86400)
    purged = 0

    for entry in archive_dir.iterdir():
        if entry.is_dir():
            mtime = entry.stat().st_mtime
            if mtime < cutoff:
                shutil.rmtree(str(entry))
                purged += 1

    return purged
