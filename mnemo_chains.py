"""
mnemo_chains.py — Chain data model and CRUD

A chain is an ordered sequence of nodes representing a coherent line of
reasoning. Chains are lightweight metadata — they organize existing nodes
without creating new ones.

== Chain fields ==

chain_id        Stable random ID: "ch_" + 12-char hex. Never changes.
head            Address of the most recent (concluding) node.
tail            Address of the originating node.
members         Ordered list of node addresses, tail → head.
domain          Primary domain (inherited from majority of members).
summary         One-line description.
score           Pre-computed relevance (updated on retrieval).
last_extended   Timestamp of last node added.
status          preliminary | active | stashed | archived | superseded
agent_id        Owning agent (null = unattributed / single-agent legacy).
authority       Inherited from the owning agent's role (0.0–1.0).
stash_reason    Why the chain was stashed (if status == stashed).

== Dual indexing ==

Membership is indexed in both directions:
  chains.json         chain_id → chain dict (chain → members)
  node.meta["chains"] list of chain_ids (node → chains)

This gives O(1) lookup in either direction. get_chains_for_node() reads
the node meta; get_chain() reads chains.json.

== Storage ==

chains.json lives at <store_root>/chains.json alongside active.json.
"""

import json
import secrets
import time
from pathlib import Path
from typing import Optional

from mnemo import Store, Node


# ===================================================================
# Chain ID generation
# ===================================================================

def _new_chain_id() -> str:
    """Generate a stable, random chain ID: ch_ + 12 hex chars."""
    return "ch_" + secrets.token_hex(6)


# ===================================================================
# chains.json I/O
# ===================================================================

def _chains_path(store: Store) -> Path:
    return store.root / "chains.json"


def get_chains(store: Store) -> dict:
    """Load full chain index. Returns {} if file doesn't exist yet."""
    path = _chains_path(store)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _set_chains(store: Store, chains: dict):
    """Write the full chain index atomically."""
    _chains_path(store).write_text(
        json.dumps(chains, indent=2), encoding="utf-8"
    )


def get_chain(store: Store, chain_id: str) -> Optional[dict]:
    """Look up a single chain by ID. Returns None if not found."""
    return get_chains(store).get(chain_id)


def _put_chain(store: Store, chain_id: str, chain: dict):
    """Upsert a single chain into the index."""
    chains = get_chains(store)
    chains[chain_id] = chain
    _set_chains(store, chains)


# ===================================================================
# Node ↔ chain membership
# ===================================================================

def get_chains_for_node(node: Node) -> list[str]:
    """Return chain IDs this node belongs to (reads node.meta)."""
    return list(node.meta.get("chains", []))


def _add_chain_to_node(store: Store, addr: str, chain_id: str):
    """Add chain_id to node.meta['chains'], persist the node."""
    node = store.get(addr)
    if not node:
        return
    existing = node.meta.get("chains", [])
    if chain_id not in existing:
        node.meta["chains"] = existing + [chain_id]
        store.put(node)


# ===================================================================
# Chain creation and extension
# ===================================================================

def create_chain(
    store: Store,
    head_addr: str,
    *,
    domain: str = "",
    summary: str = "",
    agent_id: Optional[str] = None,
    authority: float = 0.0,
    status: str = "active",
    plan_root: bool = False,
    team_config: Optional[list] = None,
    success_criteria: Optional[dict] = None,
    friction_points: Optional[list] = None,
) -> str:
    """
    Create a new chain with a single seed node as both head and tail.

    Planning chains (plan_root=True) carry structured metadata that Arachne
    writes before agents launch. Agents discover them via recall and follow
    the team_config pointer to their role node.

    Returns the new chain_id.
    """
    chain_id = _new_chain_id()
    now = time.time()

    chain = {
        "chain_id": chain_id,
        "head": head_addr,
        "tail": head_addr,
        "members": [head_addr],
        "domain": domain,
        "summary": summary,
        "score": 0.0,
        "last_extended": now,
        "created": now,
        "status": status,
        "agent_id": agent_id,
        "authority": authority,
        "stash_reason": None,
        "plan_root": plan_root,
    }
    if plan_root:
        chain["team_config"] = team_config or []
        chain["success_criteria"] = success_criteria or {}
        chain["friction_points"] = friction_points or []

    _put_chain(store, chain_id, chain)
    _add_chain_to_node(store, head_addr, chain_id)
    return chain_id


def extend_chain(store: Store, chain_id: str, node_addr: str) -> bool:
    """
    Append node_addr to an existing chain, updating head and last_extended.

    Returns True on success, False if chain not found or node already member.
    """
    chains = get_chains(store)
    chain = chains.get(chain_id)
    if not chain:
        return False

    if node_addr in chain["members"]:
        return False  # already a member

    chain["members"].append(node_addr)
    chain["head"] = node_addr
    chain["last_extended"] = time.time()

    chains[chain_id] = chain
    _set_chains(store, chains)
    _add_chain_to_node(store, node_addr, chain_id)
    return True


def add_node_to_chains(store: Store, node_addr: str, chain_ids: list[str]):
    """
    Add an existing node to multiple chains (for retrospective assembly).
    Each chain gets the node appended at head.
    """
    for chain_id in chain_ids:
        extend_chain(store, chain_id, node_addr)


# ===================================================================
# Chain status transitions
# ===================================================================

def stash_chain(store: Store, chain_id: str, reason: str = "") -> bool:
    """
    Mark a chain as stashed. Removes member nodes from the active set
    (they remain in the store, just not active).

    Returns True on success.
    """
    chains = get_chains(store)
    chain = chains.get(chain_id)
    if not chain:
        return False
    if chain["status"] == "stashed":
        return False

    chain["status"] = "stashed"
    chain["stash_reason"] = reason
    chains[chain_id] = chain
    _set_chains(store, chains)

    # Remove member nodes from the active set that aren't in other active chains.
    active = store.get_active()
    active_chains = {
        cid: c for cid, c in chains.items()
        if c["status"] == "active" and cid != chain_id
    }
    other_active_members: set[str] = set()
    for c in active_chains.values():
        other_active_members.update(c["members"])

    to_deactivate = set(chain["members"]) - other_active_members
    active -= to_deactivate
    store.set_active(active)
    return True


def pop_chain(store: Store, chain_id: str) -> bool:
    """
    Restore a stashed chain to active status. Re-adds member nodes to
    the active set.

    Returns True on success.
    """
    chains = get_chains(store)
    chain = chains.get(chain_id)
    if not chain:
        return False
    if chain["status"] != "stashed":
        return False

    chain["status"] = "active"
    chain["stash_reason"] = None
    chains[chain_id] = chain
    _set_chains(store, chains)

    active = store.get_active()
    active.update(chain["members"])
    store.set_active(active)
    return True


def supersede_chain(store: Store, chain_id: str):
    """Mark a chain as superseded (used during reconciliation)."""
    chains = get_chains(store)
    if chain_id in chains:
        chains[chain_id]["status"] = "superseded"
        _set_chains(store, chains)


def archive_chain(store: Store, chain_id: str):
    """Mark a chain as archived (session end / completed work)."""
    chains = get_chains(store)
    if chain_id in chains:
        chains[chain_id]["status"] = "archived"
        _set_chains(store, chains)


# ===================================================================
# Chain scoring
# ===================================================================

def score_chain(
    chain: dict,
    node_scores: dict[str, float],
    *,
    agent_id: Optional[str] = None,
    recently_extended_ids: Optional[set] = None,
) -> float:
    """
    Score a chain holistically for retrieval ranking.

    Args:
        chain:                  Chain dict from chains.json.
        node_scores:            {addr: score} from node-level scoring
                                (retrieve_relevant output).
        agent_id:               Current agent's ID (for continuity boost).
        recently_extended_ids:  Chain IDs the current agent extended this
                                session (continuation signal).

    Returns:
        Composite chain score (higher = more relevant).
    """
    members = chain.get("members", [])
    if not members:
        return 0.0

    # --- Member score sum ---
    member_scores = [node_scores.get(addr, 0.0) for addr in members]
    total_score = sum(member_scores)

    if total_score == 0.0:
        return 0.0

    # --- Coherence factor ---
    # Fraction of members that scored > 0 on this query.
    # A chain where every node is relevant scores higher than one
    # where only the head matched.
    scored_members = sum(1 for s in member_scores if s > 0)
    coherence_factor = scored_members / len(members)

    base = total_score * coherence_factor

    # --- Head recency boost ---
    # Chains that were recently extended are more likely to be relevant
    # to current work.
    now = time.time()
    last_extended = chain.get("last_extended", 0.0)
    days_since_extended = (now - last_extended) / 86400
    if days_since_extended < 1:
        head_recency_boost = 0.5
    elif days_since_extended < 7:
        head_recency_boost = 0.2
    elif days_since_extended < 30:
        head_recency_boost = 0.05
    else:
        head_recency_boost = 0.0

    # --- Chain length bonus ---
    # Longer chains = more developed reasoning = more useful context.
    # Soft bonus, caps at ~10 nodes.
    length = len(members)
    chain_length_bonus = min(length * 0.05, 0.5)

    # --- Continuation signal ---
    # Chains the current agent has been actively building this session.
    continuation_signal = 0.0
    if recently_extended_ids and chain.get("chain_id") in recently_extended_ids:
        continuation_signal = 1.0

    # --- Agent continuity boost ---
    # Slightly boost chains owned by the current agent.
    agent_boost = 0.0
    if agent_id and chain.get("agent_id") == agent_id:
        agent_boost = 0.2

    # --- Authority weight (Phase 3 placeholder) ---
    # authority = chain.get("authority", 0.0)
    # authority_weight = authority * 0.3

    # --- Plan root boost ---
    # Planning chains always surface near the top — agents need to find their
    # role node and objectives on turn one regardless of query content.
    plan_boost = 2.0 if chain.get("plan_root") else 0.0

    return (
        base
        + head_recency_boost
        + chain_length_bonus
        + continuation_signal
        + agent_boost
        + plan_boost
    )


def rank_chains(
    store: Store,
    node_scores: dict[str, float],
    *,
    agent_id: Optional[str] = None,
    recently_extended_ids: Optional[set] = None,
    status_filter: Optional[set] = None,
    max_chains: int = 3,
) -> list[dict]:
    """
    Given a map of {addr: score} from node-level retrieval, assemble and
    rank chains holistically.

    Args:
        store:                  The project store.
        node_scores:            {addr: score} — every scored node from
                                retrieve_relevant().
        agent_id:               Current agent for continuity boost.
        recently_extended_ids:  Chain IDs extended this session.
        status_filter:          Only consider chains with these statuses.
                                Defaults to {"active"}.
        max_chains:             Maximum chains to return.

    Returns:
        List of scored chain dicts, sorted by score desc, each with an
        injected "chain_score" field.
    """
    if status_filter is None:
        status_filter = {"active"}

    chains = get_chains(store)
    if not chains:
        return []

    scored = []
    for chain_id, chain in chains.items():
        if chain.get("status") not in status_filter:
            continue

        chain_score = score_chain(
            chain,
            node_scores,
            agent_id=agent_id,
            recently_extended_ids=recently_extended_ids,
        )
        if chain_score > 0:
            entry = dict(chain)  # shallow copy
            entry["chain_score"] = round(chain_score, 3)
            scored.append(entry)

    scored.sort(key=lambda c: -c["chain_score"])
    return scored[:max_chains]


# ===================================================================
# Freshness helpers
# ===================================================================

def _freshness_tag(node) -> str:
    """
    Return a compact age/staleness indicator for inline display.

    Examples: · 2h  · 3d  · 14d  · stale
    'stale' = last reinforcement > 21 days ago (high decay)
    """
    last_fresh = node.meta.get("last_reinforced", node.created)
    age_secs = time.time() - last_fresh
    age_days = age_secs / 86400

    if age_days > 21:
        return " · stale"
    elif age_days >= 1:
        return f" · {int(age_days)}d"
    else:
        age_h = int(age_secs / 3600)
        if age_h >= 1:
            return f" · {age_h}h"
        return ""  # very fresh — no tag needed


# ===================================================================
# Chain rendering (for recall output)
# ===================================================================

def render_chain(chain: dict, store: Store, *, max_chars: int = 6500) -> str:
    """
    Render a chain as a coherent narrative string (tail → head).

    Each member node contributes one line. Agent attribution is
    shown when agent_id is present on the node's meta. Code context
    (content_hash anchors) is noted inline when present.

    Returns a string formatted for injection into the recall preload.
    """
    chain_id = chain.get("chain_id", "?")
    domain = chain.get("domain", "")
    summary = chain.get("summary", "")
    status = chain.get("status", "active")
    members = chain.get("members", [])
    agent_id = chain.get("agent_id")

    # Header
    is_plan = chain.get("plan_root", False)
    status_tag = f", {status}" if status != "active" else ""
    plan_tag = " [PLAN]" if is_plan else ""
    header = f"── Chain {chain_id} [{domain}{status_tag}]{plan_tag}"
    if summary:
        header += f": {summary}"
    if agent_id:
        header += f" (by {agent_id})"
    header += f" ({len(members)} node{'s' if len(members) != 1 else ''}) ──"

    lines = [header]
    chars = len(header)

    # Planning chain metadata block
    if is_plan:
        tc = chain.get("team_config", [])
        sc = chain.get("success_criteria", {})
        fp = chain.get("friction_points", [])
        if tc:
            role_lines = ", ".join(
                f"{r.get('agent_id', '?')} ({r.get('role', '?')}, auth={r.get('authority', 0)})"
                for r in tc
            )
            meta_line = f"  Team: {role_lines}"
            lines.append(meta_line)
            chars += len(meta_line)
        if sc:
            expected = sc.get("expected_outputs", "")
            out_type = sc.get("output_type", "")
            metric = sc.get("completion_metric", "")
            sc_parts = []
            if expected and out_type:
                sc_parts.append(f"{expected} {out_type}")
            if metric:
                sc_parts.append(f"metric: {metric}")
            if sc_parts:
                meta_line = f"  Success: {' | '.join(sc_parts)}"
                lines.append(meta_line)
                chars += len(meta_line)
        if fp:
            def _fp_str(f) -> str:
                if not isinstance(f, dict):
                    return str(f)
                files = f.get("files", [])
                desc = f.get("description", "")
                if files:
                    return ", ".join(files) + (f" — {desc}" if desc else "")
                return desc or str(f)
            fp_strs = [_fp_str(f) for f in fp]
            meta_line = f"  Friction: {' | '.join(fp_strs[:3])}" + (" ..." if len(fp_strs) > 3 else "")
            lines.append(meta_line)
            chars += len(meta_line)

    for addr in members:
        if chars >= max_chars:
            lines.append(f"  ... ({len(members) - members.index(addr)} more nodes)")
            break

        node = store.get(addr)
        if not node:
            continue

        # Node attribution
        node_agent = node.meta.get("agent_id")
        attr = f"[{node_agent}] " if node_agent else ""

        # Content
        content = node.content

        # Code context annotation
        code_context = node.meta.get("code_context") or {}
        if code_context:
            file_ref = code_context.get("file", "")
            line_range = code_context.get("lines", [])
            if file_ref and line_range:
                lines_str = (
                    f"{line_range[0]}-{line_range[1]}"
                    if len(line_range) == 2
                    else str(line_range[0])
                )
                content += f" [{file_ref}:{lines_str}]"

            # Drifted anchor warning
            anchors = node.meta.get("anchors", [])
            if any(a.get("drifted") for a in anchors):
                content += " ⚠ drifted"

        freshness = _freshness_tag(node)
        line = f"  {attr}{content} [{addr[:8]}{freshness}]"
        lines.append(line)
        chars += len(line)

    return "\n".join(lines)


# ===================================================================
# Standalone node rendering
# ===================================================================

def render_standalone_nodes(
    nodes: list, *, max_chars: int = 3000
) -> str:
    """
    Render high-priority standalone nodes (user preferences, critical
    warnings) that don't belong to any chain.

    `nodes` should be pre-filtered and sorted by priority desc.
    """
    if not nodes:
        return ""

    lines = ["── Always active ──"]
    chars = len(lines[0])

    for item in nodes:
        if chars >= max_chars:
            break
        node = item["node"] if isinstance(item, dict) else item
        domain = node.meta.get("domain", "")
        priority = node.meta.get("priority", 0)
        scope = node.meta.get("scope", "project")
        scope_tag = "global" if scope == "global" else ""
        addr_tag = f"{scope_tag}:{node.addr[:8]}" if scope_tag else node.addr[:8]
        freshness = _freshness_tag(node)
        line = f"  {node.content} [{addr_tag}{freshness}]"
        if priority >= 1.0:
            line = f"  ⚑ {node.content} [{addr_tag}{freshness}]"
        lines.append(line)
        chars += len(line)

    return "\n".join(lines)


# ===================================================================
# Utility
# ===================================================================

def list_chains(
    store: Store,
    *,
    status_filter: Optional[set] = None,
    agent_id: Optional[str] = None,
    domain: Optional[str] = None,
    plan_root: Optional[bool] = None,
) -> list[dict]:
    """
    List chains with optional filters.

    Args:
        plan_root: If True, only planning chains. If False, only work chains.
                   If None (default), return all.

    Returns chains sorted by last_extended desc (most recently active first).
    """
    chains = get_chains(store)
    result = []
    for chain_id, chain in chains.items():
        if status_filter and chain.get("status") not in status_filter:
            continue
        if agent_id and chain.get("agent_id") != agent_id:
            continue
        if domain and chain.get("domain") != domain:
            continue
        if plan_root is not None and bool(chain.get("plan_root")) != plan_root:
            continue
        result.append(chain)

    result.sort(key=lambda c: -(c.get("last_extended") or 0))
    return result


def update_chain_summary(store: Store, chain_id: str, summary: str) -> bool:
    """Update the human-readable summary for a chain."""
    chains = get_chains(store)
    if chain_id not in chains:
        return False
    chains[chain_id]["summary"] = summary
    _set_chains(store, chains)
    return True


def update_chain_domain(store: Store, chain_id: str, domain: str) -> bool:
    """Update the domain for a chain."""
    chains = get_chains(store)
    if chain_id not in chains:
        return False
    chains[chain_id]["domain"] = domain
    _set_chains(store, chains)
    return True


def infer_chain_domain(chain: dict, store: Store) -> str:
    """
    Infer the primary domain from member nodes (majority vote).
    Falls back to the chain's stored domain if no clear winner.
    """
    domain_counts: dict[str, int] = {}
    for addr in chain.get("members", []):
        node = store.get(addr)
        if node:
            d = node.meta.get("domain", "")
            if d:
                domain_counts[d] = domain_counts.get(d, 0) + 1

    if not domain_counts:
        return chain.get("domain", "")

    return max(domain_counts, key=domain_counts.get)
