"""
mnemo_handoff — session continuity through structured handoff nodes

Solves the "losing the thread" problem:
- At session end: captures what was worked on, what's pending, corrections received
- At session start: surfaces orientation context regardless of first message content

Two entry points:
    generate_handoff()   — creates a handoff node from session state + recent logs
    build_orientation()  — assembles first-recall priming context from tree state
"""

import json
import os
import time
from mnemo import Store, Node


# ===================================================================
# Handoff generation — called at session compress time
# ===================================================================

def generate_handoff(
    store: Store,
    session_addrs: list[str],
    session_turns: int,
    logs_dir: str = None,
) -> str | None:
    """
    Create a structured handoff node capturing the session's work state.

    Returns the handoff node address, or None if there's nothing to hand off.
    """
    if not session_addrs:
        return None

    active = store.get_active()

    # --- Gather session work ---
    work_items = []
    corrections = []
    domains_touched = set()

    for addr in session_addrs:
        node = store.get(addr)
        if not node:
            continue

        domain = node.meta.get("domain", "uncategorized")
        domains_touched.add(domain)

        if node.type == "supersede":
            # This was a correction/update
            reason = node.meta.get("reason", "")
            old_addr = node.inputs[0] if node.inputs else "?"
            corrections.append({
                "old": old_addr[:8],
                "new": addr[:8],
                "reason": reason,
                "content": node.content[:80],
                "domain": domain,
            })
        elif node.type == "leaf":
            work_items.append({
                "addr": addr[:8],
                "content": node.content[:80],
                "domain": domain,
            })
        elif node.type == "compress":
            work_items.append({
                "addr": addr[:8],
                "content": f"[compressed] {node.content[:70]}",
                "domain": domain,
            })

    # --- Gather pending work from task-domain nodes ---
    pending = []
    for addr in active:
        node = store.get(addr)
        if not node:
            continue
        if node.meta.get("domain") == "tasks":
            pending.append({
                "addr": addr[:8],
                "content": node.content[:80],
            })

    # --- Extract recent corrections from logs ---
    log_corrections = _extract_recent_corrections(logs_dir) if logs_dir else []

    # --- Build handoff content ---
    lines = [f"Session handoff ({session_turns} turns, {len(session_addrs)} nodes)"]
    lines.append("")

    if work_items:
        lines.append("Worked on:")
        for item in work_items[:10]:
            lines.append(f"  [{item['domain']}] {item['content']}")

    if corrections:
        lines.append("")
        lines.append("Corrections made:")
        for c in corrections[:8]:
            reason_part = f" — {c['reason']}" if c['reason'] else ""
            lines.append(f"  {c['old']} → {c['new']}{reason_part}: {c['content']}")

    if log_corrections:
        lines.append("")
        lines.append("User feedback received:")
        for lc in log_corrections[:5]:
            lines.append(f"  {lc}")

    if pending:
        lines.append("")
        lines.append("Active tasks:")
        for p in pending[:5]:
            lines.append(f"  {p['addr']}: {p['content']}")

    if domains_touched:
        lines.append("")
        lines.append(f"Domains touched: {', '.join(sorted(domains_touched))}")

    content = "\n".join(lines)

    # --- Create handoff node ---
    node = Node(
        type="leaf",
        content=content,
        meta={
            "domain": "tasks",
            "confidence": 0.9,
            "source": "system",
            "handoff": True,
            "session_turns": session_turns,
            "session_node_count": len(session_addrs),
            "domains_touched": sorted(domains_touched),
            "priority": 0.5,  # moderate boost so it surfaces on next session start
        },
    )
    store.put(node)
    active = store.get_active()
    active.add(node.addr)
    store.set_active(active)

    return node.addr


def _extract_recent_corrections(logs_dir: str, max_age_hours: int = 4) -> list[str]:
    """
    Pull user correction signals from recent session logs.
    Looks for update events with correction-indicating reasons.
    """
    if not logs_dir or not os.path.isdir(logs_dir):
        return []

    cutoff = time.time() - (max_age_hours * 3600)
    corrections = []

    # Read log files sorted newest first
    try:
        log_files = sorted(
            [f for f in os.listdir(logs_dir) if f.endswith(".log")],
            reverse=True,
        )
    except OSError:
        return []

    for log_file in log_files[:3]:  # only check last 3 log files
        path = os.path.join(logs_dir, log_file)
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Only update events
                    if event.get("event") != "update":
                        continue

                    # Check timestamp
                    ts = event.get("ts", "")
                    detail = event.get("detail", {})
                    reason = detail.get("reason", "")

                    if reason:
                        corrections.append(reason)
        except OSError:
            continue

    return corrections[:10]


# ===================================================================
# First-recall orientation — called on turn 1
# ===================================================================

def build_orientation(
    store: Store,
    global_store: Store = None,
) -> str | None:
    """
    Build orientation context for the first recall of a new session.

    Assembles:
    1. Most recent handoff node (what was the last session doing?)
    2. Recent corrections (what keeps getting fixed?)
    3. Active tasks/blockers
    4. User profile from global store

    Returns formatted orientation text, or None if nothing useful.
    """
    sections = []

    # --- 1. Latest handoff ---
    handoff = _find_latest_handoff(store)
    if handoff:
        age_hours = (time.time() - handoff.created) / 3600
        if age_hours < 168:  # within last week
            age_str = (
                f"{age_hours:.0f}h ago" if age_hours < 24
                else f"{age_hours / 24:.0f}d ago"
            )
            sections.append(f"Last session ({age_str}):\n{handoff.content}")

    # --- 2. Recent corrections (volatile domains) ---
    correction_nodes = _find_recent_corrections(store, max_count=5)
    if correction_nodes:
        lines = ["Recent corrections:"]
        for node in correction_nodes:
            reason = node.meta.get("reason", "")
            reason_part = f" ({reason})" if reason else ""
            lines.append(f"  {node.addr[:8]}{reason_part}: {node.content[:70]}")
        sections.append("\n".join(lines))

    # --- 3. Active tasks ---
    tasks = _find_active_tasks(store)
    if tasks:
        lines = ["Active tasks:"]
        for node in tasks[:5]:
            lines.append(f"  {node.addr[:8]}: {node.content[:70]}")
        sections.append("\n".join(lines))

    # --- 4. Active work arcs ---
    from mnemo_arc import find_active_arcs
    arcs = find_active_arcs(store)
    if arcs:
        lines = ["Active work arcs:"]
        for arc in arcs[:3]:
            lines.append(f"  {arc.addr[:8]}: {arc.content}")
        sections.append("\n".join(lines))

    # --- 5. User profile from global ---
    if global_store:
        profile = _find_user_profile(global_store)
        if profile:
            sections.append(f"User profile: {profile}")

    if not sections:
        return None

    return "── Session orientation ──\n\n" + "\n\n".join(sections)


def _find_latest_handoff(store: Store) -> Node | None:
    """Find the most recent handoff node in active set."""
    active = store.get_active()
    best = None
    best_time = 0

    for addr in active:
        node = store.get(addr)
        if not node:
            continue
        if node.meta.get("handoff") and node.created > best_time:
            best = node
            best_time = node.created

    return best


def _find_recent_corrections(store: Store, max_count: int = 5) -> list[Node]:
    """Find recent supersede nodes — things that were corrected."""
    active = store.get_active()
    corrections = []

    for addr in active:
        node = store.get(addr)
        if not node:
            continue
        if node.type == "supersede":
            corrections.append(node)

    # Sort by recency
    corrections.sort(key=lambda n: n.created, reverse=True)
    return corrections[:max_count]


def _find_active_tasks(store: Store) -> list[Node]:
    """Find active task-domain nodes."""
    active = store.get_active()
    tasks = []

    for addr in active:
        node = store.get(addr)
        if not node:
            continue
        if node.meta.get("domain") == "tasks":
            tasks.append(node)

    # Most recent first
    tasks.sort(key=lambda n: n.created, reverse=True)
    return tasks


def _find_user_profile(global_store: Store) -> str | None:
    """Extract user profile summary from global store."""
    active = global_store.get_active()

    for addr in active:
        node = global_store.get(addr)
        if not node:
            continue
        content_lower = node.content.lower()
        # Look for profile/preference nodes
        if any(kw in content_lower for kw in
               ("working style", "user profile", "preferences", "user is")):
            return node.content[:200]

    return None
