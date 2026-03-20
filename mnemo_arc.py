"""
mnemo_arc — work arcs for multi-session momentum tracking

A work arc is a named, evolving goal that spans multiple sessions.
It tracks trajectory (what happened each session), direction (where
we're heading), and completion state.

Arc nodes live in the tree like everything else — content-addressed,
supersedable, compressible. No new data structures. Arcs are a
convention on top of the existing node model.

Entry points:
    create_arc()            — start a new arc
    update_arc()            — append a trajectory line (manual or auto)
    complete_arc()          — mark arc as done
    pause_arc()             — pause without completing
    find_active_arcs()      — list active/paused arcs
    match_session_to_arcs() — find arcs that overlap with session work
    detect_arc_candidates() — scan handoff chain for potential arcs
"""

import re
import time
from mnemo import Store, Node, supersede


# ===================================================================
# Arc CRUD
# ===================================================================

def create_arc(
    store: Store,
    name: str,
    goal: str,
    domains: list[str] = None,
    keywords: list[str] = None,
) -> str:
    """
    Create a new work arc.

    Returns the arc node address.
    """
    # Auto-extract keywords from goal if not provided
    if not keywords:
        keywords = _extract_keywords(goal)

    content = (
        f"Arc: {name} (0 sessions, active)\n"
        f"\n"
        f"Goal: {goal}\n"
        f"\n"
        f"Trajectory:\n"
        f"  (no sessions yet)\n"
        f"\n"
        f"Next: begin work"
    )

    node = Node(
        type="leaf",
        content=content,
        meta={
            "domain": "tasks",
            "confidence": 0.9,
            "source": "conscious",
            "arc": True,
            "arc_name": name,
            "arc_status": "active",
            "arc_sessions": 0,
            "arc_domains": domains or [],
            "arc_keywords": keywords,
            "priority": 0.5,
        },
    )
    store.put(node)
    active = store.get_active()
    active.add(node.addr)
    store.set_active(active)

    return node.addr


def update_arc(
    store: Store,
    arc_addr: str,
    progress: str,
    next_step: str = "",
    domains_touched: list[str] = None,
) -> str | None:
    """
    Append a trajectory line to an active arc. Supersedes the old arc node.

    Returns the new arc address, or None if arc not found.
    """
    old = store.get(arc_addr)
    if not old or not old.meta.get("arc"):
        return None

    sessions = old.meta.get("arc_sessions", 0) + 1
    name = old.meta.get("arc_name", "unnamed")
    status = old.meta.get("arc_status", "active")

    # Parse existing content to extract goal and trajectory
    goal, trajectory = _parse_arc_content(old.content)

    # Add new trajectory line
    age_label = _age_label(time.time())
    trajectory.append(f"Session {sessions} ({age_label}): {progress}")

    # Update keywords with new domains
    arc_keywords = old.meta.get("arc_keywords", [])
    new_words = _extract_keywords(progress)
    merged = list(set(arc_keywords + new_words))

    arc_domains = old.meta.get("arc_domains", [])
    if domains_touched:
        arc_domains = list(set(arc_domains + domains_touched))

    # Build updated content
    next_line = next_step or _extract_next(old.content) or "continue work"
    content = _build_arc_content(name, sessions, status, goal, trajectory, next_line)

    # Supersede
    new_meta = dict(old.meta)
    new_meta.update({
        "arc_sessions": sessions,
        "arc_domains": arc_domains,
        "arc_keywords": merged,
    })

    new_addr = supersede(old.addr, content, store,
                         reason=f"arc progress: {progress[:60]}")

    # Update meta on the new node
    new_node = store.get(new_addr)
    if new_node:
        new_node.meta.update(new_meta)
        store.put(new_node)

    return new_addr


def complete_arc(
    store: Store,
    arc_addr: str,
    outcome: str = "",
) -> str | None:
    """
    Mark an arc as completed. Supersedes with final status.

    Returns the new arc address, or None if not found.
    """
    old = store.get(arc_addr)
    if not old or not old.meta.get("arc"):
        return None

    sessions = old.meta.get("arc_sessions", 0)
    name = old.meta.get("arc_name", "unnamed")
    goal, trajectory = _parse_arc_content(old.content)

    if outcome:
        trajectory.append(f"Completed: {outcome}")

    content = _build_arc_content(name, sessions, "completed", goal, trajectory,
                                 next_line=None)

    new_addr = supersede(old.addr, content, store,
                         reason=f"arc completed: {outcome[:60]}")

    new_node = store.get(new_addr)
    if new_node:
        new_node.meta["arc_status"] = "completed"
        new_node.meta["priority"] = 0  # stop boosting in recall
        store.put(new_node)

    return new_addr


def pause_arc(
    store: Store,
    arc_addr: str,
    reason: str = "",
) -> str | None:
    """
    Pause an arc without completing it. Can be resumed later.

    Returns the new arc address, or None if not found.
    """
    old = store.get(arc_addr)
    if not old or not old.meta.get("arc"):
        return None

    sessions = old.meta.get("arc_sessions", 0)
    name = old.meta.get("arc_name", "unnamed")
    goal, trajectory = _parse_arc_content(old.content)

    if reason:
        trajectory.append(f"Paused: {reason}")

    content = _build_arc_content(name, sessions, "paused", goal, trajectory,
                                 next_line=f"resume: {reason}" if reason else "resume")

    new_addr = supersede(old.addr, content, store,
                         reason=f"arc paused: {reason[:60]}")

    new_node = store.get(new_addr)
    if new_node:
        new_node.meta["arc_status"] = "paused"
        store.put(new_node)

    return new_addr


# ===================================================================
# Arc queries
# ===================================================================

def find_active_arcs(store: Store) -> list[Node]:
    """Find all active and paused arc nodes."""
    active = store.get_active()
    arcs = []

    for addr in active:
        node = store.get(addr)
        if not node:
            continue
        if node.meta.get("arc") and node.meta.get("arc_status") in ("active", "paused"):
            arcs.append(node)

    # Most recently modified first
    arcs.sort(key=lambda n: n.created, reverse=True)
    return arcs


def match_session_to_arcs(
    store: Store,
    domains_touched: list[str],
    work_keywords: list[str],
) -> list[tuple[Node, float]]:
    """
    Find active arcs that overlap with session work.

    Returns list of (arc_node, overlap_score) sorted by score descending.
    Only includes arcs above 0.2 overlap threshold.
    """
    arcs = find_active_arcs(store)
    if not arcs:
        return []

    work_set = set(w.lower() for w in work_keywords)
    domain_set = set(d.lower() for d in domains_touched)

    matches = []
    for arc in arcs:
        arc_keywords = set(k.lower() for k in arc.meta.get("arc_keywords", []))
        arc_domains = set(d.lower() for d in arc.meta.get("arc_domains", []))

        # Keyword overlap (Jaccard)
        kw_union = arc_keywords | work_set
        kw_inter = arc_keywords & work_set
        kw_score = len(kw_inter) / len(kw_union) if kw_union else 0

        # Domain overlap (Jaccard)
        dm_union = arc_domains | domain_set
        dm_inter = arc_domains & domain_set
        dm_score = len(dm_inter) / len(dm_union) if dm_union else 0

        # Combined: keywords matter more than domains
        score = 0.7 * kw_score + 0.3 * dm_score

        if score >= 0.2:
            matches.append((arc, score))

    matches.sort(key=lambda x: x[1], reverse=True)
    return matches


def detect_arc_candidates(store: Store) -> list[dict]:
    """
    Scan the handoff chain for thematic overlap across sessions.
    Proposes arcs when 2+ sequential handoffs share significant overlap.

    Returns list of candidate dicts with: name, goal, evidence, domains.
    """
    active = store.get_active()

    # Collect handoff nodes sorted by creation time
    handoffs = []
    for addr in active:
        node = store.get(addr)
        if not node:
            continue
        if node.meta.get("handoff"):
            handoffs.append(node)

    handoffs.sort(key=lambda n: n.created)

    if len(handoffs) < 2:
        return []

    # Check which handoffs aren't already covered by an active arc
    existing_arcs = find_active_arcs(store)

    candidates = []

    # Sliding window: compare adjacent handoffs
    for i in range(len(handoffs) - 1):
        h1 = handoffs[i]
        h2 = handoffs[i + 1]

        kw1 = set(_extract_keywords(h1.content))
        kw2 = set(_extract_keywords(h2.content))

        overlap = kw1 & kw2
        union = kw1 | kw2

        if not union:
            continue

        score = len(overlap) / len(union)

        if score < 0.2:
            continue

        # Check if this theme is already covered by an existing arc
        already_covered = False
        for arc in existing_arcs:
            arc_kw = set(k.lower() for k in arc.meta.get("arc_keywords", []))
            if len(overlap & arc_kw) > len(overlap) * 0.5:
                already_covered = True
                break

        if already_covered:
            continue

        # Extract domains from handoffs
        domains1 = set(h1.meta.get("domains_touched", []))
        domains2 = set(h2.meta.get("domains_touched", []))
        shared_domains = sorted(domains1 & domains2) or sorted(domains1 | domains2)

        # Propose arc name from the overlapping keywords
        theme_words = sorted(overlap, key=len, reverse=True)[:3]
        proposed_name = " ".join(theme_words) if theme_words else "unnamed theme"

        candidates.append({
            "name": proposed_name,
            "goal": f"Recurring theme across sessions: {', '.join(sorted(overlap)[:6])}",
            "evidence": [h1.addr[:8], h2.addr[:8]],
            "domains": shared_domains,
            "overlap_score": score,
            "keywords": sorted(overlap),
        })

    return candidates


# ===================================================================
# Internal helpers
# ===================================================================

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "and",
    "but", "or", "nor", "not", "no", "so", "if", "then", "than", "too",
    "very", "just", "about", "up", "out", "off", "over", "under", "again",
    "further", "once", "here", "there", "when", "where", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "only", "own", "same", "that", "this", "what", "which",
    "who", "whom", "it", "its", "we", "they", "them", "their", "our",
    "your", "my", "his", "her", "she", "he", "you", "i", "me", "us",
    "session", "handoff", "turns", "nodes", "worked", "active", "tasks",
    "domains", "touched", "corrections", "made",
})


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from text, filtering stop words."""
    words = re.findall(r'[a-z_][a-z0-9_]*', text.lower())
    return list(set(
        w for w in words
        if w not in _STOP_WORDS and len(w) > 2
    ))


def _parse_arc_content(content: str) -> tuple[str, list[str]]:
    """Parse arc content into (goal, trajectory_lines)."""
    goal = ""
    trajectory = []

    in_trajectory = False
    for line in content.split("\n"):
        stripped = line.strip()

        if stripped.startswith("Goal:"):
            goal = stripped[5:].strip()
        elif stripped == "Trajectory:":
            in_trajectory = True
        elif stripped.startswith("Next:") or stripped.startswith("Blockers:"):
            in_trajectory = False
        elif in_trajectory and stripped and stripped != "(no sessions yet)":
            trajectory.append(stripped)

    return goal, trajectory


def _extract_next(content: str) -> str:
    """Extract the Next: line from arc content."""
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("Next:"):
            return stripped[5:].strip()
    return ""


def _build_arc_content(
    name: str,
    sessions: int,
    status: str,
    goal: str,
    trajectory: list[str],
    next_line: str | None,
) -> str:
    """Build formatted arc content."""
    lines = [
        f"Arc: {name} ({sessions} session{'s' if sessions != 1 else ''}, {status})",
        "",
        f"Goal: {goal}",
        "",
        "Trajectory:",
    ]

    if trajectory:
        for t in trajectory:
            # Ensure indentation
            if not t.startswith("  "):
                t = f"  {t}"
            lines.append(t)
    else:
        lines.append("  (no sessions yet)")

    if next_line:
        lines.append("")
        lines.append(f"Next: {next_line}")

    return "\n".join(lines)


def _age_label(ts: float) -> str:
    """Human-readable age label like 'today', '1d ago', '3d ago'."""
    age_days = int((time.time() - ts) / 86400)
    if age_days == 0:
        return "today"
    elif age_days == 1:
        return "1d ago"
    else:
        return f"{age_days}d ago"
