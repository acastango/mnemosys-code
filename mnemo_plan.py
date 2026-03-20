"""
mnemo_plan.py — Tree-aware planning context

Given a task description, produces a structured planning context:
1. Architecture: which modules/systems are involved?
2. Constraints: which decisions and patterns must be respected?
3. Risks: which known issues, gotchas, or blockers affect this task?
4. Affected files: where do changes need to happen?
5. Current state: what's in progress, what's blocked?
6. Related history: what was tried before?

This is NOT the plan itself — it's the tree-informed context that
should shape the plan. The calling model does the actual planning,
but with full awareness of project knowledge.

Zero LLM calls. Tree retrieval + categorization.
"""

import os
import re
from pathlib import Path
from typing import Optional

from mnemo import Store, Node
from mnemo_associate import retrieve_relevant
from mnemo_verify import _resolve_project_root


# -------------------------------------------------------------------
# Node categorization for planning
# -------------------------------------------------------------------

# Map domains to planning roles
_PLANNING_ROLES = {
    "architecture":  "architecture",
    "decisions":     "constraints",
    "patterns":      "constraints",
    "issues":        "risks",
    "tasks":         "state",
    "dependencies":  "constraints",
    "history":       "history",
    "context":       "context",
}


def _categorize_nodes(scored: list[dict]) -> dict[str, list[dict]]:
    """Group scored nodes by their planning role."""
    categories: dict[str, list[dict]] = {
        "architecture": [],
        "constraints":  [],
        "risks":        [],
        "state":        [],
        "history":      [],
        "context":      [],
    }

    for item in scored:
        node = item["node"]
        domain = node.meta.get("domain", "context")
        role = _PLANNING_ROLES.get(domain, "context")
        categories[role].append(item)

    return categories


# -------------------------------------------------------------------
# File extraction
# -------------------------------------------------------------------

_FILE_PATTERN = re.compile(
    r'([\w./\\-]+\.(?:py|js|ts|tsx|jsx|rs|go|java|toml|yaml|yml|json|cfg))'
)


def _extract_affected_files(nodes: list[Node],
                            project_root: Path) -> list[dict]:
    """Extract files referenced in nodes, check existence.

    Returns list of {path, exists, sources: [addr[:8]]}.
    """
    file_sources: dict[str, list[str]] = {}

    for node in nodes:
        for match in _FILE_PATTERN.finditer(node.content):
            path = match.group(1)
            if path not in file_sources:
                file_sources[path] = []
            file_sources[path].append(node.addr[:8])

        for anchor in node.meta.get("anchors", []):
            if anchor.get("path"):
                path = anchor["path"]
                if path not in file_sources:
                    file_sources[path] = []
                file_sources[path].append(node.addr[:8])

    results = []
    for path, sources in sorted(file_sources.items()):
        exists = (project_root / path).exists()
        results.append({
            "path": path,
            "exists": exists,
            "sources": sorted(set(sources)),
        })

    return results


# -------------------------------------------------------------------
# Tension detection between task and existing constraints
# -------------------------------------------------------------------

def _find_blockers(task: str, categories: dict[str, list[dict]]) -> list[str]:
    """Identify nodes that might block or complicate this task."""
    blockers = []

    # Check if any issues mention things the task touches
    task_lower = task.lower()
    task_words = set(re.findall(r'\w{4,}', task_lower))

    for item in categories.get("risks", []):
        node = item["node"]
        content_lower = node.content.lower()
        overlap = task_words & set(re.findall(r'\w{4,}', content_lower))
        if overlap:
            blockers.append(
                f"[{node.addr[:8]}] {node.content[:150]}"
            )

    # Check graph links — any node that "blocks" something related
    for role in categories.values():
        for item in role:
            links = item["node"].meta.get("links", [])
            for link in links:
                if link.get("rel") == "blocks":
                    blockers.append(
                        f"[{item['node'].addr[:8]}] blocks {link['addr'][:8]}: "
                        f"{item['node'].content[:100]}"
                    )

    return blockers[:5]


# -------------------------------------------------------------------
# Main plan function
# -------------------------------------------------------------------

def plan(task: str, store: Store,
         session_context: dict = None,
         project_root: Path = None) -> str:
    """Generate tree-informed planning context for a task.

    Returns a structured report organized by planning role.
    """
    if project_root is None:
        project_root = _resolve_project_root() or Path.cwd()

    parts = []

    # ── Step 1: Recall with high depth ──────────────────────────────
    scored = retrieve_relevant(task, store,
                               session_context=session_context,
                               max_nodes=12)
    nodes = [item["node"] for item in scored]

    # ── Step 2: Categorize by planning role ─────────────────────────
    categories = _categorize_nodes(scored)

    # ── Architecture ────────────────────────────────────────────────
    if categories["architecture"]:
        parts.append("## Architecture (modules and systems involved)\n")
        for item in categories["architecture"]:
            n = item["node"]
            priority_tag = f" [priority={n.meta.get('priority')}]" if n.meta.get("priority") else ""
            parts.append(f"  [{n.addr[:8]}]{priority_tag} {n.content[:250]}")
        parts.append("")

    # ── Constraints ─────────────────────────────────────────────────
    if categories["constraints"]:
        parts.append("## Constraints (decisions, patterns, dependencies to respect)\n")
        for item in categories["constraints"]:
            n = item["node"]
            domain = n.meta.get("domain", "?")
            priority_tag = f" [priority={n.meta.get('priority')}]" if n.meta.get("priority") else ""
            parts.append(f"  [{domain}] {n.addr[:8]}{priority_tag}: {n.content[:250]}")
        parts.append("")

    # ── Risks ───────────────────────────────────────────────────────
    if categories["risks"]:
        parts.append("## Risks (known issues, gotchas, fragile areas)\n")
        for item in categories["risks"]:
            n = item["node"]
            parts.append(f"  [{n.addr[:8]}] {n.content[:250]}")
        parts.append("")

    # ── Current state ───────────────────────────────────────────────
    if categories["state"]:
        parts.append("## Current state (tasks, progress, blockers)\n")
        for item in categories["state"]:
            n = item["node"]
            parts.append(f"  [{n.addr[:8]}] {n.content[:250]}")
        parts.append("")

    # ── History ─────────────────────────────────────────────────────
    if categories["history"]:
        parts.append("## History (what was tried before)\n")
        for item in categories["history"]:
            n = item["node"]
            parts.append(f"  [{n.addr[:8]}] {n.content[:250]}")
        parts.append("")

    # ── Affected files ──────────────────────────────────────────────
    affected = _extract_affected_files(nodes, project_root)
    if affected:
        parts.append("## Affected files (referenced in tree nodes)\n")
        for f in affected:
            status = "exists" if f["exists"] else "MISSING"
            sources = ", ".join(f["sources"])
            parts.append(f"  {f['path']} ({status}) -- from: {sources}")
        parts.append("")

    # ── Blockers ────────────────────────────────────────────────────
    blockers = _find_blockers(task, categories)
    if blockers:
        parts.append("## Potential blockers\n")
        for b in blockers:
            parts.append(f"  ! {b}")
        parts.append("")

    # ── Planning summary ────────────────────────────────────────────
    parts.append("## Planning context summary\n")
    parts.append(f"  Task: {task}")
    parts.append(f"  Tree nodes consulted: {len(nodes)}")
    for role, items in categories.items():
        if items:
            parts.append(f"  {role}: {len(items)} node(s)")
    parts.append(f"  Affected files: {len(affected)}")
    parts.append(f"  Potential blockers: {len(blockers)}")

    if not nodes:
        parts.append("\n  The tree has no context for this task — "
                     "consider exploring first with memory_explore.")

    return "\n".join(parts)
