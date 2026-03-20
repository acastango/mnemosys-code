"""
mnemo_graph.py — Explicit link-graph traversal for mnemo.

Provides deliberate BFS over the link graph, distinct from recall's
implicit multi-hop scoring. Where recall uses graph traversal to boost
associative scores, this module renders the graph structure explicitly
for inspection and reasoning.

Entry point: traverse_graph() + render_graph()
MCP tool: memory_graph (in mnemo_mcp.py)
"""

from __future__ import annotations

from collections import deque

_CAUSAL_RELS = frozenset({"caused_by", "depends_on", "blocks"})
_ALL_REL_TYPES = frozenset({
    "caused_by", "depends_on", "blocks", "enables", "relates_to", "contradicts",
})


def traverse_graph(
    store,
    start_addr: str,
    depth: int = 2,
    rel_types: list[str] | None = None,
    direction: str = "both",
    max_nodes: int = 40,
) -> dict:
    """
    BFS from start_addr through the link graph.

    Args:
        store:      mnemo Store instance
        start_addr: Starting node address (prefix matching supported upstream)
        depth:      Maximum hops from the root (1 = immediate neighbours only)
        rel_types:  Relationship types to follow (None = all)
        direction:  "forward" | "reverse" | "both"
        max_nodes:  Hard cap on nodes returned

    Returns:
        {
            "root":  Node | None,
            "nodes": {addr: Node},
            "edges": [(from_addr, to_addr, rel, hop_depth)],
        }
    """
    active = set(store.get_active())
    allowed = set(rel_types) if rel_types else _ALL_REL_TYPES

    root = store.get(start_addr)
    if not root:
        return {"root": None, "nodes": {}, "edges": []}

    nodes: dict[str, object] = {root.addr: root}
    edges: list[tuple[str, str, str, int]] = []
    seen_edges: set[tuple[str, str, str]] = set()  # (from, to, rel) — no duplicates
    visited: set[str] = {root.addr}
    queue: deque[tuple[str, int]] = deque([(root.addr, 0)])

    while queue and len(nodes) < max_nodes:
        current_addr, current_depth = queue.popleft()
        if current_depth >= depth:
            continue

        current_node = store.get(current_addr)
        if not current_node:
            continue

        next_depth = current_depth + 1

        # Forward links: current_node → target
        if direction in ("forward", "both"):
            for link in current_node.meta.get("links", []):
                target = link.get("addr", "")
                rel = link.get("rel", "relates_to")
                if not target or target not in active or rel not in allowed:
                    continue
                edge_key = (current_addr, target, rel)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append((current_addr, target, rel, next_depth))
                if target not in visited and len(nodes) < max_nodes:
                    visited.add(target)
                    t_node = store.get(target)
                    if t_node:
                        nodes[target] = t_node
                        queue.append((target, next_depth))

        # Reverse links: source → current_node
        if direction in ("reverse", "both"):
            for rl in store.get_reverse_links(current_addr):
                src = rl.get("source_addr", "")
                rel = rl.get("rel", "relates_to")
                if not src or src not in active or rel not in allowed:
                    continue
                edge_key = (src, current_addr, rel)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append((src, current_addr, rel, next_depth))
                if src not in visited and len(nodes) < max_nodes:
                    visited.add(src)
                    r_node = store.get(src)
                    if r_node:
                        nodes[src] = r_node
                        queue.append((src, next_depth))

    return {"root": root, "nodes": nodes, "edges": edges}


def render_graph(result: dict) -> str:
    """Render graph traversal result as a readable string."""
    root = result["root"]
    if not root:
        return "Node not found."

    nodes = result["nodes"]
    edges = result["edges"]

    root_domain = root.meta.get("domain", "?")
    root_snippet = root.content[:120].replace("\n", " ")
    lines = [
        f"Graph root [{root_domain}] {root.addr[:8]}",
        f"  {root_snippet}",
    ]

    if not edges:
        lines.append("\n(no links found from this node)")
        return "\n".join(lines)

    connected = len(nodes) - 1
    lines.append(f"\n{connected} connected node(s), {len(edges)} edge(s):\n")

    # Group by hop depth
    by_depth: dict[int, list] = {}
    for edge in edges:
        by_depth.setdefault(edge[3], []).append(edge)

    for hop in sorted(by_depth.keys()):
        lines.append(f"  Hop {hop}:")
        for from_addr, to_addr, rel, _ in by_depth[hop]:

            from_short = from_addr[:8]
            to_short = to_addr[:8]
            to_node = nodes.get(to_addr)
            domain = to_node.meta.get("domain", "?") if to_node else "?"
            snippet = to_node.content[:80].replace("\n", " ") if to_node else ""
            lines.append(
                f"    {from_short} --[{rel}]--> {to_short}  [{domain}] {snippet}"
            )

    return "\n".join(lines)
