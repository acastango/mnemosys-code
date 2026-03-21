"""
mnemo_pipeline.py - Composable memory pipelines

Pipelines are first-class nodes in the tree: stored, addressed,
supersedable, and recallable like any other knowledge.

Every step receives the current node set and returns a new one.
Sources ignore the current set and produce fresh nodes.
Sinks produce side effects and pass the set through unchanged.
No LLM in the loop - execution is pure Python over the store.

Step ops:
  Sources:   recall, search, active, spatial
  Transforms: traverse, filter, sort, limit, dedupe
  Sinks:     compress, claim, link

Variables: {varname} in any string parameter, resolved from
the params dict passed to run_pipeline().

MCP surface (in mnemo_mcp.py):
  memory_pipeline(name, steps, description?)  - define + store
  memory_run(name_or_addr, **params)          - invoke
  memory_pipelines()                          - list defined
"""

from __future__ import annotations

import re
import time
from typing import Any

from mnemo import Node, Store, compress as _compress_nodes


# ───────────────────────────────────────────────────────────────────
# Built-in pipelines
# ───────────────────────────────────────────────────────────────────

BUILTIN_PIPELINES: dict[str, dict] = {
    "session-orient": {
        "description": "Orient a new session: surface architecture + compress",
        "steps": [
            {"op": "recall",   "query": "{input}", "max_nodes": 12},
            {"op": "traverse", "depth": 1},
            {"op": "filter",   "domain": "architecture"},
            {"op": "dedupe"},
            {"op": "compress", "label": "orientation: {input}"},
        ],
    },
    "file-context": {
        "description": "Surface all tree knowledge relevant to a file",
        "steps": [
            {"op": "spatial", "file": "{input}"},
            {"op": "traverse", "depth": 1},
            {"op": "dedupe"},
        ],
    },
    "issue-cluster": {
        "description": "Cluster known issues into a summary",
        "steps": [
            {"op": "active",   "domain": "issues"},
            {"op": "dedupe"},
            {"op": "compress", "label": "known issues"},
        ],
    },
    "drift-check": {
        "description": "Find anchors that may have drifted from the codebase",
        "steps": [
            {"op": "active"},
            {"op": "filter", "has_anchors": True},
            {"op": "dedupe"},
        ],
    },
}


# ───────────────────────────────────────────────────────────────────
# Variable interpolation
# ───────────────────────────────────────────────────────────────────

def _interp(value: Any, ctx: dict) -> Any:
    """Substitute {varname} in string values. Non-strings pass through."""
    if not isinstance(value, str):
        return value
    def _replace(m):
        key = m.group(1)
        return str(ctx.get(key, m.group(0)))
    return re.sub(r"\{(\w+)\}", _replace, value)


def _interp_step(step: dict, ctx: dict) -> dict:
    """Return a copy of step with all string values interpolated."""
    return {k: _interp(v, ctx) for k, v in step.items()}


# ───────────────────────────────────────────────────────────────────
# Step implementations
# ───────────────────────────────────────────────────────────────────

def _op_recall(step: dict, _current: list[Node], store: Store) -> list[Node]:
    from mnemo_associate import retrieve_relevant
    query = step.get("query", "")
    max_nodes = int(step.get("max_nodes", 8))
    if not query:
        return _current
    results = retrieve_relevant(query, store, max_nodes=max_nodes)
    return [r["node"] for r in results]


def _op_search(step: dict, _current: list[Node], store: Store) -> list[Node]:
    """TF-IDF search — similar to recall but skips session/domain boosts."""
    from mnemo_associate import retrieve_relevant
    query = step.get("query", "")
    max_nodes = int(step.get("max_nodes", 8))
    if not query:
        return _current
    results = retrieve_relevant(query, store, max_nodes=max_nodes)
    return [r["node"] for r in results]


def _op_active(step: dict, _current: list[Node], store: Store) -> list[Node]:
    domain = step.get("domain", "")
    active = store.get_active()
    nodes = []
    for addr in active:
        node = store.get(addr)
        if node and (not domain or node.meta.get("domain") == domain):
            nodes.append(node)
    return nodes


def _op_spatial(step: dict, _current: list[Node], store: Store) -> list[Node]:
    from mnemo_anchor import get_anchors_for_file
    filepath = step.get("file", "")
    if not filepath:
        return _current
    anchored = get_anchors_for_file(filepath, store)
    return [item["node"] for item in anchored]


def _op_traverse(step: dict, current: list[Node], store: Store) -> list[Node]:
    from mnemo_graph import traverse_graph
    depth = int(step.get("depth", 2))
    rel_types = step.get("rel_types")
    direction = step.get("direction", "both")
    max_nodes = int(step.get("max_nodes", 40))

    seen: set[str] = {n.addr for n in current}
    result = list(current)

    for node in current:
        traversal = traverse_graph(store, node.addr, depth=depth,
                                   rel_types=rel_types, direction=direction,
                                   max_nodes=max_nodes)
        for addr, tnode in traversal["nodes"].items():
            if addr not in seen:
                seen.add(addr)
                result.append(tnode)

    return result


def _op_filter(step: dict, current: list[Node], _store: Store) -> list[Node]:
    domain = step.get("domain")
    min_priority = step.get("min_priority")
    min_confidence = step.get("min_confidence")
    has_anchors = step.get("has_anchors")
    node_type = step.get("type")

    result = []
    for node in current:
        if domain and node.meta.get("domain") != domain:
            continue
        if min_priority is not None and node.meta.get("priority", 0) < float(min_priority):
            continue
        if min_confidence is not None and node.meta.get("confidence", 1) < float(min_confidence):
            continue
        if has_anchors is True and not node.meta.get("anchors"):
            continue
        if node_type and node.type != node_type:
            continue
        result.append(node)
    return result


def _op_sort(step: dict, current: list[Node], _store: Store) -> list[Node]:
    by = step.get("by", "created")
    reverse = step.get("reverse", True)
    key_fns = {
        "created":  lambda n: n.created,
        "priority": lambda n: n.meta.get("priority", 0),
        "confidence": lambda n: n.meta.get("confidence", 1),
    }
    key = key_fns.get(by, lambda n: n.created)
    return sorted(current, key=key, reverse=bool(reverse))


def _op_limit(step: dict, current: list[Node], _store: Store) -> list[Node]:
    n = int(step.get("n", 10))
    return current[:n]


def _op_dedupe(step: dict, current: list[Node], _store: Store) -> list[Node]:
    seen: set[str] = set()
    result = []
    for node in current:
        if node.addr not in seen:
            seen.add(node.addr)
            result.append(node)
    return result


def _op_compress(step: dict, current: list[Node], store: Store) -> list[Node]:
    if not current:
        return current
    label = step.get("label", "pipeline compress")
    domain = step.get("domain", "context")

    addrs = [n.addr for n in current]

    # Auto-generate summary from node snippets — no LLM
    parts = []
    for node in current[:8]:
        snippet = node.content[:80].replace("\n", " ")
        d = node.meta.get("domain", "")
        parts.append(f"[{d}] {snippet}" if d else snippet)
    summary = f"{label}: " + "; ".join(parts)
    if len(current) > 8:
        summary += f" (+{len(current) - 8} more)"

    new_addr = _compress_nodes(addrs, summary, store, domain=domain)

    # Remove inputs from active, add compress node
    active = store.get_active()
    for addr in addrs:
        active.discard(addr)
    active.add(new_addr)
    store.set_active(active)

    compress_node = store.get(new_addr)
    return [compress_node] if compress_node else []


def _op_claim(step: dict, current: list[Node], store: Store) -> list[Node]:
    content = step.get("content", "")
    if not content:
        return current
    domain = step.get("domain", "context")
    priority = float(step.get("priority", 0))
    node = Node(
        type="leaf",
        content=content,
        meta={"domain": domain, "priority": priority,
              "source": "pipeline", "created": time.time()},
    )
    store.put(node)
    active = store.get_active()
    active.add(node.addr)
    store.set_active(active)
    return current  # pass through


def _op_link(step: dict, current: list[Node], store: Store) -> list[Node]:
    target_addr = step.get("target", "")
    rel = step.get("rel", "relates_to")
    if not target_addr or not current:
        return current
    for node in current:
        links = node.meta.setdefault("links", [])
        if not any(lk.get("addr") == target_addr for lk in links):
            links.append({"addr": target_addr, "rel": rel})
        store.put(node)
    return current


_OPS = {
    "recall":   _op_recall,
    "search":   _op_search,
    "active":   _op_active,
    "spatial":  _op_spatial,
    "traverse": _op_traverse,
    "filter":   _op_filter,
    "sort":     _op_sort,
    "limit":    _op_limit,
    "dedupe":   _op_dedupe,
    "compress": _op_compress,
    "claim":    _op_claim,
    "link":     _op_link,
}


# ───────────────────────────────────────────────────────────────────
# Runner
# ───────────────────────────────────────────────────────────────────

def run_pipeline(pipeline_def: dict, store: Store, params: dict | None = None) -> dict:
    """
    Execute a pipeline definition against the store.

    Args:
        pipeline_def: dict with "steps" list and optional "name"
        store:        the node store
        params:       variable substitutions for {varname} in steps

    Returns:
        {
          "nodes":         list[Node] — final node set
          "steps_run":     int
          "errors":        list[str]
          "name":          str
        }
    """
    ctx = params or {}
    steps = pipeline_def.get("steps", [])
    name = pipeline_def.get("name", "unnamed")
    current: list[Node] = []
    errors: list[str] = []

    for i, step in enumerate(steps):
        op = step.get("op", "")
        fn = _OPS.get(op)
        if not fn:
            errors.append(f"step {i}: unknown op '{op}'")
            continue
        try:
            resolved = _interp_step(step, ctx)
            current = fn(resolved, current, store)
        except Exception as e:
            errors.append(f"step {i} ({op}): {e}")

    return {
        "nodes":     current,
        "steps_run": len(steps),
        "errors":    errors,
        "name":      name,
    }


# ───────────────────────────────────────────────────────────────────
# Pipeline node management
# ───────────────────────────────────────────────────────────────────

def define_pipeline(name: str, steps: list[dict], store: Store,
                    description: str = "") -> str:
    """
    Store a pipeline definition as a node in the tree.
    Returns the node address.
    """
    import json
    content = description or f"pipeline: {name} ({len(steps)} steps)"
    node = Node(
        type="pipeline",
        content=content,
        meta={
            "domain": "context",
            "pipeline": {
                "name": name,
                "steps": steps,
                "description": description,
            },
            "source": "pipeline",
        },
    )
    store.put(node)
    active = store.get_active()
    active.add(node.addr)
    store.set_active(active)
    return node.addr


def get_pipeline(name_or_addr: str, store: Store) -> dict | None:
    """
    Look up a pipeline by name or address.

    Checks built-ins first, then the store (by addr prefix or name scan).
    Returns the pipeline def dict, or None if not found.
    """
    # Built-ins
    if name_or_addr in BUILTIN_PIPELINES:
        p = BUILTIN_PIPELINES[name_or_addr].copy()
        p["name"] = name_or_addr
        return p

    # Try as address prefix
    node = store.get(name_or_addr)
    if node and node.type == "pipeline":
        p = node.meta.get("pipeline", {})
        p["name"] = p.get("name", name_or_addr)
        return p

    # Scan active set for matching name
    for addr in store.get_active():
        node = store.get(addr)
        if node and node.type == "pipeline":
            p = node.meta.get("pipeline", {})
            if p.get("name") == name_or_addr:
                p["name"] = name_or_addr
                return p

    return None


def list_pipelines(store: Store) -> list[dict]:
    """
    List all available pipelines: built-ins + stored.
    Returns list of {name, description, steps_count, addr?, source}.
    """
    result = []

    # Built-ins
    for name, defn in BUILTIN_PIPELINES.items():
        result.append({
            "name":        name,
            "description": defn.get("description", ""),
            "steps_count": len(defn.get("steps", [])),
            "source":      "builtin",
        })

    # Stored
    for addr in store.get_active():
        node = store.get(addr)
        if node and node.type == "pipeline":
            p = node.meta.get("pipeline", {})
            name = p.get("name", addr[:8])
            # Skip if it shadows a builtin (builtin takes precedence)
            if name not in BUILTIN_PIPELINES:
                result.append({
                    "name":        name,
                    "description": p.get("description", node.content),
                    "steps_count": len(p.get("steps", [])),
                    "addr":        addr,
                    "source":      "stored",
                })

    return result


def render_result(result: dict) -> str:
    """Render a pipeline run result as a readable string."""
    name = result["name"]
    nodes = result["nodes"]
    errors = result["errors"]
    steps_run = result["steps_run"]

    lines = [f"Pipeline '{name}': {steps_run} steps, {len(nodes)} node(s) in final set"]

    if errors:
        lines.append(f"\nErrors ({len(errors)}):")
        for e in errors:
            lines.append(f"  {e}")

    if nodes:
        lines.append(f"\nOutput nodes:")
        for node in nodes[:10]:
            domain = node.meta.get("domain", "?")
            snippet = node.content[:100].replace("\n", " ")
            lines.append(f"  [{domain}] {node.addr[:8]}  {snippet}")
        if len(nodes) > 10:
            lines.append(f"  ... and {len(nodes) - 10} more")

    return "\n".join(lines)
