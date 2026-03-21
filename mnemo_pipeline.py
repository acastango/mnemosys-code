"""
mnemo_pipeline.py - Composable memory pipelines and vectors

Pipelines are first-class nodes in the tree: stored, addressed,
supersedable, and recallable like any other knowledge.

Every step receives the current node set and returns a new one.
Sources ignore the current set and produce fresh nodes.
Sinks produce side effects and pass the set through unchanged.
No LLM in the loop - execution is pure Python over the store.

Step ops:
  Sources:    recall, search, active, spatial, pipe (passthrough)
  Transforms: traverse, filter, sort, limit, dedupe
  Sinks:      compress, claim, link

Vectors are compositions of multiple pipelines (type="vector"):
  components: list of {pipeline, params} — run as fan-out or chain
  merge:      dedupe | union | intersect | ranked | sequential
  post:       optional pipeline steps applied to the merged result

sequential merge threads each component's output into the next.
All other merges run components independently and combine results.

Variables: {varname} in any string parameter, resolved from
the params dict passed to run_pipeline() / run_vector().

MCP surface (in mnemo_mcp.py):
  memory_pipeline(name, steps, description?)              - define pipeline
  memory_vector(name, components, merge, post, desc?)     - define vector
  memory_run(name_or_addr, params?)                       - invoke either
  memory_pipelines()                                      - list pipelines
  memory_vectors()                                        - list vectors
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

def _op_pipe(step: dict, current: list[Node], _store: Store) -> list[Node]:
    """Passthrough source — passes current node set through unchanged.
    Used as the first step in a pipeline that receives upstream output
    from sequential vector composition."""
    return current


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
    "pipe":     _op_pipe,
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

def run_pipeline(pipeline_def: dict, store: Store, params: dict | None = None,
                 initial_nodes: list[Node] | None = None) -> dict:
    """
    Execute a pipeline definition against the store.

    Args:
        pipeline_def:  dict with "steps" list and optional "name"
        store:         the node store
        params:        variable substitutions for {varname} in steps
        initial_nodes: seed the node set before step 1 (used by sequential
                       vector composition to thread output between components)

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
    current: list[Node] = list(initial_nodes) if initial_nodes else []
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


def learn_from_chain(chain_id: str, store: Store, name: str = "") -> dict | None:
    """
    Extract a reusable pipeline pattern from a chain.

    Analyzes the sequence of nodes in the chain and infers what operations
    produced them based on node metadata. Produces a pipeline definition
    that captures the shape of the work — not the specific content, but
    the methodology.

    Inference rules (in order of application):
      - Nodes with recall_count > 0 or source="subconscious" → recall step
      - Nodes with links to others / traversal evidence             → traverse step
      - Majority domain across chain                               → filter step
      - Compress node near end                                     → compress step
      - Explicit claims (source="conscious", no anchors)          → claim pattern noted
      - Content-hash anchored nodes                               → spatial/file step

    Returns a pipeline def dict (not stored — caller decides whether to store it).
    Returns None if chain not found or has fewer than 2 nodes.
    """
    from mnemo_chains import get_chain
    from collections import Counter

    chain = get_chain(store, chain_id)
    if not chain:
        # Try prefix match
        all_chains = store.root / "chains.json"
        if all_chains.exists():
            import json
            data = json.loads(all_chains.read_text(encoding="utf-8"))
            for cid in data:
                if cid.startswith(chain_id):
                    chain = data[cid]
                    chain_id = cid
                    break
    if not chain:
        return None

    members = chain.get("members", [])
    if len(members) < 2:
        return None

    nodes = [store.get(addr) for addr in members]
    nodes = [n for n in nodes if n is not None]
    if not nodes:
        return None

    steps: list[dict] = []

    # ── Source step ──────────────────────────────────────────────────
    # Detect recalled nodes at the start (subconscious surfacing)
    recalled = [n for n in nodes[:3]
                if n.meta.get("recall_count", 0) > 0
                or n.meta.get("source") == "subconscious"]
    if recalled:
        steps.append({"op": "recall", "query": "{input}", "max_nodes": 10})
    else:
        # Check for file-anchored nodes at the start → spatial source
        file_anchored = [n for n in nodes[:3]
                         if any(a.get("type") == "content_hash"
                                for a in n.meta.get("anchors", []))]
        if file_anchored:
            anchor = file_anchored[0].meta["anchors"][0]
            steps.append({"op": "spatial", "file": anchor.get("file", "{input}")})
        else:
            steps.append({"op": "recall", "query": "{input}", "max_nodes": 10})

    # ── Traversal ────────────────────────────────────────────────────
    # If nodes have links or chain spans multiple domains, traversal was likely used
    linked_nodes = [n for n in nodes if n.meta.get("links")]
    domain_spread = len({n.meta.get("domain") for n in nodes if n.meta.get("domain")})
    if linked_nodes or domain_spread > 1:
        steps.append({"op": "traverse", "depth": 1})

    # ── Domain filter ────────────────────────────────────────────────
    domains = [n.meta.get("domain") for n in nodes if n.meta.get("domain")]
    if domains:
        top_domain, top_count = Counter(domains).most_common(1)[0]
        # Only add filter if one domain clearly dominates (>50%)
        if top_count > len(nodes) * 0.5 and domain_spread > 1:
            steps.append({"op": "filter", "domain": top_domain})

    # ── Dedupe ───────────────────────────────────────────────────────
    steps.append({"op": "dedupe"})

    # ── Sink ─────────────────────────────────────────────────────────
    # Compress node near the end → compress step
    has_compress = any(n.type == "compress" for n in nodes[-3:])
    if has_compress:
        chain_summary = chain.get("summary", "")
        label = f"learned: {chain_summary[:40]}" if chain_summary else "learned: {input}"
        steps.append({"op": "compress", "label": label})

    # ── Build result ─────────────────────────────────────────────────
    chain_summary = chain.get("summary", chain_id)
    pipeline_name = name or f"learned-{chain_id[:8]}"
    description = (
        f"Learned from chain {chain_id[:8]}: {chain_summary[:80]}"
        if chain_summary else f"Learned from chain {chain_id[:8]}"
    )

    return {
        "name":        pipeline_name,
        "description": description,
        "steps":       steps,
        "learned_from": chain_id,
    }


def render_learned(pipeline_def: dict) -> str:
    """Render a learned pipeline definition for display."""
    name = pipeline_def["name"]
    desc = pipeline_def.get("description", "")
    steps = pipeline_def.get("steps", [])
    source = pipeline_def.get("learned_from", "")

    lines = [f"Learned pipeline: '{name}'"]
    if desc:
        lines.append(f"  {desc}")
    lines.append(f"\nSteps ({len(steps)}):")
    for i, step in enumerate(steps, 1):
        op = step["op"]
        params = {k: v for k, v in step.items() if k != "op"}
        param_str = "  " + ", ".join(f"{k}={v!r}" for k, v in params.items()) if params else ""
        lines.append(f"  {i}. {op}{param_str}")
    lines.append(f"\nTo store: memory_pipeline({name!r}, steps)")
    lines.append(f"To run:   memory_run({name!r}, params={{\"input\": \"...\"}})")
    return "\n".join(lines)


# ───────────────────────────────────────────────────────────────────
# Vector merge strategies
# ───────────────────────────────────────────────────────────────────

def _merge_dedupe(component_results: list[list[Node]]) -> list[Node]:
    """Union of all components, first-occurrence wins."""
    seen: set[str] = set()
    result = []
    for nodes in component_results:
        for n in nodes:
            if n.addr not in seen:
                seen.add(n.addr)
                result.append(n)
    return result


def _merge_intersect(component_results: list[list[Node]]) -> list[Node]:
    """Only nodes present in every component result."""
    if not component_results:
        return []
    addr_sets = [set(n.addr for n in nodes) for nodes in component_results]
    common = addr_sets[0].intersection(*addr_sets[1:])
    return [n for n in component_results[0] if n.addr in common]


def _merge_ranked(component_results: list[list[Node]]) -> list[Node]:
    """Round-robin interleave: take one from each component in turn, dedupe."""
    seen: set[str] = set()
    result = []
    max_len = max((len(r) for r in component_results), default=0)
    for i in range(max_len):
        for nodes in component_results:
            if i < len(nodes):
                n = nodes[i]
                if n.addr not in seen:
                    seen.add(n.addr)
                    result.append(n)
    return result


_MERGE_FNS = {
    "dedupe":    _merge_dedupe,
    "union":     _merge_dedupe,   # alias
    "intersect": _merge_intersect,
    "ranked":    _merge_ranked,
}


# ───────────────────────────────────────────────────────────────────
# Vector runner
# ───────────────────────────────────────────────────────────────────

def run_vector(vector_def: dict, store: Store, params: dict | None = None) -> dict:
    """
    Execute a vector: run N pipelines, merge results, apply post steps.

    merge="sequential" threads each component's output into the next —
    the output of component i becomes initial_nodes for component i+1.
    All other merge modes run components independently and combine results.

    The "post" field is a list of pipeline steps applied to the merged
    result after combining — same step schema as a regular pipeline.

    Returns:
        {
          "nodes":           list[Node] — final merged + post-processed set
          "component_count": int
          "components_run":  int
          "merge":           str
          "errors":          list[str]
          "name":            str
        }
    """
    ctx = params or {}
    components = vector_def.get("components", [])
    merge = vector_def.get("merge", "dedupe")
    post_steps = vector_def.get("post", [])
    name = vector_def.get("name", "unnamed-vector")

    errors: list[str] = []
    component_results: list[list[Node]] = []

    if merge == "sequential":
        current_nodes: list[Node] = []
        for i, comp in enumerate(components):
            pipeline_name = comp.get("pipeline", "")
            comp_params = {**ctx, **{k: _interp(v, ctx)
                                     for k, v in comp.get("params", {}).items()}}
            pipeline_def = get_pipeline(pipeline_name, store)
            if pipeline_def is None:
                errors.append(f"component {i}: pipeline '{pipeline_name}' not found")
                continue
            result = run_pipeline(pipeline_def, store, comp_params,
                                  initial_nodes=current_nodes)
            errors.extend(result["errors"])
            current_nodes = result["nodes"]
            component_results.append(current_nodes)
        merged = current_nodes

    else:
        for i, comp in enumerate(components):
            pipeline_name = comp.get("pipeline", "")
            comp_params = {**ctx, **{k: _interp(v, ctx)
                                     for k, v in comp.get("params", {}).items()}}
            pipeline_def = get_pipeline(pipeline_name, store)
            if pipeline_def is None:
                errors.append(f"component {i}: pipeline '{pipeline_name}' not found")
                component_results.append([])
                continue
            result = run_pipeline(pipeline_def, store, comp_params)
            errors.extend(result["errors"])
            component_results.append(result["nodes"])

        merge_fn = _MERGE_FNS.get(merge, _merge_dedupe)
        merged = merge_fn(component_results)

    # Post-merge pipeline steps
    if post_steps and merged:
        post_def = {"name": f"{name}:post", "steps": post_steps}
        post_result = run_pipeline(post_def, store, ctx, initial_nodes=merged)
        errors.extend(post_result["errors"])
        merged = post_result["nodes"]

    return {
        "nodes":           merged,
        "component_count": len(components),
        "components_run":  len(component_results),
        "merge":           merge,
        "errors":          errors,
        "name":            name,
    }


# ───────────────────────────────────────────────────────────────────
# Vector node management
# ───────────────────────────────────────────────────────────────────

def define_vector(name: str, components: list[dict], store: Store,
                  merge: str = "dedupe", post: list[dict] | None = None,
                  description: str = "") -> str:
    """
    Store a vector definition as a node in the tree.
    Returns the node address.
    """
    content = description or f"vector: {name} ({len(components)} components, merge={merge})"
    node = Node(
        type="vector",
        content=content,
        meta={
            "domain": "context",
            "vector": {
                "name":        name,
                "components":  components,
                "merge":       merge,
                "post":        post or [],
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


def get_vector(name_or_addr: str, store: Store) -> dict | None:
    """Look up a vector by name or address prefix. Returns the vector def or None."""
    node = store.get(name_or_addr)
    if node and node.type == "vector":
        v = node.meta.get("vector", {})
        v["name"] = v.get("name", name_or_addr)
        return v

    for addr in store.get_active():
        node = store.get(addr)
        if node and node.type == "vector":
            v = node.meta.get("vector", {})
            if v.get("name") == name_or_addr:
                v["name"] = name_or_addr
                return v

    return None


def list_vectors(store: Store) -> list[dict]:
    """List all stored vectors."""
    result = []
    for addr in store.get_active():
        node = store.get(addr)
        if node and node.type == "vector":
            v = node.meta.get("vector", {})
            result.append({
                "name":            v.get("name", addr[:8]),
                "description":     v.get("description", node.content),
                "component_count": len(v.get("components", [])),
                "merge":           v.get("merge", "dedupe"),
                "addr":            addr,
            })
    return result


def render_vector_result(result: dict) -> str:
    """Render a vector run result as a readable string."""
    name = result["name"]
    nodes = result["nodes"]
    errors = result["errors"]
    merge = result["merge"]
    n_components = result["component_count"]
    n_run = result["components_run"]

    lines = [
        f"Vector '{name}': {n_run}/{n_components} components, "
        f"merge={merge}, {len(nodes)} node(s) in final set"
    ]

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
